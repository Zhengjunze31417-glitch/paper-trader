#!/usr/bin/env python3
"""
paper_trader.py
---------------
Multi-market paper trading system to validate the CompressedSpring
institutional accumulation theory over 6 months.

Runs every hour at :01. No real orders — fully simulated.
All signals and trades logged to paper_trades.csv for statistical analysis.

Assets:
  Crypto : ETH-USD, BTC-USD         (2x leverage simulation)
  Stocks : QQQ, SPY, AAPL, GLD, USO (1x spot simulation)

── SETUP ─────────────────────────────────────────────────────────────────────
  pip install yfinance schedule requests pandas numpy scipy

  # Optional Telegram alerts
  export TELEGRAM_TOKEN="your_bot_token"
  export TELEGRAM_CHAT_ID="your_chat_id"

  python paper_trader.py

── CLOUD DEPLOYMENT ──────────────────────────────────────────────────────────
  screen -S paper
  python paper_trader.py
  # Ctrl+A then D to detach — runs forever without your computer

── OUTPUT FILES ──────────────────────────────────────────────────────────────
  paper_trades.csv   — every completed trade (for statistical analysis)
  paper_signals.csv  — every signal that fired (even if not traded)
  paper_state.json   — open positions (survives restarts)
  paper_trader.log   — full log
"""
from __future__ import annotations
import os, json, csv, logging, sys, time, math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
import schedule
import requests
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from thermo_core import Config, FeatureEngine

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler("paper_trader.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("paper.trader")

# ── Asset universe ────────────────────────────────────────────────────────────
ASSETS = {
    "ETH-USD": {"name": "Ethereum",      "leverage": 2.0, "type": "crypto"},
    "BTC-USD": {"name": "Bitcoin",       "leverage": 2.0, "type": "crypto"},
    "QQQ":     {"name": "Nasdaq-100",    "leverage": 1.0, "type": "stock"},
    "SPY":     {"name": "S&P 500",       "leverage": 1.0, "type": "stock"},
    "GLD":     {"name": "Gold ETF",      "leverage": 1.0, "type": "stock"},
    "AAPL":    {"name": "Apple",         "leverage": 1.0, "type": "stock"},
    "USO":     {"name": "Oil ETF",       "leverage": 1.0, "type": "stock"},
}

# ── Strategy parameters ───────────────────────────────────────────────────────
WARMUP        = 500
MAX_HOLD      = 24        # bars (= 24h crypto / ~3.7 days stocks)
TRAIL_H       = 8
TREND_MA      = 50
STOP_MULT     = 1.0
TARGET_MULT   = 3.0
POS_RISK      = 0.015
START_EQUITY  = 1000.0    # paper capital per asset

# Signal filters
SPRING_EMIN   = 55.0
SPRING_EMAX   = 75.0
SPRING_DIST   = 3.0
SPRING_MOM    = 0.0
VC_PCTILE     = 0.20      # adaptive: use P20 of each asset's vol_contraction

CFG = Config(min_dollar_vol=0.0, min_history_bars=50)

# ── File paths ────────────────────────────────────────────────────────────────
STATE_FILE   = Path("paper_state.json")
TRADES_FILE  = Path("paper_trades.csv")
SIGNALS_FILE = Path("paper_signals.csv")

TRADE_COLS = [
    "trade_id","asset","direction","entry_time","entry_price",
    "stop_price","target_price","vc","energy","dist","leverage",
    "exit_time","exit_price","exit_reason","pnl_pct","win",
]
SIGNAL_COLS = [
    "signal_id","asset","signal_time","direction",
    "price","vc","energy","dist","vc_thresh",
    "traded","skip_reason",
]

def _init_csv(path, cols):
    if not path.exists():
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(cols)

_init_csv(TRADES_FILE,  TRADE_COLS)
_init_csv(SIGNALS_FILE, SIGNAL_COLS)

# ── State ─────────────────────────────────────────────────────────────────────
def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"positions": {}, "trade_count": 0, "signal_count": 0}

def save_state(s: dict):
    STATE_FILE.write_text(json.dumps(s, indent=2, default=str))

# ── Telegram ──────────────────────────────────────────────────────────────────
TELE_TOKEN   = os.environ.get("TELEGRAM_TOKEN", "")
TELE_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

def notify(msg: str):
    log.info("MSG: %s", msg)
    if not TELE_TOKEN or not TELE_CHAT_ID:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELE_TOKEN}/sendMessage",
            json={"chat_id": TELE_CHAT_ID, "text": f"📋 Paper Trader\n{msg}"},
            timeout=10,
        )
    except Exception as e:
        log.warning("Telegram failed: %s", e)

# ── Data & features ───────────────────────────────────────────────────────────
def fetch_features(ticker: str) -> pd.DataFrame | None:
    try:
        df = yf.download(ticker, period="730d", interval="1h",
                         auto_adjust=True, progress=False, threads=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.dropna(subset=["Close"])
        if len(df) < WARMUP + 10:
            log.warning("%s: not enough bars (%d)", ticker, len(df))
            return None

        engine = FeatureEngine(CFG)
        fd = engine.compute(ticker, df, df["Close"], as_history=True)
        if fd is None or fd.empty:
            return None

        h, l, c = fd["High"], fd["Low"], fd["Close"]
        tr = pd.concat(
            [h-l, (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1
        ).max(axis=1)
        fd["_atr"]  = tr.rolling(14).mean()
        fd["_ma50"] = fd["Close"].rolling(TREND_MA).mean()

        # Adaptive VC threshold from warmup period
        vc_thresh = float(
            fd["vol_contraction"].iloc[:WARMUP].dropna().quantile(VC_PCTILE)
        )
        fd.attrs["vc_thresh"] = vc_thresh
        return fd
    except Exception as e:
        log.error("%s fetch_features error: %s", ticker, e)
        return None

def classify_signal(row, vc_thresh: float) -> str | None:
    vc   = float(row.get("vol_contraction",  1.0))
    eng  = float(row.get("energy_rank",      0.0))
    mom  = float(row.get("mom_5d",           0.0))
    dist = abs(float(row.get("dist_20ma_pct", 0.0)))
    cl   = float(row.get("Close",            0.0))
    ma50 = float(row.get("_ma50",            0.0))
    if any(pd.isna(x) for x in [vc, eng, mom, dist, cl, ma50]):
        return None
    spring = (vc < vc_thresh
              and SPRING_EMIN <= eng <= SPRING_EMAX
              and dist <= SPRING_DIST)
    if not spring:
        return None
    if cl > ma50 and mom >= SPRING_MOM:
        return "Long"
    if cl < ma50 and mom <= SPRING_MOM:
        return "Short"
    return None

# ── Trade log helpers ─────────────────────────────────────────────────────────
def log_signal(state: dict, asset: str, row, direction: str,
               vc_thresh: float, traded: bool, skip_reason: str = ""):
    state["signal_count"] = state.get("signal_count", 0) + 1
    sid = f"S{state['signal_count']:05d}"
    with open(SIGNALS_FILE, "a", newline="") as f:
        csv.writer(f).writerow([
            sid, asset,
            datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
            direction,
            float(row["Close"]),
            float(row.get("vol_contraction", 0)),
            float(row.get("energy_rank", 0)),
            abs(float(row.get("dist_20ma_pct", 0))),
            round(vc_thresh, 4),
            traded, skip_reason,
        ])
    return sid

def open_trade(state: dict, asset: str, row, direction: str,
               atr: float, leverage: float) -> dict:
    state["trade_count"] = state.get("trade_count", 0) + 1
    tid = f"T{state['trade_count']:05d}"
    cl  = float(row["Close"])
    sd  = STOP_MULT * atr
    pos = {
        "trade_id":    tid,
        "asset":       asset,
        "direction":   direction,
        "entry_time":  datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "entry_price": cl,
        "stop":        cl - sd if direction == "Long" else cl + sd,
        "target":      cl + TARGET_MULT * sd if direction == "Long" else cl - TARGET_MULT * sd,
        "atr":         atr,
        "leverage":    leverage,
        "bars_held":   0,
        "equity":      START_EQUITY,
        "vc":          float(row.get("vol_contraction", 0)),
        "energy":      float(row.get("energy_rank", 0)),
        "dist":        abs(float(row.get("dist_20ma_pct", 0))),
    }
    with open(TRADES_FILE, "a", newline="") as f:
        csv.writer(f).writerow([
            tid, asset, direction, pos["entry_time"], cl,
            pos["stop"], pos["target"],
            pos["vc"], pos["energy"], pos["dist"], leverage,
            "", "", "", "", "",          # exit fields filled later
        ])
    msg = (f"[PAPER] ENTER {direction.upper()} {asset}\n"
           f"Price: ${cl:,.2f}  Stop: ${pos['stop']:,.2f}  "
           f"Target: ${pos['target']:,.2f}\n"
           f"vc={pos['vc']:.3f}  energy={pos['energy']:.0f}  dist={pos['dist']:.1f}%")
    notify(msg)
    return pos

def close_trade(pos: dict, exit_price: float, reason: str):
    d    = pos["direction"]
    pnl  = (exit_price - pos["entry_price"]) / pos["entry_price"]
    pnl  = pnl if d == "Long" else -pnl
    pnl *= pos["leverage"]
    win  = pnl > 0

    # Update CSV: rewrite the row with exit data
    rows = []
    with open(TRADES_FILE, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["trade_id"] == pos["trade_id"]:
                row["exit_time"]   = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
                row["exit_price"]  = round(exit_price, 4)
                row["exit_reason"] = reason
                row["pnl_pct"]     = round(pnl * 100, 4)
                row["win"]         = "1" if win else "0"
            rows.append(row)
    with open(TRADES_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TRADE_COLS)
        writer.writeheader()
        writer.writerows(rows)

    emoji = "✅" if win else "❌"
    msg = (f"[PAPER] {emoji} CLOSE {d.upper()} {pos['asset']}\n"
           f"Exit: ${exit_price:,.2f}  Reason: {reason}\n"
           f"PnL: {pnl*100:+.2f}%  ({'WIN' if win else 'LOSS'})\n"
           f"Held: {pos['bars_held']} bars")
    notify(msg)
    log.info("CLOSE %s %s @ %.2f (%s) PnL=%+.2f%%",
             pos["asset"], d, exit_price, reason, pnl*100)
    return pnl, win

# ── Statistics ────────────────────────────────────────────────────────────────
def compute_stats() -> str:
    if not TRADES_FILE.exists():
        return "No trades yet."

    df = pd.read_csv(TRADES_FILE)
    closed = df[df["exit_reason"].notna() & (df["exit_reason"] != "")]
    if len(closed) == 0:
        open_n = len(df)
        return f"No closed trades yet. {open_n} position(s) open."

    closed = closed.copy()
    closed["pnl_pct"] = pd.to_numeric(closed["pnl_pct"], errors="coerce")
    closed["win"]     = pd.to_numeric(closed["win"],     errors="coerce")
    closed = closed.dropna(subset=["pnl_pct"])

    n   = len(closed)
    wr  = closed["win"].mean() * 100
    be  = STOP_MULT / (STOP_MULT + TARGET_MULT) * 100
    avg_w = closed[closed["win"]==1]["pnl_pct"].mean() if (closed["win"]==1).any() else 0
    avg_l = closed[closed["win"]==0]["pnl_pct"].mean() if (closed["win"]==0).any() else 0
    mean_r = closed["pnl_pct"].mean()

    lines = [f"\n{'='*52}",
             f"  PAPER TRADING STATS  |  {n} closed trades",
             f"{'='*52}"]

    # Per-asset breakdown
    lines.append(f"\n  {'Asset':<10} {'N':>4} {'WR':>7} {'Return':>9} {'Status'}")
    lines.append(f"  {'-'*44}")
    for asset in closed["asset"].unique():
        sub = closed[closed["asset"]==asset]
        awr = sub["win"].mean()*100
        aret = (np.prod(1+sub["pnl_pct"].values/100)-1)*100
        verdict = "✓" if awr>be else "✗"
        lines.append(f"  {asset:<10} {len(sub):>4} {awr:>6.1f}% {aret:>+8.1f}%  {verdict}")

    lines.append(f"\n  Overall WR    : {wr:.1f}%  (break-even={be:.1f}%)")
    lines.append(f"  Avg Win       : {avg_w:+.2f}%   Avg Loss: {avg_l:+.2f}%")

    # t-test if enough trades
    if n >= 10:
        tr_arr = closed["pnl_pct"].values / 100
        t, p = stats.ttest_1samp(tr_arr, 0)
        binom = stats.binomtest(int(closed["win"].sum()), n, be/100, alternative="greater")
        lines.append(f"\n  t-test p-value   : {p:.4f}  {'✓ significant' if p<0.05 else '✗ not yet'}")
        lines.append(f"  Binomial p-value : {binom.pvalue:.4f}  {'✓ significant' if binom.pvalue<0.05 else '✗ not yet'}")
        lines.append(f"  Trades needed for 95% confidence: ~{max(30, int(255*(be/100)/(wr/100-be/100+0.01)))} (have {n})")
    else:
        lines.append(f"\n  Need 10+ trades for statistical tests (have {n})")

    # Signals breakdown
    if SIGNALS_FILE.exists():
        sig_df = pd.read_csv(SIGNALS_FILE)
        lines.append(f"\n  Signals fired : {len(sig_df)}  |  Traded: {sig_df['traded'].sum()}")
        lines.append(f"  Skipped       : {(~sig_df['traded'].astype(bool)).sum()} (position already open)")

    lines.append(f"{'='*52}")
    return "\n".join(lines)

# ── Main hourly job ───────────────────────────────────────────────────────────
def hourly_job():
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    log.info("── Hourly check (%s UTC) ─────────────────────", now)
    state = load_state()

    for ticker, cfg in ASSETS.items():
        leverage = cfg["leverage"]
        try:
            fd = fetch_features(ticker)
            if fd is None:
                continue

            vc_thresh = fd.attrs.get("vc_thresh", 0.40)
            row = fd.iloc[-1]
            lo  = float(row["Low"])
            hi  = float(row["High"])
            cl  = float(row["Close"])
            atr = float(row.get("_atr", 0))
            if pd.isna(atr) or atr <= 0:
                continue

            # ── Manage open position ──────────────────────────────────────
            pos = state["positions"].get(ticker)
            if pos:
                pos["bars_held"] += 1

                # Breakeven stop after TRAIL_H bars
                if pos["bars_held"] == TRAIL_H:
                    d = pos["direction"]
                    if (d == "Long"  and cl > pos["entry_price"]) or \
                       (d == "Short" and cl < pos["entry_price"]):
                        pos["stop"] = pos["entry_price"]
                        log.info("%s breakeven stop set @ %.2f", ticker, pos["entry_price"])

                # Check exits
                d   = pos["direction"]
                ep  = rsn = None
                if d == "Long":
                    if lo <= pos["stop"]:    ep, rsn = pos["stop"],   "STOP"
                    elif hi >= pos["target"]: ep, rsn = pos["target"], "TARGET"
                    elif pos["bars_held"] >= MAX_HOLD: ep, rsn = cl, "TIMEOUT"
                else:
                    if hi >= pos["stop"]:    ep, rsn = pos["stop"],   "STOP"
                    elif lo <= pos["target"]: ep, rsn = pos["target"], "TARGET"
                    elif pos["bars_held"] >= MAX_HOLD: ep, rsn = cl, "TIMEOUT"

                if ep is not None:
                    close_trade(pos, ep, rsn)
                    del state["positions"][ticker]
                else:
                    state["positions"][ticker] = pos
                    pnl_now = (cl - pos["entry_price"])/pos["entry_price"]
                    pnl_now = pnl_now if d=="Long" else -pnl_now
                    log.info("%-8s %s open  bar=%2d/%d  price=%.2f  unrealPnL=%+.2f%%",
                             ticker, d, pos["bars_held"], MAX_HOLD, cl, pnl_now*100)
                continue  # one position per asset

            # ── Check for new signal ──────────────────────────────────────
            if row[list(CFG.features)].isna().any():
                continue

            signal = classify_signal(row, vc_thresh)

            if signal:
                # Log every signal regardless of whether we trade it
                sid = log_signal(state, ticker, row, signal, vc_thresh,
                                 traded=True)
                pos = open_trade(state, ticker, row, signal, atr, leverage)
                state["positions"][ticker] = pos
                log.info("%-8s signal=%s  vc=%.3f(<%.3f)  eng=%.0f  dist=%.1f%%",
                         ticker, signal,
                         float(row.get("vol_contraction",0)), vc_thresh,
                         float(row.get("energy_rank",0)),
                         abs(float(row.get("dist_20ma_pct",0))))
            else:
                log.info("%-8s no signal  vc=%.3f  eng=%.0f  dist=%.1f%%",
                         ticker,
                         float(row.get("vol_contraction",0)) if not pd.isna(row.get("vol_contraction",float("nan"))) else 0,
                         float(row.get("energy_rank",0)) if not pd.isna(row.get("energy_rank",float("nan"))) else 0,
                         abs(float(row.get("dist_20ma_pct",0))) if not pd.isna(row.get("dist_20ma_pct",float("nan"))) else 0)

        except Exception as e:
            log.error("%-8s error: %s", ticker, e)

    save_state(state)

# ── Weekly stats report ───────────────────────────────────────────────────────
def weekly_report():
    report = compute_stats()
    log.info(report)
    notify(report)

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    log.info("=" * 60)
    log.info("  PAPER TRADER — CompressedSpring Multi-Market")
    log.info("  Assets: %s", ", ".join(ASSETS.keys()))
    log.info("  Signal: vc<P20, energy 55-75, dist<3%%, 1:3 R:R")
    log.info("  Duration: 6 months  |  No real orders")
    log.info("=" * 60)

    notify(
        f"Paper Trader STARTED\n"
        f"Assets: {', '.join(ASSETS.keys())}\n"
        f"Capital per asset: ${START_EQUITY:,.0f} (simulated)\n"
        f"Running for 6 months to validate CompressedSpring theory."
    )

    # Run immediately on startup
    hourly_job()

    # Schedule
    schedule.every().hour.at(":01").do(hourly_job)
    schedule.every().monday.at("09:00").do(weekly_report)

    log.info("Scheduler running. Checks every hour at :01.")
    log.info("Weekly report every Monday at 09:00.")
    log.info("Results saved to: %s", TRADES_FILE.absolute())

    while True:
        schedule.run_pending()
        time.sleep(30)
