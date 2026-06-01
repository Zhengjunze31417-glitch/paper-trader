#!/usr/bin/env python3
"""
thermo_autopaper.py — Fully Automated Paper Trading Daemon
===========================================================
Runs automatically during Taiwan market hours (09:00–13:35 TW).

What it does every day:
  10:05 / 11:05 / 12:05 → check signal on all approved stocks (bar close)
  every 30 min           → check if any open position hit Stop or Target
  13:25                  → force-close all open positions at EOD

Logs everything to: paper_trades/paper_log.csv
macOS notification on every signal and exit.

Usage:
  python thermo_autopaper.py          # run today (waits for market open)
  python thermo_autopaper.py --cron   # install cron job (auto-start every weekday)
  python thermo_autopaper.py --test   # run one signal check immediately (for testing)
"""
from __future__ import annotations
import logging, math, sys, time, warnings, subprocess
from datetime import datetime, date, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np, pandas as pd, yfinance as yf
from scipy.stats import entropy as scipy_entropy

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("paper_trades/autopaper.log", mode="a"),
    ]
)
log = logging.getLogger("thermo.auto")

# ── Settings ──────────────────────────────────────────────────────────────────
TW = ZoneInfo("Asia/Taipei")
STOP_PCT     = 0.010    # 1.0% stop loss
TARGET_PCT   = 0.025    # 2.5% take profit  →  2.5:1 R:R, break-even WR = 28.6%
ENTRY_START  = 0
ENTRY_WINDOW = 3
LOT          = 1000

W_LOCAL=20; W_SHORT=3; W_MOM_L=10; W_ATR=14; W_ENTROPY=8; W_TREND_MA=15

# Market schedule (TW time HH:MM)
MARKET_OPEN   = "09:00"
BAR_CHECKS    = ["10:05", "11:05", "12:05"]   # 5 min after each hourly bar close
MONITOR_TIMES = ["10:35", "11:35", "12:35", "13:05"]  # intraday position checks
EOD_TIME      = "13:25"                        # force-close all
MARKET_CLOSE  = "13:35"                        # stop the daemon

LOG_DIR       = Path("paper_trades")
LOG_FILE      = LOG_DIR / "paper_log.csv"
APPROVED_FILE = LOG_DIR / "approved_stocks.csv"
LOG_COLS = ["date","ticker","name","side","entry","tp","sl",
            "exit","exit_reason","pnl_lot","status"]


# ── macOS notification ────────────────────────────────────────────────────────

def notify(title: str, message: str):
    try:
        subprocess.run(
            ["osascript", "-e",
             f'display notification "{message}" with title "{title}" sound name "Glass"'],
            capture_output=True, timeout=5
        )
    except Exception:
        pass  # notification is nice-to-have, not critical


# ── Features ──────────────────────────────────────────────────────────────────

def _pe(s, w=8, o=3):
    ld = math.log(math.factorial(o)); v = s.values; n = len(v)
    out = np.full(n, np.nan)
    for i in range(w - 1, n):
        seg = v[i-w+1:i+1]
        pats = np.array([tuple(np.argsort(seg[j:j+o])) for j in range(len(seg)-o+1)])
        _, cnt = np.unique(pats, axis=0, return_counts=True)
        p = cnt / cnt.sum(); out[i] = scipy_entropy(p) / ld
    return pd.Series(out, index=s.index)

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    c = df["Close"].copy(); ret = np.log(c / c.shift(1))
    fd = df[["High", "Low", "Close"]].copy()
    fd["energy_rank"] = (df["High"]-df["Low"]).rolling(W_LOCAL).rank(pct=True) * 100
    fd["vol_c"]       = ret.rolling(W_SHORT).std() / (ret.rolling(W_LOCAL).std() + 1e-9)
    fd["entropy"]     = (1 - _pe(ret,W_ENTROPY,3).rolling(W_LOCAL).rank(pct=True)) * 100
    ma = c.rolling(W_LOCAL).mean(); fd["dist_ma"] = (c - ma) / ma * 100
    fd["mom_s"]    = c.pct_change(W_SHORT) * 100
    fd["mom_l"]    = c.pct_change(W_MOM_L) * 100
    h, l = df["High"], df["Low"]
    tr = pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()], axis=1).max(axis=1)
    fd["atr"]      = tr.rolling(W_ATR).mean()
    fd["trend_ma"] = c.rolling(W_TREND_MA).mean()
    fd["ma_rising"]= (fd["trend_ma"].diff() > 0).astype(int)
    return fd.dropna()

def signal(row) -> str:
    er,vc,es,ms,ml,dst = (row["energy_rank"],row["vol_c"],row["entropy"],
                           row["mom_s"],row["mom_l"],row["dist_ma"])
    up = bool(row["ma_rising"])
    if up:
        if er>50 and vc<1.3 and ms>0 and ml>0: return "LONG"
        if er>40 and ml<-2 and dst<-1.0:        return "LONG"
    else:
        if er>50 and vc<1.3 and ms<0 and ml<0: return "SHORT"
        if er<55 and ml>2  and dst>1.0:         return "SHORT"
    return "NEUTRAL"


# ── Data helpers ──────────────────────────────────────────────────────────────

def fetch_hourly(ticker: str) -> pd.DataFrame | None:
    try:
        raw = yf.download(ticker, period="60d", interval="1h",
                          auto_adjust=True, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw = raw.dropna(subset=["Close"])
        raw.index = raw.index.tz_convert(TW)
        return raw
    except Exception as e:
        log.warning("fetch_hourly %s: %s", ticker, e)
        return None

def fetch_current_price(ticker: str) -> tuple[float, float, float] | None:
    """Returns (current_price, today_high, today_low) from 30m bars."""
    try:
        raw = yf.download(ticker, period="1d", interval="30m",
                          auto_adjust=True, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw = raw.dropna(subset=["Close"])
        raw.index = raw.index.tz_convert(TW)
        today = date.today()
        today_bars = raw[raw.index.date == today]
        if today_bars.empty:
            return None
        return (
            float(today_bars["Close"].iloc[-1]),
            float(today_bars["High"].max()),
            float(today_bars["Low"].min()),
        )
    except Exception as e:
        log.warning("fetch_price %s: %s", ticker, e)
        return None


# ── Paper trade log helpers ───────────────────────────────────────────────────

def load_log() -> pd.DataFrame:
    if LOG_FILE.exists():
        return pd.read_csv(LOG_FILE)
    return pd.DataFrame(columns=LOG_COLS)

def save_log(df: pd.DataFrame):
    df.to_csv(LOG_FILE, index=False)

def open_trades(df: pd.DataFrame, today: str) -> pd.DataFrame:
    return df[(df["status"] == "OPEN") & (df["date"] == today)]

def already_traded(df: pd.DataFrame, ticker: str, today: str) -> bool:
    return not df[(df["ticker"] == ticker) & (df["date"] == today)].empty

def log_entry(ticker: str, name: str, side: str, price: float):
    today = str(date.today())
    df = load_log()
    if already_traded(df, ticker, today):
        log.info("  %s already has a trade today — skipping", ticker)
        return
    tp = price*(1+TARGET_PCT) if side=="LONG" else price*(1-TARGET_PCT)
    sl = price*(1-STOP_PCT)   if side=="LONG" else price*(1+STOP_PCT)
    new_row = pd.DataFrame([{
        "date": today, "ticker": ticker, "name": name, "side": side,
        "entry": round(price, 2), "tp": round(tp, 2), "sl": round(sl, 2),
        "exit": None, "exit_reason": None, "pnl_lot": None, "status": "OPEN",
    }])
    df = pd.concat([df, new_row], ignore_index=True)
    save_log(df)
    log.info("  LOGGED  %s %s @ NT$%.1f  TP=NT$%.1f  SL=NT$%.1f", ticker, side, price, tp, sl)

def close_trade(df: pd.DataFrame, idx: int, exit_price: float, reason: str) -> pd.DataFrame:
    row = df.loc[idx]
    entry = float(row["entry"]); side = row["side"]
    pts   = (exit_price - entry) if side == "LONG" else (entry - exit_price)
    pnl   = round(pts * LOT, 1)
    df.loc[idx, "exit"]        = round(exit_price, 2)
    df.loc[idx, "exit_reason"] = reason
    df.loc[idx, "pnl_lot"]     = pnl
    df.loc[idx, "status"]      = "CLOSED"
    result = "WIN" if pnl > 0 else "LOSS"
    log.info("  CLOSED  %s %s  %s  exit=NT$%.1f  pnl=NT$%+.0f",
             row["ticker"], side, reason, exit_price, pnl)
    return df

def check_milestone(df: pd.DataFrame):
    """Send notification only when closed trade count hits 30 or 50 or 100."""
    closed = df[df["status"] == "CLOSED"]
    n = len(closed)
    for milestone in [30, 50, 100]:
        if n == milestone:
            closed_num = closed.copy()
            closed_num["pnl_lot"] = pd.to_numeric(closed_num["pnl_lot"], errors="coerce")
            total = closed_num["pnl_lot"].sum()
            nw = (closed_num["pnl_lot"] > 0).sum()
            notify(
                f"Thermo Paper — {milestone} Trades!",
                f"{nw}W/{n-nw}L  WR={nw/n*100:.0f}%  Total PnL=NT${total:+,.0f}"
            )
            log.info("MILESTONE: %d trades reached! WR=%.0f%%  PnL=NT$%+.0f",
                     milestone, nw/n*100, total)
            break


# ── Core actions ──────────────────────────────────────────────────────────────

def check_signals(approved: pd.DataFrame, bar_idx: int):
    """Called at bar close — check if signal fires on any approved stock."""
    if bar_idx >= ENTRY_WINDOW:
        log.info("Entry window closed (bar %d). No more entries today.", bar_idx)
        return

    today = str(date.today())
    log.info("── Checking signals  bar=%d  %s ──", bar_idx, datetime.now(TW).strftime("%H:%M"))
    candidates = []

    for _, row in approved.iterrows():
        ticker = row["ticker"]; name = row["name"]
        df_log = load_log()
        if already_traded(df_log, ticker, today):
            continue

        raw = fetch_hourly(ticker)
        if raw is None: continue

        fd = build_features(raw)
        if fd.empty: continue

        # Get today's bars up to this bar_idx
        today_bars = fd[fd.index.date == date.today()]
        if len(today_bars) <= bar_idx: continue

        sig_row = today_bars.iloc[bar_idx]
        sig     = signal(sig_row)
        er      = float(sig_row["energy_rank"])
        price   = float(sig_row["Close"])

        if sig != "NEUTRAL":
            candidates.append({"ticker": ticker, "name": name,
                                "side": sig, "price": price, "energy": er})
            log.info("  SIGNAL  %s (%s)  %s  energy=%.0f  price=NT$%.1f",
                     ticker, name, sig, er, price)
        else:
            log.info("  no signal  %s (%s)  energy=%.0f", ticker, name, er)

    if candidates:
        # Take highest energy_rank if multiple fire same bar
        best = max(candidates, key=lambda x: x["energy"])
        log.info("  → Best signal: %s (energy=%.0f)", best["ticker"], best["energy"])
        log_entry(best["ticker"], best["name"], best["side"], best["price"])
        if len(candidates) > 1:
            others = [c["ticker"] for c in candidates if c["ticker"] != best["ticker"]]
            log.info("  (skipped: %s — same bar, lower energy)", ", ".join(others))
    else:
        log.info("  No signals this bar.")


def monitor_positions():
    """Check if any open position hit Stop or Target."""
    today = str(date.today())
    df    = load_log()
    opens = open_trades(df, today)
    if opens.empty:
        log.info("── Monitor: no open positions")
        return

    log.info("── Monitoring %d open position(s) ──", len(opens))
    for idx, trade in opens.iterrows():
        ticker = trade["ticker"]
        prices = fetch_current_price(ticker)
        if prices is None: continue

        current, today_hi, today_lo = prices
        tp = float(trade["tp"]); sl = float(trade["sl"]); side = trade["side"]

        hit_price = hit_reason = None
        if side == "LONG":
            if today_lo <= sl:   hit_price, hit_reason = sl, "STOP"
            elif today_hi >= tp: hit_price, hit_reason = tp, "TARGET"
        else:
            if today_hi >= sl:   hit_price, hit_reason = sl, "STOP"
            elif today_lo <= tp: hit_price, hit_reason = tp, "TARGET"

        if hit_price:
            df = close_trade(df, idx, hit_price, hit_reason)
            save_log(df)
            check_milestone(df)
        else:
            log.info("  %s  current=NT$%.1f  TP=NT$%.1f  SL=NT$%.1f  (open)",
                     ticker, current, tp, sl)


def close_eod():
    """Force-close all open positions at current price (EOD)."""
    today = str(date.today())
    df    = load_log()
    opens = open_trades(df, today)
    if opens.empty:
        log.info("── EOD: no open positions to close")
        return

    log.info("── EOD CLOSE  %d position(s) ──", len(opens))
    for idx, trade in opens.iterrows():
        prices = fetch_current_price(trade["ticker"])
        exit_price = prices[0] if prices else float(trade["entry"])
        df = close_trade(df, idx, exit_price, "EOD")
    save_log(df)
    check_milestone(df)


def print_daily_summary():
    df = load_log()
    today = str(date.today())
    today_trades = df[df["date"] == today]
    closed = today_trades[today_trades["status"] == "CLOSED"]
    closed = closed.copy()
    closed["pnl_lot"] = pd.to_numeric(closed["pnl_lot"], errors="coerce")

    print(f"\n{'='*55}")
    print(f"  Daily Summary — {today}")
    print(f"{'='*55}")
    if closed.empty:
        print("  No trades today.")
    else:
        n = len(closed); nw = (closed.pnl_lot > 0).sum()
        total = closed.pnl_lot.sum()
        print(f"  Trades: {n}  ({nw}W/{n-nw}L)  PnL: NT${total:+,.0f}")
        for _, r in closed.iterrows():
            print(f"  {r['ticker']:8s} {r['side']:5s} {r['exit_reason']:6s}"
                  f"  entry=NT${float(r['entry']):.1f}"
                  f"  exit=NT${float(r['exit']):.1f}"
                  f"  PnL=NT${float(r['pnl_lot']):+,.0f}")

    all_closed = df[df["status"] == "CLOSED"].copy()
    all_closed["pnl_lot"] = pd.to_numeric(all_closed["pnl_lot"], errors="coerce")
    n_all = len(all_closed)
    if n_all > 0:
        nw_all = (all_closed.pnl_lot > 0).sum()
        total_all = all_closed.pnl_lot.sum()
        print(f"\n  All-time: {n_all} trades  WR={nw_all/n_all*100:.0f}%"
              f"  Total PnL=NT${total_all:+,.0f}")
        print(f"  Progress to 30-trade milestone: {n_all}/30"
              f"  ({max(0,30-n_all)} more needed)")
    print(f"{'='*55}\n")
    # Write daily summary CSV
    summary_file = LOG_DIR / "daily_summary.csv"
    summary_cols = ["date","ticker","name","side","entry","tp","sl",
                    "exit","exit_reason","pnl_lot","cumulative_trades",
                    "cumulative_pnl","win_rate_pct"]
    all_closed = df[df["status"] == "CLOSED"].copy()
    all_closed["pnl_lot"] = pd.to_numeric(all_closed["pnl_lot"], errors="coerce")
    all_closed = all_closed.sort_values("date").reset_index(drop=True)
    all_closed["cumulative_trades"] = range(1, len(all_closed)+1)
    all_closed["cumulative_pnl"]    = all_closed["pnl_lot"].cumsum().round(1)
    all_closed["win_rate_pct"]      = (
        (all_closed["pnl_lot"] > 0).expanding().mean() * 100
    ).round(1)
    export_cols = [c for c in summary_cols if c in all_closed.columns]
    all_closed[export_cols].to_csv(summary_file, index=False)
    log.info("Summary CSV → %s", summary_file)


# ── Time helpers ──────────────────────────────────────────────────────────────

def tw_hhmm() -> str:
    return datetime.now(TW).strftime("%H:%M")

def is_weekday() -> bool:
    return datetime.now(TW).weekday() < 5  # Mon=0 … Fri=4

def wait_until(hhmm: str):
    """Sleep until HH:MM TW time."""
    now = datetime.now(TW)
    h, m = map(int, hhmm.split(":"))
    target = now.replace(hour=h, minute=m, second=0, microsecond=0)
    if target <= now:
        return
    secs = (target - now).total_seconds()
    log.info("Waiting until %s TW  (%.0f min)", hhmm, secs/60)
    time.sleep(secs)

def past(hhmm: str) -> bool:
    return tw_hhmm() >= hhmm


# ── Main daemon ───────────────────────────────────────────────────────────────

def run_daemon():
    LOG_DIR.mkdir(exist_ok=True)

    if not APPROVED_FILE.exists():
        log.error("No approved stocks. Run: python thermo_scanner.py scan")
        sys.exit(1)

    approved = pd.read_csv(APPROVED_FILE)
    log.info("Loaded %d approved stocks: %s",
             len(approved), ", ".join(approved.ticker.tolist()))

    if not is_weekday():
        log.info("Today is a weekend — no trading. Exiting.")
        return

    today_str = str(date.today())
    log.info("=== Thermo AutoPaper — %s ===", today_str)

    # Wait for market open
    if not past(MARKET_OPEN):
        wait_until(MARKET_OPEN)

    done_bars    = set()   # track which bar checks were done
    done_monitor = set()   # track which monitor times were done
    done_eod     = False

    bar_map = {"10:05": 0, "11:05": 1, "12:05": 2}

    log.info("Market open. Running until %s TW ...", MARKET_CLOSE)

    while not past(MARKET_CLOSE):
        now = tw_hhmm()

        # Signal checks at bar close
        for check_time, bar_idx in bar_map.items():
            if now >= check_time and check_time not in done_bars:
                check_signals(approved, bar_idx)
                done_bars.add(check_time)

        # Intraday position monitors
        for mtime in MONITOR_TIMES:
            if now >= mtime and mtime not in done_monitor:
                monitor_positions()
                done_monitor.add(mtime)

        # EOD close
        if now >= EOD_TIME and not done_eod:
            close_eod()
            done_eod = True

        time.sleep(30)  # poll every 30 seconds

    print_daily_summary()
    log.info("=== Market closed. Done for today. ===")


# ── Cron setup ────────────────────────────────────────────────────────────────

def setup_cron():
    """Add a cron job to run the daemon automatically every weekday at 08:55 TW."""
    import getpass, os
    python = sys.executable
    script = Path(__file__).resolve()
    cron_line = f"55 8 * * 1-5 cd {script.parent} && {python} {script} >> paper_trades/autopaper.log 2>&1"

    result = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
    existing = result.stdout if result.returncode == 0 else ""

    if str(script) in existing:
        print("Cron job already exists:")
        for line in existing.splitlines():
            if str(script) in line:
                print(f"  {line}")
        return

    new_crontab = existing.rstrip() + "\n" + cron_line + "\n"
    proc = subprocess.run(["crontab", "-"], input=new_crontab, text=True, capture_output=True)
    if proc.returncode == 0:
        print(f"\n  Cron job installed successfully!")
        print(f"  Will run every weekday at 08:55 TW time.")
        print(f"  Cron entry: {cron_line}")
        print(f"\n  To remove: crontab -e  (delete the line)")
        print(f"  To verify: crontab -l")
    else:
        print(f"  Failed to install cron: {proc.stderr}")
        print(f"\n  Add this line manually with 'crontab -e':")
        print(f"  {cron_line}")


# ── Test mode ─────────────────────────────────────────────────────────────────

def run_test():
    """Run one signal check immediately — for testing without waiting for market hours."""
    LOG_DIR.mkdir(exist_ok=True)
    if not APPROVED_FILE.exists():
        log.error("Run scan first: python thermo_scanner.py scan")
        return
    approved = pd.read_csv(APPROVED_FILE)
    log.info("TEST MODE — running signal check on bar 0 now")
    check_signals(approved, bar_idx=0)
    monitor_positions()
    print_daily_summary()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else ""

    if arg == "--cron":
        setup_cron()

    elif arg == "--test":
        run_test()

    elif arg == "--action":
        # GitHub Actions single-shot mode: one check per workflow run
        LOG_DIR.mkdir(exist_ok=True)
        action = sys.argv[2] if len(sys.argv) > 2 else "signal"

        if not APPROVED_FILE.exists():
            log.error("approved_stocks.csv not found — commit it to the repo first.")
            sys.exit(1)

        approved = pd.read_csv(APPROVED_FILE)
        log.info("Loaded %d approved stocks", len(approved))

        if action == "signal":
            # Determine which bar based on current TW hour
            now_tw  = datetime.now(TW)
            bar_map = {10: 0, 11: 1, 12: 2}
            bar_idx = bar_map.get(now_tw.hour, 0)
            log.info("GitHub Actions: signal check  TW=%s  bar=%d",
                     now_tw.strftime("%H:%M"), bar_idx)
            check_signals(approved, bar_idx)

        elif action == "eod":
            log.info("GitHub Actions: EOD close")
            close_eod()
            print_daily_summary()

        elif action == "scan":
            # Re-run the universe scan to refresh approved_stocks.csv
            log.info("GitHub Actions: weekly scan")
            import importlib.util, pathlib
            scanner_path = pathlib.Path(__file__).parent / "thermo_scanner.py"
            spec = importlib.util.spec_from_file_location("scanner", scanner_path)
            scanner = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(scanner)
            scanner.run_scan()

    else:
        run_daemon()
