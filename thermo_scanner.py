#!/usr/bin/env python3
"""
thermo_scanner.py — Taiwan Stock Universe Scanner + Daily Paper Trade Logger
=============================================================================
Two modes:

  python thermo_scanner.py scan     → scan universe, find qualifying stocks,
                                       backtest each, show summary table

  python thermo_scanner.py today    → run today's signal check on all
                                       approved stocks, print entries to take,
                                       update open paper trades from yesterday

Paper trade log saved to: paper_trades/paper_log.csv
Approved stocks saved to: paper_trades/approved_stocks.csv
"""
from __future__ import annotations
import logging, math, sys, warnings
from datetime import datetime, date
from pathlib import Path
import numpy as np, pandas as pd, yfinance as yf
from scipy.stats import entropy as scipy_entropy

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("thermo.scanner")

# ── Settings ──────────────────────────────────────────────────────────────────
STOP_PCT     = 0.010    # 1.0% stop loss (wider — survives normal ATR noise)
TARGET_PCT   = 0.020    # 2.0% take profit  →  2:1 R:R, break-even WR = 33%
ENTRY_START  = 0        # enter bars 0–2 (10:00–12:00 TW)
ENTRY_WINDOW = 3
LOT          = 1000     # 1 lot = 1000 shares
RISK_PCT     = 0.02

# Pre-filter thresholds
MAX_ATR_PCT  = 2.0      # reject if 1h ATR > 2% of price (too noisy)
MIN_PRICE    = 15.0     # reject penny stocks
MIN_WR       = 40.0     # reject if backtest WR < 40%
MIN_PPROFIT  = 60.0     # reject if bootstrap P(profit) < 60%

# Feature windows
W_LOCAL=20; W_SHORT=3; W_MOM_L=10; W_ATR=14; W_ENTROPY=8; W_TREND_MA=15

# ── Taiwan stock universe ─────────────────────────────────────────────────────
UNIVERSE = {
    # === Semiconductors ===
    "2330.TW": "TSMC",
    "2303.TW": "UMC",
    "2454.TW": "MediaTek",
    "3034.TW": "Novatek",
    "3711.TW": "ASE Tech",
    "2379.TW": "Realtek",
    "2408.TW": "Nanya Tech",
    "2337.TW": "Macronix",
    "3008.TW": "Largan",
    "6415.TW": "Silergy",
    "2449.TW": "King Yuan",
    "3533.TW": "Parade Tech",
    # === Electronics / Contract Mfg ===
    "2317.TW": "Foxconn",
    "2382.TW": "Quanta",
    "2357.TW": "ASUS",
    "2376.TW": "Gigabyte",
    "2395.TW": "Advantech",
    "2345.TW": "Acer",
    "2385.TW": "Inventec",
    "3231.TW": "Wistron",
    "2324.TW": "Compal",
    "2301.TW": "Lite-On",
    "2354.TW": "Foxconn Tech",
    "6669.TW": "Wiwynn",
    # === Industrial / Components ===
    "2308.TW": "Delta Elec",
    "2207.TW": "Hotai Motor",
    "2409.TW": "AUO",
    "1605.TW": "Walsin Wire",
    "2049.TW": "Hiwin",
    "1590.TW": "Airtac",
    # === Telecom ===
    "2412.TW": "Chunghwa Tel",
    "3045.TW": "TW Mobile",
    "4904.TW": "Far EasTone",
    # === Financials ===
    "2881.TW": "Fubon Fin",
    "2882.TW": "Cathay Fin",
    "2884.TW": "E.Sun Fin",
    "2885.TW": "Yuanta Fin",
    "2886.TW": "Mega Fin",
    "2887.TW": "Taishin Fin",
    "2891.TW": "CTBC Fin",
    "2892.TW": "First Fin",
    "5876.TW": "SinoPac Fin",
    "2883.TW": "KGI Sec",
    # === ETFs ===
    "0050.TW": "TW50 ETF",
    "0056.TW": "TW Dividend ETF",
    "00631L.TW": "TW Bull 2x",
    # === Materials / Energy ===
    "2002.TW": "China Steel",
    "1301.TW": "Formosa Plastics",
    "1303.TW": "Nan Ya Plastics",
    "6505.TW": "Formosa Petro",
    "1326.TW": "Formosa Chemical",
    # === Aviation / Retail ===
    "2618.TW": "EVA Air",
    "2610.TW": "China Airlines",
    "2912.TW": "President Chain",
}

LOG_DIR       = Path("paper_trades")
LOG_FILE      = LOG_DIR / "paper_log.csv"
APPROVED_FILE = LOG_DIR / "approved_stocks.csv"


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
    fd["entropy"]     = (1 - _pe(ret, W_ENTROPY, 3).rolling(W_LOCAL).rank(pct=True)) * 100
    ma = c.rolling(W_LOCAL).mean(); fd["dist_ma"] = (c - ma) / ma * 100
    fd["mom_s"]     = c.pct_change(W_SHORT) * 100
    fd["mom_l"]     = c.pct_change(W_MOM_L) * 100
    h, l = df["High"], df["Low"]
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    fd["atr"]       = tr.rolling(W_ATR).mean()
    fd["trend_ma"]  = c.rolling(W_TREND_MA).mean()
    fd["ma_rising"] = (fd["trend_ma"].diff() > 0).astype(int)
    return fd.dropna()

def signal(row) -> str:
    er,vc,es,ms,ml,dst = (row["energy_rank"], row["vol_c"], row["entropy"],
                           row["mom_s"], row["mom_l"], row["dist_ma"])
    up = bool(row["ma_rising"])
    if up:
        if er>60 and vc<1.0 and ms>0 and ml>0 and es>35: return "LONG"
        if er>50 and ml<-3 and dst<-1.5 and ms>-0.5:     return "LONG"
    else:
        if er>60 and vc<1.0 and ms<0 and ml<0 and es>35: return "SHORT"
        if er<45 and ml>3  and dst>1.5 and ms<0.5:        return "SHORT"
    return "NEUTRAL"


# ── Backtest (1 fixed lot) ────────────────────────────────────────────────────

def backtest(fd: pd.DataFrame) -> pd.DataFrame:
    fd = fd.copy()
    fd.index = fd.index.tz_localize(None) if fd.index.tz else fd.index
    fd["td"] = fd.index.normalize(); trades = []
    for td, day in fd.groupby("td"):
        day = day.sort_index()
        if len(day) < 2: continue
        pos = None; done = False
        for bi, (ts, row) in enumerate(day.iterrows()):
            pr=float(row["Close"]); hi=float(row["High"]); lo=float(row["Low"])
            is_last=(ts==day.index[-1]); in_w=(bi<ENTRY_WINDOW)
            if pos:
                ep = rsn = None
                if pos["s"] == "LONG":
                    if lo <= pos["sl"]:  ep, rsn = pos["sl"],  "STOP"
                    elif hi >= pos["tp"]: ep, rsn = pos["tp"], "TARGET"
                else:
                    if hi >= pos["sl"]:  ep, rsn = pos["sl"],  "STOP"
                    elif lo <= pos["tp"]: ep, rsn = pos["tp"], "TARGET"
                if ep is None and is_last: ep, rsn = pr, "EOD"
                if ep is not None:
                    pts = (ep-pos["e"]) if pos["s"]=="LONG" else (pos["e"]-ep)
                    trades.append({"date": td, "side": pos["s"], "entry": pos["e"],
                                   "exit": ep, "pnl": pts*LOT, "pct": pts/pos["e"]*100,
                                   "reason": rsn, "result": "WIN" if pts>0 else "LOSS"})
                    pos = None; done = True
            if pos is None and not done and in_w and not is_last:
                s = signal(row)
                if s != "NEUTRAL":
                    sl = pr*(1-STOP_PCT) if s=="LONG" else pr*(1+STOP_PCT)
                    tp = pr*(1+TARGET_PCT) if s=="LONG" else pr*(1-TARGET_PCT)
                    pos = {"s": s, "e": pr, "sl": sl, "tp": tp}
    return pd.DataFrame(trades)

def bootstrap_pprofit(fd: pd.DataFrame, n_boot=500, days=300) -> float:
    fd = fd.copy()
    fd.index = fd.index.tz_localize(None) if fd.index.tz else fd.index
    fd["td"] = fd.index.normalize(); day_ret = {}
    for td, day in fd.groupby("td"):
        day = day.sort_index()
        if len(day) < 2: day_ret[td] = 0.0; continue
        pos = None; r = 0.0
        for bi, (ts, row) in enumerate(day.iterrows()):
            pr=float(row["Close"]); hi=float(row["High"]); lo=float(row["Low"])
            is_last=(ts==day.index[-1]); in_w=(bi<ENTRY_WINDOW)
            if pos:
                ep = None
                if pos["s"]=="LONG":
                    if lo<=pos["sl"]: ep=pos["sl"]
                    elif hi>=pos["tp"]: ep=pos["tp"]
                else:
                    if hi>=pos["sl"]: ep=pos["sl"]
                    elif lo<=pos["tp"]: ep=pos["tp"]
                if ep is None and is_last: ep = pr
                if ep is not None:
                    r = ((ep-pos["e"]) if pos["s"]=="LONG" else (pos["e"]-ep)) / pos["e"]
                    pos = None
            if pos is None and r==0.0 and in_w and not is_last:
                s = signal(row)
                if s != "NEUTRAL":
                    sl = pr*(1-STOP_PCT) if s=="LONG" else pr*(1+STOP_PCT)
                    tp = pr*(1+TARGET_PCT) if s=="LONG" else pr*(1-TARGET_PCT)
                    pos = {"s": s, "e": pr, "sl": sl, "tp": tp}
        day_ret[td] = r
    arr = np.array(list(day_ret.values()))
    rng = np.random.default_rng(42); finals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(arr), size=days)
        cap = 1.0
        for rv in arr[idx]:
            if rv != 0.0: cap *= (1 + rv * RISK_PCT)
        finals.append(cap - 1)
    return (np.array(finals) > 0).mean() * 100

def atr_pct(raw: pd.DataFrame) -> float:
    h, l, c = raw["High"], raw["Low"], raw["Close"]
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().dropna()
    return float(atr.iloc[-20:].mean() / c.iloc[-1] * 100)


# ── Mode 1: Universe scan ─────────────────────────────────────────────────────

def run_scan():
    LOG_DIR.mkdir(exist_ok=True)
    log.info("Scanning %d stocks in universe ...", len(UNIVERSE))
    rows = []; approved = []

    for ticker, name in UNIVERSE.items():
        try:
            raw = yf.download(ticker, period="max", interval="1h",
                              auto_adjust=True, progress=False)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            raw = raw.dropna(subset=["Close"])
            raw.index = raw.index.tz_convert("Asia/Taipei")

            price = float(raw["Close"].iloc[-1])
            if price < MIN_PRICE:
                log.info("  %-12s  SKIP  price=NT$%.0f", name, price)
                continue

            a_pct = atr_pct(raw)
            if a_pct > MAX_ATR_PCT:
                log.info("  %-12s  REJECT  ATR%%=%.2f%% (too noisy)", name, a_pct)
                rows.append({"Ticker": ticker, "Name": name, "Status": "REJECTED",
                              "ATR%": round(a_pct, 2), "Reason": f"ATR>{MAX_ATR_PCT}%"})
                continue

            fd  = build_features(raw)
            tr  = backtest(fd)
            if tr.empty:
                log.info("  %-12s  SKIP  no trades generated", name); continue

            n    = len(tr)
            wr   = (tr.result == "WIN").mean() * 100
            total= tr.pnl.sum()
            days = raw.index.normalize().nunique()
            freq = n / days * 245

            pp = bootstrap_pprofit(fd)

            ok = wr >= MIN_WR and pp >= MIN_PPROFIT
            status = "PASS" if ok else "FAIL"
            reason = "" if ok else (
                f"WR={wr:.0f}%<{MIN_WR}%" if wr < MIN_WR else f"P={pp:.0f}%<{MIN_PPROFIT}%")

            rows.append({"Ticker": ticker, "Name": name, "Status": status,
                          "Price": round(price), "ATR%": round(a_pct, 2),
                          "Trades": n, "WR%": round(wr, 1),
                          "Total PnL": round(total), "Signals/yr": round(freq),
                          "P(profit)%": round(pp, 1), "Reason": reason})

            if ok:
                approved.append({"ticker": ticker, "name": name,
                                  "atr_pct": round(a_pct, 2), "wr": round(wr, 1),
                                  "signals_yr": round(freq)})

            log.info("  %-12s  %-6s  Trades=%3d  WR=%4.0f%%  P(profit)=%4.0f%%  ATR=%.2f%%",
                     name, status, n, wr, pp, a_pct)

        except Exception as e:
            log.warning("  %-12s  ERROR: %s", name, e)

    df = pd.DataFrame(rows)
    pass_df = df[df.Status == "PASS"] if not df.empty else pd.DataFrame()
    fail_df = df[df.Status != "PASS"] if not df.empty else pd.DataFrame()

    print(f"\n{'='*78}")
    print(f"  APPROVED STOCKS — pass ATR<{MAX_ATR_PCT}%  WR>{MIN_WR}%  P(profit)>{MIN_PPROFIT}%")
    print(f"{'='*78}")
    if not pass_df.empty:
        print(pass_df[["Ticker","Name","Price","ATR%","Trades","WR%",
                        "Total PnL","Signals/yr","P(profit)%"]].to_string(index=False))

    print(f"\n{'='*78}")
    print(f"  REJECTED / FAILED")
    print(f"{'='*78}")
    if not fail_df.empty:
        cols = [c for c in ["Ticker","Name","ATR%","WR%","P(profit)%","Reason"] if c in fail_df.columns]
        print(fail_df[cols].to_string(index=False))

    if approved:
        ap_df = pd.DataFrame(approved)
        ap_df.to_csv(APPROVED_FILE, index=False)
        log.info("Approved list saved → %s", APPROVED_FILE)

        total_yr = ap_df["signals_yr"].sum()
        print(f"\n{'='*78}")
        print(f"  COMBINED SIGNAL FREQUENCY  ({len(approved)} approved stocks)")
        print(f"{'='*78}")
        print(f"  Total signals/year : ~{total_yr:.0f} across all approved stocks")
        print(f"  Reach 30 trades in : ~{30/total_yr*12:.0f} months")
        print(f"  Reach 50 trades in : ~{50/total_yr*12:.0f} months")
        print(f"  Reach 100 trades in: ~{100/total_yr*12:.0f} months")
        print(f"\n  Rule: if multiple stocks signal same day,")
        print(f"        take the one with highest energy_rank.")


# ── Mode 2: Today's signals ───────────────────────────────────────────────────

def run_today():
    LOG_DIR.mkdir(exist_ok=True)

    if not APPROVED_FILE.exists():
        log.error("Run scan first: python thermo_scanner.py scan")
        return

    approved = pd.read_csv(APPROVED_FILE)
    today    = str(date.today())
    print(f"\n  Thermo Scanner — {datetime.now().strftime('%Y-%m-%d %H:%M')} TW")
    print(f"  Scanning {len(approved)} approved stocks\n")

    signals = []
    for _, row in approved.iterrows():
        ticker = row["ticker"]; name = row["name"]
        try:
            raw = yf.download(ticker, period="5d", interval="1h",
                              auto_adjust=True, progress=False)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            raw = raw.dropna(subset=["Close"])
            raw.index = raw.index.tz_convert("Asia/Taipei")

            fd     = build_features(raw)
            if fd.empty: continue
            latest = fd.iloc[-1]
            price  = float(latest["Close"])
            sig    = signal(latest)
            ma_dir = "RISING" if bool(latest["ma_rising"]) else "FALLING"
            er     = float(latest["energy_rank"])
            vc     = float(latest["vol_c"])
            tp = price*(1+TARGET_PCT) if sig=="LONG" else price*(1-TARGET_PCT)
            sl = price*(1-STOP_PCT)   if sig=="LONG" else price*(1+STOP_PCT)
            signals.append({"ticker":ticker,"name":name,"signal":sig,
                             "price":price,"ma":ma_dir,"energy":round(er),"vol_c":round(vc,2),
                             "tp":round(tp,1),"sl":round(sl,1)})
        except Exception as e:
            log.warning("  %s ERROR: %s", ticker, e)

    active = sorted([s for s in signals if s["signal"]!="NEUTRAL"],
                    key=lambda x: -x["energy"])  # rank by energy

    if not active:
        print("  No signals today. MA directions:")
        for s in signals:
            print(f"    {s['ticker']} ({s['name']:<12})  MA15: {s['ma']}")
    else:
        print(f"  {len(active)} signal(s) today (sorted by energy rank):\n")
        for i, s in enumerate(active, 1):
            tag = " ← BEST ENTRY" if i == 1 else ""
            print(f"  [{i}] {s['ticker']} ({s['name']})  {s['signal']}{tag}")
            print(f"      Entry: NT${s['price']:.1f}  |  TP: NT${s['tp']:.1f} (+2%)  |  SL: NT${s['sl']:.1f} (-0.5%)")
            print(f"      Energy: {s['energy']}  |  Vol compression: {s['vol_c']}  |  MA15: {s['ma']}")
            print()
        log_trades(active, today)

    show_open_trades()


def log_trades(signals: list, trade_date: str):
    cols = ["date","ticker","name","side","entry","tp","sl",
            "exit","exit_reason","pnl_lot","status"]
    df = pd.read_csv(LOG_FILE) if LOG_FILE.exists() else pd.DataFrame(columns=cols)
    new = []
    for s in signals:
        if df.empty or df[(df.ticker==s["ticker"]) &
                          (df.date==trade_date) &
                          (df.status=="OPEN")].empty:
            new.append({"date":trade_date,"ticker":s["ticker"],"name":s["name"],
                        "side":s["signal"],"entry":s["price"],"tp":s["tp"],"sl":s["sl"],
                        "exit":None,"exit_reason":None,"pnl_lot":None,"status":"OPEN"})
    if new:
        df = pd.concat([df, pd.DataFrame(new)], ignore_index=True)
        df.to_csv(LOG_FILE, index=False)
        log.info("Logged %d new paper trade(s) → %s", len(new), LOG_FILE)


def show_open_trades():
    if not LOG_FILE.exists(): return
    df = pd.read_csv(LOG_FILE)
    open_t  = df[df.status == "OPEN"]
    closed  = df[df.status == "CLOSED"]

    if not open_t.empty:
        print(f"\n  ── Open paper trades ({len(open_t)}) ─────────────────────────")
        for _, r in open_t.iterrows():
            print(f"    {r['date']}  {r['ticker']} {r['side']}"
                  f"  entry=NT${r['entry']:.1f}  TP=NT${r['tp']:.1f}  SL=NT${r['sl']:.1f}")

    if not closed.empty:
        closed = closed.copy()
        closed["pnl_lot"] = pd.to_numeric(closed["pnl_lot"], errors="coerce")
        total = closed.pnl_lot.sum(); n = len(closed)
        nw = (closed.pnl_lot > 0).sum()
        print(f"\n  ── Paper trade history ({n} closed) ──────────────────────")
        print(f"    Win/Loss  : {nw}W / {n-nw}L  (WR={nw/n*100:.0f}%)")
        print(f"    Total PnL : NT${total:+,.0f}")
        print(f"    Progress  : {n}/30 trades  ({max(0,30-n)} more for first milestone)")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "scan"
    if mode == "scan":
        log.info("=== Universe Scan Mode ===")
        run_scan()
    elif mode == "today":
        log.info("=== Today Signal Mode ===")
        run_today()
    else:
        print("Usage:")
        print("  python thermo_scanner.py scan    # find qualifying stocks (run once)")
        print("  python thermo_scanner.py today   # get today's signals + log trades")
