#!/usr/bin/env python3
"""
thermo_tw_basket.py — Full Thermo Strategy on TW Stock Basket
=============================================================
Tests the bi-directional (LONG + SHORT) strategy on 7 liquid TW names.
Position sizing uses TW lot convention (1 lot = 1000 shares).
All signals are direction-gated by MA15 slope.

Tickers:
  2330.TW  TSMC
  2317.TW  Foxconn (Hon Hai)
  2454.TW  MediaTek
  2308.TW  Delta Electronics
  3008.TW  Largan Precision
  2382.TW  Quanta Computer
  0050.TW  Taiwan 50 ETF
"""
from __future__ import annotations
import logging, math, warnings
import numpy as np, pandas as pd
import yfinance as yf
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import entropy as scipy_entropy

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("thermo.basket")

# ── Settings ──────────────────────────────────────────────────────────────────
TICKERS = {
    "2330.TW": "TSMC",
    "2317.TW": "Foxconn",
    "2454.TW": "MediaTek",
    "2308.TW": "Delta",
    "3008.TW": "Largan",
    "2382.TW": "Quanta",
    "0050.TW": "TW50 ETF",
}
INTERVAL      = "1h"
INITIAL_CAP   = 1_000_000   # NT$1M — large enough for lot-based sizing
LOT_SIZE      = 1_000       # TW stocks: 1 lot = 1000 shares
RISK_PCT      = 0.02
ATR_STOP      = 0.5
ATR_TARGET    = 1.0
ENTRY_WINDOW  = 3           # bars 0,1,2 = 09:00–11:00 TW only
W_LOCAL       = 20
W_SHORT       = 3
W_MOM_L       = 10
W_ATR         = 14
W_ENTROPY     = 8
W_TREND_MA    = 15          # 3-day MA on hourly bars

# Signal thresholds
BRK_ENERGY     = 60
BRK_VOL_C      = 1.0
BRK_ENTROPY    = 35
REVL_ENERGY    = 50
REVL_MOM_L_MAX = -3
REVL_DIST      = 1.5
REVL_MOM_S_MAX = -0.5
REVS_ENERGY_MAX = 45
REVS_MOM_L_MIN  = 3
REVS_DIST       = 1.5
REVS_MOM_S_MAX  = 0.5

N_BOOT        = 1000
BOOT_DAYS     = 300


# ── Features ──────────────────────────────────────────────────────────────────

def _perm_entropy(series: pd.Series, w: int = 8, order: int = 3) -> pd.Series:
    log_d = math.log(math.factorial(order))
    vals  = series.values; n = len(vals)
    out   = np.full(n, np.nan)
    for i in range(w - 1, n):
        seg  = vals[i - w + 1: i + 1]
        pats = np.array([tuple(np.argsort(seg[j:j+order]))
                         for j in range(len(seg) - order + 1)])
        _, cnt = np.unique(pats, axis=0, return_counts=True)
        p = cnt / cnt.sum()
        out[i] = scipy_entropy(p) / log_d
    return pd.Series(out, index=series.index)

def _atr(df: pd.DataFrame, w: int) -> pd.Series:
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(w).mean()

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    c   = df["Close"].copy()
    ret = np.log(c / c.shift(1))
    fd  = df[["High", "Low", "Close"]].copy()

    fd["energy_rank"] = (df["High"] - df["Low"]).rolling(W_LOCAL).rank(pct=True) * 100
    vol_s       = ret.rolling(W_SHORT).std()
    vol_l       = ret.rolling(W_LOCAL).std()
    fd["vol_c"] = vol_s / (vol_l + 1e-9)
    pe             = _perm_entropy(ret, W_ENTROPY, 3)
    fd["entropy"]  = (1 - pe.rolling(W_LOCAL).rank(pct=True)) * 100
    ma             = c.rolling(W_LOCAL).mean()
    fd["dist_ma"]  = (c - ma) / ma * 100
    fd["mom_s"]    = c.pct_change(W_SHORT) * 100
    fd["mom_l"]    = c.pct_change(W_MOM_L) * 100
    fd["atr"]      = _atr(df, W_ATR)
    fd["trend_ma"] = c.rolling(W_TREND_MA).mean()
    fd["ma_rising"]= (fd["trend_ma"].diff() > 0).astype(int)

    return fd.dropna()


# ── Signal ────────────────────────────────────────────────────────────────────

def signal(row) -> str:
    er  = row["energy_rank"]; vc = row["vol_c"]; es = row["entropy"]
    ms  = row["mom_s"];       ml = row["mom_l"]; dst = row["dist_ma"]
    up  = bool(row["ma_rising"])

    if up:
        if er > BRK_ENERGY and vc < BRK_VOL_C and ms > 0 and ml > 0 and es > BRK_ENTROPY:
            return "LONG"
        if er > REVL_ENERGY and ml < REVL_MOM_L_MAX and dst < -REVL_DIST and ms > REVL_MOM_S_MAX:
            return "LONG"
    else:
        if er > BRK_ENERGY and vc < BRK_VOL_C and ms < 0 and ml < 0 and es > BRK_ENTROPY:
            return "SHORT"
        if er < REVS_ENERGY_MAX and ml > REVS_MOM_L_MIN and dst > REVS_DIST and ms < REVS_MOM_S_MAX:
            return "SHORT"
    return "NEUTRAL"


# ── Simulation ────────────────────────────────────────────────────────────────

def simulate(fd: pd.DataFrame) -> dict:
    fd = fd.copy()
    fd.index = fd.index.tz_localize(None) if fd.index.tz else fd.index
    fd["trade_date"] = fd.index.normalize()
    trading_days = sorted(fd["trade_date"].unique())

    cash   = INITIAL_CAP
    trades = []
    eq     = []

    for td in trading_days:
        day  = fd[fd["trade_date"] == td].sort_index()
        if len(day) < 2:
            eq.append((td, cash)); continue

        pos       = None
        day_trade = None

        for bar_idx, (ts, row) in enumerate(day.iterrows()):
            price   = float(row["Close"])
            hi      = float(row["High"])
            lo      = float(row["Low"])
            atr_val = float(row["atr"])
            is_last = ts == day.index[-1]
            in_win  = bar_idx < ENTRY_WINDOW

            if pos:
                ep = reason = None
                if pos["side"] == "LONG":
                    if lo  <= pos["stop"]:   ep, reason = pos["stop"],   "STOP"
                    elif hi >= pos["target"]: ep, reason = pos["target"], "TARGET"
                else:
                    if hi >= pos["stop"]:    ep, reason = pos["stop"],   "STOP"
                    elif lo <= pos["target"]: ep, reason = pos["target"], "TARGET"
                if ep is None and is_last:   ep, reason = price,         "EOD"

                if ep is not None:
                    pnl = ((ep - pos["entry"]) if pos["side"] == "LONG"
                           else (pos["entry"] - ep)) * pos["lots"] * LOT_SIZE
                    cash += pnl
                    day_trade = {
                        "date": td, "side": pos["side"],
                        "entry": pos["entry"], "exit": ep,
                        "pnl": pnl, "reason": reason,
                        "result": "WIN" if pnl > 0 else "LOSS",
                        "pts": (ep - pos["entry"]) if pos["side"] == "LONG"
                               else (pos["entry"] - ep),
                        "lots": pos["lots"],
                    }
                    trades.append(day_trade)
                    pos = None

            if pos is None and day_trade is None and in_win and not is_last and atr_val > 0:
                sig = signal(row)
                if sig == "NEUTRAL":
                    continue
                risk_per_lot = ATR_STOP * atr_val * LOT_SIZE
                lots = max(1, int((cash * RISK_PCT) / risk_per_lot))
                stop   = (price - ATR_STOP  * atr_val) if sig == "LONG" else (price + ATR_STOP  * atr_val)
                target = (price + ATR_TARGET * atr_val) if sig == "LONG" else (price - ATR_TARGET * atr_val)
                pos = {"side": sig, "entry": price, "stop": stop, "target": target, "lots": lots}

        eq.append((td, cash))

    return {"equity": eq, "trades": trades}


# ── Bootstrap ─────────────────────────────────────────────────────────────────

def bootstrap(fd: pd.DataFrame) -> pd.Series:
    fd = fd.copy()
    fd.index = fd.index.tz_localize(None) if fd.index.tz else fd.index
    fd["trade_date"] = fd.index.normalize()
    all_days = sorted(fd["trade_date"].unique())

    day_ret = {}
    for td in all_days:
        day = fd[fd["trade_date"] == td].sort_index()
        if len(day) < 2:
            day_ret[td] = 0.0; continue
        pos = None; ret_day = 0.0
        for bar_idx, (ts, row) in enumerate(day.iterrows()):
            price   = float(row["Close"])
            hi      = float(row["High"])
            lo      = float(row["Low"])
            atr_val = float(row["atr"])
            is_last = ts == day.index[-1]
            in_win  = bar_idx < ENTRY_WINDOW
            if pos:
                ep = reason = None
                if pos["side"] == "LONG":
                    if lo  <= pos["stop"]:   ep, reason = pos["stop"],   "STOP"
                    elif hi >= pos["target"]: ep, reason = pos["target"], "TARGET"
                else:
                    if hi >= pos["stop"]:    ep, reason = pos["stop"],   "STOP"
                    elif lo <= pos["target"]: ep, reason = pos["target"], "TARGET"
                if ep is None and is_last: ep, reason = price, "EOD"
                if ep is not None:
                    ret_day = ((ep - pos["entry"]) if pos["side"] == "LONG"
                               else (pos["entry"] - ep)) / pos["entry"]
                    pos = None
            if pos is None and ret_day == 0.0 and in_win and not is_last and atr_val > 0:
                sig = signal(row)
                if sig != "NEUTRAL":
                    stop   = (price - ATR_STOP  * atr_val) if sig == "LONG" else (price + ATR_STOP  * atr_val)
                    target = (price + ATR_TARGET * atr_val) if sig == "LONG" else (price - ATR_TARGET * atr_val)
                    pos = {"side": sig, "entry": price, "stop": stop, "target": target}
        day_ret[td] = ret_day

    arr = np.array([day_ret[d] for d in all_days])
    rng = np.random.default_rng(42)
    finals = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(arr), size=BOOT_DAYS)
        cap = INITIAL_CAP
        for r in arr[idx]:
            if r != 0.0:
                # scale: risk 2% per trade, return = r × (RISK_PCT / ATR_STOP) / ATR_STOP
                cap *= (1 + r * RISK_PCT)
        finals.append((cap / INITIAL_CAP - 1) * 100)
    return pd.Series(finals)


# ── Summary table row ─────────────────────────────────────────────────────────

def summarize(ticker: str, name: str, results: dict, bs: pd.Series) -> dict:
    dates, vals = zip(*results["equity"])
    eq     = pd.Series(list(vals), index=list(dates))
    trades = pd.DataFrame(results["trades"])

    ret    = (eq.iloc[-1] / INITIAL_CAP - 1) * 100
    dchg   = eq.pct_change().dropna()
    sharpe = dchg.mean() / dchg.std() * np.sqrt(252) if dchg.std() > 0 else 0
    dd     = (eq - eq.cummax()) / eq.cummax() * 100
    max_dd = dd.min()

    n  = len(trades) if not trades.empty else 0
    wr = (trades["result"] == "WIN").mean() * 100 if n > 0 else 0
    wl = 0.0
    if n > 0:
        aw = trades[trades["result"]=="WIN"]["pnl"].mean()  if (trades["result"]=="WIN").any()  else 0
        al = trades[trades["result"]=="LOSS"]["pnl"].mean() if (trades["result"]=="LOSS").any() else 1
        wl = abs(aw / al) if al != 0 else 0

    pp = (bs > 0).mean() * 100

    return {
        "Ticker":     ticker,
        "Name":       name,
        "Return%":    round(ret, 1),
        "Sharpe":     round(sharpe, 2),
        "MaxDD%":     round(max_dd, 1),
        "Trades":     n,
        "WR%":        round(wr, 1),
        "W/L":        round(wl, 2),
        "P(profit)%": round(pp, 1),
        "Median%":    round(bs.median(), 1),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os; os.makedirs("results_sprint", exist_ok=True)

    rows    = []
    eq_dict = {}
    dd_dict = {}
    fd_dict = {}

    for ticker, name in TICKERS.items():
        log.info("─── %s (%s) ───", ticker, name)
        try:
            raw = yf.download(ticker, period="max", interval=INTERVAL,
                              auto_adjust=True, progress=False)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            raw = raw.dropna(subset=["Close"])
            raw.index = raw.index.tz_convert("Asia/Taipei")
            log.info("  %d bars  price=NT$%.0f–NT$%.0f",
                     len(raw), raw["Close"].min(), raw["Close"].max())

            fd = build_features(raw)
            fd_dict[ticker] = fd

            results = simulate(fd)
            log.info("  Bootstrap ...")
            bs = bootstrap(fd)
            row = summarize(ticker, name, results, bs)
            rows.append(row)

            dates, vals = zip(*results["equity"])
            eq = pd.Series(list(vals), index=list(dates))
            dd = (eq - eq.cummax()) / eq.cummax() * 100
            eq_dict[ticker] = eq
            dd_dict[ticker] = dd

            log.info("  Return=%+.1f%%  Sharpe=%.2f  WR=%.1f%%  P(profit)=%.1f%%",
                     row["Return%"], row["Sharpe"], row["WR%"], row["P(profit)%"])

        except Exception as e:
            log.warning("  FAILED: %s", e)

    # ── Print comparison table ────────────────────────────────────────────────
    df_summary = pd.DataFrame(rows).set_index("Ticker")
    print("\n" + "="*80)
    print("  THERMO STRATEGY — TW STOCK BASKET  |  NT$1,000,000 capital  |  1h bars")
    print("  MA15 slope filter + thermo signal  |  Entry: 09:00–11:00  |  Exit: EOD/SL/TP")
    print("="*80)
    print(df_summary.to_string())
    print("="*80)

    # rank by Sharpe
    ranked = df_summary.sort_values("Sharpe", ascending=False)
    print("\n  Ranked by Sharpe:")
    for i, (t, r) in enumerate(ranked.iterrows(), 1):
        print(f"  {i}. {t} ({r['Name']:12s})  "
              f"Return={r['Return%']:+5.1f}%  Sharpe={r['Sharpe']:.2f}  "
              f"WR={r['WR%']:.0f}%  P(profit)={r['P(profit)%']:.0f}%")

    # ── Equity curve grid ─────────────────────────────────────────────────────
    n_tickers = len(eq_dict)
    ncols = 4; nrows = math.ceil(n_tickers / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, nrows * 4))
    axes = axes.flatten()

    colors = ["crimson","darkorange","steelblue","seagreen","purple","goldenrod","teal"]

    for i, (ticker, eq) in enumerate(eq_dict.items()):
        name = TICKERS[ticker]
        dd   = dd_dict[ticker]
        row  = df_summary.loc[ticker]
        clr  = colors[i % len(colors)]

        eq_idx = eq.index.tz_localize(None) if eq.index.tz else eq.index
        ax = axes[i]
        ax.plot(eq_idx, eq / 1e3, color=clr, lw=1.8)
        ax.fill_between(eq_idx, eq / 1e3,
                        INITIAL_CAP / 1e3, alpha=0.15,
                        color="green" if row["Return%"] > 0 else "red")
        ax.axhline(INITIAL_CAP / 1e3, color="gray", lw=0.8, ls="--")
        ax.set_title(f"{ticker} {name}\n"
                     f"Ret={row['Return%']:+.1f}%  Sharpe={row['Sharpe']:.2f}  "
                     f"MaxDD={row['MaxDD%']:.1f}%",
                     fontsize=9)
        ax.set_ylabel("Equity (NT$K)"); ax.grid(alpha=0.3)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Thermo Strategy — TW Basket  |  1h bars  |  MA15 + Thermo signal  |  "
                 "LONG when MA↑ / SHORT when MA↓  |  Entry 09:00–11:00  |  NT$1M capital",
                 fontsize=11)
    plt.tight_layout()
    out = "results_sprint/thermo_tw_basket.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    log.info("Chart -> %s", out)
    print(f"\n  Chart -> {out}")
