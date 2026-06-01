#!/usr/bin/env python3
"""
thermo_longonly.py — Thermo LONG-ONLY Intraday Strategy
========================================================
Entry: LONG signal only (MA rising + thermo breakout/reversal)
Exit:  TARGET hit  OR  STOP hit  OR  EOD (same day)
No overnight positions — all exits before market close.

Runs on:
  1. TWII Index (^TWII) — as TMF futures proxy
  2. TSMC (2330.TW)    — as individual stock

Signal gating:
  MA15 must be rising → only LONG allowed
  Thermo breakout or reversal must confirm direction
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
log = logging.getLogger("thermo.longonly")

# ── Settings ──────────────────────────────────────────────────────────────────
INTERVAL     = "1h"
ENTRY_WINDOW = 3          # only first 3 bars (09:00–11:00 TW)
RISK_PCT     = 0.02       # 2% of capital per trade
ATR_STOP     = 1.0
ATR_TARGET   = 2.0

# Feature windows
W_LOCAL   = 20
W_SHORT   = 3
W_MOM_L   = 10
W_ATR     = 14
W_ENTROPY = 8
W_TREND_MA = 15           # 3-day MA (15 hourly bars)

# Signal thresholds (same as thermo_clean.py)
BRK_ENERGY  = 60
BRK_VOL_C   = 1.0
BRK_ENTROPY = 35
REVL_ENERGY    = 50
REVL_MOM_L_MAX = -3
REVL_DIST      = 1.5
REVL_MOM_S_MAX = -0.5


# ── Feature builder ───────────────────────────────────────────────────────────

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
    fd  = df[["High", "Low", "Close", "Open"]].copy()

    candle_range      = df["High"] - df["Low"]
    fd["energy_rank"] = candle_range.rolling(W_LOCAL).rank(pct=True) * 100

    vol_s       = ret.rolling(W_SHORT).std()
    vol_l       = ret.rolling(W_LOCAL).std()
    fd["vol_c"] = vol_s / (vol_l + 1e-9)

    pe             = _perm_entropy(ret, W_ENTROPY, 3)
    ent_rank       = pe.rolling(W_LOCAL).rank(pct=True)
    fd["entropy"]  = (1 - ent_rank) * 100

    ma             = c.rolling(W_LOCAL).mean()
    fd["dist_ma"]  = (c - ma) / ma * 100

    fd["mom_s"] = c.pct_change(W_SHORT) * 100
    fd["mom_l"] = c.pct_change(W_MOM_L) * 100
    fd["atr"]   = _atr(df, W_ATR)

    fd["trend_ma"]  = c.rolling(W_TREND_MA).mean()
    fd["ma_rising"] = (fd["trend_ma"].diff() > 0).astype(int)

    return fd.dropna()


# ── Signal: LONG only when MA rising ─────────────────────────────────────────

def signal_long(row) -> bool:
    """Returns True if LONG signal fires (MA must be rising)."""
    if not bool(row["ma_rising"]):
        return False           # MA falling → skip entirely (no short)

    er  = row["energy_rank"]
    vc  = row["vol_c"]
    es  = row["entropy"]
    ms  = row["mom_s"]
    ml  = row["mom_l"]
    dst = row["dist_ma"]

    # Breakout LONG
    if er > BRK_ENERGY and vc < BRK_VOL_C and ms > 0 and ml > 0 and es > BRK_ENTROPY:
        return True
    # Reversal LONG
    if er > REVL_ENERGY and ml < REVL_MOM_L_MAX and dst < -REVL_DIST and ms > REVL_MOM_S_MAX:
        return True

    return False


# ── Simulation ────────────────────────────────────────────────────────────────

def simulate(fd: pd.DataFrame, initial_cap: float, multiplier: float = 1.0,
             margin: float = 0.0, max_margin: float = 1.0) -> dict:
    """
    multiplier = 1.0 for stocks (1 share = 1 unit)
                10.0 for TWII futures (NT$10/point)
    margin     = 0 for stocks, 25000 for TMF futures
    """
    fd = fd.copy()
    fd.index = fd.index.tz_localize(None) if fd.index.tz else fd.index
    fd["trade_date"] = fd.index.normalize()
    trading_days = sorted(fd["trade_date"].unique())

    cash     = initial_cap
    trades   = []
    eq_curve = []

    for trade_date in trading_days:
        day_bars = fd[fd["trade_date"] == trade_date].sort_index()
        if len(day_bars) < 2:
            eq_curve.append((trade_date, cash)); continue

        position  = None
        day_trade = None

        for bar_idx, (ts, row) in enumerate(day_bars.iterrows()):
            price   = float(row["Close"])
            hi      = float(row["High"])
            lo      = float(row["Low"])
            atr_val = float(row["atr"])
            is_last = (ts == day_bars.index[-1])
            in_window = bar_idx < ENTRY_WINDOW

            # ── Manage open position ──────────────────────────────────────────
            if position:
                ep = reason = None
                if lo  <= position["stop"]:    ep, reason = position["stop"],   "STOP"
                elif hi >= position["target"]: ep, reason = position["target"], "TARGET"
                if ep is None and is_last:     ep, reason = price,              "EOD"

                if ep is not None:
                    pnl = (ep - position["entry"]) * position["size"] * multiplier
                    cash += pnl
                    rec = {
                        "date":   trade_date,
                        "entry":  position["entry"],
                        "exit":   ep,
                        "pnl":    pnl,
                        "reason": reason,
                        "result": "WIN" if pnl > 0 else "LOSS",
                        "pts":    ep - position["entry"],
                        "size":   position["size"],
                    }
                    trades.append(rec)
                    day_trade = rec
                    position  = None

            # ── Entry: LONG only, first 3 bars, one trade per day ─────────────
            if position is None and day_trade is None and in_window and not is_last and atr_val > 0:
                if signal_long(row):
                    risk_per_unit = ATR_STOP * atr_val * multiplier
                    size = max(1, int((cash * RISK_PCT) / risk_per_unit))
                    if margin > 0:
                        size = min(size, max(1, int((cash * max_margin) / margin)))
                    stop   = price - ATR_STOP   * atr_val
                    target = price + ATR_TARGET * atr_val
                    position = {"entry": price, "stop": stop, "target": target, "size": size}

        eq_curve.append((trade_date, cash))

    return {"equity": eq_curve, "trades": trades}


# ── Bootstrap ────────────────────────────────────────────────────────────────

def bootstrap(fd: pd.DataFrame, initial_cap: float, multiplier: float = 1.0,
              margin: float = 0.0, max_margin: float = 1.0,
              n_samples: int = 1000, sample_days: int = 300) -> pd.Series:
    fd = fd.copy()
    fd.index = fd.index.tz_localize(None) if fd.index.tz else fd.index
    fd["trade_date"] = fd.index.normalize()
    all_days = sorted(fd["trade_date"].unique())

    # pre-compute per-day PnL with infinite capital (relative)
    day_pnl = {}
    for td in all_days:
        day_bars = fd[fd["trade_date"] == td].sort_index()
        if len(day_bars) < 2:
            day_pnl[td] = 0.0; continue
        position = None; pnl_day = 0.0
        for bar_idx, (ts, row) in enumerate(day_bars.iterrows()):
            price   = float(row["Close"])
            hi      = float(row["High"])
            lo      = float(row["Low"])
            atr_val = float(row["atr"])
            is_last = (ts == day_bars.index[-1])
            in_window = bar_idx < ENTRY_WINDOW
            if position:
                ep = reason = None
                if lo  <= position["stop"]:    ep, reason = position["stop"],   "STOP"
                elif hi >= position["target"]: ep, reason = position["target"], "TARGET"
                if ep is None and is_last:     ep, reason = price,              "EOD"
                if ep is not None:
                    pnl_day = (ep - position["entry"]) / position["entry"]  # % return
                    position = None
            if position is None and pnl_day == 0.0 and in_window and not is_last and atr_val > 0:
                if signal_long(row):
                    stop   = price - ATR_STOP   * atr_val
                    target = price + ATR_TARGET * atr_val
                    position = {"entry": price, "stop": stop, "target": target}
        day_pnl[td] = pnl_day

    day_arr = np.array([day_pnl[d] for d in all_days])
    rng     = np.random.default_rng(42)
    finals  = []

    for _ in range(n_samples):
        idx    = rng.integers(0, len(day_arr), size=sample_days)
        sample = day_arr[idx]
        cap    = initial_cap
        for r in sample:
            cap *= (1 + r * RISK_PCT / 0.02)  # scale by actual RISK_PCT
        finals.append((cap / initial_cap - 1) * 100)

    return pd.Series(finals)


# ── Report ────────────────────────────────────────────────────────────────────

def report(results: dict, initial_cap: float, label: str):
    dates, vals = zip(*results["equity"])
    eq     = pd.Series(list(vals), index=list(dates))
    trades = pd.DataFrame(results["trades"])

    final  = eq.iloc[-1]; ret = (final / initial_cap - 1) * 100
    dchg   = eq.pct_change().dropna()
    sharpe = dchg.mean() / dchg.std() * np.sqrt(252) if dchg.std() > 0 else 0
    dd     = (eq - eq.cummax()) / eq.cummax() * 100
    max_dd = dd.min()

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  Capital: NT${initial_cap:,.0f} → NT${final:,.0f}  ({ret:+.1f}%)")
    print(f"  Sharpe: {sharpe:.2f}   MaxDD: {max_dd:.1f}%")

    if not trades.empty:
        n = len(trades)
        nw = (trades["result"] == "WIN").sum()
        nl = n - nw
        aw = trades[trades["result"]=="WIN"]["pnl"].mean()  if nw > 0 else 0
        al = trades[trades["result"]=="LOSS"]["pnl"].mean() if nl > 0 else 0
        gw = trades[trades["result"]=="WIN"]["pnl"].sum()
        gl = trades[trades["result"]=="LOSS"]["pnl"].sum()
        pf = abs(gw / gl) if gl != 0 else float("inf")

        print(f"  Trades: {n}  ({nw}W/{nl}L)  WR={nw/n*100:.1f}%  "
              f"W/L={abs(aw/al) if al else 0:.2f}×  PF={pf:.2f}")
        print(f"  AvgWin: NT${aw:+,.0f}  AvgLoss: NT${al:+,.0f}")

        eb = trades.groupby("reason").agg(
            N  =("pnl","count"),
            WR =("result", lambda x: f"{(x=='WIN').mean()*100:.0f}%"),
            Pts=("pts",    lambda x: f"{x.mean():+.0f}"),
            PnL=("pnl",   lambda x: f"NT${x.sum():+,.0f}"),
        )
        print(eb.to_string())

    print(f"{'='*60}")
    return eq, dd, trades


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os; os.makedirs("results_sprint", exist_ok=True)

    # ── TWII (futures) ────────────────────────────────────────────────────────
    log.info("Downloading ^TWII 1h ...")
    twii_raw = yf.download("^TWII", period="max", interval="1h",
                            auto_adjust=True, progress=False)
    if isinstance(twii_raw.columns, pd.MultiIndex):
        twii_raw.columns = twii_raw.columns.get_level_values(0)
    twii_raw = twii_raw.dropna(subset=["Close"])
    twii_raw.index = twii_raw.index.tz_convert("Asia/Taipei")
    log.info("TWII: %d bars (%s → %s)", len(twii_raw),
             twii_raw.index[0].date(), twii_raw.index[-1].date())

    twii_fd = build_features(twii_raw)
    twii_r  = simulate(twii_fd, initial_cap=40_000,
                       multiplier=10.0, margin=25_000, max_margin=0.30)
    eq_twii, dd_twii, tr_twii = report(twii_r, 40_000,
        "LONG-ONLY  |  TWII (TMF futures)  |  Stop=1×ATR  Target=2×ATR")

    # Bootstrap TWII
    log.info("Bootstrap TWII (1000 × 300 days) ...")
    bs_twii = bootstrap(twii_fd, 40_000, multiplier=10.0,
                        margin=25_000, max_margin=0.30)
    pp_twii = (bs_twii > 0).mean() * 100
    print(f"\n  Bootstrap TWII: P(profit)={pp_twii:.1f}%  "
          f"Median={bs_twii.median():+.1f}%  "
          f"5th={bs_twii.quantile(0.05):+.1f}%  "
          f"95th={bs_twii.quantile(0.95):+.1f}%")

    # ── TSMC (stock) ──────────────────────────────────────────────────────────
    log.info("Downloading 2330.TW 1h ...")
    tsmc_raw = yf.download("2330.TW", period="max", interval="1h",
                            auto_adjust=True, progress=False)
    if isinstance(tsmc_raw.columns, pd.MultiIndex):
        tsmc_raw.columns = tsmc_raw.columns.get_level_values(0)
    tsmc_raw = tsmc_raw.dropna(subset=["Close"])
    tsmc_raw.index = tsmc_raw.index.tz_convert("Asia/Taipei")
    log.info("TSMC: %d bars (%s → %s)", len(tsmc_raw),
             tsmc_raw.index[0].date(), tsmc_raw.index[-1].date())

    tsmc_fd = build_features(tsmc_raw)
    tsmc_r  = simulate(tsmc_fd, initial_cap=200_000,
                       multiplier=1.0, margin=0.0)
    eq_tsmc, dd_tsmc, tr_tsmc = report(tsmc_r, 200_000,
        "LONG-ONLY  |  TSMC 2330.TW (stock)  |  Stop=1×ATR  Target=2×ATR")

    # Bootstrap TSMC
    log.info("Bootstrap TSMC (1000 × 300 days) ...")
    bs_tsmc = bootstrap(tsmc_fd, 200_000, multiplier=1.0)
    pp_tsmc = (bs_tsmc > 0).mean() * 100
    print(f"\n  Bootstrap TSMC: P(profit)={pp_tsmc:.1f}%  "
          f"Median={bs_tsmc.median():+.1f}%  "
          f"5th={bs_tsmc.quantile(0.05):+.1f}%  "
          f"95th={bs_tsmc.quantile(0.95):+.1f}%")

    # ── Chart ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 9))

    for col, (eq, dd, label, clr, cap) in enumerate([
        (eq_twii, dd_twii, "TWII Futures (LONG only)", "crimson",   40_000),
        (eq_tsmc, dd_tsmc, "TSMC Stock (LONG only)",   "darkorange", 200_000),
    ]):
        eq_idx = eq.index.tz_localize(None) if eq.index.tz else eq.index
        dd_idx = dd.index.tz_localize(None) if dd.index.tz else dd.index
        ret = (eq.iloc[-1] / cap - 1) * 100

        ax = axes[0][col]
        ax.plot(eq_idx, eq / 1e3, color=clr, lw=2,
                label=f"Strategy ({ret:+.1f}%)")
        ax.set_title(label); ax.legend(fontsize=9); ax.grid(alpha=0.3)
        ax.set_ylabel("Equity (NT$K)")

        ax2 = axes[1][col]
        ax2.fill_between(dd_idx, dd.values, 0, color=clr, alpha=0.35)
        ax2.set_ylabel("Drawdown (%)"); ax2.grid(alpha=0.3)
        ax2.set_title(f"Drawdown | MaxDD={dd.min():.1f}%")

    plt.suptitle("Thermo LONG-ONLY Intraday  |  MA15 Rising + Thermo Signal  |  "
                 "Entry 09:00–11:00  |  Exit: Target / Stop / EOD",
                 fontsize=11)
    plt.tight_layout()
    out = "results_sprint/thermo_longonly.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    log.info("Chart -> %s", out)
    print(f"\n  Chart -> {out}")

    # ── Bootstrap distribution chart ──────────────────────────────────────────
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for ax, bs, label, clr, pp in [
        (ax1, bs_twii, "TWII Futures", "crimson",   pp_twii),
        (ax2, bs_tsmc, "TSMC Stock",   "darkorange", pp_tsmc),
    ]:
        ax.hist(bs, bins=60, color=clr, alpha=0.7, edgecolor="white")
        ax.axvline(0, color="black", lw=1.5, ls="--")
        ax.axvline(bs.median(), color="navy", lw=1.5, label=f"Median {bs.median():+.1f}%")
        ax.set_xlabel("Return (%)"); ax.set_ylabel("Count")
        ax.set_title(f"{label} — LONG-ONLY Bootstrap\n"
                     f"P(profit)={pp:.1f}%  Median={bs.median():+.1f}%")
        ax.legend()

    plt.suptitle("Bootstrap: 1000× random 300-day samples", fontsize=11)
    plt.tight_layout()
    out2 = "results_sprint/thermo_longonly_bootstrap.png"
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    log.info("Chart -> %s", out2)
