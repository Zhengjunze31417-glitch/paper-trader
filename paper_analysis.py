#!/usr/bin/env python3
"""
paper_analysis.py
-----------------
Run this any time to get a full statistical report on paper trading results.
Usage: python paper_analysis.py
"""
import pandas as pd, numpy as np, sys, math
from scipy import stats
from pathlib import Path

TRADES_FILE  = Path("paper_trades.csv")
SIGNALS_FILE = Path("paper_signals.csv")
STOP_MULT=1.0; TARGET_MULT=3.0
BE_WR = STOP_MULT/(STOP_MULT+TARGET_MULT)*100  # 25.0%

if not TRADES_FILE.exists():
    print("No paper_trades.csv found. Run paper_trader.py first.")
    sys.exit(1)

df   = pd.read_csv(TRADES_FILE)
closed = df[df["exit_reason"].notna() & (df["exit_reason"] != "")].copy()
closed["pnl_pct"] = pd.to_numeric(closed["pnl_pct"], errors="coerce")
closed["win"]     = pd.to_numeric(closed["win"],     errors="coerce")
closed = closed.dropna(subset=["pnl_pct"])
n = len(closed)

print(f"\n{'='*65}")
print(f"  PAPER TRADING ANALYSIS  |  CompressedSpring Theory Test")
print(f"{'='*65}")

if n == 0:
    print("  No completed trades yet.")
    sys.exit(0)

# Date range
if "entry_time" in closed.columns:
    closed["entry_time"] = pd.to_datetime(closed["entry_time"])
    days = (closed["entry_time"].max() - closed["entry_time"].min()).days
    print(f"  Period  : {closed['entry_time'].min().date()} → {closed['entry_time'].max().date()} ({days} days)")
print(f"  Trades  : {n} completed")

# ── 1. Overall statistics ─────────────────────────────────────────────────────
print(f"\n  ── 1. OVERALL STATISTICS ──────────────────────────────")
wins  = int(closed["win"].sum())
wr    = wins/n*100
tr_arr= closed["pnl_pct"].values/100
mean_r= tr_arr.mean()*100
avg_w = closed[closed["win"]==1]["pnl_pct"].mean() if wins>0 else 0
avg_l = closed[closed["win"]==0]["pnl_pct"].mean() if (n-wins)>0 else 0
total_ret = (np.prod(1+tr_arr)-1)*100

print(f"  Win Rate      : {wr:.1f}%  (break-even = {BE_WR:.1f}%)")
print(f"  {'+' if wr>BE_WR else '-'}  {'ABOVE' if wr>BE_WR else 'BELOW'} break-even by {abs(wr-BE_WR):.1f}%")
print(f"  Avg Win       : {avg_w:+.2f}%")
print(f"  Avg Loss      : {avg_l:+.2f}%")
print(f"  W/L Ratio     : {abs(avg_w/avg_l):.2f}x" if avg_l!=0 else "")
print(f"  Mean return   : {mean_r:+.3f}% per trade")
print(f"  Total return  : {total_ret:+.1f}% (compounded)")

# ── 2. Per-asset breakdown ────────────────────────────────────────────────────
print(f"\n  ── 2. PER-ASSET RESULTS ───────────────────────────────")
print(f"  {'Asset':<10} {'N':>4} {'WR':>7} {'AvgWin':>8} {'AvgLoss':>9} {'Total':>8}  Verdict")
print(f"  {'-'*60}")
for asset in sorted(closed["asset"].unique()):
    sub  = closed[closed["asset"]==asset]
    awr  = sub["win"].mean()*100
    aw   = sub[sub["win"]==1]["pnl_pct"].mean() if (sub["win"]==1).any() else 0
    al   = sub[sub["win"]==0]["pnl_pct"].mean() if (sub["win"]==0).any() else 0
    aret = (np.prod(1+sub["pnl_pct"].values/100)-1)*100
    v    = "✓ WORKS" if awr>BE_WR and aret>0 else ("~ CLOSE" if aret>0 else "✗ FAILS")
    print(f"  {asset:<10} {len(sub):>4} {awr:>6.1f}% {aw:>+7.2f}% {al:>+8.2f}% {aret:>+7.1f}%  {v}")

# ── 3. Exit breakdown ─────────────────────────────────────────────────────────
print(f"\n  ── 3. EXIT REASON BREAKDOWN ───────────────────────────")
print(f"  {'Reason':<10} {'N':>4} {'WR':>7} {'AvgPnL':>9}  Insight")
print(f"  {'-'*52}")
for rsn in ["STOP","TARGET","TIMEOUT"]:
    sub = closed[closed["exit_reason"]==rsn]
    if len(sub)==0: continue
    swr = sub["win"].mean()*100
    spnl= sub["pnl_pct"].mean()
    insight = {"STOP":"hard stop hit","TARGET":"full 3xATR hit","TIMEOUT":"time limit"}.get(rsn,"")
    print(f"  {rsn:<10} {len(sub):>4} {swr:>6.1f}% {spnl:>+8.2f}%  {insight}")

# ── 4. Statistical significance ───────────────────────────────────────────────
print(f"\n  ── 4. STATISTICAL SIGNIFICANCE ────────────────────────")
if n >= 5:
    t_stat, p_ttest = stats.ttest_1samp(tr_arr, 0)
    binom = stats.binomtest(wins, n, BE_WR/100, alternative="greater")
    boot  = np.array([np.random.choice(tr_arr,n,replace=True).mean() for _ in range(10000)])
    ci_lo, ci_hi = np.percentile(boot*100,[2.5,97.5])
    print(f"  t-test (mean≠0)  : p={p_ttest:.4f}  {'✓ significant' if p_ttest<0.05 else '✗ not yet'}")
    print(f"  Binomial (WR>25%): p={binom.pvalue:.4f}  {'✓ significant' if binom.pvalue<0.05 else '✗ not yet'}")
    print(f"  95% CI mean return : [{ci_lo:+.3f}%, {ci_hi:+.3f}%]  "
          f"{'✓ all positive' if ci_lo>0 else '✗ crosses zero'}")
    needed = max(30, int(((1.645*math.sqrt((BE_WR/100)*(1-BE_WR/100)) +
                           0.842*math.sqrt((wr/100)*(1-wr/100)))/
                           max(abs(wr/100-BE_WR/100),0.001))**2))
    print(f"  Trades needed for 80% power: ~{needed}  (have {n}  →  "
          f"{'SUFFICIENT' if n>=needed else f'need {needed-n} more'})")
else:
    print(f"  Need at least 5 trades for statistics (have {n})")

# ── 5. Theory validation ──────────────────────────────────────────────────────
print(f"\n  ── 5. THEORY VALIDATION CHECKLIST ────────────────────")
checks = [
    ("WR above break-even 25%",        wr > BE_WR),
    ("Mean return per trade positive",  mean_r > 0),
    ("Winners larger than losers",      abs(avg_w) > abs(avg_l)),
    (f"At least 30 trades",            n >= 30),
    ("t-test significant (p<0.05)",     n>=5 and p_ttest<0.05),
]
passed = sum(c[1] for c in checks)
for label, ok in checks:
    print(f"  {'✓' if ok else '✗'}  {label}")
print(f"\n  Theory score: {passed}/{len(checks)}")
verdict = ("STRONG — physics confirmed, consider real trading"  if passed>=4
      else "MODERATE — promising, continue paper trading"       if passed>=3
      else "WEAK — edge not confirmed, do not use real money"  if passed>=2
      else "FAIL — strategy does not work, do not trade")
print(f"  Verdict: {verdict}")
print(f"{'='*65}")
