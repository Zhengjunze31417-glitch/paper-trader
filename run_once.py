#!/usr/bin/env python3
"""
run_once.py
-----------
Single-execution version for GitHub Actions.
Runs one hourly check then exits — GitHub handles the scheduling.
"""
import sys
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from paper_trader import hourly_job, notify, load_state, log, compute_stats, ASSETS

log.info("=== GitHub Actions — single hourly run ===")
hourly_job()

# Send Telegram every day at 09:00 UTC as a heartbeat
hour = datetime.now(timezone.utc).hour
if hour == 9:
    state  = load_state()
    open_n = len(state.get("positions", {}))
    assets = ", ".join(ASSETS.keys())
    notify(
        f"Daily Status — Paper Trader running\n"
        f"Watching: {assets}\n"
        f"Open positions: {open_n}\n"
        f"System healthy — checking every hour"
    )

# Also send a one-time confirmation right now (first run only)
state = load_state()
if state.get("trade_count", 0) == 0 and state.get("signal_count", 0) == 0:
    notify(
        f"Paper Trader is LIVE on GitHub\n"
        f"Watching 7 markets every hour:\n"
        f"ETH, BTC, QQQ, SPY, GLD, AAPL, USO\n"
        f"You will be notified when a signal fires."
    )

log.info("=== Done ===")
