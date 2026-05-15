#!/usr/bin/env python3
"""
run_once.py
-----------
Single-execution version for GitHub Actions.
Runs one hourly check then exits — GitHub handles the scheduling.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from paper_trader import hourly_job, log, compute_stats

log.info("=== GitHub Actions — single hourly run ===")
hourly_job()

# Print current stats to Actions log
print(compute_stats())
log.info("=== Done ===")
