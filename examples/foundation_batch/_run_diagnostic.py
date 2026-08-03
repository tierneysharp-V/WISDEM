#!/usr/bin/env python3
"""Run diagnostic jacket then monopile batch; log to examples/foundation_batch/logs/."""
from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, timezone

BATCH_DIR = os.path.dirname(os.path.realpath(__file__))
LOG_DIR = os.path.join(BATCH_DIR, "logs")
SCENARIOS = os.path.join(BATCH_DIR, "diagnostic_scenarios.csv")
RUNNER = os.path.join(BATCH_DIR, "run_foundation_batch.py")


def run_one(foundation: str) -> int:
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f"diagnostic_{foundation}.log")
    ec_path = os.path.join(LOG_DIR, f"diagnostic_{foundation}.exitcode")
    stamp = datetime.now(timezone.utc).isoformat()
    cmd = [sys.executable, RUNNER, SCENARIOS, "--foundation", foundation]
    with open(log_path, "w", encoding="utf-8") as log:
        log.write(f"=== START {foundation} {stamp} ===\n")
        log.write(f"cmd: {' '.join(cmd)}\n")
        log.flush()
        print(f"=== START {foundation} {stamp} ===", flush=True)
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=BATCH_DIR)
        end = datetime.now(timezone.utc).isoformat()
        log.write(f"\n=== {foundation} EXIT {proc.returncode} {end} ===\n")
    with open(ec_path, "w", encoding="utf-8") as f:
        f.write(str(proc.returncode))
    print(f"=== {foundation} EXIT {proc.returncode} {end} ===", flush=True)
    return proc.returncode


def main() -> int:
    print(f"python={sys.executable}", flush=True)
    print(f"scenarios={SCENARIOS}", flush=True)
    with open(SCENARIOS, encoding="utf-8") as f:
        print(f.read(), flush=True)
    j = run_one("jacket")
    m = run_one("monopile")
    print(f"DONE jacket={j} monopile={m}", flush=True)
    return 0 if j == 0 and m == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
