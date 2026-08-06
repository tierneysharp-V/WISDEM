#!/usr/bin/env python3
"""Launch the 22 MW monopile batch and tee output to logs/."""
from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, timezone

BATCH_DIR = os.path.dirname(os.path.realpath(__file__))
LOG_DIR = os.path.join(BATCH_DIR, "logs")
SCENARIOS = os.path.join(BATCH_DIR, "scenarios_22mw.csv")
RUNNER = os.path.join(BATCH_DIR, "run_foundation_batch.py")
TURBINE_CSV = os.path.join(BATCH_DIR, "turbine_lookup.csv")


def main() -> int:
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, "run_22mw_monopile_wd30.log")
    ec_path = os.path.join(LOG_DIR, "run_22mw_monopile_wd30.exitcode")
    stamp = datetime.now(timezone.utc).isoformat()
    cmd = [
        sys.executable,
        RUNNER,
        SCENARIOS,
        "--turbine-csv",
        TURBINE_CSV,
        "--foundation",
        "monopile",
    ]
    with open(log_path, "w", encoding="utf-8") as log:
        log.write(f"=== START monopile 22MW WD30 {stamp} ===\n")
        log.write("cmd: " + " ".join(cmd) + "\n")
        log.flush()
        print(f"=== START monopile 22MW WD30 {stamp} ===", flush=True)
        print("cmd:", " ".join(cmd), flush=True)
        with open(SCENARIOS, encoding="utf-8") as sc:
            print(sc.read(), flush=True)
            log.write(open(SCENARIOS, encoding="utf-8").read())
        proc = subprocess.Popen(
            cmd,
            cwd=BATCH_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()
        ec = proc.wait()
        end = datetime.now(timezone.utc).isoformat()
        footer = f"=== EXIT {ec} {end} ===\n"
        print(footer, end="", flush=True)
        log.write(footer)
    with open(ec_path, "w", encoding="utf-8") as f:
        f.write(str(ec) + "\n")
    return ec


if __name__ == "__main__":
    raise SystemExit(main())
