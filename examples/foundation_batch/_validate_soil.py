#!/usr/bin/env python3
"""Validate two-stage monopile soil sweep; write summary to logs/soil_validate_summary.txt."""
from __future__ import annotations

import os
import sys
import traceback

BATCH_DIR = os.path.dirname(os.path.realpath(__file__))
LOG_DIR = os.path.join(BATCH_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
SUMMARY = os.path.join(LOG_DIR, "soil_validate_summary.txt")
LOG = os.path.join(LOG_DIR, "soil_validate.log")

sys.path.insert(0, BATCH_DIR)
from run_foundation_batch import run_batch  # noqa: E402


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def main() -> int:
    csv_in = os.path.join(BATCH_DIR, "foundation_sensitivity_inputs_soil.csv")
    with open(LOG, "w", encoding="utf-8") as lf:
        sys.stdout = Tee(sys.__stdout__, lf)
        sys.stderr = Tee(sys.__stderr__, lf)
        try:
            rc = run_batch(csv_in, foundation_mode="monopile", stop_on_error=False)
        except Exception:
            traceback.print_exc()
            rc = 2
        finally:
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__

    # Summarise last 4 monopile rows
    examples = os.path.dirname(BATCH_DIR)
    results = os.path.join(examples, "monopile_foundation_results.csv")
    lines = []
    lines.append(f"exit_code={rc}")
    try:
        import csv

        with open(results, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        last = rows[-4:] if len(rows) >= 4 else rows
        lines.append("last_rows:")
        for r in last:
            lines.append(
                f"  D={r.get('pile_diameter_m')} L={r.get('pile_length_m')} "
                f"t={r.get('wall_thickness_m')} mass={r.get('support_structure_mass_kg')} "
                f"exit={r.get('opt_exit_mode')} f1={r.get('f1_Hz')}"
            )
        masses = []
        for r in last:
            try:
                masses.append(float(r.get("support_structure_mass_kg") or "nan"))
            except ValueError:
                masses.append(float("nan"))
        if len(masses) >= 2:
            mono = all(
                (not (a == a and b == b)) or b <= a * 1.05
                for a, b in zip(masses, masses[1:])
            )
            lines.append(f"mass_nonincreasing_within_5pct={mono}")
            lines.append(f"masses={masses}")
    except Exception as exc:
        lines.append(f"summary_error={exc}")

    with open(SUMMARY, "w", encoding="utf-8") as sf:
        sf.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
