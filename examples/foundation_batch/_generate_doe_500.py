#!/usr/bin/env python3
"""
Build full-factorial foundation DOE scenario CSVs (~200 rows each).

Shared levels:
  turbine_rating_W: 5, 10, 15, 22 MW
  V50_mps:          12 values from 30 to 70 m/s
  Hs50_m:           4, 9, 14, 18

Jacket is the 4 x 12 x 4 grid (192 rows). Monopile uses the same site
grid with one of two soil-G levels per cell (10 or 150 MPa), so it also
has 192 rows. Water depth is not a DOE factor (modeling YAML default).
"""
from __future__ import annotations

import csv
import itertools
import os
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))

SHARED_LEVELS = {
    "turbine_rating_W": [5000000.0, 10000000.0, 15000000.0, 22000000.0],
    "V50_mps": [30.0, 34.0, 38.0, 41.0, 45.0, 49.0, 52.0, 56.0, 60.0, 63.0, 67.0, 70.0],
    "Hs50_m": [4.0, 9.0, 14.0, 18.0],
}

SOIL_G = [10000000.0, 150000000.0]

DESIGNS = {
    "monopile": {
        "out": os.path.join(HERE, "DOE_scenarios_500_monopile.csv"),
        "headers": [
            "turbine_rating_W",
            "V50_mps",
            "Hs50_m",
            "soil_shear_modulus_Pa",
        ],
        "levels": {**SHARED_LEVELS, "soil_shear_modulus_Pa": SOIL_G},
        "sort_key": lambda r: (r[0], r[1], r[2], r[3]),
        "mode": "site_plus_one_g",
    },
    "jacket": {
        "out": os.path.join(HERE, "DOE_scenarios_500_jacket.csv"),
        "headers": ["turbine_rating_W", "V50_mps", "Hs50_m"],
        "levels": dict(SHARED_LEVELS),
        "sort_key": lambda r: (r[0], r[1], r[2]),
        "mode": "full",
    },
}


def all_grid(headers, levels):
    grids = [levels[h] for h in headers]
    return [tuple(combo) for combo in itertools.product(*grids)]


def site_plus_one_g(headers, levels):
    """One soil-G level per (rating, V50, Hs) cell so n matches the jacket grid."""
    g_key = "soil_shear_modulus_Pa"
    site_headers = [h for h in headers if h != g_key]
    gs = levels[g_key]
    rows = []
    for combo in itertools.product(*[levels[h] for h in site_headers]):
        idx = sum(levels[h].index(v) for h, v in zip(site_headers, combo))
        rec = dict(zip(site_headers, combo))
        rec[g_key] = gs[idx % len(gs)]
        rows.append(tuple(rec[h] for h in headers))
    return rows


def fmt(h, v):
    if h in {"Hs50_m", "V50_mps"}:
        return f"{v:g}"
    return str(int(v)) if float(v).is_integer() else f"{v:g}"


def report(name, headers, levels, rows):
    lines = [f"{name}: n={len(rows)}"]
    for j, h in enumerate(headers):
        c = Counter(r[j] for r in rows)
        lines.append(f"{h}:")
        for v in levels[h]:
            lines.append(f"  {v:g}: {c[v]}")
    return "\n".join(lines)


def write_design(name, spec):
    headers = spec["headers"]
    levels = spec["levels"]
    if spec.get("mode") == "site_plus_one_g":
        rows = site_plus_one_g(headers, levels)
    else:
        rows = all_grid(headers, levels)
    rows.sort(key=spec["sort_key"])
    with open(spec["out"], "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for r in rows:
            w.writerow([fmt(h, v) for h, v in zip(headers, r)])
    text = report(name, headers, levels, rows)
    print("WROTE", spec["out"])
    print(text)
    return text


def main():
    chunks = []
    for name, spec in DESIGNS.items():
        chunks.append(write_design(name, spec))
    old = os.path.join(HERE, "DOE_scenarios_500.csv")
    if os.path.isfile(old):
        os.remove(old)
        print("REMOVED", old)
    report_path = os.path.join(HERE, "_doe_500_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(chunks) + "\n")
    print("WROTE", report_path)


if __name__ == "__main__":
    main()
