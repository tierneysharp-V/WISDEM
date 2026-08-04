"""
Append foundation structural metrics from a WISDEM run to a type-specific CSV.

Jacket runs append to jacket_foundation_results.csv; monopile runs append to
monopile_foundation_results.csv. Foundation type is detected from modeling flags.
One row per run; unavailable columns are written blank (NaN).
"""

from __future__ import annotations

import csv
import os
from datetime import datetime, timezone

import numpy as np

# Slim schema shared by both jacket and monopile files (append-only)
FOUNDATION_CSV_COLUMNS = [
    "timestamp_utc",
    "foundation_type",
    "concrete_volume_m3",
    "pile_diameter_m",
    "pile_length_m",
    "wall_thickness_m",
    "number_of_piles",
    "transition_piece_mass_kg",
    "scour_protection_volume_m3",
    "jacket_height_m",
    "jacket_mass_kg",
    "support_structure_mass_kg",
    "total_foundation_mass_kg",
    "opt_exit_mode",
    "f1_Hz",
]

_EXAMPLES_DIR = os.path.dirname(os.path.realpath(__file__))
JACKET_CSV_PATH = os.path.join(_EXAMPLES_DIR, "jacket_foundation_results.csv")
MONOPILE_CSV_PATH = os.path.join(_EXAMPLES_DIR, "monopile_foundation_results.csv")


def _scalar(value, default=np.nan, reduce="first"):
    """Reduce OpenMDAO arrays / scalars to a float (or NaN)."""
    try:
        if value is None:
            return default
        arr = np.atleast_1d(np.asarray(value, dtype=float).flatten())
        if arr.size == 0:
            return default
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return default
        if reduce == "max":
            return float(np.nanmax(finite))
        return float(finite[0])
    except Exception:
        return default


def _get_val(wt_opt, name, units=None, default=np.nan, reduce="first"):
    """Safely read a promoted OpenMDAO variable."""
    try:
        if units is None:
            val = wt_opt.get_val(name)
        else:
            val = wt_opt.get_val(name, units=units)
        return _scalar(val, default=default, reduce=reduce)
    except Exception:
        return default


def _detect_foundation_type(modeling_options):
    flags = modeling_options.get("flags", {})
    if flags.get("jacket"):
        return "jacket"
    if flags.get("monopile"):
        return "monopile"
    return "unknown"


def csv_path_for_foundation_type(ftype):
    """Return the append-only CSV path for a detected foundation type."""
    if ftype == "jacket":
        return JACKET_CSV_PATH
    if ftype == "monopile":
        return MONOPILE_CSV_PATH
    raise ValueError(f"Unknown foundation type '{ftype}'; expected jacket or monopile")


def collect_foundation_row(wt_opt, modeling_options, extras=None):
    """Build one results dict matching FOUNDATION_CSV_COLUMNS."""
    ftype = _detect_foundation_type(modeling_options)
    row = {col: np.nan for col in FOUNDATION_CSV_COLUMNS}
    row["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    row["foundation_type"] = ftype

    # Offshore WISDEM does not model foundation concrete volume
    row["concrete_volume_m3"] = np.nan

    bos_on = modeling_options.get("flags", {}).get("bos", False)

    if ftype == "jacket":
        jacket_mass = _get_val(wt_opt, "fixedse.jacket_mass", units="kg")
        tp_mass = _get_val(wt_opt, "jacket.transition_piece_mass", units="kg")
        row["jacket_mass_kg"] = jacket_mass
        row["jacket_height_m"] = _get_val(wt_opt, "jacket.height", units="m")
        row["transition_piece_mass_kg"] = tp_mass
        row["support_structure_mass_kg"] = jacket_mass
        if np.isfinite(jacket_mass) and np.isfinite(tp_mass):
            row["total_foundation_mass_kg"] = jacket_mass + tp_mass
        n_legs = modeling_options.get("WISDEM", {}).get("FixedBottomSE", {}).get("n_legs", np.nan)
        try:
            row["number_of_piles"] = int(n_legs)
        except (TypeError, ValueError):
            row["number_of_piles"] = np.nan
        # Pile geometry / wall thickness not modeled for jackets
        row["pile_diameter_m"] = np.nan
        row["pile_length_m"] = np.nan
        row["wall_thickness_m"] = np.nan
        row["f1_Hz"] = _get_val(wt_opt, "fixedse.structural_frequencies", reduce="first")

    elif ftype == "monopile":
        mono_mass = _get_val(wt_opt, "fixedse.monopile_mass", units="kg")
        tp_mass = _get_val(wt_opt, "monopile.transition_piece_mass", units="kg")
        grav_mass = _get_val(wt_opt, "monopile.gravity_foundation_mass", units="kg", default=0.0)
        if not np.isfinite(grav_mass):
            grav_mass = 0.0
        row["transition_piece_mass_kg"] = tp_mass
        # monopile_mass already includes TP (+ gravity foundation); cylinder-only for support column
        if np.isfinite(mono_mass) and np.isfinite(tp_mass):
            row["support_structure_mass_kg"] = mono_mass - tp_mass - grav_mass
            row["total_foundation_mass_kg"] = mono_mass
        row["pile_diameter_m"] = _get_val(wt_opt, "monopile.diameter", units="m", reduce="max")
        row["pile_length_m"] = _get_val(wt_opt, "monopile.height", units="m")
        wall = _get_val(wt_opt, "fixedse.monopile_wall_thickness", units="m", reduce="max")
        if not np.isfinite(wall):
            wall = _get_val(wt_opt, "monopile.layer_thickness", units="m", reduce="max")
        row["wall_thickness_m"] = wall
        row["number_of_piles"] = 1
        row["jacket_height_m"] = np.nan
        row["jacket_mass_kg"] = np.nan
        row["f1_Hz"] = _get_val(wt_opt, "fixedse.structural_frequencies", reduce="first")

    if bos_on:
        row["scour_protection_volume_m3"] = _get_val(
            wt_opt, "orbit.scour_protection_volume", units="m**3"
        )

    if extras:
        for key, val in extras.items():
            if key in FOUNDATION_CSV_COLUMNS:
                row[key] = val

    return row


def append_foundation_results(wt_opt, modeling_options, csv_path=None, extras=None):
    """
    Append one foundation-results row.

    If csv_path is None, routes automatically to jacket_foundation_results.csv or
    monopile_foundation_results.csv based on modeling flags.

    extras : optional dict of extra column values (e.g. opt_exit_mode).

    Returns
    -------
    str
        Path to the CSV that was written.
    """
    ftype = _detect_foundation_type(modeling_options)
    if csv_path is None:
        csv_path = csv_path_for_foundation_type(ftype)

    row = collect_foundation_row(wt_opt, modeling_options, extras=extras)

    os.makedirs(os.path.dirname(os.path.abspath(csv_path)) or ".", exist_ok=True)

    # Migrate older CSVs that predate opt_exit_mode / f1_Hz columns
    if os.path.isfile(csv_path) and os.path.getsize(csv_path) > 0:
        with open(csv_path, newline="", encoding="utf-8") as rf:
            reader = csv.DictReader(rf)
            header = list(reader.fieldnames or [])
            existing_rows = list(reader)
        if header and header != FOUNDATION_CSV_COLUMNS:
            with open(csv_path, "w", newline="", encoding="utf-8") as wf:
                writer = csv.DictWriter(wf, fieldnames=FOUNDATION_CSV_COLUMNS, extrasaction="ignore")
                writer.writeheader()
                for erow in existing_rows:
                    writer.writerow({c: erow.get(c, "") for c in FOUNDATION_CSV_COLUMNS})
            write_header = False
        else:
            write_header = False
    else:
        write_header = True

    out = {}
    for col in FOUNDATION_CSV_COLUMNS:
        val = row.get(col, np.nan)
        if isinstance(val, (float, np.floating)) and not np.isfinite(val):
            out[col] = ""
        elif isinstance(val, (np.integer,)):
            out[col] = int(val)
        elif isinstance(val, (float, np.floating)):
            out[col] = float(val)
        else:
            out[col] = val

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FOUNDATION_CSV_COLUMNS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(out)

    print(f"Appended foundation results to {csv_path}")
    return csv_path


if __name__ == "__main__":
    print("Import append_foundation_results(wt_opt, modeling_options) after run_wisdem.")
    print(f"Jacket CSV:    {JACKET_CSV_PATH}")
    print(f"Monopile CSV:  {MONOPILE_CSV_PATH}")
