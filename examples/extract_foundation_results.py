"""
Append foundation structural metrics from a WISDEM run to type-specific CSVs.

Monopile runs append to results_monopile.csv. Jacket runs append to
results_jacket.csv (four legs) or results_jacket_3.csv (three legs).
Full LCOE runs append to results_lcoe.csv.
Foundation type and jacket leg count are detected from modeling flags/options.
One row per run; unavailable columns are written blank (NaN).
"""

from __future__ import annotations

import csv
import os
from datetime import datetime, timezone

import numpy as np

# Old result-CSV names that map onto the current schema when rewriting a file.
COLUMN_ALIASES = {
    "jacket_footprint_m": "jacket_foot_radius_m",
    "pile_length_m": "monopile_height_m",
    "pile_diameter_m": "monopile_diameter_m",
    "pile_thickness_m": "monopile_thickness_m",
}

MONOPILE_CSV_COLUMNS = [
    "timestamp",
    "turbine_rating_W",
    "tower_base_shear_N",
    "tower_base_vertical_N",
    "tower_base_overturning_moment_Nm",
    "rna_mass_kg",
    "V50_mps",
    "Hs50_m",
    "water_depth_m",
    "soil_shear_modulus_Pa",
    "monopile_height_m",
    "monopile_diameter_m",
    "monopile_thickness_m",
    "monopile_mass_kg",
    "transition_piece_mass_kg",
    "total_foundation_mass_kg",
    "opt_exit_mode",
    "f1_Hz",
]

JACKET_CSV_COLUMNS = [
    "timestamp",
    "turbine_rating_W",
    "tower_base_shear_N",
    "tower_base_vertical_N",
    "tower_base_overturning_moment_Nm",
    "rna_mass_kg",
    "V50_mps",
    "Hs50_m",
    "water_depth_m",
    "jacket_height_m",
    "jacket_foot_radius_m",
    "leg_diameter_m",
    "leg_thickness_m",
    "brace_diameter_m",
    "brace_thickness_m",
    "jacket_mass_kg",
    "transition_piece_mass_kg",
    "total_foundation_mass_kg",
    "opt_exit_mode",
]

LCOE_CSV_COLUMNS = [
    "timestamp",
    "turbine_rating_W",
    "foundation_type",
    "n_legs",
    "V50_mps",
    "Hs50_m",
    "water_depth_m",
    "soil_shear_modulus_Pa",
    "monopile_mass_kg",
    "jacket_mass_kg",
    "transition_piece_mass_kg",
    "total_foundation_mass_kg",
    "foundation_cost_usd",
    "turbine_aep_kWh",
    "tcc_usd_per_kW",
    "bos_usd_per_kW",
    "opex_usd_per_kW_yr",
    "lcoe_usd_per_kwh",
    "lcoe_usd_per_mwh",
    "bos_source",
]

_EXAMPLES_DIR = os.path.dirname(os.path.realpath(__file__))
MONOPILE_CSV_PATH = os.path.join(_EXAMPLES_DIR, "results_monopile.csv")
JACKET_CSV_PATH = os.path.join(_EXAMPLES_DIR, "results_jacket.csv")
JACKET_3_CSV_PATH = os.path.join(_EXAMPLES_DIR, "results_jacket_3.csv")
LCOE_CSV_PATH = os.path.join(_EXAMPLES_DIR, "results_lcoe.csv")


def _jacket_n_legs(modeling_options):
    """Read jacket leg count from FixedBottomSE modeling options (default 4)."""
    try:
        n = int(modeling_options["WISDEM"]["FixedBottomSE"]["n_legs"])
    except (KeyError, TypeError, ValueError):
        return 4
    return 3 if n == 3 else 4


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


def columns_for_foundation_type(ftype):
    """Return the CSV column list for a foundation type."""
    if ftype == "jacket":
        return JACKET_CSV_COLUMNS
    if ftype == "monopile":
        return MONOPILE_CSV_COLUMNS
    raise ValueError(f"Unknown foundation type '{ftype}'; expected jacket or monopile")


def results_tag_for_foundation_type(ftype, modeling_options=None):
    """Suffix used in results CSV names: jacket, jacket_3, or monopile."""
    if ftype == "jacket":
        if modeling_options is not None and _jacket_n_legs(modeling_options) == 3:
            return "jacket_3"
        return "jacket"
    if ftype == "monopile":
        return "monopile"
    raise ValueError(f"Unknown foundation type '{ftype}'; expected jacket or monopile")


def csv_path_for_foundation_type(ftype, modeling_options=None, results_prefix=None):
    """Return the append-only CSV path for a detected foundation type.

    results_prefix : optional stem such as ``.../DOE_results``. When set, the
    file is ``{prefix}_{jacket|jacket_3|monopile}.csv`` instead of the default
    ``examples/results_*.csv`` paths.
    """
    tag = results_tag_for_foundation_type(ftype, modeling_options)
    if results_prefix:
        return f"{results_prefix}_{tag}.csv"
    if tag == "jacket_3":
        return JACKET_3_CSV_PATH
    if tag == "jacket":
        return JACKET_CSV_PATH
    return MONOPILE_CSV_PATH


def _remap_existing_row(erow, columns):
    """Map an on-disk row onto the current schema (aliases + current names)."""
    out = {}
    for col in columns:
        if col in erow and erow[col] not in (None, ""):
            out[col] = erow[col]
            continue
        for old, new in COLUMN_ALIASES.items():
            if new == col and old in erow and erow[old] not in (None, ""):
                out[col] = erow[old]
                break
        else:
            out[col] = erow.get(col, "")
    return out


def collect_foundation_row(wt_opt, modeling_options, extras=None):
    """Build one results dict matching the type-specific CSV columns."""
    ftype = _detect_foundation_type(modeling_options)
    columns = columns_for_foundation_type(ftype)
    row = {col: np.nan for col in columns}
    row["timestamp"] = datetime.now(timezone.utc).isoformat()

    if ftype == "jacket":
        jacket_mass = _get_val(wt_opt, "fixedse.jacket_mass", units="kg")
        tp_mass = _get_val(wt_opt, "jacket.transition_piece_mass", units="kg")
        row["jacket_mass_kg"] = jacket_mass
        row["jacket_height_m"] = _get_val(wt_opt, "jacket.height", units="m")
        row["jacket_foot_radius_m"] = _get_val(wt_opt, "fixedse.r_foot", units="m")
        row["leg_diameter_m"] = _get_val(wt_opt, "jacket.leg_diameter", units="m")
        row["leg_thickness_m"] = _get_val(wt_opt, "jacket.leg_thickness", units="m")
        row["brace_diameter_m"] = _get_val(wt_opt, "jacket.brace_diameters", units="m")
        row["brace_thickness_m"] = _get_val(wt_opt, "jacket.brace_thicknesses", units="m")
        row["transition_piece_mass_kg"] = tp_mass
        if np.isfinite(jacket_mass) and np.isfinite(tp_mass):
            row["total_foundation_mass_kg"] = jacket_mass + tp_mass
        elif np.isfinite(jacket_mass):
            row["total_foundation_mass_kg"] = jacket_mass

    elif ftype == "monopile":
        mono_mass = _get_val(wt_opt, "fixedse.monopile_mass", units="kg")
        tp_mass = _get_val(wt_opt, "monopile.transition_piece_mass", units="kg")
        grav_mass = _get_val(wt_opt, "monopile.gravity_foundation_mass", units="kg", default=0.0)
        if not np.isfinite(grav_mass):
            grav_mass = 0.0
        row["transition_piece_mass_kg"] = tp_mass
        # monopile_mass from FixedBottomSE includes TP (+ gravity foundation)
        if np.isfinite(mono_mass) and np.isfinite(tp_mass):
            row["monopile_mass_kg"] = mono_mass - tp_mass - grav_mass
            row["total_foundation_mass_kg"] = mono_mass
        elif np.isfinite(mono_mass):
            row["monopile_mass_kg"] = mono_mass
            row["total_foundation_mass_kg"] = mono_mass
        row["monopile_diameter_m"] = _get_val(wt_opt, "monopile.diameter", units="m", reduce="max")
        row["monopile_height_m"] = _get_val(wt_opt, "monopile.height", units="m")
        wall = _get_val(wt_opt, "fixedse.monopile_wall_thickness", units="m", reduce="max")
        if not np.isfinite(wall):
            wall = _get_val(wt_opt, "monopile.layer_thickness", units="m", reduce="max")
        row["monopile_thickness_m"] = wall

    if "f1_Hz" in columns:
        row["f1_Hz"] = _get_val(wt_opt, "fixedse.structural_frequencies")

    if extras:
        for key, val in extras.items():
            if key in columns:
                row[key] = val

    env = (modeling_options or {}).get("Environment") or {}
    if "water_depth_m" in columns and not np.isfinite(_scalar(row.get("water_depth_m"))):
        wd = env.get("water_depth")
        if wd is not None:
            row["water_depth_m"] = float(wd)
    if "soil_shear_modulus_Pa" in columns and not np.isfinite(
        _scalar(row.get("soil_shear_modulus_Pa"))
    ):
        g = env.get("soil_shear_modulus")
        if g is not None:
            row["soil_shear_modulus_Pa"] = float(g)

    return row


def append_foundation_results(
    wt_opt, modeling_options, csv_path=None, extras=None, results_prefix=None
):
    """
    Append one foundation-results row.

    If csv_path is None, routes automatically to results_jacket.csv,
    results_jacket_3.csv (three-legged jackets), or results_monopile.csv
    based on modeling flags and n_legs. Pass results_prefix to write
    ``{prefix}_{jacket|jacket_3|monopile}.csv`` instead.

    extras : optional dict of extra column values (e.g. loads, site params, RNA mass).

    Returns
    -------
    str
        Path to the CSV that was written.
    """
    ftype = _detect_foundation_type(modeling_options)
    columns = columns_for_foundation_type(ftype)
    if csv_path is None:
        csv_path = csv_path_for_foundation_type(
            ftype, modeling_options, results_prefix=results_prefix
        )

    row = collect_foundation_row(wt_opt, modeling_options, extras=extras)

    os.makedirs(os.path.dirname(os.path.abspath(csv_path)) or ".", exist_ok=True)

    # If an existing file has a different header, rewrite with the current schema
    if os.path.isfile(csv_path) and os.path.getsize(csv_path) > 0:
        with open(csv_path, newline="", encoding="utf-8") as rf:
            reader = csv.DictReader(rf)
            header = list(reader.fieldnames or [])
            existing_rows = list(reader)
        if header and header != columns:
            with open(csv_path, "w", newline="", encoding="utf-8") as wf:
                writer = csv.DictWriter(wf, fieldnames=columns, extrasaction="ignore")
                writer.writeheader()
                for erow in existing_rows:
                    writer.writerow(_remap_existing_row(erow, columns))
            write_header = False
        else:
            write_header = False
    else:
        write_header = True

    out = {}
    for col in columns:
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
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(out)

    print(f"Appended foundation results to {csv_path}")
    return csv_path


def collect_lcoe_row(wt_opt, modeling_options, extras=None):
    """Build one LCOE-results dict matching LCOE_CSV_COLUMNS."""
    ftype = _detect_foundation_type(modeling_options)
    row = {col: np.nan for col in LCOE_CSV_COLUMNS}
    row["timestamp"] = datetime.now(timezone.utc).isoformat()
    row["foundation_type"] = ftype if ftype in ("jacket", "monopile") else ""
    if ftype == "jacket":
        row["n_legs"] = _jacket_n_legs(modeling_options)
        jacket_mass = _get_val(wt_opt, "fixedse.jacket_mass", units="kg")
        tp_mass = _get_val(wt_opt, "jacket.transition_piece_mass", units="kg")
        row["jacket_mass_kg"] = jacket_mass
        row["transition_piece_mass_kg"] = tp_mass
        row["foundation_cost_usd"] = _get_val(wt_opt, "fixedse.jacket_cost", units="USD")
        if np.isfinite(jacket_mass) and np.isfinite(tp_mass):
            row["total_foundation_mass_kg"] = jacket_mass + tp_mass
        elif np.isfinite(jacket_mass):
            row["total_foundation_mass_kg"] = jacket_mass
    elif ftype == "monopile":
        mono_mass = _get_val(wt_opt, "fixedse.monopile_mass", units="kg")
        tp_mass = _get_val(wt_opt, "monopile.transition_piece_mass", units="kg")
        grav_mass = _get_val(wt_opt, "monopile.gravity_foundation_mass", units="kg", default=0.0)
        if not np.isfinite(grav_mass):
            grav_mass = 0.0
        row["transition_piece_mass_kg"] = tp_mass
        row["foundation_cost_usd"] = _get_val(wt_opt, "fixedse.monopile_cost", units="USD")
        if np.isfinite(mono_mass) and np.isfinite(tp_mass):
            row["monopile_mass_kg"] = mono_mass - tp_mass - grav_mass
            row["total_foundation_mass_kg"] = mono_mass
        elif np.isfinite(mono_mass):
            row["monopile_mass_kg"] = mono_mass
            row["total_foundation_mass_kg"] = mono_mass

    row["turbine_aep_kWh"] = _get_val(wt_opt, "financese.turbine_aep", units="kW*h")
    row["tcc_usd_per_kW"] = _get_val(wt_opt, "financese.tcc_per_kW", units="USD/kW")
    row["bos_usd_per_kW"] = _get_val(wt_opt, "financese.bos_per_kW", units="USD/kW")
    row["opex_usd_per_kW_yr"] = _get_val(wt_opt, "financese.opex_per_kW", units="USD/kW/yr")
    lcoe = _get_val(wt_opt, "financese.lcoe", units="USD/kW/h")
    row["lcoe_usd_per_kwh"] = lcoe
    if np.isfinite(lcoe):
        row["lcoe_usd_per_mwh"] = lcoe * 1000.0

    if extras:
        for key, val in extras.items():
            if key in LCOE_CSV_COLUMNS:
                row[key] = val
    return row


def append_lcoe_results(wt_opt, modeling_options, csv_path=None, extras=None, results_prefix=None):
    """
    Append one full-WISDEM LCOE row to results_lcoe.csv (or ``{prefix}_lcoe.csv``).

    extras : optional dict (site params, bos_source, rating, n_legs).

    Returns
    -------
    str
        Path to the CSV that was written.
    """
    columns = LCOE_CSV_COLUMNS
    if csv_path is None:
        csv_path = f"{results_prefix}_lcoe.csv" if results_prefix else LCOE_CSV_PATH

    row = collect_lcoe_row(wt_opt, modeling_options, extras=extras)
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)) or ".", exist_ok=True)

    if os.path.isfile(csv_path) and os.path.getsize(csv_path) > 0:
        with open(csv_path, newline="", encoding="utf-8") as rf:
            reader = csv.DictReader(rf)
            header = list(reader.fieldnames or [])
            existing_rows = list(reader)
        if header and header != columns:
            with open(csv_path, "w", newline="", encoding="utf-8") as wf:
                writer = csv.DictWriter(wf, fieldnames=columns, extrasaction="ignore")
                writer.writeheader()
                for erow in existing_rows:
                    writer.writerow(_remap_existing_row(erow, columns))
            write_header = False
        else:
            write_header = False
    else:
        write_header = True

    out = {}
    for col in columns:
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
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(out)

    print(f"Appended LCOE results to {csv_path}")
    return csv_path


if __name__ == "__main__":
    print("Import append_foundation_results(wt_opt, modeling_options) after run_wisdem.")
    print(f"Monopile CSV:  {MONOPILE_CSV_PATH}")
    print(f"Jacket CSV:    {JACKET_CSV_PATH}")
    print(f"Jacket 3-leg:  {JACKET_3_CSV_PATH}")
    print(f"LCOE CSV:      {LCOE_CSV_PATH}")
