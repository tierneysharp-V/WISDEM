#!/usr/bin/env python3
"""
Run WISDEM foundation scenarios from a site CSV + turbine lookup CSV.

Usage:
  python run_foundation_batch.py scenarios.csv --turbine-csv turbine_lookup.csv --foundation jacket
  python run_foundation_batch.py scenarios.csv --turbine-csv turbine_lookup.csv --foundation monopile
  python run_foundation_batch.py scenarios.csv --turbine-csv turbine_lookup.csv --foundation both
  python run_foundation_batch.py DOE_scenarios_500.csv --doe-results

Site CSV columns (required):
  turbine_rating_W, V50_mps, Hs50_m, soil_shear_modulus_Pa, water_depth_m

Optional site CSV column:
  n_legs — jacket leg count for this row (3 or 4). When omitted, --jacket-legs applies.

Turbine lookup CSV:
  First column = parameter (YAML path or special key). Remaining columns = turbines.
  Special row `_source_yaml` may point at a full geometry YAML; other rows override.
  Rating lookup uses `assembly.rated_power` (or `turbine_rating_W`).
  `rna_mass_kg` is lookup-only (results CSV); it is not written into geometry YAML.

Workflow per scenario:
  1. Look up turbine by rating and build a temporary full geometry YAML
  2. Stage 1: RotorSE+DriveSE compute RNA loads (V_mean = V50/7)
  3. Stage 2: Loading-driven monopile/jacket sizing with n_dlc=2
     - DLC1: rated GustETM RNA + rated_V
     - DLC2: parked extreme RNA (feathered-disk drag) + Uref=V50
  4. Frequency bounds from min/max rpm (soft-stiff; rated soft-stiff if band breached)
  5. Monopile/jacket: tower-scaled cold start -> preflight repair ->
     two-stage SLSQP; soil-only warm-start when a prior case matches
"""

from __future__ import annotations

import argparse
import copy
import csv
import importlib
import importlib.util
import math
import os
import sys
import traceback
from typing import Any

import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BATCH_DIR = os.path.dirname(os.path.realpath(__file__))
EXAMPLES_DIR = os.path.dirname(BATCH_DIR)
REPO_ROOT = os.path.dirname(EXAMPLES_DIR)
JACKET_DIR = os.path.join(EXAMPLES_DIR, "17_jacket")
TMP_DIR = os.path.join(BATCH_DIR, "tmp")

REQUIRED_COLUMNS = [
    "turbine_rating_W",
    "V50_mps",
    "Hs50_m",
    "soil_shear_modulus_Pa",
    "water_depth_m",
]

# Scenario fields that must match for soil-only warm start (everything except G)
_WARMSTART_MATCH_COLS = [
    "turbine_rating_W",
    "V50_mps",
    "Hs50_m",
    "water_depth_m",
]

BASELINES = {
    "jacket": {
        "wt": os.path.join(JACKET_DIR, "nrel5mw_jacket.yaml"),
        "modeling": os.path.join(JACKET_DIR, "modeling_options_jacket.yaml"),
        "analysis": os.path.join(BATCH_DIR, "analysis_options_jacket.yaml"),
    },
    "monopile": {
        "wt": os.path.join(BATCH_DIR, "nrel5mw_monopile.yaml"),
        "modeling": os.path.join(BATCH_DIR, "modeling_options_monopile.yaml"),
        "analysis": os.path.join(BATCH_DIR, "analysis_options_monopile.yaml"),
    },
}

MODELING_ROTORSE = os.path.join(BATCH_DIR, "modeling_options_rotorse.yaml")
ANALYSIS_LOADS = os.path.join(BATCH_DIR, "analysis_options_loads.yaml")
DEFAULT_TURBINE_CSV = os.path.join(BATCH_DIR, "turbine_lookup.csv")
DOE_RESULTS_PREFIX = os.path.join(BATCH_DIR, "DOE_results")

STAGE1_MAX_ITER = 50
_WARMSTART_MAX_UTIL = 1.05
_PREFLIGHT_GLOBAL_BUCK_TARGET = 2.0  # preflight tries to reduce global buckling util below this
_RATING_MATCH_RTOL = 1e-6
_RATING_MATCH_ATOL = 1.0  # Watts
# Lookup-CSV keys that must not be written into geometry YAML (schema / results-only).
_LOOKUP_ONLY_KEYS = frozenset({"rna_mass_kg", "rna_mass", "turbine_rating_W"})

# Early-design parked rotor: feathered blades as a low-Cd disk (not IEC DLC 6.x)
PARKED_ROTOR_CD = 0.04
_G = 9.80665


def _ensure_wisdem_namespace():
    """
    Register a minimal ``wisdem`` package before loading repo .py files.

    Blocks execution of ``wisdem/__init__.py`` (which imports runWISDEM → glue_code
    → JacketSE while jacket.py is still loading). Subpackages (commonse, etc.) are
    loaded normally from ``wisdem.__path__``.
    """
    import types

    if "wisdem" in sys.modules:
        return
    wisdem_root = os.path.join(REPO_ROOT, "wisdem")
    mod = types.ModuleType("wisdem")
    mod.__path__ = [wisdem_root]
    mod.__package__ = "wisdem"
    sys.modules["wisdem"] = mod


def _load_repo_module(qualname: str, relpath: str):
    """Prefer this checkout's modules over an older installed wisdem wheel."""
    path = os.path.join(REPO_ROOT, relpath)
    if not os.path.isfile(path):
        return None
    spec = importlib.util.spec_from_file_location(qualname, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[qualname] = mod
    spec.loader.exec_module(mod)
    return mod


def _apply_repo_module_to_package(repo_mod, qualname: str):
    """Bind a repo-loaded module onto the installed wisdem package namespace."""
    if repo_mod is None:
        return None
    pkg = sys.modules.get(qualname)
    if pkg is None or pkg is repo_mod:
        sys.modules[qualname] = repo_mod
        return repo_mod
    for name in dir(repo_mod):
        if name.startswith("__"):
            continue
        setattr(pkg, name, getattr(repo_mod, name))
    sys.modules[qualname] = pkg
    return pkg


def _setup_wisdem():
    import inspect

    _ensure_wisdem_namespace()

    # Load repo Python overrides BEFORE wisdem/__init__.py (runWISDEM → glue_code → jacket).
    dnvgl_mod = _load_repo_module(
        "wisdem.commonse.utilization_dnvgl",
        os.path.join("wisdem", "commonse", "utilization_dnvgl.py"),
    )
    jacket_mod = _load_repo_module("wisdem.fixed_bottomse.jacket", os.path.join("wisdem", "fixed_bottomse", "jacket.py"))
    _load_repo_module(
        "wisdem.fixed_bottomse.conceptual_design_rules",
        os.path.join("wisdem", "fixed_bottomse", "conceptual_design_rules.py"),
    )
    _load_repo_module("wisdem.fixed_bottomse.monopile", os.path.join("wisdem", "fixed_bottomse", "monopile.py"))

    val = _load_repo_module("wisdem.inputs.validation", os.path.join("wisdem", "inputs", "validation.py"))
    if val is not None:
        import wisdem.inputs as inputs_pkg

        for name in dir(val):
            if not name.startswith("_"):
                setattr(inputs_pkg, name, getattr(val, name))
    _load_repo_module("wisdem.glue_code.gc_WT_DataStruc", os.path.join("wisdem", "glue_code", "gc_WT_DataStruc.py"))
    _load_repo_module("wisdem.glue_code.gc_WT_InitModel", os.path.join("wisdem", "glue_code", "gc_WT_InitModel.py"))
    _load_repo_module("wisdem.glue_code.gc_LoadInputs", os.path.join("wisdem", "glue_code", "gc_LoadInputs.py"))
    _load_repo_module(
        "wisdem.glue_code.gc_PoseOptimization",
        os.path.join("wisdem", "glue_code", "gc_PoseOptimization.py"),
    )
    _load_repo_module("wisdem.orbit.orbit_api", os.path.join("wisdem", "orbit", "orbit_api.py"))
    _load_repo_module("wisdem.glue_code.glue_code", os.path.join("wisdem", "glue_code", "glue_code.py"))
    run_wisdem_mod = _load_repo_module(
        "wisdem.glue_code.runWISDEM", os.path.join("wisdem", "glue_code", "runWISDEM.py")
    )
    if run_wisdem_mod is None:
        import wisdem.glue_code.runWISDEM as run_wisdem_mod

        importlib.reload(run_wisdem_mod)

    dnvgl_mod = _apply_repo_module_to_package(dnvgl_mod, "wisdem.commonse.utilization_dnvgl")
    jacket_mod = _apply_repo_module_to_package(jacket_mod, "wisdem.fixed_bottomse.jacket")
    if jacket_mod is not None:
        fb_pkg = sys.modules.get("wisdem.fixed_bottomse")
        if fb_pkg is not None:
            fb_pkg.jacket = jacket_mod

    if dnvgl_mod is not None:
        sig = inspect.signature(dnvgl_mod.CylinderBuckling.__init__)
        dnvgl_path = inspect.getfile(dnvgl_mod)
        if "k" in sig.parameters:
            print(f"  Active utilization_dnvgl: {dnvgl_path}")
        else:
            print(f"  WARNING: utilization_dnvgl at {dnvgl_path} has no per-member k parameter.")
    if jacket_mod is not None and hasattr(jacket_mod, "JACKET_GLOBAL_BUCKLING_K_LEG"):
        jpath = inspect.getfile(jacket_mod)
        print(
            f"  Active jacket.py: {jpath}\n"
            "  Jacket global buckling k: "
            f"legs={jacket_mod.JACKET_GLOBAL_BUCKLING_K_LEG}, "
            f"braces={jacket_mod.JACKET_GLOBAL_BUCKLING_K_BRACE}"
        )
    sys.path.insert(0, EXAMPLES_DIR)
    from extract_foundation_results import append_foundation_results  # noqa: E402

    import wisdem.fixed_bottomse.conceptual_design_rules as cdr

    return run_wisdem_mod.run_wisdem, append_foundation_results, cdr


def _is_blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and value != value:  # NaN
        return True
    s = str(value).strip()
    return s == "" or s.lower() in {"nan", "none", "null"}


def _to_float(value: Any) -> float:
    return float(str(value).strip().replace(",", ""))


def _parse_cell(value: Any) -> Any:
    """Parse a turbine-CSV cell as YAML (supports scalars, lists, dicts)."""
    if _is_blank(value):
        return None
    if not isinstance(value, str):
        return value
    text = value.strip()
    try:
        return yaml.safe_load(text)
    except Exception:
        return text


def _set_by_path(root: dict, path: str, value: Any) -> None:
    """Set a nested dict/list value using dotted YAML-path keys (numeric list indices OK)."""
    keys = [k for k in path.split(".") if k]
    if not keys:
        raise ValueError("Empty parameter path")
    cur: Any = root
    for key in keys[:-1]:
        if key.isdigit():
            idx = int(key)
            if not isinstance(cur, list):
                raise TypeError(f"Cannot index non-list at '{key}' in path '{path}'")
            while len(cur) <= idx:
                cur.append({})
            if cur[idx] is None:
                cur[idx] = {}
            cur = cur[idx]
        else:
            if not isinstance(cur, dict):
                raise TypeError(f"Cannot descend into non-dict at '{key}' in path '{path}'")
            nxt = cur.get(key)
            if nxt is None:
                cur[key] = {}
                nxt = cur[key]
            cur = nxt
    last = keys[-1]
    if last.isdigit():
        idx = int(last)
        if not isinstance(cur, list):
            raise TypeError(f"Cannot assign into non-list at end of '{path}'")
        while len(cur) <= idx:
            cur.append(None)
        cur[idx] = value
    else:
        if not isinstance(cur, dict):
            raise TypeError(f"Cannot assign into non-dict at end of '{path}'")
        cur[last] = value


def _get_by_path(root: Any, path: str, default=None):
    cur = root
    for key in path.split("."):
        if cur is None:
            return default
        if key.isdigit():
            if not isinstance(cur, list) or int(key) >= len(cur):
                return default
            cur = cur[int(key)]
        else:
            if not isinstance(cur, dict) or key not in cur:
                return default
            cur = cur[key]
    return cur


def read_scenarios(csv_path: str) -> list[dict[str, Any]]:
    """Read site CSV; require expected column names; ignore extras."""
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header row: {csv_path}")

        headers = [h.strip() for h in reader.fieldnames]
        missing = [c for c in REQUIRED_COLUMNS if c not in headers]
        if missing:
            raise ValueError(
                "CSV is missing required column(s): "
                + ", ".join(missing)
                + f"\nFound columns: {headers}"
            )

        rows = []
        for i, raw in enumerate(reader, start=2):
            row = {k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in raw.items() if k}
            if all(_is_blank(row.get(c, "")) for c in REQUIRED_COLUMNS):
                continue
            row["_csv_row_number"] = i
            rows.append(row)

    if not rows:
        raise ValueError(f"No scenario rows found in {csv_path}")
    return rows


def load_turbine_lookup(csv_path: str) -> dict[str, dict[str, Any]]:
    """
    Load transposed turbine CSV.

    Returns {turbine_column_name: {parameter: parsed_value, ...}, ...}
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Turbine lookup CSV not found: {csv_path}")

    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Turbine CSV has no header row: {csv_path}")

        headers = [h.strip() for h in reader.fieldnames if h and h.strip()]
        if not headers:
            raise ValueError(f"Turbine CSV has no columns: {csv_path}")

        param_key = headers[0]
        turbine_cols = headers[1:]
        if not turbine_cols:
            raise ValueError("Turbine CSV needs at least one turbine column after the parameter column")

        turbines: dict[str, dict[str, Any]] = {c: {} for c in turbine_cols}
        for raw in reader:
            param = (raw.get(param_key) or "").strip()
            if not param or param.startswith("#"):
                continue
            for col in turbine_cols:
                if col not in raw:
                    continue
                parsed = _parse_cell(raw[col])
                if parsed is not None:
                    turbines[col][param] = parsed

    return turbines


def _resolve_source_yaml(path_value: Any, turbine_csv_path: str) -> str:
    path = str(path_value).strip()
    if os.path.isabs(path) and os.path.isfile(path):
        return path
    candidates = [
        os.path.normpath(os.path.join(os.path.dirname(turbine_csv_path), path)),
        os.path.normpath(os.path.join(BATCH_DIR, path)),
        os.path.normpath(os.path.join(EXAMPLES_DIR, path)),
        os.path.normpath(os.path.join(REPO_ROOT, path)),
    ]
    for cand in candidates:
        if os.path.isfile(cand):
            return cand
    raise FileNotFoundError(
        f"_source_yaml not found: {path_value}\nTried: " + ", ".join(candidates)
    )


def build_turbine_geometry(
    turbine_params: dict[str, Any],
    turbine_csv_path: str,
    foundation_type: str,
    foundation_baseline_wt: dict,
) -> dict:
    """
    Build a full geometry YAML from turbine lookup params + foundation baseline block.
    """
    wt: dict[str, Any] = {}
    source = turbine_params.get("_source_yaml")
    if source is not None:
        src_path = _resolve_source_yaml(source, turbine_csv_path)
        wt = load_yaml(src_path)

    for key, value in turbine_params.items():
        if key.startswith("_") or key in _LOOKUP_ONLY_KEYS:
            continue
        _set_by_path(wt, key, value)

    if "components" not in wt:
        raise ValueError("Turbine geometry missing 'components' (provide _source_yaml or full rows)")

    # Attach / replace foundation component from foundation baseline
    foundation_comp = copy.deepcopy(
        foundation_baseline_wt.get("components", {}).get(foundation_type)
    )
    if foundation_comp is None:
        raise ValueError(f"Foundation baseline missing components.{foundation_type}")
    # Remove the other foundation type if present
    other = "jacket" if foundation_type == "monopile" else "monopile"
    wt["components"].pop(other, None)
    wt["components"][foundation_type] = foundation_comp

    # Ensure materials include steel from foundation baseline if missing
    if "materials" not in wt and "materials" in foundation_baseline_wt:
        wt["materials"] = copy.deepcopy(foundation_baseline_wt["materials"])

    return wt


def slim_sizing_geometry(full_wt: dict, foundation_type: str) -> dict:
    """Tower+foundation-only geometry for Loading-driven sizing (RotorSE off)."""
    comps = full_wt.get("components", {})
    if "tower" not in comps or foundation_type not in comps:
        raise ValueError(f"Full geometry missing tower or {foundation_type} for sizing stage")
    slim = {
        "windIO_version": full_wt.get("windIO_version", "2.0"),
        "name": full_wt.get("name", "foundation_sizing"),
        "assembly": copy.deepcopy(full_wt.get("assembly", {})),
        "components": {
            "tower": copy.deepcopy(comps["tower"]),
            foundation_type: copy.deepcopy(comps[foundation_type]),
        },
        "materials": copy.deepcopy(full_wt.get("materials", [])),
        "control": copy.deepcopy(full_wt.get("control", {})),
    }
    return slim


def lookup_turbine_column(
    turbines: dict[str, dict[str, Any]],
    rating_w: float,
    turbine_csv_path: str,
) -> tuple[str, dict[str, Any]]:
    """Find turbine column whose rating matches rating_w."""
    matches = []
    for name, params in turbines.items():
        rating = params.get("assembly.rated_power", params.get("turbine_rating_W"))
        if rating is None and "_source_yaml" in params:
            # Peek source YAML for rating
            try:
                src = load_yaml(_resolve_source_yaml(params["_source_yaml"], turbine_csv_path))
                rating = _get_by_path(src, "assembly.rated_power")
            except Exception:
                rating = None
        if rating is None:
            continue
        if math.isclose(float(rating), float(rating_w), rel_tol=_RATING_MATCH_RTOL, abs_tol=_RATING_MATCH_ATOL):
            matches.append((name, params))

    if not matches:
        available = []
        for name, params in turbines.items():
            r = params.get("assembly.rated_power", params.get("turbine_rating_W", "?"))
            available.append(f"{name}={r}")
        raise ValueError(
            f"No turbine in lookup matches turbine_rating_W={rating_w}. "
            f"Known: {', '.join(available)}"
        )
    if len(matches) > 1:
        names = ", ".join(m[0] for m in matches)
        raise ValueError(f"Multiple turbines match rating {rating_w}: {names}")
    return matches[0]


def lookup_rna_mass(params: dict[str, Any]) -> float:
    """RNA mass [kg] from the turbine lookup row (``rna_mass_kg`` or ``rna_mass``)."""
    for key in ("rna_mass_kg", "rna_mass"):
        val = params.get(key)
        if val is not None and not _is_blank(val):
            return _to_float(val)
    return float("nan")


def _rotor_speed_limits_rpm(wt_geom: dict) -> tuple[float, float, float, str]:
    """
    Operating rotor-speed limits [rpm] from control, else constant-TSR cut-in/out.

    Returns (min_rpm, max_rpm, rated_rpm, source_label).
    rated_rpm falls back to the mid of [min, max] if not provided.
    """
    control = wt_geom.get("control", {}) or {}
    assembly = wt_geom.get("assembly", {}) or {}

    min_rpm = control.get("min_rotor_speed")
    max_rpm = control.get("max_rotor_speed")
    rated_rpm = control.get("rated_rotor_speed")
    if min_rpm is not None and max_rpm is not None and not _is_blank(min_rpm) and not _is_blank(max_rpm):
        mn, mx = float(min_rpm), float(max_rpm)
        if rated_rpm is not None and not _is_blank(rated_rpm):
            rd = float(rated_rpm)
        else:
            rd = 0.5 * (mn + mx)
        return mn, mx, rd, "min/max_rotor_speed"

    tsr = control.get("optimal_tsr")
    v_in = assembly.get("cut_in_wind_speed")
    v_out = assembly.get("cut_out_wind_speed")
    diameter = assembly.get("rotor_diameter")
    missing = [
        n
        for n, v in (
            ("control.optimal_tsr", tsr),
            ("assembly.cut_in_wind_speed", v_in),
            ("assembly.cut_out_wind_speed", v_out),
            ("assembly.rotor_diameter", diameter),
        )
        if v is None or _is_blank(v)
    ]
    if missing:
        raise ValueError(
            "Cannot compute frequency bounds: missing min/max_rotor_speed and also "
            + ", ".join(missing)
        )
    # Ω [rpm] = TSR * V * 60 / (π D) under constant-TSR
    mn = float(tsr) * float(v_in) * 60.0 / (math.pi * float(diameter))
    mx = float(tsr) * float(v_out) * 60.0 / (math.pi * float(diameter))
    if rated_rpm is not None and not _is_blank(rated_rpm):
        rd = float(rated_rpm)
    else:
        rd = 0.5 * (mn + mx)
    return mn, mx, rd, "TSR/cut-in/cut-out"


def rotor_excitation_bands_hz(min_rpm: float, max_rpm: float) -> tuple[tuple[float, float], tuple[float, float]]:
    """
    1P and 3P excitation bands [Hz] over the rotor-speed operating range.

    1P = [min_rpm/60, max_rpm/60]
    3P = [3*min_rpm/60, 3*max_rpm/60]
    """
    f1p = (float(min_rpm) / 60.0, float(max_rpm) / 60.0)
    f3p = (3.0 * float(min_rpm) / 60.0, 3.0 * float(max_rpm) / 60.0)
    return f1p, f3p


def classical_soft_stiff_bounds_hz(
    f1p: tuple[float, float], f3p: tuple[float, float]
) -> tuple[float, float]:
    """
    Classical soft-stiff window: above all 1P and below all 3P.

    lower = max(1P) = f1p[1]
    upper = min(3P) = f3p[0]
    Valid only when lower <= upper (rotor speed ratio max/min <= 3).
    """
    return float(f1p[1]), float(f3p[0])


def rated_soft_stiff_bounds_hz(rated_rpm: float) -> tuple[float, float]:
    """
    Soft-stiff band referenced to rated rotor speed only:
    [Ω_rated/60, 3·Ω_rated/60] = above 1P_rated and below 3P_rated.
    """
    f1p_r = float(rated_rpm) / 60.0
    return f1p_r, 3.0 * f1p_r


def maximin_clearance_frequency_hz(
    f1p: tuple[float, float], f3p: tuple[float, float]
) -> tuple[float, float]:
    """
    Frequency that maximises the minimum signed clearance to the 1P/3P bands
    that define the classical soft-stiff gap.

    Let a = max(1P), b = min(3P). For a candidate f, signed clearances are
        c_1P = f - a   (positive when above the 1P band)
        c_3P = b - f   (positive when below the 3P band)
    Maximise min(c_1P, c_3P). The optimum is the midpoint
        f* = (a + b) / 2
    with half-gap δ = (b - a) / 2  (δ > 0 ⇒ valid soft-stiff gap;
    δ < 0 ⇒ overlap / breach, f* still minimises the worst intrusion).

    Returns (f_star_hz, signed_half_gap_hz).
    """
    a = float(f1p[1])  # max 1P
    b = float(f3p[0])  # min 3P
    f_star = 0.5 * (a + b)
    half_gap = 0.5 * (b - a)
    return f_star, half_gap


def compute_frequency_bounds(wt_geom: dict) -> tuple[float, float]:
    """
    Soft-stiff f1 bounds from turbine control/assembly.

    Prefer min/max rotor rpm; else constant-TSR cut-in/out speeds.

    If a classical soft-stiff band exists (max 1P <= min 3P), return that band.
    If the band is breached (1P/3P ranges overlap), fall back to rated soft-stiff
    [Ω_rated/60, 3·Ω_rated/60].
    """
    min_rpm, max_rpm, rated_rpm, source = _rotor_speed_limits_rpm(wt_geom)
    if not (math.isfinite(min_rpm) and math.isfinite(max_rpm)) or min_rpm <= 0.0 or max_rpm <= 0.0:
        raise ValueError(f"Non-finite or non-positive rotor speeds: min={min_rpm}, max={max_rpm}")
    if min_rpm > max_rpm:
        raise ValueError(f"Invalid rotor speed limits: min_rpm={min_rpm} > max_rpm={max_rpm}")
    if not math.isfinite(rated_rpm) or rated_rpm <= 0.0:
        raise ValueError(f"Non-finite or non-positive rated rotor speed: {rated_rpm}")

    f1p, f3p = rotor_excitation_bands_hz(min_rpm, max_rpm)
    f_lo, f_hi = classical_soft_stiff_bounds_hz(f1p, f3p)

    if not (math.isfinite(f_lo) and math.isfinite(f_hi)):
        raise ValueError(f"Non-finite frequency bounds: lo={f_lo}, hi={f_hi}")

    if f_lo <= f_hi:
        print(f"  Frequency bounds ({source}): [{f_lo:.5f}, {f_hi:.5f}] Hz")
        print(
            f"    1P=[{f1p[0]:.5f}, {f1p[1]:.5f}] Hz  "
            f"3P=[{f3p[0]:.5f}, {f3p[1]:.5f}] Hz"
        )
        return f_lo, f_hi

    # Classical soft-stiff window empty: use rated soft-stiff instead.
    f_lo_r, f_hi_r = rated_soft_stiff_bounds_hz(rated_rpm)
    f_star, half_gap = maximin_clearance_frequency_hz(f1p, f3p)
    print(
        "  WARNING: soft-stiff frequency breach; "
        "using rated soft-stiff [Ω_rated/60, 3·Ω_rated/60]"
    )
    print(
        f"  Frequency bounds ({source}, rated soft-stiff): [{f_lo_r:.5f}, {f_hi_r:.5f}] Hz  "
        f"(rated_rpm={rated_rpm:.4f})"
    )
    print(
        f"    1P=[{f1p[0]:.5f}, {f1p[1]:.5f}] Hz  "
        f"3P=[{f3p[0]:.5f}, {f3p[1]:.5f}] Hz  "
        f"classical would-be=[{f_lo:.5f}, {f_hi:.5f}] Hz  "
        f"maximin f*={f_star:.5f} Hz (half-gap={half_gap:.5f} Hz)"
    )
    if f_lo_r >= f_hi_r:
        raise ValueError(
            f"Rated soft-stiff bounds invalid: lower={f_lo_r:.6g} >= upper={f_hi_r:.6g}"
        )
    return f_lo_r, f_hi_r


def patch_frequency_bounds(
    analysis: dict, f_lo: float, f_hi: float, foundation_type: str = "monopile"
) -> dict:
    opts = copy.deepcopy(analysis)
    freq = (
        opts.setdefault("constraints", {})
        .setdefault(foundation_type, {})
        .setdefault("frequency_1", {})
    )
    # Monopile sizing always uses soft-stiff. Jackets only patch bounds when already enabled.
    if foundation_type == "monopile" or freq.get("flag", False):
        freq["flag"] = True
        freq["lower_bound"] = float(f_lo)
        freq["upper_bound"] = float(f_hi)
    return opts



def load_yaml(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def dump_yaml(data: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)


def patch_environment(baseline: dict, scenario: dict, for_rotorse: bool = False) -> dict:
    """Copy modeling options and apply site environment fields."""
    opts = copy.deepcopy(baseline)
    env = opts.setdefault("Environment", {})

    mapping_env = {
        "water_depth_m": "water_depth",
        "Hs50_m": "significant_wave_height",
        "soil_shear_modulus_Pa": "soil_shear_modulus",
    }
    for csv_col, yaml_key in mapping_env.items():
        if not _is_blank(scenario.get(csv_col)):
            env[yaml_key] = _to_float(scenario[csv_col])

    # IEC: V_extreme50 = 1.4*Vref, V_mean = 0.2*Vref => V_mean = V50/7
    if for_rotorse and not _is_blank(scenario.get("V50_mps")):
        env["V_mean"] = _to_float(scenario["V50_mps"]) / 7.0

    opts["Environment"] = env
    return opts


def estimate_parked_extreme_loads(
    loads: dict,
    scenario: dict,
    wt_geom: dict,
    rho_air: float = 1.225,
) -> dict:
    """
    Physics-light parked extreme RNA loads at V50 (early-design screening).

    Fx = 0.5 * rho * Cd * A * V50^2 with Cd=PARKED_ROTOR_CD (feathered disk).
    Fz = -mass * g; tower-top moments from r_cm × F; Uref = V50.
    """
    if _is_blank(scenario.get("V50_mps")):
        raise ValueError("V50_mps is required for parked extreme Loading case")
    v50 = _to_float(scenario["V50_mps"])
    if v50 <= 0.0:
        raise ValueError(f"V50_mps must be positive, got {v50}")

    diameter = (wt_geom.get("assembly") or {}).get("rotor_diameter")
    if diameter is None or _is_blank(diameter):
        raise ValueError("assembly.rotor_diameter required for parked extreme load estimate")
    d_rotor = float(diameter)
    if d_rotor <= 0.0:
        raise ValueError(f"rotor_diameter must be positive, got {d_rotor}")

    area = math.pi * (0.5 * d_rotor) ** 2
    mass = float(loads["mass"])
    cm = [float(x) for x in loads["center_of_mass"][:3]]
    while len(cm) < 3:
        cm.append(0.0)

    fx = 0.5 * float(rho_air) * PARKED_ROTOR_CD * area * v50**2
    fy = 0.0
    fz = -mass * _G
    rx, ry, rz = cm[0], cm[1], cm[2]
    # M = r_cm × F at tower top
    mx = ry * fz - rz * fy
    my = rz * fx - rx * fz
    mz = rx * fy - ry * fx

    return {
        "force": [float(fx), float(fy), float(fz)],
        "moment": [float(mx), float(my), float(mz)],
        "velocity": float(v50),
        "rotor_diameter_m": d_rotor,
        "disk_area_m2": float(area),
        "Cd": float(PARKED_ROTOR_CD),
    }


def inject_loading(
    baseline: dict, scenario: dict, loads: dict, wt_geom: dict
) -> tuple[dict, dict]:
    """Loading-driven modeling options: rated GustETM + parked extreme (n_dlc=2)."""
    opts = patch_environment(baseline, scenario, for_rotorse=False)
    wisdem = opts.setdefault("WISDEM", {})
    wisdem["n_dlc"] = 2
    wisdem.setdefault("RotorSE", {})["flag"] = False
    wisdem.setdefault("DriveSE", {})["flag"] = False
    wisdem.setdefault("TowerSE", {})["flag"] = True
    wisdem.setdefault("FixedBottomSE", {})["flag"] = True

    loading = wisdem.setdefault("Loading", {})
    loading["mass"] = float(loads["mass"])
    loading["center_of_mass"] = [float(x) for x in loads["center_of_mass"]]
    loading["moment_of_inertia"] = [float(x) for x in loads["moment_of_inertia"]]

    force = [float(x) for x in loads["force"]]
    moment = [float(x) for x in loads["moment"]]
    while len(force) < 3:
        force.append(0.0)
    while len(moment) < 3:
        moment.append(0.0)

    rho_air = float(opts.get("Environment", {}).get("air_density", 1.225))
    extreme = estimate_parked_extreme_loads(loads, scenario, wt_geom, rho_air=rho_air)

    loading["loads"] = [
        {
            "force": force[:3],
            "moment": moment[:3],
            "velocity": float(loads["velocity"]),
        },
        {
            "force": extreme["force"],
            "moment": extreme["moment"],
            "velocity": extreme["velocity"],
        },
    ]
    wisdem["Loading"] = loading
    opts["WISDEM"] = wisdem
    return opts, extreme


def patch_geometry(baseline_wt: dict, modeling: dict, foundation_type: str, cdr, embedment=None) -> dict:
    """Apply conceptual height / bay topology rules to geometry YAML."""
    return cdr.apply_wt_geometry_rules(
        baseline_wt, modeling, foundation_type, embedment=embedment
    )


def _as_float_list(val) -> list[float]:
    arr = np.atleast_1d(np.asarray(val, dtype=float).flatten())
    return [float(x) for x in arr]


def _scenario_key_for_warmstart(scenario: dict) -> tuple:
    """Hashable key of site/turbine fields (excludes soil G)."""
    parts = []
    for col in _WARMSTART_MATCH_COLS:
        v = scenario.get(col, "")
        if _is_blank(v):
            parts.append(None)
        else:
            parts.append(round(_to_float(v), 6))
    return tuple(parts)


def tower_base_diameter_from_geom(wt_geom: dict) -> float:
    """Tower base outer diameter from geometry YAML (first station)."""
    vals = (
        wt_geom.get("components", {})
        .get("tower", {})
        .get("outer_shape", {})
        .get("outer_diameter", {})
        .get("values", [])
    )
    arr = np.atleast_1d(np.asarray(vals, dtype=float).flatten())
    if arr.size == 0 or not np.isfinite(arr[0]):
        raise ValueError("Cannot read tower base diameter from geometry")
    return float(arr[0])


def monopile_diameter_floor(wt_geom: dict) -> float:
    """Strict D floor: max(YAML DV lower bound, tower base) for diameter coupling."""
    return float(max(_D_LO, tower_base_diameter_from_geom(wt_geom)))


def monopile_thickness_floor(diameter_m: float) -> float:
    """Minimum wall thickness [m]: t >= D/120 (see conceptual_design_rules)."""
    from wisdem.fixed_bottomse.conceptual_design_rules import monopile_min_wall_thickness

    return monopile_min_wall_thickness(diameter_m)


def seed_monopile_diameter_thickness(wt_geom: dict, diameter: float, thickness: float) -> dict:
    """Overwrite monopile outer diameter and wall thickness (uniform). Enforces D >= tower base."""
    wt = copy.deepcopy(wt_geom)
    d_floor = monopile_diameter_floor(wt)
    d = _clip(max(float(diameter), d_floor), _D_LO, _D_HI)
    t_floor = monopile_thickness_floor(d)
    t = _clip(max(float(thickness), t_floor), max(_T_LO, t_floor), _T_HI)
    mono = wt["components"]["monopile"]
    n_d = len(mono["outer_shape"]["outer_diameter"]["values"])
    mono["outer_shape"]["outer_diameter"]["values"] = [d] * n_d
    n_t = len(mono["structure"]["layers"][0]["thickness"]["values"])
    mono["structure"]["layers"][0]["thickness"]["values"] = [t] * n_t
    return wt


def _clip(v: float, lo: float, hi: float) -> float:
    return float(min(hi, max(lo, v)))


def _clamp_tube_wall(diameter: float, thickness: float, t_lo: float | None = None) -> float:
    """Keep tubular wall thickness within [t_min, 0.45·D] so Frame3DD Area stays > 0."""
    d = float(diameter)
    t = float(thickness)
    floor = float(_J_LEG_T_LO if t_lo is None else t_lo)
    if not np.isfinite(d) or d <= 0.0:
        return floor
    t_min = floor
    if _J_ENFORCE_D_TO_T:
        t_min = max(t_min, d / _J_D_TO_T_MAX)
    t_max = _J_T_OVER_D_MAX * d
    return _clip(t, min(t_min, t_max), t_max)


def seed_jacket_geometry(
    wt_geom: dict,
    r_head: float,
    foot_head_ratio: float,
    leg_diameter: float,
    leg_thickness: float,
    brace_diameters: list[float] | None = None,
    brace_thicknesses: list[float] | None = None,
) -> dict:
    """Overwrite jacket sizing DVs (keeps height / n_bays / bay_spacing)."""
    wt = copy.deepcopy(wt_geom)
    jacket = wt["components"]["jacket"]
    n_bays = int(jacket.get("n_bays", len(jacket.get("brace_diameters", [])) or 3))
    jacket["r_head"] = float(r_head)
    jacket["r_foot"] = float(r_head) * float(foot_head_ratio)
    leg_d = _clip(float(leg_diameter), _J_LEG_D_LO, _J_LEG_D_HI)
    leg_t = _clamp_tube_wall(leg_d, _clip(float(leg_thickness), _J_LEG_T_LO, _J_LEG_T_HI), t_lo=_J_LEG_T_LO)
    jacket["leg_diameter"] = leg_d
    jacket["leg_thickness"] = leg_t

    def _brace(key: str, vals: list[float] | None, default: float) -> list[float]:
        if vals is None:
            old = list(jacket.get(key, []))
            seed = float(old[-1]) if old else float(default)
            return [seed] * n_bays
        arr = [float(x) for x in vals]
        if len(arr) == n_bays:
            return arr
        if len(arr) == 1:
            return arr * n_bays
        if len(arr) > n_bays:
            return arr[:n_bays]
        return arr + [arr[-1]] * (n_bays - len(arr))

    brace_d = [
        _clip(d, _J_BRACE_D_LO, _J_BRACE_D_HI)
        for d in _brace("brace_diameters", brace_diameters, 1.0)
    ]
    raw_t = _brace("brace_thicknesses", brace_thicknesses, 0.1)
    if len(raw_t) != len(brace_d):
        raw_t = (raw_t + [raw_t[-1]] * len(brace_d))[: len(brace_d)]
    brace_t = [
        _clamp_tube_wall(d, _clip(t, _J_BRACE_T_LO, _J_BRACE_T_HI), t_lo=_J_BRACE_T_LO)
        for d, t in zip(brace_d, raw_t)
    ]
    jacket["brace_diameters"] = brace_d
    jacket["brace_thicknesses"] = brace_t
    return wt


def extract_jacket_dvs(wt_opt) -> tuple[float, float, float, float, list[float], list[float]]:
    """Return (r_head, foot_head_ratio, leg_D, leg_t, brace_Ds, brace_ts)."""
    r_head = float(_safe_get(wt_opt, "jacket.r_head", units="m"))
    ratio = float(_safe_get(wt_opt, "jacket.foot_head_ratio"))
    if not np.isfinite(ratio):
        r_foot = float(_safe_get(wt_opt, "fixedse.r_foot", units="m"))
        ratio = r_foot / r_head if np.isfinite(r_head) and r_head > 0 else float("nan")
    leg_d = float(_safe_get(wt_opt, "jacket.leg_diameter", units="m"))
    leg_t = float(_safe_get(wt_opt, "jacket.leg_thickness", units="m"))
    try:
        brace_d = _as_float_list(wt_opt.get_val("jacket.brace_diameters", units="m"))
    except Exception:
        brace_d = [1.0]
    try:
        brace_t = _as_float_list(wt_opt.get_val("jacket.brace_thicknesses", units="m"))
    except Exception:
        brace_t = [0.1]
    return r_head, ratio, leg_d, leg_t, brace_d, brace_t


def scale_jacket_cold_start(wt_geom: dict) -> dict:
    """
    Replace the shared NREL5MW jacket seed with a tower-scaled start.

    r_head sits on D_tower (coupling floor). Foot flare targets the mid of the
    practical batter band (r_foot - r_head)/height ≈ 0.125.
    """
    d_tow = tower_base_diameter_from_geom(wt_geom)
    r_head = _clip(d_tow, _J_RHEAD_LO, _J_RHEAD_HI)
    height = float(wt_geom["components"]["jacket"].get("height", 40.0))
    if height <= 0.0:
        height = 40.0
    # Mid of [0.1, 0.15] flare band → ratio = 1 + flare * H / r_head
    flare_target = 0.5 * (_J_FLARE_LO + _J_FLARE_HI)
    ratio = _clip(1.0 + flare_target * height / r_head, _J_RATIO_LO, _J_RATIO_HI)
    leg_d = _clip(0.28 * d_tow, _J_LEG_D_LO, _J_LEG_D_HI)
    leg_t = _clip(0.01 * d_tow, _J_LEG_T_LO, _J_LEG_T_HI)
    brace_d = _clip(0.55 * leg_d, _J_BRACE_D_LO, _J_BRACE_D_HI)
    brace_t = _clip(max(_J_BRACE_T_LO, 0.5 * leg_t), _J_BRACE_T_LO, _J_BRACE_T_HI)
    print(
        f"  Cold-start jacket from D_tower={d_tow:.3f} m: "
        f"r_head={r_head:.3f} m, ratio={ratio:.3f} "
        f"(flare≈{flare_target:.3f}), "
        f"leg_D={leg_d:.3f} m, leg_t={leg_t*1e3:.1f} mm, "
        f"brace_D={brace_d:.3f} m, brace_t={brace_t*1e3:.1f} mm"
    )
    return seed_jacket_geometry(
        wt_geom, r_head, ratio, leg_d, leg_t, [brace_d], [brace_t]
    )


def scale_monopile_cold_start(wt_geom: dict) -> dict:
    """
    Replace the shared NREL5MW monopile seed with a tower-scaled start.

    Uniform D sits on D_tower (diameter-coupling floor). Thickness keeps the
    baseline wall or a mild D-scaled minimum, whichever is larger.
    """
    d_tow = tower_base_diameter_from_geom(wt_geom)
    d = _clip(d_tow, _D_LO, _D_HI)
    try:
        t0 = float(
            wt_geom["components"]["monopile"]["structure"]["layers"][0]["thickness"]["values"][0]
        )
    except Exception:
        t0 = _T_LO
    t = _clip(max(t0, monopile_thickness_floor(d)), monopile_thickness_floor(d), _T_HI)
    print(
        f"  Cold-start monopile from D_tower={d_tow:.3f} m: "
        f"D={d:.3f} m, t={t*1e3:.1f} mm"
    )
    return seed_monopile_diameter_thickness(wt_geom, d, t)


def patch_monopile_diameter_bounds(analysis: dict, wt_geom: dict) -> dict:
    """Raise monopile outer_diameter DV lower_bound to the tower-base floor."""
    opts = copy.deepcopy(analysis)
    d_floor = monopile_diameter_floor(wt_geom)
    od = (
        opts.setdefault("design_variables", {})
        .setdefault("monopile", {})
        .setdefault("outer_diameter", {})
    )
    lo = float(od.get("lower_bound", _D_LO))
    hi = float(od.get("upper_bound", _D_HI))
    new_lo = max(lo, d_floor)
    if new_lo > hi:
        raise ValueError(
            f"Monopile D floor {new_lo:.3f} m exceeds DV upper_bound {hi:.3f} m "
            f"(tower base / coupling)"
        )
    od["lower_bound"] = new_lo
    print(
        f"  Monopile D DV bounds: lower={new_lo:.3f} m (tower-base floor), "
        f"upper={hi:.3f} m"
    )
    return opts


def patch_monopile_thickness_bounds(analysis: dict, wt_geom: dict) -> dict:
    """Raise monopile layer_thickness DV lower_bound to D/120 at D floor."""
    opts = copy.deepcopy(analysis)
    lt = (
        opts.setdefault("design_variables", {})
        .setdefault("monopile", {})
        .setdefault("layer_thickness", {})
    )
    if not lt.get("flag", False):
        return opts

    od = opts["design_variables"]["monopile"].get("outer_diameter", {})
    d_ref = float(od.get("lower_bound", monopile_diameter_floor(wt_geom)))
    t_floor = monopile_thickness_floor(d_ref)
    lo = float(lt.get("lower_bound", _T_LO))
    hi = float(lt.get("upper_bound", _T_HI))
    new_lo = max(lo, t_floor)
    if new_lo > hi:
        raise ValueError(
            f"Monopile t floor {new_lo:.6f} m (D={d_ref:.3f} m) exceeds "
            f"DV upper_bound {hi:.6f} m"
        )
    lt["lower_bound"] = new_lo

    mono_c = opts.setdefault("constraints", {}).setdefault("monopile", {})
    mono_c.setdefault("wall_thickness_min", {})["flag"] = True

    print(
        f"  Monopile t DV bounds: lower={new_lo:.6f} m "
        f"(D/120 at D={d_ref:.3f} m), upper={hi:.6f} m"
    )
    return opts


def monopile_height_dv_enabled(analysis: dict) -> bool:
    """True when monopile total height is an optimization design variable."""
    return bool(
        analysis.get("design_variables", {})
        .get("monopile", {})
        .get("height", {})
        .get("flag", False)
    )


def extract_monopile_dt(wt_opt) -> tuple[float, float]:
    """Return (diameter_m, thickness_m) from an optimised problem."""
    diam = _as_float_list(wt_opt.get_val("monopile.diameter", units="m"))
    d = float(diam[0])
    try:
        thick = np.atleast_2d(np.asarray(wt_opt.get_val("monopile.layer_thickness", units="m"), dtype=float))
        t = float(thick[0].flatten()[0])
    except Exception:
        t = float(wt_opt.get_val("fixedse.monopile_wall_thickness", units="m").flatten()[0])
    return d, t


def write_optimised_monopile_geometry(
    wt_geom: dict,
    wt_opt,
    modeling: dict,
    cdr,
    *,
    use_opt_embedment: bool = False,
) -> dict:
    """Copy optimised D/t (and embedment) into geometry YAML."""
    wt = copy.deepcopy(wt_geom)
    mono = wt["components"]["monopile"]
    d_floor = monopile_diameter_floor(wt)

    diam = _as_float_list(wt_opt.get_val("monopile.diameter", units="m"))
    diam = [max(float(x), d_floor) for x in diam]
    n = len(mono["outer_shape"]["outer_diameter"]["values"])
    if len(diam) == 1:
        diam_vals = diam * n
    elif len(diam) >= n:
        diam_vals = diam[:n]
    else:
        diam_vals = list(diam) + [diam[-1]] * (n - len(diam))
    mono["outer_shape"]["outer_diameter"]["values"] = diam_vals

    try:
        thick = np.atleast_2d(np.asarray(wt_opt.get_val("monopile.layer_thickness", units="m"), dtype=float))
        layer0 = thick[0].flatten()
        n_t = len(mono["structure"]["layers"][0]["thickness"]["values"])
        if layer0.size == 1:
            t_vals = [float(layer0[0])] * n_t
        elif layer0.size >= n_t:
            t_vals = [float(x) for x in layer0[:n_t]]
        else:
            t_vals = [float(x) for x in layer0] + [float(layer0[-1])] * (n_t - layer0.size)
        mono["structure"]["layers"][0]["thickness"]["values"] = t_vals
    except Exception:
        pass

    d_mud = float(diam_vals[0])
    t_floor = monopile_thickness_floor(d_mud)
    t_vals = [
        max(t_floor, float(x))
        for x in mono["structure"]["layers"][0]["thickness"]["values"]
    ]
    mono["structure"]["layers"][0]["thickness"]["values"] = t_vals
    t_mud = float(t_vals[0])
    if use_opt_embedment:
        l_emb = float(
            np.atleast_1d(wt_opt.get_val("fixedse.suctionpile_depth", units="m")).flatten()[0]
        )
    else:
        l_emb = cdr.embedment_length_from_wt(wt, modeling, diameter=d_mud, thickness=t_mud)
    site = cdr.resolve_site_params(modeling)
    print(f"  Post-opt embedment: L_emb = {l_emb:.4f} m (D={d_mud:.4f} m, t={t_mud:.4f} m)")
    print(f"  H_free = {site['H_free']:.4f} m; total height = {site['H_free'] + l_emb:.4f} m")

    return cdr.apply_wt_geometry_rules(wt, modeling, "monopile", embedment=l_emb)


def analysis_refresh_options(analysis: dict) -> dict:
    """Disable optimisation for the monopile embedment refresh pass."""
    opts = copy.deepcopy(analysis)
    opts.setdefault("driver", {}).setdefault("optimization", {})["flag"] = False
    mono = opts.setdefault("design_variables", {}).setdefault("monopile", {})
    mono.setdefault("height", {})["flag"] = False
    mono.setdefault("outer_diameter", {})["flag"] = False
    mono.setdefault("layer_thickness", {})["flag"] = False
    return opts


def analysis_jacket_run_only(analysis: dict) -> dict:
    """Disable optimisation for jacket preflight / diagnostic run_only passes."""
    opts = copy.deepcopy(analysis)
    opts.setdefault("driver", {}).setdefault("optimization", {})["flag"] = False
    jacket = opts.setdefault("design_variables", {}).setdefault("jacket", {})
    for key in (
        "foot_head_ratio",
        "r_head",
        "leg_thickness",
        "leg_diameter",
        "brace_diameters",
        "brace_thicknesses",
        "height",
        "bay_spacing",
    ):
        jacket.setdefault(key, {})["flag"] = False
    return opts


def analysis_stage1_strength(analysis: dict) -> dict:
    """Stage 1: full monopile constraints including frequency."""
    opts = copy.deepcopy(analysis)
    opts.setdefault("constraints", {}).setdefault("monopile", {}).setdefault("frequency_1", {})["flag"] = True
    opts.setdefault("driver", {}).setdefault("optimization", {})["max_iter"] = STAGE1_MAX_ITER
    return opts


def analysis_stage2_full(analysis: dict) -> dict:
    """Stage 2: polish with full constraints and the standard iteration budget."""
    opts = copy.deepcopy(analysis)
    opts.setdefault("constraints", {}).setdefault("monopile", {}).setdefault("frequency_1", {})["flag"] = True
    return opts


def jacket_frequency_enabled(analysis: dict) -> bool:
    """True when jacket soft-stiff frequency is requested in the base analysis YAML."""
    return bool(
        analysis.get("constraints", {}).get("jacket", {}).get("frequency_1", {}).get("flag", False)
    )


def analysis_jacket_stage1(analysis: dict) -> dict:
    """Stage 1 jacket: stress/shell/flare (global buckling and frequency off), shorter budget."""
    opts = copy.deepcopy(analysis)
    opts.setdefault("driver", {}).setdefault("optimization", {})["max_iter"] = STAGE1_MAX_ITER
    jacket_constr = opts.setdefault("constraints", {}).setdefault("jacket", {})
    jacket_constr.setdefault("global_buckling", {})["flag"] = False
    jacket_constr.setdefault("frequency_1", {})["flag"] = False
    return opts


def analysis_jacket_stage2(analysis: dict) -> dict:
    """Stage 2 jacket: full YAML constraints including global buckling and frequency."""
    opts = copy.deepcopy(analysis)
    if jacket_frequency_enabled(analysis):
        opt_drv = opts.setdefault("driver", {}).setdefault("optimization", {})
        # Stabler finite-difference gradients for the modal frequency constraint
        opt_drv["step_size"] = max(float(opt_drv.get("step_size", 1e-6)), 1e-5)
        opt_drv["form"] = "central"
    return opts


def get_opt_exit_mode(wt_opt) -> Any:
    """Best-effort SciPy / OpenMDAO exit status after run_driver."""
    import re

    driver = getattr(wt_opt, "driver", None)
    if driver is None:
        return np.nan
    for attr in ("result", "results", "_result"):
        result = getattr(driver, attr, None)
        if result is None:
            continue
        for key in ("status", "exit_status", "exit_code"):
            status = getattr(result, key, None)
            if status is not None:
                try:
                    return int(status)
                except (TypeError, ValueError):
                    pass
        msg = f"{getattr(result, 'message', '')} {result}"
        m = re.search(r"Exit mode\s+(\d+)", msg, re.I)
        if m:
            return int(m.group(1))
        success = getattr(result, "success", None)
        if success is True:
            return 0
        if success is False:
            return 8
    fail = getattr(driver, "_failed", None)
    if fail is False:
        return 0
    if fail is True:
        return 8
    return np.nan


def _safe_get(wt_opt, name, units=None, reduce="first"):
    try:
        if units is None:
            val = wt_opt.get_val(name)
        else:
            val = wt_opt.get_val(name, units=units)
        arr = np.atleast_1d(np.asarray(val, dtype=float).flatten())
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return float("nan")
        if reduce == "max":
            return float(np.nanmax(finite))
        return float(finite[0])
    except Exception:
        return float("nan")


def max_strength_utilization(wt_opt) -> float:
    """Peak stress / buckling utilization (nan if unavailable)."""
    vals = []
    for name in (
        "fixedse.post_monopile_tower.constr_stress",
        "fixedse.post_monopile_tower.constr_global_buckling",
        "fixedse.post_monopile_tower.constr_shell_buckling",
        "fixedse.post.constr_stress",
        "fixedse.post.constr_global_buckling",
        "fixedse.post.constr_shell_buckling",
        "fixedse.constr_stress",
        "fixedse.constr_global_buckling",
        "fixedse.constr_shell_buckling",
    ):
        v = _safe_get(wt_opt, name, reduce="max")
        if np.isfinite(v):
            vals.append(v)
    return float(max(vals)) if vals else float("nan")


def max_jacket_global_buckling_util(wt_opt) -> float:
    """Peak jacket global buckling utilization (nan if unavailable)."""
    for name in (
        "fixedse.constr_global_buckling",
        "fixedse.post.constr_global_buckling",
    ):
        v = _safe_get(wt_opt, name, reduce="max")
        if np.isfinite(v):
            return float(v)
    return float("nan")


def max_jacket_strength_util(wt_opt) -> float:
    """Peak jacket stress / shell buckling utilization (excludes global buckling)."""
    vals = []
    for name in (
        "fixedse.constr_stress",
        "fixedse.constr_shell_buckling",
        "fixedse.post.constr_stress",
        "fixedse.post.constr_shell_buckling",
    ):
        v = _safe_get(wt_opt, name, reduce="max")
        if np.isfinite(v):
            vals.append(v)
    return float(max(vals)) if vals else float("nan")


def _jacket_preflight_scale(strength_util: float, global_buck: float) -> float:
    """
    Member size multiplier for jacket preflight repair.

    Uses modest scaling for stress/shell, and more aggressive scaling when
    global buckling utilization is very high.
    """
    scale = 1.0
    if np.isfinite(strength_util) and strength_util > 1.0:
        scale = max(scale, min(1.5, strength_util * 1.05))
    if np.isfinite(global_buck) and global_buck > 1.0:
        if global_buck >= 10.0:
            g_scale = min(3.0, global_buck ** 0.35)
        elif global_buck >= 3.0:
            g_scale = min(2.0, global_buck ** 0.5)
        else:
            g_scale = min(1.5, global_buck * 1.05)
        scale = max(scale, g_scale)
    return scale


def jacket_preflight_is_ready(
    wt_opt,
    f_lo: float,
    f_hi: float,
    check_frequency: bool = False,
) -> bool:
    """True when jacket preflight has stress/shell and global buckling roughly feasible."""
    strength = max_jacket_strength_util(wt_opt)
    global_buck = max_jacket_global_buckling_util(wt_opt)
    if np.isfinite(strength) and strength > _WARMSTART_MAX_UTIL:
        return False
    if np.isfinite(global_buck) and global_buck > _PREFLIGHT_GLOBAL_BUCK_TARGET:
        return False
    if check_frequency:
        f1 = _safe_get(wt_opt, "fixedse.structural_frequencies")
        if not np.isfinite(f1) or f1 < f_lo or f1 > f_hi:
            return False
    return True


def design_is_warmstart_safe(wt_opt, f_lo: float, f_hi: float, check_frequency: bool = True) -> bool:
    """True if strength utils (and optionally f1) look feasible enough to seed the next case."""
    util = max_strength_utilization(wt_opt)
    if not np.isfinite(util) or util > _WARMSTART_MAX_UTIL:
        return False
    if check_frequency:
        f1 = _safe_get(wt_opt, "fixedse.structural_frequencies")
        if not np.isfinite(f1) or f1 < f_lo or f1 > f_hi:
            return False
    return True


def print_monopile_diagnostics(wt_opt, label: str, exit_mode=None) -> None:
    """Log exit mode, D/t/mass/f1 and peak utilizations."""
    d = _safe_get(wt_opt, "monopile.diameter", units="m", reduce="max")
    t = _safe_get(wt_opt, "fixedse.monopile_wall_thickness", units="m", reduce="max")
    if not np.isfinite(t):
        t = _safe_get(wt_opt, "monopile.layer_thickness", units="m", reduce="max")
    mass = _safe_get(wt_opt, "fixedse.monopile_mass", units="kg")
    f1 = _safe_get(wt_opt, "fixedse.structural_frequencies")
    stress = _safe_get(wt_opt, "fixedse.post_monopile_tower.constr_stress", reduce="max")
    if not np.isfinite(stress):
        stress = _safe_get(wt_opt, "fixedse.post.constr_stress", reduce="max")
    buck_g = _safe_get(wt_opt, "fixedse.post_monopile_tower.constr_global_buckling", reduce="max")
    if not np.isfinite(buck_g):
        buck_g = _safe_get(wt_opt, "fixedse.post.constr_global_buckling", reduce="max")
    buck_s = _safe_get(wt_opt, "fixedse.post_monopile_tower.constr_shell_buckling", reduce="max")
    if not np.isfinite(buck_s):
        buck_s = _safe_get(wt_opt, "fixedse.post.constr_shell_buckling", reduce="max")

    em = exit_mode if exit_mode is not None else get_opt_exit_mode(wt_opt)
    print(
        f"  [{label}] exit_mode={em}  D={d:.4f} m  t={t*1e3:.2f} mm  "
        f"mass={mass:.1f} kg  f1={f1:.4f} Hz"
    )
    print(
        f"  [{label}] max util: stress={stress:.3f}  global_buck={buck_g:.3f}  "
        f"shell_buck={buck_s:.3f}"
    )


def resolve_foundations(mode: str) -> list[str]:
    mode = mode.lower().strip()
    if mode == "both":
        return ["jacket", "monopile"]
    if mode in ("jacket", "monopile"):
        return [mode]
    raise ValueError(f"Invalid --foundation '{mode}'. Use jacket, monopile, or both.")


def resolve_jacket_leg_counts(scenario: dict, jacket_legs_mode: str, cdr) -> list[int]:
    """
    Leg counts to run for one scenario row.

    Per-row ``n_legs`` in the site CSV overrides ``--jacket-legs`` when set to 3 or 4.
    """
    if not _is_blank(scenario.get("n_legs", "")):
        return [cdr.validate_jacket_n_legs(int(_to_float(scenario["n_legs"])))]
    mode = jacket_legs_mode.lower().strip()
    if mode == "both":
        return [3, 4]
    return [cdr.validate_jacket_n_legs(int(mode))]


def set_jacket_n_legs(wt: dict, n_legs: int) -> dict:
    """Set jacket leg count on a geometry dict (deep copy)."""
    wt = copy.deepcopy(wt)
    wt.setdefault("components", {}).setdefault("jacket", {})["n_legs"] = int(n_legs)
    return wt


def _jacket_warmstart_key(scenario: dict, n_legs: int) -> tuple:
    """Warm-start cache key for jacket (site/turbine fields + leg count)."""
    return _scenario_key_for_warmstart(scenario) + (int(n_legs),)


_D_LO = 3.87
_D_HI = 16.0
_T_LO = 5e-2
_T_HI = 2e-1

# Jacket DV clamps (match analysis_options_jacket.yaml)
_J_RHEAD_LO, _J_RHEAD_HI = 2.0, 20.0
_J_RATIO_LO, _J_RATIO_HI = 1.1, 5.0
_J_LEG_D_LO, _J_LEG_D_HI = 0.9, 5.0
_J_LEG_T_LO, _J_LEG_T_HI = 0.03, 0.1
_J_BRACE_D_LO, _J_BRACE_D_HI = 0.5, 2.0
_J_BRACE_T_LO, _J_BRACE_T_HI = 0.03, 0.1  # keep t < 0.5 * brace_D_lo
_J_T_OVER_D_MAX = 0.45  # Tube Area → 0 when t >= D
_J_D_TO_T_MAX = 100.0
_J_ENFORCE_D_TO_T = True  # match analysis_options_jacket.yaml d_to_t upper=100
_J_FLARE_LO, _J_FLARE_HI = 0.1, 0.15  # (r_foot - r_head) / height


def repair_feasible_monopile_start(
    run_wisdem,
    wt_geom: dict,
    modeling: dict,
    analysis: dict,
    modeling_out: str,
    tag: str,
    cdr,
    f_lo: float,
    f_hi: float,
    max_passes: int = 8,
) -> dict:
    """Nudge D/t with run_only analyses until strength util and f1 are roughly feasible."""
    wt = copy.deepcopy(wt_geom)
    d_floor = monopile_diameter_floor(wt)
    analysis_ro = analysis_refresh_options(analysis)
    analysis_ro.setdefault("general", {})
    analysis_ro["general"]["folder_output"] = os.path.join(TMP_DIR, f"outputs_{tag}_preflight")
    analysis_ro["general"]["fname_output"] = "refturb_output"
    analysis_path = os.path.join(TMP_DIR, f"analysis_{tag}_preflight.yaml")
    dump_yaml(analysis_ro, analysis_path)

    for i in range(max_passes):
        d, t = (
            float(wt["components"]["monopile"]["outer_shape"]["outer_diameter"]["values"][0]),
            float(wt["components"]["monopile"]["structure"]["layers"][0]["thickness"]["values"][0]),
        )
        d = max(d, d_floor)
        wt = cdr.apply_wt_geometry_rules(
            seed_monopile_diameter_thickness(wt, d, t),
            modeling,
            "monopile",
        )
        geom_path = os.path.join(TMP_DIR, f"geometry_{tag}_preflight_{i}.yaml")
        dump_yaml(wt, geom_path)
        wt_opt, _, _ = run_wisdem(geom_path, modeling_out, analysis_path, run_only=True)
        util = max_strength_utilization(wt_opt)
        f1 = _safe_get(wt_opt, "fixedse.structural_frequencies")
        print(f"  [preflight {i}] D={d:.3f} t={t*1e3:.2f} mm  util={util:.3f}  f1={f1:.4f} Hz")
        if design_is_warmstart_safe(wt_opt, f_lo, f_hi):
            print("  Preflight: start is feasible enough for SLSQP")
            return wt

        d_new, t_new = d, t
        if np.isfinite(util) and util > 1.0:
            t_new = min(_T_HI, max(_T_LO, t * util * 1.05))
        if np.isfinite(f1) and f1 > f_hi:
            d_new = max(d_floor, d - 0.25)
            if not (np.isfinite(util) and util > 1.0):
                t_new = max(_T_LO, min(t_new, t * 0.95))
        if np.isfinite(f1) and f1 < f_lo:
            d_new = min(_D_HI, d + 0.25)
            t_new = min(_T_HI, max(t_new, t * 1.05))

        if abs(d_new - d) < 1e-9 and abs(t_new - t) < 1e-12:
            print("  Preflight: bounds hit; continuing with best-effort start")
            return wt
        wt = seed_monopile_diameter_thickness(wt, d_new, t_new)

    print("  Preflight: max passes reached; continuing with best-effort start")
    return wt


def run_monopile_two_stage(
    run_wisdem,
    wt_geom: dict,
    modeling: dict,
    analysis: dict,
    modeling_out: str,
    tag: str,
    cdr,
    f_lo: float,
    f_hi: float,
) -> tuple:
    """
    Preflight feasible seed -> Stage 1 (full, shorter) -> Stage 2 polish ->
    embedment refresh. Returns (wt_opt, modeling_options, opt_exit_mode, d, t).
    """
    height_opt = monopile_height_dv_enabled(analysis)
    print("  Preflight feasible-start repair …")
    wt_geom = repair_feasible_monopile_start(
        run_wisdem, wt_geom, modeling, analysis, modeling_out, tag, cdr, f_lo, f_hi
    )

    analysis_s1 = analysis_stage1_strength(analysis)
    analysis_s1.setdefault("general", {})
    analysis_s1["general"]["folder_output"] = os.path.join(TMP_DIR, f"outputs_{tag}_stage1")
    analysis_s1["general"]["fname_output"] = "refturb_output"
    analysis_s1_path = os.path.join(TMP_DIR, f"analysis_{tag}_stage1.yaml")
    wt_s1_path = os.path.join(TMP_DIR, f"geometry_{tag}_stage1.yaml")
    dump_yaml(analysis_s1, analysis_s1_path)
    dump_yaml(wt_geom, wt_s1_path)

    print("  Stage 1 (full constraints, shorter budget) …")
    wt_opt, modeling_options, _opt = run_wisdem(wt_s1_path, modeling_out, analysis_s1_path)
    exit1 = get_opt_exit_mode(wt_opt)
    print_monopile_diagnostics(wt_opt, "stage1", exit_mode=exit1)

    d1, t1 = extract_monopile_dt(wt_opt)
    if height_opt:
        wt_after_s1 = write_optimised_monopile_geometry(
            wt_geom, wt_opt, modeling, cdr, use_opt_embedment=True
        )
    else:
        wt_after_s1 = seed_monopile_diameter_thickness(wt_geom, d1, t1)
        wt_after_s1 = cdr.apply_wt_geometry_rules(wt_after_s1, modeling, "monopile")

    analysis_s2 = analysis_stage2_full(analysis)
    analysis_s2.setdefault("general", {})
    analysis_s2["general"]["folder_output"] = os.path.join(TMP_DIR, f"outputs_{tag}_stage2")
    analysis_s2["general"]["fname_output"] = "refturb_output"
    analysis_s2_path = os.path.join(TMP_DIR, f"analysis_{tag}_stage2.yaml")
    wt_s2_path = os.path.join(TMP_DIR, f"geometry_{tag}_stage2.yaml")
    dump_yaml(analysis_s2, analysis_s2_path)
    dump_yaml(wt_after_s1, wt_s2_path)

    print("  Stage 2 (full constraints, polish) …")
    wt_opt, modeling_options, _opt = run_wisdem(wt_s2_path, modeling_out, analysis_s2_path)
    exit2 = get_opt_exit_mode(wt_opt)
    print_monopile_diagnostics(wt_opt, "stage2", exit_mode=exit2)

    d2, t2 = extract_monopile_dt(wt_opt)
    d2 = max(d2, monopile_diameter_floor(wt_geom))
    t2 = _clip(t2, monopile_thickness_floor(d2), _T_HI)

    wt_final = write_optimised_monopile_geometry(
        wt_after_s1, wt_opt, modeling, cdr, use_opt_embedment=height_opt
    )
    wt_refresh = os.path.join(TMP_DIR, f"geometry_{tag}_embedment.yaml")
    analysis_refresh = os.path.join(TMP_DIR, f"analysis_{tag}_refresh.yaml")
    dump_yaml(wt_final, wt_refresh)
    dump_yaml(analysis_refresh_options(analysis_s2), analysis_refresh)
    if height_opt:
        print("  Refreshing analysis with optimised height/embedment …")
    else:
        print("  Refreshing analysis with final embedment …")
    wt_opt, modeling_options, _opt = run_wisdem(
        wt_refresh, modeling_out, analysis_refresh, run_only=True
    )
    print_monopile_diagnostics(wt_opt, "refresh", exit_mode=exit2)

    return wt_opt, modeling_options, exit2, d2, t2


def print_jacket_diagnostics(
    wt_opt,
    label: str,
    f_lo: float,
    f_hi: float,
    exit_mode=None,
    modeling_options: dict | None = None,
    member_diagnostics: bool = False,
    stress_diagnostics: bool = False,
    buckling_diagnostics: bool = False,
    buckling_util_threshold: float = 1.0,
    stress_util_threshold: float = 1.0,
) -> None:
    """Log jacket exit mode, geometry, mass, f1 and peak utilizations."""
    r_head = _safe_get(wt_opt, "jacket.r_head", units="m")
    r_foot = _safe_get(wt_opt, "fixedse.r_foot", units="m")
    ratio = _safe_get(wt_opt, "jacket.foot_head_ratio")
    leg_d = _safe_get(wt_opt, "jacket.leg_diameter", units="m")
    leg_t = _safe_get(wt_opt, "jacket.leg_thickness", units="m")
    jmass = _safe_get(wt_opt, "fixedse.jacket_mass", units="kg")
    f1 = _safe_get(wt_opt, "fixedse.structural_frequencies")
    stress = _safe_get(wt_opt, "fixedse.constr_stress", reduce="max")
    buck_s = _safe_get(wt_opt, "fixedse.constr_shell_buckling", reduce="max")
    buck_g = _safe_get(wt_opt, "fixedse.constr_global_buckling", reduce="max")
    em = exit_mode if exit_mode is not None else get_opt_exit_mode(wt_opt)
    print(
        f"  [{label}] exit_mode={em}  r_head={r_head:.3f} m  r_foot={r_foot:.3f} m  "
        f"ratio={ratio:.3f}  leg_D={leg_d:.3f} m  leg_t={leg_t*1e3:.1f} mm  "
        f"mass={jmass:.1f} kg  f1={f1:.4f} Hz  band=[{f_lo:.4f}, {f_hi:.4f}]"
    )
    print(
        f"  [{label}] max util: stress={stress:.3f}  global_buck={buck_g:.3f}  "
        f"shell_buck={buck_s:.3f}"
    )
    show_stress = stress_diagnostics or member_diagnostics
    show_buckling = buckling_diagnostics or member_diagnostics
    if (show_stress or show_buckling) and modeling_options is not None:
        printed = False
        if show_buckling and np.isfinite(buck_g) and buck_g >= buckling_util_threshold:
            print_jacket_member_buckling_diagnostics(
                wt_opt,
                modeling_options,
                label=label,
                util_threshold=buckling_util_threshold,
            )
            printed = True
        if show_stress and np.isfinite(stress) and stress >= stress_util_threshold:
            print_jacket_member_stress_diagnostics(
                wt_opt,
                modeling_options,
                label=label,
                util_threshold=stress_util_threshold,
            )
            printed = True
        if not printed:
            parts = []
            if show_buckling:
                parts.append(f"global>={buckling_util_threshold:.3f} (peak={buck_g:.3f})")
            if show_stress:
                parts.append(f"stress>={stress_util_threshold:.3f} (peak={stress:.3f})")
            print(
                f"  [{label}] member diagnostics skipped "
                f"(no {' or '.join(parts)} member/DLC pairs)"
            )


def _get_jacket_member_arrays(wt_opt, name: str, units=None) -> np.ndarray | None:
    """Read a per-element jacket array from fixedse (or post alias)."""
    for prefix in ("fixedse.", "fixedse.post."):
        try:
            if units is None:
                return np.asarray(wt_opt.get_val(prefix + name), dtype=float)
            return np.asarray(wt_opt.get_val(prefix + name, units=units), dtype=float)
        except Exception:
            continue
    return None


def _jacket_dlc_names(n_dlc: int) -> list[str]:
    names = [
        "DLC1_rated_GustETM",
        "DLC2_parked_extreme_V50",
    ]
    while len(names) < n_dlc:
        names.append(f"DLC{len(names) + 1}")
    return names


def _align_jacket_member_2d(arr, n_elem: int, n_dlc: int) -> np.ndarray | None:
    if arr is None:
        return None
    arr = np.atleast_2d(np.asarray(arr, dtype=float))
    if arr.shape[0] == n_dlc and arr.shape[1] == n_elem:
        arr = arr.T
    if arr.shape[0] > n_elem:
        arr = arr[:n_elem]
    if arr.shape[1] > n_dlc:
        arr = arr[:, :n_dlc]
    return arr


def _jacket_stress_gamma(modeling_options: dict) -> float:
    fb = modeling_options.get("WISDEM", {}).get("FixedBottomSE", {})
    return float(fb.get("gamma_f", 1.0)) * float(fb.get("gamma_m", 1.0)) * float(fb.get("gamma_n", 1.0))


def classify_jacket_stress_governing(
    axial_stress: float,
    hoop_stress: float,
    shear_stress: float,
    sigma_y: float,
    gamma: float,
    axial_from_force: float,
    bending_stress: float,
) -> tuple[str, dict[str, float]]:
    """
    Classify which stress mechanism drives von Mises utilization for one member/DLC.

    Replicates ``JacketPost`` / ``TubevonMisesStressUtilization`` term structure.
    """
    ax = float(axial_stress)
    hoop = float(hoop_stress)
    sh = float(shear_stress)
    sigy = float(sigma_y)
    if not (np.isfinite(ax) and np.isfinite(hoop) and np.isfinite(sh) and sigy > 0.0):
        return "unknown", {}

    term_mean = ((ax + hoop) / 2.0) ** 2
    term_diff = ((ax - hoop) / 2.0) ** 2
    term_shear = sh**2
    vm = float(np.sqrt(term_mean + 3.0 * (term_diff + term_shear)))
    vm2 = vm**2 if vm > 0.0 else 0.0

    util_vm = gamma * vm / sigy
    util_ax = gamma * abs(ax) / sigy
    util_hoop = gamma * abs(hoop) / sigy
    util_shear = gamma * np.sqrt(3.0) * abs(sh) / sigy

    detail = {
        "von_mises_MPa": vm / 1e6,
        "axial_MPa": ax / 1e6,
        "hoop_MPa": hoop / 1e6,
        "shear_MPa": sh / 1e6,
        "axial_force_MPa": float(axial_from_force) / 1e6,
        "bending_MPa": float(bending_stress) / 1e6,
        "util_vm": util_vm,
        "util_axial_component": util_ax,
        "util_hoop_component": util_hoop,
        "util_shear_component": util_shear,
        "vm_frac_mean_stress": float(term_mean / vm2) if vm2 > 0.0 else 0.0,
        "vm_frac_axial_hoop_diff": float(3.0 * term_diff / vm2) if vm2 > 0.0 else 0.0,
        "vm_frac_shear": float(3.0 * term_shear / vm2) if vm2 > 0.0 else 0.0,
    }

    if util_shear >= max(util_ax, util_hoop) and detail["vm_frac_shear"] >= 0.35:
        governing = "shear / torsion (von Mises)"
    elif util_hoop >= util_ax and util_hoop >= util_shear and abs(hoop) > 0.15 * max(abs(ax), 1.0):
        governing = "hydrodynamic hoop pressure"
    elif abs(bending_stress) > abs(axial_from_force):
        governing = "bending-dominated von Mises"
    elif axial_from_force < 0.0:
        governing = "compressive axial (+ interaction)"
    else:
        governing = "tensile axial (+ interaction)"

    return governing, detail


def print_jacket_member_stress_diagnostics(
    wt_opt,
    modeling_options: dict,
    label: str = "",
    util_threshold: float = 1.0,
    top_n: int = 5,
) -> None:
    """
    Report which jacket members drive von Mises stress utilization and why.

    Uses the same stress model as ``JacketPost`` (Frame3DD loads + DNV partial factors).
    """
    from wisdem.fixed_bottomse.jacket import jacket_element_metadata

    fb = modeling_options.get("WISDEM", {}).get("FixedBottomSE", {})
    n_legs = int(fb.get("n_legs", 4))
    n_bays = int(fb.get("n_bays", 3))
    mud_brace = bool(fb.get("mud_brace", True))
    n_dlc = int(modeling_options.get("WISDEM", {}).get("n_dlc", 1))
    gamma = _jacket_stress_gamma(modeling_options)

    meta = jacket_element_metadata(n_legs, n_bays, mud_brace)
    n_elem = len(meta)
    dlc_names = _jacket_dlc_names(n_dlc)

    stress = _align_jacket_member_2d(_get_jacket_member_arrays(wt_opt, "constr_stress"), n_elem, n_dlc)
    if stress is None:
        print(f"  [{label}] member diagnostics: constr_stress not available")
        return

    elem_L = _get_jacket_member_arrays(wt_opt, "jacket_elem_L", units="m")
    elem_D = _get_jacket_member_arrays(wt_opt, "jacket_elem_D", units="m")
    elem_t = _get_jacket_member_arrays(wt_opt, "jacket_elem_t", units="m")
    elem_A = _get_jacket_member_arrays(wt_opt, "jacket_elem_A", units="m**2")
    elem_Asx = _get_jacket_member_arrays(wt_opt, "jacket_elem_Asx", units="m**2")
    elem_Iyy = _get_jacket_member_arrays(wt_opt, "jacket_elem_Iyy", units="kg*m**2")
    elem_J0 = _get_jacket_member_arrays(wt_opt, "jacket_elem_J0", units="kg*m**2")
    elem_sigy = _get_jacket_member_arrays(wt_opt, "jacket_elem_sigma_y", units="Pa")
    qdyn = _align_jacket_member_2d(_get_jacket_member_arrays(wt_opt, "jacket_elem_qdyn", units="Pa"), n_elem, n_dlc)
    fz = _align_jacket_member_2d(_get_jacket_member_arrays(wt_opt, "jacket_Fz", units="N"), n_elem, n_dlc)
    vx = _align_jacket_member_2d(_get_jacket_member_arrays(wt_opt, "jacket_Vx", units="N"), n_elem, n_dlc)
    vy = _align_jacket_member_2d(_get_jacket_member_arrays(wt_opt, "jacket_Vy", units="N"), n_elem, n_dlc)
    mxx = _align_jacket_member_2d(_get_jacket_member_arrays(wt_opt, "jacket_Mxx", units="N*m"), n_elem, n_dlc)
    myy = _align_jacket_member_2d(_get_jacket_member_arrays(wt_opt, "jacket_Myy", units="N*m"), n_elem, n_dlc)
    mzz = _align_jacket_member_2d(_get_jacket_member_arrays(wt_opt, "jacket_Mzz", units="N*m"), n_elem, n_dlc)

    if elem_L is not None:
        elem_L = np.atleast_1d(elem_L).flatten()[:n_elem]
    if elem_D is not None:
        elem_D = np.atleast_1d(elem_D).flatten()[:n_elem]
    if elem_t is not None:
        elem_t = np.atleast_1d(elem_t).flatten()[:n_elem]
    if elem_A is not None:
        elem_A = np.atleast_1d(elem_A).flatten()[:n_elem]
    if elem_Asx is not None:
        elem_Asx = np.atleast_1d(elem_Asx).flatten()[:n_elem]
    if elem_Iyy is not None:
        elem_Iyy = np.atleast_1d(elem_Iyy).flatten()[:n_elem]
    if elem_J0 is not None:
        elem_J0 = np.atleast_1d(elem_J0).flatten()[:n_elem]
    if elem_sigy is not None:
        elem_sigy = np.atleast_1d(elem_sigy).flatten()[:n_elem]

    pairs = []
    for i in range(min(n_elem, stress.shape[0])):
        for k in range(min(n_dlc, stress.shape[1])):
            util_st = float(stress[i, k])
            if np.isfinite(util_st) and util_st >= util_threshold:
                pairs.append((util_st, i, k))
    pairs.sort(reverse=True)

    tag = f"[{label}] " if label else ""
    print(f"\n  {tag}=== Jacket member stress diagnostics (von Mises util >= {util_threshold:.2f}) ===")
    print(
        f"  {tag}Check model: DNV partial factors γ_f·γ_m·γ_n={gamma:.3f} on von Mises stress "
        f"(σ_vm = sqrt(((σ_ax+σ_h)/2)² + 3·(((σ_ax-σ_h)/2)² + τ²)); util = γ·σ_vm/σ_y)."
    )
    print(
        f"  {tag}Stress components match JacketPost: "
        "σ_ax = Fz/A + M·r/Iyy; τ = |Mzz|/(J·r) + V/Asx; σ_h = -q_dyn·r/t."
    )

    if not pairs:
        max_st = float(np.nanmax(stress)) if stress.size else float("nan")
        print(f"  {tag}No member/DLC pairs above threshold (peak stress util={max_st:.3f}).")
        return

    show = pairs[:top_n]
    print(f"  {tag}Top {len(show)} critical member(s) by von Mises stress utilization:\n")
    for rank, (util_st, i, k) in enumerate(show, start=1):
        m = meta[i]
        L = float(elem_L[i]) if elem_L is not None else float("nan")
        D = float(elem_D[i]) if elem_D is not None else float("nan")
        t = float(elem_t[i]) if elem_t is not None else float("nan")
        sigy = float(elem_sigy[i]) if elem_sigy is not None else float("nan")
        r = 0.5 * D if np.isfinite(D) else float("nan")

        axial = float(fz[i, k]) if fz is not None else float("nan")
        if mxx is not None and myy is not None:
            moment = float(np.hypot(mxx[i, k], myy[i, k]))
        else:
            moment = float("nan")
        if vx is not None and vy is not None:
            shear_force = float(np.hypot(vx[i, k], vy[i, k]))
        else:
            shear_force = float("nan")
        torsion = float(mzz[i, k]) if mzz is not None else float("nan")
        q = float(qdyn[i, k]) if qdyn is not None else float("nan")

        Az = float(elem_A[i]) if elem_A is not None else float("nan")
        Asx = float(elem_Asx[i]) if elem_Asx is not None else float("nan")
        Iyy = float(elem_Iyy[i]) if elem_Iyy is not None else float("nan")
        Jz = float(elem_J0[i]) if elem_J0 is not None else float("nan")

        axial_force_stress = float("nan")
        bending_stress = float("nan")
        axial_stress = float("nan")
        hoop_stress = float("nan")
        shear_stress = float("nan")
        if (
            np.isfinite(Az)
            and Az > 0.0
            and np.isfinite(Iyy)
            and Iyy > 0.0
            and np.isfinite(r)
            and np.isfinite(axial)
            and np.isfinite(moment)
        ):
            axial_force_stress = axial / Az
            bending_stress = moment * (r / Iyy)
            axial_stress = axial_force_stress + bending_stress
        if np.isfinite(q) and np.isfinite(r) and np.isfinite(t) and t > 0.0:
            hoop_stress = -q * ((r - 0.5 * t) / t)
        if all(np.isfinite(x) and x > 0.0 for x in (Jz, r, Asx)):
            shear_stress = abs(torsion) / (Jz * r) + (
                shear_force / Asx if np.isfinite(shear_force) else 0.0
            )

        governing, detail = classify_jacket_stress_governing(
            axial_stress,
            hoop_stress,
            shear_stress,
            sigy,
            gamma,
            axial_force_stress,
            bending_stress,
        )

        dlc_label = dlc_names[k] if k < len(dlc_names) else f"DLC{k + 1}"
        extreme_note = (
            "parked extreme (V50 thrust + RNA weight)"
            if "parked" in dlc_label or k == 1
            else "rated operational (GustETM RNA + rated wind)"
        )
        slenderness_ld = L / D if np.isfinite(L) and np.isfinite(D) and D > 0 else float("nan")

        print(f"  {tag}#{rank}  {m['label']}  (elem {i}, {dlc_label})")
        print(f"  {tag}    Member type: {m['member_type']}")
        print(
            f"  {tag}    Geometry: L={L:.3f} m, D={D:.3f} m, t={t*1e3:.1f} mm, "
            f"L/D={slenderness_ld:.1f}, σ_y={sigy/1e6:.0f} MPa"
        )
        print(
            f"  {tag}    Loads: Fz={axial/1e3:.1f} kN axial, |M|={moment/1e3:.1f} kN·m, "
            f"|V|={shear_force/1e3:.1f} kN, |Mzz|={abs(torsion)/1e3:.1f} kN·m, q_dyn={q/1e3:.1f} kPa"
        )
        print(
            f"  {tag}    Stress components: σ_ax={detail.get('axial_MPa', float('nan')):.2f} MPa "
            f"(F/A={detail.get('axial_force_MPa', float('nan')):.2f}, "
            f"M·r/I={detail.get('bending_MPa', float('nan')):.2f}), "
            f"σ_h={detail.get('hoop_MPa', float('nan')):.2f} MPa, "
            f"τ={detail.get('shear_MPa', float('nan')):.2f} MPa"
        )
        print(
            f"  {tag}    Component util (γ·|σ|/σ_y): axial={detail.get('util_axial_component', float('nan')):.3f}  "
            f"hoop={detail.get('util_hoop_component', float('nan')):.3f}  "
            f"shear≈{detail.get('util_shear_component', float('nan')):.3f}  "
            f"von Mises={util_st:.3f}"
        )
        print(
            f"  {tag}    von Mises term fractions: mean={detail.get('vm_frac_mean_stress', 0.0):.2f}  "
            f"axial–hoop diff={detail.get('vm_frac_axial_hoop_diff', 0.0):.2f}  "
            f"shear={detail.get('vm_frac_shear', 0.0):.2f}  → governing: {governing}"
        )
        print(f"  {tag}    Load case: {dlc_label} ({extreme_note})")
        print("")


def print_jacket_member_buckling_diagnostics(
    wt_opt,
    modeling_options: dict,
    label: str = "",
    util_threshold: float = 1.0,
    top_n: int = 5,
) -> None:
    """
    Report which jacket members drive global / shell / stress utilization.

    Uses DNVGL-RP-C202 checks in ``JacketPost`` (k=1.0 legs, k=0.8 braces per
    element; shell buckling on unstiffened cylinder segments).
    """
    from wisdem.fixed_bottomse.jacket import (
        JACKET_GLOBAL_BUCKLING_K_BRACE,
        JACKET_GLOBAL_BUCKLING_K_LEG,
        jacket_element_metadata,
        jacket_global_buckling_k_factors,
    )

    fb = modeling_options.get("WISDEM", {}).get("FixedBottomSE", {})
    n_legs = int(fb.get("n_legs", 4))
    n_bays = int(fb.get("n_bays", 3))
    mud_brace = bool(fb.get("mud_brace", True))
    n_dlc = int(modeling_options.get("WISDEM", {}).get("n_dlc", 1))

    meta = jacket_element_metadata(n_legs, n_bays, mud_brace)
    k_factors = jacket_global_buckling_k_factors(n_legs, n_bays, mud_brace)
    n_elem = len(meta)

    buck_g = _get_jacket_member_arrays(wt_opt, "constr_global_buckling")
    buck_s = _get_jacket_member_arrays(wt_opt, "constr_shell_buckling")
    stress = _get_jacket_member_arrays(wt_opt, "constr_stress")
    elem_L = _get_jacket_member_arrays(wt_opt, "jacket_elem_L", units="m")
    elem_D = _get_jacket_member_arrays(wt_opt, "jacket_elem_D", units="m")
    elem_t = _get_jacket_member_arrays(wt_opt, "jacket_elem_t", units="m")
    fz = _get_jacket_member_arrays(wt_opt, "jacket_Fz", units="N")
    mxx = _get_jacket_member_arrays(wt_opt, "jacket_Mxx", units="N*m")
    myy = _get_jacket_member_arrays(wt_opt, "jacket_Myy", units="N*m")

    if buck_g is None:
        print(f"  [{label}] member diagnostics: constr_global_buckling not available")
        return

    buck_g = np.atleast_2d(buck_g)
    if buck_g.shape[0] == n_dlc and buck_g.shape[1] == n_elem:
        buck_g = buck_g.T
    if buck_g.shape[0] > n_elem:
        buck_g = buck_g[:n_elem]
    if buck_g.shape[1] > n_dlc:
        buck_g = buck_g[:, :n_dlc]

    dlc_names = _jacket_dlc_names(n_dlc)

    buck_s = _align_jacket_member_2d(buck_s, n_elem, n_dlc)
    stress = _align_jacket_member_2d(stress, n_elem, n_dlc)
    fz = _align_jacket_member_2d(fz, n_elem, n_dlc)
    mxx = _align_jacket_member_2d(mxx, n_elem, n_dlc)
    myy = _align_jacket_member_2d(myy, n_elem, n_dlc)
    # Ghost members are the last n_legs elements; structural checks omit them.
    if elem_L is not None:
        elem_L = np.atleast_1d(elem_L).flatten()[:n_elem]
    if elem_D is not None:
        elem_D = np.atleast_1d(elem_D).flatten()[:n_elem]
    if elem_t is not None:
        elem_t = np.atleast_1d(elem_t).flatten()[:n_elem]

    # Rank (element, DLC) pairs by global buckling utilization
    pairs = []
    for i in range(min(n_elem, buck_g.shape[0])):
        for k in range(min(n_dlc, buck_g.shape[1])):
            util_g = float(buck_g[i, k])
            if np.isfinite(util_g) and util_g >= util_threshold:
                util_s = float(buck_s[i, k]) if buck_s is not None else float("nan")
                util_st = float(stress[i, k]) if stress is not None else float("nan")
                pairs.append((util_g, i, k, util_s, util_st))
    pairs.sort(reverse=True)

    tag = f"[{label}] " if label else ""
    print(f"\n  {tag}=== Jacket member buckling diagnostics (global util >= {util_threshold:.2f}) ===")
    print(
        f"  {tag}Check model: DNVGL-RP-C202 via JacketPost — "
        "global = column buckling + axial/bending interaction; "
        "shell = local cylinder shell buckling (separate constraint)."
    )
    print(
        f"  {tag}Boundary/end conditions (global): k={JACKET_GLOBAL_BUCKLING_K_LEG:.1f} legs, "
        f"k={JACKET_GLOBAL_BUCKLING_K_BRACE:.1f} braces (Leff = k·L_element per Frame3DD member)."
    )

    if not pairs:
        max_g = float(np.nanmax(buck_g)) if buck_g.size else float("nan")
        print(f"  {tag}No member/DLC pairs above threshold (peak global util={max_g:.3f}).")
        return

    show = pairs[:top_n]
    print(f"  {tag}Top {len(show)} critical member(s) by global buckling utilization:\n")
    for rank, (util_g, i, k, util_s, util_st) in enumerate(show, start=1):
        m = meta[i]
        L = float(elem_L[i]) if elem_L is not None else float("nan")
        D = float(elem_D[i]) if elem_D is not None else float("nan")
        t = float(elem_t[i]) if elem_t is not None else float("nan")
        axial = float(fz[i, k]) if fz is not None else float("nan")
        if mxx is not None and myy is not None:
            moment = float(np.hypot(mxx[i, k], myy[i, k]))
        else:
            moment = float("nan")

        compression = axial < 0.0 if np.isfinite(axial) else None
        k_factor = float(k_factors[i])
        leff = k_factor * L if np.isfinite(L) else float("nan")
        slenderness_ld = L / D if np.isfinite(L) and np.isfinite(D) and D > 0 else float("nan")

        if np.isfinite(util_g) and np.isfinite(util_s) and np.isfinite(util_st):
            if util_g >= util_s and util_g >= util_st:
                governing = "global column buckling (+ axial/bending interaction)"
            elif util_s >= util_g and util_s >= util_st:
                governing = "local shell buckling"
            else:
                governing = "von Mises stress"
        else:
            governing = "global buckling (reported constraint)"

        dlc_label = dlc_names[k] if k < len(dlc_names) else f"DLC{k + 1}"
        extreme_note = (
            "parked extreme (V50 thrust + RNA weight)"
            if "parked" in dlc_label or k == 1
            else "rated operational (GustETM RNA + rated wind)"
        )

        print(f"  {tag}#{rank}  {m['label']}  (elem {i}, {dlc_label})")
        print(f"  {tag}    Member type: {m['member_type']}")
        print(
            f"  {tag}    Compression-dominated: "
            f"{'yes' if compression else 'no' if compression is not None else 'unknown'} "
            f"(Fz={axial/1e3:.1f} kN)"
        )
        print(
            f"  {tag}    Geometry: L={L:.3f} m, D={D:.3f} m, t={t*1e3:.1f} mm, "
            f"L/D={slenderness_ld:.1f}"
        )
        print(
            f"  {tag}    Effective buckling length: Leff=k·L={leff:.3f} m "
            f"(k={k_factor:.1f}, {m['member_type']})"
        )
        print(
            f"  {tag}    Loads: |Fz|={abs(axial)/1e3:.1f} kN axial, "
            f"|M|={moment/1e3:.1f} kN·m bending"
        )
        print(
            f"  {tag}    Utilization: global={util_g:.3f}  shell={util_s:.3f}  "
            f"stress={util_st:.3f}  → governing: {governing}"
        )
        print(f"  {tag}    Load case: {dlc_label} ({extreme_note})")
        if slenderness_ld > 40:
            print(f"  {tag}    Note: L/D={slenderness_ld:.1f} is slender for a tubular brace/leg.")
        print("")


def _apply_jacket_frequency_nudge(
    wt: dict, f1: float, f_lo: float, f_hi: float
) -> tuple[dict, bool]:
    """
    One-step geometry tweak to move jacket f1 toward the soft-stiff band.

    Returns (updated_wt, changed).
    """
    if not (np.isfinite(f1) and np.isfinite(f_lo) and np.isfinite(f_hi)):
        return wt, False
    if f_lo <= f1 <= f_hi:
        return wt, False

    jacket = wt["components"]["jacket"]
    r_head = float(jacket["r_head"])
    ratio = float(jacket["r_foot"]) / r_head if r_head > 0 else 1.5
    leg_d = float(jacket["leg_diameter"])
    brace_d0 = float(jacket["brace_diameters"][0])
    height = float(jacket.get("height", 40.0)) or 40.0

    r_new, ratio_new = r_head, ratio
    leg_d_new = leg_d
    brace_d_new = brace_d0

    if f1 < f_lo:
        # Too soft: widen base and fatten members (keep within flare band)
        ratio_new = _clip(ratio + 0.15, _J_RATIO_LO, _J_RATIO_HI)
        flare_new = (r_new * (ratio_new - 1.0)) / height if height > 0 else 0.0
        if flare_new > _J_FLARE_HI:
            ratio_new = _clip(
                1.0 + _J_FLARE_HI * height / max(r_new, 1e-6), _J_RATIO_LO, _J_RATIO_HI
            )
        r_new = _clip(r_head + 0.25, _J_RHEAD_LO, _J_RHEAD_HI)
        leg_d_new = _clip(leg_d + 0.2, _J_LEG_D_LO, _J_LEG_D_HI)
        brace_d_new = _clip(brace_d0 + 0.1, _J_BRACE_D_LO, _J_BRACE_D_HI)
    elif f1 > f_hi:
        # Too stiff: trim flare / members (keep r_head >= D_tower)
        ratio_new = _clip(ratio - 0.1, _J_RATIO_LO, _J_RATIO_HI)
        flare_new = (r_head * (ratio_new - 1.0)) / height if height > 0 else 0.0
        if flare_new < _J_FLARE_LO:
            ratio_new = _clip(
                1.0 + _J_FLARE_LO * height / max(r_head, 1e-6), _J_RATIO_LO, _J_RATIO_HI
            )
        d_tow = tower_base_diameter_from_geom(wt)
        r_new = _clip(max(d_tow, r_head - 0.25), _J_RHEAD_LO, _J_RHEAD_HI)
        leg_d_new = _clip(leg_d * 0.95, _J_LEG_D_LO, _J_LEG_D_HI)
        brace_d_new = _clip(brace_d0 * 0.95, _J_BRACE_D_LO, _J_BRACE_D_HI)

    d_tow = tower_base_diameter_from_geom(wt)
    r_new = max(r_new, d_tow)
    leg_t = float(jacket["leg_thickness"])
    brace_t0 = float(jacket["brace_thicknesses"][0])
    unchanged = (
        abs(r_new - r_head) < 1e-9
        and abs(ratio_new - ratio) < 1e-12
        and abs(leg_d_new - leg_d) < 1e-9
        and abs(brace_d_new - brace_d0) < 1e-12
    )
    if unchanged:
        return wt, False

    wt_new = seed_jacket_geometry(
        wt,
        r_new,
        ratio_new,
        leg_d_new,
        leg_t,
        [brace_d_new],
        [brace_t0],
    )
    return wt_new, True


def repair_jacket_frequency_start(
    run_wisdem,
    wt_geom: dict,
    modeling: dict,
    analysis: dict,
    modeling_out: str,
    tag: str,
    f_lo: float,
    f_hi: float,
    max_passes: int = 6,
) -> dict:
    """Run-only nudges after Stage 1 so Stage 2 frequency SLSQP starts near the band."""
    wt = copy.deepcopy(wt_geom)
    analysis_ro = analysis_jacket_run_only(analysis)
    analysis_ro.setdefault("general", {})
    analysis_ro["general"]["folder_output"] = os.path.join(TMP_DIR, f"outputs_{tag}_preflight")
    analysis_ro["general"]["fname_output"] = "refturb_output"
    analysis_path = os.path.join(TMP_DIR, f"analysis_{tag}_preflight.yaml")
    dump_yaml(analysis_ro, analysis_path)

    for i in range(max_passes):
        geom_path = os.path.join(TMP_DIR, f"geometry_{tag}_preflight_{i}.yaml")
        dump_yaml(wt, geom_path)
        wt_opt, _, _ = run_wisdem(geom_path, modeling_out, analysis_path, run_only=True)
        f1 = _safe_get(wt_opt, "fixedse.structural_frequencies")
        print(f"  [freq repair {i}] f1={f1:.4f} Hz  band=[{f_lo:.4f}, {f_hi:.4f}]")
        if np.isfinite(f1) and f_lo <= f1 <= f_hi:
            print("  Frequency repair: f1 is within band for Stage 2")
            return wt
        wt_new, changed = _apply_jacket_frequency_nudge(wt, f1, f_lo, f_hi)
        if not changed:
            print("  Frequency repair: bounds hit; continuing with best-effort start")
            return wt
        wt = wt_new

    print("  Frequency repair: max passes reached; continuing with best-effort start")
    return wt


def repair_feasible_jacket_start(
    run_wisdem,
    wt_geom: dict,
    modeling: dict,
    analysis: dict,
    modeling_out: str,
    tag: str,
    f_lo: float,
    f_hi: float,
    max_passes: int = 8,
    member_diagnostics: bool = False,
    stress_diagnostics: bool = False,
    buckling_diagnostics: bool = False,
    buckling_util_threshold: float = 1.0,
    stress_util_threshold: float = 1.0,
) -> dict:
    """Nudge jacket DVs with run_only analyses until strength util looks roughly feasible."""
    wt = copy.deepcopy(wt_geom)
    analysis_ro = analysis_jacket_run_only(analysis)
    analysis_ro.setdefault("general", {})
    analysis_ro["general"]["folder_output"] = os.path.join(TMP_DIR, f"outputs_{tag}_preflight")
    analysis_ro["general"]["fname_output"] = "refturb_output"
    analysis_path = os.path.join(TMP_DIR, f"analysis_{tag}_preflight.yaml")
    dump_yaml(analysis_ro, analysis_path)

    for i in range(max_passes):
        jacket = wt["components"]["jacket"]
        r_head = float(jacket["r_head"])
        ratio = float(jacket["r_foot"]) / r_head if r_head > 0 else 1.5
        leg_d = float(jacket["leg_diameter"])
        leg_t = float(jacket["leg_thickness"])
        brace_d0 = float(jacket["brace_diameters"][0])
        brace_t0 = float(jacket["brace_thicknesses"][0])

        geom_path = os.path.join(TMP_DIR, f"geometry_{tag}_preflight_{i}.yaml")
        dump_yaml(wt, geom_path)
        wt_opt, _, _ = run_wisdem(geom_path, modeling_out, analysis_path, run_only=True)
        strength_util = max_jacket_strength_util(wt_opt)
        global_buck = max_jacket_global_buckling_util(wt_opt)
        util = max_strength_utilization(wt_opt)
        f1 = _safe_get(wt_opt, "fixedse.structural_frequencies")
        print(
            f"  [preflight {i}] r_head={r_head:.3f} ratio={ratio:.3f} "
            f"leg_D={leg_d:.3f} leg_t={leg_t*1e3:.1f} mm  "
            f"util={util:.3f} (stress/shell={strength_util:.3f}, global_buck={global_buck:.3f})  "
            f"f1={f1:.4f} Hz"
        )
        if jacket_preflight_is_ready(wt_opt, f_lo, f_hi, check_frequency=False):
            print("  Preflight: start is feasible enough for SLSQP")
            return wt

        show_stress = stress_diagnostics or member_diagnostics
        show_buckling = buckling_diagnostics or member_diagnostics
        global_buck_diag = max_jacket_global_buckling_util(wt_opt)
        strength_diag = max_jacket_strength_util(wt_opt)
        if show_buckling and np.isfinite(global_buck_diag) and global_buck_diag >= buckling_util_threshold:
            print_jacket_member_buckling_diagnostics(
                wt_opt,
                modeling,
                label=f"preflight {i}",
                util_threshold=buckling_util_threshold,
            )
        if show_stress and np.isfinite(strength_diag) and strength_diag >= stress_util_threshold:
            print_jacket_member_stress_diagnostics(
                wt_opt,
                modeling,
                label=f"preflight {i}",
                util_threshold=stress_util_threshold,
            )

        r_new, ratio_new = r_head, ratio
        leg_d_new, leg_t_new = leg_d, leg_t
        brace_d_new, brace_t_new = brace_d0, brace_t0

        # Keep batter in the practical flare band when enabled in analysis YAML
        height = float(jacket.get("height", 40.0)) or 40.0
        flare = (r_head * (ratio - 1.0)) / height if height > 0 else 0.0
        if flare < _J_FLARE_LO or flare > _J_FLARE_HI:
            flare_tgt = 0.5 * (_J_FLARE_LO + _J_FLARE_HI)
            ratio_new = _clip(1.0 + flare_tgt * height / max(r_head, 1e-6), _J_RATIO_LO, _J_RATIO_HI)

        scale = _jacket_preflight_scale(strength_util, global_buck)
        if scale > 1.0:
            leg_d_new = _clip(leg_d * scale, _J_LEG_D_LO, _J_LEG_D_HI)
            leg_t_new = _clamp_tube_wall(
                leg_d_new, _clip(leg_t * scale, _J_LEG_T_LO, _J_LEG_T_HI)
            )
            brace_d_new = _clip(brace_d0 * scale, _J_BRACE_D_LO, _J_BRACE_D_HI)
            brace_t_new = _clamp_tube_wall(
                brace_d_new, _clip(brace_t0 * scale, _J_BRACE_T_LO, _J_BRACE_T_HI)
            )
            if scale > 1.5 and np.isfinite(global_buck) and global_buck > 1.0:
                print(f"  [preflight {i}] aggressive scale={scale:.3f} for global_buck={global_buck:.3f}")

        d_tow = tower_base_diameter_from_geom(wt)
        r_new = max(r_new, d_tow)

        unchanged = (
            abs(r_new - r_head) < 1e-9
            and abs(ratio_new - ratio) < 1e-12
            and abs(leg_d_new - leg_d) < 1e-9
            and abs(leg_t_new - leg_t) < 1e-12
            and abs(brace_d_new - brace_d0) < 1e-9
            and abs(brace_t_new - brace_t0) < 1e-12
        )
        if unchanged:
            print("  Preflight: bounds hit; continuing with best-effort start")
            return wt

        wt = seed_jacket_geometry(
            wt, r_new, ratio_new, leg_d_new, leg_t_new, [brace_d_new], [brace_t_new]
        )

    print("  Preflight: max passes reached; continuing with best-effort start")
    return wt


def run_jacket_two_stage(
    run_wisdem,
    wt_geom: dict,
    modeling: dict,
    analysis: dict,
    modeling_out: str,
    tag: str,
    f_lo: float,
    f_hi: float,
    member_diagnostics: bool = False,
    stress_diagnostics: bool = False,
    buckling_diagnostics: bool = False,
    buckling_util_threshold: float = 1.0,
    stress_util_threshold: float = 1.0,
) -> tuple:
    """
    Preflight feasible seed -> Stage 1 (stress/shell/flare) ->
    optional frequency repair -> Stage 2 (global buckling + frequency).
    Returns (wt_opt, modeling_options, opt_exit_mode, jacket_dv_tuple).
    """
    freq_on = jacket_frequency_enabled(analysis)
    print("  Preflight feasible-start repair …")
    wt_geom = repair_feasible_jacket_start(
        run_wisdem,
        wt_geom,
        modeling,
        analysis,
        modeling_out,
        tag,
        f_lo,
        f_hi,
        member_diagnostics=member_diagnostics,
        stress_diagnostics=stress_diagnostics,
        buckling_diagnostics=buckling_diagnostics,
        buckling_util_threshold=buckling_util_threshold,
        stress_util_threshold=stress_util_threshold,
    )

    analysis_s1 = analysis_jacket_stage1(analysis)
    analysis_s1.setdefault("general", {})
    analysis_s1["general"]["folder_output"] = os.path.join(TMP_DIR, f"outputs_{tag}_stage1")
    analysis_s1["general"]["fname_output"] = "refturb_output"
    analysis_s1_path = os.path.join(TMP_DIR, f"analysis_{tag}_stage1.yaml")
    wt_s1_path = os.path.join(TMP_DIR, f"geometry_{tag}_stage1.yaml")
    dump_yaml(analysis_s1, analysis_s1_path)
    dump_yaml(wt_geom, wt_s1_path)

    print("  Stage 1 (stress/shell/flare; global buckling and frequency off, shorter budget) …")
    wt_opt, modeling_options, _opt = run_wisdem(wt_s1_path, modeling_out, analysis_s1_path)
    exit1 = get_opt_exit_mode(wt_opt)
    print_jacket_diagnostics(
        wt_opt,
        "stage1",
        f_lo,
        f_hi,
        exit_mode=exit1,
        modeling_options=modeling_options,
        member_diagnostics=member_diagnostics,
        stress_diagnostics=stress_diagnostics,
        buckling_diagnostics=buckling_diagnostics,
        buckling_util_threshold=buckling_util_threshold,
        stress_util_threshold=stress_util_threshold,
    )

    r_head, ratio, leg_d, leg_t, brace_d, brace_t = extract_jacket_dvs(wt_opt)
    wt_after_s1 = seed_jacket_geometry(
        wt_geom, r_head, ratio, leg_d, leg_t, brace_d, brace_t
    )
    if freq_on:
        print("  Frequency band repair before Stage 2 …")
        wt_after_s1 = repair_jacket_frequency_start(
            run_wisdem,
            wt_after_s1,
            modeling,
            analysis,
            modeling_out,
            f"{tag}_f1",
            f_lo,
            f_hi,
        )

    analysis_s2 = analysis_jacket_stage2(analysis)
    analysis_s2.setdefault("general", {})
    analysis_s2["general"]["folder_output"] = os.path.join(TMP_DIR, f"outputs_{tag}_stage2")
    analysis_s2["general"]["fname_output"] = "refturb_output"
    analysis_s2_path = os.path.join(TMP_DIR, f"analysis_{tag}_stage2.yaml")
    wt_s2_path = os.path.join(TMP_DIR, f"geometry_{tag}_stage2.yaml")
    dump_yaml(analysis_s2, analysis_s2_path)
    dump_yaml(wt_after_s1, wt_s2_path)

    global_on = bool(
        analysis.get("constraints", {}).get("jacket", {}).get("global_buckling", {}).get("flag", False)
    )
    if global_on and freq_on:
        print("  Stage 2 (global buckling + frequency, polish) …")
    elif global_on:
        print("  Stage 2 (full constraints incl. global buckling, polish) …")
    elif freq_on:
        print("  Stage 2 (frequency + remaining constraints, polish) …")
    else:
        print("  Stage 2 (full constraints, polish) …")
    wt_opt, modeling_options, _opt = run_wisdem(wt_s2_path, modeling_out, analysis_s2_path)
    exit2 = get_opt_exit_mode(wt_opt)
    print_jacket_diagnostics(
        wt_opt,
        "stage2",
        f_lo,
        f_hi,
        exit_mode=exit2,
        modeling_options=modeling_options,
        member_diagnostics=member_diagnostics,
        stress_diagnostics=stress_diagnostics,
        buckling_diagnostics=buckling_diagnostics,
        buckling_util_threshold=buckling_util_threshold,
        stress_util_threshold=stress_util_threshold,
    )

    dvs = extract_jacket_dvs(wt_opt)
    return wt_opt, modeling_options, exit2, dvs


def _first_dlc_vec3(val, units_name: str) -> np.ndarray:
    """Reduce DriveSE base_F/M shaped (3,) or (3, n_dlc) to a length-3 vector."""
    arr = np.asarray(val, dtype=float)
    if arr.ndim == 2 and arr.shape[0] == 3:
        return arr[:, 0]
    flat = arr.flatten()
    if flat.size < 3:
        raise ValueError(f"{units_name} has fewer than 3 components: shape={arr.shape}")
    return flat[:3]


def extract_rna_loads(wt_opt) -> dict:
    """
    Extract loads from a RotorSE+DriveSE run.

    - force/moment/mass/... : RNA (tower-top) values for Loading injection
    - shear_N / vertical_N / overturning_Nm : tower-base reactions for results CSV
      (towerse.tower.turbine_F/M; OT = My)
    """
    force = _first_dlc_vec3(wt_opt.get_val("drivese.base_F", units="N"), "drivese.base_F")
    moment = _first_dlc_vec3(wt_opt.get_val("drivese.base_M", units="N*m"), "drivese.base_M")
    turb_F = _first_dlc_vec3(
        wt_opt.get_val("towerse.tower.turbine_F", units="N"), "towerse.tower.turbine_F"
    )
    turb_M = _first_dlc_vec3(
        wt_opt.get_val("towerse.tower.turbine_M", units="N*m"), "towerse.tower.turbine_M"
    )

    mass = float(np.asarray(wt_opt.get_val("drivese.rna_mass", units="kg"), dtype=float).flatten()[0])
    cm = np.atleast_1d(np.asarray(wt_opt.get_val("drivese.rna_cm", units="m"), dtype=float)).flatten()
    moi = np.atleast_1d(np.asarray(wt_opt.get_val("drivese.rna_I_TT", units="kg*m**2"), dtype=float)).flatten()

    velocity = _safe_get(wt_opt, "rotorse.rp.powercurve.rated_V", units="m/s")
    if not np.isfinite(velocity):
        velocity = _safe_get(wt_opt, "towerse.env.Uref", units="m/s")
    if not np.isfinite(velocity):
        velocity = 11.73732

    hub_height = _safe_get(wt_opt, "high_level_tower_props.hub_height", units="m")
    rated_thrust = _safe_get(wt_opt, "rotorse.rp.powercurve.rated_T", units="N")

    loads = {
        # RNA / tower-top — used by Loading-driven sizing
        "force": [float(force[0]), float(force[1]), float(force[2])],
        "moment": [float(moment[0]), float(moment[1]), float(moment[2])],
        "velocity": float(velocity),
        "mass": mass,
        "center_of_mass": [float(x) for x in cm[:3]] + [0.0] * max(0, 3 - cm.size),
        "moment_of_inertia": [float(x) for x in moi[:6]] + [0.0] * max(0, 6 - moi.size),
        # Tower-base reactions — written to results CSV
        "shear_N": float(turb_F[0]),
        "vertical_N": float(turb_F[2]),
        "overturning_Nm": float(turb_M[1]),
        "hub_height_m": float(hub_height) if np.isfinite(hub_height) else float("nan"),
        "rated_thrust_N": float(rated_thrust) if np.isfinite(rated_thrust) else float("nan"),
    }
    return loads


def compute_loads_stage(
    run_wisdem,
    full_wt: dict,
    scenario: dict,
    foundation_type: str,
    tag: str,
    cdr,
) -> dict:
    """Stage 1: RotorSE+DriveSE run_only to compute RNA + tower-base loads."""
    modeling = patch_environment(load_yaml(MODELING_ROTORSE), scenario, for_rotorse=True)
    # Jacket FixedBottomSE options differ slightly; keep soil springs for monopile
    if foundation_type == "jacket":
        modeling.setdefault("WISDEM", {}).setdefault("FixedBottomSE", {})["soil_springs"] = False

    analysis = load_yaml(ANALYSIS_LOADS)
    analysis.setdefault("general", {})
    analysis["general"]["folder_output"] = os.path.join(TMP_DIR, f"outputs_{tag}_loads")
    analysis["general"]["fname_output"] = "refturb_output"

    wt_geom = patch_geometry(full_wt, modeling, foundation_type, cdr)

    modeling_out = os.path.join(TMP_DIR, f"modeling_{tag}_loads.yaml")
    analysis_out = os.path.join(TMP_DIR, f"analysis_{tag}_loads.yaml")
    wt_out = os.path.join(TMP_DIR, f"geometry_{tag}_loads.yaml")
    dump_yaml(modeling, modeling_out)
    dump_yaml(analysis, analysis_out)
    dump_yaml(wt_geom, wt_out)

    print("  Stage loads: RotorSE+DriveSE run_only …")
    print(f"  V_mean (from V50/7) = {modeling['Environment'].get('V_mean')}")
    wt_opt, _, _ = run_wisdem(wt_out, modeling_out, analysis_out, run_only=True)
    loads = extract_rna_loads(wt_opt)
    print(
        f"  hub_height={loads['hub_height_m']:.3g} m, "
        f"rated_thrust={loads['rated_thrust_N']:.6g} N, "
        f"RNA mass={loads['mass']:.6g} kg"
    )
    print(
        f"  Tower-base loads: shear={loads['shear_N']:.6g} N, "
        f"vertical={loads['vertical_N']:.6g} N, "
        f"OT(My)={loads['overturning_Nm']:.6g} N·m, "
        f"Uref={loads['velocity']:.4g} m/s"
    )
    print(
        f"  RNA (Loading) Fx/Fz/My = "
        f"{loads['force'][0]:.6g} / {loads['force'][2]:.6g} / {loads['moment'][1]:.6g}"
    )
    return loads


def scenario_result_extras(
    scenario: dict,
    loads: dict,
    exit_mode=None,
    rna_mass_kg: float | None = None,
) -> dict:
    """Site + tower-base load fields written into the type-specific results CSV."""
    extras = {
        "tower_base_shear_N": loads["shear_N"],
        "tower_base_vertical_N": loads["vertical_N"],
        "tower_base_overturning_moment_Nm": loads["overturning_Nm"],
        "turbine_rating_W": _to_float(scenario["turbine_rating_W"]),
        "V50_mps": _to_float(scenario["V50_mps"]),
        "Hs50_m": _to_float(scenario["Hs50_m"]),
        "soil_shear_modulus_Pa": _to_float(scenario["soil_shear_modulus_Pa"]),
        "water_depth_m": _to_float(scenario["water_depth_m"]),
    }
    if rna_mass_kg is not None and not (isinstance(rna_mass_kg, float) and not math.isfinite(rna_mass_kg)):
        extras["rna_mass_kg"] = float(rna_mass_kg)
    if exit_mode is not None and not (isinstance(exit_mode, float) and not math.isfinite(exit_mode)):
        try:
            extras["opt_exit_mode"] = int(exit_mode)
        except (TypeError, ValueError):
            extras["opt_exit_mode"] = exit_mode
    return extras


def resolve_results_prefix(doe_results: bool = False, results_prefix: str | None = None) -> str | None:
    """Stem for opt-in result CSVs, or None to keep the default examples/results_*.csv files.

    A bare name such as ``DOE_results`` is resolved next to this script. A path
    with a directory (relative or absolute) is kept as given.
    """
    if results_prefix:
        prefix = results_prefix
        if not os.path.isabs(prefix) and os.path.dirname(prefix) == "":
            prefix = os.path.join(BATCH_DIR, prefix)
        return prefix
    if doe_results:
        return DOE_RESULTS_PREFIX
    return None


def run_batch(
    csv_path: str,
    turbine_csv: str,
    foundation_mode: str = "both",
    jacket_legs_mode: str = "both",
    stop_on_error: bool = False,
    row_filter: int | None = None,
    member_diagnostics: bool = False,
    stress_diagnostics: bool = False,
    buckling_diagnostics: bool = False,
    buckling_util_threshold: float = 1.0,
    stress_util_threshold: float = 1.0,
    results_prefix: str | None = None,
) -> int:
    run_wisdem, append_foundation_results, cdr = _setup_wisdem()
    scenarios = read_scenarios(csv_path)
    if row_filter is not None:
        scenarios = [s for s in scenarios if s.get("_csv_row_number") == row_filter]
        if not scenarios:
            raise ValueError(
                f"No scenario row with CSV line number {row_filter} in {csv_path}. "
                "Row numbers match the spreadsheet line (header is row 1; first data row is 2)."
            )
    turbines = load_turbine_lookup(turbine_csv)
    foundations = resolve_foundations(foundation_mode)
    os.makedirs(TMP_DIR, exist_ok=True)

    baselines_modeling = {ftype: load_yaml(BASELINES[ftype]["modeling"]) for ftype in foundations}
    baselines_wt = {ftype: load_yaml(BASELINES[ftype]["wt"]) for ftype in foundations}

    mono_warmstart: dict[tuple, tuple[float, float]] = {}
    jacket_warmstart: dict[tuple, tuple] = {}

    n_ok = 0
    n_fail = 0
    print(f"Loaded {len(scenarios)} scenario row(s) from {csv_path}")
    if row_filter is not None:
        print(f"  (--row {row_filter}: running that CSV line only)")
    show_stress = stress_diagnostics or member_diagnostics
    show_buckling = buckling_diagnostics or member_diagnostics
    if (show_stress or show_buckling) and "jacket" in foundations:
        parts = []
        if show_stress:
            parts.append(f"stress >= {stress_util_threshold:.2f}")
        if show_buckling:
            parts.append(f"global buckling >= {buckling_util_threshold:.2f}")
        print(f"  Jacket member diagnostics enabled when {' and '.join(parts)}")
    print(f"Turbine lookup: {turbine_csv} ({len(turbines)} turbine column(s))")
    print(f"Foundation mode(s): {', '.join(foundations)}")
    if results_prefix:
        print(f"Results CSVs: {results_prefix}_{{jacket,jacket_3,monopile}}.csv")
    else:
        print("Results CSVs: examples/results_{jacket,jacket_3,monopile}.csv")
    if "jacket" in foundations:
        print(
            f"Jacket leg counts: --jacket-legs={jacket_legs_mode} "
            "(optional CSV column n_legs overrides per row)"
        )

    for scenario in scenarios:
        row_num = scenario["_csv_row_number"]
        rating = _to_float(scenario["turbine_rating_W"])
        turb_name, turb_params = lookup_turbine_column(turbines, rating, turbine_csv)
        rna_mass_kg = lookup_rna_mass(turb_params)
        print(f"\n### CSV row {row_num}: rating={rating:g} W -> turbine '{turb_name}' ###")
        if math.isfinite(rna_mass_kg):
            print(f"  RNA mass (lookup) = {rna_mass_kg:.6g} kg")

        jacket_loads = None

        for ftype in foundations:
            paths = BASELINES[ftype]
            leg_counts = (
                resolve_jacket_leg_counts(scenario, jacket_legs_mode, cdr)
                if ftype == "jacket"
                else [None]
            )

            for n_legs in leg_counts:
                tag = f"{ftype}_row{row_num}" + (f"_legs{n_legs}" if n_legs is not None else "")

                try:
                    full_wt = build_turbine_geometry(
                        turb_params, turbine_csv, ftype, baselines_wt[ftype]
                    )
                    if n_legs is not None:
                        full_wt = set_jacket_n_legs(full_wt, n_legs)
                    f_lo, f_hi = compute_frequency_bounds(full_wt)

                    # Stage 1: compute loads (reuse across 3/4-leg jacket runs on same row)
                    if ftype == "jacket":
                        if jacket_loads is None:
                            jacket_loads = compute_loads_stage(
                                run_wisdem, full_wt, scenario, ftype, tag, cdr
                            )
                        loads = jacket_loads
                    else:
                        loads = compute_loads_stage(
                            run_wisdem, full_wt, scenario, ftype, tag, cdr
                        )

                    # Stage 2: Loading-driven foundation sizing (rated + parked extreme)
                    modeling, extreme = inject_loading(
                        baselines_modeling[ftype], scenario, loads, full_wt
                    )
                    modeling_out = os.path.join(TMP_DIR, f"modeling_{tag}.yaml")
                    dump_yaml(modeling, modeling_out)

                    analysis = load_yaml(paths["analysis"])
                    analysis = patch_frequency_bounds(analysis, f_lo, f_hi, foundation_type=ftype)
                    analysis.setdefault("general", {})

                    analysis["general"]["folder_output"] = os.path.join(TMP_DIR, f"outputs_{tag}")
                    analysis["general"]["fname_output"] = "refturb_output"
                    analysis_out = os.path.join(TMP_DIR, f"analysis_{tag}.yaml")
                    dump_yaml(analysis, analysis_out)

                    site = cdr.resolve_site_params(modeling)
                    leg_label = f", n_legs={n_legs}" if n_legs is not None else ""
                    print(f"\n=== Sizing {ftype} scenario CSV row {row_num}{leg_label} ===")
                    print(
                        f"  Site: WD={site['water_depth']:.3f} m, Hs={site['hs']:.3f} m, "
                        f"H_free={site['H_free']:.3f} m"
                    )
                    g_soil = modeling.get("Environment", {}).get("soil_shear_modulus", None)
                    if g_soil is not None:
                        print(f"  soil_shear_modulus = {g_soil:.6g} Pa")
                    print(f"  n_dlc = {modeling['WISDEM'].get('n_dlc')}")
                    print(
                        f"  DLC1 rated RNA Fx/Fz/My = "
                        f"{loads['force'][0]:.6g} / {loads['force'][2]:.6g} / {loads['moment'][1]:.6g}, "
                        f"Uref={loads['velocity']:.4g} m/s"
                    )
                    print(
                        f"  DLC2 extreme (Cd={extreme['Cd']}, D_rotor={extreme['rotor_diameter_m']:.3g} m) "
                        f"Fx/Fz/My = "
                        f"{extreme['force'][0]:.6g} / {extreme['force'][2]:.6g} / {extreme['moment'][1]:.6g}, "
                        f"Uref={extreme['velocity']:.4g} m/s"
                    )
                    if ftype == "jacket":
                        print(
                            f"  Jacket topology: n_legs={n_legs}, "
                            f"n_bays={cdr.n_bays_from_height(site['H_free'])}"
                        )

                    slim_wt = slim_sizing_geometry(full_wt, ftype)
                    wt_geom = patch_geometry(slim_wt, modeling, ftype, cdr)

                    if ftype == "jacket":
                        ws_key = _jacket_warmstart_key(scenario, n_legs)
                    else:
                        ws_key = _scenario_key_for_warmstart(scenario)

                    if ftype == "monopile" and ws_key in mono_warmstart:
                        d_ws, t_ws = mono_warmstart[ws_key]
                        d_ws = max(d_ws, monopile_diameter_floor(wt_geom))
                        print(f"  Warm-start from previous soil case: D={d_ws:.4f} m, t={t_ws*1e3:.2f} mm")
                        wt_geom = seed_monopile_diameter_thickness(wt_geom, d_ws, t_ws)
                        wt_geom = cdr.apply_wt_geometry_rules(wt_geom, modeling, "monopile")
                    elif ftype == "monopile":
                        wt_geom = scale_monopile_cold_start(wt_geom)
                        wt_geom = cdr.apply_wt_geometry_rules(wt_geom, modeling, "monopile")
                    elif ftype == "jacket" and ws_key in jacket_warmstart:
                        r_h, ratio, leg_d, leg_t, brace_d, brace_t = jacket_warmstart[ws_key]
                        print(
                            f"  Warm-start from previous soil case: "
                            f"r_head={r_h:.3f} m, ratio={ratio:.3f}, leg_D={leg_d:.3f} m"
                        )
                        wt_geom = seed_jacket_geometry(
                            wt_geom, r_h, ratio, leg_d, leg_t, list(brace_d), list(brace_t)
                        )
                    elif ftype == "jacket":
                        wt_geom = scale_jacket_cold_start(wt_geom)

                    if ftype == "monopile":
                        analysis = patch_monopile_diameter_bounds(analysis, wt_geom)
                        analysis = patch_monopile_thickness_bounds(analysis, wt_geom)

                    wt_out = os.path.join(TMP_DIR, f"geometry_{tag}.yaml")
                    dump_yaml(wt_geom, wt_out)
                    dump_yaml(analysis, analysis_out)
                    print(f"  geometry: {wt_out}")
                    print(f"  modeling: {modeling_out}")

                    if ftype == "monopile":
                        wt_opt, modeling_options, exit_mode, d_opt, t_opt = run_monopile_two_stage(
                            run_wisdem, wt_geom, modeling, analysis, modeling_out, tag, cdr, f_lo, f_hi
                        )
                        if design_is_warmstart_safe(wt_opt, f_lo, f_hi):
                            mono_warmstart[ws_key] = (d_opt, t_opt)
                        else:
                            util = max_strength_utilization(wt_opt)
                            f1 = _safe_get(wt_opt, "fixedse.structural_frequencies")
                            print(
                                f"  Skipping warm-start cache (util={util:.3f}, f1={f1:.4f} Hz); "
                                "next soil row will cold-start"
                            )
                            mono_warmstart.pop(ws_key, None)
                        extras = scenario_result_extras(
                            scenario, loads, exit_mode=exit_mode, rna_mass_kg=rna_mass_kg
                        )
                        out_csv = append_foundation_results(
                            wt_opt,
                            modeling_options,
                            extras=extras,
                            results_prefix=results_prefix,
                        )
                    else:
                        wt_opt, modeling_options, exit_mode, j_dvs = run_jacket_two_stage(
                            run_wisdem,
                            wt_geom,
                            modeling,
                            analysis,
                            modeling_out,
                            tag,
                            f_lo,
                            f_hi,
                            member_diagnostics=member_diagnostics,
                            stress_diagnostics=stress_diagnostics,
                            buckling_diagnostics=buckling_diagnostics,
                            buckling_util_threshold=buckling_util_threshold,
                            stress_util_threshold=stress_util_threshold,
                        )
                        freq_on = jacket_frequency_enabled(analysis)
                        if design_is_warmstart_safe(wt_opt, f_lo, f_hi, check_frequency=freq_on):
                            r_h, ratio, leg_d, leg_t, brace_d, brace_t = j_dvs
                            jacket_warmstart[ws_key] = (
                                r_h,
                                ratio,
                                leg_d,
                                leg_t,
                                tuple(brace_d),
                                tuple(brace_t),
                            )
                        else:
                            util = max_strength_utilization(wt_opt)
                            f1 = _safe_get(wt_opt, "fixedse.structural_frequencies")
                            print(
                                f"  Skipping jacket warm-start cache (util={util:.3f}, f1={f1:.4f} Hz); "
                                "next soil row will cold-start"
                            )
                            jacket_warmstart.pop(ws_key, None)
                        extras = scenario_result_extras(
                            scenario, loads, exit_mode=exit_mode, rna_mass_kg=rna_mass_kg
                        )
                        out_csv = append_foundation_results(
                            wt_opt,
                            modeling_options,
                            extras=extras,
                            results_prefix=results_prefix,
                        )

                    print(f"  OK -> {out_csv}")
                    if exit_mode is not None and exit_mode != 0:
                        print(
                            f"  NOTE: optimizer exit_mode={exit_mode} "
                            "(0=success; 8=linesearch stuck; 9=iteration limit). "
                            "Row was still appended — check opt_exit_mode in CSV."
                        )
                    n_ok += 1
                except Exception as exc:
                    n_fail += 1
                    fail_label = f"{ftype} row {row_num}"
                    if n_legs is not None:
                        fail_label += f" (n_legs={n_legs})"
                    print(f"  FAILED {fail_label}: {exc}")
                    traceback.print_exc()
                    if stop_on_error:
                        raise

    print(f"\nBatch finished: {n_ok} succeeded, {n_fail} failed")
    return 0 if n_fail == 0 else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run WISDEM foundation scenarios from site + turbine lookup CSVs."
    )
    parser.add_argument(
        "input_csv",
        help="Site CSV with turbine_rating_W, V50_mps, Hs50_m, soil_shear_modulus_Pa, water_depth_m.",
    )
    parser.add_argument(
        "--turbine-csv",
        default=DEFAULT_TURBINE_CSV,
        help=f"Turbine lookup CSV (default: {DEFAULT_TURBINE_CSV}).",
    )
    parser.add_argument(
        "--foundation",
        choices=("jacket", "monopile", "both"),
        default="both",
        help="Which foundation baseline(s) to run (default: both).",
    )
    parser.add_argument(
        "--jacket-legs",
        choices=("3", "4", "both"),
        default="both",
        help=(
            "Jacket leg counts to run per scenario (default: both). "
            "By default, three-legged results append to examples/results_jacket_3.csv "
            "and four-legged results to examples/results_jacket.csv. "
            "Pass --doe-results to append to DOE_results_jacket_3.csv / DOE_results_jacket.csv instead. "
            "Optional site CSV column n_legs (3 or 4) overrides this for that row."
        ),
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop at the first failed scenario instead of continuing.",
    )
    parser.add_argument(
        "--row",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Run only the scenario on CSV line N (header is line 1; first data row is 2). "
            "Useful for debugging a single case from scenarios_template.csv."
        ),
    )
    parser.add_argument(
        "--stress-diagnostics",
        action="store_true",
        help=(
            "For jacket runs, print per-member von Mises stress details when utilization "
            "exceeds --stress-util-threshold (no global buckling member output)."
        ),
    )
    parser.add_argument(
        "--buckling-diagnostics",
        action="store_true",
        help=(
            "For jacket runs, print per-member global buckling details when utilization "
            "exceeds --buckling-util-threshold (no stress member output)."
        ),
    )
    parser.add_argument(
        "--member-diagnostics",
        action="store_true",
        help="Shorthand for --stress-diagnostics and --buckling-diagnostics together.",
    )
    parser.add_argument(
        "--buckling-util-threshold",
        type=float,
        default=1.0,
        help="Global buckling utilization threshold for --buckling-diagnostics (default: 1.0).",
    )
    parser.add_argument(
        "--stress-util-threshold",
        type=float,
        default=1.0,
        help="von Mises stress utilization threshold for --stress-diagnostics (default: 1.0).",
    )
    parser.add_argument(
        "--doe-results",
        action="store_true",
        help=(
            "Append this run to examples/foundation_batch/DOE_results_jacket.csv, "
            "DOE_results_jacket_3.csv, and DOE_results_monopile.csv instead of the "
            "default examples/results_*.csv files. Other runs are unchanged unless "
            "they also pass this flag."
        ),
    )
    parser.add_argument(
        "--results-prefix",
        default=None,
        metavar="PREFIX",
        help=(
            "Append to {PREFIX}_jacket.csv, {PREFIX}_jacket_3.csv, and "
            "{PREFIX}_monopile.csv. A bare name is placed in examples/foundation_batch/; "
            "a path is used as given. Overrides --doe-results."
        ),
    )
    args = parser.parse_args(argv)
    stress_diagnostics = args.stress_diagnostics or args.member_diagnostics
    buckling_diagnostics = args.buckling_diagnostics or args.member_diagnostics
    results_prefix = resolve_results_prefix(args.doe_results, args.results_prefix)
    return run_batch(
        args.input_csv,
        turbine_csv=args.turbine_csv,
        foundation_mode=args.foundation,
        jacket_legs_mode=args.jacket_legs,
        stop_on_error=args.stop_on_error,
        row_filter=args.row,
        member_diagnostics=args.member_diagnostics,
        stress_diagnostics=stress_diagnostics,
        buckling_diagnostics=buckling_diagnostics,
        buckling_util_threshold=args.buckling_util_threshold,
        stress_util_threshold=args.stress_util_threshold,
        results_prefix=results_prefix,
    )


if __name__ == "__main__":
    sys.exit(main())
