#!/usr/bin/env python3
"""
Run WISDEM foundation scenarios from any CSV with the expected column names.

Usage:
  python run_foundation_batch.py /path/to/scenarios.csv --foundation jacket
  python run_foundation_batch.py /path/to/scenarios.csv --foundation monopile
  python run_foundation_batch.py /path/to/scenarios.csv --foundation both

Each CSV row is one scenario. Extra columns are ignored. Blank cells keep the
baseline modeling-options value. Foundation type is selected via --foundation,
not from the CSV.

Conceptual design workflow (before optimisation):
  1. Read site params (water depth, Hs, Tp) and tower-base loads
  2. Set freeboard height H_free = WD + Hs + air_gap + safety_margin
  3. Jacket: assign n_bays from H_free, equal bay_spacing, n_legs=4
  4. Monopile: provisional embedment = 5 * D_initial; total length = H_free + L_emb
  5. Monopile two-stage opt: Stage 1 (full constraints, shorter) then Stage 2 polish
  6. Monopile post-pass: L_emb = 5 * D_opt, refresh analysis (run_only)
  7. Soil sweeps: warm-start D/t from previous feasible row when only G changes
"""

from __future__ import annotations

import argparse
import copy
import csv
import importlib
import importlib.util
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
    "water_depth_m",
    "Hs50_m",
    "Tp50_s",
    "soil_shear_modulus_Pa",
    "tower_base_shear_N",
    "tower_base_vertical_N",
    "tower_base_overturning_moment_Nm",
]

# Scenario fields that must match for soil-only warm start (everything except G)
_WARMSTART_MATCH_COLS = [
    "water_depth_m",
    "Hs50_m",
    "Tp50_s",
    "tower_base_shear_N",
    "tower_base_vertical_N",
    "tower_base_overturning_moment_Nm",
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

STAGE1_MAX_ITER = 50
# Reject soil warm-starts that are strength-infeasible (util > this)
_WARMSTART_MAX_UTIL = 1.05
# Match analysis_options_monopile.yaml soft-stiff frequency_1 band
_FREQ_LO = 0.22
_FREQ_HI = 0.35


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


def _setup_wisdem():
    from wisdem import run_wisdem as _run_wisdem_installed  # noqa: F401

    # Prefer this checkout for modules changed on the branch (installed wheel may lag).
    val = _load_repo_module("wisdem.inputs.validation", os.path.join("wisdem", "inputs", "validation.py"))
    if val is not None:
        import wisdem.inputs as inputs_pkg

        for name in dir(val):
            if not name.startswith("_"):
                setattr(inputs_pkg, name, getattr(val, name))
    _load_repo_module(
        "wisdem.fixed_bottomse.conceptual_design_rules",
        os.path.join("wisdem", "fixed_bottomse", "conceptual_design_rules.py"),
    )
    _load_repo_module("wisdem.fixed_bottomse.jacket", os.path.join("wisdem", "fixed_bottomse", "jacket.py"))
    _load_repo_module("wisdem.fixed_bottomse.monopile", os.path.join("wisdem", "fixed_bottomse", "monopile.py"))
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


def read_scenarios(csv_path: str) -> list[dict[str, Any]]:
    """Read any CSV; require expected column names; ignore extras."""
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
        for i, raw in enumerate(reader, start=2):  # 1-indexed data rows (header=1)
            row = {k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in raw.items() if k}
            if all(_is_blank(row.get(c, "")) for c in REQUIRED_COLUMNS):
                continue
            row["_csv_row_number"] = i
            rows.append(row)

    if not rows:
        raise ValueError(f"No scenario rows found in {csv_path}")
    return rows


def load_yaml(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def dump_yaml(data: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)


def patch_modeling_options(baseline: dict, scenario: dict) -> dict:
    """Copy baseline modeling options and apply non-blank scenario fields."""
    opts = copy.deepcopy(baseline)
    env = opts.setdefault("Environment", {})
    wisdem = opts.setdefault("WISDEM", {})
    loading = wisdem.setdefault("Loading", {})
    loads = loading.setdefault("loads", [])
    if not loads:
        loads.append({"force": [0.0, 0.0, 0.0], "moment": [0.0, 0.0, 0.0], "velocity": 0.0})
        loading["loads"] = loads

    load0 = loads[0]
    force = list(load0.get("force", [0.0, 0.0, 0.0]))
    moment = list(load0.get("moment", [0.0, 0.0, 0.0]))
    while len(force) < 3:
        force.append(0.0)
    while len(moment) < 3:
        moment.append(0.0)

    mapping_env = {
        "water_depth_m": "water_depth",
        "Hs50_m": "significant_wave_height",
        "Tp50_s": "significant_wave_period",
        "soil_shear_modulus_Pa": "soil_shear_modulus",
    }
    for csv_col, yaml_key in mapping_env.items():
        if not _is_blank(scenario.get(csv_col)):
            env[yaml_key] = _to_float(scenario[csv_col])

    if not _is_blank(scenario.get("tower_base_shear_N")):
        force[0] = _to_float(scenario["tower_base_shear_N"])
    if not _is_blank(scenario.get("tower_base_vertical_N")):
        force[2] = _to_float(scenario["tower_base_vertical_N"])
    if not _is_blank(scenario.get("tower_base_overturning_moment_Nm")):
        moment[0] = _to_float(scenario["tower_base_overturning_moment_Nm"])

    load0["force"] = force
    load0["moment"] = moment
    loads[0] = load0
    loading["loads"] = loads
    wisdem["Loading"] = loading
    opts["WISDEM"] = wisdem
    opts["Environment"] = env
    return opts


def patch_geometry(baseline_wt: dict, modeling: dict, foundation_type: str, cdr, embedment=None) -> dict:
    """Apply conceptual height / bay topology rules to geometry YAML."""
    return cdr.apply_wt_geometry_rules(
        baseline_wt, modeling, foundation_type, embedment=embedment
    )


def _as_float_list(val) -> list[float]:
    arr = np.atleast_1d(np.asarray(val, dtype=float).flatten())
    return [float(x) for x in arr]


def _scenario_key_for_warmstart(scenario: dict) -> tuple:
    """Hashable key of site/load fields (excludes soil G)."""
    parts = []
    for col in _WARMSTART_MATCH_COLS:
        v = scenario.get(col, "")
        if _is_blank(v):
            parts.append(None)
        else:
            parts.append(round(_to_float(v), 6))
    return tuple(parts)


def seed_monopile_diameter_thickness(wt_geom: dict, diameter: float, thickness: float) -> dict:
    """Overwrite monopile outer diameter and wall thickness (uniform)."""
    wt = copy.deepcopy(wt_geom)
    mono = wt["components"]["monopile"]
    n_d = len(mono["outer_shape"]["outer_diameter"]["values"])
    mono["outer_shape"]["outer_diameter"]["values"] = [float(diameter)] * n_d
    n_t = len(mono["structure"]["layers"][0]["thickness"]["values"])
    mono["structure"]["layers"][0]["thickness"]["values"] = [float(thickness)] * n_t
    return wt


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


def write_optimised_monopile_geometry(wt_geom: dict, wt_opt, modeling: dict, cdr) -> dict:
    """
    After sizing, set L_emb = 5 * D_opt and copy optimised D/t into geometry.
    """
    wt = copy.deepcopy(wt_geom)
    mono = wt["components"]["monopile"]

    diam = _as_float_list(wt_opt.get_val("monopile.diameter", units="m"))
    # Keep geometry grid length; broadcast / trim as needed
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
        # thick shape (n_layers, n_height)
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
    l_emb = cdr.embedment_length(d_mud)
    site = cdr.resolve_site_params(modeling)
    print(f"  Post-opt embedment: L_emb = 5 * {d_mud:.4f} m = {l_emb:.4f} m")
    print(f"  H_free = {site['H_free']:.4f} m; total height = {site['H_free'] + l_emb:.4f} m")

    return cdr.apply_wt_geometry_rules(wt, modeling, "monopile", embedment=l_emb)


def analysis_refresh_options(analysis: dict) -> dict:
    """Disable optimisation for the monopile embedment refresh pass."""
    opts = copy.deepcopy(analysis)
    opts.setdefault("driver", {}).setdefault("optimization", {})["flag"] = False
    # Ensure height remains a non-DV even if schema defaults differ
    mono = opts.setdefault("design_variables", {}).setdefault("monopile", {})
    mono.setdefault("height", {})["flag"] = False
    mono.setdefault("outer_diameter", {})["flag"] = False
    mono.setdefault("layer_thickness", {})["flag"] = False
    return opts


def analysis_stage1_strength(analysis: dict) -> dict:
    """
    Stage 1: full monopile constraints including frequency.

    Frequency stays on: the soft-stiff band (see analysis_options) prevents a
    strength-only pass from locking onto an over-stiff thick design that Stage 2
    then tears down into buckling-infeasible thin walls.
    """
    opts = copy.deepcopy(analysis)
    opts.setdefault("constraints", {}).setdefault("monopile", {}).setdefault("frequency_1", {})["flag"] = True
    opts.setdefault("driver", {}).setdefault("optimization", {})["max_iter"] = STAGE1_MAX_ITER
    return opts


def analysis_stage2_full(analysis: dict) -> dict:
    """Stage 2: polish with full constraints and the standard iteration budget."""
    opts = copy.deepcopy(analysis)
    opts.setdefault("constraints", {}).setdefault("monopile", {}).setdefault("frequency_1", {})["flag"] = True
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
            return 8  # typical SLSQP linesearch failure in this workflow
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
    ):
        v = _safe_get(wt_opt, name, reduce="max")
        if np.isfinite(v):
            vals.append(v)
    return float(max(vals)) if vals else float("nan")


def design_is_warmstart_safe(wt_opt) -> bool:
    """True if strength utils and f1 look feasible enough to seed the next soil case."""
    util = max_strength_utilization(wt_opt)
    f1 = _safe_get(wt_opt, "fixedse.structural_frequencies")
    if not np.isfinite(util) or util > _WARMSTART_MAX_UTIL:
        return False
    if not np.isfinite(f1) or f1 < _FREQ_LO or f1 > _FREQ_HI:
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


# Match monopile DV bounds in analysis_options_monopile.yaml
_D_LO = 3.87
_D_HI = 8.0
_T_LO = 5e-2
_T_HI = 2e-1


def repair_feasible_monopile_start(
    run_wisdem,
    wt_geom: dict,
    modeling: dict,
    analysis: dict,
    modeling_out: str,
    tag: str,
    cdr,
    max_passes: int = 8,
) -> dict:
    """
    Nudge D/t with run_only analyses until strength util and f1 are roughly feasible.

    SLSQP often exits mode 8 immediately from a badly infeasible start; this gives
    the optimiser a point inside / near the feasible region first.
    """
    wt = copy.deepcopy(wt_geom)
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
        wt = cdr.apply_wt_geometry_rules(
            seed_monopile_diameter_thickness(wt, d, t),
            modeling,
            "monopile",
            embedment=cdr.embedment_length(d),
        )
        geom_path = os.path.join(TMP_DIR, f"geometry_{tag}_preflight_{i}.yaml")
        dump_yaml(wt, geom_path)
        wt_opt, _, _ = run_wisdem(geom_path, modeling_out, analysis_path, run_only=True)
        util = max_strength_utilization(wt_opt)
        f1 = _safe_get(wt_opt, "fixedse.structural_frequencies")
        print(f"  [preflight {i}] D={d:.3f} t={t*1e3:.2f} mm  util={util:.3f}  f1={f1:.4f} Hz")
        if design_is_warmstart_safe(wt_opt):
            print("  Preflight: start is feasible enough for SLSQP")
            return wt

        d_new, t_new = d, t
        if np.isfinite(util) and util > 1.0:
            t_new = min(_T_HI, max(_T_LO, t * util * 1.05))
        # With soil springs, larger D raises stiffness/f1 — move opposite to the old heuristic
        if np.isfinite(f1) and f1 > _FREQ_HI:
            d_new = max(_D_LO, d - 0.25)
            if not (np.isfinite(util) and util > 1.0):
                t_new = max(_T_LO, min(t_new, t * 0.95))
        if np.isfinite(f1) and f1 < _FREQ_LO:
            d_new = min(_D_HI, d + 0.25)
            t_new = min(_T_HI, max(t_new, t * 1.05))

        if abs(d_new - d) < 1e-9 and abs(t_new - t) < 1e-12:
            # Cannot move further within bounds
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
) -> tuple:
    """
    Preflight feasible seed -> Stage 1 (full, shorter) -> Stage 2 polish ->
    5*D embedment refresh. Returns (wt_opt, modeling_options, opt_exit_mode, d, t).
    """
    print("  Preflight feasible-start repair …")
    wt_geom = repair_feasible_monopile_start(
        run_wisdem, wt_geom, modeling, analysis, modeling_out, tag, cdr
    )

    # --- Stage 1 ---
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
    wt_after_s1 = seed_monopile_diameter_thickness(wt_geom, d1, t1)
    # Keep provisional embedment consistent with Stage-1 D
    wt_after_s1 = cdr.apply_wt_geometry_rules(
        wt_after_s1, modeling, "monopile", embedment=cdr.embedment_length(d1)
    )

    # --- Stage 2 ---
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

    # --- Embedment refresh ---
    wt_final = write_optimised_monopile_geometry(wt_after_s1, wt_opt, modeling, cdr)
    wt_refresh = os.path.join(TMP_DIR, f"geometry_{tag}_embedment.yaml")
    analysis_refresh = os.path.join(TMP_DIR, f"analysis_{tag}_refresh.yaml")
    dump_yaml(wt_final, wt_refresh)
    dump_yaml(analysis_refresh_options(analysis_s2), analysis_refresh)
    print("  Refreshing analysis with final 5*D embedment …")
    wt_opt, modeling_options, _opt = run_wisdem(
        wt_refresh, modeling_out, analysis_refresh, run_only=True
    )
    print_monopile_diagnostics(wt_opt, "refresh", exit_mode=exit2)

    return wt_opt, modeling_options, exit2, d2, t2


def run_batch(csv_path: str, foundation_mode: str = "both", stop_on_error: bool = False) -> int:
    run_wisdem, append_foundation_results, cdr = _setup_wisdem()
    scenarios = read_scenarios(csv_path)
    foundations = resolve_foundations(foundation_mode)
    os.makedirs(TMP_DIR, exist_ok=True)

    baselines_modeling = {ftype: load_yaml(BASELINES[ftype]["modeling"]) for ftype in foundations}
    baselines_wt = {ftype: load_yaml(BASELINES[ftype]["wt"]) for ftype in foundations}

    # Warm-start cache: site/load key -> (D, t) from last successful monopile Stage-2
    mono_warmstart: dict[tuple, tuple[float, float]] = {}

    n_ok = 0
    n_fail = 0
    print(f"Loaded {len(scenarios)} scenario row(s) from {csv_path}")
    print(f"Foundation mode(s): {', '.join(foundations)}")

    for scenario in scenarios:
        row_num = scenario["_csv_row_number"]
        for ftype in foundations:
            paths = BASELINES[ftype]
            tag = f"{ftype}_row{row_num}"
            modeling_out = os.path.join(TMP_DIR, f"modeling_{tag}.yaml")
            analysis_out = os.path.join(TMP_DIR, f"analysis_{tag}.yaml")
            wt_out = os.path.join(TMP_DIR, f"geometry_{tag}.yaml")

            analysis = load_yaml(paths["analysis"])
            analysis.setdefault("general", {})
            analysis["general"]["folder_output"] = os.path.join(TMP_DIR, f"outputs_{tag}")
            analysis["general"]["fname_output"] = "refturb_output"
            dump_yaml(analysis, analysis_out)

            modeling = patch_modeling_options(baselines_modeling[ftype], scenario)
            dump_yaml(modeling, modeling_out)

            site = cdr.resolve_site_params(modeling)
            print(f"\n=== Running {ftype} scenario CSV row {row_num} ===")
            print(
                f"  Site: WD={site['water_depth']:.3f} m, Hs={site['hs']:.3f} m, "
                f"H_free={site['H_free']:.3f} m"
            )
            g_soil = modeling.get("Environment", {}).get("soil_shear_modulus", None)
            if g_soil is not None:
                print(f"  soil_shear_modulus = {g_soil:.6g} Pa")
            if ftype == "jacket":
                print(f"  Jacket topology: n_legs=4, n_bays={cdr.n_bays_from_height(site['H_free'])}")

            wt_geom = patch_geometry(baselines_wt[ftype], modeling, ftype, cdr)

            # Soil-sweep warm start: only G differs; skip if last design was infeasible
            ws_key = _scenario_key_for_warmstart(scenario)
            if ftype == "monopile" and ws_key in mono_warmstart:
                d_ws, t_ws = mono_warmstart[ws_key]
                print(f"  Warm-start from previous soil case: D={d_ws:.4f} m, t={t_ws*1e3:.2f} mm")
                wt_geom = seed_monopile_diameter_thickness(wt_geom, d_ws, t_ws)
                wt_geom = cdr.apply_wt_geometry_rules(
                    wt_geom, modeling, "monopile", embedment=cdr.embedment_length(d_ws)
                )

            dump_yaml(wt_geom, wt_out)
            print(f"  geometry: {wt_out}")
            print(f"  modeling: {modeling_out}")

            try:
                if ftype == "monopile":
                    wt_opt, modeling_options, exit_mode, d_opt, t_opt = run_monopile_two_stage(
                        run_wisdem, wt_geom, modeling, analysis, modeling_out, tag, cdr
                    )
                    if design_is_warmstart_safe(wt_opt):
                        mono_warmstart[ws_key] = (d_opt, t_opt)
                    else:
                        util = max_strength_utilization(wt_opt)
                        f1 = _safe_get(wt_opt, "fixedse.structural_frequencies")
                        print(
                            f"  Skipping warm-start cache (util={util:.3f}, f1={f1:.4f} Hz); "
                            "next soil row will cold-start"
                        )
                        mono_warmstart.pop(ws_key, None)
                    extras = {"opt_exit_mode": exit_mode}
                    out_csv = append_foundation_results(
                        wt_opt, modeling_options, extras=extras
                    )
                else:
                    wt_opt, modeling_options, _opt = run_wisdem(wt_out, modeling_out, analysis_out)
                    extras = {"opt_exit_mode": get_opt_exit_mode(wt_opt)}
                    out_csv = append_foundation_results(
                        wt_opt, modeling_options, extras=extras
                    )

                print(f"  OK -> {out_csv}")
                n_ok += 1
            except Exception as exc:
                n_fail += 1
                print(f"  FAILED {ftype} row {row_num}: {exc}")
                traceback.print_exc()
                if stop_on_error:
                    raise

    print(f"\nBatch finished: {n_ok} succeeded, {n_fail} failed")
    return 0 if n_fail == 0 else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run WISDEM foundation scenarios from a CSV of key inputs."
    )
    parser.add_argument(
        "input_csv",
        help="Path to any CSV with the required column names (extra columns ignored).",
    )
    parser.add_argument(
        "--foundation",
        choices=("jacket", "monopile", "both"),
        default="both",
        help="Which foundation baseline(s) to run (default: both).",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop at the first failed scenario instead of continuing.",
    )
    args = parser.parse_args(argv)
    return run_batch(args.input_csv, foundation_mode=args.foundation, stop_on_error=args.stop_on_error)


if __name__ == "__main__":
    sys.exit(main())
