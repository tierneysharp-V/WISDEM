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

BASELINES = {
    "jacket": {
        "wt": os.path.join(JACKET_DIR, "nrel5mw_jacket.yaml"),
        "modeling": os.path.join(JACKET_DIR, "modeling_options_jacket.yaml"),
        "analysis": os.path.join(JACKET_DIR, "analysis_options_jacket.yaml"),
    },
    "monopile": {
        "wt": os.path.join(BATCH_DIR, "nrel5mw_monopile.yaml"),
        "modeling": os.path.join(BATCH_DIR, "modeling_options_monopile.yaml"),
        "analysis": os.path.join(BATCH_DIR, "analysis_options_monopile.yaml"),
    },
}


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
    _load_repo_module("wisdem.fixed_bottomse.jacket", os.path.join("wisdem", "fixed_bottomse", "jacket.py"))
    _load_repo_module("wisdem.fixed_bottomse.monopile", os.path.join("wisdem", "fixed_bottomse", "monopile.py"))
    _load_repo_module("wisdem.glue_code.gc_WT_DataStruc", os.path.join("wisdem", "glue_code", "gc_WT_DataStruc.py"))
    _load_repo_module("wisdem.glue_code.gc_WT_InitModel", os.path.join("wisdem", "glue_code", "gc_WT_InitModel.py"))
    _load_repo_module(
        "wisdem.glue_code.gc_PoseOptimization",
        os.path.join("wisdem", "glue_code", "gc_PoseOptimization.py"),
    )
    _load_repo_module("wisdem.orbit.orbit_api", os.path.join("wisdem", "orbit", "orbit_api.py"))
    _load_repo_module("wisdem.glue_code.glue_code", os.path.join("wisdem", "glue_code", "glue_code.py"))
    import wisdem.glue_code.runWISDEM as run_wisdem_mod

    importlib.reload(run_wisdem_mod)
    sys.path.insert(0, EXAMPLES_DIR)
    from extract_foundation_results import append_foundation_results  # noqa: E402

    return run_wisdem_mod.run_wisdem, append_foundation_results


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
            # Normalize keys (strip whitespace on header names)
            row = {k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in raw.items() if k}
            # Skip fully empty rows
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


def resolve_foundations(mode: str) -> list[str]:
    mode = mode.lower().strip()
    if mode == "both":
        return ["jacket", "monopile"]
    if mode in ("jacket", "monopile"):
        return [mode]
    raise ValueError(f"Invalid --foundation '{mode}'. Use jacket, monopile, or both.")


def run_batch(csv_path: str, foundation_mode: str = "both", stop_on_error: bool = False) -> int:
    run_wisdem, append_foundation_results = _setup_wisdem()
    scenarios = read_scenarios(csv_path)
    foundations = resolve_foundations(foundation_mode)
    os.makedirs(TMP_DIR, exist_ok=True)

    # Cache baseline modeling YAMLs
    baselines_modeling = {ftype: load_yaml(BASELINES[ftype]["modeling"]) for ftype in foundations}

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

            # Point analysis output folder per run to avoid clobbering
            analysis = load_yaml(paths["analysis"])
            analysis.setdefault("general", {})
            analysis["general"]["folder_output"] = os.path.join(TMP_DIR, f"outputs_{tag}")
            analysis["general"]["fname_output"] = "refturb_output"
            dump_yaml(analysis, analysis_out)

            modeling = patch_modeling_options(baselines_modeling[ftype], scenario)
            dump_yaml(modeling, modeling_out)

            print(f"\n=== Running {ftype} scenario CSV row {row_num} ===")
            print(f"  modeling: {modeling_out}")
            try:
                wt_opt, modeling_options, _opt = run_wisdem(
                    paths["wt"], modeling_out, analysis_out
                )
                out_csv = append_foundation_results(wt_opt, modeling_options)
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
