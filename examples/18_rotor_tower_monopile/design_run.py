#!/usr/bin/env python3
import importlib
import importlib.util
import os
import sys

## File management
run_dir = os.path.dirname(os.path.realpath(__file__)) + os.sep
repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
examples_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


def _load_repo_module(qualname, relpath):
    """Prefer this checkout's ORBIT API over an older installed wisdem wheel."""
    path = os.path.join(repo_root, relpath)
    if not os.path.isfile(path):
        return
    spec = importlib.util.spec_from_file_location(qualname, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[qualname] = mod
    spec.loader.exec_module(mod)


from wisdem.glue_code.runWISDEM import run_wisdem as _run_wisdem_installed  # noqa: F401

_load_repo_module("wisdem.orbit.orbit_api", os.path.join("wisdem", "orbit", "orbit_api.py"))
_load_repo_module("wisdem.glue_code.glue_code", os.path.join("wisdem", "glue_code", "glue_code.py"))
import wisdem.glue_code.runWISDEM as _run_wisdem_mod

importlib.reload(_run_wisdem_mod)
run_wisdem = _run_wisdem_mod.run_wisdem

sys.path.insert(0, examples_dir)
from extract_foundation_results import append_foundation_results  # noqa: E402

fname_wt_input = os.path.join(run_dir, "IEA15MW_scaled.yaml")
fname_modeling_options = os.path.join(run_dir, "modeling_options.yaml")
fname_analysis_options = os.path.join(run_dir, "analysis_options.yaml")

wt_opt, modeling_options, analysis_options = run_wisdem(fname_wt_input, fname_modeling_options, fname_analysis_options)

append_foundation_results(wt_opt, modeling_options)
