#!/usr/bin/env python3
import importlib
import importlib.util
import os
import sys

import numpy as np

## File management
mydir = os.path.dirname(os.path.realpath(__file__))  # get path to this file
repo_root = os.path.dirname(os.path.dirname(mydir))
examples_dir = os.path.dirname(mydir)


def _load_repo_module(qualname, relpath):
    """Load a WISDEM submodule from this git checkout (conda install keeps compiled extensions)."""
    path = os.path.join(repo_root, relpath)
    if not os.path.isfile(path):
        return
    spec = importlib.util.spec_from_file_location(qualname, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[qualname] = mod
    spec.loader.exec_module(mod)
    return mod


# Import installed WISDEM first so compiled extensions and dependencies resolve.
from wisdem import run_wisdem as _run_wisdem_installed  # noqa: F401

# Prefer this checkout for modules changed on the branch (installed wheel may lag).
_val = _load_repo_module("wisdem.inputs.validation", os.path.join("wisdem", "inputs", "validation.py"))
if _val is not None:
    import wisdem.inputs as _inputs_pkg

    for _name in dir(_val):
        if not _name.startswith("_"):
            setattr(_inputs_pkg, _name, getattr(_val, _name))
_load_repo_module("wisdem.fixed_bottomse.jacket", os.path.join("wisdem", "fixed_bottomse", "jacket.py"))
_load_repo_module(
    "wisdem.glue_code.gc_PoseOptimization",
    os.path.join("wisdem", "glue_code", "gc_PoseOptimization.py"),
)
_load_repo_module("wisdem.orbit.orbit_api", os.path.join("wisdem", "orbit", "orbit_api.py"))
_load_repo_module("wisdem.glue_code.glue_code", os.path.join("wisdem", "glue_code", "glue_code.py"))
import wisdem.glue_code.runWISDEM as _run_wisdem_mod

importlib.reload(_run_wisdem_mod)
run_wisdem = _run_wisdem_mod.run_wisdem

sys.path.insert(0, examples_dir)
from extract_foundation_results import append_foundation_results  # noqa: E402

fname_wt_input = mydir + os.sep + "nrel5mw_jacket.yaml"
fname_modeling_options = mydir + os.sep + "modeling_options_jacket.yaml"
fname_analysis_options = mydir + os.sep + "analysis_options_jacket.yaml"

wt_opt, modeling_options, opt_options = run_wisdem(fname_wt_input, fname_modeling_options, fname_analysis_options)

# print results from the analysis
print()
print("Mass (kg) =", np.max(wt_opt.get_val("fixedse.jacket_mass", units="kg")))
print("Jacket Cost (USD)=", np.max(wt_opt.get_val("fixedse.jacket_cost", units="USD")))
print("Stress =", np.max(wt_opt.get_val("fixedse.constr_stress")))
print("Global buckling =", np.max(wt_opt.get_val("fixedse.constr_global_buckling")))
print("Shell buckling =", np.max(wt_opt.get_val("fixedse.constr_shell_buckling")))

append_foundation_results(wt_opt, modeling_options)
