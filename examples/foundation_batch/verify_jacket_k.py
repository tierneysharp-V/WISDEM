#!/usr/bin/env python3
"""Print which Python files govern jacket global-buckling k, and a quick sanity check."""

from __future__ import annotations

import importlib.util
import inspect
import os
import sys
import types
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]


def _ensure_wisdem_root_stub():
    if "wisdem" in sys.modules:
        return
    mod = types.ModuleType("wisdem")
    mod.__path__ = [str(REPO / "wisdem")]
    mod.__package__ = "wisdem"
    sys.modules["wisdem"] = mod


def _load_module(qualname: str, relpath: str):
    path = REPO / relpath
    spec = importlib.util.spec_from_file_location(qualname, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[qualname] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    _ensure_wisdem_root_stub()
    dnvgl = _load_module(
        "wisdem.commonse.utilization_dnvgl",
        "wisdem/commonse/utilization_dnvgl.py",
    )
    jacket = _load_module(
        "wisdem.fixed_bottomse.jacket",
        "wisdem/fixed_bottomse/jacket.py",
    )

    if not hasattr(jacket, "JacketSE"):
        print("ERROR: jacket module loaded but JacketSE is missing (partial import).")
        return 1

    print("\n=== Files Python is using ===")
    print(f"utilization_dnvgl: {inspect.getfile(dnvgl)}")
    print(f"jacket.py:         {inspect.getfile(jacket)}")
    print(
        f"k constants:       legs={jacket.JACKET_GLOBAL_BUCKLING_K_LEG}, "
        f"braces={jacket.JACKET_GLOBAL_BUCKLING_K_BRACE}"
    )

    sig = inspect.signature(dnvgl.CylinderBuckling.__init__)
    print(f"CylinderBuckling accepts k: {'k' in sig.parameters}")

    h, d, t = 10.365, 0.515, 0.05
    E, G, sigy = 200e9, 79.3e9, 345e6
    Fz, M = -835.6e3, 129.1e3
    sigma_x = Fz / (np.pi * (d / 2) ** 2 * 2 * t)

    old = dnvgl.CylinderBuckling(
        np.atleast_1d(h), np.atleast_1d(d), np.atleast_1d(t), E=E, G=G, sigma_y=sigy, k=2.0
    )
    new = dnvgl.CylinderBuckling(
        np.atleast_1d(h), np.atleast_1d(d), np.atleast_1d(t), E=E, G=G, sigma_y=sigy, k=0.8
    )
    g_old = float(np.asarray(old.run_buckling_checks(Fz, M, sigma_x, 0.0, 0.0)["Global"]).ravel()[0])
    g_new = float(np.asarray(new.run_buckling_checks(Fz, M, sigma_x, 0.0, 0.0)["Global"]).ravel()[0])

    print("\n=== Sanity check (brace_leg2_bay3_SW-like segment) ===")
    print(f"  global util with k=2.0: {g_old:.3f}")
    print(f"  global util with k=0.8: {g_new:.3f}")
    if abs(g_old - g_new) < 1e-6:
        print("  WARNING: k change had no effect - still on old utilization_dnvgl logic.")
        return 1
    print("  OK: k parameter changes global utilization in the active module.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
