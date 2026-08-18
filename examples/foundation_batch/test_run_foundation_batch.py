"""Unit tests for foundation_batch helpers (no full WISDEM run required)."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

BATCH_DIR = os.path.dirname(os.path.realpath(__file__))
EXAMPLES_DIR = os.path.dirname(BATCH_DIR)
REPO_ROOT = os.path.dirname(EXAMPLES_DIR)
sys.path.insert(0, BATCH_DIR)
sys.path.insert(0, EXAMPLES_DIR)
sys.path.insert(0, REPO_ROOT)

import run_foundation_batch as rfb  # noqa: E402
from extract_foundation_results import csv_path_for_foundation_type  # noqa: E402

# Load checkout conceptual_design_rules (same pattern as run_foundation_batch._setup_wisdem)
cdr = rfb._load_repo_module(
    "wisdem.fixed_bottomse.conceptual_design_rules",
    os.path.join("wisdem", "fixed_bottomse", "conceptual_design_rules.py"),
)
if cdr is None:
    import wisdem.fixed_bottomse.conceptual_design_rules as cdr  # noqa: E402


class TestFoundationBatchHelpers(unittest.TestCase):
    def test_required_baseline_files_exist(self):
        for ftype, paths in rfb.BASELINES.items():
            for key in ("wt", "modeling", "analysis"):
                self.assertTrue(os.path.isfile(paths[key]), f"Missing {ftype} {key}: {paths[key]}")
        for path in (rfb.MODELING_ROTORSE, rfb.ANALYSIS_LOADS, rfb.DEFAULT_TURBINE_CSV):
            self.assertTrue(os.path.isfile(path), path)

    def test_read_scenarios_template(self):
        rows = rfb.read_scenarios(os.path.join(BATCH_DIR, "scenarios_template.csv"))
        self.assertEqual(len(rows), 3)
        self.assertIn("turbine_rating_W", rows[0])

    def test_read_scenarios_optional_n_legs(self):
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8") as f:
            f.write(
                "turbine_rating_W,V50_mps,Hs50_m,soil_shear_modulus_Pa,water_depth_m,n_legs\n"
                "1e7,70,10,1.4e8,55,3\n"
            )
            path = f.name
        try:
            rows = rfb.read_scenarios(path)
            self.assertEqual(rows[0]["n_legs"], "3")
        finally:
            os.unlink(path)

    def test_resolve_foundations(self):
        self.assertEqual(rfb.resolve_foundations("both"), ["jacket", "monopile"])
        self.assertEqual(rfb.resolve_foundations("jacket"), ["jacket"])

    def test_resolve_jacket_leg_counts(self):
        scenario = {}
        self.assertEqual(rfb.resolve_jacket_leg_counts(scenario, "both", cdr), [3, 4])
        self.assertEqual(rfb.resolve_jacket_leg_counts(scenario, "3", cdr), [3])
        self.assertEqual(
            rfb.resolve_jacket_leg_counts({"n_legs": "4"}, "both", cdr),
            [4],
        )

    def test_set_jacket_n_legs(self):
        wt = rfb.load_yaml(rfb.BASELINES["jacket"]["wt"])
        out = rfb.set_jacket_n_legs(wt, 3)
        self.assertEqual(out["components"]["jacket"]["n_legs"], 3)

    def test_slim_sizing_geometry(self):
        wt = rfb.load_yaml(rfb.BASELINES["monopile"]["wt"])
        slim = rfb.slim_sizing_geometry(wt, "monopile")
        self.assertIn("tower", slim["components"])
        self.assertIn("monopile", slim["components"])
        self.assertNotIn("blade", slim.get("components", {}))

    def test_csv_path_routing(self):
        mo4 = {"flags": {"jacket": True}, "WISDEM": {"FixedBottomSE": {"n_legs": 4}}}
        mo3 = {"flags": {"jacket": True}, "WISDEM": {"FixedBottomSE": {"n_legs": 3}}}
        self.assertTrue(csv_path_for_foundation_type("jacket", mo4).endswith("results_jacket.csv"))
        self.assertTrue(csv_path_for_foundation_type("jacket", mo3).endswith("results_jacket_3.csv"))

    def test_apply_jacket_topology_preserves_three_legs(self):
        jacket = {"n_legs": 3, "brace_diameters": [1.0], "brace_thicknesses": [0.1]}
        out = cdr.apply_jacket_topology(jacket, 45.0)
        self.assertEqual(out["n_legs"], 3)

    def test_jacket_stage1_disables_global_buckling(self):
        analysis = {
            "constraints": {
                "jacket": {
                    "stress": {"flag": True},
                    "global_buckling": {"flag": True},
                    "shell_buckling": {"flag": True},
                }
            },
            "driver": {"optimization": {"max_iter": 50}},
        }
        s1 = rfb.analysis_jacket_stage1(analysis)
        s2 = rfb.analysis_jacket_stage2(analysis)
        self.assertFalse(s1["constraints"]["jacket"]["global_buckling"]["flag"])
        self.assertTrue(s2["constraints"]["jacket"]["global_buckling"]["flag"])

    def test_jacket_preflight_scale_aggressive_for_high_global_buck(self):
        self.assertGreater(rfb._jacket_preflight_scale(0.8, 486.0), 2.0)
        self.assertLessEqual(rfb._jacket_preflight_scale(0.8, 486.0), 3.0)
        self.assertAlmostEqual(rfb._jacket_preflight_scale(1.2, 2.0), 1.5)


if __name__ == "__main__":
    unittest.main()
