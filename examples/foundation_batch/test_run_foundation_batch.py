"""Unit tests for foundation_batch helpers (no full WISDEM run required)."""

from __future__ import annotations

import math
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
from extract_foundation_results import (  # noqa: E402
    JACKET_CSV_COLUMNS,
    MONOPILE_CSV_COLUMNS,
    csv_path_for_foundation_type,
)

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

    def test_rna_mass_not_written_to_geometry(self):
        turbines = rfb.load_turbine_lookup(rfb.DEFAULT_TURBINE_CSV)
        _, params = rfb.lookup_turbine_column(turbines, 15e6, rfb.DEFAULT_TURBINE_CSV)
        self.assertIn("rna_mass_kg", params)
        wt = rfb.build_turbine_geometry(
            params, rfb.DEFAULT_TURBINE_CSV, "jacket", rfb.load_yaml(rfb.BASELINES["jacket"]["wt"])
        )
        self.assertNotIn("rna_mass_kg", wt)

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

    def test_jacket_stage1_disables_global_buckling_and_frequency(self):
        analysis = {
            "constraints": {
                "jacket": {
                    "stress": {"flag": True},
                    "global_buckling": {"flag": True},
                    "shell_buckling": {"flag": True},
                    "frequency_1": {"flag": True, "lower_bound": 0.13, "upper_bound": 0.2},
                }
            },
            "driver": {"optimization": {"max_iter": 50, "step_size": 1.0e-6, "form": "forward"}},
        }
        s1 = rfb.analysis_jacket_stage1(analysis)
        s2 = rfb.analysis_jacket_stage2(analysis)
        self.assertFalse(s1["constraints"]["jacket"]["global_buckling"]["flag"])
        self.assertFalse(s1["constraints"]["jacket"]["frequency_1"]["flag"])
        self.assertTrue(s2["constraints"]["jacket"]["global_buckling"]["flag"])
        self.assertTrue(s2["constraints"]["jacket"]["frequency_1"]["flag"])
        self.assertEqual(s2["driver"]["optimization"]["form"], "central")
        self.assertGreaterEqual(s2["driver"]["optimization"]["step_size"], 1.0e-5)

    def test_jacket_stage2_leaves_fd_unchanged_without_frequency(self):
        analysis = {
            "constraints": {
                "jacket": {
                    "stress": {"flag": True},
                    "frequency_1": {"flag": False},
                }
            },
            "driver": {"optimization": {"step_size": 1.0e-6, "form": "forward"}},
        }
        s2 = rfb.analysis_jacket_stage2(analysis)
        self.assertEqual(s2["driver"]["optimization"]["form"], "forward")
        self.assertEqual(s2["driver"]["optimization"]["step_size"], 1.0e-6)

    def test_jacket_preflight_scale_aggressive_for_high_global_buck(self):
        self.assertGreater(rfb._jacket_preflight_scale(0.8, 486.0), 2.0)
        self.assertLessEqual(rfb._jacket_preflight_scale(0.8, 486.0), 3.0)
        self.assertAlmostEqual(rfb._jacket_preflight_scale(1.2, 2.0), 1.5)

    def test_classify_jacket_stress_governing_bending(self):
        gamma = 1.35
        sigy = 345e6
        axial_force = -20e6
        bending = 180e6
        ax = axial_force + bending
        governing, detail = rfb.classify_jacket_stress_governing(
            ax, 0.0, 0.0, sigy, gamma, axial_force, bending
        )
        self.assertEqual(governing, "bending-dominated von Mises")
        self.assertGreater(detail["util_vm"], 1.0)

    def test_classify_jacket_stress_governing_hoop(self):
        gamma = 1.35
        sigy = 345e6
        hoop = -120e6
        governing, detail = rfb.classify_jacket_stress_governing(
            10e6, hoop, 0.0, sigy, gamma, 10e6, 0.0
        )
        self.assertEqual(governing, "hydrodynamic hoop pressure")
        self.assertGreater(detail["util_hoop_component"], detail["util_axial_component"])

    def test_classify_jacket_stress_governing_shear(self):
        gamma = 1.35
        sigy = 345e6
        shear = 90e6
        governing, detail = rfb.classify_jacket_stress_governing(
            5e6, 0.0, shear, sigy, gamma, 5e6, 0.0
        )
        self.assertEqual(governing, "shear / torsion (von Mises)")
        self.assertGreater(detail["vm_frac_shear"], 0.35)

    def test_lookup_rna_mass(self):
        turbines = rfb.load_turbine_lookup(rfb.DEFAULT_TURBINE_CSV)
        _, params = rfb.lookup_turbine_column(turbines, 15e6, rfb.DEFAULT_TURBINE_CSV)
        self.assertAlmostEqual(rfb.lookup_rna_mass(params), 1017000.0)
        self.assertTrue(math.isnan(rfb.lookup_rna_mass({})))

    def test_csv_column_order(self):
        self.assertEqual(
            JACKET_CSV_COLUMNS[:6],
            [
                "timestamp",
                "turbine_rating_W",
                "tower_base_shear_N",
                "tower_base_vertical_N",
                "tower_base_overturning_moment_Nm",
                "rna_mass_kg",
            ],
        )
        self.assertEqual(JACKET_CSV_COLUMNS[-1], "opt_exit_mode")
        self.assertNotIn("f1_Hz", JACKET_CSV_COLUMNS)
        self.assertNotIn("scour_protection_volume_m3", JACKET_CSV_COLUMNS)
        self.assertEqual(MONOPILE_CSV_COLUMNS[-1], "f1_Hz")
        self.assertEqual(MONOPILE_CSV_COLUMNS[-2], "opt_exit_mode")
        self.assertIn("soil_shear_modulus_Pa", MONOPILE_CSV_COLUMNS)
        self.assertNotIn("soil_shear_modulus_Pa", JACKET_CSV_COLUMNS)


if __name__ == "__main__":
    unittest.main()
