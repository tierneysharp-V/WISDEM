"""Unit tests for foundation_batch helpers (no full WISDEM run required)."""

from __future__ import annotations

import math
import os
import sys
import tempfile
import unittest

import numpy as np

BATCH_DIR = os.path.dirname(os.path.realpath(__file__))
EXAMPLES_DIR = os.path.dirname(BATCH_DIR)
REPO_ROOT = os.path.dirname(EXAMPLES_DIR)
sys.path.insert(0, BATCH_DIR)
sys.path.insert(0, EXAMPLES_DIR)
sys.path.insert(0, REPO_ROOT)

import run_foundation_batch as rfb  # noqa: E402
from extract_foundation_results import (  # noqa: E402
    JACKET_CSV_COLUMNS,
    LCOE_CSV_COLUMNS,
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
        for path in (rfb.MODELING_ROTORSE, rfb.MODELING_LCOE, rfb.ANALYSIS_LOADS, rfb.DEFAULT_TURBINE_CSV):
            self.assertTrue(os.path.isfile(path), path)

    def test_read_scenarios_template(self):
        rows = rfb.read_scenarios(os.path.join(BATCH_DIR, "scenarios_template.csv"))
        self.assertEqual(len(rows), 4)
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
        prefix = os.path.join(BATCH_DIR, "DOE_results")
        self.assertEqual(
            csv_path_for_foundation_type("jacket", mo4, results_prefix=prefix),
            os.path.join(BATCH_DIR, "DOE_results_jacket.csv"),
        )
        self.assertEqual(
            csv_path_for_foundation_type("jacket", mo3, results_prefix=prefix),
            os.path.join(BATCH_DIR, "DOE_results_jacket_3.csv"),
        )
        self.assertEqual(
            csv_path_for_foundation_type("monopile", results_prefix=prefix),
            os.path.join(BATCH_DIR, "DOE_results_monopile.csv"),
        )

    def test_resolve_results_prefix(self):
        self.assertIsNone(rfb.resolve_results_prefix())
        self.assertEqual(rfb.resolve_results_prefix(doe_results=True), rfb.DOE_RESULTS_PREFIX)
        self.assertEqual(
            rfb.resolve_results_prefix(results_prefix="DOE_results"),
            os.path.join(BATCH_DIR, "DOE_results"),
        )

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
        self.assertIn("leg_diameter_m", JACKET_CSV_COLUMNS)
        self.assertIn("leg_thickness_m", JACKET_CSV_COLUMNS)
        self.assertIn("brace_diameter_m", JACKET_CSV_COLUMNS)
        self.assertIn("brace_thickness_m", JACKET_CSV_COLUMNS)
        self.assertLess(
            JACKET_CSV_COLUMNS.index("leg_thickness_m"),
            JACKET_CSV_COLUMNS.index("brace_diameter_m"),
        )

    def test_merge_sized_foundation_replaces_seed_and_drops_other(self):
        full = {
            "components": {
                "blade": {"name": "blade"},
                "tower": {"name": "tower"},
                "monopile": {"diameter": 6.0},
                "jacket": {"leg_diameter": 1.0},
            }
        }
        sized = {"components": {"jacket": {"leg_diameter": 2.4, "height": 55.0}}}
        out = rfb.merge_sized_foundation(full, sized, "jacket")
        self.assertEqual(out["components"]["jacket"]["leg_diameter"], 2.4)
        self.assertEqual(out["components"]["jacket"]["height"], 55.0)
        self.assertIn("blade", out["components"])
        self.assertIn("tower", out["components"])
        self.assertNotIn("monopile", out["components"])

    def test_apply_jacket_mass_user(self):
        wt = {"components": {"jacket": {"leg_diameter": 2.0}}}
        out = rfb.apply_jacket_mass_user(wt, 1.2e6)
        self.assertEqual(out["components"]["jacket"]["jacket_mass_user"], 1.2e6)
        self.assertIsNone(rfb.apply_jacket_mass_user(wt, None)["components"]["jacket"].get("jacket_mass_user"))

    def test_find_sized_geometry_prefers_sized_yaml(self):
        tag = "jacket_row2_legs4"
        with tempfile.TemporaryDirectory() as tmp:
            old_tmp = rfb.TMP_DIR
            rfb.TMP_DIR = tmp
            try:
                sized = os.path.join(tmp, f"geometry_{tag}_sized.yaml")
                stage2 = os.path.join(tmp, f"outputs_{tag}_stage2")
                os.makedirs(stage2)
                with open(sized, "w", encoding="utf-8") as f:
                    f.write("name: sized\n")
                with open(os.path.join(stage2, "refturb_output.yaml"), "w", encoding="utf-8") as f:
                    f.write("name: stage2\n")
                self.assertEqual(rfb.find_sized_geometry(tag, "jacket"), sized)
            finally:
                rfb.TMP_DIR = old_tmp

    def test_build_lcoe_modeling_enables_bos_and_patches_site(self):
        scenario = {
            "water_depth_m": "45",
            "Hs50_m": "8",
            "soil_shear_modulus_Pa": "1.4e8",
            "V50_mps": "70",
        }
        modeling = rfb.build_lcoe_modeling("jacket", scenario)
        self.assertTrue(modeling["WISDEM"]["BOS"]["flag"])
        self.assertTrue(modeling["WISDEM"]["LCOE"]["flag"])
        self.assertFalse(modeling["WISDEM"]["OpEx"]["flag"])
        self.assertFalse(modeling["WISDEM"]["FixedBottomSE"].get("soil_springs", True))
        self.assertEqual(modeling["Environment"]["water_depth"], 45.0)
        self.assertAlmostEqual(modeling["Environment"]["V_mean"], 10.0)

    def test_parked_cd_is_mw_class_iec_mapped_disk_value(self):
        # 1.5 * planform solidity 0.03-0.05 → 0.045-0.075 (not small-rotor 0.12-0.20)
        self.assertGreaterEqual(rfb.PARKED_ROTOR_CD, 0.04)
        self.assertLessEqual(rfb.PARKED_ROTOR_CD, 0.08)
        self.assertAlmostEqual(rfb.PARKED_ROTOR_CD, 0.05)

    def test_extreme_wave_period_iec_61400_3_midpoint(self):
        hs = 8.0
        t = rfb.extreme_wave_period(hs)
        t_lo = 11.1 * math.sqrt(hs / rfb._G)
        t_hi = 14.3 * math.sqrt(hs / rfb._G)
        self.assertAlmostEqual(t, 12.7 * math.sqrt(hs / rfb._G))
        self.assertGreater(t, t_lo)
        self.assertLess(t, t_hi)
        with self.assertRaises(ValueError):
            rfb.extreme_wave_period(0.0)

    def test_patch_environment_sets_wave_period_from_hs(self):
        scenario = {
            "water_depth_m": "45",
            "Hs50_m": "8",
            "soil_shear_modulus_Pa": "1.4e8",
            "V50_mps": "70",
        }
        modeling = rfb.patch_environment({"Environment": {}}, scenario, for_rotorse=False)
        self.assertAlmostEqual(modeling["Environment"]["significant_wave_height"], 8.0)
        self.assertAlmostEqual(
            modeling["Environment"]["significant_wave_period"],
            rfb.extreme_wave_period(8.0),
        )

    def test_lcoe_csv_columns(self):
        self.assertIn("lcoe_usd_per_mwh", LCOE_CSV_COLUMNS)
        self.assertIn("bos_source", LCOE_CSV_COLUMNS)
        self.assertIn("foundation_type", LCOE_CSV_COLUMNS)

    def test_read_scenarios_jacket_without_soil_or_depth(self):
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8") as f:
            f.write("turbine_rating_W,V50_mps,Hs50_m\n1e7,40,8\n")
            path = f.name
        try:
            rows = rfb.read_scenarios(path)
            self.assertEqual(len(rows), 1)
            self.assertNotIn("soil_shear_modulus_Pa", rows[0])
            self.assertNotIn("water_depth_m", rows[0])
        finally:
            os.unlink(path)

    def test_scenario_result_extras_optional_site_columns(self):
        loads = {"shear_N": 1.0, "vertical_N": 2.0, "overturning_Nm": 3.0}
        extras = rfb.scenario_result_extras(
            {"turbine_rating_W": "1e7", "V50_mps": "40", "Hs50_m": "8"},
            loads,
        )
        self.assertNotIn("soil_shear_modulus_Pa", extras)
        self.assertNotIn("water_depth_m", extras)
        extras_full = rfb.scenario_result_extras(
            {
                "turbine_rating_W": "1e7",
                "V50_mps": "40",
                "Hs50_m": "8",
                "soil_shear_modulus_Pa": "1.4e8",
                "water_depth_m": "45",
            },
            loads,
        )
        self.assertAlmostEqual(extras_full["soil_shear_modulus_Pa"], 1.4e8)
        self.assertAlmostEqual(extras_full["water_depth_m"], 45.0)

    def test_patch_environment_keeps_yaml_defaults_when_site_cols_omitted(self):
        modeling = rfb.patch_environment(
            {"Environment": {"water_depth": 30.0, "soil_shear_modulus": 1.4e8}},
            {"turbine_rating_W": "1e7", "V50_mps": "40", "Hs50_m": "8"},
            for_rotorse=False,
        )
        self.assertEqual(modeling["Environment"]["water_depth"], 30.0)
        self.assertEqual(modeling["Environment"]["soil_shear_modulus"], 1.4e8)
        self.assertAlmostEqual(modeling["Environment"]["significant_wave_height"], 8.0)

    def test_jacket_results_include_leg_and_brace_sizes(self):
        from extract_foundation_results import collect_foundation_row

        class _FakeOpt:
            def get_val(self, name, units=None):
                data = {
                    "fixedse.jacket_mass": 1000.0,
                    "jacket.transition_piece_mass": 100.0,
                    "jacket.height": 50.0,
                    "fixedse.r_foot": 12.0,
                    "jacket.leg_diameter": 1.5,
                    "jacket.leg_thickness": 0.04,
                    "jacket.brace_diameters": np.array([0.8, 0.9]),
                    "jacket.brace_thicknesses": np.array([0.03, 0.035]),
                }
                return data[name]

        modeling = {"flags": {"jacket": True}, "WISDEM": {"FixedBottomSE": {"n_legs": 4}}}
        row = collect_foundation_row(_FakeOpt(), modeling)
        self.assertAlmostEqual(row["leg_diameter_m"], 1.5)
        self.assertAlmostEqual(row["leg_thickness_m"], 0.04)
        self.assertAlmostEqual(row["brace_diameter_m"], 0.8)
        self.assertAlmostEqual(row["brace_thickness_m"], 0.03)

    def test_doe_scenario_files_split_by_foundation(self):
        mono = os.path.join(BATCH_DIR, "DOE_scenarios_500_monopile.csv")
        jacket = os.path.join(BATCH_DIR, "DOE_scenarios_500_jacket.csv")
        self.assertTrue(os.path.isfile(mono), mono)
        self.assertTrue(os.path.isfile(jacket), jacket)
        mono_rows = rfb.read_scenarios(mono)
        jacket_rows = rfb.read_scenarios(jacket)
        self.assertGreater(len(mono_rows), 0)
        self.assertGreater(len(jacket_rows), 0)
        self.assertIn("soil_shear_modulus_Pa", mono_rows[0])
        self.assertNotIn("water_depth_m", mono_rows[0])
        self.assertNotIn("soil_shear_modulus_Pa", jacket_rows[0])
        self.assertNotIn("water_depth_m", jacket_rows[0])
        v50_mono = {float(r["V50_mps"]) for r in mono_rows}
        v50_jacket = {float(r["V50_mps"]) for r in jacket_rows}
        hs_mono = {float(r["Hs50_m"]) for r in mono_rows}
        hs_jacket = {float(r["Hs50_m"]) for r in jacket_rows}
        self.assertEqual(min(v50_mono), 30.0)
        self.assertEqual(max(v50_mono), 70.0)
        self.assertEqual(len(v50_mono), 12)
        self.assertEqual(v50_mono, v50_jacket)
        self.assertEqual(hs_mono, hs_jacket)
        self.assertEqual(len(hs_mono), 4)
        self.assertEqual(len(mono_rows), 192)
        self.assertEqual(len(jacket_rows), 192)
        self.assertEqual(
            {float(r["soil_shear_modulus_Pa"]) for r in mono_rows},
            {10000000.0, 150000000.0},
        )


if __name__ == "__main__":
    unittest.main()
