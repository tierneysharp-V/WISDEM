"""Unit tests for conceptual fixed-bottom design rules."""

import unittest

import numpy as np
import numpy.testing as npt

import wisdem.fixed_bottomse.conceptual_design_rules as cdr


class TestConceptualDesignRules(unittest.TestCase):
    def test_freeboard_height(self):
        # water_depth + Hs + 1.5 + 4.0
        self.assertAlmostEqual(cdr.required_freeboard_height(30.0, 4.52), 40.02)

    def test_embedment_length(self):
        self.assertAlmostEqual(cdr.embedment_length(8.0), 40.0)
        self.assertAlmostEqual(cdr.embedment_length(6.0, factor=5.0), 30.0)

    def test_n_bays_from_height(self):
        self.assertEqual(cdr.n_bays_from_height(49.9), 3)
        self.assertEqual(cdr.n_bays_from_height(50.0), 4)
        self.assertEqual(cdr.n_bays_from_height(80.0), 4)
        self.assertEqual(cdr.n_bays_from_height(80.1), 5)

    def test_equal_bay_spacing(self):
        spacing = cdr.equal_bay_spacing(4)
        self.assertEqual(len(spacing), 5)
        # Mid-bay: linspace(0.5/5, 1-0.5/5, 5) == linspace(0.1, 0.9, 5)
        npt.assert_allclose(spacing, np.linspace(0.1, 0.9, 5))
        d = np.diff(spacing)
        npt.assert_allclose(d, d[0] * np.ones_like(d))
        # Must not coincide with leg-node fractions linspace(0,1,n+2)
        leg = np.linspace(0.0, 1.0, 6)
        self.assertTrue(np.all(np.min(np.abs(np.asarray(spacing)[:, None] - leg[None, :]), axis=1) > 1e-12))

    def test_monopile_reference_z(self):
        # WD=30, Hs=5 → H_free=40.5, z_top=10.5, L_emb=40 → height_in=80.5
        # z(0)=z_top-height=-70, z(1)=z_top=10.5
        z = cdr.monopile_reference_z(30.0, 5.0, 40.0, [0.0, 1.0])
        self.assertAlmostEqual(z[0], 10.5 - 80.5)
        self.assertAlmostEqual(z[-1], 10.5)
        # Embedment below mudline: mudline at -30, tip at z[0]
        self.assertAlmostEqual(-30.0 - z[0], 40.0)

    def test_apply_jacket_topology(self):
        jacket = {
            "height": 50.0,
            "n_legs": 3,
            "n_bays": 4,
            "brace_diameters": [1.0, 1.0, 1.0, 1.0],
            "brace_thicknesses": [0.1, 0.1, 0.1, 0.1],
            "bay_spacing": [0.1, 0.3, 0.5, 0.7, 0.9],
        }
        # H_free=45 → 3 bays
        out = cdr.apply_jacket_topology(jacket, 45.0)
        self.assertEqual(out["n_legs"], 4)
        self.assertEqual(out["n_bays"], 3)
        self.assertEqual(out["height"], 45.0)
        self.assertEqual(len(out["bay_spacing"]), 4)
        self.assertEqual(len(out["brace_diameters"]), 3)
        self.assertEqual(len(out["brace_thicknesses"]), 3)

        # H_free=90 → 5 bays; seed from last brace size
        out2 = cdr.apply_jacket_topology(jacket, 90.0)
        self.assertEqual(out2["n_bays"], 5)
        self.assertEqual(out2["brace_diameters"], [1.0] * 5)

    def test_apply_monopile_geometry_provisional(self):
        mono = {
            "outer_shape": {"outer_diameter": {"grid": [0.0, 1.0], "values": [8.0, 8.0]}},
            "reference_axis": {"z": {"grid": [0.0, 1.0], "values": [-55.0, 10.0]}},
        }
        out = cdr.apply_monopile_geometry(mono, water_depth=30.0, hs=5.0)
        # provisional emb = 5*8 = 40; H_free=40.5; height=80.5; z_top=10.5
        self.assertAlmostEqual(out["reference_axis"]["z"]["values"][0], 10.5 - 80.5)
        self.assertAlmostEqual(out["reference_axis"]["z"]["values"][-1], 10.5)


if __name__ == "__main__":
    unittest.main()
