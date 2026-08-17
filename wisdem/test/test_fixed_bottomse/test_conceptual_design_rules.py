"""Unit tests for conceptual fixed-bottom design rules."""

import unittest

import numpy as np
import numpy.testing as npt

import wisdem.fixed_bottomse.conceptual_design_rules as cdr


class TestConceptualDesignRules(unittest.TestCase):
    def test_freeboard_height(self):
        # water_depth + 1.98*Hs + 0.2*Hs
        self.assertAlmostEqual(cdr.required_freeboard_height(30.0, 4.52), 30.0 + 2.18 * 4.52)

    def test_effective_pile_modulus(self):
        d, t, e = 8.0, 0.055, 200.0e9
        ratio = (d - 2.0 * t) / d
        expected = e * (1.0 - ratio**4)
        self.assertAlmostEqual(cdr.effective_pile_modulus(d, t, e), expected)

    def test_effective_soil_shear_modulus(self):
        self.assertAlmostEqual(cdr.effective_soil_shear_modulus(140.0e6, 0.4), 182.0e6)

    def test_embedment_length(self):
        d, t = 8.0, 0.055
        e_steel = 200.0e9
        g_soil = 140.0e6
        nu_soil = 0.4
        ee = cdr.effective_pile_modulus(d, t, e_steel)
        g_star = cdr.effective_soil_shear_modulus(g_soil, nu_soil)
        expected = d * (ee / g_star) ** (2.0 / 7.0)
        self.assertAlmostEqual(
            cdr.embedment_length(d, t, e_steel, g_soil, nu_soil),
            expected,
        )

    def test_monopile_min_wall_thickness(self):
        self.assertAlmostEqual(cdr.monopile_min_wall_thickness(10.0), 10.0 / 120.0)
        self.assertAlmostEqual(cdr.monopile_min_wall_thickness(6.0), 6.0 / 120.0)

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
        # H_free=45 → 3 bays; preserve three legs from geometry
        out = cdr.apply_jacket_topology(jacket, 45.0)
        self.assertEqual(out["n_legs"], 3)
        self.assertEqual(out["n_bays"], 3)
        self.assertEqual(out["height"], 45.0)
        self.assertEqual(len(out["bay_spacing"]), 4)
        self.assertEqual(len(out["brace_diameters"]), 3)
        self.assertEqual(len(out["brace_thicknesses"]), 3)

        # Default when geometry omits n_legs is four legs
        jacket.pop("n_legs")
        out_default = cdr.apply_jacket_topology(jacket, 45.0)
        self.assertEqual(out_default["n_legs"], 4)

        # Explicit override
        out_four = cdr.apply_jacket_topology(jacket, 45.0, n_legs=4)
        self.assertEqual(out_four["n_legs"], 4)

        # H_free=90 → 5 bays; seed from last brace size
        out2 = cdr.apply_jacket_topology(jacket, 90.0, n_legs=3)
        self.assertEqual(out2["n_bays"], 5)
        self.assertEqual(out2["n_legs"], 3)
        self.assertEqual(out2["brace_diameters"], [1.0] * 5)

    def test_validate_jacket_n_legs(self):
        self.assertEqual(cdr.validate_jacket_n_legs(3), 3)
        self.assertEqual(cdr.validate_jacket_n_legs(4), 4)
        with self.assertRaises(ValueError):
            cdr.validate_jacket_n_legs(5)

    def test_apply_monopile_geometry_provisional(self):
        mono = {
            "outer_shape": {"outer_diameter": {"grid": [0.0, 1.0], "values": [8.0, 8.0]}},
            "structure": {
                "layers": [
                    {
                        "name": "tower_wall",
                        "material": "steel",
                        "thickness": {"grid": [0.0, 1.0], "values": [0.055, 0.055]},
                    }
                ]
            },
            "reference_axis": {"z": {"grid": [0.0, 1.0], "values": [-55.0, 10.0]}},
        }
        wt = {
            "components": {"monopile": mono},
            "materials": [{"name": "steel", "E": 200.0e9}],
        }
        modeling = {"Environment": {"soil_shear_modulus": 140.0e6, "soil_poisson": 0.4}}
        l_emb = cdr.embedment_length_from_wt(wt, modeling)
        out = cdr.apply_monopile_geometry(mono, water_depth=30.0, hs=5.0, embedment=l_emb)
        # H_free=40.5; height=40.5 + l_emb; z_top=10.5
        h_free = cdr.required_freeboard_height(30.0, 5.0)
        height_in = h_free + l_emb
        self.assertAlmostEqual(out["reference_axis"]["z"]["values"][0], 10.5 - height_in)
        self.assertAlmostEqual(out["reference_axis"]["z"]["values"][-1], 10.5)


if __name__ == "__main__":
    unittest.main()
