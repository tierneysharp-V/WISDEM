import unittest

import numpy as np
import openmdao.api as om
#import numpy.testing as npt

#from wisdem.commonse import gravity as g
from wisdem.fixed_bottomse.jacket import JacketSE, jacket_element_metadata, jacket_n_elem, jacket_global_buckling_k_factors


class Test(unittest.TestCase):
    def test_element_metadata_count(self):
        for n_legs, n_bays, mud in ((3, 2, True), (4, 3, True), (4, 3, False)):
            n_struct = jacket_n_elem(n_legs, n_bays, mud) - n_legs
            meta = jacket_element_metadata(n_legs, n_bays, mud)
            self.assertEqual(len(meta), n_struct)
            self.assertTrue(all(m["member_type"] in ("leg", "brace") for m in meta))

    def test_global_buckling_k_factors(self):
        k = jacket_global_buckling_k_factors(4, 3, True)
        meta = jacket_element_metadata(4, 3, True)
        for ki, m in zip(k, meta):
            if m["member_type"] == "leg":
                self.assertAlmostEqual(ki, 1.0)
            else:
                self.assertAlmostEqual(ki, 0.8)

    def testAll(self):
        modeling_options = {}
        modeling_options["WISDEM"] = {}
        modeling_options["WISDEM"]["n_dlc"] = 1
        modeling_options["WISDEM"]["FixedBottomSE"] = {}
        modeling_options["WISDEM"]["FixedBottomSE"]["mud_brace"] = True
        modeling_options["WISDEM"]["FixedBottomSE"]["n_legs"] = 3
        modeling_options["WISDEM"]["FixedBottomSE"]["n_bays"] = 2
        modeling_options["WISDEM"]["FixedBottomSE"]["gamma_f"] = 1.0
        modeling_options["WISDEM"]["FixedBottomSE"]["gamma_m"] = 1.0
        modeling_options["WISDEM"]["FixedBottomSE"]["gamma_n"] = 1.0
        modeling_options["WISDEM"]["FixedBottomSE"]["gamma_b"] = 1.0
        modeling_options["WISDEM"]["FixedBottomSE"]["material"] = "steel"
        modeling_options["WISDEM"]["FixedBottomSE"]["frame3dd"] = {}
        modeling_options["WISDEM"]["FixedBottomSE"]["frame3dd"]["shear"] = 1
        modeling_options["WISDEM"]["FixedBottomSE"]["frame3dd"]["geom"] = 1
        modeling_options["WISDEM"]["FixedBottomSE"]["save_truss_figures"] = False
        modeling_options["materials"] = {}
        modeling_options["materials"]["n_mat"] = 1

        prob = om.Problem(reports=False, model=JacketSE(modeling_options=modeling_options))

        prob.setup()

        prob["material_names"] = ["steel"]
        prob["E_mat"] = 1e10 * np.ones((1, 3))
        prob["G_mat"] = 1e8 * np.ones((1, 3))
        prob["rho_mat"] = 1e4
        prob["sigma_y_mat"] = 1e8

        prob.run_model()

    def testMemberWind(self):
        modeling_options = {}
        modeling_options["WISDEM"] = {}
        modeling_options["WISDEM"]["n_dlc"] = 1
        modeling_options["WISDEM"]["FixedBottomSE"] = {}
        modeling_options["WISDEM"]["FixedBottomSE"]["mud_brace"] = True
        modeling_options["WISDEM"]["FixedBottomSE"]["n_legs"] = 3
        modeling_options["WISDEM"]["FixedBottomSE"]["n_bays"] = 2
        modeling_options["WISDEM"]["FixedBottomSE"]["gamma_f"] = 1.0
        modeling_options["WISDEM"]["FixedBottomSE"]["gamma_m"] = 1.0
        modeling_options["WISDEM"]["FixedBottomSE"]["gamma_n"] = 1.0
        modeling_options["WISDEM"]["FixedBottomSE"]["gamma_b"] = 1.0
        modeling_options["WISDEM"]["FixedBottomSE"]["material"] = "steel"
        modeling_options["WISDEM"]["FixedBottomSE"]["member_wind"] = True
        modeling_options["WISDEM"]["FixedBottomSE"]["frame3dd"] = {}
        modeling_options["WISDEM"]["FixedBottomSE"]["frame3dd"]["shear"] = 1
        modeling_options["WISDEM"]["FixedBottomSE"]["frame3dd"]["geom"] = 1
        modeling_options["WISDEM"]["FixedBottomSE"]["save_truss_figures"] = False
        modeling_options["materials"] = {}
        modeling_options["materials"]["n_mat"] = 1

        prob = om.Problem(reports=False, model=JacketSE(modeling_options=modeling_options))
        prob.setup()

        prob["material_names"] = ["steel"]
        prob["E_mat"] = 1e10 * np.ones((1, 3))
        prob["G_mat"] = 1e8 * np.ones((1, 3))
        prob["rho_mat"] = 1e4
        prob["sigma_y_mat"] = 1e8
        prob["foot_head_ratio"] = 1.5
        prob["r_head"] = 5.0
        prob["height"] = 70.0
        prob["leg_diameter"] = 1.5
        prob["leg_thickness"] = 0.05
        prob["brace_diameters"] = np.array([1.0, 1.0])
        prob["brace_thicknesses"] = np.array([0.04, 0.04])
        prob["bay_spacing"] = np.array([0.25, 0.5, 0.75])
        prob["Uref"] = 11.0
        prob["wind_reference_height"] = 90.0
        prob["rho_air"] = 1.225
        prob["mu_air"] = 1.81e-5
        prob["shearExp"] = 0.1

        prob.run_model()

        n_elem = 2 * (3 * (2 + 1)) + 4 * (3 * 2) + 3 + 3
        px1 = prob.get_val("jacket_elem_Px1", units="N/m")
        self.assertEqual(px1.shape[0], n_elem)
        self.assertGreater(np.max(np.abs(px1[: n_elem - 3])), 0.0)

    def testMemberHydro(self):
        modeling_options = {}
        modeling_options["WISDEM"] = {}
        modeling_options["WISDEM"]["n_dlc"] = 1
        modeling_options["WISDEM"]["FixedBottomSE"] = {}
        modeling_options["WISDEM"]["FixedBottomSE"]["mud_brace"] = True
        modeling_options["WISDEM"]["FixedBottomSE"]["n_legs"] = 3
        modeling_options["WISDEM"]["FixedBottomSE"]["n_bays"] = 2
        modeling_options["WISDEM"]["FixedBottomSE"]["gamma_f"] = 1.0
        modeling_options["WISDEM"]["FixedBottomSE"]["gamma_m"] = 1.0
        modeling_options["WISDEM"]["FixedBottomSE"]["gamma_n"] = 1.0
        modeling_options["WISDEM"]["FixedBottomSE"]["gamma_b"] = 1.0
        modeling_options["WISDEM"]["FixedBottomSE"]["material"] = "steel"
        modeling_options["WISDEM"]["FixedBottomSE"]["member_hydro"] = True
        modeling_options["WISDEM"]["FixedBottomSE"]["frame3dd"] = {}
        modeling_options["WISDEM"]["FixedBottomSE"]["frame3dd"]["shear"] = 1
        modeling_options["WISDEM"]["FixedBottomSE"]["frame3dd"]["geom"] = 1
        modeling_options["WISDEM"]["FixedBottomSE"]["save_truss_figures"] = False
        modeling_options["materials"] = {}
        modeling_options["materials"]["n_mat"] = 1

        prob = om.Problem(reports=False, model=JacketSE(modeling_options=modeling_options))
        prob.setup()

        prob["material_names"] = ["steel"]
        prob["E_mat"] = 1e10 * np.ones((1, 3))
        prob["G_mat"] = 1e8 * np.ones((1, 3))
        prob["rho_mat"] = 1e4
        prob["sigma_y_mat"] = 1e8
        prob["foot_head_ratio"] = 1.5
        prob["r_head"] = 5.0
        prob["height"] = 70.0
        prob["leg_diameter"] = 1.5
        prob["leg_thickness"] = 0.05
        prob["brace_diameters"] = np.array([1.0, 1.0])
        prob["brace_thicknesses"] = np.array([0.04, 0.04])
        prob["bay_spacing"] = np.array([0.25, 0.5, 0.75])
        prob["water_depth"] = 40.0
        prob["rho_water"] = 1025.0
        prob["mu_water"] = 1.08e-3
        prob["Hsig_wave"] = 6.0
        prob["Tsig_wave"] = 10.0

        prob.run_model()

        n_elem = 2 * (3 * (2 + 1)) + 4 * (3 * 2) + 3 + 3
        px1 = prob.get_val("jacket_elem_Px1", units="N/m")
        self.assertEqual(px1.shape[0], n_elem)
        self.assertGreater(np.max(np.abs(px1[: n_elem - 3])), 0.0)

    def testMemberWindHydro(self):
        modeling_options = {}
        modeling_options["WISDEM"] = {}
        modeling_options["WISDEM"]["n_dlc"] = 1
        modeling_options["WISDEM"]["FixedBottomSE"] = {}
        modeling_options["WISDEM"]["FixedBottomSE"]["mud_brace"] = True
        modeling_options["WISDEM"]["FixedBottomSE"]["n_legs"] = 3
        modeling_options["WISDEM"]["FixedBottomSE"]["n_bays"] = 2
        modeling_options["WISDEM"]["FixedBottomSE"]["gamma_f"] = 1.0
        modeling_options["WISDEM"]["FixedBottomSE"]["gamma_m"] = 1.0
        modeling_options["WISDEM"]["FixedBottomSE"]["gamma_n"] = 1.0
        modeling_options["WISDEM"]["FixedBottomSE"]["gamma_b"] = 1.0
        modeling_options["WISDEM"]["FixedBottomSE"]["material"] = "steel"
        modeling_options["WISDEM"]["FixedBottomSE"]["member_wind"] = True
        modeling_options["WISDEM"]["FixedBottomSE"]["member_hydro"] = True
        modeling_options["WISDEM"]["FixedBottomSE"]["frame3dd"] = {}
        modeling_options["WISDEM"]["FixedBottomSE"]["frame3dd"]["shear"] = 1
        modeling_options["WISDEM"]["FixedBottomSE"]["frame3dd"]["geom"] = 1
        modeling_options["WISDEM"]["FixedBottomSE"]["save_truss_figures"] = False
        modeling_options["materials"] = {}
        modeling_options["materials"]["n_mat"] = 1

        prob = om.Problem(reports=False, model=JacketSE(modeling_options=modeling_options))
        prob.setup()

        prob["material_names"] = ["steel"]
        prob["E_mat"] = 1e10 * np.ones((1, 3))
        prob["G_mat"] = 1e8 * np.ones((1, 3))
        prob["rho_mat"] = 1e4
        prob["sigma_y_mat"] = 1e8
        prob["foot_head_ratio"] = 1.5
        prob["r_head"] = 5.0
        prob["height"] = 70.0
        prob["leg_diameter"] = 1.5
        prob["leg_thickness"] = 0.05
        prob["brace_diameters"] = np.array([1.0, 1.0])
        prob["brace_thicknesses"] = np.array([0.04, 0.04])
        prob["bay_spacing"] = np.array([0.25, 0.5, 0.75])
        prob["Uref"] = 11.0
        prob["wind_reference_height"] = 90.0
        prob["rho_air"] = 1.225
        prob["mu_air"] = 1.81e-5
        prob["shearExp"] = 0.1
        prob["water_depth"] = 40.0
        prob["rho_water"] = 1025.0
        prob["mu_water"] = 1.08e-3
        prob["Hsig_wave"] = 6.0
        prob["Tsig_wave"] = 10.0

        prob.run_model()


if __name__ == "__main__":
    unittest.main()
