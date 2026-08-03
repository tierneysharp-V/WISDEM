"""WISDEM Monopile API."""

__author__ = ["Jake Nunemaker"]
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = "jake.nunemaker@nrel.gov"


# https://stackoverflow.com/questions/8391411/how-to-block-calls-to-print
import os
import sys
from warnings import warn

import numpy as np
import openmdao.api as om
from ORBIT import ProjectManager


# Typical rock density used when ORBIT reports scour mass but not volume
_SCOUR_ROCK_DENSITY_T_PER_M3 = 2.65


def _safe_float(value, default=np.nan):
    """Convert ORBIT values to float, returning default on failure."""
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _lookup_phase_time(phase_times, *keys):
    """Return the first matching phase duration from an ORBIT phase_times mapping."""
    if not isinstance(phase_times, dict):
        return np.nan
    lowered = {str(k).lower(): v for k, v in phase_times.items()}
    for key in keys:
        if key.lower() in lowered:
            return _safe_float(lowered[key.lower()])
    return np.nan


def _sum_matching_detailed(detailed, keywords):
    """Sum numeric detailed-output entries whose keys contain any keyword."""
    if not isinstance(detailed, dict):
        return np.nan
    total = 0.0
    found = False
    for key, val in detailed.items():
        key_l = str(key).lower()
        if any(kw in key_l for kw in keywords):
            try:
                total += float(val)
                found = True
            except (TypeError, ValueError):
                continue
    return total if found else np.nan


def _extract_scour_tonnes(project):
    """Pull scour protection mass from ORBIT design phase if present."""
    phases = getattr(project, "phases", None)
    if not isinstance(phases, dict):
        return np.nan
    for name, phase in phases.items():
        if "scour" not in str(name).lower():
            continue
        for attr in (
            "scour_protection_tonnes",
            "tonnes_per_substructure",
            "scour_protection_mass",
        ):
            if hasattr(phase, attr):
                return _safe_float(getattr(phase, attr))
        # Some ORBIT versions stash results on phase.design or similar
        design = getattr(phase, "design", None)
        if design is not None:
            for attr in ("scour_protection_tonnes", "tonnes_per_substructure"):
                if hasattr(design, attr):
                    return _safe_float(getattr(design, attr))
    return np.nan


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


class Orbit(om.Group):
    """Orbit class for WISDEM API."""

    def initialize(self):
        """Initializes the API connections."""
        self.options.declare("floating", default=False)
        self.options.declare("jacket", default=False)
        self.options.declare("jacket_legs", default=0)
        self.options.declare("quiet", default=False)

    def setup(self):
        """Define all input variables from all models."""
        self.set_input_defaults("wtiv", "example_wtiv")
        self.set_input_defaults("feeder", "example_feeder")
        # self.set_input_defaults("num_feeders", 1)
        # self.set_input_defaults("num_towing", 1)
        # self.set_input_defaults("num_station_keeping", 3)
        # self.set_input_defaults(
        #    "oss_install_vessel", "example_heavy_lift_vessel",
        # )
        self.set_input_defaults("site_distance", 40.0, units="km")
        self.set_input_defaults("site_distance_to_landfall", 40.0, units="km")
        self.set_input_defaults("interconnection_distance", 40.0, units="km")
        self.set_input_defaults("plant_turbine_spacing", 7)
        self.set_input_defaults("plant_row_spacing", 7)
        self.set_input_defaults("plant_substation_distance", 1, units="km")
        # self.set_input_defaults("num_port_cranes", 1)
        # self.set_input_defaults("num_assembly_lines", 1)
        self.set_input_defaults("takt_time", 170.0, units="h")
        self.set_input_defaults("port_cost_per_month", 2e6, units="USD/mo")
        self.set_input_defaults("construction_insurance", 44.0, units="USD/kW")
        self.set_input_defaults("construction_financing", 183.0, units="USD/kW")
        self.set_input_defaults("contingency", 316.0, units="USD/kW")
        self.set_input_defaults("commissioning_cost_kW", 44.0, units="USD/kW")
        self.set_input_defaults("decommissioning_cost_kW", 58.0, units="USD/kW")
        self.set_input_defaults("site_auction_price", 100e6, units="USD")
        self.set_input_defaults("site_assessment_cost", 50e6, units="USD")
        self.set_input_defaults("construction_plan_cost", 1e6, units="USD")
        self.set_input_defaults("installation_plan_cost", 2.5e5, units="USD")
        self.set_input_defaults("boem_review_cost", 0.0, units="USD")

        self.add_subsystem(
            "orbit",
            OrbitWisdem(
                floating=self.options["floating"],
                jacket=self.options["jacket"],
                jacket_legs=self.options["jacket_legs"],
            ),
            promotes=["*"],
        )


class OrbitWisdem(om.ExplicitComponent):
    """ORBIT-WISDEM Fixed Substructure API."""

    def initialize(self):
        """Initialize the API."""
        self.options.declare("floating", default=False)
        self.options.declare("jacket", default=False)
        self.options.declare("jacket_legs", default=0)
        self.options.declare("quiet", default=False)

    def setup(self):
        """Define all the inputs."""
        # Inputs
        # self.add_discrete_input(
        #     'weather_file',
        #     'block_island',
        #     desc='Weather file to use for installation times.'
        # )

        # Vessels
        self.add_discrete_input(
            "wtiv",
            "example_wtiv",
            desc=("Vessel configuration to use for installation of foundations" " and turbines."),
        )
        self.add_discrete_input(
            "feeder",
            "future_feeder",
            desc="Vessel configuration to use for (optional) feeder barges.",
        )
        self.add_discrete_input(
            "num_feeders",
            1,
            desc=("Number of feeder barges to use for installation of" " foundations and turbines."),
        )
        self.add_discrete_input(
            "num_towing",
            1,
            desc=(
                "Number of towing vessels to use for floating platforms that"
                " are assembled at port (with or without the turbine)."
            ),
        )
        self.add_discrete_input(
            "num_station_keeping",
            3,
            desc=("Number of station keeping or AHTS vessels that attach to floating" " platforms under tow-out."),
        )
        self.add_discrete_input(
            "oss_install_vessel",
            "example_heavy_lift_vessel",
            desc="Vessel configuration to use for installation of offshore substations.",  # noqa: E501
        )

        # Site
        self.add_input("site_depth", 40.0, units="m", desc="Site depth.")
        self.add_input(
            "site_distance",
            40.0,
            units="km",
            desc="Distance from site to installation port.",
        )
        self.add_input(
            "site_distance_to_landfall",
            50.0,
            units="km",
            desc="Distance from site to landfall for export cable.",
        )
        self.add_input(
            "interconnection_distance",
            3.0,
            units="km",
            desc="Distance from landfall to interconnection.",
        )
        self.add_input(
            "site_mean_windspeed",
            9.0,
            units="m/s",
            desc="Mean windspeed of the site.",
        )

        # Plant
        self.add_discrete_input(
            "number_of_turbines",
            60,
            desc="Number of turbines.",
        )
        self.add_input(
            "plant_turbine_spacing",
            7,
            desc="Turbine spacing in rotor diameters.",
        )
        self.add_input(
            "plant_row_spacing",
            7,
            desc="Row spacing in rotor diameters. Not used in ring layouts.",
        )
        self.add_input(
            "plant_substation_distance",
            1,
            units="km",
            desc="Distance from first turbine in string to substation.",
        )

        # Turbine
        self.add_input(
            "turbine_rating",
            8.0,
            units="MW",
            desc="Rated capacity of a turbine.",
        )
        self.add_input(
            "turbine_rated_windspeed",
            11.0,
            units="m/s",
            desc="Rated windspeed of the turbine.",
        )
        self.add_input(
            "turbine_capex",
            1100,
            units="USD/kW",
            desc="Turbine CAPEX",
        )
        self.add_input(
            "hub_height",
            100.0,
            units="m",
            desc="Turbine hub height.",
        )
        self.add_input(
            "turbine_rotor_diameter",
            130,
            units="m",
            desc="Turbine rotor diameter.",
        )
        self.add_input(
            "tower_mass",
            400.0,
            units="t",
            desc="mass of the total tower.",
        )
        self.add_input(
            "tower_length",
            100.0,
            units="m",
            desc="Total length of the tower.",
        )
        self.add_input(
            "tower_deck_space",
            25.0,
            units="m**2",
            desc=(
                "Deck space required to transport the tower. Defaults to 0 in"
                " order to not be a constraint on installation."
            ),
        )
        self.add_input(
            "nacelle_mass",
            500.0,
            units="t",
            desc="mass of the rotor nacelle assembly (RNA).",
        )
        self.add_input(
            "nacelle_deck_space",
            25.0,
            units="m**2",
            desc=(
                "Deck space required to transport the rotor nacelle assembly"
                " (RNA). Defaults to 0 in order to not be a constraint on"
                " installation."
            ),
        )
        self.add_discrete_input(
            "number_of_blades",
            3,
            desc="Number of blades per turbine.",
        )
        self.add_input(
            "blade_mass",
            50.0,
            units="t",
            desc="mass of an individual blade.",
        )
        self.add_input(
            "blade_deck_space",
            100.0,
            units="m**2",
            desc=(
                "Deck space required to transport a blade. Defaults to 0 in"
                " order to not be a constraint on installation."
            ),
        )

        # Mooring
        self.add_discrete_input(
            "num_mooring_lines",
            3,
            desc="Number of mooring lines per platform.",
        )
        self.add_input(
            "mooring_line_mass",
            1e4,
            units="kg",
            desc="Total mass of a mooring line",
        )
        self.add_input(
            "mooring_line_diameter",
            0.1,
            units="m",
            desc="Cross-sectional diameter of a mooring line",
        )
        self.add_input(
            "mooring_line_length",
            1e3,
            units="m",
            desc="Unstretched mooring line length",
        )
        self.add_input(
            "anchor_mass",
            1e4,
            units="kg",
            desc="Total mass of an anchor",
        )
        self.add_input(
            "mooring_line_cost",
            0.5e6,
            units="USD",
            desc="Mooring line unit cost.",
        )
        self.add_input(
            "mooring_anchor_cost",
            0.1e6,
            units="USD",
            desc="Mooring line unit cost.",
        )
        self.add_discrete_input(
            "anchor_type",
            "drag_embedment",
            desc="Number of mooring lines per platform.",
        )

        # Port
        self.add_input(
            "port_cost_per_month",
            2e6,
            units="USD/mo",
            desc="Monthly port costs.",
        )
        self.add_input(
            "takt_time",
            170.0,
            units="h",
            desc="Substructure assembly cycle time when doing assembly at the port.",  # noqa: E501
        )
        self.add_discrete_input(
            "num_assembly_lines",
            1,
            desc="Number of assembly lines used when assembly occurs at the port.",  # noqa: E501
        )
        self.add_discrete_input(
            "num_port_cranes",
            1,
            desc=(
                "Number of cranes used at the port to load feeders / WTIVS"
                " when assembly occurs on-site or assembly cranes when"
                " assembling at port."
            ),
        )

        # Floating Substructures
        self.add_input(
            "floating_substructure_cost",
            10e6,
            units="USD",
            desc="Floating substructure unit cost.",
        )

        # Monopile
        self.add_input(
            "monopile_length",
            100.0,
            units="m",
            desc="Length of monopile (including pile).",
        )
        self.add_input(
            "monopile_diameter",
            7.0,
            units="m",
            desc="Diameter of monopile.",
        )
        self.add_input(
            "monopile_mass",
            900.0,
            units="t",
            desc="mass of an individual monopile.",
        )
        self.add_input(
            "monopile_cost",
            4e6,
            units="USD",
            desc="Monopile unit cost.",
        )

        # Jacket
        self.add_input(
            "jacket_length",
            65.0,
            units="m",
            desc="Length/height of jacket (including pile/buckets).",
        )
        self.add_input(
            "jacket_mass",
            900.0,
            units="t",
            desc="mass of an individual jacket.",
        )
        self.add_input(
            "jacket_cost",
            4e6,
            units="USD",
            desc="Jacket unit cost.",
        )
        self.add_input(
            "jacket_r_foot",
            10.0,
            units="m",
            desc="Radius of jacket legs at base from centeroid.",
        )

        # Generic fixed-bottom
        self.add_input(
            "transition_piece_mass",
            250.0,
            units="t",
            desc="mass of an individual transition piece.",
        )
        self.add_input(
            "transition_piece_deck_space",
            25.0,
            units="m**2",
            desc=(
                "Deck space required to transport a transition piece."
                " Defaults to 0 in order to not be a constraint on"
                " installation."
            ),
        )
        self.add_input(
            "transition_piece_cost",
            1.5e6,
            units="USD",
            desc="Transition piece unit cost.",
        )

        # Project
        self.add_input(
            "construction_insurance",
            44.0,
            units="USD/kW",
            desc="Cost for construction insurance",
        )
        self.add_input(
            "construction_financing",
            183.0,
            units="USD/kW",
            desc="Cost for construction financing",
        )
        self.add_input(
            "contingency",
            316.0,
            units="USD/kW",
            desc="Cost in case of contingency",
        )
        self.add_input(
            "site_auction_price",
            100e6,
            units="USD",
            desc="Cost to secure site lease",
        )
        self.add_input(
            "site_assessment_cost",
            50e6,
            units="USD",
            desc="Cost to execute site assessment",
        )
        self.add_input(
            "construction_plan_cost",
            1e6,
            units="USD",
            desc="Cost to do construction planning",
        )
        self.add_input(
            "installation_plan_cost",
            2.5e5,
            units="USD",
            desc="Cost to do construction planning",
        )
        self.add_input(
            "boem_review_cost",
            0.0,
            units="USD",
            desc=("Cost for additional review by U.S. Dept of Interior Bureau" " of Ocean Energy Management (BOEM)"),
        )
        self.add_input("commissioning_cost_kW", 44.0, units="USD/kW", desc="Commissioning cost.")
        self.add_input("decommissioning_cost_kW", 58.0, units="USD/kW", desc="Decommissioning cost.")

        # Outputs
        # Totals
        self.add_output(
            "bos_capex",
            0.0,
            units="USD",
            desc="Sum of system and installation capex",
        )
        self.add_output(
            "soft_capex",
            0.0,
            units="USD",
            desc="Project costs associated with commissioning, decommissioning and financing",
        )
        self.add_output(
            "project_capex",
            0.0,
            units="USD",
            desc="costs associated with the lease area, "
            + "the development of the construction operations plan,"
            + "and any environmental review and other upfront project costs.",
        )
        self.add_output(
            "total_capex",
            0.0,
            units="USD",
            desc="Total capex of bos + soft + project",
        )
        self.add_output(
            "total_capex_kW",
            0.0,
            units="USD/kW",
            desc="Total capex of bos + soft + project per rated project capacity in kW",
        )
        self.add_output(
            "installation_time",
            0.0,
            units="h",
            desc="Total balance of system installation time.",
        )
        self.add_output(
            "installation_capex",
            0.0,
            units="USD",
            desc="Total balance of system installation cost.",
        )
        self.add_output(
            "capacity",
            0,
            units="MW",
            desc="Wind plant capacity, in MW.",
        )
        self.add_output(
            "foundation_installation_time",
            np.nan,
            units="h",
            desc="ORBIT phase time for monopile or jacket foundation installation.",
        )
        self.add_output(
            "turbine_installation_time",
            np.nan,
            units="h",
            desc="ORBIT phase time for turbine (WTG) installation.",
        )
        self.add_output(
            "pile_driving_time",
            np.nan,
            units="h",
            desc="ORBIT pile-driving duration when available from detailed outputs.",
        )
        self.add_output(
            "grouting_time",
            np.nan,
            units="h",
            desc="ORBIT grouting duration when available from detailed outputs.",
        )
        self.add_output(
            "scour_protection_tonnes",
            np.nan,
            units="t",
            desc="Scour protection mass per substructure from ORBIT design phase.",
        )
        self.add_output(
            "scour_protection_volume",
            np.nan,
            units="m**3",
            desc="Scour protection volume per substructure (from ORBIT or tonnes/density).",
        )
        self.add_output(
            "project_days",
            np.nan,
            units="d",
            desc="ORBIT project duration in days when available.",
        )
        self.add_output(
            "vessel_days",
            np.nan,
            units="d",
            desc="Vessel days derived from installation time or ORBIT project days.",
        )
        self.add_output(
            "weather_delay_time",
            np.nan,
            units="h",
            desc="Total weather delay time when exposed by ORBIT.",
        )
        self.add_output(
            "weather_downtime_factor",
            np.nan,
            desc="Weather delay time divided by total installation time when available.",
        )
        self.add_discrete_output(
            "layout",
            None,
            desc="Farm layout to be used by WOMBAT.",
        )

    def compile_orbit_config_file(
        self,
        inputs,
        outputs,
        discrete_inputs,
        discrete_outputs,
    ):
        """Compiles the ORBIT configuration dictionary."""

        floating_flag = self.options["floating"]
        jacket_flag = self.options["jacket"]

        config = {
            # Vessels
            "wtiv": ("floating_heavy_lift_vessel" if floating_flag else discrete_inputs["wtiv"]),
            "array_cable_install_vessel": "example_cable_lay_vessel",
            "array_cable_bury_vessel": "example_cable_lay_vessel",
            "export_cable_install_vessel": "example_cable_lay_vessel",
            "export_cable_bury_vessel": "example_cable_lay_vessel",
            # Site/plant
            "site": {
                "depth": float(inputs["site_depth"][0]),
                "distance": float(inputs["site_distance"][0]),
                "distance_to_landfall": float(inputs["site_distance_to_landfall"][0]),
                "mean_windspeed": float(inputs["site_mean_windspeed"][0]),
            },
            "plant": {
                "layout": "grid",
                "num_turbines": int(discrete_inputs["number_of_turbines"]),
                "row_spacing": float(inputs["plant_row_spacing"][0]),
                "turbine_spacing": float(inputs["plant_turbine_spacing"][0]),
                "substation_distance": float(inputs["plant_substation_distance"][0]),
            },
            # Turbine + components
            "turbine": {
                "hub_height": float(inputs["hub_height"][0]),
                "rotor_diameter": float(inputs["turbine_rotor_diameter"][0]),
                "turbine_rating": float(inputs["turbine_rating"][0]),
                "rated_windspeed": float(inputs["turbine_rated_windspeed"][0]),
                "tower": {
                    "type": "Tower",
                    "deck_space": float(inputs["tower_deck_space"][0]),
                    "mass": float(inputs["tower_mass"][0]),
                    "length": float(inputs["tower_length"][0]),
                },
                "nacelle": {
                    "type": "Nacelle",
                    "deck_space": float(inputs["nacelle_deck_space"][0]),
                    "mass": float(inputs["nacelle_mass"][0]),
                },
                "blade": {
                    "type": "Blade",
                    "number": int(discrete_inputs["number_of_blades"]),
                    "deck_space": float(inputs["blade_deck_space"][0]),
                    "mass": float(inputs["blade_mass"][0]),
                },
            },
            # Substructure components
            "transition_piece": {
                "type": "Transition Piece",
                "deck_space": float(inputs["transition_piece_deck_space"][0]),
                "mass": float(inputs["transition_piece_mass"][0]),
                "unit_cost": float(inputs["transition_piece_cost"][0]),
            },
            # Electrical
            "array_system_design": {
                "cables": ["XLPE_630mm_66kV", "XLPE_185mm_66kV"],
            },
            "export_system_design": {
                "cables": "XLPE_1000mm_220kV",
                "percent_added_length": 0.1,
                "landfall": {
                    "interconnection_distance": float(
                        inputs["interconnection_distance"][0],
                    ),
                },
            },
            # Phase Specific
            "OffshoreSubstationInstallation": {
                "oss_install_vessel": ("floating_heavy_lift_vessel" if floating_flag else "example_heavy_lift_vessel"),
                "feeder": ("floating_barge" if floating_flag else "future_feeder"),
                "num_feeders": int(discrete_inputs["num_feeders"]),
            },
            # Project development costs
            "project_parameters": {
                "construction_insurance": float(inputs["construction_insurance"][0]),
                "construction_financing": float(inputs["construction_financing"][0]),
                "installation_contingency": 0.5 * float(inputs["contingency"][0]),
                "procurement_contingency": 0.5 * float(inputs["contingency"][0]),
                "site_auction_price": float(inputs["site_auction_price"][0]),
                "site_assessment_cost": float(inputs["site_assessment_cost"][0]),
                "construction_plan_cost": float(inputs["construction_plan_cost"][0]),
                "installation_plan_cost": float(inputs["installation_plan_cost"][0]),
                "boem_review_cost": float(inputs["boem_review_cost"][0]),
                "commissioning": float(inputs["commissioning_cost_kW"][0]),
                "decommissioning": float(inputs["decommissioning_cost_kW"][0]),
                "turbine_capex": float(inputs["turbine_capex"][0]),
            },
            # Phases
            # Putting monopile or semisub here would override the inputs
            # we assume to get from WISDEM
            "design_phases": [
                # 'MonopileDesign',
                # 'SemiSubmersibleDesign',
                # 'MooringSystemDesign',
                # 'ScourProtectionDesign',
                "ArraySystemDesign",
                "ExportSystemDesign",
                "OffshoreSubstationDesign",
            ],
        }

        if "landfall" in config and "interconnection_distance" in config["landfall"]:
            warn(
                "landfall dictionary will be deprecated and moved" " into [export_system_design][landfall].",
                DeprecationWarning,
                stacklevel=2,
            )

        if "export_system_design" in config and "interconnection_distance" in config["export_system_design"]:
            warn(
                "[export_system][interconnection_distance] will be deprecated"
                " and moved to"
                " [export_system_design][landfall][interconnection_distance].",
                DeprecationWarning,
                stacklevel=2,
            )

        # Unique design phases
        if floating_flag:
            config["install_phases"] = {
                "ExportCableInstallation": 0,
                "OffshoreSubstationInstallation": 0,
                "MooringSystemInstallation": 0,
                "MooredSubInstallation": ("MooringSystemInstallation", 0.25),
                "ArrayCableInstallation": ("MooredSubInstallation", 0.25),
            }
        else:
            fixedStr = "JacketInstallation" if jacket_flag else "MonopileInstallation"

            if jacket_flag:
                monopile = config.get("monopile", {})
                monopile["diameter"] = 10
                config["monopile"] = monopile

            config["design_phases"] += ["ScourProtectionDesign"]
            config["install_phases"] = {
                "ExportCableInstallation": 0,
                "OffshoreSubstationInstallation": 0,
                "ArrayCableInstallation": 0,
                fixedStr: 0,
                "ScourProtectionInstallation": 0,
                "TurbineInstallation": (fixedStr, 0.25),
            }

        # Unique vessels
        if floating_flag:
            vessels = {
                "ahts_vessel": "example_ahts_vessel",
                "towing_vessel": "example_towing_vessel",
                "mooring_install_vessel": "example_support_vessel",
                "towing_vessel_groups": {
                    "towing_vessels": int(discrete_inputs["num_towing"]),
                    "ahts_vessels": int(discrete_inputs["num_station_keeping"]),
                },
            }
        else:
            vessels = {
                "feeder": discrete_inputs["feeder"],
                "num_feeders": int(discrete_inputs["num_feeders"]),
                "spi_vessel": "example_scour_protection_vessel",
            }
        config.update(vessels)

        # Unique support structure design/assembly
        if floating_flag:
            config["port"] = {
                "sub_assembly_lines": int(discrete_inputs["num_assembly_lines"]),
                "turbine_assembly_cranes": int(discrete_inputs["num_port_cranes"]),
                "monthly_rate": float(inputs["port_cost_per_month"][0]),
            }

            config["substructure"] = {
                "takt_time": float(inputs["takt_time"][0]),
                "unit_cost": float(inputs["floating_substructure_cost"][0]),
                "towing_speed": 6.0,  # km/h
            }

            anchorstr_in = discrete_inputs["anchor_type"].lower()
            if anchorstr_in.find("drag") >= 0:
                anchorstr = "Drag Embedment"
            elif anchorstr_in.find("suction") >= 0:
                anchorstr = "Suction Pile"

            config["mooring_system"] = {
                "num_lines": int(discrete_inputs["num_mooring_lines"]),
                "line_mass": 1e-3 * float(inputs["mooring_line_mass"][0]),
                "line_diam": float(inputs["mooring_line_diameter"][0]),
                "line_length": float(inputs["mooring_line_length"][0]),
                "line_cost": float(inputs["mooring_line_cost"][0]),
                "anchor_mass": 1e-3 * float(inputs["anchor_mass"][0]),
                "anchor_type": anchorstr,
                "anchor_cost": float(inputs["mooring_anchor_cost"][0]),
            }
        else:
            config["port"] = {
                "num_cranes": int(discrete_inputs["num_port_cranes"]),
                "monthly_rate": float(inputs["port_cost_per_month"][0]),
            }

            config["scour_protection_design"] = {
                "cost_per_tonne": 20,
            }

            if jacket_flag:
                config["jacket"] = {
                    "type": "Jacket",
                    "height": float(inputs["jacket_length"][0]),
                    "num_legs": int(self.options["jacket_legs"]),
                    "deck_space": 4 * float(inputs["jacket_r_foot"][0]) ** 2,
                    "mass": float(inputs["jacket_mass"][0]),
                    "unit_cost": float(inputs["jacket_cost"][0]),
                }
            else:
                config["monopile"] = {
                    "type": "Monopile",
                    "length": float(inputs["monopile_length"][0]),
                    "diameter": float(inputs["monopile_diameter"][0]),
                    "deck_space": 0.25 * float(inputs["monopile_diameter"][0] * inputs["monopile_length"][0]),
                    "mass": float(inputs["monopile_mass"][0]),
                    "unit_cost": float(inputs["monopile_cost"][0]),
                }

        self._orbit_config = config
        return config

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        """Creates and runs the project, then gathers the results."""
        quiet_flag = self.options["quiet"]

        config = self.compile_orbit_config_file(
            inputs,
            outputs,
            discrete_inputs,
            discrete_outputs,
        )

        project = ProjectManager(config)
        if quiet_flag:
            with HiddenPrints():
                project.run()
        else:
            project.run()

        # The ORBIT version of total_capex includes turbine capex, so we do our own sum of
        # the parts here that wisdem doesn't account for
        capacity_kW = 1e3 * inputs["turbine_rating"] * discrete_inputs["number_of_turbines"]
        outputs["bos_capex"] = project.bos_capex
        outputs["soft_capex"] = project.soft_capex
        outputs["project_capex"] = project.project_capex
        outputs["total_capex"] = project.bos_capex + project.soft_capex + project.project_capex
        outputs["total_capex_kW"] = outputs["total_capex"] / capacity_kW
        outputs["installation_time"] = project.installation_time
        outputs["installation_capex"] = project.installation_capex
        outputs["capacity"] = project.capacity
        if "ArraySystemDesign" in project.phases:
            discrete_outputs["layout"] = project.phases["ArraySystemDesign"].create_layout_df()
        elif "CustomArraySystemDesign" in project.phases:
            discrete_outputs["layout"] = project.phases["CustomArraySystemDesign"].create_layout_df()

        # Optional installation metrics (attribute names vary across ORBIT versions)
        phase_times = getattr(project, "phase_times", None)
        if phase_times is None:
            phase_times = getattr(project, "phase_times_hours", {})
        outputs["foundation_installation_time"] = _lookup_phase_time(
            phase_times,
            "MonopileInstallation",
            "JacketInstallation",
            "FixedSubstructureInstallation",
            "SubstructureInstallation",
        )
        outputs["turbine_installation_time"] = _lookup_phase_time(
            phase_times,
            "TurbineInstallation",
            "TurbineInstall",
        )

        detailed = getattr(project, "detailed_outputs", None)
        if detailed is None:
            detailed = getattr(project, "logs", {})
        outputs["pile_driving_time"] = _sum_matching_detailed(
            detailed, ("drive pile", "pile drive", "pile_drive", "driving")
        )
        outputs["grouting_time"] = _sum_matching_detailed(
            detailed, ("grout",)
        )

        scour_t = _extract_scour_tonnes(project)
        outputs["scour_protection_tonnes"] = scour_t
        if np.isfinite(scour_t) and scour_t >= 0.0:
            outputs["scour_protection_volume"] = scour_t / _SCOUR_ROCK_DENSITY_T_PER_M3
        else:
            outputs["scour_protection_volume"] = np.nan

        project_days = getattr(project, "project_days", None)
        if project_days is None:
            install_h = _safe_float(project.installation_time, default=np.nan)
            project_days = install_h / 24.0 if np.isfinite(install_h) else np.nan
        outputs["project_days"] = _safe_float(project_days)
        outputs["vessel_days"] = outputs["project_days"]

        weather_delay = getattr(project, "weather_delay", None)
        if weather_delay is None:
            weather_delay = getattr(project, "total_weather_delay", None)
        if weather_delay is None and isinstance(detailed, dict):
            weather_delay = _sum_matching_detailed(detailed, ("weather delay", "weather_delay", "weatherdelay"))
        outputs["weather_delay_time"] = _safe_float(weather_delay)
        install_h = _safe_float(project.installation_time, default=np.nan)
        if np.isfinite(outputs["weather_delay_time"]) and np.isfinite(install_h) and install_h > 0.0:
            outputs["weather_downtime_factor"] = outputs["weather_delay_time"] / install_h
        else:
            outputs["weather_downtime_factor"] = np.nan
