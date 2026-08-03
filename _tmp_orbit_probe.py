"""Temporary probe of ORBIT ProjectManager attributes. Delete after use."""
from ORBIT import ProjectManager
import ORBIT
import inspect

print("ORBIT file:", ORBIT.__file__)
print("\n=== Public attrs/methods ===")
for a in sorted(dir(ProjectManager)):
    if not a.startswith("_"):
        print(a)

# Minimal monopile config similar to WISDEM
config = {
    "wtiv": "example_wtiv",
    "feeder": "future_feeder",
    "num_feeders": 1,
    "array_cable_install_vessel": "example_cable_lay_vessel",
    "array_cable_bury_vessel": "example_cable_lay_vessel",
    "export_cable_install_vessel": "example_cable_lay_vessel",
    "export_cable_bury_vessel": "example_cable_lay_vessel",
    "spi_vessel": "example_scour_protection_vessel",
    "site": {"depth": 40.0, "distance": 40.0, "distance_to_landfall": 50.0, "mean_windspeed": 9.0},
    "plant": {
        "layout": "grid",
        "num_turbines": 5,
        "row_spacing": 7,
        "turbine_spacing": 7,
        "substation_distance": 1,
    },
    "turbine": {
        "hub_height": 100.0,
        "rotor_diameter": 130,
        "turbine_rating": 8.0,
        "rated_windspeed": 11.0,
        "tower": {"type": "Tower", "deck_space": 25.0, "mass": 400.0, "length": 100.0},
        "nacelle": {"type": "Nacelle", "deck_space": 25.0, "mass": 500.0},
        "blade": {"type": "Blade", "number": 3, "deck_space": 100.0, "mass": 50.0},
    },
    "transition_piece": {
        "type": "Transition Piece",
        "deck_space": 25.0,
        "mass": 250.0,
        "unit_cost": 1.5e6,
    },
    "monopile": {
        "type": "Monopile",
        "length": 100.0,
        "diameter": 7.0,
        "deck_space": 175.0,
        "mass": 900.0,
        "unit_cost": 4e6,
    },
    "array_system_design": {"cables": ["XLPE_630mm_66kV", "XLPE_185mm_66kV"]},
    "export_system_design": {
        "cables": "XLPE_1000mm_220kV",
        "percent_added_length": 0.1,
        "landfall": {"interconnection_distance": 3.0},
    },
    "OffshoreSubstationInstallation": {
        "oss_install_vessel": "example_heavy_lift_vessel",
        "feeder": "future_feeder",
        "num_feeders": 1,
    },
    "project_parameters": {
        "construction_insurance": 44.0,
        "construction_financing": 183.0,
        "installation_contingency": 158.0,
        "procurement_contingency": 158.0,
        "site_auction_price": 100e6,
        "site_assessment_cost": 50e6,
        "construction_plan_cost": 1e6,
        "installation_plan_cost": 2.5e5,
        "boem_review_cost": 0.0,
        "commissioning": 44.0,
        "decommissioning": 58.0,
        "turbine_capex": 1100,
    },
    "design_phases": [
        "ArraySystemDesign",
        "ExportSystemDesign",
        "OffshoreSubstationDesign",
        "ScourProtectionDesign",
    ],
    "install_phases": {
        "ExportCableInstallation": 0,
        "OffshoreSubstationInstallation": 0,
        "ArrayCableInstallation": 0,
        "MonopileInstallation": 0,
        "ScourProtectionInstallation": 0,
        "TurbineInstallation": ("MonopileInstallation", 0.25),
    },
    "port": {"num_cranes": 1, "monthly_rate": 2e6},
    "scour_protection_design": {"cost_per_tonne": 20},
}

print("\n=== Running minimal project ===")
project = ProjectManager(config)
project.run()

interesting = [
    "bos_capex",
    "soft_capex",
    "project_capex",
    "total_capex",
    "installation_time",
    "installation_capex",
    "capacity",
    "phase_times",
    "project_days",
    "vessel_days",
    "scour_protection",
    "scour_protection_tonnes",
    "scour_protection_cost",
    "weather_delay",
    "downtime",
    "system_capex",
    "installation_capex_dict",
    "capex_breakdown",
    "detailed_outputs",
    "phase_dates",
]

print("\n=== Interesting attrs present ===")
for name in interesting:
    if hasattr(project, name):
        val = getattr(project, name)
        t = type(val).__name__
        s = str(val)
        if len(s) > 300:
            s = s[:300] + "..."
        print(f"{name}: ({t}) {s}")
    else:
        print(f"{name}: MISSING")

print("\n=== All non-callable instance attrs (sample) ===")
for name in sorted(vars(project).keys()):
    if name.startswith("_"):
        continue
    val = getattr(project, name)
    if callable(val):
        continue
    t = type(val).__name__
    s = str(val)
    if len(s) > 200:
        s = s[:200] + "..."
    print(f"{name}: ({t}) {s}")

print("\n=== phases keys ===")
print(list(project.phases.keys()))

# ScourProtectionDesign outputs
if "ScourProtectionDesign" in project.phases:
    spd = project.phases["ScourProtectionDesign"]
    print("\n=== ScourProtectionDesign attrs ===")
    for name in sorted(dir(spd)):
        if name.startswith("_"):
            continue
        try:
            val = getattr(spd, name)
        except Exception as e:
            print(f"{name}: ERR {e}")
            continue
        if callable(val):
            continue
        t = type(val).__name__
        s = str(val)
        if len(s) > 200:
            s = s[:200] + "..."
        print(f"{name}: ({t}) {s}")

# MonopileInstallation phase times if any
if "MonopileInstallation" in project.phases:
    mi = project.phases["MonopileInstallation"]
    print("\n=== MonopileInstallation interesting ===")
    for name in sorted(dir(mi)):
        if any(k in name.lower() for k in ["time", "day", "grout", "drive", "mobil", "weather", "vessel", "phase"]):
            try:
                val = getattr(mi, name)
            except Exception:
                continue
            if callable(val):
                continue
            t = type(val).__name__
            s = str(val)
            if len(s) > 200:
                s = s[:200] + "..."
            print(f"{name}: ({t}) {s}")
