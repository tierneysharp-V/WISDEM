"""
Conceptual design rules for fixed-bottom support structures.

These helpers set topology and freeboard height from site parameters so that
WISDEM optimisation sizes member diameters/thicknesses only. They are
intentionally simple and isolated so thresholds can be changed later without
touching Frame3DD or PoseOptimization.

Engineering rules (conceptual design):
  H_free = water_depth + 1.98*Hs + 0.2*Hs
  L_embedment = D * (Ee/G*)^(2/7)   (monopile, after sizing; D,t at mudline)
    Ee = E_steel * [1 - ((D - 2t)/D)^4]
    G* = G_soil * (1 + 3/4 * nu_soil)
  t_min = D / 120   (monopile wall thickness [m], D in m)
  n_bays from jacket freeboard height thresholds
  equal bay spacing from n_bays
  jackets use four legs
"""

from __future__ import annotations

from typing import Any

import copy

import numpy as np

# ---------------------------------------------------------------------------
# Tunable conceptual-design constants (edit here if rules change)
# ---------------------------------------------------------------------------
HSIG_CLEARANCE_PRIMARY = 1.98
HSIG_CLEARANCE_SECONDARY = 0.2
DEFAULT_SOIL_SHEAR_MODULUS_PA = 140.0e6
DEFAULT_SOIL_POISSON = 0.4
MONOPILE_WALL_T_D_TO_T_RATIO = 120.0  # t >= D / ratio
N_LEGS_JACKET = 4

# Jacket freeboard height [m] → number of bays
BAY_HEIGHT_THRESHOLDS_M = (50.0, 80.0)  # <50 → 3; 50–80 → 4; >80 → 5


def wave_clearance_above_mwl(hs: float) -> float:
    """Transition elevation above MSL from significant wave height."""
    h = float(hs)
    return HSIG_CLEARANCE_PRIMARY * h + HSIG_CLEARANCE_SECONDARY * h


def required_freeboard_height(water_depth: float, hs: float) -> float:
    """
    Mudline-to-top freeboard height (jacket height; monopile span above seabed).

    Matches the WISDEM height_airgap constraint:
        height >= water_depth + 1.98*Hs + 0.2*Hs
    """
    return float(water_depth) + wave_clearance_above_mwl(hs)


def effective_pile_modulus(diameter: float, thickness: float, youngs_modulus: float) -> float:
    """
    Effective pile Young's modulus Ee for the embedment rule.

    Ee = E_steel * [J_outer - J_inner] / J_outer with J = D^4 * pi / 64.
    """
    d = float(diameter)
    t = float(thickness)
    e = float(youngs_modulus)
    if d <= 0.0:
        raise ValueError(f"diameter must be > 0, got {d}")
    if d <= 2.0 * t:
        raise ValueError(f"diameter must exceed 2*thickness, got D={d}, t={t}")
    ratio = (d - 2.0 * t) / d
    return e * (1.0 - ratio**4)


def effective_soil_shear_modulus(g_soil: float, nu_soil: float) -> float:
    """Effective soil shear modulus G* = G_soil * (1 + 3/4 * nu_soil)."""
    return float(g_soil) * (1.0 + 0.75 * float(nu_soil))


def embedment_length(
    diameter: float,
    thickness: float,
    youngs_modulus: float,
    g_soil: float,
    nu_soil: float,
) -> float:
    """
    Minimum monopile embedment below mudline: D * (Ee/G*)^(2/7).

    D and t are outer diameter and wall thickness at the mudline station.
    """
    d = float(diameter)
    ee = effective_pile_modulus(d, thickness, youngs_modulus)
    g_star = effective_soil_shear_modulus(g_soil, nu_soil)
    if g_star <= 0.0:
        raise ValueError(f"effective soil shear modulus G* must be > 0, got {g_star}")
    stiffness_ratio = ee / g_star
    if stiffness_ratio <= 0.0:
        raise ValueError(f"Ee/G* must be > 0, got {stiffness_ratio}")
    return d * stiffness_ratio ** (2.0 / 7.0)


def mudline_thickness_from_geometry(monopile: dict[str, Any]) -> float:
    """Representative wall thickness for embedment rule (mudline / first station)."""
    values = monopile["structure"]["layers"][0]["thickness"]["values"]
    return float(np.atleast_1d(np.asarray(values, dtype=float)).flatten()[0])


def monopile_min_wall_thickness(diameter_m: float) -> float:
    """
    Minimum monopile wall thickness [m] from t >= D/120.

    With outer diameter D in meters: t_m = D / MONOPILE_WALL_T_D_TO_T_RATIO.
    """
    d = float(diameter_m)
    if d <= 0.0:
        raise ValueError(f"diameter must be > 0, got {d}")
    return d / MONOPILE_WALL_T_D_TO_T_RATIO


def monopile_youngs_modulus(wt: dict[str, Any]) -> float:
    """Young's modulus of the monopile steel layer from the windIO materials list."""
    mono = wt["components"]["monopile"]
    mat_name = mono["structure"]["layers"][0]["material"]
    for mat in wt.get("materials", []):
        if mat.get("name") == mat_name:
            return float(mat["E"])
    raise ValueError(f"Material '{mat_name}' not found in wt['materials']")


def embedment_length_from_wt(
    wt: dict[str, Any],
    modeling_options: dict[str, Any],
    *,
    diameter: float | None = None,
    thickness: float | None = None,
) -> float:
    """Compute mudline embedment from geometry, materials, and site soil properties."""
    mono = wt["components"]["monopile"]
    d = float(diameter) if diameter is not None else mudline_diameter_from_geometry(mono)
    t = float(thickness) if thickness is not None else mudline_thickness_from_geometry(mono)
    e_steel = monopile_youngs_modulus(wt)
    env = modeling_options.get("Environment", {})
    g_soil = float(env.get("soil_shear_modulus", DEFAULT_SOIL_SHEAR_MODULUS_PA))
    nu_soil = float(env.get("soil_poisson", DEFAULT_SOIL_POISSON))
    return embedment_length(d, t, e_steel, g_soil, nu_soil)


def n_bays_from_height(freeboard_height: float) -> int:
    """Assign jacket bay count from freeboard height thresholds."""
    h = float(freeboard_height)
    lo, hi = BAY_HEIGHT_THRESHOLDS_M
    if h < lo:
        return 3
    if h <= hi:
        return 4
    return 5


def equal_bay_spacing(n_bays: int) -> list[float]:
    """
    Normalized bay-node elevations of length n_bays + 1.

    Mid-bay stations between equal leg segments (matches JacketSE defaults):
    leg nodes use linspace(0, 1, n_bays+2); bay nodes sit halfway between
    consecutive leg elevations so Frame3DD leg stubs are non-zero length.
    """
    n = int(n_bays)
    if n < 1:
        raise ValueError(f"n_bays must be >= 1, got {n}")
    # linspace(0.5/(n+1), 1 - 0.5/(n+1), n+1)
    return np.linspace(0.5 / (n + 1), 1.0 - 0.5 / (n + 1), n + 1).tolist()


def monopile_z_top(hs: float) -> float:
    """Transition-piece elevation above mean water line."""
    return wave_clearance_above_mwl(hs)


def monopile_total_height(freeboard_height: float, embedment: float) -> float:
    """Full monopile length: seabed→TP freeboard plus pile embedment."""
    return float(freeboard_height) + float(embedment)


def monopile_reference_z(
    water_depth: float,
    hs: float,
    embedment: float,
    s_grid: list[float] | np.ndarray,
) -> list[float]:
    """
    Rebuild monopile reference-axis z values on an existing nondimensional grid.

    z(s) = z_top - height_in * (1 - s), with
      z_top = 1.98*Hs + 0.2*Hs
      height_in = H_free + embedment
      H_free = water_depth + 1.98*Hs + 0.2*Hs
    """
    h_free = required_freeboard_height(water_depth, hs)
    z_top = monopile_z_top(hs)
    height_in = monopile_total_height(h_free, embedment)
    s = np.asarray(s_grid, dtype=float).flatten()
    z = z_top - height_in * (1.0 - s)
    return z.tolist()


def mudline_diameter_from_geometry(monopile: dict[str, Any]) -> float:
    """Representative outer diameter for embedment rule (prefer mudline / first station)."""
    values = monopile["outer_shape"]["outer_diameter"]["values"]
    return float(np.atleast_1d(np.asarray(values, dtype=float)).flatten()[0])


def apply_jacket_topology(
    jacket: dict[str, Any],
    freeboard_height: float,
    n_legs: int = N_LEGS_JACKET,
) -> dict[str, Any]:
    """
    Set jacket height, n_legs, n_bays, equal bay_spacing; resize brace arrays.

    New bay diameter/thickness entries are seeded from the last baseline value.
    """
    jacket = dict(jacket)
    n_bays = n_bays_from_height(freeboard_height)
    jacket["height"] = float(freeboard_height)
    jacket["n_legs"] = int(n_legs)
    jacket["n_bays"] = int(n_bays)
    jacket["bay_spacing"] = equal_bay_spacing(n_bays)

    def _resize(key: str, default: float) -> None:
        old = list(jacket.get(key, []))
        seed = float(old[-1]) if old else float(default)
        jacket[key] = [seed] * n_bays

    _resize("brace_diameters", 1.0)
    _resize("brace_thicknesses", 0.1)
    return jacket


def _resize_station_values(values: list[float] | None, n: int) -> list[float]:
    """Resize a station property array, seeding new entries from the nearest old value."""
    old = [float(v) for v in (values or [])]
    if n <= 0:
        return []
    if not old:
        return [0.0] * n
    if len(old) == n:
        return old
    if len(old) == 1:
        return [old[0]] * n
    # Prefer constant section: broadcast first value when collapsing/expanding
    # around a mudline insert so tip/mudline/top share one D/t seed.
    return [old[0]] * n


def ensure_mudline_station(
    monopile: dict[str, Any],
    water_depth: float,
    hs: float,
    embedment: float,
) -> list[float]:
    """
    Ensure the nondimensional grid has a mudline station for soil-spring Frame3DD.

    FixedBottomSE requires exactly one unrefined section for the submerged pile
    (tip → mudline). A 2-station tip→TP grid fails that check when soil_springs
    is enabled, so insert s_mud = L_emb / (H_free + L_emb) when missing.
    Waterline is optional; keep an existing mid-water station if already present.
    """
    h_free = required_freeboard_height(water_depth, hs)
    height_in = monopile_total_height(h_free, embedment)
    if height_in <= 0.0:
        raise ValueError(f"monopile height_in must be > 0, got {height_in}")

    s_mud = float(embedment) / height_in
    s_wl = (float(embedment) + float(water_depth)) / height_in

    od = monopile.setdefault("outer_shape", {}).setdefault("outer_diameter", {})
    z_axis = monopile.setdefault("reference_axis", {}).setdefault("z", {})
    s_grid = z_axis.get("grid") or od.get("grid") or [0.0, 1.0]
    s = sorted({float(x) for x in s_grid})
    if 0.0 not in s:
        s.insert(0, 0.0)
    if 1.0 not in s:
        s.append(1.0)

    # Drop stale intermediate fractions; rebuild physical stations.
    # Keep tip, mudline, optional waterline (if WD span is meaningful), top.
    new_s = [0.0, s_mud]
    if 0.0 < s_wl < 1.0 and abs(s_wl - s_mud) > 1e-6 and abs(s_wl - 1.0) > 1e-6:
        # Only keep waterline if the original grid already had >2 stations
        # (classic 4-pt) or mudline-only would leave a very long water column.
        if len(s) >= 4:
            new_s.append(s_wl)
    new_s.append(1.0)
    # De-duplicate numerically
    out: list[float] = []
    for v in new_s:
        if not out or abs(v - out[-1]) > 1e-8:
            out.append(v)

    n = len(out)
    od["grid"] = list(out)
    od["values"] = _resize_station_values(od.get("values"), n)

    for layer in monopile.get("structure", {}).get("layers", []) or []:
        th = layer.setdefault("thickness", {})
        th["grid"] = list(out)
        th["values"] = _resize_station_values(th.get("values"), n)

    z_axis["grid"] = list(out)
    return out


def apply_monopile_geometry(
    monopile: dict[str, Any],
    water_depth: float,
    hs: float,
    embedment: float | None = None,
    youngs_modulus: float | None = None,
    g_soil: float | None = None,
    nu_soil: float | None = None,
) -> dict[str, Any]:
    """
    Rebuild monopile reference-axis z from freeboard rule + embedment.

    If embedment is None, use L_emb = D*(Ee/G*)^(2/7) from mudline D/t, steel E, and site soil.
    Inserts a mudline grid station when needed so soil_springs Frame3DD sees
    exactly one submerged-pile section (tip → mudline).
    """
    monopile = dict(monopile)
    if embedment is None:
        if youngs_modulus is None or g_soil is None or nu_soil is None:
            raise ValueError(
                "embedment or (youngs_modulus, g_soil, nu_soil) required to compute embedment length"
            )
        embedment = embedment_length(
            mudline_diameter_from_geometry(monopile),
            mudline_thickness_from_geometry(monopile),
            youngs_modulus,
            g_soil,
            nu_soil,
        )

    s_grid = ensure_mudline_station(monopile, water_depth, hs, embedment)
    z_axis = monopile.setdefault("reference_axis", {}).setdefault("z", {})
    z_axis["values"] = monopile_reference_z(water_depth, hs, embedment, s_grid)
    return monopile


def shift_tower_to_transition(tower: dict[str, Any], z_top: float) -> dict[str, Any]:
    """
    Shift tower reference-axis z so the tower base matches the transition elevation.

    Preserves section lengths (relative z spacing).
    """
    tower = dict(tower)
    z_axis = tower.setdefault("reference_axis", {}).setdefault("z", {})
    values = list(z_axis.get("values", []))
    if len(values) < 2:
        return tower
    z0 = float(values[0])
    dz = float(z_top) - z0
    z_axis["values"] = [float(v) + dz for v in values]
    return tower


def apply_wt_geometry_rules(
    wt: dict[str, Any],
    modeling_options: dict[str, Any],
    foundation_type: str,
    embedment: float | None = None,
) -> dict[str, Any]:
    """
    Apply conceptual topology/height rules to a windIO geometry dict.

    foundation_type: 'jacket' or 'monopile'
    """
    wt = copy_wt_components(wt)
    site = resolve_site_params(modeling_options)
    comps = wt.setdefault("components", {})

    if foundation_type == "jacket":
        comps["jacket"] = apply_jacket_topology(comps["jacket"], site["H_free"])
        # Align tower base with jacket top (mudline at 0 for jackets in examples is different;
        # jacket height is freeboard; tower typically starts at jacket top above MWL).
        # Jacket top elevation above MWL = H_free - water_depth = 1.98*Hs + 0.2*Hs.
        if "tower" in comps:
            comps["tower"] = shift_tower_to_transition(comps["tower"], site["z_top"])
    elif foundation_type == "monopile":
        mono_embedment = embedment
        if mono_embedment is None:
            mono_embedment = embedment_length_from_wt(wt, modeling_options)
        comps["monopile"] = apply_monopile_geometry(
            comps["monopile"],
            site["water_depth"],
            site["hs"],
            embedment=mono_embedment,
        )
        if "tower" in comps:
            comps["tower"] = shift_tower_to_transition(comps["tower"], site["z_top"])
    else:
        raise ValueError(f"Unknown foundation_type '{foundation_type}'")

    return wt


def copy_wt_components(wt: dict[str, Any]) -> dict[str, Any]:
    """Deep copy of geometry YAML."""
    return copy.deepcopy(wt)


def resolve_site_params(modeling_options: dict[str, Any]) -> dict[str, float]:
    """Pull water depth / Hs from modeling Environment and compute H_free."""
    env = modeling_options.get("Environment", {})
    water_depth = float(env.get("water_depth", 0.0))
    hs = float(env.get("significant_wave_height", 0.0))
    h_free = required_freeboard_height(water_depth, hs)
    return {
        "water_depth": water_depth,
        "hs": hs,
        "H_free": h_free,
        "z_top": monopile_z_top(hs),
    }
