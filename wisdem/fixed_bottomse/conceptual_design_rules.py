"""
Conceptual design rules for fixed-bottom support structures.

These helpers set topology and freeboard height from site parameters so that
WISDEM optimisation sizes member diameters/thicknesses only. They are
intentionally simple and isolated so thresholds can be changed later without
touching Frame3DD or PoseOptimization.

Engineering rules (conceptual design):
  H_free = water_depth + Hs + air_gap + safety_margin
  L_embedment = EMBEDMENT_DIAMETER_FACTOR * D_opt   (monopile, after sizing)
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
DEFAULT_AIR_GAP_M = 1.5
DEFAULT_SAFETY_MARGIN_M = 4.0
EMBEDMENT_DIAMETER_FACTOR = 5.0
N_LEGS_JACKET = 4

# Jacket freeboard height [m] → number of bays
BAY_HEIGHT_THRESHOLDS_M = (50.0, 80.0)  # <50 → 3; 50–80 → 4; >80 → 5


def required_freeboard_height(
    water_depth: float,
    hs: float,
    air_gap: float = DEFAULT_AIR_GAP_M,
    safety_margin: float = DEFAULT_SAFETY_MARGIN_M,
) -> float:
    """
    Mudline-to-top freeboard height (jacket height; monopile span above seabed).

    Matches the WISDEM height_airgap constraint equality case:
        height = water_depth + Hs + air_gap + safety_margin
    """
    return float(water_depth) + float(hs) + float(air_gap) + float(safety_margin)


def embedment_length(diameter: float, factor: float = EMBEDMENT_DIAMETER_FACTOR) -> float:
    """Monopile pile embedment below mudline: factor × outer diameter."""
    return float(factor) * float(diameter)


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


def monopile_z_top(
    hs: float,
    air_gap: float = DEFAULT_AIR_GAP_M,
    safety_margin: float = DEFAULT_SAFETY_MARGIN_M,
) -> float:
    """Transition-piece elevation above mean water line."""
    return float(hs) + float(air_gap) + float(safety_margin)


def monopile_total_height(freeboard_height: float, embedment: float) -> float:
    """Full monopile length: seabed→TP freeboard plus pile embedment."""
    return float(freeboard_height) + float(embedment)


def monopile_reference_z(
    water_depth: float,
    hs: float,
    embedment: float,
    s_grid: list[float] | np.ndarray,
    air_gap: float = DEFAULT_AIR_GAP_M,
    safety_margin: float = DEFAULT_SAFETY_MARGIN_M,
) -> list[float]:
    """
    Rebuild monopile reference-axis z values on an existing nondimensional grid.

    z(s) = z_top - height_in * (1 - s), with
      z_top = Hs + air_gap + safety_margin
      height_in = H_free + embedment
      H_free = water_depth + Hs + air_gap + safety_margin
    """
    h_free = required_freeboard_height(water_depth, hs, air_gap, safety_margin)
    z_top = monopile_z_top(hs, air_gap, safety_margin)
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
    air_gap: float = DEFAULT_AIR_GAP_M,
    safety_margin: float = DEFAULT_SAFETY_MARGIN_M,
) -> list[float]:
    """
    Ensure the nondimensional grid has a mudline station for soil-spring Frame3DD.

    FixedBottomSE requires exactly one unrefined section for the submerged pile
    (tip → mudline). A 2-station tip→TP grid fails that check when soil_springs
    is enabled, so insert s_mud = L_emb / (H_free + L_emb) when missing.
    Waterline is optional; keep an existing mid-water station if already present.
    """
    h_free = required_freeboard_height(water_depth, hs, air_gap, safety_margin)
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
    air_gap: float = DEFAULT_AIR_GAP_M,
    safety_margin: float = DEFAULT_SAFETY_MARGIN_M,
) -> dict[str, Any]:
    """
    Rebuild monopile reference-axis z from freeboard rule + embedment.

    If embedment is None, use provisional L_emb = factor × current mudline diameter.
    Inserts a mudline grid station when needed so soil_springs Frame3DD sees
    exactly one submerged-pile section (tip → mudline).
    """
    monopile = dict(monopile)
    if embedment is None:
        embedment = embedment_length(mudline_diameter_from_geometry(monopile))

    s_grid = ensure_mudline_station(
        monopile, water_depth, hs, embedment, air_gap=air_gap, safety_margin=safety_margin
    )
    z_axis = monopile.setdefault("reference_axis", {}).setdefault("z", {})
    z_axis["values"] = monopile_reference_z(
        water_depth, hs, embedment, s_grid, air_gap=air_gap, safety_margin=safety_margin
    )
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
    air_gap: float = DEFAULT_AIR_GAP_M,
    safety_margin: float = DEFAULT_SAFETY_MARGIN_M,
) -> dict[str, Any]:
    """
    Apply conceptual topology/height rules to a windIO geometry dict.

    foundation_type: 'jacket' or 'monopile'
    """
    wt = copy_wt_components(wt)
    site = resolve_site_params(modeling_options, air_gap=air_gap, safety_margin=safety_margin)
    comps = wt.setdefault("components", {})

    if foundation_type == "jacket":
        comps["jacket"] = apply_jacket_topology(comps["jacket"], site["H_free"])
        # Align tower base with jacket top (mudline at 0 for jackets in examples is different;
        # jacket height is freeboard; tower typically starts at jacket top above MWL).
        # Jacket top elevation above MWL = H_free - water_depth = Hs + air_gap + safety_margin.
        if "tower" in comps:
            comps["tower"] = shift_tower_to_transition(comps["tower"], site["z_top"])
    elif foundation_type == "monopile":
        comps["monopile"] = apply_monopile_geometry(
            comps["monopile"],
            site["water_depth"],
            site["hs"],
            embedment=embedment,
            air_gap=air_gap,
            safety_margin=safety_margin,
        )
        if "tower" in comps:
            comps["tower"] = shift_tower_to_transition(comps["tower"], site["z_top"])
    else:
        raise ValueError(f"Unknown foundation_type '{foundation_type}'")

    return wt


def copy_wt_components(wt: dict[str, Any]) -> dict[str, Any]:
    """Deep copy of geometry YAML."""
    return copy.deepcopy(wt)


def resolve_site_params(
    modeling_options: dict[str, Any],
    air_gap: float = DEFAULT_AIR_GAP_M,
    safety_margin: float = DEFAULT_SAFETY_MARGIN_M,
) -> dict[str, float]:
    """Pull water depth / Hs from modeling Environment and compute H_free."""
    env = modeling_options.get("Environment", {})
    water_depth = float(env.get("water_depth", 0.0))
    hs = float(env.get("significant_wave_height", 0.0))
    h_free = required_freeboard_height(water_depth, hs, air_gap, safety_margin)
    return {
        "water_depth": water_depth,
        "hs": hs,
        "air_gap": float(air_gap),
        "safety_margin": float(safety_margin),
        "H_free": h_free,
        "z_top": monopile_z_top(hs, air_gap, safety_margin),
    }
