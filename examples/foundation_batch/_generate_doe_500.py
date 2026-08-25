#!/usr/bin/env python3
"""
Build ~500 space-filling scenarios from the DOE_scenarios factorial.

Source grid (2000 = 4 x 4 x 5 x 5 x 5):
  turbine_rating_W: 5, 10, 15, 22 MW
  V50_mps:          35, 40, 45, 50
  Hs50_m:           4, 7.5, 11, 14.5, 18
  soil G:           10, 45, 80, 115, 150 MPa
  water_depth_m:    20, 35, 50, 65, 80
"""
from __future__ import annotations

import csv
import itertools
import os
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_CSV = os.path.join(HERE, "DOE_scenarios_500.csv")
N_TARGET = 500

HEADERS = [
    "turbine_rating_W",
    "V50_mps",
    "Hs50_m",
    "soil_shear_modulus_Pa",
    "water_depth_m",
]

LEVELS = {
    "turbine_rating_W": [5000000.0, 10000000.0, 15000000.0, 22000000.0],
    "V50_mps": [35.0, 40.0, 45.0, 50.0],
    "Hs50_m": [4.0, 7.5, 11.0, 14.5, 18.0],
    "soil_shear_modulus_Pa": [10000000.0, 45000000.0, 80000000.0, 115000000.0, 150000000.0],
    "water_depth_m": [20.0, 35.0, 50.0, 65.0, 80.0],
}


def all_grid():
    keys = HEADERS
    grids = [LEVELS[h] for h in keys]
    pts = []
    for combo in itertools.product(*grids):
        pts.append(tuple(combo))
    return pts


def coded(pt):
    u = []
    for h, x in zip(HEADERS, pt):
        lv = LEVELS[h]
        u.append(lv.index(x) / (len(lv) - 1))
    return tuple(u)


def dist2(a, b):
    s = 0.0
    for i in range(len(a)):
        d = a[i] - b[i]
        s += d * d
    return s


def lmh(h):
    """Low / middle / high specified levels."""
    lv = LEVELS[h]
    mid = lv[(len(lv) - 1) // 2]
    return [lv[0], mid, lv[-1]]


def ofat_points():
    """Every specified level, others at their middle level."""
    mids = {h: LEVELS[h][(len(LEVELS[h]) - 1) // 2] for h in HEADERS}
    out = []
    for h in HEADERS:
        for v in LEVELS[h]:
            row = dict(mids)
            row[h] = v
            out.append(tuple(row[k] for k in HEADERS))
    return out


def cube_points(level_fn):
    grids = [level_fn(h) for h in HEADERS]
    return [tuple(c) for c in itertools.product(*grids)]


def corners():
    return cube_points(lambda h: [LEVELS[h][0], LEVELS[h][-1]])


def lmh_cube():
    return cube_points(lmh)


def alt_mid_cube():
    """Second L/M/H cube using the other central level of 4-level factors (15 MW, 45 m/s)."""

    def lv(h):
        levels = LEVELS[h]
        if len(levels) == 4:
            return [levels[0], levels[2], levels[-1]]
        return lmh(h)

    return cube_points(lv)


def pairwise_lh_mid_rest():
    """Every pair of factors at {low, high}, remaining at mid."""
    mids = {h: LEVELS[h][(len(LEVELS[h]) - 1) // 2] for h in HEADERS}
    out = []
    for i, hi in enumerate(HEADERS):
        for hj in HEADERS[i + 1 :]:
            for vi in (LEVELS[hi][0], LEVELS[hi][-1]):
                for vj in (LEVELS[hj][0], LEVELS[hj][-1]):
                    row = dict(mids)
                    row[hi] = vi
                    row[hj] = vj
                    out.append(tuple(row[k] for k in HEADERS))
    return out


def triple_lh_mid_rest():
    mids = {h: LEVELS[h][(len(LEVELS[h]) - 1) // 2] for h in HEADERS}
    out = []
    n = len(HEADERS)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                hi, hj, hk = HEADERS[i], HEADERS[j], HEADERS[k]
                for vi in (LEVELS[hi][0], LEVELS[hi][-1]):
                    for vj in (LEVELS[hj][0], LEVELS[hj][-1]):
                        for vk in (LEVELS[hk][0], LEVELS[hk][-1]):
                            row = dict(mids)
                            row[hi] = vi
                            row[hj] = vj
                            row[hk] = vk
                            out.append(tuple(row[k_] for k_ in HEADERS))
    return out


def targets():
    t = {}
    for h in HEADERS:
        nlev = len(LEVELS[h])
        base, rem = divmod(N_TARGET, nlev)
        t[h] = {}
        for i, v in enumerate(LEVELS[h]):
            t[h][v] = base + (1 if i < rem else 0)
    return t


def l2_from_target(counts, targs, candidate):
    """Total squared deviation from level quotas after adding candidate."""
    s = 0.0
    for j, h in enumerate(HEADERS):
        v = candidate[j]
        for ov, tgt in targs[h].items():
            n = counts[h][ov] + (1 if ov == v else 0)
            d = n - tgt
            s += d * d
    return s


def greedy_fill(seed, universe, n_target):
    selected = []
    seen = set()
    for p in seed:
        if p not in seen:
            seen.add(p)
            selected.append(p)

    index = {p: i for i, p in enumerate(universe)}
    n_u = len(universe)
    coded_u = [coded(p) for p in universe]
    in_sel = [False] * n_u
    for p in selected:
        in_sel[index[p]] = True

    min_d2 = [1e9] * n_u
    for p in selected:
        cu = coded_u[index[p]]
        for i in range(n_u):
            if in_sel[i]:
                continue
            d = dist2(cu, coded_u[i])
            if d < min_d2[i]:
                min_d2[i] = d

    targs = targets()
    counts = {h: Counter() for h in HEADERS}
    for p in selected:
        for j, h in enumerate(HEADERS):
            counts[h][p[j]] += 1

    while len(selected) < n_target:
        # Lexicographic: keep level counts near quota, then maximin spread.
        best_i = None
        best_l2 = None
        best_d = None
        for i in range(n_u):
            if in_sel[i]:
                continue
            l2 = l2_from_target(counts, targs, universe[i])
            d = min_d2[i]
            if (
                best_i is None
                or l2 < best_l2 - 1e-12
                or (abs(l2 - best_l2) <= 1e-12 and d > best_d)
            ):
                best_i = i
                best_l2 = l2
                best_d = d
        if best_i is None:
            break
        p = universe[best_i]
        selected.append(p)
        in_sel[best_i] = True
        seen.add(p)
        for j, h in enumerate(HEADERS):
            counts[h][p[j]] += 1
        cu = coded_u[best_i]
        for i in range(n_u):
            if in_sel[i]:
                continue
            d = dist2(cu, coded_u[i])
            if d < min_d2[i]:
                min_d2[i] = d
    return selected


def fmt(h, v):
    if h == "Hs50_m":
        return f"{v:g}"
    return str(int(v)) if float(v).is_integer() else f"{v:g}"


def report(rows):
    lines = [f"n={len(rows)}"]
    targs = targets()
    for j, h in enumerate(HEADERS):
        c = Counter(r[j] for r in rows)
        lines.append(f"{h}:")
        for v in LEVELS[h]:
            lines.append(f"  {v:g}: {c[v]} (target {targs[h][v]})")
    # pairwise L/H coverage
    lo = [LEVELS[h][0] for h in HEADERS]
    hi = [LEVELS[h][-1] for h in HEADERS]
    has_lo = any(r == tuple(lo) for r in rows)
    has_hi = any(r == tuple(hi) for r in rows)
    lines.append(f"all_low_corner={has_lo} all_high_corner={has_hi}")
    # unique 2-factor cells
    for i in range(len(HEADERS)):
        for k in range(i + 1, len(HEADERS)):
            cells = {(r[i], r[k]) for r in rows}
            need = len(LEVELS[HEADERS[i]]) * len(LEVELS[HEADERS[k]])
            lines.append(f"pairs {HEADERS[i]} x {HEADERS[k]}: {len(cells)}/{need}")
    return "\n".join(lines)


def spaced_subset(points, n_keep, already):
    """Maximin subsample of `points` not already selected."""
    cand = [p for p in points if p not in already]
    if len(cand) <= n_keep:
        return cand
    chosen = [cand[0]]
    coded_c = [coded(p) for p in cand]
    chosen_i = [0]
    while len(chosen) < n_keep:
        best_i = None
        best_d = -1.0
        for i, p in enumerate(cand):
            if i in chosen_i:
                continue
            md = min(dist2(coded_c[i], coded_c[j]) for j in chosen_i)
            if md > best_d:
                best_d = md
                best_i = i
        chosen_i.append(best_i)
        chosen.append(cand[best_i])
    return chosen


def main():
    universe = all_grid()
    # Structured skeleton: corners + every specified level + 2- and 3-factor
    # high/low interactions. Do NOT dump both full 3^5 L/M/H cubes — that
    # would starve the intermediate 5-level values (7.5 m, 35 m, 45 MPa, …).
    seed = []
    seed.extend(corners())
    seed.extend(ofat_points())
    seed.extend(pairwise_lh_mid_rest())
    uniq_seed = []
    seen = set()
    for p in seed:
        if p not in seen:
            seen.add(p)
            uniq_seed.append(p)

    # Small maximin slices of L/M/H cubes and 3-factor high/low corners
    # so interactions are present without locking level quotas.
    for extra in (
        spaced_subset(triple_lh_mid_rest(), 24, seen),
        spaced_subset(lmh_cube(), 20, seen),
        spaced_subset(alt_mid_cube(), 20, seen),
    ):
        for p in extra:
            if p not in seen:
                seen.add(p)
                uniq_seed.append(p)

    selected = greedy_fill(uniq_seed, universe, N_TARGET)

    # Warm-start friendly: vary soil G last (runner reuses geometry when
    # rating/V50/Hs/depth match).
    selected.sort(key=lambda r: (r[0], r[1], r[2], r[4], r[3]))

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(HEADERS)
        for r in selected:
            w.writerow([fmt(h, v) for h, v in zip(HEADERS, r)])

    text = report(selected)
    print("WROTE", OUT_CSV)
    print(text)
    with open(os.path.join(HERE, "_doe_500_report.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")
        f.write(f"n_seed_unique={len(uniq_seed)}\n")


if __name__ == "__main__":
    main()
