"""
Polsby-Popper compactness score contribution (lower = better for total score).

PP per district = 4pi * area / perimeter^2, in (0, 1].
  1.0 = perfect circle (most compact)
  ~0  = very elongated / spiky shape

score_polsby_popper() returns  (1 - mean(PP)) * 100,  matching classic Mosaic's
polsby_popper_metric scale so that weights are comparable to other metrics:
  0.0  = all districts are perfect circles
  100.0 = maximally non-compact
"""

from __future__ import annotations

import numpy as np

from mosaic.scoring.precompute import PPData

_TWO_PI = 4.0 * 3.141592653589793


# ── Numba fused kernel ────────────────────────────────────────────────────────
# Folds the per-district area + perimeter accumulation and the final mean into
# three sequential loops with zero temporaries. Perimeter is summed in three
# isolated accumulators (base / eu-cut / ev-cut) combined in a fixed order. No
# fastmath.
from numba import njit


@njit(cache=True)
def _score_pp_numba(assignment, areas, ext_perim, eu, ev, elen,
                    n_districts, two_pi):
    n = assignment.shape[0]
    m = eu.shape[0]
    area = np.zeros(n_districts)
    # Separate accumulators sum the base / eu-cut / ev-cut perimeters in
    # isolation and combine as (base + eu) + ev, so a district's perimeter is
    # summed in a fixed, stable order regardless of district size.
    p_base = np.zeros(n_districts)
    p_eu = np.zeros(n_districts)
    p_ev = np.zeros(n_districts)
    # Pass 1: per-precinct area + exterior perimeter, in precinct order.
    for i in range(n):
        d = assignment[i]
        area[d] += areas[i]
        p_base[d] += ext_perim[i]
    # Pass 2: shared boundary of cut edges. p_eu and p_ev are independent
    # accumulators, so summing both in a single edge-order pass leaves each
    # one's summation order (and therefore its exact bit pattern) identical to
    # the old two-pass form — it just evaluates the cut test once per edge and
    # halves the edge scans.
    for i in range(m):
        du = assignment[eu[i]]
        dv = assignment[ev[i]]
        if du != dv:
            p_eu[du] += elen[i]
            p_ev[dv] += elen[i]
    # Pass 3: per-district PP, summed in district order.
    total = 0.0
    for d in range(n_districts):
        perim = (p_base[d] + p_eu[d]) + p_ev[d]
        sp = perim if perim > 0.0 else 1.0
        pp = two_pi * area[d] / (sp * sp)
        if pp > 1.0:
            pp = 1.0
        elif pp < 0.0:
            pp = 0.0
        total += pp
    return (1.0 - total / n_districts) * 100.0


def score_polsby_popper(
    assignment: np.ndarray,
    pp_data: PPData,
    n_districts: int,
) -> float:
    """
    Args:
        assignment:  (n,) int32 district indices 0..k-1
        pp_data:     precomputed geometric data (areas, ext_perimeters, edges)
        n_districts: number of districts k

    Returns:
        float in [0, 100] — (1 - mean_PP) * 100, where 0 = perfectly compact
    """
    return float(_score_pp_numba(
        assignment, pp_data.areas, pp_data.ext_perimeters,
        pp_data.edge_u, pp_data.edge_v, pp_data.edge_len,
        n_districts, _TWO_PI,
    ))
