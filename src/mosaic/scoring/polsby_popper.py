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
    dist_area = np.bincount(assignment, weights=pp_data.areas,
                             minlength=n_districts).astype(np.float64)
    dist_perim = np.bincount(assignment, weights=pp_data.ext_perimeters,
                              minlength=n_districts).astype(np.float64)

    eu, ev, elen = pp_data.edge_u, pp_data.edge_v, pp_data.edge_len
    if len(eu) > 0:
        eu_dist = assignment[eu]
        ev_dist = assignment[ev]
        is_cut = eu_dist != ev_dist
        if is_cut.any():
            cut_len = elen[is_cut]
            # bincount is a fully-vectorized C scatter-add; np.add.at is a
            # Python-level loop and is the historical bottleneck here.
            dist_perim += np.bincount(eu_dist[is_cut], weights=cut_len,
                                       minlength=n_districts)
            dist_perim += np.bincount(ev_dist[is_cut], weights=cut_len,
                                       minlength=n_districts)

    safe_perim = np.where(dist_perim > 0.0, dist_perim, 1.0)
    pp = _TWO_PI * dist_area / (safe_perim ** 2)
    pp = np.clip(pp, 0.0, 1.0)
    return float((1.0 - pp.mean()) * 100.0)
