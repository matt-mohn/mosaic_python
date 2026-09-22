"""
Reock compactness — 16-direction approximation of the Reock score.

Reock = district_area / area(min_bounding_circle), in (0, 1]. 1.0 = circular,
lower = elongated. This module approximates the bounding circle from cached
directional extrema rather than computing the textbook minimum bounding circle.

Reock approximates the MBC by:
  - Caching K=16 directional extreme vertices per precinct (one per direction).
  - Per iteration, picking the most-extreme precinct in each direction within
    each district -> K candidate boundary points per district.
  - Bounding circle diameter ~= pairwise diameter of those K points.

This is a deterministic 16-direction approximation, not the textbook minimum-
bounding-circle calculation. The approximation error is not estimated at run
time.

Cached arrays (per shapefile, computed once at load):
  - dir_ext_pts: (K, n, 2) extreme vertex coords per (direction, precinct)
  - dir_ext_proj: (n, K) projection of that vertex onto its direction
  - areas: (n,) precinct areas

dir_ext_proj is precinct-major because the hot loop's inner index is the
direction: (n, K) reads all K of a precinct's projections from one or two cache
lines, where (K, n) strides n floats per direction and takes K misses per
precinct. dir_ext_pts keeps direction-major layout — it is read only in the
short per-district loop, K*k times, not once per precinct.

At scoring time, one compiled pass over precincts builds a per-(direction,
district) max-projection table; a district loop then computes pairwise diameters
from the cached points.

score_reock() returns (1 - mean(reock)) * 100. PP uses the same algebraic
0-100 penalty orientation, but the distributions of the two metrics can differ.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Optional

import geopandas as gpd
import numpy as np
from numba import njit

log = logging.getLogger("mosaic")


# K=16 directions evenly spaced around the circle, starting at angle 0.
K_DIRS = 16
_ANGLES = np.linspace(0.0, 2.0 * np.pi, K_DIRS, endpoint=False)
DIRS = np.stack([np.cos(_ANGLES), np.sin(_ANGLES)], axis=1)


@dataclass
class ReockData:
    """Per-precinct K-direction extrema cached at shapefile load time.

    These arrays are linear under disjoint union of precincts only in the
    sense that the district-level extreme in each direction equals the max
    over assigned precincts of their per-direction extreme — the score
    function exploits that for a single fused groupby-max pass.
    """
    dir_ext_pts: np.ndarray   # (K, n, 2)
    dir_ext_proj: np.ndarray  # (n, K) — precinct-major; see module docstring
    areas: np.ndarray         # (n,)


def precompute_reock_data(gdf: gpd.GeoDataFrame) -> Optional[ReockData]:
    """Compute per-precinct K-direction extrema and projections.

    For each precinct, the K extreme vertices are taken from its convex hull.
    The extraction is performed when scoring data is prepared.

    Returns None on failure; the scoring slot then stays disabled in the GUI.
    """
    try:
        n = len(gdf)
        ext_pts = np.zeros((K_DIRS, n, 2), dtype=np.float64)
        ext_proj = np.zeros((n, K_DIRS), dtype=np.float64)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=".*geographic CRS.*", category=UserWarning,
            )
            areas = gdf.geometry.area.values.astype(np.float64)

        for i, geom in enumerate(gdf.geometry.values):
            hull = geom.convex_hull
            if hull.geom_type == "Polygon":
                coords = np.asarray(hull.exterior.coords)
            elif hull.geom_type == "Point":
                coords = np.array([[hull.x, hull.y]])
            else:
                coords = np.asarray(hull.coords)
            projs = coords @ DIRS.T              # (n_coords, K)
            best = np.argmax(projs, axis=0)      # (K,)
            ext_pts[:, i, :] = coords[best]
            ext_proj[i, :] = projs[best, np.arange(K_DIRS)]

        log.info(f"Reock data precomputed: {n} precincts, K={K_DIRS}")
        return ReockData(
            dir_ext_pts=ext_pts, dir_ext_proj=ext_proj, areas=areas,
        )
    except Exception as exc:
        log.warning(f"Reock precomputation failed: {exc}. Reock disabled.")
        return None


@njit(cache=True, fastmath=True)
def _score_reock_numba(
    dir_ext_pts: np.ndarray,
    dir_ext_proj: np.ndarray,
    assignment: np.ndarray,
    areas: np.ndarray,
    n_districts: int,
) -> float:
    """Compiled inner loop. One pass over precincts builds the per-(direction,
    district) (max_projection, argmax_precinct) table, then a district loop
    computes pairwise diameters from K cached extreme points.
    """
    k = dir_ext_proj.shape[1]
    n = assignment.shape[0]

    # District-major so the inner direction loop walks contiguous memory in both
    # passes; the tables are k*n_districts, small enough to stay resident.
    max_proj = np.full((n_districts, k), -1e30, dtype=np.float64)
    max_idx = np.full((n_districts, k), -1, dtype=np.int64)
    district_areas = np.zeros(n_districts, dtype=np.float64)

    for i in range(n):
        d = assignment[i]
        district_areas[d] += areas[i]
        for ki in range(k):
            v = dir_ext_proj[i, ki]
            if v > max_proj[d, ki]:
                max_proj[d, ki] = v
                max_idx[d, ki] = i

    total = 0.0
    for d in range(n_districts):
        d_max_sq = 0.0
        for ki in range(k):
            pi = max_idx[d, ki]
            if pi < 0:
                continue
            xi = dir_ext_pts[ki, pi, 0]
            yi = dir_ext_pts[ki, pi, 1]
            for kj in range(ki + 1, k):
                pj = max_idx[d, kj]
                if pj < 0:
                    continue
                dx = xi - dir_ext_pts[kj, pj, 0]
                dy = yi - dir_ext_pts[kj, pj, 1]
                ds = dx * dx + dy * dy
                if ds > d_max_sq:
                    d_max_sq = ds
        r = np.sqrt(d_max_sq) / 2.0
        if r > 0.0:
            rk = district_areas[d] / (np.pi * r * r)
            if rk > 1.0:
                rk = 1.0
            total += rk
    return (1.0 - total / n_districts) * 100.0


def score_reock(
    assignment: np.ndarray,
    data: ReockData,
    n_districts: int,
) -> float:
    """Args:
        assignment:  (n,) int district indices 0..k-1
        data:        precomputed ReockData
        n_districts: number of districts k

    Returns:
        float in [0, 100] — (1 - mean_Reock) * 100, where 0 = all
        districts perfectly compact under the K=16 approximation.
    """
    # Pass int32 assignment straight to Numba (used only as indices/bounds),
    # avoiding a per-iteration full-precinct int64 copy.
    return float(_score_reock_numba(
        data.dir_ext_pts,
        data.dir_ext_proj,
        assignment,
        data.areas,
        n_districts,
    ))


@njit(cache=True, fastmath=True)
def _reock_per_district_numba(
    dir_ext_pts: np.ndarray,
    dir_ext_proj: np.ndarray,
    assignment: np.ndarray,
    areas: np.ndarray,
    n_districts: int,
) -> np.ndarray:
    """Per-district Reock ratio in [0, 1] (same math as _score_reock_numba, but
    returns the per-district array instead of the (1 - mean) penalty). Used only
    for map shading, so it lives apart from the hot-loop scorer."""
    k = dir_ext_proj.shape[1]
    n = assignment.shape[0]

    max_proj = np.full((n_districts, k), -1e30, dtype=np.float64)
    max_idx = np.full((n_districts, k), -1, dtype=np.int64)
    district_areas = np.zeros(n_districts, dtype=np.float64)

    for i in range(n):
        d = assignment[i]
        district_areas[d] += areas[i]
        for ki in range(k):
            v = dir_ext_proj[i, ki]
            if v > max_proj[d, ki]:
                max_proj[d, ki] = v
                max_idx[d, ki] = i

    out = np.zeros(n_districts, dtype=np.float64)
    for d in range(n_districts):
        d_max_sq = 0.0
        for ki in range(k):
            pi = max_idx[d, ki]
            if pi < 0:
                continue
            xi = dir_ext_pts[ki, pi, 0]
            yi = dir_ext_pts[ki, pi, 1]
            for kj in range(ki + 1, k):
                pj = max_idx[d, kj]
                if pj < 0:
                    continue
                dx = xi - dir_ext_pts[kj, pj, 0]
                dy = yi - dir_ext_pts[kj, pj, 1]
                ds = dx * dx + dy * dy
                if ds > d_max_sq:
                    d_max_sq = ds
        r = np.sqrt(d_max_sq) / 2.0
        if r > 0.0:
            rk = district_areas[d] / (np.pi * r * r)
            if rk > 1.0:
                rk = 1.0
            out[d] = rk
    return out


def reock_per_district(
    assignment: np.ndarray,
    data: ReockData,
    n_districts: int,
) -> np.ndarray:
    """(n_districts,) per-district Reock ratio in [0, 1]; 0 = empty/degenerate."""
    return _reock_per_district_numba(
        data.dir_ext_pts,
        data.dir_ext_proj,
        assignment,
        data.areas,
        n_districts,
    )
