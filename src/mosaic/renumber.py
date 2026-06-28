"""Geographic district renumbering (label-only).

Produces a *label map* that renames districts by geographic position without
touching colors or the underlying assignment array. This is the "rational
numbering" several states apply to a finished map (e.g. ordering districts so
the numbers sweep across the state instead of reflecting draw order).

Rules:
  ``nw_se`` -- diagonal sweep from the northwest corner to the southeast:
              number 1 = NW-most district, number k = SE-most.
  ``n_s``   -- north to south: number 1 = northmost, ties broken west-to-east.

The map is keyed by *stable color index* (the index Mosaic already uses to keep
a region under the same color across iterations), so applying it changes the
number shown on a district without disturbing its color.
"""

from __future__ import annotations

import numpy as np


def geographic_label_map(
    assignment: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    k: int,
    rule: str = "nw_se",
) -> np.ndarray:
    """Rank districts by geographic position; return 1-indexed labels.

    Args:
        assignment: (n,) district index per precinct, 0..k-1. Pass the stable
                    color indices so the resulting numbers line up with the
                    colors and the District Info panel.
        x, y:       (n,) precinct centroid coordinates in a north-up CRS --
                    x east-positive, y north-positive (lon/lat or any standard
                    projected CRS, NOT screen pixels, whose y points down).
        weights:    (n,) per-precinct weight for the district centroid
                    (population). Zero / non-finite safe.
        k:          number of districts.
        rule:       "nw_se" (diagonal NW->SE) or "n_s" (north to south).

    Returns:
        (k,) int32 where ``label_map[d]`` is district d's 1..k rank under the
        rule. ``label_map[assignment]`` therefore gives a per-precinct number,
        and the array is a permutation of 1..k. Empty districts (no precincts)
        sort last with a deterministic, index-stable order.
    """
    if rule not in ("nw_se", "n_s"):
        raise ValueError(f"unknown renumber rule: {rule!r}")

    assignment = np.asarray(assignment)
    w = np.asarray(weights, dtype=np.float64)

    # Population-weighted centroid per district.
    wsum = np.bincount(assignment, weights=w, minlength=k).astype(np.float64)
    cx = np.bincount(assignment, weights=w * np.asarray(x, dtype=np.float64), minlength=k)
    cy = np.bincount(assignment, weights=w * np.asarray(y, dtype=np.float64), minlength=k)
    nonempty = wsum > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        cx = np.where(nonempty, cx / wsum, np.nan)
        cy = np.where(nonempty, cy / wsum, np.nan)

    # Normalize each axis to [0, 1] across district centroids so the diagonal
    # stays balanced regardless of the state's aspect ratio. Ranking only, so
    # any monotone transform is fine; this just keeps north and west weighted
    # comparably for the diagonal sweep.
    def _norm(a: np.ndarray) -> np.ndarray:
        lo = np.nanmin(a)
        hi = np.nanmax(a)
        rng = hi - lo
        if not np.isfinite(rng) or rng <= 0:
            return np.zeros_like(a)
        return (a - lo) / rng

    nx = _norm(cx)   # 0 = westmost, 1 = eastmost
    ny = _norm(cy)   # 0 = southmost, 1 = northmost

    # lexsort's LAST key is primary; ties fall through to earlier keys, with a
    # final arange(k) tiebreak so ordering is deterministic and index-stable.
    # Empty districts are pushed to the end via +inf/-inf sentinels.
    idx = np.arange(k)
    if rule == "nw_se":
        # Diagonal: highest (north + west) score is the NW corner -> label 1.
        score = np.where(nonempty, ny + (1.0 - nx), -np.inf)
        order = np.lexsort((idx, -score))
    else:  # "n_s"
        # North first; tie-break west-to-east (low nx first), then index.
        pny = np.where(nonempty, ny, -np.inf)   # north high; empties last
        pnx = np.where(nonempty, nx, np.inf)
        order = np.lexsort((idx, pnx, -pny))

    label_map = np.empty(k, dtype=np.int32)
    label_map[order] = np.arange(1, k + 1, dtype=np.int32)
    return label_map


def infer_label_map_from_reference(
    prop_assignment: np.ndarray,
    alt_assignment: np.ndarray,
    alt_labels: np.ndarray,
    weights: np.ndarray,
    k_prop: int,
    n_alt: int,
) -> np.ndarray:
    """Number proposed districts to match a reference (alignment) plan.

    Each proposed district adopts the *number* of the reference district it
    most corresponds to. Correspondence is decided by a maximum-weight bipartite
    matching on the overlap matrix M[a, p] = weight shared by reference district
    a and proposed district p, where ``weights`` is the per-precinct quantity
    being aligned (population, or a party's votes). The matching is global, not
    greedy: when one reference district is the largest source for two proposed
    districts, the match maximizing total retained weight wins and the other
    proposed district falls to its next-best free reference -- exactly the
    "two successors" collision that a per-district argmax mishandles.

    District counts need not match. Proposed districts with no positive-overlap
    match (brand-new territory, or surplus when k_prop > n_alt) receive fresh
    numbers above every reference label so nothing collides.

    Args:
        prop_assignment: (n,) proposed district index per precinct, 0..k_prop-1.
                         Pass the stable color indices so numbers line up with
                         colors and panels.
        alt_assignment:  (n,) reference district index per precinct, 0..n_alt-1.
        alt_labels:      (n_alt,) original reference district number per index.
        weights:         (n,) per-precinct weight (population / D votes / R votes).
        k_prop:          number of proposed districts.
        n_alt:           number of reference districts.

    Returns:
        (k_prop,) int32 displayed number per proposed district.
    """
    from scipy.optimize import linear_sum_assignment

    from mosaic.scoring.alignment import _overlap_matrix

    m = _overlap_matrix(
        np.asarray(alt_assignment, dtype=np.int64),
        np.asarray(prop_assignment, dtype=np.int64),
        np.asarray(weights, dtype=np.float64),
        n_alt,
        k_prop,
    )

    label_map = np.zeros(k_prop, dtype=np.int32)
    assigned = np.zeros(k_prop, dtype=bool)
    if n_alt > 0 and m.size > 0:
        # Maximize total overlap == minimize -overlap. Rectangular-safe: returns
        # min(n_alt, k_prop) pairs. Every matched proposed district adopts its
        # reference number, even on a forced zero-overlap pairing -- that keeps
        # an equal-count (or fewer-proposed) map a clean permutation of the
        # reference's numbers instead of inventing out-of-range labels.
        rows, cols = linear_sum_assignment(-m)
        for a, p in zip(rows, cols):
            label_map[p] = int(alt_labels[a])
            assigned[p] = True

    # Fresh numbers ONLY for proposed districts the matching couldn't cover --
    # i.e. genuine surplus when there are more districts than the reference has.
    # Fill the smallest unused positive integers so numbering stays compact
    # (e.g. a 15-district run against a 13-district reference uses 14, 15, not
    # 16, 17) and never collides with an adopted reference number.
    if not assigned.all():
        used = {int(v) for v in label_map[assigned]}
        nxt = 1
        for p in range(k_prop):
            if not assigned[p]:
                while nxt in used:
                    nxt += 1
                label_map[p] = nxt
                used.add(nxt)
                nxt += 1
    return label_map
