"""Single-precinct flip move for the ReCom polish phase.

A flip picks a precinct on the boundary between two districts and moves
it to the adjacent district, if the move keeps the source district
contiguous and both districts within population tolerance. Cheap per
attempt because there is no spanning-tree construction; the per-step
cost is dominated by a local igraph contiguity check on the source
district minus the flipped precinct.

Contract mirrors recom_step_ig: returns (new_assignment, success,
new_cut_edge_indices). The engine handles full PlanScore + Metropolis
above this layer. On failure the returned assignment IS the input
object (no copy), so callers can compare by identity to detect a no-op.
"""

from __future__ import annotations

import math

import numpy as np

from mosaic.recom.recombination import GraphContext
from mosaic.recom.tree import _NUMBA_OK
if _NUMBA_OK:
    from mosaic.recom.tree import _nb_district_connected_without


# Anchor points for the flip-rate ramp. The curve is pinned to these three
# values for ANY midpoint: f(0)=START, f(midpoint)=MID, f(1)=END.
FLIP_RATE_START = 0.05
FLIP_RATE_MID = 0.50
FLIP_RATE_END = 0.85
# Default 0.5-crossing (progress at which the rate hits FLIP_RATE_MID) and the
# shared logistic steepness for both halves. Steepness is a fixed constant, not
# a user knob: it only sets how S-shaped each half is, never the endpoints.
DEFAULT_FLIP_MIDPOINT = 0.835
DEFAULT_FLIP_STEEPNESS = 9.5


def _centered_logistic(x: float) -> float:
    """Logistic mapping R -> (-1, 1) with f(0)=0. Equals tanh(x/2)."""
    return math.tanh(0.5 * x)


def flip_rate_curve(
    progress: float,
    midpoint: float = DEFAULT_FLIP_MIDPOINT,
    steepness: float = DEFAULT_FLIP_STEEPNESS,
) -> float:
    """Two-piece logistic ramp giving the per-step flip probability.

    The curve is anchored at three fixed points, exactly, for any midpoint:
        f(0)        = FLIP_RATE_START (0.05)
        f(midpoint) = FLIP_RATE_MID   (0.50)
        f(1)        = FLIP_RATE_END   (0.85)

    Each side is a normalized centered logistic, so the endpoints land
    exactly (not just asymptotically). The two halves span different amounts
    (0.45 below the midpoint, 0.35 above), so the slope can change at the
    midpoint -- unavoidable given the asymmetric anchors.

    Args:
        progress: current_iteration / max_iterations; clamped to [0, 1].
        midpoint: progress at which the rate hits FLIP_RATE_MID (0.50).
                  User-adjustable.
        steepness: shared logistic k for both halves. Higher = more S-shaped;
                   lower = closer to two straight lines. Fixed constant.

    Returns:
        Flip proposal probability in [FLIP_RATE_START, FLIP_RATE_END].
    """
    if progress <= 0.0:
        return FLIP_RATE_START
    if progress >= 1.0:
        return FLIP_RATE_END

    # Degenerate midpoints collapse the ramp to a single half.
    if midpoint <= 0.0:
        denom = _centered_logistic(steepness * 1.0)
        return FLIP_RATE_MID + (FLIP_RATE_END - FLIP_RATE_MID) * (
            _centered_logistic(steepness * progress) / denom
        )
    if midpoint >= 1.0:
        denom = -_centered_logistic(steepness * -1.0)
        return FLIP_RATE_MID + (FLIP_RATE_MID - FLIP_RATE_START) * (
            _centered_logistic(steepness * (progress - 1.0)) / denom
        )

    if progress <= midpoint:
        # Lower half: rises from START at p=0 to MID at p=midpoint.
        denom = -_centered_logistic(steepness * (0.0 - midpoint))  # > 0
        return FLIP_RATE_MID + (FLIP_RATE_MID - FLIP_RATE_START) * (
            _centered_logistic(steepness * (progress - midpoint)) / denom
        )
    # Upper half: rises from MID at p=midpoint to END at p=1.
    denom = _centered_logistic(steepness * (1.0 - midpoint))  # > 0
    return FLIP_RATE_MID + (FLIP_RATE_END - FLIP_RATE_MID) * (
        _centered_logistic(steepness * (progress - midpoint)) / denom
    )


def flip_step_ig(
    ctx: GraphContext,
    assignment: np.ndarray,
    populations: np.ndarray,
    ideal_pop: float,
    tolerance: float,
    cut_edge_indices: np.ndarray | None = None,
    county_array: np.ndarray | None = None,
    county_bias: float = 1.0,
    max_attempts: int = 100,
) -> tuple[np.ndarray, bool, np.ndarray]:
    """Propose one single-precinct flip move.

    Tries up to `max_attempts` random (cut-edge, direction) picks before
    giving up, matching ReCom's internal retry behavior. Each rejection
    (would-empty source, contiguity break, pop tolerance) triggers another
    pick rather than failing the whole call.

    Args:
        ctx: Graph context (igraph + cached edge arrays).
        assignment: Current district assignment; not modified.
        populations: Population per precinct.
        ideal_pop: Target population per district.
        tolerance: Fractional population deviation tolerance.
        cut_edge_indices: Cached indices into ctx.edge_u/edge_v of cut
                          edges, or None to recompute from scratch.
        county_array: Optional precinct -> county-id array. Combined with
                      county_bias > 1.0, suppresses selection of cross-
                      county cut edges (the flip-side analog of ReCom's
                      MST weighting).
        county_bias: Multiplier (>= 1.0). 1.0 = uniform random selection
                     (no bias). Higher = cross-county cut edges are
                     correspondingly less likely to be picked as the
                     flip pivot. Only used when county_array is provided.
        max_attempts: How many random (cut-edge, direction) picks to try
                      before returning failure. Default 100 mirrors
                      recom_step_ig.

    Returns:
        (new_assignment, success, new_cut_edge_indices). On exhaustion
        (no cut edges at all, or all `max_attempts` picks rejected)
        returns (assignment, False, cut_edge_indices) — same objects.
    """
    if cut_edge_indices is None:
        cut_edge_indices = ctx.compute_cut_edges(assignment)

    n_cuts = len(cut_edge_indices)
    if n_cuts == 0:
        return assignment, False, cut_edge_indices

    lo = ideal_pop * (1.0 - tolerance)
    hi = ideal_pop * (1.0 + tolerance)

    # County-biased cut-edge selection. Built once per call; the inner
    # loop draws via np.random.random() + searchsorted, which costs the
    # same as np.random.randint at this scale.
    if county_array is not None and county_bias != 1.0:
        eu_idx = ctx.edge_u[cut_edge_indices]
        ev_idx = ctx.edge_v[cut_edge_indices]
        cross_county = county_array[eu_idx] != county_array[ev_idx]
        weights = np.where(cross_county, 1.0 / county_bias, 1.0)
        cum_weights = np.cumsum(weights)
        total_weight = float(cum_weights[-1])
    else:
        cum_weights = None
        total_weight = 0.0

    for _ in range(max_attempts):
        # Pick a random cut edge, then a random direction across it.
        if cum_weights is None:
            rand_pos = np.random.randint(n_cuts)
        else:
            rand_pos = int(np.searchsorted(
                cum_weights, np.random.random() * total_weight
            ))
            if rand_pos >= n_cuts:   # guard against floating-point edge
                rand_pos = n_cuts - 1
        edge_idx = int(cut_edge_indices[rand_pos])
        u = int(ctx.edge_u[edge_idx])
        v = int(ctx.edge_v[edge_idx])
        if np.random.random() < 0.5:
            src, dst = u, v
        else:
            src, dst = v, u

        src_district = int(assignment[src])
        dst_district = int(assignment[dst])

        # Refuse to empty the source district.
        src_mask = assignment == src_district
        if int(src_mask.sum()) <= 1:
            continue

        # Population check on the two affected districts.
        pop_src_old = float(populations[src_mask].sum())
        pop_dst_old = float(populations[assignment == dst_district].sum())
        p_src = float(populations[src])
        pop_src_new = pop_src_old - p_src
        pop_dst_new = pop_dst_old + p_src
        if not (lo <= pop_src_new <= hi and lo <= pop_dst_new <= hi):
            continue

        # Contiguity: source district must stay connected after src is removed.
        # Destination stays connected because src attaches to a node already in
        # it (that's what makes the picked edge a cut edge).
        if _NUMBA_OK:
            connected = _nb_district_connected_without(
                ctx.edge_u, ctx.edge_v, assignment, src_district, src)
        else:
            remaining = np.flatnonzero(src_mask).astype(np.int32)
            remaining = remaining[remaining != src]
            connected = ctx.graph.subgraph(remaining.tolist()).is_connected()
        if not connected:
            continue

        # All checks passed — apply.
        new_assignment = assignment.copy()
        new_assignment[src] = dst_district
        new_cut_edge_indices = ctx.compute_cut_edges(new_assignment)
        return new_assignment, True, new_cut_edge_indices

    # Exhausted max_attempts without finding a valid flip.
    return assignment, False, cut_edge_indices
