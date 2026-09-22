"""Electoral Opportunity ratings and aggregate optimizer penalty.

For each group, the scorer sums opportunity credit over the top
``round(proportional_seats)`` districts and divides by the heuristic credit
reference in ``OpportunityResult``. A per-group rating of 100 means that the
selected districts meet or exceed that reference. Groups whose proportional
target rounds below one district, or whose heuristic feasibility flag is zero,
are marked not applicable and excluded from the aggregate.

The optimizer penalty is the proportional-seat-weighted mean shortfall. In
``unclipped`` mode, a soft per-district cap is used for the penalty while the
display ratings remain hard-capped. Inputs are a demographic total and group
counts from one consistent universe.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from mosaic.scoring.opportunity import GROUPS, OpportunityResult

_SOFT_K = 8.0   # smoothness of the soft per-district cap (higher = closer to hard min)


@njit(cache=True)
def _top_credit(P, ref, target, soft_k, want_soft):
    """Per-district credit, hard- and soft-capped, reduced to the top `target`
    values in descending order.

    Credit is P/ref: a district at the solid level counts ~1.0, below
    proportionally less, above no more. The soft cap is a differentiable stand-in
    for min(x, 1) so annealing keeps a gradient there. Taking only the top
    `target` removes the reward for cracking a group across extra districts.

    The hard and soft paths are evaluated in one compiled kernel. The caller
    reduces the returned top-credit slices.
    """
    n = P.shape[0]
    t = target
    if t < 1:
        t = 1
    if t > n:
        t = n
    hard = np.empty(n, dtype=np.float64)
    soft = np.empty(n, dtype=np.float64)
    for i in range(n):
        x = P[i] / ref                                 # P is sigmoid output, > 0
        hard[i] = x if x < 1.0 else 1.0
        if want_soft:
            soft[i] = (x ** (-soft_k) + 1.0) ** (-1.0 / soft_k)
    out_h = np.empty(t, dtype=np.float64)
    hs = np.sort(hard)
    for i in range(t):
        out_h[i] = hs[n - 1 - i]
    out_s = np.empty(t, dtype=np.float64)
    if want_soft:
        ss = np.sort(soft)
        for i in range(t):
            out_s[i] = ss[n - 1 - i]
    return out_h, out_s


def representation_from_opportunity(
    opp: OpportunityResult,
    *,
    mode: str = "proportional",
    unclipped: bool = True,
) -> tuple[float, float, dict, dict]:
    """Compute per-group ratings and the aggregate opportunity penalty.

    ``mode`` is accepted for configuration compatibility but currently does not
    select a different calculation.

    Returns:
        agg_rating -- T-weighted 0-100 (display only).
        penalty    -- 0-100, lower = better; the value the optimizer minimises.
        ratings    -- {group: 0-100 or None}; None = <1 proportional seat or a
                      zero heuristic feasibility flag; 100 = at or above the
                      heuristic credit reference.
        eff        -- {group: effective opportunity districts (top-target credit)}.
    """
    ratings: dict[str, float | None] = {}
    eff: dict[str, float] = {}
    num_disp = 0.0        # display: hard-capped scorecard (reaches 100)
    num_pen = 0.0         # optimizer: soft-capped when unclipped (smooth surface)
    den = 0.0
    for g in GROUPS:
        T = opp.T[g]
        target = int(round(T))
        feas = opp.feasible.get(g)
        # N/A gate first (both conditions are run-constant): a group below one
        # proportional seat or with a zero heuristic feasibility flag contributes
        # nothing, so skip its per-district credit + sort entirely. eff (display
        # only) is 0 -- an N/A group has no counted opportunity districts.
        if target < 1 or feas == 0:
            ratings[g] = None
            eff[g] = 0.0
            continue
        ref = opp.ref[g]
        if ref <= 0.0:                # degenerate curve: no district earns credit
            top_hard = top_soft = np.zeros(1)
        else:
            top_hard, top_soft = _top_credit(
                np.ascontiguousarray(opp.P[g], dtype=np.float64), ref, target,
                _SOFT_K, unclipped)
        O_hard = float(top_hard.sum())
        eff[g] = O_hard
        # Normalize against the supplied heuristic credit reference. Fall back
        # to the raw target count when no reference was supplied.
        denom = opp.ceiling.get(g, 0.0)
        if denom <= 1e-9:
            denom = float(target)
        frac_disp = min(O_hard / denom, 1.0)
        ratings[g] = 100.0 * frac_disp
        if unclipped:
            frac_pen = min(float(top_soft.sum()) / denom, 1.0)
        else:
            frac_pen = frac_disp
        num_disp += T * frac_disp
        num_pen += T * frac_pen
        den += T

    agg_rating = 100.0 if den <= 0.0 else 100.0 * (num_disp / den)
    penalty = 0.0 if den <= 0.0 else 100.0 * (1.0 - num_pen / den)
    return agg_rating, penalty, ratings, eff
