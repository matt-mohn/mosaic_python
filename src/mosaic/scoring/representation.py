"""
Electoral Opportunity -- per-group 0-100 ratings of electoral opportunity vs a
proportional benchmark, plus a benchmark-weighted aggregate penalty for the
optimizer. Built on the shared opportunity engine (opportunity.py).
(Internal ids stay `representation` / `weight_representation` throughout;
user-facing name is "Electoral Opportunity".)

Per group the score sums opportunity credit over the top `target` = round(T)
districts (T = proportional seats) and normalises it against f(state), the max
credit those districts could reach given the geography (opp.ceiling). Each
district contributes up to 1.0, saturating at a solid majority (via opp.ref), so
the climb through 50% is rewarded and over-packing earns nothing. Scoring only
the top `target` removes the reward for cracking a group across many
sub-opportunity districts. So:

    100 = arranged the top `target` districts as well as the geography allows

Normalising against f(state) rather than the raw count matters because the count
can be unreachable as credit: NC black is 2.99 proportional seats, so target
rounds to 3, but the geography supports only ~1.93 districts' worth of solid
credit, so a count-based 100 is a phantom capping even the best map near 64. The
population-vs-geography gap is not lost -- it lives in f(state)/round(T), where NC
black ~0.64 under smart targets (~0.76 under the statewide sweeps) means the
geography allows that share of proportional representation.

A group is not applicable (rating None, dropped from the aggregate) when its
proportional share rounds below one district or the geography cannot yield one
opportunity district. Without that gate, a state whose precincts top out below
the curve would score 100 for a map of junk districts, since achievable-
normalisation alone rewards matching a near-zero ceiling.

Per-group ratings are the primary read. The aggregate the optimizer minimises is
the T-weighted mean of per-group shortfalls in penalty form.

`unclipped` softens the per-district cap for the PENALTY only, giving a
differentiable surface near the solid level; per-group ratings stay hard-capped.

VAP basis, not CVAP: see opportunity.py.
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

    Fused into one kernel because the arrays are k long, where a dozen NumPy
    calls cost more in overhead than the arithmetic. The caller reduces the
    returned slices, whose pairwise order differs from a sequential add.
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
    """
    Returns:
        agg_rating -- T-weighted 0-100 (display only).
        penalty    -- 0-100, lower = better; the value the optimizer minimises.
        ratings    -- {group: 0-100 or None}; None = <1 proportional seat or no
                      drawable opportunity district; 100 = at the geography's ceiling.
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
        # proportional seat or with no drawable opportunity district contributes
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
        # normalise against f(state), the achievable credit ceiling, so a
        # geographically-best map reaches 100 (see module docstring). Fall back to
        # the raw count if no ceiling was supplied (e.g. a hand-built opp).
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
