"""
County Congruence — bidirectional county/district splitting penalty.
(Internal id stays `holistic_splitting` throughout; user-facing name is
"County Congruence".)

Builds the CxD population matrix and computes two complementary scores:

  raw_county   = sum_c (county_pop_c / state_pop) * sum_d sqrt(CxD[c, d] / county_pop_c)
  raw_district = sum_d (district_pop_d / state_pop) * sum_c sqrt(CxD[c, d] / district_pop_d)

Both use Moon Duchin's split-score kernel `sum(sqrt(fraction))`, which is minimised
at 1.0 when nothing is split (one bucket holds the full mass) and grows as mass
gets spread across more buckets. County-direction weights by county population
(splitting a big city costs more than splitting a tiny rural county). District-
direction weights by district population (uniform across districts in practice
since districts are pop-equal).

Reduce step (both directions): whole-district-inside-one-county and whole-county-
inside-one-district overlaps are folded into a single bucket per outer entity.
Without this, a district touching many small whole counties would be penalised
the same as a district that fragments those counties — which isn't what we want.

Two penalty forms, both blended 50/50 across the county and district directions
(0 = perfect):

  - `unclipped` (DEFAULT -- the annealing form): a flat slope from the true
    no-split floor, `pen_dir = _UNCLIPPED_SLOPE * (raw - 1.0)`. raw = 1.0 (a map
    with no splits) is the only non-arbitrary anchor there is; any split above it
    registers, the gradient runs to zero splits, and it never caps. No band, no
    1.33 -- the single knob is the slope, which is pure magnitude (the weight
    does the rest).

  - clipped (scorecard form): the classic DRA-style band. 0 at a map-derived
    ideal `ref` (looser `best_target(n_c,n_d)` on the many-into-few direction,
    strict MAX_SPLITTING = 1.20 on the other), rising to 100 at `worst = ref *
    1.33` and pinned there. `pen_dir = 100 * clip((raw-ref)/(worst-ref), 0, 1)`,
    then * _PENALTY_SCALE so its magnitude sits near the unclipped form's (so
    toggling the mode doesn't reprice the weight).

The `raw_county` / `raw_district` sqrt-entropy scores (min 1.0 = no splits) are
population- and lopsidedness-weighted, so severity and county size matter -- a
plain split count is Classic Splitting's job, not this one.
"""

from __future__ import annotations

import numpy as np


_MAX_SPLITTING    = 1.20   # clipped scorecard form only
_MIN_SPLITTING    = 1.00
_WORST_MULTIPLIER = 1.33   # clipped scorecard form only
_PENALTY_SCALE    = 0.2    # clipped-form pre-weight scale
_UNCLIPPED_SLOPE  = 45.0   # unclipped penalty per unit of raw excess over 1.0


# ── Numba fused kernel for the raw split scores ───────────────────────────────
# Folds both directions into one (c, d) pass with per-c / per-d accumulators:
# the county direction sums over districts (inner d loop), the district direction
# accumulates per-d across the c outer loop. A whole district inside one county
# (or whole county inside one district) collapses into its own bucket. No
# fastmath. Exact == on pop sums is safe (integer pops, exact in float64).
from numba import njit


@njit(cache=True)
def _hsplit_raw_numba(co, county_pops, district_pops, state_pop,
                      n_counties, n_districts):
    sqrt_partial_co = np.zeros(n_counties)
    whole_co = np.zeros(n_counties)
    sqrt_partial_d = np.zeros(n_districts)
    whole_d = np.zeros(n_districts)
    for c in range(n_counties):
        cp = county_pops[c]
        safe_cp = cp if cp > 0.0 else 1.0
        for d in range(n_districts):
            v = co[c, d]
            dp = district_pops[d]
            # County direction: whole district sitting inside this county.
            if v == dp and v > 0.0:
                whole_co[c] += v
                vp_c = 0.0
            else:
                vp_c = v
            sqrt_partial_co[c] += np.sqrt(vp_c / safe_cp)
            # District direction: whole county sitting inside this district.
            safe_dp = dp if dp > 0.0 else 1.0
            if v == cp and v > 0.0:
                whole_d[d] += v
                vp_d = 0.0
            else:
                vp_d = v
            sqrt_partial_d[d] += np.sqrt(vp_d / safe_dp)
    raw_county = 0.0
    for c in range(n_counties):
        cp = county_pops[c]
        safe_cp = cp if cp > 0.0 else 1.0
        cs = sqrt_partial_co[c] + np.sqrt(whole_co[c] / safe_cp)
        raw_county += cp / state_pop * cs
    raw_district = 0.0
    for d in range(n_districts):
        dp = district_pops[d]
        safe_dp = dp if dp > 0.0 else 1.0
        ds = sqrt_partial_d[d] + np.sqrt(whole_d[d] / safe_dp)
        raw_district += dp / state_pop * ds
    return raw_county, raw_district


def _best_target(n_counties: int, n_districts: int) -> float:
    """Smoothly interpolate the rating's `best` target between MIN and MAX based
    on the count imbalance between counties and districts."""
    more = max(n_counties, n_districts)
    less = min(n_counties, n_districts)
    if more <= 0:
        return _MAX_SPLITTING
    w1 = (less - 1) / more
    return w1 * _MAX_SPLITTING + (1.0 - w1) * _MIN_SPLITTING


def _direction_penalty(raw: float, floor: float, worst: float) -> float:
    """Clipped per-direction split penalty in [0, 100]: 0 at/under `floor`,
    rising linearly to 100 at `worst`, pinned there beyond. Scorecard form only
    (the unclipped path uses a plain slope from raw = 1.0, no band)."""
    if worst <= floor:
        return 0.0
    u = max(0.0, min(1.0, (raw - floor) / (worst - floor)))
    return u * 100.0


def score_holistic_splitting(
    assignment: np.ndarray,
    county_ids: np.ndarray,
    populations: np.ndarray,
    n_districts: int,
    county_data=None,
    co_di_pop=None,
    unclipped: bool = False,
) -> tuple[float, float, float]:
    """
    Args:
        assignment:   (n,) int32 district indices 0..k-1
        county_ids:   (n,) int32 county indices 0..C-1
        populations:  (n,) int/float precinct populations
        n_districts:  number of districts k
        county_data:  optional CountyData with cached county_pops + n_counties
        unclipped:    if True (default in practice), use the flat-slope annealing
                      form (penalty = slope * (raw - 1.0), no band or cap); if
                      False, the clipped scorecard band

    Returns:
        raw_county   -- county-direction split score (1.0 = no splits in this direction)
        raw_district -- district-direction split score (1.0 = no splits in this direction)
        penalty      -- combined penalty (lower = less split); [0, inf) unclipped,
                        [0, 100 * _PENALTY_SCALE] clipped
    """
    if county_data is not None:
        n_counties  = county_data.n_counties
        county_pops = county_data.county_pops
        pops_f      = county_data.pops_f
    else:
        n_counties  = int(county_ids.max()) + 1
        pops_f      = populations.astype(np.float64)
        county_pops = np.bincount(county_ids, weights=pops_f, minlength=n_counties)

    # CxD matrix: co_di_pop[c, d] = total pop of (county c ∩ district d).
    # May be supplied by score_plan (shared with score_county_splits) to avoid
    # rebuilding the identical matrix twice per iteration.
    if co_di_pop is None:
        flat_idx   = (county_ids * n_districts + assignment).astype(np.int64)
        co_di_flat = np.bincount(flat_idx, weights=pops_f,
                                 minlength=n_counties * n_districts)
        co_di_pop  = co_di_flat.reshape(n_counties, n_districts)

    district_pops = co_di_pop.sum(axis=0)
    state_pop = float(county_pops.sum())

    if state_pop == 0.0 or n_counties == 0 or n_districts == 0:
        return 1.0, 1.0, 0.0

    # ── Raw split scores in both directions (fused Numba kernel) ──
    raw_county, raw_district = _hsplit_raw_numba(
        co_di_pop, county_pops, district_pops, state_pop,
        n_counties, n_districts,
    )
    raw_county = float(raw_county)
    raw_district = float(raw_district)

    if unclipped:
        # Annealing form (default): penalty rises at a flat slope from the true
        # no-split floor (raw = 1.0) -- no arbitrary band, no cap. _UNCLIPPED_SLOPE
        # is the only knob and it is pure magnitude (the weight tunes the rest).
        pen_c = _UNCLIPPED_SLOPE * max(0.0, raw_county - 1.0)
        pen_d = _UNCLIPPED_SLOPE * max(0.0, raw_district - 1.0)
        penalty = 0.5 * pen_c + 0.5 * pen_d
    else:
        # Clipped scorecard form: 0 at the map ideal `ref`, capped at 100 at
        # worst = ref * 1.33. _PENALTY_SCALE keeps its magnitude near the
        # unclipped form's so toggling the mode doesn't reprice the weight.
        if n_counties > n_districts:
            ref_c = _best_target(n_counties, n_districts)
            ref_d = _MAX_SPLITTING
        else:
            ref_c = _MAX_SPLITTING
            ref_d = _best_target(n_counties, n_districts)
        pen_c = _direction_penalty(raw_county,   ref_c, ref_c * _WORST_MULTIPLIER)
        pen_d = _direction_penalty(raw_district, ref_d, ref_d * _WORST_MULTIPLIER)
        penalty = _PENALTY_SCALE * (0.5 * pen_c + 0.5 * pen_d)
    return raw_county, raw_district, penalty
