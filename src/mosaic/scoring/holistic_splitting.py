"""
Holistic Splitting — bidirectional county/district splitting rating.

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

Rating curve:
  - Asymmetric `best` target: when n_counties > n_districts (typical state legs),
    county-direction uses a looser `best_target(n_c, n_d)` and district-direction
    uses the strict MAX_SPLITTING = 1.20. Swapped when n_districts > n_counties.
  - Each direction clipped to [best, best * 1.33], linear mapped, inverted → [0, 100].
  - Combined rating = 0.5 * county_rating + 0.5 * district_rating.
  - Returned as Mosaic penalty form: `100 - combined_rating` (lower = less split).

No "reserve 100" rule (that's a scorecard convention; we want 0 = perfect for
the optimizer). No rounding (continuous gradient).
"""

from __future__ import annotations

import numpy as np


_MAX_SPLITTING    = 1.20
_MIN_SPLITTING    = 1.00
_WORST_MULTIPLIER = 1.33


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


def _continuous_rating(raw: float, best: float, worst: float) -> float:
    """Linear clipped rating in [0, 100]. Continuous (no rounding)."""
    if worst <= best:
        return 100.0
    clipped = max(best, min(worst, raw))
    unitized = (clipped - best) / (worst - best)
    return (1.0 - unitized) * 100.0


def score_holistic_splitting(
    assignment: np.ndarray,
    county_ids: np.ndarray,
    populations: np.ndarray,
    n_districts: int,
    county_data=None,
    co_di_pop=None,
) -> tuple[float, float, float]:
    """
    Args:
        assignment:   (n,) int32 district indices 0..k-1
        county_ids:   (n,) int32 county indices 0..C-1
        populations:  (n,) int/float precinct populations
        n_districts:  number of districts k
        county_data:  optional CountyData with cached county_pops + n_counties

    Returns:
        raw_county   -- county-direction split score (1.0 = no splits in this direction)
        raw_district -- district-direction split score (1.0 = no splits in this direction)
        penalty      -- combined penalty in [0, 100] (lower = less split)
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

    # ── Rating curves with asymmetric best-target ───────────────────────────
    if n_counties > n_districts:
        best_c = _best_target(n_counties, n_districts)
        best_d = _MAX_SPLITTING
    else:
        best_c = _MAX_SPLITTING
        best_d = _best_target(n_counties, n_districts)

    rating_c = _continuous_rating(raw_county,   best_c, best_c * _WORST_MULTIPLIER)
    rating_d = _continuous_rating(raw_district, best_d, best_d * _WORST_MULTIPLIER)
    combined = 0.5 * rating_c + 0.5 * rating_d
    penalty  = 100.0 - combined
    return raw_county, raw_district, penalty
