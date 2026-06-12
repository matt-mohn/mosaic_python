"""
County-splits scoring (lower = better; 0 = no avoidable splits and no missed
unified districts).

Two penalty terms, returned separately so the GUI / ScoreConfig can weight
them independently:

  excess_score = SCORE_SCALE * sum_c max(0, splits[c] - allowances[c])
      "this county touches too many districts"
      allowance[c] = ceil(county_pop[c] / ideal_pop)

  unified_score = SCORE_SCALE * (max_unified - unified_districts)
      "we built fewer single-county districts than were geometrically possible"
      max_unified       = sum_c floor(county_pop[c] / min_district_pop)
      unified_districts = districts whose precincts all live in one county

SCORE_SCALE = 10 keeps each term competitive with PP / Reock at typical
weights, so weight_county_excess = 5 is the same effective pressure as
weight_polsby_popper = 5.
"""

from __future__ import annotations

import numpy as np

_SCORE_SCALE = 10.0


def score_county_splits(
    assignment: np.ndarray,
    county_ids: np.ndarray,
    populations: np.ndarray,
    ideal_pop: float,
    tolerance: float,
    n_districts: int,
    county_data=None,
) -> tuple[float, float, int, int]:
    """
    Args:
        assignment:   (n,) int32 district indices 0..k-1
        county_ids:   (n,) int32 county indices 0..C-1
        populations:  (n,) int/float precinct populations
        ideal_pop:    target population per district
        tolerance:    fractional pdev tolerance (e.g. 0.025)
        n_districts:  number of districts k
        county_data:  optional CountyData precomputed at startup; if provided,
                      the constant parts (county_pops, allowances, max_unified)
                      are taken from there instead of being recomputed.

    Returns:
        tuple of (excess_score, unified_score, excess_splits, unified_districts)
            excess_score      -- SCORE_SCALE * sum of over-allowance splits
            unified_score     -- SCORE_SCALE * (max_unified - unified_districts)
            excess_splits     -- raw integer count of over-allowance splits
            unified_districts -- raw integer count of single-county districts
    """
    if county_data is not None:
        pops_f      = county_data.pops_f
        n_counties  = county_data.n_counties
        county_pops = county_data.county_pops
        allowances  = county_data.allowances
        max_unified = county_data.max_clean
    else:
        n_counties  = int(county_ids.max()) + 1
        pops_f      = populations.astype(np.float64)
        county_pops = np.bincount(county_ids, weights=pops_f, minlength=n_counties)
        safe_ideal  = max(ideal_pop, 1.0)
        allowances  = np.ceil(county_pops / safe_ideal).astype(np.int32)
        min_dp      = max(ideal_pop * (1.0 - tolerance), 1.0)
        max_unified = int(np.floor(county_pops / min_dp).sum())

    flat_idx   = (county_ids * n_districts + assignment).astype(np.int64)
    co_di_flat = np.bincount(flat_idx, weights=pops_f,
                             minlength=n_counties * n_districts)
    co_di_pop  = co_di_flat.reshape(n_counties, n_districts)

    splits = (co_di_pop > 0).sum(axis=1)
    excess = np.maximum(0, splits - allowances)
    excess_total = int(excess.sum())

    n_co_per_dist     = (co_di_pop > 0).sum(axis=0)
    unified_districts = int((n_co_per_dist == 1).sum())

    excess_score  = _SCORE_SCALE * float(excess_total)
    unified_score = _SCORE_SCALE * float(max_unified - unified_districts)
    return excess_score, unified_score, excess_total, unified_districts
