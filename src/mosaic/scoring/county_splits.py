"""
County-splits score (lower = better, 0 = no avoidable county splits).

Algorithm:
  For each county:
    allowance = ceil(county_pop / ideal_pop)   -- how many districts it can host
    excess_splits = max(0, n_districts_present - allowance)
    count_penalty += excess_splits

  clean_districts = districts whose precincts are all in one county
  max_clean       = sum_counties floor(county_pop / min_district_pop)

  total = count_penalty + (max_clean - clean_districts)
"""

from __future__ import annotations

import numpy as np


def score_county_splits(
    assignment: np.ndarray,
    county_ids: np.ndarray,
    populations: np.ndarray,
    ideal_pop: float,
    tolerance: float,
    n_districts: int,
) -> tuple[float, int, int]:
    """
    Args:
        assignment:  (n,) int32 district indices 0..k-1
        county_ids:  (n,) int32 county indices 0..C-1
        populations: (n,) int/float precinct populations
        ideal_pop:   target population per district
        tolerance:   fractional pdev tolerance (e.g. 0.05)
        n_districts: number of districts k

    Returns:
        tuple of (score, excess_splits, clean_districts)
    """
    n_counties = int(county_ids.max()) + 1
    pops_f = populations.astype(np.float64)

    # County populations
    county_pops = np.bincount(county_ids, weights=pops_f, minlength=n_counties)

    # Joint pop matrix: co_di_pop[c, d] = pop of precincts in county c AND district d
    flat_idx = (county_ids * n_districts + assignment).astype(np.int64)
    co_di_flat = np.bincount(flat_idx, weights=pops_f,
                              minlength=n_counties * n_districts)
    co_di_pop = co_di_flat.reshape(n_counties, n_districts)

    # Splits per county = number of distinct districts with any precincts there
    splits = (co_di_pop > 0).sum(axis=1)   # (n_counties,)

    # Allowances
    safe_ideal = max(ideal_pop, 1.0)
    allowances = np.ceil(county_pops / safe_ideal).astype(np.int32)

    # Count penalty: excess districts beyond allowance
    excess = np.maximum(0, splits - allowances)
    count_penalty = float(excess.sum())

    # Clean districts: districts whose precincts are entirely within one county
    # n_counties_per_district[d] = #counties that have any precincts in district d
    n_co_per_dist = (co_di_pop > 0).sum(axis=0)   # (n_districts,)
    clean_districts = int((n_co_per_dist == 1).sum())

    # Max possible clean districts given population constraints
    min_district_pop = max(ideal_pop * (1.0 - tolerance), 1.0)
    max_clean = int(np.floor(county_pops / min_district_pop).sum())

    return count_penalty + float(max_clean - clean_districts), int(excess.sum()), clean_districts
