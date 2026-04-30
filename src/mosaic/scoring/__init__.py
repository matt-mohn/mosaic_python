"""Scoring functions for redistricting plans."""

from mosaic.scoring.population import (
    count_cut_edges,
    get_plan_stats,
    calculate_population_score,
    get_population_stats,
    PlanStats,
)
from mosaic.scoring.score import ScoreConfig, PlanScore, score_plan
from mosaic.scoring.precompute import PPData, find_county_array, precompute_pp_data
from mosaic.scoring.county_splits import score_county_splits
from mosaic.scoring.polsby_popper import score_polsby_popper
from mosaic.scoring.partisan import (
    district_dem_shares,
    eg_from_shares,
    k_to_sigma,
    p_win_gaussian,
    score_mean_median,
    score_efficiency_gap,
    score_dem_seats,
    score_competitiveness,
    score_majority_chance,
)

__all__ = [
    "count_cut_edges",
    "get_plan_stats",
    "calculate_population_score",
    "get_population_stats",
    "PlanStats",
    "ScoreConfig",
    "PlanScore",
    "score_plan",
    "PPData",
    "find_county_array",
    "precompute_pp_data",
    "score_county_splits",
    "score_polsby_popper",
    "district_dem_shares",
    "eg_from_shares",
    "k_to_sigma",
    "p_win_gaussian",
    "score_mean_median",
    "score_efficiency_gap",
    "score_dem_seats",
    "score_competitiveness",
    "score_majority_chance",
]
