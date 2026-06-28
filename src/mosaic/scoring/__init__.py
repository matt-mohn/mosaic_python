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
from mosaic.scoring.cache import (
    get_pp_cache_path, load_cached_pp_data, save_cached_pp_data,
)
from mosaic.scoring.county_splits import score_county_splits
from mosaic.scoring.polsby_popper import score_polsby_popper
from mosaic.scoring.reock import (
    ReockData, precompute_reock_data, score_reock,
)
from mosaic.scoring.alignment import (
    AlignmentData, AlignmentError, precompute_alignment_data, score_alignment,
)
from mosaic.scoring.holistic_compactness import holistic_compactness_from_scores
from mosaic.scoring.holistic_proportionality import holistic_proportionality_from_shares
from mosaic.scoring.holistic_competitiveness import holistic_competitiveness_from_shares
from mosaic.scoring.holistic_splitting import score_holistic_splitting
from mosaic.scoring.partisan import (
    district_dem_shares,
    eg_from_shares,
    k_to_sigma,
    p_win_gaussian,
    score_mean_median,
    score_efficiency_gap,
    score_dem_seats,
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
    "get_pp_cache_path",
    "load_cached_pp_data",
    "save_cached_pp_data",
    "score_county_splits",
    "score_polsby_popper",
    "ReockData",
    "precompute_reock_data",
    "score_reock",
    "AlignmentData",
    "AlignmentError",
    "precompute_alignment_data",
    "score_alignment",
    "holistic_compactness_from_scores",
    "holistic_proportionality_from_shares",
    "holistic_competitiveness_from_shares",
    "score_holistic_splitting",
    "district_dem_shares",
    "eg_from_shares",
    "k_to_sigma",
    "p_win_gaussian",
    "score_mean_median",
    "score_efficiency_gap",
    "score_dem_seats",
    "score_majority_chance",
]
