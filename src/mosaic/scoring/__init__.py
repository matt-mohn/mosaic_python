"""Scoring functions for redistricting plans."""

from mosaic.scoring.alignment import (
    AlignmentData,
    AlignmentError,
    precompute_alignment_data,
    score_alignment,
)
from mosaic.scoring.cache import (
    get_pp_cache_path,
    load_cached_pp_data,
    save_cached_pp_data,
)
from mosaic.scoring.community_congruence import (
    CommunityCongruenceData,
    precompute_community_congruence_data,
    score_community_congruence,
)
from mosaic.scoring.county_splits import score_county_splits
from mosaic.scoring.holistic_compactness import holistic_compactness_from_scores
from mosaic.scoring.holistic_competitiveness import holistic_competitiveness_from_shares
from mosaic.scoring.holistic_proportionality import holistic_proportionality_from_shares
from mosaic.scoring.holistic_splitting import score_holistic_splitting
from mosaic.scoring.minority_cohesion import (
    MinorityCohesionData,
    precompute_minority_cohesion_data,
    score_minority_cohesion,
)
from mosaic.scoring.opportunity import (
    DISCOUNT,
    GROUPS,
    OpportunityResult,
    compute_opportunity,
    precompute_opportunity_coords,
    warm_opportunity_geo,
)
from mosaic.scoring.partisan import (
    district_dem_shares,
    eg_from_shares,
    k_to_sigma,
    p_win_gaussian,
    score_dem_seats,
    score_efficiency_gap,
    score_majority_chance,
    score_mean_median,
    score_partisan_bias,
    score_partisan_gini,
)
from mosaic.scoring.polsby_popper import score_polsby_popper
from mosaic.scoring.population import (
    PlanStats,
    calculate_population_score,
    count_cut_edges,
    get_plan_stats,
)
from mosaic.scoring.precompute import PPData, find_county_array, precompute_pp_data
from mosaic.scoring.reock import (
    ReockData,
    precompute_reock_data,
    score_reock,
)
from mosaic.scoring.representation import representation_from_opportunity
from mosaic.scoring.score import PlanScore, ScoreConfig, score_plan

__all__ = [
    "count_cut_edges",
    "get_plan_stats",
    "calculate_population_score",
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
    "score_partisan_bias",
    "score_partisan_gini",
    "GROUPS",
    "DISCOUNT",
    "OpportunityResult",
    "compute_opportunity",
    "precompute_opportunity_coords",
    "warm_opportunity_geo",
    "representation_from_opportunity",
    "MinorityCohesionData",
    "precompute_minority_cohesion_data",
    "score_minority_cohesion",
    "CommunityCongruenceData",
    "precompute_community_congruence_data",
    "score_community_congruence",
]
