"""
Weighted scoring for redistricting plans.

Adding a new metric:
  1. Add `weight_<name>: float = 0.0` to ScoreConfig.
  2. Add `<name>: float = 0.0` to PlanScore.
  3. Add one guarded branch in score_plan().
  4. Add a toggle + slider in the GUI Score section.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from mosaic.scoring.precompute import PPData


@dataclass
class ScoreConfig:
    """Weights per metric.  weight=0.0 excludes the metric from the total."""
    weight_cut_edges: float = 1.0
    weight_county_splits: float = 0.0
    weight_polsby_popper: float = 0.0
    weight_pop_deviation: float = 0.0
    pop_deviation_safe_harbor: float = 0.0025   # fractional; 0.0025 = 0.25%
    # Partisan metrics (require election data; all off by default)
    weight_mean_median: float = 0.0
    target_mean_median: float = 0.0
    weight_efficiency_gap: float = 0.0
    target_efficiency_gap: float = 0.0
    use_robust_eg: bool = True
    weight_dem_seats: float = 0.0
    target_dem_seats: float = 7.0
    weight_competitiveness: float = 0.0
    weight_majority_chance_dem: float = 0.0
    weight_majority_chance_rep: float = 0.0
    election_win_prob_at_55: float = 0.9
    election_swing_sigma: float = 0.03
    weight_hinge: float = 0.0
    hinge_threshold: int = 1      # seat count for the selected party
    hinge_dem: bool = True        # True = D wants >= threshold; False = R


@dataclass
class PlanScore:
    """Per-metric raw values and the weighted total the optimizer minimises."""
    total: float
    cut_edges: int
    county_splits: float = 0.0
    polsby_popper: float = 0.0          # stored as 1 - mean_PP (penalty form)
    pop_deviation: float = 0.0          # mean squared excess dev * 10000
    county_excess_splits: int = 0
    county_clean_districts: int = 0
    # Partisan raw metric values (before target penalty; for display)
    mean_median: float = 0.0           # actual MM = mean(shares) - median(shares)
    efficiency_gap: float = 0.0        # actual EG (at swung shares when robust)
    dem_seats: float = 0.0             # expected number of Dem seats
    competitiveness: float = 0.0       # mean non-competitiveness in [0, 1]
    majority_chance_dem: float = 0.0   # P(Dems win >= ceil(n/2) districts)
    majority_chance_rep: float = 0.0   # 1 - majority_chance_dem
    hinge_chance: float = 0.0          # P(selected party wins >= hinge_threshold)


def score_plan(
    cut_edge_indices: np.ndarray,
    config: ScoreConfig,
    *,
    assignment: Optional[np.ndarray] = None,
    county_ids: Optional[np.ndarray] = None,
    populations: Optional[np.ndarray] = None,
    ideal_pop: Optional[float] = None,
    tolerance: Optional[float] = None,
    pp_data: Optional[PPData] = None,
    county_data=None,
    n_districts: Optional[int] = None,
    dem_votes: Optional[np.ndarray] = None,
    gop_votes: Optional[np.ndarray] = None,
) -> PlanScore:
    """
    Compute the weighted plan score.

    cut_edge_indices is always required (already in the hot loop).
    All other kwargs are only used when the corresponding weight is > 0.
    """
    from mosaic.scoring.county_splits import score_county_splits
    from mosaic.scoring.polsby_popper import score_polsby_popper
    from mosaic.scoring.population import score_pop_deviation
    from mosaic.scoring.precompute import CountyData
    from mosaic.scoring.partisan import (
        district_dem_shares, k_to_sigma,
        score_mean_median, score_efficiency_gap,
        score_dem_seats, score_competitiveness,
        score_majority_chance, score_hinge_chance,
    )

    cut_edges = len(cut_edge_indices)
    total = config.weight_cut_edges * cut_edges
    cs_raw = pp_raw = pd_raw = 0.0
    cs_excess = cs_clean = 0
    mm_raw = eg_raw = seats_raw = comp_raw = 0.0
    maj_d_raw = maj_r_raw = hinge_raw = 0.0

    if config.weight_county_splits and assignment is not None \
            and county_ids is not None and populations is not None \
            and ideal_pop is not None and n_districts is not None:
        cs_raw, cs_excess, cs_clean = score_county_splits(
            assignment, county_ids, populations, ideal_pop,
            tolerance or 0.05, n_districts,
            county_data=county_data,
        )
        total += config.weight_county_splits * cs_raw

    if config.weight_polsby_popper and assignment is not None \
            and pp_data is not None and n_districts is not None:
        pp_raw = score_polsby_popper(assignment, pp_data, n_districts)
        total += config.weight_polsby_popper * pp_raw

    if config.weight_pop_deviation and assignment is not None \
            and populations is not None and ideal_pop is not None \
            and n_districts is not None:
        pd_raw = score_pop_deviation(
            assignment, populations, ideal_pop, n_districts,
            safe_harbor=config.pop_deviation_safe_harbor,
        )
        total += config.weight_pop_deviation * pd_raw

    # Partisan metrics — only run when election data is available
    has_election = (dem_votes is not None and gop_votes is not None
                    and assignment is not None and n_districts is not None)

    if has_election:
        # Compute once; pass to all partisan functions to avoid redundant work.
        _shares, _total_d = district_dem_shares(
            assignment, dem_votes, gop_votes, n_districts)
        _sigma_d = k_to_sigma(config.election_win_prob_at_55)

        mm_raw, mm_penalty = score_mean_median(
            assignment, dem_votes, gop_votes, n_districts,
            target=config.target_mean_median,
            _shares=_shares,
        )
        if config.weight_mean_median:
            total += config.weight_mean_median * mm_penalty

        eg_raw, eg_penalty = score_efficiency_gap(
            assignment, dem_votes, gop_votes, n_districts,
            target=config.target_efficiency_gap,
            robust=config.use_robust_eg,
            win_prob_at_55=config.election_win_prob_at_55,
            _shares=_shares,
            _total_d=_total_d,
            _sigma_d=_sigma_d,
        )
        if config.weight_efficiency_gap:
            total += config.weight_efficiency_gap * eg_penalty

        seats_raw, seats_penalty = score_dem_seats(
            assignment, dem_votes, gop_votes, n_districts,
            target=config.target_dem_seats,
            win_prob_at_55=config.election_win_prob_at_55,
            swing_sigma=config.election_swing_sigma,
            _shares=_shares,
            _sigma_d=_sigma_d,
        )
        if config.weight_dem_seats:
            total += config.weight_dem_seats * seats_penalty * 100

        comp_raw = score_competitiveness(
            assignment, dem_votes, gop_votes, n_districts,
            win_prob_at_55=config.election_win_prob_at_55,
            swing_sigma=config.election_swing_sigma,
            _shares=_shares,
            _sigma_d=_sigma_d,
        )
        if config.weight_competitiveness:
            total += config.weight_competitiveness * comp_raw * 100

        maj_d_raw, maj_r_raw, maj_d_pen, maj_r_pen = score_majority_chance(
            assignment, dem_votes, gop_votes, n_districts,
            win_prob_at_55=config.election_win_prob_at_55,
            swing_sigma=config.election_swing_sigma,
            _shares=_shares,
            _sigma_d=_sigma_d,
        )
        if config.weight_majority_chance_dem:
            total += config.weight_majority_chance_dem * maj_d_pen
        if config.weight_majority_chance_rep:
            total += config.weight_majority_chance_rep * maj_r_pen

        # Hinge — always computed so the panel updates even when weight=0
        if config.hinge_dem:
            dem_thr = max(1, min(config.hinge_threshold, n_districts))
            p_hinge_d = score_hinge_chance(
                assignment, dem_votes, gop_votes, n_districts,
                dem_threshold=dem_thr,
                win_prob_at_55=config.election_win_prob_at_55,
                swing_sigma=config.election_swing_sigma,
                _shares=_shares,
                _sigma_d=_sigma_d,
            )
            hinge_raw = p_hinge_d
            hinge_pen = (1.0 - p_hinge_d) ** 1.5 * 100.0
        else:
            # R wants >= threshold: convert to D-perspective threshold
            dem_thr = max(1, n_districts - config.hinge_threshold + 1)
            p_hinge_d = score_hinge_chance(
                assignment, dem_votes, gop_votes, n_districts,
                dem_threshold=dem_thr,
                win_prob_at_55=config.election_win_prob_at_55,
                swing_sigma=config.election_swing_sigma,
                _shares=_shares,
                _sigma_d=_sigma_d,
            )
            hinge_raw = 1.0 - p_hinge_d   # P(R wins >= threshold)
            hinge_pen = (1.0 - hinge_raw) ** 1.5 * 100.0
        if config.weight_hinge:
            total += config.weight_hinge * hinge_pen

    return PlanScore(
        total=total,
        cut_edges=cut_edges,
        county_splits=cs_raw,
        county_excess_splits=cs_excess,
        county_clean_districts=cs_clean,
        polsby_popper=pp_raw,
        pop_deviation=pd_raw,
        mean_median=mm_raw,
        efficiency_gap=eg_raw,
        dem_seats=seats_raw,
        competitiveness=comp_raw,
        majority_chance_dem=maj_d_raw,
        majority_chance_rep=maj_r_raw,
        hinge_chance=hinge_raw,
    )
