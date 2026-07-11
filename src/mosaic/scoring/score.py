"""
Weighted scoring for redistricting plans.

Adding a new metric:
  1. Add `weight_<name>: float = 0.0` to ScoreConfig.
  2. Add `<name>: float = 0.0` to PlanScore.
  3. Add one guarded branch in score_plan().
  4. Add a toggle + slider in the GUI Score section.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from mosaic.scoring.precompute import PPData
from mosaic.scoring.reock import ReockData

# Module-scope scorer imports: nothing imports score.py back, so lazy
# per-call imports would just be hot-loop overhead.
from mosaic.scoring.county_splits import score_county_splits
from mosaic.scoring.holistic_compactness import holistic_compactness_from_scores
from mosaic.scoring.holistic_proportionality import holistic_proportionality_from_shares
from mosaic.scoring.holistic_competitiveness import holistic_competitiveness_from_shares
from mosaic.scoring.holistic_splitting import score_holistic_splitting
from mosaic.scoring.polsby_popper import score_polsby_popper
from mosaic.scoring.reock import score_reock
from mosaic.scoring.alignment import score_alignment
from mosaic.scoring.population import score_pop_deviation
from mosaic.scoring.precompute import build_county_district_matrix
from mosaic.scoring.partisan import (
    district_dem_shares, k_to_sigma, build_p_wins_matrix,
    score_mean_median, score_efficiency_gap,
    score_dem_seats,
    score_majority_chance, score_hinge_chance,
    score_partisan_bias, score_partisan_gini,
)


@dataclass
class ScoreConfig:
    """Weights per metric.  weight=0.0 excludes the metric from the total."""
    weight_cut_edges: float = 1.0
    weight_county_excess: float = 0.0   # over-allowance splits (county-side view)
    weight_county_unified: float = 0.0  # missed single-county districts
    weight_holistic_splitting: float = 0.0
    holistic_splitting_unclipped: bool = True   # uncapped penalty (annealing gradient)
    weight_polsby_popper: float = 0.0
    weight_reock: float = 0.0
    weight_holistic_compactness: float = 0.0
    weight_pop_deviation: float = 0.0
    pop_deviation_safe_harbor: float = 0.0   # fractional; 0 = no safe harbor
    weight_alignment: float = 0.0       # least-change vs a reference plan
    alignment_party_focus: str = "none"      # "none" | "rep" | "dem": whose voters
    alignment_restrict_to_party: bool = False  # only score that party's won districts
    alignment_win_threshold: float = 0.535     # two-party share to count as "won"
    # Partisan metrics (require election data; all off by default)
    weight_mean_median: float = 0.0
    mm_mode: str = "fair"               # "fair" | "favor_dem" | "favor_rep"
    mm_bound: float = 0.20
    weight_efficiency_gap: float = 0.0
    eg_mode: str = "fair"
    eg_bound: float = 0.35
    use_robust_eg: bool = True
    partisan_quadratic_penalty: bool = False   # advanced toggle in Partisanship Settings popup
    weight_partisan_bias: float = 0.0
    pbias_mode: str = "fair"            # "fair" | "favor_dem" | "favor_rep"
    pbias_bound: float = 0.25
    weight_partisan_gini: float = 0.0   # fair-only (bound is a module constant)
    weight_dem_seats: float = 0.0
    dem_seats_favor_dem: bool = True   # True = optimize toward more D seats, False = more R
    weight_holistic_proportionality: float = 0.0
    weight_holistic_competitiveness: float = 0.0
    competitiveness_unclipped: bool = True   # two-segment knee form (annealing gradient)
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
    county_excess_score: float = 0.0    # SCORE_SCALE * over-allowance splits
    county_unified_score: float = 0.0   # SCORE_SCALE * (max_unified - unified_districts)
    holistic_splitting: float = 0.0     # combined split penalty (0 = best)
    polsby_popper: float = 0.0          # stored as 1 - mean_PP (penalty form)
    reock: float = 0.0                  # stored as 1 - mean_Reock (penalty form)
    holistic_compactness: float = 0.0   # stored as 100 - rating (penalty form)
    pop_deviation: float = 0.0          # mean squared excess dev × 10,000
    pop_dev_max: float = 0.0            # max |deviation| as % (display only)
    pop_dev_mean: float = 0.0           # mean |deviation| as % (display only)
    alignment: float = 0.0             # 100 * weighted mean (1 - cohesion) penalty
    alignment_mean_ret: float = 0.0    # mean district cohesion as % (display only)
    alignment_min_ret: float = 0.0     # worst-district cohesion as % (display only)
    county_excess_splits: int = 0
    county_unified_districts: int = 0
    # Partisan raw metric values (before target penalty; for display)
    mean_median: float = 0.0           # actual MM = mean(shares) - median(shares)
    efficiency_gap: float = 0.0        # actual EG (at swung shares when robust)
    dem_seats: float = 0.0             # expected number of Dem seats (raw, for display)
    dem_seats_penalty: float = 0.0     # directional linear [0, 100] penalty (lower = better)
    holistic_proportionality: float = 0.0  # [0, 100] penalty (lower = more proportional)
    holistic_competitiveness: float = 0.0  # [0, 100] penalty (lower = more competitive)
    partisan_bias: float = 0.0             # raw: 0.5 - D seat share at a tied vote (+ = pro-R)
    partisan_gini: float = 0.0             # [0, 100] penalty (lower = more symmetric)
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
    reock_data: Optional[ReockData] = None,
    alignment_data=None,
    county_data=None,
    n_districts: Optional[int] = None,
    dem_votes: Optional[np.ndarray] = None,
    gop_votes: Optional[np.ndarray] = None,
    real_edge_mask: Optional[np.ndarray] = None,
    force_pop_components: bool = False,
) -> PlanScore:
    """
    Compute the weighted plan score.

    cut_edge_indices is always required (already in the hot loop).
    All other kwargs are only used when the corresponding weight is > 0.

    real_edge_mask, when given, is a boolean array over ctx.edge_u/edge_v that
    is False for virtual bridge edges; the cut-edge count then excludes them so
    island bridges stay invisible to scoring. Without it, all cut edges count.
    """
    if real_edge_mask is not None and len(cut_edge_indices):
        cut_edges = int(real_edge_mask[cut_edge_indices].sum())
    else:
        cut_edges = len(cut_edge_indices)
    total = config.weight_cut_edges * cut_edges
    cs_excess_score = cs_unified_score = pp_raw = reock_raw = hc_raw = hsplit_raw = pd_raw = 0.0
    align_raw = 0.0
    align_mean_ret = align_min_ret = 0.0
    cs_excess = cs_unified = 0
    mm_raw = eg_raw = seats_raw = 0.0
    seats_penalty = 0.0
    hprop_pen = hcmp_pen = 0.0
    bias_raw = 0.0
    gini_pen = 0.0
    maj_d_raw = maj_r_raw = hinge_raw = 0.0

    # excess/unified county scorers and holistic_splitting all need the same
    # CxD population matrix; build it once and share it.
    _co_di_pop = None
    _need_cxd = bool(config.weight_county_excess or config.weight_county_unified
                     or config.weight_holistic_splitting)
    if _need_cxd and assignment is not None and county_ids is not None \
            and county_data is not None and n_districts is not None:
        _co_di_pop = build_county_district_matrix(
            assignment, county_ids, n_districts, county_data,
        )

    if (config.weight_county_excess or config.weight_county_unified) \
            and assignment is not None \
            and county_ids is not None and populations is not None \
            and ideal_pop is not None and n_districts is not None:
        cs_excess_score, cs_unified_score, cs_excess, cs_unified = score_county_splits(
            assignment, county_ids, populations, ideal_pop,
            tolerance or 0.05, n_districts,
            county_data=county_data, co_di_pop=_co_di_pop,
        )
        total += config.weight_county_excess * cs_excess_score
        total += config.weight_county_unified * cs_unified_score

    if config.weight_holistic_splitting and assignment is not None \
            and county_ids is not None and populations is not None \
            and n_districts is not None:
        _hs_rc, _hs_rd, hsplit_raw = score_holistic_splitting(
            assignment, county_ids, populations, n_districts,
            county_data=county_data, co_di_pop=_co_di_pop,
            unclipped=config.holistic_splitting_unclipped,
        )
        total += config.weight_holistic_splitting * hsplit_raw

    # PP and Reock are feeders for Holistic Compactness; compute when either
    # their own weight or Holistic's weight is active so we never re-run them.
    need_pp = bool(config.weight_polsby_popper or config.weight_holistic_compactness)
    if need_pp and assignment is not None and pp_data is not None \
            and n_districts is not None:
        pp_raw = score_polsby_popper(assignment, pp_data, n_districts)
        if config.weight_polsby_popper:
            total += config.weight_polsby_popper * pp_raw

    need_reock = bool(config.weight_reock or config.weight_holistic_compactness)
    if need_reock and assignment is not None and reock_data is not None \
            and n_districts is not None:
        reock_raw = score_reock(assignment, reock_data, n_districts)
        if config.weight_reock:
            total += config.weight_reock * reock_raw

    if config.weight_holistic_compactness and pp_data is not None \
            and reock_data is not None and assignment is not None \
            and n_districts is not None:
        hc_raw = holistic_compactness_from_scores(pp_raw, reock_raw)
        total += config.weight_holistic_compactness * hc_raw

    pd_max = pd_mean = 0.0
    # force_pop_components exposes pd_max to the Tolerance Ratchet when the
    # deviation score is unweighted (the weight-0 penalty is then a no-op).
    if (config.weight_pop_deviation or force_pop_components) \
            and assignment is not None \
            and populations is not None and ideal_pop is not None \
            and n_districts is not None:
        pd_raw, pd_max, pd_mean = score_pop_deviation(
            assignment, populations, ideal_pop, n_districts,
            safe_harbor=config.pop_deviation_safe_harbor,
            return_components=True,
        )
        total += config.weight_pop_deviation * pd_raw

    if config.weight_alignment and assignment is not None \
            and alignment_data is not None and populations is not None \
            and n_districts is not None:
        # Ask 1 — whose voters: measure retention in a party's votes if focused
        # and that party's per-precinct votes are present; else fall back to pop.
        focus = config.alignment_party_focus
        if focus == "rep" and gop_votes is not None:
            align_weights = gop_votes
        elif focus == "dem" and dem_votes is not None:
            align_weights = dem_votes
        else:
            focus = "none"
            align_weights = populations

        # Ask 2 — which districts: restrict to reference districts the focus
        # party "wins" (two-party share > threshold). Uses the reference's own
        # per-district totals cached at load, so the set stays frozen to the
        # reference plan as the proposed map evolves.
        align_mask = None
        if config.alignment_restrict_to_party and focus in ("rep", "dem") \
                and alignment_data.alt_dem_by_district is not None \
                and alignment_data.alt_gop_by_district is not None:
            dd = alignment_data.alt_dem_by_district
            gg = alignment_data.alt_gop_by_district
            tot = dd + gg
            party_v = gg if focus == "rep" else dd
            with np.errstate(invalid="ignore", divide="ignore"):
                share = np.where(tot > 0, party_v / tot, 0.0)
            align_mask = share > config.alignment_win_threshold

        align_raw, align_mean_ret, align_min_ret = score_alignment(
            assignment,
            alignment_data.alt_assignment,
            align_weights,
            alignment_data.n_alt_districts,
            n_districts,
            district_mask=align_mask,
            return_components=True,
        )
        total += config.weight_alignment * align_raw

    # Partisan metrics — only run when election data is available
    has_election = (dem_votes is not None and gop_votes is not None
                    and assignment is not None and n_districts is not None)

    if has_election:
        # Compute once; pass to all partisan functions to avoid redundant work.
        _shares, _total_d = district_dem_shares(
            assignment, dem_votes, gop_votes, n_districts)
        _sigma_d = k_to_sigma(config.election_win_prob_at_55)
        _sigma_comb = float(np.sqrt(
            config.election_swing_sigma ** 2 + _sigma_d ** 2))
        # majority_chance and hinge_chance (both always computed below) need the
        # identical (M, n) Gauss-Hermite win-prob matrix. Build it once here.
        _p_wins = build_p_wins_matrix(
            _shares, _sigma_d, config.election_swing_sigma)

        mm_raw, mm_penalty = score_mean_median(
            assignment, dem_votes, gop_votes, n_districts,
            mode=config.mm_mode,
            bound=config.mm_bound,
            quadratic_penalty=config.partisan_quadratic_penalty,
            _shares=_shares,
        )
        if config.weight_mean_median:
            total += config.weight_mean_median * mm_penalty

        eg_raw, eg_penalty = score_efficiency_gap(
            assignment, dem_votes, gop_votes, n_districts,
            mode=config.eg_mode,
            bound=config.eg_bound,
            quadratic_penalty=config.partisan_quadratic_penalty,
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
            favor_dem=config.dem_seats_favor_dem,
            win_prob_at_55=config.election_win_prob_at_55,
            swing_sigma=config.election_swing_sigma,
            _shares=_shares,
            _sigma_d=_sigma_d,
        )
        if config.weight_dem_seats:
            total += config.weight_dem_seats * seats_penalty

        # Holistic scores piggyback on the shared partisan calibration so they
        # rank plans consistently with the rest of the partisan stack.
        _, hprop_pen = holistic_proportionality_from_shares(
            _shares, _total_d, _sigma_comb,
        )
        if config.weight_holistic_proportionality:
            total += config.weight_holistic_proportionality * hprop_pen

        _, hcmp_pen = holistic_competitiveness_from_shares(
            _shares, _sigma_comb, unclipped=config.competitiveness_unclipped,
        )
        if config.weight_holistic_competitiveness:
            total += config.weight_holistic_competitiveness * hcmp_pen

        # Seats-votes-curve fairness metrics (all reuse _shares/_total_d/_sigma_comb).
        bias_raw, bias_pen = score_partisan_bias(
            _shares, _total_d, _sigma_comb,
            mode=config.pbias_mode,
            bound=config.pbias_bound,
            quadratic_penalty=config.partisan_quadratic_penalty,
        )
        if config.weight_partisan_bias:
            total += config.weight_partisan_bias * bias_pen

        _, gini_pen = score_partisan_gini(_shares, _total_d, _sigma_comb)
        if config.weight_partisan_gini:
            total += config.weight_partisan_gini * gini_pen

        maj_d_raw, maj_r_raw, maj_d_pen, maj_r_pen = score_majority_chance(
            assignment, dem_votes, gop_votes, n_districts,
            win_prob_at_55=config.election_win_prob_at_55,
            swing_sigma=config.election_swing_sigma,
            _shares=_shares,
            _sigma_d=_sigma_d,
            _p_wins=_p_wins,
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
                _p_wins=_p_wins,
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
                _p_wins=_p_wins,
            )
            hinge_raw = 1.0 - p_hinge_d   # P(R wins >= threshold)
            hinge_pen = (1.0 - hinge_raw) ** 1.5 * 100.0
        if config.weight_hinge:
            total += config.weight_hinge * hinge_pen

    return PlanScore(
        total=total,
        cut_edges=cut_edges,
        county_excess_score=cs_excess_score,
        county_unified_score=cs_unified_score,
        holistic_splitting=hsplit_raw,
        county_excess_splits=cs_excess,
        county_unified_districts=cs_unified,
        polsby_popper=pp_raw,
        reock=reock_raw,
        holistic_compactness=hc_raw,
        pop_deviation=pd_raw,
        pop_dev_max=pd_max,
        pop_dev_mean=pd_mean,
        alignment=align_raw,
        alignment_mean_ret=align_mean_ret,
        alignment_min_ret=align_min_ret,
        mean_median=mm_raw,
        efficiency_gap=eg_raw,
        dem_seats=seats_raw,
        dem_seats_penalty=seats_penalty,
        holistic_proportionality=hprop_pen,
        holistic_competitiveness=hcmp_pen,
        partisan_bias=bias_raw,
        partisan_gini=gini_pen,
        majority_chance_dem=maj_d_raw,
        majority_chance_rep=maj_r_raw,
        hinge_chance=hinge_raw,
    )
