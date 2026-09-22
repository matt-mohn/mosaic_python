"""Weighted scoring and aggregation for redistricting plans.

``score_plan`` returns the raw/display components in ``PlanScore`` and the
weighted penalty total minimized by the optimizer. GUI histories and display
transformations are maintained separately from this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.special import ndtr

from mosaic.scoring.alignment import score_alignment
from mosaic.scoring.community_congruence import score_community_congruence

# Module-scope scorer imports: nothing imports score.py back, so lazy
# per-call imports would just be hot-loop overhead.
from mosaic.scoring.county_splits import score_county_splits
from mosaic.scoring.holistic_compactness import holistic_compactness_from_scores
from mosaic.scoring.holistic_competitiveness import holistic_competitiveness_from_shares
from mosaic.scoring.holistic_proportionality import holistic_proportionality_from_shares
from mosaic.scoring.holistic_splitting import score_holistic_splitting
from mosaic.scoring.minority_cohesion import score_minority_cohesion
from mosaic.scoring.opportunity import compute_opportunity
from mosaic.scoring.partisan import (
    _EG_SWING_SIGMA,
    build_p_wins_matrix,
    district_dem_shares,
    k_to_sigma,
    party_display_scores,
    score_dem_seats,
    score_efficiency_gap,
    score_hinge_chance,
    score_majority_chance,
    score_mean_median,
    score_partisan_bias,
    score_partisan_gini,
)
from mosaic.scoring.polsby_popper import score_polsby_popper
from mosaic.scoring.population import score_pop_deviation
from mosaic.scoring.precompute import PPData, build_county_district_matrix
from mosaic.scoring.reock import ReockData, score_reock
from mosaic.scoring.representation import representation_from_opportunity


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
    compactness_unclipped: bool = True   # clipped then flat-landing ease-out (gradient)
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
    proportionality_unclipped: bool = True   # probabilistic form (magnitude + swing P_inv)
    weight_holistic_competitiveness: float = 0.0
    competitiveness_unclipped: bool = True   # clipped then flat-landing ease-out (gradient)
    weight_majority_chance_dem: float = 0.0
    weight_majority_chance_rep: float = 0.0
    election_win_prob_at_55: float = 0.9
    election_swing_sigma: float = 0.03
    weight_hinge: float = 0.0
    hinge_threshold: int = 1      # seat count for the selected party
    hinge_dem: bool = True        # True = D wants >= threshold; False = R
    # Demographic metrics (require one consistent demographic universe;
    # independent of election data)
    weight_representation: float = 0.0
    representation_unclipped: bool = True    # smoothed cap (gradient) vs hard scorecard
    # Passed through to representation_from_opportunity; only the proportional
    # form is currently implemented.
    representation_mode: str = "proportional"
    weight_minority_cohesion: float = 0.0    # keep minority neighborhoods intact
    weight_community_congruence: float = 0.0  # keep minority communities whole
    opportunity_midpoint: float = 0.44       # logistic center on group share
    opportunity_steepness: float = 0.05      # logistic scale
    opportunity_solid: float = 0.55          # solid-majority share = one full opportunity district
    opportunity_smart_targets: bool = True   # local-pool feasibility + ceiling


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
    pop_deviation: float = 0.0          # sum squared excess dev × 100,000 / n_districts
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
    inversion_chance: float = 0.0          # P(vote-loser controls the chamber), [0, 1]
    holistic_competitiveness: float = 0.0  # [0, 100] penalty (lower = more competitive)
    partisan_bias: float = 0.0             # raw: 0.5 - D seat share at a tied vote (+ = pro-R)
    partisan_gini: float = 0.0             # [0, 100] penalty (lower = more symmetric)
    majority_chance_dem: float = 0.0   # P(Dems win >= ceil(n/2) districts)
    majority_chance_rep: float = 0.0   # 1 - majority_chance_dem
    hinge_chance: float = 0.0          # P(selected party wins >= hinge_threshold)
    # Demographic metrics
    representation: float = 0.0            # [0, 100] penalty (lower = more proportional)
    representation_rating: float = 0.0     # [0, 100] rating (higher = better; display)
    opportunity_black: float = 0.0         # expected opportunity districts (display)
    opportunity_latino: float = 0.0
    opportunity_asian: float = 0.0
    representation_black: float = -1.0     # per-group 0-100 rating (-1 = not applicable)
    representation_latino: float = -1.0
    representation_asian: float = -1.0
    minority_cohesion: float = 0.0         # [0, 100] penalty (0 = no minority-core edges cut)
    cohesion_black: float = -1.0           # per-group community-preservation % (-1 = n/a; display)
    cohesion_latino: float = -1.0
    cohesion_asian: float = -1.0
    community_congruence: float = 0.0      # [0, 100] penalty; band PROVISIONAL, see module
    congruence_black: float = -1.0         # per-group congruence % (-1 = n/a; display)
    congruence_latino: float = -1.0
    congruence_asian: float = -1.0


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
    vap_data: Optional[dict] = None,
    minority_cohesion_data=None,
    community_congruence_data=None,
    opportunity_coords: Optional[np.ndarray] = None,
    _compactness_scores: tuple[float, float] | None = None,
    _opportunity_prep=None,
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
    hprop_pen = hcmp_pen = hprop_inv = 0.0
    bias_raw = 0.0
    gini_pen = 0.0
    maj_d_raw = maj_r_raw = hinge_raw = 0.0
    rep_pen = rep_rating = 0.0
    opp_black = opp_latino = opp_asian = 0.0
    rep_black = rep_latino = rep_asian = -1.0
    mc_pen = 0.0
    coh_black = coh_latino = coh_asian = -1.0
    cc_pen = 0.0
    cong_black = cong_latino = cong_asian = -1.0

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
        pp_raw = (_compactness_scores[0] if _compactness_scores is not None
                  else score_polsby_popper(assignment, pp_data, n_districts))
        if config.weight_polsby_popper:
            total += config.weight_polsby_popper * pp_raw

    need_reock = bool(config.weight_reock or config.weight_holistic_compactness)
    if need_reock and assignment is not None and reock_data is not None \
            and n_districts is not None:
        reock_raw = (_compactness_scores[1] if _compactness_scores is not None
                     else score_reock(assignment, reock_data, n_districts))
        if config.weight_reock:
            total += config.weight_reock * reock_raw

    if config.weight_holistic_compactness and pp_data is not None \
            and reock_data is not None and assignment is not None \
            and n_districts is not None:
        hc_raw = holistic_compactness_from_scores(
            pp_raw, reock_raw, unclipped=config.compactness_unclipped)
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
        # Measure retention in a party's votes when that focus is selected and
        # the party's per-precinct votes are present; otherwise use population.
        focus = config.alignment_party_focus
        if focus == "rep" and gop_votes is not None:
            align_weights = gop_votes
        elif focus == "dem" and dem_votes is not None:
            align_weights = dem_votes
        else:
            focus = "none"
            align_weights = populations

        # Optionally restrict scoring to reference districts the focus party
        # "wins" (two-party share > threshold). Uses the reference's own
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
        # Shared per-iteration partisan model, reused by every metric below (and
        # by the map's partisan display): district shares/totals + the two sigmas.
        _shares, _total_d = district_dem_shares(
            assignment, dem_votes, gop_votes, n_districts)
        _sigma_d = k_to_sigma(config.election_win_prob_at_55)
        _sigma_comb = float(np.sqrt(
            config.election_swing_sigma ** 2 + _sigma_d ** 2))
        _total_votes = float(_total_d.sum())
        _vote_share = (float((_shares * _total_d).sum() / _total_votes)
                       if _total_votes > 0.0 else 0.5)
        # Only combine identical arithmetic. Expected seats uses reciprocal
        # multiplication, whereas these metrics use division. Robust EG has
        # its own fixed swing sigma, so it shares only at that calibration.
        _share_eg_probs = (config.use_robust_eg
                          and config.election_swing_sigma == _EG_SWING_SIGMA)
        _need_district_probs = (_share_eg_probs or config.weight_holistic_proportionality
                                or config.weight_holistic_competitiveness)
        _p_district = (ndtr((_shares - 0.5) / _sigma_comb)
                       if _need_district_probs else None)

        # Live display tier: evaluated every proposal. Batch the unweighted case;
        # weighted objectives retain their individual penalty calculations.
        _display = None
        if (n_districts > 0 and not (config.weight_mean_median
                or config.weight_efficiency_gap or config.weight_dem_seats
                or config.weight_partisan_bias)):
            _display = party_display_scores(
                _shares, _total_d, _sigma_comb, _sigma_d, _vote_share,
                _total_votes, config.use_robust_eg, config.dem_seats_favor_dem,
                _p_district if _share_eg_probs else None)
        mm_raw, mm_penalty = (_display[0] if _display is not None else score_mean_median(
            assignment, dem_votes, gop_votes, n_districts,
            mode=config.mm_mode,
            bound=config.mm_bound,
            quadratic_penalty=config.partisan_quadratic_penalty,
            _shares=_shares,
        ))
        if config.weight_mean_median:
            total += config.weight_mean_median * mm_penalty

        eg_raw, eg_penalty = (_display[1] if _display is not None else score_efficiency_gap(
            assignment, dem_votes, gop_votes, n_districts,
            mode=config.eg_mode,
            bound=config.eg_bound,
            quadratic_penalty=config.partisan_quadratic_penalty,
            robust=config.use_robust_eg,
            win_prob_at_55=config.election_win_prob_at_55,
            _shares=_shares,
            _total_d=_total_d,
            _sigma_d=_sigma_d,
            _p_dem_wins=_p_district if _share_eg_probs else None,
            _total_votes=_total_votes,
        ))
        if config.weight_efficiency_gap:
            total += config.weight_efficiency_gap * eg_penalty

        seats_raw, seats_penalty = (_display[2] if _display is not None else score_dem_seats(
            assignment, dem_votes, gop_votes, n_districts,
            favor_dem=config.dem_seats_favor_dem,
            win_prob_at_55=config.election_win_prob_at_55,
            swing_sigma=config.election_swing_sigma,
            _shares=_shares,
            _sigma_d=_sigma_d,
            _sigma_comb=_sigma_comb,
        ))
        if config.weight_dem_seats:
            total += config.weight_dem_seats * seats_penalty

        bias_raw, bias_pen = (_display[3] if _display is not None else score_partisan_bias(
            _shares, _total_d, _sigma_comb,
            mode=config.pbias_mode,
            bound=config.pbias_bound,
            quadratic_penalty=config.partisan_quadratic_penalty,
            _vote_share=_vote_share,
        ))
        if config.weight_partisan_bias:
            total += config.weight_partisan_bias * bias_pen

        # ── Expensive tier: gated by weight. Each does real work — a full seat-
        # curve sweep (gini) or the Poisson-binomial seat-count DP (majority,
        # hinge) — so an unweighted metric is pure waste. Unweighted -> skipped,
        # and its display panel blanks (the runner appends NaN). The (M, n)
        # Gauss-Hermite win-prob matrix is shared by majority, hinge, and
        # proportionality's inversion risk; build it once, only when one is on.
        _need_pwins = bool(config.weight_majority_chance_dem
                           or config.weight_majority_chance_rep
                           or config.weight_hinge
                           or config.weight_holistic_proportionality)
        _p_wins = (build_p_wins_matrix(_shares, _sigma_d, config.election_swing_sigma)
                   if _need_pwins else None)

        if config.weight_holistic_proportionality:
            _, hprop_pen, hprop_inv = holistic_proportionality_from_shares(
                _shares, _total_d, _sigma_comb,
                unclipped=config.proportionality_unclipped,
                swing_sigma=config.election_swing_sigma,
                p_wins=_p_wins,
                _vote_share=_vote_share,
                _total_votes=_total_votes,
                _p_district=_p_district,
            )
            total += config.weight_holistic_proportionality * hprop_pen

        if config.weight_holistic_competitiveness:
            _, hcmp_pen = holistic_competitiveness_from_shares(
                _shares, _sigma_comb, unclipped=config.competitiveness_unclipped,
                _p_district=_p_district,
            )
            total += config.weight_holistic_competitiveness * hcmp_pen

        if config.weight_partisan_gini:
            _, gini_pen = score_partisan_gini(
                _shares, _total_d, _sigma_comb, _vote_share=_vote_share)
            total += config.weight_partisan_gini * gini_pen

        if config.weight_majority_chance_dem or config.weight_majority_chance_rep:
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

        if config.weight_hinge:
            if config.hinge_dem:
                dem_thr = max(1, min(config.hinge_threshold, n_districts))
                p_hinge_d = score_hinge_chance(
                    assignment, dem_votes, gop_votes, n_districts,
                    dem_threshold=dem_thr,
                    win_prob_at_55=config.election_win_prob_at_55,
                    swing_sigma=config.election_swing_sigma,
                    _shares=_shares, _sigma_d=_sigma_d, _p_wins=_p_wins,
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
                    _shares=_shares, _sigma_d=_sigma_d, _p_wins=_p_wins,
                )
                hinge_raw = 1.0 - p_hinge_d   # P(R wins >= threshold)
                hinge_pen = (1.0 - hinge_raw) ** 1.5 * 100.0
            total += config.weight_hinge * hinge_pen

    # Demographic metrics — only run when demographic data is present (independent of
    # election data). Builds the shared opportunity engine once, per proposal.
    has_race = (vap_data is not None and assignment is not None
                and n_districts is not None)
    if has_race and config.weight_representation:
        _opp = compute_opportunity(
            assignment, vap_data, n_districts,
            midpoint=config.opportunity_midpoint,
            steepness=config.opportunity_steepness,
            solid=config.opportunity_solid,
            coords=opportunity_coords,
            smart_targets=config.opportunity_smart_targets,
            _prepared=_opportunity_prep,
        )
        rep_rating, rep_pen, _rr, _reff = representation_from_opportunity(
            _opp, mode=config.representation_mode,
            unclipped=config.representation_unclipped,
        )
        opp_black, opp_latino, opp_asian = _reff["black"], _reff["latino"], _reff["asian"]
        rep_black = _rr["black"] if _rr["black"] is not None else -1.0
        rep_latino = _rr["latino"] if _rr["latino"] is not None else -1.0
        rep_asian = _rr["asian"] if _rr["asian"] is not None else -1.0
        total += config.weight_representation * rep_pen

    # Neighborhood Severance — minority-weighted cut-edge penalty (keep minority
    # neighborhoods intact). Independent of Representation. Scores the cut set:
    # which edges the plan severs, weighted by how much minority adjacency each
    # carries. District and precinct counts set the fixed race-blind expectation
    # against which the ratio is measured.
    if (config.weight_minority_cohesion and minority_cohesion_data is not None
            and n_districts is not None):
        mc_pen, _coh = score_minority_cohesion(
            cut_edge_indices, minority_cohesion_data, n_districts)
        coh_black, coh_latino, coh_asian = _coh["black"], _coh["latino"], _coh["asian"]
        total += config.weight_minority_cohesion * mc_pen

    # Community Dispersion — partition penalty (keep a minority community inside
    # as few districts as population equality allows). The complement to Minority
    # Cohesion: that score measures boundary through minority fabric, this one
    # measures how many pieces the community lands in, normalised by how many
    # pieces its size forces. Needs the assignment itself, not the cut set.
    if (config.weight_community_congruence and community_congruence_data is not None
            and assignment is not None and n_districts is not None):
        cc_pen, _cong = score_community_congruence(
            assignment, community_congruence_data, n_districts, ideal_pop)
        cong_black, cong_latino, cong_asian = (
            _cong["black"], _cong["latino"], _cong["asian"])
        total += config.weight_community_congruence * cc_pen

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
        inversion_chance=hprop_inv,
        holistic_competitiveness=hcmp_pen,
        partisan_bias=bias_raw,
        partisan_gini=gini_pen,
        majority_chance_dem=maj_d_raw,
        majority_chance_rep=maj_r_raw,
        hinge_chance=hinge_raw,
        representation=rep_pen,
        representation_rating=rep_rating,
        opportunity_black=opp_black,
        opportunity_latino=opp_latino,
        opportunity_asian=opp_asian,
        representation_black=rep_black,
        representation_latino=rep_latino,
        representation_asian=rep_asian,
        minority_cohesion=mc_pen,
        cohesion_black=coh_black,
        cohesion_latino=coh_latino,
        cohesion_asian=coh_asian,
        community_congruence=cc_pen,
        congruence_black=cong_black,
        congruence_latino=cong_latino,
        congruence_asian=cong_asian,
    )
