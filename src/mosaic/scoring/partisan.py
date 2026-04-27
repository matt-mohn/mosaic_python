"""
Partisan scoring metrics for redistricting plans.

Metrics
-------
Mean-Median Difference  (MM)  -- exponent 2
Efficiency Gap          (EG)  -- exponent 2
  Static:  vote-weighted EG at current election environment
  Robust:  single closed-form call with sigma_combined = sqrt(sigma_swing^2 + sigma_district^2)
           Integrates out both partisan-environment uncertainty (sigma_swing = _EG_SWING_SIGMA)
           and per-district electoral noise (sigma_district derived from win_prob_at_55)
           in one shot. Mathematically exact under Gaussian swing assumption.
Expected Dem Seats      (DS)  -- exponent 2
Competitiveness         (CP)  -- exponent 1  (no target)

EG is vote-weighted: (total_wasted_dem - total_wasted_rep) / total_votes.

Logistic calibration: P(D wins | share=0.55) == win_prob_at_55
  k = log(p / (1-p)) / 0.05         (used by DS, CP)
  sigma = 0.05 / Phi^-1(p)          (used by Robust EG)

Both calibrations use the same k parameter, so adjusting win_prob_at_55
coherently updates uncertainty across all metrics.
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr, ndtri

# Robust EG: sigma of the partisan-environment uncertainty (swing distribution).
# Combined with per-district noise from win_prob_at_55 via sqrt(s_swing^2 + s_district^2).
_EG_SWING_SIGMA: float = 0.03


def election_k(win_prob_at_55: float) -> float:
    """Logistic steepness from the P(win | share=0.55) calibration point."""
    p = float(np.clip(win_prob_at_55, 0.501, 0.9999))
    return np.log(p / (1.0 - p)) / 0.05


def district_dem_shares(
    assignment: np.ndarray,
    dem_votes: np.ndarray,
    gop_votes: np.ndarray,
    n_districts: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Aggregate votes to district level.

    Returns:
        shares  -- (n_districts,) float D two-party share
        total_d -- (n_districts,) float total two-party votes per district
    """
    dem_d = np.bincount(assignment, weights=dem_votes.astype(np.float64),
                        minlength=n_districts)
    gop_d = np.bincount(assignment, weights=gop_votes.astype(np.float64),
                        minlength=n_districts)
    total_d = dem_d + gop_d
    shares = np.divide(dem_d, total_d, out=np.full(len(dem_d), 0.5), where=total_d > 0)
    return shares, total_d


def _eg_votes(shares: np.ndarray, total_d: np.ndarray) -> float:
    """
    Vote-weighted efficiency gap.

    EG = (total_wasted_dem - total_wasted_rep) / total_votes
    """
    total_votes = float(total_d.sum())
    if total_votes == 0.0:
        return 0.0
    dem_wins = shares > 0.5
    wasted_dem = np.where(dem_wins, (shares - 0.5) * total_d, shares * total_d)
    wasted_rep = np.where(dem_wins,
                          (1.0 - shares) * total_d,
                          (0.5 - shares) * total_d)
    return float((wasted_dem.sum() - wasted_rep.sum()) / total_votes)


def k_to_sigma(win_prob_at_55: float) -> float:
    """
    Convert P(D wins | share=0.55) to the equivalent Gaussian sigma on share.

    Used by Robust EG 2.0 to derive per-district uncertainty from the same k
    parameter that calibrates the logistic curve elsewhere in this module.
    """
    p = float(np.clip(win_prob_at_55, 0.501, 0.9999))
    return 0.05 / ndtri(p)


def _eg_votes_robust(shares: np.ndarray, total_d: np.ndarray,
                     sigma: float) -> float:
    """
    Expected vote-weighted EG when each district's share is N(share, sigma^2).

    Closed-form via the normal CDF -- no sampling, no RNG, fully deterministic.
    Falls back to _eg_votes when sigma <= 0.
    """
    total_votes = float(total_d.sum())
    if total_votes == 0.0:
        return 0.0
    if sigma <= 0.0:
        return _eg_votes(shares, total_d)
    p_dem_wins = ndtr((shares - 0.5) / sigma)
    contrib = total_d * ((2.0 * shares - 0.5) - p_dem_wins)
    return float(contrib.sum() / total_votes)


def eg_from_shares(shares: np.ndarray, total_d: np.ndarray) -> float:
    """Vote-weighted efficiency gap. Public alias for _eg_votes."""
    return _eg_votes(shares, total_d)


def score_mean_median(
    assignment: np.ndarray,
    dem_votes: np.ndarray,
    gop_votes: np.ndarray,
    n_districts: int,
    target: float = 0.0,
) -> tuple[float, float]:
    """
    Returns:
        raw     -- actual mean-median value (for display)
        penalty -- ((raw - target) * 100)^2  (scaled to pp for weight comparability)
    """
    shares, _ = district_dem_shares(assignment, dem_votes, gop_votes, n_districts)
    raw = float(np.mean(shares) - np.median(shares))
    return raw, ((raw - target) * 100) ** 2


def score_efficiency_gap(
    assignment: np.ndarray,
    dem_votes: np.ndarray,
    gop_votes: np.ndarray,
    n_districts: int,
    target: float = 0.0,
    robust: bool = True,
    win_prob_at_55: float = 0.9,
) -> tuple[float, float]:
    """
    Returns:
        raw     -- EG value (robust when robust=True, static when robust=False)
        penalty -- ((raw - target) * 100)^2  (scaled to pp for weight comparability)

    Robust (default): integrates out both partisan-environment uncertainty
      (sigma_swing = _EG_SWING_SIGMA = 3pp) and per-district electoral noise
      (sigma_district from win_prob_at_55) in one closed-form call:
          sigma_combined = sqrt(sigma_swing^2 + sigma_district^2)
      Mathematically exact under Gaussian swing assumption. Deterministic, no sampling.
    Static: vote-weighted EG at the current partisan split, no noise.
    """
    shares, total_d = district_dem_shares(assignment, dem_votes, gop_votes, n_districts)

    if robust:
        sigma_d    = k_to_sigma(win_prob_at_55)
        sigma_comb = float(np.sqrt(_EG_SWING_SIGMA ** 2 + sigma_d ** 2))
        raw = _eg_votes_robust(shares, total_d, sigma_comb)
    else:
        raw = _eg_votes(shares, total_d)

    return raw, ((raw - target) * 100) ** 2


def score_dem_seats(
    assignment: np.ndarray,
    dem_votes: np.ndarray,
    gop_votes: np.ndarray,
    n_districts: int,
    target: float,
    win_prob_at_55: float = 0.9,
) -> tuple[float, float]:
    """
    Returns:
        raw     -- expected number of Dem seats (for display)
        penalty -- (raw - target)^2
    """
    k = election_k(win_prob_at_55)
    shares, _ = district_dem_shares(assignment, dem_votes, gop_votes, n_districts)
    p_win = 1.0 / (1.0 + np.exp(-k * (shares - 0.5)))
    raw = float(p_win.sum())
    return raw, (raw - target) ** 2


def score_competitiveness(
    assignment: np.ndarray,
    dem_votes: np.ndarray,
    gop_votes: np.ndarray,
    n_districts: int,
    win_prob_at_55: float = 0.9,
) -> float:
    """
    Mean non-competitiveness = mean(|2*P(win) - 1|).
    0 = all seats perfectly competitive, 1 = all seats completely safe.
    """
    k = election_k(win_prob_at_55)
    shares, _ = district_dem_shares(assignment, dem_votes, gop_votes, n_districts)
    p_win = 1.0 / (1.0 + np.exp(-k * (shares - 0.5)))
    return float(np.abs(2.0 * p_win - 1.0).mean())
