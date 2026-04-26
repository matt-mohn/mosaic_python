"""
Partisan scoring metrics for redistricting plans.

Metrics
-------
Mean-Median Difference  (MM)  -- exponent 2
Efficiency Gap          (EG)  -- exponent 2
  Static:  vote-weighted EG at current election environment
  Robust:  weighted average across 9 uniform-swing scenarios
           swings = -8%..+8% in 2% steps, weights = normal(mu=0, sigma~3%)
Expected Dem Seats      (DS)  -- exponent 2
Competitiveness         (CP)  -- exponent 1  (no target)

EG is vote-weighted: (total_wasted_dem - total_wasted_rep) / total_votes.
Total votes per district are held fixed in robust swing scenarios.

Logistic calibration: P(D wins | share=0.55) == win_prob_at_55
  k = log(p / (1-p)) / 0.05
"""

from __future__ import annotations

import numpy as np

# Robust EG: uniform-swing scenarios with normal-distribution weights (sigma ~3%)
_ROBUST_SWINGS  = np.array([-0.08, -0.06, -0.04, -0.02, 0.00,
                              0.02,  0.04,  0.06,  0.08], dtype=np.float64)
_ROBUST_WEIGHTS = np.array([ 0.007,  0.037,  0.108,  0.218,  0.272,
                              0.218,  0.108,  0.037,  0.007], dtype=np.float64)


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
    shares = np.where(total_d > 0, dem_d / total_d, 0.5)
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
) -> tuple[float, float]:
    """
    Returns:
        raw     -- EG value (weighted average when robust, single-env when static)
        penalty -- ((raw - target) * 100)^2  (scaled to pp for weight comparability)

    Robust: weighted average of vote-weighted EG across 9 uniform-swing scenarios.
      District vote totals are held fixed; only the partisan split shifts.
    Static: vote-weighted EG at the current partisan split.
    """
    shares, total_d = district_dem_shares(assignment, dem_votes, gop_votes, n_districts)

    if robust:
        eg_sum = 0.0
        for swing, w in zip(_ROBUST_SWINGS, _ROBUST_WEIGHTS):
            swung = np.clip(shares + swing, 0.0, 1.0)
            eg_sum += _eg_votes(swung, total_d) * w
        raw = eg_sum
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
