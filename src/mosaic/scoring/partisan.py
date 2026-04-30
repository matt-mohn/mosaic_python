"""
Partisan scoring metrics for redistricting plans.

Metrics
-------
Mean-Median Difference  (MM)  -- exponent 2
Efficiency Gap          (EG)  -- exponent 2
  Static:  vote-weighted EG at current election environment
  Robust:  single closed-form call with sigma_combined = sqrt(sigma_swing^2 + sigma_district^2)
Expected Dem Seats      (DS)  -- exponent 2
Competitiveness         (CP)  -- exponent 1  (no target)
Chance of Majority      (CoM) -- exponent 1.5

Unified probability model (EG Robust, DS, CP, CoM):

    P(D wins district i) = Phi((share[i] - 0.5) / sigma_combined)
    sigma_combined = sqrt(sigma_swing^2 + sigma_district^2)
    sigma_district = 0.05 / Phi^-1(win_prob_at_55)

CoM integrates the Poisson Binomial over the swing via Gauss-Hermite quadrature (M=17).
EG is vote-weighted: (total_wasted_dem - total_wasted_rep) / total_votes.
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr, ndtri

# Optional Numba JIT for the Poisson Binomial DP inner loop.
try:
    from numba import njit as _njit

    @_njit(cache=True)
    def _nb_pb_ge_batched(p_wins, threshold):
        """JIT Poisson Binomial P(sum >= threshold) for M scenarios × n districts."""
        M = p_wins.shape[0]
        n = p_wins.shape[1]
        dp = np.zeros((M, n + 1))
        for i in range(M):
            dp[i, 0] = 1.0
        for j in range(n):
            for i in range(M):
                p = p_wins[i, j]
                q = 1.0 - p
                for k in range(n, 0, -1):
                    dp[i, k] = dp[i, k] * q + dp[i, k - 1] * p
                dp[i, 0] *= q
        result = np.zeros(M)
        for i in range(M):
            s = 0.0
            for k in range(threshold, n + 1):
                s += dp[i, k]
            result[i] = s
        return result

    _NUMBA_OK = True
except ImportError:
    _NUMBA_OK = False

# Shared swing uncertainty (partisan-environment sigma).
# Default 0.03 (3pp). User-configurable via ScoreConfig.election_swing_sigma.
_EG_SWING_SIGMA: float = 0.03

# Gauss-Hermite quadrature constants for CoM swing integration (probabilists' convention).
# hermegauss integrates against exp(-x^2/2); divide by sqrt(2*pi) to integrate against N(0,1).
_GH_NODES, _GH_WEIGHTS = np.polynomial.hermite_e.hermegauss(17)
_GH_NORM: float = float(np.sqrt(2.0 * np.pi))


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
    """Per-district Gaussian sigma from the P(win | share=0.55) calibration point."""
    p = float(np.clip(win_prob_at_55, 0.501, 0.9999))
    return 0.05 / ndtri(p)


def p_win_gaussian(share: np.ndarray, sigma: float) -> np.ndarray:
    """P(D wins district) = Phi((share - 0.5) / sigma). Vectorized over districts."""
    return ndtr((share - 0.5) * (1.0 / sigma))


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
    _shares: np.ndarray | None = None,
) -> tuple[float, float]:
    """
    Returns:
        raw     -- actual mean-median value (for display)
        penalty -- ((raw - target) * 100)^2  (scaled to pp for weight comparability)
    """
    if _shares is None:
        _shares, _ = district_dem_shares(assignment, dem_votes, gop_votes, n_districts)
    _s = np.sort(_shares)
    n = len(_s)
    _med = 0.5 * (_s[n // 2 - 1] + _s[n // 2]) if n % 2 == 0 else float(_s[n // 2])
    raw = float(_shares.mean() - _med)
    return raw, ((raw - target) * 100) ** 2


def score_efficiency_gap(
    assignment: np.ndarray,
    dem_votes: np.ndarray,
    gop_votes: np.ndarray,
    n_districts: int,
    target: float = 0.0,
    robust: bool = True,
    win_prob_at_55: float = 0.9,
    _shares: np.ndarray | None = None,
    _total_d: np.ndarray | None = None,
    _sigma_d: float | None = None,
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
    if _shares is None or _total_d is None:
        _shares, _total_d = district_dem_shares(assignment, dem_votes, gop_votes, n_districts)

    if robust:
        if _sigma_d is None:
            _sigma_d = k_to_sigma(win_prob_at_55)
        sigma_comb = float(np.sqrt(_EG_SWING_SIGMA ** 2 + _sigma_d ** 2))
        raw = _eg_votes_robust(_shares, _total_d, sigma_comb)
    else:
        raw = _eg_votes(_shares, _total_d)

    return raw, ((raw - target) * 100) ** 2


def score_dem_seats(
    assignment: np.ndarray,
    dem_votes: np.ndarray,
    gop_votes: np.ndarray,
    n_districts: int,
    target: float,
    win_prob_at_55: float = 0.9,
    swing_sigma: float = _EG_SWING_SIGMA,
    _shares: np.ndarray | None = None,
    _sigma_d: float | None = None,
) -> tuple[float, float]:
    """
    Returns:
        raw     -- expected number of Dem seats (for display)
        penalty -- (raw - target)^2
    """
    if _sigma_d is None:
        _sigma_d = k_to_sigma(win_prob_at_55)
    sigma_comb = float(np.sqrt(swing_sigma ** 2 + _sigma_d ** 2))
    if _shares is None:
        _shares, _ = district_dem_shares(assignment, dem_votes, gop_votes, n_districts)
    raw = float(p_win_gaussian(_shares, sigma_comb).sum())
    return raw, (raw - target) ** 2


def score_competitiveness(
    assignment: np.ndarray,
    dem_votes: np.ndarray,
    gop_votes: np.ndarray,
    n_districts: int,
    win_prob_at_55: float = 0.9,
    swing_sigma: float = _EG_SWING_SIGMA,
    _shares: np.ndarray | None = None,
    _sigma_d: float | None = None,
) -> float:
    """
    Mean non-competitiveness = mean(|2*P(win) - 1|).
    0 = all seats perfectly competitive, 1 = all seats completely safe.
    """
    if _sigma_d is None:
        _sigma_d = k_to_sigma(win_prob_at_55)
    sigma_comb = float(np.sqrt(swing_sigma ** 2 + _sigma_d ** 2))
    if _shares is None:
        _shares, _ = district_dem_shares(assignment, dem_votes, gop_votes, n_districts)
    p_win = p_win_gaussian(_shares, sigma_comb)
    return float(np.abs(2.0 * p_win - 1.0).mean())


def _poisson_binomial_ge(p_wins: np.ndarray, threshold: int) -> float:
    """P(sum of independent Bernoullis >= threshold) via exact DP. O(n^2)."""
    n = len(p_wins)
    dp = np.zeros(n + 1)
    dp[0] = 1.0
    for p in p_wins:
        dp[1:] = dp[1:] * (1.0 - p) + dp[:-1] * p
        dp[0] *= (1.0 - p)
    return float(dp[threshold:].sum())


def _poisson_binomial_ge_batched(p_wins_matrix: np.ndarray, threshold: int) -> np.ndarray:
    """
    Vectorized P(sum >= threshold) for M independent-Bernoulli scenarios at once.

    p_wins_matrix: (M, n) — row k is the win-prob vector for scenario k.
    Returns: (M,) float array.
    """
    if _NUMBA_OK:
        return _nb_pb_ge_batched(p_wins_matrix, threshold)
    M, n = p_wins_matrix.shape
    dp = np.zeros((M, n + 1))
    dp[:, 0] = 1.0
    for j in range(n):
        p = p_wins_matrix[:, j]
        q = 1.0 - p
        dp[:, 1:] = dp[:, 1:] * q[:, None] + dp[:, :-1] * p[:, None]
        dp[:, 0] *= q
    return dp[:, threshold:].sum(axis=1)


def score_hinge_chance(
    assignment: np.ndarray,
    dem_votes: np.ndarray,
    gop_votes: np.ndarray,
    n_districts: int,
    dem_threshold: int,
    win_prob_at_55: float = 0.9,
    swing_sigma: float = _EG_SWING_SIGMA,
    _shares: np.ndarray | None = None,
    _sigma_d: float | None = None,
) -> float:
    """
    P(D wins >= dem_threshold districts), integrated over the shared swing.

    Always expressed from the D perspective; callers convert for R targets via
    P(R wins >= k) = P(D wins >= n - k + 1) → pass dem_threshold = n - k + 1.

    _shares / _sigma_d: pre-computed values from score_plan to avoid redundant work.
    Returns the probability (float in [0, 1]).
    """
    if _sigma_d is None:
        _sigma_d = k_to_sigma(win_prob_at_55)
    if _shares is None:
        _shares, _ = district_dem_shares(assignment, dem_votes, gop_votes, n_districts)

    inv_sigma_d = 1.0 / _sigma_d
    centered    = (_shares - 0.5) * inv_sigma_d

    # Build (M, n) win-prob matrix for all GH nodes at once, then one batched DP call
    node_offsets = _GH_NODES * swing_sigma * inv_sigma_d   # (M,)
    p_wins_matrix = ndtr(centered[None, :] + node_offsets[:, None])  # (M, n)
    p_per_node = _poisson_binomial_ge_batched(p_wins_matrix, dem_threshold)  # (M,)
    return float(np.dot(_GH_WEIGHTS, p_per_node)) / _GH_NORM


def score_majority_chance(
    assignment: np.ndarray,
    dem_votes: np.ndarray,
    gop_votes: np.ndarray,
    n_districts: int,
    win_prob_at_55: float = 0.9,
    swing_sigma: float = _EG_SWING_SIGMA,
    _shares: np.ndarray | None = None,
    _sigma_d: float | None = None,
) -> tuple[float, float, float, float]:
    """
    Probability each party wins a majority of districts, integrated over the shared swing.

    _shares / _sigma_d: pre-computed values from score_plan to avoid redundant work.

    Returns:
        p_dem_maj   -- P(Dems win strict majority; no ties)
        p_rep_maj   -- 1 - p_dem_maj  (exact for odd n_districts)
        dem_penalty -- (1 - p_dem_maj)^1.5 * 100
        rep_penalty -- (1 - p_rep_maj)^1.5 * 100
    """
    if _sigma_d is None:
        _sigma_d = k_to_sigma(win_prob_at_55)
    if _shares is None:
        _shares, _ = district_dem_shares(assignment, dem_votes, gop_votes, n_districts)
    majority = n_districts // 2 + 1   # strict majority: more seats than opponent, no ties

    inv_sigma_d  = 1.0 / _sigma_d
    centered     = (_shares - 0.5) * inv_sigma_d
    node_offsets = _GH_NODES * swing_sigma * inv_sigma_d          # (M,)
    p_wins_matrix = ndtr(centered[None, :] + node_offsets[:, None])  # (M, n)
    p_per_node = _poisson_binomial_ge_batched(p_wins_matrix, majority)  # (M,)
    p_dem = float(np.dot(_GH_WEIGHTS, p_per_node)) / _GH_NORM
    p_rep = 1.0 - p_dem
    return p_dem, p_rep, (1.0 - p_dem) ** 1.5 * 100.0, (1.0 - p_rep) ** 1.5 * 100.0
