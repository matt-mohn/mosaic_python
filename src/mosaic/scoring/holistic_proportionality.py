"""
Holistic Proportionality — single 0-100 rating of seat-share vs vote-share fairness.

Inputs piggyback on Mosaic's shared partisan calibration (shares, total_d, the
swing model) so the probability model is consistent with Expected Dem Seats,
Competitiveness, Chance of Majority, and Hinge.

Two forms (both PENALTY form, lower = better, 0 = fair):

  - clipped (scorecard): point-estimate deviation of expected seat share from the
    rounded-proportional target, winner's-bonus forgiven, mapped linearly to a 0.20
    cap, with a hard antimajoritarian short-circuit to 100. Bounded but full of
    flat regions (the winner's-bonus basin at 0, the cap at 100, and a binary
    cliff for the antimajoritarian trigger).

  - unclipped (DEFAULT): fully probabilistic and smooth. Two terms:
      M      -- magnitude: expected seat/vote gap, winner's-bonus forgiven on a
                gentle slope (no flat basin), mapped linearly to [0, 100] over the
                realistic gap range.
      P_inv  -- swing-integrated probability the popular-vote LOSER controls the
                chamber (the antimajoritarian "hammer" as a probability, not a
                binary trip).
    penalty = M * (1 - P_inv) + 100 * P_inv
    a convex blend of M in [0, 100] and 100, so it is provably in [0, 100] and
    never negative. Seat ties (even chambers) count as nobody in control, so they
    add 0 to P_inv; the vote-winner boundary is smoothed (_VOTE_BLUR) so P_inv
    stays continuous through a statewide tie.

This is a *derived* score -- shares/total_d come from the shared partisan pass.
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr

from mosaic.scoring.partisan import (
    _GH_NODES,
    _GH_NORM,
    _GH_WEIGHTS,
    _poisson_binomial_ge_batched,
    build_p_wins_matrix,
)

# ── clipped scorecard form (unchanged behaviour) ─────────────────────────────
_AVG_SV_ERROR = 0.02   # slack on the antimajoritarian check
_WINNER_BONUS = 2.0    # 1pp extra seat share per pp of statewide vote share above 50
_CLIP_MAX_DEV = 0.20   # |adjusted| above this saturates at 100 penalty

# ── unclipped probabilistic form ─────────────────────────────────────────────
_MAG_LIGHT = 0.25   # within-bonus over-seating costs 1/4 rate (de-flattens the basin)
_MAG_FULL  = 0.42   # seat/vote gap at which M reaches 100 (linear below; flat above)
_VOTE_BLUR = 0.005  # smooths the vote-winner flip so P_inv is continuous through a tie


def _clipped_proportionality(shares, total_d, sigma_comb):
    """Original point-estimate form: linear-to-cap deviation with a binary
    antimajoritarian short-circuit. Kept as the clipped fallback."""
    n = len(shares)
    Vf = float((shares * total_d).sum() / total_d.sum())

    if sigma_comb <= 0.0:
        p_district = (shares > 0.5).astype(np.float64)
    else:
        p_district = ndtr((shares - 0.5) / sigma_comb)
    est_sf = float(p_district.sum() / n)

    best_sf = round(n * Vf - 1e-9) / n
    raw = best_sf - est_sf

    fptp_seats = int((shares > 0.5).sum())
    sf_obs = fptp_seats / n
    am_dem = (Vf < (0.5 - _AVG_SV_ERROR)) and (sf_obs > 0.5)
    am_rep = ((1.0 - Vf) < (0.5 - _AVG_SV_ERROR)) and ((1.0 - sf_obs) > 0.5)
    if am_dem or am_rep:
        return raw, 100.0, 1.0

    over_50 = abs(Vf - 0.5)
    extra = over_50 * (_WINNER_BONUS - 1.0)
    if Vf > 0.5 and raw < 0.0:
        adjusted = min(raw + extra, 0.0)
    elif Vf < 0.5 and raw > 0.0:
        adjusted = max(raw - extra, 0.0)
    else:
        adjusted = raw

    abs_adj = abs(adjusted)
    if abs_adj >= _CLIP_MAX_DEV:
        return adjusted, 100.0, 0.0
    rating = (1.0 - abs_adj / _CLIP_MAX_DEV) * 100.0
    return adjusted, 100.0 - rating, 0.0


def _magnitude(shares, total_d, sigma_comb):
    """Expected seat/vote gap -> [0, 100). Winner's-bonus forgiveness is a gentle
    slope (no flat basin); the map saturates so it can never exceed 100."""
    Vf = float((shares * total_d).sum() / total_d.sum())
    if sigma_comb <= 0.0:
        est_sf = float((shares > 0.5).mean())
    else:
        est_sf = float(ndtr((shares - 0.5) / sigma_comb).mean())
    s = est_sf - Vf                        # + = D over-seated vs its votes
    extra = abs(Vf - 0.5)                  # winner's-bonus allowance
    over = s if Vf >= 0.5 else -s          # winner's over-seating (signed)
    if over <= 0.0:
        d = -over                          # winner under-seated -> full slope
    elif over <= extra:
        d = _MAG_LIGHT * over              # over-seated within bonus -> gentle
    else:
        d = _MAG_LIGHT * extra + (over - extra)   # beyond bonus -> full slope
    M = 100.0 * min(d / _MAG_FULL, 1.0)    # linear to _MAG_FULL, in [0, 100]
    return Vf, M, Vf - est_sf              # display gap: + = D under-seated


def _p_inversion(shares, total_d, p_wins, swing_sigma):
    """Swing-integrated P(the popular-vote loser controls the chamber), in [0, 1].

    Seat ties (even chambers) are excluded from both majority terms, so they add
    0. The vote-winner is a smooth (_VOTE_BLUR) crossover rather than a hard
    threshold, so P_inv stays continuous through a statewide tie instead of
    jumping as swing nodes cross 0.5.
    """
    n = len(shares)
    maj = n // 2 + 1
    p_maj_d = _poisson_binomial_ge_batched(p_wins, maj)          # (M,) P(D holds | node)
    p_maj_r = _poisson_binomial_ge_batched(1.0 - p_wins, maj)    # (M,) P(R holds | node)
    Vf = float((shares * total_d).sum() / total_d.sum())
    vf_nodes = Vf + _GH_NODES * swing_sigma                      # shifted vote share per node
    w_dem_lost = ndtr((0.5 - vf_nodes) / _VOTE_BLUR)   # ~1 if D lost the vote, ~0 if R lost
    g = w_dem_lost * p_maj_d + (1.0 - w_dem_lost) * p_maj_r   # per-node inversion prob, [0, 1]
    return float(np.dot(_GH_WEIGHTS, g) / _GH_NORM)


def holistic_proportionality_from_shares(
    shares: np.ndarray,
    total_d: np.ndarray,
    sigma_comb: float,
    *,
    unclipped: bool = True,
    swing_sigma: float | None = None,
    sigma_d: float | None = None,
    p_wins: np.ndarray | None = None,
) -> tuple[float, float]:
    """
    Args:
        unclipped:   if True (default), the probabilistic form (magnitude escalated
                     by the swing-integrated inversion probability); if False, the
                     clipped scorecard form.
        swing_sigma: statewide swing sigma; required for the unclipped path.
        sigma_d:     per-district sigma; used only to build p_wins when it is not
                     supplied.
        p_wins:      shared (M, n) Gauss-Hermite win-prob matrix (reused from the
                     partisan pass); built from sigma_d/swing_sigma if omitted.

    Returns:
        display          -- signed seat/vote gap (+ = D under-seated); display only.
        penalty          -- [0, 100] (lower = more proportional).
        inversion_chance -- P(popular-vote loser controls the chamber), [0, 1];
                            the clipped form returns its binary antimajoritarian flag.
    """
    n = len(shares)
    if float(total_d.sum()) == 0.0 or n == 0:
        return 0.0, 0.0, 0.0

    if not unclipped or swing_sigma is None:
        return _clipped_proportionality(shares, total_d, sigma_comb)

    if p_wins is None:
        if sigma_d is None:
            return _clipped_proportionality(shares, total_d, sigma_comb)
        p_wins = build_p_wins_matrix(shares, sigma_d, swing_sigma)

    _, M, gap = _magnitude(shares, total_d, sigma_comb)
    p_inv = _p_inversion(shares, total_d, p_wins, swing_sigma)
    penalty = M * (1.0 - p_inv) + 100.0 * p_inv    # convex blend -> [0, 100], >= 0
    return gap, penalty, p_inv
