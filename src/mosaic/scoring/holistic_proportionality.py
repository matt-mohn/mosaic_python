"""
Holistic Proportionality — single 0-100 rating of seat-share vs vote-share fairness.

Inputs piggyback on Mosaic's existing partisan calibration (shares, total_d,
sigma_comb) so the probability model is consistent with Expected Dem Seats,
Competitiveness, Chance of Majority, and Hinge.

Curve:
  1. Vf      = vote-weighted statewide Dem two-party share.
  2. bestSf  = round(N*Vf) / N — the integer-proportional seat share.
  3. estSf   = sum(Phi((shares - 0.5) / sigma_comb)) / N — soft expected seat share.
  4. raw     = bestSf - estSf (signed; positive = under-seated relative to votes).
  5. Antimajoritarian short-circuit: if the FPTP winner (majority of districts)
     contradicts the statewide vote winner with 2pp slack, return max penalty.
  6. Winner's-bonus discount: if the bias points the same way as the statewide
     winner, shrink it by |Vf - 0.5| points (capped so the adjusted bias never
     crosses zero).
  7. abs(adjusted) -> clip to [0, 0.20] -> linear map -> invert -> penalty form.

Saturation: |adjusted| >= 0.20 saturates at 100 penalty. The antimajoritarian
check kicks in only when both vote and seat sides disagree by more than 2pp.
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr


_AVG_SV_ERROR = 0.02   # slack on antimajoritarian check
_WINNER_BONUS = 2.0    # 1pp extra seat share per pp of statewide vote share above 50
_CLIP_MAX_DEV = 0.20   # |adjusted| above this saturates at 100 penalty


def holistic_proportionality_from_shares(
    shares: np.ndarray,
    total_d: np.ndarray,
    sigma_comb: float,
) -> tuple[float, float]:
    """
    Returns:
        adjusted -- signed adjusted deviation (display; positive = D under-seated)
        penalty  -- [0, 100] (lower = more proportional)
    """
    n = len(shares)
    total_votes = float(total_d.sum())
    if total_votes == 0.0 or n == 0:
        return 0.0, 0.0

    Vf = float((shares * total_d).sum() / total_votes)

    if sigma_comb <= 0.0:
        p_wins = (shares > 0.5).astype(np.float64)
    else:
        p_wins = ndtr((shares - 0.5) / sigma_comb)
    est_sf = float(p_wins.sum() / n)

    best_sf = round(n * Vf - 1e-9) / n
    raw = best_sf - est_sf

    # Antimajoritarian short-circuit (uses observed FPTP outcome)
    fptp_seats = int((shares > 0.5).sum())
    sf_obs = fptp_seats / n
    am_dem = (Vf < (0.5 - _AVG_SV_ERROR)) and (sf_obs > 0.5)
    am_rep = ((1.0 - Vf) < (0.5 - _AVG_SV_ERROR)) and ((1.0 - sf_obs) > 0.5)
    if am_dem or am_rep:
        return raw, 100.0

    # Winner's-bonus discount
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
        return adjusted, 100.0
    rating = (1.0 - abs_adj / _CLIP_MAX_DEV) * 100.0
    return adjusted, 100.0 - rating
