"""
Holistic Competitiveness — single 0-100 rating using a quadratic competitiveness kernel.

Per-district soft competitiveness = 1 - 4 * (P_win - 0.5)**2:
  P_win = 0.5      -> kernel = 1.0   (perfect toss-up)
  P_win in {0, 1}  -> kernel = 0     (locked seat)

Fraction of competitive-equivalent districts cDf = mean(kernel) is clipped to
[0, 0.75] and linearly mapped to a rating. The 0.75 cap reflects that real
plans rarely exceed 75% truly competitive districts; above that the curve
saturates.

Coexists with the linear Competitiveness score: same sigma, different kernel
shape. Both rank "all P=0.5" as best and "all P=0/1" as worst, so they never
disagree at the extremes; they weigh the middle ground differently.
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr


_CLIP_BEST = 0.75  # cDf >= 0.75 saturates at 0 penalty


def holistic_competitiveness_from_shares(
    shares: np.ndarray,
    sigma_comb: float,
) -> tuple[float, float]:
    """
    Returns:
        cDf     -- soft competitive-equivalent fraction (display, in [0, 1])
        penalty -- [0, 100] (lower = more competitive)
    """
    n = len(shares)
    if n == 0:
        return 0.0, 0.0

    if sigma_comb <= 0.0:
        p_wins = (shares > 0.5).astype(np.float64)
    else:
        p_wins = ndtr((shares - 0.5) / sigma_comb)
    kernel = 1.0 - 4.0 * (p_wins - 0.5) ** 2
    cdf_val = float(kernel.mean())

    cdf_clipped = max(0.0, min(_CLIP_BEST, cdf_val))
    rating = cdf_clipped / _CLIP_BEST * 100.0
    return cdf_val, 100.0 - rating
