"""
Holistic Competitiveness — single 0-100 rating using a quadratic competitiveness kernel.

Per-district soft competitiveness = 1 - 4 * (P_win - 0.5)**2:
  P_win = 0.5      -> kernel = 1.0   (perfect toss-up)
  P_win in {0, 1}  -> kernel = 0     (locked seat)

The competitive-equivalent fraction cDf = mean(kernel) maps to a rating two ways:

  - unclipped (DEFAULT): a two-segment line with a knee at cDf = _KNEE. The
    achievable range [0, _KNEE] spends _KNEE_LOW of the scale (penalty 100 -> 5);
    the remaining [_KNEE, 1] spends the last 5% (penalty 5 -> 0). No realistic map
    bottoms out at 0, and the top few points are reserved for the (near-impossible)
    all-toss-up range, so the optimizer keeps a gradient throughout.

  - clipped (scorecard form): cDf clipped to [0, _CLIP_BEST] and linearly mapped;
    cDf >= _CLIP_BEST saturates at 0 penalty.

Both rank "all P=0.5" as best and "all P=0/1" as worst.
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr


_CLIP_BEST = 0.75   # clipped form: cDf >= this saturates at 0 penalty
_KNEE      = 0.50   # unclipped form: knee cDf (achievable range below, reserve above)
_KNEE_LOW  = 0.95   # unclipped form: fraction of the rating spent reaching the knee


def holistic_competitiveness_from_shares(
    shares: np.ndarray,
    sigma_comb: float,
    unclipped: bool = True,
) -> tuple[float, float]:
    """
    Args:
        unclipped: if True (default), two-segment knee mapping (keeps a gradient
                   above the knee); if False, the clipped scorecard form.

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

    c = max(0.0, min(1.0, cdf_val))
    if unclipped:
        if c <= _KNEE:
            rating = c / _KNEE * (_KNEE_LOW * 100.0)
        else:
            rating = (_KNEE_LOW * 100.0
                      + (c - _KNEE) / (1.0 - _KNEE) * ((1.0 - _KNEE_LOW) * 100.0))
    else:
        rating = min(_CLIP_BEST, c) / _CLIP_BEST * 100.0
    return cdf_val, 100.0 - rating
