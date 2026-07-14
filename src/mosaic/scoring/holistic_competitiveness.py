"""
Holistic Competitiveness — single 0-100 rating using a quadratic competitiveness kernel.

Per-district soft competitiveness = 1 - 4 * (P_win - 0.5)**2:
  P_win = 0.5      -> kernel = 1.0   (perfect toss-up)
  P_win in {0, 1}  -> kernel = 0     (locked seat)

The competitive-equivalent fraction cDf = mean(kernel) maps to a rating two ways:

  - unclipped (DEFAULT): ride the clipped slope down to penalty _DIVERGE_PEN, then a
    power ease-out that curves to a flat (slope-0) landing at cDf = 1. _UNCLIP_EXP is
    chosen so the tail leaves the divergence point at the clipped slope (no kink), so
    unclipped tracks the scorecard down to penalty _DIVERGE_PEN and then keeps a
    gradient across the top end where clipped saturates at 0.

  - clipped (scorecard form): cDf clipped to [0, _CLIP_BEST] and linearly mapped;
    cDf >= _CLIP_BEST saturates at 0 penalty.

Both rank "all P=0.5" as best and "all P=0/1" as worst.
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr


_CLIP_BEST   = 0.75   # clipped form: cDf >= this saturates at 0 penalty
# Unclipped: ride the clipped slope to penalty _DIVERGE_PEN, then a power ease-out
# curving to a flat (slope-0) landing at cDf = 1. _UNCLIP_EXP is derived so the tail
# leaves the divergence point at the clipped slope -- no kink at the knee.
_DIVERGE_PEN = 25.0   # penalty where unclipped peels off clipped
_UNCLIP_KNEE = _CLIP_BEST * (1.0 - _DIVERGE_PEN / 100.0)   # cDf at the divergence (0.5625)
_UNCLIP_EXP  = (100.0 / _CLIP_BEST) * (1.0 - _UNCLIP_KNEE) / _DIVERGE_PEN   # tail exp (~2.33)


def holistic_competitiveness_from_shares(
    shares: np.ndarray,
    sigma_comb: float,
    unclipped: bool = True,
) -> tuple[float, float]:
    """
    Args:
        unclipped: if True (default), ride the clipped slope to penalty _DIVERGE_PEN
                   then a flat-landing ease-out (keeps a gradient where clipped
                   saturates); if False, the clipped scorecard form.

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
        if c <= _UNCLIP_KNEE:
            rating = c / _CLIP_BEST * 100.0            # rides the clipped slope
        else:
            u = (1.0 - c) / (1.0 - _UNCLIP_KNEE)       # 1 at the knee -> 0 at cDf = 1
            rating = 100.0 - _DIVERGE_PEN * u ** _UNCLIP_EXP   # ease-out: flat at cDf = 1
    else:
        rating = min(_CLIP_BEST, c) / _CLIP_BEST * 100.0
    return cdf_val, 100.0 - rating
