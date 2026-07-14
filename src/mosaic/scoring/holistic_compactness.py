"""
Holistic Compactness — a single 0-100 rating that blends mean Polsby-Popper and
mean Reock into one number, with both inputs mapped over fixed bands and averaged
50/50. Returned in Mosaic PENALTY form (lower = better).

avg_PP, avg_Reock are the unweighted means across districts. Each maps to a
penalty two ways:

  - clipped (scorecard form): linear over a fixed band, clamped at both ends.
    Reock band [0.25, 0.50], PP band [0.10, 0.50] -> penalty 100..0. Saturates
    at penalty 0 above 0.50 and pins at 100 below the band floor.

  - unclipped (DEFAULT): ride the clipped slope down to penalty _DIVERGE_PEN,
    then a power ease-out that curves to a flat (slope-0) landing at raw _LANDING
    (a realistic excellence ceiling, not a perfect circle). Each component's
    exponent is chosen so the tail leaves the divergence point at the clipped
    slope (no kink), so it tracks the scorecard down to penalty _DIVERGE_PEN and
    then keeps a gradient across [0.50, _LANDING] where clipped saturates at 0.
    The bottom clamp (pinned at 100 below the band floor) is kept.

This is a *derived* score — both inputs are already produced by
score_polsby_popper and score_reock; we invert their penalty-form outputs back
to raw means before mapping.
"""

from __future__ import annotations


_REOCK_MIN  = 0.25
_REOCK_MAX  = 0.50
_POLSBY_MIN = 0.10
_POLSBY_MAX = 0.50

# Unclipped ski-ramp: ride the clipped band down to penalty _DIVERGE_PEN, then a
# power ease-out to a flat landing at raw _LANDING. The exponent per component is
# derived so the tail leaves the divergence point at the clipped slope -- no kink.
_DIVERGE_PEN = 15.0   # penalty where unclipped peels off the scorecard (rating 85)
_LANDING     = 0.65   # raw value where the ease-out lands flat (new saturation)


def _band_penalty(raw: float, lo: float, hi: float, unclipped: bool) -> float:
    """One component's penalty in [0, 100] (lower = more compact)."""
    if unclipped:
        if raw <= lo:                       # bottom clamp (kept)
            return 100.0
        knee = lo + (1.0 - _DIVERGE_PEN / 100.0) * (hi - lo)   # rating-85 point
        if raw <= knee:
            return 100.0 - (raw - lo) / (hi - lo) * 100.0      # rides the scorecard
        if raw >= _LANDING:
            return 0.0
        slope = 100.0 / (hi - lo)
        p = slope * (_LANDING - knee) / _DIVERGE_PEN   # tangent -> no kink
        u = (_LANDING - raw) / (_LANDING - knee)       # 1 at knee -> 0 at landing
        return _DIVERGE_PEN * u ** p
    # clipped scorecard: linear over [lo, hi], clamped at both ends
    clipped = lo if raw < lo else hi if raw > hi else raw
    return 100.0 - (clipped - lo) / (hi - lo) * 100.0


def holistic_compactness_from_scores(
    pp_score: float, reock_score: float, unclipped: bool = True,
) -> float:
    """Return Holistic Compactness in Mosaic PENALTY form (lower = better).

    Args:
        pp_score:    score_polsby_popper output, i.e. (1 - avg_PP) * 100
        reock_score: score_reock output,         i.e. (1 - avg_Reock) * 100
        unclipped:   if True (default), ride the scorecard to penalty _DIVERGE_PEN
                     then a flat-landing ease-out (keeps a gradient where the
                     scorecard saturates); if False, the clipped scorecard form.

    Returns:
        float in [0, 100]. 0 = both components at/above their best band; 100 =
        both at or below the band floor.

    Pure-scalar: no geometry work. The caller must already have computed
    pp_score and reock_score (score_plan does this when
    weight_holistic_compactness is active).
    """
    avg_pp    = 1.0 - pp_score    / 100.0
    avg_reock = 1.0 - reock_score / 100.0
    r_pen = _band_penalty(avg_reock, _REOCK_MIN,  _REOCK_MAX,  unclipped)
    p_pen = _band_penalty(avg_pp,    _POLSBY_MIN, _POLSBY_MAX, unclipped)
    return 0.5 * r_pen + 0.5 * p_pen
