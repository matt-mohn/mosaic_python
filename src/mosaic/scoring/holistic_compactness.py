"""
Holistic Compactness — a single 0-100 rating that blends mean Polsby-Popper and
mean Reock into one number, with both inputs linearly normalised over fixed
bands and averaged 50/50.

Steps (per plan):
  1. avg_PP, avg_Reock = unweighted mean across districts of the raw measures.
  2. Linearly normalize each into [0, 100] using fixed bands:
        Reock:  [0.25, 0.50] -> [0, 100]
        PP:     [0.10, 0.50] -> [0, 100]
  3. Combined rating = 0.5 * reock_rating + 0.5 * polsby_rating  (higher = more
     compact).

This is a *derived* score — both inputs are already produced by score_polsby_popper
and score_reock. We invert their penalty-form outputs back to raw means, apply the
bands, average, then convert to Mosaic's penalty form (lower = better) by
returning (100 - rating).

Saturation: the rating maxes out at avg_PP >= 0.50 and avg_Reock >= 0.50. Above
those, gradients flatten and the optimizer stops favouring further gains. This
is intentional — most "compact enough" plans live above the saturation point.
"""

from __future__ import annotations


_REOCK_MIN  = 0.25
_REOCK_MAX  = 0.50
_POLSBY_MIN = 0.10
_POLSBY_MAX = 0.50


def holistic_compactness_from_scores(pp_score: float, reock_score: float) -> float:
    """Return Holistic Compactness in Mosaic PENALTY form (lower = better).

    Args:
        pp_score:    score_polsby_popper output, i.e. (1 - avg_PP) * 100
        reock_score: score_reock output,         i.e. (1 - avg_Reock) * 100

    Returns:
        float in [0, 100]. 0 = both PP and Reock saturate the "best" band
        (avg_PP >= 0.50, avg_Reock >= 0.50). 100 = both at or below the
        "worst" band floor.

    Pure-scalar: no geometry work. The caller must already have computed
    pp_score and reock_score (score_plan does this when
    weight_holistic_compactness is active).
    """
    avg_pp    = 1.0 - pp_score    / 100.0
    avg_reock = 1.0 - reock_score / 100.0

    if avg_reock < _REOCK_MIN:
        r_clip = _REOCK_MIN
    elif avg_reock > _REOCK_MAX:
        r_clip = _REOCK_MAX
    else:
        r_clip = avg_reock

    if avg_pp < _POLSBY_MIN:
        p_clip = _POLSBY_MIN
    elif avg_pp > _POLSBY_MAX:
        p_clip = _POLSBY_MAX
    else:
        p_clip = avg_pp

    r_rating = (r_clip - _REOCK_MIN)  / (_REOCK_MAX - _REOCK_MIN)  * 100.0
    p_rating = (p_clip - _POLSBY_MIN) / (_POLSBY_MAX - _POLSBY_MIN) * 100.0

    rating = 0.5 * r_rating + 0.5 * p_rating
    return 100.0 - rating
