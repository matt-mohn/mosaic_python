"""
Neighborhood Severance -- a boundary-placement penalty that resists cutting
district boundaries through the heart of a minority neighborhood, whether or not
that neighborhood can form an opportunity district.
(Internal id stays `minority_cohesion` throughout; user-facing name is
"Neighborhood Severance".)

The complement to Electoral Opportunity, which asks "can this group ELECT?" and
so is silent on a community that can never clear the opportunity curve. This
score asks whether the plan needlessly FRACTURED a cohesive minority community,
everywhere, at any concentration.

Mechanism: a minority-weighted cut-edge penalty. Each adjacency edge (u, v)
carries, per group, share_g[u] * share_g[v] -- large only when both endpoints are
heavily that group, so the edge runs through the group's core. A cross-race seam
is free by construction; only within-race cores cost.

The statistic is an enrichment ratio against a compact race-blind map:

    severed_frac_g = severed adjacency / total adjacency, for group g
    pooled         = power mean of severed_frac over groups, weights mass_g^ALPHA
    expected       = C * (k-1)^a / N^b     race-blind rate at this k and N
    ratio          = pooled / expected     1 = race-neutral, >1 = fracturing
    penalty        = curve(ratio, R5, 1.0, R95)

Each group's statistic is a fraction of its OWN adjacency, so cutting a group's
only cluster registers heavily while nicking 1 of its 100 neighborhoods registers
~1% -- severity graded by the mechanism, not a threshold.

The denominator is frozen rather than the map's own cut fraction, which is
gameable: fractally cracking a low-minority area adds near-zero-minority cut
edges that inflate a live cut_frac while barely moving severed_frac, rewarding
the optimizer for shredding white areas. C*(k-1)^a/N^b depends only on the
district and precinct counts, so fragmenting is cohesion-neutral.

The ratio is a pure function of (assignment, adjacency, VAP) plus k, N and frozen
constants -- no run- or seed-relative anchor, so the same map always scores the
same. (k-1)^a/N^b tracks how a compact map's cut fraction grows with k, fitted to
within 5.3% worst case across 13 states at k = 4..59, and vanishes at k=1.

CALIBRATION
    Fitted offline across 13 states x k = 4..59 and frozen; scoring samples
    nothing. Floor and ceiling were measured with the real engine rather than a
    reduced dev chain, which as a weaker optimizer biased floors ~30% high and put
    R5 above where good plans land. On the corpus: cohesion-first ~5, random
    ReCom seed 33-59, adversarial 88-99, nothing pinned (0 of 44).

Per-edge weights are precomputed once; the per-iteration scorer is a
gather-and-sum over the cut set. Virtual bridge edges are zeroed via
real_edge_mask. VAP basis, not CVAP (see opportunity.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numba import njit

from mosaic.scoring.opportunity import GROUPS


@njit(cache=True)
def _severed_sums(cut_indices, w3):
    """One gather pass over the cut set: severed minority adjacency per group.

    w3 is the interleaved (n_edges, 3) weight matrix, so an edge's three group
    weights sit in one 24-byte run. Within noise of a flat per-group layout at
    this cut size -- the gather stays cache-resident either way."""
    s0 = 0.0
    s1 = 0.0
    s2 = 0.0
    for i in range(cut_indices.shape[0]):
        e = cut_indices[i]
        s0 += w3[e, 0]
        s1 += w3[e, 1]
        s2 += w3[e, 2]
    return s0, s1, s2

# ── Calibration constants ────────────────────────────────────────────────────
# Fitted offline across 13 states x k = 4..59 and frozen (dev/cohesion/).
#
# Race-blind expected severed fraction. The implied C in a C*sqrt(k/N) form is not
# constant -- it climbs with k and flattens (1.16 at k=4 to ~1.50 by k=38 in MS) --
# so both exponents float, cutting worst-case error from 31.9% to 5.3%. (k-1) so
# the expectation vanishes at k=1, where nothing is cut.
_DEN_C, _DEN_A, _DEN_B = 0.6972, 0.5150, 0.4091

# Achievable band on the enrichment ratio, measured with the real engine
# (cohesion-first for the floor, adversarial for the ceiling). A function of k
# alone: what constrains a plan is how many boundaries it must draw, not how
# finely the map is diced, and N cancels in the ratio.
_R5_A, _R5_Q = 1.1362, 0.2447      # R5  = 1 - A*(k-1)^-Q
_R95_A, _R95_Q = 13.9170, 0.6368   # R95 = 1 + A*(k-1)^-Q

# The middle anchor is the race-blind expectation itself, where a random ReCom
# seed lands by construction (measured 0.977-1.017 across all 44 cases).
_R_SEED = 1.0
_SEED_PEN = 55.0

# Clamps: R95's (k-1)^-Q diverges as k -> 1. k=1 returns early before the curve,
# but _band must not hand a direct caller a nonsense triple.
_R5_MIN = 0.02
_R95_MAX = 20.0
_MIN_GAP = 0.05    # keeps R5 < R_SEED < R95 strictly ordered

# Per-group pooling: w_g = mass_g^_ALPHA, combined by a power mean of _POWER.
# Raw mass (_ALPHA=1) made Mississippi's asian population 0.1% of the score, so
# destroying its only cluster moved less than nicking the 100th black
# neighborhood. 0.5 gives it 2.1% against a 1.4% VAP share; 0.25 overshoots.
_ALPHA = 0.5
_POWER = 2.0

# Cap on the curve's lower-bend exponent; see _rescale_ratio.
_Q_MAX = 12.0

def _band(n_districts: int) -> tuple[float, float, float]:
    """(R5, R_seed, R95) -- the three anchors at this district count.

    Closed forms in the integer k alone; see the constants above for why N is
    absent. Widest at low k, narrowing as k rises, mirroring how much freedom a
    plan actually has.
    """
    km1 = max(float(n_districts) - 1.0, 1e-9)
    r5 = max(1.0 - _R5_A * km1 ** (-_R5_Q), _R5_MIN)
    r95 = min(1.0 + _R95_A * km1 ** (-_R95_Q), _R95_MAX)
    rs = min(max(_R_SEED, r5 + _MIN_GAP), r95 - _MIN_GAP)
    if r95 <= rs:                     # degenerate only if the fit is broken
        r95 = rs + _MIN_GAP
    return r5, rs, r95


def _rescale_ratio(r: float, r5: float, rs: float, r95: float) -> float:
    """Enrichment ratio r (severed / race-blind expectation; lower = more
    cohesive) -> 0-100 penalty (lower = better). Four pieces, monotone
    throughout, and neither end ever reaches its limit:

        r <= R5      5 * (r/R5)^q                        power ease-in
        R5..R_seed   5  + 50*(r-R5)/(R_seed-R5)          good side
        R_seed..R95  55 + 40*(r-R_seed)/(R95-R_seed)     bad side
        r >= R95     100 - 5*s/(s + (r-R95))             rational tail

    Three anchors rather than two, because a single linear span cannot both let a
    cohesion-first run reach 5 and stop a random seed reading like a good plan. A
    seed sits only ~22% of the way up the achievable range (k=38: floor 0.523,
    seed 0.897, ceiling 2.217), so a linear interior forces either seeds to ~25 or
    the floor to 15-25. Pinning ratio 1.0 at 55 satisfies both (0/44 fail).

    Accepted cost: the slope breaks at R_seed, 3-10x steeper on the good side
    (k=38: 112 vs 28). Equal slopes would drag the seed penalty back to 15-29, so
    the kink is the mechanism.

    Both tails stay asymptotic so the annealer never loses signal: the upper tail
    is rational rather than exponential, which keeps a usable derivative several
    multiples past R95 where exp decays into float noise, and the lower bend
    reaches 0 only as r -> 0. _Q_MAX caps q, since a narrow band drives the exact
    slope-matching exponent high enough that (r/R5)^q underflows just below R5.
    """
    if r <= 0.0:
        return 0.0
    if r <= r5:
        q = min(18.0 * r5 / max(r95 - r5, 1e-9), _Q_MAX)
        return float(5.0 * (r / r5) ** q)
    if r <= rs:
        return float(5.0 + (_SEED_PEN - 5.0) * (r - r5) / max(rs - r5, 1e-12))
    if r <= r95:
        return float(_SEED_PEN
                     + (95.0 - _SEED_PEN) * (r - rs) / max(r95 - rs, 1e-12))
    s = max(r95 - rs, 1e-9) / 18.0
    return float(100.0 - 5.0 * s / (s + (r - r95)))


@dataclass
class MinorityCohesionData:
    """Precomputed per-edge minority adjacency, keyed by GROUPS.

    edge_w3:     (n_edges, 3) interleaved share_g[u]*share_g[v] in GROUPS order,
                 with virtual (bridge) edges zeroed. What the scorer gathers.
    edge_weight: {group: column view into edge_w3} -- the same numbers per group,
                 for display and analysis; no copy.
    total_adj:   {group: sum of edge_weight over real edges} -- the total minority
                 adjacency (denominator of severed_frac).
    n_precincts: precinct (node) count N, for the race-blind expectation and band.
    pool_weight: {group: normalised power-mean weight mass_g^_ALPHA}, precomputed
                 because it depends only on the frozen per-group adjacency mass.
    """
    edge_w3: np.ndarray = field(
        default_factory=lambda: np.empty((0, len(GROUPS)), np.float64))
    edge_weight: dict[str, np.ndarray] = field(default_factory=dict)
    total_adj: dict[str, float] = field(default_factory=dict)
    n_precincts: int = 0
    pool_weight: dict[str, float] = field(default_factory=dict)


def precompute_minority_cohesion_data(
    vap: Optional[dict],
    edge_u: Optional[np.ndarray],
    edge_v: Optional[np.ndarray],
    real_edge_mask: Optional[np.ndarray] = None,
) -> Optional[MinorityCohesionData]:
    """Build per-edge minority-adjacency weights from VAP + the adjacency graph.

    Returns None (score cleanly disabled) when VAP or the edge list is absent.
    """
    if vap is None or edge_u is None or edge_v is None or len(edge_u) == 0:
        return None
    eu = np.asarray(edge_u)
    ev = np.asarray(edge_v)
    total = np.asarray(vap["total"], dtype=np.float64)
    denom = np.where(total > 0.0, total, 1.0)
    real = (np.asarray(real_edge_mask, dtype=bool)
            if real_edge_mask is not None else np.ones(eu.shape[0], dtype=bool))
    w3 = np.zeros((eu.shape[0], len(GROUPS)), dtype=np.float64)
    total_adj: dict[str, float] = {}
    for gi, g in enumerate(GROUPS):
        share = np.asarray(vap[g], dtype=np.float64) / denom   # per-precinct group share
        w = share[eu] * share[ev]      # large only when both endpoints are core
        w[~real] = 0.0                 # bridge edges aren't real adjacency
        # Sum the contiguous per-group array, not the interleaved column: the
        # denominator must stay bit-for-bit what a flat per-group layout gives.
        total_adj[g] = float(w.sum())
        w3[:, gi] = w
    w3 = np.ascontiguousarray(w3)      # one cache line per edge in the gather
    edge_weight = {g: w3[:, gi] for gi, g in enumerate(GROUPS)}
    # Power-mean weights: frozen with the adjacency mass, so compute once here.
    raw = {g: total_adj[g] ** _ALPHA for g in GROUPS if total_adj[g] > 0.0}
    tw = sum(raw.values())
    pool_weight = {g: v / tw for g, v in raw.items()} if tw > 0 else {}
    return MinorityCohesionData(edge_w3=w3, edge_weight=edge_weight,
                                total_adj=total_adj,
                                n_precincts=int(total.shape[0]),
                                pool_weight=pool_weight)


def score_minority_cohesion(
    cut_edge_indices: np.ndarray,
    data: Optional[MinorityCohesionData],
    n_districts: int,
) -> tuple[float, dict[str, float]]:
    """Return (penalty, retention) for the current cut set.

    Each group's raw statistic is the FRACTION of its own minority adjacency the
    plan severs, so severity is graded automatically: cutting a group's only
    cluster moves its number a lot, cutting 1 of its 100 neighborhoods moves it
    ~1%. Groups then combine by a power mean weighted by mass^_ALPHA, and the
    result is measured against the race-blind expectation for this k and N.

        n_districts   -- k, for the expected severed fraction and the band.
        penalty       -- pooled 0-100 penalty; the optimizer term.
        retention[g]  -- that group's community-preservation %, 100 - its own
                         rescaled ratio (higher = more intact), or -1.0 if the
                         group has no adjacency (not applicable); display only.

    A pure function of (assignment, adjacency, VAP, k, N) -- no seed, no random
    partition, no state carried from a previous run -- so the same map always
    scores the same on any machine, and the optimizer cannot inflate the
    denominator by fragmenting low-minority areas.
    """
    retention: dict[str, float] = {g: -1.0 for g in GROUPS}
    if data is None or data.n_precincts <= 0 or not n_districts or n_districts < 1:
        return 0.0, retention
    # Expected race-blind severed fraction, and the achievable band at this k, N.
    expected = (_DEN_C * max(float(n_districts) - 1.0, 0.0) ** _DEN_A
                / float(data.n_precincts) ** _DEN_B)
    if expected <= 0.0:      # k == 1: nothing is cut, nothing to penalise
        return 0.0, retention
    r5, rs, r95 = _band(n_districts)
    # One Numba gather over the cut set yields all three groups' severed adjacency
    # (GROUPS order: black, latino, asian).
    sev = _severed_sums(cut_edge_indices, data.edge_w3)
    acc = 0.0
    wsum = 0.0
    for g, severed in zip(GROUPS, sev):
        tot = data.total_adj.get(g, 0.0)
        if tot <= 0.0:
            continue
        ratio = (severed / tot) / expected
        retention[g] = 100.0 - _rescale_ratio(ratio, r5, rs, r95)
        w = data.pool_weight.get(g, 0.0)
        acc += w * ratio ** _POWER
        wsum += w
    if wsum <= 0.0:
        return 0.0, retention
    pooled = (acc / wsum) ** (1.0 / _POWER)
    return _rescale_ratio(pooled, r5, rs, r95), retention
