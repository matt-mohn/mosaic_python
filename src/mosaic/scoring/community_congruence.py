"""Community Dispersion partition penalty.

The scorer builds connected demographic cores at nested 50%, 35%, and 20%
group-share thresholds within the selected demographic universe. For each core
``c`` it computes:

    p_i     = core population share in district i
    N_eff   = 1 / sum(p_i ** 2)
    m_c     = ceil(core_population / ideal_district_population)
    ratio_c = N_eff / m_c

Core ratios are combined with a population- and layer-weighted power mean. The
pooled ratio is mapped monotonically to a penalty using district-count-dependent
anchors. A precinct can belong to cores for more than one group and layer.
Virtual bridge edges are excluded when connected cores are built.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numba import njit

from mosaic.scoring.opportunity import GROUPS

log = logging.getLogger(__name__)

# ── Core construction ────────────────────────────────────────────────────────
# Concentration layers, densest first. Nested by construction: a precinct in the
# 50% layer is also in the 35% and 20% layers.
LAYERS: tuple[float, ...] = (0.50, 0.35, 0.20)

# A core contributes at every concentration layer it belongs to, with denser
# layers receiving larger multipliers.
_LAMBDA: dict[float, float] = {0.50: 1.0, 0.35: 0.5, 0.20: 0.25}

# Absolute, because cores are built without reference to k (m_c is computed at
# score time, so a change in district count does not invalidate them).
_MIN_CORE_POP = 20_000.0

# ── Pooling ──────────────────────────────────────────────────────────────────
# Core weights use the square root of population; the weighted power mean uses
# exponent 2.
_ALPHA = 0.5
_POWER = 2.0

# ── District-count-dependent ratio band ─────────────────────────────────────
_R5_A, _R5_Q = 0.000074, 1.784031        # R5    = 1 + A*(k-1)^Q
_RSEED_A, _RSEED_Q = 0.259077, 0.267266  # Rseed = 1 + A*(k-1)^Q
_R95_C, _R95_A, _R95_Q = 3.923603, 3.232672, 0.688527   # R95 = C - A*(k-1)^-Q

_SEED_PEN = 55.0
_MIN_GAP = 0.05
_Q_MAX = 12.0

# Clamps: R95's (k-1)^-Q diverges as k -> 1 (unclamped, -5.1e6 at k=1), which
# would invert the band. k=1 returns early before the curve, but _band must not
# hand a direct caller a nonsense triple.
_R5_MIN, _R5_MAX = 0.02, 3.0
_R95_MAX = 20.0


def _band(n_districts: int) -> tuple[float, float, float]:
    """(R5, R_seed, R95) at this district count.

    Closed forms in k alone; see the constants above. Clamped and gap-enforced so
    R5 < R_seed < R95 holds at every k, including k = 1..2 where the raw ceiling
    form diverges.
    """
    km1 = max(float(n_districts) - 1.0, 1e-9)
    r5 = min(max(1.0 + _R5_A * km1 ** _R5_Q, _R5_MIN), _R5_MAX)
    rs = max(1.0 + _RSEED_A * km1 ** _RSEED_Q, r5 + _MIN_GAP)
    r95 = min(max(_R95_C - _R95_A * km1 ** (-_R95_Q), rs + _MIN_GAP), _R95_MAX)
    return r5, rs, r95


def _rescale_ratio(r: float, r5: float, rs: float, r95: float) -> float:
    """Ratio -> 0-100 penalty. Same four-piece shape as Neighborhood Severance's:
    power ease-in below R5, linear either side of the seed anchor, rational tail
    above R95, asymptotic at both ends so the annealer never loses signal.
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
class CommunityCongruenceData:
    """Layered per-group cores, flattened for the JITed scorer.

    core_prec/core_off: concatenated precinct indices and their per-core offsets,
                        so core c owns core_prec[core_off[c]:core_off[c + 1]].
    core_prec_pop:      each core_prec entry's population, in the same order, so
                        the scorer streams populations instead of gathering them
                        from prec_pop at random precinct indices.
    core_pop:           total population of each core.
    core_weight:        pop^_ALPHA * lambda(layer), the pooling weight.
    core_group:         index into GROUPS, for per-group display values.
    prec_pop:           per-precinct population (global, not per core).
    n_cores_by_layer:   diagnostic only; logged at load.
    """
    core_prec: np.ndarray = field(default_factory=lambda: np.empty(0, np.int32))
    core_prec_pop: np.ndarray = field(
        default_factory=lambda: np.empty(0, np.float64))
    core_off: np.ndarray = field(default_factory=lambda: np.zeros(1, np.int64))
    core_pop: np.ndarray = field(default_factory=lambda: np.empty(0, np.float64))
    core_weight: np.ndarray = field(default_factory=lambda: np.empty(0, np.float64))
    core_group: np.ndarray = field(default_factory=lambda: np.empty(0, np.int8))
    prec_pop: np.ndarray = field(default_factory=lambda: np.empty(0, np.float64))
    n_cores_by_layer: dict = field(default_factory=dict)

    @property
    def n_cores(self) -> int:
        return int(self.core_pop.shape[0])


def _components(mask: np.ndarray, nbr: list[np.ndarray]) -> list[np.ndarray]:
    """Connected components of `mask`. Deterministic: seeds are visited in
    ascending precinct index and each component's members are returned sorted,
    so the core set is a pure function of the geometry and demographics."""
    seen = np.zeros(mask.shape[0], dtype=bool)
    out: list[np.ndarray] = []
    for s in np.where(mask)[0]:
        if seen[s]:
            continue
        stack = [int(s)]
        comp: list[int] = []
        seen[s] = True
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in nbr[u]:
                if mask[v] and not seen[v]:
                    seen[v] = True
                    stack.append(int(v))
        comp.sort()
        out.append(np.array(comp, dtype=np.int64))
    return out


def _neighbours(n: int, edge_u, edge_v, real_edge_mask) -> list[np.ndarray]:
    eu = np.asarray(edge_u)
    ev = np.asarray(edge_v)
    keep = (np.asarray(real_edge_mask, dtype=bool)
            if real_edge_mask is not None else np.ones(eu.shape[0], dtype=bool))
    buckets: list[list[int]] = [[] for _ in range(n)]
    for e in np.where(keep)[0]:      # bridge edges are not real adjacency
        buckets[int(eu[e])].append(int(ev[e]))
        buckets[int(ev[e])].append(int(eu[e]))
    return [np.array(sorted(b), dtype=np.int64) for b in buckets]


def precompute_community_congruence_data(
    vap: Optional[dict],
    populations: Optional[np.ndarray],
    edge_u: Optional[np.ndarray],
    edge_v: Optional[np.ndarray],
    real_edge_mask: Optional[np.ndarray] = None,
    layers: tuple[float, ...] = LAYERS,
    min_core_pop: float = _MIN_CORE_POP,
) -> Optional[CommunityCongruenceData]:
    """Build layered per-group cores. Returns None (score cleanly disabled) when
    demographics, population or the edge list are absent.

    Depends only on geometry and demographics -- never on the assignment or the
    district count -- so it runs once per load and stays valid if k changes.
    """
    if (vap is None or populations is None or edge_u is None or edge_v is None
            or len(edge_u) == 0):
        return None
    try:
        pops = np.asarray(populations, dtype=np.float64)
        total = np.asarray(vap["total"], dtype=np.float64)
        if pops.shape[0] != total.shape[0]:
            log.warning("Community Dispersion: population/demographic length mismatch "
                        f"({pops.shape[0]} vs {total.shape[0]}); disabled")
            return None
        n = int(total.shape[0])
        denom = np.where(total > 0.0, total, 1.0)
        nbr = _neighbours(n, edge_u, edge_v, real_edge_mask)

        prec: list[np.ndarray] = []
        pop_l: list[float] = []
        wt_l: list[float] = []
        grp_l: list[int] = []
        by_layer: dict[float, int] = {}
        for gi, g in enumerate(GROUPS):
            share = np.asarray(vap[g], dtype=np.float64) / denom
            for T in sorted(layers, reverse=True):
                mask = (share >= T) & (total > 0.0)
                for comp in _components(mask, nbr):
                    cp = float(pops[comp].sum())
                    if cp < min_core_pop:
                        continue
                    prec.append(comp)
                    pop_l.append(cp)
                    wt_l.append(cp ** _ALPHA * _LAMBDA.get(T, 1.0))
                    grp_l.append(gi)
                    by_layer[T] = by_layer.get(T, 0) + 1
        if not prec:
            log.info("Community Dispersion: no cores above "
                     f"{min_core_pop:,.0f} pop; disabled")
            return None
        off = np.zeros(len(prec) + 1, dtype=np.int64)
        np.cumsum([len(c) for c in prec], out=off[1:])
        core_prec = np.concatenate(prec).astype(np.int64)
        return CommunityCongruenceData(
            core_prec=core_prec,
            core_prec_pop=np.ascontiguousarray(pops[core_prec]),
            core_off=off,
            core_pop=np.asarray(pop_l, dtype=np.float64),
            core_weight=np.asarray(wt_l, dtype=np.float64),
            core_group=np.asarray(grp_l, dtype=np.int8),
            prec_pop=np.ascontiguousarray(pops),
            n_cores_by_layer=by_layer,
        )
    except Exception:                      # pragma: no cover - defensive
        log.warning("Community Dispersion precompute failed; disabled",
                    exc_info=True)
        return None


@njit(cache=True)
def _pool(assignment, core_prec, core_prec_pop, core_off, core_pop, core_weight,
          core_group, n_districts, ideal_pop, n_groups):
    """One pass over every core: accumulate the weighted power mean of
    N_eff / m, overall and per group. Returns (acc, wsum, gacc, gwsum).

    N_eff is tot^2 / sum(pop_d^2) rather than 1 / sum(p_d^2), so no per-district
    division is needed.

    The two full 0..k-1 loops allow contiguous scratch clearing and scanning.
    """
    n_cores = core_pop.shape[0]
    scratch = np.zeros(n_districts, dtype=np.float64)
    acc = 0.0
    wsum = 0.0
    gacc = np.zeros(n_groups, dtype=np.float64)
    gwsum = np.zeros(n_groups, dtype=np.float64)
    for c in range(n_cores):
        lo = core_off[c]
        hi = core_off[c + 1]
        for d in range(n_districts):
            scratch[d] = 0.0
        for i in range(lo, hi):
            scratch[assignment[core_prec[i]]] += core_prec_pop[i]
        sq = 0.0
        for d in range(n_districts):
            sq += scratch[d] * scratch[d]
        if sq <= 0.0:
            continue
        tot = core_pop[c]
        neff = tot * tot / sq
        m = math.ceil(tot / ideal_pop - 1e-9)
        if m < 1:
            m = 1
        ratio = neff / m
        w = core_weight[c]
        contrib = w * ratio ** _POWER
        acc += contrib
        wsum += w
        g = core_group[c]
        gacc[g] += contrib
        gwsum[g] += w
    return acc, wsum, gacc, gwsum


def score_community_congruence(
    assignment: np.ndarray,
    data: Optional[CommunityCongruenceData],
    n_districts: int,
    ideal_pop: Optional[float] = None,
) -> tuple[float, dict[str, float]]:
    """Return (penalty, congruence) for this assignment.

        penalty         -- [0, 100), lower = better; the optimizer term.
        congruence[g]   -- that group's 0-100 congruence, 100 - its own rescaled
                           pooled ratio (higher = more intact), or -1.0 when the
                           group has no cores (not applicable). Display only.

    ideal_pop defaults to (total population) / n_districts. The calculation is
    deterministic for fixed assignment, core, population, and district-count
    inputs; it has no run-relative denominator.
    """
    congruence: dict[str, float] = {g: -1.0 for g in GROUPS}
    if (data is None or data.n_cores == 0 or not n_districts
            or n_districts < 1):
        return 0.0, congruence
    a = np.ascontiguousarray(np.asarray(assignment, dtype=np.int32))
    if ideal_pop is None:
        ideal_pop = float(data.prec_pop.sum()) / float(n_districts)
    if not ideal_pop or ideal_pop <= 0.0:
        return 0.0, congruence
    r5, rs, r95 = _band(n_districts)
    acc, wsum, gacc, gwsum = _pool(
        a, data.core_prec, data.core_prec_pop, data.core_off, data.core_pop,
        data.core_weight, data.core_group, int(n_districts), float(ideal_pop),
        len(GROUPS),
    )
    if wsum <= 0.0:
        return 0.0, congruence
    for gi, g in enumerate(GROUPS):
        if gwsum[gi] > 0.0:
            gp = (gacc[gi] / gwsum[gi]) ** (1.0 / _POWER)
            congruence[g] = 100.0 - _rescale_ratio(gp, r5, rs, r95)
    pooled = (acc / wsum) ** (1.0 / _POWER)
    return _rescale_ratio(pooled, r5, rs, r95), congruence
