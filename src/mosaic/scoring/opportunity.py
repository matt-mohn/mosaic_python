"""
Opportunity-to-elect engine -- shared per-district minority opportunity model.

Per district and group, the probability the group can elect its candidate of
choice, on a logistic curve over the group's share of the selected demographic
universe:

    P_g,d = sigmoid((share_g,d - midpoint) / steepness)

Also per group: `T_g` = (statewide group share) * n, the proportional benchmark;
`ref_g` = the curve at a "solid" opportunity share, the credit denominator so a
solid district counts as ~one; `feasible_g`, whether the greedy bundle heuristic
finds a qualifying bundle; and `ceiling_g`, the heuristic credit reference for
the top round(T) bundles. The bundle calculations do not enforce contiguity and
do not prove that a valid plan can attain their results.
representation.py consumes all five.

The geographic pair comes from `_smart_targets` (local pools) or from the
statewide sweeps `_anchor_feasible` / `_ceiling_credit_sum` when smart targets
are disabled or coordinates are unavailable. Assignment-independent inputs are
cached; per-district opportunity arrays are recomputed for each assignment.

Groups scored: black, latino, and asian, each on its own curve. Curve parameters
may be scalars or per-group dictionaries. Inputs are a demographic total and
group counts from one consistent universe.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numba import njit

# Group order is stable; consumers iterate this tuple.
GROUPS: tuple[str, ...] = ("black", "latino", "asian")


@njit(cache=True)
def _accum_and_curve(assignment, weights, n_districts, m, tau):
    """One precinct-order pass over the (n, 1+G) weight matrix (column 0 =
    demographic total, columns 1.. = compatible group counts), then the logistic
    curve per district.

    Accumulation and curve evaluation occur in one compiled function. Sums use
    float64 even when the stored weights are integers.

    Returns (per-district column sums, (G, n_districts) opportunity probability).
    """
    ncol = weights.shape[1]
    sums = np.zeros((n_districts, ncol))
    for i in range(assignment.shape[0]):
        d = assignment[i]
        for c in range(ncol):
            sums[d, c] += weights[i, c]
    ngrp = ncol - 1
    P = np.empty((ngrp, n_districts))
    for d in range(n_districts):
        den = sums[d, 0]
        if den < 1.0:            # districts with a zero demographic total -> share 0
            den = 1.0
        for j in range(ngrp):
            share = sums[d, j + 1] / den
            P[j, d] = 1.0 / (1.0 + np.exp(-(share - m[j]) / tau[j]))
    return sums, P

# Public per-group discount constants. The current representation scorer does
# not consult this mapping.
DISCOUNT: dict[str, float] = {"black": 1.0, "latino": 1.0, "asian": 1.0}


@dataclass
class OpportunityResult:
    """Per-group opportunity summary, all keyed by GROUPS.

    T:        proportional benchmark in districts (statewide group share * n).
    P:        per-district opportunity probability array.
    ref:      curve value at the 'solid' opportunity share; the per-district
              credit denominator (P / ref capped at 1 == effective opp districts).
    feasible: whether the greedy bundle heuristic finds >=1 qualifying bundle
              (N/A gate).
    ceiling:  heuristic reference credit for the top-round(T) districts; the
              rating denominator.
    """
    T: dict[str, float]
    P: dict[str, np.ndarray]
    ref: dict[str, float]
    feasible: dict[str, int] = field(default_factory=dict)
    ceiling: dict[str, float] = field(default_factory=dict)


def _group_vap(vap: dict, group: str) -> np.ndarray:
    if group == "coalition":
        return (np.asarray(vap["total"], dtype=np.float64)
                - np.asarray(vap["white"], dtype=np.float64))
    return np.asarray(vap[group], dtype=np.float64)


def _param(value, group: str) -> float:
    """Resolve a scalar-or-per-group parameter to a scalar for this group."""
    if isinstance(value, dict):
        return float(value[group])
    return float(value)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


# Assignment-independent, so memoised by demographic-array identity, district count,
# curve parameters, and targeting settings.
_GEO_CACHE: dict[tuple, tuple[dict[str, int], dict[str, float]]] = {}


# ── Smart targets: local-pool geography ──────────────────────────────────────
# Statewide sweeps rank every precinct, so their bundles can span distant places.
# Smart targets confine each hypothetical district to the nearest precincts to a
# centre until they hold `slack` x an ideal demographic total, then run the same greedy.
# Sizing the pool in POPULATION makes one constant work at any state and k, and
# distance only RANKS neighbours, so the CRS unit cancels.
#
# slack 1 = a compact ball; 2 = may take the best half of a two-district
# neighbourhood; inf = the statewide behaviour.
_SMART_SLACK = 2.0

# Cost is centers x target rounds. A fixed center-evaluation budget bounds this
# precomputation across district counts.
_SMART_CENTRE_BUDGET = 1200
_SMART_CENTRE_MIN, _SMART_CENTRE_MAX = 100, 600

# Minimum credit the best local bundle must earn for the heuristic feasibility
# flag to be set.
_SMART_MIN_CREDIT = 0.10


def precompute_opportunity_coords(gdf) -> "np.ndarray | None":
    """(n, 2) per-precinct coordinates for smart targets, or None on failure.

    Representative points rather than centroids: they are guaranteed to lie inside
    the precinct, and only the distance ORDERING between them is ever used.
    """
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*geographic CRS.*")
            pts = gdf.geometry.representative_point()
        return np.ascontiguousarray(
            np.c_[pts.x.to_numpy(dtype=np.float64),
                  pts.y.to_numpy(dtype=np.float64)])
    except Exception:                       # pragma: no cover - defensive
        return None


def _ranking_xy(coords: np.ndarray) -> np.ndarray:
    """(n, 2) coordinates whose ORDERING by distance is meaningful. Longitude is
    scaled by cos(mean latitude) when the input looks like degrees, so a geographic
    CRS ranks neighbours sensibly without needing a projection."""
    xy = np.array(coords, dtype=np.float64, copy=True)
    if xy.shape[0] and np.abs(xy[:, 0]).max() <= 180.0 \
            and np.abs(xy[:, 1]).max() <= 90.0:
        xy[:, 0] *= np.cos(np.deg2rad(float(np.mean(xy[:, 1]))))
    return xy


def _bundle(pool, grp, total, share, ideal, stop_group):
    """Highest-share-first inside `pool`, stopping when the bundle holds
    `stop_group` of group population or reaches an ideal demographic total.
    Returns (group count, demographic total, members)."""
    p2 = pool[np.argsort(share[pool], kind="stable")[::-1]]
    cg = ct = 0.0
    for j in range(p2.shape[0]):
        i = p2[j]
        cg += grp[i]
        ct += total[i]
        if (stop_group is not None and cg >= stop_group) or ct >= ideal:
            return cg, ct, p2[:j + 1]
    return cg, ct, p2


def _local_rounds(grp, total, ideal, xy, slack, n_rounds, stop_group, score_fn,
                  tree=None):
    """Greedily take up to `n_rounds` disjoint local bundles, scoring each with
    `score_fn(cg, ct)`. Centres are selected again from the remaining precincts
    after each bundle is removed."""
    n = total.shape[0]
    if n == 0 or ideal <= 0.0 or n_rounds < 1:
        return []
    share = np.divide(grp, total, out=np.zeros_like(grp, dtype=np.float64),
                      where=total > 0.0)
    if tree is None:
        from scipy.spatial import cKDTree
        tree = cKDTree(xy)
    mean_vap = max(float(total.mean()), 1.0)
    # Neighbors per center that could hold slack x the ideal demographic total,
    # with headroom.
    kq = int(min(n, max(32, (slack * ideal / mean_vap) * 2.0)))
    n_centres = int(min(n, _SMART_CENTRE_MAX,
                        max(_SMART_CENTRE_MIN,
                            _SMART_CENTRE_BUDGET // max(n_rounds, 1))))

    avail = np.ones(n, dtype=bool)
    out: list[float] = []
    for _ in range(n_rounds):
        live = np.flatnonzero(avail)
        if live.size == 0 or float(total[live].sum()) < ideal:
            break
        centres = live[np.argsort(share[live], kind="stable")[::-1][:n_centres]]
        best_score = 0.0
        best_mem = None
        _, rows = tree.query(xy[centres], k=kq)
        rows = np.atleast_2d(np.asarray(rows))
        for r in range(centres.shape[0]):
            idx = rows[r]
            idx = idx[idx < n]
            idx = idx[avail[idx]]                # distance order is preserved
            if idx.size == 0:
                continue
            cum = np.cumsum(total[idx])
            pool = idx[:int(np.searchsorted(cum, slack * ideal) + 1)]
            if float(total[pool].sum()) < ideal * 0.999:
                continue
            cg, ct, mem = _bundle(pool, grp, total, share, ideal, stop_group)
            sc = score_fn(cg, ct)
            if sc > best_score:
                best_score = sc
                best_mem = mem
        if best_mem is None:
            break
        avail[best_mem] = False
        out.append(best_score)
    return out


def _smart_targets(grp, total, ideal, m, tau, s, ref, target, xy, slack,
                   tree=None) -> tuple[int, float]:
    """Smart-target feasibility flag and reference ceiling from one local sweep.

    Both come from the same per-district credits: the gate is the best single
    district's credit vs _SMART_MIN_CREDIT, the ceiling the credit sum over the
    best `target` disjoint districts. A bundle stopping early at solid is still
    charged the population it must absorb to become a full district.
    """
    def score(cg, ct):
        denom = ct if ct > ideal else ideal
        if denom <= 0.0:
            return 0.0
        p = 1.0 / (1.0 + np.exp(-(cg / denom - m) / tau))
        return min(p / ref, 1.0)

    credits = _local_rounds(grp, total, ideal, xy, slack, max(target, 1),
                            s * ideal, score, tree=tree)
    feasible = int(bool(credits) and credits[0] >= _SMART_MIN_CREDIT)
    ceiling = float(sum(credits[:target])) if target >= 1 else 0.0
    return feasible, ceiling


def _anchor_feasible(grp: np.ndarray, total: np.ndarray, ideal: float,
                     threshold: float) -> int:
    """Heuristic count of ideal-sized bundles that reach `threshold`
    group share. Bundle the highest-share precincts until a bundle holds
    threshold*ideal group population -- enough to be one threshold-level district once
    padded to ideal size -- then start the next; a bundle that fills a whole
    district's population without getting there is spent. This calculation
    ignores contiguity and does not construct valid districts."""
    if ideal <= 0.0:
        return 0
    share = np.divide(grp, total, out=np.zeros_like(grp, dtype=np.float64),
                      where=total > 0.0)
    order = np.argsort(share)[::-1]
    ts = total[order]
    gs = grp[order]
    need = threshold * ideal
    cg = ct = 0.0
    count = 0
    for i in range(order.shape[0]):
        cg += gs[i]
        ct += ts[i]
        if cg >= need:                # enough group population -> one opportunity district
            count += 1
            cg = ct = 0.0
        elif ct >= ideal:             # full district's population, still short -> spent
            cg = ct = 0.0
    return count


def _ceiling_credit_sum(grp: np.ndarray, total: np.ndarray, ideal: float,
                        m: float, tau: float, s: float, ref: float,
                        target: int) -> float:
    """Heuristic clipped-credit reference for the top ``target`` bundles.

    Precincts are greedily packed in descending group-share order. A bundle that
    reaches the solid-share group requirement earns full credit; a bundle that
    reaches the ideal demographic total first earns partial credit. The calculation ignores
    contiguity and is not an optimization proof or guaranteed upper bound."""
    if ideal <= 0.0 or target < 1:
        return 0.0
    share = np.divide(grp, total, out=np.zeros_like(grp, dtype=np.float64),
                      where=total > 0.0)
    order = np.argsort(share)[::-1]
    ts = total[order]
    gs = grp[order]
    need = s * ideal                      # group population for one full-credit district
    cg = ct = 0.0
    credits: list[float] = []
    for i in range(order.shape[0]):
        cg += gs[i]
        ct += ts[i]
        if cg >= need:
            credits.append(1.0)
            cg = ct = 0.0
        elif ct >= ideal:                 # full district, short of solid -> partial credit
            p = 1.0 / (1.0 + np.exp(-(cg / ideal - m) / tau))
            credits.append(min(p / ref, 1.0))
            cg = ct = 0.0
    credits.sort(reverse=True)
    return float(sum(credits[:target]))


def _geo_feasibility(vap: dict, total: np.ndarray, total_sw: float,
                     n_districts: int, midpoint, steepness, solid,
                     coords=None, smart_targets: bool = False,
                     ) -> tuple[dict[str, int], dict[str, float]]:
    """Per-group (feasible flag, reference ceiling), memoised on the demographic
    arrays so repeated proposals reuse the preparation. The gate counts
    districts reachable at the opportunity midpoint (P = 0.5); the ceiling is the
    heuristic credit reference over the top-round(T) bundles.

    With smart_targets and coordinates, both are computed over local pools rather
    than a single statewide ranking. Local pools constrain distance but do not
    enforce contiguity or construct valid districts. Without coordinates the
    scorer falls back to statewide sweeps.
    """
    if n_districts < 1 or total_sw <= 0.0:
        return {g: 0 for g in GROUPS}, {g: 0.0 for g in GROUPS}
    smart = bool(smart_targets) and coords is not None
    key = (id(vap["total"]), n_districts, round(float(total_sw), 3),
           tuple(round(_param(midpoint, g), 9) for g in GROUPS),
           tuple(round(_param(steepness, g), 9) for g in GROUPS),
           tuple(round(_param(solid, g), 9) for g in GROUPS),
           smart, id(coords) if smart else 0)
    cached = _GEO_CACHE.get(key)
    if cached is not None:
        return cached
    ideal = total_sw / n_districts
    xy = tree = None
    if smart:
        from scipy.spatial import cKDTree
        xy = _ranking_xy(coords)
        tree = cKDTree(xy)
    feasible: dict[str, int] = {}
    ceiling: dict[str, float] = {}
    for g in GROUPS:
        arr = _group_vap(vap, g)
        m = _param(midpoint, g)
        tau = max(_param(steepness, g), 1e-6)
        s = _param(solid, g)
        ref = max(_sigmoid((s - m) / tau), 1e-9)
        target = int(round((float(arr.sum()) / total_sw) * n_districts))
        if smart:
            feasible[g], ceiling[g] = _smart_targets(
                arr, total, ideal, m, tau, s, ref, target, xy, _SMART_SLACK,
                tree=tree)
        else:
            feasible[g] = _anchor_feasible(arr, total, ideal, m)
            ceiling[g] = _ceiling_credit_sum(arr, total, ideal, m, tau, s, ref,
                                             target)
    out = (feasible, ceiling)
    _GEO_CACHE[key] = out
    return out


@dataclass
class _OppPrep:
    """Assignment-independent inputs used by the fused accumulation/scoring path.

    Preparation includes statewide demographic data, curve parameters, proportional
    benchmarks, credit references, and geographic heuristic outputs.
    """
    total_f: np.ndarray                 # demographic total, pre-cast to float64
    vap_f: dict[str, np.ndarray]        # per-group counts, pre-cast to float64
    T: dict[str, float]
    ref: dict[str, float]
    m: dict[str, float]
    tau: dict[str, float]
    feasible: dict[str, int]
    ceiling: dict[str, float]
    active: dict[str, bool]             # False = group is statewide-empty (zero-filled)
    p_empty: dict[str, np.ndarray]      # constant per-district P for an empty group
    # (n_precinct, 1+n_active): col 0 = total, then active groups. int32 when the
    # Demographic table is integral and in range, halving bytes streamed per proposal;
    # float64 otherwise. Accumulation is performed in float64 either way.
    W: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    # active group -> its column in W
    active_cols: dict[str, int] = field(default_factory=dict)
    # curve params in W's active-column order, for the fused kernel
    m_arr: np.ndarray = field(default_factory=lambda: np.empty(0))
    tau_arr: np.ndarray = field(default_factory=lambda: np.empty(0))


# Constant preparation (demographic arrays, T, ref, curve parameters, and geographic
# heuristic outputs) is memoised on array identity and scalar settings.
_PREP_CACHE: dict[tuple, _OppPrep] = {}


def _param_key(value) -> object:
    """Cheap, array-free cache-key part for a scalar-or-per-group parameter."""
    if isinstance(value, dict):
        return tuple(round(float(value[g]), 9) for g in GROUPS)
    return round(float(value), 9)


def _get_prep(vap: dict, n_districts: int, midpoint, steepness, solid,
              coords=None, smart_targets: bool = False,
              prepare_targets: bool = True) -> _OppPrep:
    # Key on array identity + params only -- no full-array touch on the hot (hit)
    # path. vap arrays are stable for a run, so identity pins their contents.
    key = (id(vap["total"]), n_districts,
           _param_key(midpoint), _param_key(steepness), _param_key(solid),
           bool(smart_targets), id(coords) if coords is not None else 0,
           bool(prepare_targets))
    # Target-free exports are one-shot reads; do not retain their demographic
    # matrices or let them reuse run-scoring preparation.
    cached = _PREP_CACHE.get(key) if prepare_targets else None
    if cached is not None:
        return cached
    total_f = np.ascontiguousarray(vap["total"], dtype=np.float64)
    total_sw = float(total_f.sum())
    vap_f: dict[str, np.ndarray] = {}
    T: dict[str, float] = {}
    ref: dict[str, float] = {}
    m: dict[str, float] = {}
    tau: dict[str, float] = {}
    active: dict[str, bool] = {}
    p_empty: dict[str, np.ndarray] = {}
    for g in GROUPS:
        arr = np.ascontiguousarray(_group_vap(vap, g), dtype=np.float64)
        vap_f[g] = arr
        mg = _param(midpoint, g)
        tg = max(_param(steepness, g), 1e-6)
        sg = _param(solid, g)
        m[g] = mg
        tau[g] = tg
        ref[g] = max(_sigmoid((sg - mg) / tg), 1e-9)
        sw = float(arr.sum())
        sw_share = (sw / total_sw) if total_sw > 0.0 else 0.0
        T[g] = sw_share * n_districts
        # A statewide-empty group (zero-filled, unselected) has share 0 in every
        # district -> a constant P; skip its bincount and reuse this array.
        active[g] = sw > 0.0
        if not active[g]:
            p_empty[g] = np.full(n_districts, _sigmoid((0.0 - mg) / tg))
    if prepare_targets:
        feasible, ceiling = _geo_feasibility(
            vap, total_f, total_sw, n_districts, midpoint, steepness, solid,
            coords=coords, smart_targets=smart_targets)
    else:
        feasible, ceiling = {}, {}
    # Stack total + the active groups into one contiguous (n_precinct, 1+k)
    # matrix so the per-proposal path is a single Numba scan, not 1+k bincounts.
    # Statewide-empty groups stay out of W (their P is the constant p_empty).
    active_groups = [g for g in GROUPS if active[g]]
    cols = [total_f] + [vap_f[g] for g in active_groups]
    W = np.stack(cols, axis=1)
    # Demographic inputs are head counts, so the matrix is normally exactly representable in
    # int32 -- half the bytes to stream, and the float64 accumulation is
    # unchanged. Fall back to float64 for a fractional or out-of-range table.
    if W.size and np.all(np.isfinite(W)) and np.all(W == np.floor(W)) \
            and W.min() >= -2_147_483_648 and W.max() <= 2_147_483_647:
        W = W.astype(np.int32)
    W = np.ascontiguousarray(W)
    active_cols = {g: i + 1 for i, g in enumerate(active_groups)}
    prep = _OppPrep(total_f=total_f, vap_f=vap_f, T=T, ref=ref, m=m, tau=tau,
                    feasible=feasible, ceiling=ceiling, active=active, p_empty=p_empty,
                    W=W, active_cols=active_cols,
                    m_arr=np.array([m[g] for g in active_groups], dtype=np.float64),
                    tau_arr=np.array([tau[g] for g in active_groups],
                                     dtype=np.float64))
    if prepare_targets:
        _PREP_CACHE[key] = prep
    return prep


def warm_opportunity_geo(vap: dict, n_districts: int, *, midpoint=0.44,
                         steepness=0.05, solid=0.55, coords=None,
                         smart_targets: bool = False) -> None:
    """Precompute and cache the heuristic geographic references for these inputs.

    The preparation is memoised inside compute_opportunity. Callers may invoke
    this function before scoring so the first proposal does not pay that cost.
    """
    _get_prep(vap, n_districts, midpoint, steepness, solid,
              coords=coords, smart_targets=smart_targets)


def opportunity_targets(vap: dict, n_districts: int, *, midpoint=0.44,
                        steepness=0.05, solid=0.55, coords=None,
                        smart_targets: bool = False) -> dict[str, dict]:
    """Per-group targets for display: what the score is aiming at, before any
    assignment exists.

        target   -- opportunity districts counted (round(T), the proportional seats)
        feasible -- whether the greedy bundle heuristic finds at least one
                    qualifying bundle
        ceiling  -- credit reference from the heuristic's top target-many
                    bundles; the rating's denominator

    The bundle heuristic does not enforce district contiguity and these values
    are not proofs of plan feasibility or attainable maxima.

    Run-constant and memoised alongside the scoring path, so calling this after
    `warm_opportunity_geo` is free.
    """
    prep = _get_prep(vap, n_districts, midpoint, steepness, solid,
                     coords=coords, smart_targets=smart_targets)
    return {g: {"T": prep.T[g], "target": int(round(prep.T[g])),
                "feasible": int(prep.feasible.get(g, 0)),
                "ceiling": float(prep.ceiling.get(g, 0.0))}
            for g in GROUPS}


def compute_opportunity(
    assignment: np.ndarray,
    vap: dict,
    n_districts: int,
    *,
    midpoint=0.44,
    steepness=0.05,
    solid=0.55,
    coords=None,
    smart_targets: bool = False,
    _prepared: _OppPrep | None = None,
) -> OpportunityResult:
    """
    Args:
        assignment: per-precinct district index (int array).
        vap:        legacy-named dict of per-precinct demographic arrays with keys
                    total, white, black, latino, asian.
        n_districts: number of districts.
        midpoint/steepness/solid: logistic center/scale on the group share and the
                    "solid opportunity" reference share. Scalar or a dict keyed
                    by GROUPS.
        coords:     optional (n, 2) per-precinct coordinates. Required for smart
                    targets; only their distance ORDERING is used.
        smart_targets: build the heuristic references from distance-ranked local
                    pools instead of statewide rankings. The bundles still do
                    not enforce district contiguity. Ignored without coords.

    Returns:
        OpportunityResult with T, P, ref, feasible, ceiling keyed by GROUPS.
    """
    assignment = np.asarray(assignment)
    # The runner supplies preparation for its frozen settings. Standalone calls
    # retain the parameter-keyed lookup, including after settings change.
    prep = (_prepared if _prepared is not None else
            _get_prep(vap, n_districts, midpoint, steepness, solid,
                      coords=coords, smart_targets=smart_targets))
    # Single fused Numba pass: per-district sums for total (col 0) + every active
    # group, then the logistic curve per district.
    _, P_act = _accum_and_curve(assignment, prep.W, n_districts,
                                prep.m_arr, prep.tau_arr)
    P: dict[str, np.ndarray] = {}
    for g in GROUPS:
        if not prep.active[g]:
            P[g] = prep.p_empty[g]      # empty group: constant P, not in W
            continue
        P[g] = P_act[prep.active_cols[g] - 1]
    return OpportunityResult(T=prep.T, P=P, ref=prep.ref,
                             feasible=prep.feasible, ceiling=prep.ceiling)


def district_opportunity_credit(assignment, vap, n_districts, *, midpoint=0.44,
                                steepness=0.05, solid=0.55, groups=GROUPS):
    """Per-district normalized credit without plan-wide target preparation.

    Uses the scorer's accumulation, probability curve, and solid-share reference.
    Geographic feasibility and ceiling estimates do not affect these credits.
    """
    assignment = np.asarray(assignment)
    if (assignment.ndim != 1 or not np.issubdtype(assignment.dtype, np.integer)
            or n_districts <= 0 or np.any(assignment < 0)
            or np.any(assignment >= n_districts)):
        raise ValueError("District assignments are outside the requested district range")
    for name in ("total", *GROUPS):
        values = np.asarray(vap[name], dtype=np.float64)
        if (values.shape != assignment.shape or not np.all(np.isfinite(values))
                or np.any(values < 0)):
            raise ValueError(f"Demographic '{name}' must have one finite, "
                             "nonnegative value per precinct")
    for group in GROUPS:
        if (not all(np.isfinite(_param(p, group)) for p in (midpoint, steepness, solid))
                or _param(steepness, group) <= 0):
            raise ValueError("Opportunity parameters must be finite, with positive steepness")
    prep = _get_prep(vap, n_districts, midpoint, steepness, solid,
                     prepare_targets=False)
    opp = compute_opportunity(assignment, vap, n_districts, _prepared=prep)
    return {g: np.clip(opp.P[g] / opp.ref[g], 0.0, 1.0) for g in groups}
