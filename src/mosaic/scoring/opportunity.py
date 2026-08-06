"""
Opportunity-to-elect engine -- shared per-district minority opportunity model.

Per district and group, the probability the group can elect its candidate of
choice, on a logistic curve over the group's voting-age population share:

    P_g,d = sigmoid((share_g,d - midpoint) / steepness)

Also per group: `T_g` = (statewide group share) * n, the proportional benchmark;
`ref_g` = the curve at a "solid" opportunity share, the credit denominator so a
solid district counts as ~one without a packed super-majority; `feasible_g`,
whether one opportunity district is drawable at all; and `ceiling_g` = f(state),
the credit the top-round(T) districts could reach given the geography.
representation.py consumes all five.

The geographic pair comes from `_smart_targets` (local pools, the default) or
from the statewide sweeps `_anchor_feasible` / `_ceiling_credit_sum` without
coordinates. Built once per score_plan call, like the partisan shares/p_wins pass.

Groups scored: black, latino, asian, each on its own curve. A coalition group is
deliberately not scored -- on a VAP basis it rewards cracking one group into ~50%
non-white districts electing no single group's candidate, masking dilution.
Curve params accept a per-group dict, so absorbing the VAP-vs-CVAP gap for
latino/asian is a config change rather than a rewrite.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numba import njit

# Group order is stable; consumers iterate this. Single groups only -- coalition
# is deliberately excluded (see module docstring).
GROUPS: tuple[str, ...] = ("black", "latino", "asian")


@njit(cache=True)
def _accum_and_curve(assignment, weights, n_districts, m, tau):
    """One precinct-order pass over the (n, 1+G) weight matrix (column 0 = total
    VAP, columns 1.. = per-group VAP), then the logistic curve per district.

    Fusing the scan and the curve replaces 1+G bincounts plus ~4 tiny NumPy calls
    per group on k-length arrays, where call overhead dwarfs the arithmetic.
    float64 accumulation keeps an integer `weights` dtype bit-identical.

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
        if den < 1.0:            # districts with 0 VAP -> share 0
            den = 1.0
        for j in range(ngrp):
            share = sums[d, j + 1] / den
            P[j, d] = 1.0 / (1.0 + np.exp(-(share - m[j]) / tau[j]))
    return sums, P

# Benchmark weight per group. Kept as a dict so a future coalition term can be
# reintroduced at a discount without reworking consumers.
DISCOUNT: dict[str, float] = {"black": 1.0, "latino": 1.0, "asian": 1.0}


@dataclass
class OpportunityResult:
    """Per-group opportunity summary, all keyed by GROUPS.

    T:        proportional benchmark in districts (statewide group VAP share * n).
    P:        per-district opportunity probability array.
    ref:      curve value at the 'solid' opportunity share; the per-district
              credit denominator (P / ref capped at 1 == effective opp districts).
    feasible: whether >=1 opportunity district is drawable at all (N/A gate).
    ceiling:  f(state) -- max credit the top-round(T) districts could achieve
              given the geography; the rating's denominator so 100 is reachable.
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


# Assignment-independent, so memoised per (state VAP arrays, n, curve params)
# and computed once per run rather than once per proposal.
_GEO_CACHE: dict[tuple, tuple[dict[str, int], dict[str, float]]] = {}


# ── Smart targets: local-pool geography ──────────────────────────────────────
# Statewide sweeps rank every precinct, so their bundles can span places no
# district could reach (PA latino: 517 precincts across 20 counties). Smart
# targets confine each hypothetical district to the nearest precincts to a centre
# until they hold `slack` x an ideal district's VAP, then run the same greedy.
# Sizing the pool in POPULATION makes one constant work at any state and k, and
# distance only RANKS neighbours, so the CRS unit cancels.
#
# slack 1 = a compact ball; 2 = may take the best half of a two-district
# neighbourhood; inf = the statewide behaviour.
_SMART_SLACK = 2.0

# Cost is centres x rounds, and rounds = the group's target (up to 20 for CA
# latino), so a fixed budget of centre-evaluations holds cost flat instead of
# running slow there and coarse in a small state. Saturation is state-dependent
# (CA converges by 200 centres, NC's black ceiling still climbs at 800) and too
# low a ceiling inflates ratings.
_SMART_CENTRE_BUDGET = 1200
_SMART_CENTRE_MIN, _SMART_CENTRE_MAX = 100, 600

# Credit the best single local district must earn for the group to be drawable.
# Gating on credit rather than the midpoint avoids a knife-edge: CA's best black
# district is 42.7% share (credit 0.48), which a hard 0.44 cutoff rejects. Groups
# that truly cannot form one sit far below (PA latino 0.028).
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
    `stop_group` of group VAP (so a district is never over-packed) or a full
    district's total VAP. Returns (group VAP, total VAP, members)."""
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
    `score_fn(cg, ct)`. Centres are re-derived from the still-available precincts
    each round; deriving them once and rejecting overlaps strands the next
    district on a distant weak bundle once the densest core is consumed."""
    n = total.shape[0]
    if n == 0 or ideal <= 0.0 or n_rounds < 1:
        return []
    share = np.divide(grp, total, out=np.zeros_like(grp, dtype=np.float64),
                      where=total > 0.0)
    if tree is None:
        from scipy.spatial import cKDTree
        tree = cKDTree(xy)
    mean_vap = max(float(total.mean()), 1.0)
    # neighbours per centre that could hold slack x ideal VAP, with headroom
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
    """Smart-target (feasibility gate, f(state) ceiling) from one local sweep.

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
    """Optimistic count of ideal-sized districts that could reach `threshold`
    group VAP share. Bundle the highest-share precincts until a bundle holds
    threshold*ideal group VAP -- enough to be one threshold-level district once
    padded to ideal size -- then start the next; a bundle that fills a whole
    district's population without getting there is spent. Ignores contiguity (an
    upper-ish bound, the safe direction for a gate) but honours that
    concentrated minority VAP can't be diluted below threshold, so it returns 0
    when no district can clear the bar. Middle ground between packing (wastes
    VAP, undercounts) and a raw VAP budget (assumes perfect spread)."""
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
        if cg >= need:                # enough minority VAP -> one opportunity district
            count += 1
            cg = ct = 0.0
        elif ct >= ideal:             # full district's population, still short -> spent
            cg = ct = 0.0
    return count


def _ceiling_credit_sum(grp: np.ndarray, total: np.ndarray, ideal: float,
                        m: float, tau: float, s: float, ref: float,
                        target: int) -> float:
    """f(state): the max clipped-credit sum the top `target` districts could
    achieve given this geography -- the achievable optimum the rating normalises
    against, so a geographically-best map reaches 100.

    Credit-maximising arrangement: since credit caps at 1.0 at the solid share,
    the way to maximise total credit is to bring as many districts as possible to
    exactly solid (each uses the *minimum* group VAP for full credit, leaving more
    for others), packing highest-share precincts first; leftover bundles that fill
    a whole district without reaching solid earn partial credit. Ignores
    contiguity (an upper-ish bound), so it is mildly optimistic."""
    if ideal <= 0.0 or target < 1:
        return 0.0
    share = np.divide(grp, total, out=np.zeros_like(grp, dtype=np.float64),
                      where=total > 0.0)
    order = np.argsort(share)[::-1]
    ts = total[order]
    gs = grp[order]
    need = s * ideal                      # group VAP for one solid (credit 1.0) district
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
    """Per-group (feasible-gate, ceiling f(state)), memoised on the state's VAP
    arrays so the sweeps run once per run, not once per proposal. The gate counts
    districts reachable at the opportunity midpoint (P = 0.5); the ceiling is the
    achievable credit over the top-round(T) districts.

    With smart_targets and coordinates, both are computed over local pools rather
    than statewide, so neither can score a bundle assembled from places no district
    could span. Without coordinates it falls back to the statewide sweeps.
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
    """Run-constant opportunity inputs, memoised so the per-proposal path is just
    bincounts + the per-district sigmoid. Everything here is independent of the
    assignment (statewide VAP, curve params, benchmark T, ref, geography)."""
    total_f: np.ndarray                 # VAP total, pre-cast to float64
    vap_f: dict[str, np.ndarray]        # per-group VAP, pre-cast to float64
    T: dict[str, float]
    ref: dict[str, float]
    m: dict[str, float]
    tau: dict[str, float]
    feasible: dict[str, int]
    ceiling: dict[str, float]
    active: dict[str, bool]             # False = group is statewide-empty (zero-filled)
    p_empty: dict[str, np.ndarray]      # constant per-district P for an empty group
    # (n_precinct, 1+n_active): col 0 = total, then active groups. int32 when the
    # VAP table is integral and in range, halving the bytes streamed per proposal;
    # float64 otherwise. The float64 accumulation is bit-identical either way.
    W: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    # active group -> its column in W
    active_cols: dict[str, int] = field(default_factory=dict)
    # curve params in W's active-column order, for the fused kernel
    m_arr: np.ndarray = field(default_factory=lambda: np.empty(0))
    tau_arr: np.ndarray = field(default_factory=lambda: np.empty(0))


# Constant prep (float64 VAP arrays, T, ref, curve params, geography) memoised on
# the same (state VAP arrays, n, curve params) key as _GEO_CACHE, so it is built
# once per run rather than once per proposal.
_PREP_CACHE: dict[tuple, _OppPrep] = {}


def _param_key(value) -> object:
    """Cheap, array-free cache-key part for a scalar-or-per-group parameter."""
    if isinstance(value, dict):
        return tuple(round(float(value[g]), 9) for g in GROUPS)
    return round(float(value), 9)


def _get_prep(vap: dict, n_districts: int, midpoint, steepness, solid,
              coords=None, smart_targets: bool = False) -> _OppPrep:
    # Key on array identity + params only -- no full-array touch on the hot (hit)
    # path. vap arrays are stable for a run, so identity pins their contents.
    key = (id(vap["total"]), n_districts,
           _param_key(midpoint), _param_key(steepness), _param_key(solid),
           bool(smart_targets), id(coords) if coords is not None else 0)
    cached = _PREP_CACHE.get(key)
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
    feasible, ceiling = _geo_feasibility(
        vap, total_f, total_sw, n_districts, midpoint, steepness, solid,
        coords=coords, smart_targets=smart_targets)
    # Stack total + the active groups into one contiguous (n_precinct, 1+k)
    # matrix so the per-proposal path is a single Numba scan, not 1+k bincounts.
    # Statewide-empty groups stay out of W (their P is the constant p_empty).
    active_groups = [g for g in GROUPS if active[g]]
    cols = [total_f] + [vap_f[g] for g in active_groups]
    W = np.stack(cols, axis=1)
    # VAP is a head count, so the matrix is normally exactly representable in
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
    _PREP_CACHE[key] = prep
    return prep


def warm_opportunity_geo(vap: dict, n_districts: int, *, midpoint=0.44,
                         steepness=0.05, solid=0.55, coords=None,
                         smart_targets: bool = False) -> None:
    """Precompute and cache the geographic gate/ceiling for these inputs.

    The geography sweep is memoised inside compute_opportunity, so without this it
    runs on the first scored proposal -- which reads as an unexplained stall at run
    start. Callers warm it at a point where they can say so in the status line.
    """
    _get_prep(vap, n_districts, midpoint, steepness, solid,
              coords=coords, smart_targets=smart_targets)


def opportunity_targets(vap: dict, n_districts: int, *, midpoint=0.44,
                        steepness=0.05, solid=0.55, coords=None,
                        smart_targets: bool = False) -> dict[str, dict]:
    """Per-group targets for display: what the score is aiming at, before any
    assignment exists.

        target   -- opportunity districts counted (round(T), the proportional seats)
        feasible -- 0 when the geography cannot yield even one such district
        ceiling  -- f(state), the achievable credit those target-many districts
                    could reach; the rating's denominator

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
) -> OpportunityResult:
    """
    Args:
        assignment: per-precinct district index (int array).
        vap:        dict of per-precinct VAP arrays with keys
                    total, white, black, latino, asian.
        n_districts: number of districts.
        midpoint/steepness/solid: logistic center/scale on the VAP share and the
                    "solid opportunity" reference share. Scalar (shared, the v1
                    default) or a dict keyed by GROUPS.
        coords:     optional (n, 2) per-precinct coordinates. Required for smart
                    targets; only their distance ORDERING is used.
        smart_targets: restrict the feasibility gate and the achievable ceiling to
                    local pools instead of statewide sweeps. Ignored without
                    coords.

    Returns:
        OpportunityResult with T, P, ref, feasible, ceiling keyed by GROUPS.
    """
    assignment = np.asarray(assignment)
    prep = _get_prep(vap, n_districts, midpoint, steepness, solid,
                     coords=coords, smart_targets=smart_targets)
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
