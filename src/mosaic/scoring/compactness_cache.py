"""Exact per-proposal compactness cache; changed districts are rebuilt in order.

Every proposal rebuilds every affected district before acceptance.
Floating-point geometry uses the full scorers' precinct/edge summation order,
never add/subtract deltas. Snapshots' internal arrays must not be mutated.
"""

from dataclasses import dataclass

import numpy as np
from numba import njit

from mosaic.scoring.polsby_popper import _TWO_PI


@njit(cache=True)
def _affected(old, new, n_districts):
    dirty = np.zeros(n_districts, dtype=np.bool_)
    for i in range(len(new)):
        if new[i] < 0 or new[i] >= n_districts:
            raise ValueError("District index out of range")
        if old[i] != new[i]:
            dirty[old[i]] = True
            dirty[new[i]] = True
    return dirty


@njit(cache=True)
def _pp_update(assignment, areas, ext, eu, ev, elen, dirty, previous, nodes):
    values = previous.copy()  # area, exterior perimeter, u-cut and v-cut sums
    n_districts = len(dirty)
    for d in range(n_districts):
        if dirty[d]:
            for row in range(4):
                values[row, d] = 0.0
    for i in nodes:
        d = assignment[i]
        values[0, d] += areas[i]
        values[1, d] += ext[i]
    # Preserve full-scorer edge order and separate u/v accumulators. No deltas
    # are applied to floating-point geometry, so rounding cannot accumulate.
    for i in range(len(eu)):
        du, dv = assignment[eu[i]], assignment[ev[i]]
        if du != dv:
            if dirty[du]:
                values[2, du] += elen[i]
            if dirty[dv]:
                values[3, dv] += elen[i]
    total = 0.0
    for d in range(n_districts):
        perim = (values[1, d] + values[2, d]) + values[3, d]
        sp = perim if perim > 0.0 else 1.0
        pp = _TWO_PI * values[0, d] / (sp * sp)
        if pp > 1.0:
            pp = 1.0
        elif pp < 0.0:
            pp = 0.0
        total += pp
    return values, (1.0 - total / n_districts) * 100.0


@njit(cache=True, fastmath=True)
def _reock_update(points, projections, assignment, areas, dirty,
                   previous_proj, previous_idx, previous_areas, nodes):
    k = projections.shape[1]
    n_districts = len(dirty)
    max_proj = previous_proj.copy()
    max_idx = previous_idx.copy()
    district_areas = previous_areas.copy()
    for d in range(n_districts):
        if dirty[d]:
            district_areas[d] = 0.0
            for ki in range(k):
                max_proj[d, ki] = -1e30
                max_idx[d, ki] = -1
    # Ascending precinct order preserves full-scorer area sums and argmax
    # tie-breaking. Rebuild all extrema of changed districts, including removals.
    for i in nodes:
        d = assignment[i]
        district_areas[d] += areas[i]
        for ki in range(k):
            value = projections[i, ki]
            if value > max_proj[d, ki]:
                max_proj[d, ki] = value
                max_idx[d, ki] = i
    # Same arithmetic and district order as the full scorer, including its
    # existing fastmath convention. Every district contributes every proposal.
    total = 0.0
    for d in range(n_districts):
        d_max_sq = 0.0
        for ki in range(k):
            pi = max_idx[d, ki]
            if pi < 0:
                continue
            xi = points[ki, pi, 0]
            yi = points[ki, pi, 1]
            for kj in range(ki + 1, k):
                pj = max_idx[d, kj]
                if pj < 0:
                    continue
                dx = xi - points[kj, pj, 0]
                dy = yi - points[kj, pj, 1]
                ds = dx * dx + dy * dy
                if ds > d_max_sq:
                    d_max_sq = ds
        r = np.sqrt(d_max_sq) / 2.0
        if r > 0.0:
            rk = district_areas[d] / (np.pi * r * r)
            if rk > 1.0:
                rk = 1.0
            total += rk
    return max_proj, max_idx, district_areas, (1.0 - total / n_districts) * 100.0


@njit(cache=True)
def _update_both(old_assignment, assignment, n_districts, pp, rd,
                 previous_pp, previous_proj, previous_idx, previous_areas, initialize):
    dirty = (np.ones(n_districts, dtype=np.bool_) if initialize
             else _affected(old_assignment, assignment, n_districts))
    nodes = np.empty(len(assignment), dtype=np.int32)
    count = 0
    for i in range(len(assignment)):
        if dirty[assignment[i]]:
            nodes[count] = i
            count += 1
    nodes = nodes[:count]
    values, pp_score = _pp_update(assignment, pp[0], pp[1], pp[2], pp[3], pp[4],
                                 dirty, previous_pp, nodes)
    proj, idx, areas, reock_score = _reock_update(
        rd[0], rd[1], assignment, rd[2], dirty, previous_proj, previous_idx, previous_areas, nodes)
    return values, proj, idx, areas, pp_score, reock_score


@dataclass(frozen=True, slots=True)
class CompactnessSnapshot:
    assignment: np.ndarray
    pp_values: np.ndarray
    reock_proj: np.ndarray
    reock_idx: np.ndarray
    reock_areas: np.ndarray
    pp_score: float
    reock_score: float
    generation: int
    owner: object


class CompactnessCache:
    """One run, one accepted state; proposals never mutate accepted buffers.

    Assignment arrays are borrowed from the runner, whose move generators
    allocate a new assignment for each proposal. They must not be mutated.
    Geometry is immutable for the lifetime of the run. No periodic refresh.
    """

    def __init__(self, assignment, n_districts, pp_data, reock_data):
        self.n_districts = n_districts
        self.pp_data = pp_data
        self.reock_data = reock_data
        self._pp_inputs = (pp_data.areas, pp_data.ext_perimeters,
                           pp_data.edge_u, pp_data.edge_v, pp_data.edge_len)
        self._reock_inputs = (reock_data.dir_ext_pts, reock_data.dir_ext_proj, reock_data.areas)
        self.generation = 0
        self.owner = object()
        self._validate(assignment)
        k = reock_data.dir_ext_proj.shape[1]
        self.current = CompactnessSnapshot(
            assignment, np.zeros((4, n_districts)),
            np.full((n_districts, k), -1e30), np.full((n_districts, k), -1, dtype=np.int64),
            np.zeros(n_districts), 0.0, 0.0, -1, self.owner)
        self.current = self._propose(assignment, initialize=True)

    def _validate(self, assignment):
        if (self.n_districts <= 0 or assignment.ndim != 1
                or assignment.dtype.kind not in "iu"
                or len(assignment) != len(self.pp_data.areas)
                or len(assignment) != len(self.reock_data.areas)
                or np.any(assignment < 0) or np.any(assignment >= self.n_districts)):
            raise ValueError("Invalid compactness assignment")

    def _propose(self, assignment, initialize=False):
        old = self.current
        values, proj, idx, areas, pp_score, reock_score = _update_both(
            old.assignment, assignment, self.n_districts, self._pp_inputs, self._reock_inputs,
            old.pp_values, old.reock_proj, old.reock_idx, old.reock_areas, initialize)
        return CompactnessSnapshot(assignment, values, proj, idx, areas,
                                    pp_score, reock_score, self.generation, self.owner)

    def propose(self, assignment):
        if assignment.shape != self.current.assignment.shape or assignment.dtype.kind not in "iu":
            raise ValueError("Invalid compactness assignment")
        return self._propose(assignment)

    def accept(self, snapshot):
        if snapshot.owner is not self.owner or snapshot.generation != self.generation:
            raise ValueError("Cannot accept a foreign or stale compactness snapshot")
        self.current = snapshot
        self.generation += 1
