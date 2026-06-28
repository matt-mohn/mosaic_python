"""Core ReCom step implementation using igraph for performance."""

import numpy as np
import igraph as ig
from typing import Optional

from mosaic.recom.tree import (
    FbcScratch,
    find_balanced_cut_fast,
    find_balanced_cut_ig,
    try_residual_balanced_cut,
    _NUMBA_OK,
)
if _NUMBA_OK:
    from mosaic.recom.tree import _nb_cut_edges


class GraphContext:
    """
    Precomputed static arrays derived from an igraph Graph.

    Build once, pass to every recom_step_ig call to avoid recomputing
    edge endpoint arrays on each iteration. Also owns the persistent scratch
    buffers for the Numba fast path (find_balanced_cut_fast).
    """
    __slots__ = ('graph', 'edge_u', 'edge_v', 'n_nodes', 'n_edges', 'scratch',
                 'real_edge_mask', 'n_virtual')

    def __init__(self, graph: ig.Graph):
        self.graph = graph
        self.n_nodes = graph.vcount()
        self.n_edges = graph.ecount()
        edges = graph.get_edgelist()
        if edges:
            arr = np.array(edges, dtype=np.int32)
            self.edge_u = np.ascontiguousarray(arr[:, 0])
            self.edge_v = np.ascontiguousarray(arr[:, 1])
        else:
            self.edge_u = np.empty(0, dtype=np.int32)
            self.edge_v = np.empty(0, dtype=np.int32)
        # Virtual bridge edges keep islands reachable for moves but must not be
        # counted as cut edges. real_edge_mask[i] is False for a virtual edge.
        if "virtual" in graph.edge_attributes():
            virt = np.asarray(graph.es["virtual"], dtype=bool)
        else:
            virt = np.zeros(self.n_edges, dtype=bool)
        self.real_edge_mask = ~virt
        self.n_virtual = int(virt.sum())
        self.scratch = FbcScratch.allocate(self.n_nodes, self.n_edges)

    def compute_cut_edges(self, assignment: np.ndarray) -> np.ndarray:
        """Vectorised cut-edge detection. Returns indices into edge_u/edge_v
        of edges whose endpoints lie in different districts.

        Includes virtual edges: move proposals select from this pool so an
        island can be merged and reassigned. The cut-edge *score* masks them
        out separately (see real_edge_mask / score_plan)."""
        if _NUMBA_OK:
            out = np.empty(self.n_edges, dtype=np.int32)
            cnt = _nb_cut_edges(self.edge_u, self.edge_v, assignment, out)
            return out[:cnt]
        return np.where(
            assignment[self.edge_u] != assignment[self.edge_v]
        )[0].astype(np.int32)


def recom_step_ig(
    ctx: GraphContext,
    assignment: np.ndarray,
    populations: np.ndarray,
    ideal_pop: float,
    tolerance: float,
    cut_edge_indices: np.ndarray | None = None,
    county_array: Optional[np.ndarray] = None,
    county_bias: float = 1.0,
) -> tuple[np.ndarray, bool, np.ndarray]:
    """
    Perform one ReCom step.

    Args:
        ctx: Precomputed graph context (edge arrays, etc.)
        assignment: Current district assignment (modified in-place on success)
        populations: Population per precinct
        ideal_pop: Target population per district
        tolerance: Fractional population deviation tolerance
        cut_edge_indices: Cached indices into ctx.edge_u/edge_v of cut edges,
                          or None to recompute from scratch.

    Returns:
        (new_assignment, success, new_cut_edge_indices)
    """
    if cut_edge_indices is None:
        cut_edge_indices = ctx.compute_cut_edges(assignment)

    if len(cut_edge_indices) == 0:
        return assignment, False, cut_edge_indices

    # Pick a random cut edge.
    rand_pos = np.random.randint(len(cut_edge_indices))
    edge_idx = cut_edge_indices[rand_pos]
    u = int(ctx.edge_u[edge_idx])
    v = int(ctx.edge_v[edge_idx])

    district_a = int(assignment[u])
    district_b = int(assignment[v])

    nodes_a = np.flatnonzero(assignment == district_a).astype(np.int32)
    nodes_b = np.flatnonzero(assignment == district_b).astype(np.int32)
    merged_nodes = np.concatenate([nodes_a, nodes_b])

    # Both districts are connected (invariant maintained by construction) and the
    # selected cut edge bridges them, so the merged region is always connected.
    if _NUMBA_OK:
        subset = find_balanced_cut_fast(
            ctx.edge_u, ctx.edge_v, ctx.scratch, merged_nodes,
            populations, ideal_pop, tolerance, max_attempts=100,
            county_array=county_array, county_bias=county_bias,
        )
    else:
        subgraph = ctx.graph.subgraph(merged_nodes)
        subset = find_balanced_cut_ig(
            subgraph, populations, ideal_pop, tolerance, max_attempts=100,
            county_array=county_array, county_bias=county_bias,
        )

    if subset is None:
        return assignment, False, cut_edge_indices

    # Vectorised assignment update: seed all merged nodes to district_b, then
    # overwrite the subset with district_a.  Two numpy fancy-index ops.
    new_assignment = assignment.copy()
    new_assignment[merged_nodes] = district_b
    subset_arr = np.array(subset, dtype=np.int32)
    new_assignment[subset_arr] = district_a

    # Spanning-tree bipartition always yields two connected subtrees in the
    # original graph — contiguity checks are provably redundant and skipped.
    new_cut_edge_indices = ctx.compute_cut_edges(new_assignment)

    return new_assignment, True, new_cut_edge_indices


# ── n=3 ReCom (merge three districts, re-split into three) ────────────────────


def _pick_third_district(
    ctx: GraphContext,
    assignment: np.ndarray,
    cut_edge_indices: np.ndarray,
    district_a: int,
    district_b: int,
) -> Optional[int]:
    """Return a district C adjacent to A or B (C != A, C != B), or None.

    Choice is weighted by the count of cut edges bridging {A,B} ↔ C. This is
    naturally ergodic (every triple with a connected merged region has positive
    probability) and biases toward triples with more shared boundary, which is
    desirable for swap potential.
    """
    if len(cut_edge_indices) == 0:
        return None

    eu = ctx.edge_u[cut_edge_indices]
    ev = ctx.edge_v[cut_edge_indices]
    da = assignment[eu]
    db = assignment[ev]

    u_in_ab = (da == district_a) | (da == district_b)
    v_in_ab = (db == district_a) | (db == district_b)
    # Exactly one endpoint in {A,B} → the other endpoint sits in some C ∉ {A,B}.
    mask = u_in_ab ^ v_in_ab
    candidate_pos = np.flatnonzero(mask)
    if len(candidate_pos) == 0:
        return None

    chosen = int(candidate_pos[np.random.randint(len(candidate_pos))])
    chosen_edge = int(cut_edge_indices[chosen])
    du = int(assignment[int(ctx.edge_u[chosen_edge])])
    dv = int(assignment[int(ctx.edge_v[chosen_edge])])
    return du if du not in (district_a, district_b) else dv


def recom_step_ig_n3(
    ctx: GraphContext,
    assignment: np.ndarray,
    populations: np.ndarray,
    ideal_pop: float,
    tolerance: float,
    cut_edge_indices: np.ndarray | None = None,
    county_array: Optional[np.ndarray] = None,
    county_bias: float = 1.0,
    max_attempts_per_stage: int = 20,
    _stats: dict | None = None,
) -> tuple[np.ndarray, bool, np.ndarray]:
    """Perform one n=3 ReCom step: merge three adjacent districts, split into three.

    Hierarchical two-stage cut on the merged region:
      Stage 1 — one_sided=True, target=ideal_pop carves off a single balanced district.
      Stage 2 — two-sided cut bisects the ~2/3 remainder into the other two districts.

    Failure modes (return success=False, assignment unchanged):
      • No cut edges at all (single-district map)
      • No third district adjacent to {A, B} (only two districts touch each other)
      • Stage 1 exceeds max_attempts_per_stage without finding a balanced cut
      • Stage 2 exceeds max_attempts_per_stage without finding a balanced cut

    Args:
        max_attempts_per_stage: Cap per stage. Smaller than the n=2 default (100)
            because n=3 fails more often — better to bail early than burn time on
            hopeless merged regions.

    Returns:
        (new_assignment, success, new_cut_edge_indices)
    """
    if cut_edge_indices is None:
        cut_edge_indices = ctx.compute_cut_edges(assignment)

    if len(cut_edge_indices) == 0:
        return assignment, False, cut_edge_indices

    rand_pos = np.random.randint(len(cut_edge_indices))
    edge_idx = int(cut_edge_indices[rand_pos])
    u = int(ctx.edge_u[edge_idx])
    v = int(ctx.edge_v[edge_idx])
    district_a = int(assignment[u])
    district_b = int(assignment[v])

    district_c = _pick_third_district(
        ctx, assignment, cut_edge_indices, district_a, district_b
    )
    if district_c is None:
        return assignment, False, cut_edge_indices

    nodes_a = np.flatnonzero(assignment == district_a).astype(np.int32)
    nodes_b = np.flatnonzero(assignment == district_b).astype(np.int32)
    nodes_c = np.flatnonzero(assignment == district_c).astype(np.int32)
    merged_nodes = np.concatenate([nodes_a, nodes_b, nodes_c])

    # Merged region is connected: A∪B via the picked cut edge, C joined via the
    # cut edge selected by _pick_third_district.
    stage1_state: dict = {}
    if _NUMBA_OK:
        subset_a_new = find_balanced_cut_fast(
            ctx.edge_u, ctx.edge_v, ctx.scratch, merged_nodes,
            populations, ideal_pop, tolerance,
            max_attempts=max_attempts_per_stage,
            one_sided=True,
            county_array=county_array, county_bias=county_bias,
            out_state=stage1_state,
        )
    else:
        subgraph_3 = ctx.graph.subgraph(merged_nodes)
        subset_a_new = find_balanced_cut_ig(
            subgraph_3, populations, ideal_pop, tolerance,
            max_attempts=max_attempts_per_stage,
            one_sided=True,
            county_array=county_array, county_bias=county_bias,
            out_state=stage1_state,
        )
    if subset_a_new is None:
        return assignment, False, cut_edge_indices

    subset_a_arr = np.array(subset_a_new, dtype=np.int32)

    subset_b_new = try_residual_balanced_cut(
        stage1_state, ideal_pop, tolerance,
    )

    if subset_b_new is not None:
        if _stats is not None:
            _stats["residual_hits"] = _stats.get("residual_hits", 0) + 1
    else:
        if _stats is not None:
            _stats["residual_misses"] = _stats.get("residual_misses", 0) + 1
        # Fall back to a fresh 2-region MST on the post-stage-1 remainder.
        # node_ids is in local-index order matching carved_sub_idx, so the
        # mask maps directly — no np.isin sort needed.
        keep = np.ones(stage1_state["n"], dtype=np.bool_)
        keep[stage1_state["carved_sub_idx"]] = False
        remaining_nodes = stage1_state["node_ids"][keep].astype(np.int32, copy=False)
        if _NUMBA_OK:
            subset_b_new = find_balanced_cut_fast(
                ctx.edge_u, ctx.edge_v, ctx.scratch, remaining_nodes,
                populations, ideal_pop, tolerance,
                max_attempts=max_attempts_per_stage,
                one_sided=False,
                county_array=county_array, county_bias=county_bias,
            )
        else:
            subgraph_2 = ctx.graph.subgraph(remaining_nodes)
            subset_b_new = find_balanced_cut_ig(
                subgraph_2, populations, ideal_pop, tolerance,
                max_attempts=max_attempts_per_stage,
                one_sided=False,
                county_array=county_array, county_bias=county_bias,
            )
        if subset_b_new is None:
            return assignment, False, cut_edge_indices

    subset_b_arr = np.array(subset_b_new, dtype=np.int32)

    new_assignment = assignment.copy()
    new_assignment[merged_nodes] = district_c
    new_assignment[subset_a_arr] = district_a
    new_assignment[subset_b_arr] = district_b

    new_cut_edge_indices = ctx.compute_cut_edges(new_assignment)

    return new_assignment, True, new_cut_edge_indices
