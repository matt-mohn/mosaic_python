"""Core ReCom step implementation using igraph for performance."""

import numpy as np
import igraph as ig
from typing import Optional, Tuple

from mosaic.recom.tree import find_balanced_cut_ig


class GraphContext:
    """
    Precomputed static arrays derived from an igraph Graph.

    Build once, pass to every recom_step_ig call to avoid recomputing
    edge endpoint arrays on each iteration.
    """
    __slots__ = ('graph', 'edge_u', 'edge_v', 'n_nodes', 'n_edges')

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
    # Vectorised cut-edge detection — single numpy op over all edges.
    if cut_edge_indices is None:
        cut_edge_indices = np.where(
            assignment[ctx.edge_u] != assignment[ctx.edge_v]
        )[0].astype(np.int32)

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

    # Vectorised cut-edge recomputation.
    new_cut_edge_indices = np.where(
        new_assignment[ctx.edge_u] != new_assignment[ctx.edge_v]
    )[0].astype(np.int32)

    return new_assignment, True, new_cut_edge_indices


# ── Legacy NetworkX implementation (kept for reference / compatibility) ───────

import networkx as nx
from mosaic.recom.tree import find_balanced_cut
from mosaic.recom.partition import get_district_nodes


def get_cut_edges(graph: nx.Graph, assignment: np.ndarray) -> list:
    """Find all edges crossing district boundaries."""
    cut_edges = []
    for u, v in graph.edges():
        if assignment[u] != assignment[v]:
            cut_edges.append((u, v))
    return cut_edges


def check_contiguity(graph: nx.Graph, assignment: np.ndarray, district: int) -> bool:
    """Check if a district is contiguous."""
    nodes = get_district_nodes(assignment, district)
    if len(nodes) == 0:
        return False
    subgraph = graph.subgraph(nodes)
    return nx.is_connected(subgraph)


def recom_step(
    graph: nx.Graph,
    assignment: np.ndarray,
    populations: np.ndarray,
    ideal_pop: float,
    tolerance: float,
) -> Tuple[np.ndarray, bool]:
    """NetworkX version — slower, kept for compatibility."""
    cut_edges = get_cut_edges(graph, assignment)
    if not cut_edges:
        return assignment, False

    u, v = cut_edges[np.random.randint(len(cut_edges))]

    district_a = assignment[u]
    district_b = assignment[v]

    nodes_a = np.flatnonzero(assignment == district_a)
    nodes_b = np.flatnonzero(assignment == district_b)
    merged_nodes = np.concatenate([nodes_a, nodes_b])

    subgraph = graph.subgraph(merged_nodes)

    if not nx.is_connected(subgraph):
        return assignment, False

    subset = find_balanced_cut(
        subgraph,
        populations,
        ideal_pop,
        tolerance,
        max_attempts=100,
    )

    if subset is None:
        return assignment, False

    new_assignment = assignment.copy()
    subset_set = set(subset)

    for node in merged_nodes:
        if node in subset_set:
            new_assignment[node] = district_a
        else:
            new_assignment[node] = district_b

    if not check_contiguity(graph, new_assignment, district_a):
        return assignment, False
    if not check_contiguity(graph, new_assignment, district_b):
        return assignment, False

    return new_assignment, True
