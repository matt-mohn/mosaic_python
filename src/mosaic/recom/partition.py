"""Initial partition generation via sequential balanced bisection."""

import logging
import numpy as np
import networkx as nx
import igraph as ig
from typing import Callable

from mosaic.recom.tree import find_balanced_cut_ig

log = logging.getLogger("mosaic")

_DISTRICT_TIMEOUT = 5.0
_MAX_RESTARTS = 10


def _nx_subgraph_to_ig(graph: nx.Graph, nodes: set) -> ig.Graph:
    """Build an igraph subgraph from a NetworkX graph restricted to given nodes."""
    node_list = sorted(nodes)
    idx = {n: i for i, n in enumerate(node_list)}
    edges = [(idx[u], idx[v]) for u, v in graph.subgraph(nodes).edges()]
    g = ig.Graph(n=len(node_list), edges=edges)
    g.vs["name"] = node_list
    return g


def create_initial_partition(
    graph: nx.Graph,
    populations: np.ndarray,
    num_districts: int,
    tolerance: float,
    seed: int | None = None,
    on_progress: Callable[[int, int], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> np.ndarray | None:
    """
    Create an initial district assignment by sequentially carving one district
    at a time from the remaining precinct graph.

    Args:
        graph: Precinct adjacency graph. Nodes should be 0..N-1.
        populations: Population array indexed by node ID
        num_districts: Number of districts to create
        tolerance: Population deviation tolerance (e.g., 0.05 for 5%)
        seed: Random seed for reproducibility
        on_progress: Callback(district_num, num_districts) for progress updates
        should_cancel: Optional callable; if it returns True partitioning stops
                       and None is returned immediately.

    Returns:
        Array of district assignments (0 to num_districts-1), or None if
        cancelled via should_cancel.
    """
    if seed is not None:
        np.random.seed(seed)

    for restart in range(_MAX_RESTARTS):
        if should_cancel and should_cancel():
            return None
        result = _try_partition(
            graph, populations, num_districts, tolerance, on_progress, should_cancel
        )
        if result is not None:
            return result
        if should_cancel and should_cancel():
            return None
        log.warning(f"Partition attempt {restart + 1} timed out, restarting...")

    raise RuntimeError(
        f"Could not create partition after {_MAX_RESTARTS} attempts. "
        f"Try relaxing the population tolerance."
    )


def _try_partition(
    graph: nx.Graph,
    populations: np.ndarray,
    num_districts: int,
    tolerance: float,
    on_progress: Callable[[int, int], None] | None,
    should_cancel: Callable[[], bool] | None = None,
) -> np.ndarray | None:
    """
    Attempt a sequential partition. Returns None if any cut times out.

    Carves districts 0..N-2 one at a time using one_sided=True cuts, so only
    the carved district needs to be within tolerance. The very last cut uses
    one_sided=False to ensure both remaining districts are valid (prevents the
    last district from drifting out of tolerance and never recovering in annealing).
    """
    n = graph.number_of_nodes()
    total_pop = populations.sum()
    ideal_pop = total_pop / num_districts

    assignment = np.full(n, -1, dtype=np.int32)
    remaining_nodes = set(graph.nodes())

    for district in range(num_districts - 1):
        # The second-to-last cut must constrain both remaining districts.
        is_last_cut = (district == num_districts - 2)

        ig_sub = _nx_subgraph_to_ig(graph, remaining_nodes)

        carved = find_balanced_cut_ig(
            ig_sub,
            populations,
            ideal_pop,
            tolerance,
            max_attempts=10000,
            one_sided=not is_last_cut,
            timeout=_DISTRICT_TIMEOUT,
        )

        if carved is None:
            log.warning(
                f"District {district + 1}/{num_districts} timed out "
                f"({len(remaining_nodes)} nodes remaining)"
            )
            return None

        if should_cancel and should_cancel():
            return None

        for node in carved:
            assignment[node] = district
        remaining_nodes -= set(carved)

        if on_progress:
            on_progress(district + 1, num_districts)

    # Assign all remaining nodes to the last district
    for node in remaining_nodes:
        assignment[node] = num_districts - 1

    if on_progress:
        on_progress(num_districts, num_districts)

    return assignment


def get_district_nodes(assignment: np.ndarray, district: int) -> np.ndarray:
    """Get array of node IDs assigned to a district."""
    return np.where(assignment == district)[0]


def get_district_populations(
    populations: np.ndarray,
    assignment: np.ndarray,
    num_districts: int,
) -> np.ndarray:
    """Get population of each district."""
    return np.array([
        populations[assignment == d].sum()
        for d in range(num_districts)
    ])
