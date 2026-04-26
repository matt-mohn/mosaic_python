"""Initial partition generation."""

import logging
import time
import numpy as np
import networkx as nx
from typing import Callable

from mosaic.recom.tree import find_balanced_cut

log = logging.getLogger("mosaic")

# Per-district timeout before restarting partition (seconds)
_DISTRICT_TIMEOUT = 5.0
_MAX_RESTARTS = 10


def create_initial_partition(
    graph: nx.Graph,
    populations: np.ndarray,
    num_districts: int,
    tolerance: float,
    seed: int | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> np.ndarray:
    """
    Create an initial district assignment using recursive bipartition.

    Args:
        graph: Precinct adjacency graph. Nodes should be 0..N-1.
        populations: Population array indexed by node ID
        num_districts: Number of districts to create
        tolerance: Population deviation tolerance (e.g., 0.05 for 5%)
        seed: Random seed for reproducibility
        on_progress: Callback(district_num, num_districts) for progress updates

    Returns:
        Array of district assignments (0 to num_districts-1) indexed by node ID
    """
    if seed is not None:
        np.random.seed(seed)

    for restart in range(_MAX_RESTARTS):
        result = _try_partition(
            graph, populations, num_districts, tolerance, on_progress
        )
        if result is not None:
            return result
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
) -> np.ndarray | None:
    """
    Attempt to create a partition. Returns None if any district times out.
    """
    n = graph.number_of_nodes()
    total_pop = populations.sum()
    ideal_pop = total_pop / num_districts

    assignment = np.full(n, -1, dtype=np.int32)
    remaining_nodes = set(graph.nodes())

    for district in range(num_districts - 1):
        if on_progress:
            on_progress(district + 1, num_districts)

        log.info(f"Creating district {district + 1}/{num_districts}...")
        district_start = time.perf_counter()

        # Build subgraph of remaining nodes
        subgraph = graph.subgraph(remaining_nodes).copy()

        if not nx.is_connected(subgraph):
            # If disconnected, work with largest component
            components = list(nx.connected_components(subgraph))
            largest = max(components, key=len)
            log.warning(f"Subgraph disconnected, using largest component ({len(largest)} of {len(remaining_nodes)} nodes)")
            subgraph = graph.subgraph(largest).copy()

        remaining_pop = sum(populations[node] for node in remaining_nodes)
        log.info(f"  Remaining: {len(remaining_nodes)} nodes, {remaining_pop:,} pop")

        # Find a balanced cut.
        # For the last iteration (creating district num_districts-2), use one_sided=False
        # so that BOTH the carved district AND the remaining district (num_districts-1)
        # are guaranteed to be within population tolerance.
        is_last_cut = (district == num_districts - 2)
        subset = find_balanced_cut(
            subgraph,
            populations,
            ideal_pop,
            tolerance,
            max_attempts=10000,
            one_sided=not is_last_cut,  # Both sides must be valid on final cut
        )

        elapsed = time.perf_counter() - district_start
        if elapsed > _DISTRICT_TIMEOUT:
            log.warning(f"District {district + 1} took {elapsed:.1f}s (>{_DISTRICT_TIMEOUT}s)")
            return None

        if subset is None:
            log.warning(f"Could not find balanced cut for district {district + 1}")
            return None

        subset_pop = sum(populations[node] for node in subset)
        deviation = (subset_pop - ideal_pop) / ideal_pop * 100
        log.info(f"  District {district + 1}: {len(subset)} nodes, {subset_pop:,} pop ({deviation:+.2f}%)")

        # Assign nodes to this district
        for node in subset:
            assignment[node] = district

        remaining_nodes -= set(subset)

    # Assign remaining nodes to last district
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
