"""Initial partition generation."""

import logging
import numpy as np
import networkx as nx
from typing import Callable

from mosaic.recom.tree import find_balanced_cut

log = logging.getLogger("mosaic")


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

    n = graph.number_of_nodes()
    total_pop = populations.sum()
    ideal_pop = total_pop / num_districts

    assignment = np.full(n, -1, dtype=np.int32)
    remaining_nodes = set(graph.nodes())

    for district in range(num_districts - 1):
        if on_progress:
            on_progress(district + 1, num_districts)

        log.info(f"Creating district {district + 1}/{num_districts}...")

        # Build subgraph of remaining nodes
        subgraph = graph.subgraph(remaining_nodes).copy()

        if not nx.is_connected(subgraph):
            # If disconnected, work with largest component
            components = list(nx.connected_components(subgraph))
            largest = max(components, key=len)
            log.warning(f"Subgraph disconnected, using largest component ({len(largest)} of {len(remaining_nodes)} nodes)")
            subgraph = graph.subgraph(largest).copy()

        remaining_pop = sum(populations[n] for n in remaining_nodes)
        log.info(f"  Remaining: {len(remaining_nodes)} nodes, {remaining_pop:,} pop")

        # Find a balanced cut (one_sided=True for initial partition)
        subset = find_balanced_cut(
            subgraph,
            populations,
            ideal_pop,
            tolerance,
            max_attempts=10000,
            one_sided=True,  # Only carved-off district needs to be within tolerance
        )

        if subset is None:
            raise RuntimeError(
                f"Could not find balanced partition for district {district + 1}. "
                f"Try relaxing the population tolerance."
            )

        subset_pop = sum(populations[n] for n in subset)
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
