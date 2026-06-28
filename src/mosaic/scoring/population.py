"""Scoring functions for redistricting plans."""

import numpy as np
import networkx as nx
from dataclasses import dataclass


@dataclass
class PlanStats:
    """Basic statistics for a districting plan."""
    num_cut_edges: int      # Edges crossing district boundaries
    district_populations: np.ndarray
    max_pop_deviation: float


def count_cut_edges(graph: nx.Graph, assignment: np.ndarray) -> int:
    """
    Count edges that cross district boundaries.

    Args:
        graph: Precinct adjacency graph
        assignment: District assignment array

    Returns:
        Number of cut edges. Virtual bridge edges (attribute ``virtual``) are
        excluded so island bridges stay invisible to scoring, matching the
        hot-loop count in score_plan.
    """
    count = 0
    for u, v, data in graph.edges(data=True):
        if data.get("virtual"):
            continue
        if assignment[u] != assignment[v]:
            count += 1
    return count


def get_plan_stats(
    graph: nx.Graph,
    populations: np.ndarray,
    assignment: np.ndarray,
    num_districts: int,
) -> PlanStats:
    """
    Calculate basic statistics for a districting plan.

    Args:
        graph: Precinct adjacency graph
        populations: Population array indexed by node ID
        assignment: District assignment array
        num_districts: Number of districts

    Returns:
        PlanStats with cut edges and population info
    """
    # Count cut edges
    num_cut_edges = count_cut_edges(graph, assignment)

    # Calculate district populations
    district_pops = np.array([
        populations[assignment == d].sum()
        for d in range(num_districts)
    ])

    # Calculate max deviation
    ideal_pop = populations.sum() / num_districts
    deviations = np.abs(district_pops - ideal_pop) / ideal_pop
    max_deviation = deviations.max()

    return PlanStats(
        num_cut_edges=num_cut_edges,
        district_populations=district_pops,
        max_pop_deviation=max_deviation,
    )


def score_pop_deviation(
    assignment: np.ndarray,
    populations: np.ndarray,
    ideal_pop: float,
    n_districts: int,
    safe_harbor: float = 0.0025,
    return_components: bool = False,
) -> float | tuple[float, float, float]:
    """
    Population balance score using sum of squared excess deviations.

    Score = sum(excess²) × 100,000 / n_districts, where
    excess = max(0, |dev_frac| - safe_harbor).
    Quadratic scaling naturally prioritizes outliers while providing smooth
    gradients for annealing. Lower is better; 0 means all districts are
    within the safe-harbor.

    If return_components=True, returns (score, max_dev_pct, mean_dev_pct)
    where max_dev_pct and mean_dev_pct are actual percentages (0-100).
    """
    ideal = max(float(ideal_pop), 1.0)
    pop_d = np.bincount(assignment, weights=populations.astype(np.float64),
                        minlength=n_districts)
    abs_dev = np.abs(pop_d - ideal) / ideal
    excess = np.maximum(0.0, abs_dev - safe_harbor)
    excess_sq = excess ** 2
    # Sum-of-squares provides smoother optimization landscape than mean+max.
    # Scale factor chosen to roughly match previous (mean+max)×50k at typical
    # uniform deviations, so existing weight tuning remains valid.
    combined = float(np.sum(excess_sq) * 100_000 / n_districts)

    if return_components:
        max_dev_pct = float(abs_dev.max() * 100.0)
        mean_dev_pct = float(abs_dev.mean() * 100.0)
        return combined, max_dev_pct, mean_dev_pct
    return combined


# Keep these for backwards compatibility / future use
def calculate_population_score(
    populations: np.ndarray,
    assignment: np.ndarray,
    num_districts: int,
) -> float:
    """Calculate max population deviation (lower is better)."""
    total_pop = populations.sum()
    ideal_pop = total_pop / num_districts

    district_pops = np.array([
        populations[assignment == d].sum()
        for d in range(num_districts)
    ])

    deviations = np.abs(district_pops - ideal_pop) / ideal_pop
    return deviations.max()


def get_population_stats(populations, assignment, num_districts):
    """Backwards-compatible function."""
    from dataclasses import dataclass as dc

    @dc
    class PopulationStats:
        district_populations: np.ndarray
        ideal_population: float
        deviations: np.ndarray
        max_deviation: float
        mean_deviation: float

    total_pop = populations.sum()
    ideal_pop = total_pop / num_districts
    district_pops = np.array([
        populations[assignment == d].sum()
        for d in range(num_districts)
    ])
    deviations = (district_pops - ideal_pop) / ideal_pop

    return PopulationStats(
        district_populations=district_pops,
        ideal_population=ideal_pop,
        deviations=deviations,
        max_deviation=np.abs(deviations).max(),
        mean_deviation=np.abs(deviations).mean(),
    )
