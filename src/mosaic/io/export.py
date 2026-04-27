"""Export utilities for redistricting results."""

import logging
import numpy as np
import pandas as pd
from pathlib import Path

log = logging.getLogger("mosaic")

_FOUR_PI = 4.0 * 3.141592653589793


def save_assignments(
    assignments: np.ndarray,
    output_path: str | Path,
    precinct_ids: list | None = None,
    id_col_name: str = "precinct_id",
) -> None:
    """
    Save district assignments to CSV.

    Args:
        assignments: Array of district IDs, shape (num_precincts,)
        output_path: Path for output CSV
        precinct_ids: Optional list of precinct identifiers.  If None, uses 0-based indices.
        id_col_name: Column header to use for the ID column in the output CSV.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n = len(assignments)
    if precinct_ids is None:
        log.warning("No precinct IDs provided, using indices")
        precinct_ids = list(range(n))

    # Districts are 0-indexed internally, output as 1-indexed for readability
    df = pd.DataFrame({
        id_col_name: precinct_ids,
        "district": assignments + 1,
    })

    df.to_csv(output_path, index=False)
    log.info(f"Exported {n} assignments to {output_path}")


def save_metrics(
    assignment: np.ndarray,
    output_path: str | Path,
    *,
    populations: np.ndarray,
    ideal_pop: float,
    dem_votes: np.ndarray | None = None,
    gop_votes: np.ndarray | None = None,
    pp_data=None,
) -> None:
    """
    Save per-district metrics to CSV.

    Columns always present: district, population, pop_dev_pct
    With election data: dem_votes, rep_votes, dem_pct, rep_pct
    With pp_data: polsby_popper (0-1, higher = more compact)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_dist = int(assignment.max()) + 1

    pop_d = np.bincount(assignment, weights=populations.astype(np.float64),
                        minlength=n_dist)
    pop_dev_pct = (pop_d - ideal_pop) / ideal_pop * 100.0

    rows: dict = {
        "district": list(range(1, n_dist + 1)),
        "population": pop_d.astype(int).tolist(),
        "pop_dev_pct": [round(v, 2) for v in pop_dev_pct],
    }

    if dem_votes is not None and gop_votes is not None:
        dem_d = np.bincount(assignment, weights=dem_votes.astype(np.float64),
                            minlength=n_dist)
        gop_d = np.bincount(assignment, weights=gop_votes.astype(np.float64),
                            minlength=n_dist)
        total_d = dem_d + gop_d
        dem_pct = np.divide(dem_d, total_d, out=np.full(len(total_d), 0.5), where=total_d > 0) * 100.0
        rep_pct = np.divide(gop_d, total_d, out=np.full(len(total_d), 0.5), where=total_d > 0) * 100.0
        rows["dem_votes"] = dem_d.astype(int).tolist()
        rows["rep_votes"] = gop_d.astype(int).tolist()
        rows["dem_pct"] = [round(v, 2) for v in dem_pct]
        rows["rep_pct"] = [round(v, 2) for v in rep_pct]

    if pp_data is not None:
        dist_area  = np.bincount(assignment, weights=pp_data.areas,
                                  minlength=n_dist).astype(np.float64)
        dist_perim = np.bincount(assignment, weights=pp_data.ext_perimeters,
                                  minlength=n_dist).astype(np.float64)
        eu, ev, elen = pp_data.edge_u, pp_data.edge_v, pp_data.edge_len
        if len(eu) > 0:
            eu_d = assignment[eu]
            ev_d = assignment[ev]
            is_cut = eu_d != ev_d
            if is_cut.any():
                cut_len = elen[is_cut]
                np.add.at(dist_perim, eu_d[is_cut], cut_len)
                np.add.at(dist_perim, ev_d[is_cut], cut_len)
        safe_perim = np.where(dist_perim > 0.0, dist_perim, 1.0)
        pp_per_dist = np.clip(_FOUR_PI * dist_area / (safe_perim ** 2), 0.0, 1.0)
        rows["polsby_popper"] = [round(v, 4) for v in pp_per_dist]

    pd.DataFrame(rows).to_csv(output_path, index=False)
    log.info(f"Exported metrics for {n_dist} districts to {output_path}")
