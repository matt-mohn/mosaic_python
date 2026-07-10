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
    """Save district assignments to CSV.

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


# Band limits for per-district Holistic Compactness normalization
_HC_PP_LO, _HC_PP_HI = 0.10, 0.50
_HC_RK_LO, _HC_RK_HI = 0.25, 0.50


def save_metrics(
    assignment: np.ndarray,
    output_path: str | Path,
    *,
    populations: np.ndarray,
    ideal_pop: float,
    dem_votes: np.ndarray | None = None,
    gop_votes: np.ndarray | None = None,
    pp_data=None,
    reock_data=None,
    county_ids: np.ndarray | None = None,
    win_prob_at_55: float = 0.9,
    swing_sigma: float = 0.03,
) -> None:
    """Save per-district metrics to CSV.

    Columns:
      Always:           district, population, pop_dev_pct, pop_dev_pct_abs
      With pp_data:     cut_edges, polsby_popper
      With county_ids:  counties_touched
      With elections:   dem_votes, rep_votes, dem_pct, rep_pct, dem_margin, win_prob
      With reock_data:  reock
      With pp+reock:    holistic_compactness
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_dist = int(assignment.max()) + 1

    # ── Population ────────────────────────────────────────────────────────────
    pop_d = np.bincount(assignment, weights=populations.astype(np.float64),
                        minlength=n_dist)
    pop_dev_pct = (pop_d - ideal_pop) / ideal_pop * 100.0

    # ── Edge-derived metrics (cut_edges + Polsby-Popper share edges) ──────────
    cut_d: np.ndarray | None = None
    pp_per_dist: np.ndarray | None = None
    if pp_data is not None:
        eu = pp_data.edge_u
        ev = pp_data.edge_v
        elen = pp_data.edge_len

        cut_d = np.zeros(n_dist, dtype=np.int64)
        area_d = np.bincount(assignment, weights=pp_data.areas,
                             minlength=n_dist).astype(np.float64)
        perim_d = np.bincount(assignment, weights=pp_data.ext_perimeters,
                              minlength=n_dist).astype(np.float64)
        if len(eu) > 0:
            du = assignment[eu]
            dv = assignment[ev]
            is_cut = du != dv
            if is_cut.any():
                cut_du = du[is_cut]
                cut_dv = dv[is_cut]
                cut_len = elen[is_cut]
                np.add.at(cut_d, cut_du, 1)
                np.add.at(cut_d, cut_dv, 1)
                np.add.at(perim_d, cut_du, cut_len)
                np.add.at(perim_d, cut_dv, cut_len)
        safe_p = np.where(perim_d > 0.0, perim_d, 1.0)
        pp_per_dist = np.clip(_FOUR_PI * area_d / (safe_p ** 2), 0.0, 1.0)

    # ── County congruence ─────────────────────────────────────────────────────
    counties_d: np.ndarray | None = None
    if county_ids is not None:
        county_ids_a = np.asarray(county_ids, dtype=np.int64)
        pairs = np.column_stack([assignment.astype(np.int64), county_ids_a])
        up = np.unique(pairs, axis=0)
        counties_d = np.bincount(up[:, 0], minlength=n_dist)

    # ── Partisan ──────────────────────────────────────────────────────────────
    dem_pct_arr: np.ndarray | None = None
    partisan_rows: dict = {}
    if dem_votes is not None and gop_votes is not None:
        dem_d = np.bincount(assignment, weights=dem_votes.astype(np.float64),
                            minlength=n_dist)
        gop_d = np.bincount(assignment, weights=gop_votes.astype(np.float64),
                            minlength=n_dist)
        total_d = dem_d + gop_d
        dem_pct_arr = (np.divide(dem_d, total_d,
                                 out=np.full(n_dist, 0.5),
                                 where=total_d > 0) * 100.0)
        rep_pct_arr = (np.divide(gop_d, total_d,
                                 out=np.full(n_dist, 0.5),
                                 where=total_d > 0) * 100.0)
        partisan_rows["dem_votes"]  = dem_d.astype(int).tolist()
        partisan_rows["rep_votes"]  = gop_d.astype(int).tolist()
        partisan_rows["dem_pct"]    = [round(float(v), 2) for v in dem_pct_arr]
        partisan_rows["rep_pct"]    = [round(float(v), 2) for v in rep_pct_arr]
        partisan_rows["dem_margin"] = [round(float(v - 50.0), 2) for v in dem_pct_arr]
        try:
            from scipy.special import ndtr, ndtri
            p = float(np.clip(win_prob_at_55, 0.501, 0.9999))
            sigma_d = 0.05 / float(ndtri(p))
            sigma_c = float(np.sqrt(swing_sigma ** 2 + sigma_d ** 2))
            partisan_rows["win_prob"] = [
                round(float(ndtr((v / 100.0 - 0.5) / sigma_c)), 4)
                for v in dem_pct_arr
            ]
        except Exception:
            pass

    # ── Reock ─────────────────────────────────────────────────────────────────
    reock_per_dist: np.ndarray | None = None
    if reock_data is not None:
        try:
            from mosaic.scoring.reock import reock_per_district
            reock_per_dist = reock_per_district(assignment, reock_data, n_dist)
        except Exception:
            pass

    # ── Assemble rows in column order ─────────────────────────────────────────
    rows: dict = {
        "district":        list(range(1, n_dist + 1)),
        "population":      pop_d.astype(int).tolist(),
        "pop_dev_pct":     [round(float(v), 2) for v in pop_dev_pct],
        "pop_dev_pct_abs": [round(abs(float(v)), 2) for v in pop_dev_pct],
    }
    if cut_d is not None:
        rows["cut_edges"] = cut_d.tolist()
    if counties_d is not None:
        rows["counties_touched"] = counties_d.astype(int).tolist()
    rows.update(partisan_rows)
    if pp_per_dist is not None:
        rows["polsby_popper"] = [round(float(v), 4) for v in pp_per_dist]
    if reock_per_dist is not None:
        rows["reock"] = [round(float(v), 4) for v in reock_per_dist]
    if pp_per_dist is not None and reock_per_dist is not None:
        pp_c = ((np.clip(pp_per_dist, _HC_PP_LO, _HC_PP_HI) - _HC_PP_LO)
                / (_HC_PP_HI - _HC_PP_LO))
        rk_c = ((np.clip(reock_per_dist, _HC_RK_LO, _HC_RK_HI) - _HC_RK_LO)
                / (_HC_RK_HI - _HC_RK_LO))
        rows["holistic_compactness"] = [round(float(v), 4) for v in (pp_c + rk_c) / 2.0]

    pd.DataFrame(rows).to_csv(output_path, index=False)
    log.info(f"Exported metrics for {n_dist} districts to {output_path}")
