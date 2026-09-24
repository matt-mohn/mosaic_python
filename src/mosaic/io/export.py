"""Export utilities for redistricting results."""

import logging
from collections.abc import Sequence
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

log = logging.getLogger("mosaic")

_FOUR_PI = 4.0 * 3.141592653589793
_GROUPS = ("black", "latino", "asian")


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
    vap_data: dict | None = None,
    race_groups: Sequence[str] | None = None,
    opportunity_midpoint: float = 0.44,
    opportunity_steepness: float = 0.05,
    opportunity_solid: float = 0.55,
) -> None:
    """Save per-district metrics to CSV.

    Columns:
      Always:           district, precincts, population, pop_dev_pct,
                        pop_dev_pct_abs, pop_dev_people
      With pp_data:     cut_edges, polsby_popper, area, perimeter
      With county_ids:  counties_touched, counties_whole
      With elections:   dem_votes, rep_votes, total_votes, dem_pct, rep_pct,
                        dem_margin, win_prob
      With reock_data:  reock
      With pp+reock:    holistic_compactness
      With vap_data:    demographic_total, and per group in race_groups,
                        <group>_pop, <group>_pct, <group>_opportunity

    ``race_groups`` names the groups the user actually gave a column for;
    unselected groups are zero-filled upstream, so without it a group with no
    data is indistinguishable from a group with no people. Defaults to every
    scored group.

    Area uses squared native CRS units; perimeter uses native CRS units.
    Geographic coordinates therefore yield square degrees and degrees.
    Opportunity is clipped credit (P / solid-share reference), not raw probability.
    All requested metrics must succeed before the destination file is replaced.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    assignment = np.asarray(assignment)
    if (assignment.ndim != 1 or assignment.size == 0
            or not np.issubdtype(assignment.dtype, np.integer)
            or np.any(assignment < 0)):
        raise ValueError("District assignments must be a nonempty array of nonnegative integers")
    populations = np.asarray(populations, dtype=np.float64)
    if (populations.shape != assignment.shape or not np.all(np.isfinite(populations))
            or np.any(populations < 0)):
        raise ValueError("Population must contain one finite, nonnegative value per precinct")
    if not np.isfinite(ideal_pop) or ideal_pop <= 0:
        raise ValueError("Ideal population must be positive and finite")
    if (dem_votes is None) != (gop_votes is None):
        raise ValueError("Both Democratic and Republican votes are required")
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
                # Island bridges have no shared boundary and do not count as
                # geographic cut edges. They add zero to the perimeter too.
                real_cut = cut_len > 0
                np.add.at(cut_d, cut_du[real_cut], 1)
                np.add.at(cut_d, cut_dv[real_cut], 1)
                np.add.at(perim_d, cut_du, cut_len)
                np.add.at(perim_d, cut_dv, cut_len)
        safe_p = np.where(perim_d > 0.0, perim_d, 1.0)
        pp_per_dist = np.clip(_FOUR_PI * area_d / (safe_p ** 2), 0.0, 1.0)

    # ── County congruence ─────────────────────────────────────────────────────
    counties_d: np.ndarray | None = None
    counties_whole_d: np.ndarray | None = None
    if county_ids is not None:
        county_ids_a = np.asarray(county_ids, dtype=np.int64)
        pairs = np.column_stack([assignment.astype(np.int64), county_ids_a])
        up = np.unique(pairs, axis=0)
        counties_d = np.bincount(up[:, 0], minlength=n_dist)
        # A county touching exactly one district sits whole inside it.
        per_county = np.bincount(up[:, 1])
        whole = up[per_county[up[:, 1]] == 1]
        counties_whole_d = np.bincount(whole[:, 0], minlength=n_dist)

    # ── Partisan ──────────────────────────────────────────────────────────────
    dem_pct_arr: np.ndarray | None = None
    partisan_rows: dict = {}
    if dem_votes is not None and gop_votes is not None:
        dem_votes = np.asarray(dem_votes, dtype=np.float64)
        gop_votes = np.asarray(gop_votes, dtype=np.float64)
        for values in (dem_votes, gop_votes):
            if (values.shape != assignment.shape or not np.all(np.isfinite(values))
                    or np.any(values < 0)):
                raise ValueError("Votes must contain one finite, nonnegative value per precinct")
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
        partisan_rows["dem_votes"]   = dem_d.astype(int).tolist()
        partisan_rows["rep_votes"]   = gop_d.astype(int).tolist()
        partisan_rows["total_votes"] = total_d.astype(int).tolist()
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
        except Exception as exc:
            raise ValueError(f"Win-probability calculation failed: {exc}") from exc

    # ── Reock ─────────────────────────────────────────────────────────────────
    reock_per_dist: np.ndarray | None = None
    if reock_data is not None:
        try:
            from mosaic.scoring.reock import reock_per_district
            reock_per_dist = np.asarray(reock_per_district(assignment, reock_data, n_dist))
            if reock_per_dist.shape != (n_dist,):
                raise ValueError("Expected one value per district")
        except Exception as exc:
            raise ValueError(f"Reock calculation failed: {exc}") from exc

    # ── Demographics ──────────────────────────────────────────────────────────
    demographic_rows: dict = {}
    if vap_data is not None:
        groups = list(race_groups) if race_groups is not None else list(_GROUPS)
        if not set(groups).issubset(_GROUPS):
            raise ValueError("Unknown demographic group in metrics export")
        total_v = np.asarray(vap_data["total"], dtype=np.float64)
        if (total_v.shape != assignment.shape or not np.all(np.isfinite(total_v))
                or np.any(total_v < 0)):
            raise ValueError("Demographic total must contain one finite, "
                             "nonnegative value per precinct")
        total_vd = np.bincount(assignment, weights=total_v, minlength=n_dist)
        demographic_rows["demographic_total"] = total_vd.astype(int).tolist()

        try:
            from mosaic.scoring.opportunity import district_opportunity_credit
            credit = district_opportunity_credit(
                assignment, vap_data, n_dist,
                midpoint=opportunity_midpoint,
                steepness=opportunity_steepness,
                solid=opportunity_solid,
                groups=groups,
            ) if groups else {}
        except Exception as exc:
            raise ValueError(f"Opportunity calculation failed: {exc}") from exc

        for g in groups:
            grp_d = np.bincount(assignment,
                                weights=np.asarray(vap_data[g], dtype=np.float64),
                                minlength=n_dist)
            pct = np.divide(grp_d, total_vd, out=np.zeros(n_dist),
                            where=total_vd > 0) * 100.0
            demographic_rows[f"{g}_pop"] = grp_d.astype(int).tolist()
            demographic_rows[f"{g}_pct"] = [round(float(v), 2) for v in pct]
            demographic_rows[f"{g}_opportunity"] = [
                round(float(v), 4) for v in credit[g]
            ]

    # ── Assemble rows in column order ─────────────────────────────────────────
    rows: dict = {
        "district":        list(range(1, n_dist + 1)),
        "precincts":       np.bincount(assignment, minlength=n_dist).tolist(),
        "population":      pop_d.astype(int).tolist(),
        "pop_dev_pct":     [round(float(v), 2) for v in pop_dev_pct],
        "pop_dev_pct_abs": [round(abs(float(v)), 2) for v in pop_dev_pct],
        "pop_dev_people":  [int(round(float(v))) for v in (pop_d - ideal_pop)],
    }
    if cut_d is not None:
        rows["cut_edges"] = cut_d.tolist()
    if counties_d is not None:
        rows["counties_touched"] = counties_d.astype(int).tolist()
    if counties_whole_d is not None:
        rows["counties_whole"] = counties_whole_d.astype(int).tolist()
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
    if pp_data is not None:
        rows["area"] = [round(float(v), 4) for v in area_d]
        rows["perimeter"] = [round(float(v), 4) for v in perim_d]
    rows.update(demographic_rows)

    for column, values in rows.items():
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Nonfinite metric in '{column}'")
    # Write beside the destination so replacement is atomic on its filesystem.
    # A failed calculation or write leaves an existing export intact.
    temporary = output_path.with_name(f".mosaic-{uuid4().hex}.tmp")
    try:
        pd.DataFrame(rows).to_csv(temporary, index=False)
        temporary.replace(output_path)
    finally:
        temporary.unlink(missing_ok=True)
    log.info(f"Exported metrics for {n_dist} districts to {output_path}")
