"""
Hot-start: load a precinct -> district assignment from a CSV and validate it
against the currently-loaded shapefile + run parameters.

Expected CSV format (matches save_assignments() output):
    <id_col>, district
    <id>,    1
    <id>,    2
    ...

district values are 1-indexed in the file, converted to 0-indexed internally.
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd

log = logging.getLogger("mosaic")


class HotStartError(ValueError):
    """Raised when the uploaded assignment fails validation."""


def read_csv_columns(path: str | Path) -> list[str]:
    """Return the CSV's header row so the caller can offer a column picker."""
    path = Path(path)
    if not path.is_file():
        raise HotStartError(f"File not found: {path}")
    try:
        df = pd.read_csv(path, nrows=0)
    except Exception as e:
        raise HotStartError(f"Could not read CSV header: {e}") from e
    return list(df.columns)


def load_hot_start(
    path: str | Path,
    *,
    gdf: gpd.GeoDataFrame,
    gdf_id_col: str,
    csv_id_col: str,
    csv_district_col: str,
    populations: np.ndarray,
    graph: nx.Graph,
    num_districts: int,
    tolerance: float,
) -> tuple[np.ndarray, dict]:
    """Parse a hot-start CSV and validate it against the loaded plan parameters.

    The CSV's id column is matched against gdf[gdf_id_col] by value (as strings),
    so the CSV header doesn't need to match the shapefile's column name.

    Returns (assignment, info) on success, where:
        assignment -- (n,) int32 zero-indexed district array aligned to gdf rows
        info       -- {
            "filename":     str,
            "n_districts":  int,
            "max_dev_pct":  float,   # max(|pop - ideal| / ideal) * 100
            "max_dev_idx":  int,     # district index (0-based) with largest dev
            "csv_id_col":      str,
            "csv_district_col": str,
        }

    Raises HotStartError with a user-facing message on any validation failure.
    """
    path = Path(path)
    if not path.is_file():
        raise HotStartError(f"File not found: {path}")

    # Read the ID column as string so leading zeros survive. Pandas otherwise
    # auto-infers int for all-digit columns (e.g. GEOID '08001001001' becomes
    # 8001001001), which silently breaks the match against the shapefile.
    try:
        df = pd.read_csv(path, dtype={csv_id_col: str})
    except Exception as e:
        raise HotStartError(f"Could not read CSV: {e}") from e

    if csv_id_col not in df.columns:
        raise HotStartError(
            f"CSV does not have the selected ID column '{csv_id_col}'. "
            f"Available: {list(df.columns)}"
        )
    if csv_district_col not in df.columns:
        raise HotStartError(
            f"CSV does not have the selected district column "
            f"'{csv_district_col}'. Available: {list(df.columns)}"
        )

    n = len(gdf)
    if len(df) != n:
        raise HotStartError(
            f"CSV has {len(df)} rows; shapefile has {n} precincts. "
            f"Row count must match exactly."
        )

    gdf_ids = gdf[gdf_id_col].astype(str).str.strip().values
    csv_ids = df[csv_id_col].astype(str).str.strip().values

    if set(gdf_ids) != set(csv_ids):
        missing = set(gdf_ids) - set(csv_ids)
        extra   = set(csv_ids) - set(gdf_ids)
        bits = []
        if missing:
            bits.append(
                f"shapefile has {len(missing)} IDs not in CSV "
                f"(e.g. {sorted(missing)[:3]})"
            )
        if extra:
            bits.append(
                f"CSV has {len(extra)} IDs not in shapefile "
                f"(e.g. {sorted(extra)[:3]})"
            )
        bits.append(
            f"sample shapefile IDs: {sorted(set(gdf_ids))[:3]}; "
            f"sample CSV IDs: {sorted(set(csv_ids))[:3]}"
        )
        raise HotStartError(
            "Precinct IDs in CSV do not match shapefile: " + "; ".join(bits)
        )

    # Reorder CSV rows to match gdf order so the assignment array is correctly aligned.
    # Match on the same stripped-string representation used for the set-equality check above.
    order_map = {pid: i for i, pid in enumerate(gdf_ids)}
    df = df.assign(
        _pos=df[csv_id_col].astype(str).str.strip().map(order_map)
    ).sort_values("_pos")
    raw_districts = df[csv_district_col].values

    if not np.issubdtype(raw_districts.dtype, np.number):
        try:
            raw_districts = raw_districts.astype(np.int64)
        except (ValueError, TypeError) as e:
            raise HotStartError(f"'district' column is not numeric: {e}") from e

    if np.isnan(raw_districts.astype(np.float64)).any():
        raise HotStartError("'district' column contains NaN / missing values")

    raw_districts = raw_districts.astype(np.int64)
    unique_d = np.unique(raw_districts)
    n_dist_csv = len(unique_d)

    if n_dist_csv != num_districts:
        raise HotStartError(
            f"CSV defines {n_dist_csv} districts; current run is set to "
            f"{num_districts}. Adjust the Districts slider or fix the CSV."
        )

    # Accept either 1-indexed (export convention) or 0-indexed input.
    lo, hi = int(unique_d.min()), int(unique_d.max())
    if lo == 1 and hi == num_districts:
        assignment = (raw_districts - 1).astype(np.int32)
    elif lo == 0 and hi == num_districts - 1:
        assignment = raw_districts.astype(np.int32)
    else:
        raise HotStartError(
            f"District labels should be 1..{num_districts} (or 0..{num_districts-1}). "
            f"Got min={lo}, max={hi}."
        )

    # Contiguity: every district's induced subgraph must be connected
    nodes_arr = np.array(list(graph.nodes()))
    for d in range(num_districts):
        members = nodes_arr[assignment == d]
        if len(members) == 0:
            raise HotStartError(f"District {d + 1} is empty")
        sub = graph.subgraph(members)
        if not nx.is_connected(sub):
            n_components = nx.number_connected_components(sub)
            raise HotStartError(
                f"District {d + 1} is not contiguous "
                f"({n_components} disconnected components)"
            )

    # Population deviation
    pop_f = populations.astype(np.float64)
    ideal_pop = pop_f.sum() / num_districts
    if ideal_pop <= 0:
        raise HotStartError("Total population is zero; cannot compute deviation")

    dist_pop = np.bincount(assignment, weights=pop_f, minlength=num_districts)
    dev = (dist_pop - ideal_pop) / ideal_pop
    abs_dev = np.abs(dev)
    max_dev_idx = int(np.argmax(abs_dev))
    max_dev = float(abs_dev[max_dev_idx])

    if max_dev > tolerance:
        raise HotStartError(
            f"District {max_dev_idx + 1} population deviation is "
            f"{max_dev * 100:.2f}%, exceeding tolerance "
            f"{tolerance * 100:.2f}%. Tighten the CSV or relax tolerance."
        )

    log.info(
        f"Hot start loaded: {n} precincts, {num_districts} districts, "
        f"max pop dev {max_dev * 100:.2f}% (district {max_dev_idx + 1})"
    )

    return assignment, {
        "filename":         path.name,
        "n_districts":      num_districts,
        "max_dev_pct":      max_dev * 100.0,
        "max_dev_idx":      max_dev_idx,
        "csv_id_col":       csv_id_col,
        "csv_district_col": csv_district_col,
    }
