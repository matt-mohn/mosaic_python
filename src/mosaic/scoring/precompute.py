"""
Pre-computation of geometric data needed for Polsby-Popper scoring and
county-column detection. Called once at shapefile load time.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Optional

import geopandas as gpd
import networkx as nx
import numpy as np

log = logging.getLogger("mosaic")

# Candidate column names for county identifiers in common US precinct shapefiles
_COUNTY_COLS = [
    "CTY",
    "COUNTYFP20", "COUNTYFP", "COUNTY20", "COUNTY",
    "county", "CNTYFP", "CNTYFP20", "CTY_CODE", "CNTYVTD",
    "COUNTYFIPS", "CO_FIPS",
]


@dataclass
class PPData:
    """Precomputed geometry for Polsby-Popper scoring."""
    areas: np.ndarray           # (n,) float64 — precinct areas in CRS units²
    ext_perimeters: np.ndarray  # (n,) float64 — exterior perimeter (not shared with neighbours)
    edge_u: np.ndarray          # (m,) int32 — edge endpoints (global precinct indices)
    edge_v: np.ndarray          # (m,) int32
    edge_len: np.ndarray        # (m,) float64 — shared boundary length


def find_county_array(gdf: gpd.GeoDataFrame) -> Optional[np.ndarray]:
    """
    Scan common column names for a county identifier.
    Returns a contiguous int32 array of county indices (0..K-1) per precinct,
    or None if no county column is found.
    """
    for col in _COUNTY_COLS:
        if col in gdf.columns:
            vals = gdf[col].values
            _, ids = np.unique(vals, return_inverse=True)
            n_counties = ids.max() + 1
            log.info(f"County column '{col}': {n_counties} unique counties")
            return ids.astype(np.int32)
    log.warning(
        "No county column found in shapefile "
        f"(tried: {', '.join(_COUNTY_COLS)}). "
        "County Splits scoring and county-edge bias disabled."
    )
    return None


def precompute_pp_data(gdf: gpd.GeoDataFrame, graph: nx.Graph) -> Optional[PPData]:
    """
    Compute per-precinct areas, exterior perimeters, and shared edge lengths
    from the GeoDataFrame and the already-built adjacency graph.

    Returns None on failure (e.g., missing/invalid geometries).

    Note: PP values are computed in the shapefile's native CRS.  If the CRS
    is geographic (degrees), the absolute PP values will reflect coordinate-space
    shape rather than true geographic compactness, but relative plan rankings
    remain valid.
    """
    try:
        n = len(gdf)
        geoms = gdf.geometry
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*geographic CRS.*", category=UserWarning)
            areas = geoms.area.values.astype(np.float64)
            perimeters = geoms.length.values.astype(np.float64)

        edges = list(graph.edges())
        m = len(edges)

        if m == 0:
            return PPData(
                areas=areas,
                ext_perimeters=perimeters.copy(),
                edge_u=np.empty(0, dtype=np.int32),
                edge_v=np.empty(0, dtype=np.int32),
                edge_len=np.empty(0, dtype=np.float64),
            )

        edge_u = np.array([e[0] for e in edges], dtype=np.int32)
        edge_v = np.array([e[1] for e in edges], dtype=np.int32)
        edge_len = np.zeros(m, dtype=np.float64)

        for i, (u, v) in enumerate(edges):
            try:
                edge_len[i] = geoms[u].intersection(geoms[v]).length
            except Exception:
                edge_len[i] = 0.0

        # Exterior perimeter = total perimeter − all shared boundary lengths
        ext_perimeters = perimeters.copy()
        np.subtract.at(ext_perimeters, edge_u, edge_len)
        np.subtract.at(ext_perimeters, edge_v, edge_len)
        np.maximum(ext_perimeters, 0.0, out=ext_perimeters)

        log.info(f"PP data precomputed: {n} precincts, {m} edges")
        return PPData(
            areas=areas,
            ext_perimeters=ext_perimeters,
            edge_u=edge_u,
            edge_v=edge_v,
            edge_len=edge_len,
        )
    except Exception as exc:
        log.warning(f"PP data precomputation failed: {exc}. Polsby-Popper disabled.")
        return None
