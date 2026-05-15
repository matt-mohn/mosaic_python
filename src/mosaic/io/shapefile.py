"""Shapefile loading utilities."""

import hashlib
import geopandas as gpd
import pandas as pd
from pathlib import Path


def load_shapefile(path: str | Path) -> gpd.GeoDataFrame:
    """
    Load a shapefile and prepare it for redistricting.

    Args:
        path: Path to .shp file

    Returns:
        GeoDataFrame with geometry and population columns.
        Index is reset to 0..N-1 for use as precinct IDs.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Shapefile not found: {path}")

    gdf = gpd.read_file(path)
    gdf = gdf.reset_index(drop=True)

    # Try to identify population column
    pop_candidates = ["population", "pop", "total_pop", "totpop", "POP100", "P0010001"]
    pop_col = None
    for col in pop_candidates:
        if col in gdf.columns or col.lower() in [c.lower() for c in gdf.columns]:
            # Find actual column name (case-insensitive match)
            for c in gdf.columns:
                if c.lower() == col.lower():
                    pop_col = c
                    break
            break

    if pop_col and pop_col != "population":
        gdf = gdf.rename(columns={pop_col: "population"})

    if "population" not in gdf.columns:
        raise ValueError(
            f"Could not find population column. "
            f"Available columns: {list(gdf.columns)}"
        )

    return gdf


def shapefile_fingerprint(shapefile_path: str | Path) -> dict:
    """SHA-256 + size of the .shp and .dbf bytes.

    Used by the graph and PP caches to detect that a shapefile's *contents*
    changed even when its filename did not. Returns an empty dict if either
    sidecar is missing — callers treat that as "do not cache."
    """
    p = Path(shapefile_path)
    out: dict = {}
    for label, path in (("shp", p), ("dbf", p.with_suffix(".dbf"))):
        if not path.exists():
            return {}
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        out[label] = h.hexdigest()
        out[f"{label}_size"] = path.stat().st_size
    return out
