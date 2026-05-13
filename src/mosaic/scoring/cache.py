"""Disk cache for precomputed Polsby-Popper geometry data.

The expensive piece of ``precompute_pp_data`` is the per-edge shapely
``intersection().length`` loop — many seconds on a ~10k+ precinct shapefile.
Since PPData is a pure function of the shapefile geometry plus the adjacency
graph (both stable for a given shapefile), it's a clean caching target.

Layout: sidecar to the existing graph cache, e.g.
    cache/North_Carolina_Simplified.pkl       <- graph
    cache/North_Carolina_Simplified.pp.pkl    <- PPData

The two caches are independent so a corrupt or stale PP cache never breaks
graph loading; we just rebuild PP on the next run.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional

from mosaic.scoring.precompute import PPData

log = logging.getLogger("mosaic")


def get_pp_cache_path(shapefile_path: str | Path, cache_dir: str | Path = "cache") -> Path:
    """Sidecar path next to the graph cache: ``cache/<stem>.pp.pkl``."""
    name = Path(shapefile_path).stem
    return Path(cache_dir) / f"{name}.pp.pkl"


def save_cached_pp_data(pp_data: PPData, cache_path: str | Path) -> None:
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "areas": pp_data.areas,
        "ext_perimeters": pp_data.ext_perimeters,
        "edge_u": pp_data.edge_u,
        "edge_v": pp_data.edge_v,
        "edge_len": pp_data.edge_len,
    }
    with open(cache_path, "wb") as f:
        pickle.dump(payload, f)


def load_cached_pp_data(
    cache_path: str | Path,
    n_precincts: int,
    n_edges: int,
) -> Optional[PPData]:
    """Load cached PPData if it exists and matches the current graph/gdf.

    Returns None on any of: missing file, unreadable file, or size mismatch
    (precinct count or edge count differs from the live graph). A mismatch
    means the shapefile changed underneath us — better to recompute than
    silently use stale geometry.
    """
    cache_path = Path(cache_path)
    if not cache_path.exists():
        return None
    try:
        with open(cache_path, "rb") as f:
            payload = pickle.load(f)
        pp = PPData(**payload)
    except Exception as exc:
        log.warning(f"PP cache unreadable at {cache_path}: {exc}. Recomputing.")
        return None

    if len(pp.areas) != n_precincts or len(pp.edge_len) != n_edges:
        log.info(
            f"PP cache size mismatch (cached {len(pp.areas)} precincts / "
            f"{len(pp.edge_len)} edges vs live {n_precincts}/{n_edges}). "
            "Recomputing."
        )
        return None

    return pp
