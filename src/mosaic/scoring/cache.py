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

from mosaic.io.shapefile import shapefile_fingerprint
from mosaic.paths import cache_dir as _default_cache_dir
from mosaic.scoring.precompute import PPData

log = logging.getLogger("mosaic")


def get_pp_cache_path(
    shapefile_path: str | Path,
    cache_dir: str | Path | None = None,
) -> Path:
    """Sidecar path next to the graph cache: ``cache/<stem>.pp.pkl``."""
    cdir = Path(cache_dir) if cache_dir is not None else _default_cache_dir()
    name = Path(shapefile_path).stem
    return cdir / f"{name}.pp.pkl"


def save_cached_pp_data(
    pp_data: PPData,
    cache_path: str | Path,
    shapefile_path: str | Path,
) -> None:
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "fingerprint": shapefile_fingerprint(shapefile_path),
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
    shapefile_path: str | Path,
    n_precincts: int,
    n_edges: int,
) -> Optional[PPData]:
    """Load cached PPData iff its fingerprint matches the live shapefile.

    Returns None on: missing file, unreadable file, fingerprint mismatch, or
    size mismatch (precinct/edge count differs from the live graph — defense
    in depth in case the same shapefile produced a different graph).
    """
    cache_path = Path(cache_path)
    if not cache_path.exists():
        return None
    try:
        with open(cache_path, "rb") as f:
            payload = pickle.load(f)
    except Exception as exc:
        log.warning(f"PP cache unreadable at {cache_path}: {exc}. Recomputing.")
        return None

    live_fp = shapefile_fingerprint(shapefile_path)
    if not live_fp or payload.get("fingerprint") != live_fp:
        log.info(
            f"PP cache stale for {Path(shapefile_path).name} "
            f"(fingerprint mismatch). Recomputing."
        )
        return None

    pp_kwargs = {k: payload[k] for k in ("areas", "ext_perimeters", "edge_u", "edge_v", "edge_len")}
    pp = PPData(**pp_kwargs)

    if len(pp.areas) != n_precincts or len(pp.edge_len) != n_edges:
        log.warning(
            f"PP cache size mismatch (cached {len(pp.areas)} precincts / "
            f"{len(pp.edge_len)} edges vs live {n_precincts}/{n_edges}). "
            "Recomputing."
        )
        return None

    return pp
