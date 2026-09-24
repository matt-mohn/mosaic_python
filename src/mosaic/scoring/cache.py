"""Disk cache for precomputed Polsby-Popper geometry data.

``precompute_pp_data`` reuses intersection lengths stored on real graph edges
and calculates them for graphs without that metadata. Its disk cache avoids
reassembling the geometry arrays. Reuse requires matching source geometry and
graph edge endpoints.

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

import numpy as np

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
    """Pickle PPData alongside a fingerprint of its source.

    Best-effort: a cache write failure must never fail the caller.
    """
    cache_path = Path(cache_path)
    payload = {
        "fingerprint": shapefile_fingerprint(shapefile_path),
        "areas": pp_data.areas,
        "ext_perimeters": pp_data.ext_perimeters,
        "edge_u": pp_data.edge_u,
        "edge_v": pp_data.edge_v,
        "edge_len": pp_data.edge_len,
    }
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(payload, f)
    except Exception as exc:
        log.warning(f"Could not write PP cache to {cache_path}: {exc}.")


def load_cached_pp_data(
    cache_path: str | Path,
    shapefile_path: str | Path,
    n_precincts: int,
    n_edges: int,
    *,
    edges=None,
) -> Optional[PPData]:
    """Load cached PPData iff its fingerprint matches the live shapefile.

    Returns None on: missing file, unreadable file, fingerprint mismatch, or
    size mismatch. When live edges are supplied, their endpoints and order must
    match too: changing island bridges can leave the edge count unchanged.
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
    if not isinstance(payload, dict) or not live_fp or payload.get("fingerprint") != live_fp:
        log.info(
            f"PP cache stale for {Path(shapefile_path).name} "
            f"(fingerprint mismatch). Recomputing."
        )
        return None

    try:
        pp_kwargs = {k: np.asarray(payload[k])
                     for k in ("areas", "ext_perimeters", "edge_u", "edge_v", "edge_len")}
        pp = PPData(**pp_kwargs)
        for name, expected in (("areas", n_precincts), ("ext_perimeters", n_precincts),
                               ("edge_u", n_edges), ("edge_v", n_edges), ("edge_len", n_edges)):
            values = getattr(pp, name)
            if values.shape != (expected,) or not np.all(np.isfinite(values)):
                raise ValueError(f"invalid {name}")
        for name in ("edge_u", "edge_v"):
            endpoints = getattr(pp, name)
            if (not np.issubdtype(endpoints.dtype, np.integer)
                    or np.any(endpoints < 0) or np.any(endpoints >= n_precincts)):
                raise ValueError(f"invalid {name} indices")
        if edges is not None:
            live_edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
            if (not np.array_equal(pp.edge_u, live_edges[:, 0])
                    or not np.array_equal(pp.edge_v, live_edges[:, 1])):
                raise ValueError("graph edge endpoints changed")
    except (KeyError, TypeError, ValueError) as exc:
        log.info("PP cache invalid at %s: %s. Recomputing.", cache_path, exc)
        return None

    return pp
