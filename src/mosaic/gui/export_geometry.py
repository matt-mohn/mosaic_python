"""Stateless geometry helpers for vector map exports; nothing survives a save."""

import geopandas as gpd
import numpy as np
import shapely
from matplotlib.path import Path
from shapely.errors import ShapelyError


def polygon_path(geom):
    """Preserve the existing polygon/ring order when building a vector path."""
    if geom is None:
        return None
    if geom.geom_type == "Polygon":
        polys = [geom]
    elif geom.geom_type == "MultiPolygon":
        polys = geom.geoms
    else:
        return None
    verts, codes = [], []
    for poly in polys:
        for ring in (poly.exterior, *poly.interiors):
            coords = np.asarray(ring.coords)
            if len(coords) < 3:
                continue
            verts.append(coords)
            code = np.full(len(coords), Path.LINETO, dtype=np.uint8)
            code[0], code[-1] = Path.MOVETO, Path.CLOSEPOLY
            codes.append(code)
    if not verts:
        return None
    return Path(np.concatenate(verts), np.concatenate(codes), readonly=True)


def is_valid_coverage(geometries):
    """Check every export; older Shapely/GEOS versions use the general union."""
    validate = getattr(shapely, "coverage_is_valid", None)
    if validate is not None:
        try:
            return bool(
                np.isin(shapely.get_type_id(geometries), [3, 6]).all()
                and shapely.is_valid(geometries).all()
                and validate(geometries)
            )
        except (ShapelyError, NotImplementedError):
            pass
    return False


def union_geometry(geometries, coverage=False):
    """Only use the coverage algorithm after validating the source polygons."""
    if coverage:
        try:
            return shapely.coverage_union_all(geometries)
        except (ShapelyError, NotImplementedError):
            pass
    return shapely.union_all(geometries)


def grouped_geometry(geometry, labels, coverage=False):
    """Merge the current assignment, retaining positional precinct alignment."""
    labels = np.asarray(labels)
    if labels.shape != (len(geometry),):
        raise ValueError("Export assignment does not match loaded geometry")
    keys = np.unique(labels)
    geoms = gpd.GeoSeries(
        [union_geometry(geometry.array[labels == key], coverage) for key in keys],
        index=keys, crs=geometry.crs,
    )
    paths = {key: polygon_path(geom) for key, geom in geoms.items()}
    return geoms, paths
