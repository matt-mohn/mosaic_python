"""Strict validation of a shapefile + user column choices before a Mosaic run.

Three checks, each returning a list of human-readable error strings. An empty
list means OK. The first non-empty list halts the load — Mosaic does not
silently proceed on questionable data.

Surface points:
- ``check_geometry`` and ``check_columns`` are called from the column-picker
  dialog at confirm time. Errors appear inline in the dialog.
- ``check_connectivity`` is called from the runner after the adjacency graph
  is built. Errors appear in the red status bar via ``error_message``.

No file-logging side effects; the message strings themselves carry the detail.
"""

from __future__ import annotations

from typing import Iterable

import networkx as nx

from mosaic.io.inspect import ShapefileInspection


_MAX_ROWS_IN_MSG = 10


def _trunc_rows(rows: Iterable[int]) -> str:
    """Render a row-index list capped at ``_MAX_ROWS_IN_MSG`` entries."""
    rows = sorted(rows)
    if len(rows) <= _MAX_ROWS_IN_MSG:
        return ", ".join(str(r) for r in rows)
    head = ", ".join(str(r) for r in rows[:_MAX_ROWS_IN_MSG])
    return f"{head}, ... and {len(rows) - _MAX_ROWS_IN_MSG} more"


def check_geometry(inspection: ShapefileInspection) -> list[str]:
    """Block on geometry conditions that actually break adjacency or rendering.

    Not blocked: ``is_valid == False`` per shapely. Slivers and harmless
    self-intersections are out of scope here.
    """
    issues: list[str] = []
    n = inspection.n_precincts

    if inspection.geometry_null:
        issues.append(
            f"{inspection.geometry_null} of {n} rows have null or empty "
            f"geometry. Mosaic cannot build adjacency for those rows. "
            f"Remove them from the shapefile and reload."
        )
    if inspection.geometry_wrong_type:
        issues.append(
            f"{inspection.geometry_wrong_type} of {n} rows are not "
            f"Polygon / MultiPolygon (likely points or lines). Mosaic "
            f"requires polygonal precincts."
        )
    if inspection.geometry_zero_area:
        issues.append(
            f"{inspection.geometry_zero_area} of {n} polygons have zero "
            f"area. Mosaic cannot redistrict degenerate features."
        )
    return issues


def check_columns(
    inspection: ShapefileInspection,
    *,
    pop_col: str,
    vote_cols: Iterable[tuple[str, str]] = (),
    county_col: str | None = None,
) -> list[str]:
    """Strict checks on the user-selected columns.

    Population is non-negotiable: must exist, be numeric, fully populated
    (no NaN), non-negative everywhere, and sum > 0. Vote columns follow the
    same rules. County column is optional and only sanity-checked for shape.
    """
    issues: list[str] = []

    def _numeric_check(label: str, col: str) -> None:
        info = inspection.column_info.get(col)
        if info is None:
            issues.append(f"{label} column '{col}' is not present in the shapefile.")
            return
        if not info.is_numeric:
            issues.append(
                f"{label} column '{col}' is not numeric (dtype: {info.dtype})."
            )
            return
        if info.n_null > 0:
            issues.append(
                f"{label} column '{col}' has {info.n_null} null / NaN value(s). "
                f"Clean the data in your GIS tool and reload."
            )
        if info.min_value is not None and info.min_value < 0:
            issues.append(
                f"{label} column '{col}' has negative value(s) (min = {info.min_value:g})."
            )
        if info.col_sum is not None and info.col_sum <= 0:
            issues.append(
                f"{label} column '{col}' sums to {info.col_sum:g}; "
                f"expected a positive total."
            )

    if not pop_col:
        issues.append("Population column is required.")
    else:
        _numeric_check("Population", pop_col)

    for dem_col, gop_col in vote_cols:
        if dem_col:
            _numeric_check("DEM vote", dem_col)
        if gop_col:
            _numeric_check("GOP vote", gop_col)

    if county_col:
        info = inspection.column_info.get(county_col)
        if info is None:
            issues.append(f"County column '{county_col}' is not present in the shapefile.")
        elif info.n_unique <= 1:
            issues.append(
                f"County column '{county_col}' has only {info.n_unique} unique "
                f"value(s); county-splits scoring would be meaningless."
            )

    return issues


def check_connectivity(graph: nx.Graph) -> list[str]:
    """Block if the adjacency graph is disconnected.

    ReCom requires a connected graph. Islands / exclaves (HI, AK, barrier
    islands, etc.) are normally reconnected automatically by
    bridge_components, which adds virtual edges during graph construction, so
    a well-formed shapefile reaches here already connected. This check is the
    safety net: it still fires if bridging could not find a valid link (e.g.
    degenerate geometry), pointing the user at the offending rows.
    """
    components = list(nx.connected_components(graph))
    if len(components) <= 1:
        return []

    components.sort(key=len, reverse=True)
    mainland, *smaller = components
    smaller_nodes: list[int] = sorted(n for c in smaller for n in c)

    msg = (
        f"Adjacency graph has {len(components)} disconnected components: "
        f"mainland of {len(mainland):,} precincts plus {len(smaller)} smaller "
        f"group(s) totaling {len(smaller_nodes):,} precincts "
        f"(rows: {_trunc_rows(smaller_nodes)}). "
        f"Remove those rows from the shapefile and reload."
    )
    return [msg]
