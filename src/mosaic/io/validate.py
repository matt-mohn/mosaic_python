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


SCORED_GROUPS: tuple[tuple[str, str], ...] = (
    ("black", "Black"), ("latino", "Hispanic"), ("asian", "Asian"),
)

# Any population basis is valid -- voting-age, citizen voting-age, or
# total population, whichever the file carries -- so the ratio to total
# population is not checked from above. Far below it, the column is not a head
# count at all: a share encoded 0-1, a subgroup, or a count in thousands.
_TOTAL_MIN_SHARE = 0.25
# Named groups overlap (Hispanic is an ethnicity), so they can sum slightly
# past the total legitimately; well past it means the wrong column.
_NAMED_OVER_TOTAL = 1.25


def check_demographics(
    inspection: ShapefileInspection,
    demographics: dict | None,
    *,
    pop_col: str = "",
    vote_cols: Iterable[tuple[str, str]] = (),
) -> tuple[list[str], list[str]]:
    """Validate the demographic columns. Returns (errors, warnings).

    Errors block the load on the same terms as ``check_columns``. Warnings are
    shown but proceed: a usable-but-partial selection still scores, and saying
    so beats silently dropping groups the user believes are being scored.
    """
    errors: list[str] = []
    warnings: list[str] = []
    if not demographics:
        return errors, warnings

    total_col = demographics.get("total")
    scored = [(g, lbl) for g, lbl in SCORED_GROUPS if demographics.get(g)]
    if not total_col:
        errors.append("Demographics: a Total column is required.")
        return errors, warnings
    if not scored:
        errors.append(
            "Demographics: select at least one scored group (Black, Hispanic, "
            "Asian).")
        return errors, warnings

    def _numeric(label: str, col: str) -> bool:
        info = inspection.column_info.get(col)
        if info is None:
            errors.append(f"{label} column '{col}' is not present in the shapefile.")
            return False
        if not info.is_numeric:
            errors.append(f"{label} column '{col}' is not numeric (dtype: {info.dtype}).")
            return False
        if info.min_value is not None and info.min_value < 0:
            errors.append(
                f"{label} column '{col}' has negative value(s) "
                f"(min = {info.min_value:g}).")
        if info.n_null:
            warnings.append(
                f"{label} column '{col}' has {info.n_null} null value(s); "
                f"they are read as 0.")
        return True

    if not _numeric("Demographic total", total_col):
        return errors, warnings
    tot_info = inspection.column_info[total_col]
    tot_sum = float(tot_info.col_sum or 0.0)
    if tot_sum <= 0:
        errors.append(
            f"Demographic total column '{total_col}' sums to {tot_sum:g}; "
            f"expected a positive total.")
        return errors, warnings

    named_sum = 0.0
    for group, label in scored:
        col = demographics[group]
        if not _numeric(f"{label}", col):
            continue
        info = inspection.column_info[col]
        col_sum = float(info.col_sum or 0.0)
        named_sum += col_sum
        if col_sum > tot_sum:
            warnings.append(
                f"{label} ({col}) sums to more than the race total "
                f"({total_col}); check the column mapping.")
        elif col_sum <= 0:
            warnings.append(
                f"{label} ({col}) sums to 0, so that group is never scored.")

    if named_sum > tot_sum * _NAMED_OVER_TOTAL:
        warnings.append(
            f"The selected races sum to {named_sum / tot_sum:.0%} of "
            f"'{total_col}'; check that the total is the right column.")

    pop_info = inspection.column_info.get(pop_col) if pop_col else None
    pop_sum = float(pop_info.col_sum or 0.0) if pop_info else 0.0
    if pop_sum > 0 and tot_sum / pop_sum < _TOTAL_MIN_SHARE:
        warnings.append(
            f"Demographic total '{total_col}' is only {tot_sum / pop_sum:.0%} of total "
            f"population; check it is a head count and not a share.")

    # A column doing double duty as votes or population is a mis-pick, not a
    # data property, so it survives every distributional check above.
    taken = {c for pair in vote_cols for c in pair if c}
    if pop_col:
        taken.add(pop_col)
    for group, label in scored:
        col = demographics[group]
        if col in taken:
            warnings.append(
                f"{label} ({col}) is also selected as a population or vote "
                f"column; check the column mapping.")

    missing = [lbl for g, lbl in SCORED_GROUPS if not demographics.get(g)]
    if missing:
        warnings.append(
            f"No column for {', '.join(missing)}; "
            f"{'that group is' if len(missing) == 1 else 'those groups are'} "
            f"left out of the demographic scores.")
    return errors, warnings


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
