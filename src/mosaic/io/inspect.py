"""Shapefile inspection — reads a shapefile and profiles columns without renaming data."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import geopandas as gpd
import pandas as pd

log = logging.getLogger("mosaic")

_POP_HINTS    = {"population", "pop", "total_pop", "totpop", "pop100", "p0010001", "p001001"}
_ID_HINTS     = {"geoid20", "geoid", "vtdid", "vtd", "precinct_id", "id", "gid", "objectid"}
_COUNTY_HINTS = {
    "cty", "county", "countyfp20", "countyfp", "county20",
    "cntyfp", "cntyfp20", "cty_code", "cntyvtd", "countyfips", "co_fips",
}


@dataclass
class ColumnInfo:
    name: str
    dtype: str
    n_unique: int
    n_null: int
    is_numeric: bool
    col_sum: Optional[float] = None   # only for numeric columns


@dataclass
class ShapefileConfig:
    """Column assignments confirmed by the user in the shapefile dialog."""
    pop_col: str
    id_col: str
    county_col: Optional[str] = None
    elections: list[tuple[str, str]] = field(default_factory=list)  # [(dem_col, gop_col)]


@dataclass
class ShapefileInspection:
    """Raw GDF plus per-column statistics. Produced by inspect_shapefile()."""
    path: str
    gdf: Optional[gpd.GeoDataFrame]
    n_precincts: int
    columns: list[str]                     # non-geometry column names, in file order
    column_info: dict[str, ColumnInfo]
    geometry_valid: int
    geometry_invalid: int
    load_error: Optional[str] = None
    hint_pop_col: Optional[str] = None    # auto-detected suggestion
    hint_id_col: Optional[str] = None
    hint_county_col: Optional[str] = None


# Census GEOID column names — used only for the no-fiona heuristic fallback
_GEOID_COL_NAMES: frozenset[str] = frozenset({"geoid", "geoid20", "geoid10", "vtdid", "vtd"})
# Standard fixed widths for census GEOIDs — fallback only
_GEOID_STANDARD_WIDTHS: frozenset[int] = frozenset({5, 11, 12, 15})


def _restore_zero_padded_ids(gdf: gpd.GeoDataFrame, path: str) -> None:
    """
    Restore leading zeros for int64 columns that lost them at read time.

    When a .dbf field is stored as type 'N' (numeric), geopandas reads it as
    int64 regardless of the column name, silently dropping leading zeros (e.g.
    GEOID20 "08001000001" becomes 8001000001, or "Vtd_ID" "003" becomes 3).

    Primary fix: use fiona's schema to get the original field width for ANY
    int64 column. If schema_width > max digit count of current values, the
    difference was leading zeros and we restore them with zfill.

    Fallback (no fiona): apply only to columns whose name matches a known
    census GEOID pattern, using a standard-length heuristic. Arbitrary
    column names are left untouched — we cannot safely infer their width.
    """
    int64_cols = [
        c for c in gdf.columns
        if c != "geometry" and pd.api.types.is_integer_dtype(gdf[c])
    ]
    if not int64_cols:
        return

    fiona_props: dict[str, str] = {}
    try:
        import fiona
        with fiona.open(path) as _src:
            fiona_props = dict(_src.schema.get("properties", {}))
    except Exception:
        pass

    for col in int64_cols:
        non_null = gdf[col].dropna()
        if non_null.empty:
            continue
        max_digits = int(non_null.astype(str).str.len().max())

        target_width: int | None = None

        # Primary: fiona schema width — works for any column name/length
        field_spec = fiona_props.get(col, "")
        if ":" in field_spec:
            try:
                schema_width = int(field_spec.split(":")[1])
                if schema_width > max_digits:
                    target_width = schema_width
            except ValueError:
                pass

        # Fallback: GEOID-name heuristic only — arbitrary names skipped
        if target_width is None and (
            col.lower() in _GEOID_COL_NAMES or col.lower().startswith("geoid")
        ):
            if (max_digits + 1) in _GEOID_STANDARD_WIDTHS:
                target_width = max_digits + 1

        if target_width is not None:
            mask = gdf[col].notna()
            gdf.loc[mask, col] = gdf.loc[mask, col].astype(str).str.zfill(target_width)
            gdf[col] = gdf[col].where(mask, other=None)
            log.info(f"Column '{col}': zero-padded to width {target_width} (was int64 in .dbf)")


def inspect_shapefile(path: str | Path) -> ShapefileInspection:
    """
    Load a shapefile and collect per-column statistics without renaming anything.
    Returns a ShapefileInspection; .load_error is set on failure and .gdf is None.
    """
    path = str(path)
    try:
        gdf = gpd.read_file(path)
        gdf = gdf.reset_index(drop=True)
        _restore_zero_padded_ids(gdf, path)
        n = len(gdf)

        cols = [c for c in gdf.columns if c != "geometry"]
        col_info: dict[str, ColumnInfo] = {}
        for c in cols:
            s = gdf[c]
            is_num = pd.api.types.is_numeric_dtype(s)

            # If dtype is non-numeric but every non-null value parses as a number, coerce.
            # Skip coercion when any value has a leading zero — those are zero-padded
            # identifiers like FIPS/GEOID20 codes that must stay as strings.
            if not is_num:
                non_null = s.dropna()
                if len(non_null) > 0:
                    converted = pd.to_numeric(non_null, errors="coerce")
                    has_leading_zero = non_null.astype(str).str.match(r'^0\d').any()
                    if converted.notna().all() and not has_leading_zero:
                        gdf[c] = pd.to_numeric(s, errors="coerce")
                        s = gdf[c]
                        is_num = True
                        log.info(f"Column '{c}': coerced string-numeric values to numeric")

            col_info[c] = ColumnInfo(
                name=c,
                dtype=str(s.dtype),
                n_unique=int(s.nunique()),
                n_null=int(s.isna().sum()),
                is_numeric=is_num,
                col_sum=float(s.sum()) if is_num else None,
            )

        valid_mask = gdf.geometry.is_valid
        n_valid = int(valid_mask.sum())

        return ShapefileInspection(
            path=path,
            gdf=gdf,
            n_precincts=n,
            columns=cols,
            column_info=col_info,
            geometry_valid=n_valid,
            geometry_invalid=n - n_valid,
            hint_pop_col=_hint(cols, _POP_HINTS, numeric_only=True, col_info=col_info),
            hint_id_col=_hint(cols, _ID_HINTS),
            hint_county_col=_hint(cols, _COUNTY_HINTS),
        )
    except Exception as exc:
        log.error(f"inspect_shapefile failed: {exc}")
        return ShapefileInspection(
            path=path, gdf=None, n_precincts=0,
            columns=[], column_info={},
            geometry_valid=0, geometry_invalid=0,
            load_error=str(exc),
        )


def _hint(
    cols: list[str],
    hints: set[str],
    *,
    numeric_only: bool = False,
    col_info: dict[str, ColumnInfo] | None = None,
) -> Optional[str]:
    for c in cols:
        if c.lower() in hints:
            if numeric_only and col_info and not col_info[c].is_numeric:
                continue
            return c
    return None
