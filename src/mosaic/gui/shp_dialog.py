"""
Shapefile column-picker popup.

Shown after the shapefile is read but before the graph is built.
The user selects which column is Population, which is the Precinct ID,
an optional County column, and any number of DEM/GOP election pairs.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

import dearpygui.dearpygui as dpg

from mosaic.io.inspect import ShapefileConfig, ShapefileInspection

log = logging.getLogger("mosaic")

_W = 580
_H = 620


class ShapefileDialog:
    """Manages the shapefile setup modal popup."""

    def __init__(
        self,
        confirm_cb: Callable[[ShapefileInspection, ShapefileConfig], None],
        cancel_cb: Callable[[], None],
    ):
        self._confirm_cb = confirm_cb
        self._cancel_cb = cancel_cb
        self._inspection: Optional[ShapefileInspection] = None

        # Election row tracking: monotonic index + list of active indices
        self._election_next_id: int = 0
        self._election_active: list[int] = []

        # DPG item tags for required fields (set during build)
        self._file_text: int | str = ""
        self._status_text: int | str = ""
        self._pop_combo: int | str = ""
        self._pop_info: int | str = ""
        self._id_combo: int | str = ""
        self._id_info: int | str = ""
        self._county_combo: int | str = ""
        self._county_info: int | str = ""
        self._confirm_err: int | str = ""

    # ── Construction ──────────────────────────────────────────────────────────

    def build(self, vp_w: int, vp_h: int) -> None:
        """Build the DPG window. Call once during app setup()."""
        px = (vp_w - _W) // 2
        py = (vp_h - _H) // 2

        with dpg.window(
            tag="shp_dialog",
            label="Shapefile Setup",
            modal=True, no_close=True, show=False,
            width=_W, height=_H,
            pos=[px, py],
            no_scrollbar=True,
        ):
            # ── File / integrity header ───────────────────────────────────────
            self._file_text = dpg.add_text("No file selected")
            self._status_text = dpg.add_text("", color=(150, 150, 150))
            dpg.add_separator()

            # ── Required columns ─────────────────────────────────────────────
            dpg.add_text("Required Columns", color=(200, 200, 100))
            dpg.add_spacer(height=4)

            with dpg.group(horizontal=True):
                dpg.add_text("Population:  ", color=(180, 180, 180))
                self._pop_combo = dpg.add_combo(
                    items=[], default_value="",
                    width=260,
                    callback=self._on_pop_change,
                )
            self._pop_info = dpg.add_text(
                "Select a numeric population column",
                color=(130, 130, 130), indent=14,
            )

            dpg.add_spacer(height=6)

            with dpg.group(horizontal=True):
                dpg.add_text("Precinct ID: ", color=(180, 180, 180))
                self._id_combo = dpg.add_combo(
                    items=[], default_value="",
                    width=260,
                    callback=self._on_id_change,
                )
            self._id_info = dpg.add_text(
                "Select the unique precinct identifier column",
                color=(130, 130, 130), indent=14,
            )

            dpg.add_separator()

            # ── Optional ─────────────────────────────────────────────────────
            dpg.add_text("Optional", color=(200, 200, 100))
            dpg.add_spacer(height=4)

            with dpg.group(horizontal=True):
                dpg.add_text("County:      ", color=(180, 180, 180))
                self._county_combo = dpg.add_combo(
                    items=[], default_value="(none)",
                    width=260,
                    callback=self._on_county_change,
                )
            dpg.add_text(
                "  Required for county splits scoring, county-edge bias, and county overlay",
                color=(110, 110, 110), indent=14,
            )
            self._county_info = dpg.add_text("", color=(130, 130, 130), indent=14)

            dpg.add_separator()

            # ── Elections ────────────────────────────────────────────────────
            dpg.add_text("Elections", color=(200, 200, 100))
            dpg.add_text(
                "  Required for partisan scoring (Mean-Median, Efficiency Gap, etc.)",
                color=(110, 110, 110),
            )
            dpg.add_spacer(height=4)
            with dpg.child_window(
                tag="shp_elections_scroll", height=90, border=True,
            ):
                dpg.add_text(
                    "No elections added.",
                    tag="shp_no_elections_text",
                    color=(130, 130, 130),
                )
            dpg.add_button(
                tag="shp_add_election_btn",
                label="+ Add Election",
                callback=self._on_add_election,
                width=130,
            )
            dpg.add_text(
                "(one election supported)",
                color=(110, 110, 110),
            )

            dpg.add_separator()

            # ── Footer ───────────────────────────────────────────────────────
            dpg.add_spacer(height=2)
            self._confirm_err = dpg.add_text("", color=(220, 80, 80))
            dpg.add_spacer(height=4)
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Confirm and Load",
                    callback=self._on_confirm_click,
                    width=148,
                )
                dpg.add_spacer(width=8)
                dpg.add_button(
                    label="Cancel",
                    callback=self._on_cancel_click,
                    width=80,
                )

    # ── Populate ──────────────────────────────────────────────────────────────

    def populate(self, inspection: ShapefileInspection) -> None:
        """Fill the dialog with data from a completed inspection. Call on main thread."""
        self._inspection = inspection

        # Clear any leftover election rows from a previous load
        for i in list(self._election_active):
            if dpg.does_item_exist(f"shp_elec_{i}_row"):
                dpg.delete_item(f"shp_elec_{i}_row")
        self._election_active = []
        self._election_next_id = 0
        dpg.configure_item("shp_no_elections_text", show=True)
        dpg.configure_item("shp_add_election_btn", enabled=True)

        # ── Header ────────────────────────────────────────────────────────────
        dpg.set_value(self._file_text, f"File: {inspection.path}")

        if inspection.load_error:
            dpg.set_value(self._status_text,
                          f"Error: {inspection.load_error}")
            dpg.configure_item(self._status_text, color=(220, 80, 80))
            dpg.configure_item("shp_dialog", show=True)
            return

        if inspection.geometry_invalid == 0:
            status_str = (f"{inspection.n_precincts:,} precincts loaded  |  "
                          f"all geometries valid")
            status_col = (90, 200, 90)
        else:
            status_str = (f"{inspection.n_precincts:,} precincts  |  "
                          f"{inspection.geometry_invalid} invalid geometr"
                          + ("y" if inspection.geometry_invalid == 1 else "ies"))
            status_col = (220, 180, 60)

        dpg.set_value(self._status_text, status_str)
        dpg.configure_item(self._status_text, color=status_col)

        # ── Populate combos ───────────────────────────────────────────────────
        cols = inspection.columns
        dpg.configure_item(self._pop_combo, items=cols)
        dpg.configure_item(self._id_combo, items=cols)
        dpg.configure_item(self._county_combo, items=["(none)"] + cols)

        # Apply auto-detected hints
        if inspection.hint_pop_col:
            dpg.set_value(self._pop_combo, inspection.hint_pop_col)
            self._on_pop_change(None, inspection.hint_pop_col)
        else:
            dpg.set_value(self._pop_combo, "")
            dpg.set_value(self._pop_info, "Select a numeric population column")

        if inspection.hint_id_col:
            dpg.set_value(self._id_combo, inspection.hint_id_col)
            self._on_id_change(None, inspection.hint_id_col)
        else:
            dpg.set_value(self._id_combo, "")
            dpg.set_value(self._id_info, "Select the unique precinct identifier column")

        if inspection.hint_county_col:
            dpg.set_value(self._county_combo, inspection.hint_county_col)
            self._on_county_change(None, inspection.hint_county_col)
        else:
            dpg.set_value(self._county_combo, "(none)")
            dpg.set_value(self._county_info, "")

        dpg.set_value(self._confirm_err, "")
        dpg.configure_item("shp_dialog", show=True)

    # ── Column-change callbacks ───────────────────────────────────────────────

    def _on_pop_change(self, sender, app_data) -> None:
        col = app_data
        if not col or self._inspection is None:
            return
        info = self._inspection.column_info.get(col)
        if info is None:
            return
        if not info.is_numeric:
            dpg.set_value(self._pop_info, f"  Warning: '{col}' is not numeric (dtype: {info.dtype})")
            dpg.configure_item(self._pop_info, color=(220, 160, 60))
        else:
            pop_total = info.col_sum or 0.0
            null_str = f"  |  {info.n_null} null values" if info.n_null else ""
            dpg.set_value(self._pop_info, f"  Total population: {pop_total:,.0f}{null_str}")
            dpg.configure_item(self._pop_info, color=(130, 200, 130))

    def _on_id_change(self, sender, app_data) -> None:
        col = app_data
        if not col or self._inspection is None:
            return
        info = self._inspection.column_info.get(col)
        if info is None:
            return
        n = self._inspection.n_precincts
        if info.n_unique == n:
            dpg.set_value(self._id_info,
                          f"  {info.n_unique:,} unique values (all unique)")
            dpg.configure_item(self._id_info, color=(130, 200, 130))
        else:
            dupes = n - info.n_unique
            dpg.set_value(self._id_info,
                          f"  {info.n_unique:,} unique / {n:,} total — "
                          f"Warning: {dupes:,} duplicate value(s)")
            dpg.configure_item(self._id_info, color=(220, 160, 60))

    def _on_county_change(self, sender, app_data) -> None:
        col = app_data
        if self._inspection is None:
            return
        if not col or col == "(none)":
            dpg.set_value(self._county_info,
                          "  County splits, county bias, and county overlay will be disabled")
            dpg.configure_item(self._county_info, color=(130, 130, 130))
            return
        info = self._inspection.column_info.get(col)
        if info is None:
            return
        dpg.set_value(self._county_info,
                      f"  {info.n_unique:,} unique county values found")
        dpg.configure_item(self._county_info, color=(130, 200, 130))

    # ── Election management ───────────────────────────────────────────────────

    def _on_add_election(self) -> None:
        if self._inspection is None or self._inspection.gdf is None:
            return
        if self._election_active:
            return  # only one election supported

        i = self._election_next_id
        self._election_next_id += 1
        self._election_active.append(i)

        if len(self._election_active) == 1:
            dpg.configure_item("shp_no_elections_text", show=False)
            dpg.configure_item("shp_add_election_btn", enabled=False)

        cols = self._inspection.columns
        row_tag = f"shp_elec_{i}_row"
        hrow_tag = f"shp_elec_{i}_hrow"
        info_tag = f"shp_elec_{i}_info"

        dpg.add_group(tag=row_tag, parent="shp_elections_scroll")
        dpg.add_group(tag=hrow_tag, parent=row_tag, horizontal=True)

        dpg.add_text("DEM:", parent=hrow_tag, color=(130, 160, 255))
        dpg.add_combo(
            tag=f"shp_elec_{i}_dem",
            items=cols, default_value="",
            width=165, parent=hrow_tag,
            callback=self._on_election_change,
            user_data=i,
        )
        dpg.add_text("  GOP:", parent=hrow_tag, color=(255, 130, 130))
        dpg.add_combo(
            tag=f"shp_elec_{i}_gop",
            items=cols, default_value="",
            width=165, parent=hrow_tag,
            callback=self._on_election_change,
            user_data=i,
        )
        dpg.add_button(
            label="Remove",
            parent=hrow_tag,
            callback=self._on_remove_election,
            user_data=i,
            width=62,
        )
        dpg.add_text("", tag=info_tag, parent=row_tag, color=(130, 130, 130))
        dpg.add_spacer(height=3, parent=row_tag)

    def _on_remove_election(self, sender, app_data, user_data) -> None:
        i = user_data
        if dpg.does_item_exist(f"shp_elec_{i}_row"):
            dpg.delete_item(f"shp_elec_{i}_row")
        if i in self._election_active:
            self._election_active.remove(i)
        if not self._election_active:
            dpg.configure_item("shp_no_elections_text", show=True)
            dpg.configure_item("shp_add_election_btn", enabled=True)

    def _on_election_change(self, sender, app_data, user_data) -> None:
        i = user_data
        dem_col = dpg.get_value(f"shp_elec_{i}_dem")
        gop_col = dpg.get_value(f"shp_elec_{i}_gop")
        info_tag = f"shp_elec_{i}_info"
        gdf = self._inspection.gdf if self._inspection else None
        if gdf is None or not dem_col or not gop_col:
            return
        if dem_col in gdf.columns and gop_col in gdf.columns:
            dem_sum = int(gdf[dem_col].sum())
            gop_sum = int(gdf[gop_col].sum())
            total = dem_sum + gop_sum
            dpg.set_value(info_tag,
                          f"  DEM: {dem_sum:,}  |  GOP: {gop_sum:,}  |  "
                          f"Total: {total:,}")

    # ── Confirm / Cancel ──────────────────────────────────────────────────────

    def _collect_config(self) -> Optional[ShapefileConfig]:
        pop_col = dpg.get_value(self._pop_combo)
        id_col  = dpg.get_value(self._id_combo)
        county_val = dpg.get_value(self._county_combo)
        county_col = None if (not county_val or county_val == "(none)") else county_val

        elections: list[tuple[str, str]] = []
        for i in self._election_active:
            dem = dpg.get_value(f"shp_elec_{i}_dem")
            gop = dpg.get_value(f"shp_elec_{i}_gop")
            if dem and gop:
                elections.append((dem, gop))

        return ShapefileConfig(
            pop_col=pop_col,
            id_col=id_col,
            county_col=county_col,
            elections=elections,
        )

    def _on_confirm_click(self) -> None:
        if self._inspection is None:
            return

        config = self._collect_config()
        insp = self._inspection

        if not config.pop_col:
            dpg.set_value(self._confirm_err, "Please select a Population column.")
            return
        if not config.id_col:
            dpg.set_value(self._confirm_err, "Please select a Precinct ID column.")
            return

        pop_info = insp.column_info.get(config.pop_col)
        if pop_info and not pop_info.is_numeric:
            dpg.set_value(self._confirm_err,
                          f"'{config.pop_col}' is not numeric — choose a different column.")
            return

        dpg.configure_item("shp_dialog", show=False)
        self._confirm_cb(insp, config)

    def _on_cancel_click(self) -> None:
        dpg.configure_item("shp_dialog", show=False)
        self._cancel_cb()
