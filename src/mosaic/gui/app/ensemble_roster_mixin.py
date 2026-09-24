"""Ensemble Roster: filter runs by up to five metric ranges, list the matches."""
from ._common import dpg, np
from .ensemble_views_mixin import _BAR_RGBA, _metric_label

_ROSTER_TAG = "popup_ens_roster"
_MAX_CRITERIA = 5
_PAGE_SIZE = 100
_W, _H = 660, 460
_ROW_H = 26                  # one criterion row, including item spacing
_RUN_COL_W = 56
_CELL_H = 20                 # table row height; every cell centres in it
_COMBO_W = 230
_TRACK_W, _TRACK_H, _PAD = 230, 22, 8     # range track; _PAD keeps knobs inside
_TEXT_W = 128                             # fits e.g. "0.412 to 0.455", "12.25% to 18.50%"

# Display rule per summary column: (scale, decimals, suffix). Values are scaled
# into display units before both the range tracks and the table see them, so a
# range shown as "10.0% to 12.5%" filters on the same numbers it shows.
_FMT = {
    "score": (1, 1, ""),
    "cut_edges": (1, 0, ""),
    "pop_dev_max_pct": (1, 2, "%"),
    "pop_dev_mean_pct": (1, 2, "%"),
    "compactness": (1, 1, ""),
    "polsby_popper": (1, 3, ""),
    "reock": (1, 3, ""),
    "county_congruence_penalty": (1, 1, ""),
    "county_excess_splits": (1, 0, ""),
    "county_unified_districts": (1, 0, ""),
    "mean_median": (100, 1, "%"),
    "efficiency_gap": (100, 1, "%"),
    "partisan_bias": (100, 1, "%"),
    "partisan_gini_penalty": (1, 1, ""),
    "expected_dem_seats": (1, 2, ""),
    "proportionality": (1, 1, ""),
    "inversion_chance": (100, 1, "%"),
    "competitiveness": (1, 1, ""),
    "dem_majority_chance": (100, 1, "%"),
    "hinge_chance": (100, 1, "%"),
    "electoral_opportunity": (1, 1, ""),
    "opportunity_black": (1, 2, ""),
    "opportunity_latino": (1, 2, ""),
    "opportunity_asian": (1, 2, ""),
    "neighborhood_severance_penalty": (1, 1, ""),
    "community_dispersion_penalty": (1, 1, ""),
    "alignment_mean_retention": (1, 1, "%"),
    "alignment_min_retention": (1, 1, "%"),
    "iterations": (1, 0, ""),
    "n3_probability": (100, 0, "%"),
    "flip_midpoint": (100, 0, "%"),
    "temp_factor": (1, 3, ""),
    "guide_fraction": (100, 0, "%"),
}

# Table headers must fit a fixed-width column; full names stay in the combos.
_SHORT = {
    "score": "Score",
    "cut_edges": "Cut Edges",
    "pop_dev_max_pct": "Pop Dev Max",
    "pop_dev_mean_pct": "Pop Dev Mean",
    "compactness": "Compactness",
    "polsby_popper": "Polsby-Popper",
    "reock": "Reock",
    "county_congruence_penalty": "County Cong.",
    "county_excess_splits": "Excess Splits",
    "county_unified_districts": "1-County Dists",
    "mean_median": "Mean-Median",
    "efficiency_gap": "Eff. Gap",
    "partisan_bias": "Partisan Bias",
    "partisan_gini_penalty": "Partisan Gini",
    "expected_dem_seats": "Exp. D Seats",
    "proportionality": "Proportionality",
    "inversion_chance": "Inversion Risk",
    "competitiveness": "Competitive",
    "dem_majority_chance": "D Majority",
    "hinge_chance": "Hinge",
    "electoral_opportunity": "Elec. Opp.",
    "opportunity_black": "Opp. Black",
    "opportunity_latino": "Opp. Latino",
    "opportunity_asian": "Opp. Asian",
    "neighborhood_severance_penalty": "Severance",
    "community_dispersion_penalty": "Dispersion",
    "alignment_mean_retention": "Align. Mean",
    "alignment_min_retention": "Align. Min",
    "iterations": "Iterations",
    "n3_probability": "N3",
    "flip_midpoint": "Flip Mid.",
    "temp_factor": "Temp",
    "guide_fraction": "Guide",
}


def fmt_rule(col: str) -> tuple[float, int, str]:
    return _FMT.get(col, (1, 2, ""))


def fmt_value(col: str, raw: float) -> str:
    """Display text for one raw summary value (no '-0.0', thousands commas)."""
    scale, places, suffix = fmt_rule(col)
    v = round(float(raw) * scale, places) + 0.0   # + 0.0 folds -0.0 into 0.0
    return f"{v:,.{places}f}{suffix}"


def display_values(col: str, raws) -> np.ndarray:
    """Raw values in display units, rounded as shown, so filtering agrees
    with what the table prints."""
    scale, places, _ = fmt_rule(col)
    return np.round(np.asarray(raws, dtype=float) * scale, places)


def snap(v: float, places: int) -> float:
    return round(v, places) + 0.0


class EnsembleRosterMixin:
    """Ensemble Roster: filter runs by up to five metric ranges, list the matches."""

    def _on_open_ens_roster(self) -> None:
        if self._queue_session_action(self._on_open_ens_roster):
            return
        if dpg.does_item_exist(_ROSTER_TAG):
            dpg.configure_item(_ROSTER_TAG, show=True)
            dpg.focus_item(_ROSTER_TAG)
            return
        self._roster: list[dict] = []
        self._roster_sig = None
        self._roster_page = 0
        self._roster_ranges = {}
        self._roster_table_cols: tuple = ()
        self._roster_sort = (0, False)        # (column index, descending)
        vp_w = dpg.get_viewport_client_width() or 1300
        with dpg.window(label="Ensemble Roster", tag=_ROSTER_TAG,
                        width=_W, height=_H, pos=[max(20, vp_w - _W - 60), 80],
                        # Width is locked (min == max); height drags freely.
                        min_size=[_W, 260], max_size=[_W, 4000],
                        no_collapse=True, no_scrollbar=True,
                        no_scroll_with_mouse=True):
            with dpg.group(horizontal=True):
                self.theme.text("Criteria", "heading")
                dpg.add_spacer(width=8)
                self._roster_add_btn = dpg.add_button(
                    label="+ Add", width=64, callback=self._on_roster_add)
            # Grows one row per criterion; the table below takes what's left.
            dpg.add_group(tag="roster_rows")
            dpg.add_spacer(height=2)
            self._roster_count = self.theme.text("", "muted")
            with dpg.group(horizontal=True):
                self._roster_prev = dpg.add_button(label="Previous", enabled=False,
                    callback=lambda: self._on_roster_page(-1))
                self._roster_next = dpg.add_button(label="Next", enabled=False,
                    callback=lambda: self._on_roster_page(1))
                self._roster_page_text = self.theme.text("", "muted")
            with dpg.tooltip(self._roster_count):
                dpg.add_text("Filters and sorting cover all runs. Click a row to view its map.",
                             wrap=300)
            with dpg.child_window(height=-1, border=False, no_scrollbar=True,
                                  tag="roster_table_box"):
                pass
        self._on_roster_add()     # start with one criterion

    # ── Criteria rows ────────────────────────────────────────────────────────

    def _on_roster_add(self) -> None:
        if self._queue_session_action(self._on_roster_add):
            return
        if len(self._roster) >= _MAX_CRITERIA:
            return
        cols, labels = self._ens_columns()
        used = {c["col"] for c in self._roster}
        free = [c for c in cols if c not in used]
        # A fresh Roster opens on Score; the next suggestion is Compactness.
        pick = next((c for c in ("score", "compactness") if c in free),
                    free[0] if free else (cols[0] if cols else ""))
        # pin_lo / pin_hi: that handle sits at the data's edge and follows it
        # as new runs widen the range; a handle pulled inward stays put.
        crit = {"col": pick, "pin_lo": True, "pin_hi": True, "drag": None, "drawn": None,
                "lo": 0.0, "hi": 1.0, "b_lo": 0.0, "b_hi": 1.0}
        with dpg.group(horizontal=True, parent="roster_rows") as row:
            crit["combo"] = dpg.add_combo(
                labels, default_value=_metric_label(pick) if pick else "",
                width=_COMBO_W, callback=self._on_roster_metric, user_data=crit)
            crit["track"] = dpg.add_drawlist(width=_TRACK_W, height=_TRACK_H)
            # Fixed-width box so the range text never shifts the x button.
            with dpg.child_window(width=_TEXT_W, height=_TRACK_H, border=False,
                                  no_scrollbar=True):
                crit["text"] = dpg.add_text("", color=self.theme.color("body"))
            dpg.add_button(label="x", width=22, callback=self._on_roster_remove,
                           user_data=crit)
        crit["row"] = row
        self._roster.append(crit)
        self._roster_grow(+1)
        self._roster_bounds(crit, reset=True)
        self._sync_roster_add()
        self._roster_sig = None

    def _on_roster_remove(self, sender, app_data, crit) -> None:
        if self._queue_session_action(self._on_roster_remove, sender, app_data, crit):
            return
        if crit not in self._roster:
            return
        dpg.delete_item(crit["row"])
        self._roster.remove(crit)
        self._roster_grow(-1)
        self._sync_roster_add()
        self._roster_sig = None
        self._roster_page = 0

    def _on_roster_metric(self, sender, label, crit) -> None:
        if self._queue_session_action(self._on_roster_metric, sender, label, crit):
            return
        if crit not in self._roster:
            return
        cols, labels = self._ens_columns()
        if label in labels:
            crit["col"] = cols[labels.index(label)]
        crit["pin_lo"] = crit["pin_hi"] = True
        self._roster_bounds(crit, reset=True)
        self._roster_sig = None
        self._roster_page = 0

    def _roster_grow(self, rows: int) -> None:
        """Grow or shrink the window by whole criterion rows, so the criteria
        section changes height and the table keeps its own."""
        if len(self._roster) == 1 and rows > 0:
            return                       # the first row is part of the base height
        h = dpg.get_item_height(_ROSTER_TAG) or _H
        dpg.configure_item(_ROSTER_TAG, height=max(260, h + rows * _ROW_H))

    def _sync_roster_add(self) -> None:
        dpg.configure_item(self._roster_add_btn,
                           enabled=len(self._roster) < _MAX_CRITERIA)

    def _roster_bounds(self, crit: dict, reset: bool = False) -> None:
        """Fit the track to the ensemble's range so far. A pinned handle (left
        at the edge) follows the data outward; one set inside it never moves,
        so a new run past a chosen limit stays excluded."""
        col = crit["col"]
        _, places, _ = fmt_rule(col)
        lo, hi = self._roster_ranges.get(col, (0.0, 1.0))
        if hi == lo:
            hi = lo + 10.0 ** -places
        crit["b_lo"], crit["b_hi"] = lo, hi
        if reset or crit["pin_lo"]:
            crit["lo"] = lo
        if reset or crit["pin_hi"]:
            crit["hi"] = hi

    # ── The two-handle range track ───────────────────────────────────────────

    def _roster_track_input(self, crit: dict) -> bool:
        """Mouse handling for one track: press picks the nearer handle (or the
        handle nearer a clicked spot), drag moves it, release lets go. Returns
        True when the range changed."""
        down = dpg.is_mouse_button_down(dpg.mvMouseButton_Left)
        if not down:
            crit["drag"] = None
            return False
        x0 = dpg.get_item_rect_min(crit["track"])[0]
        mx = dpg.get_mouse_pos(local=False)[0] - x0
        span = crit["b_hi"] - crit["b_lo"]
        v = crit["b_lo"] + (min(max(mx, _PAD), _TRACK_W - _PAD) - _PAD) \
            / (_TRACK_W - 2 * _PAD) * span
        places = fmt_rule(crit["col"])[1]
        v = snap(v, places)
        if crit["drag"] is None:
            if not (dpg.is_item_hovered(crit["track"])
                    and dpg.is_mouse_button_clicked(dpg.mvMouseButton_Left)):
                return False
            crit["drag"] = "lo" if abs(v - crit["lo"]) <= abs(v - crit["hi"]) else "hi"
            # Coincident handles: pick by drag direction next frame.
            if crit["lo"] == crit["hi"]:
                crit["drag"] = "hi" if v > crit["hi"] else "lo"
        old = (crit["lo"], crit["hi"])
        if crit["drag"] == "lo":
            crit["lo"] = min(v, crit["hi"])
        else:
            crit["hi"] = max(v, crit["lo"])
        if (crit["lo"], crit["hi"]) != old:
            # Dragging a handle back onto the edge re-pins it.
            crit["pin_lo"] = crit["lo"] <= crit["b_lo"]
            crit["pin_hi"] = crit["hi"] >= crit["b_hi"]
            return True
        return False

    def _roster_draw_track(self, crit: dict) -> None:
        mode = self.theme.palette.name
        sig = (crit["lo"], crit["hi"], crit["b_lo"], crit["b_hi"], mode, crit["col"])
        if sig == crit["drawn"]:
            return
        crit["drawn"] = sig
        t = crit["track"]
        dpg.delete_item(t, children_only=True)
        span = crit["b_hi"] - crit["b_lo"]

        def px(v: float) -> float:
            return _PAD + (v - crit["b_lo"]) / span * (_TRACK_W - 2 * _PAD)

        y = _TRACK_H / 2
        rail = (200, 205, 214, 255) if mode == "light" else (78, 84, 96, 255)
        blue = _BAR_RGBA[mode]
        knob = (255, 255, 255, 255) if mode == "light" else (225, 230, 238, 255)
        dpg.draw_line((_PAD, y), (_TRACK_W - _PAD, y), color=rail, thickness=3,
                      parent=t)
        a, b = px(crit["lo"]), px(crit["hi"])
        dpg.draw_line((a, y), (b, y), color=blue, thickness=4, parent=t)
        for cx in (a, b):
            dpg.draw_circle((cx, y), 6, color=blue, fill=knob, thickness=2,
                            parent=t)
        f = lambda v: fmt_value(crit["col"], v / fmt_rule(crit["col"])[0])  # noqa: E731
        dpg.set_value(crit["text"], f"{f(crit['lo'])} to {f(crit['hi'])}")

    # ── Refresh ──────────────────────────────────────────────────────────────

    def _on_roster_page(self, step):
        if self._queue_session_action(self._on_roster_page, step):
            return
        self._roster_page = max(0, self._roster_page + step)

    def _refresh_ens_roster(self, force: bool = False) -> None:
        """Keep criteria responsive while disk queries prepare one result page."""
        if self._queue_session_action(self._refresh_ens_roster, force):
            return
        if not dpg.does_item_exist(_ROSTER_TAG) or not dpg.is_item_shown(_ROSTER_TAG):
            return
        cols, labels = self._ens_columns()
        for crit in list(self._roster):
            if crit["col"] not in cols:
                self._on_roster_remove(None, None, crit)
        revision = self._ens_revision()
        source_changed = getattr(self, "_roster_source_id", None) != revision[0]
        if source_changed:
            self._roster_source_id = revision[0]
            self._roster_reset_bounds = True
            self._roster_page = 0
            self._roster_sig = None
            if dpg.does_item_exist("roster_table"):
                dpg.delete_item("roster_table", children_only=True, slot=1)
        w = self._ens_writer
        crit_cols = tuple(c["col"] for c in self._roster)
        rules = [(c, *fmt_rule(c)[:2]) for c in crit_cols]
        ready, ranges = self._ens_query("roster_bounds", (revision, crit_cols),
            lambda: {c: w.store.bounds(c, scale, places, count=revision[1])
                     for c, scale, places in rules}
                    if w else {})
        if not ready or isinstance(ranges, Exception):
            dpg.set_value(self._roster_count, "Could not read results."
                          if isinstance(ranges, Exception) else "Updating...")
            if source_changed and dpg.does_item_exist("roster_table"):
                dpg.delete_item("roster_table", children_only=True, slot=1)
            return
        self._roster_ranges = ranges
        bounds_sig = self._ens_query_used["roster_bounds"]
        revision = bounds_sig[0]
        new_runs = bounds_sig != getattr(self, "_roster_bounds_sig", None)
        self._roster_bounds_sig = bounds_sig
        for crit in self._roster:
            if new_runs:
                dpg.configure_item(crit["combo"], items=labels)
                self._roster_bounds(crit, reset=getattr(self, "_roster_reset_bounds", False))
            self._roster_track_input(crit)
            self._roster_draw_track(crit)
        self._roster_reset_bounds = False
        crit_sig = tuple((c["col"], c["lo"], c["hi"], c["pin_lo"], c["pin_hi"])
                         for c in self._roster)
        if crit_cols != self._roster_table_cols:
            self._roster_sort = (0, False)
        if (self._roster_sig is not None
                and (source_changed or crit_sig != self._roster_sig[1]
                     or self._roster_sort != self._roster_sig[2])):
            self._roster_page = 0
        sig = (revision, crit_sig, self._roster_sort, self._roster_page)
        if sig == self._roster_sig and not force:
            return
        criteria = [(*c, *fmt_rule(c[0])[:2]) for c in crit_sig]
        sc, desc = self._roster_sort
        sort_col = crit_cols[sc - 1] if 0 < sc <= len(crit_cols) else None
        page = self._roster_page
        ready, result = self._ens_query("roster", sig,
            lambda: w.store.roster(criteria, sort_col, desc, page, _PAGE_SIZE, count=revision[1])
                    if w else ([], 0, 0))
        if not ready:
            dpg.set_value(self._roster_count, "Updating...")
            return
        self._roster_sig = self._ens_query_used["roster"]
        if isinstance(result, Exception):
            dpg.set_value(self._roster_count, "Could not read results.")
            return
        rows, matched, self._roster_page = result
        total = self._roster_sig[0][1]
        dpg.set_value(self._roster_count, f"{matched:,} of {total:,} runs match"
                      if total else "No runs yet.")
        pages = max(1, (matched + _PAGE_SIZE - 1) // _PAGE_SIZE)
        dpg.set_value(self._roster_page_text, f"Page {self._roster_page + 1:,} of {pages:,}")
        dpg.configure_item(self._roster_prev, enabled=self._roster_page > 0)
        dpg.configure_item(self._roster_next, enabled=self._roster_page + 1 < pages)
        self._roster_table(crit_cols, rows, range(len(rows)))

    def _roster_table(self, crit_cols: tuple, rows: list, idx) -> None:
        # Columns change only with the criteria; rebuild then, else just rows,
        # so the header's sort arrow survives new runs.
        if crit_cols != self._roster_table_cols or not dpg.does_item_exist("roster_table"):
            if dpg.does_item_exist("roster_table"):
                dpg.delete_item("roster_table")
            self._roster_table_cols = crit_cols
            with dpg.table(tag="roster_table", parent="roster_table_box",
                           header_row=True, row_background=True,
                           borders_outerH=True, borders_innerV=False,
                           borders_outerV=True, borders_innerH=False,
                           policy=dpg.mvTable_SizingFixedFit, scrollY=True,
                           freeze_rows=1, height=-1, width=-1, clipper=True,
                           sortable=True, sort_tristate=False,
                           callback=self._on_roster_sort):
                dpg.add_table_column(label="Run", width_fixed=True,
                                     init_width_or_weight=_RUN_COL_W,
                                     default_sort=True, prefer_sort_ascending=True)
                for col in crit_cols:
                    dpg.add_table_column(label=_SHORT.get(col, _metric_label(col)),
                                         width_stretch=True, init_width_or_weight=1.0)
            dpg.bind_item_theme("roster_table", self._roster_cell_theme())
        dpg.delete_item("roster_table", children_only=True, slot=1)
        for i in idx:
            r = rows[i]
            cells = [str(r["run_id"]).removeprefix("run_")]
            for col in crit_cols:
                v = r.get(col)
                cells.append(fmt_value(col, v) if isinstance(v, (int, float))
                             and np.isfinite(v) else "--")
            # Every cell is the same fixed-height selectable, so all of them
            # centre on one line; clicking any cell maps that run.
            with dpg.table_row(parent="roster_table", height=_CELL_H):
                for text in cells:
                    dpg.add_selectable(label=text, height=_CELL_H,
                                       callback=self._on_roster_run_click,
                                       user_data=(self._ens_writer, r["run_id"]))

    def _roster_cell_theme(self):
        """Selectable text centred vertically in its fixed-height cell."""
        if not hasattr(self, "_roster_cell_theme_id"):
            with dpg.theme() as self._roster_cell_theme_id:
                with dpg.theme_component(dpg.mvSelectable):
                    dpg.add_theme_style(dpg.mvStyleVar_SelectableTextAlign, 0.0, 0.5,
                                        category=dpg.mvThemeCat_Core)
        return self._roster_cell_theme_id

    def _on_roster_run_click(self, sender, app_data, run_id) -> None:
        if self._queue_session_action(self._on_roster_run_click, sender, app_data, run_id):
            return
        writer, rid = run_id
        if writer is not self._ens_writer:
            return
        if dpg.does_item_exist(sender):
            dpg.set_value(sender, False)      # no sticky highlight
        self._open_ens_map_run(rid)

    def _on_roster_sort(self, sender, sort_specs) -> None:
        if self._queue_session_action(self._on_roster_sort, sender, sort_specs):
            return
        if not sort_specs or not dpg.does_item_exist(sender):
            return
        col_id, direction = sort_specs[0]
        cols = dpg.get_item_children(sender, 0)
        if col_id in cols:
            self._roster_sort = (cols.index(col_id), direction < 0)
