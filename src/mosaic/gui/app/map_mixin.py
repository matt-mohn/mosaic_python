"""Map render/overlay toggles and theme synchronisation."""
from ._common import (
    _FILL_ATTRS,
    _FILL_NEEDS,
    _FILL_NONE,
    _FILL_OPTIONS,
    _MAP_DH,
    _MAP_DW,
    _build_camera_icon,
    dpg,
    np,
)


class MapMixin:
    """Map render/overlay toggles and theme synchronisation."""

    def _on_theme_change(self):
        choice = dpg.get_value(self._theme_radio)
        self.theme.apply("dark" if choice == "Dark" else "light")
        self._sync_map_bg_to_theme()
        self._sync_ref_lines_to_theme()
        self._sync_camera_icon_to_theme()
        self._phase_apply_fade()        # repaint phase trail for the new palette

    def _sync_camera_icon_to_theme(self):
        """Repaint the map-toolbar camera icon in the current palette's body color."""
        r, g, b, _ = self.theme.color("body")
        fg = (int(r), int(g), int(b))
        if dpg.does_item_exist("camera_icon_texture"):
            dpg.set_value("camera_icon_texture", _build_camera_icon(fg))
        body = self.theme.color("body")
        muted = self.theme.color("muted")
        for tag_attr in ("_save_spinner", "_adv_save_spinner"):
            tag = getattr(self, tag_attr, None)
            if tag is not None and dpg.does_item_exist(tag):
                dpg.configure_item(tag, color=body, secondary_color=muted)

    def _sync_ref_lines_to_theme(self):
        """Rebind the partisan/win-chance 50/50 and median guide lines so
        they stay readable on the current plot background (white guides on
        dark mode, black on light)."""
        t = self._partisan_ref_themes[self.theme.palette.name]
        self._partisan_ref_theme = t
        for tag in ("partisan_ref", "partisan_median",
                    "win_chance_ref", "win_chance_median",
                    "cs_clean_max"):
            if dpg.does_item_exist(tag):
                dpg.bind_item_theme(tag, t)

    def _sync_map_bg_to_theme(self):
        """Push the theme's child_bg into the map view and refresh the texture."""
        r, g, b, _ = self.theme.color("child_bg")
        # Update MapView's bg so future LUT rebuilds use it.
        if self.map_view is not None:
            self.map_view._bg_color = np.array([r, g, b, 255], dtype=np.uint8)
        # If a shapefile is loaded, force the map to re-render with the new bg.
        if self.map_view is not None and self.map_view._loaded:
            self.state.update(map_needs_update=True)
            return
        # Otherwise paint the empty texture directly.
        rgba = np.tile(
            np.array([r / 255.0, g / 255.0, b / 255.0, 1.0], dtype=np.float32),
            _MAP_DW * _MAP_DH,
        )
        dpg.set_value("map_texture", rgba)

    def _rerender_map(self) -> None:
        """Re-compose and upload the current map frame with the latest overlay flags.

        Overlay toggle callbacks call this instead of queuing map_needs_update so
        the response is immediate rather than deferred to the next render-loop tick
        (which can silently drop the update if current_assignment is transiently None).
        Falls back to queuing when no assignment is available yet.
        """
        if self.map_view is None:
            return
        with self.state._lock:
            asgn = (self.state.current_assignment.copy()
                    if self.state.current_assignment is not None else None)
            n    = self.state.num_districts
            init = (self.state.initial_assignment.copy()
                    if self.state.initial_assignment is not None else None)
        if asgn is not None:
            self.map_view.render_assignment(asgn, n, init)
        else:
            self.state.update(map_needs_update=True)

    def _on_county_overlay_toggle(self):
        if self.map_view is None:
            return
        self.map_view.county_overlay = dpg.get_value(self._county_overlay)
        self._rerender_map()

    def _on_precinct_overlay_toggle(self):
        if self.map_view is None:
            return
        self.map_view.precinct_overlay = dpg.get_value(self._precinct_overlay)
        self._rerender_map()

    # ── Map fill combo ───────────────────────────────────────────────────────
    def _fill_labels(self) -> list[str]:
        """Combo items for the current data availability, unavailable entries
        suffixed rather than dropped.

        Deliberately a flat list. Separator rows were tried and reverted: a DPG
        combo's value IS the item string, so three identical divider strings are
        ambiguous, and items carry no per-item styling to grey or colour them.
        The "Results - " / "Demographics - " prefixes carry the grouping instead.
        """
        avail = getattr(self, "_fill_avail", {})
        out = [_FILL_NONE]
        for label, _attr, need in _FILL_OPTIONS:
            out.append(label if avail.get(need, False)
                       else f"{label}  ({_FILL_NEEDS[need]})")
        return out

    def _set_fill(self, view_attr):
        """Point the map at exactly one fill (or None) and repaint."""
        if self.map_view is None:
            return
        for a in _FILL_ATTRS:
            setattr(self.map_view, a, a == view_attr)
        self._rerender_map()

    def _on_fill_combo(self):
        """Apply the chosen fill; refuse (and snap back to None) if its data is
        absent, since the item is listed but not usable."""
        choice = dpg.get_value(self._fill_combo)
        if choice == _FILL_NONE:
            self._set_fill(None)
            return
        avail = getattr(self, "_fill_avail", {})
        for label, attr, need in _FILL_OPTIONS:
            if choice.startswith(label):
                if avail.get(need, False):
                    self._set_fill(attr)
                else:
                    dpg.set_value(self._fill_combo, _FILL_NONE)
                    self._set_fill(None)
                return
        dpg.set_value(self._fill_combo, _FILL_NONE)
        self._set_fill(None)

    def _clear_fill(self):
        """Reset to None -- used on load/new so a previous file's fill can't
        linger over fresh data."""
        if getattr(self, "_fill_combo", None) is None:
            return
        dpg.set_value(self._fill_combo, _FILL_NONE)
        self._set_fill(None)

    def _sync_fill_availability(self, *, elections: bool, race: bool,
                                compact: bool, pops: bool):
        """Refresh the combo's labels for what data is loaded, and drop the
        current selection if it just became unavailable. Replaces the per-
        checkbox enable/disable bookkeeping the old toolbar needed."""
        if getattr(self, "_fill_combo", None) is None:
            return
        avail = {"elections": elections, "race": race,
                 "compact": compact, "pops": pops}
        if avail == getattr(self, "_fill_avail", None):
            return                      # per-frame call; only rebuild on change
        self._fill_avail = avail
        current = dpg.get_value(self._fill_combo)
        dpg.configure_item(self._fill_combo, items=self._fill_labels())
        for label, _attr, need in _FILL_OPTIONS:
            if current.startswith(label):
                if avail.get(need, False):
                    dpg.set_value(self._fill_combo, label)   # drop needs-suffix
                else:
                    dpg.set_value(self._fill_combo, _FILL_NONE)
                    self._set_fill(None)
                return
        dpg.set_value(self._fill_combo, _FILL_NONE)

    def _on_splits_view_toggle(self):
        if self.map_view is None:
            return
        self.map_view.splits_view = dpg.get_value(self._splits_view)
        self._rerender_map()

    def _on_labels_toggle(self):
        if self.map_view is None:
            return
        self.map_view.show_labels = dpg.get_value(self._show_labels)
        self._rerender_map()

    # ── Action callbacks ──────────────────────────────────────────────────────
