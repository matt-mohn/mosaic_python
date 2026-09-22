"""Map render/overlay toggles and theme synchronisation."""
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from queue import Empty, SimpleQueue

from mosaic.gui.map_navigation import MapViewport
from mosaic.gui.map_view import MapView

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

    def _init_map_navigation(self):
        self._map_viewport = MapViewport()
        # Windows/Linux use drawlist input; macOS trackpads use ImPlot's
        # fractional native wheel handling.
        self._map_native_input = sys.platform == "darwin"
        self._map_nav_events = SimpleQueue()
        self._map_plot_reset = True
        self._map_axes_locked_frame = None
        self._map_inputs_enabled = False
        self._map_nav_revision = 0
        self._map_nav_changed_at = 0.0
        self._map_nav_failed_revision = None
        self._map_drag_pos = None
        self._map_nav_executor = None
        self._map_nav_future = None
        self._map_nav_job = None

    def _map_can_navigate(self):
        return (self.map_view is not None and self.map_view._loaded
                and not self._map_loading and self.runner is not None
                and self.runner.gdf is not None
                and id(self.runner.gdf) == self._map_loaded_gdf_id)

    def _on_map_wheel(self, sender, app_data):
        # Callbacks only enqueue input; frame-loop code owns the viewport.
        if self._map_can_navigate() and dpg.is_item_hovered("map_canvas"):
            x, y = dpg.get_mouse_pos(local=False)
            left, top = dpg.get_item_rect_min("map_canvas")
            self._map_nav_events.put(("zoom", float(app_data),
                                      (x - left) / _MAP_DW, (y - top) / _MAP_DH))

    def _reset_map_navigation(self):
        # Invalidates an in-flight raster even if a new file uses the same path.
        self._map_nav_revision += 1
        self._map_viewport.fit()
        self._map_drag_pos = None
        self._map_nav_failed_revision = None
        self._map_plot_reset = True
        while True:
            try:
                self._map_nav_events.get_nowait()
            except Empty:
                break
        self._update_map_preview()

    def _shutdown_map_navigation(self):
        self._map_nav_revision += 1
        if self._map_nav_executor is not None:
            self._map_nav_executor.shutdown(wait=False, cancel_futures=True)

    def _update_map_preview(self):
        if self.map_view is None or not dpg.does_item_exist("map_image"):
            return
        if not self._map_native_input:
            rectangle = self._map_viewport.preview(
                self.map_view._view_bounds, _MAP_DW, _MAP_DH)
            if rectangle is None:
                dpg.configure_item("map_image", show=False)
            else:
                dpg.configure_item("map_image", show=True, **rectangle)
            return
        # ImPlot crops/transforms the existing texture using its native axes.
        # Plot y points upward; viewport/image y points downward.
        x0, y0, x1, y1 = self.map_view._view_bounds
        dpg.configure_item("map_image", bounds_min=(x0, -y1), bounds_max=(x1, -y0))

    def _read_map_mouse(self, ready):
        """Process drawlist wheel and drag input on Windows and Linux."""
        changed = False
        while True:
            try:
                event = self._map_nav_events.get_nowait()
            except Empty:
                break
            if ready:
                changed |= (self._map_viewport.fit() if event[0] == "fit"
                            else self._map_viewport.zoom(event[1], event[2:]))
        if not ready:
            self._map_drag_pos = None
            return False
        hovered = dpg.is_item_hovered("map_canvas")
        mouse = dpg.get_mouse_pos(local=False)
        left = dpg.mvMouseButton_Left
        if hovered and dpg.is_mouse_button_double_clicked(left):
            changed |= self._map_viewport.fit()
            self._map_drag_pos = None
        elif hovered and dpg.is_mouse_button_clicked(left):
            self._map_drag_pos = mouse
        if self._map_drag_pos is not None:
            if dpg.is_mouse_button_down(left):
                dx = (mouse[0] - self._map_drag_pos[0]) / _MAP_DW
                dy = (mouse[1] - self._map_drag_pos[1]) / _MAP_DH
                changed |= self._map_viewport.pan(dx, dy)
                self._map_drag_pos = mouse
            else:
                self._map_drag_pos = None
        return changed

    def _set_map_axes(self):
        x0, y0, x1, y1 = self._map_viewport.bounds
        dpg.set_axis_limits("map_x", x0, x1)
        dpg.set_axis_limits("map_y", -y1, -y0)
        # Explicit limits lock native input. Release after one rendered frame.
        self._map_axes_locked_frame = dpg.get_frame_count()

    def _read_map_plot(self, ready):
        """Read native pan/zoom, bypassing DPG's integer-only wheel callback."""
        if not dpg.does_item_exist("map_x"):
            return False
        if ready != self._map_inputs_enabled:
            dpg.configure_item("map_canvas", no_inputs=not ready)
            self._map_inputs_enabled = ready
        if self._map_plot_reset:
            self._set_map_axes()
            self._map_plot_reset = False
            return False
        if self._map_axes_locked_frame is not None:
            # A minimized/clipped plot may not render even as frames advance.
            # Do not unlock until its requested limits have actually landed.
            x0, y0, x1, y1 = self._map_viewport.bounds
            applied = (dpg.get_axis_limits("map_x"), dpg.get_axis_limits("map_y"))
            expected = ((x0, x1), (-y1, -y0))
            if (dpg.get_frame_count() > self._map_axes_locked_frame
                    and np.allclose(applied, expected, rtol=0, atol=1e-12)):
                dpg.set_axis_limits_auto("map_x")
                dpg.set_axis_limits_auto("map_y")
                self._map_axes_locked_frame = None
            return False
        if not ready:
            self._map_drag_pos = None
            return False

        hovered = dpg.is_item_hovered("map_canvas")
        left = dpg.mvMouseButton_Left
        if hovered and dpg.is_mouse_button_double_clicked(left):
            changed = self._map_viewport.fit()
            self._map_drag_pos = None
            self._set_map_axes()
            return changed
        if hovered and dpg.is_mouse_button_clicked(left):
            self._map_drag_pos = True
        if not dpg.is_mouse_button_down(left):
            self._map_drag_pos = None

        x0, x1 = dpg.get_axis_limits("map_x")
        bottom, top = dpg.get_axis_limits("map_y")
        bounds = (x0, -top, x1, -bottom)
        changed = self._map_viewport.from_plot(bounds)
        # Also guard against axis-specific modifier gestures changing aspect.
        if not np.allclose(bounds, self._map_viewport.bounds, rtol=0, atol=1e-12):
            self._set_map_axes()
        return changed

    def _tick_map_navigation(self):
        future = self._map_nav_future
        if future is not None and future.done():
            self._map_nav_future = None
            revision, gdf, source = self._map_nav_job
            try:
                raster = future.result()
            except Exception as exc:
                raster = None
                if revision == self._map_nav_revision:
                    self._map_nav_failed_revision = revision
                    self.state.update(status_message=f"Map zoom failed: {exc}")
            if (raster is not None and revision == self._map_nav_revision
                    and self._map_can_navigate() and self.runner.gdf is gdf
                    and self.map_view is source):
                # Only one displayed pixel grid. The temporary replacement is
                # discarded after installation; no previous views are retained.
                source._pixel_map = raster._pixel_map
                source._precinct_centroids = raster._precinct_centroids
                source._view_bounds = raster._view_bounds
                source._label_centers_cache = None  # existing label cache
                with self.state._lock:
                    has_assignment = self.state.current_assignment is not None
                if not has_assignment:
                    source.draw_blank()
                self.state.update(map_needs_update=True)
                self._update_map_preview()
            self._map_nav_job = None

        ready = self._map_can_navigate()
        changed = (self._read_map_plot(ready) if self._map_native_input
                   else self._read_map_mouse(ready))
        if not ready:
            self._map_drag_pos = None
            return
        if changed:
            self._map_nav_revision += 1
            self._map_nav_changed_at = time.monotonic()
            self._map_nav_failed_revision = None
            self._update_map_preview()

        if (self._map_nav_future is not None or self._map_drag_pos is not None
                or self._map_viewport.bounds == self.map_view._view_bounds
                or self._map_nav_failed_revision == self._map_nav_revision
                or time.monotonic() - self._map_nav_changed_at < 0.18):
            return

        revision = self._map_nav_revision
        gdf, source = self.runner.gdf, self.map_view
        bounds = self._map_viewport.bounds
        self._map_nav_job = (revision, gdf, source)

        def rebuild():
            raster = MapView(source._ttag, source._w, source._h)
            complete = raster.load(
                gdf, view_bounds=bounds,
                cancelled=lambda: revision != self._map_nav_revision,
            )
            return raster if complete else None

        if self._map_nav_executor is None:
            self._map_nav_executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="mosaic-map")
        self._map_nav_future = self._map_nav_executor.submit(rebuild)

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
        """Upload current overlay flags immediately when an assignment exists.

        Queue an update when no assignment is available.
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

        This is a flat list because a DPG combo identifies a selection by its
        item string and cannot disable or style individual entries. The
        "Results - " and "Demographics - " prefixes provide grouping.
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
        """Refresh labels for loaded data and clear an unavailable selection."""
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
