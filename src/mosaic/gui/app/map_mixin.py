"""Map render/overlay toggles and theme synchronisation."""
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from queue import Empty, SimpleQueue

from mosaic.gui.map_navigation import FIT_VIEW, MAX_ZOOM, MapViewport
from mosaic.gui.map_view import MapView

from ._common import (
    _FILL_ATTRS,
    _FILL_NEEDS,
    _FILL_NONE,
    _FILL_OPTIONS,
    _MAP_DH,
    _MAP_DW,
    _MAP_MIN_W,
    _build_camera_icon,
    dpg,
    log,
    np,
)


def _resized_view_bounds(view, old_size, new_size, geographic_bounds):
    """Keep the geographic center and pixels per map unit across a resize.

    View coordinates refer to the fitted image, whose padding changes with the
    panel aspect ratio. Convert its center and zoom to the new fitted image,
    clamping only at the navigation limits. A fitted map stays fitted.
    """
    if view == FIT_VIEW:
        return FIT_VIEW
    old_w, old_h = old_size
    new_w, new_h = new_size
    left, bottom, right, top = geographic_bounds
    geo_w, geo_h = max(right - left, 1e-9), max(top - bottom, 1e-9)
    # The rasterizer's constant 4% fit padding cancels in this ratio.
    ratio = min(new_w / geo_w, new_h / geo_h) / min(old_w / geo_w, old_h / geo_h)
    x0, y0, x1, y1 = view
    cx = .5 + ((x0 + x1) / 2 - .5) * old_w / new_w * ratio
    cy = .5 + ((y0 + y1) / 2 - .5) * old_h / new_h * ratio
    span = min(1.0, max(1.0 / MAX_ZOOM, (x1 - x0) * ratio))
    x = min(max(cx - span / 2, 0.0), 1.0 - span)
    y = min(max(cy - span / 2, 0.0), 1.0 - span)
    return (x, y, x + span, y + span)


class MapMixin:
    """Map render/overlay toggles and theme synchronisation."""

    def _init_map_navigation(self):
        # Current on-screen map size; the width follows the window (see
        # _sync_map_size), the height is fixed.
        self._map_w, self._map_h = _MAP_DW, _MAP_DH
        self._map_target = _MAP_DW      # width the window currently asks for
        self._map_target_at = 0.0
        self._map_resize_job = None     # (future, width, gdf, source, revision)
        self._map_resize_failure = None  # (request key, attempts, retry time)
        self._map_texture_tag = "map_texture"
        self._map_resize_registry = None
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
                                      (x - left) / self._map_w, (y - top) / self._map_h))

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
                self.map_view._view_bounds, self._map_w, self._map_h)
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
                dx = (mouse[0] - self._map_drag_pos[0]) / self._map_w
                dy = (mouse[1] - self._map_drag_pos[1]) / self._map_h
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
                    and self.map_view is source
                    and (raster._w, raster._h) == (source._w, source._h)):
                # Only one displayed pixel grid. The temporary replacement is
                # discarded after installation; no previous views are retained.
                source._pixel_map = raster._pixel_map
                source._precinct_centroids = raster._precinct_centroids
                source._view_bounds = raster._view_bounds
                source._clear_raster_caches()
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

    # ── Map width follows the window ─────────────────────────────────────────

    def _resize_source_ready(self, source, gdf):
        """A resize may use only the fully loaded map currently on screen."""
        return (source is not None and source is self.map_view and source._loaded
                and self.runner is not None and self.runner.gdf is gdf
                and gdf is not None and not self._map_loading
                and id(gdf) == self._map_data_gdf_id == self._map_loaded_gdf_id
                and source._n_precincts == len(gdf))

    def _resize_failed(self, key, now):
        previous = self._map_resize_failure
        attempts = previous[1] + 1 if previous and previous[0] == key else 1
        self._map_resize_failure = (key, attempts, now + 1.0)
        self.state.update(status_message="Map resize failed; keeping the current map.")

    def _sync_map_size(self) -> None:
        """Per frame: when the map area's width settles on a new value, redraw
        the precinct grid at that width off the GUI thread, then swap it in.
        The height stays fixed. Skipped during an ensemble (the map is frozen)."""
        if not dpg.does_item_exist("map_container"):
            return
        rect_w = dpg.get_item_rect_size("map_container")[0]
        if rect_w <= 0:
            return
        # Inner width of the bordered, padded map box (the app sets both).
        sp = self.theme.palette.spacing
        inner = int(rect_w - 2 * sp.window_padding[0] - 2 * sp.child_border_size)
        # Score and Entropy split the same width, whatever the map is doing.
        half = max(80, (inner - sp.item_spacing[0]) // 2)
        plot = "score_half_plot"
        if dpg.does_item_exist(plot) and dpg.get_item_configuration(plot)["width"] != half:
            dpg.configure_item(plot, width=half)
        # Centre the image whenever it is narrower than the box (mid-drag, or
        # at the minimum width), so the state never sits off to one side. The
        # spacer's own item spacing counts toward the offset.
        offset = max(0, (inner - self._map_w) // 2)
        pad_w = offset - sp.item_spacing[0]
        show = pad_w >= 1
        cfg = dpg.get_item_configuration("map_center_pad")
        if cfg["show"] != show or (show and cfg["width"] != pad_w):
            dpg.configure_item("map_center_pad", show=show, width=max(1, pad_w))
        target = max(_MAP_MIN_W, inner)
        now = time.monotonic()
        if target != self._map_target:
            self._map_target, self._map_target_at = target, now

        # Read the current requested width before accepting any completed job.
        job = self._map_resize_job
        if job is not None and job[0].done():
            self._map_resize_job = None
            future, width, gdf, source, revision = job
            fresh = (not self._ensemble_active and width == target
                     and revision == self._map_nav_revision
                     and self._resize_source_ready(source, gdf))
            if fresh:
                key = (id(source), id(gdf), width, revision)
                try:
                    raster = future.result()
                    if raster is None:
                        raise ValueError("Resize produced no raster")
                    self._install_map_size(width, raster)
                    self._map_resize_failure = None
                except Exception:
                    log.exception("Map resize failed at width %d", width)
                    self._resize_failed(key, now)
        if self._ensemble_active:
            return
        if (target == self._map_w or now - self._map_target_at < 0.25
                or self._map_resize_job is not None or self._map_loading
                or self._map_nav_future is not None):
            return
        source = self.map_view
        gdf = self.runner.gdf if self.runner is not None else None
        revision = self._map_nav_revision
        key = (id(source), id(gdf), target, revision)
        failed = self._map_resize_failure
        if failed and failed[0] == key and (failed[1] >= 3 or now < failed[2]):
            return
        if gdf is None and (source is None or not source._loaded):
            try:
                self._install_map_size(target, None)
                self._map_resize_failure = None
            except Exception:
                log.exception("Empty map resize failed at width %d", target)
                self._resize_failed(key, now)
            return
        if not self._resize_source_ready(source, gdf):
            return

        height, texture = self._map_h, source._ttag
        view = self._map_viewport.bounds
        old_size = (source._w, source._h)
        inputs = dict(county_array=source._county_array,
                      dem_votes=source._dem_votes, gop_votes=source._gop_votes,
                      pp_data=source._pp_data, reock_data=source._reock_data,
                      populations=source._populations, vap_data=source._vap)

        def obsolete():
            return (target != self._map_target or revision != self._map_nav_revision
                    or self._ensemble_active or not self._resize_source_ready(source, gdf))

        def rebuild():
            raster = MapView(texture, target, height)
            bounds = _resized_view_bounds(view, old_size, (target, height), gdf.total_bounds)
            ok = raster.load(gdf, view_bounds=bounds, cancelled=obsolete, **inputs)
            return raster if ok else None

        if self._map_nav_executor is None:
            self._map_nav_executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="mosaic-map")
        self._map_resize_job = (self._map_nav_executor.submit(rebuild), target,
                                gdf, source, revision)

    def _install_map_size(self, width: int, raster) -> None:
        """Prepare a replacement texture before committing matching map geometry."""
        h = self._map_h
        mv = self.map_view
        if raster is None:
            if mv is not None and mv._loaded:
                raise ValueError("A loaded map requires a replacement raster")
        elif (not raster._loaded or raster._w != width or raster._h != h
              or raster._pixel_map.shape != (h, width)
              or mv is None or raster._n_precincts != mv._n_precincts):
            raise ValueError("Replacement raster does not match the loaded map")

        old_tag = self._map_texture_tag
        new_tag = dpg.generate_uuid()
        old_image = dpg.get_item_configuration("map_image")
        old_bounds = ({} if self._map_native_input else
                      {name: old_image[name] for name in ("pmin", "pmax")})
        new_bounds = {} if self._map_native_input else dict(pmin=(0, 0), pmax=(width, h))
        registry = None
        try:
            with dpg.texture_registry() as registry:
                dpg.add_raw_texture(width=width, height=h,
                                    default_value=np.zeros(width * h * 4, dtype=np.float32),
                                    format=dpg.mvFormat_Float_rgba, tag=new_tag)
            dpg.configure_item("map_image", texture_tag=new_tag, **new_bounds)
            dpg.configure_item("map_canvas", width=width)
        except Exception:
            dpg.configure_item("map_image", texture_tag=old_tag, **old_bounds)
            dpg.configure_item("map_canvas", width=self._map_w)
            if dpg.does_item_exist(new_tag):
                dpg.delete_item(new_tag)
            if registry is not None:
                dpg.delete_item(registry)
            raise
        self._map_texture_tag = new_tag
        self._map_w = width
        log.info("Map resized to %d x %d px (map box %d px wide)", width, h,
                 self._map_target)
        if mv is not None:
            mv._ttag = new_tag
            mv._w, mv._h = width, h
            mv._last_rgba = None
        if raster is not None:
            mv._pixel_map = raster._pixel_map
            mv._precinct_centroids = raster._precinct_centroids
            mv._view_bounds = raster._view_bounds
            mv._clear_raster_caches()
        dpg.delete_item(old_tag)
        # The active texture registry owns the texture until the next swap.
        old_registry = self._map_resize_registry
        self._map_resize_registry = registry
        if old_registry is not None:
            dpg.delete_item(old_registry)
        if raster is None:
            self._reset_map_navigation()
        else:
            # Invalidate old-size navigation jobs without fitting the map.
            self._map_nav_revision += 1
            self._map_viewport.bounds = raster._view_bounds
            self._map_drag_pos = None
            self._map_nav_failed_revision = None
            self._map_plot_reset = True
            self._update_map_preview()
        if raster is not None:
            with self.state._lock:
                has_assignment = self.state.current_assignment is not None
            if has_assignment:
                self.state.update(map_needs_update=True)
            else:
                mv.draw_blank()
        else:
            self._sync_map_bg_to_theme()

    def _on_theme_change(self):
        choice = dpg.get_value(self._theme_radio)
        self.theme.apply("dark" if choice == "Dark" else "light")
        self._sync_map_bg_to_theme()
        self._sync_ref_lines_to_theme()
        self._sync_camera_icon_to_theme()
        self._phase_apply_fade()        # repaint phase trail for the new palette
        self._sync_ensemble_theme()

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
            self._map_w * self._map_h,
        )
        dpg.set_value(self._map_texture_tag, rgba)

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
