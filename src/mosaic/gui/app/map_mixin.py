"""Map render/overlay toggles and theme synchronisation."""
from ._common import _MAP_DH, _MAP_DW, _build_camera_icon, _build_more_icon, dpg, np


class MapMixin:
    """Map render/overlay toggles and theme synchronisation."""

    def _on_theme_change(self):
        choice = dpg.get_value(self._theme_radio)
        self.theme.apply("dark" if choice == "Dark" else "light")
        self._sync_map_bg_to_theme()
        self._sync_ref_lines_to_theme()
        self._sync_camera_icon_to_theme()
        self._phase_apply_fade()        # repaint phase trail for the new palette

    def _align_photo_icons(self, *_):
        """Resize the overlay-row spacer so the photo icons hug the right edge.

        Called once after first render and again on every viewport resize.
        Uses the actual measured positions/sizes rather than guessing widths
        because checkbox widths depend on the font in use.
        """
        fill = getattr(self, "_overlay_fill", None)
        cam = getattr(self, "_cam_btn", None)
        more = getattr(self, "_more_btn", None)
        spinner = getattr(self, "_save_spinner", None)
        if not all(t is not None and dpg.does_item_exist(t)
                   for t in (fill, cam, more, spinner)):
            return
        if not dpg.does_item_exist("map_container"):
            return
        container_w = dpg.get_item_rect_size("map_container")[0]
        if container_w <= 1:
            return
        cam_x = dpg.get_item_pos(cam)[0]
        cam_w = dpg.get_item_rect_size(cam)[0] or 28
        more_w = dpg.get_item_rect_size(more)[0] or 28
        spinner_w = dpg.get_item_rect_size(spinner)[0] or 0
        photo_block = cam_w + more_w + spinner_w + 16
        right_pad = 12
        target_cam_x = container_w - photo_block - right_pad
        try:
            current_w = dpg.get_item_configuration(fill).get("width", 20)
        except SystemError:
            current_w = 20
        new_w = max(8, int(current_w + (target_cam_x - cam_x)))
        dpg.configure_item(fill, width=new_w)

    def _sync_camera_icon_to_theme(self):
        """Repaint the map-toolbar icon textures in the current palette's body color."""
        r, g, b, _ = self.theme.color("body")
        fg = (int(r), int(g), int(b))
        if dpg.does_item_exist("camera_icon_texture"):
            dpg.set_value("camera_icon_texture", _build_camera_icon(fg))
        if dpg.does_item_exist("more_icon_texture"):
            dpg.set_value("more_icon_texture", _build_more_icon(fg))
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

    def _on_partisan_overlay_toggle(self):
        if dpg.get_value(self._partisan_overlay):
            for cb, attr in [
                (self._district_partisan, "district_partisan_overlay"),
                (self._compactness_view,  "compactness_view"),
                (self._pop_dev_view,      "pop_dev_view"),
            ]:
                dpg.set_value(cb, False)
                if self.map_view:
                    setattr(self.map_view, attr, False)
        if self.map_view is None:
            return
        self.map_view.partisan_overlay = dpg.get_value(self._partisan_overlay)
        self._rerender_map()

    def _on_district_partisan_toggle(self):
        if dpg.get_value(self._district_partisan):
            for cb, attr in [
                (self._partisan_overlay, "partisan_overlay"),
                (self._compactness_view, "compactness_view"),
                (self._pop_dev_view,     "pop_dev_view"),
            ]:
                dpg.set_value(cb, False)
                if self.map_view:
                    setattr(self.map_view, attr, False)
        if self.map_view is None:
            return
        self.map_view.district_partisan_overlay = dpg.get_value(self._district_partisan)
        self._rerender_map()

    def _on_splits_view_toggle(self):
        if self.map_view is None:
            return
        self.map_view.splits_view = dpg.get_value(self._splits_view)
        self._rerender_map()

    def _on_compactness_toggle(self):
        if dpg.get_value(self._compactness_view):
            for cb, attr in [
                (self._partisan_overlay, "partisan_overlay"),
                (self._district_partisan, "district_partisan_overlay"),
                (self._pop_dev_view, "pop_dev_view"),
            ]:
                dpg.set_value(cb, False)
                if self.map_view:
                    setattr(self.map_view, attr, False)
        if self.map_view is None:
            return
        self.map_view.compactness_view = dpg.get_value(self._compactness_view)
        self._rerender_map()

    def _on_pop_dev_toggle(self):
        if dpg.get_value(self._pop_dev_view):
            for cb, attr in [
                (self._partisan_overlay, "partisan_overlay"),
                (self._district_partisan, "district_partisan_overlay"),
                (self._compactness_view, "compactness_view"),
            ]:
                dpg.set_value(cb, False)
                if self.map_view:
                    setattr(self.map_view, attr, False)
        if self.map_view is None:
            return
        self.map_view.pop_dev_view = dpg.get_value(self._pop_dev_view)
        self._rerender_map()

    def _on_labels_toggle(self):
        if self.map_view is None:
            return
        self.map_view.show_labels = dpg.get_value(self._show_labels)
        self._rerender_map()

    # ── Action callbacks ──────────────────────────────────────────────────────
