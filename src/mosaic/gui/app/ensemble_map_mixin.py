"""Ensemble Map: a small static map of any completed run. Deliberately not
the main map: no zoom, pan, overlays, or labels, and its own texture, so it
never touches the main view or the chain."""
from ._common import MapView, dpg, log, np, threading

_MAP_TAG = "popup_ens_map"
_TEX_TAG = "ens_map_texture"
_W = 520                        # image width; height follows the state's shape
_H_MIN, _H_MAX = 200, 520
_NOTE_W, _BEST_W = 90, 90
# Run number + two step buttons + note, with DPG's 8 px item spacing.
_HEADER_W = 30 + 120 + 24 + 24 + _NOTE_W + 6 * 8


class EnsembleMapMixin:
    """Ensemble Map: a small static map of any completed run."""

    def _on_open_ens_map(self) -> None:
        if self._queue_session_action(self._on_open_ens_map):
            return
        if dpg.does_item_exist(_MAP_TAG):
            dpg.configure_item(_MAP_TAG, show=True)
            dpg.focus_item(_MAP_TAG)
            self._refresh_ens_map(True)
            return
        self._ens_map_view = None        # MapView once its geometry is drawn
        self._ens_map_gdf_id = 0         # gdf the geometry was drawn from
        self._ens_map_loading = False
        self._ens_map_sig = None
        self._ens_map_writer = None
        self._ens_map_error = ""
        self._ens_map_read_error = False
        vp_w = dpg.get_viewport_client_width() or 1300
        with dpg.window(label="Ensemble Map", tag=_MAP_TAG, autosize=True,
                        pos=[max(20, vp_w - _W - 90), 120], no_resize=True,
                        no_collapse=True, no_scrollbar=True):
            with dpg.group(horizontal=True):
                dpg.add_text("Run:")
                self._ens_map_run = dpg.add_input_int(
                    default_value=1, min_value=1, min_clamped=True,
                    width=120, step=0, callback=lambda: self._refresh_ens_map(True))
                with dpg.tooltip(self._ens_map_run):
                    dpg.add_text("Enter a completed run number.")
                dpg.add_button(label="<", width=24,
                               callback=lambda: self._step_ens_map(-1))
                dpg.add_button(label=">", width=24,
                               callback=lambda: self._step_ens_map(+1))
                # Fixed-width note so Best Score stays pinned at the right edge.
                with dpg.child_window(width=_NOTE_W, height=20, border=False,
                                      no_scrollbar=True):
                    self._ens_map_note = self.theme.text("", "muted")
                dpg.add_spacer(width=max(0, _W - _HEADER_W - _BEST_W))
                dpg.add_button(label="Best Score", width=_BEST_W,
                               callback=self._on_ens_map_best)
            # Sized and filled once the geometry is drawn (_ens_map_geometry).
            self._ens_map_canvas = dpg.add_drawlist(width=_W, height=_H_MIN)
        self._refresh_ens_map(True)

    def _open_ens_map_run(self, run_id: str) -> None:
        """Show one run (e.g. from a Roster row click), opening the Map if needed."""
        if self._queue_session_action(self._open_ens_map_run, run_id):
            return
        self._on_open_ens_map()
        count = self._sync_ens_map_runs()
        number = int(str(run_id).removeprefix('run_'))
        if 1 <= number <= count:
            dpg.set_value(self._ens_map_run, number)
        self._refresh_ens_map(True)

    def _on_ens_map_best(self) -> None:
        """Jump to the lowest-Score run so far (Score: lower is better)."""
        if self._queue_session_action(self._on_ens_map_best):
            return
        w = self._ens_writer
        rid = w.store.best_run() if w is not None else None
        if rid is not None:
            self._open_ens_map_run(rid)

    def _step_ens_map(self, step: int) -> None:
        if self._queue_session_action(self._step_ens_map, step):
            return
        count = self._sync_ens_map_runs()
        cur = dpg.get_value(self._ens_map_run)
        dpg.set_value(self._ens_map_run, max(1, min(cur + step, count)))
        self._refresh_ens_map(True)

    def _sync_ens_map_runs(self) -> int:
        w = self._ens_writer
        count = len(w.assignments) if w is not None else 0
        current = dpg.get_value(self._ens_map_run)
        selected = max(1, min(current, count))
        if selected != current:
            dpg.set_value(self._ens_map_run, selected)
        return count

    def _ens_map_geometry(self) -> bool:
        """Draw the precinct geometry once per shapefile, off the GUI thread.
        True when ready."""
        gdf = getattr(self, "_ens_source_gdf", None)
        if gdf is None:
            return False
        if self._ens_map_view is not None and self._ens_map_gdf_id == id(gdf):
            return True
        if not self._ens_map_loading and not self._ens_map_error:
            self._ens_map_loading = True
            x0, y0, x1, y1 = gdf.total_bounds
            h = int(np.clip(round(_W * (y1 - y0) / max(x1 - x0, 1e-9)), _H_MIN, _H_MAX))
            # New texture at this state's shape (DPG textures are fixed-size).
            if dpg.does_item_exist(_TEX_TAG):
                dpg.delete_item(self._ens_map_canvas, children_only=True)
                dpg.delete_item(_TEX_TAG)
            with dpg.texture_registry():
                dpg.add_raw_texture(width=_W, height=h,
                                    default_value=np.zeros(_W * h * 4, dtype=np.float32),
                                    format=dpg.mvFormat_Float_rgba, tag=_TEX_TAG)
            dpg.configure_item(self._ens_map_canvas, height=h)
            dpg.draw_image(_TEX_TAG, (0, 0), (_W, h), parent=self._ens_map_canvas)
            mv = MapView(_TEX_TAG, _W, h)
            writer, canvas = self._ens_writer, self._ens_map_canvas

            def stale():
                return (self._ens_map_writer is not writer
                        or self._ens_writer is not writer
                        or self._ens_map_canvas != canvas)

            def _load():
                error = ""
                try:
                    if not mv.load(gdf, cancelled=stale):
                        error = "Map unavailable."
                except Exception:
                    log.exception("Ensemble map could not be prepared")
                    error = "Map unavailable."

                def publish():
                    if stale() or not dpg.does_item_exist(canvas):
                        return
                    self._ens_map_loading = False
                    self._ens_map_error = error
                    if not error:
                        self._ens_map_view, self._ens_map_gdf_id = mv, id(gdf)
                self._queue_session_action(publish)
            threading.Thread(target=_load, name="ens-map", daemon=True).start()
        return False

    def _refresh_ens_map(self, force: bool = False) -> None:
        """Per frame: redraw when the chosen run, geometry, or theme changed."""
        if self._queue_session_action(self._refresh_ens_map, force):
            return
        if not dpg.does_item_exist(_MAP_TAG) or not dpg.is_item_shown(_MAP_TAG):
            return
        if self._ens_map_writer is not self._ens_writer:
            self._ens_map_writer = self._ens_writer
            self._ens_map_view = None
            self._ens_map_gdf_id = 0
            self._ens_map_sig = None
            self._ens_map_loading = False
            self._ens_map_error = ""
            self._ens_map_read_error = False
            dpg.set_value(self._ens_map_run, 1)
            dpg.delete_item(self._ens_map_canvas, children_only=True)
        count = self._sync_ens_map_runs()
        if not count:
            dpg.set_value(self._ens_map_note, "No runs yet.")
            return
        if not self._ens_map_geometry():
            dpg.set_value(self._ens_map_note, self._ens_map_error or "Drawing map...")
            return
        selected = dpg.get_value(self._ens_map_run)
        mode = self.theme.palette.name
        sig = (selected, self._ens_map_gdf_id, mode)
        dpg.set_value(self._ens_map_note, "Read failed." if self._ens_map_read_error
                      else f"of {count:,}")
        if sig == self._ens_map_sig and not force:
            return
        try:
            _, districts = self._ens_writer.assignments[selected - 1]
        except Exception:
            log.exception("Ensemble assignment could not be read")
            dpg.set_value(self._ens_map_note, "Read failed.")
            self._ens_map_read_error = True
            self._ens_map_view.draw_blank()
            self._ens_map_sig = sig
            return
        mv = self._ens_map_view
        r, g, b, _ = self.theme.color("child_bg")
        mv._bg_color = np.array([r, g, b, 255], dtype=np.uint8)
        n_dist = int(districts.max())
        mv.render_assignment((districts - 1).astype(np.int32), n_dist)
        self._ens_map_read_error = False
        dpg.set_value(self._ens_map_note, f"of {count:,}")
        self._ens_map_sig = sig
