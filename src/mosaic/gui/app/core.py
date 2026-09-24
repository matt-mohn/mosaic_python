"""MosaicApp, the Dear PyGui application composed from focused mixins.

The sibling ``*_mixin.py`` modules form one runtime class with shared ``self``
state. Private extensions may attach through the optional ``_internal`` seam.
"""
from queue import Empty, SimpleQueue

from ._common import (
    _DIALOG_BTN_W,
    _DIALOG_GAP,
    _DIALOG_PAD,
    _DIALOG_RM,
    _VP_H,
    _VP_W,
    AlgorithmRunner,
    MapView,
    Optional,
    ShapefileConfig,
    ShapefileDialog,
    SharedState,
    ThemeManager,
    _SeriesBuffer,
    contextmanager,
    dpg,
    threading,
)
from .ensemble_map_mixin import EnsembleMapMixin
from .ensemble_mixin import EnsembleMixin
from .ensemble_roster_mixin import EnsembleRosterMixin
from .ensemble_views_mixin import EnsembleViewsMixin
from .export_mixin import ExportMixin
from .io_mixin import IOMixin
from .map_mixin import MapMixin
from .menu_mixin import MenuMixin
from .panels_mixin import PanelsMixin
from .phase_mixin import PhaseMixin
from .popups_mixin import PopupsMixin
from .runner_mixin import RunnerMixin
from .setup_mixin import SetupMixin
from .toggles_mixin import TogglesMixin
from .updates_mixin import UpdatesMixin

# App mixins compose one class; order is irrelevant (no method names overlap).
_APP_MIXINS = (
    SetupMixin,
    PopupsMixin,
    PanelsMixin,
    PhaseMixin,
    UpdatesMixin,
    TogglesMixin,
    MapMixin,
    IOMixin,
    RunnerMixin,
    ExportMixin,
    MenuMixin,
    EnsembleMixin,
    EnsembleViewsMixin,
    EnsembleRosterMixin,
    EnsembleMapMixin,
)

try:
    from ._internal import INTERNAL_MIXINS  # private-only; absent in public checkout
except ImportError:
    INTERNAL_MIXINS = ()


class MosaicApp(*INTERNAL_MIXINS, *_APP_MIXINS):
    """Main application -- Dear PyGui interface coordinating the algorithm thread."""

    def __init__(self):
        self.state = SharedState()
        self.runner: Optional[AlgorithmRunner] = None
        self.algorithm_thread: Optional[threading.Thread] = None
        self._data_thread: Optional[threading.Thread] = None
        self._map_load_thread: Optional[threading.Thread] = None
        self._pending_session_action = None
        self._gui_thread_id = threading.get_ident()
        self._session_requests = SimpleQueue()
        self.map_view: Optional[MapView] = None
        self._shp_dialog: Optional[ShapefileDialog] = None
        self.theme = ThemeManager(initial="light")

        # Stored after the user confirms the shapefile dialog
        self._loaded_config: Optional[ShapefileConfig] = None

        # Recent shapefiles (path + column config), loaded from disk in setup()
        self._recent_shapefiles: list = []   # [{"path": str, "config": dict}]
        self._recent_presets: list = []      # preset file paths, newest first
        # Ensemble tool: while active, run() refreshes only the ensemble window.
        self._ensemble_active = False
        self._ens_writer = None
        self._ens_keep_best = False          # Ensemble Advanced > Keep best map
        self._ens_targeting = False          # Ensemble Advanced > Targeting
        self._ens_target_tail = False        # ... > Favor lower-tail scores
        self._ensemble_item = 0              # Advanced > Ensemble... item
        self._min_width_set = False          # window min width corrected for the frame
        self._file_save_asgn_item = 0        # File > Save Assignments menu item
        self._file_save_metrics_item = 0     # File > Save District Info menu item
        # last-saved assignment; drives the unsaved-changes guard
        self._saved_plan = None
        # When set, the next inspection-complete event skips the column picker
        # and uses this config directly (one-click recent-file open).
        self._pending_recent_config: Optional[ShapefileConfig] = None
        # One-shot marker for a Recent load configured with election columns;
        # consumed after map-fill availability is synchronized.
        self._restore_partisan_on_load: bool = False

        # Map background-load tracking (app-local, no SharedState). Path and
        # GeoDataFrame identity jointly distinguish reloads of edited files.
        self._map_loading: bool = False
        self._map_ready: bool = False
        self._map_loaded_path: str = ""
        self._map_loaded_gdf_id: int = 0
        # gdf id whose complete_load has fully finished (pulsed via gdf_ready).
        # The map may only load a gdf that reached this point, so it never
        # captures the runner mid-populate. See the map bg-load gate below.
        self._map_data_gdf_id: int = 0
        self._init_map_navigation()

        # Plot appearance toggle (app-local)
        self._limit_plots: int | str = ""   # DPG checkbox tag, set during setup

        self._contrib_bar_series: list = []

        # Track what data the current shapefile has
        self._has_elections: bool = False

        # Relight: a "continue-refining" mode that reseeds each run from the
        # current on-screen map (mutually exclusive with Hot Start). _saved holds
        # the annealing-control values captured when armed, restored on clear.
        self._relight_active: bool = False
        self._relight_saved: Optional[dict] = None

        # Tracks whether the last frame was in a "running" state, so we can
        # trigger a one-shot precise-label re-render on transitions out of
        # running (cheap centroid labels -> pole-of-inaccessibility labels).
        self._labels_were_fast: bool = False

        # Renumber settings. Source of truth for the Advanced > Renumber controls.
        # _renumber_enabled mirrors the "Renumber districts after run" check and
        # the radio's None option; _renumber_rule is the last chosen sweep.
        self._renumber_enabled: bool = True
        self._renumber_rule: str = "proximity"  # "nw_se" | "n_s" | "proximity"
        # Cached precinct centroids (x, y) in gdf CRS, keyed by gdf identity, so
        # repeated renumbers don't recompute geometry.centroid each time.
        self._renumber_centroids = None         # (gdf_id, x_arr, y_arr) | None

        # Local history buffers — incremental delta copy, self-compacting
        self._buf_score     = _SeriesBuffer()
        self._buf_acc       = _SeriesBuffer()
        self._buf_temp      = _SeriesBuffer()
        self._buf_cs_score  = _SeriesBuffer()
        self._buf_cs_excess = _SeriesBuffer()
        self._buf_cs_clean  = _SeriesBuffer()
        self._buf_mm        = _SeriesBuffer()
        self._buf_eg        = _SeriesBuffer()
        self._buf_pb        = _SeriesBuffer()
        self._buf_pg        = _SeriesBuffer()
        self._buf_seats     = _SeriesBuffer()
        self._buf_pp        = _SeriesBuffer()
        self._buf_reock     = _SeriesBuffer()
        self._buf_hc        = _SeriesBuffer()
        self._buf_hsplit    = _SeriesBuffer()
        self._buf_hprop     = _SeriesBuffer()
        self._buf_hcmp      = _SeriesBuffer()
        self._buf_rep_black  = _SeriesBuffer()
        self._buf_rep_latino = _SeriesBuffer()
        self._buf_rep_asian  = _SeriesBuffer()
        self._buf_rep_black_seats  = _SeriesBuffer()
        self._buf_rep_latino_seats = _SeriesBuffer()
        self._buf_rep_asian_seats  = _SeriesBuffer()
        self._buf_coh_black  = _SeriesBuffer()
        self._buf_coh_latino = _SeriesBuffer()
        self._buf_coh_asian  = _SeriesBuffer()
        self._buf_cong_black  = _SeriesBuffer()
        self._buf_cong_latino = _SeriesBuffer()
        self._buf_cong_asian  = _SeriesBuffer()
        self._buf_rep_overall = _SeriesBuffer()   # aggregate representation penalty
        self._buf_coh_overall = _SeriesBuffer()   # aggregate cohesion penalty
        self._buf_cong_overall = _SeriesBuffer()  # aggregate congruence penalty
        self._buf_popdev     = _SeriesBuffer()
        self._buf_popdev_max = _SeriesBuffer()
        self._buf_popdev_mean = _SeriesBuffer()
        self._buf_align_mean = _SeriesBuffer()
        self._buf_align_min  = _SeriesBuffer()
        self._buf_cuts      = _SeriesBuffer()
        self._buf_maj_dem   = _SeriesBuffer()
        self._buf_maj_rep   = _SeriesBuffer()
        self._buf_hinge     = _SeriesBuffer()
        self._buf_inversion = _SeriesBuffer()

        # Phase plot: selected metric labels + view / smoothing prefs.
        self._phase_x_label = "Compactness"
        self._phase_y_label = "Efficiency Gap"
        self._phase_fit_all = True         # Fit all (sticky) vs Follow dot
        self._phase_smooth = True
        self._phase_fade = True            # alpha fade with age vs solid trail
        self._phase_metric_sig = None      # last active-metric set the combos synced to
        self._phase_lim = None             # sticky (xlo,xhi,ylo,yhi) for "Fit all"
        self._phase_prev_n = 0             # detect a run reset (history shrank)

    # ── Setup ─────────────────────────────────────────────────────────────────

    def _dialog_pos(self, w: int, h: int) -> list:
        """Centre a w x h window on the live viewport (falls back to the design
        viewport size if it can't be queried yet)."""
        try:
            vp_w = dpg.get_viewport_client_width()
            vp_h = dpg.get_viewport_client_height()
        except Exception:
            vp_w, vp_h = _VP_W, _VP_H
        return [max(20, (vp_w - w) // 2), max(20, (vp_h - h) // 2)]

    @contextmanager
    def _dialog(self, title: str, tag: str, size, *,
                primary=None, secondary=None, buttons=None, show: bool = True,
                autosize: bool = True, modal: bool = True):
        """Standard modal dialog: locked chrome, **auto-fit height**, fixed width,
        centred, with a right-aligned themed footer built after the body.

        ``size`` is ``(width, height_hint)``. Width is enforced (``min_size`` +
        ``autosize``) so wrap widths and footer alignment stay predictable; height
        follows the content. ``height_hint`` is used only to centre the window
        vertically. Wrap long/dynamic text at ``width - 2 * _DIALOG_PAD`` to keep
        it within the fixed width. Long bodies need a scrollable child or an
        explicitly bounded window to fit short displays.

        Footer buttons come from either ``primary``/``secondary`` ``(label,
        callback)`` tuples (primary blue, secondary grey) or a general ``buttons``
        list for 3+ buttons -- each item ``(label, callback)`` (grey) or
        ``(label, callback, "primary")`` (blue). Body widgets go in the ``with``
        block. Callers own the lifecycle via the callbacks they pass (hide with
        ``configure_item(tag, show=False)`` for build-once dialogs, or
        ``delete_item(tag)`` for transient ones)."""
        w, h = size
        if dpg.does_item_exist(tag):
            dpg.delete_item(tag)
        win_kwargs = dict(label=title, tag=tag, modal=modal, no_close=True,
                          no_collapse=True, no_resize=True, no_scrollbar=True,
                          show=show, pos=self._dialog_pos(w, h))
        if autosize:
            # Fix width while height follows the body. Callers wrap dynamic
            # text so it fits without moving the footer beyond the fixed edge.
            win_kwargs.update(autosize=True, min_size=[w, 1],
                              max_size=[w, 100_000])
        else:
            # Fixed size (readers that scroll their own inner child_window).
            win_kwargs.update(width=w, height=h)
        with dpg.window(**win_kwargs):
            yield
            self._dialog_footer(w, primary=primary, secondary=secondary,
                                buttons=buttons)

    @staticmethod
    def _dialog_btn_w(label: str) -> int:
        """Fit a footer button to its label while holding short labels to a
        uniform minimum. Estimated (not measured), so it works even for dialogs
        built before the first frame is rendered."""
        return max(_DIALOG_BTN_W, 8 * len(label) + 26)

    def _dialog_footer(self, w: int, *, primary=None, secondary=None,
                       buttons=None) -> None:
        """Right-aligned footer row: separator, then the buttons in order, the
        primary blue (nudge) and the rest grey (anti-nudge)."""
        rows = []
        if buttons is not None:
            for b in buttons:
                rows.append((b[0], b[1], len(b) > 2 and b[2] == "primary"))
        else:
            if primary:
                rows.append((primary[0], primary[1], True))
            if secondary:
                rows.append((secondary[0], secondary[1], False))
        if not rows:
            return
        dpg.add_spacer(height=8)
        dpg.add_separator()
        dpg.add_spacer(height=6)
        widths = [self._dialog_btn_w(r[0]) for r in rows]
        block = sum(widths) + (len(rows) - 1) * _DIALOG_GAP
        content_w = w - 2 * _DIALOG_PAD
        lead = max(0, content_w - _DIALOG_RM - block)
        with dpg.group(horizontal=True):
            if lead:
                dpg.add_spacer(width=lead)
            for (label, callback, is_primary), bw in zip(rows, widths):
                btag = dpg.add_button(label=label, width=bw, callback=callback)
                dpg.bind_item_theme(
                    btag,
                    self.theme.nudge_theme if is_primary
                    else self.theme.antinudge_theme,
                )

    def run(self):
        dpg.show_viewport()
        try:
            while dpg.is_dearpygui_running():
                while True:
                    try:
                        request = self._session_requests.get_nowait()
                    except Empty:
                        break
                    self._perform_session_action(request)
                self._update_window_layout()
                if self._ensemble_active:
                    self._update_ensemble_ui()
                else:
                    self._tick_session_action()
                    if not self._session_action_pending():
                        self._update_ui()
                self._refresh_ens_hist()
                self._refresh_ens_scatter()
                self._refresh_ens_roster()
                self._refresh_ens_map()
                dpg.render_dearpygui_frame()
        finally:
            if self._ensemble_active:
                with self._ens_progress.lock:
                    self._ens_progress.stop_requested = True
            self.state.request_stop()
            self._shutdown_map_navigation()
            dpg.destroy_context()

    # ── Frame update ──────────────────────────────────────────────────────────


def main():
    # Direct module launches need the same console logging as mosaic:main.
    from mosaic import _setup_logging
    _setup_logging()

    app = MosaicApp()
    app.setup()
    app.run()


if __name__ == "__main__":
    main()
