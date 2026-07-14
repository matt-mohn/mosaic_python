"""MosaicApp -- the Dear PyGui application, assembled from focused mixins.

The class was split across sibling ``*_mixin.py`` modules purely for
navigability; at runtime it is one class with shared ``self`` state, so any
method may call any other via ``self``. Private-only extensions attach through
the optional ``_internal`` seam below and never ship to the public repo.
"""
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
        self.map_view: Optional[MapView] = None
        self._shp_dialog: Optional[ShapefileDialog] = None
        self.theme = ThemeManager(initial="light")

        # Stored after the user confirms the shapefile dialog
        self._loaded_config: Optional[ShapefileConfig] = None

        # Recent shapefiles (path + column config), loaded from disk in setup()
        self._recent_shapefiles: list = []   # [{"path": str, "config": dict}]
        self._file_save_asgn_item = 0        # File > Save Assignments menu item
        self._file_save_metrics_item = 0     # File > Save District Info menu item
        self._saved_plan = None              # last-saved assignment; drives the unsaved-changes guard
        # When set, the next inspection-complete event skips the column picker
        # and uses this config directly (one-click recent-file open).
        self._pending_recent_config: Optional[ShapefileConfig] = None
        # If True, auto-enable partisan overlay after the next Recent load completes.
        self._restore_partisan_on_load: bool = False

        # Map background-load tracking (app-local, no SharedState).
        # Tracking by (path, gdf id) so a re-import of the same path with
        # edited content still triggers a fresh MapView load — otherwise the
        # map keeps stale dimensions and render_assignment indexes past the
        # new assignment array.
        self._map_loading: bool = False
        self._map_ready: bool = False
        self._map_loaded_path: str = ""
        self._map_loaded_gdf_id: int = 0
        # gdf id whose complete_load has fully finished (pulsed via gdf_ready).
        # The map may only load a gdf that reached this point, so it never
        # captures the runner mid-populate. See the map bg-load gate below.
        self._map_data_gdf_id: int = 0

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
        auto-fits the content, so a dialog can never show a scrollbar or clip its
        text -- size it once and it always fits. ``height_hint`` is used only to
        centre the window vertically. Wrap long/dynamic text at ``width - 2 *
        _DIALOG_PAD`` so it never forces the window wider.

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
            # Auto-fit height, and pin the width to exactly w (min == max on the
            # x axis). The window can never scroll, clip vertically, or -- the
            # part that bites -- balloon sideways when a body item is wider than
            # expected, which would leave the right-aligned footer floating away
            # from the edge. Wrap long body text at w - 2 * _DIALOG_PAD so it
            # doesn't clip against the pinned width.
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
        dpg.set_viewport_resize_callback(self._align_photo_icons)
        _aligned = False
        while dpg.is_dearpygui_running():
            self._update_ui()
            dpg.render_dearpygui_frame()
            if not _aligned and dpg.get_frame_count() >= 2:
                self._align_photo_icons()
                _aligned = True
        dpg.destroy_context()

    # ── Frame update ──────────────────────────────────────────────────────────


def main():
    # The .bat / .command / .sh launchers invoke this entry point directly
    # via `python -m mosaic.gui.app`, bypassing mosaic:main. Wire up logging
    # here too so the 'mosaic' logger always has a console + file handler.
    from mosaic import _setup_logging
    _setup_logging()

    app = MosaicApp()
    app.setup()
    app.run()


if __name__ == "__main__":
    main()
