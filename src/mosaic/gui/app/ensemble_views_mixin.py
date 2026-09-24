"""Ensemble pop-out views: histogram of one metric, scatterplot of two."""
from concurrent.futures import Future

from ._common import _ASSETS_DIR, dpg, log, np, threading

_HIST_TAG = "popup_ens_hist"
_SCATTER_TAG = "popup_ens_scatter"
_NOT_METRICS = {"run_id", "seed", "iterations", "seconds", "phase"}
_MAX_BINS = 40
_FREEZE_ALPHA = 0.45        # main window opacity while an ensemble runs
# One sequential hue; the lighter step reads on the dark plot background.
_BAR_RGBA = {"light": (52, 110, 180, 230), "dark": (110, 165, 230, 230)}


_LABELS = {
    "score": "Score",
    "cut_edges": "Cut Edges",
    "pop_dev_max_pct": "Population Deviation, Max (%)",
    "pop_dev_mean_pct": "Population Deviation, Mean (%)",
    "compactness": "Compactness",
    "polsby_popper": "Polsby-Popper",
    "reock": "Reock",
    "county_congruence_penalty": "County Congruence (penalty)",
    "county_excess_splits": "County Excess Splits",
    "county_unified_districts": "Single-County Districts",
    "mean_median": "Mean-Median",
    "efficiency_gap": "Efficiency Gap",
    "partisan_bias": "Partisan Bias",
    "partisan_gini_penalty": "Partisan Gini (penalty)",
    "expected_dem_seats": "Expected Dem Seats",
    "proportionality": "Proportionality",
    "inversion_chance": "Inversion Risk",
    "competitiveness": "Competitiveness",
    "dem_majority_chance": "Chance of Dem Majority",
    "hinge_chance": "Hinge Chance",
    "electoral_opportunity": "Electoral Opportunity",
    "opportunity_black": "Opportunity Districts, Black",
    "opportunity_latino": "Opportunity Districts, Latino",
    "opportunity_asian": "Opportunity Districts, Asian",
    "neighborhood_severance_penalty": "Neighborhood Severance (penalty)",
    "community_dispersion_penalty": "Community Dispersion (penalty)",
    "alignment_mean_retention": "Alignment, Mean Retention",
    "alignment_min_retention": "Alignment, Min Retention",
    # Targeting's tuned settings, recorded per run.
    "iterations": "Iterations",
    "n3_probability": "N3 Probability",
    "flip_midpoint": "Flip Midpoint",
    "temp_factor": "Initial Temp Factor",
    "guide_fraction": "Guide Point",
}


def _metric_label(col: str) -> str:
    return _LABELS.get(col) or col.replace("_", " ").capitalize()


def pearson_r(x: np.ndarray, y: np.ndarray) -> "float | None":
    """Pearson r, or None with fewer than 3 points or a constant axis."""
    if x.size < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _padded(v: np.ndarray, hi_extra: float = 0.0) -> tuple[float, float]:
    """Axis limits with a 6% margin; hi_extra widens the top/right side,
    where point labels sit."""
    lo, hi = float(v.min()), float(v.max())
    span = (hi - lo) or (abs(lo) * 0.8 or 16.0)
    return lo - span * 0.06, hi + span * (0.06 + hi_extra)


def hist_edges(v: np.ndarray) -> np.ndarray:
    """Bin edges: one bar per value for small-range integer metrics (cut
    edges, splits), otherwise numpy's 'auto' rule capped at _MAX_BINS."""
    lo, hi = float(v.min()), float(v.max())
    if (np.all(v == np.round(v)) and hi - lo < _MAX_BINS
            and max(abs(lo), abs(hi)) < 2**52):
        return np.arange(lo - 0.5, hi + 1.5)
    if hi == lo:
        d = abs(lo) * 0.01 or 0.5
        return np.array([lo - d, lo + d])
    iqr = float(np.subtract(*np.percentile(v, [75, 25])))
    width = 2 * iqr / np.cbrt(v.size)
    fd = int(np.ceil(min(_MAX_BINS, (hi - lo) / width))) if width > 0 else 1
    sturges = int(np.ceil(np.log2(v.size) + 1))
    return np.linspace(lo, hi, min(_MAX_BINS, max(fd, sturges)) + 1)


class EnsembleViewsMixin:
    """Ensemble pop-out views: histogram of one metric, scatterplot of two."""

    def _ens_columns(self) -> tuple[list[str], list[str]]:
        """Metric names for the current ensemble; no result rows are loaded."""
        w = self._ens_writer
        if w is None:
            return [], []
        cols = [c for c in w.columns if c not in _NOT_METRICS]
        if "phase" in w.columns:
            cols.insert(1, "iterations")
        return cols, [_metric_label(c) for c in cols]

    def _ens_revision(self):
        w = self._ens_writer
        return (id(w), len(w.rows) if w is not None else 0)

    def _ens_query(self, name, signature, query):
        """Poll one bounded result query per view without blocking a GUI frame.

        A changed request waits for its predecessor to finish before starting;
        obsolete results are discarded instead of building a worker backlog.
        """
        jobs = self.__dict__.setdefault("_ens_queries", {})
        used = self.__dict__.setdefault("_ens_query_used", {})
        job = jobs.get(name)
        if job is not None and job[0] == signature:
            used[name] = signature
            return (True, job[1].result()) if job[1].done() else (False, None)
        if job is not None and not job[1].done():
            return False, None
        if (job is not None and isinstance(signature, tuple)
                and isinstance(signature[0], tuple)
                and job[0][0][0] == signature[0][0] and job[0][1:] == signature[1:]):
            # Publish a coherent earlier count even if new runs arrived while
            # querying. The next frame requests the newer count; fast producers
            # cannot starve a view by continually invalidating completed work.
            used[name] = job[0]
            del jobs[name]
            return True, job[1].result()
        future = Future()
        jobs[name] = (signature, future)

        def work():
            try:
                result = query()
            except Exception as exc:
                log.exception("Ensemble results query failed")
                result = exc
            future.set_result(result)
        threading.Thread(target=work, name="ensemble-" + name, daemon=True).start()
        return False, None

    def _view_themes(self) -> dict:
        """Palette-keyed series themes shared by the views (built once)."""
        if not hasattr(self, "_view_theme_cache"):
            self._view_theme_cache = {}
            for mode, rgba in _BAR_RGBA.items():
                with dpg.theme() as t:
                    with dpg.theme_component(dpg.mvBarSeries):
                        dpg.add_theme_color(dpg.mvPlotCol_Fill, rgba,
                                            category=dpg.mvThemeCat_Plots)
                        dpg.add_theme_color(dpg.mvPlotCol_Line, (0, 0, 0, 0),
                                            category=dpg.mvThemeCat_Plots)
                    with dpg.theme_component(dpg.mvScatterSeries):
                        dpg.add_theme_color(dpg.mvPlotCol_MarkerFill,
                                            (*rgba[:3], 150),
                                            category=dpg.mvThemeCat_Plots)
                        dpg.add_theme_color(dpg.mvPlotCol_MarkerOutline,
                                            (*rgba[:3], 0),
                                            category=dpg.mvThemeCat_Plots)
                        dpg.add_theme_style(dpg.mvPlotStyleVar_Marker,
                                            dpg.mvPlotMarker_Circle,
                                            category=dpg.mvThemeCat_Plots)
                        dpg.add_theme_style(dpg.mvPlotStyleVar_MarkerSize, 3.5,
                                            category=dpg.mvThemeCat_Plots)
                self._view_theme_cache[mode] = t
        return self._view_theme_cache

    def _tiny_font(self):
        """10px Inter for point labels (0 if the font file is missing)."""
        if not hasattr(self, "_tiny_font_id"):
            self._tiny_font_id = 0
            path = _ASSETS_DIR / "fonts" / "inter" / "Inter-Regular.ttf"
            if path.exists():
                with dpg.font_registry():
                    self._tiny_font_id = dpg.add_font(str(path), 10)
        return self._tiny_font_id

    @staticmethod
    def _sync_combo(combo, labels: list[str], default: str) -> None:
        if dpg.get_item_configuration(combo)["items"] != labels:
            dpg.configure_item(combo, items=labels)
            if dpg.get_value(combo) not in labels and labels:
                dpg.set_value(combo, default if default in labels else labels[0])

    # ── Scatterplot ──────────────────────────────────────────────────────────

    def _on_open_ens_scatter(self) -> None:
        if self._queue_session_action(self._on_open_ens_scatter):
            return
        if dpg.does_item_exist(_SCATTER_TAG):
            dpg.configure_item(_SCATTER_TAG, show=True)
            dpg.focus_item(_SCATTER_TAG)
            return
        self._scatter_sig = None
        self._scatter_notes: list = []
        vp_w = dpg.get_viewport_client_width() or 1300
        with dpg.window(label="Ensemble Scatterplot", tag=_SCATTER_TAG,
                        width=560, height=500, pos=[max(20, vp_w - 600), 100],
                        min_size=[420, 320], no_collapse=True, no_scrollbar=True,
                        no_scroll_with_mouse=True):
            with dpg.group(horizontal=True):
                dpg.add_text("X:")
                self._scatter_xc = dpg.add_combo(
                    [], width=170, callback=lambda: self._refresh_ens_scatter(True))
                dpg.add_spacer(width=6)
                dpg.add_text("Y:")
                self._scatter_yc = dpg.add_combo(
                    [], width=170, callback=lambda: self._refresh_ens_scatter(True))
            self._scatter_r = self.theme.text("No runs yet.", "muted")
            with dpg.group(horizontal=True):
                self._scatter_trend = dpg.add_checkbox(
                    label="Trendline", default_value=False,
                    callback=lambda: self._refresh_ens_scatter(True))
                self._scatter_labels = dpg.add_checkbox(
                    label="Run Labels", default_value=False,
                    callback=lambda: self._refresh_ens_scatter(True))
            with dpg.tooltip(self._scatter_r):
                dpg.add_text("Correlation and trendline use all finite pairs. "
                             "Large ensembles show up to 2,000 evenly spaced runs.", wrap=300)
            with dpg.tooltip(self._scatter_labels):
                dpg.add_text("Run labels appear when there are at most 100 points.")
            with dpg.plot(height=-1, width=-1, no_menus=True, no_mouse_pos=True) \
                    as self._scatter_plot:
                self._scatter_x = dpg.add_plot_axis(dpg.mvXAxis, label="")
                with dpg.plot_axis(dpg.mvYAxis, label="") as self._scatter_y:
                    self._scatter_pts = dpg.add_scatter_series([], [])
                    self._scatter_fit = dpg.add_line_series([], [], label="##trend")
        self._refresh_ens_scatter(True)

    def _refresh_ens_scatter(self, force: bool = False) -> None:
        """Publish a completed query; unchanged frames do not read result data."""
        if self._queue_session_action(self._refresh_ens_scatter, force):
            return
        if not dpg.does_item_exist(_SCATTER_TAG) or not dpg.is_item_shown(_SCATTER_TAG):
            return
        cols, labels = self._ens_columns()
        self._sync_combo(self._scatter_xc, labels, "Compactness")
        self._sync_combo(self._scatter_yc, labels, "Efficiency Gap"
                         if "Efficiency Gap" in labels else "Cut Edges")
        lx, ly = dpg.get_value(self._scatter_xc), dpg.get_value(self._scatter_yc)
        mode = self.theme.palette.name
        trend = dpg.get_value(self._scatter_trend)
        annotated = dpg.get_value(self._scatter_labels)
        revision = self._ens_revision()
        sig = (revision, lx, ly, mode, trend, annotated)
        if sig == self._scatter_sig and not force:
            return
        w = self._ens_writer
        cx = cols[labels.index(lx)] if lx in labels else None
        cy = cols[labels.index(ly)] if ly in labels else None
        ready, data = self._ens_query("scatter", (revision, cx, cy),
                                      lambda: w.store.scatter(cx, cy, count=revision[1])
                                              if w else None)
        if not ready:
            dpg.set_value(self._scatter_r, "Updating...")
            if self._scatter_sig is None or self._scatter_sig[0][0] != revision[0]:
                dpg.set_value(self._scatter_pts, [[], []])
                dpg.set_value(self._scatter_fit, [[], []])
                for note in self._scatter_notes:
                    dpg.delete_item(note)
                self._scatter_notes = []
            return
        revision = self._ens_query_used["scatter"][0]
        self._scatter_sig = (revision, lx, ly, mode, trend, annotated)
        dpg.bind_item_theme(self._scatter_pts, self._view_themes()[mode])
        dpg.bind_item_theme(self._scatter_fit, self._partisan_ref_theme)
        dpg.set_value(self._scatter_fit, [[], []])
        dpg.configure_item(self._scatter_x, label=lx)
        dpg.configure_item(self._scatter_y, label=ly)
        for note in self._scatter_notes:
            dpg.delete_item(note)
        self._scatter_notes = []
        if data is None or isinstance(data, Exception):
            dpg.set_value(self._scatter_pts, [[], []])
            dpg.set_value(self._scatter_r, "Could not read results." if isinstance(data, Exception)
                          else "No finite pairs." if revision[1] else "No runs yet.")
            return
        x, y = data["points"].T
        dpg.set_value(self._scatter_pts, [x.tolist(), y.tolist()])
        show_labels = annotated and len(x) <= 100
        extra = .08 if show_labels else 0.0
        dpg.set_axis_limits(self._scatter_x, *_padded(data["xbounds"], extra))
        dpg.set_axis_limits(self._scatter_y, *_padded(data["ybounds"], extra))
        if show_labels:
            font = self._tiny_font()
            box = (255, 255, 255, 0) if mode == "light" else (0, 0, 0, 0)
            for rid, px, py in zip(data["ids"], x, y):
                note = dpg.add_plot_annotation(
                    label=rid, default_value=(float(px), float(py)),
                    offset=(5, -5), color=box, clamped=False, parent=self._scatter_plot)
                if font:
                    dpg.bind_item_font(note, font)
                self._scatter_notes.append(note)
        r = data["r"]
        if r is not None and trend:
            xs = data["xbounds"].tolist()
            dpg.set_value(self._scatter_fit,
                          [xs, [data["intercept"] + data["slope"] * v for v in xs]])
        count = data["n"]
        scope = f"{count:,} runs" + (f"; {len(x):,} shown" if len(x) < count else "")
        if annotated and not show_labels:
            scope += "; labels hidden"
        dpg.set_value(self._scatter_r, (f"r = {r:.2f}" if r is not None else "r = n/a")
                      + f"    ({scope})")

    # ── Histogram ────────────────────────────────────────────────────────────

    def _on_open_ens_hist(self) -> None:
        if self._queue_session_action(self._on_open_ens_hist):
            return
        if dpg.does_item_exist(_HIST_TAG):
            dpg.configure_item(_HIST_TAG, show=True)
            dpg.focus_item(_HIST_TAG)
            return
        self._hist_sig = None
        self._hist_edges = None
        self._hist_counts = None
        vp_w = dpg.get_viewport_client_width() or 1300
        with dpg.window(label="Ensemble Histograms", tag=_HIST_TAG,
                        width=600, height=460, pos=[max(20, vp_w - 640), 60],
                        min_size=[420, 300],   # fits the two stats rows
                        no_collapse=True, no_scrollbar=True,
                        no_scroll_with_mouse=True):
            with dpg.group(horizontal=True):
                dpg.add_text("Metric:")
                self._hist_metric = dpg.add_combo(
                    [], width=260, callback=lambda: self._refresh_ens_hist(True))
            self._hist_stats = self.theme.text("No completed runs yet.", "muted")
            self._hist_stats2 = self.theme.text("", "muted")
            # Leave room below the plot for the one-line hover readout.
            with dpg.plot(height=-30, width=-1, no_menus=True, no_mouse_pos=True,
                          tag="ens_hist_plot"):
                self._hist_x = dpg.add_plot_axis(dpg.mvXAxis, label="")
                with dpg.plot_axis(dpg.mvYAxis, label="Runs") as self._hist_y:
                    self._hist_bars = dpg.add_bar_series([], [], weight=1.0)
                    self._hist_median = dpg.add_inf_line_series(
                        [], label="##median", tag="ens_hist_median")
            with dpg.tooltip(self._hist_stats):
                dpg.add_text("All finite values are included. The vertical line marks the median.")
            self._hist_hover = self.theme.text("", "muted")
        self._refresh_ens_hist(True)

    def _refresh_ens_hist(self, force: bool = False) -> None:
        """Histogram statistics cover all finite values; queries run off-frame."""
        if self._queue_session_action(self._refresh_ens_hist, force):
            return
        if not dpg.does_item_exist(_HIST_TAG) or not dpg.is_item_shown(_HIST_TAG):
            return
        cols, labels = self._ens_columns()
        self._sync_combo(self._hist_metric, labels, "Compactness")
        label = dpg.get_value(self._hist_metric)
        mode = self.theme.palette.name
        revision = self._ens_revision()
        sig = (revision, label, mode)
        if sig == self._hist_sig and not force:
            self._update_ens_hist_hover()
            return
        col = cols[labels.index(label)] if label in labels else None
        w = self._ens_writer
        ready, data = self._ens_query("hist", (revision, col),
                                      lambda: w.store.histogram(col, count=revision[1])
                                              if w else None)
        if not ready:
            dpg.set_value(self._hist_stats, "Updating...")
            if self._hist_sig is None or self._hist_sig[0][0] != revision[0]:
                dpg.set_value(self._hist_bars, [[], []])
                dpg.set_value(self._hist_median, [[]])
                dpg.set_value(self._hist_stats2, "")
                self._hist_edges = self._hist_counts = None
                dpg.set_value(self._hist_hover, "")
            return
        revision = self._ens_query_used["hist"][0]
        self._hist_sig = (revision, label, mode)
        dpg.bind_item_theme(self._hist_bars, self._view_themes()[mode])
        dpg.bind_item_theme(self._hist_median, self._partisan_ref_theme)
        dpg.configure_item(self._hist_x, label=label)
        if data is None or isinstance(data, Exception):
            dpg.set_value(self._hist_bars, [[], []])
            dpg.set_value(self._hist_median, [[]])
            dpg.set_value(self._hist_stats, "Could not read results." if isinstance(data, Exception)
                          else "No finite values." if revision[1] else "No runs yet.")
            dpg.set_value(self._hist_stats2, "")
            self._hist_edges = self._hist_counts = None
            dpg.set_value(self._hist_hover, "")
            return
        edges, counts = data["edges"], data["counts"]
        centers = (edges[:-1] + edges[1:]) / 2
        dpg.configure_item(self._hist_bars, weight=float(edges[1] - edges[0]) * .9)
        dpg.set_value(self._hist_bars, [centers.tolist(), counts.astype(float).tolist()])
        dpg.set_value(self._hist_median, [[data["median"]]])
        dpg.set_axis_limits(self._hist_x, float(edges[0]), float(edges[-1]))
        ticks = _count_ticks(int(counts.max()))
        dpg.set_axis_limits(self._hist_y, 0.0, ticks[-1] * 1.05)
        dpg.set_axis_ticks(self._hist_y, tuple((str(t), float(t)) for t in ticks))
        f = _formatter_bounds(data["lo"], data["hi"], data["mean"], data["integral"])
        self._hist_fmt = f
        dpg.set_value(self._hist_stats,
                      f"{data['n']:,} runs    median {f(data['median'])}    mean {f(data['mean'])}")
        dpg.set_value(self._hist_stats2,
                      f"range {f(data['lo'])} to {f(data['hi'])}    sd {f(data['sd'])}")
        self._hist_edges, self._hist_counts = edges, counts
        self._update_ens_hist_hover()

    def _update_ens_hist_hover(self) -> None:
        """Readout for the bin under the mouse."""
        e = self._hist_edges
        text = ""
        if e is not None and dpg.is_item_hovered("ens_hist_plot"):
            x = dpg.get_plot_mouse_pos()[0]
            i = int(np.searchsorted(e, x, side="right")) - 1
            if x == e[-1]:
                i = len(e) - 2
            if 0 <= i < len(self._hist_counts):
                n = int(self._hist_counts[i])
                f = self._hist_fmt
                text = (f"{f(e[i])} to {f(e[i + 1])}: "
                        f"{n:,} run{'s' if n != 1 else ''}")
        dpg.set_value(self._hist_hover, text)

    # ── Main-window freeze while an ensemble runs ────────────────────────────

    def _freeze_main(self, on: bool) -> None:
        """Dim the main window and disable its menus. Its action handlers also
        return early while _ensemble_active, so nothing can disturb the run;
        pop-out views stay live."""
        if not hasattr(self, "_freeze_theme"):
            with dpg.theme() as self._freeze_theme:
                with dpg.theme_component(dpg.mvAll):
                    dpg.add_theme_style(dpg.mvStyleVar_Alpha, _FREEZE_ALPHA,
                                        category=dpg.mvThemeCat_Core)
        dpg.bind_item_theme("main_window", self._freeze_theme if on else 0)
        for menu in dpg.get_item_children("main_menu_bar", 1) or []:
            dpg.configure_item(menu, enabled=not on)
        # The map is a drawn image, which style alpha skips; tint it to match.
        if dpg.does_item_exist("map_image"):
            tint = (255, 255, 255, round(255 * _FREEZE_ALPHA) if on else 255)
            key = "tint_color" if self._map_native_input else "color"
            dpg.configure_item("map_image", **{key: tint})


def _formatter(vals: np.ndarray):
    """One decimal count for every number shown for a metric, from its spread:
    whole numbers stay whole, 54.0-69.1 gets 1 place, 0.035-0.183 gets 3."""
    return _formatter_bounds(float(vals.min()), float(vals.max()), float(vals.mean()),
                             bool(np.all(vals == np.round(vals))))


def _formatter_bounds(lo, hi, mean, integral):
    if integral:
        places = 0
    else:
        spread = hi - lo or abs(mean) or 1.0
        places = int(min(4, max(1, 2 - np.floor(np.log10(spread)))))
    return lambda x: f"{float(x):,.{places}f}"


def _count_ticks(peak: int) -> list[int]:
    """Whole-number y ticks from 0 up past the tallest bar, in 1/2/5 x 10^k
    steps, about five of them."""
    raw = max(1.0, peak / 5)
    mag = 10 ** np.floor(np.log10(raw))
    step = int(next(m * mag for m in (1, 2, 5, 10) if m * mag >= raw))
    top = -(-max(peak, 1) // step) * step
    return list(range(0, top + 1, step))
