"""Phase plot: metric-vs-metric trajectory panel and its controls."""
from ._common import (
    _LEFT_W,
    _PHASE_ATTR,
    _PHASE_AXIS_UNIT,
    _PHASE_BANDS,
    _PHASE_BUDGET,
    _PHASE_FOLLOW,
    _PHASE_HEAD,
    _PHASE_KIND,
    _PHASE_LABELS,
    _PHASE_MIN_SPAN,
    _PHASE_RAMP,
    _PHASE_RAMP_LIGHT,
    _PHASE_SMOOTH_WIN,
    _PHASE_TAU,
    _phase_transform,
    _ramp_color,
    dpg,
    np,
)


class PhaseMixin:
    """Phase plot: metric-vs-metric trajectory panel and its controls."""

    def _build_phase_themes(self):
        """Recency-ramp line themes per palette, in faded and full-alpha variants
        (the Fade toggle), plus the palette's head-marker theme. Built for both
        palettes so a light/dark swap only rebinds -- see _phase_apply_fade."""
        self._phase_band_themes = {}    # palette -> {"fade": [...], "solid": [...]}
        self._phase_head_themes = {}    # palette -> marker theme
        for pal, ramp in (("dark", _PHASE_RAMP), ("light", _PHASE_RAMP_LIGHT)):
            fade_list, solid_list = [], []
            for k in range(_PHASE_BANDS):
                t = (k + 0.5) / _PHASE_BANDS
                rgb = _ramp_color(t, ramp)
                weight = 1.0 + 2.0 * t
                for alpha, store in ((int(75 + 180 * t), fade_list),
                                     (255, solid_list)):
                    with dpg.theme() as th:
                        with dpg.theme_component(dpg.mvLineSeries):
                            dpg.add_theme_color(dpg.mvPlotCol_Line, (*rgb, alpha),
                                                category=dpg.mvThemeCat_Plots)
                            dpg.add_theme_style(dpg.mvPlotStyleVar_LineWeight, weight,
                                                category=dpg.mvThemeCat_Plots)
                    store.append(th)
            self._phase_band_themes[pal] = {"fade": fade_list, "solid": solid_list}
            dot_fill = (0, 0, 0, 255) if pal == "dark" else (255, 255, 255, 255)
            dot_line = (255, 255, 255, 255) if pal == "dark" else (0, 0, 0, 255)
            with dpg.theme() as hd:
                with dpg.theme_component(dpg.mvScatterSeries):
                    dpg.add_theme_color(dpg.mvPlotCol_MarkerFill, dot_fill,
                                        category=dpg.mvThemeCat_Plots)
                    dpg.add_theme_color(dpg.mvPlotCol_MarkerOutline, dot_line,
                                        category=dpg.mvThemeCat_Plots)
                    dpg.add_theme_style(dpg.mvPlotStyleVar_Marker, dpg.mvPlotMarker_Circle,
                                        category=dpg.mvThemeCat_Plots)
                    dpg.add_theme_style(dpg.mvPlotStyleVar_MarkerSize, 4.0,
                                        category=dpg.mvThemeCat_Plots)
                    dpg.add_theme_style(dpg.mvPlotStyleVar_MarkerWeight, 2.0,
                                        category=dpg.mvThemeCat_Plots)
            self._phase_head_themes[pal] = hd

    def _build_phase_panel(self):
        """Comet Plot: a metric-vs-metric trajectory -- two axis pickers, a tail
        that cools with age, and a bright dot on the live plan."""
        self._build_phase_themes()
        with dpg.window(
            label="Comet Plot", tag="panel_phase", show=False,
            width=420, height=440, pos=[_LEFT_W + 100, 80],
            on_close=lambda: dpg.set_value(self._panel_phase_item, False),
        ):
            with dpg.group(horizontal=True):
                dpg.add_text("X")
                dpg.add_combo(_PHASE_LABELS, default_value="Compactness",
                              width=200, tag="phase_x_combo",
                              callback=self._on_phase_x_change)
            with dpg.group(horizontal=True):
                dpg.add_text("Y")
                dpg.add_combo(_PHASE_LABELS, default_value="Efficiency Gap",
                              width=200, tag="phase_y_combo",
                              callback=self._on_phase_y_change)
            with dpg.group(horizontal=True):            # options row (all on by default)
                dpg.add_checkbox(label="Smooth", default_value=True,
                                 callback=self._on_phase_smooth_change)
                dpg.add_checkbox(label="Fit all", default_value=True,
                                 callback=self._on_phase_fit_change)
                dpg.add_checkbox(label="Fade", default_value=True,
                                 callback=self._on_phase_fade_change)
            dpg.add_spacer(height=4)
            with dpg.plot(height=-1, width=-1, tag="phase_plot", no_menus=True):
                dpg.add_plot_axis(dpg.mvXAxis, label="Compactness", tag="phase_x")
                with dpg.plot_axis(dpg.mvYAxis, label="Efficiency Gap",
                                   tag="phase_y"):
                    for k in range(_PHASE_BANDS):
                        dpg.add_line_series([], [], label=f"##phase_band_{k}",
                                            tag=f"phase_band_{k}")
                    dpg.add_scatter_series([], [], label="##phase_head",
                                           tag="phase_head")
            # Stamp the default axis titles with their unit hints (later picks
            # route through _set_phase_axis, which does the same).
            self._set_phase_axis("x", self._phase_x_label)
            self._set_phase_axis("y", self._phase_y_label)
            self._phase_apply_fade()    # bind bands + head to current palette / fade

    def _set_phase_axis(self, which: str, label: str) -> None:
        """Point an axis at a metric (updates combo, stored label, axis title).
        The axis title carries a unit hint (e.g. "(%)") while the picker and the
        stored label stay bare -- see _PHASE_AXIS_UNIT."""
        self._phase_lim = None      # new metric/scale -> refit bounds
        unit = _PHASE_AXIS_UNIT.get(label)
        title = f"{label} {unit}" if unit else label
        if which == "x":
            self._phase_x_label = label
            dpg.set_value("phase_x_combo", label)
            dpg.configure_item("phase_x", label=title)
        else:
            self._phase_y_label = label
            dpg.set_value("phase_y_combo", label)
            dpg.configure_item("phase_y", label=title)

    def _on_phase_x_change(self, sender, app_data):
        self._set_phase_axis("x", app_data)

    def _on_phase_y_change(self, sender, app_data):
        self._set_phase_axis("y", app_data)

    def _on_phase_fit_change(self, sender, app_data):
        self._phase_fit_all = bool(app_data)
        self._phase_lim = None

    def _on_phase_fade_change(self, sender, app_data):
        self._phase_fade = bool(app_data)
        self._phase_apply_fade()

    def _phase_apply_fade(self) -> None:
        """Bind every trail band + head marker to the current palette's faded or
        full-alpha theme set. Called on Fade toggle and on light/dark swap."""
        pal = self.theme.palette.name
        if pal not in self._phase_band_themes:
            pal = "dark"
        bands = self._phase_band_themes[pal]["fade" if self._phase_fade else "solid"]
        for k in range(_PHASE_BANDS):
            dpg.bind_item_theme(f"phase_band_{k}", bands[k])
        dpg.bind_item_theme("phase_head", self._phase_head_themes[pal])

    def _on_phase_smooth_change(self, sender, app_data):
        self._phase_smooth = bool(app_data)

    def _on_panel_phase_toggle(self):
        dpg.configure_item("panel_phase",
                           show=dpg.get_value(self._panel_phase_item))

    def _phase_active_labels(self) -> list:
        """Metrics whose history carries real values right now.

        Non-partisan display metrics are 0.0 unless their weight (or a coupled
        weight / the pop-dev ratchet) is on, so they gate on the run's config --
        this keeps the picker from offering a metric that flatlines at 0. The
        coupling below mirrors scoring/score.py; keep the two in step when scores
        change. Partisan metrics are always computed once election data is
        loaded, so they gate on elections alone. Hinge is the exception: its
        value is P(party wins >= threshold), meaningless until the user
        establishes a hinge, so it also needs the Hinge score enabled. Cut Edges
        / Score / Temperature are intrinsic and always offered.
        """
        cfg = self.state.score_config
        has = getattr(self, "_has_elections", False)
        ratchet = self.state.tolerance_ratchet_mode not in (None, "off", "Off")
        active = set()
        if cfg.weight_holistic_compactness:
            active.add("holistic_compactness_history")
        if cfg.weight_polsby_popper or cfg.weight_holistic_compactness:
            active.add("pp_history")
        if cfg.weight_reock or cfg.weight_holistic_compactness:
            active.add("reock_history")
        if cfg.weight_holistic_splitting:
            active.add("holistic_splitting_history")
        if cfg.weight_county_excess or cfg.weight_county_unified:
            active.add("county_excess_splits_history")
        if cfg.weight_pop_deviation or ratchet:
            active.update(("pop_dev_max_history", "pop_dev_mean_history"))
        if has:
            # Always computed from the election data -> available with elections.
            active.update((
                "mm_history", "eg_history", "dem_seats_history",
                "partisan_bias_history",
                "partisan_gini_history",
                "holistic_proportionality_history",
                "inversion_history",
                "holistic_competitiveness_history",
                "majority_dem_history", "majority_rep_history",
            ))
            # Hinge needs a user-established threshold (the Hinge score toggle),
            # not just election data -- else its value is against a stale default.
            if dpg.get_value(self._hinge_enabled):
                active.add("hinge_history")
        always = {"cut_edges_history", "score_history", "temperature_history"}
        return [l for l in _PHASE_LABELS
                if _PHASE_ATTR[l] in always or _PHASE_ATTR[l] in active]

    def _phase_sync_available_metrics(self) -> None:
        """Rebuild the axis pickers to the metrics active for this run's config;
        bump a now-hidden axis to a safe intrinsic. Re-syncs only on a change
        (config is set at run start, so the picker tracks the live histories)."""
        labels = self._phase_active_labels()
        sig = tuple(labels)
        if self._phase_metric_sig == sig:
            return
        self._phase_metric_sig = sig
        dpg.configure_item("phase_x_combo", items=labels)
        dpg.configure_item("phase_y_combo", items=labels)
        if self._phase_x_label not in labels:
            self._set_phase_axis("x", "Cut Edges")
        if self._phase_y_label not in labels:
            self._set_phase_axis("y", "Score (total)")

    def _update_phase_plot(self) -> None:
        """Redraw the phase trajectory. Free when hidden; bounded to _PHASE_HEAD
        full-res recent points plus a strided tail (<= _PHASE_BUDGET)."""
        if not dpg.is_item_shown("panel_phase"):
            return
        self._phase_sync_available_metrics()
        xa, ya = _PHASE_ATTR[self._phase_x_label], _PHASE_ATTR[self._phase_y_label]
        with self.state._lock:
            xh = getattr(self.state, xa)
            yh = getattr(self.state, ya)
            n = min(len(xh), len(yh))
            if n == 0:
                xs_l = ys_l = []
                head_len = tail_n = stride = 0
            else:
                head_len = min(n, _PHASE_HEAD)
                tail_n = n - head_len
                budget_tail = max(1, _PHASE_BUDGET - head_len)
                # Power-of-two stride: it only ever doubles, so the sampled tail
                # indices stay nested -- points thin out, never reshuffle (no shimmer).
                stride = 1
                while tail_n // stride > budget_tail:
                    stride *= 2
                xs_l = (xh[0:tail_n:stride] if tail_n else []) + xh[tail_n:n]
                ys_l = (yh[0:tail_n:stride] if tail_n else []) + yh[tail_n:n]
        if n < self._phase_prev_n:          # run reset -> forget sticky bounds
            self._phase_lim = None
        self._phase_prev_n = n
        if not xs_l:
            for k in range(_PHASE_BANDS):
                dpg.set_value(f"phase_band_{k}", [[], []])
            dpg.set_value("phase_head", [[], []])
            return
        xs = _phase_transform(_PHASE_KIND[self._phase_x_label],
                              np.asarray(xs_l, dtype=np.float64))
        ys = _phase_transform(_PHASE_KIND[self._phase_y_label],
                              np.asarray(ys_l, dtype=np.float64))
        hx, hy = float(xs[-1]), float(ys[-1])           # dot = true current plan
        # Age (iterations back from the head) per plotted point, oldest first, so
        # recency t = exp(-age/TAU) is ascending -> bands are contiguous by t.
        tail_ages = ((n - 1) - np.arange(0, tail_n, stride)) if tail_n else np.empty(0)
        head_ages = np.arange(head_len - 1, -1, -1)
        ages = np.concatenate([tail_ages, head_ages]).astype(np.float64)
        t = np.exp(-ages / _PHASE_TAU)
        if self._phase_smooth and len(xs) >= _PHASE_SMOOTH_WIN:
            # Edge-aware moving average: normalise by the real (non-padded) window
            # overlap, else convolve's zero-padding drags both ends to the origin.
            k = np.ones(_PHASE_SMOOTH_WIN)
            norm = np.convolve(np.ones(len(xs)), k, mode="same")
            xs = np.convolve(xs, k, mode="same") / norm
            ys = np.convolve(ys, k, mode="same") / norm
        edges = np.searchsorted(t, [k / _PHASE_BANDS for k in range(1, _PHASE_BANDS)])
        edges = [0, *edges.tolist(), len(t)]
        for k in range(_PHASE_BANDS):
            a, b = edges[k], edges[k + 1]
            if k < _PHASE_BANDS - 1:
                b = min(b + 1, len(t))          # overlap one point so bands join
            dpg.set_value(f"phase_band_{k}",
                          [xs[a:b].tolist(), ys[a:b].tolist()] if b - a >= 1
                          else [[], []])
        dpg.set_value("phase_head", [[hx], [hy]])
        self._phase_fit_axes(xs, ys, ages, hx, hy)

    def _phase_fit_axes(self, xs, ys, ages, hx: float, hy: float) -> None:
        """Axis limits with margin. 'Fit all' holds sticky total bounds that only
        ever grow, so the view never jitters or shrinks mid-run; 'Follow dot'
        fits the last _PHASE_FOLLOW iterations. The live dot always stays inside.
        abs metrics anchor the axis floor at 0 (0 = perfectly fair)."""
        def _pad(lo, hi, floor0=False):
            lo, hi = min(lo, hi), max(lo, hi)
            if floor0:
                # 0 is the hard floor; keep min-span breathing room but grow it
                # upward from 0 instead of centering (which would dip below 0).
                hi = max(hi, _PHASE_MIN_SPAN)
                return 0.0, hi + hi * 0.08
            if hi - lo < _PHASE_MIN_SPAN:      # floor tiny ranges (Follow dot zoom)
                c = 0.5 * (lo + hi)
                lo, hi = c - _PHASE_MIN_SPAN / 2, c + _PHASE_MIN_SPAN / 2
            pad = (hi - lo) * 0.08
            return lo - pad, hi + pad

        fx = _PHASE_KIND[self._phase_x_label] == "abs"
        fy = _PHASE_KIND[self._phase_y_label] == "abs"
        if not self._phase_fit_all:
            m = ages <= _PHASE_FOLLOW
            xv, yv = xs[m], ys[m]
            xlo, xhi = _pad(min(float(xv.min()), hx), max(float(xv.max()), hx), fx)
            ylo, yhi = _pad(min(float(yv.min()), hy), max(float(yv.max()), hy), fy)
        else:
            dxlo, dxhi = min(float(xs.min()), hx), max(float(xs.max()), hx)
            dylo, dyhi = min(float(ys.min()), hy), max(float(ys.max()), hy)
            if self._phase_lim is None:
                lim = (dxlo, dxhi, dylo, dyhi)
            else:
                pxlo, pxhi, pylo, pyhi = self._phase_lim
                lim = (min(pxlo, dxlo), max(pxhi, dxhi),      # sticky: grow only,
                       min(pylo, dylo), max(pyhi, dyhi))      # never shrink
            self._phase_lim = lim
            xlo, xhi = _pad(lim[0], lim[1], fx)
            ylo, yhi = _pad(lim[2], lim[3], fy)
        dpg.set_axis_limits("phase_x", xlo, xhi)
        dpg.set_axis_limits("phase_y", ylo, yhi)
