"""Score/metric side-panel builders and reference-line themes."""
from ._common import _CONTRIB_BAR_METRICS, _LEFT_W, dpg


class PanelsMixin:
    """Score/metric side-panel builders and reference-line themes."""

    def _build_temperature_panel(self):
        with dpg.window(
            label="Temperature",
            tag="panel_temperature",
            show=False,
            width=520, height=280,
            pos=[_LEFT_W + 60, 80],
            on_close=self._on_temp_panel_close,
        ):
            with dpg.plot(height=-1, width=-1, tag="panel_temp_plot", no_menus=True):
                dpg.add_plot_legend()
                dpg.add_plot_axis(
                    dpg.mvXAxis, label="Iteration", tag="panel_temp_x",
                )
                with dpg.plot_axis(
                    dpg.mvYAxis, label="Temp", tag="panel_temp_y",
                ):
                    dpg.add_line_series([], [], label="Temp",
                                        tag="panel_temp_series")
    def _build_county_splits_panel(self):
        with dpg.window(
            label="Classic Splitting", tag="panel_county_splits",
            show=False, width=540, height=460,
            pos=[_LEFT_W + 80, 80],
            on_close=lambda: dpg.set_value(self._panel_cs_item, False),
        ):
            with dpg.group(tag="cs_charts_grp"):
                # Two panels, one per real county score -- there is no single
                # combined "county splits score", so it isn't charted.
                self.theme.text("Excess Splits", "heading")
                with dpg.plot(height=165, width=-1, no_menus=True):
                    dpg.add_plot_legend()
                    dpg.add_plot_axis(dpg.mvXAxis, label="Iteration", tag="cs_excess_x")
                    with dpg.plot_axis(dpg.mvYAxis, label="Count", tag="cs_excess_y"):
                        dpg.add_line_series([], [], label="Excess", tag="cs_excess_series")
                dpg.add_spacer(height=6)
                self.theme.text("Single-County Districts", "heading")
                with dpg.plot(height=165, width=-1, no_menus=True):
                    dpg.add_plot_legend()
                    dpg.add_plot_axis(dpg.mvXAxis, label="Iteration", tag="cs_clean_x")
                    with dpg.plot_axis(dpg.mvYAxis, label="Count", tag="cs_clean_y"):
                        dpg.add_line_series([], [], label="Single-County", tag="cs_clean_series")
                        _cs_max = dpg.add_line_series(
                            [], [], label="##cs_max", tag="cs_clean_max",
                        )
                        dpg.bind_item_theme(_cs_max, self._partisan_ref_theme)
                self.theme.track(
                    dpg.add_text("", tag="cs_max_clean_note"),
                    "success_soft",
                )
            self._tooltip(
                "cs_charts_grp",
                "Fewer Excess Splits is better; more Single-County Districts is better.",
            )
            self.theme.track(
                dpg.add_text(
                    "Turn on this score to chart it.",
                    tag="cs_inactive_lbl", show=False,
                ),
                "muted",
            )

    def _build_ref_line_themes(self):
        """Build the palette-aware reference-line themes used across panels
        (50/50 guides, median guides, max-feasible line, etc.). Must run
        BEFORE any panel that binds these themes."""
        self._partisan_ref_themes: dict[str, int] = {}
        for mode, rgba in (("dark",  (255, 255, 255, 140)),
                           ("light", (0,   0,   0,   140))):
            with dpg.theme() as t:
                with dpg.theme_component(dpg.mvLineSeries):
                    dpg.add_theme_color(
                        dpg.mvPlotCol_Line, rgba,
                        category=dpg.mvThemeCat_Plots,
                    )
                    dpg.add_theme_style(
                        dpg.mvPlotStyleVar_LineWeight, 1.0,
                        category=dpg.mvThemeCat_Plots,
                    )
            self._partisan_ref_themes[mode] = t
        self._partisan_ref_theme = self._partisan_ref_themes[
            self.theme.palette.name
        ]

    def _build_partisanship_panel(self):
        # One bar theme per partisan-palette bucket (using map_view palette)
        from mosaic.gui.map_view import _PARTISAN_RGBA
        self._partisan_bar_themes = []
        for rgba in _PARTISAN_RGBA:
            with dpg.theme() as t:
                with dpg.theme_component(dpg.mvBarSeries):
                    dpg.add_theme_color(
                        dpg.mvPlotCol_Fill,
                        (int(rgba[0]), int(rgba[1]), int(rgba[2]), 220),
                        category=dpg.mvThemeCat_Plots,
                    )
            self._partisan_bar_themes.append(t)

        with dpg.window(
            label="Partisanship", tag="panel_partisanship",
            show=False, width=540, height=320,
            pos=[_LEFT_W + 80, 80],
            on_close=lambda: dpg.set_value(self._panel_partisan_item, False),
        ):
            with dpg.plot(height=-1, width=-1, tag="partisan_plot", no_menus=True):
                dpg.add_plot_axis(
                    dpg.mvXAxis, tag="partisan_x",
                    label="Districts (least to most Democratic)",
                    no_tick_labels=True,
                )
                with dpg.plot_axis(dpg.mvYAxis, label="D Vote Share (%)", tag="partisan_y"):
                    self._partisan_bar_series = []
                    for i in range(len(self._partisan_bar_themes)):
                        s = dpg.add_bar_series([], [], weight=0.85, show=True)
                        dpg.bind_item_theme(s, self._partisan_bar_themes[i])
                        self._partisan_bar_series.append(s)
                    # Reference lines drawn after bars so they render on top;
                    # ##-prefix suppresses legend entries.
                    _ref = dpg.add_line_series(
                        [0, 200], [50.0, 50.0], label="##ref50", tag="partisan_ref",
                    )
                    dpg.bind_item_theme(_ref, self._partisan_ref_theme)
                    _med = dpg.add_line_series(
                        [], [], label="##refmed", tag="partisan_median",
                    )
                    dpg.bind_item_theme(_med, self._partisan_ref_theme)
        dpg.set_axis_limits("partisan_y", 0.0, 100.0)

    def _build_win_chance_panel(self):
        with dpg.window(
            label="Win Chance", tag="panel_win_chance",
            show=False, width=540, height=320,
            pos=[_LEFT_W + 120, 110],
            on_close=lambda: dpg.set_value(self._panel_win_chance_item, False),
        ):
            with dpg.plot(height=-1, width=-1, tag="win_chance_plot", no_menus=True):
                dpg.add_plot_axis(
                    dpg.mvXAxis, tag="win_chance_x",
                    label="Districts (least to most likely D win)",
                    no_tick_labels=True,
                )
                with dpg.plot_axis(dpg.mvYAxis, label="P(D wins) (%)", tag="win_chance_y"):
                    self._win_chance_bar_series = []
                    for i in range(len(self._partisan_bar_themes)):
                        s = dpg.add_bar_series([], [], weight=0.85, show=True)
                        dpg.bind_item_theme(s, self._partisan_bar_themes[i])
                        self._win_chance_bar_series.append(s)
                    _wref = dpg.add_line_series(
                        [0, 200], [50.0, 50.0], label="##wref50", tag="win_chance_ref",
                    )
                    dpg.bind_item_theme(_wref, self._partisan_ref_theme)
                    _wmed = dpg.add_line_series(
                        [], [], label="##wrefmed", tag="win_chance_median",
                    )
                    dpg.bind_item_theme(_wmed, self._partisan_ref_theme)
        dpg.set_axis_limits("win_chance_y", 0.0, 100.0)

    def _build_mm_panel(self):
        with dpg.window(
            label="Mean-Median Difference", tag="panel_mm",
            show=False, width=500, height=280,
            pos=[_LEFT_W + 80, 80],
            on_close=lambda: dpg.set_value(self._panel_mm_item, False),
        ):
            with dpg.group(tag="mm_plot_grp"):
                with dpg.plot(height=-1, width=-1, no_menus=True):
                    dpg.add_plot_legend()
                    dpg.add_plot_axis(dpg.mvXAxis, label="Iteration", tag="mm_x")
                    with dpg.plot_axis(dpg.mvYAxis, label="Mean-Median", tag="mm_y"):
                        dpg.add_line_series([], [], label="MM", tag="mm_series")
            self.theme.track(
                dpg.add_text(
                    "Load election data to use this panel.",
                    tag="mm_inactive_lbl", show=False,
                ),
                "muted",
            )

    def _build_eg_panel(self):
        with dpg.window(
            label="Efficiency Gap", tag="panel_eg",
            show=False, width=500, height=280,
            pos=[_LEFT_W + 80, 80],
            on_close=lambda: dpg.set_value(self._panel_eg_item, False),
        ):
            with dpg.group(tag="eg_plot_grp"):
                with dpg.plot(height=-1, width=-1, no_menus=True):
                    dpg.add_plot_legend()
                    dpg.add_plot_axis(dpg.mvXAxis, label="Iteration", tag="eg_x")
                    with dpg.plot_axis(dpg.mvYAxis, label="Efficiency Gap", tag="eg_y"):
                        dpg.add_line_series([], [], label="EG", tag="eg_series")
            self.theme.track(
                dpg.add_text(
                    "Load election data to use this panel.",
                    tag="eg_inactive_lbl", show=False,
                ),
                "muted",
            )

    def _build_dem_seats_panel(self):
        with dpg.window(
            label="Expected Dem Seats", tag="panel_dem_seats",
            show=False, width=500, height=280,
            pos=[_LEFT_W + 80, 80],
            on_close=lambda: dpg.set_value(self._panel_seats_item, False),
        ):
            with dpg.group(tag="seats_plot_grp"):
                with dpg.plot(height=-1, width=-1, no_menus=True):
                    dpg.add_plot_legend()
                    dpg.add_plot_axis(dpg.mvXAxis, label="Iteration", tag="seats_x")
                    with dpg.plot_axis(dpg.mvYAxis, label="Expected D Seats", tag="seats_y"):
                        dpg.add_line_series([], [], label="D Seats", tag="seats_series")
            self.theme.track(
                dpg.add_text(
                    "Load election data to use this panel.",
                    tag="seats_inactive_lbl", show=False,
                ),
                "muted",
            )

    def _build_pp_panel(self):
        with dpg.window(
            label="Polsby-Popper", tag="panel_pp",
            show=False, width=500, height=280,
            pos=[_LEFT_W + 80, 80],
            on_close=lambda: dpg.set_value(self._panel_pp_item, False),
        ):
            with dpg.group(tag="pp_plot_grp"):
                with dpg.plot(height=-1, width=-1, no_menus=True):
                    dpg.add_plot_legend()
                    dpg.add_plot_axis(dpg.mvXAxis, label="Iteration", tag="pp_x")
                    with dpg.plot_axis(
                            dpg.mvYAxis, label="Polsby-Popper (100 = circle)",
                            tag="pp_y"):
                        dpg.add_line_series([], [], label="PP", tag="pp_series")
            self._tooltip(
                "pp_plot_grp",
                "Shape rating from 0 to 100. Higher is more compact.",
            )
            self.theme.track(
                dpg.add_text(
                    "Turn on this score to chart it.",
                    tag="pp_inactive_lbl", show=False,
                ),
                "muted",
            )
        dpg.set_axis_limits("pp_y", 0.0, 100.0)

    def _build_reock_panel(self):
        with dpg.window(
            label="Reock", tag="panel_reock",
            show=False, width=500, height=280,
            pos=[_LEFT_W + 80, 80],
            on_close=lambda: dpg.set_value(self._panel_reock_item, False),
        ):
            with dpg.group(tag="reock_plot_grp"):
                with dpg.plot(height=-1, width=-1, no_menus=True):
                    dpg.add_plot_legend()
                    dpg.add_plot_axis(dpg.mvXAxis, label="Iteration", tag="reock_x")
                    with dpg.plot_axis(dpg.mvYAxis, label="Reock (100 = circle)", tag="reock_y"):
                        dpg.add_line_series([], [], label="Reock", tag="reock_series")
            self._tooltip(
                "reock_plot_grp",
                "Shape rating from 0 to 100. Higher is more compact.",
            )
            self.theme.track(
                dpg.add_text(
                    "Turn on this score to chart it.",
                    tag="reock_inactive_lbl", show=False,
                ),
                "muted",
            )
        dpg.set_axis_limits("reock_y", 0.0, 100.0)

    def _build_representation_panel(self):
        with dpg.window(
            label="Electoral Opportunity", tag="panel_representation",
            show=False, width=500, height=280,
            pos=[_LEFT_W + 80, 80],
            on_close=lambda: dpg.set_value(self._panel_representation_item, False),
        ):
            self._rep_chart_mode = dpg.add_radio_button(
                items=["Overall", "Rating", "Opportunity count"], default_value="Overall",
                horizontal=True,
            )
            self._tooltip(
                self._rep_chart_mode,
                "Overall: lower penalty is better. Rating: higher group rating is "
                "better. Opportunity count: modeled full and partial district credit.",
            )
            with dpg.group(tag="representation_plot_grp"):
                with dpg.plot(height=-1, width=-1, no_menus=True):
                    dpg.add_plot_legend()
                    dpg.add_plot_axis(dpg.mvXAxis, label="Iteration", tag="representation_x")
                    with dpg.plot_axis(dpg.mvYAxis,
                                       label="Penalty (0 = best)",
                                       tag="representation_y"):
                        dpg.add_line_series([], [], label="Overall",
                                            tag="representation_overall_series")
                        dpg.add_line_series([], [], label="Black",
                                            tag="representation_black_series")
                        dpg.add_line_series([], [], label="Latino",
                                            tag="representation_latino_series")
                        dpg.add_line_series([], [], label="Asian",
                                            tag="representation_asian_series")
            # Match the line colors to the demographic ("D") map overlay so the
            # chart and map agree: Black=blue, Latino=green, Asian=purple.
            from mosaic.gui.map_view import _DEMOGRAPHIC_RGB
            for grp, series in (("black",  "representation_black_series"),
                                ("latino", "representation_latino_series"),
                                ("asian",  "representation_asian_series")):
                r, g, b = _DEMOGRAPHIC_RGB[grp]
                with dpg.theme() as _line_theme:
                    with dpg.theme_component(dpg.mvLineSeries):
                        dpg.add_theme_color(dpg.mvPlotCol_Line, (r, g, b, 255),
                                            category=dpg.mvThemeCat_Plots)
                dpg.bind_item_theme(series, _line_theme)
            self._tooltip(
                "representation_plot_grp",
                "Group ratings and opportunity counts are higher-is-better. "
                "Overall penalty is lower-is-better.",
            )
            self.theme.track(
                dpg.add_text(
                    "Turn on this score to chart it.",
                    tag="representation_inactive_lbl", show=False,
                ),
                "muted",
            )
        # Pad past 0-100 so a rating pinned to the proportional 100 (or to 0)
        # sits inside the frame instead of clipped on the border.
        dpg.set_axis_limits("representation_y", -4.0, 104.0)

    def _build_minority_cohesion_panel(self):
        with dpg.window(
            label="Neighborhood Severance", tag="panel_minority_cohesion",
            show=False, width=500, height=280,
            pos=[_LEFT_W + 100, 100],
            on_close=lambda: dpg.set_value(self._panel_minority_cohesion_item, False),
        ):
            self._cohesion_chart_mode = dpg.add_radio_button(
                items=["Overall", "By group"], default_value="Overall",
                horizontal=True,
            )
            self._tooltip(
                self._cohesion_chart_mode,
                "Overall: lower penalty is better. By group: higher intact share "
                "is better.",
            )
            with dpg.group(tag="minority_cohesion_plot_grp"):
                with dpg.plot(height=-1, width=-1, no_menus=True):
                    dpg.add_plot_legend()
                    dpg.add_plot_axis(dpg.mvXAxis, label="Iteration",
                                      tag="minority_cohesion_x")
                    with dpg.plot_axis(dpg.mvYAxis,
                                       label="Penalty (0 = best)",
                                       tag="minority_cohesion_y"):
                        dpg.add_line_series([], [], label="Overall",
                                            tag="minority_cohesion_overall_series")
                        dpg.add_line_series([], [], label="Black",
                                            tag="minority_cohesion_black_series")
                        dpg.add_line_series([], [], label="Latino",
                                            tag="minority_cohesion_latino_series")
                        dpg.add_line_series([], [], label="Asian",
                                            tag="minority_cohesion_asian_series")
            # Same palette as the Electoral Opportunity chart and the "D" overlay.
            from mosaic.gui.map_view import _DEMOGRAPHIC_RGB
            for grp, series in (("black",  "minority_cohesion_black_series"),
                                ("latino", "minority_cohesion_latino_series"),
                                ("asian",  "minority_cohesion_asian_series")):
                r, g, b = _DEMOGRAPHIC_RGB[grp]
                with dpg.theme() as _line_theme:
                    with dpg.theme_component(dpg.mvLineSeries):
                        dpg.add_theme_color(dpg.mvPlotCol_Line, (r, g, b, 255),
                                            category=dpg.mvThemeCat_Plots)
                dpg.bind_item_theme(series, _line_theme)
            self._tooltip(
                "minority_cohesion_plot_grp",
                "Share of each selected group's weighted neighborhood kept together. "
                "Higher is better.",
            )
            self.theme.track(
                dpg.add_text(
                    "Turn on this score to chart it.",
                    tag="minority_cohesion_inactive_lbl", show=False,
                ),
                "muted",
            )
        dpg.set_axis_limits("minority_cohesion_y", -4.0, 104.0)

    def _build_community_congruence_panel(self):
        with dpg.window(
            label="Community Dispersion", tag="panel_community_congruence",
            show=False, width=500, height=280,
            pos=[_LEFT_W + 120, 120],
            on_close=lambda: dpg.set_value(
                self._panel_community_congruence_item, False),
        ):
            self._congruence_chart_mode = dpg.add_radio_button(
                items=["Overall", "By group"], default_value="Overall",
                horizontal=True,
            )
            self._tooltip(
                self._congruence_chart_mode,
                "Overall: lower penalty is better. By group: higher community "
                "congruence is better.",
            )
            with dpg.group(tag="community_congruence_plot_grp"):
                with dpg.plot(height=-1, width=-1, no_menus=True):
                    dpg.add_plot_legend()
                    dpg.add_plot_axis(dpg.mvXAxis, label="Iteration",
                                      tag="community_congruence_x")
                    with dpg.plot_axis(dpg.mvYAxis,
                                       label="Penalty (0 = best)",
                                       tag="community_congruence_y"):
                        dpg.add_line_series([], [], label="Overall",
                                            tag="community_congruence_overall_series")
                        dpg.add_line_series([], [], label="Black",
                                            tag="community_congruence_black_series")
                        dpg.add_line_series([], [], label="Latino",
                                            tag="community_congruence_latino_series")
                        dpg.add_line_series([], [], label="Asian",
                                            tag="community_congruence_asian_series")
            # Same palette as the other demographic charts and the "D" overlay.
            from mosaic.gui.map_view import _DEMOGRAPHIC_RGB
            for grp, series in (("black",  "community_congruence_black_series"),
                                ("latino", "community_congruence_latino_series"),
                                ("asian",  "community_congruence_asian_series")):
                r, g, b = _DEMOGRAPHIC_RGB[grp]
                with dpg.theme() as _line_theme:
                    with dpg.theme_component(dpg.mvLineSeries):
                        dpg.add_theme_color(dpg.mvPlotCol_Line, (r, g, b, 255),
                                            category=dpg.mvThemeCat_Plots)
                dpg.bind_item_theme(series, _line_theme)
            self._tooltip(
                "community_congruence_plot_grp",
                "Rates how little each selected group is spread across districts, "
                "after a population-based allowance. Higher is better.",
            )
            self.theme.track(
                dpg.add_text(
                    "Turn on this score to chart it.",
                    tag="community_congruence_inactive_lbl", show=False,
                ),
                "muted",
            )
        dpg.set_axis_limits("community_congruence_y", -4.0, 104.0)

    def _build_hc_panel(self):
        with dpg.window(
            label="Compactness", tag="panel_hc",
            show=False, width=500, height=280,
            pos=[_LEFT_W + 80, 80],
            on_close=lambda: dpg.set_value(self._panel_hc_item, False),
        ):
            with dpg.group(tag="hc_plot_grp"):
                with dpg.plot(height=-1, width=-1, no_menus=True):
                    dpg.add_plot_legend()
                    dpg.add_plot_axis(dpg.mvXAxis, label="Iteration", tag="hc_x")
                    with dpg.plot_axis(dpg.mvYAxis, label="Compactness (100 = best)", tag="hc_y"):
                        dpg.add_line_series([], [], label="Compactness", tag="hc_series")
            self._tooltip(
                "hc_plot_grp",
                "Combined Polsby-Popper and Reock rating. Higher is more compact.",
            )
            self.theme.track(
                dpg.add_text(
                    "Turn on this score to chart it.",
                    tag="hc_inactive_lbl", show=False,
                ),
                "muted",
            )
        dpg.set_axis_limits("hc_y", 0.0, 100.0)

    def _build_hsplit_panel(self):
        with dpg.window(
            label="County Congruence", tag="panel_hsplit",
            show=False, width=500, height=280,
            pos=[_LEFT_W + 80, 80],
            on_close=lambda: dpg.set_value(self._panel_hsplit_item, False),
        ):
            with dpg.group(tag="hsplit_plot_grp"):
                with dpg.plot(height=-1, width=-1, no_menus=True):
                    dpg.add_plot_legend()
                    dpg.add_plot_axis(dpg.mvXAxis, label="Iteration", tag="hsplit_x")
                    with dpg.plot_axis(
                            dpg.mvYAxis,
                            label="County Congruence (penalty, 0 = best)",
                            tag="hsplit_y"):
                        dpg.add_line_series([], [], label="Cty Cong", tag="hsplit_series")
            self._tooltip(
                "hsplit_plot_grp",
                "County-splitting penalty. Lower is better; Unclipped mode may exceed 100.",
            )
            self.theme.track(
                dpg.add_text(
                    "Turn on this score to chart it.",
                    tag="hsplit_inactive_lbl", show=False,
                ),
                "muted",
            )
        # Penalty is >= 0 and unbounded above: pin the floor at 0, auto-fit the top
        # each frame in the render block.
        dpg.set_axis_limits("hsplit_y", 0.0, 100.0)

    def _build_hprop_panel(self):
        with dpg.window(
            label="Proportionality", tag="panel_hprop",
            show=False, width=500, height=280,
            pos=[_LEFT_W + 80, 80],
            on_close=lambda: dpg.set_value(self._panel_hprop_item, False),
        ):
            with dpg.group(tag="hprop_plot_grp"):
                with dpg.plot(height=-1, width=-1, no_menus=True):
                    dpg.add_plot_legend()
                    dpg.add_plot_axis(dpg.mvXAxis, label="Iteration", tag="hprop_x")
                    with dpg.plot_axis(
                            dpg.mvYAxis, label="Proportionality (100 = best)",
                            tag="hprop_y"):
                        dpg.add_line_series([], [], label="Proportionality", tag="hprop_series")
                        dpg.add_line_series(
                            [], [], label="Inversion Risk",
                            tag="hprop_inversion_series")
            self._tooltip(
                "hprop_plot_grp",
                "Proportionality rating (higher = closer to proportional), and "
                "the modeled chance that the statewide vote winner loses the chamber.",
            )
            self.theme.track(
                dpg.add_text(
                    "Load election data to use this panel.",
                    tag="hprop_inactive_lbl", show=False,
                ),
                "muted",
            )
        dpg.set_axis_limits("hprop_y", 0.0, 100.0)

    def _build_hcmp_panel(self):
        with dpg.window(
            label="Competitiveness", tag="panel_hcmp",
            show=False, width=500, height=280,
            pos=[_LEFT_W + 80, 80],
            on_close=lambda: dpg.set_value(self._panel_hcmp_item, False),
        ):
            with dpg.group(tag="hcmp_plot_grp"):
                with dpg.plot(height=-1, width=-1, no_menus=True):
                    dpg.add_plot_legend()
                    dpg.add_plot_axis(dpg.mvXAxis, label="Iteration", tag="hcmp_x")
                    with dpg.plot_axis(
                            dpg.mvYAxis, label="Competitiveness (100 = best)",
                            tag="hcmp_y"):
                        dpg.add_line_series([], [], label="Competitiveness", tag="hcmp_series")
            self._tooltip(
                "hcmp_plot_grp",
                "Higher means the election model treats more districts as close contests.",
            )
            self.theme.track(
                dpg.add_text(
                    "Load election data to use this panel.",
                    tag="hcmp_inactive_lbl", show=False,
                ),
                "muted",
            )
        dpg.set_axis_limits("hcmp_y", 0.0, 100.0)

    def _build_pb_panel(self):
        with dpg.window(
            label="Partisan Bias", tag="panel_pb",
            show=False, width=500, height=280,
            pos=[_LEFT_W + 80, 80],
            on_close=lambda: dpg.set_value(self._panel_pb_item, False),
        ):
            with dpg.group(tag="pb_plot_grp"):
                with dpg.plot(height=-1, width=-1, no_menus=True):
                    dpg.add_plot_legend()
                    dpg.add_plot_axis(dpg.mvXAxis, label="Iteration", tag="pb_x")
                    with dpg.plot_axis(dpg.mvYAxis, label="Bias (+ = R)", tag="pb_y"):
                        dpg.add_line_series([], [], label="Bias", tag="pb_series")
            self.theme.track(
                dpg.add_text(
                    "Load election data to use this panel.",
                    tag="pb_inactive_lbl", show=False,
                ),
                "muted",
            )

    def _build_pg_panel(self):
        with dpg.window(
            label="Partisan Gini", tag="panel_pg",
            show=False, width=500, height=280,
            pos=[_LEFT_W + 80, 80],
            on_close=lambda: dpg.set_value(self._panel_pg_item, False),
        ):
            with dpg.group(tag="pg_plot_grp"):
                with dpg.plot(height=-1, width=-1, no_menus=True):
                    dpg.add_plot_legend()
                    dpg.add_plot_axis(dpg.mvXAxis, label="Iteration", tag="pg_x")
                    with dpg.plot_axis(dpg.mvYAxis, label="Penalty (0 = best)", tag="pg_y"):
                        dpg.add_line_series([], [], label="Gini", tag="pg_series")
            self.theme.track(
                dpg.add_text(
                    "Load election data to use this panel.",
                    tag="pg_inactive_lbl", show=False,
                ),
                "muted",
            )
        dpg.set_axis_limits("pg_y", 0.0, 100.0)

    def _build_popdev_panel(self):
        with dpg.window(
            label="Population Deviation", tag="panel_popdev",
            show=False, width=500, height=280,
            pos=[_LEFT_W + 100, 100],
            on_close=lambda: dpg.set_value(self._panel_popdev_item, False),
        ):
            with dpg.group(tag="popdev_plot_grp"):
                with dpg.plot(height=-1, width=-1, no_menus=True):
                    dpg.add_plot_legend()
                    dpg.add_plot_axis(dpg.mvXAxis, label="Iteration", tag="popdev_x")
                    with dpg.plot_axis(dpg.mvYAxis, label="% Deviation", tag="popdev_y"):
                        dpg.add_line_series([], [], label="Max %",
                                            tag="popdev_max_series")
                        dpg.add_line_series([], [], label="Mean %",
                                            tag="popdev_mean_series")
                # Set y-axis floor to 0; constrain so pan/zoom can't go below.
                dpg.set_axis_limits("popdev_y", 0.0, 5.0)
                dpg.set_axis_limits_constraints("popdev_y", 0.0, float("inf"))
            self._tooltip(
                "popdev_plot_grp",
                "Max and mean absolute deviation from ideal population. Lower is better.",
            )
            self.theme.track(
                dpg.add_text(
                    "Turn on this score to chart it.",
                    tag="popdev_inactive_lbl", show=False,
                ),
                "muted",
            )

    def _build_alignment_panel(self):
        with dpg.window(
            label="Alignment", tag="panel_alignment",
            show=False, width=500, height=280,
            pos=[_LEFT_W + 100, 100],
            on_close=lambda: dpg.set_value(self._panel_alignment_item, False),
        ):
            with dpg.group(tag="alignment_plot_grp"):
                with dpg.plot(height=-1, width=-1, no_menus=True):
                    dpg.add_plot_legend()
                    dpg.add_plot_axis(dpg.mvXAxis, label="Iteration",
                                      tag="alignment_x")
                    with dpg.plot_axis(dpg.mvYAxis,
                                       label="Cohesion % (100 = intact)",
                                       tag="alignment_y"):
                        dpg.add_line_series([], [], label="Mean",
                                            tag="alignment_mean_series")
                        dpg.add_line_series([], [], label="Worst district",
                                            tag="alignment_min_series")
            self._tooltip(
                "alignment_plot_grp",
                "Share of each reference district that stays intact. "
                "Higher is closer to the reference plan.",
            )
            self.theme.track(
                dpg.add_text(
                    "Load a reference plan and turn on Alignment to chart it.",
                    tag="alignment_inactive_lbl", show=False,
                ),
                "muted",
            )
        dpg.set_axis_limits("alignment_y", 0.0, 100.0)

    def _build_cut_edges_panel(self):
        with dpg.window(
            label="Cut Edges", tag="panel_cut_edges",
            show=False, width=500, height=280,
            pos=[_LEFT_W + 80, 80],
            on_close=lambda: dpg.set_value(self._panel_cuts_item, False),
        ):
            with dpg.plot(height=-1, width=-1, no_menus=True):
                dpg.add_plot_legend()
                dpg.add_plot_axis(dpg.mvXAxis, label="Iteration", tag="cuts_x")
                with dpg.plot_axis(dpg.mvYAxis, label="Cut Edges", tag="cuts_y"):
                    dpg.add_line_series([], [], label="Cuts", tag="cuts_series")

    def _build_majority_panel(self):
        with dpg.window(
            label="Chance of Majority", tag="panel_majority",
            show=False, width=500, height=280,
            pos=[_LEFT_W + 80, 80],
            on_close=lambda: dpg.set_value(self._panel_majority_item, False),
        ):
            with dpg.group(tag="majority_plot_grp"):
                with dpg.plot(height=-1, width=-1, no_menus=True):
                    dpg.add_plot_legend()
                    dpg.add_plot_axis(dpg.mvXAxis, label="Iteration", tag="maj_x")
                    with dpg.plot_axis(dpg.mvYAxis, label="Chance of Majority (%)", tag="maj_y"):
                        dpg.add_line_series([], [], label="Dem Majority", tag="maj_dem_series")
                        dpg.add_line_series([], [], label="Rep Majority", tag="maj_rep_series")
            self.theme.track(
                dpg.add_text(
                    "Load election data to use this panel.",
                    tag="majority_inactive_lbl", show=False,
                ),
                "muted",
            )
        dpg.set_axis_limits("maj_y", 0.0, 100.0)

    def _build_hinge_panel(self):
        with dpg.window(
            label="Supermajority/Hinge", tag="panel_hinge",
            show=False, width=500, height=280,
            pos=[_LEFT_W + 80, 80],
            on_close=lambda: dpg.set_value(self._panel_hinge_item, False),
        ):
            with dpg.group(tag="hinge_plot_grp"):
                with dpg.plot(height=-1, width=-1, no_menus=True):
                    dpg.add_plot_legend()
                    dpg.add_plot_axis(dpg.mvXAxis, label="Iteration", tag="hinge_x")
                    with dpg.plot_axis(dpg.mvYAxis, label="Chance of Hinge (%)", tag="hinge_y"):
                        dpg.add_line_series([], [], label="Hinge", tag="hinge_series")
            self.theme.track(
                dpg.add_text(
                    "Load election data to use this panel.",
                    tag="hinge_inactive_lbl", show=False,
                ),
                "muted",
            )
        dpg.set_axis_limits("hinge_y", 0.0, 100.0)

    # Column spec for the District Info table.  (header, pixel_width, key)
    _DIST_COLS = [
        ("Dist",        50,  "dist"),
        ("Population",  95,  "pop"),
        ("Pop Dev",     80,  "pdev"),
        ("Polsby-Popper", 110, "pp"),
        ("Dem %",       70,  "demp"),
        ("Rep %",       70,  "repp"),
        ("Margin",      75,  "margin"),
    ]

    def _build_district_info_panel(self):
        # Tracks how many rows the table currently has, so we only rebuild
        # rows when the district count changes (not every frame).
        self._dist_table_built_rows: int = 0
        # Iteration at which we last refreshed the table.  -1 forces an
        # update on first paint and after the panel re-opens.
        self._dist_info_last_iter: int = -1
        total_w = sum(w for _, w, _ in self._DIST_COLS) + 32  # column widths + chrome
        with dpg.window(
            label="District Info", tag="panel_district_info",
            show=False, width=total_w, height=400,
            pos=[_LEFT_W + 80, 80],
            on_close=self._on_district_panel_close,
        ):
            self.theme.text("Per-district metrics for the current plan.", "muted")
            with dpg.group(horizontal=True):
                self._dist_info_update_every = dpg.add_slider_int(
                    label="Update every (iters)",
                    default_value=1000, min_value=50, max_value=10000,
                    width=200,
                    callback=self._on_dist_info_interval_change,
                )
                self._tooltip(
                    self._dist_info_update_every,
                    "How often this table refreshes during a run. This does not "
                    "change scoring.",
                )
            dpg.add_spacer(height=4)
            with dpg.table(
                tag="district_info_table",
                header_row=True,
                borders_innerH=True,
                borders_innerV=False,
                borders_outerH=False,
                borders_outerV=False,
                row_background=True,
                policy=dpg.mvTable_SizingFixedFit,
                scrollY=True,
                no_host_extendX=True,
                width=-1, height=-1,
            ):
                for label, w, _ in self._DIST_COLS:
                    dpg.add_table_column(
                        label=label,
                        width_fixed=True,
                        init_width_or_weight=w,
                        no_resize=True,
                        no_reorder=True,
                    )
            self.theme.track(
                dpg.add_text(
                    "Load a shapefile to populate this table.",
                    tag="district_info_empty_lbl", show=False,
                ),
                "muted",
            )

    def _build_score_contrib_panel(self):
        """Horizontal bar chart: metric names on the category (Y) axis, magnitude
        on the value (X) axis. Horizontal because the names are long enough that a
        vertical chart forced an abbreviation table.
        """
        self._contrib_bar_themes = []
        for _, rgba in _CONTRIB_BAR_METRICS:
            with dpg.theme() as t:
                with dpg.theme_component(dpg.mvBarSeries):
                    dpg.add_theme_color(
                        dpg.mvPlotCol_Fill, rgba,
                        category=dpg.mvThemeCat_Plots,
                    )
            self._contrib_bar_themes.append(t)

        # Width is pinned (min_size[0] == max_size[0]): the category axis is sized
        # for the metric names, and a fixed width keeps the intro text's wrap in
        # step with it. Height stays draggable, since the bar count varies.
        w = 332
        with dpg.window(
            label="Score Contributors", tag="panel_score_contrib",
            show=False, width=w, height=340,
            min_size=[w, 200], max_size=[w, 10_000],
            pos=[_LEFT_W + 80, 80],
            on_close=lambda: dpg.set_value(self._panel_contrib_item, False),
        ):
            self.theme.text(
                "Share of the current total from each weighted score. Longer bars "
                "have more influence.",
                "muted",
                wrap=w - 20,
            )
            dpg.add_spacer(height=2)
            with dpg.plot(height=-1, width=-1, no_mouse_pos=True, no_menus=True):
                # X is the value axis, Y the category axis (horizontal bars).
                dpg.add_plot_axis(dpg.mvXAxis, tag="contrib_x", label="% of total")
                with dpg.plot_axis(dpg.mvYAxis, tag="contrib_y",
                                   no_gridlines=True):
                    self._contrib_bar_series = []
                    for i, (name, _) in enumerate(_CONTRIB_BAR_METRICS):
                        s = dpg.add_bar_series(
                            [0.0], [float(i + 1)], weight=0.7, horizontal=True,
                            label="##cb{}".format(i),
                        )
                        dpg.bind_item_theme(s, self._contrib_bar_themes[i])
                        self._contrib_bar_series.append(s)
            dpg.set_axis_limits("contrib_x", 0.0, 100.0)
            dpg.set_axis_limits("contrib_y", 0.5, len(_CONTRIB_BAR_METRICS) + 0.5)
