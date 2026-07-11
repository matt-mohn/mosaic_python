"""Modal popup builders (settings, help, confirmations)."""
from ._common import _DOCS_SHAPEFILE_URL, _DOCS_URL, __version__, dpg, webbrowser


class PopupsMixin:
    """Modal popup builders (settings, help, confirmations)."""

    def _build_population_popup(self):
        with self._dialog(
            "Population", "popup_population", (420, 340),
            show=False,
            secondary=("Close",
                       lambda: dpg.configure_item("popup_population", show=False)),
        ):
            self._tolerance = dpg.add_slider_float(
                label="Population Tolerance",
                default_value=2.5, min_value=0.1, max_value=10.0,
                format="%.1f %%", width=260,
            )
            self._tooltip(
                self._tolerance,
                "Mosaic will only explore solutions where each district differs "
                "from ideal by no more than this percentage in either direction.",
            )
            dpg.add_spacer(height=6)
            with dpg.group(horizontal=True):
                dpg.add_text("Tolerance Ratchet:")
                self._tolerance_ratchet_mode = dpg.add_radio_button(
                    ["Off", "Standard", "Strict"],
                    default_value="Off", horizontal=True,
                    callback=self._on_popdev_score_toggle,
                )
            self._tooltip(
                self._tolerance_ratchet_mode,
                "Gradually tightens Population Tolerance toward 0.25% over the "
                "back of the run - never below the deviation the map already "
                "hit, so it can't strand a plan.\n"
                "  Off: fixed.\n"
                "  Standard: tightens on each new best.\n"
                "  Strict: tightens every eligible step.",
            )
            dpg.add_spacer(height=10)
            dpg.add_separator()
            dpg.add_spacer(height=6)
            self.theme.text("Population Deviation Score - Safe Harbor", "heading")
            self._pop_dev_harbor = dpg.add_slider_float(
                label="Safe Harbor",
                default_value=0.0, min_value=0.0, max_value=5.0,
                format="%.2f %%", width=260,
            )
            self._tooltip(
                self._pop_dev_harbor,
                "Districts within this % of ideal are not penalized by the "
                "population deviation score. Cannot exceed Population Tolerance "
                "(clamped on run).",
            )

    def _build_seed_popup(self):
        with self._dialog(
            "Seed", "popup_seed", (340, 160), show=False,
            secondary=("Close",
                       lambda: dpg.configure_item("popup_seed", show=False)),
        ):
            self._seed = dpg.add_input_int(
                label="Random Seed  (0 = random)",
                default_value=0, min_value=0, width=120,
            )
            self._tooltip(
                self._seed,
                "Set a non-zero seed to make a run reproducible; 0 leaves the "
                "RNG random. Best-effort: identical results need the same "
                "machine, Mosaic version, and shapefile - float ordering in "
                "numpy/igraph can differ across environments.",
            )

    def _build_advanced_save_popup(self):
        # Fixed size (not autosize): the inline spinner/status toggle on save and
        # we don't want the window resizing mid-export.
        with self._dialog(
            "Export District Map", "popup_adv_save", (420, 300),
            show=False, autosize=False,
        ):
            self.theme.text(
                "Map exports mirror the settings you currently have in your "
                "district map view.",
                "muted", wrap=400,
            )
            dpg.add_spacer(height=8)
            self._adv_save_title = dpg.add_input_text(
                label="Title (optional)", default_value="", width=260,
                hint="Leave blank for no title",
            )
            dpg.add_spacer(height=8)
            dpg.add_text("Format")
            self._adv_save_fmt = dpg.add_radio_button(
                ["PNG (raster)", "PDF (vector, slower)"],
                default_value="PNG (raster)", horizontal=True,
                callback=self._on_adv_fmt_changed,
            )
            with dpg.group() as self._adv_dpi_group:
                dpg.add_spacer(height=4)
                self._adv_save_dpi = dpg.add_combo(
                    label="Raster DPI",
                    items=["96 (screen)", "144 (1.5x)", "192 (2x)",
                           "288 (3x)", "384 (4x)", "576 (6x)"],
                    default_value="288 (3x)", width=200,
                )
            dpg.add_spacer(height=10)
            dpg.add_separator()
            dpg.add_spacer(height=6)
            with dpg.group(horizontal=True):
                self._adv_save_btn = dpg.add_button(
                    label="Save", width=80,
                    callback=self._on_advanced_save_confirm,
                )
                dpg.bind_item_theme(self._adv_save_btn, self.theme.nudge_theme)
                self._adv_save_as_btn = dpg.add_button(
                    label="Save As...", width=90,
                    callback=self._on_advanced_save_as,
                )
                dpg.bind_item_theme(self._adv_save_as_btn,
                                    self.theme.nudge_theme)
                self._adv_close_btn = dpg.add_button(
                    label="Close", width=80,
                    callback=lambda: dpg.configure_item(
                        "popup_adv_save", show=False),
                )
                dpg.bind_item_theme(self._adv_close_btn,
                                    self.theme.antinudge_theme)
                dpg.add_spacer(width=4)
                self._adv_save_spinner = dpg.add_loading_indicator(
                    style=0, radius=2.0, show=False,
                    color=self.theme.color("body"),
                    secondary_color=self.theme.color("muted"),
                )
            self._adv_save_status = dpg.add_text("", show=False)

    def _build_opt_popup(self):
        with self._dialog(
            "Annealing Settings", "popup_opt", (460, 460),
            show=False,
            secondary=("Close",
                       lambda: dpg.configure_item("popup_opt", show=False)),
        ):
            self._ann_enabled = dpg.add_checkbox(
                label="Enable simulated annealing", default_value=True,
                callback=self._on_ann_toggle,
            )
            dpg.add_separator()

            with dpg.group(tag="ann_body"):
                self._temp_factor = dpg.add_slider_float(
                    label="Initial Temp Factor",
                    default_value=0.2, min_value=0.01, max_value=2.0,
                    format="%.3f", width=260,
                )
                self._tooltip(
                    self._temp_factor,
                    "initial_temp = factor x initial_score",
                )
                dpg.add_spacer(height=6)

                dpg.add_text("Cooling mode:")
                self._cool_mode = dpg.add_radio_button(
                    items=["Guided (recommended)", "Static"],
                    default_value="Guided (recommended)",
                    callback=self._on_cool_mode,
                    horizontal=True,
                )
                dpg.add_spacer(height=4)

                with dpg.group(tag="guided_controls"):
                    self._guide_frac = dpg.add_slider_float(
                        label="Guide Point",
                        default_value=0.9, min_value=0.5, max_value=1.0,
                        format="%.2f", width=200,
                    )
                    self._tooltip(
                        self._guide_frac,
                        "Fraction of iterations to cool over.",
                    )
                    self._target_temp = dpg.add_input_float(
                        label="Target Temp",
                        default_value=1.0, min_value=0.001,
                        format="%.3f", width=120,
                    )
                    self._tooltip(
                        self._target_temp,
                        "Temperature at the guide point (absolute).",
                    )

                with dpg.group(tag="static_controls", show=False):
                    self._cooling_rate = dpg.add_slider_float(
                        label="Cooling Rate / iteration",
                        default_value=0.9995, min_value=0.990,
                        max_value=0.99999, format="%.5f", width=260,
                    )
                    self._tooltip(
                        self._cooling_rate,
                        "Per-iteration temperature multiplier (default "
                        "0.9995). Lower values cool faster.",
                    )

                dpg.add_spacer(height=8)
                dpg.add_separator()
                dpg.add_spacer(height=4)
                self._launch_watch_enabled = dpg.add_checkbox(
                    label="Launch Watch",
                    default_value=True,
                    callback=self._on_launch_watch_toggle,
                )
                with dpg.group(tag="launch_watch_controls"):
                    self._launch_watch_iter = dpg.add_input_int(
                        label="Re-anchor after iter",
                        default_value=250, min_value=10, max_value=10_000,
                        step=50, width=120,
                    )
                    self._tooltip(
                        self._launch_watch_iter,
                        "Once past this iteration, reset initial_temp to "
                        "factor x current_score (instead of initial_score). "
                        "Helps when scores nose-dive in the first ~250 steps.",
                    )

            dpg.add_spacer(height=10)
            dpg.add_separator()
            dpg.add_spacer(height=6)
            self.theme.text("n=3 ReCom Mix", "heading")
            self._n3_pct = dpg.add_slider_int(
                label="% of iterations",
                default_value=25, min_value=0, max_value=50,
                format="%d %%", width=200,
            )
            self._tooltip(
                self._n3_pct,
                "Fraction of steps that merge 3 districts (vs 2) and re-split. "
                "Escapes local minima better, at ~2.4x per-step cost. "
                "Set to 0 for mass-generation runs.",
            )

            dpg.add_spacer(height=6)
            self.theme.text("Polish Flips", "heading")
            self._flip_enabled = dpg.add_checkbox(
                label="Enable single-precinct flips in tail",
                default_value=True,
            )
            self._tooltip(
                self._flip_enabled,
                "Adds single-precinct boundary flips alongside ReCom to polish "
                "borders late in the run. The flip rate ramps from 5% early to "
                "85% at the end, always leaving at least 15% ReCom.",
            )
            self._flip_midpoint = dpg.add_slider_int(
                label="50% crossover (% of run)",
                default_value=84, min_value=1, max_value=99,
                format="%d %%", width=200,
            )
            self._tooltip(
                self._flip_midpoint,
                "Where in the run the flip rate hits 50%. "
                "Higher = flips stay rare until later.",
            )

            dpg.add_spacer(height=10)
            dpg.add_separator()
            dpg.add_spacer(height=6)
            self.theme.text("County Congruence", "heading")
            self._hsplit_unclipped = dpg.add_checkbox(
                label="Unclipped County Congruence", default_value=True,
            )
            self._tooltip(
                self._hsplit_unclipped,
                "Let the County Congruence penalty climb past the scorecard cap so "
                "the optimizer keeps a gradient on heavily-split plans. Off = clipped "
                "scorecard form (pins at the cap).",
            )

    def _build_alignment_settings_popup(self):
        # Non-modal exception: this window spawns a file dialog ("Load reference
        # plan..."), and a modal-over-modal stack misbehaves in DPG.
        with self._dialog(
            "Alignment Settings", "popup_alignment_settings", (400, 340),
            show=False, modal=False,
            secondary=("Close", lambda: dpg.configure_item(
                "popup_alignment_settings", show=False)),
        ):
            self.theme.text(
                "Load a reference plan, then choose whose voters to align "
                "and which districts to score.", "muted", wrap=380,
            )
            dpg.add_separator()
            dpg.add_spacer(height=6)

            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Load reference plan...",
                    callback=self._on_load_alignment,
                )
                dpg.add_button(
                    label="Clear",
                    callback=self._on_clear_alignment,
                )
            dpg.add_spacer(height=8)

            # Ask 1 — whose share retention is measured in (needs election data).
            dpg.add_text("Focus:")
            self._alignment_focus = dpg.add_radio_button(
                items=["All residents", "Republican", "Democratic"],
                default_value="All residents", horizontal=True,
                callback=self._on_alignment_focus,
            )
            self._hint(self._alignment_focus, "alignment_focus")
            dpg.add_spacer(height=6)

            # Ask 2 — restrict scoring to the focus party's won districts.
            # Meaningless without a party focus, so disabled when neutral.
            self._alignment_restrict = dpg.add_checkbox(
                label="Only districts that party wins",
                default_value=False, enabled=False,
            )
            self._hint(self._alignment_restrict, "alignment_restrict")
            dpg.add_spacer(height=6)

            self._alignment_win_threshold = dpg.add_slider_float(
                label="District win threshold",
                default_value=0.535, min_value=0.50, max_value=0.70,
                format="%.3f", width=200,
            )
            self._tooltip(
                self._alignment_win_threshold,
                "A reference district counts as 'won' (and is scored) when the "
                "focus party's two-party share exceeds this. 0.535 ~ win by 7pts; "
                "raise it to ignore near-coin-flip seats.",
            )

    def _build_partisan_popup(self):
        with self._dialog(
            "Partisanship Settings", "popup_partisan", (460, 420),
            show=False,
            secondary=("Close",
                       lambda: dpg.configure_item("popup_partisan", show=False)),
        ):
            self.theme.text(
                "Applied when partisan metrics are enabled.",
                "muted",
            )
            dpg.add_separator()
            dpg.add_spacer(height=6)

            self._win_prob = dpg.add_slider_float(
                label="Win Prob at 55% vote share",
                default_value=0.9, min_value=0.51, max_value=0.999,
                format="%.3f", width=220,
            )
            self._tooltip(
                self._win_prob,
                "P(D wins district | D has 55% of two-party vote)",
            )
            dpg.add_spacer(height=6)

            self._swing_sigma = dpg.add_slider_float(
                label="Swing sigma (shared)",
                default_value=0.03, min_value=0.005, max_value=0.10,
                format="%.3f", width=220,
            )
            self._tooltip(
                self._swing_sigma,
                "Std dev of partisan-environment swing shared across all districts.",
            )
            dpg.add_spacer(height=8)

            dpg.add_text("Efficiency Gap mode:")
            self._eg_mode = dpg.add_radio_button(
                items=["Robust (recommended)", "Static"],
                default_value="Robust (recommended)",
                horizontal=True,
            )
            self._tooltip(
                self._eg_mode,
                "Robust EG integrates out both swing sigma and per-district noise.",
            )
            dpg.add_spacer(height=10)
            dpg.add_separator()
            dpg.add_spacer(height=6)

            self.theme.text("Mean-Median / Efficiency Gap shape", "heading")
            self._partisan_quadratic_penalty = dpg.add_checkbox(
                label="Use quadratic penalty",
                default_value=False,
            )
            self._tooltip(
                self._partisan_quadratic_penalty,
                "When ON, all three modes (Fair / D / R) use a quadratic curve: "
                "less urgency near the favored end, more urgency near the unfavored "
                "end. Off = linear.",
            )
            dpg.add_spacer(height=8)

            self._mm_bound = dpg.add_slider_float(
                label="MM bound",
                default_value=0.20, min_value=0.05, max_value=0.30,
                format="%.2f", width=220,
            )
            self._tooltip(
                self._mm_bound,
                "MM penalty reaches 100 at this |raw| value, then saturates.",
            )
            self._eg_bound = dpg.add_slider_float(
                label="EG bound",
                default_value=0.35, min_value=0.10, max_value=0.50,
                format="%.2f", width=220,
            )
            self._tooltip(
                self._eg_bound,
                "EG penalty reaches 100 at this |raw| value, then saturates.",
            )
            self._pbias_bound = dpg.add_slider_float(
                label="Partisan Bias bound",
                default_value=0.25, min_value=0.05, max_value=0.50,
                format="%.2f", width=220,
            )
            self._tooltip(
                self._pbias_bound,
                "Partisan Bias penalty reaches 100 at this |raw| seat-tilt, then saturates.",
            )
            dpg.add_spacer(height=10)
            dpg.add_separator()
            dpg.add_spacer(height=6)
            self.theme.text("Competitiveness", "heading")
            self._comp_unclipped = dpg.add_checkbox(
                label="Unclipped Competitiveness", default_value=True,
            )
            self._tooltip(
                self._comp_unclipped,
                "Two-segment knee mapping: the achievable competitive range spends most "
                "of the penalty scale and reserves the top few points for the near-"
                "impossible all-toss-up range, so the optimizer keeps a gradient. Off = "
                "clipped scorecard form (saturates once ~75% of seats are toss-ups).",
            )

    def _build_help_popup(self):
        # Fixed-size reader: the doc text scrolls inside its own child_window,
        # so this one keeps a fixed height instead of auto-fitting.
        with self._dialog(
            "Help", "popup_help", (460, 400), show=False, autosize=False,
            secondary=("Close",
                       lambda: dpg.configure_item("popup_help", show=False)),
        ):
            with dpg.child_window(height=-48, border=False):
                self.theme.text(
                    "Mosaic from Matt Mohn (@mattmxhn)",
                    "title",
                )
                self.theme.text(
                    f"Version {__version__}",
                    "muted",
                )
                dpg.add_spacer(height=6)
                dpg.add_text(
                    "Mosaic uses simulated annealing plus recombination to generate "
                    "redistricting plans. Cooling makes it more selective over time; "
                    "recombination (merge two districts, draw a new boundary) is the "
                    "edit it tries on each step.",
                    wrap=420,
                )
                dpg.add_spacer(height=14)

                self.theme.text("Basic usage", "heading")
                dpg.add_separator()
                dpg.add_text(
                    "1. Import a shapefile\n"
                    "2. Map columns (population, county, votes) in the picker\n"
                    "3. Set district count and iterations\n"
                    "4. Enable and weight the scores in the left-hand panel\n"
                    "5. Optionally load an existing plan as a hot start via "
                    "Advanced > Load Hot Start\n"
                    "6. Start - pause / reset / revert to best as needed\n"
                    "7. Save Assignments writes a CSV",
                    wrap=420,
                )
                dpg.add_spacer(height=14)

                self.theme.text("Full documentation", "heading")
                dpg.add_separator()
                dpg.add_text(
                    "The website has the complete reference: scoring formulas, "
                    "shapefile sources and requirements, install steps, "
                    "troubleshooting, and methodology notes.",
                    wrap=420,
                )
                dpg.add_spacer(height=8)
                with dpg.group(horizontal=True):
                    dpg.add_button(
                        label="Open docs",
                        callback=lambda: webbrowser.open(_DOCS_URL),
                        width=110,
                    )
                    dpg.add_button(
                        label="Shapefile guide",
                        callback=lambda: webbrowser.open(_DOCS_SHAPEFILE_URL),
                        width=140,
                    )

        # Recent menu is built directly from _recent_shapefiles during
        # window construction above; _refresh_recent_menu handles updates.

    # ── Main loop ─────────────────────────────────────────────────────────────

    def _build_new_confirm_popup(self) -> None:
        with self._dialog(
            "New Map", "popup_new_confirm", (380, 120),
            show=False,
            buttons=[
                ("Discard & New", self._do_new, "primary"),
                ("Cancel",
                 lambda: dpg.configure_item("popup_new_confirm", show=False)),
            ],
        ):
            dpg.add_text(
                "The current run has results that have not been saved.\n"
                "Discard them and start a new map?",
                wrap=380 - 2 * 16,
            )

    def _build_close_confirm_popup(self) -> None:
        with self._dialog(
            "Close Mosaic", "popup_close_confirm", (380, 120),
            show=False,
            buttons=[
                ("Close Anyway", lambda: dpg.stop_dearpygui(), "primary"),
                ("Cancel",
                 lambda: dpg.configure_item("popup_close_confirm", show=False)),
            ],
        ):
            dpg.add_text(
                "The current run has results that have not been saved.\n"
                "Close anyway?",
                wrap=380 - 2 * 16,
            )
