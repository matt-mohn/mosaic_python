"""Modal popup builders (settings, help, confirmations)."""
from mosaic.scoring.opportunity import GROUPS

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
                "Every district must stay within this percentage of ideal "
                "population. Proposals outside the limit are rejected.",
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
                "Tightens Population Tolerance later in the run.\n"
                "Off: fixed. Standard: tighten on a new best. "
                "Strict: tighten at every eligible step.",
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
                "Differences inside this band add no Population Deviation penalty. "
                "This does not change the hard Population Tolerance.",
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
                "Use a nonzero value to repeat a run; zero chooses a new value. "
                "For best repeatability, turn off Fast tree generation and reuse "
                "the same inputs and settings.",
            )

    def _build_advanced_save_popup(self):
        # Fixed size (not autosize): the inline spinner/status toggle on save and
        # we don't want the window resizing mid-export.
        with self._dialog(
            "Save Map Image", "popup_adv_save", (420, 300),
            show=False, autosize=False,
        ):
            self.theme.text(
                "PNG and PDF export the full map, regardless of zoom.\n"
                "Current colors, overlays, and labels are preserved.",
                "muted", wrap=380,
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
            "Annealing Settings", "popup_opt", (460, 500),
            show=False,
            secondary=("Close",
                       lambda: dpg.configure_item("popup_opt", show=False)),
        ):
            self._fast_trees = dpg.add_checkbox(
                label="Fast tree generation", default_value=True,
            )
            self._tooltip(
                self._fast_trees,
                "Build proposal trees faster. Turn this off for the strongest "
                "repeatability with a fixed seed.",
            )
            dpg.add_separator()
            self._ann_enabled = dpg.add_checkbox(
                label="Enable simulated annealing", default_value=True,
                callback=self._on_ann_toggle,
            )
            self._tooltip(
                self._ann_enabled,
                "Sometimes accept a higher-scoring map so the search can keep "
                "exploring. When off, every valid proposal is accepted.",
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
                    "Sets the starting willingness to accept a higher score. "
                    "Larger values allow more exploration.",
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
                        "Point in the run where Guided cooling reaches Target Temp.",
                    )
                    self._target_temp = dpg.add_input_float(
                        label="Target Temp",
                        default_value=1.0, min_value=0.001,
                        format="%.3f", width=120,
                    )
                    self._tooltip(
                        self._target_temp,
                        "Temperature Guided cooling reaches at Guide Point.",
                    )

                with dpg.group(tag="static_controls", show=False):
                    self._cooling_rate = dpg.add_slider_float(
                        label="Cooling Rate / iteration",
                        default_value=0.9995, min_value=0.990,
                        max_value=0.99999, format="%.5f", width=260,
                    )
                    self._tooltip(
                        self._cooling_rate,
                        "Multiplies temperature after each iteration. Lower values "
                        "cool faster.",
                    )

                dpg.add_spacer(height=8)
                dpg.add_separator()
                dpg.add_spacer(height=4)
                self._launch_watch_enabled = dpg.add_checkbox(
                    label="Launch Watch",
                    default_value=True,
                    callback=self._on_launch_watch_toggle,
                )
                self._tooltip(
                    self._launch_watch_enabled,
                    "Recalculate the Guided cooling schedule once after the "
                    "opening part of the run.",
                )
                with dpg.group(tag="launch_watch_controls"):
                    self._launch_watch_iter = dpg.add_input_int(
                        label="Re-anchor after iter",
                        default_value=250, min_value=10, max_value=10_000,
                        step=50, width=120,
                    )
                    self._tooltip(
                        self._launch_watch_iter,
                        "Earliest iteration when Launch Watch may recalculate the schedule.",
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
                "Chance of redrawing three neighboring districts after no flip is "
                "selected. These proposals usually take longer than ordinary ReCom.",
            )

            dpg.add_spacer(height=6)
            self.theme.text("Polish Flips", "heading")
            self._flip_enabled = dpg.add_checkbox(
                label="Enable single-precinct flips",
                default_value=True,
            )
            self._tooltip(
                self._flip_enabled,
                "Allow small boundary changes throughout the run. They are rare "
                "early and more common late.",
            )
            self._flip_midpoint = dpg.add_slider_int(
                label="50% crossover (% of run)",
                default_value=84, min_value=1, max_value=99,
                format="%d %%", width=200,
            )
            self._tooltip(
                self._flip_midpoint,
                "Point in the run where flips reach half of proposal attempts. "
                "Higher values delay them.",
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
                "Keep distinguishing heavily split maps after the standard penalty "
                "reaches its cap.",
            )

            dpg.add_spacer(height=10)
            dpg.add_separator()
            dpg.add_spacer(height=6)
            self.theme.text("Compactness", "heading")
            self._hcompact_unclipped = dpg.add_checkbox(
                label="Unclipped Compactness", default_value=True,
            )
            self._tooltip(
                self._hcompact_unclipped,
                "Keep distinguishing compact maps after the standard penalty "
                "reaches its cap.",
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
                "Include a reference district when the selected party's two-party "
                "vote share is above this value.",
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
                "Modeled chance that a party wins a district where it has 55% of "
                "the two-party vote.",
            )
            dpg.add_spacer(height=6)

            self._swing_sigma = dpg.add_slider_float(
                label="Swing sigma (shared)",
                default_value=0.03, min_value=0.005, max_value=0.10,
                format="%.3f", width=220,
            )
            self._tooltip(
                self._swing_sigma,
                "Controls how much the modeled statewide election environment varies.",
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
                "Robust averages Efficiency Gap across modeled election conditions. "
                "Static uses the loaded vote totals directly.",
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
                "Make larger departures from the selected target count more heavily. "
                "When off, the penalty is linear.",
            )
            dpg.add_spacer(height=8)

            self._mm_bound = dpg.add_slider_float(
                label="MM bound",
                default_value=0.20, min_value=0.05, max_value=0.30,
                format="%.2f", width=220,
            )
            self._tooltip(
                self._mm_bound,
                "Mean-Median penalty reaches 100 at this absolute value.",
            )
            self._eg_bound = dpg.add_slider_float(
                label="EG bound",
                default_value=0.35, min_value=0.10, max_value=0.50,
                format="%.2f", width=220,
            )
            self._tooltip(
                self._eg_bound,
                "Efficiency Gap penalty reaches 100 at this absolute value.",
            )
            self._pbias_bound = dpg.add_slider_float(
                label="Partisan Bias bound",
                default_value=0.25, min_value=0.05, max_value=0.50,
                format="%.2f", width=220,
            )
            self._tooltip(
                self._pbias_bound,
                "Partisan Bias penalty reaches 100 at this absolute seat tilt.",
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
                "Keep distinguishing highly competitive maps after the standard "
                "penalty reaches its cap.",
            )

            dpg.add_spacer(height=10)
            dpg.add_separator()
            dpg.add_spacer(height=6)
            self.theme.text("Proportionality", "heading")
            self._prop_unclipped = dpg.add_checkbox(
                label="Unclipped Proportionality", default_value=True,
            )
            self._tooltip(
                self._prop_unclipped,
                "Use a smoother proportionality penalty instead of the standard "
                "capped version.",
            )

    def _build_representation_popup(self):
        with self._dialog(
            "Electoral Opportunity Settings", "popup_representation", (460, 380),
            show=False,
            secondary=("Close",
                       lambda: dpg.configure_item("popup_representation", show=False)),
        ):
            self.theme.text(
                "Uses selected demographic shares as a planning proxy for electoral "
                "opportunity. Requires demographic columns.",
                "muted", wrap=440,
            )
            dpg.add_separator()
            dpg.add_spacer(height=6)

            self._representation_unclipped = dpg.add_checkbox(
                label="Unclipped Electoral Opportunity", default_value=True,
            )
            self._tooltip(
                self._representation_unclipped,
                "Keep distinguishing districts above the standard full-credit level. "
                "This changes the Overall penalty, not the displayed group ratings.",
            )
            self._opportunity_smart_targets = dpg.add_checkbox(
                label="Smart Targets", default_value=True,
            )
            self._tooltip(
                self._opportunity_smart_targets,
                "Estimate each group's reference from nearby map units. This is a "
                "heuristic, not proof that a valid district can be drawn.",
            )
            dpg.add_spacer(height=10)
            dpg.add_separator()
            dpg.add_spacer(height=6)
            self.theme.text("Opportunity curve", "heading")

            self._opportunity_midpoint = dpg.add_slider_float(
                label="Midpoint",
                default_value=0.44, min_value=0.30, max_value=0.65,
                format="%.2f", width=200,
            )
            self._tooltip(
                self._opportunity_midpoint,
                "Group share assigned a 50% opportunity value by the score curve.",
            )
            self._opportunity_steepness = dpg.add_slider_float(
                label="Steepness",
                default_value=0.05, min_value=0.01, max_value=0.15,
                format="%.3f", width=200,
            )
            self._tooltip(
                self._opportunity_steepness,
                "Controls how quickly opportunity value rises around Midpoint. "
                "Smaller values make a sharper threshold.",
            )
            self._opportunity_solid = dpg.add_slider_float(
                label="Solid level",
                default_value=0.55, min_value=0.45, max_value=0.65,
                format="%.2f", width=200,
            )
            self._tooltip(
                self._opportunity_solid,
                "Group share that receives full standard credit.",
            )
            dpg.add_spacer(height=10)
            dpg.add_separator()
            dpg.add_spacer(height=6)
            self.theme.text("Estimated reference", "heading")
            self.theme.text(
                "Heuristic opportunity count by group.",
                "muted", wrap=440,
            )
            dpg.add_spacer(height=4)
            # One item per group so a dispersed group can grey out on its own.
            self._repr_target_lbls = [
                self.theme.text("", "body", wrap=440) for _ in range(len(GROUPS))
            ]
            self._repr_counts_lbl = self.theme.text(
                "Start a run to estimate the reference.", "muted", wrap=440)

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
                    "Mosaic searches for district maps that meet the population limit "
                    "and improve the scores you choose. Most changes redraw two "
                    "neighboring districts; optional n=3 moves and boundary flips "
                    "provide larger or smaller changes.",
                    wrap=420,
                )
                dpg.add_spacer(height=14)

                self.theme.text("Basic workflow", "heading")
                dpg.add_separator()
                dpg.add_text(
                    "Load a shapefile and confirm its columns. Choose the district "
                    "count, run length, and scores. Start the search, compare the "
                    "current and best maps, then save the result you want.",
                    wrap=420,
                )
                dpg.add_spacer(height=14)

                self.theme.text("Learn more", "heading")
                dpg.add_separator()
                dpg.add_text(
                    "The user guide explains settings, scores, saving, and "
                    "troubleshooting. Technical details are in Methodology.",
                    wrap=420,
                )
                dpg.add_spacer(height=8)
                with dpg.group(horizontal=True):
                    dpg.add_button(
                        label="Open user guide",
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
