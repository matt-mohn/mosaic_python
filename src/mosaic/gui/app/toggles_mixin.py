"""Series/panel visibility toggles, tooltips, and hints."""
from ._common import _HINTS, dpg


class TogglesMixin:
    """Series/panel visibility toggles, tooltips, and hints."""

    def _on_ann_toggle(self):
        dpg.configure_item("ann_body",
                           show=dpg.get_value(self._ann_enabled))

    def _on_cool_mode(self):
        guided = dpg.get_value(self._cool_mode) == "Guided (recommended)"
        dpg.configure_item("guided_controls", show=guided)
        dpg.configure_item("static_controls", show=not guided)

    def _on_launch_watch_toggle(self):
        dpg.configure_item("launch_watch_controls",
                           show=dpg.get_value(self._launch_watch_enabled))

    def _on_cut_toggle(self):
        en = dpg.get_value(self._cut_enabled)
        self.theme.retoken(self._cut_lbl,
                           "accent_green" if en else "disabled")
        dpg.configure_item("cut_edge_controls", show=en)

    def _on_cs_toggle(self):
        en = dpg.get_value(self._cs_enabled)
        self.theme.retoken(self._cs_lbl,
                           "accent_green" if en else "disabled")
        dpg.configure_item("cs_controls", show=en)

    def _on_pp_toggle(self):
        en = dpg.get_value(self._pp_enabled)
        self.theme.retoken(self._pp_lbl,
                           "accent_green" if en else "disabled")
        dpg.configure_item("pp_controls", show=en)

    def _on_reock_toggle(self):
        en = dpg.get_value(self._reock_enabled)
        self.theme.retoken(self._reock_lbl,
                           "accent_green" if en else "secondary")
        dpg.configure_item("reock_controls", show=en)

    def _on_alignment_toggle(self):
        en = dpg.get_value(self._alignment_enabled)
        self.theme.retoken(self._alignment_lbl,
                           "accent_green" if en else "secondary")
        dpg.configure_item("alignment_controls", show=en)

    def _on_alignment_focus(self):
        # "Only districts that party wins" is meaningless without a party;
        # gray it out and force it off when focus is neutral.
        neutral = dpg.get_value(self._alignment_focus) == "All residents"
        if neutral:
            dpg.set_value(self._alignment_restrict, False)
        dpg.configure_item(self._alignment_restrict, enabled=not neutral)

    def _on_hc_toggle(self):
        en = dpg.get_value(self._hc_enabled)
        self.theme.retoken(self._hc_lbl,
                           "accent_green" if en else "secondary")
        dpg.configure_item("hc_controls", show=en)

    def _on_hsplit_toggle(self):
        en = dpg.get_value(self._hsplit_enabled)
        self.theme.retoken(self._hsplit_lbl,
                           "accent_green" if en else "disabled")
        dpg.configure_item("hsplit_controls", show=en)
        # County-Edge Bias is paired with County Congruence: enabling the score
        # turns the bias on too (user can still switch it off separately).
        if en:
            dpg.set_value(self._county_bias_enabled, True)
            dpg.configure_item("county_bias_controls", show=True)

    def _on_hprop_toggle(self):
        en = dpg.get_value(self._hprop_enabled)
        self.theme.retoken(self._hprop_lbl,
                           "accent_green" if en else "disabled")
        dpg.configure_item("hprop_controls", show=en)

    def _on_hcmp_toggle(self):
        en = dpg.get_value(self._hcmp_enabled)
        self.theme.retoken(self._hcmp_lbl,
                           "accent_green" if en else "disabled")
        dpg.configure_item("hcmp_controls", show=en)

    def _on_popdev_score_toggle(self):
        en = dpg.get_value(self._popdev_enabled)
        # Green when weighted OR when the Tolerance Ratchet is on -- the ratchet
        # drives population deviation even if this score's own weight is 0.
        ratchet_on = dpg.get_value(self._tolerance_ratchet_mode) != "Off"
        self.theme.retoken(self._popdev_lbl,
                           "accent_green" if (en or ratchet_on) else "secondary")
        dpg.configure_item("popdev_controls", show=en)

    def _on_mm_toggle(self):
        en = dpg.get_value(self._mm_enabled)
        self.theme.retoken(self._mm_lbl,
                           "accent_green" if en else "disabled")
        dpg.configure_item("mm_controls", show=en)

    def _on_eg_toggle(self):
        en = dpg.get_value(self._eg_enabled)
        self.theme.retoken(self._eg_lbl,
                           "accent_green" if en else "disabled")
        dpg.configure_item("eg_controls", show=en)

    def _on_pb_toggle(self):
        en = dpg.get_value(self._pb_enabled)
        self.theme.retoken(self._pb_lbl,
                           "accent_green" if en else "disabled")
        dpg.configure_item("pb_controls", show=en)

    def _on_resp_toggle(self):
        en = dpg.get_value(self._resp_enabled)
        self.theme.retoken(self._resp_lbl,
                           "accent_green" if en else "disabled")
        dpg.configure_item("resp_controls", show=en)

    def _on_pg_toggle(self):
        en = dpg.get_value(self._pg_enabled)
        self.theme.retoken(self._pg_lbl,
                           "accent_green" if en else "disabled")
        dpg.configure_item("pg_controls", show=en)

    def _on_seats_toggle(self):
        en = dpg.get_value(self._seats_enabled)
        self.theme.retoken(self._seats_lbl,
                           "accent_green" if en else "disabled")
        dpg.configure_item("seats_controls", show=en)

    def _on_majority_toggle(self):
        en = dpg.get_value(self._majority_enabled)
        self.theme.retoken(self._majority_lbl,
                           "accent_green" if en else "disabled")
        dpg.configure_item("majority_controls", show=en)

    def _on_majority_dem_chk(self):
        if dpg.get_value(self._majority_dem_chk):
            dpg.set_value(self._majority_rep_chk, False)

    def _on_majority_rep_chk(self):
        if dpg.get_value(self._majority_rep_chk):
            dpg.set_value(self._majority_dem_chk, False)

    def _on_hinge_toggle(self):
        en = dpg.get_value(self._hinge_enabled)
        self.theme.retoken(self._hinge_lbl,
                           "accent_green" if en else "disabled")
        dpg.configure_item("hinge_controls", show=en)

    def _on_hinge_dem_chk(self):
        if dpg.get_value(self._hinge_dem_chk):
            dpg.set_value(self._hinge_rep_chk, False)

    def _on_hinge_rep_chk(self):
        if dpg.get_value(self._hinge_rep_chk):
            dpg.set_value(self._hinge_dem_chk, False)

    def _set_score_row_vis(self, row_tag: str, show: bool,
                           chk_item: int, toggle_cb) -> None:
        """Show/hide a score row; if hiding, force-disable the checkbox."""
        dpg.configure_item(row_tag, show=show)
        if not show and dpg.get_value(chk_item):
            dpg.set_value(chk_item, False)
            toggle_cb()

    def _on_county_bias_toggle(self):
        dpg.configure_item("county_bias_controls",
                           show=dpg.get_value(self._county_bias_enabled))

    def _hint(self, widget: int | str, key: str, delay: float = 0.6) -> None:
        """Attach a hover tooltip to ``widget`` using the named entry from ``_HINTS``.

        ``delay`` is the hover time (seconds) before the tooltip appears.
        """
        text = _HINTS.get(key)
        if text is None:
            return
        with dpg.tooltip(parent=widget, delay=delay):
            dpg.add_text(text, wrap=320)

    def _tooltip(self, widget: int | str, text: str, delay: float = 0.4) -> None:
        """Attach a hover tooltip with literal ``text`` (no _HINTS lookup)."""
        with dpg.tooltip(parent=widget, delay=delay):
            dpg.add_text(text, wrap=320)

    def _show_panel(self, panel_tag: str, menu_item) -> None:
        visible = dpg.is_item_shown(panel_tag)
        dpg.set_value(menu_item, not visible)
        dpg.configure_item(panel_tag, show=not visible)

    def _on_panel_temp_toggle(self):
        dpg.configure_item(
            "panel_temperature", show=dpg.get_value(self._panel_temp_item),
        )

    def _on_temp_panel_close(self):
        dpg.set_value(self._panel_temp_item, False)

    def _on_panel_cs_toggle(self):
        dpg.configure_item("panel_county_splits", show=dpg.get_value(self._panel_cs_item))

    def _on_panel_partisan_toggle(self):
        dpg.configure_item("panel_partisanship", show=dpg.get_value(self._panel_partisan_item))

    def _on_panel_win_chance_toggle(self):
        dpg.configure_item("panel_win_chance", show=dpg.get_value(self._panel_win_chance_item))

    def _on_panel_mm_toggle(self):
        dpg.configure_item("panel_mm", show=dpg.get_value(self._panel_mm_item))

    def _on_panel_eg_toggle(self):
        dpg.configure_item("panel_eg", show=dpg.get_value(self._panel_eg_item))

    def _on_panel_pb_toggle(self):
        dpg.configure_item("panel_pb", show=dpg.get_value(self._panel_pb_item))

    def _on_panel_resp_toggle(self):
        dpg.configure_item("panel_resp", show=dpg.get_value(self._panel_resp_item))

    def _on_panel_pg_toggle(self):
        dpg.configure_item("panel_pg", show=dpg.get_value(self._panel_pg_item))

    def _on_panel_seats_toggle(self):
        dpg.configure_item("panel_dem_seats", show=dpg.get_value(self._panel_seats_item))

    def _on_panel_pp_toggle(self):
        dpg.configure_item("panel_pp", show=dpg.get_value(self._panel_pp_item))

    def _on_panel_reock_toggle(self):
        dpg.configure_item("panel_reock", show=dpg.get_value(self._panel_reock_item))

    def _on_panel_hc_toggle(self):
        dpg.configure_item("panel_hc", show=dpg.get_value(self._panel_hc_item))

    def _on_panel_hsplit_toggle(self):
        dpg.configure_item("panel_hsplit", show=dpg.get_value(self._panel_hsplit_item))

    def _on_panel_hprop_toggle(self):
        dpg.configure_item("panel_hprop", show=dpg.get_value(self._panel_hprop_item))

    def _on_panel_hcmp_toggle(self):
        dpg.configure_item("panel_hcmp", show=dpg.get_value(self._panel_hcmp_item))

    def _on_panel_popdev_toggle(self):
        dpg.configure_item("panel_popdev", show=dpg.get_value(self._panel_popdev_item))

    def _on_panel_alignment_toggle(self):
        dpg.configure_item("panel_alignment", show=dpg.get_value(self._panel_alignment_item))

    def _on_panel_cuts_toggle(self):
        dpg.configure_item("panel_cut_edges", show=dpg.get_value(self._panel_cuts_item))

    def _on_panel_majority_toggle(self):
        dpg.configure_item("panel_majority", show=dpg.get_value(self._panel_majority_item))

    def _on_panel_hinge_toggle(self):
        dpg.configure_item("panel_hinge", show=dpg.get_value(self._panel_hinge_item))

    def _on_panel_contrib_toggle(self):
        dpg.configure_item("panel_score_contrib", show=dpg.get_value(self._panel_contrib_item))

    def _on_panel_district_toggle(self):
        showing = dpg.get_value(self._panel_district_item)
        dpg.configure_item("panel_district_info", show=showing)
        # Force a fresh update next time we render the panel.
        self._dist_info_last_iter = -1

    def _on_district_panel_close(self):
        dpg.set_value(self._panel_district_item, False)
        self._dist_info_last_iter = -1

    def _on_dist_info_interval_change(self):
        # Lower threshold should take effect immediately.
        self._dist_info_last_iter = -1
