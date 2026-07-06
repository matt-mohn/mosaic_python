"""Color palettes, global themes, and runtime swap manager for the Mosaic GUI.

Design
------
The GUI uses semantic *tokens* (``heading``, ``muted``, ``accent_green`` ...)
instead of raw RGB tuples.  Each token resolves to a color via the active
``Palette``.  Two palettes ship today (``DARK``, ``LIGHT``); more can be added
without touching call sites.

``ThemeManager`` owns the active palette, builds one DPG global theme per
palette, and tracks every inline-colored text item so it can re-color them
when the palette swaps.  The same registry is the natural place to later
extend for fonts and spacing tokens.

Usage
-----
    self.theme = ThemeManager()             # in __init__
    self.theme.build()                      # after dpg.create_context()
    self.theme.apply("modern")              # bind initial theme

    self.theme.text("Heading", "heading")   # tracked text widget
    self.theme.retoken(tag, "muted")        # change a tag's token at runtime
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import dearpygui.dearpygui as dpg

log = logging.getLogger("mosaic")


# ── Palette ───────────────────────────────────────────────────────────────────

Color = tuple[int, int, int, int]
_FONTS_DIR = Path(__file__).resolve().parent.parent / "assets" / "fonts"


def _rgb(r: int, g: int, b: int, a: int = 255) -> Color:
    return (r, g, b, a)


@dataclass(frozen=True)
class Typography:
    """Font definitions for a design mode.

    None paths → use DPG's default font.  Sizes are honored even when paths
    are None (DPG ignores them in that case, but the structure stays uniform).
    """
    body_path: Optional[Path] = None     # primary UI font
    bold_path: Optional[Path] = None     # used for headings/title (optional)
    body_size: int = 13                  # px
    heading_size: int = 14               # px (uses bold_path if set)
    title_size: int = 18                 # px (uses bold_path if set)


@dataclass(frozen=True)
class Spacing:
    """Layout style tokens — rounding, padding, border thickness."""
    frame_rounding: float = 0.0
    child_rounding: float = 0.0
    popup_rounding: float = 0.0
    window_rounding: float = 0.0
    grab_rounding: float = 0.0
    tab_rounding: float = 0.0
    scrollbar_rounding: float = 0.0
    frame_padding: tuple[int, int] = (4, 3)
    item_spacing: tuple[int, int] = (8, 4)
    item_inner_spacing: tuple[int, int] = (4, 4)
    window_padding: tuple[int, int] = (8, 8)
    frame_border_size: float = 0.0
    child_border_size: float = 1.0
    window_border_size: float = 1.0
    popup_border_size: float = 1.0


@dataclass(frozen=True)
class Palette:
    """Semantic color tokens for one design mode (dark or light)."""

    name: str

    # ── Surfaces ──────────────────────────────────────────────────────────────
    window_bg: Color
    child_bg: Color
    popup_bg: Color
    menubar_bg: Color
    frame_bg: Color
    frame_bg_hovered: Color
    frame_bg_active: Color
    border: Color
    separator: Color
    scrollbar_bg: Color
    scrollbar_grab: Color
    scrollbar_grab_hovered: Color
    scrollbar_grab_active: Color

    # ── Text tokens (used by add_text(..., color=...)) ────────────────────────
    body: Color           # default body text
    title: Color          # app title accent
    heading: Color        # section heading
    subheading: Color     # input labels
    muted: Color          # info text
    secondary: Color      # semi-active label
    accent_green: Color   # active score row label
    success_soft: Color   # soft green text
    success_pale: Color   # success indicator
    disabled: Color       # inactive score row label
    disabled_deep: Color  # heavily disabled
    disabled_subtle: Color  # special disabled label
    dialog_muted: Color   # dialog hint / path text
    error: Color          # error text
    warning: Color        # warning text
    ok: Color             # success confirmation
    dem: Color            # Dem party color
    gop: Color            # Rep party color

    # ── Buttons (default DPG state) ──────────────────────────────────────────
    button: Color
    button_hovered: Color
    button_active: Color
    button_text: Color

    # ── Nudge buttons (recommended primary action) ───────────────────────────
    nudge_button: Color
    nudge_hovered: Color
    nudge_active: Color
    nudge_text: Color

    # ── Anti-nudge buttons (available but not primary) ───────────────────────
    antinudge_button: Color
    antinudge_hovered: Color
    antinudge_active: Color
    antinudge_text: Color

    # ── Check / slider grabs ─────────────────────────────────────────────────
    check_mark: Color
    slider_grab: Color
    slider_grab_active: Color

    # ── Menu / header (selectable bg) ────────────────────────────────────────
    header: Color
    header_hovered: Color
    header_active: Color

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab: Color
    tab_hovered: Color
    tab_active: Color

    # ── Plots ────────────────────────────────────────────────────────────────
    plot_bg: Color
    plot_border: Color
    plot_axis_grid: Color
    plot_axis_text: Color
    plot_legend_bg: Color
    plot_legend_border: Color

    # ── Tables ───────────────────────────────────────────────────────────────
    table_header_bg: Color
    table_border_strong: Color
    table_border_light: Color
    table_row_bg: Color
    table_row_bg_alt: Color

    # ── Typography + spacing ─────────────────────────────────────────────────
    typography: Typography = field(default_factory=Typography)
    spacing: Spacing = field(default_factory=Spacing)


# ── Shared Modern styling (typography + spacing) ─────────────────────────────
# Both LIGHT and DARK palettes use these — only colors differ between them.

_INTER_REGULAR  = _FONTS_DIR / "inter" / "Inter-Regular.ttf"
_INTER_SEMIBOLD = _FONTS_DIR / "inter" / "Inter-SemiBold.ttf"

_MODERN_TYPOGRAPHY = Typography(
    body_path=_INTER_REGULAR,
    bold_path=_INTER_SEMIBOLD,
    body_size=14,
    heading_size=15,
    title_size=20,
)

_MODERN_SPACING = Spacing(
    frame_rounding=5.0,
    child_rounding=4.0,
    popup_rounding=6.0,
    window_rounding=0.0,        # main window stays square — feels grounded
    grab_rounding=5.0,
    tab_rounding=4.0,
    scrollbar_rounding=4.0,
    frame_padding=(8, 5),
    item_spacing=(8, 5),
    item_inner_spacing=(5, 4),
    window_padding=(10, 10),
    frame_border_size=0.0,      # bg contrast carries inputs
    child_border_size=1.0,
    window_border_size=0.0,
    popup_border_size=1.0,
)


# ── LIGHT — light surfaces with Modern styling (default) ─────────────────────

LIGHT = Palette(
    name="light",
    # Surfaces — cool, neutral light
    window_bg=_rgb(248, 249, 250),
    child_bg=_rgb(255, 255, 255),
    popup_bg=_rgb(252, 252, 253),
    menubar_bg=_rgb(238, 240, 244),
    frame_bg=_rgb(240, 243, 247),
    frame_bg_hovered=_rgb(228, 232, 240),
    frame_bg_active=_rgb(212, 220, 232),
    border=_rgb(220, 226, 234),
    separator=_rgb(228, 232, 240),
    scrollbar_bg=_rgb(240, 242, 245),
    scrollbar_grab=_rgb(195, 200, 208),
    scrollbar_grab_hovered=_rgb(165, 172, 182),
    scrollbar_grab_active=_rgb(140, 148, 160),
    # Text — ~7:1 contrast on light surfaces
    body=_rgb(28, 30, 34),
    title=_rgb(184, 110, 0),
    heading=_rgb(140, 95, 0),
    subheading=_rgb(85, 92, 105),
    muted=_rgb(110, 116, 124),
    secondary=_rgb(95, 102, 115),
    accent_green=_rgb(30, 130, 60),
    success_soft=_rgb(60, 130, 80),
    success_pale=_rgb(40, 120, 70),
    disabled=_rgb(120, 126, 138),
    disabled_deep=_rgb(140, 146, 158),
    disabled_subtle=_rgb(130, 136, 148),
    dialog_muted=_rgb(115, 122, 134),
    error=_rgb(190, 35, 40),
    warning=_rgb(180, 110, 0),
    ok=_rgb(35, 125, 65),
    dem=_rgb(50, 95, 185),
    gop=_rgb(190, 50, 50),
    # Buttons — near-white with strong text contrast
    button=_rgb(244, 246, 250),
    button_hovered=_rgb(225, 230, 240),
    button_active=_rgb(205, 213, 228),
    button_text=_rgb(22, 26, 35),
    # Nudge — saturated indigo
    nudge_button=_rgb(80, 105, 175),
    nudge_hovered=_rgb(100, 128, 200),
    nudge_active=_rgb(125, 152, 220),
    nudge_text=_rgb(255, 255, 255),
    # Anti-nudge
    antinudge_button=_rgb(216, 220, 228),
    antinudge_hovered=_rgb(200, 206, 218),
    antinudge_active=_rgb(186, 194, 210),
    antinudge_text=_rgb(85, 92, 105),
    # Checks / sliders
    check_mark=_rgb(60, 100, 180),
    slider_grab=_rgb(125, 145, 185),
    slider_grab_active=_rgb(80, 110, 175),
    # Headers
    header=_rgb(220, 228, 240),
    header_hovered=_rgb(200, 212, 232),
    header_active=_rgb(175, 192, 220),
    # Tabs
    tab=_rgb(228, 232, 238),
    tab_hovered=_rgb(210, 218, 230),
    tab_active=_rgb(245, 247, 250),
    # Plots
    plot_bg=_rgb(252, 252, 253),
    plot_border=_rgb(210, 214, 220),
    plot_axis_grid=_rgb(225, 228, 233),
    plot_axis_text=_rgb(70, 78, 90),
    plot_legend_bg=_rgb(252, 252, 253, 230),
    plot_legend_border=_rgb(210, 214, 220),
    # Tables — subtle stripe contrast, blue-tinted header
    table_header_bg=_rgb(228, 234, 244),
    table_border_strong=_rgb(208, 215, 224),
    table_border_light=_rgb(232, 236, 242),
    table_row_bg=_rgb(255, 255, 255),
    table_row_bg_alt=_rgb(244, 247, 252),
    typography=_MODERN_TYPOGRAPHY,
    spacing=_MODERN_SPACING,
)


# ── DARK — dark surfaces with Modern styling ─────────────────────────────────

DARK = Palette(
    name="dark",
    # Surfaces — neutral dark with a hint of cool
    window_bg=_rgb(28, 30, 34),
    child_bg=_rgb(36, 39, 44),
    popup_bg=_rgb(32, 34, 38),
    menubar_bg=_rgb(24, 26, 30),
    frame_bg=_rgb(52, 56, 64),
    frame_bg_hovered=_rgb(64, 70, 80),
    frame_bg_active=_rgb(78, 86, 100),
    border=_rgb(58, 64, 74),
    separator=_rgb(54, 60, 70),
    scrollbar_bg=_rgb(24, 26, 30),
    scrollbar_grab=_rgb(70, 76, 88),
    scrollbar_grab_hovered=_rgb(95, 102, 116),
    scrollbar_grab_active=_rgb(120, 128, 145),
    # Text
    body=_rgb(232, 234, 240),
    title=_rgb(255, 200, 60),
    heading=_rgb(220, 200, 110),
    subheading=_rgb(180, 184, 195),
    muted=_rgb(150, 156, 168),
    secondary=_rgb(195, 200, 212),
    accent_green=_rgb(110, 220, 110),
    success_soft=_rgb(140, 180, 145),
    success_pale=_rgb(160, 210, 170),
    disabled=_rgb(110, 116, 128),
    disabled_deep=_rgb(85, 90, 100),
    disabled_subtle=_rgb(125, 130, 142),
    dialog_muted=_rgb(135, 140, 152),
    error=_rgb(230, 95, 95),
    warning=_rgb(230, 170, 70),
    ok=_rgb(130, 210, 140),
    dem=_rgb(130, 160, 255),
    gop=_rgb(255, 130, 130),
    # Buttons
    button=_rgb(58, 64, 74),
    button_hovered=_rgb(76, 84, 98),
    button_active=_rgb(95, 105, 122),
    button_text=_rgb(232, 234, 240),
    # Nudge
    nudge_button=_rgb(95, 110, 165),
    nudge_hovered=_rgb(125, 142, 200),
    nudge_active=_rgb(155, 172, 225),
    nudge_text=_rgb(255, 255, 255),
    # Anti-nudge
    antinudge_button=_rgb(42, 46, 54),
    antinudge_hovered=_rgb(56, 62, 72),
    antinudge_active=_rgb(72, 80, 92),
    antinudge_text=_rgb(140, 148, 162),
    # Checks / sliders
    check_mark=_rgb(160, 200, 245),
    slider_grab=_rgb(115, 130, 165),
    slider_grab_active=_rgb(150, 170, 215),
    # Headers
    header=_rgb(70, 80, 100),
    header_hovered=_rgb(90, 102, 125),
    header_active=_rgb(115, 130, 158),
    # Tabs
    tab=_rgb(40, 44, 52),
    tab_hovered=_rgb(65, 72, 88),
    tab_active=_rgb(90, 100, 122),
    # Plots
    plot_bg=_rgb(24, 26, 30),
    plot_border=_rgb(58, 64, 74),
    plot_axis_grid=_rgb(50, 55, 64),
    plot_axis_text=_rgb(190, 195, 205),
    plot_legend_bg=_rgb(32, 34, 38, 230),
    plot_legend_border=_rgb(58, 64, 74),
    # Tables
    table_header_bg=_rgb(52, 58, 68),
    table_border_strong=_rgb(58, 64, 74),
    table_border_light=_rgb(48, 54, 64),
    table_row_bg=_rgb(36, 39, 44),
    table_row_bg_alt=_rgb(44, 48, 56),
    typography=_MODERN_TYPOGRAPHY,
    spacing=_MODERN_SPACING,
)


PALETTES = {"light": LIGHT, "dark": DARK}


# ── Theme builder ─────────────────────────────────────────────────────────────

def _add_style_entries(p: Palette) -> None:
    """Inside an active theme_component, add style entries from p.spacing."""
    s = p.spacing
    # Vec2 styles (x, y)
    dpg.add_theme_style(dpg.mvStyleVar_FramePadding, *s.frame_padding)
    dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, *s.item_spacing)
    dpg.add_theme_style(dpg.mvStyleVar_ItemInnerSpacing, *s.item_inner_spacing)
    dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, *s.window_padding)
    # Scalar styles
    dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, s.frame_rounding)
    dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, s.child_rounding)
    dpg.add_theme_style(dpg.mvStyleVar_PopupRounding, s.popup_rounding)
    dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, s.window_rounding)
    dpg.add_theme_style(dpg.mvStyleVar_GrabRounding, s.grab_rounding)
    dpg.add_theme_style(dpg.mvStyleVar_TabRounding, s.tab_rounding)
    dpg.add_theme_style(dpg.mvStyleVar_ScrollbarRounding, s.scrollbar_rounding)
    dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, s.frame_border_size)
    dpg.add_theme_style(dpg.mvStyleVar_ChildBorderSize, s.child_border_size)
    dpg.add_theme_style(dpg.mvStyleVar_WindowBorderSize, s.window_border_size)
    dpg.add_theme_style(dpg.mvStyleVar_PopupBorderSize, s.popup_border_size)


def _build_global_theme(p: Palette) -> int:
    """Build a DPG global theme tag from a palette."""
    with dpg.theme() as theme_tag:
        # ── Default component (applies to most widgets, incl. combo arrows) ───
        # NOTE: button colors live here (NOT in a mvButton sub-component) so
        # combo dropdown arrows, selectables, and other button-rendering
        # widgets pick them up too.  Per-item themes (nudge/antinudge) still
        # override for specific buttons.
        with dpg.theme_component(dpg.mvAll):
            # Surfaces
            dpg.add_theme_color(dpg.mvThemeCol_WindowBg, p.window_bg)
            dpg.add_theme_color(dpg.mvThemeCol_ChildBg, p.child_bg)
            dpg.add_theme_color(dpg.mvThemeCol_PopupBg, p.popup_bg)
            dpg.add_theme_color(dpg.mvThemeCol_MenuBarBg, p.menubar_bg)
            dpg.add_theme_color(dpg.mvThemeCol_FrameBg, p.frame_bg)
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, p.frame_bg_hovered)
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive, p.frame_bg_active)
            dpg.add_theme_color(dpg.mvThemeCol_Border, p.border)
            dpg.add_theme_color(dpg.mvThemeCol_Separator, p.separator)
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarBg, p.scrollbar_bg)
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrab, p.scrollbar_grab)
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrabHovered, p.scrollbar_grab_hovered)
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrabActive, p.scrollbar_grab_active)
            # Buttons (global — applies to all button-rendering widgets)
            dpg.add_theme_color(dpg.mvThemeCol_Button, p.button)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, p.button_hovered)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, p.button_active)
            # Text
            dpg.add_theme_color(dpg.mvThemeCol_Text, p.body)
            dpg.add_theme_color(dpg.mvThemeCol_TextDisabled, p.disabled)
            dpg.add_theme_color(dpg.mvThemeCol_InputTextCursor, p.body)
            # Headers (selectables, collapsing headers, tree nodes — also combo items)
            dpg.add_theme_color(dpg.mvThemeCol_Header, p.header)
            dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered, p.header_hovered)
            dpg.add_theme_color(dpg.mvThemeCol_HeaderActive, p.header_active)
            # Title bar
            dpg.add_theme_color(dpg.mvThemeCol_TitleBg, p.menubar_bg)
            dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive, p.window_bg)
            dpg.add_theme_color(dpg.mvThemeCol_TitleBgCollapsed, p.menubar_bg)
            # Tabs
            dpg.add_theme_color(dpg.mvThemeCol_Tab, p.tab)
            dpg.add_theme_color(dpg.mvThemeCol_TabHovered, p.tab_hovered)
            dpg.add_theme_color(dpg.mvThemeCol_TabActive, p.tab_active)

            dpg.add_theme_color(dpg.mvThemeCol_TableHeaderBg, p.table_header_bg)
            dpg.add_theme_color(dpg.mvThemeCol_TableBorderStrong, p.table_border_strong)
            dpg.add_theme_color(dpg.mvThemeCol_TableBorderLight, p.table_border_light)
            dpg.add_theme_color(dpg.mvThemeCol_TableRowBg, p.table_row_bg)
            dpg.add_theme_color(dpg.mvThemeCol_TableRowBgAlt, p.table_row_bg_alt)
            # Checks / sliders
            dpg.add_theme_color(dpg.mvThemeCol_CheckMark, p.check_mark)
            dpg.add_theme_color(dpg.mvThemeCol_SliderGrab, p.slider_grab)
            dpg.add_theme_color(dpg.mvThemeCol_SliderGrabActive, p.slider_grab_active)
            # Spacing / rounding / border styles
            _add_style_entries(p)

        # ── Plots ────────────────────────────────────────────────────────────
        with dpg.theme_component(dpg.mvPlot):
            dpg.add_theme_color(dpg.mvPlotCol_FrameBg, p.plot_bg,
                                category=dpg.mvThemeCat_Plots)
            dpg.add_theme_color(dpg.mvPlotCol_PlotBg, p.plot_bg,
                                category=dpg.mvThemeCat_Plots)
            dpg.add_theme_color(dpg.mvPlotCol_PlotBorder, p.plot_border,
                                category=dpg.mvThemeCat_Plots)
            dpg.add_theme_color(dpg.mvPlotCol_LegendBg, p.plot_legend_bg,
                                category=dpg.mvThemeCat_Plots)
            dpg.add_theme_color(dpg.mvPlotCol_LegendBorder, p.plot_legend_border,
                                category=dpg.mvThemeCat_Plots)
            dpg.add_theme_color(dpg.mvPlotCol_AxisGrid, p.plot_axis_grid,
                                category=dpg.mvThemeCat_Plots)
            dpg.add_theme_color(dpg.mvPlotCol_AxisText, p.plot_axis_text,
                                category=dpg.mvThemeCat_Plots)
            dpg.add_theme_color(dpg.mvPlotCol_AxisTick, p.plot_axis_text,
                                category=dpg.mvThemeCat_Plots)

    return theme_tag


def _build_nudge_theme(p: Palette, nudge: bool) -> int:
    """Build a button theme tag for nudge / anti-nudge buttons.

    Colours are defined for both the enabled and the disabled state. Without an
    explicit disabled-state component, a disabled button falls back to the
    global button colour (a lighter grey); pinning the disabled colours keeps a
    greyed-out button the same dark anti-nudge grey as its enabled siblings.
    """
    if nudge:
        btn, hov, act, txt = (p.nudge_button, p.nudge_hovered,
                              p.nudge_active, p.nudge_text)
    else:
        btn, hov, act, txt = (p.antinudge_button, p.antinudge_hovered,
                              p.antinudge_active, p.antinudge_text)
    with dpg.theme() as theme_tag:
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, btn)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, hov)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, act)
            dpg.add_theme_color(dpg.mvThemeCol_Text, txt)
        with dpg.theme_component(dpg.mvButton, enabled_state=False):
            dpg.add_theme_color(dpg.mvThemeCol_Button, btn)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, btn)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, btn)
            dpg.add_theme_color(dpg.mvThemeCol_Text, p.disabled)
    return theme_tag


# ── ThemeManager ──────────────────────────────────────────────────────────────

class ThemeManager:
    """Owns the active palette and re-applies themed colors when it changes.

    Tracks each text widget by ``(tag, token_name)`` so on palette swap every
    inline color can be rebound without rebuilding the UI.
    """

    def __init__(self, initial: str = "dark"):
        self.palette: Palette = PALETTES[initial]
        self._global_themes: dict[str, int] = {}
        self._nudge_themes: dict[str, int] = {}      # palette name -> theme tag
        self._antinudge_themes: dict[str, int] = {}
        # palette name -> body font tag (0 = DPG default)
        self._body_fonts: dict[str, int] = {}
        # (item_tag, token_name) for every tracked text widget
        self._tracked: list[tuple[int | str, str]] = []

    # ── Build (call after dpg.create_context()) ───────────────────────────────

    def build(self) -> None:
        # Shared font registry — created once, even if no palette uses fonts
        dpg.add_font_registry()
        for name, p in PALETTES.items():
            self._global_themes[name] = _build_global_theme(p)
            self._nudge_themes[name] = _build_nudge_theme(p, nudge=True)
            self._antinudge_themes[name] = _build_nudge_theme(p, nudge=False)
            self._body_fonts[name] = self._register_font(p.typography)

    def _register_font(self, typo: Typography) -> int:
        """Register the body font from a Typography spec; 0 if default."""
        if typo.body_path is None:
            return 0
        if not typo.body_path.exists():
            log.warning(f"theme: font not found at {typo.body_path}; using DPG default")
            return 0
        with dpg.font_registry():
            font_tag = dpg.add_font(str(typo.body_path), typo.body_size)
        return font_tag

    # ── Token lookup ──────────────────────────────────────────────────────────

    def color(self, token: str) -> Color:
        return getattr(self.palette, token)

    @property
    def nudge_theme(self) -> int:
        return self._nudge_themes[self.palette.name]

    @property
    def antinudge_theme(self) -> int:
        return self._antinudge_themes[self.palette.name]

    # ── Tracked widget helpers ────────────────────────────────────────────────

    def text(self, label: str, token: str = "body", **kwargs) -> int | str:
        """``dpg.add_text`` that remembers its token for later re-color."""
        item = dpg.add_text(label, color=self.color(token), **kwargs)
        self._tracked.append((item, token))
        return item

    def track(self, item: int | str, token: str) -> int | str:
        """Mark an existing text item for theme-managed coloring."""
        dpg.configure_item(item, color=self.color(token))
        self._tracked.append((item, token))
        return item

    def retoken(self, item: int | str, token: str) -> None:
        """Change the token bound to a tracked item (e.g. enabled <-> disabled)."""
        for i, (tag, _) in enumerate(self._tracked):
            if tag == item:
                self._tracked[i] = (tag, token)
                break
        else:
            self._tracked.append((item, token))
        dpg.configure_item(item, color=self.color(token))

    # ── Apply / swap ──────────────────────────────────────────────────────────

    def apply(self, name: str) -> None:
        """Switch palettes and re-color everything tracked."""
        if name not in PALETTES:
            return
        self.palette = PALETTES[name]
        dpg.bind_theme(self._global_themes[name])
        # Bind font (0 = DPG bitmap default).  DPG accepts 0 to reset to default.
        try:
            dpg.bind_font(self._body_fonts.get(name, 0))
        except SystemError:
            # Older DPG can't bind 0 — fall back to leaving the current font.
            pass
        for tag, token in self._tracked:
            if dpg.does_item_exist(tag):
                dpg.configure_item(tag, color=self.color(token))
