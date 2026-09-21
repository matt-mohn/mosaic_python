"""
Choropleth district map rendered via PIL rasterisation + DPG raw texture.

Why PIL instead of DPG draw_polygon:
  DPG's draw_polygon internally calls ImGui's AddConvexPolyFilled, which
  produces incorrect fills for non-convex (concave) polygons -- the "random
  black splotches" seen on real precinct shapefiles.  PIL's ImageDraw.polygon
  uses a proper scanline fill that handles any simple polygon correctly.

Render pipeline:
  load()              -- project coords, rasterise each precinct into a
                         (H, W) int32 pixel_map once per shapefile load.
                         Thread-safe; no DPG calls.
  draw_blank()        -- fill all precincts with neutral grey, upload texture.
  render_assignment() -- numpy LUT lookup to colorise pixel_map in O(W*H),
                         draw 1-px district borders, upload texture.
                         Both GUI-thread only.

Overlay modes (instance flags):
  county_overlay    -- grey county-border lines
  partisan_overlay  -- recolour precincts by per-precinct dem share
                       (Classic Mosaic 12-step red/blue palette)
  splits_view       -- dim non-split counties + draw county borders
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import dearpygui.dearpygui as dpg
import geopandas as gpd
import numpy as np
import shapely
from PIL import Image, ImageDraw, ImageFont

log = logging.getLogger("mosaic")

# District-label font: bundled Inter SemiBold; fall back to Arial then default.
_LABEL_FONT_PATH = (
    Path(__file__).resolve().parent.parent / "assets" / "fonts" / "inter"
    / "Inter-SemiBold.ttf"
)

# Original classic Mosaic 50-colour district palette (from graphics.R)
_HEX = [
    "#b86e6e", "#6e6ec2", "#bbffad", "#ff6e6e", "#ffe86e",
    "#6eb7b7", "#e7ac80", "#aca3e5", "#6effff", "#ff79c2",
    "#b6ff6e", "#a7c3f5", "#f2c3b3", "#b9b96f", "#ffbe6e",
    "#6eff6e", "#9791bd", "#ffff6e", "#c6e38a", "#ffdbe1",
    "#b76e6e", "#ca9d88", "#b8ffea", "#996eb8", "#ebaec2",
    "#6e6eb7", "#6effb6", "#ecc9ec", "#bfd9bf", "#f6b7b7",
    "#f6f1be", "#ff956e", "#93aaee", "#8ae38a", "#c58ae2",
    "#ffcc6e", "#7fc0ff", "#a8e3cf", "#fff0d5", "#d38181",
    "#d2ddec", "#b5fd6e", "#ff6eff", "#df7aba", "#92ede4",
    "#ffa696", "#e9dbe9", "#f9cba5", "#e3a2a2", "#d2edf1",
]
DISTRICT_COLORS: list[tuple[int, int, int]] = [
    (int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)) for h in _HEX
]

# Partisan colour scale on Dem two-party share. Both ends sit 2.0:1 against the
# black district border (relative luminance 0.05); steps are even in L* inward.
# Regenerate: dev/gen_spectrum.py
_PARTISAN_BREAKS = np.array(
    [0.00, 0.10, 0.20, 0.30, 0.35, 0.40, 0.45,
     0.50, 0.55, 0.60, 0.65, 0.70, 0.80, 0.90],
    dtype=np.float64,
)
_PARTISAN_RGBA = np.array([
    [132,   0,  36, 255],  # #840024  wine
    [174,   8,  42, 255],  # #AE082A  deep red
    [218,  22,  41, 255],  # #DA1629  red
    [227,  91,  76, 255],  # #E35B4C  medium red
    [234, 133, 122, 255],  # #EA857A  light red
    [243, 170, 162, 255],  # #F3AAA2  pale red
    [252, 204, 200, 255],  # #FCCCC8  very pale red
    [204, 214, 253, 255],  # #CCD6FD  very pale blue
    [166, 187, 246, 255],  # #A6BBF6  pale blue
    [127, 159, 241, 255],  # #7F9FF1  light blue
    [ 82, 131, 240, 255],  # #5283F0  medium blue
    [ 59,  96, 241, 255],  # #3B60F1  blue
    [ 52,  48, 244, 255],  # #3430F4  deep blue
    [ 49,   0, 204, 255],  # #3100CC  blue-violet
], dtype=np.uint8)

# Demographic chart palette, group order ("white","black","latino","asian"). Hues
# sit off the partisan red/blue axis so a demographic map cannot be mistaken for a
# partisan one. Shared with the demographic score charts
# (panels_mixin) so the three never drift; each is its ramp sampled at 55% share.
_DEMOGRAPHIC_GROUPS = ("white", "black", "latino", "asian")
_DEMOGRAPHIC_RGB = {
    "white":  (183, 88,  0  ),   # orange
    "black":  (16,  126, 146),   # cyan
    "latino": (11,  133, 37 ),   # green
    "asian":  (193, 11,  225),   # magenta
}

# Demographic-overlay ramp, indexed by the dominant group's share in whole
# percent from 35 to 100. Below 35% a district is flat grey; crossing 50% snaps
# chroma from muted to full; 70-100% is compressed; the 100% end sits 2.0:1
# against the black district border. Regenerate: dev/gen_race_v3.py
_DEMO_GRAY = np.array([210, 210, 210], dtype=np.uint8)
_DEMO_RAMP_FLOOR = 0.35
_DEMO_RAMP_HEX = {
    "white":
        "D2D2D2 D4CBC7 D4C6BD D3C0B4 D2BAAB D2B4A3 D0AF9B CFA992 CEA38B CB9E82 CA987A "
        "C9987A CA9779 C99778 C99778 C36007 C05F0C BF5C00 BB5C06 B85B0B B75800 B35704 "
        "B0560B AE5400 AB5305 A8520B A65001 A34F06 A14D00 9E4C02 9A4B06 994900 964803 "
        "924706 914500 8D4403 8D4301 8C4200 894204 894101 884000 854004 853F01 843E00 "
        "823E04 813D01 803C00 7F3B00 7D3B02 7C3A00 7B3900 793902 783800 783700 753702 "
        "753600 743500 723502 713400 703300 6E3302 6D3200 6D3100 6C3000 6A3001 692F00 ",
    "black":
        "D2D2D2 C5CFD1 BACBD0 AFC8CE A5C4CB 9CC0CA 92BDC7 87B9C5 7DB5C2 73B1C0 69ADBD "
        "68ADBD 67ACBC 66ACBC 66ACBC 06889E 0D869B 00849A 048297 0B8095 107E92 037C90 "
        "0A7A8E 0F788B 03768A 097487 007386 067183 0B6F81 016D7F 066B7D 0A697A 016778 "
        "066576 006375 046272 006171 086070 045F6F 005E6E 075D6C 045C6B 005B6A 075A69 "
        "035968 005867 005767 045665 025665 005564 055462 025361 005260 04515F 01505E "
        "004F5D 044E5B 024D5B 004D5A 004C59 024B58 004A57 004956 024854 004753 004653 ",
    "latino":
        "D2D2D2 C8CFC8 C0CBC0 B8C8B8 B0C4B0 A9C0A8 A1BDA1 9AB99A 93B593 8BB18B 84AD84 "
        "84AD84 83AD84 82AC83 82AC82 009025 088E26 0F8B28 008A23 078824 0B8525 008421 "
        "068122 0B7F23 017E1F 057B20 0B7921 01771E 05751E 00741B 01711C 066F1D 006E1A "
        "036B1A 07691B 016818 07661A 056519 016518 006416 056319 026217 006115 056018 "
        "025F16 005E14 055D17 025C15 005B13 055A16 025914 005812 055715 025613 005512 "
        "055414 025312 005211 005210 025012 015011 004F0F 034E11 014D10 004C0E 034B10 ",
    "asian":
        "D2D2D2 D1CBD2 CFC5D1 CDBFD0 CBB9CE C8B4CD C5AECA C2A8C8 C0A2C6 BD9DC4 BA97C1 "
        "BA97C2 B996C2 B996C1 B995C1 CE16F0 CD03F0 C90EEB C614E6 C501E6 C10BE1 BD12DC "
        "BC01DC B80BD7 B413D2 B303D2 AF0CCD AC12C8 AB04C7 A70CC2 A600C2 A205BD 9E0CB9 "
        "9D00B8 9908B3 960DAF 9508AE 9403AE 920DAA 9108A9 9003A9 8D0DA5 8D08A4 8C03A4 "
        "8B00A3 8809A0 88049F 87009E 84099B 84059A 83009A 800996 7F0596 7F0095 7C0991 "
        "7C0591 7B0190 7A008F 77068C 77018B 76008A 730588 730287 720086 6F0583 6F0282 ",
}
_DEMO_RAMP = np.array(
    [[[int(h[i:i + 2], 16) for i in (0, 2, 4)]
      for h in _DEMO_RAMP_HEX[g].split()] for g in _DEMOGRAPHIC_GROUPS],
    dtype=np.uint8,
)   # (4 groups, 66 share steps, 3)

_BG_COLOR           = np.array([18,  18,  18,  255], dtype=np.uint8)
_BLANK_COLOR        = np.array([55,  55,  55,  220], dtype=np.uint8)
_BORDER_RGBA        = np.array([0,   0,   0,   255], dtype=np.uint8)
_COUNTY_BORDER_RGBA = np.array([180, 180, 180, 255], dtype=np.uint8)
_SPLITS_DIM_RGBA    = np.array([28,  28,  28,  255], dtype=np.uint8)

PRECINCT_EDGE_ALPHA = 0.30   # white precinct hairlines when the Precincts overlay is on

# Compactness (Polsby-Popper 0→1): red = not compact, green = compact
_COMPACT_STOPS = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
_COMPACT_RGB   = np.array([
    [190, 50,  50 ],
    [210, 130, 50 ],
    [210, 200, 70 ],
    [100, 185, 85 ],
    [40,  155, 90 ],
], dtype=np.float64)

# Pop. deviation (signed %, mapped through ±_POP_DEV_MAX): blue = under, red = over
_POP_DEV_MAX   = 0.10   # clamp to ±10 %
_POP_DEV_STOPS = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
_POP_DEV_RGB   = np.array([
    [65,  105, 225],
    [135, 165, 225],
    [185, 185, 185],
    [225, 155, 100],
    [200, 60,  60 ],
], dtype=np.float64)

_FOUR_PI = 4.0 * np.pi


def _interp_palette(stops: np.ndarray, rgb: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Interpolate an RGB palette at positions t ∈ [0, 1]. Returns (N, 3) uint8."""
    r = np.interp(t, stops, rgb[:, 0])
    g = np.interp(t, stops, rgb[:, 1])
    b = np.interp(t, stops, rgb[:, 2])
    return np.stack([r, g, b], axis=-1).clip(0, 255).astype(np.uint8)


def _demo_ramp_rgb(shares: np.ndarray) -> np.ndarray:
    """Map (N, 4) group shares to (N, 3) RGB through the demographic ramp.

    Each row takes its largest group's ramp indexed by that group's share in
    whole percent; rows under the floor come back flat grey. Shared by the
    precinct and district overlays so the two cannot drift.
    """
    dom = shares.argmax(axis=1)
    s_dom = shares.max(axis=1)
    # The epsilon keeps a share of exactly 0.50 off the muted side of the break.
    idx = np.clip(np.floor(s_dom * 100.0 + 1e-9).astype(np.int64) - 35,
                  0, _DEMO_RAMP.shape[1] - 1)
    rgb = _DEMO_RAMP[dom, idx]
    return np.where((s_dom < _DEMO_RAMP_FLOOR)[:, None], _DEMO_GRAY, rgb)


def stable_color_mapping(
    current: np.ndarray,
    initial: np.ndarray,
    k: int,
) -> np.ndarray:
    """
    Map current district indices to stable colour indices that best match the
    initial assignment (ported from calculatestable_color_mapping in
    original Classic Mosaic graphics.R).

    Returns per-precinct array of colour indices in [0, k).
    """
    # k < 2 has no colours to disambiguate, and the confidence step below reads
    # s[:, 1] -- the runner-up overlap -- which does not exist on a (1, 1) array.
    # A single-district plan reached this via auto-renumber on run completion.
    if k < 2:
        return np.zeros(len(current), dtype=np.int32)
    overlap = np.zeros((k, k), dtype=np.int32)
    for d in range(k):
        mask = current == d
        if mask.any():
            np.add.at(overlap[d], initial[mask], 1)

    s = np.sort(overlap, axis=1)[:, ::-1]
    conf = s[:, 0].astype(np.int64) - s[:, 1].astype(np.int64)

    color_map = np.arange(k, dtype=np.int32)
    taken = np.zeros(k, dtype=bool)
    for d in np.argsort(conf)[::-1]:
        for pref in np.argsort(overlap[d])[::-1]:
            pref = int(pref)
            if not taken[pref]:
                color_map[d] = pref
                taken[pref] = True
                break

    return color_map[current]


class MapView:
    """
    Rasterises a GeoDataFrame into a per-pixel precinct lookup (pixel_map)
    at load time, then recolours it in O(W*H) numpy ops on each render.
    """

    def __init__(self, texture_tag: str, draw_w: int, draw_h: int):
        self._ttag = texture_tag
        self._w = int(draw_w)
        self._h = int(draw_h)
        self._pixel_map: Optional[np.ndarray] = None   # (H, W) int32; -1 = bg
        self._n_precincts: int = 0
        self._loaded: bool = False
        self._view_bounds = (0.0, 0.0, 1.0, 1.0)
        # Last rendered RGBA (uint8, HxWx4); cached so the GUI can save it to disk.
        self._last_rgba: Optional[np.ndarray] = None
        # Background color used outside polygon pixels; theme can override.
        self._bg_color: np.ndarray = _BG_COLOR.copy()
        # Fill for un-split counties in splits view; exports override it so the
        # dim recedes toward the (white) page instead of going near-black.
        self._splits_dim: np.ndarray = _SPLITS_DIM_RGBA.copy()
        self._county_array: Optional[np.ndarray] = None
        self._dem_votes: Optional[np.ndarray] = None
        self._gop_votes: Optional[np.ndarray] = None
        self._vap: Optional[dict] = None
        self._pp_data = None
        self._reock_data = None
        self._populations: Optional[np.ndarray] = None
        self._precinct_centroids: Optional[np.ndarray] = None  # (N, 2) projected pixel coords
        # Geographic renumbering: (k,) 1-indexed label per stable color index,
        # or None for default stable_index+1. Label-only; colors are unaffected.
        self.district_label_map: Optional[np.ndarray] = None
        # Cache of precise label positions: (assignment_copy, [(d, cx, cy), ...]).
        # Lets a renumber (text-only change) skip the distance transform.
        self._label_centers_cache = None
        # Overlay mode flags (set by GUI callbacks)
        self.county_overlay: bool = False
        self.partisan_overlay: bool = False          # colour each precinct by its own partisan lean
        # colour by racial composition, per district and per precinct
        self.demographic_overlay: bool = False
        self.precinct_demographic_overlay: bool = False
        # colour each district by its aggregate partisan lean
        self.district_partisan_overlay: bool = False
        self.splits_view: bool = False
        # colour each district by combined PP+Reock compactness
        self.compactness_view: bool = False
        self.pop_dev_view: bool = False              # colour each district by population deviation
        self.show_labels: bool = False               # show district number labels
        # True: cheap centroid; False: pole-of-inaccessibility
        self.fast_labels: bool = False
        self.precinct_overlay: bool = False          # faint white precinct boundaries
        self.state_outline: bool = False             # black outline around the state's geometry
        # Multiplier for state/county/district borders and label font size.
        # 1 = native (on-screen); offscreen export sets this from the DPI scale.
        # Precinct overlay is intentionally not scaled — it stays a hairline hint.
        self.border_thickness: int = 1

    # ── Load (thread-safe, no DPG) ────────────────────────────────────────────

    def load(
        self,
        gdf: gpd.GeoDataFrame,
        county_array: Optional[np.ndarray] = None,
        dem_votes: Optional[np.ndarray] = None,
        gop_votes: Optional[np.ndarray] = None,
        pp_data=None,
        reock_data=None,
        populations: Optional[np.ndarray] = None,
        vap_data: Optional[dict] = None,
        view_bounds=None,
        cancelled=None,
    ) -> bool:
        """
        Project geometries and rasterise each precinct into pixel_map.
        Safe to call from any thread; does not touch DPG. Navigation builds a
        separate temporary view and may cancel it before installing its grid.
        """
        self._county_array = county_array
        self._dem_votes = dem_votes
        self._gop_votes = gop_votes
        self._pp_data = pp_data
        self._reock_data = reock_data
        self._populations = populations
        self._vap = vap_data
        W, H = self._w, self._h
        if cancelled is not None and cancelled():
            return False
        geometries = np.asarray(gdf.geometry.array).copy()
        bounds = gdf.total_bounds
        gw = max(bounds[2] - bounds[0], 1e-9)
        gh = max(bounds[3] - bounds[1], 1e-9)
        scale = min(W / gw, H / gh) * 0.96
        ox = (W - gw * scale) / 2.0
        oy = (H - gh * scale) / 2.0
        b0, b1 = float(bounds[0]), float(bounds[1])
        fh = float(H)
        view = tuple(view_bounds) if view_bounds is not None else (0.0, 0.0, 1.0, 1.0)
        vx0, vy0, vx1, vy1 = view
        if not (0 <= vx0 < vx1 <= 1 and 0 <= vy0 < vy1 <= 1):
            raise ValueError("Invalid map viewport")
        sx, sy = vx1 - vx0, vy1 - vy0

        def project(coords):
            # Keep the scalar operation order: pixel rounding must not change.
            pts = np.empty((len(coords), 2), dtype=np.float64)
            pts[:, 0] = ((coords[:, 0] - b0) * scale + ox - vx0 * W) / sx
            pts[:, 1] = (fh - ((coords[:, 1] - b1) * scale + oy) - vy0 * H) / sy
            return pts

        # Cheap geometry bounds rejection, no spatial index or tile cache.
        crop_left = b0 + (vx0 * W - ox) / scale
        crop_right = b0 + (vx1 * W - ox) / scale
        crop_bottom = b1 + (H - vy1 * H - oy) / scale
        crop_top = b1 + (H - vy0 * H - oy) / scale
        visible = np.isin(shapely.get_type_id(geometries), [3, 6])
        if view != (0.0, 0.0, 1.0, 1.0):
            boxes = shapely.bounds(geometries)
            visible &= ((boxes[:, 2] >= crop_left) & (boxes[:, 0] <= crop_right)
                        & (boxes[:, 3] >= crop_bottom) & (boxes[:, 1] <= crop_top))
        precinct_ids = np.flatnonzero(visible)

        img = Image.new("I", (W, H), -1)
        draw = ImageDraw.Draw(img)

        # Batch GEOS extraction and NumPy projection, not individual Python
        # points. Every original ring is still drawn in the original order.
        # Small transient batches also bound memory and cancellation latency.
        for start in range(0, len(precinct_ids), 64):
            if cancelled is not None and cancelled():
                return False
            ids = precinct_ids[start:start + 64]
            parts, parents = shapely.get_parts(geometries[ids], return_index=True)
            rings = shapely.get_exterior_ring(parts)
            counts = shapely.get_num_coordinates(rings)
            points = project(shapely.get_coordinates(rings))
            offset = 0
            for parent, count in zip(parents, counts):
                end = offset + int(count)
                if count >= 4:
                    # The closing coordinate repeats the first, as before.
                    draw.polygon(points[offset:end - 1].ravel().tolist(),
                                 fill=int(ids[parent]))
                offset = end

        # Precompute precinct centroids in pixel coordinates for label placement
        centroids = np.empty((len(geometries), 2), dtype=np.float64)
        for start in range(0, len(geometries), 256):
            if cancelled is not None and cancelled():
                return False
            batch = geometries[start:start + 256]
            centers = shapely.centroid(batch)
            coords = np.column_stack((shapely.get_x(centers), shapely.get_y(centers)))
            projected = project(coords)
            projected[shapely.is_missing(batch) | shapely.is_empty(batch)] = 0.0
            centroids[start:start + len(batch)] = projected
        if cancelled is not None and cancelled():
            return False
        self._pixel_map = np.array(img, dtype=np.int32)
        self._n_precincts = len(gdf)
        self._precinct_centroids = centroids
        self._label_centers_cache = None   # geometry changed; drop stale positions

        self._loaded = True
        self._view_bounds = view
        return True

    # ── DPG upload helpers (GUI thread only) ──────────────────────────────────

    @staticmethod
    def _to_dpg(rgba: np.ndarray) -> np.ndarray:
        return (rgba.astype(np.float32) * (1.0 / 255.0)).ravel()

    def _colorise(self, lut: np.ndarray) -> np.ndarray:
        pm = self._pixel_map
        safe = np.where(pm >= 0, pm, self._n_precincts)
        return lut[safe]

    def _build_partisan_lut(self) -> np.ndarray:
        """Per-precinct RGBA LUT using the partisan palette."""
        n = self._n_precincts
        dem = self._dem_votes.astype(np.float64)
        gop = self._gop_votes.astype(np.float64)
        total = dem + gop
        shares = np.divide(dem, total, out=np.full(len(dem), 0.5), where=total > 0)
        shares_clamped = np.clip(shares, 0.0, 1.0)
        idx = np.searchsorted(_PARTISAN_BREAKS, shares_clamped, side="right") - 1
        idx = np.clip(idx, 0, len(_PARTISAN_RGBA) - 1)
        lut = np.zeros((n + 1, 4), dtype=np.uint8)
        lut[:n] = _PARTISAN_RGBA[idx]
        lut[n] = self._bg_color
        return lut

    def _build_district_partisan_lut(
        self, assignment: np.ndarray, n_districts: int,
    ) -> np.ndarray:
        """Per-precinct LUT coloured by the district's aggregate dem share."""
        n = self._n_precincts
        dem_d = np.bincount(assignment,
                            weights=self._dem_votes.astype(np.float64),
                            minlength=n_districts)
        gop_d = np.bincount(assignment,
                            weights=self._gop_votes.astype(np.float64),
                            minlength=n_districts)
        total_d = dem_d + gop_d
        shares_d = np.divide(dem_d, total_d, out=np.full(len(dem_d), 0.5), where=total_d > 0)
        idx_d = np.searchsorted(_PARTISAN_BREAKS, np.clip(shares_d, 0.0, 1.0),
                                side="right") - 1
        idx_d = np.clip(idx_d, 0, len(_PARTISAN_RGBA) - 1)
        lut = np.zeros((n + 1, 4), dtype=np.uint8)
        lut[:n] = _PARTISAN_RGBA[idx_d[assignment]]
        lut[n] = self._bg_color
        return lut

    def _build_demographic_lut(self, assignment: np.ndarray, n_districts: int) -> np.ndarray:
        """Per-precinct RGBA coloured by each DISTRICT's racial composition: the
        district takes its largest group's ramp, indexed by that group's share --
        grey below 35%, muted up to 50%, full chroma above. orange=White,
        cyan=Black, green=Hispanic, magenta=Asian.

        Future variant to keep in mind: colour by each PRECINCT's own composition
        (aggregate per precinct instead of by district) -- likely a separate
        toggle later."""
        n = self._n_precincts
        vap = self._vap
        tot_d = np.bincount(assignment, weights=np.asarray(vap["total"], dtype=np.float64),
                            minlength=n_districts)
        denom_d = np.where(tot_d > 0, tot_d, 1.0)
        shares_d = np.stack([
            np.bincount(assignment, weights=np.asarray(vap[g], dtype=np.float64),
                        minlength=n_districts) / denom_d
            for g in _DEMOGRAPHIC_GROUPS], axis=1)
        rgb_d = _demo_ramp_rgb(shares_d)
        lut = np.zeros((n + 1, 4), dtype=np.uint8)
        lut[:n, :3] = rgb_d[assignment]
        lut[:n, 3] = 255
        lut[n] = self._bg_color
        return lut

    def _build_precinct_demographic_lut(self) -> np.ndarray:
        """Per-precinct RGBA from each PRECINCT's own racial composition, on the
        same ramp the district overlay uses."""
        n = self._n_precincts
        vap = self._vap
        tot = np.asarray(vap["total"], dtype=np.float64)
        denom = np.where(tot > 0, tot, 1.0)
        shares = np.stack([np.asarray(vap[g], dtype=np.float64) / denom
                           for g in _DEMOGRAPHIC_GROUPS], axis=1)
        lut = np.zeros((n + 1, 4), dtype=np.uint8)
        lut[:n, :3] = _demo_ramp_rgb(shares)
        lut[:n, 3] = 255
        lut[n] = self._bg_color
        return lut

    def _build_compactness_lut(self, assignment: np.ndarray, n_districts: int) -> np.ndarray:
        """Per-precinct LUT coloured by each district's combined compactness: a
        50/50 blend of Polsby-Popper and Reock (the same mix the Compactness
        score uses). Falls back to Polsby-Popper alone if Reock data is absent."""
        n = self._n_precincts
        pd = self._pp_data
        dist_area  = np.bincount(assignment, weights=pd.areas,
                                  minlength=n_districts).astype(np.float64)
        dist_perim = np.bincount(assignment, weights=pd.ext_perimeters,
                                  minlength=n_districts).astype(np.float64)
        eu, ev, elen = pd.edge_u, pd.edge_v, pd.edge_len
        if len(eu) > 0:
            eu_d = assignment[eu]
            ev_d = assignment[ev]
            is_cut = eu_d != ev_d
            if is_cut.any():
                np.add.at(dist_perim, eu_d[is_cut], elen[is_cut])
                np.add.at(dist_perim, ev_d[is_cut], elen[is_cut])
        safe_perim = np.where(dist_perim > 0, dist_perim, 1.0)
        pp_d = np.clip(_FOUR_PI * dist_area / safe_perim ** 2, 0.0, 1.0)
        if self._reock_data is not None:
            from mosaic.scoring.reock import reock_per_district
            reock_d = reock_per_district(assignment, self._reock_data, n_districts)
            compact_d = 0.5 * pp_d + 0.5 * reock_d
        else:
            compact_d = pp_d
        colors_d = _interp_palette(_COMPACT_STOPS, _COMPACT_RGB, compact_d)
        lut = np.zeros((n + 1, 4), dtype=np.uint8)
        lut[:n, :3] = colors_d[assignment]
        lut[:n, 3]  = 255
        lut[n] = self._bg_color
        return lut

    def _build_pop_dev_lut(self, assignment: np.ndarray, n_districts: int) -> np.ndarray:
        """Per-precinct LUT coloured by each district's population deviation from ideal."""
        n = self._n_precincts
        pop_d = np.bincount(assignment,
                            weights=self._populations.astype(np.float64),
                            minlength=n_districts)
        ideal = pop_d.mean() if pop_d.mean() > 0 else 1.0
        dev   = (pop_d - ideal) / ideal
        t     = np.clip(dev / _POP_DEV_MAX, -1.0, 1.0) * 0.5 + 0.5
        colors_d = _interp_palette(_POP_DEV_STOPS, _POP_DEV_RGB, t)
        lut = np.zeros((n + 1, 4), dtype=np.uint8)
        lut[:n, :3] = colors_d[assignment]
        lut[:n, 3]  = 255
        lut[n] = self._bg_color
        return lut

    def _thicken(self, mask: np.ndarray) -> np.ndarray:
        """Dilate a boolean border mask by (border_thickness - 1) pixels."""
        if self.border_thickness <= 1:
            return mask
        from scipy.ndimage import binary_dilation
        return binary_dilation(mask, iterations=self.border_thickness - 1)

    def _county_border_mask(self, pm: np.ndarray) -> np.ndarray:
        """Boolean mask of pixels that lie on county borders."""
        ca = self._county_array
        nca = len(ca)
        cmap = np.where(pm >= 0, ca[np.clip(pm, 0, nca - 1)], -1)
        cbh = cmap[:-1, :] != cmap[1:, :]
        cbv = cmap[:, :-1] != cmap[:, 1:]
        vch = (cmap[:-1, :] >= 0) & (cmap[1:, :] >= 0)
        vcv = (cmap[:, :-1] >= 0) & (cmap[:, 1:] >= 0)
        cb = np.zeros(pm.shape, dtype=bool)
        cb[:-1, :] |= cbh & vch
        cb[1:,  :] |= cbh & vch
        cb[:, :-1] |= cbv & vcv
        cb[:, 1:]  |= cbv & vcv
        return cb

    def draw_blank(self) -> None:
        """Upload neutral-grey map (all precincts same colour). GUI thread."""
        if not self._loaded:
            return
        n = self._n_precincts
        lut = np.full((n + 1, 4), _BLANK_COLOR, dtype=np.uint8)
        lut[n] = self._bg_color
        rgba = self._colorise(lut)
        self._last_rgba = rgba
        dpg.set_value(self._ttag, self._to_dpg(rgba))

    def wipe(self) -> None:
        """Clear the canvas to solid background and drop all loaded geometry.

        Called by File > New so the old shapefile outline doesn't persist.
        """
        bg = np.full((self._h, self._w, 4), self._bg_color, dtype=np.uint8)
        dpg.set_value(self._ttag, self._to_dpg(bg))
        self._loaded = False
        self._pixel_map = None
        self._last_rgba = None
        self._n_precincts = 0
        self._view_bounds = (0.0, 0.0, 1.0, 1.0)
        self._county_array = None
        self._dem_votes = None
        self._gop_votes = None
        self._vap = None
        self._pp_data = None
        self._reock_data = None
        self._populations = None
        self._precinct_centroids = None
        self._label_centers_cache = None

    def render_assignment(
        self,
        assignment: np.ndarray,
        n_districts: int,
        initial: Optional[np.ndarray] = None,
    ) -> None:
        """Compose the current frame's rgba and upload to DPG. GUI thread only."""
        rgba = self.compose_rgba(assignment, n_districts, initial)
        if rgba is None:
            return
        self._last_rgba = rgba
        dpg.set_value(self._ttag, self._to_dpg(rgba))

    def compose_rgba(
        self,
        assignment: np.ndarray,
        n_districts: int,
        initial: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """
        Build the colourised RGBA frame without uploading. Returns None if the
        view isn't loaded or the assignment doesn't match.
        Overlays are applied in order: splits view, county borders, district borders.
        """
        if not self._loaded:
            return None

        # Safety: if assignment length != loaded precinct count, skip rather
        # than index past the array. Guards an IndexError when an edited
        # re-import of the same path slips past the reload trigger; the real
        # fix tracks gdf identity, this is the backstop.
        if len(assignment) != self._n_precincts:
            log.warning(
                f"MapView render skipped: assignment length {len(assignment)} "
                f"does not match loaded precincts {self._n_precincts}. "
                "Map is stale; reload the shapefile."
            )
            return None

        pm = self._pixel_map
        n = self._n_precincts

        # ── Base colorization ─────────────────────────────────────────────────
        if self.partisan_overlay and self._dem_votes is not None:
            lut = self._build_partisan_lut()
        elif self.district_partisan_overlay and self._dem_votes is not None:
            lut = self._build_district_partisan_lut(assignment, n_districts)
        elif self.compactness_view and self._pp_data is not None:
            lut = self._build_compactness_lut(assignment, n_districts)
        elif self.pop_dev_view and self._populations is not None:
            lut = self._build_pop_dev_lut(assignment, n_districts)
        elif self.precinct_demographic_overlay and self._vap is not None:
            lut = self._build_precinct_demographic_lut()
        elif self.demographic_overlay and self._vap is not None:
            lut = self._build_demographic_lut(assignment, n_districts)
        else:
            if initial is not None and len(initial) == len(assignment):
                ci = stable_color_mapping(assignment, initial, n_districts)
            else:
                ci = assignment
            nc = len(DISTRICT_COLORS)
            lut = np.zeros((n + 1, 4), dtype=np.uint8)
            for pi in range(n):
                r, g, b = DISTRICT_COLORS[int(ci[pi]) % nc]
                lut[pi] = (r, g, b, 255)
            lut[n] = self._bg_color

        rgba = self._colorise(lut).copy()

        # ── Splits view (dim non-split counties, always draw county borders) ──
        if self.splits_view and self._county_array is not None:
            ca = self._county_array
            n_counties = int(ca.max()) + 1
            flat_idx = (ca * n_districts + assignment).astype(np.int64)
            co_di = np.bincount(
                flat_idx, minlength=n_counties * n_districts,
            ).reshape(n_counties, n_districts)
            county_is_clean = (co_di > 0).sum(axis=1) <= 1  # True = not split

            nca = len(ca)
            cmap_vals = np.where(pm >= 0, ca[np.clip(pm, 0, nca - 1)], -1)
            valid = pm >= 0
            clean_mask = np.zeros(pm.shape, dtype=bool)
            clean_mask[valid] = county_is_clean[cmap_vals[valid]]

            if clean_mask.any():
                rgba[clean_mask] = self._splits_dim

            # County borders always visible in splits view
            rgba[self._thicken(self._county_border_mask(pm))] = _COUNTY_BORDER_RGBA

        # ── County overlay (border lines only, when not using splits view) ────
        elif self.county_overlay and self._county_array is not None:
            rgba[self._thicken(self._county_border_mask(pm))] = _COUNTY_BORDER_RGBA

        # ── Precinct boundaries (faint white, alpha-blended) ─────────────────
        if self.precinct_overlay:
            pb_h = (pm[:-1, :] != pm[1:, :]) & (pm[:-1, :] >= 0) & (pm[1:, :] >= 0)
            pb_v = (pm[:, :-1] != pm[:, 1:]) & (pm[:, :-1] >= 0) & (pm[:, 1:] >= 0)
            pb_mask = np.zeros(pm.shape, dtype=bool)
            pb_mask[:-1, :] |= pb_h
            pb_mask[1:,  :] |= pb_h
            pb_mask[:, :-1] |= pb_v
            pb_mask[:, 1:]  |= pb_v
            if pb_mask.any():
                alpha = PRECINCT_EDGE_ALPHA
                blended = rgba[pb_mask].astype(np.float32)
                blended[:, :3] = blended[:, :3] * (1.0 - alpha) + 255.0 * alpha
                rgba[pb_mask] = blended.astype(np.uint8)

        # ── District borders (over precinct boundaries) ──────────────────────
        dist_map = np.where(pm >= 0, assignment[np.clip(pm, 0, n - 1)], -1)
        bh = dist_map[:-1, :] != dist_map[1:, :]
        bv = dist_map[:, :-1] != dist_map[:, 1:]
        vh = (dist_map[:-1, :] >= 0) & (dist_map[1:, :] >= 0)
        vv = (dist_map[:, :-1] >= 0) & (dist_map[:, 1:] >= 0)
        border = np.zeros(pm.shape, dtype=bool)
        border[:-1, :] |= bh & vh
        border[1:,  :] |= bh & vh
        border[:, :-1] |= bv & vv
        border[:, 1:]  |= bv & vv
        rgba[self._thicken(border)] = _BORDER_RGBA

        # ── State outline (above district borders so it's a clean edge) ──────
        if self.state_outline:
            valid = pm >= 0
            so = np.zeros(pm.shape, dtype=bool)
            so[:-1, :] |= valid[:-1, :] & ~valid[1:,  :]
            so[1:,  :] |= valid[1:,  :] & ~valid[:-1, :]
            so[:, :-1] |= valid[:, :-1] & ~valid[:, 1:]
            so[:, 1:]  |= valid[:, 1:]  & ~valid[:, :-1]
            rgba[self._thicken(so)] = _BORDER_RGBA

        # ── District labels (if enabled) ─────────────────────────────────────
        if self.show_labels and self._precinct_centroids is not None:
            # Compute stable label numbers (matching color assignment)
            if initial is not None and len(initial) == len(assignment):
                stable_colors = stable_color_mapping(assignment, initial, n_districts)
            else:
                stable_colors = assignment

            # Build mapping: current district -> stable label number. With a
            # geographic renumber active, the displayed number is label_map of
            # the stable color index (label-only; the color is still the stable
            # index, so renumbering moves numbers, not colors).
            lm = self.district_label_map
            use_lm = lm is not None and len(lm) == n_districts
            dist_to_label = {}
            for d in range(n_districts):
                mask = assignment == d
                if mask.any():
                    si = int(stable_colors[mask][0])
                    dist_to_label[d] = int(lm[si]) if use_lm else si + 1

            # Label placement is the hot path while annealing runs (the map
            # re-renders every accepted step).  We have two modes:
            #   fast_labels=True  -> cheap mean of precinct centroids per
            #     district.  Can drift outside a concave district but is
            #     microseconds, so safe to run every frame.
            #   fast_labels=False -> pole of inaccessibility via per-district
            #     scipy distance transform.  Guaranteed on-surface even for
            #     U-shaped districts, but costs ~10-100ms per render.  Used
            #     when the algorithm is paused/idle and the user is actually
            #     inspecting the map.
            # Label POSITIONS (d, cx, cy) depend only on the assignment and the
            # placement mode, NOT on the displayed numbers. The precise path
            # runs a per-district distance transform (~10-100ms); cache its
            # output keyed by assignment so a geographic renumber -- which moves
            # no district, only the text -- reuses positions instead of paying
            # the transform again. Only the precise path is cached (the fast
            # path is microseconds and its assignment changes every frame).
            centers = None
            if not self.fast_labels and self._label_centers_cache is not None:
                c_assign, c_centers = self._label_centers_cache
                if (len(c_assign) == len(assignment)
                        and np.array_equal(c_assign, assignment)):
                    centers = c_centers
            if centers is None:
                centers = []
                if self.fast_labels:
                    pc = self._precinct_centroids
                    for d in range(n_districts):
                        mask = assignment == d
                        if not mask.any():
                            continue
                        cx, cy = pc[mask].mean(axis=0)
                        centers.append((d, float(cx), float(cy)))
                else:
                    from scipy.ndimage import distance_transform_edt
                    pm = self._pixel_map
                    pm_valid = pm >= 0
                    # Per-pixel district id (only meaningful where pm_valid)
                    pm_safe = np.where(pm_valid, pm, 0)
                    pixel_district = assignment[pm_safe]

                    for d in range(n_districts):
                        d_mask = pm_valid & (pixel_district == d)
                        if not d_mask.any():
                            continue
                        # One background pixel around the district contains
                        # every possible nearest-zero boundary. Cropping the
                        # EDT preserves distances and row-major argmax ties.
                        rows = np.flatnonzero(d_mask.any(axis=1))
                        cols = np.flatnonzero(d_mask.any(axis=0))
                        y0 = max(0, int(rows[0]) - 1)
                        y1 = min(d_mask.shape[0], int(rows[-1]) + 2)
                        x0 = max(0, int(cols[0]) - 1)
                        x1 = min(d_mask.shape[1], int(cols[-1]) + 2)
                        dist = distance_transform_edt(d_mask[y0:y1, x0:x1])
                        cy, cx = np.unravel_index(int(dist.argmax()), dist.shape)
                        centers.append((d, float(cx + x0), float(cy + y0)))
                    self._label_centers_cache = (assignment.copy(), centers)

            # Sort by approximate area (larger districts first for priority).
            # sorted() returns a new list so we never mutate the cached centers.
            areas = np.bincount(assignment, minlength=n_districts)
            centers = sorted(centers, key=lambda c: -areas[c[0]])

            # Greedy collision avoidance — scale glyph size with border_thickness
            # so labels stay legible at high-DPI exports.
            font_h = 14 * self.border_thickness
            char_w = 8 * self.border_thickness
            placed_boxes = []
            labels_to_draw = []

            for d, cx, cy in centers:
                label = str(dist_to_label.get(d, d + 1))
                half_w = len(label) * char_w / 2 + 2
                half_h = font_h / 2 + 2
                box = (cx - half_w, cy - half_h, cx + half_w, cy + half_h)

                # Check collision
                collision = False
                for pb in placed_boxes:
                    if not (box[2] < pb[0] or box[0] > pb[2] or
                            box[3] < pb[1] or box[1] > pb[3]):
                        collision = True
                        break

                if not collision:
                    placed_boxes.append(box)
                    labels_to_draw.append((int(cx), int(cy), label))

            # Draw labels onto rgba via PIL
            if labels_to_draw:
                img = Image.fromarray(rgba, mode="RGBA")
                draw = ImageDraw.Draw(img)
                font = None
                use_anchor = False
                if _LABEL_FONT_PATH.exists():
                    try:
                        font = ImageFont.truetype(str(_LABEL_FONT_PATH), font_h)
                        use_anchor = True
                    except OSError:
                        font = None
                if font is None:
                    try:
                        font = ImageFont.truetype("arial.ttf", font_h)
                        use_anchor = True
                    except OSError:
                        font = ImageFont.load_default()
                        use_anchor = False

                # Outline scales with border_thickness so it stays visible
                # against high-DPI glyphs (PIL's built-in stroke_width).
                stroke_w = max(1, self.border_thickness)

                for px, py, text in labels_to_draw:
                    if use_anchor:
                        draw.text((px, py), text,
                                  fill=(255, 255, 255, 255), font=font,
                                  anchor="mm",
                                  stroke_width=stroke_w,
                                  stroke_fill=(0, 0, 0, 255))
                    else:
                        bbox = draw.textbbox((0, 0), text, font=font)
                        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                        tx, ty = px - tw // 2, py - th // 2
                        draw.text((tx, ty), text,
                                  fill=(255, 255, 255, 255), font=font,
                                  stroke_width=stroke_w,
                                  stroke_fill=(0, 0, 0, 255))

                rgba = np.array(img, dtype=np.uint8)

        return rgba
