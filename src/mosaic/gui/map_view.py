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

from typing import Optional

import geopandas as gpd
import numpy as np
from PIL import Image, ImageDraw

import dearpygui.dearpygui as dpg

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

# Classic Mosaic partisan colour scale (PARTISAN_COLORS / PARTISAN_BREAKS from graphics.R)
_PARTISAN_BREAKS = np.array(
    [0.0, 0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9],
    dtype=np.float64,
)
_PARTISAN_RGBA = np.array([
    [168,   0,   0, 255],  # #A80000  deep red
    [194,  27,  24, 255],  # #C21B18  dark red
    [215,  47,  48, 255],  # #D72F30  red
    [215,  93,  93, 255],  # #D75D5D  medium red
    [226, 127, 127, 255],  # #E27F7F  light red
    [255, 178, 178, 255],  # #FFB2B2  very light red/pink
    [211, 217, 255, 255],  # #D3D9FF  very light blue/lavender
    [121, 150, 226, 255],  # #7996E2  light blue
    [102, 116, 222, 255],  # #6674DE  medium blue
    [ 88,  76, 222, 255],  # #584CDE  blue
    [ 57,  51, 229, 255],  # #3933E5  dark blue
    [ 37,  33, 152, 255],  # #252198  deep blue/navy
], dtype=np.uint8)

_BG_COLOR           = np.array([18,  18,  18,  255], dtype=np.uint8)
_BLANK_COLOR        = np.array([55,  55,  55,  220], dtype=np.uint8)
_BORDER_RGBA        = np.array([0,   0,   0,   255], dtype=np.uint8)
_COUNTY_BORDER_RGBA = np.array([180, 180, 180, 255], dtype=np.uint8)
_SPLITS_DIM_RGBA    = np.array([28,  28,  28,  255], dtype=np.uint8)


def _stable_color_mapping(
    current: np.ndarray,
    initial: np.ndarray,
    k: int,
) -> np.ndarray:
    """
    Map current district indices to stable colour indices that best match the
    initial assignment (ported from calculate_stable_color_mapping in
    original Classic Mosaic graphics.R).

    Returns per-precinct array of colour indices in [0, k).
    """
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
        self._county_array: Optional[np.ndarray] = None
        self._dem_votes: Optional[np.ndarray] = None
        self._gop_votes: Optional[np.ndarray] = None
        # Overlay mode flags (set by GUI callbacks)
        self.county_overlay: bool = False
        self.partisan_overlay: bool = False          # colour each precinct by its own partisan lean
        self.district_partisan_overlay: bool = False  # colour each district by its aggregate partisan lean
        self.splits_view: bool = False

    # ── Load (thread-safe, no DPG) ────────────────────────────────────────────

    def load(
        self,
        gdf: gpd.GeoDataFrame,
        county_array: Optional[np.ndarray] = None,
        dem_votes: Optional[np.ndarray] = None,
        gop_votes: Optional[np.ndarray] = None,
    ) -> None:
        """
        Project geometries and rasterise each precinct into pixel_map.
        Safe to call from any thread; does not touch DPG.
        """
        self._county_array = county_array
        self._dem_votes = dem_votes
        self._gop_votes = gop_votes
        W, H = self._w, self._h
        bounds = gdf.total_bounds
        gw = max(bounds[2] - bounds[0], 1e-9)
        gh = max(bounds[3] - bounds[1], 1e-9)
        scale = min(W / gw, H / gh) * 0.96
        ox = (W - gw * scale) / 2.0
        oy = (H - gh * scale) / 2.0
        b0, b1 = float(bounds[0]), float(bounds[1])
        fh = float(H)

        def proj(x: float, y: float) -> tuple[float, float]:
            return ((x - b0) * scale + ox,
                    fh - ((y - b1) * scale + oy))

        img = Image.new("I", (W, H), -1)
        draw = ImageDraw.Draw(img)

        for prec_idx, geom in enumerate(gdf.geometry):
            if geom is None:
                continue
            gt = geom.geom_type
            if gt == "Polygon":
                exteriors = [geom.exterior]
            elif gt == "MultiPolygon":
                exteriors = [part.exterior for part in geom.geoms]
            else:
                continue
            for ring in exteriors:
                pts = [proj(x, y) for x, y in ring.coords[:-1]]
                if len(pts) >= 3:
                    flat = [c for xy in pts for c in xy]
                    draw.polygon(flat, fill=prec_idx)

        self._pixel_map = np.array(img, dtype=np.int32)
        self._n_precincts = len(gdf)
        self._loaded = True

    # ── DPG upload helpers (GUI thread only) ──────────────────────────────────

    @staticmethod
    def _to_dpg(rgba: np.ndarray) -> np.ndarray:
        return (rgba.astype(np.float32) * (1.0 / 255.0)).ravel()

    def _colorise(self, lut: np.ndarray) -> np.ndarray:
        pm = self._pixel_map
        safe = np.where(pm >= 0, pm, self._n_precincts)
        return lut[safe]

    def _build_partisan_lut(self) -> np.ndarray:
        """Per-precinct RGBA LUT using Classic Mosaic's 12-step partisan palette."""
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
        lut[n] = _BG_COLOR
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
        lut[n] = _BG_COLOR
        return lut

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
        lut[n] = _BG_COLOR
        rgba = self._colorise(lut)
        dpg.set_value(self._ttag, self._to_dpg(rgba))

    def render_assignment(
        self,
        assignment: np.ndarray,
        n_districts: int,
        initial: Optional[np.ndarray] = None,
    ) -> None:
        """
        Colourise by district assignment (or partisan lean) and upload texture.
        Overlays are applied in order: splits view, county borders, district borders.
        GUI thread only.
        """
        if not self._loaded:
            return

        pm = self._pixel_map
        n = self._n_precincts

        # ── Base colorization ─────────────────────────────────────────────────
        if self.partisan_overlay and self._dem_votes is not None:
            lut = self._build_partisan_lut()
        elif self.district_partisan_overlay and self._dem_votes is not None:
            lut = self._build_district_partisan_lut(assignment, n_districts)
        else:
            if initial is not None and len(initial) == len(assignment):
                ci = _stable_color_mapping(assignment, initial, n_districts)
            else:
                ci = assignment
            nc = len(DISTRICT_COLORS)
            lut = np.zeros((n + 1, 4), dtype=np.uint8)
            for pi in range(n):
                r, g, b = DISTRICT_COLORS[int(ci[pi]) % nc]
                lut[pi] = (r, g, b, 255)
            lut[n] = _BG_COLOR

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
                rgba[clean_mask] = _SPLITS_DIM_RGBA

            # County borders always visible in splits view
            rgba[self._county_border_mask(pm)] = _COUNTY_BORDER_RGBA

        # ── County overlay (border lines only, when not using splits view) ────
        elif self.county_overlay and self._county_array is not None:
            rgba[self._county_border_mask(pm)] = _COUNTY_BORDER_RGBA

        # ── District borders (always last, on top of everything) ─────────────
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
        rgba[border] = _BORDER_RGBA

        dpg.set_value(self._ttag, self._to_dpg(rgba))
