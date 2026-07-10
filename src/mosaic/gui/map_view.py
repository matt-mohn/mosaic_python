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

import geopandas as gpd
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import dearpygui.dearpygui as dpg

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
        # Last rendered RGBA (uint8, HxWx4); cached so the GUI can save it to disk.
        self._last_rgba: Optional[np.ndarray] = None
        # Background color used outside polygon pixels; theme can override.
        self._bg_color: np.ndarray = _BG_COLOR.copy()
        self._county_array: Optional[np.ndarray] = None
        self._dem_votes: Optional[np.ndarray] = None
        self._gop_votes: Optional[np.ndarray] = None
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
        self.district_partisan_overlay: bool = False  # colour each district by its aggregate partisan lean
        self.splits_view: bool = False
        self.compactness_view: bool = False          # colour each district by combined PP+Reock compactness
        self.pop_dev_view: bool = False              # colour each district by population deviation
        self.show_labels: bool = False               # show district number labels
        self.fast_labels: bool = False               # True: cheap centroid; False: pole-of-inaccessibility
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
    ) -> None:
        """
        Project geometries and rasterise each precinct into pixel_map.
        Safe to call from any thread; does not touch DPG.
        """
        self._county_array = county_array
        self._dem_votes = dem_votes
        self._gop_votes = gop_votes
        self._pp_data = pp_data
        self._reock_data = reock_data
        self._populations = populations
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

        # Precompute precinct centroids in pixel coordinates for label placement
        centroids = []
        for geom in gdf.geometry:
            if geom is not None:
                c = geom.centroid
                centroids.append(proj(c.x, c.y))
            else:
                centroids.append((0.0, 0.0))
        self._precinct_centroids = np.array(centroids, dtype=np.float64)
        self._label_centers_cache = None   # geometry changed; drop stale positions

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
        self._county_array = None
        self._dem_votes = None
        self._gop_votes = None
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
                rgba[clean_mask] = _SPLITS_DIM_RGBA

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
                        dist = distance_transform_edt(d_mask)
                        cy, cx = np.unravel_index(int(dist.argmax()), dist.shape)
                        centers.append((d, float(cx), float(cy)))
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
