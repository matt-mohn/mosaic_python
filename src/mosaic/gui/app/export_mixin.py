"""CSV/metric export and map image save (PNG/PDF workers)."""
from ._common import (
    _ASSETS_DIR,
    _MAP_DW,
    _PDF_PRECINCT_OFF_ALPHA,
    PRECINCT_EDGE_ALPHA,
    MapView,
    Optional,
    Path,
    dpg,
    np,
    output_dir,
    threading,
)

# Exports always render on white with dark text, regardless of the app theme.
_EXPORT_BG = (255, 255, 255, 255)
_EXPORT_FG = (28, 30, 34, 255)
# Splits view dims un-split counties; on the white page they recede to light grey.
_EXPORT_SPLITS_DIM = (230, 232, 236, 255)


class ExportMixin:
    """CSV/metric export and map image save (PNG/PDF workers)."""

    def _on_export(self):
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir() / f"assignments_{timestamp}.csv"
        self._do_export_to_path(output_path)

    def _do_export_to_path(self, output_path: Path) -> None:
        """Export labeled assignment CSV to a specific path."""
        with self.state._lock:
            current = (self.state.current_assignment.copy()
                       if self.state.current_assignment is not None else None)
        if self.runner is None or current is None:
            return
        from mosaic.io import save_assignments
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        precinct_ids = None
        id_col_name = "precinct_id"
        if self.runner.gdf is not None and self._loaded_config is not None:
            col = self._loaded_config.id_col
            if col and col in self.runner.gdf.columns:
                precinct_ids = self.runner.gdf[col].tolist()
                id_col_name = col

        save_assignments(
            self._export_labeled_assignment(current),
            output_path,
            precinct_ids=precinct_ids,
            id_col_name=id_col_name,
        )
        self.state.update(status_message=f"Assignments saved to {output_path}")
        self._saved_plan = current.copy()   # mark the plan clean for the unsaved-changes guard

    def _on_export_metrics(self):
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir() / f"metrics_{timestamp}.csv"
        self._do_export_metrics_to_path(output_path)

    def _do_export_metrics_to_path(self, output_path: Path) -> None:
        """Export per-district metrics CSV to a specific path."""
        with self.state._lock:
            current = (self.state.current_assignment.copy()
                       if self.state.current_assignment is not None else None)
        if self.runner is None or current is None:
            return
        from mosaic.io import save_metrics
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        n_dist = self.state.num_districts
        ideal_pop = (float(self.runner.populations.sum()) / n_dist
                     if n_dist > 0 else 1.0)
        dem = self.runner.election_arrays[0][0] if self.runner.election_arrays else None
        gop = self.runner.election_arrays[0][1] if self.runner.election_arrays else None
        cfg = self.state.score_config

        save_metrics(
            self._export_labeled_assignment(current),
            output_path,
            populations=self.runner.populations,
            ideal_pop=ideal_pop,
            dem_votes=dem,
            gop_votes=gop,
            pp_data=self.runner.pp_data,
            reock_data=self.runner.reock_data,
            county_ids=self.runner.county_array,
            win_prob_at_55=cfg.election_win_prob_at_55,
            swing_sigma=cfg.election_swing_sigma,
        )
        self.state.update(status_message=f"Metrics saved to {output_path}")

    # ── File menu: New / recent files / named saves ───────────────────────────

    def _on_file_save_assignments(self) -> None:
        """File > Save Assignments: native OS save dialog into output/, with an
        assignments_<timestamp>.csv default name."""
        if self.state.best_assignment is None:
            return
        import os
        from datetime import datetime
        default_name = f"assignments_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if os.name == "nt":
            path = self._native_save_csv(default_name)
            if not path:
                return
            out = Path(path)
            if out.suffix.lower() != ".csv":
                out = out.with_suffix(".csv")
            self._do_export_to_path(out)
        else:
            # No native save dialog off Windows (tkinter/DPG run-loop conflict);
            # fall back to an auto-named file in output/.
            self._do_export_to_path(output_dir() / f"{default_name}.csv")

    def _native_save_csv(self, default_name: str) -> str:
        """Windows native Save-As for a CSV, opened to output/. Returns the
        chosen path, or "" on cancel."""
        import subprocess
        out_dir = output_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        # TopMost owner so the dialog can't hide behind the app (see Save As).
        ps = (
            "Add-Type -AssemblyName System.Windows.Forms; "
            "$owner = New-Object System.Windows.Forms.Form; "
            "$owner.TopMost = $true; "
            "$d = New-Object System.Windows.Forms.SaveFileDialog; "
            "$d.Filter = 'CSV (*.csv)|*.csv'; "
            "$d.DefaultExt = 'csv'; $d.AddExtension = $true; "
            f"$d.InitialDirectory = '{out_dir}'; "
            f"$d.FileName = '{default_name}'; "
            "$null = $d.ShowDialog($owner); "
            "$owner.Dispose(); "
            "Write-Output $d.FileName"
        )
        try:
            r = subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
                capture_output=True, text=True, timeout=120,
            )
            return r.stdout.strip()
        except Exception:
            return ""

    def _on_file_save_metrics(self) -> None:
        """File > Save District Info: native OS save dialog into output/, with a
        metrics_<timestamp>.csv default name."""
        if self.state.best_assignment is None:
            return
        import os
        from datetime import datetime
        default_name = f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if os.name == "nt":
            path = self._native_save_csv(default_name)
            if not path:
                return
            out = Path(path)
            if out.suffix.lower() != ".csv":
                out = out.with_suffix(".csv")
            self._do_export_metrics_to_path(out)
        else:
            # No native save dialog off Windows (tkinter/DPG run-loop conflict);
            # fall back to an auto-named file in output/.
            self._do_export_metrics_to_path(output_dir() / f"{default_name}.csv")

    def _render_map_at_scale(self, scale: float,
                             state_outline: bool = False) -> Optional[np.ndarray]:
        """Re-rasterize the current map at `scale`x resolution, cropped to the
        state's aspect ratio, and return rgba.

        Long edge is `_MAP_DW * scale` regardless of orientation; short edge is
        derived from the gdf's bounding-box aspect. Falls back to LANCZOS
        upscale of the cached frame if the runner/gdf isn't available.
        """
        src = self.map_view
        if src is None or not src._loaded:
            return None

        if (self.runner is None or self.runner.gdf is None
                or self.state.current_assignment is None):
            if src._last_rgba is None:
                return None
            if scale == 1.0:
                return src._last_rgba.copy()
            from PIL import Image
            img = Image.fromarray(src._last_rgba, mode="RGBA")
            new_size = (int(img.width * scale), int(img.height * scale))
            return np.asarray(
                img.resize(new_size, Image.LANCZOS), dtype=np.uint8,
            )

        bounds = self.runner.gdf.total_bounds
        gw = max(float(bounds[2] - bounds[0]), 1e-9)
        gh = max(float(bounds[3] - bounds[1]), 1e-9)
        long_edge = max(1, int(_MAP_DW * scale))
        if gw >= gh:
            w = long_edge
            h = max(1, int(round(long_edge * gh / gw)))
        else:
            h = long_edge
            w = max(1, int(round(long_edge * gw / gh)))
        offscreen = MapView(texture_tag="__offscreen", draw_w=w, draw_h=h)
        offscreen._bg_color = np.array(_EXPORT_BG, dtype=np.uint8)
        offscreen._splits_dim = np.array(_EXPORT_SPLITS_DIM, dtype=np.uint8)
        for flag in ("county_overlay", "partisan_overlay",
                     "district_partisan_overlay", "splits_view",
                     "compactness_view", "pop_dev_view", "show_labels",
                     "precinct_overlay"):
            setattr(offscreen, flag, getattr(src, flag))
        offscreen.fast_labels = False  # precise label placement for export
        offscreen.state_outline = state_outline
        # Half-rate proportionality: doubling DPI grows lines/labels by ~50%,
        # not 100%. Keeps high-DPI exports from over-emphasizing borders.
        offscreen.border_thickness = max(1, int(round(1 + 0.5 * (scale - 1))))

        dem = (self.runner.election_arrays[0][0]
               if self.runner.election_arrays else None)
        gop = (self.runner.election_arrays[0][1]
               if self.runner.election_arrays else None)
        offscreen.load(
            self.runner.gdf,
            county_array=self.runner.county_array,
            dem_votes=dem, gop_votes=gop,
            pp_data=self.runner.pp_data,
            reock_data=self.runner.reock_data,
            populations=self.runner.populations,
        )

        with self.state._lock:
            assignment = self.state.current_assignment.copy()
            initial = (self.state.initial_assignment.copy()
                       if self.state.initial_assignment is not None else None)
            n_dist = self.state.num_districts
        return offscreen.compose_rgba(assignment, n_dist, initial)

    def _on_save_map(self):
        if getattr(self, "_saving", False):
            return
        if self.map_view is None or not self.map_view._loaded:
            self.state.update(status_message="Nothing to save: load a shapefile first.")
            return
        self._saving = True
        dpg.configure_item(self._save_spinner, show=True)
        self.state.update(status_message="Saving map...")

        def _worker():
            from datetime import datetime

            from PIL import Image
            try:
                rgba = self._render_map_at_scale(1.0)
                if rgba is None:
                    self.state.update(
                        status_message="Save failed: map not ready.",
                    )
                    return
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = output_dir() / f"map_{timestamp}.png"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(rgba, mode="RGBA").save(output_path)
                self.state.update(status_message=f"Map saved to {output_path}")
            finally:
                dpg.configure_item(self._save_spinner, show=False)
                self._saving = False
        threading.Thread(target=_worker, daemon=True).start()

    def _on_advanced_save_open(self):
        if self.map_view is None or self.map_view._last_rgba is None:
            self.state.update(status_message="Nothing to save: load a shapefile first.")
            return
        dpg.configure_item("popup_adv_save", show=True)

    def _on_adv_fmt_changed(self):
        """Show Raster DPI row only when PNG is selected."""
        is_pdf = "PDF" in dpg.get_value(self._adv_save_fmt)
        dpg.configure_item(self._adv_dpi_group, show=not is_pdf)

    def _adv_save_ready(self) -> bool:
        return (self.map_view is not None
                and self.map_view._loaded
                and self.state.current_assignment is not None)

    def _adv_begin(self, status: str) -> None:
        self._saving = True
        for btn in (self._adv_save_btn, self._adv_save_as_btn,
                    self._adv_close_btn):
            dpg.configure_item(btn, enabled=False)
        dpg.configure_item(self._adv_save_spinner, show=True)
        dpg.set_value(self._adv_save_status, status)
        dpg.configure_item(self._adv_save_status, show=True)

    def _adv_finish(self, close: bool = True) -> None:
        dpg.configure_item(self._adv_save_spinner, show=False)
        dpg.configure_item(self._adv_save_status, show=False)
        for btn in (self._adv_save_btn, self._adv_save_as_btn,
                    self._adv_close_btn):
            dpg.configure_item(btn, enabled=True)
        if close:
            dpg.configure_item("popup_adv_save", show=False)
        self._saving = False

    def _adv_get_dpi(self) -> int:
        if not hasattr(self, "_adv_save_dpi") or self._adv_save_dpi is None:
            return 192
        return int(dpg.get_value(self._adv_save_dpi).split()[0])

    def _adv_dispatch(self, output_path) -> None:
        """Start the PNG or PDF worker for the current format selection."""
        fmt_str = dpg.get_value(self._adv_save_fmt) if self._adv_save_fmt else ""
        is_pdf = "PDF" in fmt_str
        title = dpg.get_value(self._adv_save_title).strip()
        dpi = self._adv_get_dpi()
        if is_pdf:
            threading.Thread(
                target=self._pdf_vector_worker,
                args=(title, output_path),
                daemon=True,
            ).start()
        else:
            threading.Thread(
                target=self._png_worker,
                args=(title, dpi, output_path),
                daemon=True,
            ).start()

    def _on_advanced_save_confirm(self):
        """Save to an auto-timestamped path."""
        if getattr(self, "_saving", False):
            return
        if not self._adv_save_ready():
            self.state.update(status_message="Nothing to save: load a shapefile first.")
            dpg.configure_item("popup_adv_save", show=False)
            return
        fmt_str = dpg.get_value(self._adv_save_fmt) if self._adv_save_fmt else ""
        is_pdf = "PDF" in fmt_str
        dpi = self._adv_get_dpi()
        self._adv_begin("Rendering PDF..." if is_pdf else f"Rendering {dpi} DPI...")
        self._adv_dispatch(output_path=None)

    def _on_advanced_save_as(self):
        """Save to a chosen path (Windows native dialog); auto-name elsewhere."""
        if getattr(self, "_saving", False):
            return
        if not self._adv_save_ready():
            self.state.update(status_message="Nothing to save: load a shapefile first.")
            return
        fmt_str = dpg.get_value(self._adv_save_fmt) if self._adv_save_fmt else ""
        is_pdf = "PDF" in fmt_str
        import os
        if os.name != "nt":
            # No native save dialog off Windows: PowerShell is Windows-only, and a
            # tkinter dialog would risk the mac/DPG run-loop conflict the Open
            # picker already avoids. Fall back to an auto-named file in output/.
            dpi = self._adv_get_dpi()
            self._adv_begin("Rendering PDF..." if is_pdf else f"Rendering {dpi} DPI...")
            self._adv_dispatch(output_path=None)
            return
        self._adv_begin("Waiting for save dialog...")

        def _ask():
            import subprocess
            from datetime import datetime as _dt
            ext = ".pdf" if is_pdf else ".png"
            filt = ("PDF Document|*.pdf" if is_pdf else "PNG Image|*.png")
            ts = _dt.now().strftime("%Y%m%d_%H%M%S")
            default_name = f"map_{ts}"
            out_dir = output_dir()
            out_dir.mkdir(parents=True, exist_ok=True)
            init_dir = str(out_dir)
            # TopMost owner keeps the dialog above the (still-responsive) DPG
            # window: without it, clicking back on Mosaic buries the ownerless
            # dialog and the export hangs at "Waiting for save dialog...".
            ps = (
                "Add-Type -AssemblyName System.Windows.Forms; "
                "$owner = New-Object System.Windows.Forms.Form; "
                "$owner.TopMost = $true; "
                "$d = New-Object System.Windows.Forms.SaveFileDialog; "
                f"$d.Filter = '{filt}'; "
                f"$d.DefaultExt = '{ext.lstrip('.')}'; "
                f"$d.InitialDirectory = '{init_dir}'; "
                f"$d.FileName = '{default_name}'; "
                "$null = $d.ShowDialog($owner); "
                "$owner.Dispose(); "
                "Write-Output $d.FileName"
            )
            try:
                r = subprocess.run(
                    ["powershell", "-NoProfile", "-NonInteractive",
                     "-Command", ps],
                    capture_output=True, text=True, timeout=120,
                )
                path = r.stdout.strip()
            except Exception:
                path = ""
            if not path:
                self._adv_finish(close=False)
                return
            dpi = self._adv_get_dpi()
            dpg.set_value(self._adv_save_status,
                          "Rendering PDF..." if is_pdf
                          else f"Rendering {dpi} DPI...")
            self._adv_dispatch(output_path=Path(path))

        threading.Thread(target=_ask, daemon=True).start()

    # ── Save workers ─────────────────────────────────────────────────────────

    def _pdf_vector_worker(self, title: str, output_path) -> None:
        """Build a true vector PDF on US Letter paper (auto landscape/portrait).
        Precinct fills and all outlines are real vector paths, not a raster."""
        from datetime import datetime
        try:
            if self.runner is None or self.runner.gdf is None:
                self.state.update(
                    status_message="PDF save failed: no shapefile loaded.")
                return
            with self.state._lock:
                assignment = self.state.current_assignment.copy()
                initial = (self.state.initial_assignment.copy()
                           if self.state.initial_assignment is not None
                           else None)
                n_dist = self.state.num_districts

            gdf = self.runner.gdf
            mv  = self.map_view
            n   = len(gdf)

            # ── Per-precinct fill colours (same logic as MapView LUTs) ────────
            if mv.partisan_overlay and mv._dem_votes is not None:
                lut = mv._build_partisan_lut()
                fill_rgba = lut[:n]
            elif mv.district_partisan_overlay and mv._dem_votes is not None:
                lut = mv._build_district_partisan_lut(assignment, n_dist)
                fill_rgba = lut[:n]
            elif mv.compactness_view and mv._pp_data is not None:
                lut = mv._build_compactness_lut(assignment, n_dist)
                fill_rgba = lut[:n]
            elif mv.pop_dev_view and mv._populations is not None:
                lut = mv._build_pop_dev_lut(assignment, n_dist)
                fill_rgba = lut[:n]
            else:
                from mosaic.gui.map_view import DISTRICT_COLORS, stable_color_mapping
                nc = len(DISTRICT_COLORS)
                ci_arr = (stable_color_mapping(assignment, initial, n_dist)
                          if initial is not None and len(initial) == len(assignment)
                          else assignment)
                fill_rgba = np.zeros((n, 4), dtype=np.uint8)
                for i in range(n):
                    r, g, b = DISTRICT_COLORS[int(ci_arr[i]) % nc]
                    fill_rgba[i] = (r, g, b, 255)

            fill_mpl = fill_rgba[:, :3] / 255.0

            # ── Matplotlib setup ──────────────────────────────────────────────
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
            import matplotlib.patheffects as _pe
            from matplotlib.collections import PatchCollection
            from matplotlib.font_manager import FontProperties
            from matplotlib.patches import PathPatch
            from matplotlib.path import Path as MplPath
            from shapely.ops import unary_union as _shapely_union

            bounds = gdf.total_bounds
            bx0, by0, bx1, by1 = (float(v) for v in bounds)
            gw = max(bx1 - bx0, 1e-9)
            gh = max(by1 - by0, 1e-9)
            map_ratio = gw / gh

            br, bg_c, bb, _ = _EXPORT_BG
            bg_mpl = (br / 255.0, bg_c / 255.0, bb / 255.0)

            # ── Page layout: US Letter, fixed-inch title/caption bands ────────
            # The map is fit as large as possible inside the printable area
            # (page minus PAGE_MARGIN) after reserving fixed-height bands, in
            # whichever orientation (portrait 8.5x11 / landscape 11x8.5) yields
            # the larger MAP. Bands are a fixed number of inches, so they never
            # scale with — and shrink — the map. The caption+map+title stack is
            # centered vertically; the map is centered horizontally.
            PAGE_MARGIN = 0.4                        # inches, printer safe area
            TITLE_BAND  = 0.5 if title else 0.0      # inches, fixed title band
            CAP_BAND    = 0.35                       # inches, fixed caption band

            def _fit_map(fw, fh):
                aw = fw - 2 * PAGE_MARGIN
                ah = fh - 2 * PAGE_MARGIN - TITLE_BAND - CAP_BAND
                if aw <= 0 or ah <= 0:
                    return 0.0, 0.0, 0.0
                if aw / ah >= map_ratio:
                    mh, mw = ah, ah * map_ratio
                else:
                    mw, mh = aw, aw / map_ratio
                return mw, mh, mw * mh

            _pw, _ph, p_area = _fit_map(8.5, 11.0)
            _lw, _lh, l_area = _fit_map(11.0, 8.5)
            if p_area >= l_area:
                fig_w, fig_h = 8.5, 11.0
                map_w, map_h = _pw, _ph
            else:
                fig_w, fig_h = 11.0, 8.5
                map_w, map_h = _lw, _lh

            map_left     = (fig_w - map_w) / 2.0
            stack_bottom = (PAGE_MARGIN
                            + ((fig_h - 2 * PAGE_MARGIN)
                               - (CAP_BAND + map_h + TITLE_BAND)) / 2.0)

            ax_l   = map_left / fig_w
            ax_b   = (stack_bottom + CAP_BAND) / fig_h
            ax_w   = map_w / fig_w
            ax_h_f = map_h / fig_h

            # Text anchors (figure fractions): caption at 90% width, centered in
            # the bottom band; title at 50% width, centered in the top band.
            cap_x   = (map_left + 0.90 * map_w) / fig_w
            cap_y   = (stack_bottom + 0.5 * CAP_BAND) / fig_h
            title_x = (map_left + 0.50 * map_w) / fig_w
            title_y = (stack_bottom + CAP_BAND + map_h + 0.5 * TITLE_BAND) / fig_h

            fig = plt.figure(figsize=(fig_w, fig_h))
            fig.patch.set_facecolor(bg_mpl)

            ax = fig.add_axes([ax_l, ax_b, ax_w, ax_h_f])
            ax.set_facecolor(bg_mpl)
            ax.set_aspect("equal")
            ax.set_axis_off()
            ax.set_xlim(bx0, bx1)
            ax.set_ylim(by0, by1)

            def _to_mpl_path(geom):
                """Shapely Polygon/MultiPolygon → matplotlib Path with holes."""
                if geom is None:
                    return None
                gtype = geom.geom_type
                if gtype == "Polygon":
                    polys = [geom]
                elif gtype == "MultiPolygon":
                    polys = list(geom.geoms)
                else:
                    return None
                verts, codes = [], []
                for poly in polys:
                    coords = np.asarray(poly.exterior.coords)
                    if len(coords) < 3:
                        continue
                    verts.append(coords)
                    codes.append(np.array(
                        [MplPath.MOVETO]
                        + [MplPath.LINETO] * (len(coords) - 2)
                        + [MplPath.CLOSEPOLY], dtype=np.uint8))
                    for ring in poly.interiors:
                        coords = np.asarray(ring.coords)
                        if len(coords) < 3:
                            continue
                        verts.append(coords)
                        codes.append(np.array(
                            [MplPath.MOVETO]
                            + [MplPath.LINETO] * (len(coords) - 2)
                            + [MplPath.CLOSEPOLY], dtype=np.uint8))
                if not verts:
                    return None
                return MplPath(np.concatenate(verts, axis=0),
                               np.concatenate(codes))

            # ── Gentle geometry simplification to shrink the vector PDF ───────
            # Douglas-Peucker at a fraction of a print pixel (fitted map size at
            # 300 DPI), so the deviation stays sub-pixel and invisible on paper.
            # Per-precinct vertex count is the dominant size driver and the
            # precinct grid reuses these same paths, so simplifying here shrinks
            # both the fills and the grid. The district/county dissolve below
            # keeps the original geometry (see note there). Raise _SIMPLIFY_PX to
            # shrink more aggressively (watch for hairline gaps between precincts).
            _SIMPLIFY_PX = 0.5
            _data_per_px = (gw / map_w) / 300.0 if map_w > 0 else 0.0
            _simp_tol = _data_per_px * _SIMPLIFY_PX
            geom_simplified = (gdf.geometry.simplify(_simp_tol)
                               if _simp_tol > 0 else gdf.geometry)

            # ── Precinct fills (batched PatchCollection) ──────────────────────
            patches, colors = [], []
            for i, geom in enumerate(geom_simplified):
                path = _to_mpl_path(geom)
                if path is None:
                    continue
                patches.append(PathPatch(path))
                colors.append(fill_mpl[i])
            if patches:
                # Fills: no edges (keeps PDF edge transparency independent)
                ax.add_collection(
                    PatchCollection(patches, facecolors=colors, edgecolors='none'))
                # Precinct boundaries: prominent when the Precincts overlay is on,
                # near-invisible when off. Stroke-only; set_alpha() writes a PDF /CA.
                edge_coll = PatchCollection(patches, facecolors='none',
                                            edgecolors='white', linewidths=0.2)
                edge_coll.set_alpha(PRECINCT_EDGE_ALPHA if mv.precinct_overlay
                                    else _PDF_PRECINCT_OFF_ALPHA)
                ax.add_collection(edge_coll)

            # ── Pre-dissolve district & county geometries (fast, avoids
            #    looping n_dist × unary_union over all precincts) ───────────────
            # Dissolve from ORIGINAL geometry: independently-simplified precincts
            # no longer share exact edges, so their union leaves interior slivers
            # that the district/county borders would trace as jagged lines.
            gdf_work = gdf[["geometry"]].copy()
            gdf_work["_dist"] = assignment
            have_county = (mv._county_array is not None
                           and (mv.county_overlay or mv.splits_view))
            if have_county:
                gdf_work["_cty"] = mv._county_array
                cty_geoms = gdf_work.dissolve(by="_cty").geometry
            dist_geoms = gdf_work.dissolve(by="_dist").geometry

            # ── Splits view: dim clean (un-split) counties ────────────────────
            if mv.splits_view and mv._county_array is not None:
                ca  = mv._county_array
                nct = int(ca.max()) + 1
                flt = (ca * n_dist + assignment).astype(np.int64)
                co_di = (np.bincount(flt, minlength=nct * n_dist)
                           .reshape(nct, n_dist))
                county_clean = (co_di > 0).sum(axis=1) <= 1
                dim_c = (_EXPORT_SPLITS_DIM[0] / 255, _EXPORT_SPLITS_DIM[1] / 255,
                         _EXPORT_SPLITS_DIM[2] / 255)
                dim_patches = []
                for ci_idx in range(nct):
                    if county_clean[ci_idx] and ci_idx in cty_geoms.index:
                        p = _to_mpl_path(cty_geoms.loc[ci_idx])
                        if p:
                            dim_patches.append(PathPatch(p))
                if dim_patches:
                    ax.add_collection(
                        PatchCollection(dim_patches,
                                        facecolors=[dim_c] * len(dim_patches),
                                        edgecolors="none", linewidths=0))

            # ── County borders ────────────────────────────────────────────────
            if have_county:
                cty_color = (180 / 255, 180 / 255, 180 / 255)
                for geom in cty_geoms:
                    p = _to_mpl_path(geom)
                    if p:
                        ax.add_patch(PathPatch(p, facecolor="none",
                                               edgecolor=cty_color,
                                               linewidth=0.5))

            # ── District borders (dissolved geoms, no per-precinct loop) ──────
            for geom in dist_geoms:
                p = _to_mpl_path(geom)
                if p:
                    ax.add_patch(PathPatch(p, facecolor="none",
                                           edgecolor="black", linewidth=0.7))

            # ── State outline (union of already-dissolved district geoms) ──────
            state_geom = _shapely_union(list(dist_geoms.values))
            p = _to_mpl_path(state_geom)
            if p:
                ax.add_patch(PathPatch(p, facecolor="none",
                                       edgecolor="black", linewidth=1.2))

            # ── District labels ───────────────────────────────────────────────
            if mv.show_labels:
                from shapely.ops import polylabel as _polylabel

                from mosaic.gui.map_view import stable_color_mapping
                ci_arr = (stable_color_mapping(assignment, initial, n_dist)
                          if initial is not None and len(initial) == len(assignment)
                          else assignment)
                lm     = mv.district_label_map
                use_lm = lm is not None and len(lm) == n_dist
                font_path = _ASSETS_DIR / "fonts" / "inter" / "Inter-SemiBold.ttf"
                fp = (FontProperties(fname=str(font_path))
                      if font_path.exists() else None)
                # Cascade per label: BASE_PT if it fits, else shrink to the
                # district's inscribed circle, else drop it (below MIN_PT it is
                # too small to place without overlapping a neighbour). Never
                # grows past BASE_PT, so the common case is unchanged.
                BASE_PT, MIN_PT = 7.0, 3.5
                pts_per_data = 72.0 * map_w / gw if gw > 0 else 0.0
                # Pole of inaccessibility (deepest interior point) rather than
                # representative_point(), which can sit on a concave edge.
                _lbl_tol = max(gw, gh) / 1000.0
                for d in range(n_dist):
                    if d not in dist_geoms.index:
                        continue
                    mask = assignment == d
                    if not mask.any():
                        continue
                    si       = int(ci_arr[mask][0])
                    num      = int(lm[si]) if use_lm else si + 1
                    _dgeom = dist_geoms.loc[d]
                    if _dgeom.geom_type == "MultiPolygon":
                        _dgeom = max(_dgeom.geoms, key=lambda g: g.area)
                    try:
                        label_pt = _polylabel(_dgeom, tolerance=_lbl_tol)
                    except Exception:
                        label_pt = dist_geoms.loc[d].representative_point()
                    # Radius of the largest empty circle at the label point, in
                    # points on the page.
                    r_pt = label_pt.distance(_dgeom.boundary) * pts_per_data
                    # Largest size whose digits fit inside that circle (diagonal
                    # fit, ~0.9 fill), capped at BASE_PT; drop if it can't reach
                    # MIN_PT.
                    if r_pt > 0.0:
                        fit = 0.9 * r_pt / np.hypot(0.275 * len(str(num)), 0.36)
                        if fit < MIN_PT:
                            continue
                        size = min(BASE_PT, fit)
                    else:
                        size = BASE_PT
                    kw = dict(ha="center", va="center", fontsize=size,
                              color="white",
                              path_effects=[_pe.withStroke(
                                  linewidth=1.75 * size / BASE_PT,
                                  foreground="black")])
                    if fp:
                        ax.text(label_pt.x, label_pt.y, str(num),
                                fontproperties=fp, **kw)
                    else:
                        ax.text(label_pt.x, label_pt.y, str(num),
                                fontweight="bold", **kw)

            # ── Title (50% width, centered in the top band) ───────────────────
            if title:
                body_r, body_g, body_b, _ = _EXPORT_FG
                fig.text(title_x, title_y, title,
                         ha="center", va="center",
                         fontsize=14, fontweight="bold",
                         color=(body_r / 255, body_g / 255, body_b / 255))

            # ── Caption (80% width, centered in the bottom band) ──────────────
            cap_r, cap_g, cap_b, _ = _EXPORT_FG
            fig.text(cap_x, cap_y, "Made with Mosaic",
                     ha="center", va="center", fontsize=7,
                     color=(cap_r / 255, cap_g / 255, cap_b / 255))

            # ── Write file ────────────────────────────────────────────────────
            if output_path is None:
                timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = output_dir() / f"map_{timestamp}.pdf"
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(str(output_path), format="pdf")
            plt.close(fig)
            self.state.update(status_message=f"Map saved to {output_path}")
            self._open_in_os(output_path)

        except Exception as exc:
            self.state.update(status_message=f"PDF export failed: {exc}")
        finally:
            self._adv_finish()

    def _png_worker(self, title: str, dpi: int, output_path) -> None:
        """Rasterise the map at the given DPI and save as PNG."""
        from datetime import datetime

        from PIL import Image, ImageDraw, ImageFont
        scale = dpi / 96.0
        try:
            rgba = self._render_map_at_scale(scale, state_outline=True)
            if rgba is None:
                self.state.update(status_message="Save failed: map not ready.")
                return

            img = Image.fromarray(rgba, mode="RGBA")
            font_path = _ASSETS_DIR / "fonts" / "inter" / "Inter-SemiBold.ttf"

            br, bg, bb, _ = _EXPORT_BG

            cap_size       = max(10, int(11 * scale))
            cap_margin     = max(8,  int(12 * scale))
            bottom_strip_h = cap_size + 2 * cap_margin
            try:
                cap_font = ImageFont.truetype(str(font_path), cap_size)
            except OSError:
                cap_font = ImageFont.load_default()

            top_strip_h = 0
            title_font  = None
            if title:
                r, g, b, _ = _EXPORT_FG
                top_strip_h = max(28, int(36 * scale))
                font_size   = max(14, int(18 * scale))
                try:
                    title_font = ImageFont.truetype(str(font_path), font_size)
                except OSError:
                    title_font = ImageFont.load_default()

            new_img = Image.new(
                "RGBA",
                (img.width, img.height + top_strip_h + bottom_strip_h),
                (br, bg, bb, 255),
            )
            new_img.paste(img, (0, top_strip_h), img)
            draw = ImageDraw.Draw(new_img)
            if title:
                draw.text(
                    (img.width // 2, top_strip_h // 2), title,
                    fill=(r, g, b, 255), font=title_font, anchor="mm",
                )
            cap_r, cap_g, cap_b, _ = _EXPORT_FG
            draw.text(
                (new_img.width - cap_margin, new_img.height - cap_margin),
                "Made with Mosaic",
                fill=(cap_r, cap_g, cap_b, 255),
                font=cap_font, anchor="rs",
            )

            if output_path is None:
                timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = output_dir() / f"map_{timestamp}.png"
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            new_img.save(output_path, dpi=(dpi, dpi))
            self.state.update(status_message=f"Map saved to {output_path}")
            self._open_in_os(output_path)
        finally:
            self._adv_finish()
