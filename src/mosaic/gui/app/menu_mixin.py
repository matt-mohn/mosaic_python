"""File/session menu: recent files, new/close, updates, output dir."""
from ._common import (
    _DOWNLOAD_URL,
    _RECENT_FILE,
    _SETTINGS_DIR,
    _UPDATE_CHECK_URL,
    AlgorithmRunner,
    Path,
    ShapefileConfig,
    __version__,
    dpg,
    np,
    output_dir,
    webbrowser,
)


class MenuMixin:
    """File/session menu: recent files, new/close, updates, output dir."""

    @staticmethod
    def _version_tuple(v):
        """Parse 'X.Y.Z' to a comparable tuple; () if unparseable."""
        try:
            return tuple(int(x) for x in v.split("."))
        except (ValueError, AttributeError):
            return ()

    def _on_open_output_dir(self):
        """Open output/ (saved maps + assignment/metric CSVs) in the OS file browser."""
        import os
        import subprocess
        import sys
        out = output_dir()
        out.mkdir(parents=True, exist_ok=True)  # create on first use so opening never fails
        try:
            if sys.platform == "darwin":
                subprocess.run(["open", str(out)], check=False)
            elif os.name == "nt":
                os.startfile(str(out))  # type: ignore[attr-defined]
            else:
                subprocess.run(["xdg-open", str(out)], check=False)
            self.state.update(status_message=f"Opened {out}")
        except Exception as e:
            self.state.update(status_message=f"Could not open output folder: {e}")

    def _open_in_os(self, path) -> None:
        """Open a saved file (or folder) in the OS default handler. Best-effort."""
        import os
        import subprocess
        import sys
        try:
            if sys.platform == "darwin":
                subprocess.run(["open", str(path)], check=False)
            elif os.name == "nt":
                os.startfile(str(path))  # type: ignore[attr-defined]
            else:
                subprocess.run(["xdg-open", str(path)], check=False)
        except Exception:
            pass

    def _on_check_updates(self):
        """Manual update check (Advanced menu). Compares the local version to the
        public repo's pyproject version. Synchronous — it's user-initiated — with
        a short timeout so it can't hang the UI for long."""
        import re as _re
        import urllib.request
        latest = None
        try:
            with urllib.request.urlopen(_UPDATE_CHECK_URL, timeout=5) as resp:
                text = resp.read().decode("utf-8")
            m = _re.search(r'^version\s*=\s*["\']([^"\']+)["\']', text, _re.M)
            latest = m.group(1) if m else None
        except Exception:
            latest = None
        self._show_update_result(latest)

    def _show_update_result(self, latest) -> None:
        cur = __version__
        update_available = False
        if latest is None:
            msg = ("Couldn't reach GitHub to check for updates. "
                   "Check your connection and try again.")
        elif self._version_tuple(latest) > self._version_tuple(cur):
            update_available = True
            msg = (f"A newer version is available: v{latest} (you have v{cur}).\n\n"
                   "Click Download to open the install page, then follow the "
                   "download-and-run steps for your computer.")
        else:
            msg = f"You're up to date (v{cur})."
        if update_available:
            buttons = [
                ("Download", lambda: webbrowser.open(_DOWNLOAD_URL), "primary"),
                ("Close", lambda: dpg.delete_item("popup_update")),
            ]
        else:
            buttons = [("Close", lambda: dpg.delete_item("popup_update"))]
        with self._dialog("Check for updates", "popup_update", (520, 220),
                          buttons=buttons):
            dpg.add_text(msg, wrap=500)

    def _load_recent_shapefiles(self) -> None:
        """Read recent-shapefile list from disk."""
        import json
        if not _RECENT_FILE.exists():
            return
        try:
            data = json.loads(_RECENT_FILE.read_text(encoding="utf-8"))
            self._recent_shapefiles = data if isinstance(data, list) else []
        except Exception:
            self._recent_shapefiles = []

    def _save_recent_shapefiles(self) -> None:
        """Persist recent-shapefile list to disk."""
        import json
        try:
            _SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
            _RECENT_FILE.write_text(
                json.dumps(self._recent_shapefiles[:5], indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass

    def _push_recent_shapefile(self, path: str,
                               config: "ShapefileConfig") -> None:
        """Prepend path+config to the recent list, cap at 5, refresh menu."""
        entry = {
            "path": path,
            "config": {
                "pop_col":    config.pop_col,
                "id_col":     config.id_col,
                "county_col": config.county_col,
                "elections":  config.elections,
            },
        }
        self._recent_shapefiles = [
            e for e in self._recent_shapefiles if e.get("path") != path
        ]
        self._recent_shapefiles.insert(0, entry)
        self._recent_shapefiles = self._recent_shapefiles[:5]
        self._save_recent_shapefiles()
        self._refresh_recent_menu()

    def _refresh_recent_menu(self) -> None:
        """Rebuild the Open Recent submenu from _recent_shapefiles."""
        if not dpg.does_item_exist("file_recent_menu"):
            return
        dpg.delete_item("file_recent_menu", children_only=True)
        if not self._recent_shapefiles:
            dpg.add_menu_item(
                label="(no recent files)", enabled=False,
                parent="file_recent_menu",
            )
            return
        for entry in self._recent_shapefiles:
            dpg.add_menu_item(
                label=Path(entry["path"]).name,
                callback=self._on_open_recent,
                user_data=entry,
                parent="file_recent_menu",
            )

    def _on_open_recent(self, sender, app_data, user_data) -> None:
        """Open a recently used shapefile, skipping the column picker."""
        if not user_data:
            return
        entry = user_data
        path = entry.get("path", "")
        if not path or not Path(path).exists():
            self.state.update(
                status_message=f"Recent file not found: {path}")
            return
        cfg_d = entry.get("config", {})
        # JSON round-trips tuples as lists; restore tuples so the runner's
        # for dem_col, gop_col in config.elections unpacking works correctly.
        raw_elections = cfg_d.get("elections", [])
        elections = [tuple(e) for e in raw_elections if len(e) == 2]
        config = ShapefileConfig(
            pop_col=cfg_d.get("pop_col", ""),
            id_col=cfg_d.get("id_col", ""),
            county_col=cfg_d.get("county_col"),
            elections=elections,
        )
        self._pending_recent_config = config
        self._restore_partisan_on_load = bool(elections)
        self._on_shapefile_selected(None, {"file_path_name": path})

    def _plan_unsaved(self) -> bool:
        """True if a run result exists and differs from the last saved assignment
        (Save Assignments writes current_assignment, so compare against that)."""
        with self.state._lock:
            best = self.state.best_assignment
            cur = (self.state.current_assignment.copy()
                   if self.state.current_assignment is not None else None)
        if best is None:
            return False
        if cur is None or self._saved_plan is None:
            return True
        return not np.array_equal(cur, self._saved_plan)

    def _on_close(self) -> None:
        """File > Close / Ctrl+W: warn only if the plan has unsaved changes."""
        if self._plan_unsaved():
            dpg.configure_item("popup_close_confirm", show=True)
        else:
            dpg.stop_dearpygui()

    def _on_new(self) -> None:
        """File > New: warn only if the plan has unsaved changes."""
        if self._plan_unsaved():
            dpg.configure_item("popup_new_confirm", show=True)
        else:
            self._do_new()

    def _do_new(self) -> None:
        """Discard current results and return to a clean slate."""
        dpg.configure_item("popup_new_confirm", show=False)
        self._on_reset()
        self._saved_plan = None
        self.runner = AlgorithmRunner(self.state)
        self._loaded_config = None
        self._has_elections = False
        self._map_loaded_path = ""
        self._map_loaded_gdf_id = 0
        self._map_data_gdf_id = 0
        self._map_loading = False
        self._map_ready = False
        self.state.update(shapefile_path="", status_message="")
        self._update_hot_start_display(None)
        dpg.set_value(self._shp_info, "Load a shapefile to begin.")
        self.theme.retoken(self._shp_info, "muted")
        if self.map_view is not None:
            self.map_view.wipe()
