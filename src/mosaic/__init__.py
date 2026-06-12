"""
Mosaic - Redistricting toolkit for Python

A port of the original R-based Mosaic redistricting toolkit.
Uses recombination (ReCom) to generate and optimize district maps.
"""

import sys

__version__ = "0.1.0"


def _die_friendly(msg: str) -> None:
    """Print a clear, multi-line error message and exit non-zero.

    Used for first-contact failures (missing deps, missing system libs)
    so users see something actionable instead of a raw Python traceback.
    """
    bar = "=" * 60
    print(f"\n{bar}\nMosaic cannot start\n{bar}\n{msg}\n", file=sys.stderr)
    sys.exit(1)


def _check_vcredist() -> None:
    """Windows-only: catch missing VC++ runtime before DPG fails to load.

    ``run_mosaic.bat`` already does this for the launcher path; this check
    covers users who invoke the ``mosaic`` console script directly.
    """
    if sys.platform != "win32":
        return
    import ctypes
    missing = []
    for dll in ("vcruntime140.dll", "msvcp140.dll"):
        try:
            ctypes.WinDLL(dll)
        except OSError:
            missing.append(dll)
    if missing:
        _die_friendly(
            f"The Microsoft Visual C++ Runtime is not installed on this PC.\n"
            f"Dear PyGui needs it to draw the window.\n"
            f"\n"
            f"Missing: {', '.join(missing)}\n"
            f"\n"
            f"Fix:\n"
            f"  1) Download and run: https://aka.ms/vs/17/release/vc_redist.x64.exe\n"
            f"  2) Click Yes if Windows asks for permission\n"
            f"  3) Re-launch Mosaic"
        )


def _try_import(modname: str, purpose: str, install_hint: str = "") -> None:
    import importlib
    try:
        importlib.import_module(modname)
    except ImportError as exc:
        msg = (
            f"Mosaic could not load '{modname}' ({purpose}).\n"
            f"\n"
            f"  {type(exc).__name__}: {exc}\n"
        )
        if install_hint:
            msg += f"\n{install_hint}\n"
        msg += (
            "\n"
            "If you launched via run_mosaic.bat (Windows) or run_mosaic.command\n"
            "(macOS), close this window and double-click the launcher again —\n"
            "it re-installs missing dependencies automatically."
        )
        _die_friendly(msg)


def _preflight() -> None:
    """Validate the runtime environment before touching the GUI module.

    On failure prints a friendly diagnostic to stderr and exits — never
    surfaces a raw ImportError traceback to the user.
    """
    _check_vcredist()

    mac_gdal_hint = (
        "On macOS you may need to install GDAL via Homebrew:\n"
        "  brew install gdal"
        if sys.platform == "darwin" else ""
    )
    _try_import(
        "dearpygui.dearpygui",
        "the graphics library that draws Mosaic's window",
    )
    _try_import(
        "geopandas",
        "the GIS library that reads shapefiles",
        install_hint=mac_gdal_hint,
    )
    _try_import("shapely", "the geometry library used for scoring")
    _try_import("numpy", "numerical core")


def _setup_logging() -> None:
    """Wire up a console handler for the 'mosaic' logger.

    Without this the runner-added NullHandler swallows every log line, so
    diagnostic checkpoints (e.g. the hot-start flow) leave no trace when
    something goes wrong. ASCII-only formatter to dodge cp1252 crashes
    when the console code page is Windows-1252.

    Note: no persistent file handler, and level is WARNING so routine
    per-iteration progress (INFO) does not stream to the console. Only
    warnings/errors surface. Durable on-disk diagnostics for failures are
    handled separately by crash.write_crash_log (crashes/*.log).
    """
    import logging

    logger = logging.getLogger("mosaic")
    if any(
        not isinstance(h, logging.NullHandler) for h in logger.handlers
    ):
        return  # already configured (e.g. headless CLI ran basicConfig)

    logger.setLevel(logging.WARNING)
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    console = logging.StreamHandler(stream=sys.stderr)
    console.setFormatter(fmt)
    logger.addHandler(console)


def main():
    """Launch the Mosaic GUI application."""
    _preflight()
    _setup_logging()

    try:
        from mosaic.gui import MosaicApp
    except ImportError as exc:
        _die_friendly(
            f"Failed to import Mosaic GUI module:\n"
            f"  {type(exc).__name__}: {exc}\n"
            f"\n"
            f"This usually means a dependency is installed but broken.\n"
            f"Try deleting the .venv folder next to run_mosaic and re-launching."
        )

    try:
        app = MosaicApp()
        app.setup()
        app.run()
    except Exception as exc:
        from mosaic.crash import write_crash_log
        path = write_crash_log(exc, context={"phase": "main"})
        print(
            f"\nMosaic crashed.\n"
            f"Crash log: {path}\n",
            file=sys.stderr,
        )
        raise


__all__ = ["main"]
