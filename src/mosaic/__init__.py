"""
Mosaic - Redistricting toolkit for Python

A port of the original R-based Mosaic redistricting toolkit.
Uses recombination (ReCom) to generate and optimize district maps.
"""

import sys

__version__ = "0.1.0"


def main():
    """Launch the Mosaic GUI application."""
    try:
        from mosaic.gui import MosaicApp
        app = MosaicApp()
        app.setup()
        app.run()
    except Exception as exc:
        from mosaic.crash import write_crash_log
        path = write_crash_log(exc, context={"phase": "main"})
        print(f"Mosaic crashed. Crash log: {path}", file=sys.stderr)
        raise


__all__ = ["main"]
