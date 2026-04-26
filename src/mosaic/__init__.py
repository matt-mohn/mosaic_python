"""
Mosaic - Redistricting toolkit for Python

A port of the original R-based Mosaic redistricting toolkit.
Uses recombination (ReCom) to generate and optimize district maps.
"""

__version__ = "0.1.0"

from mosaic.gui import MosaicApp


def main():
    """Launch the Mosaic GUI application."""
    app = MosaicApp()
    app.setup()
    app.run()


__all__ = ["MosaicApp", "main"]
