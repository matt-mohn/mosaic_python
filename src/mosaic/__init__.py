"""
Mosaic - Redistricting toolkit for Python

A port of the original R-based Mosaic redistricting toolkit.
Uses recombination (ReCom) to generate and optimize district maps.
"""

__version__ = "0.1.0"


def main():
    """Launch the Mosaic GUI application."""
    from mosaic.gui import MosaicApp
    app = MosaicApp()
    app.setup()
    app.run()


__all__ = ["main"]
