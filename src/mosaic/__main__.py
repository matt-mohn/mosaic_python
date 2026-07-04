"""Entry point for ``python -m mosaic`` -- launches the GUI.

Preferred over ``python -m mosaic.gui.app``: running a submodule with ``-m``
when its parent package (``mosaic.gui``) already imports it triggers a
``RuntimeWarning: found in sys.modules ...`` from runpy. ``mosaic.__init__``
does not import ``__main__``, so this stays clean.
"""

from mosaic import main

if __name__ == "__main__":
    main()
