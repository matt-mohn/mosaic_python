"""User-data directory resolution for crash logs, caches, and shapefiles.

Two install shapes:

* Editable install (the launcher workflow): the package lives inside a
  checked-out repo, so ``parents[2]`` from this file is the repo root —
  which is where the shipped ``cache/`` lives and where crash logs should
  go too (one folder beside the launcher scripts).
* Wheel install (site-packages): ``parents[2]`` lands somewhere read-only
  inside the Python install. Fall back to ``~/.mosaic/`` there.

Override with ``MOSAIC_DATA_DIR`` for tests or unusual layouts.
"""

from __future__ import annotations

import os
from pathlib import Path


def mosaic_data_dir() -> Path:
    env = os.environ.get("MOSAIC_DATA_DIR")
    if env:
        return Path(env).expanduser()
    repo_root = Path(__file__).resolve().parents[2]
    if (repo_root / "pyproject.toml").is_file():
        return repo_root
    return Path.home() / ".mosaic"


def crash_dir() -> Path:
    return mosaic_data_dir() / "crashes"


def cache_dir() -> Path:
    return mosaic_data_dir() / "cache"


def shapefiles_dir() -> Path:
    return mosaic_data_dir() / "shapefiles"
