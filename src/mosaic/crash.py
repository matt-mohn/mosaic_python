"""On-disk crash logging.

Top-level exception handlers call ``write_crash_log`` to drop a structured
log file in ``crashes/YYYYMMDD-HHMMSS.log``. The file is what a user attaches
to a bug report and what future maintainers debrief on return — distinct
from Python's ``logging`` module, which is for ongoing operational info that
doesn't survive the session.

Best-effort by design: any failure inside this module is swallowed and a
sentinel path is returned, because a crash logger that itself raises masks
the original exception and frustrates everyone.
"""

from __future__ import annotations

import platform
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from mosaic import __version__


CRASH_DIR = Path("crashes")


def write_crash_log(
    exc: BaseException,
    context: Mapping[str, Any] | None = None,
    crash_dir: Path | str = CRASH_DIR,
) -> Path:
    """Write a crash log file and return its path.

    Filename is ``YYYYMMDD-HHMMSS.log`` with a numeric suffix if a same-second
    crash already exists. Contents include Mosaic version, Python version,
    platform, optional caller-supplied context dict, and the full traceback.
    """
    crash_dir = Path(crash_dir)
    try:
        crash_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = crash_dir / f"{stamp}.log"
        counter = 1
        while path.exists():
            path = crash_dir / f"{stamp}-{counter}.log"
            counter += 1

        lines: list[str] = [
            f"Mosaic crash log - {datetime.now().isoformat()}",
            f"Mosaic version: {__version__}",
            f"Python: {sys.version.split()[0]}",
            f"Platform: {platform.platform()}",
            "",
        ]
        if context:
            lines.append("Context:")
            for key, value in context.items():
                lines.append(f"  {key}: {value}")
            lines.append("")
        lines.append("Traceback:")
        lines.append(
            "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        )

        path.write_text("\n".join(lines), encoding="utf-8")
        return path
    except Exception:
        return crash_dir / "(crash log write failed)"
