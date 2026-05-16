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
from mosaic.paths import crash_dir as _default_crash_dir


def write_crash_log(
    exc: BaseException,
    context: Mapping[str, Any] | None = None,
    crash_dir: Path | str | None = None,
) -> Path:
    """Write a crash log file and return its absolute path.

    Filename is ``YYYYMMDD-HHMMSS.log`` with a numeric suffix if a same-second
    crash already exists. Contents include Mosaic version, Python version,
    platform, optional caller-supplied context dict, and the full traceback.

    If the log write itself fails the traceback is dumped to stderr so the
    information is not silently lost.
    """
    cdir = Path(crash_dir) if crash_dir is not None else _default_crash_dir()
    try:
        cdir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = cdir / f"{stamp}.log"
        counter = 1
        while path.exists():
            path = cdir / f"{stamp}-{counter}.log"
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
        return path.resolve()
    except Exception as log_exc:
        print(
            f"\n[mosaic] Could not write crash log to {cdir}: {log_exc}\n"
            f"[mosaic] Original traceback follows:\n",
            file=sys.stderr,
        )
        traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.stderr)
        return cdir / "(crash log write failed)"
