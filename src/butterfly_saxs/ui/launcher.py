"""Crash-visible GUI entry point for desktop launchers.

``pythonw.exe`` does not provide a console, so an import or Qt start-up failure
can otherwise look like a double-click that did nothing.  This module records a
traceback and presents the log location while leaving the scientific pipeline
unchanged.
"""

from __future__ import annotations

import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence


def default_log_path() -> Path:
    """Return a writable per-user launcher log path."""

    if os.name == "nt":
        root = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        root = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state"))
    return root / "LamellarSAXS2D" / "launcher.log"


def _append_failure(
    path: Path,
    *,
    argv: Sequence[str],
    traceback_text: str,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(f"\n[{timestamp}] GUI launch failed\n")
        handle.write(f"argv={list(argv)!r}\n")
        handle.write(traceback_text.rstrip())
        handle.write("\n")
    return path


def _present_failure(title: str, message: str) -> None:
    """Show a native Windows error dialog, with stderr as a safe fallback."""

    if os.name == "nt":
        try:
            import ctypes

            ctypes.windll.user32.MessageBoxW(0, message, title, 0x10)
            return
        except Exception:  # pragma: no cover - platform fallback
            pass
    try:
        sys.stderr.write(f"{title}\n{message}\n")
        sys.stderr.flush()
    except Exception:  # pragma: no cover - pythonw may not expose stderr
        pass


def run(
    argv: Sequence[str] | None = None,
    *,
    launch_fn: Callable[[list[str]], Any] | None = None,
    log_path: str | os.PathLike[str] | None = None,
    presenter: Callable[[str, str], None] | None = None,
) -> int:
    """Run the public workbench and make start-up exceptions visible."""

    arguments = list(sys.argv[1:] if argv is None else argv)
    try:
        if launch_fn is None:
            from . import launch as launch_fn
        result = launch_fn(arguments)
        return 0 if result is None else int(result)
    except Exception as exc:  # noqa: BLE001 - this is the top-level crash boundary
        traceback_text = traceback.format_exc()
        destination = Path(log_path) if log_path is not None else default_log_path()
        try:
            written = _append_failure(
                destination,
                argv=arguments,
                traceback_text=traceback_text,
            )
            location = str(written)
        except Exception as log_error:  # pragma: no cover - unwritable user profile
            location = f"unavailable ({type(log_error).__name__}: {log_error})"
        show = presenter or _present_failure
        show(
            "LamellarSAXS2D start-up failed",
            (
                "The graphical workbench could not start.\n"
                "图形界面启动失败。\n\n"
                f"Error: {type(exc).__name__}: {exc}\n"
                f"Diagnostic log / 诊断日志: {location}\n\n"
                "Run `bsaxs-doctor --require-ui` in a terminal for an "
                "environment check."
            ),
        )
        return 1


def main(argv: Sequence[str] | None = None) -> int:
    return run(argv)


if __name__ == "__main__":  # pragma: no cover - desktop entry point
    raise SystemExit(main())
