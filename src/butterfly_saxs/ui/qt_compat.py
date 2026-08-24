"""Optional Qt bindings used by the LamellarSAXS2D workbench.

The scientific/core modules deliberately do not depend on Qt.  This module is
the single import boundary for the optional UI stack, so importing
``butterfly_saxs.ui`` (or any of its non-Qt helpers) remains safe on a headless
machine where PySide6 is not installed.
"""

from __future__ import annotations

from typing import Any

QT_AVAILABLE = False
QT_IMPORT_ERROR: Exception | None = None

try:  # pragma: no cover - the fallback is covered on minimal installations
    from PySide6 import QtCore, QtGui, QtWidgets

    QT_AVAILABLE = True
except Exception as exc:  # pragma: no cover - depends on the host install
    QtCore = None  # type: ignore[assignment]
    QtGui = None  # type: ignore[assignment]
    QtWidgets = None  # type: ignore[assignment]
    QT_IMPORT_ERROR = exc


def require_qt() -> None:
    """Raise an actionable error when a Qt-only entry point is requested."""

    if not QT_AVAILABLE:
        detail = f": {QT_IMPORT_ERROR}" if QT_IMPORT_ERROR else ""
        raise RuntimeError(
            "The LamellarSAXS2D workbench requires the optional UI dependencies "
            "(PySide6 and pyqtgraph)" + detail
        )


def qt_enum(name: str, fallback: Any = None) -> Any:
    """Return a Qt enum by dotted path without importing Qt at module import.

    A couple of small helpers use this for compatibility with PySide6 minor
    releases whose enum aliases moved from ``Qt`` to ``Qt.ItemDataRole``.
    """

    if not QT_AVAILABLE:
        return fallback
    current: Any = QtCore.Qt
    for part in name.split("."):
        current = getattr(current, part)
    return current


__all__ = ["QT_AVAILABLE", "QT_IMPORT_ERROR", "QtCore", "QtGui", "QtWidgets", "require_qt"]
