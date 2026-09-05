"""Optional LamellarSAXS2D Qt workbench.

No Qt module is imported until a UI symbol is requested.  This keeps
``import butterfly_saxs`` and core-only scripts independent of PySide6.
"""

from __future__ import annotations

from .qt_compat import QT_AVAILABLE, QT_IMPORT_ERROR

__all__ = [
    "QT_AVAILABLE",
    "QT_IMPORT_ERROR",
    "AnalysisWorker",
    "AnalysisService",
    "ButterflyAnalysisService",
    "GenerationGuard",
    "MainWindow",
    "OverlayView",
    "ParameterRow",
    "ParameterTableModel",
    "ParameterModel",
    "PatternView",
    "RefinementMainWindow",
    "RefinementWindow",
    "ViewGrid",
    "Workbench",
    "WorkbenchWindow",
    "coerce_parameter_rows",
    "create_app",
    "launch",
    "symmetric_ellipses",
    "upgrade_window",
]


def __getattr__(name: str):
    if name in {"AnalysisService", "ButterflyAnalysisService"}:
        from ..service import AnalysisService, ButterflyAnalysisService

        return {
            "AnalysisService": AnalysisService,
            "ButterflyAnalysisService": ButterflyAnalysisService,
        }[name]
    if name in {
        "ParameterRow",
        "ParameterTableModel",
        "ParameterModel",
        "coerce_parameter_rows",
    }:
        from .models import (
            ParameterRow,
            ParameterTableModel,
            coerce_parameter_rows,
        )

        return {
            "ParameterRow": ParameterRow,
            "ParameterTableModel": ParameterTableModel,
            "ParameterModel": ParameterTableModel,
            "coerce_parameter_rows": coerce_parameter_rows,
        }[name]
    if name in {"AnalysisWorker", "GenerationGuard"}:
        from .workers import AnalysisWorker, GenerationGuard

        return {
            "AnalysisWorker": AnalysisWorker,
            "GenerationGuard": GenerationGuard,
        }[name]
    if name in {"PatternView", "OverlayView", "ViewGrid"}:
        from .views import OverlayView, PatternView, ViewGrid

        return {
            "PatternView": PatternView,
            "OverlayView": OverlayView,
            "ViewGrid": ViewGrid,
        }[name]
    if name in {
        "MainWindow",
        "RefinementMainWindow",
        "RefinementWindow",
        "Workbench",
        "WorkbenchWindow",
        "create_app",
        "launch",
        "symmetric_ellipses",
        "upgrade_window",
    }:
        from .workbench import (
            MainWindow,
            RefinementMainWindow,
            RefinementWindow,
            Workbench,
            WorkbenchWindow,
            create_app,
            launch,
            symmetric_ellipses,
            upgrade_window,
        )

        return {
            "MainWindow": MainWindow,
            "RefinementMainWindow": RefinementMainWindow,
            "RefinementWindow": RefinementWindow,
            "Workbench": Workbench,
            "WorkbenchWindow": WorkbenchWindow,
            "create_app": create_app,
            "launch": launch,
            "symmetric_ellipses": symmetric_ellipses,
            "upgrade_window": upgrade_window,
        }[name]
    raise AttributeError(name)
