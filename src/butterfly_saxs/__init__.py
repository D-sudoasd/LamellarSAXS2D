"""LamellarSAXS2D quantitative 2D SAXS refinement toolkit.

The package root is intentionally dependency-light.  Scientific modules are
imported when their public object is first requested, so ``bsaxs-doctor`` can
diagnose a missing NumPy/SciPy installation instead of failing during package
initialization.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__version__ = "0.1.0"

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "AnalysisConfig": (".models", "AnalysisConfig"),
    "AnalysisResult": (".models", "AnalysisResult"),
    "ParameterSet": (".models", "ParameterSet"),
    "ParameterSpec": (".models", "ParameterSpec"),
    "RidgePoint": (".models", "RidgePoint"),
    "LoadedImage": (".io", "LoadedImage"),
    "GeometryMaps": (".geometry", "GeometryMaps"),
    "AnalysisDomain": (".validation", "AnalysisDomain"),
    "AnalysisDomainError": (".validation", "AnalysisDomainError"),
    "ResultSchemaError": (".validation", "ResultSchemaError"),
    "build_analysis_domain": (".validation", "build_analysis_domain"),
    "validate_result_schema": (".validation", "validate_result_schema"),
    "PreflightError": (".preflight", "PreflightError"),
    "run_preflight": (".preflight", "run_preflight"),
    "P3_GATE_SCHEMA_VERSION": (".p3_gate", "P3_GATE_SCHEMA_VERSION"),
    "evaluate_p3_gate": (".p3_gate", "evaluate_p3_gate"),
    "write_p3_gate_report": (".p3_gate", "write_p3_gate_report"),
    "AnalysisCancelled": (".cancellation", "AnalysisCancelled"),
}

_ALIASES: dict[str, str] = {"ImageFrame": "LoadedImage", "QMap": "GeometryMaps"}

__all__ = [
    "AnalysisConfig",
    "AnalysisResult",
    "ImageFrame",
    "ParameterSet",
    "ParameterSpec",
    "QMap",
    "RidgePoint",
    "LoadedImage",
    "GeometryMaps",
    "AnalysisDomain",
    "AnalysisDomainError",
    "ResultSchemaError",
    "build_analysis_domain",
    "validate_result_schema",
    "PreflightError",
    "run_preflight",
    "P3_GATE_SCHEMA_VERSION",
    "evaluate_p3_gate",
    "write_p3_gate_report",
    "AnalysisCancelled",
]


def __getattr__(name: str) -> Any:
    target = _ALIASES.get(name, name)
    export = _LAZY_EXPORTS.get(target)
    if export is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = export
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[target] = value
    if name != target:
        globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
