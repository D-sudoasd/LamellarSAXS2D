"""LamellarSAXS2D quantitative 2D SAXS refinement toolkit."""

from .geometry import GeometryMaps
from .io import LoadedImage
from .models import AnalysisConfig, AnalysisResult, ParameterSet, ParameterSpec, RidgePoint
from .preflight import PreflightError, run_preflight
from .p3_gate import P3_GATE_SCHEMA_VERSION, evaluate_p3_gate, write_p3_gate_report
from .validation import (
    AnalysisDomain,
    AnalysisDomainError,
    ResultSchemaError,
    build_analysis_domain,
    validate_result_schema,
)

# Canonical public data objects.  ``io.LoadedImage`` carries the positive
# valid_mask and ``geometry.GeometryMaps`` carries physical q units plus the
# detector-style mask alias; the older models.ImageFrame/QMap remain available
# only to legacy/internal callers and are not exported under ambiguous names.
ImageFrame = LoadedImage
QMap = GeometryMaps

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
]

__version__ = "0.1.0"
