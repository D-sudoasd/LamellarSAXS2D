"""LamellarSAXS2D quantitative 2D SAXS refinement toolkit."""

from .geometry import GeometryMaps
from .io import LoadedImage
from .models import AnalysisConfig, AnalysisResult, ParameterSet, ParameterSpec, RidgePoint

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
]

__version__ = "0.1.0"
