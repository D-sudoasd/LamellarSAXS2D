"""Small domain dataclasses shared by loaders, analysis and the UI.

These classes intentionally contain data and light validation only.  Image
decoding and detector calibration belong to their respective modules; keeping
the common objects here makes single-frame and in-situ batch workflows use the
same contracts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .parameters import ParameterSet, ParameterSpec


def _array(value: Any, name: str, *, ndim: int | None = None) -> np.ndarray:
    result = np.asarray(value)
    if ndim is not None and result.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got {result.ndim}")
    if result.size == 0:
        raise ValueError(f"{name} cannot be empty")
    return result


@dataclass
class ImageFrame:
    """Legacy in-memory frame adapter.

    ``mask=True`` means invalid, matching FabIO/pyFAI detector-mask polarity.
    File-backed public loading uses :class:`butterfly_saxs.io.LoadedImage`.
    """

    data: np.ndarray
    path: str | Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    mask: np.ndarray | None = None
    frame_id: str | int | None = None
    timestamp: float | None = None

    def __post_init__(self) -> None:
        self.data = _array(self.data, "data", ndim=2)
        self.metadata = dict(self.metadata)
        if self.mask is not None:
            self.mask = np.asarray(self.mask, dtype=bool)
            if self.mask.shape != self.data.shape:
                raise ValueError("mask must have the same shape as data")
        if self.path is not None:
            self.path = str(self.path)

    @property
    def shape(self) -> tuple[int, int]:
        return self.data.shape


@dataclass
class QMap:
    """Legacy in-memory calibrated q-map adapter.

    ``mask=True`` means invalid.  PONI-backed public geometry uses
    :class:`butterfly_saxs.geometry.GeometryMaps`, whose canonical positive
    mask is ``valid_mask`` and whose ``mask`` property is its inverse.  A
    missing ``q_unit`` stays ``unknown``; physical spacing consumers must not
    infer units from the numeric q values.
    """

    qx: np.ndarray
    qy: np.ndarray
    q: np.ndarray | None = None
    chi: np.ndarray | None = None
    mask: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    # A q map without a declared unit is intentionally not treated as a
    # physical reciprocal-space map by the observable adapters.  Calibrated
    # callers should pass ``q_unit`` (or put it in metadata) explicitly.
    q_unit: str = "unknown"

    def __post_init__(self) -> None:
        self.qx = _array(self.qx, "qx", ndim=2).astype(float, copy=False)
        self.qy = _array(self.qy, "qy", ndim=2).astype(float, copy=False)
        if self.qx.shape != self.qy.shape:
            raise ValueError("qx and qy must have the same shape")
        if self.q is None:
            self.q = np.hypot(self.qx, self.qy)
        else:
            self.q = _array(self.q, "q", ndim=2).astype(float, copy=False)
            if self.q.shape != self.qx.shape:
                raise ValueError("q must have the same shape as qx/qy")
        if self.chi is not None:
            self.chi = _array(self.chi, "chi", ndim=2).astype(float, copy=False)
            if self.chi.shape != self.qx.shape:
                raise ValueError("chi must have the same shape as qx/qy")
        if self.mask is not None:
            self.mask = np.asarray(self.mask, dtype=bool)
            if self.mask.shape != self.qx.shape:
                raise ValueError("mask must have the same shape as qx/qy")
        self.metadata = dict(self.metadata)
        metadata_unit = self.metadata.get("q_unit", self.metadata.get("unit"))
        if str(self.q_unit or "unknown").strip().lower() == "unknown" and metadata_unit is not None:
            self.q_unit = str(metadata_unit)
        self.q_unit = str(self.q_unit or "unknown")
        self.metadata.setdefault("q_unit", self.q_unit)

    @property
    def shape(self) -> tuple[int, int]:
        return self.qx.shape


@dataclass
class RidgePoint:
    """One intensity-ridge point in q space.

    ``component`` is optional; when supplied it can label one member of a
    symmetric pair (0 for ``+theta`` and 1 for ``-theta``).  Unlabelled points
    are assigned to whichever ellipse gives the smaller residual during fit.
    """

    qx: float
    qy: float
    intensity: float | None = None
    weight: float = 1.0
    component: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    q_unit: str = "unknown"

    def __post_init__(self) -> None:
        self.qx = float(self.qx)
        self.qy = float(self.qy)
        if not np.isfinite(self.qx) or not np.isfinite(self.qy):
            raise ValueError("RidgePoint coordinates must be finite")
        self.weight = float(self.weight)
        if not np.isfinite(self.weight) or self.weight <= 0:
            raise ValueError("RidgePoint weight must be finite and positive")
        if self.intensity is not None:
            self.intensity = float(self.intensity)
            if not np.isfinite(self.intensity):
                raise ValueError("RidgePoint intensity must be finite")
        if self.component is not None and self.component not in (0, 1):
            raise ValueError("RidgePoint component must be 0, 1 or None")
        self.metadata = dict(self.metadata)
        metadata_unit = self.metadata.get("q_unit", self.metadata.get("unit"))
        if str(self.q_unit or "unknown").strip().lower() == "unknown" and metadata_unit is not None:
            self.q_unit = str(metadata_unit)
        self.q_unit = str(self.q_unit or "unknown")
        self.metadata.setdefault("q_unit", self.q_unit)

    @property
    def x(self) -> float:
        return self.qx

    @property
    def y(self) -> float:
        return self.qy


@dataclass
class AnalysisConfig:
    """Configuration shared by one-frame and batch analyses."""

    parameters: ParameterSet = field(default_factory=ParameterSet)
    residual: str = "sampson"
    loss: str = "soft_l1"
    f_scale: float = 1.0
    max_nfev: int | None = None
    min_intensity: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.parameters, ParameterSet):
            self.parameters = ParameterSet(self.parameters)
        self.residual = str(self.residual).lower()
        if self.residual not in {"sampson", "geometric"}:
            raise ValueError("residual must be 'sampson' or 'geometric'")
        if self.f_scale <= 0 or not np.isfinite(self.f_scale):
            raise ValueError("f_scale must be finite and positive")
        if self.max_nfev is not None and (not isinstance(self.max_nfev, int) or self.max_nfev <= 0):
            raise ValueError("max_nfev must be a positive integer or None")
        if self.min_intensity is not None and not np.isfinite(self.min_intensity):
            raise ValueError("min_intensity must be finite or None")
        self.metadata = dict(self.metadata)


@dataclass
class AnalysisResult:
    """Container for a completed analysis and its provenance."""

    parameters: ParameterSet | None = None
    values: dict[str, float] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    frame_id: str | int | None = None
    timestamp: float | None = None
    source: str | None = None

    def __post_init__(self) -> None:
        if self.parameters is not None and not isinstance(self.parameters, ParameterSet):
            self.parameters = ParameterSet(self.parameters)
        if self.parameters is not None and not self.values:
            self.values = self.parameters.resolve()
        else:
            self.values = {str(key): float(value) for key, value in self.values.items()}
        self.diagnostics = dict(self.diagnostics)

    @property
    def parameter_set(self) -> ParameterSet | None:
        return self.parameters


__all__ = [
    "AnalysisConfig",
    "AnalysisResult",
    "ImageFrame",
    "ParameterSet",
    "ParameterSpec",
    "QMap",
    "RidgePoint",
]
