"""Public settings and geometry records for the lamellar schematic."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass, fields
from numbers import Integral
from typing import Any

import numpy as np

from .lamellar_utils import _copy_metadata, _finite, _finite_positive


@dataclass(frozen=True)
class LamellarSettings:
    mode: str = "single"
    period_source: str = "radial"
    layer_count: int = 8
    stack_count: int = 12
    thickness_ratio: float = 0.25
    width_ratio: float = 4.0
    depth_ratio: float = 4.0
    spread_deg: float = 0.0
    lateral_shift_ratio: float = 0.0
    out_of_plane_deg: float = 0.0
    seed: int = 0
    manual_period: float = 1.0
    manual_angle_deg: float = 30.0
    manual_unit: str = "relative"
    selected_branch: int = -1
    reference_period: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", str(self.mode).strip().lower())
        object.__setattr__(
            self, "period_source", str(self.period_source).strip().lower()
        )
        object.__setattr__(self, "manual_unit", str(self.manual_unit).strip().lower())
        for name in ("layer_count", "stack_count", "seed", "selected_branch"):
            value = getattr(self, name)
            if isinstance(value, np.generic):
                value = value.item()
            if isinstance(value, Integral) and not isinstance(value, bool):
                object.__setattr__(self, name, int(value))
        self.validate()

    def validate(self) -> "LamellarSettings":
        if self.mode not in {"single", "multi"}:
            raise ValueError("mode must be 'single' or 'multi'")
        if self.period_source not in {"radial", "ellipse", "manual"}:
            raise ValueError("period_source must be 'radial', 'ellipse', or 'manual'")
        if self.manual_unit not in {"nm", "relative"}:
            raise ValueError("manual_unit must be 'nm' or 'relative'")
        for name, maximum in (("layer_count", 512), ("stack_count", 256)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise ValueError(f"{name} must be a nonnegative integer")
            if value < 0 or value > maximum:
                raise ValueError(f"{name} must be between 0 and {maximum}")
        if isinstance(self.seed, bool) or not isinstance(self.seed, Integral):
            raise ValueError("seed must be an integer")
        if self.selected_branch not in {-1, 0, 1}:
            raise ValueError("selected_branch must be -1, 0, or 1")
        for name in ("thickness_ratio", "width_ratio", "depth_ratio", "manual_period"):
            value = _finite(getattr(self, name))
            if value is None or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0.0 < float(self.thickness_ratio) < 1.0:
            raise ValueError("thickness_ratio must be greater than 0 and less than 1")
        spread = _finite(self.spread_deg)
        lateral = _finite(self.lateral_shift_ratio)
        tilt = _finite(self.out_of_plane_deg)
        angle = _finite(self.manual_angle_deg)
        if spread is None or spread < 0.0 or spread > 180.0:
            raise ValueError("spread_deg must be finite in [0, 180]")
        if lateral is None:
            raise ValueError("lateral_shift_ratio must be finite")
        if tilt is None or abs(tilt) > 90.0:
            raise ValueError("out_of_plane_deg must be finite in [-90, 90]")
        if angle is None:
            raise ValueError("manual_angle_deg must be finite")
        if self.reference_period is not None:
            reference = _finite_positive(self.reference_period)
            if reference is None:
                raise ValueError(
                    "reference_period must be finite and positive when provided"
                )
        return self

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any] | "LamellarSettings" | None = None
    ) -> "LamellarSettings":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return cls(**value.to_dict())
        if not isinstance(value, Mapping):
            raise TypeError("LamellarSettings.from_mapping expects a mapping")
        nested = value
        for key in ("lamellar", "lamellar_settings", "settings"):
            candidate = value.get(key)
            if isinstance(candidate, Mapping):
                nested = candidate
                break
        names = {field.name for field in fields(cls)}
        selected = {name: nested[name] for name in names if name in nested}
        return cls(**selected)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "period_source": self.period_source,
            "layer_count": int(self.layer_count),
            "stack_count": int(self.stack_count),
            "thickness_ratio": float(self.thickness_ratio),
            "width_ratio": float(self.width_ratio),
            "depth_ratio": float(self.depth_ratio),
            "spread_deg": float(self.spread_deg),
            "lateral_shift_ratio": float(self.lateral_shift_ratio),
            "out_of_plane_deg": float(self.out_of_plane_deg),
            "seed": int(self.seed),
            "manual_period": float(self.manual_period),
            "manual_angle_deg": float(self.manual_angle_deg),
            "manual_unit": self.manual_unit,
            "selected_branch": int(self.selected_branch),
            "reference_period": None
            if self.reference_period is None
            else float(self.reference_period),
        }


@dataclass
class LamellarScene:
    centers: np.ndarray
    sizes: np.ndarray
    orientations: np.ndarray
    vertices: np.ndarray
    stack_ids: np.ndarray
    branch_ids: np.ndarray
    colors: np.ndarray
    bounds: np.ndarray
    length_unit: str
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        self.centers = np.array(self.centers, dtype=float, copy=True).reshape((-1, 3))
        self.sizes = np.array(self.sizes, dtype=float, copy=True).reshape((-1, 3))
        self.orientations = np.array(self.orientations, dtype=float, copy=True).reshape(
            (-1, 3, 3)
        )
        self.vertices = np.array(self.vertices, dtype=float, copy=True).reshape(
            (-1, 8, 3)
        )
        self.stack_ids = np.array(self.stack_ids, dtype=int, copy=True).reshape((-1,))
        self.branch_ids = np.array(self.branch_ids, dtype=int, copy=True).reshape((-1,))
        self.colors = np.array(self.colors, dtype=float, copy=True).reshape((-1, 4))
        n = len(self.centers)
        if any(
            len(array) != n
            for array in (
                self.sizes,
                self.orientations,
                self.vertices,
                self.stack_ids,
                self.branch_ids,
                self.colors,
            )
        ):
            raise ValueError("LamellarScene arrays must share the same first dimension")
        if not all(
            np.all(np.isfinite(array))
            for array in (
                self.centers,
                self.sizes,
                self.orientations,
                self.vertices,
                self.colors,
            )
        ):
            raise ValueError("LamellarScene geometry arrays must be finite")
        self.colors = np.clip(self.colors, 0.0, 1.0)
        bounds = np.asarray(self.bounds, dtype=float)
        if bounds.shape != (2, 3) or not np.all(np.isfinite(bounds)):
            raise ValueError("LamellarScene bounds must have finite shape (2, 3)")
        self.bounds = np.array(bounds, dtype=float, copy=True)
        unit = str(self.length_unit or "relative")
        if unit not in {"nm", "relative"}:
            raise ValueError("LamellarScene.length_unit must be 'nm' or 'relative'")
        self.length_unit = unit
        self.metadata = _copy_metadata(dict(self.metadata or {}))

    @property
    def status(self) -> str:
        return str(self.metadata.get("status", "schematic"))

    @property
    def message(self) -> str:
        return str(self.metadata.get("message", ""))

    @property
    def source_identity(self) -> dict[str, Any]:
        identity = self.metadata.get("source_identity", {})
        return identity if isinstance(identity, dict) else {}

    def __getattr__(self, name: str) -> Any:
        metadata = self.__dict__.get("metadata", {})
        if name == "assumptions":
            return list(metadata.get(name, ()))
        if name in {"settings", "parameter_sources", "populations"}:
            return metadata.get(name, {} if name == "settings" else [])
        if name == "scientific_boundary":
            return str(metadata.get(name, ""))
        if name == "draw_axis_deg":
            return float(metadata.get(name, 90.0))
        if name == "reference_period":
            return float(metadata.get(name, 1.0))
        raise AttributeError(name)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def to_mapping(self) -> dict[str, Any]:
        return {
            name: getattr(self, name).copy()
            for name in (
                "centers",
                "sizes",
                "orientations",
                "vertices",
                "stack_ids",
                "branch_ids",
                "colors",
                "bounds",
            )
        } | {"length_unit": self.length_unit, "metadata": copy.deepcopy(self.metadata)}


def _empty_scene(
    settings: LamellarSettings, metadata: Mapping[str, Any]
) -> LamellarScene:
    unit = (
        str(metadata.get("length_unit", "relative"))
        if isinstance(metadata, Mapping)
        else "relative"
    )
    if unit not in {"nm", "relative"}:
        unit = "relative"
    return LamellarScene(
        centers=np.empty((0, 3), dtype=float),
        sizes=np.empty((0, 3), dtype=float),
        orientations=np.empty((0, 3, 3), dtype=float),
        vertices=np.empty((0, 8, 3), dtype=float),
        stack_ids=np.empty((0,), dtype=int),
        branch_ids=np.empty((0,), dtype=int),
        colors=np.empty((0, 4), dtype=float),
        bounds=np.asarray(((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)), dtype=float),
        length_unit=unit,
        metadata=metadata,
    )


__all__ = ["LamellarScene", "LamellarSettings"]
