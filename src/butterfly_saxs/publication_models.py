"""Presentation-only contracts for publication figures.

These records never alter fitted or schematic geometry parameters.  Dimensions
are physical output sizes; surface appearance is explicitly a drawing choice.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields
import math
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PublicationStyle:
    palette: str = "editorial"
    bevel_fraction: float = .12
    roughness: float = .72
    ambient_occlusion: bool = True

    def __post_init__(self) -> None:
        if self.palette not in {"editorial", "grayscale"}:
            raise ValueError("Unsupported publication palette")
        if not math.isfinite(self.bevel_fraction) or not 0 <= self.bevel_fraction <= .24:
            raise ValueError("bevel_fraction must be within [0, .24] of thickness")
        if not math.isfinite(self.roughness) or not .35 <= self.roughness <= 1.:
            raise ValueError("roughness must be within [.35, 1]")

    @property
    def colors(self) -> tuple[tuple[float, float, float, float], ...]:
        if self.palette == "grayscale":
            return ((.31, .36, .39, 1.), (.68, .70, .71, 1.))
        return ((.34, .54, .62, 1.), (.75, .56, .40, 1.))

    @classmethod
    def from_mapping(cls, value: Any = None) -> "PublicationStyle":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("PublicationStyle expects a mapping")
        return cls(**{item.name: value[item.name] for item in fields(cls) if item.name in value})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PublicationFigureSpec:
    template: str = "structure"
    width_mm: float = 183.
    height_mm: float = 95.
    dpi: int = 600
    language: str = "en"
    background: str = "white"
    quality: str = "publication"
    title: str = ""
    show_inset: bool = True
    annotations: bool = True
    annotation_positions: dict[str, list[float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.template not in {"structure", "evidence"}:
            raise ValueError("template must be structure or evidence")
        if self.language not in {"en", "zh", "zh_CN"}:
            raise ValueError("Unsupported publication language")
        if self.background not in {"white", "transparent"}:
            raise ValueError("background must be white or transparent")
        if self.quality not in {"preview", "publication"}:
            raise ValueError("quality must be preview or publication")
        if not (math.isfinite(self.width_mm) and 50 <= self.width_mm <= 300):
            raise ValueError("width_mm must be within [50, 300]")
        if not (math.isfinite(self.height_mm) and 40 <= self.height_mm <= 250):
            raise ValueError("height_mm must be within [40, 250]")
        if isinstance(self.dpi, bool) or not isinstance(self.dpi, int) or not 72 <= self.dpi <= 1200:
            raise ValueError("dpi must be an integer within [72, 1200]")
        clean = {}
        for name, position in self.annotation_positions.items():
            array = np.asarray(position, dtype=float)
            if array.shape != (2,) or not np.all(np.isfinite(array)) or np.any((array < 0) | (array > 1)):
                raise ValueError("Annotation positions must be normalized figure coordinates")
            clean[str(name)] = array.tolist()
        object.__setattr__(self, "annotation_positions", clean)

    @property
    def pixel_size(self) -> tuple[int, int]:
        return round(self.width_mm / 25.4 * self.dpi), round(self.height_mm / 25.4 * self.dpi)

    @property
    def main_panel(self) -> tuple[float, float, float, float]:
        if self.width_mm < 120:
            if self.template == "structure" and not self.show_inset:
                return (.03, .10, .94, .83)
            return (.03, .43 if self.template == "evidence" else .32, .94,
                    .50 if self.template == "evidence" else .61)
        if self.template == "evidence":
            return (.32, .12, .65, .80)
        return (.03, .12, .74 if self.show_inset else .94, .80)

    @property
    def auxiliary_panels(self) -> dict[str, tuple[float, float, float, float]]:
        """Physical-size aware supporting panels; no geometry is stretched."""
        if self.width_mm < 120:
            if self.template == "evidence":
                panels = {"saxs": (.09, .13, .32, .23), "projection": (.54, .25, .38, .14)}
                if self.show_inset:
                    panels["detail"] = (.55, .075, .38, .115)
                return panels
            return {"detail": (.56, .08, .38, .20)} if self.show_inset else {}
        if self.template == "evidence":
            panels = {"saxs": (.065, .68, .205, .245), "projection": (.06, .38, .215, .22)}
            if self.show_inset:
                panels["detail"] = (.06, .095, .215, .20)
            return panels
        return {"detail": (.79, .38, .19, .32)} if self.show_inset else {}

    @property
    def main_pixel_size(self) -> tuple[int, int]:
        width, height = self.pixel_size
        return max(1, round(width * self.main_panel[2])), max(1, round(height * self.main_panel[3]))

    @classmethod
    def from_mapping(cls, value: Any = None) -> "PublicationFigureSpec":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("PublicationFigureSpec expects a mapping")
        return cls(**{item.name: value[item.name] for item in fields(cls) if item.name in value})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PublicationRenderResult:
    rgba: np.ndarray
    camera: dict[str, Any]
    bounds: np.ndarray
    projected: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        image = np.asarray(self.rgba)
        if image.ndim != 3 or image.shape[2] != 4 or image.dtype != np.uint8 or not image.size:
            raise ValueError("Publication render must be a nonempty uint8 RGBA image")
        self.rgba = image
        self.bounds = np.asarray(self.bounds, dtype=float)
        if self.bounds.shape != (2, 3) or not np.all(np.isfinite(self.bounds)):
            raise ValueError("Publication bounds must have finite shape (2, 3)")


__all__ = ["PublicationStyle", "PublicationFigureSpec", "PublicationRenderResult"]
