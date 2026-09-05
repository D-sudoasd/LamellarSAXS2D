"""Auditable detector-mask and exclusion-ROI utilities.

Every public mask in this module uses detector polarity: ``True`` means the
pixel is excluded.  This is the convention accepted by the measurement and
full-pixel refinement functions; positive ``valid_mask`` arrays are inverted
explicitly at I/O boundaries.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np


class MaskSpecError(ValueError):
    """An exclusion ROI is malformed or incompatible with the image."""


def _finite_float(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise MaskSpecError(f"{name} must be a finite number") from exc
    if not np.isfinite(number):
        raise MaskSpecError(f"{name} must be a finite number")
    return number


def _required(spec: Mapping[str, Any], name: str) -> Any:
    try:
        return spec[name]
    except KeyError as exc:
        raise MaskSpecError(f"ROI is missing required field {name!r}") from exc


def _shape(value: Any) -> tuple[int, int]:
    try:
        height, width = (int(item) for item in value)
    except Exception as exc:
        raise MaskSpecError("shape must be a (height, width) pair") from exc
    if height <= 0 or width <= 0:
        raise MaskSpecError("shape dimensions must be positive")
    return height, width


def rectangle_mask(
    shape: tuple[int, int],
    *,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
) -> np.ndarray:
    """Return a pixel-space rectangular exclusion mask."""

    height, width = _shape(shape)
    left = _finite_float(x0, "x0")
    right = _finite_float(x1, "x1")
    top = _finite_float(y0, "y0")
    bottom = _finite_float(y1, "y1")
    if right < left or bottom < top:
        raise MaskSpecError("rectangle bounds must satisfy x1 >= x0 and y1 >= y0")
    yy, xx = np.indices((height, width), dtype=float)
    return (xx >= left) & (xx <= right) & (yy >= top) & (yy <= bottom)


def ellipse_mask(
    shape: tuple[int, int],
    *,
    cx: float,
    cy: float,
    rx: float,
    ry: float,
    angle_deg: float = 0.0,
) -> np.ndarray:
    """Return a rotated pixel-space elliptical exclusion mask."""

    height, width = _shape(shape)
    cx = _finite_float(cx, "cx")
    cy = _finite_float(cy, "cy")
    rx = _finite_float(rx, "rx")
    ry = _finite_float(ry, "ry")
    angle_deg = _finite_float(angle_deg, "angle_deg")
    if rx <= 0 or ry <= 0:
        raise MaskSpecError("ellipse radii must be finite and positive")
    yy, xx = np.indices((height, width), dtype=float)
    angle = np.deg2rad(angle_deg)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    dx, dy = xx - cx, yy - cy
    local_x = cos_a * dx + sin_a * dy
    local_y = -sin_a * dx + cos_a * dy
    return (local_x / rx) ** 2 + (local_y / ry) ** 2 <= 1.0


def q_sector_mask(
    qx: Any,
    qy: Any,
    *,
    q_min: float | None = None,
    q_max: float | None = None,
    chi_min_deg: float = -180.0,
    chi_max_deg: float = 180.0,
) -> np.ndarray:
    """Return an exclusion mask for a calibrated q/azimuth sector.

    Azimuth limits may cross the -180/180 boundary (for example 170 to -170).
    """

    try:
        qx_array, qy_array = np.broadcast_arrays(
            np.asarray(qx, dtype=float), np.asarray(qy, dtype=float)
        )
    except ValueError as exc:
        raise MaskSpecError("qx/qy shapes are not broadcast-compatible") from exc
    if qx_array.ndim != 2:
        raise MaskSpecError("qx/qy must be two-dimensional")
    lower = None if q_min is None else _finite_float(q_min, "q_min")
    upper = None if q_max is None else _finite_float(q_max, "q_max")
    if lower is not None and lower < 0:
        raise MaskSpecError("q_min must be non-negative")
    if upper is not None and upper < 0:
        raise MaskSpecError("q_max must be non-negative")
    if lower is not None and upper is not None and upper < lower:
        raise MaskSpecError("q bounds must satisfy q_max >= q_min")
    chi_min = _finite_float(chi_min_deg, "chi_min_deg")
    chi_max = _finite_float(chi_max_deg, "chi_max_deg")
    q = np.hypot(qx_array, qy_array)
    chi = (np.degrees(np.arctan2(qy_array, qx_array)) + 180.0) % 360.0 - 180.0
    selected = np.isfinite(q) & np.isfinite(chi)
    if lower is not None:
        selected &= q >= lower
    if upper is not None:
        selected &= q <= upper
    lo = (chi_min + 180.0) % 360.0 - 180.0
    hi = (chi_max + 180.0) % 360.0 - 180.0
    if np.isclose((chi_max - chi_min) % 360.0, 0.0) and not np.isclose(
        chi_max, chi_min
    ):
        angular = np.ones_like(selected)
    elif lo <= hi:
        angular = (chi >= lo) & (chi <= hi)
    else:
        angular = (chi >= lo) | (chi <= hi)
    return selected & angular


def mask_from_roi(
    shape: tuple[int, int],
    spec: Mapping[str, Any],
    *,
    qx: Any = None,
    qy: Any = None,
) -> np.ndarray:
    """Build one exclusion mask from a serializable ROI specification."""

    if not isinstance(spec, Mapping):
        raise MaskSpecError("ROI specification must be a mapping")
    kind = str(spec.get("type", spec.get("kind", ""))).strip().lower().replace("-", "_")
    if kind in {"rectangle", "rect", "box"}:
        return rectangle_mask(
            shape,
            x0=_required(spec, "x0"),
            x1=_required(spec, "x1"),
            y0=_required(spec, "y0"),
            y1=_required(spec, "y1"),
        )
    if kind in {"ellipse", "elliptical"}:
        return ellipse_mask(
            shape,
            cx=_required(spec, "cx"),
            cy=_required(spec, "cy"),
            rx=_required(spec, "rx"),
            ry=_required(spec, "ry"),
            angle_deg=spec.get("angle_deg", 0.0),
        )
    if kind in {"q_sector", "sector", "annular_sector"}:
        if qx is None or qy is None:
            raise MaskSpecError("q-sector ROI requires qx and qy maps")
        mask = q_sector_mask(
            qx,
            qy,
            q_min=spec.get("q_min"),
            q_max=spec.get("q_max"),
            chi_min_deg=spec.get("chi_min_deg", -180.0),
            chi_max_deg=spec.get("chi_max_deg", 180.0),
        )
        if mask.shape != _shape(shape):
            raise MaskSpecError("q-sector map shape does not match image")
        return mask
    raise MaskSpecError(f"unknown ROI type: {kind or '<missing>'}")


def combine_exclusion_masks(
    shape: tuple[int, int],
    *,
    masks: Iterable[Any] = (),
    rois: Iterable[Mapping[str, Any]] = (),
    qx: Any = None,
    qy: Any = None,
) -> np.ndarray:
    """OR-combine detector masks and ROI specifications (``True`` invalid)."""

    canonical_shape = _shape(shape)
    combined = np.zeros(canonical_shape, dtype=bool)
    for value in masks:
        array = np.asarray(value, dtype=bool)
        if array.shape != canonical_shape:
            raise MaskSpecError(f"mask shape {array.shape} does not match {canonical_shape}")
        combined |= array
    for spec in rois:
        combined |= mask_from_roi(canonical_shape, spec, qx=qx, qy=qy)
    return combined


__all__ = [
    "MaskSpecError",
    "combine_exclusion_masks",
    "ellipse_mask",
    "mask_from_roi",
    "q_sector_mask",
    "rectangle_mask",
]
