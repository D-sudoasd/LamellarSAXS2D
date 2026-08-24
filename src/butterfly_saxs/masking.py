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
    left, right = sorted((float(x0), float(x1)))
    top, bottom = sorted((float(y0), float(y1)))
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
    rx, ry = float(rx), float(ry)
    if not np.isfinite(rx) or not np.isfinite(ry) or rx <= 0 or ry <= 0:
        raise MaskSpecError("ellipse radii must be finite and positive")
    yy, xx = np.indices((height, width), dtype=float)
    angle = np.deg2rad(float(angle_deg))
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    dx, dy = xx - float(cx), yy - float(cy)
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

    qx_array, qy_array = np.broadcast_arrays(np.asarray(qx, dtype=float), np.asarray(qy, dtype=float))
    if qx_array.ndim != 2:
        raise MaskSpecError("qx/qy must be two-dimensional")
    q = np.hypot(qx_array, qy_array)
    chi = (np.degrees(np.arctan2(qy_array, qx_array)) + 180.0) % 360.0 - 180.0
    selected = np.isfinite(q) & np.isfinite(chi)
    if q_min is not None:
        selected &= q >= float(q_min)
    if q_max is not None:
        selected &= q <= float(q_max)
    lo = (float(chi_min_deg) + 180.0) % 360.0 - 180.0
    hi = (float(chi_max_deg) + 180.0) % 360.0 - 180.0
    if np.isclose((float(chi_max_deg) - float(chi_min_deg)) % 360.0, 0.0) and not np.isclose(
        float(chi_max_deg), float(chi_min_deg)
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

    kind = str(spec.get("type", spec.get("kind", ""))).strip().lower().replace("-", "_")
    if kind in {"rectangle", "rect", "box"}:
        return rectangle_mask(
            shape,
            x0=spec["x0"],
            x1=spec["x1"],
            y0=spec["y0"],
            y1=spec["y1"],
        )
    if kind in {"ellipse", "elliptical"}:
        return ellipse_mask(
            shape,
            cx=spec["cx"],
            cy=spec["cy"],
            rx=spec["rx"],
            ry=spec["ry"],
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
