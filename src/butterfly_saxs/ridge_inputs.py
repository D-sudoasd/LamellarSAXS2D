"""Shared image/q-map input and detector-coordinate helpers.

The butterfly ridge, normal-profile, and uncertainty paths all consume the
same physical q-map seam.  This module keeps aliases and mask polarity in one
place while leaving the source objects untouched.  ``mask``/``bad_mask`` /
``invalid_mask`` fields mean *invalid* pixels; ``valid_mask``/``valid`` fields
mean *valid* pixels.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

try:  # scipy is a declared project dependency; keep a small NumPy fallback.
    from scipy.ndimage import map_coordinates
except Exception:  # pragma: no cover - source inspection in partial installs
    map_coordinates = None


IMAGE_DATA_ALIASES = ("data", "intensity", "image", "values")
INVALID_MASK_ALIASES = ("mask", "bad_mask", "invalid_mask")
VALID_MASK_ALIASES = ("valid_mask", "valid")
QX_ALIASES = ("qx", "q_x", "x", "qx_map")
QY_ALIASES = ("qy", "q_y", "y", "qy_map")
Q_ALIASES = ("q", "q_map", "q_abs", "radius", "q_magnitude")


def field(value: Any, names: Sequence[str], default: Any = None) -> Any:
    """Read the first available field from a mapping or light-weight object."""

    if value is None:
        return default
    if isinstance(value, Mapping):
        for name in names:
            if name in value:
                return value[name]
        return default
    for name in names:
        try:
            result = getattr(value, name)
        except Exception:
            continue
        if result is not None:
            return result
    return default


def array_field(value: Any, names: Sequence[str], default: Any = None) -> np.ndarray | None:
    """Return a field as an ndarray, or ``None`` when absent/unreadable."""

    candidate = field(value, names, default)
    if candidate is None:
        return None
    try:
        return np.asarray(candidate)
    except Exception:
        return None


def _broadcast_bool(value: Any, shape: tuple[int, int], message: str) -> np.ndarray:
    try:
        return np.asarray(np.broadcast_to(np.asarray(value, dtype=bool), shape), dtype=bool)
    except (TypeError, ValueError) as exc:
        raise ValueError(message) from exc


def _broadcast_float(value: Any, shape: tuple[int, int], message: str) -> np.ndarray:
    try:
        return np.asarray(np.broadcast_to(np.asarray(value, dtype=float), shape), dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(message) from exc


def invalid_mask(value: Any, shape: tuple[int, int], *, label: str) -> np.ndarray | None:
    """Resolve detector-style and positive-valid mask aliases to invalid=True."""

    raw = array_field(value, INVALID_MASK_ALIASES)
    valid = array_field(value, VALID_MASK_ALIASES)
    result: np.ndarray | None = None
    if raw is not None:
        result = _broadcast_bool(raw, shape, f"{label} mask does not broadcast to image shape {shape}").copy()
    if valid is not None:
        valid_array = _broadcast_bool(valid, shape, f"{label} valid mask does not broadcast to image shape {shape}")
        if result is None:
            result = ~valid_array
        else:
            result |= ~valid_array
    return result


def as_image(image: Any) -> tuple[np.ndarray, np.ndarray | None]:
    """Return a 2-D floating image and its optional canonical invalid mask."""

    data = np.asarray(image) if isinstance(image, np.ndarray) else array_field(image, IMAGE_DATA_ALIASES)
    if data is None and not isinstance(image, Mapping):
        data = np.asarray(image)
    if data is None:
        raise ValueError("image must be a two-dimensional ndarray or expose data/intensity/image/values")
    array = np.asarray(data, dtype=float)
    if array.ndim != 2:
        raise ValueError("image must be two-dimensional")
    return array, invalid_mask(image, tuple(array.shape), label="image")


def as_qmap(
    qmap: Any,
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Return broadcasted ``qx``, ``qy``, ``q`` and invalid mask arrays.

    The coordinate and q arrays are views when the source can be broadcast
    without a copy.  The adapter never writes to them or to the source q-map.
    """

    if len(shape) != 2:
        raise ValueError("image shape must be two-dimensional")
    qx_raw = array_field(qmap, QX_ALIASES)
    qy_raw = array_field(qmap, QY_ALIASES)
    if qx_raw is None or qy_raw is None:
        raise ValueError("qmap must expose qx/qy arrays")
    qx = _broadcast_float(qx_raw, shape, f"qmap qx does not broadcast to image shape {shape}")
    qy = _broadcast_float(qy_raw, shape, f"qmap qy does not broadcast to image shape {shape}")
    q_raw = array_field(qmap, Q_ALIASES)
    q = np.hypot(qx, qy) if q_raw is None else _broadcast_float(q_raw, shape, f"qmap q does not broadcast to image shape {shape}")
    return qx, qy, q, invalid_mask(qmap, shape, label="qmap")


def canonical_inputs(
    image: Any,
    qmap: Any,
    *,
    mask: Any = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return image/q coordinates and one invalid mask for a complete input."""

    data, image_invalid = as_image(image)
    qx, qy, q, qmap_invalid = as_qmap(qmap, tuple(data.shape))
    invalid = np.zeros(data.shape, dtype=bool)
    if image_invalid is not None:
        invalid |= image_invalid
    if qmap_invalid is not None:
        invalid |= qmap_invalid
    if mask is not None:
        invalid |= _broadcast_bool(mask, tuple(data.shape), f"mask does not broadcast to image shape {data.shape}")
    return data, qx, qy, q, invalid


def _gradient(array: np.ndarray, axis: int) -> np.ndarray:
    """Finite-difference gradient that handles one-pixel detector axes."""

    if array.shape[axis] < 2:
        return np.zeros_like(array, dtype=float)
    return np.asarray(np.gradient(array, axis=axis), dtype=float)


def representative_q_step(qmap: Any, qy: Any = None) -> float | None:
    """Return a robust physical q step using both detector axes.

    ``qmap`` may be a mapping/object exposing qx/qy aliases, or a qx array
    when ``qy`` is supplied explicitly.  Degenerate or wholly non-finite maps
    return ``None``; callers that need a numerical fallback should choose it
    explicitly rather than treating a singular Jacobian as one physical unit.
    """

    if qy is None:
        qx_raw = array_field(qmap, QX_ALIASES)
        qy_raw = array_field(qmap, QY_ALIASES)
    else:
        qx_raw, qy_raw = qmap, qy
    if qx_raw is None or qy_raw is None:
        return None
    try:
        qx, qy_array = np.broadcast_arrays(np.asarray(qx_raw, dtype=float), np.asarray(qy_raw, dtype=float))
    except (TypeError, ValueError):
        return None
    if qx.ndim != 2:
        return None
    qx_y, qx_x = _gradient(qx, 0), _gradient(qx, 1)
    qy_y, qy_x = _gradient(qy_array, 0), _gradient(qy_array, 1)
    steps = np.concatenate((np.hypot(qx_x, qy_x).ravel(), np.hypot(qx_y, qy_y).ravel()))
    finite = steps[np.isfinite(steps) & (steps > np.finfo(float).eps)]
    return float(np.median(finite)) if finite.size else None


def coordinate_derivatives(
    qx: np.ndarray,
    qy: np.ndarray,
) -> tuple[dict[str, np.ndarray], np.ndarray, float]:
    """Build detector-to-q Jacobian/derivatives once for a trace.

    The returned arrays are independent derivative buffers.  Singular
    Jacobians retain NaNs in their inverse entries; ``q_step`` falls back to
    ``1.0`` only for numerical compatibility with the ridge equations when a
    map has no finite positive step.
    """

    qx_array, qy_array = np.broadcast_arrays(np.asarray(qx, dtype=float), np.asarray(qy, dtype=float))
    qx_y, qx_x = _gradient(qx_array, 0), _gradient(qx_array, 1)
    qy_y, qy_x = _gradient(qy_array, 0), _gradient(qy_array, 1)
    qx_yy, qx_yx_from_y = _gradient(qx_y, 0), _gradient(qx_y, 1)
    qx_xy_from_x, qx_xx = _gradient(qx_x, 0), _gradient(qx_x, 1)
    qy_yy, qy_yx_from_y = _gradient(qy_y, 0), _gradient(qy_y, 1)
    qy_xy_from_x, qy_xx = _gradient(qy_x, 0), _gradient(qy_x, 1)
    qx_yx = 0.5 * (qx_yx_from_y + qx_xy_from_x)
    qy_yx = 0.5 * (qy_yx_from_y + qy_xy_from_x)
    determinant = qx_x * qy_y - qx_y * qy_x
    with np.errstate(divide="ignore", invalid="ignore"):
        inv00 = qy_y / determinant
        inv01 = -qx_y / determinant
        inv10 = -qy_x / determinant
        inv11 = qx_x / determinant
    step_x = np.hypot(qx_x, qy_x)
    step_y = np.hypot(qx_y, qy_y)
    steps = np.concatenate((step_x.ravel(), step_y.ravel()))
    finite_steps = steps[np.isfinite(steps) & (steps > np.finfo(float).eps)]
    q_step = float(np.median(finite_steps)) if finite_steps.size else 1.0
    derivatives = {
        "qx_x": qx_x,
        "qx_y": qx_y,
        "qy_x": qy_x,
        "qy_y": qy_y,
        "qx_xx": qx_xx,
        "qx_xy": qx_yx,
        "qx_yy": qx_yy,
        "qy_xx": qy_xx,
        "qy_xy": qy_yx,
        "qy_yy": qy_yy,
        "determinant": determinant,
        "inv00": inv00,
        "inv01": inv01,
        "inv10": inv10,
        "inv11": inv11,
        "step_x": step_x,
        "step_y": step_y,
    }
    inverse = np.asarray([inv00, inv01, inv10, inv11])
    return derivatives, inverse, q_step


def sample(array: np.ndarray, y: np.ndarray, x: np.ndarray, *, order: int = 1) -> np.ndarray:
    """Sample a detector array at floating detector coordinates."""

    if map_coordinates is None:  # pragma: no cover
        yi = np.clip(np.rint(y).astype(int), 0, array.shape[0] - 1)
        xi = np.clip(np.rint(x).astype(int), 0, array.shape[1] - 1)
        return np.asarray(array[yi, xi], dtype=float)
    values = array if getattr(array, "dtype", None) == np.float64 else np.asarray(array, dtype=float)
    sample_y = y if getattr(y, "dtype", None) == np.float64 else np.asarray(y, dtype=float)
    sample_x = x if getattr(x, "dtype", None) == np.float64 else np.asarray(x, dtype=float)
    return np.asarray(
        map_coordinates(
            values,
            [sample_y, sample_x],
            order=order,
            mode="nearest",
            prefilter=False,
        ),
        dtype=float,
    )


def robust_noise(values: np.ndarray) -> float:
    """Estimate noise from finite values using MAD with a standard-deviation fallback."""

    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size < 2:
        return float("nan")
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    if np.isfinite(mad) and mad > 0:
        return 1.4826 * mad
    return float(np.std(finite, ddof=1)) if finite.size > 1 else float("nan")


__all__ = [
    "IMAGE_DATA_ALIASES",
    "INVALID_MASK_ALIASES",
    "VALID_MASK_ALIASES",
    "QX_ALIASES",
    "QY_ALIASES",
    "Q_ALIASES",
    "array_field",
    "as_image",
    "as_qmap",
    "canonical_inputs",
    "coordinate_derivatives",
    "field",
    "invalid_mask",
    "representative_q_step",
    "robust_noise",
    "sample",
]
