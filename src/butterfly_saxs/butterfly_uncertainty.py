"""Image-level empirical uncertainty evaluation for butterfly refinements.

The public entry point, :func:`resample_butterfly`, perturbs an image by
resampling spatial residual blocks and sends every perturbed image through a
caller-supplied refit function.  The result is an empirical central 95 percent
interval.  It is deliberately labelled as an empirical interval rather than a
calibrated confidence interval: the routine has no instrument calibration,
point-spread, or counting-statistics knowledge unless the caller supplies it.

The callback receives ``resamples=0`` and ``stage='evaluate'`` in its options
on every call.  This makes recursion into the uncertainty routine explicit
and prevents a nested resampling run.  Center and q-calibration perturbations
are passed through both a perturbed q-map and callback options when their
standard deviations are supplied.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import copy
import math
from typing import Any

import numpy as np

from .cancellation import raise_if_cancelled
from .butterfly_settings import MAX_RESAMPLES
from .ridge_inputs import (
    Q_ALIASES,
    as_image as _shared_as_image,
    as_qmap as _shared_as_qmap,
    invalid_mask as _shared_invalid_mask,
)
from .settings import strict_int

try:  # scipy is a runtime dependency, but this module keeps a NumPy fallback.
    from scipy.ndimage import gaussian_filter
except Exception:  # pragma: no cover - only for deliberately minimal runtimes
    gaussian_filter = None


DEFAULT_RESAMPLES = 32
# Keep the image-level helper aligned with the public butterfly recipe limit.
# Interactive callers normally use 32; larger campaigns must still opt in by
# requesting a count explicitly within the shared hard cap.
DEFAULT_MAX_RESAMPLES = MAX_RESAMPLES
DEFAULT_BLOCK_SHAPE = (8, 8)
UNCERTAINTY_METHOD = "empirical_image_residual_spatial_block"
UNCERTAINTY_INTERVAL = "central95_empirical_interval"


class _CanonicalPerturbedQMap(dict[str, Any]):
    """Mapping seam for perturbed q maps, with attribute compatibility.

    Production ``GeometryMaps`` exposes read-only qx/qy/q properties.  A
    mapping avoids attempting to mutate those properties while retaining
    ``qmap.qx`` compatibility for lightweight callback adapters.
    """

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:  # pragma: no cover - normal attribute errors
            raise AttributeError(name) from exc


_TOPOLOGY_FAILURE_CODES = frozenset(
    {
        "topology",
        "topology_failed",
        "topology_failure",
        "topology_loss",
        "arc_topology_failed",
        "missing_branch",
        "missing_counterpart",
        "missing_counterpart_branch",
        "missing_side",
        "no_observable_arcs",
        "no_observable_arc",
        "insufficient_geometry",
        "insufficient_side_support",
        "insufficient_independent_side_support",
        "coverage_gap",
        "unresolved_topology",
    }
)
_REFIT_FAILURE_CODES = frozenset(
    {
        "refit",
        "refit_failed",
        "solver_failed",
        "solver_error",
        "optimizer_failed",
        "invalid_fit",
    }
)


def _field(value: Any, names: Sequence[str], default: Any = None) -> Any:
    if value is None:
        return default
    resolver = getattr(value, "resolve", None)
    if callable(resolver):
        try:
            resolved = resolver()
        except Exception:
            resolved = None
        if isinstance(resolved, Mapping):
            for name in names:
                if name in resolved:
                    return resolved[name]
    if isinstance(value, Mapping):
        for name in names:
            if name in value:
                result = value[name]
                # ParameterSpec-like wrappers expose their numeric value as
                # an attribute.  Resolve() above handles tied expressions;
                # this fallback keeps ordinary parameter mappings usable.
                numeric = getattr(result, "value", None)
                return numeric if numeric is not None else result
        return default
    for name in names:
        try:
            result = getattr(value, name)
        except Exception:
            continue
        if result is not None:
            return result
    return default


_MISSING = object()


def _present_field(value: Any, name: str) -> tuple[bool, Any]:
    """Return whether a field exists, preserving an explicit ``None``."""

    if value is None:
        return False, _MISSING
    if isinstance(value, Mapping):
        if name in value:
            return True, value[name]
        return False, _MISSING
    try:
        return True, getattr(value, name)
    except AttributeError:
        return False, _MISSING
    except Exception:
        return True, _MISSING


def _array_field(value: Any, names: Sequence[str], default: Any = None) -> np.ndarray | None:
    raw = _field(value, names, default)
    if raw is None:
        return None
    try:
        return np.asarray(raw)
    except Exception:
        return None


def _extract_image(image: Any) -> np.ndarray:
    data, _image_invalid = _shared_as_image(image)
    data = np.asarray(data, dtype=float)
    if data.ndim != 2 or data.size == 0:
        raise ValueError("image must be a non-empty two-dimensional array")
    if not np.any(np.isfinite(data)):
        raise ValueError("image must contain at least one finite pixel")
    return data.copy()


def _extract_q(qmap: Any, shape: tuple[int, int]) -> np.ndarray | None:
    if qmap is None:
        return None
    q = _array_field(qmap, Q_ALIASES)
    if q is None:
        qx = _array_field(qmap, ("qx", "q_x", "x", "qx_map"))
        qy = _array_field(qmap, ("qy", "q_y", "y", "qy_map"))
        if qx is not None and qy is not None:
            q = np.hypot(qx, qy)
    if q is None:
        return None
    try:
        q = np.broadcast_to(np.asarray(q, dtype=float), shape)
    except ValueError as exc:
        raise ValueError(f"q map shape does not broadcast to image shape {shape}") from exc
    return np.asarray(q, dtype=float)


def _extract_qxy(qmap: Any, shape: tuple[int, int]) -> tuple[np.ndarray | None, np.ndarray | None]:
    if qmap is None:
        return None, None
    try:
        qx, qy, _q, _mask = _shared_as_qmap(qmap, shape)
        return qx, qy
    except ValueError as exc:
        if "qx/qy" not in str(exc):
            raise
        return None, None


def _extract_qmap_mask(qmap: Any, shape: tuple[int, int]) -> np.ndarray:
    result = _shared_invalid_mask(qmap, shape, label="q-map")
    if result is None:
        return np.zeros(shape, dtype=bool)
    return np.asarray(result, dtype=bool)


def _extract_image_mask(image: Any, shape: tuple[int, int]) -> np.ndarray:
    result = _shared_invalid_mask(image, shape, label="image")
    if result is None:
        return np.zeros(shape, dtype=bool)
    return np.asarray(result, dtype=bool)


def _parse_q_window(value: Any) -> tuple[float, float] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        low = _field(value, ("min", "q_min", "low", "start"))
        high = _field(value, ("max", "q_max", "high", "stop"))
    else:
        try:
            low, high = tuple(value)
        except Exception as exc:
            raise ValueError("q_window must be a (min, max) pair") from exc
    try:
        low_f, high_f = float(low), float(high)
    except (TypeError, ValueError) as exc:
        raise ValueError("q_window limits must be finite numbers") from exc
    if not np.isfinite(low_f) or not np.isfinite(high_f) or high_f <= low_f:
        raise ValueError("q_window must contain finite max > min")
    return low_f, high_f


def _validate_integer(value: Any, name: str, *, minimum: int = 0) -> int:
    # Keep the uncertainty API's historical TypeError for booleans and
    # decimal values while delegating integer parsing/range checks to the
    # repository-wide strict helper.
    if isinstance(value, (bool, np.bool_)) or isinstance(value, (float, np.floating)):
        raise TypeError(f"{name} must be an integer")
    return strict_int(value, name, minimum=int(minimum))


def _parse_block_shape(value: Any) -> tuple[int, int]:
    if value is None:
        return DEFAULT_BLOCK_SHAPE
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("block_shape must contain positive integers")
    if isinstance(value, (int, np.integer)):
        row = col = _validate_integer(value, "block_shape", minimum=1)
        return row, col
    try:
        values = tuple(value)
    except TypeError as exc:
        raise TypeError("block_shape must be an integer or a pair of integers") from exc
    if len(values) != 2:
        raise ValueError("block_shape must contain exactly two integers")
    return (
        _validate_integer(values[0], "block_shape[0]", minimum=1),
        _validate_integer(values[1], "block_shape[1]", minimum=1),
    )


def _validate_mask(mask: Any, shape: tuple[int, int]) -> np.ndarray:
    if mask is None:
        return np.zeros(shape, dtype=bool)
    try:
        result = np.broadcast_to(np.asarray(mask, dtype=bool), shape)
    except ValueError as exc:
        raise ValueError(f"mask shape does not match image shape {shape}") from exc
    return np.asarray(result, dtype=bool).copy()


def _crop_for_q_window(valid: np.ndarray, q: np.ndarray | None, q_window: tuple[float, float] | None) -> tuple[slice, slice, np.ndarray]:
    if q_window is None or q is None:
        rows, cols = valid.shape
        return slice(0, rows), slice(0, cols), valid.copy()
    q_valid = np.isfinite(q) & (q >= q_window[0]) & (q <= q_window[1])
    selected = valid & q_valid
    positions = np.argwhere(q_valid)
    if positions.size == 0:
        raise ValueError("q_window selects no q-map pixels")
    row_lo, col_lo = np.min(positions, axis=0)
    row_hi, col_hi = np.max(positions, axis=0) + 1
    row_slice = slice(int(row_lo), int(row_hi))
    col_slice = slice(int(col_lo), int(col_hi))
    return row_slice, col_slice, selected[row_slice, col_slice]


def _fallback_smooth(image: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0.0:
        return np.asarray(image, dtype=float)
    radius = max(1, int(math.ceil(3.0 * float(sigma))))
    coordinate = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (coordinate / float(sigma)) ** 2)
    kernel /= np.sum(kernel)
    padded = np.pad(np.asarray(image, dtype=float), ((radius, radius), (radius, radius)), mode="reflect")
    horizontal = np.empty_like(padded)
    for row in range(padded.shape[0]):
        horizontal[row] = np.convolve(padded[row], kernel, mode="same")
    vertical = np.empty_like(horizontal)
    for col in range(horizontal.shape[1]):
        vertical[:, col] = np.convolve(horizontal[:, col], kernel, mode="same")
    return vertical[radius:-radius, radius:-radius]


def _smooth(image: np.ndarray, sigma: float) -> np.ndarray:
    if float(sigma) <= 0.0:
        return np.asarray(image, dtype=float)
    if gaussian_filter is not None:
        return np.asarray(gaussian_filter(np.asarray(image, dtype=float), sigma=float(sigma), mode="reflect"), dtype=float)
    return _fallback_smooth(image, float(sigma))


def _robust_fill(image: np.ndarray, valid: np.ndarray) -> np.ndarray:
    finite = valid & np.isfinite(image)
    if np.any(finite):
        fill = float(np.nanmedian(image[finite]))
    else:
        finite_all = np.isfinite(image)
        fill = float(np.nanmedian(image[finite_all])) if np.any(finite_all) else 0.0
    return np.where(np.isfinite(image), image, fill).astype(float, copy=False)


def _safe_gradient(array: np.ndarray, axis: int) -> np.ndarray:
    if array.shape[axis] < 2:
        return np.zeros_like(array, dtype=float)
    return np.asarray(np.gradient(array, axis=axis), dtype=float)


def _residual_field(image: np.ndarray, valid: np.ndarray, smoothing_sigma: float) -> np.ndarray:
    filled = _robust_fill(image, valid)
    smooth = _smooth(filled, smoothing_sigma)
    residual = np.asarray(filled - smooth, dtype=float)
    finite_valid = valid & np.isfinite(residual)
    if np.any(finite_valid):
        residual = residual - float(np.median(residual[finite_valid]))
    residual[~finite_valid] = 0.0
    return residual


def _sample_block_residual(
    residual: np.ndarray,
    valid: np.ndarray,
    block_shape: tuple[int, int],
    rng: np.random.Generator,
) -> np.ndarray:
    """Moving spatial-block residual bootstrap on a cropped image."""

    rows, cols = residual.shape
    block_rows = max(1, min(int(block_shape[0]), rows))
    block_cols = max(1, min(int(block_shape[1]), cols))
    candidate_origins: list[tuple[int, int]] = []
    for row in range(0, rows, block_rows):
        for col in range(0, cols, block_cols):
            row_end = min(rows, row + block_rows)
            col_end = min(cols, col + block_cols)
            if np.any(valid[row:row_end, col:col_end]):
                candidate_origins.append((row, col))
    if not candidate_origins:
        return np.zeros_like(residual, dtype=float)
    result = np.zeros_like(residual, dtype=float)
    for row in range(0, rows, block_rows):
        for col in range(0, cols, block_cols):
            height = min(block_rows, rows - row)
            width = min(block_cols, cols - col)
            candidate_row, candidate_col = candidate_origins[int(rng.integers(0, len(candidate_origins)))]
            source_row = min(candidate_row, max(0, rows - height))
            source_col = min(candidate_col, max(0, cols - width))
            source = residual[source_row:source_row + height, source_col:source_col + width]
            if source.shape != (height, width):
                # The only possible mismatch is a candidate edge block.  A
                # periodic resize keeps the correlation structure explicit.
                source = np.resize(source, (height, width))
            result[row:row + height, col:col + width] = source
    result[~valid] = 0.0
    return result


def _as_float_pair(value: Any, name: str) -> tuple[float, float]:
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, (bool, np.bool_)):
        scalar = float(value)
        if not np.isfinite(scalar) or scalar < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
        return scalar, scalar
    try:
        values = tuple(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be a scalar or a pair") from exc
    if len(values) != 2:
        raise ValueError(f"{name} must contain exactly two values")
    result = (float(values[0]), float(values[1]))
    if any(not np.isfinite(v) or v < 0.0 for v in result):
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _readonly_view(value: Any, *, dtype: Any = None) -> np.ndarray:
    """Return a non-writeable view, preserving source storage when possible."""

    array = np.asarray(value, dtype=dtype)
    view = array.view()
    view.flags.writeable = False
    return view


def _qmap_alias_is_readonly(qmap: Any) -> bool:
    for names in (
        ("qx", "q_x", "x", "qx_map"),
        ("qy", "q_y", "y", "qy_map"),
        Q_ALIASES,
        ("mask", "bad_mask", "invalid_mask"),
        ("valid_mask", "valid"),
    ):
        raw = _array_field(qmap, names)
        if raw is not None and np.asarray(raw).flags.writeable:
            return False
    return True


def _readonly_qmap_alias(qmap: Any, shape: tuple[int, int]) -> Any:
    """Expose a zero-perturbation q-map without writable source aliases."""

    if _qmap_alias_is_readonly(qmap):
        return qmap
    qx, qy = _extract_qxy(qmap, shape)
    q = _extract_q(qmap, shape)
    invalid = _shared_invalid_mask(qmap, shape, label="q-map")
    if qx is None or qy is None:
        raw_q = _array_field(qmap, Q_ALIASES)
        if raw_q is None:
            return qmap
        q_view = _readonly_view(np.broadcast_to(np.asarray(raw_q, dtype=float), shape))
        return _CanonicalPerturbedQMap({"q": q_view, "q_map": q_view, "shape": shape})
    qx_view, qy_view = _readonly_view(qx), _readonly_view(qy)
    q_view = _readonly_view(q if q is not None else np.hypot(qx, qy))
    chi = np.arctan2(qy, qx)
    if invalid is None:
        detector_mask = _readonly_view(np.zeros(shape, dtype=bool))
        valid_mask = _readonly_view(np.ones(shape, dtype=bool))
    else:
        detector_mask = _readonly_view(invalid, dtype=bool)
        valid_mask = _readonly_view(~np.asarray(invalid, dtype=bool))
    metadata_value = _field(qmap, ("metadata",), {})
    metadata = copy.deepcopy(dict(metadata_value)) if isinstance(metadata_value, Mapping) else {}
    q_unit = str(_field(qmap, ("q_unit", "unit"), metadata.get("q_unit", "unknown")) or "unknown")
    fingerprint = _field(qmap, ("fingerprint", "geometry_fingerprint"), "")
    return _CanonicalPerturbedQMap(
        {
            "qx": qx_view,
            "qy": qy_view,
            "q": q_view,
            "q_map": q_view,
            "chi": _readonly_view(chi),
            "angle": _readonly_view(chi),
            "azimuth": _readonly_view(chi),
            "mask": detector_mask,
            "bad_mask": detector_mask,
            "valid_mask": valid_mask,
            "valid": valid_mask,
            "q_unit": q_unit,
            "unit": q_unit,
            "metadata": metadata,
            "fingerprint": fingerprint,
            "geometry_fingerprint": fingerprint,
            "shape": shape,
        }
    )


def _copy_qmap_with_perturbation(
    qmap: Any,
    shape: tuple[int, int],
    center_delta_px: tuple[float, float],
    qcal_scale: float,
) -> Any:
    if qmap is None:
        if any(float(value) != 0. for value in center_delta_px) or not np.isclose(float(qcal_scale), 1.):
            raise ValueError("declared center/q-calibration perturbation requires a qmap")
        return None
    zero_perturbation = (
        all(abs(float(value)) <= np.finfo(float).eps for value in center_delta_px)
        and np.isclose(float(qcal_scale), 1.0)
    )
    if zero_perturbation:
        return _readonly_qmap_alias(qmap, shape)
    qx, qy = _extract_qxy(qmap, shape)
    if qx is None or qy is None:
        if any(abs(float(value)) > np.finfo(float).eps for value in center_delta_px) or not np.isclose(float(qcal_scale), 1.0):
            raise ValueError("declared center/q-calibration perturbation requires qmap qx/qy arrays")
        return qmap
    qcal_scale = float(qcal_scale)
    if not np.isfinite(qcal_scale) or qcal_scale <= 0.0:
        raise ValueError("qcal_scale must be finite and positive")
    row_delta, col_delta = center_delta_px
    # The gradient-based shift respects warped/nonuniform q maps while
    # remaining a small, explicit detector-center perturbation.
    qx_row, qx_col = _safe_gradient(qx, 0), _safe_gradient(qx, 1)
    qy_row, qy_col = _safe_gradient(qy, 0), _safe_gradient(qy, 1)
    perturbed_qx = (qx - row_delta * qx_row - col_delta * qx_col) * float(qcal_scale)
    perturbed_qy = (qy - row_delta * qy_row - col_delta * qy_col) * float(qcal_scale)
    perturbed_q = np.hypot(perturbed_qx, perturbed_qy)
    chi = np.arctan2(perturbed_qy, perturbed_qx)

    raw_mask = _array_field(qmap, ("mask", "bad_mask", "invalid_mask"))
    raw_valid = _array_field(qmap, ("valid_mask", "valid"))
    if raw_valid is None and raw_mask is None:
        valid_mask = np.ones(shape, dtype=bool)
        detector_mask = np.zeros(shape, dtype=bool)
    elif raw_valid is None:
        detector_mask = np.asarray(np.broadcast_to(raw_mask, shape), dtype=bool).copy()
        valid_mask = ~detector_mask
    elif raw_mask is None:
        valid_mask = np.asarray(np.broadcast_to(raw_valid, shape), dtype=bool).copy()
        detector_mask = ~valid_mask
    else:
        detector_mask = np.asarray(np.broadcast_to(raw_mask, shape), dtype=bool).copy()
        valid_mask = np.asarray(np.broadcast_to(raw_valid, shape), dtype=bool).copy()

    metadata_value = _field(qmap, ("metadata",), {})
    metadata = copy.deepcopy(dict(metadata_value)) if isinstance(metadata_value, Mapping) else {}
    q_unit = _field(qmap, ("q_unit", "unit"), metadata.get("q_unit", "unknown"))
    q_unit = str(q_unit or "unknown")
    fingerprint = _field(qmap, ("fingerprint", "geometry_fingerprint"), "")
    return _CanonicalPerturbedQMap(
        {
            "qx": np.asarray(perturbed_qx, dtype=float),
            "qy": np.asarray(perturbed_qy, dtype=float),
            "q": np.asarray(perturbed_q, dtype=float),
            "chi": np.asarray(chi, dtype=float),
            "angle": np.asarray(chi, dtype=float),
            "azimuth": np.asarray(chi, dtype=float),
            "chi_deg": np.degrees(chi) % 360.0,
            "mask": detector_mask,
            "valid_mask": valid_mask,
            "valid": valid_mask,
            "q_unit": q_unit,
            "unit": q_unit,
            "metadata": metadata,
            "fingerprint": fingerprint,
            "geometry_fingerprint": fingerprint,
            "shape": shape,
        }
    )


def _normalise_noise_model(options: Mapping[str, Any]) -> str:
    value = options.get("noise_model", options.get("residual_model", "empirical_residual_block"))
    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "block": "empirical_residual_block",
        "spatial_block": "empirical_residual_block",
        "residual_block": "empirical_residual_block",
        "empirical_block": "empirical_residual_block",
        "counting": "poisson",
        "counting_poisson": "poisson",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"empirical_residual_block", "poisson"}:
        raise ValueError("noise_model must be empirical_residual_block or explicit poisson")
    return normalized


def _json_number(value: Any) -> float | int | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return int(number) if number.is_integer() else number


def _source_mapping(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    keys = getattr(value, "keys", None)
    if callable(keys):
        try:
            return {str(key): value[key] for key in keys()}
        except Exception:
            pass
    return None


def _candidate_sources(result: Any) -> list[Any]:
    sources: list[Any] = [result]
    if result is None:
        return sources
    for name in ("candidate_fit", "fit", "root_result", "root", "fit_result", "analysis_result", "result"):
        child = _field(result, (name,), None)
        if child is not None and child is not result:
            sources.append(child)
    for source in tuple(sources):
        child = _field(source, ("values", "parameters", "parameter_values"), None)
        if child is not None and child is not source:
            sources.append(child)
    return sources


def _value_from_sources(sources: Sequence[Any], names: Sequence[str], default: Any = None) -> Any:
    for source in sources:
        result = _field(source, names, None)
        if result is not None:
            return result
    return default


def _normalise_code(value: Any) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def _parse_boolean_status(value: Any, field_name: str) -> tuple[bool | None, str | None]:
    if isinstance(value, (bool, np.bool_)):
        return bool(value), None
    if isinstance(value, (int, np.integer, float, np.floating)) and not isinstance(value, (bool, np.bool_)):
        number = float(value)
        if np.isfinite(number) and number in (0.0, 1.0):
            return bool(number), None
    if isinstance(value, str):
        normalized = _normalise_code(value)
        if normalized in {"true", "yes", "ok", "success", "succeeded", "converged", "pass", "passed"}:
            return True, None
        if normalized in {"false", "no", "failed", "failure", "error", "fail", "not_fitted", "not_run", "cancelled", "canceled", "invalid"}:
            return False, None
    return None, f"solver status field {field_name!r} must be boolean or an explicitly recognized status"


def _parse_success(result: Any, sources: Sequence[Any]) -> tuple[bool | None, str | None]:
    states: list[bool] = []
    for source in sources:
        for name in ("success", "ok", "valid", "status"):
            present, value = _present_field(source, name)
            if present and value is not None:
                if value is _MISSING:
                    return None, f"solver status field {name!r} could not be read"
                parsed, error = _parse_boolean_status(value, name)
                if error is not None or parsed is None:
                    return None, error
                states.append(parsed)
    if states:
        if any(states) and not all(states):
            return False, "conflicting callback solver status fields"
        return all(states), None
    return None, "solver status is missing; callback must provide a boolean success/status field"


def _structured_failure(result: Any) -> tuple[str, str]:
    """Classify only explicit failure metadata, never arbitrary prose."""

    for source in _candidate_sources(result):
        topology_flag = _field(source, ("topology_failed", "topology_failure"), None)
        if isinstance(topology_flag, (bool, np.bool_)) and bool(topology_flag):
            code = _field(source, ("topology_code", "failure_code", "code"), "topology_failed")
            code = _normalise_code(code)
            return "topology", code if code in _TOPOLOGY_FAILURE_CODES else "topology_failed"
        failure_kind = _field(source, ("failure_kind",), None)
        if failure_kind is not None:
            normalized_kind = _normalise_code(failure_kind)
            if normalized_kind in _TOPOLOGY_FAILURE_CODES:
                code = _normalise_code(_field(source, ("topology_code", "failure_code", "code"), normalized_kind))
                return "topology", code if code in _TOPOLOGY_FAILURE_CODES else "topology_failed"
            if normalized_kind in _REFIT_FAILURE_CODES:
                return "refit", normalized_kind
        for name in ("topology_code", "failure_code", "code"):
            code_value = _field(source, (name,), None)
            if code_value is None:
                continue
            code = _normalise_code(code_value)
            if code in _TOPOLOGY_FAILURE_CODES:
                return "topology", code
            if code in _REFIT_FAILURE_CODES:
                return "refit", code
        status_value = _field(source, ("status",), None)
        if status_value is not None:
            status_code = _normalise_code(status_value)
            if status_code in _TOPOLOGY_FAILURE_CODES:
                return "topology", status_code
            if status_code in _REFIT_FAILURE_CODES:
                return "refit", status_code
        flags = _field(source, ("flags",), None)
        if isinstance(flags, str):
            flags = (flags,)
        if isinstance(flags, Sequence):
            for flag in flags:
                code = _normalise_code(flag)
                if code in _TOPOLOGY_FAILURE_CODES:
                    return "topology", code
    return "refit", "refit_failed"


def _canonical_fit(result: Any) -> tuple[dict[str, float | bool] | None, str | None, str]:
    """Extract the stable five-field fit record from callback variants."""

    sources = _candidate_sources(result)
    structured_kind, structured_code = _structured_failure(result)
    if structured_kind == "topology":
        return None, str(_value_from_sources(sources, ("message", "error", "reason"), structured_code)), "topology"
    a_raw = _value_from_sources(sources, ("a", "semi_major", "major_axis", "ellipse_a"), None)
    b_raw = _value_from_sources(sources, ("b", "semi_minor", "minor_axis", "ellipse_b"), None)
    ratio_raw = _value_from_sources(sources, ("axis_ratio", "b_over_a", "ratio"), None)
    theta_deg_raw = _value_from_sources(sources, ("theta_deg", "theta_degrees", "orientation_deg", "tilt_deg"), None)
    theta_raw = _value_from_sources(sources, ("theta", "orientation", "tilt", "ellipse_theta"), None)
    success, status_error = _parse_success(result, sources)
    if status_error is not None or success is None:
        return None, status_error or "solver status is missing", "refit"
    try:
        a = float(a_raw)
        b = float(b_raw) if b_raw is not None else float(a) * float(ratio_raw)
        ratio = float(ratio_raw) if ratio_raw is not None else float(b) / float(a)
        if theta_deg_raw is not None:
            theta_deg = float(theta_deg_raw)
        elif theta_raw is not None:
            unit = str(_value_from_sources(sources, ("theta_unit",), "rad")).lower()
            theta_deg = float(theta_raw) if unit.startswith("deg") else float(np.degrees(float(theta_raw)))
        else:
            raise ValueError("theta/theta_deg missing")
    except (TypeError, ValueError, ZeroDivisionError, OverflowError) as exc:
        return None, f"candidate fit missing finite a/b/axis_ratio/theta_deg: {exc}", "refit"
    values = (a, b, ratio, theta_deg)
    if not all(np.isfinite(value) for value in values) or a <= 0.0 or b <= 0.0 or ratio <= 0.0:
        return None, "candidate fit contains non-finite or non-positive geometry", "refit"
    if not success:
        return None, str(_value_from_sources(sources, ("message", "error", "reason"), "callback reported success=False")), "refit"
    candidate = {
        "a": float(a),
        "b": float(b),
        "axis_ratio": float(ratio),
        "theta_deg": float(theta_deg),
        "success": True,
    }
    return candidate, None, "success"


def _failure_kind(result: Any, error: BaseException | str | None) -> str:
    del error  # Free-form prose is intentionally not a topology classifier.
    return _structured_failure(result)[0]


def _failure_code(result: Any, error: BaseException | str | None) -> str:
    del error
    return _structured_failure(result)[1]


def _call_options(base: Mapping[str, Any], *, stage: str, replicate: int | None, mask: np.ndarray | None, qmap: Any, perturbation: Mapping[str, Any] | None = None) -> dict[str, Any]:
    result = dict(base)
    result["resamples"] = 0
    result["stage"] = "evaluate"
    result["uncertainty_stage"] = str(stage)
    if replicate is not None:
        result["replicate"] = int(replicate)
    if mask is not None:
        mask_array = np.asarray(mask, dtype=bool)
        if mask_array.flags.writeable:
            mask_array = mask_array.copy()
        mask_view = mask_array.view()
        mask_view.flags.writeable = False
        result["mask"] = mask_view
    if qmap is not None:
        result["qmap"] = qmap
    if perturbation is not None:
        result["qmap_perturbation"] = dict(perturbation)
        if "center_delta_px" in perturbation:
            result["center_delta_px"] = list(perturbation["center_delta_px"])
        if "qcal_scale" in perturbation:
            result["qcal_scale"] = float(perturbation["qcal_scale"])
    return result


def _failure_record(index: int, *, error: BaseException | str, result: Any = None) -> dict[str, Any]:
    kind = _failure_kind(result, error)
    code = _failure_code(result, error)
    message = str(error)
    return {
        "replicate": int(index),
        "error": message,
        "failure_kind": kind,
        "failure_code": code,
        "topology_failed": kind == "topology",
        "refit_failed": kind == "refit",
    }


def _interval(values: Sequence[float]) -> list[float | None]:
    if not values:
        return [None, None]
    low, high = np.percentile(np.asarray(values, dtype=float), [2.5, 97.5], method="linear")
    return [float(low), float(high)]


def _declared_qmap_uncertainty(options: Mapping[str, Any]) -> tuple[tuple[float, float], float, dict[str, Any]]:
    center_value = options.get(
        "center_sigma_px",
        options.get("center_sigma", options.get("instrument_center_sigma_px", options.get("instrument_center_sigma", None))),
    )
    qcal_value = options.get(
        "qcal_sigma",
        options.get("qcal_sigma_fraction", options.get("q_scale_sigma", options.get("instrument_qcal_sigma", None))),
    )
    center = _as_float_pair(center_value, "center_sigma_px") if center_value is not None else (0.0, 0.0)
    qcal = float(qcal_value) if qcal_value is not None else 0.0
    if not np.isfinite(qcal) or qcal < 0.0:
        raise ValueError("qcal_sigma must be finite and non-negative")
    declared = {
        "center_sigma_px": list(center) if center_value is not None else None,
        "qcal_sigma_fraction": qcal if qcal_value is not None else None,
        "center_status": "declared" if center_value is not None else "absent",
        "qcal_status": "declared" if qcal_value is not None else "absent",
        "psf_status": "declared" if options.get("psf_sigma", None) is not None else "absent",
    }
    return center, qcal, declared


def _poisson_draw(image: np.ndarray, valid: np.ndarray, options: Mapping[str, Any], rng: np.random.Generator) -> tuple[np.ndarray, dict[str, Any]]:
    declaration = options.get("actual_counts", options.get("counts_declared", False))
    counts_raw = options.get("counts", None)
    if counts_raw is None and declaration is not None and not isinstance(declaration, (bool, np.bool_)):
        counts_raw = declaration
        declaration = True
    declared = bool(declaration) or counts_raw is not None
    if not declared:
        raise ValueError("poisson resampling requires an explicit actual_counts declaration")
    scale_raw = options.get("count_scale", options.get("poisson_scale", None))
    if counts_raw is None and scale_raw is None:
        raise ValueError("poisson resampling requires counts or a declared count_scale")
    if counts_raw is not None:
        counts = np.asarray(counts_raw, dtype=float)
        if counts.shape != image.shape:
            raise ValueError("actual counts array must match image shape")
        if np.any(~np.isfinite(counts[valid])) or np.any(counts[valid] < 0.0):
            raise ValueError("actual counts must be finite and non-negative on valid pixels")
        scale = float(scale_raw) if scale_raw is not None else 1.0
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("count_scale must be finite and positive")
    else:
        scale = float(scale_raw)
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("count_scale must be finite and positive")
        counts = np.clip(np.asarray(image, dtype=float), 0.0, None) * scale
    poisson_mean = np.where(np.isfinite(counts), np.clip(counts, 0.0, None), 0.0)
    sampled = rng.poisson(poisson_mean) / max(scale, np.finfo(float).eps)
    result = np.asarray(image, dtype=float).copy()
    result[valid] = sampled[valid]
    return result, {
        "noise_model": "poisson",
        "actual_counts_declared": True,
        "count_scale": float(scale),
        "counts_source": "supplied_array" if counts_raw is not None else "image_times_declared_scale",
    }


def resample_butterfly(
    image: Any,
    qmap: Any,
    *,
    mask: Any = None,
    q_window: Any = None,
    refit: Callable[[np.ndarray, dict[str, Any]], Any],
    options: Mapping[str, Any] | None = None,
    cancel_event: Any = None,
) -> dict[str, Any]:
    """Run empirical image-level spatial-block resampling and refitting.

    The returned intervals summarize successful callback fits only.  Failed
    topology/refit attempts remain visible in ``draws``, ``errors`` and the
    failure counters; they are never silently converted into numeric bounds.
    """

    if not callable(refit):
        raise TypeError("refit must be callable")
    base_options: dict[str, Any] = dict(options or {})
    image_array = _extract_image(image)
    shape = image_array.shape
    q = _extract_q(qmap, shape)
    base_mask = _validate_mask(mask, shape)
    base_mask |= _extract_image_mask(image, shape)
    base_mask |= ~np.isfinite(image_array)
    base_mask |= _extract_qmap_mask(qmap, shape)
    q_limits = _parse_q_window(q_window)
    row_slice, col_slice, crop_valid = _crop_for_q_window(~base_mask, q, q_limits)
    if not np.any(crop_valid):
        raise ValueError("mask and q_window leave no finite pixels for uncertainty evaluation")

    requested = _validate_integer(base_options.get("resamples", DEFAULT_RESAMPLES), "resamples", minimum=1)
    cap = _validate_integer(base_options.get("max_resamples", DEFAULT_MAX_RESAMPLES), "max_resamples", minimum=1)
    if cap > MAX_RESAMPLES:
        raise ValueError(f"max_resamples must be <= {MAX_RESAMPLES}")
    if requested > cap:
        raise ValueError(f"resamples={requested} exceeds configured max_resamples={cap}")
    seed = _validate_integer(base_options.get("seed", 0), "seed", minimum=0)
    block_shape = _parse_block_shape(base_options.get("block_shape", base_options.get("block_size", None)))
    smoothing_sigma = float(
        base_options.get(
            "smoothing_sigma",
            base_options.get("residual_smoothing_sigma", base_options.get("smoothing_sigma_px", 1.0)),
        )
    )
    if not np.isfinite(smoothing_sigma) or smoothing_sigma < 0.0:
        raise ValueError("smoothing_sigma must be finite and non-negative")
    noise_model = _normalise_noise_model(base_options)
    center_sigma, qcal_sigma, qmap_uncertainty = _declared_qmap_uncertainty(base_options)
    if qmap_uncertainty["center_status"] == "declared" or qmap_uncertainty["qcal_status"] == "declared":
        calibrated_qx, calibrated_qy = _extract_qxy(qmap, shape)
        if calibrated_qx is None or calibrated_qy is None:
            raise ValueError("declared center/q-calibration uncertainty requires qmap qx/qy arrays")
    rng = np.random.default_rng(seed)
    crop_image = image_array[row_slice, col_slice]
    crop_mask = base_mask[row_slice, col_slice]
    if noise_model == "empirical_residual_block":
        residual = _residual_field(crop_image, crop_valid & ~crop_mask, smoothing_sigma)
    else:
        residual = None

    zero_uncertainty_qmap = None
    if not any(center_sigma) and qcal_sigma == 0.0:
        # Build one read-only alias once.  Replicates then reuse it without
        # allocating or differentiating a full q-map on every draw.
        zero_uncertainty_qmap = _copy_qmap_with_perturbation(qmap, shape, (0.0, 0.0), 1.0)

    common_callback_options = dict(base_options)
    callback_mask = base_mask.copy()
    callback_mask.flags.writeable = False
    common_callback_options["mask"] = callback_mask
    common_callback_options["q_window"] = list(q_limits) if q_limits is not None else None
    common_callback_options["uncertainty_method"] = UNCERTAINTY_METHOD if noise_model == "empirical_residual_block" else "explicit_poisson_counts"
    common_callback_options["instrument_uncertainty"] = dict(qmap_uncertainty)
    common_callback_options.pop("refit", None)

    draws: list[dict[str, Any]] = []
    errors: list[dict[str, Any] | None] = []
    successful_values: dict[str, list[float]] = {"a": [], "b": [], "axis_ratio": [], "theta_deg": []}
    failure_counts = {"topology": 0, "refit": 0, "cancelled": 0}
    for replicate in range(requested):
        raise_if_cancelled(cancel_event, f"uncertainty:replicate-{replicate}")
        perturbation = {
            "center_delta_px": [
                float(rng.normal(0.0, center_sigma[0])),
                float(rng.normal(0.0, center_sigma[1])),
            ],
            "qcal_scale": (
                float(max(np.finfo(float).eps, 1.0 + rng.normal(0.0, qcal_sigma)))
                if qcal_sigma > 0.0
                else 1.0
            ),
            "center_sigma_px": list(center_sigma) if any(center_sigma) else None,
            "qcal_sigma_fraction": float(qcal_sigma) if qcal_sigma > 0.0 else None,
        }
        qmap_draw = (
            zero_uncertainty_qmap
            if zero_uncertainty_qmap is not None or (not any(center_sigma) and qcal_sigma == 0.0)
            else _copy_qmap_with_perturbation(
                qmap,
                shape,
                tuple(float(value) for value in perturbation["center_delta_px"]),
                float(perturbation["qcal_scale"]),
            )
        )
        if noise_model == "empirical_residual_block":
            assert residual is not None
            sampled_residual = _sample_block_residual(residual, crop_valid & ~crop_mask, block_shape, rng)
            perturbed = image_array.copy()
            target = perturbed[row_slice, col_slice]
            valid_crop = crop_valid & ~crop_mask
            target[valid_crop] = target[valid_crop] + sampled_residual[valid_crop]
            perturbed[row_slice, col_slice] = target
        else:
            perturbed, poisson_settings = _poisson_draw(image_array, ~base_mask, base_options, rng)
        callback_options = _call_options(
            common_callback_options,
            stage="resample",
            replicate=replicate,
            mask=callback_mask,
            qmap=qmap_draw,
            perturbation=perturbation,
        )
        callback_options["seed"] = int(seed)
        callback_options["replicate_seed"] = int(rng.integers(0, 2**32 - 1))
        callback_options["image_shape"] = tuple(shape)
        if noise_model == "poisson":
            callback_options["poisson_settings"] = poisson_settings
        try:
            raise_if_cancelled(cancel_event, f"uncertainty:before-refit-{replicate}")
            raw_result = refit(np.asarray(perturbed, dtype=float), callback_options)
            candidate, parse_error, parse_kind = _canonical_fit(raw_result)
            if candidate is None:
                failure = _failure_record(replicate, error=parse_error or "callback did not provide a canonical fit", result=raw_result)
                if parse_kind == "topology":
                    failure["failure_kind"] = "topology"
                failure_counts[failure["failure_kind"]] += 1
                errors.append(failure)
                draws.append({
                    "replicate": int(replicate),
                    "success": False,
                    "fit": None,
                    "error": failure["error"],
                    "failure_kind": failure["failure_kind"],
                    "failure_code": failure["failure_code"],
                    "topology_failed": failure["topology_failed"],
                    "refit_failed": failure["refit_failed"],
                    "center_delta_px": list(perturbation["center_delta_px"]),
                    "qcal_scale": float(perturbation["qcal_scale"]),
                })
                continue
            for key in successful_values:
                successful_values[key].append(float(candidate[key]))
            errors.append(None)
            draws.append({
                "replicate": int(replicate),
                "success": True,
                "fit": candidate,
                "error": None,
                "failure_kind": None,
                "topology_failed": False,
                "refit_failed": False,
                "center_delta_px": list(perturbation["center_delta_px"]),
                "qcal_scale": float(perturbation["qcal_scale"]),
            })
        except Exception as exc:
            # Cooperative cancellation is intentionally propagated so callers
            # can distinguish a stopped run from a failed fit.
            from .cancellation import AnalysisCancelled

            if isinstance(exc, AnalysisCancelled):
                failure_counts["cancelled"] += 1
                raise
            failure = _failure_record(replicate, error=exc)
            failure_counts[failure["failure_kind"]] += 1
            errors.append(failure)
            draws.append({
                "replicate": int(replicate),
                "success": False,
                "fit": None,
                "error": failure["error"],
                "failure_kind": failure["failure_kind"],
                "failure_code": failure["failure_code"],
                "topology_failed": failure["topology_failed"],
                "refit_failed": failure["refit_failed"],
                "center_delta_px": list(perturbation["center_delta_px"]),
                "qcal_scale": float(perturbation["qcal_scale"]),
            })

    intervals = {key: _interval(values) for key, values in successful_values.items()}
    success_count = int(sum(1 for draw in draws if draw["success"]))
    settings: dict[str, Any] = {
        "requested_resamples": int(requested),
        "resamples": int(requested),
        "max_resamples": int(cap),
        "seed": int(seed),
        "block_shape": list(block_shape),
        "smoothing_sigma": float(smoothing_sigma),
        "noise_model": noise_model,
        "q_window": list(q_limits) if q_limits is not None else None,
        "cropped_shape": [int(crop_image.shape[0]), int(crop_image.shape[1])],
        "output_shape": [int(shape[0]), int(shape[1])],
        "image_shape_preserved": True,
        "instrument_uncertainty": qmap_uncertainty,
    }
    if noise_model == "empirical_residual_block":
        settings.update({
            "residual_block_model": "spatial_blocks",
            "residual_block_status": "provisional_empirical_model",
            "correlation_preserved": True,
            "iid_pixel_bootstrap": False,
        })
    else:
        settings.update({"actual_counts_declared": True, "calibrated_counting_noise": False})
    source = {
        "label": UNCERTAINTY_METHOD if noise_model == "empirical_residual_block" else "explicit_poisson_counts",
        "method": UNCERTAINTY_METHOD if noise_model == "empirical_residual_block" else "explicit_poisson_counts",
        "interval": UNCERTAINTY_INTERVAL,
        "calibrated_confidence": False,
        "instrument_center_uncertainty": qmap_uncertainty["center_status"],
        "instrument_qcal_uncertainty": qmap_uncertainty["qcal_status"],
        "instrument_psf_uncertainty": qmap_uncertainty["psf_status"],
        "limitations": [
            "empirical central interval over successful refits",
            "topology failures are tallied and excluded from quantiles",
            "instrument and PSF uncertainty are absent unless explicitly supplied",
        ],
    }
    if "source" in base_options:
        source["caller_source"] = base_options["source"]
    result: dict[str, Any] = {
        "intervals": intervals,
        "central95": {key: list(value) for key, value in intervals.items()},
        "draws": draws,
        "draw_values": {key: list(values) for key, values in successful_values.items()},
        "success_draws": [draw["fit"] for draw in draws if draw["success"]],
        "successes": success_count,
        "n_successes": success_count,
        "n_draws": int(len(draws)),
        "failed": int(len(draws) - success_count),
        "success_fraction": float(success_count / max(1, requested)),
        "failed_topology": int(failure_counts["topology"]),
        "failed_refits": int(failure_counts["refit"]),
        "failure_counts": dict(failure_counts),
        "per_replicate_errors": errors,
        "errors": errors,
        "error_messages": [error["error"] if error is not None else None for error in errors],
        "settings": settings,
        "source": source,
        "source_label": source["label"],
        "uncertainty_source": source["label"],
        "calibrated": False,
        "coverage_calibrated": False,
        "interval_kind": UNCERTAINTY_INTERVAL,
        "shape": tuple(shape),
        "uncertainty_status": "partial" if not all(draw["success"] for draw in draws) else "complete",
    }
    sensitivity_option = base_options.get("sensitivity", None)
    if bool(base_options.get("include_sensitivity", False)) or isinstance(sensitivity_option, Mapping):
        result["sensitivity"] = evaluate_refit_sensitivity(
            image_array,
            qmap,
            refit,
            mask=callback_mask,
            q_window=q_limits,
            options=base_options,
        )
    return result


def evaluate_refit_sensitivity(
    image: Any,
    qmap: Any,
    refit: Callable[[np.ndarray, dict[str, Any]], Any],
    *,
    mask: Any = None,
    q_window: Any = None,
    options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate deterministic image-scale, smoothing and mask sensitivity.

    This diagnostic intentionally uses the same callback contract as
    :func:`resample_butterfly`, but each invocation has ``resamples=0`` and
    ``stage='evaluate'``.  It reports fit failures instead of turning them into
    a range.  Defaults are conservative; callers may pass
    ``options['sensitivity']`` with ``image_scales``, ``smoothing_sigmas`` or
    explicit ``masks`` sequences.
    """

    if not callable(refit):
        raise TypeError("refit must be callable")
    base = dict(options or {})
    image_array = _extract_image(image)
    shape = image_array.shape
    base_mask = _validate_mask(mask, shape)
    base_mask |= _extract_image_mask(image, shape)
    base_mask |= ~np.isfinite(image_array)
    sensitivity = base.get("sensitivity", {})
    # The shared butterfly recipe uses a boolean ``sensitivity`` switch;
    # treat that as the helper's default variant set.  A mapping customises
    # the three variant families below.
    if isinstance(sensitivity, (bool, np.bool_)):
        sensitivity = {}
    if not isinstance(sensitivity, Mapping):
        raise TypeError("options['sensitivity'] must be a mapping or boolean")
    scales = sensitivity.get("image_scales", base.get("image_scale_factors", (0.95, 1.05)))
    sigmas = sensitivity.get("smoothing_sigmas", base.get("smoothing_sensitivity", (0.0, 1.0)))
    masks = sensitivity.get("masks", base.get("mask_variants", None))
    scale_values = tuple(float(value) for value in scales)
    sigma_values = tuple(float(value) for value in sigmas)
    if any(not np.isfinite(value) or value <= 0.0 for value in scale_values):
        raise ValueError("image sensitivity scales must be finite and positive")
    if any(not np.isfinite(value) or value < 0.0 for value in sigma_values):
        raise ValueError("smoothing sensitivity sigmas must be finite and non-negative")
    q_limits = _parse_q_window(q_window)
    records: list[dict[str, Any]] = []
    callback_qmap = _readonly_qmap_alias(qmap, shape) if qmap is not None else None

    def run_variant(label: str, variant: str, variant_image: np.ndarray, variant_mask: np.ndarray) -> None:
        callback_options = _call_options(
            base,
            stage="sensitivity",
            replicate=None,
            mask=variant_mask,
            qmap=callback_qmap,
            perturbation={"sensitivity_variant": label, "variant_type": variant},
        )
        callback_options.update({"sensitivity_variant": label, "sensitivity_type": variant, "q_window": list(q_limits) if q_limits else None})
        try:
            raw = refit(np.asarray(variant_image, dtype=float), callback_options)
            candidate, parse_error, _kind = _canonical_fit(raw)
            if candidate is None:
                failure = _failure_record(len(records), error=parse_error or "callback did not provide a fit", result=raw)
                records.append({"label": label, "type": variant, "success": False, "fit": None, "error": failure["error"], "failure_kind": failure["failure_kind"], "failure_code": failure["failure_code"]})
            else:
                records.append({"label": label, "type": variant, "success": True, "fit": candidate, "error": None, "failure_kind": None})
        except Exception as exc:
            failure = _failure_record(len(records), error=exc)
            records.append({"label": label, "type": variant, "success": False, "fit": None, "error": failure["error"], "failure_kind": failure["failure_kind"], "failure_code": failure["failure_code"]})

    for value in scale_values:
        run_variant(f"image_scale_{value:g}", "image_scale", image_array * value, base_mask)
    for sigma in sigma_values:
        run_variant(f"smoothing_sigma_{sigma:g}", "smoothing", _smooth(_robust_fill(image_array, ~base_mask), sigma), base_mask)
    if masks is None:
        # A standard sensitivity point checks the effect of excluding the
        # outermost one-pixel border while preserving the user mask.
        border_mask = base_mask.copy()
        if border_mask.shape[0] > 2 and border_mask.shape[1] > 2:
            border_mask[[0, -1], :] = True
            border_mask[:, [0, -1]] = True
        masks_to_run = (border_mask,)
    else:
        masks_to_run = tuple(_validate_mask(value, shape) for value in masks)
    for index, variant_mask in enumerate(masks_to_run):
        run_variant(f"mask_variant_{index}", "mask", image_array, variant_mask)
    successes = sum(1 for record in records if record["success"])
    return {
        "variants": records,
        "successes": int(successes),
        "failed": int(len(records) - successes),
        "settings": {
            "image_scales": list(scale_values),
            "smoothing_sigmas": list(sigma_values),
            "mask_variant_count": len(masks_to_run),
            "q_window": list(q_limits) if q_limits is not None else None,
        },
        "source": "deterministic_refit_sensitivity_diagnostic",
        "calibrated": False,
    }


refit_sensitivity = evaluate_refit_sensitivity
assess_refit_sensitivity = evaluate_refit_sensitivity


__all__ = [
    "DEFAULT_BLOCK_SHAPE",
    "DEFAULT_MAX_RESAMPLES",
    "DEFAULT_RESAMPLES",
    "UNCERTAINTY_INTERVAL",
    "UNCERTAINTY_METHOD",
    "assess_refit_sensitivity",
    "evaluate_refit_sensitivity",
    "refit_sensitivity",
    "resample_butterfly",
]
