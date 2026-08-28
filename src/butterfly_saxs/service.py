"""Application service used by the Qt workbench.

The service is deliberately free of Qt.  It is the narrow seam between the
detector/geometry/measurement engines and the interactive workbench, so the
same preview, refinement and batch operations can be exercised from tests or
from another front end.  Values crossing this seam use the public UI spelling
(``theta_deg`` and ``lobe_angle_deg``); conversion to the core radian model is
kept here and is never hidden in a widget.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
from pathlib import Path
import re
from typing import Any

import numpy as np

from .batch import run_batch
from .geometry import build_geometry
from .intensity import (
    DEFAULT_PARAMETERS,
    default_intensity_parameters,
    double_ellipse_intensity,
    fit_intensity_model,
    parameter_values,
)
from .io import LoadedImage, load_image as read_image
from .observables import measure_observables
from .parameters import ParameterSet, ParameterSpec
from .validation import AnalysisDomain, build_analysis_domain, normalise_q_arrays


SERVICE_FLAGS = (
    "apparent_geometry_only",
    "nonunique_inverse_problem",
    "empirical_model_only",
)


# These are the workbench-facing controls for the quantitative measurement
# chain.  ``None`` is the serializable representation of an ``Auto`` q bound;
# ``max_pixels=0`` deliberately means all pixels and is normalized to
# ``None`` only at the intensity-fitting seam.
DEFAULT_ANALYSIS_SETTINGS: dict[str, Any] = {
    "q_min": None,
    "q_max": None,
    "draw_axis_deg": 90.0,
    "ridge_method": "radial_peak",
    "n_angular_bins": 180,
    "n_ridge_angles": 72,
    "n_radial_bins": 192,
    "curvature_sigma": 2.0,
    "curvature_percentile": 25.0,
    "normal_step": 1.0,
    "max_pixels": 0,
    "scales": (0.25, 0.5, 1.0),
    "seed": 0,
    "robust_loss": "soft_l1",
    "f_scale": 1.0,
    "max_nfev": 800,
}
# Public alias retained for callers that describe these as measurement
# settings rather than analysis settings.
DEFAULT_MEASUREMENT_SETTINGS = DEFAULT_ANALYSIS_SETTINGS


def _finite_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _as_public_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (int, str, bool)) or value is None:
        return value
    return value


def _json_safe(value: Any) -> Any:
    """Return a JSON-safe value, converting non-finite numbers to ``None``."""

    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (int, str, bool)) or value is None:
        return value
    return str(value)


_Q_PARAMETER_NAMES = frozenset(
    {
        "a",
        "b",
        "q",
        "background_width",
        "radial_gamma",
        "radial_sigma",
        "ridge_width",
        "spacing",
        "lamellar_spacing",
    }
)


def _q_unit(qmap: Any) -> str:
    """Read the declared q unit from either a map or its metadata."""

    unit = _read(qmap, ("q_unit", "unit"), None) if qmap is not None else None
    if unit is None:
        metadata = _read(qmap, ("metadata",), {}) if qmap is not None else {}
        unit = _read(metadata, ("q_unit", "unit"), None)
    return str(unit or "unknown")


def _display_q_unit(qmap: Any) -> str:
    """Return an honest editable-table unit for the supplied q map.

    ``a``/``b`` and radial widths are in the same coordinate system as the q
    map.  A bare q array is not evidence of a physical calibration, so it must
    never inherit the historical ``nm⁻¹`` display default.
    """

    normalized = _q_unit(qmap).strip().lower().replace(" ", "")
    if normalized in {"pixel-q", "pixel_q", "pixelq", "pixel"}:
        return "pixel-q"
    if normalized in {"1/nm", "nm^-1", "nm^−1", "nm−1", "nm-1", "nm⁻¹"}:
        return "nm⁻¹"
    if normalized in {
        "1/a",
        "a^-1",
        "a−1",
        "a-1",
        "angstrom^-1",
        "å^-1",
        "å^−1",
        "å−1",
        "å⁻¹",
    }:
        return "Å⁻¹"
    return "unknown"


def _normalise_service_qmap(qmap: Any, shape: tuple[int, int]) -> Any:
    """Canonicalize explicit service q arrays without changing mask metadata."""

    if not isinstance(qmap, Mapping):
        return qmap
    qx = _read(qmap, ("qx", "qx_nm_inv"), None)
    qy = _read(qmap, ("qy", "qy_nm_inv"), None)
    if qx is None or qy is None:
        return qmap
    qx_array = np.asarray(qx, dtype=float)
    qy_array = np.asarray(qy, dtype=float)
    if qx_array.shape != shape or qy_array.shape != shape:
        raise ValueError(f"qmap shape must match image shape {shape!r}")
    q = _read(qmap, ("q", "q_nm_inv"), None)
    q_array = np.hypot(qx_array, qy_array) if q is None else np.asarray(q, dtype=float)
    if q_array.shape != shape:
        raise ValueError(f"qmap q shape must match image shape {shape!r}")
    qx_array, qy_array, q_array, unit_info = normalise_q_arrays(
        qx_array,
        qy_array,
        q_array,
        _q_unit(qmap),
    )
    if qmap.get("q_unit") == "nm^-1" and "q_conversion_factor_to_nm_inv" in qmap:
        unit_info["source_q_unit"] = qmap.get("source_q_unit")
        unit_info["q_conversion_factor_to_nm_inv"] = qmap.get(
            "q_conversion_factor_to_nm_inv"
        )
    result = dict(qmap)
    result.update({"qx": qx_array, "qy": qy_array, "q": q_array, **unit_info})
    metadata = result.get("metadata")
    result["metadata"] = {
        **(dict(metadata) if isinstance(metadata, Mapping) else {}),
        **unit_info,
    }
    return result


def _is_q_parameter(name: Any) -> bool:
    key = str(name).strip().lower()
    return key in _Q_PARAMETER_NAMES or key.startswith("q_") or key.endswith("_q")


def _q_parameter_specs(
    specs: Mapping[str, Any],
    qmap: Any,
) -> dict[str, dict[str, Any]]:
    """Apply the q-map unit to q-valued parameter rows.

    The copy keeps the service's internal optimizer records independent from
    the UI display adapter.  This is intentionally based on the q map rather
    than on whether qx/qy arrays happen to be present.
    """

    result = deepcopy(dict(specs))
    unit = _display_q_unit(qmap)
    for name, raw in result.items():
        if not _is_q_parameter(name):
            continue
        if isinstance(raw, Mapping):
            row = raw
        else:
            row = {"value": raw}
            result[str(name)] = row
        row["unit"] = unit
    return result


def _payload_option(payload: Mapping[str, Any], names: tuple[str, ...], default: Any = None) -> Any:
    """Read a fit option from the root payload or an optional ``fit`` block."""

    for source in (payload, payload.get("fit")):
        if not isinstance(source, Mapping):
            continue
        for name in names:
            if name in source:
                return source[name]
    return default


def _analysis_payload(payload: Any) -> dict[str, Any]:
    """Collect analysis controls from the worker payload.

    New workbench requests carry an ``analysis`` mapping.  Root-level aliases
    and the older ``fit.analysis`` spelling are accepted so injected engines
    and saved projects can be migrated without changing the worker contract.
    Explicit root values win over nested values.
    """

    if not isinstance(payload, Mapping):
        analysis = getattr(payload, "analysis", None)
        return dict(analysis) if isinstance(analysis, Mapping) else {}
    result: dict[str, Any] = {}
    for source_name in ("measurement", "analysis_settings", "analysis"):
        source = payload.get(source_name)
        if isinstance(source, Mapping):
            result.update(source)
    fit = payload.get("fit")
    if isinstance(fit, Mapping):
        nested = fit.get("analysis", fit.get("measurement"))
        if isinstance(nested, Mapping):
            result = {**nested, **result}
    for name in DEFAULT_ANALYSIS_SETTINGS:
        if name in payload:
            result[name] = payload[name]
    for name in ("q_window", "q_range", "q_min", "q_max"):
        if name in payload:
            result[name] = payload[name]
    if "loss" in payload and "robust_loss" not in result:
        result["robust_loss"] = payload["loss"]
    return result


def _optional_float(value: Any, name: str) -> float | None:
    """Parse a finite float, treating blank/``Auto`` as an automatic value."""

    if value is None or (isinstance(value, str) and value.strip().lower() in {"", "auto"}):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number or Auto") from exc
    if not np.isfinite(number):
        raise ValueError(f"{name} must be a finite number or Auto")
    return number


def _validated_analysis_settings(
    settings: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Normalize and validate analysis controls independent of q-map data."""

    merged = dict(DEFAULT_ANALYSIS_SETTINGS)
    if isinstance(settings, Mapping):
        merged.update(settings)
    configured_window = merged.get("q_window", merged.get("q_range"))
    if configured_window is not None:
        if isinstance(configured_window, Mapping):
            window_min = configured_window.get(
                "min", configured_window.get("q_min", configured_window.get("low"))
            )
            window_max = configured_window.get(
                "max", configured_window.get("q_max", configured_window.get("high"))
            )
        else:
            try:
                window_min, window_max = configured_window
            except (TypeError, ValueError) as exc:
                raise ValueError("q_window must be a (min, max) pair") from exc
        if window_min is None or window_max is None:
            raise ValueError("q_window mapping must provide min/max")
        if merged.get("q_min") is None:
            merged["q_min"] = window_min
        if merged.get("q_max") is None:
            merged["q_max"] = window_max
    merged["q_min"] = _optional_float(merged.get("q_min"), "q_min")
    merged["q_max"] = _optional_float(merged.get("q_max"), "q_max")
    try:
        draw_axis = float(merged.get("draw_axis_deg", 90.0))
    except (TypeError, ValueError) as exc:
        raise ValueError("draw_axis_deg must be finite") from exc
    if not np.isfinite(draw_axis):
        raise ValueError("draw_axis_deg must be finite")
    merged["draw_axis_deg"] = draw_axis

    method = str(merged.get("ridge_method", "radial_peak")).strip().lower().replace("-", "_")
    if method == "curvature":
        method = "surface_curvature"
    if method not in {"radial_peak", "surface_curvature"}:
        raise ValueError("ridge_method must be 'radial_peak' or 'surface_curvature'")
    merged["ridge_method"] = method

    integer_rules = {
        "n_angular_bins": 8,
        "n_ridge_angles": 1,
        "n_radial_bins": 8,
    }
    for name, minimum in integer_rules.items():
        try:
            number = int(merged.get(name, DEFAULT_ANALYSIS_SETTINGS[name]))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be an integer") from exc
        if number < minimum:
            raise ValueError(f"{name} must be >= {minimum}")
        merged[name] = number

    float_rules = {
        "curvature_sigma": (0.0, False),
        "curvature_percentile": (0.0, True),
        "normal_step": (0.0, False),
    }
    for name, (minimum, include_minimum) in float_rules.items():
        try:
            number = float(merged.get(name, DEFAULT_ANALYSIS_SETTINGS[name]))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be finite") from exc
        if not np.isfinite(number) or (number < minimum if include_minimum else number <= minimum):
            comparator = ">= 0" if include_minimum else "> 0"
            raise ValueError(f"{name} must be finite and {comparator}")
        if name == "curvature_percentile" and number > 100.0:
            raise ValueError("curvature_percentile must be in [0, 100]")
        if name == "normal_step" and number > 2.0:
            raise ValueError("normal_step must be in (0, 2]")
        merged[name] = number

    try:
        max_pixels = int(merged.get("max_pixels", 0))
    except (TypeError, ValueError) as exc:
        raise ValueError("max_pixels must be a non-negative integer; 0 means all") from exc
    if max_pixels < 0:
        raise ValueError("max_pixels must be a non-negative integer; 0 means all")
    merged["max_pixels"] = max_pixels
    try:
        seed = int(merged.get("seed", 0))
    except (TypeError, ValueError) as exc:
        raise ValueError("seed must be an integer") from exc
    merged["seed"] = seed
    try:
        max_nfev = int(merged.get("max_nfev", 800))
    except (TypeError, ValueError) as exc:
        raise ValueError("max_nfev must be a positive integer") from exc
    if max_nfev <= 0:
        raise ValueError("max_nfev must be a positive integer")
    merged["max_nfev"] = max_nfev
    try:
        f_scale = float(merged.get("f_scale", 1.0))
    except (TypeError, ValueError) as exc:
        raise ValueError("f_scale must be finite and > 0") from exc
    if not np.isfinite(f_scale) or f_scale <= 0:
        raise ValueError("f_scale must be finite and > 0")
    merged["f_scale"] = f_scale
    loss = str(merged.get("robust_loss", merged.get("loss", "soft_l1"))).strip()
    if loss not in {"linear", "soft_l1", "huber", "cauchy", "arctan"}:
        raise ValueError(
            "robust_loss must be linear, soft_l1, huber, cauchy, or arctan"
        )
    merged["robust_loss"] = loss
    raw_scales = merged.get("scales", (0.25, 0.5, 1.0))
    if isinstance(raw_scales, (str, bytes)):
        raise ValueError("scales must be a sequence of positive finite numbers")
    try:
        scales = tuple(float(value) for value in raw_scales)
    except (TypeError, ValueError) as exc:
        raise ValueError("scales must be a sequence of positive finite numbers") from exc
    if not scales or any(not np.isfinite(value) or value <= 0 for value in scales):
        raise ValueError("scales must be a sequence of positive finite numbers")
    merged["scales"] = scales
    return merged


def _resolved_analysis_settings(
    settings: Mapping[str, Any] | None,
    qmap: Any,
    shape: tuple[int, int],
) -> tuple[dict[str, Any], tuple[float, float]]:
    """Validate controls and resolve independent ``Auto`` q bounds."""

    normalized = _validated_analysis_settings(settings)
    q_min, q_max = normalized["q_min"], normalized["q_max"]
    q_values = _read(qmap, ("q", "q_nm_inv", "q_map"), None)
    if q_values is None:
        qx = _read(qmap, ("qx", "qx_nm_inv"), None)
        qy = _read(qmap, ("qy", "qy_nm_inv"), None)
        if qx is None or qy is None:
            raise ValueError(f"q map has no q/qx/qy values for image shape {shape!r}")
        q_values = np.hypot(np.asarray(qx, dtype=float), np.asarray(qy, dtype=float))
    finite = np.asarray(q_values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        raise ValueError(f"q map has no finite pixels for image shape {shape!r}")
    auto_min, auto_max = float(np.min(finite)), float(np.max(finite))
    q_min = auto_min if q_min is None else float(q_min)
    q_max = auto_max if q_max is None else float(q_max)
    if not np.isfinite(q_min) or not np.isfinite(q_max) or q_max <= q_min:
        raise ValueError("q window must contain finite max > min")
    return normalized, (q_min, q_max)


def _service_analysis_domain(
    observed: np.ndarray,
    qmap: Any,
    *,
    mask: Any = None,
    analysis: Mapping[str, Any] | None = None,
    sigma: Any = None,
    weights: Any = None,
    detector_valid: Any = None,
    roi_exclusion: Any = None,
) -> tuple[AnalysisDomain, dict[str, Any]]:
    """Build the shared measurement/refinement domain for one service request."""

    settings, q_window = _resolved_analysis_settings(analysis, qmap, observed.shape)
    qx = np.asarray(_read(qmap, ("qx", "qx_nm_inv")), dtype=float)
    qy = np.asarray(_read(qmap, ("qy", "qy_nm_inv")), dtype=float)
    q = _read(qmap, ("q", "q_nm_inv", "q_map"), None)
    qmap_valid = _read(qmap, ("valid_mask", "valid"), None)
    if detector_valid is None:
        detector_valid = qmap_valid
    elif qmap_valid is not None:
        detector_valid = (
            np.asarray(detector_valid, dtype=bool)
            & np.asarray(qmap_valid, dtype=bool)
        )
    if detector_valid is None:
        qmap_mask = _read(qmap, ("mask", "invalid_mask", "bad_mask"), None)
        if qmap_mask is not None:
            detector_valid = ~np.asarray(qmap_mask, dtype=bool)
    domain = build_analysis_domain(
        observed,
        qx,
        qy,
        q=q,
        detector_valid=detector_valid,
        external_mask=mask,
        roi_exclusion=roi_exclusion,
        q_window=q_window,
        sigma=sigma,
        weights=weights,
    )
    return domain, settings


def _combined_exclusion_mask(
    detector_valid: Any = None,
    external_mask: Any = None,
    roi_exclusion: Any = None,
) -> np.ndarray | None:
    """Combine already-validated masks for display/error-result fallbacks."""

    exclusions: list[np.ndarray] = []
    if detector_valid is not None:
        exclusions.append(~np.asarray(detector_valid, dtype=bool))
    if external_mask is not None:
        exclusions.append(np.asarray(external_mask, dtype=bool))
    if roi_exclusion is not None:
        exclusions.append(np.asarray(roi_exclusion, dtype=bool))
    return np.logical_or.reduce(exclusions) if exclusions else None


def _reference_axis_deg(settings: Mapping[str, Any] | None) -> float:
    """Return the model-frame reference implied by the draw-axis control."""

    try:
        draw_axis = float(_validated_analysis_settings(settings)["draw_axis_deg"])
    except (TypeError, ValueError):
        draw_axis = float(DEFAULT_ANALYSIS_SETTINGS["draw_axis_deg"])
    return draw_axis - 90.0


def _resolve_fit_array(value: Any, name: str) -> np.ndarray:
    """Load a per-pixel sigma/weight array from memory or a data file."""

    dataset = None
    candidate = value
    if isinstance(value, Mapping):
        dataset = _read(value, ("dataset", "key"), None)
        candidate = _read(value, ("path", "file", "source", "array", "values", "data"), None)
        if candidate is None:
            raise ValueError(f"{name} mapping must contain path or array data")
    if isinstance(candidate, (str, Path)):
        loaded = read_image(candidate, dataset=dataset)
        array = np.asarray(loaded.data, dtype=float)
    else:
        try:
            array = np.asarray(candidate, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a numeric array or image path") from exc
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2-D per-pixel array, got {array.shape!r}")
    return array


def _read(source: Any, names: tuple[str, ...], default: Any = None) -> Any:
    if isinstance(source, Mapping):
        for name in names:
            if name in source:
                return source[name]
    else:
        for name in names:
            if hasattr(source, name):
                value = getattr(source, name)
                if callable(value):
                    try:
                        value = value()
                    except TypeError:
                        pass
                return value
    return default


def _frame_selectors(frame: Any) -> tuple[int | None, str | None]:
    """Resolve per-entry image selectors from a frame reference or manifest row."""

    metadata = _read(frame, ("metadata",), {})
    if not isinstance(metadata, Mapping):
        metadata = {}
    frame_value = _read(frame, ("frame", "frame_index"), None)
    if frame_value is None:
        frame_value = _read(metadata, ("frame", "frame_index"), None)
    if isinstance(frame_value, str) and not frame_value.strip():
        frame_value = None
    if frame_value is not None:
        try:
            numeric_frame = int(frame_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"frame selector must be an integer, got {frame_value!r}") from exc
        if isinstance(frame_value, (float, np.floating)) and float(frame_value) != numeric_frame:
            raise ValueError(f"frame selector must be an integer, got {frame_value!r}")
        frame_value = numeric_frame

    dataset_value = _read(frame, ("dataset", "dataset_id", "dataset_name"), None)
    if dataset_value is None:
        dataset_value = _read(metadata, ("dataset", "dataset_id", "dataset_name"), None)
    if dataset_value is not None:
        dataset_value = str(dataset_value).strip() or None
    return frame_value, dataset_value


def _is_spec_mapping(parameters: Any) -> bool:
    if not isinstance(parameters, Mapping) or not parameters:
        return False
    return any(isinstance(value, Mapping) for value in parameters.values())


def _parameter_specs(parameters: Any, fallback: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Normalize scalar or rich parameter mappings to editable row records."""

    source: Mapping[str, Any]
    spec_items = getattr(parameters, "spec_items", None)
    if callable(spec_items):
        try:
            source = dict(spec_items())
        except (TypeError, ValueError):
            source = {}
    elif isinstance(parameters, Mapping):
        source = parameters
    else:
        source = {}
    result = deepcopy(dict(fallback))
    for name, raw in source.items():
        key = str(name)
        if isinstance(raw, Mapping):
            item = dict(raw)
            if "value" not in item and "initial" in item:
                item["value"] = item["initial"]
        elif hasattr(raw, "value"):
            item = {
                "value": getattr(raw, "value", None),
                "min": getattr(raw, "min", None),
                "max": getattr(raw, "max", None),
                "vary": getattr(raw, "vary", True),
                "expr": getattr(raw, "expr", "") or "",
            }
        else:
            item = {"value": raw}
        result[key] = item
    return result


def _ui_to_core_name(name: str) -> str:
    aliases = {
        "theta_deg": "theta",
        "lobe_angle_deg": "lobe_angle",
        "angular_width_deg": "angular_width",
    }
    return aliases.get(str(name), str(name))


def _angle_scale(name: str) -> float:
    return float(np.pi / 180.0) if str(name) in {"theta_deg", "lobe_angle_deg", "angular_width_deg"} else 1.0


def _core_values(specs: Mapping[str, Mapping[str, Any]], scalar_values: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Convert UI-facing values to the core intensity model mapping."""

    values: dict[str, Any] = dict(DEFAULT_PARAMETERS)
    scalar_values = scalar_values or {}
    for name, spec in specs.items():
        raw = scalar_values.get(name, _read(spec, ("value", "val", "initial"), None))
        if raw is None:
            continue
        try:
            number = float(raw)
        except (TypeError, ValueError):
            continue
        core_name = _ui_to_core_name(name)
        if core_name in {"theta", "lobe_angle", "angular_width"} and name.endswith("_deg"):
            number *= _angle_scale(name)
        values[core_name] = number
    # Keep the useful intensity shorthand consistent with its branch values.
    if "amplitude" in values:
        if "amplitude_plus" not in scalar_values and "amplitude_plus" not in specs:
            values["amplitude_plus"] = values["amplitude"]
        if "amplitude_minus" not in scalar_values and "amplitude_minus" not in specs:
            values["amplitude_minus"] = values["amplitude"]
    return values


def _uses_default_intensity_scale(parameters: Any) -> bool:
    """Whether the editable rows still carry the untouched scale defaults."""

    values = parameter_values(parameters)
    return all(
        np.isclose(float(values.get(name, np.nan)), float(DEFAULT_PARAMETERS[name]))
        for name in ("amplitude_plus", "amplitude_minus", "background")
    )


def _has_explicit_intensity_scale(*sources: Any) -> bool:
    """Whether a caller supplied any intensity-scale parameter explicitly."""

    scale_names = {"amplitude_plus", "amplitude_minus", "background"}
    for source in sources:
        if isinstance(source, Mapping):
            if scale_names.intersection(str(name) for name in source):
                return True
            continue
        spec_items = getattr(source, "spec_items", None)
        if callable(spec_items):
            try:
                if scale_names.intersection(str(name) for name, _ in spec_items()):
                    return True
            except (TypeError, ValueError):
                continue
    return False


_EXPRESSION_IDENTIFIER = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
_DEGREE_REFERENCES = {
    "theta_deg": "(theta*180/pi)",
    "lobe_angle_deg": "(lobe_angle*180/pi)",
    "angular_width_deg": "(angular_width*180/pi)",
}


def _core_expression(expression: str, target_name: str) -> str:
    """Translate degree-labelled UI expressions into the core radian graph."""

    translated = _EXPRESSION_IDENTIFIER.sub(
        lambda match: _DEGREE_REFERENCES.get(match.group(0), match.group(0)),
        str(expression),
    )
    if str(target_name) in _DEGREE_REFERENCES:
        translated = f"({translated})*pi/180"
    return translated


def _scaled_optional(value: Any, scale: float) -> float | None:
    if value is None or value == "":
        return None
    number = float(value)
    return number * scale


def _core_parameter_set(
    specs: Mapping[str, Mapping[str, Any]],
    scalar_values: Mapping[str, Any] | None = None,
) -> ParameterSet:
    """Build the optimizer's authoritative tied/bounded parameter graph.

    UI rows use degrees for angular values.  The returned set uses radians,
    preserves fixed/free state and evaluates every expression for each trial
    vector, rather than freezing a tied row at its last displayed value.
    """

    scalar_values = scalar_values or {}
    base = default_intensity_parameters()
    definitions: dict[str, Any] = {
        name: spec.copy(name=name) for name, spec in base.spec_items()
    }
    base_values = base.resolve()
    for display_name, raw_spec in specs.items():
        if not isinstance(raw_spec, Mapping):
            raw_spec = {"value": raw_spec}
        core_name = _ui_to_core_name(display_name)
        scale = _angle_scale(display_name)
        expression = str(_read(raw_spec, ("expr", "expression", "constraint"), "") or "").strip()
        low = _scaled_optional(_read(raw_spec, ("min", "minimum", "lower", "min_value"), None), scale)
        high = _scaled_optional(_read(raw_spec, ("max", "maximum", "upper", "max_value"), None), scale)
        if expression:
            definitions[core_name] = ParameterSpec(
                value=0.0,
                min=low,
                max=high,
                vary=False,
                expr=_core_expression(expression, display_name),
                name=core_name,
            )
            continue
        raw_value = scalar_values.get(
            display_name,
            _read(raw_spec, ("value", "val", "initial"), base_values.get(core_name)),
        )
        if raw_value is None:
            raise ValueError(f"parameter {display_name!r} has no value")
        definitions[core_name] = ParameterSpec(
            value=float(raw_value) * scale,
            min=low,
            max=high,
            vary=bool(_read(raw_spec, ("vary", "free", "variable"), True)),
            name=core_name,
        )
    return ParameterSet(definitions)


def _fit_controls(specs: Mapping[str, Mapping[str, Any]]) -> tuple[dict[str, bool], dict[str, tuple[float, float]]]:
    fixed: dict[str, bool] = {}
    bounds: dict[str, tuple[float, float]] = {}
    for name, spec in specs.items():
        core_name = _ui_to_core_name(name)
        if bool(_read(spec, ("expr", "expression", "constraint"), "")) or not bool(
            _read(spec, ("vary", "free", "variable"), True)
        ):
            fixed[core_name] = True
        low = _finite_or_none(_read(spec, ("min", "minimum", "lower", "min_value"), None))
        high = _finite_or_none(_read(spec, ("max", "maximum", "upper", "max_value"), None))
        scale = _angle_scale(name)
        if low is not None or high is not None:
            # ``fit_intensity_model`` requires finite bounds only where they
            # are supplied.  Its own defaults handle open sides.
            bounds[core_name] = (
                -np.inf if low is None else low * scale,
                np.inf if high is None else high * scale,
            )
    return fixed, bounds


def _core_to_ui_value(name: str, values: Mapping[str, Any]) -> Any:
    core_name = _ui_to_core_name(name)
    value = values.get(core_name)
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    return number / _angle_scale(name) if name.endswith("_deg") else number


def _updated_specs(
    specs: Mapping[str, Mapping[str, Any]],
    values: Mapping[str, Any],
    *,
    stderr: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    result = deepcopy(dict(specs))
    for name, spec in result.items():
        if not isinstance(spec, Mapping):
            spec = {"value": spec}
            result[name] = spec
        value = _core_to_ui_value(name, values)
        if value is not None:
            spec["value"] = _as_public_scalar(value)
        spec.setdefault("unit", "degree" if name.endswith("_deg") else "")
        if stderr and _ui_to_core_name(name) in stderr:
            error = _core_to_ui_value(name, stderr)
            spec["stderr"] = _as_public_scalar(error)
        else:
            # Stderr is unknown for the robust intensity fit.  ``None`` is
            # explicit and keeps strict JSON valid; it is never a fake zero.
            spec.setdefault("stderr", None)
    return result


def _default_parameter_specs() -> dict[str, dict[str, Any]]:
    """Build UI rows from the canonical intensity ``ParameterSet``.

    The optimizer owns the authoritative defaults and ties (not this service).
    Only the display adapter changes angle fields to degree-labelled editable
    rows; all other values, bounds and the ``b=a*axis_ratio`` tie are copied.
    """

    canonical = default_intensity_parameters()
    resolved = canonical.resolve()
    result: dict[str, dict[str, Any]] = {}
    units = {
        # The service starts without a calibration map.  q-valued rows are
        # relabelled from the active q map by ``_q_parameter_specs``; keeping
        # this baseline unknown avoids inventing nm^-1 before that point.
        "a": "unknown",
        "b": "unknown",
        "radial_sigma": "unknown",
        "radial_gamma": "unknown",
        "background_width": "unknown",
        "amplitude_plus": "a.u.",
        "amplitude_minus": "a.u.",
        "background": "a.u.",
        "background_slope": "a.u.",
        "background_curvature": "a.u.",
        "background_amplitude": "a.u.",
    }
    for name, spec in canonical.spec_items():
        if name in {"theta", "lobe_angle", "angular_width", "theta_deg", "lobe_angle_deg"}:
            continue
        result[name] = {
            "value": resolved[name],
            "min": spec.min,
            "max": spec.max,
            "vary": spec.vary,
            "expr": spec.expr,
            "unit": units.get(name, ""),
            "stderr": None,
        }
    angle_fields = {
        "theta_deg": ("theta", -90.0, 90.0),
        "lobe_angle_deg": ("lobe_angle", 0.0, 180.0),
        "angular_width_deg": ("angular_width", 0.01, 90.0),
    }
    for display_name, (core_name, low, high) in angle_fields.items():
        result[display_name] = {
            "value": float(np.degrees(resolved[core_name])),
            "min": low,
            "max": high,
            "vary": canonical[core_name].vary,
            "expr": "",
            "unit": "degree",
            "stderr": None,
        }
    return result


def _public_ridges(observables: Any) -> list[dict[str, Any]]:
    ridge = _read(observables, ("ridge", "ridges"), None)
    points = _read(ridge, ("points", "observed_points"), []) if ridge is not None else []
    result: list[dict[str, Any]] = []
    for point in points or []:
        angle = _read(point, ("angle", "azimuth", "phi"), None)
        q = _read(point, ("q", "q_star", "q_position"), None)
        qx = _read(point, ("qx", "x"), None)
        qy = _read(point, ("qy", "y"), None)
        valid = bool(_read(point, ("valid",), _read(point, ("accepted",), True)))
        accepted = bool(_read(point, ("accepted",), valid))
        try:
            if (qx is None or qy is None) and angle is not None and q is not None:
                qx, qy = float(q) * np.cos(float(angle)), float(q) * np.sin(float(angle))
            q_number = _finite_or_none(q)
            qx_number = _finite_or_none(qx)
            qy_number = _finite_or_none(qy)
            angle_number = _finite_or_none(angle)
            row = {
                "qx": qx_number,
                "qy": qy_number,
                "q": q_number if q_number is not None else (
                    float(np.hypot(qx_number, qy_number))
                    if qx_number is not None and qy_number is not None
                    else None
                ),
                "theta_deg": float(np.degrees(angle_number)) if angle_number is not None else None,
                "intensity": _as_public_scalar(_read(point, ("intensity",), None)),
                "baseline": _as_public_scalar(_read(point, ("baseline",), None)),
                "snr": _as_public_scalar(_read(point, ("snr",), None)),
                "method": str(_read(point, ("method",), "observed")),
                "coverage": _as_public_scalar(_read(point, ("coverage",), None)),
                "score": _as_public_scalar(_read(point, ("score", "point_score"), None)),
                "continuity_score": _as_public_scalar(_read(point, ("continuity_score",), None)),
                "trajectory_id": _read(point, ("trajectory_id",), None),
                "branch_id": _read(point, ("branch_id", "component"), None),
                "radial_fwhm": _as_public_scalar(_read(point, ("radial_fwhm",), None)),
                "azimuthal_fwhm": _as_public_scalar(_read(point, ("azimuthal_fwhm",), None)),
                "local_q_step": _as_public_scalar(_read(point, ("local_q_step",), None)),
                "q_normal_step": _as_public_scalar(_read(point, ("q_normal_step",), None)),
                "q_scale_anisotropy": _as_public_scalar(
                    _read(point, ("q_scale_anisotropy",), None)
                ),
                "pixel_x": _as_public_scalar(_read(point, ("pixel_x",), None)),
                "pixel_y": _as_public_scalar(_read(point, ("pixel_y",), None)),
                "n_pixels": _read(point, ("n_pixels",), None),
                "flags": list(_read(point, ("flags",), ())),
                "valid": valid,
                "accepted": accepted,
                "reason": str(_read(point, ("reason",), "accepted" if accepted else "rejected")),
                "q_unit": str(_read(point, ("q_unit",), _read(ridge, ("q_unit",), "unknown")) or "unknown"),
            }
        except (TypeError, ValueError):
            continue
        result.append(_json_safe(row))
    return result


def _public_ellipse(ellipse: Any) -> dict[str, Any] | None:
    if ellipse is None:
        return None
    members = _read(ellipse, ("ellipses", "ellipse_pair"), None)
    if members is None:
        members = []
    public_members: list[dict[str, Any]] = []
    for member in members:
        a = _read(member, ("a", "major", "semi_major"), None)
        b = _read(member, ("b", "minor", "semi_minor"), None)
        theta = _read(member, ("theta", "angle", "orientation"), None)
        theta_deg = _read(member, ("theta_deg", "angle_deg", "orientation_deg"), None)
        if theta_deg is None and theta is not None:
            try:
                theta_deg = float(np.degrees(float(theta)))
            except (TypeError, ValueError):
                theta_deg = None
        center = _read(member, ("center", "centre", "origin"), (0.0, 0.0))
        try:
            center = [float(center[0]), float(center[1])]
        except (TypeError, ValueError, IndexError):
            center = [0.0, 0.0]
        row = {"a": _as_public_scalar(a), "b": _as_public_scalar(b), "theta_deg": _as_public_scalar(theta_deg), "angle_deg": _as_public_scalar(theta_deg), "center": center}
        if a is not None and b is not None:
            try:
                ratio = float(b) / float(a)
                row.update({"axis_ratio": _as_public_scalar(ratio), "ellipticity": _as_public_scalar(np.sqrt(max(0.0, 1.0 - ratio * ratio)))})
            except (TypeError, ValueError, ZeroDivisionError):
                pass
        public_members.append(_json_safe(row))
    values = _read(ellipse, ("parameter_values", "parameters"), {})
    theta = _read(ellipse, ("theta_deg",), None)
    if theta is None:
        raw_theta = _read(ellipse, ("theta",), None)
        theta = float(np.degrees(float(raw_theta))) if raw_theta is not None else None
    success = bool(_read(ellipse, ("success",), False))
    quality = _read(ellipse, ("quality",), {}) or {}
    quality_status = str(_read(quality, ("status", "engineering_status"), "") or "").upper()
    solver_status = "ok" if success else "failed"
    branch_assignment = _read(
        ellipse, ("branch_assignment", "branch_assignments"), []
    )
    residual_values = _read(ellipse, ("residuals",), [])
    if branch_assignment is None:
        branch_assignment = []
    if residual_values is None:
        residual_values = []
    result = {
        "status": solver_status,
        "solver_status": solver_status,
        "quality_status": quality_status or None,
        "quality": _json_safe(quality),
        "a": _as_public_scalar(_read(ellipse, ("a",), None)),
        "b": _as_public_scalar(_read(ellipse, ("b",), None)),
        "q_unit": str(_read(ellipse, ("q_unit",), "unknown") or "unknown"),
        "Ln_from_minor_axis_nm": _as_public_scalar(
            _read(ellipse, ("Ln_from_minor_axis_nm",), None)
        ),
        "Lz_from_draw_axis_nm": _as_public_scalar(
            _read(ellipse, ("Lz_from_draw_axis_nm",), None)
        ),
        "theta_deg": _as_public_scalar(theta),
        "angle_deg": _as_public_scalar(theta),
        "ellipticity": _as_public_scalar(_read(ellipse, ("ellipticity", "eccentricity"), None)),
        "axis_ratio": _as_public_scalar(_read(ellipse, ("axes_ratio", "axis_ratio"), None)),
        "rmse": _as_public_scalar(_read(ellipse, ("rmse", "residual_rms"), None)),
        "rss": _as_public_scalar(_read(ellipse, ("rss",), None)),
        "n_points": _read(ellipse, ("n_points", "n_data"), None),
        "success": success,
        "flags": list(_read(ellipse, ("flags",), ())),
        "ellipses": public_members,
        "parameter_values": _json_safe(values),
        "stderr": _json_safe(_read(ellipse, ("stderr",), {}) or {}),
        "condition": _as_public_scalar(
            _read(ellipse, ("condition_number", "condition"), None)
        ),
        "coverage": _json_safe(_read(ellipse, ("coverage",), {}) or {}),
        "bound_flags": _json_safe(_read(ellipse, ("bound_flags",), {}) or {}),
        "bound_status": _json_safe(_read(ellipse, ("bound_status",), {}) or {}),
        "branch_counts": _json_safe(_read(ellipse, ("branch_counts",), (0, 0))),
        "branch_assignment": _json_safe(branch_assignment),
        "residuals": _json_safe(residual_values),
        "candidate_solutions": _json_safe(
            _read(ellipse, ("candidate_solutions",), ()) or ()
        ),
        "selected_start_index": int(_read(ellipse, ("selected_start_index",), 0) or 0),
        "multistart_count": int(_read(ellipse, ("multistart_count",), 1) or 1),
    }
    return _json_safe(result)


class ButterflyAnalysisService:
    """Stateful, Qt-free service backing one workbench document."""

    DEFAULT_PARAMETER_SPECS: dict[str, dict[str, Any]] = _default_parameter_specs()

    def __init__(
        self,
        *,
        poni: str | Path | Any | None = None,
        parameters: Any = None,
        analysis_settings: Mapping[str, Any] | None = None,
    ) -> None:
        self.poni_path: str | None = None
        self._poni: Any = None
        self._loaded: LoadedImage | None = None
        self._qmap: Any = None
        self._parameter_specs = _parameter_specs(parameters, self.DEFAULT_PARAMETER_SPECS)
        self._analysis_settings = _validated_analysis_settings(analysis_settings)
        if poni is not None:
            self.set_poni(poni)

    @property
    def parameters(self) -> dict[str, dict[str, Any]]:
        return _q_parameter_specs(self._parameter_specs, self._qmap)

    @property
    def analysis_settings(self) -> dict[str, Any]:
        """Return a copy of the current quantitative-measurement controls."""

        return deepcopy(self._analysis_settings)

    def set_analysis_settings(self, settings: Mapping[str, Any] | None) -> None:
        """Merge validated analysis controls for requests without overrides."""

        merged = dict(self._analysis_settings)
        if isinstance(settings, Mapping):
            merged.update(settings)
        self._analysis_settings = _validated_analysis_settings(merged)

    @property
    def observed(self) -> np.ndarray | None:
        return None if self._loaded is None else self._loaded.data

    @property
    def qmap(self) -> Any:
        return self._qmap

    def set_parameters(self, parameters: Any) -> None:
        self._parameter_specs = _parameter_specs(parameters, self._parameter_specs)

    def set_poni(self, poni: str | Path | Any | None) -> Any:
        if poni is None:
            self.poni_path, self._poni = None, None
            return None
        from .geometry import load_poni

        candidate = load_poni(poni)
        candidate_qmap = self._qmap
        if self._loaded is not None:
            # Validate shape/rotations before changing document state.  A
            # mismatched PONI must remain an explicit error in the UI.
            candidate_qmap = build_geometry(
                self._loaded.data.shape,
                candidate,
                valid_mask=self._loaded.valid_mask,
            )
        self._poni = candidate
        self.poni_path = str(poni) if isinstance(poni, (str, Path)) else "in-memory"
        self._qmap = candidate_qmap
        return self._qmap

    def _fallback_qmap(self, shape: tuple[int, int], *, valid_mask: Any = None) -> dict[str, Any]:
        yy, xx = np.indices(shape, dtype=float)
        cx, cy = (shape[1] - 1) / 2.0, (shape[0] - 1) / 2.0
        qx, qy = xx - cx, yy - cy
        result: dict[str, Any] = {
            "qx": qx,
            "qy": qy,
            "q": np.hypot(qx, qy),
            "q_unit": "pixel-q",
            "flags": ["uncalibrated_pixel_q"],
        }
        if valid_mask is not None:
            valid = np.asarray(valid_mask, dtype=bool)
            if valid.shape != shape:
                raise ValueError(f"valid_mask shape {valid.shape} does not match image shape {shape}")
            result["valid_mask"] = valid
            result["mask"] = ~valid
        return result

    def load_image(
        self,
        path: str | Path,
        *,
        frame: int | None = None,
        dataset: str | None = None,
        valid_mask: Any | None = None,
        external_mask: Any | None = None,
        mask_frame: int | None = None,
        mask_dataset: str | None = None,
        poni: str | Path | Any | None = None,
    ) -> dict[str, Any]:
        if poni is not None:
            self.set_poni(poni)
        loaded = read_image(
            path,
            frame=frame,
            dataset=dataset,
            valid_mask=valid_mask,
            mask_frame=mask_frame,
            mask_dataset=mask_dataset,
        )
        if self._poni is not None:
            qmap = build_geometry(loaded.data.shape, self._poni, valid_mask=loaded.valid_mask)
        else:
            qmap = self._fallback_qmap(loaded.data.shape, valid_mask=loaded.valid_mask)
        self._loaded = loaded
        self._qmap = qmap
        # Keep the loaded pair local: another batch worker may replace the
        # document state before it starts optimization.
        payload = self._payload_from_loaded(loaded, qmap)
        if external_mask is not None:
            mask_value = (
                read_image(
                    external_mask,
                    frame=mask_frame,
                    dataset=mask_dataset,
                ).data
                if isinstance(external_mask, (str, Path))
                else external_mask
            )
            mask_array = np.asarray(mask_value)
            if mask_array.shape != loaded.data.shape:
                raise ValueError(
                    f"external_mask shape {mask_array.shape} does not match image shape {loaded.data.shape}"
                )
            payload["external_mask"] = np.asarray(mask_array != 0, dtype=bool)
        return payload

    read_image = load_image

    def set_observed(self, data: Any, *, qx: Any = None, qy: Any = None, qmap: Any = None, metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
        array = np.asarray(data)
        supplied_valid = _read(qmap, ("valid_mask", "valid"), None) if qmap is not None else None
        self._loaded = LoadedImage(array, metadata=dict(metadata or {}), valid_mask=supplied_valid)
        if qmap is not None:
            self._qmap = _normalise_service_qmap(qmap, array.shape)
        elif qx is not None and qy is not None:
            qx_array, qy_array = np.asarray(qx, dtype=float), np.asarray(qy, dtype=float)
            self._qmap = {"qx": qx_array, "qy": qy_array, "q": np.hypot(qx_array, qy_array), "q_unit": "provided"}
        elif self._poni is not None:
            self._qmap = build_geometry(array.shape, self._poni)
        else:
            self._qmap = self._fallback_qmap(array.shape)
        return self.current_payload()

    def _payload_from_loaded(self, loaded: LoadedImage | None, qmap: Any) -> dict[str, Any]:
        """Build a payload from one loaded image/qmap pair.

        ``load_image`` updates the document's shared state for the GUI, but a
        batch worker must retain the pair it loaded even if another worker
        updates the document before optimization starts.
        """

        if loaded is None:
            return {
                "observed": None,
                "qmap": qmap,
                "qx": None,
                "qy": None,
                "poni": self.poni_path,
                "analysis": self.analysis_settings,
            }
        valid_mask = loaded.valid_mask
        if valid_mask is None:
            valid_mask = _read(qmap, ("valid_mask", "valid"), None)
        return {
            "observed": loaded.data,
            "qx": _read(qmap, ("qx", "qx_nm_inv"), None),
            "qy": _read(qmap, ("qy", "qy_nm_inv"), None),
            "qmap": qmap,
            "valid_mask": valid_mask,
            "metadata": dict(loaded.metadata),
            "source": loaded.path,
            "poni": self.poni_path,
            "analysis": self.analysis_settings,
        }

    def current_payload(self) -> dict[str, Any]:
        return self._payload_from_loaded(self._loaded, self._qmap)

    def _state(
        self, payload: Any = None
    ) -> tuple[
        np.ndarray | None,
        Any,
        list[str],
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray | None,
    ]:
        payload = payload if isinstance(payload, Mapping) else {}
        observed = payload.get("observed", self.observed)
        if observed is None:
            return (
                None,
                payload.get("qmap", self._qmap),
                ["no_observed"],
                None,
                None,
                None,
            )
        observed = np.asarray(observed)
        qmap = payload.get("qmap", self._qmap)
        flags: list[str] = []
        if qmap is None:
            qmap = self._fallback_qmap(observed.shape)
        else:
            qmap = _normalise_service_qmap(qmap, observed.shape)
        if _display_q_unit(qmap) == "pixel-q":
            flags.append("uncalibrated_pixel_q")
        detector_masks: list[np.ndarray] = []
        valid_mask = payload.get("valid_mask")
        if valid_mask is None:
            valid_mask = _read(qmap, ("valid_mask", "valid"), None)
        # A frame payload may explicitly carry ``valid_mask=None``.  That is
        # different from omitting the field: the former means this frame has
        # no loaded-image mask and must not inherit another worker's document
        # state.  The legacy document fallback remains for payloads that do
        # not provide either image state field.
        if (
            valid_mask is None
            and "valid_mask" not in payload
            and "qmap" not in payload
            and self._loaded is not None
        ):
            valid_mask = self._loaded.valid_mask
        if valid_mask is not None:
            valid_array = np.asarray(valid_mask, dtype=bool)
            if valid_array.shape == observed.shape:
                detector_masks.append(valid_array)
            else:
                raise ValueError(
                    f"valid_mask shape {valid_array.shape} does not match image shape {observed.shape}"
                )
        qmap_valid = _read(qmap, ("valid_mask", "valid"), None)
        if qmap_valid is not None:
            qmap_valid_array = np.asarray(qmap_valid, dtype=bool)
            if qmap_valid_array.shape == observed.shape:
                detector_masks.append(qmap_valid_array)
            else:
                raise ValueError(
                    f"qmap valid_mask shape {qmap_valid_array.shape} does not match image shape {observed.shape}"
                )
        qmap_mask = _read(qmap, ("mask", "invalid_mask", "bad_mask"), None)
        if qmap_mask is not None:
            qmap_mask_array = np.asarray(qmap_mask, dtype=bool)
            if qmap_mask_array.shape == observed.shape:
                detector_masks.append(~qmap_mask_array)
            else:
                raise ValueError(
                    f"qmap mask shape {qmap_mask_array.shape} does not match image shape {observed.shape}"
                )
        detector_valid = (
            np.logical_and.reduce(detector_masks) if detector_masks else None
        )
        external_mask = payload.get("external_mask")
        if external_mask is not None:
            external_array = np.asarray(external_mask, dtype=bool)
            if external_array.shape == observed.shape:
                external_mask = external_array
            else:
                raise ValueError(
                    f"external_mask shape {external_array.shape} does not match image shape {observed.shape}"
                )
        rois = payload.get("rois", ())
        roi_exclusion = None
        if rois:
            try:
                from .masking import combine_exclusion_masks

                qx = _read(qmap, ("qx", "qx_nm_inv"), None)
                qy = _read(qmap, ("qy", "qy_nm_inv"), None)
                roi_exclusion = combine_exclusion_masks(
                    observed.shape,
                    rois=rois,
                    qx=qx,
                    qy=qy,
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid ROI exclusion specification: {exc}") from exc
        return observed, qmap, flags, external_mask, roi_exclusion, detector_valid

    def _q_window(self, qmap: Any, shape: tuple[int, int]) -> tuple[float, float]:
        q = _read(qmap, ("q", "q_nm_inv", "q_map"), None)
        if q is None:
            qx = np.asarray(_read(qmap, ("qx", "qx_nm_inv")))
            qy = np.asarray(_read(qmap, ("qy", "qy_nm_inv")))
            q = np.hypot(qx, qy)
        values = np.asarray(q, dtype=float)
        finite = values[np.isfinite(values)]
        if not finite.size:
            raise ValueError(f"q map has no finite pixels for image shape {shape!r}")
        return float(np.min(finite)), float(np.max(finite))

    def _measure(
        self,
        observed: np.ndarray,
        qmap: Any,
        flags: list[str],
        *,
        mask: Any = None,
        analysis: Mapping[str, Any] | None = None,
        analysis_domain: AnalysisDomain | None = None,
    ) -> Any:
        try:
            merged = dict(self._analysis_settings)
            if isinstance(analysis, Mapping):
                merged.update(analysis)
            if analysis_domain is None:
                analysis_domain, settings = _service_analysis_domain(
                    observed, qmap, mask=mask, analysis=merged
                )
            else:
                settings = _validated_analysis_settings(merged)
            return measure_observables(
                observed,
                qmap,
                analysis_domain.q_window,
                n_angular_bins=settings["n_angular_bins"],
                n_ridge_angles=settings["n_ridge_angles"],
                n_radial_bins=settings["n_radial_bins"],
                mask=~analysis_domain.fit_valid_mask,
                ridge_method=settings["ridge_method"],
                draw_axis_deg=settings["draw_axis_deg"],
                curvature_sigma=settings["curvature_sigma"],
                curvature_percentile=settings["curvature_percentile"],
                curvature_normal_step=settings["normal_step"],
                p4_quality_thresholds=merged.get("p4_quality_thresholds"),
            )
        except Exception as exc:
            flags.append(f"observables_failed:{type(exc).__name__}")
            flags.append(f"analysis_validation_failed:{exc}")
            return None

    def _result_mapping(self, observed: np.ndarray | None, qmap: Any, model: Any = None, residual: Any = None, *, parameters: Mapping[str, Mapping[str, Any]] | None = None, observables: Any = None, flags: Iterable[str] = (), fit: Any = None, mask: Any = None, analysis: Mapping[str, Any] | None = None, analysis_domain: AnalysisDomain | None = None) -> dict[str, Any]:
        analysis_settings = dict(self._analysis_settings)
        if isinstance(analysis, Mapping):
            analysis_settings.update(analysis)
        try:
            analysis_settings = _validated_analysis_settings(analysis_settings)
        except ValueError:
            # Measurement validation is reported through ``metrics.flags``;
            # keep the JSON/result boundary usable even for a failed request.
            analysis_settings = _json_safe(analysis_settings)
        if observed is None:
            public_parameters = _q_parameter_specs(
                parameters if parameters is not None else self._parameter_specs,
                qmap,
            )
            return {
                "observed": None,
                "model": None,
                "residual": None,
                "valid_mask": None,
                "mask": None,
                "parameters": public_parameters,
                "analysis": _json_safe(analysis_settings),
                "analysis_domain": None,
                "metrics": {
                    "rmse": None,
                    "ndata": 0,
                    "flags": list(flags) + ["no_observed"],
                },
            }
        observed_array = np.asarray(observed, dtype=float)
        model_array = (
            None
            if model is None
            else np.array(np.asarray(model, dtype=float).reshape(observed_array.shape), copy=True)
        )
        if residual is None and model_array is not None:
            residual = observed_array - model_array
        residual_array = (
            None
            if residual is None
            else np.array(
                np.asarray(residual, dtype=float).reshape(observed_array.shape),
                copy=True,
            )
        )
        ridge_rows = _public_ridges(observables) if observables is not None else []
        ellipse = _public_ellipse(_read(observables, ("ellipse",), None)) if observables is not None else None
        all_flags = list(flags) + list(_read(observables, ("flags",), ()) if observables is not None else ())
        if fit is not None:
            all_flags.extend(list(_read(fit, ("flags",), ())))
        display_valid = np.ones(observed_array.shape, dtype=bool)
        if mask is not None:
            mask_array = np.asarray(mask, dtype=bool)
            if mask_array.shape != observed_array.shape:
                raise ValueError(
                    f"mask shape {mask_array.shape} does not match image shape {observed_array.shape}"
                )
            display_valid &= ~mask_array
        valid = (
            np.asarray(analysis_domain.fit_valid_mask, dtype=bool)
            if analysis_domain is not None
            else display_valid & np.isfinite(observed_array)
        )
        # NaN is intentional for display arrays: pyqtgraph/matplotlib treat
        # it as transparent, so masked detector pixels cannot look like a
        # fitted signal.  ``valid_mask`` remains the explicit JSON/UI mask.
        if model_array is not None:
            model_array[~valid] = np.nan
        if residual_array is not None:
            residual_array[~valid] = np.nan
        finite_residual = residual_array[valid & np.isfinite(residual_array)] if residual_array is not None else np.asarray([], dtype=float)
        rmse = float(np.sqrt(np.mean(finite_residual**2))) if finite_residual.size else None
        public_parameters = _q_parameter_specs(
            parameters if parameters is not None else self._parameter_specs,
            qmap,
        )
        metrics = {
            "rmse": _as_public_scalar(_read(fit, ("rmse",), rmse) if fit is not None else rmse),
            "ndata": int(_read(fit, ("ndata",), int(valid.sum()))) if fit is not None else int(valid.sum()),
            "valid_fraction": float(valid.mean()) if valid.size else None,
            "flags": sorted(set(str(item) for item in all_flags)),
            "domain_counts": (
                None if analysis_domain is None else analysis_domain.counts
            ),
        }
        if fit is not None:
            metrics.update(
                {
                    "success": bool(_read(fit, ("success",), False)),
                    "nfev": _read(fit, ("nfev",), None),
                    "weighted_rmse": _as_public_scalar(_read(fit, ("weighted_rmse",), None)),
                    "condition_number": _as_public_scalar(_read(fit, ("condition_number",), None)),
                    "bound_flags": _json_safe(_read(fit, ("bound_flags",), {})),
                    "stderr": _json_safe(_read(fit, ("stderr",), {})),
                }
            )
        result = {
            "observed": observed_array,
            "model": model_array,
            "residual": residual_array,
            "valid_mask": valid,
            "mask": ~valid,
            "ridges": ridge_rows,
            "ridge_points": ridge_rows,
            "ellipse_fit": ellipse,
            "ellipses": [] if ellipse is None else ellipse.get("ellipses", []),
            "observables": observables,
            "parameters": public_parameters,
            "analysis": _json_safe(analysis_settings),
            "analysis_domain": (
                None if analysis_domain is None else analysis_domain.to_summary()
            ),
            "fit_valid_mask": valid,
            "sampled_valid_mask": (
                valid
                if analysis_domain is None
                else analysis_domain.sampled_valid_mask
            ),
            "metrics": metrics,
            "flags": metrics["flags"],
        }
        return result

    def preview(
        self,
        *,
        parameters: Mapping[str, Any] | None = None,
        parameter_specs: Mapping[str, Any] | None = None,
        payload: Any = None,
        analysis_settings: Mapping[str, Any] | None = None,
        **extra: Any,
    ) -> dict[str, Any]:
        payload_mapping = payload if isinstance(payload, Mapping) else {}
        (
            observed,
            qmap,
            flags,
            external_mask,
            roi_exclusion,
            detector_valid,
        ) = self._state(payload_mapping)
        analysis = dict(self._analysis_settings)
        analysis.update(_analysis_payload(payload_mapping))
        if isinstance(analysis_settings, Mapping):
            analysis.update(analysis_settings)
        for alias in ("analysis", "measurement"):
            if isinstance(extra.get(alias), Mapping):
                analysis.update(extra[alias])
        specs = _q_parameter_specs(
            _parameter_specs(parameter_specs or parameters, self._parameter_specs),
            qmap,
        )
        scalar = {str(name): _read(value, ("value",), value) for name, value in (parameters or {}).items()} if isinstance(parameters, Mapping) else {}
        values = _core_parameter_set(specs, scalar)
        if observed is None:
            return self._result_mapping(None, qmap, parameters=specs, flags=flags, analysis=analysis)
        display_mask = _combined_exclusion_mask(
            detector_valid, external_mask, roi_exclusion
        )
        qx = np.asarray(_read(qmap, ("qx", "qx_nm_inv")), dtype=float)
        qy = np.asarray(_read(qmap, ("qy", "qy_nm_inv")), dtype=float)
        model = double_ellipse_intensity(
            qx,
            qy,
            values,
            reference_axis_deg=_reference_axis_deg(analysis),
        )
        observables = None
        try:
            domain, _ = _service_analysis_domain(
                observed,
                qmap,
                mask=external_mask,
                analysis=analysis,
                detector_valid=detector_valid,
                roi_exclusion=roi_exclusion,
            )
        except ValueError as exc:
            flags.append("observables_failed:ValueError")
            flags.append(f"analysis_validation_failed:{exc}")
            domain = None
        if domain is not None:
            observables = self._measure(
                observed,
                qmap,
                flags,
                mask=display_mask,
                analysis=analysis,
                analysis_domain=domain,
            )
        return self._result_mapping(observed, qmap, model, parameters=specs, observables=observables, flags=flags, mask=display_mask, analysis=analysis, analysis_domain=domain)

    def optimize(
        self,
        *,
        parameters: Mapping[str, Any] | None = None,
        parameter_specs: Mapping[str, Any] | None = None,
        payload: Any = None,
        sigma: Any = None,
        weights: Any = None,
        max_pixels: int | None = None,
        speed_cap: int | None = None,
        analysis_settings: Mapping[str, Any] | None = None,
        **extra: Any,
    ) -> dict[str, Any]:
        payload_mapping = payload if isinstance(payload, Mapping) else {}
        (
            observed,
            qmap,
            flags,
            external_mask,
            roi_exclusion,
            detector_valid,
        ) = self._state(payload_mapping)
        analysis = dict(self._analysis_settings)
        analysis.update(_analysis_payload(payload_mapping))
        if isinstance(analysis_settings, Mapping):
            analysis.update(analysis_settings)
        for alias in ("analysis", "measurement"):
            if isinstance(extra.get(alias), Mapping):
                analysis.update(extra[alias])
        specs = _q_parameter_specs(
            _parameter_specs(parameter_specs or parameters, self._parameter_specs),
            qmap,
        )
        scalar = {str(name): _read(value, ("value",), value) for name, value in (parameters or {}).items()} if isinstance(parameters, Mapping) else {}
        values = _core_parameter_set(specs, scalar)
        if observed is None:
            return self._result_mapping(None, qmap, parameters=specs, flags=flags, analysis=analysis)
        display_mask = _combined_exclusion_mask(
            detector_valid, external_mask, roi_exclusion
        )
        frame = LoadedImage(observed)
        try:
            normalized_analysis, _ = _resolved_analysis_settings(
                analysis,
                qmap,
                observed.shape,
            )
        except ValueError as exc:
            flags.append("intensity_fit_failed:ValueError")
            flags.append(f"analysis_validation_failed:{exc}")
            return self._result_mapping(
                observed,
                qmap,
                parameters=specs,
                flags=flags,
                analysis=analysis,
                mask=display_mask,
            )
        if max_pixels is None:
            max_pixels = _payload_option(
                payload_mapping,
                ("max_pixels", "fit_max_pixels", "speed_cap"),
                normalized_analysis.get("max_pixels", speed_cap),
            )
        try:
            max_pixels = int(max_pixels) if max_pixels is not None else None
        except (TypeError, ValueError) as exc:
            flags.append(f"intensity_fit_failed:{type(exc).__name__}")
            return self._result_mapping(observed, qmap, parameters=specs, flags=flags, analysis=analysis, mask=display_mask)
        if max_pixels == 0:
            max_pixels = None
        if max_pixels is not None and max_pixels < 0:
            flags.append("intensity_fit_failed:ValueError")
            return self._result_mapping(observed, qmap, parameters=specs, flags=flags, analysis=analysis, mask=display_mask)
        try:
            sigma_source = sigma if sigma is not None else _payload_option(payload_mapping, ("sigma",))
            weights_source = weights if weights is not None else _payload_option(payload_mapping, ("weights", "weight"))
            sigma_array = None if sigma_source is None else _resolve_fit_array(sigma_source, "sigma")
            weights_array = None if weights_source is None else _resolve_fit_array(weights_source, "weights")
            fit_kwargs: dict[str, Any] = {
                "initial": values,
                # Full-pixel refinement is the service default.  Sampling is
                # opt-in through the explicit max_pixels/speed_cap payload.
                "max_pixels": max_pixels,
                "scales": normalized_analysis["scales"],
                "seed": normalized_analysis["seed"],
                "robust_loss": normalized_analysis["robust_loss"],
                "f_scale": normalized_analysis["f_scale"],
                "max_nfev": normalized_analysis["max_nfev"],
                "reference_axis_deg": _reference_axis_deg(analysis),
                "auto_scale_initial": bool(
                    analysis.get(
                        "auto_scale_initial",
                        _uses_default_intensity_scale(values)
                        and not _has_explicit_intensity_scale(parameters, parameter_specs),
                    )
                ),
            }
            if sigma_array is not None:
                fit_kwargs["sigma"] = sigma_array
            if weights_array is not None:
                fit_kwargs["weights"] = weights_array
            domain, _ = _service_analysis_domain(
                observed,
                qmap,
                mask=external_mask,
                analysis=normalized_analysis,
                sigma=sigma_array,
                weights=weights_array,
                detector_valid=detector_valid,
                roi_exclusion=roi_exclusion,
            )
            fit_mask = ~domain.fit_valid_mask
            fit_kwargs["mask"] = fit_mask if np.any(fit_mask) else None
            fit_kwargs["q_window"] = domain.q_window
            fit = fit_intensity_model(frame, qmap, **fit_kwargs)
        except Exception as exc:
            flags.append(f"intensity_fit_failed:{type(exc).__name__}")
            return self._result_mapping(observed, qmap, parameters=specs, flags=flags, analysis=analysis, mask=display_mask)
        fitted_values = parameter_values(_read(fit, ("parameters",), values))
        result_specs = _updated_specs(specs, fitted_values, stderr=_read(fit, ("stderr",), {}))
        if bool(_payload_option(payload_mapping, ("commit_parameters", "commit"), True)):
            self._parameter_specs = deepcopy(result_specs)
        sampled_indices = _read(fit, ("sampled_indices",), None)
        if sampled_indices is not None:
            domain = domain.with_sampled_indices(sampled_indices)
        observables = self._measure(
            observed,
            qmap,
            flags,
            mask=display_mask,
            analysis=analysis,
            analysis_domain=domain,
        )
        return self._result_mapping(observed, qmap, fit.model_image, observed - fit.model_image, parameters=result_specs, observables=observables, flags=flags, fit=fit, mask=display_mask, analysis=analysis, analysis_domain=domain)

    def analyze_frame(self, frame: Any, initial: Any = None, *, warm_start: bool = False, config: Any = None) -> dict[str, Any]:
        path = _read(frame, ("path", "input_path"), frame)
        frame_selector, dataset = _frame_selectors(frame)
        payload = self.load_image(path, frame=frame_selector, dataset=dataset)
        params = initial if warm_start and initial is not None else self.parameters
        if isinstance(initial, Mapping):
            params = initial.get("parameters", initial)
        config_analysis = _analysis_payload(config)
        result = self.optimize(
            parameters=params,
            parameter_specs=params if _is_spec_mapping(params) else self.parameters,
            payload=payload,
            analysis_settings=config_analysis,
        )
        result["frame"] = str(path)
        result["frame_selector"] = frame_selector
        result["dataset"] = dataset
        result["time"] = _read(frame, ("time", "timestamp"), None)
        return result

    def batch(self, *, parameters: Mapping[str, Any] | None = None, parameter_specs: Mapping[str, Any] | None = None, payload: Any = None, **_: Any) -> dict[str, Any]:
        specs = _parameter_specs(parameter_specs or parameters, self._parameter_specs)
        payload_mapping = payload if isinstance(payload, Mapping) else {}
        frames = list(payload_mapping.get("frames", ()))
        mode = str(payload_mapping.get("mode", "warm_start"))
        if not frames:
            return {"records": [], "results": [], "mode": mode, "flags": [*SERVICE_FLAGS, "no_batch_frames"]}

        # Carry the rich UI state into every frame; the core batch runner still
        # owns natural ordering, failure isolation, warm-start lineage and
        # optional checkpointing.
        original_mask = payload_mapping.get("external_mask")
        original_rois = payload_mapping.get("rois", ())
        original_analysis = _analysis_payload(payload_mapping)

        def analyze_with_state(frame: Any, initial: Any = None, *, warm_start: bool = False, config: Any = None) -> dict[str, Any]:
            del config
            path = _read(frame, ("path", "input_path"), frame)
            frame_selector, dataset = _frame_selectors(frame)
            state = self.load_image(path, frame=frame_selector, dataset=dataset)
            selected = initial if warm_start and initial is not None else specs
            if isinstance(initial, Mapping):
                selected = initial.get("parameters", initial)
            frame_payload = dict(state)
            # Keep this frame's loaded image, mask, and q-map in the worker
            # payload.  Do not let optimize() reconstruct them from the
            # service document, which another concurrent frame may replace.
            frame_payload.update(
                {
                    "observed": state.get("observed"),
                    "valid_mask": state.get("valid_mask"),
                    "qmap": state.get("qmap"),
                }
            )
            if original_mask is not None:
                frame_payload["external_mask"] = original_mask
            if original_rois:
                frame_payload["rois"] = original_rois
            if original_analysis:
                frame_payload["analysis"] = dict(original_analysis)
            frame_payload["commit_parameters"] = False
            result = self.optimize(
                parameters=selected,
                parameter_specs=selected if _is_spec_mapping(selected) else specs,
                payload=frame_payload,
            )
            result["frame"] = str(path)
            result["frame_selector"] = frame_selector
            result["dataset"] = dataset
            result["time"] = _read(frame, ("time", "timestamp"), None)
            return result

        # Each frame request is explicitly side-effect free.  Do not restore a
        # pre-batch snapshot afterwards: a cancelled/stale batch may finish
        # after the user has already committed newer parameters.
        run = run_batch(
            frames,
            analyze_with_state,
            mode=mode,
            config={"parameters": specs},
            manifest=payload_mapping.get("manifest"),
            checkpoint=payload_mapping.get("checkpoint"),
            resume=bool(payload_mapping.get("resume", False)),
        )
        records: list[dict[str, Any]] = []
        for item in run.frame_results:
            result = item.result if item.result is not None else {}
            metrics = _read(result, ("metrics",), {})
            frame_selector, dataset = _frame_selectors(item.frame)
            records.append({"frame": str(_read(item.frame, ("path",), item.frame)), "frame_selector": frame_selector, "dataset": dataset, "time": _read(item.frame, ("time",), None), "status": item.status, "rmse": _read(metrics, ("rmse",), None), "flags": _read(metrics, ("flags",), []), "parameters": _read(result, ("parameters",), {})})
        outputs: dict[str, str] = {}
        output_dir = payload_mapping.get("output_dir", payload_mapping.get("output"))
        if output_dir:
            from .export import export_batch

            outputs = {
                key: str(path)
                for key, path in export_batch(
                    run,
                    output_dir,
                    provenance={"source": "ButterflySAXS UI"},
                    force=bool(payload_mapping.get("force", False)),
                ).items()
            }
        return {
            "records": _json_safe(records),
            "results": records,
            "mode": run.mode,
            "outputs": outputs,
            "checkpoint": str(run.checkpoint) if run.checkpoint is not None else None,
            "flags": list(SERVICE_FLAGS),
        }

    def preflight(self, package: str | Path, **kwargs: Any) -> dict[str, Any]:
        """Run the same read-only preflight contract used by the CLI."""

        from .preflight import run_preflight

        if kwargs.get("poni") is None and self.poni_path not in {None, "in-memory"}:
            kwargs["poni"] = self.poni_path
        return run_preflight(package, **kwargs)


AnalysisService = ButterflyAnalysisService

__all__ = [
    "AnalysisService",
    "ButterflyAnalysisService",
    "DEFAULT_ANALYSIS_SETTINGS",
    "DEFAULT_MEASUREMENT_SETTINGS",
    "SERVICE_FLAGS",
]
