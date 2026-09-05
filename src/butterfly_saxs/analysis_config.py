"""Dependency-light canonical analysis and ellipse configuration boundary."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .settings import (
    ELLIPSE_PRESET_DEFAULTS,
    canonical_ellipse_preset,
    ellipse_preset_defaults,
    strict_int,
)

DEFAULT_ANALYSIS_SETTINGS: dict[str, Any] = {
    "q_min": None,
    "q_max": None,
    "draw_axis_deg": 90.0,
    "ridge_method": "radial_peak",
    # The peak-localisation controls are shared by the Qt, CLI and batch
    # seams.  ``None`` keeps the engine default for backwards compatibility;
    # explicit values are recorded in the result and checkpoint identity.
    "ridge_snr_threshold": 2.0,
    "ridge_min_peak_fraction": 0.0,
    "ridge_min_coverage": 0.0,
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
    # A flat ellipse is an empirical geometry preset.  It is opt-in so old
    # projects retain the historical unconstrained ridge fit, while NO50S
    # users can constrain the poorly supported short axis explicitly.
    "ellipse_preset": "standard",
    "ellipse": None,
    "ellipse_residual": "sampson",
    "ellipse_multistart": 7,
    "full2d_multistart": 1,
}

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


def _ellipse_scalar(value: Any, name: str) -> Any:
    """Read a scalar or ParameterSpec-like mapping from a config row."""

    if isinstance(value, Mapping):
        return value.get("value", value.get("val", value.get("initial")))
    return value


_ELLIPSE_PRESET_DEFAULTS: dict[str, dict[str, Any]] = ELLIPSE_PRESET_DEFAULTS

# Keep the service's historical private name as an import-compatible alias;
# the canonical preset values live in settings.py for CLI and Qt restoration.

def normalize_ellipse_settings(settings: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Normalize the shared constrained-ellipse configuration.

    The accepted mapping is intentionally usable from TOML, the Qt controls
    and Python callers.  A standard preset with no explicit constraints
    returns ``None`` so the canonical solver can continue to derive a
    deterministic data-driven initial guess.  Any flat preset or explicit
    constraint returns a complete, auditable mapping.
    """

    source = dict(settings or {})
    raw = source.get("ellipse")
    nested: dict[str, Any] = {}
    if isinstance(raw, Mapping):
        nested.update(raw)
    elif isinstance(raw, str) and raw.strip():
        source.setdefault("ellipse_preset", raw)
    aliases = {
        "preset": "ellipse_preset",
        "axis_ratio_min": "ellipse_axis_ratio_min",
        "axis_ratio_max": "ellipse_axis_ratio_max",
        "a_min": "ellipse_a_min",
        "a_max": "ellipse_a_max",
        "b_min": "ellipse_b_min",
        "b_max": "ellipse_b_max",
        "theta_min_deg": "ellipse_theta_min_deg",
        "theta_max_deg": "ellipse_theta_max_deg",
        "angle_min_deg": "ellipse_theta_min_deg",
        "angle_max_deg": "ellipse_theta_max_deg",
        "fixed_center": "ellipse_fixed_center",
        "center_qx": "ellipse_center_qx",
        "center_qy": "ellipse_center_qy",
        "fixed_angle": "ellipse_fixed_angle",
        "angle_deg": "ellipse_angle_deg",
        "fixed_a": "ellipse_fixed_a",
        "fixed_axis_ratio": "ellipse_fixed_axis_ratio",
        "residual": "ellipse_residual",
        "multistart": "ellipse_multistart",
    }
    explicit = bool(nested)
    # ``{"preset": "standard"}`` is a no-op marker commonly emitted by UI
    # persistence.  Treat it like an omitted preset so standard fitting keeps
    # its data-driven initial guess unless a real constraint is supplied.
    if set(nested).issubset({"preset"}) and str(nested.get("preset", "")).strip().lower().replace("-", "_") == "standard":
        explicit = False
    for nested_name, root_name in aliases.items():
        if nested_name not in nested and root_name in source:
            nested[nested_name] = source[root_name]
            explicit = True
    preset_raw = nested.get("preset", source.get("ellipse_preset", "standard"))
    preset = str(preset_raw or "standard").strip().lower().replace("-", "_")
    preset = canonical_ellipse_preset(preset)
    if preset != "standard":
        explicit = True
    result = ellipse_preset_defaults(preset)
    result["preset"] = preset
    result["residual"] = str(
        nested.get("residual", source.get("ellipse_residual", "sampson"))
    ).strip().lower().replace("-", "_")
    if result["residual"] in {"closest", "distance", "geometric_distance"}:
        result["residual"] = "geometric"
    if result["residual"] not in {"sampson", "geometric"}:
        raise ValueError("ellipse residual must be 'sampson' or 'geometric'")
    multistart = strict_int(
        nested.get("multistart", source.get("ellipse_multistart", 7)),
        "ellipse multistart",
        minimum=1,
    )
    result["multistart"] = multistart
    for name in (
        "axis_ratio_min",
        "axis_ratio_max",
        "a_min",
        "a_max",
        "b_min",
        "b_max",
        "theta_min_deg",
        "theta_max_deg",
    ):
        if name in nested:
            explicit = True
            result[name] = _optional_float(
                _ellipse_scalar(nested[name], name), f"ellipse {name}"
            )
        value = result.get(name)
        if value is not None and not np.isfinite(float(value)):
            raise ValueError(f"ellipse {name} must be finite or Auto")
    ratio_min = result["axis_ratio_min"]
    ratio_max = result["axis_ratio_max"]
    if ratio_min is not None and (ratio_min <= 0.0 or ratio_min > 1.0):
        raise ValueError("ellipse axis_ratio_min must be in (0, 1]")
    if ratio_max is not None and (ratio_max <= 0.0 or ratio_max > 1.0):
        raise ValueError("ellipse axis_ratio_max must be in (0, 1]")
    if ratio_min is not None and ratio_max is not None and ratio_min > ratio_max:
        raise ValueError("ellipse axis_ratio_min must not exceed axis_ratio_max")
    for low_name, high_name in (("a_min", "a_max"), ("b_min", "b_max")):
        low, high = result[low_name], result[high_name]
        if low is not None and low <= 0.0:
            raise ValueError(f"ellipse {low_name} must be > 0")
        if high is not None and high <= 0.0:
            raise ValueError(f"ellipse {high_name} must be > 0")
        if low is not None and high is not None and low > high:
            raise ValueError(f"ellipse {low_name} must not exceed {high_name}")
    theta_min, theta_max = result["theta_min_deg"], result["theta_max_deg"]
    if theta_min is not None and not -90.0 <= theta_min <= 90.0:
        raise ValueError("ellipse theta_min_deg must be in [-90, 90]")
    if theta_max is not None and not -90.0 <= theta_max <= 90.0:
        raise ValueError("ellipse theta_max_deg must be in [-90, 90]")
    if theta_min is not None and theta_max is not None and theta_min > theta_max:
        raise ValueError("ellipse theta_min_deg must not exceed theta_max_deg")
    result["fixed_center"] = bool(nested.get("fixed_center", result["fixed_center"]))
    result["fixed_angle"] = bool(nested.get("fixed_angle", result["fixed_angle"]))
    result["fixed_a"] = bool(nested.get("fixed_a", result["fixed_a"]))
    result["fixed_axis_ratio"] = bool(
        nested.get("fixed_axis_ratio", result["fixed_axis_ratio"])
    )
    for name in ("center_qx", "center_qy", "angle_deg"):
        value = nested.get(name, result[name])
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"ellipse {name} must be finite") from exc
        if not np.isfinite(value):
            raise ValueError(f"ellipse {name} must be finite")
        result[name] = value
    center = nested.get("center")
    if center is not None:
        if isinstance(center, Mapping):
            center = (center.get("qx", center.get("x")), center.get("qy", center.get("y")))
        try:
            if len(center) < 2:
                raise ValueError
            result["center_qx"] = float(center[0])
            result["center_qy"] = float(center[1])
        except (TypeError, ValueError, IndexError) as exc:
            raise ValueError("ellipse center must provide finite qx and qy") from exc
        if not np.isfinite(result["center_qx"]) or not np.isfinite(result["center_qy"]):
            raise ValueError("ellipse center must provide finite qx and qy")
        explicit = True
    # Optional starting values are passed through to the canonical solver but
    # remain separate from bounds so the export clearly distinguishes both.
    for angle_name in ("angle_deg", "theta_deg", "angle", "theta"):
        if angle_name not in nested:
            continue
        raw_angle = nested[angle_name]
        if isinstance(raw_angle, Mapping):
            if result["theta_min_deg"] is None and raw_angle.get("min") is not None:
                result["theta_min_deg"] = _optional_float(
                    raw_angle["min"], "ellipse theta_min_deg"
                )
            if result["theta_max_deg"] is None and raw_angle.get("max") is not None:
                result["theta_max_deg"] = _optional_float(
                    raw_angle["max"], "ellipse theta_max_deg"
                )
            if raw_angle.get("vary") is False:
                result["fixed_angle"] = True
            raw_angle = _ellipse_scalar(raw_angle, angle_name)
        value = _optional_float(raw_angle, "ellipse angle_deg")
        if value is None:
            raise ValueError("ellipse angle_deg must be finite")
        result["angle_deg"] = value
        explicit = True
        break
    theta_min, theta_max = result["theta_min_deg"], result["theta_max_deg"]
    if theta_min is not None and not -90.0 <= theta_min <= 90.0:
        raise ValueError("ellipse theta_min_deg must be in [-90, 90]")
    if theta_max is not None and not -90.0 <= theta_max <= 90.0:
        raise ValueError("ellipse theta_max_deg must be in [-90, 90]")
    if theta_min is not None and theta_max is not None and theta_min > theta_max:
        raise ValueError("ellipse theta_min_deg must not exceed theta_max_deg")
    for source_name, result_name in (("a", "a"), ("b", "b"), ("axis_ratio", "axis_ratio")):
        if source_name in nested:
            raw_value = nested[source_name]
            if isinstance(raw_value, Mapping):
                # A full ParameterSpec-style row is accepted as a compact
                # alternative to separate ``a_min``/``a_max`` keys.
                for bound_name in ("min", "max"):
                    target = f"{source_name}_{bound_name}"
                    if target in result and result[target] is None and bound_name in raw_value:
                        result[target] = _optional_float(
                            raw_value[bound_name], f"ellipse {target}"
                        )
                if raw_value.get("vary") is False:
                    if source_name == "a":
                        result["fixed_a"] = True
                    elif source_name == "axis_ratio":
                        result["fixed_axis_ratio"] = True
                expression_only = raw_value.get("expr", raw_value.get("expression")) and _ellipse_scalar(raw_value, source_name) is None
                raw_value = _ellipse_scalar(raw_value, source_name)
                if expression_only:
                    continue
            value = _optional_float(raw_value, f"ellipse {source_name}")
            if value is None or value <= 0.0 or (result_name == "axis_ratio" and value > 1.0):
                raise ValueError(f"ellipse {source_name} must be positive and finite")
            result[result_name] = value
            explicit = True
    # ``b`` is a derived parameter in the canonical solver.  A b-only
    # starting value therefore defines b/a; accepting an inconsistent pair
    # would silently replace the user's starting point with a different one.
    if result.get("b") is not None and result.get("a") is not None:
        derived_ratio = float(result["b"]) / float(result["a"])
        explicit_ratio = result.get("axis_ratio")
        if explicit_ratio is None:
            result["axis_ratio"] = derived_ratio
        elif not np.isclose(
            float(explicit_ratio), derived_ratio, rtol=1e-7, atol=1e-12
        ):
            raise ValueError(
                "ellipse b and axis_ratio are inconsistent; provide b/a-consistent values"
            )
    return result if explicit else None


def ellipse_parameter_specs(
    settings: Mapping[str, Any] | None,
    *,
    q_window: tuple[float, float] | None = None,
) -> dict[str, dict[str, Any]] | None:
    """Build canonical ellipse specs consumed by the measured-ridge fit."""

    normalized = normalize_ellipse_settings(settings)
    if normalized is None:
        return None
    q_mid = (
        0.5 * (float(q_window[0]) + float(q_window[1]))
        if q_window is not None
        else 1.0
    )
    a_value = float(normalized.get("a", q_mid))
    if normalized.get("b") is not None and normalized.get("axis_ratio") is None:
        if a_value <= 0:
            raise ValueError("ellipse a must be positive when deriving axis_ratio from b")
        normalized = dict(normalized)
        normalized["axis_ratio"] = float(normalized["b"]) / a_value
    ratio_min = normalized.get("axis_ratio_min")
    ratio_max = normalized.get("axis_ratio_max")
    ratio_value = float(normalized.get("axis_ratio", (
        0.5 * (ratio_min + ratio_max)
        if ratio_min is not None and ratio_max is not None
        else ratio_max if ratio_max is not None
        else ratio_min if ratio_min is not None
        else 0.7
    )))
    if ratio_min is not None:
        ratio_value = max(ratio_value, float(ratio_min))
    if ratio_max is not None:
        ratio_value = min(ratio_value, float(ratio_max))
    b_value = float(normalized.get("b", a_value * ratio_value))
    if not np.isclose(b_value, a_value * ratio_value, rtol=1e-7, atol=1e-12):
        raise ValueError("ellipse b must equal a*axis_ratio")
    specs: dict[str, dict[str, Any]] = {
        "cx": {
            "value": normalized["center_qx"],
            "vary": not normalized["fixed_center"],
        },
        "cy": {
            "value": normalized["center_qy"],
            "vary": not normalized["fixed_center"],
        },
        "a": {
            "value": a_value,
            "min": normalized["a_min"],
            "max": normalized["a_max"],
            "vary": not normalized["fixed_a"],
        },
        "axis_ratio": {
            "value": ratio_value,
            "min": ratio_min,
            "max": ratio_max,
            "vary": not normalized["fixed_axis_ratio"],
        },
        "b": {
            "value": b_value,
            "min": normalized["b_min"],
            "max": normalized["b_max"],
            "vary": False,
            "expr": "a*axis_ratio",
        },
        "theta_deg": {
            "value": normalized["angle_deg"],
            "min": normalized["theta_min_deg"],
            "max": normalized["theta_max_deg"],
            "vary": not normalized["fixed_angle"],
        },
    }
    # ``None`` bounds are omitted for cleaner ParameterSet error messages and
    # stable project/checkpoint JSON.
    return {
        name: {key: value for key, value in spec.items() if value is not None}
        for name, spec in specs.items()
    }


def validate_analysis_settings(
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
    if (
        merged["q_min"] is not None
        and merged["q_max"] is not None
        and merged["q_max"] <= merged["q_min"]
    ):
        raise ValueError("q window requires q_max greater than q_min")
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
    if method in {"azimuthal", "azimuthal_max", "angular_peak", "azimuth_peak"}:
        method = "azimuthal_peak"
    if method not in {"radial_peak", "surface_curvature", "azimuthal_peak"}:
        raise ValueError(
            "ridge_method must be 'radial_peak', 'azimuthal_peak', or 'surface_curvature'"
        )
    merged["ridge_method"] = method

    integer_rules = {
        "n_angular_bins": 8,
        "n_ridge_angles": 1,
        "n_radial_bins": 8,
    }
    for name, minimum in integer_rules.items():
        number = strict_int(
            merged.get(name, DEFAULT_ANALYSIS_SETTINGS[name]),
            name,
            minimum=minimum,
        )
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

    for name, minimum in (
        ("ridge_snr_threshold", 0.0),
        ("ridge_min_peak_fraction", 0.0),
        ("ridge_min_coverage", 0.0),
    ):
        try:
            number = float(merged.get(name, DEFAULT_ANALYSIS_SETTINGS[name]))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be finite") from exc
        if not np.isfinite(number) or number < minimum:
            raise ValueError(f"{name} must be finite and >= {minimum:g}")
        if name in {"ridge_min_peak_fraction", "ridge_min_coverage"} and number > 1.0:
            raise ValueError(f"{name} must be in [0, 1]")
        merged[name] = number

    max_pixels = strict_int(
        merged.get("max_pixels", 0),
        "max_pixels",
        minimum=0,
    )
    merged["max_pixels"] = max_pixels
    seed = strict_int(merged.get("seed", 0), "seed")
    merged["seed"] = seed
    max_nfev = strict_int(merged.get("max_nfev", 800), "max_nfev", minimum=1)
    merged["max_nfev"] = max_nfev
    full2d_multistart = strict_int(
        merged.get("full2d_multistart", 1),
        "full2d_multistart",
        minimum=1,
    )
    merged["full2d_multistart"] = full2d_multistart
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
    ellipse_settings = normalize_ellipse_settings(merged)
    merged["ellipse"] = ellipse_settings
    if ellipse_settings is not None:
        merged["ellipse_preset"] = ellipse_settings["preset"]
        merged["ellipse_residual"] = ellipse_settings["residual"]
        merged["ellipse_multistart"] = ellipse_settings["multistart"]
    return merged
