"""Shared boundary helpers for analysis and batch configuration.

The scientific modules intentionally keep their public numerical APIs small.
This module contains the non-scientific normalization rules that must be
identical when a request enters through the service, pipeline, CLI, or a
project file.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import math
from typing import Any


ELLIPSE_PRESET_DEFAULTS: dict[str, dict[str, Any]] = {
    "standard": {
        "preset": "standard",
        "axis_ratio_min": None,
        "axis_ratio_max": None,
        "a_min": None,
        "a_max": None,
        "b_min": None,
        "b_max": None,
        "theta_min_deg": None,
        "theta_max_deg": None,
        "fixed_center": False,
        "center_qx": 0.0,
        "center_qy": 0.0,
        "fixed_angle": False,
        "angle_deg": 0.0,
        "fixed_a": False,
        "fixed_axis_ratio": False,
    },
    "flat_ellipse": {
        "preset": "flat_ellipse",
        "axis_ratio_min": 0.005,
        "axis_ratio_max": 0.35,
        "a_min": None,
        "a_max": None,
        "b_min": None,
        "b_max": None,
        "theta_min_deg": 0.0,
        "theta_max_deg": 90.0,
        "fixed_center": True,
        "center_qx": 0.0,
        "center_qy": 0.0,
        "fixed_angle": False,
        "angle_deg": 0.0,
        "fixed_a": False,
        "fixed_axis_ratio": False,
    },
    "very_flat_ellipse": {
        "preset": "very_flat_ellipse",
        "axis_ratio_min": 0.005,
        "axis_ratio_max": 0.35,
        "a_min": None,
        "a_max": None,
        "b_min": None,
        "b_max": None,
        "theta_min_deg": 0.0,
        "theta_max_deg": 90.0,
        "fixed_center": True,
        "center_qx": 0.0,
        "center_qy": 0.0,
        "fixed_angle": False,
        "angle_deg": 0.0,
        "fixed_a": False,
        "fixed_axis_ratio": False,
    },
}


def canonical_ellipse_preset(value: Any) -> str:
    """Return the shared preset name used by service, CLI, and Qt."""

    preset = str(value or "standard").strip().lower().replace("-", "_")
    if preset in {"flat", "flatellipse", "butterfly", "butterfly_flat"}:
        return "flat_ellipse"
    if preset in {"very_flat", "veryflat"}:
        return "very_flat_ellipse"
    if preset not in ELLIPSE_PRESET_DEFAULTS:
        raise ValueError(
            "ellipse preset must be 'standard', 'flat_ellipse', or 'very_flat_ellipse'"
        )
    return preset


def ellipse_preset_defaults(value: Any) -> dict[str, Any]:
    """Return editable defaults without overwriting explicit user settings."""

    return deepcopy(ELLIPSE_PRESET_DEFAULTS[canonical_ellipse_preset(value)])


def resolve_analysis_settings(value: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Public normalization seam shared by service, pipeline, and CLI.

    The implementation remains lazily imported for compatibility with the
    Qt-free bootstrap path; callers depend on this public boundary rather
    than reaching into the application service's private parser.
    """

    from .analysis_config import validate_analysis_settings

    return validate_analysis_settings(value)


def normalize_ellipse_settings(value: Mapping[str, Any] | None = None) -> dict[str, Any] | None:
    """Public constrained-ellipse normalization compatibility seam."""

    from .analysis_config import normalize_ellipse_settings

    return normalize_ellipse_settings(value)


def ellipse_parameter_specs(
    value: Mapping[str, Any] | None = None,
    *,
    q_window: tuple[float, float] | None = None,
) -> dict[str, dict[str, Any]] | None:
    """Public canonical ParameterSpec graph for the measured ellipse."""

    from .analysis_config import ellipse_parameter_specs

    return ellipse_parameter_specs(value, q_window=q_window)


def deep_merge_mapping(base: Mapping[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    """Merge nested mappings without discarding a sibling configuration.

    Analysis settings are edited incrementally by the GUI and CLI.  A shallow
    ``dict.update`` would make ``--ellipse-ratio-max`` erase the user's
    preset, center, and lower bound, so mappings are merged recursively while
    scalar/list values retain normal replacement semantics.
    """

    result = {str(key): value for key, value in base.items()}
    for key, value in update.items():
        key_text = str(key)
        previous = result.get(key_text)
        if isinstance(previous, Mapping) and isinstance(value, Mapping):
            result[key_text] = deep_merge_mapping(previous, value)
        else:
            result[key_text] = value
    return result


def strict_int(value: Any, name: str, *, minimum: int | None = None) -> int:
    """Parse an integer without accepting booleans or truncating decimals."""

    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    if isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            raise ValueError(f"{name} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    # Strings such as "1.5" must not be accepted by int-like third-party
    # scalar objects after an implicit conversion.
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lstrip("+-").isdigit() is False:
            raise ValueError(f"{name} must be an integer")
    if minimum is not None and parsed < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return parsed


def canonical_q_unit(value: Any, *, aliases: tuple[str, ...] = ()) -> str:
    """Normalize common q-unit spellings while preserving pixel-q."""

    text = str(value or "").strip().lower().replace(" ", "")
    if text in {"pixel-q", "pixel_q", "pixelq", "pixel", "px"}:
        return "pixel-q"
    if text in {
        "1/nm", "nm^-1", "nm^−1", "nm−1", "nm-1", "nm⁻¹",
        "nm_inv", "nm^-1", "q_nm_inv",
    }:
        return "nm⁻¹"
    if text in {
        "1/a", "a^-1", "a−1", "a-1", "angstrom^-1", "å^-1", "å^−1",
        "å−1", "å⁻¹", "1/å", "1/angstrom", "angstrom_inv", "a_inv",
    }:
        return "Å⁻¹"
    if any(text == str(alias).strip().lower().replace(" ", "") for alias in aliases):
        return "nm⁻¹"
    return str(value) if value is not None else "unknown"


def infer_q_unit_from_keys(mapping: Mapping[str, Any]) -> str | None:
    """Infer the physical unit from explicit ``*_nm_inv`` field names."""

    keys = {str(key).casefold() for key in mapping}
    if any(key in keys for key in {"qx_nm_inv", "qy_nm_inv", "q_nm_inv"}):
        # Use the validation module's canonical token so source_q_unit
        # provenance is identical across service and pipeline seams.
        return "nm^-1"
    return None


__all__ = [
    "canonical_q_unit",
    "canonical_ellipse_preset",
    "deep_merge_mapping",
    "ELLIPSE_PRESET_DEFAULTS",
    "ellipse_preset_defaults",
    "ellipse_parameter_specs",
    "infer_q_unit_from_keys",
    "strict_int",
    "normalize_ellipse_settings",
    "resolve_analysis_settings",
]
