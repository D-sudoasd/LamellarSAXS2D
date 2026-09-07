"""Shared, Qt-free scalar and source adaptation helpers."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def _read(value: Any, names: Sequence[str], default: Any = None) -> Any:
    for name in names:
        if isinstance(value, Mapping) and name in value:
            candidate = value[name]
        elif hasattr(value, name):
            try:
                candidate = getattr(value, name)
            except Exception:  # pragma: no cover - defensive property access
                continue
        else:
            continue
        if callable(candidate):
            try:
                candidate = candidate()
            except TypeError:
                pass
        if candidate is not None:
            return candidate
    return default


def _as_mapping(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    converter = getattr(value, "to_mapping", None)
    if callable(converter):
        try:
            converted = converter(include_arrays=False)
        except TypeError:
            converted = converter()
        if isinstance(converted, Mapping):
            return converted
    converter = getattr(value, "as_dict", None)
    if callable(converter):
        converted = converter()
        if isinstance(converted, Mapping):
            return converted
    if hasattr(value, "__dict__"):
        try:
            return vars(value)
        except TypeError:
            return None
    return None


def _finite(value: Any, default: float | None = None) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return result if np.isfinite(result) else default


def _finite_positive(value: Any) -> float | None:
    result = _finite(value)
    return result if result is not None and result > 0.0 else None


def _normalise_q_unit(unit: Any) -> str:
    text = str(unit or "unknown").strip().casefold()
    return (
        text.replace(" ", "")
        .replace("⁻¹", "^-1")
        .replace("⁻", "^")
        .replace("−", "-")
        .replace("å", "a")
        .replace("Å", "a")
    )


def _q_scale_to_nm(unit: Any) -> float | None:
    normal = _normalise_q_unit(unit)
    if normal in {"nm^-1", "nm-1", "nm**-1", "1/nm"}:
        return 1.0
    if normal in {
        "a^-1",
        "a-1",
        "a**-1",
        "1/a",
        "angstrom^-1",
        "angstrom-1",
        "angstrom**-1",
        "1/angstrom",
    }:
        return 10.0
    return None


def _is_physical_unit(unit: Any) -> bool:
    return _q_scale_to_nm(unit) is not None


def _copy_metadata(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _copy_metadata(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_metadata(item) for item in value]
    if isinstance(value, tuple):
        return [_copy_metadata(item) for item in value]
    if isinstance(value, np.ndarray):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    try:
        return copy.deepcopy(value)
    except Exception:  # pragma: no cover - unusual third-party metadata
        return str(value)


__all__ = [
    "_as_mapping",
    "_copy_metadata",
    "_finite",
    "_finite_positive",
    "_is_physical_unit",
    "_normalise_q_unit",
    "_q_scale_to_nm",
    "_read",
]
