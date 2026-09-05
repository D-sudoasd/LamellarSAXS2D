"""Geometry, q-map, and detector-mask construction for preflight.

All path and read operations are explicit callbacks so the public facade keeps
its authorization and loader seams without introducing a dependency cycle.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
import os
from typing import Any

import numpy as np

from .validation import AnalysisDomainError, normalise_q_arrays, validate_q_coordinates

def qmap_value(value: Any, names: Sequence[str], default: Any = None) -> Any:
    if isinstance(value, Mapping):
        for name in names:
            if name in value:
                return value[name]
        return default
    for name in names:
        candidate = getattr(value, name, None)
        if candidate is not None:
            return candidate
    return default


def qmap_from_value(value: Any, shape: tuple[int, int], *, error_type: type[Exception] = ValueError) -> dict[str, Any]:
    qx = qmap_value(value, ("qx", "qx_nm_inv"))
    qy = qmap_value(value, ("qy", "qy_nm_inv"))
    if qx is None or qy is None:
        raise error_type("explicit qmap must provide qx and qy arrays")
    try:
        qx_array = np.asarray(qx, dtype=float)
        qy_array = np.asarray(qy, dtype=float)
    except (TypeError, ValueError) as exc:
        raise error_type("explicit qmap qx/qy arrays must be numeric") from exc
    if qx_array.shape != shape or qy_array.shape != shape:
        raise error_type(
            f"qmap shape must match image shape {shape!r}; got qx={qx_array.shape!r}, qy={qy_array.shape!r}"
        )
    q_value = qmap_value(value, ("q", "q_nm_inv"))
    try:
        q_array = np.hypot(qx_array, qy_array) if q_value is None else np.asarray(q_value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise error_type("explicit qmap q array must be numeric") from exc
    if q_array.shape != shape:
        raise error_type(f"qmap q shape {q_array.shape!r} does not match image shape {shape!r}")

    # The q-map contract has two mask polarities.  A positive valid mask and a
    # negative invalid mask are both authoritative when supplied; silently
    # preferring one loses detector exclusions at the preflight boundary.
    valid_value = qmap_value(value, ("valid_mask", "valid"))
    invalid_value = qmap_value(value, ("mask", "bad_mask", "invalid_mask"))
    detector_valid = np.ones(shape, dtype=bool)
    for raw, polarity, label in (
        (valid_value, True, "valid_mask"),
        (invalid_value, False, "mask"),
    ):
        if raw is None:
            continue
        try:
            array = np.asarray(raw, dtype=bool)
        except (TypeError, ValueError) as exc:
            raise error_type(f"qmap {label} must be boolean-like") from exc
        if array.shape != shape:
            raise error_type(
                f"qmap {label} shape {array.shape!r} does not match image shape {shape!r}"
            )
        detector_valid &= array if polarity else ~array
    metadata = qmap_value(value, ("metadata",), {})
    if not isinstance(metadata, Mapping):
        metadata = {}
    q_unit = qmap_value(value, ("q_unit", "unit"))
    if q_unit is None:
        q_unit = metadata.get("q_unit", metadata.get("unit"))
    if q_unit is None and any(
        qmap_value(value, (name,)) is not None
        for name in ("qx_nm_inv", "qy_nm_inv", "q_nm_inv")
    ):
        q_unit = "nm^-1"
    qx_array, qy_array, q_array, unit_info = normalise_q_arrays(
        qx_array,
        qy_array,
        q_array,
        q_unit,
    )
    try:
        validate_q_coordinates(qx_array, qy_array, q_array)
    except AnalysisDomainError as exc:
        raise error_type(f"explicit qmap coordinates are inconsistent: {exc}") from exc
    metadata = {**dict(metadata), **unit_info}
    return {
        "qx": qx_array,
        "qy": qy_array,
        "q": q_array,
        "detector_valid": detector_valid,
        **unit_info,
        "metadata": metadata,
        "source": "explicit_qmap",
    }


def build_qmap(
    shape: tuple[int, int],
    poni: Any,
    package: Path,
    records: list[dict[str, Any]],
    external_roots: Sequence[str | os.PathLike[str] | Path] | None = None,
    *,
    resolve_path: Callable[..., Path],
    inline_record: Callable[..., None],
    read_file_record: Callable[..., Any],
    error_type: type[Exception] = ValueError,
) -> dict[str, Any]:
    if poni is None:
        rows, columns = shape
        yy, xx = np.indices(shape, dtype=float)
        qx = xx - (columns - 1.0) / 2.0
        qy = yy - (rows - 1.0) / 2.0
        return {
            "qx": qx,
            "qy": qy,
            "q": np.hypot(qx, qy),
            "detector_valid": np.ones(shape, dtype=bool),
            "q_unit": "pixel-q",
            "source_q_unit": None,
            "q_conversion_factor_to_nm_inv": None,
            "metadata": {"q_unit": "pixel-q", "uncalibrated_pixel_q": True},
            "source": "pixel-q",
        }
    if isinstance(poni, Mapping) or (
        not isinstance(poni, (str, os.PathLike, Path))
        and qmap_value(poni, ("qx", "qx_nm_inv")) is not None
    ):
        inline_record(poni, "poni:qmap-in-memory", records)
        return qmap_from_value(poni, shape, error_type=error_type)
    if not isinstance(poni, (str, os.PathLike, Path)) and all(
        callable(getattr(poni, name, None)) for name in ("qArray", "center_array")
    ):
        inline_record(poni, "poni:integrator-in-memory", records)
        try:
            from .geometry import build_geometry

            geometry = build_geometry(shape, poni)
        except (ImportError, OSError, ValueError, RuntimeError) as exc:
            raise error_type(f"could not build qmap from in-memory PONI: {exc}") from exc
        return {
            "qx": np.asarray(geometry.qx, dtype=float),
            "qy": np.asarray(geometry.qy, dtype=float),
            "q": np.asarray(geometry.q, dtype=float),
            "detector_valid": np.asarray(geometry.valid_mask, dtype=bool),
            "q_unit": str(geometry.metadata.get("q_unit", "nm^-1")),
            "source_q_unit": None,
            "q_conversion_factor_to_nm_inv": 1.0,
            "metadata": dict(geometry.metadata),
            "source": "poni",
        }
    source = resolve_path(package, poni, external_roots=external_roots, label="PONI")
    if not source.exists() or not source.is_file():
        raise error_type(f"PONI file does not exist: {source}")
    try:
        from .geometry import build_geometry

        geometry = read_file_record(
            source,
            package,
            lambda: build_geometry(shape, source),
            records,
        )
    except (ImportError, OSError, ValueError, RuntimeError) as exc:
        if isinstance(exc, error_type):
            raise
        raise error_type(f"could not build qmap from PONI {source}: {exc}") from exc
    return {
        "qx": np.asarray(geometry.qx, dtype=float),
        "qy": np.asarray(geometry.qy, dtype=float),
        "q": np.asarray(geometry.q, dtype=float),
        "detector_valid": np.asarray(geometry.valid_mask, dtype=bool),
        "q_unit": str(geometry.metadata.get("q_unit", "nm^-1")),
        "source_q_unit": None,
        "q_conversion_factor_to_nm_inv": 1.0,
        "metadata": dict(geometry.metadata),
        "source": "poni",
    }


def load_mask(
    mask: Any,
    package: Path,
    shape: tuple[int, int],
    mask_frame: int | None,
    mask_dataset: str | None,
    records: list[dict[str, Any]],
    external_roots: Sequence[str | os.PathLike[str] | Path] | None = None,
    *,
    resolve_path: Callable[..., Path],
    inline_record: Callable[..., None],
    read_file_record: Callable[..., Any],
    load_image: Callable[..., Any],
    error_type: type[Exception] = ValueError,
) -> tuple[np.ndarray | None, Path | None]:
    if mask is None:
        return None, None
    source: Path | None = None
    if isinstance(mask, (str, os.PathLike, Path)):
        source = resolve_path(package, mask, external_roots=external_roots, label="mask")
        if not source.exists() or not source.is_file():
            raise error_type(f"mask file does not exist: {source}")
        try:
            loaded = read_file_record(
                source,
                package,
                lambda: load_image(source, frame=mask_frame, dataset=mask_dataset).data,
                records,
            )
        except (OSError, TypeError, ValueError, RuntimeError) as exc:
            if isinstance(exc, error_type):
                raise
            raise error_type(f"could not read mask {source}: {exc}") from exc
        array = np.asarray(loaded)
    else:
        array = np.asarray(mask)
        inline_record(array, "mask:in-memory", records)
        if array.ndim > 2:
            if mask_frame is None:
                raise error_type(
                    "in-memory mask contains multiple frames; select mask_frame explicitly"
                )
            if array.ndim != 3 or mask_frame >= array.shape[0]:
                raise error_type(
                    f"mask_frame {mask_frame} is outside in-memory mask with {array.shape[0]} frames"
                )
            array = array[int(mask_frame)]
    if array.shape != shape:
        raise error_type(f"mask shape {array.shape!r} does not match image shape {shape!r}")
    if array.dtype.kind == "O":
        raise error_type("mask cannot be object-valued")
    return np.asarray(array != 0, dtype=bool), source


def normalise_mask_convention(value: Any, *, error_type: type[Exception] = ValueError) -> str:
    convention = "0_valid_1_invalid" if value is None else str(value)
    if convention not in {"0_valid_1_invalid", "1_valid_0_invalid"}:
        raise error_type(
            "mask_convention must be '0_valid_1_invalid' or '1_valid_0_invalid'"
        )
    return convention



__all__ = [
    "build_qmap",
    "load_mask",
    "normalise_mask_convention",
    "qmap_from_value",
    "qmap_value",
]
