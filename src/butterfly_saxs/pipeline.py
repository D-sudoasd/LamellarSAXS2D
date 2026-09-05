"""Public analysis pipeline for butterfly-shaped 2D SAXS patterns.

The module is the deliberately thin seam shared by the command line and GUI.
Beamline readers, q-space conversion, observable extraction, ridge detection,
and full-2D fitting are loaded lazily.  That keeps this first vertical slice
usable with a NumPy fixture while allowing the specialised modules to mature
independently.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import csv
import glob
import importlib
import inspect as _inspect
import json
import math
import os
import re

import numpy as np

from .batch import BatchRunResult, FrameRef, run_batch as run_batch_frames
from .project import ProjectConfig, load_project
from .validation import (
    AnalysisDomain,
    AnalysisDomainError,
    build_analysis_domain,
    normalise_q_arrays,
    validate_q_coordinates,
)
from .cancellation import AnalysisCancelled, raise_if_cancelled
from .path_utils import filter_supported_image_paths
from .csv_utils import safe_csv_cell
from .public_ellipse import canonical_ellipse_payload


class PipelineError(RuntimeError):
    """A user-facing, recoverable pipeline error."""


def _config_value(config: Any, name: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, Mapping):
        if name in config:
            return config[name]
        for group_name in ("analysis", "project", "input", "inputs", "output"):
            group = config.get(group_name)
            if isinstance(group, Mapping) and name in group:
                return group[name]
        return default
    direct = getattr(config, name, None)
    if direct is not None:
        return direct
    # ProjectConfig stores analysis knobs in a dedicated mapping.  Looking
    # there explicitly keeps the public pipeline independent from the TOML
    # representation and avoids silently ignoring q_window/mask/ridge knobs.
    analysis = getattr(config, "analysis", None)
    if isinstance(analysis, Mapping) and name in analysis:
        return analysis[name]
    return default


def _as_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    resolver = getattr(value, "resolve", None)
    if callable(resolver):
        try:
            resolved = resolver()
        except (TypeError, ValueError):
            resolved = None
        if isinstance(resolved, Mapping):
            return dict(resolved)
    if isinstance(value, Mapping):
        return dict(value)
    for method_name in ("to_dict", "as_dict", "to_mapping"):
        method = getattr(value, method_name, None)
        if callable(method):
            try:
                result = method()
            except TypeError:
                continue
            if isinstance(result, Mapping):
                return dict(result)
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "__dict__"):
        return {key: item for key, item in vars(value).items() if not key.startswith("_")}
    return {"value": value}


def _public_angles(value: Any) -> Any:
    """Convert radians to explicit, field-specific public degree keys.

    ``theta`` is the ellipse rotation, ``angle`` is a single azimuth,
    ``angles`` is an azimuth vector, and ``chi`` is the detector azimuth.
    Keeping these namespaces separate prevents a geometric ellipse angle from
    being mistaken for a lamellar tilt or a detector coordinate.
    """

    if is_dataclass(value):
        value = _as_mapping(value)
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            text = str(key)
            if text in {"theta", "theta_rad"}:
                try:
                    result["theta_deg"] = np.degrees(item)
                    continue
                except (TypeError, ValueError):
                    pass
            if text in {"angle", "angle_rad", "azimuth", "azimuth_rad"}:
                try:
                    result["angle_deg"] = np.degrees(item)
                    continue
                except (TypeError, ValueError):
                    pass
            if text in {"angles", "angles_rad", "azimuths", "azimuths_rad"}:
                try:
                    result["angles_deg"] = np.degrees(item)
                    continue
                except (TypeError, ValueError):
                    pass
            if text in {"chi", "chi_rad"}:
                try:
                    result["chi_deg"] = np.degrees(item)
                    continue
                except (TypeError, ValueError):
                    pass
            result[text] = _public_angles(item)
        return result
    if isinstance(value, (list, tuple)):
        return type(value)(_public_angles(item) for item in value)
    return value


def _jsonable(value: Any, *, array_summary: bool = True) -> Any:
    """Convert result values to strict JSON-compatible values."""

    if isinstance(value, np.ndarray):
        if array_summary:
            finite = np.asarray(value)[np.isfinite(value)] if value.size else np.asarray([])
            summary: dict[str, Any] = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
            if finite.size:
                summary.update(min=float(np.min(finite)), max=float(np.max(finite)))
            return summary
        return value.tolist()
    if isinstance(value, np.generic):
        # Recurse so np.float32/64 NaN and infinities receive the same strict
        # JSON treatment as native Python floats.
        return _jsonable(value.item(), array_summary=array_summary)
    if isinstance(value, Path):
        return os.fspath(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item, array_summary=array_summary) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item, array_summary=array_summary) for item in value]
    if is_dataclass(value):
        return _jsonable(asdict(value), array_summary=array_summary)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _module_candidates(names: Sequence[str]) -> Iterable[Any]:
    for name in names:
        try:
            yield importlib.import_module(name)
        except (ImportError, ModuleNotFoundError):
            continue


def _find_callable(module_names: Sequence[str], function_names: Sequence[str]) -> Any | None:
    for module in _module_candidates(module_names):
        for name in function_names:
            candidate = getattr(module, name, None)
            if callable(candidate):
                return candidate
    return None


def _call_adapter(fn: Any, *, image: np.ndarray, qmap: Any = None, config: Any = None, **extra: Any) -> Any:
    """Call an optional adapter without retrying exceptions from its body.

    Older versions tried several positional forms after *any* ``TypeError``.
    That made a genuine engine bug look like a signature mismatch and could
    execute a stateful adapter more than once.  Signature inspection now
    chooses one call shape; the resulting exception is allowed to propagate.
    """

    values = {
        "image": image,
        "data": image,
        "frame": image,
        "intensity": image,
        "qmap": qmap,
        "q_map": qmap,
        "config": config,
        **extra,
    }
    try:
        signature = _inspect.signature(fn)
    except (TypeError, ValueError):
        signature = None
    if signature is None:
        # There is no safe way to distinguish a positional-only contract from
        # a body exception without a signature.  Use the documented richest
        # positional form exactly once and preserve its original exception.
        return fn(image, qmap) if qmap is not None else fn(image)

    parameters = tuple(signature.parameters.values())
    accepts_kwargs = any(
        parameter.kind == _inspect.Parameter.VAR_KEYWORD for parameter in parameters
    )
    positional_args: list[Any] = []
    keyword_args: dict[str, Any] = {}
    for parameter in parameters:
        if parameter.kind == _inspect.Parameter.VAR_POSITIONAL:
            continue
        if parameter.name not in values:
            continue
        value = values[parameter.name]
        if value is None and parameter.default is not _inspect.Parameter.empty:
            continue
        if parameter.kind == _inspect.Parameter.POSITIONAL_ONLY:
            positional_args.append(value)
        elif parameter.kind in (
            _inspect.Parameter.POSITIONAL_OR_KEYWORD,
            _inspect.Parameter.KEYWORD_ONLY,
        ):
            keyword_args[parameter.name] = value
    if accepts_kwargs:
        for name, value in values.items():
            if value is not None and name not in keyword_args:
                keyword_args[name] = value

    required = [
        parameter
        for parameter in parameters
        if parameter.default is _inspect.Parameter.empty
        and parameter.kind
        in (
            _inspect.Parameter.POSITIONAL_ONLY,
            _inspect.Parameter.POSITIONAL_OR_KEYWORD,
            _inspect.Parameter.KEYWORD_ONLY,
        )
    ]
    missing = [
        parameter.name
        for parameter in required
        if parameter.name not in values
        or (values[parameter.name] is None and parameter.default is _inspect.Parameter.empty)
    ]
    if missing:
        # Small third-party adapters often use neutral names such as ``x``
        # and ``y``.  Keep that compatibility with one arity-based call while
        # still avoiding retries after an exception from the call itself.
        if all(
            parameter.kind
            in (
                _inspect.Parameter.POSITIONAL_ONLY,
                _inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
            for parameter in required
        ) and len(required) in {1, 2}:
            positional_args = [image] if len(required) == 1 else [image, qmap]
            keyword_args = {
                name: value
                for name, value in keyword_args.items()
                if name not in {parameter.name for parameter in required}
            }
        else:
            names = ", ".join(missing)
            raise TypeError(
                f"adapter {getattr(fn, '__name__', fn)!r} has unsupported required argument(s): {names}"
            )
    # This is the sole invocation.  In particular, do not catch TypeError:
    # it may have been raised by the adapter's numerical implementation.
    return fn(*positional_args, **keyword_args)


def _coerce_array(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        array = value
    elif hasattr(value, "data") and not isinstance(value, Mapping):
        array = np.asarray(value.data)
    else:
        array = np.asarray(value)
    if array.ndim != 2:
        raise PipelineError(f"二维 SAXS 图像必须是二维数组，实际形状为 {array.shape}")
    if not np.issubdtype(array.dtype, np.number):
        raise PipelineError("二维 SAXS 图像必须包含数值强度")
    return np.asarray(array, dtype=float)


def _coerce_qmap(value: Any, shape: tuple[int, int]) -> Any:
    if value is None:
        return None

    def coerce_field(key: str, field: Any) -> np.ndarray | None:
        # Optional q-map providers commonly expose a property as ``None``.
        # Do not turn that sentinel into a scalar object array: it would pass
        # through the adapter and fail much later in the observable stage.
        if field is None:
            return None
        array = np.asarray(field)
        if array.ndim != 2 or array.shape != shape:
            raise PipelineError(
                f"qmap 的 {key} 必须是严格二维且形状为 {shape}，实际为 {array.shape}"
            )
        return array

    def normalise_mapping(result: dict[str, Any]) -> dict[str, Any]:
        qx = result.get("qx", result.get("qx_nm_inv", result.get("q_x")))
        qy = result.get("qy", result.get("qy_nm_inv", result.get("q_y")))
        radial = result.get(
            "q",
            result.get(
                "q_nm_inv",
                result.get("radius", result.get("q_abs", result.get("q_map"))),
            ),
        )
        if (qx is None) != (qy is None):
            raise PipelineError("qmap 必须同时提供 qx 和 qy")
        if qx is None:
            if radial is not None:
                raise PipelineError("二维分析不能仅使用 q/radius；qmap 必须提供 qx 和 qy")
            return result
        q = radial
        if q is None:
            q = np.hypot(np.asarray(qx, dtype=float), np.asarray(qy, dtype=float))
        metadata = result.get("metadata")
        metadata = dict(metadata) if isinstance(metadata, Mapping) else {}
        source_unit = result.get("q_unit", result.get("unit"))
        if source_unit is None:
            source_unit = metadata.get("q_unit", metadata.get("unit"))
        if source_unit is None:
            from .settings import infer_q_unit_from_keys

            source_unit = infer_q_unit_from_keys(result)
        qx_array, qy_array, q_array, unit_info = normalise_q_arrays(
            qx,
            qy,
            q,
            source_unit,
        )
        try:
            validate_q_coordinates(qx_array, qy_array, q_array)
        except AnalysisDomainError as exc:
            raise PipelineError(f"qmap 坐标不一致：{exc}") from exc
        if (
            result.get("q_unit") == "nm^-1"
            and "q_conversion_factor_to_nm_inv" in result
        ):
            unit_info["source_q_unit"] = result.get("source_q_unit")
            unit_info["q_conversion_factor_to_nm_inv"] = result.get(
                "q_conversion_factor_to_nm_inv"
            )
        result["qx"] = qx_array
        result["qy"] = qy_array
        result["q"] = q_array
        result.update(unit_info)
        metadata.update(unit_info)
        result["metadata"] = metadata
        return result

    if isinstance(value, Mapping):
        result = dict(value)
        for key in (
            "qx", "qy", "q", "radius", "qx_nm_inv", "qy_nm_inv", "q_nm_inv",
            "q_x", "q_y", "q_map", "q_abs", "theta", "angle", "azimuth",
            "phi", "azimuth_map", "chi", "chi_rad", "mask", "bad_mask",
            "invalid_mask", "valid_mask", "valid",
        ):
            if key not in result:
                continue
            array = coerce_field(key, result[key])
            if array is None:
                result.pop(key, None)
            else:
                result[key] = array
        return normalise_mapping(result)
    if isinstance(value, (tuple, list)) and len(value) >= 2:
        result = {}
        qx = coerce_field("qx", value[0])
        qy = coerce_field("qy", value[1])
        if qx is not None:
            result["qx"] = qx
        if qy is not None:
            result["qy"] = qy
        if len(value) >= 3:
            q = coerce_field("q", value[2])
            if q is not None:
                result["q"] = q
        return normalise_mapping(result)
    attrs: dict[str, Any] = {}
    aliases = {
        "qx": ("qx", "qx_nm_inv", "q_x"),
        "qy": ("qy", "qy_nm_inv", "q_y"),
        "q": ("q", "q_nm_inv", "radius", "q_abs", "q_map"),
        "theta": ("theta", "chi_rad", "angle"),
        "chi": ("chi", "chi_rad"),
        "mask": ("mask", "bad_mask", "invalid_mask"),
        "valid_mask": ("valid_mask", "valid"),
    }
    for target, names in aliases.items():
        for name in names:
            if not hasattr(value, name):
                continue
            field = getattr(value, name)
            array = coerce_field(target, field)
            if array is not None:
                attrs[target] = array
                break
    metadata = getattr(value, "metadata", None)
    if isinstance(metadata, Mapping):
        attrs["metadata"] = dict(metadata)
        if "q_unit" in metadata:
            attrs["q_unit"] = metadata["q_unit"]
        elif "unit" in metadata:
            attrs["q_unit"] = metadata["unit"]
    if "q_unit" not in attrs:
        for name in ("q_unit", "unit"):
            if not hasattr(value, name):
                continue
            unit = getattr(value, name)
            if unit is not None:
                attrs["q_unit"] = unit
                break
    if attrs:
        return normalise_mapping({"object": value, **attrs})
    array = np.asarray(value)
    if array.shape == shape + (2,):
        return normalise_mapping({"qx": array[..., 0], "qy": array[..., 1]})
    raise PipelineError("qmap 必须包含 qx/qy 数组，或形状为 (高, 宽, 2) 的数组")


def _qmap_arrays(qmap: Any, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    qmap = _coerce_qmap(qmap, shape)
    if qmap is None:
        raise PipelineError("缺少 qmap")
    qx = qmap.get("qx", qmap.get("qx_nm_inv"))
    qy = qmap.get("qy", qmap.get("qy_nm_inv"))
    if qx is None or qy is None:
        raise PipelineError("二维分析的 qmap 必须同时提供 qx 和 qy")
    qx = np.asarray(qx, dtype=float)
    qy = np.asarray(qy, dtype=float)
    if qx.shape != shape or qy.shape != shape:
        raise PipelineError(f"qmap 与图像形状不一致：qx={qx.shape}, qy={qy.shape}, image={shape}")
    q = qmap.get("q", qmap.get("q_nm_inv"))
    if q is None:
        q = np.hypot(qx, qy)
    q = np.asarray(q, dtype=float)
    if q.shape != shape:
        raise PipelineError(f"qmap 的 q 数组形状不一致：{q.shape} != {shape}")
    return qx, qy, q


@dataclass
class _FrameBundle:
    """Internal read contract retaining both positive and negative masks."""

    frame: Any
    image: np.ndarray
    metadata: dict[str, Any]
    qmap: Any | None
    valid_mask: np.ndarray | None = None
    external_mask: np.ndarray | None = None


def _combine_valid_masks(
    shape: tuple[int, int],
    *,
    valid_masks: Iterable[Any] = (),
    masks: Iterable[Any] = (),
    frame: int | None = None,
    dataset: str | None = None,
) -> np.ndarray | None:
    """Combine positive and negative mask conventions without overwriting.

    Frame sources and project configuration can each provide both forms.  The
    IO layer is the single source of truth for path loading and exact-shape
    validation; this adapter only applies the logical intersection/union over
    all supplied values.
    """

    from .io import combine_masks, load_image

    combined: np.ndarray | None = None
    try:
        for value in valid_masks:
            if value is None:
                continue
            raw = (
                load_image(value, frame=frame, dataset=dataset).data
                if isinstance(value, (str, os.PathLike, Path))
                else value
            )
            current = combine_masks(shape, valid_mask=raw)
            if current is not None:
                combined = current if combined is None else (combined & current)
        for value in masks:
            if value is None:
                continue
            current = combine_masks(shape, external_mask=value)
            if current is not None:
                combined = current if combined is None else (combined & current)
    except Exception as exc:  # noqa: BLE001 - normalize source/config errors
        raise PipelineError(f"掩膜无法与图像形状 {shape} 合并：{exc}") from exc
    return combined


def _combine_external_masks(
    shape: tuple[int, int],
    masks: Iterable[Any] = (),
    *,
    frame: int | None = None,
    dataset: str | None = None,
) -> np.ndarray | None:
    """Load and OR negative-polarity masks without mixing detector validity."""

    from .io import combine_masks, load_image

    combined: np.ndarray | None = None
    try:
        for value in masks:
            if value is None:
                continue
            raw = (
                load_image(value, frame=frame, dataset=dataset).data
                if isinstance(value, (str, os.PathLike, Path))
                else value
            )
            valid = combine_masks(shape, external_mask=raw)
            if valid is not None:
                current = ~valid
                combined = current if combined is None else (combined | current)
    except Exception as exc:  # noqa: BLE001 - normalize mask boundary errors
        raise PipelineError(f"外部掩膜无法与图像形状 {shape} 合并：{exc}") from exc
    return combined


def _merge_qmap_masks(
    qmap: Mapping[str, Any],
    shape: tuple[int, int],
    inherited_valid_mask: Any = None,
) -> tuple[dict[str, Any], np.ndarray | None]:
    """Attach source validity to a q-map while retaining q-map mask polarity."""

    result = dict(qmap)
    qmap_valid = [result.get(key) for key in ("valid_mask", "valid") if result.get(key) is not None]
    qmap_masks = [
        result.get(key)
        for key in ("mask", "bad_mask", "invalid_mask")
        if result.get(key) is not None
    ]
    combined = _combine_valid_masks(
        shape,
        valid_masks=(inherited_valid_mask, *qmap_valid),
        masks=qmap_masks,
    )
    if combined is not None:
        # Keep an explicit q-map ``mask`` as the q-map-local negative mask;
        # the combined positive field carries source/config validity too.
        result["valid_mask"] = combined
    return result, combined


def _loaded_frame(
    data: Any,
    *,
    metadata: Mapping[str, Any] | None = None,
    source: Any = None,
    valid_mask: Any = None,
    external_mask: Any = None,
) -> Any:
    """Construct the canonical ``io.LoadedImage`` without importing it early."""

    # ``LoadedImage`` and ``combine_masks`` are part of the core input
    # contract.  Do not replace them with a permissive duck-typed fallback:
    # that would turn a malformed mask or a non-2-D image into an apparently
    # valid analysis frame.
    from .io import LoadedImage, combine_masks

    array = np.asarray(data)
    combined = combine_masks(
        array.shape,
        valid_mask=valid_mask,
        external_mask=external_mask,
    )
    return LoadedImage(
        array,
        metadata=dict(metadata or {}),
        source=Path(source) if isinstance(source, (str, os.PathLike)) else None,
        valid_mask=combined,
    )


def _read_frame_bundle(
    source: Any,
    *,
    config: Any = None,
    frame: int | None = None,
    dataset: str | None = None,
    valid_mask: Any = None,
    external_mask: Any = None,
    mask_frame: int | None = None,
    mask_dataset: str | None = None,
) -> _FrameBundle:
    """Read one frame while retaining ``LoadedImage.valid_mask`` and qmap mask."""

    configured_frame = frame if frame is not None else _config_value(config, "frame", None)
    configured_dataset = dataset if dataset is not None else _config_value(config, "dataset", None)
    configured_valid_mask = valid_mask if valid_mask is not None else _config_value(config, "valid_mask", None)
    configured_mask = external_mask if external_mask is not None else _config_value(config, "mask", None)
    configured_mask_frame = (
        mask_frame if mask_frame is not None else _config_value(config, "mask_frame", None)
    )
    configured_mask_dataset = (
        mask_dataset
        if mask_dataset is not None
        else _config_value(config, "mask_dataset", None)
    )
    if isinstance(source, Mapping):
        data = source.get("data", source.get("image", source.get("intensity")))
        if data is None:
            raise PipelineError("帧映射缺少 data/image/intensity 字段")
        image = _coerce_array(data)
        metadata = dict(source.get("metadata", {}))
        source_valid = [source.get(key) for key in ("valid_mask", "valid") if source.get(key) is not None]
        source_masks = [
            source.get(key)
            for key in ("mask", "external_mask", "bad_mask", "invalid_mask")
            if source.get(key) is not None
        ]
        valid = _combine_valid_masks(
            image.shape,
            valid_masks=(*source_valid, configured_valid_mask),
            frame=configured_mask_frame,
            dataset=configured_mask_dataset,
        )
        exclusion = _combine_external_masks(
            image.shape,
            (*source_masks, configured_mask),
            frame=configured_mask_frame,
            dataset=configured_mask_dataset,
        )
        frame = _loaded_frame(data, metadata=metadata, valid_mask=valid)
        return _FrameBundle(
            frame=frame,
            image=image,
            metadata=metadata,
            qmap=source.get("qmap"),
            valid_mask=getattr(frame, "valid_mask", None),
            external_mask=exclusion,
        )

    if isinstance(source, np.ndarray) or (
        hasattr(source, "data") and not isinstance(source, (str, os.PathLike))
    ):
        data = source if isinstance(source, np.ndarray) else getattr(source, "data")
        image = _coerce_array(data)
        metadata = dict(getattr(source, "metadata", {}) or {})
        source_valid = [
            getattr(source, key, None)
            for key in ("valid_mask", "valid")
            if getattr(source, key, None) is not None
        ]
        source_masks = [
            getattr(source, key, None)
            for key in ("mask", "external_mask", "bad_mask", "invalid_mask")
            if getattr(source, key, None) is not None
        ]
        valid = _combine_valid_masks(
            image.shape,
            valid_masks=(*source_valid, configured_valid_mask),
            frame=configured_mask_frame,
            dataset=configured_mask_dataset,
        )
        exclusion = _combine_external_masks(
            image.shape,
            (*source_masks, configured_mask),
            frame=configured_mask_frame,
            dataset=configured_mask_dataset,
        )
        frame = _loaded_frame(data, metadata=metadata, valid_mask=valid)
        return _FrameBundle(
            frame=frame,
            image=image,
            metadata=metadata,
            qmap=getattr(source, "qmap", None),
            valid_mask=getattr(frame, "valid_mask", valid),
            external_mask=exclusion,
        )

    path = Path(source)
    if not path.exists():
        raise PipelineError(f"找不到输入图像：{path}")
    suffix = path.suffix.lower()

    # NPZ files produced by the synthetic/beamline seam may contain one image
    # array plus reserved q-map arrays.  Select that sole non-q-map dataset
    # explicitly before calling the canonical loader.  If there is more than
    # one actual dataset, leave selection unset so ``io.load_image`` raises its
    # strict ambiguity error; never pick the first array as a fallback.
    loader_dataset = configured_dataset
    if suffix == ".npz" and loader_dataset is None:
        qmap_keys = {"qx", "qy", "q", "theta", "chi", "mask", "valid_mask", "q_unit"}
        try:
            with np.load(path, allow_pickle=False) as bundle_npz:
                candidates = [key for key in bundle_npz.files if key not in qmap_keys]
        except Exception as exc:  # noqa: BLE001 - normalize archive errors
            raise PipelineError(f"读取输入图像失败：{path}（{exc}）") from exc
        if len(candidates) == 1:
            loader_dataset = candidates[0]

    try:
        from .io import load_image

        reader_kwargs = {
            "frame": configured_frame,
            "dataset": loader_dataset,
            "valid_mask": configured_valid_mask,
            "mask_frame": configured_mask_frame,
            "mask_dataset": configured_mask_dataset,
        }
        reader_kwargs = {key: value for key, value in reader_kwargs.items() if value is not None}
        loaded = load_image(path, **reader_kwargs) if reader_kwargs else load_image(path)
    except Exception as exc:  # noqa: BLE001 - normalize strict IO errors
        raise PipelineError(f"读取输入图像失败：{path}（{exc}）") from exc

    data = _coerce_array(getattr(loaded, "data", loaded))
    metadata = dict(getattr(loaded, "metadata", {}) or {})
    metadata.setdefault("path", os.fspath(path))
    valid = getattr(loaded, "valid_mask", None)
    exclusion = _combine_external_masks(
        data.shape,
        (configured_mask,),
        frame=configured_mask_frame,
        dataset=configured_mask_dataset,
    )
    embedded_qmap = getattr(loaded, "qmap", None)
    if embedded_qmap is None and suffix == ".npz":
        # The selected image itself is always read by io.load_image.  Reading
        # these reserved auxiliary arrays is only for preserving the q-map
        # beside that selected dataset; no intensity array is chosen here.
        try:
            with np.load(path, allow_pickle=False) as bundle_npz:
                qkeys = {"qx", "qy", "q", "theta", "chi", "mask", "valid_mask"}
                embedded = {
                    key: np.asarray(bundle_npz[key])
                    for key in qkeys
                    if key in bundle_npz.files
                }
                if "q_unit" in bundle_npz.files:
                    raw_unit = np.asarray(bundle_npz["q_unit"])
                    if raw_unit.ndim != 0:
                        raise PipelineError("NPZ q_unit 必须是标量字符串")
                    embedded["q_unit"] = str(raw_unit.item())
            embedded_qmap = embedded or None
        except PipelineError:
            raise
        except Exception as exc:  # noqa: BLE001 - normalize archive errors
            raise PipelineError(f"读取 NPZ qmap 失败：{path}（{exc}）") from exc
    return _FrameBundle(
        # Keep the canonical LoadedImage returned by io.load_image so frame,
        # dataset and mask provenance cannot be lost at this seam.
        frame=loaded,
        image=data,
        metadata=metadata,
        qmap=embedded_qmap,
        valid_mask=valid,
        external_mask=exclusion,
    )


def read_frame(source: Any) -> tuple[np.ndarray, dict[str, Any], Any | None]:
    """Read one frame using the historical three-value compatibility seam."""

    bundle = _read_frame_bundle(source)
    return bundle.image, bundle.metadata, bundle.qmap


def build_qmap(
    image: np.ndarray,
    *,
    poni: str | os.PathLike[str] | None = None,
    qmap: Any | None = None,
    config: Any = None,
    valid_mask: Any | None = None,
) -> Any:
    """Build q-space coordinates, preferring the specialised qmap module."""

    if qmap is not None:
        coerced = _coerce_qmap(qmap, image.shape)
        if isinstance(coerced, Mapping):
            merged, _ = _merge_qmap_masks(coerced, image.shape, valid_mask)
            return merged
        return coerced
    adapter = _find_callable(
        (
            "butterfly_saxs.qmap",
            "butterfly_saxs.geometry",
            "butterfly_saxs.calibration",
        ),
        ("build_qmap", "compute_qmap", "qmap_from_poni", "make_qmap", "pixel_qmap"),
    )
    if adapter is not None:
        try:
            result = _call_adapter(
                adapter,
                image=image,
                config=config,
                poni=poni,
                valid_mask=valid_mask,
            )
            coerced = _coerce_qmap(result, image.shape)
            if isinstance(coerced, Mapping):
                merged, _ = _merge_qmap_masks(coerced, image.shape, valid_mask)
                return merged
            return coerced
        except (TypeError, ValueError, KeyError, PipelineError) as exc:
            # A supplied PONI is a physical calibration contract; silently
            # replacing it with pixel coordinates would make the result look
            # valid while changing its units.  Only the no-PONI fixture path
            # is allowed to use the deterministic NumPy fallback.
            if poni is not None:
                raise PipelineError(f"PONI 几何转换失败：{exc}") from exc
    if poni is not None:
        raise PipelineError("PONI 几何转换器不可用，无法生成物理 qmap")

    configured_q_scale = _config_value(config, "q_scale", None)
    configured_pixel_scale = _config_value(config, "pixel_q_scale", None)
    q_scale = configured_q_scale
    if q_scale is None:
        q_scale = 1.0 if configured_pixel_scale is None else configured_pixel_scale
    try:
        q_scale = float(q_scale)
    except (TypeError, ValueError) as exc:
        raise PipelineError(f"q_scale 必须是数字，实际为 {q_scale!r}") from exc
    if not math.isfinite(q_scale) or q_scale <= 0:
        raise PipelineError("q_scale 必须是正数")

    center = _config_value(config, "center", None)
    center_x = _config_value(config, "center_x", None)
    center_y = _config_value(config, "center_y", None)
    if isinstance(center, Sequence) and not isinstance(center, (str, bytes)) and len(center) >= 2:
        center_x, center_y = center[0], center[1]
    height, width = image.shape
    cx = float(width - 1) / 2.0 if center_x is None else float(center_x)
    cy = float(height - 1) / 2.0 if center_y is None else float(center_y)
    y, x = np.indices(image.shape, dtype=float)
    qx = (x - cx) * q_scale
    qy = (y - cy) * q_scale
    result = {
        "qx": qx,
        "qy": qy,
        "q": np.hypot(qx, qy),
        "q_unit": (
            _config_value(config, "q_unit", "1/nm")
            if configured_q_scale is not None
            else "pixel-q"
        ),
        "flags": ([] if configured_q_scale is not None else ["uncalibrated_pixel_q"]),
    }
    if valid_mask is not None:
        result["valid_mask"] = np.asarray(valid_mask, dtype=bool)
        result["mask"] = ~result["valid_mask"]
    return _coerce_qmap(result, image.shape)


def _q_window(image: np.ndarray, qmap: Any, config: Any = None) -> tuple[float, float] | Any:
    configured = _config_value(config, "q_window", _config_value(config, "q_range", None))
    explicit_q_min = _config_value(config, "q_min", None)
    explicit_q_max = _config_value(config, "q_max", None)
    configured_bounds: tuple[Any, Any] | None = None
    if configured is not None:
        if isinstance(configured, Mapping):
            configured_bounds = (
                configured.get("min", configured.get("q_min", configured.get("low"))),
                configured.get("max", configured.get("q_max", configured.get("high"))),
            )
        else:
            try:
                configured_bounds = tuple(configured)  # type: ignore[arg-type]
            except (TypeError, ValueError) as exc:
                raise PipelineError("q_window must be a (min, max) pair") from exc
            if len(configured_bounds) != 2:
                raise PipelineError("q_window must be a (min, max) pair")
        if explicit_q_min is None and explicit_q_max is None:
            return configured_bounds
    q_min = _config_value(config, "q_min", None)
    q_max = _config_value(config, "q_max", None)
    if q_min is None and configured_bounds is not None:
        q_min = configured_bounds[0]
    if q_max is None and configured_bounds is not None:
        q_max = configured_bounds[1]
    if q_min is not None or q_max is not None:
        if isinstance(q_min, str) and q_min.strip().casefold() in {"", "auto"}:
            q_min = None
        if isinstance(q_max, str) and q_max.strip().casefold() in {"", "auto"}:
            q_max = None
        try:
            finite_q_min = None if q_min is None else float(q_min)
            finite_q_max = None if q_max is None else float(q_max)
        except (TypeError, ValueError) as exc:
            raise PipelineError("q_min/q_max must be finite numbers or Auto") from exc
        if finite_q_min is not None and not math.isfinite(finite_q_min):
            raise PipelineError("q_min must be finite or Auto")
        if finite_q_max is not None and not math.isfinite(finite_q_max):
            raise PipelineError("q_max must be finite or Auto")
        try:
            _, _, q = _qmap_arrays(qmap, image.shape)
            finite = q[np.isfinite(q)]
            if not finite.size:
                raise PipelineError("q map has no finite pixels")
            return (
                float(np.min(finite)) if finite_q_min is None else finite_q_min,
                float(np.max(finite)) if finite_q_max is None else finite_q_max,
            )
        except (PipelineError, ValueError):
            if finite_q_min is None or finite_q_max is None:
                raise
            return finite_q_min, finite_q_max
    try:
        _, _, q = _qmap_arrays(qmap, image.shape)
        finite = q[np.isfinite(q)]
        if finite.size:
            return float(np.min(finite)), float(np.max(finite))
    except (PipelineError, ValueError):
        pass
    return None


def _analysis_options(image: np.ndarray, qmap: Any, config: Any = None) -> dict[str, Any]:
    """Resolve the analysis mapping once for every observable entry point."""
    # The service resolver is the canonical non-qmap settings boundary.  Map
    # historical pipeline aliases into it once, then only resolve q bounds
    # from the actual q map below.
    from .settings import resolve_analysis_settings

    raw: dict[str, Any] = {}
    if isinstance(config, Mapping):
        for group_name in ("analysis", "project", "input", "inputs", "output"):
            group = config.get(group_name)
            if isinstance(group, Mapping):
                raw.update(group)
        raw.update(
            {
                str(key): value
                for key, value in config.items()
                if key not in {"analysis", "project", "input", "inputs", "output"}
            }
        )
    else:
        analysis = getattr(config, "analysis", None) if config is not None else None
        if isinstance(analysis, Mapping):
            raw.update(analysis)
    if "n_angles" in raw and "n_ridge_angles" not in raw:
        raw["n_ridge_angles"] = raw["n_angles"]
    if "ridge_bins" in raw and "n_angular_bins" not in raw:
        raw["n_angular_bins"] = raw["ridge_bins"]
    if "curvature_normal_step" in raw and "normal_step" not in raw:
        raw["normal_step"] = raw["curvature_normal_step"]
    curvature = raw.get("curvature")
    if isinstance(curvature, Mapping):
        raw.setdefault("curvature_sigma", curvature.get("sigma", curvature.get("smooth_sigma", 2.0)))
        raw.setdefault("curvature_percentile", curvature.get("percentile", 25.0))
        raw.setdefault("normal_step", curvature.get("normal_step", 1.0))
    elif curvature is not None and not isinstance(curvature, bool):
        raw.setdefault("curvature_sigma", curvature)
    canonical = resolve_analysis_settings(raw)
    method = canonical["ridge_method"]
    ellipse = canonical.get("ellipse")
    return {
        "q_window": _q_window(image, qmap, canonical),
        "mask": raw.get("mask"),
        "ridge_method": method,
        "ridge_snr_threshold": canonical["ridge_snr_threshold"],
        "ridge_min_peak_fraction": canonical["ridge_min_peak_fraction"],
        "ridge_min_coverage": canonical["ridge_min_coverage"],
        "n_angles": canonical["n_ridge_angles"],
        "n_angular_bins": canonical["n_angular_bins"],
        "n_radial_bins": canonical["n_radial_bins"],
        "max_pixels": canonical["max_pixels"],
        "draw_axis_deg": canonical["draw_axis_deg"],
        "curvature_sigma": canonical["curvature_sigma"],
        "curvature_percentile": canonical["curvature_percentile"],
        "curvature_normal_step": canonical["normal_step"],
        "ellipse": ellipse,
        "ellipse_residual": canonical["ellipse_residual"],
        "ellipse_multistart": canonical["ellipse_multistart"],
        "full2d_multistart": canonical["full2d_multistart"],
    }


def _call_supported(function: Any, *args: Any, **kwargs: Any) -> Any:
    """Call an evolving scientific seam without hiding body exceptions."""

    try:
        signature = _inspect.signature(function)
    except (TypeError, ValueError):
        return function(*args, **kwargs)
    parameters = signature.parameters
    if any(item.kind == _inspect.Parameter.VAR_KEYWORD for item in parameters.values()):
        return function(*args, **kwargs)
    accepted = {
        name: value
        for name, value in kwargs.items()
        if name in parameters
        and parameters[name].kind != _inspect.Parameter.POSITIONAL_ONLY
    }
    return function(*args, **accepted)


def _configured_external_mask(
    value: Any,
    shape: tuple[int, int],
    *,
    config: Any = None,
) -> np.ndarray | None:
    """Resolve one configured negative-polarity mask."""

    if value is not None:
        if isinstance(value, (str, os.PathLike)):
            try:
                from .io import combine_masks, load_image

                loaded_mask = load_image(
                    value,
                    frame=_config_value(config, "mask_frame", None),
                    dataset=_config_value(config, "mask_dataset", None),
                ).data
                valid = combine_masks(shape, external_mask=loaded_mask)
                return None if valid is None else ~valid
            except Exception as exc:  # noqa: BLE001 - report a useful config error
                raise PipelineError(f"无法读取 analysis.mask：{value}（{exc}）") from exc
        array = np.asarray(value, dtype=bool)
        if array.shape != shape:
            raise PipelineError(f"analysis.mask 形状 {array.shape} 与图像 {shape} 不一致")
        return array
    return None


def _roi_exclusion_mask(
    shape: tuple[int, int], *, qmap: Any = None, config: Any = None
) -> np.ndarray | None:
    """Resolve configured ROIs separately so their count remains auditable."""

    rois = _config_value(config, "rois", ())
    if rois:
        try:
            from .masking import combine_exclusion_masks

            qx = qy = None
            if qmap is not None:
                qx, qy, _ = _qmap_arrays(qmap, shape)
            return combine_exclusion_masks(shape, rois=rois, qx=qx, qy=qy)
        except Exception as exc:  # noqa: BLE001 - fail closed for ROI config
            raise PipelineError(f"无法解析 analysis.rois：{exc}") from exc
    return None


def _observable_mask(value: Any, shape: tuple[int, int], *, qmap: Any = None, config: Any = None) -> Any:
    """Resolve configured external mask and ROIs to one legacy exclusion mask."""

    masks = [
        item
        for item in (
            _configured_external_mask(value, shape, config=config),
            _roi_exclusion_mask(shape, qmap=qmap, config=config),
        )
        if item is not None
    ]
    return None if not masks else np.logical_or.reduce(masks)


def _analysis_domain(
    image: np.ndarray,
    qmap: Any,
    *,
    config: Any = None,
    sigma: Any = None,
    weights: Any = None,
    detector_valid: Any = None,
    external_mask: Any = None,
    include_config_mask: bool = True,
) -> AnalysisDomain:
    """Resolve the single auditable pixel population for all analysis stages."""

    qx, qy, q = _qmap_arrays(qmap, image.shape)
    options = _analysis_options(image, qmap, config)
    qmap_detector_valid = None
    if isinstance(qmap, Mapping):
        qmap_detector_valid = qmap.get("valid_mask", qmap.get("valid"))
    if detector_valid is None:
        detector_valid = qmap_detector_valid
    elif qmap_detector_valid is not None:
        detector_valid = (
            np.asarray(detector_valid, dtype=bool)
            & np.asarray(qmap_detector_valid, dtype=bool)
        )
    exclusions: list[np.ndarray] = []
    if external_mask is not None:
        explicit = np.asarray(external_mask, dtype=bool)
        if explicit.shape != image.shape:
            raise PipelineError(
                f"external_mask 形状 {explicit.shape} 与图像 {image.shape} 不一致"
            )
        exclusions.append(explicit)
    if include_config_mask:
        configured = _configured_external_mask(
            options["mask"], image.shape, config=config
        )
        if configured is not None:
            exclusions.append(configured)
    exclusion = None if not exclusions else np.logical_or.reduce(exclusions)
    roi_exclusion = _roi_exclusion_mask(image.shape, qmap=qmap, config=config)
    if sigma is None:
        sigma = _config_value(config, "sigma", None)
    if weights is None:
        weights = _config_value(config, "weights", None)
    if sigma is not None and not isinstance(sigma, np.ndarray):
        sigma = _resolve_full2d_array(
            sigma, name="sigma", shape=image.shape, config=config
        )
    if weights is not None and not isinstance(weights, np.ndarray):
        weights = _resolve_full2d_array(
            weights, name="weights", shape=image.shape, config=config
        )
    return build_analysis_domain(
        image,
        qx,
        qy,
        q=q,
        detector_valid=detector_valid,
        external_mask=exclusion,
        roi_exclusion=roi_exclusion,
        q_window=options["q_window"],
        sigma=sigma,
        weights=weights,
    )


def measure_observables(
    image: np.ndarray,
    qmap: Any,
    *,
    config: Any = None,
    frame: Any = None,
    fit_ellipse: bool = True,
    analysis_domain: AnalysisDomain | None = None,
    cancel_event: Any = None,
) -> dict[str, Any]:
    """Measure angular/lobe/ridge observables with the declared config."""

    from . import observables as observable_module

    options = _analysis_options(image, qmap, config)
    domain = analysis_domain or _analysis_domain(image, qmap, config=config)
    observed_frame = frame if frame is not None else _loaded_frame(image)
    from .settings import ellipse_parameter_specs

    ellipse_parameters = ellipse_parameter_specs(
        {"ellipse": options["ellipse"]} if options["ellipse"] is not None else None,
        q_window=domain.q_window,
    )
    result = _call_supported(
        observable_module.measure_observables,
        observed_frame,
        qmap,
        domain.q_window,
        n_angular_bins=options["n_angular_bins"],
        n_ridge_angles=options["n_angles"],
        n_radial_bins=options["n_radial_bins"],
        fit_ellipse=bool(fit_ellipse),
        mask=~domain.fit_valid_mask,
        ridge_method=options["ridge_method"],
        ridge_snr_threshold=options["ridge_snr_threshold"],
        ridge_min_peak_fraction=options["ridge_min_peak_fraction"],
        ridge_min_coverage=options["ridge_min_coverage"],
        draw_axis_deg=options["draw_axis_deg"],
        curvature_sigma=options["curvature_sigma"],
        curvature_percentile=options["curvature_percentile"],
        curvature_normal_step=options["curvature_normal_step"],
        p4_quality_thresholds=_config_value(config, "p4_quality_thresholds", None),
        ellipse_parameters=ellipse_parameters,
        ellipse_residual=options["ellipse_residual"],
        ellipse_multistart=options["ellipse_multistart"],
        cancel_event=cancel_event,
    )
    # Do not infer alpha/phi from the fitted ellipse rotation.  The papers'
    # microscopic tilts are not identifiable from this apparent trajectory.
    public = _public_angles(_as_mapping(result))
    if isinstance(public, Mapping):
        normalized_ridges = _ridges_from_observable_bundle(public)
        ridge = public.get("ridge")
        if normalized_ridges is not None and isinstance(ridge, Mapping):
            ridge = dict(ridge)
            ridge["points"] = normalized_ridges
            public = dict(public)
            public["ridge"] = ridge
    return public


def extract_ridges(
    image: np.ndarray,
    qmap: Any,
    *,
    config: Any = None,
    analysis_domain: AnalysisDomain | None = None,
) -> list[dict[str, float]]:
    """Extract the observed radial ridge with all configured safeguards."""

    from . import observables as observable_module

    options = _analysis_options(image, qmap, config)
    domain = analysis_domain or _analysis_domain(image, qmap, config=config)
    frame = _loaded_frame(image)
    track = _call_supported(
        observable_module.measure_radial_ridges,
        frame,
        qmap,
        domain.q_window,
        n_angles=options["n_angles"],
        n_bins=options["n_radial_bins"],
        mask=~domain.fit_valid_mask,
        ridge_method=options["ridge_method"],
        snr_threshold=options["ridge_snr_threshold"],
        ridge_min_peak_fraction=options["ridge_min_peak_fraction"],
        ridge_min_coverage=options["ridge_min_coverage"],
        curvature_sigma=options["curvature_sigma"],
        curvature_percentile=options["curvature_percentile"],
        curvature_normal_step=options["curvature_normal_step"],
    )
    raw_points = getattr(track, "points", track)
    points: list[dict[str, Any]] = []
    for item in raw_points or []:
        point = _public_angles(_as_mapping(item))
        if not isinstance(point, Mapping):
            continue
        metadata = point.get("metadata")
        if not isinstance(metadata, Mapping):
            candidate_metadata = getattr(item, "metadata", None)
            metadata = candidate_metadata if isinstance(candidate_metadata, Mapping) else None
        if isinstance(metadata, Mapping):
            point = dict(point)
            for key in (
                "quadrant",
                "quadrant_pair",
                "branch_assignment_source",
                "symmetry_flags",
            ):
                if key in metadata:
                    point.setdefault(key, metadata[key])
        point = {str(key): _jsonable(value, array_summary=False) for key, value in point.items()}
        # Invalid sectors are retained as records for coverage diagnostics but
        # do not contribute artificial (0, 0) coordinates to the ellipse fit.
        if point.get("valid") is False:
            point.pop("qx", None)
            point.pop("qy", None)
            point.pop("q", None)
        points.append(point)
    return points


def _ridges_from_observable_bundle(observables: Any) -> list[dict[str, Any]] | None:
    """Reuse the ridge already measured by ``measure_observables``.

    Surface-curvature extraction is one of the most expensive preprocessing
    stages.  The high-level measurement call already performed it, so the
    single-frame pipeline must not repeat the same detector-grid calculation
    merely to feed the ellipse adapter.
    """

    bundle = observables if isinstance(observables, Mapping) else _as_mapping(observables)
    if not isinstance(bundle, Mapping):
        return None
    ridge = bundle.get("ridge")
    ridge_mapping = ridge if isinstance(ridge, Mapping) else _as_mapping(ridge)
    if not isinstance(ridge_mapping, Mapping):
        return None
    raw_points = ridge_mapping.get("points")
    if not isinstance(raw_points, Sequence) or isinstance(raw_points, (str, bytes)):
        return None
    ellipse = bundle.get("ellipse")
    ellipse_mapping = ellipse if isinstance(ellipse, Mapping) else _as_mapping(ellipse)
    reference_axis_deg = float(
        ellipse_mapping.get("reference_axis_deg", 0.0)
        if isinstance(ellipse_mapping, Mapping)
        else 0.0
    )
    parameter_values = (
        ellipse_mapping.get("parameters", {})
        if isinstance(ellipse_mapping, Mapping)
        else {}
    )
    if not isinstance(parameter_values, Mapping):
        parameter_values = {}
    center_qx = float(parameter_values.get("center_qx", parameter_values.get("cx", 0.0)) or 0.0)
    center_qy = float(parameter_values.get("center_qy", parameter_values.get("cy", 0.0)) or 0.0)
    points: list[dict[str, Any]] = []
    for item in raw_points:
        mapping = _as_mapping(item)
        if not mapping:
            continue
        metadata = mapping.get("metadata")
        if not isinstance(metadata, Mapping):
            candidate_metadata = getattr(item, "metadata", None)
            metadata = candidate_metadata if isinstance(candidate_metadata, Mapping) else None
        point = {str(key): _jsonable(value, array_summary=False) for key, value in mapping.items()}
        if isinstance(metadata, Mapping):
            point.update(
                {
                    key: _jsonable(metadata[key], array_summary=False)
                    for key in (
                        "quadrant",
                        "quadrant_pair",
                        "branch_assignment_source",
                        "symmetry_flags",
                    )
                    if key in metadata
                }
            )
        try:
            qx_value = float(point.get("qx"))
            qy_value = float(point.get("qy"))
            local_angle = float(
                (
                    np.arctan2(qy_value - center_qy, qx_value - center_qx)
                    - np.deg2rad(reference_axis_deg)
                )
                % (2.0 * np.pi)
            )
            quadrant_index = int(np.floor(local_angle / (0.5 * np.pi))) % 4
            quadrant_name = ("QI", "QII", "QIII", "QIV")[quadrant_index]
            point.setdefault("quadrant", quadrant_name)
            point.setdefault(
                "quadrant_pair",
                "QI+QIII" if quadrant_index in (0, 2) else "QII+QIV",
            )
            point.setdefault("branch_assignment_source", "reference_quadrant")
            point.setdefault("symmetry_flags", [])
        except (TypeError, ValueError, OverflowError):
            pass
        if point.get("valid") is False:
            point.pop("qx", None)
            point.pop("qy", None)
            point.pop("q", None)
        points.append(point)
    return points


def _ellipse_residual(parameters: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    a = math.exp(float(parameters[0]))
    b = math.exp(float(parameters[1]))
    angle = float(parameters[2])
    c, s = math.cos(angle), math.sin(angle)
    xp = c * x + s * y
    yp = -s * x + c * y
    return (xp / a) ** 2 + (yp / b) ** 2 - 1.0


def _qmap_unit(qmap: Any, config: Any = None) -> str:
    unit = None
    if isinstance(qmap, Mapping):
        unit = qmap.get("q_unit")
        if unit is None:
            unit = qmap.get("unit")
        if unit is None and isinstance(qmap.get("metadata"), Mapping):
            unit = qmap["metadata"].get("q_unit", qmap["metadata"].get("unit"))
    elif qmap is not None:
        unit = getattr(qmap, "q_unit", None)
        if unit is None:
            unit = getattr(qmap, "unit", None)
        if unit is None:
            metadata = getattr(qmap, "metadata", None)
            if isinstance(metadata, Mapping):
                unit = metadata.get("q_unit", metadata.get("unit"))
    if unit is not None:
        return str(unit)
    configured = _config_value(config, "q_unit", None)
    if configured is not None:
        return str(configured)
    return "unknown"


def _ellipse_value(value: Any, name: str, default: Any = None) -> Any:
    """Read a field from either a fitted object or its public mapping."""

    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _as_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _public_ellipse_fit(
    fit: Any,
    *,
    n_points: int,
    qmap: Any = None,
    config: Any = None,
) -> dict[str, Any]:
    """Compatibility wrapper around the shared canonical ellipse payload."""

    payload = canonical_ellipse_payload(
        fit, n_points=n_points, qmap=qmap, config=config
    )
    assignment = payload.get("branch_assignment")
    if assignment is not None and not isinstance(assignment, Mapping):
        try:
            assignment_array = np.asarray(assignment)
            payload["branch_assignment_values"] = assignment_array
            # Historical pipeline callers received a compact array descriptor;
            # retain that shape while exposing the lossless aligned vector.
            payload["branch_assignment"] = {
                "shape": list(assignment_array.shape),
                "dtype": str(assignment_array.dtype),
            }
        except (TypeError, ValueError):
            pass
    return payload


def fit_symmetric_ellipses(
    points: Any,
    *,
    config: Any = None,
    qmap: Any = None,
) -> dict[str, Any]:
    """Fit a shared-centre pair and expose only explicit public quantities."""

    from . import observables as observable_module

    if isinstance(points, Mapping):
        source_points = points.get("points", points.get("ridges", points))
    else:
        source_points = getattr(points, "points", points)
    rows: list[dict[str, Any]] = []
    for item in source_points or []:
        if isinstance(item, Mapping):
            if item.get("valid") is False:
                continue
            x, y = item.get("qx", item.get("x")), item.get("qy", item.get("y"))
            row = {
                "qx": x,
                "qy": y,
                "branch_id": item.get("branch_id", item.get("component")),
                "weight": item.get("weight"),
            }
        else:
            x, y = getattr(item, "qx", getattr(item, "x", None)), getattr(item, "qy", getattr(item, "y", None))
            row = {
                "qx": x,
                "qy": y,
                "branch_id": getattr(item, "branch_id", getattr(item, "component", None)),
                "weight": getattr(item, "weight", None),
            }
        try:
            if x is not None and y is not None and math.isfinite(float(x)) and math.isfinite(float(y)) and np.hypot(float(x), float(y)) > 0:
                row["qx"] = float(x)
                row["qy"] = float(y)
                rows.append(row)
        except (TypeError, ValueError):
            continue
    if len(rows) < 5:
        return {"status": "insufficient_data", "n_points": len(rows), "ellipses": [], "rmse": float("nan")}
    values_cfg = _config_value(config, "ellipse", None)
    kwargs: dict[str, Any] = {}
    if isinstance(values_cfg, Mapping):
        from .settings import ellipse_parameter_specs, normalize_ellipse_settings

        normalized = normalize_ellipse_settings({"ellipse": values_cfg})
        if normalized is not None:
            radii = np.asarray([[row["qx"], row["qy"]] for row in rows], dtype=float)
            ellipse_q_window = (
                float(np.min(np.hypot(radii[:, 0], radii[:, 1]))),
                float(np.max(np.hypot(radii[:, 0], radii[:, 1]))),
            )
            kwargs["parameters"] = ellipse_parameter_specs(
                {"ellipse": normalized},
                q_window=ellipse_q_window,
            )
            kwargs["residual"] = normalized["residual"]
            kwargs["multistart"] = normalized["multistart"]
        else:
            kwargs["parameters"] = dict(values_cfg)
    for key in ("residual", "loss", "f_scale", "max_nfev"):
        value = _config_value(config, key, None)
        if value is not None:
            kwargs[key] = value
    draw_axis_deg = float(_config_value(config, "draw_axis_deg", 90.0))
    reference_axis_deg = draw_axis_deg - 90.0
    strict_symmetry = bool(
        _config_value(
            config,
            "strict_symmetry",
            any(row.get("branch_id") is not None for row in rows),
        )
    )
    try:
        fit = observable_module.fit_symmetric_double_ellipse(
            rows,
            **kwargs,
            reference_axis_deg=reference_axis_deg,
            q_unit=_qmap_unit(qmap, config),
            strict_symmetry=strict_symmetry,
        )
    except (TypeError, ValueError, FloatingPointError) as exc:
        return {"status": "failed", "n_points": len(rows), "ellipses": [], "rmse": float("nan"), "message": str(exc)}
    return _public_ellipse_fit(fit, n_points=len(rows), qmap=qmap, config=config)


def _resolve_full2d_array(
    value: Any,
    *,
    name: str,
    shape: tuple[int, int],
    config: Any = None,
) -> np.ndarray:
    """Resolve an in-memory or project-relative sigma/weights array."""

    source = value
    if isinstance(source, (str, os.PathLike)):
        path = Path(source)
        # ``run_project``/the CLI call ``ProjectConfig.resolve_paths`` first,
        # so normal project-file paths are already absolute.  The optional
        # base-dir aliases make direct mapping configs equally explicit.
        base_dir = _config_value(config, "base_dir", _config_value(config, "config_dir", None))
        if base_dir is None and isinstance(config, Mapping):
            base_dir = config.get("project_base_dir", config.get("config_path"))
            if base_dir is not None and Path(base_dir).suffix:
                base_dir = Path(base_dir).parent
        if base_dir is None and config is not None:
            metadata = getattr(config, "metadata", None)
            if isinstance(metadata, Mapping):
                base_dir = metadata.get("base_dir", metadata.get("config_dir"))
        if not path.is_absolute() and base_dir is not None:
            path = Path(base_dir) / path
        try:
            if path.suffix.lower() == ".npy":
                source = np.load(path, allow_pickle=False)
            elif path.suffix.lower() == ".npz":
                with np.load(path, allow_pickle=False) as bundle:
                    if name in bundle.files:
                        source = bundle[name]
                    elif "data" in bundle.files:
                        source = bundle["data"]
                    elif len(bundle.files) == 1:
                        source = bundle[bundle.files[0]]
                    else:
                        raise PipelineError(
                            f"{name} 文件 {path} 包含多个数组，请使用 {name!r} 或 data 键"
                        )
            else:
                from .io import load_image

                source = load_image(path).data
        except PipelineError:
            raise
        except Exception as exc:  # noqa: BLE001 - normalize array path errors
            raise PipelineError(f"无法读取 full2d {name}：{path}（{exc}）") from exc
    array = np.asarray(source, dtype=float)
    if array.shape != shape:
        raise PipelineError(f"full2d {name} 形状 {array.shape} 与图像 {shape} 不一致")
    return array


def fit_full2d(
    image: np.ndarray,
    qmap: Any,
    ellipse_fit: Mapping[str, Any],
    *,
    config: Any = None,
    frame: Any = None,
    initial_parameters: Any = None,
    analysis_domain: AnalysisDomain | None = None,
    cancel_event: Any = None,
) -> Any:
    """Run the empirical pixel-wise intensity refinement when requested."""

    from .intensity import default_intensity_parameters, fit_intensity_model, parameter_values
    from .parameters import ParameterSet

    options = _analysis_options(image, qmap, config)
    observed_frame = frame if frame is not None else _loaded_frame(image)
    auto_initial = initial_parameters is None
    initial = initial_parameters
    if isinstance(initial, ParameterSet):
        # ParameterSet.items() intentionally exposes resolved numeric values,
        # so converting it through dict() would discard bounds, vary/fixed
        # state, and tied expressions (notably b=a*axis_ratio).  Copy the
        # editable specification graph while keeping the caller's warm start
        # untouched.
        initial = initial.copy()
    elif isinstance(initial, Mapping):
        # Public fit results intentionally expose both core radians and degree
        # display adapters.  A warm start is internal model state, so retain
        # the authoritative core values and remove only redundant adapters.
        initial = dict(initial)
        for radians_name, degrees_name in (
            ("theta", "theta_deg"),
            ("lobe_angle", "lobe_angle_deg"),
            ("angular_width", "angular_width_deg"),
        ):
            if radians_name in initial and degrees_name in initial:
                initial.pop(degrees_name)
    if initial is None:
        params = _as_mapping(ellipse_fit.get("parameters", {}) if isinstance(ellipse_fit, Mapping) else {})
        try:
            a = float(params.get("a", 1.0))
        except (TypeError, ValueError):
            a = 1.0
        if not np.isfinite(a) or a <= 0:
            a = 1.0
        try:
            ratio = float(params.get("axis_ratio"))
        except (TypeError, ValueError):
            ratio = float("nan")
        if not np.isfinite(ratio):
            try:
                ratio = float(params.get("b")) / a
            except (TypeError, ValueError, ZeroDivisionError):
                ratio = 0.7
        if not np.isfinite(ratio) or not 0 < ratio <= 1:
            ratio = 0.7
        theta_value = params.get("theta_deg", params.get("angle_deg"))
        if theta_value is None and params.get("theta") is not None:
            try:
                theta_value = np.degrees(float(params["theta"]))
            except (TypeError, ValueError):
                theta_value = 0.0
        try:
            theta_deg = float(0.0 if theta_value is None else theta_value)
        except (TypeError, ValueError):
            theta_deg = 0.0
        if not np.isfinite(theta_deg):
            theta_deg = 0.0
        initial = default_intensity_parameters(a=a, axis_ratio=ratio, theta_deg=theta_deg)
    analysis = getattr(config, "analysis", None) if config is not None else None
    if not isinstance(analysis, Mapping):
        analysis = (
            config.get("analysis", config)
            if isinstance(config, Mapping)
            else {}
        )
    kwargs: dict[str, Any] = {
        "q_window": options["q_window"],
        "fixed": analysis.get("fixed"),
        "bounds": analysis.get("bounds"),
        # Precision-first default: every valid detector pixel participates.
        # A deterministic speed cap is opt-in through analysis.max_pixels.
        # ``0`` is the public spelling for all valid detector pixels.  The
        # intensity engine uses ``None`` for that mode; forwarding zero would
        # otherwise create a one-pixel sample.
        "max_pixels": None if options["max_pixels"] == 0 else options["max_pixels"],
        "seed": analysis.get("seed", 0),
        "scales": analysis.get("scales", (0.25, 0.5, 1.0)),
        "robust_loss": analysis.get("robust_loss", analysis.get("loss", "soft_l1")),
        "f_scale": analysis.get("f_scale", 1.0),
        "max_nfev": analysis.get("max_nfev", 800),
        "multistart": analysis.get("full2d_multistart", options["full2d_multistart"]),
        "reference_axis_deg": float(options["draw_axis_deg"]) - 90.0,
        # Detector counts/absolute intensity can differ by many orders of
        # magnitude.  Scale only internally generated defaults; explicit or
        # warm-started values remain authoritative unless requested in config.
        "auto_scale_initial": bool(
            analysis.get("auto_scale_initial", auto_initial)
        ),
        "cancel_event": cancel_event,
    }
    for name in ("sigma", "weights"):
        value = analysis.get(name)
        if value is not None:
            kwargs[name] = _resolve_full2d_array(
                value,
                name=name,
                shape=image.shape,
                config=config,
            )
    domain = analysis_domain or _analysis_domain(
        image,
        qmap,
        config=config,
        sigma=kwargs.get("sigma"),
        weights=kwargs.get("weights"),
    )
    kwargs["q_window"] = domain.q_window
    kwargs["mask"] = ~domain.fit_valid_mask
    kwargs = {key: value for key, value in kwargs.items() if value is not None}
    try:
        fit = _call_supported(fit_intensity_model, observed_frame, qmap, initial, **kwargs)
    except AnalysisCancelled:
        raise
    except Exception as exc:  # noqa: BLE001 - full2d is an explicit optional stage
        return {
            "status": "error",
            "message": f"full2d 精修失败：{exc}",
            "flags": ("empirical_model_only", "nonunique_inverse_problem"),
        }
    values = parameter_values(getattr(fit, "parameters", None))
    for radians_name, degrees_name in (
        ("theta", "theta_deg"),
        ("lobe_angle", "lobe_angle_deg"),
        ("angular_width", "angular_width_deg"),
    ):
        if radians_name in values:
            values[degrees_name] = float(np.degrees(float(values[radians_name])))
    # ``IntensityFitResult`` retains flattened prediction/residual arrays;
    # expose image-shaped aliases as well so UI/export consumers cannot lose
    # the full 2-D feedback at a JSON-summary boundary.
    model = np.asarray(getattr(fit, "model_image", getattr(fit, "prediction", np.asarray([]))))
    residual = np.asarray(getattr(fit, "residual_image", getattr(fit, "residual", np.asarray([]))))
    raw = _as_mapping(fit)
    raw.update(
        status="ok" if bool(getattr(fit, "success", False)) else "failed",
        success=bool(getattr(fit, "success", False)),
        parameters=values if values else parameter_values(getattr(fit, "parameters", None)),
        model=model,
        residual=residual,
        model_image=model,
        residual_image=residual,
        reference_axis_deg=float(options["draw_axis_deg"]) - 90.0,
        # Keep the scientific boundary visible even if an older intensity
        # result class did not carry an explicit flags field.
        flags=tuple(getattr(fit, "flags", ("empirical_model_only", "nonunique_inverse_problem"))),
    )
    for name in (
        "sample_cost",
        "full_cost",
        "selection_objective",
        "candidate_solutions",
        "selected_start_index",
        "multistart_count",
    ):
        if name not in raw:
            value = getattr(fit, name, None)
            if value is not None:
                raw[name] = _jsonable(value, array_summary=False)
    raw["fit_audit"] = {
        name: raw.get(name)
        for name in (
            "sample_cost",
            "full_cost",
            "selection_objective",
            "candidate_solutions",
            "selected_start_index",
            "multistart_count",
        )
        if raw.get(name) is not None
    }
    if "condition" not in raw and "condition_number" in raw:
        raw["condition"] = raw["condition_number"]
    return raw


@dataclass
class PipelineResult:
    """Result object shared by CLI, GUI, and batch consumers."""

    image: np.ndarray
    qmap: Any
    observables: dict[str, Any]
    ridges: list[dict[str, Any]]
    ellipse_fit: dict[str, Any]
    full2d: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
    flags: dict[str, Any] = field(default_factory=dict)
    output_paths: list[str] = field(default_factory=list)
    valid_mask: np.ndarray | None = None
    analysis_domain: AnalysisDomain | None = None
    analysis: dict[str, Any] = field(default_factory=dict)
    analysis_arrays: dict[str, np.ndarray] = field(default_factory=dict, repr=False)

    @property
    def data(self) -> np.ndarray:
        return self.image

    @property
    def ellipse(self) -> dict[str, Any]:
        return self.ellipse_fit

    @property
    def parameters(self) -> Any:
        """Preferred longitudinal parameter source for batch exporters."""

        if isinstance(self.full2d, Mapping) and self.full2d.get("parameters") is not None:
            return self.full2d.get("parameters")
        if isinstance(self.ellipse_fit, Mapping):
            return self.ellipse_fit.get("parameters", {})
        return {}

    @property
    def lobe_radial_profiles(self) -> Any:
        """Independent narrow-sector radial profiles around observed lobes."""

        return self.observables.get("lobe_radial_profiles") if isinstance(self.observables, Mapping) else None

    @property
    def lobe_radial_peaks(self) -> Any:
        """Radial peak summaries paired with :attr:`lobe_radial_profiles`."""

        return self.observables.get("lobe_radial_peaks") if isinstance(self.observables, Mapping) else None

    def __getitem__(self, key: str) -> Any:
        return self.to_mapping()[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.to_mapping().get(key, default)

    def to_mapping(self, *, include_arrays: bool = False) -> dict[str, Any]:
        result = {
            "metadata": self.metadata,
            "flags": self.flags,
            "observables": self.observables,
            "ridges": self.ridges,
            "ellipse_fit": self.ellipse_fit,
            # Batch/checkpoint consumers resume from this stable top-level
            # parameter source rather than knowing which fit stage produced it.
            "parameters": self.parameters,
            "lobe_radial_profiles": self.lobe_radial_profiles,
            "lobe_radial_peaks": self.lobe_radial_peaks,
            "full2d": self.full2d,
            "output_paths": self.output_paths,
            "image": self.image,
            "qmap": self.qmap,
            "valid_mask": self.valid_mask,
            "analysis_domain": (
                None if self.analysis_domain is None else self.analysis_domain.to_summary()
            ),
            "analysis": self.analysis,
        }
        if include_arrays and self.analysis_domain is not None:
            for name in (
                "finite_mask",
                "detector_valid_mask",
                "external_valid_mask",
                "q_window_mask",
                "roi_exclusion_mask",
                "weight_valid_mask",
                "fit_valid_mask",
                "sampled_valid_mask",
            ):
                result[name] = np.asarray(getattr(self.analysis_domain, name), dtype=bool)
        if include_arrays and self.analysis_arrays:
            result.update(self.analysis_arrays)
        return _jsonable(result, array_summary=not include_arrays)


def inspect_frame(
    source: Any,
    *,
    qmap: Any | None = None,
    poni: str | os.PathLike[str] | None = None,
    config: Any = None,
    frame: int | None = None,
    dataset: str | None = None,
    valid_mask: Any = None,
    mask: Any = None,
    mask_frame: int | None = None,
    mask_dataset: str | None = None,
    fit_ellipse: bool = False,
) -> dict[str, Any]:
    """Return frame/q-space diagnostics, with ellipse fitting explicit.

    Inspection is intentionally lightweight by default.  Pass
    ``fit_ellipse=True`` (or set ``analysis.fit_ellipse=true``) when the
    diagnostic report should include the canonical ellipse solver output.
    """

    bundle = _read_frame_bundle(
        source,
        config=config,
        frame=frame,
        dataset=dataset,
        valid_mask=valid_mask,
        external_mask=mask,
        mask_frame=mask_frame,
        mask_dataset=mask_dataset,
    )
    image, metadata, embedded_qmap = bundle.image, bundle.metadata, bundle.qmap
    selected_qmap = qmap if qmap is not None else embedded_qmap
    qmap_obj = build_qmap(
        image,
        poni=poni or _config_value(config, "poni_path", None),
        qmap=selected_qmap,
        config=config,
        valid_mask=bundle.valid_mask,
    )
    qx, qy, q = _qmap_arrays(qmap_obj, image.shape)
    domain = _analysis_domain(
        image,
        qmap_obj,
        config=config,
        detector_valid=bundle.valid_mask,
        external_mask=bundle.external_mask,
        include_config_mask=False,
    )
    finite = np.isfinite(image)
    requested_fit_ellipse = bool(_config_value(config, "fit_ellipse", fit_ellipse))
    measured = measure_observables(
        image,
        qmap_obj,
        config=config,
        frame=bundle.frame,
        fit_ellipse=requested_fit_ellipse,
        analysis_domain=domain,
    )
    return {
        "metadata": metadata,
        "shape": list(image.shape),
        "dtype": str(image.dtype),
        "finite_fraction": float(np.mean(finite)),
        "intensity_min": float(np.nanmin(image)) if np.any(finite) else None,
        "intensity_max": float(np.nanmax(image)) if np.any(finite) else None,
        "q_range": [float(np.nanmin(q)), float(np.nanmax(q))],
        "qx_range": [float(np.nanmin(qx)), float(np.nanmax(qx))],
        "qy_range": [float(np.nanmin(qy)), float(np.nanmax(qy))],
        "q_unit": _qmap_unit(qmap_obj, config),
        # ``inspect`` is a CLI-facing report, so collapse profile arrays to
        # strict JSON summaries at this boundary rather than leaking NumPy
        # objects into the command handler.
        "observables": _jsonable(measured),
        "ellipse_measured": requested_fit_ellipse,
        "ellipse_fit": _jsonable(measured.get("ellipse")),
        "valid_mask": _jsonable(domain.fit_valid_mask),
        "analysis_domain": domain.to_summary(),
        "flags": {
            "empirical_model_only": True,
            "mechanism_under_determined": True,
            "forward_simulation_only": False,
            "nonunique_inverse_problem": True,
            "uncalibrated_pixel_q": bool(
                (qmap_obj.get("q_unit") if isinstance(qmap_obj, Mapping) else None) == "pixel-q"
            ),
        },
    }


def analyze_frame(
    source: Any,
    *,
    qmap: Any | None = None,
    poni: str | os.PathLike[str] | None = None,
    config: Any = None,
    full2d: bool | None = None,
    initial_parameters: Any = None,
    frame: int | None = None,
    dataset: str | None = None,
    valid_mask: Any = None,
    mask: Any = None,
    mask_frame: int | None = None,
    mask_dataset: str | None = None,
    output: str | os.PathLike[str] | None = None,
    force: bool = False,
    geometry_cache: Any = None,
    cancel_event: Any = None,
) -> PipelineResult:
    """Run read → qmap → observables → ridge → symmetric ellipse → full2d."""

    raise_if_cancelled(cancel_event, "frame-load")
    bundle = _read_frame_bundle(
        source,
        config=config,
        frame=frame,
        dataset=dataset,
        valid_mask=valid_mask,
        external_mask=mask,
        mask_frame=mask_frame,
        mask_dataset=mask_dataset,
    )
    image, metadata, embedded_qmap = bundle.image, bundle.metadata, bundle.qmap
    if isinstance(source, (str, os.PathLike)):
        metadata.setdefault("path", os.fspath(source))
    # Selector identity travels with the result so directory exports cannot
    # overwrite two frames from one multi-frame/dataset container.
    if frame is not None:
        metadata["frame_selector"] = int(frame)
    elif _config_value(config, "frame", None) is not None:
        metadata["frame_selector"] = int(_config_value(config, "frame", None))
    if dataset is not None:
        metadata["dataset_selector"] = str(dataset)
    elif _config_value(config, "dataset", None) is not None:
        metadata["dataset_selector"] = str(_config_value(config, "dataset", None))
    selected_poni = poni or _config_value(config, "poni_path", _config_value(config, "poni", None))
    selected_embedded_qmap = qmap if qmap is not None else embedded_qmap
    if geometry_cache is not None and selected_poni is not None and selected_embedded_qmap is None:
        if isinstance(selected_poni, (str, os.PathLike)):
            try:
                stat = Path(selected_poni).stat()
                cache_key = (os.fspath(Path(selected_poni).resolve()), stat.st_size, stat.st_mtime_ns, image.shape)
            except OSError:
                cache_key = (os.fspath(selected_poni), image.shape)
        else:
            cache_key = (id(selected_poni), image.shape)
        base_qmap = geometry_cache.get(cache_key) if isinstance(geometry_cache, Mapping) else None
        if base_qmap is None:
            base_qmap = build_qmap(
                image,
                poni=selected_poni,
                config=config,
                valid_mask=None,
            )
            if isinstance(geometry_cache, dict):
                geometry_cache[cache_key] = base_qmap
        qmap_obj = build_qmap(
            image,
            qmap=base_qmap,
            config=config,
            valid_mask=bundle.valid_mask,
        )
    else:
        qmap_obj = build_qmap(
            image,
            poni=selected_poni,
            qmap=selected_embedded_qmap,
            config=config,
            valid_mask=bundle.valid_mask,
        )
    options = _analysis_options(image, qmap_obj, config)
    domain = _analysis_domain(
        image,
        qmap_obj,
        config=config,
        detector_valid=bundle.valid_mask,
        external_mask=bundle.external_mask,
        include_config_mask=False,
    )
    # ``measure_observables`` owns the ridge and ellipse calculation.  Keep
    # ellipse fitting enabled here and adapt its result below; calling the
    # standalone ridge/ellipse solver afterwards would fit the same pixels a
    # second time and can make nested/top-level results diverge.
    observables = measure_observables(
        image,
        qmap_obj,
        config=config,
        frame=bundle.frame,
        fit_ellipse=True,
        analysis_domain=domain,
        cancel_event=cancel_event,
    )
    observable_ellipse = observables.get("ellipse") if isinstance(observables, Mapping) else None
    if isinstance(observable_ellipse, Mapping):
        # Keep the established verbose names while exposing the compact
        # longitudinal aliases used by batch/checkpoint consumers.
        observable_ellipse = dict(observable_ellipse)
        if "L_N" not in observable_ellipse and "Ln_from_minor_axis_nm" in observable_ellipse:
            observable_ellipse["L_N"] = observable_ellipse["Ln_from_minor_axis_nm"]
        if "L_z" not in observable_ellipse and "Lz_from_draw_axis_nm" in observable_ellipse:
            observable_ellipse["L_z"] = observable_ellipse["Lz_from_draw_axis_nm"]
        observables["ellipse"] = observable_ellipse
    ridges = _ridges_from_observable_bundle(observables)
    if ridges is None:
        # Compatibility fallback for an injected/legacy observable adapter
        # that does not expose its measured ridge track.  It does not trigger
        # an additional ellipse solver; the measured ellipse remains the
        # authoritative result for this frame.
        ridges = extract_ridges(
            image, qmap_obj, config=config, analysis_domain=domain
        )
    if isinstance(observables, Mapping) and isinstance(observables.get("ridge"), Mapping):
        public_ridge = dict(observables["ridge"])
        public_ridge["points"] = ridges
        observables = dict(observables)
        observables["ridge"] = public_ridge
    ellipse_fit = _public_ellipse_fit(
        observable_ellipse,
        n_points=len(ridges),
        qmap=qmap_obj,
        config=config,
    ) if observable_ellipse is not None else {
        "status": "unavailable",
        "success": False,
        "message": "measure_observables did not return an ellipse",
        "n_points": len(ridges),
        "ellipses": [],
        "parameters": {},
        "flags": ("ellipse_unavailable",),
    }
    if options["ellipse"] is not None:
        ellipse_fit["constraint_config"] = _jsonable(options["ellipse"], array_summary=False)
        ellipse_fit["flags"] = tuple(
            dict.fromkeys(tuple(ellipse_fit.get("flags", ())) + ("ellipse_constraints_active",))
        )
    # Make the nested and top-level public views literally share the same
    # adapter output.  This prevents degree/radian or alias drift at export.
    observables["ellipse"] = ellipse_fit
    # ``phi_app_deg`` is measured from the angular lobe profile.  The ellipse
    # rotation remains ``theta_deg``; no apparent fit angle is relabelled as
    # alpha or phi.
    observables.setdefault("phi_app_deg", None)
    observables.setdefault("alpha_candidate_deg", None)
    run_full2d = _config_value(config, "full2d", False) if full2d is None else bool(full2d)
    full2d_result = (
        fit_full2d(
            image,
            qmap_obj,
            ellipse_fit,
            config=config,
            frame=bundle.frame,
            initial_parameters=initial_parameters,
            analysis_domain=domain,
            cancel_event=cancel_event,
        )
        if run_full2d
        else None
    )
    if isinstance(full2d_result, Mapping) and full2d_result.get("sampled_indices") is not None:
        domain = domain.with_sampled_indices(full2d_result["sampled_indices"])
    result = PipelineResult(
        image=image,
        qmap=qmap_obj,
        observables=observables,
        ridges=ridges,
        ellipse_fit=ellipse_fit,
        full2d=full2d_result,
        metadata=metadata,
        flags={
            "empirical_model_only": True,
            "mechanism_under_determined": True,
            "forward_simulation_only": False,
            "nonunique_inverse_problem": True,
            "ellipse_constraints_active": options["ellipse"] is not None,
        },
        valid_mask=domain.fit_valid_mask,
        analysis_domain=domain,
        analysis={
            **options,
            "q_window": list(domain.q_window),
        },
    )
    if output is not None:
        result.output_paths = [os.fspath(path) for path in export_result(result, output, force=force)]
    return result


def _safe_stem(source: Any, fallback: str = "frame") -> str:
    if isinstance(source, (str, os.PathLike)):
        stem = Path(source).stem
    else:
        stem = fallback
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._")
    return stem or fallback


def _result_output_stem(result: PipelineResult) -> str:
    """Build a selector-aware directory-export stem.

    A multi-frame HDF5/NPZ/TIFF source can yield several valid results with
    the same filename stem.  Keep the historical stem for ordinary single
    frames, but append explicit selector tokens whenever they are present so
    writing a directory cannot silently target the wrong frame.
    """

    metadata = result.metadata if isinstance(result.metadata, Mapping) else {}
    stem = _safe_stem(metadata.get("path"), "frame")
    frame_value = metadata.get("frame_selector", metadata.get("frame"))
    dataset_value = metadata.get("dataset_selector", metadata.get("dataset"))
    frame_id = metadata.get("frame_id")
    tokens: list[str] = []
    if frame_value is not None and frame_value != "":
        try:
            tokens.append(f"frame_{int(frame_value):04d}")
        except (TypeError, ValueError):
            tokens.append(f"frame_{_safe_stem(frame_value, 'selected')}")
    if dataset_value is not None and str(dataset_value).strip():
        tokens.append(f"dataset_{_safe_stem(dataset_value, 'selected')}")
    if frame_id is not None and str(frame_id).strip() and str(frame_id) != Path(str(metadata.get("path", ""))).stem:
        tokens.append(f"id_{_safe_stem(frame_id, 'selected')}")
    parent_token = metadata.get("source_parent_token")
    if parent_token is not None and str(parent_token).strip():
        tokens.append(f"src_{_safe_stem(parent_token, 'source')}")
    return stem if not tokens else f"{stem}__{'__'.join(tokens)}"


def _result_arrays(result: PipelineResult) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {"image": np.asarray(result.image)}
    if result.valid_mask is not None:
        arrays["valid_mask"] = np.asarray(result.valid_mask, dtype=bool)
    if result.analysis_domain is not None:
        for name in (
            "finite_mask",
            "detector_valid_mask",
            "external_valid_mask",
            "q_window_mask",
            "roi_exclusion_mask",
            "weight_valid_mask",
            "fit_valid_mask",
            "sampled_valid_mask",
        ):
            arrays[name] = np.asarray(
                getattr(result.analysis_domain, name),
                dtype=bool,
            )
    for name, value in result.analysis_arrays.items():
        array = np.asarray(value)
        if array.shape == result.image.shape:
            arrays[name] = array
    if isinstance(result.qmap, Mapping):
        for key in ("qx", "qy", "q", "theta"):
            if key in result.qmap:
                value = np.asarray(result.qmap[key])
                if value.shape == result.image.shape:
                    arrays[key] = value
    if result.ridges:
        for key in ("qx", "qy", "q", "intensity", "theta_deg"):
            values = [row[key] for row in result.ridges if isinstance(row, Mapping) and key in row]
            if values:
                arrays[f"ridge_{key}"] = np.asarray(values, dtype=float)
    for index, profile in enumerate(result.lobe_radial_profiles or ()):
        mapping = _as_mapping(profile)
        if not mapping:
            continue
        for key in ("angle_deg", "q", "intensity", "counts", "candidate_counts", "coverage"):
            if key not in mapping:
                continue
            value = np.asarray(mapping[key])
            if value.ndim:
                arrays[f"lobe_radial_profile_{index}_{key}"] = value
    for index, peak in enumerate(result.lobe_radial_peaks or ()):
        mapping = _as_mapping(peak)
        if not mapping:
            continue
        for key in ("angle_deg", "q", "q_star", "intensity", "baseline", "snr", "radial_fwhm", "area"):
            if key not in mapping:
                continue
            try:
                arrays[f"lobe_radial_peak_{index}_{key}"] = np.asarray(mapping[key], dtype=float)
            except (TypeError, ValueError):
                continue
    if isinstance(result.full2d, Mapping):
        for key in ("model", "residual", "model_image", "residual_image"):
            value = result.full2d.get(key)
            if isinstance(value, np.ndarray):
                arrays[f"full2d_{key}"] = value
    return arrays


def export_result(
    result: PipelineResult | Mapping[str, Any],
    output: str | os.PathLike[str],
    *,
    force: bool = False,
) -> list[Path]:
    """Export JSON/NPZ (or CSV) while refusing accidental overwrite.

    A directory target creates ``<stem>.json`` and ``<stem>.npz``.  A target
    ending in ``.json``, ``.npz``, or ``.csv`` writes just that representation.
    """

    if not isinstance(result, PipelineResult):
        mapping = dict(result)
        image = np.asarray(mapping.get("image", mapping.get("data")))
        metadata = dict(mapping.get("metadata", {}) or {})
        if mapping.get("frame_selector", mapping.get("frame")) is not None:
            metadata.setdefault(
                "frame_selector", mapping.get("frame_selector", mapping.get("frame"))
            )
        if mapping.get("dataset", mapping.get("dataset_selector")) is not None:
            metadata.setdefault(
                "dataset_selector", mapping.get("dataset", mapping.get("dataset_selector"))
            )
        result = PipelineResult(
            image=image,
            qmap=mapping.get("qmap", {}),
            observables=dict(mapping.get("observables", {})),
            ridges=list(mapping.get("ridges", [])),
            ellipse_fit=dict(mapping.get("ellipse_fit", mapping.get("ellipse", {}))),
            full2d=mapping.get("full2d"),
            metadata=metadata,
            flags=dict(mapping.get("flags", {})),
            valid_mask=(
                np.asarray(mapping["valid_mask"], dtype=bool)
                if mapping.get("valid_mask") is not None
                else None
            ),
            analysis=dict(mapping.get("analysis", {}) or {}),
            analysis_arrays={
                str(name): np.asarray(mapping[name], dtype=bool)
                for name in (
                    "finite_mask",
                    "detector_valid_mask",
                    "external_valid_mask",
                    "q_window_mask",
                    "roi_exclusion_mask",
                    "weight_valid_mask",
                    "fit_valid_mask",
                    "sampled_valid_mask",
                )
                if name in mapping
            },
        )
    target = Path(output)
    suffix = target.suffix.lower()
    if suffix in {".json", ".npz", ".csv"}:
        paths = [target]
    else:
        stem = _result_output_stem(result)
        paths = [target / f"{stem}.json", target / f"{stem}.npz"]
    existing = [path for path in paths if path.exists()]
    if existing and not force:
        joined = "、".join(os.fspath(path) for path in existing)
        raise FileExistsError(f"输出已存在，未覆盖：{joined}（需要 --force）")
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)

    summary = result.to_mapping()
    written: list[Path] = []
    for path in paths:
        if path.suffix.lower() == ".json":
            path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        elif path.suffix.lower() == ".npz":
            arrays = _result_arrays(result)
            # Observable profiles are modest in size compared with detector
            # arrays.  Keep them lossless in the NPZ sidecar so annulus/lobe
            # q bins, counts and coverage survive JSON-summary compaction.
            arrays["observables_json"] = np.asarray(
                json.dumps(
                    _jsonable(result.observables, array_summary=False),
                    ensure_ascii=False,
                )
            )
            np.savez_compressed(path, **arrays)
        elif path.suffix.lower() == ".csv":
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(("parameter", "value"))
                def csv_value(value: Any) -> Any:
                    converted = _jsonable(value, array_summary=False)
                    if isinstance(converted, (Mapping, list, tuple)):
                        converted = json.dumps(
                            converted,
                            ensure_ascii=False,
                            allow_nan=False,
                            separators=(",", ":"),
                        )
                    return safe_csv_cell(converted)

                for key, value in result.observables.items():
                    writer.writerow(
                        (
                            safe_csv_cell(key),
                            csv_value(value),
                        )
                    )
                for key, value in result.ellipse_fit.get("parameters", {}).items():
                    writer.writerow(
                        (
                            safe_csv_cell(f"ellipse.{key}"),
                            csv_value(value),
                        )
                    )
        else:
            raise PipelineError(f"不支持的输出格式：{path.suffix}")
        written.append(path)
    return written


def _expand_inputs(inputs: Iterable[Any]) -> list[Any]:
    expanded: list[Any] = []
    for value in inputs:
        if not isinstance(value, (str, os.PathLike)):
            expanded.append(value)
            continue
        text = os.fspath(value)
        if any(char in text for char in "*?[]"):
            expanded.extend(sorted(filter_supported_image_paths(glob.glob(text))))
        else:
            expanded.append(value)
    return expanded


def batch_analyze(
    inputs: Iterable[Any],
    *,
    poni: str | os.PathLike[str] | None = None,
    config: Any = None,
    output_dir: str | os.PathLike[str] | None = None,
    full2d: bool | None = None,
    force: bool = False,
    manifest: Any = None,
    series: str | None = None,
    start: int | None = None,
    stop: int | None = None,
    stride: int = 1,
    frame_range: str | Sequence[int] | None = None,
    bounded: bool = False,
    checkpoint: str | os.PathLike[str] | None = None,
    resume: bool = False,
    cancel_event: Any = None,
    progress: Any = None,
    result_sink: Any = None,
) -> list[PipelineResult] | BatchRunResult:
    """Analyze a sequence of frames and optionally export one bundle per frame."""

    from .batch import build_frame_refs, parse_frame_range, select_frame_refs

    values = _expand_inputs(inputs)
    if not values:
        raise PipelineError("没有可处理的输入帧")
    refs = build_frame_refs(values, manifest=manifest)
    if frame_range is not None:
        range_start, range_stop, range_stride = parse_frame_range(frame_range)
        if start is not None or stop is not None or stride != 1:
            raise PipelineError("frame_range cannot be combined with start/stop/stride")
        start, stop, stride = range_start, range_stop, range_stride
    refs = select_frame_refs(refs, series=series, start=start, stop=stop, stride=stride)
    if not refs:
        raise PipelineError("批量选择没有匹配任何输入帧")
    geometry_cache: dict[Any, Any] = {}
    destination = output_dir or _config_value(config, "output_dir", _config_value(config, "output", None))
    stem_counts: dict[str, int] = {}
    for ref in refs:
        stem_counts[ref.path.stem.casefold()] = stem_counts.get(ref.path.stem.casefold(), 0) + 1

    def analyze_for_batch(
        ref: FrameRef,
        initial_parameters: Any = None,
        config: Any = None,
    ) -> PipelineResult:
        result = analyze_frame(
            ref.path,
            poni=poni,
            config=config,
            full2d=full2d,
            initial_parameters=initial_parameters,
            frame=ref.frame_selector,
            dataset=ref.dataset_id or None,
            geometry_cache=geometry_cache,
            cancel_event=cancel_event,
        )
        result.metadata["frame_id"] = ref.id
        if stem_counts.get(ref.path.stem.casefold(), 0) > 1:
            result.metadata["source_parent_token"] = ref.path.parent.name
        if ref.source is not None:
            result.metadata["series"] = ref.source
        if destination is not None:
            target = Path(destination)
            result.output_paths = [
                os.fspath(path)
                for path in export_result(result, target, force=force or resume)
            ]
        return result

    run = run_batch_frames(
        refs,
        analyze_for_batch,
        mode=str(_config_value(config, "batch_mode", "independent")),
        config=config,
        checkpoint=checkpoint,
        resume=resume,
        series=series,
        start=None,
        stop=None,
        stride=1,
        cancel_event=cancel_event,
        progress=progress,
        result_sink=result_sink,
        retain_results=result_sink is None,
        allow_mixed_series=series is not None,
    )
    if bounded:
        return run
    results: list[PipelineResult] = []
    for item in run.frame_results:
        if item.traceback:
            raise PipelineError(item.error or "batch analyzer failed")
        if item.result is not None:
            results.append(item.result)
    return results


def batch_analyze_bounded(*args: Any, **kwargs: Any) -> BatchRunResult:
    """Explicit common-runner API for bounded batch processing."""

    kwargs["bounded"] = True
    return batch_analyze(*args, **kwargs)  # type: ignore[return-value]


def _run_project_bounded(
    project: ProjectConfig | str | os.PathLike[str],
    *,
    force: bool = False,
) -> BatchRunResult:
    """Run a project TOML with per-frame failure and quality isolation."""

    if isinstance(project, (str, os.PathLike)):
        project_path = Path(project)
        config = load_project(project_path)
        config = config.resolve_paths(project_path.parent)
    else:
        config = project
    if not isinstance(config, ProjectConfig):
        config = ProjectConfig.from_mapping(config)
    if not config.input_paths:
        raise PipelineError("项目配置没有 inputs.files/input_paths")

    values = _expand_inputs(config.input_paths)
    if not values:
        raise PipelineError("项目输入通配符没有匹配任何文件")
    analysis = config.analysis
    mode = str(analysis.get("batch_mode", analysis.get("mode", "independent")))
    manifest = analysis.get("manifest")
    checkpoint = analysis.get("checkpoint")
    resume = bool(analysis.get("resume", False))
    series = analysis.get("series")
    start = analysis.get("start")
    stop = analysis.get("stop")
    stride = analysis.get("stride", 1)
    frame_range = analysis.get("frame_range")
    destination = Path(config.output_dir)
    project_config = config
    geometry_cache: dict[Any, Any] = {}
    project_stem_counts: dict[str, int] = {}
    for value in values:
        candidate_path = Path(value.path if isinstance(value, FrameRef) else value)
        key = candidate_path.stem.casefold()
        project_stem_counts[key] = project_stem_counts.get(key, 0) + 1
    batch_config = ProjectConfig(
        input_paths=project_config.input_paths,
        poni_path=project_config.poni_path,
        output_dir=project_config.output_dir,
        q_unit=project_config.q_unit,
        full2d=project_config.full2d,
        analysis={
            key: value
            for key, value in project_config.analysis.items()
            if key not in {"batch_mode", "mode", "manifest", "checkpoint", "resume", "stream", "stream_results"}
        },
        export=project_config.export,
        metadata=project_config.metadata,
    )
    stream_writer = None
    if bool(analysis.get("stream", analysis.get("stream_results", False))):
        from .export import StreamingBatchExporter

        stream_writer = StreamingBatchExporter(
            destination,
            provenance={"source": "project", "stream": True},
            force=bool(force or resume),
            resume=resume,
        )

    def analyze_for_project(
        frame_ref: FrameRef,
        initial_parameters: Any = None,
        config: Any = None,
    ) -> PipelineResult:
        active_config = config if config is not None else batch_config
        result = analyze_frame(
            frame_ref.path,
            poni=project_config.poni_path,
            config=active_config,
            full2d=project_config.full2d,
            initial_parameters=initial_parameters,
            frame=frame_ref.frame_selector,
            dataset=frame_ref.dataset_id or None,
            geometry_cache=geometry_cache,
        )
        if isinstance(result.metadata, Mapping):
            result.metadata["frame_id"] = frame_ref.id
            if project_stem_counts.get(frame_ref.path.stem.casefold(), 0) > 1:
                result.metadata["source_parent_token"] = frame_ref.path.parent.name
            if frame_ref.source is not None:
                result.metadata["series"] = frame_ref.source
        if stream_writer is None:
            result.output_paths = [
                os.fspath(path)
                for path in export_result(result, destination, force=bool(force or resume))
            ]
        return result
    try:
        run = run_batch_frames(
            values,
            analyze_for_project,
            mode=mode,
            config=batch_config,
            manifest=manifest,
            checkpoint=checkpoint,
            resume=resume,
            series=series,
            start=start,
            stop=stop,
            stride=stride,
            frame_range=frame_range,
            result_sink=None if stream_writer is None else stream_writer.write,
            retain_results=stream_writer is None,
        )
        if stream_writer is not None:
            stream_writer.finalize(run)
        return run
    except Exception:
        if stream_writer is not None:
            stream_writer.abort()
        raise


class LegacyProjectResults(list[Any]):
    """Historical list-shaped project result with batch metadata attached."""

    def __init__(self, run: BatchRunResult) -> None:
        super().__init__(
            item.result for item in run.frame_results if item.result is not None
        )
        self._run = run
        self.frame_results = run.frame_results

    def __getattr__(self, name: str) -> Any:
        return getattr(self._run, name)

    @property
    def successful(self) -> list[Any]:
        return list(self._run.successful)

    @property
    def failures(self) -> list[Any]:
        return list(self._run.failures)


def run_project(
    project: ProjectConfig | str | os.PathLike[str],
    *,
    force: bool = False,
) -> LegacyProjectResults:
    """Preserve the historical list-shaped project API by default."""

    return LegacyProjectResults(_run_project_bounded(project, force=force))


def run_project_bounded(
    project: ProjectConfig | str | os.PathLike[str],
    *,
    force: bool = False,
) -> BatchRunResult:
    """Versioned name for the resumable common project runner."""

    return _run_project_bounded(project, force=force)


def run_project_legacy(
    project: ProjectConfig | str | os.PathLike[str],
    *,
    force: bool = False,
) -> list[Any]:
    """Compatibility adapter returning the historical per-frame list."""

    run = _run_project_bounded(project, force=force)
    return [item.result for item in run.frame_results if item.result is not None]


def synthetic_butterfly(
    shape: tuple[int, int] | Sequence[int] = (128, 128),
    *,
    q0: float = 28.0,
    width: float = 2.0,
    ellipticity: float = 2.0,
    angle_deg: float = 28.0,
    amplitude: float = 1.0,
    background: float = 0.01,
    noise: float = 0.0,
    seed: int = 0,
    q_scale: float = 1.0,
    return_qmap: bool = False,
) -> Any:
    """Generate a deterministic two-lobe butterfly pattern for smoke tests."""

    if len(shape) != 2 or int(shape[0]) < 4 or int(shape[1]) < 4:
        raise PipelineError("synthetic 图像 shape 必须是至少 4×4 的二维尺寸")
    height, width_pixels = int(shape[0]), int(shape[1])
    if q0 <= 0 or width <= 0 or ellipticity <= 0 or q_scale <= 0:
        raise PipelineError("synthetic 的 q0、width、ellipticity、q_scale 必须为正数")
    y, x = np.indices((height, width_pixels), dtype=float)
    cx, cy = (width_pixels - 1) / 2.0, (height - 1) / 2.0
    qx, qy = (x - cx) * q_scale, (y - cy) * q_scale
    q = np.hypot(qx, qy)
    theta = np.arctan2(qy, qx)
    image = np.full_like(q, float(background), dtype=float)
    elongation = max(float(ellipticity), np.finfo(float).eps)
    angle = math.radians(float(angle_deg))
    for sign in (1.0, -1.0):
        delta = theta - sign * angle
        denominator = np.sqrt((np.cos(delta) / elongation) ** 2 + np.sin(delta) ** 2)
        target_q = q0 / np.maximum(denominator, np.finfo(float).eps)
        image += amplitude * np.exp(-0.5 * ((q - target_q) / width) ** 2)
    # A small central beam-stop-compatible diffuse component keeps moments
    # defined for very small fixtures without dominating the lobes.
    image += 0.03 * amplitude * np.exp(-0.5 * (q / max(width * 2.0, 1e-9)) ** 2)
    if noise:
        rng = np.random.default_rng(seed)
        image += rng.normal(0.0, float(noise), size=image.shape)
    image = np.maximum(image, 0.0)
    qmap = {"qx": qx, "qy": qy, "q": q, "q_unit": "pixel-q"}
    return (image, qmap) if return_qmap else image


def synthetic_frame(*args: Any, **kwargs: Any) -> Any:
    """Compatibility alias for :func:`synthetic_butterfly`."""

    return synthetic_butterfly(*args, **kwargs)


def generate_synthetic(*args: Any, **kwargs: Any) -> Any:
    return synthetic_butterfly(*args, **kwargs)


def launch_gui(*args: Any, **kwargs: Any) -> Any:
    """Delegate to an optional GUI while keeping the CLI seam stable."""

    launcher = _find_callable(
        ("butterfly_saxs.gui", "butterfly_saxs.gui.app", "butterfly_saxs.ui"),
        ("launch", "run", "main"),
    )
    if launcher is None:
        raise PipelineError("GUI 模块未安装；请安装项目的 ui 可选依赖")
    try:
        return launcher(*args, **kwargs)
    except TypeError:
        # The bundled Qt workbench exposes ``launch(argv=None)``.  Keep the
        # richer keyword seam for GUI implementations that support it, and
        # degrade to a normal argv launch for this minimal contract.
        argv: list[str] = []
        input_path = kwargs.get("input_path", kwargs.get("input"))
        if input_path:
            argv.append(os.fspath(input_path))
        config_path = kwargs.get("config_path")
        if config_path:
            argv.extend(("--config", os.fspath(config_path)))
        poni = kwargs.get("poni")
        if poni and isinstance(poni, (str, os.PathLike)):
            argv.extend(("--poni", os.fspath(poni)))
        frame = kwargs.get("frame")
        if frame is not None:
            argv.extend(("--frame", str(frame)))
        dataset = kwargs.get("dataset")
        if dataset:
            argv.extend(("--dataset", str(dataset)))
        return launcher(argv or None)


def gui_seam(*args: Any, **kwargs: Any) -> Any:
    return launch_gui(*args, **kwargs)


# Short names mirror the CLI verbs and make notebook/GUI wiring discoverable.
inspect = inspect_frame
analyze = analyze_frame
batch = batch_analyze
synthetic = synthetic_butterfly
gui = launch_gui


__all__ = [
    "PipelineError",
    "PipelineResult",
    "LegacyProjectResults",
    "read_frame",
    "build_qmap",
    "measure_observables",
    "extract_ridges",
    "fit_symmetric_ellipses",
    "fit_full2d",
    "inspect_frame",
    "analyze_frame",
    "batch_analyze",
    "batch_analyze_bounded",
    "run_project",
    "run_project_bounded",
    "run_project_legacy",
    "export_result",
    "synthetic_butterfly",
    "synthetic_frame",
    "generate_synthetic",
    "launch_gui",
    "gui_seam",
    "inspect",
    "analyze",
    "batch",
    "synthetic",
    "gui",
]
