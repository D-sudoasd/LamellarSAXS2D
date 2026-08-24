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

from .project import ProjectConfig, load_project


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
    """Call an optional adapter across the small API variants used in this repo."""

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
    if signature is not None:
        accepts_kwargs = any(
            parameter.kind == _inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        kwargs = {
            name: value
            for name, value in values.items()
            if value is not None
            and (accepts_kwargs or name in signature.parameters)
        }
        try:
            return fn(**kwargs)
        except TypeError:
            # A positional-only function is common in small scientific helper
            # modules.  Fall through to the conservative forms below.
            pass
    attempts = [
        ((image, qmap), {}),
        ((image,), {"qmap": qmap}),
        ((image,), {}),
    ]
    last_error: TypeError | None = None
    for args, kwargs in attempts:
        if qmap is None and args == (image, qmap):
            continue
        try:
            return fn(*args, **kwargs)
        except TypeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    return fn(image)


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
        return result
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
        return result
    attrs: dict[str, Any] = {}
    aliases = {
        "qx": ("qx", "qx_nm_inv"),
        "qy": ("qy", "qy_nm_inv"),
        "q": ("q", "q_nm_inv", "radius"),
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
        return {"object": value, **attrs}
    array = np.asarray(value)
    if array.shape == shape + (2,):
        return {"qx": array[..., 0], "qy": array[..., 1]}
    raise PipelineError("qmap 必须包含 qx/qy 数组，或形状为 (高, 宽, 2) 的数组")


def _qmap_arrays(qmap: Any, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    qmap = _coerce_qmap(qmap, shape)
    if qmap is None:
        raise PipelineError("缺少 qmap")
    qx = qmap.get("qx", qmap.get("qx_nm_inv"))
    qy = qmap.get("qy", qmap.get("qy_nm_inv"))
    if qx is None or qy is None:
        q = qmap.get("q", qmap.get("q_nm_inv", qmap.get("radius")))
        if q is None:
            raise PipelineError("qmap 缺少 qx/qy（或 q/radius）数组")
        q = np.asarray(q, dtype=float)
        if q.shape != shape:
            raise PipelineError(f"qmap 与图像形状不一致：{q.shape} != {shape}")
        y, x = np.indices(shape, dtype=float)
        cx = float(np.nanmean(x))
        cy = float(np.nanmean(y))
        qx = q * (x - cx) / np.maximum(np.hypot(x - cx, y - cy), 1e-12)
        qy = q * (y - cy) / np.maximum(np.hypot(x - cx, y - cy), 1e-12)
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


def _combine_valid_masks(
    shape: tuple[int, int],
    *,
    valid_masks: Iterable[Any] = (),
    masks: Iterable[Any] = (),
) -> np.ndarray | None:
    """Combine positive and negative mask conventions without overwriting.

    Frame sources and project configuration can each provide both forms.  The
    IO layer is the single source of truth for path loading and exact-shape
    validation; this adapter only applies the logical intersection/union over
    all supplied values.
    """

    from .io import combine_masks

    combined: np.ndarray | None = None
    try:
        for value in valid_masks:
            if value is None:
                continue
            current = combine_masks(shape, valid_mask=value)
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
) -> _FrameBundle:
    """Read one frame while retaining ``LoadedImage.valid_mask`` and qmap mask."""

    configured_frame = frame if frame is not None else _config_value(config, "frame", None)
    configured_dataset = dataset if dataset is not None else _config_value(config, "dataset", None)
    configured_valid_mask = valid_mask if valid_mask is not None else _config_value(config, "valid_mask", None)
    configured_mask = external_mask if external_mask is not None else _config_value(config, "mask", None)
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
            masks=(*source_masks, configured_mask),
        )
        frame = _loaded_frame(data, metadata=metadata, valid_mask=valid)
        return _FrameBundle(
            frame=frame,
            image=image,
            metadata=metadata,
            qmap=source.get("qmap"),
            valid_mask=getattr(frame, "valid_mask", None),
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
            masks=(*source_masks, configured_mask),
        )
        frame = _loaded_frame(data, metadata=metadata, valid_mask=valid)
        return _FrameBundle(
            frame=frame,
            image=image,
            metadata=metadata,
            qmap=getattr(source, "qmap", None),
            valid_mask=getattr(frame, "valid_mask", valid),
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
            "external_mask": configured_mask,
        }
        reader_kwargs = {key: value for key, value in reader_kwargs.items() if value is not None}
        loaded = load_image(path, **reader_kwargs) if reader_kwargs else load_image(path)
    except Exception as exc:  # noqa: BLE001 - normalize strict IO errors
        raise PipelineError(f"读取输入图像失败：{path}（{exc}）") from exc

    data = _coerce_array(getattr(loaded, "data", loaded))
    metadata = dict(getattr(loaded, "metadata", {}) or {})
    metadata.setdefault("path", os.fspath(path))
    valid = getattr(loaded, "valid_mask", None)
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
            configured_mask = _config_value(config, "mask", None)
            result = _call_adapter(
                adapter,
                image=image,
                config=config,
                poni=poni,
                valid_mask=valid_mask,
                mask=configured_mask,
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
    return result


def _q_window(image: np.ndarray, qmap: Any, config: Any = None) -> tuple[float, float] | Any:
    configured = _config_value(config, "q_window", _config_value(config, "q_range", None))
    if configured is not None:
        return configured
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

    curvature = _config_value(config, "curvature", None)
    if isinstance(curvature, Mapping):
        curvature_sigma = curvature.get("sigma", curvature.get("smooth_sigma", 2.0))
        curvature_percentile = curvature.get("percentile", 25.0)
        curvature_normal_step = curvature.get("normal_step", 1.0)
    elif curvature is not None and not isinstance(curvature, bool):
        curvature_sigma, curvature_percentile, curvature_normal_step = curvature, 25.0, 1.0
    else:
        curvature_sigma = _config_value(config, "curvature_sigma", 2.0)
        curvature_percentile = _config_value(config, "curvature_percentile", 25.0)
        curvature_normal_step = _config_value(config, "curvature_normal_step", 1.0)
    try:
        n_angles = max(4, int(_config_value(config, "n_angles", _config_value(config, "n_ridge_angles", 72))))
    except (TypeError, ValueError):
        n_angles = 72
    try:
        n_angular_bins = max(12, int(_config_value(config, "n_angular_bins", _config_value(config, "ridge_bins", 360))))
    except (TypeError, ValueError):
        n_angular_bins = 360
    try:
        n_radial_bins = max(32, int(_config_value(config, "n_radial_bins", 256)))
    except (TypeError, ValueError):
        n_radial_bins = 256
    return {
        "q_window": _q_window(image, qmap, config),
        "mask": _config_value(config, "mask", None),
        "ridge_method": str(_config_value(config, "ridge_method", "radial_peak")),
        "n_angles": n_angles,
        "n_angular_bins": n_angular_bins,
        "n_radial_bins": n_radial_bins,
        "draw_axis_deg": float(_config_value(config, "draw_axis_deg", 90.0)),
        "curvature_sigma": float(curvature_sigma),
        "curvature_percentile": float(curvature_percentile),
        "curvature_normal_step": float(curvature_normal_step),
    }


def _observable_mask(value: Any, shape: tuple[int, int], *, qmap: Any = None, config: Any = None) -> Any:
    """Resolve detector masks and serializable q/pixel ROIs to True=invalid."""

    masks: list[np.ndarray] = []
    if value is not None:
        if isinstance(value, (str, os.PathLike)):
            try:
                from .io import combine_masks

                valid = combine_masks(shape, external_mask=value)
                if valid is not None:
                    masks.append(~valid)
            except Exception as exc:  # noqa: BLE001 - report a useful config error
                raise PipelineError(f"无法读取 analysis.mask：{value}（{exc}）") from exc
        else:
            array = np.asarray(value, dtype=bool)
            if array.shape != shape:
                raise PipelineError(f"analysis.mask 形状 {array.shape} 与图像 {shape} 不一致")
            masks.append(array)
    rois = _config_value(config, "rois", ())
    if rois:
        try:
            from .masking import combine_exclusion_masks

            qx = qy = None
            if qmap is not None:
                qx, qy, _ = _qmap_arrays(qmap, shape)
            masks.append(combine_exclusion_masks(shape, rois=rois, qx=qx, qy=qy))
        except Exception as exc:  # noqa: BLE001 - fail closed for ROI config
            raise PipelineError(f"无法解析 analysis.rois：{exc}") from exc
    if not masks:
        return None
    return np.logical_or.reduce(masks)


def measure_observables(
    image: np.ndarray,
    qmap: Any,
    *,
    config: Any = None,
    frame: Any = None,
    fit_ellipse: bool = True,
) -> dict[str, Any]:
    """Measure angular/lobe/ridge observables with the declared config."""

    from . import observables as observable_module

    options = _analysis_options(image, qmap, config)
    options["mask"] = _observable_mask(options["mask"], image.shape, qmap=qmap, config=config)
    observed_frame = frame if frame is not None else _loaded_frame(image)
    result = observable_module.measure_observables(
        observed_frame,
        qmap,
        options["q_window"],
        n_angular_bins=options["n_angular_bins"],
        n_ridge_angles=options["n_angles"],
        n_radial_bins=options["n_radial_bins"],
        fit_ellipse=bool(fit_ellipse),
        mask=options["mask"],
        ridge_method=options["ridge_method"],
        draw_axis_deg=options["draw_axis_deg"],
        curvature_sigma=options["curvature_sigma"],
        curvature_percentile=options["curvature_percentile"],
        curvature_normal_step=options["curvature_normal_step"],
    )
    # Do not infer alpha/phi from the fitted ellipse rotation.  The papers'
    # microscopic tilts are not identifiable from this apparent trajectory.
    return _public_angles(_as_mapping(result))


def extract_ridges(image: np.ndarray, qmap: Any, *, config: Any = None) -> list[dict[str, float]]:
    """Extract the observed radial ridge with all configured safeguards."""

    from . import observables as observable_module

    options = _analysis_options(image, qmap, config)
    options["mask"] = _observable_mask(options["mask"], image.shape, qmap=qmap, config=config)
    frame = _loaded_frame(image)
    track = observable_module.measure_radial_ridges(
        frame,
        qmap,
        options["q_window"],
        n_angles=options["n_angles"],
        n_bins=options["n_radial_bins"],
        mask=options["mask"],
        ridge_method=options["ridge_method"],
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

    if not isinstance(observables, Mapping):
        return None
    ridge = observables.get("ridge")
    if not isinstance(ridge, Mapping):
        return None
    raw_points = ridge.get("points")
    if not isinstance(raw_points, Sequence) or isinstance(raw_points, (str, bytes)):
        return None
    points: list[dict[str, Any]] = []
    for item in raw_points:
        mapping = _as_mapping(item)
        if not mapping:
            continue
        point = {str(key): _jsonable(value, array_summary=False) for key, value in mapping.items()}
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
    """Adapt one already-computed observable ellipse without solving again.

    ``observables.measure_observables`` performs the canonical solver once.
    Both the high-level pipeline and external ridge callers use this adapter
    to expose the stable degree/public schema, so a nested observable and the
    top-level ``ellipse_fit`` cannot silently diverge through a second fit.
    """

    values = _ellipse_value(fit, "parameter_values", {})
    if not isinstance(values, Mapping) or not values:
        values = _ellipse_value(fit, "values", {})
    if not isinstance(values, Mapping):
        values = {}
    a = _as_float(values.get("a", _ellipse_value(fit, "a", np.nan)))
    ratio = _as_float(
        values.get(
            "axis_ratio",
            _ellipse_value(fit, "axis_ratio", _ellipse_value(fit, "axes_ratio", np.nan)),
        )
    )
    b = _as_float(values.get("b", _ellipse_value(fit, "b", np.nan)))
    if not np.isfinite(b) and np.isfinite(a * ratio):
        b = float(a * ratio)
    if not np.isfinite(ratio) and np.isfinite(a) and a != 0:
        ratio = float(b / a)

    theta_value = _ellipse_value(fit, "theta_deg", None)
    if theta_value is None:
        theta_value = values.get("theta_deg")
    if theta_value is None:
        theta_value = np.degrees(_as_float(values.get("theta", np.nan)))
    theta_deg = _as_float(theta_value)

    centre = _ellipse_value(fit, "center", None)
    if centre is None:
        centre = (
            values.get("cx", _ellipse_value(fit, "center_qx", 0.0)),
            values.get("cy", _ellipse_value(fit, "center_qy", 0.0)),
        )
    try:
        cx, cy = float(centre[0]), float(centre[1])
    except (IndexError, TypeError, ValueError):
        cx = _as_float(_ellipse_value(fit, "center_qx", values.get("cx", 0.0)), 0.0)
        cy = _as_float(_ellipse_value(fit, "center_qy", values.get("cy", 0.0)), 0.0)

    reference_axis_deg = _as_float(
        _ellipse_value(fit, "reference_axis_deg", float("nan")),
        float("nan"),
    )
    if not np.isfinite(reference_axis_deg):
        reference_axis_deg = float(_config_value(config, "draw_axis_deg", 90.0)) - 90.0

    fit_q_unit = _ellipse_value(fit, "q_unit", None)
    q_unit = str(fit_q_unit) if fit_q_unit is not None else _qmap_unit(qmap, config)
    l_n = _as_float(
        _ellipse_value(fit, "Ln_from_minor_axis_nm", _ellipse_value(fit, "L_N", np.nan))
    )
    l_z = _as_float(
        _ellipse_value(fit, "Lz_from_draw_axis_nm", _ellipse_value(fit, "L_z", np.nan))
    )
    eccentricity = _as_float(
        _ellipse_value(
            fit,
            "eccentricity",
            _ellipse_value(fit, "ellipticity", np.sqrt(max(0.0, 1.0 - ratio * ratio))),
        )
    )
    common = {
        "a": a,
        "b": b,
        "semi_major": a,
        "semi_minor": b,
        "axis_ratio": ratio,
        "center_qx": cx,
        "center_qy": cy,
        "reference_axis_deg": reference_axis_deg,
        "ellipse_axis_tilt_deg": theta_deg,
        "q_unit": q_unit,
        "eccentricity": eccentricity,
        "ellipticity": eccentricity,
        "L_N": l_n,
        "L_z": l_z,
        "Ln_from_minor_axis_nm": l_n,
        "Lz_from_draw_axis_nm": l_z,
    }
    coverage = _as_mapping(_ellipse_value(fit, "coverage", None))
    rmse = _as_float(_ellipse_value(fit, "rmse", np.nan))
    members: list[dict[str, Any]] = []
    for member in _ellipse_value(fit, "ellipses", ()) or ():
        member_mapping = _as_mapping(member)
        member_theta = member_mapping.get("theta_deg")
        if member_theta is None:
            member_theta = np.degrees(_as_float(member_mapping.get("theta", np.nan)))
        members.append(dict(common, theta_deg=_as_float(member_theta)))
    if not members:
        members = [dict(common, theta_deg=theta_deg), dict(common, theta_deg=-theta_deg)]
    parameters = dict(common, theta_deg=theta_deg)
    raw_flags = _ellipse_value(fit, "flags", ()) or ()
    if isinstance(raw_flags, str):
        raw_flags = (raw_flags,)
    configured_flags = _config_value(config, "flags", ()) or ()
    if isinstance(configured_flags, str):
        configured_flags = (configured_flags,)
    public_flags = tuple(
        dict.fromkeys(
            (
                "apparent_geometry_only",
                "nonunique_inverse_problem",
                *(str(item) for item in raw_flags),
                *(str(item) for item in configured_flags),
            )
        )
    )
    success = bool(_ellipse_value(fit, "success", False))
    status = _ellipse_value(fit, "status", None)
    if status is None:
        status = "ok" if success else "failed"
    residuals = np.asarray(_ellipse_value(fit, "residuals", np.asarray([])), dtype=float)
    stderr = _ellipse_value(fit, "stderr", {}) or {}
    condition = _as_float(
        _ellipse_value(fit, "condition_number", _ellipse_value(fit, "condition", np.nan))
    )
    return {
        "status": str(status),
        "success": success,
        "message": str(_ellipse_value(fit, "message", "")),
        "n_points": int(_ellipse_value(fit, "n_points", coverage.get("n_points", n_points))),
        "ellipses": members,
        "parameters": parameters,
        "a": a,
        "b": b,
        "axis_ratio": ratio,
        "theta_deg": theta_deg,
        "ellipse_axis_tilt_deg": theta_deg,
        "reference_axis_deg": reference_axis_deg,
        "eccentricity": eccentricity,
        "ellipticity": eccentricity,
        "q_unit": q_unit,
        "L_N": l_n,
        "L_z": l_z,
        "Ln_from_minor_axis_nm": l_n,
        "Lz_from_draw_axis_nm": l_z,
        "rmse": rmse,
        "residual_rms": rmse,
        "residuals": residuals,
        "stderr": dict(stderr),
        "coverage": coverage,
        "condition": condition,
        "flags": public_flags,
    }


def fit_symmetric_ellipses(
    points: Any,
    *,
    config: Any = None,
    qmap: Any = None,
) -> dict[str, Any]:
    """Fit a shared-centre pair and expose only explicit public quantities."""

    from . import observables as observable_module

    if isinstance(points, Mapping):
        points = points.get("points", points.get("ridges", points))
    rows: list[tuple[float, float]] = []
    for item in points or []:
        if isinstance(item, Mapping):
            if item.get("valid") is False:
                continue
            x, y = item.get("qx", item.get("x")), item.get("qy", item.get("y"))
        else:
            x, y = getattr(item, "qx", getattr(item, "x", None)), getattr(item, "qy", getattr(item, "y", None))
        try:
            if x is not None and y is not None and math.isfinite(float(x)) and math.isfinite(float(y)) and np.hypot(float(x), float(y)) > 0:
                rows.append((float(x), float(y)))
        except (TypeError, ValueError):
            continue
    if len(rows) < 5:
        return {"status": "insufficient_data", "n_points": len(rows), "ellipses": [], "rmse": float("nan")}
    values_cfg = _config_value(config, "ellipse", None)
    kwargs: dict[str, Any] = {}
    if isinstance(values_cfg, Mapping):
        kwargs["parameters"] = dict(values_cfg)
    for key in ("residual", "loss", "f_scale", "max_nfev"):
        value = _config_value(config, key, None)
        if value is not None:
            kwargs[key] = value
    draw_axis_deg = float(_config_value(config, "draw_axis_deg", 90.0))
    reference_axis_deg = draw_axis_deg - 90.0
    try:
        fit = observable_module.fit_symmetric_double_ellipse(
            np.asarray(rows, dtype=float),
            **kwargs,
            reference_axis_deg=reference_axis_deg,
            q_unit=_qmap_unit(qmap, config),
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
) -> Any:
    """Run the empirical pixel-wise intensity refinement when requested."""

    from .intensity import default_intensity_parameters, fit_intensity_model, parameter_values
    from .parameters import ParameterSet

    options = _analysis_options(image, qmap, config)
    mask = _observable_mask(options["mask"], image.shape, qmap=qmap, config=config)
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
        "mask": mask,
        "fixed": analysis.get("fixed"),
        "bounds": analysis.get("bounds"),
        # Precision-first default: every valid detector pixel participates.
        # A deterministic speed cap is opt-in through analysis.max_pixels.
        "max_pixels": analysis.get("max_pixels"),
        "seed": analysis.get("seed", 0),
        "scales": analysis.get("scales", (0.25, 0.5, 1.0)),
        "robust_loss": analysis.get("robust_loss", analysis.get("loss", "soft_l1")),
        "f_scale": analysis.get("f_scale", 1.0),
        "max_nfev": analysis.get("max_nfev", 800),
        "reference_axis_deg": float(options["draw_axis_deg"]) - 90.0,
        # Detector counts/absolute intensity can differ by many orders of
        # magnitude.  Scale only internally generated defaults; explicit or
        # warm-started values remain authoritative unless requested in config.
        "auto_scale_initial": bool(
            analysis.get("auto_scale_initial", auto_initial)
        ),
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
    kwargs = {key: value for key, value in kwargs.items() if value is not None}
    try:
        fit = fit_intensity_model(observed_frame, qmap, initial, **kwargs)
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
            "full2d": self.full2d,
            "output_paths": self.output_paths,
            "image": self.image,
            "qmap": self.qmap,
            "valid_mask": self.valid_mask,
        }
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
    report_valid_mask = bundle.valid_mask
    if isinstance(qmap_obj, Mapping) and qmap_obj.get("valid_mask") is not None:
        report_valid_mask = np.asarray(qmap_obj["valid_mask"], dtype=bool)
    qx, qy, q = _qmap_arrays(qmap_obj, image.shape)
    finite = np.isfinite(image)
    requested_fit_ellipse = bool(_config_value(config, "fit_ellipse", fit_ellipse))
    measured = measure_observables(
        image,
        qmap_obj,
        config=config,
        frame=bundle.frame,
        fit_ellipse=requested_fit_ellipse,
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
        "valid_mask": _jsonable(report_valid_mask),
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
    output: str | os.PathLike[str] | None = None,
    force: bool = False,
) -> PipelineResult:
    """Run read → qmap → observables → ridge → symmetric ellipse → full2d."""

    bundle = _read_frame_bundle(
        source,
        config=config,
        frame=frame,
        dataset=dataset,
        valid_mask=valid_mask,
        external_mask=mask,
    )
    image, metadata, embedded_qmap = bundle.image, bundle.metadata, bundle.qmap
    if isinstance(source, (str, os.PathLike)):
        metadata.setdefault("path", os.fspath(source))
    qmap_obj = build_qmap(
        image,
        poni=poni or _config_value(config, "poni_path", _config_value(config, "poni", None)),
        qmap=qmap if qmap is not None else embedded_qmap,
        config=config,
        valid_mask=bundle.valid_mask,
    )
    result_valid_mask = bundle.valid_mask
    if isinstance(qmap_obj, Mapping) and qmap_obj.get("valid_mask") is not None:
        result_valid_mask = np.asarray(qmap_obj["valid_mask"], dtype=bool)
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
        ridges = extract_ridges(image, qmap_obj, config=config)
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
        )
        if run_full2d
        else None
    )
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
        },
        valid_mask=result_valid_mask,
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


def _result_arrays(result: PipelineResult) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {"image": np.asarray(result.image)}
    if result.valid_mask is not None:
        arrays["valid_mask"] = np.asarray(result.valid_mask, dtype=bool)
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
        result = PipelineResult(
            image=image,
            qmap=mapping.get("qmap", {}),
            observables=dict(mapping.get("observables", {})),
            ridges=list(mapping.get("ridges", [])),
            ellipse_fit=dict(mapping.get("ellipse_fit", mapping.get("ellipse", {}))),
            full2d=mapping.get("full2d"),
            metadata=dict(mapping.get("metadata", {})),
            flags=dict(mapping.get("flags", {})),
        )
    target = Path(output)
    suffix = target.suffix.lower()
    if suffix in {".json", ".npz", ".csv"}:
        paths = [target]
    else:
        stem = _safe_stem(result.metadata.get("path"), "frame")
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
            arrays["observables_json"] = np.asarray(json.dumps(_jsonable(result.observables), ensure_ascii=False))
            np.savez_compressed(path, **arrays)
        elif path.suffix.lower() == ".csv":
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(("parameter", "value"))
                for key, value in result.observables.items():
                    writer.writerow((key, _jsonable(value, array_summary=False)))
                for key, value in result.ellipse_fit.get("parameters", {}).items():
                    writer.writerow((f"ellipse.{key}", _jsonable(value, array_summary=False)))
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
            expanded.extend(sorted(Path(match) for match in glob.glob(text)))
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
) -> list[PipelineResult]:
    """Analyze a sequence of frames and optionally export one bundle per frame."""

    values = _expand_inputs(inputs)
    if not values:
        raise PipelineError("没有可处理的输入帧")
    destination = output_dir or _config_value(config, "output_dir", _config_value(config, "output", None))
    results: list[PipelineResult] = []
    for item in values:
        result = analyze_frame(item, poni=poni, config=config, full2d=full2d)
        if destination is not None:
            target = Path(destination)
            result.output_paths = [os.fspath(path) for path in export_result(result, target, force=force)]
        results.append(result)
    return results


def run_project(
    project: ProjectConfig | str | os.PathLike[str],
    *,
    force: bool = False,
) -> list[PipelineResult]:
    """Run a project TOML file, resolving relative paths beside the file."""

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
    return batch_analyze(
        config.input_paths,
        poni=config.poni_path,
        config=config,
        output_dir=config.output_dir,
        full2d=config.full2d,
        force=force,
    )


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
    "read_frame",
    "build_qmap",
    "measure_observables",
    "extract_ridges",
    "fit_symmetric_ellipses",
    "fit_full2d",
    "inspect_frame",
    "analyze_frame",
    "batch_analyze",
    "run_project",
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
