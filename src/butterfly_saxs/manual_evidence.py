"""Export one manually reviewed fit as a small, auditable evidence bundle.

This module deliberately keeps the manual-evidence seam independent from Qt.
Callers provide numeric arrays and plain result/context mappings; the exporter
renders the four diagnostic images from those arrays and writes exactly the
seven files defined by the single-frame evidence contract.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import platform
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


OUTPUT_NAMES = (
    "observed.png",
    "model.png",
    "residual.png",
    "overlay.png",
    "parameters.csv",
    "fit_session.json",
    "provenance.json",
)
PARAMETER_COLUMNS = ("name", "value", "min", "max", "vary", "expr", "unit", "stderr")
_MISSING = object()
_INPUT_ROLES = ("source", "poni", "mask")


def _read(value: Any, names: Sequence[str], default: Any = _MISSING) -> Any:
    """Read one field from either a mapping or a small result object."""

    for name in names:
        if isinstance(value, Mapping) and name in value:
            return value[name]
        try:
            return getattr(value, name)
        except (AttributeError, KeyError, TypeError):
            continue
    return default


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if is_dataclass(value):
        return {field.name: getattr(value, field.name) for field in fields(value)}
    if hasattr(value, "__dict__"):
        return {str(key): item for key, item in vars(value).items() if not str(key).startswith("_")}
    if value is None:
        return {}
    raise TypeError("result/context/review 必须是 mapping 或带公开属性的对象")


def _first(*values: Any, default: Any = None) -> Any:
    for value in values:
        if value is not _MISSING and value is not None:
            return value
    return default


def _safe_float(value: Any, field_name: str, *, allow_none: bool = True) -> float | None:
    if value is None or value is _MISSING or value == "":
        if allow_none:
            return None
        raise ValueError(f"{field_name} 不能为空")
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{field_name} 必须是有限数值")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field_name} 必须是有限数值") from exc
    if not math.isfinite(result):
        if allow_none:
            return None
        raise ValueError(f"{field_name} 必须是有限数值")
    return result


def _json_safe(value: Any) -> Any:
    """Convert metadata to strict JSON without embedding detector-sized arrays."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _json_safe(value.item())
        summary: dict[str, Any] = {"shape": list(value.shape), "dtype": str(value.dtype)}
        if np.issubdtype(value.dtype, np.number):
            finite = np.isfinite(value)
            summary["finite_count"] = int(np.count_nonzero(finite))
            if np.any(finite):
                finite_values = np.asarray(value[finite], dtype=float)
                summary["min"] = _json_safe(float(np.min(finite_values)))
                summary["max"] = _json_safe(float(np.max(finite_values)))
        else:
            summary["size"] = int(value.size)
        return summary
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_safe(item())
        except (TypeError, ValueError, OverflowError):
            pass
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    for method_name in ("to_mapping", "as_dict", "to_dict"):
        method = getattr(value, method_name, None)
        if callable(method):
            try:
                converted = method(include_specs=True)
            except TypeError:
                try:
                    converted = method()
                except TypeError:
                    continue
            if converted is not value:
                return _json_safe(converted)
    if is_dataclass(value):
        return {field.name: _json_safe(getattr(value, field.name)) for field in fields(value)}
    if hasattr(value, "__dict__"):
        return _json_safe({key: item for key, item in vars(value).items() if not str(key).startswith("_")})
    return str(value)


def _strict_json_bytes(value: Any) -> bytes:
    return (json.dumps(_json_safe(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _coerce_image(value: Any, name: str, shape: tuple[int, int] | None = None) -> np.ndarray:
    if value is _MISSING or value is None:
        raise ValueError(f"result 缺少 {name}")
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是二维数值数组") from exc
    if array.ndim != 2 or array.size == 0:
        raise ValueError(f"{name} 必须是非空二维数值数组")
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} shape {array.shape} 与 observed shape {shape} 不一致")
    return array


def _coerce_mask(value: Any, name: str, shape: tuple[int, int]) -> np.ndarray:
    if value is _MISSING or value is None:
        return np.ones(shape, dtype=bool)
    try:
        mask = np.asarray(value, dtype=bool)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是布尔/0-1二维数组") from exc
    if mask.shape != shape:
        raise ValueError(f"{name} shape {mask.shape} 与图像 shape {shape} 不一致")
    return mask


def _extract_arrays(result: Any, context: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, np.ndarray, np.ndarray]:
    full2d = _read(result, ("full2d",), default={})
    observed = _first(_read(result, ("observed", "image", "data")), _read(context, ("observed", "image", "data")))
    observed_array = _coerce_image(observed, "observed")
    model = _first(
        _read(result, ("model", "model_image", "predicted", "fit")),
        _read(full2d, ("model", "model_image", "prediction")),
    )
    residual = _first(
        _read(result, ("residual", "residual_image", "difference", "resid")),
        _read(full2d, ("residual", "residual_image")),
    )
    model_array = _coerce_image(model, "model", observed_array.shape)
    residual_array = _coerce_image(residual, "residual", observed_array.shape)

    qmap = _first(_read(result, ("qmap",)), _read(context, ("qmap",)), default={})
    qx_value = _first(_read(result, ("qx", "qx_nm_inv")), _read(qmap, ("qx", "qx_nm_inv")), _read(context, ("qx", "qx_nm_inv")))
    qy_value = _first(_read(result, ("qy", "qy_nm_inv")), _read(qmap, ("qy", "qy_nm_inv")), _read(context, ("qy", "qy_nm_inv")))
    if qx_value is _MISSING or qx_value is None or qy_value is _MISSING or qy_value is None:
        rows, cols = observed_array.shape
        qx_array, qy_array = np.meshgrid(np.arange(cols, dtype=float), np.arange(rows, dtype=float))
        q_unit = "pixel"
    else:
        qx_array = _coerce_image(qx_value, "qx", observed_array.shape)
        qy_array = _coerce_image(qy_value, "qy", observed_array.shape)
        q_unit = str(
            _first(
                _read(result, ("q_unit", "unit")),
                _read(qmap, ("q_unit", "unit")),
                _read(_read(qmap, ("metadata",), default={}), ("q_unit", "unit")),
                _read(context, ("q_unit", "unit")),
                default="unknown",
            )
            or "unknown"
        )
    valid = np.isfinite(observed_array) & np.isfinite(model_array) & np.isfinite(residual_array)
    valid_mask = _first(_read(result, ("valid_mask", "fit_valid_mask")), _read(context, ("valid_mask", "fit_valid_mask")), _read(qmap, ("valid_mask", "valid")))
    valid &= _coerce_mask(valid_mask, "valid_mask", observed_array.shape)
    external_mask = _first(
        _read(result, ("external_mask", "mask")),
        _read(context, ("external_mask", "mask")),
        _read(qmap, ("mask",)),
    )
    if external_mask is not _MISSING and external_mask is not None:
        valid &= ~_coerce_mask(external_mask, "external_mask", observed_array.shape)
    domain = _first(_read(result, ("analysis_domain",)), _read(context, ("analysis_domain",)))
    domain_valid = _read(domain, ("fit_valid_mask",), default=_MISSING)
    if domain_valid is not _MISSING and domain_valid is not None:
        valid &= _coerce_mask(domain_valid, "analysis_domain.fit_valid_mask", observed_array.shape)
    if not np.any(valid):
        raise ValueError("observed/model/residual 与 mask 组合后没有有效像素")
    if qx_value is not _MISSING and qx_value is not None:
        valid &= np.isfinite(qx_array) & np.isfinite(qy_array)
        if not np.any(valid):
            raise ValueError("qx/qy 在有效像素中没有有限坐标")
    return observed_array, model_array, residual_array, qx_array, q_unit, valid, qy_array


def _parameter_source(result: Any, context: Mapping[str, Any]) -> tuple[Any, Mapping[str, Any]]:
    parameters = _read(result, ("parameters", "params", "parameter_specs"), default=_MISSING)
    if parameters is _MISSING or parameters is None:
        full2d = _read(result, ("full2d",), default={})
        parameters = _read(full2d, ("parameters", "params"), default=_MISSING)
    if parameters is _MISSING or parameters is None:
        raise ValueError("result 缺少 parameters")
    return parameters, context


def _parameter_rows(result: Any, context: Mapping[str, Any]) -> list[dict[str, Any]]:
    source, _ = _parameter_source(result, context)
    result_stderr = _first(_read(result, ("stderr", "parameter_stderr")), default={})
    context_stderr = _read(context, ("stderr", "parameter_stderr"), default={})
    stderr_map = result_stderr if isinstance(result_stderr, Mapping) else {}
    if not stderr_map and isinstance(context_stderr, Mapping):
        stderr_map = context_stderr
    units = _first(_read(result, ("parameter_units", "units")), _read(context, ("parameter_units", "units")), default={})
    units_map = units if isinstance(units, Mapping) else {}

    rows: list[dict[str, Any]] = []
    if hasattr(source, "spec_items") and callable(source.spec_items):
        items = list(source.spec_items())
    elif isinstance(source, Mapping):
        items = list(source.items())
    elif isinstance(source, Sequence) and not isinstance(source, (str, bytes)):
        items = []
        for index, item in enumerate(source):
            row = _as_mapping(item)
            name = _first(_read(row, ("name", "parameter")), default=f"parameter_{index}")
            items.append((str(name), row))
    else:
        raise ValueError("parameters 必须是 mapping、ParameterSet 或参数行序列")

    for raw_name, raw_spec in items:
        name = str(raw_name)
        spec = _as_mapping(raw_spec) if isinstance(raw_spec, Mapping) or hasattr(raw_spec, "__dict__") or is_dataclass(raw_spec) else {}
        if spec:
            value = _read(spec, ("value",), default=raw_spec if not isinstance(raw_spec, Mapping) else _MISSING)
            minimum = _read(spec, ("min", "minimum", "lower"), default=None)
            maximum = _read(spec, ("max", "maximum", "upper"), default=None)
            vary = _read(spec, ("vary",), default=True)
            expr = _read(spec, ("expr",), default=None)
            unit = _first(_read(spec, ("unit",)), units_map.get(name), default="")
            stderr = _first(_read(spec, ("stderr", "uncertainty")), stderr_map.get(name), default=None)
        else:
            value = raw_spec
            minimum = maximum = None
            vary = True
            expr = None
            unit = units_map.get(name, "")
            stderr = stderr_map.get(name)
        numeric_value = _safe_float(value, f"parameter {name}.value", allow_none=False)
        numeric_min = _safe_float(minimum, f"parameter {name}.min")
        numeric_max = _safe_float(maximum, f"parameter {name}.max")
        numeric_stderr = _safe_float(stderr, f"parameter {name}.stderr")
        rows.append(
            {
                "name": name,
                "value": numeric_value,
                "min": numeric_min,
                "max": numeric_max,
                "vary": bool(vary),
                "expr": "" if expr is None else str(expr),
                "unit": "" if unit is None else str(unit),
                "stderr": numeric_stderr,
            }
        )
    if not rows:
        raise ValueError("parameters 不能为空")
    return rows


def _normalise_review(review: Any) -> dict[str, Any]:
    if review is None:
        payload: dict[str, Any] = {}
    else:
        payload = dict(_as_mapping(review))
    status = str(payload.get("manual_status", payload.get("status", "unreviewed"))).strip().lower()
    if status not in {"unreviewed", "accepted", "rejected"}:
        raise ValueError("manual_status 只能是 unreviewed、accepted 或 rejected")
    reviewed_by = payload.get("reviewed_by")
    if reviewed_by is not None:
        reviewed_by = str(reviewed_by).strip()
    reviewed_at = payload.get("reviewed_at")
    if isinstance(reviewed_at, datetime):
        reviewed_at = reviewed_at.isoformat()
    elif reviewed_at is not None:
        reviewed_at = str(reviewed_at).strip()
    if status in {"accepted", "rejected"}:
        if not reviewed_by:
            raise ValueError(f"manual_status={status} 时 reviewed_by 不能为空")
        if not reviewed_at:
            raise ValueError(f"manual_status={status} 时 reviewed_at 不能为空")
        try:
            datetime.fromisoformat(reviewed_at.replace("Z", "+00:00"))
        except (TypeError, ValueError) as exc:
            raise ValueError("reviewed_at 必须是可解析的 ISO 时间") from exc
    payload["manual_status"] = status
    payload["reviewed_by"] = reviewed_by
    payload["reviewed_at"] = reviewed_at
    payload["review_notes"] = str(payload.get("review_notes", payload.get("notes", "")) or "")
    return payload


def _coordinates(value: Any) -> np.ndarray:
    if value is None or value is _MISSING:
        return np.empty((0, 2), dtype=float)
    if isinstance(value, Mapping):
        nested = _first(value.get("points"), value.get("ridges"), value.get("ridge_points"), default=_MISSING)
        if nested is not _MISSING:
            return _coordinates(nested)
        if "qx" in value and "qy" in value:
            x, y = np.asarray(value["qx"], dtype=float).ravel(), np.asarray(value["qy"], dtype=float).ravel()
            return np.column_stack((x, y)) if x.size == y.size else np.empty((0, 2), dtype=float)
        x = _first(value.get("x"), value.get("q_x"), default=_MISSING)
        y = _first(value.get("y"), value.get("q_y"), default=_MISSING)
        if x is not _MISSING and y is not _MISSING:
            return np.asarray([[float(x), float(y)]], dtype=float)
        q = _first(value.get("q"), value.get("q_star"), default=_MISSING)
        angle = _first(value.get("angle_deg"), value.get("angle"), value.get("azimuth"), default=_MISSING)
        if q is not _MISSING and angle is not _MISSING:
            angle_value = float(angle)
            if "angle_deg" not in value and abs(angle_value) <= 2 * np.pi:
                angle_value = float(np.degrees(angle_value))
            angle_rad = np.radians(angle_value)
            q_value = float(q)
            return np.asarray([[q_value * np.cos(angle_rad), q_value * np.sin(angle_rad)]], dtype=float)
        branch_rows = [_coordinates(item) for item in value.values()]
        nonempty = [row for row in branch_rows if row.size]
        if nonempty:
            return np.vstack(nonempty)
        return np.empty((0, 2), dtype=float)
    if isinstance(value, np.ndarray):
        array = np.asarray(value, dtype=float)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if not value:
            return np.empty((0, 2), dtype=float)
        try:
            array = np.asarray(value, dtype=float)
        except (TypeError, ValueError):
            rows = [_coordinates(item) for item in value]
            return np.vstack([row for row in rows if row.size]) if any(row.size for row in rows) else np.empty((0, 2), dtype=float)
    else:
        return np.empty((0, 2), dtype=float)
    if array.ndim == 1 and array.size == 2:
        array = array.reshape(1, 2)
    if array.ndim != 2 or array.shape[1] < 2:
        return np.empty((0, 2), dtype=float)
    return np.asarray(array[:, :2], dtype=float)


def _ellipse_curves(value: Any) -> list[np.ndarray]:
    """Return explicit curves or generate curves from basic ellipse values."""

    if value is None or value is _MISSING:
        return []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, np.ndarray, Mapping)):
        curves: list[np.ndarray] = []
        for item in value:
            curves.extend(_ellipse_curves(item))
        return curves
    if isinstance(value, np.ndarray):
        points = _coordinates(value)
        return [points] if points.size else []
    mapping = _as_mapping(value) if isinstance(value, Mapping) or hasattr(value, "__dict__") or is_dataclass(value) else {}
    if mapping:
        for key in ("curve", "points", "xy", "coordinates"):
            if key in mapping:
                points = _coordinates(mapping[key])
                if points.size:
                    return [points]
        pair = _first(mapping.get("ellipses"), mapping.get("ellipse_pair"), mapping.get("curves"), default=_MISSING)
        if pair is not _MISSING:
            return _ellipse_curves(pair)
        parameters = _first(mapping.get("parameters"), mapping.get("values"), default=_MISSING)
        if parameters is not _MISSING:
            nested = _ellipse_curves(parameters)
            if nested:
                return nested
        def number(*names: str) -> float | None:
            return _safe_float(_first(*(mapping.get(name, _MISSING) for name in names), default=None), names[0])

        cx, cy, a = number("cx", "center_x"), number("cy", "center_y"), number("a", "semi_major", "major_axis")
        b = number("b", "semi_minor", "minor_axis")
        ratio = number("axis_ratio", "ratio")
        if b is None and a is not None and ratio is not None:
            b = a * ratio
        if cx is None or cy is None or a is None or b is None or a <= 0 or b <= 0:
            return []
        theta = number("theta", "angle")
        if theta is None:
            theta_deg = number("theta_deg", "angle_deg")
            theta = None if theta_deg is None else float(np.radians(theta_deg))
        theta = 0.0 if theta is None else theta
        phi = np.linspace(0.0, 2.0 * np.pi, 361)
        c, s = np.cos(theta), np.sin(theta)
        u, v = a * np.cos(phi), b * np.sin(phi)
        return [np.column_stack((cx + c * u - s * v, cy + s * u + c * v))]
    return []


def _result_ridges(result: Any) -> np.ndarray:
    value = _read(result, ("ridges", "ridge_points", "ridge"), default=None)
    return _coordinates(value)


def _result_ellipses(result: Any) -> list[np.ndarray]:
    direct = _read(result, ("observed_ellipses", "ellipses", "ellipse_fits"), default=_MISSING)
    fit = _read(result, ("ellipse_fit", "ellipse", "ellipse_result"), default=_MISSING)
    curves: list[np.ndarray] = []
    if direct is not _MISSING:
        curves.extend(_ellipse_curves(direct))
    if not curves and fit is not _MISSING:
        curves.extend(_ellipse_curves(fit))
    return curves


def _context_model_ellipses(context: Mapping[str, Any]) -> list[np.ndarray]:
    value = _first(
        _read(context, ("current_model_ellipses", "model_ellipses", "model_ellipse")),
        default=_MISSING,
    )
    return _ellipse_curves(value) if value is not _MISSING else []


def _finite_limits(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 1.0
    lower, upper = float(np.min(finite)), float(np.max(finite))
    if upper <= lower:
        delta = max(abs(lower) * 0.01, 1.0)
        return lower - delta, upper + delta
    return lower, upper


def _extent(qx: np.ndarray, qy: np.ndarray, valid: np.ndarray) -> tuple[float, float, float, float]:
    x, y = qx[valid], qy[valid]
    return float(np.min(x)), float(np.max(x)), float(np.min(y)), float(np.max(y))


def _render_png(
    array: np.ndarray,
    qx: np.ndarray,
    qy: np.ndarray,
    valid: np.ndarray,
    *,
    title: str,
    q_unit: str,
    vmin: float,
    vmax: float,
    residual: bool = False,
    ridges: np.ndarray | None = None,
    ellipses: Iterable[np.ndarray] = (),
    model_ellipses: Iterable[np.ndarray] = (),
) -> bytes:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    image = np.where(valid, array, np.nan)
    figure, axis = plt.subplots(figsize=(5.0, 4.4), constrained_layout=True)
    try:
        extent = _extent(qx, qy, valid)
        cmap = "PuOr" if residual else "cividis"
        shown = axis.imshow(
            image,
            origin="lower",
            extent=extent,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
            aspect="equal",
        )
        axis.set_title(title)
        axis.set_xlabel(f"qx ({q_unit})")
        axis.set_ylabel(f"qy ({q_unit})")
        figure.colorbar(shown, ax=axis, shrink=0.84, label="Data - model" if residual else "Intensity")
        if not residual:
            if ridges is not None and ridges.size:
                points = ridges[np.isfinite(ridges).all(axis=1)]
                if points.size:
                    axis.scatter(points[:, 0], points[:, 1], s=11, facecolors="none", edgecolors="#F0E442", linewidths=0.7, label="Observed ridge")
            for index, curve in enumerate(ellipses):
                points = np.asarray(curve, dtype=float)
                if points.ndim == 2 and points.shape[1] == 2 and points.size:
                    axis.plot(points[:, 0], points[:, 1], color="#E69F00", lw=1.2, label="Observed ellipse" if index == 0 else None)
            for index, curve in enumerate(model_ellipses):
                points = np.asarray(curve, dtype=float)
                if points.ndim == 2 and points.shape[1] == 2 and points.size:
                    axis.plot(points[:, 0], points[:, 1], color="#FFFFFF", lw=1.2, ls="--", label="Current model" if index == 0 else None)
            handles, _ = axis.get_legend_handles_labels()
            if handles:
                axis.legend(frameon=False, fontsize=7, loc="best")
        output = io.BytesIO()
        figure.savefig(output, format="png", dpi=300, bbox_inches="tight")
        return output.getvalue()
    finally:
        plt.close(figure)


def _csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=PARAMETER_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        serialised = dict(row)
        for name in ("value", "min", "max", "stderr"):
            value = serialised.get(name)
            serialised[name] = "" if value is None else value
        writer.writerow(serialised)
    return output.getvalue().encode("utf-8")


def _summary_mapping(value: Any) -> Any:
    mapping = _as_mapping(value) if value is not None and value is not _MISSING else {}
    return _json_safe(mapping)


def _metrics(result: Any, residual: np.ndarray, valid: np.ndarray) -> dict[str, Any]:
    finite = residual[valid]
    metrics: dict[str, Any] = {"valid_pixel_count": int(np.count_nonzero(valid))}
    if finite.size:
        metrics["rmse"] = float(np.sqrt(np.mean(np.square(finite))))
        metrics["residual_mean"] = float(np.mean(finite))
        metrics["residual_p95_abs"] = float(np.percentile(np.abs(finite), 95.0))
    source = _first(_read(result, ("metrics", "diagnostics")), _read(_read(result, ("full2d",), default={}), ("metrics", "diagnostics")), default={})
    if isinstance(source, Mapping):
        for key, value in source.items():
            if key not in metrics:
                metrics[str(key)] = _json_safe(value)
    return metrics


def _file_record(value: Any) -> dict[str, Any]:
    if value is None or value is _MISSING or isinstance(value, (np.ndarray, list, tuple, Mapping)):
        return {"path": None, "exists": False}
    if isinstance(value, str) and value.strip().casefold() in {"in-memory", "in_memory"}:
        return {"path": None, "exists": False}
    path = Path(value).expanduser().resolve(strict=False)
    record: dict[str, Any] = {"path": str(path), "exists": bool(path.exists() and path.is_file())}
    if record["exists"]:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        record["sha256"] = digest.hexdigest()
    return record


def capture_input_records(
    *,
    source: Any = None,
    poni: Any = None,
    mask: Any = None,
) -> dict[str, dict[str, Any]]:
    """Capture path, existence, and SHA-256 for fit-defining files."""

    return {
        "source": _file_record(source),
        "poni": _file_record(poni),
        "mask": _file_record(mask),
    }


def _canonical_record_path(value: Any) -> str | None:
    if value is None or value == "":
        return None
    return str(Path(value).expanduser().resolve(strict=False)).casefold()


def _verify_input_records(
    expected: Any,
    current: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    if expected is None or expected is _MISSING:
        if any(current[role].get("path") is not None for role in _INPUT_ROLES):
            raise ValueError("缺少拟合时输入哈希，不能为文件来源的结果导出人工证据")
        return {role: dict(current[role]) for role in _INPUT_ROLES}
    if not isinstance(expected, Mapping):
        raise ValueError("fit_input_records 必须是 source/poni/mask 记录表")

    verified: dict[str, dict[str, Any]] = {}
    for role in _INPUT_ROLES:
        record = expected.get(role)
        if not isinstance(record, Mapping):
            raise ValueError(f"fit_input_records 缺少 {role} 记录")
        expected_record = dict(record)
        current_record = dict(current[role])
        if _canonical_record_path(expected_record.get("path")) != _canonical_record_path(
            current_record.get("path")
        ):
            raise ValueError(f"{role} 输入路径与拟合时记录不一致")
        expected_exists = bool(expected_record.get("exists", False))
        current_exists = bool(current_record.get("exists", False))
        if expected_exists != current_exists:
            raise ValueError(f"{role} 输入在拟合后已变化：文件存在状态不一致")
        if expected_exists:
            expected_hash = expected_record.get("sha256")
            current_hash = current_record.get("sha256")
            if not isinstance(expected_hash, str) or not expected_hash:
                raise ValueError(f"{role} 的拟合时记录缺少 SHA-256")
            if expected_hash != current_hash:
                raise ValueError(f"{role} 输入在拟合后已变化：SHA-256 不一致")
        verified[role] = expected_record
    return verified


def _input_values(result: Any, context: Mapping[str, Any]) -> tuple[Any, Any, Any]:
    metadata = _first(_read(result, ("metadata", "provenance")), default={})
    if not isinstance(metadata, Mapping):
        metadata = {}
    source = _first(_read(context, ("source", "source_path", "path")), _read(result, ("source", "source_path")), _read(metadata, ("source", "path", "source_path")))
    poni = _first(_read(context, ("poni", "poni_path")), _read(result, ("poni", "poni_path")), _read(metadata, ("poni", "poni_path")))
    mask = _first(_read(context, ("mask_path", "mask_file", "mask")), _read(result, ("mask_path", "mask_file")), _read(metadata, ("mask_path", "mask_file")))
    return source, poni, mask


def _provenance(
    result: Any,
    context: Mapping[str, Any],
    q_unit: str,
    *,
    current_inputs: Mapping[str, Mapping[str, Any]],
    fit_inputs: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    metadata = _first(_read(result, ("metadata", "provenance")), default={})
    if not isinstance(metadata, Mapping):
        metadata = {}
    source, poni, mask = _input_values(result, context)
    frame = _first(_read(context, ("frame", "frame_id", "index")), _read(result, ("frame", "frame_id")), _read(metadata, ("frame", "frame_id")))
    dataset = _first(_read(context, ("dataset", "dataset_id")), _read(result, ("dataset", "dataset_id")), _read(metadata, ("dataset", "dataset_id")))
    roi = _first(_read(context, ("roi", "rois", "exclusion_roi")), _read(result, ("roi", "rois", "exclusion_roi")), _read(metadata, ("roi", "rois")))
    return {
        "schema_version": "lamellarsaxs2d.manual_fit_provenance.v1",
        "software": {"name": "LamellarSAXS2D", "version": _module_version("butterfly_saxs") or "unknown"},
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": _module_version("scipy"),
            "matplotlib": _module_version("matplotlib"),
        },
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "source": _json_safe(source),
        "frame": _json_safe(frame),
        "dataset": _json_safe(dataset),
        "poni": _json_safe(poni),
        "mask": _json_safe(mask),
        "roi": _json_safe(roi),
        "q_unit": q_unit,
        "inputs": _json_safe(current_inputs),
        "fit_time_inputs": _json_safe(fit_inputs),
        "input_binding_verified": True,
        "output_files": list(OUTPUT_NAMES),
    }


def _module_version(name: str) -> str | None:
    try:
        module = __import__(name)
        return str(getattr(module, "__version__", "unknown"))
    except ImportError:
        return None


def export_manual_fit(
    result: Any,
    output_dir: str | Path,
    *,
    context: Mapping[str, Any] | None = None,
    review: Mapping[str, Any] | None = None,
    force: bool = False,
) -> dict[str, Path]:
    """Write the seven-file evidence bundle for one manual fit.

    ``review=None`` deliberately produces ``manual_status=unreviewed``.  A
    caller may pass an explicit human review record for ``accepted`` or
    ``rejected``, but the exporter never promotes an unreviewed result itself.
    All validation and serialisation happen before the first target is opened.
    """

    context_mapping = _as_mapping(context or {})
    result_mapping = _as_mapping(result)
    output = Path(output_dir)
    if output.exists() and not output.is_dir():
        raise NotADirectoryError(f"输出路径不是目录：{output}")
    targets = {name: output / name for name in OUTPUT_NAMES}
    existing = [path for path in targets.values() if path.exists()]
    if existing and not force:
        raise FileExistsError("输出已存在，未覆盖：" + "、".join(str(path) for path in existing) + "（需要 force=True）")

    review_payload = _normalise_review(review)
    source, poni, mask = _input_values(result_mapping, context_mapping)
    current_inputs = capture_input_records(source=source, poni=poni, mask=mask)
    expected_inputs = _first(
        _read(context_mapping, ("fit_input_records", "input_records")),
        _read(result_mapping, ("fit_input_records", "input_records")),
        default=None,
    )
    fit_inputs = _verify_input_records(expected_inputs, current_inputs)
    observed, model, residual, qx, q_unit, valid, qy = _extract_arrays(result, context_mapping)
    rows = _parameter_rows(result, context_mapping)
    ridges = _result_ridges(result)
    observed_ellipses = _result_ellipses(result)
    model_ellipses = _context_model_ellipses(context_mapping)

    data_values = np.concatenate((observed[valid], model[valid]))
    data_vmin, data_vmax = _finite_limits(data_values)
    residual_limit = float(np.max(np.abs(residual[valid]))) if np.any(valid) else 1.0
    if not math.isfinite(residual_limit) or residual_limit <= 0:
        residual_limit = 1.0
    files: dict[str, bytes] = {
        "observed.png": _render_png(observed, qx, qy, valid, title="Observed", q_unit=q_unit, vmin=data_vmin, vmax=data_vmax),
        "model.png": _render_png(model, qx, qy, valid, title="Model", q_unit=q_unit, vmin=data_vmin, vmax=data_vmax),
        "residual.png": _render_png(residual, qx, qy, valid, title="Residual", q_unit=q_unit, vmin=-residual_limit, vmax=residual_limit, residual=True),
        "overlay.png": _render_png(observed, qx, qy, valid, title="Overlay", q_unit=q_unit, vmin=data_vmin, vmax=data_vmax, ridges=ridges, ellipses=observed_ellipses, model_ellipses=model_ellipses),
    }
    metrics = _metrics(result, residual, valid)
    flags = _first(_read(result, ("flags", "quality_flags", "scientific_flags")), default={})
    flags_mapping = dict(flags) if isinstance(flags, Mapping) else {"values": _json_safe(flags)}
    flags_mapping.update({"empirical_model_only": True, "human_review_required": True})
    analysis = _first(_read(result, ("analysis", "diagnostics")), default={})
    analysis_safe = _summary_mapping(analysis)
    if not isinstance(analysis_safe, Mapping):
        analysis_safe = {"value": analysis_safe}
    session = {
        "schema_version": "lamellarsaxs2d.manual_fit_session.v1",
        "manual_status": review_payload["manual_status"],
        "reviewed_by": review_payload["reviewed_by"],
        "reviewed_at": review_payload["reviewed_at"],
        "review_notes": review_payload["review_notes"],
        "review": review_payload,
        "context": _json_safe(context_mapping),
        "metrics": metrics,
        "flags": flags_mapping,
        "analysis": {**analysis_safe, "q_unit": q_unit, "valid_pixel_count": int(np.count_nonzero(valid))},
        "ridges": {"count": int(len(ridges)), "points": _json_safe(ridges)},
        "ellipse": {"observed_count": len(observed_ellipses), "model_count": len(model_ellipses)},
        "parameters": rows,
        "outputs": list(OUTPUT_NAMES),
        "empirical_model_only": True,
        "human_review_required": True,
    }
    files["parameters.csv"] = _csv_bytes(rows)
    files["fit_session.json"] = _strict_json_bytes(session)
    files["provenance.json"] = _strict_json_bytes(
        _provenance(
            result_mapping,
            context_mapping,
            q_unit,
            current_inputs=current_inputs,
            fit_inputs=fit_inputs,
        )
    )

    output.mkdir(parents=True, exist_ok=True)
    for name in OUTPUT_NAMES:
        targets[name].write_bytes(files[name])
    return targets


__all__ = [
    "OUTPUT_NAMES",
    "PARAMETER_COLUMNS",
    "capture_input_records",
    "export_manual_fit",
]
