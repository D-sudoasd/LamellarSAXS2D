"""Auditable exports for batch LamellarSAXS2D analyses.

Exports intentionally retain the full scientific result objects in JSON/NPZ
sidecars while providing compact CSV tables for plotting and downstream
kinetic analysis.  The functions accept :class:`~butterfly_saxs.batch.BatchRunResult`
as well as a plain sequence of ``FrameFitResult`` objects.
"""

from __future__ import annotations

import csv
import importlib.metadata as importlib_metadata
import json
import math
import os
import platform
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from .batch import BatchRunResult, FrameFitResult, FrameRef, _json_safe


_MISSING = object()
_FLAG_NAMES = (
    "scientific_flags",
    "flags",
    "quality_flags",
    "fit_flags",
    "warnings",
)


def _value(value: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(value, Mapping) and name in value:
            return value[name]
        if hasattr(value, name):
            return getattr(value, name)
    return default


def _json_text(value: Any) -> str:
    if value is None:
        return ""
    return json.dumps(
        _json_safe(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _scalar(value: Any) -> Any:
    """Convert numpy/Python scalar values for CSV without coercing arrays."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else ""
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _scalar(item())
        except Exception:  # pragma: no cover
            pass
    if isinstance(value, (list, tuple, Mapping)):
        return _json_text(value)
    return value if isinstance(value, (str, int, float, bool)) else _json_text(value)


def _as_mapping(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    if is_dataclass(value):
        return {field.name: getattr(value, field.name) for field in fields(value)}
    if hasattr(value, "__dict__"):
        return vars(value)
    return None


def _result_mapping(item: FrameFitResult) -> Mapping[str, Any] | None:
    return _as_mapping(item.result)


def _frame_results(batch: Any) -> list[FrameFitResult]:
    if isinstance(batch, BatchRunResult):
        return list(batch.frame_results)
    if hasattr(batch, "frame_results"):
        return list(batch.frame_results)
    result: list[FrameFitResult] = []
    for index, item in enumerate(batch or []):
        if isinstance(item, FrameFitResult):
            result.append(item)
        elif isinstance(item, Mapping):
            frame = item.get("frame", item.get("frame_ref", item.get("path", f"frame_{index}")))
            frame_ref = frame if isinstance(frame, FrameRef) else FrameRef(frame)
            result.append(
                FrameFitResult(
                    frame=frame_ref,
                    result=item.get("result", item.get("fit_result")),
                    status=item.get("status", "ok"),
                    error=item.get("error"),
                    warm_start_from=item.get("warm_start_from", item.get("lineage")),
                )
            )
        else:
            metadata = _value(item, "metadata", default={})
            if not isinstance(metadata, Mapping):
                metadata = {}
            source = metadata.get("path", metadata.get("source", f"frame_{index}"))
            frame_ref = FrameRef(
                source,
                frame_id=_value(item, "frame_id", "id", default=None),
                time=_value(item, "timestamp", "time", default=metadata.get("time")),
                metadata=metadata,
            )
            result.append(FrameFitResult(frame=frame_ref, result=item))
    return result


def _frame_base(item: FrameFitResult, index: int) -> dict[str, Any]:
    frame = item.frame
    return {
        "frame_index": index,
        "frame_id": _scalar(frame.frame_id),
        "path": str(frame.path),
        "time": _scalar(frame.time),
        "status": item.status,
        "error": _scalar(item.error) if item.error else "",
        "warm_start_from": _scalar(item.warm_start_from) if item.warm_start_from else "",
        "elapsed_s": _scalar(item.elapsed_s),
        "resumed": bool(item.resumed),
    }


def _empty_parameter_source(value: Any) -> bool:
    if value is None or value is _MISSING:
        return True
    if isinstance(value, Mapping):
        return not value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return not value
    try:
        return len(value) == 0
    except (TypeError, AttributeError):
        return False


def _parameter_source(value: Any) -> tuple[Any, list[Any]]:
    """Return the preferred parameter source and metadata contexts.

    ``PipelineResult`` exposes a convenience ``parameters`` property that may
    resolve to ``full2d.parameters``.  Looking at the explicit instance fields
    first keeps a genuine top-level ``parameters`` result authoritative while
    still allowing the nested full2d result to be exported when no top-level
    field exists.
    """

    contexts: list[Any] = [value]
    if isinstance(value, Mapping):
        full2d = _value(value, "full2d", default=_MISSING)
        nested = _value(full2d, "parameters", "params", default=_MISSING)
        for name in ("parameters", "params"):
            if name in value and not _empty_parameter_source(value[name]):
                source = value[name]
                if not _empty_parameter_source(nested):
                    try:
                        is_nested_alias = _json_safe(source) == _json_safe(nested)
                    except (TypeError, ValueError):
                        is_nested_alias = source is nested
                    if is_nested_alias:
                        return source, [full2d, value]
                return source, contexts
    else:
        own = vars(value) if hasattr(value, "__dict__") else {}
        for name in ("parameters", "params"):
            if name in own and not _empty_parameter_source(own[name]):
                return own[name], contexts

    source = _value(value, "parameters", "params", default=_MISSING)
    if not _empty_parameter_source(source):
        # PipelineResult's property is an alias for its nested full2d source;
        # retain that context so stderr/flags beside the nested parameters are
        # not lost.  A distinct top-level attribute remains authoritative.
        full2d = _value(value, "full2d", default=_MISSING)
        nested = _value(full2d, "parameters", "params", default=_MISSING)
        if not _empty_parameter_source(nested) and source is nested:
            return nested, [full2d, value]
        return source, contexts

    full2d = _value(value, "full2d", default=_MISSING)
    if full2d is not _MISSING and full2d is not None:
        nested = _value(full2d, "parameters", "params", default=_MISSING)
        if not _empty_parameter_source(nested):
            return nested, [full2d, value]
    source = _value(value, "values", default=_MISSING)
    if not _empty_parameter_source(source):
        return source, contexts
    # PipelineResult calls its scalar observables ``observables``.  They are
    # still frame parameters for a longitudinal export.
    source = _value(value, "observables", default=_MISSING)
    if not _empty_parameter_source(source):
        return source, contexts
    ellipse = _value(value, "ellipse_fit", "ellipse", default=None)
    nested = _value(ellipse, "parameters", "params", default=_MISSING)
    if not _empty_parameter_source(nested):
        return nested, [ellipse, value]
    return _MISSING, contexts


def _first_context_value(contexts: Sequence[Any], names: Sequence[str]) -> Any:
    for context in contexts:
        candidate = _value(context, *names, default=_MISSING)
        if candidate is not _MISSING and candidate is not None:
            return candidate
    return _MISSING


def _context_parameter_value(contexts: Sequence[Any], names: Sequence[str], name: Any) -> Any:
    candidate = _first_context_value(contexts, names)
    if candidate is _MISSING:
        return _MISSING
    if isinstance(candidate, Mapping):
        return candidate.get(name, _MISSING)
    return candidate


def _result_flags(value: Any) -> Any:
    """Prefer flags beside the selected parameter source."""

    _, contexts = _parameter_source(value)
    flags = _first_context_value(contexts, _FLAG_NAMES)
    return None if flags is _MISSING else flags


def _parameters(value: Any) -> list[dict[str, Any]]:
    """Normalise parameter sources while retaining fit diagnostics.

    The public pipeline can supply a top-level mapping, a nested
    ``full2d.parameters`` mapping, or a ``ParameterSet``/fit object.  Keep the
    source values untouched until the CSV boundary so non-finite values become
    blank rather than an invented zero.
    """

    source, contexts = _parameter_source(value)
    if source is _MISSING or source is None:
        return []

    spec_items = getattr(source, "spec_items", None)
    if callable(spec_items):
        try:
            iterable = list(spec_items())
        except Exception:  # pragma: no cover - defensive for custom sets
            iterable = []
    elif isinstance(source, Mapping):
        iterable = list(source.items())
    elif isinstance(source, Sequence) and not isinstance(source, (str, bytes)):
        iterable = []
        for item in source:
            if isinstance(item, Mapping):
                name = item.get("name", item.get("parameter", item.get("key")))
                if name is not None:
                    iterable.append((name, item))
    else:
        source_mapping = _as_mapping(source)
        iterable = list(source_mapping.items()) if source_mapping is not None else []

    rows: list[dict[str, Any]] = []
    for name, spec in iterable:
        parameter_name = str(name)
        if np.isscalar(spec) or isinstance(spec, str):
            # NumPy scalars expose unrelated array attributes such as
            # ``flags``.  They are parameter values, not rich parameter specs.
            value_field = spec
            stderr = uncertainty = fixed = unit = flags = bound_flags = _MISSING
        else:
            spec_mapping = spec if isinstance(spec, Mapping) else _as_mapping(spec)
            if spec_mapping is not None:
                value_field = _value(spec_mapping, "value", "val", "estimate", default=None)
                stderr = _value(spec_mapping, "stderr", default=_MISSING)
                uncertainty = _value(
                    spec_mapping,
                    "uncertainty",
                    "sigma",
                    "error",
                    "std",
                    default=_MISSING,
                )
                fixed = _value(spec_mapping, "fixed", "is_fixed", default=_MISSING)
                unit = _value(spec_mapping, "unit", default=_MISSING)
                flags = _value(spec_mapping, "flags", "scientific_flags", default=_MISSING)
                bound_flags = _value(spec_mapping, "bound_flags", default=_MISSING)
                if fixed is _MISSING and ("vary" in spec_mapping or "expr" in spec_mapping):
                    fixed = spec_mapping.get("expr") is None and not bool(spec_mapping.get("vary", True))
                if flags is _MISSING and spec_mapping.get("expr") is not None:
                    flags = {"tied": True}
            else:
                value_field = getattr(spec, "value", spec)
                stderr = getattr(spec, "stderr", _MISSING)
                uncertainty = getattr(spec, "uncertainty", _MISSING)
                fixed = getattr(spec, "is_fixed", _MISSING)
                if fixed is _MISSING and hasattr(spec, "vary"):
                    fixed = not bool(getattr(spec, "vary"))
                unit = getattr(spec, "unit", _MISSING)
                flags = getattr(spec, "flags", _MISSING)
                if flags is _MISSING and getattr(spec, "is_tied", False):
                    flags = {"tied": True}
                bound_flags = getattr(spec, "bound_flags", _MISSING)

        if stderr is _MISSING:
            stderr = _context_parameter_value(contexts, ("stderr",), name)
        if uncertainty is _MISSING:
            uncertainty = stderr
        if stderr is _MISSING:
            stderr = uncertainty
        if fixed is _MISSING:
            fixed = _context_parameter_value(contexts, ("fixed", "is_fixed"), name)
        if unit is _MISSING:
            unit = _context_parameter_value(contexts, ("units", "unit"), name)
        if flags is _MISSING:
            flags = _context_parameter_value(
                contexts,
                ("flags", "scientific_flags", "quality_flags", "fit_flags"),
                name,
            )
        if bound_flags is _MISSING:
            bound_flags = _context_parameter_value(contexts, ("bound_flags",), name)

        rows.append(
            {
                "parameter": parameter_name,
                "value": _scalar(value_field),
                "stderr": _scalar(stderr) if stderr is not _MISSING else "",
                "uncertainty": _scalar(uncertainty) if uncertainty is not _MISSING else "",
                "fixed": _scalar(fixed) if fixed is not _MISSING else "",
                "unit": _scalar(unit) if unit is not _MISSING else "",
                "flags": (
                    _json_text(flags)
                    if isinstance(flags, (Mapping, list, tuple, set, frozenset))
                    else (_scalar(flags) if flags is not _MISSING else "")
                ),
                "bound_flags": (
                    _json_text(bound_flags)
                    if isinstance(bound_flags, (Mapping, list, tuple, set, frozenset))
                    else (_scalar(bound_flags) if bound_flags is not _MISSING else "")
                ),
            }
        )
    return rows


def _frame_summary_rows(results: Sequence[FrameFitResult]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(results):
        row = _frame_base(item, index)
        result = item.result
        mapping = _result_mapping(item)
        params = _parameters(result)
        # Scalar top-level observables are useful in a quick summary.  Arrays
        # and nested mappings stay in their lossless JSON sidecars.
        if mapping:
            for key, value in mapping.items():
                key_text = str(key)
                if key_text in {"parameters", "params", "ridge_points", "ellipse_fit", "ellipse"}:
                    continue
                if isinstance(value, (str, int, float, bool)) or value is None:
                    row[key_text] = _scalar(value)
                elif key_text in _FLAG_NAMES:
                    row[key_text] = _json_text(value)
            if "flags" not in row:
                row["flags"] = _json_text(_value(result, "flags", default=None))
            if "scientific_flags" not in row:
                row["scientific_flags"] = _json_text(
                    _value(result, "scientific_flags", default=None)
                )
        for section_name, prefix in (("metrics", "fit"), ("full2d", "full2d")):
            section = _value(result, section_name, default=None)
            section_mapping = _as_mapping(section)
            if not section_mapping:
                continue
            for metric_name in (
                "status",
                "success",
                "rmse",
                "weighted_rmse",
                "ndata",
                "sampled_n",
                "sample_rmse",
                "nfev",
                "condition_number",
                "condition",
            ):
                metric = _value(section_mapping, metric_name, default=_MISSING)
                if metric is _MISSING or not (metric is None or np.isscalar(metric)):
                    continue
                if metric_name in {"status", "success"}:
                    output_name = f"{prefix}_{metric_name}"
                elif metric_name == "condition":
                    output_name = "condition_number"
                else:
                    output_name = metric_name
                row.setdefault(output_name, _scalar(metric))
        for parameter in params:
            name = parameter["parameter"]
            row.setdefault(name, parameter["value"])
        row["parameters_json"] = _json_text(params)
        row["frame_metadata_json"] = _json_text(item.frame.metadata)
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str] | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = list(columns or [])
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    if not keys:
        keys = ["frame_index"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _scalar(row.get(key, "")) for key in keys})
    return path


def _ridge_points(value: Any) -> list[dict[str, Any]]:
    points = _value(value, "ridge_points", "ridges", "ridge", default=None)
    if points is None:
        return []
    if isinstance(points, Mapping):
        # A named collection such as {"upper": [...], "lower": [...]} is
        # retained by adding the branch name to each point.
        rows: list[dict[str, Any]] = []
        for branch, branch_points in points.items():
            for row in _ridge_points_from_sequence(branch_points):
                row.setdefault("branch", branch)
                rows.append(row)
        return rows
    return _ridge_points_from_sequence(points)


def _ridge_points_from_sequence(points: Any) -> list[dict[str, Any]]:
    if points is None:
        return []
    tolist = getattr(points, "tolist", None)
    if callable(tolist):
        try:
            points = tolist()
        except Exception:  # pragma: no cover
            pass
    if isinstance(points, Mapping):
        return [dict(points)]
    if isinstance(points, Sequence) and not isinstance(points, (str, bytes)):
        rows: list[dict[str, Any]] = []
        for index, point in enumerate(points):
            if isinstance(point, Mapping):
                rows.append({str(key): _scalar(value) for key, value in point.items()})
            elif isinstance(point, Sequence) and not isinstance(point, (str, bytes)):
                row = {"point_index": index}
                for axis, coordinate in enumerate(point):
                    row["q_x" if axis == 0 else "q_y" if axis == 1 else f"coordinate_{axis}"] = _scalar(coordinate)
                rows.append(row)
            else:
                rows.append({"point_index": index, "value": _scalar(point)})
        return rows
    return [{"value": _scalar(points)}]


def _ellipse_fit(value: Any) -> Any:
    return _value(value, "ellipse_fit", "ellipse", default=None)


def _walk_arrays(value: Any, prefix: str, output: dict[str, Any]) -> None:
    tolist = getattr(value, "tolist", None)
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if callable(tolist) and shape is not None and dtype is not None:
        output[prefix] = value
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _walk_arrays(item, f"{prefix}__{_safe_key(key)}", output)
        return
    if is_dataclass(value):
        for field in fields(value):
            _walk_arrays(getattr(value, field.name), f"{prefix}__{_safe_key(field.name)}", output)
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _walk_arrays(item, f"{prefix}__{index}", output)
        return
    if hasattr(value, "__dict__"):
        for key, item in vars(value).items():
            if not str(key).startswith("_"):
                _walk_arrays(item, f"{prefix}__{_safe_key(key)}", output)


def _safe_key(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return text or "value"


def _contains_omitted_array(value: Any, _visited: set[int] | None = None) -> bool:
    """Detect an explicit checkpoint marker without guessing from absence.

    Results can retain third-party metadata objects.  In particular, pyFAI
    stores detector orientation as an ``Enum`` whose private ``__objclass__``
    points back to the enum class.  Treat scalar-like values as leaves and
    only walk public object attributes so that this provenance scan cannot
    recurse through implementation details or cyclic metadata graphs.
    """

    # A numpy scalar may expose implementation metadata on some versions, but
    # it is never a container for an omission marker.  Check it before
    # ``__dict__`` traversal; Enum also covers pyFAI IntEnum members that are
    # not caught by the builtin scalar tuple below.
    if value is None or isinstance(value, (str, bytes, int, float, complex, bool, Enum, np.generic)):
        return False

    if _visited is None:
        _visited = set()
    object_id = id(value)
    if object_id in _visited:
        return False
    _visited.add(object_id)

    if isinstance(value, Mapping):
        for key in ("array_omitted", "arrays_omitted"):
            marker = value.get(key)
            if isinstance(marker, (bool, np.bool_)) and bool(marker):
                return True
        return any(_contains_omitted_array(item, _visited) for item in value.values())
    if is_dataclass(value):
        return any(
            _contains_omitted_array(getattr(value, field.name), _visited)
            for field in fields(value)
            if not field.name.startswith("_")
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_omitted_array(item, _visited) for item in value)
    if hasattr(value, "__dict__"):
        try:
            attributes = vars(value)
        except TypeError:
            return False
        return any(
            _contains_omitted_array(item, _visited)
            for key, item in attributes.items()
            if not str(key).startswith("_")
        )
    return False


def _write_npz(path: Path, results: Sequence[FrameFitResult]) -> Path:
    import numpy as np

    arrays: dict[str, Any] = {}
    missing_frames: list[int] = []
    missing_frame_ids: list[Any] = []
    missing_frame_paths: list[str] = []
    for index, item in enumerate(results):
        _walk_arrays(item.result, f"frame_{index:04d}", arrays)
        if (
            item.result is None
            or item.status != "ok"
            or _contains_omitted_array(item.result)
        ):
            missing_frames.append(index)
            missing_frame_ids.append(_scalar(item.frame.frame_id))
            missing_frame_paths.append(str(item.frame.path))
    metadata = {
        "arrays": list(arrays),
        "frame_count": len(results),
        "complete": not missing_frames,
        "missing_frames": missing_frames,
        "missing_frame_ids": missing_frame_ids,
        "missing_frame_paths": missing_frame_paths,
    }
    arrays["__metadata__"] = np.asarray(_json_text(metadata))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return path


def _write_evolution(path: Path, results: Sequence[FrameFitResult]) -> Path:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    series: dict[tuple[str, str | None], list[tuple[float, float]]] = {}
    for index, item in enumerate(results):
        x_value = item.frame.time
        try:
            x = float(x_value) if x_value is not None else float(index)
        except (TypeError, ValueError):
            x = float(index)
        if not math.isfinite(x):
            x = float(index)
        for parameter in _parameters(item.result):
            value = parameter.get("value")
            try:
                y = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(y):
                unit = parameter.get("unit")
                unit_key = str(unit).strip() if unit not in (None, "") else None
                series.setdefault((parameter["parameter"], unit_key), []).append((x, y))
    path.parent.mkdir(parents=True, exist_ok=True)
    groups: dict[str, list[tuple[str, str | None]]] = {}
    for key in series:
        name, unit = key
        # Parameters without a declared unit are kept on separate panels: a
        # missing unit is not evidence that two quantities are commensurate.
        group = unit if unit is not None else f"__unknown__:{name}"
        groups.setdefault(group, []).append(key)
    panel_count = max(1, len(groups))
    fig, axes = plt.subplots(
        nrows=panel_count,
        ncols=1,
        figsize=(9, max(5, 3.2 * panel_count)),
        squeeze=False,
        constrained_layout=True,
    )
    axes_flat = list(axes[:, 0])
    if series:
        x_label = "time" if any(item.frame.time is not None for item in results) else "frame"
        for axis, (group, keys) in zip(axes_flat, groups.items()):
            for name, unit in keys:
                points = series[(name, unit)]
                points.sort(key=lambda point: point[0])
                axis.plot(
                    [point[0] for point in points],
                    [point[1] for point in points],
                    "o-",
                    label=name,
                )
            unit_label = group if not group.startswith("__unknown__:") else "unit unspecified"
            axis.set_ylabel(f"parameter value ({unit_label})")
            axis.legend(loc="best", fontsize="small")
            axis.set_xlabel(x_label)
            axis.grid(True, alpha=0.25)
    else:
        axis = axes_flat[0]
        axis.text(0.5, 0.5, "No scalar parameter evolution available", ha="center", va="center")
        axis.set_axis_off()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


@dataclass
class ExportResult(dict[str, Path]):
    """Dictionary-like output index with a convenient output directory."""

    output_dir: Path = field(default_factory=Path)

    def __post_init__(self) -> None:
        dict.__init__(self)


def _distribution_version(*names: str) -> str | None:
    for name in names:
        try:
            version = importlib_metadata.version(name)
            return str(version) if version is not None else None
        except importlib_metadata.PackageNotFoundError:
            continue
        except Exception:  # pragma: no cover - broken local metadata
            continue
    return None


def _software_versions() -> dict[str, str | None]:
    try:
        from . import __version__ as package_version
    except Exception:  # pragma: no cover - import-time fallback
        package_version = None
    return {
        "python": platform.python_version(),
        "ButterflySAXS": (
            str(package_version)
            if package_version is not None
            else _distribution_version("butterfly-saxs", "butterfly_saxs")
        ),
        "numpy": _distribution_version("numpy"),
        "scipy": _distribution_version("scipy"),
        "fabio": _distribution_version("fabio"),
        "pyFAI": _distribution_version("pyFAI", "pyfai"),
    }


def _provenance_payload(
    *,
    created_at: str,
    mode: str,
    input_hash: str | None,
    config_hash: str | None,
    source_count: int,
    user_provenance: Mapping[str, Any] | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "created_at": created_at,
        "tool": "ButterflySAXS",
        "mode": mode,
        "input_hash": input_hash,
        "config_hash": config_hash,
        "source_count": source_count,
    }
    if user_provenance:
        safe = _json_safe(user_provenance)
        if isinstance(safe, Mapping):
            payload.update(safe)
    user_versions = payload.get("versions")
    merged_versions = dict(user_versions) if isinstance(user_versions, Mapping) else {}
    # Auto-detected versions are authoritative for the audited package set;
    # user provenance may still add other version keys.
    merged_versions.update(_software_versions())
    payload["versions"] = merged_versions
    return payload


def export_batch(
    batch: BatchRunResult | Iterable[FrameFitResult],
    output_dir: str | os.PathLike[str],
    *,
    provenance: Mapping[str, Any] | None = None,
    prefix: str = "",
) -> dict[str, Path]:
    """Write CSV, JSON/JSONL, NPZ and evolution-plot exports.

    The returned mapping uses stable logical keys (``frame_summary``,
    ``parameters_long``, ``ridge_points``, ``ellipse_fit``, ``npz``, and
    ``evolution_png``) and points to the actual files.
    """

    # Also tolerate export_batch(output_dir, batch), a common notebook form.
    if isinstance(batch, (str, os.PathLike, Path)) and not isinstance(output_dir, (str, os.PathLike, Path)):
        batch, output_dir = output_dir, batch
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    results = _frame_results(batch)
    stem = f"{prefix}" if prefix else ""

    frame_summary = output / f"{stem}frame_summary.csv"
    _write_csv(
        frame_summary,
        _frame_summary_rows(results),
        columns=[
            "frame_index",
            "frame_id",
            "path",
            "time",
            "status",
            "error",
            "warm_start_from",
            "elapsed_s",
            "resumed",
            "flags",
            "scientific_flags",
            "parameters_json",
        ],
    )

    parameter_rows: list[dict[str, Any]] = []
    for index, item in enumerate(results):
        base = _frame_base(item, index)
        result_flags = _result_flags(item.result)
        for parameter in _parameters(item.result):
            row = {
                **base,
                **parameter,
                "scientific_flags": _json_text(result_flags),
            }
            parameter_rows.append(row)
    parameters_long = output / f"{stem}parameters_long.csv"
    _write_csv(
        parameters_long,
        parameter_rows,
        columns=[
            "frame_index",
            "frame_id",
            "path",
            "time",
            "status",
            "parameter",
            "value",
            "stderr",
            "uncertainty",
            "fixed",
            "unit",
            "flags",
            "bound_flags",
            "scientific_flags",
        ],
    )

    ridge_rows: list[dict[str, Any]] = []
    for index, item in enumerate(results):
        for point_index, point in enumerate(_ridge_points(item.result)):
            ridge_rows.append({
                **_frame_base(item, index),
                "point_index": point_index,
                **point,
            })
    ridge_points = output / f"{stem}ridge_points.csv"
    _write_csv(ridge_points, ridge_rows)

    ellipse_rows = []
    for index, item in enumerate(results):
        ellipse_rows.append({
            **_frame_base(item, index),
            "ellipse_fit": _json_safe(_ellipse_fit(item.result)),
        })
    ellipse_fit = output / f"{stem}ellipse_fit.json"
    ellipse_fit.write_text(
        json.dumps({"frames": ellipse_rows}, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    ellipse_jsonl = output / f"{stem}ellipse_fit.jsonl"
    with ellipse_jsonl.open("w", encoding="utf-8", newline="\n") as handle:
        for row in ellipse_rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")

    if isinstance(batch, BatchRunResult):
        batch_manifest = batch.manifest
        input_hash = batch.input_hash
        config_hash = batch.config_hash
        mode = batch.mode
    else:
        batch_manifest = None
        input_hash = None
        config_hash = None
        mode = "independent"
    created_at = datetime.now(timezone.utc).isoformat()
    provenance_value = _provenance_payload(
        created_at=created_at,
        mode=mode,
        input_hash=input_hash,
        config_hash=config_hash,
        source_count=len(results),
        user_provenance=provenance,
    )
    manifest_value = {
        "created_at": created_at,
        "mode": mode,
        "input_hash": input_hash,
        "config_hash": config_hash,
        "frames": [_json_safe(item.frame.to_dict()) for item in results],
        "user_manifest": _json_safe(batch_manifest),
        "provenance": provenance_value,
    }
    manifest_path = output / f"{stem}manifest.json"
    manifest_path.write_text(json.dumps(manifest_value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    provenance_path = output / f"{stem}provenance.json"
    provenance_path.write_text(json.dumps(provenance_value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    npz_path = output / f"{stem}results.npz"
    _write_npz(npz_path, results)
    evolution_path = output / f"{stem}evolution.png"
    _write_evolution(evolution_path, results)
    return {
        "frame_summary": frame_summary,
        "parameters_long": parameters_long,
        "ridge_points": ridge_points,
        "ellipse_fit": ellipse_fit,
        "ellipse_fit_jsonl": ellipse_jsonl,
        "manifest": manifest_path,
        "provenance": provenance_path,
        "npz": npz_path,
        "evolution_png": evolution_path,
    }


export_results = export_batch
write_exports = export_batch
export_batch_results = export_batch


__all__ = [
    "ExportResult",
    "export_batch",
    "export_batch_results",
    "export_results",
    "write_exports",
]
