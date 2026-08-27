"""Batch orchestration primitives for LamellarSAXS2D.

The batch layer deliberately knows very little about the scientific fitting
implementation.  A caller supplies an ``analyze_frame`` callable and gets a
stable, serialisable record for every input frame.  This keeps the UI, CLI and
in-situ workflows on the same seam and also makes it possible to test the
orchestration with a small fake analyser.
"""

from __future__ import annotations

import csv
import hashlib
import inspect
import json
import math
import os
import re
import tempfile
import traceback as traceback_module
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal


_NATURAL_PART = re.compile(r"(\d+)")
_MISSING = object()
_PATH_NAMES = {"source", "path", "file", "filename", "filepath", "input", "input_path"}
_CONFIG_NAMES = {"config", "analysis_config", "settings", "options"}
_INITIAL_NAMES = {
    "initial",
    "initial_parameters",
    "initial_result",
    "previous",
    "previous_result",
    "prior",
    "prior_result",
    "warm_start",
    "warm_start_result",
    "seed",
    "seed_result",
}


def _json_safe(value: Any) -> Any:
    """Return a JSON-compatible copy without losing nested scientific flags."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    if is_dataclass(value):
        return {
            item.name: _json_safe(getattr(value, item.name))
            for item in fields(value)
        }
    # numpy is an optional import for the batch layer.  Duck typing keeps the
    # module useful in lightweight CLI environments and for test doubles.
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return _json_safe(tolist())
        except Exception:  # pragma: no cover - defensive for unusual arrays
            pass
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_safe(item())
        except Exception:  # pragma: no cover
            pass
    for method_name in ("to_dict", "as_dict", "to_mapping"):
        method = getattr(value, method_name, None)
        if callable(method):
            try:
                try:
                    converted = method(include_specs=True)
                except TypeError:
                    converted = method()
            except TypeError:
                try:
                    converted = method(resolved=False)
                except Exception:  # pragma: no cover
                    continue
            except Exception:  # pragma: no cover
                continue
            if converted is not value:
                return _json_safe(converted)
    if hasattr(value, "__dict__"):
        try:
            return _json_safe(
                {key: item for key, item in vars(value).items() if not key.startswith("_")}
            )
        except Exception:  # pragma: no cover
            pass
    return repr(value)


def _named_value(value: Any, name: str) -> Any:
    """Read one public result field without invoking mapping conversion."""

    if isinstance(value, Mapping):
        return value.get(name, _MISSING)
    try:
        return getattr(value, name)
    except (AttributeError, KeyError, TypeError):
        return _MISSING


def _checkpoint_safe(value: Any) -> Any:
    """Serialize restart state without embedding detector-sized arrays.

    A checkpoint is control state, not the scientific evidence archive.  Full
    image/model/residual/q-map arrays are written by :mod:`butterfly_saxs.export`;
    here they are represented by a small shape/dtype/range descriptor so an
    in-situ run cannot grow the checkpoint by several images per frame.
    """

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is not None and dtype is not None:
        summary: dict[str, Any] = {
            "array_omitted": True,
            "array_omission_reason": "detector_array_not_stored_in_checkpoint",
            "shape": [int(item) for item in shape],
            "dtype": str(dtype),
        }
        try:
            import numpy as np

            array = np.asarray(value)
            finite = array[np.isfinite(array)] if array.dtype.kind in "fciu" else np.asarray([])
            if finite.size:
                summary["min"] = float(np.min(finite))
                summary["max"] = float(np.max(finite))
        except Exception:  # pragma: no cover - optional diagnostics only
            pass
        return summary
    if isinstance(value, Mapping):
        return {str(key): _checkpoint_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_checkpoint_safe(item) for item in value]
    if is_dataclass(value):
        converted = {
            item.name: _checkpoint_safe(getattr(value, item.name))
            for item in fields(value)
        }
        # PipelineResult exposes the longitudinal parameters as a property,
        # not as a dataclass field.  Keep that property in the restart state
        # alongside the nested full2d parameters.
        parameters = _named_value(value, "parameters")
        if parameters is not _MISSING and "parameters" not in converted:
            converted["parameters"] = _checkpoint_safe(parameters)
        return converted
    # Prefer a public mapping method for result objects, but merge back public
    # restart/array attributes that a compact mapping may intentionally omit.
    for method_name in ("to_mapping", "to_dict", "as_dict"):
        method = getattr(value, method_name, None)
        if not callable(method):
            continue
        try:
            if method_name == "to_mapping":
                converted = method(include_arrays=False)
            else:
                converted = method(include_specs=True)
        except TypeError:
            try:
                converted = method()
            except Exception:  # pragma: no cover
                continue
        except Exception:  # pragma: no cover
            continue
        if converted is value:
            continue
        if isinstance(converted, Mapping):
            merged = dict(converted)
            parameters = _named_value(value, "parameters")
            if parameters is _MISSING:
                parameters = _named_value(value, "params")
            if parameters is not _MISSING and "parameters" not in merged:
                merged["parameters"] = parameters
            # Compact result mappings often contain only array summaries.  If
            # the original object exposes the payload, replace that summary
            # with the explicit omission descriptor generated above.
            for name in (
                "image",
                "qmap",
                "valid_mask",
                "model",
                "residual",
                "model_image",
                "residual_image",
                "full2d",
            ):
                original = _named_value(value, name)
                if original is not _MISSING:
                    merged[name] = original
            return _checkpoint_safe(merged)
    if hasattr(value, "__dict__"):
        return _checkpoint_safe(
            {key: item for key, item in vars(value).items() if not key.startswith("_")}
        )
    return repr(value)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _json_safe(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _hash_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def natural_sort_key(value: str | os.PathLike[str]) -> tuple[Any, ...]:
    """Build a case-insensitive natural-sort key (``frame2`` before ``frame10``)."""

    text = str(value)
    parts: list[tuple[int, Any]] = []
    for part in _NATURAL_PART.split(text):
        if not part:
            continue
        if part.isdigit():
            parts.append((0, int(part)))
        else:
            parts.append((1, part.casefold()))
    return tuple(parts)


def _canonical_path(value: str | os.PathLike[str] | Path) -> str:
    """Return a stable, case-normalised path for identity/fingerprint use."""

    try:
        path = Path(value).expanduser().resolve(strict=False)
    except (OSError, RuntimeError):
        path = Path(os.path.abspath(os.fspath(value)))
    return os.path.normcase(path.as_posix()).replace("\\", "/")


def _coerce_order(value: Any | None) -> int | float | None:
    """Convert an input order to a finite numeric value at the boundary."""

    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    if isinstance(value, bool):
        raise ValueError("order must be a finite number")
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("order must be a finite number") from exc
    if not math.isfinite(numeric):
        raise ValueError("order must be a finite number")
    if numeric.is_integer():
        return int(numeric)
    return numeric


def _as_ref(value: Any, *, order: int | float | None = None) -> "FrameRef":
    if isinstance(value, FrameRef):
        if order is None or value.order is not None:
            return value
        return FrameRef(
            value.path,
            time=value.time,
            frame_id=value.frame_id,
            metadata=value.metadata,
            order=order,
            source=value.source,
            dataset=value.dataset,
            frame=value.frame,
        )
    if isinstance(value, Mapping):
        path = value.get("path")
        if path is None:
            path = value.get("input_path", value.get("file", value.get("filename")))
        if path is None:
            raise ValueError("manifest frame entry is missing path")
        known = {
            "path",
            "input_path",
            "file",
            "filename",
            "time",
            "timestamp",
            "frame_id",
            "id",
            "metadata",
            "order",
            "source",
            "dataset",
            "dataset_id",
            "dataset_name",
            "frame",
            "frame_index",
        }
        metadata = dict(value.get("metadata") or {})
        metadata.update({str(k): v for k, v in value.items() if k not in known})
        entry_order = value.get("order", order)
        return FrameRef(
            path,
            time=value.get("time", value.get("timestamp")),
            frame_id=value.get("frame_id", value.get("id")),
            metadata=metadata,
            order=entry_order,
            source=value.get("source"),
            dataset=value.get("dataset", value.get("dataset_id", value.get("dataset_name"))),
            frame=value.get("frame", value.get("frame_index")),
        )
    return FrameRef(value, order=order)


@dataclass(frozen=True, init=False)
class FrameRef:
    """Reference to one detector frame and its optional acquisition metadata."""

    path: Path | str
    time: float | int | str | None = None
    frame_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    order: int | float | None = None
    source: str | None = None
    dataset: str | None = None
    frame: int | None = None

    def __init__(
        self,
        path: Path | str | None = None,
        time: float | int | str | None = None,
        frame_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        order: int | float | str | None = None,
        source: str | None = None,
        dataset: str | None = None,
        frame: int | None = None,
        *,
        input_path: Path | str | None = None,
        timestamp: float | int | str | None = None,
        id: str | None = None,
        index: int | None = None,
        frame_index: int | None = None,
    ) -> None:
        """Create a frame reference, accepting common timestamp/index aliases."""

        if path is None:
            path = input_path
        if path is None:
            raise TypeError("FrameRef requires path or input_path")
        if time is None:
            time = timestamp
        if frame_id is None:
            frame_id = id
        if order is None:
            order = index
        order = _coerce_order(order)
        if frame is None:
            frame = frame_index
        if frame is not None:
            if isinstance(frame, bool):
                raise ValueError("frame selector must be a non-negative integer")
            try:
                numeric_frame = int(frame)
                numeric_value = float(frame)
            except (TypeError, ValueError) as exc:
                raise ValueError("frame selector must be a non-negative integer") from exc
            if not math.isfinite(numeric_value) or numeric_value != numeric_frame:
                raise ValueError("frame selector must be a non-negative integer")
            frame = numeric_frame
            if frame < 0:
                raise ValueError("frame selector must be a non-negative integer")
        object.__setattr__(self, "path", Path(path))
        object.__setattr__(self, "time", time)
        object.__setattr__(
            self,
            "frame_id",
            frame_id if frame_id is not None else Path(path).stem,
        )
        object.__setattr__(self, "metadata", dict(metadata or {}))
        object.__setattr__(self, "order", order)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "dataset", dataset)
        object.__setattr__(self, "frame", frame)

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))
        if self.frame_id is None:
            object.__setattr__(self, "frame_id", self.path.stem)
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @property
    def input_path(self) -> Path:
        return self.path

    @property
    def timestamp(self) -> float | int | str | None:
        return self.time

    @property
    def id(self) -> str:
        return str(self.frame_id)

    @property
    def index(self) -> int | float | None:
        return self.order

    @property
    def frame_selector(self) -> int | str | None:
        """Return the selector understood by the service for this frame."""

        value = self.frame
        if value is None:
            for name in ("frame", "frame_index"):
                candidate = self.metadata.get(name)
                if candidate is not None:
                    value = candidate
                    break
        if value is None or (isinstance(value, str) and not value.strip()):
            return None
        try:
            numeric = int(value)
            numeric_value = float(value)
        except (TypeError, ValueError):
            # The service will report malformed metadata when it is used; keep
            # its raw identity here rather than collapsing two requests.
            return str(value)
        if not math.isfinite(numeric_value) or numeric_value != numeric:
            return str(value)
        return numeric

    @property
    def dataset_id(self) -> str:
        value = self.dataset
        if value is None:
            for name in ("dataset", "dataset_id", "dataset_name"):
                candidate = self.metadata.get(name)
                if candidate is not None:
                    value = candidate
                    break
        return "" if value is None else str(value)

    @property
    def key(self) -> str:
        return _canonical_json(
            {
                "path": _canonical_path(self.path),
                "frame": self.frame_selector,
                "frame_id": self.id,
                "dataset": self.dataset_id,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "frame_id": self.frame_id,
            "time": self.time,
            "metadata": _json_safe(self.metadata),
            "order": self.order,
            "source": self.source,
            "dataset": self.dataset,
            "frame": self.frame,
        }


def _manifest_rows(manifest: Any) -> list[Any]:
    if manifest is None:
        return []
    if isinstance(manifest, (str, os.PathLike)):
        path = Path(manifest)
        if path.suffix.casefold() == ".csv":
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                loaded_manifest: Any = list(csv.DictReader(handle))
        else:
            with path.open("r", encoding="utf-8") as handle:
                loaded_manifest = json.load(handle)
        rows = _manifest_rows(loaded_manifest)
        resolved_rows: list[Any] = []
        for row in rows:
            if not isinstance(row, Mapping):
                resolved_rows.append(row)
                continue
            resolved = dict(row)
            for key in ("path", "input_path", "file", "filename"):
                candidate = resolved.get(key)
                if candidate is None or not isinstance(candidate, (str, os.PathLike)):
                    continue
                candidate_path = Path(candidate)
                if not candidate_path.is_absolute():
                    resolved[key] = os.fspath(path.parent / candidate_path)
                break
            resolved_rows.append(resolved)
        return resolved_rows
    if isinstance(manifest, Mapping):
        for key in ("frames", "frame_manifest", "manifest", "data", "items"):
            if key in manifest and isinstance(manifest[key], Sequence) and not isinstance(
                manifest[key], (str, bytes)
            ):
                return list(manifest[key])
        # A mapping from filename to metadata is a convenient manifest form.
        rows: list[dict[str, Any]] = []
        for path, metadata in manifest.items():
            if isinstance(metadata, Mapping):
                row = dict(metadata)
            else:
                row = {"time": metadata}
            row.setdefault("path", path)
            rows.append(row)
        return rows
    if isinstance(manifest, Sequence) and not isinstance(manifest, (str, bytes)):
        return list(manifest)
    raise TypeError("manifest must be a path, mapping, or sequence")


def _expand_inputs(inputs: Any) -> list[Any]:
    if isinstance(inputs, (str, os.PathLike, Path)):
        path = Path(inputs)
        if path.is_dir():
            return [item for item in path.iterdir() if item.is_file()]
        # A wildcard is useful for CLI callers; a literal missing path is kept
        # so a loader/analyser can report the failure per frame.
        if any(char in str(path) for char in "*?["):
            return list(path.parent.glob(path.name))
        return [path]
    paths: list[Any] = []
    for item in inputs or []:
        if isinstance(item, (str, os.PathLike, Path)):
            paths.append(Path(item))
        elif isinstance(item, FrameRef):
            paths.append(item)
        elif isinstance(item, Mapping):
            paths.append(_as_ref(item))
        else:
            raise TypeError(f"unsupported frame input: {type(item)!r}")
    return paths


def build_frame_refs(
    inputs: Iterable[Any] | Any,
    manifest: Any = None,
) -> list[FrameRef]:
    """Resolve frames using manifest order/time, otherwise natural filename order.

    A manifest is authoritative: entries in it are returned in explicit
    ``order`` order when supplied, otherwise by numeric ``time`` when present,
    and otherwise in manifest order.  Without a manifest paths use natural
    sorting.  This policy prevents lexical ordering from silently scrambling
    in-situ kinetics while retaining the convenient ``frame2``/``frame10``
    behaviour for directory scans.
    """

    rows = _manifest_rows(manifest)
    if rows:
        refs = [_as_ref(row, order=index) for index, row in enumerate(rows)]

        def explicit_key(item: FrameRef) -> tuple[int, Any, int]:
            if item.order is not None:
                try:
                    return (0, float(item.order), refs.index(item))
                except (TypeError, ValueError):
                    pass
            if item.time is not None:
                try:
                    return (1, float(item.time), refs.index(item))
                except (TypeError, ValueError):
                    pass
            return (2, refs.index(item), refs.index(item))

        # If the manifest has no explicit order values, numeric acquisition
        # time is the next strongest ordering signal.  ``order`` generated by
        # _as_ref is only the source position, not an explicit user order.
        has_explicit_order = any(
            (
                isinstance(row, Mapping)
                and row.get("order") is not None
                and not (
                    isinstance(row.get("order"), str)
                    and not str(row.get("order")).strip()
                )
            )
            or (isinstance(row, FrameRef) and row.order is not None)
            for row in rows
        )
        has_time = any(item.time is not None for item in refs)
        if has_explicit_order:
            refs.sort(key=lambda item: (item.order is None, item.order or 0))
        elif has_time:
            def time_key(item: FrameRef) -> tuple[int, float, int]:
                try:
                    numeric = float(item.time)  # type: ignore[arg-type]
                    return (0, numeric, item.order or 0)
                except (TypeError, ValueError):
                    return (1, float("inf"), item.order or 0)

            refs.sort(key=time_key)
        return refs

    refs: list[FrameRef] = []
    for item in (inputs if isinstance(inputs, Sequence) and not isinstance(inputs, (str, bytes, Path)) else _expand_inputs(inputs)):
        # Source-list position is not part of natural ordering or resume
        # identity; preserving it would make an equivalent reordered input
        # list produce a spurious checkpoint hash mismatch.
        refs.append(_as_ref(item))
    refs.sort(key=lambda item: natural_sort_key(item.path.name))
    return refs


# Public aliases used by the CLI and by older prototype notebooks.
make_frame_refs = build_frame_refs
resolve_frame_refs = build_frame_refs
discover_frames = build_frame_refs


def input_fingerprint(refs: Iterable[FrameRef]) -> str:
    records: list[dict[str, Any]] = []
    for ref in refs:
        path = Path(ref.path)
        stat: dict[str, Any] = {"exists": path.exists()}
        if path.exists():
            try:
                info = path.stat()
                stat.update({"size": info.st_size, "mtime_ns": info.st_mtime_ns})
                # Size/mtime alone can be unchanged by an in-place rewrite.
                # Stream a SHA-256 digest in bounded chunks so a checkpoint
                # cannot silently resume against different detector bytes.
                digest = hashlib.sha256()
                with path.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
                stat.update(
                    {
                        "content_hash_algorithm": "sha256",
                        "content_sha256": digest.hexdigest(),
                    }
                )
            except OSError:
                # Keep the stat identity when content access is unavailable;
                # the explicit marker makes the reduced guarantee auditable.
                stat["content_hash_algorithm"] = "sha256"
                stat["content_sha256"] = None
                stat["content_hash_unavailable"] = True
        ref_record = ref.to_dict()
        ref_record["path"] = _canonical_path(path)
        records.append({"ref": ref_record, "file": stat})
    return _hash_json(records)


def config_fingerprint(config: Any = None, *, mode: str = "independent") -> str:
    return _hash_json({"mode": mode, "config": config})


_QUALITY_FAILURE_STATUSES = {
    "error",
    "failed",
    "failure",
    "invalid",
    "insufficient_data",
}
_QUALITY_FAILURE_FLAG_PREFIXES = (
    "intensity_fit_failed:",
    "analysis_validation_failed:",
)
_EMPTY_OBSERVATION_FLAGS = {"no_observed"}


def _is_explicit_false(value: Any) -> bool:
    if isinstance(value, bool):
        return value is False
    # Avoid importing numpy solely for a scalar bool while still handling
    # numpy.bool_(False) returned by scientific result objects.
    if type(value).__name__ == "bool_":
        try:
            return not bool(value)
        except Exception:  # pragma: no cover - unusual scalar proxy
            return False
    return False


def _is_failure_status(value: Any) -> bool:
    return isinstance(value, str) and value.strip().casefold() in _QUALITY_FAILURE_STATUSES


def _quality_failure_flag(value: Any) -> str | None:
    """Return the first explicit batch-failing flag in a flag container."""

    if isinstance(value, str):
        flag = value.strip()
        if flag.casefold() in _EMPTY_OBSERVATION_FLAGS:
            return flag
        if any(flag.startswith(prefix) for prefix in _QUALITY_FAILURE_FLAG_PREFIXES):
            return flag
        return None
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key).strip()
            if key_text.casefold() in _EMPTY_OBSERVATION_FLAGS and item is True:
                return key_text
            if key_text in {prefix.rstrip(":") for prefix in _QUALITY_FAILURE_FLAG_PREFIXES}:
                if item is True:
                    return key_text
                if isinstance(item, str) and item.strip():
                    return f"{key_text}:{item.strip()}"
            failure = _quality_failure_flag(item)
            if failure is not None:
                return failure
        return None
    if isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            failure = _quality_failure_flag(item)
            if failure is not None:
                return failure
    return None


def _observed_array_failure(name: str, observed: Any) -> str | None:
    """Reject empty/all-NaN observed arrays, including compact checkpoints."""

    # ``_checkpoint_safe`` replaces arrays with a shape/dtype/range summary.
    # Keep enough information in that summary to reject a legacy checkpoint
    # that mislabeled an empty or all-NaN observed array as successful.
    if isinstance(observed, Mapping) and observed.get("array_omitted") is True:
        shape = observed.get("shape")
        if isinstance(shape, (list, tuple)):
            try:
                if any(int(dimension) == 0 for dimension in shape):
                    return f"{name}.observed.size=0"
            except (TypeError, ValueError, OverflowError):
                pass
        dtype = observed.get("dtype")
        if isinstance(dtype, str) and "min" not in observed and "max" not in observed:
            try:
                import numpy as np

                if np.dtype(dtype).kind in "fc":
                    return f"{name}.observed.all_nan"
            except (TypeError, ValueError):
                pass

    size = getattr(observed, "size", _MISSING)
    if size is _MISSING:
        return None
    try:
        if int(size) == 0:
            return f"{name}.observed.size=0"
    except (TypeError, ValueError, OverflowError):
        return None

    try:
        import numpy as np

        array = np.asarray(observed)
        if array.dtype.kind in "fc" and bool(np.isnan(array).all()):
            return f"{name}.observed.all_nan"
    except (ImportError, TypeError, ValueError):
        # The batch layer stays import-light when numpy is unavailable or a
        # third-party array proxy does not support numpy's NaN predicate.
        pass
    return None


def _empty_observation_failure(name: str, result: Any) -> str | None:
    """Return explicit empty-observation signals from one result stage."""

    for field_name in ("ndata", "n_data"):
        value = _named_value(result, field_name)
        if value is _MISSING or value is None or isinstance(value, bool):
            continue
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric_value) and numeric_value == 0:
            return f"{name}.{field_name}=0"
    observed = _named_value(result, "observed")
    if observed is _MISSING:
        return None
    if observed is None:
        return f"{name}.observed=None"
    return _observed_array_failure(name, observed)


def _nested_quality_failure(name: str, result: Any) -> str | None:
    """Check one named result stage for explicit failure signals."""

    if result is _MISSING or result is None:
        return None
    success = _named_value(result, "success")
    if _is_explicit_false(success):
        return f"{name}.success=False"
    status = _named_value(result, "status")
    if _is_failure_status(status):
        return f"{name}.status={status}"
    flag = _quality_failure_flag(_named_value(result, "flags"))
    if flag is not None:
        return f"{name}.flags={flag}"
    empty_observation = _empty_observation_failure(name, result)
    if empty_observation is not None:
        return empty_observation
    return None


def _quality_failure_reason(result: Any) -> str | None:
    """Return an explicit result failure, without inventing numeric cutoffs."""

    if result is None:
        return "result=None"
    success = _named_value(result, "success")
    if _is_explicit_false(success):
        return "success=False"
    status = _named_value(result, "status")
    if _is_failure_status(status):
        return f"status={status}"

    for name in ("metrics", "ellipse_fit", "full2d"):
        failure = _nested_quality_failure(name, _named_value(result, name))
        if failure is not None:
            return failure
    flag = _quality_failure_flag(_named_value(result, "flags"))
    if flag is not None:
        return f"flags={flag}"
    empty_observation = _empty_observation_failure("result", result)
    if empty_observation is not None:
        return empty_observation
    return None


def _warm_start_seed(result: Any) -> Any:
    """Extract the parameter state expected by an analyzer when available."""

    for name in ("parameters", "params"):
        parameters = _named_value(result, name)
        if parameters is not _MISSING and parameters is not None:
            return parameters
    full2d = _named_value(result, "full2d")
    if full2d is not _MISSING and full2d is not None:
        parameters = _named_value(full2d, "parameters")
        if parameters is not _MISSING and parameters is not None:
            return parameters
    # Keep compatibility with lightweight analyzers whose result is already
    # the warm-start state rather than a result envelope.
    return result


@dataclass(init=False)
class FrameFitResult:
    """One isolated frame outcome, including failures and warm-start lineage."""

    frame: FrameRef
    result: Any = None
    status: Literal["ok", "failed", "skipped"] = "ok"
    error: str | None = None
    traceback: str | None = None
    warm_start_from: str | None = None
    elapsed_s: float | None = None
    resumed: bool = False

    def __init__(
        self,
        frame: FrameRef | str | os.PathLike[str] | None = None,
        result: Any = None,
        status: Literal["ok", "failed", "skipped"] = "ok",
        error: str | None = None,
        traceback: str | None = None,
        warm_start_from: str | None = None,
        elapsed_s: float | None = None,
        resumed: bool = False,
        *,
        frame_ref: FrameRef | str | os.PathLike[str] | None = None,
        ref: FrameRef | str | os.PathLike[str] | None = None,
        fit_result: Any = _MISSING,
        fit: Any = _MISSING,
        lineage: str | None = None,
        path: str | os.PathLike[str] | None = None,
    ) -> None:
        if frame is None:
            frame = frame_ref if frame_ref is not None else ref
        if frame is None:
            frame = path
        if frame is None:
            raise TypeError("FrameFitResult requires frame/frame_ref/ref/path")
        if not isinstance(frame, FrameRef):
            frame = FrameRef(frame)
        if fit_result is not _MISSING:
            result = fit_result
        elif fit is not _MISSING:
            result = fit
        if lineage is not None and warm_start_from is None:
            warm_start_from = lineage
        self.frame = frame
        self.result = result
        self.status = status
        self.error = error
        self.traceback = traceback
        self.warm_start_from = warm_start_from
        self.elapsed_s = elapsed_s
        self.resumed = bool(resumed)

    @property
    def frame_ref(self) -> FrameRef:
        return self.frame

    @property
    def fit_result(self) -> Any:
        return self.result

    @property
    def ref(self) -> FrameRef:
        return self.frame

    @property
    def fit(self) -> Any:
        return self.result

    @property
    def analysis_result(self) -> Any:
        return self.result

    @property
    def lineage(self) -> str | None:
        return self.warm_start_from

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    @property
    def success(self) -> bool:
        return self.ok

    @property
    def failed(self) -> bool:
        return self.status == "failed"

    @property
    def parameters(self) -> Any:
        value = self.result
        if isinstance(value, Mapping):
            return value.get("parameters", value.get("params", value))
        return getattr(value, "parameters", getattr(value, "params", None))

    def to_record(self) -> dict[str, Any]:
        return {
            "frame": self.frame.to_dict(),
            "status": self.status,
            "error": self.error,
            "traceback": self.traceback,
            "warm_start_from": self.warm_start_from,
            "elapsed_s": self.elapsed_s,
            "resumed": self.resumed,
            "result": _json_safe(self.result),
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "FrameFitResult":
        frame_value = record.get("frame", record)
        return cls(
            frame=_as_ref(frame_value),
            result=record.get("result"),
            status=record.get("status", "ok"),
            error=record.get("error"),
            traceback=record.get("traceback"),
            warm_start_from=record.get("warm_start_from", record.get("lineage")),
            elapsed_s=record.get("elapsed_s"),
            resumed=bool(record.get("resumed", False)),
        )


@dataclass
class BatchRunResult:
    frame_results: list[FrameFitResult]
    mode: str = "independent"
    input_hash: str | None = None
    config_hash: str | None = None
    checkpoint: Path | None = None
    manifest: Any = None

    # Sequence behaviour makes this object drop-in compatible with the early
    # prototype API, which returned ``list[FrameFitResult]``.
    def __iter__(self):
        return iter(self.frame_results)

    def __len__(self) -> int:
        return len(self.frame_results)

    def __getitem__(self, index: int) -> FrameFitResult:
        return self.frame_results[index]

    @property
    def results(self) -> list[FrameFitResult]:
        return self.frame_results

    @property
    def successful(self) -> list[FrameFitResult]:
        return [item for item in self.frame_results if item.ok]

    @property
    def failures(self) -> list[FrameFitResult]:
        return [item for item in self.frame_results if item.failed]

    def to_records(self) -> list[dict[str, Any]]:
        return [item.to_record() for item in self.frame_results]


def _call_analyzer(
    analyze_frame: Callable[..., Any],
    frame: FrameRef,
    initial: Any,
    *,
    warm_start: bool,
    config: Any = None,
) -> Any:
    """Call analyzers with a small, explicit compatibility seam.

    Supported signatures include ``fn(frame)``, ``fn(path)``,
    ``fn(frame, initial)``, and keyword forms such as
    ``fn(frame, warm_start=...)``.  Signature inspection avoids catching a
    TypeError raised *inside* the user analyser and accidentally running it a
    second time.
    """

    try:
        signature = inspect.signature(analyze_frame)
    except (TypeError, ValueError):
        return analyze_frame(frame)

    parameters = list(signature.parameters.values())
    positional = [
        item
        for item in parameters
        if item.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    accepts_var_keyword = any(
        item.kind == inspect.Parameter.VAR_KEYWORD for item in parameters
    )
    first_name = positional[0].name.casefold() if positional else "frame"
    first_value = frame.path if first_name in _PATH_NAMES else frame
    args: list[Any] = [first_value]
    kwargs: dict[str, Any] = {}
    supplied_initial = False
    supplied_config = False
    for parameter in parameters[1:]:
        name = parameter.name.casefold()
        if name in _CONFIG_NAMES:
            value = config
            supplied_config = True
        elif name in _INITIAL_NAMES:
            value = initial
            supplied_initial = True
        else:
            continue
        if parameter.kind == inspect.Parameter.POSITIONAL_ONLY:
            args.append(value)
        elif parameter.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            kwargs[parameter.name] = value
    # A generic second positional argument is conventionally the warm-start
    # state.  This keeps ``fn(frame, initial=None)`` concise while named
    # config/initial arguments above remain unambiguous.
    if warm_start and not supplied_initial:
        unknown_positional = [
            item
            for item in positional[1:]
            if item.name.casefold() not in _CONFIG_NAMES
        ]
        if unknown_positional:
            parameter = unknown_positional[0]
            if parameter.kind == inspect.Parameter.POSITIONAL_ONLY:
                args.append(initial)
            else:
                kwargs[parameter.name] = initial
            supplied_initial = True
        elif any(item.kind == inspect.Parameter.VAR_POSITIONAL for item in parameters):
            args.append(initial)
            supplied_initial = True
    if accepts_var_keyword:
        if config is not None and not supplied_config:
            kwargs["config"] = config
        if warm_start and not supplied_initial:
            kwargs["initial_parameters"] = initial
    return analyze_frame(*args, **kwargs)


def _checkpoint_payload(
    run: BatchRunResult,
    refs: Sequence[FrameRef],
    *,
    config: Any,
) -> dict[str, Any]:
    frame_records: list[dict[str, Any]] = []
    for item in run.frame_results:
        record = item.to_record()
        safe_result = _checkpoint_safe(item.result)
        if isinstance(safe_result, Mapping):
            safe_result = dict(safe_result)
            # Preserve both the public top-level parameters and the nested
            # full2d parameters used by longitudinal fitting.  Some result
            # classes expose the former as a property rather than a mapping
            # field, so it must be copied explicitly at this boundary.
            parameters = _named_value(item.result, "parameters")
            if parameters is _MISSING:
                parameters = _named_value(item.result, "params")
            if parameters is not _MISSING and "parameters" not in safe_result:
                safe_result["parameters"] = _checkpoint_safe(parameters)
        record["result"] = safe_result
        frame_records.append(record)
    return {
        "version": 1,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "mode": run.mode,
        "input_hash": run.input_hash,
        "config_hash": run.config_hash,
        "config": _json_safe(config),
        "frames": frame_records,
        "frame_count": len(refs),
    }


def write_checkpoint(path: str | os.PathLike[str], payload: Mapping[str, Any]) -> Path:
    """Write a checkpoint with replace-on-success semantics."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    # NamedTemporaryFile is used in the target directory so os.replace remains
    # atomic on Windows as well as POSIX filesystems.
    fd, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(_json_safe(payload), handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    return target


def read_checkpoint(path: str | os.PathLike[str]) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("checkpoint must contain a JSON object")
    return value


def run_batch(
    frames: Iterable[Any] | Any = (),
    analyze_frame: Callable[..., Any] | None = None,
    *,
    mode: Literal["independent", "warm_start"] = "independent",
    config: Any = None,
    manifest: Any = None,
    checkpoint: str | os.PathLike[str] | None = None,
    checkpoint_path: str | os.PathLike[str] | None = None,
    resume: bool = False,
    strategy: Literal["independent", "warm_start"] | None = None,
) -> BatchRunResult:
    """Analyze a sequence of frames with failure isolation and optional resume."""

    # A few callers naturally write run_batch(analyze_frame, frames).  Keep
    # that form harmlessly supported while retaining the documented order.
    if callable(frames) and analyze_frame is not None and not callable(analyze_frame):
        frames, analyze_frame = analyze_frame, frames
    if analyze_frame is None or not callable(analyze_frame):
        raise TypeError("analyze_frame callable is required")
    if strategy is not None:
        mode = strategy
    if mode not in {"independent", "warm_start"}:
        raise ValueError("mode must be 'independent' or 'warm_start'")
    if checkpoint is not None and checkpoint_path is not None and Path(checkpoint) != Path(checkpoint_path):
        raise ValueError("checkpoint and checkpoint_path refer to different files")
    checkpoint_file = Path(checkpoint_path or checkpoint) if (checkpoint_path or checkpoint) else None

    refs = build_frame_refs(frames, manifest=manifest)
    input_hash = input_fingerprint(refs)
    config_hash = config_fingerprint(config, mode=mode)
    run = BatchRunResult(
        frame_results=[],
        mode=mode,
        input_hash=input_hash,
        config_hash=config_hash,
        checkpoint=checkpoint_file,
        manifest=manifest,
    )

    prior_records: dict[str, Mapping[str, Any]] = {}
    if resume:
        if checkpoint_file is None or not checkpoint_file.exists():
            raise FileNotFoundError("resume requested but checkpoint does not exist")
        payload = read_checkpoint(checkpoint_file)
        if payload.get("input_hash") != input_hash:
            raise ValueError("checkpoint input hash mismatch; refusing resume")
        if payload.get("config_hash") != config_hash:
            raise ValueError("checkpoint config hash mismatch; refusing resume")
        if payload.get("mode", mode) != mode:
            raise ValueError("checkpoint mode mismatch; refusing resume")
        for record in payload.get("frames", []):
            if isinstance(record, Mapping):
                frame_record = record.get("frame", record)
                if isinstance(frame_record, Mapping):
                    try:
                        prior_ref = _as_ref(frame_record)
                    except (TypeError, ValueError):
                        prior_ref = None
                    if prior_ref is not None:
                        prior_records[prior_ref.key] = record
                    # Checkpoints written before the path/frame/dataset key
                    # change remain readable when their legacy key is unique.
                    legacy_key = frame_record.get("frame_id") or frame_record.get("path")
                    if legacy_key is not None:
                        prior_records[str(legacy_key)] = record

    previous_result: Any = None
    previous_frame_key: str | None = None
    for frame in refs:
        restored = prior_records.get(frame.key)
        # Successful frames are safe to restore.  Failed frames are retried so
        # a transient detector/read error does not become permanent.
        if restored is not None and restored.get("status") == "ok":
            # Do not trust a legacy checkpoint that called an explicitly
            # unsuccessful result "ok".  Falling through retries the frame;
            # importantly, the invalid payload never becomes a warm seed.
            restored_quality_error = _quality_failure_reason(restored.get("result"))
            if restored_quality_error is None:
                item = FrameFitResult.from_record(restored)
                item.frame = frame
                item.resumed = True
                run.frame_results.append(item)
                if mode == "warm_start":
                    previous_result = _warm_start_seed(item.result)
                    previous_frame_key = frame.key
                continue

        initial = previous_result if mode == "warm_start" else None
        lineage = previous_frame_key if mode == "warm_start" and previous_frame_key else None
        started = __import__("time").perf_counter()
        try:
            result = _call_analyzer(
                analyze_frame,
                frame,
                initial,
                warm_start=mode == "warm_start",
                config=config,
            )
        except Exception as exc:
            item = FrameFitResult(
                frame=frame,
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
                traceback=traceback_module.format_exc(),
                warm_start_from=lineage,
                elapsed_s=__import__("time").perf_counter() - started,
            )
            # Crucially, previous_result is unchanged.  A failed frame must
            # never seed the next warm-start fit.
        else:
            quality_error = _quality_failure_reason(result)
            item = FrameFitResult(
                frame=frame,
                result=result,
                status="failed" if quality_error is not None else "ok",
                error=quality_error,
                warm_start_from=lineage,
                elapsed_s=__import__("time").perf_counter() - started,
            )
            if mode == "warm_start" and quality_error is None:
                previous_result = _warm_start_seed(result)
                previous_frame_key = frame.key
        run.frame_results.append(item)
        if checkpoint_file is not None:
            write_checkpoint(
                checkpoint_file,
                _checkpoint_payload(run, refs, config=config),
            )

    if checkpoint_file is not None:
        write_checkpoint(checkpoint_file, _checkpoint_payload(run, refs, config=config))
    return run


analyze_batch = run_batch
batch_analyze = run_batch


class BatchAnalyzer:
    """Reusable object wrapper for GUI/CLI dependency injection."""

    def __init__(
        self,
        analyze_frame: Callable[..., Any],
        *,
        mode: Literal["independent", "warm_start"] = "independent",
        config: Any = None,
    ) -> None:
        self.analyze_frame = analyze_frame
        self.mode = mode
        self.config = config

    def run(self, frames: Iterable[Any] | Any, **kwargs: Any) -> BatchRunResult:
        kwargs.setdefault("mode", self.mode)
        kwargs.setdefault("config", self.config)
        return run_batch(frames, self.analyze_frame, **kwargs)


BatchRunner = BatchAnalyzer


__all__ = [
    "BatchAnalyzer",
    "BatchRunResult",
    "BatchRunner",
    "FrameFitResult",
    "FrameRef",
    "analyze_batch",
    "batch_analyze",
    "build_frame_refs",
    "config_fingerprint",
    "discover_frames",
    "input_fingerprint",
    "make_frame_refs",
    "natural_sort_key",
    "read_checkpoint",
    "resolve_frame_refs",
    "run_batch",
    "write_checkpoint",
]
