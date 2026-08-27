"""Read-only package preflight for two-dimensional SAXS data.

The preflight layer deliberately stops before ridge extraction or any fitting.
It resolves the same frame and selector contracts used by :mod:`batch` and
:mod:`io`, then asks :func:`validation.build_analysis_domain` for the pixel
population that a later analysis would be allowed to use.
"""

from __future__ import annotations

import csv
import glob as glob_module
import hashlib
from importlib import metadata as importlib_metadata
import json
import math
import os
import platform
import subprocess
import tempfile
import uuid
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path, PureWindowsPath
from typing import Any, Callable

import numpy as np

from .batch import FrameRef, build_frame_refs, natural_sort_key
from .io import load_image
from .validation import (
    RESULT_SCHEMA_VERSION,
    AnalysisDomainError,
    build_analysis_domain,
    normalise_q_arrays,
    validate_result_schema,
)


PREFLIGHT_SCHEMA_VERSION = "lamellarsaxs2d.preflight.v1"
_IMAGE_SUFFIXES = {
    ".cbf",
    ".edf",
    ".tif",
    ".tiff",
    ".npy",
    ".npz",
    ".h5",
    ".hdf5",
    ".hdf",
    ".csv",
    ".txt",
}
_MANIFEST_KEYS = ("frames", "frame_manifest", "manifest", "data", "items")
_PATH_KEYS = ("path", "input_path", "file", "filename")


class PreflightError(ValueError):
    """Raised when a package cannot satisfy the preflight contract."""


def _json_safe(value: Any) -> Any:
    """Convert values to strict-JSON-compatible Python objects.

    Scientific arrays are never placed in the report except for small inline
    context values.  Non-finite numbers are represented as ``None`` so the
    final ``allow_nan=False`` dump cannot silently produce non-standard JSON.
    """

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if hasattr(value, "tolist"):
        try:
            return _json_safe(value.tolist())
        except (AttributeError, TypeError, ValueError):
            pass
    return str(value)


def _strict_json(value: Any) -> str:
    return json.dumps(
        _json_safe(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise PreflightError(f"cannot read input for SHA-256: {path}: {exc}") from exc
    return digest.hexdigest()


def _hashable_inline(value: Any) -> Any:
    """Describe inline arrays without expanding detector-sized data to JSON."""

    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return {
            "__ndarray__": True,
            "dtype": str(array.dtype),
            "shape": [int(item) for item in array.shape],
            "sha256": _sha256_bytes(array.tobytes()),
        }
    if isinstance(value, Mapping):
        return {str(key): _hashable_inline(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_hashable_inline(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _hashable_inline(value.item())
    return _json_safe(value)


def _inline_hash(value: Any) -> str:
    return _sha256_bytes(_strict_json(_hashable_inline(value)).encode("utf-8"))


def _source_tree_sha256() -> str:
    digest = hashlib.sha256()
    source_root = Path(__file__).parent
    for path in sorted(source_root.glob("*.py"), key=lambda item: item.name.casefold()):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _dependency_versions() -> dict[str, str | None]:
    names = (
        "numpy",
        "scipy",
        "matplotlib",
        "h5py",
        "fabio",
        "pyFAI",
        "PySide6",
        "pyqtgraph",
    )
    versions: dict[str, str | None] = {"python": platform.python_version()}
    for name in names:
        try:
            versions[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (
        FileNotFoundError,
        OSError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ):
        return None
    value = completed.stdout.strip()
    return value if len(value) == 40 else None


def _display_path(path: Path, package: Path) -> str:
    try:
        return path.relative_to(package).as_posix()
    except ValueError:
        return path.as_posix()


def _resolve_path(package: Path, value: str | os.PathLike[str] | Path) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = package / candidate
    return candidate.resolve(strict=False)


def _resolve_output_path(value: str | os.PathLike[str] | Path) -> Path:
    """Resolve output relative to the caller's current working directory."""

    return Path(value).expanduser().resolve(strict=False)


def _read_file_record(
    path: Path,
    package: Path,
    reader: Callable[[], Any],
    records: list[dict[str, Any]],
) -> Any:
    """Run a reader bracketed by SHA-256 checks and record both digests."""

    before = _sha256_file(path)
    value = reader()
    after = _sha256_file(path)
    record = {
        "path": _display_path(path, package),
        "algorithm": "sha256",
        "sha256_before": before,
        "sha256_after": after,
        # Short aliases make the read-before/read-after contract convenient
        # for callers while retaining the explicit names above.
        "before": before,
        "after": after,
        "unchanged": before == after,
    }
    records.append(record)
    if before != after:
        raise PreflightError(f"input changed while being read: {path}")
    return value


def _inline_record(
    value: Any,
    label: str,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    digest = _inline_hash(value)
    record = {
        "path": label,
        "algorithm": "sha256",
        "sha256_before": digest,
        "sha256_after": digest,
        "before": digest,
        "after": digest,
        "unchanged": True,
        "source": "in-memory",
    }
    records.append(record)
    return record


def _require_package(package: str | os.PathLike[str] | Path) -> Path:
    root = Path(package).expanduser().resolve(strict=False)
    if not root.exists() or not root.is_dir():
        raise PreflightError(f"package must be an existing directory: {root}")
    return root


def _load_yaml_context(
    path: Path,
    package: Path,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - exercised without optional dep
        raise PreflightError(
            "YAML context requires PyYAML; install PyYAML before passing a YAML context"
        ) from exc

    def read() -> str:
        try:
            return path.read_text(encoding="utf-8-sig")
        except OSError as exc:
            raise PreflightError(f"cannot read context file {path}: {exc}") from exc

    text = _read_file_record(path, package, read, records)
    try:
        value = yaml.safe_load(text)
    except Exception as exc:  # yaml exposes several parser exception classes
        raise PreflightError(f"could not parse YAML context {path}: {exc}") from exc
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise PreflightError("context YAML must contain a mapping at its top level")
    return dict(value)


def _load_context(
    package: Path,
    context: Any,
    records: list[dict[str, Any]],
) -> tuple[dict[str, Any], Path | None]:
    source: Path | None = None
    if context is None:
        for name in ("project_context.yaml", "project_context.yml"):
            candidate = package / name
            if candidate.is_file():
                source = candidate.resolve()
                break
        if source is None:
            return {}, None
        return _load_yaml_context(source, package, records), source
    if isinstance(context, Mapping):
        value = dict(context)
        _inline_record(value, "context:in-memory", records)
        return value, None
    if not isinstance(context, (str, os.PathLike, Path)):
        raise PreflightError("context must be a mapping, YAML path, or None")
    source = _resolve_path(package, context)
    if not source.exists() or not source.is_file():
        raise PreflightError(f"context file does not exist: {source}")
    if source.suffix.casefold() not in {".yaml", ".yml"}:
        raise PreflightError("context path must use .yaml or .yml")
    return _load_yaml_context(source, package, records), source


def _parse_manifest_file(path: Path) -> Any:
    suffix = path.suffix.casefold()
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise PreflightError(f"cannot read manifest {path}: {exc}") from exc
    try:
        if suffix == ".csv":
            text = raw.decode("utf-8-sig")
            return list(csv.DictReader(text.splitlines()))
        if suffix == ".json":
            return json.loads(raw.decode("utf-8-sig"))
        if suffix == ".toml":
            import tomllib

            return tomllib.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise PreflightError(f"could not parse manifest {path}: {exc}") from exc
    except ImportError as exc:  # pragma: no cover - tomllib is stdlib on supported Python
        raise PreflightError("TOML manifest requires Python 3.11 or newer") from exc
    raise PreflightError("manifest must be CSV, JSON, or TOML")


def _manifest_rows(value: Any) -> list[Any]:
    """Flatten supported manifest envelopes without changing row order."""

    if isinstance(value, Mapping):
        for key in _MANIFEST_KEYS:
            candidate = value.get(key)
            if isinstance(candidate, Sequence) and not isinstance(candidate, (str, bytes)):
                return list(candidate)
        if any(key in value for key in _PATH_KEYS):
            return [value]
        rows: list[dict[str, Any]] = []
        for path, metadata in value.items():
            row = dict(metadata) if isinstance(metadata, Mapping) else {"time": metadata}
            row.setdefault("path", path)
            rows.append(row)
        return rows
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return list(value)
    raise PreflightError("manifest must contain a sequence of frame rows")


def _normalise_manifest_row(row: Any, package: Path) -> Any:
    if isinstance(row, FrameRef):
        return FrameRef(
            _resolve_path(package, row.path),
            time=row.time,
            frame_id=row.frame_id,
            metadata=row.metadata,
            order=row.order,
            source=row.source,
            dataset=row.dataset,
            frame=row.frame,
        )
    if isinstance(row, Mapping):
        result = dict(row)
        # CSV readers retain empty selector cells as ``""``.  Empty values
        # mean "not selected"; forwarding them to ``FrameRef`` would turn a
        # perfectly valid row into an integer-conversion error.
        for key in (
            "frame",
            "frame_index",
            "dataset",
            "dataset_id",
            "dataset_name",
            "order",
            "time",
            "timestamp",
        ):
            if key in result and isinstance(result[key], str) and not result[key].strip():
                result[key] = None
        for key in _PATH_KEYS:
            candidate = result.get(key)
            if candidate is None:
                continue
            if isinstance(candidate, (str, os.PathLike, Path)):
                result[key] = str(_resolve_path(package, candidate))
            break
        return result
    if isinstance(row, (str, os.PathLike, Path)):
        return {"path": str(_resolve_path(package, row))}
    raise PreflightError(f"manifest row is not a path or mapping: {row!r}")


def _resolve_manifest(
    package: Path,
    manifest: Any,
    records: list[dict[str, Any]],
) -> tuple[list[FrameRef] | None, Path | None, dict[str, Any] | None]:
    if manifest is None:
        return None, None, None
    source: Path | None = None
    if isinstance(manifest, (str, os.PathLike, Path)):
        source = _resolve_path(package, manifest)
        if not source.exists() or not source.is_file():
            raise PreflightError(f"manifest file does not exist: {source}")
        parsed = _read_file_record(source, package, lambda: _parse_manifest_file(source), records)
    else:
        parsed = manifest
        _inline_record(parsed, "manifest:in-memory", records)
    rows = [_normalise_manifest_row(row, package) for row in _manifest_rows(parsed)]
    try:
        refs = build_frame_refs([], manifest=rows)
    except (TypeError, ValueError) as exc:
        raise PreflightError(f"could not resolve manifest frame references: {exc}") from exc
    order_values: list[float] = []
    order_indices: list[int] = []
    paths: list[str] = []
    selector_identities: list[tuple[str, str | None, int | None]] = []
    for index, row in enumerate(rows):
        if isinstance(row, Mapping):
            order = row.get("order")
        else:
            order = None
        if order is not None and not (isinstance(order, str) and not order.strip()):
            try:
                numeric_order = float(order)
            except (TypeError, ValueError, OverflowError):
                numeric_order = float("nan")
            if math.isfinite(numeric_order):
                order_values.append(numeric_order)
                order_indices.append(index)
    time_values: list[float | None] = []
    invalid_time_indices: list[int] = []
    missing_time_indices: list[int] = []
    # All sequence-sensitive evidence follows the same authoritative order as
    # the returned refs: explicit numeric order, otherwise numeric time, then
    # original manifest position.
    for index, ref in enumerate(refs):
        normalized_path = os.path.normcase(ref.path.resolve(strict=False).as_posix())
        paths.append(normalized_path)
        selector_identities.append((normalized_path, ref.dataset, ref.frame))
        time = ref.time
        if time is None or (isinstance(time, str) and not time.strip()):
            time_values.append(None)
            missing_time_indices.append(index)
        else:
            try:
                numeric_time = float(time)
            except (TypeError, ValueError, OverflowError):
                numeric_time = float("nan")
            if math.isfinite(numeric_time):
                time_values.append(numeric_time)
            else:
                time_values.append(None)
                invalid_time_indices.append(index)
    duplicate_orders = sorted(
        value
        for value in set(order_values)
        if order_values.count(value) > 1
    )
    duplicate_paths = sorted(
        value
        for value in set(paths)
        if paths.count(value) > 1
    )
    duplicate_selectors = sorted(
        (
            identity
            for identity in set(selector_identities)
            if selector_identities.count(identity) > 1
        ),
        key=lambda identity: (
            identity[0],
            identity[1] or "",
            -1 if identity[2] is None else identity[2],
        ),
    )
    finite_times = [value for value in time_values if value is not None]
    time_monotonic = all(left <= right for left, right in zip(finite_times, finite_times[1:]))
    authoritative_order_values = [
        float(ref.order) for ref in refs if ref.order is not None
    ] if order_indices else []
    quality = {
        "explicit_order": bool(order_indices),
        "order_values": authoritative_order_values,
        "input_order_values": order_values,
        "order_unique": len(order_values) == len(set(order_values)),
        "duplicate_orders": duplicate_orders,
        "paths": paths,
        "path_unique": len(paths) == len(set(paths)),
        "duplicate_paths": duplicate_paths,
        "selector_unique": len(selector_identities) == len(set(selector_identities)),
        "duplicate_selectors": [
            {"path": path, "dataset": dataset, "frame": frame}
            for path, dataset, frame in duplicate_selectors
        ],
        "time_values": time_values,
        "time_finite": not invalid_time_indices,
        "invalid_time_indices": invalid_time_indices,
        "missing_time_indices": missing_time_indices,
        "time_monotonic": time_monotonic,
    }
    return refs, source, quality


def _nested_mapping(context: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = context.get(key)
    return value if isinstance(value, Mapping) else {}


def _first_value(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _context_settings(context: Mapping[str, Any]) -> dict[str, Any]:
    preferred = _nested_mapping(context, "preferred_project_input")
    return {
        "manifest": _first_value(
            context.get("manifest"), context.get("manifest_path"), preferred.get("manifest")
        ),
        "image_glob": _first_value(
            preferred.get("image_glob"),
            preferred.get("preferred_image_glob"),
            preferred.get("preferred_edf_glob"),
            context.get("image_glob"),
            context.get("preferred_edf_glob"),
        ),
        "poni": _first_value(
            preferred.get("poni"),
            preferred.get("poni_path"),
            preferred.get("qmap"),
            context.get("poni"),
            context.get("poni_path"),
            context.get("qmap"),
        ),
        "mask": _first_value(
            preferred.get("mask"), preferred.get("mask_path"), context.get("mask"), context.get("mask_path")
        ),
        "mask_convention": _first_value(
            preferred.get("mask_convention"), context.get("mask_convention")
        ),
        "q_window": _first_value(
            preferred.get("q_window_nm_inv"),
            preferred.get("q_window"),
            preferred.get("q_range"),
            context.get("q_window_nm_inv"),
            context.get("q_window"),
        ),
        "q_unit": _first_value(preferred.get("q_unit"), context.get("q_unit")),
    }


def _glob_inputs(package: Path, pattern: Any) -> list[Path]:
    if pattern is None:
        patterns = [f"**/*{suffix}" for suffix in sorted(_IMAGE_SUFFIXES)]
        matches: list[Path] = []
        for candidate in patterns:
            matches.extend(Path(item).resolve() for item in glob_module.glob(str(package / candidate), recursive=True))
    else:
        text = os.fspath(pattern)
        glob_pattern = text if Path(text).expanduser().is_absolute() else str(package / text)
        matches = [Path(item).resolve() for item in glob_module.glob(glob_pattern, recursive=True)]
    return sorted(
        {path for path in matches if path.is_file() and path.suffix.casefold() in _IMAGE_SUFFIXES},
        key=lambda path: natural_sort_key(path.name),
    )


def _qmap_value(value: Any, names: Sequence[str], default: Any = None) -> Any:
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


def _qmap_from_value(value: Any, shape: tuple[int, int]) -> dict[str, Any]:
    qx = _qmap_value(value, ("qx", "qx_nm_inv"))
    qy = _qmap_value(value, ("qy", "qy_nm_inv"))
    if qx is None or qy is None:
        raise PreflightError("explicit qmap must provide qx and qy arrays")
    qx_array = np.asarray(qx, dtype=float)
    qy_array = np.asarray(qy, dtype=float)
    if qx_array.shape != shape or qy_array.shape != shape:
        raise PreflightError(
            f"qmap shape must match image shape {shape!r}; got qx={qx_array.shape!r}, qy={qy_array.shape!r}"
        )
    q_value = _qmap_value(value, ("q", "q_nm_inv"))
    q_array = np.hypot(qx_array, qy_array) if q_value is None else np.asarray(q_value, dtype=float)
    if q_array.shape != shape:
        raise PreflightError(f"qmap q shape {q_array.shape!r} does not match image shape {shape!r}")
    valid = _qmap_value(value, ("valid_mask", "valid"))
    if valid is None and isinstance(value, Mapping) and value.get("mask") is not None:
        detector_valid = ~np.asarray(value["mask"], dtype=bool)
    else:
        detector_valid = np.ones(shape, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    if detector_valid.shape != shape:
        raise PreflightError(
            f"qmap valid_mask shape {detector_valid.shape!r} does not match image shape {shape!r}"
        )
    metadata = _qmap_value(value, ("metadata",), {})
    if not isinstance(metadata, Mapping):
        metadata = {}
    q_unit = _qmap_value(value, ("q_unit", "unit"))
    if q_unit is None:
        q_unit = metadata.get("q_unit", metadata.get("unit"))
    if q_unit is None and any(
        _qmap_value(value, (name,)) is not None
        for name in ("qx_nm_inv", "qy_nm_inv", "q_nm_inv")
    ):
        q_unit = "nm^-1"
    qx_array, qy_array, q_array, unit_info = normalise_q_arrays(
        qx_array,
        qy_array,
        q_array,
        q_unit,
    )
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


def _build_qmap(
    shape: tuple[int, int],
    poni: Any,
    package: Path,
    records: list[dict[str, Any]],
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
        and _qmap_value(poni, ("qx", "qx_nm_inv")) is not None
    ):
        _inline_record(poni, "poni:qmap-in-memory", records)
        return _qmap_from_value(poni, shape)
    if not isinstance(poni, (str, os.PathLike, Path)) and all(
        callable(getattr(poni, name, None)) for name in ("qArray", "center_array")
    ):
        _inline_record(poni, "poni:integrator-in-memory", records)
        try:
            from .geometry import build_geometry

            geometry = build_geometry(shape, poni)
        except (ImportError, OSError, ValueError, RuntimeError) as exc:
            raise PreflightError(f"could not build qmap from in-memory PONI: {exc}") from exc
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
    source = _resolve_path(package, poni)
    if not source.exists() or not source.is_file():
        raise PreflightError(f"PONI file does not exist: {source}")
    try:
        from .geometry import build_geometry

        geometry = _read_file_record(
            source,
            package,
            lambda: build_geometry(shape, source),
            records,
        )
    except (ImportError, OSError, ValueError, RuntimeError) as exc:
        if isinstance(exc, PreflightError):
            raise
        raise PreflightError(f"could not build qmap from PONI {source}: {exc}") from exc
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


def _image_summary(array: np.ndarray) -> dict[str, Any]:
    if array.dtype.kind not in "biufc":
        raise PreflightError(f"image dtype {array.dtype!s} is not numeric")
    finite = np.isfinite(array)
    finite_count = int(np.count_nonzero(finite))
    image_count = int(array.size)
    result: dict[str, Any] = {
        "shape": [int(item) for item in array.shape],
        "dtype": str(array.dtype),
        "pixel_count": image_count,
        "finite_count": finite_count,
        "finite_fraction": finite_count / image_count if image_count else 0.0,
        "negative_count": 0,
        "negative_fraction": 0.0,
        "robust_high_count": 0,
        "robust_high_fraction": 0.0,
        "robust_high": {
            "method": "median_plus_6_mad",
            "median": None,
            "mad": None,
            "threshold": None,
            "count": 0,
            "fraction": 0.0,
        },
    }
    if not finite_count:
        return result
    values = np.asarray(array[finite], dtype=float)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    threshold = median + 6.0 * 1.4826 * mad
    if not math.isfinite(threshold):
        threshold = float(np.max(values))
    negative_count = int(np.count_nonzero(values < 0))
    high_count = int(np.count_nonzero(values > threshold))
    result.update(
        {
            "negative_count": negative_count,
            "negative_fraction": negative_count / finite_count,
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "median": median,
            "p95": float(np.percentile(values, 95)),
            "p99": float(np.percentile(values, 99)),
            "robust_high_count": high_count,
            "robust_high_fraction": high_count / finite_count,
            "robust_high": {
                "method": "median_plus_6_mad",
                "median": median,
                "mad": mad,
                "threshold": threshold,
                "count": high_count,
                "fraction": high_count / finite_count,
            },
        }
    )
    return result


def _load_mask(
    mask: Any,
    package: Path,
    shape: tuple[int, int],
    mask_frame: int | None,
    mask_dataset: str | None,
    records: list[dict[str, Any]],
) -> tuple[np.ndarray | None, Path | None]:
    if mask is None:
        return None, None
    source: Path | None = None
    if isinstance(mask, (str, os.PathLike, Path)):
        source = _resolve_path(package, mask)
        if not source.exists() or not source.is_file():
            raise PreflightError(f"mask file does not exist: {source}")
        try:
            loaded = _read_file_record(
                source,
                package,
                lambda: load_image(source, frame=mask_frame, dataset=mask_dataset).data,
                records,
            )
        except (OSError, TypeError, ValueError, RuntimeError) as exc:
            if isinstance(exc, PreflightError):
                raise
            raise PreflightError(f"could not read mask {source}: {exc}") from exc
        array = np.asarray(loaded)
    else:
        array = np.asarray(mask)
        _inline_record(array, "mask:in-memory", records)
        if array.ndim > 2:
            if mask_frame is None:
                raise PreflightError(
                    "in-memory mask contains multiple frames; select mask_frame explicitly"
                )
            if array.ndim != 3 or mask_frame >= array.shape[0]:
                raise PreflightError(
                    f"mask_frame {mask_frame} is outside in-memory mask with {array.shape[0]} frames"
                )
            array = array[int(mask_frame)]
    if array.shape != shape:
        raise PreflightError(f"mask shape {array.shape!r} does not match image shape {shape!r}")
    if array.dtype.kind == "O":
        raise PreflightError("mask cannot be object-valued")
    return np.asarray(array != 0, dtype=bool), source


def _normalise_mask_convention(value: Any) -> str:
    convention = "0_valid_1_invalid" if value is None else str(value)
    if convention not in {"0_valid_1_invalid", "1_valid_0_invalid"}:
        raise PreflightError(
            "mask_convention must be '0_valid_1_invalid' or '1_valid_0_invalid'"
        )
    return convention


def _state_value(explicit: Any, context: Mapping[str, Any], key: str) -> Any:
    if explicit is not None:
        return explicit
    if key in context:
        return context[key]
    source = context.get("source_intensity")
    if isinstance(source, Mapping):
        if key == "uncertainty_state":
            return _first_value(source.get("uncertainty_state"), source.get("uncertainty_status"))
        if key == "correction_state":
            return source
    return None


def _resolve_correction_value(explicit: Any, context: Mapping[str, Any]) -> tuple[Any, Any]:
    """Resolve correction status separately from uncertainty metadata."""

    if explicit is not None:
        return explicit, explicit
    if "correction_state" in context:
        value = context["correction_state"]
        return value, value
    source = context.get("source_intensity")
    if not isinstance(source, Mapping):
        return None, None
    declared = _first_value(source.get("correction_state"), source.get("correction_status"))
    if declared is not None:
        return declared, source
    if "already_applied" in source or "not_burned_into_2d_values" in source:
        details = dict(source)
        details["status"] = "external_recipe_declared"
        return "external_recipe_declared", details
    # ``uncertainty_status`` belongs to the independent uncertainty contract;
    # it must never be interpreted as a correction state.
    return None, None


def _state_status(value: Any, kind: str) -> tuple[str, str, dict[str, Any]]:
    safe = _json_safe(value)
    if value is None:
        return "warn", f"{kind} is not declared", {"value": None}
    text = str(value).casefold() if not isinstance(value, Mapping) else ""
    if text in {"partial", "unknown", "provisional", "incomplete", "none"}:
        return "warn", f"{kind} is {text}", {"value": safe}
    if isinstance(value, Mapping):
        status_names = ("status", "state", "uncertainty_status") if kind == "uncertainty_state" else (
            "status",
            "state",
            "correction_status",
            "correction_state",
        )
        status = str(_first_value(*(value.get(name) for name in status_names), "unknown")).casefold()
        if status in {"partial", "unknown", "provisional", "incomplete", "none"}:
            return "warn", f"{kind} is {status}", {"value": safe}
        if status in {
            "complete",
            "completed",
            "fully_corrected",
            "fully_corrected_external",
            "external_recipe_declared",
            "pass",
            "ok",
        }:
            return "pass", f"{kind} is {status}", {"value": safe}
        return "warn", f"{kind} status is not confirmed", {"value": safe}
    return "pass", f"{kind} is {text or 'declared'}", {"value": safe}


def _correction_check(value: Any) -> tuple[str, str, dict[str, Any]]:
    status, reason, evidence = _state_status(value, "correction_state")
    not_burned: list[str] = []
    if isinstance(value, Mapping):
        raw = _first_value(
            value.get("not_burned_into_2d_values"),
            value.get("not_applied"),
            value.get("not_applied_corrections"),
        )
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            not_burned = [str(item).casefold().replace("-", "_").replace(" ", "_") for item in raw]
        for key in ("solid_angle", "solid_angle_applied", "polarization", "polarisation", "polarization_applied"):
            if key in value and value[key] is False:
                not_burned.append(key)
        if "status" not in value and "state" not in value and "correction_status" not in value and (
            "already_applied" in value or "not_burned_into_2d_values" in value
        ):
            status = "pass"
            reason = "correction_state is external_recipe_declared"
    required = {"solid_angle", "solid_angle_correction", "polarization", "polarisation"}
    omitted = sorted(item for item in not_burned if item in required or any(token in item for token in required))
    if omitted:
        status = "warn"
        reason = "solid-angle/polarization correction is declared not applied"
        evidence = {**evidence, "not_burned_into_2d_values": omitted}
    return status, reason, evidence


def _uncertainty_check(value: Any) -> tuple[str, str, dict[str, Any]]:
    status, reason, evidence = _state_status(value, "uncertainty_state")
    if isinstance(value, Mapping):
        raw = _first_value(value.get("status"), value.get("state"), value.get("uncertainty_status"))
        if raw is not None:
            status, reason, evidence = _state_status(str(raw), "uncertainty_state")
    return status, reason, evidence


def _uncertainty_provenance(
    image_metadata: Sequence[Mapping[str, Any]],
    package: Path,
    hash_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize declared per-frame uncertainty files and reference datasets."""

    resolved: list[Path] = []
    declared_basenames: list[str] = []
    missing_basenames: list[str] = []
    for metadata in image_metadata:
        header = metadata.get("header")
        if not isinstance(header, Mapping):
            continue
        normalized = {str(key).casefold(): value for key, value in header.items()}
        raw_source = normalized.get("uncertaintyhdf5")
        if raw_source is None:
            continue
        declared = str(raw_source)
        windows_path = PureWindowsPath(declared)
        basename = windows_path.name
        if not basename:
            continue
        declared_basenames.append(basename)
        direct = Path(declared)
        candidates = [
            direct,
            package / windows_path.parent.name / basename,
            package / basename,
        ]
        local = next((path for path in candidates if path.is_file()), None)
        if local is None:
            missing_basenames.append(basename)
        else:
            resolved.append(local.resolve())

    unique_resolved = list(dict.fromkeys(resolved))
    unique_declared = list(dict.fromkeys(declared_basenames))
    unique_missing = list(dict.fromkeys(missing_basenames))
    datasets: list[dict[str, Any]] = []
    inventory_status = "not_declared"
    reference_file = unique_resolved[0] if unique_resolved else None
    if reference_file is not None:
        try:
            import h5py
        except ImportError:
            inventory_status = "h5py_unavailable"
        else:
            try:
                def read_reference_schema() -> tuple[str, list[dict[str, Any]]]:
                    records: list[dict[str, Any]] = []
                    with h5py.File(reference_file, "r") as handle:
                        group = handle.get("entry/data/uncertainty")
                        if group is None:
                            return "uncertainty_group_missing", records
                        for name, dataset in group.items():
                            if not hasattr(dataset, "shape"):
                                continue
                            records.append(
                                {
                                    "dataset": f"entry/data/uncertainty/{name}",
                                    "component": str(name),
                                    "shape": list(dataset.shape),
                                    "unit": _json_safe(dataset.attrs.get("units")),
                                }
                            )
                    return "reference_schema_read", records

                inventory_status, datasets = _read_file_record(
                    reference_file,
                    package,
                    read_reference_schema,
                    hash_records,
                )
            except OSError:
                inventory_status = "reference_hdf5_unreadable"

    return {
        "declared_source_kind": "per_frame_hdf5" if unique_declared else None,
        "declared_file_count": len(unique_declared),
        "resolved_file_count": len(unique_resolved),
        "files": [_display_path(path, package) for path in unique_resolved],
        "missing_declared_files": unique_missing,
        "reference_schema_file": (
            _display_path(reference_file, package) if reference_file is not None else None
        ),
        "dataset_inventory_status": inventory_status,
        "datasets": datasets,
        "covered_components": [item["component"] for item in datasets],
        "units": sorted(
            {
                str(item["unit"])
                for item in datasets
                if item.get("unit") is not None
            }
        ),
    }


def _header_geometry_fields(metadata: Mapping[str, Any]) -> dict[str, float]:
    header = metadata.get("header")
    if not isinstance(header, Mapping):
        return {}
    aliases = {
        "dist": "dist",
        "distance": "dist",
        "poni1": "poni1",
        "poni2": "poni2",
        "rot1": "rot1",
        "rot2": "rot2",
        "rot3": "rot3",
        "wavelength": "wavelength",
    }
    result: dict[str, float] = {}
    for key, value in header.items():
        normal = aliases.get(str(key).casefold().replace(" ", ""))
        if normal is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric):
            result[normal] = numeric
    return result


def _header_geometry_check(
    image_metadata: Sequence[Mapping[str, Any]],
    qmap_metadata: Mapping[str, Any],
    has_poni: bool,
) -> tuple[str, str, dict[str, Any]]:
    if not has_poni:
        return "pass", "header geometry comparison is not applicable to pixel-q", {"applicable": False}
    poni_fields: dict[str, float] = {}
    for key in ("dist", "poni1", "poni2", "rot1", "rot2", "rot3", "wavelength"):
        try:
            number = float(qmap_metadata[key])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(number):
            poni_fields[key] = number
    headers = [_header_geometry_fields(metadata) for metadata in image_metadata]
    comparable = sorted(set(poni_fields).intersection(*(set(item) for item in headers))) if headers else []
    differences: dict[str, Any] = {}
    for key in comparable:
        values = [item[key] for item in headers]
        if not all(math.isclose(item, poni_fields[key], rel_tol=1e-6, abs_tol=1e-9) for item in values):
            differences[key] = {"poni": poni_fields[key], "headers": values}
    if not comparable:
        return (
            "warn",
            "header geometry fields are unavailable for comparison with PONI",
            {"poni_fields": sorted(poni_fields), "comparable_fields": [], "differences": {}},
        )
    if differences:
        return "warn", "header geometry differs from PONI fields", {
            "poni_fields": sorted(poni_fields),
            "comparable_fields": comparable,
            "differences": differences,
        }
    return "pass", "available header geometry agrees with PONI", {
        "poni_fields": sorted(poni_fields),
        "comparable_fields": comparable,
        "differences": {},
    }


def _add_check(
    checks: list[dict[str, Any]],
    warnings: list[str],
    errors: list[str],
    check_id: str,
    status: str,
    reason: str,
    evidence: Any,
) -> None:
    check = {
        "id": check_id,
        "status": status,
        "reason": str(reason),
        "evidence": _json_safe(evidence),
    }
    checks.append(check)
    if status == "warn":
        warnings.append(str(reason))
    elif status == "fail":
        errors.append(str(reason))


def _status_from_checks(checks: Sequence[Mapping[str, Any]]) -> tuple[str, str]:
    values = {str(item.get("status", "pass")) for item in checks}
    if "fail" in values:
        return "red", "FAIL"
    if "warn" in values:
        return "yellow", "WARN"
    return "green", "PASS"


def _prepare_outputs(output: Any, force: bool) -> tuple[dict[str, Path] | None, list[Path]]:
    if output is None:
        return None, []
    if not isinstance(output, (str, os.PathLike, Path)):
        raise PreflightError("output must be a directory path or None")
    target = _resolve_output_path(output)
    if target.exists() and not target.is_dir():
        raise PreflightError(f"output must be a directory: {target}")
    paths = {
        "preflight_json": target / "preflight.json",
        "arrays_npz": target / "arrays.npz",
        "run_report": target / "run_report.md",
    }
    existing = [path for path in paths.values() if path.exists()]
    directory_contents = list(target.iterdir()) if target.exists() else []
    if not force and directory_contents:
        names = ", ".join(str(path) for path in directory_contents)
        raise FileExistsError(f"preflight output exists; pass force=True to overwrite: {names}")
    return paths, existing


def _atomic_text(path: Path, text: str) -> None:
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(text, encoding="utf-8", newline="\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_npz(path: Path, arrays: Mapping[str, Any]) -> None:
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=path.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _markdown_report(report: Mapping[str, Any]) -> str:
    status = report.get("status", {})
    quality = report.get("quality", {})
    checks = quality.get("checks", []) if isinstance(quality, Mapping) else []
    extensions = report.get("extensions", {})
    preflight_extension = (
        extensions.get("preflight", {}) if isinstance(extensions, Mapping) else {}
    )
    frames = preflight_extension.get("frames", [])
    lines = [
        "# SAXS package preflight",
        "",
        f"- Status: `{status.get('status_color')}` / scientific status `{status.get('scientific_status')}`",
        f"- Exit code: `{status.get('exit_code')}`",
        f"- Frames: `{len(frames)}`",
        f"- q unit: `{report.get('geometry', {}).get('q_unit', 'unknown')}`",
        f"- Fit-valid pixels (reference frame): `{report.get('analysis_domain', {}).get('counts', {}).get('fit_pixel_count', 'n/a')}`",
        f"- Solver status: `{status.get('solver_status', 'not_run')}` (preflight does not fit data)",
        "",
        "## Checks",
        "",
        "| id | status | reason |",
        "|---|---|---|",
    ]
    for item in checks:
        reason = str(item.get("message", item.get("reason", ""))).replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {item.get('id', '')} | {item.get('status', '')} | {reason} |")
    warnings = preflight_extension.get("warnings", [])
    errors = preflight_extension.get("errors", [])
    if warnings:
        lines.extend(["", "## Warnings", "", *[f"- {item}" for item in warnings]])
    if errors:
        lines.extend(["", "## Errors", "", *[f"- {item}" for item in errors]])
    return "\n".join(lines) + "\n"


def _read_one_frame(
    ref: FrameRef,
    package: Path,
    frame_override: int | None,
    dataset_override: str | None,
    records: list[dict[str, Any]],
) -> tuple[np.ndarray, Any, Path]:
    """Read one frame and immediately release the loader object to the caller."""

    source = _resolve_path(package, ref.path)
    if not source.exists() or not source.is_file():
        raise PreflightError(f"manifest frame path does not exist: {source}")
    selected_frame = frame_override if frame_override is not None else ref.frame_selector
    selected_dataset = dataset_override if dataset_override is not None else (ref.dataset_id or None)
    try:
        loaded = _read_file_record(
            source,
            package,
            lambda source=source, selected_frame=selected_frame, selected_dataset=selected_dataset: load_image(
                source,
                frame=selected_frame,
                dataset=selected_dataset,
            ),
            records,
        )
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        if isinstance(exc, PreflightError):
            raise
        raise PreflightError(f"could not read frame {source}: {exc}") from exc
    array = np.asarray(loaded.data)
    if array.ndim != 2 or array.size == 0:
        raise PreflightError(f"frame {source} must be a non-empty 2-D image; got {array.shape!r}")
    return array, loaded, source


def _frame_record(
    index: int,
    ref: FrameRef,
    source: Path,
    loaded: Any,
    array: np.ndarray,
    package: Path,
    image_metadata: list[Mapping[str, Any]],
) -> dict[str, Any]:
    summary = _image_summary(array)
    metadata = loaded.metadata if isinstance(loaded.metadata, Mapping) else {}
    image_metadata.append(metadata)
    return {
        "index": index,
        "id": ref.id,
        "path": _display_path(source, package),
        "frame": loaded.frame,
        "dataset": loaded.dataset,
        "manifest_frame": ref.to_dict(),
        "summary": summary,
        "shape": summary["shape"],
        "dtype": summary["dtype"],
        "finite_fraction": summary["finite_fraction"],
        "negative_fraction": summary["negative_fraction"],
        "robust_high": summary["robust_high"],
    }


def _canonical_state(value: Any, kind: str) -> str:
    if isinstance(value, Mapping):
        names = (
            ("status", "state", "correction_status", "correction_state")
            if kind == "correction"
            else ("status", "state", "uncertainty_status", "uncertainty_state")
        )
        value = _first_value(*(value.get(name) for name in names))
    token = str(value or "unknown").casefold()
    if kind == "correction":
        aliases = {
            "raw": "raw_counts",
            "raw_counts": "raw_counts",
            "external_recipe_declared": "external_recipe_declared",
            "partial": "partially_corrected",
            "partially_corrected": "partially_corrected",
            "complete": "fully_corrected_external",
            "completed": "fully_corrected_external",
            "fully_corrected": "fully_corrected_external",
            "fully_corrected_external": "fully_corrected_external",
        }
    else:
        aliases = {
            "none": "none",
            "partial": "partial",
            "complete": "complete",
            "completed": "complete",
        }
    return aliases.get(token, "unknown")


def _structured_check(check: Mapping[str, Any]) -> dict[str, Any]:
    status = str(check.get("status", "pass"))
    status_name = {"pass": "PASS", "warn": "WARN", "fail": "FAIL"}[status]
    color = {"pass": "green", "warn": "yellow", "fail": "red"}[status]
    check_id = str(check.get("id", "unknown"))
    return {
        "id": check_id,
        "name": check_id,
        "status": status_name,
        "status_color": color,
        "observed": check.get("evidence"),
        "threshold": "p0-p2-contract-v1",
        "comparison": "contract",
        "evidence": f"extensions.preflight.checks.{check_id}",
        "message": str(check.get("reason", "")),
    }


def _status_item(check: Mapping[str, Any]) -> dict[str, Any]:
    status = str(check.get("status", "warn"))
    return {
        "code": f"preflight_{check.get('id', 'unknown')}",
        "severity": "red" if status == "fail" else "yellow",
        "message": str(check.get("reason", "")),
        "evidence": f"quality.checks.{check.get('id', 'unknown')}",
    }


def run_preflight(
    package: str | os.PathLike[str] | Path,
    *,
    manifest: Any = None,
    poni: Any = None,
    mask: Any = None,
    context: Any = None,
    image_glob: str | os.PathLike[str] | None = None,
    frame: int | None = None,
    dataset: str | None = None,
    mask_frame: int | None = None,
    mask_dataset: str | None = None,
    q_window: Any = None,
    mask_convention: str | None = None,
    correction_state: Any = None,
    uncertainty_state: Any = None,
    output: str | os.PathLike[str] | Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Run a read-only preflight and optionally write its three artifacts.

    Relative input paths (including manifest rows, context paths, PONI, mask,
    and image globs) are resolved below ``package``.  A relative ``output`` is
    interpreted relative to the caller's current working directory, while an
    absolute output remains unchanged.  The function raises
    :class:`PreflightError`/``ValueError`` for blocking input errors and
    returns a strict-JSON-compatible dictionary for green or yellow runs.
    """

    package_root = _require_package(package)
    if isinstance(frame, bool) or (frame is not None and (not isinstance(frame, (int, np.integer)) or int(frame) < 0)):
        raise PreflightError("frame must be a non-negative integer or None")
    if isinstance(mask_frame, bool) or (
        mask_frame is not None and (not isinstance(mask_frame, (int, np.integer)) or int(mask_frame) < 0)
    ):
        raise PreflightError("mask_frame must be a non-negative integer or None")
    if dataset is not None and not isinstance(dataset, str):
        raise PreflightError("dataset must be a string or None")
    if mask_dataset is not None and not isinstance(mask_dataset, str):
        raise PreflightError("mask_dataset must be a string or None")

    hash_groups: dict[str, list[dict[str, Any]]] = {
        "inputs": [],
        "manifest": [],
        "poni": [],
        "mask": [],
        "context": [],
        "uncertainty": [],
    }
    context_value, context_source = _load_context(package_root, context, hash_groups["context"])
    settings = _context_settings(context_value)
    selected_manifest = manifest if manifest is not None else settings["manifest"]
    refs, manifest_source, manifest_quality = _resolve_manifest(
        package_root,
        selected_manifest,
        hash_groups["manifest"],
    )
    selected_glob = image_glob if image_glob is not None else settings["image_glob"]
    selected_poni = poni if poni is not None else settings["poni"]
    selected_mask = mask if mask is not None else settings["mask"]
    selected_q_window = q_window if q_window is not None else settings["q_window"]
    selected_convention = _normalise_mask_convention(
        mask_convention if mask_convention is not None else settings["mask_convention"]
    )

    if refs is None:
        paths = _glob_inputs(package_root, selected_glob)
        exclusions: set[Path] = set()
        for candidate in (selected_mask, selected_poni):
            if isinstance(candidate, (str, os.PathLike, Path)):
                exclusions.add(_resolve_path(package_root, candidate))
        if context_source is not None:
            exclusions.add(context_source)
        output_root = (
            _resolve_output_path(output)
            if isinstance(output, (str, os.PathLike, Path))
            else None
        )
        paths = [
            path
            for path in paths
            if path not in exclusions
            and (output_root is None or not path.is_relative_to(output_root))
        ]
        refs = build_frame_refs(paths)
    if not refs:
        source_hint = "manifest" if selected_manifest is not None else "image_glob/package scan"
        raise PreflightError(f"no input frames found from {source_hint}")

    # Read one reference frame first.  Once its shape is known, resolve the
    # independent mask and geometry, then stream every remaining frame through
    # summary/domain construction without retaining detector-sized arrays.
    image_metadata: list[Mapping[str, Any]] = []
    frames: list[dict[str, Any]] = []
    reference_array, reference_loaded, reference_source = _read_one_frame(
        refs[0],
        package_root,
        frame,
        dataset,
        hash_groups["inputs"],
    )
    first_shape = tuple(int(item) for item in reference_array.shape)
    frames.append(
        _frame_record(
            0,
            refs[0],
            reference_source,
            reference_loaded,
            reference_array,
            package_root,
            image_metadata,
        )
    )

    mask_array, mask_source = _load_mask(
        selected_mask,
        package_root,
        first_shape,
        mask_frame,
        mask_dataset,
        hash_groups["mask"],
    )
    external_mask: np.ndarray | None = None
    valid_mask: np.ndarray | None = None
    if mask_array is not None:
        if selected_convention == "0_valid_1_invalid":
            external_mask = mask_array
        else:
            valid_mask = mask_array
            external_mask = ~valid_mask

    qmap = _build_qmap(first_shape, selected_poni, package_root, hash_groups["poni"])
    if qmap["qx"].shape != first_shape or qmap["qy"].shape != first_shape:
        raise PreflightError("qmap shape does not match input image shape")
    domain_q_window = selected_q_window
    if domain_q_window is None:
        finite_q = np.asarray(qmap["q"])[np.isfinite(qmap["q"])]
        if finite_q.size and float(np.min(finite_q)) == float(np.max(finite_q)):
            # A tiny 1x1/2x2 detector can have one unique radial distance.
            # The validation contract still requires an ordered window, so
            # widen only the implicit full-range default by machine-scale
            # padding; an explicit degenerate q_window remains an error.
            value = float(finite_q[0])
            delta = max(1.0, abs(value)) * 1e-12
            domain_q_window = (value - delta, value + delta)

    def build_domain(index: int, array: np.ndarray) -> Any:
        try:
            return build_analysis_domain(
                array,
                qmap["qx"],
                qmap["qy"],
                q=qmap["q"],
                detector_valid=qmap["detector_valid"],
                external_mask=external_mask,
                q_window=domain_q_window,
            )
        except (AnalysisDomainError, TypeError, ValueError) as exc:
            raise PreflightError(f"could not build analysis domain for frame {index}: {exc}") from exc

    reference_domain = build_domain(0, reference_array)
    domain_records: list[dict[str, Any]] = [
        {
            "index": 0,
            "frame_id": refs[0].id,
            "summary": reference_domain.to_summary(),
        }
    ]
    for index, ref in enumerate(refs[1:], start=1):
        array, loaded, source = _read_one_frame(
            ref,
            package_root,
            frame,
            dataset,
            hash_groups["inputs"],
        )
        if tuple(array.shape) != first_shape:
            raise PreflightError(
                f"all frames must share one shape; frame {index} has {array.shape!r}, expected {first_shape!r}"
            )
        frames.append(
            _frame_record(
                index,
                ref,
                source,
                loaded,
                array,
                package_root,
                image_metadata,
            )
        )
        domain = build_domain(index, array)
        domain_records.append(
            {
                "index": index,
                "frame_id": ref.id,
                "summary": domain.to_summary(),
            }
        )
        del array, loaded

    convention_for_report = selected_convention if mask_array is not None else None
    finite_q_for_report = np.asarray(qmap["q"])[np.isfinite(qmap["q"])]
    checks: list[dict[str, Any]] = []
    warnings: list[str] = []
    errors: list[str] = []
    _add_check(checks, warnings, errors, "package", "pass", "package directory is readable", {"path": package_root.as_posix()})
    manifest_status = "pass"
    if manifest_quality is not None:
        manifest_evidence = dict(manifest_quality)
        manifest_reasons: list[str] = []
        if manifest_quality["duplicate_orders"]:
            manifest_status = "fail"
            manifest_reasons.append("manifest order values must be unique")
        if not manifest_quality["selector_unique"]:
            manifest_status = "fail"
            manifest_reasons.append(
                "manifest path/dataset/frame selector identities must be unique"
            )
        if not manifest_quality["time_finite"]:
            manifest_status = "fail"
            manifest_reasons.append("manifest time values must be finite numbers")
        if not manifest_quality["time_monotonic"]:
            manifest_status = "fail"
            manifest_reasons.append("manifest finite time values must be monotonic")
        if manifest_quality["missing_time_indices"] and manifest_status == "pass":
            manifest_status = "warn"
            manifest_reasons.append("manifest time is missing for one or more frames")
        manifest_reason = "; ".join(manifest_reasons) or "manifest structure is valid"
    else:
        manifest_evidence = {"provided": False, "frame_count": len(refs)}
        manifest_reason = "frames use natural filename order"
    _add_check(
        checks,
        warnings,
        errors,
        "manifest",
        manifest_status,
        manifest_reason,
        {
            **manifest_evidence,
            "provided": selected_manifest is not None,
            "path": _display_path(manifest_source, package_root) if manifest_source else None,
            "frame_count": len(refs),
        },
    )
    _add_check(checks, warnings, errors, "frames", "pass", "all selected frames are readable and shape-consistent", {"count": len(frames), "shape": list(first_shape)})
    intensity_unit = _nested_mapping(context_value, "source_intensity").get("unit")
    if intensity_unit is None:
        _add_check(
            checks,
            warnings,
            errors,
            "intensity_unit",
            "warn",
            "source intensity unit is not declared",
            {"unit": None},
        )
    else:
        _add_check(
            checks,
            warnings,
            errors,
            "intensity_unit",
            "pass",
            "source intensity unit is declared",
            {"unit": str(intensity_unit)},
        )
    if qmap["source"] == "pixel-q":
        _add_check(checks, warnings, errors, "q_unit", "warn", "pixel-q is uncalibrated and has no physical reciprocal-space unit", {"q_unit": "pixel-q"})
        geometry_reason = "PONI not supplied; deterministic pixel-q fallback used"
        geometry_status = "warn"
    else:
        geometry_reason = "q map built from supplied geometry"
        geometry_status = "pass"
        if str(qmap.get("q_unit", "unknown")).casefold() in {"unknown", "", "none"}:
            geometry_status = "warn"
            geometry_reason = "q map unit is unknown"
    _add_check(checks, warnings, errors, "geometry", geometry_status, geometry_reason, {"source": qmap["source"], "q_unit": qmap["q_unit"], "shape": list(first_shape)})
    if mask_array is None:
        _add_check(checks, warnings, errors, "mask", "pass", "no external mask supplied", {"provided": False})
    else:
        invalid_count = int(np.count_nonzero(external_mask)) if external_mask is not None else int(np.count_nonzero(~valid_mask))
        _add_check(
            checks,
            warnings,
            errors,
            "mask",
            "pass",
            "mask shape and convention are valid",
            {
                "provided": True,
                "path": _display_path(mask_source, package_root) if mask_source else "in-memory",
                "convention": convention_for_report,
                "invalid_count": invalid_count,
                "invalid_fraction": invalid_count / mask_array.size,
            },
        )
    _add_check(checks, warnings, errors, "analysis_domain", "pass", "all frames share the unified fit domain contract", {"counts": reference_domain.counts, "per_frame": domain_records})

    negative_count = sum(int(item["summary"].get("negative_count", 0)) for item in frames)
    if negative_count:
        _add_check(checks, warnings, errors, "negative_intensity", "warn", "negative finite intensity values were observed; no values were rewritten", {"count": negative_count})
    else:
        _add_check(checks, warnings, errors, "negative_intensity", "pass", "no negative finite intensity values observed", {"count": 0})

    correction_value, correction_check_value = _resolve_correction_value(
        correction_state,
        context_value,
    )
    correction_status, correction_reason, correction_evidence = _correction_check(correction_check_value)
    _add_check(checks, warnings, errors, "correction_state", correction_status, correction_reason, correction_evidence)
    uncertainty_value = _state_value(uncertainty_state, context_value, "uncertainty_state")
    uncertainty_status, uncertainty_reason, uncertainty_evidence = _uncertainty_check(uncertainty_value)
    uncertainty_provenance = _uncertainty_provenance(
        image_metadata,
        package_root,
        hash_groups["uncertainty"],
    )
    uncertainty_evidence = {**uncertainty_evidence, **uncertainty_provenance}
    if isinstance(uncertainty_value, Mapping):
        uncertainty_label = str(
            _first_value(
                uncertainty_value.get("status"),
                uncertainty_value.get("state"),
                uncertainty_value.get("uncertainty_status"),
                "unknown",
            )
        ).casefold()
    else:
        uncertainty_label = str(uncertainty_value or "unknown").casefold()
    complete_uncertainty = uncertainty_label in {"complete", "completed"}
    complete_inventory = (
        uncertainty_provenance["declared_file_count"] > 0
        and uncertainty_provenance["resolved_file_count"]
        == uncertainty_provenance["declared_file_count"]
        and uncertainty_provenance["dataset_inventory_status"]
        == "reference_schema_read"
        and bool(uncertainty_provenance["datasets"])
        and bool(uncertainty_provenance["units"])
    )
    if complete_uncertainty and not complete_inventory:
        uncertainty_status = "fail"
        uncertainty_reason = (
            "complete uncertainty was declared without a complete file/dataset/unit inventory"
        )
    _add_check(checks, warnings, errors, "uncertainty_state", uncertainty_status, uncertainty_reason, uncertainty_evidence)

    frame_structure = _nested_mapping(context_value, "frame_structure")
    time_policy = str(_first_value(frame_structure.get("time_zero_policy"), "")).casefold()
    provisional = "provisional" in time_policy or any(
        "provisional" in str(item.get("manifest_frame", {}).get("metadata", {}).get("time_basis", "")).casefold()
        for item in frames
    )
    if provisional:
        _add_check(checks, warnings, errors, "time_basis", "warn", "time basis is provisional", {"time_zero_policy": time_policy})
    else:
        _add_check(checks, warnings, errors, "time_basis", "pass", "no provisional time basis was declared", {"time_zero_policy": time_policy or None})

    header_status, header_reason, header_evidence = _header_geometry_check(
        image_metadata,
        qmap.get("metadata", {}),
        qmap["source"] == "poni",
    )
    _add_check(checks, warnings, errors, "header_geometry_comparison", header_status, header_reason, header_evidence)

    if output is None:
        _add_check(
            checks,
            warnings,
            errors,
            "evidence_persistence",
            "warn",
            "output was not requested; JSON/NPZ evidence was not persisted",
            {"output": None, "arrays_persisted": False},
        )
    else:
        _add_check(
            checks,
            warnings,
            errors,
            "evidence_persistence",
            "pass",
            "strict JSON and NPZ evidence targets were selected",
            {"output": str(output), "arrays_persisted": True},
        )

    status_color, scientific_status = _status_from_checks(checks)
    structural_failure_ids = {
        "package",
        "manifest",
        "frames",
        "q_unit",
        "geometry",
        "mask",
        "analysis_domain",
    }
    structural_failure = any(
        item.get("status") == "fail" and item.get("id") in structural_failure_ids
        for item in checks
    )
    exit_code = 0 if status_color == "green" else (2 if structural_failure else 1)
    output_paths, overwritten_paths = _prepare_outputs(output, force)
    hash_summary = dict(hash_groups)
    for key in ("manifest", "poni", "mask", "context", "uncertainty"):
        entries = hash_groups[key]
        hash_summary[f"{key}_sha256"] = entries[0]["sha256_after"] if entries else None
    hash_summary["input_sha256"] = [item["sha256_after"] for item in hash_groups["inputs"]]
    all_hash_records = [
        record
        for entries in hash_groups.values()
        for record in entries
    ]
    provenance_files: list[dict[str, Any]] = []
    for role, entries in hash_groups.items():
        provenance_role = "image" if role == "inputs" else role
        provenance_files.extend(
            {
                "role": provenance_role,
                "path": record["path"],
                "sha256": record["sha256_after"],
                "before": record["sha256_before"],
                "after": record["sha256_after"],
                "unchanged": bool(record["unchanged"]),
            }
            for record in entries
        )

    output_directory = (
        output_paths["preflight_json"].parent if output_paths is not None else None
    )
    arrays_path = "arrays.npz" if output_paths is not None else None
    mask_keys = {
        "finite_mask": "finite_mask",
        "detector_valid_mask": "detector_valid_mask",
        "external_valid_mask": "external_valid_mask",
        "q_window_mask": "q_window_mask",
        "roi_exclusion_mask": "roi_exclusion_mask",
        "weight_valid_mask": "weight_valid_mask",
        "fit_valid_mask": "fit_valid_mask",
        "sampled_valid_mask": "sampled_valid_mask",
    }
    q_unit = str(qmap["q_unit"])
    if q_unit == "nm^-1":
        q_keys = {"qx": "qx_nm_inv", "qy": "qy_nm_inv", "q": "q_nm_inv", "chi": "chi_rad"}
        q_key_units = {
            "qx_nm_inv": "nm^-1",
            "qy_nm_inv": "nm^-1",
            "q_nm_inv": "nm^-1",
            "chi_rad": "rad",
        }
        coordinate_system = "physical_q"
        q_axis_label = "nm^-1"
    elif q_unit == "pixel-q":
        q_keys = {
            "qx": "qx_pixel_q",
            "qy": "qy_pixel_q",
            "q": "q_pixel_q",
            "chi": "chi_rad",
        }
        q_key_units = {
            "qx_pixel_q": "pixel-q",
            "qy_pixel_q": "pixel-q",
            "q_pixel_q": "pixel-q",
            "chi_rad": "rad",
        }
        coordinate_system = "pixel_q"
        q_axis_label = "pixel-q"
    else:
        q_keys = {"qx": "qx", "qy": "qy", "q": "q", "chi": "chi_rad"}
        q_key_units = {"qx": "unknown", "qy": "unknown", "q": "unknown", "chi_rad": "rad"}
        coordinate_system = "unknown"
        q_axis_label = "unknown"

    source_intensity = _nested_mapping(context_value, "source_intensity")
    local_data_policy = _nested_mapping(context_value, "local_data_policy")
    frame_structure = _nested_mapping(context_value, "frame_structure")
    input_images: list[dict[str, Any]] = []
    input_hash_records = hash_groups["inputs"]
    for index, frame_record in enumerate(frames):
        hash_record = input_hash_records[index] if index < len(input_hash_records) else None
        input_images.append(
            {
                "path": frame_record["path"],
                "sha256": hash_record["sha256_after"] if hash_record else None,
                "frame_id": frame_record["id"],
                "shape": frame_record["shape"],
                "dtype": frame_record["dtype"],
            }
        )

    first_ref = refs[0]
    first_manifest_metadata = dict(first_ref.metadata or {})
    selector_time: float | None = None
    if first_ref.time is not None:
        try:
            candidate_time = float(first_ref.time)
        except (TypeError, ValueError, OverflowError):
            candidate_time = float("nan")
        if math.isfinite(candidate_time):
            selector_time = candidate_time
    mask_hash = hash_groups["mask"][0]["sha256_after"] if hash_groups["mask"] else None
    poni_hash = hash_groups["poni"][0]["sha256_after"] if hash_groups["poni"] else None
    correction_details = (
        correction_check_value
        if isinstance(correction_check_value, Mapping)
        else source_intensity
    )
    declared_steps = correction_details.get("already_applied", [])
    not_applied_steps = _first_value(
        correction_details.get("not_burned_into_2d_values"),
        correction_details.get("not_applied"),
        [],
    )
    if not isinstance(declared_steps, Sequence) or isinstance(declared_steps, (str, bytes)):
        declared_steps = []
    if not isinstance(not_applied_steps, Sequence) or isinstance(not_applied_steps, (str, bytes)):
        not_applied_steps = []
    correction_label = _canonical_state(correction_value, "correction")
    uncertainty_label = _canonical_state(uncertainty_value, "uncertainty")
    uncertainty_components = {
        str(item.get("component", item.get("dataset", "unknown"))): {
            "dataset": item.get("dataset"),
            "shape": item.get("shape"),
            "unit": item.get("unit"),
        }
        for item in uncertainty_provenance.get("datasets", [])
    }

    structured_checks = [_structured_check(item) for item in checks]
    status_flags = [_status_item(item) for item in checks if item.get("status") != "pass"]
    failure_reasons = [_status_item(item) for item in checks if item.get("status") == "fail"]
    created_at = datetime.now().astimezone()
    run_id = f"preflight-{created_at.strftime('%Y%m%dT%H%M%S%z')}-{uuid.uuid4().hex[:8]}"
    dependencies = _dependency_versions()
    schema_source = Path(__file__).with_name("validation.py")
    code_sha256 = _sha256_file(Path(__file__))
    source_tree_sha256 = _source_tree_sha256()
    schema_sha256 = _sha256_file(schema_source)
    dependencies_sha256 = _inline_hash(dependencies)
    source_data_local = any(part.casefold() == "data_local" for part in package_root.parts)
    upload_allowed = bool(
        local_data_policy.get("upload_allowed", not source_data_local)
    )

    report: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "result_type": "preflight",
        "run_id": run_id,
        "created_at": created_at.isoformat(timespec="seconds"),
        "tool": {"name": "LamellarSAXS2D", "version": "0.1.0"},
        "status": {
            "status_color": status_color,
            "scientific_status": scientific_status,
            "solver_status": "not_run",
            "numerical_status": "NOT_TESTED",
            "exit_code": exit_code,
            "flags": status_flags,
            "failure_reasons": failure_reasons,
        },
        "input": {
            "source_kind": "manifest" if selected_manifest is not None else "image",
            "images": input_images,
            "manifest_path": (
                _display_path(manifest_source, package_root) if manifest_source else None
            ),
            "intensity_unit": source_intensity.get("unit"),
            "read_only": True,
        },
        "selector": {
            "image": {
                "path": frames[0]["path"],
                "frame": frames[0]["frame"],
                "dataset": frames[0]["dataset"],
            },
            "mask": {
                "path": (
                    _display_path(mask_source, package_root)
                    if mask_source is not None
                    else ("in-memory" if mask_array is not None else None)
                ),
                "frame": mask_frame,
                "dataset": mask_dataset,
            },
            "manifest": {
                "path": (
                    _display_path(manifest_source, package_root)
                    if manifest_source is not None
                    else None
                ),
                "row_id": first_manifest_metadata.get("row_id"),
                "frame_id": first_ref.id,
                "order": (
                    first_ref.order
                    if manifest_quality is not None
                    and manifest_quality.get("explicit_order")
                    else None
                ),
                "time": selector_time,
                "time_unit": (
                    first_manifest_metadata.get("time_unit")
                    or ("s" if first_ref.time is not None else None)
                ),
                "time_source": first_manifest_metadata.get(
                    "time_basis",
                    frame_structure.get("time_zero_policy", "unknown"),
                ),
            },
        },
        "geometry": {
            "poni": {
                "path": (
                    hash_groups["poni"][0]["path"] if hash_groups["poni"] else None
                ),
                "sha256": poni_hash,
                "valid": qmap["source"] == "poni",
            },
            "q_unit": q_unit,
            "source_q_unit": qmap.get("source_q_unit"),
            "q_conversion_factor_to_nm_inv": qmap.get(
                "q_conversion_factor_to_nm_inv"
            ),
            "coordinate_system": coordinate_system,
            "q_window": {
                "min": float(reference_domain.q_window[0]),
                "max": float(reference_domain.q_window[1]),
                "unit": q_unit,
            },
            "q_range": {
                "min": float(np.min(finite_q_for_report)),
                "max": float(np.max(finite_q_for_report)),
                "unit": q_unit,
            },
            "axis_labels": {
                "qx": f"qx ({q_axis_label})",
                "qy": f"qy ({q_axis_label})",
            },
            "arrays_npz": {
                "path": arrays_path,
                "keys": q_keys,
                "key_units": q_key_units,
            },
        },
        "mask": {
            "source": {
                "path": (
                    _display_path(mask_source, package_root)
                    if mask_source is not None
                    else ("in-memory" if mask_array is not None else None)
                ),
                "sha256": mask_hash,
                "raw_polarity": convention_for_report,
            },
            "shape": list(first_shape),
            "valid_mask_polarity": "true_valid",
            "external_mask_polarity": "true_invalid",
            "roi_exclusion_polarity": "true_invalid",
        },
        "correction_state": correction_label,
        "correction": {
            "source_files": [
                record["path"] for record in hash_groups["context"]
            ],
            "declared_steps": [str(item) for item in declared_steps],
            "not_applied_steps": [str(item) for item in not_applied_steps],
            "software_reapply_prohibited": True,
            "absolute_intensity_comparable": bool(
                correction_label == "fully_corrected_external"
                and not not_applied_steps
            ),
        },
        "uncertainty_state": uncertainty_label,
        "uncertainty": {
            "sources": list(uncertainty_provenance.get("files", [])),
            "components": uncertainty_components,
            "units": list(uncertainty_provenance.get("units", [])),
            "stderr_scope": (
                "complete" if uncertainty_label == "complete" else uncertainty_label
            ),
            "separate_from_selection_uncertainty": True,
            "declared_file_count": uncertainty_provenance.get("declared_file_count", 0),
            "resolved_file_count": uncertainty_provenance.get("resolved_file_count", 0),
            "dataset_inventory_status": uncertainty_provenance.get(
                "dataset_inventory_status"
            ),
        },
        "analysis_domain": {
            "schema_version": reference_domain.schema_version,
            "status": "computed",
            "image_shape": list(first_shape),
            "q_window": {
                "min": float(reference_domain.q_window[0]),
                "max": float(reference_domain.q_window[1]),
                "unit": q_unit,
            },
            "weight_kind": reference_domain.weight_kind,
            "counts": reference_domain.counts,
            "arrays_npz": {"path": arrays_path, "keys": mask_keys},
        },
        "quality": {
            "status": scientific_status,
            "status_color": status_color,
            "thresholds_version": "p0-p2-contract-v1",
            "checks": structured_checks,
            "flags": status_flags,
            "metrics": {
                "coverage": None,
                "condition_number_scaled": None,
                "bound_flags": [],
                "fit_ndata": None,
                "sampled_n": None,
                "withheld": None,
                "residual": None,
            },
        },
        "measurements": None,
        "fit": None,
        "interpretation": {
            "model_scope": "empirical",
            "interpretation_limit": "nonunique_inverse_problem",
            "claims_allowed": ["只读输入、几何、掩膜和像素域预检"],
            "claims_forbidden": [
                "由预检结果推断物理周期",
                "由单张二维 SAXS 唯一确定三维层片结构",
                "把预检或拟合趋势作为材料机理证明",
            ],
            "flags": ["apparent_geometry_only", "nonunique_inverse_problem"],
        },
        "outputs": {
            "directory": output_directory.as_posix() if output_directory else None,
            "paths_relative": True,
            "files": (
                {
                    "preflight_json": "preflight.json",
                    "arrays_npz": "arrays.npz",
                    "run_report": "run_report.md",
                }
                if output_paths is not None
                else {}
            ),
            "overwrite": bool(overwritten_paths),
            "force": bool(force),
            "overwritten_paths": [path.name for path in overwritten_paths],
        },
        "provenance": {
            "command": "bsaxs preflight (see provenance.arguments for the exact resolved inputs)",
            "arguments": {
                "package": package_root.as_posix(),
                "manifest": (
                    _display_path(manifest_source, package_root)
                    if manifest_source is not None
                    else None
                ),
                "poni": (
                    str(selected_poni)
                    if isinstance(selected_poni, (str, os.PathLike))
                    else "in-memory"
                    if selected_poni is not None
                    else None
                ),
                "mask": (
                    str(selected_mask)
                    if isinstance(selected_mask, (str, os.PathLike))
                    else "in-memory"
                    if selected_mask is not None
                    else None
                ),
                "context": str(context_source) if context_source is not None else None,
                "image_glob": str(selected_glob) if selected_glob is not None else None,
                "frame": frame,
                "dataset": dataset,
                "mask_frame": mask_frame,
                "mask_dataset": mask_dataset,
                "q_window": domain_q_window,
                "output": output_directory.as_posix() if output_directory else None,
                "force": bool(force),
            },
            "working_directory": Path.cwd().as_posix(),
            "git_commit": _git_commit(),
            "dependencies": dependencies,
            "hashes": {
                "algorithm": "SHA-256",
                "files": provenance_files,
                "config_sha256": hash_summary.get("context_sha256"),
                "code_sha256": code_sha256,
                "source_tree_sha256": source_tree_sha256,
                "schema_sha256": schema_sha256,
                "dependencies_sha256": dependencies_sha256,
            },
            "input_unchanged": {
                "checked": bool(all_hash_records),
                "before_after_equal": all(
                    bool(item.get("unchanged")) for item in all_hash_records
                ),
            },
            "privacy": {
                "source_data_local": source_data_local,
                "upload_allowed": upload_allowed,
            },
        },
        "extensions": {
            "preflight": {
                "schema_version": PREFLIGHT_SCHEMA_VERSION,
                "package": package_root.as_posix(),
                "image_glob": _json_safe(selected_glob),
                "frames": frames,
                "per_frame_analysis_domain": domain_records,
                "reference_frame_id": refs[0].id,
                "manifest_quality": manifest_quality,
                "checks": checks,
                "hashes": hash_summary,
                "warnings": warnings,
                "errors": errors,
                "analysis_stages": {
                    "ridge": "NOT_TESTED",
                    "lobe": "NOT_TESTED",
                    "ellipse": "NOT_TESTED",
                    "full2d": "NOT_TESTED",
                },
            }
        },
    }
    safe_report = _json_safe(report)
    validate_result_schema(safe_report)
    if output_paths is not None:
        output_paths["preflight_json"].parent.mkdir(parents=True, exist_ok=True)
        q_arrays = {
            q_keys["qx"]: np.asarray(qmap["qx"], dtype=float),
            q_keys["qy"]: np.asarray(qmap["qy"], dtype=float),
            q_keys["q"]: np.asarray(qmap["q"], dtype=float),
            q_keys["chi"]: np.arctan2(
                np.asarray(qmap["qy"], dtype=float),
                np.asarray(qmap["qx"], dtype=float),
            ),
        }
        _atomic_npz(
            output_paths["arrays_npz"],
            {
                **q_arrays,
                "finite_mask": np.asarray(reference_domain.finite_mask, dtype=bool),
                "detector_valid_mask": np.asarray(
                    reference_domain.detector_valid_mask, dtype=bool
                ),
                "external_valid_mask": np.asarray(
                    reference_domain.external_valid_mask, dtype=bool
                ),
                "q_window_mask": np.asarray(reference_domain.q_window_mask, dtype=bool),
                "roi_exclusion_mask": np.asarray(
                    reference_domain.roi_exclusion_mask, dtype=bool
                ),
                "weight_valid_mask": np.asarray(
                    reference_domain.weight_valid_mask, dtype=bool
                ),
                "fit_valid_mask": np.asarray(reference_domain.fit_valid_mask, dtype=bool),
                "sampled_valid_mask": np.asarray(
                    reference_domain.sampled_valid_mask, dtype=bool
                ),
                "frame_id": np.asarray(refs[0].id),
            },
        )
        _atomic_text(
            output_paths["preflight_json"],
            json.dumps(safe_report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        )
        _atomic_text(output_paths["run_report"], _markdown_report(safe_report))
        return safe_report
    return safe_report


__all__ = ["PREFLIGHT_SCHEMA_VERSION", "PreflightError", "run_preflight"]
