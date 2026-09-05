"""Resolved, read-only inputs for the preflight orchestration boundary."""

from __future__ import annotations

from dataclasses import dataclass
import csv
import glob as glob_module
import json
import math
from pathlib import Path
import os
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .batch import FrameRef, natural_sort_key


MANIFEST_KEYS = ("frames", "frame_manifest", "manifest", "data", "items")
PATH_KEYS = ("path", "input_path", "file", "filename")
IMAGE_SUFFIXES = {
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



def inline_record(
    value: Any,
    label: str,
    records: list[dict[str, Any]],
    *,
    inline_hash_fn: Callable[[Any], str],
) -> dict[str, Any]:
    digest = inline_hash_fn(value)
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

def require_package(
    package: str | os.PathLike[str] | Path,
    *,
    error_type: type[Exception] = ValueError,
) -> Path:
    root = Path(package).expanduser().resolve(strict=False)
    if not root.exists() or not root.is_dir():
        raise error_type(f"package must be an existing directory: {root}")
    return root

def load_yaml_context(
    path: Path,
    package: Path,
    records: list[dict[str, Any]],
    *,
    read_file_record: Callable[..., Any],
    error_type: type[Exception] = ValueError,
) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - exercised without optional dep
        raise error_type(
            "YAML context requires PyYAML; install PyYAML before passing a YAML context"
        ) from exc

    def read() -> str:
        try:
            return path.read_text(encoding="utf-8-sig")
        except OSError as exc:
            raise error_type(f"cannot read context file {path}: {exc}") from exc

    text = read_file_record(path, package, read, records)
    try:
        value = yaml.safe_load(text)
    except Exception as exc:  # yaml exposes several parser exception classes
        raise error_type(f"could not parse YAML context {path}: {exc}") from exc
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise error_type("context YAML must contain a mapping at its top level")
    return dict(value)

def load_context(
    package: Path,
    context: Any,
    records: list[dict[str, Any]],
    external_roots: Sequence[str | os.PathLike[str] | Path] | None = None,
    *,
    resolve_path: Callable[..., Path],
    load_yaml_context: Callable[..., dict[str, Any]],
    read_file_record: Callable[..., Any],
    inline_record: Callable[..., dict[str, Any]],
    error_type: type[Exception] = ValueError,
) -> tuple[dict[str, Any], Path | None]:
    source: Path | None = None
    if context is None:
        for name in ("project_context.yaml", "project_context.yml"):
            try:
                candidate = resolve_path(
                    package,
                    name,
                    external_roots=external_roots,
                    label="auto-discovered context",
                )
            except error_type:
                # A package-relative symlink that resolves outside the package
                # is not an auto-discoverable context.  Do not fall back to
                # the unresolved link path and reopen the escape.
                continue
            if candidate.is_file():
                source = candidate
                break
        if source is None:
            return {}, None
        return load_yaml_context(
            source,
            package,
            records,
            read_file_record=read_file_record,
            error_type=error_type,
        ), source
    if isinstance(context, Mapping):
        value = dict(context)
        inline_record(value, "context:in-memory", records)
        return value, None
    if not isinstance(context, (str, os.PathLike, Path)):
        raise error_type("context must be a mapping, YAML path, or None")
    source = resolve_path(package, context, external_roots=external_roots, label="context")
    if not source.exists() or not source.is_file():
        raise error_type(f"context file does not exist: {source}")
    if source.suffix.casefold() not in {".yaml", ".yml"}:
        raise error_type("context path must use .yaml or .yml")
    return load_yaml_context(
        source,
        package,
        records,
        read_file_record=read_file_record,
        error_type=error_type,
    ), source

def parse_manifest_file(path: Path, *, error_type: type[Exception] = ValueError) -> Any:
    suffix = path.suffix.casefold()
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise error_type(f"cannot read manifest {path}: {exc}") from exc
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
        raise error_type(f"could not parse manifest {path}: {exc}") from exc
    except ImportError as exc:  # pragma: no cover - tomllib is stdlib on supported Python
        raise error_type("TOML manifest requires Python 3.11 or newer") from exc
    raise error_type("manifest must be CSV, JSON, or TOML")

def manifest_rows(
    value: Any,
    *,
    error_type: type[Exception] = ValueError,
) -> list[Any]:
    """Flatten supported manifest envelopes without changing row order."""

    if isinstance(value, Mapping):
        for key in MANIFEST_KEYS:
            candidate = value.get(key)
            if isinstance(candidate, Sequence) and not isinstance(candidate, (str, bytes)):
                return list(candidate)
        if any(key in value for key in PATH_KEYS):
            return [value]
        rows: list[dict[str, Any]] = []
        for path, metadata in value.items():
            row = dict(metadata) if isinstance(metadata, Mapping) else {"time": metadata}
            row.setdefault("path", path)
            rows.append(row)
        return rows
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return list(value)
    raise error_type("manifest must contain a sequence of frame rows")

def normalise_manifest_row(
    row: Any,
    package: Path,
    *,
    base_dir: Path | None = None,
    external_roots: Sequence[str | os.PathLike[str] | Path] | None = None,
    resolve_path: Callable[..., Path],
    error_type: type[Exception] = ValueError,
) -> Any:
    anchor = package if base_dir is None else base_dir
    if isinstance(row, FrameRef):
        return FrameRef(
            resolve_path(package, row.path, base_dir=anchor, external_roots=external_roots, label="manifest row"),
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
        for key in PATH_KEYS:
            candidate = result.get(key)
            if candidate is None:
                continue
            if isinstance(candidate, (str, os.PathLike, Path)):
                result[key] = str(
                    resolve_path(
                        package,
                        candidate,
                        base_dir=anchor,
                        external_roots=external_roots,
                        label="manifest row",
                    )
                )
            break
        return result
    if isinstance(row, (str, os.PathLike, Path)):
        return {
            "path": str(
                resolve_path(
                    package,
                    row,
                    base_dir=anchor,
                    external_roots=external_roots,
                    label="manifest row",
                )
            )
        }
    raise error_type(f"manifest row is not a path or mapping: {row!r}")

def resolve_manifest(
    package: Path,
    manifest: Any,
    records: list[dict[str, Any]],
    external_roots: Sequence[str | os.PathLike[str] | Path] | None = None,
    *,
    resolve_path: Callable[..., Path],
    read_file_record: Callable[..., Any],
    inline_record: Callable[..., dict[str, Any]],
    build_frame_refs: Callable[..., list[FrameRef]],
    parse_manifest_file: Callable[..., Any],
    error_type: type[Exception] = ValueError,
) -> tuple[list[FrameRef] | None, Path | None, dict[str, Any] | None]:
    if manifest is None:
        return None, None, None
    source: Path | None = None
    if isinstance(manifest, (str, os.PathLike, Path)):
        source = resolve_path(package, manifest, external_roots=external_roots, label="manifest")
        if not source.exists() or not source.is_file():
            raise error_type(f"manifest file does not exist: {source}")
        parsed = read_file_record(
            source,
            package,
            lambda: parse_manifest_file(source, error_type=error_type),
            records,
        )
    else:
        parsed = manifest
        inline_record(parsed, "manifest:in-memory", records)
    rows = [
        normalise_manifest_row(
            row,
            package,
            base_dir=source.parent if source is not None else package,
            external_roots=external_roots,
            resolve_path=resolve_path,
            error_type=error_type,
        )
        for row in manifest_rows(parsed, error_type=error_type)
    ]
    try:
        refs = build_frame_refs([], manifest=rows)
    except (TypeError, ValueError) as exc:
        raise error_type(f"could not resolve manifest frame references: {exc}") from exc
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

def nested_mapping(context: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = context.get(key)
    return value if isinstance(value, Mapping) else {}

def first_value(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None

def context_settings(context: Mapping[str, Any]) -> dict[str, Any]:
    preferred = nested_mapping(context, "preferred_project_input")
    return {
        "manifest": first_value(
            context.get("manifest"), context.get("manifest_path"), preferred.get("manifest")
        ),
        "image_glob": first_value(
            preferred.get("image_glob"),
            preferred.get("preferred_image_glob"),
            preferred.get("preferred_edf_glob"),
            context.get("image_glob"),
            context.get("preferred_edf_glob"),
        ),
        "poni": first_value(
            preferred.get("poni"),
            preferred.get("poni_path"),
            preferred.get("qmap"),
            context.get("poni"),
            context.get("poni_path"),
            context.get("qmap"),
        ),
        "mask": first_value(
            preferred.get("mask"), preferred.get("mask_path"), context.get("mask"), context.get("mask_path")
        ),
        "mask_convention": first_value(
            preferred.get("mask_convention"), context.get("mask_convention")
        ),
        "q_window": first_value(
            preferred.get("q_window_nm_inv"),
            preferred.get("q_window"),
            preferred.get("q_range"),
            context.get("q_window_nm_inv"),
            context.get("q_window"),
        ),
        "q_unit": first_value(preferred.get("q_unit"), context.get("q_unit")),
        "external_roots": first_value(
            preferred.get("external_roots"), context.get("external_roots")
        ),
    }

def glob_inputs(
    package: Path,
    pattern: Any,
    external_roots: Sequence[str | os.PathLike[str] | Path] | None = None,
    *,
    resolve_path: Callable[..., Path],
) -> list[Path]:
    if pattern is None:
        patterns = [f"**/*{suffix}" for suffix in sorted(IMAGE_SUFFIXES)]
        matches: list[Path] = []
        for candidate in patterns:
            matches.extend(Path(item).resolve() for item in glob_module.glob(str(package / candidate), recursive=True))
    else:
        text = os.fspath(pattern)
        glob_pattern = text if Path(text).expanduser().is_absolute() else str(package / text)
        matches = [Path(item).resolve() for item in glob_module.glob(glob_pattern, recursive=True)]
    authorized: list[Path] = []
    for path in matches:
        if not path.is_file() or path.suffix.casefold() not in IMAGE_SUFFIXES:
            continue
        authorized.append(
            resolve_path(
                package,
                path,
                external_roots=external_roots,
                label="image glob",
            )
        )
    return sorted(
        set(authorized),
        key=lambda path: natural_sort_key(path.name),
    )


__all__ = [
    "IMAGE_SUFFIXES",
    "MANIFEST_KEYS",
    "PATH_KEYS",
    "PreflightContext",
    "ResolvedPreflightInputs",
    "context_settings",
    "first_value",
    "glob_inputs",
    "inline_record",
    "load_context",
    "load_yaml_context",
    "manifest_rows",
    "nested_mapping",
    "normalise_manifest_row",
    "parse_manifest_file",
    "require_package",
    "resolve_inputs",
    "resolve_manifest",
]



@dataclass(frozen=True)
class PreflightContext:
    """The selected input contract after path/manifest resolution.

    Detector arrays and mutable hash records remain outside this value.  The
    facade keeps them in its streaming locals, while this object makes the
    selected roots, selectors, and frame order explicit for checks and report
    construction.
    """

    package_root: Path
    context_value: Mapping[str, Any]
    context_source: Path | None
    external_roots: tuple[Path, ...]
    selected_manifest: Any
    selected_glob: Any
    selected_poni: Any
    selected_mask: Any
    selected_q_window: Any
    selected_convention: str
    refs: tuple[Any, ...]
    manifest_source: Path | None
    manifest_quality: Mapping[str, Any] | None


@dataclass(frozen=True)
class ResolvedPreflightInputs:
    """Input-resolution result consumed by the streaming read/check stage."""

    context: PreflightContext
    hash_groups: dict[str, list[dict[str, Any]]]


def resolve_inputs(
    package: str | os.PathLike[str] | Path,
    *,
    manifest: Any,
    poni: Any,
    mask: Any,
    context: Any,
    image_glob: str | os.PathLike[str] | None,
    frame: int | None,
    dataset: str | None,
    mask_frame: int | None,
    mask_dataset: str | None,
    q_window: Any,
    mask_convention: str | None,
    output: Any,
    external_roots: Sequence[str | os.PathLike[str] | Path] | None,
    require_package: Callable[[Any], Path],
    load_context: Callable[..., tuple[dict[str, Any], Path | None]],
    context_settings: Callable[[Mapping[str, Any]], dict[str, Any]],
    resolve_manifest: Callable[..., tuple[list[Any] | None, Path | None, dict[str, Any] | None]],
    glob_inputs: Callable[..., list[Path]],
    resolve_path: Callable[..., Path],
    resolve_output_path: Callable[[Any], Path],
    build_frame_refs: Callable[..., list[Any]],
    normalise_mask_convention: Callable[[Any], str],
    error_type: type[Exception] = ValueError,
) -> ResolvedPreflightInputs:
    """Resolve selectors and authorized roots without reading detector arrays."""

    package_root = require_package(package)
    import numpy as np

    if isinstance(frame, bool) or (
        frame is not None
        and (not isinstance(frame, (int, np.integer)) or int(frame) < 0)
    ):
        raise error_type("frame must be a non-negative integer or None")
    if isinstance(mask_frame, bool) or (
        mask_frame is not None
        and (not isinstance(mask_frame, (int, np.integer)) or int(mask_frame) < 0)
    ):
        raise error_type("mask_frame must be a non-negative integer or None")
    if dataset is not None and not isinstance(dataset, str):
        raise error_type("dataset must be a string or None")
    if mask_dataset is not None and not isinstance(mask_dataset, str):
        raise error_type("mask_dataset must be a string or None")

    hash_groups: dict[str, list[dict[str, Any]]] = {
        "inputs": [],
        "manifest": [],
        "poni": [],
        "mask": [],
        "context": [],
        "uncertainty": [],
    }
    context_value, context_source = load_context(
        package_root,
        context,
        hash_groups["context"],
        external_roots=external_roots,
    )
    settings = context_settings(context_value)
    configured_external_roots = external_roots
    if configured_external_roots is None:
        configured_external_roots = settings.get("external_roots")
    if configured_external_roots is None:
        configured_external_roots = None
    if configured_external_roots is not None and isinstance(
        configured_external_roots, (str, os.PathLike, Path)
    ):
        configured_external_roots = (configured_external_roots,)
    selected_manifest = manifest if manifest is not None else settings["manifest"]
    refs, manifest_source, manifest_quality = resolve_manifest(
        package_root,
        selected_manifest,
        hash_groups["manifest"],
        external_roots=configured_external_roots,
    )
    selected_glob = image_glob if image_glob is not None else settings["image_glob"]
    selected_poni = poni if poni is not None else settings["poni"]
    selected_mask = mask if mask is not None else settings["mask"]
    selected_q_window = q_window if q_window is not None else settings["q_window"]
    selected_convention = normalise_mask_convention(
        mask_convention if mask_convention is not None else settings["mask_convention"]
    )

    if refs is None:
        paths = glob_inputs(package_root, selected_glob, configured_external_roots)
        exclusions: set[Path] = set()
        for candidate in (selected_mask, selected_poni):
            if isinstance(candidate, (str, os.PathLike, Path)):
                exclusions.add(
                    resolve_path(
                        package_root,
                        candidate,
                        external_roots=configured_external_roots,
                        label="input exclusion",
                    )
                )
        if context_source is not None:
            exclusions.add(context_source)
        output_root = (
            resolve_output_path(output)
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
        raise error_type(f"no input frames found from {source_hint}")

    resolved_context = PreflightContext(
        package_root=package_root,
        context_value=context_value,
        context_source=context_source,
        external_roots=tuple(
            Path(item).expanduser().resolve(strict=False)
            for item in (configured_external_roots or ())
        ),
        selected_manifest=selected_manifest,
        selected_glob=selected_glob,
        selected_poni=selected_poni,
        selected_mask=selected_mask,
        selected_q_window=selected_q_window,
        selected_convention=selected_convention,
        refs=tuple(refs),
        manifest_source=manifest_source,
        manifest_quality=manifest_quality,
    )
    return ResolvedPreflightInputs(resolved_context, hash_groups)


__all__ = ["PreflightContext", "ResolvedPreflightInputs", "resolve_inputs"]
