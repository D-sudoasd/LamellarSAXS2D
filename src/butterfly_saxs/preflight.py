"""Read-only package preflight for two-dimensional SAXS data.

The preflight layer deliberately stops before ridge extraction or any fitting.
It resolves the same frame and selector contracts used by :mod:`batch` and
:mod:`io`, then asks :func:`validation.build_analysis_domain` for the pixel
population that a later analysis would be allowed to use.
"""

from __future__ import annotations

import hashlib
from importlib import metadata as importlib_metadata
import json
import math
import os
import platform
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .batch import FrameRef, build_frame_refs
from .io import load_image
from .path_contract import PathContractError, display_path, resolve_authorized_path
from .preflight_artifacts import (
    PreflightArtifactWriter,
    atomic_npz,
    atomic_text,
    markdown_report,
    prepare_outputs,
)
from .preflight_checks import CheckBuilder, build_preflight_checks
from .preflight_context import (
    context_settings,
    first_value,
    glob_inputs,
    inline_record,
    load_context,
    load_yaml_context,
    manifest_rows,
    nested_mapping,
    normalise_manifest_row,
    parse_manifest_file,
    require_package,
    resolve_inputs,
    resolve_manifest,
)
from .preflight_io import (
    frame_record,
    image_summary,
    read_one_frame,
    read_preflight_inputs,
)
from .preflight_hash import (
    hashable_inline as _hashable_inline_impl,
    inline_hash as _inline_hash_impl,
    read_file_record as _read_file_record_impl,
    sha256_bytes as _sha256_bytes_impl,
    sha256_file as _sha256_file_impl,
)
from .preflight_geometry import (
    build_qmap,
    load_mask,
    normalise_mask_convention,
    qmap_from_value,
    qmap_value,
)
from .preflight_uncertainty import (
    correction_check,
    resolve_correction_value,
    state_status,
    state_value,
    uncertainty_check,
    uncertainty_provenance,
)
from .preflight_report import (
    PreflightReportHelpers,
    build_preflight_report_from_stages,
)
from .validation import (
    RESULT_SCHEMA_VERSION,
    AnalysisDomainError,
    build_analysis_domain,
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
    return _sha256_bytes_impl(data)


def _sha256_file(path: Path) -> str:
    return _sha256_file_impl(path, error_type=PreflightError)


def _hashable_inline(value: Any) -> Any:
    return _hashable_inline_impl(value, json_safe=_json_safe)


def _inline_hash(value: Any) -> str:
    return _inline_hash_impl(value, json_safe=_json_safe)

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
        "tifffile",
        "PyYAML",
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
    return display_path(path, package)


def _resolve_path(
    package: Path,
    value: str | os.PathLike[str] | Path,
    *,
    base_dir: Path | None = None,
    external_roots: Sequence[str | os.PathLike[str] | Path] | None = None,
    label: str = "input",
) -> Path:
    try:
        return resolve_authorized_path(
            value,
            package_root=package,
            base_dir=base_dir,
            external_roots=external_roots,
            label=label,
        )
    except PathContractError as exc:
        raise PreflightError(str(exc)) from exc


def _resolve_output_path(value: str | os.PathLike[str] | Path) -> Path:
    """Resolve output relative to the caller's current working directory."""

    return Path(value).expanduser().resolve(strict=False)


def _read_file_record(
    path: Path,
    package: Path,
    reader: Callable[[], Any],
    records: list[dict[str, Any]],
) -> Any:
    return _read_file_record_impl(
        path,
        package,
        reader,
        records,
        display_path=_display_path,
        error_type=PreflightError,
    )

def _inline_record(
    value: Any,
    label: str,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    return inline_record(value, label, records, inline_hash_fn=_inline_hash)


def _require_package(package: str | os.PathLike[str] | Path) -> Path:
    return require_package(package, error_type=PreflightError)


def _load_yaml_context(
    path: Path,
    package: Path,
    records: list[dict[str, Any]],
    *,
    read_file_record: Callable[..., Any] | None = None,
    error_type: type[Exception] = PreflightError,
) -> dict[str, Any]:
    return load_yaml_context(
        path,
        package,
        records,
        read_file_record=read_file_record or _read_file_record,
        error_type=PreflightError,
    )


def _load_context(
    package: Path,
    context: Any,
    records: list[dict[str, Any]],
    external_roots: Sequence[str | os.PathLike[str] | Path] | None = None,
) -> tuple[dict[str, Any], Path | None]:
    return load_context(
        package,
        context,
        records,
        external_roots,
        resolve_path=_resolve_path,
        load_yaml_context=_load_yaml_context,
        read_file_record=_read_file_record,
        inline_record=_inline_record,
        error_type=PreflightError,
    )


def _parse_manifest_file(path: Path) -> Any:
    return parse_manifest_file(path, error_type=PreflightError)


def _manifest_rows(value: Any) -> list[Any]:
    return manifest_rows(value, error_type=PreflightError)


def _normalise_manifest_row(
    row: Any,
    package: Path,
    *,
    base_dir: Path | None = None,
    external_roots: Sequence[str | os.PathLike[str] | Path] | None = None,
) -> Any:
    return normalise_manifest_row(
        row,
        package,
        base_dir=base_dir,
        external_roots=external_roots,
        resolve_path=_resolve_path,
        error_type=PreflightError,
    )


def _resolve_manifest(
    package: Path,
    manifest: Any,
    records: list[dict[str, Any]],
    external_roots: Sequence[str | os.PathLike[str] | Path] | None = None,
) -> tuple[list[FrameRef] | None, Path | None, dict[str, Any] | None]:
    return resolve_manifest(
        package,
        manifest,
        records,
        external_roots,
        resolve_path=_resolve_path,
        read_file_record=_read_file_record,
        inline_record=_inline_record,
        build_frame_refs=build_frame_refs,
        parse_manifest_file=lambda path, **_: _parse_manifest_file(path),
        error_type=PreflightError,
    )


def _nested_mapping(context: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    return nested_mapping(context, key)


def _first_value(*values: Any) -> Any:
    return first_value(*values)


def _context_settings(context: Mapping[str, Any]) -> dict[str, Any]:
    return context_settings(context)


def _glob_inputs(
    package: Path,
    pattern: Any,
    external_roots: Sequence[str | os.PathLike[str] | Path] | None = None,
) -> list[Path]:
    return glob_inputs(
        package,
        pattern,
        external_roots,
        resolve_path=_resolve_path,
    )



def _qmap_value(value: Any, names: Sequence[str], default: Any = None) -> Any:
    return qmap_value(value, names, default)


def _qmap_from_value(value: Any, shape: tuple[int, int]) -> dict[str, Any]:
    return qmap_from_value(value, shape, error_type=PreflightError)


def _build_qmap(
    shape: tuple[int, int],
    poni: Any,
    package: Path,
    records: list[dict[str, Any]],
    external_roots: Sequence[str | os.PathLike[str] | Path] | None = None,
) -> dict[str, Any]:
    return build_qmap(
        shape,
        poni,
        package,
        records,
        external_roots,
        resolve_path=_resolve_path,
        inline_record=_inline_record,
        read_file_record=_read_file_record,
        error_type=PreflightError,
    )

def _image_summary(array: np.ndarray) -> dict[str, Any]:
    return image_summary(array, error_type=PreflightError)



def _load_mask(
    mask: Any,
    package: Path,
    shape: tuple[int, int],
    mask_frame: int | None,
    mask_dataset: str | None,
    records: list[dict[str, Any]],
    external_roots: Sequence[str | os.PathLike[str] | Path] | None = None,
) -> tuple[np.ndarray | None, Path | None]:
    return load_mask(
        mask,
        package,
        shape,
        mask_frame,
        mask_dataset,
        records,
        external_roots,
        resolve_path=_resolve_path,
        inline_record=_inline_record,
        read_file_record=_read_file_record,
        load_image=load_image,
        error_type=PreflightError,
    )


def _normalise_mask_convention(value: Any) -> str:
    return normalise_mask_convention(value, error_type=PreflightError)


def _state_value(explicit: Any, context: Mapping[str, Any], key: str) -> Any:
    return state_value(explicit, context, key, first_value=_first_value)


def _resolve_correction_value(explicit: Any, context: Mapping[str, Any]) -> tuple[Any, Any]:
    return resolve_correction_value(explicit, context, first_value=_first_value)


def _state_status(value: Any, kind: str) -> tuple[str, str, dict[str, Any]]:
    return state_status(
        value,
        kind,
        json_safe=_json_safe,
        first_value=_first_value,
    )


def _correction_check(value: Any) -> tuple[str, str, dict[str, Any]]:
    return correction_check(
        value,
        state_status=_state_status,
        first_value=_first_value,
    )


def _uncertainty_check(value: Any) -> tuple[str, str, dict[str, Any]]:
    return uncertainty_check(
        value,
        state_status=_state_status,
        first_value=_first_value,
    )


def _uncertainty_provenance(
    image_metadata: Sequence[Mapping[str, Any]],
    package: Path,
    hash_records: list[dict[str, Any]],
    external_roots: Sequence[str | os.PathLike[str] | Path] | None = None,
) -> dict[str, Any]:
    return uncertainty_provenance(
        image_metadata,
        package,
        hash_records,
        external_roots,
        resolve_path=_resolve_path,
        display_path=_display_path,
        read_file_record=_read_file_record,
        json_safe=_json_safe,
        error_type=PreflightError,
    )

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
    CheckBuilder.append(
        checks,
        warnings,
        errors,
        check_id,
        status,
        reason,
        _json_safe(evidence),
    )


def _status_from_checks(checks: Sequence[Mapping[str, Any]]) -> tuple[str, str]:
    values = {str(item.get("status", "pass")) for item in checks}
    if "fail" in values:
        return "red", "FAIL"
    if "warn" in values:
        return "yellow", "WARN"
    return "green", "PASS"


def _prepare_outputs(output: Any, force: bool) -> tuple[dict[str, Path] | None, list[Path]]:
    """Compatibility seam for callers/tests that patch the old helper."""

    return prepare_outputs(
        output,
        force,
        resolve_output_path=_resolve_output_path,
        error_type=PreflightError,
    )


def _atomic_text(path: Path, text: str) -> None:
    atomic_text(path, text)


def _atomic_npz(path: Path, arrays: Mapping[str, Any]) -> None:
    atomic_npz(path, arrays)


def _markdown_report(report: Mapping[str, Any]) -> str:
    return markdown_report(report)


def _read_one_frame(
    ref: FrameRef,
    package: Path,
    frame_override: int | None,
    dataset_override: str | None,
    records: list[dict[str, Any]],
    external_roots: Sequence[str | os.PathLike[str] | Path] | None = None,
) -> tuple[np.ndarray, Any, Path]:
    return read_one_frame(
        ref,
        package,
        frame_override,
        dataset_override,
        records,
        external_roots,
        resolve_path=_resolve_path,
        read_file_record=_read_file_record,
        load_image=load_image,
        error_type=PreflightError,
    )


def _frame_record(
    index: int,
    ref: FrameRef,
    source: Path,
    loaded: Any,
    array: np.ndarray,
    package: Path,
    image_metadata: list[Mapping[str, Any]],
) -> dict[str, Any]:
    return frame_record(
        index,
        ref,
        source,
        loaded,
        array,
        package,
        image_metadata,
        image_summary=_image_summary,
        display_path=_display_path,
    )



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
    external_roots: Sequence[str | os.PathLike[str] | Path] | None = None,
) -> dict[str, Any]:
    """Run a read-only preflight and optionally write its three artifacts.

    Relative input paths (including manifest rows, context paths, PONI, mask,
    and image globs) are resolved below ``package``.  A relative ``output`` is
    interpreted relative to the caller's current working directory, while an
    absolute output remains unchanged.  The function raises
    :class:`PreflightError`/``ValueError`` for blocking input errors and
    returns a strict-JSON-compatible dictionary for green or yellow runs.
    """

    artifact_writer = PreflightArtifactWriter(
        prepare_outputs=_prepare_outputs,
        atomic_text=_atomic_text,
        atomic_npz=_atomic_npz,
        markdown_report=_markdown_report,
    )
    resolved_inputs = resolve_inputs(
        package,
        manifest=manifest,
        poni=poni,
        mask=mask,
        context=context,
        image_glob=image_glob,
        frame=frame,
        dataset=dataset,
        mask_frame=mask_frame,
        mask_dataset=mask_dataset,
        q_window=q_window,
        mask_convention=mask_convention,
        output=output,
        external_roots=external_roots,
        require_package=_require_package,
        load_context=_load_context,
        context_settings=_context_settings,
        resolve_manifest=_resolve_manifest,
        glob_inputs=_glob_inputs,
        resolve_path=_resolve_path,
        resolve_output_path=_resolve_output_path,
        build_frame_refs=build_frame_refs,
        normalise_mask_convention=_normalise_mask_convention,
        error_type=PreflightError,
    )
    preflight_context = resolved_inputs.context
    hash_groups = resolved_inputs.hash_groups

    read_state = read_preflight_inputs(
        preflight_context,
        frame=frame,
        dataset=dataset,
        mask_frame=mask_frame,
        mask_dataset=mask_dataset,
        hash_groups=hash_groups,
        read_one_frame=_read_one_frame,
        frame_record=_frame_record,
        load_mask=_load_mask,
        build_qmap=_build_qmap,
        build_analysis_domain=build_analysis_domain,
        preflight_error=PreflightError,
        analysis_domain_error=AnalysisDomainError,
    )

    check_state = build_preflight_checks(
        context=preflight_context,
        read_state=read_state,
        hash_groups=hash_groups,
        configured_external_roots=preflight_context.external_roots,
        correction_state=correction_state,
        uncertainty_state=uncertainty_state,
        image_metadata=read_state.image_metadata,
        output=output,
        add_check=_add_check,
        display_path=_display_path,
        nested_mapping=_nested_mapping,
        first_value=_first_value,
        resolve_correction_value=_resolve_correction_value,
        correction_check=_correction_check,
        state_value=_state_value,
        uncertainty_check=_uncertainty_check,
        uncertainty_provenance=_uncertainty_provenance,
        header_geometry_check=_header_geometry_check,
        status_from_checks=_status_from_checks,
    )
    output_paths, overwritten_paths = artifact_writer.prepare(output, force)
    report_result = build_preflight_report_from_stages(
        context=preflight_context,
        read_state=read_state,
        check_state=check_state,
        hash_groups=hash_groups,
        frame=frame,
        dataset=dataset,
        mask_frame=mask_frame,
        mask_dataset=mask_dataset,
        correction_state=correction_state,
        uncertainty_state=uncertainty_state,
        output_paths=output_paths,
        overwritten_paths=overwritten_paths,
        force=force,
        schema_source=Path(__file__).with_name("validation.py"),
        code_source=Path(__file__),
        result_schema_version=RESULT_SCHEMA_VERSION,
        preflight_schema_version=PREFLIGHT_SCHEMA_VERSION,
        helpers=PreflightReportHelpers(
            json_safe=_json_safe,
            display_path=_display_path,
            first_value=_first_value,
            nested_mapping=_nested_mapping,
            canonical_state=_canonical_state,
            structured_check=_structured_check,
            status_item=_status_item,
            dependency_versions=_dependency_versions,
            git_commit=_git_commit,
            inline_hash=_inline_hash,
            sha256_file=_sha256_file,
            source_tree_sha256=_source_tree_sha256,
            validate_result_schema=validate_result_schema,
        ),
    )
    safe_report = report_result.report
    q_keys = report_result.q_keys

    if output_paths is not None:
        q_arrays = {
            q_keys["qx"]: np.asarray(read_state.qmap["qx"], dtype=float),
            q_keys["qy"]: np.asarray(read_state.qmap["qy"], dtype=float),
            q_keys["q"]: np.asarray(read_state.qmap["q"], dtype=float),
            q_keys["chi"]: np.arctan2(
                np.asarray(read_state.qmap["qy"], dtype=float),
                np.asarray(read_state.qmap["qx"], dtype=float),
            ),
        }
        artifact_writer.write(
            output_paths,
            safe_report,
            {
                **q_arrays,
                "finite_mask": np.asarray(read_state.reference_domain.finite_mask, dtype=bool),
                "detector_valid_mask": np.asarray(
                    read_state.reference_domain.detector_valid_mask, dtype=bool
                ),
                "external_valid_mask": np.asarray(
                    read_state.reference_domain.external_valid_mask, dtype=bool
                ),
                "q_window_mask": np.asarray(read_state.reference_domain.q_window_mask, dtype=bool),
                "roi_exclusion_mask": np.asarray(
                    read_state.reference_domain.roi_exclusion_mask, dtype=bool
                ),
                "weight_valid_mask": np.asarray(
                    read_state.reference_domain.weight_valid_mask, dtype=bool
                ),
                "fit_valid_mask": np.asarray(read_state.reference_domain.fit_valid_mask, dtype=bool),
                "sampled_valid_mask": np.asarray(
                    read_state.reference_domain.sampled_valid_mask, dtype=bool
                ),
                "frame_id": np.asarray(preflight_context.refs[0].id),
            },
        )
        return safe_report
    return safe_report


__all__ = ["PREFLIGHT_SCHEMA_VERSION", "PreflightError", "run_preflight"]
