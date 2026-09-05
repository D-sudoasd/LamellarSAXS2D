"""Strict report assembly for the preflight contract.

The builder receives explicit read/check evidence and produces the stable
result schema.  It does not resolve paths or read detector arrays.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any
from datetime import datetime
import os
import uuid

import numpy as np

from .preflight_checks import PreflightCheckState
from .preflight_context import PreflightContext
from .preflight_io import PreflightReadState


@dataclass(frozen=True)
class PreflightReportHelpers:
    """Small callback bundle retaining the facade's established seams."""

    json_safe: Callable[[Any], Any]
    display_path: Callable[[Path, Path], str]
    first_value: Callable[..., Any]
    nested_mapping: Callable[[Mapping[str, Any], str], Mapping[str, Any]]
    canonical_state: Callable[[Any, str], str]
    structured_check: Callable[[Mapping[str, Any]], dict[str, Any]]
    status_item: Callable[[Mapping[str, Any]], dict[str, Any]]
    dependency_versions: Callable[[], Mapping[str, Any]]
    git_commit: Callable[[], str | None]
    inline_hash: Callable[[Any], str]
    sha256_file: Callable[[Path], str]
    source_tree_sha256: Callable[[], str]
    validate_result_schema: Callable[[Mapping[str, Any]], None]


@dataclass(frozen=True)
class PreflightReportInputs:
    """Explicit evidence required to assemble one preflight result."""

    package_root: Path
    context_value: Mapping[str, Any]
    context_source: Path | None
    preflight_context: Any
    refs: Sequence[Any]
    selected_manifest: Any
    selected_glob: Any
    selected_poni: Any
    selected_mask: Any
    configured_external_roots: Sequence[Any] | None
    manifest_source: Path | None
    manifest_quality: Mapping[str, Any] | None
    frames: Sequence[Mapping[str, Any]]
    first_shape: tuple[int, ...]
    mask_array: np.ndarray | None
    mask_source: Path | None
    mask_frame: int | None
    mask_dataset: str | None
    qmap: Mapping[str, Any]
    domain_q_window: Any
    reference_domain: Any
    domain_records: Sequence[Mapping[str, Any]]
    checks: Sequence[Mapping[str, Any]]
    warnings: Sequence[str]
    errors: Sequence[str]
    status_color: str
    scientific_status: str
    exit_code: int
    convention_for_report: str | None
    finite_q_for_report: np.ndarray
    correction_value: Any
    correction_check_value: Any
    uncertainty_value: Any
    uncertainty_provenance: Mapping[str, Any]
    hash_groups: Mapping[str, list[dict[str, Any]]]
    frame: int | None
    dataset: str | None
    correction_state: Any
    uncertainty_state: Any
    output_paths: Mapping[str, Path] | None
    overwritten_paths: Sequence[Path]
    force: bool
    schema_source: Path
    code_source: Path
    result_schema_version: str
    preflight_schema_version: str
    helpers: PreflightReportHelpers


@dataclass(frozen=True)
class PreflightReportResult:
    """Report plus the array-key maps needed by the publication stage."""

    report: dict[str, Any]
    q_keys: dict[str, str]
    mask_keys: dict[str, str]


def build_preflight_report(inputs: PreflightReportInputs) -> PreflightReportResult:
    """Assemble the strict report without performing I/O or validation side effects."""

    _json_safe = inputs.helpers.json_safe
    _display_path = inputs.helpers.display_path
    _first_value = inputs.helpers.first_value
    _nested_mapping = inputs.helpers.nested_mapping
    _canonical_state = inputs.helpers.canonical_state
    _structured_check = inputs.helpers.structured_check
    _status_item = inputs.helpers.status_item
    _dependency_versions = inputs.helpers.dependency_versions
    _git_commit = inputs.helpers.git_commit
    _inline_hash = inputs.helpers.inline_hash
    _sha256_file = inputs.helpers.sha256_file
    _source_tree_sha256 = inputs.helpers.source_tree_sha256
    validate_result_schema = inputs.helpers.validate_result_schema
    RESULT_SCHEMA_VERSION = inputs.result_schema_version
    PREFLIGHT_SCHEMA_VERSION = inputs.preflight_schema_version
    package_root = inputs.package_root
    context_value = inputs.context_value
    context_source = inputs.context_source
    preflight_context = inputs.preflight_context
    refs = inputs.refs
    selected_manifest = inputs.selected_manifest
    selected_glob = inputs.selected_glob
    selected_poni = inputs.selected_poni
    selected_mask = inputs.selected_mask
    configured_external_roots = inputs.configured_external_roots
    manifest_source = inputs.manifest_source
    manifest_quality = inputs.manifest_quality
    frames = inputs.frames
    first_shape = inputs.first_shape
    mask_array = inputs.mask_array
    mask_source = inputs.mask_source
    mask_frame = inputs.mask_frame
    mask_dataset = inputs.mask_dataset
    qmap = inputs.qmap
    domain_q_window = inputs.domain_q_window
    reference_domain = inputs.reference_domain
    domain_records = inputs.domain_records
    checks = inputs.checks
    warnings = inputs.warnings
    errors = inputs.errors
    status_color = inputs.status_color
    scientific_status = inputs.scientific_status
    exit_code = inputs.exit_code
    convention_for_report = inputs.convention_for_report
    finite_q_for_report = inputs.finite_q_for_report
    correction_value = inputs.correction_value
    correction_check_value = inputs.correction_check_value
    uncertainty_value = inputs.uncertainty_value
    uncertainty_provenance = inputs.uncertainty_provenance
    hash_groups = inputs.hash_groups
    frame = inputs.frame
    dataset = inputs.dataset
    output_paths = inputs.output_paths
    overwritten_paths = inputs.overwritten_paths
    force = inputs.force

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
        for record in entries:
            provenance_record = {
                "role": provenance_role,
                "path": record["path"],
                "sha256": record["sha256_after"],
                "before": record["sha256_before"],
                "after": record["sha256_after"],
                "unchanged": bool(record["unchanged"]),
            }
            for key in ("read_status", "read_error", "post_read_hash_error"):
                if key in record:
                    provenance_record[key] = record[key]
            provenance_files.append(provenance_record)

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

    first_ref = preflight_context.refs[0]
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
    schema_source = inputs.schema_source
    code_sha256 = _sha256_file(inputs.code_source)
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
            "declared_values": list(uncertainty_provenance.get("declared_values", [])),
            "missing_declared_files": list(
                uncertainty_provenance.get("missing_declared_files", [])
            ),
            "file_inventory": list(uncertainty_provenance.get("file_inventory", [])),
            "datasets": list(uncertainty_provenance.get("datasets", [])),
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
                    hash_groups["poni"][0]["path"]
                    if isinstance(selected_poni, (str, os.PathLike))
                    and hash_groups["poni"]
                    else "in-memory"
                    if selected_poni is not None
                    else None
                ),
                "mask": (
                    hash_groups["mask"][0]["path"]
                    if isinstance(selected_mask, (str, os.PathLike))
                    and hash_groups["mask"]
                    else "in-memory"
                    if selected_mask is not None
                    else None
                ),
                "context": (
                    _display_path(context_source, package_root)
                    if context_source is not None
                    else None
                ),
                "external_roots": [
                    str(
                        (
                            Path(item).expanduser()
                            if Path(item).expanduser().is_absolute()
                            else package_root / Path(item).expanduser()
                        ).resolve(strict=False)
                    )
                    for item in (configured_external_roots or ())
                ],
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
    return PreflightReportResult(
        report=safe_report,
        q_keys=q_keys,
        mask_keys=mask_keys,
    )


def build_preflight_report_from_stages(
    *,
    context: PreflightContext,
    read_state: PreflightReadState,
    check_state: PreflightCheckState,
    hash_groups: Mapping[str, list[dict[str, Any]]],
    frame: int | None,
    dataset: str | None,
    mask_frame: int | None,
    mask_dataset: str | None,
    correction_state: Any,
    uncertainty_state: Any,
    output_paths: Mapping[str, Path] | None,
    overwritten_paths: Sequence[Path],
    force: bool,
    schema_source: Path,
    code_source: Path,
    result_schema_version: str,
    preflight_schema_version: str,
    helpers: PreflightReportHelpers,
) -> PreflightReportResult:
    """Bridge the three stage values into the explicit report input contract."""

    inputs = PreflightReportInputs(
        package_root=context.package_root,
        context_value=context.context_value,
        context_source=context.context_source,
        preflight_context=context,
        refs=context.refs,
        selected_manifest=context.selected_manifest,
        selected_glob=context.selected_glob,
        selected_poni=context.selected_poni,
        selected_mask=context.selected_mask,
        configured_external_roots=context.external_roots,
        manifest_source=context.manifest_source,
        manifest_quality=context.manifest_quality,
        frames=read_state.frames,
        first_shape=read_state.first_shape,
        mask_array=read_state.mask_array,
        mask_source=read_state.mask_source,
        mask_frame=mask_frame,
        mask_dataset=mask_dataset,
        qmap=read_state.qmap,
        domain_q_window=read_state.domain_q_window,
        reference_domain=read_state.reference_domain,
        domain_records=read_state.domain_records,
        checks=check_state.checks,
        warnings=check_state.warnings,
        errors=check_state.errors,
        status_color=check_state.status_color,
        scientific_status=check_state.scientific_status,
        exit_code=check_state.exit_code,
        convention_for_report=check_state.convention_for_report,
        finite_q_for_report=check_state.finite_q_for_report,
        correction_value=check_state.correction_value,
        correction_check_value=check_state.correction_check_value,
        uncertainty_value=check_state.uncertainty_value,
        uncertainty_provenance=check_state.uncertainty_provenance,
        hash_groups=hash_groups,
        frame=frame,
        dataset=dataset,
        correction_state=correction_state,
        uncertainty_state=uncertainty_state,
        output_paths=output_paths,
        overwritten_paths=overwritten_paths,
        force=force,
        schema_source=schema_source,
        code_source=code_source,
        result_schema_version=result_schema_version,
        preflight_schema_version=preflight_schema_version,
        helpers=helpers,
    )
    return build_preflight_report(inputs)


__all__ = [
    "PreflightReportHelpers",
    "PreflightReportInputs",
    "PreflightReportResult",
    "build_preflight_report",
    "build_preflight_report_from_stages",
]
