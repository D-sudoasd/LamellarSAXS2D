"""Check accumulation for the preflight orchestration facade."""

from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from .preflight_context import PreflightContext
from .preflight_io import PreflightReadState

@dataclass
class CheckBuilder:
    """Build ordered checks while retaining legacy warning/error lists."""

    checks: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def add(self, check_id: str, status: str, reason: str, evidence: Any) -> None:
        check = {
            "id": check_id,
            "status": status,
            "reason": str(reason),
            "evidence": evidence,
        }
        self.checks.append(check)
        if status == "warn":
            self.warnings.append(str(reason))
        elif status == "fail":
            self.errors.append(str(reason))

    @staticmethod
    def append(
        checks: list[dict[str, Any]],
        warnings: list[str],
        errors: list[str],
        check_id: str,
        status: str,
        reason: str,
        evidence: Any,
    ) -> None:
        """Compatibility adapter for callers/tests using the old helper."""

        builder = CheckBuilder(checks=checks, warnings=warnings, errors=errors)
        builder.add(check_id, status, reason, evidence)


@dataclass(frozen=True)
class PreflightCheckState:
    """Ordered scientific and structural checks plus their derived status."""

    checks: list[dict[str, Any]]
    warnings: list[str]
    errors: list[str]
    convention_for_report: str | None
    finite_q_for_report: np.ndarray
    correction_value: Any
    correction_check_value: Any
    uncertainty_value: Any
    uncertainty_label: str
    uncertainty_provenance: dict[str, Any]
    status_color: str
    scientific_status: str
    exit_code: int


def build_preflight_checks(
    *,
    context: PreflightContext,
    read_state: PreflightReadState,
    hash_groups: Mapping[str, list[dict[str, Any]]],
    configured_external_roots: Sequence[Any] | None,
    correction_state: Any,
    uncertainty_state: Any,
    image_metadata: Sequence[Mapping[str, Any]],
    output: Any,
    add_check: Callable[..., None],
    display_path: Callable[..., str],
    nested_mapping: Callable[..., Mapping[str, Any]],
    first_value: Callable[..., Any],
    resolve_correction_value: Callable[..., tuple[Any, Any]],
    correction_check: Callable[..., tuple[str, str, dict[str, Any]]],
    state_value: Callable[..., Any],
    uncertainty_check: Callable[..., tuple[str, str, dict[str, Any]]],
    uncertainty_provenance: Callable[..., dict[str, Any]],
    header_geometry_check: Callable[..., tuple[str, str, dict[str, Any]]],
    status_from_checks: Callable[..., tuple[str, str]],
) -> PreflightCheckState:
    """Build checks in the historical order without reading detector arrays."""

    package_root = context.package_root
    refs = context.refs
    selected_manifest = context.selected_manifest
    manifest_source = context.manifest_source
    manifest_quality = context.manifest_quality
    selected_convention = context.selected_convention
    context_value = context.context_value
    frames = read_state.frames
    first_shape = read_state.first_shape
    qmap = read_state.qmap
    mask_array = read_state.mask_array
    mask_source = read_state.mask_source
    external_mask = read_state.external_mask
    valid_mask = read_state.valid_mask
    reference_domain = read_state.reference_domain
    domain_records = read_state.domain_records
    _add_check = add_check
    _display_path = display_path
    _nested_mapping = nested_mapping
    _first_value = first_value
    _resolve_correction_value = resolve_correction_value
    _correction_check = correction_check
    _state_value = state_value
    _uncertainty_check = uncertainty_check
    _uncertainty_provenance = uncertainty_provenance
    _header_geometry_check = header_geometry_check
    _status_from_checks = status_from_checks
    convention_for_report = selected_convention if mask_array is not None else None
    finite_q_for_report = np.asarray(qmap["q"])[np.isfinite(qmap["q"])]

    check_builder = CheckBuilder()
    checks = check_builder.checks
    warnings = check_builder.warnings
    errors = check_builder.errors
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
        configured_external_roots,
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
    return PreflightCheckState(
        checks=checks,
        warnings=warnings,
        errors=errors,
        convention_for_report=convention_for_report,
        finite_q_for_report=finite_q_for_report,
        correction_value=correction_value,
        correction_check_value=correction_check_value,
        uncertainty_value=uncertainty_value,
        uncertainty_label=uncertainty_label,
        uncertainty_provenance=uncertainty_provenance,
        status_color=status_color,
        scientific_status=scientific_status,
        exit_code=exit_code,
    )


__all__ = ["CheckBuilder", "PreflightCheckState", "build_preflight_checks"]
