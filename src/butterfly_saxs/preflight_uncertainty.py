"""Correction and uncertainty provenance checks used by preflight."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import os
from pathlib import Path
from typing import Any

def state_value(explicit: Any, context: Mapping[str, Any], key: str, *, first_value: Callable[..., Any]) -> Any:
    if explicit is not None:
        return explicit
    if key in context:
        return context[key]
    source = context.get("source_intensity")
    if isinstance(source, Mapping):
        if key == "uncertainty_state":
            return first_value(source.get("uncertainty_state"), source.get("uncertainty_status"))
        if key == "correction_state":
            return source
    return None

def resolve_correction_value(explicit: Any, context: Mapping[str, Any], *, first_value: Callable[..., Any]) -> tuple[Any, Any]:
    """Resolve correction status separately from uncertainty metadata."""

    if explicit is not None:
        return explicit, explicit
    if "correction_state" in context:
        value = context["correction_state"]
        return value, value
    source = context.get("source_intensity")
    if not isinstance(source, Mapping):
        return None, None
    declared = first_value(source.get("correction_state"), source.get("correction_status"))
    if declared is not None:
        return declared, source
    if "already_applied" in source or "not_burned_into_2d_values" in source:
        details = dict(source)
        details["status"] = "external_recipe_declared"
        return "external_recipe_declared", details
    # ``uncertainty_status`` belongs to the independent uncertainty contract;
    # it must never be interpreted as a correction state.
    return None, None

def state_status(value: Any, kind: str, *, json_safe: Callable[[Any], Any], first_value: Callable[..., Any]) -> tuple[str, str, dict[str, Any]]:
    safe = json_safe(value)
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
        status = str(first_value(*(value.get(name) for name in status_names), "unknown")).casefold()
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

def correction_check(value: Any, *, state_status: Callable[..., tuple[str, str, dict[str, Any]]], first_value: Callable[..., Any]) -> tuple[str, str, dict[str, Any]]:
    status, reason, evidence = state_status(value, "correction_state")
    not_burned: list[str] = []
    if isinstance(value, Mapping):
        raw = first_value(
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

def uncertainty_check(value: Any, *, state_status: Callable[..., tuple[str, str, dict[str, Any]]], first_value: Callable[..., Any]) -> tuple[str, str, dict[str, Any]]:
    status, reason, evidence = state_status(value, "uncertainty_state")
    if isinstance(value, Mapping):
        raw = first_value(value.get("status"), value.get("state"), value.get("uncertainty_status"))
        if raw is not None:
            status, reason, evidence = state_status(str(raw), "uncertainty_state")
    return status, reason, evidence

def uncertainty_provenance(
    image_metadata: Sequence[Mapping[str, Any]],
    package: Path,
    hash_records: list[dict[str, Any]],
    external_roots: Sequence[str | os.PathLike[str] | Path] | None = None,
    *,
    resolve_path: Callable[..., Path],
    display_path: Callable[[Path, Path], str],
    read_file_record: Callable[..., Any],
    json_safe: Callable[[Any], Any],
    error_type: type[Exception] = ValueError,
) -> dict[str, Any]:
    """Inventory every declared uncertainty file and its datasets.

    Header values are treated as package-relative references.  An absolute or
    outside-package value is opened only when its root was explicitly
    authorized by the caller.  Every resolved file is hashed before and after
    inventory, even when its HDF5 group is malformed, so a ``complete`` state
    cannot be inferred from one representative file.
    """

    declarations: list[dict[str, Any]] = []
    for metadata in image_metadata:
        header = metadata.get("header")
        if not isinstance(header, Mapping):
            continue
        normalized = {str(key).casefold(): value for key, value in header.items()}
        raw_source = normalized.get("uncertaintyhdf5")
        if raw_source is None:
            continue
        declared = str(raw_source).strip()
        if not declared:
            continue
        try:
            resolved = resolve_path(
                package,
                declared,
                external_roots=external_roots,
                label="uncertainty file",
            )
        except error_type as exc:
            declarations.append(
                {
                    "declared": declared,
                    "path": None,
                    "status": "unauthorized_reference",
                    "reason": str(exc),
                    "datasets": [],
                }
            )
            continue
        existing = next(
            (item for item in declarations if item.get("path") == display_path(resolved, package)),
            None,
        )
        if existing is not None:
            existing.setdefault("declared_values", []).append(declared)
            continue
        if not resolved.is_file():
            declarations.append(
                {
                    "declared": declared,
                    "path": display_path(resolved, package),
                    "status": "missing",
                    "reason": "declared uncertainty file does not exist",
                    "datasets": [],
                }
            )
            continue

        record: dict[str, Any] = {
            "declared": declared,
            "declared_values": [declared],
            "path": display_path(resolved, package),
            "status": "unknown",
            "reason": None,
            "datasets": [],
        }
        try:
            def read_reference_schema() -> tuple[str, list[dict[str, Any]]]:
                try:
                    import h5py
                except ImportError:
                    return "h5py_unavailable", []
                rows: list[dict[str, Any]] = []
                with h5py.File(resolved, "r") as handle:
                    group_path = "entry/data/uncertainty"
                    group = handle.get(group_path)
                    if group is None:
                        return "uncertainty_group_missing", rows
                    group_unit = group.attrs.get("units", group.attrs.get("unit"))
                    for name, dataset in group.items():
                        if not hasattr(dataset, "shape"):
                            continue
                        rows.append(
                            {
                                "dataset": f"{group_path}/{name}",
                                "group": group_path,
                                "component": str(name),
                                "shape": [int(item) for item in dataset.shape],
                                "dtype": str(dataset.dtype),
                                "unit": json_safe(
                                    dataset.attrs.get(
                                        "units",
                                        dataset.attrs.get("unit", group_unit),
                                    )
                                ),
                            }
                        )
                return ("reference_schema_read", rows) if rows else (
                    "uncertainty_dataset_missing",
                    rows,
                )

            status, rows = read_file_record(
                resolved,
                package,
                read_reference_schema,
                hash_records,
            )
            record["status"] = status
            record["datasets"] = rows
            if status != "reference_schema_read":
                record["reason"] = status
        except (OSError, ValueError, RuntimeError) as exc:
            # read_file_record has already raised for a read-time hash
            # change.  Other HDF5 failures remain explicit and the file
            # still has a before/after record when possible.
            record["status"] = "reference_hdf5_unreadable"
            record["reason"] = f"{type(exc).__name__}: {exc}"
        declarations.append(record)

    declared_values = [str(item.get("declared", "")) for item in declarations]
    resolved_files = [
        str(item["path"])
        for item in declarations
        if item.get("path") is not None and item.get("status") not in {"missing", "unauthorized_reference"}
    ]
    datasets = [
        {**dict(dataset), "file": item.get("path")}
        for item in declarations
        for dataset in item.get("datasets", [])
    ]
    valid_files = [item for item in declarations if item.get("status") == "reference_schema_read"]
    all_files_valid = bool(declarations) and len(valid_files) == len(declarations)
    all_datasets_have_units = bool(datasets) and all(
        item.get("unit") not in (None, "", "unknown") for item in datasets
    )
    if not declarations:
        inventory_status = "not_declared"
    elif all_files_valid and all_datasets_have_units:
        inventory_status = "reference_schema_read"
    elif any(item.get("status") in {"missing", "unauthorized_reference"} for item in declarations):
        inventory_status = "partial_missing"
    else:
        inventory_status = "partial_invalid"
    return {
        "declared_source_kind": "per_frame_hdf5" if declarations else None,
        "declared_file_count": len(declarations),
        "resolved_file_count": len(resolved_files),
        "files": resolved_files,
        "declared_values": declared_values,
        "missing_declared_files": [
            item.get("declared")
            for item in declarations
            if item.get("status") in {"missing", "unauthorized_reference"}
        ],
        "file_inventory": declarations,
        "reference_schema_file": resolved_files[0] if resolved_files else None,
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


__all__ = [
    "correction_check",
    "resolve_correction_value",
    "state_status",
    "state_value",
    "uncertainty_check",
    "uncertainty_provenance",
]
