"""Source-to-observable adaptation helpers for the lamellar schematic."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .lamellar_models import LamellarSettings
from .lamellar_utils import (
    _as_mapping,
    _copy_metadata,
    _finite,
    _normalise_q_unit,
    _q_scale_to_nm,
    _read,
)

_MISSING = object()
_SCENE_FLAGS = (
    "apparent_geometry_only",
    "nonunique_inverse_problem",
    "lamellar_schematic_only",
)


def _source_mapping(source: Any) -> Mapping[str, Any]:
    mapping = _as_mapping(source)
    if mapping is None:
        return {}
    return mapping


def _source_identity(source: Any, root: Mapping[str, Any]) -> dict[str, Any]:
    identity = _read(root, ("source_identity",), {})
    result = _copy_metadata(identity) if isinstance(identity, Mapping) else {}
    if not isinstance(result, dict):
        result = {}
    metadata = _read(root, ("metadata",), {})
    if isinstance(metadata, Mapping):
        for key, value in metadata.items():
            if key in {"observed", "image", "data", "qmap", "qx", "qy"}:
                continue
            if key in {
                "source",
                "path",
                "frame",
                "frame_selector",
                "dataset",
                "dataset_selector",
                "id",
                "frame_id",
                "time",
                "timestamp",
                "source_parent_token",
            }:
                result.setdefault(str(key), _copy_metadata(value))
    for key in (
        "source",
        "path",
        "frame",
        "frame_selector",
        "dataset",
        "dataset_selector",
        "id",
        "frame_id",
        "time",
        "timestamp",
        "source_parent_token",
    ):
        value = _read(root, (key,), _MISSING)
        if value is not _MISSING and value is not None:
            result.setdefault(key, _copy_metadata(value))
    if "source" not in result and "path" in result:
        result["source"] = result["path"]
    if "frame" not in result and "frame_selector" in result:
        result["frame"] = result["frame_selector"]
    if "dataset" not in result and "dataset_selector" in result:
        result["dataset"] = result["dataset_selector"]
    return result


def _q_unit_from_source(
    root: Mapping[str, Any], peaks: Sequence[Mapping[str, Any]] = ()
) -> str:
    direct = _read(root, ("q_unit", "source_q_unit"), None)
    if direct not in (None, "") and _normalise_q_unit(direct) not in {
        "unknown",
        "none",
        "",
    }:
        return str(direct)
    qmap = _read(root, ("qmap", "geometry"), None)
    direct = _read(qmap, ("q_unit", "source_q_unit", "unit"), None)
    if direct not in (None, "") and _normalise_q_unit(direct) not in {
        "unknown",
        "none",
        "",
    }:
        return str(direct)
    observables = _read(root, ("observables", "measurements"), None)
    direct = _read(observables, ("q_unit", "source_q_unit"), None)
    if direct not in (None, "") and _normalise_q_unit(direct) not in {
        "unknown",
        "none",
        "",
    }:
        return str(direct)
    ellipse = _read(root, ("ellipse_fit", "ellipse"), None)
    direct = _read(ellipse, ("q_unit", "source_q_unit"), None)
    if direct not in (None, "") and _normalise_q_unit(direct) not in {
        "unknown",
        "none",
        "",
    }:
        return str(direct)
    for peak in peaks:
        direct = _read(peak, ("q_unit", "source_q_unit"), None)
        if direct not in (None, ""):
            return str(direct)
    return "unknown"


def _execution_failure(root: Mapping[str, Any]) -> str | None:
    """Return a failed execution status from the source/result envelope."""

    failure_tokens = {
        "failed",
        "failure",
        "cancelled",
        "canceled",
        "skipped",
        "error",
        "aborted",
        "incomplete",
        "invalid",
        "not_run",
    }
    containers = [root]
    for name in ("result", "summary", "metadata", "metrics"):
        candidate = _read(root, (name,), None)
        if isinstance(candidate, Mapping):
            containers.append(candidate)
    for container in containers:
        explicit = _read(container, ("execution_success",), _MISSING)
        if explicit is False:
            return "execution_success=false"
        status = _read(
            container,
            ("execution_status", "measurement_status", "status", "state"),
            _MISSING,
        )
        if isinstance(status, Mapping):
            status = _read(status, ("status", "state", "scientific_status"), _MISSING)
        if (
            status is not _MISSING
            and str(status or "").strip().casefold() in failure_tokens
        ):
            return str(status)
    return None


def _sequence(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, Mapping) or isinstance(value, (str, bytes)):
        return [value]
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return [value.item()]
    try:
        return list(value)
    except TypeError:
        return [value]


def _raw_radial_peak_sequence(root: Mapping[str, Any]) -> list[Any]:
    direct: list[Any] = []
    nested: list[Any] = []
    value = _read(root, ("lobe_radial_peaks", "radial_peaks"), _MISSING)
    if value is not _MISSING and value is not None:
        direct = _sequence(value)
    observables = _read(root, ("observables", "measurements"), None)
    value = _read(observables, ("lobe_radial_peaks", "radial_peaks"), _MISSING)
    if value is not _MISSING and value is not None:
        nested = _sequence(value)
    if direct:
        recognized = any(
            _read(
                item,
                ("angle", "angle_deg", "q_star", "q", "qx", "qy"),
                _read(
                    _read(item, ("metadata",), {}),
                    ("angle", "angle_deg", "q_star", "q", "qx", "qy"),
                    _MISSING,
                ),
            )
            is not _MISSING
            for item in direct
        )
        if recognized or not nested:
            return direct
    return nested or direct


def _angle_deg(peak: Any) -> float | None:
    metadata = _read(peak, ("metadata",), {})
    value = _read(
        peak,
        ("angle_deg", "azimuth_deg", "phi_deg"),
        _read(metadata, ("angle_deg", "azimuth_deg", "phi_deg"), _MISSING),
    )
    if value is not _MISSING:
        return _finite(value)
    value = _read(
        peak,
        ("angle", "azimuth", "phi"),
        _read(metadata, ("angle", "azimuth", "phi"), _MISSING),
    )
    if value is _MISSING:
        return None
    unit = str(
        _read(
            peak,
            ("angle_unit", "azimuth_unit"),
            _read(metadata, ("angle_unit", "azimuth_unit"), "rad"),
        )
        or "rad"
    ).lower()
    number = _finite(value)
    if number is None:
        return None
    if "deg" in unit or "degree" in unit:
        return number
    return float(np.degrees(number))


def _branch_value(peak: Any) -> int | None:
    metadata = _read(peak, ("metadata",), {})
    value = _read(
        peak,
        ("branch_id", "branch", "component"),
        _read(metadata, ("branch_id", "branch", "component"), _MISSING),
    )
    if value is _MISSING:
        return None
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number


def _radial_peaks(root: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = _raw_radial_peak_sequence(root)
    result: list[dict[str, Any]] = []
    source_unit = _q_unit_from_source(root)
    for index, peak in enumerate(raw):
        if not isinstance(peak, Mapping) and not hasattr(peak, "__dict__"):
            continue
        metadata = _read(peak, ("metadata",), {})
        method_value = _read(
            peak,
            ("method", "source", "measurement_kind", "source_method"),
            _read(
                metadata,
                ("method", "source", "measurement_kind", "source_method"),
                _MISSING,
            ),
        )
        flags_value = _read(peak, ("flags",), _read(metadata, ("flags",), ()))
        method = str(method_value or "").lower()
        flags = tuple(str(item).lower() for item in (flags_value or ()))
        radial_evidence = any(
            "radial_peak" in token or "radial_lobe" in token
            for token in (method, *flags)
        )
        if not method and radial_evidence:
            method = "radial_peak"
        if (
            "azimuthal_peak" in method
            or "annulus" in method
            or any("annulus" in item for item in flags)
        ):
            continue
        valid_value = _read(
            peak, ("valid", "accepted"), _read(metadata, ("valid", "accepted"), True)
        )
        if valid_value is not None and not bool(valid_value):
            continue
        angle = _angle_deg(peak)
        if angle is None and radial_evidence:
            qx = _finite(
                _read(peak, ("qx", "q_x"), _read(metadata, ("qx", "q_x"), _MISSING))
            )
            qy = _finite(
                _read(peak, ("qy", "q_y"), _read(metadata, ("qy", "q_y"), _MISSING))
            )
            if (
                qx is not None
                and qy is not None
                and np.isfinite(qx)
                and np.isfinite(qy)
                and (qx != 0.0 or qy != 0.0)
            ):
                angle = float(np.degrees(np.arctan2(qy, qx)))
        if angle is None:
            continue
        q_unit = str(
            _read(
                peak,
                ("q_unit", "source_q_unit"),
                _read(metadata, ("q_unit", "source_q_unit"), source_unit),
            )
            or source_unit
        )
        q_star_nm = _finite(
            _read(
                peak, ("q_star_nm_inv",), _read(metadata, ("q_star_nm_inv",), _MISSING)
            )
        )
        q_star_a = _finite(
            _read(
                peak,
                ("q_star_Ainv", "q_star_A_inv"),
                _read(metadata, ("q_star_Ainv", "q_star_A_inv"), _MISSING),
            )
        )
        if q_star_nm is None and q_star_a is not None and q_star_a > 0.0:
            q_star_nm = q_star_a * 10.0
            q_unit = "Å^-1"
        raw_q = _read(
            peak,
            ("q_star", "q_position", "q"),
            _read(metadata, ("q_star", "q_position", "q"), _MISSING),
        )
        if raw_q is _MISSING and radial_evidence:
            qx = _finite(
                _read(peak, ("qx", "q_x"), _read(metadata, ("qx", "q_x"), _MISSING))
            )
            qy = _finite(
                _read(peak, ("qy", "q_y"), _read(metadata, ("qy", "q_y"), _MISSING))
            )
            if qx is not None and qy is not None:
                raw_q = float(np.hypot(qx, qy))
        q_value = _finite(raw_q)
        if q_star_nm is None and (q_value is None or q_value <= 0.0):
            continue
        if q_star_nm is not None and q_star_nm <= 0.0:
            q_star_nm = None
        scale = _q_scale_to_nm(q_unit)
        q_nm = (
            q_star_nm
            if q_star_nm is not None
            else (
                q_value * scale if scale is not None and q_value is not None else None
            )
        )
        if q_value is None and q_star_nm is not None:
            q_value = q_star_nm if scale in (None, 1.0) else q_star_nm / scale
        if q_value is None or q_value <= 0.0:
            continue
        status = str(
            _read(
                peak,
                ("status", "identifiability_status"),
                _read(metadata, ("status", "identifiability_status"), "available"),
            )
            or "available"
        ).lower()
        if status in {"invalid", "rejected", "failed", "unavailable"}:
            continue
        result.append(
            {
                "index": index,
                "angle_deg": float(angle),
                "q": float(q_value),
                "q_nm_inv": (None if q_nm is None else float(q_nm)),
                "q_unit": q_unit,
                "branch_id": _branch_value(peak),
                "status": status,
                "reason": str(
                    _read(peak, ("reason",), _read(metadata, ("reason",), "")) or ""
                ),
            }
        )
    return result


def _axial_distance_deg(first: float, second: float) -> float:
    difference = (float(second) - float(first) + 90.0) % 180.0 - 90.0
    return abs(difference)


def _assign_branches(peaks: list[dict[str, Any]]) -> None:
    explicit = [peak["branch_id"] for peak in peaks if peak["branch_id"] is not None]
    if len(explicit) == len(peaks):
        return
    anchor = float(peaks[0]["angle_deg"])
    for peak in peaks:
        if peak["branch_id"] is None:
            peak["branch_id"] = (
                0 if _axial_distance_deg(anchor, peak["angle_deg"]) <= 45.0 else 1
            )


def _group_populations(
    peaks: list[dict[str, Any]],
) -> list[tuple[int, list[dict[str, Any]]]]:
    _assign_branches(peaks)
    groups: dict[int, list[dict[str, Any]]] = {}
    for peak in peaks:
        branch = int(peak["branch_id"])
        groups.setdefault(branch, []).append(peak)
    return sorted(groups.items(), key=lambda item: (item[0], item[1][0]["index"]))


def _direction_for_group(group: Sequence[Mapping[str, Any]]) -> tuple[float, bool]:
    reference = float(group[0]["angle_deg"])
    doubled = np.radians([float(item["angle_deg"]) * 2.0 for item in group])
    mean = 0.5 * float(
        np.degrees(np.arctan2(np.mean(np.sin(doubled)), np.mean(np.cos(doubled))))
    )
    candidate = mean
    while candidate - reference >= 90.0:
        candidate -= 180.0
    while candidate - reference < -90.0:
        candidate += 180.0
    inconsistent = any(
        _axial_distance_deg(reference, float(item["angle_deg"])) > 15.0
        for item in group[1:]
    )
    return float(candidate), inconsistent


def _ellipse_payload(root: Mapping[str, Any]) -> Any:
    for container in (
        root,
        _read(root, ("observables", "measurements"), None),
        _read(root, ("butterfly",), None),
    ):
        candidate = _read(
            container, ("ellipse_fit", "ellipse", "candidate_fit"), _MISSING
        )
        if candidate is not _MISSING and candidate is not None:
            return candidate
    return None


def _parameter_value(value: Any) -> tuple[float | None, bool, str]:
    if isinstance(value, Mapping):
        status = str(
            value.get("status", value.get("identifiability_status", "available"))
            or "available"
        ).lower()
        candidate = value.get("candidate_value", value.get("candidate", _MISSING))
        number = _finite(
            value.get("value", value.get("estimate", value.get("val", _MISSING)))
        )
        candidate_number = _finite(candidate) if candidate is not _MISSING else None
        candidate_status = (
            status in {"candidate", "provisional", "undetermined"}
            or candidate_number is not None
            and status != "available"
        )
        if candidate_status:
            return (
                (candidate_number if candidate_number is not None else number),
                True,
                status,
            )
        return number, False, status
    number = _finite(value)
    return number, False, "available"


def _ellipse_components(
    ellipse: Any,
) -> tuple[float | None, bool, str, tuple[float, float] | None, str, float | None]:
    if ellipse is None:
        return None, False, "unavailable", None, "unknown", None
    q_unit = str(_read(ellipse, ("q_unit", "source_q_unit"), "unknown") or "unknown")
    parameters = _read(ellipse, ("parameters", "quantitative_parameters"), {})
    b_value = _read(ellipse, ("b", "semi_minor", "minor_axis"), _MISSING)
    b_candidate = False
    b_status = str(
        _read(ellipse, ("status", "identifiability_status"), "available") or "available"
    ).lower()
    if b_value is _MISSING and isinstance(parameters, Mapping):
        b_value = parameters.get(
            "b", parameters.get("semi_minor", parameters.get("minor_axis", _MISSING))
        )
    if b_value is _MISSING:
        b_value = None
    b, candidate, status = _parameter_value(b_value)
    b_candidate |= candidate
    if status != "available":
        b_status = status
    centre = _read(ellipse, ("center", "centre"), _MISSING)
    if centre is _MISSING and isinstance(parameters, Mapping):
        centre = parameters.get("center", parameters.get("centre", _MISSING))
    center_pair: tuple[float, float] | None = None
    if centre is not _MISSING and centre is not None:
        try:
            values = list(centre)
            if (
                len(values) >= 2
                and _finite(values[0]) is not None
                and _finite(values[1]) is not None
            ):
                center_pair = (float(values[0]), float(values[1]))
        except (TypeError, ValueError):
            center_pair = None
    if center_pair is None:
        cx = _read(ellipse, ("cx", "center_qx"), _MISSING)
        cy = _read(ellipse, ("cy", "center_qy"), _MISSING)
        if cx is _MISSING and isinstance(parameters, Mapping):
            cx = parameters.get("cx", parameters.get("center_qx", _MISSING))
        if cy is _MISSING and isinstance(parameters, Mapping):
            cy = parameters.get("cy", parameters.get("center_qy", _MISSING))
        if (
            cx is not _MISSING
            and cy is not _MISSING
            and _finite(cx) is not None
            and _finite(cy) is not None
        ):
            center_pair = (float(cx), float(cy))
    if center_pair is None:
        existing = _finite(
            _read(ellipse, ("Ln_from_minor_axis_nm", "L_N", "Ln_nm"), _MISSING)
        )
        if existing is not None and existing > 0.0:
            center_pair = (0.0, 0.0)
    if b_status in {"candidate", "provisional", "undetermined"}:
        b_candidate = True
    return (
        b,
        b_candidate,
        b_status,
        center_pair,
        q_unit,
        _finite(_read(ellipse, ("Ln_from_minor_axis_nm", "L_N", "Ln_nm"), _MISSING)),
    )


def _ellipse_parameter_records(ellipse: Any) -> list[dict[str, Any]]:
    if ellipse is None:
        return []
    containers: list[Mapping[str, Any]] = []
    for name in ("quantitative_parameters", "parameters"):
        candidate = _read(ellipse, (name,), None)
        if isinstance(candidate, Mapping):
            containers.append(candidate)
    if not containers and isinstance(ellipse, Mapping):
        direct = {
            name: ellipse[name]
            for name in ("a", "b", "axis_ratio", "theta_deg")
            if name in ellipse
        }
        if direct:
            containers.append(direct)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    ellipse_unit = str(
        _read(ellipse, ("q_unit", "source_q_unit"), "unknown") or "unknown"
    )
    for container in containers:
        for name, spec in container.items():
            label = str(name)
            if label in seen:
                continue
            seen.add(label)
            mapping = spec if isinstance(spec, Mapping) else {}
            value, candidate, status = _parameter_value(spec)
            if mapping and mapping.get("unit") not in (None, ""):
                unit = mapping["unit"]
            elif label in {
                "a",
                "b",
                "semi_major",
                "semi_minor",
                "major_axis",
                "minor_axis",
                "cx",
                "cy",
                "center_qx",
                "center_qy",
            }:
                unit = ellipse_unit
            elif label.endswith("_deg") or label in {
                "theta",
                "reference_axis",
                "reference_axis_deg",
                "angle",
            }:
                unit = "degree"
            else:
                unit = "dimensionless"
            interval = (
                mapping.get("interval", mapping.get("confidence_interval"))
                if mapping
                else None
            )
            if (
                isinstance(interval, Sequence)
                and not isinstance(interval, (str, bytes))
                and len(interval) == 2
            ):
                interval = [_finite(interval[0]), _finite(interval[1])]
            else:
                interval = None
            reason = (
                str(
                    mapping.get("reason", mapping.get("identifiability_reason", ""))
                    or ""
                )
                if mapping
                else ""
            )
            candidate_value = value if candidate else None
            formal_value = None if candidate else value
            rows.append(
                _period_record(
                    label,
                    formal_value,
                    str(unit or ""),
                    "ellipse_fit",
                    status,
                    reason,
                    candidate_value=candidate_value,
                    interval=interval,
                )
            )
    return rows


def _stable_seed(seed: int, branch: int, stack_index: int) -> int:
    value = int(seed) + 1000003 * (int(branch) + 257) + 104729 * (int(stack_index) + 1)
    return value % (2**63 - 1)


def _metadata_base(
    settings: LamellarSettings,
    source_identity: Mapping[str, Any],
    *,
    status: str,
    available: bool,
    message: str,
    draw_axis_deg: float,
    reference_period: float,
    assumptions: Sequence[str] = (),
    parameter_sources: Sequence[Mapping[str, Any]] = (),
    populations: Sequence[Mapping[str, Any]] = (),
    flags: Sequence[str] = (),
    q_unit: str = "unknown",
) -> dict[str, Any]:
    merged_flags = list(dict.fromkeys([*_SCENE_FLAGS, *(str(item) for item in flags)]))
    return {
        "available": bool(available),
        "status": str(status),
        "message": str(message),
        "source_identity": _copy_metadata(dict(source_identity)),
        "assumptions": [str(item) for item in assumptions],
        "settings": settings.to_dict(),
        "parameter_sources": [_copy_metadata(dict(item)) for item in parameter_sources],
        "populations": [_copy_metadata(dict(item)) for item in populations],
        "scientific_boundary": (
            "Parameter-driven lamellar schematic from apparent SAXS geometry; "
            "not a unique 3-D structural inversion or calibrated forward simulation."
        ),
        "draw_axis_deg": float(draw_axis_deg),
        "reference_period": float(reference_period),
        "q_unit": str(q_unit),
        "length_unit": "nm"
        if any(
            str(item.get("unit", "")) == "nm" and item.get("value") is not None
            for item in parameter_sources
        )
        else "relative",
        "flags": merged_flags,
    }


def _period_record(
    name: str,
    value: float | None,
    unit: str,
    source: str,
    status: str,
    reason: str = "",
    *,
    candidate_value: float | None = None,
    interval: Any = None,
) -> dict[str, Any]:
    return {
        "name": str(name),
        "value": None if value is None else float(value),
        "unit": str(unit),
        "source": str(source),
        "status": str(status),
        "interval": _copy_metadata(interval) if interval is not None else None,
        "reason": str(reason),
        **(
            {"candidate_value": float(candidate_value)}
            if candidate_value is not None
            else {}
        ),
    }
