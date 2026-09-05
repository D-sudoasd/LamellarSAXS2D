"""Canonical public payload normalization for measured ellipse fits."""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any



def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "__dict__"):
        return {
            str(key): item
            for key, item in vars(value).items()
            if not str(key).startswith("_")
        }
    return {}


def public_ellipse_payload(value: Any) -> dict[str, Any]:
    """Add stable aliases while preserving every existing fit diagnostic.

    Service and pipeline adapters may expose different envelope fields, but
    their nested ellipse body passes through this helper.  No scientific value
    is recomputed here; aliases are copied from the already-authoritative fit.
    """

    result = _mapping(value)
    values = result.get("parameters")
    if not isinstance(values, Mapping) or not values:
        values = result.get("parameter_values", {})
    values = dict(values) if isinstance(values, Mapping) else {}
    result.setdefault("parameters", dict(values))
    result.setdefault("parameter_values", dict(values))

    aliases = {
        "semi_major": "a",
        "semi_minor": "b",
        "axes_ratio": "axis_ratio",
        "ellipse_axis_tilt_deg": "theta_deg",
    }
    for target, source in aliases.items():
        if target not in result and source in result:
            result[target] = result[source]
    for target, source in aliases.items():
        if target not in values and source in result:
            values[target] = result[source]
    result["parameters"] = dict(values)
    result["parameter_values"] = dict(values)

    members = result.get("ellipses")
    if isinstance(members, (list, tuple)):
        normalized_members: list[dict[str, Any]] = []
        for member in members:
            row = _mapping(member)
            if "theta_deg" not in row and "angle_deg" in row:
                row["theta_deg"] = row["angle_deg"]
            if "angle_deg" not in row and "theta_deg" in row:
                row["angle_deg"] = row["theta_deg"]
            normalized_members.append(row)
        result["ellipses"] = normalized_members
    return result


def _field(value: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(value, Mapping) and name in value:
            return value[name]
        if hasattr(value, name):
            return getattr(value, name)
    return default


def _finite(value: Any, default: float = float("nan")) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def canonical_ellipse_payload(
    fit: Any,
    *,
    n_points: int = 0,
    qmap: Any = None,
    config: Any = None,
) -> dict[str, Any]:
    """Build the shared public geometry body from one raw fit object."""

    values = _field(fit, "parameter_values", "values", "parameters", default={})
    values = dict(values) if isinstance(values, Mapping) else {}
    a = _finite(values.get("a", _field(fit, "a", default=float("nan"))))
    ratio = _finite(
        values.get("axis_ratio", _field(fit, "axis_ratio", "axes_ratio", default=float("nan")))
    )
    b = _finite(values.get("b", _field(fit, "b", default=float("nan"))))
    if not math.isfinite(b) and math.isfinite(a * ratio):
        b = a * ratio
    if not math.isfinite(ratio) and math.isfinite(a) and a != 0:
        ratio = b / a
    theta_deg = _finite(_field(fit, "theta_deg", "angle_deg", default=values.get("theta_deg")))
    if not math.isfinite(theta_deg):
        theta_deg = math.degrees(_finite(values.get("theta", float("nan"))))
    center = _field(fit, "center", "centre", default=None)
    if center is None:
        center = (
            values.get("cx", _field(fit, "center_qx", default=0.0)),
            values.get("cy", _field(fit, "center_qy", default=0.0)),
        )
    try:
        cx, cy = float(center[0]), float(center[1])
    except (TypeError, ValueError, IndexError):
        cx = _finite(values.get("cx", 0.0), 0.0)
        cy = _finite(values.get("cy", 0.0), 0.0)
    q_unit = _field(fit, "q_unit", default=None)
    if q_unit is None and isinstance(qmap, Mapping):
        q_unit = qmap.get("q_unit", qmap.get("unit"))
    if q_unit is None:
        q_unit = "unknown"
    reference_axis = _finite(_field(fit, "reference_axis_deg", default=float("nan")))
    if not math.isfinite(reference_axis):
        analysis = config.get("analysis", config) if isinstance(config, Mapping) else getattr(config, "analysis", {})
        reference_axis = _finite(
            analysis.get("draw_axis_deg", 90.0) if isinstance(analysis, Mapping) else 90.0,
            90.0,
        ) - 90.0
    quality = _field(fit, "quality", default={}) or {}
    if not isinstance(quality, Mapping):
        quality = {}
    success = bool(_field(fit, "success", default=False))
    solver_status = str(_field(fit, "status", "solver_status", default="ok" if success else "failed"))
    raw_flags = _field(fit, "flags", default=()) or ()
    if isinstance(raw_flags, str):
        raw_flags = (raw_flags,)
    flags = tuple(dict.fromkeys(("apparent_geometry_only", "nonunique_inverse_problem", *(str(x) for x in raw_flags))))
    parameters = {
        **values,
        "a": a,
        "b": b,
        "axis_ratio": ratio,
        "center_qx": cx,
        "center_qy": cy,
        "theta_deg": theta_deg,
    }
    common = {
        "a": a,
        "b": b,
        "semi_major": a,
        "semi_minor": b,
        "axis_ratio": ratio,
        "center_qx": cx,
        "center_qy": cy,
        "reference_axis_deg": reference_axis,
        "ellipse_axis_tilt_deg": theta_deg,
        "q_unit": str(q_unit),
        "eccentricity": _finite(_field(fit, "eccentricity", "ellipticity", default=float("nan"))),
        "ellipticity": _finite(_field(fit, "ellipticity", "eccentricity", default=float("nan"))),
        "L_N": _finite(_field(fit, "L_N", "Ln_from_minor_axis_nm", default=float("nan"))),
        "L_z": _finite(_field(fit, "L_z", "Lz_from_draw_axis_nm", default=float("nan"))),
    }
    common["Ln_from_minor_axis_nm"] = common["L_N"]
    common["Lz_from_draw_axis_nm"] = common["L_z"]
    # Longitudinal/export consumers read the parameter mapping directly.
    # Preserve every canonical public scalar there, including unavailable
    # values represented as NaN until the JSON boundary turns them into null.
    parameters.update(common)
    parameters["theta_deg"] = theta_deg
    members: list[dict[str, Any]] = []
    for member in _field(fit, "ellipses", "ellipse_pair", default=()) or ():
        row = _mapping(member)
        member_theta = _finite(row.get("theta_deg", row.get("angle_deg", float("nan"))))
        if not math.isfinite(member_theta):
            member_theta = math.degrees(_finite(row.get("theta", float("nan"))))
        members.append({**common, **row, "theta_deg": member_theta, "angle_deg": member_theta})
    if not members:
        members = [
            {**common, "theta_deg": theta_deg, "angle_deg": theta_deg},
            {**common, "theta_deg": -theta_deg, "angle_deg": -theta_deg},
        ]
    payload = {
        "status": solver_status,
        "solver_status": solver_status,
        "quality_status": str(quality.get("status", quality.get("engineering_status", "")) or "").upper() or None,
        "quality": quality,
        "success": success,
        "message": str(_field(fit, "message", default="")),
        "n_points": int(_field(fit, "n_points", "n_data", default=n_points) or n_points),
        "ellipses": members,
        "parameters": parameters,
        "parameter_values": parameters,
        **common,
        "theta_deg": theta_deg,
        "angle_deg": theta_deg,
        "rmse": _finite(_field(fit, "rmse", "residual_rms", default=float("nan"))),
        "residual_rms": _finite(_field(fit, "residual_rms", "rmse", default=float("nan"))),
        "rss": _finite(_field(fit, "rss", default=float("nan"))),
        "stderr": _field(fit, "stderr", default={}) or {},
        "condition": _finite(_field(fit, "condition", "condition_number", default=float("nan"))),
        "coverage": _field(fit, "coverage", default={}) or {},
        "bound_flags": _field(fit, "bound_flags", default={}) or {},
        "bound_status": _field(fit, "bound_status", default={}) or {},
        "branch_counts": _field(fit, "branch_counts", default=(0, 0)),
        "branch_assignment": _field(fit, "branch_assignment", "branch_assignments", default=[]),
        "branch_assignment_indices": _field(
            fit, "branch_assignment_indices", default=[]
        ),
        "residuals": _field(fit, "residuals", default=[]),
        "symmetry": _field(fit, "symmetry", default={}) or {},
        "candidate_solutions": _field(fit, "candidate_solutions", default=()) or (),
        "selected_start_index": int(_field(fit, "selected_start_index", default=0) or 0),
        "multistart_count": int(_field(fit, "multistart_count", default=1) or 1),
        "flags": flags,
    }
    return public_ellipse_payload(payload)


__all__ = ["canonical_ellipse_payload", "public_ellipse_payload"]
