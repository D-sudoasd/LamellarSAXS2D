"""Canonical public payload normalization for measured ellipse fits."""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

from .observables import _SPACING_CENTER_REL_TOL, _q_to_nm_inverse_scale, ellipse_radius



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


def _origin_centered_periods(
    *,
    a: float,
    b: float,
    theta_deg: float,
    cx: float,
    cy: float,
    q_unit: Any,
    draw_axis_deg: float = 90.0,
) -> tuple[float, float, float, tuple[str, ...]]:
    """Apparent Ln/Lz/L_major from an origin-centred q ellipse, or NaN with flags."""

    unavailable = ("spacing_unavailable_unknown_q_unit",)
    scale = _q_to_nm_inverse_scale(q_unit)
    if scale is None:
        return float("nan"), float("nan"), float("nan"), unavailable
    if not (math.isfinite(a) and math.isfinite(b) and a > 0.0 and b > 0.0):
        return float("nan"), float("nan"), float("nan"), ("spacing_unavailable_degenerate_axes",)
    center_scale = max(1.0, abs(a), abs(b))
    center_tol = _SPACING_CENTER_REL_TOL * center_scale
    if not (math.isfinite(cx) and math.isfinite(cy) and math.hypot(cx, cy) <= center_tol):
        return float("nan"), float("nan"), float("nan"), (
            "spacing_unavailable_nonzero_center",
            "spacing_requires_origin_centered_ellipse_assumption",
        )
    ln = 2.0 * math.pi / (b * scale)
    l_major = 2.0 * math.pi / (a * scale)
    try:
        draw = float(draw_axis_deg)
    except (TypeError, ValueError):
        draw = 90.0
    if not math.isfinite(draw):
        draw = 90.0
    qz = float(ellipse_radius(math.radians(draw), a, b, math.radians(theta_deg)))
    lz = 2.0 * math.pi / (qz * scale) if math.isfinite(qz) and qz > 0.0 else float("nan")
    return ln, lz, l_major, ("spacing_requires_origin_centered_ellipse_assumption",)


def observed_arc_radius_period(
    points: Any,
    q_unit: Any,
    first_order_q: Any = None,
) -> tuple[float, float, tuple[str, ...]]:
    """Apparent Bragg period of the first-order ring.

    When a two-ellipse fit pins ``b/a`` on a bound, ``2π/b`` is not a usable
    lamellar period.  The I(q)* hint is the azimuthally averaged first-order
    peak; the median |q| of accepted arcs is kept only when that hint is
    missing.  Curvature tips sit outside I(q)* and must not walk the published
    period.
    """

    radii: list[float] = []
    if isinstance(points, (list, tuple)):
        rows = points
    else:
        rows = ()
    for point in rows:
        if not isinstance(point, Mapping):
            continue
        try:
            radius = math.hypot(float(point["qx"]), float(point["qy"]))
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(radius) and radius > 0.0:
            radii.append(radius)
    radii.sort()
    mid = len(radii) // 2
    arc_q = (
        radii[mid]
        if len(radii) % 2 == 1
        else (0.5 * (radii[mid - 1] + radii[mid]) if len(radii) >= 2 else float("nan"))
    )
    if len(radii) < 5:
        arc_q = float("nan")
    try:
        hint = float(first_order_q) if first_order_q is not None else float("nan")
    except (TypeError, ValueError):
        hint = float("nan")
    scale = _q_to_nm_inverse_scale(q_unit)
    if math.isfinite(hint) and hint > 0.0:
        flags = ["spacing_from_first_order_iq"]
        if math.isfinite(arc_q) and arc_q > 1.8 * hint:
            flags.append("arc_radius_on_secondary_population")
        elif not math.isfinite(arc_q):
            flags.append("spacing_iq_without_dense_arc_radius")
        if scale is None:
            return hint, float("nan"), tuple(flags + ["spacing_unavailable_unknown_q_unit"])
        return hint, 2.0 * math.pi / (hint * scale), tuple(flags)
    if not math.isfinite(arc_q):
        return float("nan"), float("nan"), ("spacing_unavailable_insufficient_arc_radius",)
    if scale is None:
        return arc_q, float("nan"), ("spacing_unavailable_unknown_q_unit",)
    return arc_q, 2.0 * math.pi / (arc_q * scale), ("spacing_from_observed_arc_radius",)


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
        "L_from_major_axis_nm": _finite(
            _field(fit, "L_from_major_axis_nm", default=float("nan"))
        ),
    }
    if not math.isfinite(common["ellipticity"]) and math.isfinite(ratio):
        common["ellipticity"] = math.sqrt(max(0.0, 1.0 - ratio * ratio))
        common["eccentricity"] = common["ellipticity"]
    if (
        not math.isfinite(common["L_N"])
        or not math.isfinite(common["L_z"])
        or not math.isfinite(common["L_from_major_axis_nm"])
    ):
        ln, lz, l_major, spacing_flags = _origin_centered_periods(
            a=a,
            b=b,
            theta_deg=theta_deg,
            cx=cx,
            cy=cy,
            q_unit=q_unit,
            draw_axis_deg=reference_axis + 90.0,
        )
        if not math.isfinite(common["L_N"]):
            common["L_N"] = ln
        if not math.isfinite(common["L_z"]):
            common["L_z"] = lz
        if not math.isfinite(common["L_from_major_axis_nm"]):
            common["L_from_major_axis_nm"] = l_major
        flags = tuple(dict.fromkeys((*flags, *spacing_flags)))
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
    for name in ("method_version", "candidate_fit", "quantitative_parameters",
                 "parameter_identifiability", "arc_diagnostics", "point_diagnostics",
                 "uncertainty", "sensitivity", "warm_start_eligible", "measurement_status", "branch_swap_applied"):
        value = _field(fit, name, default=None)
        if value is not None:
            payload[name] = value
    return public_ellipse_payload(payload)


__all__ = ["canonical_ellipse_payload", "public_ellipse_payload"]
