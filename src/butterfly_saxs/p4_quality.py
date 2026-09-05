"""Fail-closed engineering quality checks for P4 apparent geometry.

The checks in this module are deliberately separated from scientific
acceptance.  They prevent a numerically converged ellipse from being reported
as reliable when the ridge support is insufficient, the solution is
ill-conditioned, or the residual is large compared with the measured ridge
width.  The numerical limits are the provisional P4 contract from the v2.0
plan; they do not become final scientific thresholds until the external P3
evidence has been completed and frozen.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


ENGINEERING_THRESHOLDS_VERSION = "p4-engineering-provisional-v1"

DEFAULT_ENGINEERING_THRESHOLDS: dict[str, float] = {
    "minimum_points": 5.0,
    "minimum_valid_fraction_fail": 0.05,
    "minimum_valid_fraction_warn": 0.15,
    "minimum_continuity_fraction_fail": 0.25,
    "minimum_continuity_fraction_warn": 0.75,
    "minimum_angular_coverage_fail": 0.50,
    "minimum_angular_coverage_warn": 0.75,
    "residual_local_fwhm_fraction_max": 0.25,
    "condition_warn": 1.0e8,
    "condition_fail": 1.0e12,
    "near_circular_axis_ratio": 0.98,
    # These diagnostics are deliberately conservative.  They expose when a
    # thin ellipse is being extrapolated from wing arcs; they do not turn a
    # finite optimizer result into a physical acceptance claim.
    "short_arc_span_fail_deg": 90.0,
    "short_arc_span_warn_deg": 180.0,
    "flat_axis_ratio_warn": 0.10,
    "major_axis_extrapolation_fraction": 0.05,
}


def _read(source: Any, name: str, default: Any = None) -> Any:
    if isinstance(source, Mapping):
        return source.get(name, default)
    return getattr(source, name, default)


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _physical_q_unit(value: Any) -> bool:
    normalized = (
        str(value or "unknown")
        .strip()
        .casefold()
        .replace(" ", "")
        .replace("⁻¹", "^-1")
        .replace("−", "-")
        .replace("å", "a")
        .replace("Å", "a")
    )
    return normalized in {
        "1/nm",
        "nm^-1",
        "nm-1",
        "nm**-1",
        "1/a",
        "a^-1",
        "a-1",
        "a**-1",
        "1/angstrom",
        "angstrom^-1",
        "angstrom-1",
        "angstrom**-1",
    }


def _point_rows(ridge: Any) -> list[Any]:
    points = _read(ridge, "points", ridge)
    if points is None or isinstance(points, (str, bytes, Mapping)):
        return []
    if isinstance(points, Sequence) or isinstance(points, np.ndarray):
        return list(points)
    return []


def _coverage_value(ellipse: Any) -> float | None:
    coverage = _read(ellipse, "coverage", None)
    return _finite(_read(coverage, "angular_coverage", None))


def _candidate_theta_spread_deg(ellipse: Any) -> float | None:
    candidates = _read(ellipse, "candidate_solutions", ()) or ()
    accepted: list[tuple[float, float]] = []
    for candidate in candidates:
        if not bool(_read(candidate, "success", False)):
            continue
        cost = _finite(_read(candidate, "cost", None))
        values = _read(candidate, "values", {})
        theta = _finite(_read(values, "theta", None))
        if cost is not None and theta is not None:
            accepted.append((cost, float(np.degrees(theta))))
    if len(accepted) < 2:
        return None
    best = min(item[0] for item in accepted)
    tolerance = max(1.0e-12, abs(best) * 0.01)
    equivalent = [theta for cost, theta in accepted if cost <= best + tolerance]
    if len(equivalent) < 2:
        return 0.0
    return float(max(equivalent) - min(equivalent))


def evaluate_p4_ellipse_quality(
    ridge: Any,
    ellipse: Any,
    *,
    thresholds: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return engineering PASS/WARN/FAIL without claiming scientific acceptance."""

    limits = dict(DEFAULT_ENGINEERING_THRESHOLDS)
    if thresholds:
        for name in limits:
            if name in thresholds:
                value = _finite(thresholds[name])
                if value is None:
                    raise ValueError(f"P4 quality threshold {name!r} must be finite")
                if name == "minimum_points" and not float(value).is_integer():
                    raise ValueError("P4 quality threshold 'minimum_points' must be an integer")
                limits[name] = value

    rows = _point_rows(ridge)
    valid_rows = [
        point
        for point in rows
        if bool(_read(point, "valid", _read(point, "accepted", True)))
    ]
    widths = np.asarray(
        [
            value
            for point in valid_rows
            if (value := _finite(_read(point, "radial_fwhm", None))) is not None
            and value > 0.0
        ],
        dtype=float,
    )
    scores = np.asarray(
        [
            value
            for point in valid_rows
            if (value := _finite(_read(point, "score", None))) is not None
        ],
        dtype=float,
    )
    snr = np.asarray(
        [
            value
            for point in valid_rows
            if (value := _finite(_read(point, "snr", None))) is not None
        ],
        dtype=float,
    )

    rmse = _finite(_read(ellipse, "rmse", _read(ellipse, "residual_rms", None)))
    median_width = float(np.median(widths)) if widths.size else None
    residual_width_ratio = (
        float(rmse / median_width)
        if rmse is not None and median_width is not None and median_width > 0.0
        else None
    )
    condition = _finite(
        _read(ellipse, "condition_number", _read(ellipse, "condition", None))
    )
    coverage = _coverage_value(ellipse)
    branch_counts_raw = _read(ellipse, "branch_counts", (0, 0))
    try:
        branch_counts = [int(branch_counts_raw[0]), int(branch_counts_raw[1])]
    except (IndexError, TypeError, ValueError):
        branch_counts = [0, 0]
    values = _read(ellipse, "parameter_values", _read(ellipse, "values", {}))
    axis_ratio = _finite(
        _read(ellipse, "axes_ratio", _read(ellipse, "axis_ratio", _read(values, "axis_ratio", None)))
    )
    bound_flags = _read(ellipse, "bound_flags", {}) or {}
    critical_bound_names = [
        name
        for name in ("a", "axis_ratio", "theta")
        if bool(_read(bound_flags, name, False))
    ]
    theta_spread_deg = _candidate_theta_spread_deg(ellipse)
    valid_fraction = _finite(_read(ridge, "valid_fraction", None))
    continuity_fraction = _finite(_read(ridge, "continuity_fraction", None))
    continuity_score = _finite(_read(ridge, "continuity_score", None))
    trajectory_ids = {
        int(value)
        for point in valid_rows
        if (value := _read(point, "trajectory_id", None)) is not None
    }
    q_unit = _read(ridge, "q_unit", None)
    if q_unit is None and rows:
        q_unit = _read(rows[0], "q_unit", None)
    if q_unit is None:
        q_unit = _read(ellipse, "q_unit", "unknown")

    # Estimate how much of the fitted major-axis extent is directly observed.
    # This is a support diagnostic only.  A large fitted ``a`` can be a valid
    # empirical continuation of a wing, but it is not identified by data that
    # never reach that extent.
    observed_major_extent: float | None = None
    observed_radius_max: float | None = None
    fit_a = _finite(_read(ellipse, "a", _read(values, "a", None)))
    fit_theta = _finite(_read(ellipse, "theta", _read(values, "theta", None)))
    fit_cx = _finite(_read(ellipse, "cx", _read(values, "cx", None)))
    fit_cy = _finite(_read(ellipse, "cy", _read(values, "cy", None)))
    reference_axis_deg = _finite(_read(ellipse, "reference_axis_deg", 0.0)) or 0.0
    if fit_a is not None and fit_a > 0.0 and fit_theta is not None and fit_cx is not None and fit_cy is not None:
        qx_values: list[float] = []
        qy_values: list[float] = []
        for point in valid_rows:
            x = _finite(_read(point, "qx", _read(point, "x", None)))
            y = _finite(_read(point, "qy", _read(point, "y", None)))
            if x is None or y is None:
                q_value = _finite(_read(point, "q", _read(point, "q_star", None)))
                angle_value = _finite(_read(point, "angle", _read(point, "azimuth", None)))
                if q_value is None or angle_value is None:
                    continue
                x = q_value * float(np.cos(angle_value))
                y = q_value * float(np.sin(angle_value))
            qx_values.append(float(x))
            qy_values.append(float(y))
        if qx_values:
            qx_array = np.asarray(qx_values, dtype=float)
            qy_array = np.asarray(qy_values, dtype=float)
            laboratory_theta = np.deg2rad(reference_axis_deg) + fit_theta
            longitudinal = (
                np.cos(laboratory_theta) * (qx_array - fit_cx)
                + np.sin(laboratory_theta) * (qy_array - fit_cy)
            )
            observed_major_extent = float(np.max(np.abs(longitudinal)) / fit_a)
            observed_radius_max = float(
                np.max(np.hypot(qx_array - fit_cx, qy_array - fit_cy))
            )

    checks: list[dict[str, Any]] = []

    def add(name: str, status: str, value: Any, limit: Any, message: str) -> None:
        checks.append(
            {
                "name": name,
                "status": status,
                "value": value,
                "limit": limit,
                "message": message,
            }
        )

    solver_success = bool(_read(ellipse, "success", False))
    add(
        "solver_success",
        "PASS" if solver_success else "FAIL",
        solver_success,
        True,
        "solver converged" if solver_success else "ellipse solver did not converge",
    )

    physical_q = _physical_q_unit(q_unit)
    add(
        "physical_q_declared",
        "PASS" if physical_q else "WARN",
        str(q_unit or "unknown"),
        "nm^-1 or Å^-1",
        (
            "ridge coordinates have a declared physical reciprocal-space unit"
            if physical_q
            else "ridge coordinates are uncalibrated; geometry may run but cannot receive an engineering PASS"
        ),
    )

    finite_geometry = all(
        _finite(_read(ellipse, name, _read(values, name, None))) is not None
        for name in ("a", "b", "theta")
    )
    add(
        "finite_geometry",
        "PASS" if finite_geometry else "FAIL",
        finite_geometry,
        True,
        "ellipse geometry is finite" if finite_geometry else "ellipse geometry is non-finite",
    )

    minimum_points = int(limits["minimum_points"])
    add(
        "point_support",
        "PASS" if len(valid_rows) >= minimum_points else "FAIL",
        len(valid_rows),
        {"minimum": minimum_points},
        "enough observed ridge points" if len(valid_rows) >= minimum_points else "insufficient observed ridge points",
    )

    if valid_fraction is None:
        valid_fraction_status = "WARN"
        valid_fraction_message = "ridge valid fraction is unavailable"
    elif valid_fraction < limits["minimum_valid_fraction_fail"]:
        valid_fraction_status = "FAIL"
        valid_fraction_message = "too few angular sectors contain an observed ridge"
    elif valid_fraction < limits["minimum_valid_fraction_warn"]:
        valid_fraction_status = "WARN"
        valid_fraction_message = "ridge support covers only a small angular fraction"
    else:
        valid_fraction_status = "PASS"
        valid_fraction_message = "ridge valid fraction is adequate"
    add(
        "ridge_valid_fraction",
        valid_fraction_status,
        valid_fraction,
        {
            "fail_below": limits["minimum_valid_fraction_fail"],
            "warn_below": limits["minimum_valid_fraction_warn"],
        },
        valid_fraction_message,
    )

    if continuity_fraction is None:
        continuity_status = "WARN"
        continuity_message = "ridge continuity is unavailable"
    elif continuity_fraction < limits["minimum_continuity_fraction_fail"]:
        continuity_status = "FAIL"
        continuity_message = "ridge trajectory is dominated by gaps or jumps"
    elif continuity_fraction < limits["minimum_continuity_fraction_warn"]:
        continuity_status = "WARN"
        continuity_message = "ridge trajectory contains material gaps or jumps"
    else:
        continuity_status = "PASS"
        continuity_message = "ridge trajectory continuity is adequate"
    add(
        "ridge_continuity",
        continuity_status,
        {
            "continuity_fraction": continuity_fraction,
            "continuity_score": continuity_score,
            "trajectory_count": len(trajectory_ids),
        },
        {
            "fail_below": limits["minimum_continuity_fraction_fail"],
            "warn_below": limits["minimum_continuity_fraction_warn"],
        },
        continuity_message,
    )

    both_branches = min(branch_counts) > 0
    add(
        "branch_support",
        "PASS" if both_branches else "FAIL",
        branch_counts,
        {"each_branch_minimum": 1},
        "both fitted branches have observed support" if both_branches else "one fitted branch has no observed support",
    )

    symmetry = _read(ellipse, "symmetry", {}) or {}
    if isinstance(symmetry, Mapping) and symmetry:
        branch_leaks = _read(symmetry, "branch_leaks", {}) or {}
        selected_leaks = int(_finite(_read(branch_leaks, "selected", 0)) or 0)
        paired_support = _read(symmetry, "paired_support", {}) or {}
        paired_values = paired_support.values() if isinstance(paired_support, Mapping) else ()
        missing_opposite = int(
            sum(
                int(_finite(_read(value, "missing_opposite_count", 0)) or 0)
                for value in paired_values
                if isinstance(value, Mapping)
            )
        )
        unassigned = int(_finite(_read(symmetry, "unassigned_count", 0)) or 0)
        symmetry_status = str(_read(symmetry, "symmetry_status", "WARN"))
        if selected_leaks:
            symmetry_check_status = "FAIL"
            symmetry_message = "observed points leak across the fixed opposite-quadrant branch pairing"
        elif missing_opposite:
            symmetry_check_status = "WARN"
            symmetry_message = "one or more opposite quadrants lack an observed counterpart"
        elif symmetry_status not in {"PASS"} or unassigned:
            symmetry_check_status = "WARN"
            symmetry_message = "symmetry pairing is observationally incomplete or center-unverified"
        else:
            symmetry_check_status = "PASS"
            symmetry_message = "observed quadrant pairing is consistent"
        add(
            "symmetry_quadrant_pairing",
            symmetry_check_status,
            {
                "status": symmetry_status,
                "quadrant_counts": _read(symmetry, "quadrant_counts", {}),
                "branch_quadrant_counts": _read(symmetry, "branch_quadrant_counts", {}),
                "paired_support": paired_support,
                "branch_leaks": branch_leaks,
                "unassigned_count": unassigned,
            },
            {"missing_opposite_count": 0, "branch_leaks": 0},
            symmetry_message,
        )

    if coverage is None:
        coverage_status = "FAIL"
        coverage_message = "angular coverage is unavailable"
    elif coverage < limits["minimum_angular_coverage_fail"]:
        coverage_status = "FAIL"
        coverage_message = "angular coverage is too low"
    elif coverage < limits["minimum_angular_coverage_warn"]:
        coverage_status = "WARN"
        coverage_message = "angular coverage is limited"
    else:
        coverage_status = "PASS"
        coverage_message = "angular coverage is adequate"
    add(
        "angular_coverage",
        coverage_status,
        coverage,
        {
            "fail_below": limits["minimum_angular_coverage_fail"],
            "warn_below": limits["minimum_angular_coverage_warn"],
        },
        coverage_message,
    )

    if condition is None:
        condition_status = "FAIL"
        condition_message = "condition number is non-finite"
    elif condition > limits["condition_fail"]:
        condition_status = "FAIL"
        condition_message = "fit is severely ill-conditioned"
    elif condition > limits["condition_warn"]:
        condition_status = "WARN"
        condition_message = "fit is weakly identifiable"
    else:
        condition_status = "PASS"
        condition_message = "condition number is finite"
    add(
        "condition_number",
        condition_status,
        condition,
        {
            "warn_above": limits["condition_warn"],
            "fail_above": limits["condition_fail"],
        },
        condition_message,
    )

    add(
        "critical_parameter_bounds",
        "FAIL" if critical_bound_names else "PASS",
        critical_bound_names,
        [],
        "critical parameter reached a bound" if critical_bound_names else "critical parameters are away from bounds",
    )
    add(
        "bound_saturation",
        "FAIL" if critical_bound_names else "PASS",
        {
            "critical_bound_names": critical_bound_names,
            # Retain this explicit name for UI/export consumers while making
            # the compatibility relationship with the older check clear.
            "alias_of": "critical_parameter_bounds",
        },
        {"critical_parameters": ("a", "axis_ratio", "theta")},
        "fit is saturated at a critical parameter bound" if critical_bound_names else "critical fit parameters are not bound-saturated",
    )

    if residual_width_ratio is None:
        residual_status = "WARN"
        residual_message = "local ridge width is unavailable for residual normalization"
    elif residual_width_ratio > limits["residual_local_fwhm_fraction_max"]:
        residual_status = "FAIL"
        residual_message = "ellipse residual exceeds the provisional local-width limit"
    else:
        residual_status = "PASS"
        residual_message = "ellipse residual is small relative to local ridge width"
    add(
        "residual_vs_local_width",
        residual_status,
        residual_width_ratio,
        {"maximum": limits["residual_local_fwhm_fraction_max"]},
        residual_message,
    )

    near_circular = axis_ratio is not None and axis_ratio >= limits["near_circular_axis_ratio"]
    add(
        "orientation_identifiability",
        "WARN" if near_circular else "PASS",
        {
            "axis_ratio": axis_ratio,
            "equivalent_theta_spread_deg": theta_spread_deg,
        },
        {"axis_ratio_warn_at_or_above": limits["near_circular_axis_ratio"]},
        "ellipse is nearly circular; theta is not reliably identifiable" if near_circular else "axis ratio supports an orientation estimate",
    )

    coverage_span = _finite(_read(_read(ellipse, "coverage", None), "angular_span", None))
    if coverage_span is None and coverage is not None:
        coverage_span = float(coverage * 2.0 * np.pi)
    span_deg = float(np.degrees(coverage_span)) if coverage_span is not None else None
    if span_deg is None:
        add(
            "short_arc",
            "WARN",
            None,
            {
                "fail_below_deg": limits["short_arc_span_fail_deg"],
                "warn_below_deg": limits["short_arc_span_warn_deg"],
            },
            "fitted angular span is unavailable",
        )
    elif span_deg < limits["short_arc_span_fail_deg"]:
        add(
            "short_arc",
            "FAIL",
            span_deg,
            {
                "fail_below_deg": limits["short_arc_span_fail_deg"],
                "warn_below_deg": limits["short_arc_span_warn_deg"],
            },
            "ellipse is fitted from a short angular arc",
        )
    elif span_deg < limits["short_arc_span_warn_deg"]:
        add(
            "short_arc",
            "WARN",
            span_deg,
            {
                "fail_below_deg": limits["short_arc_span_fail_deg"],
                "warn_below_deg": limits["short_arc_span_warn_deg"],
            },
            "ellipse angular support is limited",
        )
    else:
        add(
            "short_arc",
            "PASS",
            span_deg,
            {
                "fail_below_deg": limits["short_arc_span_fail_deg"],
                "warn_below_deg": limits["short_arc_span_warn_deg"],
            },
            "ellipse angular support spans a broad arc",
        )

    if observed_major_extent is None:
        add(
            "major_axis_extrapolated",
            "WARN" if axis_ratio is not None and axis_ratio <= limits["flat_axis_ratio_warn"] else "PASS",
            None,
            {"minimum_direct_extent_fraction": 1.0 - limits["major_axis_extrapolation_fraction"]},
            (
                "direct major-axis extent is unavailable for a flat ellipse"
                if axis_ratio is not None and axis_ratio <= limits["flat_axis_ratio_warn"]
                else "direct major-axis extent is unavailable"
            ),
        )
    elif observed_major_extent < 1.0 - limits["major_axis_extrapolation_fraction"]:
        add(
            "major_axis_extrapolated",
            "WARN",
            observed_major_extent,
            {"minimum_direct_extent_fraction": 1.0 - limits["major_axis_extrapolation_fraction"]},
            "fitted major axis extends beyond the observed ridge support",
        )
    else:
        add(
            "major_axis_extrapolated",
            "PASS",
            observed_major_extent,
            {"minimum_direct_extent_fraction": 1.0 - limits["major_axis_extrapolation_fraction"]},
            "fitted major axis is directly supported by observed ridge extent",
        )

    flat_nonidentifiable = bool(
        axis_ratio is not None
        and axis_ratio <= limits["flat_axis_ratio_warn"]
        and (
            observed_major_extent is None
            or observed_major_extent < 1.0 - limits["major_axis_extrapolation_fraction"]
            or (span_deg is not None and span_deg < limits["short_arc_span_warn_deg"])
        )
    )
    add(
        "flat_ellipse_nonidentifiable",
        "WARN" if flat_nonidentifiable else "PASS",
        {
            "axis_ratio": axis_ratio,
            "direct_major_extent_fraction": observed_major_extent,
            "angular_span_deg": span_deg,
        },
        {
            "axis_ratio_warn_below": limits["flat_axis_ratio_warn"],
            "minimum_direct_extent_fraction": 1.0 - limits["major_axis_extrapolation_fraction"],
        },
        "thin ellipse parameters are weakly identified by the available arc support"
        if flat_nonidentifiable
        else "thin-ellipse extrapolation diagnostic is not triggered",
    )

    if any(item["status"] == "FAIL" for item in checks):
        status = "FAIL"
    elif any(item["status"] == "WARN" for item in checks):
        status = "WARN"
    else:
        status = "PASS"

    return {
        "status": status,
        "engineering_status": status,
        "scientific_status": "NOT_ACCEPTED",
        "scientific_reason": "P3 human consensus and frozen evidence-backed thresholds are not available",
        "thresholds_version": ENGINEERING_THRESHOLDS_VERSION,
        "thresholds_frozen": False,
        "checks": checks,
        "flags": [
            item["name"]
            for item in checks
            if item["status"] in {"WARN", "FAIL"}
        ],
        "metrics": {
            "n_points_total": len(rows),
            "n_points_valid": len(valid_rows),
            "q_unit": str(q_unit or "unknown"),
            "valid_fraction": valid_fraction,
            "continuity_fraction": continuity_fraction,
            "continuity_score": continuity_score,
            "trajectory_count": len(trajectory_ids),
            "median_score": float(np.median(scores)) if scores.size else None,
            "median_snr": float(np.median(snr)) if snr.size else None,
            "median_radial_fwhm": median_width,
            "rmse": rmse,
            "residual_local_fwhm_fraction": residual_width_ratio,
            "angular_coverage": coverage,
            "condition_number": condition,
            "branch_counts": branch_counts,
            "critical_bound_names": critical_bound_names,
            "axis_ratio": axis_ratio,
            "equivalent_theta_spread_deg": theta_spread_deg,
            "multistart_count": int(_read(ellipse, "multistart_count", 1) or 1),
            "angular_span_deg": span_deg,
            "direct_major_extent_fraction": observed_major_extent,
            "observed_radius_max": observed_radius_max,
            "symmetry": symmetry,
        },
    }


__all__ = [
    "DEFAULT_ENGINEERING_THRESHOLDS",
    "ENGINEERING_THRESHOLDS_VERSION",
    "evaluate_p4_ellipse_quality",
]
