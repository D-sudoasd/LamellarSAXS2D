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
        },
    }


__all__ = [
    "DEFAULT_ENGINEERING_THRESHOLDS",
    "ENGINEERING_THRESHOLDS_VERSION",
    "evaluate_p4_ellipse_quality",
]
