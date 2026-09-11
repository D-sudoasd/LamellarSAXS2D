from __future__ import annotations

import numpy as np
import pytest

from butterfly_saxs.arc_geometry import (
    _local_coordinates,
    _observed_radius_seed,
    _project_ellipse_arc,
    _project_point_to_support,
    _rectangle_intervals,
    _rectangle_reachable,
    _side_domain,
    fit_arc_ellipses,
)
from butterfly_saxs.analysis_config import ellipse_parameter_specs, validate_analysis_settings
from butterfly_saxs.ellipse import EllipseGeometry, fit_symmetric_ellipses
from butterfly_saxs.parameters import default_ellipse_parameters


def _arc_points(*, branch_swap: bool = False) -> list[dict[str, object]]:
    points: list[dict[str, object]] = []
    for branch, sign in enumerate((1.0, -1.0)):
        geometry = EllipseGeometry(0.0, 0.0, 1.5, 0.1, sign * 0.35)
        for side, values in (
            ("upper", np.linspace(0.35, 0.8, 10)),
            ("lower", np.linspace(3.35, 3.8, 10)),
        ):
            for value in values:
                qx, qy = geometry.point(value)
                points.append(
                    {
                        "point_id": len(points),
                        "qx": float(qx),
                        "qy": float(qy),
                        "branch_id": 1 - branch if branch_swap else branch,
                        "side": side,
                        "arc_id": branch * 2 + int(side == "lower"),
                        "accepted": True,
                        "localization_sigma_q": 0.001,
                        "arc_t_min": float(values[0]),
                        "arc_t_max": float(values[-1]),
                    }
                )
    return points


def _fixed_truth() -> dict[str, dict[str, object]]:
    return {
        "cx": {"value": 0.0, "vary": False},
        "cy": {"value": 0.0, "vary": False},
        "a": {"value": 1.5, "vary": False},
        "axis_ratio": {"value": 0.1, "vary": False},
        "theta": {"value": 0.35, "vary": False},
    }


def test_arc_fit_recovers_shared_geometry_and_point_aligned_projection() -> None:
    result = fit_arc_ellipses(_arc_points(), parameters=_fixed_truth(), multistart=1)

    fit = result["fit"]
    assert fit is not None and fit.success
    assert fit.values["a"] == 1.5
    assert fit.values["axis_ratio"] == 0.1
    assert len(result["point_diagnostics"]) == 40
    first = result["point_diagnostics"][0]
    assert first["point_id"] == 0
    assert first["projected_qx"] == pytest.approx(first["qx"])
    assert first["projected_qy"] == pytest.approx(first["qy"])
    assert first["distance_q"] < 1e-10
    assert "normal_residual_q" in first
    assert len(result["arc_diagnostics"]) == 4


def test_arc_fit_is_invariant_to_global_branch_id_swap() -> None:
    first = fit_arc_ellipses(_arc_points(), parameters=_fixed_truth(), multistart=1)
    swapped = fit_arc_ellipses(
        _arc_points(branch_swap=True), parameters=_fixed_truth(), multistart=1
    )

    assert swapped["fit"] is not None
    assert swapped["fit"].values == first["fit"].values
    assert swapped["fit"].cost == first["fit"].cost
    assert first["branch_swap_applied"] is False
    assert swapped["branch_swap_applied"] is True


@pytest.mark.parametrize("axis_ratio", (0.005, 0.02))
def test_free_flat_arc_fit_selects_low_cost_nonconverged_candidate_honestly(axis_ratio: float) -> None:
    points: list[dict[str, object]] = []
    for branch, sign in enumerate((1.0, -1.0)):
        geometry = EllipseGeometry(0.0, 0.0, 1.5, axis_ratio, sign * 0.35)
        for side, values in (
            ("upper", np.linspace(0.3, 0.8, 4)),
            ("lower", np.linspace(3.3, 3.8, 4)),
        ):
            for value in values:
                qx, qy = geometry.point(value)
                points.append(
                    {
                        "point_id": len(points),
                        "qx": float(qx),
                        "qy": float(qy),
                        "branch_id": branch,
                        "side": side,
                        "arc_id": branch * 2 + int(side == "lower"),
                        "accepted": True,
                        "localization_sigma_q": 0.001,
                        "arc_t_min": float(values[0]),
                        "arc_t_max": float(values[-1]),
                    }
                )
    parameters = {
        "cx": {"value": 0.05, "min": -0.5, "max": 0.5},
        "cy": {"value": -0.04, "min": -0.5, "max": 0.5},
        "a": {"value": 1.0, "min": 0.5, "max": 2.5},
        "axis_ratio": {"value": 0.1, "min": 0.005, "max": 0.5},
        "theta": {"value": 0.6, "min": 0.0, "max": np.pi / 2.0},
    }

    result = fit_arc_ellipses(points, parameters=parameters, multistart=1, max_nfev=800)
    fit = result["fit"]
    assert fit is not None and fit.success
    assert fit.values["cx"] == pytest.approx(0.0, abs=1e-6)
    assert fit.values["cy"] == pytest.approx(0.0, abs=1e-6)
    assert fit.values["a"] == pytest.approx(1.5, rel=1e-6)
    assert fit.values["axis_ratio"] == pytest.approx(axis_ratio, rel=1e-5)
    assert fit.values["theta"] == pytest.approx(0.35, abs=1e-6)
    finite_costs = [candidate["cost"] for candidate in fit.candidate_solutions]
    assert fit.cost == pytest.approx(min(finite_costs))
    assert result["branch_swap_applied"] is False


def test_arc_projection_reports_endpoint_clipping_and_excludes_unknown_labels() -> None:
    points: list[dict[str, object]] = []
    geometry = EllipseGeometry(0.0, 0.0, 1.5, 0.1, 0.35)
    for index, value in enumerate(np.linspace(0.0, 0.1, 6)):
        qx, qy = geometry.point(value)
        points.append(
            {
                "point_id": index,
                "qx": float(qx),
                "qy": float(qy),
                "branch_id": 0,
                "side": "upper",
                "arc_id": 3,
                "accepted": True,
                "arc_t_min": 0.2,
                "arc_t_max": 0.8,
            }
        )
    points.append(
        {
            "point_id": 99,
            "qx": 1.0,
            "qy": 1.0,
            "branch_id": 0,
            "side": "unknown",
            "arc_id": 4,
            "accepted": True,
        }
    )
    result = fit_arc_ellipses(points, parameters=_fixed_truth(), multistart=1)
    diagnostics = result["point_diagnostics"]
    assert all(item["endpoint_clipped"] for item in diagnostics[:6])
    assert diagnostics[-1]["used"] is False
    assert diagnostics[-1]["excluded_reason"] == "unknown_side"


def test_coverage_uses_connected_support_length_for_disconnected_arcs() -> None:
    parameters = _fixed_truth()
    points = []
    labels = []
    for branch, sign in enumerate((1.0, -1.0)):
        geometry = EllipseGeometry(0.0, 0.0, 1.5, 0.1, sign * 0.35)
        for value in np.linspace((0.2 if branch == 0 else 3.35) - 0.02, (0.2 if branch == 0 else 3.35) + 0.02, 5):
            points.append(geometry.point(value))
            labels.append(branch)

    fit = fit_symmetric_ellipses(
        np.asarray(points), parameters=parameters, labels=labels, multistart=1
    )
    assert fit.coverage.angular_span > 3.0
    assert fit.coverage.angular_coverage == pytest.approx(0.04 / (2.0 * np.pi))
    assert fit.coverage.observed_angular_length == pytest.approx(0.08)
    assert fit.coverage.branch_coverage == pytest.approx((0.04 / (2.0 * np.pi),) * 2)
    assert fit.coverage.support_components == 2


def test_all_excluded_arc_input_returns_structured_failure() -> None:
    result = fit_arc_ellipses(
        [
            {
                "point_id": 4,
                "qx": 1.0,
                "qy": 0.0,
                "branch_id": 0,
                "side": "unknown",
                "arc_id": 0,
                "accepted": True,
            }
        ],
        parameters=_fixed_truth(),
        multistart=1,
    )
    assert result["fit"] is None
    assert "no accepted points" in result["error"]


def _tip_arc_points(a: float, axis_ratio: float, theta: float) -> list[dict[str, object]]:
    points: list[dict[str, object]] = []
    for branch, sign in enumerate((1.0, -1.0)):
        geometry = EllipseGeometry(0.0, 0.0, a, axis_ratio, sign * theta)
        for side, values in (
            ("upper", np.linspace(0.06, 0.22, 9)),
            ("lower", np.linspace(np.pi + 0.06, np.pi + 0.22, 9)),
        ):
            for value in values:
                qx, qy = geometry.point(value)
                points.append(
                    {
                        "point_id": len(points),
                        "qx": float(qx),
                        "qy": float(qy),
                        "branch_id": branch,
                        "side": side,
                        "arc_id": branch * 2 + int(side == "lower"),
                        "accepted": True,
                        "localization_sigma_q": 0.001,
                        "arc_t_min": float(values[0]),
                        "arc_t_max": float(values[-1]),
                    }
                )
    return points


def test_observed_radius_seed_replaces_chord_length_guess_on_tip_arcs() -> None:
    a_true = 0.72
    points = _tip_arc_points(a_true, 0.02, np.radians(17.0))
    xy = np.asarray([[row["qx"], row["qy"]] for row in points], dtype=float)
    parameters = default_ellipse_parameters(center=(0.0, 0.0), a=1.6, axis_ratio=0.1, theta=0.4)
    parameters["cx"] = parameters["cx"].copy(vary=False, name="cx")
    parameters["cy"] = parameters["cy"].copy(vary=False, name="cy")
    parameters["a"] = parameters["a"].copy(value=1.6, min=float(np.finfo(float).eps), max=None, name="a")

    seeded = _observed_radius_seed(xy, parameters)

    assert seeded["a"].value == pytest.approx(a_true, rel=0.12)
    assert seeded["a"].max is not None
    assert seeded["a"].max == pytest.approx(1.03 * seeded["a"].value, rel=0.05)
    assert seeded["a"].min == pytest.approx(0.97 * seeded["a"].value, rel=0.05)


def test_observed_radius_seed_preserves_caller_finite_a_max() -> None:
    points = _tip_arc_points(0.72, 0.02, np.radians(17.0))
    xy = np.asarray([[row["qx"], row["qy"]] for row in points], dtype=float)
    parameters = default_ellipse_parameters(center=(0.0, 0.0), a=0.8, axis_ratio=0.1, theta=0.3)
    parameters["cx"] = parameters["cx"].copy(vary=False, name="cx")
    parameters["cy"] = parameters["cy"].copy(vary=False, name="cy")
    parameters["a"] = parameters["a"].copy(value=0.8, min=0.2, max=3.0, name="a")

    seeded = _observed_radius_seed(xy, parameters)

    assert seeded["a"].min == pytest.approx(0.2)
    assert seeded["a"].max == pytest.approx(3.0)


def test_fixed_center_tip_arcs_recover_major_axis_not_chord_length() -> None:
    a_true = 0.72
    ratio = 0.02
    theta = float(np.radians(17.0))
    result = fit_arc_ellipses(
        _tip_arc_points(a_true, ratio, theta),
        parameters={
            "cx": {"value": 0.0, "vary": False},
            "cy": {"value": 0.0, "vary": False},
            "a": {"value": 1.55, "min": 1.0e-12},
            "axis_ratio": {"value": 0.12, "min": 0.005, "max": 0.4},
            "theta": {"value": 0.45, "min": 0.0, "max": float(np.pi / 2.0)},
        },
        multistart=5,
        max_nfev=800,
    )
    fit = result["fit"]
    assert fit is not None and fit.success
    assert fit.values["a"] == pytest.approx(a_true, rel=0.12)
    assert fit.values["axis_ratio"] == pytest.approx(ratio, rel=0.25)
    assert fit.values["theta"] == pytest.approx(theta, abs=0.08)


def test_infeasible_support_penalty_matches_domain_arc_distance() -> None:
    geometry = EllipseGeometry(0.0, 0.0, 1.0, 0.2, 0.0)
    point = np.asarray([1.8, 0.4], dtype=float)
    rectangle = {
        "frame_origin_q": [1.8, 0.4],
        "tangent_q": [1.0, 0.0],
        "normal_q": [0.0, 1.0],
        "tangent_bounds": [-1.0e-4, 1.0e-4],
        "normal_bounds": [-1.0e-4, 1.0e-4],
        "component_id": 0,
        "segment_index": 0,
    }

    result = _project_point_to_support(point, geometry, "upper", None, [rectangle])

    domain_lo, domain_hi = _side_domain("upper")
    local_u, local_v = _local_coordinates(point.reshape(1, 2), geometry)
    expected = _project_ellipse_arc(
        local_u,
        local_v,
        geometry.a,
        geometry.b,
        np.asarray([domain_lo]),
        np.asarray([domain_hi]),
    )
    assert result["support_infeasible"] is True
    assert result["support_violation_q"] == pytest.approx(float(expected["distance"][0]))
    assert result["objective_distance"] == pytest.approx(float(expected["distance"][0]))


def test_flat_ellipse_specs_cap_near_circular_and_runaway_major_axis() -> None:
    settings = validate_analysis_settings(
        {
            "ridge_method": "butterfly_curvature",
            "ellipse_preset": "flat_ellipse",
            "ellipse": {"preset": "flat_ellipse"},
            "q_min": 0.05,
            "q_max": 0.80,
        }
    )
    specs = ellipse_parameter_specs(settings, q_window=(0.05, 0.80))
    assert specs is not None
    assert specs["axis_ratio"]["max"] == pytest.approx(0.35)
    assert specs["cx"]["vary"] is False
    assert specs["a"].get("max") in {None}

    points: list[dict[str, object]] = []
    for branch, sign in enumerate((1.0, -1.0)):
        geometry = EllipseGeometry(0.0, 0.0, 0.10, 0.90, sign * 0.25)
        for side, values in (
            ("upper", np.linspace(0.30, 0.85, 8)),
            ("lower", np.linspace(3.30, 3.85, 8)),
        ):
            for value in values:
                qx, qy = geometry.point(value)
                points.append(
                    {
                        "point_id": len(points),
                        "qx": float(qx),
                        "qy": float(qy),
                        "branch_id": branch,
                        "side": side,
                        "arc_id": branch * 2 + int(side == "lower"),
                        "accepted": True,
                        "localization_sigma_q": 0.001,
                        "arc_t_min": float(values[0]),
                        "arc_t_max": float(values[-1]),
                    }
                )

    result = fit_arc_ellipses(points, parameters=specs, multistart=3, max_nfev=200)
    fit = result["fit"]
    assert fit is not None
    assert fit.values["axis_ratio"] <= 0.35 + 1e-12
    assert fit.values["a"] < 1.0


def test_unreachable_support_rectangle_skips_interval_enumeration() -> None:
    geometry = EllipseGeometry(0.0, 0.0, 1.0, 0.2, 0.0)
    rectangle = {
        "frame_origin_q": [8.0, 0.0],
        "tangent_q": [1.0, 0.0],
        "normal_q": [0.0, 1.0],
        "tangent_bounds": [-1.0e-3, 1.0e-3],
        "normal_bounds": [-1.0e-3, 1.0e-3],
        "component_id": 0,
        "segment_index": 0,
    }
    assert _rectangle_reachable(geometry, rectangle) is False
    assert _rectangle_intervals(geometry, rectangle, [(0.0, float(np.pi))]) == []
