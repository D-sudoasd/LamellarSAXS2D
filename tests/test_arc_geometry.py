from __future__ import annotations

import numpy as np
import pytest

from butterfly_saxs.arc_geometry import fit_arc_ellipses
from butterfly_saxs.ellipse import EllipseGeometry, fit_symmetric_ellipses


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
