from __future__ import annotations

import numpy as np
import pytest

from butterfly_saxs.ellipse import (
    EllipseGeometry,
    ellipse_geometric_residuals,
    ellipse_sampson_residuals,
    ellipse_points,
    fit_ellipse,
    fit_symmetric_ellipses,
    symmetric_ellipse_points,
)
from butterfly_saxs.parameters import ParameterSpec, default_ellipse_parameters


def test_single_ellipse_recovers_independent_synthetic_q_points() -> None:
    truth = default_ellipse_parameters(center=(0.12, -0.08), a=1.8, axis_ratio=0.62, theta=0.31)
    phi = np.linspace(0.0, 2.0 * np.pi, 96, endpoint=False)
    points = ellipse_points(truth, phi)
    result = fit_ellipse(points, residual="geometric", max_nfev=800)

    assert result.success
    assert result.values["cx"] == pytest.approx(0.12, abs=1e-5)
    assert result.values["cy"] == pytest.approx(-0.08, abs=1e-5)
    assert result.values["a"] == pytest.approx(1.8, rel=1e-5)
    assert result.values["axis_ratio"] == pytest.approx(0.62, rel=1e-5)
    assert result.values["theta_deg"] == pytest.approx(np.degrees(0.31), rel=1e-5)
    assert np.max(np.abs(result.residuals)) < 1e-5
    assert result.covariance is not None
    assert result.coverage.angular_coverage > 0.95


def test_fixed_parameter_and_sampson_residual() -> None:
    truth = default_ellipse_parameters(center=(0.0, 0.0), a=2.0, axis_ratio=0.5, theta=0.2)
    points = ellipse_points(truth, np.linspace(0, 2 * np.pi, 80, endpoint=False))
    assert np.max(np.abs(ellipse_sampson_residuals(points, truth))) < 1e-10
    assert np.max(np.abs(ellipse_geometric_residuals(points, truth))) < 1e-10
    result = fit_ellipse(
        points,
        {
            "cx": {"value": 0.0, "vary": False},
            "cy": {"value": 0.0, "vary": False},
            "theta_deg": {"value": np.degrees(0.2), "vary": False},
        },
    )
    assert result.success
    assert result.values["cx"] == 0.0
    assert result.values["cy"] == 0.0
    assert result.bound_flags["cx"] is False
    assert np.isnan(result.stderr["cx"])


def test_shared_symmetric_double_ellipse_recovers_pair() -> None:
    truth = default_ellipse_parameters(center=(0.15, 0.03), a=2.1, axis_ratio=0.68, theta=0.27)
    phi = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    plus, minus = symmetric_ellipse_points(truth, phi)
    points = np.vstack((plus, minus))
    labels = np.r_[np.zeros(len(phi), dtype=int), np.ones(len(phi), dtype=int)]
    result = fit_symmetric_ellipses(points, labels=labels, residual="sampson", max_nfev=800)

    assert result.success
    assert result.values["cx"] == pytest.approx(0.15, abs=1e-5)
    assert result.values["cy"] == pytest.approx(0.03, abs=1e-5)
    assert result.values["a"] == pytest.approx(2.1, rel=1e-5)
    assert result.values["axis_ratio"] == pytest.approx(0.68, rel=1e-5)
    assert result.values["theta"] == pytest.approx(0.27, rel=1e-5)
    assert result.coverage.components == 2


def test_symmetric_multistart_is_deterministic_and_auditable() -> None:
    truth = default_ellipse_parameters(center=(0.1, -0.05), a=1.8, axis_ratio=0.58, theta=0.34)
    phi = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
    plus, minus = symmetric_ellipse_points(truth, phi)
    points = np.vstack((plus, minus))
    labels = np.r_[np.zeros(len(phi), dtype=int), np.ones(len(phi), dtype=int)]

    first = fit_symmetric_ellipses(points, labels=labels, max_nfev=600)
    second = fit_symmetric_ellipses(points, labels=labels, max_nfev=600)

    assert first.multistart_count == 7
    assert len(first.candidate_solutions) == 7
    assert first.selected_start_index == second.selected_start_index
    assert first.candidate_solutions == second.candidate_solutions
    assert first.values == pytest.approx(second.values)
    np.testing.assert_array_equal(first.branch_assignment, labels)


def test_symmetric_multistart_recovers_from_poor_initial_guess() -> None:
    truth = default_ellipse_parameters(center=(0.0, 0.0), a=2.0, axis_ratio=0.45, theta=0.4)
    phi = np.r_[np.linspace(-0.6, 0.6, 30), np.linspace(2.5, 3.7, 30), np.linspace(5.6, 6.8, 30)]
    plus, minus = symmetric_ellipse_points(truth, phi)
    points = np.vstack((plus, minus))
    bad_initial = {
        "cx": ParameterSpec(0.0, min=-5.0, max=5.0),
        "cy": ParameterSpec(0.0, min=-5.0, max=5.0),
        "a": ParameterSpec(0.2, min=0.05, max=5.0),
        "axis_ratio": ParameterSpec(0.95, min=0.05, max=1.0),
        "theta": ParameterSpec(1.4, min=0.0, max=np.pi / 2.0),
    }

    result = fit_symmetric_ellipses(points, parameters=bad_initial, max_nfev=300)

    assert result.success
    assert result.selected_start_index == 1
    assert result.cost < 1e-10
    assert result.values["a"] == pytest.approx(2.0, rel=1e-5)
    assert result.values["axis_ratio"] == pytest.approx(0.45, rel=1e-5)


def test_symmetric_branch_assignment_is_nearest_branch_without_labels() -> None:
    truth = default_ellipse_parameters(center=(0.0, 0.0), a=1.7, axis_ratio=0.62, theta=0.25)
    phi = np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False)
    plus, minus = symmetric_ellipse_points(truth, phi)
    points = np.vstack((plus, minus))

    result = fit_symmetric_ellipses(points, multistart=1, max_nfev=500)

    assert result.branch_assignment is not None
    assert result.branch_assignment.shape == (points.shape[0],)
    assert set(np.unique(result.branch_assignment)) <= {0, 1}
    assert np.count_nonzero(result.branch_assignment == 0) > 0
    assert np.count_nonzero(result.branch_assignment == 1) > 0


def test_symmetric_pair_uses_nonnegative_canonical_tilt() -> None:
    """The pair +/-theta has no second solution at negative theta."""

    truth = default_ellipse_parameters(center=(0.0, 0.0), a=1.7, axis_ratio=0.55, theta=0.24)
    phi = np.linspace(0, 2 * np.pi, 80, endpoint=False)
    plus, minus = symmetric_ellipse_points(truth, phi)
    result = fit_symmetric_ellipses(
        np.vstack((plus, minus)),
        parameters={"theta_deg": {"value": -10.0, "min": -90.0, "max": 90.0}},
        max_nfev=800,
    )

    assert result.success
    assert 0.0 <= result.values["theta"] <= np.pi / 2
    assert result.values["theta"] == pytest.approx(0.24, rel=1e-4)
    assert result.parameters["theta"].min == pytest.approx(0.0)


def test_invalid_geometry_is_rejected() -> None:
    with pytest.raises(ValueError):
        EllipseGeometry(0.0, 0.0, 0.0, 0.5)
    with pytest.raises(ValueError):
        default_ellipse_parameters(a=1.0, axis_ratio=1.1)


def test_grubb_ellipticity_is_eccentricity_not_one_minus_axis_ratio() -> None:
    geometry = EllipseGeometry(cx=0.0, cy=0.0, a=2.0, axis_ratio=0.5, theta=0.0)
    assert geometry.eccentricity == pytest.approx(np.sqrt(0.75))
    assert geometry.ellipticity == pytest.approx(geometry.eccentricity)
