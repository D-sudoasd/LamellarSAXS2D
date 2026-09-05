from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import minimize_scalar
from types import SimpleNamespace

from butterfly_saxs.ellipse import (
    EllipseGeometry,
    ellipse_geometric_residuals,
    fit_symmetric_ellipses,
)
from butterfly_saxs.parameters import ParameterSpec
from butterfly_saxs.p4_quality import evaluate_p4_ellipse_quality


def _distance_oracle(point: np.ndarray, geometry: EllipseGeometry) -> float:
    """Independent dense-plus-scalar global projection oracle."""

    phi = np.linspace(-np.pi, np.pi, 8192, endpoint=False)
    curve = geometry.point(phi)
    squared = np.sum((curve - point[None, :]) ** 2, axis=1)
    index = int(np.argmin(squared))
    step = 2.0 * np.pi / phi.size
    centre = float(phi[index])
    result = minimize_scalar(
        lambda value: float(np.sum((geometry.point(value) - point) ** 2)),
        bounds=(centre - step, centre + step),
        method="bounded",
        options={"xatol": 1.0e-14, "maxiter": 200},
    )
    return float(np.sqrt(max(0.0, result.fun)))


@pytest.mark.parametrize("axis_ratio", (0.02, 0.05, 0.10))
def test_geometric_residual_matches_global_oracle_for_flat_ellipse(axis_ratio: float) -> None:
    geometry = EllipseGeometry(cx=0.13, cy=-0.08, a=1.0, axis_ratio=axis_ratio, theta=0.37)
    local_points = np.asarray(
        (
            (0.10, 0.0),
            (0.90, 0.001),
            (0.0, 0.0),
            (0.25, 0.75 * axis_ratio),
            (-1.2, 0.1),
        ),
        dtype=float,
    )
    cosine, sine = np.cos(geometry.theta), np.sin(geometry.theta)
    points = np.column_stack(
        (
            geometry.cx + cosine * local_points[:, 0] - sine * local_points[:, 1],
            geometry.cy + sine * local_points[:, 0] + cosine * local_points[:, 1],
        )
    )
    actual = np.abs(ellipse_geometric_residuals(points, geometry))
    expected = np.asarray([_distance_oracle(point, geometry) for point in points])
    np.testing.assert_allclose(actual, expected, rtol=2.0e-6, atol=2.0e-8)


def test_flat_fit_uses_fixed_center_and_recovers_partial_noisy_arcs() -> None:
    rng = np.random.default_rng(20260905)
    phi = np.concatenate(
        (
            np.linspace(-0.8, 0.8, 35),
            np.linspace(1.2, 1.9, 25),
            np.linspace(2.5, 3.3, 35),
        )
    )
    centre = np.asarray((0.0, 0.0))
    theta = 0.37
    for axis_ratio in (0.02, 0.05, 0.10):
        a = 1.25
        branches = []
        for sign in (1.0, -1.0):
            geometry = EllipseGeometry(0.0, 0.0, a, axis_ratio, sign * theta)
            branches.append(geometry.point(phi))
        points = np.vstack(branches) + rng.normal(0.0, 1.0e-4, (2 * phi.size, 2))
        parameters = {
            "cx": ParameterSpec(centre[0], vary=False),
            "cy": ParameterSpec(centre[1], vary=False),
            "a": ParameterSpec(1.0, min=0.5, max=2.0),
            "axis_ratio": ParameterSpec(0.2, min=0.005, max=0.5),
            "theta": ParameterSpec(0.7, min=0.0, max=np.pi / 2.0),
        }
        result = fit_symmetric_ellipses(
            points,
            parameters=parameters,
            residual="geometric",
            multistart=3,
            max_nfev=160,
        )
        assert result.success
        assert result.values["cx"] == pytest.approx(0.0, abs=1.0e-12)
        assert result.values["cy"] == pytest.approx(0.0, abs=1.0e-12)
        assert result.values["a"] == pytest.approx(a, rel=2.0e-3)
        assert result.values["axis_ratio"] == pytest.approx(axis_ratio, rel=2.0e-2, abs=2.0e-4)
        assert result.values["theta"] == pytest.approx(theta, abs=2.0e-3)


def test_flat_parameter_multistart_preserves_tied_and_fixed_specs() -> None:
    phi = np.linspace(-0.7, 0.7, 40)
    a, ratio, theta = 1.1, 0.02, 0.28
    branches = [
        EllipseGeometry(0.0, 0.0, a, ratio, sign * theta).point(phi)
        for sign in (1.0, -1.0)
    ]
    parameters = {
        "cx": {"value": 0.0, "vary": False},
        "cy": {"value": 0.0, "vary": False},
        "a": {"value": a, "vary": False},
        "axis_ratio": {"value": ratio, "min": 0.005, "max": 0.5},
        "b": {"value": a * ratio, "expr": "a*axis_ratio"},
        "theta": {"value": theta, "min": 0.0, "max": np.pi / 2.0},
    }
    result = fit_symmetric_ellipses(
        np.vstack(branches),
        parameters=parameters,
        residual="geometric",
        multistart=5,
        max_nfev=120,
    )
    assert result.success
    assert result.parameters["a"].is_fixed
    assert result.parameters["b"].is_tied
    assert result.values["b"] == pytest.approx(result.values["a"] * result.values["axis_ratio"])
    assert result.values["axis_ratio"] == pytest.approx(ratio, abs=1.0e-8)
    assert result.multistart_count == 5
    assert len(result.candidate_solutions) == 5


def test_p4_flags_short_arc_extrapolation_and_bound_saturation() -> None:
    points = [
        SimpleNamespace(
            qx=0.1 * np.cos(angle),
            qy=0.1 * np.sin(angle),
            q=0.1,
            valid=True,
            radial_fwhm=0.01,
            snr=8.0,
            score=8.0,
            trajectory_id=0,
        )
        for angle in np.linspace(-0.2, 0.2, 12)
    ]
    ridge = SimpleNamespace(
        points=points,
        valid_fraction=1.0,
        continuity_fraction=1.0,
        continuity_score=1.0,
        q_unit="nm^-1",
    )
    ellipse = SimpleNamespace(
        a=1.0,
        b=0.02,
        theta=0.2,
        axes_ratio=0.02,
        rmse=0.01,
        success=True,
        condition_number=10.0,
        coverage=SimpleNamespace(angular_coverage=0.1, angular_span=0.4),
        branch_counts=(6, 6),
        bound_flags={"a": True},
        candidate_solutions=(),
        multistart_count=3,
        cx=0.0,
        cy=0.0,
        reference_axis_deg=0.0,
    )
    quality = evaluate_p4_ellipse_quality(ridge, ellipse)
    assert "short_arc" in quality["flags"]
    assert "major_axis_extrapolated" in quality["flags"]
    assert "flat_ellipse_nonidentifiable" in quality["flags"]
    assert "bound_saturation" in quality["flags"]
    assert quality["scientific_status"] == "NOT_ACCEPTED"
