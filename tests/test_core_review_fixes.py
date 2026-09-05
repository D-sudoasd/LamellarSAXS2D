from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from butterfly_saxs.intensity import (
    default_intensity_parameters,
    double_ellipse_intensity,
    fit_intensity_model,
)
from butterfly_saxs.models import ImageFrame, QMap
from butterfly_saxs.observables import RadialProfile, _radial_peak, measure_radial_ridges
from butterfly_saxs.p4_quality import evaluate_p4_ellipse_quality


def _ring_fixture() -> tuple[ImageFrame, QMap]:
    axis = np.linspace(-1.0, 1.0, 129)
    qx, qy = np.meshgrid(axis, axis)
    q = np.hypot(qx, qy)
    image = 0.1 + 5.0 * np.exp(-0.5 * ((q - 0.58) / 0.018) ** 2)
    image += np.random.default_rng(4).normal(0.0, 0.02, image.shape)
    return ImageFrame(image), QMap(qx, qy, q_unit="nm^-1")


def test_radial_continuity_honors_configured_snr_threshold() -> None:
    frame, qmap = _ring_fixture()
    ordinary = measure_radial_ridges(
        frame,
        qmap,
        (0.35, 0.9),
        n_angles=36,
        n_bins=96,
        ridge_snr_threshold=2.0,
    )
    strict = measure_radial_ridges(
        frame,
        qmap,
        (0.35, 0.9),
        n_angles=36,
        n_bins=96,
        ridge_snr_threshold=1.0e6,
    )
    assert np.count_nonzero(ordinary.valid) > 0
    assert np.count_nonzero(strict.valid) == 0


def test_radial_coverage_gate_uses_selected_peak_neighbourhood() -> None:
    q = np.linspace(0.3, 0.8, 40)
    peak_index = 22
    intensity = 0.1 + 4.0 * np.exp(-0.5 * ((q - q[peak_index]) / 0.025) ** 2)
    candidate_counts = np.full(q.size, 25, dtype=int)
    counts = np.full(q.size, 25, dtype=int)
    counts[peak_index] = 2
    coverage = counts / candidate_counts
    profile = RadialProfile(
        angle=0.4,
        q=q,
        intensity=intensity,
        counts=counts,
        candidate_counts=candidate_counts,
        coverage=coverage,
        q_min=float(q[0]),
        q_max=float(q[-1]),
        q_unit="nm^-1",
    )
    accepted = _radial_peak(profile, snr_threshold=2.0, min_coverage=0.0)
    rejected = _radial_peak(profile, snr_threshold=2.0, min_coverage=0.8)
    assert accepted.valid
    assert accepted.metadata["peak_coverage"] == pytest.approx(2.0 / 25.0)
    assert not rejected.valid
    assert rejected.reason == "low_peak_support"
    assert rejected.metadata["peak_coverage"] == pytest.approx(2.0 / 25.0)
    assert np.isnan(rejected.q_star)


def test_full2d_failure_and_full_objective_selection_are_explicit() -> None:
    axis = np.linspace(-1.0, 1.0, 24)
    qx, qy = np.meshgrid(axis, axis)
    image = double_ellipse_intensity(
        qx,
        qy,
        {"a": 0.8, "b": 0.2, "theta": 0.4, "lobe_angle": 0.5, "angular_width": 0.1, "amplitude": 5.0},
    )
    initial = default_intensity_parameters(a=0.5, axis_ratio=0.5, theta_deg=10.0)
    fixed = {name: name not in {"a", "axis_ratio", "theta"} for name in initial.names}
    initial["a"] = initial["a"].copy(min=0.2, max=1.5)
    initial["axis_ratio"] = initial["axis_ratio"].copy(min=0.02, max=0.8)
    initial["theta"] = initial["theta"].copy(min=-1.2, max=1.2)
    failed = fit_intensity_model(
        {"data": image},
        {"qx": qx, "qy": qy},
        initial=initial,
        fixed=fixed,
        max_pixels=100,
        scales=(1.0,),
        multistart=3,
        max_nfev=1,
    )
    assert not failed.success
    assert "solver_failed" in failed.flags
    assert "all_candidates_failed" in failed.flags
    assert all(not record["success"] for record in failed.candidate_solutions)

    fitted = fit_intensity_model(
        {"data": image},
        {"qx": qx, "qy": qy},
        initial=initial,
        fixed=fixed,
        max_pixels=100,
        scales=(1.0,),
        multistart=3,
        max_nfev=60,
    )
    successful = [
        record
        for record in fitted.candidate_solutions
        if record["success"] and record["scale"] == 1.0
    ]
    assert successful
    assert fitted.full_cost == pytest.approx(
        min(record["full_cost"] for record in successful)
    )
    assert fitted.cost == pytest.approx(fitted.full_cost)
    assert all("sample_cost" in record and "full_cost" in record for record in fitted.candidate_solutions)
    assert fitted.selection_objective == "full_valid_weighted_robust_cost"


def _quality_fixture(*, span: float, extent: float, bound: bool = False):
    points = [
        SimpleNamespace(
            qx=extent * np.cos(angle),
            qy=extent * np.sin(angle),
            q=extent,
            valid=True,
            radial_fwhm=0.01,
            snr=8.0,
            score=8.0,
            trajectory_id=0,
        )
        for angle in np.linspace(-0.25, 0.25, 12)
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
        theta=0.0,
        axes_ratio=0.02,
        rmse=0.001,
        success=True,
        condition_number=10.0,
        coverage=SimpleNamespace(angular_coverage=span / (2.0 * np.pi), angular_span=span),
        branch_counts=(6, 6),
        bound_flags={"a": bound},
        candidate_solutions=(),
        multistart_count=3,
        cx=0.0,
        cy=0.0,
        reference_axis_deg=0.0,
    )
    return ridge, ellipse


def test_p4_short_arc_extent_and_bound_diagnostics_have_explicit_statuses() -> None:
    short_ridge, short_ellipse = _quality_fixture(span=0.4, extent=0.1, bound=True)
    short = evaluate_p4_ellipse_quality(short_ridge, short_ellipse)
    assert short["checks"]
    assert next(check for check in short["checks"] if check["name"] == "short_arc")["status"] == "FAIL"
    assert next(check for check in short["checks"] if check["name"] == "major_axis_extrapolated")["status"] == "WARN"
    assert "flat_ellipse_nonidentifiable" in short["flags"]
    assert "bound_saturation" in short["flags"]

    broad_ridge, broad_ellipse = _quality_fixture(span=2.0 * np.pi, extent=1.0)
    broad = evaluate_p4_ellipse_quality(broad_ridge, broad_ellipse)
    assert next(check for check in broad["checks"] if check["name"] == "short_arc")["status"] == "PASS"
    assert next(check for check in broad["checks"] if check["name"] == "major_axis_extrapolated")["status"] == "PASS"

    middle_ridge, middle_ellipse = _quality_fixture(span=2.5, extent=0.8)
    middle = evaluate_p4_ellipse_quality(middle_ridge, middle_ellipse)
    assert next(check for check in middle["checks"] if check["name"] == "short_arc")["status"] == "WARN"
    assert next(check for check in middle["checks"] if check["name"] == "major_axis_extrapolated")["status"] == "WARN"
    serialized = json.dumps(middle, allow_nan=False)
    assert "flat_ellipse_nonidentifiable" in serialized
