from __future__ import annotations

import numpy as np
import pytest

from butterfly_saxs.observables import (
    AngularSpectrum,
    LobeMetrics,
    RidgePoint,
    apparent_lamellar_tilt,
    ellipse_radius,
    fit_symmetric_double_ellipse,
    measure_angular_spectrum,
    measure_four_lobe_peaks,
    measure_observables,
    measure_radial_ridges,
    RidgeTrack,
    _annotate_ridge_continuity,
    _write_fit_branch_ids,
)
from butterfly_saxs.synthetic import SyntheticFrame, SyntheticQMap, make_butterfly_sequence


def test_angular_spectrum_reports_mask_coverage_without_mirror_completion():
    y, x = np.mgrid[-24:24, -24:24]
    qmap = SyntheticQMap(x / 24.0, y / 24.0)
    data = np.ones_like(x, dtype=float)
    mask = np.zeros_like(data, dtype=bool)
    mask[:, :12] = True
    spectrum = measure_angular_spectrum(SyntheticFrame(data, mask), qmap, (0.25, 1.1), n_bins=72)
    assert 0.0 < spectrum.global_coverage < 1.0
    assert np.nanmin(spectrum.coverage) < np.nanmax(spectrum.coverage)
    assert np.nanmax(spectrum.coverage) <= 1.0


def test_phi_app_is_measured_from_lobes_relative_to_draw_axis_not_ellipse_theta():
    angles_deg = [70.0, 110.0, 250.0, 290.0]
    lobes = [{"angle": np.deg2rad(value), "valid": True} for value in angles_deg]
    phi, spread = apparent_lamellar_tilt(lobes, draw_axis_deg=90.0)
    assert phi == pytest.approx(20.0)
    assert spread == pytest.approx(0.0, abs=1e-12)


def test_four_lobes_and_ridges_use_real_observed_pixels():
    sequence = make_butterfly_sequence(
        1,
        shape=(96, 96),
        parameters={
            "a": 0.78,
            "b": 0.55,
            "theta": 0.18,
            "lobe_angle": 0.48,
            "angular_width": 0.12,
            "radial_width": 0.035,
            "amplitude": 8.0,
            "background": 0.1,
        },
        seed=2,
    )
    frame, qmap = sequence.frames[0], sequence.qmaps[0]
    spectrum = measure_angular_spectrum(frame, qmap, (0.35, 1.0), n_bins=180)
    lobes = measure_four_lobe_peaks(spectrum, snr_threshold=2.0)
    assert len(lobes) == 4
    angles = np.asarray([lobe.angle for lobe in lobes])
    ridge = measure_radial_ridges(frame, qmap, (0.35, 1.0), angles=angles, n_bins=128, sector_width=0.16)
    assert len(ridge.points) == 4
    assert all(point.source == "observed" for point in ridge.points)
    assert all(point.q_star > 0 and np.isclose(point.Ln, 2.0 * np.pi / point.q_star) for point in ridge.points)
    assert all(np.isclose(point.q_star_Ainv, point.q_star_nm_inv / 10.0) for point in ridge.points)
    assert all(np.isclose(point.Ln_nm, point.Ln) for point in ridge.points)


def test_ellipse_fit_recovers_known_apparent_geometry():
    from butterfly_saxs.observables import RidgePoint, ellipse_radius

    angles = np.linspace(-np.pi, np.pi, 80, endpoint=False)
    a, b, theta = 0.82, 0.51, 0.23
    q = np.where(np.sin(angles) >= 0, ellipse_radius(angles, a, b, theta), ellipse_radius(angles, a, b, -theta))
    points = [RidgePoint(float(phi), float(radius), float(radius), 2.0 * np.pi / float(radius), 1.0, 0.0, 20.0, 0.03, np.nan, 1.0, 1.0, 1) for phi, radius in zip(angles, q)]
    result = fit_symmetric_double_ellipse(points)
    assert result.success
    assert np.isclose(result.a, a, atol=0.02)
    assert np.isclose(result.b, b, atol=0.02)
    assert np.isclose(abs(result.theta), theta, atol=0.03)
    assert "apparent_geometry_only" in result.flags
    assert "nonunique_inverse_problem" in result.flags


def test_observable_ellipse_adapter_exposes_canonical_diagnostics_and_units():
    angles = np.linspace(-np.pi, np.pi, 96, endpoint=False)
    a, b, theta = 0.82, 0.51, 0.23
    q = np.where(
        np.sin(angles) >= 0,
        ellipse_radius(angles, a, b, theta),
        ellipse_radius(angles, a, b, -theta),
    )
    points = [
        RidgePoint(float(phi), float(radius), float(radius), 2.0 * np.pi / float(radius), 1.0, 0.0, 20.0, 0.03, np.nan, 1.0, 1.0, 1)
        for phi, radius in zip(angles, q)
    ]

    result = fit_symmetric_double_ellipse(
        points,
        initial={"a": 0.8, "b": 0.5, "theta_deg": np.degrees(0.2)},
    )

    assert result.success
    assert result.center == pytest.approx((0.0, 0.0), abs=1e-5)
    assert result.ellipses[0].center == pytest.approx(result.center, abs=1e-12)
    assert result.eccentricity == pytest.approx(np.sqrt(1.0 - (result.b / result.a) ** 2))
    assert result.ellipticity == pytest.approx(result.eccentricity)
    assert result.covariance is not None
    assert np.isfinite(result.condition_number)
    assert result.coverage.angular_coverage > 0.8
    assert set(("cx", "cy", "a", "axis_ratio", "theta")).issubset(result.bound_flags)
    assert "theta_deg" in result.stderr
    assert result.branch_counts == (48, 48)


def test_observable_ellipse_adapter_accepts_fixed_degree_parameters():
    angles = np.linspace(-np.pi, np.pi, 64, endpoint=False)
    a, b, theta = 0.8, 0.5, 0.18
    q = np.where(np.sin(angles) >= 0, ellipse_radius(angles, a, b, theta), ellipse_radius(angles, a, b, -theta))
    points = [
        RidgePoint(float(phi), float(radius), float(radius), 2.0 * np.pi / float(radius), 1.0, 0.0, 20.0, 0.03, np.nan, 1.0, 1.0, 1)
        for phi, radius in zip(angles, q)
    ]
    result = fit_symmetric_double_ellipse(
        points,
        parameters={
            "cx": {"value": 0.0, "vary": False},
            "cy": {"value": 0.0, "vary": False},
            "theta_deg": {"value": np.degrees(theta), "vary": False},
        },
    )
    assert result.success
    assert result.center == pytest.approx((0.0, 0.0), abs=1e-12)
    assert result.theta_deg == pytest.approx(np.degrees(theta), abs=1e-7)
    assert np.isnan(result.stderr["cx"])
    assert np.isnan(result.stderr["theta_deg"])


def test_double_ellipse_reference_axis_and_spacing_outputs() -> None:
    phi = np.linspace(0.0, 2.0 * np.pi, 120, endpoint=False)
    a, b, tilt_deg, reference_deg = 0.82, 0.49, 17.0, 31.0
    tilt = np.deg2rad(tilt_deg)
    reference = np.deg2rad(reference_deg)

    def branch(sign: float) -> np.ndarray:
        c, s = np.cos(sign * tilt), np.sin(sign * tilt)
        x_local = c * a * np.cos(phi) - s * b * np.sin(phi)
        y_local = s * a * np.cos(phi) + c * b * np.sin(phi)
        cr, sr = np.cos(reference), np.sin(reference)
        return np.column_stack((cr * x_local - sr * y_local, sr * x_local + cr * y_local))

    points = np.vstack((branch(1.0), branch(-1.0)))
    labels = np.r_[np.zeros(phi.size, dtype=int), np.ones(phi.size, dtype=int)]
    fit = fit_symmetric_double_ellipse(
        points,
        initial={
            "a": 0.8,
            "axis_ratio": 0.6,
            "theta_deg": 15.0,
            "cx": {"value": 0.0, "vary": False},
            "cy": {"value": 0.0, "vary": False},
        },
        labels=labels,
        reference_axis_deg=reference_deg,
        q_unit="nm^-1",
    )
    assert fit.success
    assert fit.reference_axis_deg == pytest.approx(reference_deg)
    assert fit.ellipse_axis_tilt_deg == pytest.approx(tilt_deg, abs=0.1)
    assert sorted(ellipse.theta_deg for ellipse in fit.ellipses) == pytest.approx(
        sorted((reference_deg - tilt_deg, reference_deg + tilt_deg)), abs=0.1
    )
    assert fit.Ln_from_minor_axis_nm == pytest.approx(2.0 * np.pi / b, rel=2e-3)
    expected_qz = a * b / np.sqrt(
        (b * np.cos(np.pi / 2 - tilt)) ** 2 + (a * np.sin(np.pi / 2 - tilt)) ** 2
    )
    assert fit.Lz_from_draw_axis_nm == pytest.approx(2.0 * np.pi / expected_qz, rel=2e-3)


def test_wraparound_lobe_area_uses_periodic_coordinates() -> None:
    angle = np.linspace(-np.pi, np.pi, 720, endpoint=False)
    distance = np.angle(np.exp(1j * (angle - (np.pi - 0.015))))
    sigma = 0.08
    intensity = 0.2 + 4.0 * np.exp(-0.5 * (distance / sigma) ** 2)
    spectrum = AngularSpectrum(
        angle=angle,
        intensity=intensity,
        counts=np.ones(angle.size, dtype=int),
        candidate_counts=np.ones(angle.size, dtype=int),
        coverage=np.ones(angle.size),
        q_min=0.2,
        q_max=1.0,
        q_center=0.6,
    )
    peak = measure_four_lobe_peaks(spectrum, expected=1, snr_threshold=0.1)[0]
    assert 0.1 < peak.area < 2.0


def test_symmetric_lobe_refinement_recovers_coarse_four_lobe_angle() -> None:
    angle = np.linspace(-np.pi, np.pi, 90, endpoint=False)
    phi = np.deg2rad(23.4)
    sigma = np.deg2rad(6.0)
    centres = np.asarray((phi, -phi, np.pi + phi, np.pi - phi))
    centres = np.angle(np.exp(1j * centres))
    intensity = np.full(angle.size, 0.2)
    for index, centre in enumerate(centres):
        distance = np.angle(np.exp(1j * (angle - centre)))
        intensity += (4.0 + 0.3 * index) * np.exp(-0.5 * (distance / sigma) ** 2)
    intensity += 0.03 * np.sin(7.0 * angle)
    spectrum = AngularSpectrum(
        angle=angle,
        intensity=intensity,
        counts=np.full(angle.size, 20, dtype=int),
        candidate_counts=np.full(angle.size, 20, dtype=int),
        coverage=np.ones(angle.size),
        q_min=0.2,
        q_max=1.0,
        q_center=0.6,
    )

    lobes = measure_four_lobe_peaks(
        spectrum,
        min_prominence=0.2,
        snr_threshold=0.5,
        symmetric_refine=True,
    )

    assert len(lobes) == 4
    assert all(lobe.valid for lobe in lobes)
    assert all(lobe.refinement == "symmetric_cauchy" for lobe in lobes)
    for expected in centres:
        error = min(abs(np.angle(np.exp(1j * (lobe.angle - expected)))) for lobe in lobes)
        assert np.degrees(error) < 1.0


def test_low_snr_sector_stays_invalid_instead_of_being_mirrored():
    y, x = np.mgrid[-32:32, -32:32]
    q = np.hypot(x, y) / 32.0
    angle = np.arctan2(y, x)
    data = np.full_like(q, 1.0)
    data[(q > 0.45) & (q < 0.55) & (np.abs(np.angle(np.exp(1j * (angle - 0.5)))) < 0.12)] = 1.01
    mask = np.abs(np.angle(np.exp(1j * (angle - 0.5)))) < 0.16
    ridge = measure_radial_ridges(
        SyntheticFrame(data, mask),
        SyntheticQMap(x / 32.0, y / 32.0),
        (0.3, 0.8),
        angles=[0.5, -0.5],
        sector_width=0.2,
        n_bins=64,
        snr_threshold=3.0,
    )
    assert ridge.points[0].valid is False
    assert ridge.points[1].source == "observed"


def test_surface_curvature_uses_2d_hessian_and_subpixel_track():
    axis = np.linspace(-1.05, 1.05, 129)
    qx, qy = np.meshgrid(axis, axis)
    q = np.hypot(qx, qy)
    phi = np.arctan2(qy, qx)
    target = ellipse_radius(phi, 0.78, 0.52, 0.21)
    image = 0.05 + 7.0 * np.exp(-0.5 * ((q - target) / 0.025) ** 2)
    angles = np.linspace(-np.pi, np.pi, 48, endpoint=False)

    curvature = measure_radial_ridges(
        SyntheticFrame(image, None),
        SyntheticQMap(qx, qy),
        (0.35, 0.95),
        angles=angles,
        sector_width=0.10,
        ridge_method="surface_curvature",
        curvature_sigma=1.6,
        snr_threshold=1.0,
    )
    radial = measure_radial_ridges(
        SyntheticFrame(image, None),
        SyntheticQMap(qx, qy),
        (0.35, 0.95),
        angles=angles,
        sector_width=0.10,
        ridge_method="radial_peak",
        n_bins=96,
        snr_threshold=1.0,
    )

    valid = curvature.valid
    assert np.count_nonzero(valid) >= 36
    expected = ellipse_radius(curvature.angles[valid], 0.78, 0.52, 0.21)
    assert np.nanmedian(np.abs(curvature.q[valid] - expected)) < 0.035
    assert "detector_pixel_principal_curvature" in curvature.flags
    assert "surface_curvature_approximation" not in curvature.flags
    assert not np.array_equal(curvature.q[valid], radial.q[valid])
    assert all(point.method == "surface_curvature" for point in curvature.points)


def test_surface_curvature_respects_qmap_valid_mask_and_rejects_plane():
    axis = np.linspace(-1.0, 1.0, 81)
    qx, qy = np.meshgrid(axis, axis)
    valid_mask = np.ones_like(qx, dtype=bool)
    valid_mask[:, 37:44] = False
    qmap = {"qx": qx, "qy": qy, "valid_mask": valid_mask}
    plane = 0.2 + 0.1 * qx + 0.05 * qy
    track = measure_radial_ridges(
        SyntheticFrame(plane, None),
        qmap,
        (0.25, 0.9),
        angles=np.linspace(-np.pi, np.pi, 24, endpoint=False),
        ridge_method="surface_curvature",
        curvature_sigma=1.5,
        snr_threshold=2.0,
    )
    assert np.count_nonzero(track.valid) == 0
    assert np.nanmin(track.coverage) < 1.0


def test_surface_curvature_reports_pixel_to_q_scale_without_changing_pixel_track():
    axis = np.linspace(-1.0, 1.0, 97)
    qx, qy = np.meshgrid(axis, axis)
    q = np.hypot(qx, qy)
    image = np.exp(-0.5 * ((q - 0.58) / 0.025) ** 2)
    angles = np.linspace(-np.pi, np.pi, 24, endpoint=False)

    base = measure_radial_ridges(
        SyntheticFrame(image, None),
        SyntheticQMap(qx, qy),
        (0.35, 0.8),
        angles=angles,
        ridge_method="surface_curvature",
        curvature_sigma=1.5,
        snr_threshold=1.0,
    )
    scaled = measure_radial_ridges(
        SyntheticFrame(image, None),
        SyntheticQMap(2.0 * qx, 2.0 * qy),
        (0.70, 1.6),
        angles=angles,
        ridge_method="surface_curvature",
        curvature_sigma=1.5,
        snr_threshold=1.0,
    )

    paired = [
        (left, right)
        for left, right in zip(base.points, scaled.points)
        if left.valid and right.valid
    ]
    assert len(paired) >= 16
    for left, right in paired:
        assert right.pixel_x == pytest.approx(left.pixel_x, abs=1e-8)
        assert right.pixel_y == pytest.approx(left.pixel_y, abs=1e-8)
        assert right.q_normal_step == pytest.approx(2.0 * left.q_normal_step, rel=1e-6)
        assert np.isfinite(left.q_scale_anisotropy)


def test_ridge_point_spacing_requires_an_explicit_physical_q_unit():
    undeclared = RidgePoint(angle=0.0, q=2.0)
    nm = RidgePoint(angle=0.0, q=2.0, q_unit="nm^-1")
    angstrom = RidgePoint(angle=0.0, q=0.2, q_unit="Å^-1")
    assert nm.q_star_Ainv == pytest.approx(0.2)
    assert nm.q_star_nm_inv == pytest.approx(2.0)
    assert nm.Ln_nm == pytest.approx(np.pi)
    assert angstrom.q_star_Ainv == pytest.approx(0.2)
    assert angstrom.q_star_nm_inv == pytest.approx(2.0)
    assert angstrom.Ln_nm == pytest.approx(np.pi)
    assert np.isnan(undeclared.q_star_Ainv)
    assert np.isnan(undeclared.Ln_nm)

    for unit in ("pixel-q", "unknown"):
        point = RidgePoint(angle=0.0, q=2.0, q_unit=unit)
        assert np.isnan(point.q_star_Ainv)
        assert np.isnan(point.Ln_nm)
        assert "spacing_unavailable_unknown_q_unit" in point.flags


def test_ridge_measurement_propagates_q_unit_and_gates_pixel_q_spacing():
    axis = np.linspace(-1.0, 1.0, 65)
    qx, qy = np.meshgrid(axis, axis)
    q = np.hypot(qx, qy)
    image = np.exp(-0.5 * ((q - 0.55) / 0.025) ** 2)
    qmap = {"qx": qx, "qy": qy, "q": q, "q_unit": "pixel-q"}
    ridge = measure_radial_ridges(
        SyntheticFrame(image, None),
        qmap,
        (0.35, 0.8),
        angles=[0.0],
        n_bins=96,
        snr_threshold=1.0,
    )
    assert ridge.q_unit == "pixel-q"
    assert ridge.points[0].q_unit == "pixel-q"
    assert np.isnan(ridge.points[0].q_star_Ainv)
    assert np.isnan(ridge.points[0].Ln_nm)
    assert "spacing_unavailable_unknown_q_unit" in ridge.points[0].flags


def test_lobe_fwhm_degree_alias_and_phi_app_mad_alias():
    lobe = LobeMetrics(
        angle=0.0,
        intensity=1.0,
        baseline=0.0,
        snr=5.0,
        fwhm=np.deg2rad(12.0),
        area=1.0,
        index=0,
        coverage=1.0,
        n_pixels=10,
    )
    assert lobe.fwhm_deg == pytest.approx(12.0)
    assert lobe.as_dict()["fwhm_deg"] == pytest.approx(12.0)

    sequence = make_butterfly_sequence(1, shape=(48, 48), seed=4)
    result = measure_observables(sequence.frames[0], sequence.qmaps[0], (0.2, 1.0), fit_ellipse=False)
    assert result.phi_app_mad_deg == pytest.approx(result.phi_app_std_deg)
    assert result.as_dict()["phi_app_mad_deg"] == pytest.approx(result.phi_app_std_deg)


def test_physical_spacing_is_not_reported_for_a_shifted_q_ellipse():
    phi = np.linspace(0.0, 2.0 * np.pi, 96, endpoint=False)
    a, b, tilt = 0.82, 0.49, np.deg2rad(17.0)
    shift = np.array([0.08, -0.05])

    def branch(sign: float) -> np.ndarray:
        c, s = np.cos(sign * tilt), np.sin(sign * tilt)
        x = c * a * np.cos(phi) - s * b * np.sin(phi)
        y = s * a * np.cos(phi) + c * b * np.sin(phi)
        return np.column_stack((x, y)) + shift

    points = np.vstack((branch(1.0), branch(-1.0)))
    labels = np.r_[np.zeros(phi.size, dtype=int), np.ones(phi.size, dtype=int)]
    fit = fit_symmetric_double_ellipse(
        points,
        labels=labels,
        q_unit="nm^-1",
        initial={
            "a": 0.8,
            "axis_ratio": 0.6,
            "theta_deg": 15.0,
            "cx": {"value": shift[0], "vary": False},
            "cy": {"value": shift[1], "vary": False},
        },
    )
    assert fit.success
    assert fit.center == pytest.approx(tuple(shift), abs=1e-6)
    assert np.isnan(fit.Ln_from_minor_axis_nm)
    assert np.isnan(fit.Lz_from_draw_axis_nm)
    assert "spacing_unavailable_nonzero_center" in fit.flags


def test_ridge_point_exposes_quality_continuity_and_identity_contract():
    point = RidgePoint(
        angle=0.0,
        q=0.5,
        q_unit="nm^-1",
        score=3.5,
        continuity_score=1.0,
        trajectory_id=2,
        branch_id=1,
        local_q_step=0.01,
    )
    values = point.as_dict()
    assert values["score"] == pytest.approx(3.5)
    assert values["point_score"] == pytest.approx(3.5)
    assert values["continuity_score"] == pytest.approx(1.0)
    assert values["trajectory_id"] == 2
    assert values["branch_id"] == 1
    assert values["local_q_step"] == pytest.approx(0.01)


def test_ridge_continuity_marks_jump_without_inventing_points():
    angles = np.linspace(-np.pi, np.pi, 8, endpoint=False)
    q_values = [0.50, 0.51, 0.52, 0.53, 1.20, 0.55, 0.56, 0.57]
    points = [RidgePoint(float(angle), float(q), valid=True) for angle, q in zip(angles, q_values)]
    valid_fraction, continuity_fraction, continuity_score, flags = _annotate_ridge_continuity(points, angles)
    assert len(points) == 8
    assert valid_fraction == pytest.approx(1.0)
    assert continuity_fraction < 1.0
    assert continuity_score < 1.0
    assert "continuity_jump" in flags
    assert any("continuity_jump" in point.flags for point in points)
    assert len({point.trajectory_id for point in points}) >= 2


def test_radial_continuity_keeps_ring_and_rejects_isolated_peak() -> None:
    axis = np.linspace(-1.0, 1.0, 129)
    qx, qy = np.meshgrid(axis, axis)
    q = np.hypot(qx, qy)
    angle = np.arctan2(qy, qx)
    ring = 4.0 * np.exp(-0.5 * ((q - 0.58) / 0.018) ** 2)
    isolated = 20.0 * (
        (np.abs(q - 0.78) < 0.012)
        & (np.abs(np.angle(np.exp(1j * angle))) < np.deg2rad(1.0))
    )

    track = measure_radial_ridges(
        SyntheticFrame(0.05 + ring + isolated, None),
        SyntheticQMap(qx, qy),
        (0.35, 0.9),
        n_angles=36,
        n_bins=96,
    )
    valid_q = track.q[track.valid]
    assert valid_q.size >= 30
    assert np.nanmedian(np.abs(valid_q - 0.58)) < 0.015
    assert np.nanmax(valid_q) < 0.7
    assert "radial_continuity_tracking" in track.flags

    sparse = measure_radial_ridges(
        SyntheticFrame(0.05 + isolated, None),
        SyntheticQMap(qx, qy),
        (0.35, 0.9),
        n_angles=36,
        n_bins=96,
    )
    assert np.count_nonzero(sparse.valid) == 0
    assert all(point.reason != "accepted" for point in sparse.points)


def test_fit_branch_assignment_is_written_to_valid_measured_points_only():
    points = [
        RidgePoint(0.0, 0.5, valid=True),
        RidgePoint(np.pi / 2.0, float("nan"), valid=False),
        RidgePoint(np.pi, 0.6, valid=True),
    ]
    track = RidgeTrack(
        points=points,
        angles=np.array([0.0, np.pi / 2.0, np.pi]),
        q=np.array([0.5, np.nan, 0.6]),
        valid=np.array([True, False, True]),
        coverage=np.ones(3),
    )

    class Fit:
        branch_assignment = np.array([1, 0])

    _write_fit_branch_ids(track, Fit())
    assert points[0].branch_id == 1
    assert points[1].branch_id is None
    assert points[2].branch_id == 0
