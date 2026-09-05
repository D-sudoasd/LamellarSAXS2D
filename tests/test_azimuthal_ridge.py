from __future__ import annotations

import numpy as np

from butterfly_saxs.models import ImageFrame, QMap
from butterfly_saxs.observables import (
    fit_symmetric_double_ellipse,
    measure_observables,
    measure_radial_ridges,
)


def _butterfly_frame(
    *,
    angle_centres: tuple[float, ...] = (0.42, -0.42, np.pi + 0.42, np.pi - 0.42),
    amplitudes: tuple[float, ...] | None = None,
    noise: float = 0.03,
    seed: int = 20260905,
    mask: np.ndarray | None = None,
) -> tuple[ImageFrame, QMap]:
    axis = np.linspace(-1.0, 1.0, 241)
    qx, qy = np.meshgrid(axis, axis)
    q = np.hypot(qx, qy)
    angle = np.arctan2(qy, qx)
    image = np.full_like(q, 0.15)
    if amplitudes is None:
        amplitudes = tuple(8.0 for _ in angle_centres)
    for centre, amplitude in zip(angle_centres, amplitudes):
        angular_distance = np.angle(np.exp(1j * (angle - centre)))
        image += amplitude * np.exp(-0.5 * (angular_distance / 0.045) ** 2)
    image *= np.exp(-0.5 * ((q - 0.50) / 0.26) ** 2)
    image += np.random.default_rng(seed).normal(0.0, noise, image.shape)
    return ImageFrame(image, mask=mask), QMap(qx, qy, q_unit="nm^-1")


def test_azimuthal_peak_returns_direct_observed_points_with_branch_ids() -> None:
    frame, qmap = _butterfly_frame()
    ridge = measure_radial_ridges(
        frame,
        qmap,
        (0.22, 0.78),
        ridge_method="azimuthal_peak",
        n_bins=24,
        n_angles=144,
        ridge_snr_threshold=3.0,
        ridge_min_peak_fraction=0.25,
    )
    assert ridge.points
    assert all(point.method == "azimuthal_peak" for point in ridge.points)
    assert all(point.source == "observed" for point in ridge.points)
    assert all(point.q_unit == "nm^-1" for point in ridge.points)
    assert all(not np.isfinite(point.q_star) for point in ridge.points)
    assert all(not np.isfinite(point.lamellar_spacing) for point in ridge.points)
    assert all(point.branch_id in (0, 1) for point in ridge.points)
    assert all(point.trajectory_id is not None for point in ridge.points)
    assert set(ridge.flags) >= {"azimuthal_peak_ridge", "observed_angular_maxima"}
    assert all("spacing_unavailable_azimuthal_trajectory" in point.flags for point in ridge.points)
    assert not any("radial_continuity_tracking" == flag for flag in ridge.flags)
    assert all(0.22 <= point.q <= 0.78 for point in ridge.points)


def test_azimuthal_peak_does_not_mirror_a_single_observed_wing() -> None:
    frame, qmap = _butterfly_frame(angle_centres=(0.42,), amplitudes=(8.0,))
    ridge = measure_radial_ridges(
        frame,
        qmap,
        (0.25, 0.75),
        ridge_method="azimuthal_peak",
        n_bins=20,
        n_angles=144,
        ridge_snr_threshold=3.0,
    )
    assert ridge.points
    observed_angles = np.asarray([point.angle for point in ridge.points])
    close_to_observed = np.abs(np.angle(np.exp(1j * (observed_angles - 0.42)))) < 0.12
    assert np.count_nonzero(close_to_observed) >= 0.75 * len(observed_angles)
    assert not any(
        np.abs(np.angle(np.exp(1j * (angle + 0.42)))) < 0.12
        for angle in observed_angles
    )


def test_azimuthal_peak_rejects_mask_gap_boundaries_and_no_signal() -> None:
    axis = np.linspace(-1.0, 1.0, 241)
    qx, qy = np.meshgrid(axis, axis)
    q = np.hypot(qx, qy)
    angle = np.arctan2(qy, qx)
    mask_gap = np.abs(np.angle(np.exp(1j * (angle - 0.42)))) < 0.09
    frame, qmap = _butterfly_frame(mask=mask_gap)
    ridge = measure_radial_ridges(
        frame,
        qmap,
        (0.22, 0.78),
        ridge_method="azimuthal_peak",
        n_bins=24,
        n_angles=144,
        ridge_snr_threshold=3.0,
    )
    assert ridge.points
    assert "masked_gap_or_boundary_peak_rejected" in ridge.flags
    assert not any(
        np.abs(np.angle(np.exp(1j * (point.angle - 0.42)))) < 0.12
        for point in ridge.points
    )

    no_signal = ImageFrame(np.ones_like(q))
    empty = measure_radial_ridges(
        no_signal,
        qmap,
        (0.22, 0.78),
        ridge_method="azimuthal_peak",
        n_bins=24,
        n_angles=144,
        ridge_snr_threshold=5.0,
    )
    assert empty.points == []
    assert "no_azimuthal_peak" in empty.flags


def test_peak_fraction_is_prominence_filter_and_coverage_is_independent() -> None:
    frame, qmap = _butterfly_frame(amplitudes=(8.0, 2.0, 8.0, 2.0), noise=0.02)
    all_peaks = measure_radial_ridges(
        frame,
        qmap,
        (0.25, 0.75),
        ridge_method="azimuthal_peak",
        n_bins=20,
        n_angles=144,
        ridge_snr_threshold=1.0,
        ridge_min_peak_fraction=0.0,
    )
    strong_peaks = measure_radial_ridges(
        frame,
        qmap,
        (0.25, 0.75),
        ridge_method="azimuthal_peak",
        n_bins=20,
        n_angles=144,
        ridge_snr_threshold=1.0,
        ridge_min_peak_fraction=0.6,
    )
    assert len(all_peaks.points) > len(strong_peaks.points)
    assert all(point.support > 0.0 for point in strong_peaks.points)


def test_measure_observables_exposes_ellipse_controls_and_keeps_method_label() -> None:
    frame, qmap = _butterfly_frame()
    result = measure_observables(
        frame,
        qmap,
        (0.25, 0.75),
        fit_ellipse=False,
        ridge_method="azimuthal_peak",
        n_ridge_angles=144,
        n_radial_bins=20,
        ridge_snr_threshold=3.0,
        ridge_min_peak_fraction=0.25,
    )
    assert result.ellipse is None
    assert result.ridge.points
    assert all(point.method == "azimuthal_peak" for point in result.ridge.points)
    assert result.lobe_radial_profiles
    assert result.lobe_radial_peaks
    assert any(point.valid and np.isfinite(point.q_star) for point in result.lobe_radial_peaks)

    ellipse = fit_symmetric_double_ellipse(
        result.ridge,
        initial={"a": {"value": 0.8, "min": 0.2, "max": 2.0},
                 "axis_ratio": {"value": 0.1, "min": 0.005, "max": 0.8},
                 "theta_deg": {"value": 25.0, "min": 0.0, "max": 90.0},
                 "cx": {"value": 0.0, "vary": False},
                 "cy": {"value": 0.0, "vary": False}},
        q_unit="nm^-1",
        multistart=2,
    )
    assert ellipse.success
    assert np.isnan(ellipse.Ln_from_minor_axis_nm)
    assert np.isnan(ellipse.Lz_from_draw_axis_nm)
    assert "spacing_unavailable_azimuthal_trajectory" in ellipse.flags

    strict = measure_observables(
        frame,
        qmap,
        (0.25, 0.75),
        fit_ellipse=False,
        ridge_method="azimuthal_peak",
        n_ridge_angles=144,
        n_radial_bins=20,
        ridge_snr_threshold=1.0e6,
    )
    assert all(
        (not point.valid) and (not np.isfinite(point.q_star))
        for point in strict.lobe_radial_peaks
    )
