from __future__ import annotations

import warnings

import numpy as np
import pytest

from butterfly_saxs.peak_landmarks import compute_peak_landmarks


_NOISE_ONLY_CASES = (
    (20260912, 0.0, False, "full"),
    (20260913, 11.0, False, "sector"),
    (20260914, 30.0, False, "column_strip"),
    (20260915, 37.0, False, "sector_and_column_strip"),
    (20260916, 0.0, False, "full"),
    (20260917, 11.0, True, "sector"),
    (20260918, 30.0, True, "column_strip"),
    (20260919, 37.0, True, "sector_and_column_strip"),
    (20260920, 0.0, True, "full"),
    (20260921, 11.0, True, "sector"),
)


def _q_map(
    size: int,
    *,
    rotation_deg: float = 0.0,
    warped: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    axis = np.linspace(-0.4, 0.4, size, dtype=np.float64)
    qx, qy = np.meshgrid(axis, axis)
    if warped:
        rows, cols = np.indices((size, size), dtype=np.float64)
        qx = qx + 0.0015 * np.sin(rows / 13.0) + 0.0008 * np.cos(cols / 7.0)
        qy = qy + 0.0012 * np.sin(cols / 15.0) - 0.0006 * np.cos(rows / 9.0)
    rotation = np.deg2rad(rotation_deg)
    return (
        qx * np.cos(rotation) - qy * np.sin(rotation),
        qx * np.sin(rotation) + qy * np.cos(rotation),
    )


def _ring(
    qx: np.ndarray,
    qy: np.ndarray,
    centers_deg: tuple[float, ...] = (),
    *,
    amplitude: float = 0.0,
    radius: float = 0.22,
    noise_sigma: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    radial_q = np.hypot(qx, qy)
    angle_deg = np.mod(np.degrees(np.arctan2(qy, qx)), 360.0)
    radial_signal = np.exp(-0.5 * ((radial_q - radius) / 0.025) ** 2)
    image = 2.0 + 20.0 * radial_signal
    for center in centers_deg:
        angular_distance = np.abs((angle_deg - center + 180.0) % 360.0 - 180.0)
        image += amplitude * radial_signal * np.exp(
            -0.5 * (angular_distance / 13.0) ** 2
        )
    if noise_sigma > 0.0:
        image += np.random.default_rng(seed).normal(0.0, noise_sigma, image.shape)
    return image


def _analyze(
    observed: np.ndarray,
    qx: np.ndarray,
    qy: np.ndarray,
    **kwargs: object,
) -> dict:
    return compute_peak_landmarks(
        observed,
        qx,
        qy,
        q_window=(0.08, 0.34),
        signal_q_window=(0.15, 0.29),
        **kwargs,
    )


def _angle_distance(first: float, second: float) -> float:
    return abs((first - second + 180.0) % 360.0 - 180.0)


@pytest.mark.parametrize("amplitude", (0.6, 2.0))
def test_frozen_broad_four_lobe_controls_keep_the_four_sigma_rule(amplitude: float) -> None:
    axis = (np.arange(241, dtype=np.float64) - 120.0) * 0.003
    qx, qy = np.meshgrid(axis, axis)
    radial_q = np.hypot(qx, qy)
    angle_deg = np.mod(np.degrees(np.arctan2(qy, qx)), 360.0)
    shell = np.exp(-0.5 * ((radial_q - 0.22) / 0.025) ** 2)
    angular = np.zeros(qx.shape, dtype=np.float64)
    expected = (25.0, 115.0, 205.0, 295.0)
    for center in expected:
        distance = np.abs((angle_deg - center + 180.0) % 360.0 - 180.0)
        angular += np.exp(-0.5 * (distance / 22.0) ** 2)
    observed = 2.0 + 20.0 * shell + amplitude * shell * angular
    observed += np.random.default_rng(20260912).normal(0.0, 0.04, qx.shape)

    result = _analyze(observed, qx, qy)

    assert result["peak_count"] == 4
    assert result["options"]["minimum_prominence_sigma"] == 4.0
    assert result["options"]["angular_noise_smoothing_sigma_bins"] == 1.0
    for expected_angle, peak in zip(expected, result["peaks"], strict=True):
        assert _angle_distance(peak["angular_peak_deg"], expected_angle) <= 5.0
        assert peak["angular_snr"] >= 4.0


@pytest.mark.parametrize("size", (96, 128, 161, 241, 321))
@pytest.mark.parametrize("rotation_deg", (0.0, 11.0, 30.0))
def test_isotropic_ring_has_no_supported_lobes_across_resolution_and_rotation(
    size: int, rotation_deg: float
) -> None:
    qx, qy = _q_map(size, rotation_deg=rotation_deg)
    result = _analyze(_ring(qx, qy), qx, qy)

    assert result["peak_count"] == 0
    assert result["status"] == "no_supported_lobes"
    assert all(not candidate["accepted"] for candidate in result["candidate_diagnostics"])
    assert result["angular_detection"]["reference_estimator"] == (
        "median_raw_intensity_per_radial_bin"
    )
    assert result["angular_detection"]["radial_reference"]["bin_count"] > 0
    angular = result["profiles"]["angular"]
    measured = np.asarray(angular["intensity_smoothed"], dtype=np.float64)
    corrected = np.asarray(angular["intensity_detection"], dtype=np.float64)
    reference = np.asarray(angular["intensity_isotropic_reference"], dtype=np.float64)
    np.testing.assert_allclose(corrected, measured - reference, equal_nan=True)
    assert np.nanstd(corrected) < np.nanstd(measured)


@pytest.mark.parametrize(("size", "rotation_deg", "seed"), ((161, 11.0, 3), (241, 30.0, 9)))
def test_noisy_warped_masked_isotropic_ring_has_no_supported_lobes(
    size: int, rotation_deg: float, seed: int
) -> None:
    qx, qy = _q_map(size, rotation_deg=rotation_deg, warped=True)
    radial_q = np.hypot(qx, qy)
    valid = np.ones(qx.shape, dtype=bool)
    valid[:, : max(1, size // 9)] = False
    valid[radial_q < 0.045] = False
    angle_deg = np.mod(np.degrees(np.arctan2(qy, qx)), 360.0)
    valid[(angle_deg > 128.0) & (angle_deg < 151.0)] = False
    observed = _ring(qx, qy, noise_sigma=0.05, seed=seed)

    result = _analyze(observed, qx, qy, valid_mask=valid)

    assert result["peak_count"] == 0
    assert result["status"] == "no_supported_lobes"


@pytest.mark.parametrize(
    ("seed", "rotation_deg", "warped", "mask_kind"), _NOISE_ONLY_CASES
)
def test_noise_only_seeds_do_not_gain_lobes_from_the_local_noise_estimator(
    seed: int, rotation_deg: float, warped: bool, mask_kind: str
) -> None:
    qx, qy = _q_map(241, rotation_deg=rotation_deg, warped=warped)
    valid = np.ones(qx.shape, dtype=bool)
    angle_deg = np.mod(np.degrees(np.arctan2(qy, qx)), 360.0)
    if mask_kind in {"sector", "sector_and_column_strip"}:
        valid[(angle_deg > 62.5) & (angle_deg < 97.5)] = False
    if mask_kind in {"column_strip", "sector_and_column_strip"}:
        valid[:, : qx.shape[1] // 9] = False
    observed = 5.0 + np.random.default_rng(seed).normal(0.0, 0.15, qx.shape)

    result = _analyze(observed, qx, qy, valid_mask=valid)

    assert result["peak_count"] == 0
    assert result["status"] == "no_supported_lobes"
    assert all(not candidate["accepted"] for candidate in result["candidate_diagnostics"])


@pytest.mark.parametrize(
    ("centers_deg", "amplitude"),
    (
        ((35.0, 215.0), 4.0),
        ((35.0, 215.0), 8.0),
        ((20.0, 140.0, 270.0), 4.0),
        ((20.0, 140.0, 270.0), 8.0),
        ((25.0, 115.0, 205.0, 295.0), 4.0),
        ((25.0, 115.0, 205.0, 295.0), 8.0),
    ),
)
def test_weak_and_moderate_two_three_and_four_lobes_remain_supported(
    centers_deg: tuple[float, ...], amplitude: float
) -> None:
    qx, qy = _q_map(241, rotation_deg=11.0, warped=True)
    observed = _ring(
        qx,
        qy,
        centers_deg,
        amplitude=amplitude,
        noise_sigma=0.05,
        seed=5,
    )
    model = _ring(qx, qy, centers_deg, amplitude=amplitude, radius=0.231)
    result = _analyze(observed, qx, qy, model=model)

    assert result["peak_count"] == len(centers_deg)
    for expected, peak in zip(sorted(centers_deg), result["peaks"], strict=True):
        assert _angle_distance(peak["angular_peak_deg"], expected) <= 4.0
        row, col = peak["pixel_y"], peak["pixel_x"]
        assert peak["qx"] == qx[row, col]
        assert peak["qy"] == qy[row, col]
        assert peak["raw_intensity"] == observed[row, col]
        assert peak["model_peak"] is not None
        assert peak["delta_q"] == pytest.approx(0.011, abs=0.007)


def test_isotropic_ring_and_single_in_window_hot_pixel_do_not_create_a_lobe() -> None:
    qx, qy = _q_map(241, rotation_deg=11.0, warped=True)
    observed = _ring(qx, qy)
    hot_row, hot_col = np.unravel_index(
        int(np.argmin(np.abs(np.hypot(qx, qy) - 0.22))), qx.shape
    )
    observed[hot_row, hot_col] = 1.0e6

    result = _analyze(observed, qx, qy)

    assert result["peak_count"] == 0
    assert result["raw_global_max"]["pixel_x"] == hot_col
    assert result["raw_global_max"]["pixel_y"] == hot_row
    assert result["raw_global_max"]["raw_intensity"] == 1.0e6
    assert result["noise"]["raw_support_pixel_count_in_signal_band"] == 1
    assert any(
        candidate["reason"] == "insufficient_unfiltered_isotropic_residual_support"
        for candidate in result["candidate_diagnostics"]
    )


def test_start_stop_window_aliases_match_canonical_q_window_pairs() -> None:
    qx, qy = _q_map(161, rotation_deg=11.0)
    observed = _ring(qx, qy, (25.0, 115.0, 205.0), amplitude=8.0)
    canonical = compute_peak_landmarks(
        observed,
        qx,
        qy,
        q_window=(0.08, 0.34),
        signal_q_window=(0.15, 0.29),
    )
    aliases = compute_peak_landmarks(
        observed,
        qx,
        qy,
        q_window={"start": 0.08, "stop": 0.34},
        signal_q_window={"start": 0.15, "stop": 0.29},
    )

    assert aliases == canonical


def test_nonfinite_q_and_observation_samples_are_excluded_from_reference() -> None:
    qx, qy = _q_map(161)
    observed = _ring(qx, qy)
    baseline = _analyze(observed, qx, qy)
    for row in (80, 81, 82):
        assert .15 < np.hypot(qx[row, 120], qy[row, 120]) < .29
    qx[80, 120] = np.nan
    qy[81, 120] = np.inf
    observed[82, 120] = np.nan

    result = _analyze(observed, qx, qy)

    assert result["peak_count"] == 0
    assert result["domain"]["raw_search_pixel_count"] == baseline["domain"]["raw_search_pixel_count"] - 3
    assert result["domain"]["signal_pixel_count"] == baseline["domain"]["signal_pixel_count"] - 3


def test_signal_window_excludes_ring_from_lobes_but_keeps_raw_maximum() -> None:
    qx, qy = _q_map(161)
    observed = _ring(qx, qy)
    result = compute_peak_landmarks(
        observed,
        qx,
        qy,
        q_window=(0.08, 0.34),
        signal_q_window=(0.27, 0.32),
    )

    assert result["peak_count"] == 0
    assert result["raw_global_max"]["q"] == pytest.approx(0.22, abs=0.01)
    assert result["domain"]["effective_signal_q_window"] == [0.27, 0.32]


def test_constant_signal_with_mask_has_no_zero_prominence_candidates_or_warnings() -> None:
    axis = np.linspace(-0.5, 0.5, 41)
    qx, qy = np.meshgrid(axis, axis)
    observed = np.ones(qx.shape, dtype=np.float64)
    observed[20, 30] = 1000.0
    valid = ~((qx >= 0.21) & (qx <= 0.29) & (qy >= -0.04) & (qy <= 0.04))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = compute_peak_landmarks(
            observed,
            qx,
            qy,
            valid_mask=valid,
            q_window=(0.05, 0.45),
        )

    assert result["peak_count"] == 0
    assert result["candidate_diagnostics"] == []
    assert caught == []
