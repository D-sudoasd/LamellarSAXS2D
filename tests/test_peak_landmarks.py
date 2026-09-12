from __future__ import annotations

import json
import numpy as np
import pytest

from butterfly_saxs.peak_landmarks import compute_peak_landmarks


def _q_map(
    shape: tuple[int, int] = (161, 161),
    *,
    step: float = 0.003,
    reversed_x: bool = False,
    warped: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = np.indices(shape, dtype=np.float64)
    row_center = 0.5 * (shape[0] - 1)
    col_center = 0.5 * (shape[1] - 1)
    sign_x = -1.0 if reversed_x else 1.0
    qx = sign_x * (cols - col_center) * step
    qy = (rows - row_center) * step
    if warped:
        qx = qx + 0.00045 * np.sin(rows / 13.0) + 0.0002 * np.cos(cols / 7.0)
        qy = qy + 0.00035 * np.sin(cols / 15.0) - 0.00015 * np.cos(rows / 9.0)
    return qx, qy


def _lobes(
    qx: np.ndarray,
    qy: np.ndarray,
    centers: tuple[float, ...] = (25.0, 115.0, 205.0, 295.0),
    *,
    radius: float = 0.22,
    radial_sigma: float = 0.025,
    angular_sigma_deg: float = 13.0,
    amplitude: float = 20.0,
    baseline: float = 2.0,
) -> np.ndarray:
    q_radius = np.hypot(qx, qy)
    chi = np.mod(np.degrees(np.arctan2(qy, qx)), 360.0)
    image = np.full(qx.shape, baseline, dtype=np.float64)
    for center in centers:
        angular_distance = np.abs((chi - center + 180.0) % 360.0 - 180.0)
        image += (
            amplitude
            * np.exp(-0.5 * ((q_radius - radius) / radial_sigma) ** 2)
            * np.exp(-0.5 * (angular_distance / angular_sigma_deg) ** 2)
        )
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
        q_unit="nm^-1",
        **kwargs,
    )


def test_recovers_four_broad_lobes_at_actual_detector_pixels() -> None:
    qx, qy = _q_map()
    observed = _lobes(qx, qy)
    observed += np.random.default_rng(8).normal(0.0, 0.04, observed.shape)
    original = observed.copy()
    qx_original, qy_original = qx.copy(), qy.copy()

    result = _analyze(observed, qx, qy)

    assert result["schema_version"] == "peak-landmarks-v2"
    assert result["method_version"] == "supported-angular-lobes-v2"
    assert result["status"] == "ok"
    assert [peak["peak_id"] for peak in result["peaks"]] == ["P1", "P2", "P3", "P4"]
    for expected, peak in zip((25.0, 115.0, 205.0, 295.0), result["peaks"], strict=True):
        assert abs((peak["angular_peak_deg"] - expected + 180.0) % 360.0 - 180.0) <= 4.0
        assert peak["q"] == pytest.approx(0.22, abs=0.025)
        assert peak["qx"] == qx[peak["pixel_y"], peak["pixel_x"]]
        assert peak["qy"] == qy[peak["pixel_y"], peak["pixel_x"]]
        assert peak["raw_intensity"] == observed[peak["pixel_y"], peak["pixel_x"]]
        assert peak["smoothed_intensity"] >= peak["raw_intensity"] - 1.0
        assert peak["support_pixel_count"] >= 5
    assert result["domain"]["signal_q_window_source"] == "explicit_signal_q_window"
    assert result["raw_global_max"]["interpretation"] == "raw_maximum_only_not_an_identified_reflection"
    assert result["raw_global_max"]["label"] == "G"
    assert result["profiles"]["angular"]["angle_deg"]
    assert result["profiles"]["radial"]["q"]
    angular = result["profiles"]["angular"]
    np.testing.assert_allclose(
        np.asarray(angular["intensity_detection"], dtype=float),
        np.asarray(angular["intensity_smoothed"], dtype=float)
        - np.asarray(angular["intensity_isotropic_reference"], dtype=float),
        equal_nan=True,
    )
    assert result["noise"]["pixel_highpass_sigma"] >= 0.0
    np.testing.assert_array_equal(observed, original)
    np.testing.assert_array_equal(qx, qx_original)
    np.testing.assert_array_equal(qy, qy_original)
    json.dumps(result, allow_nan=False)


def test_missing_lobe_is_not_completed_or_mirrored() -> None:
    qx, qy = _q_map()
    observed = _lobes(qx, qy, (20.0, 130.0, 250.0))

    result = _analyze(observed, qx, qy)

    assert result["peak_count"] == 3
    assert len(result["peaks"]) == 3
    assert [peak["peak_id"] for peak in result["peaks"]] == ["P1", "P2", "P3"]
    angles = [peak["angular_peak_deg"] for peak in result["peaks"]]
    assert all(abs((angle - expected + 180.0) % 360.0 - 180.0) < 4.0 for angle, expected in zip(angles, (20.0, 130.0, 250.0), strict=True))


def test_noise_only_frame_has_no_supported_angular_lobes() -> None:
    qx, qy = _q_map()
    observed = 5.0 + np.random.default_rng(981).normal(0.0, 0.15, qx.shape)

    result = _analyze(observed, qx, qy)

    assert result["status"] == "no_supported_lobes"
    assert result["peaks"] == []
    assert "no_supported_angular_lobes_found" in result["flags"]
    assert all(not candidate["accepted"] for candidate in result["candidate_diagnostics"])


def test_central_tail_and_masked_beamstop_hot_pixel_do_not_create_lobes() -> None:
    qx, qy = _q_map()
    radius = np.hypot(qx, qy)
    observed = _lobes(qx, qy) + 180.0 * np.exp(-0.5 * (radius / 0.035) ** 2)
    center = (observed.shape[0] // 2, observed.shape[1] // 2)
    observed[center] = 1.0e9
    valid = np.ones(observed.shape, dtype=bool)
    valid[center] = False
    raw_window = (0.001, 0.34)
    result = compute_peak_landmarks(
        observed,
        qx,
        qy,
        valid_mask=valid,
        q_window=raw_window,
        signal_q_window=(0.15, 0.29),
        q_unit="nm^-1",
    )

    assert result["peak_count"] == 4
    assert result["raw_global_max"]["raw_intensity"] < 1.0e9
    assert (result["raw_global_max"]["pixel_y"], result["raw_global_max"]["pixel_x"]) != center
    assert result["domain"]["signal_q_window"] == [0.15, 0.29]
    assert all(0.15 <= peak["q"] <= 0.29 for peak in result["peaks"])


def test_valid_raw_hot_pixel_is_preserved_outside_lobe_search_band() -> None:
    qx, qy = _q_map()
    observed = _lobes(qx, qy)
    row, col = (observed.shape[0] // 2, observed.shape[1] // 2 + 10)
    observed[row, col] = 1.0e6
    result = compute_peak_landmarks(
        observed,
        qx,
        qy,
        q_window=(0.0, 0.34),
        signal_q_window=(0.15, 0.29),
    )

    assert result["raw_global_max"]["pixel_x"] == col
    assert result["raw_global_max"]["pixel_y"] == row
    assert result["raw_global_max"]["raw_intensity"] == 1.0e6
    assert "raw_maximum_only_not_an_identified_reflection" == result["raw_global_max"]["interpretation"]
    assert result["peak_count"] == 4


def test_reversed_warped_q_map_uses_coordinates_at_selected_pixels() -> None:
    qx, qy = _q_map(reversed_x=True, warped=True)
    observed = _lobes(qx, qy, (33.0, 123.0, 213.0, 303.0))
    result = _analyze(observed, qx, qy)

    assert result["peak_count"] == 4
    for expected, peak in zip((33.0, 123.0, 213.0, 303.0), result["peaks"], strict=True):
        assert abs((peak["angular_peak_deg"] - expected + 180.0) % 360.0 - 180.0) < 5.0
        row, col = peak["pixel_y"], peak["pixel_x"]
        assert peak["qx"] == qx[row, col]
        assert peak["qy"] == qy[row, col]
        assert peak["chi_deg"] == pytest.approx(
            np.mod(np.degrees(np.arctan2(qy[row, col], qx[row, col])), 360.0)
        )


def test_affine_intensity_scale_preserves_locations_and_model_delta_q() -> None:
    qx, qy = _q_map()
    observed = _lobes(qx, qy)
    model = _lobes(qx, qy, radius=0.231)
    first = _analyze(observed, qx, qy, model=model)
    scaled = _analyze(7.0 * observed - 12.0, qx, qy)

    assert [
        (peak["pixel_x"], peak["pixel_y"]) for peak in first["peaks"]
    ] == [(peak["pixel_x"], peak["pixel_y"]) for peak in scaled["peaks"]]
    assert all(peak["model_peak"] is not None for peak in first["peaks"])
    assert all(peak["delta_q"] == pytest.approx(0.011, abs=0.006) for peak in first["peaks"])
    assert first["model_comparison"]["supplied"] is True
    assert first["model_comparison"]["paired_lobe_count"] == 4
    assert "2D q-vector displacement magnitude" in first["model_comparison"]["delta_q_definition"]


def test_all_invalid_data_returns_empty_result_without_mutation() -> None:
    qx, qy = _q_map((31, 35))
    observed = np.full(qx.shape, np.nan)
    valid = np.ones(qx.shape, dtype=bool)
    before = valid.copy()

    result = compute_peak_landmarks(observed, qx, qy, valid_mask=valid)

    assert result["status"] == "no_valid_pixels"
    assert result["raw_global_max"] is None
    assert result["peaks"] == []
    assert result["domain"]["raw_search_pixel_count"] == 0
    np.testing.assert_array_equal(valid, before)


def test_signal_window_disjoint_from_user_window_fails_closed() -> None:
    qx, qy = _q_map()
    observed = _lobes(qx, qy)

    result = compute_peak_landmarks(
        observed,
        qx,
        qy,
        q_window=(0.08, 0.20),
        signal_q_window=(0.22, 0.30),
    )

    assert result["status"] == "no_supported_lobes"
    assert result["peaks"] == []
    assert result["domain"]["effective_signal_q_window"] is None


def test_boundary_maximum_is_retained_and_flagged() -> None:
    qx, qy = _q_map((101, 101))
    observed = _lobes(qx, qy, (35.0, 125.0, 215.0, 305.0))
    observed[0, 50] = 1.0e6

    result = compute_peak_landmarks(
        observed,
        qx,
        qy,
        q_window=(0.08, 0.34),
        signal_q_window=(0.15, 0.29),
    )

    assert result["raw_global_max"]["pixel_y"] == 0
    assert "detector_boundary" in result["raw_global_max"]["flags"]
    assert "raw_global_maximum_at_detector_boundary" in result["flags"]


def test_wrong_shapes_and_non_boolean_mask_are_rejected() -> None:
    qx, qy = _q_map((21, 23))
    observed = np.ones(qx.shape)
    with pytest.raises(ValueError, match="same shape"):
        compute_peak_landmarks(observed, qx[:-1], qy)
    with pytest.raises(TypeError, match="boolean"):
        compute_peak_landmarks(observed, qx, qy, valid_mask=np.ones(qx.shape, dtype=int))
