from __future__ import annotations

import numpy as np
import pytest

from butterfly_saxs.sector_peaks import measure_sector_peaks


def _qmap(size: int = 181, step: float = 0.005, *, unit: str = "nm^-1") -> tuple[np.ndarray, dict[str, np.ndarray | str]]:
    coordinates = np.arange(size, dtype=float) - 0.5 * (size - 1)
    qy, qx = np.meshgrid(coordinates * step, coordinates * step, indexing="ij")
    q = np.hypot(qx, qy)
    return q, {"qx": qx, "qy": qy, "q": q, "metadata": {"q_unit": unit}}


def _ring_image(q: np.ndarray, peak: float = 0.30, width: float = 0.012) -> np.ndarray:
    return 0.08 + np.exp(-0.5 * ((q - peak) / width) ** 2)


def _sector(result: dict, angle: float) -> dict:
    sectors = result["sectors"]
    return min(sectors, key=lambda item: abs(((item["angle_deg"] - angle + 180.0) % 360.0) - 180.0))


def test_ring_and_four_lobe_amplitude_measure_one_observed_peak_per_sector() -> None:
    q, qmap = _qmap()
    angles = np.mod(np.degrees(np.arctan2(qmap["qy"], qmap["qx"])), 360.0)
    intensity = _ring_image(q)
    amplitude = 0.25 + 0.75 * np.square(np.cos(np.radians(2.0 * angles)))
    image = intensity * amplitude
    result = measure_sector_peaks(image, qmap, (0.18, 0.42))

    assert result["method_version"].startswith("sector-peaks-v1")
    assert result["q_unit"] == "nm^-1"
    assert result["sector_overlap"]["overlapping"] is True
    for sector in result["sectors"]:
        assert sector["raw_mean"].shape == sector["raw_sum"].shape == sector["raw_count"].shape
        assert sector["geometry_count"].shape == sector["coverage"].shape
    strong = _sector(result, 0.0)
    assert strong["selected_peak"] is not None
    assert strong["selected_peak"]["q_star"] == pytest.approx(0.30, abs=0.018)
    assert strong["selected_peak"]["pixel_x"] is not None
    assert strong["selected_peak"]["pixel_y"] is not None
    assert strong["selected_peak"]["source_pixel_count"] > 0
    assert strong["selected_peak"]["sampling_sigma_basis"].endswith("sampling_resolution_not_CI")
    assert strong["sector_coverage"] == pytest.approx(1.0)


def test_low_q_monotonic_tail_does_not_win_against_supported_outer_peak() -> None:
    q, qmap = _qmap()
    image = 1.8 / np.maximum(q, 0.02) + 1.5 * np.exp(-0.5 * ((q - 0.34) / 0.014) ** 2)
    result = measure_sector_peaks(image, qmap, (0.10, 0.48))
    sector = _sector(result, 0.0)

    assert sector["selected_peak"] is not None
    assert sector["selected_peak"]["q_star"] == pytest.approx(0.34, abs=0.025)
    assert sector["selected_peak"]["peak_bin_index"] > 2


def test_monotonic_and_pure_noise_profiles_are_not_promoted_to_peaks() -> None:
    q, qmap = _qmap()
    monotonic = 2.0 / np.maximum(q, 0.02)
    monotonic_result = measure_sector_peaks(monotonic, qmap, (0.10, 0.48))
    monotonic_sector = _sector(monotonic_result, 0.0)
    assert monotonic_sector["selected_peak"] is None
    assert "boundary" in monotonic_sector["reason"] or "monotonic" in monotonic_sector["reason"]

    rng = np.random.default_rng(1234)
    noisy = 1.0 + 0.02 * rng.normal(size=q.shape)
    noise_result = measure_sector_peaks(noisy, qmap, (0.10, 0.48))
    assert all(sector["selected_peak"] is None for sector in noise_result["sectors"])


def test_two_similar_supported_peaks_remain_ambiguous() -> None:
    q, qmap = _qmap()
    image = (
        0.05
        + 1.0 * np.exp(-0.5 * ((q - 0.25) / 0.012) ** 2)
        + 0.98 * np.exp(-0.5 * ((q - 0.36) / 0.012) ** 2)
    )
    result = measure_sector_peaks(
        image,
        qmap,
        (0.12, 0.46),
        options={"selection_prominence_ratio": 1.25},
    )
    sector = _sector(result, 0.0)

    assert sector["selected_peak"] is None
    assert sector["reason"] == "ambiguous_multiple_peaks"
    usable = [candidate for candidate in sector["candidates"] if candidate["status"] == "ambiguous"]
    assert len(usable) >= 2
    assert {round(candidate["q_star"], 2) for candidate in usable} >= {0.25, 0.36}


def test_noisy_single_hot_pixel_is_rejected_by_effective_source_support() -> None:
    axis = np.linspace(-0.8, 0.8, 161)
    qy, qx = np.meshgrid(axis, axis, indexing="ij")
    q = np.hypot(qx, qy)
    rng = np.random.default_rng(11)
    image = 3.0 + rng.normal(0.0, 0.05, q.shape)
    image[80, 120] = 10_000.0
    result = measure_sector_peaks(image, {"qx": qx, "qy": qy, "q": q}, (0.10, 0.70))

    for angle in (0.0, 5.0, 355.0):
        sector = _sector(result, angle)
        assert sector["selected_peak"] is None
        assert sector["reason"] == "single_pixel_dominated_support"
        assert sector["candidates"]
        candidate = sector["candidates"][0]
        assert candidate["n_eff"] < 2.0
        assert candidate["max_contribution_fraction"] > 0.75


def test_two_negative_hot_pixels_do_not_make_a_positive_between_valleys() -> None:
    axis = np.linspace(-0.8, 0.8, 161)
    qy, qx = np.meshgrid(axis, axis, indexing="ij")
    q = np.hypot(qx, qy)
    rng = np.random.default_rng(11)
    image = 3.0 + rng.normal(0.0, 0.05, q.shape)
    image[80, 110] = -10_000.0
    image[80, 130] = -10_000.0
    result = measure_sector_peaks(image, {"qx": qx, "qy": qy, "q": q}, (0.10, 0.70))

    for angle in (0.0, 5.0, 355.0):
        sector = _sector(result, angle)
        assert sector["selected_peak"] is None
        assert sector["reason"] == "height_below_baseline_noise"
        candidate = sector["candidates"][0]
        assert candidate["prominence_snr"] > 100.0
        assert candidate["height_snr"] < 4.0
        assert candidate["raw_baseline"] > 2.8
        assert np.nanmin(sector["raw_mean"]) < -100.0


def test_mask_gap_is_visible_and_never_smoothed_into_a_peak() -> None:
    q, qmap = _qmap()
    image = _ring_image(q, peak=0.30, width=0.010)
    mask = (q > 0.285) & (q < 0.315)
    result = measure_sector_peaks(image, qmap, (0.18, 0.42), mask=mask)
    sector = _sector(result, 0.0)
    gap = (sector["raw_count"] == 0) & (sector["geometry_count"] > 0)

    assert np.any(gap)
    assert np.all(np.isnan(sector["smoothed_intensity"][gap]))
    assert sector["selected_peak"] is None or not (0.285 < sector["selected_peak"]["q_star"] < 0.315)
    assert np.min(sector["coverage"][gap]) == 0.0


def test_unrelated_masked_quadrant_does_not_remove_a_well_supported_peak() -> None:
    q, qmap = _qmap()
    image = _ring_image(q)
    mask = (qmap["qx"] < 0.0) & (qmap["qy"] > 0.0)
    result = measure_sector_peaks(image, qmap, (0.18, 0.42), mask=mask)
    sector = _sector(result, 0.0)

    assert sector["selected_peak"] is not None
    assert sector["selected_peak"]["n_eff"] >= 2.0
    assert sector["selected_peak"]["max_contribution_fraction"] <= 0.75


def test_boundary_maximum_is_not_reported_without_two_sided_support() -> None:
    q, qmap = _qmap()
    image = np.asarray(q, dtype=float)  # increasing tail: maximum at q-window edge
    result = measure_sector_peaks(image, qmap, (0.10, 0.32))
    sector = _sector(result, 0.0)

    assert sector["selected_peak"] is None
    assert "boundary" in sector["reason"] or "monotonic" in sector["reason"]


def test_wide_plateau_is_retained_as_unresolved_instead_of_getting_a_fake_q() -> None:
    q, qmap = _qmap()
    image = np.where((q > 0.24) & (q < 0.41), 1.0, 0.05)
    result = measure_sector_peaks(image, qmap, (0.10, 0.50))
    sector = _sector(result, 45.0)

    assert sector["selected_peak"] is None
    assert sector["reason"] == "flat_top_unresolved_peak"
    assert any(candidate["reason"] == "flat_top_unresolved_peak" for candidate in sector["candidates"])


def test_wide_gaussian_and_lorentzian_profiles_are_valid_when_both_tails_return() -> None:
    q, qmap = _qmap(size=401)
    qmap["q_unit"] = "nm^-1"
    profiles = (
        0.1 + np.exp(-0.5 * ((q - 0.40) / 0.12) ** 2),
        0.1 + 1.0 / (1.0 + ((q - 0.40) / 0.12) ** 2),
    )

    for image in profiles:
        result = measure_sector_peaks(image, qmap, (0.10, 0.90))
        sector = _sector(result, 0.0)
        assert sector["selected_peak"] is not None
        assert sector["selected_peak"]["q_star"] == pytest.approx(0.40, abs=0.02)
        assert sector["selected_peak"]["radial_fwhm"] > 0.15
        assert sector["selected_peak"]["flat_top_ratio"] < 0.45


def test_positive_peak_on_negative_raw_baseline_is_kept_and_raw_values_remain_signed() -> None:
    q, qmap = _qmap(size=401)
    image = -2.0 + 5.0 * np.exp(-0.5 * ((q - 0.40) / 0.025) ** 2)
    result = measure_sector_peaks(image, qmap, (0.10, 0.90))
    sector = _sector(result, 0.0)
    peak = sector["selected_peak"]

    assert peak is not None
    assert peak["q_star"] == pytest.approx(0.40, abs=0.012)
    assert peak["raw_baseline"] < -1.5
    assert peak["height"] > peak["height_threshold"]
    assert np.nanmin(sector["raw_mean"]) < -1.5


def test_non_affine_qmap_units_and_explicit_bin_cap_are_recorded_without_mutation() -> None:
    q, qmap = _qmap(size=121, step=0.007, unit="A^-1")
    qx_before = qmap["qx"].copy()
    qy_before = qmap["qy"].copy()
    image = _ring_image(q, peak=0.28, width=0.018)
    image_before = image.copy()
    # A smooth non-affine radial coordinate is supplied explicitly.  The
    # sector measurement must use it as data and retain its declared unit.
    qmap["q"] = q * (1.0 + 0.025 * np.sin(qmap["qx"] * 2.0) * np.cos(qmap["qy"] * 3.0))
    qmap_before = qmap["q"].copy()
    result = measure_sector_peaks(
        image,
        qmap,
        (0.12, 0.42),
        options={"radial_bins": 512, "max_radial_bins": 192},
    )

    assert result["q_unit"] == "A^-1"
    assert result["settings"]["effective_radial_bins"] <= 192
    assert result["settings"]["effective_radial_bins"] <= result["settings"]["q_resolution_limited_bins"]
    assert result["sampling"]["representative_q_step"] > 0.0
    np.testing.assert_array_equal(qmap["qx"], qx_before)
    np.testing.assert_array_equal(qmap["qy"], qy_before)
    np.testing.assert_array_equal(qmap["q"], qmap_before)
    np.testing.assert_array_equal(image, image_before)


def test_heterogeneous_local_q_steps_use_conservative_neighbor_upper_bound() -> None:
    size = 161
    coordinate = np.arange(size, dtype=float) - 0.5 * (size - 1)
    qx_profile = np.where(coordinate < 0.0, coordinate * 0.02, coordinate * 0.24)
    qy_profile = coordinate * 0.02
    qx = np.broadcast_to(qx_profile[None, :], (size, size))
    qy = np.broadcast_to(qy_profile[:, None], (size, size))
    q = np.hypot(qx, qy)
    image = np.ones_like(q)
    result = measure_sector_peaks(
        image,
        {"qx": qx, "qy": qy, "q": q, "q_unit": "nm^-1"},
        (0.10, 0.70),
    )

    details = result["sampling"]["q_step_details"]
    assert result["settings"]["representative_q_step"] == pytest.approx(0.24)
    assert details["vector_step_max"] == pytest.approx(0.24)
    assert details["radial_step_max"] == pytest.approx(0.24)
    assert result["sampling"]["effective_radial_bin_width"] >= 0.24
    assert "conservative maximum" in result["settings"]["q_step_source"]


def test_mask_true_excludes_observed_statistics_but_remains_in_geometry_denominator() -> None:
    q, qmap = _qmap(size=101)
    image = _ring_image(q)
    mask = np.zeros_like(q, dtype=bool)
    mask[:, q.shape[1] // 2 :] = True
    result = measure_sector_peaks(image, qmap, (0.18, 0.42), mask=mask)
    right = _sector(result, 0.0)

    assert right["measured_total_count"] < right["geometry_total_count"]
    assert np.nanmin(right["coverage"]) == 0.0
    assert result["input"]["mask_true_is_excluded"] is True


def test_numeric_options_reject_bool_values_instead_of_coercing_them() -> None:
    q, qmap = _qmap()
    image = _ring_image(q)

    with pytest.raises(ValueError, match="sector_width_deg"):
        measure_sector_peaks(image, qmap, (0.18, 0.42), options={"sector_width_deg": True})
    with pytest.raises(ValueError, match="smoothing_sigma_bins"):
        measure_sector_peaks(image, qmap, (0.18, 0.42), options={"smoothing_sigma_bins": np.bool_(False)})
