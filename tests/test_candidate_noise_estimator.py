from __future__ import annotations

import numpy as np
import pytest

from butterfly_saxs.butterfly_ridge import (
    CANDIDATE_NOISE_MIN_SAFE_SAMPLES,
    _normalise_options,
    _scaled_surface_field,
    trace_butterfly_ridges,
)


def _synthetic_ring(shape: int = 256):
    axis = np.linspace(-0.8, 0.8, shape)
    qx, qy = np.meshgrid(axis, axis)
    q = np.hypot(qx, qy)
    valid = (q >= 0.2) & (q <= 0.5)
    detector_mask = np.zeros(q.shape, dtype=bool)
    detector_mask[:, shape // 2 - 2 : shape // 2 + 2] = True
    valid &= ~detector_mask
    rng = np.random.default_rng(20260912)
    image = 12.0 + 25.0 * np.exp(-0.5 * ((q - 0.34) / 0.085) ** 2)
    image += rng.normal(0.0, 0.8, q.shape)
    return image, qx, qy, q, valid, detector_mask


def test_masked_highpass_mad_tracks_known_noise_and_is_scale_invariant() -> None:
    image, qx, qy, _q, valid, _detector_mask = _synthetic_ring()
    options = _normalise_options({"candidate_snr_min": 3.0, "support_min": 0.78})

    for sigma in (1.2, 2.2, 3.6):
        field = _scaled_surface_field(image, qx, qy, valid, sigma=sigma, options=options)
        scaled = _scaled_surface_field(7.0 + 3.0 * image, qx, qy, valid, sigma=sigma, options=options)
        masked_values_changed = image.copy()
        masked_values_changed[~valid] = 1e8
        masked = _scaled_surface_field(
            masked_values_changed, qx, qy, valid, sigma=sigma, options=options
        )

        assert field.noise_status == "ok"
        assert field.noise == pytest.approx(0.8, abs=0.32)
        assert field.noise_support_count >= CANDIDATE_NOISE_MIN_SAFE_SAMPLES
        assert 0.0 < field.noise_support_fraction < 1.0
        assert scaled.noise == pytest.approx(3.0 * field.noise, rel=1e-10)
        assert scaled.structural_spread == pytest.approx(3.0 * field.structural_spread, rel=1e-10)
        assert scaled.height_scale == pytest.approx(3.0 * field.height_scale, rel=1e-10)
        assert np.array_equal(field.candidate, scaled.candidate)
        assert np.array_equal(field.accepted, scaled.accepted)
        assert masked.noise == pytest.approx(field.noise, rel=1e-12)
        assert masked.noise_support_count == field.noise_support_count


def test_zero_highpass_mad_fails_closed_instead_of_using_structural_spread() -> None:
    axis = np.linspace(-0.8, 0.8, 128)
    qx, qy = np.meshgrid(axis, axis)
    q = np.hypot(qx, qy)
    valid = (q >= 0.2) & (q <= 0.5)
    image = np.full(q.shape, 12.0)
    options = _normalise_options({"candidate_snr_min": 3.0, "support_min": 0.78})

    field = _scaled_surface_field(image, qx, qy, valid, sigma=1.2, options=options)

    assert field.noise_status == "degenerate_highpass_mad"
    assert np.isinf(field.noise)
    assert field.noise_support_count >= CANDIDATE_NOISE_MIN_SAFE_SAMPLES
    assert not np.any(field.candidate)
    assert not np.any(field.accepted)


def test_trace_diagnostics_report_candidate_noise_and_unchanged_three_sigma_gate() -> None:
    image, qx, qy, _q, _valid, detector_mask = _synthetic_ring(shape=128)

    trace = trace_butterfly_ridges(
        image,
        {"qx": qx, "qy": qy},
        (0.2, 0.5),
        mask=detector_mask,
        options={"smoothing_scales": (1.2,), "candidate_snr_min": 3.0, "run_wang_check": False},
    )

    noise = trace["diagnostics"]["candidate_noise"]
    estimate = noise["estimates_by_scale"][0]
    assert noise["method"] == "mask_normalized_raw_minus_smoothed_mad"
    assert noise["threshold_basis"] == "mask-normalized raw-minus-smoothed high-pass MAD"
    assert noise["minimum_safe_samples"] == CANDIDATE_NOISE_MIN_SAFE_SAMPLES
    assert estimate["candidate_snr_min"] == 3.0
    assert estimate["status"] == "ok"
    assert estimate["noise_sample_count"] >= CANDIDATE_NOISE_MIN_SAFE_SAMPLES
    assert estimate["noise_threshold_delta"] == pytest.approx(3.0 * estimate["noise_sigma"])
