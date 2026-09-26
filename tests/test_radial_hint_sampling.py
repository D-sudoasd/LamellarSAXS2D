from __future__ import annotations

import warnings

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter1d

from butterfly_saxs.benchmark_t2 import generate_case
from butterfly_saxs.butterfly_ridge import (
    _finite_scale_stability_summary,
    _first_order_q_hint,
)


def _independent_local_annular_peak(
    q: np.ndarray,
    intensity: np.ndarray,
    excluded: np.ndarray,
    q_center: float,
) -> float:
    """Measure a known local ring without calling the application hint code."""

    q_min, q_max = 0.76 * q_center, 1.24 * q_center
    valid = (
        ~excluded
        & np.isfinite(q)
        & np.isfinite(intensity)
        & (q >= q_min)
        & (q <= q_max)
    )
    edges = np.linspace(q_min, q_max, 122)
    counts, _ = np.histogram(q[valid], bins=edges)
    sums, _ = np.histogram(q[valid], bins=edges, weights=intensity[valid])
    populated = counts > 0
    assert np.count_nonzero(populated) >= 3
    profile = np.divide(sums, counts, out=np.zeros_like(sums), where=populated)
    profile = np.interp(
        np.arange(profile.size), np.flatnonzero(populated), profile[populated]
    )
    smooth = gaussian_filter1d(profile, sigma=1.2, mode="nearest")
    index = int(np.argmax(smooth))
    delta = 0.0
    if 0 < index < smooth.size - 1:
        left, center, right = smooth[index - 1 : index + 2]
        curvature = left - 2.0 * center + right
        if curvature < 0.0:
            delta = float(np.clip(0.5 * (left - right) / curvature, -1.0, 1.0))
    return float(0.5 * (edges[index] + edges[index + 1]) + delta * (edges[1] - edges[0]))


@pytest.fixture(scope="module")
def butterfly_case() -> dict[str, object]:
    return generate_case("butterfly", shape=(128, 128), noise_sigma=0.005)


def test_sparse_detector_grid_hint_recovers_the_noise_free_local_ring(
    butterfly_case: dict[str, object],
) -> None:
    q = np.asarray(butterfly_case["q"], dtype=float)
    excluded = np.asarray(butterfly_case["mask"], dtype=bool)
    clean = np.asarray(butterfly_case["intensity_noiseless"], dtype=float)
    q0 = 2.0 * np.pi / float(butterfly_case["structure_truth"]["layer_spacing_nm"])
    local_peak = _independent_local_annular_peak(q, clean, excluded, q0)

    assert local_peak == pytest.approx(0.5274, abs=0.003)
    for key in ("intensity_noiseless", "intensity"):
        image = np.asarray(butterfly_case[key], dtype=float)
        hint = _first_order_q_hint(
            q,
            image,
            np.isfinite(image) & ~excluded,
            0.25,
            0.95,
        )
        assert hint["selection_status"] == "selected"
        assert hint["q_star"] == pytest.approx(local_peak, abs=0.02)
        assert hint["radial_q_pixel_step"] == pytest.approx(0.049, abs=0.003)
        assert hint["profile_smoothing_sigma_bins"] > 1.0
        assert hint["significance_prominence_noise_multiple"] == 3.0
        assert hint["profile_support_method"] == "q_pixel_resolution_smoothed_annular_bin_counts"


def test_sampling_smoothing_does_not_promote_a_single_detector_hot_pixel(
    butterfly_case: dict[str, object],
) -> None:
    q = np.asarray(butterfly_case["q"], dtype=float)
    excluded = np.asarray(butterfly_case["mask"], dtype=bool)
    image = np.zeros_like(q)
    pixel = np.unravel_index(int(np.argmin(np.abs(q - 0.53))), q.shape)
    image[pixel] = 1.0

    hint = _first_order_q_hint(q, image, np.isfinite(q) & ~excluded, 0.25, 0.95)

    assert hint["selection_status"] == "no_hint"
    assert hint["q_star"] is None
    assert hint["reason"] == "outlier_dominated_peak_only"
    assert hint["candidate_peaks"]
    assert all(peak["support_status"] == "outlier_dominated" for peak in hint["candidate_peaks"])


def test_scale_stability_summary_handles_all_nonfinite_values_without_warning() -> None:
    points = [
        {"scale_stability": float("nan")},
        {"scale_stability": float("inf")},
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        summary = _finite_scale_stability_summary(points)

    assert summary == {"mean": None, "min": None, "n_finite": 0, "n_candidates": 2}
