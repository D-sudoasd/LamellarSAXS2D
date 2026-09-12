from __future__ import annotations

import numpy as np
import pytest

from butterfly_saxs.butterfly_ridge import _first_order_q_hint


def _profile_samples(
    profile: np.ndarray,
    q_min: float,
    q_max: float,
    *,
    counts: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edges = np.linspace(q_min, q_max, profile.size + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    sample_counts = np.full(profile.size, 12, dtype=int) if counts is None else counts
    q = np.repeat(centres, sample_counts)
    intensity = np.repeat(profile, sample_counts)
    return q, intensity, np.ones(q.shape, dtype=bool)


def test_significant_inner_ring_remains_ahead_of_a_brighter_harmonic() -> None:
    q_min, q_max = 0.05, 0.80
    edges = np.linspace(q_min, q_max, 97)
    centres = 0.5 * (edges[:-1] + edges[1:])
    profile = (
        8.0 * np.exp(-0.5 * ((centres - 0.10) / 0.012) ** 2)
        + 22.0 * np.exp(-0.5 * ((centres - 0.40) / 0.020) ** 2)
    )
    q, intensity, valid = _profile_samples(profile, q_min, q_max)

    hint = _first_order_q_hint(q, intensity, valid, q_min, q_max)

    assert hint["selection_status"] == "selected"
    assert hint["q_star"] == pytest.approx(0.10, abs=0.02)
    assert hint["band"][1] < 0.22
    assert hint["selection_method"] == "lowest_significant_supported_radial_peak_family"
    assert hint["minimum_bin_count"] == 8
    assert hint["significance_prominence_noise_multiple"] == 3.0
    assert hint["selected_peak"]["prominence_noise_ratio"] >= 3.0


def test_weak_lower_fundamental_blocks_automatic_promotion_of_bright_harmonic() -> None:
    q_min, q_max = 0.05, 0.80
    edges = np.linspace(q_min, q_max, 97)
    centres = 0.5 * (edges[:-1] + edges[1:])
    rng = np.random.default_rng(29)
    profile = (
        6.0 * np.exp(-centres / 0.05)
        + 1.2 * np.exp(-0.5 * ((centres - 0.10) / 0.02) ** 2)
        + 8.0 * np.exp(-0.5 * ((centres - 0.40) / 0.02) ** 2)
        + rng.normal(0.0, 0.20, centres.size)
    )
    q, intensity, valid = _profile_samples(profile, q_min, q_max)

    hint = _first_order_q_hint(q, intensity, valid, q_min, q_max)

    assert hint["selection_status"] == "ambiguous"
    assert hint["reason"] == "ambiguous_lower_q_peak_family"
    assert hint["q_star"] is None
    assert hint["band"] is None
    unresolved = hint["ambiguous_peaks"]
    assert unresolved
    assert min(peak["q_bin"] for peak in unresolved) == pytest.approx(0.10078125)
    assert all(
        hint["ambiguity_prominence_noise_multiple"]
        <= peak["prominence_noise_ratio"]
        < hint["significance_prominence_noise_multiple"]
        and peak["width_bins"] >= hint["ambiguity_min_width_bins"]
        for peak in unresolved
    )
    assert hint["n_significant_peaks"] > 0  # the higher-q harmonic is detectable


def test_prominent_inner_ring_is_not_dropped_by_its_absolute_height() -> None:
    q_min, q_max = 0.05, 0.80
    edges = np.linspace(q_min, q_max, 97)
    centres = 0.5 * (edges[:-1] + edges[1:])
    rng = np.random.default_rng(8)
    profile = (
        5.0
        + 15.0 * np.exp(-0.5 * ((centres - 0.10) / 0.018) ** 2)
        + 100.0 * np.exp(-0.5 * ((centres - 0.40) / 0.020) ** 2)
        + rng.normal(0.0, 0.8, centres.size)
    )
    q, intensity, valid = _profile_samples(
        profile, q_min, q_max, counts=np.full(centres.size, 100, dtype=int)
    )

    hint = _first_order_q_hint(q, intensity, valid, q_min, q_max)

    assert hint["selection_status"] == "selected"
    assert hint["q_star"] == pytest.approx(0.10, abs=0.02)
    assert hint["selected_peak"]["prominence_noise_ratio"] >= 3.0
    assert hint["selected_peak"]["height"] < 0.25 * max(profile)


def test_monotone_profile_and_isolated_supported_spike_do_not_fabricate_peak() -> None:
    q_min, q_max = 0.05, 0.80
    edges = np.linspace(q_min, q_max, 97)
    centres = 0.5 * (edges[:-1] + edges[1:])
    profile = np.linspace(1.0, 10.0, centres.size)
    sample_counts = np.full(centres.size, 12, dtype=int)
    isolated_index = 55
    sample_counts[isolated_index - 1] = 0
    sample_counts[isolated_index + 1] = 0
    profile[isolated_index] = 100.0
    q, intensity, valid = _profile_samples(
        profile, q_min, q_max, counts=sample_counts
    )

    hint = _first_order_q_hint(q, intensity, valid, q_min, q_max)

    assert hint["selection_status"] == "no_hint"
    assert hint["reason"] == "no_supported_local_peak"
    assert hint["q_star"] is None
    assert hint["band"] is None
    assert hint["n_supported_runs"] == 2
    assert hint["candidate_peaks"] == []


def test_under_supported_radial_profiles_cannot_borrow_noise_from_other_bins() -> None:
    q_min, q_max = 0.05, 0.80
    sparse_profile = np.zeros(96, dtype=float)
    sparse_profile[45:48] = [0.0, 100.0, 0.0]
    sparse_counts = np.zeros(96, dtype=int)
    sparse_counts[45:48] = 100
    q, intensity, valid = _profile_samples(
        sparse_profile, q_min, q_max, counts=sparse_counts
    )
    sparse_hint = _first_order_q_hint(q, intensity, valid, q_min, q_max)
    assert sparse_hint["selection_status"] == "no_hint"
    assert sparse_hint["reason"] == "sparse_radial_profile"
    assert sparse_hint["n_supported_bins"] == 3
    assert sparse_hint["q_star"] is None
    assert sparse_hint["band"] is None

    edges = np.linspace(q_min, q_max, 97)
    centres = 0.5 * (edges[:-1] + edges[1:])
    profile = np.linspace(1.0, 10.0, centres.size)
    profile[:3] = [0.0, 100.0, 0.0]
    sample_counts = np.full(centres.size, 12, dtype=int)
    sample_counts[:3] = 100
    sample_counts[3] = 0
    q, intensity, valid = _profile_samples(
        profile, q_min, q_max, counts=sample_counts
    )

    short_run_hint = _first_order_q_hint(q, intensity, valid, q_min, q_max)

    assert short_run_hint["n_supported_bins"] >= 8
    assert short_run_hint["minimum_contiguous_run_bins"] == 8
    assert short_run_hint["n_supported_runs"] == 1
    assert short_run_hint["selection_status"] == "no_hint"
    assert short_run_hint["reason"] == "no_supported_local_peak"
    assert short_run_hint["candidate_peaks"] == []


def test_single_detector_outlier_cannot_define_a_radial_peak() -> None:
    q_min, q_max = 0.05, 0.80
    edges = np.linspace(q_min, q_max, 97)
    centres = 0.5 * (edges[:-1] + edges[1:])
    q = np.tile(centres, (12, 1))
    image = np.tile(np.linspace(1.0, 10.0, centres.size), (12, 1))
    image[0, 45] = 1e6

    hint = _first_order_q_hint(q, image, np.ones(q.shape, dtype=bool), q_min, q_max)

    assert hint["selection_status"] == "no_hint"
    assert hint["reason"] == "outlier_dominated_peak_only"
    assert hint["q_star"] is None
    assert hint["band"] is None
    dominated = [peak for peak in hint["candidate_peaks"] if peak["support_status"] == "outlier_dominated"]
    assert len(dominated) == 1
    assert dominated[0]["q_bin"] == pytest.approx(centres[45])
    assert dominated[0]["effective_samples"] < 2.0


def test_narrow_radial_ring_with_many_contributors_remains_eligible() -> None:
    q_min, q_max = 0.05, 0.80
    edges = np.linspace(q_min, q_max, 97)
    centres = 0.5 * (edges[:-1] + edges[1:])
    profile = 5.0 + 20.0 * np.exp(-0.5 * ((centres - 0.10) / 0.0035) ** 2)
    q, intensity, valid = _profile_samples(
        profile, q_min, q_max, counts=np.full(centres.size, 12, dtype=int)
    )

    hint = _first_order_q_hint(q, intensity, valid, q_min, q_max)

    assert hint["selection_status"] == "selected"
    assert hint["q_star"] == pytest.approx(0.10, abs=0.02)
    assert hint["selected_peak"]["width_bins"] < 2.0
    assert hint["selected_peak"]["effective_samples"] == pytest.approx(12.0)
