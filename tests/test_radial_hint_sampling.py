from __future__ import annotations

import warnings

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter1d

from butterfly_saxs.benchmark_sequence import SequenceSettings, generate_sequence
from butterfly_saxs.benchmark_t2 import generate_case
from butterfly_saxs.butterfly_ridge import (
    _apply_hint_independent_arc_support,
    _finite_scale_stability_summary,
    _first_order_q_hint,
    trace_butterfly_ridges,
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


def _radial_hint_summary(peaks: list[tuple[float, float, float]]) -> dict[str, object]:
    return {
        "q_star": peaks[0][0],
        "band": None,
        "n_bins": 96,
        "selection_status": "selected",
        "reason": "ok",
        "selection_method": "lowest_significant_supported_radial_peak_family",
        "peak_family_span": 1.45,
        "significance_prominence_threshold": 3.0,
        "ambiguity_prominence_floor": 1.0,
        "ambiguity_min_width_bins": 2.0,
        "candidate_peaks": [
            {
                "q_bin": q_peak,
                "height": height,
                "prominence": prominence,
                "prominence_noise_ratio": prominence / 0.5,
                "width_bins": 4.0,
                "effective_samples": 20.0,
                "support_status": "eligible",
            }
            for q_peak, height, prominence in peaks
        ],
    }


def _observed_arc_support(
    q_values: list[float],
    *,
    n_arcs: int,
    points_per_arc: int,
    span_deg: float,
    start_deg: float = 15.0,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    arcs: list[dict[str, object]] = []
    points: list[dict[str, object]] = []
    for q_value in q_values:
        for arc_index in range(n_arcs):
            point_ids = []
            start = start_deg + 90.0 * arc_index
            for point_index, angle_deg in enumerate(
                np.linspace(start, start + span_deg, points_per_arc)
            ):
                point_id = f"q{q_value:.3f}-arc{arc_index}-point{point_index}"
                angle = np.deg2rad(angle_deg)
                points.append(
                    {
                        "point_id": point_id,
                        "qx": float(q_value * np.cos(angle)),
                        "qy": float(q_value * np.sin(angle)),
                    }
                )
                point_ids.append(point_id)
            arcs.append(
                {
                    "arc_id": len(arcs),
                    "point_ids": point_ids,
                    "ordered_point_ids": point_ids,
                    "valid": True,
                    "branch_ids": [arc_index % 2],
                    "side": "upper" if arc_index % 2 == 0 else "lower",
                }
            )
    return arcs, points


def test_clear_observed_arc_support_reconciles_the_512px_low_q_side_peak() -> None:
    frame = generate_sequence(
        SequenceSettings(
            n_signal_frames=2,
            noise_control=False,
            buried_signal_stress=False,
            local_missing_lobe_frame=1,
        )
    )[0]
    result = trace_butterfly_ridges(
        frame["intensity"],
        {"qx": frame["qx"], "qy": frame["qy"], "q": frame["q"]},
        (0.15, 0.85),
        mask=frame["mask"],
    )
    hint = result["diagnostics"]["first_order_q_hint"]
    q0 = float(frame["structural_q0_nm_inv"])
    low_q_peak = min(hint["candidate_peaks"], key=lambda peak: abs(peak["q_bin"] - 0.1755))
    low_family = next(
        family for family in hint["candidate_families"]
        if family["family_id"] == low_q_peak["family_id"]
    )
    main_arc_family = max(
        hint["candidate_families"],
        key=lambda family: family["mean_matched_arc_span_deg"] or 0.0,
    )
    unsupported_lower_family = next(
        family for family in hint["candidate_families"]
        if family["q_max"] < q0 and family["radial_evidence_status"] == "significant"
        and family["n_valid_arcs"] == 0
    )
    accepted_radii = [
        np.hypot(point["qx"], point["qy"])
        for point in result["points"]
        if point.get("accepted")
    ]
    observed_peak = _independent_local_annular_peak(
        frame["q"],
        frame["intensity"],
        frame["mask"],
        q0,
    )
    competing_q = {peak["q_bin"] for peak in hint["competing_peaks"]}

    assert q0 == pytest.approx(0.55116, abs=0.001)
    assert observed_peak == pytest.approx(q0, abs=0.015)
    assert low_q_peak["prominence"] < main_arc_family["max_prominence"]
    assert low_family["weaker_observed_support_than_family_ids"] == [main_arc_family["family_id"]]
    assert unsupported_lower_family["n_valid_arcs"] == 0
    assert hint["selection_status"] == "ambiguous"
    assert hint["q_star"] is None
    assert hint["selection_method"] == "radial_peak_with_hint_independent_observed_arc_support"
    assert hint["order_assignment"] == "provisional"
    assert all(
        peak["q_bin"] in competing_q
        for peak in hint["candidate_peaks"]
        if peak["q_bin"] < main_arc_family["q_min"]
    )
    assert hint["applied"] is False
    assert len(accepted_radii) > 20
    assert np.median(accepted_radii) == pytest.approx(observed_peak, abs=0.01)
    assert result["diagnostics"]["radial_population"]["demoted"] == 0


def test_brighter_higher_order_peak_does_not_replace_a_supported_lower_q_family() -> None:
    lower_q, higher_q = 0.10, 0.40
    arcs, points = _observed_arc_support(
        [lower_q, higher_q], n_arcs=2, points_per_arc=10, span_deg=25.0
    )
    summary = _apply_hint_independent_arc_support(
        _radial_hint_summary([(lower_q, 8.0, 8.0), (higher_q, 25.0, 22.0)]),
        arcs,
        points,
        q_min=0.05,
        q_max=0.80,
    )

    assert summary["selection_status"] == "selected"
    assert summary["q_star"] == pytest.approx(lower_q)
    assert summary["selected_peak"]["height"] < max(
        peak["height"] for peak in summary["candidate_peaks"]
    )
    assert summary["candidate_families"][1]["weaker_observed_support_than_family_ids"] == []


def test_missing_lobe_does_not_make_a_supported_lower_q_ring_lose_to_a_bright_peak() -> None:
    lower_q, higher_q = 0.10, 0.20
    lower_arcs, lower_points = _observed_arc_support(
        [lower_q], n_arcs=3, points_per_arc=9, span_deg=24.0
    )
    higher_arcs, higher_points = _observed_arc_support(
        [higher_q], n_arcs=4, points_per_arc=9, span_deg=24.0
    )
    arcs = lower_arcs + [dict(arc, arc_id=int(arc["arc_id"]) + len(lower_arcs)) for arc in higher_arcs]
    points = lower_points + higher_points
    summary = _apply_hint_independent_arc_support(
        _radial_hint_summary([(lower_q, 8.0, 8.0), (higher_q, 25.0, 22.0)]),
        arcs,
        points,
        q_min=0.05,
        q_max=0.80,
    )

    assert summary["selection_status"] == "selected"
    assert summary["q_star"] == pytest.approx(lower_q)
    assert summary["candidate_families"][0]["n_valid_arcs"] == 3
    assert summary["candidate_families"][1]["n_valid_arcs"] == 4
    assert summary["candidate_families"][0]["weaker_observed_support_than_family_ids"] == []


def test_an_isolated_radial_hit_cannot_promote_a_higher_q_peak() -> None:
    low_q, high_q = 0.10, 0.40
    points = [{"point_id": "low-hit", "qx": low_q, "qy": 0.0}]
    point_ids = ["low-hit"]
    for index, angle_deg in enumerate(np.linspace(20.0, 45.0, 10)):
        angle = np.deg2rad(angle_deg)
        points.append(
            {
                "point_id": f"high-{index}",
                "qx": float(high_q * np.cos(angle)),
                "qy": float(high_q * np.sin(angle)),
            }
        )
        point_ids.append(f"high-{index}")
    arcs = [
        {
            "arc_id": 0,
            "point_ids": point_ids,
            "ordered_point_ids": point_ids,
            "valid": True,
        }
    ]
    summary = _apply_hint_independent_arc_support(
        _radial_hint_summary([(low_q, 8.0, 8.0), (high_q, 25.0, 22.0)]),
        arcs,
        points,
        q_min=0.05,
        q_max=0.80,
    )

    assert summary["selection_status"] == "ambiguous"
    assert summary["reason"] == "lower_q_peak_without_independent_arc_support"
    assert summary["q_star"] is None
    assert summary["candidate_families"][0]["n_valid_arcs"] == 0
    assert summary["candidate_families"][0]["n_isolated_arc_hits"] == 1
    assert summary["candidate_families"][1]["n_valid_arcs"] == 1


def test_masked_lower_q_ring_remains_first_order_ahead_of_a_brighter_higher_ring() -> None:
    shape = (512, 512)
    axis = np.linspace(-0.9, 0.9, shape[0])
    qy, qx = np.meshgrid(axis, axis, indexing="ij")
    q = np.hypot(qx, qy)
    angle = np.arctan2(qy, qx)
    rng = np.random.default_rng(29)
    intensity = np.clip(
        40.0 * np.exp(-0.5 * ((q - 0.10) / 0.012) ** 2)
        + 100.0 * np.exp(-0.5 * ((q - 0.40) / 0.030) ** 2)
        + rng.normal(0.0, 0.20, size=shape),
        0.0,
        None,
    )
    mask = (
        (q > 0.07)
        & (q < 0.13)
        & (angle > np.deg2rad(40.0))
        & (angle < np.deg2rad(120.0))
    )
    result = trace_butterfly_ridges(
        intensity,
        {"qx": qx, "qy": qy, "q": q},
        (0.05, 0.80),
        mask=mask,
    )
    hint = result["diagnostics"]["first_order_q_hint"]
    lower_family = hint["candidate_families"][0]
    higher_family = next(
        family for family in hint["candidate_families"]
        if family["q_min"] > lower_family["q_max"]
        and family["radial_evidence_status"] == "significant"
    )

    assert higher_family["max_prominence"] > lower_family["max_prominence"]
    assert lower_family["n_valid_arcs"] >= 2
    assert lower_family["mean_matched_arc_span_deg"] > higher_family["mean_matched_arc_span_deg"]
    assert hint["selection_status"] == "selected"
    assert hint["q_star"] == pytest.approx(0.10, abs=0.01)
    assert hint["selected_family_id"] == lower_family["family_id"]
    assert any(peak["family_id"] == higher_family["family_id"] for peak in hint["competing_peaks"])
