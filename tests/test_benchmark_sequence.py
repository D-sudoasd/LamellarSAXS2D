from __future__ import annotations

import numpy as np

from butterfly_saxs.butterfly_ridge import trace_butterfly_ridges
from butterfly_saxs.benchmark_sequence import (
    ObliqueStackSettings,
    SequenceSettings,
    generate_oblique_stack_sequence,
    generate_sequence,
    measure_annular_local_peak,
    pipeline_analysis_config,
)


def test_sequence_is_deterministic_continuous_and_retains_noise_control() -> None:
    settings = SequenceSettings(
        n_signal_frames=4,
        shape=(128, 128),
        local_missing_lobe_frame=2,
        seed=17,
    )
    first = generate_sequence(settings)
    second = generate_sequence(settings)

    assert len(first) == 6
    assert [frame["frame_index"] for frame in first] == list(range(len(first)))
    assert [frame["frame_id"] for frame in first] == [
        "signal_00", "signal_01", "signal_02", "signal_03",
        "buried_signal_stress", "noise_control",
    ]
    assert [frame["structural_q0_nm_inv"] for frame in first[:4]] == sorted(
        (frame["structural_q0_nm_inv"] for frame in first[:4]), reverse=True
    )
    assert [frame["structure_truth"]["orientation_spread_scale"] for frame in first[:4]] == sorted(
        frame["structure_truth"]["orientation_spread_scale"] for frame in first[:4]
    )
    for left, right in zip(first, second):
        assert np.array_equal(left["intensity_noisy"], right["intensity_noisy"])
        assert np.array_equal(left["mask"], right["mask"])

    assert first[2]["mask_diagnostics"]["missing_lobe_pixels"] > 0
    assert all(frame["mask_diagnostics"]["missing_lobe_pixels"] == 0
               for index, frame in enumerate(first[:4]) if index != 2)
    missing_lobe = first[2]["missing_lobe_mask"]
    assert np.any(first[2]["intensity_noiseless"][missing_lobe] > 0)
    assert np.count_nonzero(first[0]["detector_beamstop_mask"]) == np.count_nonzero(
        first[2]["detector_beamstop_mask"]
    )

    control = first[-1]
    assert control["sequence_role"] == "noise_only_control"
    assert control["frame_index"] == 5
    assert control["structural_q0_nm_inv"] is None
    assert control["structure_truth"] is None
    assert np.count_nonzero(control["intensity_noiseless"]) == 0
    assert np.std(control["intensity_noisy"]) > 0
    signal_sigmas = [float(frame["noise_sigma"]) for frame in first[:4]]
    assert np.isclose(first[-2]["noise_sigma"], settings.buried_signal_stress_sigma)
    assert np.isclose(control["noise_sigma"], np.median(signal_sigmas))
    assert all(np.isclose(frame["noise_reference"]["p90_snr"], 15.0) for frame in first[:4])


def test_annular_noise_free_reference_is_pixel_measured_and_near_spacing_q() -> None:
    settings = SequenceSettings(
        n_signal_frames=3,
        shape=(128, 128),
        local_missing_lobe_frame=1,
        seed=20260926,
    )
    frames = generate_sequence(settings)
    for frame in frames[:3]:
        q0 = float(frame["structural_q0_nm_inv"])
        measured = measure_annular_local_peak(
            frame["intensity_noiseless"], frame["q"], frame["mask"], q0
        )
        assert measured["method"] == "independent_noise_free_annular_mean_local_peak"
        assert measured["valid_pixel_count"] > 100
        assert measured["populated_bin_count"] > 10
        assert measured["q_peak_nm_inv"] > 0
        assert abs(measured["q_peak_nm_inv"] - q0) < 0.06
        assert np.isclose(measured["spacing_nm"], 2 * np.pi / measured["q_peak_nm_inv"])
        assert frame["projection_truth"]["truth_scope"].startswith("projection_only")


def test_pipeline_config_keeps_curvature_and_radial_paths_explicit() -> None:
    curvature = pipeline_analysis_config()
    radial = pipeline_analysis_config(ridge_method="radial_peak")
    assert curvature["analysis"]["ridge_method"] == "butterfly_curvature"
    assert radial["analysis"]["ridge_method"] == "radial_peak"
    assert curvature["analysis"]["ellipse"]["preset"] == "standard"
    assert curvature["analysis"]["butterfly"]["stage"] == "evaluate"


def test_oblique_stack_sequence_uses_finite_density_fft_and_matched_noise_control() -> None:
    settings = ObliqueStackSettings(
        n_signal_frames=3,
        shape=(128, 128),
        local_missing_lobe_frame=1,
        seed=31,
    )
    first = generate_oblique_stack_sequence(settings)
    second = generate_oblique_stack_sequence(settings)
    assert len(first) == 5
    assert [frame["sequence_role"] for frame in first] == [
        "oblique_stack_signal", "oblique_stack_signal", "oblique_stack_signal",
        "noise_free_signal_control", "noise_only_control",
    ]
    assert [frame["structure_truth"]["layer_spacing_nm"] for frame in first[:3]] == sorted(
        frame["structure_truth"]["layer_spacing_nm"] for frame in first[:3]
    )
    for left, right in zip(first, second):
        assert np.array_equal(left["intensity_noiseless"], right["intensity_noiseless"])
        assert np.array_equal(left["intensity_noisy"], right["intensity_noisy"])
    truth = first[0]["structure_truth"]
    assert truth["model"] == "finite_gaussian_slab_stacks_incoherent_fft_power_sum"
    assert truth["stack_width_nm"] == settings.stack_width_nm
    assert truth["stack_height_nm"] == settings.stack_height_nm
    assert truth["positive_population_stack_rotation_deg"] < 0
    assert truth["negative_population_stack_rotation_deg"] > 0
    assert truth["geometry_reference_type"].endswith("not_projection_target")
    assert first[1]["mask_diagnostics"]["missing_lobe_pixels"] > 0
    assert np.any(first[1]["intensity_noiseless"][first[1]["missing_lobe_mask"]] > 0)
    assert first[-2]["noise_sigma"] == 0
    assert np.array_equal(first[-2]["intensity_noisy"], first[-2]["intensity_noiseless"])
    assert first[-1]["structural_q0_nm_inv"] is None
    assert np.isclose(
        first[-1]["noise_sigma"], np.median([frame["noise_sigma"] for frame in first[:3]])
    )


def test_512_t2_signal_01_retains_both_radial_populations_without_q_hint() -> None:
    frame = next(
        item
        for item in generate_sequence(SequenceSettings(shape=(512, 512)))
        if item["frame_id"] == "signal_01"
    )
    qmap = {
        "qx": frame["qx"],
        "qy": frame["qy"],
        "q": frame["q"],
        "q_unit": frame["q_unit"],
    }

    # Only the measured image, q coordinates, and detector mask reach the
    # tracer; structural q0 remains an independent test reference.
    trace = trace_butterfly_ridges(
        frame["intensity_noisy"], qmap, (0.15, 0.85), mask=frame["mask"]
    )

    diagnostics = trace["diagnostics"]
    hint = diagnostics["first_order_q_hint"]
    radial = diagnostics["radial_population"]
    assert hint["selection_status"] == "ambiguous"
    assert hint["q_star"] is None
    assert radial["split"] is True
    assert radial["selection_status"] == "ambiguous"
    assert radial["keep"] == "all_observed"
    assert radial["demoted"] == 0
    assert radial["low_q_population"]["n_points"] > 0
    assert radial["high_q_population"]["n_points"] > 0

    threshold = float(radial["threshold"])
    accepted_radii = [
        float(np.hypot(point["qx"], point["qy"]))
        for point in trace["points"]
        if point.get("accepted")
        and point.get("branch_id") in (0, 1)
        and point.get("side") in ("upper", "lower")
    ]
    low_count = sum(radius < threshold for radius in accepted_radii)
    high_count = sum(radius >= threshold for radius in accepted_radii)
    assert low_count >= radial["low_q_population"]["n_points"] > 0
    assert high_count >= radial["high_q_population"]["n_points"] > 0
