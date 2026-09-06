from __future__ import annotations

from dataclasses import replace
import threading

import numpy as np
import pytest

from butterfly_saxs.butterfly_ridge import (
    METHOD_VERSION,
    _annotate_scale_stability,
    _apply_seeds,
    _graph_arcs,
    _nms,
    _normalise_options,
    _raw_candidates,
    _scaled_surface_field,
    trace_butterfly_ridges,
)
from butterfly_saxs.cancellation import AnalysisCancelled
from butterfly_saxs.ridge_profiles import extract_normal_profile
from butterfly_saxs.ridge_inputs import as_image, as_qmap, representative_q_step


def _nonelliptic_butterfly(shape: tuple[int, int] = (101, 101), seed: int = 4):
    axis = np.linspace(-1.0, 1.0, shape[0])
    qx, qy = np.meshgrid(axis, axis)
    q = np.hypot(qx, qy)
    angle = np.arctan2(qy, qx)
    centres = (0.43, -0.43, np.pi - 0.43, -np.pi + 0.43)
    image = np.full_like(q, 0.08)
    truth_mask = np.zeros_like(q, dtype=bool)
    target = 0.52 + 0.045 * np.sin(2.0 * angle) + 0.018 * np.sin(5.0 * angle)
    for centre in centres:
        distance = np.abs(np.angle(np.exp(1j * (angle - centre))))
        lobe = distance < 0.19
        truth_mask |= lobe
        image += 7.0 * np.exp(-0.5 * ((q - target) / 0.026) ** 2) * lobe
    image += np.random.default_rng(seed).normal(0.0, 0.04, image.shape)
    detector_mask = np.abs(qx) < 0.025
    return image, {"qx": qx, "qy": qy, "q": q}, detector_mask, target, truth_mask


def _options() -> dict[str, object]:
    return {
        "smoothing_scales": (1.0, 1.8, 2.8),
        "max_points": 700,
        "run_wang_check": False,
    }


def test_trace_returns_observed_points_and_topology_before_ellipse_fit() -> None:
    image, qmap, detector_mask, target, truth_mask = _nonelliptic_butterfly()
    result = trace_butterfly_ridges(
        image,
        qmap,
        (0.36, 0.70),
        mask=detector_mask,
        reference_axis_deg=0.0,
        options=_options(),
    )

    assert result["method_version"] == METHOD_VERSION
    assert isinstance(result["points"], list)
    assert isinstance(result["arcs"], list)
    assert isinstance(result["profiles"], dict)
    assert result["diagnostics"]["topology_before_ellipse_fit"] is True
    assert result["diagnostics"]["ellipse_fit"] is None
    assert result["points"]
    assert len(result["profiles"]) == len(result["points"])

    required = {
        "qx",
        "qy",
        "pixel_x",
        "pixel_y",
        "branch_id",
        "quadrant",
        "arc_id",
        "side",
        "accepted",
        "valid",
        "reason",
        "point_id",
        "intensity",
        "snr",
        "normal_qx",
        "normal_qy",
        "normal_fwhm_q",
        "localization_sigma_q",
        "sampling_sigma_q",
        "observed_support",
        "support_frozen",
        "q_normal_step",
        "topology_flags",
        "scale_stability",
    }
    assert required <= set(result["points"][0])
    assert all(point["point_id"] in result["profiles"] for point in result["points"])
    assert result["diagnostics"]["observed_support"]["support_version"] == "observed_q_corridor_v1"
    assert result["diagnostics"]["observed_support"]["n_arcs"] >= len(result["arcs"])
    assert all(point["branch_id"] in {-1, 0, 1} for point in result["points"])
    assert all(point["side"] in {"upper", "lower", "unknown"} for point in result["points"])
    assert all("topology_before_ellipse_fit" in arc["topology_flags"] for arc in result["arcs"])

    accepted = [point for point in result["points"] if point["accepted"]]
    assert accepted
    radial_error = np.asarray(
        [
            abs(np.hypot(point["qx"], point["qy"]) - target[np.rint(point["pixel_y"]).astype(int), np.rint(point["pixel_x"]).astype(int)])
            for point in accepted
            if truth_mask[np.clip(int(round(point["pixel_y"])), 0, target.shape[0] - 1), np.clip(int(round(point["pixel_x"])), 0, target.shape[1] - 1)]
        ],
        dtype=float,
    )
    assert radial_error.size > 8
    assert float(np.nanmedian(radial_error)) < 0.06
    # Rejected/ambiguous candidates remain in the UI-facing list.
    assert any(not point["accepted"] for point in result["points"])


def test_topology_does_not_force_four_arcs_and_preserves_masked_gaps() -> None:
    image, qmap, detector_mask, _target, _truth_mask = _nonelliptic_butterfly(shape=(81, 81), seed=9)
    # Remove one complete wing.  A valid result may contain fewer than four
    # arcs; the test guards against synthetic counterpart generation.
    qx, qy = qmap["qx"], qmap["qy"]
    angle = np.arctan2(qy, qx)
    removed_wing = np.abs(np.angle(np.exp(1j * (angle - 0.43)))) < 0.10
    detector_mask |= removed_wing
    result = trace_butterfly_ridges(image, qmap, (0.36, 0.70), mask=detector_mask, options=_options())
    assert result["points"]
    assert any(point["reason"] in {"profile_unavailable", "ambiguous_normal_profile", "weak_curvature", "accepted"} for point in result["points"])
    removed_points = [
        point
        for point in result["points"]
        if np.abs(np.angle(np.exp(1j * (np.arctan2(point["qy"], point["qx"]) - 0.43)))) < 0.10
    ]
    assert not any(point["accepted"] for point in removed_points)
    assert not any(point.get("source") == "mirrored" for point in result["points"])


def test_point_ids_are_stable_and_seed_snaps_only_to_observed_candidates() -> None:
    image, qmap, detector_mask, _target, _truth_mask = _nonelliptic_butterfly(shape=(81, 81), seed=6)
    base = trace_butterfly_ridges(image, qmap, (0.36, 0.70), mask=detector_mask, options=_options())
    assert base["points"]
    selected = base["points"][0]
    seed_action = {"type": "seed", "qx": selected["qx"], "qy": selected["qy"]}
    if selected["branch_id"] in {0, 1}:
        seed_action["branch_id"] = selected["branch_id"]
    if selected["side"] in {"upper", "lower"}:
        seed_action["side"] = selected["side"]
    edits = [
        seed_action,
        {"type": "exclude_point", "point_id": selected["point_id"]},
    ]
    edited = trace_butterfly_ridges(image, qmap, (0.36, 0.70), mask=detector_mask, options=_options(), edits=edits)
    base_ids = {point["point_id"] for point in base["points"]}
    edited_ids = {point["point_id"] for point in edited["points"]}
    assert base_ids == edited_ids
    excluded = next(point for point in edited["points"] if point["point_id"] == selected["point_id"])
    assert excluded["accepted"] is False
    assert excluded["reason"] == "excluded_point_edit"
    assert edited["diagnostics"]["seed_matches"]
    assert edited["diagnostics"]["seed_matches"][0]["matched"] is True


def test_flat_surface_has_no_principal_curvature_ridge() -> None:
    axis = np.linspace(-1.0, 1.0, 61)
    qx, qy = np.meshgrid(axis, axis)
    image = np.full_like(qx, 1.0)
    result = trace_butterfly_ridges(image, {"qx": qx, "qy": qy}, (0.25, 0.85), options=_options())
    assert result["points"] == []
    assert result["arcs"] == []
    assert result["diagnostics"]["n_raw_candidates"] == 0


def test_curvature_metadata_surface_keeps_original_intensity_units_without_changing_fit_inputs() -> None:
    from scipy.ndimage import gaussian_filter

    axis = np.linspace(-1.0, 1.0, 81)
    qx, qy = np.meshgrid(axis, axis)
    q = np.hypot(qx, qy)
    image = 1250.0 + 850.0 * np.exp(-0.5 * ((q - 0.55) / 0.04) ** 2)
    valid = (q >= 0.35) & (q <= 0.75)
    options = _normalise_options({"smoothing_scales": (1.6,), "run_wang_check": False})
    field = _scaled_surface_field(image, qx, qy, valid, sigma=1.6, options=options)

    support = gaussian_filter(valid.astype(float), sigma=1.6, mode="nearest")
    expected = np.divide(
        gaussian_filter(np.where(valid, image, 0.0), sigma=1.6, mode="nearest"),
        support,
        out=np.full_like(image, np.nan),
        where=support > 1e-8,
    )
    comparison = valid & (support > 0.90)
    assert field.height_scale > 1.0
    np.testing.assert_allclose(field.smooth[comparison], expected[comparison], rtol=1e-12, atol=1e-10)
    assert float(np.nanmedian(field.smooth[comparison])) > 1000.0

    current_raw = _raw_candidates(field, row_offset=0, col_offset=0, max_candidates=10000, options=options)
    legacy_field = replace(field, smooth=field.smooth * field.height_scale + field.baseline)
    legacy_raw = _raw_candidates(legacy_field, row_offset=0, col_offset=0, max_candidates=10000, options=options)
    assert len(current_raw) == len(legacy_raw)
    for current, legacy in zip(current_raw, legacy_raw):
        for key in ("qx", "qy", "pixel_x", "pixel_y", "score", "accepted"):
            assert current[key] == pytest.approx(legacy[key], abs=1e-12)
    current_nms = _nms(current_raw, options, field.q_step)
    legacy_nms = _nms(legacy_raw, options, field.q_step)
    assert [(p["qx"], p["qy"]) for p in current_nms] == pytest.approx([(p["qx"], p["qy"]) for p in legacy_nms], abs=1e-12)
    current_groups, current_edges = _graph_arcs(current_nms, options, field.q_step)
    legacy_groups, legacy_edges = _graph_arcs(legacy_nms, options, field.q_step)
    assert [len(group) for group in current_groups] == [len(group) for group in legacy_groups]
    assert current_edges == legacy_edges

    observed = next(point for point in current_raw if point["accepted"])
    qmap = {"qx": qx, "qy": qy, "q": q}
    current_profile = extract_normal_profile(image, qmap, observed, mask=~valid, options={"profile_samples": 33})
    legacy_profile = extract_normal_profile(image, qmap, legacy_raw[current_raw.index(observed)], mask=~valid, options={"profile_samples": 33})
    assert current_profile["localization_sigma_q"] == pytest.approx(legacy_profile["localization_sigma_q"], abs=1e-12)
    assert current_profile["normal_fwhm_q"] == pytest.approx(legacy_profile["normal_fwhm_q"], abs=1e-12)


def test_cancel_event_is_honoured_at_stage_boundaries() -> None:
    axis = np.linspace(-1.0, 1.0, 41)
    qx, qy = np.meshgrid(axis, axis)
    event = threading.Event()
    event.set()
    with pytest.raises(AnalysisCancelled):
        trace_butterfly_ridges(np.ones_like(qx), {"qx": qx, "qy": qy}, (0.2, 0.8), cancel_event=event)


def test_profile_refinement_keeps_curvature_seed_and_updates_observed_coordinate() -> None:
    image, qmap, detector_mask, _target, _truth_mask = _nonelliptic_butterfly(shape=(81, 81), seed=12)
    result = trace_butterfly_ridges(image, qmap, (0.36, 0.70), mask=detector_mask, options=_options())
    refined = [point for point in result["points"] if point.get("profile_refinement_applied")]
    assert refined
    for point in refined:
        seed_distance = np.hypot(point["qx"] - point["curvature_seed_qx"], point["qy"] - point["curvature_seed_qy"])
        assert point["profile_refinement_reason"] == "observed_single_normal_peak"
        assert seed_distance == pytest.approx(point["profile_shift_q"], abs=1e-12)
        assert seed_distance <= 1.5 * point["q_normal_step"] + 1e-12
        assert point["localization_sigma_q"] >= point["q_normal_step"] / np.sqrt(12.0) - 1e-12
        assert point["uncertainty_source"].startswith("empirical_profile_sampling")


def test_profile_shift_boundary_gate_preserves_curvature_coordinate() -> None:
    image, qmap, detector_mask, _target, _truth_mask = _nonelliptic_butterfly(shape=(81, 81), seed=13)
    options = {**_options(), "profile_refinement_max_q_step": 0.001}
    result = trace_butterfly_ridges(image, qmap, (0.36, 0.70), mask=detector_mask, options=options)
    gated = [point for point in result["points"] if point.get("profile_refinement_reason") == "profile_shift_exceeds_local_scale"]
    assert gated
    for point in gated:
        assert point["profile_refinement_applied"] is False
        assert point["qx"] == pytest.approx(point["curvature_seed_qx"], abs=1e-12)
        assert point["qy"] == pytest.approx(point["curvature_seed_qy"], abs=1e-12)


def test_seed_annotation_does_not_promote_a_rejected_candidate() -> None:
    image, qmap, detector_mask, _target, _truth_mask = _nonelliptic_butterfly(shape=(81, 81), seed=14)
    base = trace_butterfly_ridges(image, qmap, (0.36, 0.70), mask=detector_mask, options=_options())
    rejected = next(point for point in base["points"] if not point["accepted"])
    result = trace_butterfly_ridges(
        image,
        qmap,
        (0.36, 0.70),
        mask=detector_mask,
        options=_options(),
        edits=[{"type": "seed", "qx": rejected["qx"], "qy": rejected["qy"]}],
    )
    seeded = next(point for point in result["points"] if point["point_id"] == rejected["point_id"])
    assert seeded["accepted"] is False
    assert "seeded" not in seeded
    assert result["diagnostics"]["seed_matches"][0]["reason"] in {
        "seed_no_supported_observation",
        "seed_outside_snap_scale",
        "selected_supported_observed_arc",
    }


def test_seed_annotation_overrides_labels_only_on_the_snapped_observation() -> None:
    image, qmap, detector_mask, _target, _truth_mask = _nonelliptic_butterfly(shape=(81, 81), seed=15)
    base = trace_butterfly_ridges(image, qmap, (0.36, 0.70), mask=detector_mask, options=_options())
    accepted = next(point for point in base["points"] if point["accepted"])
    result = trace_butterfly_ridges(
        image,
        qmap,
        (0.36, 0.70),
        mask=detector_mask,
        options=_options(),
        edits=[{"type": "seed", "qx": accepted["qx"], "qy": accepted["qy"], "branch_id": accepted["branch_id"], "side": accepted["side"]}],
    )
    seeded = next(point for point in result["points"] if point["point_id"] == accepted["point_id"])
    assert seeded["seeded"] is True
    assert seeded["accepted"] is True
    assert seeded["branch_id"] == accepted["branch_id"]
    assert seeded["side"] == accepted["side"]
    assert seeded["seed_selection_status"] == "selected_observed_arc"
    assert result["diagnostics"]["seed_matches"][0]["selected_arc_id"] == seeded["arc_id"]


def test_auto_seed_selects_one_supported_component_and_retains_competitor_reason() -> None:
    points = [
        {"point_id": "a0", "qx": 0.000, "qy": 0.0, "arc_id": 0, "accepted": True, "valid": True, "branch_id": 0, "side": "upper", "snr": 8.0, "q_normal_step": 0.02, "topology_flags": []},
        {"point_id": "a1", "qx": 0.010, "qy": 0.0, "arc_id": 0, "accepted": True, "valid": True, "branch_id": 0, "side": "upper", "snr": 7.0, "q_normal_step": 0.02, "topology_flags": []},
        {"point_id": "b0", "qx": 0.045, "qy": 0.0, "arc_id": 1, "accepted": True, "valid": True, "branch_id": 0, "side": "lower", "snr": 6.0, "q_normal_step": 0.02, "topology_flags": []},
        {"point_id": "b1", "qx": 0.055, "qy": 0.0, "arc_id": 1, "accepted": False, "valid": False, "branch_id": 0, "side": "lower", "snr": 1.0, "q_normal_step": 0.02, "reason": "low_snr", "topology_flags": []},
    ]
    arcs = [
        {"arc_id": 0, "accepted_points": 2, "scale_stability": 1.0, "branch_ids": [0], "sides": ["upper"]},
        {"arc_id": 1, "accepted_points": 1, "scale_stability": 0.8, "branch_ids": [0], "sides": ["lower"]},
    ]
    records = _apply_seeds(points, arcs, [{"type": "seed", "qx": 0.02, "qy": 0.0}], {"seed_snap_factor": 3.0}, 0.02)
    assert records[0]["matched"] is True
    assert records[0]["selected_arc_id"] == 0
    assert records[0]["affected_point_ids"] == ["a0", "a1"]
    assert records[0]["competing_arc_ids"] == [1]
    assert all(point.get("seed_selected") is True for point in points[:2])
    assert points[2]["accepted"] is False
    assert points[2]["reason"] == "competing_seed_component"
    assert points[3]["accepted"] is False
    assert points[3]["reason"] == "low_snr"


def test_explicit_side_prior_is_applied_coherently_to_selected_arc() -> None:
    points = [
        {"point_id": "p0", "qx": 0.0, "qy": 0.0, "arc_id": 0, "accepted": True, "valid": True, "branch_id": 0, "side": "unknown", "snr": 8.0, "q_normal_step": 0.02, "topology_flags": []},
        {"point_id": "p1", "qx": 0.01, "qy": 0.0, "arc_id": 0, "accepted": True, "valid": True, "branch_id": 0, "side": "unknown", "snr": 7.0, "q_normal_step": 0.02, "topology_flags": []},
    ]
    arcs = [{"arc_id": 0, "accepted_points": 2, "scale_stability": 1.0, "branch_ids": [0], "sides": ["unknown"]}]
    records = _apply_seeds(points, arcs, [{"type": "seed", "qx": 0.0, "qy": 0.0, "branch_id": 0, "side": "lower"}], {"seed_snap_factor": 3.0}, 0.02)
    assert records[0]["matched"] is True
    assert all(point["side"] == "lower" for point in points)
    assert arcs[0]["sides"] == ["lower"]
    assert arcs[0]["seed_coherent"] is True


def test_shared_input_adapter_accepts_production_aliases_broadcasts_and_mask_polarity() -> None:
    shape = (5, 7)
    image = {"data": np.ones(shape), "bad_mask": np.asarray([[False, True, False, False, False, False, False]])}
    qx = np.linspace(-1.0, 1.0, shape[0], dtype=float)[:, None]
    qy = np.linspace(-1.0, 1.0, shape[1], dtype=float)[None, :]
    qmap = {
        "qx_map": qx,
        "qy_map": qy,
        "q_map": np.hypot(qx, qy),
        "bad_mask": np.asarray([[False, False, False, True, False, False, False]]),
        "valid_mask": np.asarray([[True], [True], [False], [True], [True]]),
    }
    data, image_invalid = as_image(image)
    got_qx, got_qy, got_q, qmap_invalid = as_qmap(qmap, shape)
    assert data.shape == shape
    assert image_invalid is not None and image_invalid[0, 1]
    assert got_qx.shape == got_qy.shape == got_q.shape == shape
    assert qmap_invalid is not None
    assert np.all(qmap_invalid[2, :])
    assert np.all(qmap_invalid[:, 3])
    assert representative_q_step(qmap) == pytest.approx(0.4166666666666667)

    # The same aliases and broadcast mask are accepted by both tracing and
    # normal-profile setup; no source array is modified by either path.
    trace = trace_butterfly_ridges(image, qmap, (0.0, 2.0), options={"run_wang_check": False})
    assert trace["diagnostics"]["mask_fraction_in_q_window"] > 0.0
    point = {"pixel_x": 3.0, "pixel_y": 2.0, "qx": 0.0, "qy": 0.0, "normal_qx": 1.0, "normal_qy": 0.0, "q_normal_step": 1.0}
    profile = extract_normal_profile(image, qmap, point, options={"profile_samples": 9})
    assert profile["valid"] is False


def test_scale_stability_spatial_query_matches_exact_bruteforce_support() -> None:
    raw = [
        {"qx": 0.0, "qy": 0.0, "scale": 1.0},
        {"qx": 1.5, "qy": 0.0, "scale": 2.0},
        {"qx": 4.0, "qy": 0.0, "scale": 3.0},
        {"qx": 0.0, "qy": 2.0, "scale": 1.0},
    ]
    candidates = [
        {"qx": 0.0, "qy": 0.0, "topology_flags": []},
        {"qx": 3.9, "qy": 0.0, "topology_flags": []},
    ]
    _annotate_scale_stability(candidates, raw, 1.0, (1.0, 2.0, 3.0))
    expected = []
    for point in candidates:
        support = [other["scale"] for other in raw if np.hypot(other["qx"] - point["qx"], other["qy"] - point["qy"]) <= 1.8]
        expected.append(len(set(support)) / 3.0)
    assert [point["scale_stability"] for point in candidates] == pytest.approx(expected)


def test_graph_connectivity_does_not_bridge_preassigned_branches() -> None:
    options = _normalise_options({"graph_radius_factor": 4.0})
    candidates = []
    for index, (qx, branch) in enumerate(((0.00, 0), (0.04, 0), (0.08, 1), (0.12, 1))):
        candidates.append(
            {
                "qx": qx,
                "qy": 0.0,
                "scale": 1.0,
                "accepted": True,
                "branch_id": branch,
                "tangent_qx": 1.0,
                "tangent_qy": 0.0,
                "normal_qx": 0.0,
                "normal_qy": 1.0,
                "point_id": f"p{index}",
            }
        )
    groups, _edges = _graph_arcs(candidates, options, 0.05)
    assert all(len({candidates[index]["branch_id"] for index in group}) == 1 for group in groups)


def test_public_points_expose_sampling_uncertainty() -> None:
    image, qmap, detector_mask, _target, _truth_mask = _nonelliptic_butterfly(shape=(61, 61), seed=22)
    result = trace_butterfly_ridges(image, qmap, (0.36, 0.70), mask=detector_mask, options=_options())
    assert result["points"]
    assert all("sampling_sigma_q" in point for point in result["points"])


def test_trace_builds_coordinate_derivatives_once_for_all_scales(monkeypatch: pytest.MonkeyPatch) -> None:
    import butterfly_saxs.butterfly_ridge as ridge_module

    image, qmap, detector_mask, _target, _truth_mask = _nonelliptic_butterfly(shape=(41, 41), seed=23)
    original = ridge_module._coordinate_derivatives
    calls = 0

    def counted(qx: np.ndarray, qy: np.ndarray):
        nonlocal calls
        calls += 1
        return original(qx, qy)

    monkeypatch.setattr(ridge_module, "_coordinate_derivatives", counted)
    trace_butterfly_ridges(
        image,
        qmap,
        (0.36, 0.70),
        mask=detector_mask,
        options={"smoothing_scales": (1.0, 1.8, 2.8), "max_points": 100, "run_wang_check": False},
    )
    assert calls == 1
