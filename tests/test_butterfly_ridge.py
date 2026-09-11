from __future__ import annotations

from dataclasses import replace
import math
import threading

import numpy as np
import pytest

from butterfly_saxs.butterfly_ridge import (
    METHOD_VERSION,
    _annotate_scale_stability,
    _apply_seeds,
    _demote_secondary_radial_population,
    _arc_topology,
    _assign_component_identities,
    _first_order_q_hint,
    _select_first_order_peak_index,
    _graph_arcs,
    _prefer_first_order_scores,
    _fill_sparse_first_order_ring,
    _point_identity_key,
    _sparse_first_order_coverage,
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


def _ellipse_wing_candidates():
    from butterfly_saxs.ellipse import EllipseGeometry

    geometry = EllipseGeometry(0.0, 0.0, 1.5, 0.12, 0.80)
    samples = (0.55, 0.40, 0.25, 0.0, -0.25, -0.40, -0.55)
    candidates = []
    for index, t in enumerate(samples):
        qx, qy = geometry.point(t)
        candidates.append(
            {
                "point_id": f"p{index}",
                "qx": float(qx[0] if np.ndim(qx) else qx),
                "qy": float(qy[0] if np.ndim(qy) else qy),
                "tangent_qx": float(-math.sin(t)),
                "tangent_qy": float(math.cos(t)),
                "accepted": True,
                "valid": True,
                "reason": "accepted",
                "scale": 2.0,
                "topology_flags": [],
                "arc_id": -1,
                "branch_id": -1,
                "side": "unknown",
            }
        )
    return candidates


def test_connected_two_sided_wing_splits_instead_of_rejecting_all_points() -> None:
    from butterfly_saxs.butterfly_ridge import _arc_topology, _refresh_arc_identity

    candidates = _ellipse_wing_candidates()
    options = _normalise_options({"min_arc_points": 3, "reference_axis_deg": 0.0})
    groups = [list(range(len(candidates)))]
    edges = [(index, index + 1) for index in range(len(candidates) - 1)]
    arcs = _arc_topology(groups, edges, candidates, options, 0.02)
    _refresh_arc_identity(arcs, candidates)

    accepted = [point for point in candidates if point["accepted"]]
    sides = {point["side"] for point in accepted}
    assert "upper" in sides
    assert "lower" in sides
    assert all(point["reason"] != "unresolved_arc_identity" for point in accepted)
    resolved = [arc for arc in arcs if arc["identity_resolved"]]
    assert len(resolved) >= 2
    assert {tuple(arc["accepted_sides"]) for arc in resolved} >= {("lower",), ("upper",)}
    tip = next(point for point in candidates if point["point_id"] == "p3")
    assert tip["accepted"] is False
    assert tip["reason"] in {"side_axis_boundary", "unresolved_point_identity"}
    assert tip["side"] == "unknown"


def test_flat_wing_keeps_side_labels_when_pixel_step_exceeds_minor_axis() -> None:
    from butterfly_saxs.butterfly_ridge import _arc_topology
    from butterfly_saxs.ellipse import EllipseGeometry

    geometry = EllipseGeometry(0.0, 0.0, 0.72, 0.02, math.radians(17.0))
    samples = (0.28, 0.18, 0.10, 0.0, -0.10, -0.18, -0.28)
    candidates = []
    for index, value in enumerate(samples):
        qx, qy = geometry.point(value)
        candidates.append(
            {
                "point_id": f"p{index}",
                "qx": float(qx[0] if np.ndim(qx) else qx),
                "qy": float(qy[0] if np.ndim(qy) else qy),
                "tangent_qx": float(-math.sin(value)),
                "tangent_qy": float(math.cos(value)),
                "accepted": True,
                "valid": True,
                "reason": "accepted",
                "scale": 2.0,
                "topology_flags": [],
                "arc_id": -1,
                "branch_id": -1,
                "side": "unknown",
            }
        )
    options = _normalise_options({"min_arc_points": 3, "reference_axis_deg": 0.0})
    groups = [list(range(len(candidates)))]
    edges = [(index, index + 1) for index in range(len(candidates) - 1)]
    # Pixel step is larger than b=0.0144; a one-pixel side band would
    # swallow the whole wing.
    _arc_topology(groups, edges, candidates, options, 0.025)
    sides = {point["side"] for point in candidates if point["accepted"]}
    assert "upper" in sides
    assert "lower" in sides
    tip = next(point for point in candidates if point["point_id"] == "p3")
    assert tip["side"] == "unknown"


def test_coarse_flat_ellipse_trace_recovers_four_identity_sides() -> None:
    from butterfly_saxs.benchmark_arcs import generate_arc_case

    case = generate_arc_case("ellipse_ratio_020", seed=502, shape=(96, 96))
    result = trace_butterfly_ridges(
        case["image"],
        case["qmap"],
        (0.05, 1.1),
        mask=case.get("mask"),
        reference_axis_deg=0.0,
        options={"run_wang_check": False},
    )
    sides = {
        (point.get("branch_id"), point.get("side"))
        for point in result["points"]
        if point.get("accepted") and point.get("side") in {"upper", "lower"}
    }
    assert sides == {(0, "upper"), (0, "lower"), (1, "upper"), (1, "lower")}


def test_secondary_radial_population_is_demoted_and_unimodal_wings_are_kept() -> None:
    mixed = []
    for index, radius in enumerate([0.11] * 8 + [0.48] * 10):
        angle = 0.4 + 0.05 * (index % 4)
        mixed.append(
            {
                "qx": radius * math.cos(angle),
                "qy": radius * math.sin(angle),
                "branch_id": index % 2,
                "side": "upper" if index % 2 == 0 else "lower",
                "accepted": True,
            }
        )
    summary = _demote_secondary_radial_population(mixed)
    assert summary["split"] is True
    assert summary["keep"] == "low_q"
    assert summary["demoted"] == 10
    assert sum(1 for point in mixed if point["accepted"]) == 8
    assert all(point["reason"] == "secondary_radial_population" for point in mixed if not point["accepted"])
    harmonic = [dict(point) for point in mixed]
    for point in harmonic:
        point["accepted"] = True
        point.pop("reason", None)
    hinted = _demote_secondary_radial_population(harmonic, prefer_radius=0.48)
    assert hinted["keep"] == "high_q"
    assert hinted["demoted"] == 8

    unimodal = []
    for index in range(16):
        radius = 0.70 + 0.02 * (index % 3)
        angle = 0.3 + 0.04 * index
        unimodal.append(
            {
                "qx": radius * math.cos(angle),
                "qy": radius * math.sin(angle),
                "branch_id": 0,
                "side": "upper",
                "accepted": True,
            }
        )
    kept = _demote_secondary_radial_population(unimodal)
    assert kept["split"] is False
    assert all(point["accepted"] for point in unimodal)

    continuum = []
    for index, radius in enumerate(np.linspace(0.09, 0.55, 24)):
        angle = 0.35 + 0.03 * (index % 5)
        continuum.append(
            {
                "qx": float(radius * math.cos(angle)),
                "qy": float(radius * math.sin(angle)),
                "branch_id": index % 2,
                "side": "upper" if index % 2 == 0 else "lower",
                "accepted": True,
            }
        )
    trimmed = _demote_secondary_radial_population(continuum, prefer_radius=0.092)
    assert trimmed["demoted"] > 0
    assert trimmed["keep"] == "first_order_hint"
    assert all(
        math.hypot(float(point["qx"]), float(point["qy"])) <= 3.5 * 0.092 + 1e-12
        for point in continuum
        if point["accepted"]
    )


def test_first_order_q_hint_selects_inner_ring_not_brighter_harmonic() -> None:
    axis = np.linspace(-0.8, 0.8, 81)
    qx, qy = np.meshgrid(axis, axis)
    radius = np.hypot(qx, qy)
    image = (
        8.0 * np.exp(-0.5 * ((radius - 0.10) / 0.012) ** 2)
        + 22.0 * np.exp(-0.5 * ((radius - 0.40) / 0.020) ** 2)
    )
    hint = _first_order_q_hint(radius, image, np.isfinite(image), 0.05, 0.80)
    assert hint["reason"] == "ok"
    assert hint["q_star"] == pytest.approx(0.10, abs=0.03)
    assert hint["band"][1] < 0.22


def test_first_order_q_hint_refines_off_the_coarse_bin_center() -> None:
    axis = np.linspace(-0.4, 0.4, 161)
    qx, qy = np.meshgrid(axis, axis)
    radius = np.hypot(qx, qy)
    image = 10.0 * np.exp(-0.5 * ((radius - 0.0925) / 0.008) ** 2)
    hint = _first_order_q_hint(radius, image, np.isfinite(image), 0.05, 0.80)
    assert hint["reason"] == "ok"
    assert hint["q_star"] == pytest.approx(0.0925, abs=0.003)
    assert abs(hint["q_star"] - 0.0925) <= abs(float(hint["q_star_bin"]) - 0.0925) + 1e-12


def test_first_order_refine_does_not_snap_to_a_two_value_clamp() -> None:
    axis = np.linspace(-0.4, 0.4, 161)
    qx, qy = np.meshgrid(axis, axis)
    radius = np.hypot(qx, qy)
    values = []
    for centre in (0.0918, 0.0925, 0.0934):
        image = 10.0 * np.exp(-0.5 * ((radius - centre) / 0.005) ** 2)
        hint = _first_order_q_hint(radius, image, np.isfinite(image), 0.05, 0.80)
        values.append(float(hint["q_star"]))
    assert values[1] == pytest.approx(0.0925, abs=0.003)
    assert values[0] < values[1] < values[2]
    assert {round(value, 6) for value in values} != {0.091903, 0.093066}


def test_select_first_order_peak_keeps_stronger_bin_of_the_same_ring() -> None:
    peak_q = np.array([0.0852, 0.0930, 0.370])
    peak_intensity = np.array([0.90, 1.20, 3.00])
    peak_indices = np.array([4, 5, 40])
    assert _select_first_order_peak_index(peak_q, peak_intensity, peak_indices) == 5


def test_first_order_q_hint_ignores_low_q_shoulder_of_the_same_ring() -> None:
    axis = np.linspace(-0.8, 0.8, 161)
    qx, qy = np.meshgrid(axis, axis)
    radius = np.hypot(qx, qy)
    image = (
        6.5 * np.exp(-0.5 * ((radius - 0.082) / 0.0045) ** 2)
        + 10.0 * np.exp(-0.5 * ((radius - 0.093) / 0.006) ** 2)
        + 24.0 * np.exp(-0.5 * ((radius - 0.38) / 0.018) ** 2)
    )
    hint = _first_order_q_hint(radius, image, np.isfinite(image), 0.05, 0.80)
    assert hint["reason"] == "ok"
    assert hint["q_star"] == pytest.approx(0.093, abs=0.008)
    assert hint["band"][1] < 0.22


def test_unresolved_major_keeps_reference_axis_side_instead_of_dropping_the_wing() -> None:
    candidates = [
        {
            "qx": -0.096,
            "qy": -0.034,
            "accepted": True,
            "valid": True,
            "reason": "accepted",
            "topology_flags": [],
            "branch_id": -1,
            "side": "unknown",
        },
        {
            "qx": -0.103,
            "qy": -0.030,
            "accepted": True,
            "valid": True,
            "reason": "accepted",
            "topology_flags": [],
            "branch_id": -1,
            "side": "unknown",
        },
    ]
    options = _normalise_options({"min_arc_points": 6, "reference_axis_deg": 0.0})
    _assign_component_identities(
        [0, 1],
        candidates,
        options,
        0.01,
        center_x=0.0,
        center_y=0.0,
        major=np.asarray([np.nan, np.nan], dtype=float),
        resolved=False,
    )
    assert candidates[0]["branch_id"] == 0
    assert candidates[0]["side"] == "lower"
    assert candidates[0]["accepted"] is True
    assert "side_from_reference_axis" in candidates[0]["topology_flags"]
    assert candidates[1]["side"] == "lower"
    assert candidates[1]["accepted"] is True


def test_short_arc_uses_reference_side_even_when_local_major_resolves() -> None:
    candidates = [
        {
            "qx": -0.100,
            "qy": -0.020,
            "accepted": True,
            "valid": True,
            "reason": "accepted",
            "topology_flags": [],
            "branch_id": -1,
            "side": "unknown",
        },
        {
            "qx": -0.105,
            "qy": -0.018,
            "accepted": True,
            "valid": True,
            "reason": "accepted",
            "topology_flags": [],
            "branch_id": -1,
            "side": "unknown",
        },
    ]
    options = _normalise_options({"min_arc_points": 6, "reference_axis_deg": 0.0})
    _assign_component_identities(
        [0, 1],
        candidates,
        options,
        0.01,
        center_x=0.0,
        center_y=0.0,
        major=np.asarray([0.0, 1.0], dtype=float),
        resolved=True,
    )
    assert candidates[0]["branch_id"] == 0
    assert candidates[0]["side"] == "lower"
    assert candidates[0]["accepted"] is True
    assert "side_from_reference_axis" in candidates[0]["topology_flags"]


def test_unconnected_branched_point_gets_reference_axis_side() -> None:
    candidates = [
        {
            "qx": -0.10,
            "qy": 0.03,
            "accepted": True,
            "valid": True,
            "reason": "accepted",
            "topology_flags": [],
            "arc_id": -1,
            "branch_id": -1,
            "side": "unknown",
            "tangent_qx": 0.0,
            "tangent_qy": 1.0,
            "scale": 2.0,
        }
    ]
    options = _normalise_options({"reference_axis_deg": 0.0})
    _arc_topology([], [], candidates, options, 0.01)
    assert candidates[0]["branch_id"] == 1
    assert candidates[0]["side"] == "upper"
    assert candidates[0]["accepted"] is True
    assert "side_from_reference_axis" in candidates[0]["topology_flags"]


def test_first_order_score_weight_keeps_tips_and_suppresses_harmonics() -> None:
    points = [
        {"qx": 0.11, "qy": 0.0, "score": 1.0},
        {"qx": 0.40, "qy": 0.0, "score": 1.0},
    ]
    updated = _prefer_first_order_scores(points, 0.10)
    assert updated == 2
    assert points[0]["score"] > 0.7
    assert points[1]["score"] < 0.05


def test_unknown_tip_point_does_not_poison_an_identity_pure_arc() -> None:
    from butterfly_saxs.butterfly_ridge import _refresh_arc_identity

    points = [
        {
            "point_id": "upper-1",
            "arc_id": 0,
            "branch_id": 0,
            "side": "upper",
            "accepted": True,
            "valid": True,
            "reason": "accepted",
            "topology_flags": [],
        },
        {
            "point_id": "upper-2",
            "arc_id": 0,
            "branch_id": 0,
            "side": "upper",
            "accepted": True,
            "valid": True,
            "reason": "accepted",
            "topology_flags": [],
        },
        {
            "point_id": "tip",
            "arc_id": 0,
            "branch_id": 0,
            "side": "unknown",
            "accepted": True,
            "valid": True,
            "reason": "accepted",
            "topology_flags": [],
        },
    ]
    arcs = [
        {
            "arc_id": 0,
            "valid": True,
            "reason": "accepted",
            "topology_flags": ["topology_before_ellipse_fit"],
        }
    ]
    _refresh_arc_identity(arcs, points)
    assert points[0]["accepted"] is True
    assert points[1]["accepted"] is True
    assert points[2]["accepted"] is False
    assert points[2]["reason"] == "side_axis_boundary"
    assert arcs[0]["identity_resolved"] is True
    assert arcs[0]["valid"] is True


def test_axial_identity_points_are_sparse_coverage() -> None:
    points = [
        {"qx": 0.10, "qy": 0.01, "accepted": True, "branch_id": 0, "side": "upper"},
        {"qx": 0.11, "qy": -0.01, "accepted": True, "branch_id": 0, "side": "lower"},
        {"qx": -0.10, "qy": 0.01, "accepted": True, "branch_id": 1, "side": "upper"},
        {"qx": -0.11, "qy": -0.01, "accepted": True, "branch_id": 1, "side": "lower"},
    ]
    sparse = _sparse_first_order_coverage(
        points, center_x=0.0, center_y=0.0, reference_axis_deg=0.0
    )
    assert sparse["sparse"] is True
    assert sparse["n"] == 4
    wide = [
        {**points[0], "qx": 0.07, "qy": 0.08},
        {**points[1], "qx": 0.07, "qy": -0.08},
        {**points[2], "qx": -0.07, "qy": 0.08},
        {**points[3], "qx": -0.07, "qy": -0.08},
        {"qx": 0.05, "qy": 0.09, "accepted": True, "branch_id": 0, "side": "upper"},
        {"qx": 0.05, "qy": -0.09, "accepted": True, "branch_id": 0, "side": "lower"},
        {"qx": -0.05, "qy": 0.09, "accepted": True, "branch_id": 1, "side": "upper"},
        {"qx": -0.05, "qy": -0.09, "accepted": True, "branch_id": 1, "side": "lower"},
    ]
    assert _sparse_first_order_coverage(
        wide, center_x=0.0, center_y=0.0, reference_axis_deg=0.0
    )["sparse"] is False


def test_sparse_ring_fill_adds_observed_peaks_not_empty_sectors() -> None:
    axis = np.linspace(-1.0, 1.0, 81)
    qx, qy = np.meshgrid(axis, axis)
    q = np.hypot(qx, qy)
    image = 0.2 + 4.0 * np.exp(-0.5 * ((q - 0.50) / 0.03) ** 2)
    valid = np.isfinite(image)
    points = [
        {
            "qx": 0.50,
            "qy": 0.02,
            "accepted": True,
            "branch_id": 0,
            "side": "upper",
            "point_id": "axial-plus",
        },
        {
            "qx": -0.50,
            "qy": -0.02,
            "accepted": True,
            "branch_id": 0,
            "side": "lower",
            "point_id": "axial-minus",
        },
    ]
    options = _normalise_options({"reference_axis_deg": 0.0, "run_wang_check": False})
    arcs: list[dict[str, object]] = []
    summary = _fill_sparse_first_order_ring(
        points,
        qx=qx,
        qy=qy,
        q=q,
        intensity=image,
        valid=valid,
        hint=0.50,
        options=options,
        q_step=0.025,
        signature="test-ring",
        arcs=arcs,
    )
    assert summary["applied"] is True
    assert summary["added"] >= 4
    added = [point for point in points if point.get("reason") == "first_order_ring_sample"]
    assert added
    assert all(_point_identity_key(point) is not None for point in added)
    assert len({int(point["arc_id"]) for point in added}) == len(added)
    assert len(arcs) == len(added)
    assert max(abs(math.degrees(math.atan2(float(p["qy"]), float(p["qx"])))) for p in added) > 25.0

    empty = np.full_like(image, 0.2)
    axial_only = [
        {
            "qx": 0.50,
            "qy": 0.02,
            "accepted": True,
            "branch_id": 0,
            "side": "upper",
            "point_id": "axial-plus",
        }
    ]
    none = _fill_sparse_first_order_ring(
        axial_only,
        qx=qx,
        qy=qy,
        q=q,
        intensity=empty,
        valid=valid,
        hint=0.50,
        options=options,
        q_step=0.025,
        signature="test-empty",
    )
    assert none["applied"] is False
    assert none["added"] == 0
    assert none["reason"] == "no_observed_sector_peak"
