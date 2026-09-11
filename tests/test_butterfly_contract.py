from __future__ import annotations

import numpy as np
import pytest

from butterfly_saxs.analysis_config import validate_analysis_settings
from butterfly_saxs.batch import FrameRef, run_batch
from butterfly_saxs.butterfly_quality import (
    classify_ellipse_publication,
    evaluate_arc_evidence,
    unpublished_ellipse_shape,
)
from butterfly_saxs.butterfly_settings import normalize_butterfly_settings
from butterfly_saxs.export import _parameters
from butterfly_saxs.pipeline import _ridges_from_observable_bundle


def test_butterfly_reads_poni_q_unit_from_geometry_metadata() -> None:
    from butterfly_saxs.butterfly import _qmap_unit

    class GeometryMapsLike:
        metadata = {"q_unit": "nm^-1"}

    assert _qmap_unit(GeometryMapsLike()) == "nm^-1"
    assert _qmap_unit({"qx": 1, "metadata": {"q_unit": "nm^-1"}}) == "nm^-1"
    assert _qmap_unit({"q_unit": "pixel-q", "metadata": {"q_unit": "nm^-1"}}) == "pixel-q"
    assert _qmap_unit({"qx": 1}) == "unknown"


def test_old_method_is_preserved_and_new_recipe_is_independent():
    assert validate_analysis_settings({})["ridge_method"] == "radial_peak"
    assert validate_analysis_settings({})["ellipse"]["preset"] == "standard"
    recipe = {"edits": [{"type": "seed", "qx": .2, "qy": .3, "side": "upper"}]}
    settings = validate_analysis_settings({"ridge_method": "butterfly_curvature", "butterfly": recipe})
    settings["butterfly"]["edits"][0]["qx"] = 10.
    assert recipe["edits"][0]["qx"] == .2
    assert settings["ellipse"]["preset"] == "flat_ellipse"
    assert settings["ellipse"]["axis_ratio_min"] == pytest.approx(0.005)
    assert settings["ellipse"]["axis_ratio_max"] == pytest.approx(0.35)
    named = validate_analysis_settings(
        {"ridge_method": "butterfly_curvature", "ellipse_preset": "very_flat_ellipse"}
    )
    assert named["ellipse"]["preset"] == "very_flat_ellipse"
    standard = validate_analysis_settings(
        {"ridge_method": "butterfly_curvature", "ellipse_preset": "standard"}
    )
    assert standard["ellipse"]["preset"] == "standard"


@pytest.mark.parametrize("recipe", [
    {"resamples": 2.5}, {"resamples": -1}, {"seed": float("nan")},
    {"stage": "anything"}, {"edits": [{"type": "seed", "qx": 0., "qy": float("inf")}]},
    {"edits": [{"type": "exclude_polygon", "points": [[0, 0], [1, 1], [2, 2]]}]},
])
def test_invalid_recipe_fails_before_any_image_processing(recipe):
    with pytest.raises((ValueError, TypeError)):
        normalize_butterfly_settings(recipe)


def test_ellipse_publication_classifies_ring_vs_interior_fit() -> None:
    assert classify_ellipse_publication(quality_status="FAIL") == "fail"
    assert (
        classify_ellipse_publication(
            quality_status="WARN",
            axis_ratio=0.30,
        )
        == "ellipse"
    )
    assert (
        classify_ellipse_publication(
            quality_status="WARN",
            axis_ratio=0.35,
            flags=["axis_ratio_at_bound"],
        )
        == "ring"
    )
    assert unpublished_ellipse_shape(
        quality_status="WARN",
        flags=["axis_ratio_at_bound"],
    )
    assert not unpublished_ellipse_shape(
        quality_status="WARN",
        axis_ratio=0.30,
    )


def test_two_occupied_sides_are_an_engineering_fail() -> None:
    points = [
        {
            "accepted": True,
            "branch_id": 0,
            "side": side,
            "localization_sigma_q": 0.001,
        }
        for side in ("upper", "lower")
        for _ in range(5)
    ]
    candidate = {
        "success": True,
        "a": 1.0,
        "b": 0.01,
        "axis_ratio": 0.01,
        "theta_deg": 10.0,
        "condition": 2.0,
        "rmse": 0.0001,
    }
    result = evaluate_arc_evidence({"points": points}, candidate)
    assert result["quality"]["status"] == "FAIL"
    assert result["quality"]["metrics"]["occupied_sides"] == 2
    assert "insufficient_occupied_sides" in result["quality"]["flags"]


def test_axis_ratio_floor_is_a_ring_warning() -> None:
    points = [
        {
            "accepted": True,
            "branch_id": branch,
            "side": side,
            "localization_sigma_q": 0.001,
        }
        for branch in (0, 1)
        for side in ("upper", "lower")
        for _ in range(5)
    ]
    candidate = {
        "success": True,
        "a": 0.11,
        "b": 0.00055,
        "axis_ratio": 0.005,
        "theta_deg": 8.0,
        "condition": 2.0,
        "rmse": 0.0001,
        "bound_flags": {"axis_ratio": True},
    }
    result = evaluate_arc_evidence({"points": points}, candidate)
    assert result["quality"]["status"] == "WARN"
    assert "axis_ratio_collapsed_to_line" in result["quality"]["flags"]
    flagged = dict(candidate)
    flagged.pop("bound_flags")
    flagged["flags"] = ["axis_ratio_at_bound"]
    via_flag = evaluate_arc_evidence({"points": points}, flagged)
    assert via_flag["quality"]["status"] == "WARN"
    assert classify_ellipse_publication(
        quality_status="WARN",
        axis_ratio=None,
        flags=["axis_ratio_collapsed_to_line", "axis_ratio_at_bound"],
    ) == "ring"


def test_major_axis_beyond_observed_ridge_is_a_ring_warning() -> None:
    points = [
        {
            "accepted": True,
            "branch_id": branch,
            "side": side,
            "qx": 0.11 if branch == 0 else -0.10,
            "qy": 0.02 if side == "upper" else -0.02,
            "localization_sigma_q": 0.001,
        }
        for branch in (0, 1)
        for side in ("upper", "lower")
        for _ in range(3)
    ]
    q_extent = max((0.11**2 + 0.02**2) ** 0.5, (0.10**2 + 0.02**2) ** 0.5)
    result = evaluate_arc_evidence(
        {"points": points},
        {
            "success": True,
            "a": 1.8 * q_extent,
            "b": 0.16 * q_extent,
            "axis_ratio": 0.087,
            "theta_deg": 8.0,
            "condition": 2.0,
            "rmse": 0.0001,
        },
    )
    assert result["quality"]["status"] == "WARN"
    assert "major_axis_exceeds_observed_extent" in result["quality"]["flags"]
    assert classify_ellipse_publication(
        quality_status="WARN",
        axis_ratio=0.087,
        flags=["major_axis_exceeds_observed_extent"],
    ) == "ring"
    supported = evaluate_arc_evidence(
        {"points": points},
        {
            "success": True,
            "a": 0.95 * q_extent,
            "b": 0.30 * q_extent,
            "axis_ratio": 0.30,
            "theta_deg": 8.0,
            "condition": 2.0,
            "rmse": 0.0001,
        },
    )
    assert "major_axis_exceeds_observed_extent" not in supported["quality"]["flags"]
    harmonic = dict(points[0])
    harmonic.update({"qx": 0.55, "qy": 0.02, "branch_id": 0, "side": "upper"})
    inflated = evaluate_arc_evidence(
        {
            "points": [*points, harmonic],
            "diagnostics": {"first_order_q_hint": {"q_star": 0.092}},
        },
        {
            "success": True,
            "a": 1.8 * q_extent,
            "b": 0.16 * q_extent,
            "axis_ratio": 0.087,
            "theta_deg": 8.0,
            "condition": 2.0,
            "rmse": 0.0001,
        },
    )
    assert "major_axis_exceeds_observed_extent" in inflated["quality"]["flags"]


def test_major_axis_beyond_first_order_q_star_is_unpublished() -> None:
    points = [
        {
            "accepted": True,
            "branch_id": branch,
            "side": side,
            "qx": 0.22 if branch == 0 else -0.22,
            "qy": 0.05 if side == "upper" else -0.05,
            "localization_sigma_q": 0.001,
        }
        for branch in (0, 1)
        for side in ("upper", "lower")
        for _ in range(3)
    ]
    q_extent = (0.22**2 + 0.05**2) ** 0.5
    result = evaluate_arc_evidence(
        {
            "points": points,
            "diagnostics": {"first_order_q_hint": {"q_star": 0.092}},
        },
        {
            "success": True,
            "a": 0.26,
            "b": 0.048,
            "axis_ratio": 0.18,
            "theta_deg": 24.0,
            "condition": 2.0,
            "rmse": 0.0001,
        },
    )
    assert 0.26 < 1.2 * q_extent
    assert 0.26 > 2.0 * 0.092
    assert result["quality"]["status"] == "WARN"
    assert "major_axis_exceeds_observed_extent" in result["quality"]["flags"]


def test_unknown_side_and_missing_uncertainty_cannot_become_quantitative():
    points = [{"accepted": True, "branch_id": branch, "side": "unknown",
               "localization_sigma_q": .001} for branch in (0, 1) for _ in range(20)]
    candidate = {"success": True, "a": 1., "b": .01, "axis_ratio": .01,
                 "theta_deg": 40., "condition": 2., "rmse": .0001}
    result = evaluate_arc_evidence({"points": points}, candidate)
    assert not result["warm_start_eligible"]
    for row in result["quantitative_parameters"].values():
        assert row["value"] is None
        assert "insufficient_independent_side_support" in row["reasons"]


def test_unidentified_parameter_csv_retains_candidate_without_publishing_it():
    result = {"parameters": {"a": 1., "b": .01}, "ellipse_fit": {
        "quantitative_parameters": {"a": {"value": None, "candidate_value": .5,
            "status": "undetermined", "reason": "short_arc", "interval": [.3, .8],
            "interval_kind": "empirical_central_95_percent", "unit": "nm^-1"}}}}
    rows = {r["parameter"]: r for r in _parameters(result)}
    assert rows["a"]["value"] == ""
    assert rows["a"]["candidate_value"] == .5
    assert rows["a"]["interval_low"] == .3
    assert rows["a"]["identifiability_reason"] == "short_arc"


def test_rejected_new_point_remains_visible_for_correction():
    data = {"ridge": {"points": [{"point_id": "pixel:3:4", "qx": .2, "qy": .3,
                                   "valid": False, "side": "unknown"}]}}
    point = _ridges_from_observable_bundle(data)[0]
    assert point["qx"] == .2
    assert point["qy"] == .3
    assert point["valid"] is False


def test_unidentified_frame_cannot_seed_the_next_frame(tmp_path):
    refs = []
    for index in range(2):
        path = tmp_path / f"{index}.npy"
        np.save(path, np.ones((3, 3)))
        refs.append(FrameRef(path, frame_id=str(index)))
    received = []

    def analyzer(frame, initial=None, **kwargs):
        received.append(initial)
        return {"parameters": {"a": 1.}, "ellipse_fit": {
            "success": True, "warm_start_eligible": False, "quality_status": "WARN"}}

    run = run_batch(refs, analyzer, mode="warm_start")
    assert received == [None, None]
    assert run.frame_results[1].warm_start_from is None


def test_flat_butterfly_evaluate_recovers_axes_and_origin_centered_periods() -> None:
    from butterfly_saxs.benchmark_arcs import generate_arc_case
    from butterfly_saxs.butterfly import analyze_butterfly

    case = generate_arc_case("ellipse_ratio_020", seed=502, shape=(96, 96))
    result = analyze_butterfly(
        case["image"],
        case["qmap"],
        (0.05, 1.1),
        mask=case.get("mask"),
        options={"stage": "evaluate", "resamples": 0, "sensitivity": False, "run_wang_check": False},
        reference_axis_deg=0.0,
        multistart=5,
    )
    fit = result["candidate_fit"]
    sides = {
        (point.get("branch_id"), point.get("side"))
        for point in result.get("points", [])
        if point.get("accepted") and point.get("side") in ("upper", "lower")
    }
    assert fit["success"] is True
    assert sides == {(0, "upper"), (0, "lower"), (1, "upper"), (1, "lower")}
    assert fit["a"] == pytest.approx(case["truth"]["a"], rel=0.08)
    assert fit["axis_ratio"] == pytest.approx(case["truth"]["axis_ratio"], rel=0.15)
    assert fit["theta_deg"] == pytest.approx(17.0, abs=1.0)
    assert fit["Ln_from_minor_axis_nm"] == pytest.approx(2.0 * np.pi / fit["b"], rel=1e-6)
    assert fit["ellipticity"] == pytest.approx((1.0 - fit["axis_ratio"] ** 2) ** 0.5, rel=1e-6)
    assert "spacing_requires_origin_centered_ellipse_assumption" in fit["flags"]
    assert fit["L_from_observed_radius_nm"] == pytest.approx(
        2.0 * np.pi / fit["q_star_from_arcs"], rel=1e-6
    )
    assert "spacing_from_first_order_iq" in fit["flags"]
    assert fit.get("q_star_source") == "first_order_iq"


def test_very_flat_butterfly_evaluate_keeps_ratio_above_line_collapse() -> None:
    from butterfly_saxs.benchmark_arcs import generate_arc_case
    from butterfly_saxs.butterfly import analyze_butterfly

    case = generate_arc_case("ellipse_ratio_005", seed=501, shape=(80, 80))
    result = analyze_butterfly(
        case["image"],
        case["qmap"],
        (0.05, 1.1),
        mask=case.get("mask"),
        options={"stage": "evaluate", "resamples": 0, "sensitivity": False, "run_wang_check": False},
        reference_axis_deg=0.0,
        multistart=5,
    )
    fit = result["candidate_fit"]
    sides = {
        (point.get("branch_id"), point.get("side"))
        for point in result.get("points", [])
        if point.get("accepted") and point.get("side") in ("upper", "lower")
    }
    assert fit["success"] is True
    assert sides == {(0, "upper"), (0, "lower"), (1, "upper"), (1, "lower")}
    assert fit["a"] == pytest.approx(case["truth"]["a"], rel=0.15)
    assert fit["axis_ratio"] >= 0.005 - 1e-9
    assert fit["axis_ratio"] == pytest.approx(case["truth"]["axis_ratio"], rel=0.8, abs=0.004)
    assert fit["theta_deg"] == pytest.approx(17.0, abs=2.0)
    assert fit["Ln_from_minor_axis_nm"] == pytest.approx(2.0 * np.pi / fit["b"], rel=1e-6)


def test_service_and_pipeline_share_actual_butterfly_measurement(tmp_path):
    from butterfly_saxs.benchmark_arcs import generate_arc_case
    from butterfly_saxs.pipeline import analyze_frame
    from butterfly_saxs.service import ButterflyAnalysisService

    case = generate_arc_case("ellipse_ratio_400", seed=123, shape=(64, 64))
    settings = {"ridge_method": "butterfly_curvature", "q_window": [.05, 1.1],
                "ellipse_multistart": 2,
                "butterfly": {"stage": "evaluate", "resamples": 0, "sensitivity": False}}
    service = ButterflyAnalysisService(analysis_settings=settings)
    state = service.set_observed(case["image"], qmap=case["qmap"])
    gui_result = service.measure_geometry(payload=state)
    cli_result = analyze_frame(case["image"], qmap=case["qmap"],
                               config={"analysis": settings}, full2d=False)
    assert gui_result["butterfly"] is not None
    for key in ("a", "b", "axis_ratio", "theta_deg", "rmse"):
        assert gui_result["ellipse_fit"][key] == pytest.approx(cli_result.ellipse_fit[key], nan_ok=True)
    ui_points = gui_result["butterfly"]["points"]
    pipeline_points = cli_result.butterfly["points"]
    assert [(p["branch_id"], p["side"], p["accepted"]) for p in ui_points] == [
        (p["branch_id"], p["side"], p["accepted"]) for p in pipeline_points]
    np.testing.assert_allclose([[p["qx"], p["qy"]] for p in ui_points],
                               [[p["qx"], p["qy"]] for p in pipeline_points])
    import csv
    from butterfly_saxs.pipeline import export_result

    target = tmp_path / "parameters.csv"
    export_result(cli_result, target)
    with target.open(encoding="utf-8") as handle:
        rows = {row["parameter"]: row for row in csv.DictReader(handle)}
    assert rows["a"]["value"] == ""
    assert float(rows["a"]["candidate_value"]) == pytest.approx(cli_result.ellipse_fit["a"])
    assert rows["a"]["identifiability_status"] == "undetermined"
    assert set(cli_result.butterfly["ellipse_local"]) == {"0", "1"}
    assert any(profile.get("residual") for profile in cli_result.butterfly["profiles"].values())


def test_public_extract_ridges_routes_new_method_without_fitting():
    from butterfly_saxs.benchmark_arcs import generate_arc_case
    from butterfly_saxs.pipeline import extract_ridges

    case = generate_arc_case("ellipse_ratio_400", seed=71, shape=(48, 48))
    points = extract_ridges(case["image"], case["qmap"], config={
        "ridge_method": "butterfly_curvature", "q_window": [.05, 1.1]})
    assert points
    assert all("arc_id" in point and "side" in point for point in points)


def test_real_pipeline_trace_checkpoint_does_not_recompute_completed_frame(tmp_path):
    import threading
    from butterfly_saxs.benchmark_arcs import generate_arc_case
    from butterfly_saxs.pipeline import analyze_frame

    case = generate_arc_case("ellipse_ratio_400", seed=73, shape=(48, 48))
    refs = []
    for index in (1, 2):
        path = tmp_path / f"sample_{index:05d}.npy"
        np.save(path, case["image"])
        refs.append(FrameRef(path, frame_id=str(index)))
    settings = {"ridge_method": "butterfly_curvature", "q_window": [.05, 1.1],
                "butterfly": {"stage": "trace", "resamples": 0, "sensitivity": False}}
    cancel = threading.Event()
    calls = []

    def analyzer(frame, **kwargs):
        calls.append(frame.frame_id)
        return analyze_frame(frame.path, qmap=case["qmap"], config={"analysis": settings}, full2d=False)

    def stop_after_first(event):
        if event.get("completed", 0) >= 1:
            cancel.set()

    checkpoint = tmp_path / "checkpoint.json"
    first = run_batch(refs, analyzer, config=settings, checkpoint=checkpoint,
                       cancel_event=cancel, progress=stop_after_first)
    assert first.cancelled
    cancel.clear()
    resumed = run_batch(refs, analyzer, config=settings, checkpoint=checkpoint,
                        resume=True, cancel_event=cancel)
    assert calls == ["1", "2"]
    assert resumed.frame_results[0].resumed
