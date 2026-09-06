from __future__ import annotations

import numpy as np
import pytest

from butterfly_saxs.analysis_config import validate_analysis_settings
from butterfly_saxs.batch import FrameRef, run_batch
from butterfly_saxs.butterfly_quality import evaluate_arc_evidence
from butterfly_saxs.butterfly_settings import normalize_butterfly_settings
from butterfly_saxs.export import _parameters
from butterfly_saxs.pipeline import _ridges_from_observable_bundle


def test_old_method_is_preserved_and_new_recipe_is_independent():
    assert validate_analysis_settings({})["ridge_method"] == "radial_peak"
    recipe = {"edits": [{"type": "seed", "qx": .2, "qy": .3, "side": "upper"}]}
    settings = validate_analysis_settings({"ridge_method": "butterfly_curvature", "butterfly": recipe})
    settings["butterfly"]["edits"][0]["qx"] = 10.
    assert recipe["edits"][0]["qx"] == .2


@pytest.mark.parametrize("recipe", [
    {"resamples": 2.5}, {"resamples": -1}, {"seed": float("nan")},
    {"stage": "anything"}, {"edits": [{"type": "seed", "qx": 0., "qy": float("inf")}]},
    {"edits": [{"type": "exclude_polygon", "points": [[0, 0], [1, 1], [2, 2]]}]},
])
def test_invalid_recipe_fails_before_any_image_processing(recipe):
    with pytest.raises((ValueError, TypeError)):
        normalize_butterfly_settings(recipe)


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
