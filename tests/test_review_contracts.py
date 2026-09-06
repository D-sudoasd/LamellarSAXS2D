from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import threading

import numpy as np
import pytest

from butterfly_saxs import butterfly, butterfly_ridge, observables, pipeline
from butterfly_saxs.batch import FrameRef, _warm_start_seed, run_batch
from butterfly_saxs.cancellation import AnalysisCancelled
from butterfly_saxs.project import ProjectConfig, load_project, save_project
from butterfly_saxs.service import ButterflyAnalysisService


def test_new_butterfly_trace_never_becomes_a_warm_start_seed() -> None:
    trace = {
        "method_version": "butterfly-curvature-arcs-v2",
        "settings": {"stage": "trace"},
        "points": [{"accepted": True, "qx": 0.1, "qy": 0.2}],
    }
    assert _warm_start_seed(trace) is None
    assert _warm_start_seed({"observables": {"butterfly": trace}}) is None
    assert _warm_start_seed({"butterfly": {"parameters": {"a": 1.0}}}) is None

    eligible = {
        "method_version": "butterfly-curvature-arcs-v2",
        "warm_start_eligible": True,
        "candidate_fit": {
            "parameters": {
                "a": 1.0,
                "b": 0.5,
                "axis_ratio": 0.5,
                "theta_deg": 12.0,
            }
        },
    }
    assert _warm_start_seed(eligible) == {
        "a": 1.0,
        "b": 0.5,
        "axis_ratio": 0.5,
        "theta_deg": 12.0,
    }
    assert _warm_start_seed(
        {
            "butterfly": {
                "warm_start_eligible": True,
                "candidate_fit": {
                    "a": 1.0,
                    "b": 0.5,
                    "axis_ratio": 0.5,
                    "theta_deg": 12.0,
                },
            }
        }
    ) == {"a": 1.0, "b": 0.5, "axis_ratio": 0.5, "theta_deg": 12.0}
    eligible["candidate_fit"]["parameters"]["b"] = float("nan")
    assert _warm_start_seed(eligible) is None
    assert _warm_start_seed(
        {"ellipse_fit": {"warm_start_eligible": True}, "parameters": {"a": 2.0}}
    ) == {"a": 2.0}


def test_root_trace_payload_does_not_seed_batch_next_frame(tmp_path: Path) -> None:
    paths = []
    for index in range(2):
        path = tmp_path / f"frame_{index}.npy"
        np.save(path, np.ones((3, 3)))
        paths.append(FrameRef(path, frame_id=str(index)))
    received: list[object] = []

    def analyzer(_frame, initial=None, **_kwargs):
        received.append(initial)
        return {
            "method_version": "butterfly-curvature-arcs-v2",
            "settings": {"stage": "trace"},
            "points": [{"accepted": True}],
        }

    result = run_batch(paths, analyzer, mode="warm_start")
    assert received == [None, None]
    assert [item.warm_start_from for item in result.frame_results] == [None, None]


def test_service_direct_analyze_frame_ignores_trace_envelope_as_initial(monkeypatch) -> None:
    service = ButterflyAnalysisService()
    captured: dict[str, object] = {}

    monkeypatch.setattr(service, "load_image", lambda *_args, **_kwargs: {})

    def fake_optimize(**kwargs):
        captured["parameters"] = kwargs["parameters"]
        return {}

    monkeypatch.setattr(service, "optimize", fake_optimize)
    trace = {
        "method_version": "butterfly-curvature-arcs-v2",
        "settings": {"stage": "trace"},
        "points": [{"accepted": True}],
    }
    service.analyze_frame("frame.npy", initial=trace, warm_start=True)
    assert captured["parameters"] == service.parameters


def test_pipeline_analysis_emits_replayable_butterfly_recipe(monkeypatch) -> None:
    def fake_measure(*_args, **_kwargs):
        return {
            "ridge": {"points": []},
            "ellipse": {"success": False, "parameters": {}, "ellipses": []},
        }

    monkeypatch.setattr(pipeline, "measure_observables", fake_measure)
    image = np.ones((5, 5), dtype=float)
    y, x = np.indices(image.shape, dtype=float)
    qmap = {"qx": x - 2.0, "qy": y - 2.0, "q_unit": "1/nm"}
    recipe = {"stage": "trace", "resamples": 3, "sensitivity": False}
    result = pipeline.analyze_frame(
        image,
        qmap=qmap,
        config={
            "analysis": {
                "ridge_method": "butterfly_curvature",
                "q_window": [0.1, 2.9],
                "butterfly": recipe,
            }
        },
        full2d=False,
    )
    assert result.analysis["butterfly"] == {
        "stage": "trace",
        "resamples": 3,
        "seed": 20260906,
        "edits": [],
        "sensitivity": False,
    }
    assert "butterfly_options" not in result.analysis


def test_project_toml_round_trip_preserves_butterfly_mapping_list(tmp_path: Path) -> None:
    edits = [
        {"type": "seed", "qx": 0.2, "qy": -0.3, "branch_id": 1, "side": "upper"},
        {
            "type": "exclude_polygon",
            "points": [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        },
        {"type": "exclude_point", "point_id": "pixel:2:3"},
    ]
    source = ProjectConfig(
        inputs=["frame.npy"],
        analysis={
            "ridge_method": "butterfly_curvature",
            "butterfly": {"stage": "trace", "edits": edits},
        },
    )
    path = tmp_path / "project.toml"
    save_project(source, path)
    loaded = load_project(path)
    assert loaded.analysis["butterfly"]["edits"] == edits
    assert "edits = [" in path.read_text(encoding="utf-8")


def test_analyze_butterfly_returns_strict_json_without_mutating_inputs(monkeypatch) -> None:
    raw_qx = np.arange(4, dtype=float).reshape(2, 2)
    raw_qy = raw_qx + 10.0
    qmap = {"qx": raw_qx.copy(), "qy": raw_qy.copy(), "q_unit": "1/nm"}
    image = np.ones((2, 2), dtype=float)
    original_qx = qmap["qx"].copy()

    def fake_trace(*_args, **_kwargs):
        return {
            "points": [
                {
                    "accepted": False,
                    "valid": False,
                    "branch_id": -1,
                    "side": "unknown",
                    "normal_fwhm_q": np.float32(np.nan),
                    "qx": np.float64(0.25),
                }
            ],
            "arcs": [],
            "profiles": {"p": {"raw_intensity": np.array([1.0, np.nan])}},
            "diagnostics": {},
            "method_version": "butterfly-observed-ridges-v1.0",
        }

    monkeypatch.setattr(butterfly_ridge, "trace_butterfly_ridges", fake_trace)
    result = butterfly.analyze_butterfly(
        image,
        qmap,
        (0.0, 20.0),
        options={"stage": "trace", "sensitivity": False},
    )
    json.dumps(result, allow_nan=False)
    assert result["points"][0]["normal_fwhm_q"] is None
    assert result["points"][0]["qx"] == pytest.approx(0.25)
    assert result["profiles"]["p"]["raw_intensity"] == [1.0, None]
    np.testing.assert_array_equal(qmap["qx"], original_qx)


def test_post_trace_lobe_stages_honor_cancellation(monkeypatch) -> None:
    cancel = threading.Event()
    monkeypatch.setattr(
        butterfly,
        "analyze_butterfly",
        lambda *_args, **_kwargs: {"settings": {"snr_threshold": 1.0}},
    )

    def cancelled_angular(*_args, **_kwargs):
        cancel.set()
        return None

    monkeypatch.setattr(observables, "measure_angular_spectrum", cancelled_angular)
    with pytest.raises(AnalysisCancelled, match="angular-spectrum"):
        butterfly.measure_butterfly_observables(
            np.ones((3, 3)),
            {"qx": np.zeros((3, 3)), "qy": np.zeros((3, 3))},
            (0.0, 1.0),
            cancel_event=cancel,
        )


def test_fit_trace_uses_canonical_envelope_and_preserves_arc_mapping(monkeypatch) -> None:
    fit = SimpleNamespace(
        values={"cx": 0.0, "cy": 0.0, "a": 1.0, "axis_ratio": 0.5, "b": 0.5, "theta": 0.2},
        residuals=np.asarray([0.01, 0.02]),
        success=True,
        message="ok",
        cost=0.1,
        optimality=0.0,
        nfev=2,
        covariance=None,
        stderr={},
        condition=2.0,
        bound_flags={},
        bound_status={},
        coverage={"n_points": 6},
        candidate_solutions=(),
        selected_start_index=0,
        multistart_count=1,
        branch_assignment=np.asarray([0, 1]),
    )
    arc_diagnostics = [{"arc_id": 17, "projection_span": 0.4}]
    point_diagnostics = [
        {
            "distance_q": 0.01,
            "arc_id": 17,
            "branch_id": 0,
            "side": "upper",
        }
    ]

    def fake_fit(*_args, **_kwargs):
        return {
            "fit": fit,
            "point_diagnostics": point_diagnostics,
            "arc_diagnostics": arc_diagnostics,
            "branch_swap_applied": True,
        }

    monkeypatch.setattr(butterfly, "strict_jsonable", lambda value: value)
    monkeypatch.setattr(
        __import__("butterfly_saxs.arc_geometry", fromlist=["fit_arc_ellipses"]),
        "fit_arc_ellipses",
        fake_fit,
    )
    points = [
        {"accepted": True, "branch_id": branch, "side": side}
        for branch in (0, 1)
        for side in ("upper", "lower")
        for _ in range(2)
    ]
    result = butterfly._fit_trace(
        {"points": points},
        parameters=None,
        reference=10.0,
        multistart=1,
        unit="1/nm",
        cancel_event=None,
    )
    assert result["parameters"]["theta_deg"] == pytest.approx(np.degrees(0.2))
    assert [row["branch_id"] for row in result["ellipses"]] == [1, 0]
    assert result["branch_label_mapping"]["fit_branch_to_observed"] == {"0": 1, "1": 0}
    assert result["arc_diagnostics"] == arc_diagnostics
    assert result["point_diagnostics"] == point_diagnostics


def test_failed_arc_fit_keeps_observed_diagnostics(monkeypatch) -> None:
    from butterfly_saxs import arc_geometry

    points = [
        {"accepted": True, "branch_id": branch, "side": side}
        for branch in (0, 1)
        for side in ("upper", "lower")
        for _ in range(2)
    ]
    trace = {
        "points": points,
        "arcs": [{"arc_id": 4, "support_valid": False}],
        "diagnostics": {"trace_status": "partial"},
    }
    monkeypatch.setattr(
        arc_geometry,
        "fit_arc_ellipses",
        lambda *_args, **_kwargs: {
            "fit": None,
            "message": "support infeasible",
            "point_diagnostics": [{"index": 0, "excluded_reason": "support_infeasible"}],
            "arc_diagnostics": [{"arc_id": 4, "support_valid": False}],
        },
    )
    result = butterfly._fit_trace(
        trace,
        parameters=None,
        reference=0.0,
        multistart=1,
        unit="1/nm",
        cancel_event=None,
    )
    assert result["success"] is False
    assert result["arc_diagnostics"] == [{"arc_id": 4, "support_valid": False}]
    assert result["point_diagnostics"] == [{"index": 0, "excluded_reason": "support_infeasible"}]


def test_fit_trace_forwards_observed_center_without_defaulting(monkeypatch) -> None:
    from butterfly_saxs import arc_geometry

    points = [
        {"accepted": True, "branch_id": branch, "side": side}
        for branch in (0, 1)
        for side in ("upper", "lower")
        for _ in range(2)
    ]
    seen: list[object] = []

    def fake_fit(*_args, **kwargs):
        seen.append(kwargs.get("reference_center"))
        return {"fit": None, "message": "diagnostic-only"}

    monkeypatch.setattr(arc_geometry, "fit_arc_ellipses", fake_fit)
    butterfly._fit_trace(
        {"points": points, "diagnostics": {"center_q": [0.21, -0.14]}},
        parameters=None,
        reference=0.0,
        multistart=1,
        unit="1/nm",
        cancel_event=None,
    )
    butterfly._fit_trace(
        {"points": points, "diagnostics": {}},
        parameters=None,
        reference=0.0,
        multistart=1,
        unit="1/nm",
        cancel_event=None,
    )
    assert seen == [(0.21, -0.14), None]


def test_holdout_projection_uses_frozen_support_and_counts_invalid_points(monkeypatch) -> None:
    from butterfly_saxs import arc_geometry

    captured: dict[str, object] = {}

    def fake_project(records, values, **kwargs):
        captured["records"] = records
        captured["values"] = values
        captured["kwargs"] = kwargs
        return {
            "point_diagnostics": [
                {"point_id": "p0", "distance_q": 0.1, "projection_valid": True},
                {"point_id": "p1", "distance_q": None, "projection_valid": False},
            ]
        }

    monkeypatch.setattr(
        arc_geometry,
        "project_arc_points",
        fake_project,
    )
    records = [
        {"point_id": "p0", "accepted": True, "support_status": "frozen"},
        {"point_id": "p1", "accepted": True, "support_status": "frozen"},
    ]
    prediction = butterfly._holdout_prediction(
        records,
        {"cx": 0.0, "cy": 0.0, "a": 1.0, "axis_ratio": 0.5, "theta": 0.2},
        reference_axis_deg=7.0,
        branch_swap_applied=True,
    )
    assert captured["records"] is records
    assert captured["kwargs"] == {
        "reference_axis_deg": 7.0,
        "branch_swap_applied": True,
    }
    assert prediction["success"] is False
    assert prediction["n_points"] == 2
    assert prediction["n_valid"] == 1
    assert prediction["n_invalid"] == 1
    assert prediction["invalid_point_ids"] == ["p1"]
    assert prediction["predictive_rmse_q"] is None


def test_real_arc_projection_helper_is_used_for_frozen_support() -> None:
    from butterfly_saxs.arc_geometry import project_arc_points

    records = [
        {
            "point_id": "p0",
            "qx": 1.0,
            "qy": 0.0,
            "accepted": True,
            "branch_id": 0,
            "side": "upper",
            "arc_id": 0,
            "localization_sigma_q": 0.1,
            "support_status": "frozen",
            "support_components": [
                {
                    "rectangles": [
                        {
                            "frame_origin_q": [1.0, 0.0],
                            "tangent_q": [0.0, 1.0],
                            "normal_q": [-1.0, 0.0],
                            "tangent_bounds": [-1.0, 1.0],
                            "normal_bounds": [-1.0, 1.0],
                        }
                    ]
                }
            ],
        }
    ]
    result = project_arc_points(
        records,
        {"cx": 0.0, "cy": 0.0, "a": 1.0, "axis_ratio": 0.5, "theta": 0.0},
    )
    assert result["projection_version"] == "observed_arc_projection_v1"
    assert result["n_input"] == 1
    assert result["n_valid"] == 1
    assert result["n_invalid"] == 0
