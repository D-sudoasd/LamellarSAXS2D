from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6 import QtCore, QtTest

from butterfly_saxs.ui import MainWindow
from butterfly_saxs.ui.butterfly_workbench import ButterflyWorkbench


def _non_affine_frame(shape: tuple[int, int] = (20, 20)):
    yy, xx = np.indices(shape, dtype=float)
    observed = 1.0 + 0.1 * xx + 0.2 * yy
    qx = 0.01 * xx + 0.002 * yy**2
    qy = 0.008 * yy + 0.001 * xx**2
    return observed, qx, qy


def _landmark_result(qx: np.ndarray, qy: np.ndarray) -> dict:
    gx, gy = 2, 8
    px, py = 15, 3
    return {
        "measurement_status": "evaluated",
        "points": [
            {
                "point_id": "ridge-1",
                "qx": 0.48,
                "qy": 0.22,
                "branch_id": 0,
                "side": "upper",
                "valid": True,
                "accepted": True,
            }
        ],
        "peak_landmarks": {
            "schema_version": "peak-landmarks-v1",
            "status": "ok",
            "q_unit": "1/nm",
            "domain": {"search_q_window": [0.02, 1.1]},
            "raw_global_max": {
                "pixel_x": gx,
                "pixel_y": gy,
                "qx": float(qx[gy, gx]),
                "qy": float(qy[gy, gx]),
                "q": float(math.hypot(qx[gy, gx], qy[gy, gx])),
                "raw_intensity": 100.0,
                "flags": [],
                "interpretation": "raw_maximum_only_not_an_identified_reflection",
            },
            "peaks": [
                {
                    "peak_id": "P1",
                    "angular_peak_deg": 56.0,
                    "chi_deg": 55.0,
                    "pixel_x": px,
                    "pixel_y": py,
                    "qx": float(qx[py, px]),
                    "qy": float(qy[py, px]),
                    "q": float(math.hypot(qx[py, px], qy[py, px])),
                    "raw_intensity": 80.0,
                    "smoothed_intensity": 78.0,
                    "angular_prominence": 9.5,
                    "support_pixel_count": 12,
                    "flags": [],
                },
                {
                    # A q and detector coordinate without qx/qy must not be
                    # converted to a displayed location by interpolation.
                    "peak_id": "P2",
                    "pixel_x": 4,
                    "pixel_y": 12,
                    "q": 0.72,
                    "raw_intensity": 60.0,
                    "smoothed_intensity": 58.0,
                    "flags": [],
                },
            ],
            "profiles": {
                "angular": {
                    "angle_deg": [-180.0, -60.0, 0.0, 56.0, 120.0, 180.0],
                    "intensity_raw": [1.0, 2.0, 1.5, 8.0, 2.0, 1.0],
                    "intensity_smoothed": [1.1, 2.1, 1.7, 7.5, 2.0, 1.1],
                    "coverage_fraction": [1.0] * 6,
                },
                "radial": {
                    "q": [0.0, 0.15, 0.30, 0.45, 0.60],
                    "mean_intensity_raw": [1.0, 4.0, 7.0, 3.0, 1.0],
                    "mean_intensity_smoothed": [1.1, 3.8, 6.5, 3.2, 1.1],
                    "coverage_fraction": [1.0] * 5,
                },
            },
        },
        "profiles": {},
    }


def _poor_geometry_result(*, residual_ratio: float = 10.2) -> dict:
    return {
        "measurement_status": "evaluated",
        "points": [],
        "profiles": {},
        "candidate_fit": {
            "success": True,
            "a": 0.42,
            "b": 0.042,
            "axis_ratio": 0.1,
            "theta_deg": 5.0,
            "center_qx": 0.0,
            "center_qy": 0.0,
            "reference_axis_deg": 0.0,
        },
        "quality": {
            "status": "WARN",
            "metrics": {
                "residual_sigma_ratio": residual_ratio,
                "median_localization_sigma_q": 0.01,
            },
            "provisional_limits": {"residual_sigma_ratio_max": 3.0},
            "flags": [],
        },
    }


def test_landmarks_are_separate_and_use_exact_non_affine_q_coordinates(qtbot):
    observed, qx, qy = _non_affine_frame()
    page = ButterflyWorkbench(language="en")
    qtbot.addWidget(page)
    page.resize(1180, 820)
    page.set_data(observed, qx=qx, qy=qy, q_unit="1/nm")
    page.set_q_window((0.0, 1.2))
    page.set_result(_landmark_result(qx, qy))
    page.show()
    qtbot.wait(80)

    assert page.point_list.count() == 1
    assert list(page.qspace._point_by_id) == ["ridge-1"]
    assert page.edits == []
    assert page.peak_table.rowCount() == 3  # G, P1, P2; no four-peak padding
    assert [item[0] for item in page.qspace._landmark_items()] == ["G", "P1"]

    # The landmark position is the exact q-array value at (row=3, col=15),
    # not an affine conversion of the detector pixel index.
    expected_q = (float(qx[3, 15]), float(qy[3, 15]))
    pixel_linear_guess = (15.0, 3.0)
    record = page._peak_landmarks["peaks"][0]
    assert (record["qx"], record["qy"]) == pytest.approx(expected_q)
    assert expected_q != pytest.approx(pixel_linear_guess)

    clicked: list[dict] = []
    page.qspace.landmarkSelected.connect(clicked.append)
    QtTest.QTest.mouseClick(
        page.qspace,
        QtCore.Qt.MouseButton.LeftButton,
        pos=page.qspace.q_to_widget(*expected_q),
    )
    assert clicked and clicked[-1]["peak_id"] == "P1"
    assert page._selected_landmark_id == "P1"
    assert page.qspace.selected_landmark_id == "P1"
    assert page.diagnostics_tabs.currentIndex() == 1
    assert "P1" in page.peak_angular_profile.title_label.text()
    assert page.peak_angular_profile.plot is not None
    assert len(page.peak_angular_profile.plot.listDataItems()) == 2
    assert page.qspace.q_window is not None
    focused_window = page.qspace.q_window
    assert focused_window != pytest.approx((0.0, 1.2))

    # Selecting the table row has the same focus behavior. The malformed P2
    # record retains its q value in the table but receives no invented qx/qy.
    page.peak_table.setCurrentCell(2, 0)
    assert page._selected_landmark_id == "P2"
    assert page.qspace.selected_landmark_id == "P2"
    assert page.qspace.q_window == pytest.approx(focused_window)
    assert all(item[0] != "P2" for item in page.qspace._landmark_items())
    page.close()


def test_exact_ridge_click_wins_over_a_nearby_landmark(qtbot):
    observed, qx, qy = _non_affine_frame()
    result = _landmark_result(qx, qy)
    ridge_point = result["points"][0]
    raw_max = result["peak_landmarks"]["raw_global_max"]
    raw_max["qx"] = float(ridge_point["qx"] + 0.025)
    raw_max["qy"] = float(ridge_point["qy"])
    raw_max["q"] = math.hypot(raw_max["qx"], raw_max["qy"])

    page = ButterflyWorkbench(language="en")
    qtbot.addWidget(page)
    page.resize(1180, 820)
    page.set_data(observed, qx=qx, qy=qy, q_unit="1/nm")
    page.set_q_window((0.0, 1.2))
    page.set_result(result)
    page.show()
    qtbot.wait(80)

    selected_points: list[dict] = []
    selected_landmarks: list[dict] = []
    page.qspace.pointSelected.connect(selected_points.append)
    page.qspace.landmarkSelected.connect(selected_landmarks.append)
    QtTest.QTest.mouseClick(
        page.qspace,
        QtCore.Qt.MouseButton.LeftButton,
        pos=page.qspace.q_to_widget(ridge_point["qx"], ridge_point["qy"]),
    )

    assert selected_points and selected_points[-1]["point_id"] == "ridge-1"
    assert selected_landmarks == []
    assert page.qspace.selected_point_id == "ridge-1"

    page.close()


def test_real_geometry_result_populates_only_core_supported_landmarks(qtbot):
    from butterfly_saxs.benchmark_arcs import generate_arc_case
    from butterfly_saxs.service import ButterflyAnalysisService

    case = generate_arc_case("ellipse_ratio_400", seed=231, shape=(48, 48))
    settings = {
        "ridge_method": "butterfly_curvature",
        "q_window": [0.05, 1.1],
        "ellipse_multistart": 2,
        "butterfly": {"stage": "evaluate", "resamples": 0, "sensitivity": False},
    }
    service = ButterflyAnalysisService(analysis_settings=settings)
    state = service.set_observed(case["image"], qmap=case["qmap"])
    result = service.measure_geometry(payload=state)["butterfly"]

    page = ButterflyWorkbench(language="en")
    qtbot.addWidget(page)
    page.set_data(
        case["image"],
        qx=case["qmap"].qx,
        qy=case["qmap"].qy,
        valid_mask=case["qmap"].valid_mask,
        q_unit=case["qmap"].q_unit,
    )
    page.set_result(result)
    page.resize(1180, 820)
    page.show()
    qtbot.wait(80)

    landmarks = result["peak_landmarks"]
    assert landmarks["status"] == "ok"
    assert page.peak_table.rowCount() == 1 + len(landmarks["peaks"])
    assert [
        page.peak_table.item(row, 0).text()
        for row in range(page.peak_table.rowCount())
    ] == ["G", *(peak["peak_id"] for peak in landmarks["peaks"])]
    assert len(page.qspace._landmark_items()) == 1 + len(landmarks["peaks"])
    assert len(page.qspace._point_by_id) == len(result["points"])
    assert not any(
        row["point_id"].startswith("P")
        for row in result["points"]
        if isinstance(row, dict)
    )
    angular_profile = landmarks["profiles"]["angular"]
    assert "intensity_isotropic_reference" in angular_profile
    assert "intensity_detection" in angular_profile
    bounds = page.qspace.base_q_bounds
    assert bounds is not None
    old_q_tolerance = max(bounds[1] - bounds[0], bounds[3] - bounds[2]) * 0.04
    visible_landmarks = [record for _, _, record in page.qspace._landmark_items()]
    nearby_ridge = next(
        (
            point
            for point in result["points"]
            if min(
                math.hypot(
                    float(point["qx"]) - float(landmark["qx"]),
                    float(point["qy"]) - float(landmark["qy"]),
                )
                for landmark in visible_landmarks
            )
            <= old_q_tolerance
        ),
        None,
    )
    assert nearby_ridge is not None
    selected_ridges: list[dict] = []
    selected_peaks: list[dict] = []
    page.qspace.pointSelected.connect(selected_ridges.append)
    page.qspace.landmarkSelected.connect(selected_peaks.append)
    QtTest.QTest.mouseClick(
        page.qspace,
        QtCore.Qt.MouseButton.LeftButton,
        pos=page.qspace.q_to_widget(nearby_ridge["qx"], nearby_ridge["qy"]),
    )
    assert selected_ridges and selected_ridges[-1]["point_id"] == nearby_ridge["point_id"]
    assert selected_peaks == []
    if landmarks["peaks"]:
        page.peak_table.setCurrentCell(1, 0)
        angular_table = page.peak_angular_profile.table
        assert angular_table.rowCount() == len(angular_profile["angle_deg"])
        assert [
            angular_table.horizontalHeaderItem(column).text()
            for column in range(angular_table.columnCount())
        ] == [
            "chi (deg)",
            "raw",
            "smoothed",
            "isotropic reference",
            "detection",
        ]


def test_fit_sources_and_poor_geometry_readout_are_separate_display_layers(qtbot):
    observed, qx, qy = _non_affine_frame()
    page = ButterflyWorkbench(language="en")
    qtbot.addWidget(page)
    page.set_data(observed, qx=qx, qy=qy, q_unit="1/nm")
    result = _poor_geometry_result()
    page.set_result(result)
    page.show()
    qtbot.wait(80)

    assert page.fit_assessment_label.isVisible()
    assert "Poor geometry fit" in page.fit_assessment_label.text()
    assert "10.2" in page.fit_assessment_label.text()
    assert "3" in page.fit_assessment_label.text()
    assert "Poor geometry fit" in page.status_label.text()
    # The interior b/a value does not override a poor residual/localization fit.
    assert result["candidate_fit"]["axis_ratio"] == pytest.approx(0.1)

    revision = page.result_revision
    page.set_manual_review(
        {
            "manual_status": "accepted",
            "reviewed_by": "reviewer",
            "result_revision": revision,
        }
    )
    analysis_events: list[dict] = []
    page.analysisChanged.connect(analysis_events.append)
    page.overlay_mode_combo.setCurrentIndex(
        page.overlay_mode_combo.findData("full2d_model")
    )
    assert page._fit_layers["intensity_model"]["curves"] == []
    assert "No current full2d Optimize geometry" in page.fit_source_label.text()
    page.set_model_fit_context(
        {"a": 0.45, "b": 0.20, "theta": math.radians(8.0)},
        reference_axis_deg=0.0,
        solver_status="failed",
        diagnostics={
            "rmse": 0.45,
            "condition_number": 5.0e11,
            "bound_flags": {"a": True},
        },
    )

    geometry = page._fit_layers["geometry"]
    model = page._fit_layers["intensity_model"]
    assert geometry["curves"]
    assert model["curves"]
    assert model["status"] == "failed"
    assert model["diagnostics"]["rmse"] == pytest.approx(0.45)

    for mode in (
        "measured_only",
        "observed_ridges",
        "geometry_candidate",
        "full2d_model",
        "compare",
    ):
        page.overlay_mode_combo.setCurrentIndex(page.overlay_mode_combo.findData(mode))
        assert page.qspace.overlay_mode == mode
    assert "Full2D intensity-model ellipse" in page.fit_source_label.text()
    assert "failed" in page.fit_source_label.text()
    assert "RMSE 0.45" in page.fit_source_label.text()
    assert "condition 5e+11" in page.fit_source_label.text()

    page.global_max_check.setChecked(False)
    page.supported_peaks_check.setChecked(False)
    assert analysis_events == []
    assert page.result_revision == revision
    assert page.manual_review["manual_status"] == "accepted"
    assert page.result_fresh

    page.set_language("zh_CN")
    assert "几何拟合较差" in page.fit_assessment_label.text()
    assert page.overlay_mode_combo.itemText(
        page.overlay_mode_combo.findData("geometry_candidate")
    ) == "几何候选 · 脊线拟合"
    assert page.peak_table.horizontalHeaderItem(2).text() == "原始 I"
    page.close()


def test_cancel_stale_and_new_frame_clear_diagnostic_layers(qtbot):
    observed, qx, qy = _non_affine_frame()
    page = ButterflyWorkbench(language="en")
    qtbot.addWidget(page)
    page.set_data(observed, qx=qx, qy=qy, q_unit="1/nm")
    result = _landmark_result(qx, qy)

    page.set_result(result)
    assert page.peak_table.rowCount() == 3
    page.set_job_status("cancelled", "refine_geometry")
    assert not page.result_fresh
    assert page.peak_table.rowCount() == 0
    assert page.qspace.peak_landmarks == {}
    assert page.qspace.fit_layers == {}

    page.set_result(result)
    page.set_model_fit_context(
        {"a": 0.45, "b": 0.2, "theta": 0.1},
        reference_axis_deg=0.0,
        solver_status="success",
    )
    page.invalidate_result()
    assert page.peak_table.rowCount() == 0
    assert page.qspace.fit_layers == {}

    page.set_result(result)
    page.set_analysis_settings({"q_min": 0.1, "q_max": 0.8})
    assert not page.result_fresh
    assert page.peak_table.rowCount() == 0

    page.set_result(result)
    mask = np.ones_like(observed, dtype=bool)
    mask[0, 0] = False
    page.set_data(observed + 1.0, qx=qx, qy=qy, valid_mask=mask, q_unit="1/nm")
    assert not page.result_fresh
    assert page.peak_table.rowCount() == 0
    assert page.qspace.peak_landmarks == {}
    page.close()


def test_full2d_overlay_context_comes_only_from_current_optimize_result(qtbot):
    observed, qx, qy = _non_affine_frame((12, 12))
    window = MainWindow(engine=object(), auto_preview=False, language="en")
    qtbot.addWidget(window)
    window.set_observed_data(observed, qx=qx, qy=qy)
    result = {
        "observed": observed,
        "model": observed.copy(),
        "residual": np.zeros_like(observed),
        "status": "failed",
        "parameters": {
            "a": 0.45,
            "b": 0.20,
            "theta": math.radians(8.0),
        },
        "pixel_model_condition_number": 5.0e11,
        "pixel_model_bound_flags": {"a": True},
        "pixel_model_rmse": 0.45,
        "butterfly": _poor_geometry_result(),
    }

    preview_generation = window._generation.next()
    window._on_worker_finished(preview_generation, "preview", result)
    page = window.butterfly_workbench
    assert page._fit_layers["intensity_model"]["curves"] == []

    optimize_generation = window._generation.next()
    window._on_worker_finished(optimize_generation, "optimize", result)
    assert page._fit_layers["intensity_model"]["curves"]
    assert page._fit_layers["intensity_model"]["status"] == "failed"
    assert page._fit_layers["intensity_model"]["diagnostics"]["condition_number"] == pytest.approx(
        5.0e11
    )
    assert window._last_model_diagnostic_signature == window._fit_state_signature()

    # Editing the fit state invalidates the current result and its diagnostic
    # curve before an old Optimize generation can paint over it.
    assert window.parameter_model.set_parameter("a", 0.48)
    assert page._fit_layers.get("intensity_model", {}).get("curves", []) == []
    assert page.peak_table.rowCount() == 0
    assert not page.result_fresh

    window.set_observed_data(observed + 2.0, qx=qx, qy=qy)
    assert page.qspace.fit_layers == {}
    assert page.qspace.peak_landmarks == {}
    window.close()


def test_real_service_optimize_metrics_reach_live_layer_and_export_context(
    qtbot, tmp_path, monkeypatch
):
    from butterfly_saxs import butterfly_figure
    from butterfly_saxs.benchmark_arcs import generate_arc_case
    from butterfly_saxs.service import ButterflyAnalysisService

    case = generate_arc_case("ellipse_ratio_400", seed=912, shape=(16, 16))
    analysis = {
        "q_min": 0.01,
        "q_max": 1.1,
        "max_nfev": 2,
        "full2d_multistart": 1,
        "max_pixels": 0,
        "ridge_method": "butterfly_curvature",
        "butterfly": {
            "stage": "evaluate",
            "resamples": 0,
            "sensitivity": False,
        },
    }
    service = ButterflyAnalysisService(analysis_settings=analysis)
    window = MainWindow(
        engine=service,
        parameters=service.parameters,
        analysis_settings=analysis,
        auto_preview=False,
        language="en",
    )
    qtbot.addWidget(window)
    window.set_observed_data(
        case["image"],
        qx=case["qmap"].qx,
        qy=case["qmap"].qy,
        qmap=case["qmap"],
    )
    result = service.optimize(
        parameters=window.parameter_model.parameter_dict(),
        payload=window._payload(),
    )
    metrics = result["metrics"]
    assert metrics["rmse"] is not None
    assert metrics["condition_number"] is not None
    assert metrics["bound_flags"]
    assert metrics["effective_bounds"]
    assert metrics["bound_flag_intensity_scale"] is not None

    generation = window._generation.next()
    window._on_worker_finished(generation, "optimize", result)
    page = window.butterfly_workbench
    page.overlay_mode_combo.setCurrentIndex(
        page.overlay_mode_combo.findData("full2d_model")
    )
    model = page._fit_layers["intensity_model"]
    assert model["curves"]
    assert model["status"] == "failed"  # metrics.success=False is solver status only
    assert model["diagnostics"]["rmse"] == pytest.approx(metrics["rmse"])
    assert model["diagnostics"]["condition_number"] == pytest.approx(
        metrics["condition_number"]
    )
    assert model["diagnostics"]["bound_flags"] == metrics["bound_flags"]
    assert model["diagnostics"]["effective_bounds"] == metrics["effective_bounds"]
    assert "failed" in page.fit_source_label.text()
    assert "RMSE" in page.fit_source_label.text()
    assert "condition" in page.fit_source_label.text()
    assert window._last_model_diagnostic_signature == window._fit_state_signature()

    captured: list[dict] = []

    def capture_export(target, **kwargs):
        captured.append(kwargs)
        target.mkdir(parents=True, exist_ok=False)
        manifest = target / "manifest.json"
        manifest.write_text("{}", encoding="utf-8")
        return {"manifest": manifest}

    monkeypatch.setattr(butterfly_figure, "export_butterfly_figure", capture_export)
    window._start_butterfly_figure_export(
        tmp_path / "service-model-export",
        page.figure_export_snapshot(),
        width_mm=183,
    )
    qtbot.waitUntil(lambda: not window._workers, timeout=5_000)

    context = captured[0]["context"]
    assert context["pixel_model_status"] == "failed"
    assert context["pixel_model_success"] is metrics["success"]
    assert context["pixel_model_rmse"] == pytest.approx(metrics["rmse"])
    assert context["pixel_model_condition_number"] == pytest.approx(
        metrics["condition_number"]
    )
    assert context["pixel_model_bound_flags"] == metrics["bound_flags"]
    assert context["pixel_model_effective_bounds"] == metrics["effective_bounds"]
    assert context["pixel_model_bound_flag_intensity_scale"] == pytest.approx(
        metrics["bound_flag_intensity_scale"]
    )
    assert context["pixel_model_reference_axis_deg"] == pytest.approx(
        window._reference_axis_deg()
    )
    assert context["pixel_model_parameters"]["a"] == pytest.approx(
        result["parameters"]["a"]["value"]
    )
    assert context["pixel_model_parameters"]["b"] == pytest.approx(
        result["parameters"]["b"]["value"]
    )
    window.close()


def test_constructor_evaluate_resamples_override_cached_budget_and_trace_preserves_it(
    qtbot,
):
    window = MainWindow(
        engine=object(),
        auto_preview=False,
        language="en",
        analysis_settings={
            "butterfly": {
                "stage": "evaluate",
                "resamples": 0,
                "sensitivity": False,
            }
        },
    )
    qtbot.addWidget(window)
    page = window.butterfly_workbench

    # MainWindow applies a project/config recipe over the page's default
    # cached evaluation budget of 32.
    assert page.butterfly_settings["stage"] == "evaluate"
    assert page.butterfly_settings["resamples"] == 0
    assert page._evaluation_resamples() == 0
    assert page.evaluation_resamples_combo.currentData() == 0

    # An explicitly persisted separate budget wins, while a later trace
    # request's zero resamples do not overwrite the user's chosen budget.
    page.set_analysis_settings(
        {"stage": "evaluate", "resamples": 0, "evaluation_resamples": 128}
    )
    assert page._evaluation_resamples() == 128
    page.set_analysis_settings({"stage": "trace", "resamples": 0})
    assert page.butterfly_settings["resamples"] == 0
    assert page._evaluation_resamples() == 128
    window.close()
