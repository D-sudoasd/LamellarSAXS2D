from __future__ import annotations

import json
import threading

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

from PySide6 import QtCore, QtWidgets

from butterfly_saxs.ui import MainWindow
from butterfly_saxs.ui.views import ViewGrid
from butterfly_saxs.visualization import plot_fit_diagnostics


class _Engine:
    def __init__(self) -> None:
        self.analysis: list[dict] = []

    def set_analysis_settings(self, settings):
        self.analysis.append(settings)


class _BatchEngine:
    def __init__(self) -> None:
        self.payloads: list[dict] = []

    def batch(self, *, parameters, payload):
        del parameters
        self.payloads.append(payload)
        callback = payload.get("progress")
        if callable(callback):
            callback({"completed": 1, "total": 1, "elapsed_s": 0.25})
        return {"records": []}


class _ConcurrentPreviewEngine:
    def __init__(self) -> None:
        self.started: list[threading.Event] = []
        self.release: list[threading.Event] = []
        self.cancel_seen: list[bool] = []
        self._lock = threading.Lock()

    def preview(self, *, parameters, payload):
        del parameters
        with self._lock:
            started = threading.Event()
            release = threading.Event()
            self.started.append(started)
            self.release.append(release)
        started.set()
        release.wait(3.0)
        cancel_event = payload.get("cancel_event") if isinstance(payload, dict) else None
        self.cancel_seen.append(bool(cancel_event is not None and cancel_event.is_set()))
        observed = np.asarray(payload["observed"], dtype=float)
        return {"observed": observed, "model": observed.copy(), "residual": np.zeros_like(observed)}


def _flat_settings() -> dict:
    return {
        "q_window": [0.1, 0.5],
        "ridge_method": "azimuthal_peak",
        "ridge_snr_threshold": 3.0,
        "ridge_min_peak_fraction": 0.3,
        "ridge_min_coverage": 0.2,
        "scales": [0.5, 1.0],
        "seed": 17,
        "robust_loss": "huber",
        "f_scale": 2.5,
        "max_nfev": 123,
        "n_ridge_angles": 180,
        "n_radial_bins": 48,
        "ellipse_residual": "geometric",
        "ellipse": {
            "preset": "flat_ellipse",
            "a": 0.8,
            "axis_ratio": 0.08,
            "a_min": 0.2,
            "a_max": 4.0,
            "axis_ratio_min": 0.005,
            "axis_ratio_max": 0.35,
            "fixed_center": True,
            "angle_deg": 40.0,
            "multistart": 7,
        },
    }


def test_flat_controls_round_trip_and_fit_tabs_are_compact(qtbot) -> None:
    engine = _Engine()
    window = MainWindow(
        engine=engine,
        analysis_settings=_flat_settings(),
        auto_preview=False,
        language="en",
    )
    qtbot.addWidget(window)
    window.resize(1280, 800)
    window.show()
    qtbot.wait(50)

    assert isinstance(window.parameters_dock.widget(), QtWidgets.QScrollArea)
    assert window.parameters_scroll_area.horizontalScrollBarPolicy() == (
        QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded
    )
    assert window.parameters_dock.width() <= 520
    assert [window.parameter_control_tabs.tabText(i) for i in range(4)] == [
        "Analysis",
        "Geometry",
        "Mask / ROI",
        "Review",
    ]
    assert window.q_min_edit.text() == "0.1"
    assert window.q_max_edit.text() == "0.5"
    assert window.ridge_method_combo.currentData() == "azimuthal_peak"
    assert window.ridge_min_coverage_spin.value() == pytest.approx(0.2)
    assert window.ellipse_residual_combo.currentData() == "geometric"
    assert window.ellipse_a_init_spin.value() == pytest.approx(0.8)
    assert window.ellipse_ratio_init_spin.value() == pytest.approx(0.08)
    assert window.ellipse_angle_deg_spin.value() == pytest.approx(40.0)
    assert window.ellipse_fixed_center_check.isChecked()
    assert window.ellipse_a_init_spin.toolTip()
    assert window.ellipse_ratio_init_spin.toolTip()
    assert window.ellipse_a_init_spin.accessibleName()
    window.parameter_control_tabs.setCurrentIndex(1)
    for widget in (
        window.ellipse_ratio_max_spin,
        window.ellipse_a_max_spin,
        window.ellipse_center_qy_spin,
        window.ellipse_residual_combo,
        window.refine_geometry_button,
    ):
        widget.setFocus()
        window.parameters_scroll_area.ensureWidgetVisible(widget)
        qtbot.wait(5)
        top_left = widget.mapTo(window.parameters_scroll_area.viewport(), widget.rect().topLeft())
        bottom_right = widget.mapTo(
            window.parameters_scroll_area.viewport(), widget.rect().bottomRight()
        )
        viewport = window.parameters_scroll_area.viewport().rect()
        assert viewport.left() <= top_left.x() <= viewport.right()
        assert viewport.left() <= bottom_right.x() <= viewport.right()
    window.resize(980, 680)
    window.show()
    qtbot.wait(30)
    window.parameter_control_tabs.setCurrentIndex(1)
    for widget in (
        window.ellipse_ratio_max_spin,
        window.ellipse_center_qy_spin,
        window.ellipse_residual_combo,
        window.refine_geometry_button,
    ):
        widget.setFocus()
        window.parameters_scroll_area.ensureWidgetVisible(widget)
        qtbot.wait(5)
        top_left = widget.mapTo(
            window.parameters_scroll_area.viewport(), widget.rect().topLeft()
        )
        bottom_right = widget.mapTo(
            window.parameters_scroll_area.viewport(), widget.rect().bottomRight()
        )
        viewport = window.parameters_scroll_area.viewport().rect()
        assert viewport.left() <= top_left.x() <= viewport.right()
        assert viewport.left() <= bottom_right.x() <= viewport.right()

    payload = window.analysis_settings
    assert payload["q_min"] == pytest.approx(0.1)
    assert payload["q_max"] == pytest.approx(0.5)
    assert payload["ridge_min_coverage"] == pytest.approx(0.2)
    assert payload["ellipse"]["a"] == pytest.approx(0.8)
    assert payload["ellipse"]["axis_ratio"] == pytest.approx(0.08)
    assert payload["ellipse"]["residual"] == "geometric"
    assert payload["ellipse"]["fixed_center"] is True
    assert payload["scales"] == [0.5, 1.0]
    assert payload["seed"] == 17
    assert payload["robust_loss"] == "huber"
    assert payload["f_scale"] == pytest.approx(2.5)
    assert payload["max_nfev"] == 123
    assert engine.analysis[-1]["ellipse"]["a"] == pytest.approx(0.8)
    window.close()


def test_geometry_result_does_not_claim_a_full2d_model_or_residual(qtbot) -> None:
    window = MainWindow(engine=_Engine(), auto_preview=False, language="zh_CN")
    qtbot.addWidget(window)
    observed = np.ones((8, 8), dtype=float)
    yy, xx = np.indices(observed.shape, dtype=float)
    window.set_observed_data(
        observed,
        qx=(xx - 3.5) / 16.0,
        qy=(yy - 3.5) / 16.0,
        qmap={
            "qx": (xx - 3.5) / 16.0,
            "qy": (yy - 3.5) / 16.0,
            "q_unit": "nm^-1",
        },
    )
    window._last_result_kind = "refine_geometry"
    result = {
        "geometry_action": "refine",
        "observed": observed,
        # A legacy geometry delegate may still return these shaped arrays;
        # they must be ignored by the UI.
        "model": np.full_like(observed, 9.0),
        "residual": np.full_like(observed, -8.0),
        "ellipse_fit": {
            "a": 0.5,
            "b": 0.04,
            "axis_ratio": 0.08,
            "theta_deg": 40.0,
            "q_unit": "nm^-1",
            "rmse": 0.004534,
            "n_points": 109,
            "quality_status": "WARN",
        },
        "metrics": {
            "rmse": 394.891,
            "ndata": observed.size,
            "valid_fraction": 0.41,
            "flags": ["geometry_refined"],
        },
    }
    window._last_result = result
    window._apply_result(result)
    window._set_busy(False, "refine_geometry", result_ok=True)
    assert window._geometry_only_result is True
    assert window.views.model.image_data is None
    assert window.views.residual.image_data is None
    assert "仅几何" in window.views.model.state_label.text()
    assert "整幅强度精修" in window.views.residual.state_label.text()
    assert window.rmse_label.text().endswith("0.004534")
    assert window.ndata_label.text().endswith("109")
    assert "geometry_only" in window.flags_label.text()
    assert "intensity_fit_not_run" in window.flags_label.text()
    assert "q_unit=nm^-1" in window.flags_label.text()
    assert "质量：WARN" in window.status_message.text() or "quality: WARN" in window.status_message.text()
    assert "尚未运行整幅强度精修" in window.workflow_status_label.text()
    assert "检查 Observed 中的 ridge 与椭圆" in window.workflow_status_label.text()
    assert window._current_result_is_reviewable() is False
    window.set_language("en", persist=False)
    assert "geometry only" in window.status_message.text()
    assert "whole-pixel intensity fit not run" in window.status_message.text()
    assert "Model not run" in window.views.model.state_label.text()
    assert "Review the observed ridge and ellipse" in window.workflow_status_label.text()
    assert "whole-pixel intensity fit has not run" in window.workflow_status_label.text()
    window.close()


def test_c3_symmetry_metadata_decorates_overlay_reference_pairs(qtbot) -> None:
    window = MainWindow(engine=_Engine(), auto_preview=False, language="zh_CN")
    qtbot.addWidget(window)
    observed = np.ones((12, 12), dtype=float)
    window.set_observed_data(observed)
    window._apply_result(
        {
            "observed": observed,
            "ridge_points": [
                {"qx": 0.2, "qy": 0.1, "branch_id": 0},
                {"qx": -0.2, "qy": -0.1, "branch_id": 1},
            ],
            "ellipse_fit": {
                "ellipses": [
                    {"a": 0.5, "b": 0.02, "angle_deg": 20.0},
                    {"a": 0.5, "b": 0.02, "angle_deg": -20.0},
                ],
                "symmetry": {
                    "paired_support": {
                        "0": {"quadrant_pair": "QI+QIII"},
                        "1": {"quadrant_pair": "QII+QIV"},
                    },
                    "branch_leaks": {"global_swap": False},
                },
            },
        }
    )
    assert window.views.overlay._ridge_records[0]["quadrant_pair"] == "QI+QIII"
    assert window.views.overlay._ridge_records[1]["quadrant_pair"] == "QII+QIV"
    assert window.views.overlay.ellipses[0]["quadrant_pair"] == "QI+QIII"
    assert window.views.overlay.ellipses[1]["quadrant_pair"] == "QII+QIV"
    window.close()


def test_current_worker_error_clears_previous_fit_views(qtbot) -> None:
    window = MainWindow(engine=_Engine(), auto_preview=False, language="en")
    qtbot.addWidget(window)
    observed = np.ones((6, 6), dtype=float)
    window.set_observed_data(observed)
    window._apply_result(
        {
            "observed": observed,
            "model": observed * 0.8,
            "residual": observed * 0.2,
            "ridge_points": [{"qx": 0.1, "qy": 0.2}],
            "ellipses": [{"a": 0.4, "b": 0.1}],
        }
    )
    generation = window._generation.next()
    window._on_worker_error(generation, "preview", RuntimeError("boom"))
    assert window.views.model.image_data is None
    assert window.views.residual.image_data is None
    assert window.views.overlay.ridge_points == []
    assert window.views.overlay.ellipses == []
    assert "boom" in window.status_message.text()
    window.close()


def test_busy_state_tracks_all_live_workers_and_cancel_event(qtbot) -> None:
    engine = _ConcurrentPreviewEngine()
    window = MainWindow(engine=engine, auto_preview=False, language="en")
    qtbot.addWidget(window)
    window.set_observed_data(np.ones((8, 8), dtype=float))
    window.request_preview()
    qtbot.waitUntil(lambda: len(engine.started) >= 1, timeout=2_000)
    window.request_preview()
    qtbot.waitUntil(lambda: len(engine.started) >= 2, timeout=2_000)

    # The current request completes while the stale request remains alive;
    # Cancel must remain available until both workers drain.
    engine.release[1].set()
    qtbot.waitUntil(
        lambda: window._last_result_kind == "preview" and len(window._workers) == 1,
        timeout=2_000,
    )
    assert window.cancel_button.isEnabled()
    assert window._cancel_events
    window.cancel_jobs()
    engine.release[0].set()
    qtbot.waitUntil(lambda: not window._workers, timeout=2_000)
    assert not window.cancel_button.isEnabled()
    assert True in engine.cancel_seen
    window.close()


def test_chinese_locale_retranslates_public_tabs_controls_and_geometry_hint(qtbot) -> None:
    window = MainWindow(engine=_Engine(), auto_preview=False, language="zh_CN")
    qtbot.addWidget(window)
    assert window.pages.tabText(0) == "精修"
    assert window.pages.tabText(1) == "测量 / 剖面"
    assert window.parameters_dock.windowTitle() == "参数"
    assert window.preview_button.text() == "预览"
    assert window.analysis_group.title() == "分析 / 测量"
    assert window.ellipse_group.title() == "观测椭圆约束"
    assert window.batch_stage_combo.itemText(0) == "几何测量"
    assert window.batch_stage_combo.itemText(1) == "Full2D 强度精修"
    assert window.views.observed.title_label.text() == "观测"
    assert "下限" in window.ellipse_a_min_spin.accessibleName()
    assert "快照" in window.snapshot_note_edit.accessibleName()
    assert window.cancel_button.shortcut().toString() == "Esc"
    window.set_language("en", persist=False)
    assert window.pages.tabText(0) == "Refinement"
    assert window.parameters_dock.windowTitle() == "Parameters"
    assert window.preview_button.text() == "Preview"
    assert window.analysis_group.title() == "Analysis / Measurement"
    assert window.batch_stage_combo.itemText(0) == "Geometry measurement"
    assert "min" in window.ellipse_a_min_spin.accessibleName()
    assert window.cancel_button.shortcut().toString() == "Esc"
    window.set_language("zh_CN", persist=False)
    assert window.parameters_dock.windowTitle() == "参数"
    assert window.preview_button.text() == "预览"
    window.close()


def test_batch_stage_forwards_geometry_or_full2d_explicitly(qtbot) -> None:
    engine = _BatchEngine()
    window = MainWindow(engine=engine, auto_preview=False, language="en")
    qtbot.addWidget(window)
    window.set_batch_frames(["frame-001.tif"])
    window.batch_output_edit.setText("results")
    window.batch_stream_check.setChecked(True)
    window.batch_stage_combo.setCurrentIndex(
        window.batch_stage_combo.findData("geometry")
    )
    window.run_batch()
    qtbot.waitUntil(lambda: bool(engine.payloads), timeout=2_000)
    assert engine.payloads[0]["stage"] == "geometry"
    assert engine.payloads[0]["full2d"] is False
    assert engine.payloads[0]["stream"] is True
    assert window._batch_progress_state["completed"] == 1
    window.cancel_jobs()
    window.batch_stage_combo.setCurrentIndex(
        window.batch_stage_combo.findData("full2d")
    )
    payload = window.project_to_dict()["batch"]
    assert payload["stage"] == "full2d"
    assert payload["full2d"] is True
    window.close()


def test_workflow_guide_routes_failed_results_to_failure_help(qtbot) -> None:
    from butterfly_saxs.ui.workbench import _refresh_workflow_guide

    window = MainWindow(engine=_Engine(), auto_preview=False, language="en")
    qtbot.addWidget(window)
    window.set_observed_data(np.ones((4, 4), dtype=float))
    window._last_result = {"status": "failed", "metrics": {"flags": ["boom"]}}
    window._last_result_kind = "preview"
    _refresh_workflow_guide(window)
    assert "latest result failed" in window.workflow_status_label.text()
    assert "Inspect the error" in window.workflow_status_label.text()
    window.close()


def test_workflow_guide_matches_batch_failure_signals(qtbot) -> None:
    from butterfly_saxs.ui.workbench import _refresh_workflow_guide

    window = MainWindow(engine=_Engine(), auto_preview=False, language="en")
    qtbot.addWidget(window)
    window.set_observed_data(np.ones((4, 4), dtype=float))
    for result in (
        {"success": False},
        {"full2d": {"quality_status": "FAIL"}},
        {"ellipse_fit": {"observed": None}},
    ):
        window._last_result = result
        window._last_result_kind = "preview"
        _refresh_workflow_guide(window)
        assert "latest result failed" in window.workflow_status_label.text()
    window.close()


def test_missing_project_input_rolls_back_small_ui_state(qtbot, tmp_path) -> None:
    window = MainWindow(engine=_Engine(), auto_preview=False, language="en")
    qtbot.addWidget(window)
    assert window.set_parameter("theta_deg", 10.0)
    project = tmp_path / "broken.json"
    project.write_text(
        '{"parameters": {"theta_deg": {"value": 20.0}}, "input": "missing.npy"}',
        encoding="utf-8",
    )
    assert window.load_project(project) is False
    assert window.parameters["theta_deg"] == pytest.approx(10.0)
    window.close()


def test_failed_project_load_restores_active_document_and_selectors(qtbot, tmp_path) -> None:
    image_path = tmp_path / "frames.npz"
    mask_path = tmp_path / "masks.npz"
    bad_path = tmp_path / "bad.npy"
    frames = np.stack([np.full((4, 5), index, dtype=np.float32) for index in range(9)])
    masks = np.zeros((9, 4, 5), dtype=np.uint8)
    masks[8, 0, 0] = 1
    np.savez(image_path, series=frames)
    np.savez(mask_path, masks=masks)
    bad_path.write_bytes(b"not a NumPy array")

    window = MainWindow(auto_preview=False, language="en")
    qtbot.addWidget(window)
    assert window.open_image(
        image_path,
        frame=7,
        dataset="series",
        external_mask=mask_path,
        mask_frame=8,
        mask_dataset="masks",
    )
    old_observed = window._observed
    old_qmap = window._qmap
    old_result = {
        "observed": old_observed,
        "model": np.asarray(old_observed, dtype=float) * 0.9,
        "residual": np.asarray(old_observed, dtype=float) * 0.1,
    }
    window._last_result = old_result
    window._last_result_kind = "preview"
    window._apply_result(old_result)
    window._last_result_signature = window._fit_state_signature()

    rejected = tmp_path / "rejected.json"
    rejected.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "input": str(bad_path),
                "frame": 1,
                "dataset": "new-dataset",
                "mask": str(mask_path),
                "mask_frame": 2,
                "mask_dataset": "new-mask",
                "parameters": {},
            }
        ),
        encoding="utf-8",
    )
    assert window.load_project(rejected) is False
    assert window._source_path == str(image_path)
    assert window._frame == 7
    assert window._dataset == "series"
    assert window._mask_path == str(mask_path)
    assert window._mask_frame == 8
    assert window._mask_dataset == "masks"
    assert window._observed is old_observed
    assert window._qmap is old_qmap
    assert window._last_result is old_result
    assert window._loaded_input_records["source"]["frame"] == 7
    assert window._loaded_input_records["mask"]["frame"] == 8

    new_image = tmp_path / "new.npy"
    np.save(new_image, np.full((4, 5), 11.0, dtype=np.float32))
    invalid_roi = tmp_path / "invalid-roi-existing-document.json"
    invalid_roi.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "input": str(new_image),
                "rois": [{"type": "rectangle", "x0": 4, "x1": 1, "y0": 0, "y1": 3}],
                "parameters": {},
            }
        ),
        encoding="utf-8",
    )
    assert window.load_project(invalid_roi) is False
    assert window._source_path == str(image_path)
    assert window._frame == 7
    assert window._dataset == "series"
    assert window._mask_path == str(mask_path)
    assert window._mask_frame == 8
    assert window._mask_dataset == "masks"
    assert window._observed is old_observed
    assert window._qmap is old_qmap
    assert window._last_result is old_result
    window.close()


def test_q_window_focus_uses_native_pixel_ranges_for_all_detector_views(qtbot) -> None:
    grid = ViewGrid()
    qtbot.addWidget(grid)
    observed = np.ones((64, 80), dtype=float)
    yy, xx = np.indices(observed.shape, dtype=float)
    qx = (xx - 40.0) / 80.0
    qy = (yy - 32.0) / 80.0
    grid.set_images(observed, observed * 2.0, observed * 0.5, qx=qx, qy=qy, q_unit="nm^-1")

    full_ranges = {
        name: view.plot.viewRange()
        for name, view in grid.views.items()
        if view.plot is not None
    }
    grid.set_q_view((0.1, 0.5))
    focused_ranges = {
        name: view.plot.viewRange()
        for name, view in grid.views.items()
        if view.plot is not None
    }
    for name in ("observed", "model", "residual"):
        assert focused_ranges[name][0][1] - focused_ranges[name][0][0] < (
            full_ranges[name][0][1] - full_ranges[name][0][0]
        )
        assert focused_ranges[name][1][1] - focused_ranges[name][1][0] < (
            full_ranges[name][1][1] - full_ranges[name][1][0]
        )
    assert focused_ranges["overlay"][0][1] - focused_ranges["overlay"][0][0] < (
        full_ranges["overlay"][0][1] - full_ranges["overlay"][0][0]
    )
    assert grid.overlay.image_extent is not None
    grid.set_q_view(full=True)
    restored = grid.views["observed"].plot.viewRange()
    assert restored[0][1] - restored[0][0] == pytest.approx(
        full_ranges["observed"][0][1] - full_ranges["observed"][0][0],
        rel=0.05,
    )
    grid.close()


def test_display_contrast_keeps_raw_counts_and_measurement_profiles_visible(qtbot) -> None:
    window = MainWindow(engine=_Engine(), auto_preview=False, language="en")
    qtbot.addWidget(window)
    observed = np.arange(64, dtype=float).reshape(8, 8) + 1.0
    window.set_observed_data(observed)
    generation = window._generation.current
    window.set_display_settings({"scale": "asinh", "percentile": 90.0})
    np.testing.assert_array_equal(window.views.observed.raw_image_data, observed)
    assert not np.array_equal(window.views.observed.image_data, observed)
    assert window._generation.current == generation

    result = {
        "observables": {
            "angular": {
                "angle_deg": [-90.0, 0.0, 90.0],
                "intensity": [1.0, 4.0, 1.0],
                "coverage": [0.8, 1.0, 0.9],
            },
            "ridge": {
                "q_unit": "nm^-1",
                "points": [{"angle_deg": 0.0, "q": 0.25, "accepted": True}],
            },
            "lobes": [{"angle_deg": 0.0, "intensity": 4.0, "valid": True}],
            "lobe_radial_profiles": [
                {
                    "angle": 0.0,
                    "q": np.asarray([0.1, 0.2, 0.3]),
                    "intensity": np.asarray([1.0, 4.0, 2.0]),
                    "q_unit": "nm^-1",
                }
            ],
            "lobe_radial_peaks": [
                {
                    "angle_deg": 0.0,
                    "q_star": 0.2,
                    "lamellar_spacing": 31.4,
                    "snr": 4.0,
                    "radial_fwhm": 0.03,
                    "coverage": 0.9,
                    "valid": True,
                    "flags": ["lobe_radial_peak"],
                }
            ],
            "flags": ["apparent_geometry_only"],
        },
        "ellipse_fit": {
            "a": 0.5,
            "b": 0.04,
            "axis_ratio": 0.08,
            "theta_deg": 40.0,
            "success": True,
            "stderr": {"a": 0.01},
            "bound_flags": {"a": False},
            "quality_status": "WARN",
            "flags": ["flat_ellipse_nonidentifiable"],
        },
    }
    window._update_measurements(result)
    assert window.profile_tabs.count() == 4
    assert window.radial_table.rowCount() == 1
    assert window.radial_table.item(0, 1).text() == "0.2"
    assert window.radial_table.item(0, 2).text() == "31.4"
    assert window.coverage_plot is not None
    assert window.coverage_plot.listDataItems()
    assert window.radial_profile_plot is not None
    assert window.radial_profile_plot.listDataItems()
    assert window.ellipse_table.rowCount() >= 19
    window.close()


def test_export_diagnostics_supports_display_only_contrast(tmp_path) -> None:
    observed = np.arange(36, dtype=float).reshape(6, 6) + 1.0
    model = observed * 0.8
    yy, xx = np.indices(observed.shape, dtype=float)
    output = tmp_path / "diagnostic-asinh.png"
    figure = plot_fit_diagnostics(
        observed,
        model,
        xx,
        yy,
        display_scale="asinh",
        display_percentile=92.0,
        output=output,
    )
    assert output.exists() and output.stat().st_size > 0
    assert figure.axes[0].images[0].get_array().max() == pytest.approx(
        np.arcsinh(observed).max()
    )
