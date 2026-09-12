from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6 import QtCore, QtWidgets

from butterfly_saxs.cancellation import AnalysisCancelled
from butterfly_saxs.ui import MainWindow


@pytest.mark.parametrize("kind, current_fit, status, has_model", [
    ("optimize", True, "ok", True), ("optimize", True, "failed", True),
    ("optimize", False, "ok", False), ("preview", True, "ok", False),
    ("measure_geometry", True, "ok", False),
])
def test_only_completed_pixel_fit_supplies_frozen_image_comparison(qtbot, tmp_path, monkeypatch, kind, current_fit, status, has_model):
    from butterfly_saxs import butterfly_figure

    window, _, observed, *_ = _window(qtbot)
    _set_page_result(window)
    model = observed * .9
    expected = model.copy()
    window._last_result = {"observed": observed, "model": model, "status": status}
    window._last_result_kind = kind
    window._last_model_diagnostic_signature = "current-signature" if current_fit else "old-signature"
    monkeypatch.setattr(window, "_fit_state_signature", lambda: "current-signature")
    captured = []

    def capture(target, **kwargs):
        captured.append(kwargs)
        target.mkdir()
        path = target / "manifest.json"
        path.write_text("{}", encoding="utf-8")
        return {"manifest": path}

    monkeypatch.setattr(butterfly_figure, "export_butterfly_figure", capture)
    snapshot = window.butterfly_workbench.figure_export_snapshot()
    window._start_butterfly_figure_export(tmp_path / "figure", snapshot, width_mm=183)
    model[:] = -123.0
    qtbot.waitUntil(lambda: not window._workers, timeout=5000)
    assert captured
    assert ("model" in captured[0]) is has_model
    if has_model:
        np.testing.assert_array_equal(captured[0]["model"], expected)
        assert not captured[0]["model"].flags.writeable
        assert captured[0]["context"]["pixel_model_status"] == status
    window.close()


def test_failed_finite_model_completion_retains_diagnostics_but_not_review_acceptance(qtbot, tmp_path, monkeypatch):
    from butterfly_saxs import butterfly_figure

    window, _, observed, qx, qy, mask = _window(qtbot)
    butterfly = _set_page_result(window)
    model = observed * .8
    result = {
        "observed": observed, "model": model, "model_image": model,
        "residual": observed - model, "qx": qx, "qy": qy, "valid_mask": mask,
        "status": "failed", "metrics": {"success": False}, "butterfly": butterfly,
    }
    generation = window._generation.next()
    window._on_worker_finished(generation, "optimize", result)
    assert window._last_result_signature is None
    assert not window._current_result_is_reviewable()
    assert window._last_model_diagnostic_signature == window._fit_state_signature()
    assert window.butterfly_workbench.result_fresh
    captured = []

    def capture(target, **kwargs):
        captured.append(kwargs)
        target.mkdir()
        manifest = target / "manifest.json"
        manifest.write_text("{}", encoding="utf-8")
        return {"manifest": manifest}

    monkeypatch.setattr(butterfly_figure, "export_butterfly_figure", capture)
    window._start_butterfly_figure_export(
        tmp_path / "diagnostic", window.butterfly_workbench.figure_export_snapshot(), width_mm=183,
    )
    qtbot.waitUntil(lambda: not window._workers, timeout=5000)
    assert captured[0]["context"]["pixel_model_status"] == "failed"
    np.testing.assert_array_equal(captured[0]["model"], model)
    assert not window._current_result_is_reviewable()
    window.close()


class _WorkflowEngine:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    @staticmethod
    def _result(payload):
        observed = np.asarray(payload["observed"], dtype=float)
        return {
            "observed": observed,
            "qx": payload.get("qx"),
            "qy": payload.get("qy"),
            "valid_mask": np.asarray(payload.get("valid_mask"), dtype=bool),
            "butterfly": {
                "points": [
                    {
                        "point_id": "ridge-0",
                        "qx": 0.1,
                        "qy": 0.2,
                        "pixel_x": 2.0,
                        "pixel_y": 3.0,
                        "accepted": True,
                        "valid": True,
                    }
                ],
                "profiles": {},
                "quantitative_parameters": {},
                "method_version": "workflow-test-v1",
            },
        }

    def measure_geometry(self, *, parameters, payload):
        del parameters
        self.calls.append(("measure_geometry", payload))
        return self._result(payload)

    def refine_geometry(self, *, parameters, payload):
        del parameters
        self.calls.append(("refine_geometry", payload))
        return self._result(payload)


def _window(qtbot, *, language="en"):
    engine = _WorkflowEngine()
    window = MainWindow(engine=engine, auto_preview=False, language=language)
    qtbot.addWidget(window)
    observed = np.arange(64, dtype=float).reshape(8, 8)
    yy, xx = np.indices(observed.shape, dtype=float)
    qx = (xx - 3.5) / 10.0
    qy = (yy - 3.5) / 10.0
    valid_mask = np.ones(observed.shape, dtype=bool)
    valid_mask[0, 0] = False
    window.set_observed_data(
        observed,
        qmap={"qx": qx, "qy": qy, "valid_mask": valid_mask, "q_unit": "1/nm"},
    )
    return window, engine, observed, qx, qy, valid_mask


def _set_page_result(window):
    result = {
        "points": [
            {
                "point_id": "ridge-0",
                "qx": 0.1,
                "qy": 0.2,
                "pixel_x": 2.0,
                "pixel_y": 3.0,
                "accepted": True,
                "valid": True,
            }
        ],
        "profiles": {},
        "quantitative_parameters": {},
        "diagnostic_array": np.arange(4, dtype=float),
        "method_version": "workflow-test-v1",
    }
    window.butterfly_workbench.set_result(result)
    return result


def test_invalid_q_range_is_visible_blocks_analysis_and_recovers_without_losing_display_semantics(qtbot):
    window, engine, *_ = _window(qtbot)
    page = window.butterfly_workbench
    assert "identify" in page.workflow_hint_label.text().lower()
    emitted = []
    page.analysisChanged.connect(emitted.append)
    original_analysis = dict(window._analysis_settings)

    page.q_min_edit.setText("NaN")
    page.q_max_edit.setText("Auto")
    page._on_analysis_range_changed()
    assert not page.q_range_error_label.isHidden()
    assert "finite" in page.q_range_error_label.text().lower()
    assert not page.identify_button.isEnabled()
    assert not page.evaluate_button.isEnabled()
    assert emitted == []
    assert window._analysis_settings == original_analysis
    page.request_identify()
    page.request_evaluate()
    assert engine.calls == []

    page.q_min_edit.setText("0.6")
    page.q_max_edit.setText("0.2")
    page._on_analysis_range_changed()
    assert not page.q_range_error_label.isHidden()
    assert "smaller" in page.q_range_error_label.text().lower()
    assert emitted == []
    assert window._analysis_settings == original_analysis

    page.q_min_edit.setText("0.1")
    page.q_max_edit.setText("0.6")
    page._on_analysis_range_changed()
    assert page.q_range_error_label.isHidden()
    assert page.identify_button.isEnabled()
    assert page.evaluate_button.isEnabled()
    assert len(emitted) == 1
    assert window._analysis_settings["q_min"] == pytest.approx(0.1)
    assert window._analysis_settings["q_max"] == pytest.approx(0.6)

    committed = dict(window._analysis_settings)
    page.display_scale_combo.setCurrentIndex(page.display_scale_combo.findData("log1p"))
    assert window._analysis_settings == committed
    window.close()


def test_evaluation_budget_default_quick_fit_and_extended_budget_survive_trace_and_project_roundtrip(
    qtbot, tmp_path
):
    window, engine, *_ = _window(qtbot)
    page = window.butterfly_workbench
    assert page.evaluation_resamples_combo.currentData() == 32
    assert page.sensitivity_check.isChecked()

    page.evaluation_resamples_combo.setCurrentIndex(
        page.evaluation_resamples_combo.findData(0)
    )
    page.evaluate_button.click()
    qtbot.waitUntil(lambda: any(kind == "refine_geometry" for kind, _ in engine.calls), timeout=2_000)
    quick = [payload for kind, payload in engine.calls if kind == "refine_geometry"][-1]
    assert quick["analysis"]["butterfly"]["resamples"] == 0
    assert quick["analysis"]["butterfly"]["evaluation_resamples"] == 0
    qtbot.waitUntil(lambda: not window._workers, timeout=2_000)
    assert "review" in page.workflow_hint_label.text().lower()

    page.evaluation_resamples_combo.setCurrentIndex(
        page.evaluation_resamples_combo.findData(128)
    )
    page.identify_button.click()
    qtbot.waitUntil(lambda: any(kind == "measure_geometry" for kind, _ in engine.calls), timeout=2_000)
    trace = [payload for kind, payload in engine.calls if kind == "measure_geometry"][-1]
    butterfly = trace["analysis"]["butterfly"]
    assert butterfly["stage"] == "trace"
    assert butterfly["resamples"] == 0
    assert butterfly["evaluation_resamples"] == 128
    assert page.butterfly_settings["evaluation_resamples"] == 128
    qtbot.waitUntil(lambda: not window._workers, timeout=2_000)
    assert "128" in page.workflow_hint_label.text()
    page.set_result(
        {
            "points": [],
            "quantitative_parameters": {"axis_ratio": {"status": "not_evaluated"}},
        }
    )
    assert "128" in page.quantity_table.item(0, 5).text()

    project = tmp_path / "evaluation-budget.json"
    assert window.save_project(project)
    saved = json.loads(project.read_text(encoding="utf-8"))
    assert saved["analysis"]["butterfly"]["evaluation_resamples"] == 128
    restored, _, *_ = _window(qtbot)
    assert restored.load_project(project)
    assert restored.butterfly_workbench.evaluation_resamples_combo.currentData() == 128
    assert restored.butterfly_workbench.sensitivity_check.isChecked()
    window.close()
    restored.close()


@pytest.mark.parametrize("choice_index, expected_width", [(0, 89.0), (1, 183.0)])
def test_figure_button_uses_selected_nature_width_600dpi_and_frozen_measurement_inputs(
    qtbot, tmp_path, monkeypatch, choice_index, expected_width
):
    import butterfly_saxs.butterfly_figure as figure_module

    window, _, observed, qx, qy, valid_mask = _window(qtbot)
    page = window.butterfly_workbench
    expected_result = _set_page_result(window)
    captured = []
    started = threading.Event()
    release = threading.Event()
    dialog_state = {}

    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getExistingDirectory",
        lambda *args, **kwargs: str(tmp_path),
    )

    def choose_item(parent, title, label, items, current, editable):
        dialog_state["items"] = list(items)
        dialog_state["default_index"] = current
        dialog_state["editable"] = editable
        return items[choice_index], True

    def choose_dpi(parent, title, label, value, minimum, maximum, step):
        dialog_state["dpi_default"] = value
        return value, True

    monkeypatch.setattr(QtWidgets.QInputDialog, "getItem", choose_item)
    monkeypatch.setattr(QtWidgets.QInputDialog, "getInt", choose_dpi)

    def fake_export(target, **kwargs):
        captured.append((target, kwargs))
        kwargs["progress"](47, "render")
        started.set()
        assert release.wait(5)
        target.mkdir(parents=True, exist_ok=False)
        written = {}
        for name in ("manifest", "svg", "pdf", "tiff", "png"):
            path = target / f"{name}.txt"
            path.write_text(name, encoding="utf-8")
            written[name] = path
        return written

    monkeypatch.setattr(figure_module, "export_butterfly_figure", fake_export)
    try:
        controls_scroll = page.findChild(QtWidgets.QScrollArea, "butterflyControlsScroll")
        assert (
            controls_scroll.horizontalScrollBarPolicy()
            == QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        assert page.figure_export_button.text() == "Export figure"
        assert page.figure_export_button.isEnabled()
        page.figure_export_button.click()
        qtbot.waitUntil(started.is_set, timeout=2_000)
        qtbot.waitUntil(lambda: "47%" in page.status_label.text(), timeout=2_000)
        assert "running" in page.status_label.text().lower()
        assert "s" in page.status_label.text().lower()
        assert dialog_state["items"] == ["Single column · 89 mm", "Double column · 183 mm"]
        assert dialog_state["default_index"] == 1
        assert dialog_state["dpi_default"] == 600
        target, kwargs = captured[0]
        assert kwargs["width_mm"] == expected_width
        assert kwargs["dpi"] == 600
        assert kwargs["q_unit"] == "1/nm"
        assert kwargs["display_scale"] == "linear"
        np.testing.assert_array_equal(kwargs["observed"], observed)
        np.testing.assert_array_equal(kwargs["qx"], qx)
        np.testing.assert_array_equal(kwargs["qy"], qy)
        np.testing.assert_array_equal(kwargs["valid_mask"], valid_mask)
        assert not kwargs["observed"].flags.writeable
        assert not kwargs["qx"].flags.writeable
        assert not kwargs["qy"].flags.writeable
        assert not kwargs["valid_mask"].flags.writeable
        assert not kwargs["result"]["diagnostic_array"].flags.writeable
        assert kwargs["result"]["points"] == expected_result["points"]
        assert Path(target) == tmp_path / "butterfly-figure"
        assert target.exists() is False
        release.set()
        qtbot.waitUntil(lambda: not window._workers, timeout=3_000)
        assert page.result_fresh
        assert page.figure_export_button.isEnabled()
        assert window._last_butterfly_figure_paths["manifest"].exists()
        assert window._status_key == "status.butterfly_figure_exported"
    finally:
        release.set()
        window.close()


def test_stale_result_is_blocked_but_fresh_failed_result_can_be_exported_without_upgrading_it(
    qtbot, tmp_path, monkeypatch
):
    import butterfly_saxs.butterfly_figure as figure_module

    window, *_ = _window(qtbot)
    page = window.butterfly_workbench
    _set_page_result(window)
    assert page.figure_export_button.isEnabled()
    page.invalidate_result()
    assert not page.figure_export_button.isEnabled()

    calls = []

    def fake_export(target, **kwargs):
        calls.append(kwargs)
        return {"manifest": Path(target) / "manifest.json"}

    monkeypatch.setattr(figure_module, "export_butterfly_figure", fake_export)
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getExistingDirectory",
        lambda *args, **kwargs: str(tmp_path),
    )
    monkeypatch.setattr(
        QtWidgets.QInputDialog,
        "getItem",
        lambda parent, title, label, items, current, editable: (items[0], True),
    )
    monkeypatch.setattr(
        QtWidgets.QInputDialog,
        "getInt",
        lambda parent, title, label, value, minimum, maximum, step: (value, True),
    )
    page.set_result({"status": "failed", "points": [], "quantitative_parameters": {}})
    assert page.result_fresh
    assert page.figure_export_button.isEnabled()
    assert "failed" in page.status_label.text().lower()
    page.figure_export_button.click()
    qtbot.waitUntil(lambda: bool(calls), timeout=2_000)
    qtbot.waitUntil(lambda: not window._workers, timeout=3_000)
    assert calls[0]["result"]["status"] == "failed"
    assert page.current_result["status"] == "failed"
    assert page.figure_export_button.isEnabled()
    window.close()


def test_q_range_validation_feedback_is_translated_to_chinese(qtbot):
    window, *_ = _window(qtbot, language="zh_CN")
    page = window.butterfly_workbench
    page.q_min_edit.setText("inf")
    page._on_analysis_range_changed()
    assert not page.q_range_error_label.isHidden()
    assert "有限" in page.q_range_error_label.text()
    assert not page.identify_button.isEnabled()
    window.close()


def test_long_failure_detail_stays_in_tooltip_across_language_roundtrip(qtbot):
    window, *_ = _window(qtbot)
    page = window.butterfly_workbench
    error = "solver failed: " + ("unexpected residual; " * 8)
    page.set_job_status("error", "measure_geometry", error=error)
    assert "Failed" in page.status_label.text()
    assert "solver failed" in page.status_label.text()
    assert error not in page.status_label.text()
    assert page.status_label.toolTip() == error
    page.set_language("zh_CN")
    assert "失败" in page.status_label.text()
    assert page.status_label.toolTip() == error
    window.close()


def test_figure_export_failure_preserves_current_measurement_and_reports_error(qtbot, tmp_path, monkeypatch):
    import butterfly_saxs.butterfly_figure as figure_module

    window, _, *_ = _window(qtbot)
    page = window.butterfly_workbench
    result = _set_page_result(window)

    def fail_export(*args, **kwargs):
        raise RuntimeError("render failed")

    monkeypatch.setattr(figure_module, "export_butterfly_figure", fail_export)
    snapshot = page.figure_export_snapshot(context={"sample": "test"})
    window._start_butterfly_figure_export(
        tmp_path / "failed-figure",
        snapshot,
        width_mm=89,
        dpi=600,
    )
    qtbot.waitUntil(lambda: not window._workers, timeout=3_000)
    assert page.result_fresh
    assert page.current_result["points"] == result["points"]
    assert page.figure_export_button.isEnabled()
    assert "failed" in page.status_label.text().lower()
    assert window._status_key == "status.butterfly_figure_export_failed"
    window.close()


def test_cancel_shows_draining_state_until_worker_exits(qtbot, tmp_path, monkeypatch):
    import butterfly_saxs.butterfly_figure as figure_module

    window, _, *_ = _window(qtbot)
    page = window.butterfly_workbench
    _set_page_result(window)
    started = threading.Event()
    release = threading.Event()

    def wait_then_cancel(target, **kwargs):
        kwargs["progress"](20, "render")
        started.set()
        assert release.wait(5)
        raise AnalysisCancelled("test export cancelled")

    monkeypatch.setattr(figure_module, "export_butterfly_figure", wait_then_cancel)
    window._start_butterfly_figure_export(
        tmp_path / "cancelled-figure",
        page.figure_export_snapshot(),
        width_mm=183,
        dpi=600,
    )
    qtbot.waitUntil(started.is_set, timeout=2_000)
    page.cancel_button.click()
    assert window._workers
    assert page.cancel_button.isEnabled()
    assert "cancelling" in page.status_label.text().lower()
    assert "cancelled" not in page.status_label.text().lower()

    release.set()
    qtbot.waitUntil(lambda: not window._workers, timeout=3_000)
    assert not page.cancel_button.isEnabled()
    assert "cancelled" in page.status_label.text().lower()
    assert page.result_fresh
    assert page.figure_export_button.isEnabled()
    window.close()


def test_frame_change_during_export_does_not_attach_old_result_to_new_frame(qtbot, tmp_path, monkeypatch):
    import butterfly_saxs.butterfly_figure as figure_module

    window, _, _, _, _, _ = _window(qtbot)
    page = window.butterfly_workbench
    _set_page_result(window)
    started = threading.Event()
    release = threading.Event()

    def delayed_export(target, **kwargs):
        started.set()
        assert release.wait(5)
        target.mkdir(parents=True, exist_ok=False)
        manifest = target / "manifest.json"
        manifest.write_text("{}", encoding="utf-8")
        return {"manifest": manifest}

    monkeypatch.setattr(figure_module, "export_butterfly_figure", delayed_export)
    window._start_butterfly_figure_export(
        tmp_path / "old-frame-figure",
        page.figure_export_snapshot(context={"frame": "old"}),
        width_mm=89,
        dpi=600,
    )
    qtbot.waitUntil(started.is_set, timeout=2_000)
    replacement = np.full((8, 8), 5.0)
    yy, xx = np.indices(replacement.shape, dtype=float)
    window.set_observed_data(
        replacement,
        qmap={
            "qx": (xx - 3.5) / 5.0,
            "qy": (yy - 3.5) / 5.0,
            "q_unit": "1/nm",
        },
    )
    assert not page.result_fresh
    release.set()
    qtbot.waitUntil(lambda: not window._workers, timeout=3_000)
    assert page.current_result == {}
    assert not page.result_fresh
    assert page.figure_export_button.isEnabled() is False
    assert window._last_butterfly_figure_paths["manifest"].parent == tmp_path / "old-frame-figure"
    window.close()
