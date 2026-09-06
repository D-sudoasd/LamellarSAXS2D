from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("PySide6")

from butterfly_saxs.ui import MainWindow
from butterfly_saxs.ui.butterfly_export import export_butterfly_analysis
from butterfly_saxs.ui.qspace import QSpaceView


class _IdleEngine:
    """Engine seam sufficient for state-only UI checks."""

    def set_observed(self, *args, **kwargs):
        del args, kwargs
        return None


def _frame(shape: tuple[int, int] = (8, 8)) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    observed = np.arange(np.prod(shape), dtype=float).reshape(shape)
    yy, xx = np.indices(shape, dtype=float)
    return observed, xx / 10.0, yy / 10.0


def test_loaded_analysis_controls_match_recipe_and_preserve_other_fields_on_edit(qtbot):
    window = MainWindow(
        engine=_IdleEngine(),
        auto_preview=False,
        language="en",
        analysis_settings={
            "q_window": [0.1, 0.5],
            "draw_axis_deg": 95.0,
            "ridge_method": "butterfly_curvature",
            "butterfly": {"stage": "trace", "resamples": 0, "sensitivity": False},
        },
    )
    qtbot.addWidget(window)
    page = window.butterfly_workbench
    assert float(page.q_min_edit.text()) == pytest.approx(0.1)
    assert float(page.q_max_edit.text()) == pytest.approx(0.5)
    assert page.reference_axis_spin.value() == pytest.approx(5.0)

    # Editing one visible control must not replace the other configured values.
    page.reference_axis_spin.setValue(7.0)
    assert window.analysis_settings["q_min"] == pytest.approx(0.1)
    assert window.analysis_settings["q_max"] == pytest.approx(0.5)
    assert window.analysis_settings["draw_axis_deg"] == pytest.approx(97.0)
    assert page.butterfly_settings["sensitivity"] is False

    # Programmatic changes without a butterfly recipe still synchronize controls.
    window.set_analysis_settings({"q_window": [0.2, 0.6]}, trigger_preview=False)
    assert float(page.q_min_edit.text()) == pytest.approx(0.2)
    assert float(page.q_max_edit.text()) == pytest.approx(0.6)
    assert page.reference_axis_spin.value() == pytest.approx(7.0)
    page.q_min_edit.setText("0.25")
    page.q_min_edit.editingFinished.emit()
    assert window.analysis_settings["q_min"] == pytest.approx(0.25)
    assert window.analysis_settings["q_max"] == pytest.approx(0.6)
    assert window.analysis_settings["draw_axis_deg"] == pytest.approx(97.0)
    window.close()


def test_new_frame_clears_butterfly_overlay_and_selection(qtbot):
    window = MainWindow(engine=_IdleEngine(), auto_preview=False, language="en")
    qtbot.addWidget(window)
    observed, qx, qy = _frame()
    window.set_observed_data(observed, qx=qx, qy=qy, metadata={})
    page = window.butterfly_workbench
    page.set_result(
        {
            "points": [{"point_id": "old", "qx": 0.1, "qy": 0.1, "valid": True, "accepted": True}],
            "arcs": [],
        }
    )
    page.point_list.setCurrentRow(0)
    page.qspace.set_selected_point("old")
    assert page.result_fresh and page.qspace.selected_point_id == "old"

    new_observed, new_qx, new_qy = _frame((6, 6))
    window.set_observed_data(new_observed, qx=new_qx, qy=new_qy, metadata={})
    assert page.current_result == {}
    assert not page.result_fresh
    assert page.point_list.count() == 0
    assert page.qspace.selected_point_id is None
    assert page.qspace._butterfly == {}
    window.close()


def test_recipe_roundtrip_preserves_validated_fields_and_legacy_load_resets_edits(qtbot, tmp_path):
    window = MainWindow(engine=_IdleEngine(), auto_preview=False, language="en")
    qtbot.addWidget(window)
    observed, qx, qy = _frame()
    window.set_observed_data(observed, qx=qx, qy=qy, metadata={})
    page = window.butterfly_workbench
    page.set_analysis_settings(
        {
            "stage": "evaluate",
            "resamples": 4,
            "seed": 17,
            "sensitivity": False,
            "max_nfev": 321,
        }
    )
    recipe = page.butterfly_settings
    assert recipe["sensitivity"] is False
    assert recipe["max_nfev"] == 321
    page._on_edit_requested({"type": "seed", "qx": 0.1, "qy": 0.2})
    legacy = tmp_path / "legacy.json"
    legacy.write_text(
        json.dumps({"schema_version": 2, "analysis": {"ridge_method": "radial_peak"}}),
        encoding="utf-8",
    )
    assert window.load_project(legacy)
    assert page.edits == []
    assert page.butterfly_settings["sensitivity"] is True
    assert "max_nfev" not in page.butterfly_settings
    assert window.analysis_settings["ridge_method"] == "radial_peak"
    window.close()


def test_qspace_splits_physical_gaps_and_reuses_mesh_cache(qtbot):
    view = QSpaceView()
    qtbot.addWidget(view)
    observed, qx, qy = _frame((20, 20))
    view.set_data(observed, qx=qx, qy=qy)
    points = [
        {"point_id": "p0", "qx": 0.00, "qy": 0.00, "q_normal_step": 0.01, "branch_id": 0, "side": "upper"},
        {"point_id": "p1", "qx": 0.01, "qy": 0.00, "q_normal_step": 0.01, "branch_id": 0, "side": "upper"},
        {"point_id": "p2", "qx": 0.10, "qy": 0.00, "q_normal_step": 0.01, "branch_id": 0, "side": "upper"},
        {"point_id": "p3", "qx": 0.11, "qy": 0.00, "q_normal_step": 0.01, "branch_id": 0, "side": "upper"},
    ]
    arc = {"arc_id": 3, "ordered_point_ids": [item["point_id"] for item in points], "valid": True}
    view.set_butterfly({"points": points, "arcs": [arc]})
    segments = view._resolve_arc_segments(arc)
    assert [len(segment["points"]) for segment in segments] == [2, 2]
    invalid = dict(arc, arc_id=4, valid=False)
    assert not view._arc_visible(invalid)

    view.resize(500, 360)
    view.show()
    qtbot.waitUntil(lambda: view.q_bounds is not None, timeout=1_000)
    view.grab()
    original_corner = view._corner
    calls: list[tuple[int, int]] = []

    def counted_corner(row: int, col: int):
        calls.append((row, col))
        return original_corner(row, col)

    view._corner = counted_corner
    view._hover_q = (0.2, 0.2)
    view.update()
    view.grab()
    assert calls == []


def test_butterfly_export_keeps_diagnostic_failed_candidate_and_review_audit(qtbot, tmp_path):
    window = MainWindow(engine=_IdleEngine(), auto_preview=False, language="en")
    qtbot.addWidget(window)
    page = window.butterfly_workbench
    page.set_result({"status": "failed", "points": [], "arcs": []})
    page.set_manual_review(
        {
            "manual_status": "accepted",
            "reviewed_by": "reviewer",
            "reviewed_at": "2026-09-06T10:00:00+00:00",
            "review_notes": "diagnostic only",
        }
    )
    target = tmp_path / "failed-candidate"
    page.export_analysis(target)
    analysis = json.loads((target / "butterfly_analysis.json").read_text(encoding="utf-8"))
    assert analysis["manual_status"] == "accepted"
    assert analysis["scientific_acceptance"] is False
    assert analysis["reviewed_by"] == "reviewer"
    window.close()


def test_butterfly_busy_focuses_visible_cancel_and_readiness_is_gated(qtbot):
    window = MainWindow(engine=_IdleEngine(), auto_preview=False, language="en")
    qtbot.addWidget(window)
    window.show()
    qtbot.waitExposed(window)
    page = window.butterfly_workbench
    assert not page.identify_button.isEnabled()
    assert not page.evaluate_button.isEnabled()
    observed, qx, qy = _frame()
    window.set_observed_data(observed, qx=qx, qy=qy, metadata={})
    assert page.identify_button.isEnabled()
    window._set_busy(True, "measure_geometry")
    qtbot.wait(20)
    assert page.cancel_button.isVisible()
    assert window.focusWidget() is page.cancel_button
    assert "Running" in page.status_label.text()
    window._set_busy(False, "measure_geometry", result_ok=True)
    assert page.cancel_button.isEnabled() is False
    window.close()


def test_page_status_and_empty_canvas_survive_language_roundtrip(qtbot):
    window = MainWindow(engine=_IdleEngine(), auto_preview=False, language="zh_CN")
    qtbot.addWidget(window)
    page = window.butterfly_workbench
    page.set_job_status("empty")
    assert "载入" in page.status_label.text()
    page.set_language("en")
    assert "Load a frame" in page.status_label.text()
    page.set_job_status("running", "measure_geometry")
    page.set_language("zh_CN")
    assert "运行中" in page.status_label.text()
    page.set_job_status("error", "measure_geometry", error="solver failed")
    page.set_language("en")
    assert "Failed" in page.status_label.text() and "solver failed" in page.status_label.text()
    page.set_result({"status": "failed", "points": []})
    page.set_language("zh_CN")
    assert "失败" in page.status_label.text()

    view = QSpaceView()
    qtbot.addWidget(view)
    view.set_language("zh_CN")
    assert "载入" in view._empty_state_message()
    view.set_language("en")
    assert "Load a frame" in view._empty_state_message()
    window.close()


def test_butterfly_export_target_is_a_unique_child_bundle_directory(qtbot, tmp_path):
    window = MainWindow(engine=_IdleEngine(), auto_preview=False, language="en")
    qtbot.addWidget(window)
    parent = tmp_path / "exports"
    parent.mkdir()
    (parent / "butterfly-analysis").mkdir()
    target = window._butterfly_export_target(parent)
    assert target.parent == parent.resolve()
    assert target.name == "butterfly-analysis-2"
    window.close()


class _FalsePixmap:
    def save(self, path: str) -> bool:
        del path
        return False


class _FalseGrab:
    def grab(self):
        return _FalsePixmap()


def test_export_rolls_back_when_screenshot_save_returns_false(tmp_path):
    workbench = _FalseGrab()
    workbench.result_fresh = True
    workbench.current_result = {"points": []}
    workbench.butterfly_settings = {}
    workbench.export_context = {}
    workbench.display_settings = {}
    workbench._frame_data = {}
    workbench.qspace = workbench
    with pytest.raises(OSError, match="qspace"):
        export_butterfly_analysis(workbench, tmp_path / "rollback")
    assert not (tmp_path / "rollback").exists()
