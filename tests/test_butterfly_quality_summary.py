from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from butterfly_saxs.ui.butterfly_summary import build_butterfly_quality_summary
from butterfly_saxs.ui import MainWindow
from butterfly_saxs.ui.butterfly_workbench import ButterflyWorkbench


def _points(*, excluded: int = 0) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, (branch, side) in enumerate(
        ((0, "upper"), (0, "lower"), (1, "upper"), (1, "lower"))
    ):
        rows.append(
            {
                "point_id": f"p{index}",
                "qx": 0.1 + index * 0.01,
                "qy": 0.1,
                "branch_id": branch,
                "side": side,
                "accepted": index >= excluded,
                "valid": True,
            }
        )
    return rows


def _result(
    *,
    quality_status: str = "WARN",
    engineering_status: str = "WARN",
    flags: list[str] | None = None,
    q_unit: str = "nm^-1",
    stage: str = "evaluate",
) -> dict[str, object]:
    return {
        "q_unit": q_unit,
        "points": _points(),
        "settings": {"stage": stage},
        "quality": {
            "status": quality_status,
            "engineering_status": engineering_status,
            "flags": list(flags or []),
            "metrics": {
                "side_counts": {
                    "0:upper": 3,
                    "0:lower": 3,
                    "1:upper": 3,
                    "1:lower": 3,
                }
            },
        },
        "candidate_fit": {
            "axis_ratio": 0.35,
            "q_star_from_arcs": 0.092,
            "L_from_observed_radius_nm": 68.3,
            "q_unit": q_unit,
        },
        "quantitative_parameters": {
            "a": {"value": 0.2, "status": "available"},
        },
    }


def _page(qtbot, *, language: str = "zh_CN") -> ButterflyWorkbench:
    page = ButterflyWorkbench(language=language)
    qtbot.addWidget(page)
    return page


def test_summary_trace_and_evaluate_are_distinct(qtbot) -> None:
    page = _page(qtbot)
    page.set_analysis_settings({"stage": "trace", "resamples": 0})
    page.set_result(_result(stage="trace", quality_status="NOT_EVALUATED", engineering_status="NOT_EVALUATED"))
    assert page.quality_summary.state.status_key == "pending_evaluation"
    assert "评估" in page.quality_summary.status_label.text()

    page.set_analysis_settings({"stage": "evaluate", "resamples": 32})
    page.set_result(_result())
    assert page.quality_summary.state.status_key == "ellipse_candidate"
    assert page.quality_summary.state.side_support == "4/4"
    assert page.quality_summary.state.q_star == pytest.approx(0.092)
    assert page.quality_summary.state.l_ring == pytest.approx(68.3)
    page.close()


def test_parameter_estimate_status_is_visible_without_promoting_it(qtbot) -> None:
    page = _page(qtbot)
    page.set_analysis_settings({"stage": "evaluate", "resamples": 0})
    result = _result()
    result["quantitative_parameters"] = {
        "a": {
            "value": 0.2,
            "status": "estimate",
            "confidence": "empirical",
            "publication_status": "not_assessed",
            "reason": "interval is not calibrated",
        }
    }
    page.set_result(result)
    assert page.quality_summary.state.status_key == "ellipse_candidate"
    assert "parameter_candidates_require_review" in page.quality_summary.state.reasons
    assert "估计值或候选值" in page.quality_summary.reason_label.text()
    page.close()


def test_summary_ring_only_and_poor_match_reasons_are_visible(qtbot) -> None:
    page = _page(qtbot)
    page.set_analysis_settings({"stage": "evaluate", "resamples": 32})
    page.set_result(
        _result(flags=["analysis_choice_sensitivity_unassessed", "axis_ratio_at_bound"])
    )
    # The bound flag is a ring-only classification even when a candidate ratio exists.
    assert page.quality_summary.state.status_key == "ring_only"
    assert "axis_ratio_at_bound" in page.quality_summary.state.reasons
    assert "#fff5d6" in page.quality_summary.styleSheet()
    assert "观测弧迹/峰轨迹" in page.quality_summary.next_label.text()

    near_circle = _result(flags=["near_circular_ellipse_axis_unidentifiable"])
    near_circle["candidate_fit"]["axis_ratio"] = 0.98
    page.set_result(near_circle)
    assert page.quality_summary.state.status_key == "ring_only"
    assert "近圆形" in page.quality_summary.reason_label.text()

    page.set_result(_result(flags=["annular_outer_window_truncated"]))
    assert page.quality_summary.state.status_key == "ring_only"
    assert "窗口边界" in page.quality_summary.reason_label.text()

    page.set_result(
        _result(flags=["analysis_choice_sensitivity_unassessed", "insufficient_occupied_sides", "poor_match"])
    )
    assert page.quality_summary.state.status_key == "ellipse_candidate"
    assert page.quality_summary.state.reasons[0] == "poor_match"
    assert "insufficient_occupied_sides" in page.quality_summary.state.reasons
    assert "#fff0ee" in page.quality_summary.styleSheet()
    page.set_result(
        _result(
            flags=[
                "arc_endpoint_or_manual_bound_dependent",
                "disconnected_observed_support",
                "observed_support_infeasible",
            ]
        )
    )
    assert "依赖弧端点或手动边界" in page.quality_summary.reason_label.text()
    assert "观测弧支持不连续" in page.quality_summary.reason_label.text()
    assert "部分拟合投影超出观测支持" in page.quality_summary.reason_label.text()
    page.close()


def test_pixel_q_never_displays_physical_ring_period(qtbot) -> None:
    page = _page(qtbot)
    page.set_analysis_settings({"stage": "evaluate", "resamples": 0})
    page.set_result(_result(q_unit="pixel-q"))
    state = page.quality_summary.state
    assert state.q_star == pytest.approx(0.092)
    assert state.q_star_unit == "pixel-q"
    assert state.l_ring is None
    assert state.calibrated is False
    assert "uncalibrated_pixel_q" in state.reasons
    assert "未标定" in page.quality_summary.unit_label.text()
    page.close()


def test_explicit_nm_inverse_q_field_keeps_its_declared_unit() -> None:
    result = _result(q_unit="Å^-1")
    candidate = result["candidate_fit"]
    assert isinstance(candidate, dict)
    candidate.pop("q_star_from_arcs")
    candidate["q_star_nm_inv"] = 0.92
    state = build_butterfly_quality_summary(result, stage="evaluate")
    assert state.q_star == pytest.approx(0.92)
    assert state.q_star_unit == "nm^-1"


def test_cancel_invalidates_summary_and_language_refreshes_labels(qtbot) -> None:
    page = _page(qtbot, language="zh_CN")
    page.set_analysis_settings({"stage": "evaluate", "resamples": 32})
    page.set_result(_result())
    assert page.quality_summary.state.q_star is not None

    page.set_job_status("cancelled", "evaluate")
    assert page.quality_summary.state.q_star is None
    assert page.quality_summary.state.l_ring is None
    assert page.quality_summary.state.status_key == "empty"

    page.set_result(_result())
    page.set_language("en")
    assert "Ellipse" in page.quality_summary.status_label.text()
    assert "engineering" in page.quality_summary.engineering_label.text()
    assert "Scientific acceptance" in page.quality_summary.scientific_label.text()
    page.set_language("zh_CN")
    assert "科学接受" in page.quality_summary.scientific_label.text()
    page.close()


def test_old_failed_payload_does_not_survive_retry_running_state(qtbot) -> None:
    page = _page(qtbot)
    page.set_analysis_settings({"stage": "evaluate", "resamples": 32})
    page.set_result(_result(quality_status="FAIL", engineering_status="FAIL"))
    assert page.quality_summary.state.status_key == "failed"

    page.set_job_status("running", "evaluate")
    assert page.quality_summary.state.status_key == "running"
    assert page.quality_summary.state.q_star is None
    assert page.quality_summary.state.l_ring is None

    page.set_job_status("cancelled", "evaluate")
    assert page.quality_summary.state.status_key == "empty"
    page.close()


def test_analysis_error_and_cancel_clear_old_measurement_and_export_state(qtbot) -> None:
    page = _page(qtbot)
    page.set_analysis_settings({"stage": "evaluate", "resamples": 32})
    page.set_result(_result())
    assert page.result_fresh
    assert page.quality_summary.state.q_star is not None
    assert page.export_button.isEnabled()

    page.set_job_status("error", "measure_geometry", error="worker failed")
    assert not page.result_fresh
    assert page.quality_summary.state.q_star is None
    assert page.quality_summary.state.l_ring is None
    assert page.quantity_table.rowCount() == 0
    assert not page.export_button.isEnabled()
    assert page.quality_summary.state.status_key == "failed"

    page.set_result(_result())
    page.set_job_status("cancelled", "evaluate")
    assert not page.result_fresh
    assert page.quality_summary.state.q_star is None
    assert page.quality_summary.state.l_ring is None
    assert page.quantity_table.rowCount() == 0
    assert not page.export_button.isEnabled()
    page.close()


def test_completed_quality_failure_keeps_new_diagnostic_payload(qtbot) -> None:
    page = _page(qtbot)
    page.set_analysis_settings({"stage": "evaluate", "resamples": 32})
    failed = _result(quality_status="FAIL", engineering_status="FAIL", flags=["poor_match"])
    page.set_result(failed)
    q_star = page.quality_summary.state.q_star
    l_ring = page.quality_summary.state.l_ring

    page.set_job_status("completed", "evaluate", result_ok=False)
    assert page.result_fresh
    assert page.quality_summary.state.status_key == "failed"
    assert page.quality_summary.state.q_star == q_star
    assert page.quality_summary.state.l_ring == l_ring
    assert page.quantity_table.rowCount() > 0
    assert page.export_button.isEnabled()
    page.close()


def test_detached_figure_export_error_preserves_fresh_measurement(qtbot) -> None:
    page = _page(qtbot)
    page.set_analysis_settings({"stage": "evaluate", "resamples": 32})
    page.set_result(_result())
    q_star = page.quality_summary.state.q_star
    l_ring = page.quality_summary.state.l_ring

    # Cover the normal MainWindow sequence and a direct terminal error without
    # depending on the export worker or filesystem dialog.
    page.set_busy(True)
    page.set_job_status("running", "butterfly_figure_export")
    page.set_busy(False)
    page.set_job_status("error", "butterfly_figure_export", error="export failed")
    assert page.result_fresh
    assert page.quality_summary.state.q_star == q_star
    assert page.quality_summary.state.l_ring == l_ring
    assert page.quality_summary.state.status_key == "ellipse_candidate"
    assert page.export_button.isEnabled()

    page.set_result(_result())
    page.set_busy(True)
    page.set_job_status("running", "butterfly_figure_export")
    page.set_busy(False)
    page.set_job_status("cancelled", "cancelled")
    assert page.result_fresh
    assert page.quality_summary.state.q_star == q_star
    assert page.quality_summary.state.l_ring == l_ring

    failed = _result(quality_status="FAIL", engineering_status="FAIL", flags=["poor_match"])
    page.set_result(failed)
    failed_q_star = page.quality_summary.state.q_star
    page.set_busy(True)
    page.set_job_status("running", "butterfly_figure_export")
    page.set_busy(False)
    # MainWindow restores the pre-export diagnostic state after a detached
    # export error; a fresh FAIL payload must remain available for review.
    page.set_job_status("failed", "evaluate")
    assert page.result_fresh
    assert page.quality_summary.state.status_key == "failed"
    assert page.quality_summary.state.q_star == failed_q_star
    page.close()


def test_excluded_count_batch_feedback_and_cell_tooltips_follow_language(qtbot) -> None:
    page = _page(qtbot, language="zh_CN")
    result = _result()
    result["points"] = _points(excluded=1)
    result["quantitative_parameters"] = {
        "a": {
            "value": 0.2,
            "status": "undetermined",
            "reason": "a long diagnostic reason that should remain available in a tooltip",
        }
    }
    page.set_result(result)
    assert "1" in page.excluded_count_label.text()
    cell = page.quantity_table.item(0, 5)
    assert cell is not None
    assert cell.toolTip() == cell.text()

    page.set_batch_feedback(["frame-1"], ["frame-2 failed"])
    assert "批处理" in page.batch_feedback_label.text()
    page.set_language("en")
    assert "excluded" in page.excluded_count_label.text()
    assert "Batch completed" in page.batch_feedback_label.text()
    refreshed = page.quantity_table.item(0, 5)
    assert refreshed is not None
    assert refreshed.toolTip() == refreshed.text()
    page.close()


def test_summary_long_observables_keep_full_bbox_at_980_and_1600(qtbot, tmp_path) -> None:
    window = MainWindow(engine=object(), auto_preview=False, language="en")
    qtbot.addWidget(window)
    page = window.butterfly_workbench
    result = _result()
    candidate = result["candidate_fit"]
    assert isinstance(candidate, dict)
    candidate["q_star_from_arcs"] = 0.330123456
    candidate["L_from_observed_radius_nm"] = 18.9123456
    result["quality"]["flags"] = ["poor_match"]
    page.set_analysis_settings({"stage": "evaluate", "resamples": 32})
    page.set_result(result)
    window.show()
    qtbot.waitForWindowShown(window)
    for width, name in ((980, "980"), (1600, "1600")):
        window.resize(width, 680 if width == 980 else 1050)
        qtbot.waitUntil(lambda: window.width() == width, timeout=1_000)
        assert page.quality_summary.width() <= width
        for label in (
            page.quality_summary.status_label,
            page.quality_summary.engineering_label,
            page.quality_summary.support_label,
            page.quality_summary.q_star_label,
            page.quality_summary.l_ring_label,
            page.quality_summary.unit_label,
        ):
            required = label.fontMetrics().horizontalAdvance(label.text())
            assert label.geometry().width() >= required, (name, label.objectName(), label.text())
            assert label.geometry().height() >= label.fontMetrics().height()
        assert "#fff0ee" in page.quality_summary.styleSheet()
        assert page.quality_summary.state.engineering_status == "WARN"
        assert page.save_screenshot(tmp_path / f"quality-summary-{name}.png").exists()
    window.close()
