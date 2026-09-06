from __future__ import annotations

import csv
import json
import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6 import QtCore, QtTest

from butterfly_saxs.ui import MainWindow
from butterfly_saxs.ui.qspace import QSpaceView


class _ButterflyEngine:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def measure_geometry(self, *, parameters, payload):
        del parameters
        self.calls.append(("measure_geometry", payload))
        observed = np.asarray(payload["observed"], dtype=float)
        return {
            "observed": observed,
            "qx": payload.get("qx"),
            "qy": payload.get("qy"),
            "valid_mask": np.ones_like(observed, dtype=bool),
            "butterfly": {
                "points": [
                    {
                        "point_id": "p0",
                        "qx": 0.0,
                        "qy": 0.0,
                        "pixel_x": 3.0,
                        "pixel_y": 3.0,
                        "branch_id": 0,
                        "arc_id": 0,
                        "side": "upper",
                        "accepted": True,
                        "valid": True,
                        "reason": "supported",
                    }
                ],
                "arcs": [],
                "profiles": {"p0": {"offset_q": [-0.1, 0.0, 0.1], "raw": [1, 2, 1], "fit": [1, 1.8, 1]}},
                "candidate_fit": {"ellipses": []},
                "quantitative_parameters": {
                    "a": {"value": 0.2, "status": "ok", "reason": "supported", "interval": [0.1, 0.3]}
                },
                "diagnostics": {"display_magnification": 8},
                "method_version": "test-v1",
                "edits": payload["analysis"]["butterfly"]["edits"],
            },
        }

    def refine_geometry(self, *, parameters, payload):
        self.calls.append(("refine_geometry", payload))
        result = self.measure_geometry(parameters=parameters, payload=payload)
        return result


def _set_frame(window):
    observed = np.arange(64, dtype=float).reshape(8, 8)
    yy, xx = np.indices(observed.shape, dtype=float)
    qx = (xx - 3.5) / 10.0 + 0.015 * np.sin(yy / 2.0)
    qy = (yy - 3.5) / 10.0 + 0.01 * np.cos(xx / 2.0)
    window.set_observed_data(observed, qx=qx, qy=qy)


def test_butterfly_page_actions_pass_trace_and_evaluate_recipes(qtbot):
    engine = _ButterflyEngine()
    window = MainWindow(engine=engine, auto_preview=False, language="en")
    qtbot.addWidget(window)
    _set_frame(window)

    assert window.pages.currentIndex() == 0
    assert "Butterfly" in window.pages.tabText(0)
    window.butterfly_workbench.identify_button.click()
    qtbot.waitUntil(lambda: any(kind == "measure_geometry" for kind, _ in engine.calls), timeout=2_000)
    trace = [payload for kind, payload in engine.calls if kind == "measure_geometry"][-1]
    assert trace["analysis"]["ridge_method"] == "butterfly_curvature"
    assert trace["analysis"]["butterfly"]["stage"] == "trace"
    assert trace["analysis"]["butterfly"]["resamples"] == 0
    assert trace["analysis"]["butterfly"]["seed"] == 20260906

    window.butterfly_workbench.evaluate_button.click()
    qtbot.waitUntil(lambda: any(kind == "refine_geometry" for kind, _ in engine.calls), timeout=2_000)
    evaluate = [payload for kind, payload in engine.calls if kind == "refine_geometry"][-1]
    assert evaluate["analysis"]["butterfly"]["stage"] == "evaluate"
    assert evaluate["analysis"]["butterfly"]["resamples"] == 32
    window.close()


def test_butterfly_edit_undo_project_roundtrip_and_cancel(qtbot, tmp_path):
    window = MainWindow(engine=_ButterflyEngine(), auto_preview=False, language="en")
    qtbot.addWidget(window)
    _set_frame(window)
    page = window.butterfly_workbench
    window.show()
    qtbot.waitForWindowShown(window)

    page.correction_mode_combo.setCurrentIndex(page.correction_mode_combo.findData("seed"))
    page.correct_button.click()
    QtTest.QTest.mouseClick(
        page.qspace,
        QtCore.Qt.MouseButton.LeftButton,
        pos=page.qspace.q_to_widget(0.0, 0.0),
    )
    qtbot.waitUntil(lambda: len(page.edits) == 1, timeout=1_000)
    assert page.edits[0]["type"] == "seed"
    generation_before_cancel = window._generation.current
    page.set_busy(True)
    page.cancel_button.click()
    assert window._generation.current > generation_before_cancel
    page.request_identify()
    qtbot.waitUntil(lambda: window.analysis_settings["ridge_method"] == "butterfly_curvature", timeout=1_000)

    project_path = tmp_path / "butterfly-edits.json"
    assert window.save_project(project_path)
    saved = json.loads(project_path.read_text(encoding="utf-8"))
    assert saved["analysis"]["ridge_method"] == "butterfly_curvature"
    assert saved["analysis"]["butterfly"]["edits"] == page.edits

    page.undo_button.click()
    assert page.edits == []
    assert window.load_project(project_path)
    assert page.edits == saved["analysis"]["butterfly"]["edits"]
    assert not page.legacy_banner.isVisible()
    window.close()


def test_old_project_keeps_legacy_method_and_offscreen_pages_are_capturable(qtbot, tmp_path):
    project_path = tmp_path / "legacy.json"
    project_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "parameters": {},
                "analysis": {"ridge_method": "radial_peak"},
            }
        ),
        encoding="utf-8",
    )
    window = MainWindow(engine=_ButterflyEngine(), auto_preview=False, language="en")
    qtbot.addWidget(window)
    assert window.load_project(project_path)
    assert window.analysis_settings["ridge_method"] == "radial_peak"
    window.show()
    qtbot.waitForWindowShown(window)
    assert window.butterfly_workbench.legacy_banner.isVisible()
    window.resize(980, 680)
    window.show()
    qtbot.waitForWindowShown(window)
    first = tmp_path / "butterfly-980x680.png"
    second = tmp_path / "butterfly-1280x800.png"
    window.butterfly_workbench.save_screenshot(first)
    window.resize(1280, 800)
    window.butterfly_workbench.save_screenshot(second)
    assert first.stat().st_size > 0 and second.stat().st_size > 0
    window.close()


def test_main_window_q_window_method_authority_and_keyboard_point_exclusion(qtbot):
    window = MainWindow(engine=_ButterflyEngine(), auto_preview=False, language="zh_CN")
    qtbot.addWidget(window)
    _set_frame(window)
    window.set_analysis_settings({"q_min": 0.2, "q_max": 0.4}, trigger_preview=False)
    assert window.butterfly_workbench.qspace.q_window == pytest.approx((0.2, 0.4))
    window.reset_q_view()
    assert window.butterfly_workbench.qspace.q_window is None

    window.ridge_method_combo.setCurrentIndex(window.ridge_method_combo.findData("radial_peak"))
    window._on_analysis_changed()
    assert window.analysis_settings["ridge_method"] == "radial_peak"

    point = {"point_id": "keyboard-point", "qx": 0.0, "qy": 0.0, "accepted": True, "valid": True}
    window.butterfly_workbench.set_result({"points": [point], "profiles": {}})
    window.butterfly_workbench.point_list.setCurrentRow(0)
    QtTest.QTest.keyClick(
        window.butterfly_workbench.point_list,
        QtCore.Qt.Key.Key_Delete,
    )
    qtbot.waitUntil(lambda: window.butterfly_workbench.edits == [{"type": "exclude_point", "point_id": "keyboard-point"}], timeout=1_000)
    window.close()


def test_backend_ellipse_local_bundle_and_redo_reset(qtbot):
    window = MainWindow(engine=_ButterflyEngine(), auto_preview=False, language="zh_CN")
    qtbot.addWidget(window)
    _set_frame(window)
    point = {"point_id": "p0", "qx": 0.0, "qy": 0.0, "branch_id": 0, "side": "upper", "accepted": True, "valid": True}
    result = {
        "points": [point],
        "profiles": {"p0": {"offset_q": [-1, 0, 1], "raw_intensity": [1, 2, 1], "fit_intensity": [1, 1.8, 1], "residual": [0, .2, 0]}},
        "ellipse_local": {
            "points": [{"point_id": "p0", "u": 0.1, "v": 0.02, "side": "upper", "accepted": True}],
            "curves": {"upper": {"u": [-.2, 0, .2], "v": [.03, .05, .03]}},
            "q_unit": "1/nm",
            "source": "candidate_geometry",
        },
        "quantitative_parameters": {},
    }
    window.butterfly_workbench.set_result(result)
    window.butterfly_workbench.point_list.setCurrentRow(0)
    assert window.butterfly_workbench.normal_profile.plot.listDataItems()
    assert window.butterfly_workbench.ellipse_diagnostic.plot.listDataItems()

    page = window.butterfly_workbench
    page._on_edit_requested({"type": "seed", "qx": 0.1, "qy": 0.2})
    page.undo_edit()
    assert page.redo_button.isEnabled()
    page.set_analysis_settings({"butterfly": {"stage": "trace", "edits": []}})
    assert not page.redo_button.isEnabled()
    window.close()


def test_evaluated_undetermined_keeps_reason_and_exposes_unvalidated_candidate(qtbot):
    window = MainWindow(engine=_ButterflyEngine(), auto_preview=False, language="zh_CN")
    qtbot.addWidget(window)
    window.butterfly_workbench.set_analysis_settings({"stage": "evaluate", "resamples": 32})
    window.butterfly_workbench.set_result(
        {
            "points": [],
            "profiles": {},
            "quantitative_parameters": {
                "a": {
                    "value": None,
                    "candidate_value": 0.42,
                    "status": "undetermined",
                    "reason": "candidate geometry is ill-conditioned after evaluation",
                    "interval": None,
                }
            },
        }
    )
    assert window.butterfly_workbench.quantity_table.item(0, 2).text() == "未确定"
    assert window.butterfly_workbench.quantity_table.item(0, 3).text() == "0.42"
    assert "ill-conditioned" in window.butterfly_workbench.quantity_table.item(0, 5).text()
    assert "追踪阶段" not in window.butterfly_workbench.quantity_table.item(0, 5).text()
    window.close()


def test_actual_service_arc_ids_resolve_to_contiguous_visible_segments(qtbot):
    from butterfly_saxs.benchmark_arcs import generate_arc_case
    from butterfly_saxs.service import ButterflyAnalysisService

    case = generate_arc_case("ellipse_ratio_400", seed=123, shape=(48, 48))
    settings = {
        "ridge_method": "butterfly_curvature",
        "q_window": [0.05, 1.1],
        "ellipse_multistart": 2,
        "butterfly": {"stage": "evaluate", "resamples": 0, "sensitivity": False},
    }
    service = ButterflyAnalysisService(analysis_settings=settings)
    state = service.set_observed(case["image"], qmap=case["qmap"])
    result = service.measure_geometry(payload=state)
    payload = result["butterfly"]
    view = QSpaceView()
    qtbot.addWidget(view)
    view.set_data(case["image"], qx=case["qmap"].qx, qy=case["qmap"].qy)
    view.set_butterfly(payload)
    points = {row["point_id"]: row for row in payload["points"]}
    arc = next(arc for arc in payload["arcs"] if arc.get("ordered_point_ids"))
    segments = view._resolve_arc_segments(arc)
    assert segments
    for segment in segments:
        assert len(segment["points"]) >= 2
        for point_id, q in zip(segment["point_ids"], segment["points"]):
            assert tuple(q) == pytest.approx((points[point_id]["qx"], points[point_id]["qy"]))
    first = segments[0]
    view.set_visible_branch(first["branch_id"], first["side"], False)
    assert not view._visible_branches[(first["branch_id"], first["side"])]


def test_programmatic_butterfly_export_is_authoritative_non_overwriting_and_stale_safe(qtbot, tmp_path):
    from butterfly_saxs.benchmark_arcs import generate_arc_case
    from butterfly_saxs.service import ButterflyAnalysisService

    case = generate_arc_case("ellipse_ratio_400", seed=124, shape=(40, 40))
    settings = {
        "ridge_method": "butterfly_curvature",
        "q_window": [0.05, 1.1],
        "butterfly": {"stage": "evaluate", "resamples": 0, "sensitivity": False},
    }
    service = ButterflyAnalysisService(analysis_settings=settings)
    state = service.set_observed(case["image"], qmap=case["qmap"])
    result = service.measure_geometry(payload=state)

    window = MainWindow(engine=service, auto_preview=False, language="en")
    qtbot.addWidget(window)
    window.butterfly_workbench.set_result(result["butterfly"])
    window.butterfly_workbench.set_export_context(
        {"source": "synthetic-service-frame", "frame": 0, "dataset": None, "q_unit": "1/nm"}
    )
    target = tmp_path / "butterfly-export"
    window.butterfly_workbench.export_analysis(target)
    expected = {
        "butterfly_analysis.json",
        "parameters.csv",
        "ridge_points.csv",
        "provenance.json",
        "manifest.json",
        "qspace.png",
        "normal_profile.png",
        "ellipse_local.png",
    }
    assert expected <= {path.name for path in target.iterdir()}
    analysis = json.loads((target / "butterfly_analysis.json").read_text(encoding="utf-8"))
    assert analysis["manual_status"] == "unreviewed"
    assert analysis["scientific_acceptance"] is False
    assert analysis["butterfly"]["points"]
    csv_rows = list(csv.DictReader((target / "parameters.csv").open(encoding="utf-8", newline="")))
    authoritative = analysis["butterfly"]["quantitative_parameters"]
    for row in csv_rows:
        source = authoritative[row["parameter"]]
        assert row["candidate_value"] == ("" if source.get("candidate_value") is None else str(source.get("candidate_value")))
    with pytest.raises(FileExistsError):
        window.butterfly_workbench.export_analysis(target)
    window.butterfly_workbench._on_edit_requested({"type": "seed", "qx": 0.1, "qy": 0.2})
    with pytest.raises(ValueError, match="stale"):
        window.butterfly_workbench.export_analysis(tmp_path / "stale-export")
    window.close()


def test_shared_input_mutations_stale_geometry_export_but_display_only_does_not(qtbot, tmp_path):
    from butterfly_saxs.benchmark_arcs import generate_arc_case
    from butterfly_saxs.service import ButterflyAnalysisService

    case = generate_arc_case("ellipse_ratio_400", seed=125, shape=(32, 32))
    settings = {
        "ridge_method": "butterfly_curvature",
        "q_window": [0.05, 1.1],
        "butterfly": {"stage": "evaluate", "resamples": 0, "sensitivity": False},
    }
    service = ButterflyAnalysisService(analysis_settings=settings)
    state = service.set_observed(case["image"], qmap=case["qmap"])
    result = service.measure_geometry(payload=state)
    window = MainWindow(engine=service, auto_preview=False, language="en")
    qtbot.addWidget(window)
    window.set_observed_data(case["image"], qmap=case["qmap"])

    def fresh() -> None:
        window.butterfly_workbench.set_result(result["butterfly"])
        window.butterfly_workbench.set_export_context({"source": "real-service-case"})

    def assert_stale(label: str) -> None:
        with pytest.raises(ValueError, match="stale"):
            window.butterfly_workbench.export_analysis(tmp_path / f"stale-{label}")

    fresh()
    window.set_parameter("theta_deg", 3.0)
    assert_stale("parameter")
    fresh()
    service.set_poni = lambda _path: case["qmap"]
    assert window.set_poni("cached-poni")
    assert_stale("poni")
    fresh()
    window._file_mask = np.zeros_like(case["image"], dtype=bool)
    window._external_mask = np.zeros_like(case["image"], dtype=bool)
    window.clear_external_mask()
    assert_stale("mask")
    fresh()
    assert window.set_exclusion_roi({"type": "rectangle", "x0": 0, "y0": 0, "x1": 3, "y1": 3})
    assert_stale("roi")
    fresh()
    window.set_display_settings({"scale": "log1p", "percentile": 95.0})
    assert window.butterfly_workbench.result_fresh
    window.butterfly_workbench.export_analysis(tmp_path / "display-only")
    window.close()


def test_trace_geometry_is_not_reported_as_fit_failure_but_evaluate_failure_is(qtbot):
    window = MainWindow(engine=_ButterflyEngine(), auto_preview=False, language="en")
    qtbot.addWidget(window)
    observed = np.ones((4, 4), dtype=float)
    trace = {
        "observed": observed,
        "valid_mask": np.ones_like(observed, dtype=bool),
        "analysis": {
            "ridge_method": "butterfly_curvature",
            "butterfly": {"stage": "trace", "resamples": 0},
        },
        "butterfly": {
            "settings": {"stage": "trace"},
            "points": [{"point_id": "p0", "qx": 0.1, "qy": 0.1, "accepted": True, "valid": True}],
            "arcs": [],
            "profiles": {},
            "candidate_fit": {"status": "not_fitted"},
        },
        "ellipse_fit": {"status": "not_fitted", "quality_status": "FAIL"},
        "metrics": {"success": False, "flags": ["solver_or_arc_support_unavailable"]},
    }
    generation = window._generation.next()
    window._on_worker_finished(generation, "measure_geometry", trace)
    assert "result_failed" not in window.flags_label.text()
    assert "not evaluated" in window.status_message.text().lower()

    evaluate = dict(trace)
    evaluate["analysis"] = {"ridge_method": "butterfly_curvature", "butterfly": {"stage": "evaluate"}}
    evaluate["butterfly"] = dict(trace["butterfly"], settings={"stage": "evaluate"})
    generation = window._generation.next()
    window._on_worker_finished(generation, "refine_geometry", evaluate)
    assert "result_failed" in window.flags_label.text()
    window.close()
