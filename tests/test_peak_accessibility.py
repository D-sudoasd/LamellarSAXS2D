from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6 import QtCore, QtGui, QtTest

from butterfly_saxs.ui.butterfly_workbench import ButterflyWorkbench


def _frame(shape: tuple[int, int] = (12, 12)) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    yy, xx = np.indices(shape, dtype=float)
    observed = 2.0 + 0.2 * xx + 0.1 * yy
    qx = 0.01 * xx + 0.002 * yy**2
    qy = 0.008 * yy + 0.001 * xx**2
    return observed, qx, qy


def _result(qx: np.ndarray, qy: np.ndarray, *, profiles: bool = True) -> dict:
    peak_x, peak_y = 8, 2
    peak_qx = float(qx[peak_y, peak_x])
    peak_qy = float(qy[peak_y, peak_x])
    angular = {
        "angle_deg": [-90.0, 0.0, 45.0, 90.0],
        "intensity_raw": [1.0, 2.0, 8.0, 1.5],
        "intensity_smoothed": [1.1, 2.1, 7.5, 1.4],
        "intensity_isotropic_reference": [1.0, 1.2, 1.0, 1.1],
        "intensity_detection": [0.0, 0.8, 6.5, 0.2],
        "topangular_detection": "isotropic_compensated",
    }
    radial = {
        "q": [0.1, 0.2, 0.3],
        "mean_intensity_raw": [2.0, 9.0, 3.0],
        "mean_intensity_smoothed": [2.1, 8.4, 3.1],
    }
    return {
        "measurement_status": "evaluated",
        "points": [],
        "peak_landmarks": {
            "status": "ok",
            "q_unit": "1/nm",
            "raw_global_max": {
                "qx": peak_qx,
                "qy": peak_qy,
                "q": math.hypot(peak_qx, peak_qy),
                "raw_intensity": 10.0,
            },
            "peaks": [
                {
                    "peak_id": "P1",
                    "qx": peak_qx,
                    "qy": peak_qy,
                    "q": math.hypot(peak_qx, peak_qy),
                    "angular_peak_deg": 45.0,
                    "raw_intensity": 8.0,
                    "smoothed_intensity": 7.5,
                }
            ],
            "profiles": {"angular": angular, "radial": radial} if profiles else {},
        },
        "profiles": {},
    }


def _accessible_name(widget) -> str:
    interface = QtGui.QAccessible.queryAccessibleInterface(widget)
    assert interface is not None
    return interface.text(QtGui.QAccessible.Text.Name)


def test_dynamic_workflow_and_geometry_warning_are_accessible_after_translation(qtbot):
    observed, qx, qy = _frame()
    page = ButterflyWorkbench(language="en")
    qtbot.addWidget(page)
    page.resize(1180, 820)
    page.set_data(observed, qx=qx, qy=qy, q_unit="1/nm")
    page.set_result(
        {
            "measurement_status": "evaluated",
            "points": [],
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
                    "residual_sigma_ratio": 10.2,
                    "median_localization_sigma_q": 0.01,
                },
                "provisional_limits": {"residual_sigma_ratio_max": 3.0},
                "flags": [],
            },
        }
    )
    page.show()
    qtbot.wait(40)

    warning = page.fit_assessment_label.text()
    workflow = page.workflow_hint_label.text()
    assert "10.2" in warning and "3" in warning
    assert _accessible_name(page.fit_assessment_label) == warning
    assert _accessible_name(page.workflow_hint_label) == workflow

    page.set_language("zh_CN")
    translated_warning = page.fit_assessment_label.text()
    translated_workflow = page.workflow_hint_label.text()
    assert "10.2" in translated_warning and "3" in translated_warning
    assert _accessible_name(page.fit_assessment_label) == translated_warning
    assert _accessible_name(page.workflow_hint_label) == translated_workflow
    assert translated_workflow != workflow
    page.close()


def test_peak_profile_table_is_available_with_plot_and_keeps_series_units(qtbot):
    observed, qx, qy = _frame()
    result = _result(qx, qy)
    page = ButterflyWorkbench(language="en")
    qtbot.addWidget(page)
    page.resize(1180, 820)
    page.set_data(observed, qx=qx, qy=qy, q_unit="1/nm")
    page.set_result(result)
    page.show()
    qtbot.wait(40)
    revision = page.result_revision
    analysis_events: list[dict] = []
    page.analysisChanged.connect(analysis_events.append)

    page.peak_table.setCurrentCell(1, 0)
    panel = page.peak_angular_profile
    assert panel.plot is not None  # the project environment includes pyqtgraph
    assert panel.table is not None
    assert panel.view_tabs is not None
    assert panel.view_tabs.currentIndex() == 0
    assert panel.view_tabs.tabText(0) == "Plot"
    assert panel.view_tabs.tabText(1) == "Data"
    assert panel.table.rowCount() == 4
    assert [panel.table.horizontalHeaderItem(column).text() for column in range(5)] == [
        "chi (deg)",
        "raw",
        "smoothed",
        "isotropic reference",
        "detection",
    ]
    assert panel.table.item(2, 1).text() == "8"
    assert panel.table.item(2, 2).text() == "7.5"
    assert panel.table.item(2, 3).text() == "1"
    assert panel.table.item(2, 4).text() == "6.5"
    assert "chi (deg)" in panel.table.accessibleDescription()
    assert "detection" in panel.table.accessibleDescription()
    radial = page.peak_radial_profile
    assert [radial.table.horizontalHeaderItem(column).text() for column in range(3)] == [
        "q (1/nm)",
        "raw",
        "smoothed",
    ]
    assert radial.table.item(1, 1).text() == "9"
    assert radial.table.item(1, 2).text() == "8.4"

    panel.view_tabs.tabBar().setFocus()
    QtTest.QTest.keyClick(panel.view_tabs.tabBar(), QtCore.Qt.Key.Key_Right)
    assert panel.view_tabs.currentIndex() == 1
    panel.table.setFocus()
    panel.table.setCurrentCell(0, 0)
    QtTest.QTest.keyClick(panel.table, QtCore.Qt.Key.Key_Down)
    assert panel.table.currentRow() == 1

    page.set_language("zh_CN")
    assert panel.view_tabs.tabText(0) == "曲线"
    assert panel.view_tabs.tabText(1) == "数据"
    assert panel.table.horizontalHeaderItem(0).text() == "方位角 chi（deg）"
    assert panel.table.horizontalHeaderItem(1).text() == "原始"
    assert panel.table.horizontalHeaderItem(2).text() == "平滑"
    assert panel.table.horizontalHeaderItem(3).text() == "各向同性参考"
    assert panel.table.horizontalHeaderItem(4).text() == "检测曲线"
    assert "方位角 chi（deg）" in panel.table.accessibleDescription()
    assert panel.table.item(2, 1).text() == "8"
    assert radial.table.horizontalHeaderItem(0).text() == "q（1/nm）"
    assert radial.table.horizontalHeaderItem(1).text() == "原始"
    assert radial.table.horizontalHeaderItem(2).text() == "平滑"
    assert radial.table.item(1, 1).text() == "9"
    assert page.result_revision == revision
    assert page.result_fresh
    assert analysis_events == []
    page.close()


def test_peak_profile_data_view_clears_stale_values_and_handles_missing_data(qtbot):
    observed, qx, qy = _frame()
    page = ButterflyWorkbench(language="en")
    qtbot.addWidget(page)
    page.resize(1180, 820)
    page.set_data(observed, qx=qx, qy=qy, q_unit="1/nm")
    page.set_result(_result(qx, qy))
    page.peak_table.setCurrentCell(1, 0)
    panel = page.peak_angular_profile
    assert panel.table.rowCount() == 4

    page.set_job_status("cancelled", "trace")
    assert panel.table.rowCount() == 0
    assert panel.plot.listDataItems() == []
    assert panel.table.accessibleDescription() == "No profile values are available."
    assert panel.table.horizontalHeaderItem(0).text() == "Coordinate"

    page.set_language("zh_CN")
    assert panel.table.accessibleDescription() == "当前没有可读取的剖面数值。"
    page.set_result(_result(qx, qy, profiles=False))
    page.peak_table.setCurrentCell(1, 0)
    assert panel.table.rowCount() == 0
    assert panel.empty_label.text() == "当前结果没有可用的峰位剖面。"
    assert panel.table.accessibleDescription() == "当前没有可读取的剖面数值。"
    page.close()
