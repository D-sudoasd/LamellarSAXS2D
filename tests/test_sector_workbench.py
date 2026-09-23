from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from butterfly_saxs.ui.butterfly_workbench import ButterflyWorkbench


def _sector_result() -> dict:
    profiles = {
        "sector-000": {
            "profile_axis": "radial",
            "q_unit": "nm^-1",
            "q": [0.12, 0.16, 0.20, 0.24],
            "raw_intensity": [1.0, 3.0, 2.0, 1.2],
            "smoothed_intensity": [1.1, 2.7, 2.1, 1.3],
            "counts": [80, 82, 81, 79],
            "coverage": [0.9, 0.92, 0.91, 0.88],
            "selected_peak_q": 0.16,
        },
        "sector-001": {
            "profile_axis": "radial",
            "q_unit": "nm^-1",
            "q": [0.12, 0.16, 0.20, 0.24],
            "raw_intensity": [0.8, 0.9, 0.85, 0.7],
            "smoothed_intensity": [0.8, 0.88, 0.84, 0.72],
            "counts": [20, 18, 19, 17],
            "coverage": [0.22, 0.20, 0.21, 0.19],
            "selected_peak_q": None,
            "failure_reason": "low_coverage",
        },
    }
    sectors = [
        {
            "point_id": "sector-000",
            "source_method": "radial_sector",
            "sector_center_deg": 35.0,
            "sector_width_deg": 10.0,
            "q_star": 0.16,
            "accepted": True,
            "valid": True,
        },
        {
            "point_id": "sector-001",
            "source_method": "radial_sector",
            "sector_center_deg": 40.0,
            "sector_width_deg": 10.0,
            "q_star": None,
            "failure_reason": "low_coverage",
            "accepted": False,
            "valid": False,
            "profile_only": True,
        },
    ]
    return {
        "points": [
            {
                "point_id": "sector-000",
                "qx": 0.16,
                "qy": 0.0,
                "source_method": "radial_sector",
                "sector_center_deg": 35.0,
                "sector_width_deg": 10.0,
                "q_star": 0.16,
                "accepted": True,
                "valid": True,
            }
        ],
        "profiles": profiles,
        "sector_peaks": {"sectors": sectors},
        "peak_landmarks": {
            "raw_global_max": {"q": 0.3, "qx": 0.3, "qy": 0.0, "raw_intensity": 9},
            "peaks": [{"peak_id": "P1", "q": 0.16, "qx": 0.16, "qy": 0.0}],
        },
    }


def test_sector_controls_default_roundtrip_and_invalidate(qtbot):
    page = ButterflyWorkbench(language="zh_CN")
    qtbot.addWidget(page)
    page.show()
    qtbot.wait(10)

    assert page.trace_method_combo.currentData() == "annular_peak"
    page.trace_method_combo.setCurrentIndex(page.trace_method_combo.findData("radial_sector"))
    assert page.sector_width_spin.value() == pytest.approx(10.0)
    assert page.sector_step_spin.value() == pytest.approx(5.0)
    assert not page.global_max_check.isChecked()
    assert not page.supported_peaks_check.isChecked()
    assert not page.ellipse_diagnostic.isVisible()

    page.trace_method_combo.setCurrentIndex(page.trace_method_combo.findData("curvature"))
    qtbot.wait(10)
    assert page.ellipse_diagnostic.isVisible()
    page.trace_method_combo.setCurrentIndex(page.trace_method_combo.findData("radial_sector"))
    qtbot.wait(10)
    assert not page.ellipse_diagnostic.isVisible()

    page.set_result(_sector_result())
    assert page.result_fresh
    events: list[dict] = []
    page.analysisChanged.connect(events.append)
    page.sector_width_spin.setValue(12.0)

    assert not page.result_fresh
    assert events[-1]["butterfly"]["trace_method"] == "radial_sector"
    assert events[-1]["butterfly"]["sector_width_deg"] == pytest.approx(12.0)
    assert events[-1]["butterfly"]["sector_step_deg"] == pytest.approx(5.0)


def test_loading_legacy_curvature_result_restores_landmarks_without_overriding_user(qtbot):
    legacy = _sector_result()
    legacy.pop("sector_peaks")
    legacy["method_version"] = "butterfly-curvature-arcs-v2.1"
    legacy["points"][0]["source_method"] = "butterfly_curvature"

    page = ButterflyWorkbench(language="en")
    qtbot.addWidget(page)
    assert not page.global_max_check.isChecked()
    assert not page.supported_peaks_check.isChecked()
    page.set_result(legacy)
    assert page.butterfly_settings["trace_method"] == "curvature"
    assert page.global_max_check.isChecked()
    assert page.supported_peaks_check.isChecked()

    manual = ButterflyWorkbench(language="en")
    qtbot.addWidget(manual)
    manual.global_max_check.setChecked(True)
    manual.set_result(legacy)
    assert manual.butterfly_settings["trace_method"] == "curvature"
    assert manual.global_max_check.isChecked()
    assert not manual.supported_peaks_check.isChecked()


def test_replaced_legacy_recipe_without_trace_method_stays_curvature(qtbot):
    page = ButterflyWorkbench(language="en")
    qtbot.addWidget(page)

    page.set_analysis_settings({"stage": "trace", "edits": []}, replace=True)
    assert page.butterfly_settings["trace_method"] == "curvature"
    assert page.trace_method_combo.currentData() == "curvature"
    assert page.global_max_check.isChecked()
    assert page.supported_peaks_check.isChecked()

    page.set_analysis_settings(
        {
            "stage": "trace",
            "trace_method": "radial_sector",
            "sector_width_deg": 14,
            "sector_step_deg": 7,
        },
        replace=True,
    )
    assert page.butterfly_settings["trace_method"] == "radial_sector"
    assert page.sector_width_spin.value() == pytest.approx(14.0)
    assert page.sector_step_spin.value() == pytest.approx(7.0)


def test_sector_list_includes_profile_only_rows_without_fake_qspace_points(qtbot):
    page = ButterflyWorkbench(language="zh_CN")
    qtbot.addWidget(page)
    page.set_result(_sector_result())

    assert page.point_list.count() == 2
    assert "χ=35" in page.point_list.item(0).text()
    assert "q*=0.16" in page.point_list.item(0).text()
    assert "low_coverage" in page.point_list.item(1).text()
    assert "sector-001" not in page.point_list.item(1).text()

    page.point_list.setCurrentRow(1)
    assert getattr(page.qspace, "_selected_point_id", None) is None
    page._exclude_selected_point()
    assert page.edits == []


def test_radial_profile_uses_absolute_q_and_keeps_smoothed_as_locator_only(qtbot):
    page = ButterflyWorkbench(language="zh_CN")
    qtbot.addWidget(page)
    page.set_result(_sector_result())
    page.point_list.setCurrentRow(0)

    assert page.normal_profile._x_label == "q（nm^-1）"
    assert page.normal_profile._series_names == ("raw", "smoothed", "counts", "coverage")
    assert "sector-000" not in page.normal_profile.title_label.text()
    assert "source=" not in page.normal_profile.title_label.text()
    assert "point_id=sector-000" in page.normal_profile.title_label.toolTip()
    headers = [
        page.normal_profile.table.horizontalHeaderItem(column).text()
        for column in range(page.normal_profile.table.columnCount())
    ]
    assert "有效像素数（像素）" in headers
    assert "覆盖率（无量纲）" in headers
    assert "平滑（仅用于定位）" in headers
    assert "拟合" not in page.normal_profile.title_label.text()
    if page.normal_profile.plot is not None:
        assert "仅用于定位" in page.normal_profile.plot.accessibleDescription()
        assert len(page.normal_profile.plot.listDataItems()) == 2
        assert not page.normal_profile.plot.getAxis("bottom").autoSIPrefix

    page.point_list.setCurrentRow(1)
    assert "覆盖不足" in page.normal_profile.title_label.text()
    assert "reason=low_coverage" in page.normal_profile.title_label.toolTip()
    assert page.normal_profile._x_label == "q（nm^-1）"
    assert page.normal_profile.table.rowCount() == 4


def test_sector_profile_and_method_label_follow_language(qtbot):
    page = ButterflyWorkbench(language="zh_CN")
    qtbot.addWidget(page)
    page.set_result(_sector_result())
    page.point_list.setCurrentRow(0)
    assert "扇区积分 I(q)" in page.method_label.text()
    assert "扇区积分 I(q)" in page.normal_profile.title_label.text()

    page.set_language("en")
    assert "Radial sector" in page.method_label.text()
    assert "Radial sector I(q)" in page.normal_profile.title_label.text()
    assert "Sector-integrated" in page.trace_method_combo.itemText(page.trace_method_combo.findData("radial_sector"))


def test_profile_only_sector_without_profile_keeps_failure_context(qtbot):
    page = ButterflyWorkbench(language="zh_CN")
    qtbot.addWidget(page)
    result = _sector_result()
    result["profiles"].pop("sector-001")
    page.set_result(result)
    page.point_list.setCurrentRow(1)

    assert "覆盖不足" in page.normal_profile.title_label.text()
    assert "reason=low_coverage" in page.normal_profile.title_label.toolTip()
    assert "暂无扇区径向剖面" in page.normal_profile.empty_label.text()
    assert getattr(page.qspace, "_selected_point_id", None) is None


def test_sector_review_uses_unassigned_peak_definitions_without_recomputing_values(qtbot):
    page = ButterflyWorkbench(language="zh_CN")
    qtbot.addWidget(page)
    result = _sector_result()
    result["candidate_fit"] = {
        "q_star_from_arcs": 0.16,
        "L_from_observed_radius_nm": 39.27,
    }
    result["measurement_summary"] = {
        "q_star_sector_median": 0.17,
        "q_star_sector_median_unit": "nm^-1",
        "apparent_period_from_sector_median_nm": 36.96,
        "aggregation": "median of selected finite sector-profile peaks after manual exclusion",
    }
    result["quality"] = {"status": "WARN", "flags": []}
    page.set_result(result)
    labels = [
        page.quantity_table.item(row, 0).text()
        for row in range(page.quantity_table.rowCount())
    ]

    assert "主峰 q*中位数（未定级）" in labels
    assert "2π/q*（表观）" in labels
    assert page.quantity_table.item(labels.index("主峰 q*中位数（未定级）"), 1).text() == "0.17"
    assert page.quantity_table.item(labels.index("2π/q*（表观）"), 1).text() == "36.96"
