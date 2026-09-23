from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from butterfly_saxs.ui.butterfly_workbench import ButterflyWorkbench
from butterfly_saxs.ui.butterfly_summary import build_butterfly_quality_summary


def _annular_result() -> dict:
    angles = [0.0, 90.0, 180.0, 270.0]
    profiles = {}
    annuli = []
    points = []
    for index, q_center in enumerate((0.20, 0.30)):
        profile_id = f"annulus-{index:03d}"
        profiles[profile_id] = {
            "profile_axis": "azimuthal",
            "angle_deg": list(range(0, 360, 90)),
            "raw_intensity": [1.0 + index, 4.0, 1.2 + index, 3.5],
            "smoothed_intensity": [1.1 + index, 3.8, 1.3 + index, 3.2],
            "counts": [80, 81, 79, 82],
            "coverage": [0.9, 0.91, 0.88, 0.92],
            "peak_angles_deg": angles,
            "q_center": q_center,
            "q_min": q_center - 0.025,
            "q_max": q_center + 0.025,
            "q_unit": "nm^-1",
            "reason": "selected",
        }
        selected = [
            {
                "point_id": f"track-{branch}-{index}",
                "chi_deg": angle,
                "qx": q_center,
                "qy": 0.0,
                "trajectory_id": f"trajectory-{branch}",
                "branch_id": branch // 2,
                "side": "upper" if branch % 2 == 0 else "lower",
                "accepted": True,
                "valid": True,
            }
            for branch, angle in enumerate(angles)
        ]
        annuli.append(
            {
                "annulus_index": index,
                "q_center": q_center,
                "q_min": q_center - 0.025,
                "q_max": q_center + 0.025,
                "profile_id": profile_id,
                "selected_peaks": selected,
                "status": "selected",
                "reason": "selected",
            }
        )
        if index == 0:
            points.extend(selected)
    return {
        "method_version": "butterfly-annular-v1",
        "trace_method": "annular_peak",
        "q_unit": "nm^-1",
        "points": points,
        "profiles": profiles,
        "annular_peaks": {
            "q_unit": "nm^-1",
            "q_edges": [0.175, 0.225, 0.275, 0.325],
            "angle_centers_deg": [0.0, 90.0, 180.0, 270.0],
            "settings": {"annular_radial_bins": 40, "annular_angle_bins": 72},
            "annuli": annuli,
        },
        "quality": {
            "status": "WARN",
            "engineering_status": "WARN",
            "flags": ["ellipse_not_evaluated"],
        },
    }


def test_fresh_workbench_defaults_to_annular_controls(qtbot):
    page = ButterflyWorkbench(language="zh_CN")
    qtbot.addWidget(page)
    page.show()
    qtbot.wait(10)

    assert page.trace_method_combo.currentData() == "annular_peak"
    assert page.analysis_settings["ridge_method"] == "butterfly_curvature"
    assert page.annular_radial_bins.value() == 40
    assert page.annular_angle_bins.value() == 72
    assert page.annular_radial_bins.isVisible()
    assert page.annular_angle_bins.isVisible()
    assert not page.sector_width_spin.isVisible()
    assert not page.sector_step_spin.isVisible()
    assert not page.global_max_check.isChecked()
    assert not page.supported_peaks_check.isChecked()
    assert not page.ellipse_diagnostic.isVisible()


def test_annular_result_lists_every_ring_and_renders_full_angular_profile(qtbot):
    page = ButterflyWorkbench(language="zh_CN")
    qtbot.addWidget(page)
    page.set_result(_annular_result())

    assert page.trace_method_combo.currentData() == "annular_peak"
    assert page.point_list.count() == 2
    assert "q=[0.175, 0.225]" in page.point_list.item(0).text()
    assert "4 个峰" in page.point_list.item(0).text()
    assert "q=[0.275, 0.325]" in page.point_list.item(1).text()

    page.point_list.setCurrentRow(0)
    assert page.normal_profile._x_label == "χ（deg）"
    assert page.normal_profile._series_names == ("raw", "smoothed", "counts", "coverage")
    assert page.normal_profile.table.rowCount() == 4
    headers = [
        page.normal_profile.table.horizontalHeaderItem(column).text()
        for column in range(page.normal_profile.table.columnCount())
    ]
    assert "有效像素数（像素）" in headers
    assert "覆盖率（无量纲）" in headers
    assert "平滑（仅用于定位）" in headers
    assert len(page.normal_profile.plot.listDataItems()) == 2
    assert len(page.normal_profile.plot.items()) >= 4
    assert not page.ellipse_diagnostic.isVisible()
    assert getattr(page.qspace, "_selected_point_id", None) is None


def test_annular_mode_roundtrip_invalidation_and_compatibility_modes(qtbot):
    page = ButterflyWorkbench(language="en")
    qtbot.addWidget(page)
    page.show()
    qtbot.wait(10)
    page.set_result(_annular_result())
    assert page.result_fresh

    events: list[dict] = []
    page.analysisChanged.connect(events.append)
    page.annular_angle_bins.setValue(96)
    assert not page.result_fresh
    assert events[-1]["butterfly"]["trace_method"] == "annular_peak"
    assert events[-1]["butterfly"]["annular_angle_bins"] == 96

    page.set_analysis_settings(
        {
            "stage": "trace",
            "trace_method": "annular_peak",
            "annular_radial_bins": 44,
            "annular_angle_bins": 96,
        },
        replace=True,
    )
    assert page.butterfly_settings["annular_radial_bins"] == 44
    assert page.butterfly_settings["annular_angle_bins"] == 96

    page.trace_method_combo.setCurrentIndex(page.trace_method_combo.findData("radial_sector"))
    assert page.trace_method_combo.currentData() == "radial_sector"
    assert page.sector_width_spin.isVisible()
    assert not page.annular_angle_bins.isVisible()
    assert page.ellipse_diagnostic.isVisible() is False

    page.trace_method_combo.setCurrentIndex(page.trace_method_combo.findData("curvature"))
    assert page.trace_method_combo.currentData() == "curvature"
    assert page.global_max_check.isChecked()
    assert page.supported_peaks_check.isChecked()
    assert page.ellipse_diagnostic.isVisible()


def test_legacy_payload_without_trace_method_remains_curvature(qtbot):
    page = ButterflyWorkbench(language="en")
    qtbot.addWidget(page)
    legacy = {
        "method_version": "butterfly-curvature-arcs-v2.1",
        "points": [{"point_id": "legacy-0", "qx": 0.1, "qy": 0.0}],
        "profiles": {},
    }
    page.set_result(legacy)

    assert page.trace_method_combo.currentData() == "curvature"
    assert page.global_max_check.isChecked()
    assert page.supported_peaks_check.isChecked()


def test_annular_summary_reports_tracks_without_q_median_or_spacing():
    state = build_butterfly_quality_summary(
        _annular_result(),
        trace_method="annular_peak",
        result_fresh=True,
        data_ready=True,
    )
    assert state.trace_method == "annular_peak"
    assert state.q_star is None
    assert state.l_ring is None
    assert state.next_step_key in {"evaluate", "ellipse_review", "ring_review"}
