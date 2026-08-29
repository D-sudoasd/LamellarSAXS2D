from __future__ import annotations

import re

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6 import QtCore, QtWidgets

from butterfly_saxs.ui import MainWindow, create_app
from butterfly_saxs.ui.i18n import CATALOGS, LANGUAGE_SETTING_KEY, validate_language


def _settings(path) -> QtCore.QSettings:
    return QtCore.QSettings(str(path), QtCore.QSettings.Format.IniFormat)


class _PreviewEngine:
    def __init__(self) -> None:
        self.payloads: list[dict] = []

    def preview(self, *, parameters, payload):
        del parameters
        self.payloads.append(payload)
        observed = np.asarray(payload["observed"], dtype=float)
        return {
            "observed": observed,
            "model": observed.copy(),
            "residual": np.zeros_like(observed),
        }


def test_translation_catalogs_match_and_language_validation_is_strict() -> None:
    assert set(CATALOGS["zh_CN"]) == set(CATALOGS["en"])
    assert validate_language("zh_CN") == "zh_CN"
    assert validate_language("en") == "en"
    with pytest.raises(ValueError, match="unsupported UI language"):
        validate_language("fr")


def test_chinese_scientific_term_contract() -> None:
    canonical_english_keys = {
        "tab.refinement",
        "tab.measurements",
        "tab.batch",
        "tab.evolution",
        "dock.parameters",
        "button.preview",
        "button.optimize",
        "check.auto_preview",
        "check.resume_checkpoint",
        "group.roi",
        "group.fit_session",
        "group.analysis",
        "label.type",
        "label.q_min",
        "label.q_max",
        "label.draw_axis",
        "label.ridge_method",
        "label.angular_bins",
        "label.ridge_angles",
        "label.radial_bins",
        "label.curvature_sigma",
        "label.curvature_percentile",
        "label.normal_step",
        "label.max_pixels",
        "label.mode",
        "label.manifest",
        "label.checkpoint",
        "label.output",
        "label.y_parameter",
        "placeholder.auto",
        "special.all_pixels",
        "combo.rectangle",
        "combo.ellipse",
        "combo.radial_peak",
        "combo.surface_curvature",
        "combo.independent",
        "combo.warm_start",
        "view.observed",
        "view.model",
        "view.residual",
        "view.overlay",
        "axis.x_pixel",
        "axis.y_pixel",
        "axis.azimuth",
        "axis.angular_intensity",
        "axis.ridge_q",
        "axis.frame_time",
        "axis.value",
        "measurement.lobes",
        "measurement.ridge",
        "measurement.ellipse",
        "header.parameter",
        "header.value",
        "header.min",
        "header.max",
        "header.vary",
        "header.expr",
        "header.unit",
        "header.stderr",
        "header.angle_deg",
        "header.intensity",
        "header.baseline",
        "header.snr",
        "header.fwhm_deg",
        "header.coverage",
        "header.valid",
        "header.flags",
        "header.accepted",
        "header.method",
        "header.quantity",
        "header.frame",
        "header.status",
        "metric.ndata",
        "metric.flags",
        "metric.coverage",
        "boolean.true",
        "boolean.false",
        "ellipse.a",
        "ellipse.b",
        "ellipse.ellipticity",
        "ellipse.rmse",
        "ellipse.rss",
        "ellipse.flags",
    }
    assert {key: CATALOGS["zh_CN"][key] for key in canonical_english_keys} == {
        key: CATALOGS["en"][key] for key in canonical_english_keys
    }
    assert {
        key: CATALOGS["zh_CN"][key]
        for key in (
            "ellipse.axis_ratio",
            "ellipse.theta",
            "ellipse.ln",
            "ellipse.lz",
            "ellipse.n_points",
            "ellipse.quality",
            "ellipse.phi_app",
            "ellipse.alpha_candidate",
            "ellipse.psi_candidate",
        )
    } == {
        "ellipse.axis_ratio": "axis_ratio",
        "ellipse.theta": "theta_deg",
        "ellipse.ln": "L_N (nm)",
        "ellipse.lz": "L_z (nm)",
        "ellipse.n_points": "n_points",
        "ellipse.quality": "success",
        "ellipse.phi_app": "phi_app_deg",
        "ellipse.alpha_candidate": "alpha_candidate_deg",
        "ellipse.psi_candidate": "psi_candidate_deg",
    }
    assert CATALOGS["zh_CN"]["button.cancel"] == "取消"
    assert CATALOGS["zh_CN"]["status.ready"] == "就绪"
    assert not any(
        re.search(r"[A-Za-z0-9]\s*（", value)
        for value in CATALOGS["zh_CN"].values()
    )


def test_default_chinese_switch_and_global_persistence(qtbot, tmp_path) -> None:
    settings_path = tmp_path / "ui.ini"
    settings = _settings(settings_path)
    settings.clear()
    settings.sync()

    window = MainWindow(engine=object(), auto_preview=False, settings=settings)
    qtbot.addWidget(window)
    assert window.language == "zh_CN"
    assert window.windowTitle() == "LamellarSAXS2D · 二维精修"
    assert window.preview_button.text() == "Preview"
    assert window.parameter_model.headerData(0, QtCore.Qt.Orientation.Horizontal) == "Parameter"
    assert window.views.observed.title_label.text() == "Observed"
    assert window.fit_session_group.title() == "Fit session"
    assert window.manual_status_label.text() == "未审核"
    assert window.chinese_action.isChecked()

    window.set_language("en")
    assert window.language == "en"
    assert window.windowTitle() == "LamellarSAXS2D · 2D Refinement"
    assert window.preview_button.text() == "Preview"
    assert window.parameter_model.headerData(0, QtCore.Qt.Orientation.Horizontal) == "Parameter"
    assert window.views.observed.title_label.text() == "Observed"
    assert window.english_action.isChecked()
    assert settings.value(LANGUAGE_SETTING_KEY) == "en"
    window.close()

    restored = MainWindow(
        engine=object(),
        auto_preview=False,
        settings=_settings(settings_path),
    )
    qtbot.addWidget(restored)
    assert restored.language == "en"
    restored.close()

    invalid = _settings(tmp_path / "invalid.ini")
    invalid.setValue(LANGUAGE_SETTING_KEY, "fr")
    invalid.sync()
    fallback = MainWindow(engine=object(), auto_preview=False, settings=invalid)
    qtbot.addWidget(fallback)
    assert fallback.language == "zh_CN"
    fallback.close()


def test_chinese_analysis_panel_uses_english_scientific_labels(qtbot, tmp_path) -> None:
    window = MainWindow(
        engine=object(),
        auto_preview=False,
        language="zh_CN",
        settings=_settings(tmp_path / "analysis-labels.ini"),
    )
    qtbot.addWidget(window)

    assert [
        window.pages.tabText(window.pages.indexOf(page))
        for page in (
            window.refinement_page,
            window.measurements_page,
            window.batch_page,
            window.evolution_page,
        )
    ] == ["Refinement", "Measurements / Profiles", "Batch", "Evolution"]
    assert window.parameters_dock.windowTitle() == "Parameters"
    assert window.preview_button.text() == "Preview"
    assert window.optimize_button.text() == "Optimize"
    assert window.auto_preview_check.text() == "Auto Preview"
    assert window.batch_resume_check.text() == "Resume checkpoint"
    assert window.cancel_button.text() == "取消"
    assert window.apply_roi_button.text() == "应用"
    assert window.open_project_action.text() == "打开项目…"

    expected_labels = (
        (window.q_min_edit, "q min"),
        (window.q_max_edit, "q max"),
        (window.draw_axis_deg_spin, "draw axis (deg)"),
        (window.ridge_method_combo, "ridge method"),
        (window.n_angular_bins_spin, "angular bins"),
        (window.n_ridge_angles_spin, "ridge angles"),
        (window.n_radial_bins_spin, "radial bins"),
        (window.curvature_sigma_spin, "curvature sigma"),
        (window.curvature_percentile_spin, "curvature percentile"),
        (window.normal_step_spin, "normal step"),
        (window.max_pixels_spin, "max pixels"),
    )
    assert window.analysis_group.title() == "Analysis / Measurement"
    assert [
        window.analysis_form.labelForField(widget).text() for widget, _ in expected_labels
    ] == [expected for _, expected in expected_labels]
    assert window.q_min_edit.text() == "Auto"
    assert window.q_max_edit.text() == "Auto"
    assert window.max_pixels_spin.specialValueText() == "0 (all)"
    assert window.ridge_method_combo.currentText() == "Radial peak"
    assert window.ridge_method_combo.currentData() == "radial_peak"
    assert "ridge（" not in window.ridge_method_combo.toolTip()
    assert "ridge 位置" in window.ridge_method_combo.toolTip()

    assert window.roi_group.title() == "Exclusion ROI (pixel)"
    assert window.roi_type_label.text() == "Type"
    assert window.roi_type_combo.currentText() == "Rectangle"
    assert window.roi_type_combo.currentData() == "rectangle"
    assert window.fit_session_group.title() == "Fit session"
    assert window.fit_session_form.labelForField(window.reviewer_edit).text() == "审核人"
    assert window.batch_form.labelForField(window.batch_mode_combo).text() == "Mode"
    assert window.batch_form.labelForField(window.batch_manifest_edit).text() == "Manifest"
    assert window.batch_form.labelForField(window.batch_checkpoint_edit).text() == "Checkpoint"
    assert window.batch_form.labelForField(window.batch_output_edit).text() == "Output"
    assert window.batch_mode_combo.currentText() == "Independent"
    assert window.batch_mode_combo.currentData() == "independent"
    assert window.lobe_panel_label.text() == "Four-lobe measurements"
    assert window.ridge_panel_label.text() == "Ridge q-angle / accepted"
    assert window.ellipse_panel_label.text() == "Ellipse quantities"
    window.close()


def test_create_app_accepts_explicit_language(qtbot) -> None:
    app, window = create_app([], analysis_service=object(), language="en")
    del app
    qtbot.addWidget(window)
    assert window.language == "en"
    assert window.preview_button.text() == "Preview"
    window.close()


def test_default_chinese_auto_values_can_start_preview(qtbot, tmp_path) -> None:
    engine = _PreviewEngine()
    window = MainWindow(
        engine=engine,
        auto_preview=False,
        settings=_settings(tmp_path / "preview.ini"),
    )
    qtbot.addWidget(window)
    window.set_observed_data(np.ones((3, 4), dtype=float))

    assert window.q_min_edit.text() == "Auto"
    assert window.q_max_edit.text() == "Auto"
    generation = window.request_preview()
    qtbot.waitUntil(lambda: generation not in window._workers, timeout=3000)

    assert engine.payloads[0]["analysis"]["q_min"] is None
    assert engine.payloads[0]["analysis"]["q_max"] is None
    assert window.status_message.text() == "Preview 完成"
    window.close()


def test_special_value_and_batch_statuses_retranslate_without_changing_codes(
    qtbot, tmp_path
) -> None:
    window = MainWindow(
        engine=object(),
        auto_preview=False,
        settings=_settings(tmp_path / "batch.ini"),
    )
    qtbot.addWidget(window)
    records = [
        {"frame": "a.edf", "status": "ok"},
        {"frame": "b.edf", "status": "failed"},
        {"frame": "c.edf", "status": "skipped"},
        {"frame": "d.edf", "status": "CUSTOM_FLAG"},
    ]
    window._update_batch_rows(records)

    assert window.max_pixels_spin.specialValueText() == "0 (all)"
    assert [window.batch_table.item(row, 1).text() for row in range(4)] == [
        "成功",
        "失败",
        "已跳过",
        "CUSTOM_FLAG",
    ]
    assert [
        window.batch_table.item(row, 1).data(QtCore.Qt.ItemDataRole.UserRole)
        for row in range(4)
    ] == ["ok", "failed", "skipped", "CUSTOM_FLAG"]

    window.set_language("en", persist=False)
    assert window.max_pixels_spin.specialValueText() == "0 (all)"
    assert [window.batch_table.item(row, 1).text() for row in range(4)] == [
        "OK",
        "Failed",
        "Skipped",
        "CUSTOM_FLAG",
    ]
    assert [
        window.batch_table.item(row, 1).data(QtCore.Qt.ItemDataRole.UserRole)
        for row in range(4)
    ] == ["ok", "failed", "skipped", "CUSTOM_FLAG"]
    window.close()


def test_measurement_terms_and_booleans_retranslate_without_changing_raw_data(
    qtbot, tmp_path
) -> None:
    window = MainWindow(
        engine=object(),
        auto_preview=False,
        language="zh_CN",
        settings=_settings(tmp_path / "measurements.ini"),
    )
    qtbot.addWidget(window)
    result = {
        "observables": {
            "ridge": {
                "q_unit": "nm^-1",
                "points": [
                    {
                        "angle_deg": 15.0,
                        "q": 0.21,
                        "accepted": True,
                        "method": "surface_curvature",
                    }
                ],
            },
            "lobes": [
                {
                    "angle_deg": 45.0,
                    "intensity": 120.0,
                    "baseline": 5.0,
                    "snr": 8.0,
                    "fwhm_deg": 4.5,
                    "coverage": 0.9,
                    "valid": False,
                    "flags": ["raw_lobe_flag"],
                }
            ],
            "ellipse": {
                "a": 0.24,
                "b": 0.18,
                "axis_ratio": 0.75,
                "ellipticity": 0.25,
                "theta_deg": 12.0,
                "Ln_from_minor_axis_nm": 34.9,
                "Lz_from_draw_axis_nm": 26.2,
                "rmse": 0.02,
                "rss": 0.004,
                "n_points": 48,
                "success": False,
                "flags": ["raw_ellipse_flag"],
            },
            "phi_app_deg": 22.5,
            "alpha_candidate_deg": 11.25,
            "psi_candidate_deg": 33.75,
        }
    }
    window._update_measurements(result)

    user_role = QtCore.Qt.ItemDataRole.UserRole
    assert window.lobe_table.horizontalHeaderItem(4).text() == "FWHM (deg)"
    assert window.ridge_table.item(0, 2).text() == "True"
    assert window.ridge_table.item(0, 2).data(user_role) is True
    assert window.ridge_table.item(0, 3).text() == "surface_curvature"
    assert window.lobe_table.item(0, 6).text() == "False"
    assert window.lobe_table.item(0, 6).data(user_role) is False
    assert window.lobe_table.item(0, 7).text() == "raw_lobe_flag"
    assert window.ellipse_table.item(0, 0).text() == "a (major q)"
    assert window.ellipse_table.item(2, 0).text() == "axis_ratio"
    assert window.ellipse_table.item(4, 0).text() == "theta_deg"
    assert window.ellipse_table.item(5, 0).text() == "L_N (nm)"
    assert window.ellipse_table.item(6, 0).text() == "L_z (nm)"
    assert window.ellipse_table.item(9, 0).text() == "n_points"
    assert window.ellipse_table.item(10, 0).text() == "success"
    assert window.ellipse_table.item(10, 1).text() == "False"
    assert window.ellipse_table.item(10, 1).data(user_role) is False
    assert window.ellipse_table.item(11, 1).text() == "raw_ellipse_flag"
    assert window.ellipse_table.item(12, 0).text() == "phi_app_deg"
    assert window.ellipse_table.item(13, 0).text() == "alpha_candidate_deg"
    assert window.ellipse_table.item(14, 0).text() == "psi_candidate_deg"

    raw_observables = window.measurement_observables
    window.set_language("en", persist=False)

    assert window.ridge_table.item(0, 2).text() == "True"
    assert window.ridge_table.item(0, 2).data(user_role) is True
    assert window.ridge_table.item(0, 3).text() == "surface_curvature"
    assert window.lobe_table.item(0, 6).text() == "False"
    assert window.lobe_table.item(0, 6).data(user_role) is False
    assert window.lobe_table.item(0, 7).text() == "raw_lobe_flag"
    assert window.ellipse_table.item(10, 1).text() == "False"
    assert window.ellipse_table.item(10, 1).data(user_role) is False
    assert window.ellipse_table.item(11, 1).text() == "raw_ellipse_flag"
    assert window.ellipse_table.item(2, 0).text() == "axis ratio"
    assert window.ellipse_table.item(4, 0).text() == "theta (ellipse axis, deg)"
    assert window.ellipse_table.item(5, 0).text() == "Ln from minor axis (nm)"
    assert window.measurement_observables is raw_observables
    window.close()


def test_language_switch_preserves_scientific_and_review_state(qtbot, tmp_path) -> None:
    settings = _settings(tmp_path / "state.ini")
    window = MainWindow(
        engine=object(),
        auto_preview=False,
        language="zh_CN",
        settings=settings,
    )
    qtbot.addWidget(window)
    observed = np.arange(24, dtype=float).reshape(4, 6)
    window.set_observed_data(observed)
    assert window.set_parameter("theta_deg", 18.0)
    window.ridge_method_combo.setCurrentIndex(
        window.ridge_method_combo.findData("surface_curvature")
    )
    window.batch_mode_combo.setCurrentIndex(
        window.batch_mode_combo.findData("warm_start")
    )
    window.pages.setCurrentWidget(window.measurements_page)
    assert window.save_snapshot("基准 snapshot")
    window._fit_session["manual_status"] = "accepted"
    window._fit_session["reviewed_by"] = "Reviewer A"
    window._fit_session["review_notes"] = "keep raw note"
    window._sync_fit_session_controls()
    window.reviewer_edit.setText("Reviewer draft")
    window.review_notes_edit.setText("uncommitted draft note")
    window._update_metrics(
        {
            "metrics": {
                "rmse": 0.25,
                "ndata": observed.size,
                "flags": ["ok"],
                "valid_fraction": 0.75,
            }
        },
        observed,
        None,
    )
    window._set_status("status.image_failed", flags="error", error="decoder boom")

    project_before = window.project_to_dict()
    parameters_before = window.parameter_model.parameter_dict()
    fit_session_before = window.fit_session
    generation_before = window._generation.current
    observed_before = np.array(window.views.observed.image_data, copy=True)
    ridge_data_before = window.ridge_method_combo.currentData()
    batch_data_before = window.batch_mode_combo.currentData()
    page_before = window.pages.currentWidget()
    snapshot_data_before = window.snapshot_combo.currentData()

    window.set_language("en", persist=False)

    assert window.status_message.text() == "Image load failed: decoder boom"
    assert window.flags_label.text() == "flags: error"
    assert window.ndata_label.text() == f"ndata: {observed.size}"
    assert window.manual_status_label.text() == "accepted"
    assert window.snapshot_combo.currentText().endswith("基准 snapshot")
    assert window.parameter_model.parameter_dict() == parameters_before
    assert window.fit_session == fit_session_before
    assert window.project_to_dict() == project_before
    assert "language" not in project_before
    assert window._generation.current == generation_before
    np.testing.assert_array_equal(window.views.observed.image_data, observed_before)
    assert window.ridge_method_combo.currentData() == ridge_data_before
    assert window.batch_mode_combo.currentData() == batch_data_before
    assert window.pages.currentWidget() is page_before
    assert window.snapshot_combo.currentData() == snapshot_data_before
    assert window.reviewer_edit.text() == "Reviewer draft"
    assert window.review_notes_edit.text() == "uncommitted draft note"

    window.set_language("zh_CN", persist=False)
    assert window.status_message.text() == "图像载入失败：decoder boom"
    assert window.flags_label.text() == "flags: error"
    assert window.manual_status_label.text() == "已接受"
    window._set_status("status.running", kind_key="job.preview")
    assert window.status_message.text() == "正在运行：Preview…"
    window.set_language("en", persist=False)
    assert window.status_message.text() == "Running preview…"
    window.close()


def test_language_updates_dialog_titles_and_filters(qtbot, tmp_path, monkeypatch) -> None:
    window = MainWindow(
        engine=object(),
        auto_preview=False,
        language="zh_CN",
        settings=_settings(tmp_path / "dialogs.ini"),
    )
    qtbot.addWidget(window)
    open_calls: list[tuple] = []
    directory_calls: list[tuple] = []

    def fake_open(*args):
        open_calls.append(args)
        return "", ""

    def fake_directory(*args):
        directory_calls.append(args)
        return ""

    monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileName", fake_open)
    monkeypatch.setattr(QtWidgets.QFileDialog, "getExistingDirectory", fake_directory)
    monkeypatch.setattr(window, "_current_result_is_reviewable", lambda: True)

    assert not window.open_image()
    assert open_calls[-1][1] == "打开二维 SAXS 图像"
    assert "所有文件" in open_calls[-1][3]
    assert not window.export_manual_evidence()
    assert directory_calls[-1][1] == "选择用于人工拟合证据的空文件夹"

    window.set_language("en", persist=False)
    assert not window.open_image()
    assert open_calls[-1][1] == "Open 2D SAXS image"
    assert "All files" in open_calls[-1][3]
    assert not window.export_manual_evidence()
    assert directory_calls[-1][1] == "Select an empty folder for manual-fit evidence"
    window.close()
