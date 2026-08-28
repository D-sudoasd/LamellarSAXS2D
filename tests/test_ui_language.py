from __future__ import annotations

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


def test_default_chinese_switch_and_global_persistence(qtbot, tmp_path) -> None:
    settings_path = tmp_path / "ui.ini"
    settings = _settings(settings_path)
    settings.clear()
    settings.sync()

    window = MainWindow(engine=object(), auto_preview=False, settings=settings)
    qtbot.addWidget(window)
    assert window.language == "zh_CN"
    assert window.windowTitle() == "LamellarSAXS2D · 二维精修"
    assert window.preview_button.text() == "预览"
    assert window.parameter_model.headerData(0, QtCore.Qt.Orientation.Horizontal) == "参数"
    assert window.views.observed.title_label.text() == "观测图"
    assert window.fit_session_group.title() == "拟合会话"
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

    assert window.q_min_edit.text() == "自动"
    assert window.q_max_edit.text() == "自动"
    generation = window.request_preview()
    qtbot.waitUntil(lambda: generation not in window._workers, timeout=3000)

    assert engine.payloads[0]["analysis"]["q_min"] is None
    assert engine.payloads[0]["analysis"]["q_max"] is None
    assert window.status_message.text() == "预览完成"
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

    assert window.max_pixels_spin.specialValueText() == "0（全部）"
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
    assert window.flags_label.text() == "标记: error"
    assert window.manual_status_label.text() == "已接受"
    window._set_status("status.running", kind_key="job.preview")
    assert window.status_message.text() == "正在运行预览…"
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
