from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from butterfly_saxs.ui.figure_export_dialog import FigureExportDialog, _pixel_size


def test_dialog_preset_dimensions_are_explicit_and_static_preview_is_not_claimed_as_rendered(qtbot):
    dialog = FigureExportDialog(language="en", q_unit="1/nm")
    qtbot.addWidget(dialog)

    assert [dialog.width_combo.itemData(i) for i in range(dialog.width_combo.count())] == [
        89.0,
        183.0,
    ]
    assert dialog.width_combo.itemText(0) == "Single column · 89 mm"
    assert dialog.width_combo.itemText(1) == "Double column · 183 mm"
    assert [dialog.dpi_combo.itemData(i) for i in range(dialog.dpi_combo.count())] == [
        300,
        600,
        1200,
    ]
    assert _pixel_size(89.0, 300) == (1051, 1771)
    assert _pixel_size(183.0, 1200) == (8645, 4440)
    assert "not a rendered" in dialog._preview_caption.text().lower()
    assert dialog.start_button.isDefault()


def test_dialog_uses_non_overwriting_subfolder_and_surfaces_uncalibrated_q(tmp_path, qtbot):
    (tmp_path / "butterfly-figure").mkdir()
    dialog = FigureExportDialog(
        language="zh_CN",
        q_unit="pixel-q",
        stage="evaluate",
        quality_status="WARN",
        scientific_status="NOT_ACCEPTED",
        has_qx=False,
        has_qy=True,
    )
    qtbot.addWidget(dialog)
    dialog.parent_dir_edit.setText(str(tmp_path))

    assert "butterfly-figure-2" in dialog.target_preview_label.text()
    assert "无物理 q 标定" in dialog.q_unit_label.text()
    assert "诊断图" in dialog.q_unit_label.text()
    assert "NOT_ACCEPTED" in dialog.scientific_label.text()
    assert "qx" in dialog.coordinates_label.text()
    assert dialog._parent_label.buddy() is dialog.parent_dir_edit
    assert dialog._width_label.buddy() is dialog.width_combo


def test_dialog_start_is_single_signal_and_open_button_requires_explicit_click(tmp_path, qtbot):
    dialog = FigureExportDialog(language="en")
    qtbot.addWidget(dialog)
    dialog.parent_dir_edit.setText(str(tmp_path))
    emitted = []
    opened = []
    dialog.startRequested.connect(emitted.append)
    dialog.openRequested.connect(opened.append)

    dialog.start_button.click()
    qtbot.wait(10)
    assert len(emitted) == 1
    assert emitted[0]["parent"] == tmp_path.resolve()
    assert emitted[0]["width_mm"] == 183.0
    assert emitted[0]["dpi"] == 600
    assert dialog.is_running
    assert not dialog.start_button.isEnabled()

    index = tmp_path / "bundle" / "index.html"
    index.parent.mkdir()
    index.write_text("<html></html>", encoding="utf-8")
    dialog.set_exported_paths({"index": index})
    assert dialog.open_button.isEnabled()
    dialog.open_button.click()
    assert opened == [index.resolve()]


def test_dialog_retranslates_while_open_and_refreshes_context(qtbot):
    dialog = FigureExportDialog(language="en", q_unit="1/nm", stage="trace")
    qtbot.addWidget(dialog)
    dialog.set_measurement_context(
        {
            "q_unit": "unknown",
            "stage": "evaluate",
            "quality_status": "FAIL",
            "scientific_status": "not_assessed",
            "has_qx": True,
            "has_qy": False,
        }
    )
    dialog.set_language("zh_CN")
    assert "评估" in dialog.stage_label.text()
    assert "无物理 q 标定" in dialog.q_unit_label.text()
    assert "qy" in dialog.coordinates_label.text()
    assert "失败" not in dialog.quality_label.text()
    assert "FAIL" in dialog.quality_label.text()


def test_dialog_cancel_emits_worker_cancel_when_running(qtbot):
    dialog = FigureExportDialog(language="en")
    qtbot.addWidget(dialog)
    cancelled = []
    dialog.cancelRequested.connect(lambda: cancelled.append(True))
    dialog.set_export_running(True)
    dialog.cancel_button.click()
    assert cancelled == [True]
    assert not dialog.cancel_button.isEnabled()


def test_dialog_retranslates_running_error_and_stale_statuses(tmp_path, qtbot):
    dialog = FigureExportDialog(language="en")
    qtbot.addWidget(dialog)
    dialog.set_export_running(True)
    dialog.set_language("zh_CN")
    assert "后台导出" in dialog.status_label.text()
    dialog.set_export_error("render failed")
    dialog.set_language("en")
    assert "Export failed" in dialog.status_label.text()
    index = tmp_path / "bundle" / "index.html"
    index.parent.mkdir()
    index.write_text("<html></html>", encoding="utf-8")
    dialog.set_exported_paths(
        {"index": index},
        source_context={"source": "frame-A", "frame": 4},
        stale=True,
    )
    dialog.set_language("zh_CN")
    assert "此前" in dialog.status_label.text()
    assert "frame-A" in dialog.snapshot_label.text()
