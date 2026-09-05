from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6 import QtCore

from butterfly_saxs.ui import MainWindow
from butterfly_saxs.ui.models import ParameterTableModel


_ACTION_NAMES = (
    "open_project_action",
    "save_project_action",
    "open_image_action",
    "open_poni_action",
    "open_mask_action",
    "clear_mask_action",
    "export_evidence_action",
    "close_action",
    "chinese_action",
    "english_action",
)

_WIDGET_NAMES = (
    "parameter_table",
    "preview_button",
    "optimize_button",
    "cancel_button",
    "ignore_late_result_button",
    "auto_preview_check",
    "clear_mask_button",
    "roi_type_combo",
    "roi_x0",
    "roi_y0",
    "roi_x1",
    "roi_y1",
    "roi_cx",
    "roi_cy",
    "roi_rx",
    "roi_ry",
    "roi_angle",
    "apply_roi_button",
    "clear_roi_button",
    "reviewer_edit",
    "review_notes_edit",
    "accept_current_button",
    "reject_current_button",
    "restore_before_optimize_button",
    "snapshot_note_edit",
    "save_snapshot_button",
    "snapshot_combo",
    "restore_snapshot_button",
    "q_min_edit",
    "q_max_edit",
    "draw_axis_deg_spin",
    "ridge_method_combo",
    "n_angular_bins_spin",
    "n_ridge_angles_spin",
    "n_radial_bins_spin",
    "curvature_sigma_spin",
    "curvature_percentile_spin",
    "normal_step_spin",
    "max_pixels_spin",
    "batch_add_button",
    "batch_run_button",
    "batch_mode_combo",
    "batch_manifest_edit",
    "batch_checkpoint_edit",
    "batch_resume_check",
    "batch_output_edit",
    "evolution_y_label",
    "evolution_parameter_combo",
)

_KNOWN_PARAMETER_NAMES = (
    "a",
    "b",
    "cx",
    "cy",
    "axis_ratio",
    "theta",
    "theta_deg",
    "lobe_angle",
    "lobe_angle_deg",
    "angular_width",
    "angular_width_deg",
    "radial_sigma",
    "radial_gamma",
    "radial_fwhm",
    "radial_width",
    "eta",
    "amplitude",
    "amplitude_plus",
    "amplitude_minus",
    "background",
    "background_slope",
    "background_curvature",
    "background_amplitude",
    "background_width",
    "q_center",
    "q_major",
    "q_minor",
    "ellipticity",
    "intensity",
    "ridge_width",
)


def _assert_nonempty_fixed_tooltips(window: MainWindow) -> None:
    for name in _ACTION_NAMES:
        assert getattr(window, name).toolTip().strip(), name
    for name in _WIDGET_NAMES:
        assert getattr(window, name).toolTip().strip(), name

    assert window.project_menu.toolTipsVisible()
    assert window.language_menu.toolTipsVisible()
    assert window.project_menu.menuAction().toolTip().strip()
    assert window.language_menu.menuAction().toolTip().strip()

    for page in (
        window.refinement_page,
        window.measurements_page,
        window.batch_page,
        window.evolution_page,
    ):
        assert window.pages.tabToolTip(window.pages.indexOf(page)).strip()

    form_fields = (
        (window.fit_session_form, window.reviewer_edit),
        (window.fit_session_form, window.review_notes_edit),
        (window.fit_session_form, window.snapshot_save_row),
        (window.fit_session_form, window.snapshot_restore_row),
        (window.analysis_form, window.q_min_edit),
        (window.analysis_form, window.q_max_edit),
        (window.analysis_form, window.draw_axis_deg_spin),
        (window.analysis_form, window.ridge_method_combo),
        (window.analysis_form, window.n_angular_bins_spin),
        (window.analysis_form, window.n_ridge_angles_spin),
        (window.analysis_form, window.n_radial_bins_spin),
        (window.analysis_form, window.curvature_sigma_spin),
        (window.analysis_form, window.curvature_percentile_spin),
        (window.analysis_form, window.normal_step_spin),
        (window.analysis_form, window.max_pixels_spin),
        (window.batch_form, window.batch_mode_combo),
        (window.batch_form, window.batch_manifest_edit),
        (window.batch_form, window.batch_checkpoint_edit),
        (window.batch_form, window.batch_output_edit),
    )
    for form, field in form_fields:
        label = form.labelForField(field)
        assert label is not None
        assert label.toolTip().strip()

    assert window.roi_type_label.toolTip().strip()
    for label, spin in (*window._rectangle_roi_widgets, *window._ellipse_roi_widgets):
        assert label.toolTip().strip()
        assert spin.toolTip().strip()


def _assert_combo_item_tooltips(window: MainWindow) -> None:
    for combo in (
        window.roi_type_combo,
        window.ridge_method_combo,
        window.batch_mode_combo,
    ):
        assert combo.count() > 0
        for index in range(combo.count()):
            tooltip = combo.itemData(index, QtCore.Qt.ItemDataRole.ToolTipRole)
            assert isinstance(tooltip, str) and tooltip.strip(), (
                combo.objectName(),
                index,
            )


def test_all_fixed_controls_and_options_have_bilingual_tooltips(qtbot) -> None:
    window = MainWindow(engine=object(), auto_preview=False, language="zh_CN")
    qtbot.addWidget(window)

    _assert_nonempty_fixed_tooltips(window)
    _assert_combo_item_tooltips(window)
    chinese_preview = window.preview_button.toolTip()
    chinese_q_min = window.q_min_edit.toolTip()
    chinese_radial_peak = window.ridge_method_combo.itemData(
        window.ridge_method_combo.findData("radial_peak"),
        QtCore.Qt.ItemDataRole.ToolTipRole,
    )

    window.set_language("en", persist=False)

    _assert_nonempty_fixed_tooltips(window)
    _assert_combo_item_tooltips(window)
    assert window.preview_button.toolTip() != chinese_preview
    assert window.q_min_edit.toolTip() != chinese_q_min
    assert (
        window.ridge_method_combo.itemData(
            window.ridge_method_combo.findData("radial_peak"),
            QtCore.Qt.ItemDataRole.ToolTipRole,
        )
        != chinese_radial_peak
    )
    window.close()


def test_dynamic_combo_tooltips_retranslate_without_changing_project_state(qtbot) -> None:
    window = MainWindow(engine=object(), auto_preview=False, language="zh_CN")
    qtbot.addWidget(window)
    assert window.save_snapshot("基准参数")
    window.plot_evolution(
        [
            {"frame": 0, "rmse": 0.2, "parameters": {"q_major": 0.4}},
            {"frame": 1, "rmse": 0.1, "parameters": {"q_major": 0.5}},
        ]
    )

    assert window.snapshot_combo.count() == 1
    assert window.evolution_parameter_combo.count() > 0
    snapshot_tip_zh = window.snapshot_combo.itemData(
        0,
        QtCore.Qt.ItemDataRole.ToolTipRole,
    )
    evolution_tips_zh = [
        window.evolution_parameter_combo.itemData(
            index,
            QtCore.Qt.ItemDataRole.ToolTipRole,
        )
        for index in range(window.evolution_parameter_combo.count())
    ]
    assert "基准参数" in snapshot_tip_zh
    assert all(isinstance(value, str) and value.strip() for value in evolution_tips_zh)

    project_before = window.project_to_dict()
    parameters_before = window.parameter_model.parameter_dict()
    snapshot_before = window.snapshot_combo.currentData()
    evolution_before = window.evolution_parameter_combo.currentText()
    generation_before = window._generation.current

    window.set_language("en", persist=False)

    assert window.project_to_dict() == project_before
    assert window.parameter_model.parameter_dict() == parameters_before
    assert window.snapshot_combo.currentData() == snapshot_before
    assert window.evolution_parameter_combo.currentText() == evolution_before
    assert window._generation.current == generation_before
    assert (
        window.snapshot_combo.itemData(0, QtCore.Qt.ItemDataRole.ToolTipRole)
        != snapshot_tip_zh
    )
    assert [
        window.evolution_parameter_combo.itemData(
            index,
            QtCore.Qt.ItemDataRole.ToolTipRole,
        )
        for index in range(window.evolution_parameter_combo.count())
    ] != evolution_tips_zh
    window.close()


def test_parameter_table_headers_cells_unknown_fallback_and_unit_refresh(qtbot) -> None:
    parameters = {
        "a": {"value": 0.8, "unit": "unknown"},
        "theta_deg": {"value": 15.0, "unit": "degree"},
        "custom_peak_factor": {"value": 2.0, "unit": "a.u."},
    }
    window = MainWindow(
        engine=object(),
        parameters=parameters,
        auto_preview=False,
        language="zh_CN",
    )
    qtbot.addWidget(window)
    model: ParameterTableModel = window.parameter_model

    for column in range(model.columnCount()):
        tooltip = model.headerData(
            column,
            QtCore.Qt.Orientation.Horizontal,
            QtCore.Qt.ItemDataRole.ToolTipRole,
        )
        assert isinstance(tooltip, str) and tooltip.strip(), column
    for row in range(model.rowCount()):
        for column in range(model.columnCount()):
            tooltip = model.data(
                model.index(row, column),
                QtCore.Qt.ItemDataRole.ToolTipRole,
            )
            assert isinstance(tooltip, str) and tooltip.strip(), (row, column)

    theta_tip_zh = model.data(
        model.index(1, 1),
        QtCore.Qt.ItemDataRole.ToolTipRole,
    )
    unknown_tip_zh = model.data(
        model.index(2, 0),
        QtCore.Qt.ItemDataRole.ToolTipRole,
    )
    assert "不是唯一结构角" in theta_tip_zh
    assert "custom_peak_factor" in unknown_tip_zh
    assert "当前模型文档" in unknown_tip_zh

    window._refresh_q_parameter_units("nm^-1")
    a_tip_zh = model.data(
        model.index(0, 1),
        QtCore.Qt.ItemDataRole.ToolTipRole,
    )
    assert "nm^-1" in a_tip_zh

    window.set_language("en", persist=False)
    theta_tip_en = model.data(
        model.index(1, 1),
        QtCore.Qt.ItemDataRole.ToolTipRole,
    )
    unknown_tip_en = model.data(
        model.index(2, 0),
        QtCore.Qt.ItemDataRole.ToolTipRole,
    )
    assert theta_tip_en != theta_tip_zh
    assert "not a unique structural angle" in theta_tip_en
    assert "custom_peak_factor" in unknown_tip_en
    assert "active model documentation" in unknown_tip_en
    window.close()


def test_all_builtin_parameter_names_use_specific_bilingual_descriptions(qtbot) -> None:
    del qtbot
    parameters = {
        name: {"value": 1.0, "unit": "degree" if name.endswith("_deg") else ""}
        for name in _KNOWN_PARAMETER_NAMES
    }
    model = ParameterTableModel(parameters, language="zh_CN")

    for row, name in enumerate(_KNOWN_PARAMETER_NAMES):
        tooltip = model.data(
            model.index(row, 0),
            QtCore.Qt.ItemDataRole.ToolTipRole,
        )
        assert "没有其内置科学定义" not in tooltip, name

    model.set_language("en")
    for row, name in enumerate(_KNOWN_PARAMETER_NAMES):
        tooltip = model.data(
            model.index(row, 0),
            QtCore.Qt.ItemDataRole.ToolTipRole,
        )
        assert "no built-in scientific definition" not in tooltip, name
