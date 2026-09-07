"""Compact, bilingual controls for geometry assumptions (never fit parameters)."""

from __future__ import annotations

from typing import Any

from .qt_compat import QT_AVAILABLE, QtCore, QtWidgets, require_qt


TEXT = {
    "title": ("实空间片层", "Real-space lamellae"),
    "subtitle": ("从散射参数理解片层几何", "Explore lamellar geometry from scattering parameters"),
    "settings": ("示意设置", "Schematic settings"),
    "import": ("导入结果", "Open results"),
    "export": ("导出", "Export"),
    "publication": ("发表画板", "Publication artboard"),
    "export_one": ("当前帧 · 图片与来源", "Current frame · figures and provenance"),
    "export_sequence": ("序列 · PNG 与 GIF", "Sequence · PNG and GIF"),
    "cancel": ("取消任务", "Cancel task"),
    "file": ("打开结果 JSON…", "Open result JSON…"),
    "folder": ("打开结果文件夹…", "Open result folder…"),
    "mode": ("结构尺度", "Structure scale"),
    "single": ("单堆栈", "Single stack"),
    "multi": ("多堆栈", "Multiple stacks"),
    "period_source": ("周期来源", "Period source"),
    "radial": ("径向峰位", "Radial peak"),
    "ellipse": ("椭圆派生周期", "Ellipse-derived period"),
    "manual": ("手动示意", "Manual schematic"),
    "branch": ("显示分支", "Branches"),
    "all": ("全部分支", "All branches"),
    "thickness_ratio": ("厚度 / 周期", "Thickness / period"),
    "width_ratio": ("宽度 / 周期", "Width / period"),
    "depth_ratio": ("深度 / 周期", "Depth / period"),
    "layer_count": ("每堆栈层数", "Layers per stack"),
    "stack_count": ("堆栈数量", "Stack count"),
    "spread_deg": ("方向分散 (°)", "Orientation spread (°)"),
    "advanced": ("高级假设", "Advanced assumptions"),
    "palette": ("显示配色", "Display palette"),
    "blue_orange": ("蓝橙分支", "Blue / orange"),
    "grayscale": ("灰度出图", "Grayscale"),
    "lateral_shift_ratio": ("每层错移 / 周期", "Slip per layer / period"),
    "out_of_plane_deg": ("出平面角度 (°)", "Out-of-plane angle (°)"),
    "seed": ("随机种子", "Random seed"),
    "manual_period": ("设定周期", "Assumed period"),
    "manual_angle_deg": ("设定法向角度 (°)", "Assumed normal angle (°)"),
    "manual_unit": ("设定长度单位", "Assumed length unit"),
    "relative": ("相对单位", "Relative units"),
    "reset": ("恢复数据驱动默认值", "Restore data-driven defaults"),
    "undo": ("撤销", "Undo"),
    "redo": ("重做", "Redo"),
    "assumption_note": ("厚度、宽度、层数与分散程度是示意假设。", "Thickness, width, layer count and spread are schematic assumptions."),
    "saxs": ("SAXS · 观测与拟合", "SAXS · observation and fit"),
    "two_d": ("二维 · 样品面内投影", "2D · in-plane projection"),
    "three_d": ("三维 · 可拖动旋转", "3D · drag to orbit"),
    "focus": ("放大视图", "Focus view"),
    "unfocus": ("恢复布局", "Restore layout"),
    "reset_camera": ("复位视角", "Reset view"),
    "isometric": ("立体视角", "Isometric"),
    "front": ("正视", "Front"),
    "side": ("侧视", "Side"),
    "top": ("俯视", "Top"),
    "play": ("播放", "Play"),
    "pause": ("暂停", "Pause"),
    "previous": ("上一帧", "Previous frame"),
    "next": ("下一帧", "Next frame"),
    "sources": ("参数来源与假设", "Parameter sources and assumptions"),
    "empty": ("尚无可用参数。请先分析一帧、导入结果，或选择手动示意。", "No parameters yet. Analyze a frame, open results, or choose a manual schematic."),
    "stale": ("数据或分析设置已改变，请重新分析后查看。", "Data or analysis settings changed. Run analysis again."),
    "building": ("正在更新片层…", "Updating lamellae…"),
    "loading": ("正在读取结果…", "Reading results…"),
    "exporting": ("正在导出…", "Exporting…"),
    "cancelled": ("任务已取消", "Task cancelled"),
    "saved": ("已导出至", "Exported to"),
    "candidate": ("候选参数驱动示意", "Candidate-driven schematic"),
    "schematic": ("参数驱动示意", "Parameter-driven schematic"),
    "manual_status": ("手动假设示意", "Manual schematic assumptions"),
    "unavailable": ("当前帧缺少可用参数", "No usable parameters for this frame"),
    "no_image": ("此结果未附可读取的原始图像", "No readable original image in this result"),
    "trajectory": ("表观周期", "Apparent period"),
    "frame": ("帧序号", "Frame index"),
    "frame_count": ("帧", "frames"),
    "orthographic": ("正交投影 · z 为出平面方向", "Orthographic · z is out of plane"),
    "source_note": ("片层法向沿散射峰方向为示意假设；三维位置不代表唯一结构。", "Normal along the scattering peak is an assumption; 3D positions are not a unique structure."),
}


def tr(key: str, language: str = "zh_CN") -> str:
    pair = TEXT.get(key, (key, key))
    return pair[1 if str(language).startswith("en") else 0]


if QT_AVAILABLE:
    class LamellarControls(QtWidgets.QWidget):
        settingsChanged = QtCore.Signal(object)
        presentationChanged = QtCore.Signal(str)
        resetRequested = QtCore.Signal()

        def __init__(self, parent: Any = None, *, language: str = "zh_CN") -> None:
            super().__init__(parent)
            self.language = language
            self._loading = False
            self._values: dict[str, Any] = {}
            self.controls: dict[str, Any] = {}
            self.labels: dict[str, Any] = {}
            root = QtWidgets.QVBoxLayout(self)
            root.setContentsMargins(12, 8, 12, 8)
            self.form = QtWidgets.QFormLayout()
            self.form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
            self.form.setRowWrapPolicy(QtWidgets.QFormLayout.RowWrapPolicy.WrapLongRows)
            self.form.setVerticalSpacing(10)
            root.addLayout(self.form)
            self._combo("mode", ("single", "multi"))
            self._combo("period_source", ("radial", "ellipse", "manual"))
            self._combo("selected_branch", (("all", -1), ("A", 0), ("B", 1)), label="branch")
            self._spin("thickness_ratio", .01, .99, .05, decimals=2)
            self._spin("width_ratio", .1, 50., .5)
            self._spin("layer_count", 1, 64, 1, integer=True)
            self._spin("stack_count", 1, 128, 1, integer=True)
            self._spin("spread_deg", 0., 90., 1.)
            self.manual_group = QtWidgets.QGroupBox()
            self.manual_form = QtWidgets.QFormLayout(self.manual_group)
            self._spin("manual_period", .001, 1000000., .5, form=self.manual_form, decimals=3)
            self._spin("manual_angle_deg", -180., 180., 1., form=self.manual_form)
            self._combo("manual_unit", ("relative", "nm"), form=self.manual_form)
            root.addWidget(self.manual_group)
            self.advanced_toggle = QtWidgets.QToolButton()
            self.advanced_toggle.setCheckable(True)
            self.advanced_toggle.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
            self.advanced_toggle.setArrowType(QtCore.Qt.ArrowType.RightArrow)
            root.addWidget(self.advanced_toggle)
            self.advanced = QtWidgets.QWidget()
            advanced_form = QtWidgets.QFormLayout(self.advanced)
            advanced_form.setContentsMargins(0, 0, 0, 0)
            self._spin("depth_ratio", .1, 50., .5, form=advanced_form)
            self._spin("lateral_shift_ratio", -5., 5., .05, form=advanced_form, decimals=2)
            self._spin("out_of_plane_deg", -89., 89., 1., form=advanced_form)
            self._spin("seed", 0, 2147483647, 1, form=advanced_form, integer=True)
            self.palette_label = QtWidgets.QLabel()
            self.palette_combo = QtWidgets.QComboBox()
            self.palette_combo.addItem("", "blue_orange")
            self.palette_combo.addItem("", "grayscale")
            self.palette_combo.currentIndexChanged.connect(lambda: self.presentationChanged.emit(self.palette_combo.currentData()))
            advanced_form.addRow(self.palette_label, self.palette_combo)
            root.addWidget(self.advanced)
            self.advanced.hide()
            self.advanced_toggle.toggled.connect(self._toggle_advanced)
            self.note = QtWidgets.QLabel()
            self.note.setWordWrap(True)
            self.note.setStyleSheet("color: #65717c; font-size: 11px;")
            root.addWidget(self.note)
            self.reset_button = QtWidgets.QPushButton()
            self.reset_button.clicked.connect(self.resetRequested)
            root.addWidget(self.reset_button)
            root.addStretch(1)
            self.set_language(language)

        def _combo(self, key: str, options: Any, *, label: str | None = None, form: Any = None) -> None:
            widget = QtWidgets.QComboBox()
            widget.setObjectName("lamellar_" + key)
            widget.setMinimumWidth(112)
            for item in options:
                title, value = item if isinstance(item, tuple) else (item, item)
                widget.addItem(tr(title, self.language), value)
                widget.setItemData(widget.count() - 1, title, QtCore.Qt.ItemDataRole.UserRole + 1)
            text = QtWidgets.QLabel(tr(label or key, self.language))
            text.setBuddy(widget)
            (form or self.form).addRow(text, widget)
            self.controls[key], self.labels[key] = widget, (text, label or key)
            widget.currentIndexChanged.connect(self._changed)

        def _spin(self, key: str, low: float, high: float, step: float, *, form: Any = None,
                  integer: bool = False, decimals: int = 1) -> None:
            widget = QtWidgets.QSpinBox() if integer else QtWidgets.QDoubleSpinBox()
            widget.setObjectName("lamellar_" + key)
            if not integer:
                widget.setDecimals(decimals)
            widget.setRange(low, high)
            widget.setSingleStep(step)
            widget.setKeyboardTracking(False)
            text = QtWidgets.QLabel(tr(key, self.language))
            text.setBuddy(widget)
            (form or self.form).addRow(text, widget)
            self.controls[key], self.labels[key] = widget, (text, key)
            widget.valueChanged.connect(self._changed)

        def _toggle_advanced(self, checked: bool) -> None:
            self.advanced.setVisible(checked)
            self.advanced_toggle.setArrowType(QtCore.Qt.ArrowType.DownArrow if checked else QtCore.Qt.ArrowType.RightArrow)

        def set_settings(self, values: dict[str, Any]) -> None:
            self._loading = True
            try:
                self._values = dict(values)
                for key, widget in self.controls.items():
                    if key not in values:
                        continue
                    if isinstance(widget, QtWidgets.QComboBox):
                        widget.setCurrentIndex(max(0, widget.findData(values[key])))
                    else:
                        widget.setValue(values[key])
                self.manual_group.setVisible(values.get("period_source") == "manual")
                self.controls["stack_count"].setEnabled(values.get("mode") == "multi")
            finally:
                self._loading = False

        def _changed(self, *_: Any) -> None:
            if self._loading:
                return
            values = dict(self._values)
            for key, widget in self.controls.items():
                values[key] = widget.currentData() if isinstance(widget, QtWidgets.QComboBox) else widget.value()
            self.set_settings(values)
            self.settingsChanged.emit(values)

        def set_language(self, language: str) -> None:
            self.language = language
            for key, widget in self.controls.items():
                label, title = self.labels[key]
                label.setText(tr(title, language))
                widget.setAccessibleName(tr(title, language))
                widget.setToolTip(tr(title, language) + "\n" + tr("assumption_note", language))
                if isinstance(widget, QtWidgets.QComboBox):
                    for i in range(widget.count()):
                        widget.setItemText(i, tr(widget.itemData(i, QtCore.Qt.ItemDataRole.UserRole + 1), language))
            self.manual_group.setTitle(tr("manual", language))
            self.advanced_toggle.setText(tr("advanced", language))
            self.note.setText(tr("assumption_note", language))
            self.reset_button.setText(tr("reset", language))
            self.palette_label.setText(tr("palette", language))
            self.palette_combo.setAccessibleName(tr("palette", language))
            for i in range(self.palette_combo.count()):
                self.palette_combo.setItemText(i, tr(self.palette_combo.itemData(i), language))

        def set_palette(self, palette: str) -> None:
            self.palette_combo.blockSignals(True)
            self.palette_combo.setCurrentIndex(max(0, self.palette_combo.findData(palette)))
            self.palette_combo.blockSignals(False)

else:
    class LamellarControls:
        def __init__(self, *_: Any, **__: Any) -> None:
            require_qt()
