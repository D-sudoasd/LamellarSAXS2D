"""Publication artboard and export controller; no scientific fitting here."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import threading
from typing import Any

from ..publication_models import PublicationFigureSpec, PublicationStyle
from .qt_compat import QT_AVAILABLE, QtCore, QtGui, QtWidgets
from .workers import AnalysisWorker


if QT_AVAILABLE:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.figure import Figure

    class _PublicationCanvas(FigureCanvasQTAgg):
        """Fit the paper by DPI, keeping typography at its specified mm size."""

        def attach_artboard(self, figure):
            self.figure = figure
            figure.set_canvas(self)
            self._paper_inches = tuple(figure.get_size_inches())
            self._sync_paper_pixels()

        def _sync_paper_pixels(self):
            inches = getattr(self, "_paper_inches", None)
            if inches is None or self.width() < 1:
                return
            self.figure.set_dpi(self.width() * self.device_pixel_ratio / inches[0])
            self.figure.set_size_inches(inches, forward=False)
            self.draw_idle()

        def resizeEvent(self, event):  # noqa: N802
            if getattr(self, "_paper_inches", None) is None:
                super().resizeEvent(event)
            else:
                QtWidgets.QWidget.resizeEvent(self, event)
                self._sync_paper_pixels()

    class _ArtboardView(QtWidgets.QWidget):
        def __init__(self, canvas, parent=None):
            super().__init__(parent)
            self.canvas = canvas
            self.canvas.setParent(self)
            self.ratio = 183. / 126.
            self.setMinimumSize(320, 260)

        def resizeEvent(self, event):  # noqa: N802
            width = min(self.width(), self.height() * self.ratio)
            height = width / self.ratio
            self.canvas.setGeometry(round((self.width() - width) / 2), round((self.height() - height) / 2),
                                    max(1, round(width)), max(1, round(height)))
            super().resizeEvent(event)

        def set_ratio(self, ratio):
            self.ratio = float(ratio)
            event = QtGui.QResizeEvent(self.size(), self.size())
            self.resizeEvent(event)

    class PublicationDialog(QtWidgets.QDialog):
        documentChanged = QtCore.Signal(object)

        def __init__(self, parent=None, *, document=None, context_provider=None):
            super().__init__(parent)
            self.setWindowTitle("发表画板 · Publication artboard")
            self.setObjectName("publicationDialog")
            self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, False)
            self._provider = context_provider
            self._context: dict[str, Any] | None = None
            self._style = PublicationStyle()
            self._spec = PublicationFigureSpec()
            self._camera: dict[str, Any] | None = None
            self._camera_auto = True
            self._serial = 0
            self._event = threading.Event()
            self._worker = None
            self._rendering = False
            self._pending = False
            self._loading = False
            self._dragged: Any = None
            self._rendered = None
            self._figure = Figure(figsize=(8, 5), facecolor="white")
            self._pool = QtCore.QThreadPool(self)
            self._pool.setMaxThreadCount(1)
            self._preview_timer = QtCore.QTimer(self)
            self._preview_timer.setSingleShot(True)
            self._preview_timer.setInterval(180)
            self._preview_timer.timeout.connect(self.refresh_preview)
            self._build_ui()
            self.restore_document(document or {})
            screen = QtGui.QGuiApplication.primaryScreen()
            available = screen.availableGeometry() if screen else QtCore.QRect(0, 0, 1366, 900)
            self.resize(min(1240, available.width() - 40), min(830, available.height() - 50))

        def _build_ui(self):
            self.setStyleSheet(
                "#publicationDialog { background: #eef1f3; }"
                "#publicationDialog QPushButton, #publicationDialog QToolButton { padding: 6px 10px; }"
                "#publicationDialog QComboBox, #publicationDialog QDoubleSpinBox, #publicationDialog QSpinBox { min-height: 25px; }"
            )
            outer = QtWidgets.QVBoxLayout(self)
            outer.setContentsMargins(16, 12, 16, 12)
            header = QtWidgets.QHBoxLayout()
            title = QtWidgets.QLabel("发表画板")
            title.setStyleSheet("font-size: 22px; font-weight: 600; color: #2a3b44;")
            header.addWidget(title)
            subtitle = QtWidgets.QLabel("平直片层 · 精细三维 · 可编辑标注")
            subtitle.setStyleSheet("color: #627581;")
            header.addWidget(subtitle, 1)
            self.export_button = QtWidgets.QPushButton("导出发表图…")
            self.export_button.setObjectName("publicationExportButton")
            self.export_button.setStyleSheet("background: #426f81; color: white; border: 0; border-radius: 4px;")
            self.export_button.clicked.connect(self._choose_export)
            header.addWidget(self.export_button)
            outer.addLayout(header)
            body = QtWidgets.QHBoxLayout()
            sidebar = QtWidgets.QWidget()
            sidebar.setFixedWidth(240)
            controls = QtWidgets.QVBoxLayout(sidebar)
            controls.setContentsMargins(0, 5, 8, 0)
            self.form = QtWidgets.QFormLayout()
            self.form.setRowWrapPolicy(QtWidgets.QFormLayout.RowWrapPolicy.WrapLongRows)
            controls.addLayout(self.form)
            self.template_combo = self._combo("构图", (("结构主图", "structure"), ("SAXS＋结构", "evidence")))
            self.width_preset = self._combo("画板", (("双栏 · 183 mm", 183.), ("单栏 · 89 mm", 89.), ("自定义", 0.)))
            self.width_spin = self._spin("宽度 (mm)", 50., 300., 183.)
            self.height_spin = self._spin("高度 (mm)", 40., 250., 95.)
            self.dpi_combo = self._combo("分辨率", (("600 dpi", 600), ("450 dpi", 450), ("1200 dpi", 1200)))
            self.language_combo = self._combo("图中文字", (("English", "en"), ("中文", "zh_CN")))
            self.palette_combo = self._combo("配色", (("冷暖哑光", "editorial"), ("灰度", "grayscale")))
            self.background_combo = self._combo("背景", (("白色", "white"), ("透明", "transparent")))
            self.bevel_spin = self._spin("边缘精修", 0., .24, .12, step=.02, decimals=2)
            self.roughness_spin = self._spin("哑光程度", .35, 1., .72, step=.05, decimals=2)
            self.inset_check = QtWidgets.QCheckBox("显示层间关系特写")
            self.inset_check.setChecked(True)
            controls.addWidget(self.inset_check)
            self.labels_check = QtWidgets.QCheckBox("显示参数与法向标注")
            self.labels_check.setChecked(True)
            controls.addWidget(self.labels_check)
            self.ao_check = QtWidgets.QCheckBox("柔和层间阴影")
            self.ao_check.setChecked(True)
            controls.addWidget(self.ao_check)
            self.reset_labels_button = QtWidgets.QPushButton("恢复自动标注位置")
            self.reset_labels_button.clicked.connect(self.reset_annotation_positions)
            controls.addWidget(self.reset_labels_button)
            self.auto_camera_button = QtWidgets.QPushButton("平衡视角与自动取景")
            self.auto_camera_button.clicked.connect(self.reset_camera)
            controls.addWidget(self.auto_camera_button)
            self.use_camera_button = QtWidgets.QPushButton("采用工作台当前视角")
            self.use_camera_button.clicked.connect(self.use_workbench_camera)
            controls.addWidget(self.use_camera_button)
            note = QtWidgets.QLabel("可拖动图中标注调整位置。\n图内厚度与横向尺寸为示意设定；完整来源随导出保存。")
            note.setWordWrap(True)
            note.setStyleSheet("color: #647782; font-size: 11px; padding-top: 12px;")
            controls.addWidget(note)
            controls.addStretch(1)
            scroll = QtWidgets.QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFixedWidth(258)
            scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
            scroll.setWidget(sidebar)
            body.addWidget(scroll)
            paper = QtWidgets.QWidget()
            paper_layout = QtWidgets.QVBoxLayout(paper)
            paper_layout.setContentsMargins(10, 10, 10, 10)
            paper.setStyleSheet("background: white;")
            self.canvas = _PublicationCanvas(self._figure)
            self.canvas.setObjectName("publicationCanvas")
            self.canvas.mpl_connect("button_press_event", self._on_press)
            self.canvas.mpl_connect("motion_notify_event", self._on_motion)
            self.canvas.mpl_connect("button_release_event", self._on_release)
            self.artboard = _ArtboardView(self.canvas)
            paper_layout.addWidget(self.artboard, 1)
            self.size_label = QtWidgets.QLabel()
            self.size_label.setStyleSheet("color: #647782; font-size: 11px;")
            paper_layout.addWidget(self.size_label)
            body.addWidget(paper, 1)
            outer.addLayout(body, 1)
            footer = QtWidgets.QHBoxLayout()
            self.status = QtWidgets.QLabel("等待当前场景")
            self.status.setWordWrap(True)
            footer.addWidget(self.status, 1)
            self.cancel_button = QtWidgets.QPushButton("取消")
            self.cancel_button.clicked.connect(self.cancel)
            footer.addWidget(self.cancel_button)
            self.close_button = QtWidgets.QPushButton("关闭")
            self.close_button.clicked.connect(self.close)
            footer.addWidget(self.close_button)
            outer.addLayout(footer)
            self.width_preset.currentIndexChanged.connect(self._preset_changed)
            for widget in (self.template_combo, self.language_combo, self.dpi_combo,
                           self.palette_combo, self.background_combo):
                widget.currentIndexChanged.connect(self._controls_changed)
            for widget in (self.width_spin, self.height_spin, self.bevel_spin, self.roughness_spin):
                widget.valueChanged.connect(self._controls_changed)
            for widget in (self.inset_check, self.labels_check, self.ao_check):
                widget.toggled.connect(self._controls_changed)
            self._set_busy(False)

        def _combo(self, label, options):
            widget = QtWidgets.QComboBox()
            for text, value in options:
                widget.addItem(text, value)
            widget.setAccessibleName(label)
            self.form.addRow(label, widget)
            return widget

        def _spin(self, label, low, high, initial, *, step=1., decimals=1):
            widget = QtWidgets.QDoubleSpinBox()
            widget.setRange(low, high)
            widget.setDecimals(decimals)
            widget.setSingleStep(step)
            widget.setValue(initial)
            widget.setKeyboardTracking(False)
            widget.setAccessibleName(label)
            self.form.addRow(label, widget)
            return widget

        def _preset_changed(self):
            if self._loading:
                return
            width = self.width_preset.currentData()
            if width:
                self._loading = True
                self.width_spin.setValue(width)
                evidence = self.template_combo.currentData() == "evidence"
                self.height_spin.setValue((90. if evidence else 70.) if width == 89. else (126. if evidence else 95.))
                self._loading = False
                self._controls_changed()

        def _controls_changed(self, *_):
            if self._loading:
                return
            if self.template_combo.currentData() != self._spec.template:
                self._loading = True
                narrow = self.width_spin.value() < 120
                height = (90. if narrow else 126.) if self.template_combo.currentData() == "evidence" else (70. if narrow else 95.)
                self.height_spin.setValue(height)
                self._loading = False
            self._spec = replace(self._spec, template=self.template_combo.currentData(),
                                 width_mm=self.width_spin.value(), height_mm=self.height_spin.value(),
                                 dpi=self.dpi_combo.currentData(), language=self.language_combo.currentData(),
                                 background=self.background_combo.currentData(),
                                 show_inset=self.inset_check.isChecked(), annotations=self.labels_check.isChecked())
            self._style = PublicationStyle(palette=self.palette_combo.currentData(),
                                           bevel_fraction=self.bevel_spin.value(),
                                           roughness=self.roughness_spin.value(), ambient_occlusion=self.ao_check.isChecked())
            self.artboard.set_ratio(self._spec.width_mm / self._spec.height_mm)
            self.documentChanged.emit(self.document())
            self.request_preview()

        def document(self):
            return {"version": 1, "style": self._style.to_dict(), "figure": self._spec.to_dict(),
                    "camera": deepcopy(self._camera), "camera_auto": self._camera_auto}

        def restore_document(self, document):
            if document and document.get("version", 1) != 1:
                raise ValueError("Unsupported publication document version")
            self._style = PublicationStyle.from_mapping(document.get("style"))
            self._spec = PublicationFigureSpec.from_mapping(document.get("figure"))
            self._camera = deepcopy(document.get("camera"))
            self._camera_auto = bool(document.get("camera_auto", self._camera is None))
            self._loading = True
            try:
                for widget, value in ((self.template_combo, self._spec.template), (self.dpi_combo, self._spec.dpi),
                                      (self.language_combo, self._spec.language), (self.palette_combo, self._style.palette),
                                      (self.background_combo, self._spec.background)):
                    widget.setCurrentIndex(max(0, widget.findData(value)))
                self.width_spin.setValue(self._spec.width_mm)
                self.height_spin.setValue(self._spec.height_mm)
                preset = self.width_preset.findData(self._spec.width_mm)
                self.width_preset.setCurrentIndex(preset if preset >= 0 else self.width_preset.findData(0.))
                self.bevel_spin.setValue(self._style.bevel_fraction)
                self.roughness_spin.setValue(self._style.roughness)
                self.inset_check.setChecked(self._spec.show_inset)
                self.labels_check.setChecked(self._spec.annotations)
                self.ao_check.setChecked(self._style.ambient_occlusion)
            finally:
                self._loading = False
            self.artboard.set_ratio(self._spec.width_mm / self._spec.height_mm)

        def invalidate_source(self):
            self.cancel()
            self._context = None
            self._rendered = None
            self._figure.clear()
            self.canvas.draw_idle()
            self.export_button.setEnabled(False)
            self.status.setText("数据已改变，等待当前场景更新")

        def context_ready(self):
            if self.isVisible():
                self.request_preview()

        def request_preview(self):
            self._event.set()
            self._serial += 1
            self._pending = True
            self._rendered = None
            self.export_button.setEnabled(False)
            if self.isVisible():
                self._preview_timer.start()

        def _read_context(self):
            context = self._provider() if callable(self._provider) else self._context
            if not context or context.get("scene") is None or not context["scene"].metadata.get("available"):
                raise ValueError("当前没有可用场景，请先完成测量或选择手动示意。")
            return context

        def refresh_preview(self):
            if self._rendering or self._worker is not None:
                self._pending = True
                return
            token = self._serial
            self._event = threading.Event()
            event = self._event
            self._rendering = True
            self._pending = False
            self._set_busy(True)
            self.status.setText("正在绘制发表预览…")
            try:
                from .lamellar_3d import render_publication_scene
                from ..publication import render_publication_figure

                context = self._read_context()
                scene = deepcopy(context["scene"])
                preview_spec = replace(self._spec, dpi=140, quality="preview")
                width, height = preview_spec.main_pixel_size
                frame = render_publication_scene(scene, style=self._style, width=width, height=height,
                                                 camera=None if self._camera_auto else self._camera,
                                                 quality="preview", transparent=True, cancel_event=event)
                if token != self._serial or event.is_set():
                    return
                figure = render_publication_figure(scene, spec=preview_spec, style=self._style,
                                                   rendered_scene=frame, observed=context.get("observed"),
                                                   qx=context.get("qx"), qy=context.get("qy"))
                self._replace_figure(figure)
                self._rendered = frame
                self._context = context
                self._camera = deepcopy(frame.camera)
                self.status.setText("预览已更新 · 可拖动文字调整位置")
                self.size_label.setText(f"{self._spec.width_mm:g} × {self._spec.height_mm:g} mm  ·  {self._spec.dpi} dpi  ·  6.5 pt text")
                self.documentChanged.emit(self.document())
            except Exception as exc:
                if token == self._serial:
                    self.status.setText("预览未完成：" + str(exc))
                    self._rendered = None
            finally:
                self._rendering = False
                self._set_busy(False)
                if self._pending:
                    self._preview_timer.start()

        def _replace_figure(self, figure):
            self._figure = figure
            self.canvas.attach_artboard(figure)
            self.canvas.mpl_connect("button_press_event", self._on_press)
            self.canvas.mpl_connect("motion_notify_event", self._on_motion)
            self.canvas.mpl_connect("button_release_event", self._on_release)
            self.canvas.draw_idle()

        def reset_camera(self):
            self._camera = None
            self._camera_auto = True
            self.documentChanged.emit(self.document())
            self.request_preview()

        def use_workbench_camera(self):
            try:
                self._camera = deepcopy(self._read_context().get("camera"))
                self._camera_auto = False
                self.request_preview()
            except ValueError as exc:
                self.status.setText(str(exc))

        def reset_annotation_positions(self):
            self._spec = replace(self._spec, annotation_positions={})
            self.documentChanged.emit(self.document())
            self.request_preview()

        def _on_press(self, event):
            if self._rendering or self._worker or event.button != 1:
                return
            artists = getattr(self._figure, "publication_annotation_artists", {})
            for name, artist in artists.items():
                if artist.contains(event)[0]:
                    self._dragged = (name, artist)
                    break

        def _on_motion(self, event):
            if self._dragged is None or event.x is None or event.y is None:
                return
            _, artist = self._dragged
            position = self._figure.transFigure.inverted().transform((event.x, event.y))
            position = [min(.98, max(.02, float(v))) for v in position]
            if hasattr(artist, "xyann"):
                artist.xyann = tuple(position)
            else:
                artist.set_transform(self._figure.transFigure)
                artist.set_position(position)
            self.canvas.draw_idle()

        def _on_release(self, event):
            if self._dragged is None:
                return
            name, artist = self._dragged
            position = artist.xyann if hasattr(artist, "xyann") else artist.get_position()
            values = dict(self._spec.annotation_positions)
            values[name] = [float(v) for v in position]
            self._spec = replace(self._spec, annotation_positions=values)
            self._dragged = None
            self.documentChanged.emit(self.document())

        def _choose_export(self):
            directory = QtWidgets.QFileDialog.getExistingDirectory(self, "发表图导出目录")
            if not directory:
                return
            target = Path(directory) / "publication-figure"
            index = 2
            while target.exists():
                target = Path(directory) / f"publication-figure-{index}"
                index += 1
            self.export_to(target)

        def export_to(self, destination):
            if self._rendering or self._worker:
                raise RuntimeError("Publication renderer is busy")
            token = self._serial = self._serial + 1
            self._event = threading.Event()
            event = self._event
            self._rendering = True
            self._set_busy(True)
            self.status.setText("正在生成原始高分辨率三维图…")
            try:
                from .lamellar_3d import render_publication_scene

                context = self._read_context()
                scene = deepcopy(context["scene"])
                spec, style = self._spec, self._style
                width, height = spec.main_pixel_size
                frame = render_publication_scene(scene, style=style, width=width, height=height,
                                                 camera=self._camera, quality="publication", transparent=True,
                                                 cancel_event=event)
                if event.is_set() or token != self._serial:
                    raise RuntimeError("Export cancelled")
                self.status.setText("正在保存矢量文字、图像与来源…")

                def job(**_):
                    from ..publication import export_publication_figure

                    return export_publication_figure(scene, destination, spec=spec, style=style,
                        rendered_scene=frame, camera=frame.camera, observed=context.get("observed"),
                        qx=context.get("qx"), qy=context.get("qy"), cancel_event=event,
                        progress=lambda current, total=11: worker.report_progress((current, total)))

                worker = AnalysisWorker(job, generation=token, kind="publication_export")
                worker.signals.finished.connect(self._export_finished)
                worker.signals.error.connect(self._export_failed)
                worker.signals.progress.connect(self._export_progress)
                self._worker = worker
                self._pool.start(worker)
            except Exception as exc:
                if event.is_set():
                    self.status.setText("已取消")
                elif token == self._serial:
                    self.status.setText("导出未完成：" + str(exc))
            finally:
                self._rendering = False
                self._set_busy(self._worker is not None)
                if self._worker is None and self._pending:
                    self._preview_timer.start()

        @QtCore.Slot(int, str, object)
        def _export_progress(self, token, kind, value):
            if token == self._serial:
                current, total = value
                self.status.setText(f"正在保存发表图与来源… {current} / {total}")

        @QtCore.Slot(int, str, object)
        def _export_finished(self, token, kind, result):
            self._worker = None
            if token == self._serial:
                self.status.setText("已导出：" + str(Path(next(iter(result.values()))).parent))
            self._set_busy(False)
            if self._pending and self.isVisible():
                self._preview_timer.start()

        @QtCore.Slot(int, str, object)
        def _export_failed(self, token, kind, error):
            self._worker = None
            if token == self._serial:
                self.status.setText("导出未完成：" + str(error))
            self._set_busy(False)
            if self._pending and self.isVisible():
                self._preview_timer.start()

        def _set_busy(self, busy):
            self.cancel_button.setVisible(busy)
            self.export_button.setEnabled(not busy and self._rendered is not None)

        def cancel(self):
            self._event.set()
            self._serial += 1
            self._pending = False
            self._preview_timer.stop()
            self.status.setText("已取消")

        def jobs_running(self):
            return self._rendering or self._worker is not None

        def showEvent(self, event):  # noqa: N802
            super().showEvent(event)
            self.request_preview()

        def closeEvent(self, event):  # noqa: N802
            self.cancel()
            self.documentChanged.emit(self.document())
            event.accept()

else:
    class PublicationDialog:
        def __init__(self, *args, **kwargs):
            from .qt_compat import require_qt

            require_qt()
