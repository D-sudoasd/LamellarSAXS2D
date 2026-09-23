"""Settings dialog for the detached butterfly measurement-figure export.

The dialog deliberately contains no rendering code.  It gives the user one
place to choose the paper width, raster resolution and a parent directory,
then emits a small configuration mapping to :class:`MainWindow`.  The actual
figure export remains on the worker pool and keeps the frozen-snapshot and
generation guards owned by the main window.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .butterfly_figure_export import new_figure_export_target
from .qt_compat import QT_AVAILABLE, QtCore, QtGui, QtWidgets, require_qt


_WIDTHS_MM = (89.0, 183.0)
_DPIS = (300, 600, 1200)


def _figure_height_mm(width_mm: float) -> float:
    """Return the fixed physical height used by the measurement exporter."""

    return 150.0 if float(width_mm) == 89.0 else 94.0


def _pixel_size(width_mm: float, dpi: int) -> tuple[int, int]:
    """Return the exporter-compatible raster dimensions."""

    height_mm = _figure_height_mm(width_mm)
    return (
        int(float(width_mm) / 25.4 * int(dpi)),
        int(height_mm / 25.4 * int(dpi)),
    )


if QT_AVAILABLE:

    class _PaperRatioPreview(QtWidgets.QWidget):
        """Small static paper-ratio cue; this is not a rendered figure preview."""

        def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
            super().__init__(parent)
            self._width_mm = 183.0
            self._height_mm = _figure_height_mm(self._width_mm)
            self.setMinimumHeight(76)
            self.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Expanding,
                QtWidgets.QSizePolicy.Policy.Fixed,
            )
            self.setAccessibleName("Paper aspect ratio preview")
            self.setAccessibleDescription(
                "Static paper ratio cue; it is not a rendered measurement figure."
            )

        def set_width_mm(self, width_mm: float) -> None:
            self._width_mm = float(width_mm)
            self._height_mm = _figure_height_mm(self._width_mm)
            self.update()

        def paintEvent(self, event: Any) -> None:  # noqa: N802
            del event
            painter = QtGui.QPainter(self)
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
            painter.fillRect(self.rect(), QtGui.QColor("#f6f8fa"))
            ratio = self._width_mm / max(self._height_mm, 1.0)
            margin = 10.0
            available_width = max(1.0, float(self.width()) - 2.0 * margin)
            available_height = max(1.0, float(self.height()) - 2.0 * margin)
            width = min(available_width, available_height * ratio)
            height = width / ratio
            x = (float(self.width()) - width) / 2.0
            y = (float(self.height()) - height) / 2.0
            paper = QtCore.QRectF(x, y, width, height)
            painter.setPen(QtGui.QPen(QtGui.QColor("#667784"), 1.0))
            painter.setBrush(QtGui.QBrush(QtGui.QColor("#ffffff")))
            painter.drawRect(paper)
            painter.setPen(QtGui.QColor("#536572"))
            painter.drawText(
                paper,
                int(QtCore.Qt.AlignmentFlag.AlignCenter),
                f"{self._width_mm:g} × {self._height_mm:g} mm",
            )
            painter.end()


    class FigureExportDialog(QtWidgets.QDialog):
        """Single-step settings and status window for a figure export.

        ``startRequested`` is emitted with ``parent``, ``width_mm`` and
        ``dpi``.  The main window owns the worker lifecycle and calls
        :meth:`set_exported_paths`, :meth:`set_export_error` or
        :meth:`set_export_cancelled` when that lifecycle completes.
        """

        startRequested = QtCore.Signal(object)
        cancelRequested = QtCore.Signal()
        openRequested = QtCore.Signal(object)

        def __init__(
            self,
            parent: QtWidgets.QWidget | None = None,
            *,
            language: str = "en",
            q_unit: str = "unknown",
            stage: str = "trace",
            quality_status: str = "unknown",
            measurement_status: str = "unknown",
            scientific_status: str | None = None,
            scientific_acceptance: str = "not_assessed",
            has_qx: bool = True,
            has_qy: bool = True,
            default_parent: str | Path | None = None,
        ) -> None:
            super().__init__(parent)
            self.setObjectName("figureExportDialog")
            self.setWindowModality(QtCore.Qt.WindowModality.NonModal)
            self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, False)
            self.setSizeGripEnabled(True)
            self._language = str(language or "en")
            self._q_unit = str(q_unit or "unknown")
            self._stage = str(stage or "trace")
            self._quality_status = str(quality_status or "unknown")
            self._measurement_status = str(measurement_status or "unknown")
            self._scientific_status = str(
                scientific_status or scientific_acceptance or "not_assessed"
            )
            self._scientific_acceptance = str(
                scientific_acceptance or "not_assessed"
            )
            self._has_qx = bool(has_qx)
            self._has_qy = bool(has_qy)
            self._default_parent = (
                Path(default_parent).expanduser().resolve()
                if default_parent
                else None
            )
            self._running = False
            self._exported_paths: dict[str, Path] = {}
            self._last_error: str | None = None
            self._status_key = "idle"
            self._status_detail = ""
            self._status_error = False
            self._export_stale = False
            self._export_source_context: dict[str, Any] = {}
            self._build_ui()
            self._retranslate_ui()
            if self._default_parent is not None:
                self.parent_dir_edit.setText(str(self._default_parent))
            else:
                self._update_target_preview()
            self._resize_for_screen()

        # ---- public state -------------------------------------------------

        @property
        def language(self) -> str:
            return self._language

        @property
        def is_running(self) -> bool:
            return self._running

        @property
        def exported_paths(self) -> dict[str, Path]:
            return dict(self._exported_paths)

        @property
        def has_exported_result(self) -> bool:
            return bool(self._exported_paths)

        @property
        def export_is_stale(self) -> bool:
            return bool(self._export_stale)

        def settings(self) -> dict[str, Any]:
            """Return validated user settings without creating an output dir."""

            parent_text = self.parent_dir_edit.text().strip()
            parent = Path(parent_text).expanduser().resolve()
            if not parent_text:
                raise ValueError(self._text("Choose an output parent folder.", "请选择输出父目录。"))
            if not parent.is_dir():
                raise ValueError(
                    self._text(
                        "The output parent folder does not exist.",
                        "输出父目录不存在。",
                    )
                )
            width_mm = float(self.width_combo.currentData())
            dpi = int(self.dpi_combo.currentData())
            return {
                "parent": parent,
                "width_mm": width_mm,
                "dpi": dpi,
            }

        def set_language(self, language: str) -> None:
            self._language = str(language or "en")
            self._retranslate_ui()

        def set_measurement_context(self, context: Mapping[str, Any]) -> None:
            """Refresh display-only source state after the page changed."""

            values = dict(context)
            self._q_unit = str(values.get("q_unit", self._q_unit) or "unknown")
            self._stage = str(values.get("stage", self._stage) or "trace")
            self._quality_status = str(
                values.get("quality_status", self._quality_status) or "unknown"
            )
            self._measurement_status = str(
                values.get("measurement_status", self._measurement_status) or "unknown"
            )
            self._scientific_status = str(
                values.get("scientific_status", self._scientific_status)
                or "not_assessed"
            )
            self._has_qx = bool(values.get("has_qx", self._has_qx))
            self._has_qy = bool(values.get("has_qy", self._has_qy))
            self._retranslate_ui()

        def set_context_changed(self) -> None:
            """Stop a just-requested start when the source changed meanwhile."""

            self._running = False
            self.set_export_running(False)
            self._set_status(key="context_changed", error=True)

        def set_export_running(self, running: bool = True) -> None:
            self._running = bool(running)
            for widget in (
                self.parent_dir_edit,
                self.browse_button,
                self.width_combo,
                self.dpi_combo,
                self.start_button,
            ):
                widget.setEnabled(not self._running)
            self.cancel_button.setEnabled(True)
            if self._running:
                self.open_button.setEnabled(False)
                self._set_status(key="running")
                self.cancel_button.setText(self._text("Cancel export", "取消导出"))
            else:
                self.cancel_button.setText(self._text("Close", "关闭"))

        def set_cancelling(self) -> None:
            self._running = True
            self.start_button.setEnabled(False)
            self.cancel_button.setEnabled(False)
            self._set_status(key="cancelling")

        def set_exported_paths(
            self,
            paths: Mapping[str, str | Path],
            *,
            source_context: Mapping[str, Any] | None = None,
            stale: bool = False,
        ) -> None:
            self._running = False
            self._exported_paths = {
                str(name): Path(path).expanduser().resolve()
                for name, path in paths.items()
            }
            self._export_stale = bool(stale)
            self._export_source_context = dict(source_context or {})
            self.set_export_running(False)
            index = self._exported_paths.get("index")
            if index is None:
                manifest = self._exported_paths.get("manifest")
                index = manifest.parent / "index.html" if manifest is not None else None
            has_index = bool(index is not None and index.is_file())
            self.open_button.setEnabled(has_index)
            self.open_button.setProperty("packagePath", str(index) if index else "")
            self.open_button.setProperty("snapshotStale", self._export_stale)
            self._render_snapshot_context()
            self._set_status(key="stale_complete" if self._export_stale else "complete")

        def set_export_error(self, error: Any) -> None:
            self._last_error = str(error)
            self._running = False
            self.set_export_running(False)
            self._update_target_preview()
            self._set_status(key="error", detail=self._last_error, error=True)

        def set_export_cancelled(self) -> None:
            self._running = False
            self.set_export_running(False)
            self._set_status(key="cancelled")

        def mark_export_stale(self, current_context: Mapping[str, Any] | None = None) -> None:
            """Mark a completed bundle as belonging to an older page snapshot."""

            if not self._exported_paths:
                return
            self._export_stale = True
            self.open_button.setProperty("snapshotStale", True)
            self._render_snapshot_context(current_context=current_context)
            self._set_status(key="stale_complete")

        # ---- construction -------------------------------------------------

        def _build_ui(self) -> None:
            self.setStyleSheet(
                "#figureExportDialog QPushButton, #figureExportDialog QComboBox, "
                "#figureExportDialog QLineEdit { min-height: 28px; }"
            )
            outer = QtWidgets.QVBoxLayout(self)
            outer.setContentsMargins(12, 10, 12, 10)
            outer.setSpacing(8)

            self.title_label = QtWidgets.QLabel()
            self.title_label.setObjectName("figureExportTitle")
            self.title_label.setStyleSheet("font-size: 17px; font-weight: 600;")
            outer.addWidget(self.title_label)
            self.subtitle_label = QtWidgets.QLabel()
            self.subtitle_label.setObjectName("figureExportSubtitle")
            self.subtitle_label.setWordWrap(True)
            outer.addWidget(self.subtitle_label)

            scroll = QtWidgets.QScrollArea(self)
            scroll.setObjectName("figureExportSettingsScroll")
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
            scroll.setHorizontalScrollBarPolicy(
                QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
            )
            content = QtWidgets.QWidget(scroll)
            content.setObjectName("figureExportSettingsContent")
            content_layout = QtWidgets.QVBoxLayout(content)
            content_layout.setContentsMargins(2, 2, 2, 2)
            content_layout.setSpacing(8)

            self.form = QtWidgets.QFormLayout()
            self.form.setFieldGrowthPolicy(
                QtWidgets.QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow
            )
            self.form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
            content_layout.addLayout(self.form)

            self.parent_dir_edit = QtWidgets.QLineEdit(content)
            self.parent_dir_edit.setObjectName("figureExportParentDirectory")
            self.parent_dir_edit.setClearButtonEnabled(True)
            self.parent_dir_edit.textChanged.connect(self._update_target_preview)
            self.browse_button = QtWidgets.QPushButton(content)
            self.browse_button.setObjectName("figureExportBrowseButton")
            self.browse_button.clicked.connect(self._browse_parent)
            parent_row = QtWidgets.QWidget(content)
            parent_layout = QtWidgets.QHBoxLayout(parent_row)
            parent_layout.setContentsMargins(0, 0, 0, 0)
            parent_layout.addWidget(self.parent_dir_edit, 1)
            parent_layout.addWidget(self.browse_button)
            self._parent_label = QtWidgets.QLabel(parent_row)
            self._parent_label.setBuddy(self.parent_dir_edit)
            self.form.addRow(self._parent_label, parent_row)

            self.target_preview_label = QtWidgets.QLabel(content)
            self.target_preview_label.setObjectName("figureExportTargetPreview")
            self.target_preview_label.setWordWrap(True)
            self.target_preview_label.setTextInteractionFlags(
                QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
            )
            self._target_label = QtWidgets.QLabel(content)
            self._target_label.setBuddy(self.parent_dir_edit)
            self.form.addRow(self._target_label, self.target_preview_label)

            self.width_combo = QtWidgets.QComboBox(content)
            self.width_combo.setObjectName("figureExportWidthCombo")
            self.width_combo.addItem("", 89.0)
            self.width_combo.addItem("", 183.0)
            self.width_combo.setCurrentIndex(1)
            self.width_combo.currentIndexChanged.connect(self._controls_changed)
            self._width_label = QtWidgets.QLabel(content)
            self._width_label.setBuddy(self.width_combo)
            self.form.addRow(self._width_label, self.width_combo)
            # Compatibility alias for callers that call this a preset control.
            self.width_preset = self.width_combo

            self.dpi_combo = QtWidgets.QComboBox(content)
            self.dpi_combo.setObjectName("figureExportDpiCombo")
            for dpi in _DPIS:
                self.dpi_combo.addItem(f"{dpi} dpi", dpi)
            self.dpi_combo.setCurrentIndex(1)
            self.dpi_combo.currentIndexChanged.connect(self._controls_changed)
            self._dpi_label = QtWidgets.QLabel(content)
            self._dpi_label.setBuddy(self.dpi_combo)
            self.form.addRow(self._dpi_label, self.dpi_combo)

            self.size_label = QtWidgets.QLabel(content)
            self.size_label.setObjectName("figureExportSizeLabel")
            self.size_label.setWordWrap(True)
            self.form.addRow(QtWidgets.QLabel(content), self.size_label)
            self._size_label = self.form.labelForField(self.size_label)

            content_layout.addWidget(self._make_preview_group(content))

            self.state_group = QtWidgets.QGroupBox(content)
            self.state_group.setObjectName("figureExportStateGroup")
            state_layout = QtWidgets.QFormLayout(self.state_group)
            self.q_unit_label = QtWidgets.QLabel()
            self.stage_label = QtWidgets.QLabel()
            self.quality_label = QtWidgets.QLabel()
            self.scientific_label = QtWidgets.QLabel()
            self.coordinates_label = QtWidgets.QLabel()
            for value in (
                self.q_unit_label,
                self.stage_label,
                self.quality_label,
                self.scientific_label,
                self.coordinates_label,
            ):
                value.setWordWrap(True)
            self._q_unit_caption = QtWidgets.QLabel(self.state_group)
            self._stage_caption = QtWidgets.QLabel(self.state_group)
            self._quality_caption = QtWidgets.QLabel(self.state_group)
            self._scientific_caption = QtWidgets.QLabel(self.state_group)
            self._coordinates_caption = QtWidgets.QLabel(self.state_group)
            self._snapshot_caption = QtWidgets.QLabel(self.state_group)
            self.snapshot_label = QtWidgets.QLabel()
            state_layout.addRow(self._q_unit_caption, self.q_unit_label)
            state_layout.addRow(self._stage_caption, self.stage_label)
            state_layout.addRow(self._quality_caption, self.quality_label)
            state_layout.addRow(self._scientific_caption, self.scientific_label)
            state_layout.addRow(self._coordinates_caption, self.coordinates_label)
            state_layout.addRow(self._snapshot_caption, self.snapshot_label)
            self.snapshot_label.setObjectName("figureExportSnapshotLabel")
            self.snapshot_label.setWordWrap(True)
            content_layout.addWidget(self.state_group)

            self.notes_label = QtWidgets.QLabel(content)
            self.notes_label.setObjectName("figureExportNotes")
            self.notes_label.setWordWrap(True)
            self.notes_label.setTextInteractionFlags(
                QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
            )
            content_layout.addWidget(self.notes_label)
            content_layout.addStretch(1)
            scroll.setWidget(content)
            outer.addWidget(scroll, 1)

            self.status_label = QtWidgets.QLabel()
            self.status_label.setObjectName("figureExportStatus")
            self.status_label.setWordWrap(True)
            outer.addWidget(self.status_label)
            buttons = QtWidgets.QHBoxLayout()
            buttons.addStretch(1)
            self.open_button = QtWidgets.QPushButton()
            self.open_button.setObjectName("openFigureExportButton")
            self.open_button.setEnabled(False)
            self.open_button.clicked.connect(self._open_exported_package)
            buttons.addWidget(self.open_button)
            self.cancel_button = QtWidgets.QPushButton()
            self.cancel_button.setObjectName("cancelFigureExportButton")
            self.cancel_button.clicked.connect(self._cancel_or_close)
            buttons.addWidget(self.cancel_button)
            self.start_button = QtWidgets.QPushButton()
            self.start_button.setObjectName("startFigureExportButton")
            self.start_button.setDefault(True)
            self.start_button.clicked.connect(self._request_start)
            buttons.addWidget(self.start_button)
            outer.addLayout(buttons)

        def _make_preview_group(self, parent: QtWidgets.QWidget) -> QtWidgets.QGroupBox:
            group = QtWidgets.QGroupBox(parent)
            group.setObjectName("figureExportPaperGroup")
            layout = QtWidgets.QVBoxLayout(group)
            self.paper_preview = _PaperRatioPreview(group)
            layout.addWidget(self.paper_preview)
            self._preview_caption = QtWidgets.QLabel(group)
            self._preview_caption.setWordWrap(True)
            layout.addWidget(self._preview_caption)
            return group

        # ---- interaction --------------------------------------------------

        def _text(self, english: str, chinese: str) -> str:
            return english if self._language.lower().startswith("en") else chinese

        def _browse_parent(self) -> None:
            chosen = QtWidgets.QFileDialog.getExistingDirectory(
                self,
                self._text("Choose output parent folder", "选择输出父目录"),
                self.parent_dir_edit.text().strip(),
            )
            if chosen:
                self.parent_dir_edit.setText(chosen)

        def _controls_changed(self, *_: Any) -> None:
            width_mm = float(self.width_combo.currentData())
            dpi = int(self.dpi_combo.currentData())
            self.paper_preview.set_width_mm(width_mm)
            pixels = _pixel_size(width_mm, dpi)
            self.size_label.setText(
                self._text(
                    f"Physical size: {width_mm:g} × {_figure_height_mm(width_mm):g} mm\n"
                    f"Raster size: {pixels[0]:,} × {pixels[1]:,} px at {dpi} dpi",
                    f"物理尺寸：{width_mm:g} × {_figure_height_mm(width_mm):g} mm\n"
                    f"栅格尺寸：{pixels[0]:,} × {pixels[1]:,} px（{dpi} dpi）",
                )
            )
            self._update_target_preview()

        def _update_target_preview(self, *_: Any) -> None:
            parent_text = self.parent_dir_edit.text().strip()
            if not parent_text:
                self.target_preview_label.setText(
                    self._text(
                        "A new subfolder will be created after you choose a parent folder.",
                        "选择父目录后将创建新的子目录。",
                    )
                )
                return
            parent = Path(parent_text).expanduser()
            try:
                target = new_figure_export_target(parent)
            except (OSError, RuntimeError, ValueError):
                target = parent / "butterfly-figure"
            self.target_preview_label.setText(
                self._text(
                    f"New non-overwriting bundle: {target}",
                    f"新建且不覆盖已有内容的图包目录：{target}",
                )
            )

        def _request_start(self) -> None:
            try:
                values = self.settings()
            except (OSError, TypeError, ValueError) as exc:
                self._set_status(key="error", detail=str(exc), error=True)
                self.parent_dir_edit.setFocus()
                return
            self.set_export_running(True)
            self.startRequested.emit(values)

        def _cancel_or_close(self) -> None:
            if self._running:
                self.set_cancelling()
                self.cancelRequested.emit()
                return
            self.reject()

        def _open_exported_package(self) -> None:
            path = self.open_button.property("packagePath")
            if not path:
                return
            self.openRequested.emit(Path(str(path)))

        def closeEvent(self, event: Any) -> None:  # noqa: N802
            if self._running:
                self.set_cancelling()
                self.cancelRequested.emit()
                event.ignore()
                return
            event.accept()

        def _set_status(
            self,
            text: str | None = None,
            *,
            key: str | None = None,
            detail: Any = "",
            error: bool = False,
        ) -> None:
            """Store a language-neutral status and render it in the active locale."""

            if key is not None:
                self._status_key = str(key)
            if text is not None:
                self._status_detail = str(text)
            elif detail not in (None, ""):
                self._status_detail = str(detail)
            elif key in {"running", "cancelling", "complete", "stale_complete", "cancelled"}:
                self._status_detail = ""
            self._status_error = bool(error)
            self._render_status()

        def _render_status(self) -> None:
            key = self._status_key
            detail = self._status_detail
            if key == "running":
                text = self._text("Export is running…", "正在后台导出…")
            elif key == "cancelling":
                text = self._text(
                    "Cancelling; waiting for the worker…",
                    "正在取消，等待后台任务结束…",
                )
            elif key == "complete":
                text = self._text(
                    "Export complete. Scientific status is recorded only; acceptance is not inferred.",
                    "导出完成；科学状态仅作记录，不据此认定科学接纳。",
                )
            elif key == "stale_complete":
                text = self._text(
                    "Export complete for a previous frame/settings snapshot; the bundle is not tagged as the current measurement.",
                    "导出完成，但图包属于此前的帧/设置快照；不会标记为当前测量结果。",
                )
            elif key == "cancelled":
                text = self._text("Export cancelled.", "导出已取消。")
            elif key == "context_changed":
                text = self._text(
                    "The measurement changed; settings were refreshed. Review them and start again.",
                    "测量数据已变化；设置已刷新。请检查后再次开始导出。",
                )
            elif key == "error":
                text = self._text("Export failed: ", "导出失败：") + detail
            else:
                text = detail
            self.status_label.setText(str(text))
            self.status_label.setStyleSheet(
                "color: #9b1c1c;" if self._status_error else "color: #3b4b57;"
            )
            self.status_label.setToolTip(str(text))

        def _render_snapshot_context(
            self, *, current_context: Mapping[str, Any] | None = None
        ) -> None:
            context = dict(self._export_source_context)
            source = context.get("source") or context.get("frame") or "unknown"
            frame = context.get("frame")
            dataset = context.get("dataset")
            parts = [f"source={source}"]
            if frame not in (None, "") and str(frame) != str(source):
                parts.append(f"frame={frame}")
            if dataset not in (None, ""):
                parts.append(f"dataset={dataset}")
            if context.get("q_unit") not in (None, ""):
                parts.append(f"q_unit={context['q_unit']}")
            analysis = context.get("analysis")
            if isinstance(analysis, Mapping):
                butterfly = analysis.get("butterfly")
                stage = (
                    butterfly.get("stage")
                    if isinstance(butterfly, Mapping)
                    else analysis.get("stage")
                )
                if stage not in (None, ""):
                    parts.append(f"stage={stage}")
            if context.get("figure_width_mm") not in (None, ""):
                parts.append(f"width={context['figure_width_mm']} mm")
            if context.get("figure_dpi") not in (None, ""):
                parts.append(f"dpi={context['figure_dpi']}")
            snapshot = "; ".join(str(item) for item in parts)
            if self._export_stale:
                prefix = self._text("Previous export snapshot: ", "此前导出快照：")
                current = ""
                if current_context:
                    current_source = current_context.get("source") or current_context.get("frame")
                    if current_source not in (None, ""):
                        current = self._text(
                            f"; current page source={current_source}",
                            f"；当前页面来源={current_source}",
                        )
                text = prefix + snapshot + current
            else:
                text = self._text("Export snapshot: ", "导出快照：") + snapshot
            self.snapshot_label.setText(text)
            self.snapshot_label.setToolTip(text)

        def _resize_for_screen(self) -> None:
            screen = QtGui.QGuiApplication.primaryScreen()
            available = screen.availableGeometry() if screen else QtCore.QRect(0, 0, 1280, 800)
            # The settings body is scrollable, so the dialog follows a small
            # screen instead of imposing a minimum that would be clipped.
            width = max(1, min(680, available.width() - 24))
            height = max(1, min(760, available.height() - 32))
            self.resize(width, height)

        # ---- language -----------------------------------------------------

        def _retranslate_ui(self) -> None:
            english = self._language.lower().startswith("en")
            self.setWindowTitle(
                "Export butterfly measurement figure" if english else "导出蝴蝶测量图"
            )
            self.title_label.setText(
                "Butterfly measurement figure" if english else "蝴蝶花样测量图"
            )
            self.subtitle_label.setText(
                "One export setup · rendering runs in the background"
                if english
                else "一次设置完成导出 · 高分辨率绘图在后台运行"
            )
            self.browse_button.setText("Browse…" if english else "浏览…")
            self._parent_label.setText("Output parent" if english else "输出父目录")
            self._target_label.setText("New bundle" if english else "新图包目录")
            self._width_label.setText("Figure width" if english else "图稿宽度")
            self._dpi_label.setText("Raster resolution" if english else "栅格分辨率")
            self._size_label.setText("Size" if english else "尺寸")
            self.width_combo.setItemText(
                0, "Single column · 89 mm" if english else "单栏 · 89 mm"
            )
            self.width_combo.setItemText(
                1, "Double column · 183 mm" if english else "双栏 · 183 mm"
            )
            for index, dpi in enumerate(_DPIS):
                self.dpi_combo.setItemText(index, f"{dpi} dpi")
            self._preview_caption.setText(
                "Static paper-ratio cue only; this is not a rendered preview."
                if english
                else "这里只显示纸面比例示意，不是真实绘图预览。"
            )
            self.state_group.setTitle(
                "Current measurement state" if english else "当前测量状态"
            )
            self._q_unit_caption.setText("q unit" if english else "q 单位")
            self._stage_caption.setText("workflow" if english else "流程阶段")
            self._quality_caption.setText("engineering quality" if english else "工程质量")
            self._scientific_caption.setText(
                "scientific status" if english else "科学状态"
            )
            self._coordinates_caption.setText("coordinates" if english else "坐标")
            self._snapshot_caption.setText(
                "export source snapshot" if english else "导出来源快照"
            )
            self.q_unit_label.setText(self._q_unit_display())
            self.stage_label.setText(self._stage_display())
            self.quality_label.setText(
                f"{self._measurement_status} / {self._quality_status}"
            )
            scientific_status = self._scientific_status.strip() or "not_assessed"
            if scientific_status.lower() in {"not_assessed", "unknown", "none"}:
                self.scientific_label.setText(
                    "Not assessed; export does not mean accepted"
                    if english
                    else "未评估；导出不等于科学接纳"
                )
            else:
                self.scientific_label.setText(
                    f"{scientific_status} · acceptance is never inferred"
                    if english
                    else f"{scientific_status} · 不会据此推断科学接纳"
                )
            self.coordinates_label.setText(self._coordinates_display())
            self.notes_label.setText(
                "Outputs include editable-text SVG and PDF, raster PNG/TIFF, source_data.npz, and provenance files. "
                "Figure colors use the exporter-wide 0.5th–99.5th percentile of valid pixels; the main-view percentile control is not copied into this figure."
                if english
                else "输出包含可编辑文字的 SVG/PDF、PNG/TIFF 栅格图、source_data.npz 源数据和来源记录。"
                "图稿统一使用有效像素第 0.5–99.5 百分位颜色范围；主图分位数控件不会完整沿用。"
            )
            self.open_button.setText(
                (
                    "Open previous-snapshot bundle"
                    if self._export_stale and english
                    else "打开此前快照图包"
                    if self._export_stale
                    else "Open exported figure bundle"
                    if english
                    else "打开已导出图包"
                )
            )
            if not self._running:
                self.cancel_button.setText("Close" if english else "关闭")
                self.start_button.setText(
                    "Start background export" if english else "开始后台导出"
                )
            self._controls_changed()
            self._render_snapshot_context()
            self._render_status()

        def _q_unit_display(self) -> str:
            unit = self._q_unit.strip() or "unknown"
            if unit.lower() in {"unknown", "pixel-q", "pixel_q", "pixel", "px"}:
                return self._text(
                    f"{unit} · no physical q calibration (diagnostic export allowed)",
                    f"{unit} · 无物理 q 标定（仍允许导出诊断图）",
                )
            return unit

        def _stage_display(self) -> str:
            normalized = self._stage.strip().lower()
            label = "Evaluate" if normalized == "evaluate" else "Trace"
            return label if self._language.lower().startswith("en") else (
                "评估" if normalized == "evaluate" else "追踪"
            )

        def _coordinates_display(self) -> str:
            missing = []
            if not self._has_qx:
                missing.append("qx")
            if not self._has_qy:
                missing.append("qy")
            if not missing:
                return self._text("qx/qy available", "qx/qy 已提供")
            values = ", ".join(missing)
            return self._text(
                f"Missing {values}; the worker will report a readable export error.",
                f"缺少 {values}；后台导出会返回可读的错误提示。",
            )


else:

    class FigureExportDialog:  # pragma: no cover - exercised without Qt only
        """Import-safe placeholder when the optional Qt UI is unavailable."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs
            require_qt()


__all__ = [
    "FigureExportDialog",
    "_DPIS",
    "_WIDTHS_MM",
    "_figure_height_mm",
    "_pixel_size",
]
