"""Parameter-driven schematic studio, isolated from scientific optimization."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import threading
from typing import Any

import numpy as np

from ..lamellar import LamellarSettings, build_lamellar_scene, load_lamellar_sources
from ..serialization import json_safe
from .lamellar_controls import LamellarControls, tr
from .lamellar_state import FrameImageReader, compact_source, field, frame_label, padded_bounds, rgba_image, source_images
from .qt_compat import QT_AVAILABLE, QtCore, QtGui, QtWidgets, require_qt
from .workers import AnalysisWorker


def _build_scenes(sources: list[Any], values: dict[str, Any], cancel: threading.Event) -> list[Any]:
    settings = LamellarSettings.from_mapping(values)
    scenes = []
    for source in sources:
        if cancel.is_set():
            raise RuntimeError("cancelled")
        scenes.append(build_lamellar_scene(source, settings))
        scenes[-1].metadata["display_q_window"] = field(field(source, "analysis", {}), "q_window")
    if len(scenes) > 1 and settings.reference_period is None:
        periods = [float(scene.metadata["reference_period"]) for scene in scenes
                   if scene.metadata.get("available") and scene.metadata.get("reference_period")]
        if periods:
            settings = replace(settings, reference_period=max(periods))
            scenes = []
            for source in sources:
                if cancel.is_set():
                    raise RuntimeError("cancelled")
                scenes.append(build_lamellar_scene(source, settings))
                scenes[-1].metadata["display_q_window"] = field(field(source, "analysis", {}), "q_window")
    return scenes


if QT_AVAILABLE:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.figure import Figure
    import pyqtgraph as pg

    from .lamellar_3d import Lamellar3DView
    from .qspace import QSpaceView

    class LamellarPage(QtWidgets.QWidget):
        """Independent view settings and immutable input snapshots for each frame."""

        frameSelected = QtCore.Signal(object)
        documentChanged = QtCore.Signal()

        def __init__(self, parent: Any = None, *, language: str = "zh_CN") -> None:
            super().__init__(parent)
            self.setObjectName("lamellarPage")
            self.language = language
            self.settings = LamellarSettings().to_dict()
            self._palette = "blue_orange"
            self._publication_document: dict[str, Any] = {}
            self._publication_dialog: Any = None
            self.sources: list[Any] = []
            self.scenes: list[Any] = []
            self.current_index = 0
            self.current_scene: Any = None
            self._fresh = False
            self._context_signature: str | None = None
            self._import_path: str | None = None
            self._image_cache: OrderedDict[int, tuple[Any, Any, Any]] = OrderedDict()
            self._images: tuple[Any, Any, Any] = (None, None, None)
            self._image_error: str | None = None
            self._undo: list[dict[str, Any]] = []
            self._redo: list[dict[str, Any]] = []
            self._pool = QtCore.QThreadPool(self)
            self._pool.setMaxThreadCount(2)
            self._workers: dict[int, AnalysisWorker] = {}
            self._cancel_events: dict[int, threading.Event] = {}
            self._active: dict[str, int] = {}
            self._serial = 0
            self._closed = False
            self._bounds: Any = None
            self._focus: str | None = None
            self._debounce = QtCore.QTimer(self)
            self._debounce.setSingleShot(True)
            self._debounce.setInterval(140)
            self._debounce.timeout.connect(self._queue_build)
            self._play_timer = QtCore.QTimer(self)
            self._play_timer.setInterval(200)
            self._play_timer.timeout.connect(self._advance)
            self._build_ui()
            self.controls.set_settings(self.settings)
            self.set_language(language)
            self._clear_views("empty")

        def _button(self, key: str, callback: Any, *, tool: bool = False) -> Any:
            widget = QtWidgets.QToolButton() if tool else QtWidgets.QPushButton()
            widget.setObjectName("lamellar_" + key)
            widget.setProperty("textKey", key)
            widget.setText(tr(key, self.language))
            widget.setToolTip(tr(key, self.language))
            widget.setAccessibleName(tr(key, self.language))
            widget.clicked.connect(callback)
            return widget

        def _build_ui(self) -> None:
            self.setStyleSheet(
                "#lamellarPage { background: #f6f7f8; color: #253340; }"
                "#lamellarPage QGroupBox { border: 1px solid #dce2e6; border-radius: 5px; margin-top: 10px; padding-top: 7px; }"
                "#lamellarPage QGroupBox::title { subcontrol-origin: margin; left: 9px; color: #536472; }"
                "#lamellarPage QPushButton, #lamellarPage QToolButton { padding: 5px 8px; }"
                "#lamellarPage QToolButton:checked { background: #dcebf1; color: #285a70; border: 1px solid #b9d1dc; border-radius: 4px; }"
                "#lamellarPage QComboBox, #lamellarPage QSpinBox, #lamellarPage QDoubleSpinBox { min-height: 24px; }"
            )
            root = QtWidgets.QVBoxLayout(self)
            root.setContentsMargins(14, 10, 14, 8)
            root.setSpacing(8)
            header = QtWidgets.QHBoxLayout()
            titles = QtWidgets.QVBoxLayout()
            self.title = QtWidgets.QLabel()
            self.title.setStyleSheet("font-size: 22px; font-weight: 600;")
            self.subtitle = QtWidgets.QLabel()
            self.subtitle.setStyleSheet("color: #647381; font-size: 11px;")
            titles.addWidget(self.title)
            titles.addWidget(self.subtitle)
            header.addLayout(titles, 1)
            self.import_button = self._button("import", lambda: None, tool=True)
            self.import_button.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
            import_menu = QtWidgets.QMenu(self.import_button)
            self.import_actions = []
            for key, callback in (("file", self._open_file), ("folder", self._open_folder)):
                action = import_menu.addAction(tr(key, self.language))
                action.setProperty("textKey", key)
                action.triggered.connect(callback)
                self.import_actions.append(action)
            self.import_button.setMenu(import_menu)
            header.addWidget(self.import_button)
            self.undo_button = self._button("undo", self.undo, tool=True)
            self.redo_button = self._button("redo", self.redo, tool=True)
            header.addWidget(self.undo_button)
            header.addWidget(self.redo_button)
            self.settings_button = self._button("settings", self._toggle_settings, tool=True)
            self.settings_button.setCheckable(True)
            self.settings_button.setChecked(True)
            header.addWidget(self.settings_button)
            self.publication_button = self._button("publication", self.open_publication, tool=True)
            header.addWidget(self.publication_button)
            self.export_button = self._button("export", lambda: None, tool=True)
            self.export_button.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
            export_menu = QtWidgets.QMenu(self.export_button)
            self.export_actions = []
            for key, sequence in (("export_one", False), ("export_sequence", True)):
                action = export_menu.addAction(tr(key, self.language))
                action.setProperty("textKey", key)
                action.triggered.connect(lambda _checked=False, seq=sequence: self._choose_export(seq))
                self.export_actions.append(action)
            self.export_button.setMenu(export_menu)
            header.addWidget(self.export_button)
            root.addLayout(header)
            self.status = QtWidgets.QLabel()
            self.status.setWordWrap(True)
            self.status.setStyleSheet("color: #8a601f; padding: 3px 0;")
            root.addWidget(self.status)
            self.splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
            self.left = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
            self.panels: dict[str, Any] = {}
            self.q_view = QSpaceView()
            self.q_view.set_compact_mode(True)
            self.q_view.setMinimumSize(190, 90)
            self.left.addWidget(self._panel("saxs", self.q_view))
            self.figure = Figure(figsize=(4, 3), facecolor="#f7f8fa")
            self.canvas = FigureCanvasQTAgg(self.figure)
            self.canvas.setMinimumSize(190, 90)
            self.canvas.mpl_connect("scroll_event", self._zoom_2d)
            self.left.addWidget(self._panel("two_d", self.canvas))
            self.left.setSizes([330, 320])
            self.splitter.addWidget(self.left)
            self.view3d = Lamellar3DView()
            self.view3d.setMinimumSize(240, 180)
            self.view3d.cameraChanged.connect(lambda _: self.documentChanged.emit())
            self.splitter.addWidget(self._panel("three_d", self.view3d))
            self.control_scroll = QtWidgets.QScrollArea()
            self.control_scroll.setWidgetResizable(True)
            self.control_scroll.setMinimumWidth(248)
            self.control_scroll.setMaximumWidth(320)
            self.control_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
            self.controls = LamellarControls(language=self.language)
            self.controls.settingsChanged.connect(self.update_settings)
            self.controls.presentationChanged.connect(self.set_palette)
            self.controls.resetRequested.connect(self.reset_settings)
            self.control_scroll.setWidget(self.controls)
            self.splitter.addWidget(self.control_scroll)
            self.splitter.setStretchFactor(0, 3)
            self.splitter.setStretchFactor(1, 5)
            self.splitter.setStretchFactor(2, 0)
            self.splitter.setSizes([330, 520, 260])
            self.splitter.setChildrenCollapsible(False)
            root.addWidget(self.splitter, 1)
            self.frame_text = QtWidgets.QLabel()
            self.frame_text.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
            root.addWidget(self.frame_text)
            timeline = QtWidgets.QHBoxLayout()
            self.previous_button = self._button("previous", lambda: self.select_frame(self.current_index - 1), tool=True)
            self.play_button = self._button("play", self.toggle_play, tool=True)
            self.next_button = self._button("next", lambda: self.select_frame(self.current_index + 1), tool=True)
            for widget in (self.previous_button, self.play_button, self.next_button):
                timeline.addWidget(widget)
            self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            self.slider.setObjectName("lamellarFrameSlider")
            self.slider.setRange(0, 0)
            self.slider.valueChanged.connect(self.select_frame)
            timeline.addWidget(self.slider, 1)
            self.speed = QtWidgets.QComboBox()
            for fps in (1, 2, 5, 10):
                self.speed.addItem(f"{fps} fps", fps)
            self.speed.setCurrentIndex(2)
            self.speed.currentIndexChanged.connect(lambda: self._play_timer.setInterval(int(1000 / self.speed.currentData())))
            timeline.addWidget(self.speed)
            self.sources_button = self._button("sources", self.show_sources, tool=True)
            timeline.addWidget(self.sources_button)
            root.addLayout(timeline)
            self.trajectory = pg.PlotWidget(background="#f6f7f8")
            self.trajectory.setObjectName("lamellarTrajectory")
            self.trajectory.setMinimumHeight(44)
            self.trajectory.setMaximumHeight(80)
            self.trajectory.setMouseEnabled(x=False, y=False)
            self.trajectory.showGrid(x=False, y=True, alpha=.15)
            self.trajectory.scene().sigMouseClicked.connect(self._trajectory_click)
            root.addWidget(self.trajectory)
            footer = QtWidgets.QHBoxLayout()
            self.boundary = QtWidgets.QLabel()
            self.boundary.setWordWrap(True)
            self.boundary.setStyleSheet("color: #6b7782; font-size: 10px;")
            footer.addWidget(self.boundary, 1)
            self.cancel_button = self._button("cancel", self.cancel_jobs, tool=True)
            footer.addWidget(self.cancel_button)
            root.addLayout(footer)
            self.undo_action = QtGui.QAction(self)
            self.undo_action.setShortcut(QtGui.QKeySequence.StandardKey.Undo)
            self.undo_action.setShortcutContext(QtCore.Qt.ShortcutContext.WidgetWithChildrenShortcut)
            self.undo_action.triggered.connect(self.undo)
            self.addAction(self.undo_action)
            self.redo_action = QtGui.QAction(self)
            self.redo_action.setShortcut(QtGui.QKeySequence.StandardKey.Redo)
            self.redo_action.setShortcutContext(QtCore.Qt.ShortcutContext.WidgetWithChildrenShortcut)
            self.redo_action.triggered.connect(self.redo)
            self.addAction(self.redo_action)

        def _panel(self, key: str, content: Any) -> Any:
            panel = QtWidgets.QWidget()
            layout = QtWidgets.QVBoxLayout(panel)
            layout.setContentsMargins(0, 0, 0, 0)
            bar = QtWidgets.QHBoxLayout()
            label = QtWidgets.QLabel(tr(key, self.language))
            label.setProperty("textKey", key)
            label.setStyleSheet("font-weight: 600; color: #52616c;")
            bar.addWidget(label, 1)
            if key == "three_d":
                self.camera_combo = QtWidgets.QComboBox()
                for name in ("isometric", "front", "side", "top"):
                    self.camera_combo.addItem(tr(name, self.language), name)
                self.camera_combo.currentIndexChanged.connect(lambda: self.view3d.set_standard_view(self.camera_combo.currentData()))
                bar.addWidget(self.camera_combo)
                bar.addWidget(self._button("reset_camera", lambda: self.view3d.reset_camera(), tool=True))
            button = self._button("focus", lambda: self.focus_view(key), tool=True)
            bar.addWidget(button)
            layout.addLayout(bar)
            if key == "three_d":
                self.period_readout = QtWidgets.QLabel()
                self.period_readout.setStyleSheet("color: #52616c; padding-bottom: 2px; font-size: 11px;")
                layout.addWidget(self.period_readout)
            layout.addWidget(content, 1)
            self.panels[key] = panel
            return panel

        def set_language(self, language: str) -> None:
            self.language = language
            self.title.setText(tr("title", language))
            self.subtitle.setText(tr("subtitle", language))
            self.boundary.setText(tr("source_note", language))
            for widget in self.findChildren(QtWidgets.QWidget):
                key = widget.property("textKey")
                if key and hasattr(widget, "setText"):
                    widget.setText(tr(key, language))
                    widget.setToolTip(tr(key, language))
                    widget.setAccessibleName(tr(key, language))
            for action in self.import_actions + self.export_actions:
                action.setText(tr(action.property("textKey"), language))
            for i in range(self.camera_combo.count()):
                self.camera_combo.setItemText(i, tr(self.camera_combo.itemData(i), language))
            self.controls.set_language(language)
            self.view3d.set_language(language)
            self.q_view.set_language(language)
            self.slider.setAccessibleName(tr("frame", language))
            self.speed.setAccessibleName("Playback frames per second" if language == "en" else "播放速度：每秒帧数")
            self._refresh_buttons()
            if self.scenes:
                self._render_scene()
            else:
                self.status.setText(tr("empty" if self._fresh else "stale", language))

        def set_source(self, source: Any, *, context_signature: str | None = None) -> None:
            self.set_series([source], context_signature=context_signature)

        def set_series(self, sources: Any, *, context_signature: str | None = None) -> None:
            self.cancel_jobs()
            self.sources = list(sources)
            self.scenes = []
            self._context_signature = context_signature
            self._fresh = True
            self._import_path = None
            self.current_index = 0
            self._image_cache.clear()
            self._clear_views("building")
            if not self.sources and self.settings["period_source"] != "manual":
                self._fresh = False
                self._clear_views("empty")
                return
            self._queue_build()

        def invalidate(self, reason: str = "stale") -> None:
            self.cancel_jobs()
            self._fresh = False
            self.scenes = []
            self.current_scene = None
            self._image_cache.clear()
            self._clear_views(reason)

        def update_settings(self, values: Mapping[str, Any], *, record: bool = True) -> None:
            try:
                candidate = LamellarSettings.from_mapping(dict(values)).to_dict()
            except (ValueError, TypeError) as exc:
                self.controls.set_settings(self.settings)
                self.status.setText(str(exc))
                return
            if candidate == self.settings:
                return
            if record:
                self._undo.append(self._history_settings())
                self._undo = self._undo[-100:]
                self._redo.clear()
            self.settings = candidate
            self.controls.set_settings(candidate)
            self._stop_job("build")
            self._stop_job("export")
            self._play_timer.stop()
            self.scenes = []
            self.current_scene = None
            if self._publication_dialog is not None:
                self._publication_dialog.invalidate_source()
            self._refresh_buttons()
            self._debounce.start()
            self.documentChanged.emit()

        def reset_settings(self) -> None:
            self.update_settings(LamellarSettings().to_dict())

        def _history_settings(self) -> dict[str, Any]:
            return {**deepcopy(self.settings), "_palette": self._palette}

        def _restore_history(self, values: dict[str, Any]) -> None:
            values = dict(values)
            self._palette = values.pop("_palette", self._palette)
            self.controls.set_palette(self._palette)
            self.update_settings(values, record=False)
            self._apply_palette()
            if self.scenes:
                self._render_scene()

        def set_palette(self, palette: str) -> None:
            if palette not in ("blue_orange", "grayscale") or palette == self._palette:
                return
            self._undo.append(self._history_settings())
            self._redo.clear()
            self._palette = palette
            self.controls.set_palette(palette)
            self._apply_palette()
            if self.scenes:
                self._render_scene()
            self.documentChanged.emit()

        def _apply_palette(self) -> None:
            from ..publication_models import PublicationStyle

            style = PublicationStyle.from_mapping({**self._publication_document.get("style", {}),
                "palette": "editorial" if self._palette == "blue_orange" else "grayscale"})
            colors = style.colors
            self.q_view.set_branch_colors(colors)
            self._publication_document["style"] = style.to_dict()
            setter = getattr(self.view3d, "set_publication_style", None)
            if callable(setter):
                setter(style)
            for scene in self.scenes:
                scene.colors[:] = np.asarray([colors[int(branch) % 2] for branch in scene.branch_ids]).reshape((-1, 4))
                scene.metadata["presentation"] = {"palette": self._palette}

        def undo(self) -> None:
            if self._undo:
                values = self._undo.pop()
                self._redo.append(self._history_settings())
                self._restore_history(values)

        def redo(self) -> None:
            if self._redo:
                values = self._redo.pop()
                self._undo.append(self._history_settings())
                self._restore_history(values)

        def _queue_build(self) -> None:
            if self._closed:
                return
            if not self._fresh and self.settings["period_source"] != "manual":
                self._clear_views("stale" if self.sources else "empty")
                return
            sources = self.sources or [{}]
            if not self._fresh:
                sources = [{}]  # explicit manual scenario has no inherited measurement authority
            values = dict(self.settings)
            self.status.setText(tr("building", self.language))
            self._start_job("build", lambda cancel, progress: _build_scenes(sources, values, cancel))

        def _start_job(self, kind: str, operation: Any) -> None:
            self._stop_job(kind)
            self._serial += 1
            token = self._serial
            event = threading.Event()
            worker = AnalysisWorker(lambda **_: operation(event, worker.report_progress), generation=token, kind=kind)
            self._workers[token] = worker
            self._cancel_events[token] = event
            self._active[kind] = token
            worker.signals.finished.connect(self._finished)
            worker.signals.error.connect(self._failed)
            worker.signals.progress.connect(self._progress)
            self._refresh_buttons()
            self._pool.start(worker)

        def _stop_job(self, kind: str) -> None:
            token = self._active.pop(kind, None)
            if token in self._cancel_events:
                self._cancel_events[token].set()

        def cancel_jobs(self) -> None:
            was_running = bool(self._active) or self._debounce.isActive()
            if "images" in self._active:
                self._image_error = tr("cancelled", self.language)
            self._play_timer.stop()
            self._debounce.stop()
            for kind in tuple(self._active):
                self._stop_job(kind)
            if was_running:
                self.status.setText(tr("cancelled", self.language))
            self._refresh_buttons()

        def _take_job(self, token: int, kind: str) -> bool:
            self._workers.pop(token, None)
            self._cancel_events.pop(token, None)
            current = self._active.get(kind) == token and not self._closed
            if current:
                self._active.pop(kind, None)
            self._refresh_buttons()
            return current

        @QtCore.Slot(int, str, object)
        def _finished(self, token: int, kind: str, result: Any) -> None:
            if not self._take_job(token, kind):
                return
            if kind == "build":
                self.scenes = result
                self._apply_palette()
                self._bounds = padded_bounds(result)
                self.slider.blockSignals(True)
                self.slider.setRange(0, max(0, len(result) - 1))
                self.slider.blockSignals(False)
                self.select_frame(min(self.current_index, max(0, len(result) - 1)))
            elif kind == "images":
                index, images = result
                self._image_cache[index] = images
                while len(self._image_cache) > 3:
                    self._image_cache.popitem(last=False)
                if index == self.current_index:
                    self._apply_images(images)
            elif kind == "import":
                sources, path = result
                self.set_series(sources)
                self._import_path = path
            elif kind == "export":
                self.status.setText(tr("saved", self.language) + " " + str(result))
            self._refresh_buttons()

        @QtCore.Slot(int, str, object)
        def _failed(self, token: int, kind: str, error: Any) -> None:
            if not self._take_job(token, kind):
                return
            if kind == "images":
                self._apply_images((None, None, None))
                self._image_error = str(error)
                self.q_view.set_empty_message("Image could not be read\nSee status below" if self.language == "en" else "原始图像读取失败\n请查看状态说明")
                self.q_view.setToolTip(str(error))
                self.status.setText(str(error))
                self._refresh_buttons()
            else:
                self.status.setText(str(error))

        @QtCore.Slot(int, str, object)
        def _progress(self, token: int, kind: str, value: Any) -> None:
            if self._active.get(kind) == token:
                key = {"export": "exporting", "import": "loading"}.get(kind, "building")
                self.status.setText(tr(key, self.language) + " " + str(value))

        def select_frame(self, index: int) -> None:
            if not self.scenes:
                return
            index = max(0, min(int(index), len(self.scenes) - 1))
            self.current_index = index
            self.slider.blockSignals(True)
            self.slider.setValue(index)
            self.slider.blockSignals(False)
            self._render_scene()
            self._stop_job("images")
            if index in self._image_cache:
                self._image_cache.move_to_end(index)
                self._apply_images(self._image_cache[index])
            else:
                self._apply_images((None, None, None))
                if self.sources and index < len(self.sources):
                    source = self.sources[index]
                    self._start_job("images", lambda cancel, progress: (index, source_images(source)))
            self.frameSelected.emit(self.current_scene.metadata.get("source_identity", {}))

        def _render_scene(self) -> None:
            if not self.scenes:
                return
            if self._publication_dialog is not None and self._publication_dialog.isVisible():
                self._publication_dialog.invalidate_source()
            from ..lamellar_render import render_lamellar_2d

            self.current_scene = self.scenes[self.current_index]
            scene = self.current_scene
            population_text = []
            for population in scene.metadata.get("populations", []):
                if population.get("period") is not None:
                    name = "AB"[population["branch_id"]] if population["branch_id"] in (0, 1) else str(population["branch_id"])
                    population_text.append(f"{name}: L_app = {population['period']:.4g} {scene.length_unit}")
            self.period_readout.setText("    ·    ".join(population_text))
            self.figure.clear()
            self.figure.subplots_adjust(left=.18, right=.97, bottom=.28, top=.97)
            axis = self.figure.add_subplot(111)
            render_lamellar_2d(scene, ax=axis, language=self.language, bounds=self._bounds, decorate=False)
            axis.tick_params(labelsize=8, pad=2)
            axis.xaxis.label.set_size(8)
            axis.yaxis.label.set_size(8)
            self.canvas.draw_idle()
            self.view3d.set_scene(scene, bounds=self._bounds, reset_camera=False)
            self.view3d.set_selected_branch(self.settings["selected_branch"])
            status = scene.metadata.get("status", "unavailable")
            status_key = "manual_status" if status == "manual" else status
            message = scene.metadata.get("message", "")
            self.status.setText(tr(status_key, self.language) + (" · " + message if message and not scene.metadata.get("available") else ""))
            self.status.setToolTip(message)
            source = self.sources[self.current_index] if self.sources and self.current_index < len(self.sources) else {}
            self.frame_text.setText(f"{self.current_index + 1} / {len(self.scenes)}  ·  {frame_label(source, self.current_index)}")
            self._draw_trajectory()
            self._refresh_buttons()

        def _apply_images(self, images: tuple[Any, Any, Any]) -> None:
            self._images = images
            self._image_error = None
            observed, qx, qy = images
            self.q_view.set_empty_message(tr("no_image", self.language) if observed is None else None)
            source = self.sources[self.current_index] if self.sources and self.current_index < len(self.sources) else {}
            q_unit = field(source, "q_unit", "unknown") if qx is not None and qy is not None else "pixel-q"
            self.q_view.set_data(observed, qx=qx, qy=qy, q_unit=q_unit, valid_mask=field(source, "valid_mask"))
            self.q_view.set_display_settings("log", 99.5)
            analysis = field(source, "analysis", {}) or {}
            self.q_view.set_q_window(field(analysis, "q_window"))
            payload = field(source, "butterfly", field(field(source, "observables", {}), "butterfly"))
            self.q_view.set_butterfly(payload)
            if qx is None or qy is None:
                self.q_view.set_butterfly(None)
            branch = self.settings["selected_branch"]
            self.q_view.set_visible_branches({(i, side): branch < 0 or branch == i
                                             for i in (0, 1) for side in ("upper", "lower", "unknown")})
            self._refresh_buttons()
            if self._publication_dialog is not None:
                self._publication_dialog.context_ready()

        def _draw_trajectory(self) -> None:
            self.trajectory.clear()
            unit = self.current_scene.length_unit if self.current_scene else "relative"
            self.trajectory.setLabel("left", tr("trajectory", self.language), units="nm" if unit == "nm" else None)
            self.trajectory.setLabel("bottom", tr("frame", self.language))
            for branch, color in ((0, "#2ba4df"), (1, "#e39a38")):
                y = []
                for scene in self.scenes:
                    population = next((p for p in scene.metadata.get("populations", []) if p["branch_id"] == branch), None)
                    y.append(population["period"] if population and scene.length_unit == unit and scene.metadata.get("available") else np.nan)
                if not np.any(np.isfinite(y)):
                    continue
                self.trajectory.plot(np.arange(1, len(y) + 1), y, pen=pg.mkPen(color, width=1.8),
                                     symbol="o", symbolSize=4, symbolBrush=color, connect="finite")
            self.trajectory.addItem(pg.InfiniteLine(self.current_index + 1, angle=90, pen=pg.mkPen("#6b7982", width=1)))

        def _trajectory_click(self, event: Any) -> None:
            position = self.trajectory.plotItem.vb.mapSceneToView(event.scenePos())
            self.select_frame(round(position.x()) - 1)

        def _zoom_2d(self, event: Any) -> None:
            if event.inaxes is None or event.xdata is None or event.ydata is None:
                return
            factor = .8 if event.button == "up" else 1.25
            ax = event.inaxes
            for getter, setter, center in ((ax.get_xlim, ax.set_xlim, event.xdata), (ax.get_ylim, ax.set_ylim, event.ydata)):
                low, high = getter()
                setter(center + (low - center) * factor, center + (high - center) * factor)
            self.canvas.draw_idle()

        def toggle_play(self) -> None:
            if self._play_timer.isActive():
                self._play_timer.stop()
            elif len(self.scenes) > 1:
                if self.current_index == len(self.scenes) - 1:
                    self.select_frame(0)
                self._play_timer.start()
            self._refresh_buttons()

        def _advance(self) -> None:
            if "images" in self._active:
                return  # display every physical frame instead of skipping while I/O catches up
            if self.current_index + 1 >= len(self.scenes):
                self._play_timer.stop()
                self._refresh_buttons()
            else:
                self.select_frame(self.current_index + 1)

        def focus_view(self, name: str) -> None:
            self._focus = None if self._focus == name else name
            for key, panel in self.panels.items():
                panel.setVisible(self._focus in (None, key))
            self.left.setVisible(self._focus in (None, "saxs", "two_d"))
            self.control_scroll.setVisible(self._focus is None and self.settings_button.isChecked())

        def _toggle_settings(self) -> None:
            self.control_scroll.setVisible(self.settings_button.isChecked() and self._focus is None)

        def _clear_views(self, key: str) -> None:
            if self._publication_dialog is not None:
                self._publication_dialog.invalidate_source()
            self.current_scene = None
            self.period_readout.clear()
            self.view3d.clear(tr(key, self.language))
            self.q_view.set_data(None)
            self.q_view.set_butterfly(None)
            self.figure.clear()
            self.canvas.draw_idle()
            self.trajectory.clear()
            self.status.setText(tr(key, self.language))
            self.frame_text.clear()
            self._images = (None, None, None)
            self._image_error = None
            self._refresh_buttons()

        def _refresh_buttons(self) -> None:
            if not hasattr(self, "export_button"):
                return
            ready = bool(self.current_scene is not None and self.current_scene.metadata.get("available") and not self._image_error)
            busy = bool(self._active)
            self.export_button.setEnabled(ready and not any(kind in self._active for kind in ("build", "export", "images")))
            self.publication_button.setEnabled(ready and not any(kind in self._active for kind in ("build", "images")))
            self.export_actions[1].setEnabled(len(self.scenes) > 1 and ready)
            self.cancel_button.setVisible(busy)
            self.undo_button.setEnabled(bool(self._undo))
            self.redo_button.setEnabled(bool(self._redo))
            self.play_button.setEnabled(len(self.scenes) > 1)
            self.trajectory.setVisible(len(self.scenes) > 1)
            self.play_button.setText(tr("pause" if self._play_timer.isActive() else "play", self.language))
            self.previous_button.setEnabled(bool(self.scenes) and self.current_index > 0)
            self.next_button.setEnabled(self.current_index + 1 < len(self.scenes))

        def _open_file(self) -> None:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, tr("import", self.language), "", "Result (*.json)")
            if path:
                self.import_results(path)

        def _open_folder(self) -> None:
            path = QtWidgets.QFileDialog.getExistingDirectory(self, tr("import", self.language))
            if path:
                self.import_results(path)

        def import_results(self, path: str | Path) -> None:
            self.cancel_jobs()
            self.invalidate("loading")
            self.status.setText(tr("loading", self.language))
            path = str(Path(path).resolve())
            self._start_job("import", lambda cancel, progress: (load_lamellar_sources(path), path))

        def _choose_export(self, sequence: bool) -> None:
            parent = QtWidgets.QFileDialog.getExistingDirectory(self, tr("export", self.language))
            if not parent:
                return
            target = Path(parent) / ("lamellar-sequence" if sequence else "lamellar-schematic")
            count = 2
            while target.exists():
                target = Path(parent) / f"lamellar-{'sequence' if sequence else 'schematic'}-{count}"
                count += 1
            self.export_to(target, sequence=sequence)

        def export_to(self, path: str | Path, *, sequence: bool = False) -> None:
            if self.current_scene is None or not self.current_scene.metadata.get("available"):
                raise ValueError("No current scene to export")
            if self._image_error:
                raise ValueError("Original image could not be read: " + self._image_error)
            if "images" in self._active:
                raise ValueError("The selected frame image is still loading")
            self._play_timer.stop()
            path = Path(path)
            scene = deepcopy(self.current_scene)
            scenes = deepcopy(self.scenes)
            language, camera = self.language, self.view3d.camera_state()
            observed, qx, qy = self._images
            capture = None
            if not sequence and self.view3d.available:
                try:
                    capture = rgba_image(self.view3d.capture_image(width=2400, height=1800))
                except RuntimeError as exc:
                    self.status.setText(str(exc))
                    return
            sources = list(self.sources)
            fps = float(self.speed.currentData())
            self.status.setText(tr("exporting", language))

            def export(cancel: threading.Event, progress: Any) -> Path:
                from ..lamellar_export import export_lamellar_scene, export_lamellar_sequence

                if sequence:
                    reader = FrameImageReader(sources)
                    export_lamellar_sequence(scenes, path, images=reader.view(), qmaps=reader.view(qmaps=True), camera=camera,
                                             language=language, fps=fps, cancel_event=cancel, progress=progress)
                else:
                    export_lamellar_scene(scene, path, observed=observed, qx=qx, qy=qy,
                                          image_3d=capture, camera=camera, language=language, cancel_event=cancel)
                return path

            self._start_job("export", export)

        def show_sources(self) -> None:
            dialog = QtWidgets.QDialog(self)
            dialog.setWindowTitle(tr("sources", self.language))
            dialog.resize(660, 480)
            layout = QtWidgets.QVBoxLayout(dialog)
            metadata = self.current_scene.metadata if self.current_scene is not None else {"message": tr("empty", self.language)}
            tabs = QtWidgets.QTabWidget()
            summary = QtWidgets.QWidget()
            summary_layout = QtWidgets.QVBoxLayout(summary)
            note = QtWidgets.QLabel(tr("source_note", self.language))
            note.setWordWrap(True)
            summary_layout.addWidget(note)
            table = QtWidgets.QTreeWidget()
            table.setRootIsDecorated(False)
            english = self.language.startswith("en")
            table.setHeaderLabels(["Parameter", "Value", "Unit", "Origin", "Status"] if english else ["参数", "数值", "单位", "来源", "状态"])
            names = {"period": "表观周期", "q_star": "径向峰位", "normal_angle": "示意法向", "thickness_ratio": "厚度 / 周期"}
            statuses = {"candidate": "候选", "available": "有观测支持", "assumed": "设定", "manual": "手动设定", "undetermined": "未确定", "unavailable": "不可用"}
            for row in metadata.get("parameter_sources", []):
                name = str(row.get("name", ""))
                value = row.get("value")
                value_text = "—" if value is None else (f"{value:.6g}" if isinstance(value, (float, int)) else str(value))
                if row.get("candidate_value") is not None and value is None:
                    value_text = f"({row['candidate_value']:.6g})"
                item = QtWidgets.QTreeWidgetItem([
                    name if english else names.get(name, name), value_text, str(row.get("unit", "")),
                    str(row.get("source", "")), str(row.get("status", "")) if english else statuses.get(str(row.get("status", "")), str(row.get("status", ""))),
                ])
                item.setToolTip(1, str(row.get("interval") or row.get("reason") or ""))
                table.addTopLevelItem(item)
            table.header().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
            table.header().setStretchLastSection(True)
            summary_layout.addWidget(table, 1)
            assumptions = QtWidgets.QPlainTextEdit()
            assumptions.setReadOnly(True)
            assumptions.setPlainText("\n".join(str(value) for value in metadata.get("assumptions", [])))
            assumptions.setMaximumHeight(120)
            summary_layout.addWidget(assumptions)
            tabs.addTab(summary, "Parameters and assumptions" if english else "参数与假设")
            text = QtWidgets.QPlainTextEdit()
            text.setReadOnly(True)
            text.setPlainText(json.dumps(json_safe(metadata), ensure_ascii=False, indent=2, allow_nan=False))
            tabs.addTab(text, "Detailed provenance" if english else "详细来源记录")
            layout.addWidget(tabs)
            buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
            buttons.rejected.connect(dialog.reject)
            layout.addWidget(buttons)
            dialog.exec()

        def document(self) -> dict[str, Any]:
            return {"version": 1, "settings": deepcopy(self.settings), "camera": self.view3d.camera_state(),
                    "presentation": {"palette": self._palette},
                    "publication": deepcopy(self._publication_document),
                    "frame_index": self.current_index, "fps": self.speed.currentData(), "fresh": self._fresh,
                    "context_sha256": self._context_hash(self._context_signature), "import_path": self._import_path,
                    "sources": [compact_source(source) for source in self.sources]}

        @staticmethod
        def _context_hash(signature: str | None) -> str | None:
            if not signature:
                return None
            try:
                context = json.loads(signature)
            except (json.JSONDecodeError, TypeError):
                context = None
            if isinstance(context, dict):
                from ..lamellar_utils import _q_scale_to_nm

                parameters = context.get("parameters", {})
                if isinstance(parameters, dict):
                    for spec in parameters.values():
                        if isinstance(spec, dict) and "unit" in spec:
                            scale = _q_scale_to_nm(spec["unit"])
                            if scale is not None:
                                spec["unit"] = "nm^-1" if scale == 1. else "angstrom^-1"
                signature = json.dumps(context, sort_keys=True, separators=(",", ":"), allow_nan=False)
            return hashlib.sha256(signature.encode()).hexdigest()

        def restore_document(self, document: Mapping[str, Any] | None, *, context_signature: str | None = None,
                             observed: Any = None, qx: Any = None, qy: Any = None) -> None:
            self.cancel_jobs()
            document = dict(document or {})
            if document and document.get("version") != 1:
                raise ValueError("Unsupported lamellar_view version")
            values = LamellarSettings.from_mapping(document.get("settings", {})).to_dict()
            self.settings = values
            self._publication_document = deepcopy(document.get("publication") or {})
            if self._publication_dialog is not None:
                self._publication_dialog.restore_document(self._publication_document)
            self._palette = document.get("presentation", {}).get("palette", "blue_orange")
            if self._palette not in ("blue_orange", "grayscale"):
                raise ValueError("Unsupported lamellar palette")
            self.controls.set_palette(self._palette)
            self.controls.set_settings(values)
            self._undo.clear()
            self._redo.clear()
            self.sources = list(document.get("sources") or [])
            self._image_cache.clear()
            self.scenes = []
            self.current_index = min(max(0, int(document.get("frame_index", 0))), max(0, len(self.sources) - 1))
            self._context_signature = context_signature
            self._import_path = document.get("import_path")
            saved_context = document.get("context_sha256")
            matches = saved_context is None or saved_context == self._context_hash(context_signature)
            self._fresh = bool(document.get("fresh") and matches)
            if not matches:
                self._publication_document.update(camera=None, camera_auto=True)
                if self._publication_dialog is not None:
                    self._publication_dialog.restore_document(self._publication_document)
            if self._fresh and len(self.sources) == 1 and observed is not None and saved_context is not None:
                self.sources[0] = {**self.sources[0], "observed": observed, "qx": qx, "qy": qy}
            self.view3d.set_camera_state(document.get("camera", {}))
            index = self.speed.findData(document.get("fps", 5))
            self.speed.setCurrentIndex(index if index >= 0 else 2)
            self._clear_views("building" if self._fresh else "stale")
            self._queue_build()

        def snapshot_state(self) -> dict[str, Any]:
            return {"document": self.document(), "sources": list(self.sources), "signature": self._context_signature}

        def restore_state(self, snapshot: Mapping[str, Any]) -> None:
            self.restore_document(snapshot.get("document"), context_signature=snapshot.get("signature"))
            self.sources = list(snapshot.get("sources") or [])
            self._queue_build()

        def shutdown(self) -> None:
            self._closed = True
            self.cancel_jobs()
            if self._publication_dialog is not None:
                self._publication_dialog.cancel()
                self._publication_dialog.close()

        def jobs_running(self) -> bool:
            return bool(self._workers) or bool(self._publication_dialog is not None and self._publication_dialog.jobs_running())

        def publication_context(self) -> dict[str, Any] | None:
            if (self.current_scene is None or not self.current_scene.metadata.get("available")
                    or self._image_error or any(kind in self._active for kind in ("build", "images"))):
                return None
            observed, qx, qy = self._images
            camera = getattr(self.view3d, "publication_camera_state", self.view3d.camera_state)()
            return {"scene": self.current_scene, "observed": observed, "qx": qx, "qy": qy,
                    "camera": camera}

        def open_publication(self) -> None:
            if self.publication_context() is None:
                return
            from .publication_panel import PublicationDialog

            if self._publication_dialog is None:
                self._publication_dialog = PublicationDialog(self, document=self._publication_document,
                                                              context_provider=self.publication_context)
                self._publication_dialog.documentChanged.connect(self._save_publication_document)
                self._publication_dialog.finished.connect(self._publication_closed)
            self._publication_dialog.show()
            self._publication_dialog.raise_()

        def _save_publication_document(self, document: Mapping[str, Any]) -> None:
            self._publication_document = deepcopy(dict(document))
            self.documentChanged.emit()

        def _publication_closed(self, *_: Any) -> None:
            style = self._publication_document.get("style", {})
            self._palette = "grayscale" if style.get("palette") == "grayscale" else "blue_orange"
            self.controls.set_palette(self._palette)
            self._apply_palette()
            if self.scenes:
                self._render_scene()

        def hideEvent(self, event: Any) -> None:  # noqa: N802
            self._play_timer.stop()
            super().hideEvent(event)

else:
    class LamellarPage:
        def __init__(self, *_: Any, **__: Any) -> None:
            require_qt()
