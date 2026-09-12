"""Qt butterfly-analysis page.

This page is a thin workflow adapter.  It owns interaction state (selected
frame, branch visibility, serializable edits and display contrast), while the
existing :class:`RefinementMainWindow` continues to own the worker lifecycle
and service calls.  In particular, changing contrast or selecting a point
never starts an analysis request.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import math
from pathlib import Path
from typing import Any

from ..butterfly_quality import classify_ellipse_publication, unpublished_ellipse_shape
from ..butterfly_settings import normalize_butterfly_settings
from ..fit_overlays import fit_geometry_layers
from .qt_compat import QT_AVAILABLE, QtCore, QtGui, QtWidgets, require_qt
from .qspace import QSpaceView
from .butterfly_export import export_butterfly_analysis
from .i18n import translate

try:
    import numpy as _np
except Exception:  # pragma: no cover - numpy is a core dependency normally
    _np = None

try:
    import pyqtgraph as _pg
except Exception:  # pragma: no cover - optional plotting dependency
    _pg = None


DEFAULT_BUTTERFLY_SETTINGS: dict[str, Any] = {
    "stage": "trace",
    "edits": [],
    "resamples": 0,
    "evaluation_resamples": 32,
    "seed": 20260906,
    "sensitivity": True,
}


def _read(source: Any, names: tuple[str, ...], default: Any = None) -> Any:
    if isinstance(source, Mapping):
        for name in names:
            if name in source:
                return source[name]
    else:
        for name in names:
            if hasattr(source, name):
                return getattr(source, name)
    return default


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _fmt(value: Any) -> str:
    if value is None or value == "":
        return "—"
    if isinstance(value, float):
        return f"{value:.6g}" if math.isfinite(value) else "—"
    return str(value)


if QT_AVAILABLE:

    class _ProfilePanel(QtWidgets.QWidget):
        """Small plot/table fallback used below the q-space canvas."""

        def __init__(
            self,
            title: str,
            parent: Any = None,
            *,
            data_view: bool = True,
        ) -> None:
            super().__init__(parent)
            self._title = str(title)
            self._english = True
            self._empty_message_key = "profile.empty.select"
            self._x_label: str | None = None
            self._y_label: str | None = None
            self._series_names: tuple[str, ...] = ()
            self.setObjectName(title.replace(" ", "") + "Panel")
            layout = QtWidgets.QVBoxLayout(self)
            layout.setContentsMargins(4, 4, 4, 4)
            self.title_label = QtWidgets.QLabel(title, self)
            self.title_label.setStyleSheet("font-weight: 600;")
            self.title_label.setWordWrap(True)
            self.title_label.setMaximumHeight(32)
            layout.addWidget(self.title_label)
            self.plot = None
            self.table = QtWidgets.QTableWidget(0, 1, self)
            self.table.setHorizontalHeaderLabels((translate("en", "profile.column.coordinate"),))
            self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
            self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectItems)
            self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
            self.table.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
            self.table.horizontalHeader().setStretchLastSection(True)
            self.table.setMinimumHeight(112)
            self.table.setAccessibleName(self._title)
            self.table.setAccessibleDescription(translate("en", "a11y.profile_data_empty"))
            self.view_tabs = None
            if _pg is not None:
                self.plot = _pg.PlotWidget(self)
                self.plot.setBackground("#16181e")
                self.plot.showGrid(x=True, y=True, alpha=0.18)
                self.plot.setMinimumHeight(112)
                self.plot.setAccessibleName(self._title)
                self.plot.setAccessibleDescription(
                    f"{self._title}; measured and fitted diagnostic series"
                )
                self.view_tabs = QtWidgets.QTabWidget(self)
                self.view_tabs.setObjectName("profileDataViews")
                self.view_tabs.setAccessibleName(translate("en", "a11y.profile_tabs"))
                self.view_tabs.addTab(self.plot, translate("en", "profile.view.plot"))
                if data_view:
                    self.view_tabs.addTab(self.table, translate("en", "profile.view.data"))
                self.view_tabs.setCurrentWidget(self.plot)
                layout.addWidget(self.view_tabs, 1)
            else:
                layout.addWidget(self.table, 1)
            self.empty_label = QtWidgets.QLabel("Select a measured point", self)
            self.empty_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self.empty_label.setStyleSheet("color: #8c95a5;")
            layout.addWidget(self.empty_label)

        def clear(
            self,
            message: str | None = None,
            *,
            message_key: str | None = None,
        ) -> None:
            if self.plot is not None:
                self.plot.clear()
            self.table.setRowCount(0)
            self.table.setColumnCount(1)
            language = "en" if self._english else "zh_CN"
            self.table.setHorizontalHeaderLabels(
                (translate(language, "profile.column.coordinate"),)
            )
            self._x_label = None
            self._y_label = None
            self._series_names = ()
            self._empty_message_key = message_key
            if message_key is not None:
                message = translate(language, message_key)
            elif message is None:
                self._empty_message_key = "profile.empty.select"
                message = translate(language, self._empty_message_key)
            self.empty_label.setText(message or "")
            self.empty_label.setVisible(True)
            self._update_data_accessibility()

        def set_language(self, *, english: bool) -> None:
            """Refresh visible and assistive text without changing data."""

            self._english = bool(english)
            language = "en" if english else "zh_CN"
            title = self._title
            if not english:
                title = {
                    "Angular peak signal": "方位峰信号",
                    "Radial peak signal": "径向峰信号",
                }.get(title, title)
                title = {
                    "Normal profile · raw / fit / residual": "法向剖面 · 原始 / 拟合 / 残差",
                    "Ellipse-local u/v · narrow-axis": "椭圆局部 u/v · 短轴诊断",
                }.get(title, title)
            self.title_label.setText(title)
            if self._empty_message_key is not None:
                self.empty_label.setText(translate(language, self._empty_message_key))
            if self.plot is not None:
                self.plot.setAccessibleName(title)
                self.plot.setAccessibleDescription(
                    f"{title}; measured and fitted diagnostic series"
                    if english
                    else f"{title}；显示测量与拟合诊断序列"
                )
            if self.view_tabs is not None:
                self.view_tabs.setAccessibleName(translate(language, "a11y.profile_tabs"))
                self.view_tabs.setTabText(0, translate(language, "profile.view.plot"))
                if self.view_tabs.count() > 1:
                    self.view_tabs.setTabText(1, translate(language, "profile.view.data"))
            self._update_data_accessibility()

        def _display_series_names(self) -> list[str]:
            return [self._display_series_name(name) for name in self._series_names]

        def _display_series_name(self, name: str) -> str:
            if name in {
                "raw",
                "smoothed",
                "isotropic_reference",
                "detection",
                "fit",
                "residual",
                "u",
                "v",
            }:
                language = "en" if self._english else "zh_CN"
                return translate(language, f"profile.series.{name}")
            return name

        def _update_data_accessibility(self) -> None:
            language = "en" if self._english else "zh_CN"
            self.table.setAccessibleName(self.title_label.text())
            if self.table.rowCount() == 0 or self._x_label is None:
                self.table.setColumnCount(1)
                self.table.setHorizontalHeaderLabels(
                    (translate(language, "profile.column.coordinate"),)
                )
                description = translate(language, "a11y.profile_data_empty")
            else:
                display_names = self._display_series_names()
                self.table.setColumnCount(1 + len(display_names))
                self.table.setHorizontalHeaderLabels([self._x_label, *display_names])
                description = translate(
                    language,
                    "a11y.profile_data_table",
                    x_axis=self._x_label,
                    y_axis=self._y_label or "",
                    series=", ".join(display_names),
                )
            self.table.setAccessibleDescription(description)

        def set_series(
            self,
            x: Sequence[Any],
            series: Mapping[str, Sequence[Any]],
            *,
            x_label: str = "q offset",
            y_label: str = "value",
            markers: Sequence[tuple[float, str]] = (),
        ) -> None:
            values = []
            if _np is not None:
                try:
                    values = _np.asarray(x, dtype=float)
                except (TypeError, ValueError):
                    values = []
            names = list(series)
            x_values = list(values) if hasattr(values, "__len__") and len(values) else list(x)
            self._x_label = str(x_label)
            self._y_label = str(y_label)
            self._series_names = tuple(str(name) for name in names)
            self.table.setColumnCount(1 + len(names))
            self.table.setRowCount(len(x_values))
            for row, value in enumerate(x_values):
                self.table.setItem(row, 0, QtWidgets.QTableWidgetItem(_fmt(value)))
                for col, name in enumerate(names, 1):
                    data = series[name]
                    cell = data[row] if row < len(data) else None
                    self.table.setItem(row, col, QtWidgets.QTableWidgetItem(_fmt(cell)))
            self.table.resizeColumnsToContents()
            self._update_data_accessibility()
            if self.plot is not None:
                self.plot.clear()
                colors = {
                    "raw": (220, 230, 238),
                    "smoothed": (110, 211, 255),
                    "isotropic_reference": (155, 139, 232),
                    "detection": (255, 145, 106),
                    "fit": (72, 190, 242),
                    "residual": (243, 157, 73),
                    "u": (72, 190, 242),
                    "v": (243, 157, 73),
                }
                for name, y in series.items():
                    if not name:
                        continue
                    try:
                        self.plot.plot(
                            values,
                            _np.asarray(y, dtype=float) if _np is not None else list(y),
                            pen=_pg.mkPen(colors.get(name, (190, 190, 200)), width=2),
                            symbol="o" if name == "raw" else None,
                            symbolSize=4,
                            name=self._display_series_name(str(name)),
                        )
                    except (TypeError, ValueError):
                        continue
                for marker_value, marker_label in markers:
                    marker_position = _finite(marker_value)
                    if marker_position is None:
                        continue
                    marker = _pg.InfiniteLine(
                        pos=marker_position,
                        angle=90,
                        movable=False,
                        pen=_pg.mkPen(
                            (255, 205, 72),
                            width=1.3,
                            style=QtCore.Qt.PenStyle.DashLine,
                        ),
                    )
                    marker.setZValue(10)
                    marker.setToolTip(str(marker_label))
                    self.plot.addItem(marker)
                self.plot.setLabel("bottom", x_label)
                self.plot.setLabel("left", y_label)
                self.empty_label.setVisible(not bool(x_values) or not bool(series))
                return
            self.empty_label.setVisible(not bool(x_values) or not bool(series))

        def set_uv_series(
            self,
            points: Sequence[Mapping[str, Any]],
            curves: Mapping[str, Mapping[str, Sequence[Any]]],
            *,
            v_scale: float = 1.0,
            x_label: str = "u",
            y_label: str = "v",
        ) -> None:
            if self.plot is not None:
                self.clear(message_key="profile.empty.select")
                self.plot.clear()
                curve_colors = {
                    "upper": (72, 190, 242),
                    "lower": (243, 157, 73),
                }
                for side, curve in curves.items():
                    if not isinstance(curve, Mapping):
                        continue
                    try:
                        u_values = _np.asarray(curve.get("u", []), dtype=float)
                        v_values = _np.asarray(curve.get("v", []), dtype=float) * float(v_scale)
                        if u_values.size and v_values.size:
                            self.plot.plot(
                                u_values,
                                v_values,
                                pen=_pg.mkPen(curve_colors.get(str(side), (190, 190, 200)), width=2),
                                name=f"{side} fit",
                            )
                    except (TypeError, ValueError):
                        continue
                grouped: dict[str, tuple[list[float], list[float]]] = {}
                for point in points or ():
                    if not isinstance(point, Mapping) or not bool(point.get("accepted", True)):
                        continue
                    try:
                        side = str(point.get("side", "unknown"))
                        grouped.setdefault(side, ([], []))[0].append(float(point.get("u")))
                        grouped.setdefault(side, ([], []))[1].append(float(point.get("v")) * float(v_scale))
                    except (TypeError, ValueError):
                        continue
                for side, (u_values, v_values) in grouped.items():
                    if u_values:
                        self.plot.plot(
                            u_values,
                            v_values,
                            pen=None,
                            symbol="o" if side == "upper" else "s",
                            symbolSize=6,
                            symbolBrush=_pg.mkBrush(curve_colors.get(side, (180, 180, 185))),
                            name=f"{side} points",
                        )
                self.plot.setLabel("bottom", x_label)
                self.plot.setLabel("left", y_label)
                self.empty_label.setVisible(not bool(grouped) and not bool(curves))
                return
            self.clear(message_key="profile.empty.select")


    class ButterflyWorkbench(QtWidgets.QWidget):
        """First-page butterfly tracing/evaluation workflow."""

        identifyRequested = QtCore.Signal(object)
        evaluateRequested = QtCore.Signal(object)
        applyToBatchRequested = QtCore.Signal(object)
        cancelRequested = QtCore.Signal()
        editChanged = QtCore.Signal(object)
        frameSelected = QtCore.Signal(object)
        pointSelected = QtCore.Signal(object)
        displayChanged = QtCore.Signal(str, float)
        analysisChanged = QtCore.Signal(object)
        exportRequested = QtCore.Signal()
        figureExportRequested = QtCore.Signal()

        def __init__(self, parent: Any = None, *, language: str = "zh_CN") -> None:
            super().__init__(parent)
            self.setObjectName("butterflyWorkbench")
            self._language = str(language)
            self._settings = deepcopy(DEFAULT_BUTTERFLY_SETTINGS)
            self._result: dict[str, Any] = {}
            self._frames: list[Any] = []
            self._profiles: dict[str, Any] = {}
            self._ellipse_local: dict[str, Any] = {}
            self._uv_display_magnification = 1.0
            self._selected_ellipse_point: dict[str, Any] = {}
            self._edits: list[dict[str, Any]] = []
            self._redo_edits: list[dict[str, Any]] = []
            self._legacy_method: str | None = None
            self._frame_data: dict[str, Any] = {}
            self._export_context: dict[str, Any] = {}
            self._result_fresh = False
            self._current_frame: Any = None
            self._diagnostic_magnification = 1.0
            self._busy = False
            self._has_loaded_data = False
            self._page_status_state = "ready"
            self._page_status_kind = ""
            self._page_status_error: Any = None
            self._q_range_error = ""
            self._job_elapsed_s: float | None = None
            self._job_progress_percent: int | None = None
            self._job_progress_phase = ""
            self._last_valid_analysis_range: tuple[float | None, float | None, float] = (
                None,
                None,
                0.0,
            )
            self._result_revision = 0
            self._peak_landmarks: dict[str, Any] = {}
            self._fit_layers: dict[str, Any] = {}
            self._model_parameters: Any = None
            self._model_reference_axis_deg: float | None = None
            self._model_status: str | None = None
            self._model_diagnostics: dict[str, Any] = {}
            self._selected_landmark: dict[str, Any] = {}
            self._selected_landmark_id: str | None = None
            self._requested_q_window: tuple[float, float] | None = None
            self._landmark_zoomed = False
            self._manual_review: dict[str, Any] = {
                "manual_status": "unreviewed",
                "reviewed_by": "",
                "reviewed_at": None,
                "review_notes": "",
                "result_revision": None,
            }

            root = QtWidgets.QVBoxLayout(self)
            root.setContentsMargins(8, 6, 8, 6)
            root.setSpacing(6)

            header = QtWidgets.QHBoxLayout()
            self.title_label = QtWidgets.QLabel("蝴蝶分析 / Butterfly analysis", self)
            self.title_label.setObjectName("butterflyTitle")
            self.title_label.setStyleSheet("font-size: 17px; font-weight: 700;")
            header.addWidget(self.title_label)
            self.method_label = QtWidgets.QLabel("method: butterfly_curvature", self)
            self.method_label.setObjectName("butterflyMethodLabel")
            self.method_label.setToolTip("ridge_method=butterfly_curvature · arc tracing method")
            self.method_label.setStyleSheet("color: #56667a;")
            header.addWidget(self.method_label)
            header.addStretch(1)
            self.status_label = QtWidgets.QLabel("Ready · trace", self)
            self.status_label.setObjectName("butterflyStatusLabel")
            header.addWidget(self.status_label)
            root.addLayout(header)
            self.workflow_hint_label = QtWidgets.QLabel(self)
            self.workflow_hint_label.setObjectName("butterflyWorkflowHint")
            self.workflow_hint_label.setWordWrap(True)
            self.workflow_hint_label.setStyleSheet("color: #56667a; font-size: 11px;")
            root.addWidget(self.workflow_hint_label)

            self.legacy_banner = QtWidgets.QLabel(self)
            self.legacy_banner.setObjectName("butterflyLegacyBanner")
            self.legacy_banner.setWordWrap(True)
            self.legacy_banner.setStyleSheet("color: #875d08; background: #fff4cc; padding: 4px;")
            self.legacy_banner.setVisible(False)
            root.addWidget(self.legacy_banner)

            splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal, self)
            splitter.setObjectName("butterflyMainSplitter")
            root.addWidget(splitter, 1)

            left = QtWidgets.QWidget(splitter)
            left_layout = QtWidgets.QHBoxLayout(left)
            left_layout.setContentsMargins(0, 0, 4, 0)
            frame_rail = QtWidgets.QWidget(left)
            frame_rail.setObjectName("butterflyFrameRail")
            frame_rail.setMinimumWidth(142)
            frame_rail.setMaximumWidth(190)
            frame_layout = QtWidgets.QVBoxLayout(frame_rail)
            frame_layout.setContentsMargins(0, 0, 4, 0)
            frame_row = QtWidgets.QHBoxLayout()
            self.frame_title_label = QtWidgets.QLabel("Frames / 帧", frame_rail)
            frame_row.addWidget(self.frame_title_label)
            self.frame_count_label = QtWidgets.QLabel("0", frame_rail)
            frame_row.addWidget(self.frame_count_label)
            frame_row.addStretch(1)
            self.frame_source_label = QtWidgets.QLabel("loaded frame", frame_rail)
            self.frame_source_label.setWordWrap(True)
            self.frame_source_label.setStyleSheet("color: #56667a;")
            frame_layout.addLayout(frame_row)
            frame_layout.addWidget(self.frame_source_label)
            self.frame_list = QtWidgets.QListWidget(frame_rail)
            self.frame_list.setObjectName("butterflyFrameList")
            self.frame_list.setAccessibleName("Frames")
            self.frame_list.setAccessibleDescription("Select the active SAXS frame")
            self.frame_list.setMaximumHeight(160)
            self.frame_list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
            self.frame_list.currentRowChanged.connect(self._on_frame_changed)
            frame_layout.addWidget(self.frame_list, 1)
            self.point_title_label = QtWidgets.QLabel("Points / 点", frame_rail)
            self.point_list = QtWidgets.QListWidget(frame_rail)
            self.point_list.setObjectName("butterflyPointList")
            self.point_list.setAccessibleName("Points")
            self.point_list.setAccessibleDescription("Select a measured butterfly point")
            self.point_list.setMaximumHeight(150)
            self.point_list.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
            self.point_list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
            self.point_list.currentRowChanged.connect(self._on_point_list_changed)
            self.point_list.itemActivated.connect(lambda _item: self._on_point_list_changed(self.point_list.currentRow()))
            self.point_list.installEventFilter(self)
            QtGui.QShortcut(QtGui.QKeySequence("Delete"), self.point_list).activated.connect(
                self._exclude_selected_point
            )
            frame_layout.addWidget(self.point_title_label)
            frame_layout.addWidget(self.point_list)
            frame_layout.addStretch(1)
            left_layout.addWidget(frame_rail, 0)
            canvas = QtWidgets.QWidget(left)
            canvas_layout = QtWidgets.QVBoxLayout(canvas)
            canvas_layout.setContentsMargins(0, 0, 0, 0)
            self.qspace = QSpaceView(canvas)
            self.qspace.setObjectName("butterflyQSpace")
            canvas_layout.addWidget(self.qspace, 5)

            self.diagnostics_tabs = QtWidgets.QTabWidget(canvas)
            self.diagnostics_tabs.setObjectName("butterflyDiagnosticsTabs")
            diagnostics = QtWidgets.QSplitter(
                QtCore.Qt.Orientation.Horizontal, self.diagnostics_tabs
            )
            diagnostics.setObjectName("butterflyDiagnosticsSplitter")
            self.normal_profile = _ProfilePanel("Normal profile · raw / fit / residual", diagnostics)
            self.normal_profile.setObjectName("normalProfilePanel")
            self.ellipse_diagnostic = _ProfilePanel(
                "Ellipse-local u/v · narrow-axis", diagnostics, data_view=False
            )
            self.ellipse_diagnostic.setObjectName("ellipseDiagnosticPanel")
            diagnostics.addWidget(self.normal_profile)
            diagnostics.addWidget(self.ellipse_diagnostic)
            diagnostics.setStretchFactor(0, 1)
            diagnostics.setStretchFactor(1, 1)
            self.diagnostics_tabs.addTab(diagnostics, "Point diagnostics")
            peak_profiles = QtWidgets.QSplitter(
                QtCore.Qt.Orientation.Horizontal, self.diagnostics_tabs
            )
            peak_profiles.setObjectName("peakLandmarkProfiles")
            self.peak_angular_profile = _ProfilePanel("Angular peak signal", peak_profiles)
            self.peak_angular_profile.setObjectName("peakAngularProfilePanel")
            self.peak_radial_profile = _ProfilePanel("Radial peak signal", peak_profiles)
            self.peak_radial_profile.setObjectName("peakRadialProfilePanel")
            peak_profiles.addWidget(self.peak_angular_profile)
            peak_profiles.addWidget(self.peak_radial_profile)
            peak_profiles.setStretchFactor(0, 1)
            peak_profiles.setStretchFactor(1, 1)
            self.diagnostics_tabs.addTab(peak_profiles, "Peak profiles")
            canvas_layout.addWidget(self.diagnostics_tabs, 2)
            self.magnification_label = QtWidgets.QLabel(
                "Display magnification: ×1.0 (diagnostic display only)", canvas
            )
            self.magnification_label.setObjectName("diagnosticMagnificationLabel")
            self.magnification_label.setStyleSheet("color: #56667a; font-size: 11px;")
            canvas_layout.addWidget(self.magnification_label)
            left_layout.addWidget(canvas, 1)
            splitter.addWidget(left)

            right_scroll = QtWidgets.QScrollArea(splitter)
            right_scroll.setObjectName("butterflyControlsScroll")
            right_scroll.setWidgetResizable(True)
            right_scroll.setMinimumWidth(340)
            right_scroll.setHorizontalScrollBarPolicy(
                QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
            )
            right_panel = QtWidgets.QWidget(right_scroll)
            right_panel.setObjectName("butterflyControlsPanel")
            right_layout = QtWidgets.QVBoxLayout(right_panel)
            right_layout.setContentsMargins(4, 2, 4, 4)
            right_layout.setSpacing(7)

            analysis_group = QtWidgets.QGroupBox("Analysis range", right_panel)
            analysis_group.setObjectName("butterflyAnalysisRange")
            analysis_form = QtWidgets.QFormLayout(analysis_group)
            self.q_min_edit = QtWidgets.QLineEdit("Auto", analysis_group)
            self.q_min_edit.setObjectName("butterflyQMin")
            self.q_max_edit = QtWidgets.QLineEdit("Auto", analysis_group)
            self.q_max_edit.setObjectName("butterflyQMax")
            self.q_range_error_label = QtWidgets.QLabel(analysis_group)
            self.q_range_error_label.setObjectName("butterflyQRangeError")
            self.q_range_error_label.setWordWrap(True)
            self.q_range_error_label.setStyleSheet("color: #b42318; font-size: 11px;")
            self.q_range_error_label.setVisible(False)
            self.reference_axis_spin = QtWidgets.QDoubleSpinBox(analysis_group)
            self.reference_axis_spin.setObjectName("butterflyReferenceAxis")
            self.reference_axis_spin.setRange(-360.0, 360.0)
            self.reference_axis_spin.setDecimals(2)
            analysis_form.addRow("q min", self.q_min_edit)
            analysis_form.addRow("q max", self.q_max_edit)
            analysis_form.addRow(self.q_range_error_label)
            analysis_form.addRow("Reference axis (deg)", self.reference_axis_spin)
            self.q_min_edit.editingFinished.connect(self._on_analysis_range_changed)
            self.q_max_edit.editingFinished.connect(self._on_analysis_range_changed)
            self.reference_axis_spin.valueChanged.connect(self._on_analysis_range_changed)
            right_layout.addWidget(analysis_group)

            evaluation_group = QtWidgets.QGroupBox("Evaluation", right_panel)
            evaluation_group.setObjectName("butterflyEvaluationControls")
            evaluation_layout = QtWidgets.QFormLayout(evaluation_group)
            self.evaluation_resamples_combo = QtWidgets.QComboBox(evaluation_group)
            self.evaluation_resamples_combo.setObjectName("butterflyEvaluationResamples")
            for label, value in (("Quick fit · 0", 0), ("Standard · 32", 32), ("Extended · 128", 128)):
                self.evaluation_resamples_combo.addItem(label, value)
            self.evaluation_resamples_combo.setCurrentIndex(1)
            evaluation_layout.addRow("Uncertainty resamples", self.evaluation_resamples_combo)
            self.sensitivity_check = QtWidgets.QCheckBox("Run sensitivity checks", evaluation_group)
            self.sensitivity_check.setObjectName("butterflySensitivityChecks")
            self.sensitivity_check.setChecked(True)
            evaluation_layout.addRow(self.sensitivity_check)
            self.evaluation_resamples_combo.currentIndexChanged.connect(self._on_evaluation_settings_changed)
            self.sensitivity_check.toggled.connect(self._on_evaluation_settings_changed)
            right_layout.addWidget(evaluation_group)

            display_group = QtWidgets.QGroupBox("Display", right_panel)
            display_group.setObjectName("butterflyDisplayControls")
            display_form = QtWidgets.QFormLayout(display_group)
            self.display_scale_combo = QtWidgets.QComboBox(display_group)
            self.display_scale_combo.setObjectName("butterflyDisplayScale")
            self.display_scale_combo.addItem("Linear", "linear")
            self.display_scale_combo.addItem("Log1p", "log1p")
            self.display_scale_combo.addItem("Asinh", "asinh")
            display_form.addRow("Scale", self.display_scale_combo)
            self.display_percentile_spin = QtWidgets.QDoubleSpinBox(display_group)
            self.display_percentile_spin.setObjectName("butterflyDisplayPercentile")
            self.display_percentile_spin.setRange(50.0, 100.0)
            self.display_percentile_spin.setDecimals(1)
            self.display_percentile_spin.setValue(99.5)
            self.display_percentile_spin.setSuffix(" %")
            display_form.addRow("Upper percentile", self.display_percentile_spin)
            self.uv_magnification_spin = QtWidgets.QDoubleSpinBox(display_group)
            self.uv_magnification_spin.setObjectName("butterflyUvMagnification")
            self.uv_magnification_spin.setRange(0.1, 100.0)
            self.uv_magnification_spin.setDecimals(1)
            self.uv_magnification_spin.setValue(1.0)
            self.uv_magnification_spin.setSuffix(" ×")
            display_form.addRow("u/v magnification", self.uv_magnification_spin)
            self.display_reset_button = QtWidgets.QPushButton("Reset contrast", display_group)
            self.display_reset_button.setObjectName("butterflyDisplayReset")
            display_form.addRow(self.display_reset_button)
            self.display_scale_combo.currentIndexChanged.connect(self._on_display_changed)
            self.display_percentile_spin.valueChanged.connect(self._on_display_changed)
            self.uv_magnification_spin.valueChanged.connect(self._on_uv_magnification_changed)
            self.display_reset_button.clicked.connect(self._reset_display)
            right_layout.addWidget(display_group)

            overlay_group = QtWidgets.QGroupBox("Fit-source overlays", right_panel)
            overlay_group.setObjectName("butterflyFitOverlays")
            overlay_layout = QtWidgets.QVBoxLayout(overlay_group)
            overlay_layout.setContentsMargins(7, 5, 7, 5)
            overlay_form = QtWidgets.QFormLayout()
            self.overlay_mode_combo = QtWidgets.QComboBox(overlay_group)
            self.overlay_mode_combo.setObjectName("butterflyOverlayMode")
            for key, mode in (
                ("overlay.measured_only", "measured_only"),
                ("overlay.observed_ridges", "observed_ridges"),
                ("overlay.geometry_candidate", "geometry_candidate"),
                ("overlay.full2d_model", "full2d_model"),
                ("overlay.compare", "compare"),
            ):
                self.overlay_mode_combo.addItem(self._tr(key), mode)
            self.overlay_mode_combo.setCurrentIndex(
                self.overlay_mode_combo.findData("observed_ridges")
            )
            overlay_form.addRow("Displayed layer", self.overlay_mode_combo)
            overlay_layout.addLayout(overlay_form)
            self.fit_source_label = QtWidgets.QLabel(overlay_group)
            self.fit_source_label.setObjectName("butterflyFitSourceLabel")
            self.fit_source_label.setWordWrap(True)
            self.fit_source_label.setStyleSheet("color: #56667a; font-size: 11px;")
            overlay_layout.addWidget(self.fit_source_label)
            self.fit_assessment_label = QtWidgets.QLabel(overlay_group)
            self.fit_assessment_label.setObjectName("butterflyFitAssessment")
            self.fit_assessment_label.setWordWrap(True)
            self.fit_assessment_label.setStyleSheet(
                "color: #9f1d16; background: #fff0ee; padding: 5px; font-weight: 700;"
            )
            self.fit_assessment_label.setVisible(False)
            overlay_layout.addWidget(self.fit_assessment_label)
            self.overlay_mode_combo.currentIndexChanged.connect(
                self._on_overlay_mode_changed
            )
            right_layout.addWidget(overlay_group)

            landmark_group = QtWidgets.QGroupBox(
                "Peak landmarks · display only", right_panel
            )
            landmark_group.setObjectName("butterflyPeakLandmarks")
            landmark_layout = QtWidgets.QVBoxLayout(landmark_group)
            landmark_layout.setContentsMargins(7, 5, 7, 5)
            landmark_toggles = QtWidgets.QHBoxLayout()
            self.global_max_check = QtWidgets.QCheckBox(landmark_group)
            self.global_max_check.setObjectName("showGlobalRawMaximum")
            self.global_max_check.setChecked(True)
            landmark_toggles.addWidget(self.global_max_check)
            self.supported_peaks_check = QtWidgets.QCheckBox(landmark_group)
            self.supported_peaks_check.setObjectName("showSupportedPeaks")
            self.supported_peaks_check.setChecked(True)
            landmark_toggles.addWidget(self.supported_peaks_check)
            landmark_layout.addLayout(landmark_toggles)
            self.peak_table = QtWidgets.QTableWidget(0, 4, landmark_group)
            self.peak_table.setObjectName("butterflyPeakLandmarkTable")
            self.peak_table.setEditTriggers(
                QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
            )
            self.peak_table.setSelectionBehavior(
                QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
            )
            self.peak_table.setSelectionMode(
                QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
            )
            self.peak_table.setHorizontalScrollBarPolicy(
                QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
            )
            self.peak_table.verticalHeader().hide()
            self.peak_table.verticalHeader().setDefaultSectionSize(19)
            self.peak_table.horizontalHeader().setMinimumSectionSize(28)
            self.peak_table.horizontalHeader().setStretchLastSection(True)
            self.peak_table.setMinimumHeight(76)
            self.peak_table.setMaximumHeight(136)
            self.peak_table.setAccessibleName("Peak landmark diagnostics")
            self.peak_table.setAccessibleDescription(
                "Global raw maximum and supported lobe peaks; selecting a row focuses its actual q coordinate"
            )
            for column, width in enumerate((34, 66, 72, 138)):
                self.peak_table.setColumnWidth(column, width)
            landmark_layout.addWidget(self.peak_table)
            self.reset_peak_zoom_button = QtWidgets.QPushButton(landmark_group)
            self.reset_peak_zoom_button.setObjectName("resetPeakLandmarkZoom")
            self.reset_peak_zoom_button.setEnabled(False)
            self.reset_peak_zoom_button.clicked.connect(self._reset_landmark_zoom)
            landmark_layout.addWidget(self.reset_peak_zoom_button)
            self.peak_table.currentCellChanged.connect(
                self._on_peak_table_selection_changed
            )
            self.global_max_check.toggled.connect(self._on_landmark_visibility_changed)
            self.supported_peaks_check.toggled.connect(
                self._on_landmark_visibility_changed
            )
            right_layout.addWidget(landmark_group)

            branch_group = QtWidgets.QGroupBox("Branch / side visibility", right_panel)
            branch_group.setObjectName("branchSideVisibility")
            branch_layout = QtWidgets.QVBoxLayout(branch_group)
            branch_layout.setContentsMargins(8, 5, 8, 5)
            self._branch_checks: dict[tuple[int, str], QtWidgets.QCheckBox] = {}
            rows = (
                (0, "upper", "A · upper", QtGui.QColor(42, 154, 220)),
                (0, "lower", "A · lower", QtGui.QColor(42, 154, 220)),
                (1, "upper", "B · upper", QtGui.QColor(239, 143, 44)),
                (1, "lower", "B · lower", QtGui.QColor(239, 143, 44)),
            )
            for branch, side, text, color in rows:
                row_widget = QtWidgets.QWidget(branch_group)
                row_widget.setFixedHeight(20)
                row_layout = QtWidgets.QHBoxLayout(row_widget)
                row_layout.setContentsMargins(0, 0, 0, 0)
                swatch = QtWidgets.QLabel("●", row_widget)
                swatch.setStyleSheet(f"color: {color.name()};")
                row_layout.addWidget(swatch)
                check = QtWidgets.QCheckBox(text, row_widget)
                check.setObjectName(f"branch{branch}{side.capitalize()}Visible")
                check.setChecked(True)
                check.toggled.connect(
                    lambda enabled, branch=branch, side=side: self.qspace.set_visible_branch(branch, side, enabled)
                )
                row_layout.addWidget(check, 1)
                branch_layout.addWidget(row_widget)
                self._branch_checks[(branch, side)] = check
            unknown_check = QtWidgets.QCheckBox("Unknown / 未知", branch_group)
            unknown_check.setObjectName("branchUnknownVisible")
            unknown_check.setFixedHeight(20)
            unknown_check.setChecked(True)
            unknown_check.toggled.connect(lambda enabled: self.qspace.set_visible_branch(-1, "unknown", enabled))
            self._branch_checks[(-1, "unknown")] = unknown_check
            excluded_row = QtWidgets.QHBoxLayout()
            excluded_row.addWidget(unknown_check)
            self.show_excluded_check = QtWidgets.QCheckBox("Show excluded", branch_group)
            self.show_excluded_check.setObjectName("showExcludedPoints")
            self.show_excluded_check.setFixedHeight(20)
            self.show_excluded_check.setChecked(False)
            self.show_excluded_check.toggled.connect(self.qspace.set_show_excluded)
            excluded_row.addWidget(self.show_excluded_check)
            self.excluded_count_label = QtWidgets.QLabel("0", branch_group)
            self.excluded_count_label.setStyleSheet("color: #56667a;")
            excluded_row.addWidget(self.excluded_count_label)
            excluded_row.addStretch(1)
            branch_layout.addLayout(excluded_row)
            branch_group.setMaximumHeight(132)
            right_layout.addWidget(branch_group)

            quantity_group = QtWidgets.QGroupBox("Quantitative parameters", right_panel)
            quantity_group.setObjectName("butterflyQuantitativeParameters")
            quantity_layout = QtWidgets.QVBoxLayout(quantity_group)
            quantity_layout.setContentsMargins(5, 5, 5, 5)
            self.quantity_table = QtWidgets.QTableWidget(0, 6, quantity_group)
            self.quantity_table.setObjectName("butterflyQuantitativeTable")
            self.quantity_table.setHorizontalHeaderLabels(
                ("Param", "Value", "State", "Cand.", "CI", "Why")
            )
            self.quantity_table.horizontalHeader().setStretchLastSection(True)
            self.quantity_table.horizontalHeader().setMinimumSectionSize(34)
            self.quantity_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
            self.quantity_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
            self.quantity_table.setMinimumHeight(112)
            self.quantity_table.setMaximumHeight(146)
            self.quantity_table.verticalHeader().setDefaultSectionSize(18)
            self.quantity_table.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            for column, width in enumerate((42, 38, 42, 46, 40, 48)):
                self.quantity_table.setColumnWidth(column, width)
            quantity_layout.addWidget(self.quantity_table)
            quantity_group.setMinimumHeight(126)
            right_layout.addWidget(quantity_group)

            correction_group = QtWidgets.QGroupBox("Corrections / 编辑", right_panel)
            correction_group.setObjectName("butterflyCorrections")
            correction_layout = QtWidgets.QFormLayout(correction_group)
            self.correction_mode_combo = QtWidgets.QComboBox(correction_group)
            self.correction_mode_combo.setObjectName("butterflyCorrectionMode")
            self.correction_mode_combo.addItem("Seed point", "seed")
            self.correction_mode_combo.addItem("Exclude point", "exclude_point")
            self.correction_mode_combo.addItem("Exclude rectangle", "rectangle_exclude")
            self.correction_mode_combo.addItem("Include rectangle", "rectangle_include")
            self.correction_mode_combo.addItem("Exclude polygon", "polygon_exclude")
            self.correction_mode_combo.addItem("Include polygon", "polygon_include")
            correction_layout.addRow("Edit mode", self.correction_mode_combo)
            self.seed_branch_combo = QtWidgets.QComboBox(correction_group)
            self.seed_branch_combo.addItem("Auto / unknown", -1)
            self.seed_branch_combo.addItem("A", 0)
            self.seed_branch_combo.addItem("B", 1)
            correction_layout.addRow("Seed branch", self.seed_branch_combo)
            self.seed_side_combo = QtWidgets.QComboBox(correction_group)
            self.seed_side_combo.addItem("Unknown", "unknown")
            self.seed_side_combo.addItem("Upper", "upper")
            self.seed_side_combo.addItem("Lower", "lower")
            correction_layout.addRow("Seed side", self.seed_side_combo)
            edit_buttons = QtWidgets.QHBoxLayout()
            self.correct_button = QtWidgets.QPushButton("Correct / 编辑", correction_group)
            self.correct_button.setObjectName("butterflyCorrectButton")
            self.correct_button.clicked.connect(self.start_correction)
            edit_buttons.addWidget(self.correct_button)
            self.undo_button = QtWidgets.QPushButton("Undo", correction_group)
            self.undo_button.setObjectName("butterflyUndoButton")
            self.undo_button.clicked.connect(self.undo_edit)
            edit_buttons.addWidget(self.undo_button)
            self.redo_button = QtWidgets.QPushButton("Redo", correction_group)
            self.redo_button.setObjectName("butterflyRedoButton")
            self.redo_button.clicked.connect(self.redo_edit)
            edit_buttons.addWidget(self.redo_button)
            correction_layout.addRow(edit_buttons)
            self.numeric_edit_button = QtWidgets.QPushButton("Keyboard edit / 键盘编辑", correction_group)
            self.numeric_edit_button.setObjectName("butterflyNumericEditButton")
            self.numeric_edit_button.clicked.connect(self._open_numeric_edit_dialog)
            correction_layout.addRow(self.numeric_edit_button)
            right_layout.addWidget(correction_group)

            actions_group = QtWidgets.QGroupBox("Workflow", right_panel)
            actions_group.setObjectName("butterflyWorkflow")
            actions_layout = QtWidgets.QVBoxLayout(actions_group)
            actions_layout.setContentsMargins(4, 3, 4, 3)
            actions_layout.setSpacing(2)
            self.identify_button = QtWidgets.QPushButton("Identify arcs / 识别", actions_group)
            self.identify_button.setObjectName("butterflyIdentifyButton")
            self.identify_button.setToolTip("Trace observed arcs using butterfly curvature")
            self.identify_button.clicked.connect(self.request_identify)
            actions_layout.addWidget(self.identify_button)
            self.evaluate_button = QtWidgets.QPushButton("Evaluate / 评估", actions_group)
            self.evaluate_button.setObjectName("butterflyEvaluateButton")
            self.evaluate_button.setToolTip("Evaluate the traced arcs with 32 uncertainty resamples")
            self.evaluate_button.clicked.connect(self.request_evaluate)
            actions_layout.addWidget(self.evaluate_button)
            self.apply_batch_button = QtWidgets.QPushButton("Apply to batch / 应用到批处理", actions_group)
            self.apply_batch_button.setObjectName("butterflyApplyBatchButton")
            self.apply_batch_button.clicked.connect(self.apply_to_batch)
            actions_layout.addWidget(self.apply_batch_button)
            self.cancel_button = QtWidgets.QPushButton("Cancel", actions_group)
            self.cancel_button.setObjectName("butterflyCancelButton")
            self.cancel_button.setShortcut(QtGui.QKeySequence("Esc"))
            self.cancel_button.clicked.connect(lambda _checked=False: self.cancelRequested.emit())
            actions_layout.addWidget(self.cancel_button)
            self.export_button = QtWidgets.QPushButton("Export analysis / 导出分析", actions_group)
            self.export_button.setObjectName("butterflyExportButton")
            self.export_button.clicked.connect(lambda _checked=False: self.exportRequested.emit())
            export_row = QtWidgets.QHBoxLayout()
            export_row.setSpacing(4)
            export_row.addWidget(self.export_button, 1)
            self.figure_export_button = QtWidgets.QPushButton("Measurement figure", actions_group)
            self.figure_export_button.setObjectName("butterflyFigureExportButton")
            self.figure_export_button.clicked.connect(
                lambda _checked=False: self.figureExportRequested.emit()
            )
            export_row.addWidget(self.figure_export_button, 1)
            actions_layout.addLayout(export_row)
            for action_button in (
                self.identify_button,
                self.evaluate_button,
                self.apply_batch_button,
                self.cancel_button,
                self.export_button,
                self.figure_export_button,
            ):
                action_button.setMinimumHeight(30)
                action_button.setMaximumHeight(34)
            actions_group.setMaximumHeight(210)
            right_layout.addWidget(actions_group)
            self.batch_feedback_label = QtWidgets.QLabel(actions_group)
            self.batch_feedback_label.setObjectName("butterflyBatchFeedback")
            self.batch_feedback_label.setWordWrap(True)
            self.batch_feedback_label.setStyleSheet("color: #56667a;")
            actions_layout.addWidget(self.batch_feedback_label)
            right_layout.addStretch(1)
            right_layout.removeWidget(actions_group)
            right_layout.insertWidget(1, actions_group)
            right_scroll.setWidget(right_panel)
            splitter.addWidget(right_scroll)
            splitter.setStretchFactor(0, 5)
            splitter.setStretchFactor(1, 2)

            self.qspace.pointSelected.connect(self._on_point_selected)
            self.qspace.landmarkSelected.connect(self._on_landmark_selected)
            self.qspace.editRequested.connect(self._on_edit_requested)
            self.qspace.set_overlay_mode(str(self.overlay_mode_combo.currentData()))
            self.qspace.set_landmark_visibility(
                global_raw_max=True,
                supported_peaks=True,
            )
            self._update_edit_buttons()
            self.set_language(self._language)
            self._sync_action_state()

        @property
        def butterfly_settings(self) -> dict[str, Any]:
            result = deepcopy(self._settings)
            result["edits"] = deepcopy(self._edits)
            return result

        @property
        def analysis_settings(self) -> dict[str, Any]:
            return {"ridge_method": "butterfly_curvature", "butterfly": self.butterfly_settings}

        @property
        def edits(self) -> list[dict[str, Any]]:
            return deepcopy(self._edits)

        @property
        def current_result(self) -> dict[str, Any]:
            return deepcopy(self._result)

        @property
        def manual_review(self) -> dict[str, Any]:
            """Return the page-local review state tied to this result revision."""

            review = deepcopy(self._manual_review)
            if review.get("result_revision") != self._result_revision or not self._result_fresh:
                review.update(
                    {
                        "manual_status": "unreviewed",
                        "reviewed_by": "",
                        "reviewed_at": None,
                        "review_notes": "",
                        "result_revision": None,
                    }
                )
            return review

        @property
        def result_revision(self) -> int:
            return int(self._result_revision)

        def set_manual_review(self, review: Mapping[str, Any] | None) -> None:
            """Record an explicit page-local review for the current result.

            The shared intensity-fit review session is deliberately not reused
            here: butterfly geometry jobs have a separate result lifecycle.
            """

            values = dict(review or {}) if isinstance(review, Mapping) else {}
            status = str(values.get("manual_status", values.get("status", "unreviewed")) or "unreviewed").lower()
            if status not in {"unreviewed", "accepted", "rejected"}:
                status = "unreviewed"
            revision = values.get("result_revision", self._result_revision)
            try:
                revision = int(revision)
            except (TypeError, ValueError):
                revision = self._result_revision
            if revision != self._result_revision or not self._result_fresh:
                status = "unreviewed"
                revision = None
            self._manual_review = {
                "manual_status": status,
                "reviewed_by": str(values.get("reviewed_by", values.get("reviewer", "")) or ""),
                "reviewed_at": values.get("reviewed_at"),
                "review_notes": str(values.get("review_notes", values.get("notes", "")) or ""),
                "result_revision": revision,
            }

        def _tr(self, key: str, **values: Any) -> str:
            try:
                return translate(self._language, key, **values)
            except (KeyError, ValueError):
                return key

        def _refresh_fit_layers(self) -> None:
            if not self._result_fresh or not self._result:
                self._fit_layers = {}
            else:
                try:
                    self._fit_layers = fit_geometry_layers(
                        self._result,
                        model_parameters=self._model_parameters,
                        model_reference_axis_deg=self._model_reference_axis_deg,
                        model_status=self._model_status,
                        model_diagnostics=self._model_diagnostics,
                    )
                except (TypeError, ValueError, OverflowError):
                    self._fit_layers = {}
            self.qspace.set_fit_layers(self._fit_layers)
            self._render_fit_source_state()
            self._render_fit_assessment()

        def set_model_fit_context(
            self,
            parameters: Any,
            *,
            reference_axis_deg: Any,
            solver_status: Any,
            diagnostics: Mapping[str, Any] | None = None,
        ) -> None:
            """Set the actual current Optimize model geometry for diagnosis.

            MainWindow calls this only for a generation-current Optimize result
            whose diagnostic signature still matches the live fit state.
            Preview arrays and edited/stale parameter tables never enter here.
            """

            if not self._result_fresh:
                self.clear_model_fit_context()
                return
            self._model_parameters = deepcopy(parameters)
            self._model_reference_axis_deg = _finite(reference_axis_deg)
            self._model_status = (
                None if solver_status in (None, "") else str(solver_status)
            )
            self._model_diagnostics = deepcopy(dict(diagnostics or {}))
            self._refresh_fit_layers()
            self._render_page_status()

        def clear_model_fit_context(self) -> None:
            self._model_parameters = None
            self._model_reference_axis_deg = None
            self._model_status = None
            self._model_diagnostics = {}
            self._refresh_fit_layers()

        def _geometry_assessment(self) -> Mapping[str, Any]:
            assessment = _read(self._fit_layers, ("assessment",), {})
            return assessment if isinstance(assessment, Mapping) else {}

        def _poor_geometry_fit(self) -> bool:
            assessment = self._geometry_assessment()
            if str(assessment.get("geometry_status", "")).lower() == "poor_match":
                return True
            ratio = _finite(assessment.get("residual_sigma_ratio"))
            limit = _finite(assessment.get("residual_limit"))
            return ratio is not None and limit is not None and ratio > limit

        def _render_fit_source_state(self) -> None:
            mode = str(self.overlay_mode_combo.currentData() or "measured_only")
            self.qspace.set_overlay_mode(mode)
            geometry = _read(self._fit_layers, ("geometry",), {})
            model = _read(self._fit_layers, ("intensity_model",), {})
            geometry_status = str(_read(geometry, ("status",), "unavailable"))
            model_status = str(_read(model, ("status",), "unavailable"))
            if mode == "measured_only":
                key = "fit.source_measured_only"
                values: dict[str, Any] = {}
            elif mode == "observed_ridges":
                key = "fit.source_observed_ridges"
                values = {}
            elif mode == "geometry_candidate":
                key = "fit.source_geometry"
                values = {"status": geometry_status}
            elif mode == "full2d_model":
                model_curves = _read(model, ("curves",), ())
                if not model_curves:
                    key = "fit.model_unavailable"
                    values = {}
                else:
                    key = "fit.source_model"
                    diagnostics = _read(model, ("diagnostics",), {})
                    condition = _finite(_read(diagnostics, ("condition_number",), None))
                    rmse = _finite(_read(diagnostics, ("rmse",), None))
                    bound_flags = _read(diagnostics, ("bound_flags",), {})
                    bound_names = (
                        ", ".join(
                            str(name)
                            for name, active in bound_flags.items()
                            if active
                        )
                        if isinstance(bound_flags, Mapping)
                        else ""
                    )
                    detail_parts = []
                    if rmse is not None:
                        detail_parts.append(f"RMSE {rmse:.3g}")
                    if condition is not None:
                        detail_parts.append(f"condition {condition:.3g}")
                    if bound_names:
                        detail_parts.append(f"bound flags: {bound_names}")
                    values = {
                        "status": model_status,
                        "details": "; ".join(detail_parts) or "diagnostics unavailable",
                    }
            else:
                key = "fit.source_compare"
                diagnostics = _read(model, ("diagnostics",), {})
                condition = _finite(_read(diagnostics, ("condition_number",), None))
                rmse = _finite(_read(diagnostics, ("rmse",), None))
                bound_flags = _read(diagnostics, ("bound_flags",), {})
                bound_names = (
                    ", ".join(
                        str(name)
                        for name, active in bound_flags.items()
                        if active
                    )
                    if isinstance(bound_flags, Mapping)
                    else ""
                )
                detail_parts = []
                if rmse is not None:
                    detail_parts.append(f"RMSE {rmse:.3g}")
                if condition is not None:
                    detail_parts.append(f"condition {condition:.3g}")
                if bound_names:
                    detail_parts.append(f"bound flags: {bound_names}")
                values = {
                    "geometry_status": geometry_status,
                    "model_status": model_status,
                    "details": "; ".join(detail_parts) or "diagnostics unavailable",
                }
            self.fit_source_label.setText(self._tr(key, **values))

        def _render_fit_assessment(self) -> None:
            assessment = self._geometry_assessment()
            if not assessment:
                self.fit_assessment_label.clear()
                self.fit_assessment_label.setAccessibleName("")
                self.fit_assessment_label.setVisible(False)
                return
            if self._poor_geometry_fit():
                ratio = _finite(assessment.get("residual_sigma_ratio"))
                limit = _finite(assessment.get("residual_limit"))
                text = self._tr(
                    "fit.geometry_poor",
                    ratio=_fmt(ratio),
                    limit=_fmt(limit),
                )
                self.fit_assessment_label.setStyleSheet(
                    "color: #9f1d16; background: #fff0ee; padding: 5px; font-weight: 700;"
                )
            else:
                status = str(assessment.get("geometry_status", "unavailable"))
                text = self._tr("fit.geometry_status", status=status)
                self.fit_assessment_label.setStyleSheet(
                    "color: #664d03; background: #fff8dc; padding: 5px;"
                )
            self._set_dynamic_accessible_text(self.fit_assessment_label, text)
            self.fit_assessment_label.setVisible(True)

        @staticmethod
        def _set_dynamic_accessible_text(label: Any, text: str) -> None:
            label.setText(text)
            label.setAccessibleName(text)
            try:
                QtGui.QAccessible.updateAccessibility(
                    QtGui.QAccessibleEvent(label, QtGui.QAccessible.Event.NameChanged)
                )
            except (AttributeError, TypeError):
                pass

        def _clear_diagnostic_layers(self) -> None:
            self._peak_landmarks = {}
            self._fit_layers = {}
            self._model_parameters = None
            self._model_reference_axis_deg = None
            self._model_status = None
            self._model_diagnostics = {}
            self._selected_landmark = {}
            self._selected_landmark_id = None
            self.qspace.set_peak_landmarks(None)
            self.qspace.set_selected_landmark(None)
            self.qspace.set_fit_layers(None)
            self.peak_table.blockSignals(True)
            self.peak_table.setRowCount(0)
            self.peak_table.clearSelection()
            self.peak_table.blockSignals(False)
            self.peak_angular_profile.clear(message_key="landmark.no_profiles")
            self.peak_radial_profile.clear(message_key="landmark.no_profiles")
            self.diagnostics_tabs.setCurrentIndex(0)
            self._reset_landmark_zoom()
            self._render_fit_source_state()
            self.fit_assessment_label.clear()
            self.fit_assessment_label.setAccessibleName("")
            self.fit_assessment_label.setVisible(False)

        def _render_peak_table(self) -> None:
            self.peak_table.blockSignals(True)
            self.peak_table.setRowCount(0)
            q_unit = str(
                self._peak_landmarks.get("q_unit")
                or self._frame_data.get("q_unit")
                or "q"
            )
            headers = (
                self._tr("header.landmark_id"),
                f"{self._tr('header.landmark_q')} ({q_unit})",
                self._tr("header.landmark_intensity"),
                self._tr("header.landmark_kind_status"),
            )
            self.peak_table.setHorizontalHeaderLabels(headers)
            english = self._language.lower().startswith("en")
            self.peak_table.setAccessibleName(
                "Peak landmark diagnostics" if english else "峰位标记诊断"
            )
            self.peak_table.setAccessibleDescription(
                "Global raw maximum and supported lobe peaks; selecting a row focuses its actual q coordinate"
                if english
                else "全局原始最大值与受支持瓣峰；选择行后聚焦该记录的实际 q 坐标"
            )
            rows: list[tuple[str, str, Mapping[str, Any]]] = []
            raw_max = self._peak_landmarks.get("raw_global_max")
            if isinstance(raw_max, Mapping):
                rows.append(("G", "raw_global_max", raw_max))
            peaks = self._peak_landmarks.get("peaks", ())
            if isinstance(peaks, Sequence) and not isinstance(peaks, (str, bytes)):
                for index, peak in enumerate(peaks):
                    if not isinstance(peak, Mapping):
                        continue
                    peak_id = str(_read(peak, ("peak_id",), f"P{index + 1}") or "")
                    if peak_id:
                        rows.append((peak_id, "supported_peak", peak))
            for landmark_id, kind, record in rows:
                row = self.peak_table.rowCount()
                self.peak_table.insertRow(row)
                if kind == "raw_global_max":
                    kind_text = self._tr("landmark.kind_raw_max")
                    status_text = self._tr("landmark.status_raw_only")
                else:
                    kind_text = self._tr("landmark.kind_supported")
                    flags = _read(record, ("flags",), ())
                    if isinstance(flags, str):
                        flags = (flags,) if flags else ()
                    flag_text = ", ".join(str(flag) for flag in (flags or ()) if flag)
                    status_text = (
                        self._tr("landmark.status_flags", flags=flag_text)
                        if flag_text
                        else self._tr("landmark.status_supported")
                    )
                values = (
                    landmark_id,
                    _fmt(_read(record, ("q",), None)),
                    _fmt(_read(record, ("raw_intensity",), None)),
                    f"{kind_text} · {status_text}",
                )
                for column, value in enumerate(values):
                    item = QtWidgets.QTableWidgetItem(value)
                    item.setData(
                        QtCore.Qt.ItemDataRole.UserRole,
                        {
                            "landmark_id": landmark_id,
                            "kind": kind,
                            "record": dict(record),
                        },
                    )
                    item.setToolTip(
                        str(record.get("interpretation"))
                        if kind == "raw_global_max"
                        else str(record.get("flags", ""))
                    )
                    self.peak_table.setItem(row, column, item)
            self.peak_table.blockSignals(False)

        def _on_landmark_visibility_changed(self, *_: Any) -> None:
            self.qspace.set_landmark_visibility(
                global_raw_max=self.global_max_check.isChecked(),
                supported_peaks=self.supported_peaks_check.isChecked(),
            )

        def _on_overlay_mode_changed(self, *_: Any) -> None:
            self._render_fit_source_state()

        def _peak_table_row(self, landmark_id: str) -> int | None:
            for row in range(self.peak_table.rowCount()):
                item = self.peak_table.item(row, 0)
                payload = item.data(QtCore.Qt.ItemDataRole.UserRole) if item else None
                if isinstance(payload, Mapping) and payload.get("landmark_id") == landmark_id:
                    return row
            return None

        def _on_landmark_selected(self, record: Any) -> None:
            if not isinstance(record, Mapping):
                return
            landmark_id = str(_read(record, ("peak_id",), "G") or "G")
            raw_max = self._peak_landmarks.get("raw_global_max")
            if record is raw_max:
                landmark_id = "G"
            row = self._peak_table_row(landmark_id)
            if row is None:
                return
            self.peak_table.setCurrentCell(row, 0)
            self._focus_landmark_row(row)

        def _on_peak_table_selection_changed(
            self,
            row: int,
            _column: int,
            _previous_row: int,
            _previous_column: int,
        ) -> None:
            self._focus_landmark_row(row)

        def _focus_landmark_row(self, row: int) -> None:
            if row < 0:
                return
            item = self.peak_table.item(row, 0)
            payload = item.data(QtCore.Qt.ItemDataRole.UserRole) if item else None
            if not isinstance(payload, Mapping):
                return
            record = payload.get("record")
            if not isinstance(record, Mapping):
                return
            landmark_id = str(payload.get("landmark_id", ""))
            self._selected_landmark = dict(record)
            self._selected_landmark_id = landmark_id
            self.qspace.set_selected_landmark(landmark_id)
            self._focus_landmark_q(record)
            self._render_peak_profiles(record, landmark_id)
            self.diagnostics_tabs.setCurrentIndex(1)

        def _focus_landmark_q(self, record: Mapping[str, Any]) -> None:
            qx = _finite(_read(record, ("qx",), None))
            qy = _finite(_read(record, ("qy",), None))
            if qx is None or qy is None:
                return
            radius = _finite(_read(record, ("q",), None))
            if radius is None:
                radius = math.hypot(qx, qy)
            bounds = self.qspace.base_q_bounds
            if bounds is None:
                return
            span = max(bounds[1] - bounds[0], bounds[3] - bounds[2])
            delta = max(abs(radius) * 0.12, span * 0.05, 1e-12)
            low = max(0.0, radius - delta)
            high = radius + delta
            if high <= low:
                high = low + max(delta, 1e-12)
            self.qspace.set_q_window((low, high))
            self._landmark_zoomed = True
            self.reset_peak_zoom_button.setEnabled(True)

        def _reset_landmark_zoom(self) -> None:
            self.qspace.set_q_window(self._requested_q_window)
            self._landmark_zoomed = False
            if hasattr(self, "reset_peak_zoom_button"):
                self.reset_peak_zoom_button.setEnabled(False)

        def _render_peak_profiles(self, record: Mapping[str, Any], landmark_id: str) -> None:
            profiles = _read(self._peak_landmarks, ("profiles",), {})
            if not isinstance(profiles, Mapping):
                profiles = {}
            angular = _read(profiles, ("angular",), {})
            radial = _read(profiles, ("radial",), {})
            if not isinstance(angular, Mapping):
                angular = {}
            if not isinstance(radial, Mapping):
                radial = {}
            angle = _finite(
                _read(record, ("angular_peak_deg", "chi_deg"), None)
            )
            qx = _finite(_read(record, ("qx",), None))
            qy = _finite(_read(record, ("qy",), None))
            if angle is None and qx is not None and qy is not None:
                angle = math.degrees(math.atan2(qy, qx))
            radius = _finite(_read(record, ("q",), None))
            if radius is None and qx is not None and qy is not None:
                radius = math.hypot(qx, qy)
            angular_x, angular_raw = self._series(
                angular, ("angle_deg",), ("intensity_raw",)
            )
            _, angular_smoothed = self._series(
                angular, ("angle_deg",), ("intensity_smoothed",)
            )
            _, angular_reference = self._series(
                angular, ("angle_deg",), ("intensity_isotropic_reference",)
            )
            _, angular_detection = self._series(
                angular, ("angle_deg",), ("intensity_detection",)
            )
            radial_x, radial_raw = self._series(
                radial, ("q",), ("mean_intensity_raw",)
            )
            _, radial_smoothed = self._series(
                radial, ("q",), ("mean_intensity_smoothed",)
            )
            angular_count = min(
                len(angular_x),
                len(angular_raw),
                len(angular_smoothed) if angular_smoothed else len(angular_x),
            )
            radial_count = min(
                len(radial_x),
                len(radial_raw),
                len(radial_smoothed) if radial_smoothed else len(radial_x),
            )
            if angular_count:
                angular_series = {"raw": angular_raw[:angular_count]}
                if angular_smoothed:
                    angular_series["smoothed"] = angular_smoothed[:angular_count]
                if angular_reference:
                    angular_series["isotropic_reference"] = (
                        angular_reference[:angular_count]
                        + [math.nan] * max(0, angular_count - len(angular_reference))
                    )
                if angular_detection:
                    angular_series["detection"] = (
                        angular_detection[:angular_count]
                        + [math.nan] * max(0, angular_count - len(angular_detection))
                    )
                self.peak_angular_profile._title = "Angular peak signal"
                self.peak_angular_profile.title_label.setText(
                    self._tr("profile.angular_selected", id=landmark_id)
                )
                self.peak_angular_profile.set_series(
                    angular_x[:angular_count],
                    angular_series,
                    x_label="chi (deg)" if self._language.lower().startswith("en") else "方位角 chi（deg）",
                    y_label="intensity" if self._language.lower().startswith("en") else "强度",
                    markers=[] if angle is None else [(angle, landmark_id)],
                )
            else:
                self.peak_angular_profile.clear(message_key="landmark.no_profiles")
            if radial_count:
                radial_series = {"raw": radial_raw[:radial_count]}
                if radial_smoothed:
                    radial_series["smoothed"] = radial_smoothed[:radial_count]
                self.peak_radial_profile._title = "Radial peak signal"
                self.peak_radial_profile.title_label.setText(
                    self._tr("profile.radial_selected", id=landmark_id)
                )
                self.peak_radial_profile.set_series(
                    radial_x[:radial_count],
                    radial_series,
                    x_label=(
                        f"q ({self._peak_landmarks.get('q_unit') or 'q'})"
                        if self._language.lower().startswith("en")
                        else f"q（{self._peak_landmarks.get('q_unit') or 'q'}）"
                    ),
                    y_label="intensity" if self._language.lower().startswith("en") else "强度",
                    markers=[] if radius is None else [(radius, landmark_id)],
                )
            else:
                self.peak_radial_profile.clear(message_key="landmark.no_profiles")

        def _render_frame_source(self) -> None:
            frame = self._current_frame
            if frame is None or not str(frame):
                self.frame_source_label.setText(
                    "in-memory frame" if self._language.lower().startswith("en") else "内存帧"
                )
                return
            self.frame_source_label.setText(Path(str(frame)).name or str(frame))

        def _render_magnification_label(self) -> None:
            value = _fmt(self._diagnostic_magnification)
            self.magnification_label.setText(
                f"Display magnification: ×{value} (diagnostic display only)"
                if self._language.lower().startswith("en")
                else f"显示放大倍数：×{value}（仅用于诊断显示）"
            )

        def _data_ready(self) -> bool:
            observed = getattr(self.qspace, "observed", None)
            return bool(self._has_loaded_data and observed is not None)

        def _sync_action_state(self) -> None:
            ready = self._data_ready()
            busy = bool(self._busy)
            valid_range = not bool(self._q_range_error)
            self.identify_button.setEnabled(ready and valid_range and not busy)
            self.evaluate_button.setEnabled(ready and valid_range and not busy)
            self.apply_batch_button.setEnabled(ready and valid_range and not busy)
            self.cancel_button.setEnabled(busy)
            self.export_button.setEnabled(bool(self._result_fresh and not busy))
            self.figure_export_button.setEnabled(bool(self._result_fresh and not busy))

        @staticmethod
        def _result_is_failed(result: Mapping[str, Any]) -> bool:
            for key in ("status", "measurement_status", "solver_status", "quality_status"):
                value = str(result.get(key, "") or "").strip().lower()
                if value in {"fail", "failed", "error", "invalid"}:
                    return True
            metrics = result.get("metrics", result.get("statistics", result.get("summary", {})))
            if isinstance(metrics, Mapping) and metrics.get("success") is False:
                return True
            return False

        def _render_page_status(self) -> None:
            """Render the retained readiness/job/result state in the active language."""

            english = self._language.lower().startswith("en")
            state = self._page_status_state
            kind = self._page_status_kind or "analysis"
            kind_label = self._tr(f"job.{kind}")
            if kind_label == f"job.{kind}":
                kind_label = kind
            if state == "empty":
                text = (
                    "Load a frame before Identify or Evaluate"
                    if english
                    else "请先载入图像，再识别或评估"
                )
            elif state == "running":
                elapsed = max(0.0, float(self._job_elapsed_s or 0.0))
                text = self._tr("workflow.running", kind=kind_label, elapsed_s=elapsed)
                if self._job_progress_percent is not None:
                    text += self._tr(
                        "workflow.progress_suffix",
                        percent=self._job_progress_percent,
                        phase=self._job_progress_phase,
                    )
            elif state == "cancelling":
                elapsed = max(0.0, float(self._job_elapsed_s or 0.0))
                text = self._tr("workflow.cancelling", kind=kind_label, elapsed_s=elapsed)
            elif state == "cancelled":
                text = "Cancelled" if english else "已取消"
            elif state == "failed":
                detail = str(self._page_status_error or "").strip()
                visible_detail = detail if len(detail) <= 100 else detail[:97] + "…"
                suffix = f": {visible_detail}" if visible_detail else ""
                text = f"Failed · {kind_label}{suffix}" if english else f"失败 · {kind_label}{suffix}"
            elif state == "result":
                count = len(self._result.get("points", []) or [])
                stage = self._settings.get("stage", "trace")
                text = (
                    f"Result · {stage} · {count} points"
                    if english
                    else f"结果 · {'评估' if stage == 'evaluate' else '追踪'} · {count} 个点"
                )
            elif state == "completed":
                count = len(self._result.get("points", []) or [])
                text = (
                    f"Completed · {kind_label} · {count} points"
                    if english
                    else f"已完成 · {kind_label} · {count} 个点"
                )
            else:
                if english:
                    text = "Ready · identify arcs" if self._data_ready() else "Load a frame to begin"
                else:
                    text = "就绪 · 请识别弧线" if self._data_ready() else "请先载入一帧图像"
            self.status_label.setText(text)
            if state in {"result", "completed"} and self._poor_geometry_fit():
                self.status_label.setText(
                    f"{text} · {self._tr('fit.poor_short')}"
                )
            self.status_label.setToolTip(
                str(self._page_status_error or "") if state == "failed" else ""
            )
            self._render_workflow_hint()

        def _render_workflow_hint(self) -> None:
            """Show the next useful operation without implying scientific acceptance."""

            if self._q_range_error:
                key = "workflow.fix_q_range"
                values: dict[str, Any] = {}
            elif self._page_status_state in {"running", "cancelling"}:
                key = "workflow.running_hint"
                values = {}
            elif not self._data_ready():
                key = "workflow.next_load"
                values = {}
            elif self._page_status_state == "failed":
                key = "workflow.next_retry"
                values = {}
            elif self._result_fresh:
                if self._settings.get("stage", "trace") == "trace":
                    key = "workflow.next_evaluate"
                    values = {"resamples": self._evaluation_resamples()}
                else:
                    key = "workflow.next_review"
                    values = {}
            else:
                key = "workflow.next_identify"
                values = {}
            self._set_dynamic_accessible_text(
                self.workflow_hint_label, self._tr(key, **values)
            )

        def set_language(self, language: str) -> None:
            self._language = str(language)
            english = self._language.lower().startswith("en")
            self.qspace.set_language(self._language)
            self.title_label.setText("Butterfly analysis" if english else "蝴蝶分析 / Butterfly analysis")
            self.method_label.setText(
                "Curvature ridge" if english else "论文曲率脊线"
            )
            self.frame_title_label.setText("Frames" if english else "帧 / Frames")
            self.point_title_label.setText("Points" if english else "点 / Points")
            self.frame_list.setAccessibleName("Frames" if english else "帧列表")
            self.frame_list.setAccessibleDescription(
                "Select the active SAXS frame" if english else "选择当前 SAXS 帧"
            )
            self.point_list.setAccessibleName("Points" if english else "测量点")
            self.point_list.setAccessibleDescription(
                "Select a measured butterfly point" if english else "选择一个蝴蝶测量点"
            )
            self._render_frame_source()
            analysis_group = self.findChild(QtWidgets.QGroupBox, "butterflyAnalysisRange")
            analysis_group.setTitle("Analysis range" if english else "分析范围")
            analysis_form = analysis_group.layout()
            analysis_form.labelForField(self.q_min_edit).setText("q min" if english else "q 下限")
            analysis_form.labelForField(self.q_max_edit).setText("q max" if english else "q 上限")
            analysis_form.labelForField(self.reference_axis_spin).setText(
                "Reference axis (deg)" if english else "图样参考轴（deg）"
            )
            evaluation_group = self.findChild(QtWidgets.QGroupBox, "butterflyEvaluationControls")
            evaluation_group.setTitle("Evaluation" if english else "评估设置")
            evaluation_form = evaluation_group.layout()
            evaluation_form.labelForField(self.evaluation_resamples_combo).setText(
                "Uncertainty resamples" if english else "不确定度重采样次数"
            )
            self.sensitivity_check.setText(
                "Run sensitivity checks" if english else "运行敏感性检查"
            )
            for index in range(self.evaluation_resamples_combo.count()):
                value = int(self.evaluation_resamples_combo.itemData(index))
                if value == 0:
                    preset = "Quick fit" if english else "快速拟合"
                    label = self._tr("evaluation.budget", preset=preset, count=value)
                elif value == 32:
                    preset = "Standard" if english else "标准"
                    label = self._tr("evaluation.budget", preset=preset, count=value)
                elif value == 128:
                    preset = "Extended" if english else "扩展"
                    label = self._tr("evaluation.budget", preset=preset, count=value)
                else:
                    label = self._tr("evaluation.custom", count=value)
                self.evaluation_resamples_combo.setItemText(index, label)
            self._sync_evaluation_controls()
            self.findChild(QtWidgets.QGroupBox, "butterflyFitOverlays").setTitle(
                self._tr("group.fit_overlays")
            )
            overlay_form = self.findChild(
                QtWidgets.QGroupBox, "butterflyFitOverlays"
            ).layout().itemAt(0).layout()
            overlay_form.labelForField(self.overlay_mode_combo).setText(
                self._tr("label.overlay_mode")
            )
            overlay_keys = (
                ("overlay.measured_only", "measured_only"),
                ("overlay.observed_ridges", "observed_ridges"),
                ("overlay.geometry_candidate", "geometry_candidate"),
                ("overlay.full2d_model", "full2d_model"),
                ("overlay.compare", "compare"),
            )
            for key, mode in overlay_keys:
                index = self.overlay_mode_combo.findData(mode)
                if index >= 0:
                    self.overlay_mode_combo.setItemText(index, self._tr(key))
            self.findChild(QtWidgets.QGroupBox, "butterflyPeakLandmarks").setTitle(
                self._tr("group.peak_landmarks")
            )
            self.global_max_check.setText(self._tr("check.raw_global_max"))
            self.supported_peaks_check.setText(self._tr("check.supported_peaks"))
            self.reset_peak_zoom_button.setText(self._tr("button.reset_peak_zoom"))
            previous_landmark = self._selected_landmark_id
            self._render_peak_table()
            if previous_landmark:
                selected_row = self._peak_table_row(previous_landmark)
                if selected_row is not None:
                    self.peak_table.setCurrentCell(selected_row, 0)
            self.normal_profile.set_language(english=english)
            self.ellipse_diagnostic.set_language(english=english)
            self.peak_angular_profile.set_language(english=english)
            self.peak_radial_profile.set_language(english=english)
            self.diagnostics_tabs.setTabText(
                0, "Point diagnostics" if english else "测量点诊断"
            )
            self.diagnostics_tabs.setTabText(
                1, "Peak profiles" if english else "峰位剖面"
            )
            self._render_peak_profiles(
                self._selected_landmark, self._selected_landmark_id
            ) if self._selected_landmark and self._selected_landmark_id else None
            self.findChild(QtWidgets.QGroupBox, "butterflyDisplayControls").setTitle(
                "Display" if english else "显示"
            )
            display_form = self.findChild(QtWidgets.QGroupBox, "butterflyDisplayControls").layout()
            display_form.labelForField(self.display_scale_combo).setText("Scale" if english else "对比度")
            display_form.labelForField(self.display_percentile_spin).setText(
                "Upper percentile" if english else "上分位数"
            )
            display_form.labelForField(self.uv_magnification_spin).setText(
                "u/v magnification" if english else "u/v 显示放大"
            )
            self.display_reset_button.setText("Reset contrast" if english else "重置对比度")
            self.findChild(QtWidgets.QGroupBox, "branchSideVisibility").setTitle(
                "Branch / side visibility" if english else "分支 / 侧边可见性"
            )
            labels = {
                (0, "upper"): "A · upper" if english else "A · 上侧",
                (0, "lower"): "A · lower" if english else "A · 下侧",
                (1, "upper"): "B · upper" if english else "B · 上侧",
                (1, "lower"): "B · lower" if english else "B · 下侧",
            }
            for key, check in self._branch_checks.items():
                if key in labels:
                    check.setText(labels[key])
                elif key == (-1, "unknown"):
                    check.setText("Unknown" if english else "未知")
            self.show_excluded_check.setText("Show excluded" if english else "显示排除点")
            self.findChild(QtWidgets.QGroupBox, "butterflyQuantitativeParameters").setTitle(
                "Quantitative parameters" if english else "定量参数"
            )
            self.quantity_table.setHorizontalHeaderLabels(
                ("Param", "Value", "State", "Cand.", "CI", "Why")
                if english
                else ("参数", "数值", "状态", "候选", "区间", "原因")
            )
            self.quantity_table.setToolTip(
                "Candidate values are cached/unvalidated and are not quantitative values."
                if english
                else "候选值为缓存且未验证，仅供诊断，不等同于定量值。"
            )
            self.findChild(QtWidgets.QGroupBox, "butterflyCorrections").setTitle(
                "Corrections" if english else "校正 / 编辑"
            )
            self.findChild(QtWidgets.QGroupBox, "butterflyWorkflow").setTitle(
                "Workflow" if english else "工作流"
            )
            correction_text = (
                ("Seed point", "Exclude point", "Exclude rectangle", "Include rectangle", "Exclude polygon", "Include polygon")
                if english
                else ("种子点", "排除点", "排除矩形", "包含矩形", "排除多边形", "包含多边形")
            )
            for index, text in enumerate(correction_text):
                self.correction_mode_combo.setItemText(index, text)
            self.seed_branch_combo.setItemText(0, "Auto / unknown" if english else "自动 / 未知")
            self.seed_branch_combo.setItemText(1, "A" if english else "A 分支")
            self.seed_branch_combo.setItemText(2, "B" if english else "B 分支")
            self.seed_side_combo.setItemText(0, "Unknown" if english else "未知")
            self.seed_side_combo.setItemText(1, "Upper" if english else "上侧")
            self.seed_side_combo.setItemText(2, "Lower" if english else "下侧")
            self.correct_button.setText("Correct" if english else "校正 / 编辑")
            self.undo_button.setText("Undo" if english else "撤销")
            self.redo_button.setText("Redo" if english else "重做")
            self.numeric_edit_button.setText("Keyboard edit" if english else "键盘编辑")
            correction_form = self.findChild(QtWidgets.QGroupBox, "butterflyCorrections").layout()
            correction_form.labelForField(self.correction_mode_combo).setText(
                "Edit mode" if english else "编辑模式"
            )
            correction_form.labelForField(self.seed_branch_combo).setText(
                "Seed branch" if english else "种子分支"
            )
            correction_form.labelForField(self.seed_side_combo).setText(
                "Seed side" if english else "种子侧边"
            )
            self.identify_button.setText("Identify arcs" if english else "识别弧线 / Identify")
            self.evaluate_button.setText("Evaluate" if english else "评估 / Evaluate")
            self.apply_batch_button.setText("Apply to batch" if english else "应用到批处理 / Apply")
            self.cancel_button.setText("Cancel" if english else "取消 / Cancel")
            self.export_button.setText("Export evidence" if english else "导出分析证据")
            self.figure_export_button.setText(self._tr("button.butterfly_figure"))
            self.figure_export_button.setToolTip(self._tr("tooltip.butterfly_figure"))
            self.evaluate_button.setToolTip(
                self._tr("tooltip.butterfly_evaluate", resamples=self._evaluation_resamples())
            )
            _, _, q_error = self._parse_analysis_q_range()
            self._q_range_error = q_error or ""
            self._render_q_range_feedback()
            self._render_fit_source_state()
            self._render_fit_assessment()
            self.normal_profile.empty_label.setText(
                "Select a measured point" if english else "请选择测量点"
            )
            self.ellipse_diagnostic.empty_label.setText(
                "Select a measured point" if english else "请选择测量点"
            )
            self._render_magnification_label()
            self._render_page_status()
            self._sync_action_state()

        def eventFilter(self, watched: Any, event: Any) -> bool:  # noqa: N802 - Qt API
            if watched is getattr(self, "point_list", None) and event.type() == QtCore.QEvent.Type.KeyPress:
                if event.key() == QtCore.Qt.Key.Key_Delete:
                    self._exclude_selected_point()
                    return True
                if event.key() in (QtCore.Qt.Key.Key_Return, QtCore.Qt.Key.Key_Enter):
                    self._on_point_list_changed(self.point_list.currentRow())
                    return True
            return super().eventFilter(watched, event)

        def set_frames(self, frames: Sequence[Any] | None, *, current: Any = None) -> None:
            self._frames = list(frames or ())
            self.frame_list.blockSignals(True)
            self.frame_list.clear()
            for frame in self._frames:
                item = QtWidgets.QListWidgetItem(str(frame))
                item.setData(QtCore.Qt.ItemDataRole.UserRole, frame)
                self.frame_list.addItem(item)
            self.frame_list.blockSignals(False)
            self.frame_count_label.setText(str(len(self._frames)))
            if current is not None:
                for index, frame in enumerate(self._frames):
                    if str(frame) == str(current):
                        self.frame_list.setCurrentRow(index)
                        break
            elif self._frames and self.frame_list.currentRow() < 0:
                self.frame_list.setCurrentRow(0)

        def set_current_frame(self, frame: Any) -> None:
            self._current_frame = frame
            self._render_frame_source()
            if frame is None:
                return
            for index, value in enumerate(self._frames):
                if str(value) == str(frame):
                    self.frame_list.setCurrentRow(index)
                    return

        def _on_frame_changed(self, row: int) -> None:
            if row < 0 or row >= len(self._frames):
                return
            frame = self._frames[row]
            if str(frame) != str(self._current_frame):
                self.invalidate_result()
            self.set_current_frame(frame)
            self.frameSelected.emit(frame)

        def set_data(
            self,
            observed: Any = None,
            *,
            qx: Any = None,
            qy: Any = None,
            valid_mask: Any = None,
            q_unit: str | None = None,
            source: Any = None,
        ) -> None:
            self._frame_data = {
                "observed": observed,
                "qx": qx,
                "qy": qy,
                "valid_mask": valid_mask,
                "q_unit": q_unit,
            }
            self.set_current_frame(source)
            self.qspace.set_data(observed, qx=qx, qy=qy, valid_mask=valid_mask, q_unit=q_unit)
            self.qspace.set_q_window(self._requested_q_window)
            self._has_loaded_data = getattr(self.qspace, "observed", None) is not None
            # A frame is a new scientific input.  Remove every result/selection
            # that could otherwise be painted over the new q-map.
            self.clear_result(
                message=(
                    "Ready · identify arcs"
                    if self._has_loaded_data and self._language.lower().startswith("en")
                    else "就绪 · 请识别弧线"
                    if self._has_loaded_data
                    else "Load a frame to begin"
                    if self._language.lower().startswith("en")
                    else "请先载入一帧图像"
                )
            )
            self._page_status_state = "ready" if self._has_loaded_data else "empty"
            self._render_page_status()
            self._sync_action_state()

        set_observed_data = set_data

        def set_display_settings(self, scale: str, percentile: float) -> None:
            # Contrast is intentionally display-only and does not emit an edit
            # or an analysis request.
            mode = str(scale or "linear").strip().lower().replace("-", "_")
            index = self.display_scale_combo.findData(mode)
            self.display_scale_combo.blockSignals(True)
            self.display_percentile_spin.blockSignals(True)
            try:
                self.display_scale_combo.setCurrentIndex(max(0, index))
                self.display_percentile_spin.setValue(float(percentile))
            finally:
                self.display_scale_combo.blockSignals(False)
                self.display_percentile_spin.blockSignals(False)
            self.qspace.set_display_settings(scale, percentile)

        @property
        def result_fresh(self) -> bool:
            return bool(self._result_fresh)

        @property
        def export_context(self) -> dict[str, Any]:
            return deepcopy(self._export_context)

        @property
        def display_settings(self) -> dict[str, Any]:
            return {
                "scale": str(self.display_scale_combo.currentData() or "linear"),
                "percentile": float(self.display_percentile_spin.value()),
            }

        def set_export_context(self, context: Mapping[str, Any] | None) -> None:
            self._export_context = deepcopy(dict(context or {}))

        def export_analysis(self, path: str | Path) -> dict[str, Path]:
            return export_butterfly_analysis(self, path, context=self._export_context)

        def figure_export_snapshot(
            self,
            *,
            context: Mapping[str, Any] | None = None,
        ) -> dict[str, Any]:
            """Copy the current physical image/result inputs for a worker export."""

            if not self._result_fresh or not self._result:
                raise ValueError("a fresh butterfly result is required for figure export")
            if _np is None:
                raise RuntimeError("NumPy is required for measurement figure export")

            def frozen_array(name: str, value: Any, *, optional: bool = False) -> Any:
                if value is None:
                    if optional:
                        return None
                    raise ValueError(f"measurement figure export requires {name}")
                try:
                    array = _np.array(value, copy=True, subok=True)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"measurement figure input {name} is not an array") from exc
                if array.ndim != 2 or not array.size:
                    raise ValueError(f"measurement figure input {name} must be a non-empty 2D array")
                array.setflags(write=False)
                return array

            observed = frozen_array("observed", self._frame_data.get("observed"))
            qx = frozen_array("qx", self._frame_data.get("qx"))
            qy = frozen_array("qy", self._frame_data.get("qy"))
            mask = frozen_array("valid_mask", self._frame_data.get("valid_mask"), optional=True)
            if qx.shape != observed.shape or qy.shape != observed.shape:
                raise ValueError("observed, qx and qy arrays must have the same 2D shape")
            if mask is not None and mask.shape != observed.shape:
                raise ValueError("valid_mask must have the same 2D shape as observed")

            result = deepcopy(self._result)

            def freeze_nested(value: Any) -> None:
                if isinstance(value, _np.ndarray):
                    value.setflags(write=False)
                elif isinstance(value, Mapping):
                    for item in value.values():
                        freeze_nested(item)
                elif isinstance(value, (list, tuple)):
                    for item in value:
                        freeze_nested(item)

            freeze_nested(result)
            return {
                "observed": observed,
                "qx": qx,
                "qy": qy,
                "valid_mask": mask,
                "result": result,
                "q_unit": str(self._frame_data.get("q_unit") or "unknown"),
                "display_scale": self.display_settings["scale"],
                "context": deepcopy(dict(context or self._export_context)),
                "result_revision": self._result_revision,
            }

        def _sync_export_state(self) -> None:
            self._sync_action_state()

        def invalidate_result(self) -> None:
            """Remove every displayed measurement derived from a stale input."""

            self.clear_result()

        def _on_analysis_range_changed(self, *_: Any) -> None:
            q_min, q_max, error = self._parse_analysis_q_range()
            self._q_range_error = error or ""
            self._render_q_range_feedback()
            self._sync_action_state()
            if error:
                self._render_page_status()
                return
            values = (q_min, q_max, float(self.reference_axis_spin.value()))
            if values == self._last_valid_analysis_range:
                self._render_page_status()
                return
            self._last_valid_analysis_range = values
            self.invalidate_result()
            self.analysisChanged.emit(
                {
                    "q_min": q_min,
                    "q_max": q_max,
                    "draw_axis_deg": float(self.reference_axis_spin.value()) + 90.0,
                }
            )

        def _on_display_changed(self, *_: Any) -> None:
            scale = str(self.display_scale_combo.currentData() or "linear")
            percentile = float(self.display_percentile_spin.value())
            self.qspace.set_display_settings(scale, percentile)
            self.displayChanged.emit(scale, percentile)

        def _on_uv_magnification_changed(self, value: float) -> None:
            self._uv_display_magnification = float(value)
            self._render_ellipse_local()

        def _reset_display(self) -> None:
            self.set_display_settings("linear", 99.5)
            self.displayChanged.emit("linear", 99.5)

        def _parse_analysis_q_range(
            self,
        ) -> tuple[float | None, float | None, str | None]:
            values: list[float | None] = []
            for key, edit in (("label.q_min", self.q_min_edit), ("label.q_max", self.q_max_edit)):
                text = edit.text().strip()
                if text.lower() in {"", "auto", "自动"}:
                    values.append(None)
                    continue
                try:
                    value = float(text)
                except (TypeError, ValueError):
                    return None, None, self._tr("workflow.q_range_number", field=self._tr(key))
                if not math.isfinite(value):
                    return None, None, self._tr("workflow.q_range_number", field=self._tr(key))
                values.append(value)
            q_min, q_max = values
            if q_min is not None and q_max is not None and q_min >= q_max:
                return q_min, q_max, self._tr("workflow.q_range_order")
            return q_min, q_max, None

        def _render_q_range_feedback(self) -> None:
            error = str(self._q_range_error or "")
            self.q_range_error_label.setText(error)
            self.q_range_error_label.setVisible(bool(error))
            for edit in (self.q_min_edit, self.q_max_edit):
                edit.setProperty("invalid", bool(error))
                edit.setToolTip(error)
                edit.style().unpolish(edit)
                edit.style().polish(edit)

        def _evaluation_resamples(self) -> int:
            try:
                return int(self._settings.get("evaluation_resamples", 32))
            except (TypeError, ValueError):
                return 32

        def _sync_evaluation_controls(self) -> None:
            value = self._evaluation_resamples()
            index = self.evaluation_resamples_combo.findData(value)
            if index < 0:
                label = self._tr("evaluation.custom", count=value)
                self.evaluation_resamples_combo.addItem(label, value)
                index = self.evaluation_resamples_combo.findData(value)
            self.evaluation_resamples_combo.blockSignals(True)
            self.sensitivity_check.blockSignals(True)
            try:
                self.evaluation_resamples_combo.setCurrentIndex(max(0, index))
                self.sensitivity_check.setChecked(bool(self._settings.get("sensitivity", True)))
            finally:
                self.evaluation_resamples_combo.blockSignals(False)
                self.sensitivity_check.blockSignals(False)

        def _on_evaluation_settings_changed(self, *_: Any) -> None:
            selected = self.evaluation_resamples_combo.currentData()
            if selected is None:
                return
            value = int(selected)
            changed = (
                value != self._evaluation_resamples()
                or bool(self.sensitivity_check.isChecked())
                != bool(self._settings.get("sensitivity", True))
            )
            if not changed:
                return
            self._settings["evaluation_resamples"] = value
            if self._settings.get("stage") == "evaluate":
                self._settings["resamples"] = value
            self._settings["sensitivity"] = bool(self.sensitivity_check.isChecked())
            self.clear_result()
            self.analysisChanged.emit({"butterfly": self.butterfly_settings})
            self._sync_action_state()

        def set_q_window(self, q_window: Sequence[Any] | None = None) -> None:
            if q_window is None:
                self._requested_q_window = None
            else:
                try:
                    low, high = (float(value) for value in q_window)
                    if math.isfinite(low) and math.isfinite(high) and high > low:
                        self._requested_q_window = (low, high)
                except (TypeError, ValueError):
                    return
            self._landmark_zoomed = False
            self.reset_peak_zoom_button.setEnabled(False)
            self.qspace.set_q_window(q_window)

        def set_analysis_settings(
            self,
            settings: Mapping[str, Any] | None,
            *,
            replace: bool = False,
        ) -> None:
            if not isinstance(settings, Mapping):
                return
            previous_range = self._last_valid_analysis_range
            if any(key in settings for key in ("q_min", "q_max", "draw_axis_deg")):
                widgets = (self.q_min_edit, self.q_max_edit, self.reference_axis_spin)
                for widget in widgets:
                    widget.blockSignals(True)
                try:
                    for edit, key in ((self.q_min_edit, "q_min"), (self.q_max_edit, "q_max")):
                        if key in settings:
                            value = settings.get(key)
                            edit.setText("Auto" if value in (None, "") else str(value))
                    if settings.get("draw_axis_deg") is not None:
                        self.reference_axis_spin.setValue(float(settings["draw_axis_deg"]) - 90.0)
                except (TypeError, ValueError):
                    pass
                finally:
                    for widget in widgets:
                        widget.blockSignals(False)
            q_min, q_max, q_error = self._parse_analysis_q_range()
            self._q_range_error = q_error or ""
            q_changed = False
            if not q_error:
                values = (q_min, q_max, float(self.reference_axis_spin.value()))
                q_changed = values != previous_range
                self._last_valid_analysis_range = values
            self._render_q_range_feedback()
            recipe_keys = {
                "stage",
                "resamples",
                "evaluation_resamples",
                "seed",
                "edits",
                "sensitivity",
                "max_nfev",
            }
            nested = settings.get("butterfly")
            if not isinstance(nested, Mapping):
                analysis = settings.get("analysis")
                if isinstance(analysis, Mapping) and isinstance(analysis.get("butterfly"), Mapping):
                    nested = analysis["butterfly"]
                elif isinstance(analysis, Mapping):
                    nested = {key: analysis[key] for key in recipe_keys if key in analysis}
                else:
                    nested = {key: settings[key] for key in recipe_keys if key in settings}
            if not nested:
                if q_changed:
                    self.invalidate_result()
                self._sync_action_state()
                self._render_page_status()
                return
            if not isinstance(nested, Mapping):
                return
            recipe_source = {} if replace else deepcopy(self._settings)
            recipe_source.update(deepcopy(dict(nested)))
            if "evaluation_resamples" in nested:
                recipe_source["evaluation_resamples"] = nested["evaluation_resamples"]
            elif recipe_source.get("stage") == "evaluate" and "resamples" in nested:
                # An Evaluate recipe's explicit resampling count is its
                # effective uncertainty budget unless the separate saved
                # budget was explicitly supplied. Do not let a default or
                # previous-stage cached budget override that project value.
                recipe_source["evaluation_resamples"] = recipe_source.get("resamples")
            elif "evaluation_resamples" not in recipe_source:
                recipe_source["evaluation_resamples"] = (
                    32 if replace else self._evaluation_resamples()
                )
            try:
                recipe_source["evaluation_resamples"] = normalize_butterfly_settings(
                    {"resamples": recipe_source["evaluation_resamples"]}
                )["resamples"]
            except ValueError as exc:
                raise ValueError(
                    "butterfly evaluation_resamples must be a non-negative integer"
                ) from exc
            normalized = normalize_butterfly_settings(recipe_source)
            old_settings = normalize_butterfly_settings(self._settings)
            self._settings = normalized
            edits = normalized.get("edits", [])
            self._edits = [dict(edit) for edit in edits if isinstance(edit, Mapping)]
            if replace or "edits" in nested or normalized.get("edits") != old_settings.get("edits"):
                self._redo_edits.clear()
            self.qspace.set_edits(self._edits)
            self._update_edit_buttons()
            self._sync_evaluation_controls()
            if normalized != old_settings:
                self.clear_result(
                    message=(
                        "Settings changed · rerun Identify or Evaluate"
                        if self._language.lower().startswith("en")
                        else "设置已更改 · 请重新识别或评估"
                    )
                )
            elif q_changed:
                self.invalidate_result()
            self._sync_action_state()
            self._render_page_status()

        def reset_analysis_settings(self) -> None:
            """Reset the page recipe when a legacy project has no recipe."""

            self.set_analysis_settings(DEFAULT_BUTTERFLY_SETTINGS, replace=True)

        def set_legacy_method(self, method: Any) -> None:
            value = str(method or "radial_peak")
            self._legacy_method = None if value == "butterfly_curvature" else value
            if self._legacy_method:
                self.legacy_banner.setText(
                    f"Loaded legacy ridge method '{self._legacy_method}'. "
                    "Choose Identify arcs to explicitly use butterfly_curvature; the project method is preserved."
                )
            self.legacy_banner.setVisible(bool(self._legacy_method))

        def set_result(self, result: Any = None) -> None:
            butterfly = result if isinstance(result, Mapping) else {}
            self._result = deepcopy(dict(butterfly))
            self._result_fresh = bool(self._result)
            self._model_parameters = None
            self._model_reference_axis_deg = None
            self._model_status = None
            self._model_diagnostics = {}
            self._result_revision += 1
            self._manual_review = {
                "manual_status": "unreviewed",
                "reviewed_by": "",
                "reviewed_at": None,
                "review_notes": "",
                "result_revision": self._result_revision if self._result_fresh else None,
            }
            self.qspace.set_butterfly(self._result)
            self._peak_landmarks = dict(
                _read(self._result, ("peak_landmarks",), {}) or {}
            )
            self._selected_landmark = {}
            self._selected_landmark_id = None
            self._reset_landmark_zoom()
            self.qspace.set_peak_landmarks(self._peak_landmarks)
            self._render_peak_table()
            self.peak_angular_profile.clear(message_key="landmark.no_profiles")
            self.peak_radial_profile.clear(message_key="landmark.no_profiles")
            self._refresh_fit_layers()
            self.point_list.blockSignals(True)
            self.point_list.clear()
            for point in self._result.get("points", []) or []:
                if not isinstance(point, Mapping):
                    continue
                point_id = str(_read(point, ("point_id",), "") or "")
                qx = _fmt(_read(point, ("qx",), None))
                qy = _fmt(_read(point, ("qy",), None))
                accepted = bool(_read(point, ("accepted",), True))
                marker = "✓" if accepted else "×"
                item = QtWidgets.QListWidgetItem(f"{marker} {point_id}  ({qx}, {qy})")
                item.setData(QtCore.Qt.ItemDataRole.UserRole, dict(point))
                self.point_list.addItem(item)
            self.point_list.blockSignals(False)
            excluded_count = sum(
                1
                for point in (self._result.get("points", []) or [])
                if isinstance(point, Mapping)
                and (
                    not bool(_read(point, ("valid",), True))
                    or not bool(_read(point, ("accepted",), True))
                )
            )
            self.excluded_count_label.setText(f"{excluded_count} excluded")
            self._profiles = dict(_read(self._result, ("profiles",), {}) or {})
            self._ellipse_local = dict(_read(self._result, ("ellipse_local",), {}) or {})
            self._render_quantities(_read(self._result, ("quantitative_parameters",), {}) or {})
            diagnostics = _read(self._result, ("diagnostics",), {}) or {}
            magnification = _read(diagnostics, ("display_magnification", "magnification"), 1.0)
            if _finite(magnification) is not None:
                self._diagnostic_magnification = float(magnification)
                if self._ellipse_local:
                    self.uv_magnification_spin.blockSignals(True)
                    self.uv_magnification_spin.setValue(max(0.1, min(100.0, float(magnification))))
                    self.uv_magnification_spin.blockSignals(False)
                    self._uv_display_magnification = float(self.uv_magnification_spin.value())
            self._render_magnification_label()
            self._page_status_state = (
                "failed"
                if self._result and self._result_is_failed(self._result)
                else "result"
                if self._result
                else "ready"
                if self._data_ready()
                else "empty"
            )
            self._page_status_kind = str(self._settings.get("stage", "trace"))
            self._page_status_error = self._result.get("error")
            self._render_page_status()
            self._sync_export_state()

        set_butterfly_result = set_result

        def clear_result(self, *, message: str | None = None, state: str | None = None) -> None:
            self._result = {}
            self._result_fresh = False
            self._result_revision += 1
            self._manual_review = {
                "manual_status": "unreviewed",
                "reviewed_by": "",
                "reviewed_at": None,
                "review_notes": "",
                "result_revision": None,
            }
            self._profiles = {}
            self._ellipse_local = {}
            self.qspace.set_butterfly({})
            self.qspace.set_selected_point(None)
            self._clear_diagnostic_layers()
            self.point_list.blockSignals(True)
            self.point_list.clear()
            self.point_list.blockSignals(False)
            self._selected_ellipse_point = {}
            self.excluded_count_label.setText("0 excluded")
            self.quantity_table.setRowCount(0)
            self.normal_profile.clear()
            self.ellipse_diagnostic.clear()
            del message  # The structured state is rendered afresh on language changes.
            self._page_status_state = str(state or ("ready" if self._data_ready() else "empty"))
            self._page_status_kind = ""
            self._page_status_error = None
            self._render_page_status()
            self._sync_export_state()

        def _render_quantities(self, quantities: Mapping[str, Any]) -> None:
            self.quantity_table.setRowCount(0)
            for name, payload in quantities.items():
                if not isinstance(payload, Mapping):
                    payload = {"value": payload}
                row = self.quantity_table.rowCount()
                self.quantity_table.insertRow(row)
                value = _read(payload, ("value",), None)
                candidate_value = _read(payload, ("candidate_value", "candidate"), None)
                status = _read(payload, ("status",), "unknown")
                interval = _read(payload, ("interval", "ci", "confidence_interval"), None)
                if isinstance(interval, Sequence) and not isinstance(interval, (str, bytes)):
                    interval_text = "[" + ", ".join(_fmt(item) for item in interval) + "]"
                else:
                    interval_text = _fmt(interval)
                reason = _read(payload, ("reason",), "")
                display_name = {
                    "axis_ratio": "b/a",
                    "theta_deg": "θ",
                    "theta": "θ",
                }.get(str(name), str(name))
                status_text = str(status)
                reason_text = str(reason or "")
                if not self._language.lower().startswith("en"):
                    status_text = {
                        "ok": "通过",
                        "pass": "通过",
                        "warn": "警告",
                        "warning": "警告",
                        "not_evaluated": "待评估",
                        "not evaluated": "待评估",
                        "pending": "待评估",
                        "pending_evaluation": "待评估",
                        "undetermined": "未确定",
                        "fail": "失败",
                        "failed": "失败",
                    }.get(status_text.lower(), status_text)
                    if status_text == "待评估" and not reason_text:
                        reason_text = self._tr(
                            "workflow.trace_evaluate_reason",
                            resamples=self._evaluation_resamples(),
                        )
                elif status_text.lower() in {"not_evaluated", "not evaluated", "pending", "pending_evaluation"}:
                    status_text = "not evaluated"
                    if not reason_text:
                        reason_text = self._tr(
                            "workflow.trace_evaluate_reason",
                            resamples=self._evaluation_resamples(),
                        )
                for column, text in enumerate(
                    (display_name, _fmt(value), status_text, _fmt(candidate_value), interval_text, reason_text)
                ):
                    self.quantity_table.setItem(row, column, QtWidgets.QTableWidgetItem(text))
            self._render_review_observables()

        def _render_review_observables(self) -> None:
            """Show observed period and coverage even when Ln/b/a stay unpublished."""

            result = self._result if isinstance(self._result, Mapping) else {}
            candidate = result.get("candidate_fit")
            if not isinstance(candidate, Mapping):
                candidate = {}
            quality = result.get("quality")
            if not isinstance(quality, Mapping):
                quality = {}
            metrics = quality.get("metrics")
            side_counts = metrics.get("side_counts") if isinstance(metrics, Mapping) else None
            sides = None
            if isinstance(side_counts, Mapping) and side_counts:
                present = sum(
                    1
                    for value in side_counts.values()
                    if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0
                )
                sides = f"{present}/4"
            english = self._language.lower().startswith("en")
            bound_flags = candidate.get("bound_flags")
            flag_names = [str(item) for item in (candidate.get("flags") or ()) if item]
            flag_names.extend(str(item) for item in (quality.get("flags") or ()) if item)
            if isinstance(bound_flags, Mapping) and bound_flags.get("axis_ratio"):
                flag_names.append("axis_ratio_at_bound")
            kind = classify_ellipse_publication(
                quality_status=quality.get("status"),
                axis_ratio=candidate.get("axis_ratio"),
                flags=flag_names,
            )
            unpublished_shape = unpublished_ellipse_shape(
                quality_status=quality.get("status"),
                axis_ratio=candidate.get("axis_ratio"),
                flags=flag_names,
            )
            if kind == "ring":
                reading = "ring only" if english else "仅一阶环"
            elif kind == "fail":
                reading = "fail" if english else "失败"
            else:
                reading = "ellipse" if english else "椭圆"
            review_rows = (
                (
                    "reading" if english else "判读",
                    reading,
                    quality.get("status"),
                    None,
                    None,
                    None,
                ),
                (
                    "quality" if english else "质量",
                    quality.get("status"),
                    quality.get("status"),
                    None,
                    None,
                    ", ".join(str(item) for item in (quality.get("flags") or ()) if item),
                ),
                (
                    "arcs" if english else "弧",
                    sides,
                    None,
                    None,
                    None,
                    None,
                ),
                (
                    "q* (first-order)" if english else "一阶 q*",
                    candidate.get("q_star_from_arcs", result.get("q_star_from_arcs")),
                    candidate.get("q_star_source", result.get("q_star_source")),
                    None,
                    None,
                    None,
                ),
                (
                    "L ring (nm)" if english else "环 L（nm）",
                    candidate.get(
                        "L_from_observed_radius_nm",
                        result.get("L_from_observed_radius_nm"),
                    ),
                    None,
                    None,
                    None,
                    None,
                ),
            )
            if not unpublished_shape:
                extra = []
                ln = candidate.get("Ln_from_minor_axis_nm", candidate.get("L_N"))
                lz = candidate.get("Lz_from_draw_axis_nm", candidate.get("L_z"))
                l_major = candidate.get("L_from_major_axis_nm")
                reason = (
                    "apparent; unpublished until independently supported"
                    if english
                    else "表观值；尚未独立支持，故不发表"
                )
                if ln not in (None, ""):
                    extra.append(
                        (
                            "Ln candidate (nm)" if english else "Ln 候选（nm）",
                            None,
                            None,
                            ln,
                            None,
                            reason,
                        )
                    )
                if lz not in (None, ""):
                    extra.append(
                        (
                            "Lz candidate (nm)" if english else "Lz 候选（nm）",
                            None,
                            None,
                            lz,
                            None,
                            reason,
                        )
                    )
                if l_major not in (None, ""):
                    extra.append(
                        (
                            "L major candidate (nm)" if english else "长轴 L 候选（nm）",
                            None,
                            None,
                            l_major,
                            None,
                            reason,
                        )
                    )
                if extra:
                    review_rows = review_rows + tuple(extra)
            if not any(row[1] not in (None, "", []) for row in review_rows):
                return
            for name, value, status, candidate_value, interval, reason in review_rows:
                if value in (None, "") and not status and not reason:
                    continue
                row = self.quantity_table.rowCount()
                self.quantity_table.insertRow(row)
                for column, text in enumerate(
                    (
                        str(name),
                        _fmt(value),
                        "" if status is None else str(status),
                        _fmt(candidate_value),
                        "" if interval is None else str(interval),
                        "" if reason is None else str(reason),
                    )
                ):
                    self.quantity_table.setItem(row, column, QtWidgets.QTableWidgetItem(text))

        def _on_point_selected(self, point: Any) -> None:
            if not isinstance(point, Mapping):
                return
            point_id = str(_read(point, ("point_id",), "") or "")
            self.point_list.blockSignals(True)
            for row in range(self.point_list.count()):
                item = self.point_list.item(row)
                data = item.data(QtCore.Qt.ItemDataRole.UserRole)
                if isinstance(data, Mapping) and str(data.get("point_id", "")) == point_id:
                    self.point_list.setCurrentRow(row)
                    break
            self.point_list.blockSignals(False)
            self._render_profile(point_id, point)
            self.pointSelected.emit(dict(point))

        def _on_point_list_changed(self, row: int) -> None:
            if row < 0:
                return
            item = self.point_list.item(row)
            point = item.data(QtCore.Qt.ItemDataRole.UserRole) if item is not None else None
            if not isinstance(point, Mapping):
                return
            self.qspace.set_selected_point(_read(point, ("point_id",), None))
            self._on_point_selected(point)

        def _exclude_selected_point(self) -> None:
            item = self.point_list.currentItem()
            point = item.data(QtCore.Qt.ItemDataRole.UserRole) if item is not None else None
            point_id = _read(point, ("point_id",), None)
            if point_id not in (None, ""):
                self._on_edit_requested({"type": "exclude_point", "point_id": str(point_id)})

        @staticmethod
        def _series(profile: Mapping[str, Any], x_names: tuple[str, ...], y_names: tuple[str, ...]) -> tuple[list[Any], list[Any]]:
            x = _read(profile, x_names, [])
            y = _read(profile, y_names, [])
            if isinstance(x, Mapping):
                x = _read(x, ("values", "data"), [])
            if isinstance(y, Mapping):
                y = _read(y, ("values", "data"), [])
            try:
                return list(x or []), list(y or [])
            except TypeError:
                return [], []

        def _render_profile(self, point_id: str, point: Mapping[str, Any] | None = None) -> None:
            profile = self._profiles.get(point_id)
            if profile is None:
                profile = self._profiles.get(str(point_id))
            if not isinstance(profile, Mapping):
                self.normal_profile.clear()
                self.ellipse_diagnostic.clear()
                return
            x, raw = self._series(profile, ("offset_q", "q_offset", "normal_q", "x"), ("raw", "raw_intensity", "intensity", "observed"))
            _, fit = self._series(profile, ("offset_q", "q_offset", "normal_q", "x"), ("fit", "fit_intensity", "fitted", "model"))
            _, residual = self._series(profile, ("offset_q", "q_offset", "normal_q", "x"), ("residual", "resid", "fit_residual"))
            series = {name: values for name, values in (("raw", raw), ("fit", fit), ("residual", residual)) if values}
            self.normal_profile.set_series(x, series, x_label="normal q offset", y_label="intensity") if x and series else self.normal_profile.clear()
            self._selected_ellipse_point = dict(point or {})
            self._render_ellipse_local()

        def _render_ellipse_local(self) -> None:
            bundle = self._ellipse_local
            if not isinstance(bundle, Mapping) or not bundle:
                self.ellipse_diagnostic.clear(
                    "No ellipse-local diagnostic" if self._language.lower().startswith("en") else "暂无椭圆局部诊断"
                )
                return
            selected = getattr(self, "_selected_ellipse_point", {})
            source_branch = _read(selected, ("branch_id", "source_branch", "branch"), None)
            side = str(_read(selected, ("side",), "") or "").lower()
            if "points" not in bundle and "curves" not in bundle:
                branch_bundle = None
                if source_branch is not None:
                    branch_bundle = bundle.get(str(source_branch), bundle.get(source_branch))
                if branch_bundle is None and side:
                    branch_bundle = bundle.get(side)
                if isinstance(branch_bundle, Mapping):
                    bundle = branch_bundle
            point_rows = _read(bundle, ("points",), []) or []
            curves = _read(bundle, ("curves",), {}) or {}
            selected_points = [
                row for row in point_rows
                if isinstance(row, Mapping)
                and (source_branch is None or str(row.get("branch_id", row.get("source_branch", source_branch))) == str(source_branch))
                and (not side or str(row.get("side", "")).lower() == side)
            ]
            if not selected_points:
                selected_points = [row for row in point_rows if isinstance(row, Mapping)]
            self.ellipse_diagnostic.set_uv_series(
                selected_points,
                curves if isinstance(curves, Mapping) else {},
                v_scale=self._uv_display_magnification,
                x_label="u",
                y_label=(
                    f"v ×{self._uv_display_magnification:g}"
                    if self._language.lower().startswith("en")
                    else f"v ×{self._uv_display_magnification:g}（显示）"
                ),
            )
            self._diagnostic_magnification = float(self._uv_display_magnification)
            self._render_magnification_label()

        def _on_edit_requested(self, edit: Any) -> None:
            if not isinstance(edit, Mapping):
                return
            item = dict(edit)
            if item.get("type") == "seed":
                branch = self.seed_branch_combo.currentData()
                side = self.seed_side_combo.currentData()
                if branch in (0, 1):
                    item["branch_id"] = int(branch)
                item["side"] = str(side or "unknown")
            self._edits.append(deepcopy(item))
            self._redo_edits.clear()
            self._settings["edits"] = deepcopy(self._edits)
            self.qspace.set_edits(self._edits)
            self.clear_result()
            self.editChanged.emit(self.butterfly_settings)
            self._update_edit_buttons()

        def start_correction(self) -> None:
            mode = str(self.correction_mode_combo.currentData() or "seed")
            self.qspace.set_interaction_mode(mode)
            self.status_label.setText(
                f"Correction mode · {mode}"
                if self._language.lower().startswith("en")
                else f"校正模式 · {mode}"
            )
            self.qspace.setFocus(QtCore.Qt.FocusReason.OtherFocusReason)

        def _open_numeric_edit_dialog(self) -> None:
            english = self._language.lower().startswith("en")
            dialog = QtWidgets.QDialog(self)
            dialog.setWindowTitle("Keyboard edit" if english else "键盘编辑")
            form = QtWidgets.QFormLayout(dialog)
            mode = QtWidgets.QComboBox(dialog)
            mode.addItem("Seed point" if english else "种子点", "seed")
            mode.addItem("Exclude polygon" if english else "排除多边形", "exclude_polygon")
            mode.addItem("Include polygon" if english else "包含多边形", "include_polygon")
            mode.addItem("Exclude rectangle" if english else "排除矩形", "exclude_rectangle")
            mode.addItem("Include rectangle" if english else "包含矩形", "include_rectangle")
            form.addRow("Mode" if english else "模式", mode)
            qx_edit = QtWidgets.QLineEdit(dialog)
            qy_edit = QtWidgets.QLineEdit(dialog)
            form.addRow("qx" if english else "qx", qx_edit)
            form.addRow("qy" if english else "qy", qy_edit)
            points_edit = QtWidgets.QLineEdit(dialog)
            points_edit.setPlaceholderText("qx,qy; qx,qy; qx,qy")
            form.addRow("Points" if english else "顶点", points_edit)
            buttons = QtWidgets.QDialogButtonBox(
                QtWidgets.QDialogButtonBox.StandardButton.Ok
                | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
                parent=dialog,
            )
            form.addRow(buttons)
            buttons.accepted.connect(dialog.accept)
            buttons.rejected.connect(dialog.reject)
            if dialog.exec() != int(QtWidgets.QDialog.DialogCode.Accepted):
                return
            kind = str(mode.currentData())
            if kind == "seed":
                try:
                    edit = {"type": "seed", "qx": float(qx_edit.text()), "qy": float(qy_edit.text())}
                except ValueError:
                    return
            else:
                pairs: list[list[float]] = []
                try:
                    for token in points_edit.text().split(";"):
                        x_text, y_text = token.strip().split(",", 1)
                        pairs.append([float(x_text), float(y_text)])
                except (TypeError, ValueError):
                    return
                if kind in {"exclude_rectangle", "include_rectangle"} and len(pairs) == 2:
                    (x0, y0), (x1, y1) = pairs
                    pairs = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
                    kind = "exclude_polygon" if kind == "exclude_rectangle" else "include_polygon"
                if len(pairs) < 3:
                    return
                edit = {"type": kind, "points": pairs}
            self._on_edit_requested(edit)

        def undo_edit(self) -> None:
            if not self._edits:
                return
            self._redo_edits.append(self._edits.pop())
            self._settings["edits"] = deepcopy(self._edits)
            self.qspace.set_edits(self._edits)
            self.clear_result()
            self.editChanged.emit(self.butterfly_settings)
            self._update_edit_buttons()

        def redo_edit(self) -> None:
            if not self._redo_edits:
                return
            self._edits.append(self._redo_edits.pop())
            self._settings["edits"] = deepcopy(self._edits)
            self.qspace.set_edits(self._edits)
            self.clear_result()
            self.editChanged.emit(self.butterfly_settings)
            self._update_edit_buttons()

        def _update_edit_buttons(self) -> None:
            self.undo_button.setEnabled(bool(self._edits))
            self.redo_button.setEnabled(bool(self._redo_edits))

        def _request_payload(self, *, stage: str, resamples: int) -> dict[str, Any]:
            self._settings["stage"] = stage
            if stage == "evaluate":
                selected_resamples = self._evaluation_resamples()
                self._settings["evaluation_resamples"] = selected_resamples
                self._settings["resamples"] = selected_resamples
            else:
                # Identify is a zero-resample trace request. Keep the user's
                # selected evaluation budget separately for the next stage.
                self._settings["resamples"] = int(resamples)
            self._settings["edits"] = deepcopy(self._edits)
            self._page_status_state = "ready"
            self._page_status_kind = stage
            self._page_status_error = None
            self._render_page_status()
            return self.butterfly_settings

        def request_identify(self) -> None:
            if self._q_range_error:
                self._render_q_range_feedback()
                self._render_page_status()
                return
            if not self._data_ready():
                self.set_job_status("empty")
                return
            self.identifyRequested.emit(self._request_payload(stage="trace", resamples=0))

        def request_evaluate(self) -> None:
            if self._q_range_error:
                self._render_q_range_feedback()
                self._render_page_status()
                return
            if not self._data_ready():
                self.set_job_status("empty")
                return
            self.evaluateRequested.emit(
                self._request_payload(
                    stage="evaluate",
                    resamples=self._evaluation_resamples(),
                )
            )

        def apply_to_batch(self) -> None:
            if self._q_range_error:
                self._render_q_range_feedback()
                self._render_page_status()
                return
            self.applyToBatchRequested.emit({"analysis": self.analysis_settings, "edits": self.edits})

        def set_batch_feedback(self, successes: Sequence[Any] = (), failures: Sequence[Any] = ()) -> None:
            success_count = len(list(successes))
            failure_items = list(failures)
            if failure_items:
                details = "; ".join(str(item) for item in failure_items[:4])
                if len(failure_items) > 4:
                    details += f" (+{len(failure_items) - 4})"
                self.batch_feedback_label.setText(f"Batch applied: {success_count} ok; failures: {details}")
            else:
                self.batch_feedback_label.setText(f"Batch applied: {success_count} frame(s) ready")

        def set_busy(self, busy: bool) -> None:
            self._busy = bool(busy)
            self._sync_action_state()
            if busy:
                self.set_job_status("running")

        def set_job_status(
            self,
            state: str,
            kind: str = "",
            *,
            error: Any = None,
            result_ok: bool | None = None,
            elapsed_s: float | None = None,
            progress_percent: int | None = None,
            progress_phase: str | None = None,
        ) -> None:
            """Route asynchronous worker outcomes into the page-local status."""

            state = str(state or "ready").lower()
            label = str(kind or "analysis")
            retain_error = state in {"error", "failed"} or result_ok is False
            if state == "canceled":
                state = "cancelled"
            if result_ok is False:
                state = "failed"
            if state in {"cancelled", "ignored", "stale"} and label in {
                "preview",
                "optimize",
                "measure_geometry",
                "refine_geometry",
                "trace",
                "evaluate",
            }:
                self._result_fresh = False
                self._clear_diagnostic_layers()
            if state in {"running", "cancelling"}:
                if elapsed_s is not None:
                    self._job_elapsed_s = max(0.0, float(elapsed_s))
                if progress_percent is not None:
                    self._job_progress_percent = max(0, min(100, int(progress_percent)))
                if progress_phase is not None:
                    self._job_progress_phase = str(progress_phase)
            else:
                self._job_elapsed_s = None
                self._job_progress_percent = None
                self._job_progress_phase = ""
            self._page_status_state = {
                "error": "failed",
                "complete": "completed",
            }.get(state, state)
            self._page_status_kind = label
            self._page_status_error = error if retain_error else None
            self._render_page_status()
            try:
                QtGui.QAccessible.updateAccessibility(
                    QtGui.QAccessibleEvent(self.status_label, QtGui.QAccessible.Event.NameChanged)
                )
            except (AttributeError, TypeError):
                pass

        def finish_detached_job_if_active(self, kind: str) -> None:
            """Clear a detached worker label without replacing a newer page state."""

            if (
                self._page_status_kind == str(kind)
                and self._page_status_state in {"running", "cancelling"}
            ):
                self.set_job_status("ready")

        def save_screenshot(self, path: str | Path) -> Path:
            target = Path(path).expanduser().resolve()
            target.parent.mkdir(parents=True, exist_ok=True)
            if not bool(self.grab().save(str(target))):
                raise OSError(f"could not save screenshot: {target}")
            return target


else:

    class ButterflyWorkbench:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs
            require_qt()


__all__ = ["ButterflyWorkbench", "DEFAULT_BUTTERFLY_SETTINGS"]
