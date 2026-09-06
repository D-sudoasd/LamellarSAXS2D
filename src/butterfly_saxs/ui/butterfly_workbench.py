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

from ..butterfly_settings import normalize_butterfly_settings
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

        def __init__(self, title: str, parent: Any = None) -> None:
            super().__init__(parent)
            self._title = str(title)
            self.setObjectName(title.replace(" ", "") + "Panel")
            layout = QtWidgets.QVBoxLayout(self)
            layout.setContentsMargins(4, 4, 4, 4)
            self.title_label = QtWidgets.QLabel(title, self)
            self.title_label.setStyleSheet("font-weight: 600;")
            self.title_label.setWordWrap(True)
            self.title_label.setMaximumHeight(32)
            layout.addWidget(self.title_label)
            self.plot = None
            self.table = None
            if _pg is not None:
                self.plot = _pg.PlotWidget(self)
                self.plot.setBackground("#16181e")
                self.plot.showGrid(x=True, y=True, alpha=0.18)
                self.plot.setMinimumHeight(112)
                self.plot.setAccessibleName(self._title)
                self.plot.setAccessibleDescription(
                    f"{self._title}; measured and fitted diagnostic series"
                )
                layout.addWidget(self.plot, 1)
            else:
                self.table = QtWidgets.QTableWidget(0, 3, self)
                self.table.setHorizontalHeaderLabels(("x", "raw", "fit"))
                self.table.horizontalHeader().setStretchLastSection(True)
                self.table.setMinimumHeight(112)
                self.table.setAccessibleName(self._title)
                self.table.setAccessibleDescription(
                    f"{self._title}; measured and fitted diagnostic values"
                )
                layout.addWidget(self.table, 1)
            self.empty_label = QtWidgets.QLabel("Select a measured point", self)
            self.empty_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self.empty_label.setStyleSheet("color: #8c95a5;")
            layout.addWidget(self.empty_label)

        def clear(self, message: str = "Select a measured point") -> None:
            if self.plot is not None:
                self.plot.clear()
            if self.table is not None:
                self.table.setRowCount(0)
            self.empty_label.setText(message)
            self.empty_label.setVisible(True)

        def set_language(self, *, english: bool) -> None:
            """Refresh visible and assistive text without changing data."""

            title = self._title
            if not english:
                title = {
                    "Normal profile · raw / fit / residual": "法向剖面 · 原始 / 拟合 / 残差",
                    "Ellipse-local u/v · narrow-axis": "椭圆局部 u/v · 短轴诊断",
                }.get(title, title)
            self.title_label.setText(title)
            empty = "Select a measured point" if english else "请选择测量点"
            self.empty_label.setText(empty)
            if self.plot is not None:
                self.plot.setAccessibleName(title)
                self.plot.setAccessibleDescription(
                    f"{title}; measured and fitted diagnostic series"
                    if english
                    else f"{title}；显示测量与拟合诊断序列"
                )
            if self.table is not None:
                self.table.setAccessibleName(title)
                self.table.setAccessibleDescription(
                    f"{title}; measured and fitted diagnostic values"
                    if english
                    else f"{title}；显示测量与拟合诊断数值"
                )

        def set_series(
            self,
            x: Sequence[Any],
            series: Mapping[str, Sequence[Any]],
            *,
            x_label: str = "q offset",
            y_label: str = "value",
        ) -> None:
            values = []
            if _np is not None:
                try:
                    values = _np.asarray(x, dtype=float)
                except (TypeError, ValueError):
                    values = []
            if self.plot is not None:
                self.plot.clear()
                colors = {
                    "raw": (220, 230, 238),
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
                            name=name,
                        )
                    except (TypeError, ValueError):
                        continue
                self.plot.setLabel("bottom", x_label)
                self.plot.setLabel("left", y_label)
                self.empty_label.setVisible(not bool(series))
                return
            if self.table is not None:
                names = list(series)
                self.table.setColumnCount(1 + len(names))
                self.table.setHorizontalHeaderLabels([x_label, *names])
                count = len(values) if hasattr(values, "__len__") else len(x)
                self.table.setRowCount(count)
                for row in range(count):
                    self.table.setItem(row, 0, QtWidgets.QTableWidgetItem(_fmt(x[row])))
                    for col, name in enumerate(names, 1):
                        data = series[name]
                        self.table.setItem(row, col, QtWidgets.QTableWidgetItem(_fmt(data[row])))
                self.empty_label.setVisible(count == 0)

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
            self.clear("Select a measured point")


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
            self._result_revision = 0
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

            diagnostics = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal, canvas)
            diagnostics.setObjectName("butterflyDiagnosticsSplitter")
            self.normal_profile = _ProfilePanel("Normal profile · raw / fit / residual", diagnostics)
            self.normal_profile.setObjectName("normalProfilePanel")
            self.ellipse_diagnostic = _ProfilePanel("Ellipse-local u/v · narrow-axis", diagnostics)
            self.ellipse_diagnostic.setObjectName("ellipseDiagnosticPanel")
            diagnostics.addWidget(self.normal_profile)
            diagnostics.addWidget(self.ellipse_diagnostic)
            diagnostics.setStretchFactor(0, 1)
            diagnostics.setStretchFactor(1, 1)
            canvas_layout.addWidget(diagnostics, 2)
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
            right_scroll.setMinimumWidth(306)
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
            self.reference_axis_spin = QtWidgets.QDoubleSpinBox(analysis_group)
            self.reference_axis_spin.setObjectName("butterflyReferenceAxis")
            self.reference_axis_spin.setRange(-360.0, 360.0)
            self.reference_axis_spin.setDecimals(2)
            analysis_form.addRow("q min", self.q_min_edit)
            analysis_form.addRow("q max", self.q_max_edit)
            analysis_form.addRow("Reference axis (deg)", self.reference_axis_spin)
            self.q_min_edit.editingFinished.connect(self._on_analysis_range_changed)
            self.q_max_edit.editingFinished.connect(self._on_analysis_range_changed)
            self.reference_axis_spin.valueChanged.connect(self._on_analysis_range_changed)
            right_layout.addWidget(analysis_group)

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
                ("Parameter", "Value", "Status", "Candidate*", "Interval", "Reason")
            )
            self.quantity_table.horizontalHeader().setStretchLastSection(True)
            self.quantity_table.horizontalHeader().setMinimumSectionSize(54)
            self.quantity_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
            self.quantity_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
            self.quantity_table.setMinimumHeight(112)
            self.quantity_table.setMaximumHeight(146)
            self.quantity_table.verticalHeader().setDefaultSectionSize(18)
            self.quantity_table.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            for column, width in enumerate((45, 34, 45, 52, 42, 62)):
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
            actions_layout.addWidget(self.export_button)
            for action_button in (
                self.identify_button,
                self.evaluate_button,
                self.apply_batch_button,
                self.cancel_button,
                self.export_button,
            ):
                action_button.setMinimumHeight(30)
                action_button.setMaximumHeight(34)
            actions_group.setMaximumHeight(150)
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
            self.qspace.editRequested.connect(self._on_edit_requested)
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
            self.identify_button.setEnabled(ready and not busy)
            self.evaluate_button.setEnabled(ready and not busy)
            self.apply_batch_button.setEnabled(ready and not busy)
            self.cancel_button.setEnabled(busy)
            self.export_button.setEnabled(bool(self._result_fresh and not busy))

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
            if state == "empty":
                text = (
                    "Load a frame before Identify or Evaluate"
                    if english
                    else "请先载入图像，再识别或评估"
                )
            elif state == "running":
                text = f"Running · {kind}…" if english else f"运行中 · {kind}…"
            elif state == "cancelled":
                text = "Cancelled" if english else "已取消"
            elif state == "failed":
                suffix = f": {self._page_status_error}" if self._page_status_error else ""
                text = f"Failed · {kind}{suffix}" if english else f"失败 · {kind}{suffix}"
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
                text = f"Completed · {kind} · {count} points" if english else f"已完成 · {kind} · {count} 个点"
            else:
                if english:
                    text = "Ready · identify arcs" if self._data_ready() else "Load a frame to begin"
                else:
                    text = "就绪 · 请识别弧线" if self._data_ready() else "请先载入一帧图像"
            self.status_label.setText(text)

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
                ("Parameter", "Value", "Status", "Candidate*", "Interval", "Reason")
                if english
                else ("参数", "数值", "状态", "候选值*", "区间", "原因")
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
            self.export_button.setText("Export analysis" if english else "导出分析")
            self.normal_profile.set_language(english=english)
            self.ellipse_diagnostic.set_language(english=english)
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

        def _sync_export_state(self) -> None:
            self._sync_action_state()

        def invalidate_result(self) -> None:
            """Invalidate derived geometry without changing the editable recipe."""

            self._result_fresh = False
            if self._page_status_state in {"result", "failed", "completed"}:
                self._page_status_state = "ready" if self._data_ready() else "empty"
                self._page_status_kind = ""
                self._page_status_error = None
                self._render_page_status()
            self._sync_export_state()

        def _on_analysis_range_changed(self, *_: Any) -> None:
            def scalar(edit: Any) -> float | None:
                text = edit.text().strip()
                if text.lower() in {"", "auto", "自动"}:
                    return None
                try:
                    value = float(text)
                except ValueError:
                    return None
                return value if math.isfinite(value) else None

            q_min = scalar(self.q_min_edit)
            q_max = scalar(self.q_max_edit)
            if q_min is not None and q_max is not None and q_min >= q_max:
                return
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

        def set_q_window(self, q_window: Sequence[Any] | None = None) -> None:
            self.qspace.set_q_window(q_window)

        def set_analysis_settings(
            self,
            settings: Mapping[str, Any] | None,
            *,
            replace: bool = False,
        ) -> None:
            if not isinstance(settings, Mapping):
                return
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
            recipe_keys = {"stage", "resamples", "seed", "edits", "sensitivity", "max_nfev"}
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
                return
            if not isinstance(nested, Mapping):
                return
            recipe_source = {} if replace else deepcopy(self._settings)
            recipe_source.update(deepcopy(dict(nested)))
            normalized = normalize_butterfly_settings(recipe_source)
            old_settings = normalize_butterfly_settings(self._settings)
            self._settings = normalized
            edits = normalized.get("edits", [])
            self._edits = [dict(edit) for edit in edits if isinstance(edit, Mapping)]
            if replace or "edits" in nested or normalized.get("edits") != old_settings.get("edits"):
                self._redo_edits.clear()
            self.qspace.set_edits(self._edits)
            self._update_edit_buttons()
            if normalized != old_settings:
                self.clear_result(
                    message=(
                        "Settings changed · rerun Identify or Evaluate"
                        if self._language.lower().startswith("en")
                        else "设置已更改 · 请重新识别或评估"
                    )
                )

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
            self._result_revision += 1
            self._manual_review = {
                "manual_status": "unreviewed",
                "reviewed_by": "",
                "reviewed_at": None,
                "review_notes": "",
                "result_revision": self._result_revision if self._result_fresh else None,
            }
            self.qspace.set_butterfly(self._result)
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
                        reason_text = "追踪阶段；点击“评估”运行 32 次重采样"
                elif status_text.lower() in {"not_evaluated", "not evaluated", "pending", "pending_evaluation"}:
                    status_text = "not evaluated"
                    if not reason_text:
                        reason_text = "Trace stage; click Evaluate for 32 resamples"
                for column, text in enumerate(
                    (display_name, _fmt(value), status_text, _fmt(candidate_value), interval_text, reason_text)
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
            self._settings["resamples"] = int(resamples)
            self._settings["edits"] = deepcopy(self._edits)
            self._page_status_state = "ready"
            self._page_status_kind = stage
            self._page_status_error = None
            self._render_page_status()
            return self.butterfly_settings

        def request_identify(self) -> None:
            if not self._data_ready():
                self.set_job_status("empty")
                return
            self.identifyRequested.emit(self._request_payload(stage="trace", resamples=0))

        def request_evaluate(self) -> None:
            if not self._data_ready():
                self.set_job_status("empty")
                return
            self.evaluateRequested.emit(self._request_payload(stage="evaluate", resamples=32))

        def apply_to_batch(self) -> None:
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
        ) -> None:
            """Route asynchronous worker outcomes into the page-local status."""

            english = self._language.lower().startswith("en")
            state = str(state or "ready").lower()
            label = str(kind or "analysis")
            del english
            if state == "canceled":
                state = "cancelled"
            if result_ok is False:
                state = "failed"
            self._page_status_state = {
                "error": "failed",
                "complete": "completed",
            }.get(state, state)
            self._page_status_kind = label
            self._page_status_error = error
            self._render_page_status()
            try:
                QtGui.QAccessible.updateAccessibility(
                    QtGui.QAccessibleEvent(self.status_label, QtGui.QAccessible.Event.NameChanged)
                )
            except (AttributeError, TypeError):
                pass

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
