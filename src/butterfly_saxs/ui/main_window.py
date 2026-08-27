"""Interactive 2D SAXS refinement workbench.

The GUI is intentionally an adapter layer.  Scientific engines can return
plain mappings/dataclasses/arrays and the workbench takes care of rendering,
editing, persistence and background-job lifetime.  Importing this module is
safe without the optional Qt stack; constructing the window then gives a
focused installation hint.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import inspect
import json
import math
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Callable

from .models import ParameterRow, ParameterTableModel
from .qt_compat import QT_AVAILABLE, QtCore, QtGui, QtWidgets, require_qt
from .views import PLOT_AVAILABLE, ViewGrid
from .workers import AnalysisWorker, GenerationGuard
from ..service import DEFAULT_ANALYSIS_SETTINGS

try:
    import numpy as _np
except Exception:  # pragma: no cover - numpy is a core dependency normally
    _np = None

if PLOT_AVAILABLE:  # import only after the optional Qt boundary succeeds
    try:
        import pyqtgraph as _pg
    except Exception:  # pragma: no cover - handled by the text fallback
        _pg = None
else:
    _pg = None


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


def _jsonable(value: Any) -> Any:
    """Convert values to strict-JSON data (NaN/Inf become JSON ``null``)."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if _np is not None:
        if isinstance(value, _np.ndarray):
            return _jsonable(value.tolist())
        if isinstance(value, _np.generic):
            return _jsonable(value.item())
    return str(value)


def _resolve_project_path(value: Any, base: Path) -> Any:
    """Resolve one persisted path relative to its project file."""

    if value is None or value == "" or not isinstance(value, (str, Path)):
        return value
    if str(value) == "in-memory":
        return value
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = base / candidate
    return str(candidate.resolve())


def _resolve_project_frame(value: Any, base: Path) -> Any:
    """Resolve a path-like batch frame while preserving selector mappings."""

    if isinstance(value, Mapping):
        resolved = dict(value)
        for key in ("path", "file", "source"):
            if key in resolved:
                resolved[key] = _resolve_project_path(resolved[key], base)
                break
        return resolved
    return _resolve_project_path(value, base)


def _default_parameter_rows() -> list[ParameterRow]:
    """Return UI-only defaults when no engine parameter set is supplied."""

    return [
        ParameterRow("q_center", 0.10, 0.0, None, True, "", "nm⁻¹"),
        ParameterRow("q_major", 0.16, 0.0, None, True, "", "nm⁻¹"),
        ParameterRow("q_minor", 0.08, 0.0, None, True, "", "nm⁻¹"),
        # UI always exposes the angle in degrees.  Engines that use radians
        # can convert it in their adapter while retaining a stable, explicit
        # ``theta_deg`` name for project files and batch exports.
        ParameterRow("theta_deg", 0.0, -180.0, 180.0, True, "", "degree"),
        ParameterRow("ellipticity", 2.0, 1.0, None, True, "", ""),
        ParameterRow("intensity", 1.0, 0.0, None, True, "", "a.u."),
        ParameterRow("background", 0.0, 0.0, None, True, "", "a.u."),
        ParameterRow("ridge_width", 0.01, 0.0, None, True, "", "nm⁻¹"),
    ]


def _call_with_adapter(
    function: Callable[..., Any],
    *,
    kind: str,
    parameters: dict[str, Any],
    payload: Any,
    parameter_specs: Mapping[str, Any] | None = None,
) -> Any:
    """Invoke an engine method with the least surprising compatible signature."""

    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return function(kind, parameters, payload)
    available = {
        "kind": kind,
        "parameters": parameters,
        "payload": payload,
        # Rich rows are supplied in addition to scalar values.  Existing
        # adapters that accept only ``parameters`` keep their old behaviour,
        # while the real service can enforce bounds/fixed/tied expressions.
        "parameter_specs": parameter_specs,
    }
    args = signature.parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in args.values()):
        return function(**available)
    accepted = {name: value for name, value in available.items() if name in args}
    required = [
        p
        for p in args.values()
        if p.default is inspect.Parameter.empty
        and p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    if accepted or not required:
        return function(**accepted)
    if len(required) == 1:
        return function(parameters)
    if len(required) == 2:
        return function(parameters, payload)
    return function(kind, parameters, payload)


def _engine_job(engine: Any) -> Callable[..., Any]:
    """Create a worker-safe adapter which captures only the engine object."""

    def run(*, kind: str, parameters: dict[str, Any], payload: Any = None, **_: Any) -> Any:
        if engine is None:
            if kind == "batch":
                return {"records": [], "flags": ["no_engine"]}
            return {"flags": ["no_engine"]}
        # ``theta_deg`` is the stable UI/export spelling.  Keep it intact for
        # adapters that consume degree-labelled tables, while also supplying
        # the radians spelling used by the low-level ellipse optimizer.
        raw_parameters = dict(parameters)
        parameter_specs: Mapping[str, Any] | None = None
        if any(isinstance(value, Mapping) for value in raw_parameters.values()):
            parameter_specs = raw_parameters
            engine_parameters = {
                str(name): (
                    value.get("value", value.get("val", value.get("initial")))
                    if isinstance(value, Mapping)
                    else value
                )
                for name, value in raw_parameters.items()
            }
        else:
            engine_parameters = raw_parameters
        if "theta_deg" in engine_parameters:
            try:
                engine_parameters.setdefault("theta", math.radians(float(engine_parameters["theta_deg"])))
            except (TypeError, ValueError):
                pass
        if callable(engine) and not any(
            callable(getattr(engine, name, None))
            for name in ("preview", "predict", "evaluate", "render", "simulate", "run", "optimize", "fit", "refine", "batch")
        ):
            return _call_with_adapter(
                engine,
                kind=kind,
                parameters=engine_parameters,
                payload=payload,
                parameter_specs=parameter_specs,
            )
        method_names = {
            "preview": ("preview", "predict", "evaluate", "render", "simulate", "run"),
            "optimize": ("optimize", "fit", "refine", "run_fit", "run"),
            "batch": ("batch", "process_batch", "analyze_batch", "run_batch", "run"),
        }.get(kind, (kind,))
        for name in method_names:
            method = getattr(engine, name, None)
            if callable(method):
                return _call_with_adapter(
                    method,
                    kind=kind,
                    parameters=engine_parameters,
                    payload=payload,
                    parameter_specs=parameter_specs,
                )
        raise AttributeError(f"Engine does not provide a {kind} adapter")

    return run


def symmetric_ellipses(ellipse: Any, *, axis: str = "y") -> list[Any]:
    """Return an ellipse and its mirror image for a visual overlay.

    Core fitting remains responsible for the physical model.  This helper is
    only a deterministic presentation seam for the two mirror-symmetric
    trajectories described by the reference workflow.
    """

    if ellipse is None:
        return []
    if isinstance(ellipse, Mapping):
        mirror = dict(ellipse)
        center = _read(ellipse, ("center", "centre", "origin"), None)
        if center is not None:
            try:
                center = list(center)
                index = 1 if axis.lower().startswith("y") else 0
                center[index] = -float(center[index])
                mirror["center"] = center
            except (TypeError, ValueError, IndexError):
                pass
        else:
            key = "cy" if axis.lower().startswith("y") else "cx"
            if key in mirror:
                try:
                    mirror[key] = -float(mirror[key])
                except (TypeError, ValueError):
                    pass
        return [ellipse, mirror]
    return [ellipse]


def _result_value(result: Any, names: tuple[str, ...], default: Any = None) -> Any:
    value = _read(result, names, default)
    return default if value is None else value


def _analysis_scalar(value: Any, *, default: Any = None) -> Any:
    """Parse a line-edit analysis value while preserving ``Auto`` as ``None``."""

    if value is None or (isinstance(value, str) and value.strip().lower() in {"", "auto"}):
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _sequence(value: Any) -> list[Any]:
    """Convert optional NumPy/iterable profile fields without truth testing arrays."""

    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        return [value]
    try:
        return list(value)
    except TypeError:
        return [value]


if QT_AVAILABLE:

    class RefinementMainWindow(QtWidgets.QMainWindow):
        """Main 2D SAXS refinement window with explicit preview/optimize paths."""

        batchRequested = QtCore.Signal(object)
        batchPayloadRequested = QtCore.Signal(object)
        previewRequested = QtCore.Signal(int)
        optimizeRequested = QtCore.Signal(int)

        def __init__(
            self,
            engine: Any = None,
            parent: Any = None,
            *,
            parameters: Any = None,
            analysis_service: Any = None,
            analysis_settings: Mapping[str, Any] | None = None,
            mask_frame: int | None = None,
            mask_dataset: str | None = None,
            auto_preview: bool = True,
            debounce_ms: int = 250,
        ) -> None:
            super().__init__(parent)
            if analysis_service is not None:
                self.engine = analysis_service
            elif engine is not None:
                self.engine = engine
            else:
                # The standalone workbench is useful immediately after
                # installation: it owns the real I/O/geometry/measurement
                # service by default.  Tests and notebooks can still inject
                # a small engine through ``engine=``.
                from ..service import ButterflyAnalysisService

                self.engine = ButterflyAnalysisService()
            self.auto_preview = bool(auto_preview)
            self.debounce_ms = max(0, int(debounce_ms))
            # The service may estimate only amplitude/background *initial*
            # scales for its own untouched defaults.  Any explicit project or
            # public parameter commit disables that one-shot convenience.
            self._auto_scale_initial = parameters is None
            self._generation = GenerationGuard()
            self._thread_pool = QtCore.QThreadPool(self)
            self._workers: dict[int, AnalysisWorker] = {}
            self._observed: Any = None
            self._qx: Any = None
            self._qy: Any = None
            self._qmap: Any = None
            self._poni_path: str | None = None
            self._source_path: str | None = None
            self._frame: int | None = None
            self._dataset: str | None = None
            self._mask_frame: int | None = mask_frame
            self._mask_dataset: str | None = mask_dataset
            self._mask_path: str | None = None
            self._file_mask: Any = None
            self._external_mask: Any = None
            self._exclusion_roi: Any = None
            self._roi_specs: list[dict[str, Any]] = []
            self._project_path: Path | None = None
            self._config_path: str | None = None
            self._last_result: Any = None
            self._last_error: str | None = None
            self.last_metrics: dict[str, Any] = {}
            self._analysis_settings: dict[str, Any] = deepcopy(DEFAULT_ANALYSIS_SETTINGS)
            self.evolution_records: list[Any] = []
            self._evolution_rows: list[Mapping[str, Any]] = []
            self.evolution_y_key = "rmse"
            self.batch_frames: list[Any] = []

            source_parameters = parameters
            if source_parameters is None:
                source_parameters = _read(self.engine, ("parameters", "parameter_set", "params"), None)
            if source_parameters is None:
                source_parameters = _default_parameter_rows()
            self.parameter_model = ParameterTableModel(source_parameters, self)

            self._build_actions()
            self._build_central_pages()
            self._build_parameter_dock()
            self._build_batch_page()
            self._build_evolution_page()
            self._build_status_bar()
            self.set_analysis_settings(analysis_settings or self._analysis_settings, trigger_preview=False)

            self._debounce_timer = QtCore.QTimer(self)
            self._debounce_timer.setSingleShot(True)
            self._debounce_timer.timeout.connect(self._on_debounce_timeout)
            self.parameter_model.parameterChanged.connect(self._on_parameter_changed)

            self.setWindowTitle("LamellarSAXS2D · 2D Refinement")
            self.setMinimumSize(980, 680)
            self.resize(1440, 900)
            self._set_status("Ready")

        # ----- UI construction -------------------------------------------------

        def _build_actions(self) -> None:
            file_menu = self.menuBar().addMenu("&Project")
            self.open_project_action = QtGui.QAction("Open project…", self)
            self.open_project_action.setObjectName("openProjectAction")
            self.open_project_action.triggered.connect(self.load_project)
            file_menu.addAction(self.open_project_action)
            self.save_project_action = QtGui.QAction("Save project…", self)
            self.save_project_action.setObjectName("saveProjectAction")
            self.save_project_action.triggered.connect(self.save_project)
            file_menu.addAction(self.save_project_action)
            self.open_image_action = QtGui.QAction("Open image…", self)
            self.open_image_action.setObjectName("openImageAction")
            self.open_image_action.triggered.connect(self.open_image)
            file_menu.addAction(self.open_image_action)
            self.open_poni_action = QtGui.QAction("Select PONI…", self)
            self.open_poni_action.setObjectName("openPoniAction")
            self.open_poni_action.triggered.connect(self.select_poni)
            file_menu.addAction(self.open_poni_action)
            self.open_mask_action = QtGui.QAction("Select external mask…", self)
            self.open_mask_action.setObjectName("openMaskAction")
            self.open_mask_action.triggered.connect(self.select_mask)
            file_menu.addAction(self.open_mask_action)
            self.clear_mask_action = QtGui.QAction("Clear mask", self)
            self.clear_mask_action.setObjectName("clearMaskAction")
            self.clear_mask_action.triggered.connect(self.clear_external_mask)
            file_menu.addAction(self.clear_mask_action)
            file_menu.addSeparator()
            close_action = QtGui.QAction("Close", self)
            close_action.triggered.connect(self.close)
            file_menu.addAction(close_action)

            self.file_toolbar = self.addToolBar("Project")
            self.file_toolbar.setObjectName("projectToolbar")
            self.file_toolbar.addAction(self.open_project_action)
            self.file_toolbar.addAction(self.save_project_action)
            self.file_toolbar.addAction(self.open_image_action)
            self.file_toolbar.addAction(self.open_poni_action)
            self.file_toolbar.addAction(self.open_mask_action)
            self.file_toolbar.addAction(self.clear_mask_action)

        def _build_central_pages(self) -> None:
            self.pages = QtWidgets.QTabWidget(self)
            self.pages.setObjectName("mainPages")
            self.refinement_page = QtWidgets.QWidget(self.pages)
            refinement_layout = QtWidgets.QVBoxLayout(self.refinement_page)
            refinement_layout.setContentsMargins(4, 4, 4, 4)
            self.views = ViewGrid(self.refinement_page)
            self.views.setObjectName("viewGrid")
            refinement_layout.addWidget(self.views, 1)
            self.pages.addTab(self.refinement_page, "Refinement")
            self._build_measurements_page()
            self.setCentralWidget(self.pages)

        def _build_parameter_dock(self) -> None:
            self.parameters_dock = QtWidgets.QDockWidget("Parameters", self)
            self.parameters_dock.setObjectName("parametersDock")
            panel = QtWidgets.QWidget(self.parameters_dock)
            layout = QtWidgets.QVBoxLayout(panel)
            layout.setContentsMargins(4, 4, 4, 4)
            self.parameter_table = QtWidgets.QTableView(panel)
            self.parameter_table.setObjectName("parameterTable")
            self.parameter_table.setModel(self.parameter_model)
            self.parameter_table.setAlternatingRowColors(True)
            self.parameter_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectItems)
            self.parameter_table.setEditTriggers(
                QtWidgets.QAbstractItemView.EditTrigger.DoubleClicked
                | QtWidgets.QAbstractItemView.EditTrigger.EditKeyPressed
                | QtWidgets.QAbstractItemView.EditTrigger.SelectedClicked
            )
            self.parameter_table.horizontalHeader().setStretchLastSection(True)
            self.parameter_table.horizontalHeader().setMinimumSectionSize(58)
            self.parameter_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Interactive)
            for column, width in enumerate(getattr(self.parameter_model, "COLUMN_WIDTHS", (128, 82, 72, 72, 58, 112, 70, 76))):
                self.parameter_table.setColumnWidth(column, width)
            self.parameters_dock.setMinimumWidth(640)
            self.parameter_table.verticalHeader().setDefaultSectionSize(24)
            layout.addWidget(self.parameter_table, 1)

            controls = QtWidgets.QHBoxLayout()
            self.preview_button = QtWidgets.QPushButton("Preview")
            self.preview_button.setObjectName("previewButton")
            self.preview_button.setToolTip("根据当前参数生成模型预览（不改变参数）")
            self.preview_button.setStyleSheet("QPushButton { background: #2d6cdf; color: white; font-weight: 600; }")
            self.preview_button.clicked.connect(self.request_preview)
            controls.addWidget(self.preview_button)
            self.optimize_button = QtWidgets.QPushButton("Optimize")
            self.optimize_button.setObjectName("optimizeButton")
            self.optimize_button.setToolTip("在后台精修可变参数")
            self.optimize_button.setStyleSheet("QPushButton { background: #238636; color: white; font-weight: 600; }")
            self.optimize_button.clicked.connect(self.request_optimize)
            controls.addWidget(self.optimize_button)
            self.cancel_button = QtWidgets.QPushButton("Cancel")
            self.cancel_button.setObjectName("cancelButton")
            self.cancel_button.setToolTip("取消当前请求并使迟到结果失效")
            self.cancel_button.clicked.connect(self.cancel_jobs)
            controls.addWidget(self.cancel_button)
            self.ignore_late_result_button = QtWidgets.QPushButton("Ignore late result")
            self.ignore_late_result_button.setObjectName("ignoreLateResultButton")
            self.ignore_late_result_button.setToolTip("仅使迟到结果失效，不改变当前数据")
            self.ignore_late_result_button.clicked.connect(self.ignore_late_result)
            controls.addWidget(self.ignore_late_result_button)
            self.cancel_button.setEnabled(False)
            self.ignore_late_result_button.setEnabled(False)
            layout.addLayout(controls)

            options = QtWidgets.QHBoxLayout()
            self.auto_preview_check = QtWidgets.QCheckBox("Auto preview")
            self.auto_preview_check.setObjectName("autoPreviewCheck")
            self.auto_preview_check.setChecked(self.auto_preview)
            self.auto_preview_check.toggled.connect(self.set_auto_preview)
            options.addWidget(self.auto_preview_check)
            self.clear_mask_button = QtWidgets.QPushButton("Clear mask")
            self.clear_mask_button.setObjectName("clearMaskButton")
            self.clear_mask_button.setToolTip("清除外部 detector mask；ROI 保持独立")
            self.clear_mask_button.clicked.connect(self.clear_external_mask)
            options.addWidget(self.clear_mask_button)
            options.addStretch(1)
            layout.addLayout(options)

            self._build_analysis_controls(panel, layout)

            roi_box = QtWidgets.QGroupBox("Exclusion ROI (pixel)", panel)
            roi_layout = QtWidgets.QGridLayout(roi_box)
            self.roi_type_combo = QtWidgets.QComboBox(roi_box)
            self.roi_type_combo.setObjectName("roiTypeCombo")
            self.roi_type_combo.addItem("Rectangle", "rectangle")
            self.roi_type_combo.addItem("Ellipse", "ellipse")
            self.roi_type_combo.currentIndexChanged.connect(self._update_roi_controls)
            roi_layout.addWidget(QtWidgets.QLabel("Type"), 0, 0)
            roi_layout.addWidget(self.roi_type_combo, 0, 1, 1, 3)
            self.roi_x0 = QtWidgets.QDoubleSpinBox(roi_box)
            self.roi_y0 = QtWidgets.QDoubleSpinBox(roi_box)
            self.roi_x1 = QtWidgets.QDoubleSpinBox(roi_box)
            self.roi_y1 = QtWidgets.QDoubleSpinBox(roi_box)
            self.roi_cx = QtWidgets.QDoubleSpinBox(roi_box)
            self.roi_cy = QtWidgets.QDoubleSpinBox(roi_box)
            self.roi_rx = QtWidgets.QDoubleSpinBox(roi_box)
            self.roi_ry = QtWidgets.QDoubleSpinBox(roi_box)
            self.roi_angle = QtWidgets.QDoubleSpinBox(roi_box)
            for spin in (
                self.roi_x0,
                self.roi_y0,
                self.roi_x1,
                self.roi_y1,
                self.roi_cx,
                self.roi_cy,
                self.roi_rx,
                self.roi_ry,
            ):
                spin.setRange(-1e9, 1e9)
                spin.setDecimals(3)
            self.roi_rx.setMinimum(0.001)
            self.roi_ry.setMinimum(0.001)
            self.roi_angle.setRange(-360.0, 360.0)
            self.roi_angle.setDecimals(3)
            self._rectangle_roi_widgets = []
            for row, (label, spin) in enumerate(
                (("x0", self.roi_x0), ("y0", self.roi_y0), ("x1", self.roi_x1), ("y1", self.roi_y1))
            ):
                label_widget = QtWidgets.QLabel(label)
                self._rectangle_roi_widgets.append((label_widget, spin))
                roi_layout.addWidget(label_widget, 1 + row // 2, 2 * (row % 2))
                roi_layout.addWidget(spin, 1 + row // 2, 2 * (row % 2) + 1)
            self._ellipse_roi_widgets = [
                ("cx", self.roi_cx),
                ("cy", self.roi_cy),
                ("rx", self.roi_rx),
                ("ry", self.roi_ry),
                ("angle", self.roi_angle),
            ]
            for index, (label, spin) in enumerate(self._ellipse_roi_widgets):
                label_widget = QtWidgets.QLabel(label)
                self._ellipse_roi_widgets[index] = (label_widget, spin)
                roi_layout.addWidget(label_widget, 1 + index // 2, 2 * (index % 2))
                roi_layout.addWidget(spin, 1 + index // 2, 2 * (index % 2) + 1)
            self.apply_roi_button = QtWidgets.QPushButton("Apply")
            self.apply_roi_button.setObjectName("applyRoiButton")
            self.apply_roi_button.clicked.connect(self.apply_exclusion_roi)
            self.clear_roi_button = QtWidgets.QPushButton("Clear")
            self.clear_roi_button.setObjectName("clearRoiButton")
            self.clear_roi_button.clicked.connect(self.clear_exclusion_roi)
            roi_layout.addWidget(self.apply_roi_button, 4, 0, 1, 2)
            roi_layout.addWidget(self.clear_roi_button, 4, 2, 1, 2)
            layout.addWidget(roi_box)
            self._update_roi_controls()
            self.parameters_dock.setWidget(panel)
            self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self.parameters_dock)

        def _build_measurements_page(self) -> None:
            """Build the measured-profile page without making pyqtgraph required."""

            self.measurements_page = QtWidgets.QWidget(self.pages)
            root = QtWidgets.QVBoxLayout(self.measurements_page)
            root.setContentsMargins(4, 4, 4, 4)
            self.measurement_observables: Any = None

            profile_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical, self.measurements_page)
            profile_splitter.setObjectName("profileSplitter")
            self.angular_plot = None
            self.ridge_plot = None
            if _pg is not None:
                self.angular_plot = _pg.PlotWidget(self.measurements_page)
                self.angular_plot.setObjectName("angularProfilePlot")
                self.angular_plot.setLabel("bottom", "Azimuth (deg)")
                self.angular_plot.setLabel("left", "Angular intensity (a.u.)")
                self.angular_plot.showGrid(x=True, y=True, alpha=0.22)
                profile_splitter.addWidget(self.angular_plot)
                self.ridge_plot = _pg.PlotWidget(self.measurements_page)
                self.ridge_plot.setObjectName("ridgeProfilePlot")
                self.ridge_plot.setLabel("bottom", "Azimuth (deg)")
                self.ridge_plot.setLabel("left", "Ridge q (map unit)")
                self.ridge_plot.showGrid(x=True, y=True, alpha=0.22)
                profile_splitter.addWidget(self.ridge_plot)
            else:
                self.angular_placeholder = QtWidgets.QLabel("安装 pyqtgraph 后显示角向强度与 coverage")
                self.angular_placeholder.setObjectName("angularProfilePlaceholder")
                self.angular_placeholder.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                profile_splitter.addWidget(self.angular_placeholder)
                self.ridge_placeholder = QtWidgets.QLabel("安装 pyqtgraph 后显示 ridge q-angle/accepted")
                self.ridge_placeholder.setObjectName("ridgeProfilePlaceholder")
                self.ridge_placeholder.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                profile_splitter.addWidget(self.ridge_placeholder)
            root.addWidget(profile_splitter, 2)

            lower = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal, self.measurements_page)
            lower.setObjectName("measurementTablesSplitter")
            lobe_panel = QtWidgets.QWidget(lower)
            lobe_layout = QtWidgets.QVBoxLayout(lobe_panel)
            lobe_layout.addWidget(QtWidgets.QLabel("Four-lobe measurements"))
            self.lobe_table = QtWidgets.QTableWidget(0, 8, lobe_panel)
            self.lobe_table.setObjectName("lobeTable")
            self.lobe_table.setHorizontalHeaderLabels(
                ["Angle (deg)", "Intensity", "Baseline", "SNR", "FWHM (deg)", "Coverage", "Valid", "Flags"]
            )
            self.lobe_table.horizontalHeader().setStretchLastSection(True)
            self.lobe_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
            lobe_layout.addWidget(self.lobe_table, 1)
            lower.addWidget(lobe_panel)

            ridge_panel = QtWidgets.QWidget(lower)
            ridge_layout = QtWidgets.QVBoxLayout(ridge_panel)
            ridge_layout.addWidget(QtWidgets.QLabel("Ridge q vs angle / accepted"))
            self.ridge_table = QtWidgets.QTableWidget(0, 4, ridge_panel)
            self.ridge_table.setObjectName("ridgeTable")
            self.ridge_table.setHorizontalHeaderLabels(["Angle (deg)", "q", "Accepted", "Method"])
            self.ridge_table.horizontalHeader().setStretchLastSection(True)
            self.ridge_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
            ridge_layout.addWidget(self.ridge_table, 1)
            lower.addWidget(ridge_panel)

            ellipse_panel = QtWidgets.QWidget(lower)
            ellipse_layout = QtWidgets.QVBoxLayout(ellipse_panel)
            ellipse_layout.addWidget(QtWidgets.QLabel("Ellipse core quantities / quality"))
            self.ellipse_table = QtWidgets.QTableWidget(0, 2, ellipse_panel)
            self.ellipse_table.setObjectName("ellipseTable")
            self.ellipse_table.setHorizontalHeaderLabels(["Quantity", "Value"])
            self.ellipse_table.horizontalHeader().setStretchLastSection(True)
            self.ellipse_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
            ellipse_layout.addWidget(self.ellipse_table, 1)
            lower.addWidget(ellipse_panel)
            root.addWidget(lower, 2)
            self.pages.addTab(self.measurements_page, "Measurements / Profiles")

        def _build_analysis_controls(self, parent: Any, layout: Any) -> None:
            """Build explicit q/measurement controls beside the fit table."""

            self.analysis_group = QtWidgets.QGroupBox("Analysis / Measurement", parent)
            form = QtWidgets.QFormLayout(self.analysis_group)
            self.q_min_edit = QtWidgets.QLineEdit("Auto", self.analysis_group)
            self.q_min_edit.setObjectName("qMinEdit")
            self.q_min_edit.setPlaceholderText("Auto")
            self.q_max_edit = QtWidgets.QLineEdit("Auto", self.analysis_group)
            self.q_max_edit.setObjectName("qMaxEdit")
            self.q_max_edit.setPlaceholderText("Auto")
            form.addRow("q min", self.q_min_edit)
            form.addRow("q max", self.q_max_edit)

            self.draw_axis_deg_spin = QtWidgets.QDoubleSpinBox(self.analysis_group)
            self.draw_axis_deg_spin.setObjectName("drawAxisDegSpin")
            self.draw_axis_deg_spin.setRange(-360.0, 360.0)
            self.draw_axis_deg_spin.setDecimals(3)
            form.addRow("draw axis (deg)", self.draw_axis_deg_spin)

            self.ridge_method_combo = QtWidgets.QComboBox(self.analysis_group)
            self.ridge_method_combo.setObjectName("ridgeMethodCombo")
            self.ridge_method_combo.addItem("Radial peak", "radial_peak")
            self.ridge_method_combo.addItem("Surface curvature", "surface_curvature")
            form.addRow("ridge method", self.ridge_method_combo)

            def integer_spin(object_name: str, minimum: int, maximum: int = 1_000_000) -> Any:
                spin = QtWidgets.QSpinBox(self.analysis_group)
                spin.setObjectName(object_name)
                spin.setRange(minimum, maximum)
                return spin

            self.n_angular_bins_spin = integer_spin("nAngularBinsSpin", 8)
            self.n_ridge_angles_spin = integer_spin("nRidgeAnglesSpin", 1)
            self.n_radial_bins_spin = integer_spin("nRadialBinsSpin", 8)
            form.addRow("angular bins", self.n_angular_bins_spin)
            form.addRow("ridge angles", self.n_ridge_angles_spin)
            form.addRow("radial bins", self.n_radial_bins_spin)

            self.curvature_sigma_spin = QtWidgets.QDoubleSpinBox(self.analysis_group)
            self.curvature_sigma_spin.setObjectName("curvatureSigmaSpin")
            self.curvature_sigma_spin.setRange(0.001, 100.0)
            self.curvature_sigma_spin.setDecimals(3)
            form.addRow("curvature sigma", self.curvature_sigma_spin)
            self.curvature_percentile_spin = QtWidgets.QDoubleSpinBox(self.analysis_group)
            self.curvature_percentile_spin.setObjectName("curvaturePercentileSpin")
            self.curvature_percentile_spin.setRange(0.0, 100.0)
            self.curvature_percentile_spin.setDecimals(2)
            form.addRow("curvature percentile", self.curvature_percentile_spin)
            self.normal_step_spin = QtWidgets.QDoubleSpinBox(self.analysis_group)
            self.normal_step_spin.setObjectName("normalStepSpin")
            self.normal_step_spin.setRange(0.001, 2.0)
            self.normal_step_spin.setDecimals(3)
            form.addRow("normal step", self.normal_step_spin)
            self.max_pixels_spin = integer_spin("maxPixelsSpin", 0)
            self.max_pixels_spin.setSpecialValueText("0 (all)")
            form.addRow("max pixels", self.max_pixels_spin)

            self.q_min_edit.editingFinished.connect(self._on_analysis_changed)
            self.q_max_edit.editingFinished.connect(self._on_analysis_changed)
            self.draw_axis_deg_spin.valueChanged.connect(self._on_analysis_changed)
            self.ridge_method_combo.currentIndexChanged.connect(self._on_analysis_changed)
            for spin in (
                self.n_angular_bins_spin,
                self.n_ridge_angles_spin,
                self.n_radial_bins_spin,
                self.curvature_sigma_spin,
                self.curvature_percentile_spin,
                self.normal_step_spin,
                self.max_pixels_spin,
            ):
                spin.valueChanged.connect(self._on_analysis_changed)
            layout.addWidget(self.analysis_group)

        def _build_batch_page(self) -> None:
            self.batch_page = QtWidgets.QWidget(self.pages)
            layout = QtWidgets.QVBoxLayout(self.batch_page)
            toolbar = QtWidgets.QHBoxLayout()
            self.batch_add_button = QtWidgets.QPushButton("Add frames…")
            self.batch_add_button.setObjectName("batchAddButton")
            self.batch_add_button.clicked.connect(self._choose_batch_files)
            toolbar.addWidget(self.batch_add_button)
            self.batch_run_button = QtWidgets.QPushButton("Run batch")
            self.batch_run_button.setObjectName("batchRunButton")
            self.batch_run_button.clicked.connect(self.run_batch)
            toolbar.addWidget(self.batch_run_button)
            toolbar.addStretch(1)
            layout.addLayout(toolbar)
            options = QtWidgets.QFormLayout()
            self.batch_mode_combo = QtWidgets.QComboBox(self.batch_page)
            self.batch_mode_combo.setObjectName("batchModeCombo")
            self.batch_mode_combo.addItem("Independent", "independent")
            self.batch_mode_combo.addItem("Warm start", "warm_start")
            options.addRow("Mode", self.batch_mode_combo)
            self.batch_manifest_edit = QtWidgets.QLineEdit(self.batch_page)
            self.batch_manifest_edit.setObjectName("batchManifestEdit")
            self.batch_manifest_edit.setPlaceholderText("optional manifest.csv/json")
            options.addRow("Manifest", self.batch_manifest_edit)
            self.batch_checkpoint_edit = QtWidgets.QLineEdit(self.batch_page)
            self.batch_checkpoint_edit.setObjectName("batchCheckpointEdit")
            self.batch_checkpoint_edit.setPlaceholderText("optional checkpoint.json")
            options.addRow("Checkpoint", self.batch_checkpoint_edit)
            self.batch_resume_check = QtWidgets.QCheckBox("Resume checkpoint", self.batch_page)
            self.batch_resume_check.setObjectName("batchResumeCheck")
            options.addRow("", self.batch_resume_check)
            self.batch_output_edit = QtWidgets.QLineEdit(self.batch_page)
            self.batch_output_edit.setObjectName("batchOutputEdit")
            self.batch_output_edit.setPlaceholderText("optional output directory")
            options.addRow("Output", self.batch_output_edit)
            layout.addLayout(options)
            self.batch_table = QtWidgets.QTableWidget(0, 3, self.batch_page)
            self.batch_table.setObjectName("batchTable")
            self.batch_table.setHorizontalHeaderLabels(["Frame", "Status", "RMSE"])
            self.batch_table.horizontalHeader().setStretchLastSection(True)
            self.batch_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
            layout.addWidget(self.batch_table, 1)
            self.batch_progress = QtWidgets.QProgressBar(self.batch_page)
            self.batch_progress.setObjectName("batchProgress")
            self.batch_progress.setRange(0, 0)
            self.batch_progress.setVisible(False)
            layout.addWidget(self.batch_progress)
            self.pages.addTab(self.batch_page, "Batch")

        def _build_evolution_page(self) -> None:
            self.evolution_page = QtWidgets.QWidget(self.pages)
            layout = QtWidgets.QVBoxLayout(self.evolution_page)
            selector_row = QtWidgets.QHBoxLayout()
            selector_row.addWidget(QtWidgets.QLabel("Y parameter"))
            self.evolution_parameter_combo = QtWidgets.QComboBox(self.evolution_page)
            self.evolution_parameter_combo.setObjectName("evolutionParameterCombo")
            self.evolution_parameter_combo.currentTextChanged.connect(self._render_evolution)
            selector_row.addWidget(self.evolution_parameter_combo, 1)
            layout.addLayout(selector_row)
            self.evolution_plot = None
            if _pg is not None:
                self.evolution_plot = _pg.PlotWidget(self.evolution_page)
                self.evolution_plot.setObjectName("evolutionPlot")
                self.evolution_plot.setLabel("left", "RMSE")
                self.evolution_plot.setLabel("bottom", "Frame / time")
                self.evolution_plot.showGrid(x=True, y=True, alpha=0.22)
                layout.addWidget(self.evolution_plot, 2)
            else:
                self.evolution_placeholder = QtWidgets.QLabel("安装 pyqtgraph 后显示参数演化曲线")
                self.evolution_placeholder.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                layout.addWidget(self.evolution_placeholder, 1)
            self.evolution_table = QtWidgets.QTableWidget(0, 0, self.evolution_page)
            self.evolution_table.setObjectName("evolutionTable")
            layout.addWidget(self.evolution_table, 1)
            self.pages.addTab(self.evolution_page, "Evolution")

        def _build_status_bar(self) -> None:
            status = self.statusBar()
            self.status_message = QtWidgets.QLabel("Ready")
            self.status_message.setObjectName("statusMessage")
            status.addWidget(self.status_message, 1)
            self.rmse_label = QtWidgets.QLabel("RMSE: —")
            self.rmse_label.setObjectName("rmseLabel")
            status.addPermanentWidget(self.rmse_label)
            self.ndata_label = QtWidgets.QLabel("ndata: —")
            self.ndata_label.setObjectName("ndataLabel")
            status.addPermanentWidget(self.ndata_label)
            self.flags_label = QtWidgets.QLabel("flags: —")
            self.flags_label.setObjectName("flagsLabel")
            status.addPermanentWidget(self.flags_label)
            self.coverage_label = QtWidgets.QLabel("coverage: —")
            self.coverage_label.setObjectName("coverageLabel")
            status.addPermanentWidget(self.coverage_label)

        # ----- data and parameters ---------------------------------------------

        def _update_roi_controls(self) -> None:
            """Toggle the rectangle/ellipse editor without losing values."""

            if not hasattr(self, "roi_type_combo"):
                return
            is_ellipse = self.roi_type_combo.currentData() == "ellipse"
            for label, spin in getattr(self, "_rectangle_roi_widgets", ()):
                label.setVisible(not is_ellipse)
                spin.setVisible(not is_ellipse)
            for label, spin in getattr(self, "_ellipse_roi_widgets", ()):
                label.setVisible(is_ellipse)
                spin.setVisible(is_ellipse)

        @property
        def parameters(self) -> dict[str, Any]:
            return self.parameter_model.parameter_values()

        def set_parameters(self, parameters: Any) -> None:
            self._invalidate_pending_work(clear_fit=False)
            self._auto_scale_initial = False
            self.parameter_model.set_rows(parameters)
            setter = getattr(self.engine, "set_parameters", None)
            if callable(setter):
                setter(self.parameter_model.parameter_dict())

        def set_parameter(self, name: str, value: Any) -> bool:
            return self.parameter_model.set_parameter(name, value)

        @property
        def analysis_settings(self) -> dict[str, Any]:
            """Return the serializable Analysis/Measurement control state."""

            return {
                "q_min": _analysis_scalar(self.q_min_edit.text(), default=None),
                "q_max": _analysis_scalar(self.q_max_edit.text(), default=None),
                "draw_axis_deg": float(self.draw_axis_deg_spin.value()),
                "ridge_method": str(self.ridge_method_combo.currentData() or "radial_peak"),
                "n_angular_bins": int(self.n_angular_bins_spin.value()),
                "n_ridge_angles": int(self.n_ridge_angles_spin.value()),
                "n_radial_bins": int(self.n_radial_bins_spin.value()),
                "curvature_sigma": float(self.curvature_sigma_spin.value()),
                "curvature_percentile": float(self.curvature_percentile_spin.value()),
                "normal_step": float(self.normal_step_spin.value()),
                "max_pixels": int(self.max_pixels_spin.value()),
            }

        def set_analysis_settings(
            self,
            settings: Mapping[str, Any] | None,
            *,
            trigger_preview: bool = True,
        ) -> None:
            """Restore analysis controls from a project/config mapping."""

            if not isinstance(settings, Mapping):
                return
            merged = dict(self._analysis_settings)
            nested = settings.get("measurement", settings.get("analysis_settings"))
            if isinstance(nested, Mapping):
                merged.update(nested)
            merged.update({key: settings[key] for key in DEFAULT_ANALYSIS_SETTINGS if key in settings})
            self._analysis_settings = merged
            widgets = (
                self.q_min_edit,
                self.q_max_edit,
                self.draw_axis_deg_spin,
                self.ridge_method_combo,
                self.n_angular_bins_spin,
                self.n_ridge_angles_spin,
                self.n_radial_bins_spin,
                self.curvature_sigma_spin,
                self.curvature_percentile_spin,
                self.normal_step_spin,
                self.max_pixels_spin,
            )
            for widget in widgets:
                widget.blockSignals(True)
            try:
                self.q_min_edit.setText("Auto" if merged.get("q_min") in (None, "") else str(merged.get("q_min")))
                self.q_max_edit.setText("Auto" if merged.get("q_max") in (None, "") else str(merged.get("q_max")))
                self.draw_axis_deg_spin.setValue(float(merged.get("draw_axis_deg", 90.0)))
                method = str(merged.get("ridge_method", "radial_peak")).lower().replace("-", "_")
                if method == "curvature":
                    method = "surface_curvature"
                method_index = self.ridge_method_combo.findData(method)
                self.ridge_method_combo.setCurrentIndex(max(0, method_index))
                self.n_angular_bins_spin.setValue(max(8, int(merged.get("n_angular_bins", 180))))
                self.n_ridge_angles_spin.setValue(max(1, int(merged.get("n_ridge_angles", 72))))
                self.n_radial_bins_spin.setValue(max(8, int(merged.get("n_radial_bins", 192))))
                self.curvature_sigma_spin.setValue(max(0.001, float(merged.get("curvature_sigma", 2.0))))
                self.curvature_percentile_spin.setValue(min(100.0, max(0.0, float(merged.get("curvature_percentile", 25.0)))))
                self.normal_step_spin.setValue(min(2.0, max(0.001, float(merged.get("normal_step", 1.0)))))
                self.max_pixels_spin.setValue(max(0, int(merged.get("max_pixels", 0))))
            finally:
                for widget in widgets:
                    widget.blockSignals(False)
            setter = getattr(self.engine, "set_analysis_settings", None)
            if callable(setter):
                try:
                    setter(self.analysis_settings)
                except Exception:
                    pass
            if trigger_preview:
                self._on_analysis_changed()

        set_measurement_settings = set_analysis_settings

        def set_auto_preview(self, enabled: bool) -> None:
            self.auto_preview = bool(enabled)
            if hasattr(self, "auto_preview_check") and self.auto_preview_check.isChecked() != self.auto_preview:
                self.auto_preview_check.setChecked(self.auto_preview)

        def _invalidate_pending_work(self, *, clear_fit: bool = False) -> None:
            """Make every older worker result stale before changing input state."""

            self._generation.next()
            self._debounce_timer.stop()
            self._set_busy(False)
            if clear_fit:
                self.views.clear_fit()
                self._last_result = None
                self._last_error = None
                self.last_metrics = {}

        def _clear_incompatible_external_mask(self, data: Any) -> bool:
            """Drop a file/combined mask when the incoming image shape changes."""

            if self._observed is None or _np is None:
                return False
            try:
                old_shape = tuple(_np.asarray(self._observed).shape)
                new_shape = tuple(_np.asarray(data).shape)
            except Exception:
                return False
            if old_shape == new_shape:
                return False
            if self._mask_path is None and self._file_mask is None and self._external_mask is None:
                return False
            self._mask_path = None
            self._file_mask = None
            self._external_mask = None
            return True

        def set_observed_data(
            self,
            data: Any,
            *,
            qx: Any = None,
            qy: Any = None,
            qmap: Any = None,
            metadata: Mapping[str, Any] | None = None,
        ) -> None:
            self._invalidate_pending_work(clear_fit=True)
            mask_cleared_for_shape = self._clear_incompatible_external_mask(data)
            self._observed = data
            self._qx, self._qy = qx, qy
            if qmap is not None:
                self._qmap = qmap
                self._qx = _read(qmap, ("qx", "qx_nm_inv"), self._qx)
                self._qy = _read(qmap, ("qy", "qy_nm_inv"), self._qy)
            elif qx is not None and qy is not None:
                self._qmap = {"qx": qx, "qy": qy}
            setter = getattr(self.engine, "set_observed", None)
            if callable(setter):
                try:
                    state = setter(data, qx=qx, qy=qy, qmap=qmap, metadata=metadata)
                    if isinstance(state, Mapping):
                        self._qmap = state.get("qmap", self._qmap)
                        self._qx = _read(self._qmap, ("qx", "qx_nm_inv"), self._qx)
                        self._qy = _read(self._qmap, ("qy", "qy_nm_inv"), self._qy)
                except Exception as exc:
                    self._set_status(f"Geometry update failed: {exc}", flags="error")
            # Keep the overlay background synchronized even before the first
            # preview result arrives.  q extent is computed by ViewGrid only
            # for the overlay; the other three views remain pixel-space.
            self.views.set_images(
                data,
                qx=self._qx,
                qy=self._qy,
                q_unit=str(_read(self._qmap, ("q_unit", "unit"), "unknown") or "unknown"),
                valid_mask=_read(self._qmap, ("valid_mask", "valid"), None),
                external_mask=self._external_mask,
            )
            if self._roi_specs or self._file_mask is not None:
                self._recompute_external_mask(update_widgets=False)
            if mask_cleared_for_shape:
                self._set_status(
                    "External mask cleared: image shape changed",
                    flags="mask_cleared_shape_changed",
                )

        set_observed = set_observed_data

        def set_poni(self, path: str | Path | Any) -> bool:
            setter = getattr(self.engine, "set_poni", None)
            if not callable(setter):
                self._set_status("Current engine does not support PONI", flags="error")
                return False
            self._invalidate_pending_work(clear_fit=True)
            try:
                qmap = setter(path)
                self._poni_path = str(path) if isinstance(path, (str, Path)) else "in-memory"
                if qmap is not None:
                    self._qmap = qmap
                    self._qx = _read(qmap, ("qx", "qx_nm_inv"), self._qx)
                    self._qy = _read(qmap, ("qy", "qy_nm_inv"), self._qy)
                    if self._observed is not None:
                        self.views.set_images(
                            self._observed,
                            qx=self._qx,
                            qy=self._qy,
                            q_unit=str(_read(self._qmap, ("q_unit", "unit"), "unknown") or "unknown"),
                            valid_mask=_read(self._qmap, ("valid_mask", "valid"), None),
                            external_mask=self._external_mask,
                        )
                self._set_status(f"PONI loaded: {Path(path).name if isinstance(path, (str, Path)) else self._poni_path}")
                return True
            except Exception as exc:
                self._set_status(f"PONI load failed: {exc}", flags="error")
                return False

        def select_poni(self, path: str | Path | bool | None = None) -> bool:
            if isinstance(path, bool) or path is None:
                chosen, _ = QtWidgets.QFileDialog.getOpenFileName(
                    self, "Select pyFAI PONI", "", "PONI files (*.poni);;All files (*)"
                )
                if not chosen:
                    return False
                path = chosen
            return self.set_poni(path)

        def open_image(
            self,
            path: str | Path | bool | None = None,
            *,
            frame: int | None = None,
            dataset: str | None = None,
            poni: str | Path | Any | None = None,
            external_mask: Any | None = None,
            mask_frame: int | None = None,
            mask_dataset: str | None = None,
        ) -> bool:
            if isinstance(path, bool) or path is None:
                chosen, _ = QtWidgets.QFileDialog.getOpenFileName(
                    self,
                    "Open 2D SAXS image",
                    "",
                    "Detector images (*.cbf *.edf *.tif *.tiff *.npy *.npz *.h5 *.hdf5);;All files (*)",
                )
                if not chosen:
                    return False
                path = chosen
            loader = getattr(self.engine, "load_image", None)
            if not callable(loader):
                self._set_status("Current engine does not support image loading", flags="error")
                return False
            selected_mask_frame = self._mask_frame if mask_frame is None else mask_frame
            selected_mask_dataset = self._mask_dataset if mask_dataset is None else mask_dataset
            try:
                load_kwargs = {
                    "frame": frame,
                    "dataset": dataset,
                    "poni": poni,
                    "external_mask": external_mask,
                }
                if external_mask is not None:
                    load_kwargs.update(
                        mask_frame=selected_mask_frame,
                        mask_dataset=selected_mask_dataset,
                    )
                state = loader(path, **load_kwargs)
                if not isinstance(state, Mapping):
                    raise TypeError("image loader must return a mapping state")
                self._source_path = str(path)
                self._frame = frame
                self._dataset = dataset
                self._mask_frame = selected_mask_frame
                self._mask_dataset = selected_mask_dataset
                if external_mask is not None:
                    # A supplied mask replaces the previous detector mask.  Do
                    # this only after loading succeeds so a failed image/mask
                    # selection leaves the current document intact.
                    self._mask_path = None
                    self._file_mask = None
                    self._external_mask = None
                self._poni_path = state.get("poni", self._poni_path)
                self.set_observed_data(
                    state.get("observed", state.get("data")),
                    qx=state.get("qx"),
                    qy=state.get("qy"),
                    qmap=state.get("qmap"),
                    metadata=state.get("metadata"),
                )
                if external_mask is not None:
                    if isinstance(external_mask, (str, Path)):
                        self._load_external_mask(
                            external_mask,
                            frame=selected_mask_frame,
                            dataset=selected_mask_dataset,
                        )
                    else:
                        self._file_mask = _np.asarray(external_mask, dtype=bool) if _np is not None else external_mask
                    self._recompute_external_mask(update_widgets=False)
                self._set_status(f"Loaded {Path(path).name}")
                return True
            except Exception as exc:
                self._set_status(f"Image load failed: {exc}", flags="error")
                return False

        load_image = open_image
        load_frame = set_observed_data

        def _load_external_mask(
            self,
            path: str | Path,
            *,
            frame: int | None = None,
            dataset: str | None = None,
        ) -> Any:
            """Load a detector mask using only the mask's selectors."""

            from ..io import load_image

            selected_frame = self._mask_frame if frame is None else frame
            selected_dataset = self._mask_dataset if dataset is None else dataset
            loaded = load_image(path, frame=selected_frame, dataset=selected_dataset)
            array = _np.asarray(loaded.data != 0, dtype=bool) if _np is not None else loaded.data
            if self._observed is not None and _np is not None:
                expected = tuple(_np.asarray(self._observed).shape)
                if tuple(_np.asarray(array).shape) != expected:
                    raise ValueError(f"mask shape {getattr(array, 'shape', None)!r} does not match {expected!r}")
            self._mask_path = str(path)
            self._file_mask = array
            self._mask_frame = selected_frame
            self._mask_dataset = selected_dataset
            return array

        def _recompute_external_mask(self, *, update_widgets: bool = True) -> bool:
            """OR-combine file mask and all ROI specs (True means excluded)."""

            if self._observed is None or _np is None:
                return False
            self._invalidate_pending_work(clear_fit=True)
            try:
                from ..masking import combine_exclusion_masks

                shape = tuple(_np.asarray(self._observed).shape)
                masks = () if self._file_mask is None else (self._file_mask,)
                self._external_mask = combine_exclusion_masks(
                    shape,
                    masks=masks,
                    rois=self._roi_specs,
                    qx=self._qx,
                    qy=self._qy,
                )
                self.views.set_roi(self._roi_specs)
                self.views.set_images(
                    self._observed,
                    qx=self._qx,
                    qy=self._qy,
                    q_unit=str(_read(self._qmap, ("q_unit", "unit"), "unknown") or "unknown"),
                    valid_mask=_read(self._qmap, ("valid_mask", "valid"), None),
                    external_mask=self._external_mask,
                )
                if update_widgets:
                    self._sync_roi_widgets()
                excluded = int(_np.count_nonzero(self._external_mask))
                self._set_status(f"Mask/ROI applied ({excluded} excluded pixels)")
                return True
            except Exception as exc:
                self._set_status(f"Mask/ROI apply failed: {exc}", flags="error")
                return False

        def select_mask(
            self,
            path: str | Path | bool | None = None,
            *,
            mask_frame: int | None = None,
            mask_dataset: str | None = None,
        ) -> bool:
            """Select an external detector mask; its polarity is True=excluded."""

            if isinstance(path, bool) or path is None:
                chosen, _ = QtWidgets.QFileDialog.getOpenFileName(
                    self,
                    "Select external detector mask",
                    "",
                    "Mask files (*.npy *.npz *.cbf *.edf *.tif *.tiff *.h5 *.hdf5 *.csv *.txt);;All files (*)",
                )
                if not chosen:
                    return False
                path = chosen
            try:
                self._load_external_mask(path, frame=mask_frame, dataset=mask_dataset)
                if self._observed is not None and _np is not None:
                    self._recompute_external_mask(update_widgets=False)
                else:
                    # Selecting a mask is still an input-state change before
                    # an image is loaded, so older worker results must expire.
                    self._invalidate_pending_work(clear_fit=True)
                self._set_status(f"Mask loaded: {Path(path).name}")
                return True
            except Exception as exc:
                self._set_status(f"Mask load failed: {exc}", flags="error")
                return False

        open_mask = select_mask

        def clear_external_mask(self) -> bool:
            self._invalidate_pending_work(clear_fit=True)
            self._mask_path = None
            self._file_mask = None
            if self._observed is not None and self._roi_specs:
                return self._recompute_external_mask(update_widgets=False)
            self._external_mask = None
            if self._observed is not None:
                self.views.set_images(
                    self._observed,
                    qx=self._qx,
                    qy=self._qy,
                    q_unit=str(_read(self._qmap, ("q_unit", "unit"), "unknown") or "unknown"),
                    valid_mask=_read(self._qmap, ("valid_mask", "valid"), None),
                    external_mask=None,
                )
            self._set_status("External mask cleared")
            return True

        def _sync_roi_widgets(self) -> None:
            if not self._roi_specs:
                return
            spec = self._roi_specs[-1]
            kind = str(spec.get("type", "rectangle")).lower()
            index = self.roi_type_combo.findData("ellipse" if kind in {"ellipse", "elliptical"} else "rectangle")
            if index >= 0 and self.roi_type_combo.currentIndex() != index:
                self.roi_type_combo.setCurrentIndex(index)
            if kind in {"ellipse", "elliptical"}:
                for spin, key in (
                    (self.roi_cx, "cx"),
                    (self.roi_cy, "cy"),
                    (self.roi_rx, "rx"),
                    (self.roi_ry, "ry"),
                    (self.roi_angle, "angle_deg"),
                ):
                    if key in spec:
                        spin.setValue(float(spec[key]))
            else:
                for spin, key in (
                    (self.roi_x0, "x0"),
                    (self.roi_y0, "y0"),
                    (self.roi_x1, "x1"),
                    (self.roi_y1, "y1"),
                ):
                    if key in spec:
                        spin.setValue(float(spec[key]))

        def set_exclusion_roi(self, roi: Iterable[float] | Mapping[str, Any] | None) -> bool:
            if roi is None:
                return self.clear_exclusion_roi()
            if isinstance(roi, Mapping):
                spec = dict(roi)
                kind = str(spec.get("type", spec.get("kind", "rectangle"))).lower()
                if kind in {"ellipse", "elliptical"}:
                    try:
                        spec = {
                            "type": "ellipse",
                            "cx": float(spec["cx"]),
                            "cy": float(spec["cy"]),
                            "rx": float(spec["rx"]),
                            "ry": float(spec["ry"]),
                            "angle_deg": float(spec.get("angle_deg", spec.get("angle", 0.0))),
                        }
                    except (KeyError, TypeError, ValueError):
                        self._set_status("Ellipse ROI requires cx,cy,rx,ry,angle", flags="error")
                        return False
                    if not all(math.isfinite(float(spec[key])) for key in ("cx", "cy", "rx", "ry", "angle_deg")) or spec["rx"] <= 0 or spec["ry"] <= 0:
                        self._set_status("Ellipse ROI requires finite positive radii", flags="error")
                        return False
                    self._exclusion_roi = dict(spec)
                else:
                    try:
                        spec = {"type": "rectangle", **{key: float(spec[key]) for key in ("x0", "y0", "x1", "y1")}}
                    except (KeyError, TypeError, ValueError):
                        self._set_status("Rectangle ROI requires x0,y0,x1,y1", flags="error")
                        return False
                    if not all(math.isfinite(float(spec[key])) for key in ("x0", "y0", "x1", "y1")) or spec["x1"] <= spec["x0"] or spec["y1"] <= spec["y0"]:
                        self._set_status("Rectangle ROI bounds are invalid", flags="error")
                        return False
                    self._exclusion_roi = tuple(spec[key] for key in ("x0", "y0", "x1", "y1"))
            else:
                try:
                    values = tuple(float(item) for item in roi)
                except (TypeError, ValueError):
                    self._set_status("ROI requires x0,y0,x1,y1", flags="error")
                    return False
                if len(values) != 4 or not all(math.isfinite(item) for item in values):
                    self._set_status("ROI requires four finite coordinates", flags="error")
                    return False
                x0, y0, x1, y1 = values
                if x1 <= x0 or y1 <= y0:
                    self._set_status("ROI upper bounds must exceed lower bounds", flags="error")
                    return False
                self._exclusion_roi = values
                spec = {"type": "rectangle", "x0": x0, "x1": x1, "y0": y0, "y1": y1}
            self._roi_specs = [spec]
            if self._observed is None:
                # Keep the existing API (ROI is pending until an image exists),
                # but invalidate any worker that belongs to the old input state.
                self._invalidate_pending_work(clear_fit=True)
                return False
            return self._apply_roi_mask(update_widgets=True)

        def _apply_roi_mask(self, *, update_widgets: bool = True) -> bool:
            if self._observed is None or not self._roi_specs:
                return False
            return self._recompute_external_mask(update_widgets=update_widgets)

        def apply_exclusion_roi(self) -> bool:
            if self.roi_type_combo.currentData() == "ellipse":
                spec = {
                    "type": "ellipse",
                    "cx": self.roi_cx.value(),
                    "cy": self.roi_cy.value(),
                    "rx": self.roi_rx.value(),
                    "ry": self.roi_ry.value(),
                    "angle_deg": self.roi_angle.value(),
                }
            else:
                spec = {
                    "type": "rectangle",
                    "x0": self.roi_x0.value(),
                    "y0": self.roi_y0.value(),
                    "x1": self.roi_x1.value(),
                    "y1": self.roi_y1.value(),
                }
            return self.set_exclusion_roi(spec)

        def clear_exclusion_roi(self) -> bool:
            self._invalidate_pending_work(clear_fit=True)
            self._exclusion_roi = None
            self._roi_specs = []
            self.views.set_roi(None)
            if self._file_mask is not None and self._observed is not None:
                return self._recompute_external_mask(update_widgets=False)
            self._external_mask = None
            if self._observed is not None:
                self.views.set_images(
                    self._observed,
                    qx=self._qx,
                    qy=self._qy,
                    q_unit=str(_read(self._qmap, ("q_unit", "unit"), "unknown") or "unknown"),
                    valid_mask=_read(self._qmap, ("valid_mask", "valid"), None),
                    external_mask=None,
                )
            self._set_status("Exclusion ROI cleared")
            return True

        def set_fit_overlay(self, ridge_points: Any = None, ellipses: Any = None) -> None:
            self.views.set_overlay(ridge_points, ellipses)

        # ----- preview/optimization lifecycle ---------------------------------

        def _on_parameter_changed(self, name: str, field: str, value: Any) -> None:
            del field, value
            # An edit changes the request identity immediately, before the
            # debounced preview starts.  A result from the preceding values
            # must never overwrite the user's new table state.
            self._generation.next()
            if name in {"amplitude", "amplitude_plus", "amplitude_minus", "background"}:
                self._auto_scale_initial = False
            setter = getattr(self.engine, "set_parameters", None)
            if callable(setter):
                try:
                    setter(self.parameter_model.parameter_dict())
                except Exception:
                    pass
            self._set_busy(False, "edited")
            self._set_status("Parameters changed")
            if self.auto_preview:
                self._debounce_timer.start(self.debounce_ms)

        def _on_analysis_changed(self, *_: Any) -> None:
            self._generation.next()
            self._analysis_settings = self.analysis_settings
            setter = getattr(self.engine, "set_analysis_settings", None)
            if callable(setter):
                try:
                    setter(self._analysis_settings)
                except Exception:
                    # A legacy/injected engine may expose a similarly named
                    # method with a narrower contract; payload remains the
                    # authoritative worker seam in that case.
                    pass
            self._set_busy(False, "edited")
            self._set_status("Analysis settings changed")
            if self.auto_preview:
                self._debounce_timer.start(self.debounce_ms)

        def _on_debounce_timeout(self) -> None:
            self.request_preview()

        def _payload(self) -> dict[str, Any]:
            analysis = self.analysis_settings
            analysis["auto_scale_initial"] = bool(self._auto_scale_initial)
            return {
                "observed": self._observed,
                "qx": self._qx,
                "qy": self._qy,
                "qmap": self._qmap,
                "valid_mask": _read(self._qmap, ("valid_mask", "valid"), None),
                "external_mask": self._external_mask,
                "rois": list(self._roi_specs),
                "source": self._source_path,
                "poni": self._poni_path,
                "frame": self._frame,
                "dataset": self._dataset,
                "mask_path": self._mask_path,
                "mask_frame": self._mask_frame,
                "mask_dataset": self._mask_dataset,
                "analysis": analysis,
                # Background workers must remain side-effect free.  Only the
                # generation-current result is committed on the GUI thread.
                "commit_parameters": False,
            }

        def _start_job(self, kind: str, payload: Any = None) -> int:
            generation = self._generation.next()
            request_payload = self._payload() if payload is None else payload
            worker = AnalysisWorker(
                _engine_job(self.engine),
                generation=generation,
                kind=kind,
                # Keep the complete editable state (bounds, vary, ties, unit,
                # stderr) in the worker request.  ``_engine_job`` supplies the
                # scalar compatibility view to legacy injected engines.
                parameters=self.parameter_model.parameter_dict(),
                payload=request_payload,
            )
            worker.signals.finished.connect(self._on_worker_finished)
            worker.signals.error.connect(self._on_worker_error)
            self._workers[generation] = worker
            self._set_busy(True, kind)
            self._thread_pool.start(worker)
            if kind == "preview":
                self.previewRequested.emit(generation)
            elif kind == "optimize":
                self.optimizeRequested.emit(generation)
            return generation

        def request_preview(self) -> int:
            return self._start_job("preview")

        preview = request_preview

        def request_optimize(self) -> int:
            return self._start_job("optimize")

        optimize = request_optimize
        refine = request_optimize

        def cancel_jobs(self) -> None:
            # QThreadPool cannot interrupt arbitrary user code.  Advancing the
            # generation is sufficient to make every late result harmless.
            self._generation.next()
            self._set_busy(False, "cancelled")
            self._set_status("Cancelled; late results ignored")

        def ignore_late_result(self) -> None:
            """Advance the generation gate while preserving the current view."""

            self._generation.next()
            self._set_busy(False, "ignored")
            self._set_status("Late result ignored")

        ignore_late_results = ignore_late_result

        def _on_worker_finished(self, generation: int, kind: str, result: Any) -> None:
            self._workers.pop(generation, None)
            if not self._generation.is_current(generation):
                return
            self._last_result = result
            self._last_error = None
            if kind == "batch":
                records = _result_value(result, ("records", "results", "evolution"), [])
                if records:
                    self.plot_evolution(records)
                    self._update_batch_rows(records)
            else:
                self._apply_result(result)
                if kind == "optimize":
                    self._auto_scale_initial = False
            self._set_busy(False, kind)

        def _on_worker_error(self, generation: int, kind: str, error: Exception) -> None:
            self._workers.pop(generation, None)
            if not self._generation.is_current(generation):
                return
            self._last_error = str(error)
            self._set_busy(False, kind)
            self._set_status(f"{kind} failed: {error}", flags="error")

        def _apply_result(self, result: Any) -> None:
            if result is None:
                result = {}
            if not isinstance(result, Mapping) and _np is not None:
                # A bare 2D array is a convenient engine preview return value.
                try:
                    if _np.asarray(result).ndim == 2:
                        result = {"model": result}
                except Exception:
                    pass
            observed = _result_value(result, ("observed", "data", "image"), self._observed)
            model = _result_value(result, ("model", "predicted", "fit", "intensity", "simulation"), None)
            residual = _result_value(result, ("residual", "difference", "resid"), None)
            if residual is None and observed is not None and model is not None and _np is not None:
                try:
                    obs_array, model_array = _np.asarray(observed), _np.asarray(model)
                    if obs_array.shape == model_array.shape:
                        residual = obs_array - model_array
                except Exception:
                    residual = None
            result_qx = _result_value(result, ("qx", "qx_nm_inv"), self._qx)
            result_qy = _result_value(result, ("qy", "qy_nm_inv"), self._qy)
            result_valid_mask = _result_value(result, ("valid_mask",), _read(self._qmap, ("valid_mask", "valid"), None))
            result_external_mask = _result_value(result, ("mask", "external_mask"), self._external_mask)
            self.views.set_images(
                observed,
                model,
                residual,
                qx=result_qx,
                qy=result_qy,
                q_unit=str(_read(result, ("q_unit",), _read(self._qmap, ("q_unit", "unit"), "unknown")) or "unknown"),
                valid_mask=result_valid_mask,
                external_mask=result_external_mask,
            )
            ridge_points = _result_value(result, ("ridge_points", "ridges", "ridge"), [])
            ellipse_result = _result_value(result, ("ellipse_fit", "ellipse", "ellipse_result"), None)
            ellipses = _result_value(result, ("ellipses", "ellipse_fits"), None)
            if not ellipses and ellipse_result is not None:
                ellipses = _result_value(ellipse_result, ("ellipses", "ellipse_pair"), [])
            if ellipses and isinstance(ellipses, Mapping):
                ellipses = [ellipses]
            self.set_fit_overlay(ridge_points, ellipses)
            self._update_metrics(result, observed, residual)
            self._update_measurements(result)

            fitted = _result_value(result, ("parameters", "fitted_parameters", "params"), None)
            if fitted:
                self.parameter_model.set_rows(fitted)
                setter = getattr(self.engine, "set_parameters", None)
                if callable(setter):
                    try:
                        setter(fitted)
                    except Exception:
                        # Injected/legacy engines may expose a narrower setter;
                        # the accepted UI state remains authoritative.
                        pass

        def _update_batch_rows(self, records: Iterable[Any]) -> None:
            rows = list(records)
            if not rows:
                return
            self.batch_table.setRowCount(len(rows))
            for row_index, record in enumerate(rows):
                mapping = record if isinstance(record, Mapping) else {"value": record}
                frame = mapping.get("frame", mapping.get("path", row_index))
                status = mapping.get("status", "ok")
                rmse = mapping.get("rmse", mapping.get("metrics", {}).get("rmse") if isinstance(mapping.get("metrics"), Mapping) else None)
                self.batch_table.setItem(row_index, 0, QtWidgets.QTableWidgetItem(str(frame)))
                self.batch_table.setItem(row_index, 1, QtWidgets.QTableWidgetItem(str(status)))
                self.batch_table.setItem(row_index, 2, QtWidgets.QTableWidgetItem(_format_metric(rmse)))

        def _update_measurements(self, result: Any) -> None:
            """Render measured profiles and quality tables from plain results."""

            observables = _result_value(result, ("observables", "measurements"), None)
            self.measurement_observables = observables
            angular = _read(observables, ("angular", "angular_spectrum"), None)
            ridge = _read(observables, ("ridge", "ridges", "ridge_track"), None)
            lobes = _sequence(_read(observables, ("lobes", "lobe_metrics", "four_lobes"), []))
            ellipse = _result_value(result, ("ellipse_fit", "ellipse", "ellipse_result"), None)
            if ellipse is None:
                ellipse = _read(observables, ("ellipse",), None)

            if self.angular_plot is not None:
                self.angular_plot.clear()
                angle = _read(angular, ("angle_deg",), None)
                if angle is None:
                    raw_angle = _sequence(_read(angular, ("angle", "azimuth"), []))
                    angle = [math.degrees(float(item)) for item in raw_angle]
                intensity = _sequence(_read(angular, ("intensity", "profile"), []))
                coverage = _sequence(_read(angular, ("coverage",), []))
                try:
                    self.angular_plot.plot(
                        list(angle),
                        list(intensity),
                        pen=_pg.mkPen(50, 150, 255, width=2),
                        name="intensity",
                    )
                    if len(coverage):
                        self.angular_plot.plot(
                            list(angle),
                            list(coverage),
                            pen=_pg.mkPen(255, 190, 55, width=1),
                            name="coverage",
                        )
                except (TypeError, ValueError):
                    pass

            point_source = _read(ridge, ("points", "observed_points"), None)
            if point_source is None:
                point_source = _result_value(result, ("ridge_points", "ridges"), [])
            point_rows = _sequence(point_source)
            if self.ridge_plot is not None:
                self.ridge_plot.clear()
                accepted_x: list[float] = []
                accepted_y: list[float] = []
                rejected_x: list[float] = []
                rejected_y: list[float] = []
                q_unit = str(_read(ridge, ("q_unit",), "unknown") or "unknown")
                self.ridge_plot.setLabel("left", f"Ridge q ({q_unit})")
                for point in point_rows:
                    raw_angle = _read(point, ("angle_deg",), None)
                    if raw_angle is None:
                        raw_angle = _read(point, ("angle", "azimuth", "phi"), None)
                        raw_angle = math.degrees(float(raw_angle)) if raw_angle is not None else None
                    q_value = _read(point, ("q", "q_star", "q_position"), None)
                    if raw_angle is None or q_value is None:
                        continue
                    try:
                        x_value, y_value = float(raw_angle), float(q_value)
                    except (TypeError, ValueError):
                        continue
                    accepted = bool(_read(point, ("accepted", "valid"), True))
                    (accepted_x if accepted else rejected_x).append(x_value)
                    (accepted_y if accepted else rejected_y).append(y_value)
                if accepted_x:
                    self.ridge_plot.plot(
                        accepted_x,
                        accepted_y,
                        pen=None,
                        symbol="o",
                        symbolSize=6,
                        symbolBrush=_pg.mkBrush(70, 210, 120, 210),
                        name="accepted",
                    )
                if rejected_x:
                    self.ridge_plot.plot(
                        rejected_x,
                        rejected_y,
                        pen=None,
                        symbol="x",
                        symbolSize=8,
                        symbolPen=_pg.mkPen(230, 90, 90, width=2),
                        name="rejected",
                    )

            self.ridge_table.setRowCount(len(point_rows))
            for row_index, point in enumerate(point_rows):
                raw_angle = _read(point, ("angle_deg",), None)
                if raw_angle is None:
                    raw_angle = _read(point, ("angle", "azimuth", "phi"), None)
                    raw_angle = math.degrees(float(raw_angle)) if raw_angle is not None else None
                values = (
                    _format_metric(raw_angle),
                    _format_metric(_read(point, ("q", "q_star", "q_position"), None)),
                    str(bool(_read(point, ("accepted", "valid"), True))),
                    str(_read(point, ("method",), "observed")),
                )
                for column, value in enumerate(values):
                    self.ridge_table.setItem(row_index, column, QtWidgets.QTableWidgetItem(value))

            self.lobe_table.setRowCount(len(lobes))
            for row_index, lobe in enumerate(lobes):
                angle = _read(lobe, ("angle_deg",), None)
                if angle is None:
                    raw_angle = _read(lobe, ("angle", "azimuth"), None)
                    angle = math.degrees(float(raw_angle)) if raw_angle is not None else None
                values = (
                    _format_metric(angle),
                    _format_metric(_read(lobe, ("intensity",), None)),
                    _format_metric(_read(lobe, ("baseline",), None)),
                    _format_metric(_read(lobe, ("snr",), None)),
                    _format_metric(_read(lobe, ("fwhm_deg",), None)),
                    _format_metric(_read(lobe, ("coverage",), None)),
                    str(bool(_read(lobe, ("valid", "accepted"), True))),
                    ", ".join(str(item) for item in _sequence(_read(lobe, ("flags",), ()))),
                )
                for column, value in enumerate(values):
                    self.lobe_table.setItem(row_index, column, QtWidgets.QTableWidgetItem(value))

            ellipse_rows = (
                ("a (major q)", _read(ellipse, ("a",), None)),
                ("b (minor q)", _read(ellipse, ("b",), None)),
                ("axis ratio", _read(ellipse, ("axis_ratio", "axes_ratio"), None)),
                ("ellipticity", _read(ellipse, ("ellipticity", "eccentricity"), None)),
                # Keep ellipse theta semantically separate from lobe-derived
                # phi/alpha/psi; no relabelling is performed here.
                ("theta (ellipse axis, deg)", _read(ellipse, ("theta_deg", "angle_deg"), None)),
                ("Ln from minor axis (nm)", _read(ellipse, ("Ln_from_minor_axis_nm",), None)),
                ("Lz from draw axis (nm)", _read(ellipse, ("Lz_from_draw_axis_nm",), None)),
                ("RMSE", _read(ellipse, ("rmse", "residual_rms"), None)),
                ("RSS", _read(ellipse, ("rss",), None)),
                ("n points", _read(ellipse, ("n_points", "n_data"), None)),
                ("quality / success", _read(ellipse, ("success",), None)),
                ("flags", ", ".join(str(item) for item in _sequence(_read(ellipse, ("flags",), ())))),
                ("phi app (lobe-derived, deg)", _read(observables, ("phi_app_deg",), None)),
                ("alpha candidate (not inferred)", _read(observables, ("alpha_candidate_deg",), None)),
                ("psi candidate (not inferred)", _read(observables, ("psi_candidate_deg",), None)),
            )
            self.ellipse_table.setRowCount(len(ellipse_rows) if ellipse is not None or observables is not None else 0)
            for row_index, (name, value) in enumerate(ellipse_rows if ellipse is not None or observables is not None else ()):
                self.ellipse_table.setItem(row_index, 0, QtWidgets.QTableWidgetItem(str(name)))
                self.ellipse_table.setItem(row_index, 1, QtWidgets.QTableWidgetItem(_format_metric(value)))

        def _update_metrics(self, result: Any, observed: Any, residual: Any) -> None:
            metrics = _result_value(result, ("metrics", "statistics", "summary"), {})
            if not isinstance(metrics, Mapping):
                metrics = {}
            ellipse_result = _result_value(result, ("ellipse_fit", "ellipse", "ellipse_result"), None)
            rmse = _result_value(
                metrics,
                ("rmse", "RMSE"),
                _result_value(
                    result,
                    ("rmse", "RMSE"),
                    _result_value(ellipse_result, ("rmse", "residual_rms"), None),
                ),
            )
            ndata = _result_value(metrics, ("ndata", "n_data", "points"), _result_value(result, ("ndata", "n_data"), None))
            if rmse is None and residual is not None and _np is not None:
                try:
                    values = _np.asarray(residual, dtype=float)
                    finite = values[_np.isfinite(values)]
                    if finite.size:
                        rmse = float(_np.sqrt(_np.mean(finite**2)))
                except Exception:
                    pass
            if ndata is None and observed is not None and _np is not None:
                try:
                    values = _np.asarray(observed)
                    ndata = int(_np.isfinite(values).sum()) if values.dtype.kind in "fc" else int(values.size)
                except Exception:
                    pass
            flags = _result_value(metrics, ("flags", "flag"), _result_value(result, ("flags", "flag"), []))
            coverage = _result_value(metrics, ("valid_fraction", "coverage", "valid_coverage"), None)
            if isinstance(flags, str):
                flags_text = flags
            else:
                flags_text = ", ".join(str(item) for item in (flags or [])) or "—"
            self.last_metrics = {"rmse": rmse, "ndata": ndata, "flags": flags, "valid_fraction": coverage}
            self.rmse_label.setText(f"RMSE: {_format_metric(rmse)}")
            self.ndata_label.setText(f"ndata: {ndata if ndata is not None else '—'}")
            self.flags_label.setText(f"flags: {flags_text}")
            try:
                coverage_text = _format_metric(float(coverage) * 100.0) + "%" if coverage is not None else "—"
            except (TypeError, ValueError):
                coverage_text = "—"
            self.coverage_label.setText(f"coverage: {coverage_text}")

        # ----- project persistence ---------------------------------------------

        def project_to_dict(self) -> dict[str, Any]:
            return {
                "schema_version": 1,
                "parameters": self.parameter_model.parameter_dict(),
                "analysis": _jsonable(self.analysis_settings),
                "input": self._source_path,
                "poni": self._poni_path,
                "frame": self._frame,
                "dataset": self._dataset,
                "mask": self._mask_path,
                "mask_frame": self._mask_frame,
                "mask_dataset": self._mask_dataset,
                "roi_exclusion": _jsonable(self._exclusion_roi),
                "rois": list(self._roi_specs),
                "batch": {
                    "mode": self.batch_mode_combo.currentData(),
                    "frames": _jsonable(list(self.batch_frames)),
                    "manifest": self.batch_manifest_edit.text() or None,
                    "checkpoint": self.batch_checkpoint_edit.text() or None,
                    "resume": self.batch_resume_check.isChecked(),
                    "output": self.batch_output_edit.text() or None,
                },
                "metadata": {
                    "project_path": str(self._project_path) if self._project_path else None,
                    "config_path": self._config_path,
                    "qmap_shape": list(getattr(self._qmap, "shape", ())) if self._qmap is not None else None,
                },
            }

        def save_project(self, path: str | Path | bool | None = None) -> bool:
            if isinstance(path, bool):
                path = None
            if path is None:
                chosen, _ = QtWidgets.QFileDialog.getSaveFileName(
                    self, "Save LamellarSAXS2D project", "", "JSON project (*.json)"
                )
                if not chosen:
                    return False
                path = chosen
            target = Path(path)
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(
                    json.dumps(_jsonable(self.project_to_dict()), indent=2, allow_nan=False),
                    encoding="utf-8",
                )
            except (OSError, TypeError, ValueError) as exc:
                self._set_status(f"Save failed: {exc}", flags="error")
                return False
            self._project_path = target
            self._set_status(f"Saved {target.name}")
            return True

        def load_project(self, path: str | Path | bool | None = None) -> bool:
            if isinstance(path, bool):
                path = None
            if path is None:
                chosen, _ = QtWidgets.QFileDialog.getOpenFileName(
                    self, "Open LamellarSAXS2D project", "", "JSON project (*.json)"
                )
                if not chosen:
                    return False
                path = chosen
            target = Path(path).resolve()
            try:
                data = json.loads(
                    target.read_text(encoding="utf-8"),
                    parse_constant=lambda value: (_ for _ in ()).throw(
                        ValueError(f"non-finite JSON constant is not allowed: {value}")
                    ),
                )
                if not isinstance(data, Mapping):
                    raise ValueError("project root must be an object")
                self._invalidate_pending_work(clear_fit=False)
                self._auto_scale_initial = False
                self.parameter_model.set_rows(data.get("parameters", {}))
                analysis = data.get("analysis", data.get("measurement", data.get("analysis_settings", {})))
                if isinstance(analysis, Mapping):
                    self.set_analysis_settings(analysis, trigger_preview=False)
                project_base = target.parent
                source = _resolve_project_path(data.get("input", data.get("input_path")), project_base)
                poni = _resolve_project_path(data.get("poni", data.get("poni_path")), project_base)
                frame = data.get("frame")
                dataset = data.get("dataset")
                mask = _resolve_project_path(data.get("mask", data.get("mask_path")), project_base)
                mask_frame = data.get("mask_frame")
                mask_dataset = data.get("mask_dataset")
                roi = data.get("roi_exclusion")
                rois = data.get("rois")
                self._frame = int(frame) if frame is not None else None
                self._dataset = str(dataset) if dataset is not None else None
                self._mask_frame = int(mask_frame) if mask_frame is not None else None
                self._mask_dataset = str(mask_dataset) if mask_dataset is not None else None
                batch = data.get("batch")
                if isinstance(batch, Mapping):
                    mode_index = self.batch_mode_combo.findData(batch.get("mode", "independent"))
                    if mode_index >= 0:
                        self.batch_mode_combo.setCurrentIndex(mode_index)
                    for widget, key in (
                        (self.batch_manifest_edit, "manifest"),
                        (self.batch_checkpoint_edit, "checkpoint"),
                        (self.batch_output_edit, "output"),
                    ):
                        value = _resolve_project_path(batch.get(key), project_base)
                        widget.setText("" if value is None else str(value))
                    self.batch_resume_check.setChecked(bool(batch.get("resume", False)))
                    frames = batch.get("frames", data.get("batch_frames", []))
                    self.set_batch_frames(
                        _resolve_project_frame(item, project_base) for item in (frames or [])
                    )
                if poni:
                    self.set_poni(poni)
                if source:
                    if not self.open_image(
                        source,
                        frame=self._frame,
                        dataset=self._dataset,
                        poni=poni,
                        external_mask=mask,
                        mask_frame=self._mask_frame,
                        mask_dataset=self._mask_dataset,
                    ):
                        raise ValueError(f"could not load project input: {source}")
                elif mask:
                    if not self.select_mask(
                        mask,
                        mask_frame=self._mask_frame,
                        mask_dataset=self._mask_dataset,
                    ):
                        raise ValueError(f"could not load project mask: {mask}")
                if mask and self._file_mask is None:
                    self._load_external_mask(
                        mask,
                        frame=self._mask_frame,
                        dataset=self._mask_dataset,
                    )
                if rois and isinstance(rois, Iterable):
                    self._roi_specs = [dict(spec) for spec in rois if isinstance(spec, Mapping)]
                    self._exclusion_roi = self._roi_specs[-1] if self._roi_specs else None
                    if self._roi_specs and not self._recompute_external_mask(update_widgets=True):
                        raise ValueError("could not apply project ROIs")
                elif roi is not None:
                    if not self.set_exclusion_roi(roi):
                        raise ValueError("could not apply project ROI")
                setter = getattr(self.engine, "set_parameters", None)
                if callable(setter):
                    setter(self.parameter_model.parameter_dict())
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                self._set_status(f"Load failed: {exc}", flags="error")
                return False
            self._project_path = target
            self._set_status(f"Loaded {target.name}")
            return True

        open_project = load_project

        # ----- batch and evolution ---------------------------------------------

        def _choose_batch_files(self) -> None:
            files, _ = QtWidgets.QFileDialog.getOpenFileNames(
                self,
                "Select 2D SAXS frames",
                "",
                "Detector images (*.cbf *.edf *.tif *.tiff);;All files (*)",
            )
            if files:
                self.set_batch_frames(files)

        def set_batch_frames(self, frames: Iterable[Any]) -> None:
            self.batch_frames = list(frames)
            self.batch_table.setRowCount(len(self.batch_frames))
            for row, frame in enumerate(self.batch_frames):
                self.batch_table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(frame)))
                self.batch_table.setItem(row, 1, QtWidgets.QTableWidgetItem("Ready"))
                self.batch_table.setItem(row, 2, QtWidgets.QTableWidgetItem(""))

        def run_batch(self, frames: Iterable[Any] | bool | None = None) -> int:
            if frames is not None and not isinstance(frames, bool):
                self.set_batch_frames(frames)
            payload = {
                "frames": list(self.batch_frames),
                "parameters": self.parameter_model.parameter_dict(),
                "parameter_specs": self.parameter_model.parameter_dict(),
                "analysis": self.analysis_settings,
                "external_mask": self._external_mask,
                "qmap": self._qmap,
                "mode": self.batch_mode_combo.currentData(),
                "manifest": self.batch_manifest_edit.text() or None,
                "checkpoint": self.batch_checkpoint_edit.text() or None,
                "resume": self.batch_resume_check.isChecked(),
                "output": self.batch_output_edit.text() or None,
                "source": self._source_path,
                "poni": self._poni_path,
            }
            self.batchRequested.emit(payload["frames"])
            self.batchPayloadRequested.emit(payload)
            self.batch_progress.setVisible(True)
            return self._start_job("batch", payload)

        def plot_evolution(self, records: Iterable[Any]) -> None:
            self.evolution_records = list(records)
            if not self.evolution_records:
                self.evolution_table.setRowCount(0)
                self.evolution_parameter_combo.clear()
                if self.evolution_plot is not None:
                    self.evolution_plot.clear()
                return
            rows: list[Mapping[str, Any]] = []
            for index, record in enumerate(self.evolution_records):
                rows.append(_flatten_evolution_record(record, index))
            self._evolution_rows = rows
            keys = list(dict.fromkeys(key for row in rows for key in row.keys()))
            self.evolution_table.setColumnCount(len(keys))
            self.evolution_table.setHorizontalHeaderLabels([str(key) for key in keys])
            self.evolution_table.setRowCount(len(rows))
            for row_index, row in enumerate(rows):
                for column, key in enumerate(keys):
                    self.evolution_table.setItem(row_index, column, QtWidgets.QTableWidgetItem(str(row.get(key, ""))))
            ignored = {"time", "time_s", "frame", "index", "status", "path"}
            numeric_keys = [
                str(key)
                for key in keys
                if key not in ignored and any(_is_finite(_numeric(row.get(key), float("nan"))) for row in rows)
            ]
            self.evolution_parameter_combo.blockSignals(True)
            self.evolution_parameter_combo.clear()
            self.evolution_parameter_combo.addItems(numeric_keys)
            preferred = self.evolution_y_key if self.evolution_y_key in numeric_keys else (
                "rmse" if "rmse" in numeric_keys else (numeric_keys[0] if numeric_keys else "")
            )
            if preferred:
                self.evolution_parameter_combo.setCurrentText(preferred)
            self.evolution_parameter_combo.blockSignals(False)
            self.evolution_y_key = preferred
            self._render_evolution(preferred)

        def _render_evolution(self, y_key: str | None = None) -> None:
            """Render the selected flattened parameter against time/frame."""

            if y_key is None:
                y_key = self.evolution_parameter_combo.currentText()
            self.evolution_y_key = str(y_key or "")
            if self.evolution_plot is None:
                return
            self.evolution_plot.clear()
            self.evolution_plot.setLabel("left", self.evolution_y_key or "value")
            rows = self._evolution_rows
            x_values = [
                _numeric(row.get("time", row.get("time_s", row.get("frame", index))), index)
                for index, row in enumerate(rows)
            ]
            y_values = [_numeric(row.get(self.evolution_y_key, float("nan")), float("nan")) for row in rows]
            finite = [(x, y) for x, y in zip(x_values, y_values) if _is_finite(x) and _is_finite(y)]
            if finite:
                self.evolution_plot.plot(
                    [point[0] for point in finite],
                    [point[1] for point in finite],
                    pen=_pg.mkPen(70, 170, 255, width=2),
                    symbol="o",
                    symbolSize=6,
                )

        # ----- status and lifetime ---------------------------------------------

        def _set_busy(self, busy: bool, kind: str = "") -> None:
            self.preview_button.setEnabled(not busy)
            self.optimize_button.setEnabled(not busy)
            self.batch_run_button.setEnabled(not busy)
            self.cancel_button.setEnabled(bool(busy))
            self.ignore_late_result_button.setEnabled(bool(busy))
            self.batch_progress.setVisible(busy and kind == "batch")
            if busy:
                self._set_status(f"Running {kind}…")
            elif kind:
                status = {
                    "preview": "Preview complete",
                    "optimize": "Optimize complete",
                    "batch": "Batch complete",
                    "cancelled": "Cancelled",
                    "ignored": "Late result ignored",
                }.get(kind, "Ready")
                self._set_status(status)

        def _set_status(self, text: str, *, flags: str | None = None) -> None:
            self.status_message.setText(text)
            if flags is not None:
                self.flags_label.setText(f"flags: {flags}")

        def closeEvent(self, event: Any) -> None:  # noqa: N802 - Qt API
            self.cancel_jobs()
            self._thread_pool.clear()
            self._thread_pool.waitForDone(1500)
            event.accept()


else:

    class RefinementMainWindow:
        """Import-safe placeholder when PySide6 is not installed."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs
            require_qt()


MainWindow = RefinementMainWindow
Workbench = RefinementMainWindow
RefinementWindow = RefinementMainWindow
WorkbenchWindow = RefinementMainWindow


def _flatten_evolution_record(record: Any, index: int) -> dict[str, Any]:
    """Flatten batch ``parameters`` specs into scalar evolution columns."""

    if isinstance(record, Mapping):
        row: dict[str, Any] = dict(record)
    else:
        row = {"frame": index, "value": record}
    parameters = row.pop("parameters", None)
    if isinstance(parameters, Mapping):
        for name, value in parameters.items():
            scalar = _read(value, ("value", "val", "initial", "best"), None)
            if scalar is None and not isinstance(value, Mapping):
                scalar = value
            row[str(name)] = scalar
    metrics = row.get("metrics")
    if isinstance(metrics, Mapping):
        for name, value in metrics.items():
            row.setdefault(str(name), _read(value, ("value", "val"), value))
    return row


def _format_metric(value: Any) -> str:
    if value is None:
        return "—"
    try:
        number = float(value)
        return "—" if not math.isfinite(number) else f"{number:.6g}"
    except (TypeError, ValueError):
        return str(value)


def _numeric(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _is_finite(value: Any) -> bool:
    try:
        return bool(_np.isfinite(value)) if _np is not None else value == value
    except (TypeError, ValueError):
        return False


def _gui_options(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("input", nargs="?")
    parser.add_argument("--input", dest="input_option")
    parser.add_argument("--poni")
    parser.add_argument("-c", "--config")
    parser.add_argument("--frame", type=int)
    parser.add_argument("--dataset")
    parser.add_argument("--mask-frame", type=int)
    parser.add_argument("--mask-dataset")
    parser.add_argument("--no-auto-preview", action="store_true")
    raw = list(sys.argv[1:] if argv is None else argv)
    # QApplication should never be asked to interpret our scientific options;
    # only this small parser consumes them.
    return parser.parse_known_args(raw)[0]


def create_app(
    argv: list[str] | None = None,
    *,
    analysis_service: Any = None,
    input_path: str | Path | None = None,
    poni: str | Path | Any | None = None,
    config: Any = None,
    config_path: str | Path | None = None,
    mask_frame: int | None = None,
    mask_dataset: str | None = None,
) -> tuple[Any, RefinementMainWindow]:
    """Create a QApplication and a real-service workbench without showing it."""

    require_qt()
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    options = _gui_options(argv)
    project_config = config
    selected_config_path = config_path or getattr(options, "config", None)
    if project_config is None and selected_config_path:
        from ..project import load_project

        project_config = load_project(selected_config_path).resolve_paths(Path(selected_config_path).parent)
    elif isinstance(project_config, (str, Path)):
        from ..project import load_project

        selected_config_path = str(project_config)
        project_config = load_project(project_config).resolve_paths(Path(project_config).parent)
    elif project_config is not None and not hasattr(project_config, "input_paths"):
        from ..project import ProjectConfig

        project_config = ProjectConfig.from_mapping(project_config)
    analysis_options = getattr(project_config, "analysis", {}) if project_config is not None else {}
    configured_parameters = analysis_options.get("parameters") if isinstance(analysis_options, Mapping) else None
    configured_frame = analysis_options.get("frame") if isinstance(analysis_options, Mapping) else None
    configured_dataset = analysis_options.get("dataset") if isinstance(analysis_options, Mapping) else None
    configured_mask = analysis_options.get("mask") if isinstance(analysis_options, Mapping) else None
    configured_mask_frame = analysis_options.get("mask_frame") if isinstance(analysis_options, Mapping) else None
    configured_mask_dataset = analysis_options.get("mask_dataset") if isinstance(analysis_options, Mapping) else None
    selected_poni = poni or getattr(options, "poni", None) or getattr(project_config, "poni_path", None)
    selected_frame = getattr(options, "frame", None)
    if selected_frame is None:
        selected_frame = configured_frame
    selected_dataset = getattr(options, "dataset", None) or configured_dataset
    selected_mask_frame = mask_frame
    if selected_mask_frame is None:
        selected_mask_frame = getattr(options, "mask_frame", None)
    if selected_mask_frame is None:
        selected_mask_frame = configured_mask_frame
    selected_mask_dataset = mask_dataset or getattr(options, "mask_dataset", None) or configured_mask_dataset
    service = analysis_service
    if service is None:
        from ..service import ButterflyAnalysisService

        service = ButterflyAnalysisService(
            poni=selected_poni,
            parameters=configured_parameters,
            analysis_settings=analysis_options if isinstance(analysis_options, Mapping) else None,
        )
    window = RefinementMainWindow(
        analysis_service=service,
        parameters=configured_parameters,
        analysis_settings=analysis_options if isinstance(analysis_options, Mapping) else None,
        mask_frame=selected_mask_frame,
        mask_dataset=selected_mask_dataset,
        auto_preview=not bool(options.no_auto_preview),
    )
    selected_input = (
        input_path
        or getattr(options, "input_option", None)
        or getattr(options, "input", None)
        or (getattr(project_config, "input_paths", [None]) or [None])[0]
    )
    if selected_input:
        window.open_image(
            selected_input,
            frame=selected_frame,
            dataset=selected_dataset,
            poni=selected_poni,
            external_mask=configured_mask,
            mask_frame=selected_mask_frame,
            mask_dataset=selected_mask_dataset,
        )
    elif selected_poni:
        window.set_poni(selected_poni)
    if selected_config_path:
        window._config_path = str(selected_config_path)
    return app, window


def launch(
    argv: list[str] | None = None,
    *,
    input_path: str | Path | None = None,
    poni: str | Path | Any | None = None,
    analysis_service: Any = None,
    config: Any = None,
    config_path: str | Path | None = None,
    mask_frame: int | None = None,
    mask_dataset: str | None = None,
) -> int:
    """Run the workbench as a small standalone entry point."""

    app, window = create_app(
        argv,
        analysis_service=analysis_service,
        input_path=input_path,
        poni=poni,
        config=config,
        config_path=config_path,
        mask_frame=mask_frame,
        mask_dataset=mask_dataset,
    )
    window.show()
    return int(app.exec())


__all__ = [
    "MainWindow",
    "RefinementMainWindow",
    "RefinementWindow",
    "Workbench",
    "WorkbenchWindow",
    "create_app",
    "launch",
    "symmetric_ellipses",
]
