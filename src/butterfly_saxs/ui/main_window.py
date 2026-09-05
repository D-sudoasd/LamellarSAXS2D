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
from datetime import datetime, timezone
import inspect
import json
import math
import sys
import threading
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Callable

from .i18n import DEFAULT_LANGUAGE, LANGUAGE_SETTING_KEY, translate, validate_language
from .models import ParameterRow, ParameterTableModel
from .project_document import ProjectDocumentController
from .qt_compat import QT_AVAILABLE, QtCore, QtGui, QtWidgets, require_qt
from .views import PLOT_AVAILABLE, ViewGrid, _disable_auto_si_prefix
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
    """Compatibility alias for the Qt-free project document boundary."""

    return ProjectDocumentController.resolve_path(value, base)


def _resolve_project_frame(value: Any, base: Path) -> Any:
    """Compatibility alias for the Qt-free project document boundary."""

    return ProjectDocumentController.resolve_frame(value, base)


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
            for name in (
                "preview", "predict", "evaluate", "render", "simulate", "run",
                "optimize", "fit", "refine", "batch", "measure_geometry",
                "remeasure_geometry", "refine_geometry", "fit_geometry",
            )
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
            "measure_geometry": ("measure_geometry", "remeasure_geometry", "preview", "predict", "evaluate", "run"),
            "refine_geometry": ("refine_geometry", "fit_geometry", "measure_geometry", "remeasure_geometry", "preview", "run"),
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

    if value is None or (
        isinstance(value, str) and value.strip().lower() in {"", "auto", "自动"}
    ):
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


_BATCH_STATUS_KEYS = {
    "ready": "status.ready",
    "ok": "status.batch_ok",
    "failed": "status.batch_failed",
    "skipped": "status.batch_skipped",
}


def _parameter_number(parameters: Mapping[str, Any], names: tuple[str, ...]) -> float | None:
    """Read one scalar from the editable table's plain parameter mapping."""

    for name in names:
        value = parameters.get(name)
        if isinstance(value, Mapping):
            value = _read(value, ("value", "val", "initial", "best"), None)
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            return number
    return None


def model_ellipse_pair(parameters: Mapping[str, Any], *, reference_axis_deg: float = 0.0) -> list[dict[str, Any]]:
    """Build the two editable model ellipses for the q-space overlay.

    These curves are generated only from the current parameter table.  They
    are intentionally returned separately from measured-fit ellipses so a
    preview cannot be mistaken for an observation-derived fit.
    """

    a = _parameter_number(parameters, ("a", "q_major"))
    ratio = _parameter_number(parameters, ("axis_ratio",))
    b = _parameter_number(parameters, ("b", "q_minor"))
    if b is None and a is not None and ratio is not None:
        b = a * ratio if ratio <= 1.0 else a / ratio
    if a is None or b is None or a <= 0.0 or b <= 0.0:
        return []
    theta_deg = _parameter_number(parameters, ("theta_deg", "orientation_deg", "angle_deg"))
    if theta_deg is None:
        theta = _parameter_number(parameters, ("theta", "orientation", "angle"))
        theta_deg = math.degrees(theta) if theta is not None else 0.0
    return [
        {
            "cx": 0.0,
            "cy": 0.0,
            "a": float(a),
            "b": float(b),
            "angle_deg": float(reference_axis_deg) + float(theta_deg),
            "source": "model",
            "branch_id": 0,
            "branch": "ellipse_a",
        },
        {
            "cx": 0.0,
            "cy": 0.0,
            "a": float(a),
            "b": float(b),
            "angle_deg": float(reference_axis_deg) - float(theta_deg),
            "source": "model",
            "branch_id": 1,
            "branch": "ellipse_b",
        },
    ]


def _result_has_failure(result: Any) -> bool:
    """Return whether a result carries an explicit failure condition."""

    try:
        # Batch and interactive review must agree on explicit failure signals
        # (including success=False, nested full2d/ellipse failures, and empty
        # observations).  Keep the UI fallback below for lightweight probes
        # that intentionally do not import the batch layer.
        from ..batch import _quality_failure_reason

        if _quality_failure_reason(result) is not None:
            return True
    except Exception:
        pass
    status = str(_read(result, ("status", "solver_status"), "") or "").lower()
    if status in {"fail", "failed", "error"}:
        return True
    quality = str(_read(result, ("quality_status",), "") or "").upper()
    if quality == "FAIL":
        return True
    metrics = _read(result, ("metrics", "statistics", "summary"), {})
    if isinstance(metrics, Mapping) and metrics.get("success") is False:
        return True
    ellipse = _read(result, ("ellipse_fit", "ellipse", "ellipse_result"), None)
    ellipse_status = str(_read(ellipse, ("quality_status", "status"), "") or "").upper()
    ellipse_quality = _read(ellipse, ("quality",), {})
    nested_quality_status = str(
        _read(ellipse_quality, ("status", "engineering_status"), "") or ""
    ).upper()
    if ellipse_status in {"FAIL", "FAILED", "INVALID"} or nested_quality_status in {
        "FAIL",
        "FAILED",
        "INVALID",
    }:
        return True
    flags = _read(metrics, ("flags", "flag"), _read(result, ("flags", "flag"), []))
    if isinstance(flags, str):
        flags = [flags]
    return any(
        any(token in str(flag).lower() for token in ("failed", "error", "invalid", "no_engine", "exception"))
        for flag in (flags or [])
    )


def _new_fit_session() -> dict[str, Any]:
    """Return the small, JSON-friendly manual-review session state."""

    return {
        "manual_status": "unreviewed",
        "reviewed_by": "",
        "reviewed_at": None,
        "review_notes": "",
        "optimize_before": None,
        "optimize_after": None,
        "accepted_parameters": None,
        "snapshots": [],
    }


def _utc_timestamp() -> str:
    """Return an explicit UTC timestamp for human-review audit fields."""

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


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

    class CompactDoubleSpinBox(QtWidgets.QDoubleSpinBox):
        """Bounded-width numeric field that still accepts large scientific values."""

        _COMPACT_WIDTH = 88

        def sizeHint(self) -> Any:  # noqa: N802 - Qt API
            size = super().sizeHint()
            size.setWidth(min(size.width(), self._COMPACT_WIDTH))
            return size

        def minimumSizeHint(self) -> Any:  # noqa: N802 - Qt API
            size = super().minimumSizeHint()
            size.setWidth(min(size.width(), self._COMPACT_WIDTH))
            return size

    class RefinementMainWindow(QtWidgets.QMainWindow):
        """Main 2D SAXS refinement window with explicit preview/optimize paths."""

        batchRequested = QtCore.Signal(object)
        batchPayloadRequested = QtCore.Signal(object)
        previewRequested = QtCore.Signal(int)
        optimizeRequested = QtCore.Signal(int)
        geometryMeasureRequested = QtCore.Signal(int)
        geometryRefineRequested = QtCore.Signal(int)

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
            language: str | None = None,
            settings: Any = None,
        ) -> None:
            super().__init__(parent)
            self._settings = (
                settings
                if settings is not None
                else QtCore.QSettings("LamellarSAXS2D", "LamellarSAXS2D")
            )
            if language is None:
                stored_language = self._settings.value(
                    LANGUAGE_SETTING_KEY,
                    DEFAULT_LANGUAGE,
                )
                try:
                    self._language = validate_language(stored_language)
                except ValueError:
                    self._language = DEFAULT_LANGUAGE
            else:
                self._language = validate_language(language)
            self._status_key = "status.ready"
            self._status_values: dict[str, Any] = {}
            self._displayed_flags_text = "—"
            self._metric_display = {"rmse": "—", "ndata": "—", "coverage": "—"}
            self._ridge_plot_q_unit = "map unit"
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
            self._cancel_events: dict[int, threading.Event] = {}
            self._closing = False
            self._batch_progress_state: dict[str, Any] = {
                "completed": 0,
                "total": 0,
                "elapsed_s": 0.0,
            }
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
            self._last_result_signature: str | None = None
            self._last_result_kind: str | None = None
            self._geometry_only_result = False
            self._last_result_input_records: dict[str, Any] | None = None
            self._pending_input_records: dict[int, dict[str, Any]] = {}
            self._loaded_input_records: dict[str, dict[str, Any]] = {
                role: {"path": None, "exists": False}
                for role in ("source", "poni", "mask")
            }
            self._last_evidence_paths: dict[str, Path] = {}
            self._last_error: str | None = None
            self._fit_ridge_points: Any = []
            self._rejected_ridge_points: list[Any] = []
            self._observed_fit_ellipses: list[Any] = []
            self._model_ellipses: list[Any] = []
            self.last_metrics: dict[str, Any] = {}
            self._analysis_settings: dict[str, Any] = deepcopy(DEFAULT_ANALYSIS_SETTINGS)
            self._display_scale = "linear"
            self._display_percentile = 99.5
            self.evolution_records: list[Any] = []
            self._evolution_rows: list[Mapping[str, Any]] = []
            self.evolution_y_key = "rmse"
            self.batch_frames: list[Any] = []
            self._batch_cancel_event: Any = None
            self._fit_session: dict[str, Any] = _new_fit_session()
            # Project loading and programmatic restoration update several
            # widgets in sequence.  The flag keeps those internal updates
            # from being interpreted as a new manual review action.
            self._fit_session_restore_active = False

            source_parameters = parameters
            if source_parameters is None:
                source_parameters = _read(self.engine, ("parameters", "parameter_set", "params"), None)
            if source_parameters is None:
                source_parameters = _default_parameter_rows()
            self.parameter_model = ParameterTableModel(
                source_parameters,
                self,
                language=self._language,
            )

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

            self.setMinimumSize(980, 680)
            self.resize(1440, 900)
            self._project_controller = ProjectDocumentController(
                snapshot=self._snapshot_project_document,
                restore=self._restore_project_document,
                apply=self._apply_project_document,
                serialize=_jsonable,
            )
            self._retranslate_ui()
            self._set_status("status.ready")

        # ----- UI construction -------------------------------------------------

        def _build_actions(self) -> None:
            self.project_menu = self.menuBar().addMenu("&Project")
            self.project_menu.setToolTipsVisible(True)
            self.open_project_action = QtGui.QAction("Open project…", self)
            self.open_project_action.setObjectName("openProjectAction")
            self.open_project_action.triggered.connect(self.load_project)
            self.project_menu.addAction(self.open_project_action)
            self.save_project_action = QtGui.QAction("Save project…", self)
            self.save_project_action.setObjectName("saveProjectAction")
            self.save_project_action.triggered.connect(self.save_project)
            self.project_menu.addAction(self.save_project_action)
            self.open_image_action = QtGui.QAction("Open image…", self)
            self.open_image_action.setObjectName("openImageAction")
            self.open_image_action.triggered.connect(self.open_image)
            self.project_menu.addAction(self.open_image_action)
            self.open_poni_action = QtGui.QAction("Select PONI…", self)
            self.open_poni_action.setObjectName("openPoniAction")
            self.open_poni_action.triggered.connect(self.select_poni)
            self.project_menu.addAction(self.open_poni_action)
            self.open_mask_action = QtGui.QAction("Select external mask…", self)
            self.open_mask_action.setObjectName("openMaskAction")
            self.open_mask_action.triggered.connect(self.select_mask)
            self.project_menu.addAction(self.open_mask_action)
            self.clear_mask_action = QtGui.QAction("Clear mask", self)
            self.clear_mask_action.setObjectName("clearMaskAction")
            self.clear_mask_action.triggered.connect(self.clear_external_mask)
            self.project_menu.addAction(self.clear_mask_action)
            self.export_evidence_action = QtGui.QAction("Export evidence…", self)
            self.export_evidence_action.setObjectName("exportEvidenceAction")
            self.export_evidence_action.setToolTip(
                "导出当前 Preview/Optimize 的四图、参数、审核记录与 provenance；不会自动判定 scientific PASS"
            )
            self.export_evidence_action.triggered.connect(self.export_manual_evidence)
            self.project_menu.addAction(self.export_evidence_action)
            self.project_menu.addSeparator()
            self.close_action = QtGui.QAction("Close", self)
            self.close_action.setObjectName("closeAction")
            self.close_action.triggered.connect(self.close)
            self.project_menu.addAction(self.close_action)

            self.language_menu = self.menuBar().addMenu("&Language")
            self.language_menu.setToolTipsVisible(True)
            self.language_action_group = QtGui.QActionGroup(self)
            self.language_action_group.setExclusive(True)
            self.chinese_action = QtGui.QAction("中文", self)
            self.chinese_action.setObjectName("chineseLanguageAction")
            self.chinese_action.setCheckable(True)
            self.chinese_action.setData("zh_CN")
            self.english_action = QtGui.QAction("English", self)
            self.english_action.setObjectName("englishLanguageAction")
            self.english_action.setCheckable(True)
            self.english_action.setData("en")
            for action in (self.chinese_action, self.english_action):
                self.language_action_group.addAction(action)
                self.language_menu.addAction(action)
            self.language_action_group.triggered.connect(self._on_language_action)

            self.file_toolbar = self.addToolBar("Project")
            self.file_toolbar.setObjectName("projectToolbar")
            self.file_toolbar.addAction(self.open_project_action)
            self.file_toolbar.addAction(self.save_project_action)
            self.file_toolbar.addAction(self.open_image_action)
            self.file_toolbar.addAction(self.open_poni_action)
            self.file_toolbar.addAction(self.open_mask_action)
            self.file_toolbar.addAction(self.clear_mask_action)
            self.file_toolbar.addAction(self.export_evidence_action)

        def _build_central_pages(self) -> None:
            self.pages = QtWidgets.QTabWidget(self)
            self.pages.setObjectName("mainPages")
            self.refinement_page = QtWidgets.QWidget(self.pages)
            refinement_layout = QtWidgets.QVBoxLayout(self.refinement_page)
            refinement_layout.setContentsMargins(4, 4, 4, 4)
            self.views = ViewGrid(self.refinement_page, language=self._language)
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
            self.parameter_table_title = QtWidgets.QLabel(
                self._tr("label.parameter_table_title"),
                panel,
            )
            self.parameter_table_title.setObjectName("parameterTableTitle")
            self.parameter_table_title.setAccessibleName(
                self._tr("label.parameter_table_title")
            )
            self.parameter_table_title.setAccessibleDescription(
                self._tr("tooltip.parameter_table")
            )
            layout.addWidget(self.parameter_table_title, 0)
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
            # Keep enough room for the parameter table while allowing the
            # central four-view plot to remain useful on a 1280 px display.
            # The presentation wrapper adds a scroll area; this width is the
            # lower bound for readable labels rather than a fixed dock size.
            self.parameters_dock.setMinimumWidth(360)
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
            self.cancel_button.setShortcut(QtGui.QKeySequence("Esc"))
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

            self.roi_group = QtWidgets.QGroupBox("Exclusion ROI (pixel)", panel)
            roi_layout = QtWidgets.QGridLayout(self.roi_group)
            self.roi_type_combo = QtWidgets.QComboBox(self.roi_group)
            self.roi_type_combo.setObjectName("roiTypeCombo")
            self.roi_type_combo.addItem("Rectangle", "rectangle")
            self.roi_type_combo.addItem("Ellipse", "ellipse")
            self.roi_type_combo.currentIndexChanged.connect(self._update_roi_controls)
            self.roi_type_label = QtWidgets.QLabel("Type", self.roi_group)
            self.roi_type_label.setBuddy(self.roi_type_combo)
            roi_layout.addWidget(self.roi_type_label, 0, 0)
            roi_layout.addWidget(self.roi_type_combo, 0, 1, 1, 3)
            self.roi_x0 = CompactDoubleSpinBox(self.roi_group)
            self.roi_y0 = CompactDoubleSpinBox(self.roi_group)
            self.roi_x1 = CompactDoubleSpinBox(self.roi_group)
            self.roi_y1 = CompactDoubleSpinBox(self.roi_group)
            self.roi_cx = CompactDoubleSpinBox(self.roi_group)
            self.roi_cy = CompactDoubleSpinBox(self.roi_group)
            self.roi_rx = CompactDoubleSpinBox(self.roi_group)
            self.roi_ry = CompactDoubleSpinBox(self.roi_group)
            self.roi_angle = CompactDoubleSpinBox(self.roi_group)
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
                label_widget.setBuddy(spin)
                spin.setAccessibleName(self._tr("a11y.roi_rectangle", field=label))
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
                label_widget.setBuddy(spin)
                spin.setAccessibleName(self._tr("a11y.roi_ellipse", field=label))
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
            layout.addWidget(self.roi_group)

            self.fit_session_group = QtWidgets.QGroupBox("Fit session", panel)
            self.fit_session_form = QtWidgets.QFormLayout(self.fit_session_group)
            session_form = self.fit_session_form
            session_box = self.fit_session_group
            self.manual_status_label = QtWidgets.QLabel("unreviewed", session_box)
            self.manual_status_label.setObjectName("manualStatusLabel")
            session_form.addRow("Manual status", self.manual_status_label)
            self.reviewer_edit = QtWidgets.QLineEdit(session_box)
            self.reviewer_edit.setObjectName("reviewerEdit")
            self.reviewer_edit.setAccessibleName(self._tr("label.reviewer"))
            self.reviewer_edit.setPlaceholderText("required for Accept/Reject")
            session_form.addRow("Reviewer", self.reviewer_edit)
            self.review_notes_edit = QtWidgets.QLineEdit(session_box)
            self.review_notes_edit.setObjectName("reviewNotesEdit")
            self.review_notes_edit.setAccessibleName(self._tr("label.review_notes"))
            self.review_notes_edit.setPlaceholderText("optional review note")
            session_form.addRow("Review notes", self.review_notes_edit)

            review_buttons = QtWidgets.QVBoxLayout()
            self.accept_current_button = QtWidgets.QPushButton("Accept current", session_box)
            self.accept_current_button.setObjectName("acceptCurrentButton")
            self.accept_current_button.setToolTip(
                "显式接受当前 Preview/Optimize 结果；仅代表人工会话审核，不是 scientific PASS"
            )
            self.accept_current_button.clicked.connect(self.accept_current)
            review_buttons.addWidget(self.accept_current_button)
            self.reject_current_button = QtWidgets.QPushButton("Reject current", session_box)
            self.reject_current_button.setObjectName("rejectCurrentButton")
            self.reject_current_button.setToolTip("显式拒绝当前 Preview/Optimize 结果并保留审计记录")
            self.reject_current_button.clicked.connect(self.reject_current)
            review_buttons.addWidget(self.reject_current_button)
            self.restore_before_optimize_button = QtWidgets.QPushButton("Restore before optimize", session_box)
            self.restore_before_optimize_button.setObjectName("restoreBeforeOptimizeButton")
            self.restore_before_optimize_button.setToolTip("恢复最近一次 Optimize 启动前的完整参数表")
            self.restore_before_optimize_button.clicked.connect(self.restore_before_optimize)
            review_buttons.addWidget(self.restore_before_optimize_button)
            session_form.addRow(review_buttons)

            self.snapshot_note_edit = QtWidgets.QLineEdit(session_box)
            self.snapshot_note_edit.setObjectName("snapshotNoteEdit")
            self.snapshot_note_edit.setAccessibleName(self._tr("a11y.snapshot_note"))
            self.snapshot_note_edit.setPlaceholderText("required snapshot note")
            self.snapshot_save_row = QtWidgets.QHBoxLayout()
            snapshot_save_row = self.snapshot_save_row
            snapshot_save_row.addWidget(self.snapshot_note_edit, 1)
            self.save_snapshot_button = QtWidgets.QPushButton("Save snapshot", session_box)
            self.save_snapshot_button.setObjectName("saveSnapshotButton")
            self.save_snapshot_button.setMinimumWidth(0)
            self.save_snapshot_button.setMaximumWidth(140)
            self.save_snapshot_button.clicked.connect(lambda _checked=False: self.save_snapshot())
            snapshot_save_row.addWidget(self.save_snapshot_button)
            session_form.addRow("Snapshot note", snapshot_save_row)

            self.snapshot_combo = QtWidgets.QComboBox(session_box)
            self.snapshot_combo.setObjectName("snapshotCombo")
            self.snapshot_combo.setAccessibleName(self._tr("a11y.snapshot_selector"))
            self.restore_snapshot_button = QtWidgets.QPushButton("Restore snapshot", session_box)
            self.restore_snapshot_button.setObjectName("restoreSnapshotButton")
            self.restore_snapshot_button.setMinimumWidth(0)
            self.restore_snapshot_button.setMaximumWidth(140)
            self.restore_snapshot_button.clicked.connect(lambda _checked=False: self.restore_snapshot())
            self.snapshot_restore_row = QtWidgets.QHBoxLayout()
            snapshot_restore_row = self.snapshot_restore_row
            snapshot_restore_row.addWidget(self.snapshot_combo, 1)
            snapshot_restore_row.addWidget(self.restore_snapshot_button)
            session_form.addRow("Saved snapshots", snapshot_restore_row)
            layout.addWidget(self.fit_session_group)

            # The controls are intentionally tabbed.  q/ridge controls stay
            # above the fold in the first tab, geometry constraints have their
            # own compact vertical space, and review/ROI controls no longer
            # push scientific settings out of reach on a laptop display.  The
            # surrounding presentation layer still provides a vertical scroll
            # area for the long parameter table and individual tabs.
            self._build_control_tabs(panel, layout)
            self._sync_fit_session_controls()
            self._update_roi_controls()
            self.parameters_dock.setWidget(panel)
            self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self.parameters_dock)

        def _build_control_tabs(self, panel: Any, layout: Any) -> None:
            """Move long-lived control groups into compact, labelled tabs."""

            tabs = QtWidgets.QTabWidget(panel)
            tabs.setObjectName("parameterControlTabs")
            tabs.setTabBarAutoHide(False)
            tabs.setDocumentMode(True)
            tabs.setAccessibleName(self._tr("a11y.controls_tabs"))
            self.parameter_control_tabs = tabs

            def compact_form_labels(form: Any, maximum_width: int = 150) -> None:
                for row in range(form.rowCount()):
                    item = form.itemAt(row, QtWidgets.QFormLayout.LabelRole)
                    label = item.widget() if item is not None else None
                    if isinstance(label, QtWidgets.QLabel):
                        label.setWordWrap(True)
                        label.setMaximumWidth(maximum_width)
                        label.setSizePolicy(
                            QtWidgets.QSizePolicy.Policy.Preferred,
                            QtWidgets.QSizePolicy.Policy.Preferred,
                        )

            compact_form_labels(self.analysis_form)
            compact_form_labels(self.ellipse_form, maximum_width=72)
            compact_form_labels(self.fit_session_form)
            self.ellipse_residual_combo.setMinimumWidth(0)
            self.ellipse_residual_combo.setMaximumWidth(240)
            for button in (self.measure_geometry_button, self.refine_geometry_button):
                button.setMaximumWidth(300)

            def add_page(title: str, page_name: str, groups: tuple[Any, ...]) -> None:
                page = QtWidgets.QWidget(tabs)
                page.setObjectName(page_name)
                page_layout = QtWidgets.QVBoxLayout(page)
                page_layout.setContentsMargins(4, 4, 4, 4)
                for group in groups:
                    layout.removeWidget(group)
                    group.setParent(page)
                    page_layout.addWidget(group)
                page_layout.addStretch(1)
                tabs.addTab(page, title)

            # Add analysis before geometry so the q window and ridge method
            # are the first controls a user sees after loading a frame.  Keep
            # the flat-ellipse editor on its own page so every initial/bound/
            # fixed field is visible without an extra long scroll.
            add_page(
                "Analysis",
                "analysisControlsPage",
                (self.analysis_group,),
            )
            add_page("Geometry", "geometryControlsPage", (self.ellipse_group,))
            add_page("Mask / ROI", "roiControlsPage", (self.roi_group,))
            add_page("Review", "reviewControlsPage", (self.fit_session_group,))
            layout.addWidget(tabs, 0)

        def _build_measurements_page(self) -> None:
            """Build the measured-profile page without making pyqtgraph required."""

            self.measurements_page = QtWidgets.QWidget(self.pages)
            root = QtWidgets.QVBoxLayout(self.measurements_page)
            root.setContentsMargins(4, 4, 4, 4)
            self.measurement_observables: Any = None

            profile_tabs = QtWidgets.QTabWidget(self.measurements_page)
            profile_tabs.setObjectName("profileTabs")
            profile_tabs.setDocumentMode(True)
            profile_tabs.setAccessibleName(self._tr("a11y.profile_tabs"))
            self.profile_tabs = profile_tabs
            # Keep the old attribute name for lightweight integrations that
            # only need to locate the profile container.
            self.profile_splitter = profile_tabs
            self.angular_plot = None
            self.coverage_plot = None
            self.ridge_plot = None
            self.radial_profile_plot = None
            if _pg is not None:
                self.angular_plot = _pg.PlotWidget(self.measurements_page)
                self.angular_plot.setObjectName("angularProfilePlot")
                self.angular_plot.setAccessibleName(self._tr("a11y.profile_angular_name"))
                self.angular_plot.setAccessibleDescription(self._tr("a11y.profile_angular_description"))
                self.angular_plot.setLabel("bottom", "Azimuth (deg)")
                self.angular_plot.setLabel("left", "Angular intensity (a.u.)")
                self.angular_plot.showGrid(x=True, y=True, alpha=0.22)
                _disable_auto_si_prefix(self.angular_plot)
                profile_tabs.addTab(self.angular_plot, "Angular intensity")
                self.coverage_plot = _pg.PlotWidget(self.measurements_page)
                self.coverage_plot.setObjectName("coverageProfilePlot")
                self.coverage_plot.setAccessibleName(self._tr("a11y.profile_coverage_name"))
                self.coverage_plot.setAccessibleDescription(self._tr("a11y.profile_coverage_description"))
                self.coverage_plot.setLabel("bottom", "Azimuth (deg)")
                self.coverage_plot.setLabel("left", "Detector coverage (0–1)")
                self.coverage_plot.showGrid(x=True, y=True, alpha=0.22)
                _disable_auto_si_prefix(self.coverage_plot)
                profile_tabs.addTab(self.coverage_plot, "Coverage")
                self.ridge_plot = _pg.PlotWidget(self.measurements_page)
                self.ridge_plot.setObjectName("ridgeProfilePlot")
                self.ridge_plot.setAccessibleName(self._tr("a11y.profile_ridge_name"))
                self.ridge_plot.setAccessibleDescription(self._tr("a11y.profile_ridge_description"))
                self.ridge_plot.setLabel("bottom", "Azimuth (deg)")
                self.ridge_plot.setLabel("left", "Ridge q (map unit)")
                self.ridge_plot.showGrid(x=True, y=True, alpha=0.22)
                _disable_auto_si_prefix(self.ridge_plot)
                profile_tabs.addTab(self.ridge_plot, "Ridge q-angle")
                self.radial_profile_plot = _pg.PlotWidget(self.measurements_page)
                self.radial_profile_plot.setObjectName("radialProfilePlot")
                self.radial_profile_plot.setAccessibleName(self._tr("a11y.profile_radial_name"))
                self.radial_profile_plot.setAccessibleDescription(self._tr("a11y.profile_radial_description"))
                self.radial_profile_plot.setLabel("bottom", "q (map unit)")
                self.radial_profile_plot.setLabel("left", "Radial intensity (a.u.)")
                self.radial_profile_plot.showGrid(x=True, y=True, alpha=0.22)
                _disable_auto_si_prefix(self.radial_profile_plot)
                profile_tabs.addTab(self.radial_profile_plot, "Lobe radial profiles")
            else:
                self.angular_placeholder = QtWidgets.QLabel("安装 pyqtgraph 后显示角向强度与 coverage")
                self.angular_placeholder.setObjectName("angularProfilePlaceholder")
                self.angular_placeholder.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                profile_tabs.addTab(self.angular_placeholder, "Angular intensity")
                self.coverage_placeholder = QtWidgets.QLabel("安装 pyqtgraph 后显示 detector coverage")
                self.coverage_placeholder.setObjectName("coverageProfilePlaceholder")
                self.coverage_placeholder.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                profile_tabs.addTab(self.coverage_placeholder, "Coverage")
                self.ridge_placeholder = QtWidgets.QLabel("安装 pyqtgraph 后显示 ridge q-angle/accepted")
                self.ridge_placeholder.setObjectName("ridgeProfilePlaceholder")
                self.ridge_placeholder.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                profile_tabs.addTab(self.ridge_placeholder, "Ridge q-angle")
                self.radial_profile_placeholder = QtWidgets.QLabel("安装 pyqtgraph 后显示 lobe radial profiles")
                self.radial_profile_placeholder.setObjectName("radialProfilePlaceholder")
                self.radial_profile_placeholder.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                profile_tabs.addTab(self.radial_profile_placeholder, "Lobe radial profiles")
            root.addWidget(profile_tabs, 2)
            self.profile_summary_label = QtWidgets.QLabel(
                self._tr("profile.summary_empty"),
                self.measurements_page,
            )
            self.profile_summary_label.setObjectName("profileSummaryLabel")
            self.profile_summary_label.setWordWrap(True)
            self.profile_summary_label.setAccessibleName(self._tr("a11y.profile_tabs"))
            self.profile_summary_label.setAccessibleDescription(
                self._tr("profile.summary_empty")
            )
            self._set_profile_summary(self._tr("profile.summary_empty"))
            root.addWidget(self.profile_summary_label)

            lower = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal, self.measurements_page)
            lower.setObjectName("measurementTablesSplitter")
            lobe_panel = QtWidgets.QWidget(lower)
            lobe_layout = QtWidgets.QVBoxLayout(lobe_panel)
            self.lobe_panel_label = QtWidgets.QLabel("Four-lobe measurements", lobe_panel)
            lobe_layout.addWidget(self.lobe_panel_label)
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
            self.ridge_panel_label = QtWidgets.QLabel("Ridge q vs angle / accepted", ridge_panel)
            ridge_layout.addWidget(self.ridge_panel_label)
            self.ridge_table = QtWidgets.QTableWidget(0, 4, ridge_panel)
            self.ridge_table.setObjectName("ridgeTable")
            self.ridge_table.setHorizontalHeaderLabels(["Angle (deg)", "q", "Accepted", "Method"])
            self.ridge_table.horizontalHeader().setStretchLastSection(True)
            self.ridge_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
            ridge_layout.addWidget(self.ridge_table, 1)
            lower.addWidget(ridge_panel)

            ellipse_panel = QtWidgets.QWidget(lower)
            ellipse_layout = QtWidgets.QVBoxLayout(ellipse_panel)
            self.ellipse_panel_label = QtWidgets.QLabel(
                "Ellipse core quantities / quality",
                ellipse_panel,
            )
            ellipse_layout.addWidget(self.ellipse_panel_label)
            self.ellipse_table = QtWidgets.QTableWidget(0, 2, ellipse_panel)
            self.ellipse_table.setObjectName("ellipseTable")
            self.ellipse_table.setHorizontalHeaderLabels(["Quantity", "Value"])
            self.ellipse_table.horizontalHeader().setStretchLastSection(True)
            self.ellipse_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
            ellipse_layout.addWidget(self.ellipse_table, 1)
            lower.addWidget(ellipse_panel)

            radial_panel = QtWidgets.QWidget(lower)
            radial_layout = QtWidgets.QVBoxLayout(radial_panel)
            self.radial_panel_label = QtWidgets.QLabel(
                "Lobe radial q-star / spacing",
                radial_panel,
            )
            radial_layout.addWidget(self.radial_panel_label)
            self.radial_table = QtWidgets.QTableWidget(0, 8, radial_panel)
            self.radial_table.setObjectName("radialTable")
            self.radial_table.setHorizontalHeaderLabels(
                [
                    "Angle (deg)",
                    "q-star",
                    "Ln (nm)",
                    "SNR",
                    "FWHM (q)",
                    "Coverage",
                    "Valid",
                    "Flags",
                ]
            )
            self.radial_table.horizontalHeader().setStretchLastSection(True)
            self.radial_table.setEditTriggers(
                QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
            )
            radial_layout.addWidget(self.radial_table, 1)
            lower.addWidget(radial_panel)
            root.addWidget(lower, 2)
            self.pages.addTab(self.measurements_page, "Measurements / Profiles")

        def _build_analysis_controls(self, parent: Any, layout: Any) -> None:
            """Build explicit q/measurement controls beside the fit table."""

            self.analysis_group = QtWidgets.QGroupBox("Analysis / Measurement", parent)
            self.analysis_form = QtWidgets.QFormLayout(self.analysis_group)
            form = self.analysis_form
            self.q_min_edit = QtWidgets.QLineEdit("Auto", self.analysis_group)
            self.q_min_edit.setObjectName("qMinEdit")
            self.q_min_edit.setPlaceholderText("Auto")
            self.q_max_edit = QtWidgets.QLineEdit("Auto", self.analysis_group)
            self.q_max_edit.setObjectName("qMaxEdit")
            self.q_max_edit.setPlaceholderText("Auto")
            form.addRow("q min", self.q_min_edit)
            form.addRow("q max", self.q_max_edit)
            self.focus_q_window_check = QtWidgets.QCheckBox("Focus q window", self.analysis_group)
            self.focus_q_window_check.setObjectName("focusQWindowCheck")
            self.focus_q_window_check.setChecked(True)
            form.addRow("view", self.focus_q_window_check)
            q_view_buttons = QtWidgets.QVBoxLayout()
            q_view_buttons.setContentsMargins(0, 0, 0, 0)
            self.focus_q_window_button = QtWidgets.QPushButton("Fit q window", self.analysis_group)
            self.focus_q_window_button.setObjectName("focusQWindowButton")
            self.focus_q_window_button.clicked.connect(self.focus_q_window)
            q_view_buttons.addWidget(self.focus_q_window_button)
            self.reset_q_view_button = QtWidgets.QPushButton("Full detector", self.analysis_group)
            self.reset_q_view_button.setObjectName("resetQViewButton")
            self.reset_q_view_button.clicked.connect(self.reset_q_view)
            q_view_buttons.addWidget(self.reset_q_view_button)
            form.addRow(q_view_buttons)

            self.display_scale_combo = QtWidgets.QComboBox(self.analysis_group)
            self.display_scale_combo.setObjectName("displayScaleCombo")
            self.display_scale_combo.addItem("Linear", "linear")
            self.display_scale_combo.addItem("Log1p", "log1p")
            self.display_scale_combo.addItem("Asinh", "asinh")
            self.display_scale_combo.setAccessibleName(self._tr("label.display_scale"))
            form.addRow("display scale", self.display_scale_combo)
            self.display_percentile_spin = QtWidgets.QDoubleSpinBox(self.analysis_group)
            self.display_percentile_spin.setObjectName("displayPercentileSpin")
            self.display_percentile_spin.setRange(50.0, 100.0)
            self.display_percentile_spin.setDecimals(2)
            self.display_percentile_spin.setValue(self._display_percentile)
            self.display_percentile_spin.setSuffix(" %")
            self.display_percentile_spin.setAccessibleName(self._tr("label.display_percentile"))
            form.addRow("display upper percentile", self.display_percentile_spin)

            self.draw_axis_deg_spin = QtWidgets.QDoubleSpinBox(self.analysis_group)
            self.draw_axis_deg_spin.setObjectName("drawAxisDegSpin")
            self.draw_axis_deg_spin.setRange(-360.0, 360.0)
            self.draw_axis_deg_spin.setDecimals(3)
            form.addRow("draw axis (deg)", self.draw_axis_deg_spin)

            self.ridge_method_combo = QtWidgets.QComboBox(self.analysis_group)
            self.ridge_method_combo.setObjectName("ridgeMethodCombo")
            self.ridge_method_combo.addItem("Radial peak", "radial_peak")
            self.ridge_method_combo.addItem("Azimuthal peak", "azimuthal_peak")
            self.ridge_method_combo.addItem("Surface curvature", "surface_curvature")
            form.addRow("ridge method", self.ridge_method_combo)

            self.ridge_snr_threshold_spin = QtWidgets.QDoubleSpinBox(self.analysis_group)
            self.ridge_snr_threshold_spin.setObjectName("ridgeSnrThresholdSpin")
            self.ridge_snr_threshold_spin.setRange(0.0, 1e6)
            self.ridge_snr_threshold_spin.setDecimals(3)
            form.addRow("ridge SNR threshold", self.ridge_snr_threshold_spin)
            self.ridge_min_peak_fraction_spin = QtWidgets.QDoubleSpinBox(self.analysis_group)
            self.ridge_min_peak_fraction_spin.setObjectName("ridgeMinPeakFractionSpin")
            self.ridge_min_peak_fraction_spin.setRange(0.0, 1.0)
            self.ridge_min_peak_fraction_spin.setDecimals(4)
            form.addRow("ridge support fraction", self.ridge_min_peak_fraction_spin)
            self.ridge_min_coverage_spin = QtWidgets.QDoubleSpinBox(self.analysis_group)
            self.ridge_min_coverage_spin.setObjectName("ridgeMinCoverageSpin")
            self.ridge_min_coverage_spin.setRange(0.0, 1.0)
            self.ridge_min_coverage_spin.setDecimals(4)
            form.addRow("ridge coverage minimum", self.ridge_min_coverage_spin)

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
            self.full2d_multistart_spin = QtWidgets.QSpinBox(self.analysis_group)
            self.full2d_multistart_spin.setObjectName("full2dMultistartSpin")
            self.full2d_multistart_spin.setRange(1, 32)
            form.addRow("full2d starts", self.full2d_multistart_spin)

            self.ellipse_group = QtWidgets.QGroupBox("Measured ellipse constraints", parent)
            ellipse_form = QtWidgets.QFormLayout(self.ellipse_group)
            self.ellipse_form = ellipse_form
            self.ellipse_preset_combo = QtWidgets.QComboBox(self.ellipse_group)
            self.ellipse_preset_combo.setObjectName("ellipsePresetCombo")
            self.ellipse_preset_combo.addItem("Standard", "standard")
            self.ellipse_preset_combo.addItem("Flat ellipse", "flat_ellipse")
            ellipse_form.addRow("preset", self.ellipse_preset_combo)

            def optional_spin(object_name: str, maximum: float = 1e9) -> Any:
                spin = CompactDoubleSpinBox(self.ellipse_group)
                spin.setObjectName(object_name)
                spin.setRange(0.0, maximum)
                spin.setDecimals(6)
                spin.setSpecialValueText("Auto")
                return spin

            def bounded_spin(object_name: str, minimum: float, maximum: float) -> Any:
                spin = CompactDoubleSpinBox(self.ellipse_group)
                spin.setObjectName(object_name)
                spin.setRange(minimum, maximum)
                spin.setDecimals(6)
                return spin

            def compact_row(
                fields: tuple[tuple[str, Any], ...],
                *,
                object_name: str,
            ) -> tuple[Any, dict[str, Any]]:
                """Build a dense labelled row while keeping every field named."""

                container = QtWidgets.QWidget(self.ellipse_group)
                container.setObjectName(object_name)
                row = QtWidgets.QHBoxLayout(container)
                row.setContentsMargins(0, 0, 0, 0)
                row.setSpacing(4)
                labels: dict[str, Any] = {}
                for index, (name, widget) in enumerate(fields):
                    label = QtWidgets.QLabel(name, container)
                    label.setObjectName(f"{object_name}{index}Label")
                    labels[name] = label
                    row.addWidget(label)
                    row.addWidget(widget, 1)
                return container, labels

            # These are explicit measured-ellipse starting values.  They are
            # kept separate from the full2d parameter table so a user can
            # guide a flat butterfly fit without silently changing intensity
            # parameters.  The value is only sent to the service when the
            # flat preset is selected or the user has edited it.
            self.ellipse_a_init_spin = bounded_spin("ellipseAInitSpin", 1.0e-9, 1.0e9)
            self.ellipse_ratio_init_spin = bounded_spin("ellipseRatioInitSpin", 1.0e-6, 1.0)
            self.ellipse_angle_deg_spin = bounded_spin("ellipseAngleDegSpin", -90.0, 90.0)
            self.ellipse_a_init_spin.setValue(0.8)
            self.ellipse_ratio_init_spin.setValue(0.08)
            self.ellipse_a_init_spin.setAccessibleName(
                f"{self._tr('label.ellipse_a')} {self._tr('label.ellipse_initial')}"
            )
            self.ellipse_ratio_init_spin.setAccessibleName(
                f"{self._tr('label.ellipse_ratio')} {self._tr('label.ellipse_initial')}"
            )
            self.ellipse_angle_deg_spin.setAccessibleName(
                self._tr("label.ellipse_angle_init")
            )
            self._ellipse_initial_explicit = False

            initial_row, initial_labels = compact_row(
                (
                    ("a", self.ellipse_a_init_spin),
                    ("b/a", self.ellipse_ratio_init_spin),
                ),
                object_name="ellipseInitialRow",
            )
            self.ellipse_angle_init_spin = self.ellipse_angle_deg_spin
            self._ellipse_initial_labels = initial_labels
            self._ellipse_initial_row = initial_row
            ellipse_form.addRow("initial", initial_row)

            self.ellipse_ratio_min_spin = optional_spin("ellipseRatioMinSpin", 1.0)
            self.ellipse_ratio_max_spin = optional_spin("ellipseRatioMaxSpin", 1.0)
            self.ellipse_a_min_spin = optional_spin("ellipseAMinSpin")
            self.ellipse_a_max_spin = optional_spin("ellipseAMaxSpin")
            self.ellipse_b_min_spin = optional_spin("ellipseBMinSpin")
            self.ellipse_b_max_spin = optional_spin("ellipseBMaxSpin")
            self.ellipse_fixed_a_check = QtWidgets.QCheckBox("Fixed a", self.ellipse_group)
            self.ellipse_fixed_a_check.setObjectName("ellipseFixedACheck")
            self.ellipse_fixed_ratio_check = QtWidgets.QCheckBox("Fixed axis ratio", self.ellipse_group)
            self.ellipse_fixed_ratio_check.setObjectName("ellipseFixedRatioCheck")

            self.ellipse_angle_min_spin = CompactDoubleSpinBox(self.ellipse_group)
            self.ellipse_angle_min_spin.setObjectName("ellipseAngleMinSpin")
            self.ellipse_angle_min_spin.setRange(-90.0, 90.0)
            self.ellipse_angle_min_spin.setDecimals(3)
            self.ellipse_angle_max_spin = CompactDoubleSpinBox(self.ellipse_group)
            self.ellipse_angle_max_spin.setObjectName("ellipseAngleMaxSpin")
            self.ellipse_angle_max_spin.setRange(-90.0, 90.0)
            self.ellipse_angle_max_spin.setDecimals(3)

            self.ellipse_fixed_center_check = QtWidgets.QCheckBox("Fixed center", self.ellipse_group)
            self.ellipse_fixed_center_check.setObjectName("ellipseFixedCenterCheck")
            center_container = QtWidgets.QWidget(self.ellipse_group)
            center_container.setObjectName("ellipseCenterRow")
            center_row = QtWidgets.QHBoxLayout(center_container)
            center_row.setContentsMargins(0, 0, 0, 0)
            center_row.setSpacing(4)
            self.ellipse_center_qx_spin = CompactDoubleSpinBox(self.ellipse_group)
            self.ellipse_center_qx_spin.setObjectName("ellipseCenterQxSpin")
            self.ellipse_center_qx_spin.setRange(-1e9, 1e9)
            self.ellipse_center_qx_spin.setDecimals(6)
            self.ellipse_center_qy_spin = CompactDoubleSpinBox(self.ellipse_group)
            self.ellipse_center_qy_spin.setObjectName("ellipseCenterQySpin")
            self.ellipse_center_qy_spin.setRange(-1e9, 1e9)
            self.ellipse_center_qy_spin.setDecimals(6)
            self._ellipse_center_labels = {
                "qx": QtWidgets.QLabel("qx", center_container),
                "qy": QtWidgets.QLabel("qy", center_container),
            }
            center_row.addWidget(self._ellipse_center_labels["qx"])
            center_row.addWidget(self.ellipse_center_qx_spin)
            center_row.addWidget(self._ellipse_center_labels["qy"])
            center_row.addWidget(self.ellipse_center_qy_spin)

            angle_row = QtWidgets.QHBoxLayout()
            angle_container = QtWidgets.QWidget(self.ellipse_group)
            angle_row = QtWidgets.QHBoxLayout(angle_container)
            angle_row.setContentsMargins(0, 0, 0, 0)
            angle_row.setSpacing(4)
            self.ellipse_fixed_angle_check = QtWidgets.QCheckBox("Fixed angle", self.ellipse_group)
            self.ellipse_fixed_angle_check.setObjectName("ellipseFixedAngleCheck")
            angle_row.addWidget(self.ellipse_angle_deg_spin)
            self._ellipse_angle_unit_label = QtWidgets.QLabel("deg", angle_container)
            angle_row.addWidget(self._ellipse_angle_unit_label)

            self.ellipse_residual_combo = QtWidgets.QComboBox(self.ellipse_group)
            self.ellipse_residual_combo.setObjectName("ellipseResidualCombo")
            self.ellipse_residual_combo.addItem("Sampson", "sampson")
            self.ellipse_residual_combo.addItem("Geometric distance", "geometric")
            self.ellipse_multistart_spin = QtWidgets.QSpinBox(self.ellipse_group)
            self.ellipse_multistart_spin.setObjectName("ellipseMultistartSpin")
            self.ellipse_multistart_spin.setRange(1, 64)

            self._ellipse_a_bounds_row, self._ellipse_a_bounds_labels = compact_row(
                (("min", self.ellipse_a_min_spin), ("max", self.ellipse_a_max_spin)),
                object_name="ellipseABoundsRow",
            )
            self._ellipse_ratio_bounds_row, self._ellipse_ratio_bounds_labels = compact_row(
                (("min", self.ellipse_ratio_min_spin), ("max", self.ellipse_ratio_max_spin)),
                object_name="ellipseRatioBoundsRow",
            )
            self._ellipse_b_bounds_row, self._ellipse_b_bounds_labels = compact_row(
                (("min", self.ellipse_b_min_spin), ("max", self.ellipse_b_max_spin)),
                object_name="ellipseBBoundsRow",
            )
            self._ellipse_angle_bounds_row, self._ellipse_angle_bounds_labels = compact_row(
                (("min", self.ellipse_angle_min_spin), ("max", self.ellipse_angle_max_spin)),
                object_name="ellipseAngleBoundsRow",
            )
            self._ellipse_fixed_rows = {
                "a": self.ellipse_fixed_a_check,
                "axis_ratio": self.ellipse_fixed_ratio_check,
                "center": self.ellipse_fixed_center_check,
                "angle": self.ellipse_fixed_angle_check,
            }
            for field_key, spin in (
                ("label.ellipse_a_min", self.ellipse_a_min_spin),
                ("label.ellipse_a_max", self.ellipse_a_max_spin),
                ("label.ellipse_ratio_min", self.ellipse_ratio_min_spin),
                ("label.ellipse_ratio_max", self.ellipse_ratio_max_spin),
                ("label.ellipse_b_min", self.ellipse_b_min_spin),
                ("label.ellipse_b_max", self.ellipse_b_max_spin),
                ("label.ellipse_angle_min", self.ellipse_angle_min_spin),
                ("label.ellipse_angle_max", self.ellipse_angle_max_spin),
                ("label.ellipse_center_q", self.ellipse_center_qx_spin),
                ("label.ellipse_center_q", self.ellipse_center_qy_spin),
            ):
                suffix = (
                    " qx"
                    if spin is self.ellipse_center_qx_spin
                    else " qy"
                    if spin is self.ellipse_center_qy_spin
                    else ""
                )
                spin.setAccessibleName(f"{self._tr(field_key)}{suffix}")
            # Keep the paired bounds rows narrow enough for a 360–440 px dock;
            # fixed-state checkboxes occupy the following full-width row so a
            # translated label cannot push the maximum field off-screen.
            self._ellipse_a_control_row = self._ellipse_a_bounds_row
            self._ellipse_ratio_control_row = self._ellipse_ratio_bounds_row
            self._ellipse_angle_control_row = self._ellipse_angle_bounds_row
            ellipse_form.addRow("a", self._ellipse_a_bounds_row)
            ellipse_form.addRow("", self.ellipse_fixed_a_check)
            ellipse_form.addRow("axis ratio", self._ellipse_ratio_bounds_row)
            ellipse_form.addRow("", self.ellipse_fixed_ratio_check)
            ellipse_form.addRow("b (derived)", self._ellipse_b_bounds_row)
            ellipse_form.addRow("angle", self._ellipse_angle_bounds_row)
            ellipse_form.addRow("", self.ellipse_fixed_angle_check)
            self._ellipse_angle_initial_row = angle_container
            ellipse_form.addRow("angle initial", angle_container)
            ellipse_form.addRow("center", self.ellipse_fixed_center_check)
            ellipse_form.addRow("center q", center_container)
            ellipse_form.addRow("residual", self.ellipse_residual_combo)
            ellipse_form.addRow("starts", self.ellipse_multistart_spin)

            geometry_buttons = QtWidgets.QVBoxLayout()
            self.measure_geometry_button = QtWidgets.QPushButton("Remeasure geometry", self.ellipse_group)
            self.measure_geometry_button.setObjectName("measureGeometryButton")
            self.measure_geometry_button.clicked.connect(self.request_geometry_measure)
            geometry_buttons.addWidget(self.measure_geometry_button)
            self.refine_geometry_button = QtWidgets.QPushButton("Refine geometry", self.ellipse_group)
            self.refine_geometry_button.setObjectName("refineGeometryButton")
            self.refine_geometry_button.clicked.connect(self.request_geometry_refine)
            geometry_buttons.addWidget(self.refine_geometry_button)
            ellipse_form.addRow(geometry_buttons)
            layout.addWidget(self.ellipse_group)

            self.q_min_edit.editingFinished.connect(self._on_analysis_changed)
            self.q_max_edit.editingFinished.connect(self._on_analysis_changed)
            self.draw_axis_deg_spin.valueChanged.connect(self._on_analysis_changed)
            self.ridge_method_combo.currentIndexChanged.connect(self._on_analysis_changed)
            self.focus_q_window_check.toggled.connect(self._on_q_view_setting_changed)
            self.display_scale_combo.currentIndexChanged.connect(
                self._on_display_setting_changed
            )
            self.display_percentile_spin.valueChanged.connect(
                self._on_display_setting_changed
            )
            for spin in (
                self.n_angular_bins_spin,
                self.n_ridge_angles_spin,
                self.n_radial_bins_spin,
                self.curvature_sigma_spin,
                self.curvature_percentile_spin,
                self.normal_step_spin,
                self.max_pixels_spin,
                self.full2d_multistart_spin,
                self.ellipse_ratio_min_spin,
                self.ellipse_ratio_max_spin,
                self.ellipse_a_min_spin,
                self.ellipse_a_max_spin,
                self.ellipse_b_min_spin,
                self.ellipse_b_max_spin,
                self.ellipse_angle_min_spin,
                self.ellipse_angle_max_spin,
                self.ellipse_center_qx_spin,
                self.ellipse_center_qy_spin,
                self.ellipse_angle_deg_spin,
                self.ellipse_multistart_spin,
            ):
                spin.valueChanged.connect(self._on_analysis_changed)
            self.ellipse_a_init_spin.valueChanged.connect(
                self._mark_ellipse_initial_explicit
            )
            self.ellipse_ratio_init_spin.valueChanged.connect(
                self._mark_ellipse_initial_explicit
            )
            self.ellipse_angle_deg_spin.valueChanged.connect(
                self._mark_ellipse_initial_explicit
            )
            for widget in (
                self.ellipse_preset_combo,
                self.ellipse_fixed_center_check,
                self.ellipse_fixed_angle_check,
                self.ellipse_fixed_a_check,
                self.ellipse_fixed_ratio_check,
                self.ellipse_residual_combo,
            ):
                widget.currentIndexChanged.connect(self._on_analysis_changed) if isinstance(widget, QtWidgets.QComboBox) else widget.toggled.connect(self._on_analysis_changed)
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
            self.batch_form = QtWidgets.QFormLayout()
            options = self.batch_form
            self.batch_mode_combo = QtWidgets.QComboBox(self.batch_page)
            self.batch_mode_combo.setObjectName("batchModeCombo")
            self.batch_mode_combo.addItem("Independent", "independent")
            self.batch_mode_combo.addItem("Warm start", "warm_start")
            options.addRow("Mode", self.batch_mode_combo)
            self.batch_stage_combo = QtWidgets.QComboBox(self.batch_page)
            self.batch_stage_combo.setObjectName("batchStageCombo")
            self.batch_stage_combo.addItem("Geometry measurement", "geometry")
            self.batch_stage_combo.addItem("Full2D refinement", "full2d")
            self.batch_stage_combo.setCurrentIndex(1)
            self.batch_stage_combo.setAccessibleName(self._tr("a11y.batch_stage"))
            options.addRow("Stage", self.batch_stage_combo)
            self.batch_stream_check = QtWidgets.QCheckBox(
                "Stream results (bounded memory)", self.batch_page
            )
            self.batch_stream_check.setObjectName("batchStreamCheck")
            self.batch_stream_check.setAccessibleName(self._tr("a11y.batch_stream"))
            options.addRow("Retention", self.batch_stream_check)
            self.batch_stream_help = QtWidgets.QLabel(
                "Streaming requires an output directory and writes per-frame artifacts; "
                "leave it off for an in-memory exploratory run.",
                self.batch_page,
            )
            self.batch_stream_help.setObjectName("batchStreamHelp")
            self.batch_stream_help.setWordWrap(True)
            self.batch_stream_help.setAccessibleName(self._tr("a11y.batch_help"))
            options.addRow("", self.batch_stream_help)
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
            self.batch_series_edit = QtWidgets.QLineEdit(self.batch_page)
            self.batch_series_edit.setObjectName("batchSeriesEdit")
            self.batch_series_edit.setPlaceholderText("optional series/group")
            options.addRow("Series", self.batch_series_edit)
            self.batch_start_spin = QtWidgets.QSpinBox(self.batch_page)
            self.batch_start_spin.setObjectName("batchStartSpin")
            self.batch_start_spin.setRange(-1, 1_000_000_000)
            self.batch_start_spin.setSpecialValueText("Auto")
            self.batch_start_spin.setValue(-1)
            options.addRow("Start", self.batch_start_spin)
            self.batch_stop_spin = QtWidgets.QSpinBox(self.batch_page)
            self.batch_stop_spin.setObjectName("batchStopSpin")
            self.batch_stop_spin.setRange(-1, 1_000_000_000)
            self.batch_stop_spin.setSpecialValueText("Auto")
            self.batch_stop_spin.setValue(-1)
            options.addRow("Stop", self.batch_stop_spin)
            self.batch_stride_spin = QtWidgets.QSpinBox(self.batch_page)
            self.batch_stride_spin.setObjectName("batchStrideSpin")
            self.batch_stride_spin.setRange(1, 1_000_000_000)
            self.batch_stride_spin.setValue(1)
            options.addRow("Stride", self.batch_stride_spin)
            self.batch_range_edit = QtWidgets.QLineEdit(self.batch_page)
            self.batch_range_edit.setObjectName("batchRangeEdit")
            self.batch_range_edit.setPlaceholderText("START:STOP[:STEP]")
            options.addRow("Range", self.batch_range_edit)
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
            self.batch_progress.setRange(0, 1)
            self.batch_progress.setVisible(False)
            self.batch_progress.setAccessibleName(self._tr("a11y.batch_progress"))
            self.batch_progress.setAccessibleDescription(
                self._tr("a11y.batch_progress_description")
            )
            layout.addWidget(self.batch_progress)
            self.batch_progress_label = QtWidgets.QLabel("", self.batch_page)
            self.batch_progress_label.setObjectName("batchProgressLabel")
            self.batch_progress_label.setAccessibleName(self._tr("a11y.batch_progress"))
            self.batch_progress_label.setVisible(False)
            layout.addWidget(self.batch_progress_label)
            self.pages.addTab(self.batch_page, "Batch")

        def _build_evolution_page(self) -> None:
            self.evolution_page = QtWidgets.QWidget(self.pages)
            layout = QtWidgets.QVBoxLayout(self.evolution_page)
            selector_row = QtWidgets.QHBoxLayout()
            self.evolution_y_label = QtWidgets.QLabel("Y parameter", self.evolution_page)
            selector_row.addWidget(self.evolution_y_label)
            self.evolution_parameter_combo = QtWidgets.QComboBox(self.evolution_page)
            self.evolution_parameter_combo.setObjectName("evolutionParameterCombo")
            self.evolution_parameter_combo.setAccessibleName(self._tr("a11y.evolution_parameter"))
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

        # ----- runtime language -----------------------------------------------

        @property
        def language(self) -> str:
            """Current UI language code (``zh_CN`` or ``en``)."""

            return self._language

        def _tr(self, key: str, **values: Any) -> str:
            return translate(self._language, key, **values)

        def _on_language_action(self, action: Any) -> None:
            self.set_language(str(action.data()))

        def set_language(self, language: str, persist: bool = True) -> None:
            """Switch visible text without touching scientific or project state."""

            resolved = validate_language(language)
            changed = resolved != self._language
            self._language = resolved
            if changed:
                self._retranslate_ui()
            else:
                self.chinese_action.setChecked(resolved == "zh_CN")
                self.english_action.setChecked(resolved == "en")
            if persist:
                self._settings.setValue(LANGUAGE_SETTING_KEY, resolved)
                self._settings.sync()

        def _set_form_label(self, form: Any, field: Any, key: str) -> None:
            label = form.labelForField(field)
            if label is not None:
                label.setText(self._tr(key))

        def _set_combo_text(self, combo: Any, data: str, key: str) -> None:
            index = combo.findData(data)
            if index >= 0:
                combo.setItemText(index, self._tr(key))

        def _set_form_tooltip(self, form: Any, field: Any, key: str) -> None:
            """Apply one translated tooltip to a form field and its label."""

            tooltip = self._tr(key)
            setter = getattr(field, "setToolTip", None)
            if callable(setter):
                setter(tooltip)
            label = form.labelForField(field)
            if label is not None:
                label.setToolTip(tooltip)

        def _set_combo_item_tooltip(
            self,
            combo: Any,
            data: Any,
            key: str,
            **values: Any,
        ) -> None:
            """Set the popup tooltip for one stable-data combo option."""

            index = combo.findData(data)
            if index >= 0:
                combo.setItemData(
                    index,
                    self._tr(key, **values),
                    QtCore.Qt.ItemDataRole.ToolTipRole,
                )

        def _refresh_snapshot_item_tooltips(self) -> None:
            if not hasattr(self, "snapshot_combo"):
                return
            snapshots = self._fit_session.get("snapshots", [])
            if not isinstance(snapshots, list):
                return
            for combo_index in range(self.snapshot_combo.count()):
                snapshot_index = self.snapshot_combo.itemData(combo_index)
                if not isinstance(snapshot_index, int) or not 0 <= snapshot_index < len(snapshots):
                    continue
                snapshot = snapshots[snapshot_index]
                if not isinstance(snapshot, Mapping):
                    continue
                note = str(snapshot.get("note", "") or "") or self._tr("snapshot.no_note")
                self.snapshot_combo.setItemData(
                    combo_index,
                    self._tr(
                        "tooltip.combo.snapshot",
                        index=snapshot_index + 1,
                        note=note,
                    ),
                    QtCore.Qt.ItemDataRole.ToolTipRole,
                )

        def _refresh_evolution_item_tooltips(self) -> None:
            if not hasattr(self, "evolution_parameter_combo"):
                return
            for index in range(self.evolution_parameter_combo.count()):
                name = self.evolution_parameter_combo.itemText(index)
                self.evolution_parameter_combo.setItemData(
                    index,
                    self._tr("tooltip.combo.evolution", name=name),
                    QtCore.Qt.ItemDataRole.ToolTipRole,
                )

        def _apply_tooltips(self) -> None:
            """Refresh all application-owned hover help in the active language."""

            for action, key in (
                (self.project_menu.menuAction(), "tooltip.menu.project"),
                (self.language_menu.menuAction(), "tooltip.menu.language"),
                (self.open_project_action, "tooltip.open_project"),
                (self.save_project_action, "tooltip.save_project"),
                (self.open_image_action, "tooltip.open_image"),
                (self.open_poni_action, "tooltip.select_poni"),
                (self.open_mask_action, "tooltip.select_mask"),
                (self.clear_mask_action, "tooltip.clear_mask"),
                (self.export_evidence_action, "tooltip.export_evidence"),
                (self.close_action, "tooltip.close"),
                (self.chinese_action, "tooltip.language.zh_CN"),
                (self.english_action, "tooltip.language.en"),
            ):
                action.setToolTip(self._tr(key))

            for page, key in (
                (self.refinement_page, "tooltip.tab.refinement"),
                (self.measurements_page, "tooltip.tab.measurements"),
                (self.batch_page, "tooltip.tab.batch"),
                (self.evolution_page, "tooltip.tab.evolution"),
            ):
                self.pages.setTabToolTip(self.pages.indexOf(page), self._tr(key))

            if hasattr(self, "parameter_control_tabs"):
                control_tabs = self.parameter_control_tabs
                for index, (title_key, tooltip_key) in enumerate(
                    (
                        ("tab.controls.analysis", "tooltip.tab.controls.analysis"),
                        ("tab.controls.geometry", "tooltip.tab.controls.geometry"),
                        ("tab.controls.roi", "tooltip.tab.controls.roi"),
                        ("tab.controls.review", "tooltip.tab.controls.review"),
                    )
                ):
                    if index < control_tabs.count():
                        control_tabs.setTabText(index, self._tr(title_key))
                        control_tabs.setTabToolTip(index, self._tr(tooltip_key))

            for widget, key in (
                (self.parameter_table, "tooltip.parameter_table"),
                (self.preview_button, "tooltip.preview"),
                (self.optimize_button, "tooltip.optimize"),
                (self.cancel_button, "tooltip.cancel"),
                (self.ignore_late_result_button, "tooltip.ignore_late"),
                (self.auto_preview_check, "tooltip.auto_preview"),
                (self.clear_mask_button, "tooltip.clear_mask"),
                (self.roi_type_combo, "tooltip.roi_type"),
                (self.apply_roi_button, "tooltip.apply_roi"),
                (self.clear_roi_button, "tooltip.clear_roi"),
                (self.reviewer_edit, "tooltip.reviewer"),
                (self.review_notes_edit, "tooltip.review_notes"),
                (self.accept_current_button, "tooltip.accept_current"),
                (self.reject_current_button, "tooltip.reject_current"),
                (self.restore_before_optimize_button, "tooltip.restore_before_optimize"),
                (self.snapshot_note_edit, "tooltip.snapshot_note"),
                (self.save_snapshot_button, "tooltip.save_snapshot"),
                (self.snapshot_combo, "tooltip.snapshot_selector"),
                (self.restore_snapshot_button, "tooltip.restore_snapshot"),
                (self.batch_add_button, "tooltip.batch_add"),
                (self.batch_run_button, "tooltip.batch_run"),
                (self.batch_resume_check, "tooltip.resume_checkpoint"),
                (self.measure_geometry_button, "tooltip.measure_geometry"),
                (self.refine_geometry_button, "tooltip.refine_geometry"),
                (self.focus_q_window_button, "tooltip.q_view_focus"),
                (self.reset_q_view_button, "tooltip.q_view_reset"),
                (self.display_scale_combo, "tooltip.display_scale"),
                (self.display_percentile_spin, "tooltip.display_percentile"),
                (self.evolution_y_label, "tooltip.evolution_parameter"),
                (self.evolution_parameter_combo, "tooltip.evolution_parameter"),
            ):
                widget.setToolTip(self._tr(key))

            for form, field, key in (
                (self.fit_session_form, self.reviewer_edit, "tooltip.reviewer"),
                (self.fit_session_form, self.review_notes_edit, "tooltip.review_notes"),
                (self.fit_session_form, self.snapshot_save_row, "tooltip.snapshot_note"),
                (self.fit_session_form, self.snapshot_restore_row, "tooltip.snapshot_selector"),
                (self.analysis_form, self.q_min_edit, "tooltip.q_min"),
                (self.analysis_form, self.q_max_edit, "tooltip.q_max"),
                (self.analysis_form, self.focus_q_window_check, "tooltip.q_view_focus"),
                (self.analysis_form, self.display_scale_combo, "tooltip.display_scale"),
                (self.analysis_form, self.display_percentile_spin, "tooltip.display_percentile"),
                (self.analysis_form, self.draw_axis_deg_spin, "tooltip.draw_axis"),
                (self.analysis_form, self.ridge_method_combo, "tooltip.ridge_method"),
                (self.analysis_form, self.ridge_snr_threshold_spin, "tooltip.ridge_snr_threshold"),
                (self.analysis_form, self.ridge_min_peak_fraction_spin, "tooltip.ridge_min_peak_fraction"),
                (self.analysis_form, self.ridge_min_coverage_spin, "tooltip.ridge_min_coverage"),
                (self.analysis_form, self.n_angular_bins_spin, "tooltip.angular_bins"),
                (self.analysis_form, self.n_ridge_angles_spin, "tooltip.ridge_angles"),
                (self.analysis_form, self.n_radial_bins_spin, "tooltip.radial_bins"),
                (self.analysis_form, self.curvature_sigma_spin, "tooltip.curvature_sigma"),
                (
                    self.analysis_form,
                    self.curvature_percentile_spin,
                    "tooltip.curvature_percentile",
                ),
                (self.analysis_form, self.normal_step_spin, "tooltip.normal_step"),
                (self.analysis_form, self.max_pixels_spin, "tooltip.max_pixels"),
                (self.analysis_form, self.full2d_multistart_spin, "tooltip.full2d_multistart"),
                (self.batch_form, self.batch_mode_combo, "tooltip.batch_mode"),
                (self.batch_form, self.batch_stage_combo, "tooltip.batch_stage"),
                (self.batch_form, self.batch_stream_check, "tooltip.batch_stream"),
                (self.batch_form, self.batch_manifest_edit, "tooltip.manifest"),
                (self.batch_form, self.batch_checkpoint_edit, "tooltip.checkpoint"),
                (self.batch_form, self.batch_output_edit, "tooltip.output"),
                (self.batch_form, self.batch_series_edit, "tooltip.batch_series"),
                (self.batch_form, self.batch_start_spin, "tooltip.batch_start"),
                (self.batch_form, self.batch_stop_spin, "tooltip.batch_stop"),
                (self.batch_form, self.batch_stride_spin, "tooltip.batch_stride"),
                (self.batch_form, self.batch_range_edit, "tooltip.batch_range"),
            ):
                self._set_form_tooltip(form, field, key)

            self.ellipse_group.setToolTip(self._tr("tooltip.ellipse_constraints"))
            for widget, key in (
                (self.ellipse_preset_combo, "tooltip.ellipse_preset"),
                (self.ellipse_a_init_spin, "tooltip.ellipse_a_init"),
                (self.ellipse_ratio_init_spin, "tooltip.ellipse_ratio_init"),
                (self.ellipse_ratio_min_spin, "tooltip.ellipse_ratio_min"),
                (self.ellipse_ratio_max_spin, "tooltip.ellipse_ratio_max"),
                (self.ellipse_a_min_spin, "tooltip.ellipse_a_min"),
                (self.ellipse_a_max_spin, "tooltip.ellipse_a_max"),
                (self.ellipse_fixed_a_check, "tooltip.ellipse_fixed_a"),
                (self.ellipse_b_min_spin, "tooltip.ellipse_b_min"),
                (self.ellipse_b_max_spin, "tooltip.ellipse_b_max"),
                (self.ellipse_fixed_ratio_check, "tooltip.ellipse_fixed_ratio"),
                (self.ellipse_angle_min_spin, "tooltip.ellipse_angle"),
                (self.ellipse_angle_max_spin, "tooltip.ellipse_angle"),
                (self.ellipse_fixed_center_check, "tooltip.ellipse_center"),
                (self.ellipse_center_qx_spin, "tooltip.ellipse_center"),
                (self.ellipse_center_qy_spin, "tooltip.ellipse_center"),
                (self.ellipse_fixed_angle_check, "tooltip.ellipse_angle"),
                (self.ellipse_angle_deg_spin, "tooltip.ellipse_angle"),
                (self.ellipse_residual_combo, "tooltip.ellipse_residual"),
                (self.ellipse_multistart_spin, "tooltip.ellipse_multistart"),
            ):
                widget.setToolTip(self._tr(key))

            self.roi_type_label.setToolTip(self._tr("tooltip.roi_type"))
            rectangle_keys = (
                "tooltip.roi_x0",
                "tooltip.roi_y0",
                "tooltip.roi_x1",
                "tooltip.roi_y1",
            )
            for (label, spin), key in zip(self._rectangle_roi_widgets, rectangle_keys):
                tooltip = self._tr(key)
                label.setToolTip(tooltip)
                spin.setToolTip(tooltip)
            ellipse_keys = (
                "tooltip.roi_cx",
                "tooltip.roi_cy",
                "tooltip.roi_rx",
                "tooltip.roi_ry",
                "tooltip.roi_angle",
            )
            for (label, spin), key in zip(self._ellipse_roi_widgets, ellipse_keys):
                tooltip = self._tr(key)
                label.setToolTip(tooltip)
                spin.setToolTip(tooltip)

            for combo, data, key in (
                (self.roi_type_combo, "rectangle", "tooltip.combo.rectangle"),
                (self.roi_type_combo, "ellipse", "tooltip.combo.ellipse"),
                (self.ridge_method_combo, "radial_peak", "tooltip.combo.radial_peak"),
                (
                    self.ridge_method_combo,
                    "surface_curvature",
                    "tooltip.combo.surface_curvature",
                ),
                (self.ridge_method_combo, "azimuthal_peak", "tooltip.combo.azimuthal_peak"),
                (self.display_scale_combo, "linear", "tooltip.combo.display_linear"),
                (self.display_scale_combo, "log1p", "tooltip.combo.display_log1p"),
                (self.display_scale_combo, "asinh", "tooltip.combo.display_asinh"),
                (self.batch_mode_combo, "independent", "tooltip.combo.independent"),
                (self.batch_mode_combo, "warm_start", "tooltip.combo.warm_start"),
                (self.batch_stage_combo, "geometry", "tooltip.combo.batch_geometry"),
                (self.batch_stage_combo, "full2d", "tooltip.combo.batch_full2d"),
            ):
                self._set_combo_item_tooltip(combo, data, key)
            self._refresh_snapshot_item_tooltips()
            self._refresh_evolution_item_tooltips()

        def _render_metric_labels(self) -> None:
            self.rmse_label.setText(f"RMSE: {self._metric_display['rmse']}")
            self.ndata_label.setText(
                f"{self._tr('metric.ndata')}: {self._metric_display['ndata']}"
            )
            self.flags_label.setText(
                f"{self._tr('metric.flags')}: {self._displayed_flags_text}"
            )
            self.coverage_label.setText(
                f"{self._tr('metric.coverage')}: {self._metric_display['coverage']}"
            )

        def _render_status(self) -> None:
            values = dict(self._status_values)
            kind_key = values.pop("kind_key", None)
            if kind_key is not None:
                values["kind"] = self._tr(str(kind_key))
            self.status_message.setText(self._tr(self._status_key, **values))

        def _boolean_table_item(self, value: Any) -> QtWidgets.QTableWidgetItem:
            """Create a localized boolean item while retaining its raw value."""

            raw_value = bool(value)
            key = "boolean.true" if raw_value else "boolean.false"
            item = QtWidgets.QTableWidgetItem(self._tr(key))
            item.setData(QtCore.Qt.ItemDataRole.UserRole, raw_value)
            return item

        def _retranslate_measurement_booleans(self) -> None:
            for table in (
                self.ridge_table,
                self.lobe_table,
                self.ellipse_table,
                self.radial_table,
            ):
                for row in range(table.rowCount()):
                    for column in range(table.columnCount()):
                        item = table.item(row, column)
                        if item is None:
                            continue
                        raw_value = item.data(QtCore.Qt.ItemDataRole.UserRole)
                        if isinstance(raw_value, bool):
                            key = "boolean.true" if raw_value else "boolean.false"
                            item.setText(self._tr(key))

        def _set_profile_summary(self, text: str) -> None:
            """Keep the keyboard-readable profile summary text and description identical."""

            if not hasattr(self, "profile_summary_label"):
                return
            summary = str(text or self._tr("profile.summary_empty"))
            self.profile_summary_label.setText(summary)
            self.profile_summary_label.setAccessibleDescription(summary)

        def _retranslate_accessible_names(self) -> None:
            """Keep the accessibility tree in the same language as labels."""

            roi_fields = {
                "roiX0": "x0",
                "roiY0": "y0",
                "roiX1": "x1",
                "roiY1": "y1",
                "roiCx": "cx",
                "roiCy": "cy",
                "roiRx": "rx",
                "roiRy": "ry",
                "roiAngle": "angle",
            }
            for label, spin in getattr(self, "_rectangle_roi_widgets", ()):
                del label
                spin.setAccessibleName(
                    self._tr(
                        "a11y.roi_rectangle",
                        field=roi_fields.get(spin.objectName(), spin.objectName()),
                    )
                )
            for label, spin in getattr(self, "_ellipse_roi_widgets", ()):
                del label
                spin.setAccessibleName(
                    self._tr(
                        "a11y.roi_ellipse",
                        field=roi_fields.get(spin.objectName(), spin.objectName()),
                    )
                )
            for field_key, spin in (
                ("label.ellipse_a_min", self.ellipse_a_min_spin),
                ("label.ellipse_a_max", self.ellipse_a_max_spin),
                ("label.ellipse_ratio_min", self.ellipse_ratio_min_spin),
                ("label.ellipse_ratio_max", self.ellipse_ratio_max_spin),
                ("label.ellipse_b_min", self.ellipse_b_min_spin),
                ("label.ellipse_b_max", self.ellipse_b_max_spin),
                ("label.ellipse_angle_min", self.ellipse_angle_min_spin),
                ("label.ellipse_angle_max", self.ellipse_angle_max_spin),
            ):
                spin.setAccessibleName(self._tr(field_key))
            self.ellipse_center_qx_spin.setAccessibleName(
                f"{self._tr('label.ellipse_center_q')} qx"
            )
            self.ellipse_center_qy_spin.setAccessibleName(
                f"{self._tr('label.ellipse_center_q')} qy"
            )
            self.ellipse_a_init_spin.setAccessibleName(
                f"{self._tr('label.ellipse_a')} {self._tr('label.ellipse_initial')}"
            )
            self.ellipse_ratio_init_spin.setAccessibleName(
                f"{self._tr('label.ellipse_ratio')} {self._tr('label.ellipse_initial')}"
            )
            self.ellipse_angle_deg_spin.setAccessibleName(
                self._tr("label.ellipse_angle_init")
            )
            self.reviewer_edit.setAccessibleName(self._tr("label.reviewer"))
            self.review_notes_edit.setAccessibleName(self._tr("label.review_notes"))
            self.snapshot_note_edit.setAccessibleName(self._tr("a11y.snapshot_note"))
            self.snapshot_combo.setAccessibleName(self._tr("a11y.snapshot_selector"))
            self.evolution_parameter_combo.setAccessibleName(
                self._tr("a11y.evolution_parameter")
            )
            self.display_scale_combo.setAccessibleName(self._tr("label.display_scale"))
            self.display_percentile_spin.setAccessibleName(
                self._tr("label.display_percentile")
            )
            self.batch_stage_combo.setAccessibleName(self._tr("a11y.batch_stage"))
            self.batch_stream_check.setAccessibleName(self._tr("a11y.batch_stream"))
            self.batch_stream_help.setAccessibleName(self._tr("a11y.batch_help"))
            self.batch_progress.setAccessibleName(self._tr("a11y.batch_progress"))
            self.batch_progress.setAccessibleDescription(
                self._tr("a11y.batch_progress_description")
            )
            self.batch_progress_label.setAccessibleName(self._tr("a11y.batch_progress"))
            if hasattr(self, "parameter_control_tabs"):
                self.parameter_control_tabs.setAccessibleName(
                    self._tr("a11y.controls_tabs")
                )
            if hasattr(self, "profile_tabs"):
                self.profile_tabs.setAccessibleName(self._tr("a11y.profile_tabs"))
            if hasattr(self, "profile_summary_label"):
                self.profile_summary_label.setAccessibleName(
                    self._tr("a11y.profile_tabs")
                )
                self.profile_summary_label.setAccessibleDescription(
                    self.profile_summary_label.text()
                    or self._tr("profile.summary_empty")
                )
            if self.angular_plot is not None:
                self.angular_plot.setAccessibleName(self._tr("a11y.profile_angular_name"))
                self.angular_plot.setAccessibleDescription(
                    self._tr("a11y.profile_angular_description")
                )
                self.coverage_plot.setAccessibleName(self._tr("a11y.profile_coverage_name"))
                self.coverage_plot.setAccessibleDescription(
                    self._tr("a11y.profile_coverage_description")
                )
                self.ridge_plot.setAccessibleName(self._tr("a11y.profile_ridge_name"))
                self.ridge_plot.setAccessibleDescription(
                    self._tr("a11y.profile_ridge_description")
                )
                self.radial_profile_plot.setAccessibleName(
                    self._tr("a11y.profile_radial_name")
                )
                self.radial_profile_plot.setAccessibleDescription(
                    self._tr("a11y.profile_radial_description")
                )
            if hasattr(self, "workflow_status_label"):
                self.workflow_status_label.setAccessibleName(
                    self._tr("a11y.workflow_status")
                )
            if hasattr(self, "parameters_scroll_area"):
                self.parameters_scroll_area.setAccessibleName(
                    self._tr("a11y.scroll_controls")
                )

        def _retranslate_ui(self) -> None:
            """Refresh every user-facing label while preserving widget data."""

            self.setWindowTitle(self._tr("app.title"))
            self.project_menu.setTitle(self._tr("menu.project"))
            self.language_menu.setTitle(self._tr("menu.language"))
            self.chinese_action.setText(self._tr("language.zh_CN"))
            self.english_action.setText(self._tr("language.en"))
            self.chinese_action.setChecked(self._language == "zh_CN")
            self.english_action.setChecked(self._language == "en")
            for action, key in (
                (self.open_project_action, "action.open_project"),
                (self.save_project_action, "action.save_project"),
                (self.open_image_action, "action.open_image"),
                (self.open_poni_action, "action.select_poni"),
                (self.open_mask_action, "action.select_mask"),
                (self.clear_mask_action, "action.clear_mask"),
                (self.export_evidence_action, "action.export_evidence"),
                (self.close_action, "action.close"),
            ):
                action.setText(self._tr(key))
            self.export_evidence_action.setToolTip(self._tr("tooltip.export_evidence"))
            self.file_toolbar.setWindowTitle(self._tr("toolbar.project"))

            for page, key in (
                (self.refinement_page, "tab.refinement"),
                (self.measurements_page, "tab.measurements"),
                (self.batch_page, "tab.batch"),
                (self.evolution_page, "tab.evolution"),
            ):
                self.pages.setTabText(self.pages.indexOf(page), self._tr(key))
            self.parameters_dock.setWindowTitle(self._tr("dock.parameters"))
            self.parameter_table_title.setText(self._tr("label.parameter_table_title"))
            self.parameter_table_title.setAccessibleName(
                self._tr("label.parameter_table_title")
            )
            self.parameter_table_title.setAccessibleDescription(
                self._tr("tooltip.parameter_table")
            )

            for widget, key in (
                (self.preview_button, "button.preview"),
                (self.optimize_button, "button.optimize"),
                (self.cancel_button, "button.cancel"),
                (self.ignore_late_result_button, "button.ignore_late"),
                (self.clear_mask_button, "button.clear_mask"),
                (self.apply_roi_button, "button.apply"),
                (self.clear_roi_button, "button.clear"),
                (self.accept_current_button, "button.accept_current"),
                (self.reject_current_button, "button.reject_current"),
                (self.restore_before_optimize_button, "button.restore_before_optimize"),
                (self.save_snapshot_button, "button.save_snapshot"),
                (self.restore_snapshot_button, "button.restore_snapshot"),
                (self.batch_add_button, "button.add_frames"),
                (self.batch_run_button, "button.run_batch"),
                (self.measure_geometry_button, "button.remeasure_geometry"),
                (self.refine_geometry_button, "button.refine_geometry"),
                (self.focus_q_window_button, "button.q_view_focus"),
                (self.reset_q_view_button, "button.q_view_reset"),
            ):
                widget.setText(self._tr(key))
            self.auto_preview_check.setText(self._tr("check.auto_preview"))
            self.batch_resume_check.setText(self._tr("check.resume_checkpoint"))
            for widget, key in (
                (self.preview_button, "tooltip.preview"),
                (self.optimize_button, "tooltip.optimize"),
                (self.cancel_button, "tooltip.cancel"),
                (self.ignore_late_result_button, "tooltip.ignore_late"),
                (self.clear_mask_button, "tooltip.clear_mask"),
                (self.accept_current_button, "tooltip.accept_current"),
                (self.reject_current_button, "tooltip.reject_current"),
                (self.restore_before_optimize_button, "tooltip.restore_before_optimize"),
            ):
                widget.setToolTip(self._tr(key))

            self.roi_group.setTitle(self._tr("group.roi"))
            self.roi_type_label.setText(self._tr("label.type"))
            self._set_combo_text(self.roi_type_combo, "rectangle", "combo.rectangle")
            self._set_combo_text(self.roi_type_combo, "ellipse", "combo.ellipse")

            self.fit_session_group.setTitle(self._tr("group.fit_session"))
            for field, key in (
                (self.manual_status_label, "label.manual_status"),
                (self.reviewer_edit, "label.reviewer"),
                (self.review_notes_edit, "label.review_notes"),
                (self.snapshot_save_row, "label.snapshot_note"),
                (self.snapshot_restore_row, "label.saved_snapshots"),
            ):
                self._set_form_label(self.fit_session_form, field, key)
            self.reviewer_edit.setPlaceholderText(self._tr("placeholder.reviewer"))
            self.review_notes_edit.setPlaceholderText(self._tr("placeholder.review_notes"))
            self.snapshot_note_edit.setPlaceholderText(self._tr("placeholder.snapshot_note"))

            self.analysis_group.setTitle(self._tr("group.analysis"))
            for field, key in (
                (self.q_min_edit, "label.q_min"),
                (self.q_max_edit, "label.q_max"),
                (self.focus_q_window_check, "label.q_view"),
                (self.display_scale_combo, "label.display_scale"),
                (self.display_percentile_spin, "label.display_percentile"),
                (self.draw_axis_deg_spin, "label.draw_axis"),
                (self.ridge_method_combo, "label.ridge_method"),
                (self.ridge_snr_threshold_spin, "label.ridge_snr_threshold"),
                (self.ridge_min_peak_fraction_spin, "label.ridge_min_peak_fraction"),
                (self.ridge_min_coverage_spin, "label.ridge_min_coverage"),
                (self.n_angular_bins_spin, "label.angular_bins"),
                (self.n_ridge_angles_spin, "label.ridge_angles"),
                (self.n_radial_bins_spin, "label.radial_bins"),
                (self.curvature_sigma_spin, "label.curvature_sigma"),
                (self.curvature_percentile_spin, "label.curvature_percentile"),
                (self.normal_step_spin, "label.normal_step"),
                (self.max_pixels_spin, "label.max_pixels"),
                (self.full2d_multistart_spin, "label.full2d_multistart"),
            ):
                self._set_form_label(self.analysis_form, field, key)
            auto_text = self._tr("placeholder.auto")
            for edit in (self.q_min_edit, self.q_max_edit):
                if edit.text().strip().lower() in {"", "auto", "自动"}:
                    edit.setText(auto_text)
                edit.setPlaceholderText(auto_text)
            self.max_pixels_spin.setSpecialValueText(self._tr("special.all_pixels"))
            self._set_combo_text(self.ridge_method_combo, "radial_peak", "combo.radial_peak")
            self._set_combo_text(
                self.ridge_method_combo,
                "surface_curvature",
                "combo.surface_curvature",
            )
            self._set_combo_text(
                self.ridge_method_combo,
                "azimuthal_peak",
                "combo.azimuthal_peak",
            )
            self._set_combo_text(self.display_scale_combo, "linear", "combo.display_linear")
            self._set_combo_text(self.display_scale_combo, "log1p", "combo.display_log1p")
            self._set_combo_text(self.display_scale_combo, "asinh", "combo.display_asinh")

            self.ellipse_group.setTitle(self._tr("group.ellipse_constraints"))
            for field, key in (
                (self.ellipse_preset_combo, "label.ellipse_preset"),
                (self._ellipse_initial_row, "label.ellipse_initial"),
                (self._ellipse_a_control_row, "label.ellipse_a"),
                (self._ellipse_ratio_control_row, "label.ellipse_ratio"),
                (self._ellipse_b_bounds_row, "label.ellipse_b"),
                (self._ellipse_angle_control_row, "label.ellipse_angle"),
                (self._ellipse_angle_initial_row, "label.ellipse_angle_init"),
                (self.ellipse_fixed_center_check, "label.ellipse_center"),
                (self.ellipse_center_qx_spin, "label.ellipse_center_q"),
                (self.ellipse_center_qy_spin, "label.ellipse_center_q"),
                (self.ellipse_residual_combo, "label.ellipse_residual"),
                (self.ellipse_multistart_spin, "label.ellipse_multistart"),
            ):
                self._set_form_label(self.ellipse_form, field, key)
            self.ellipse_fixed_a_check.setText(self._tr("check.ellipse_fixed_a"))
            self.ellipse_fixed_ratio_check.setText(self._tr("check.ellipse_fixed_ratio"))
            self.ellipse_fixed_center_check.setText(self._tr("check.ellipse_fixed_center"))
            self.ellipse_fixed_angle_check.setText(self._tr("check.ellipse_fixed_angle"))
            self._ellipse_initial_labels["a"].setText(self._tr("label.ellipse_a"))
            self._ellipse_initial_labels["b/a"].setText(self._tr("label.ellipse_ratio"))
            self._ellipse_angle_unit_label.setText("deg")
            self.ellipse_a_init_spin.setAccessibleName(
                f"{self._tr('label.ellipse_a')} {self._tr('label.ellipse_initial')}"
            )
            self.ellipse_ratio_init_spin.setAccessibleName(
                f"{self._tr('label.ellipse_ratio')} {self._tr('label.ellipse_initial')}"
            )
            self.ellipse_angle_deg_spin.setAccessibleName(
                f"{self._tr('label.ellipse_angle_init')}"
            )
            for labels in (
                self._ellipse_a_bounds_labels,
                self._ellipse_ratio_bounds_labels,
                self._ellipse_b_bounds_labels,
                self._ellipse_angle_bounds_labels,
            ):
                labels["min"].setText(self._tr("header.min"))
                labels["max"].setText(self._tr("header.max"))
            self._ellipse_center_labels["qx"].setText("qx")
            self._ellipse_center_labels["qy"].setText("qy")
            self._set_combo_text(self.ellipse_preset_combo, "standard", "combo.standard")
            self._set_combo_text(self.ellipse_preset_combo, "flat_ellipse", "combo.flat_ellipse")
            self._set_combo_text(self.ellipse_residual_combo, "sampson", "combo.sampson")
            self._set_combo_text(self.ellipse_residual_combo, "geometric", "combo.geometric")

            for field, key in (
                (self.batch_mode_combo, "label.mode"),
                (self.batch_stage_combo, "label.batch_stage"),
                (self.batch_stream_check, "label.batch_retention"),
                (self.batch_manifest_edit, "label.manifest"),
                (self.batch_checkpoint_edit, "label.checkpoint"),
                (self.batch_output_edit, "label.output"),
                (self.batch_series_edit, "label.batch_series"),
                (self.batch_start_spin, "label.batch_start"),
                (self.batch_stop_spin, "label.batch_stop"),
                (self.batch_stride_spin, "label.batch_stride"),
                (self.batch_range_edit, "label.batch_range"),
            ):
                self._set_form_label(self.batch_form, field, key)
            self._set_combo_text(self.batch_mode_combo, "independent", "combo.independent")
            self._set_combo_text(self.batch_mode_combo, "warm_start", "combo.warm_start")
            self._set_combo_text(self.batch_stage_combo, "geometry", "combo.batch_geometry")
            self._set_combo_text(self.batch_stage_combo, "full2d", "combo.batch_full2d")
            self.batch_stream_check.setText(self._tr("check.batch_stream"))
            self.batch_stream_help.setText(self._tr("help.batch_stream"))
            self.batch_manifest_edit.setPlaceholderText(self._tr("placeholder.manifest"))
            self.batch_checkpoint_edit.setPlaceholderText(self._tr("placeholder.checkpoint"))
            self.batch_output_edit.setPlaceholderText(self._tr("placeholder.output"))
            self.batch_series_edit.setPlaceholderText(self._tr("placeholder.batch_series"))
            self.batch_range_edit.setPlaceholderText(self._tr("placeholder.batch_range"))
            self.batch_table.setHorizontalHeaderLabels(
                [self._tr("header.frame"), self._tr("header.status"), "RMSE"]
            )
            for row in range(self.batch_table.rowCount()):
                item = self.batch_table.item(row, 1)
                if item is None:
                    continue
                raw_status = str(item.data(QtCore.Qt.ItemDataRole.UserRole) or "")
                status_key = _BATCH_STATUS_KEYS.get(raw_status.casefold())
                if status_key is not None:
                    item.setText(self._tr(status_key))

            self.evolution_y_label.setText(self._tr("label.y_parameter"))
            if self.evolution_plot is not None:
                self.evolution_plot.setLabel(
                    "left",
                    self.evolution_y_key or self._tr("axis.value"),
                )
                self.evolution_plot.setLabel("bottom", self._tr("axis.frame_time"))
            elif hasattr(self, "evolution_placeholder"):
                self.evolution_placeholder.setText(
                    self._tr("measurement.evolution_placeholder")
                )

            self.lobe_panel_label.setText(self._tr("measurement.lobes"))
            self.ridge_panel_label.setText(self._tr("measurement.ridge"))
            self.ellipse_panel_label.setText(self._tr("measurement.ellipse"))
            self.radial_panel_label.setText(self._tr("measurement.radial"))
            if hasattr(self, "profile_tabs"):
                for index, (title_key, tooltip_key) in enumerate(
                    (
                        ("tab.profile.angular", "tooltip.tab.profile.angular"),
                        ("tab.profile.coverage", "tooltip.tab.profile.coverage"),
                        ("tab.profile.ridge", "tooltip.tab.profile.ridge"),
                        ("tab.profile.radial", "tooltip.tab.profile.radial"),
                    )
                ):
                    if index < self.profile_tabs.count():
                        self.profile_tabs.setTabText(index, self._tr(title_key))
                        self.profile_tabs.setTabToolTip(index, self._tr(tooltip_key))
            self.lobe_table.setHorizontalHeaderLabels(
                [
                    self._tr("header.angle_deg"),
                    self._tr("header.intensity"),
                    self._tr("header.baseline"),
                    self._tr("header.snr"),
                    self._tr("header.fwhm_deg"),
                    self._tr("header.coverage"),
                    self._tr("header.valid"),
                    self._tr("header.flags"),
                ]
            )
            self.ridge_table.setHorizontalHeaderLabels(
                [
                    self._tr("header.angle_deg"),
                    self._tr("header.q"),
                    self._tr("header.accepted"),
                    self._tr("header.method"),
                ]
            )
            self.radial_table.setHorizontalHeaderLabels(
                [
                    self._tr("header.angle_deg"),
                    self._tr("header.q_star"),
                    self._tr("header.ln"),
                    self._tr("header.snr"),
                    self._tr("header.radial_fwhm"),
                    self._tr("header.coverage"),
                    self._tr("header.valid"),
                    self._tr("header.flags"),
                ]
            )
            self.ellipse_table.setHorizontalHeaderLabels(
                [self._tr("header.quantity"), self._tr("header.value")]
            )
            if self.angular_plot is not None:
                self.angular_plot.setAccessibleName(
                    f"{self._tr('axis.angular_intensity')} plot"
                )
                self.coverage_plot.setAccessibleName(
                    f"{self._tr('axis.coverage')} plot"
                )
                self.ridge_plot.setAccessibleName(
                    f"{self._tr('axis.ridge_q', unit=self._ridge_plot_q_unit)} plot"
                )
                self.radial_profile_plot.setAccessibleName(
                    f"{self._tr('axis.radial_intensity')} plot"
                )
                self.angular_plot.setLabel("bottom", self._tr("axis.azimuth"))
                self.angular_plot.setLabel("left", self._tr("axis.angular_intensity"))
                self.coverage_plot.setLabel("bottom", self._tr("axis.azimuth"))
                self.coverage_plot.setLabel("left", self._tr("axis.coverage"))
                self.ridge_plot.setLabel("bottom", self._tr("axis.azimuth"))
                self.ridge_plot.setLabel(
                    "left",
                    self._tr("axis.ridge_q", unit=self._ridge_plot_q_unit),
                )
                self.radial_profile_plot.setLabel(
                    "bottom",
                    self._tr("axis.radial_q", unit=self._ridge_plot_q_unit),
                )
                self.radial_profile_plot.setLabel(
                    "left", self._tr("axis.radial_intensity")
                )
            else:
                self.angular_placeholder.setText(
                    self._tr("measurement.angular_placeholder")
                )
                self.coverage_placeholder.setText(
                    self._tr("measurement.coverage_placeholder")
                )
                self.ridge_placeholder.setText(
                    self._tr("measurement.ridge_placeholder")
                )
                self.radial_profile_placeholder.setText(
                    self._tr("measurement.radial_placeholder")
                )

            ellipse_keys = (
                "ellipse.a",
                "ellipse.b",
                "ellipse.axis_ratio",
                "ellipse.ellipticity",
                "ellipse.theta",
                "ellipse.ln",
                "ellipse.lz",
                "ellipse.rmse",
                "ellipse.rss",
                "ellipse.n_points",
                "ellipse.quality",
                "ellipse.flags",
                "ellipse.phi_app",
                "ellipse.alpha_candidate",
                "ellipse.psi_candidate",
                "ellipse.stderr",
                "ellipse.bound_flags",
                "ellipse.quality_status",
                "ellipse.p4_flags",
                "ellipse.q_unit",
                "ellipse.geometry_action",
                "ellipse.symmetry_status",
                "ellipse.reference_axis",
                "ellipse.quadrant_counts",
                "ellipse.paired_support",
                "ellipse.branch_leaks",
            )
            for row, key in enumerate(ellipse_keys):
                item = self.ellipse_table.item(row, 0)
                if item is not None:
                    item.setText(self._tr(key))
            self._retranslate_measurement_booleans()

            self.parameter_model.set_language(self._language)
            self.views.set_language(self._language)
            if getattr(self, "_geometry_only_result", False):
                self.views.model.state_label.setText(self._tr("view.model_unfitted"))
                self.views.residual.state_label.setText(
                    self._tr("view.residual_unavailable")
                )
            self.cancel_button.setShortcut(
                QtGui.QKeySequence(QtCore.Qt.Key.Key_Escape)
            )
            self._retranslate_accessible_names()
            if isinstance(self._last_result, Mapping):
                self._update_measurements(self._last_result)
            elif hasattr(self, "profile_summary_label"):
                self._set_profile_summary(self._tr("profile.summary_empty"))
            if hasattr(self, "workflow_status_label"):
                try:
                    from .workbench import _refresh_workflow_guide

                    _refresh_workflow_guide(self)
                except Exception:
                    pass
            self._sync_fit_session_controls(preserve_edits=True)
            self._apply_tooltips()
            self._render_status()
            self._render_metric_labels()

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
            self._mark_manual_unreviewed()

        def set_parameter(self, name: str, value: Any) -> bool:
            return self.parameter_model.set_parameter(name, value)

        @property
        def analysis_settings(self) -> dict[str, Any]:
            """Return the serializable Analysis/Measurement control state."""
            result: dict[str, Any] = {
                "q_min": _analysis_scalar(self.q_min_edit.text(), default=None),
                "q_max": _analysis_scalar(self.q_max_edit.text(), default=None),
                "draw_axis_deg": float(self.draw_axis_deg_spin.value()),
                "ridge_method": str(self.ridge_method_combo.currentData() or "radial_peak"),
                "ridge_snr_threshold": float(self.ridge_snr_threshold_spin.value()),
                "ridge_min_peak_fraction": float(self.ridge_min_peak_fraction_spin.value()),
                "ridge_min_coverage": float(self.ridge_min_coverage_spin.value()),
                "n_angular_bins": int(self.n_angular_bins_spin.value()),
                "n_ridge_angles": int(self.n_ridge_angles_spin.value()),
                "n_radial_bins": int(self.n_radial_bins_spin.value()),
                "curvature_sigma": float(self.curvature_sigma_spin.value()),
                "curvature_percentile": float(self.curvature_percentile_spin.value()),
                "normal_step": float(self.normal_step_spin.value()),
                "max_pixels": int(self.max_pixels_spin.value()),
                "full2d_multistart": int(self.full2d_multistart_spin.value()),
            }
            # Preserve service/CLI settings that have no dedicated widget.
            # Rebuilding this mapping from controls alone silently dropped
            # multiscale, seed, robust-loss and solver controls on every UI
            # edit/project round trip.
            for key in ("scales", "seed", "robust_loss", "f_scale", "max_nfev"):
                if key in self._analysis_settings:
                    result[key] = deepcopy(self._analysis_settings[key])
            preset = str(self.ellipse_preset_combo.currentData() or "standard")
            ellipse: dict[str, Any] = {"preset": preset}
            if preset != "standard":
                def optional_value(spin: Any) -> float | None:
                    value = float(spin.value())
                    return None if value == 0.0 else value

                ellipse.update(
                    {
                        "a": float(self.ellipse_a_init_spin.value()),
                        "axis_ratio": float(self.ellipse_ratio_init_spin.value()),
                        "axis_ratio_min": optional_value(self.ellipse_ratio_min_spin),
                        "axis_ratio_max": optional_value(self.ellipse_ratio_max_spin),
                        "a_min": optional_value(self.ellipse_a_min_spin),
                        "a_max": optional_value(self.ellipse_a_max_spin),
                        "b_min": optional_value(self.ellipse_b_min_spin),
                        "b_max": optional_value(self.ellipse_b_max_spin),
                        "theta_min_deg": float(self.ellipse_angle_min_spin.value()),
                        "theta_max_deg": float(self.ellipse_angle_max_spin.value()),
                        "fixed_center": bool(self.ellipse_fixed_center_check.isChecked()),
                        "center_qx": float(self.ellipse_center_qx_spin.value()),
                        "center_qy": float(self.ellipse_center_qy_spin.value()),
                        "fixed_angle": bool(self.ellipse_fixed_angle_check.isChecked()),
                        "fixed_a": bool(self.ellipse_fixed_a_check.isChecked()),
                        "fixed_axis_ratio": bool(self.ellipse_fixed_ratio_check.isChecked()),
                        "angle_deg": float(self.ellipse_angle_deg_spin.value()),
                        "residual": str(self.ellipse_residual_combo.currentData() or "sampson"),
                        "multistart": int(self.ellipse_multistart_spin.value()),
                    }
                )
            elif self._ellipse_initial_explicit:
                ellipse.update(
                    {
                        "a": float(self.ellipse_a_init_spin.value()),
                        "axis_ratio": float(self.ellipse_ratio_init_spin.value()),
                        "angle_deg": float(self.ellipse_angle_deg_spin.value()),
                    }
                )
            if preset == "standard" and (
                self.ellipse_residual_combo.currentData() != "sampson"
                or self.ellipse_multistart_spin.value() != 7
            ):
                ellipse.update(
                    {
                        "residual": str(self.ellipse_residual_combo.currentData() or "sampson"),
                        "multistart": int(self.ellipse_multistart_spin.value()),
                    }
                )
            if len(ellipse) > 1 or preset != "standard":
                result["ellipse"] = ellipse
            return result

        def _validate_analysis_controls(self) -> None:
            """Reject malformed q bounds before starting a worker."""

            values: dict[str, float | None] = {}
            for name, widget in (("q_min", self.q_min_edit), ("q_max", self.q_max_edit)):
                text = widget.text().strip()
                if text.lower() in {"", "auto", "自动"}:
                    values[name] = None
                    continue
                try:
                    number = float(text)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"{name} must be a finite number or Auto") from exc
                if not math.isfinite(number):
                    raise ValueError(f"{name} must be a finite number or Auto")
                values[name] = number
            if values["q_min"] is not None and values["q_max"] is not None and values["q_min"] >= values["q_max"]:
                raise ValueError("q min must be smaller than q max")
            if self.ellipse_preset_combo.currentData() == "flat_ellipse":
                if self.ellipse_ratio_min_spin.value() > self.ellipse_ratio_max_spin.value() and self.ellipse_ratio_max_spin.value() > 0:
                    raise ValueError("ellipse axis ratio min must not exceed max")
                if self.ellipse_angle_min_spin.value() > self.ellipse_angle_max_spin.value():
                    raise ValueError("ellipse angle min must not exceed max")
                if self.ellipse_a_min_spin.value() > self.ellipse_a_max_spin.value() and self.ellipse_a_max_spin.value() > 0:
                    raise ValueError("ellipse a min must not exceed max")
                if self.ellipse_b_min_spin.value() > self.ellipse_b_max_spin.value() and self.ellipse_b_max_spin.value() > 0:
                    raise ValueError("ellipse b min must not exceed max")
                a_init = float(self.ellipse_a_init_spin.value())
                ratio_init = float(self.ellipse_ratio_init_spin.value())
                b_init = a_init * ratio_init
                if self.ellipse_a_min_spin.value() > 0 and a_init < self.ellipse_a_min_spin.value():
                    raise ValueError("ellipse a initial value is below its minimum")
                if self.ellipse_a_max_spin.value() > 0 and a_init > self.ellipse_a_max_spin.value():
                    raise ValueError("ellipse a initial value exceeds its maximum")
                if self.ellipse_ratio_min_spin.value() > 0 and ratio_init < self.ellipse_ratio_min_spin.value():
                    raise ValueError("ellipse axis ratio initial value is below its minimum")
                if self.ellipse_ratio_max_spin.value() > 0 and ratio_init > self.ellipse_ratio_max_spin.value():
                    raise ValueError("ellipse axis ratio initial value exceeds its maximum")
                if self.ellipse_b_min_spin.value() > 0 and b_init < self.ellipse_b_min_spin.value():
                    raise ValueError("derived ellipse b initial value is below its minimum")
                if self.ellipse_b_max_spin.value() > 0 and b_init > self.ellipse_b_max_spin.value():
                    raise ValueError("derived ellipse b initial value exceeds its maximum")
                if self.ellipse_angle_deg_spin.value() < self.ellipse_angle_min_spin.value() or self.ellipse_angle_deg_spin.value() > self.ellipse_angle_max_spin.value():
                    raise ValueError("ellipse angle initial value is outside its bounds")

        def set_analysis_settings(
            self,
            settings: Mapping[str, Any] | None,
            *,
            trigger_preview: bool = True,
        ) -> None:
            """Restore analysis controls from a project/config mapping."""

            if not isinstance(settings, Mapping):
                return
            # This public/config seam can run while a worker is active.  Make
            # that older result stale just like a direct widget edit does.
            if hasattr(self, "_debounce_timer"):
                self._invalidate_pending_work(clear_fit=False)
            else:
                # During __init__ the first settings commit precedes timer
                # construction, while the generation guard already exists.
                self._generation.next()
            merged = dict(self._analysis_settings)
            nested = settings.get(
                "measurement",
                settings.get("analysis_settings", settings.get("analysis")),
            )
            if isinstance(nested, Mapping):
                merged.update(nested)
            merged.update({key: settings[key] for key in DEFAULT_ANALYSIS_SETTINGS if key in settings})
            # ``q_window`` and the flat aliases are intentionally accepted at
            # the project/config seam even though they are derived into the
            # widget pair below.  Keeping them in the merge prevents a direct
            # ProjectConfig round-trip from silently reverting to Auto/Sampson.
            for key in (
                "q_window",
                "q_range",
                "ellipse",
                "ellipse_preset",
                "ellipse_residual",
                "ellipse_multistart",
            ):
                if key in settings:
                    merged[key] = settings[key]
            configured_window = merged.get("q_window", merged.get("q_range"))
            window_explicit = "q_window" in settings or "q_range" in settings
            if isinstance(nested, Mapping):
                window_explicit = window_explicit or (
                    "q_window" in nested or "q_range" in nested
                )
            if configured_window is not None:
                if isinstance(configured_window, Mapping):
                    window_min = configured_window.get(
                        "min", configured_window.get("q_min", configured_window.get("low"))
                    )
                    window_max = configured_window.get(
                        "max", configured_window.get("q_max", configured_window.get("high"))
                    )
                else:
                    try:
                        window_min, window_max = configured_window
                    except (TypeError, ValueError) as exc:
                        raise ValueError("q_window must be a (min, max) pair") from exc
                if window_explicit or merged.get("q_min") in (None, ""):
                    merged["q_min"] = window_min
                if window_explicit or merged.get("q_max") in (None, ""):
                    merged["q_max"] = window_max
            self._analysis_settings = merged
            display = settings.get("display")
            if not isinstance(display, Mapping) and isinstance(nested, Mapping):
                display = nested.get("display")
            if not isinstance(display, Mapping):
                display = {}
            display_scale = display.get(
                "scale",
                settings.get("display_scale", self._display_scale),
            )
            display_percentile = display.get(
                "percentile",
                settings.get("display_percentile", self._display_percentile),
            )
            widgets = (
                self.q_min_edit,
                self.q_max_edit,
                self.display_scale_combo,
                self.display_percentile_spin,
                self.draw_axis_deg_spin,
                self.ridge_method_combo,
                self.ridge_snr_threshold_spin,
                self.ridge_min_peak_fraction_spin,
                self.ridge_min_coverage_spin,
                self.n_angular_bins_spin,
                self.n_ridge_angles_spin,
                self.n_radial_bins_spin,
                self.curvature_sigma_spin,
                self.curvature_percentile_spin,
                self.normal_step_spin,
                self.max_pixels_spin,
                self.full2d_multistart_spin,
                self.ellipse_preset_combo,
                self.ellipse_a_init_spin,
                self.ellipse_ratio_init_spin,
                self.ellipse_fixed_a_check,
                self.ellipse_fixed_ratio_check,
                self.ellipse_ratio_min_spin,
                self.ellipse_ratio_max_spin,
                self.ellipse_a_min_spin,
                self.ellipse_a_max_spin,
                self.ellipse_b_min_spin,
                self.ellipse_b_max_spin,
                self.ellipse_angle_min_spin,
                self.ellipse_angle_max_spin,
                self.ellipse_fixed_center_check,
                self.ellipse_center_qx_spin,
                self.ellipse_center_qy_spin,
                self.ellipse_fixed_angle_check,
                self.ellipse_angle_deg_spin,
                self.ellipse_residual_combo,
                self.ellipse_multistart_spin,
            )
            for widget in widgets:
                widget.blockSignals(True)
            try:
                auto_text = self._tr("placeholder.auto")
                self.q_min_edit.setText(
                    auto_text
                    if merged.get("q_min") in (None, "")
                    else str(merged.get("q_min"))
                )
                self.q_max_edit.setText(
                    auto_text
                    if merged.get("q_max") in (None, "")
                    else str(merged.get("q_max"))
                )
                display_mode = str(display_scale or "linear").lower().replace("-", "_")
                display_index = self.display_scale_combo.findData(display_mode)
                self.display_scale_combo.setCurrentIndex(max(0, display_index))
                try:
                    self.display_percentile_spin.setValue(float(display_percentile))
                except (TypeError, ValueError):
                    self.display_percentile_spin.setValue(99.5)
                self._display_scale = str(
                    self.display_scale_combo.currentData() or "linear"
                )
                self._display_percentile = float(self.display_percentile_spin.value())
                self.draw_axis_deg_spin.setValue(float(merged.get("draw_axis_deg", 90.0)))
                method = str(merged.get("ridge_method", "radial_peak")).lower().replace("-", "_")
                if method == "curvature":
                    method = "surface_curvature"
                method_index = self.ridge_method_combo.findData(method)
                self.ridge_method_combo.setCurrentIndex(max(0, method_index))
                self.ridge_snr_threshold_spin.setValue(max(0.0, float(merged.get("ridge_snr_threshold", 2.0))))
                self.ridge_min_peak_fraction_spin.setValue(min(1.0, max(0.0, float(merged.get("ridge_min_peak_fraction", 0.0)))))
                self.ridge_min_coverage_spin.setValue(min(1.0, max(0.0, float(merged.get("ridge_min_coverage", 0.0)))))
                self.n_angular_bins_spin.setValue(max(8, int(merged.get("n_angular_bins", 180))))
                self.n_ridge_angles_spin.setValue(max(1, int(merged.get("n_ridge_angles", 72))))
                self.n_radial_bins_spin.setValue(max(8, int(merged.get("n_radial_bins", 192))))
                self.curvature_sigma_spin.setValue(max(0.001, float(merged.get("curvature_sigma", 2.0))))
                self.curvature_percentile_spin.setValue(min(100.0, max(0.0, float(merged.get("curvature_percentile", 25.0)))))
                self.normal_step_spin.setValue(min(2.0, max(0.001, float(merged.get("normal_step", 1.0)))))
                self.max_pixels_spin.setValue(max(0, int(merged.get("max_pixels", 0))))
                self.full2d_multistart_spin.setValue(max(1, min(32, int(merged.get("full2d_multistart", 1)))))
                ellipse = merged.get("ellipse")
                ellipse = ellipse if isinstance(ellipse, Mapping) else {}
                from ..settings import canonical_ellipse_preset, ellipse_preset_defaults

                preset = canonical_ellipse_preset(
                    ellipse.get("preset", merged.get("ellipse_preset", "standard"))
                )
                if preset != "standard":
                    # Canonical defaults admit the verified very-flat regime;
                    # explicit project/UI values remain authoritative.
                    defaults = ellipse_preset_defaults(preset)
                    defaults.update(
                        {
                            "residual": merged.get("ellipse_residual", "sampson"),
                            "multistart": merged.get("ellipse_multistart", 7),
                        }
                    )
                    defaults.update(dict(ellipse))
                    ellipse = defaults
                display_preset = preset
                if self.ellipse_preset_combo.findData(display_preset) < 0:
                    # Older bundled Qt layouts expose only the compatibility
                    # entries. Add the canonical very-flat choice lazily so
                    # round-tripping keeps its explicit preset name.
                    self.ellipse_preset_combo.addItem("Very flat ellipse", display_preset)
                preset_index = self.ellipse_preset_combo.findData(display_preset)
                self.ellipse_preset_combo.setCurrentIndex(max(0, preset_index))
                def optional_set(spin: Any, value: Any) -> None:
                    try:
                        spin.setValue(0.0 if value in (None, "") else float(value))
                    except (TypeError, ValueError):
                        spin.setValue(0.0)

                optional_set(self.ellipse_ratio_min_spin, ellipse.get("axis_ratio_min"))
                optional_set(self.ellipse_ratio_max_spin, ellipse.get("axis_ratio_max"))
                optional_set(self.ellipse_a_min_spin, ellipse.get("a_min"))
                optional_set(self.ellipse_a_max_spin, ellipse.get("a_max"))
                optional_set(self.ellipse_b_min_spin, ellipse.get("b_min"))
                optional_set(self.ellipse_b_max_spin, ellipse.get("b_max"))
                try:
                    if ellipse.get("a") not in (None, ""):
                        self.ellipse_a_init_spin.setValue(float(ellipse["a"]))
                    if ellipse.get("axis_ratio") not in (None, ""):
                        self.ellipse_ratio_init_spin.setValue(float(ellipse["axis_ratio"]))
                except (TypeError, ValueError):
                    pass
                self._ellipse_initial_explicit = any(
                    key in ellipse for key in ("a", "axis_ratio", "angle_deg")
                )
                self.ellipse_angle_min_spin.setValue(float(ellipse.get("theta_min_deg", 0.0)))
                self.ellipse_angle_max_spin.setValue(float(ellipse.get("theta_max_deg", 90.0)))
                self.ellipse_fixed_center_check.setChecked(bool(ellipse.get("fixed_center", False)))
                self.ellipse_center_qx_spin.setValue(float(ellipse.get("center_qx", 0.0)))
                self.ellipse_center_qy_spin.setValue(float(ellipse.get("center_qy", 0.0)))
                self.ellipse_fixed_angle_check.setChecked(bool(ellipse.get("fixed_angle", False)))
                self.ellipse_fixed_a_check.setChecked(bool(ellipse.get("fixed_a", False)))
                self.ellipse_fixed_ratio_check.setChecked(bool(ellipse.get("fixed_axis_ratio", False)))
                self.ellipse_angle_deg_spin.setValue(float(ellipse.get("angle_deg", 0.0)))
                residual = str(ellipse.get("residual", merged.get("ellipse_residual", "sampson"))).lower()
                residual_index = self.ellipse_residual_combo.findData(residual)
                self.ellipse_residual_combo.setCurrentIndex(max(0, residual_index))
                self.ellipse_multistart_spin.setValue(
                    max(
                        1,
                        min(
                            64,
                            int(ellipse.get("multistart", merged.get("ellipse_multistart", 7))),
                        ),
                    )
                )
            finally:
                for widget in widgets:
                    widget.blockSignals(False)
            self._display_scale = str(
                self.display_scale_combo.currentData() or "linear"
            )
            self._display_percentile = float(self.display_percentile_spin.value())
            self.views.set_display_settings(
                self._display_scale,
                self._display_percentile,
            )
            setter = getattr(self.engine, "set_analysis_settings", None)
            if callable(setter):
                try:
                    setter(self.analysis_settings)
                except Exception:
                    pass
            self._mark_manual_unreviewed()
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
            self._pending_input_records.clear()
            self._debounce_timer.stop()
            self._set_busy(False)
            if clear_fit:
                self.views.clear_fit()
                self._fit_ridge_points = []
                self._rejected_ridge_points = []
                self._observed_fit_ellipses = []
                self._model_ellipses = []
                self._last_result = None
                self._last_result_signature = None
                self._last_result_kind = None
                self._geometry_only_result = False
                self._last_result_input_records = None
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
            self._capture_loaded_input_record("mask", None)
            return True

        def _capture_loaded_input_record(self, role: str, value: Any) -> None:
            """Remember the exact file bytes used to build the in-memory state."""

            from ..manual_evidence import capture_input_records

            selector_kwargs: dict[str, Any] = {}
            # Selector values are part of the fit provenance, just like the
            # file digest.  Do not attach a stale selector to an in-memory or
            # cleared record: it would make a later export look as if that
            # file-backed frame had participated in the fit.
            if value is not None and role == "source":
                selector_kwargs = {
                    "frame": self._frame,
                    "dataset": self._dataset,
                }
            elif value is not None and role == "mask":
                selector_kwargs = {
                    "mask_frame": self._mask_frame,
                    "mask_dataset": self._mask_dataset,
                }
            self._loaded_input_records[role] = capture_input_records(
                **{role: value},
                **selector_kwargs,
            )[role]

        def _active_q_unit(self, result: Any = None) -> str:
            """Read the calibrated q unit from result or qmap metadata."""

            candidates = [
                _read(result, ("q_unit", "unit"), None),
                _read(
                    _result_value(result, ("ellipse_fit", "ellipse", "ellipse_result"), None),
                    ("q_unit", "unit"),
                    None,
                ),
                _read(
                    _result_value(result, ("observables", "measurements"), None),
                    ("q_unit", "unit"),
                    None,
                ),
                _read(self._qmap, ("q_unit", "unit"), None),
                _read(_read(self._qmap, ("metadata",), {}), ("q_unit", "unit"), None),
            ]
            for candidate in candidates:
                unit = str(candidate or "").strip()
                if unit and unit.lower() != "unknown":
                    return unit
            return "unknown"

        def _refresh_q_parameter_units(self, q_unit: Any = None) -> None:
            """Apply the active physical q unit to q-valued table rows."""

            unit = self._active_q_unit({"q_unit": q_unit}) if q_unit is not None else self._active_q_unit()
            if not unit or unit.lower() in {"unknown", "pixel", "pixels", "pixel-q", "pixel_q"}:
                return
            q_names = {"a", "b", "q_center", "q_major", "q_minor", "radial_sigma", "radial_gamma", "radial_fwhm", "background_width"}
            for row_index, row in enumerate(self.parameter_model.rows):
                if row.name not in q_names or row.unit == unit:
                    continue
                row.unit = unit
                data_changed = getattr(self.parameter_model, "dataChanged", None)
                index = getattr(self.parameter_model, "index", None)
                if data_changed is not None and hasattr(data_changed, "emit") and callable(index):
                    data_changed.emit(
                        index(row_index, 0),
                        index(row_index, self.parameter_model.columnCount() - 1),
                        [
                            QtCore.Qt.ItemDataRole.DisplayRole,
                            QtCore.Qt.ItemDataRole.ToolTipRole,
                        ],
                    )

        @property
        def fit_session(self) -> dict[str, Any]:
            """Return a detached copy of the manual fit-review session."""

            return deepcopy(self._fit_session)

        def _fit_state_signature(self) -> str:
            """Return the small deterministic identity of the current fit inputs."""

            state = {
                "parameters": self.parameter_model.parameter_dict(),
                "analysis": self.analysis_settings,
                "input": {
                    "path": self._source_path,
                    "frame": self._frame,
                    "dataset": self._dataset,
                },
                "poni": self._poni_path,
                "mask": {
                    "path": self._mask_path,
                    "frame": self._mask_frame,
                    "dataset": self._mask_dataset,
                },
                "rois": self._roi_specs,
                "input_records": self._loaded_input_records,
            }
            return json.dumps(
                _jsonable(state),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )

        def _result_has_fit_images(self, result: Any) -> bool:
            """Return whether one result can support visual human review."""

            if (
                result is None
                or _result_has_failure(result)
                or self._is_geometry_only_result(result)
            ):
                return False
            observed = _result_value(result, ("observed", "data", "image"), self._observed)
            model = _result_value(
                result,
                ("model", "predicted", "fit", "intensity", "simulation"),
                result if not isinstance(result, Mapping) else None,
            )
            if observed is None or model is None:
                return False
            if _np is None:
                return True
            try:
                observed_array = _np.asarray(observed)
                model_array = _np.asarray(model)
            except (TypeError, ValueError):
                return False
            return (
                observed_array.ndim == 2
                and observed_array.size > 0
                and model_array.shape == observed_array.shape
            )

        def _is_geometry_only_result(self, result: Any = None) -> bool:
            """Identify geometry actions that intentionally omit intensity fit."""

            candidate = self._last_result if result is None else result
            action = _result_value(
                candidate,
                ("geometry_action", "geometry_stage", "analysis_stage"),
                None,
            )
            if str(action or "").strip().lower() in {
                "remeasure",
                "refine",
                "geometry",
                "geometry_only",
            }:
                return True
            stage = _result_value(candidate, ("stage",), None)
            if str(stage or "").strip().lower() in {
                "geometry",
                "geometry_only",
                "measurement",
            }:
                return True
            model_status = _result_value(candidate, ("model_status",), None)
            if str(model_status or "").strip().lower() in {
                "unfitted_preview",
                "geometry_only",
                "not_run",
                "unfitted",
            }:
                return True
            return str(getattr(self, "_last_result_kind", "")).lower() in {
                "measure_geometry",
                "refine_geometry",
            }

        def _current_result_is_reviewable(self) -> bool:
            """Require a successful current result, not a stale displayed image."""

            return (
                self._last_result_signature is not None
                and self._last_result_signature == self._fit_state_signature()
                and self._result_has_fit_images(self._last_result)
            )

        def _capture_fit_context(
            self,
            parameters: Mapping[str, Any] | None = None,
            *,
            include_arrays: bool = True,
        ) -> dict[str, Any]:
            """Capture the current fit context before a background request.

            Optimize receives detached detector/q/mask arrays when available;
            project persistence later removes those arrays and keeps only the
            reproducible file selectors and analysis settings.
            """

            if parameters is None:
                parameters = self.parameter_model.parameter_dict()
            input_context: dict[str, Any] = {
                "path": self._source_path,
                "frame": self._frame,
                "dataset": self._dataset,
            }
            mask_context: dict[str, Any] = {
                "path": self._mask_path,
                "frame": self._mask_frame,
                "dataset": self._mask_dataset,
            }
            qmap_context: dict[str, Any] = {}
            if include_arrays:
                input_context["data"] = deepcopy(self._observed)
                mask_context["file"] = deepcopy(self._file_mask)
                mask_context["external"] = deepcopy(self._external_mask)
                qmap_context = {
                    "qx": deepcopy(self._qx),
                    "qy": deepcopy(self._qy),
                    "value": deepcopy(self._qmap),
                }
            context = {
                "parameters": deepcopy(parameters),
                "input": input_context,
                "frame": self._frame,
                "dataset": self._dataset,
                "poni": deepcopy(self._poni_path),
                "mask": mask_context,
                "analysis": deepcopy(self.analysis_settings),
                "roi": {
                    "exclusion": deepcopy(self._exclusion_roi),
                    "specs": deepcopy(self._roi_specs),
                },
            }
            if qmap_context:
                context["qmap"] = qmap_context
            return context

        @staticmethod
        def _fit_context_for_project(context: Any) -> Any:
            """Remove detector-sized arrays from a persisted fit context."""

            if not isinstance(context, Mapping):
                return None
            clean = deepcopy(dict(context))
            input_context = clean.get("input")
            if isinstance(input_context, Mapping):
                input_context = dict(input_context)
                input_context.pop("data", None)
                clean["input"] = input_context
            mask_context = clean.get("mask")
            if isinstance(mask_context, Mapping):
                mask_context = dict(mask_context)
                for key in ("file", "external", "data", "array"):
                    mask_context.pop(key, None)
                clean["mask"] = mask_context
            clean.pop("qmap", None)
            return _jsonable(clean)

        @staticmethod
        def _fit_result_summary(result: Any) -> dict[str, Any]:
            """Keep review-relevant result metadata without detector images."""

            if not isinstance(result, Mapping):
                return {}
            summary: dict[str, Any] = {}
            for key in ("status", "solver_status", "quality_status", "flags", "q_unit"):
                value = result.get(key)
                if value is not None:
                    summary[key] = deepcopy(value)
            observed = _result_value(result, ("observed", "data", "image"), None)
            if observed is not None:
                try:
                    summary["shape"] = [int(value) for value in _np.asarray(observed).shape]
                except (AttributeError, TypeError, ValueError):
                    pass
            metrics = _read(result, ("metrics", "statistics", "summary"), None)
            if isinstance(metrics, Mapping):
                def metadata_value(value: Any) -> Any:
                    if _np is not None and isinstance(value, _np.ndarray):
                        return None
                    if isinstance(value, Mapping):
                        return {str(key): metadata_value(item) for key, item in value.items()}
                    if isinstance(value, (list, tuple)):
                        return [metadata_value(item) for item in value]
                    return deepcopy(value)

                summary["metrics"] = {
                    str(key): metadata_value(value)
                    for key, value in metrics.items()
                }
            return _jsonable(summary)

        def _normalise_fit_session(self, source: Any) -> dict[str, Any]:
            """Load a schema-1/2 session while treating absent fields as empty."""

            session = _new_fit_session()
            if not isinstance(source, Mapping):
                return session
            status = str(source.get("manual_status", "unreviewed") or "unreviewed").lower()
            session["manual_status"] = status if status in {"unreviewed", "accepted", "rejected"} else "unreviewed"
            session["reviewed_by"] = str(source.get("reviewed_by", source.get("reviewer", "")) or "")
            session["reviewed_at"] = source.get("reviewed_at")
            session["review_notes"] = str(source.get("review_notes", "") or "")
            for key in ("optimize_before", "optimize_after"):
                value = source.get(key)
                session[key] = deepcopy(value) if isinstance(value, Mapping) else None
            accepted = source.get("accepted_parameters")
            session["accepted_parameters"] = deepcopy(accepted) if isinstance(accepted, Mapping) else None
            snapshots = source.get("snapshots", [])
            if isinstance(snapshots, Iterable) and not isinstance(snapshots, (str, bytes, Mapping)):
                for snapshot in snapshots:
                    if not isinstance(snapshot, Mapping):
                        continue
                    parameters = snapshot.get("parameters")
                    if not isinstance(parameters, Mapping):
                        continue
                    item = {
                        "order": len(session["snapshots"]) + 1,
                        "note": str(snapshot.get("note", "") or ""),
                        "created_at": snapshot.get("created_at"),
                        "parameters": deepcopy(parameters),
                    }
                    if isinstance(snapshot.get("context"), Mapping):
                        item["context"] = deepcopy(snapshot["context"])
                    session["snapshots"].append(item)
            return session

        def _sync_fit_session_controls(self, *, preserve_edits: bool = False) -> None:
            """Refresh the compact right-dock controls from session state."""

            if not hasattr(self, "manual_status_label"):
                return
            status = str(self._fit_session.get("manual_status", "unreviewed"))
            self.manual_status_label.setText(self._tr(f"manual.{status}"))
            if not preserve_edits:
                self.reviewer_edit.setText(
                    str(self._fit_session.get("reviewed_by", "") or "")
                )
                self.review_notes_edit.setText(
                    str(self._fit_session.get("review_notes", "") or "")
                )
            selected = self.snapshot_combo.currentData() if self.snapshot_combo.count() else None
            self.snapshot_combo.blockSignals(True)
            self.snapshot_combo.clear()
            snapshots = self._fit_session.get("snapshots", [])
            if isinstance(snapshots, list):
                for index, snapshot in enumerate(snapshots):
                    if not isinstance(snapshot, Mapping):
                        continue
                    note = str(snapshot.get("note", "") or "")
                    label = self._tr(
                        "snapshot.label",
                        index=index + 1,
                        note=note or self._tr("snapshot.no_note"),
                    )
                    self.snapshot_combo.addItem(label, index)
            if self.snapshot_combo.count():
                if isinstance(selected, int) and 0 <= selected < self.snapshot_combo.count():
                    self.snapshot_combo.setCurrentIndex(selected)
                else:
                    self.snapshot_combo.setCurrentIndex(self.snapshot_combo.count() - 1)
            self._refresh_snapshot_item_tooltips()
            self.snapshot_combo.blockSignals(False)
            self.restore_snapshot_button.setEnabled(bool(self.snapshot_combo.count()))
            self.restore_before_optimize_button.setEnabled(
                isinstance(self._fit_session.get("optimize_before"), Mapping)
            )
            reviewable = self._current_result_is_reviewable()
            self.accept_current_button.setEnabled(reviewable)
            self.reject_current_button.setEnabled(reviewable)
            if hasattr(self, "export_evidence_action"):
                self.export_evidence_action.setEnabled(reviewable)

        def _mark_manual_unreviewed(self, *, clear_candidate: bool = True) -> None:
            """Invalidate a previous human decision after a state edit."""

            if self._fit_session_restore_active:
                return
            self._fit_session["manual_status"] = "unreviewed"
            self._fit_session["reviewed_by"] = ""
            self._fit_session["reviewed_at"] = None
            self._fit_session["review_notes"] = ""
            self._fit_session["accepted_parameters"] = None
            self._last_result_signature = None
            self._last_result_input_records = None
            if clear_candidate:
                self._fit_session["optimize_after"] = None
            self._sync_fit_session_controls()

        def _review_candidate(self, status: str) -> bool:
            """Apply an explicit Accept/Reject decision to the current fit."""

            reviewer = self.reviewer_edit.text().strip()
            if not reviewer:
                self._set_status("status.reviewer_required", flags="review_required")
                return False
            if not self._current_result_is_reviewable():
                self._set_status(
                    "status.review_result_required",
                    flags="review_required",
                )
                return False
            self._fit_session["manual_status"] = status
            self._fit_session["reviewed_by"] = reviewer
            self._fit_session["reviewed_at"] = _utc_timestamp()
            self._fit_session["review_notes"] = self.review_notes_edit.text().strip()
            self._fit_session["accepted_parameters"] = (
                deepcopy(self.parameter_model.parameter_dict()) if status == "accepted" else None
            )
            self._sync_fit_session_controls()
            self._set_status(
                "status.manual_accepted" if status == "accepted" else "status.manual_rejected",
                flags=f"manual_{status}",
            )
            return True

        def accept_current(self) -> bool:
            """Explicitly accept the current Preview/Optimize result."""

            return self._review_candidate("accepted")

        def reject_current(self) -> bool:
            """Explicitly reject the current Preview/Optimize result."""

            return self._review_candidate("rejected")

        def _manual_evidence_result(self) -> dict[str, Any]:
            """Build the exporter payload from the committed UI result."""

            if isinstance(self._last_result, Mapping):
                payload = dict(self._last_result)
            elif self._last_result is not None:
                payload = {"model": self._last_result}
            else:
                raise ValueError("No Preview/Optimize result is available")
            observed = _result_value(payload, ("observed", "data", "image"), self._observed)
            model = _result_value(
                payload,
                ("model", "predicted", "fit", "intensity", "simulation"),
                None,
            )
            residual = _result_value(payload, ("residual", "difference", "resid"), None)
            if residual is None and observed is not None and model is not None and _np is not None:
                observed_array = _np.asarray(observed, dtype=float)
                model_array = _np.asarray(model, dtype=float)
                if observed_array.shape == model_array.shape:
                    residual = observed_array - model_array
            payload["observed"] = observed
            payload["model"] = model
            payload["residual"] = residual
            # The visible parameter table is authoritative for manual export.
            payload["parameters"] = deepcopy(self.parameter_model.parameter_dict())
            payload.setdefault("qx", self._qx)
            payload.setdefault("qy", self._qy)
            payload["q_unit"] = self._active_q_unit(payload)
            payload.setdefault("valid_mask", _read(self._qmap, ("valid_mask", "valid"), None))
            payload.setdefault("external_mask", self._external_mask)
            payload.setdefault("ridge_points", self._fit_ridge_points)
            payload.setdefault("ellipses", self._observed_fit_ellipses)
            return payload

        def _manual_evidence_context(self) -> dict[str, Any]:
            """Return arrays/selectors needed by the seven-file exporter."""

            return {
                "source": self._source_path,
                "frame": self._frame,
                "dataset": self._dataset,
                "poni": self._poni_path,
                "mask_path": self._mask_path,
                "roi": deepcopy(self._roi_specs),
                "analysis": deepcopy(self.analysis_settings),
                "result_kind": self._last_result_kind,
                "qmap": self._qmap,
                "qx": self._qx,
                "qy": self._qy,
                "q_unit": self._active_q_unit(self._last_result),
                "valid_mask": _read(self._qmap, ("valid_mask", "valid"), None),
                "external_mask": self._external_mask,
                "current_model_ellipses": deepcopy(self._model_ellipses),
                "fit_input_records": deepcopy(self._last_result_input_records),
            }

        def export_manual_evidence(
            self,
            path: str | Path | bool | None = None,
            *,
            force: bool = False,
        ) -> bool:
            """Export the current fit as exactly seven auditable files."""

            if not self._current_result_is_reviewable():
                self._set_status(
                    "status.evidence_stale",
                    flags="evidence_stale",
                )
                return False
            status = str(self._fit_session.get("manual_status", "unreviewed"))
            if status in {"accepted", "rejected"} and (
                self.reviewer_edit.text().strip() != str(self._fit_session.get("reviewed_by", "") or "")
                or self.review_notes_edit.text().strip()
                != str(self._fit_session.get("review_notes", "") or "")
            ):
                self._set_status(
                    "status.review_fields_changed",
                    flags="review_required",
                )
                return False
            if isinstance(path, bool) or path is None:
                chosen = QtWidgets.QFileDialog.getExistingDirectory(
                    self,
                    self._tr("dialog.evidence_folder"),
                    "",
                )
                if not chosen:
                    return False
                path = chosen
            try:
                from ..manual_evidence import export_manual_fit

                written = export_manual_fit(
                    self._manual_evidence_result(),
                    Path(path),
                    context=self._manual_evidence_context(),
                    review={
                        "manual_status": status,
                        "reviewed_by": self._fit_session.get("reviewed_by"),
                        "reviewed_at": self._fit_session.get("reviewed_at"),
                        "review_notes": self._fit_session.get("review_notes", ""),
                    },
                    force=bool(force),
                )
            except (OSError, TypeError, ValueError) as exc:
                self._set_status(
                    "status.evidence_failed",
                    flags="evidence_error",
                    error=exc,
                )
                return False
            self._last_evidence_paths = dict(written)
            self._set_status(
                "status.evidence_exported",
                flags="evidence_exported",
                path=Path(path),
            )
            return True

        export_evidence = export_manual_evidence

        def restore_before_optimize(self, *_: Any) -> bool:
            """Restore the most recent Optimize-before parameter table."""

            before = self._fit_session.get("optimize_before")
            parameters = before.get("parameters") if isinstance(before, Mapping) else None
            if not isinstance(parameters, Mapping):
                self._set_status("status.no_optimize_snapshot", flags="snapshot_missing")
                return False
            self._invalidate_pending_work(clear_fit=True)
            self._fit_session_restore_active = True
            try:
                self.parameter_model.set_rows(deepcopy(parameters))
                setter = getattr(self.engine, "set_parameters", None)
                if callable(setter):
                    setter(self.parameter_model.parameter_dict())
            finally:
                self._fit_session_restore_active = False
            self._fit_session["manual_status"] = "unreviewed"
            self._fit_session["reviewed_by"] = ""
            self._fit_session["reviewed_at"] = None
            self._fit_session["review_notes"] = ""
            self._fit_session["accepted_parameters"] = None
            self._fit_session["optimize_after"] = None
            self._sync_fit_session_controls()
            self._refresh_model_overlay()
            self._set_status("status.restored_before_optimize", flags="snapshot_restored")
            return True

        def save_snapshot(self, note: str | None = None) -> bool:
            """Append a detached, ordered parameter snapshot with a note."""

            if isinstance(note, bool) or note is None:
                note = self.snapshot_note_edit.text()
            note = str(note).strip()
            if not note:
                self._set_status("status.snapshot_note_required", flags="snapshot_note_required")
                return False
            snapshots = self._fit_session.setdefault("snapshots", [])
            snapshot = {
                "order": len(snapshots) + 1,
                "note": note,
                "created_at": _utc_timestamp(),
                "parameters": deepcopy(self.parameter_model.parameter_dict()),
                "context": self._capture_fit_context(parameters={}, include_arrays=False),
            }
            snapshots.append(snapshot)
            self._sync_fit_session_controls()
            self.snapshot_combo.setCurrentIndex(self.snapshot_combo.count() - 1)
            self.snapshot_note_edit.clear()
            self._set_status(
                "status.snapshot_saved",
                index=snapshot["order"],
                note=note,
            )
            return True

        def restore_snapshot(self, index: int | None = None, *_: Any) -> bool:
            """Restore one saved snapshot and clear views from the old fit."""

            if isinstance(index, bool):
                index = None
            snapshots = self._fit_session.get("snapshots", [])
            if not isinstance(snapshots, list) or not snapshots:
                self._set_status("status.no_snapshot", flags="snapshot_missing")
                return False
            if index is None:
                selected = self.snapshot_combo.currentData()
                index = int(selected) if selected is not None else self.snapshot_combo.currentIndex()
            try:
                snapshot = snapshots[int(index)]
            except (IndexError, TypeError, ValueError):
                self._set_status("status.snapshot_invalid", flags="snapshot_invalid")
                return False
            parameters = snapshot.get("parameters") if isinstance(snapshot, Mapping) else None
            if not isinstance(parameters, Mapping):
                self._set_status("status.snapshot_no_parameters", flags="snapshot_invalid")
                return False
            self._invalidate_pending_work(clear_fit=True)
            self._fit_session_restore_active = True
            try:
                self.parameter_model.set_rows(deepcopy(parameters))
                setter = getattr(self.engine, "set_parameters", None)
                if callable(setter):
                    setter(self.parameter_model.parameter_dict())
            finally:
                self._fit_session_restore_active = False
            self._fit_session["manual_status"] = "unreviewed"
            self._fit_session["reviewed_by"] = ""
            self._fit_session["reviewed_at"] = None
            self._fit_session["review_notes"] = ""
            self._fit_session["accepted_parameters"] = None
            self._fit_session["optimize_after"] = None
            self._sync_fit_session_controls()
            self.snapshot_combo.setCurrentIndex(int(index))
            self._refresh_model_overlay()
            note = str(snapshot.get("note", "") or "")
            self._set_status(
                "status.snapshot_restored",
                flags="snapshot_restored",
                index=int(index) + 1,
                note=note,
            )
            return True

        # Descriptive aliases keep the small public API discoverable for
        # scripts while the button-facing names remain concise.
        accept_current_result = accept_current
        reject_current_result = reject_current
        restore_before_fit = restore_before_optimize
        save_parameter_snapshot = save_snapshot
        restore_parameter_snapshot = restore_snapshot

        def set_observed_data(
            self,
            data: Any,
            *,
            qx: Any = None,
            qy: Any = None,
            qmap: Any = None,
            metadata: Mapping[str, Any] | None = None,
            _preserve_file_context: bool = False,
        ) -> None:
            self._invalidate_pending_work(clear_fit=True)
            self._mark_manual_unreviewed()
            if not _preserve_file_context:
                self._source_path = None
                self._frame = None
                self._dataset = None
                self._capture_loaded_input_record("source", None)
            mask_cleared_for_shape = self._clear_incompatible_external_mask(data)
            self._observed = data
            self._qx, self._qy = qx, qy
            if qmap is not None:
                self._qmap = qmap
                self._qx = _read(qmap, ("qx", "qx_nm_inv"), self._qx)
                self._qy = _read(qmap, ("qy", "qy_nm_inv"), self._qy)
            elif qx is not None and qy is not None:
                self._qmap = {"qx": qx, "qy": qy}
            self._refresh_q_parameter_units()
            setter = getattr(self.engine, "set_observed", None)
            if callable(setter):
                try:
                    state = setter(data, qx=qx, qy=qy, qmap=qmap, metadata=metadata)
                    if isinstance(state, Mapping):
                        self._qmap = state.get("qmap", self._qmap)
                        self._qx = _read(self._qmap, ("qx", "qx_nm_inv"), self._qx)
                        self._qy = _read(self._qmap, ("qy", "qy_nm_inv"), self._qy)
                        self._refresh_q_parameter_units()
                except Exception as exc:
                    self._set_status("status.geometry_failed", flags="error", error=exc)
            # Keep the overlay background synchronized even before the first
            # preview result arrives.  q extent is computed by ViewGrid only
            # for the overlay; the other three views remain pixel-space.
            self.views.set_images(
                data,
                qx=self._qx,
                qy=self._qy,
                q_unit=self._active_q_unit(),
                valid_mask=_read(self._qmap, ("valid_mask", "valid"), None),
                external_mask=self._external_mask,
            )
            if self._roi_specs or self._file_mask is not None:
                self._recompute_external_mask(update_widgets=False)
            if mask_cleared_for_shape:
                self._set_status(
                    "status.mask_shape_changed",
                    flags="mask_cleared_shape_changed",
                )

        set_observed = set_observed_data

        def set_poni(self, path: str | Path | Any) -> bool:
            setter = getattr(self.engine, "set_poni", None)
            if not callable(setter):
                self._set_status("status.no_poni_support", flags="error")
                return False
            self._invalidate_pending_work(clear_fit=True)
            self._mark_manual_unreviewed()
            try:
                qmap = setter(path)
                self._poni_path = str(path) if isinstance(path, (str, Path)) else "in-memory"
                self._capture_loaded_input_record(
                    "poni",
                    path if isinstance(path, (str, Path)) else None,
                )
                if qmap is not None:
                    self._qmap = qmap
                    self._qx = _read(qmap, ("qx", "qx_nm_inv"), self._qx)
                    self._qy = _read(qmap, ("qy", "qy_nm_inv"), self._qy)
                    self._refresh_q_parameter_units()
                    if self._observed is not None:
                        self.views.set_images(
                            self._observed,
                            qx=self._qx,
                            qy=self._qy,
                            q_unit=self._active_q_unit(),
                            valid_mask=_read(self._qmap, ("valid_mask", "valid"), None),
                            external_mask=self._external_mask,
                        )
                self._set_status(
                    "status.poni_loaded",
                    name=Path(path).name if isinstance(path, (str, Path)) else self._poni_path,
                )
                return True
            except Exception as exc:
                self._set_status("status.poni_failed", flags="error", error=exc)
                return False

        def select_poni(self, path: str | Path | bool | None = None) -> bool:
            if isinstance(path, bool) or path is None:
                chosen, _ = QtWidgets.QFileDialog.getOpenFileName(
                    self,
                    self._tr("dialog.select_poni"),
                    "",
                    self._tr(
                        "filter.poni",
                        all_files=self._tr("filter.all_files"),
                    ),
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
                    self._tr("dialog.open_image"),
                    "",
                    self._tr(
                        "filter.images",
                        all_files=self._tr("filter.all_files"),
                    ),
                )
                if not chosen:
                    return False
                path = chosen
            loader = getattr(self.engine, "load_image", None)
            if not callable(loader):
                self._set_status("status.no_image_support", flags="error")
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
                self._capture_loaded_input_record("source", path)
                if external_mask is not None:
                    # A supplied mask replaces the previous detector mask.  Do
                    # this only after loading succeeds so a failed image/mask
                    # selection leaves the current document intact.
                    self._mask_path = None
                    self._file_mask = None
                    self._external_mask = None
                    self._capture_loaded_input_record("mask", None)
                self._poni_path = state.get("poni", self._poni_path)
                if poni is not None:
                    self._capture_loaded_input_record(
                        "poni",
                        poni if isinstance(poni, (str, Path)) else None,
                    )
                self.set_observed_data(
                    state.get("observed", state.get("data")),
                    qx=state.get("qx"),
                    qy=state.get("qy"),
                    qmap=state.get("qmap"),
                    metadata=state.get("metadata"),
                    _preserve_file_context=True,
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
                        self._capture_loaded_input_record("mask", None)
                    self._recompute_external_mask(update_widgets=False)
                self._set_status("status.image_loaded", name=Path(path).name)
                return True
            except Exception as exc:
                self._set_status("status.image_failed", flags="error", error=exc)
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
            self._capture_loaded_input_record("mask", path)
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
                    q_unit=self._active_q_unit(),
                    valid_mask=_read(self._qmap, ("valid_mask", "valid"), None),
                    external_mask=self._external_mask,
                )
                if update_widgets:
                    self._sync_roi_widgets()
                excluded = int(_np.count_nonzero(self._external_mask))
                self._set_status("status.mask_applied", count=excluded)
                return True
            except Exception as exc:
                self._set_status("status.mask_apply_failed", flags="error", error=exc)
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
                    self._tr("dialog.select_mask"),
                    "",
                    self._tr(
                        "filter.masks",
                        all_files=self._tr("filter.all_files"),
                    ),
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
                self._mark_manual_unreviewed()
                self._set_status("status.mask_loaded", name=Path(path).name)
                return True
            except Exception as exc:
                self._set_status("status.mask_failed", flags="error", error=exc)
                return False

        open_mask = select_mask

        def clear_external_mask(self) -> bool:
            self._invalidate_pending_work(clear_fit=True)
            self._mark_manual_unreviewed()
            self._mask_path = None
            self._file_mask = None
            self._capture_loaded_input_record("mask", None)
            if self._observed is not None and self._roi_specs:
                return self._recompute_external_mask(update_widgets=False)
            self._external_mask = None
            if self._observed is not None:
                self.views.set_images(
                    self._observed,
                    qx=self._qx,
                    qy=self._qy,
                    q_unit=self._active_q_unit(),
                    valid_mask=_read(self._qmap, ("valid_mask", "valid"), None),
                    external_mask=None,
                )
            self._set_status("status.mask_cleared")
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
                        self._set_status("status.ellipse_roi_fields", flags="error")
                        return False
                    if not all(math.isfinite(float(spec[key])) for key in ("cx", "cy", "rx", "ry", "angle_deg")) or spec["rx"] <= 0 or spec["ry"] <= 0:
                        self._set_status("status.ellipse_roi_radii", flags="error")
                        return False
                    self._exclusion_roi = dict(spec)
                else:
                    try:
                        spec = {"type": "rectangle", **{key: float(spec[key]) for key in ("x0", "y0", "x1", "y1")}}
                    except (KeyError, TypeError, ValueError):
                        self._set_status("status.rectangle_roi_fields", flags="error")
                        return False
                    if not all(math.isfinite(float(spec[key])) for key in ("x0", "y0", "x1", "y1")) or spec["x1"] <= spec["x0"] or spec["y1"] <= spec["y0"]:
                        self._set_status("status.rectangle_roi_bounds", flags="error")
                        return False
                    self._exclusion_roi = tuple(spec[key] for key in ("x0", "y0", "x1", "y1"))
            else:
                try:
                    values = tuple(float(item) for item in roi)
                except (TypeError, ValueError):
                    self._set_status("status.roi_fields", flags="error")
                    return False
                if len(values) != 4 or not all(math.isfinite(item) for item in values):
                    self._set_status("status.roi_finite", flags="error")
                    return False
                x0, y0, x1, y1 = values
                if x1 <= x0 or y1 <= y0:
                    self._set_status("status.roi_bounds", flags="error")
                    return False
                self._exclusion_roi = values
                spec = {"type": "rectangle", "x0": x0, "x1": x1, "y0": y0, "y1": y1}
            self._roi_specs = [spec]
            self._mark_manual_unreviewed()
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
            self._mark_manual_unreviewed()
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
                    q_unit=self._active_q_unit(),
                    valid_mask=_read(self._qmap, ("valid_mask", "valid"), None),
                    external_mask=None,
                )
            self._set_status("status.roi_cleared")
            return True

        def _reference_axis_deg(self) -> float:
            try:
                return float(self.analysis_settings.get("draw_axis_deg", 90.0)) - 90.0
            except (TypeError, ValueError):
                return 0.0

        def focus_q_window(self) -> None:
            """Zoom the reciprocal-space overlay to the active radial window."""

            result = self._last_result
            domain = _result_value(result, ("analysis_domain",), None)
            q_window = _read(domain, ("q_window",), None)
            if q_window is None:
                analysis = self.analysis_settings
                q_window = (
                    analysis.get("q_min"),
                    analysis.get("q_max"),
                )
            if q_window is None or any(value is None for value in q_window):
                q_values = _read(self._qmap, ("q", "q_nm_inv"), None)
                if q_values is not None and _np is not None:
                    try:
                        finite = _np.asarray(q_values, dtype=float)
                        finite = finite[_np.isfinite(finite)]
                        q_window = (
                            float(_np.min(finite)),
                            float(_np.max(finite)),
                        ) if finite.size else None
                    except (TypeError, ValueError):
                        q_window = None
            self.views.set_q_view(q_window)

        def reset_q_view(self) -> None:
            """Restore the full detector/q-map extent in the overlay."""

            self.views.set_q_view(full=True)

        def _on_q_view_setting_changed(self, enabled: bool) -> None:
            if enabled:
                self.focus_q_window()
            else:
                self.reset_q_view()

        @property
        def display_settings(self) -> dict[str, Any]:
            return {
                "scale": str(self.display_scale_combo.currentData() or "linear"),
                "percentile": float(self.display_percentile_spin.value()),
            }

        def set_display_settings(self, settings: Mapping[str, Any] | None) -> None:
            """Restore display-only contrast settings without invalidating fits."""

            if not isinstance(settings, Mapping):
                return
            mode = str(settings.get("scale", "linear") or "linear").lower().replace(
                "-", "_"
            )
            index = self.display_scale_combo.findData(mode)
            self.display_scale_combo.blockSignals(True)
            self.display_percentile_spin.blockSignals(True)
            try:
                self.display_scale_combo.setCurrentIndex(max(0, index))
                try:
                    self.display_percentile_spin.setValue(
                        float(settings.get("percentile", 99.5))
                    )
                except (TypeError, ValueError):
                    self.display_percentile_spin.setValue(99.5)
            finally:
                self.display_scale_combo.blockSignals(False)
                self.display_percentile_spin.blockSignals(False)
            self._on_display_setting_changed()

        def _on_display_setting_changed(self, *_: Any) -> None:
            """Refresh contrast only; fitted evidence remains current."""

            self._display_scale = str(
                self.display_scale_combo.currentData() or "linear"
            )
            self._display_percentile = float(self.display_percentile_spin.value())
            self.views.set_display_settings(
                self._display_scale,
                self._display_percentile,
            )

        def _refresh_model_overlay(self) -> None:
            self._model_ellipses = model_ellipse_pair(
                self.parameter_model.parameter_values(),
                reference_axis_deg=self._reference_axis_deg(),
            )
            self.views.set_overlay(
                self._fit_ridge_points,
                self._observed_fit_ellipses,
                model_ellipses=self._model_ellipses,
                rejected_ridge_points=getattr(self, "_rejected_ridge_points", []),
            )

        def set_fit_overlay(self, ridge_points: Any = None, ellipses: Any = None) -> None:
            source = [] if ridge_points is None else ridge_points
            rows = [source] if isinstance(source, Mapping) else _sequence(source)
            accepted: list[Any] = []
            rejected: list[Any] = []
            for row in rows:
                if bool(_read(row, ("accepted", "valid"), True)):
                    accepted.append(row)
                else:
                    rejected.append(row)
            self._fit_ridge_points = accepted if rows else source
            self._rejected_ridge_points = rejected
            self._observed_fit_ellipses = [] if ellipses is None else ellipses
            if isinstance(self._observed_fit_ellipses, Mapping):
                self._observed_fit_ellipses = [self._observed_fit_ellipses]
            self._refresh_model_overlay()

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
            active = next(iter(self._workers.values()), None)
            self._set_busy(bool(active), getattr(active, "kind", "edited"))
            self._set_status("status.parameters_changed")
            self._mark_manual_unreviewed()
            self._refresh_model_overlay()
            if self.auto_preview:
                self._debounce_timer.start(self.debounce_ms)

        def _mark_ellipse_initial_explicit(self, *_: Any) -> None:
            self._ellipse_initial_explicit = True

        def _on_analysis_changed(self, *_: Any) -> None:
            try:
                self._validate_analysis_controls()
            except ValueError as exc:
                self._generation.next()
                self._mark_manual_unreviewed()
                active = next(iter(self._workers.values()), None)
                self._set_busy(bool(active), getattr(active, "kind", "edited"))
                self._set_status(
                    "status.analysis_invalid",
                    flags="invalid_analysis",
                    error=exc,
                )
                return
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
            active = next(iter(self._workers.values()), None)
            self._set_busy(bool(active), getattr(active, "kind", "edited"))
            self._set_status("status.analysis_changed")
            self._mark_manual_unreviewed()
            self._refresh_model_overlay()
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
            try:
                self._validate_analysis_controls()
            except ValueError as exc:
                generation = self._generation.next()
                self._pending_input_records.clear()
                if kind in {"preview", "optimize", "measure_geometry", "refine_geometry"}:
                    self._mark_manual_unreviewed()
                self._set_busy(False, "edited")
                self._set_status(
                    "status.analysis_invalid",
                    flags="invalid_analysis",
                    error=exc,
                )
                return generation
            generation = self._generation.next()
            self._pending_input_records.clear()
            if kind in {"preview", "optimize", "measure_geometry", "refine_geometry"}:
                # Once a new request starts, an older displayed result is no
                # longer eligible for review/export even if its image remains
                # visible until the new worker finishes.
                self._mark_manual_unreviewed()
                self._pending_input_records[generation] = deepcopy(
                    self._loaded_input_records
                )
            if kind == "optimize":
                # Freeze every editable field and the complete current input
                # context before handing work to QThreadPool.  The worker must
                # never observe a later table edit or a changed mask/ROI.
                parameter_snapshot = deepcopy(self.parameter_model.parameter_dict())
                self._fit_session["optimize_before"] = self._capture_fit_context(
                    parameters=parameter_snapshot,
                    include_arrays=False,
                )
                self._fit_session["optimize_after"] = None
                self._sync_fit_session_controls()
                request_payload = self._payload() if payload is None else payload
            else:
                parameter_snapshot = self.parameter_model.parameter_dict()
                request_payload = self._payload() if payload is None else payload
            cancel_event = (
                request_payload.get("cancel_event")
                if isinstance(request_payload, Mapping)
                else None
            )
            if cancel_event is None:
                cancel_event = threading.Event()
            self._cancel_events[generation] = cancel_event
            if isinstance(request_payload, Mapping):
                request_payload = dict(request_payload)
                request_payload["cancel_event"] = cancel_event
            worker = AnalysisWorker(
                _engine_job(self.engine),
                generation=generation,
                kind=kind,
                # Keep the complete editable state (bounds, vary, ties, unit,
                # stderr) in the worker request.  ``_engine_job`` supplies the
                # scalar compatibility view to legacy injected engines.
                parameters=parameter_snapshot,
                payload=request_payload,
            )
            if isinstance(request_payload, Mapping):
                request_payload["progress"] = worker.report_progress
            worker.signals.progress.connect(self._on_worker_progress)
            worker.signals.finished.connect(self._on_worker_finished)
            worker.signals.error.connect(self._on_worker_error)
            self._workers[generation] = worker
            self._set_busy(True, kind)
            self._thread_pool.start(worker)
            if kind == "preview":
                self.previewRequested.emit(generation)
            elif kind == "optimize":
                self.optimizeRequested.emit(generation)
            elif kind == "measure_geometry":
                self.geometryMeasureRequested.emit(generation)
            elif kind == "refine_geometry":
                self.geometryRefineRequested.emit(generation)
            return generation

        def request_preview(self) -> int:
            return self._start_job("preview")

        preview = request_preview

        def request_optimize(self) -> int:
            return self._start_job("optimize")

        optimize = request_optimize
        refine = request_optimize

        def request_geometry_measure(self) -> int:
            """Run ridge and measured-ellipse extraction as a geometry job."""

            return self._start_job("measure_geometry")

        def request_geometry_refine(self) -> int:
            """Run the constrained geometry refinement as a separate job."""

            return self._start_job("refine_geometry")

        remeasure_geometry = request_geometry_measure
        refine_geometry = request_geometry_refine

        def cancel_jobs(self) -> None:
            # QThreadPool cannot interrupt arbitrary user code.  Advancing the
            # generation makes late results harmless; the batch runner also
            # polls this event between frames for cooperative cancellation.
            if self._batch_cancel_event is not None:
                self._batch_cancel_event.set()
            for cancel_event in tuple(self._cancel_events.values()):
                cancel_event.set()
            self._generation.next()
            self._pending_input_records.clear()
            self._last_result_signature = None
            self._last_result_input_records = None
            self._sync_fit_session_controls()
            active = next(iter(self._workers.values()), None)
            self._set_busy(bool(active), getattr(active, "kind", "cancelled"))
            self._set_status("status.cancelled_late")

        def ignore_late_result(self) -> None:
            """Advance the generation gate while preserving the current view."""

            self._generation.next()
            self._pending_input_records.clear()
            self._last_result_signature = None
            self._last_result_input_records = None
            self._sync_fit_session_controls()
            active = next(iter(self._workers.values()), None)
            self._set_busy(bool(active), getattr(active, "kind", "ignored"))
            self._set_status("status.late_ignored")

        ignore_late_results = ignore_late_result

        def _on_worker_progress(
            self,
            generation: int,
            kind: str,
            payload: Any,
        ) -> None:
            """Apply structured worker progress on the GUI thread only."""

            if not self._generation.is_current(generation) or kind != "batch":
                return
            if not isinstance(payload, Mapping):
                return
            completed = int(payload.get("completed", payload.get("index", 0)) or 0)
            total = int(payload.get("total", len(self.batch_frames)) or 0)
            elapsed_s = float(payload.get("elapsed_s", 0.0) or 0.0)
            self._batch_progress_state = {
                "completed": completed,
                "total": total,
                "elapsed_s": elapsed_s,
            }
            self.batch_progress.setRange(0, max(1, total))
            self.batch_progress.setValue(min(max(0, completed), max(1, total)))
            self.batch_progress_label.setText(
                self._tr(
                    "progress.batch_running",
                    completed=completed,
                    total=total,
                    elapsed_s=elapsed_s,
                )
            )
            self._set_status(
                "status.batch_progress",
                completed=completed,
                total=total,
                elapsed_s=elapsed_s,
            )

        def _on_worker_finished(self, generation: int, kind: str, result: Any) -> None:
            self._workers.pop(generation, None)
            self._cancel_events.pop(generation, None)
            input_records = self._pending_input_records.pop(generation, None)
            if not self._generation.is_current(generation):
                if not self._workers:
                    self._set_busy(False, "cancelled")
                return
            self._last_error = None
            if kind == "batch":
                records = _result_value(result, ("records", "results", "evolution"), [])
                if records:
                    self.plot_evolution(records)
                    self._update_batch_rows(records)
                self._batch_cancel_event = None
            else:
                self._last_result = result
                self._last_result_kind = kind
                self._apply_result(result)
                self._last_result_signature = (
                    self._fit_state_signature()
                    if self._result_has_fit_images(result) and not _result_has_failure(result)
                    else None
                )
                if self._last_result_signature is None:
                    self._last_result_input_records = None
                else:
                    self._last_result_input_records = deepcopy(
                        input_records
                        if input_records is not None
                        else self._loaded_input_records
                    )
                if kind == "optimize":
                    self._auto_scale_initial = False
                    if _result_has_failure(result):
                        self._fit_session["optimize_after"] = None
                    else:
                        self._fit_session["optimize_after"] = self._capture_fit_context(
                            parameters=self.parameter_model.parameter_dict(),
                            # The worker already used the current frame.  Keep
                            # the review snapshot to selectors/settings and a
                            # scalar result summary; copying detector/q/mask
                            # arrays here would block the GUI thread.
                            include_arrays=False,
                        )
                        self._fit_session["optimize_after"]["result_summary"] = self._fit_result_summary(result)
                    # An optimization result is a candidate only.  It is
                    # deliberately left unreviewed until the user presses
                    # Accept current or Reject current.
                    self._fit_session["manual_status"] = "unreviewed"
                self._sync_fit_session_controls()
            if self._workers:
                active_kind = next(iter(self._workers.values())).kind
                self._set_busy(True, active_kind)
            else:
                self._set_busy(False, kind, result_ok=not _result_has_failure(result))

        def _on_worker_error(self, generation: int, kind: str, error: Exception) -> None:
            self._workers.pop(generation, None)
            self._cancel_events.pop(generation, None)
            self._pending_input_records.pop(generation, None)
            if not self._generation.is_current(generation):
                if not self._workers:
                    self._set_busy(False, "cancelled")
                return
            self._last_error = str(error)
            self._last_result_signature = None
            self._last_result_input_records = None
            self._last_result = None
            self._last_result_kind = None
            self._geometry_only_result = False
            self.views.clear_fit()
            if kind == "optimize":
                self._fit_session["optimize_after"] = None
                self._fit_session["manual_status"] = "unreviewed"
            self._sync_fit_session_controls()
            if self._workers:
                active_kind = next(iter(self._workers.values())).kind
                self._set_busy(True, active_kind)
            else:
                self._set_busy(False, kind)
            self._set_status(
                "status.job_error",
                flags="error",
                kind_key=f"job.{kind}",
                error=error,
            )

        @staticmethod
        def _overlay_branch_number(value: Any) -> int | None:
            try:
                branch = int(value)
            except (TypeError, ValueError):
                text = str(value or "").strip().lower()
                branch = {"ellipse_a": 0, "a": 0, "ellipse_b": 1, "b": 1}.get(text)
            return branch if branch in (0, 1) else None

        def _decorate_symmetry_overlay(
            self,
            result: Any,
            ridge_points: Any,
            ellipses: Any,
        ) -> tuple[Any, Any]:
            """Attach C3 reference-quadrant labels only when diagnostics certify them."""

            ellipse_result = _result_value(
                result,
                ("ellipse_fit", "ellipse", "ellipse_result"),
                None,
            )
            symmetry = _read(ellipse_result, ("symmetry",), None)
            paired_support = _read(symmetry, ("paired_support",), None)
            if not isinstance(paired_support, Mapping):
                return ridge_points, ellipses
            pair_by_branch: dict[int, str] = {}
            for raw_branch, payload in paired_support.items():
                branch = self._overlay_branch_number(raw_branch)
                pair = _read(payload, ("quadrant_pair",), None)
                if branch is None or pair in (None, ""):
                    continue
                pair_by_branch[branch] = str(pair)
            if not pair_by_branch:
                return ridge_points, ellipses
            branch_leaks = _read(symmetry, ("branch_leaks",), {})
            global_swap = bool(
                _read(symmetry, ("global_swap",), _read(branch_leaks, ("global_swap",), False))
            )

            def decorate(value: Any, default_branch: int | None = None) -> Any:
                if not isinstance(value, Mapping):
                    return value
                output = dict(value)
                raw_branch = _read(
                    value,
                    ("branch_id", "branch", "component"),
                    default_branch,
                )
                branch = self._overlay_branch_number(raw_branch)
                if branch is None:
                    return output
                effective_branch = branch ^ int(global_swap)
                pair = pair_by_branch.get(effective_branch)
                if pair is None:
                    return output
                output["overlay_branch_id"] = effective_branch
                output["quadrant_pair"] = pair
                return output

            if isinstance(ridge_points, Mapping):
                ridge_points = [ridge_points]
            decorated_ridges = [decorate(point) for point in (ridge_points or [])]
            if isinstance(ellipses, Mapping):
                ellipses = [ellipses]
            decorated_ellipses = [
                decorate(ellipse, index)
                for index, ellipse in enumerate(ellipses or ())
            ]
            return decorated_ridges, decorated_ellipses

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
            geometry_only = self._is_geometry_only_result(result)
            self._geometry_only_result = geometry_only
            observed = _result_value(result, ("observed", "data", "image"), self._observed)
            model = _result_value(result, ("model", "predicted", "fit", "intensity", "simulation"), None)
            residual = _result_value(result, ("residual", "difference", "resid"), None)
            if geometry_only:
                # ``measure_geometry`` delegates to a service measurement
                # path that may include a diagnostic intensity-shaped array.
                # It is not a full-pixel fit and must never be presented as a
                # Model/Residual candidate for human review.
                model = None
                residual = None
            if residual is None and observed is not None and model is not None and _np is not None:
                try:
                    obs_array, model_array = _np.asarray(observed), _np.asarray(model)
                    if obs_array.shape == model_array.shape:
                        residual = obs_array - model_array
                except Exception:
                    residual = None
            # A partial/failed result must not leave a stale image visible.
            if model is None:
                self.views.model.clear_image(
                    self._tr("view.model_unfitted") if geometry_only else None
                )
            if residual is None:
                self.views.residual.clear_image(
                    self._tr("view.residual_unavailable") if geometry_only else None
                )
            result_qx = _result_value(result, ("qx", "qx_nm_inv"), self._qx)
            result_qy = _result_value(result, ("qy", "qy_nm_inv"), self._qy)
            result_q_unit = self._active_q_unit(result)
            self._refresh_q_parameter_units(result_q_unit)
            result_valid_mask = _result_value(result, ("valid_mask",), _read(self._qmap, ("valid_mask", "valid"), None))
            result_external_mask = _result_value(result, ("mask", "external_mask"), self._external_mask)
            self.views.set_images(
                observed,
                model,
                residual,
                qx=result_qx,
                qy=result_qy,
                q_unit=result_q_unit,
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
            ridge_points, ellipses = self._decorate_symmetry_overlay(
                result,
                ridge_points,
                ellipses,
            )
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
            self.set_fit_overlay(ridge_points, ellipses)
            if self.focus_q_window_check.isChecked():
                self.focus_q_window()
            self._update_metrics(
                result,
                observed,
                residual,
                geometry_only=geometry_only,
            )
            self._update_measurements(result)

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
                raw_status = str(status)
                status_key = _BATCH_STATUS_KEYS.get(raw_status.casefold())
                status_item = QtWidgets.QTableWidgetItem(
                    self._tr(status_key) if status_key is not None else raw_status
                )
                status_item.setData(QtCore.Qt.ItemDataRole.UserRole, raw_status)
                self.batch_table.setItem(row_index, 1, status_item)
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
                for plot in (
                    self.angular_plot,
                    self.coverage_plot,
                    self.ridge_plot,
                    self.radial_profile_plot,
                ):
                    if plot.plotItem.legend is None:
                        plot.addLegend(offset=(8, 8))
                angle = _read(angular, ("angle_deg",), None)
                if angle is None:
                    raw_angle = _sequence(_read(angular, ("angle", "azimuth"), []))
                    angle = [math.degrees(float(item)) for item in raw_angle]
                intensity = _sequence(_read(angular, ("intensity", "profile"), []))
                coverage = _sequence(_read(angular, ("coverage",), []))
                try:
                    angle_values = [float(value) for value in _sequence(angle) if math.isfinite(float(value))]
                    intensity_values = [
                        float(value)
                        for value in intensity
                        if math.isfinite(float(value))
                    ]
                    coverage_values = [
                        float(value)
                        for value in coverage
                        if math.isfinite(float(value))
                    ]
                except (TypeError, ValueError):
                    angle_values, intensity_values, coverage_values = [], [], []
                if angle_values or intensity_values or coverage_values:
                    self._set_profile_summary(
                        self._tr(
                            "profile.summary",
                            count=max(len(angle_values), len(intensity_values), len(coverage_values)),
                            amin=_format_metric(min(angle_values)) if angle_values else "—",
                            amax=_format_metric(max(angle_values)) if angle_values else "—",
                            imin=_format_metric(min(intensity_values)) if intensity_values else "—",
                            imax=_format_metric(max(intensity_values)) if intensity_values else "—",
                            cmin=_format_metric(min(coverage_values)) if coverage_values else "—",
                            cmax=_format_metric(max(coverage_values)) if coverage_values else "—",
                        )
                    )
                else:
                    self._set_profile_summary(self._tr("profile.summary_empty"))
                try:
                    self.angular_plot.plot(
                        list(angle),
                        list(intensity),
                        pen=_pg.mkPen(50, 150, 255, width=2),
                        name=self._tr("legend.intensity"),
                    )
                except (TypeError, ValueError):
                    pass
                self.coverage_plot.clear()
                try:
                    if len(coverage):
                        self.coverage_plot.setYRange(0.0, 1.0, padding=0.02)
                        self.coverage_plot.plot(
                            list(angle),
                            list(coverage),
                            pen=_pg.mkPen(255, 190, 55, width=2),
                            symbol="o",
                            symbolSize=4,
                            symbolBrush=_pg.mkBrush(255, 190, 55, 180),
                            name=self._tr("legend.coverage"),
                        )
                except (TypeError, ValueError):
                    pass

            point_source = _read(ridge, ("points", "observed_points"), None)
            if point_source is None:
                point_source = _result_value(result, ("ridge_points", "ridges"), [])
            point_rows = _sequence(point_source)
            radial_profiles = _sequence(
                _read(
                    observables,
                    ("lobe_radial_profiles", "radial_profiles"),
                    _result_value(result, ("lobe_radial_profiles", "radial_profiles"), []),
                )
            )
            radial_peaks = _sequence(
                _read(
                    observables,
                    ("lobe_radial_peaks", "radial_peaks"),
                    _result_value(result, ("lobe_radial_peaks", "radial_peaks"), []),
                )
            )
            if self.radial_profile_plot is not None:
                self.radial_profile_plot.clear()
                profile_q_unit = "unknown"
                for profile_index, profile in enumerate(radial_profiles):
                    q_values = _sequence(_read(profile, ("q",), []))
                    intensities = _sequence(_read(profile, ("intensity", "profile"), []))
                    if not q_values or not intensities:
                        continue
                    profile_q_unit = str(_read(profile, ("q_unit",), profile_q_unit) or profile_q_unit)
                    try:
                        self.radial_profile_plot.plot(
                            [float(value) for value in q_values],
                            [float(value) for value in intensities],
                            pen=_pg.mkPen(
                                70 + (profile_index * 53) % 160,
                                170,
                                220 - (profile_index * 37) % 130,
                                width=1.5,
                            ),
                            name=self._tr("legend.lobe", index=profile_index + 1),
                        )
                    except (TypeError, ValueError):
                        continue
                self._ridge_plot_q_unit = profile_q_unit
                self.radial_profile_plot.setLabel(
                    "bottom",
                    self._tr("axis.radial_q", unit=profile_q_unit),
                )

            if self.ridge_plot is not None:
                self.ridge_plot.clear()
                accepted_x: list[float] = []
                accepted_y: list[float] = []
                rejected_x: list[float] = []
                rejected_y: list[float] = []
                q_unit = str(_read(ridge, ("q_unit",), "unknown") or "unknown")
                self._ridge_plot_q_unit = q_unit
                self.ridge_plot.setLabel(
                    "left",
                    self._tr("axis.ridge_q", unit=q_unit),
                )
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
                        name=self._tr("legend.accepted"),
                    )
                if rejected_x:
                    self.ridge_plot.plot(
                        rejected_x,
                        rejected_y,
                        pen=None,
                        symbol="x",
                        symbolSize=8,
                        symbolPen=_pg.mkPen(230, 90, 90, width=2),
                        name=self._tr("legend.rejected"),
                    )

            self.ridge_table.setRowCount(len(point_rows))
            for row_index, point in enumerate(point_rows):
                raw_angle = _read(point, ("angle_deg",), None)
                if raw_angle is None:
                    raw_angle = _read(point, ("angle", "azimuth", "phi"), None)
                    raw_angle = math.degrees(float(raw_angle)) if raw_angle is not None else None
                accepted = bool(_read(point, ("accepted", "valid"), True))
                method_value = str(_read(point, ("method",), "observed"))
                branch_value = _read(point, ("branch_id", "branch"), None)
                quadrant_value = _read(point, ("quadrant_pair", "quadrant"), None)
                if branch_value is not None or quadrant_value is not None:
                    method_value += (
                        f"; branch={branch_value if branch_value is not None else '—'}"
                        f"; quadrant={quadrant_value if quadrant_value is not None else '—'}"
                    )
                values = (
                    _format_metric(raw_angle),
                    _format_metric(_read(point, ("q", "q_star", "q_position"), None)),
                    accepted,
                    method_value,
                )
                for column, value in enumerate(values):
                    item = (
                        self._boolean_table_item(value)
                        if column == 2
                        else QtWidgets.QTableWidgetItem(str(value))
                    )
                    self.ridge_table.setItem(row_index, column, item)

            self.lobe_table.setRowCount(len(lobes))
            for row_index, lobe in enumerate(lobes):
                angle = _read(lobe, ("angle_deg",), None)
                if angle is None:
                    raw_angle = _read(lobe, ("angle", "azimuth"), None)
                    angle = math.degrees(float(raw_angle)) if raw_angle is not None else None
                valid = bool(_read(lobe, ("valid", "accepted"), True))
                values = (
                    _format_metric(angle),
                    _format_metric(_read(lobe, ("intensity",), None)),
                    _format_metric(_read(lobe, ("baseline",), None)),
                    _format_metric(_read(lobe, ("snr",), None)),
                    _format_metric(_read(lobe, ("fwhm_deg",), None)),
                    _format_metric(_read(lobe, ("coverage",), None)),
                    valid,
                    ", ".join(str(item) for item in _sequence(_read(lobe, ("flags",), ()))),
                )
                for column, value in enumerate(values):
                    item = (
                        self._boolean_table_item(value)
                        if column == 6
                        else QtWidgets.QTableWidgetItem(str(value))
                    )
                    self.lobe_table.setItem(row_index, column, item)

            self.radial_table.setRowCount(len(radial_peaks))
            for row_index, peak in enumerate(radial_peaks):
                raw_angle = _read(peak, ("angle_deg",), None)
                if raw_angle is None:
                    raw_angle = _read(peak, ("angle", "azimuth"), None)
                    raw_angle = (
                        math.degrees(float(raw_angle))
                        if raw_angle is not None
                        else None
                    )
                raw_q_star = _read(peak, ("q_star", "q"), None)
                values = (
                    _format_metric(raw_angle),
                    _format_metric(raw_q_star),
                    _format_metric(
                        _read(peak, ("lamellar_spacing", "spacing", "Ln"), None)
                    ),
                    _format_metric(_read(peak, ("snr",), None)),
                    _format_metric(_read(peak, ("radial_fwhm", "fwhm"), None)),
                    _format_metric(_read(peak, ("coverage",), None)),
                    bool(_read(peak, ("valid", "accepted"), True)),
                    ", ".join(
                        str(item)
                        for item in _sequence(_read(peak, ("flags",), ()))
                    ),
                )
                for column, value in enumerate(values):
                    item = (
                        self._boolean_table_item(value)
                        if column == 6
                        else QtWidgets.QTableWidgetItem(str(value))
                    )
                    self.radial_table.setItem(row_index, column, item)

            def compact_summary(value: Any) -> str:
                if value is None:
                    return "—"
                if isinstance(value, Mapping):
                    return "; ".join(
                        f"{key}={compact_summary(item)}"
                        for key, item in value.items()
                    ) or "—"
                if isinstance(value, (list, tuple)):
                    return "[" + ", ".join(compact_summary(item) for item in value) + "]"
                return str(value)

            quality_payload = _read(ellipse, ("quality",), {})
            quality_metrics = _read(quality_payload, ("metrics",), {})
            symmetry_payload = _read(ellipse, ("symmetry",), None)
            if not isinstance(symmetry_payload, Mapping):
                symmetry_payload = _read(quality_metrics, ("symmetry",), None)
            symmetry_summary = None
            if isinstance(symmetry_payload, Mapping):
                symmetry_status = _read(
                    symmetry_payload,
                    ("symmetry_status", "status"),
                    _read(quality_metrics, ("symmetry_status",), "unknown"),
                )
                reference_axis = _read(
                    symmetry_payload,
                    ("reference_axis_deg",),
                    _read(ellipse, ("reference_axis_deg",), None),
                )
                quadrant_counts = _read(
                    symmetry_payload,
                    ("quadrant_counts", "quadrant_count"),
                    None,
                )
                branch_quadrant_counts = _read(
                    symmetry_payload,
                    ("branch_quadrant_counts",),
                    None,
                )
                paired_support = _read(symmetry_payload, ("paired_support",), None)
                branch_leaks = _read(
                    symmetry_payload,
                    ("branch_leaks", "leaks"),
                    None,
                )
                unassigned = _read(
                    symmetry_payload,
                    ("unassigned_count", "unassigned"),
                    None,
                )
                symmetry_summary = self._tr(
                    "profile.symmetry_summary",
                    status=compact_summary(symmetry_status),
                    axis=compact_summary(reference_axis),
                    quadrants=(
                        f"{compact_summary(quadrant_counts)}; "
                        f"branches={compact_summary(branch_quadrant_counts)}"
                    ),
                    support=compact_summary(paired_support),
                    leaks=compact_summary(branch_leaks),
                    unassigned=compact_summary(unassigned),
                )
                self._set_profile_summary(
                    f"{self.profile_summary_label.text()}\n{symmetry_summary}"
                )
            quality_flags = _sequence(_read(quality_payload, ("flags",), ()))
            ellipse_flags = _sequence(_read(ellipse, ("flags",), ()))
            p4_flags = list(dict.fromkeys(str(item) for item in (*quality_flags, *ellipse_flags)))
            ellipse_rows = (
                ("ellipse.a", _read(ellipse, ("a",), None)),
                ("ellipse.b", _read(ellipse, ("b",), None)),
                ("ellipse.axis_ratio", _read(ellipse, ("axis_ratio", "axes_ratio"), None)),
                ("ellipse.ellipticity", _read(ellipse, ("ellipticity", "eccentricity"), None)),
                # Keep ellipse theta semantically separate from lobe-derived
                # phi/alpha/psi; no relabelling is performed here.
                ("ellipse.theta", _read(ellipse, ("theta_deg", "angle_deg"), None)),
                ("ellipse.ln", _read(ellipse, ("Ln_from_minor_axis_nm",), None)),
                ("ellipse.lz", _read(ellipse, ("Lz_from_draw_axis_nm",), None)),
                ("ellipse.rmse", _read(ellipse, ("rmse", "residual_rms"), None)),
                ("ellipse.rss", _read(ellipse, ("rss",), None)),
                ("ellipse.n_points", _read(ellipse, ("n_points", "n_data"), None)),
                ("ellipse.quality", _read(ellipse, ("success",), None)),
                ("ellipse.flags", ", ".join(str(item) for item in _sequence(_read(ellipse, ("flags",), ())))),
                ("ellipse.phi_app", _read(observables, ("phi_app_deg",), None)),
                ("ellipse.alpha_candidate", _read(observables, ("alpha_candidate_deg",), None)),
                ("ellipse.psi_candidate", _read(observables, ("psi_candidate_deg",), None)),
                (
                    "ellipse.stderr",
                    ", ".join(
                        f"{key}={_format_metric(value)}"
                        for key, value in (
                            _read(ellipse, ("stderr",), {}) or {}
                        ).items()
                    ),
                ),
                (
                    "ellipse.bound_flags",
                    ", ".join(
                        f"{key}={bool(value)}"
                        for key, value in (
                            _read(ellipse, ("bound_flags", "bound_status"), {}) or {}
                        ).items()
                    ),
                ),
                (
                    "ellipse.quality_status",
                    _read(
                        ellipse,
                        ("quality_status", "status"),
                        _read(quality_payload, ("status",), None),
                    ),
                ),
                (
                    "ellipse.p4_flags",
                    ", ".join(p4_flags)
                    or ", ".join(
                        str(item)
                        for item in _sequence(_read(observables, ("flags",), ()))
                    ),
                ),
                (
                    "ellipse.q_unit",
                    _read(
                        ellipse,
                        ("q_unit",),
                        _read(observables, ("q_unit",), "unknown"),
                    ),
                ),
                (
                    "ellipse.geometry_action",
                    _result_value(result, ("geometry_action",), None),
                ),
                (
                    "ellipse.symmetry_status",
                    _read(symmetry_payload, ("symmetry_status", "status"), None),
                ),
                (
                    "ellipse.reference_axis",
                    _read(symmetry_payload, ("reference_axis_deg",), None),
                ),
                (
                    "ellipse.quadrant_counts",
                    (
                        f"{compact_summary(_read(symmetry_payload, ('quadrant_counts', 'quadrant_count'), None))}; "
                        f"branches={compact_summary(_read(symmetry_payload, ('branch_quadrant_counts',), None))}"
                    )
                    if isinstance(symmetry_payload, Mapping)
                    else None,
                ),
                (
                    "ellipse.paired_support",
                    compact_summary(_read(symmetry_payload, ("paired_support",), None))
                    if isinstance(symmetry_payload, Mapping)
                    else None,
                ),
                (
                    "ellipse.branch_leaks",
                    compact_summary(
                        {
                            "leaks": _read(symmetry_payload, ("branch_leaks", "leaks"), None),
                            "unassigned": _read(
                                symmetry_payload,
                                ("unassigned_count", "unassigned"),
                                None,
                            ),
                        }
                    )
                    if isinstance(symmetry_payload, Mapping)
                    else None,
                ),
            )
            self.ellipse_table.setRowCount(len(ellipse_rows) if ellipse is not None or observables is not None else 0)
            for row_index, (key, value) in enumerate(ellipse_rows if ellipse is not None or observables is not None else ()):
                self.ellipse_table.setItem(row_index, 0, QtWidgets.QTableWidgetItem(self._tr(key)))
                value_item = (
                    self._boolean_table_item(value)
                    if key == "ellipse.quality" and value is not None
                    else QtWidgets.QTableWidgetItem(_format_metric(value))
                )
                self.ellipse_table.setItem(row_index, 1, value_item)

        def _update_metrics(
            self,
            result: Any,
            observed: Any,
            residual: Any,
            *,
            geometry_only: bool = False,
        ) -> None:
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
            geometry_quality = _result_value(
                ellipse_result,
                ("quality_status", "status"),
                _read(_result_value(ellipse_result, ("quality",), {}), ("status",), None),
            )
            if geometry_only:
                # Geometry refinement has no valid image residual.  Surface
                # the ellipse fit residual and point support instead of the
                # service's diagnostic/full-pixel placeholders.
                rmse = _result_value(
                    ellipse_result,
                    ("rmse", "residual_rms"),
                    _result_value(
                        metrics,
                        ("geometry_rmse",),
                        _result_value(result, ("geometry_rmse",), None),
                    ),
                )
                ndata = _result_value(
                    ellipse_result,
                    ("n_points", "n_data"),
                    ndata,
                )
                coverage_payload = _result_value(
                    ellipse_result,
                    ("coverage",),
                    {},
                )
                coverage = _read(
                    coverage_payload,
                    ("angular_coverage", "angular_coverage_fraction"),
                    _result_value(
                        metrics,
                        ("geometry_coverage", "valid_fraction", "coverage", "valid_coverage"),
                        None,
                    ),
                )
            else:
                coverage = _result_value(metrics, ("valid_fraction", "coverage", "valid_coverage"), None)
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
            if geometry_only:
                if isinstance(flags, str):
                    flags = [flags]
                flags = list(flags or ())
                q_unit = self._active_q_unit(result)
                flags.extend((
                    "geometry_only",
                    "intensity_fit_not_run",
                    f"q_unit={q_unit}",
                ))
                if geometry_quality:
                    flags.append(f"quality={geometry_quality}")
            if isinstance(flags, str):
                flags_text = flags
            else:
                flags_text = ", ".join(str(item) for item in (flags or [])) or "—"
            self.last_metrics = {
                "rmse": rmse,
                "ndata": ndata,
                "flags": flags,
                "valid_fraction": coverage,
                "geometry_only": geometry_only,
                "q_unit": self._active_q_unit(result) if geometry_only else None,
                "quality_status": geometry_quality if geometry_only else None,
            }
            self._metric_display["rmse"] = _format_metric(rmse)
            self._metric_display["ndata"] = ndata if ndata is not None else "—"
            self._displayed_flags_text = flags_text
            try:
                coverage_text = _format_metric(float(coverage) * 100.0) + "%" if coverage is not None else "—"
            except (TypeError, ValueError):
                coverage_text = "—"
            self._metric_display["coverage"] = coverage_text
            self._render_metric_labels()

        # ----- project persistence ---------------------------------------------

        def _fit_session_for_project(self) -> dict[str, Any]:
            """Return fit-session JSON without detector-sized in-memory data."""

            persisted = _new_fit_session()
            for key in ("manual_status", "reviewed_by", "reviewed_at", "review_notes"):
                persisted[key] = deepcopy(self._fit_session.get(key, persisted[key]))
            for key in ("optimize_before", "optimize_after"):
                persisted[key] = self._fit_context_for_project(self._fit_session.get(key))
            accepted = self._fit_session.get("accepted_parameters")
            persisted["accepted_parameters"] = deepcopy(accepted) if isinstance(accepted, Mapping) else None
            snapshots = self._fit_session.get("snapshots", [])
            if isinstance(snapshots, list):
                for index, snapshot in enumerate(snapshots):
                    if not isinstance(snapshot, Mapping) or not isinstance(snapshot.get("parameters"), Mapping):
                        continue
                    item = {
                        "order": index + 1,
                        "note": str(snapshot.get("note", "") or ""),
                        "created_at": snapshot.get("created_at"),
                        "parameters": deepcopy(snapshot["parameters"]),
                    }
                    if isinstance(snapshot.get("context"), Mapping):
                        item["context"] = self._fit_context_for_project(snapshot["context"])
                    persisted["snapshots"].append(item)
            return _jsonable(persisted)

        def project_to_dict(self) -> dict[str, Any]:
            return {
                "schema_version": 2,
                "parameters": self.parameter_model.parameter_dict(),
                "analysis": _jsonable(self.analysis_settings),
                "display": _jsonable(self.display_settings),
                "input": self._source_path,
                "poni": self._poni_path,
                "frame": self._frame,
                "dataset": self._dataset,
                "mask": self._mask_path,
                "mask_frame": self._mask_frame,
                "mask_dataset": self._mask_dataset,
                "roi_exclusion": _jsonable(self._exclusion_roi),
                "rois": list(self._roi_specs),
                "fit_session": self._fit_session_for_project(),
                "batch": {
                    "mode": self.batch_mode_combo.currentData(),
                    "stage": self.batch_stage_combo.currentData(),
                    "full2d": self.batch_stage_combo.currentData() == "full2d",
                    "stream": self.batch_stream_check.isChecked(),
                    "frames": _jsonable(list(self.batch_frames)),
                    "manifest": self.batch_manifest_edit.text() or None,
                    "checkpoint": self.batch_checkpoint_edit.text() or None,
                    "resume": self.batch_resume_check.isChecked(),
                    "series": self.batch_series_edit.text().strip() or None,
                    "start": None if self.batch_start_spin.value() < 0 else self.batch_start_spin.value(),
                    "stop": None if self.batch_stop_spin.value() < 0 else self.batch_stop_spin.value(),
                    "stride": self.batch_stride_spin.value(),
                    "range": self.batch_range_edit.text().strip() or None,
                    "output": self.batch_output_edit.text() or None,
                },
                "metadata": {
                    "project_path": str(self._project_path) if self._project_path else None,
                    "config_path": self._config_path,
                    "qmap_shape": list(getattr(self._qmap, "shape", ())) if self._qmap is not None else None,
                },
            }

        @staticmethod
        def _snapshot_project_value(value: Any) -> Any:
            """Copy project metadata while retaining detector-array ownership."""

            if _np is not None and isinstance(value, _np.ndarray):
                return value
            if isinstance(value, Mapping):
                return {
                    key: RefinementMainWindow._snapshot_project_value(item)
                    for key, item in value.items()
                }
            if isinstance(value, list):
                return [RefinementMainWindow._snapshot_project_value(item) for item in value]
            if isinstance(value, tuple):
                return tuple(RefinementMainWindow._snapshot_project_value(item) for item in value)
            if isinstance(value, set):
                return {RefinementMainWindow._snapshot_project_value(item) for item in value}
            try:
                return deepcopy(value)
            except Exception:
                return value

        def _snapshot_engine_document(self) -> dict[str, Any]:
            """Keep service state references so a failed load can be undone."""

            snapshot: dict[str, Any] = {}
            for name in (
                "_poni",
                "_loaded",
                "_qmap",
                "poni_path",
                "_analysis_settings",
                "_parameter_specs",
            ):
                try:
                    if hasattr(self.engine, name):
                        snapshot[name] = getattr(self.engine, name)
                except Exception:
                    continue
            try:
                cache = getattr(self.engine, "_geometry_cache")
            except Exception:
                cache = None
            if isinstance(cache, Mapping):
                snapshot["_geometry_cache"] = dict(cache)
            return snapshot

        def _restore_engine_document(self, snapshot: Mapping[str, Any]) -> None:
            for name, value in snapshot.items():
                try:
                    setattr(self.engine, name, value)
                except Exception:
                    # Lightweight injected engines may expose only the public
                    # preview seam; their absence of private document state
                    # must not hide the original project-load error.
                    continue

        def _snapshot_project_document(self) -> dict[str, Any]:
            """Capture document references before a potentially failing load."""

            direct_fields = (
                "_source_path",
                "_frame",
                "_dataset",
                "_mask_frame",
                "_mask_dataset",
                "_poni_path",
                "_mask_path",
                "_observed",
                "_qx",
                "_qy",
                "_qmap",
                "_file_mask",
                "_external_mask",
                "_last_result",
                "_last_result_signature",
                "_last_result_kind",
                "_geometry_only_result",
                "_last_error",
                "_display_scale",
                "_display_percentile",
                "_project_path",
                "_config_path",
                "_auto_scale_initial",
                "_batch_cancel_event",
                "measurement_observables",
                "evolution_y_key",
            )
            copied_fields = (
                "_exclusion_roi",
                "_roi_specs",
                "_last_result_input_records",
                "_pending_input_records",
                "_loaded_input_records",
                "_last_evidence_paths",
                "_fit_ridge_points",
                "_rejected_ridge_points",
                "_observed_fit_ellipses",
                "_model_ellipses",
                "last_metrics",
                "_analysis_settings",
                "evolution_records",
                "_evolution_rows",
                "batch_frames",
                "_fit_session",
                "_metric_display",
                "_status_values",
            )
            snapshot = {
                name: getattr(self, name)
                for name in direct_fields
                if hasattr(self, name)
            }
            snapshot.update(
                {
                    name: self._snapshot_project_value(getattr(self, name))
                    for name in copied_fields
                    if hasattr(self, name)
                }
            )
            snapshot["_displayed_flags_text"] = self._displayed_flags_text
            snapshot["_status_key"] = self._status_key
            snapshot["_fit_session_restore_active"] = self._fit_session_restore_active
            snapshot["ui_parameter_values"] = self._snapshot_project_value(
                self.parameter_model.parameter_dict()
            )
            snapshot["ui_analysis_settings"] = self._snapshot_project_value(
                self.analysis_settings
            )
            snapshot["ui_display_settings"] = self._snapshot_project_value(
                self.display_settings
            )
            snapshot["ui_batch"] = {
                "mode": self.batch_mode_combo.currentData(),
                "stage": self.batch_stage_combo.currentData(),
                "stream": self.batch_stream_check.isChecked(),
                "manifest": self.batch_manifest_edit.text(),
                "checkpoint": self.batch_checkpoint_edit.text(),
                "output": self.batch_output_edit.text(),
                "series": self.batch_series_edit.text(),
                "range": self.batch_range_edit.text(),
                "resume": self.batch_resume_check.isChecked(),
                "start": self.batch_start_spin.value(),
                "stop": self.batch_stop_spin.value(),
                "stride": self.batch_stride_spin.value(),
                "frames": list(self.batch_frames),
            }
            snapshot["_view_active_q_window"] = getattr(
                self.views,
                "_active_q_window",
                None,
            )
            snapshot["engine"] = self._snapshot_engine_document()
            return snapshot

        def _restore_project_document(self, snapshot: Mapping[str, Any]) -> None:
            parameter_values = snapshot.get("ui_parameter_values")
            if isinstance(parameter_values, Mapping):
                self.parameter_model.set_rows(parameter_values)
            analysis_settings = snapshot.get("ui_analysis_settings")
            if isinstance(analysis_settings, Mapping):
                self.set_analysis_settings(analysis_settings, trigger_preview=False)
            display_settings = snapshot.get("ui_display_settings")
            if isinstance(display_settings, Mapping):
                self.set_display_settings(display_settings)
            batch_state = snapshot.get("ui_batch")
            if isinstance(batch_state, Mapping):
                mode_index = self.batch_mode_combo.findData(batch_state.get("mode"))
                if mode_index >= 0:
                    self.batch_mode_combo.setCurrentIndex(mode_index)
                stage_index = self.batch_stage_combo.findData(batch_state.get("stage"))
                if stage_index >= 0:
                    self.batch_stage_combo.setCurrentIndex(stage_index)
                self.batch_stream_check.setChecked(bool(batch_state.get("stream")))
                for widget, key in (
                    (self.batch_manifest_edit, "manifest"),
                    (self.batch_checkpoint_edit, "checkpoint"),
                    (self.batch_output_edit, "output"),
                    (self.batch_series_edit, "series"),
                    (self.batch_range_edit, "range"),
                ):
                    widget.setText(str(batch_state.get(key) or ""))
                self.batch_resume_check.setChecked(bool(batch_state.get("resume")))
                self.batch_start_spin.setValue(int(batch_state.get("start", -1)))
                self.batch_stop_spin.setValue(int(batch_state.get("stop", -1)))
                self.batch_stride_spin.setValue(int(batch_state.get("stride", 1)))
                self.set_batch_frames(batch_state.get("frames", ()))
            direct_fields = (
                "_source_path",
                "_frame",
                "_dataset",
                "_mask_frame",
                "_mask_dataset",
                "_poni_path",
                "_mask_path",
                "_observed",
                "_qx",
                "_qy",
                "_qmap",
                "_file_mask",
                "_external_mask",
                "_last_result",
                "_last_result_signature",
                "_last_result_kind",
                "_geometry_only_result",
                "_last_error",
                "_display_scale",
                "_display_percentile",
                "_project_path",
                "_config_path",
                "_auto_scale_initial",
                "_batch_cancel_event",
                "measurement_observables",
                "evolution_y_key",
            )
            copied_fields = (
                "_exclusion_roi",
                "_roi_specs",
                "_last_result_input_records",
                "_pending_input_records",
                "_loaded_input_records",
                "_last_evidence_paths",
                "_fit_ridge_points",
                "_rejected_ridge_points",
                "_observed_fit_ellipses",
                "_model_ellipses",
                "last_metrics",
                "_analysis_settings",
                "evolution_records",
                "_evolution_rows",
                "batch_frames",
                "_fit_session",
                "_metric_display",
                "_status_values",
            )
            for name in direct_fields:
                if name in snapshot:
                    setattr(self, name, snapshot[name])
            for name in copied_fields:
                if name in snapshot:
                    setattr(self, name, self._snapshot_project_value(snapshot[name]))
            self._displayed_flags_text = str(
                snapshot.get("_displayed_flags_text", self._displayed_flags_text)
            )
            self._status_key = str(snapshot.get("_status_key", self._status_key))
            self._fit_session_restore_active = bool(
                snapshot.get(
                    "_fit_session_restore_active",
                    self._fit_session_restore_active,
                )
            )
            self._restore_engine_document(snapshot.get("engine", {}))

            # Rebuild the visible views from the restored references.  This
            # keeps rollback zero-copy for detector arrays while removing any
            # model/residual/ROI paint left by the rejected project.
            self.views.clear_fit()
            result = self._last_result
            observed = _result_value(result, ("observed", "data", "image"), self._observed)
            model = _result_value(
                result,
                ("model", "predicted", "fit", "intensity", "simulation"),
                None,
            )
            residual = _result_value(result, ("residual", "difference", "resid"), None)
            if self._geometry_only_result:
                model = None
                residual = None
            if residual is None and observed is not None and model is not None and _np is not None:
                try:
                    observed_array = _np.asarray(observed)
                    model_array = _np.asarray(model)
                    if observed_array.shape == model_array.shape:
                        residual = observed_array - model_array
                except (TypeError, ValueError):
                    residual = None
            q_unit = self._active_q_unit(result)
            self._refresh_q_parameter_units(q_unit)
            self.views.set_images(
                observed,
                model,
                residual,
                qx=_result_value(result, ("qx", "qx_nm_inv"), self._qx),
                qy=_result_value(result, ("qy", "qy_nm_inv"), self._qy),
                q_unit=q_unit,
                valid_mask=_result_value(
                    result,
                    ("valid_mask",),
                    _read(self._qmap, ("valid_mask", "valid"), None),
                ),
                external_mask=_result_value(
                    result,
                    ("mask", "external_mask"),
                    self._external_mask,
                ),
            )
            self.views.set_roi(self._roi_specs)
            self.views.set_overlay(
                self._fit_ridge_points,
                self._observed_fit_ellipses,
                model_ellipses=self._model_ellipses,
                rejected_ridge_points=self._rejected_ridge_points,
            )
            self.views.set_display_settings(
                self._display_scale,
                self._display_percentile,
            )
            q_window = snapshot.get("_view_active_q_window")
            if q_window is None:
                self.views.set_q_view(full=True)
            else:
                self.views.set_q_view(q_window)
            self._sync_roi_widgets()
            self._sync_fit_session_controls(preserve_edits=True)
            self._render_metric_labels()
            self._render_status()

        def _apply_project_document(self, data: Mapping[str, Any], target: Path) -> None:
            """Apply an already parsed/normalized document through the UI seam."""

            project_base = target.parent
            source = data.get("input", data.get("input_path"))
            poni = data.get("poni", data.get("poni_path"))
            mask = data.get("mask", data.get("mask_path"))
            self._invalidate_pending_work(clear_fit=False)
            self._auto_scale_initial = False
            self.parameter_model.set_rows(data.get("parameters", {}))
            analysis = data.get("analysis", data.get("measurement", data.get("analysis_settings", {})))
            if isinstance(analysis, Mapping):
                self.set_analysis_settings(analysis, trigger_preview=False)
            display = data.get("display")
            if isinstance(display, Mapping):
                self.set_display_settings(display)
            frame = data.get("frame")
            dataset = data.get("dataset")
            mask_frame = data.get("mask_frame")
            mask_dataset = data.get("mask_dataset")
            selected_frame = int(frame) if frame is not None else None
            selected_dataset = str(dataset) if dataset is not None else None
            selected_mask_frame = int(mask_frame) if mask_frame is not None else None
            selected_mask_dataset = str(mask_dataset) if mask_dataset is not None else None
            roi = data.get("roi_exclusion")
            rois = data.get("rois")
            batch = data.get("batch")
            if isinstance(batch, Mapping):
                mode_index = self.batch_mode_combo.findData(batch.get("mode", "independent"))
                if mode_index >= 0:
                    self.batch_mode_combo.setCurrentIndex(mode_index)
                stage_value = batch.get("stage")
                if stage_value is None and "full2d" in batch:
                    stage_value = "full2d" if bool(batch.get("full2d")) else "geometry"
                if stage_value is not None:
                    stage_index = self.batch_stage_combo.findData(stage_value)
                    if stage_index >= 0:
                        self.batch_stage_combo.setCurrentIndex(stage_index)
                self.batch_stream_check.setChecked(bool(batch.get("stream", False)))
                for widget, key in (
                    (self.batch_manifest_edit, "manifest"),
                    (self.batch_checkpoint_edit, "checkpoint"),
                    (self.batch_output_edit, "output"),
                ):
                    value = _resolve_project_path(batch.get(key), project_base)
                    widget.setText("" if value is None else str(value))
                self.batch_series_edit.setText(str(batch.get("series") or ""))
                self.batch_range_edit.setText(str(batch.get("range") or ""))
                self.batch_resume_check.setChecked(bool(batch.get("resume", False)))
                self.batch_start_spin.setValue(int(batch.get("start", -1) if batch.get("start") is not None else -1))
                self.batch_stop_spin.setValue(int(batch.get("stop", -1) if batch.get("stop") is not None else -1))
                self.batch_stride_spin.setValue(max(1, int(batch.get("stride", 1))))
                frames = batch.get("frames", data.get("batch_frames", []))
                self.set_batch_frames(
                    _resolve_project_frame(item, project_base) for item in (frames or [])
                )
            if source:
                if not self.open_image(
                    source,
                    frame=selected_frame,
                    dataset=selected_dataset,
                    poni=poni,
                    external_mask=mask,
                    mask_frame=selected_mask_frame,
                    mask_dataset=selected_mask_dataset,
                ):
                    raise ValueError(f"could not load project input: {source}")
            elif mask:
                if not self.select_mask(
                    mask,
                    mask_frame=selected_mask_frame,
                    mask_dataset=selected_mask_dataset,
                ):
                    raise ValueError(f"could not load project mask: {mask}")
            if mask and self._file_mask is None:
                self._load_external_mask(
                    mask,
                    frame=selected_mask_frame,
                    dataset=selected_mask_dataset,
                )
            if rois and isinstance(rois, Iterable):
                self._roi_specs = [dict(spec) for spec in rois if isinstance(spec, Mapping)]
                self._exclusion_roi = self._roi_specs[-1] if self._roi_specs else None
                if self._roi_specs and not self._recompute_external_mask(update_widgets=True):
                    raise ValueError("could not apply project ROIs")
            elif roi is not None:
                if not self.set_exclusion_roi(roi):
                    raise ValueError("could not apply project ROI")
            if not source and poni and not self.set_poni(poni):
                raise ValueError(f"could not load project PONI: {poni}")
            if not source:
                # A parameter/mask-only project has no open_image call to
                # commit its selectors, so commit them only after every
                # dependent operation above has succeeded.
                self._frame = selected_frame
                self._dataset = selected_dataset
                self._mask_frame = selected_mask_frame
                self._mask_dataset = selected_mask_dataset
            setter = getattr(self.engine, "set_parameters", None)
            if callable(setter):
                setter(self.parameter_model.parameter_dict())
            self._fit_session_restore_active = True
            try:
                self._fit_session = self._normalise_fit_session(data.get("fit_session"))
            finally:
                self._fit_session_restore_active = False
            self._sync_fit_session_controls()

        def save_project(self, path: str | Path | bool | None = None) -> bool:
            if isinstance(path, bool):
                path = None
            if path is None:
                chosen, _ = QtWidgets.QFileDialog.getSaveFileName(
                    self,
                    self._tr("dialog.save_project"),
                    "",
                    self._tr("filter.project"),
                )
                if not chosen:
                    return False
                path = chosen
            try:
                target = self._project_controller.save(path, self.project_to_dict())
            except (OSError, TypeError, ValueError) as exc:
                self._set_status("status.save_failed", flags="error", error=exc)
                return False
            self._project_path = target
            self._set_status("status.saved", name=target.name)
            return True

        def load_project(self, path: str | Path | bool | None = None) -> bool:
            if isinstance(path, bool):
                path = None
            if path is None:
                chosen, _ = QtWidgets.QFileDialog.getOpenFileName(
                    self,
                    self._tr("dialog.open_project"),
                    "",
                    self._tr("filter.project"),
                )
                if not chosen:
                    return False
                path = chosen
            try:
                target = self._project_controller.load(path)
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                self._set_status("status.load_failed", flags="error", error=exc)
                return False
            self._project_path = target
            self._set_status("status.loaded", name=target.name)
            return True

        open_project = load_project

        # ----- batch and evolution ---------------------------------------------

        def _choose_batch_files(self) -> None:
            files, _ = QtWidgets.QFileDialog.getOpenFileNames(
                self,
                self._tr("dialog.select_frames"),
                "",
                self._tr(
                    "filter.images_batch",
                    all_files=self._tr("filter.all_files"),
                ),
            )
            if files:
                self.set_batch_frames(files)

        def set_batch_frames(self, frames: Iterable[Any]) -> None:
            self.batch_frames = list(frames)
            self.batch_table.setRowCount(len(self.batch_frames))
            for row, frame in enumerate(self.batch_frames):
                self.batch_table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(frame)))
                status_item = QtWidgets.QTableWidgetItem(self._tr("status.ready"))
                status_item.setData(QtCore.Qt.ItemDataRole.UserRole, "ready")
                self.batch_table.setItem(row, 1, status_item)
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
                "stage": self.batch_stage_combo.currentData(),
                "full2d": self.batch_stage_combo.currentData() == "full2d",
                "stream": bool(self.batch_stream_check.isChecked()),
                "manifest": self.batch_manifest_edit.text() or None,
                "checkpoint": self.batch_checkpoint_edit.text() or None,
                "resume": self.batch_resume_check.isChecked(),
                "series": self.batch_series_edit.text().strip() or None,
                "start": None if self.batch_start_spin.value() < 0 else self.batch_start_spin.value(),
                "stop": None if self.batch_stop_spin.value() < 0 else self.batch_stop_spin.value(),
                "stride": self.batch_stride_spin.value(),
                "frame_range": self.batch_range_edit.text().strip() or None,
                "output": self.batch_output_edit.text() or None,
                "source": self._source_path,
                "poni": self._poni_path,
            }
            self._batch_cancel_event = threading.Event()
            payload["cancel_event"] = self._batch_cancel_event
            self._batch_progress_state = {
                "completed": 0,
                "total": len(self.batch_frames),
                "elapsed_s": 0.0,
            }
            self.batch_progress.setRange(0, max(1, len(self.batch_frames)))
            self.batch_progress.setValue(0)
            self.batch_progress_label.setText(
                self._tr(
                    "progress.batch_running",
                    completed=0,
                    total=len(self.batch_frames),
                    elapsed_s=0.0,
                )
            )
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
            self._refresh_evolution_item_tooltips()
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
            self.evolution_plot.setLabel(
                "left",
                self.evolution_y_key or self._tr("axis.value"),
            )
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

        def _set_busy(self, busy: bool, kind: str = "", *, result_ok: bool | None = None) -> None:
            self.preview_button.setEnabled(not busy)
            self.optimize_button.setEnabled(not busy)
            self.measure_geometry_button.setEnabled(not busy)
            self.refine_geometry_button.setEnabled(not busy)
            self.batch_run_button.setEnabled(not busy)
            self.cancel_button.setEnabled(bool(busy))
            self.ignore_late_result_button.setEnabled(bool(busy))
            self.batch_progress.setVisible(busy and kind == "batch")
            self.batch_progress_label.setVisible(busy and kind == "batch")
            if busy:
                QtCore.QTimer.singleShot(0, self.cancel_button.setFocus)
                self._set_status("status.running", kind_key=f"job.{kind}")
            elif kind:
                if result_ok is False:
                    self._set_status(
                        "status.job_failed",
                        flags="result_failed",
                        kind_key=f"job.{kind}",
                    )
                    return
                status_key = {
                    "preview": "status.preview_complete",
                    "optimize": "status.optimize_complete",
                    "measure_geometry": "status.geometry_measure_complete",
                    "refine_geometry": "status.geometry_refine_complete",
                    "batch": "status.batch_complete",
                    "cancelled": "status.cancelled",
                    "ignored": "status.late_ignored",
                }.get(kind, "status.ready")
                if kind in {"measure_geometry", "refine_geometry"}:
                    result = self._last_result
                    ellipse = _result_value(
                        result,
                        ("ellipse_fit", "ellipse", "ellipse_result"),
                        None,
                    )
                    quality = _result_value(
                        ellipse,
                        ("quality_status", "status"),
                        _read(
                            _result_value(ellipse, ("quality",), {}),
                            ("status",),
                            "WARN",
                        ),
                    ) or "WARN"
                    self._set_status(
                        status_key,
                        q_unit=self._active_q_unit(result),
                        quality=quality,
                    )
                else:
                    self._set_status(status_key)

        def _set_status(
            self,
            key: str,
            *,
            flags: str | None = None,
            **values: Any,
        ) -> None:
            self._status_key = key
            self._status_values = dict(values)
            self._render_status()
            if flags is not None:
                self._displayed_flags_text = str(flags)
                self._render_metric_labels()
            if hasattr(self, "workflow_status_label"):
                try:
                    from .workbench import _refresh_workflow_guide

                    _refresh_workflow_guide(self)
                except Exception:
                    pass

        def closeEvent(self, event: Any) -> None:  # noqa: N802 - Qt API
            if self._closing:
                event.ignore()
                return
            self._closing = True
            self.cancel_jobs()
            self._thread_pool.clear()
            if self._workers:
                self._set_status("status.closing")
                event.ignore()
                QtCore.QTimer.singleShot(50, self._finish_close_when_idle)
                return
            self._thread_pool.waitForDone(1500)
            event.accept()

        def _finish_close_when_idle(self) -> None:
            if self._workers:
                QtCore.QTimer.singleShot(50, self._finish_close_when_idle)
                return
            self._thread_pool.clear()
            self._thread_pool.waitForDone(1500)
            self._closing = False
            self.close()


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
    language: str | None = None,
    window_cls: Any = None,
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
    window_type = window_cls or RefinementMainWindow
    window = window_type(
        analysis_service=service,
        parameters=configured_parameters,
        analysis_settings=analysis_options if isinstance(analysis_options, Mapping) else None,
        mask_frame=selected_mask_frame,
        mask_dataset=selected_mask_dataset,
        auto_preview=not bool(options.no_auto_preview),
        language=language,
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
    # Keep the direct ``butterfly_saxs.ui.main_window.create_app`` seam as
    # usable as the public ``butterfly_saxs.ui`` entry point.  The latter
    # installs this presentation layer from its subclass constructor; the
    # direct seam needs one late, idempotent call after input loading.
    from .workbench import upgrade_window

    upgrade_window(window)
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
