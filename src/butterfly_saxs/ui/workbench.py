"""Public Qt workbench with first-run and small-screen usability safeguards.

The large scientific window remains implemented in :mod:`main_window`.  This
thin public layer adds only presentation safeguards: a scrollable parameter
dock and a compact workflow/readiness guide.  Numerical settings and result
schemas are not changed here.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from . import main_window as _base
from .qt_compat import QT_AVAILABLE, QtCore, QtWidgets

_BASE_WINDOW = _base.RefinementMainWindow

_GUIDE_TEXT = {
    "zh_CN": {
        "title": "工作流状态",
        "input": "输入",
        "q": "q 标定",
        "mask": "Mask / ROI",
        "result": "结果",
        "next": "建议下一步",
        "missing": "未加载",
        "loaded": "已加载：{name}",
        "q_missing": "未建立",
        "q_pixel": "pixel-q（仅像素坐标，不可作物理尺度解释）",
        "q_physical": "{unit}（物理 q 坐标）",
        "mask_none": "未设置",
        "mask_set": "{parts}",
        "file_mask": "外部 mask",
        "roi_count": "{count} 个 ROI",
        "result_none": "尚未运行 Preview / Optimize",
        "result_failed": "最近结果失败；请检查 flags 与残差",
        "result_ready": "{kind} 已完成；人工状态：{review}",
        "geometry_result": "{kind} 已完成；仅几何测量，尚未运行整幅强度精修；质量：{quality}",
        "geometry_review": "检查 Observed 中的 ridge 与椭圆，以及几何质量 flags；整幅强度精修尚未运行。",
        "open_image": "打开二维 SAXS 图像。",
        "select_poni": "选择与该数据对应的 PONI；未标定前不要解释间距。",
        "preview": "设置 q 范围和 mask 后运行 Preview。",
        "inspect_failure": "检查错误提示、有效像素、q 范围和 mask，再重新运行。",
        "review": "检查 Observed / Model / Residual / Overlay 后接受或拒绝结果。",
        "export": "导出证据包，或在一致配置下进入批处理。",
    },
    "en": {
        "title": "Workflow status",
        "input": "Input",
        "q": "q calibration",
        "mask": "Mask / ROI",
        "result": "Result",
        "next": "Recommended next step",
        "missing": "not loaded",
        "loaded": "loaded: {name}",
        "q_missing": "not available",
        "q_pixel": "pixel-q (pixel coordinates only; not a physical length scale)",
        "q_physical": "{unit} (physical q coordinates)",
        "mask_none": "none",
        "mask_set": "{parts}",
        "file_mask": "external mask",
        "roi_count": "{count} ROI(s)",
        "result_none": "Preview / Optimize has not been run",
        "result_failed": "latest result failed; inspect flags and residuals",
        "result_ready": "{kind} complete; manual status: {review}",
        "geometry_result": "{kind} complete; geometry only, whole-pixel intensity fit not run; quality: {quality}",
        "geometry_review": "Review the observed ridge and ellipse plus geometry quality flags; whole-pixel intensity fit has not run.",
        "open_image": "Open a two-dimensional SAXS image.",
        "select_poni": "Select the matching PONI before interpreting physical spacing.",
        "preview": "Set the q range and mask, then run Preview.",
        "inspect_failure": (
            "Inspect the error, valid pixels, q range, and mask, then rerun."
        ),
        "review": (
            "Inspect all four views, then explicitly accept or reject the result."
        ),
        "export": (
            "Export the evidence bundle or start a consistently configured batch."
        ),
    },
}


def _language(window: Any) -> str:
    language = str(getattr(window, "_language", "zh_CN")).lower()
    return "en" if language.startswith("en") else "zh_CN"


def _q_unit(window: Any) -> str:
    getter = getattr(window, "_active_q_unit", None)
    if callable(getter):
        try:
            return str(getter() or "unknown")
        except Exception:
            pass
    qmap = getattr(window, "_qmap", None)
    if isinstance(qmap, Mapping):
        return str(qmap.get("q_unit", qmap.get("unit", "unknown")) or "unknown")
    return "unknown"


def _is_physical_q(unit: str) -> bool:
    normalized = (
        str(unit)
        .strip()
        .lower()
        .replace(" ", "")
        .replace("−", "-")
        .replace("⁻¹", "^-1")
    )
    return normalized in {
        "1/nm",
        "nm^-1",
        "1/a",
        "a^-1",
        "angstrom^-1",
        "å^-1",
    }


def _result_failed(window: Any) -> bool:
    result = getattr(window, "_last_result", None)
    if result is None:
        return False
    try:
        return bool(_base._result_has_failure(result))
    except Exception:
        return False


def _geometry_only(window: Any) -> bool:
    result = getattr(window, "_last_result", None)
    action = _base._read(
        result,
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
    model_status = _base._read(result, ("model_status",), None)
    if str(model_status or "").strip().lower() in {
        "unfitted_preview",
        "geometry_only",
        "not_run",
        "unfitted",
    }:
        return True
    return str(getattr(window, "_last_result_kind", "")).lower() in {
        "measure_geometry",
        "refine_geometry",
    }


def _workflow_lines(window: Any) -> tuple[str, bool]:
    text = _GUIDE_TEXT[_language(window)]
    observed = getattr(window, "_observed", None)
    source_path = getattr(window, "_source_path", None)
    if observed is None:
        input_value = text["missing"]
    else:
        input_value = text["loaded"].format(
            name=Path(source_path).name if source_path else "in-memory array"
        )

    unit = _q_unit(window)
    has_qmap = getattr(window, "_qmap", None) is not None
    if not has_qmap or unit.strip().lower() in {"", "unknown", "map unit"}:
        q_value = text["q_missing"]
        q_ready = False
    elif _is_physical_q(unit):
        q_value = text["q_physical"].format(unit=unit)
        q_ready = True
    else:
        q_value = text["q_pixel"]
        q_ready = False

    mask_parts: list[str] = []
    if getattr(window, "_file_mask", None) is not None:
        mask_parts.append(text["file_mask"])
    roi_count = len(getattr(window, "_roi_specs", ()) or ())
    if roi_count:
        mask_parts.append(text["roi_count"].format(count=roi_count))
    mask_value = (
        text["mask_none"]
        if not mask_parts
        else text["mask_set"].format(parts=", ".join(mask_parts))
    )

    result = getattr(window, "_last_result", None)
    failed = _result_failed(window)
    fit_session = getattr(window, "_fit_session", {})
    review = (
        str(fit_session.get("manual_status", "unreviewed"))
        if isinstance(fit_session, Mapping)
        else "unreviewed"
    )
    geometry_only = _geometry_only(window)
    if result is None:
        result_value = text["result_none"]
    elif failed:
        result_value = text["result_failed"]
    elif geometry_only:
        ellipse = _base._read(
            result,
            ("ellipse_fit", "ellipse", "ellipse_result"),
            None,
        )
        quality = _base._read(
            ellipse,
            ("quality_status", "status"),
            _base._read(
                _base._read(ellipse, ("quality",), {}),
                ("status",),
                "WARN",
            ),
        ) or "WARN"
        raw_kind = str(getattr(window, "_last_result_kind", None) or "geometry")
        if _language(window) == "zh_CN":
            kind = "几何测量" if raw_kind == "measure_geometry" else "几何精修"
        else:
            kind = (
                "geometry measurement"
                if raw_kind == "measure_geometry"
                else "geometry refinement"
            )
        result_value = text["geometry_result"].format(
            kind=kind,
            quality=quality,
        )
    else:
        kind = str(getattr(window, "_last_result_kind", None) or "result")
        result_value = text["result_ready"].format(kind=kind, review=review)

    if observed is None:
        next_step = text["open_image"]
    elif failed:
        next_step = text["inspect_failure"]
    elif not q_ready:
        next_step = text["select_poni"]
    elif result is None:
        next_step = text["preview"]
    elif geometry_only:
        next_step = text["geometry_review"]
    elif review in {"unreviewed", ""}:
        next_step = text["review"]
    else:
        next_step = text["export"]

    body = "\n".join(
        (
            f"{text['input']}: {input_value}",
            f"{text['q']}: {q_value}",
            f"{text['mask']}: {mask_value}",
            f"{text['result']}: {result_value}",
            f"{text['next']}: {next_step}",
        )
    )
    return body, bool(observed is not None and not q_ready)


def _refresh_workflow_guide(window: Any) -> None:
    label = getattr(window, "workflow_status_label", None)
    group = getattr(window, "workflow_status_group", None)
    if label is None or group is None:
        return
    text = _GUIDE_TEXT[_language(window)]
    if group.title() != text["title"]:
        group.setTitle(text["title"])
    body, warning = _workflow_lines(window)
    if label.text() != body:
        label.setText(body)
    if bool(label.property("calibrationWarning")) != warning:
        label.setProperty("calibrationWarning", warning)
        label.setStyleSheet(
            "QLabel[calibrationWarning='true'] { font-weight: 600; }"
            if warning
            else ""
        )


def _ensure_widget_visible_exact(scroll: Any, widget: Any) -> None:
    """Scroll a focused control until both horizontal edges are visible."""

    if scroll is None or widget is None or not widget.isVisible():
        return
    scroll.ensureWidgetVisible(widget, 0, 0)
    viewport = scroll.viewport()
    bar = scroll.horizontalScrollBar()
    for _ in range(2):
        top_left = widget.mapTo(viewport, widget.rect().topLeft())
        bottom_right = widget.mapTo(viewport, widget.rect().bottomRight())
        delta = 0
        if top_left.x() < viewport.rect().left():
            delta = top_left.x() - viewport.rect().left()
        elif bottom_right.x() > viewport.rect().right():
            delta = bottom_right.x() - viewport.rect().right()
        if not delta:
            break
        bar.setValue(bar.value() + delta)


class _FocusVisibilityFilter(QtCore.QObject):
    """Bring keyboard-focused dock controls fully into the scroll viewport."""

    def __init__(self, scroll: Any, parent: Any = None) -> None:
        super().__init__(parent)
        self.scroll = scroll

    def eventFilter(self, watched: Any, event: Any) -> bool:  # noqa: N802 - Qt API
        if event.type() == QtCore.QEvent.Type.FocusIn:
            QtCore.QTimer.singleShot(
                0,
                lambda: _ensure_widget_visible_exact(self.scroll, watched),
            )
        return False


def upgrade_window(window: Any) -> Any:
    """Add idempotent workflow guidance and scroll-safe dock presentation."""

    if not QT_AVAILABLE or getattr(window, "_usability_upgrade_installed", False):
        return window
    dock = getattr(window, "parameters_dock", None)
    if dock is None:
        return window
    dock_widget = dock.widget()
    if dock_widget is None:
        return window

    # ``main_window.create_app`` can be imported directly by lightweight
    # launchers and probes, so the presentation layer may be installed after
    # the dock has already been built.  Reuse an existing scroll area instead
    # of wrapping it a second time.
    if isinstance(dock_widget, QtWidgets.QScrollArea):
        scroll = dock_widget
        panel = scroll.widget()
        if panel is None:
            return window
        layout = panel.layout()
    else:
        panel = dock_widget
        layout = panel.layout()

    if layout is not None and not hasattr(window, "workflow_status_group"):
        guide = QtWidgets.QGroupBox(panel)
        guide.setObjectName("workflowStatusGroup")
        guide_layout = QtWidgets.QVBoxLayout(guide)
        guide_layout.setContentsMargins(8, 6, 8, 6)
        label = QtWidgets.QLabel(guide)
        label.setObjectName("workflowStatusLabel")
        label.setWordWrap(True)
        label.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        label.setAccessibleName(window._tr("a11y.workflow_status"))
        # Bound the guide's long prose so it participates in the dock's
        # narrow layout instead of imposing a 700 px minimum width.
        label.setFixedWidth(320)
        guide_layout.addWidget(label)
        layout.insertWidget(0, guide)
        window.workflow_status_group = guide
        window.workflow_status_label = label

    table = getattr(window, "parameter_table", None)
    if table is not None:
        table.setMinimumHeight(max(170, table.minimumSizeHint().height()))

    if not isinstance(dock_widget, QtWidgets.QScrollArea):
        panel.setParent(None)
        if layout is not None:
            layout.setSizeConstraint(QtWidgets.QLayout.SizeConstraint.SetMinimumSize)
        panel.setMinimumHeight(max(720, panel.minimumSizeHint().height()))

        scroll = QtWidgets.QScrollArea(dock)
        scroll.setObjectName("parametersScrollArea")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        # The control panel has a wider natural layout than a narrow dock.
        # Keep horizontal scrolling available so ensureWidgetVisible(), focus
        # traversal, and keyboard users can reach every right-hand field.
        scroll.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        scroll.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        scroll.setAccessibleName(window._tr("a11y.scroll_controls"))
        scroll.setWidget(panel)
        dock.setWidget(scroll)
    else:
        # Keep a deterministic minimum content height so the scroll bar is
        # available on 980x680 windows while still allowing the dock to shrink
        # when the user maximizes the central plot.
        panel.setMinimumHeight(max(720, panel.minimumSizeHint().height()))

    dock.setMinimumWidth(360)
    dock.setMaximumWidth(520)
    try:
        target_width = min(440, max(380, int(window.width() * 0.32)))
        window.resizeDocks(
            [dock],
            [target_width],
            QtCore.Qt.Orientation.Horizontal,
        )
    except Exception:
        pass
    window.parameters_scroll_area = scroll
    window.parameters_panel = panel
    focus_filter = _FocusVisibilityFilter(scroll, window)
    for widget in panel.findChildren(QtWidgets.QWidget):
        widget.installEventFilter(focus_filter)
    window._focus_visibility_filter = focus_filter
    if callable(getattr(window, "_retranslate_accessible_names", None)):
        window._retranslate_accessible_names()

    timer = QtCore.QTimer(window)
    timer.setObjectName("workflowStatusTimer")
    timer.setInterval(600)
    timer.timeout.connect(lambda: _refresh_workflow_guide(window))
    timer.start()
    window.workflow_status_timer = timer
    window._usability_upgrade_installed = True
    _refresh_workflow_guide(window)
    return window


if QT_AVAILABLE:

    class RefinementMainWindow(_BASE_WINDOW):
        """Public workbench with the presentation upgrade installed."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            upgrade_window(self)

else:
    RefinementMainWindow = _BASE_WINDOW


MainWindow = RefinementMainWindow
Workbench = RefinementMainWindow
RefinementWindow = RefinementMainWindow
WorkbenchWindow = RefinementMainWindow
symmetric_ellipses = _base.symmetric_ellipses

def create_app(argv: list[str] | None = None, **kwargs: Any) -> tuple[Any, Any]:
    """Create the public workbench through an explicit factory hook."""

    kwargs["window_cls"] = RefinementMainWindow
    return _base.create_app(argv, **kwargs)


def launch(argv: list[str] | None = None, **kwargs: Any) -> int:
    """Run the public workbench without mutating ``main_window`` globals."""

    app, window = create_app(argv, **kwargs)
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
    "upgrade_window",
]
