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
    if result is None:
        result_value = text["result_none"]
    elif failed:
        result_value = text["result_failed"]
    else:
        kind = str(getattr(window, "_last_result_kind", None) or "result")
        result_value = text["result_ready"].format(kind=kind, review=review)

    if observed is None:
        next_step = text["open_image"]
    elif not q_ready:
        next_step = text["select_poni"]
    elif result is None:
        next_step = text["preview"]
    elif failed:
        next_step = text["inspect_failure"]
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


def upgrade_window(window: Any) -> Any:
    """Add idempotent workflow guidance and scroll-safe dock presentation."""

    if not QT_AVAILABLE or getattr(window, "_usability_upgrade_installed", False):
        return window
    dock = getattr(window, "parameters_dock", None)
    if dock is None:
        return window
    panel = dock.widget()
    if panel is None:
        return window

    layout = panel.layout()
    if layout is not None:
        guide = QtWidgets.QGroupBox(panel)
        guide.setObjectName("workflowStatusGroup")
        guide_layout = QtWidgets.QVBoxLayout(guide)
        guide_layout.setContentsMargins(8, 6, 8, 6)
        label = QtWidgets.QLabel(guide)
        label.setObjectName("workflowStatusLabel")
        label.setWordWrap(True)
        label.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        label.setAccessibleName("LamellarSAXS2D workflow status")
        guide_layout.addWidget(label)
        layout.insertWidget(0, guide)
        window.workflow_status_group = guide
        window.workflow_status_label = label

    table = getattr(window, "parameter_table", None)
    if table is not None:
        table.setMinimumHeight(max(220, table.minimumSizeHint().height()))

    panel.setParent(None)
    if layout is not None:
        layout.setSizeConstraint(QtWidgets.QLayout.SizeConstraint.SetMinimumSize)
    panel.setMinimumHeight(max(840, panel.minimumSizeHint().height()))

    scroll = QtWidgets.QScrollArea(dock)
    scroll.setObjectName("parametersScrollArea")
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
    scroll.setHorizontalScrollBarPolicy(
        QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded
    )
    scroll.setVerticalScrollBarPolicy(
        QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded
    )
    scroll.setAccessibleName("Scrollable analysis controls")
    scroll.setWidget(panel)
    dock.setWidget(scroll)
    dock.setMinimumWidth(460)
    window.parameters_scroll_area = scroll
    window.parameters_panel = panel

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

# The base factories perform all configuration/project loading.  Redirect their
# class lookup to the public presentation subclass rather than duplicating that
# scientific start-up logic here.
_base.RefinementMainWindow = RefinementMainWindow
_base.MainWindow = RefinementMainWindow
_base.Workbench = RefinementMainWindow
_base.RefinementWindow = RefinementMainWindow
_base.WorkbenchWindow = RefinementMainWindow

create_app = _base.create_app
launch = _base.launch


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
