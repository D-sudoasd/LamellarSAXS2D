"""Compact, evidence-bounded quality summary for the butterfly workbench.

The summary is deliberately a view adapter.  It reads the result fields that
the analysis service already produced and does not infer physical quantities or
scientific acceptance from a candidate fit.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any

from ..butterfly_quality import classify_ellipse_publication
from ..settings import canonical_q_unit
from .qt_compat import QT_AVAILABLE, QtWidgets, require_qt


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


def _text_values(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, Mapping):
        return [str(key) for key, active in value.items() if active]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [str(item) for item in value if item not in (None, "")]
    return [str(value)]


def _dedupe(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(value) for value in values if str(value).strip()))


def _normalise_q_unit(value: Any) -> str:
    canonical = canonical_q_unit(value)
    if canonical == "nm⁻¹":
        return "nm^-1"
    if canonical == "Å⁻¹":
        return "Å^-1"
    return canonical or "unknown"


def _is_physical_q_unit(unit: str) -> bool:
    return unit in {"nm^-1", "Å^-1"}


def _published_scalar(
    result: Mapping[str, Any],
    names: tuple[str, ...],
) -> float | None:
    """Read an existing observable without deriving it from another field.

    The service historically stores the observed first-order values directly
    on ``candidate_fit``.  They are accepted here as direct values, while
    ``candidate_value`` rows and axis-derived expressions are intentionally
    ignored.  In particular, this function never computes ``L = 2*pi/q``.
    """

    blocks: list[Mapping[str, Any]] = [result]
    for key in ("published", "public", "observables", "geometry_parameters", "candidate_fit"):
        block = result.get(key)
        if isinstance(block, Mapping):
            blocks.append(block)
    quantitative = result.get("quantitative_parameters")
    if isinstance(quantitative, Mapping):
        for name in names:
            row = quantitative.get(name)
            if isinstance(row, Mapping):
                value = _finite(row.get("value"))
                if value is not None:
                    return value
    for block in blocks:
        for name in names:
            value = _finite(block.get(name))
            if value is not None:
                return value
    return None


def _published_scalar_with_unit(
    result: Mapping[str, Any],
    names: tuple[str, ...],
    *,
    fallback_unit: str,
) -> tuple[float | None, str]:
    """Return a direct observable and the unit declared by its field."""

    blocks: list[Mapping[str, Any]] = [result]
    for key in ("published", "public", "observables", "geometry_parameters", "candidate_fit"):
        block = result.get(key)
        if isinstance(block, Mapping):
            blocks.append(block)
    quantitative = result.get("quantitative_parameters")
    if isinstance(quantitative, Mapping):
        for name in names:
            row = quantitative.get(name)
            if isinstance(row, Mapping):
                value = _finite(row.get("value"))
                if value is not None:
                    return value, _normalise_q_unit(row.get("unit", fallback_unit))
    for block in blocks:
        for name in names:
            value = _finite(block.get(name))
            if value is not None:
                declared = "nm^-1" if name == "q_star_nm_inv" else block.get("q_unit", fallback_unit)
                return value, _normalise_q_unit(declared)
    return None, fallback_unit


def _sector_measurement_field(result: Mapping[str, Any], name: str) -> Any:
    """Read the sector-profile statistic before any legacy fit aliases."""

    summary = result.get("measurement_summary")
    if isinstance(summary, Mapping) and name in summary:
        return summary.get(name)
    return result.get(name)


def _sector_measurement_scalar(
    result: Mapping[str, Any],
    *,
    fallback_unit: str,
) -> tuple[float | None, str]:
    """Return the selected-sector q* median and its declared q unit."""

    value = _finite(_sector_measurement_field(result, "q_star_sector_median"))
    declared = _sector_measurement_field(result, "q_star_sector_median_unit")
    if declared in (None, ""):
        summary = result.get("measurement_summary")
        declared = summary.get("q_unit") if isinstance(summary, Mapping) else None
    return value, _normalise_q_unit(declared or fallback_unit)


def _sector_measurement_period(result: Mapping[str, Any], *, unit: str) -> float | None:
    """Return the explicitly reported apparent period for a physical q unit.

    A pixel-q or otherwise unknown unit must never acquire an ``nm`` label in
    the view layer, even if an old or hand-built payload contains a stale
    period value.
    """

    if not _is_physical_q_unit(unit):
        return None
    return _finite(_sector_measurement_field(result, "apparent_period_from_sector_median_nm"))


def _side_support(result: Mapping[str, Any]) -> str:
    quality = result.get("quality")
    quality = quality if isinstance(quality, Mapping) else {}
    metrics = quality.get("metrics")
    metrics = metrics if isinstance(metrics, Mapping) else {}
    counts = metrics.get("side_counts")
    if isinstance(counts, Mapping) and counts:
        occupied = sum(1 for value in counts.values() if _finite(value) not in (None, 0.0))
        return f"{occupied}/4"
    occupied = _finite(metrics.get("occupied_sides"))
    if occupied is not None:
        return f"{max(0, min(4, int(occupied)))}/4"

    points = result.get("points")
    groups: set[tuple[int, str]] = set()
    if isinstance(points, Sequence) and not isinstance(points, (str, bytes)):
        for point in points:
            if not isinstance(point, Mapping):
                continue
            if not bool(point.get("accepted", point.get("valid", False))) or not bool(
                point.get("valid", True)
            ):
                continue
            branch = point.get("branch_id")
            side = str(point.get("side", "")).lower()
            if branch in (0, 1) and side in {"upper", "lower"}:
                groups.add((int(branch), side))
    return f"{len(groups)}/4" if groups else "—"


def _quality_flags(result: Mapping[str, Any]) -> list[str]:
    quality = result.get("quality")
    quality = quality if isinstance(quality, Mapping) else {}
    candidate = result.get("candidate_fit")
    candidate = candidate if isinstance(candidate, Mapping) else {}
    flags: list[str] = []
    for source in (result, quality, candidate):
        flags.extend(_text_values(source.get("flags")))
    bound_flags = candidate.get("bound_flags")
    if isinstance(bound_flags, Mapping) and bound_flags.get("axis_ratio"):
        flags.append("axis_ratio_at_bound")
    metrics = quality.get("metrics")
    if isinstance(metrics, Mapping):
        flags.extend(_text_values(metrics.get("flags")))
    return list(_dedupe(flags))


def _result_failed(result: Mapping[str, Any]) -> bool:
    for source in (result, result.get("quality"), result.get("metrics")):
        if not isinstance(source, Mapping):
            continue
        for key in ("status", "measurement_status", "solver_status", "quality_status"):
            if str(source.get(key, "") or "").strip().lower() in {
                "fail",
                "failed",
                "error",
                "invalid",
            }:
                return True
        if source.get("success") is False:
            return True
    return False


_REASON_TEXT: dict[str, tuple[str, str]] = {
    "axis_ratio_at_bound": ("轴比触及边界，按仅一阶环处理", "axis ratio at bound; ring only"),
    "axis_ratio_collapsed_to_line": ("轴比塌缩，椭圆不可分辨", "axis ratio collapsed; ellipse unresolved"),
    "major_axis_exceeds_observed_extent": ("长轴超出观测范围", "major axis exceeds observed extent"),
    "insufficient_occupied_sides": ("有效侧边不足", "insufficient occupied sides"),
    "insufficient_independent_side_support": ("独立侧边支持不足", "insufficient independent side support"),
    "solver_or_arc_support_unavailable": ("求解器或弧线支持不可用", "solver or arc support unavailable"),
    "ill_conditioned_geometry": ("几何拟合病态", "ill-conditioned geometry"),
    "residual_exceeds_localization_scale": ("残差超过定位不确定度尺度", "residual exceeds localization scale"),
    "per_arc_support_evidence_unavailable": ("逐弧支持证据不可用", "per-arc support evidence unavailable"),
    "poor_match": ("观测与拟合几何失配", "poor match between observed and fitted geometry"),
    "arc_endpoint_or_manual_bound_dependent": (
        "依赖弧端点或手动边界",
        "depends on arc endpoints or manual bounds",
    ),
    "disconnected_observed_support": (
        "观测弧支持不连续",
        "observed arc support is disconnected",
    ),
    "observed_support_infeasible": (
        "部分拟合投影超出观测支持",
        "some fitted projections exceed observed support",
    ),
    "geometry_not_evaluated": ("椭圆参数尚未评估", "ellipse parameters not evaluated"),
    "uncalibrated_pixel_q": ("pixel-q 未标定，不能解释物理周期", "pixel-q is uncalibrated; physical period unavailable"),
    "spacing_unavailable_unknown_q_unit": ("q 单位未确认，L ring 不可用", "q unit is unknown; L ring unavailable"),
    "cancelled": ("任务已取消，旧结果已失效", "job cancelled; previous result invalidated"),
}


def _reason_text(code: str, *, english: bool) -> str:
    pair = _REASON_TEXT.get(str(code))
    if pair is not None:
        return pair[1 if english else 0]
    return str(code).replace("_", " ")


def _reason_priority(code: str) -> tuple[int, str]:
    token = str(code).lower()
    if token == "poor_match":
        return (0, token)
    if "bound" in token or "collapsed" in token:
        return (1, token)
    if "support" in token or "occupied_sides" in token:
        return (2, token)
    if "residual" in token or "condition" in token:
        return (3, token)
    if "sensitivity" in token or "interval" in token:
        return (8, token)
    if "uncalibrated" in token or "unknown_q_unit" in token:
        return (9, token)
    if "not_evaluated" in token:
        return (10, token)
    return (5, token)


def _format_value(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.6g}"


@dataclass(frozen=True)
class ButterflyQualitySummaryState:
    """Language-neutral state rendered by :class:`ButterflyQualitySummary`."""

    status_key: str
    engineering_status: str
    side_support: str
    q_star: float | None
    q_star_unit: str
    l_ring: float | None
    q_unit: str
    calibrated: bool
    reasons: tuple[str, ...]
    next_step_key: str
    scientific_acceptance: bool = False
    trace_method: str = "curvature"

    def as_dict(self) -> dict[str, Any]:
        return {
            "status_key": self.status_key,
            "status": self.status_key,
            "engineering_status": self.engineering_status,
            "side_support": self.side_support,
            "q_star": self.q_star,
            "q_star_unit": self.q_star_unit,
            "l_ring": self.l_ring,
            "q_unit": self.q_unit,
            "calibrated": self.calibrated,
            "reasons": list(self.reasons),
            "next_step_key": self.next_step_key,
            "scientific_acceptance": self.scientific_acceptance,
            "trace_method": self.trace_method,
        }

    def reasons_text(self, *, english: bool = False) -> str:
        return "；".join(_reason_text(code, english=english) for code in self.reasons)


def build_butterfly_quality_summary(
    result: Mapping[str, Any] | None = None,
    *,
    stage: str = "trace",
    page_state: str = "ready",
    result_fresh: bool | None = None,
    data_ready: bool | None = None,
    busy: bool = False,
    q_unit: Any = None,
    poor_match: bool = False,
    error: Any = None,
    trace_method: str | None = None,
) -> ButterflyQualitySummaryState:
    """Build a compact view state from existing result evidence.

    ``result_fresh`` is separate from ``result`` on purpose: cancellation and
    input edits can leave an old payload in memory while making it invalid for
    display/export.  The summary must then clear all old observables.
    """

    payload = dict(result) if isinstance(result, Mapping) else {}
    settings = payload.get("settings")
    settings = settings if isinstance(settings, Mapping) else {}
    resolved_trace_method = str(
        trace_method
        or settings.get("trace_method")
        or (
            "annular_peak"
            if isinstance(payload.get("annular_peaks"), Mapping)
            or str(payload.get("method_version", "")).startswith("butterfly-annular-")
            else
            "radial_sector"
            if isinstance(payload.get("sector_peaks"), Mapping)
            or str(payload.get("method_version", "")).startswith("butterfly-radial-sector-")
            else "curvature"
        )
    ).strip().lower()
    if resolved_trace_method in {
        "azimuthal_peak",
        "annular_peaks",
        "annular_trajectory",
        "q_ring_trajectory",
    }:
        resolved_trace_method = "annular_peak"
    if str(payload.get("ridge_method", "") or "").strip().lower() in {
        "azimuthal_peak",
        "annular_peak",
        "annular_trajectory",
    }:
        resolved_trace_method = "annular_peak"
    if result_fresh is None:
        result_fresh = bool(payload)
    if data_ready is None:
        data_ready = bool(result_fresh)
    state = str(page_state or "ready").lower()
    active_stage = str(stage or "trace").lower()
    # A stale payload may remain in memory while the current input is being
    # rerun. Never let its quality, flags, or observables leak into the card.
    visible_result = bool(
        result_fresh and not busy and state not in {"running", "cancelling"}
    )
    effective_payload = payload if visible_result else {}
    quality = effective_payload.get("quality")
    quality = quality if isinstance(quality, Mapping) else {}
    quality_status = str(
        quality.get("status", effective_payload.get("quality_status", "")) or ""
    ).strip().upper()
    engineering = str(
        quality.get(
            "engineering_status",
            effective_payload.get("engineering_status", quality_status),
        )
        or ""
    ).strip().upper()
    if state in {"failed", "error"} or _result_failed(effective_payload) or quality_status in {"FAIL", "FAILED", "INVALID"}:
        engineering = "FAIL"
    elif engineering in {"FAILED", "INVALID", "ERROR"}:
        engineering = "FAIL"
    elif quality_status == "WARN" or engineering == "WARN":
        engineering = "WARN"
    elif engineering in {"OK", "AVAILABLE"} or quality_status in {"OK", "PASS", "AVAILABLE"}:
        engineering = "PASS"
    elif quality_status in {"NOT_EVALUATED", "PENDING", "PENDING_EVALUATION"}:
        engineering = "NOT_EVALUATED"
    elif not engineering:
        engineering = "NOT_EVALUATED"

    unit_value = q_unit
    if unit_value in (None, ""):
        unit_value = effective_payload.get("q_unit")
    if unit_value in (None, ""):
        metrics = quality.get("metrics")
        unit_value = metrics.get("q_unit") if isinstance(metrics, Mapping) else None
    if unit_value in (None, ""):
        geometry = effective_payload.get("geometry_parameters")
        unit_value = geometry.get("q_unit") if isinstance(geometry, Mapping) else None
    if unit_value in (None, ""):
        candidate_unit = effective_payload.get("candidate_fit")
        unit_value = candidate_unit.get("q_unit") if isinstance(candidate_unit, Mapping) else None
    unit = _normalise_q_unit(unit_value)
    calibrated = _is_physical_q_unit(unit)

    flags = _quality_flags(effective_payload)
    if poor_match:
        flags.append("poor_match")
    flags = list(_dedupe(flags))
    candidate = effective_payload.get("candidate_fit")
    candidate = candidate if isinstance(candidate, Mapping) else {}
    quantitative = effective_payload.get("quantitative_parameters")
    quantitative = quantitative if isinstance(quantitative, Mapping) else {}
    ratio_row = quantitative.get("axis_ratio")
    published_ratio = (
        _finite(ratio_row.get("value")) if isinstance(ratio_row, Mapping) else None
    )
    ratio = published_ratio if published_ratio is not None else _finite(candidate.get("axis_ratio"))
    kind = classify_ellipse_publication(
        quality_status=quality_status,
        axis_ratio=ratio,
        flags=flags,
    )

    if resolved_trace_method == "annular_peak":
        # Annular angular maxima are trajectory coordinates.  They are not a
        # radial q* statistic and must not be converted to a spacing here.
        q_star, q_star_unit = None, unit
        l_ring = None
    elif resolved_trace_method == "radial_sector":
        q_star, q_star_unit = _sector_measurement_scalar(
            effective_payload, fallback_unit=unit
        )
        l_ring = _sector_measurement_period(effective_payload, unit=q_star_unit)
    else:
        q_star, q_star_unit = _published_scalar_with_unit(
            effective_payload,
            ("q_star_from_arcs", "q_star_nm_inv"),
            fallback_unit=unit,
        )
        l_ring = _published_scalar(effective_payload, ("L_from_observed_radius_nm", "L_ring_nm", "L_ring"))
    reasons: list[str] = []

    if busy or state in {"running", "cancelling"}:
        status_key = "running"
        next_step_key = "wait"
    elif state in {"failed", "error"}:
        status_key = "failed"
        if error not in (None, ""):
            reasons.append(str(error))
        reasons.extend(flags)
        next_step_key = "retry"
    elif not visible_result:
        status_key = "ready" if data_ready else "empty"
        next_step_key = "identify" if data_ready else "load"
        if state in {"cancelled", "canceled", "ignored", "stale"}:
            reasons.append("cancelled")
    elif active_stage == "trace" or quality_status in {
        "NOT_EVALUATED",
        "PENDING",
        "PENDING_EVALUATION",
    }:
        status_key = "pending_evaluation"
        next_step_key = "evaluate"
        reasons.append("geometry_not_evaluated")
    elif _result_failed(effective_payload) or kind == "fail":
        status_key = "failed"
        next_step_key = "retry"
        reasons.extend(flags)
    elif kind == "ring":
        status_key = "ring_only"
        next_step_key = "ring_review"
        reasons.extend(flags)
    else:
        status_key = "ellipse_candidate"
        next_step_key = "ellipse_review"
        reasons.extend(flags)

    if visible_result and not calibrated:
        reasons.append("uncalibrated_pixel_q" if unit == "pixel-q" else "spacing_unavailable_unknown_q_unit")
        l_ring = None
    bound_flags = candidate.get("bound_flags")
    if (
        kind == "ring"
        and "axis_ratio_at_bound" not in reasons
        and isinstance(bound_flags, Mapping)
        and bound_flags.get("axis_ratio")
    ):
        reasons.append("axis_ratio_at_bound")
    reasons = sorted(_dedupe(reasons), key=_reason_priority)[:3]
    if not visible_result:
        q_star = None
        l_ring = None
        side_support = "—"
    else:
        side_support = _side_support(effective_payload)

    return ButterflyQualitySummaryState(
        status_key=status_key,
        engineering_status=engineering,
        side_support=side_support,
        q_star=q_star,
        q_star_unit=q_star_unit,
        l_ring=l_ring,
        q_unit=unit,
        calibrated=calibrated,
        reasons=tuple(reasons),
        next_step_key=next_step_key,
        trace_method=resolved_trace_method,
    )


# Short alias for callers/tests that prefer the noun form.
summarize_butterfly_result = build_butterfly_quality_summary


if QT_AVAILABLE:

    class ButterflyQualitySummary(QtWidgets.QFrame):
        """A small, persistent status card kept next to the butterfly title."""

        _STATUS_TEXT = {
            "empty": ("待载入", "Load frame"),
            "ready": ("待识别", "Ready to identify"),
            "running": ("处理中", "Running"),
            "pending_evaluation": ("待评估", "Awaiting evaluation"),
            "ring_only": ("仅一阶环", "First-order ring only"),
            "ellipse_candidate": ("椭圆候选", "Ellipse candidate"),
            "failed": ("失败", "Failed"),
        }
        _NEXT_TEXT = {
            "load": ("载入经 q 标定的二维帧", "Load a calibrated 2D frame"),
            "identify": ("识别弧线以获得实际侧边支持", "Identify arcs to measure side support"),
            "wait": ("等待当前任务完成或取消", "Wait for the current job to finish or cancel it"),
            "evaluate": ("运行 Evaluate，完成椭圆质量评估", "Run Evaluate to assess ellipse quality"),
            "retry": ("检查输入、掩膜和 q 范围后重试", "Check input, mask and q range, then retry"),
            "ring_review": ("保留一阶环 L；独立支持前不发布 Ln/Lz", "Keep ring L; do not publish Ln/Lz without independent support"),
            "ellipse_review": ("复核质量和支持后再导出；科学接受仍需外部证据", "Review quality/support before export; scientific acceptance still needs external evidence"),
        }

        def __init__(self, parent: Any = None, *, language: str = "zh_CN") -> None:
            super().__init__(parent)
            self.setObjectName("butterflyQualitySummary")
            self.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Expanding,
                QtWidgets.QSizePolicy.Policy.Maximum,
            )
            self.setMinimumHeight(62)
            self.setMaximumHeight(88)
            self._language = str(language)
            self._state = ButterflyQualitySummaryState(
                "empty", "NOT_EVALUATED", "—", None, "unknown", None, "unknown", False, (), "load"
            )
            outer = QtWidgets.QVBoxLayout(self)
            outer.setContentsMargins(8, 4, 8, 4)
            outer.setSpacing(2)
            top = QtWidgets.QHBoxLayout()
            top.setSpacing(8)
            self.status_label = QtWidgets.QLabel(self)
            self.status_label.setObjectName("butterflySummaryStatus")
            self.status_label.setStyleSheet("font-weight: 700;")
            top.addWidget(self.status_label)
            self.engineering_label = QtWidgets.QLabel(self)
            self.engineering_label.setObjectName("butterflySummaryEngineering")
            top.addWidget(self.engineering_label)
            top.addStretch(1)
            self.scientific_label = QtWidgets.QLabel(self)
            self.scientific_label.setObjectName("butterflySummaryScientific")
            self.scientific_label.setStyleSheet("font-size: 10px;")
            top.addWidget(self.scientific_label)
            outer.addLayout(top)

            metrics = QtWidgets.QHBoxLayout()
            metrics.setSpacing(12)
            self.support_label = QtWidgets.QLabel(self)
            self.support_label.setObjectName("butterflySummarySideSupport")
            metrics.addWidget(self.support_label)
            self.q_star_label = QtWidgets.QLabel(self)
            self.q_star_label.setObjectName("butterflySummaryQStar")
            metrics.addWidget(self.q_star_label)
            self.l_ring_label = QtWidgets.QLabel(self)
            self.l_ring_label.setObjectName("butterflySummaryLRing")
            metrics.addWidget(self.l_ring_label)
            self.unit_label = QtWidgets.QLabel(self)
            self.unit_label.setObjectName("butterflySummaryUnit")
            metrics.addWidget(self.unit_label)
            metrics.addStretch(1)
            self._metrics_layout = metrics
            self._metric_labels = (
                self.support_label,
                self.q_star_label,
                self.l_ring_label,
                self.unit_label,
            )
            outer.addLayout(metrics)

            bottom = QtWidgets.QHBoxLayout()
            bottom.setSpacing(10)
            self.reason_label = QtWidgets.QLabel(self)
            self.reason_label.setObjectName("butterflySummaryReasons")
            self.reason_label.setWordWrap(True)
            self.reason_label.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Ignored,
                QtWidgets.QSizePolicy.Policy.Preferred,
            )
            bottom.addWidget(self.reason_label, 1)
            self.next_label = QtWidgets.QLabel(self)
            self.next_label.setObjectName("butterflySummaryNext")
            self.next_label.setWordWrap(True)
            self.next_label.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Ignored,
                QtWidgets.QSizePolicy.Policy.Preferred,
            )
            bottom.addWidget(self.next_label, 1)
            outer.addLayout(bottom)
            self.setAccessibleName("Butterfly quality summary")
            self._render()

        def resizeEvent(self, event: Any) -> None:  # noqa: N802 - Qt API
            self._sync_metric_layout()
            super().resizeEvent(event)

        @property
        def state(self) -> ButterflyQualitySummaryState:
            return self._state

        def snapshot(self) -> dict[str, Any]:
            return self._state.as_dict()

        def set_language(self, language: str) -> None:
            self._language = str(language)
            self._render()

        def set_state(
            self,
            result: Mapping[str, Any] | None = None,
            *,
            stage: str = "trace",
            page_state: str = "ready",
            result_fresh: bool | None = None,
            data_ready: bool | None = None,
            busy: bool = False,
            q_unit: Any = None,
            poor_match: bool = False,
            error: Any = None,
            trace_method: str | None = None,
        ) -> None:
            self._state = build_butterfly_quality_summary(
                result,
                stage=stage,
                page_state=page_state,
                result_fresh=result_fresh,
                data_ready=data_ready,
                busy=busy,
                q_unit=q_unit,
                poor_match=poor_match,
                error=error,
                trace_method=trace_method,
            )
            self._render()

        def clear(self, *, data_ready: bool = False) -> None:
            self.set_state({}, data_ready=data_ready, result_fresh=False)

        def _render(self) -> None:
            english = self._language.lower().startswith("en")
            state = self._state
            status_pair = self._STATUS_TEXT.get(state.status_key, (state.status_key, state.status_key))
            status_text = status_pair[1 if english else 0]
            if state.trace_method in {"radial_sector", "annular_peak"} and state.status_key == "ring_only":
                status_text = "Peaks only · ellipse unsupported" if english else "仅主峰轨迹 · 椭圆未支持"
            engineering = state.engineering_status
            if engineering == "NOT_EVALUATED":
                engineering_text = "not evaluated" if english else "待评估"
            elif engineering:
                engineering_text = f"engineering {engineering}" if english else f"工程 {engineering}"
            else:
                engineering_text = "—"
            side_text = (
                f"tracks {state.side_support}"
                if english
                else f"轨迹支持 {state.side_support}"
            ) if state.trace_method == "annular_peak" else (
                f"sides {state.side_support}" if english else f"侧支持 {state.side_support}"
            )
            q_text = _format_value(state.q_star)
            if state.q_star is not None and state.q_star_unit:
                q_text = f"{q_text} ({state.q_star_unit})"
            l_text = _format_value(state.l_ring)
            if state.l_ring is not None:
                l_text = f"{l_text} nm"
            unit_text = (
                f"q unit {state.q_unit}"
                if english
                else f"q 单位 {state.q_unit}"
            )
            if not state.calibrated:
                unit_text += " · uncalibrated" if english else " · 未标定"
            next_pair = self._NEXT_TEXT.get(state.next_step_key, (state.next_step_key, state.next_step_key))
            if state.trace_method == "annular_peak":
                next_pair = (
                    ("复核各环峰位后，评估对角配对的双椭圆", "Review ring peaks, then evaluate the paired ellipses")
                    if state.next_step_key == "evaluate" else
                    ("检查双椭圆残差和长短轴稳定性；区分可测量值与外推", "Check ellipse residuals and axis stability; separate measured support from extrapolation")
                )
            elif state.trace_method == "radial_sector" and state.next_step_key == "ring_review":
                next_pair = (
                    "检查扇区剖面；反射级次与椭圆解释尚未确认",
                    "Review sector profiles; reflection order and ellipse interpretation are unconfirmed",
                )
            reason_text = state.reasons_text(english=english)
            self.status_label.setText(status_text)
            self.engineering_label.setText(engineering_text)
            self.support_label.setText(side_text)
            if state.trace_method == "annular_peak":
                q_label = "pairing" if english else "配对"
                q_text = "QI↔QIII; QII↔QIV" if english else "对角花瓣 A / B"
                length_label = "spacing" if english else "周期"
                l_text = "not reported" if english else "不由此方法给出"
            else:
                q_label = (
                    "q* sector median" if english else "主峰 q*中位数"
                ) if state.trace_method == "radial_sector" else "q*"
                length_label = (
                    ("2π/q* (apparent)" if english else "2π/q*（表观）")
                    if state.trace_method == "radial_sector"
                    else ("L ring" if english else "环 L")
                )
            self.q_star_label.setText(f"{q_label} {q_text}")
            self.l_ring_label.setText(f"{length_label} {l_text}")
            self.unit_label.setText(unit_text)
            self.reason_label.setText(
                ("Reasons: " if english else "原因：") + (reason_text or ("—"))
            )
            self.next_label.setText(
                ("Next: " if english else "下一步：") + next_pair[1 if english else 0]
            )
            self.reason_label.setToolTip(reason_text or ("—"))
            self.next_label.setToolTip(next_pair[1 if english else 0])
            self.scientific_label.setText(
                "Scientific acceptance: not inferred"
                if english
                else "科学接受：未自动判定"
            )
            self._ensure_label_width(self.status_label)
            self._ensure_label_width(self.engineering_label)
            self._ensure_label_width(self.scientific_label)
            for label in self._metric_labels:
                self._ensure_label_width(label)
            self._sync_metric_layout()
            self.setProperty("summaryState", state.status_key)
            self.style().unpolish(self)
            self.style().polish(self)
            if state.engineering_status == "FAIL" or "poor_match" in state.reasons:
                style_key = "poor_match"
            elif state.engineering_status == "WARN":
                style_key = "warning"
            else:
                style_key = state.status_key
            self.setStyleSheet(
                "QFrame#butterflyQualitySummary {"
                + {
                    "failed": "background:#fdeceb;color:#7f1d1d;border:1px solid #e0aaa5;",
                    "poor_match": "background:#fff0ee;color:#7f1d1d;border:1px solid #e0aaa5;",
                    "warning": "background:#fff5d6;color:#6b4d00;border:1px solid #e4c878;",
                    "ring_only": "background:#fff5d6;color:#6b4d00;border:1px solid #e4c878;",
                    "ellipse_candidate": "background:#eaf5ec;color:#165b2a;border:1px solid #a9d3b2;",
                    "pending_evaluation": "background:#fff8e6;color:#634d00;border:1px solid #e3cb8f;",
                    "running": "background:#eaf2ff;color:#124e8c;border:1px solid #9fc2ea;",
                    "ready": "background:#eef4f8;color:#173b56;border:1px solid #bfd0dc;",
                    "empty": "background:#f2f4f6;color:#3f4a54;border:1px solid #ccd3d9;",
                }.get(style_key, "background:#f2f4f6;color:#27313a;border:1px solid #ccd3d9;")
                + "border-radius:4px;}"
            )
            for label in (
                self.status_label,
                self.engineering_label,
                self.scientific_label,
                *self._metric_labels,
            ):
                label.updateGeometry()
            self.layout().invalidate()
            self.layout().activate()
            self.updateGeometry()
            self.setAccessibleDescription(
                " | ".join(
                    (
                        status_text,
                        engineering_text,
                        side_text,
                        self.q_star_label.text(),
                        self.l_ring_label.text(),
                        self.unit_label.text(),
                        self.reason_label.text(),
                        self.next_label.text(),
                        self.scientific_label.text(),
                    )
                )
            )

        @staticmethod
        def _ensure_label_width(label: Any) -> None:
            """Keep a one-line readout wide enough for its complete value."""

            try:
                required = int(label.fontMetrics().horizontalAdvance(label.text())) + 8
            except (AttributeError, TypeError, ValueError):
                return
            label.setMinimumWidth(max(0, required))

        def _sync_metric_layout(self) -> None:
            """Use two compact wrapped rows only when the card is genuinely narrow."""

            if not hasattr(self, "_metrics_layout"):
                return
            widths = []
            for label in self._metric_labels:
                try:
                    widths.append(int(label.fontMetrics().horizontalAdvance(label.text())) + 8)
                except (AttributeError, TypeError, ValueError):
                    widths.append(label.minimumWidth())
            required = sum(widths)
            required += max(0, len(self._metric_labels) - 1) * 12 + 20
            top_labels = (self.status_label, self.engineering_label, self.scientific_label)
            top_widths = []
            for label in top_labels:
                try:
                    top_widths.append(int(label.fontMetrics().horizontalAdvance(label.text())) + 8)
                except (AttributeError, TypeError, ValueError):
                    top_widths.append(label.minimumWidth())
            required = max(required, sum(top_widths) + 16)
            parent = self.parentWidget()
            available = parent.width() if parent is not None else self.width()
            # Keep the hidden card soft while its parent is being laid out;
            # otherwise its first long result can establish an oversized
            # minimum width before the real page width is known.
            compact = (not self.isVisible()) or (
                available > 0 and available < max(required, 760)
            )
            for label, width in zip(self._metric_labels, widths):
                label.setWordWrap(compact)
                label.setMinimumWidth(0 if compact else width)
                label.setSizePolicy(
                    QtWidgets.QSizePolicy.Policy.Ignored if compact else QtWidgets.QSizePolicy.Policy.Preferred,
                    QtWidgets.QSizePolicy.Policy.Preferred,
                )
            for label, width in zip(top_labels, top_widths):
                label.setWordWrap(compact)
                label.setMinimumWidth(0 if compact else width)
                label.setSizePolicy(
                    QtWidgets.QSizePolicy.Policy.Ignored if compact else QtWidgets.QSizePolicy.Policy.Preferred,
                    QtWidgets.QSizePolicy.Policy.Preferred,
                )
            self._metrics_layout.setSpacing(6 if compact else 12)
            self.setMinimumHeight(70 if compact else 62)
            self.setMaximumHeight(96 if compact else 88)

else:

    class ButterflyQualitySummary:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs
            require_qt()


__all__ = [
    "ButterflyQualitySummary",
    "ButterflyQualitySummaryState",
    "build_butterfly_quality_summary",
    "summarize_butterfly_result",
]
