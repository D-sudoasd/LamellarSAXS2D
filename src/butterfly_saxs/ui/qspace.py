"""Curvilinear reciprocal-space canvas for the butterfly workbench.

The legacy refinement view uses pyqtgraph's image item for detector pixels and
an extent for its q-space overlay.  That is sufficient for the established
workflow, but it is not a faithful representation of a general 2-D q map.  A
butterfly edit is a q-space operation, so this module draws a small quadrilateral
mesh: every detector sample is painted at the q coordinates supplied for that
sample.  The mesh keeps qx/qy geometry and a one-to-one display aspect without
silently stretching a bounding rectangle.

The widget deliberately owns no analysis state.  It receives plain mappings
from the service and emits serializable edit records for the workbench.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from pathlib import Path
from typing import Any

try:
    import numpy as _np
except Exception:  # pragma: no cover - numpy is a core dependency normally
    _np = None

from .qt_compat import QT_AVAILABLE, QtCore, QtGui, QtWidgets, require_qt


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


def _as_array(value: Any, *, dtype: Any = float) -> Any:
    if _np is None or value is None:
        return None
    try:
        return _np.asarray(value, dtype=dtype)
    except (TypeError, ValueError):
        return None


def _finite_pair(value: Any) -> tuple[float, float] | None:
    try:
        if len(value) != 2:
            return None
        pair = (float(value[0]), float(value[1]))
    except (TypeError, ValueError, IndexError):
        return None
    return pair if all(math.isfinite(item) for item in pair) else None


def _point_q(point: Any) -> tuple[float, float] | None:
    if isinstance(point, Mapping):
        return _finite_pair(
            (
                _read(point, ("qx", "x", "q_x"), None),
                _read(point, ("qy", "y", "q_y"), None),
            )
        )
    return _finite_pair(point)


def _display_values(
    data: Any,
    *,
    scale: str,
    percentile: float,
    level_mask: Any = None,
) -> tuple[Any, tuple[float, float] | None]:
    """Return a display-only array and robust levels; raw values stay untouched."""

    if _np is None or data is None:
        return data, None
    try:
        array = _np.asarray(data, dtype=float)
        mode = str(scale or "linear").strip().lower().replace("-", "_")
        if mode in {"log", "log1p", "signed_log"}:
            shown = _np.sign(array) * _np.log1p(_np.abs(array))
        elif mode in {"asinh", "arcsinh"}:
            shown = _np.arcsinh(array)
        else:
            shown = array
        finite_mask = _np.isfinite(shown)
        if level_mask is not None:
            mask = _np.asarray(level_mask, dtype=bool)
            if mask.shape == shown.shape:
                finite_mask &= mask
        finite = shown[finite_mask]
        if finite.size == 0:
            return shown, None
        upper = min(100.0, max(50.0, float(percentile)))
        lower = min(upper - 0.1, 1.0)
        low, high = _np.percentile(finite, (lower, upper))
        if not _np.isfinite(low) or not _np.isfinite(high) or high <= low:
            low, high = float(finite.min()), float(finite.max())
        if high <= low:
            high = low + 1.0
        return shown, (float(low), float(high))
    except (TypeError, ValueError):
        return data, None


def _colour(value: float, levels: tuple[float, float] | None) -> Any:
    if not QT_AVAILABLE:
        return None
    if levels is None or not math.isfinite(value):
        return QtGui.QColor(35, 35, 42, 0)
    low, high = levels
    fraction = min(1.0, max(0.0, (value - low) / (high - low)))
    # A restrained blue-to-warm map keeps white/black annotations legible.
    if fraction < 0.5:
        t = fraction * 2.0
        return QtGui.QColor(int(20 + 45 * t), int(42 + 110 * t), int(88 + 115 * t), 235)
    t = (fraction - 0.5) * 2.0
    return QtGui.QColor(int(65 + 185 * t), int(152 + 70 * t), int(203 - 120 * t), 235)


if QT_AVAILABLE:

    class QSpaceView(QtWidgets.QWidget):
        """Paint observed data and butterfly overlays in true q-space.

        ``interaction_mode`` is one of ``select_point``, ``seed``,
        ``exclude_point``, ``rectangle_exclude``, ``rectangle_include``,
        ``polygon_exclude`` or ``polygon_include``.  The two polygon modes are
        finalized with :meth:`finish_polygon`; this makes them usable from a
        button or a keyboard shortcut as well as from mouse double-click.
        """

        pointSelected = QtCore.Signal(object)
        editRequested = QtCore.Signal(object)
        roiPreviewChanged = QtCore.Signal(object)
        interactionChanged = QtCore.Signal(str)

        _MAX_CELLS = 60_000

        def __init__(self, parent: Any = None) -> None:
            super().__init__(parent)
            self.setObjectName("qSpaceView")
            # Keep the canvas useful on the supported 980x680 window while
            # allowing the profile diagnostics below it to remain reachable.
            self.setMinimumSize(360, 220)
            self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
            self.setMouseTracking(True)
            self.setAttribute(QtCore.Qt.WidgetAttribute.WA_OpaquePaintEvent, False)
            self._observed: Any = None
            self._qx: Any = None
            self._qy: Any = None
            self._valid_mask: Any = None
            self._butterfly: dict[str, Any] = {}
            self._edits: list[dict[str, Any]] = []
            self._display_scale = "linear"
            self._display_percentile = 99.5
            self._display: Any = None
            self._levels: tuple[float, float] | None = None
            self._q_bounds: tuple[float, float, float, float] | None = None
            self._base_q_bounds: tuple[float, float, float, float] | None = None
            self._q_window: tuple[float, float] | None = None
            self._q_unit = "q"
            self._language = "zh_CN"
            self._compact_mode = False
            self._empty_message_override: str | None = None
            self._plot_rect = QtCore.QRectF()
            self._interaction_mode = "select_point"
            self._draft_points: list[tuple[float, float]] = []
            self._selected_point_id: str | None = None
            self._hover_q: tuple[float, float] | None = None
            self._mesh_revision = 0
            self._mesh_cache_key: tuple[Any, ...] | None = None
            self._mesh_cache_image: Any = None
            self._point_by_id: dict[str, Mapping[str, Any]] = {}
            self._arc_segments_cache: dict[str, list[dict[str, Any]]] = {}
            self._butterfly_revision = 0
            self._visible_branches: dict[tuple[int, str], bool] = {
                (0, "upper"): True,
                (0, "lower"): True,
                (1, "upper"): True,
                (1, "lower"): True,
                (-1, "unknown"): True,
            }
            self._show_excluded = False
            self._branch_colors = ((42, 154, 220), (239, 143, 44))
            self._custom_branch_colors = False
            self.setAccessibleName("Butterfly reciprocal-space image")
            self.setAccessibleDescription(
                "Curvilinear q-space image with selectable butterfly points and editable regions"
            )

        @property
        def interaction_mode(self) -> str:
            return self._interaction_mode

        @property
        def selected_point_id(self) -> str | None:
            return self._selected_point_id

        @property
        def q_bounds(self) -> tuple[float, float, float, float] | None:
            return self._q_bounds

        @property
        def q_window(self) -> tuple[float, float] | None:
            return self._q_window

        @property
        def observed(self) -> Any:
            return self._observed

        @property
        def qx(self) -> Any:
            return self._qx

        @property
        def qy(self) -> Any:
            return self._qy

        def set_display_settings(self, scale: str = "linear", percentile: float = 99.5) -> None:
            mode = str(scale or "linear").strip().lower().replace("-", "_")
            self._display_scale = mode if mode in {"linear", "log1p", "asinh"} else "linear"
            try:
                self._display_percentile = min(100.0, max(50.0, float(percentile)))
            except (TypeError, ValueError):
                self._display_percentile = 99.5
            self._prepare_display()
            self._mesh_revision += 1
            self._mesh_cache_key = None
            self.update()

        def _prepare_display(self) -> None:
            """Rebuild only the painted contrast from the retained raw frame."""

            if self._observed is None:
                self._display = None
                self._levels = None
                return
            values = self._observed.copy()
            if self._valid_mask is not None:
                values[~self._valid_mask] = _np.nan
            self._display, self._levels = _display_values(
                values,
                scale=self._display_scale,
                percentile=self._display_percentile,
                level_mask=self._active_render_mask(),
            )

        def set_data(
            self,
            observed: Any = None,
            *,
            qx: Any = None,
            qy: Any = None,
            valid_mask: Any = None,
            q_unit: str | None = None,
            display_scale: str | None = None,
            display_percentile: float | None = None,
        ) -> None:
            self._q_unit = str(q_unit or "q")
            if display_scale is not None or display_percentile is not None:
                self.set_display_settings(
                    self._display_scale if display_scale is None else display_scale,
                    self._display_percentile if display_percentile is None else display_percentile,
                )
            self._observed = _as_array(observed, dtype=float)
            if self._observed is None or self._observed.ndim != 2:
                self._observed = None
                self._qx = self._qy = self._valid_mask = None
                self._q_bounds = None
                self._base_q_bounds = None
                self._display = None
                self._mesh_revision += 1
                self._mesh_cache_key = None
                self.update()
                return
            shape = self._observed.shape
            self._qx, self._qy = self._normalise_qmap(qx, qy, shape)
            valid = _as_array(valid_mask, dtype=bool)
            self._valid_mask = valid if valid is not None and valid.shape == shape else _np.isfinite(self._observed)
            self._valid_mask = self._valid_mask & _np.isfinite(self._qx) & _np.isfinite(self._qy)
            values = self._observed.copy()
            values[~self._valid_mask] = _np.nan
            self._observed = self._observed  # preserve the raw array reference/value
            self._q_bounds = self._compute_bounds(self._qx, self._qy, self._valid_mask)
            self._base_q_bounds = self._q_bounds
            if self._q_window is not None:
                self._apply_q_window()
            active_mask = self._active_render_mask()
            self._display, self._levels = _display_values(
                values,
                scale=self._display_scale,
                percentile=self._display_percentile,
                level_mask=active_mask,
            )
            self._draft_points.clear()
            self._selected_point_id = None
            self._mesh_revision += 1
            self._mesh_cache_key = None
            self.update()

        def set_q_window(self, q_window: Sequence[Any] | None = None) -> None:
            """Focus the mesh on a radial q interval without affine warping."""

            if q_window is None:
                self._q_window = None
                self._q_bounds = self._base_q_bounds
            else:
                try:
                    low, high = (float(value) for value in q_window)
                    if not math.isfinite(low) or not math.isfinite(high) or high <= low:
                        return
                    self._q_window = (low, high)
                    self._apply_q_window()
                except (TypeError, ValueError):
                    return
            self._mesh_revision += 1
            self._mesh_cache_key = None
            self.update()

        def _apply_q_window(self) -> None:
            if self._qx is None or self._qy is None or self._valid_mask is None or self._base_q_bounds is None:
                return
            if self._q_window is None:
                self._q_bounds = self._base_q_bounds
                return
            radius = _np.hypot(self._qx, self._qy)
            low, high = self._q_window
            selected = self._valid_mask & _np.isfinite(radius) & (radius >= low) & (radius <= high)
            if _np.any(selected):
                self._q_bounds = self._compute_bounds(self._qx, self._qy, selected)

        def _active_render_mask(self) -> Any:
            if self._valid_mask is None:
                return None
            if self._q_window is None or self._qx is None or self._qy is None:
                return self._valid_mask
            radius = _np.hypot(self._qx, self._qy)
            low, high = self._q_window
            return self._valid_mask & _np.isfinite(radius) & (radius >= low) & (radius <= high)

        set_images = set_data

        def set_language(self, language: str) -> None:
            self._language = str(language or "zh_CN")
            self.update()

        def _empty_state_message(self) -> str:
            if self._empty_message_override:
                return self._empty_message_override
            return (
                "Load a frame to view reciprocal space"
                if self._language.lower().startswith("en")
                else "请先载入图像以查看倒易空间"
            )

        def set_empty_message(self, message: str | None) -> None:
            self._empty_message_override = message
            self.update()

        @staticmethod
        def _normalise_qmap(qx: Any, qy: Any, shape: tuple[int, int]) -> tuple[Any, Any]:
            if _np is None:
                return qx, qy
            rows, cols = shape
            x = _as_array(qx, dtype=float)
            y = _as_array(qy, dtype=float)
            if x is not None and y is not None:
                if x.ndim == 1 and y.ndim == 1 and x.size == cols and y.size == rows:
                    return _np.meshgrid(x, y)
                if x.shape == shape and y.shape == shape:
                    return x, y
            detector_y, detector_x = _np.indices(shape, dtype=float)
            return detector_x, detector_y

        @staticmethod
        def _compute_bounds(qx: Any, qy: Any, valid: Any) -> tuple[float, float, float, float] | None:
            try:
                finite = _np.asarray(valid, dtype=bool) & _np.isfinite(qx) & _np.isfinite(qy)
                if not _np.any(finite):
                    return None
                x = _np.asarray(qx)[finite]
                y = _np.asarray(qy)[finite]
                return float(x.min()), float(x.max()), float(y.min()), float(y.max())
            except (TypeError, ValueError):
                return None

        def set_butterfly(self, payload: Any = None) -> None:
            self._butterfly = dict(payload) if isinstance(payload, Mapping) else {}
            points = self._butterfly.get("points", [])
            if isinstance(points, Mapping):
                points = [points]
            self._point_by_id = {
                str(_read(point, ("point_id",), "")): point
                for point in (points or ())
                if isinstance(point, Mapping) and _read(point, ("point_id",), None) not in (None, "")
            }
            self._arc_segments_cache = {}
            self._butterfly_revision += 1
            edits = self._butterfly.get("edits")
            if isinstance(edits, list):
                self.set_edits(edits)
            if self._selected_point_id is not None and not any(
                str(_read(item, ("point_id",), "")) == self._selected_point_id for item in points
            ):
                self._selected_point_id = None
            self.update()

        set_overlay = set_butterfly

        def set_edits(self, edits: Sequence[Any] | None) -> None:
            """Render the persisted include/exclude polygons on the q image."""

            self._edits = [dict(edit) for edit in (edits or ()) if isinstance(edit, Mapping)]
            self.update()

        @property
        def edits(self) -> list[dict[str, Any]]:
            return [dict(edit) for edit in self._edits]

        def set_visible_branch(self, branch_id: int, side: str, visible: bool) -> None:
            self._visible_branches[(int(branch_id), str(side or "unknown"))] = bool(visible)
            self.update()

        def set_visible_branches(self, visibility: Mapping[Any, Any]) -> None:
            for key, value in visibility.items():
                if isinstance(key, tuple) and len(key) == 2:
                    self.set_visible_branch(int(key[0]), str(key[1]), bool(value))
                elif isinstance(value, Mapping):
                    for side, enabled in value.items():
                        self.set_visible_branch(int(key), str(side), bool(enabled))

        def set_show_excluded(self, visible: bool) -> None:
            self._show_excluded = bool(visible)
            self.update()

        def set_selected_point(self, point_id: Any) -> None:
            self._selected_point_id = None if point_id in (None, "") else str(point_id)
            self.update()

        def set_interaction_mode(self, mode: str) -> None:
            allowed = {
                "select_point",
                "seed",
                "exclude_point",
                "rectangle_exclude",
                "rectangle_include",
                "polygon_exclude",
                "polygon_include",
            }
            normalized = str(mode or "select_point").strip().lower()
            if normalized not in allowed:
                normalized = "select_point"
            self._draft_points.clear()
            self._interaction_mode = normalized
            self.interactionChanged.emit(normalized)
            self.update()

        def clear_draft(self) -> None:
            self._draft_points.clear()
            self.roiPreviewChanged.emit(None)
            self.update()

        def _plot_geometry(self) -> None:
            bounds = self._q_bounds
            if bounds is None:
                self._plot_rect = QtCore.QRectF()
                return
            x0, x1, y0, y1 = bounds
            span_x = max(1e-12, x1 - x0)
            span_y = max(1e-12, y1 - y0)
            margin = 28.0
            available = (self.rect().adjusted(42, 12, -12, -34) if self._compact_mode
                         else self.rect().adjusted(margin, margin, -margin, -margin))
            if available.width() <= 1 or available.height() <= 1:
                self._plot_rect = QtCore.QRectF()
                return
            scale = min(available.width() / span_x, available.height() / span_y)
            width = span_x * scale
            height = span_y * scale
            self._plot_rect = QtCore.QRectF(
                available.center().x() - width / 2.0,
                available.center().y() - height / 2.0,
                width,
                height,
            )

        def set_compact_mode(self, compact: bool = True) -> None:
            """Use sparse, non-overlapping axes in a small read-only panel."""
            self._compact_mode = bool(compact)
            self._mesh_cache_key = None
            self.update()

        def set_branch_colors(self, colors: Any) -> None:
            """Optional display palette; measurement/branch identities are unchanged."""
            if _np is None:
                raise RuntimeError("NumPy is required for display palettes")
            values = _np.asarray(colors, dtype=float)
            if values.shape not in {(2, 3), (2, 4)} or not _np.all(_np.isfinite(values)):
                raise ValueError("branch colors must contain two finite RGB or RGBA colors")
            self._branch_colors = tuple(tuple(int(round(v * 255)) for v in _np.clip(row[:3], 0, 1)) for row in values)
            self._custom_branch_colors = True
            self.update()

        def _branch_color(self, branch: int, alpha: int = 255) -> Any:
            return QtGui.QColor(*self._branch_colors[int(branch) % 2], alpha)

        def _q_to_screen(self, qx: float, qy: float) -> QtCore.QPointF | None:
            if self._q_bounds is None or self._plot_rect.isNull():
                return None
            x0, x1, y0, y1 = self._q_bounds
            if not all(math.isfinite(v) for v in (qx, qy)):
                return None
            sx = self._plot_rect.left() + (qx - x0) / max(1e-12, x1 - x0) * self._plot_rect.width()
            # Detector/q arrays follow image convention (row down); qy itself
            # remains physically signed, so the display maps larger qy upward.
            sy = self._plot_rect.bottom() - (qy - y0) / max(1e-12, y1 - y0) * self._plot_rect.height()
            return QtCore.QPointF(sx, sy)

        def q_to_widget(self, qx: float, qy: float) -> QtCore.QPoint:
            """Map a q coordinate to a widget pixel for keyboard/test clients."""

            self._plot_geometry()
            point = self._q_to_screen(float(qx), float(qy))
            if point is None:
                return QtCore.QPoint()
            return QtCore.QPoint(round(point.x()), round(point.y()))

        def _screen_to_q(self, point: QtCore.QPointF) -> tuple[float, float] | None:
            if self._q_bounds is None or self._plot_rect.isNull() or not self._plot_rect.contains(point):
                return None
            x0, x1, y0, y1 = self._q_bounds
            qx = x0 + (point.x() - self._plot_rect.left()) / self._plot_rect.width() * (x1 - x0)
            qy = y0 + (self._plot_rect.bottom() - point.y()) / self._plot_rect.height() * (y1 - y0)
            return float(qx), float(qy)

        def _nearest_point(self, q: tuple[float, float]) -> Mapping[str, Any] | None:
            points = self._butterfly.get("points", [])
            if not isinstance(points, Sequence) or isinstance(points, (str, bytes)):
                return None
            best: Mapping[str, Any] | None = None
            best_distance = float("inf")
            span = 0.05
            if self._q_bounds is not None:
                span = max(self._q_bounds[1] - self._q_bounds[0], self._q_bounds[3] - self._q_bounds[2]) * 0.04
            for item in points:
                if not isinstance(item, Mapping):
                    continue
                candidate = _point_q(item)
                if candidate is None:
                    continue
                distance = math.hypot(candidate[0] - q[0], candidate[1] - q[1])
                if distance < best_distance:
                    best_distance, best = distance, item
            return best if best is not None and best_distance <= max(span, 1e-12) else None

        def _edit_for_point(self, point: Mapping[str, Any]) -> dict[str, Any] | None:
            point_id = _read(point, ("point_id",), None)
            if point_id in (None, ""):
                return None
            return {"type": "exclude_point", "point_id": str(point_id)}

        def mousePressEvent(self, event: Any) -> None:  # noqa: N802 - Qt API
            if event.button() != QtCore.Qt.MouseButton.LeftButton:
                super().mousePressEvent(event)
                return
            q = self._screen_to_q(event.position())
            if q is None:
                return
            mode = self._interaction_mode
            if mode == "select_point":
                point = self._nearest_point(q)
                if point is not None:
                    self._selected_point_id = str(_read(point, ("point_id",), "") or "") or None
                    self.pointSelected.emit(dict(point))
                    self.update()
                return
            if mode == "seed":
                self.editRequested.emit({"type": "seed", "qx": q[0], "qy": q[1]})
                return
            if mode == "exclude_point":
                point = self._nearest_point(q)
                if point is not None:
                    edit = self._edit_for_point(point)
                    if edit is not None:
                        self.editRequested.emit(edit)
                return
            if mode.startswith("rectangle"):
                self._draft_points = [q]
                self.roiPreviewChanged.emit({"type": "rectangle", "start": q, "end": q, "action": mode.removeprefix("rectangle_")})
                self.update()
                return
            if mode.startswith("polygon"):
                self._draft_points.append(q)
                self.roiPreviewChanged.emit({"type": mode.removeprefix("polygon_"), "points": list(self._draft_points)})
                self.update()

        def mouseMoveEvent(self, event: Any) -> None:  # noqa: N802 - Qt API
            self._hover_q = self._screen_to_q(event.position())
            self.update()
            if self._interaction_mode.startswith("rectangle") and self._draft_points:
                q = self._screen_to_q(event.position())
                if q is not None:
                    self.roiPreviewChanged.emit({
                        "type": "rectangle",
                        "start": self._draft_points[0],
                        "end": q,
                        "action": self._interaction_mode.removeprefix("rectangle_"),
                    })
                    self.update()
            super().mouseMoveEvent(event)

        def mouseReleaseEvent(self, event: Any) -> None:  # noqa: N802 - Qt API
            if event.button() == QtCore.Qt.MouseButton.LeftButton and self._interaction_mode.startswith("rectangle"):
                q = self._screen_to_q(event.position())
                if q is not None and self._draft_points:
                    start = self._draft_points[0]
                    x0, x1 = sorted((start[0], q[0]))
                    y0, y1 = sorted((start[1], q[1]))
                    if x1 > x0 and y1 > y0:
                        action = self._interaction_mode.removeprefix("rectangle_")
                        self.editRequested.emit({
                            "type": f"{action}_polygon",
                            "points": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
                        })
                self.clear_draft()
            super().mouseReleaseEvent(event)

        def mouseDoubleClickEvent(self, event: Any) -> None:  # noqa: N802 - Qt API
            if event.button() == QtCore.Qt.MouseButton.LeftButton and self._interaction_mode.startswith("polygon"):
                self.finish_polygon()
                return
            super().mouseDoubleClickEvent(event)

        def finish_polygon(self) -> None:
            if not self._interaction_mode.startswith("polygon") or len(self._draft_points) < 3:
                return
            action = self._interaction_mode.removeprefix("polygon_")
            self.editRequested.emit({
                "type": f"{action}_polygon",
                "points": [[float(x), float(y)] for x, y in self._draft_points],
            })
            self.clear_draft()

        def keyPressEvent(self, event: Any) -> None:  # noqa: N802 - Qt API
            if event.key() == QtCore.Qt.Key.Key_Escape:
                self.clear_draft()
                self.set_interaction_mode("select_point")
                return
            super().keyPressEvent(event)

        def _corner(self, row: int, col: int) -> tuple[float, float] | None:
            """Interpolate one mesh corner from the surrounding q centres."""

            if self._qx is None or self._qy is None:
                return None
            rows, cols = self._qx.shape
            candidates: list[tuple[float, float]] = []
            for rr in (row - 1, row):
                for cc in (col - 1, col):
                    if 0 <= rr < rows and 0 <= cc < cols:
                        pair = _finite_pair((self._qx[rr, cc], self._qy[rr, cc]))
                        if pair is not None:
                            candidates.append(pair)
            if not candidates:
                return None
            return (
                sum(pair[0] for pair in candidates) / len(candidates),
                sum(pair[1] for pair in candidates) / len(candidates),
            )

        def _mesh_cache_signature(self) -> tuple[Any, ...]:
            rect = self._plot_rect
            rect_key = (
                round(float(rect.left()), 3),
                round(float(rect.top()), 3),
                round(float(rect.width()), 3),
                round(float(rect.height()), 3),
            )
            bounds = tuple(round(float(value), 12) for value in (self._q_bounds or ()))
            levels = tuple(round(float(value), 12) for value in (self._levels or ()))
            return (self._mesh_revision, self.width(), self.height(), rect_key, bounds, levels)

        def _build_mesh_image(self) -> Any:
            if self._display is None or self._qx is None or self._qy is None:
                return None
            rows, cols = self._display.shape
            active = self._active_render_mask()
            if active is None or not _np.any(active):
                return None
            image = QtGui.QImage(
                max(1, self.width()),
                max(1, self.height()),
                QtGui.QImage.Format.Format_ARGB32_Premultiplied,
            )
            image.fill(QtCore.Qt.GlobalColor.transparent)
            mesh_painter = QtGui.QPainter(image)
            mesh_painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, False)
            active_rows, active_cols = _np.nonzero(active)
            visible_count = int(active_rows.size)
            # Decimate only after the active q-window crop is known.  The
            # resulting image is reused for pointer-only repaints.
            target_cells = max(1, min(self._MAX_CELLS, int(self.width() * self.height() * 0.45)))
            stride = max(1, int(math.ceil(math.sqrt(visible_count / target_cells))))
            row_start, row_stop = int(active_rows.min()), int(active_rows.max()) + 1
            col_start, col_stop = int(active_cols.min()), int(active_cols.max()) + 1
            for row in range(row_start, row_stop, stride):
                for col in range(col_start, col_stop, stride):
                    if not bool(active[row, col]):
                        continue
                    value = float(self._display[row, col])
                    if not math.isfinite(value):
                        continue
                    corners = [
                        self._corner(row, col),
                        self._corner(min(rows, row + stride), col),
                        self._corner(min(rows, row + stride), min(cols, col + stride)),
                        self._corner(row, min(cols, col + stride)),
                    ]
                    screen = [self._q_to_screen(*pair) for pair in corners if pair is not None]
                    if len(screen) != 4 or any(point is None for point in screen):
                        continue
                    mesh_painter.setPen(QtCore.Qt.PenStyle.NoPen)
                    mesh_painter.setBrush(_colour(value, self._levels))
                    mesh_painter.drawPolygon(QtGui.QPolygonF(screen))  # type: ignore[arg-type]
            mesh_painter.end()
            return image

        def _draw_mesh(self, painter: Any) -> None:
            key = self._mesh_cache_signature()
            if key != self._mesh_cache_key:
                self._mesh_cache_image = self._build_mesh_image()
                self._mesh_cache_key = key
            if self._mesh_cache_image is not None:
                painter.drawImage(0, 0, self._mesh_cache_image)

        def _point_visible(self, point: Mapping[str, Any]) -> bool:
            if not bool(_read(point, ("valid",), True)) or not bool(_read(point, ("accepted",), True)):
                return self._show_excluded
            branch = _read(point, ("branch_id",), -1)
            side = str(_read(point, ("side",), "unknown") or "unknown").lower()
            try:
                branch = int(branch)
            except (TypeError, ValueError):
                branch = -1
            return self._visible_branches.get((branch, side), self._visible_branches.get((branch, "unknown"), True))

        def _arc_visible(self, arc: Mapping[str, Any]) -> bool:
            if not bool(_read(arc, ("valid",), True)) or not bool(_read(arc, ("accepted",), True)):
                return self._show_excluded
            branch = _read(arc, ("branch_id", "branch", "component"), -1)
            side = str(_read(arc, ("side",), "unknown") or "unknown").lower()
            try:
                branch = int(branch)
            except (TypeError, ValueError):
                branch = -1
            return self._visible_branches.get(
                (branch, side), self._visible_branches.get((branch, "unknown"), True)
            )

        @staticmethod
        def _arc_is_extrapolated(arc: Mapping[str, Any]) -> bool:
            return bool(_read(arc, ("extrapolated", "is_extrapolated"), False)) or (
                _read(arc, ("supported",), None) is False
            )

        def _arc_gap_threshold(
            self,
            arc: Mapping[str, Any],
            points: Sequence[Mapping[str, Any]],
        ) -> float | None:
            """Return the recorded local q support threshold for an arc."""

            explicit = _read(
                arc,
                ("max_gap_q", "support_gap_q", "local_support_q", "q_step", "q_normal_step"),
                None,
            )
            candidates: list[float] = []
            if explicit is not None:
                try:
                    value = float(explicit)
                    if math.isfinite(value) and value > 0.0:
                        candidates.append(value)
                except (TypeError, ValueError):
                    pass
            for point in points:
                for key in ("q_normal_step", "q_step", "local_step_q", "support_q_step"):
                    try:
                        value = float(_read(point, (key,), None))
                    except (TypeError, ValueError):
                        continue
                    if math.isfinite(value) and value > 0.0:
                        candidates.append(value)
            if not candidates:
                return None
            local_step = float(_np.median(candidates)) if _np is not None else sorted(candidates)[len(candidates) // 2]
            try:
                factor = float(_read(arc, ("gap_factor", "support_gap_factor"), 3.5))
            except (TypeError, ValueError):
                factor = 3.5
            return local_step * max(1.0, factor)

        def _path_crosses_invalid(
            self,
            first: Mapping[str, Any],
            second: Mapping[str, Any],
        ) -> bool:
            """Detect a detector-mask hole between two retained points."""

            if _np is None or self._valid_mask is None:
                return False
            shape = getattr(self._valid_mask, "shape", ())
            if len(shape) != 2:
                return False
            try:
                x0, y0 = float(_read(first, ("pixel_x", "x_pixel", "col"), math.nan)), float(
                    _read(first, ("pixel_y", "y_pixel", "row"), math.nan)
                )
                x1, y1 = float(_read(second, ("pixel_x", "x_pixel", "col"), math.nan)), float(
                    _read(second, ("pixel_y", "y_pixel", "row"), math.nan)
                )
            except (TypeError, ValueError):
                return False
            if not all(math.isfinite(value) for value in (x0, y0, x1, y1)):
                return False
            count = max(2, int(math.ceil(math.hypot(x1 - x0, y1 - y0))) + 1)
            rows, cols = shape
            for fraction in _np.linspace(0.0, 1.0, count):
                col = int(round(x0 + fraction * (x1 - x0)))
                row = int(round(y0 + fraction * (y1 - y0)))
                if not (0 <= row < rows and 0 <= col < cols) or not bool(self._valid_mask[row, col]):
                    return True
            return False

        def _resolve_arc_segments(self, arc: Mapping[str, Any]) -> list[dict[str, Any]]:
            """Resolve real arc IDs into contiguous, identity-consistent q paths."""

            cache_key = str(_read(arc, ("arc_id", "id"), id(arc)))
            cached = self._arc_segments_cache.get(cache_key)
            if cached is not None:
                return cached
            ordered_ids = _read(arc, ("ordered_point_ids", "point_ids"), None)
            if not isinstance(ordered_ids, Sequence) or isinstance(ordered_ids, (str, bytes)):
                return []
            branch_hints = set()
            for value in (_read(arc, ("branch_ids",), ()) or ()):
                try:
                    branch_hints.add(int(value))
                except (TypeError, ValueError):
                    pass
            side_hints = {
                str(value).lower()
                for value in (_read(arc, ("sides",), ()) or ())
                if str(value).lower() in {"upper", "lower"}
            }
            ordered_points = [self._point_by_id.get(str(raw_id)) for raw_id in ordered_ids]
            valid_points = [point for point in ordered_points if isinstance(point, Mapping)]
            gap_threshold = self._arc_gap_threshold(arc, valid_points)
            segments: list[dict[str, Any]] = []
            current: dict[str, Any] | None = None
            previous_point: Mapping[str, Any] | None = None
            previous_q: tuple[float, float] | None = None
            pending_break_reason: str | None = None

            def flush() -> None:
                nonlocal current, previous_point, previous_q
                if current is not None and len(current["points"]) >= 2:
                    segments.append(current)
                current = None
                previous_point = None
                previous_q = None

            for raw_id in ordered_ids:
                point = self._point_by_id.get(str(raw_id))
                if point is None:
                    pending_break_reason = "missing_point"
                    flush()
                    continue
                if not bool(_read(point, ("valid",), True)) or not bool(_read(point, ("accepted",), True)):
                    pending_break_reason = "invalid_or_rejected"
                    flush()
                    continue
                q = _point_q(point)
                side = str(_read(point, ("side",), "unknown") or "unknown").lower()
                try:
                    branch = int(_read(point, ("branch_id",), -1))
                except (TypeError, ValueError):
                    branch = -1
                if q is None or branch not in (0, 1) or side not in {"upper", "lower"}:
                    pending_break_reason = "invalid_identity"
                    flush()
                    continue
                if branch_hints and branch not in branch_hints:
                    pending_break_reason = "branch_mismatch"
                    flush()
                    continue
                if side_hints and side not in side_hints:
                    pending_break_reason = "side_mismatch"
                    flush()
                    continue
                if previous_q is not None:
                    distance = math.hypot(q[0] - previous_q[0], q[1] - previous_q[1])
                    if (gap_threshold is not None and distance > gap_threshold) or (
                        previous_point is not None and self._path_crosses_invalid(previous_point, point)
                    ):
                        pending_break_reason = (
                            "q_gap"
                            if gap_threshold is not None and distance > gap_threshold
                            else "masked_path"
                        )
                        flush()
                identity = (branch, side)
                if current is None or current["identity"] != identity:
                    flush()
                    current = {
                        "identity": identity,
                        "branch_id": branch,
                        "side": side,
                        "points": [],
                        "point_ids": [],
                        "break_reason": pending_break_reason,
                    }
                    pending_break_reason = None
                current["points"].append(q)
                current["point_ids"].append(str(raw_id))
                previous_point = point
                previous_q = q
            flush()
            self._arc_segments_cache[cache_key] = segments
            return segments

        def _draw_butterfly(self, painter: Any) -> None:
            painter.save()
            if not self._plot_rect.isNull():
                painter.setClipRect(self._plot_rect, QtCore.Qt.ClipOperation.IntersectClip)
            points = self._butterfly.get("points", [])
            if isinstance(points, Mapping):
                points = [points]
            for point in points if isinstance(points, Sequence) else ():
                if not isinstance(point, Mapping):
                    continue
                q = _point_q(point)
                if q is None or not self._point_visible(point):
                    continue
                screen = self._q_to_screen(*q)
                if screen is None:
                    continue
                valid = bool(_read(point, ("valid", "accepted"), True))
                accepted = bool(_read(point, ("accepted", "valid"), valid))
                branch = _read(point, ("branch_id",), -1)
                side = str(_read(point, ("side",), "unknown") or "unknown").lower()
                try:
                    branch = int(branch)
                except (TypeError, ValueError):
                    branch = -1
                if not valid or not accepted:
                    pen = QtGui.QPen(QtGui.QColor(150, 150, 155, 135), 1.0)
                    painter.setPen(pen)
                    painter.drawLine(screen.x() - 3, screen.y() - 3, screen.x() + 3, screen.y() + 3)
                    painter.drawLine(screen.x() - 3, screen.y() + 3, screen.x() + 3, screen.y() - 3)
                    continue
                upper = side == "upper"
                if branch == 0:
                    color = self._branch_color(0)
                elif branch == 1:
                    color = self._branch_color(1)
                else:
                    color = QtGui.QColor(180, 180, 185, 255)
                painter.setPen(QtGui.QPen(color, 1.5))
                painter.setBrush(QtGui.QBrush(color))
                if side == "unknown" or branch not in (0, 1):
                    painter.drawEllipse(screen, 4.0, 4.0)
                elif upper:
                    painter.drawEllipse(screen, 4.5, 4.5)
                else:
                    painter.drawRect(QtCore.QRectF(screen.x() - 4.5, screen.y() - 4.5, 9.0, 9.0))
                if str(_read(point, ("point_id",), "")) == self._selected_point_id:
                    painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
                    painter.setPen(QtGui.QPen(QtGui.QColor(255, 245, 120, 255), 2.0))
                    painter.drawEllipse(screen, 8.0, 8.0)

            arcs = self._butterfly.get("arcs", [])
            if isinstance(arcs, Mapping):
                arcs = list(arcs.values())
            for arc in arcs if isinstance(arcs, Sequence) else ():
                if not isinstance(arc, Mapping):
                    continue
                # Visibility is an arc-level contract.  Do this before
                # resolving ordered IDs so hidden/invalid arcs cannot paint
                # through their individually valid points.
                if not self._arc_visible(arc):
                    continue
                extrapolated = self._arc_is_extrapolated(arc)
                segments = self._resolve_arc_segments(arc)
                if segments:
                    for segment in segments:
                        branch = int(segment["branch_id"])
                        side = str(segment["side"])
                        if not self._visible_branches.get((branch, side), True):
                            continue
                        screen_points = [self._q_to_screen(*pair) for pair in segment["points"]]
                        color = self._branch_color(branch, 190)
                        dashed = extrapolated or bool(segment.get("break_reason"))
                        painter.setPen(QtGui.QPen(color, 2.0, QtCore.Qt.PenStyle.DashLine if dashed else QtCore.Qt.PenStyle.SolidLine))
                        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
                        self._draw_polyline_segments(painter, screen_points)
                    continue
                raw_points = _read(arc, ("points", "vertices", "path"), None)
                if raw_points is None:
                    qxs = _read(arc, ("qx", "q_x"), [])
                    qys = _read(arc, ("qy", "q_y"), [])
                    raw_points = list(zip(qxs or [], qys or []))
                raw_points = list(raw_points or ())
                gap_threshold = self._arc_gap_threshold(
                    arc,
                    [item for item in raw_points if isinstance(item, Mapping)],
                )
                screen_points: list[Any] = []
                previous_q: tuple[float, float] | None = None
                previous_point: Mapping[str, Any] | None = None
                for item in raw_points:
                    if isinstance(item, Mapping) and (
                        not bool(_read(item, ("valid",), True))
                        or not bool(_read(item, ("accepted",), True))
                    ):
                        screen_points.append(None)
                        previous_q = None
                        previous_point = None
                        continue
                    pair = _point_q(item)
                    if pair is None:
                        screen_points.append(None)
                        previous_q = None
                        previous_point = None
                        continue
                    if previous_q is not None and gap_threshold is not None and math.hypot(
                        pair[0] - previous_q[0], pair[1] - previous_q[1]
                    ) > gap_threshold:
                        screen_points.append(None)
                    if previous_point is not None and isinstance(item, Mapping) and self._path_crosses_invalid(
                        previous_point, item
                    ):
                        screen_points.append(None)
                    screen_points.append(self._q_to_screen(*pair))
                    previous_q = pair
                    previous_point = item if isinstance(item, Mapping) else None
                color = self._branch_color(0 if str(_read(arc, ("side",), "upper")) == "upper" else 1, 190)
                painter.setPen(QtGui.QPen(color, 2.0, QtCore.Qt.PenStyle.DashLine if extrapolated else QtCore.Qt.PenStyle.SolidLine))
                painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
                self._draw_polyline_segments(painter, screen_points)

            fit = self._butterfly.get("candidate_fit", {})
            ellipses = _read(fit, ("ellipses", "ellipse_pair"), [])
            if isinstance(ellipses, Mapping):
                ellipses = [ellipses]
            for ellipse in ellipses if isinstance(ellipses, Sequence) else ():
                if not isinstance(ellipse, Mapping):
                    continue
                raw_branch = _read(ellipse, ("branch_id", "branch", "component"), -1)
                try:
                    branch = int(raw_branch)
                except (TypeError, ValueError):
                    branch = -1
                side = str(_read(ellipse, ("side",), "") or "").lower()
                visible = (
                    self._visible_branches.get((branch, side), True)
                    if side in {"upper", "lower"}
                    else any(self._visible_branches.get((branch, item), True) for item in ("upper", "lower"))
                )
                if not visible:
                    continue
                raw_curve = _read(ellipse, ("points", "path", "vertices"), None)
                screen_points: list[Any] = []
                if raw_curve is not None:
                    for item in raw_curve or ():
                        if isinstance(item, Mapping):
                            item_side = str(item.get("side", "") or "").lower()
                            if item_side in {"upper", "lower"} and not self._visible_branches.get(
                                (branch, item_side), True
                            ):
                                screen_points.append(None)
                                continue
                        pair = _point_q(item)
                        screen_points.append(self._q_to_screen(*pair) if pair is not None else None)
                else:
                    center = _read(ellipse, ("center", "centre"), None)
                    center_pair = _finite_pair(center) if center is not None else _finite_pair(
                        (_read(ellipse, ("cx",), 0.0), _read(ellipse, ("cy",), 0.0))
                    )
                    if center_pair is None:
                        continue
                    try:
                        a = float(_read(ellipse, ("a", "semi_major"), 0.0))
                        b = float(_read(ellipse, ("b", "semi_minor"), 0.0))
                        angle = math.radians(float(_read(ellipse, ("angle_deg", "theta_deg"), 0.0)))
                    except (TypeError, ValueError):
                        continue
                    if a <= 0.0 or b <= 0.0:
                        continue
                    if self._q_bounds is not None:
                        q_span = max(
                            self._q_bounds[1] - self._q_bounds[0],
                            self._q_bounds[3] - self._q_bounds[2],
                        )
                        if a > 2.0 * q_span or b > 2.0 * q_span:
                            continue
                    ca, sa = math.cos(angle), math.sin(angle)
                    for step in range(121):
                        phi = 2.0 * math.pi * step / 120.0
                        ex = a * math.cos(phi)
                        ey = b * math.sin(phi)
                        pair = (
                            center_pair[0] + ex * ca - ey * sa,
                            center_pair[1] + ex * sa + ey * ca,
                        )
                        screen_points.append(self._q_to_screen(*pair))
                color = (
                    QtGui.QColor(100, 195, 240, 175)
                    if branch == 0
                    else QtGui.QColor(250, 175, 95, 175)
                )
                if self._custom_branch_colors:
                    color = self._branch_color(branch, 175)
                painter.setPen(QtGui.QPen(color, 1.2, QtCore.Qt.PenStyle.DashLine))
                painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
                self._draw_polyline_segments(painter, screen_points)

            painter.restore()

        def _draw_edits(self, painter: Any) -> None:
            for edit in self._edits:
                kind = str(edit.get("type", ""))
                if kind not in {"exclude_polygon", "include_polygon"}:
                    continue
                vertices = []
                for item in edit.get("points", ()):
                    pair = _finite_pair(item)
                    screen = self._q_to_screen(*pair) if pair is not None else None
                    if screen is not None:
                        vertices.append(screen)
                if len(vertices) < 3:
                    continue
                include = kind == "include_polygon"
                color = QtGui.QColor(72, 213, 154, 220) if include else QtGui.QColor(242, 104, 104, 235)
                painter.setPen(QtGui.QPen(color, 2.0, QtCore.Qt.PenStyle.SolidLine if include else QtCore.Qt.PenStyle.DashLine))
                painter.setBrush(QtGui.QBrush(QtGui.QColor(color.red(), color.green(), color.blue(), 28)))
                painter.drawPolygon(QtGui.QPolygonF(vertices))


        @staticmethod
        def _draw_polyline_segments(painter: Any, points: Sequence[Any]) -> None:
            segment: list[Any] = []
            for point in list(points) + [None]:
                if point is None:
                    if len(segment) >= 2:
                        painter.drawPolyline(QtGui.QPolygonF(segment))
                    segment = []
                else:
                    segment.append(point)

        def _draw_draft(self, painter: Any) -> None:
            if not self._draft_points:
                return
            screen_points = [self._q_to_screen(*pair) for pair in self._draft_points]
            screen_points = [point for point in screen_points if point is not None]
            if not screen_points:
                return
            painter.setPen(QtGui.QPen(QtGui.QColor(255, 245, 120, 230), 2.0, QtCore.Qt.PenStyle.DashLine))
            painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            if len(screen_points) >= 2:
                painter.drawPolyline(QtGui.QPolygonF(screen_points))
            for point in screen_points:
                painter.drawEllipse(point, 3.0, 3.0)

        def paintEvent(self, event: Any) -> None:  # noqa: N802 - Qt API
            del event
            self._plot_geometry()
            painter = QtGui.QPainter(self)
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
            painter.fillRect(self.rect(), QtGui.QColor(21, 24, 30))
            if self._q_bounds is None or self._plot_rect.isNull():
                painter.setPen(QtGui.QColor(205, 210, 220))
                painter.drawText(
                    self.rect(),
                    QtCore.Qt.AlignmentFlag.AlignCenter,
                    self._empty_state_message(),
                )
                painter.end()
                return
            painter.fillRect(self._plot_rect, QtGui.QColor(16, 18, 22))
            self._draw_mesh(painter)
            painter.setPen(QtGui.QPen(QtGui.QColor(165, 170, 180), 1.0))
            painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            painter.drawRect(self._plot_rect)
            self._draw_butterfly(painter)
            self._draw_edits(painter)
            self._draw_draft(painter)
            if self._compact_mode:
                painter.setPen(QtGui.QColor(198, 204, 215))
                font = painter.font()
                font.setPointSizeF(8.0)
                painter.setFont(font)
                x0, x1, y0, y1 = self._q_bounds
                for fraction in (0., .5, 1.):
                    xvalue, yvalue = x0 + fraction * (x1 - x0), y0 + fraction * (y1 - y0)
                    if fraction == .5:
                        xvalue = 0. if x0 < 0 < x1 else xvalue
                        yvalue = 0. if y0 < 0 < y1 else yvalue
                    xp, yp = self._q_to_screen(xvalue, y0), self._q_to_screen(x0, yvalue)
                    painter.drawText(QtCore.QRectF(xp.x() - 28, self._plot_rect.bottom() + 2, 56, 14),
                                     QtCore.Qt.AlignmentFlag.AlignCenter, f"{xvalue:.2g}")
                    painter.drawText(QtCore.QRectF(self._plot_rect.left() - 39, yp.y() - 7, 34, 14),
                                     QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter, f"{yvalue:.2g}")
                painter.drawText(QtCore.QRectF(0, self.height() - 17, self.width(), 15),
                                 QtCore.Qt.AlignmentFlag.AlignCenter, f"qx ({self._q_unit})")
                painter.drawText(int(self._plot_rect.left() + 3), int(self._plot_rect.top() + 11), "qy")
                painter.end()
                return
            painter.setPen(QtGui.QColor(198, 204, 215))
            x0, x1, y0, y1 = self._q_bounds
            tick_count = 4
            for index in range(tick_count + 1):
                fraction = index / tick_count
                qx_tick = x0 + fraction * (x1 - x0)
                qy_tick = y0 + fraction * (y1 - y0)
                x_point = self._q_to_screen(qx_tick, y0)
                y_point = self._q_to_screen(x0, qy_tick)
                if x_point is not None:
                    painter.drawLine(x_point.x(), self._plot_rect.bottom(), x_point.x(), self._plot_rect.bottom() + 4)
                    painter.drawText(x_point.x() - 20, self._plot_rect.bottom() + 16, f"{qx_tick:.3g}")
                if y_point is not None:
                    painter.drawLine(self._plot_rect.left() - 4, y_point.y(), self._plot_rect.left(), y_point.y())
                    painter.drawText(2, y_point.y() + 4, f"{qy_tick:.3g}")
            qx_label = f"qx ({self._q_unit})"
            qy_label = f"qy ({self._q_unit})"
            painter.drawText(
                int(self._plot_rect.center().x() - 32),
                int(self._plot_rect.bottom() + 31),
                qx_label,
            )
            painter.save()
            painter.translate(14, self._plot_rect.center().y() + 34)
            painter.rotate(-90)
            painter.drawText(0, 0, qy_label)
            painter.restore()
            mode_text = {
                "select_point": "选择点",
                "seed": "种子点",
                "exclude_point": "排除点",
                "rectangle_exclude": "排除矩形",
                "rectangle_include": "包含矩形",
                "polygon_exclude": "排除多边形",
                "polygon_include": "包含多边形",
            }.get(self._interaction_mode, self._interaction_mode)
            if self._language.lower().startswith("en"):
                mode_text = self._interaction_mode
            painter.drawText(self.width() - 180, 18, f"mode: {mode_text}")
            if self._hover_q is not None:
                hover_text = (
                    f"hover q=({self._hover_q[0]:.4g}, {self._hover_q[1]:.4g})"
                    if self._language.lower().startswith("en")
                    else f"悬停 q=({self._hover_q[0]:.4g}, {self._hover_q[1]:.4g})"
                )
                painter.drawText(8, 18, hover_text)
            painter.end()

        def save_screenshot(self, path: str | Path) -> Path:
            target = Path(path).expanduser().resolve()
            target.parent.mkdir(parents=True, exist_ok=True)
            self.grab().save(str(target))
            return target


else:

    class QSpaceView:
        """Qt-free state sink with the same data/edit seam."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs
            require_qt()


__all__ = ["QSpaceView"]
