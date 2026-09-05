"""Small pyqtgraph-backed views used by the refinement workbench."""

from __future__ import annotations

from collections.abc import Mapping
from math import cos, pi, sin
from typing import Any

from .i18n import translate, validate_language
from .qt_compat import QT_AVAILABLE, QtCore, QtWidgets


_VIEW_TITLE_KEYS = {
    "Observed": "view.observed",
    "Model": "view.model",
    "Residual": "view.residual",
    "Overlay": "view.overlay",
}

try:  # pyqtgraph is optional even when PySide6 is available
    import pyqtgraph as _pg

    PLOT_AVAILABLE = QT_AVAILABLE
except Exception:  # pragma: no cover - depends on optional UI install
    _pg = None
    PLOT_AVAILABLE = False

try:
    import numpy as _np
except Exception:  # pragma: no cover - numpy is a core dependency in normal installs
    _np = None


def _as_array(data: Any) -> Any:
    if data is None:
        return None
    if _np is None:
        return data
    try:
        result = _np.asarray(data)
    except Exception:
        return None
    if result.ndim != 2 or result.size == 0:
        return None
    return result


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


def _display_array(data: Any, *, valid_mask: Any = None, external_mask: Any = None) -> Any:
    """Return a display copy with invalid detector pixels transparent."""

    array = _as_array(data)
    if array is None or _np is None:
        return array
    try:
        result = _np.array(array, copy=True)
        invalid = ~_np.isfinite(result)
        if valid_mask is not None:
            valid = _np.asarray(valid_mask, dtype=bool)
            if valid.shape == result.shape:
                invalid |= ~valid
        if external_mask is not None:
            mask = _np.asarray(external_mask, dtype=bool)
            if mask.shape == result.shape:
                invalid |= mask
        result = result.astype(float, copy=False)
        result[invalid] = _np.nan
        return result
    except Exception:
        return array


def _display_transform(data: Any, scale: str) -> Any:
    """Apply a display-only contrast transform to a finite float array."""

    if data is None or _np is None:
        return data
    try:
        array = _np.asarray(data, dtype=float)
        mode = str(scale or "linear").strip().lower().replace("-", "_")
        if mode in {"linear", "raw", "none"}:
            return array
        if mode in {"log", "log1p", "signed_log"}:
            # Signed log preserves residual polarity while remaining stable at
            # zero; the raw data arrays are retained separately by ViewGrid.
            return _np.sign(array) * _np.log1p(_np.abs(array))
        if mode in {"asinh", "arcsinh"}:
            return _np.arcsinh(array)
    except Exception:
        return data
    return data


def _color_map(name: str) -> Any:
    """Resolve a pyqtgraph 0.14/matplotlib colour map with a safe fallback."""

    if _pg is None:
        return None
    getter = getattr(_pg.colormap, "getFromMatplotlib", None)
    if callable(getter):
        try:
            return getter(name)
        except Exception:
            pass
    try:
        return _pg.colormap.get(name)
    except Exception:
        try:
            colors = (
                [[40, 70, 180, 255], [245, 245, 245, 255], [180, 40, 40, 255]]
                if name.lower().startswith("rdbu")
                else [[30, 30, 30, 255], [230, 230, 80, 255]]
            )
            return _pg.ColorMap([0.0, 0.5, 1.0] if len(colors) == 3 else [0.0, 1.0], colors)
        except Exception:
            return None


def _disable_auto_si_prefix(plot: Any) -> None:
    """Keep q and pixel tick labels in the declared native units."""

    if plot is None:
        return
    for axis_name in ("bottom", "left"):
        try:
            plot.getAxis(axis_name).enableAutoSIPrefix(False)
        except Exception:
            pass


def _point_xy(point: Any) -> tuple[float, float] | None:
    if isinstance(point, Mapping):
        x = _read(point, ("x", "qx", "q", "q_x", "u"), None)
        y = _read(point, ("y", "qy", "q_y", "v"), None)
    else:
        try:
            if len(point) >= 2:
                x, y = point[0], point[1]
            else:
                return None
        except (TypeError, IndexError):
            x = _read(point, ("x", "qx", "q"), None)
            y = _read(point, ("y", "qy"), None)
    try:
        return float(x), float(y)
    except (TypeError, ValueError):
        return None


def _ellipse_xy(ellipse: Any, count: int = 180) -> tuple[list[float], list[float]] | None:
    center = _read(ellipse, ("center", "centre", "origin"), None)
    if center is None:
        cx = _read(ellipse, ("cx", "x0", "center_x", "q0"), 0.0)
        cy = _read(ellipse, ("cy", "y0", "center_y"), 0.0)
    else:
        try:
            cx, cy = center[0], center[1]
        except (TypeError, IndexError):
            cx = cy = 0.0
    axes = _read(ellipse, ("axes", "radii", "semi_axes"), None)
    if axes is None:
        a = _read(ellipse, ("a", "major", "semi_major", "radius_x"), 1.0)
        b = _read(ellipse, ("b", "minor", "semi_minor", "radius_y"), 1.0)
    else:
        try:
            a, b = axes[0], axes[1]
        except (TypeError, IndexError):
            a = b = 1.0
    angle_degrees = _read(ellipse, ("angle_deg", "theta_deg", "orientation_deg"), None)
    angle = _read(ellipse, ("angle", "theta", "orientation"), 0.0)
    if angle_degrees is not None:
        angle = angle_degrees
    try:
        cx, cy, a, b, angle = map(float, (cx, cy, a, b, angle))
    except (TypeError, ValueError):
        return None
    if angle_degrees is not None or abs(angle) > 2 * pi:
        angle = angle * pi / 180.0
    if a <= 0 or b <= 0:
        return None
    c, s = cos(angle), sin(angle)
    xs: list[float] = []
    ys: list[float] = []
    for i in range(count + 1):
        t = 2 * pi * i / count
        ex, ey = a * cos(t), b * sin(t)
        xs.append(cx + ex * c - ey * s)
        ys.append(cy + ex * s + ey * c)
    return xs, ys


def _q_extent(qx: Any, qy: Any) -> tuple[float, float, float, float] | None:
    """Return a conservative ``(xmin, xmax, ymin, ymax)`` q-space extent.

    pyFAI maps are generally curvilinear arrays rather than separable x/y
    vectors.  ``ImageItem`` cannot represent that full warp, but a finite
    bounding rectangle is an explicit, deterministic mapping for the overlay
    view.  The ridge/ellipse coordinates are then drawn in the same q units.
    """

    if _np is None or qx is None or qy is None:
        return None
    try:
        x, y = _np.broadcast_arrays(_np.asarray(qx, dtype=float), _np.asarray(qy, dtype=float))
        if x.ndim != 2 or y.ndim != 2:
            return None
        finite = _np.isfinite(x) & _np.isfinite(y)
        if not _np.any(finite):
            return None
        xmin, xmax = float(_np.min(x[finite])), float(_np.max(x[finite]))
        ymin, ymax = float(_np.min(y[finite])), float(_np.max(y[finite]))
        if not all(_np.isfinite((xmin, xmax, ymin, ymax))):
            return None
        # A one-pixel or degenerate q extent is still renderable after a tiny
        # expansion; this also prevents QRectF from collapsing to a line.
        dx = max(abs(xmax - xmin), 1.0e-12)
        dy = max(abs(ymax - ymin), 1.0e-12)
        if xmax <= xmin:
            xmin, xmax = xmin - dx / 2.0, xmax + dx / 2.0
        if ymax <= ymin:
            ymin, ymax = ymin - dy / 2.0, ymax + dy / 2.0
        return xmin, xmax, ymin, ymax
    except Exception:
        return None


def _roi_specs(roi: Any) -> list[Any]:
    """Normalize rectangle/ellipse ROI inputs for display."""

    if roi is None:
        return []
    if isinstance(roi, Mapping):
        return [roi]
    if isinstance(roi, (tuple, list)):
        if len(roi) == 4 and all(not isinstance(item, Mapping) for item in roi):
            return [{"type": "rectangle", "x0": roi[0], "y0": roi[1], "x1": roi[2], "y1": roi[3]}]
        return list(roi)
    return [roi]


if QT_AVAILABLE:

    class PatternView(QtWidgets.QWidget):
        """One observed/model/residual/overlay view.

        The class intentionally has a tiny public surface (``set_image`` and
        ``set_overlay``), making it possible for core engines to return plain
        arrays and dictionaries without knowing anything about Qt.
        """

        def __init__(self, title: str, parent: Any = None, *, language: str = "en") -> None:
            super().__init__(parent)
            self.title = title
            self._title_key = _VIEW_TITLE_KEYS.get(title)
            self._language = validate_language(language)
            self.image_data: Any = None
            self.raw_image_data: Any = None
            self.image_extent: tuple[float, float, float, float] | None = None
            self.ridge_points: list[tuple[float, float]] = []
            self.rejected_ridge_points: list[tuple[float, float]] = []
            self._ridge_records: list[Any] = []
            self._rejected_ridge_records: list[Any] = []
            self.ellipses: list[Any] = []
            self.model_ellipses: list[Any] = []
            self._overlay_items: list[Any] = []
            self.roi: tuple[float, float, float, float] | None = None
            self._roi_item: Any = None
            self.roi_specs: list[Any] = []
            self._roi_items: list[Any] = []
            layout = QtWidgets.QVBoxLayout(self)
            layout.setContentsMargins(2, 2, 2, 2)
            self.title_label = QtWidgets.QLabel(title)
            self.title_label.setObjectName("viewTitle")
            layout.addWidget(self.title_label)
            self.state_label = QtWidgets.QLabel(self)
            self.state_label.setObjectName("viewStateLabel")
            self.state_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self.state_label.setWordWrap(True)
            self.state_label.setVisible(False)
            layout.addWidget(self.state_label)
            if PLOT_AVAILABLE:
                self.plot = _pg.PlotWidget()
                self.plot.setObjectName(f"{title.lower()}Plot")
                self.plot.showGrid(x=True, y=True, alpha=0.22)
                self.plot.setAspectLocked(True)
                self.plot.setLabel("bottom", "x (pixel)")
                self.plot.setLabel("left", "y (pixel)")
                _disable_auto_si_prefix(self.plot)
                translated_title = (
                    translate(self._language, self._title_key)
                    if self._title_key is not None
                    else title
                )
                self.plot.setAccessibleName(
                    f"{translated_title} {translate(self._language, 'a11y.plot_suffix')}"
                )
                self.plot.setAccessibleDescription(
                    translate(
                        self._language,
                        "a11y.overlay_description"
                        if str(title).lower() == "overlay"
                        else "a11y.detector_description",
                    )
                )
                self.image_item = _pg.ImageItem()
                try:
                    cmap_name = "RdBu_r" if title.lower() == "residual" else "viridis"
                    colour_map = _color_map(cmap_name)
                    if colour_map is not None:
                        self.image_item.setColorMap(colour_map)
                except Exception:
                    # A failed optional colour map must never make a view
                    # unusable; ImageItem retains its default lookup table.
                    pass
                self.plot.addItem(self.image_item)
                if str(title).lower() == "overlay":
                    self.plot.addLegend(offset=(8, 8))
                layout.addWidget(self.plot, 1)
            else:
                self.plot = None
                self.image_item = None
                self.placeholder = QtWidgets.QLabel("pyqtgraph 不可用")
                self.placeholder.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)  # type: ignore[name-defined]
                layout.addWidget(self.placeholder, 1)
            self.set_language(self._language)

        @property
        def language(self) -> str:
            return self._language

        def set_language(self, language: str) -> None:
            self._language = validate_language(language)
            if self._title_key is not None:
                self.title_label.setText(translate(self._language, self._title_key))
            if self.plot is not None:
                self.plot.setAccessibleName(
                    f"{translate(self._language, self._title_key)} "
                    f"{translate(self._language, 'a11y.plot_suffix')}"
                    if self._title_key is not None
                    else f"{translate(self._language, 'a11y.plot_suffix')}"
                )
                self.plot.setAccessibleDescription(
                    translate(
                        self._language,
                        "a11y.overlay_description"
                        if str(self.title).lower() == "overlay"
                        else "a11y.detector_description",
                    )
                )
                if str(self.title).lower() == "overlay":
                    self.set_overlay(
                        self._ridge_records or self.ridge_points,
                        self.ellipses,
                        model_ellipses=self.model_ellipses,
                        rejected_ridge_points=(
                            self._rejected_ridge_records or self.rejected_ridge_points
                        ),
                    )
                self.plot.setLabel("bottom", translate(self._language, "axis.x_pixel"))
                self.plot.setLabel("left", translate(self._language, "axis.y_pixel"))
            elif hasattr(self, "placeholder"):
                self.placeholder.setText(
                    translate(self._language, "view.pyqtgraph_unavailable")
                )

        def set_image(
            self,
            data: Any,
            *,
            levels: Any = None,
            extent: tuple[float, float, float, float] | None = None,
        ) -> None:
            self.image_data = _as_array(data)
            self.image_extent = extent
            if self.image_item is None or self.image_data is None:
                return
            self.state_label.clear()
            self.state_label.setVisible(False)
            kwargs = {}
            if levels is not None:
                kwargs["levels"] = levels
            self.image_item.setImage(
                self.image_data,
                autoLevels=levels is None,
                axisOrder="row-major",
                **kwargs,
            )
            if extent is not None:
                xmin, xmax, ymin, ymax = extent
                try:
                    self.image_item.setRect(
                        QtCore.QRectF(float(xmin), float(ymin), float(xmax - xmin), float(ymax - ymin))
                    )
                except Exception:
                    # Older pyqtgraph releases accept a tuple-like QRectF but
                    # still render the image in pixel coordinates if the
                    # transform is unavailable.  Keep the numeric extent on
                    # the view for callers and tests in that case.
                    pass
            else:
                try:
                    self.image_item.resetTransform()
                    height, width = self.image_data.shape
                    self.image_item.setRect(QtCore.QRectF(0.0, 0.0, float(width), float(height)))
                except Exception:
                    pass
            self.plot.autoRange()

        set_data = set_image

        def clear_image(self, message: str | None = None) -> None:
            self.image_data = None
            self.raw_image_data = None
            self.image_extent = None
            if self.image_item is not None:
                self.image_item.clear()
            self.state_label.setText(str(message or ""))
            self.state_label.setVisible(bool(message))

        def _remove_overlay_items(self) -> None:
            if self.plot is None:
                self._overlay_items.clear()
                return
            for item in self._overlay_items:
                try:
                    self.plot.removeItem(item)
                except Exception:
                    pass
            self._overlay_items.clear()

        def set_roi(self, roi: Any = None, *, render: bool = True) -> None:
            """Show pixel-space rectangle and/or ellipse exclusion contours."""

            specs = _roi_specs(roi)
            self.roi_specs = list(specs)
            self.roi = None
            if len(specs) == 1 and isinstance(specs[0], Mapping):
                spec = specs[0]
                kind = str(spec.get("type", spec.get("kind", ""))).lower()
                if kind in {"rectangle", "rect", "box"}:
                    try:
                        self.roi = tuple(float(spec[key]) for key in ("x0", "y0", "x1", "y1"))
                    except (KeyError, TypeError, ValueError):
                        self.roi = None
            if self.plot is None:
                return
            for item in self._roi_items:
                try:
                    self.plot.removeItem(item)
                except Exception:
                    pass
            self._roi_items.clear()
            self._roi_item = None
            if not specs or not PLOT_AVAILABLE or not render:
                return
            pen = _pg.mkPen(255, 220, 40, width=2, style=QtCore.Qt.PenStyle.DashLine)
            for spec in specs:
                if isinstance(spec, Mapping):
                    kind = str(spec.get("type", spec.get("kind", ""))).lower()
                    if kind in {"rectangle", "rect", "box"}:
                        try:
                            x0, y0, x1, y1 = (float(spec[key]) for key in ("x0", "y0", "x1", "y1"))
                        except (KeyError, TypeError, ValueError):
                            continue
                        xs = [x0, x1, x1, x0, x0]
                        ys = [y0, y0, y1, y1, y0]
                    elif kind in {"ellipse", "elliptical"}:
                        xy = _ellipse_xy(
                            {
                                "cx": spec.get("cx"),
                                "cy": spec.get("cy"),
                                "a": spec.get("rx", spec.get("a")),
                                "b": spec.get("ry", spec.get("b")),
                                "angle_deg": spec.get("angle_deg", spec.get("angle", 0.0)),
                            }
                        )
                        if xy is None:
                            continue
                        xs, ys = xy
                    else:
                        continue
                else:
                    continue
                item = _pg.PlotDataItem(xs, ys, pen=pen)
                self.plot.addItem(item)
                self._roi_items.append(item)

        def set_overlay(
            self,
            ridge_points: Any = None,
            ellipses: Any = None,
            *,
            ellipse_parameters: Any = None,
            model_ellipses: Any = None,
            rejected_ridge_points: Any = None,
        ) -> None:
            if ellipses is None and ellipse_parameters is not None:
                ellipses = ellipse_parameters
            self._remove_overlay_items()
            point_source = [] if ridge_points is None else ridge_points
            ellipse_source = [] if ellipses is None else ellipses
            if isinstance(point_source, Mapping):
                point_source = [point_source]
            if isinstance(ellipse_source, Mapping):
                ellipse_source = [ellipse_source]
            model_source = [] if model_ellipses is None else model_ellipses
            if isinstance(model_source, Mapping):
                model_source = [model_source]
            self._ridge_records = [
                point for point in point_source if isinstance(point, Mapping)
            ]
            self.ridge_points = [point for point in (_point_xy(p) for p in point_source) if point]
            rejected_source = [] if rejected_ridge_points is None else rejected_ridge_points
            if isinstance(rejected_source, Mapping):
                rejected_source = [rejected_source]
            self._rejected_ridge_records = [
                point for point in rejected_source if isinstance(point, Mapping)
            ]
            self.rejected_ridge_points = [
                point for point in (_point_xy(p) for p in rejected_source) if point
            ]
            self.ellipses = list(ellipse_source)
            self.model_ellipses = list(model_source)
            if self.plot is None or not PLOT_AVAILABLE:
                return

            def record_branch(record: Any) -> int | None:
                raw = _read(
                    record,
                    ("overlay_branch_id", "branch_id", "branch", "component"),
                    None,
                )
                try:
                    value = int(raw)
                except (TypeError, ValueError):
                    return None
                return value if value in (0, 1) else None

            def record_pair(record: Any) -> str | None:
                pair = _read(record, ("quadrant_pair", "lobe_pair", "pair_label"), None)
                if isinstance(pair, (list, tuple)):
                    pair = "+".join(str(item) for item in pair)
                return None if pair in (None, "") else str(pair)

            def ridge_name(record: Any, *, rejected: bool) -> str:
                pair = record_pair(record)
                if pair is not None:
                    return translate(self._language, "legend.ridge_pair", pair=pair)
                return translate(
                    self._language,
                    "legend.rejected" if rejected else "legend.accepted",
                )

            branch_brushes = {
                0: _pg.mkBrush(230, 120, 40, 220),
                1: _pg.mkBrush(190, 80, 190, 220),
            }
            branch_pens = {
                0: _pg.mkPen(150, 65, 20, 220),
                1: _pg.mkPen(120, 45, 120, 220),
            }
            for rejected, records, fallback_points in (
                (False, self._ridge_records, self.ridge_points),
                (True, self._rejected_ridge_records, self.rejected_ridge_points),
            ):
                grouped: dict[int | None, list[tuple[float, float]]] = {}
                grouped_records: dict[int | None, list[Any]] = {}
                if records:
                    for record in records:
                        point = _point_xy(record)
                        if point:
                            branch = record_branch(record)
                            grouped.setdefault(branch, []).append(point)
                            grouped_records.setdefault(branch, []).append(record)
                elif fallback_points:
                    grouped[None] = list(fallback_points)
                for branch, points in grouped.items():
                    if not points:
                        continue
                    if rejected:
                        pen = branch_pens.get(branch, _pg.mkPen(230, 90, 90, width=2))
                        brush = None
                        symbol = "x"
                    else:
                        pen = branch_pens.get(branch, _pg.mkPen(40, 35, 20, 180))
                        brush = branch_brushes.get(branch, _pg.mkBrush(255, 218, 70, 210))
                        symbol = "o"
                    representative = grouped_records.get(branch, [None])[0]
                    scatter = _pg.ScatterPlotItem(
                        [point[0] for point in points],
                        [point[1] for point in points],
                        size=8 if rejected else 7,
                        symbol=symbol,
                        brush=brush,
                        pen=pen,
                        name=ridge_name(representative, rejected=rejected),
                    )
                    self.plot.addItem(scatter)
                    self._overlay_items.append(scatter)
            def branch_index(ellipse: Any, index: int, total: int) -> int | None:
                raw = _read(ellipse, ("branch_id", "branch", "component"), None)
                if isinstance(raw, str):
                    text = raw.strip().lower()
                    if text in {"ellipse_a", "a", "0", "plus", "positive"}:
                        return 0
                    if text in {"ellipse_b", "b", "1", "minus", "negative"}:
                        return 1
                try:
                    value = int(raw)
                except (TypeError, ValueError):
                    value = None
                if value in (0, 1):
                    return value
                # A returned pair is an identity-ordered pair.  This does not
                # assign physical quadrants; the reference axis remains the
                # source of any scientific orientation interpretation.
                return index if total == 2 else None

            def branch_name(ellipse: Any, branch: int | None, *, model: bool) -> str:
                pair = _read(
                    ellipse,
                    ("quadrant_pair", "lobe_pair", "pair_label"),
                    None,
                )
                if isinstance(pair, (list, tuple)):
                    pair = "+".join(str(item) for item in pair)
                if pair not in (None, ""):
                    return translate(
                        self._language,
                        "legend.quadrant_pair",
                        pair=str(pair),
                    )
                if model:
                    return translate(
                        self._language,
                        "legend.model_ellipse_a" if branch == 0 else "legend.model_ellipse_b",
                    )
                if branch == 0:
                    return translate(self._language, "legend.ellipse_a")
                if branch == 1:
                    return translate(self._language, "legend.ellipse_b")
                return translate(self._language, "legend.measured_ellipse")

            for index, ellipse in enumerate(self.ellipses):
                xy = _ellipse_xy(ellipse)
                if xy is None:
                    continue
                branch = branch_index(ellipse, index, len(self.ellipses))
                if branch == 0:
                    pen = _pg.mkPen(230, 120, 40, width=2)
                elif branch == 1:
                    pen = _pg.mkPen(190, 80, 190, width=2)
                else:
                    pen = _pg.mkPen(255, 90, 90, width=2)
                name = branch_name(ellipse, branch, model=False)
                curve = _pg.PlotDataItem(
                    xy[0], xy[1], pen=pen, name=name
                )
                self.plot.addItem(curve)
                self._overlay_items.append(curve)
            for index, ellipse in enumerate(self.model_ellipses):
                xy = _ellipse_xy(ellipse)
                if xy is None:
                    continue
                branch = branch_index(ellipse, index, len(self.model_ellipses))
                name = branch_name(ellipse, branch, model=True)
                curve = _pg.PlotDataItem(
                    xy[0], xy[1],
                    pen=_pg.mkPen(70, 220, 255, width=2, style=QtCore.Qt.PenStyle.DashLine),
                    name=name,
                )
                self.plot.addItem(curve)
                self._overlay_items.append(curve)
            # The image background and overlays must share the same coordinate
            # system.  Re-ranging after both are present avoids clipping a
            # ridge point when q maps are sparse or asymmetric.
            try:
                self.plot.autoRange()
            except Exception:
                pass

        def set_ridge_and_ellipses(self, ridge_points: Any, ellipses: Any) -> None:
            self.set_overlay(ridge_points, ellipses)

        def set_ellipses(self, ellipses: Any) -> None:
            self.set_overlay(self.ridge_points, ellipses, model_ellipses=self.model_ellipses)


    class ViewGrid(QtWidgets.QWidget):
        """Four-view layout shared by the refinement and batch pages."""

        def __init__(self, parent: Any = None, *, language: str = "en") -> None:
            super().__init__(parent)
            self._language = validate_language(language)
            self._q_unit = "unknown"
            self._qx: Any = None
            self._qy: Any = None
            self._raw_observed: Any = None
            self._raw_model: Any = None
            self._raw_residual: Any = None
            self._valid_mask: Any = None
            self._external_mask: Any = None
            self._q_extent: Any = None
            self._display_scale = "linear"
            self._display_percentile = 99.5
            self._active_q_window: tuple[float, float] | None = None
            layout = QtWidgets.QGridLayout(self)
            layout.setContentsMargins(2, 2, 2, 2)
            self.observed = PatternView("Observed", self, language=self._language)
            self.model = PatternView("Model", self, language=self._language)
            self.residual = PatternView("Residual", self, language=self._language)
            self.overlay = PatternView("Overlay", self, language=self._language)
            layout.addWidget(self.observed, 0, 0)
            layout.addWidget(self.model, 0, 1)
            layout.addWidget(self.residual, 1, 0)
            layout.addWidget(self.overlay, 1, 1)
            self.views = {
                "observed": self.observed,
                "model": self.model,
                "residual": self.residual,
                "overlay": self.overlay,
            }

        @property
        def language(self) -> str:
            return self._language

        def _set_overlay_axis_labels(self) -> None:
            if self.overlay.plot is not None:
                _disable_auto_si_prefix(self.overlay.plot)
                self.overlay.plot.setLabel("bottom", f"qx ({self._q_unit})")
                self.overlay.plot.setLabel("left", f"qy ({self._q_unit})")

        def set_language(self, language: str) -> None:
            self._language = validate_language(language)
            for view in self.views.values():
                view.set_language(self._language)
            self._set_overlay_axis_labels()

        def set_images(
            self,
            observed: Any = None,
            model: Any = None,
            residual: Any = None,
            *,
            qx: Any = None,
            qy: Any = None,
            q_extent: tuple[float, float, float, float] | None = None,
            q_unit: str | None = None,
            valid_mask: Any = None,
            external_mask: Any = None,
        ) -> None:
            self._raw_observed = _display_array(observed, valid_mask=valid_mask, external_mask=external_mask)
            self._raw_model = _display_array(model, valid_mask=valid_mask, external_mask=external_mask)
            self._raw_residual = _display_array(residual, valid_mask=valid_mask, external_mask=external_mask)
            self._valid_mask = valid_mask
            self._external_mask = external_mask
            if _np is not None:
                try:
                    self._qx = None if qx is None else _np.asarray(qx, dtype=float)
                    self._qy = None if qy is None else _np.asarray(qy, dtype=float)
                except (TypeError, ValueError):
                    self._qx = None
                    self._qy = None
            else:
                self._qx = qx
                self._qy = qy
            self._q_extent = q_extent if q_extent is not None else _q_extent(qx, qy)
            self._q_unit = str(q_unit or "unknown")
            self._render_images()

        def _render_images(self) -> None:
            """Render stored raw arrays using the current display contrast."""

            observed_array = _display_transform(self._raw_observed, self._display_scale)
            model_array = _display_transform(self._raw_model, self._display_scale)
            # Keep Observed and Model visually comparable.  Residuals use a
            # symmetric range around zero so the red/blue colors carry sign.
            shared = _shared_levels(
                observed_array,
                model_array,
                percentile=self._display_percentile,
            )
            if observed_array is not None:
                self.observed.set_image(observed_array, levels=shared)
                self.observed.raw_image_data = self._raw_observed
            if model_array is not None:
                self.model.set_image(model_array, levels=shared)
                self.model.raw_image_data = self._raw_model
            residual_array = _display_transform(self._raw_residual, self._display_scale)
            if residual_array is not None:
                self.residual.set_image(
                    residual_array,
                    levels=_symmetric_levels(
                        residual_array,
                        percentile=self._display_percentile,
                    ),
                )
                self.residual.raw_image_data = self._raw_residual
            # The overlay is an observed-data background in q space.  A
            # conservative q extent is sufficient for a general curvilinear
            # qmap while keeping annotations in the same units.
            if observed_array is not None:
                self.overlay.set_image(
                    observed_array,
                    levels=shared,
                    extent=self._q_extent,
                )
                self.overlay.raw_image_data = self._raw_observed
            self._set_overlay_axis_labels()
            if self._active_q_window is not None:
                self.set_q_view(self._active_q_window)

        def set_display_settings(
            self,
            scale: str = "linear",
            percentile: float = 99.5,
        ) -> None:
            """Change only the rendered contrast; scientific arrays stay raw."""

            mode = str(scale or "linear").strip().lower().replace("-", "_")
            if mode not in {"linear", "log1p", "asinh"}:
                mode = "linear"
            try:
                value = float(percentile)
            except (TypeError, ValueError):
                value = 99.5
            self._display_scale = mode
            self._display_percentile = min(100.0, max(50.0, value))
            self._render_images()

        @property
        def display_settings(self) -> dict[str, Any]:
            return {
                "scale": self._display_scale,
                "percentile": float(self._display_percentile),
            }

        def set_q_unit(self, q_unit: str | None) -> None:
            self._q_unit = str(q_unit or "unknown")
            self._set_overlay_axis_labels()

        def set_overlay(
            self,
            ridge_points: Any = None,
            ellipses: Any = None,
            *,
            ellipse_parameters: Any = None,
            model_ellipses: Any = None,
            rejected_ridge_points: Any = None,
        ) -> None:
            if ellipses is None and ellipse_parameters is not None:
                ellipses = ellipse_parameters
            self.overlay.set_overlay(
                ridge_points,
                ellipses,
                ellipse_parameters=ellipse_parameters,
                model_ellipses=model_ellipses,
                rejected_ridge_points=rejected_ridge_points,
            )

        def set_q_view(self, q_window: Any = None, *, full: bool = False) -> None:
            """Focus all image views on a q-window without warping pixels.

            The detector images remain in their native pixel coordinates.  A
            q-window is converted to a conservative pixel bounding box using
            the supplied curvilinear q maps, while the overlay receives the
            corresponding q-space bounding box.  This keeps annotations and
            image pixels aligned without making a false separable q-grid
            assumption.
            """

            plot = self.overlay.plot
            if full or q_window is None:
                self._active_q_window = None
                for view in self.views.values():
                    if view.plot is None:
                        continue
                    try:
                        view.plot.autoRange()
                    except Exception:
                        pass
                return
            try:
                q_min, q_max = (float(value) for value in q_window)
                if not _np.isfinite(q_min) or not _np.isfinite(q_max) or q_min < 0 or q_max <= q_min:
                    return
                self._active_q_window = (q_min, q_max)
                qx = _np.asarray(self._qx, dtype=float)
                qy = _np.asarray(self._qy, dtype=float)
                if qx.shape != qy.shape or qx.ndim != 2:
                    return
                finite = _np.isfinite(qx) & _np.isfinite(qy)
                radius_map = _np.hypot(qx, qy)
                selected = finite & (radius_map >= q_min) & (radius_map <= q_max)
                if not _np.any(selected):
                    selected = finite & (radius_map <= q_max)
                if not _np.any(selected):
                    return
                rows, cols = _np.nonzero(selected)
                x0, x1 = float(cols.min()), float(cols.max() + 1)
                y0, y1 = float(rows.min()), float(rows.max() + 1)
                qx0, qx1 = float(qx[selected].min()), float(qx[selected].max())
                qy0, qy1 = float(qy[selected].min()), float(qy[selected].max())
                if plot is not None:
                    plot.setRange(
                        xRange=(qx0, qx1),
                        yRange=(qy0, qy1),
                        padding=0.02,
                    )
                for name in ("observed", "model", "residual"):
                    image_plot = self.views[name].plot
                    if image_plot is None:
                        continue
                    image_plot.setRange(
                        xRange=(x0, x1),
                        yRange=(y0, y1),
                        padding=0.02,
                    )
            except (TypeError, ValueError):
                return

        def clear_fit(self) -> None:
            """Clear results that belong to the previous observed frame."""

            self._raw_model = None
            self._raw_residual = None
            self.model.clear_image()
            self.residual.clear_image()
            self.overlay.set_overlay([], [], model_ellipses=[])
            try:
                self.overlay.plot.autoRange()
            except Exception:
                pass

        def set_roi(self, roi: Any = None) -> None:
            for view in (self.observed, self.model, self.residual):
                view.set_roi(roi)
            # ROI coordinates are detector pixels.  Keep the metadata on the
            # overlay for project/UI state, but never draw those values in its
            # q-space coordinate system (which would distort auto-ranging).
            self.overlay.set_roi(roi, render=False)


else:

    class PatternView:
        """Qt-free data sink matching :class:`PatternView`'s API."""

        def __init__(self, title: str, parent: Any = None, *, language: str = "en") -> None:
            del parent
            self.title = title
            self._title_key = _VIEW_TITLE_KEYS.get(title)
            self._language = validate_language(language)
            self.image_data: Any = None
            self.image_extent: tuple[float, float, float, float] | None = None
            self.ridge_points: list[tuple[float, float]] = []
            self.ellipses: list[Any] = []
            self.model_ellipses: list[Any] = []
            self.roi: tuple[float, float, float, float] | None = None
            self.roi_specs: list[Any] = []

        @property
        def language(self) -> str:
            return self._language

        def set_language(self, language: str) -> None:
            self._language = validate_language(language)
            if self._title_key is not None:
                self.title = translate(self._language, self._title_key)

        def set_image(
            self,
            data: Any,
            *,
            levels: Any = None,
            extent: tuple[float, float, float, float] | None = None,
        ) -> None:
            del levels
            self.image_data = _as_array(data)
            self.image_extent = extent

        def set_roi(self, roi: Any = None, *, render: bool = True) -> None:
            del render
            specs = _roi_specs(roi)
            self.roi_specs = list(specs)
            self.roi = None
            if len(specs) == 1 and isinstance(specs[0], Mapping):
                spec = specs[0]
                if str(spec.get("type", spec.get("kind", ""))).lower() in {"rectangle", "rect", "box"}:
                    try:
                        self.roi = tuple(float(spec[key]) for key in ("x0", "y0", "x1", "y1"))
                    except (KeyError, TypeError, ValueError):
                        self.roi = None

        set_data = set_image

        def clear_image(self) -> None:
            self.image_data = None
            self.image_extent = None

        def set_overlay(
            self,
            ridge_points: Any = None,
            ellipses: Any = None,
            *,
            ellipse_parameters: Any = None,
            model_ellipses: Any = None,
            rejected_ridge_points: Any = None,
        ) -> None:
            if ellipses is None and ellipse_parameters is not None:
                ellipses = ellipse_parameters
            point_source = [] if ridge_points is None else ridge_points
            ellipse_source = [] if ellipses is None else ellipses
            if isinstance(point_source, Mapping):
                point_source = [point_source]
            if isinstance(ellipse_source, Mapping):
                ellipse_source = [ellipse_source]
            model_source = [] if model_ellipses is None else model_ellipses
            if isinstance(model_source, Mapping):
                model_source = [model_source]
            self.ridge_points = [point for point in (_point_xy(p) for p in point_source) if point]
            rejected_source = [] if rejected_ridge_points is None else rejected_ridge_points
            if isinstance(rejected_source, Mapping):
                rejected_source = [rejected_source]
            self.rejected_ridge_points = [
                point for point in (_point_xy(p) for p in rejected_source) if point
            ]
            self.ellipses = list(ellipse_source)
            self.model_ellipses = list(model_source)

        set_ridge_and_ellipses = set_overlay

        def set_ellipses(self, ellipses: Any) -> None:
            self.set_overlay(self.ridge_points, ellipses, model_ellipses=self.model_ellipses)


    class ViewGrid:
        """Qt-free four-view container useful for batch scripts."""

        def __init__(self, parent: Any = None, *, language: str = "en") -> None:
            del parent
            self._language = validate_language(language)
            self.observed = PatternView("Observed", language=self._language)
            self.model = PatternView("Model", language=self._language)
            self.residual = PatternView("Residual", language=self._language)
            self.overlay = PatternView("Overlay", language=self._language)
            self.views = {
                "observed": self.observed,
                "model": self.model,
                "residual": self.residual,
                "overlay": self.overlay,
            }

        @property
        def language(self) -> str:
            return self._language

        def set_language(self, language: str) -> None:
            self._language = validate_language(language)
            for view in self.views.values():
                view.set_language(self._language)

        def set_images(
            self,
            observed: Any = None,
            model: Any = None,
            residual: Any = None,
            *,
            qx: Any = None,
            qy: Any = None,
            q_extent: tuple[float, float, float, float] | None = None,
            q_unit: str | None = None,
            valid_mask: Any = None,
            external_mask: Any = None,
        ) -> None:
            if observed is not None:
                self.observed.set_image(_display_array(observed, valid_mask=valid_mask, external_mask=external_mask))
            if model is not None:
                self.model.set_image(_display_array(model, valid_mask=valid_mask, external_mask=external_mask))
            if residual is not None:
                self.residual.set_image(_display_array(residual, valid_mask=valid_mask, external_mask=external_mask))
            if observed is not None:
                self.overlay.set_image(
                    _display_array(observed, valid_mask=valid_mask, external_mask=external_mask),
                    extent=q_extent if q_extent is not None else _q_extent(qx, qy),
                )

        def set_overlay(
            self,
            ridge_points: Any = None,
            ellipses: Any = None,
            *,
            ellipse_parameters: Any = None,
            model_ellipses: Any = None,
            rejected_ridge_points: Any = None,
        ) -> None:
            self.overlay.set_overlay(
                ridge_points,
                ellipses,
                ellipse_parameters=ellipse_parameters,
                model_ellipses=model_ellipses,
                rejected_ridge_points=rejected_ridge_points,
            )

        def set_q_view(self, q_window: Any = None, *, full: bool = False) -> None:
            self.overlay.set_q_view(q_window, full=full)

        def clear_fit(self) -> None:
            """Clear results that belong to the previous observed frame."""

            self.model.clear_image()
            self.residual.clear_image()
            self.overlay.set_overlay([], [], model_ellipses=[])

        def set_roi(self, roi: Any = None) -> None:
            for view in (self.observed, self.model, self.residual):
                view.set_roi(roi)
            self.overlay.set_roi(roi, render=False)


OverlayView = PatternView


def _finite_limits(
    array: Any,
    *,
    percentile: float = 99.5,
) -> tuple[float, float] | None:
    if array is None or _np is None:
        return None
    try:
        values = _np.asarray(array, dtype=float)
        finite = values[_np.isfinite(values)]
        if finite.size == 0:
            return None
        # Detector hot pixels and beamstop tails can span orders of magnitude.
        # Display levels use robust percentiles while the underlying arrays
        # and exported values remain untouched.
        upper = min(100.0, max(50.0, float(percentile)))
        lower = min(upper - 0.1, 1.0)
        lo, hi = _np.percentile(finite, (lower, upper))
        if not _np.isfinite(lo) or not _np.isfinite(hi) or hi <= lo:
            return float(finite.min()), float(finite.max())
        return float(lo), float(hi)
    except Exception:
        return None


def _shared_levels(
    observed: Any,
    model: Any,
    *,
    percentile: float = 99.5,
) -> tuple[float, float] | None:
    limits = [
        item
        for item in (
            _finite_limits(observed, percentile=percentile),
            _finite_limits(model, percentile=percentile),
        )
        if item
    ]
    if not limits:
        return None
    return min(item[0] for item in limits), max(item[1] for item in limits)


def _symmetric_levels(
    array: Any,
    *,
    percentile: float = 99.5,
) -> tuple[float, float] | None:
    limits = _finite_limits(array, percentile=percentile)
    if limits is None:
        return None
    magnitude = max(abs(limits[0]), abs(limits[1]))
    return -magnitude, magnitude

__all__ = ["PLOT_AVAILABLE", "PatternView", "OverlayView", "ViewGrid"]
