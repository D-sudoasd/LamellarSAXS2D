"""Small pyqtgraph-backed views used by the refinement workbench."""

from __future__ import annotations

from collections.abc import Mapping
from math import cos, pi, sin
from typing import Any

from .qt_compat import QT_AVAILABLE, QtCore, QtWidgets

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

        def __init__(self, title: str, parent: Any = None) -> None:
            super().__init__(parent)
            self.title = title
            self.image_data: Any = None
            self.image_extent: tuple[float, float, float, float] | None = None
            self.ridge_points: list[tuple[float, float]] = []
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
            if PLOT_AVAILABLE:
                self.plot = _pg.PlotWidget()
                self.plot.setObjectName(f"{title.lower()}Plot")
                self.plot.showGrid(x=True, y=True, alpha=0.22)
                self.plot.setAspectLocked(True)
                self.plot.setLabel("bottom", "x (pixel)")
                self.plot.setLabel("left", "y (pixel)")
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
                layout.addWidget(self.plot, 1)
            else:
                self.plot = None
                self.image_item = None
                self.placeholder = QtWidgets.QLabel("pyqtgraph 不可用")
                self.placeholder.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)  # type: ignore[name-defined]
                layout.addWidget(self.placeholder, 1)

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

        def clear_image(self) -> None:
            self.image_data = None
            self.image_extent = None
            if self.image_item is not None:
                self.image_item.clear()

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
            self.ridge_points = [point for point in (_point_xy(p) for p in point_source) if point]
            self.ellipses = list(ellipse_source)
            self.model_ellipses = list(model_source)
            if self.plot is None or not PLOT_AVAILABLE:
                return
            if self.ridge_points:
                scatter = _pg.ScatterPlotItem(
                    [point[0] for point in self.ridge_points],
                    [point[1] for point in self.ridge_points],
                    size=7,
                    brush=_pg.mkBrush(255, 218, 70, 210),
                    pen=_pg.mkPen(40, 35, 20, 180),
                )
                self.plot.addItem(scatter)
                self._overlay_items.append(scatter)
            for ellipse in self.ellipses:
                xy = _ellipse_xy(ellipse)
                if xy is None:
                    continue
                curve = _pg.PlotDataItem(
                    xy[0], xy[1], pen=_pg.mkPen(255, 90, 90, width=2)
                )
                self.plot.addItem(curve)
                self._overlay_items.append(curve)
            for ellipse in self.model_ellipses:
                xy = _ellipse_xy(ellipse)
                if xy is None:
                    continue
                curve = _pg.PlotDataItem(
                    xy[0], xy[1],
                    pen=_pg.mkPen(70, 220, 255, width=2, style=QtCore.Qt.PenStyle.DashLine),
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

        def __init__(self, parent: Any = None) -> None:
            super().__init__(parent)
            layout = QtWidgets.QGridLayout(self)
            layout.setContentsMargins(2, 2, 2, 2)
            self.observed = PatternView("Observed", self)
            self.model = PatternView("Model", self)
            self.residual = PatternView("Residual", self)
            self.overlay = PatternView("Overlay", self)
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
            observed_array = _display_array(observed, valid_mask=valid_mask, external_mask=external_mask)
            model_array = _display_array(model, valid_mask=valid_mask, external_mask=external_mask)
            # Keep Observed and Model visually comparable.  Residuals use a
            # symmetric range around zero so the red/blue colors carry sign.
            shared = _shared_levels(observed_array, model_array)
            if observed_array is not None:
                self.observed.set_image(observed_array, levels=shared)
            if model_array is not None:
                self.model.set_image(model_array, levels=shared)
            residual_array = _display_array(residual, valid_mask=valid_mask, external_mask=external_mask)
            if residual_array is not None:
                self.residual.set_image(residual_array, levels=_symmetric_levels(residual_array))
            # The overlay is an observed-data background in q space.  A
            # conservative q extent is sufficient for a general curvilinear
            # qmap while keeping annotations in the same units.
            extent = q_extent if q_extent is not None else _q_extent(qx, qy)
            if observed_array is not None:
                self.overlay.set_image(observed_array, levels=shared, extent=extent)
            if self.overlay.plot is not None:
                unit = str(q_unit or "unknown")
                self.overlay.plot.setLabel("bottom", f"qx ({unit})")
                self.overlay.plot.setLabel("left", f"qy ({unit})")

        def set_overlay(
            self,
            ridge_points: Any = None,
            ellipses: Any = None,
            *,
            ellipse_parameters: Any = None,
            model_ellipses: Any = None,
        ) -> None:
            if ellipses is None and ellipse_parameters is not None:
                ellipses = ellipse_parameters
            self.overlay.set_overlay(
                ridge_points,
                ellipses,
                ellipse_parameters=ellipse_parameters,
                model_ellipses=model_ellipses,
            )

        def clear_fit(self) -> None:
            """Clear results that belong to the previous observed frame."""

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

        def __init__(self, title: str, parent: Any = None) -> None:
            del parent
            self.title = title
            self.image_data: Any = None
            self.image_extent: tuple[float, float, float, float] | None = None
            self.ridge_points: list[tuple[float, float]] = []
            self.ellipses: list[Any] = []
            self.model_ellipses: list[Any] = []
            self.roi: tuple[float, float, float, float] | None = None
            self.roi_specs: list[Any] = []

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
            self.ellipses = list(ellipse_source)
            self.model_ellipses = list(model_source)

        set_ridge_and_ellipses = set_overlay

        def set_ellipses(self, ellipses: Any) -> None:
            self.set_overlay(self.ridge_points, ellipses, model_ellipses=self.model_ellipses)


    class ViewGrid:
        """Qt-free four-view container useful for batch scripts."""

        def __init__(self, parent: Any = None) -> None:
            del parent
            self.observed = PatternView("Observed")
            self.model = PatternView("Model")
            self.residual = PatternView("Residual")
            self.overlay = PatternView("Overlay")
            self.views = {
                "observed": self.observed,
                "model": self.model,
                "residual": self.residual,
                "overlay": self.overlay,
            }

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
        ) -> None:
            self.overlay.set_overlay(
                ridge_points,
                ellipses,
                ellipse_parameters=ellipse_parameters,
                model_ellipses=model_ellipses,
            )

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


def _finite_limits(array: Any) -> tuple[float, float] | None:
    if array is None or _np is None:
        return None
    try:
        values = _np.asarray(array, dtype=float)
        finite = values[_np.isfinite(values)]
        if finite.size == 0:
            return None
        return float(finite.min()), float(finite.max())
    except Exception:
        return None


def _shared_levels(observed: Any, model: Any) -> tuple[float, float] | None:
    limits = [item for item in (_finite_limits(observed), _finite_limits(model)) if item]
    if not limits:
        return None
    return min(item[0] for item in limits), max(item[1] for item in limits)


def _symmetric_levels(array: Any) -> tuple[float, float] | None:
    limits = _finite_limits(array)
    if limits is None:
        return None
    magnitude = max(abs(limits[0]), abs(limits[1]))
    return -magnitude, magnitude

__all__ = ["PLOT_AVAILABLE", "PatternView", "OverlayView", "ViewGrid"]
