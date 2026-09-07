"""Native Qt Quick 3D view for the real-space lamellar schematic.

The core lamellar builder deliberately returns plain NumPy arrays.  This
module is the optional UI boundary that turns those arrays into a small
Qt Quick 3D scene.  The renderer does not infer any scientific quantities:
it displays the supplied centres, local dimensions, basis matrices, branch
labels and RGBA colours in the declared specimen frame (x/y in the specimen
plane, +y as the draw axis, z out of plane).

Qt Quick and Qt Quick 3D are imported lazily.  Importing this module is
therefore safe for core-only scripts and for test hosts without a graphical
context.  A real ``View3D`` is used whenever the optional modules and a Qt
application are available; otherwise the widget shows an explicit renderer
diagnostic instead of pretending that a 3-D frame was produced.
"""

from __future__ import annotations

from collections.abc import Mapping
import math
from pathlib import Path
import threading
import time
from typing import Any

try:  # NumPy is a normal project dependency, but keep module import defensive.
    import numpy as np
except Exception:  # pragma: no cover - exercised only by unusual minimal hosts
    np = None  # type: ignore[assignment]

from .qt_compat import QT_AVAILABLE, QtCore, QtGui, QtWidgets

try:
    from ..publication_models import PublicationRenderResult, PublicationStyle
except Exception:  # pragma: no cover - publication layer is optional during import
    PublicationRenderResult = None  # type: ignore[assignment,misc]
    PublicationStyle = None  # type: ignore[assignment,misc]


_QML_PATH = Path(__file__).resolve().parent / "qml" / "LamellarScene.qml"
_NO_BRANCH = -2_147_483_648
_STANDARD_VIEWS: dict[str, tuple[float, float]] = {
    "isometric": (-38.0, -26.0),
    "front": (0.0, 0.0),
    "side": (90.0, 0.0),
    "top": (0.0, 90.0),
}


def _read(source: Any, names: tuple[str, ...], default: Any = None) -> Any:
    """Read a field from a mapping or a small dataclass-like object."""

    if isinstance(source, Mapping):
        for name in names:
            if name in source:
                return source[name]
    else:
        for name in names:
            if hasattr(source, name):
                return getattr(source, name)
    return default


def _as_float_array(value: Any, *, ndim: int | None = None) -> Any:
    if np is None or value is None:
        return None
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if ndim is not None and result.ndim != ndim:
        return None
    return result


def _finite_bounds(value: Any) -> np.ndarray | None:
    """Return valid ``(2, 3)`` bounds, or ``None`` when the input is unusable."""

    if np is None:
        return None
    array = _as_float_array(value, ndim=2)
    if array is None or array.shape != (2, 3) or not np.all(np.isfinite(array)):
        return None
    low = np.minimum(array[0], array[1])
    high = np.maximum(array[0], array[1])
    if not np.any(high > low):
        return None
    # Keep a zero-width coordinate renderable while retaining the supplied
    # extent on the other axes.
    span = high - low
    nonzero = span[span > 0]
    pad = float(np.max(nonzero)) * 1.0e-6 if nonzero.size else 1.0
    for axis in range(3):
        if high[axis] <= low[axis]:
            low[axis] -= pad / 2.0
            high[axis] += pad / 2.0
    return np.stack((low, high), axis=0)


class _ResponsiveCancellation:
    """Observe GUI cancellation during CPU mesh work without moving Qt off-thread."""

    def __init__(self, event: Any = None) -> None:
        self.event = event
        self.last_pump = 0.

    def is_set(self) -> bool:
        if QT_AVAILABLE and time.monotonic() - self.last_pump >= .05:
            self.last_pump = time.monotonic()
            app = QtWidgets.QApplication.instance()
            if app is not None and QtCore.QThread.currentThread() == app.thread():
                app.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents, 5)
        method = getattr(self.event, "is_set", None)
        return bool(method()) if callable(method) else bool(self.event() if callable(self.event) else self.event)


def _camera_viewport_bounds(camera: Any) -> np.ndarray | None:
    """Validate optional actual-coordinate viewport bounds carried by a camera."""

    if not isinstance(camera, Mapping) or "viewport_bounds" not in camera:
        return None
    value = _finite_bounds(camera.get("viewport_bounds"))
    if value is None:
        raise ValueError("camera.viewport_bounds must be finite with shape (2, 3)")
    return value


def _matrix_to_quaternion(matrix: Any) -> tuple[float, float, float, float]:
    """Convert a column-basis 3x3 rotation matrix to ``(w, x, y, z)``.

    The scene contract supplies orthonormal basis columns.  The small
    normalization below prevents harmless floating point drift from creating
    a non-unit quaternion, while leaving the supplied orientation unchanged in
    the normal case.
    """

    if np is None:
        return 1.0, 0.0, 0.0, 0.0
    m = np.asarray(matrix, dtype=float)
    trace = float(m[0, 0] + m[1, 1] + m[2, 2])
    if trace > 0.0:
        s = math.sqrt(max(trace + 1.0, 1.0e-18)) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(max(1.0 + float(m[0, 0] - m[1, 1] - m[2, 2]), 1.0e-18)) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(max(1.0 + float(m[1, 1] - m[0, 0] - m[2, 2]), 1.0e-18)) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(max(1.0 + float(m[2, 2] - m[0, 0] - m[1, 1]), 1.0e-18)) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    quaternion = np.asarray((w, x, y, z), dtype=float)
    norm = float(np.linalg.norm(quaternion))
    if not math.isfinite(norm) or norm <= 1.0e-12:
        return 1.0, 0.0, 0.0, 0.0
    return tuple(float(item / norm) for item in quaternion)  # type: ignore[return-value]


def _as_rgb(color: Any) -> tuple[float, float, float, float]:
    """Normalize RGBA values supplied in either [0, 1] or [0, 255]."""

    try:
        values = [float(item) for item in color]
    except (TypeError, ValueError):
        values = []
    if len(values) < 3:
        return 0.28, 0.50, 0.78, 1.0
    values = (values + [1.0])[:4]
    if max(abs(item) for item in values) > 1.001:
        values = [item / 255.0 for item in values]
    return tuple(min(1.0, max(0.0, item)) for item in values)  # type: ignore[return-value]


def _placeholder_image(width: int, height: int, message: str) -> Any:
    """Create a truthful diagnostic image when no render context exists."""

    if not QT_AVAILABLE or QtGui is None:
        return None
    image = QtGui.QImage(int(width), int(height), QtGui.QImage.Format.Format_ARGB32)
    image.fill(QtGui.QColor("#f4f1ea"))
    painter = QtGui.QPainter(image)
    try:
        painter.setPen(QtGui.QColor("#24364b"))
        painter.setFont(QtGui.QFont("Segoe UI", max(10, min(22, width // 45))))
        painter.drawText(
            image.rect().adjusted(28, 28, -28, -28),
            QtCore.Qt.AlignmentFlag.AlignCenter | QtCore.Qt.TextFlag.TextWordWrap,
            message,
        )
    finally:
        painter.end()
    return image


class _ScenePayload:
    """Validated, renderer-ready scene data kept separate from the core object."""

    def __init__(self, items: list[dict[str, float]], bounds: np.ndarray, axis_length: float) -> None:
        self.items = items
        self.bounds = bounds
        self.axis_length = float(axis_length)


def _scene_payload(scene: Any, bounds: Any = None) -> _ScenePayload:
    """Validate and normalize the public lamellar scene contract.

    World coordinates are uniformly scaled to a compact range for the
    orthographic camera.  Uniform scaling preserves every centre, local size,
    orientation and box corner relationship, while the original bounds remain
    available on the view for camera fitting and sequence consistency.
    """

    if np is None:
        raise RuntimeError("NumPy is required to display a lamellar scene")
    centers = _as_float_array(_read(scene, ("centers",), None), ndim=2)
    sizes = _as_float_array(_read(scene, ("sizes",), None), ndim=2)
    orientations = _as_float_array(_read(scene, ("orientations",), None), ndim=3)
    vertices = _as_float_array(_read(scene, ("vertices",), None), ndim=3)
    if centers is None and vertices is not None and vertices.ndim == 3 and vertices.shape[1:] == (8, 3):
        centers = np.mean(vertices, axis=1)
    if sizes is None and vertices is not None and vertices.ndim == 3 and vertices.shape[1:] == (8, 3):
        sizes = np.ptp(vertices, axis=1)
    if orientations is None and centers is not None:
        orientations = np.repeat(np.eye(3, dtype=float)[None, :, :], len(centers), axis=0)
    if centers is None or sizes is None or orientations is None:
        raise ValueError("lamellar scene requires centers, sizes and orientations")
    if centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError("scene.centers must have shape (N, 3)")
    count = int(centers.shape[0])
    if sizes.shape != (count, 3):
        raise ValueError("scene.sizes must have shape (N, 3)")
    if orientations.shape != (count, 3, 3):
        raise ValueError("scene.orientations must have shape (N, 3, 3)")
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sizes)):
        raise ValueError("scene centres and sizes must be finite")
    if np.any(sizes <= 0.0):
        raise ValueError("scene.sizes must be positive")
    if not np.all(np.isfinite(orientations)):
        raise ValueError("scene.orientations must be finite")

    resolved_bounds = _finite_bounds(bounds)
    if resolved_bounds is None:
        resolved_bounds = _finite_bounds(_read(scene, ("bounds",), None))
    if resolved_bounds is None:
        # Use the oriented box corners when available.  This is only a
        # fallback for callers that omit the optional fixed sequence bounds.
        if vertices is not None and vertices.shape == (count, 8, 3) and np.all(np.isfinite(vertices)):
            resolved_bounds = np.stack((vertices.min(axis=(0, 1)), vertices.max(axis=(0, 1))))
        else:
            half = np.abs(sizes) / 2.0
            resolved_bounds = np.stack((np.min(centers - half, axis=0), np.max(centers + half, axis=0)))
    resolved_bounds = _finite_bounds(resolved_bounds)
    if resolved_bounds is None:  # defensive: the fallback above should always work
        raise ValueError("scene bounds are empty or non-finite")

    span = np.asarray(resolved_bounds[1] - resolved_bounds[0], dtype=float)
    scale = float(np.max(span))
    if not math.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    world_scale = 8.0 / scale
    scene_origin = (resolved_bounds[0] + resolved_bounds[1]) / 2.0
    centers_world = (centers - scene_origin) * world_scale
    sizes_world = sizes * world_scale

    stack_ids = _as_float_array(_read(scene, ("stack_ids",), None), ndim=1)
    branch_ids = _as_float_array(_read(scene, ("branch_ids",), None), ndim=1)
    if stack_ids is None or len(stack_ids) != count:
        stack_ids = np.arange(count, dtype=float)
    if branch_ids is None or len(branch_ids) != count:
        branch_ids = np.zeros(count, dtype=float)
    colors = _as_float_array(_read(scene, ("colors",), None), ndim=2)
    if colors is None or colors.shape != (count, 4):
        colors = np.tile(np.asarray((0.25, 0.48, 0.78, 1.0), dtype=float), (count, 1))

    items: list[dict[str, float]] = []
    for index in range(count):
        quaternion = _matrix_to_quaternion(orientations[index])
        rgba = _as_rgb(colors[index])
        item = {
            "x": float(centers_world[index, 0]),
            "y": float(centers_world[index, 1]),
            "z": float(centers_world[index, 2]),
            "sx": float(sizes_world[index, 0]),
            "sy": float(sizes_world[index, 1]),
            "sz": float(sizes_world[index, 2]),
            "qw": float(quaternion[0]),
            "qx": float(quaternion[1]),
            "qy": float(quaternion[2]),
            "qz": float(quaternion[3]),
            "r": rgba[0],
            "g": rgba[1],
            "b": rgba[2],
            "a": rgba[3],
            "stack": int(round(float(stack_ids[index]))),
            "branch": int(round(float(branch_ids[index]))),
        }
        items.append(item)
    # The helper sits at the normalized specimen centre and remains readable
    # for both a single slab and a long sequence.
    axis_length = max(0.95, min(2.6, float(np.max(span)) * world_scale * 0.22))
    return _ScenePayload(items, resolved_bounds, axis_length)


if QT_AVAILABLE:

    class Lamellar3DView(QtWidgets.QWidget):
        """Interactive native Qt Quick 3D lamellar viewer.

        The class owns presentation state only.  ``set_scene`` accepts the
        plain scene record returned by the core builder; no analysis object is
        retained by the renderer.  All QQuickWidget and QML calls are made on
        the GUI thread, as required by Qt Quick.
        """

        cameraChanged = QtCore.Signal(object)

        def __init__(self, parent: Any = None) -> None:
            super().__init__(parent)
            self.setObjectName("lamellar3DView")
            self.setMinimumSize(300, 220)
            self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
            self._language = "en"
            self._scene: Any = None
            self._payload: _ScenePayload | None = None
            self._bounds: np.ndarray | None = None
            self._has_scene = False
            self._selected_branch: int | None = None
            self._qml_root: Any = None
            self._quick: Any = None
            self._publication_style: Any = PublicationStyle() if PublicationStyle is not None else None
            self._publication_mesh: Any = None
            self._publication_groups: list[Any] = []
            self._publication_group_maps: list[dict[str, Any]] = []
            self._publication_geometry_refs: list[Any] = []
            self._publication_bounds: np.ndarray | None = None
            self._publication_error = ""
            self._publication_active = False
            self._publication_camera_raw: dict[str, Any] | None = None
            self._last_publication_fit_raw: dict[str, Any] | None = None
            self._qt_modules: tuple[Any, ...] | None = None
            self._renderer_attempted = False
            self._renderer_failed = False
            self._camera_state: dict[str, Any] = {
                "yaw": _STANDARD_VIEWS["isometric"][0],
                "pitch": _STANDARD_VIEWS["isometric"][1],
                "distance": 12.0,
                "ortho_height": 8.8,
                "target": (0.0, 0.0, 0.0),
            }
            self.available = False
            self.error_message = ""
            self._initialise_timer = QtCore.QTimer(self)
            self._initialise_timer.setSingleShot(True)
            self._initialise_timer.timeout.connect(self._ensure_renderer)

            self._stack = QtWidgets.QStackedWidget(self)
            self._message = QtWidgets.QLabel(self._stack)
            self._message.setObjectName("lamellar3DMessage")
            self._message.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self._message.setWordWrap(True)
            self._message.setStyleSheet(
                "QLabel#lamellar3DMessage { background: #f4f1ea; color: #24364b; "
                "border: 1px solid #d6d0c7; border-radius: 10px; padding: 18px; }"
            )
            self._stack.addWidget(self._message)
            layout = QtWidgets.QVBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(self._stack)
            self._set_message("Preparing the native 3-D renderer…")
            # Module probing is inexpensive and does not allocate a graphics
            # surface.  The QQuickWidget itself is created on first show/use.
            self._probe_modules()

        @property
        def bounds(self) -> np.ndarray | None:
            return None if self._bounds is None else self._bounds.copy()

        @property
        def qml_path(self) -> Path:
            return _QML_PATH

        @property
        def publication_style(self) -> Any:
            return self._publication_style

        @property
        def publication_geometry_hash(self) -> str | None:
            if self._publication_mesh is None:
                return None
            value = _read(self._publication_mesh, ("geometry_hash",), None)
            return None if value in (None, "") else str(value)

        def _prepare_publication_geometry(
            self,
            scene: Any,
            *,
            bounds: Any = None,
            style: Any = None,
            cancel_event: Any = None,
        ) -> bool:
            """Build the Qt adapter for the approved publication mesh contract."""

            self._publication_active = False
            self._publication_groups = []
            self._publication_group_maps = []
            self._publication_geometry_refs = []
            self._publication_mesh = None
            self._publication_bounds = None
            self._publication_error = ""
            self._last_publication_fit_raw = None
            if PublicationStyle is None:
                self._publication_error = "PublicationStyle is unavailable."
                return False
            try:
                resolved_style = PublicationStyle.from_mapping(style) if style is not None else self._publication_style
                from ..publication_geometry import build_publication_mesh
                from .publication_qt_geometry import build_publication_geometry_groups

                if self._cancelled(cancel_event):
                    raise RuntimeError("publication geometry build cancelled")
                mesh = build_publication_mesh(
                    scene,
                    style=resolved_style,
                    quality="publication",
                    cancel_event=cancel_event,
                )
                groups, normalized_bounds = build_publication_geometry_groups(
                    mesh,
                    bounds=bounds,
                    cancel_event=cancel_event,
                )
                if not groups:
                    raise ValueError("publication mesh contains no triangles")
                self._publication_style = resolved_style
                self._publication_mesh = mesh
                self._publication_groups = groups
                self._publication_geometry_refs = [group.geometry for group in groups]
                self._publication_bounds = np.asarray(normalized_bounds, dtype=float)
                self._publication_group_maps = [
                    {
                        "geometry": group.geometry,
                        "branch": int(group.branch_id),
                        "triangleCount": int(group.triangle_count),
                        "r": float(group.color[0]),
                        "g": float(group.color[1]),
                        "b": float(group.color[2]),
                    }
                    for group in groups
                ]
                self._publication_active = True
                return True
            except Exception as exc:  # pragma: no cover - optional renderer contract
                if self._cancelled(cancel_event):
                    raise RuntimeError("publication geometry build cancelled") from exc
                self._publication_error = f"Publication Qt geometry unavailable: {exc}"
                return False

        def set_publication_style(self, style: Any, *, cancel_event: Any = None) -> None:
            """Set the appearance policy and rebuild only presentation geometry."""

            if PublicationStyle is None:
                raise RuntimeError("PublicationStyle is unavailable")
            self._publication_style = PublicationStyle.from_mapping(style)
            if self._scene is not None:
                self._prepare_publication_geometry(
                    self._scene,
                    bounds=self._bounds,
                    style=self._publication_style,
                    cancel_event=cancel_event,
                )
                if self._qml_root is not None:
                    self._apply_root_state()

        def fit_view(self, camera: Mapping[str, Any] | None = None) -> dict[str, Any]:
            """Fit the publication view to the current fixed scene envelope."""

            if camera is not None:
                self.set_camera_state(camera)
                return self.camera_state()
            fitted = self._fit_camera()
            self._camera_state.update(fitted)
            if self._qml_root is not None:
                self._apply_root_state()
            self.cameraChanged.emit(self.camera_state())
            return self.camera_state()

        def balanced_view(self) -> dict[str, Any]:
            """Alias used by publication controls for the default balanced view."""

            return self.fit_view()

        def _probe_modules(self) -> bool:
            if self._renderer_attempted and self._qt_modules is not None:
                return True
            try:
                from PySide6 import QtQuick3D, QtQuickWidgets

                # Retain module references so plugin types cannot be unloaded
                # while the QML engine is alive.
                self._qt_modules = (QtQuickWidgets, QtQuick3D)
                self.error_message = ""
                return True
            except Exception as exc:  # pragma: no cover - host dependent
                self._qt_modules = None
                self.available = False
                self.error_message = f"Qt Quick 3D is unavailable: {exc}"
                self._renderer_failed = True
                self._set_message(self.error_message)
                self._renderer_attempted = True
                return False

        def _set_message(self, message: str) -> None:
            self._message.setText(str(message))
            if self._stack is not None:
                self._stack.setCurrentWidget(self._message)

        def _show_renderer(self) -> None:
            if self._quick is not None:
                self._stack.setCurrentWidget(self._quick)

        def _ensure_renderer(self) -> bool:
            if self._quick is not None and self._qml_root is not None and self.available:
                return True
            if self._renderer_failed:
                return False
            if not self._probe_modules():
                return False
            if QtWidgets.QApplication.instance() is None:
                self.available = False
                self.error_message = "A QApplication is required before creating the 3-D renderer."
                self._renderer_failed = True
                self._set_message(self.error_message)
                return False
            if not _QML_PATH.exists():
                self.available = False
                self.error_message = f"Missing Qt Quick 3D scene: {_QML_PATH}"
                self._renderer_failed = True
                self._set_message(self.error_message)
                return False
            try:
                QtQuickWidgets = self._qt_modules[0]
                quick = QtQuickWidgets.QQuickWidget(self._stack)
                quick.setObjectName("lamellar3DQuickWidget")
                quick.setResizeMode(QtQuickWidgets.QQuickWidget.ResizeMode.SizeRootObjectToView)
                quick.setClearColor(QtGui.QColor("#f4f1ea"))
                quick.statusChanged.connect(self._on_qml_status)
                self._stack.addWidget(quick)
                self._quick = quick
                quick.setSource(QtCore.QUrl.fromLocalFile(str(_QML_PATH)))
                # A source can become Ready synchronously, but processing one
                # bounded event turn also covers offscreen Windows RHI setup.
                QtWidgets.QApplication.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents, 100)
                root = quick.rootObject()
                if root is None:
                    errors = self._qml_errors(quick)
                    raise RuntimeError(errors or "Qt Quick 3D scene has no root object")
                self._qml_root = root
                window = quick.quickWindow()
                if window is not None:
                    window.sceneGraphError.connect(self._on_scene_graph_error)
                backend_error = self._rhi_backend_error(quick)
                if backend_error:
                    raise RuntimeError(backend_error)
                if hasattr(root, "cameraChanged"):
                    root.cameraChanged.connect(self._on_qml_camera_changed)
                self.available = True
                self.error_message = ""
                self._apply_root_state()
                self._show_renderer()
                return True
            except Exception as exc:  # pragma: no cover - renderer/driver dependent
                self.available = False
                self._renderer_failed = True
                self.error_message = f"Qt Quick 3D renderer could not start: {exc}"
                errors = self._qml_errors(self._quick)
                if errors:
                    self.error_message = f"{self.error_message}\n{errors}"
                self._set_message(self.error_message)
                return False

        @staticmethod
        def _rhi_backend_error(quick: Any) -> str:
            """Reject a software/unknown backend that cannot render Qt3D."""

            try:
                window = quick.quickWindow()
                renderer = window.rendererInterface() if window is not None else None
                api = renderer.graphicsApi() if renderer is not None else None
                text = str(api or "").casefold()
            except Exception:
                return "Qt Quick 3D has no usable graphics backend."
            if any(token in text for token in ("software", "unknown", "null")):
                return (
                    "Qt Quick 3D requires a native RHI graphics backend; "
                    f"the current backend is {api}."
                )
            return ""

        def _on_scene_graph_error(self, _error: Any, message: str) -> None:
            self.available = False
            self._renderer_failed = True
            self.error_message = f"Qt Quick 3D scene-graph error: {message}"
            self._set_message(self.error_message)

        @staticmethod
        def _qml_errors(quick: Any) -> str:
            if quick is None:
                return ""
            try:
                errors = quick.errors()
                return "\n".join(str(error) for error in errors)
            except Exception:
                return ""

        def _on_qml_status(self, status: Any) -> None:
            if self._quick is None:
                return
            try:
                ready = status == self._quick.Status.Ready
                error = status == self._quick.Status.Error
            except Exception:
                ready = False
                error = False
            if error:
                self.available = False
                self.error_message = self._qml_errors(self._quick) or "Qt Quick 3D scene failed to load."
                self._set_message(self.error_message)
            elif ready and self._qml_root is None:
                self._qml_root = self._quick.rootObject()
                if self._qml_root is not None and hasattr(self._qml_root, "cameraChanged"):
                    self._qml_root.cameraChanged.connect(self._on_qml_camera_changed)
                self.available = self._qml_root is not None
                if self.available:
                    self._apply_root_state()
                    self._show_renderer()

        def _on_qml_camera_changed(self, state: Any) -> None:
            if getattr(self, "_updating_camera", False):
                return
            parsed = self._parse_camera_state(state)
            if parsed:
                self._camera_state.update(parsed)
                if self._publication_active and self._bounds is not None:
                    raw = dict(self._camera_state)
                    raw["target"] = list(self._camera_target_raw(raw))
                    span = np.asarray(self._bounds[1] - self._bounds[0], dtype=float)
                    world_scale = 8.0 / max(float(np.max(span)), 1.0e-12)
                    raw["distance"] = float(raw.get("distance", 43.75)) / world_scale
                    raw["viewport_bounds"] = np.asarray(self._bounds, dtype=float).tolist()
                    self._publication_camera_raw = raw
                self.cameraChanged.emit(self.camera_state())

        @staticmethod
        def _parse_target(value: Any) -> tuple[float, float, float] | None:
            if isinstance(value, Mapping):
                values = (value.get("x"), value.get("y"), value.get("z"))
            else:
                try:
                    values = (value[0], value[1], value[2])
                except (TypeError, IndexError, KeyError):
                    return None
            try:
                result = tuple(float(item) for item in values)
            except (TypeError, ValueError):
                return None
            return result if all(math.isfinite(item) for item in result) else None  # type: ignore[return-value]

        def _parse_camera_state(self, state: Any) -> dict[str, Any]:
            # QML `var` signals arrive as QJSValue on PySide6.  Native mouse
            # gestures must update the same plain state used by save/export.
            if not isinstance(state, Mapping) and callable(getattr(state, "toVariant", None)):
                state = state.toVariant()
            if not isinstance(state, Mapping):
                return {}
            parsed: dict[str, Any] = {}
            for key in ("yaw", "pitch", "distance", "ortho_height"):
                try:
                    value = float(state[key])
                except (KeyError, TypeError, ValueError):
                    continue
                if math.isfinite(value):
                    parsed[key] = value
            target = self._parse_target(state.get("target"))
            if target is not None:
                parsed["target"] = target
            return parsed

        def _apply_root_state(self) -> None:
            root = self._qml_root
            if root is None:
                return
            payload = self._payload
            items = payload.items if payload is not None else []
            root.setProperty("slabData", items)
            root.setProperty("selectedBranch", self._selected_branch if self._selected_branch is not None else _NO_BRANCH)
            root.setProperty("language", self._language)
            root.setProperty("publicationActive", bool(self._publication_active))
            root.setProperty("publicationGroups", self._publication_group_maps)
            root.setProperty("publicationRoughness", float(getattr(self._publication_style, "roughness", 0.72)))
            root.setProperty("publicationAO", bool(getattr(self._publication_style, "ambient_occlusion", True)))
            root.setProperty("publicationQuality", "preview")
            root.setProperty("publicationMode", False)
            root.setProperty("showOverlays", True)
            root.setProperty("transparentBackground", False)
            if self._quick is not None:
                self._quick.setClearColor(
                    QtGui.QColor("#ffffff" if self._publication_active else "#f4f1ea")
                )
            if payload is not None:
                root.setProperty("axisLength", float(payload.axis_length))
            state = dict(self._camera_state)
            self._updating_camera = True
            try:
                root.setProperty("cameraYaw", float(state["yaw"]))
                root.setProperty("cameraPitch", float(state["pitch"]))
                root.setProperty("cameraDistance", float(state["distance"]))
                root.setProperty("orthoHeight", float(state["ortho_height"]))
                target = state.get("target", (0.0, 0.0, 0.0))
                root.setProperty("targetX", float(target[0]))
                root.setProperty("targetY", float(target[1]))
                root.setProperty("targetZ", float(target[2]))
            finally:
                self._updating_camera = False

        def _fit_publication_camera_for_size(self, width: int, height: int, *, update_state: bool = True) -> dict[str, Any]:
            if not self._publication_active or self._scene is None or self._bounds is None:
                return {"distance": 43.75, "ortho_height": 35.0, "target": (0.0, 0.0, 0.0)}
            try:
                from ..publication_geometry import fit_publication_camera

                fitted = fit_publication_camera(
                    self._scene,
                    bounds=self._bounds,
                    width=max(1, int(width)),
                    height=max(1, int(height)),
                )
                raw_target = np.asarray(fitted.get("target", (0.0, 0.0, 0.0)), dtype=float)
                branch_values = np.asarray(_read(self._scene, ("branch_ids",), ()), dtype=int).reshape(-1)
                fitted_yaw = float(fitted.get("yaw", -38.0))
                fitted_pitch = float(fitted.get("pitch", -26.0))
                # A single flat family is more legible in its broad face than
                # edge-on. Multi-branch family separation keeps the fitter's
                # candidate view.
                if len(np.unique(branch_values)) <= 1 and abs(fitted_yaw) >= 70.0:
                    fitted_yaw, fitted_pitch = 0.0, -12.0
                limits = np.asarray(self._bounds, dtype=float)
                span = limits[1] - limits[0]
                world_scale = 8.0 / max(float(np.max(span)), 1.0e-12)
                origin = (limits[0] + limits[1]) / 2.0
                target = tuple(float(value) for value in ((raw_target - origin) * world_scale))
                raw_state = dict(fitted)
                raw_state.update({"yaw": fitted_yaw, "pitch": fitted_pitch})
                raw_state["viewport_bounds"] = np.asarray(self._bounds, dtype=float).tolist()
                self._last_publication_fit_raw = raw_state
                if update_state:
                    self._publication_camera_raw = raw_state
                return {
                    "yaw": fitted_yaw,
                    "pitch": fitted_pitch,
                    "distance": 43.75,
                    "ortho_height": float(fitted.get("ortho_height", 35.0)),
                    "target": target,
                }
            except Exception:
                return {"distance": 43.75, "ortho_height": 35.0, "target": (0.0, 0.0, 0.0)}

        def _fit_camera(self) -> dict[str, Any]:
            if self._payload is None:
                return {
                    "distance": 12.0,
                    "ortho_height": 8.8,
                    "target": (0.0, 0.0, 0.0),
                }
            if self._publication_active and self._scene is not None:
                return self._fit_publication_camera_for_size(
                    max(300, int(self.width() or 800)),
                    max(220, int(self.height() or 600)),
                )
            # The adapter normalizes the fixed scene extent to eight world
            # units.  Keep the established fallback fit for older scenes or
            # hosts where the publication helper is unavailable.
            ortho = 35.0
            return {"distance": 43.75, "ortho_height": ortho, "target": (0.0, 0.0, 0.0)}

        def _process_events(self, timeout_ms: int = 180) -> None:
            app = QtWidgets.QApplication.instance()
            if app is not None:
                app.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents, int(timeout_ms))

        def showEvent(self, event: Any) -> None:  # noqa: N802 - Qt API
            super().showEvent(event)
            # D3D/COM initialization cannot safely call out from a Windows
            # synchronous show callback (RPC_E_CANTCALLOUT_ININPUTSYNCCALL).
            self._initialise_timer.start(0)

        def set_scene(
            self,
            scene: Any,
            *,
            bounds: Any = None,
            reset_camera: bool = False,
            cancel_event: Any = None,
        ) -> None:
            """Display a core lamellar scene while preserving camera state.

            ``bounds`` is intended for a fixed sequence extent.  When supplied,
            it is retained exactly as the view's camera fitting reference even
            if a later frame contains fewer slabs.
            """

            self._scene = scene
            try:
                payload = _scene_payload(scene, bounds)
            except Exception as exc:
                self._payload = None
                self._bounds = _finite_bounds(bounds)
                self._has_scene = False
                self.available = bool(self._quick is not None and self._qml_root is not None)
                self.error_message = f"Lamellar scene cannot be displayed: {exc}"
                self._set_message(self.error_message)
                if self._qml_root is not None:
                    self._qml_root.setProperty("slabData", [])
                return
            self._payload = payload
            self._bounds = payload.bounds.copy()
            self._prepare_publication_geometry(
                scene,
                bounds=self._bounds,
                style=self._publication_style,
                cancel_event=cancel_event,
            )
            first_scene = not self._has_scene
            if reset_camera or first_scene:
                fitted = self._fit_camera()
                self._camera_state.update(fitted)
                if "yaw" not in fitted or "pitch" not in fitted:
                    self._camera_state["yaw"], self._camera_state["pitch"] = _STANDARD_VIEWS["isometric"]
            self._has_scene = True
            if self._qml_root is not None:
                self._apply_root_state()
                if self.available:
                    # ``clear(message=...)`` intentionally switches the
                    # stacked widget to its status page.  A later frame can
                    # reuse the existing QML root; restore the renderer page
                    # explicitly instead of leaving the placeholder visible.
                    self._show_renderer()
            elif self.isVisible() and self._ensure_renderer():
                self._apply_root_state()
            else:
                # Keep a valid scene pending so the first show/explicit capture
                # can render it without the caller having to resend the frame.
                if self.error_message:
                    self._set_message(self.error_message)

        def clear(self, message: str = "") -> None:
            """Remove all slabs and optionally show a neutral state message."""

            self._scene = None
            self._payload = None
            self._bounds = None
            self._has_scene = False
            self._publication_mesh = None
            self._publication_groups = []
            self._publication_group_maps = []
            self._publication_geometry_refs = []
            self._publication_bounds = None
            self._publication_active = False
            self._publication_camera_raw = None
            self._last_publication_fit_raw = None
            if self._qml_root is not None:
                self._qml_root.setProperty("slabData", [])
                self._qml_root.setProperty("publicationActive", False)
                self._qml_root.setProperty("publicationGroups", [])
            if message:
                self._set_message(message)
            elif self.available:
                self._show_renderer()
            else:
                self._set_message("Preparing the native 3-D renderer…")

        def camera_state(self) -> dict[str, Any]:
            """Return a JSON-friendly snapshot of the current orbit camera."""

            state = dict(self._camera_state)
            state["target"] = tuple(float(item) for item in state.get("target", (0.0, 0.0, 0.0)))
            return state

        def publication_camera_state(self) -> dict[str, Any]:
            """Return publication camera values in the original scene units."""

            state = self.camera_state()
            if self._bounds is not None:
                state["target"] = list(self._camera_target_raw(state))
                span = np.asarray(self._bounds[1] - self._bounds[0], dtype=float)
                world_scale = 8.0 / max(float(np.max(span)), 1.0e-12)
                state["distance"] = float(state.get("distance", 43.75)) / world_scale
                state["viewport_bounds"] = np.asarray(self._bounds, dtype=float).tolist()
                return state
            return dict(self._publication_camera_raw or state)

        def set_camera_state(self, state: Mapping[str, Any]) -> None:
            if not isinstance(state, Mapping):
                raise TypeError("camera state must be a mapping")
            parsed = self._parse_camera_state(state)
            if "zoom" in state and "ortho_height" not in parsed:
                try:
                    zoom = float(state["zoom"])
                    if math.isfinite(zoom) and zoom > 0.0:
                        parsed["ortho_height"] = self._fit_camera()["ortho_height"] * zoom
                except (TypeError, ValueError):
                    pass
            self._camera_state.update(parsed)
            if self._publication_active and self._bounds is not None:
                raw = dict(self._camera_state)
                raw["target"] = list(self._camera_target_raw(raw))
                span = np.asarray(self._bounds[1] - self._bounds[0], dtype=float)
                world_scale = 8.0 / max(float(np.max(span)), 1.0e-12)
                raw["distance"] = float(raw.get("distance", 43.75)) / world_scale
                raw["viewport_bounds"] = np.asarray(self._bounds, dtype=float).tolist()
                self._publication_camera_raw = raw
            if self._qml_root is not None:
                self._apply_root_state()
            elif self.isVisible() and self._ensure_renderer():
                self._apply_root_state()
            self.cameraChanged.emit(self.camera_state())

        def set_standard_view(self, name: str) -> None:
            key = str(name).strip().lower()
            if key not in _STANDARD_VIEWS:
                raise ValueError(f"unknown lamellar camera view: {name!r}")
            fitted = self._fit_camera()
            self._camera_state.update(fitted)
            self._camera_state["yaw"], self._camera_state["pitch"] = _STANDARD_VIEWS[key]
            if self._qml_root is not None:
                self._apply_root_state()
            elif self.isVisible() and self._ensure_renderer():
                self._apply_root_state()
            self.cameraChanged.emit(self.camera_state())

        def reset_camera(self) -> None:
            self.set_standard_view("isometric")

        def set_language(self, language: str) -> None:
            value = str(language).strip().lower()
            self._language = "zh" if value.startswith("zh") else "en"
            if self._qml_root is not None:
                self._qml_root.setProperty("language", self._language)
            if not self.available and self.error_message:
                self._set_message(self.error_message)

        def set_selected_branch(self, branch: int) -> None:
            value = int(branch)
            self._selected_branch = None if value < 0 else value
            if self._qml_root is not None:
                self._qml_root.setProperty("selectedBranch", self._selected_branch if self._selected_branch is not None else _NO_BRANCH)

        def _make_capture_widget(self, *, wait_ms: int = 120) -> Any:
            """Create an independent QQuickWidget for true-size capture."""

            if self._qt_modules is None:
                return None
            QtQuickWidgets = self._qt_modules[0]
            quick = QtQuickWidgets.QQuickWidget()
            quick.setAttribute(QtCore.Qt.WidgetAttribute.WA_DontShowOnScreen, True)
            quick.setResizeMode(QtQuickWidgets.QQuickWidget.ResizeMode.SizeRootObjectToView)
            quick.setClearColor(QtGui.QColor("#f4f1ea"))
            quick.setSource(QtCore.QUrl.fromLocalFile(str(_QML_PATH)))
            if wait_ms > 0:
                self._process_events(wait_ms)
            return quick

        def _capture_group_maps(self, capture: Any, cancel_event: Any = None) -> list[dict[str, Any]]:
            """Each Qt scene owns its render objects; the CPU mesh is shared."""
            if not self._publication_active:
                return []
            from .publication_qt_geometry import build_publication_geometry_groups

            groups, _ = build_publication_geometry_groups(
                self._publication_mesh, bounds=self._bounds, cancel_event=cancel_event)
            # A QQuick3DGeometry cannot be attached to two windows at once.
            # Keep these independent objects alive for this capture only.
            capture._publication_geometry_refs = groups
            return [{"geometry": group.geometry, "branch": int(group.branch_id),
                     "triangleCount": int(group.triangle_count), "r": float(group.color[0]),
                     "g": float(group.color[1]), "b": float(group.color[2])} for group in groups]

        def capture_image(self, width: int = 2400, height: int = 1800) -> Any:
            """Render and return a true-size ``QImage`` of the current scene."""

            try:
                width = int(width)
                height = int(height)
            except (TypeError, ValueError) as exc:
                raise ValueError("capture dimensions must be positive integers") from exc
            if width <= 0 or height <= 0:
                raise ValueError("capture dimensions must be positive integers")
            if not self._ensure_renderer():
                message = self.error_message or "Qt Quick 3D renderer is unavailable; no 3-D image was rendered."
                raise RuntimeError(message)
            capture = None
            try:
                capture = self._make_capture_widget()
                if capture is None:
                    raise RuntimeError("Qt Quick 3D renderer is unavailable; no 3-D image was rendered.")
                capture.setClearColor(QtGui.QColor("#ffffff" if self._publication_active else "#f4f1ea"))
                capture.resize(width, height)
                capture.show()
                self._process_events(260)
                root = capture.rootObject()
                if root is None:
                    raise RuntimeError(self._qml_errors(capture) or "capture scene has no root object")
                root.setProperty("slabData", self._payload.items if self._payload is not None else [])
                root.setProperty("selectedBranch", self._selected_branch if self._selected_branch is not None else _NO_BRANCH)
                root.setProperty("language", self._language)
                root.setProperty("publicationActive", bool(self._publication_active))
                root.setProperty("publicationGroups", self._capture_group_maps(capture))
                root.setProperty("publicationRoughness", float(getattr(self._publication_style, "roughness", 0.72)))
                root.setProperty("publicationAO", bool(getattr(self._publication_style, "ambient_occlusion", True)))
                root.setProperty("publicationQuality", "preview")
                root.setProperty("publicationMode", False)
                root.setProperty("showOverlays", True)
                root.setProperty("transparentBackground", False)
                root.setProperty("axisLength", self._payload.axis_length if self._payload is not None else 1.0)
                state = self._camera_state
                for key, value in (
                    ("cameraYaw", state["yaw"]),
                    ("cameraPitch", state["pitch"]),
                    ("cameraDistance", state["distance"]),
                    ("orthoHeight", state["ortho_height"]),
                ):
                    root.setProperty(key, float(value))
                target = state.get("target", (0.0, 0.0, 0.0))
                root.setProperty("targetX", float(target[0]))
                root.setProperty("targetY", float(target[1]))
                root.setProperty("targetZ", float(target[2]))
                self._process_events(260)
                # ``QQuickWidget`` sizes itself in logical pixels.  On a
                # Windows display with 150% scaling, for example, a 320x240
                # widget produces a 480x360 framebuffer.  Render at a logical
                # size that is at least the requested physical size, then
                # crop the freshly rendered framebuffer if DPI rounding made
                # it one or two pixels larger.  This keeps the returned image
                # exact-size without upscaling a screenshot.
                dpr = max(1.0, float(capture.devicePixelRatioF()))
                logical_width = max(1, int(math.ceil(width / dpr)))
                logical_height = max(1, int(math.ceil(height / dpr)))
                image = None
                for _ in range(3):
                    capture.resize(logical_width, logical_height)
                    self._process_events(180)
                    candidate = capture.grabFramebuffer()
                    if candidate is None or candidate.isNull():
                        image = candidate
                        break
                    image = candidate
                    if image.width() >= width and image.height() >= height:
                        break
                    logical_width += max(1, int(math.ceil((width - image.width()) / dpr)))
                    logical_height += max(1, int(math.ceil((height - image.height()) / dpr)))
                if image is None or image.isNull():
                    raise RuntimeError("Qt Quick 3D returned an empty framebuffer")
                if image.width() != width or image.height() != height:
                    if image.width() < width or image.height() < height:
                        raise RuntimeError(
                            f"Qt Quick 3D returned {image.width()}x{image.height()} pixels; "
                            f"requested {width}x{height}"
                        )
                    image = image.copy(0, 0, width, height)
                return image.copy()
            except Exception as exc:  # pragma: no cover - renderer/driver dependent
                self.error_message = f"Qt Quick 3D capture failed: {exc}"
                self.available = False
                self._set_message(self.error_message)
                raise RuntimeError(self.error_message) from exc
            finally:
                if capture is not None:
                    capture.hide()
                    capture.deleteLater()
                    self._process_events(60)

        @staticmethod
        def _cancelled(cancel_event: Any = None) -> bool:
            if cancel_event is None:
                return False
            try:
                method = getattr(cancel_event, "is_set", None)
                return bool(method()) if callable(method) else bool(cancel_event)
            except Exception:
                return False

        def _wait_publication_ready(self, quick: Any, *, cancel_event: Any = None, timeout_ms: int = 4_000) -> Any:
            """Wait for an actual QQuickWidget frame without a fixed sleep."""

            deadline = time.monotonic() + max(0.1, float(timeout_ms) / 1000.0)
            root = None
            while time.monotonic() < deadline:
                if self._cancelled(cancel_event):
                    raise RuntimeError("publication capture cancelled")
                try:
                    status = quick.status()
                    if status == quick.Status.Error:
                        raise RuntimeError(self._qml_errors(quick) or "publication QML scene failed to load")
                    root = quick.rootObject()
                    if status == quick.Status.Ready and root is not None:
                        return root
                except RuntimeError:
                    raise
                except Exception:
                    pass
                self._process_events(16)
            raise RuntimeError(self._qml_errors(quick) or "timed out waiting for publication Qt3D scene")

        @staticmethod
        def _qimage_rgba(image: Any) -> np.ndarray:
            converted = image.convertToFormat(QtGui.QImage.Format.Format_RGBA8888)
            pixels = np.frombuffer(bytes(converted.constBits()), dtype=np.uint8)
            rows = pixels.reshape(converted.height(), converted.bytesPerLine())
            return rows[:, : converted.width() * 4].reshape(converted.height(), converted.width(), 4).copy()

        def _camera_target_raw(self, state: Mapping[str, Any]) -> tuple[float, float, float]:
            limits = np.asarray(self._bounds, dtype=float)
            span = limits[1] - limits[0]
            world_scale = 8.0 / max(float(np.max(span)), 1.0e-12)
            target = np.asarray(state.get("target", (0.0, 0.0, 0.0)), dtype=float)
            return tuple(float(value) for value in ((limits[0] + limits[1]) / 2.0 + target / world_scale))

        def _camera_target_normalized(self, state: Mapping[str, Any]) -> tuple[float, float, float]:
            limits = np.asarray(self._bounds, dtype=float)
            span = limits[1] - limits[0]
            world_scale = 8.0 / max(float(np.max(span)), 1.0e-12)
            target = np.asarray(state.get("target", (0.0, 0.0, 0.0)), dtype=float)
            return tuple(float(value) for value in ((target - (limits[0] + limits[1]) / 2.0) * world_scale))

        def _publication_projection(
            self,
            scene: Any,
            camera: Mapping[str, Any],
            *,
            normalized_camera: bool = False,
            width: int | None = None,
            height: int | None = None,
        ) -> dict[str, Any]:
            """Build publisher-facing normalized projections when the core helper exists."""

            result: dict[str, Any] = {"stack_centers": [], "normal_vectors": []}
            try:
                from ..publication_geometry import camera_basis, project_points

                centers = np.asarray(_read(scene, ("centers",), np.empty((0, 3))), dtype=float)
                points = np.asarray(centers, dtype=float).reshape((-1, 3))
                raw_camera = dict(camera)
                if normalized_camera:
                    raw_camera["target"] = list(self._camera_target_raw(raw_camera))
                projected = project_points(
                    points,
                    camera=raw_camera,
                    bounds=self._bounds,
                    width=max(1, int(width or self.width() or 800)),
                    height=max(1, int(height or self.height() or 600)),
                )
                if isinstance(projected, Mapping):
                    values = projected.get("points", projected.get("xy", projected))
                else:
                    values = projected
                values = np.asarray(values, dtype=float).reshape((-1, 2))
                stacks = np.asarray(_read(scene, ("stack_ids",), np.arange(len(values))), dtype=int).reshape(-1)
                branches = np.asarray(_read(scene, ("branch_ids",), np.zeros(len(values))), dtype=int).reshape(-1)
                result["stack_centers"] = [
                    {"stack_id": int(stacks[index]), "branch_id": int(branches[index]), "xy": [float(values[index, 0]), float(values[index, 1])]}
                    for index in range(min(len(values), len(stacks), len(branches)))
                ]
                basis = camera_basis(camera)
                result["camera_basis"] = np.asarray(basis, dtype=float).tolist()
            except Exception:
                # Projection is supplementary publisher metadata.  The render
                # itself remains valid when an older geometry helper lacks the
                # optional projection convenience functions.
                pass
            return result

        def capture_publication(
            self,
            scene: Any = None,
            *,
            style: Any = None,
            width: int = 2400,
            height: int = 1800,
            camera: Mapping[str, Any] | None = None,
            quality: str = "publication",
            transparent: bool = True,
            cancel_event: Any = None,
        ) -> Any:
            """Render a controls-free, literal-resolution publication image."""

            if PublicationRenderResult is None:
                raise RuntimeError("PublicationRenderResult is unavailable")
            try:
                width, height = int(width), int(height)
            except (TypeError, ValueError) as exc:
                raise ValueError("publication capture dimensions must be positive integers") from exc
            if width <= 0 or height <= 0:
                raise ValueError("publication capture dimensions must be positive integers")
            if not isinstance(cancel_event, _ResponsiveCancellation):
                cancel_event = _ResponsiveCancellation(cancel_event)
            if width > 8192 or height > 8192 or width * height > 64_000_000:
                raise ValueError("publication capture is limited to 8192x8192 and 64 megapixels")
            if quality not in {"preview", "publication"}:
                raise ValueError("quality must be 'preview' or 'publication'")
            if self._cancelled(cancel_event):
                raise RuntimeError("publication capture cancelled")
            camera_bounds = _camera_viewport_bounds(camera)
            if scene is not None and scene is not self._scene:
                self.set_scene(
                    scene,
                    bounds=camera_bounds if camera_bounds is not None else _read(scene, ("bounds",), None),
                    reset_camera=True,
                    cancel_event=cancel_event,
                )
            scene = self._scene if scene is None else scene
            if scene is None:
                raise ValueError("publication capture requires a scene")
            if camera_bounds is not None and self._bounds is not None and not np.allclose(camera_bounds, self._bounds):
                self.set_scene(scene, bounds=camera_bounds, reset_camera=False, cancel_event=cancel_event)
            if style is not None:
                self.set_publication_style(style, cancel_event=cancel_event)
            if not self._publication_active:
                self._prepare_publication_geometry(
                    scene,
                    bounds=self._bounds,
                    style=self._publication_style,
                    cancel_event=cancel_event,
                )
            if not self._publication_active:
                raise RuntimeError(self._publication_error or "publication geometry is unavailable")
            if not self._ensure_renderer():
                raise RuntimeError(self.error_message or "Qt Quick 3D renderer is unavailable")
            capture = None
            try:
                capture = self._make_capture_widget(wait_ms=0)
                if capture is None:
                    raise RuntimeError("Qt Quick 3D capture widget is unavailable")
                capture.setClearColor(QtGui.QColor(0, 0, 0, 0) if transparent else QtGui.QColor("#ffffff"))
                capture.show()
                root = self._wait_publication_ready(capture, cancel_event=cancel_event)
                if camera is None:
                    camera_state = self._fit_publication_camera_for_size(
                        width,
                        height,
                        update_state=False,
                    )
                    raw_camera = dict(self._last_publication_fit_raw or self.publication_camera_state())
                    qml_camera = camera_state
                else:
                    raw_camera = dict(camera)
                    camera_state = dict(camera)
                    camera_state["target"] = self._camera_target_normalized(raw_camera)
                    if self._bounds is not None:
                        span = np.asarray(self._bounds[1] - self._bounds[0], dtype=float)
                        world_scale = 8.0 / max(float(np.max(span)), 1.0e-12)
                        camera_state["distance"] = min(200.0, max(5.0, float(raw_camera.get("distance", 43.75)) * world_scale))
                    qml_camera = camera_state
                for name, value in (
                    ("publicationActive", True),
                    ("publicationMode", True),
                    ("publicationQuality", quality),
                    ("showOverlays", False),
                    ("transparentBackground", bool(transparent)),
                    ("publicationGroups", self._capture_group_maps(capture, cancel_event)),
                    ("publicationRoughness", float(getattr(self._publication_style, "roughness", 0.72))),
                    ("publicationAO", bool(getattr(self._publication_style, "ambient_occlusion", True))),
                    ("slabData", []),
                ):
                    root.setProperty(name, value)
                target = qml_camera.get("target", (0.0, 0.0, 0.0))
                for name, value in (
                    ("cameraYaw", camera_state.get("yaw", -38.0)),
                    ("cameraPitch", camera_state.get("pitch", -26.0)),
                    ("cameraDistance", camera_state.get("distance", 43.75)),
                    ("orthoHeight", camera_state.get("ortho_height", 35.0)),
                    ("targetX", target[0]),
                    ("targetY", target[1]),
                    ("targetZ", target[2]),
                ):
                    root.setProperty(name, float(value))
                dpr = max(1.0, float(capture.devicePixelRatioF()))
                logical_width = max(1, int(math.ceil(width / dpr)))
                logical_height = max(1, int(math.ceil(height / dpr)))
                image = None
                deadline = time.monotonic() + 8.0
                while time.monotonic() < deadline:
                    if self._cancelled(cancel_event):
                        raise RuntimeError("publication capture cancelled")
                    capture.resize(logical_width, logical_height)
                    self._process_events(16)
                    candidate = capture.grabFramebuffer()
                    if candidate is not None and not candidate.isNull() and candidate.width() >= width and candidate.height() >= height:
                        image = candidate
                        break
                    self._process_events(16)
                if image is None or image.isNull():
                    raise RuntimeError("publication Qt3D did not produce a framebuffer before timeout")
                if image.width() != width or image.height() != height:
                    if image.width() < width or image.height() < height:
                        raise RuntimeError(f"publication framebuffer is {image.width()}x{image.height()}, requested {width}x{height}")
                    image = image.copy(0, 0, width, height)
                rgba = self._qimage_rgba(image)
                if transparent and not np.any(rgba[:, :, 3] > 0):
                    raise RuntimeError("publication Qt3D returned a fully transparent framebuffer")
                if not transparent:
                    background = np.asarray((255, 255, 255), dtype=np.uint8)
                    if not np.any(np.any(rgba[:, :, :3] != background, axis=2)):
                        raise RuntimeError("publication Qt3D returned a blank background framebuffer")
                if not transparent:
                    rgba[:, :, 3] = 255
                window = capture.quickWindow()
                rhi = None
                try:
                    rhi = str(window.rendererInterface().graphicsApi()) if window is not None else None
                except Exception:
                    rhi = None
                provenance = {
                    "renderer": "qtquick3d_publication",
                    "renderer_backend": "qtquick3d",
                    "rhi_backend": rhi,
                    "geometry_hash": self.publication_geometry_hash,
                    "source_geometry_hash": _read(self._publication_mesh, ("metadata",), {}).get("source_geometry_hash") if isinstance(_read(self._publication_mesh, ("metadata",), {}), Mapping) else None,
                    "style": getattr(self._publication_style, "to_dict", lambda: {})(),
                    "quality": quality,
                    "image_size": [int(width), int(height)],
                    "resolution": [int(width), int(height)],
                    "transparent": bool(transparent),
                    "antialiasing": "SSAA" if quality == "publication" else "MSAA",
                    "controls_burned_in": False,
                }
                if self._bounds is None:
                    raise RuntimeError("publication viewport bounds are unavailable")
                bounds = np.asarray(self._bounds, dtype=float).copy()
                return PublicationRenderResult(
                    rgba=rgba,
                    camera=dict(raw_camera),
                    bounds=bounds,
                    projected=self._publication_projection(
                        scene,
                        raw_camera,
                        normalized_camera=False,
                        width=width,
                        height=height,
                    ),
                    provenance=provenance,
                )
            except Exception as exc:
                raise RuntimeError(f"publication Qt3D capture failed: {exc}") from exc
            finally:
                if capture is not None:
                    capture.hide()
                    capture.deleteLater()
                    self._process_events(16)


else:

    class Lamellar3DView:  # pragma: no cover - only used without PySide6
        """Import-safe placeholder for a host without the optional UI stack."""

        cameraChanged = None

        def __init__(self, parent: Any = None) -> None:
            del parent
            self.available = False
            self.error_message = "PySide6 is unavailable; the native 3-D renderer cannot start."
            self._language = "en"
            self._scene = None
            self._bounds = None
            self._camera_state = {
                "yaw": _STANDARD_VIEWS["isometric"][0],
                "pitch": _STANDARD_VIEWS["isometric"][1],
                "distance": 12.0,
                "ortho_height": 8.8,
                "target": (0.0, 0.0, 0.0),
            }

        def set_scene(self, scene: Any, *, bounds: Any = None, reset_camera: bool = False) -> None:
            del reset_camera
            self._scene = scene
            self._bounds = _finite_bounds(bounds)

        def clear(self, message: str = "") -> None:
            del message
            self._scene = None

        def camera_state(self) -> dict[str, Any]:
            return dict(self._camera_state)

        def set_camera_state(self, state: Mapping[str, Any]) -> None:
            if not isinstance(state, Mapping):
                raise TypeError("camera state must be a mapping")
            self._camera_state.update({key: state[key] for key in self._camera_state if key in state})

        def set_standard_view(self, name: str) -> None:
            key = str(name).strip().lower()
            if key not in _STANDARD_VIEWS:
                raise ValueError(f"unknown lamellar camera view: {name!r}")
            self._camera_state["yaw"], self._camera_state["pitch"] = _STANDARD_VIEWS[key]

        def reset_camera(self) -> None:
            self.set_standard_view("isometric")

        def set_language(self, language: str) -> None:
            self._language = "zh" if str(language).strip().lower().startswith("zh") else "en"

        def set_selected_branch(self, branch: int) -> None:
            del branch

        def set_publication_style(self, style: Any) -> None:
            del style
            raise RuntimeError("PySide6 is unavailable; publication Qt3D cannot start.")

        def fit_view(self, camera: Mapping[str, Any] | None = None) -> dict[str, Any]:
            if camera is not None:
                self.set_camera_state(camera)
            return self.camera_state()

        def balanced_view(self) -> dict[str, Any]:
            return self.fit_view()

        def capture_image(self, width: int = 2400, height: int = 1800) -> Any:
            del width, height
            raise RuntimeError("PySide6 is unavailable; no 3-D image was rendered.")

        def capture_publication(self, *args: Any, **kwargs: Any) -> Any:
            del args, kwargs
            raise RuntimeError("PySide6 is unavailable; publication Qt3D cannot start.")


def render_publication_scene(
    scene: Any,
    *,
    style: Any = None,
    width: int = 2400,
    height: int = 1800,
    camera: Mapping[str, Any] | None = None,
    quality: str = "publication",
    transparent: bool = True,
    cancel_event: Any = None,
) -> Any:
    """Render a publication scene through a minimal native Qt application.

    Qt Quick scene-graph creation is GUI-thread-only.  If a worker calls this
    function while an application belongs to another thread, fail explicitly
    so a caller cannot mistake a partial/static image for a publication render.
    """

    if not QT_AVAILABLE:
        raise RuntimeError("PySide6 is unavailable; publication Qt3D cannot start.")
    try:
        width, height = int(width), int(height)
    except (TypeError, ValueError) as exc:
        raise ValueError("publication capture dimensions must be positive integers") from exc
    if width <= 0 or height <= 0:
        raise ValueError("publication capture dimensions must be positive integers")
    if width > 8192 or height > 8192 or width * height > 64_000_000:
        raise ValueError("publication capture is limited to 8192x8192 and 64 megapixels")
    if Lamellar3DView._cancelled(cancel_event):
        raise RuntimeError("publication capture cancelled")
    viewport_bounds = _camera_viewport_bounds(camera)
    app = QtWidgets.QApplication.instance()
    if app is not None:
        current = QtCore.QThread.currentThread()
        if current != app.thread():
            raise RuntimeError("publication Qt3D rendering must run on the QApplication GUI thread")
    elif threading.current_thread() is not threading.main_thread():
        raise RuntimeError("publication Qt3D cannot create QApplication from a worker thread")
    if app is None:
        app = QtWidgets.QApplication([])
    cancel_event = _ResponsiveCancellation(cancel_event)
    view = Lamellar3DView()
    try:
        view.setAttribute(QtCore.Qt.WidgetAttribute.WA_DontShowOnScreen, True)
        view.resize(max(320, min(int(width), 1200)), max(240, min(int(height), 900)))
        view.set_scene(
            scene,
            bounds=viewport_bounds if viewport_bounds is not None else _read(scene, ("bounds",), None),
            reset_camera=True,
            cancel_event=cancel_event,
        )
        if style is not None:
            view.set_publication_style(style, cancel_event=cancel_event)
        view.show()
        app.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents, 16)
        return view.capture_publication(
            scene,
            style=style,
            width=width,
            height=height,
            camera=camera,
            quality=quality,
            transparent=transparent,
            cancel_event=cancel_event,
        )
    finally:
        view.close()
        app.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents, 16)


__all__ = ["Lamellar3DView", "render_publication_scene"]
