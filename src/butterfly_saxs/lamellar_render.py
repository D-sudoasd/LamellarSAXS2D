"""Qt-free rendering helpers for the LamellarSAXS2D geometry scene.

The functions in this module deliberately consume the public ``LamellarScene``
contract through attribute or mapping lookup.  Importing the scientific core is
therefore unnecessary, which keeps exports usable in batch jobs and avoids a
Qt dependency.  Every geometry mark is made from ``scene.vertices``; the
renderer never regenerates a second approximation from centre/size values.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib import font_manager
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


_OFFWHITE = "#faf8f4"
_INK = "#25323d"
_MUTED = "#64727d"
_GRID = "#d8d8d2"
_BLUE = np.asarray((0.20, 0.45, 0.72, 0.84), dtype=float)
_ORANGE = np.asarray((0.92, 0.47, 0.18, 0.84), dtype=float)
_PALETTE = (_BLUE, _ORANGE)
_FORMAT_VERSION = "lamellarsaxs2d.lamellar-render.v1"


def _get(scene: Any, name: str, default: Any = None) -> Any:
    if isinstance(scene, Mapping):
        return scene.get(name, default)
    return getattr(scene, name, default)


def _as_float_array(value: Any, shape: tuple[int, ...] | None = None) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        array = np.empty((0,), dtype=float)
    if shape is not None and array.shape != shape:
        return np.empty((0,), dtype=float)
    return array


def _vertices(scene: Any) -> np.ndarray:
    value = _get(scene, "vertices")
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return np.empty((0, 8, 3), dtype=float)
    if array.ndim != 3 or array.shape[1:] != (8, 3):
        return np.empty((0, 8, 3), dtype=float)
    finite = np.all(np.isfinite(array), axis=(1, 2))
    return array[finite]


def _centres(scene: Any, count: int) -> np.ndarray:
    value = _as_float_array(_get(scene, "centers"), (count, 3))
    if value.size:
        return value
    vertices = _vertices(scene)
    return vertices.mean(axis=1) if len(vertices) == count else np.zeros((count, 3))


def _status(scene: Any) -> str:
    value = _get(scene, "status", "schematic")
    text = str(value or "schematic").strip().lower()
    return text


def _message(scene: Any) -> str:
    return str(_get(scene, "message", "") or "").strip()


def _source_label(scene: Any) -> str:
    source = _get(scene, "source_identity", "")
    if isinstance(source, Mapping):
        for key in ("frame_id", "source_id", "id", "name", "path", "source"):
            value = source.get(key)
            if value not in (None, ""):
                return str(value)
        return "Lamellar scene"
    if source in (None, ""):
        metadata = _get(scene, "metadata", {})
        if isinstance(metadata, Mapping):
            for key in ("frame_id", "source_id", "id", "name", "path", "source"):
                value = metadata.get(key)
                if value not in (None, ""):
                    return str(value)
        return "Lamellar scene"
    return str(source)


def _unit(scene: Any) -> str:
    value = str(_get(scene, "length_unit", "relative") or "relative").strip()
    return value if value in {"nm", "relative"} else "relative"


def _unit_label(scene: Any) -> str:
    return "nm" if _unit(scene) == "nm" else "relative scale"


def _q_unit_label(scene: Any) -> str:
    metadata = _get(scene, "metadata", {})
    value = metadata.get("q_unit") if isinstance(metadata, Mapping) else None
    if value in (None, ""):
        value = _get(scene, "q_unit", None)
    return str(value or "unknown")


def _font() -> FontProperties:
    """Return a local CJK-capable font when one is present.

    The font is attached to individual text artists.  No matplotlib global
    font configuration is changed, which keeps this module safe for notebooks
    and applications that own their own style sheet.
    """

    candidates = (
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path(r"C:\Windows\Fonts\simsun.ttc"),
        Path(r"C:\Windows\Fonts\NotoSansCJK-Regular.ttc"),
    )
    for candidate in candidates:
        if candidate.is_file():
            try:
                return FontProperties(fname=str(candidate))
            except (OSError, ValueError):
                continue
    try:
        path = font_manager.findfont(
            FontProperties(family=["Noto Sans CJK SC", "Microsoft YaHei", "DejaVu Sans"]),
            fallback_to_default=True,
        )
        return FontProperties(fname=path)
    except (OSError, ValueError):  # pragma: no cover - defensive matplotlib path
        return FontProperties(family="DejaVu Sans")


def _title_suffix(scene: Any, language: str) -> str:
    status = _status(scene)
    source = _source_label(scene)
    if str(language).lower().startswith("zh"):
        labels = {
            "schematic": "参数驱动示意",
            "candidate": "候选参数驱动示意",
            "manual": "手动假设示意",
            "unavailable": "数据不可用",
            "stale": "结果已过期",
        }
    else:
        labels = {
            "schematic": "parameter-driven schematic",
            "candidate": "candidate parameter-driven schematic",
            "manual": "manual-assumption schematic",
            "unavailable": "data unavailable",
            "stale": "stale result",
        }
    label = labels.get(status, f"{status} schematic")
    relative = " · 相对尺度" if str(language).lower().startswith("zh") and _unit(scene) == "relative" else " · relative scale" if _unit(scene) == "relative" else ""
    return f"{source} · {label}{relative}"


def _reason_text(scene: Any, language: str) -> str:
    message = _message(scene)
    if message:
        return message
    status = _status(scene)
    if str(language).lower().startswith("zh"):
        return {
            "unavailable": "没有可用的层状几何数据",
            "stale": "场景已过期，不能代表当前参数",
        }.get(status, "没有可绘制的层状几何数据")
    return {
        "unavailable": "No lamellar geometry is available",
        "stale": "The scene is stale and does not represent current parameters",
    }.get(status, "No lamellar geometry is available")


def _scene_bounds(scene: Any, vertices: np.ndarray | None = None) -> np.ndarray:
    value = _as_float_array(_get(scene, "bounds"))
    if value.shape == (2, 3) and np.all(np.isfinite(value)) and np.all(value[1] > value[0]):
        padding = .04 * (value[1] - value[0])
        return value + np.vstack((-padding, padding))
    points = _vertices(scene) if vertices is None else vertices
    if points.size:
        flat = points.reshape(-1, 3)
        result = np.vstack((np.min(flat, axis=0), np.max(flat, axis=0)))
    else:
        result = np.asarray(((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)), dtype=float)
    spans = result[1] - result[0]
    spans[~np.isfinite(spans) | (spans <= 0)] = 1.0
    result[0] -= 0.04 * spans
    result[1] += 0.04 * spans
    return result


def _coerce_bounds(bounds: Any, scene: Any, vertices: np.ndarray) -> np.ndarray:
    if bounds is None:
        result = _scene_bounds(scene, vertices)
    else:
        result = _as_float_array(bounds)
        if result.shape != (2, 3) or not np.all(np.isfinite(result)) or not np.all(result[1] > result[0]):
            raise ValueError("bounds must be finite with shape (2, 3) and positive spans")
        # An explicit viewport is an export contract.  Keep it exact so a
        # sequence can be compared frame by frame without hidden padding.
    return np.asarray(result, dtype=float).copy()


def _colors(scene: Any, count: int) -> np.ndarray:
    raw = _get(scene, "colors")
    try:
        colors = np.asarray(raw, dtype=float)
    except (TypeError, ValueError):
        colors = np.empty((0, 4), dtype=float)
    if colors.shape != (count, 4) or not np.all(np.isfinite(colors)):
        branches = _get(scene, "branch_ids")
        try:
            branch_array = np.asarray(branches, dtype=int).reshape(-1)
        except (TypeError, ValueError):
            branch_array = np.arange(count, dtype=int)
        colors = np.vstack([_PALETTE[int(branch_array[i]) % len(_PALETTE)] for i in range(count)])
    elif np.nanmax(np.abs(colors)) > 1.001:
        colors = colors / 255.0
    return np.clip(colors, 0.0, 1.0)


def _ensure_canvas(fig: Figure) -> Figure:
    # Explicitly attach Agg.  This does not touch matplotlib's global backend.
    FigureCanvasAgg(fig)
    return fig


def _style_axes(ax: Any, *, title: str, font: FontProperties) -> None:
    ax.set_facecolor(_OFFWHITE)
    ax.set_title(title, color=_INK, fontproperties=font, fontsize=13, pad=12)
    ax.tick_params(colors=_MUTED, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(_GRID)


def _draw_banner(fig: Figure, scene: Any, language: str, *, y: float = 0.985) -> None:
    font = _font()
    fig.text(
        0.02,
        y,
        _title_suffix(scene, language),
        ha="left",
        va="top",
        color=_INK,
        fontproperties=font,
        fontsize=11,
        weight="semibold",
    )


def _draw_blank(
    ax: Any,
    scene: Any,
    language: str,
    *,
    bounds: np.ndarray | None = None,
    compact: bool = False,
) -> None:
    limits = bounds if bounds is not None else _scene_bounds(scene)
    ax.set_xlim(float(limits[0, 0]), float(limits[1, 0]))
    ax.set_ylim(float(limits[0, 1]), float(limits[1, 1]))
    ax.set_aspect("equal", adjustable="box")
    ax.text(
        0.5,
        0.5,
        _reason_text(scene, language),
        transform=ax.transAxes,
        ha="center",
        va="center",
        color=_MUTED,
        fontsize=10 if compact else 13,
        fontproperties=_font(),
        wrap=True,
    )
    ax.set_xlabel(f"x ({_unit_label(scene)})", color=_MUTED)
    ax.set_ylabel(f"y ({_unit_label(scene)})", color=_MUTED)


def _draw_axis_arrow(ax: Any, bounds: np.ndarray, language: str, *, compact: bool = False) -> None:
    span = bounds[1] - bounds[0]
    x0 = bounds[0, 0] + 0.08 * span[0]
    y0 = bounds[0, 1] + 0.08 * span[1]
    length = (0.10 if compact else 0.16) * span[1]
    if str(language).lower().startswith("en"):
        label = "+y · ref" if compact else "+y · reference direction"
    else:
        label = "+y · 参考" if compact else "+y · 参考方向"
    ax.annotate(
        label,
        xy=(x0, y0 + length),
        xytext=(x0, y0),
        arrowprops={"arrowstyle": "-|>", "color": _INK, "lw": 1.6},
        color=_INK,
        fontsize=8 if compact else 10,
        fontproperties=_font(),
        ha="center",
        va="bottom",
        zorder=20,
    )


def _draw_box_projection(ax: Any, vertices: np.ndarray, colors: np.ndarray) -> None:
    """Draw each real box once as a projected hull plus its visible edges.

    The projected polygon is made directly from the actual eight corners.  A
    monotonic angle sort makes a stable hull without depending on scipy.  Four
    local ``z`` edges are then overlaid with low opacity to expose slab
    thickness while retaining the same geometry.
    """

    for corners, color in zip(vertices, colors):
        xy = corners[:, :2]
        centre = np.mean(xy, axis=0)
        order = np.argsort(np.arctan2(xy[:, 1] - centre[1], xy[:, 0] - centre[0]))
        polygon = xy[order]
        face = np.asarray(color, dtype=float).copy()
        face[3] = min(float(face[3]), 0.42)
        ax.add_patch(Polygon(polygon, closed=True, facecolor=face, edgecolor=color, linewidth=1.1, zorder=4))
        for start, end in ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)):
            ax.plot(
                (xy[start, 0], xy[end, 0]),
                (xy[start, 1], xy[end, 1]),
                color=color,
                linewidth=0.75,
                alpha=0.75,
                zorder=5,
            )


def render_lamellar_2d(
    scene: Any,
    *,
    ax: Any | None = None,
    language: str = "zh",
    bounds: Any | None = None,
    decorate: bool = True,
) -> Figure:
    """Render a 2D top projection from the scene's actual vertices.

    The world ``x-y`` projection is chosen because the +y direction is the
    stable visual reference in the LamellarScene contract.  Thin z edges remain
    visible in the projection, so the view communicates both lamella footprint
    and slab cross-section without fabricating a second geometry model.
    """

    vertices = _vertices(scene)
    limits = _coerce_bounds(bounds, scene, vertices)
    supplied_ax = ax is not None
    if ax is None:
        fig = _ensure_canvas(Figure(figsize=(8.0, 6.4), facecolor=_OFFWHITE))
        ax = fig.add_subplot(111)
    else:
        fig = ax.figure
        ax.clear()
        fig.patch.set_facecolor(_OFFWHITE)
    font = _font()
    title = "Lamellar projection" if str(language).lower().startswith("en") else "层状结构投影"
    _style_axes(ax, title=title if decorate else "", font=font)
    ax.set_xlabel(f"x ({_unit_label(scene)})", color=_MUTED, fontproperties=font)
    ax.set_ylabel(f"y ({_unit_label(scene)})", color=_MUTED, fontproperties=font)
    if vertices.size and _status(scene) not in {"unavailable", "stale"}:
        _draw_box_projection(ax, vertices, _colors(scene, len(vertices)))
    else:
        _draw_blank(ax, scene, language, bounds=limits, compact=not decorate)
    ax.set_xlim(float(limits[0, 0]), float(limits[1, 0]))
    ax.set_ylim(float(limits[0, 1]), float(limits[1, 1]))
    ax.set_aspect("equal", adjustable="box")
    _draw_axis_arrow(ax, limits, language, compact=not decorate)
    if decorate:
        _draw_banner(fig, scene, language)
    if not supplied_ax and decorate:
        # Keep enough room for the readable source/status line.
        fig.subplots_adjust(top=0.88)
    return fig


def _box_faces(corners: np.ndarray) -> list[np.ndarray]:
    indexes = ((0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7))
    return [corners[list(face)] for face in indexes]


def _camera_source(camera: Any) -> Mapping[str, Any]:
    if isinstance(camera, Mapping):
        return camera
    if isinstance(camera, Sequence) and not isinstance(camera, (str, bytes)):
        values = list(camera)
        return {"elev": values[0], "azim": values[1]} if len(values) >= 2 else {}
    return {
        name: getattr(camera, name, None)
        for name in ("elev", "azim", "roll", "yaw", "pitch", "distance", "ortho_height", "zoom", "target")
    }


def _camera_values(camera: Any) -> dict[str, float]:
    if camera is None:
        return {}
    source = _camera_source(camera)
    result: dict[str, float] = {}
    aliases = {
        "distance": ("distance",),
        "ortho_height": ("ortho_height",),
        "zoom": ("zoom",),
    }
    for name, candidates in aliases.items():
        raw = next((source.get(candidate) for candidate in candidates if source.get(candidate) is not None), None)
        try:
            value = float(raw)
        except (AttributeError, TypeError, ValueError):
            continue
        if np.isfinite(value):
            result[name] = value
    return result


def _camera_orientation(camera: Any) -> tuple[float, float, float, str] | None:
    """Return Matplotlib view angles, preserving Qt Quick 3D's +Y up frame.

    Qt's camera is placed on local ``+Z`` and looks along ``-Z``.  Its QML
    rig applies ``pitch`` about X followed by ``yaw`` about Y, so the target to
    camera direction is ``(sin(yaw) cos(pitch), -sin(pitch),
    cos(yaw) cos(pitch))`` in the scene's x/y/z frame.  Matplotlib supports a
    y vertical axis directly; converting that direction to ``elev``/``azim``
    avoids treating Qt pitch as a Matplotlib z elevation.
    """

    if camera is None:
        return None
    source = _camera_source(camera)
    has_qt_angles = source.get("yaw") is not None or source.get("pitch") is not None
    if has_qt_angles:
        try:
            yaw = float(source.get("yaw", 0.0))
            pitch = float(source.get("pitch", 0.0))
        except (TypeError, ValueError):
            return None
        if not np.isfinite(yaw) or not np.isfinite(pitch):
            return None
        yaw_rad, pitch_rad = np.radians((yaw, pitch))
        eye = np.asarray(
            (
                np.sin(yaw_rad) * np.cos(pitch_rad),
                -np.sin(pitch_rad),
                np.cos(yaw_rad) * np.cos(pitch_rad),
            ),
            dtype=float,
        )
        norm = float(np.linalg.norm(eye))
        if not np.isfinite(norm) or norm <= 1.0e-12:
            return None
        eye /= norm
        elev = float(np.degrees(np.arcsin(np.clip(eye[1], -1.0, 1.0))))
        azim = float(np.degrees(np.arctan2(eye[0], eye[2])))
        return elev, azim, 0.0, "y"
    try:
        elev = float(source.get("elev", source.get("elevation")))
        azim = float(source.get("azim", source.get("azimuth")))
    except (TypeError, ValueError):
        return None
    roll = source.get("roll", 0.0)
    try:
        roll_value = float(roll)
    except (TypeError, ValueError):
        roll_value = 0.0
    if not all(np.isfinite(value) for value in (elev, azim, roll_value)):
        return None
    return elev, azim, roll_value, "z"


def _clean_3d_axes(ax: Any) -> None:
    """Keep the 3D export focused on boxes and a small coordinate helper."""

    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_visible(False)
        axis.set_ticks([])
        try:
            axis.pane.set_visible(False)
            axis.pane.set_facecolor((0.0, 0.0, 0.0, 0.0))
            axis.pane.set_edgecolor((0.0, 0.0, 0.0, 0.0))
            axis.line.set_visible(False)
            axis.line.set_color((0.0, 0.0, 0.0, 0.0))
            axis._axinfo["grid"]["linewidth"] = 0.0
            axis._axinfo["axisline"]["linewidth"] = 0.0
        except (AttributeError, KeyError):  # pragma: no cover - Matplotlib compatibility
            continue


def _draw_3d_blank(ax: Any, scene: Any, language: str, limits: np.ndarray) -> None:
    ax.text2D(0.5, 0.5, _reason_text(scene, language), transform=ax.transAxes, ha="center", va="center", color=_MUTED, fontsize=13, fontproperties=_font())
    ax.set_xlim(float(limits[0, 0]), float(limits[1, 0]))
    ax.set_ylim(float(limits[0, 1]), float(limits[1, 1]))
    ax.set_zlim(float(limits[0, 2]), float(limits[1, 2]))


def render_lamellar_3d(
    scene: Any,
    *,
    ax: Any | None = None,
    camera: Any | None = None,
    language: str = "zh",
    bounds: Any | None = None,
    decorate: bool = True,
) -> Figure:
    """Render the same scene boxes in a Matplotlib 3D export view."""

    vertices = _vertices(scene)
    limits = _coerce_bounds(bounds, scene, vertices)
    supplied_ax = ax is not None
    if ax is None:
        fig = _ensure_canvas(Figure(figsize=(8.0, 6.4), facecolor=_OFFWHITE))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure
        ax.clear()
        fig.patch.set_facecolor(_OFFWHITE)
    font = _font()
    title = "Lamellar boxes" if str(language).lower().startswith("en") else "层状结构三维示意"
    ax.set_title(title if decorate else "", color=_INK, fontproperties=font, fontsize=13, pad=12)
    ax.set_facecolor(_OFFWHITE)
    if vertices.size and _status(scene) not in {"unavailable", "stale"}:
        colors = _colors(scene, len(vertices))
        for corners, color in zip(vertices, colors):
            face_colors = np.tile(color, (6, 1))
            face_colors[:, 3] = min(float(color[3]), 0.35)
            collection = Poly3DCollection(_box_faces(corners), facecolors=face_colors, edgecolors=[color], linewidths=0.8, alpha=0.86)
            ax.add_collection3d(collection)
    else:
        _draw_3d_blank(ax, scene, language, limits)
    labels = ("x", "y", "z")
    for index, label in enumerate(labels):
        suffix = _unit_label(scene)
        getattr(ax, f"set_{label}label")(f"{label} ({suffix})", color=_MUTED, fontproperties=font)
    ax.set_xlim(float(limits[0, 0]), float(limits[1, 0]))
    ax.set_ylim(float(limits[0, 1]), float(limits[1, 1]))
    ax.set_zlim(float(limits[0, 2]), float(limits[1, 2]))
    spans = np.maximum(limits[1] - limits[0], 1e-9)
    view = _camera_values(camera)
    orientation = _camera_orientation(camera)
    try:
        # The interactive Qt view uses an orthographic camera.  Matching that
        # projection keeps the software fallback/export visually comparable.
        ax.set_proj_type("ortho")
    except (AttributeError, TypeError):  # pragma: no cover - old Matplotlib
        pass
    if orientation is not None:
        elev, azim, roll, vertical_axis = orientation
        ax.view_init(elev=elev, azim=azim, roll=roll, vertical_axis=vertical_axis)
    zoom = 1.0
    if "ortho_height" in view:
        # The QML view uses 35 world units as the stable first-frame fit.
        # Matplotlib's supported zoom parameter is the equivalent for an
        # orthographic export; camera distance only affects Qt clipping.
        zoom = float(np.clip(float(view["ortho_height"]) / 35.0, .01, 3.0))
    elif "zoom" in view:
        zoom = float(np.clip(view["zoom"], .01, 3.0))
    try:
        ax.set_box_aspect(spans, zoom=zoom)
    except TypeError:  # pragma: no cover - Matplotlib < 3.8
        ax.set_box_aspect(spans)
    _clean_3d_axes(ax)
    helper = "+y reference" if str(language).lower().startswith("en") else "+y 为参考方向"
    helper += f" · {_unit_label(scene)}"
    ax.text2D(0.025, 0.92, helper, transform=ax.transAxes, color=_MUTED, fontsize=8, fontproperties=font)
    # A small orientation triad replaces a crowded 3-D cage.  Project actual
    # world directions through this camera, so its arrows follow rotation.
    from mpl_toolkits.mplot3d import proj3d

    center = limits.mean(axis=0)
    step = max(float(np.max(spans)) * .01, 1e-9)
    matrix = ax.get_proj()
    start = np.asarray(proj3d.proj_transform(*center, matrix)[:2])
    vectors = [np.asarray(proj3d.proj_transform(*(center + step * axis), matrix)[:2]) - start for axis in np.eye(3)]
    scale = max(float(np.linalg.norm(vector)) for vector in vectors)
    for label, vector in zip(("x", "y", "z"), vectors):
        if np.linalg.norm(vector) <= scale * 1e-5:
            continue
        endpoint = np.asarray((.11, .14)) + vector / scale * .075
        ax.annotate("", xy=endpoint, xytext=(.11, .14), xycoords="axes fraction",
                    textcoords="axes fraction", color=_MUTED, fontsize=8,
                    arrowprops={"arrowstyle": "->", "color": _MUTED, "lw": .8})
        text_point = endpoint + vector / scale * .025
        ax.text2D(*text_point, "+" + label, transform=ax.transAxes, color=_MUTED,
                  fontsize=8, ha="center", va="center")
    if decorate:
        _draw_banner(fig, scene, language)
    if not supplied_ax and decorate:
        fig.subplots_adjust(top=0.88)
    return fig


def _image_extent(qx: Any, qy: Any, shape: tuple[int, int]) -> tuple[float, float, float, float] | None:
    if qx is None or qy is None:
        return None
    try:
        x = np.asarray(qx, dtype=float)
        y = np.asarray(qy, dtype=float)
    except (TypeError, ValueError):
        return None
    try:
        x_values = x if x.ndim == 1 else x[0, :]
        y_values = y if y.ndim == 1 else y[:, 0]
        if x_values.size == 0 or y_values.size == 0:
            return None
        return (float(np.nanmin(x_values)), float(np.nanmax(x_values)), float(np.nanmin(y_values)), float(np.nanmax(y_values)))
    except (TypeError, ValueError):
        return None


def _display_q_window(scene: Any) -> tuple[float, float] | None:
    """Read a display-only radial q window from scene metadata."""

    metadata = _get(scene, "metadata", {})
    raw = metadata.get("display_q_window") if isinstance(metadata, Mapping) else None
    if isinstance(raw, Mapping):
        nested = raw.get("q_window", raw)
        if isinstance(nested, Mapping):
            low = nested.get("min", nested.get("q_min", nested.get("low", nested.get("start"))))
            high = nested.get("max", nested.get("q_max", nested.get("high", nested.get("stop"))))
        else:
            low = high = None
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        values = list(raw)
        low, high = (values + [None, None])[:2]
    else:
        low = high = None
    try:
        low_value, high_value = float(low), float(high)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(low_value) or not np.isfinite(high_value) or low_value < 0.0 or high_value <= low_value:
        return None
    return low_value, high_value


def _q_radius(qx: Any, qy: Any, shape: tuple[int, int]) -> np.ndarray | None:
    """Broadcast native q coordinates to an image-shaped radial map."""

    try:
        x, y = np.asarray(qx, dtype=float), np.asarray(qy, dtype=float)
    except (TypeError, ValueError):
        return None
    rows, columns = shape
    try:
        if x.ndim == y.ndim == 1:
            if x.size != columns or y.size != rows:
                return None
            x, y = np.meshgrid(x, y)
        else:
            x = np.broadcast_to(x, shape)
            y = np.broadcast_to(y, shape)
        if x.shape != shape or y.shape != shape:
            return None
        radius = np.hypot(x, y)
    except (TypeError, ValueError):
        return None
    return radius if np.any(np.isfinite(radius)) else None


def _draw_observed(ax: Any, observed: Any, qx: Any, qy: Any, scene: Any, language: str) -> None:
    try:
        image = np.asarray(observed, dtype=float)
    except (TypeError, ValueError):
        image = np.empty((0, 0), dtype=float)
    if image.ndim != 2 or image.size == 0 or not np.any(np.isfinite(image)):
        ax.set_facecolor(_OFFWHITE)
        ax.text(0.5, 0.5, "No observed frame" if str(language).lower().startswith("en") else "没有观测帧", transform=ax.transAxes, ha="center", va="center", color=_MUTED, fontproperties=_font())
        return
    display_window = _display_q_window(scene)
    display_mask = np.zeros(image.shape, dtype=bool)
    radius = _q_radius(qx, qy, image.shape) if display_window is not None and qx is not None and qy is not None else None
    if display_window is not None and radius is not None:
        low_q, high_q = display_window
        display_mask = ~np.isfinite(radius) | (radius < low_q) | (radius > high_q)
    display_values = image[~display_mask & np.isfinite(image)]
    finite = display_values if display_values.size else image[np.isfinite(image)]
    low, high = np.nanpercentile(finite, (1.0, 99.5)) if finite.size > 2 else (float(np.nanmin(finite)), float(np.nanmax(finite)))
    if not np.isfinite(low):
        low = 0.0
    if not np.isfinite(high) or high <= low:
        high = low + 1.0
    extent = _image_extent(qx, qy, image.shape)
    kwargs = {"origin": "lower", "cmap": "cividis", "vmin": low, "vmax": high, "interpolation": "nearest", "aspect": "equal"}
    if extent is not None:
        kwargs["extent"] = extent
    shown = np.ma.array(image, mask=display_mask) if np.any(display_mask) else image
    im = ax.imshow(shown, **kwargs)
    if display_window is not None and radius is not None:
        low_q, high_q = display_window
        ax.set_xlim(-high_q, high_q)
        ax.set_ylim(-high_q, high_q)
    q_unit = _q_unit_label(scene)
    ax.set_xlabel(f"qx ({q_unit})", color=_MUTED, fontproperties=_font())
    ax.set_ylabel(f"qy ({q_unit})", color=_MUTED, fontproperties=_font())
    title = "Observed" if str(language).lower().startswith("en") else "观测图"
    if display_window is not None:
        low_q, high_q = display_window
        title += f" · q ∈ [{low_q:g}, {high_q:g}]"
    ax.set_title(title, color=_INK, fontproperties=_font(), fontsize=12)
    ax.figure.colorbar(im, ax=ax, shrink=0.78, pad=0.04)


def render_lamellar_combined(
    scene: Any,
    *,
    observed: Any | None = None,
    qx: Any | None = None,
    qy: Any | None = None,
    image_3d: Any | None = None,
    camera: Any | None = None,
    language: str = "zh",
    bounds: Any | None = None,
) -> Figure:
    """Render observed data, geometry projection and optional 3D capture."""

    fig = _ensure_canvas(Figure(figsize=(13.0, 5.4), facecolor=_OFFWHITE))
    grid = fig.add_gridspec(1, 3, width_ratios=(1.0, 1.25, 1.0), wspace=0.4)
    observed_ax = fig.add_subplot(grid[0, 0])
    geometry_ax = fig.add_subplot(grid[0, 1])
    image_ax = fig.add_subplot(grid[0, 2])
    _draw_observed(observed_ax, observed, qx, qy, scene, language)
    # Draw directly into the combined figure to avoid creating a duplicate
    # temporary geometry figure and to keep the axes layout predictable.
    geometry_ax.set_facecolor(_OFFWHITE)
    geometry_ax.set_title("Lamellar projection" if str(language).lower().startswith("en") else "层状结构投影", color=_INK, fontproperties=_font(), fontsize=12, pad=12)
    vertices = _vertices(scene)
    limits = _coerce_bounds(bounds, scene, vertices)
    if vertices.size and _status(scene) not in {"unavailable", "stale"}:
        _draw_box_projection(geometry_ax, vertices, _colors(scene, len(vertices)))
    else:
        _draw_blank(geometry_ax, scene, language, bounds=limits)
    geometry_ax.set_xlim(float(limits[0, 0]), float(limits[1, 0]))
    geometry_ax.set_ylim(float(limits[0, 1]), float(limits[1, 1]))
    geometry_ax.set_aspect("equal", adjustable="box")
    geometry_ax.set_xlabel(f"x ({_unit_label(scene)})", color=_MUTED, fontproperties=_font())
    geometry_ax.set_ylabel(f"y ({_unit_label(scene)})", color=_MUTED, fontproperties=_font())
    _draw_axis_arrow(geometry_ax, limits, language, compact=True)
    image_ax.set_facecolor(_OFFWHITE)
    image_array = _image_to_array(image_3d)
    if image_array is not None:
        image_ax.imshow(image_array)
        image_ax.set_axis_off()
        image_ax.set_title("3D capture" if str(language).lower().startswith("en") else "三维视图捕获", color=_INK, fontproperties=_font(), fontsize=12, pad=12)
    else:
        image_ax.remove()
        image_ax = fig.add_subplot(grid[0, 2], projection="3d")
        render_lamellar_3d(scene, ax=image_ax, camera=camera, language=language, bounds=bounds, decorate=False)
    _draw_banner(fig, scene, language, y=0.99)
    fig.subplots_adjust(top=0.86)
    return fig


def _image_to_array(value: Any) -> np.ndarray | None:
    """Convert common QImage/array captures without importing Qt."""

    if value is None:
        return None
    if isinstance(value, np.ndarray):
        array = value
    else:
        array = None
        # Some capture wrappers expose to_numpy/toarray directly.
        for name in ("to_numpy", "toarray", "asarray"):
            method = getattr(value, name, None)
            if callable(method):
                try:
                    array = np.asarray(method())
                    break
                except Exception:  # pragma: no cover - optional Qt wrappers
                    continue
        if array is None and hasattr(value, "toImage"):
            try:
                array = _image_to_array(value.toImage())
            except Exception:  # pragma: no cover - optional Qt wrappers
                array = None
        if array is None and hasattr(value, "save"):
            # Saving directly is handled by the exporter; the renderer cannot
            # display an arbitrary QImage without depending on Qt's bindings.
            return None
    if array is None or array.ndim not in {2, 3} or array.size == 0:
        return None
    if array.dtype.kind in "fc":
        finite = np.isfinite(array)
        if not np.any(finite):
            return None
        if array.ndim == 3 and array.shape[-1] not in {3, 4}:
            return None
    return np.asarray(array)


__all__ = [
    "render_lamellar_2d",
    "render_lamellar_3d",
    "render_lamellar_combined",
    "_image_to_array",
    "_FORMAT_VERSION",
]
