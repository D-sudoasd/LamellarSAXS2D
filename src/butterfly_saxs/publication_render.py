"""Fixed artboards: native 3D surfaces with editable, calibrated annotations."""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import LogNorm, Normalize
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Rectangle
from matplotlib.text import Annotation
from matplotlib.ticker import MaxNLocator
from scipy.spatial import ConvexHull

from .publication_geometry import camera_basis, project_points
from .publication_models import PublicationFigureSpec, PublicationRenderResult, PublicationStyle

_INK = "#263a44"
_MUTED = "#64757c"


def _get(value, name, default=None):
    return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)


def _font(size=6.5, *, weight="normal", language="en"):
    families = ["Microsoft YaHei", "DejaVu Sans"] if language.startswith("zh") else ["Arial", "DejaVu Sans"]
    return FontProperties(family=families, size=size, weight=weight)


def _scene_status(scene):
    return str(_get(scene, "status", _get(scene, "metadata", {}).get("status", "schematic"))).lower()


def _scene_source(scene):
    identity = _get(scene, "source_identity", {})
    return str(identity.get("id", identity.get("frame", "Lamellar scene"))) if isinstance(identity, Mapping) else str(identity)


def _unit(scene):
    return "nm" if _get(scene, "length_unit") == "nm" else "rel. u."


def _as_render_result(value):
    if value is None:
        raise ValueError("rendered_scene is required for publication figure authoring")
    if isinstance(value, PublicationRenderResult):
        return value
    return PublicationRenderResult(
        rgba=np.asarray(_get(value, "rgba")), camera=dict(_get(value, "camera", {})),
        bounds=np.asarray(_get(value, "bounds")), projected=dict(_get(value, "projected", {})),
        provenance=dict(_get(value, "provenance", {})))


def _period_symbol(scene):
    source = _get(scene, "settings", {}).get("period_source", "radial")
    subscript = "set" if source == "manual" else "ell" if source == "ellipse" else "app"
    return r"$L_{\mathrm{" + subscript + r"}}$"


def _text(fig, spec, name, position, text, *, size=6.5, color=_INK, weight="normal", ha="left"):
    position = spec.annotation_positions.get(name, position)
    artist = fig.text(*position, text, ha=ha, va="top", color=color,
                      fontproperties=_font(size, weight=weight, language=spec.language))
    artist.set_gid(f"annotation-{name}")
    fig.publication_annotation_artists[name] = artist
    return artist


def _leader(fig, spec, name, anchor, position, text, *, color=_INK):
    position = spec.annotation_positions.get(name, position)
    artist = Annotation(text, xy=anchor, xytext=position, xycoords=fig.transFigure,
                        textcoords=fig.transFigure, ha="center", va="bottom", color=color,
                        fontproperties=_font(language=spec.language),
                        arrowprops={"arrowstyle": "-", "lw": .45, "color": color,
                                    "shrinkA": 3, "shrinkB": 2, "connectionstyle": "arc3,rad=0"})
    artist.set_gid(f"annotation-{name}")
    fig.add_artist(artist)
    fig.publication_annotation_artists[name] = artist
    return artist


def _project_figure(points, result, spec):
    height, width = result.rgba.shape[:2]
    screen = project_points(np.asarray(points), result.camera, result.bounds, width, height)
    x, y, w, h = spec.main_panel
    return np.column_stack((x + screen[:, 0] * w, y + (1 - screen[:, 1]) * h))


def _nice_length(value):
    power = 10. ** np.floor(np.log10(max(value, 1e-15)))
    return max(factor * power for factor in (.1, .2, .5, 1., 2., 5.) if factor * power <= value)


def _hero_annotations(fig, scene, spec, style, result):
    populations = {int(p["branch_id"]): p for p in scene.metadata.get("populations", ())}
    overall_top = float(np.max(_project_figure(scene.vertices.reshape(-1, 3), result, spec)[:, 1]))
    for branch in np.unique(scene.branch_ids):
        indices = np.flatnonzero(scene.branch_ids == branch)
        projected = _project_figure(scene.vertices[indices].reshape(-1, 3), result, spec)
        stacks = [indices[scene.stack_ids[indices] == value] for value in np.unique(scene.stack_ids[indices])]
        stack = max(stacks, key=lambda items: float(np.max(_project_figure(scene.vertices[items].reshape(-1, 3), result, spec)[:, 1])))
        representative = _project_figure(scene.vertices[stack].reshape(-1, 3), result, spec)
        top = representative[np.argmax(representative[:, 1])]
        x = float(np.mean(representative[:, 0]))
        y = min(.91, overall_top + (.05 if spec.width_mm >= 120 else .065))
        period = populations.get(int(branch), {}).get("period")
        label = chr(65 + int(branch)) if 0 <= branch < 26 else str(branch)
        text = f"Family {label}" if spec.language == "en" else f"取向族 {label}"
        if period is not None:
            text += f"\n{_period_symbol(scene)} = {period:.3g} {_unit(scene)}"
        color = tuple(v * .8 for v in style.colors[int(branch) % 2][:3])
        _leader(fig, spec, f"family_{branch}", top, (x, y), text, color=color)
        # Normal at the end of a real stack, projected through the same camera.
        normal = scene.orientations[stack[0], :, 2]
        points = _project_figure(np.array([np.zeros(3), normal]), result, spec)
        physical_direction = (points[1] - points[0]) * [spec.width_mm, spec.height_mm]
        magnitude = np.linalg.norm(physical_direction)
        if magnitude > 1e-8:
            direction = physical_direction / magnitude
            # Place the arrow outside the silhouette's supporting plane, so
            # broad projected surfaces cannot cover it at oblique views.
            support = projected[np.argmax((projected * [spec.width_mm, spec.height_mm]) @ direction)]
            start = support + direction * 1.5 / [spec.width_mm, spec.height_mm]
            points = np.array([start, start + direction * 5 / [spec.width_mm, spec.height_mm]])
            arrow = Annotation("", xy=points[1], xytext=points[0], xycoords=fig.transFigure,
                               textcoords=fig.transFigure, arrowprops={"arrowstyle": "-|>",
                               "mutation_scale": 6, "lw": .65, "color": color})
            fig.add_artist(arrow)
            _text(fig, spec, f"normal_{branch}", points[1] + [.008, .008],
                  r"$\mathbf{n}_{" + label + "}$", color=color)

    x, y, w, h = spec.main_panel
    # Screen horizontal corresponds to the camera's unit right vector.
    right, _, _ = camera_basis(result.camera)
    projection = _project_figure(np.array([np.zeros(3), right]), result, spec)
    fraction_per_unit = np.linalg.norm(projection[1] - projection[0])
    scale = _nice_length(w * .17 / fraction_per_unit)
    baseline = .20 if spec.width_mm < 120 and spec.template == "structure" else y + (.06 if spec.width_mm < 120 else .035)
    start = np.array([x + w * .08, baseline])
    stop = start + [scale * fraction_per_unit, 0]
    line = Line2D([start[0], stop[0]], [start[1], stop[1]], transform=fig.transFigure,
                  lw=1.1, color=_INK, solid_capstyle="butt")
    line.set_gid("physical-scale-bar")
    fig.add_artist(line)
    _text(fig, spec, "scale", ((start[0] + stop[0]) / 2, start[1] - .018),
          f"{scale:g} {_unit(scene)}", ha="center")
    fig.publication_scale = {"length": scale, "unit": _unit(scene),
                             "figure_endpoints": [start.tolist(), stop.tolist()]}
    # Preserve the recorded reference, including non-90-degree directions.
    angle = np.deg2rad(float(scene.metadata.get("draw_axis_deg", 90)))
    reference = np.array([np.cos(angle), np.sin(angle), 0.])
    points = _project_figure(np.array([np.zeros(3), reference]), result, spec)
    vector = points[1] - points[0]
    physical = vector * [spec.width_mm, spec.height_mm]
    if np.linalg.norm(physical) > 1e-9:
        vector *= 6 / np.linalg.norm(physical)
        start = np.array([.37, .18]) if spec.width_mm < 120 and spec.template == "structure" else np.array([x + .9 * w, baseline])
        stop = start + vector
        arrow = Annotation("", xy=stop, xytext=start, xycoords=fig.transFigure,
                           textcoords=fig.transFigure, arrowprops={"arrowstyle": "-|>",
                           "lw": .65, "mutation_scale": 7, "color": _MUTED})
        fig.add_artist(arrow)
        _text(fig, spec, "reference", (start[0], start[1] - .018),
              "Reference" if spec.language == "en" else "参考方向", ha="center", color=_MUTED)


def _axes_title(fig, spec, name, bounds, title, letter):
    x, y, w, h = bounds
    _text(fig, spec, letter, (x - .02, min(.98, y + h + .035)), letter, size=8, weight="bold")
    _text(fig, spec, f"title_{name}", (x + .5 * w, min(.98, y + h + .033)), title, ha="center")


def _draw_saxs(fig, ax, scene, observed, qx, qy, spec):
    if observed is None or np.asarray(observed).ndim != 2:
        ax.text(.5, .5, "SAXS unavailable", transform=ax.transAxes, ha="center",
                fontproperties=_font(language=spec.language), color=_MUTED)
        ax.set_axis_off()
        return
    values = np.asarray(observed, dtype=float)
    valid = np.isfinite(values)
    if not np.any(valid):
        ax.set_axis_off()
        return
    positive = values[valid & (values > 0)]
    if positive.size:
        low, high = np.percentile(positive, [1., 99.5])
        low = max(low, np.min(positive))
        norm = LogNorm(low, max(high, low * 1.01))
        valid &= values > 0
    else:
        norm = Normalize(float(np.min(values[valid])), float(np.max(values[valid])) + 1)
    x = np.asarray(qx) if qx is not None else None
    y = np.asarray(qy) if qy is not None else None
    if (x is None) != (y is None):
        raise ValueError("Both qx and qy are required for a calibrated SAXS panel")
    calibrated = x is not None and y is not None
    if calibrated:
        if x.ndim == y.ndim == 1:
            x, y = np.meshgrid(x, y)
        if x.shape != values.shape or y.shape != values.shape:
            raise ValueError("SAXS qx/qy shapes must match the observed image")
    if calibrated:
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            raise ValueError("SAXS q coordinates must be finite for a calibrated publication panel")
        ax.pcolormesh(x, y, np.ma.array(values, mask=~valid), cmap="cividis", norm=norm,
                      shading="auto", rasterized=True)
        q_unit = scene.metadata.get("q_unit", "unknown").replace("^-1", "⁻¹")
        ax.set_xlabel(f"qx ({q_unit})", fontproperties=_font(language=spec.language), labelpad=1)
        ax.set_ylabel(f"qy ({q_unit})", fontproperties=_font(language=spec.language), labelpad=1)
        ax.xaxis.set_major_locator(MaxNLocator(3))
        ax.yaxis.set_major_locator(MaxNLocator(3))
    else:
        ax.imshow(np.ma.array(values, mask=~valid), norm=norm, cmap="cividis", origin="lower", interpolation="nearest")
        ax.set_xlabel("Detector pixel", fontproperties=_font(language=spec.language))
        ax.set_ylabel("Detector pixel", fontproperties=_font(language=spec.language))
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(labelsize=5.5, width=.45, length=2, pad=2, colors=_MUTED)
    for spine in ax.spines.values():
        spine.set_linewidth(.45)
        spine.set_color(_MUTED)
    ax.text(.98, .97, "log I" if positive.size else "I", ha="right", va="top",
            transform=ax.transAxes, fontproperties=_font(5.5), color="white",
            bbox={"facecolor": "#263a44", "alpha": .6, "edgecolor": "none", "pad": 1})


def _draw_projection(ax, scene, style):
    for vertices, branch in zip(scene.vertices, scene.branch_ids):
        points = np.unique(vertices[:, :2], axis=0)
        if len(points) < 3 or np.linalg.matrix_rank(points - points[0]) < 2:
            continue
        hull = points[ConvexHull(points).vertices]
        ax.add_patch(Polygon(hull, facecolor=style.colors[int(branch) % 2], edgecolor="white", linewidth=.2))
    ax.update_datalim(scene.vertices.reshape(-1, 3)[:, :2])
    ax.autoscale_view()
    ax.margins(.12)
    ax.set_aspect("equal")
    ax.set_axis_off()


def _local_geometry(ax, scene, spec, style):
    stack = scene.stack_ids[0]
    indices = np.flatnonzero(scene.stack_ids == stack)
    basis = scene.orientations[indices[0]]
    indices = indices[np.argsort(scene.centers[indices] @ basis[:, 2])]
    # A local detail crops three existing plates. All plates remain in the
    # hero, the whole-scene projection, and the exported source arrays.
    shown = indices[:3]
    origin = scene.centers[shown[0]]
    u = (scene.centers[shown] - origin) @ basis[:, 0]
    v = (scene.centers[shown] - origin) @ basis[:, 2]
    width, _, thickness = scene.sizes[shown[0]]
    period = float(np.median(np.diff(v))) if len(v) > 1 else float(scene.reference_period)
    branch = int(scene.branch_ids[shown[0]])
    color = style.colors[branch % 2]
    for index, cx, cy in zip(shown, u, v):
        w, _, t = scene.sizes[index]
        ax.add_patch(Rectangle((cx - w / 2, cy - t / 2), w, t, facecolor=color, edgecolor="none"))
    ax.set_aspect("equal")
    ax.set_xlim(-width * .85, width * .9)
    ax.set_ylim(-period * .65, max(v) + period * .9)
    ax.set_axis_off()
    if spec.annotations:
        arrow = {"arrowstyle": "|-|", "lw": .6, "color": _INK, "mutation_scale": 3}
        x = -width * .65
        if len(v) > 1:
            ax.annotate("", xy=(x, v[1]), xytext=(x, v[0]), arrowprops=arrow)
            ax.text(x - width * .04, (v[1] + v[0]) / 2, _period_symbol(scene), ha="right", va="center",
                    fontproperties=_font(6, language=spec.language), color=_INK)
        x = width * .65
        ax.annotate("", xy=(x, thickness / 2), xytext=(x, -thickness / 2), arrowprops=arrow)
        ax.text(x + width * .08, thickness / 2 + period * .16, "$t$", va="bottom",
                fontproperties=_font(language=spec.language), color=_INK)
        ax.text(.5, -.02, f"t = {thickness:.3g} {_unit(scene)} (set)", ha="center", va="top",
                transform=ax.transAxes, fontproperties=_font(6, language=spec.language), color=_MUTED)
    return {"stack_id": int(stack), "branch_id": branch, "shown_plate_indices": shown.tolist(),
            "l_app": period, "t_schem": float(thickness), "unit": _unit(scene)}


def render_publication_figure(scene, *, spec=None, style=None, rendered_scene=None, observed=None, qx=None, qy=None):
    spec = PublicationFigureSpec.from_mapping(spec)
    style = PublicationStyle.from_mapping(style)
    result = _as_render_result(rendered_scene)
    if _scene_status(scene) in {"unavailable", "stale"} or not len(scene.vertices):
        raise ValueError("Cannot author unavailable/stale publication scene")
    fig = Figure(figsize=(spec.width_mm / 25.4, spec.height_mm / 25.4), dpi=spec.dpi,
                 facecolor="none" if spec.background == "transparent" else "white")
    FigureCanvasAgg(fig)
    fig.publication_annotation_artists = {}
    hero = fig.add_axes(spec.main_panel)
    hero.set_gid("native-3d-surface-layer")
    hero.imshow(result.rgba, interpolation="none", aspect="auto")
    hero.set_axis_off()
    hero.patch.set_visible(False)
    if spec.annotations:
        _hero_annotations(fig, scene, spec, style, result)
        _text(fig, spec, "a", (.03 if spec.width_mm < 120 else spec.main_panel[0], .97), "a", size=8, weight="bold")
    local_metrics = {}
    for index, (name, bounds) in enumerate(spec.auxiliary_panels.items()):
        axis = fig.add_axes(bounds, facecolor="none")
        axis.set_gid(f"vector-{name}-layer")
        if name == "saxs":
            _draw_saxs(fig, axis, scene, observed, qx, qy, spec)
            title = "SAXS" if spec.language == "en" else "原始 SAXS"
        elif name == "projection":
            _draw_projection(axis, scene, style)
            title = "xy projection" if spec.language == "en" else "xy 投影"
        else:
            local_metrics = _local_geometry(axis, scene, spec, style)
            label = chr(65 + local_metrics["branch_id"])
            title = f"Layer detail · {label}" if spec.language == "en" else f"层间特写 · {label}"
        if spec.annotations:
            _axes_title(fig, spec, name, bounds, title, chr(98 + index))
    if spec.title:
        _text(fig, spec, "figure_title", (.5, .992), spec.title, size=8, weight="bold", ha="center")
    labels = {"candidate": "Candidate parameter-driven schematic", "manual": "Manual-assumption schematic",
              "schematic": "Parameter-driven schematic"}
    if spec.language.startswith("zh"):
        labels = {"candidate": "候选参数驱动示意", "manual": "手动假设示意", "schematic": "参数驱动示意"}
    footer = labels.get(_scene_status(scene), labels["schematic"])
    if _unit(scene) != "nm":
        footer += " · relative scale" if spec.language == "en" else " · 相对尺度"
    _text(fig, spec, "provenance", (.03, .038), footer, size=5.5, color=_MUTED)
    fig.publication_style = style
    fig.publication_spec = spec
    fig.publication_render_result = result
    fig.publication_local_metrics = local_metrics
    return fig


__all__ = ["render_publication_figure", "_as_render_result", "_scene_status", "_scene_source"]
