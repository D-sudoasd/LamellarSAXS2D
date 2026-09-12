"""Shared input preparation and map drawing for butterfly figure exports."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
import threading
from typing import Any

from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import numpy as np

from .serialization import json_safe
from .visualization import _diagnostic_q_axis_labels, _display_transform

_SUPPORTED_WIDTHS_MM = (89.0, 183.0)
_EXPORT_LOCK = threading.RLock()
_FIGURE_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 6.5,
    "axes.labelsize": 6.5,
    "axes.titlesize": 6.5,
    "xtick.labelsize": 5.5,
    "ytick.labelsize": 5.5,
    "legend.fontsize": 5.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "axes.unicode_minus": False,
}

def _numeric_array(value: Any, name: str) -> tuple[np.ndarray, np.ndarray]:
    """Return read-only input data and its explicit masked-array mask."""

    try:
        array = np.asanyarray(value)
    except Exception as exc:
        raise TypeError(f"{name} must be a numeric array") from exc
    data = np.asarray(np.ma.getdata(array))
    if data.dtype.kind not in "iuf":
        raise TypeError(f"{name} must contain real numeric values")
    mask = np.asarray(np.ma.getmaskarray(array), dtype=bool)
    return data, mask


def _result_q_window(
    result: Mapping[str, Any],
) -> tuple[tuple[float, float] | None, str | None]:
    candidates: list[tuple[Any, str]] = []
    direct = result.get("q_window")
    if direct is not None:
        candidates.append((direct, "result.q_window"))
    diagnostics = result.get("diagnostics")
    if isinstance(diagnostics, Mapping) and diagnostics.get("q_window") is not None:
        candidates.append((diagnostics["q_window"], "result.diagnostics.q_window"))
    settings = result.get("settings")
    if isinstance(settings, Mapping) and settings.get("q_window") is not None:
        candidates.append((settings["q_window"], "result.settings.q_window"))
    analysis_settings = result.get("analysis_settings")
    if (
        isinstance(analysis_settings, Mapping)
        and analysis_settings.get("q_window") is not None
    ):
        candidates.append(
            (analysis_settings["q_window"], "result.analysis_settings.q_window")
        )
    for value, source in candidates:
        if isinstance(value, Mapping):
            low = value.get("q_min", value.get("min", value.get("low")))
            high = value.get("q_max", value.get("max", value.get("high")))
        elif (
            isinstance(value, Sequence)
            and not isinstance(value, (str, bytes))
            and len(value) == 2
        ):
            low, high = value
        else:
            continue
        try:
            pair = float(low), float(high)
        except (TypeError, ValueError, OverflowError):
            continue
        if all(math.isfinite(item) for item in pair) and pair[0] < pair[1]:
            return pair, source
    return None, None


def _prepare_inputs(
    *,
    observed: Any,
    qx: Any,
    qy: Any,
    valid_mask: Any,
    result: Mapping[str, Any],
    q_unit: str,
    context: Any,
    display_scale: str,
    width_mm: float,
    dpi: int,
) -> dict[str, Any]:
    if not isinstance(result, Mapping):
        raise TypeError("result must be a Mapping containing the butterfly page result")
    width = float(width_mm)
    if width not in _SUPPORTED_WIDTHS_MM:
        raise ValueError("width_mm must be 89 or 183")
    if isinstance(dpi, bool) or int(dpi) != dpi or int(dpi) <= 0:
        raise ValueError("dpi must be a positive integer")
    dpi = int(dpi)
    scale = str(display_scale).strip().lower()
    if scale not in {"linear", "log1p", "asinh"}:
        raise ValueError("display_scale must be 'linear', 'log1p', or 'asinh'")

    observed_raw, observed_mask = _numeric_array(observed, "observed")
    qx_raw, qx_mask = _numeric_array(qx, "qx")
    qy_raw, qy_mask = _numeric_array(qy, "qy")
    if observed_raw.ndim != 2 or 0 in observed_raw.shape:
        raise ValueError("observed must be a non-empty 2D array")
    shape = observed_raw.shape
    if qx_raw.shape != shape or qy_raw.shape != shape:
        raise ValueError("qx and qy must have the same shape as observed")
    if observed_mask.shape != shape or qx_mask.shape != shape or qy_mask.shape != shape:
        raise ValueError("masked-array masks must match observed shape")

    if valid_mask is None:
        source_valid_mask = np.ones(shape, dtype=bool)
        valid_mask_was_supplied = False
        source_mask_mask = np.zeros(shape, dtype=bool)
    else:
        raw_mask = np.asanyarray(valid_mask)
        source_mask_mask = np.asarray(np.ma.getmaskarray(raw_mask), dtype=bool)
        raw_mask_data = np.asarray(np.ma.getdata(raw_mask))
        if raw_mask_data.shape != shape:
            raise ValueError("valid_mask must have the same shape as observed")
        if raw_mask_data.dtype.kind != "b":
            raise TypeError("valid_mask must contain boolean values")
        source_valid_mask = np.asarray(raw_mask_data, dtype=bool)
        valid_mask_was_supplied = True
    supplied_valid_mask_role = (
        "caller-supplied selection mask (True=valid); may represent detector, q-window, ROI, or combined analysis-domain validity; not assumed to be detector-only"
        if valid_mask_was_supplied
        else "no caller mask supplied; all-true placeholder; effective validity comes from finite observations/q coordinates and array masks"
    )

    try:
        observed_float = np.asarray(observed_raw, dtype=np.float64)
        qx_float = np.asarray(qx_raw, dtype=np.float64)
        qy_float = np.asarray(qy_raw, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError("observed, qx, and qy must be convertible to float64") from exc
    finite_observed = np.isfinite(observed_float)
    finite_q = np.isfinite(qx_float) & np.isfinite(qy_float)
    effective_valid = (
        source_valid_mask
        & ~source_mask_mask
        & ~observed_mask
        & ~qx_mask
        & ~qy_mask
        & finite_observed
        & finite_q
    )
    full_valid_count = int(np.count_nonzero(effective_valid))
    if full_valid_count == 0:
        raise ValueError(
            "no valid finite observations remain after applying valid_mask"
        )
    q_window, q_window_source = _result_q_window(result)
    radius = np.hypot(qx_float, qy_float)
    display_valid = effective_valid.copy()
    q_window_applied = False
    q_window_fallback_reason = None
    if q_window is not None:
        window_valid = (
            effective_valid & (radius >= q_window[0]) & (radius <= q_window[1])
        )
        if np.any(window_valid):
            display_valid = window_valid
            q_window_applied = True
        else:
            q_window_fallback_reason = "no valid source pixels fall inside the result q_window; full valid frame shown"
    selected = observed_float[display_valid]
    selected_display = _display_transform(selected, scale)
    transform_scale = None
    if scale == "linear":
        transform_label = "I"
    elif scale == "log1p":
        transform_label = "sign(I) * log1p(abs(I))"
    else:
        transform_label = "asinh(I)"

    if not np.all(np.isfinite(selected_display)):
        raise ValueError(
            "display transform produced non-finite values for valid observations"
        )
    low, high = (float(x) for x in np.percentile(selected_display, [0.5, 99.5]))
    if not math.isfinite(low) or not math.isfinite(high):
        raise ValueError("could not determine finite color limits")
    if high <= low:
        if scale == "log1p" and low <= 0.0:
            low, high = 0.0, 1.0
        else:
            margin = max(abs(high) * 0.01, 0.5)
            low, high = low - margin, high + margin

    display_count = int(np.count_nonzero(display_valid))
    radii = radius[display_valid]
    radial_max = float(np.max(radii)) if radii.size else 0.0
    bin_count = min(64, max(16, int(math.ceil(math.sqrt(display_count)))))
    radial_lower = (
        max(0.0, q_window[0]) if q_window_applied and q_window is not None else 0.0
    )
    radial_upper = (
        q_window[1] if q_window_applied and q_window is not None else radial_max
    )
    if radial_upper <= radial_lower:
        radial_upper = radial_lower + max(abs(radial_lower) * 0.05, 1.0)
    radial_edges = np.linspace(
        radial_lower, radial_upper, bin_count + 1, dtype=np.float64
    )
    radial_sum, _ = np.histogram(radii, bins=radial_edges, weights=selected)
    radial_counts, _ = np.histogram(radii, bins=radial_edges)
    radial_mean = np.full(bin_count, np.nan, dtype=np.float64)
    occupied = radial_counts > 0
    radial_mean[occupied] = radial_sum[occupied] / radial_counts[occupied]
    radial_centers = 0.5 * (radial_edges[:-1] + radial_edges[1:])
    radial_abs_nonzero = np.abs(selected[np.nonzero(selected)])
    radial_linthresh = (
        max(float(np.percentile(radial_abs_nonzero, 10.0)), np.finfo(np.float64).tiny)
        if radial_abs_nonzero.size
        else 1.0
    )
    display_coordinates = np.argwhere(display_valid)
    row_start, col_start = (int(value) for value in display_coordinates.min(axis=0))
    row_stop, col_stop = (int(value) + 1 for value in display_coordinates.max(axis=0))
    display_slice = (slice(row_start, row_stop), slice(col_start, col_stop))
    crop_q_finite = bool(
        np.all(np.isfinite(qx_float[display_slice]))
        and np.all(np.isfinite(qy_float[display_slice]))
        and not np.any(qx_mask[display_slice])
        and not np.any(qy_mask[display_slice])
    )
    crop_valid = display_valid[display_slice]
    display = _display_transform(
        np.where(crop_valid, observed_float[display_slice], np.nan), scale
    )

    unit = str(q_unit or "unknown")
    result_safe = json_safe(result)
    point_rows = result.get("points", ())
    if not isinstance(point_rows, Sequence) or isinstance(point_rows, (str, bytes)):
        point_rows = ()
    points = [dict(row) for row in point_rows if isinstance(row, Mapping)]
    point_overlay = []
    for point in points:
        reason = _point_overlay_reason(
            point,
            effective_valid=effective_valid,
            display_valid=display_valid,
        )
        point_overlay.append(
            {
                "status": "drawn" if reason is None else "omitted",
                "reason": reason,
            }
        )
    plot_points = [
        point
        for point, overlay in zip(points, point_overlay, strict=True)
        if overlay["status"] == "drawn"
    ]
    omitted_point_reasons: dict[str, int] = {}
    for overlay in point_overlay:
        if overlay["status"] == "omitted":
            reason = str(overlay["reason"])
            omitted_point_reasons[reason] = omitted_point_reasons.get(reason, 0) + 1
    arc_rows = result.get("arcs", ())
    if not isinstance(arc_rows, Sequence) or isinstance(arc_rows, (str, bytes)):
        arc_rows = ()
    arcs = [dict(row) for row in arc_rows if isinstance(row, Mapping)]

    display_low_count = int(np.count_nonzero(selected_display < low))
    display_high_count = int(np.count_nonzero(selected_display > high))
    return {
        "observed_raw": observed_raw,
        "qx_raw": qx_raw,
        "qy_raw": qy_raw,
        "observed_float": observed_float,
        "qx_float": qx_float,
        "qy_float": qy_float,
        "observed_mask": observed_mask,
        "qx_mask": qx_mask,
        "qy_mask": qy_mask,
        "source_valid_mask": source_valid_mask,
        "source_valid_mask_mask": source_mask_mask,
        "effective_valid": effective_valid,
        "display_valid": display_valid,
        "display": display,
        "display_low": low,
        "display_high": high,
        "display_scale": scale,
        "display_label": transform_label,
        "transform_scale": transform_scale,
        "display_low_count": display_low_count,
        "display_high_count": display_high_count,
        "radial_edges": radial_edges,
        "radial_centers": radial_centers,
        "radial_mean": radial_mean,
        "radial_sum": radial_sum,
        "radial_counts": radial_counts,
        "radial_linthresh": radial_linthresh,
        "radial_bin_count": bin_count,
        "q_window": q_window,
        "q_window_source": q_window_source,
        "q_window_applied": q_window_applied,
        "q_window_fallback_reason": q_window_fallback_reason,
        "display_crop": {
            "row_start": row_start,
            "row_stop": row_stop,
            "col_start": col_start,
            "col_stop": col_stop,
            "shape": [row_stop - row_start, col_stop - col_start],
            "coordinates_finite": crop_q_finite,
        },
        "display_render_primitive": (
            "pcolormesh"
            if crop_q_finite and min(row_stop - row_start, col_stop - col_start) >= 2
            else "valid_q_sample_scatter"
        ),
        "width_mm": width,
        "height_mm": 150.0 if width == 89.0 else 94.0,
        "dpi": dpi,
        "q_unit": unit,
        "context": json_safe(context),
        "result_safe": result_safe,
        "points": points,
        "point_overlay": point_overlay,
        "plot_points": plot_points,
        "omitted_point_reasons": omitted_point_reasons,
        "arcs": arcs,
        "valid_mask_was_supplied": valid_mask_was_supplied,
        "supplied_valid_mask_role": supplied_valid_mask_role,
        "input_pixel_count": int(observed_raw.size),
        "input_valid_mask_count": int(
            np.count_nonzero(source_valid_mask & ~source_mask_mask)
        ),
        "valid_pixel_count": full_valid_count,
        "invalid_pixel_count": int(observed_raw.size - full_valid_count),
        "display_pixel_count": display_count,
        "color_clip_percentiles": [0.5, 99.5],
        "color_clip_low_count": display_low_count,
        "color_clip_high_count": display_high_count,
        "q_axis_interpretation": (
            "unknown unit; no physical period is inferred"
            if unit.strip().lower() in {"", "unknown", "pixel", "pixel_q", "px"}
            else "caller-supplied unit; no period conversion is performed"
        ),
    }


def _point_coordinates(point: Mapping[str, Any]) -> tuple[float, float] | None:
    try:
        x, y = float(point["qx"]), float(point["qy"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(x) or not math.isfinite(y):
        return None
    return x, y


def _flag_is(value: Any, expected: bool) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value) is expected
    return False


def _point_display_status(point: Mapping[str, Any]) -> str:
    accepted = point.get("accepted")
    valid = point.get("valid")
    if _flag_is(accepted, False) or _flag_is(valid, False):
        return "rejected"
    if _flag_is(accepted, True) and not _flag_is(valid, False):
        return "accepted"
    return "unspecified"


def _pixel_xy(point: Mapping[str, Any]) -> tuple[float, float] | None:
    try:
        x, y = float(point["pixel_x"]), float(point["pixel_y"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(x) or not math.isfinite(y):
        return None
    return x, y


def _point_overlay_reason(
    point: Mapping[str, Any],
    *,
    effective_valid: np.ndarray,
    display_valid: np.ndarray,
) -> str | None:
    """Require a source pixel and valid q/image support before drawing a point."""

    if _point_coordinates(point) is None:
        return "missing_or_nonfinite_q_coordinates"
    if "pixel_x" not in point or "pixel_y" not in point:
        return "missing_pixel_coordinates"
    pixel = _pixel_xy(point)
    if pixel is None:
        return "nonfinite_pixel_coordinates"
    x, y = pixel
    rows, cols = effective_valid.shape
    if x < 0.0 or y < 0.0 or x > cols - 1 or y > rows - 1:
        return "pixel_coordinate_out_of_bounds"
    col, row = int(round(x)), int(round(y))
    if not effective_valid[row, col]:
        return "source_pixel_invalid_or_masked"
    if not display_valid[row, col]:
        return "outside_display_q_window"
    return None


def _segment_is_valid(
    first: Mapping[str, Any], second: Mapping[str, Any], valid_pixels: np.ndarray
) -> bool:
    """Keep an observed arc segment only when its displayed pixel path is valid."""

    p0, p1 = _pixel_xy(first), _pixel_xy(second)
    if p0 is None or p1 is None:
        return False
    x0, y0 = p0
    x1, y1 = p1
    steps = max(1, int(math.ceil(max(abs(x1 - x0), abs(y1 - y0)) * 2.0)))
    rows, cols = valid_pixels.shape
    for fraction in np.linspace(0.0, 1.0, steps + 1):
        col = int(round(x0 + fraction * (x1 - x0)))
        row = int(round(y0 + fraction * (y1 - y0)))
        if (
            row < 0
            or row >= rows
            or col < 0
            or col >= cols
            or not valid_pixels[row, col]
        ):
            return False
    return True


def _supported_arc_segments(data: Mapping[str, Any]) -> list[np.ndarray]:
    points = data["points"]
    by_id: dict[str, list[dict[str, Any]]] = {}
    for point in points:
        if "point_id" in point and point["point_id"] is not None:
            by_id.setdefault(str(point["point_id"]), []).append(point)
    segments: list[np.ndarray] = []
    for arc in data["arcs"]:
        if not _flag_is(arc.get("valid"), True):
            continue
        point_ids = arc.get("ordered_point_ids", arc.get("point_ids", ()))
        if not isinstance(point_ids, Sequence) or isinstance(point_ids, (str, bytes)):
            continue
        ordered: list[dict[str, Any]] = []
        for point_id in point_ids:
            candidates = by_id.get(str(point_id), ())
            if len(candidates) != 1:
                ordered = []
                break
            point = candidates[0]
            if _point_display_status(point) != "accepted":
                ordered = []
                break
            if (
                _point_overlay_reason(
                    point,
                    effective_valid=data["effective_valid"],
                    display_valid=data["display_valid"],
                )
                is not None
            ):
                ordered = []
                break
            ordered.append(point)
        if len(ordered) < 2:
            continue
        for first, second in zip(ordered, ordered[1:]):
            xy0, xy1 = _point_coordinates(first), _point_coordinates(second)
            if xy0 is None or xy1 is None:
                continue
            if _segment_is_valid(first, second, data["display_valid"]):
                segments.append(np.asarray([xy0, xy1], dtype=np.float64))
    return segments


def _draw_map(
    fig: Figure,
    ax: Any,
    colorbar_ax: Any,
    data: Mapping[str, Any],
    *,
    draw_overlays: bool = True,
    cmap: str = "cividis",
    colorbar_orientation: str = "vertical",
    legend_loc: str = "best",
) -> None:
    crop = data["display_crop"]
    display_slice = (
        slice(crop["row_start"], crop["row_stop"]),
        slice(crop["col_start"], crop["col_stop"]),
    )
    image_values = data["display"]
    render_valid = data["display_valid"][display_slice]
    image = np.ma.masked_where(~render_valid | ~np.isfinite(image_values), image_values)
    qx = data["qx_float"][display_slice]
    qy = data["qy_float"][display_slice]
    norm = Normalize(data["display_low"], data["display_high"], clip=True)
    if data["display_render_primitive"] == "pcolormesh":
        mesh = ax.pcolormesh(
            qx,
            qy,
            image,
            cmap=cmap,
            norm=norm,
            shading="auto",
            rasterized=True,
        )
    else:
        valid = (
            render_valid
            & np.isfinite(qx)
            & np.isfinite(qy)
            & ~data["qx_mask"][display_slice]
            & ~data["qy_mask"][display_slice]
        )
        mesh = ax.scatter(
            qx[valid],
            qy[valid],
            c=data["display"][valid],
            cmap=cmap,
            norm=norm,
            marker="s",
            s=8.0,
            rasterized=True,
        )
    qx_valid = qx[render_valid]
    qy_valid = qy[render_valid]
    xmin, xmax = float(np.min(qx_valid)), float(np.max(qx_valid))
    ymin, ymax = float(np.min(qy_valid)), float(np.max(qy_valid))
    if xmax <= xmin:
        margin = max(abs(xmin) * 0.05, 0.5)
        xmin, xmax = xmin - margin, xmax + margin
    if ymax <= ymin:
        margin = max(abs(ymin) * 0.05, 0.5)
        ymin, ymax = ymin - margin, ymax + margin
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    qx_label, qy_label = _diagnostic_q_axis_labels(data["q_unit"])
    ax.set_xlabel(qx_label)
    ax.set_ylabel(qy_label)
    ax.tick_params(direction="out", length=2.0, width=0.5, pad=1.5)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)

    if not draw_overlays:
        colorbar = fig.colorbar(mesh, cax=colorbar_ax, orientation=colorbar_orientation)
        colorbar.set_label(data["display_label"], fontsize=5.7, labelpad=2.0)
        colorbar.ax.tick_params(labelsize=5.0, length=1.5, width=0.4, pad=1.0)
        colorbar.outline.set_linewidth(0.45)
        return

    segments = _supported_arc_segments(data)
    for segment in segments:
        ax.plot(
            segment[:, 0],
            segment[:, 1],
            color="#f0e442",
            linewidth=0.65,
            alpha=0.95,
            solid_capstyle="round",
            zorder=3,
        )

    plotted: dict[str, list[tuple[float, float]]] = {
        "accepted": [],
        "rejected": [],
        "unspecified": [],
    }
    for point in data["plot_points"]:
        coordinates = _point_coordinates(point)
        if coordinates is not None:
            plotted[_point_display_status(point)].append(coordinates)
    styles = {
        "accepted": {
            "marker": "o",
            "color": "#0072b2",
            "label": "accepted source point",
        },
        "rejected": {
            "marker": "x",
            "color": "#d55e00",
            "label": "rejected source point",
        },
        "unspecified": {
            "marker": "s",
            "color": "#666666",
            "label": "status not supplied",
        },
    }
    for status, coordinates in plotted.items():
        if not coordinates:
            continue
        values = np.asarray(coordinates, dtype=np.float64)
        style = styles[status]
        if status == "unspecified":
            ax.scatter(
                values[:, 0],
                values[:, 1],
                marker=style["marker"],
                s=12.0,
                facecolors="none",
                edgecolors=style["color"],
                linewidths=0.8,
                label=style["label"],
                zorder=4,
            )
        elif status == "rejected":
            ax.scatter(
                values[:, 0],
                values[:, 1],
                marker=style["marker"],
                s=13.0,
                c=style["color"],
                linewidths=0.8,
                label=style["label"],
                zorder=4,
            )
        else:
            ax.scatter(
                values[:, 0],
                values[:, 1],
                marker=style["marker"],
                s=13.0,
                c=style["color"],
                linewidths=0.8,
                edgecolors="white",
                label=style["label"],
                zorder=4,
            )
    if not any(plotted.values()):
        ax.text(
            0.5,
            0.5,
            "No measured ridge points",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="black",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 2.0},
        )

    legend_handles = (
        [
            Line2D(
                [0], [0], color="#f0e442", linewidth=0.8, label="supported observed arc"
            )
        ]
        if segments
        else []
    )
    for status, coordinates in plotted.items():
        if not coordinates:
            continue
        style = styles[status]
        legend_handles.append(
            Line2D(
                [0],
                [0],
                linestyle="none",
                marker=style["marker"],
                markerfacecolor="none" if status == "unspecified" else style["color"],
                markeredgecolor=style["color"],
                color=style["color"],
                markersize=3.5,
                label=style["label"],
            )
        )
    if legend_handles:
        ax.legend(
            handles=legend_handles,
            loc=legend_loc,
            frameon=True,
            framealpha=0.88,
            edgecolor="#cccccc",
            borderpad=0.25,
            handlelength=1.3,
            labelspacing=0.25,
        )
    colorbar = fig.colorbar(mesh, cax=colorbar_ax, orientation=colorbar_orientation)
    colorbar.set_label(data["display_label"], fontsize=5.7, labelpad=2.0)
    colorbar.ax.tick_params(labelsize=5.0, length=1.5, width=0.4, pad=1.0)
    colorbar.outline.set_linewidth(0.45)
