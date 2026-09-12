"""Measured-image, candidate-geometry, and fit-residual comparison exports.

This module only renders data already present in the caller's arrays/result.
It never synthesizes an intensity field from ellipse geometry.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
import math
from pathlib import Path
from typing import Any

import matplotlib as mpl
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import numpy as np

from .cancellation import raise_if_cancelled

from .candidate_geometry import (
    _CURVE_PHI,
    _candidate,
    _candidate_curves,
    _ellipse_state as _ellipse_state,
    _finite,
    _flag,
    _mapping_rows,
    _member_geometry as _member_geometry,
)
from .serialization import json_safe
from .visualization import _diagnostic_q_axis_labels, _display_transform


_CURVE_COLORS = ("#cc79a7", "#e69f00", "#56b4e9", "#009e73")


def _candidate_point_diagnostics(
    result: Mapping[str, Any], candidate: Mapping[str, Any] | None
) -> list[dict[str, Any]]:
    if candidate is not None and candidate.get("point_diagnostics") is not None:
        return _mapping_rows(candidate.get("point_diagnostics"))
    return _mapping_rows(result.get("point_diagnostics"))


def _profile_arrays(profile: Mapping[str, Any]) -> dict[str, np.ndarray] | None:
    offset = np.asarray(profile.get("offset_q", []), dtype=np.float64)
    raw = np.asarray(
        profile.get("raw_intensity", profile.get("raw", [])), dtype=np.float64
    )
    fit = np.asarray(
        profile.get("fit_intensity", profile.get("fit", [])), dtype=np.float64
    )
    if offset.ndim != 1 or raw.ndim != 1 or fit.ndim != 1 or not offset.size:
        return None
    if offset.shape != raw.shape or raw.shape != fit.shape:
        return None
    residual = np.asarray(profile.get("residual", []), dtype=np.float64)
    residual_definition = str(profile.get("residual_definition", ""))
    if residual.shape != raw.shape:
        residual = raw - fit
        residual_definition = "raw_intensity_minus_fit_intensity"
    return {
        "offset_q": offset,
        "raw_intensity": raw,
        "fit_intensity": fit,
        "residual": residual,
        "residual_definition": np.asarray(residual_definition),
    }


def _profile_export_rows(
    result: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, dict[str, Any] | None]:
    raw_profiles = result.get("profiles")
    if not isinstance(raw_profiles, Mapping):
        return [], None, None
    profiles: list[dict[str, Any]] = []
    for key, value in raw_profiles.items():
        if not isinstance(value, Mapping):
            continue
        arrays = _profile_arrays(value)
        if arrays is None:
            continue
        point_id = str(value.get("point_id", key))
        profile = {"point_id": point_id, **arrays}
        profiles.append(profile)

    point_by_id = {
        str(point.get("point_id")): point
        for point in _mapping_rows(result.get("points"))
        if point.get("point_id") is not None
    }
    supported = []
    for profile in profiles:
        point = point_by_id.get(profile["point_id"])
        if (
            point is None
            or _flag(point.get("accepted")) is not True
            or _flag(point.get("valid")) is False
        ):
            continue
        qx, qy = _finite(point.get("qx")), _finite(point.get("qy"))
        if qx is None or qy is None:
            continue
        profile_valid = _flag(
            result.get("profiles", {}).get(profile["point_id"], {}).get("valid")
        )
        if profile_valid is False:
            continue
        supported.append((math.hypot(qx, qy), profile))
    if not supported:
        return profiles, None, None
    median_radius = float(np.median([item[0] for item in supported]))
    radius, chosen = min(
        supported,
        key=lambda item: (abs(item[0] - median_radius), item[1]["point_id"]),
    )
    selected = {
        "point_id": chosen["point_id"],
        "q_radius": radius,
        "selection": "accepted/valid profiled source point nearest the median q radius",
    }
    return profiles, selected, chosen


def _csv_value(value: Any) -> Any:
    value = json_safe(value)
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        import json

        return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
    return value


def _write_curve_sidecars(
    directory: Path, curves: Sequence[Mapping[str, Any]], state: str
) -> dict[str, Path]:
    csv_path = directory / "ellipse_curves.csv"
    fields = (
        "curve_index",
        "point_index",
        "phi_rad",
        "qx",
        "qy",
        "branch_id",
        "fit_branch_id",
        "curve_role",
        "scientific_status",
        "fit_state",
    )
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for curve in curves:
            xy = np.asarray(curve["points"], dtype=np.float64)
            phi = np.asarray(curve["phi_rad"], dtype=np.float64)
            for point_index, (angle, point) in enumerate(zip(phi, xy, strict=True)):
                writer.writerow(
                    {
                        "curve_index": curve["curve_index"],
                        "point_index": point_index,
                        "phi_rad": float(angle),
                        "qx": float(point[0]),
                        "qy": float(point[1]),
                        "branch_id": _csv_value(curve.get("branch_id")),
                        "fit_branch_id": _csv_value(curve.get("fit_branch_id")),
                        "curve_role": curve["curve_role"],
                        "scientific_status": curve["scientific_status"],
                        "fit_state": state,
                    }
                )
    point_count = _CURVE_PHI.size
    points_array = (
        np.stack([np.asarray(curve["points"], dtype=np.float64) for curve in curves])
        if curves
        else np.empty((0, point_count, 2), dtype=np.float64)
    )
    npz_path = directory / "ellipse_curves.npz"
    np.savez_compressed(
        npz_path,
        phi_rad=_CURVE_PHI,
        q_points=points_array,
        branch_id=np.asarray(
            [str(curve.get("branch_id", "")) for curve in curves], dtype="U64"
        ),
        fit_branch_id=np.asarray(
            [str(curve.get("fit_branch_id", "")) for curve in curves], dtype="U64"
        ),
        fit_state=np.asarray(state),
    )
    return {"ellipse_curves_csv": csv_path, "ellipse_curves_npz": npz_path}


def _write_point_residuals(
    path: Path,
    points: Sequence[Mapping[str, Any]],
    diagnostics: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], str | None]:
    diag_by_id: dict[str, dict[str, Any]] = {}
    diag_by_index: dict[int, dict[str, Any]] = {}
    for diagnostic in diagnostics:
        point_id = diagnostic.get("point_id")
        if point_id is not None:
            diag_by_id[str(point_id)] = dict(diagnostic)
        index = diagnostic.get("index")
        if isinstance(index, (int, np.integer)) and not isinstance(
            index, (bool, np.bool_)
        ):
            diag_by_index[int(index)] = dict(diagnostic)
    rows: list[dict[str, Any]] = []
    for index, point in enumerate(points):
        point_id = point.get("point_id")
        diagnostic = diag_by_id.get(str(point_id)) if point_id is not None else None
        if diagnostic is None:
            diagnostic = diag_by_index.get(index, {})
        qx, qy = _finite(point.get("qx")), _finite(point.get("qy"))
        radius = math.hypot(qx, qy) if qx is not None and qy is not None else None
        rows.append(
            {
                "source_index": index,
                "point_id": point_id,
                "arc_id": point.get("arc_id", diagnostic.get("arc_id")),
                "branch_id": point.get("branch_id", diagnostic.get("branch_id")),
                "side": point.get("side", diagnostic.get("side")),
                "accepted": point.get("accepted"),
                "qx": qx,
                "qy": qy,
                "q_radius": radius,
                "used": diagnostic.get("used"),
                "projection_valid": diagnostic.get("projection_valid"),
                "projection_t": diagnostic.get("projection_t"),
                "projection_residual_q": diagnostic.get("projection_residual_q"),
                "normal_residual_q": diagnostic.get("normal_residual_q"),
                "projection_distance_q": diagnostic.get(
                    "projection_distance_q", diagnostic.get("distance_q")
                ),
                "projection_reason": diagnostic.get(
                    "projection_reason", diagnostic.get("excluded_reason")
                ),
            }
        )
    residual_field = None
    for name in ("projection_residual_q", "normal_residual_q"):
        if any(_finite(row.get(name)) is not None for row in rows):
            residual_field = name
            break
    fields = (
        "source_index",
        "point_id",
        "arc_id",
        "branch_id",
        "side",
        "accepted",
        "qx",
        "qy",
        "q_radius",
        "used",
        "projection_valid",
        "projection_t",
        "projection_residual_q",
        "normal_residual_q",
        "projection_distance_q",
        "projection_reason",
    )
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(value) for key, value in row.items()})
    return rows, residual_field


def _write_profiles(path: Path, profiles: Sequence[Mapping[str, Any]]) -> None:
    fields = (
        "point_id",
        "sample_index",
        "offset_q",
        "raw_intensity",
        "fit_intensity",
        "residual",
        "residual_definition",
    )
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for profile in profiles:
            count = len(profile["offset_q"])
            for index in range(count):
                writer.writerow(
                    {
                        "point_id": profile["point_id"],
                        "sample_index": index,
                        "offset_q": _csv_value(profile["offset_q"][index]),
                        "raw_intensity": _csv_value(profile["raw_intensity"][index]),
                        "fit_intensity": _csv_value(profile["fit_intensity"][index]),
                        "residual": _csv_value(profile["residual"][index]),
                        "residual_definition": str(
                            profile["residual_definition"].item()
                        ),
                    }
                )


def _set_q_axes(ax: Any, q_unit: str, q_bounds: Sequence[float]) -> None:
    xmin, xmax, ymin, ymax = (float(value) for value in q_bounds)
    if xmax <= xmin:
        xmin, xmax = xmin - 0.5, xmax + 0.5
    if ymax <= ymin:
        ymin, ymax = ymin - 0.5, ymax + 0.5
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    qx_label, qy_label = _diagnostic_q_axis_labels(q_unit)
    ax.set_xlabel(qx_label, fontsize=5.8)
    ax.set_ylabel(qy_label, fontsize=5.8)
    ax.tick_params(direction="out", length=1.7, width=0.45, pad=1.1, labelsize=5.0)
    for spine in ax.spines.values():
        spine.set_linewidth(0.45)


def _map_bounds(data: Mapping[str, Any]) -> tuple[float, float, float, float]:
    crop = data["display_crop"]
    display_slice = (
        slice(crop["row_start"], crop["row_stop"]),
        slice(crop["col_start"], crop["col_stop"]),
    )
    valid = data["display_valid"][display_slice]
    qx = data["qx_float"][display_slice][valid]
    qy = data["qy_float"][display_slice][valid]
    return float(np.min(qx)), float(np.max(qx)), float(np.min(qy)), float(np.max(qy))


def _curve_bounds(
    data: Mapping[str, Any], curves: Sequence[Mapping[str, Any]]
) -> tuple[float, float, float, float]:
    xmin, xmax, ymin, ymax = _map_bounds(data)
    if curves:
        all_points = np.concatenate(
            [np.asarray(curve["points"], dtype=np.float64) for curve in curves]
        )
        xmin, xmax = (
            min(xmin, float(np.min(all_points[:, 0]))),
            max(xmax, float(np.max(all_points[:, 0]))),
        )
        ymin, ymax = (
            min(ymin, float(np.min(all_points[:, 1]))),
            max(ymax, float(np.max(all_points[:, 1]))),
        )
    return xmin, xmax, ymin, ymax


def _curve_only_bounds(
    curves: Sequence[Mapping[str, Any]],
) -> tuple[float, float, float, float]:
    points = np.concatenate(
        [np.asarray(curve["points"], dtype=np.float64) for curve in curves], axis=0
    )
    return (
        float(np.min(points[:, 0])),
        float(np.max(points[:, 0])),
        float(np.min(points[:, 1])),
        float(np.max(points[:, 1])),
    )


def _curve_bounds_with_margin(
    curves: Sequence[Mapping[str, Any]], fraction: float = 0.04
) -> tuple[float, float, float, float]:
    xmin, xmax, ymin, ymax = _curve_only_bounds(curves)
    x_span, y_span = xmax - xmin, ymax - ymin
    fallback_span = max(x_span, y_span, np.finfo(np.float64).eps)
    x_margin = fraction * (x_span if x_span > 0.0 else fallback_span)
    y_margin = fraction * (y_span if y_span > 0.0 else fallback_span)
    return (
        xmin - x_margin,
        xmax + x_margin,
        ymin - y_margin,
        ymax + y_margin,
    )


def _curve_style(curve: Mapping[str, Any], fallback_index: int) -> tuple[str, str]:
    branch = curve.get("branch_id")
    if branch == 0:
        color = "#cc79a7"
    elif branch == 1:
        color = "#e69f00"
    else:
        color = _CURVE_COLORS[fallback_index % len(_CURVE_COLORS)]
    return color, str(curve["label"])


def _draw_curves(
    ax: Any,
    curves: Sequence[Mapping[str, Any]],
    *,
    add_legend: bool,
    legend_loc: str = "best",
    label_prefix: str = "",
) -> list[Line2D]:
    handles = []
    for index, curve in enumerate(curves):
        points = np.asarray(curve["points"], dtype=np.float64)
        color, base_label = _curve_style(curve, index)
        label = f"{label_prefix}{base_label}"
        ax.plot(
            points[:, 0],
            points[:, 1],
            color=color,
            linewidth=0.95,
            linestyle="--",
            zorder=5,
            label=label,
        )
        handles.append(
            Line2D([0], [0], color=color, linewidth=0.95, linestyle="--", label=label)
        )
    if add_legend and handles:
        ax.legend(
            handles=handles,
            loc=legend_loc,
            frameon=True,
            framealpha=0.88,
            fontsize=5.2,
        )
    return handles


def _plot_point_residuals(
    ax: Any, rows: Sequence[Mapping[str, Any]], field: str, q_unit: str
) -> bool:
    usable = []
    for row in rows:
        x, y = _finite(row.get("q_radius")), _finite(row.get(field))
        if x is None or y is None:
            continue
        if row.get("used") is False or row.get("projection_valid") is False:
            continue
        usable.append((x, y, row.get("branch_id")))
    if not usable:
        return False
    for branch in sorted({row[2] for row in usable if row[2] in (0, 1)}):
        values = np.asarray(
            [(x, y) for x, y, bid in usable if bid == branch], dtype=np.float64
        )
        ax.scatter(
            values[:, 0],
            values[:, 1],
            s=10.0,
            color="#0072b2" if branch == 0 else "#d55e00",
            label=f"branch {branch}",
        )
    unknown = np.asarray(
        [(x, y) for x, y, bid in usable if bid not in (0, 1)], dtype=np.float64
    )
    if unknown.size:
        ax.scatter(
            unknown[:, 0],
            unknown[:, 1],
            s=10.0,
            color="#666666",
            marker="s",
            label="branch unspecified",
        )
    ax.axhline(0.0, color="#555555", linewidth=0.5, linestyle="--")
    ax.set_xlabel(f"Observed point q radius ({q_unit})")
    residual_label = (
        "Projection residual" if field == "projection_residual_q" else "Normal residual"
    )
    ax.set_ylabel(f"{residual_label} ({q_unit})")
    ax.tick_params(labelsize=5.0, length=1.7, width=0.45, pad=1.1)
    ax.grid(axis="y", color="#dddddd", linewidth=0.35)
    ax.legend(loc="best", frameon=True, framealpha=0.88, fontsize=5.0)
    return True


def _plot_profile_pair(
    fig: Figure,
    ax: Any,
    profile: Mapping[str, Any] | None,
    *,
    width_mm: float,
    height_mm: float,
    q_unit: str,
) -> bool:
    if profile is None:
        return False
    offset = np.asarray(profile["offset_q"], dtype=np.float64)
    raw = np.asarray(profile["raw_intensity"], dtype=np.float64)
    fit = np.asarray(profile["fit_intensity"], dtype=np.float64)
    residual = np.asarray(profile["residual"], dtype=np.float64)
    if not offset.size:
        return False
    box = ax.get_position()
    ax.remove()
    gap = 0.03
    top_h = (box.height - gap) * 0.62
    lower_h = box.height - gap - top_h
    ax_top = fig.add_axes([box.x0, box.y0 + lower_h + gap, box.width, top_h])
    ax_bottom = fig.add_axes([box.x0, box.y0, box.width, lower_h], sharex=ax_top)
    ax_top.plot(offset, raw, color="#0072b2", linewidth=0.8, label="profile raw")
    ax_top.plot(
        offset, fit, color="#d55e00", linewidth=0.8, linestyle="--", label="profile fit"
    )
    ax_top.set_title(
        f"Normal profile · source point {profile['point_id']}",
        loc="left",
        fontsize=5.3,
        pad=1.0,
    )
    ax_top.set_ylabel("Intensity", fontsize=5.3)
    ax_top.tick_params(labelsize=5.0, length=1.5, width=0.4, pad=0.7, labelbottom=False)
    ax_top.legend(loc="best", frameon=True, fontsize=5.0)
    ax_bottom.plot(offset, residual, color="#0072b2", linewidth=0.75)
    ax_bottom.axhline(0.0, color="#555555", linewidth=0.45, linestyle="--")
    ax_bottom.set_xlabel(f"Normal offset ({q_unit})", fontsize=5.3)
    ax_bottom.set_ylabel("Raw − fit", fontsize=5.3)
    ax_bottom.tick_params(labelsize=5.0, length=1.5, width=0.4, pad=0.7)
    return True


def _comparison_figure(
    data: Mapping[str, Any],
    *,
    curves: Sequence[Mapping[str, Any]],
    fit_state: str,
    residual_rows: Sequence[Mapping[str, Any]],
    residual_field: str | None,
    selected_profile_data: Mapping[str, Any] | None,
) -> tuple[Figure, str]:
    from .figure_support import _draw_map

    width_mm = data["width_mm"]
    height_mm = 150.0
    fig = Figure(
        figsize=(width_mm / 25.4, height_mm / 25.4),
        dpi=data["dpi"],
        facecolor="white",
        edgecolor="white",
    )
    FigureCanvasAgg(fig)
    if width_mm == 89.0:
        map_a = fig.add_axes([0.13, 0.77, 0.74, 0.18])
        curve_ax = fig.add_axes([0.13, 0.53, 0.74, 0.18])
        map_c = fig.add_axes([0.13, 0.29, 0.74, 0.18])
        residual_ax = fig.add_axes([0.13, 0.05, 0.74, 0.14])
        colorbar_ax = fig.add_axes([0.90, 0.29, 0.025, 0.66])
    else:
        row_h, top_y, bottom_y = 0.34, 0.57, 0.10
        map_a = fig.add_axes([0.055, top_y, 0.39, row_h])
        map_c = fig.add_axes([0.055, bottom_y, 0.39, row_h])
        curve_ax = fig.add_axes([0.56, top_y, 0.385, row_h])
        residual_ax = fig.add_axes([0.56, bottom_y, 0.385, row_h])
        colorbar_ax = fig.add_axes([0.465, bottom_y + 0.03, 0.014, 0.81])
    map_a.set_gid("butterfly-measured-image")
    map_c.set_gid("butterfly-measured-overlay")
    curve_ax.set_gid("butterfly-candidate-curves")
    residual_ax.set_gid("butterfly-residual-diagnostic")
    _draw_map(fig, map_a, colorbar_ax, data, draw_overlays=False)
    _draw_map(fig, map_c, colorbar_ax, data, draw_overlays=True, legend_loc="center")
    source_legend = map_c.get_legend()
    source_legend_entries: list[tuple[Any, str]] = []
    if source_legend is not None:
        source_handles = getattr(source_legend, "legend_handles", None)
        if source_handles is None:
            source_handles = getattr(source_legend, "legendHandles", ())
        source_labels = [text.get_text() for text in source_legend.get_texts()]
        source_legend_entries = list(
            zip(source_handles, source_labels, strict=True)
        )
        source_legend.remove()
    _set_q_axes(curve_ax, data["q_unit"], _curve_bounds(data, curves))
    curve_ax.grid(color="#eeeeee", linewidth=0.3)
    if (
        fit_state in {"candidate", "candidate_unconverged", "ring_only_candidate"}
        and curves
    ):
        _draw_curves(curve_ax, curves, add_legend=True, legend_loc="upper right")
        fit_label = (
            "Ring-only / bound-limited candidate · not quantitative"
            if fit_state == "ring_only_candidate"
            else "Unsuccessful candidate · diagnostic only"
            if fit_state == "candidate_unconverged"
            else "Candidate ellipse · not scientifically accepted"
        )
        curve_ax.set_title(fit_label, loc="left", fontsize=5.3, pad=1.5)
        _draw_curves(
            map_c,
            curves,
            add_legend=False,
            label_prefix="candidate · ",
        )
    else:
        message = {
            "ring_only_candidate": "Ring-only/bound-limited candidate geometry unavailable",
            "fit_unavailable": "Candidate ellipse unavailable",
            "not_available": "Trace only; no ellipse fit",
        }.get(fit_state, "No fitted ellipse curves")
        curve_ax.text(
            0.5,
            0.5,
            message,
            transform=curve_ax.transAxes,
            ha="center",
            va="center",
            fontsize=5.7,
            wrap=True,
        )

    map_handles, map_labels = map_c.get_legend_handles_labels()
    legend_entries = list(source_legend_entries)
    seen_legend_labels = {label for _handle, label in legend_entries}
    for handle, label in zip(map_handles, map_labels, strict=True):
        if label not in seen_legend_labels:
            legend_entries.append((handle, label))
            seen_legend_labels.add(label)
    if legend_entries:
        legend_position = (
            {"loc": "lower center", "bbox_to_anchor": (0.5, 1.05)}
            if width_mm == 183.0
            else {"loc": "upper center", "bbox_to_anchor": (0.5, -0.28)}
        )
        map_c.legend(
            handles=[handle for handle, _label in legend_entries],
            labels=[label for _handle, label in legend_entries],
            frameon=False,
            ncol=2,
            fontsize=5.0,
            columnspacing=0.8,
            handlelength=1.1,
            handletextpad=0.4,
            labelspacing=0.2,
            borderaxespad=0.0,
            **legend_position,
        )

    if residual_field is not None and _plot_point_residuals(
        residual_ax, residual_rows, residual_field, data["q_unit"]
    ):
        diagnostic_selection = f"Point residual source: {residual_field}"
    elif selected_profile_data is not None and _plot_profile_pair(
        fig,
        residual_ax,
        selected_profile_data,
        width_mm=width_mm,
        height_mm=height_mm,
        q_unit=data["q_unit"],
    ):
        diagnostic_selection = (
            "Normal profile for source point "
            + str(selected_profile_data["point_id"])
            + " (accepted/valid point nearest median q radius)"
        )
    else:
        residual_ax.text(
            0.5,
            0.5,
            "No fit residual or fitted normal profile available",
            transform=residual_ax.transAxes,
            ha="center",
            va="center",
            fontsize=5.6,
            wrap=True,
        )
        residual_ax.set_axis_off()
        diagnostic_selection = "unavailable; no residual data fabricated"
    for ax, label in ((map_a, "a"), (curve_ax, "b"), (map_c, "c"), (residual_ax, "d")):
        box = ax.get_position()
        fig.text(
            max(0.025, box.x0 - (0.07 if width_mm == 89.0 else 0.025)),
            min(0.97, box.y1 + 0.018),
            label,
            ha="left",
            va="bottom",
            fontsize=8.0,
            fontweight="bold",
        )
    return fig, diagnostic_selection


def _ellipse_only_figure(
    data: Mapping[str, Any], curves: Sequence[Mapping[str, Any]], fit_state: str
) -> Figure:
    width_mm = data["width_mm"]
    height_mm = 150.0 if width_mm == 89.0 else 94.0
    fig = Figure(
        figsize=(width_mm / 25.4, height_mm / 25.4),
        dpi=data["dpi"],
        facecolor="white",
        edgecolor="white",
    )
    FigureCanvasAgg(fig)
    ax = fig.add_axes([0.17, 0.16, 0.72, 0.75])
    _set_q_axes(ax, data["q_unit"], _curve_bounds_with_margin(curves))
    ax.grid(color="#eeeeee", linewidth=0.35)
    _draw_curves(ax, curves, add_legend=True, legend_loc="upper right")
    label = (
        "Ring-only candidate · bound-limited, not quantitative"
        if fit_state == "ring_only_candidate"
        else "Unsuccessful candidate · diagnostic only"
        if fit_state == "candidate_unconverged"
        else "Candidate ellipse · not scientifically accepted"
    )
    ax.set_title(label, loc="left", fontsize=5.8, pad=2.0)
    return fig


def _model_arrays(
    model: Any, data: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from .figure_support import _numeric_array

    raw, masked = _numeric_array(model, "model")
    if raw.shape != data["observed_raw"].shape:
        raise ValueError("model must have the same shape as observed")
    try:
        model_float = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError("model must be convertible to float64") from exc
    common_valid = data["display_valid"] & ~masked & np.isfinite(model_float)
    if not np.any(common_valid):
        raise ValueError("model has no finite samples in the valid display domain")
    return raw, model_float, common_valid


def _crop_data(
    data: Mapping[str, Any],
    valid: np.ndarray,
    values: np.ndarray,
    *,
    low: float,
    high: float,
    label: str,
) -> dict[str, Any]:
    coords = np.argwhere(valid)
    r0, c0 = (int(value) for value in coords.min(axis=0))
    r1, c1 = (int(value) + 1 for value in coords.max(axis=0))
    crop = (slice(r0, r1), slice(c0, c1))
    qx, qy = data["qx_float"][crop], data["qy_float"][crop]
    q_finite = bool(
        np.all(np.isfinite(qx))
        and np.all(np.isfinite(qy))
        and not np.any(data["qx_mask"][crop])
        and not np.any(data["qy_mask"][crop])
    )
    result = dict(data)
    result.update(
        {
            "display_valid": valid,
            "display": np.where(valid[crop], values[crop], np.nan),
            "display_low": low,
            "display_high": high,
            "display_label": label,
            "display_crop": {
                "row_start": r0,
                "row_stop": r1,
                "col_start": c0,
                "col_stop": c1,
                "shape": [r1 - r0, c1 - c0],
                "coordinates_finite": q_finite,
            },
            "display_render_primitive": (
                "pcolormesh"
                if q_finite and min(r1 - r0, c1 - c0) >= 2
                else "valid_q_sample_scatter"
            ),
        }
    )
    return result


def _model_comparison_figure(
    data: Mapping[str, Any],
    model_raw: np.ndarray,
    model_float: np.ndarray,
    common_valid: np.ndarray,
) -> tuple[Figure, dict[str, Any]]:
    from .figure_support import _draw_map

    measured = data["observed_float"]
    transform = data["display_scale"]
    measured_selected = _display_transform(measured[common_valid], transform)
    model_selected = _display_transform(model_float[common_valid], transform)
    combined = np.concatenate((measured_selected, model_selected))
    low, high = (float(value) for value in np.percentile(combined, [0.5, 99.5]))
    if not math.isfinite(low) or not math.isfinite(high):
        raise ValueError("could not determine shared measured/model color limits")
    if high <= low:
        delta = max(abs(high) * 0.01, 0.5)
        low, high = low - delta, high + delta
    residual = np.full(measured.shape, np.nan, dtype=np.float64)
    residual[common_valid] = measured[common_valid] - model_float[common_valid]
    residual_selected = residual[common_valid]
    residual_limit = float(np.percentile(np.abs(residual_selected), 99.5))
    if not math.isfinite(residual_limit) or residual_limit <= 0.0:
        residual_limit = max(float(np.max(np.abs(residual_selected))), 1.0)
    selected_display = common_valid & data["effective_valid"]
    measured_display = _display_transform(
        np.where(selected_display, measured, np.nan), transform
    )
    model_display = _display_transform(
        np.where(selected_display, model_float, np.nan), transform
    )
    residual_display = np.where(selected_display, residual, np.nan)
    measured_data = _crop_data(
        data,
        selected_display,
        measured_display,
        low=low,
        high=high,
        label=data["display_label"],
    )
    model_data = _crop_data(
        data,
        selected_display,
        model_display,
        low=low,
        high=high,
        label=data["display_label"],
    )
    residual_data = _crop_data(
        data,
        selected_display,
        residual_display,
        low=-residual_limit,
        high=residual_limit,
        label="Observed − model (raw intensity)",
    )
    width_mm = data["width_mm"]
    context = data.get("context")
    raw_model_status = (
        context.get("pixel_model_status") if isinstance(context, Mapping) else None
    )
    model_status = str(raw_model_status or "unknown")
    normalized_status = model_status.strip().lower().replace("-", "_")
    if normalized_status in {"failed", "fail", "error"}:
        model_title = "Supplied model · failed"
    elif normalized_status in {"unconverged", "not_converged", "notconverged"}:
        model_title = "Supplied model · unconverged"
    elif normalized_status in {"success", "succeeded", "ok", "converged"}:
        model_title = "Supplied model · solver success"
    else:
        model_title = "Supplied model · status unknown"
    height_mm = 150.0 if width_mm == 89.0 else 100.0
    fig = Figure(
        figsize=(width_mm / 25.4, height_mm / 25.4),
        dpi=data["dpi"],
        facecolor="white",
        edgecolor="white",
    )
    FigureCanvasAgg(fig)
    if width_mm == 183.0:
        axes = [fig.add_axes([x, 0.25, 0.24, 0.62]) for x in (0.11, 0.42, 0.73)]
        shared_cbar = fig.add_axes([0.11, 0.115, 0.55, 0.022])
        residual_cbar = fig.add_axes([0.73, 0.115, 0.24, 0.022])
        cbars = [shared_cbar, shared_cbar, residual_cbar]
        panel_positions = ((0.11, 0.93), (0.42, 0.93), (0.73, 0.93))
    else:
        axes = [fig.add_axes([0.13, y, 0.72, 0.20]) for y in (0.70, 0.405, 0.11)]
        cbars = [
            fig.add_axes([0.89, y + 0.02, 0.025, 0.16]) for y in (0.70, 0.405, 0.11)
        ]
        panel_positions = ((0.13, 0.94), (0.13, 0.645), (0.13, 0.35))
    cbar_orientation = "horizontal" if width_mm == 183.0 else "vertical"
    _draw_map(
        fig,
        axes[0],
        cbars[0],
        measured_data,
        draw_overlays=False,
        colorbar_orientation=cbar_orientation,
    )
    _draw_map(
        fig,
        axes[1],
        cbars[1],
        model_data,
        draw_overlays=False,
        colorbar_orientation=cbar_orientation,
    )
    _draw_map(
        fig,
        axes[2],
        cbars[2],
        residual_data,
        draw_overlays=False,
        cmap="coolwarm",
        colorbar_orientation=cbar_orientation,
    )
    if width_mm == 183.0:
        axes[1].set_ylabel("")
        axes[2].set_ylabel("")
    titles = ("Observed", model_title, "Observed − model (raw)")
    for ax, title in zip(axes, titles, strict=True):
        ax.set_title(title, fontsize=6.2, pad=2.0)
    metadata = {
        "model_supplied": True,
        "model_source": "caller-supplied array; no model inferred from ellipse geometry",
        "pixel_model_status": model_status,
        "visible_model_status_label": model_title,
        "scientific_acceptance_inferred": False,
        "residual_definition": "observed - model in original intensity units",
        "shared_display_transform": transform,
        "shared_measured_model_color_limits_transformed": [low, high],
        "residual_color_limits_raw_symmetric": [-residual_limit, residual_limit],
        "common_valid_pixel_count": int(np.count_nonzero(common_valid)),
        "model_mask_applied": True,
        "map_coordinates": "supplied qx/qy; no affine reconstruction",
    }
    for (x, y), panel in zip(panel_positions, ("a", "b", "c"), strict=True):
        fig.text(
            x - 0.018, y, panel, ha="left", va="top", fontsize=8.0, fontweight="bold"
        )
    return fig, metadata


def _save_figure(
    fig: Figure, directory: Path, stem: str, dpi: int, cancel_event: Any
) -> dict[str, Path]:
    result = {}
    for kind, suffix in (
        ("svg", "svg"),
        ("pdf", "pdf"),
        ("tiff", "tiff"),
        ("png", "png"),
    ):
        raise_if_cancelled(cancel_event, f"butterfly-comparison:save-{kind}")
        path = directory / f"{stem}.{suffix}"
        if kind in {"svg", "pdf"}:
            fig.savefig(path, format=kind, facecolor="white", transparent=False)
        else:
            fig.savefig(
                path, format=kind, dpi=dpi, facecolor="white", transparent=False
            )
        result[kind] = path
    return result


def export_butterfly_comparison(
    directory: str | Path,
    *,
    data: Mapping[str, Any],
    model: Any = None,
    cancel_event: Any = None,
) -> tuple[dict[str, Path], dict[str, Any]]:
    """Write comparison artifacts into an existing private staging directory.

    The caller owns the atomic directory transaction and final SHA-256 manifest.
    """

    from .figure_support import _EXPORT_LOCK, _FIGURE_RC, _numeric_array

    stage = Path(directory)
    raise_if_cancelled(cancel_event, "butterfly-comparison:prepare")
    result = data["result_safe"]
    if not isinstance(result, Mapping):
        result = {}
    candidate = _candidate(result)
    fit_state, curves, fit_metadata = _candidate_curves(result)
    point_diagnostics = _candidate_point_diagnostics(result, candidate)
    point_residual_path = stage / "point_residuals.csv"
    residual_rows, residual_field = _write_point_residuals(
        point_residual_path, data["points"], point_diagnostics
    )
    profiles, selected_profile, selected_profile_data = _profile_export_rows(result)
    profile_paths: dict[str, Path] = {}
    if profiles:
        profile_path = stage / "normal_profiles.csv"
        _write_profiles(profile_path, profiles)
        profile_paths["normal_profiles_csv"] = profile_path

    outputs = _write_curve_sidecars(stage, curves, fit_state)
    outputs["point_residuals_csv"] = point_residual_path
    outputs.update(profile_paths)
    comparison_caption = stage / "comparison_caption.txt"
    outputs["comparison_caption"] = comparison_caption

    comparison_metadata: dict[str, Any] = {
        "fit_state": fit_state,
        "fit_state_metadata": fit_metadata,
        "candidate_ellipse_curves": len(curves),
        "curve_role": "diagnostic candidate geometry; not accepted or observed support",
        "candidate_ellipse_scientific_acceptance_inferred": False,
        "point_residual_field_available": residual_field,
        "point_residual_field_plotted": None,
        "point_residual_input_count": len(residual_rows),
        "point_residual_finite_count": sum(
            _finite(row.get(residual_field)) is not None for row in residual_rows
        )
        if residual_field
        else 0,
        "normal_profile_count": len(profiles),
        "normal_profile_selection": selected_profile,
        "residual_panel_source": residual_field
        or ("normal_profile_raw_fit_residual" if selected_profile else "unavailable"),
        "ellipse_curve_sidecars": "ellipse_curves.csv and ellipse_curves.npz; empty arrays when no drawable candidate fit",
        "profile_sidecar": "normal_profiles.csv" if profiles else None,
        "point_residual_sidecar": "point_residuals.csv",
        "model_comparison_supplied": model is not None,
        "standalone_ellipse_exported": bool(curves),
    }
    with _EXPORT_LOCK, mpl.rc_context(_FIGURE_RC):
        figure, diagnostic_selection = _comparison_figure(
            data,
            curves=curves,
            fit_state=fit_state,
            residual_rows=residual_rows,
            residual_field=residual_field,
            selected_profile_data=selected_profile_data,
        )
        figure.canvas.draw()
        raise_if_cancelled(cancel_event, "butterfly-comparison:render")
        comparison_metadata["residual_panel_source"] = diagnostic_selection
        comparison_metadata["point_residual_field_plotted"] = (
            residual_field
            if diagnostic_selection.startswith("Point residual source:")
            else None
        )
        saved = _save_figure(
            figure, stage, "butterfly_comparison", data["dpi"], cancel_event
        )
        figure.clear()
    if curves:
        candidate_description = (
            "Finite ring-only/bound-limited solver geometry is shown as a labeled dashed candidate; it is not quantitative."
            if fit_state == "ring_only_candidate"
            else "Finite geometry from an unsuccessful solver is shown as a dashed candidate for diagnosis."
            if fit_state == "candidate_unconverged"
            else "Finite solver geometry is shown as a dashed candidate diagnostic."
        )
        panel_c_description = (
            "source-supported measured points/arcs with the dashed candidate curves."
        )
    else:
        candidate_description = "No candidate curves are available because fitted geometry is missing, non-finite, or unphysical."
        panel_c_description = (
            "source-supported measured points/arcs; no candidate curves are drawn."
        )
    comparison_caption.write_text(
        "Butterfly fit comparison. (a) Observed intensity. (b) "
        + candidate_description
        + " Any full dashed ellipse curve is candidate geometry, not observed support, and is not scientifically accepted. "
        + "(c) Observed intensity with "
        + panel_c_description
        + " (d) "
        + diagnostic_selection
        + ". No intensity map is inferred from ellipse geometry.\n",
        encoding="utf-8",
    )
    outputs.update({f"comparison_{key}": path for key, path in saved.items()})

    if (
        fit_state in {"candidate", "candidate_unconverged", "ring_only_candidate"}
        and curves
    ):
        with _EXPORT_LOCK, mpl.rc_context(_FIGURE_RC):
            ellipse_figure = _ellipse_only_figure(data, curves, fit_state)
            ellipse_figure.canvas.draw()
            raise_if_cancelled(cancel_event, "butterfly-comparison:ellipse-only-render")
            ellipse_saved = _save_figure(
                ellipse_figure, stage, "ellipse_only", data["dpi"], cancel_event
            )
            ellipse_figure.clear()
        outputs.update(
            {f"ellipse_only_{key}": path for key, path in ellipse_saved.items()}
        )
        ellipse_caption = stage / "ellipse_only_caption.txt"
        if fit_state == "ring_only_candidate":
            ellipse_description = "Ring-only/bound-limited candidate geometry; curves are diagnostic and not quantitative."
        elif fit_state == "candidate_unconverged":
            ellipse_description = "The solver did not report success; finite candidate curves are shown for diagnostic inspection only."
        else:
            ellipse_description = (
                "Candidate fit curves are shown for diagnostic inspection only."
            )
        ellipse_caption.write_text(
            "Standalone full candidate ellipse curves generated from the candidate_fit geometry. Dashed curves are not measured arc support "
            "and are not scientifically accepted quantities. "
            + ellipse_description
            + "\n",
            encoding="utf-8",
        )
        outputs["ellipse_only_caption"] = ellipse_caption

    if model is not None:
        raise_if_cancelled(cancel_event, "butterfly-comparison:model")
        model_raw, model_float, common_valid = _model_arrays(model, data)
        with _EXPORT_LOCK, mpl.rc_context(_FIGURE_RC):
            model_figure, model_metadata = _model_comparison_figure(
                data, model_raw, model_float, common_valid
            )
            model_figure.canvas.draw()
            raise_if_cancelled(cancel_event, "butterfly-comparison:model-save")
            model_saved = _save_figure(
                model_figure, stage, "model_comparison", data["dpi"], cancel_event
            )
            model_figure.clear()
        outputs.update(
            {f"model_comparison_{key}": path for key, path in model_saved.items()}
        )
        residual = np.full(data["observed_float"].shape, np.nan, dtype=np.float64)
        residual[common_valid] = (
            data["observed_float"][common_valid] - model_float[common_valid]
        )
        data_path = stage / "model_comparison_data.npz"
        np.savez_compressed(
            data_path,
            model=model_raw,
            observed_minus_model_raw=residual,
            common_valid_mask=common_valid,
            model_mask=np.asarray(_numeric_array(model, "model")[1], dtype=bool),
            q_unit=np.asarray(data["q_unit"]),
            shared_color_limits_transformed=np.asarray(
                model_metadata["shared_measured_model_color_limits_transformed"],
                dtype=np.float64,
            ),
            residual_color_limits_raw=np.asarray(
                model_metadata["residual_color_limits_raw_symmetric"], dtype=np.float64
            ),
        )
        model_caption = stage / "model_comparison_caption.txt"
        model_caption.write_text(
            f"Model comparison. Pixel model status from caller context: {model_metadata['pixel_model_status']}. "
            "The supplied measured image and full-2D model share the recorded display transform and color limits. "
            "The signed residual is observed intensity minus the caller-supplied model in original intensity units, with symmetric zero-centered limits. "
            "Only common valid q/image/model samples are shown. Solver status and this visual comparison do not imply scientific acceptance; "
            "no model is inferred from ellipse geometry.\n",
            encoding="utf-8",
        )
        outputs["model_comparison_data"] = data_path
        outputs["model_comparison_caption"] = model_caption
        comparison_metadata["model_comparison"] = model_metadata
    return outputs, comparison_metadata


__all__ = ["export_butterfly_comparison"]
