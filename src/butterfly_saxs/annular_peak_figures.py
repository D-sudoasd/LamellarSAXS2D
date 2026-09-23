"""Publication assets for annulus-wise angular SAXS peak trajectories.

The annular measurement is performed upstream.  This module is intentionally
only a renderer and evidence writer: it consumes ``result['annular_peaks']``
and never re-integrates pixels, chooses a different peak, bridges a missing
ring, or fits an ellipse.  A point's ``q`` coordinate is the centre of its
sampled q annulus.  It is therefore not a radial ``q*`` and is never converted
to ``2*pi/q`` here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import numpy as np

from .cancellation import raise_if_cancelled
from .csv_utils import safe_csv_cell
from .figure_support import _EXPORT_LOCK, _FIGURE_RC
from .serialization import json_safe


ANNULAR_FIGURE_METHOD_VERSION = "annular-peak-figures-v1"
ANNULAR_FIGURE_SCHEMA_VERSION = "annular-peak-figure-export-v1"

_COLORS = ("#0072B2", "#D55E00", "#009E73")
_TARGET_Q = (0.25, 0.40, 0.60)


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if np.isfinite(number) else None


def _number(value: Any, default: float = float("nan")) -> float:
    number = _finite(value)
    return default if number is None else number


def _csv_number(value: Any) -> str:
    number = _finite(value)
    return "" if number is None else format(number, ".17g")


def _array(value: Any, *, dtype: Any = float) -> np.ndarray:
    """Read a result array without changing the supplied result."""

    if value is None:
        return np.asarray([], dtype=dtype)
    try:
        values = value.tolist() if isinstance(value, np.ndarray) else value
        if dtype is float:
            return np.asarray(
                [np.nan if item is None else float(item) for item in values],
                dtype=np.float64,
            )
        return np.asarray(values, dtype=dtype)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("annular result arrays must be one-dimensional numeric values") from exc


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _payload(result: Mapping[str, Any]) -> Mapping[str, Any]:
    """Accept either the annular mapping or a full page result."""

    if not isinstance(result, Mapping):
        raise TypeError("result must be a mapping")
    nested = result.get("annular_peaks")
    if isinstance(nested, Mapping):
        return nested
    return result


def _angle_delta(left: float, right: float) -> float:
    return abs((float(left) - float(right) + 180.0) % 360.0 - 180.0)


def _normalise(result: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Normalize the already-measured annuli for rendering and tabulation."""

    payload = _payload(result)
    q_unit = str(payload.get("q_unit", "unknown") or "unknown")
    q_edges = _array(payload.get("q_edges"), dtype=float)
    angle_centers = _array(payload.get("angle_centers_deg"), dtype=float)
    raw_annuli = payload.get("annuli", ())
    if raw_annuli is None:
        raw_annuli = ()
    if isinstance(raw_annuli, (str, bytes)) or not isinstance(raw_annuli, Sequence):
        raise TypeError("annular_peaks.annuli must be a sequence")
    if angle_centers.size and not np.all(np.isfinite(angle_centers)):
        raise ValueError("angle_centers_deg must contain finite values")

    rows: list[dict[str, Any]] = []
    for fallback_index, raw in enumerate(raw_annuli):
        annulus = dict(_mapping(raw, f"annular_peaks.annuli[{fallback_index}]"))
        index_value = annulus.get("annulus_index", fallback_index)
        try:
            annulus_index = int(index_value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("annulus_index must be an integer") from exc
        angles = _array(annulus.get("angle_centers_deg"), dtype=float)
        if angles.size == 0:
            angles = angle_centers.copy()
        if angles.size and not np.all(np.isfinite(angles)):
            raise ValueError(f"annulus {annulus_index} angle centers must be finite")
        raw_mapping = annulus.get("raw")
        raw_mapping = raw_mapping if isinstance(raw_mapping, Mapping) else {}
        arrays = {
            "raw_mean": _array(annulus.get("raw_mean", raw_mapping.get("mean"))),
            "raw_sum": _array(annulus.get("raw_sum", raw_mapping.get("sum"))),
            "counts": _array(annulus.get("counts", raw_mapping.get("count")), dtype=np.int64),
            "geometry_counts": _array(
                annulus.get("geometry_counts", annulus.get("geometric_counts")),
                dtype=np.int64,
            ),
            "coverage": _array(annulus.get("coverage")),
            "smoothed_intensity": _array(annulus.get("smoothed_intensity")),
        }
        lengths = {key: len(value) for key, value in arrays.items() if len(value)}
        if angles.size == 0 and lengths:
            raise ValueError(f"annulus {annulus_index} needs angle_centers_deg")
        if lengths and len(set(lengths.values())) != 1:
            raise ValueError(f"annulus {annulus_index} arrays have inconsistent lengths: {lengths}")
        length = len(angles)
        if lengths and next(iter(lengths.values())) != length:
            raise ValueError(f"annulus {annulus_index} arrays must match angle_centers_deg")
        for key, values in tuple(arrays.items()):
            if len(values) == 0 and length:
                fill: Any = np.zeros(length, dtype=np.int64) if key in {"counts", "geometry_counts"} else np.full(length, np.nan)
                arrays[key] = fill
        selected = annulus.get("selected_peaks", ())
        if selected is None:
            selected = ()
        if isinstance(selected, (str, bytes)) or not isinstance(selected, Sequence):
            raise TypeError(f"annulus {annulus_index} selected_peaks must be a sequence")
        selected_rows = [dict(_mapping(item, f"annulus {annulus_index} selected peak")) for item in selected]
        if len(selected_rows) > 4:
            raise ValueError(f"annulus {annulus_index} contains more than four selected peaks")
        candidates = annulus.get("candidates", ())
        if candidates is None:
            candidates = ()
        if isinstance(candidates, (str, bytes)) or not isinstance(candidates, Sequence):
            raise TypeError(f"annulus {annulus_index} candidates must be a sequence")
        candidate_rows = [dict(_mapping(item, f"annulus {annulus_index} candidate")) for item in candidates]
        annulus.update(
            {
                "annulus_index": annulus_index,
                "q_center": _number(annulus.get("q_center")),
                "q_min": _number(annulus.get("q_min")),
                "q_max": _number(annulus.get("q_max")),
                "profile_id": str(annulus.get("profile_id", f"annulus-{annulus_index}")),
                "angle_centers_deg": angles,
                **arrays,
                "selected_peaks": selected_rows,
                "candidates": candidate_rows,
                "status": str(annulus.get("status", "unknown")),
                "reason": str(annulus.get("reason") or ""),
            }
        )
        rows.append(annulus)
    rows.sort(key=lambda row: row["annulus_index"])
    if q_edges.size not in {0, len(rows) + 1}:
        raise ValueError("q_edges must have one more value than annuli")
    if q_edges.size and not np.all(np.isfinite(q_edges)):
        raise ValueError("q_edges must contain finite values")
    if angle_centers.size == 0 and rows:
        angle_centers = np.asarray(rows[0]["angle_centers_deg"], dtype=float)
    if angle_centers.size and any(len(row["angle_centers_deg"]) != len(angle_centers) for row in rows):
        raise ValueError("all annuli must use the same angle-centre grid")
    # The detector accumulator may use a conventional -180..180 grid while
    # refined point angles are reported as 0..360.  Canonicalize only the
    # display grid and carry every profile array along with it.  The source
    # arrays remain untouched for CSV/NPZ auditability; ``_display_*`` fields
    # are internal render-only views.
    display_angles = np.mod(np.asarray(angle_centers, dtype=float), 360.0)
    angle_order = np.argsort(display_angles, kind="stable") if display_angles.size else np.asarray([], dtype=int)
    display_angles = display_angles[angle_order]
    for row in rows:
        source_angles = np.asarray(row["angle_centers_deg"], dtype=float)
        source_display_angles = np.mod(source_angles, 360.0)
        source_order = np.argsort(source_display_angles, kind="stable") if source_display_angles.size else np.asarray([], dtype=int)
        for key in ("raw_mean", "raw_sum", "counts", "geometry_counts", "coverage", "smoothed_intensity"):
            values = np.asarray(row[key])
            if len(values) and len(source_order) == len(values):
                row[f"_display_{key}"] = values[source_order]
            else:
                row[f"_display_{key}"] = values.copy()
        row["_source_angle_centers_deg"] = source_angles.copy()
        row["_display_angle_centers_deg"] = source_display_angles[source_order]
        display_indices: list[int | None] = []
        for peak in row["selected_peaks"]:
            source_index = peak.get("angular_bin_index")
            try:
                source_index_int = int(source_index)
            except (TypeError, ValueError, OverflowError):
                source_index_int = -1
            if 0 <= source_index_int < len(source_order):
                display_index = int(np.flatnonzero(source_order == source_index_int)[0])
            else:
                chi = _finite(peak.get("chi_deg"))
                if chi is None or not len(row["_display_angle_centers_deg"]):
                    display_index = None
                else:
                    display_index = int(np.argmin([_angle_delta(value, chi) for value in row["_display_angle_centers_deg"]]))
            display_indices.append(display_index)
        row["_selected_display_indices"] = display_indices
    payload_norm = {
        "q_unit": q_unit,
        "q_edges": q_edges,
        "angle_centers_deg": np.asarray(angle_centers, dtype=float),
        "display_angle_centers_deg": display_angles,
        "settings": payload.get("settings") if isinstance(payload.get("settings"), Mapping) else {},
        "method_version": payload.get("method_version"),
        "q_window": payload.get("q_window"),
        "sector_overlap": payload.get("sector_overlap"),
    }
    return payload_norm, rows


def _intensity_unit(data: Mapping[str, Any]) -> tuple[str, str]:
    context = data.get("context")
    if isinstance(context, Mapping):
        metadata = context.get("metadata")
        if isinstance(metadata, Mapping):
            header = metadata.get("header")
            if isinstance(header, Mapping):
                value = header.get("IntensityUnit")
                if value is not None and str(value).strip():
                    return str(value).strip(), "context.metadata.header.IntensityUnit"
    return "input intensity units", "fallback: caller-supplied intensity scale; no unit inferred"


def _q_bounds(payload: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> tuple[float | None, float | None]:
    edges = np.asarray(payload.get("q_edges", ()), dtype=float)
    finite = edges[np.isfinite(edges)]
    if finite.size:
        return float(np.min(finite)), float(np.max(finite))
    centers = np.asarray([_number(row.get("q_center")) for row in rows], dtype=float)
    centers = centers[np.isfinite(centers)]
    if centers.size:
        return float(np.min(centers)), float(np.max(centers))
    return None, None


def _metadata(data: Mapping[str, Any], payload: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    intensity_unit, intensity_source = _intensity_unit(data)
    q_min, q_max = _q_bounds(payload, rows)
    settings = payload.get("settings") if isinstance(payload.get("settings"), Mapping) else {}
    overlap = payload.get("sector_overlap") if isinstance(payload.get("sector_overlap"), Mapping) else {}
    selected_count = sum(len(row["selected_peaks"]) for row in rows)
    return {
        "schema_version": ANNULAR_FIGURE_SCHEMA_VERSION,
        "method_version": ANNULAR_FIGURE_METHOD_VERSION,
        "source_measurement_method_version": payload.get("method_version"),
        "q_unit": str(payload.get("q_unit", "unknown")),
        "intensity_unit": intensity_unit,
        "intensity_unit_source": intensity_source,
        "q_range": [q_min, q_max] if q_min is not None and q_max is not None else None,
        "settings": json_safe(settings),
        "sector_overlap": json_safe(overlap),
        "overlap_correlation_warning": str(overlap.get("correlation_statement", "overlapping q annuli may share detector pixels and are correlated; they are not independent samples")),
        "annulus_count": len(rows),
        "angle_bin_count": int(len(payload.get("angle_centers_deg", ()))),
        "selected_peak_count": selected_count,
        "max_selected_peaks_per_annulus": 4,
        "q_coordinate_is_annulus_center": True,
        "radial_q_star_not_used": True,
        "spacing_conversion_not_performed": True,
        "quadrants_are_not_synthesized": True,
        "missing_bins_are_left_blank": True,
        "smooth_is_locator_only": True,
        "no_ellipse_fitted_or_drawn": True,
        "candidate_selection_repeated_at_export": False,
    }


def _q_edges_for_rows(payload: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    provided = np.asarray(payload.get("q_edges", ()), dtype=float)
    if len(provided) == len(rows) + 1 and np.all(np.isfinite(provided)):
        return provided.copy()
    centers = np.asarray([_number(row.get("q_center")) for row in rows], dtype=float)
    if not len(rows):
        return np.asarray([0.0, 1.0])
    edges = np.full(len(rows) + 1, np.nan, dtype=float)
    for index, row in enumerate(rows):
        left = _finite(row.get("q_min"))
        right = _finite(row.get("q_max"))
        if left is not None:
            edges[index] = left
        if right is not None:
            edges[index + 1] = right
    finite_centers = np.isfinite(centers)
    if np.any(finite_centers):
        for index in range(len(edges)):
            if np.isfinite(edges[index]):
                continue
            if index == 0:
                edges[index] = centers[finite_centers][0]
            elif index == len(edges) - 1:
                edges[index] = centers[finite_centers][-1]
            else:
                edges[index] = np.nanmean(centers[max(0, index - 1): min(len(centers), index + 1)])
    else:
        edges = np.arange(len(rows) + 1, dtype=float)
    return edges


def _angle_edges(centers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if centers.size == 0:
        return np.asarray([0.0, 360.0]), np.asarray([], dtype=int)
    order = np.argsort(centers)
    sorted_centers = centers[order]
    if sorted_centers.size == 1:
        half = 1.0
        return np.asarray([sorted_centers[0] - half, sorted_centers[0] + half]), order
    gaps = np.diff(sorted_centers)
    first_gap = float(gaps[0])
    last_gap = float(gaps[-1])
    edges = np.empty(sorted_centers.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (sorted_centers[:-1] + sorted_centers[1:])
    edges[0] = sorted_centers[0] - 0.5 * first_gap
    edges[-1] = sorted_centers[-1] + 0.5 * last_gap
    return edges, order


def _selected_intensity(peak: Mapping[str, Any]) -> float | None:
    for key in ("raw_intensity", "intensity", "peak_intensity"):
        value = _finite(peak.get(key))
        if value is not None:
            return value
    return None


def _build_qchi(payload: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], metadata: Mapping[str, Any], *, width_mm: float, dpi: int) -> Figure:
    height_mm = 108.0 if width_mm == 183.0 else 132.0
    fig = Figure(figsize=(width_mm / 25.4, height_mm / 25.4), dpi=dpi, facecolor="white", edgecolor="white")
    FigureCanvasAgg(fig)
    if width_mm == 183.0:
        ax = fig.add_axes([0.10, 0.20, 0.77, 0.64])
        cax = fig.add_axes([0.90, 0.27, 0.018, 0.48])
        footer_y = 0.055
    else:
        ax = fig.add_axes([0.17, 0.25, 0.67, 0.56])
        cax = fig.add_axes([0.88, 0.34, 0.027, 0.38])
        footer_y = 0.045
    if not rows:
        ax.text(0.5, 0.5, "No annular profiles supplied", transform=ax.transAxes, ha="center", va="center", fontsize=5.5, color="#555555")
        ax.set_axis_off()
        fig.suptitle("Raw annular I(q, χ) · no data", y=0.96, fontsize=7.0)
        return fig
    n_angles = max(len(row["_display_angle_centers_deg"]) for row in rows)
    if n_angles == 0:
        ax.text(0.5, 0.5, "No angular bins supplied", transform=ax.transAxes, ha="center", va="center", fontsize=5.5, color="#555555")
        ax.set_axis_off()
        return fig
    matrix = np.full((len(rows), n_angles), np.nan, dtype=float)
    for row_index, row in enumerate(rows):
        values = np.asarray(row["_display_raw_mean"], dtype=float)
        matrix[row_index, : len(values)] = values
    q_edges = _q_edges_for_rows(payload, rows)
    angles = np.asarray(payload.get("display_angle_centers_deg", rows[0]["_display_angle_centers_deg"]), dtype=float)
    angle_edges, angle_order = _angle_edges(angles)
    matrix = matrix[:, angle_order].T
    masked = np.ma.masked_invalid(matrix)
    finite = masked.compressed()
    if finite.size:
        low, high = np.nanpercentile(finite, [0.5, 99.5])
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            low, high = float(np.nanmin(finite)), float(np.nanmax(finite))
            if high <= low:
                low, high = low - 0.5, high + 0.5
        mesh = ax.pcolormesh(q_edges, angle_edges, masked, shading="auto", cmap="cividis", vmin=float(low), vmax=float(high), rasterized=True)
        cbar = fig.colorbar(mesh, cax=cax)
        cbar.set_label(f"raw mean I ({metadata['intensity_unit']})", fontsize=5.5, labelpad=2.0)
        cbar.ax.tick_params(labelsize=5.0, length=1.2, width=0.4, pad=1.0)
    for row in rows:
        q_center = _finite(row.get("q_center"))
        if q_center is None:
            continue
        for peak in row["selected_peaks"]:
            chi = _finite(peak.get("chi_deg"))
            if chi is not None:
                ax.scatter([q_center], [float(np.mod(chi, 360.0))], s=15, marker="o", facecolors="none", edgecolors="#D55E00", linewidths=0.75, zorder=5)
    ax.set_xlabel(f"q annulus centre ({metadata['q_unit']})")
    ax.set_ylabel("Azimuth χ (degree)")
    ax.set_title("Raw annular intensity I(q, χ) + observed angular maxima", fontsize=7.0)
    ax.tick_params(direction="out", length=2.0, width=0.5, pad=1.5)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    marker = Line2D([0], [0], marker="o", markerfacecolor="none", markeredgecolor="#D55E00", linestyle="None", markersize=3.0, label="observed angular maximum (≤4 per q annulus)")
    # Keep the marker key below the title so it does not collide with the q
    # xlabel or the two-line scientific boundary footer on the narrow panel.
    fig.legend(handles=[marker], loc="upper center", bbox_to_anchor=(0.5, 0.925), frameon=False, fontsize=5.0, handletextpad=0.35)
    footer = "Blank cells = missing measured angular support; no missing quadrant is synthesized.\nq is a sampled annulus coordinate; no 2π/q conversion or ellipse fit is applied."
    fig.text(0.5, footer_y, footer, ha="center", va="center", fontsize=5.0, linespacing=1.25, color="#555555")
    return fig


def _representative_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[list[Mapping[str, Any]], list[dict[str, Any]]]:
    finite_rows = [row for row in rows if _finite(row.get("q_center")) is not None]
    if not finite_rows:
        return [], []
    q_values = np.asarray([float(row["q_center"]) for row in finite_rows], dtype=float)
    q_low, q_high = float(np.min(q_values)), float(np.max(q_values))
    selected: list[Mapping[str, Any]] = []
    info: list[dict[str, Any]] = []
    used: set[int] = set()
    for index, target in enumerate(_TARGET_Q):
        in_range = q_low <= target <= q_high
        effective = target if in_range else float(np.quantile(q_values, (0.2, 0.5, 0.8)[index]))
        order = sorted(range(len(finite_rows)), key=lambda item: abs(q_values[item] - effective))
        choice = next((item for item in order if item not in used), order[0])
        used.add(choice)
        selected.append(finite_rows[choice])
        info.append({"requested_q": target, "effective_q": effective, "annulus_index": int(finite_rows[choice]["annulus_index"]), "selection": "target_q" if in_range else "effective_q_quantile"})
    return selected, info


def _build_profiles(payload: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], metadata: Mapping[str, Any], *, width_mm: float, dpi: int) -> Figure:
    height_mm = 120.0 if width_mm == 183.0 else 145.0
    fig = Figure(figsize=(width_mm / 25.4, height_mm / 25.4), dpi=dpi, facecolor="white", edgecolor="white")
    FigureCanvasAgg(fig)
    reps, _ = _representative_rows(rows)
    fig.suptitle("Representative annular I(χ) · raw mean + locator-only smooth" if width_mm == 183.0 else "Representative annular I(χ)\nraw mean + locator-only smooth", x=0.5, y=0.965, fontsize=7.0, linespacing=1.05)
    legend_handles = [Line2D([0], [0], color="#555555", marker="o", markerfacecolor="none", linewidth=0.75, markersize=2.5, label="raw annular mean"), Line2D([0], [0], color="#555555", linestyle="--", linewidth=0.95, label="locator-only smooth")]
    fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 0.91), frameon=False, ncol=2, fontsize=5.0, handlelength=1.3, handletextpad=0.3, columnspacing=0.7)
    if not reps:
        ax = fig.add_axes([0.18, 0.24, 0.68, 0.52])
        ax.text(0.5, 0.5, "No finite annular q centres supplied", transform=ax.transAxes, ha="center", va="center", fontsize=5.5, color="#555555")
        ax.set_axis_off()
    else:
        left, right = (0.16, 0.89) if width_mm == 183.0 else (0.21, 0.86)
        bottom = 0.14
        top = 0.82
        gap = 0.055
        height = (top - bottom - gap * (len(reps) - 1)) / max(1, len(reps))
        for index, row in enumerate(reps):
            y = top - (index + 1) * height - index * gap
            ax = fig.add_axes([left, y, right - left, height])
            angles = np.asarray(row["_display_angle_centers_deg"], dtype=float)
            raw = np.asarray(row["_display_raw_mean"], dtype=float)
            smooth = np.asarray(row["_display_smoothed_intensity"], dtype=float)
            color = _COLORS[index % len(_COLORS)]
            finite_raw = np.isfinite(raw)
            if finite_raw.any():
                # Keep NaN bins in the plotted arrays so Matplotlib breaks the
                # trace at an unsupported angular bin instead of connecting
                # measurements across a masked gap.
                ax.plot(angles, raw, color=color, linewidth=0.7, marker="o", markersize=1.4, markerfacecolor="none", label="raw annular mean")
            finite_smooth = np.isfinite(smooth)
            if finite_smooth.any():
                ax.plot(angles, smooth, color=color, linewidth=1.0, linestyle="--", label="locator-only smooth")
            peak_x: list[float] = []
            peak_y: list[float] = []
            for peak in row["selected_peaks"]:
                chi = _finite(peak.get("chi_deg"))
                intensity = _selected_intensity(peak)
                if intensity is None and chi is not None and angles.size:
                    # A few measurement payloads carry the selected angle but
                    # omit its duplicate intensity.  Read the supplied raw
                    # bin at that angle for display; this does not reselect a
                    # peak or alter the stored selected record.
                    nearest = int(np.argmin([_angle_delta(value, chi) for value in angles]))
                    if nearest < len(raw):
                        intensity = _finite(raw[nearest])
                if chi is not None and intensity is not None:
                    peak_x.append(float(np.mod(chi, 360.0)))
                    peak_y.append(intensity)
            if peak_x:
                ax.scatter(peak_x, peak_y, s=17, marker="o", facecolors="white", edgecolors=color, linewidths=0.9, zorder=6)
            q_min, q_max = _finite(row.get("q_min")), _finite(row.get("q_max"))
            q_center = _finite(row.get("q_center"))
            q_text = f"q annulus [{q_min:.4g}, {q_max:.4g}]" if q_min is not None and q_max is not None else f"q annulus centre {q_center:.4g}" if q_center is not None else "q annulus"
            ax.set_title(f"{q_text} · {len(row['selected_peaks'])} observed angular maxima", fontsize=5.4, loc="left", pad=1.5, color=color)
            ax.set_ylabel(f"I ({metadata['intensity_unit']})", fontsize=5.4)
            if index == len(reps) - 1:
                ax.set_xlabel("Azimuth χ (degree)", fontsize=5.4)
            else:
                ax.tick_params(labelbottom=False)
            # The display convention is a full azimuthal turn even when the
            # first/last bin centres sit half a bin inside 0 and 360 degrees.
            ax.set_xlim(0.0, 360.0 if angles.size else 1.0)
            ax.tick_params(direction="out", length=2.0, width=0.5, pad=1.3, labelsize=5.0)
            ax.grid(axis="y", color="#dddddd", linewidth=0.3)
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)
    footer = (
        "Raw I(χ) and locator-only smooth are copied from the result; markers = selected peaks.\n"
        "q is an annulus coordinate; no 2π/q conversion or gap filling."
        if width_mm == 89.0
        else "Raw I(χ) and locator-only smoothing are copied from the supplied annular result; markers are supplied selected peaks.\n"
        "q is an annulus coordinate, not radial q*; no missing peaks or quadrants are filled."
    )
    fig.text(0.5, 0.045, footer, ha="center", va="center", fontsize=5.0, linespacing=1.25, color="#555555")
    return fig


def _selected_for_angle(row: Mapping[str, Any], angle: float) -> list[Mapping[str, Any]]:
    centers = np.asarray(row["_display_angle_centers_deg"], dtype=float)
    if centers.size == 0:
        return []
    target_index = int(np.argmin([_angle_delta(value, angle) for value in centers]))
    peaks = []
    display_indices = row.get("_selected_display_indices", ())
    for peak_index, peak in enumerate(row["selected_peaks"]):
        if peak_index < len(display_indices) and display_indices[peak_index] is not None:
            if int(display_indices[peak_index]) == target_index:
                peaks.append(peak)
            continue
        chi = _finite(peak.get("chi_deg"))
        if chi is not None:
            nearest_index = int(np.argmin([_angle_delta(value, chi) for value in centers]))
            if nearest_index == target_index:
                peaks.append(peak)
    return peaks


def _selected_for_source_angle(row: Mapping[str, Any], angle: float) -> list[Mapping[str, Any]]:
    """Associate supplied peaks with a source bin for the audit CSV."""

    centers = np.asarray(row["_source_angle_centers_deg"], dtype=float)
    if centers.size == 0:
        return []
    target_index = int(np.argmin([_angle_delta(value, angle) for value in centers]))
    peaks = []
    for peak in row["selected_peaks"]:
        source_index = peak.get("angular_bin_index")
        try:
            source_index_int = int(source_index)
        except (TypeError, ValueError, OverflowError):
            source_index_int = -1
        if 0 <= source_index_int < len(centers):
            if source_index_int == target_index:
                peaks.append(peak)
            continue
        chi = _finite(peak.get("chi_deg"))
        if chi is not None:
            nearest_index = int(np.argmin([_angle_delta(value, chi) for value in centers]))
            if nearest_index == target_index:
                peaks.append(peak)
    return peaks


_PROFILE_FIELDS = (
    "record_type", "annulus_index", "profile_id", "q_unit", "intensity_unit", "q_min", "q_max", "q_center", "angle_index", "chi_deg", "raw_mean", "raw_sum", "counts", "geometry_counts", "coverage", "smoothed_intensity", "status", "reason", "selected", "selected_point_ids", "candidate_count", "candidate_json",
)


def _write_profiles_csv(path: Path, rows: Sequence[Mapping[str, Any]], metadata: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=_PROFILE_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            angles = np.asarray(row["_source_angle_centers_deg"], dtype=float)
            for index, angle in enumerate(angles):
                selected = _selected_for_source_angle(row, float(angle))
                values = {
                    "record_type": "profile", "annulus_index": row["annulus_index"], "profile_id": row["profile_id"], "q_unit": metadata["q_unit"], "intensity_unit": metadata["intensity_unit"], "q_min": _csv_number(row.get("q_min")), "q_max": _csv_number(row.get("q_max")), "q_center": _csv_number(row.get("q_center")), "angle_index": index, "chi_deg": _csv_number(angle),
                    "raw_mean": _csv_number(row["raw_mean"][index]), "raw_sum": _csv_number(row["raw_sum"][index]), "counts": int(row["counts"][index]) if len(row["counts"]) > index else "", "geometry_counts": int(row["geometry_counts"][index]) if len(row["geometry_counts"]) > index else "", "coverage": _csv_number(row["coverage"][index]), "smoothed_intensity": _csv_number(row["smoothed_intensity"][index]), "status": row["status"], "reason": row["reason"], "selected": bool(selected), "selected_point_ids": json.dumps([str(peak.get("point_id", "")) for peak in selected], ensure_ascii=False), "candidate_count": len(row["candidates"]), "candidate_json": json.dumps(json_safe(row["candidates"]), ensure_ascii=False, sort_keys=True, allow_nan=False),
                }
                writer.writerow({key: safe_csv_cell(value) for key, value in values.items()})


_PEAK_FIELDS = (
    "record_type", "annulus_index", "profile_id", "q_unit", "intensity_unit", "q_min", "q_max", "q_center", "point_id", "trajectory_id", "chi_deg", "qx", "qy", "accepted", "status", "reason", "raw_intensity", "intensity", "prominence", "snr", "candidate_count", "selected_count", "candidate_json",
)


def _write_peaks_csv(path: Path, rows: Sequence[Mapping[str, Any]], metadata: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=_PEAK_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            base = {"annulus_index": row["annulus_index"], "profile_id": row["profile_id"], "q_unit": metadata["q_unit"], "intensity_unit": metadata["intensity_unit"], "q_min": _csv_number(row.get("q_min")), "q_max": _csv_number(row.get("q_max")), "q_center": _csv_number(row.get("q_center")), "candidate_count": len(row["candidates"]), "selected_count": len(row["selected_peaks"])}
            for record_type, values in (("selected", row["selected_peaks"]), ("candidate", row["candidates"])):
                for peak in values:
                    output = {**base, "record_type": record_type, "point_id": peak.get("point_id", ""), "trajectory_id": peak.get("trajectory_id", ""), "chi_deg": _csv_number(peak.get("chi_deg")), "qx": _csv_number(peak.get("qx")), "qy": _csv_number(peak.get("qy")), "accepted": peak.get("accepted", ""), "status": peak.get("status", row["status"]), "reason": peak.get("reason", row["reason"]), "raw_intensity": _csv_number(peak.get("raw_intensity")), "intensity": _csv_number(peak.get("intensity")), "prominence": _csv_number(peak.get("prominence")), "snr": _csv_number(peak.get("snr")), "candidate_json": json.dumps(json_safe(peak), ensure_ascii=False, sort_keys=True, allow_nan=False)}
                    writer.writerow({key: safe_csv_cell(value) for key, value in output.items()})


def _write_npz(path: Path, payload: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], metadata: Mapping[str, Any]) -> None:
    n = len(rows)
    m = max((len(row["angle_centers_deg"]) for row in rows), default=0)
    def matrix(key: str, *, dtype: Any = float, fill: Any = np.nan) -> np.ndarray:
        output = np.full((n, m), fill, dtype=dtype)
        for index, row in enumerate(rows):
            values = np.asarray(row[key], dtype=dtype)
            output[index, : min(m, len(values))] = values[:m]
        return output
    np.savez_compressed(path, annulus_index=np.asarray([row["annulus_index"] for row in rows], dtype=np.int64), profile_id=np.asarray([str(row["profile_id"]) for row in rows], dtype="U128"), q_center=np.asarray([_number(row.get("q_center")) for row in rows], dtype=float), q_min=np.asarray([_number(row.get("q_min")) for row in rows], dtype=float), q_max=np.asarray([_number(row.get("q_max")) for row in rows], dtype=float), q_edges=np.asarray(payload.get("q_edges", ()), dtype=float), angle_centers_deg=np.asarray(payload.get("angle_centers_deg", ()), dtype=float), display_angle_centers_deg=np.asarray(payload.get("display_angle_centers_deg", ()), dtype=float), raw_mean=matrix("raw_mean"), raw_sum=matrix("raw_sum"), counts=matrix("counts", dtype=np.int64, fill=0), geometry_counts=matrix("geometry_counts", dtype=np.int64, fill=0), coverage=matrix("coverage"), smoothed_intensity=matrix("smoothed_intensity"), selected_peak_count=np.asarray([len(row["selected_peaks"]) for row in rows], dtype=np.int64), q_unit=np.asarray(str(metadata["q_unit"])), intensity_unit=np.asarray(str(metadata["intensity_unit"])), schema_version=np.asarray(ANNULAR_FIGURE_SCHEMA_VERSION))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(json_safe(payload), stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _caption(metadata: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], representative_info: Sequence[Mapping[str, Any]]) -> str:
    q_range = metadata.get("q_range")
    q_text = f"[{q_range[0]:.6g}, {q_range[1]:.6g}] {metadata['q_unit']}" if q_range else f"unknown range in {metadata['q_unit']}"
    selected = metadata["selected_peak_count"]
    rep_text = ", ".join(f"{item['effective_q']:.5g} ({item['selection']})" for item in representative_info) or "none"
    return (
        "Annular angular-peak trajectory evidence. Each raw profile is the supplied arithmetic-mean I(χ) in one sampled q annulus; "
        f"the q range is {q_text} and {len(rows)} annuli contribute {selected} supplied selected angular maxima. "
        "The q–χ panel retains finite measured means and leaves missing support blank. At most four selected peaks per q annulus are drawn; "
        "candidate lists and rejected annuli remain in the CSV/JSON evidence, and no missing peak or quadrant is synthesized. "
        "The representative profiles use target q values 0.25, 0.40 and 0.60 when in range, otherwise the nearest effective q quantiles "
        f"({rep_text}). Smoothing is locator-only and is copied from the supplied result. A peak q is the sampled annulus coordinate, not a radial q*; "
        "no 2π/q spacing conversion, ellipse fit, or scientific acceptance is performed. Counts and coverage are support statistics, and overlapping annuli may be correlated. "
        f"The source intensity scale is {metadata['intensity_unit']} ({metadata['intensity_unit_source']})."
    )


def render_annular_peak_figures(result: Mapping[str, Any], *, data: Mapping[str, Any] | None = None, width_mm: float = 183.0, dpi: int = 600) -> dict[str, Figure]:
    if not isinstance(result, Mapping):
        raise TypeError("result must be an annular_peaks mapping")
    width = float(width_mm)
    if width not in {89.0, 183.0}:
        raise ValueError("width_mm must be 89 or 183")
    if isinstance(dpi, bool) or int(dpi) != dpi or int(dpi) <= 0:
        raise ValueError("dpi must be a positive integer")
    payload, rows = _normalise(result)
    metadata = _metadata(data or {}, payload, rows)
    with _EXPORT_LOCK, mpl.rc_context(_FIGURE_RC):
        figures = {"annular_qchi": _build_qchi(payload, rows, metadata, width_mm=width, dpi=int(dpi)), "annular_profiles": _build_profiles(payload, rows, metadata, width_mm=width, dpi=int(dpi))}
        for figure in figures.values():
            figure.canvas.draw()
        return figures


def export_annular_peak_figures(stage: str | Path, *, data: Mapping[str, Any], result: Mapping[str, Any], cancel_event: Any = None) -> tuple[dict[str, Path], dict[str, Any]]:
    """Export annular assets into a caller-owned existing stage directory."""

    raise_if_cancelled(cancel_event, "annular-peak-figures:validate")
    stage_path = Path(stage).expanduser().resolve(strict=True)
    if not stage_path.is_dir():
        raise ValueError("stage must be an existing directory")
    if not isinstance(data, Mapping) or not isinstance(result, Mapping):
        raise TypeError("data and result must be mappings")
    width = float(data.get("width_mm", 183.0))
    dpi_raw = data.get("dpi", 600)
    if width not in {89.0, 183.0}:
        raise ValueError("data.width_mm must be 89 or 183")
    if isinstance(dpi_raw, bool) or int(dpi_raw) != dpi_raw or int(dpi_raw) <= 0:
        raise ValueError("data.dpi must be a positive integer")
    dpi = int(dpi_raw)
    payload, rows = _normalise(result)
    metadata = _metadata(data, payload, rows)
    _, representative_info = _representative_rows(rows)
    outputs = {key: stage_path / name for key, name in {"annular_qchi_svg": "annular_qchi.svg", "annular_qchi_pdf": "annular_qchi.pdf", "annular_qchi_png": "annular_qchi.png", "annular_qchi_tiff": "annular_qchi.tiff", "annular_profiles_svg": "annular_profiles.svg", "annular_profiles_pdf": "annular_profiles.pdf", "annular_profiles_png": "annular_profiles.png", "annular_profiles_tiff": "annular_profiles.tiff", "annular_profiles_csv": "annular_profiles.csv", "annular_peaks_csv": "annular_peaks.csv", "annular_profiles_npz": "annular_profiles.npz", "annular_caption": "annular_caption.txt", "annular_manifest": "annular_manifest.json"}.items()}
    existing = next((path for path in outputs.values() if path.exists()), None)
    if existing is not None:
        raise FileExistsError(f"annular figure output already exists in stage: {existing.name}")
    # Keep the normalized rows for tabulation, but pass the caller's original
    # result back through the renderer.  ``payload`` intentionally contains
    # only shared metadata and does not duplicate the annulus arrays.
    figures = render_annular_peak_figures(result, data=data, width_mm=width, dpi=dpi)
    try:
        with _EXPORT_LOCK, mpl.rc_context(_FIGURE_RC):
            for stem, figure in figures.items():
                raise_if_cancelled(cancel_event, f"annular-peak-figures:render-{stem}")
                for suffix, file_format in (("svg", "svg"), ("pdf", "pdf"), ("png", "png"), ("tiff", "tiff")):
                    raise_if_cancelled(cancel_event, f"annular-peak-figures:save-{stem}-{suffix}")
                    figure.savefig(outputs[f"{stem}_{suffix}"], format=file_format, dpi=dpi if suffix in {"png", "tiff"} else None, facecolor="white", transparent=False)
                figure.clear()
    finally:
        for figure in figures.values():
            figure.clear()
    raise_if_cancelled(cancel_event, "annular-peak-figures:write-data")
    _write_profiles_csv(outputs["annular_profiles_csv"], rows, metadata)
    _write_peaks_csv(outputs["annular_peaks_csv"], rows, metadata)
    _write_npz(outputs["annular_profiles_npz"], payload, rows, metadata)
    outputs["annular_caption"].write_text(_caption(metadata, rows, representative_info) + "\n", encoding="utf-8", newline="\n")
    for path in outputs.values():
        if path.parent != stage_path:
            raise RuntimeError("annular exporter attempted to write outside its caller-owned stage")
    file_hashes = {path.name: _sha256(path) for key, path in outputs.items() if key != "annular_manifest"}
    manifest = {**metadata, "width_mm": width, "dpi": dpi, "representative_profiles": json_safe(representative_info), "files": list(file_hashes), "sha256": file_hashes, "manifest_excluded_from_own_sha256": True, "caption_file": "annular_caption.txt", "profiles_csv_file": "annular_profiles.csv", "peaks_csv_file": "annular_peaks.csv", "profiles_npz_file": "annular_profiles.npz", "figure_files": ["annular_qchi.svg", "annular_qchi.pdf", "annular_qchi.png", "annular_qchi.tiff", "annular_profiles.svg", "annular_profiles.pdf", "annular_profiles.png", "annular_profiles.tiff"]}
    _write_json(outputs["annular_manifest"], manifest)
    return outputs, json_safe(manifest)


__all__ = ["ANNULAR_FIGURE_METHOD_VERSION", "ANNULAR_FIGURE_SCHEMA_VERSION", "export_annular_peak_figures", "render_annular_peak_figures"]
