"""Publication assets for measured azimuthal sector-integrated SAXS peaks.

The sector measurement is performed upstream by :mod:`sector_peaks`.  This
module is deliberately a *figure and evidence writer*: it consumes the
already-measured ``result['sector_peaks']`` mapping and never re-integrates
the detector image.  Missing bins remain missing, overlapping sectors retain
their correlation warning, and a selected point is drawn at most once per
sector.

The output is split into two figures.  ``sector_qchi`` shows the complete raw
mean ``I(q, chi)`` field.  ``sector_profiles`` shows a small set of readable
sector profiles with the raw mean and locator-only smoothing explicitly
separated.  The latter is a diagnostic candidate display; smoothing is not a
fit and FWHM is not a confidence interval.
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


SECTOR_FIGURE_METHOD_VERSION = "sector-peak-figures-v1"
SECTOR_FIGURE_SCHEMA_VERSION = "sector-peak-figure-export-v1"

_SECTOR_COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
_REPRESENTATIVE_ANGLES = (0.0, 45.0, 90.0, 135.0)


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if np.isfinite(number) else None


def _csv_number(value: Any) -> str:
    number = _finite(value)
    return "" if number is None else format(number, ".17g")


def _number(value: Any, default: float = float("nan")) -> float:
    number = _finite(value)
    return default if number is None else number


def _array(value: Any, *, dtype: Any = float) -> np.ndarray:
    """Convert JSON-safe lists and native arrays while preserving holes."""

    if value is None:
        return np.asarray([], dtype=dtype)
    if dtype is float:
        if isinstance(value, np.ndarray):
            raw = value.tolist() if value.dtype.kind not in "fiu" else value
        else:
            raw = value
        try:
            return np.asarray(
                [np.nan if item is None else float(item) for item in raw],
                dtype=np.float64,
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("sector profile values must be numeric or null") from exc
    try:
        return np.asarray(value, dtype=dtype)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("sector profile values have an invalid shape") from exc


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _sector_rows(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Normalize the existing sector result without calculating measurements."""

    sectors_value = result.get("sectors", ())
    if sectors_value is None:
        return []
    if isinstance(sectors_value, (str, bytes)) or not isinstance(sectors_value, Sequence):
        raise TypeError("sector_peaks.sectors must be a sequence")
    q_root = _array(result.get("q_centers"), dtype=float)
    edge_root = _array(result.get("q_edges"), dtype=float)
    rows: list[dict[str, Any]] = []
    for fallback_index, raw in enumerate(sectors_value):
        sector = dict(_mapping(raw, f"sector_peaks.sectors[{fallback_index}]"))
        index_value = sector.get("sector_index", fallback_index)
        try:
            sector_index = int(index_value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("sector_index must be an integer") from exc
        angle = _number(sector.get("angle_deg", sector.get("center_angle_deg")))
        width = _number(sector.get("width_deg"))
        q = _array(sector.get("q_centers", sector.get("q")), dtype=float)
        if q.size == 0 and q_root.size:
            q = q_root.copy()
        edges = _array(sector.get("q_edges"), dtype=float)
        if edges.size == 0 and edge_root.size:
            edges = edge_root.copy()
        raw_mapping = sector.get("raw")
        raw_mapping = raw_mapping if isinstance(raw_mapping, Mapping) else {}
        raw_mean = _array(
            sector.get("raw_mean", sector.get("intensity", raw_mapping.get("mean"))),
            dtype=float,
        )
        raw_sum = _array(sector.get("raw_sum", raw_mapping.get("sum")), dtype=float)
        raw_count = _array(
            sector.get("raw_count", sector.get("counts", raw_mapping.get("count"))),
            dtype=np.int64,
        )
        geometric_count = _array(
            sector.get("geometric_counts", sector.get("geometry_count")), dtype=np.int64
        )
        coverage = _array(
            sector.get("coverage", sector.get("geometry_coverage")), dtype=float
        )
        smooth = _array(
            sector.get("smoothed_intensity", sector.get("smoothed")), dtype=float
        )
        lengths = {
            "q_centers": len(q),
            "raw_mean": len(raw_mean),
            "raw_sum": len(raw_sum),
            "raw_count": len(raw_count),
            "geometric_count": len(geometric_count),
            "coverage": len(coverage),
            "smoothed_intensity": len(smooth),
        }
        nonzero = {length for length in lengths.values() if length}
        if len(nonzero) > 1:
            raise ValueError(
                f"sector {sector_index} profile arrays have inconsistent lengths: {lengths}"
            )
        length = next(iter(nonzero), 0)
        if len(q) == 0 and length:
            raise ValueError(f"sector {sector_index} q_centers are required for export")
        if len(edges) not in {0, length + 1}:
            raise ValueError(f"sector {sector_index} q_edges must have len(q)+1 values")
        if len(edges) == 0 and length:
            edges = np.full(length + 1, np.nan, dtype=np.float64)
        for key in ("raw_mean", "raw_sum", "raw_count", "geometric_count", "coverage", "smoothed_intensity"):
            if len(sector.get(key, ())) == 0 and key not in sector:
                # Fill absent optional arrays with explicit missing values.  This
                # keeps a malformed/legacy sector visible in the CSV instead of
                # silently manufacturing a zero intensity profile.
                fill = np.full(length, np.nan, dtype=np.float64)
                if key in {"raw_count", "geometric_count"}:
                    fill = np.zeros(length, dtype=np.int64)
                sector[key] = fill
        supported = _array(sector.get("supported_bin_mask"), dtype=bool)
        if supported.size == 0 and length:
            supported = np.isfinite(raw_mean) & (raw_count > 0)
        if supported.size not in {0, length}:
            raise ValueError(f"sector {sector_index} supported_bin_mask length mismatch")
        if supported.size == 0:
            supported = np.zeros(length, dtype=bool)
        sector.update(
            {
                "sector_index": sector_index,
                "angle_deg": angle,
                "width_deg": width,
                "q_centers": q,
                "q_edges": edges,
                "raw_mean": raw_mean,
                "raw_sum": raw_sum,
                "raw_count": raw_count,
                "geometric_counts": geometric_count,
                "coverage": coverage,
                "smoothed_intensity": smooth,
                "supported_bin_mask": supported,
            }
        )
        rows.append(sector)
    return rows


def _intensity_unit(data: Mapping[str, Any], result: Mapping[str, Any]) -> tuple[str, str]:
    """Use only the explicit EDF header field when available.

    File names and paths are intentionally never inspected for units.  The
    fallback phrase describes the caller-supplied scale without guessing one.
    """

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


def _q_window(result: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> tuple[float | None, float | None]:
    value = result.get("q_window")
    if not isinstance(value, (str, bytes)):
        try:
            pair = list(value)
        except (TypeError, ValueError):
            pair = []
    else:
        pair = []
    if len(pair) == 2:
        low, high = _finite(pair[0]), _finite(pair[1])
        if low is not None and high is not None:
            return low, high
    edges: list[np.ndarray] = [np.asarray(row["q_edges"], dtype=float) for row in rows if len(row["q_edges"])]
    finite = np.concatenate([array[np.isfinite(array)] for array in edges]) if edges else np.asarray([], dtype=float)
    if finite.size:
        return float(np.min(finite)), float(np.max(finite))
    return None, None


def _metadata(data: Mapping[str, Any], result: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    intensity_unit, intensity_source = _intensity_unit(data, result)
    units = result.get("units")
    units = units if isinstance(units, Mapping) else {}
    q_unit = str(result.get("q_unit", units.get("q", data.get("q_unit", "unknown"))) or "unknown")
    q_min, q_max = _q_window(result, rows)
    settings = result.get("settings")
    settings = settings if isinstance(settings, Mapping) else {}
    sampling = result.get("sampling")
    sampling = sampling if isinstance(sampling, Mapping) else {}
    overlap = result.get("sector_overlap")
    overlap = overlap if isinstance(overlap, Mapping) else {}
    q_step = _finite(
        sampling.get("effective_radial_bin_width", settings.get("q_bin_step"))
    )
    representative_step = _finite(
        sampling.get("representative_q_step", settings.get("representative_q_step"))
    )
    return {
        "schema_version": SECTOR_FIGURE_SCHEMA_VERSION,
        "method_version": SECTOR_FIGURE_METHOD_VERSION,
        "source_measurement_method_version": result.get("method_version"),
        "q_unit": q_unit,
        "intensity_unit": intensity_unit,
        "intensity_unit_source": intensity_source,
        "q_window": [q_min, q_max] if q_min is not None and q_max is not None else None,
        "q_bin_width": q_step,
        "representative_q_step": representative_step,
        "settings": json_safe(settings),
        "sector_overlap": json_safe(overlap),
        "overlap_correlation_warning": str(
            overlap.get(
                "correlation_statement",
                "overlapping sectors may share detector pixels and are correlated; they are not independent samples",
            )
        ),
        "sector_count": len(rows),
        "selected_peak_count": sum(
            isinstance(row.get("selected_peak"), Mapping) for row in rows
        ),
        "missing_bins_are_left_blank": True,
        "quadrants_are_not_synthesized": True,
        "smooth_is_locator_only": True,
        "fwhm_is_not_confidence_interval": True,
    }


def _angle_edges(angles: np.ndarray) -> np.ndarray:
    if angles.size == 0:
        return np.asarray([0.0, 360.0], dtype=float)
    order = np.argsort(angles)
    sorted_angles = np.asarray(angles[order], dtype=float)
    if sorted_angles.size == 1:
        width = 1.0
        return np.asarray([sorted_angles[0] - width, sorted_angles[0] + width])
    edges = np.empty(sorted_angles.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (sorted_angles[:-1] + sorted_angles[1:])
    edges[0] = sorted_angles[0] - 0.5 * (sorted_angles[1] - sorted_angles[0])
    edges[-1] = sorted_angles[-1] + 0.5 * (sorted_angles[-1] - sorted_angles[-2])
    return edges


def _selected(row: Mapping[str, Any]) -> Mapping[str, Any] | None:
    value = row.get("selected_peak")
    return value if isinstance(value, Mapping) else None


_STATUS_REASON_LABELS = {
    "low_q_boundary_without_two_sided_support": "low-q boundary",
    "insufficient_two_sided_support": "insufficient two-sided support",
    "insufficient_two_sided_background_return": "no two-sided background return",
    "ambiguous_multiple_peaks": "ambiguous candidates",
    "flat_top_unresolved_peak": "flat-top unresolved",
    "no_supported_bins_after_count_coverage_gate": "insufficient count/coverage",
    "no_finite_intensity_support": "no finite intensity support",
    "monotonic_profile": "monotonic profile",
}


def _compact_status(row: Mapping[str, Any]) -> str:
    """Return a short visual status; the full reason remains in CSV/JSON."""

    status = str(row.get("status") or "unknown").replace("_", " ")
    reason = str(row.get("reason") or "")
    if status == "selected":
        return "selected"
    if reason:
        reason = _STATUS_REASON_LABELS.get(reason, reason.replace("_", " "))
        if len(reason) > 28:
            reason = reason[:25].rstrip() + "..."
        return f"{status} · {reason}"
    return status


def _build_qchi_figure(rows: Sequence[Mapping[str, Any]], metadata: Mapping[str, Any], *, width_mm: float, dpi: int) -> Figure:
    height_mm = 108.0 if width_mm == 183.0 else 132.0
    fig = Figure(figsize=(width_mm / 25.4, height_mm / 25.4), dpi=dpi, facecolor="white", edgecolor="white")
    FigureCanvasAgg(fig)
    if width_mm == 183.0:
        ax = fig.add_axes([0.10, 0.19, 0.78, 0.68])
        cax = fig.add_axes([0.90, 0.25, 0.018, 0.52])
    else:
        ax = fig.add_axes([0.16, 0.24, 0.68, 0.59])
        cax = fig.add_axes([0.87, 0.32, 0.028, 0.40])
    if not rows:
        ax.text(0.5, 0.5, "No sector profiles supplied", transform=ax.transAxes, ha="center", va="center", color="#555555")
        ax.set_axis_off()
        fig.suptitle("Sector-integrated I(q, χ) · no data", y=0.96, fontsize=7.0)
        return fig
    order = np.argsort([_number(row.get("angle_deg")) for row in rows])
    ordered = [rows[int(index)] for index in order]
    angles = np.asarray([_number(row.get("angle_deg")) for row in ordered], dtype=float)
    lengths = [len(row["q_centers"]) for row in ordered]
    max_len = max(lengths, default=0)
    q_reference = next((np.asarray(row["q_centers"], dtype=float) for row in ordered if len(row["q_centers"])), np.asarray([], dtype=float))
    q_edges = next((np.asarray(row["q_edges"], dtype=float) for row in ordered if len(row["q_edges"])), np.asarray([], dtype=float))
    matrix = np.full((len(ordered), max_len), np.nan, dtype=float)
    for row_index, row in enumerate(ordered):
        values = np.asarray(row["raw_mean"], dtype=float)
        matrix[row_index, : len(values)] = values
    masked = np.ma.masked_invalid(matrix)
    if len(q_edges) == max_len + 1 and max_len:
        angle_edges = _angle_edges(angles)
        mesh = ax.pcolormesh(q_edges, angle_edges, masked, shading="auto", cmap="cividis", rasterized=True)
    elif max_len:
        mesh = ax.pcolormesh(q_reference, np.arange(len(ordered) + 1), masked, shading="nearest", cmap="cividis", rasterized=True)
        ax.set_yticks(np.arange(len(ordered)) + 0.5, [f"{value:.1f}" for value in angles])
    else:
        ax.text(0.5, 0.5, "No finite q bins supplied", transform=ax.transAxes, ha="center", va="center", color="#555555")
        mesh = None
    if mesh is not None:
        cbar = fig.colorbar(mesh, cax=cax)
        cbar.set_label(f"raw sector mean I ({metadata['intensity_unit']})", fontsize=5.8, labelpad=2.0)
        cbar.ax.tick_params(labelsize=5.0, length=1.2, width=0.4, pad=1.0)
    for row, angle in zip(ordered, angles, strict=True):
        selected = _selected(row)
        q_star = _finite(selected.get("q_star")) if selected is not None else None
        if q_star is not None:
            ax.scatter([q_star], [angle], s=17, marker="o", facecolors="none", edgecolors="#D55E00", linewidths=0.8, zorder=5)
    q_unit = str(metadata["q_unit"])
    ax.set_xlabel(f"q ({q_unit})")
    ax.set_ylabel("Azimuth χ (degree)")
    ax.set_title("Raw sector-integrated intensity I(q, χ)", fontsize=7.0)
    ax.tick_params(direction="out", length=2.0, width=0.5, pad=1.5)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    if width_mm == 183.0:
        footer = (
            "Blank cells = missing measured support; circles = one selected candidate per χ.\n"
            "No missing quadrants are synthesized; overlapping sectors are correlated."
        )
    else:
        footer = (
            "Blank cells = missing measured support.\n"
            "Circles = one selected candidate per χ; no missing quadrants are synthesized.\n"
            "Overlapping sectors are correlated."
        )
    fig.text(
        0.5,
        0.055 if width_mm == 183.0 else 0.045,
        footer,
        ha="center",
        va="center",
        fontsize=5.0,
        linespacing=1.25,
        color="#555555",
    )
    return fig


def _representative_rows(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    if not rows:
        return []
    remaining = list(rows)
    selected_rows: list[Mapping[str, Any]] = []
    for target in _REPRESENTATIVE_ANGLES:
        if not remaining:
            break
        index = min(
            range(len(remaining)),
            key=lambda item: abs(((_number(remaining[item].get("angle_deg")) - target + 180.0) % 360.0) - 180.0),
        )
        selected_rows.append(remaining.pop(index))
    return selected_rows


def _build_profiles_figure(rows: Sequence[Mapping[str, Any]], metadata: Mapping[str, Any], *, width_mm: float, dpi: int) -> Figure:
    height_mm = 120.0 if width_mm == 183.0 else 145.0
    fig = Figure(figsize=(width_mm / 25.4, height_mm / 25.4), dpi=dpi, facecolor="white", edgecolor="white")
    FigureCanvasAgg(fig)
    if width_mm == 183.0:
        profile_ax = fig.add_axes([0.09, 0.46, 0.84, 0.34])
        support_ax = fig.add_axes([0.09, 0.15, 0.84, 0.18], sharex=profile_ax)
    else:
        profile_ax = fig.add_axes([0.16, 0.47, 0.70, 0.33])
        support_ax = fig.add_axes([0.16, 0.15, 0.70, 0.18], sharex=profile_ax)
    reps = _representative_rows(rows)
    if not reps:
        profile_ax.text(0.5, 0.5, "No sector profiles supplied", transform=profile_ax.transAxes, ha="center", va="center", color="#555555")
        support_ax.set_axis_off()
    else:
        coverage_ax = support_ax.twinx()
        handles: list[Any] = []
        for index, row in enumerate(reps):
            color = _SECTOR_COLORS[index % len(_SECTOR_COLORS)]
            angle = _number(row.get("angle_deg"))
            q = np.asarray(row["q_centers"], dtype=float)
            raw = np.asarray(row["raw_mean"], dtype=float)
            smooth = np.asarray(row["smoothed_intensity"], dtype=float)
            count = np.asarray(row["raw_count"], dtype=float)
            coverage = np.asarray(row["coverage"], dtype=float)
            q_len = min(len(q), len(raw), len(smooth))
            if q_len:
                q, raw, smooth = q[:q_len], raw[:q_len], smooth[:q_len]
                (raw_line,) = profile_ax.plot(q, raw, color=color, linewidth=0.65, marker="o", markersize=1.4, markerfacecolor="none", alpha=0.9, label=f"χ={angle:.1f}° raw sector mean")
                (smooth_line,) = profile_ax.plot(q, smooth, color=color, linewidth=1.0, linestyle="--", alpha=0.95, label=f"χ={angle:.1f}° locator-only smooth")
                handles.extend([raw_line, smooth_line])
            if len(count):
                q_count = np.asarray(row["q_centers"], dtype=float)[: len(count)]
                support_ax.plot(q_count, count, color=color, linewidth=0.65, alpha=0.8)
            if len(coverage):
                q_cov = np.asarray(row["q_centers"], dtype=float)[: len(coverage)]
                coverage_ax.plot(q_cov, coverage[: len(q_cov)], color=color, linewidth=0.65, linestyle=":", alpha=0.85)
            selected = _selected(row)
            q_star = _finite(selected.get("q_star")) if selected is not None else None
            if q_star is not None and q_len:
                intensity = _finite(selected.get("raw_intensity"))
                if intensity is None:
                    intensity = _finite(selected.get("intensity"))
                if intensity is None:
                    finite = raw[np.isfinite(raw)]
                    intensity = float(np.max(finite)) if finite.size else None
                if intensity is not None:
                    profile_ax.scatter([q_star], [intensity], s=18, marker="o", facecolors="white", edgecolors=color, linewidths=0.9, zorder=6)
            profile_ax.text(
                0.985,
                0.96 - index * 0.13,
                f"χ={angle:.1f}° · {_compact_status(row)}",
                transform=profile_ax.transAxes,
                ha="right",
                va="top",
                fontsize=5.0,
                color=color,
                clip_on=False,
            )
        profile_ax.set_ylabel(f"Raw intensity ({metadata['intensity_unit']})")
        support_ax.set_ylabel("valid count", fontsize=5.8)
        coverage_ax.set_ylabel("coverage", fontsize=5.8, color="#555555")
        coverage_ax.set_ylim(0.0, 1.05)
        coverage_ax.tick_params(axis="y", labelsize=5.0, colors="#555555", length=1.5, width=0.4)
        support_ax.set_xlabel(f"q ({metadata['q_unit']})")
        support_ax.set_title("Support: count + coverage", fontsize=5.8, pad=2.0)
        if handles:
            # Keep the role legend at figure level so the support axes cannot
            # cover it.  Representative-sector colours are identified by the
            # compact status labels in the profile panel.
            legend_handles = [
                Line2D(
                    [0], [0], color="#555555", marker="o",
                    markerfacecolor="none", linewidth=0.75, markersize=2.5,
                    label="raw sector mean",
                ),
                Line2D(
                    [0], [0], color="#555555", linestyle="--", linewidth=0.95,
                    label="locator-only smooth",
                ),
                Line2D(
                    [0], [0], color="#555555", linestyle=":", linewidth=0.75,
                    label="coverage",
                ),
            ]
            fig.legend(
                handles=legend_handles,
                loc="lower center",
                bbox_to_anchor=(0.5, 0.355),
                frameon=False,
                ncol=1,
                fontsize=5.0,
                handlelength=1.3,
                handletextpad=0.3,
                columnspacing=0.7,
            )
    for axis in (profile_ax, support_ax):
        axis.tick_params(direction="out", length=2.0, width=0.5, pad=1.4, labelsize=5.2)
        axis.grid(axis="y", color="#dddddd", linewidth=0.3)
        for spine in axis.spines.values():
            spine.set_linewidth(0.5)
    if width_mm == 183.0:
        footer = (
            "Selected points are diagnostic candidates; smoothing is locator-only and FWHM is not a confidence interval.\n"
            "Missing bins stay blank."
        )
    else:
        footer = (
            "Markers are diagnostic candidates; raw data remain primary.\n"
            "Missing bins stay blank; no fit or confidence interval is added."
        )
    fig.suptitle(
        "Representative sector I(q | χ) · raw mean + locator-only smooth"
        if width_mm == 183.0
        else "Representative sector I(q | χ)\nraw mean + locator-only smooth",
        x=0.5,
        y=0.965,
        fontsize=7.0,
        linespacing=1.05,
    )
    fig.text(
        0.5,
        0.055 if width_mm == 183.0 else 0.045,
        footer,
        ha="center",
        va="center",
        fontsize=5.0,
        linespacing=1.25,
        color="#555555",
    )
    return fig


def render_sector_peak_figures(result: Mapping[str, Any], *, data: Mapping[str, Any] | None = None, width_mm: float = 183.0, dpi: int = 600) -> dict[str, Figure]:
    """Build the two sector figures from an existing sector result."""

    if not isinstance(result, Mapping):
        raise TypeError("result must be a sector_peaks mapping")
    width = float(width_mm)
    if width not in {89.0, 183.0}:
        raise ValueError("width_mm must be 89 or 183")
    if isinstance(dpi, bool) or int(dpi) != dpi or int(dpi) <= 0:
        raise ValueError("dpi must be a positive integer")
    data = data or {}
    rows = _sector_rows(result)
    metadata = _metadata(data, result, rows)
    with _EXPORT_LOCK, mpl.rc_context(_FIGURE_RC):
        figures = {
            "sector_qchi": _build_qchi_figure(rows, metadata, width_mm=width, dpi=int(dpi)),
            "sector_profiles": _build_profiles_figure(rows, metadata, width_mm=width, dpi=int(dpi)),
        }
        for figure in figures.values():
            figure.canvas.draw()
        return figures


_PROFILE_FIELDS = (
    "sector_index", "angle_deg", "width_deg", "q_unit", "intensity_unit",
    "q_window_min", "q_window_max", "q_bin_width", "representative_q_step",
    "q_bin_index", "q_left", "q_right", "q_center", "raw_mean", "raw_sum",
    "raw_count", "geometric_count", "coverage", "supported", "smoothed_intensity",
    "candidate_count", "selected_q_star", "selected_prominence", "selected_snr",
    "selected_radial_fwhm", "sector_status", "sector_reason", "overlap_fraction",
    "overlapping_sectors", "overlap_correlation_statement",
)

_PEAK_FIELDS = (
    "record_type", "sector_index", "angle_deg", "width_deg", "q_unit", "intensity_unit",
    "q_window_min", "q_window_max", "q_bin_width", "representative_q_step",
    "status", "reason", "selected", "q_star", "q_bin_center", "raw_intensity",
    "smoothed_intensity", "prominence", "snr", "radial_fwhm", "sampling_sigma_q",
    "half_bin_resolution_q", "peak_bin_index", "source_pixel_count", "pixel_x",
    "pixel_y", "source_angle_deg", "candidate_count", "candidate_reason",
    "candidate_json", "overlap_fraction", "overlapping_sectors",
)


def _safe_csv_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Protect every CSV cell while leaving signed numeric values unchanged."""

    return {key: safe_csv_cell(value) for key, value in row.items()}


def _write_profiles_csv(path: Path, rows: Sequence[Mapping[str, Any]], metadata: Mapping[str, Any]) -> None:
    q_window = metadata.get("q_window") or (None, None)
    overlap = metadata.get("sector_overlap") or {}
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=_PROFILE_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            q = np.asarray(row["q_centers"], dtype=float)
            edges = np.asarray(row["q_edges"], dtype=float)
            raw = np.asarray(row["raw_mean"], dtype=float)
            sums = np.asarray(row["raw_sum"], dtype=float)
            counts = np.asarray(row["raw_count"], dtype=float)
            geometric = np.asarray(row["geometric_counts"], dtype=float)
            coverage = np.asarray(row["coverage"], dtype=float)
            supported = np.asarray(row["supported_bin_mask"], dtype=bool)
            smooth = np.asarray(row["smoothed_intensity"], dtype=float)
            length = max(map(len, (q, raw, sums, counts, geometric, coverage, supported, smooth)), default=0)
            selected = _selected(row)
            candidate_count = len(row.get("candidates", ())) if isinstance(row.get("candidates", ()), Sequence) else 0
            for index in range(length):
                edge_left = edges[index] if index < len(edges) else np.nan
                edge_right = edges[index + 1] if index + 1 < len(edges) else np.nan
                writer.writerow(_safe_csv_row({
                    "sector_index": row["sector_index"], "angle_deg": _csv_number(row.get("angle_deg")), "width_deg": _csv_number(row.get("width_deg")),
                    "q_unit": metadata["q_unit"], "intensity_unit": metadata["intensity_unit"], "q_window_min": _csv_number(q_window[0]), "q_window_max": _csv_number(q_window[1]),
                    "q_bin_width": _csv_number(metadata.get("q_bin_width")), "representative_q_step": _csv_number(metadata.get("representative_q_step")), "q_bin_index": index,
                    "q_left": _csv_number(edge_left), "q_right": _csv_number(edge_right), "q_center": _csv_number(q[index] if index < len(q) else np.nan),
                    "raw_mean": _csv_number(raw[index] if index < len(raw) else np.nan), "raw_sum": _csv_number(sums[index] if index < len(sums) else np.nan),
                    "raw_count": _csv_number(counts[index] if index < len(counts) else np.nan), "geometric_count": _csv_number(geometric[index] if index < len(geometric) else np.nan),
                    "coverage": _csv_number(coverage[index] if index < len(coverage) else np.nan), "supported": bool(supported[index]) if index < len(supported) else False,
                    "smoothed_intensity": _csv_number(smooth[index] if index < len(smooth) else np.nan), "candidate_count": candidate_count,
                    "selected_q_star": _csv_number(selected.get("q_star")) if selected else "", "selected_prominence": _csv_number(selected.get("prominence")) if selected else "",
                    "selected_snr": _csv_number(selected.get("snr")) if selected else "", "selected_radial_fwhm": _csv_number(selected.get("radial_fwhm")) if selected else "",
                    "sector_status": row.get("status", ""), "sector_reason": row.get("reason") or "", "overlap_fraction": _csv_number(overlap.get("overlap_fraction")),
                    "overlapping_sectors": bool(overlap.get("overlapping", False)), "overlap_correlation_statement": metadata["overlap_correlation_warning"],
                }))


def _write_peaks_csv(path: Path, rows: Sequence[Mapping[str, Any]], metadata: Mapping[str, Any]) -> None:
    q_window = metadata.get("q_window") or (None, None)
    overlap = metadata.get("sector_overlap") or {}
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=_PEAK_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            candidates = row.get("candidates", ())
            if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
                candidates = ()
            entries: list[tuple[str, Mapping[str, Any] | None]] = [("sector_summary", _selected(row))]
            entries.extend(("candidate", item if isinstance(item, Mapping) else None) for item in candidates)
            for record_type, candidate in entries:
                candidate = candidate or {}
                writer.writerow(_safe_csv_row({
                    "record_type": record_type, "sector_index": row["sector_index"], "angle_deg": _csv_number(row.get("angle_deg")), "width_deg": _csv_number(row.get("width_deg")),
                    "q_unit": metadata["q_unit"], "intensity_unit": metadata["intensity_unit"], "q_window_min": _csv_number(q_window[0]), "q_window_max": _csv_number(q_window[1]),
                    "q_bin_width": _csv_number(metadata.get("q_bin_width")), "representative_q_step": _csv_number(metadata.get("representative_q_step")),
                    "status": candidate.get("status", row.get("status", "")), "reason": candidate.get("reason") or row.get("reason") or "", "selected": bool(candidate.get("selected", False)),
                    "q_star": _csv_number(candidate.get("q_star")), "q_bin_center": _csv_number(candidate.get("q_bin_center")), "raw_intensity": _csv_number(candidate.get("raw_intensity", candidate.get("intensity"))),
                    "smoothed_intensity": _csv_number(candidate.get("smoothed_intensity")), "prominence": _csv_number(candidate.get("prominence")), "snr": _csv_number(candidate.get("snr")),
                    "radial_fwhm": _csv_number(candidate.get("radial_fwhm")), "sampling_sigma_q": _csv_number(candidate.get("sampling_sigma_q")), "half_bin_resolution_q": _csv_number(candidate.get("half_bin_resolution_q")),
                    "peak_bin_index": candidate.get("peak_bin_index", ""), "source_pixel_count": candidate.get("source_pixel_count", ""), "pixel_x": candidate.get("pixel_x", ""), "pixel_y": candidate.get("pixel_y", ""), "source_angle_deg": _csv_number(candidate.get("source_angle_deg")),
                    "candidate_count": len(candidates), "candidate_reason": candidate.get("reason") or row.get("reason") or "", "candidate_json": json.dumps(json_safe(candidate), ensure_ascii=False, sort_keys=True, allow_nan=False),
                    "overlap_fraction": _csv_number(overlap.get("overlap_fraction")), "overlapping_sectors": bool(overlap.get("overlapping", False)),
                }))


def _write_profiles_npz(path: Path, rows: Sequence[Mapping[str, Any]], metadata: Mapping[str, Any]) -> None:
    max_len = max((len(row["q_centers"]) for row in rows), default=0)
    max_edges = max((len(row["q_edges"]) for row in rows), default=max_len + 1)
    def matrix(key: str, *, dtype: Any = float, fill: Any = np.nan) -> np.ndarray:
        output = np.full((len(rows), max_len), fill, dtype=dtype)
        for index, row in enumerate(rows):
            values = np.asarray(row[key], dtype=dtype)
            output[index, : min(max_len, len(values))] = values[:max_len]
        return output
    edges = np.full((len(rows), max_edges), np.nan, dtype=float)
    for index, row in enumerate(rows):
        values = np.asarray(row["q_edges"], dtype=float)
        edges[index, : min(max_edges, len(values))] = values[:max_edges]
    selected_q = np.asarray([_number((_selected(row) or {}).get("q_star")) for row in rows], dtype=float)
    payload = {
        "sector_index": np.asarray([int(row["sector_index"]) for row in rows], dtype=np.int64),
        "angle_deg": np.asarray([_number(row.get("angle_deg")) for row in rows], dtype=float),
        "width_deg": np.asarray([_number(row.get("width_deg")) for row in rows], dtype=float),
        "q_centers": matrix("q_centers"), "q_edges": edges, "raw_mean": matrix("raw_mean"), "raw_sum": matrix("raw_sum"),
        "raw_count": matrix("raw_count", dtype=np.int64, fill=0), "geometric_counts": matrix("geometric_counts", dtype=np.int64, fill=0), "coverage": matrix("coverage"),
        "supported_bin_mask": matrix("supported_bin_mask", dtype=bool, fill=False), "smoothed_intensity": matrix("smoothed_intensity"),
        "selected_q_star": selected_q, "sector_status": np.asarray([str(row.get("status", "")) for row in rows], dtype="U32"),
        "sector_reason": np.asarray([str(row.get("reason") or "") for row in rows], dtype="U160"), "q_unit": np.asarray(str(metadata["q_unit"])),
        "intensity_unit": np.asarray(str(metadata["intensity_unit"])), "q_window": np.asarray(metadata.get("q_window") or [np.nan, np.nan], dtype=float),
        "q_bin_width": np.asarray(_number(metadata.get("q_bin_width"))), "representative_q_step": np.asarray(_number(metadata.get("representative_q_step"))),
        "schema_version": np.asarray(SECTOR_FIGURE_SCHEMA_VERSION), "source_method_version": np.asarray(str(metadata.get("source_measurement_method_version") or "")),
    }
    np.savez_compressed(path, **payload)


def _caption(metadata: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> str:
    window = metadata.get("q_window")
    window_text = "unknown q window" if not window else f"[{window[0]:.6g}, {window[1]:.6g}] {metadata['q_unit']}"
    selected = sum(_selected(row) is not None for row in rows)
    return (
        "Sector-integrated butterfly SAXS evidence. The q–χ panel displays the raw arithmetic mean intensity "
        f"for each measured azimuthal sector over q window {window_text}; bins without measured intensity remain blank. "
        "Finite raw means remain visible even when count/coverage is insufficient for peak detection; "
        "no opposite quadrant or missing sector is synthesized. The representative profiles show the same raw "
        "sector mean and a separately named locator-only smooth copied from the supplied sector result. Smoothing is "
        "not a fit. At most one selected diagnostic candidate is marked per sector ("
        f"{selected} of {len(rows)} sectors in this bundle); a no_peak or ambiguous status is retained with its reason. "
        f"The source intensity scale is {metadata['intensity_unit']} ({metadata['intensity_unit_source']}). "
        "Counts and coverage are detector support statistics, not independent experimental replicates. Overlapping "
        "sectors share pixels and are correlated. Candidate prominence, SNR, and FWHM are diagnostic fields; FWHM "
        "is a radial width and not a confidence interval, and no scientific acceptance is inferred."
    )


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


def export_sector_peak_figures(stage: str | Path, *, data: Mapping[str, Any], result: Mapping[str, Any], cancel_event: Any = None) -> tuple[dict[str, Path], dict[str, Any]]:
    """Write sector assets into an existing caller-owned atomic stage."""

    raise_if_cancelled(cancel_event, "sector-peak-figures:validate")
    stage_path = Path(stage).expanduser().resolve(strict=True)
    if not stage_path.is_dir():
        raise ValueError("stage must be an existing directory")
    if not isinstance(data, Mapping) or not isinstance(result, Mapping):
        raise TypeError("data and result must be mappings")
    width_mm = float(data.get("width_mm", 183.0))
    dpi_raw = data.get("dpi", 600)
    if width_mm not in {89.0, 183.0}:
        raise ValueError("data.width_mm must be 89 or 183")
    if isinstance(dpi_raw, bool) or int(dpi_raw) != dpi_raw or int(dpi_raw) <= 0:
        raise ValueError("data.dpi must be a positive integer")
    dpi = int(dpi_raw)
    rows = _sector_rows(result)
    metadata = _metadata(data, result, rows)
    filenames = {
        "sector_qchi_svg": "sector_qchi.svg", "sector_qchi_pdf": "sector_qchi.pdf", "sector_qchi_png": "sector_qchi.png", "sector_qchi_tiff": "sector_qchi.tiff",
        "sector_profiles_svg": "sector_profiles.png.svg", "sector_profiles_pdf": "sector_profiles.png.pdf", "sector_profiles_png": "sector_profiles.png", "sector_profiles_tiff": "sector_profiles.tiff",
        "sector_profiles_csv": "sector_profiles.csv", "sector_peaks_csv": "sector_peaks.csv", "sector_profiles_npz": "sector_profiles.npz", "sector_caption": "sector_caption.txt", "sector_manifest": "sector_manifest.json",
    }
    # Keep the source stem simple for consumers while avoiding a collision with
    # the machine-readable CSV and NPZ files.
    filenames["sector_profiles_svg"] = "sector_profiles.svg"
    filenames["sector_profiles_pdf"] = "sector_profiles.pdf"
    outputs = {key: stage_path / name for key, name in filenames.items()}
    existing = next((path for path in outputs.values() if path.exists()), None)
    if existing is not None:
        raise FileExistsError(f"sector figure output already exists in stage: {existing.name}")
    figures = render_sector_peak_figures(result, data=data, width_mm=width_mm, dpi=dpi)
    try:
        with _EXPORT_LOCK, mpl.rc_context(_FIGURE_RC):
            for stem, figure in figures.items():
                raise_if_cancelled(cancel_event, f"sector-peak-figures:render-{stem}")
                figure.canvas.draw()
                for suffix, file_format in (("svg", "svg"), ("pdf", "pdf"), ("png", "png"), ("tiff", "tiff")):
                    raise_if_cancelled(cancel_event, f"sector-peak-figures:save-{stem}-{suffix}")
                    figure.savefig(outputs[f"{stem}_{suffix}"], format=file_format, dpi=dpi if suffix in {"png", "tiff"} else None, facecolor="white", transparent=False)
                figure.clear()
    finally:
        for figure in figures.values():
            if figure.canvas is not None:
                figure.clear()
    raise_if_cancelled(cancel_event, "sector-peak-figures:write-data")
    _write_profiles_csv(outputs["sector_profiles_csv"], rows, metadata)
    _write_peaks_csv(outputs["sector_peaks_csv"], rows, metadata)
    _write_profiles_npz(outputs["sector_profiles_npz"], rows, metadata)
    caption = _caption(metadata, rows)
    outputs["sector_caption"].write_text(caption + "\n", encoding="utf-8", newline="\n")
    for path in outputs.values():
        if path.parent != stage_path:
            raise RuntimeError("sector exporter attempted to write outside its caller-owned stage")
    file_hashes = {path.name: _sha256(path) for key, path in outputs.items() if key != "sector_manifest"}
    manifest = {
        **metadata,
        "width_mm": width_mm,
        "dpi": dpi,
        "representative_angles_deg": list(_REPRESENTATIVE_ANGLES),
        "files": list(file_hashes),
        "sha256": file_hashes,
        "manifest_excluded_from_own_sha256": True,
        "caption_file": "sector_caption.txt",
        "profiles_csv_file": "sector_profiles.csv",
        "peaks_csv_file": "sector_peaks.csv",
        "profiles_npz_file": "sector_profiles.npz",
        "figure_files": ["sector_qchi.svg", "sector_qchi.pdf", "sector_qchi.png", "sector_qchi.tiff", "sector_profiles.svg", "sector_profiles.pdf", "sector_profiles.png", "sector_profiles.tiff"],
    }
    _write_json(outputs["sector_manifest"], manifest)
    return outputs, json_safe(manifest)


__all__ = [
    "SECTOR_FIGURE_METHOD_VERSION",
    "SECTOR_FIGURE_SCHEMA_VERSION",
    "export_sector_peak_figures",
    "render_sector_peak_figures",
]
