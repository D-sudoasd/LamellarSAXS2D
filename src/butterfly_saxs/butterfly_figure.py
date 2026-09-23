"""Native, measurement-based butterfly SAXS publication figures.

The primary panel is the supplied intensity image on its original qx/qy mesh.
Only source-provided ridge points and fully supported ordered observed arcs are
overlaid. Candidate ellipse geometry is retained in the result sidecar, never
drawn or promoted to an accepted measurement by this exporter.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import csv
import hashlib
import html
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import matplotlib as mpl
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np

from .cancellation import raise_if_cancelled
from .serialization import json_safe

from .figure_support import (
    _EXPORT_LOCK,
    _FIGURE_RC,
    _SUPPORTED_WIDTHS_MM as _SUPPORTED_WIDTHS_MM,
    _draw_map,
    _numeric_array as _numeric_array,
    _point_display_status,
    _prepare_inputs,
    _supported_arc_segments,
)


FIGURE_METHOD_VERSION = "butterfly-measurement-figure-v1"
NATURE_FIGURE_GUIDE_URL = (
    "https://research-figure-guide.nature.com/figures/building-and-exporting-figure-panels/"
)


_RIDGE_CSV_FIELDS = (
    "point_index",
    "point_id",
    "qx",
    "qy",
    "pixel_x",
    "pixel_y",
    "arc_id",
    "branch_id",
    "side",
    "accepted",
    "valid",
    "display_status",
    "overlay_status",
    "overlay_reason",
    "reason",
    "source_json",
)

_RADIAL_PROFILE_FIELDS = (
    "bin_index",
    "q_unit",
    "edge_left",
    "edge_right",
    "center",
    "raw_sum",
    "raw_mean",
    "count",
)


def _notify(
    progress: Callable[[int, str], Any] | None, percent: int, phase: str
) -> None:
    if progress is not None:
        progress(int(percent), str(phase))


def _draw_radial_diagnostic(
    ax_profile: Any, ax_count: Any, data: Mapping[str, Any], *, counts_bottom: bool
) -> None:
    from matplotlib.ticker import MaxNLocator, ScalarFormatter

    centers = data["radial_centers"]
    means = data["radial_mean"]
    counts = data["radial_counts"]
    ax_profile.plot(
        centers, means, color="#0072b2", linewidth=0.85, marker="o", markersize=1.6
    )
    ax_profile.set_yscale("symlog", linthresh=data["radial_linthresh"])
    # Matplotlib's default SymmetricalLogLocator only places major ticks at
    # powers of ten.  A narrow profile can therefore have a valid visible
    # range with no labelled major tick at all (for example 3.2--4.2).  Keep
    # the symlog transform, but use a bounded linear-in-data locator/formatter
    # as a display fallback when fewer than two major ticks are visible.
    y_low, y_high = (float(value) for value in ax_profile.get_ylim())
    locator = ax_profile.yaxis.get_major_locator()
    try:
        major_ticks = np.asarray(
            locator.tick_values(y_low, y_high), dtype=np.float64
        )
    except (AttributeError, TypeError, ValueError, OverflowError):
        major_ticks = np.asarray(ax_profile.get_yticks(), dtype=np.float64)
    visible_ticks = major_ticks[
        np.isfinite(major_ticks)
        & (major_ticks >= y_low)
        & (major_ticks <= y_high)
    ]
    if visible_ticks.size < 2:
        ax_profile.yaxis.set_major_locator(
            MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10], prune="both")
        )
        formatter = ScalarFormatter(useOffset=False)
        formatter.set_powerlimits((-3, 3))
        ax_profile.yaxis.set_major_formatter(formatter)
    ax_profile.set_ylabel("Mean input intensity (symlog)")
    ax_profile.set_xlim(float(data["radial_edges"][0]), float(data["radial_edges"][-1]))
    ax_profile.tick_params(direction="out", length=2.0, width=0.5, pad=1.5)
    ax_profile.grid(axis="y", color="#dddddd", linewidth=0.35, zorder=0)
    for spine in ax_profile.spines.values():
        spine.set_linewidth(0.5)
    max_count = max(1, int(np.max(counts)))
    widths = np.diff(data["radial_edges"])
    ax_count.bar(
        centers,
        counts,
        width=widths,
        align="center",
        color="#b8b8b8",
        edgecolor="none",
        rasterized=True,
    )
    ax_count.set_ylim(0, max_count * 1.1)
    ax_count.set_ylabel("n")
    if counts_bottom:
        ax_count.set_xlabel(f"q radius from supplied origin ({data['q_unit']})")
    else:
        ax_profile.set_xlabel(f"q radius from supplied origin ({data['q_unit']})")
        ax_count.tick_params(labelbottom=False)
    ax_count.tick_params(direction="out", length=2.0, width=0.5, pad=1.5, labelsize=5.0)
    ax_count.set_yticks([0, max_count] if max_count > 1 else [0, 1])
    for spine in ax_count.spines.values():
        spine.set_linewidth(0.5)


def _build_figure(data: Mapping[str, Any]) -> Figure:
    width_mm = data["width_mm"]
    height_mm = data["height_mm"]
    fig = Figure(
        figsize=(width_mm / 25.4, height_mm / 25.4),
        dpi=data["dpi"],
        facecolor="white",
        edgecolor="white",
    )
    FigureCanvasAgg(fig)
    if width_mm == 183.0:
        map_ax = fig.add_axes([0.055, 0.12, 0.405, 0.77])
        cbar_ax = fig.add_axes([0.475, 0.19, 0.014, 0.62])
        profile_ax = fig.add_axes([0.57, 0.43, 0.37, 0.38])
        count_ax = fig.add_axes([0.57, 0.16, 0.37, 0.17], sharex=profile_ax)
        panel_a_x, panel_b_x = 0.035, 0.545
        panel_a_y = panel_b_y = 0.965
    else:
        map_ax = fig.add_axes([0.13, 0.47, 0.74, 0.48])
        cbar_ax = fig.add_axes([0.90, 0.52, 0.025, 0.38])
        profile_ax = fig.add_axes([0.14, 0.17, 0.76, 0.18])
        count_ax = fig.add_axes([0.14, 0.39, 0.76, 0.045], sharex=profile_ax)
        panel_a_x, panel_b_x = 0.055, 0.055
        panel_a_y, panel_b_y = 0.975, 0.46
    _draw_map(fig, map_ax, cbar_ax, data)
    _draw_radial_diagnostic(profile_ax, count_ax, data, counts_bottom=width_mm == 183.0)
    fig.text(
        panel_a_x, panel_a_y, "a", ha="left", va="top", fontsize=8.0, fontweight="bold"
    )
    fig.text(
        panel_b_x, panel_b_y, "b", ha="left", va="top", fontsize=8.0, fontweight="bold"
    )
    return fig


def render_butterfly_figure(
    *,
    observed: Any,
    qx: Any,
    qy: Any,
    valid_mask: Any = None,
    result: Mapping[str, Any],
    q_unit: str = "unknown",
    context: Any = None,
    display_scale: str = "log1p",
    width_mm: float = 183.0,
    dpi: int = 600,
) -> Figure:
    """Render a measurement-based figure without exporting files or importing Qt."""

    data = _prepare_inputs(
        observed=observed,
        qx=qx,
        qy=qy,
        valid_mask=valid_mask,
        result=result,
        q_unit=q_unit,
        context=context,
        display_scale=display_scale,
        width_mm=width_mm,
        dpi=dpi,
    )
    with _EXPORT_LOCK, mpl.rc_context(_FIGURE_RC):
        figure = _build_figure(data)
        figure.canvas.draw()
        return figure


def _software_version() -> str:
    try:
        return importlib.metadata.version("butterfly-saxs")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _q_axis_interpretation(data: Mapping[str, Any]) -> str:
    """Keep pixel-q labels explicitly non-physical at the export boundary."""

    unit = str(data["q_unit"]).strip().lower()
    if unit in {"pixel-q", "pixel_q"}:
        return "uncalibrated pixel-q; no physical period is inferred"
    return str(data["q_axis_interpretation"])


def _settings(data: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "figure_method_version": FIGURE_METHOD_VERSION,
        "software": {"package": "butterfly-saxs", "version": _software_version()},
        "analysis_method_version": str(
            data["result_safe"].get("method_version", "unknown")
        ),
        "figure": {
            "width_mm": data["width_mm"],
            "height_mm": data["height_mm"],
            "dpi": data["dpi"],
            "pixel_size": [
                int(data["width_mm"] / 25.4 * data["dpi"]),
                int(data["height_mm"] / 25.4 * data["dpi"]),
            ],
            "font_family": "sans-serif",
            "body_font_pt": 6.5,
            "panel_label_font_pt": 8.0,
            "panel_labels": ["a", "b"],
            "pdf_fonttype": 42,
            "svg_text": "editable text elements",
        },
        "q": {
            "unit": data["q_unit"],
            "axis_interpretation": _q_axis_interpretation(data),
            "intensity_unit": "raw input scale (caller-supplied; not inferred)",
            "image_coordinates": "supplied qx/qy mesh; no affine reconstruction or sorting",
            "radial_origin": [0.0, 0.0],
            "radial_aggregation": "raw observed intensity sum and unweighted mean across valid pixels in equal-width radial bins",
            "radial_profile_uses_display_transform": False,
            "radial_profile_color_clip_applied": False,
            "radial_profile_y_scale": "symlog",
            "radial_profile_linthresh_raw_intensity": data["radial_linthresh"],
            "radial_count_label": "n = valid source pixels per radial bin",
            "radial_bin_count": data["radial_bin_count"],
            "q_window": data["q_window"],
            "q_window_source": data["q_window_source"],
            "q_window_applied_to_figure": data["q_window_applied"],
            "q_window_fallback_reason": data["q_window_fallback_reason"],
            "display_crop": data["display_crop"],
        },
        "intensity_display": {
            "display_scale": data["display_scale"],
            "display_transform": data["display_label"],
            "asinh_intensity_scale": data["transform_scale"],
            "color_limits_transformed": [data["display_low"], data["display_high"]],
            "color_limits_percentiles_of_valid_pixels": data["color_clip_percentiles"],
            "color_clip_low_pixel_count": data["color_clip_low_count"],
            "color_clip_high_pixel_count": data["color_clip_high_count"],
            "negative_values_preserved_by_signed_transform": data["display_scale"]
            in {"log1p", "asinh"},
            "negative_values_preserved": True,
            "invalid_pixels_masked_in_display": True,
            "display_transform_changes_source_data": False,
        },
        "input": {
            "shape": list(data["observed_raw"].shape),
            "input_pixel_count": data["input_pixel_count"],
            "input_valid_mask_supplied": data["valid_mask_was_supplied"],
            "input_valid_mask_pixel_count": data["input_valid_mask_count"],
            "caller_selection_mask_supplied": data["valid_mask_was_supplied"],
            "caller_selection_mask_pixel_count": data["input_valid_mask_count"],
            "supplied_valid_mask_role": data["supplied_valid_mask_role"],
            "valid_pixel_count": data["valid_pixel_count"],
            "invalid_pixel_count": data["invalid_pixel_count"],
            "display_pixel_count": data["display_pixel_count"],
            "display_crop": data["display_crop"],
            "display_render_primitive": data["display_render_primitive"],
            "ridge_point_count": len(data["points"]),
            "ridge_point_overlay": {
                "input_count": len(data["points"]),
                "drawn_count": len(data["plot_points"]),
                "omitted_count": len(data["points"]) - len(data["plot_points"]),
                "omitted_by_reason": data["omitted_point_reasons"],
                "missing_pixel_coordinate_policy": "omit point from figure overlay; do not infer a source pixel from q coordinates",
            },
            "arcs_rendered_as_observed_segments": len(_supported_arc_segments(data)),
        },
        "scientific_boundary": {
            "drawn_geometry": "source-provided ridge points and fully supported ordered observed arcs",
            "candidate_ellipse_drawn": False,
            "candidate_fit_preserved_in_result_json": True,
            "scientific_acceptance_inferred": False,
            "mirrored_points_or_unknown_branch_sides_synthesized": False,
            "fitted_intensity_or_residual_map_inferred": False,
        },
        "context": data["context"],
    }


def _number_cell(value: Any) -> str:
    value = json_safe(value)
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
    return str(value)


def _write_ridge_csv(
    path: Path,
    points: Sequence[Mapping[str, Any]],
    point_overlay: Sequence[Mapping[str, Any]],
) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=_RIDGE_CSV_FIELDS, extrasaction="ignore"
        )
        writer.writeheader()
        for index, (point, overlay) in enumerate(
            zip(points, point_overlay, strict=True)
        ):
            writer.writerow(
                {
                    "point_index": index,
                    "point_id": _number_cell(point.get("point_id")),
                    "qx": _number_cell(point.get("qx")),
                    "qy": _number_cell(point.get("qy")),
                    "pixel_x": _number_cell(point.get("pixel_x")),
                    "pixel_y": _number_cell(point.get("pixel_y")),
                    "arc_id": _number_cell(point.get("arc_id")),
                    "branch_id": _number_cell(point.get("branch_id")),
                    "side": _number_cell(point.get("side")),
                    "accepted": _number_cell(point.get("accepted")),
                    "valid": _number_cell(point.get("valid")),
                    "display_status": _point_display_status(point),
                    "overlay_status": overlay["status"],
                    "overlay_reason": _number_cell(overlay.get("reason")),
                    "reason": _number_cell(point.get("reason")),
                    "source_json": json.dumps(
                        json_safe(point),
                        ensure_ascii=False,
                        sort_keys=True,
                        allow_nan=False,
                    ),
                }
            )


def _csv_number(value: Any) -> str:
    """Serialize one finite numeric value without changing its measurement."""

    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return ""
    if not np.isfinite(number):
        return ""
    return format(number, ".17g")


def _write_radial_profile_csv(path: Path, data: Mapping[str, Any]) -> None:
    """Write the raw, one-pixel-per-sample radial aggregation.

    ``count`` is the number of valid source pixels in each bin.  Empty bins
    retain their exact edges and center, while the undefined raw sum/mean are
    left blank rather than serializing a misleading zero or NaN.
    """

    edges = np.asarray(data["radial_edges"], dtype=np.float64)
    centers = np.asarray(data["radial_centers"], dtype=np.float64)
    raw_sum = np.asarray(data["radial_sum"], dtype=np.float64)
    raw_mean = np.asarray(data["radial_mean"], dtype=np.float64)
    counts = np.asarray(data["radial_counts"], dtype=np.int64)
    bin_count = len(centers)
    if len(edges) != bin_count + 1 or len(raw_sum) != bin_count:
        raise ValueError("radial aggregation arrays have inconsistent lengths")
    if len(raw_mean) != bin_count or len(counts) != bin_count:
        raise ValueError("radial aggregation arrays have inconsistent lengths")

    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=_RADIAL_PROFILE_FIELDS)
        writer.writeheader()
        for index in range(bin_count):
            count = int(counts[index])
            writer.writerow(
                {
                    "bin_index": index,
                    "q_unit": str(data["q_unit"]),
                    "edge_left": _csv_number(edges[index]),
                    "edge_right": _csv_number(edges[index + 1]),
                    "center": _csv_number(centers[index]),
                    "raw_sum": _csv_number(raw_sum[index]) if count else "",
                    "raw_mean": _csv_number(raw_mean[index]) if count else "",
                    "count": count,
                }
            )


def _figure_qa(
    data: Mapping[str, Any], *, model_supplied: bool
) -> dict[str, Any]:
    """Describe export integrity and scientific limits without accepting data."""

    dpi = int(data["dpi"])
    low_dpi = dpi < 300
    width_mm = float(data["width_mm"])
    height_mm = float(data["height_mm"])
    pixel_width = int(width_mm / 25.4 * dpi)
    pixel_height = int(height_mm / 25.4 * dpi)
    display_scale = str(data["display_scale"])
    qa_status = "WARN" if low_dpi else "PASS"
    result_safe = data["result_safe"]
    result_quality = result_safe.get("quality")
    result_quality = result_quality if isinstance(result_quality, Mapping) else {}
    source_scientific_status = str(
        result_safe.get("scientific_status")
        or result_quality.get("scientific_status")
        or "NOT_ACCEPTED"
    )
    dpi_check: dict[str, Any] = {
        "status": "WARN" if low_dpi else "PASS",
        "dpi": dpi,
        "minimum_for_raster_submission": 300,
    }
    if low_dpi:
        dpi_check["message"] = (
            "Raster DPI is below 300; use a higher DPI for publication raster output."
        )

    scientific_boundary = {
        "status": "recorded_only",
        "scientific_acceptance": "not_assessed",
        "source_scientific_status": source_scientific_status,
        "scientific_acceptance_inferred": False,
        "fabricated_points": False,
        "main_measurement_panel_candidate_ellipse_drawn": False,
        "candidate_assets_separately_labeled": True,
        "source_provided_ridge_points_only": True,
        "mirrored_or_synthesized_points": False,
        "candidate_fit_preserved_in_result_json": True,
        "model_overlay_is_diagnostic": True,
        "caller_model_supplied": bool(model_supplied),
        "fitted_intensity_or_residual_map_inferred": False,
        "statement": (
            "Figure QA records export integrity and provenance boundaries; it does "
            "not constitute scientific acceptance."
        ),
    }
    no_fabricated_points = {
        "status": "PASS",
        "value": True,
        "source_ridge_point_count": len(data["points"]),
        "drawn_source_ridge_point_count": len(data["plot_points"]),
        "omitted_source_ridge_point_count": len(data["points"])
        - len(data["plot_points"]),
        "mirrored_or_interpolated_points": False,
        "unsupported_points_retained_with_reasons": True,
    }
    editable_text = {
        "status": "configured",
        "verification": "matplotlib export settings recorded; vector files are not parsed by this QA step",
        "svg_fonttype": _FIGURE_RC["svg.fonttype"],
        "pdf_fonttype": _FIGURE_RC["pdf.fonttype"],
        "editable_svg_text": _FIGURE_RC["svg.fonttype"] == "none",
        "editable_pdf_text_setting": _FIGURE_RC["pdf.fonttype"] == 42,
        "text_as_paths": False,
    }
    physical_size = {
        "status": "configured",
        "verification": "derived from the requested Matplotlib canvas geometry; output files are not parsed by this QA step",
        "width_mm": width_mm,
        "height_mm": height_mm,
        "width_in": width_mm / 25.4,
        "height_in": height_mm / 25.4,
        "raster_pixel_width": pixel_width,
        "raster_pixel_height": pixel_height,
    }
    q_unit = str(data["q_unit"])
    q_units = {
        "status": "recorded",
        "unit": q_unit,
        "axis_interpretation": _q_axis_interpretation(data),
        "physical_period_conversion_performed": False,
        "pixel_q_or_unknown_is_not_physical_period": q_unit.strip().lower()
        in {"", "unknown", "pixel", "pixel_q", "pixel-q", "px"},
    }
    return {
        "schema_version": "butterfly.figure_qa.v1",
        "status": qa_status,
        "status_scope": "export_integrity_only",
        "warnings": [dpi_check["message"]] if low_dpi else [],
        "scope": {
            "measurement_panel": "butterfly_figure.*",
            "candidate_assets": "ellipse_only.* and comparison diagnostics",
            "model_assets": "model_comparison.* and fit overlay diagnostics when supplied",
            "all_bundle_assets_hashed_in_manifest": True,
        },
        "physical_size_mm": {
            "width": width_mm,
            "height": height_mm,
        },
        "dpi": dpi,
        "editable_text": editable_text,
        "q_unit": q_unit,
        "display_scale": display_scale,
        "radial_profile": {
            "file": "radial_profile.csv",
            "fields": list(_RADIAL_PROFILE_FIELDS),
            "aggregation": "raw observed intensity sum and unweighted mean",
            "raw_intensity_unit": "raw input scale (caller-supplied; not inferred)",
            "count_definition": "one valid source pixel contributes one count; not a statistical replicate count",
            "display_transform_used": False,
            "empty_bin_raw_sum_and_mean": "blank",
        },
        "scientific_acceptance": "not_assessed",
        "source_scientific_status": source_scientific_status,
        "no_fabricated_points": True,
        "fabricated_points": False,
        "nature_reference": NATURE_FIGURE_GUIDE_URL,
        "target_specification": {
            "reference": NATURE_FIGURE_GUIDE_URL,
            "supported_width_mm": [89.0, 183.0],
            "selected_width_mm": width_mm,
            "body_font_pt": 6.5,
            "panel_label_font_pt": 8.0,
            "editable_text_requested": True,
            "acceptance_claim": False,
        },
        "figure": {
            "physical_size": physical_size,
            "dpi": dpi_check,
            "display_scale": display_scale,
            "q_unit": q_unit,
        },
        "checks": {
            "physical_size": physical_size,
            "dpi": dpi_check,
            "editable_text": editable_text,
            "q_units": q_units,
            "scientific_boundary": scientific_boundary,
            "no_fabricated_points": no_fabricated_points,
        },
        "scientific_boundary": scientific_boundary,
    }


def _html_link(path: Path, *, label: str | None = None) -> str:
    """Return a local-only link using a basename, never a private path."""

    name = html.escape(path.name, quote=True)
    text = html.escape(label or path.name)
    return f'<a href="{name}" download>{text}</a>'


def _write_index_html(
    path: Path,
    *,
    data: Mapping[str, Any],
    outputs: Mapping[str, Path],
    qa: Mapping[str, Any],
    caption: str,
) -> None:
    """Write a dependency-free, local figure-bundle browser."""

    # Keep paths that will be written later in the same transaction (notably
    # manifest.json) so the browser exposes every promised download link.
    available = {item.name: item for item in outputs.values()}
    measured_names = (
        "butterfly_figure.png",
        "butterfly_figure.svg",
        "butterfly_figure.pdf",
        "butterfly_figure.tiff",
        "measured_only.png",
        "measured_only.svg",
        "measured_only.pdf",
        "measured_only.tiff",
    )
    candidate_files = {
        path.name
        for key, path in outputs.items()
        if key.startswith("ellipse_only")
        or key.startswith("ellipse_curves")
        or key.startswith("comparison")
        or key.startswith("geometry_overlay")
        or key in {"point_residuals_csv", "normal_profiles_csv"}
    }
    model_files = {
        path.name
        for key, path in outputs.items()
        if key.startswith("model_comparison")
        or key.startswith("intensity_model_overlay")
        or key.startswith("fit_")
    }
    sector_files = {
        path.name
        for key, path in outputs.items()
        if key.startswith("sector_")
    }
    measured_files = [name for name in measured_names if name in available]

    def links(names: Sequence[str] | set[str]) -> str:
        selected = [available[name] for name in names if name in available]
        if not selected:
            return '<span class="muted">No file in this bundle.</span>'
        return " ".join(_html_link(item) for item in selected)

    def preview(name: str, alt: str) -> str:
        image = available.get(name)
        if image is None or not image.is_file():
            return '<div class="preview muted">Preview unavailable</div>'
        return (
            f'<a href="{html.escape(image.name, quote=True)}">'
            f'<img src="{html.escape(image.name, quote=True)}" alt="{html.escape(alt, quote=True)}" '
            'loading="lazy"></a>'
        )

    result_safe = data["result_safe"]
    measurement_status = html.escape(str(result_safe.get("measurement_status", "unknown")))
    candidate_status = html.escape(
        str((result_safe.get("candidate_fit") or {}).get("status", "not_available"))
    )
    quality = result_safe.get("quality")
    quality = quality if isinstance(quality, Mapping) else {}
    scientific_status = html.escape(
        str(
            result_safe.get("scientific_status")
            or quality.get("scientific_status")
            or "NOT_ACCEPTED"
        )
    )
    caption_text = html.escape(caption.strip())
    q_unit = html.escape(str(data["q_unit"]))
    qa_status = html.escape(str(qa["status"]))
    measured_files_text = links(measured_files)
    candidate_files_text = links(sorted(candidate_files))
    model_files_text = links(sorted(model_files))
    sector_files_text = links(sorted(sector_files))
    has_sector_figures = bool(sector_files)
    annular_files = tuple(name for name in available if name.startswith("annular_"))
    annular_files_text = links(annular_files)
    provenance_names = (
        "source_data.npz",
        "radial_profile.csv",
        "ridge_points.csv",
        "result.json",
        "settings.json",
        "figure_qa.json",
        "caption.txt",
        "manifest.json",
        "sector_qchi.svg",
        "sector_qchi.pdf",
        "sector_qchi.png",
        "sector_qchi.tiff",
        "sector_profiles.svg",
        "sector_profiles.pdf",
        "sector_profiles.png",
        "sector_profiles.tiff",
        "sector_profiles.csv",
        "sector_peaks.csv",
        "sector_profiles.npz",
        "sector_caption.txt",
        "sector_manifest.json",
    )
    provenance_links = links(provenance_names)
    html_text = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Butterfly SAXS figure bundle</title>
<style>
:root {{ color-scheme: light; font-family: Arial, Helvetica, sans-serif; color: #17212b; background: #f7f8fa; }}
body {{ max-width: 1120px; margin: 0 auto; padding: 28px; line-height: 1.45; }}
h1 {{ margin: 0 0 8px; font-size: 1.65rem; }} h2 {{ margin-top: 28px; font-size: 1.15rem; }}
.status {{ display: inline-block; padding: 3px 9px; border-radius: 999px; background: #f5d36b; font-size: .78rem; font-weight: 700; letter-spacing: .03em; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; }}
.card {{ background: white; border: 1px solid #d9dee5; border-radius: 8px; padding: 14px; box-shadow: 0 1px 2px #0000000b; }}
.card h3 {{ margin: 0 0 6px; font-size: 1rem; }}
.preview {{ min-height: 120px; display: grid; place-items: center; background: #edf0f3; border-radius: 5px; margin: 9px 0; overflow: hidden; }}
.preview img {{ width: 100%; max-height: 240px; object-fit: contain; display: block; }}
.links {{ display: flex; flex-wrap: wrap; gap: 8px 12px; font-size: .85rem; }}
a {{ color: #075a9e; }} .muted {{ color: #5f6b76; font-size: .88rem; }}
table {{ border-collapse: collapse; background: white; min-width: 320px; }} th, td {{ border: 1px solid #d9dee5; padding: 5px 9px; text-align: left; font-size: .88rem; }}
th {{ background: #eef2f5; }} pre {{ white-space: pre-wrap; background: #fff; border: 1px solid #d9dee5; padding: 12px; border-radius: 6px; font-size: .84rem; }}
</style>
</head>
<body>
<h1>Butterfly SAXS measurement figure bundle / 蝴蝶花样测量图包</h1>
<p><span class="status">Engineering export · scientific acceptance not assessed · QA {qa_status}</span></p>
<p>This offline browser uses only local relative files. 实测、候选几何和模型/诊断资产分开标记；图包 QA 不等于科学验收。</p>
<table aria-label="Figure bundle summary">
<tr><th>Measured status</th><td>{measurement_status}</td></tr>
<tr><th>Candidate geometry status</th><td>{candidate_status}</td></tr>
<tr><th>Scientific status</th><td>{scientific_status}</td></tr>
<tr><th>q unit</th><td>{q_unit}</td></tr>
<tr><th>Display scale</th><td>{html.escape(str(data["display_scale"]))}</td></tr>
<tr><th>Physical size</th><td>{data["width_mm"]} × {data["height_mm"]} mm</td></tr>
<tr><th>Raster resolution</th><td>{data["dpi"]} DPI</td></tr>
</table>
{f'''<h2>Annular flower trajectories · 逐环花瓣轨迹</h2>
<div class="grid"><article class="card"><h3>Fixed-q angular maxima</h3><div class="preview">{preview("annular_qchi.png", "Azimuthal peaks on successive q rings")}</div></article>
<article class="card"><h3>I(χ) on individual rings</h3><div class="preview">{preview("annular_profiles.png", "Raw angular profiles of individual annuli")}</div></article></div>
<p class="links">{annular_files_text}</p><p class="muted">Each ring supplies observed angular peaks to the outward petal tracks. Missing lobes remain missing. Ring q is a sampling coordinate, not a measured radial reflection position.</p>''' if annular_files else ""}
<h2>Measured / observed · 实测</h2>
<div class="grid"><article class="card"><h3>Native measurement figure</h3><div class="preview">{preview("butterfly_figure.png", "Measured butterfly SAXS figure")}</div><div class="links">{measured_files_text}</div><p class="muted">Observed intensity, source-provided ridge points, supported observed segments, and raw radial profile.</p></article></div>
<h2>Candidate geometry · 候选</h2>
<div class="grid"><article class="card"><h3>Candidate ellipse diagnostics</h3><div class="preview">{preview("ellipse_only.png", "Candidate ellipse diagnostic")}</div><div class="links">{candidate_files_text}</div><p class="muted">Candidate fits remain candidates and are not presented as accepted measurements.</p></article></div>
<h2>Model / diagnostic · 模型</h2>
<div class="grid"><article class="card"><h3>Caller model and fit diagnostics</h3><div class="preview">{preview("model_comparison.png", "Caller model comparison diagnostic")}</div><div class="links">{model_files_text}</div><p class="muted">Model or residual assets are diagnostic comparisons and do not add measured points.</p></article></div>
{f'''<h2>Sector-integrated primary peaks · 扇区积分主峰</h2>
<div class="grid"><article class="card"><h3>Raw I(q, χ) sector field</h3><div class="preview">{preview("sector_qchi.png", "Raw sector-integrated q-chi intensity")}</div></article><article class="card"><h3>Representative I(q | χ)</h3><div class="preview">{preview("sector_profiles.png", "Representative sector-integrated profiles")}</div></article></div>
<p class="links">{sector_files_text}</p><p class="muted">The q–χ field and I(q|χ) profiles are copied from the existing sector measurement result. Raw means remain separate from locator-only smoothing; blank bins are missing support, overlapping sectors are correlated, and candidates are not scientific acceptance.</p>''' if has_sector_figures else ""}
<h2>Pixel-brightness ancillary diagnostics · 像素亮点附属诊断</h2>
<div class="grid"><article class="card"><h3>Pixel landmark maps</h3><div class="preview">{preview("peak_map.png", "Pixel-brightness ancillary diagnostic")}</div><div class="links">{links(tuple(name for name in ("peak_map.png", "peak_diagnostics.png", "peak_zooms.png") if name in available))}</div><p class="muted">G/P labels are pixel-brightness diagnostics only; they are not sector-integrated peak positions and must not be mixed with the I(q|χ) candidates.</p></article></div>
<h2>Source data and caption · 源数据与图注</h2>
<p class="links">{provenance_links}</p>
<details><summary>Figure caption</summary><pre>{caption_text}</pre></details>
</body>
</html>
"""
    path.write_text(html_text, encoding="utf-8", newline="\n")


def _caption(data: Mapping[str, Any]) -> str:
    unit = data["q_unit"]
    display_scale = str(data["display_scale"])
    if display_scale == "log1p":
        display_description = (
            "The image display uses the signed log1p transform "
            "sign(I) * log1p(abs(I)); this transform preserves the sign of negative values."
        )
    elif display_scale == "asinh":
        display_description = (
            "The image display uses the asinh(I) transform; radial values remain "
            "the signed raw observed intensities."
        )
    else:
        display_description = (
            "The image display uses the linear, untransformed intensity; radial values "
            "remain the signed raw observed intensities."
        )
    return (
        "Butterfly SAXS measurement figure. (a) Observed two-dimensional intensity on the supplied qx/qy "
        f"coordinate mesh ({unit}), with source-provided ridge points and supported observed arc segments. "
        "Accepted and rejected points retain their source status; unspecified status is shown separately. "
        "No opposite quadrants or branch sides are synthesized. (b) Unweighted mean of the raw observed "
        "intensity in the caller-supplied raw input scale across valid source pixels in equal-width bins of q radius from the supplied q=(0, 0) origin; bars show n, the number "
        "of valid source pixels per bin. The radial profile uses a recorded symlog display scale but retains signed raw means. "
        "Image color limits use the 0.5th–99.5th percentiles of valid pixels. "
        f"{display_description} Candidate fits are retained in result.json but are not drawn or "
        "represented as scientifically accepted. No fitted intensity or residual map is inferred from ellipse geometry. "
        "The caller-supplied selection mask and coordinate arrays are preserved in source_data.npz. "
        "Ridge points without valid source-pixel support are omitted from the overlay but retained with omission "
        "reasons in ridge_points.csv; a missing source pixel is never inferred from q coordinates."
    )


def _write_source_data(path: Path, data: Mapping[str, Any]) -> None:
    np.savez_compressed(
        path,
        observed=data["observed_raw"],
        qx=data["qx_raw"],
        qy=data["qy_raw"],
        # ``valid_mask`` remains as a compatibility alias; it is not assumed
        # to be a detector-only mask.
        valid_mask=data["source_valid_mask"],
        supplied_valid_mask=data["source_valid_mask"],
        valid_mask_mask=data["source_valid_mask_mask"],
        supplied_valid_mask_role=np.asarray(data["supplied_valid_mask_role"]),
        effective_valid_mask=data["effective_valid"],
        display_valid_mask=data["display_valid"],
        observed_mask=data["observed_mask"],
        qx_mask=data["qx_mask"],
        qy_mask=data["qy_mask"],
        q_unit=np.asarray(data["q_unit"]),
        radial_q_edges=data["radial_edges"],
        radial_intensity_sum_raw=data["radial_sum"],
        radial_intensity_mean_raw=data["radial_mean"],
        radial_valid_pixel_counts=data["radial_counts"],
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(
            json_safe(payload),
            stream,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        stream.write("\n")


def export_butterfly_figure(
    target: str | os.PathLike[str],
    *,
    observed: Any,
    qx: Any,
    qy: Any,
    valid_mask: Any = None,
    result: Mapping[str, Any],
    model: Any = None,
    q_unit: str = "unknown",
    context: Any = None,
    display_scale: str = "log1p",
    width_mm: float = 183.0,
    dpi: int = 600,
    cancel_event: Any = None,
    progress: Callable[[int, str], Any] | None = None,
) -> dict[str, Path]:
    """Export a complete, non-overwriting native figure bundle.

    ``target`` is a new directory. ``progress`` receives ``(percent, phase)``
    with integer percentages from 0 to 100. Cancellation raises the shared
    :class:`~butterfly_saxs.cancellation.AnalysisCancelled`; all staged files
    are removed unless the complete directory has been atomically published.
    """

    raise_if_cancelled(cancel_event, "butterfly-figure:validate")
    target_path = Path(target).expanduser().resolve(strict=False)
    if target_path.exists():
        raise FileExistsError(f"figure target already exists: {target_path}")
    if not target_path.name:
        raise ValueError("target must name a new output directory")
    _notify(progress, 0, "validate")
    data = _prepare_inputs(
        observed=observed,
        qx=qx,
        qy=qy,
        valid_mask=valid_mask,
        result=result,
        q_unit=q_unit,
        context=context,
        display_scale=display_scale,
        width_mm=width_mm,
        dpi=dpi,
    )
    raise_if_cancelled(cancel_event, "butterfly-figure:render")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(prefix=f".{target_path.name}.staging-", dir=target_path.parent)
    )
    committed = False
    output_names = {
        "svg": "butterfly_figure.svg",
        "pdf": "butterfly_figure.pdf",
        "tiff": "butterfly_figure.tiff",
        "png": "butterfly_figure.png",
        "source_data": "source_data.npz",
        "ridge_points": "ridge_points.csv",
        "result": "result.json",
        "settings": "settings.json",
        "caption": "caption.txt",
        "radial_profile": "radial_profile.csv",
        "figure_qa": "figure_qa.json",
        "index": "index.html",
        "manifest": "manifest.json",
    }
    staged = {key: stage / name for key, name in output_names.items()}
    try:
        _notify(progress, 5, "render")
        raise_if_cancelled(cancel_event, "butterfly-figure:render")
        with _EXPORT_LOCK, mpl.rc_context(_FIGURE_RC):
            figure = _build_figure(data)
            figure.canvas.draw()
            raise_if_cancelled(cancel_event, "butterfly-figure:save-vector")
            figure.savefig(
                staged["svg"], format="svg", facecolor="white", transparent=False
            )
            figure.savefig(
                staged["pdf"], format="pdf", facecolor="white", transparent=False
            )
            raise_if_cancelled(cancel_event, "butterfly-figure:save-raster")
            figure.savefig(
                staged["tiff"],
                format="tiff",
                dpi=data["dpi"],
                facecolor="white",
                transparent=False,
            )
            figure.savefig(
                staged["png"],
                format="png",
                dpi=data["dpi"],
                facecolor="white",
                transparent=False,
            )
            figure.clear()
        _notify(progress, 35, "source_data")
        raise_if_cancelled(cancel_event, "butterfly-figure:source-data")
        _write_source_data(staged["source_data"], data)
        _write_ridge_csv(staged["ridge_points"], data["points"], data["point_overlay"])
        _write_json(staged["result"], data["result_safe"])
        _notify(progress, 55, "comparison")
        raise_if_cancelled(cancel_event, "butterfly-figure:comparison")
        from .butterfly_comparison import export_butterfly_comparison

        comparison_outputs, comparison_metadata = export_butterfly_comparison(
            stage,
            data=data,
            model=model,
            cancel_event=cancel_event,
        )
        from .fit_overlays import export_fit_overlays

        overlay_outputs, overlay_metadata = export_fit_overlays(stage, data=data, cancel_event=cancel_event)
        from .peak_landmarks import compute_peak_landmarks
        from .peak_landmark_figures import export_peak_landmark_figures
        from .butterfly_ridge import _apply_edits

        stored_landmarks = data["result_safe"].get("peak_landmarks") or {}
        hint = data["result_safe"].get("diagnostics", {}).get("first_order_q_hint", {})
        signal_window = hint.get("band") if hint.get("selection_status") == "selected" else None
        peak_options = dict(stored_landmarks.get("options") or {})
        if signal_window is not None:
            peak_options["signal_q_window_origin"] = (
                f"first_order_q_hint:{hint.get('q_star')}; method={hint.get('selection_method')}"
            )
        # Replay the same domain edits as the public measurement adapter.
        # Include polygons may restore earlier exclusions, never input-invalid
        # pixels or pixels outside the analysis q window.
        peak_valid = data["effective_valid"].copy()
        if data.get("q_window") is not None:
            radius = np.hypot(data["qx_raw"], data["qy_raw"])
            peak_valid &= (radius >= data["q_window"][0]) & (radius <= data["q_window"][1])
        peak_valid, peak_edits, _ = _apply_edits(
            peak_valid, data["qx_raw"], data["qy_raw"], data["result_safe"].get("edits", [])
        )
        landmarks = compute_peak_landmarks(
            data["observed_raw"], data["qx_raw"], data["qy_raw"],
            valid_mask=peak_valid, q_window=data.get("q_window"),
            signal_q_window=signal_window, q_unit=data["q_unit"], model=model,
            options=peak_options, cancel_event=cancel_event,
        )
        landmarks["domain"]["applied_polygon_edits"] = peak_edits
        peak_outputs, peak_metadata = export_peak_landmark_figures(
            stage, data={"observed": data["observed_raw"], "qx": data["qx_raw"], "qy": data["qy_raw"],
                         "valid_mask": peak_valid, "q_unit": data["q_unit"],
                         "context": data["context"], "width_mm": data["width_mm"], "dpi": data["dpi"],
                         "display_scale": data["display_scale"]},
            landmarks=landmarks, model=model, cancel_event=cancel_event,
        )
        sector_outputs: dict[str, Path] = {}
        sector_metadata: dict[str, Any] = {}
        # Sector figures are a pure export of the caller's existing sector
        # measurement.  They are intentionally kept separate from the
        # curvature/landmark calculations above: no image re-integration or
        # replacement peak selection occurs in the figure writer.
        if "sector_peaks" in result:
            sector_result = result.get("sector_peaks")
            if not isinstance(sector_result, Mapping):
                raise TypeError("result.sector_peaks must be a mapping when supplied")
            _notify(progress, 66, "sector-peaks")
            raise_if_cancelled(cancel_event, "butterfly-figure:sector-peaks")
            from .sector_peak_figures import export_sector_peak_figures

            sector_outputs, sector_metadata = export_sector_peak_figures(
                stage,
                data={
                    "q_unit": data["q_unit"],
                    "context": data["context"],
                    "width_mm": data["width_mm"],
                    "dpi": data["dpi"],
                },
                result=sector_result,
                cancel_event=cancel_event,
            )
        figure_settings = _settings(data)
        annular_outputs: dict[str, Path] = {}
        annular_metadata: dict[str, Any] = {}
        if "annular_peaks" in result:
            from .annular_peak_figures import export_annular_peak_figures

            annular_outputs, annular_metadata = export_annular_peak_figures(
                stage, data={"q_unit": data["q_unit"], "context": data["context"],
                             "width_mm": data["width_mm"], "dpi": data["dpi"]},
                result=result["annular_peaks"], cancel_event=cancel_event,
            )
            figure_settings["annular_peaks"] = annular_metadata
        figure_settings["comparison"] = comparison_metadata
        figure_settings["fit_sources"] = overlay_metadata
        figure_settings["peak_landmarks"] = peak_metadata
        if sector_metadata:
            figure_settings["sector_peaks"] = sector_metadata
        _write_json(staged["settings"], figure_settings)
        caption = _caption(data)
        caption += "\n\nComparison assets: see comparison_caption.txt."
        if model is not None:
            caption += " A model-comparison plate is also supplied from the caller-provided model array."
        if sector_outputs:
            caption += "\n\nSector-integrated primary peaks: see sector_caption.txt. Pixel-brightness landmark diagnostics remain ancillary and are not sector peak positions."
        if annular_outputs:
            caption += "\n\nFixed-q annular trajectories: see annular_caption.txt. Angular peak positions trace each petal outwards; the sampled ring radii are not radial reflection peaks."
        staged["caption"].write_text(caption + "\n", encoding="utf-8")
        _notify(progress, 68, "package")
        raise_if_cancelled(cancel_event, "butterfly-figure:package")
        _write_radial_profile_csv(staged["radial_profile"], data)
        figure_qa = _figure_qa(data, model_supplied=model is not None)
        _write_json(staged["figure_qa"], figure_qa)
        all_staged = {
            **staged,
            **comparison_outputs,
            **overlay_outputs,
            **peak_outputs,
            **sector_outputs,
            **annular_outputs,
        }
        _write_index_html(
            staged["index"],
            data=data,
            outputs=all_staged,
            qa=figure_qa,
            caption=caption,
        )
        _notify(progress, 72, "hash")
        raise_if_cancelled(cancel_event, "butterfly-figure:hash")
        file_hashes = {
            path.name: _sha256(path)
            for key, path in all_staged.items()
            if key != "manifest"
        }
        manifest = {
            "manifest_schema": "butterfly-figure-export-v1",
            "figure_method_version": FIGURE_METHOD_VERSION,
            "comparison_method_version": "butterfly-comparison-v1",
            "q_unit": data["q_unit"],
            "input_shape": list(data["observed_raw"].shape),
            "valid_pixel_count": data["valid_pixel_count"],
            "display_pixel_count": data["display_pixel_count"],
            "display_crop": data["display_crop"],
            "q_window": data["q_window"],
            "ridge_point_count": len(data["points"]),
            "ridge_point_overlay_count": len(data["plot_points"]),
            "radial_profile_file": "radial_profile.csv",
            "radial_profile_fields": list(_RADIAL_PROFILE_FIELDS),
            "figure_qa_file": "figure_qa.json",
            "index_file": "index.html",
            "figure_qa_status": figure_qa["status"],
            "atomic_publish": {
                "staging_directory": True,
                "same_volume_rename": True,
                "target_must_not_exist": True,
            },
            "files": list(file_hashes),
            "sha256": file_hashes,
            "manifest_excluded_from_own_sha256": True,
        }
        if sector_outputs:
            manifest["sector_manifest_file"] = "sector_manifest.json"
            manifest["sector_peak_figure_method_version"] = sector_metadata.get(
                "method_version"
            )
        if annular_outputs:
            manifest["annular_manifest_file"] = "annular_manifest.json"
            manifest["annular_peak_figure_method_version"] = annular_metadata.get("method_version")
        _write_json(staged["manifest"], manifest)
        _notify(progress, 100, "publish")
        raise_if_cancelled(cancel_event, "butterfly-figure:publish")
        if target_path.exists():
            raise FileExistsError(f"figure target already exists: {target_path}")
        # The staging directory is a sibling, so this rename is an atomic
        # same-volume publish on supported local filesystems.
        os.rename(stage, target_path)
        committed = True
        return {key: target_path / path.name for key, path in all_staged.items()}
    finally:
        if not committed and stage.exists():
            try:
                shutil.rmtree(stage)
            except OSError:
                pass


__all__ = [
    "FIGURE_METHOD_VERSION",
    "NATURE_FIGURE_GUIDE_URL",
    "export_butterfly_figure",
    "render_butterfly_figure",
]
