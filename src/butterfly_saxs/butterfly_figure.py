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


def _notify(
    progress: Callable[[int, str], Any] | None, percent: int, phase: str
) -> None:
    if progress is not None:
        progress(int(percent), str(phase))


def _draw_radial_diagnostic(
    ax_profile: Any, ax_count: Any, data: Mapping[str, Any], *, counts_bottom: bool
) -> None:
    centers = data["radial_centers"]
    means = data["radial_mean"]
    counts = data["radial_counts"]
    ax_profile.plot(
        centers, means, color="#0072b2", linewidth=0.85, marker="o", markersize=1.6
    )
    ax_profile.set_yscale("symlog", linthresh=data["radial_linthresh"])
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
            "axis_interpretation": data["q_axis_interpretation"],
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
            == "log1p",
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


def _caption(data: Mapping[str, Any]) -> str:
    unit = data["q_unit"]
    return (
        "Butterfly SAXS measurement figure. (a) Observed two-dimensional intensity on the supplied qx/qy "
        f"coordinate mesh ({unit}), with source-provided ridge points and supported observed arc segments. "
        "Accepted and rejected points retain their source status; unspecified status is shown separately. "
        "No opposite quadrants or branch sides are synthesized. (b) Pixel-weighted mean of the raw observed "
        "intensity in equal-width bins of q radius from the supplied q=(0, 0) origin; bars show n, the number "
        "of valid source pixels per bin. The radial profile uses a recorded symlog display scale but retains signed raw means. "
        "Image color limits use the 0.5th–99.5th percentiles of valid pixels; signed log1p preserves negative "
        "values in the display transform. Candidate fits are retained in result.json but are not drawn or "
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
        figure_settings = _settings(data)
        figure_settings["comparison"] = comparison_metadata
        figure_settings["fit_sources"] = overlay_metadata
        figure_settings["peak_landmarks"] = peak_metadata
        _write_json(staged["settings"], figure_settings)
        caption = _caption(data)
        caption += "\n\nComparison assets: see comparison_caption.txt."
        if model is not None:
            caption += " A model-comparison plate is also supplied from the caller-provided model array."
        staged["caption"].write_text(caption + "\n", encoding="utf-8")
        _notify(progress, 72, "hash")
        raise_if_cancelled(cancel_event, "butterfly-figure:hash")
        all_staged = {**staged, **comparison_outputs, **overlay_outputs, **peak_outputs}
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
            "files": list(file_hashes),
            "sha256": file_hashes,
            "manifest_excluded_from_own_sha256": True,
        }
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
    "export_butterfly_figure",
    "render_butterfly_figure",
]
