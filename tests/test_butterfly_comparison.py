from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import matplotlib as mpl
from PIL import Image

from butterfly_saxs.butterfly_figure import export_butterfly_figure
from butterfly_saxs.butterfly_figure import _prepare_inputs
from butterfly_saxs.butterfly_figure import _FIGURE_RC
from butterfly_saxs.butterfly_comparison import (
    _candidate_curves,
    _comparison_figure,
    _ellipse_only_figure,
    _model_comparison_figure,
)
from butterfly_saxs.ellipse import EllipseGeometry


def _pdf_box(path: Path) -> tuple[float, float]:
    match = re.search(
        rb"/MediaBox\s*\[\s*([0-9.+-]+)\s+([0-9.+-]+)\s+([0-9.+-]+)\s+([0-9.+-]+)",
        path.read_bytes(),
    )
    assert match is not None
    return float(match.group(3)), float(match.group(4))


def _frame(
    shape: tuple[int, int] = (32, 36),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows, cols = np.indices(shape, dtype=np.float64)
    qx = (cols - (shape[1] - 1) / 2.0) * 0.005 + 0.0002 * np.sin(rows / 3.0)
    qy = (rows - (shape[0] - 1) / 2.0) * 0.005 + 0.0002 * np.cos(cols / 4.0)
    angle = np.arctan2(qy, qx)
    radius = np.hypot(qx, qy)
    target = 0.075 + 0.012 * np.cos(2.0 * angle)
    observed = 2.0 + 25.0 * np.exp(-0.5 * ((radius - target) / 0.004) ** 2)
    return observed, qx, qy


def _candidate_result(qx: np.ndarray, qy: np.ndarray) -> dict:
    point_specs = ((8, 8, 0), (11, 10, 1), (16, 18, 0), (21, 24, 1))
    points = []
    diagnostics = []
    for index, (row, col, branch) in enumerate(point_specs):
        point_id = f"p{index}"
        point = {
            "point_id": point_id,
            "qx": float(qx[row, col]),
            "qy": float(qy[row, col]),
            "pixel_x": float(col),
            "pixel_y": float(row),
            "arc_id": branch,
            "branch_id": branch,
            "side": "upper" if index % 2 == 0 else "lower",
            "accepted": True,
            "valid": True,
        }
        points.append(point)
        diagnostics.append(
            {
                "point_id": point_id,
                "qx": point["qx"],
                "qy": point["qy"],
                "arc_id": branch,
                "branch_id": branch,
                "side": point["side"],
                "used": True,
                "projection_valid": True,
                "projection_t": 0.1 * (index + 1),
                "projection_residual_q": (-1.0) ** index * (index + 1) * 1e-4,
                "normal_residual_q": (-1.0) ** index * (index + 1) * 1.2e-4,
                "projection_distance_q": (index + 1) * 1e-4,
            }
        )
    profiles = {
        "p0": {
            "point_id": "p0",
            "valid": True,
            "offset_q": [-0.02, -0.01, 0.0, 0.01, 0.02],
            "raw_intensity": [2.0, 4.0, 8.0, 4.5, 2.2],
            "fit_intensity": [2.1, 4.2, 7.7, 4.2, 2.1],
        }
    }
    return {
        "method_version": "fixture-trace-v1",
        "measurement_status": "supported",
        "points": points,
        "arcs": [],
        "profiles": profiles,
        "candidate_fit": {
            "success": True,
            "status": "ok",
            "a": 0.08,
            "b": 0.032,
            "axis_ratio": 0.4,
            "center_qx": 0.0,
            "center_qy": 0.0,
            "theta_deg": 20.0,
            "reference_axis_deg": 0.0,
            "quality": {"status": "WARN", "scientific_status": "NOT_ACCEPTED"},
            "point_diagnostics": diagnostics,
            "ellipses": [
                {
                    "a": 0.08,
                    "b": 0.032,
                    "axis_ratio": 0.4,
                    "center": [0.0, 0.0],
                    "angle_deg": 20.0,
                    "branch_id": 0,
                    "fit_branch_id": 0,
                },
                {
                    "a": 0.08,
                    "b": 0.032,
                    "axis_ratio": 0.4,
                    "center": [0.0, 0.0],
                    "angle_deg": -20.0,
                    "branch_id": 1,
                    "fit_branch_id": 1,
                },
            ],
        },
        "quantitative_parameters": {
            "axis_ratio": {
                "value": None,
                "candidate_value": 0.4,
                "status": "undetermined",
            }
        },
        "diagnostics": {"reference_axis_deg": 0.0},
    }


def test_candidate_curves_use_shared_ellipse_geometry_and_are_separately_exported(
    tmp_path: Path,
) -> None:
    observed, qx, qy = _frame()
    result = _candidate_result(qx, qy)
    observed_before, qx_before, qy_before = (
        array.copy() for array in (observed, qx, qy)
    )
    target = tmp_path / "candidate-comparison"
    outputs = export_butterfly_figure(
        target,
        observed=observed,
        qx=qx,
        qy=qy,
        valid_mask=np.ones(observed.shape, dtype=bool),
        result=result,
        q_unit="nm^-1",
        width_mm=183,
        dpi=72,
    )

    expected_extra = {
        "comparison_svg",
        "comparison_pdf",
        "comparison_tiff",
        "comparison_png",
        "comparison_caption",
        "ellipse_curves_csv",
        "ellipse_curves_npz",
        "ellipse_only_svg",
        "ellipse_only_pdf",
        "ellipse_only_tiff",
        "ellipse_only_png",
        "ellipse_only_caption",
        "point_residuals_csv",
        "normal_profiles_csv",
    }
    assert expected_extra <= set(outputs)
    assert all(
        outputs[key].is_file() and outputs[key].stat().st_size > 0
        for key in expected_extra
    )
    np.testing.assert_array_equal(observed, observed_before)
    np.testing.assert_array_equal(qx, qx_before)
    np.testing.assert_array_equal(qy, qy_before)

    with np.load(outputs["ellipse_curves_npz"], allow_pickle=False) as curves:
        assert curves["q_points"].shape == (2, 361, 2)
        assert list(curves["branch_id"]) == ["0", "1"]
        geometry = EllipseGeometry(0.0, 0.0, 0.08, 0.4, math.radians(20.0))
        np.testing.assert_allclose(
            curves["q_points"][0], geometry.point(curves["phi_rad"])
        )
    with outputs["ellipse_curves_csv"].open(encoding="utf-8-sig", newline="") as stream:
        curve_rows = list(csv.DictReader(stream))
    assert len(curve_rows) == 722
    assert {row["branch_id"] for row in curve_rows} == {"0", "1"}
    assert {row["curve_role"] for row in curve_rows} == {
        "candidate_full_curve_not_observed_support"
    }
    assert "not scientifically accepted" in outputs["ellipse_only_caption"].read_text(
        encoding="utf-8"
    )
    svg_root = ET.parse(outputs["ellipse_only_svg"]).getroot()
    assert "<text" in outputs["ellipse_only_svg"].read_text(encoding="utf-8")
    assert float(svg_root.attrib["width"].removesuffix("pt")) == pytest.approx(
        183.0 / 25.4 * 72.0, abs=0.02
    )
    comparison_svg_root = ET.parse(outputs["comparison_svg"]).getroot()
    assert float(
        comparison_svg_root.attrib["width"].removesuffix("pt")
    ) == pytest.approx(183.0 / 25.4 * 72.0, abs=0.02)
    for key, height_mm in (
        ("comparison_pdf", 150.0),
        ("ellipse_only_pdf", 94.0),
    ):
        media_width, media_height = _pdf_box(outputs[key])
        assert media_width == pytest.approx(183.0 / 25.4 * 72.0, abs=0.02)
        assert media_height == pytest.approx(height_mm / 25.4 * 72.0, abs=0.02)
        assert b"/FontFile2" in outputs[key].read_bytes()
    expected_comparison_pixels = (
        int(183.0 / 25.4 * 72),
        int(150.0 / 25.4 * 72),
    )
    expected_ellipse_pixels = (
        int(183.0 / 25.4 * 72),
        int(94.0 / 25.4 * 72),
    )
    for key in ("comparison_png", "comparison_tiff"):
        with Image.open(outputs[key]) as raster:
            assert raster.size == expected_comparison_pixels
    for key in ("ellipse_only_png", "ellipse_only_tiff"):
        with Image.open(outputs[key]) as raster:
            assert raster.size == expected_ellipse_pixels

    with outputs["point_residuals_csv"].open(
        encoding="utf-8-sig", newline=""
    ) as stream:
        residual_rows = list(csv.DictReader(stream))
    assert len(residual_rows) == len(result["points"])
    assert float(residual_rows[0]["projection_residual_q"]) == pytest.approx(1e-4)
    settings = json.loads(outputs["settings"].read_text(encoding="utf-8"))
    assert settings["comparison"]["fit_state"] == "candidate"
    assert (
        settings["comparison"]["point_residual_field_plotted"]
        == "projection_residual_q"
    )
    assert settings["comparison"]["normal_profile_count"] == 1
    manifest = json.loads(outputs["manifest"].read_text(encoding="utf-8"))
    assert "ellipse_only.svg" in manifest["sha256"]
    for name, checksum in manifest["sha256"].items():
        assert hashlib.sha256((target / name).read_bytes()).hexdigest() == checksum

    prepared = _prepare_inputs(
        observed=observed,
        qx=qx,
        qy=qy,
        valid_mask=np.ones(observed.shape, dtype=bool),
        result=result,
        q_unit="nm^-1",
        context=None,
        display_scale="log1p",
        width_mm=183,
        dpi=72,
    )
    fit_state, curves, _metadata = _candidate_curves(prepared["result_safe"])
    comparison_figure, _selection = _comparison_figure(
        prepared,
        curves=curves,
        fit_state=fit_state,
        residual_rows=[],
        residual_field=None,
        selected_profile_data=None,
    )
    overlay_ax = comparison_figure.axes[1]
    assert len(overlay_ax.lines) == len(curves) == 2
    for line, curve in zip(overlay_ax.lines, curves, strict=True):
        np.testing.assert_allclose(line.get_xdata(), curve["points"][:, 0])
        np.testing.assert_allclose(line.get_ydata(), curve["points"][:, 1])
        assert line.get_linestyle() == "--"
    _legend_handles, legend_labels = overlay_ax.get_legend_handles_labels()
    assert "candidate · branch 0" in legend_labels
    assert "candidate · branch 1" in legend_labels
    comparison_figure.clear()

    ellipse_figure = _ellipse_only_figure(prepared, curves, fit_state)
    ellipse_figure.canvas.draw()
    ellipse_axes = ellipse_figure.axes[0]
    all_curve_points = np.concatenate(
        [curve["points"] for curve in curves], axis=0
    )
    xmin, xmax = ellipse_axes.get_xlim()
    ymin, ymax = ellipse_axes.get_ylim()
    x_span = float(np.ptp(all_curve_points[:, 0]))
    y_span = float(np.ptp(all_curve_points[:, 1]))
    x_axis_span, y_axis_span = xmax - xmin, ymax - ymin
    assert x_span < x_axis_span <= 1.2 * x_span
    assert y_span < y_axis_span <= 1.2 * y_span
    ellipse_figure.clear()


def test_wide_model_comparison_reserves_left_ylabel_margin() -> None:
    observed, qx, qy = _frame((18, 20))
    prepared = _prepare_inputs(
        observed=observed,
        qx=qx,
        qy=qy,
        valid_mask=np.ones(observed.shape, dtype=bool),
        result={"points": [], "arcs": []},
        q_unit="nm^-1",
        context=None,
        display_scale="log1p",
        width_mm=183,
        dpi=72,
    )
    model = observed * 0.85 + 0.5
    model_float = np.asarray(model, dtype=np.float64)
    common_valid = prepared["display_valid"] & np.isfinite(model_float)
    with mpl.rc_context(_FIGURE_RC):
        figure, _metadata = _model_comparison_figure(
            prepared, model, model_float, common_valid
        )
    figure.canvas.draw()
    map_axes = figure.axes[:3]
    first_map = map_axes[0]
    ylabel_box = first_map.yaxis.label.get_window_extent(
        figure.canvas.get_renderer()
    )
    assert ylabel_box.x0 > 0.0
    assert first_map.get_ylabel()
    assert all(not ax.get_ylabel() for ax in map_axes[1:])
    renderer = figure.canvas.get_renderer()
    first_tick_boxes = [
        tick.get_window_extent(renderer)
        for tick in first_map.get_yticklabels()
        if tick.get_visible() and tick.get_text()
    ]
    assert first_tick_boxes
    assert min(box.x0 for box in first_tick_boxes) >= 0.0
    for previous, current in zip(map_axes[:-1], map_axes[1:], strict=True):
        previous_right = previous.get_window_extent(renderer).x1
        tick_boxes = [
            tick.get_window_extent(renderer)
            for tick in current.get_yticklabels()
            if tick.get_visible() and tick.get_text()
        ]
        assert tick_boxes
        assert min(box.x0 for box in tick_boxes) >= previous_right
    figure.clear()


@pytest.mark.parametrize("width_mm", [89, 183])
def test_overlay_legend_is_outside_map_for_candidate_and_trace_only(
    width_mm: int,
) -> None:
    observed, qx, qy = _frame()
    candidate_result = _candidate_result(qx, qy)
    candidate_result["arcs"] = [
        {
            "arc_id": "arc-0",
            "valid": True,
            "ordered_point_ids": ["p0", "p1"],
        }
    ]
    trace_only_result = dict(candidate_result)
    trace_only_result.pop("candidate_fit")

    for result, expect_candidate in (
        (candidate_result, True),
        (trace_only_result, False),
    ):
        prepared = _prepare_inputs(
            observed=observed,
            qx=qx,
            qy=qy,
            valid_mask=np.ones(observed.shape, dtype=bool),
            result=result,
            q_unit="nm^-1",
            context=None,
            display_scale="log1p",
            width_mm=width_mm,
            dpi=72,
        )
        fit_state, curves, _metadata = _candidate_curves(prepared["result_safe"])
        with mpl.rc_context(_FIGURE_RC):
            figure, _selection = _comparison_figure(
                prepared,
                curves=curves,
                fit_state=fit_state,
                residual_rows=[],
                residual_field=None,
                selected_profile_data=None,
            )
            figure.canvas.draw()

        renderer = figure.canvas.get_renderer()
        axes_by_gid = {ax.get_gid(): ax for ax in figure.axes if ax.get_gid()}
        overlay = axes_by_gid["butterfly-measured-overlay"]
        assert len(overlay.lines) == 1 + len(curves)
        legend = overlay.get_legend()
        assert legend is not None
        assert not legend.get_frame().get_visible()
        assert legend._ncols == 2
        legend_box = legend.get_window_extent(renderer)
        overlay_box = overlay.get_window_extent(renderer)
        assert not legend_box.overlaps(overlay_box)
        assert not legend_box.overlaps(
            overlay.xaxis.label.get_window_extent(renderer)
        )
        if width_mm == 183:
            measured = axes_by_gid["butterfly-measured-image"]
            assert legend_box.y0 >= overlay_box.y1
            assert legend_box.y1 <= measured.get_window_extent(renderer).y0
        else:
            residual = axes_by_gid["butterfly-residual-diagnostic"]
            assert legend_box.y1 <= overlay_box.y0
            assert legend_box.y0 >= residual.get_window_extent(renderer).y1

        labels = [text.get_text() for text in legend.get_texts()]
        assert "accepted source point" in labels
        assert "supported observed arc" in labels
        if expect_candidate:
            assert "candidate · branch 0" in labels
            assert "candidate · branch 1" in labels
        else:
            assert not any(label.startswith("candidate ·") for label in labels)
        figure.clear()


def test_ring_only_candidate_is_labeled_and_exported_as_diagnostic(
    tmp_path: Path,
) -> None:
    observed, qx, qy = _frame((14, 16))
    result = {
        "points": [],
        "arcs": [],
        "candidate_fit": {
            "success": True,
            "status": "ring_only",
            "a": 0.08,
            "b": 0.002,
            "axis_ratio": 0.025,
            "center_qx": 0.0,
            "center_qy": 0.0,
            "theta_deg": 18.0,
            "ellipses": [
                {"a": 0.08, "b": 0.002, "axis_ratio": 0.025, "angle_deg": 18.0},
                {"a": 0.08, "b": 0.002, "axis_ratio": 0.025, "angle_deg": -18.0},
            ],
            "flags": ["axis_ratio_at_bound"],
        },
    }
    outputs = export_butterfly_figure(
        tmp_path / "ring-only",
        observed=observed,
        qx=qx,
        qy=qy,
        result=result,
        width_mm=89,
        dpi=72,
    )
    settings = json.loads(outputs["settings"].read_text(encoding="utf-8"))
    assert settings["comparison"]["fit_state"] == "ring_only_candidate"
    assert settings["comparison"]["candidate_ellipse_curves"] == 2
    assert (
        settings["comparison"]["candidate_ellipse_scientific_acceptance_inferred"]
        is False
    )
    with np.load(outputs["ellipse_curves_npz"], allow_pickle=False) as curves:
        assert curves["q_points"].shape == (2, 361, 2)
    caption = outputs["ellipse_only_caption"].read_text(encoding="utf-8")
    assert "not scientifically accepted" in caption
    assert (
        "candidate" in outputs["ellipse_only_svg"].read_text(encoding="utf-8").lower()
    )


def test_trace_only_exports_comparison_without_fabricating_fit_curves(
    tmp_path: Path,
) -> None:
    observed, qx, qy = _frame((12, 12))
    result = {"points": [], "arcs": [], "profiles": {}}
    outputs = export_butterfly_figure(
        tmp_path / "trace-only-comparison",
        observed=observed,
        qx=qx,
        qy=qy,
        result=result,
        dpi=72,
    )
    assert outputs["comparison_png"].is_file()
    assert "ellipse_only_svg" not in outputs
    settings = json.loads(outputs["settings"].read_text(encoding="utf-8"))
    assert settings["comparison"]["fit_state"] == "not_available"
    assert settings["comparison"]["candidate_ellipse_curves"] == 0
    with np.load(outputs["ellipse_curves_npz"], allow_pickle=False) as curves:
        assert curves["q_points"].shape == (0, 361, 2)
    assert "not scientifically accepted" in outputs["comparison_caption"].read_text(
        encoding="utf-8"
    )


def test_incomplete_candidate_does_not_assume_origin_or_reference_axis(
    tmp_path: Path,
) -> None:
    observed, qx, qy = _frame((12, 12))
    result = {
        "points": [],
        "arcs": [],
        "candidate_fit": {
            "success": True,
            "status": "ok",
            "a": 0.08,
            "b": 0.032,
            "axis_ratio": 0.4,
            "theta_deg": 20.0,
        },
    }
    outputs = export_butterfly_figure(
        tmp_path / "incomplete-candidate",
        observed=observed,
        qx=qx,
        qy=qy,
        result=result,
        dpi=72,
    )
    settings = json.loads(outputs["settings"].read_text(encoding="utf-8"))
    assert settings["comparison"]["fit_state"] == "fit_unavailable"
    assert settings["comparison"]["candidate_ellipse_curves"] == 0
    assert "ellipse_only_svg" not in outputs


def test_caption_reports_profile_fallback_when_all_point_residuals_are_invalid(
    tmp_path: Path,
) -> None:
    observed, qx, qy = _frame()
    result = _candidate_result(qx, qy)
    for diagnostic in result["candidate_fit"]["point_diagnostics"]:
        diagnostic["used"] = False
        diagnostic["projection_valid"] = False
    outputs = export_butterfly_figure(
        tmp_path / "invalid-point-residuals",
        observed=observed,
        qx=qx,
        qy=qy,
        valid_mask=np.ones(observed.shape, dtype=bool),
        result=result,
        width_mm=183,
        dpi=72,
    )
    settings = json.loads(outputs["settings"].read_text(encoding="utf-8"))
    assert settings["comparison"]["point_residual_field_available"] == (
        "projection_residual_q"
    )
    assert settings["comparison"]["point_residual_field_plotted"] is None
    assert settings["comparison"]["residual_panel_source"].startswith(
        "Normal profile for source point"
    )
    caption = outputs["comparison_caption"].read_text(encoding="utf-8")
    assert (
        "Normal profile for source point p0 (accepted/valid point nearest median q radius)"
        in caption
    )
    assert "uses source field projection_residual_q" not in caption


def test_optional_model_comparison_uses_supplied_model_and_signed_raw_difference(
    tmp_path: Path,
) -> None:
    observed, qx, qy = _frame((18, 20))
    model = observed * 0.8 + 1.25
    model[0, 0] = np.nan
    valid = np.ones(observed.shape, dtype=bool)
    valid[2, 3] = False
    model_before = model.copy()
    target = tmp_path / "model-comparison"
    outputs = export_butterfly_figure(
        target,
        observed=observed,
        qx=qx,
        qy=qy,
        valid_mask=valid,
        result={"points": [], "arcs": []},
        model=model,
        q_unit="nm^-1",
        context={"pixel_model_status": "failed"},
        width_mm=89,
        dpi=72,
    )
    assert {
        "model_comparison_svg",
        "model_comparison_pdf",
        "model_comparison_png",
        "model_comparison_tiff",
        "model_comparison_data",
        "model_comparison_caption",
    } <= set(outputs)
    np.testing.assert_array_equal(model, model_before)
    with np.load(outputs["model_comparison_data"], allow_pickle=False) as comparison:
        np.testing.assert_array_equal(comparison["model"], model_before)
        common = comparison["common_valid_mask"]
        expected = np.full(observed.shape, np.nan)
        expected[common] = observed[common] - model_before[common]
        np.testing.assert_allclose(
            comparison["observed_minus_model_raw"], expected, equal_nan=True
        )
        assert not common[0, 0]
        assert not common[2, 3]
        vmin, vmax = comparison["shared_color_limits_transformed"]
        assert vmin < vmax
        residual_low, residual_high = comparison["residual_color_limits_raw"]
        assert residual_low == pytest.approx(-residual_high)
    settings = json.loads(outputs["settings"].read_text(encoding="utf-8"))
    assert settings["comparison"]["model_comparison_supplied"] is True
    assert settings["comparison"]["model_comparison"]["pixel_model_status"] == "failed"
    assert (
        "failed"
        in settings["comparison"]["model_comparison"]["visible_model_status_label"]
    )
    assert settings["comparison"]["model_comparison"]["residual_definition"] == (
        "observed - model in original intensity units"
    )
    assert "Pixel model status from caller context: failed" in outputs[
        "model_comparison_caption"
    ].read_text(encoding="utf-8")
    manifest = json.loads(outputs["manifest"].read_text(encoding="utf-8"))
    assert "model_comparison_data.npz" in manifest["sha256"]

    unknown = export_butterfly_figure(
        tmp_path / "unknown-model-status",
        observed=observed,
        qx=qx,
        qy=qy,
        valid_mask=valid,
        result={"points": [], "arcs": []},
        model=model,
        q_unit="nm^-1",
        width_mm=89,
        dpi=72,
    )
    unknown_settings = json.loads(unknown["settings"].read_text(encoding="utf-8"))
    assert (
        unknown_settings["comparison"]["model_comparison"]["pixel_model_status"]
        == "unknown"
    )
    assert (
        "status unknown"
        in unknown_settings["comparison"]["model_comparison"][
            "visible_model_status_label"
        ]
    )
    assert "status from caller context: unknown" in unknown[
        "model_comparison_caption"
    ].read_text(encoding="utf-8")
