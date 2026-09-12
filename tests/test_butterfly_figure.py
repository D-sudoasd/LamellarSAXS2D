from __future__ import annotations

import csv
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from butterfly_saxs.butterfly_figure import (
    export_butterfly_figure,
    render_butterfly_figure,
)
from butterfly_saxs.cancellation import AnalysisCancelled


def _frame(
    shape: tuple[int, int] = (24, 30),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows, cols = np.indices(shape, dtype=np.float64)
    observed = 10.0 + np.exp(
        -((rows - shape[0] / 2.0) ** 2 + (cols - shape[1] / 2.0) ** 2) / 20.0
    )
    qx = (cols - (shape[1] - 1) / 2.0) * 0.01 + 0.0005 * np.sin(rows / 2.0)
    qy = (rows - (shape[0] - 1) / 2.0) * 0.01 + 0.0003 * np.cos(cols / 3.0)
    return observed, qx, qy


def _result() -> dict:
    return {
        "method_version": "test-trace-v1",
        "measurement_status": "traced",
        "points": [
            {
                "point_id": "p0",
                "qx": 0.02,
                "qy": 0.03,
                "pixel_x": 16.0,
                "pixel_y": 14.0,
                "arc_id": 1,
                "branch_id": None,
                "side": "unknown",
                "accepted": True,
                "valid": True,
            },
            {
                "point_id": "p1",
                "qx": 0.04,
                "qy": 0.035,
                "pixel_x": 18.0,
                "pixel_y": 14.0,
                "arc_id": 1,
                "branch_id": None,
                "side": "unknown",
                "accepted": True,
                "valid": True,
            },
            {
                "point_id": "p2",
                "qx": 0.06,
                "qy": 0.04,
                "pixel_x": 20.0,
                "pixel_y": 14.0,
                "arc_id": 1,
                "branch_id": 9,
                "side": "unknown",
                "accepted": False,
                "valid": False,
                "reason": "manual exclusion",
            },
        ],
        "arcs": [
            {
                "arc_id": 1,
                "valid": True,
                "ordered_point_ids": ["p0", "p1"],
                "branch_ids": [],
                "sides": [],
            }
        ],
        "profiles": {},
        "candidate_fit": {
            "success": True,
            "status": "ring_only",
            "a": 0.3,
            "b": 0.02,
            "theta_deg": 36.0,
            "center_qx": 0.0,
            "center_qy": 0.0,
            "ellipses": [{"a": 0.3, "b": 0.02, "theta_deg": 36.0}],
        },
        "quantitative_parameters": {
            "a": {"value": None, "candidate_value": 0.3, "status": "undetermined"}
        },
        "diagnostics": {"display_magnification": 8.0, "q_window": [0.02, 0.08]},
    }


def test_export_bundle_preserves_source_and_writes_editable_vector_and_hashes(
    tmp_path: Path,
) -> None:
    observed, qx, qy = _frame()
    mask = np.ones(observed.shape, dtype=bool)
    mask[2:5, 7:10] = False
    observed[3, 8] = 1.0e12  # masked outlier must not set the color range
    observed_before, qx_before, qy_before, mask_before = (
        array.copy() for array in (observed, qx, qy, mask)
    )
    result = _result()
    result_before = json.loads(json.dumps(result))
    progress: list[tuple[int, str]] = []

    output = export_butterfly_figure(
        tmp_path / "figure",
        observed=observed,
        qx=qx,
        qy=qy,
        valid_mask=mask,
        result=result,
        q_unit="nm^-1",
        context={"frame": 7, "source": "synthetic-test"},
        width_mm=183,
        dpi=120,
        progress=lambda percent, phase: progress.append((percent, phase)),
    )

    assert set(output) == {
        "svg",
        "pdf",
        "tiff",
        "png",
        "source_data",
        "ridge_points",
        "result",
        "settings",
        "caption",
        "manifest",
        "comparison_svg",
        "comparison_pdf",
        "comparison_tiff",
        "comparison_png",
        "ellipse_curves_csv",
        "ellipse_curves_npz",
        "point_residuals_csv",
        "comparison_caption",
        "ellipse_only_svg",
        "ellipse_only_pdf",
        "ellipse_only_tiff",
        "ellipse_only_png",
        "ellipse_only_caption",
        "measured_only_svg", "measured_only_pdf", "measured_only_png", "measured_only_tiff", "measured_only_caption",
        "peak_map_svg", "peak_map_pdf", "peak_map_png", "peak_map_tiff",
        "peak_diagnostics_svg", "peak_diagnostics_pdf", "peak_diagnostics_png", "peak_diagnostics_tiff",
        "peak_zooms_svg", "peak_zooms_pdf", "peak_zooms_png", "peak_zooms_tiff",
        "peak_landmarks_json", "peak_landmarks_csv", "peak_profiles_csv", "peak_profiles_npz", "peak_manifest",
        "geometry_overlay_svg", "geometry_overlay_pdf", "geometry_overlay_png", "geometry_overlay_tiff", "geometry_overlay_caption",
        "fit_assessment", "fit_source_parameters", "fit_overlay_curves",
    }
    assert all(path.is_file() and path.stat().st_size > 0 for path in output.values())
    np.testing.assert_array_equal(observed, observed_before)
    np.testing.assert_array_equal(qx, qx_before)
    np.testing.assert_array_equal(qy, qy_before)
    np.testing.assert_array_equal(mask, mask_before)
    assert result == result_before

    with np.load(output["source_data"], allow_pickle=False) as source:
        np.testing.assert_array_equal(source["observed"], observed_before)
        np.testing.assert_array_equal(source["qx"], qx_before)
        np.testing.assert_array_equal(source["qy"], qy_before)
        np.testing.assert_array_equal(source["valid_mask"], mask_before)
        assert not source["effective_valid_mask"][3, 8]
        assert source["q_unit"].item() == "nm^-1"
        assert (
            source["supplied_valid_mask_role"]
            .item()
            .startswith("caller-supplied selection mask")
        )
        np.testing.assert_array_equal(source["supplied_valid_mask"], mask_before)
        q_radius = np.hypot(source["qx"], source["qy"])
        selected = (
            source["effective_valid_mask"] & (q_radius >= 0.02) & (q_radius <= 0.08)
        )
        radial_edges = source["radial_q_edges"]
        raw_sum, _ = np.histogram(
            q_radius[selected], bins=radial_edges, weights=source["observed"][selected]
        )
        counts, _ = np.histogram(q_radius[selected], bins=radial_edges)
        np.testing.assert_allclose(source["radial_intensity_sum_raw"], raw_sum)
        np.testing.assert_array_equal(source["radial_valid_pixel_counts"], counts)
        np.testing.assert_array_equal(source["display_valid_mask"], selected)

    with output["svg"].open("rb") as stream:
        root = ET.parse(stream).getroot()
    assert "<text" in output["svg"].read_text(encoding="utf-8")
    expected_width_pt = 183.0 / 25.4 * 72.0
    assert float(root.attrib["width"].removesuffix("pt")) == pytest.approx(
        expected_width_pt, abs=0.02
    )
    pdf_bytes = output["pdf"].read_bytes()
    media_box = re.search(
        rb"/MediaBox\s*\[\s*([0-9.+-]+)\s+([0-9.+-]+)\s+([0-9.+-]+)\s+([0-9.+-]+)",
        pdf_bytes,
    )
    assert media_box is not None
    assert float(media_box.group(3)) == pytest.approx(expected_width_pt, abs=0.02)
    assert float(media_box.group(4)) == pytest.approx(94.0 / 25.4 * 72.0, abs=0.02)
    assert b"/FontFile2" in pdf_bytes
    expected_pixels = (int(183.0 / 25.4 * 120), int(94.0 / 25.4 * 120))
    for key in ("png", "tiff"):
        with Image.open(output[key]) as raster:
            assert raster.size == expected_pixels
    with Image.open(output["comparison_png"]) as comparison_raster:
        assert comparison_raster.size == (expected_pixels[0], int(150.0 / 25.4 * 120))
    assert "<text" in output["comparison_svg"].read_text(encoding="utf-8")

    manifest = json.loads(output["manifest"].read_text(encoding="utf-8"))
    assert manifest["manifest_excluded_from_own_sha256"] is True
    for name, expected_hash in manifest["sha256"].items():
        assert (
            hashlib.sha256((output["png"].parent / name).read_bytes()).hexdigest()
            == expected_hash
        )
    assert "manifest.json" not in manifest["sha256"]

    settings = json.loads(output["settings"].read_text(encoding="utf-8"))
    assert settings["figure"]["width_mm"] == 183.0
    assert settings["figure"]["panel_label_font_pt"] == 8.0
    assert settings["figure"]["body_font_pt"] == 6.5
    assert settings["q"]["image_coordinates"].startswith("supplied qx/qy mesh")
    assert settings["q"]["q_window_applied_to_figure"] is True
    assert settings["q"]["radial_profile_uses_display_transform"] is False
    assert settings["intensity_display"]["color_limits_transformed"][1] < 10.0
    assert settings["input"]["arcs_rendered_as_observed_segments"] == 1
    assert settings["input"]["ridge_point_overlay"]["drawn_count"] == 3
    assert settings["input"]["ridge_point_overlay"]["omitted_count"] == 0
    assert (
        "not assumed to be detector-only"
        in settings["input"]["supplied_valid_mask_role"]
    )
    assert settings["scientific_boundary"]["candidate_ellipse_drawn"] is False
    assert settings["scientific_boundary"]["scientific_acceptance_inferred"] is False
    assert settings["context"]["frame"] == 7
    assert settings["comparison"]["fit_state"] == "ring_only_candidate"
    assert settings["comparison"]["candidate_ellipse_curves"] == 1
    assert (
        settings["comparison"]["candidate_ellipse_scientific_acceptance_inferred"]
        is False
    )
    with np.load(output["ellipse_curves_npz"], allow_pickle=False) as curves:
        assert curves["q_points"].shape == (1, 361, 2)
        assert curves["fit_state"].item() == "ring_only_candidate"
    assert "Ring-only/bound-limited" in output["ellipse_only_caption"].read_text(
        encoding="utf-8"
    )
    result_json = json.loads(output["result"].read_text(encoding="utf-8"))
    assert result_json["candidate_fit"]["status"] == "ring_only"
    assert result_json["quantitative_parameters"]["a"]["value"] is None

    with output["ridge_points"].open(encoding="utf-8-sig", newline="") as stream:
        ridge_rows = list(csv.DictReader(stream))
    assert [row["display_status"] for row in ridge_rows] == [
        "accepted",
        "accepted",
        "rejected",
    ]
    assert [row["side"] for row in ridge_rows] == ["unknown", "unknown", "unknown"]
    assert [row["branch_id"] for row in ridge_rows] == ["", "", "9"]
    assert [row["overlay_status"] for row in ridge_rows] == ["drawn"] * 3
    assert all(not row["overlay_reason"] for row in ridge_rows)
    assert progress[0] == (0, "validate")
    assert progress[-1] == (100, "publish")
    assert [value for value, _ in progress] == sorted(value for value, _ in progress)


def test_render_handles_non_affine_reversed_q_mesh_and_ring_only_without_ellipse() -> (
    None
):
    observed, qx, qy = _frame((18, 22))
    qx = qx[:, ::-1]
    qy = qy[::-1, :]
    valid = np.ones(observed.shape, dtype=bool)
    valid[4:8, 5:8] = False
    result = _result()
    result["points"] = []
    result["arcs"] = []

    figure = render_butterfly_figure(
        observed=observed,
        qx=qx,
        qy=qy,
        valid_mask=valid,
        result=result,
        q_unit="pixel_q",
        width_mm=89,
        dpi=120,
    )
    map_ax = figure.axes[0]
    assert map_ax.get_xlim()[0] < map_ax.get_xlim()[1]
    assert map_ax.get_ylim()[0] < map_ax.get_ylim()[1]
    # The image uses its two-dimensional quadrilateral mesh, not an affine imshow extent.
    assert map_ax.collections
    mesh = map_ax.collections[0]
    assert hasattr(mesh, "get_coordinates")
    mesh_coordinates = mesh.get_coordinates()
    in_window = valid & (np.hypot(qx, qy) >= 0.02) & (np.hypot(qx, qy) <= 0.08)
    window_pixels = np.argwhere(in_window)
    expected_mesh_shape = (
        int(window_pixels[:, 0].max() - window_pixels[:, 0].min() + 2),
        int(window_pixels[:, 1].max() - window_pixels[:, 1].min() + 2),
        2,
    )
    assert mesh_coordinates.shape == expected_mesh_shape
    x_edge_steps = np.diff(mesh_coordinates[..., 0], axis=0)
    assert not np.allclose(x_edge_steps, x_edge_steps[0:1, :])
    assert (
        len(map_ax.lines) == 0
    )  # candidate ellipse is not rendered in ring-only/trace data
    assert len(figure.axes) == 4
    assert figure.get_size_inches()[0] * 25.4 == pytest.approx(89.0)
    figure.clear()


def test_nonfinite_masked_q_coordinates_use_only_valid_samples() -> None:
    observed, qx, qy = _frame((8, 10))
    valid = np.ones(observed.shape, dtype=bool)
    valid[2:4, 3:6] = False
    qx[~valid] = np.nan
    qy[~valid] = np.nan
    figure = render_butterfly_figure(
        observed=observed,
        qx=qx,
        qy=qy,
        valid_mask=valid,
        result={},
        dpi=72,
    )
    map_ax = figure.axes[0]
    assert map_ax.collections
    # A sample collection uses the original finite q coordinates; no q values
    # are filled into the masked cells to make an artificial regular mesh.
    assert not hasattr(map_ax.collections[0], "get_coordinates")
    figure.clear()


def test_masked_q_coordinate_storage_never_enters_mesh_geometry() -> None:
    rows, cols = np.indices((5, 5), dtype=np.float64)
    coordinate_mask = np.zeros((5, 5), dtype=bool)
    coordinate_mask[2, 2] = True
    qx = np.ma.array(cols - 2.0, mask=coordinate_mask, copy=True)
    qy = np.ma.array(rows - 2.0, mask=coordinate_mask, copy=True)
    qx.data[2, 2] = 1000.0
    qy.data[2, 2] = 1000.0
    figure = render_butterfly_figure(
        observed=np.ones((5, 5)),
        qx=qx,
        qy=qy,
        valid_mask=np.ones((5, 5), dtype=bool),
        result={},
        dpi=72,
    )
    map_ax = figure.axes[0]
    image = map_ax.collections[0]
    assert not hasattr(image, "get_coordinates")
    offsets = np.asarray(image.get_offsets())
    assert offsets.shape == (24, 2)
    assert np.all(np.abs(offsets) <= 2.0)
    figure.clear()


def test_unsupported_ridge_points_are_omitted_but_preserved_with_reasons(
    tmp_path: Path,
) -> None:
    rows, cols = np.indices((5, 5), dtype=np.float64)
    observed = np.ones((5, 5), dtype=np.float64)
    qx, qy = cols - 2.0, rows - 2.0
    valid = np.ones((5, 5), dtype=bool)
    valid[2, 2] = False
    result = {
        "points": [
            {
                "point_id": "masked",
                "qx": 0.0,
                "qy": 0.0,
                "pixel_x": 2.0,
                "pixel_y": 2.0,
                "accepted": True,
                "valid": True,
            },
            {
                "point_id": "outside",
                "qx": -3.0,
                "qy": 0.0,
                "pixel_x": -1.0,
                "pixel_y": 2.0,
                "accepted": True,
                "valid": True,
            },
            {"point_id": "legacy-q-only", "qx": 0.5, "qy": 0.5, "accepted": True},
        ],
        "arcs": [],
    }
    original_points = json.loads(json.dumps(result["points"]))
    figure = render_butterfly_figure(
        observed=observed,
        qx=qx,
        qy=qy,
        valid_mask=valid,
        result=result,
        dpi=72,
    )
    map_ax = figure.axes[0]
    assert (
        len(map_ax.collections) == 1
    )  # measured image only; no unsupported point markers
    figure.clear()

    output = export_butterfly_figure(
        tmp_path / "unsupported-points",
        observed=observed,
        qx=qx,
        qy=qy,
        valid_mask=valid,
        result=result,
        dpi=72,
    )
    settings = json.loads(output["settings"].read_text(encoding="utf-8"))
    point_overlay = settings["input"]["ridge_point_overlay"]
    assert point_overlay["input_count"] == 3
    assert point_overlay["drawn_count"] == 0
    assert point_overlay["omitted_count"] == 3
    assert point_overlay["omitted_by_reason"] == {
        "missing_pixel_coordinates": 1,
        "pixel_coordinate_out_of_bounds": 1,
        "source_pixel_invalid_or_masked": 1,
    }
    with output["ridge_points"].open(encoding="utf-8-sig", newline="") as stream:
        rows_out = list(csv.DictReader(stream))
    assert [row["point_id"] for row in rows_out] == [
        "masked",
        "outside",
        "legacy-q-only",
    ]
    assert [row["overlay_status"] for row in rows_out] == ["omitted"] * 3
    assert [row["overlay_reason"] for row in rows_out] == [
        "source_pixel_invalid_or_masked",
        "pixel_coordinate_out_of_bounds",
        "missing_pixel_coordinates",
    ]
    result_json = json.loads(output["result"].read_text(encoding="utf-8"))
    assert result_json["points"] == original_points


def test_supported_arc_line_does_not_cross_an_invalid_source_pixel() -> None:
    rows, cols = np.indices((5, 5), dtype=np.float64)
    valid = np.ones((5, 5), dtype=bool)
    valid[2, 2] = False
    result = {
        "points": [
            {
                "point_id": "left",
                "qx": -1.0,
                "qy": 0.0,
                "pixel_x": 1.0,
                "pixel_y": 2.0,
                "accepted": True,
                "valid": True,
            },
            {
                "point_id": "right",
                "qx": 1.0,
                "qy": 0.0,
                "pixel_x": 3.0,
                "pixel_y": 2.0,
                "accepted": True,
                "valid": True,
            },
        ],
        "arcs": [{"valid": True, "ordered_point_ids": ["left", "right"]}],
    }
    figure = render_butterfly_figure(
        observed=np.ones((5, 5)),
        qx=cols - 2.0,
        qy=rows - 2.0,
        valid_mask=valid,
        result=result,
        dpi=72,
    )
    map_ax = figure.axes[0]
    assert not map_ax.lines
    assert len(map_ax.collections) == 2  # image plus two supported measured points
    figure.clear()


def test_signed_log1p_display_and_radial_means_keep_negative_raw_intensity(
    tmp_path: Path,
) -> None:
    observed, qx, qy = _frame((10, 10))
    observed = observed - 10.25
    result = {"points": [], "arcs": []}
    output = export_butterfly_figure(
        tmp_path / "signed",
        observed=observed,
        qx=qx,
        qy=qy,
        result=result,
        display_scale="log1p",
        width_mm=89,
        dpi=72,
    )
    settings = json.loads(output["settings"].read_text(encoding="utf-8"))
    assert (
        settings["intensity_display"]["display_transform"] == "sign(I) * log1p(abs(I))"
    )
    assert (
        settings["intensity_display"]["negative_values_preserved_by_signed_transform"]
        is True
    )
    with np.load(output["source_data"], allow_pickle=False) as source:
        np.testing.assert_array_equal(source["observed"], observed)
        assert np.nanmin(source["radial_intensity_mean_raw"]) < 0.0


def test_empty_points_and_masked_image_are_supported_but_all_invalid_is_rejected(
    tmp_path: Path,
) -> None:
    observed, qx, qy = _frame((8, 9))
    result = _result()
    result["points"] = []
    result["arcs"] = []
    output = export_butterfly_figure(
        tmp_path / "trace-only",
        observed=observed,
        qx=qx,
        qy=qy,
        valid_mask=np.zeros(observed.shape, dtype=bool)
        | np.eye(*observed.shape, dtype=bool),
        result=result,
        q_unit="unknown",
        width_mm=89,
        dpi=72,
    )
    assert output["png"].is_file()
    assert (
        "unknown unit"
        in json.loads(output["settings"].read_text(encoding="utf-8"))["q"][
            "axis_interpretation"
        ]
    )
    assert "No opposite quadrants" in output["caption"].read_text(encoding="utf-8")

    invalid_target = tmp_path / "all-invalid"
    with pytest.raises(ValueError, match="no valid finite observations"):
        export_butterfly_figure(
            invalid_target,
            observed=observed,
            qx=qx,
            qy=qy,
            valid_mask=np.zeros(observed.shape, dtype=bool),
            result=result,
            dpi=72,
        )
    assert not invalid_target.exists()


def test_export_refuses_overwrite_and_cleans_cancelled_stage(tmp_path: Path) -> None:
    observed, qx, qy = _frame((10, 10))
    target = tmp_path / "existing"
    target.mkdir()
    sentinel = target / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        export_butterfly_figure(
            target,
            observed=observed,
            qx=qx,
            qy=qy,
            result={},
            dpi=72,
        )
    assert sentinel.read_text(encoding="utf-8") == "keep"

    class Event:
        cancelled = False

        def is_set(self) -> bool:
            return self.cancelled

    event = Event()
    cancelled_target = tmp_path / "cancelled"

    def cancel_during_source_data(percent: int, phase: str) -> None:
        if phase == "source_data":
            event.cancelled = True

    with pytest.raises(AnalysisCancelled):
        export_butterfly_figure(
            cancelled_target,
            observed=observed,
            qx=qx,
            qy=qy,
            result={},
            dpi=72,
            cancel_event=event,
            progress=cancel_during_source_data,
        )
    assert not cancelled_target.exists()
    assert list(tmp_path.glob(".cancelled.staging-*")) == []


def test_shape_units_width_and_display_scale_are_validated() -> None:
    observed, qx, qy = _frame((8, 8))
    with pytest.raises(ValueError, match="same shape"):
        render_butterfly_figure(observed=observed, qx=qx[:, :-1], qy=qy, result={})
    with pytest.raises(ValueError, match="width_mm"):
        render_butterfly_figure(
            observed=observed, qx=qx, qy=qy, result={}, width_mm=100
        )
    with pytest.raises(ValueError, match="display_scale"):
        render_butterfly_figure(
            observed=observed, qx=qx, qy=qy, result={}, display_scale="gamma"
        )
