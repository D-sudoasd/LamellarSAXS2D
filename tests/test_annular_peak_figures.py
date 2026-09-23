from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from butterfly_saxs.annular_peak_figures import (
    export_annular_peak_figures,
    render_annular_peak_figures,
)


def _result() -> dict:
    angles = np.asarray([-180.0, -90.0, 0.0, 90.0], dtype=float)
    q_edges = np.asarray([0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70], dtype=float)
    annuli = []
    for index, q_center in enumerate((0.15, 0.25, 0.35, 0.45, 0.55, 0.65)):
        raw = np.asarray([1.0, 4.0, 2.0, 3.0], dtype=float) + index
        selected = [
            {
                "point_id": f"ring-{index}-a",
                "trajectory_id": "lobe-a",
                "chi_deg": -90.0,
                "qx": 0.0,
                "qy": -q_center,
                "accepted": True,
                "raw_intensity": float(raw[1]),
                "intensity": float(raw[1]),
            },
            {
                "point_id": f"ring-{index}-b",
                "trajectory_id": "lobe-b",
                "chi_deg": 90.0,
                "qx": 0.0,
                "qy": q_center,
                "accepted": True,
                "raw_intensity": float(raw[3]),
                "intensity": float(raw[3]),
            },
        ]
        annuli.append(
            {
                "annulus_index": index,
                "q_center": q_center,
                "q_min": q_edges[index],
                "q_max": q_edges[index + 1],
                "profile_id": f"profile-{index}",
                "raw_mean": raw,
                "raw_sum": raw * 10.0,
                "counts": np.asarray([10, 11, 12, 13]),
                "geometry_counts": np.asarray([12, 12, 12, 14]),
                "coverage": np.asarray([0.8, 0.9, 1.0, 0.7]),
                "smoothed_intensity": raw + 0.25,
                "candidates": [
                    {"point_id": f"candidate-{index}", "chi_deg": -180.0, "prominence": 2.5},
                ],
                "selected_peaks": selected,
                "status": "selected",
                "reason": "",
            }
        )
    return {
        "q_unit": "nm^-1",
        "q_edges": q_edges,
        "angle_centers_deg": angles,
        "settings": {"n_annuli": 6, "n_angle_bins": 4},
        "method_version": "annular-test-v1",
        "annuli": annuli,
    }


def _assert_text_inside(figure) -> None:
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    container = figure.bbox
    artists = list(figure.texts)
    for axis in figure.axes:
        artists.extend((axis.title, axis.xaxis.label, axis.yaxis.label))
        artists.extend(axis.texts)
    for legend in figure.legends:
        artists.extend(legend.get_texts())
    for artist in artists:
        if not artist.get_visible() or not artist.get_text().strip():
            continue
        bbox = artist.get_window_extent(renderer)
        assert bbox.x0 >= container.x0 - 1.0
        assert bbox.y0 >= container.y0 - 1.0
        assert bbox.x1 <= container.x1 + 1.0
        assert bbox.y1 <= container.y1 + 1.0


def test_annular_figures_keep_supplied_gaps_and_mark_all_selected_peaks() -> None:
    result = _result()
    result["annuli"][1]["raw_mean"][1] = np.nan
    result["annuli"][1]["smoothed_intensity"][1] = np.nan
    # Refined angular maxima need not equal the centre of an angular bin.
    result["annuli"][1]["selected_peaks"][0]["chi_deg"] = -89.5
    for width in (89.0, 183.0):
        figures = render_annular_peak_figures(
            result,
            data={"context": {"metadata": {"header": {"IntensityUnit": "cm^-1"}}}},
            width_mm=width,
            dpi=120,
        )
        assert figures["annular_qchi"].get_size_inches()[0] * 25.4 == pytest.approx(width)
        assert figures["annular_profiles"].get_size_inches()[0] * 25.4 == pytest.approx(width)
        _assert_text_inside(figures["annular_qchi"])
        _assert_text_inside(figures["annular_profiles"])
        assert any(collection.get_offsets().shape[0] == 1 for collection in figures["annular_qchi"].axes[0].collections)
        raw_lines = [line for line in figures["annular_profiles"].axes[0].lines if line.get_label() == "raw annular mean"]
        assert raw_lines
        assert len(raw_lines[0].get_ydata()) == len(result["angle_centers_deg"])
        assert np.isnan(raw_lines[0].get_ydata()).any()
        for figure in figures.values():
            figure.clear()


def test_annular_narrow_layout_stays_contained_at_export_dpi() -> None:
    figures = render_annular_peak_figures(_result(), data={}, width_mm=89, dpi=600)
    _assert_text_inside(figures["annular_qchi"])
    _assert_text_inside(figures["annular_profiles"])
    for figure in figures.values():
        figure.clear()


def test_annular_export_roundtrips_raw_arrays_and_selected_candidates(tmp_path: Path) -> None:
    result = _result()
    original_raw = np.asarray([row["raw_mean"] for row in result["annuli"]], dtype=float)
    original_angles = np.asarray(result["angle_centers_deg"], dtype=float).copy()
    output, metadata = export_annular_peak_figures(
        tmp_path,
        data={
            "context": {"metadata": {"header": {"IntensityUnit": "cm^-1"}}},
            "width_mm": 183,
            "dpi": 120,
        },
        result=result,
    )
    assert metadata["q_coordinate_is_annulus_center"] is True
    assert metadata["radial_q_star_not_used"] is True
    assert metadata["selected_peak_count"] == 12
    with np.load(output["annular_profiles_npz"], allow_pickle=False) as arrays:
        np.testing.assert_array_equal(arrays["raw_mean"], original_raw)
        np.testing.assert_array_equal(arrays["q_edges"], result["q_edges"])
        assert arrays["q_unit"].item() == "nm^-1"
    np.testing.assert_array_equal(result["angle_centers_deg"], original_angles)
    np.testing.assert_array_equal(result["annuli"][0]["raw_mean"], original_raw[0])
    with output["annular_profiles_csv"].open(encoding="utf-8-sig", newline="") as stream:
        profile_rows = list(csv.DictReader(stream))
    assert len(profile_rows) == 6 * 4
    assert any(row["selected"] == "True" for row in profile_rows)
    # The raw evidence table retains source order; the figure view is
    # canonicalized separately to 0..360.
    first_annulus = [row for row in profile_rows if row["annulus_index"] == "0"]
    assert [row["chi_deg"] for row in first_annulus] == ["-180", "-90", "0", "90"]
    with output["annular_peaks_csv"].open(encoding="utf-8-sig", newline="") as stream:
        peak_rows = list(csv.DictReader(stream))
    assert sum(row["record_type"] == "selected" for row in peak_rows) == 12
    assert sum(row["record_type"] == "candidate" for row in peak_rows) == 6
    manifest = json.loads(output["annular_manifest"].read_text(encoding="utf-8"))
    for name, digest in manifest["sha256"].items():
        assert hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() == digest
    caption = output["annular_caption"].read_text(encoding="utf-8")
    assert "annulus coordinate" in caption
    assert "2π/q" in caption
    assert "ellipse fit" in caption


def test_annular_export_preserves_candidate_text_and_does_not_reselect(tmp_path: Path) -> None:
    result = _result()
    result["annuli"][0]["status"] = "=STATUS()"
    result["annuli"][0]["reason"] = "=SUM(A1:A2)"
    result["annuli"][0]["selected_peaks"] = [result["annuli"][0]["selected_peaks"][0]]
    result["annuli"][0]["candidates"] = [
        {"point_id": "candidate", "reason": "=DROP_ME()", "chi_deg": 0.0}
    ]
    output, _ = export_annular_peak_figures(
        tmp_path,
        data={"width_mm": 89, "dpi": 120},
        result={"annular_peaks": result},
    )
    with output["annular_peaks_csv"].open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    first = next(row for row in rows if row["annulus_index"] == "0" and row["record_type"] == "candidate")
    assert first["candidate_json"].startswith("{")
    with output["annular_profiles_csv"].open(encoding="utf-8-sig", newline="") as stream:
        profiles = list(csv.DictReader(stream))
    assert profiles[0]["status"] == "'=STATUS()"
    assert profiles[0]["reason"] == "'=SUM(A1:A2)"
    assert sum(row["record_type"] == "selected" for row in rows) == 11
    with pytest.raises(FileExistsError):
        export_annular_peak_figures(tmp_path, data={"width_mm": 89, "dpi": 120}, result=result)


def test_annular_export_refuses_more_than_four_supplied_peaks(tmp_path: Path) -> None:
    result = _result()
    result["annuli"][0]["selected_peaks"] = result["annuli"][0]["selected_peaks"] * 3
    with pytest.raises(ValueError, match="more than four"):
        export_annular_peak_figures(tmp_path, data={"width_mm": 89, "dpi": 120}, result=result)
