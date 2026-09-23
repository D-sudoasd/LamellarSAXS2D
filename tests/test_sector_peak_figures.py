from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from butterfly_saxs.sector_peak_figures import (
    export_sector_peak_figures,
    render_sector_peak_figures,
)
from butterfly_saxs.sector_peaks import measure_sector_peaks
from butterfly_saxs.butterfly_figure import export_butterfly_figure


def _sector_result() -> dict:
    axis = np.arange(81, dtype=float) - 40.0
    qy, qx = np.meshgrid(axis * 0.006, axis * 0.006, indexing="ij")
    q = np.hypot(qx, qy)
    image = 0.1 + np.exp(-0.5 * ((q - 0.18) / 0.014) ** 2)
    return measure_sector_peaks(
        image,
        {"qx": qx, "qy": qy, "q": q, "metadata": {"q_unit": "nm^-1"}},
        (0.08, 0.28),
        options={"requested_radial_bins": 28},
    )


def _assert_visible_text_bboxes_are_inside_figure(figure) -> list[str]:
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    container = figure.bbox
    artists = list(figure.texts)
    for axis in figure.axes:
        artists.extend((axis.title, axis.xaxis.label, axis.yaxis.label))
        artists.extend(axis.texts)
    for legend in figure.legends:
        artists.extend(legend.get_texts())
    visible_text = []
    for artist in artists:
        if not artist.get_visible() or not artist.get_text().strip():
            continue
        bbox = artist.get_window_extent(renderer)
        assert bbox.x0 >= container.x0 - 1.0
        assert bbox.y0 >= container.y0 - 1.0
        assert bbox.x1 <= container.x1 + 1.0
        assert bbox.y1 <= container.y1 + 1.0
        visible_text.append(artist.get_text())
    return visible_text


def test_sector_figures_keep_missing_bins_blank_and_selected_markers_are_points() -> None:
    result = _sector_result()
    result["sectors"][0]["raw_mean"][5:8] = np.nan
    result["sectors"][0]["smoothed_intensity"][5:8] = np.nan
    figures = render_sector_peak_figures(
        result,
        data={"q_unit": "nm^-1", "context": {"metadata": {"header": {"IntensityUnit": "cm^-1"}}}},
        width_mm=89,
        dpi=120,
    )

    qchi = figures["sector_qchi"]
    assert qchi.get_size_inches()[0] * 25.4 == 89.0
    assert any(collection.get_offsets().shape[0] >= 0 for collection in qchi.axes[0].collections)
    profiles = figures["sector_profiles"]
    labels = [line.get_label() for line in profiles.axes[0].lines]
    assert any("raw sector mean" in label for label in labels)
    assert any("locator-only smooth" in label for label in labels)
    assert all("fit" not in label.lower() for label in labels)
    for figure in figures.values():
        figure.clear()


def test_sector_figure_text_layout_is_contained_at_both_publication_widths() -> None:
    result = _sector_result()
    for width in (89.0, 183.0):
        figures = render_sector_peak_figures(
            result,
            data={"q_unit": "nm^-1", "context": {"metadata": {"header": {"IntensityUnit": "cm^-1"}}}},
            width_mm=width,
            dpi=120,
        )
        qchi_text = _assert_visible_text_bboxes_are_inside_figure(figures["sector_qchi"])
        profile_text = _assert_visible_text_bboxes_are_inside_figure(figures["sector_profiles"])
        qchi_lower = [text.lower() for text in qchi_text]
        profile_lower = [text.lower() for text in profile_text]
        assert any("raw sector-integrated" in text for text in qchi_lower)
        assert any("raw sector mean" in text for text in profile_lower)
        assert any("locator-only smooth" in text for text in profile_lower)
        assert any("coverage" in text for text in profile_lower)
        for figure in figures.values():
            figure.clear()


def test_sector_export_writes_auditable_profiles_peaks_npz_and_hashes(tmp_path: Path) -> None:
    result = _sector_result()
    result["sectors"][0]["selected_peak"] = None
    result["sectors"][0]["status"] = "no_peak"
    result["sectors"][0]["reason"] = "insufficient_two_sided_support"
    output, metadata = export_sector_peak_figures(
        tmp_path,
        data={
            "q_unit": "nm^-1",
            "context": {"metadata": {"header": {"IntensityUnit": "cm^-1"}}},
            "width_mm": 183,
            "dpi": 120,
        },
        result=result,
    )

    expected = {
        "sector_qchi_svg", "sector_qchi_pdf", "sector_qchi_png", "sector_qchi_tiff",
        "sector_profiles_svg", "sector_profiles_pdf", "sector_profiles_png", "sector_profiles_tiff",
        "sector_profiles_csv", "sector_peaks_csv", "sector_profiles_npz", "sector_caption", "sector_manifest",
    }
    assert set(output) == expected
    assert metadata["intensity_unit"] == "cm^-1"
    assert metadata["smooth_is_locator_only"] is True
    assert metadata["fwhm_is_not_confidence_interval"] is True
    with output["sector_profiles_csv"].open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows
    assert {"raw_mean", "raw_count", "geometric_count", "coverage", "sector_reason"} <= set(rows[0])
    assert any(row["sector_reason"] == "insufficient_two_sided_support" for row in rows)
    with np.load(output["sector_profiles_npz"], allow_pickle=False) as arrays:
        assert arrays["raw_mean"].ndim == 2
        assert arrays["supported_bin_mask"].dtype == np.dtype(bool)
        assert arrays["intensity_unit"].item() == "cm^-1"
        np.testing.assert_allclose(arrays["q_window"], result["q_window"])
    caption = output["sector_caption"].read_text(encoding="utf-8")
    assert "locator-only" in caption
    assert "not a confidence interval" in caption
    assert "correlated" in caption
    assert "Finite raw means remain visible" in caption
    assert "unsupported bins remain blank" not in caption
    manifest = json.loads(output["sector_manifest"].read_text(encoding="utf-8"))
    assert manifest["manifest_excluded_from_own_sha256"] is True
    for name, digest in manifest["sha256"].items():
        assert hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() == digest


def test_intensity_unit_does_not_come_from_a_filename(tmp_path: Path) -> None:
    result = _sector_result()
    result["q_unit"] = "-0.25"
    result["sectors"][0]["status"] = "=STATUS()"
    result["sectors"][0]["reason"] = "=SUM(A1:A2)"
    result["sectors"][0]["candidates"] = [
        {
            "status": "rejected",
            "reason": "flat_top_unresolved_peak",
            "height_snr": 7.25,
            "n_eff": 13.5,
            "flat_top_ratio": 0.42,
            "settings": {"min_prominence_sigma": 3.0},
        }
    ]
    stage = tmp_path / "looks_like_cm-1_data"
    stage.mkdir()
    output, metadata = export_sector_peak_figures(
        stage,
        data={"q_unit": "nm^-1", "context": {}, "width_mm": 89, "dpi": 120},
        result=result,
    )
    assert metadata["intensity_unit"] == "input intensity units"
    with output["sector_profiles_csv"].open(encoding="utf-8-sig", newline="") as stream:
        profile_rows = list(csv.DictReader(stream))
    assert profile_rows[0]["q_unit"] == "-0.25"
    assert profile_rows[0]["sector_status"] == "'=STATUS()"
    assert profile_rows[0]["sector_reason"] == "'=SUM(A1:A2)"
    with output["sector_peaks_csv"].open(encoding="utf-8-sig", newline="") as stream:
        peak_rows = list(csv.DictReader(stream))
    candidate_row = next(row for row in peak_rows if row["record_type"] == "candidate")
    candidate = json.loads(candidate_row["candidate_json"])
    assert candidate["height_snr"] == 7.25
    assert candidate["n_eff"] == 13.5
    assert candidate["flat_top_ratio"] == 0.42
    assert candidate["settings"]["min_prominence_sigma"] == 3.0
    assert "no unit inferred" in output["sector_caption"].read_text(encoding="utf-8")


def test_butterfly_bundle_adds_sector_assets_only_when_result_contains_measurement(
    tmp_path: Path,
) -> None:
    axis = np.arange(12, dtype=float) - 5.5
    qy, qx = np.meshgrid(axis * 0.01, axis * 0.01, indexing="ij")
    q = np.hypot(qx, qy)
    observed = 0.1 + np.exp(-0.5 * ((q - 0.035) / 0.007) ** 2)
    sector_result = measure_sector_peaks(
        observed,
        {"qx": qx, "qy": qy, "q": q, "metadata": {"q_unit": "nm^-1"}},
        (0.02, 0.08),
        options={"requested_radial_bins": 10},
    )
    result = {"points": [], "arcs": [], "sector_peaks": sector_result}
    output = export_butterfly_figure(
        tmp_path / "bundle",
        observed=observed,
        qx=qx,
        qy=qy,
        result=result,
        q_unit="nm^-1",
        context={"metadata": {"header": {"IntensityUnit": "cm^-1"}}},
        dpi=72,
    )
    assert output["sector_qchi_png"].is_file()
    assert output["sector_profiles_csv"].is_file()
    main_manifest = json.loads(output["manifest"].read_text(encoding="utf-8"))
    assert main_manifest["sector_manifest_file"] == "sector_manifest.json"
    assert "sector_qchi.png" in main_manifest["files"]
    html = output["index"].read_text(encoding="utf-8")
    assert "Sector-integrated primary peaks" in html
    assert "Pixel-brightness ancillary diagnostics" in html
