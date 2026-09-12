from __future__ import annotations

import hashlib
import csv
import io
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
from matplotlib.legend import Legend
from matplotlib.text import Text

from butterfly_saxs.cancellation import AnalysisCancelled
from butterfly_saxs.peak_landmark_figures import export_peak_landmark_figures
from butterfly_saxs.peak_landmarks import compute_peak_landmarks


def _frame(shape: tuple[int, int] = (81, 81)) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows, cols = np.indices(shape, dtype=np.float64)
    center_y = 0.5 * (shape[0] - 1)
    center_x = 0.5 * (shape[1] - 1)
    qx = (cols - center_x) * 0.006
    qy = (rows - center_y) * 0.006
    radius = np.hypot(qx, qy)
    angle = np.mod(np.degrees(np.arctan2(qy, qx)), 360.0)
    observed = np.full(shape, 2.0, dtype=np.float64)
    for center in (25.0, 115.0, 205.0, 295.0):
        delta = np.abs((angle - center + 180.0) % 360.0 - 180.0)
        observed += (
            20.0
            * np.exp(-0.5 * ((radius - 0.22) / 0.035) ** 2)
            * np.exp(-0.5 * (delta / 16.0) ** 2)
        )
    return observed, qx, qy


def _landmarks(
    observed: np.ndarray, qx: np.ndarray, qy: np.ndarray, *, model: np.ndarray | None = None
) -> dict:
    return compute_peak_landmarks(
        observed,
        qx,
        qy,
        valid_mask=np.ones(observed.shape, dtype=bool),
        q_window=(0.08, 0.34),
        signal_q_window=(0.15, 0.29),
        q_unit="nm^-1",
        model=model,
    )


def test_export_writes_peak_figures_profiles_and_hashed_manifest(tmp_path: Path) -> None:
    stage = tmp_path / "stage"
    stage.mkdir()
    observed, qx, qy = _frame()
    model = observed.copy()
    before = observed.copy()
    landmarks = _landmarks(observed, qx, qy, model=model)

    paths, metadata = export_peak_landmark_figures(
        stage,
        data={
            "observed": observed,
            "qx": qx,
            "qy": qy,
            "valid_mask": np.ones(observed.shape, dtype=bool),
            "q_unit": "nm^-1",
            "width_mm": 89.0,
            "dpi": 72,
        },
        landmarks=landmarks,
        model=model,
    )

    assert set(paths) == {
        "peak_map_svg",
        "peak_map_pdf",
        "peak_map_png",
        "peak_map_tiff",
        "peak_diagnostics_svg",
        "peak_diagnostics_pdf",
        "peak_diagnostics_png",
        "peak_diagnostics_tiff",
        "peak_zooms_svg",
        "peak_zooms_pdf",
        "peak_zooms_png",
        "peak_zooms_tiff",
        "peak_landmarks_json",
        "peak_landmarks_csv",
        "peak_profiles_csv",
        "peak_profiles_npz",
        "peak_manifest",
    }
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths.values())
    assert all(path.parent == stage for path in paths.values())
    np.testing.assert_array_equal(observed, before)
    assert metadata["supported_lobe_count"] == 4
    assert metadata["model_supplied"] is True
    assert metadata["model_peak_pair_count"] == 4

    svg_root = ET.parse(paths["peak_map_svg"]).getroot()
    assert "text" in paths["peak_map_svg"].read_text(encoding="utf-8")
    assert "G · raw max" in paths["peak_map_svg"].read_text(encoding="utf-8")
    assert "Smoothed model max (same basin)" in paths["peak_map_svg"].read_text(encoding="utf-8")
    assert "P1 local pixel q" in paths["peak_diagnostics_svg"].read_text(encoding="utf-8")
    assert "not radial-profile fits" in paths["peak_diagnostics_svg"].read_text(encoding="utf-8")
    assert "asinh(I)" in paths["peak_zooms_svg"].read_text(encoding="utf-8")
    expected_width_pt = 89.0 / 25.4 * 72.0
    assert float(svg_root.attrib["width"].removesuffix("pt")) == pytest.approx(
        expected_width_pt, abs=0.03
    )
    pdf = paths["peak_map_pdf"].read_bytes()
    box = re.search(rb"/MediaBox\s*\[\s*([0-9.+-]+)\s+([0-9.+-]+)\s+([0-9.+-]+)\s+([0-9.+-]+)", pdf)
    assert box is not None
    assert float(box.group(3)) == pytest.approx(expected_width_pt, abs=0.03)
    for key in ("peak_map_png", "peak_map_tiff"):
        with Image.open(paths[key]) as image:
            assert image.width == round(89.0 / 25.4 * 72)

    landmark_json = json.loads(paths["peak_landmarks_json"].read_text(encoding="utf-8"))
    assert landmark_json["raw_global_max"]["interpretation"] == "raw_maximum_only_not_an_identified_reflection"
    assert landmark_json["raw_global_max"]["label"] == "G"
    csv_text = paths["peak_landmarks_csv"].read_text(encoding="utf-8-sig")
    assert "raw_global_maximum_only" in csv_text
    assert "supported_angular_lobe_peak" in csv_text
    profile_text = paths["peak_profiles_csv"].read_text(encoding="utf-8-sig")
    assert "angular" in profile_text and "radial" in profile_text
    profile_rows = list(csv.DictReader(io.StringIO(profile_text)))
    angular_row = next(row for row in profile_rows if row["profile_type"] == "angular")
    assert float(angular_row["intensity_isotropic_reference"]) == pytest.approx(
        landmarks["profiles"]["angular"]["intensity_isotropic_reference"][0]
    )
    assert float(angular_row["intensity_detection"]) == pytest.approx(
        landmarks["profiles"]["angular"]["intensity_detection"][0]
    )
    with np.load(paths["peak_profiles_npz"], allow_pickle=False) as profiles:
        assert profiles["angular_angle_deg"].shape == (180,)
        assert profiles["radial_q"].shape == (96,)
        np.testing.assert_allclose(
            profiles["angular_intensity_isotropic_reference"],
            landmarks["profiles"]["angular"]["intensity_isotropic_reference"],
        )
        np.testing.assert_allclose(
            profiles["angular_intensity_detection"],
            landmarks["profiles"]["angular"]["intensity_detection"],
        )
        assert profiles["q_unit"].item() == "nm^-1"

    manifest = json.loads(paths["peak_manifest"].read_text(encoding="utf-8"))
    assert manifest["manifest_excluded_from_own_sha256"] is True
    assert manifest["signal_q_window_source"] == "explicit_signal_q_window"
    assert "maximum mask-normalized smoothed observed pixel" in manifest["peak_selection_method"]
    assert manifest["angular_detection_method"] == landmarks["angular_detection"]["method"]
    assert manifest["angular_reference_profile_key"] == "profiles.angular.intensity_isotropic_reference"
    assert manifest["angular_detection_profile_key"] == "profiles.angular.intensity_detection"
    assert "2D q-vector displacement magnitude" in manifest["delta_q_definition"]
    for name, digest in manifest["sha256"].items():
        assert hashlib.sha256((stage / name).read_bytes()).hexdigest() == digest
    assert len(list(stage.iterdir())) == len(paths)


def test_angular_detection_profiles_render_and_legacy_schema_falls_back() -> None:
    from butterfly_saxs.peak_landmark_figures import _build_peak_diagnostics

    observed, qx, qy = _frame()
    landmarks = _landmarks(observed, qx, qy)
    expected_angular = landmarks["profiles"]["angular"]
    figure = _build_peak_diagnostics(landmarks, width_mm=89.0, dpi=72)
    try:
        angular_ax = figure.axes[0]
        lines = {line.get_label(): line for line in angular_ax.lines}
        reference = lines["isotropic reference"]
        detection = lines["detection (smoothed − reference)"]
        np.testing.assert_allclose(
            reference.get_ydata(), expected_angular["intensity_isotropic_reference"]
        )
        np.testing.assert_allclose(
            detection.get_ydata(), expected_angular["intensity_detection"]
        )
        assert reference.get_linestyle() == "--"
        assert detection.get_linestyle() == ":"
        legend = angular_ax.get_legend()
        assert legend is not None
        assert "coverage (right axis)" in {text.get_text() for text in legend.get_texts()}
    finally:
        figure.clear()

    legacy = dict(landmarks)
    profiles = dict(landmarks["profiles"])
    profiles["angular"] = {
        key: value
        for key, value in expected_angular.items()
        if key not in {"intensity_isotropic_reference", "intensity_detection"}
    }
    legacy["profiles"] = profiles
    figure = _build_peak_diagnostics(legacy, width_mm=89.0, dpi=72)
    try:
        angular_ax = figure.axes[0]
        assert angular_ax.get_legend() is None
        assert angular_ax.get_title() == (
            "Angular · raw gray / smoothed blue / coverage green"
        )
        assert not any(
            line.get_label() in {
                "isotropic reference",
                "detection (smoothed − reference)",
            }
            for line in angular_ax.lines
        )
    finally:
        figure.clear()


def test_export_handles_empty_data_and_only_writes_inside_caller_stage(tmp_path: Path) -> None:
    stage = tmp_path / "stage"
    stage.mkdir()
    observed, qx, qy = _frame((21, 23))
    observed[:] = np.nan
    valid = np.ones(observed.shape, dtype=bool)
    landmarks = _landmarks(observed, qx, qy)

    paths, metadata = export_peak_landmark_figures(
        stage,
        data={"observed": observed, "qx": qx, "qy": qy, "valid_mask": valid, "dpi": 72},
        landmarks=landmarks,
    )

    assert metadata["valid_pixel_count"] == 0
    assert all(path.parent == stage and path.exists() for path in paths.values())
    assert "No valid finite pixels" in paths["peak_map_svg"].read_text(encoding="utf-8")
    assert "No supported peaks to zoom" in paths["peak_zooms_svg"].read_text(encoding="utf-8")


def test_export_requires_an_existing_stage_and_respects_cancellation(tmp_path: Path) -> None:
    observed, qx, qy = _frame((21, 23))
    landmarks = _landmarks(observed, qx, qy)
    data = {"observed": observed, "qx": qx, "qy": qy}

    with pytest.raises(FileNotFoundError):
        export_peak_landmark_figures(tmp_path / "missing", data=data, landmarks=landmarks)
    stage = tmp_path / "cancelled"
    stage.mkdir()
    with pytest.raises(AnalysisCancelled):
        export_peak_landmark_figures(
            stage,
            data=data,
            landmarks=landmarks,
            cancel_event=lambda: True,
        )
    assert list(stage.iterdir()) == []


def test_all_publication_text_and_legends_fit_89_and_183_mm_canvases() -> None:
    import matplotlib as mpl

    from butterfly_saxs.butterfly_figure import _FIGURE_RC, _prepare_inputs
    from butterfly_saxs.peak_landmark_figures import (
        _build_peak_diagnostics,
        _build_peak_map,
        _build_peak_zooms,
    )

    observed, qx, qy = _frame()
    landmarks = _landmarks(observed, qx, qy, model=observed.copy())
    valid = np.ones(observed.shape, dtype=bool)
    source = {"q_unit": "nm^-1"}
    with mpl.rc_context(_FIGURE_RC):
        for width_mm in (89.0, 183.0):
            base = _prepare_inputs(
                observed=observed,
                qx=qx,
                qy=qy,
                valid_mask=valid,
                result={},
                q_unit="nm^-1",
                context=None,
                display_scale="asinh",
                width_mm=width_mm,
                dpi=72,
            )
            figures = (
                _build_peak_map(
                    base=base,
                    source=source,
                    landmarks=landmarks,
                    width_mm=width_mm,
                    dpi=72,
                ),
                _build_peak_diagnostics(landmarks, width_mm=width_mm, dpi=72),
                _build_peak_zooms(
                    base=base,
                    source=source,
                    landmarks=landmarks,
                    width_mm=width_mm,
                    dpi=72,
                ),
            )
            for figure in figures:
                try:
                    figure.canvas.draw()
                    renderer = figure.canvas.get_renderer()
                    canvas = figure.bbox
                    for artist in figure.findobj(match=lambda item: isinstance(item, Text)):
                        if not artist.get_visible() or not artist.get_text().strip():
                            continue
                        assert artist.get_fontsize() >= 5.0, (
                            width_mm,
                            artist.get_text(),
                            artist.get_fontsize(),
                        )
                        box = artist.get_window_extent(renderer)
                        assert box.x0 >= canvas.x0 - 1.0, (width_mm, artist.get_text(), box)
                        assert box.y0 >= canvas.y0 - 1.0, (width_mm, artist.get_text(), box)
                        assert box.x1 <= canvas.x1 + 1.0, (width_mm, artist.get_text(), box)
                        assert box.y1 <= canvas.y1 + 1.0, (width_mm, artist.get_text(), box)
                    for legend in figure.findobj(match=lambda item: isinstance(item, Legend)):
                        box = legend.get_window_extent(renderer)
                        assert legend.prop.get_size_in_points() >= 5.0
                        if width_mm == 89.0:
                            assert legend._ncols <= 2
                        assert box.x0 >= canvas.x0 - 1.0, (width_mm, "legend", box)
                        assert box.y0 >= canvas.y0 - 1.0, (width_mm, "legend", box)
                        assert box.x1 <= canvas.x1 + 1.0, (width_mm, "legend", box)
                        assert box.y1 <= canvas.y1 + 1.0, (width_mm, "legend", box)
                finally:
                    figure.clear()
