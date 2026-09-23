from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from butterfly_saxs.butterfly_figure import (
    NATURE_FIGURE_GUIDE_URL,
    _caption,
    _draw_radial_diagnostic,
    _prepare_inputs,
    _settings,
    export_butterfly_figure,
)


def _fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    rows, cols = np.indices((9, 9), dtype=np.float64)
    qx = (cols - 4.0) * 0.01
    qy = (rows - 4.0) * 0.01
    observed = 2.0 + np.exp(-((qx / 0.018) ** 2 + (qy / 0.027) ** 2))
    result: dict[str, object] = {
        "measurement_status": "traced",
        "scientific_status": "NOT_ACCEPTED",
        "points": [],
        "arcs": [],
        "candidate_fit": {"status": "ring_only"},
        "diagnostics": {"q_window": [0.0, 0.06]},
    }
    return observed, qx, qy, result


def _radial_axes(
    means: np.ndarray, linthresh: float, *, width_mm: float = 89.0
) -> tuple[Figure, object]:
    figure = Figure(figsize=(width_mm / 25.4, 150.0 / 25.4), dpi=120)
    FigureCanvasAgg(figure)
    profile_ax = figure.add_axes([0.14, 0.17, 0.76, 0.18])
    count_ax = figure.add_axes([0.14, 0.39, 0.76, 0.045], sharex=profile_ax)
    edges = np.linspace(0.1, 0.5, len(means) + 1)
    _draw_radial_diagnostic(
        profile_ax,
        count_ax,
        {
            "radial_centers": 0.5 * (edges[:-1] + edges[1:]),
            "radial_edges": edges,
            "radial_mean": means,
            "radial_counts": np.arange(1, len(means) + 1),
            "radial_linthresh": linthresh,
            "q_unit": "nm^-1",
        },
        counts_bottom=False,
    )
    figure.canvas.draw()
    return figure, profile_ax


@pytest.mark.parametrize(
    ("means", "linthresh"),
    (
        (np.array([3.2, 3.4, 3.6, 3.8, 4.0, 4.2]), 2.8),
        (np.array([-4.2, -4.0, -3.8, -3.6, -3.4, -3.2]), 2.8),
        (np.array([-0.004, -0.002, 0.0, 0.001, 0.003, 0.005]), 0.001),
    ),
)
def test_radial_symlog_keeps_visible_labels_for_narrow_ranges(
    means: np.ndarray, linthresh: float
) -> None:
    figure, profile_ax = _radial_axes(means, linthresh)
    try:
        labels = [
            label
            for label in profile_ax.get_yticklabels()
            if label.get_visible() and label.get_text().strip()
        ]
        assert profile_ax.get_yscale() == "symlog"
        assert len(labels) >= 2
    finally:
        figure.clear()


def test_radial_fallback_labels_fit_89mm_profile_panel() -> None:
    figure, profile_ax = _radial_axes(
        np.array([3.2, 3.4, 3.6, 3.8, 4.0, 4.2]), 2.8
    )
    try:
        renderer = figure.canvas.get_renderer()
        canvas = figure.bbox
        boxes = [
            label.get_window_extent(renderer)
            for label in profile_ax.get_yticklabels()
            if label.get_visible() and label.get_text().strip()
        ]
        assert len(boxes) >= 2
        assert all(
            box.x0 >= canvas.x0 - 1.0
            and box.y0 >= canvas.y0 - 1.0
            and box.x1 <= canvas.x1 + 1.0
            and box.y1 <= canvas.y1 + 1.0
            for box in boxes
        )
        assert all(not first.overlaps(second) for index, first in enumerate(boxes) for second in boxes[index + 1 :])
    finally:
        figure.clear()


@pytest.mark.parametrize("display_scale", ("linear", "log1p", "asinh"))
def test_display_metadata_records_signed_input_preservation(
    display_scale: str,
) -> None:
    observed = np.array([[-3.0, -1.0], [0.0, 2.0]], dtype=np.float64)
    qx = np.array([[-0.02, 0.02], [-0.02, 0.02]], dtype=np.float64)
    qy = np.array([[-0.02, -0.02], [0.02, 0.02]], dtype=np.float64)
    data = _prepare_inputs(
        observed=observed,
        qx=qx,
        qy=qy,
        valid_mask=None,
        result={},
        q_unit="nm^-1",
        context=None,
        display_scale=display_scale,
        width_mm=89.0,
        dpi=120,
    )
    settings = _settings(data)
    assert settings["intensity_display"]["negative_values_preserved"] is True
    assert settings["intensity_display"]["negative_values_preserved_by_signed_transform"] is (
        display_scale in {"log1p", "asinh"}
    )
    caption = _caption(data)
    if display_scale == "linear":
        assert "linear, untransformed" in caption
    elif display_scale == "asinh":
        assert "asinh(I)" in caption
    else:
        assert "signed log1p" in caption


def test_figure_bundle_has_offline_browser_raw_radial_profile_and_qa(
    tmp_path: Path,
) -> None:
    observed, qx, qy, result = _fixture()
    valid = np.ones(observed.shape, dtype=bool)
    valid[0, :] = False
    valid[:, 0] = False

    output = export_butterfly_figure(
        tmp_path / "bundle",
        observed=observed,
        qx=qx,
        qy=qy,
        valid_mask=valid,
        result=result,
        q_unit="nm^-1",
        display_scale="asinh",
        width_mm=89.0,
        dpi=120,
    )

    for key in ("index", "radial_profile", "figure_qa", "manifest"):
        assert output[key].is_file()

    with np.load(output["source_data"], allow_pickle=False) as source:
        edges = source["radial_q_edges"]
        centers = 0.5 * (edges[:-1] + edges[1:])
        raw_sum = source["radial_intensity_sum_raw"]
        raw_mean = source["radial_intensity_mean_raw"]
        counts = source["radial_valid_pixel_counts"]
        display_pixel_count = int(np.count_nonzero(source["display_valid_mask"]))

    with output["radial_profile"].open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert list(rows[0]) == [
        "bin_index",
        "q_unit",
        "edge_left",
        "edge_right",
        "center",
        "raw_sum",
        "raw_mean",
        "count",
    ]
    assert len(rows) == len(centers)
    for index, row in enumerate(rows):
        assert int(row["bin_index"]) == index
        assert row["q_unit"] == "nm^-1"
        assert float(row["edge_left"]) == np.float64(edges[index])
        assert float(row["edge_right"]) == np.float64(edges[index + 1])
        assert float(row["center"]) == np.float64(centers[index])
        assert int(row["count"]) == int(counts[index])
        if counts[index] == 0:
            assert row["raw_sum"] == ""
            assert row["raw_mean"] == ""
        else:
            assert float(row["raw_sum"]) == np.float64(raw_sum[index])
            assert float(row["raw_mean"]) == np.float64(raw_mean[index])
    assert sum(int(row["count"]) for row in rows) == display_pixel_count

    qa = json.loads(output["figure_qa"].read_text(encoding="utf-8"))
    assert qa["status"] == "WARN"
    assert qa["dpi"] == 120
    assert qa["checks"]["dpi"]["status"] == "WARN"
    assert qa["checks"]["physical_size"]["status"] == "configured"
    assert qa["checks"]["editable_text"]["status"] == "configured"
    assert qa["q_unit"] == "nm^-1"
    assert qa["scientific_acceptance"] == "not_assessed"
    assert qa["no_fabricated_points"] is True
    assert qa["nature_reference"] == NATURE_FIGURE_GUIDE_URL

    browser = output["index"].read_text(encoding="utf-8")
    for text in (
        "Measured",
        "Candidate",
        "Model",
        "Scientific status",
        "NOT_ACCEPTED",
        "radial_profile.csv",
        "source_data.npz",
        "caption.txt",
        "manifest.json",
        "peak_map.png",
    ):
        assert text in browser
    assert "http://" not in browser
    assert "https://" not in browser
    assert "file:///" not in browser
    assert "E:\\" not in browser
    for relative in re.findall(r'href="([^"]+)"|src="([^"]+)"', browser):
        target = relative[0] or relative[1]
        assert not target.startswith(("http:", "https:", "file:"))
        assert (output["index"].parent / target).is_file(), target

    caption = output["caption"].read_text(encoding="utf-8")
    assert "asinh(I)" in caption
    assert "signed log1p" not in caption

    manifest = json.loads(output["manifest"].read_text(encoding="utf-8"))
    for name in ("index.html", "radial_profile.csv", "figure_qa.json"):
        assert name in manifest["sha256"]
        assert hashlib.sha256((output["index"].parent / name).read_bytes()).hexdigest() == manifest["sha256"][name]
