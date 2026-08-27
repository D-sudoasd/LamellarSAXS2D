from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from butterfly_saxs.batch import FrameFitResult, FrameRef
from butterfly_saxs.export import export_batch
from butterfly_saxs.visualization import plot_fit_diagnostics


def _frame(tmp_path: Path) -> FrameFitResult:
    return FrameFitResult(
        frame=FrameRef(tmp_path / "frame1.tif"),
        result={"parameters": {"spacing": {"value": 1.0, "unit": "nm"}}},
    )


def test_export_batch_checks_all_targets_before_writing(tmp_path: Path) -> None:
    output = tmp_path / "exports"
    output.mkdir()
    sentinel = output / "provenance.json"
    sentinel.write_text('{"sentinel": true}\n', encoding="utf-8")

    with pytest.raises(FileExistsError, match="force=True"):
        export_batch([_frame(tmp_path)], output)

    assert sentinel.read_text(encoding="utf-8") == '{"sentinel": true}\n'
    assert not (output / "frame_summary.csv").exists()

    outputs = export_batch([_frame(tmp_path)], output, force=True)
    assert set(outputs) == {
        "frame_summary",
        "parameters_long",
        "ridge_points",
        "ellipse_fit",
        "ellipse_fit_jsonl",
        "manifest",
        "provenance",
        "npz",
        "evolution_png",
    }
    assert json.loads(sentinel.read_text(encoding="utf-8"))["tool"] == "ButterflySAXS"


@pytest.mark.parametrize(
    ("q_unit", "expected"),
    [
        ("nm^-1", r"$q_x$ (nm$^{-1}$)"),
        ("Å^-1", r"$q_x$ (Å$^{-1}$)"),
        ("pixel-q", "$q_x$ (pixel-q)"),
        ("unknown", "$q_x$ (unknown)"),
    ],
)
def test_fit_diagnostics_axis_labels_follow_q_unit(q_unit: str, expected: str) -> None:
    q = np.linspace(-1.0, 1.0, 8)
    qx, qy = np.meshgrid(q, q)
    observed = np.ones_like(qx)
    model = observed * 0.9

    fig = plot_fit_diagnostics(observed, model, qx, qy, q_unit=q_unit)
    image_axes = [axis for axis in fig.axes if axis.images]
    assert len(image_axes) == 4
    assert image_axes[0].get_xlabel() == expected
    assert image_axes[0].get_ylabel() == expected.replace("q_x", "q_y")
    if q_unit != "nm^-1":
        assert all("nm" not in axis.get_xlabel() for axis in image_axes)
        assert all("nm" not in axis.get_ylabel() for axis in image_axes)
