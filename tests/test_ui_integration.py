from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

from butterfly_saxs.intensity import DEFAULT_PARAMETERS, double_ellipse_intensity
from butterfly_saxs.service import ButterflyAnalysisService
from butterfly_saxs.ui import create_app


def test_service_preview_and_roi_change_valid_pixel_count() -> None:
    service = ButterflyAnalysisService()
    image = np.ones((12, 14), dtype=float)
    qy, qx = np.indices(image.shape, dtype=float)
    state = service.set_observed(image, qx=qx, qy=qy)
    before = service.preview(parameters=service.parameters, payload=state)
    assert before["metrics"]["ndata"] == image.size
    state["rois"] = [{"type": "rectangle", "x0": 0, "y0": 0, "x1": 4, "y1": 5}]
    after = service.preview(parameters=service.parameters, payload=state)
    assert after["metrics"]["ndata"] < before["metrics"]["ndata"]
    assert after["metrics"]["valid_fraction"] < 1.0


def test_service_preserves_loaded_detector_valid_mask(tmp_path) -> None:
    path = tmp_path / "masked.npy"
    image = np.ones((16, 18), dtype=float)
    np.save(path, image)
    valid = np.ones_like(image, dtype=bool)
    valid[:3, :5] = False
    service = ButterflyAnalysisService()
    state = service.load_image(path, valid_mask=valid)
    result = service.preview(parameters=service.parameters, payload=state)
    assert result["metrics"]["ndata"] == int(valid.sum())
    assert result["metrics"]["valid_fraction"] == pytest.approx(float(valid.mean()))


def test_service_preview_resolves_ties_and_degree_parameters() -> None:
    service = ButterflyAnalysisService()
    y, x = np.indices((18, 20), dtype=float)
    qx = (x - 9.5) / 5.0
    qy = (y - 8.5) / 5.0
    state = service.set_observed(np.zeros_like(qx), qx=qx, qy=qy)
    specs = service.parameters
    specs["a"]["value"] = 1.2
    specs["axis_ratio"]["value"] = 0.5
    # Stale values must be ignored because these two rows are tied.
    specs["b"]["value"] = 9.0
    specs["b"]["expr"] = "a*axis_ratio"
    specs["theta_deg"]["value"] = 17.0
    specs["amplitude_plus"]["value"] = 2.0
    specs["amplitude_minus"]["value"] = 99.0
    specs["amplitude_minus"]["expr"] = "amplitude_plus/2"

    result = service.preview(parameter_specs=specs, parameters=specs, payload=state)
    expected_parameters = dict(DEFAULT_PARAMETERS)
    expected_parameters.update(
        {
            "a": 1.2,
            "b": 0.6,
            "theta": np.deg2rad(17.0),
            "amplitude_plus": 2.0,
            "amplitude_minus": 1.0,
        }
    )
    expected = double_ellipse_intensity(qx, qy, expected_parameters)
    np.testing.assert_allclose(result["model"], expected)


def test_real_service_is_default_and_project_json_rejects_nan(qtbot, tmp_path) -> None:
    app, window = create_app([])
    del app
    qtbot.addWidget(window)
    assert isinstance(window.engine, ButterflyAnalysisService)
    row = window.parameter_model.parameter_dict()["theta_deg"]
    assert set(("value", "minimum", "maximum", "vary", "expression", "unit", "stderr")) <= set(row)

    project = tmp_path / "project.json"
    assert window.save_project(project)
    parsed = json.loads(project.read_text(encoding="utf-8"))
    assert "NaN" not in project.read_text(encoding="utf-8")
    assert parsed["parameters"]["theta_deg"]["unit"] == "degree"
    bad = tmp_path / "bad.json"
    bad.write_text('{"parameters": {"theta_deg": {"value": NaN}}}', encoding="utf-8")
    assert not window.load_project(bad)
    window.close()


def test_gui_input_option_loads_image_and_roi(qtbot, tmp_path) -> None:
    image_path = tmp_path / "frame.npy"
    np.save(image_path, np.ones((10, 10), dtype=np.float32))
    app, window = create_app(["--input", str(image_path), "--no-auto-preview"])
    del app
    qtbot.addWidget(window)
    assert window._source_path == str(image_path)
    assert window._observed.shape == (10, 10)
    assert window.set_exclusion_roi((0, 0, 4, 4))
    assert int(np.count_nonzero(window._external_mask)) > 0
    window.close()
