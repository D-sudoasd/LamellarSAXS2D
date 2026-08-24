from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

from PySide6 import QtCore

from butterfly_saxs.ui import MainWindow


class _Engine:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, float]]] = []

    def preview(self, *, parameters, payload):
        self.calls.append(("preview", dict(parameters)))
        observed = np.asarray(payload["observed"], dtype=float)
        model = observed * 0.9
        return {
            "observed": observed,
            "model": model,
            "residual": observed - model,
            "ridges": [{"qx": 0.1, "qy": 0.2}, {"qx": 0.1, "qy": -0.2}],
            "ellipse_fit": {
                "ellipses": [
                    {"a": 0.2, "b": 0.1, "angle_deg": 15.0},
                    {"a": 0.2, "b": 0.1, "angle_deg": -15.0},
                ]
            },
            "metrics": {"rmse": 0.1, "ndata": int(observed.size), "flags": ["ok"]},
        }


def test_workbench_offscreen_smoke(qtbot, tmp_path):
    engine = _Engine()
    window = MainWindow(engine=engine, auto_preview=False)
    qtbot.addWidget(window)
    window.set_observed_data(np.ones((12, 12)))

    assert window.windowTitle() == "LamellarSAXS2D · 2D Refinement"
    assert {"observed", "model", "residual", "overlay"} <= set(window.views.views)
    assert window.parameter_table.model() is window.parameter_model
    assert [window.parameter_model.headerData(i, QtCore.Qt.Orientation.Horizontal) for i in range(8)] == [
        "Parameter",
        "Value",
        "Min",
        "Max",
        "Vary",
        "Expr",
        "Unit",
        "Stderr",
    ]

    assert window.set_parameter("theta_deg", 25.0)
    generation = window.request_preview()
    assert generation > 0
    qtbot.waitUntil(lambda: bool(engine.calls), timeout=2_000)
    qtbot.waitUntil(lambda: window.rmse_label.text().startswith("RMSE: 0.1"), timeout=2_000)
    assert window.ndata_label.text() == "ndata: 144"
    assert window.flags_label.text() == "flags: ok"
    assert len(window.views.overlay.ridge_points) == 2
    assert len(window.views.overlay.ellipses) == 2

    project = tmp_path / "smoke.json"
    assert window.save_project(project)
    assert window.set_parameter("theta_deg", 5.0)
    assert window.load_project(project)
    assert window.parameters["theta_deg"] == pytest.approx(25.0)
    window.close()


def test_parameter_change_is_debounced(qtbot):
    engine = _Engine()
    window = MainWindow(engine=engine, auto_preview=True, debounce_ms=35)
    qtbot.addWidget(window)
    window.set_observed_data(np.ones((4, 4)))
    assert window.set_parameter("theta_deg", 10.0)
    assert window.set_parameter("theta_deg", 20.0)
    qtbot.waitUntil(lambda: len(engine.calls) == 1, timeout=2_000)
    assert engine.calls[0][1]["theta_deg"] == pytest.approx(20.0)
    window.close()
