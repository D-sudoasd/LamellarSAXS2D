from __future__ import annotations

import numpy as np
import pytest

from butterfly_saxs.visualization import plot_fit_diagnostics, plot_parameter_evolution


def test_fit_diagnostics_uses_shared_data_scale_and_centered_residual(tmp_path):
    q = np.linspace(-0.2, 0.2, 32)
    qx, qy = np.meshgrid(q, q)
    observed = np.exp(-((qx / 0.08) ** 2 + (qy / 0.04) ** 2))
    model = observed * 0.9
    output = tmp_path / "diagnostic.png"

    fig = plot_fit_diagnostics(observed, model, qx, qy, output=output)

    assert output.exists() and output.stat().st_size > 1_000
    observed_image = fig.axes[0].images[0]
    model_image = fig.axes[1].images[0]
    residual_image = fig.axes[2].images[0]
    assert observed_image.get_clim() == model_image.get_clim()
    lo, hi = residual_image.get_clim()
    assert lo == pytest.approx(-hi)


def test_parameter_evolution_retains_failed_frames(tmp_path):
    rows = [
        {"time_s": 0.0, "status": "ok", "theta_deg": 10.0, "theta_deg_stderr": 0.2},
        {"time_s": 1.0, "status": "failed", "theta_deg": None},
        {"time_s": 2.0, "status": "ok", "theta_deg": 12.0, "theta_deg_stderr": 0.3},
    ]
    output = tmp_path / "evolution.png"

    fig = plot_parameter_evolution(rows, parameters=("theta_deg",), output=output)

    assert output.exists() and output.stat().st_size > 1_000
    assert len(fig.axes[0].collections) >= 1
