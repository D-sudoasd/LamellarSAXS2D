from __future__ import annotations

import numpy as np
import pytest

from butterfly_saxs.intensity import (
    default_intensity_parameters,
    double_ellipse_intensity,
    fit_intensity_model,
)


def test_full2d_multistart_is_deterministic_and_keeps_tied_bounds() -> None:
    axis = np.linspace(-1.0, 1.0, 36)
    qx, qy = np.meshgrid(axis, axis)
    truth = {
        "a": 0.82,
        "axis_ratio": 0.16,
        "theta": 0.33,
        "lobe_angle": 0.48,
        "angular_width": 0.12,
        "radial_sigma": 0.05,
        "radial_gamma": 0.05,
        "eta": 0.25,
        "amplitude_plus": 8.0,
        "amplitude_minus": 6.5,
        "background": 0.12,
    }
    image = double_ellipse_intensity(qx, qy, truth)
    initial = default_intensity_parameters(a=0.42, axis_ratio=0.7, theta_deg=8.0)
    for name, spec in list(initial.spec_items()):
        if name not in {"a", "axis_ratio", "theta", "lobe_angle"} and not spec.is_tied:
            initial[name] = spec.copy(vary=False)
    initial["a"] = initial["a"].copy(min=0.2, max=1.5)
    initial["axis_ratio"] = initial["axis_ratio"].copy(min=0.02, max=0.9)
    initial["theta"] = initial["theta"].copy(min=-1.2, max=1.2)
    initial["lobe_angle"] = initial["lobe_angle"].copy(min=0.05, max=1.3)

    first = fit_intensity_model(
        {"data": image},
        {"qx": qx, "qy": qy},
        initial,
        q_window=(0.15, 1.0),
        max_pixels=500,
        seed=17,
        scales=(1.0,),
        multistart=3,
        max_nfev=50,
    )
    second = fit_intensity_model(
        {"data": image},
        {"qx": qx, "qy": qy},
        initial,
        q_window=(0.15, 1.0),
        max_pixels=500,
        seed=17,
        scales=(1.0,),
        multistart=3,
        max_nfev=50,
    )

    assert first.success
    assert first.multistart_count == 3
    assert len(first.candidate_solutions) == 3
    assert first.selected_start_index == second.selected_start_index
    assert len(first.candidate_solutions) == len(second.candidate_solutions)
    for left, right in zip(first.candidate_solutions, second.candidate_solutions):
        assert left["start_index"] == right["start_index"]
        assert left["success"] == right["success"]
        assert left["start_values"].keys() == right["start_values"].keys()
        np.testing.assert_allclose(
            list(left["start_values"].values()),
            list(right["start_values"].values()),
            rtol=1.0e-12,
            atol=1.0e-12,
        )
        assert left["values"].keys() == right["values"].keys()
        np.testing.assert_allclose(
            list(left["values"].values()),
            list(right["values"].values()),
            rtol=1.0e-8,
            atol=1.0e-10,
        )
        assert left["cost"] == pytest.approx(right["cost"], rel=1.0e-8, abs=1.0e-10)
        assert left["full_cost"] == pytest.approx(right["full_cost"], rel=1.0e-8, abs=1.0e-10)
    assert "deterministic_multistart" in first.flags
    assert first.parameters["b"].is_tied
    resolved = first.parameters.resolve()
    assert resolved["b"] == pytest.approx(resolved["a"] * resolved["axis_ratio"])
    assert resolved["a"] <= 1.5 + 1.0e-8
    assert resolved["axis_ratio"] <= 0.9 + 1.0e-8
    assert first.scale_history[0]["multistart_count"] == 3


def test_full2d_multistart_rejects_non_integer_count() -> None:
    axis = np.linspace(-0.8, 0.8, 12)
    qx, qy = np.meshgrid(axis, axis)
    with pytest.raises((TypeError, ValueError), match="multistart"):
        fit_intensity_model(
            {"data": np.ones_like(qx)},
            {"qx": qx, "qy": qy},
            multistart=1.5,
            max_nfev=2,
        )


def test_full2d_multistart_alias_is_accepted() -> None:
    axis = np.linspace(-0.8, 0.8, 12)
    qx, qy = np.meshgrid(axis, axis)
    result = fit_intensity_model(
        {"data": np.ones_like(qx)},
        {"qx": qx, "qy": qy},
        full2d_multistart=2,
        max_nfev=2,
    )
    assert result.multistart_count == 2
    assert len(result.candidate_solutions) == 2
