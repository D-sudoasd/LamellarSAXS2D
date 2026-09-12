from __future__ import annotations

import numpy as np
import pytest

from butterfly_saxs.intensity import (
    DEFAULT_PARAMETERS,
    default_intensity_parameters,
    double_ellipse_intensity,
    fit_intensity_model,
    parameter_values,
)
from butterfly_saxs.parameters import ParameterSpec


BACKGROUND_TERMS = ("background_slope", "background_curvature", "background_amplitude")


def _background_case(scale: float = 1.0, *, masked_hot_pixel: bool = False):
    axis = np.linspace(-0.9, 0.9, 33)
    qx, qy = np.meshgrid(axis, axis)
    truth = dict(DEFAULT_PARAMETERS)
    truth.update(
        {
            "a": 0.85,
            "b": 0.60,
            "amplitude_plus": 0.0,
            "amplitude_minus": 0.0,
            "background": 0.0,
            "background_width": 0.7,
            "background_slope": 5.0 * scale,
            "background_curvature": 8.0 * scale,
            "background_amplitude": 3.0 * scale,
        }
    )
    observed = np.asarray(double_ellipse_intensity(qx, qy, truth), dtype=float)
    initial = dict(truth)
    for name in BACKGROUND_TERMS:
        initial[name] = 0.0
    fixed = {name: name not in BACKGROUND_TERMS for name in DEFAULT_PARAMETERS}
    valid = np.ones(observed.shape, dtype=bool)
    valid[:3, :3] = False
    if masked_hot_pixel:
        observed[:3, :3] = 1.0e9 * scale
    return observed, qx, qy, valid, truth, initial, fixed


def _fit_background_case(scale: float = 1.0, *, masked_hot_pixel: bool = False):
    observed, qx, qy, valid, truth, initial, fixed = _background_case(
        scale, masked_hot_pixel=masked_hot_pixel
    )
    result = fit_intensity_model(
        {"data": observed, "valid_mask": valid},
        {"qx": qx, "qy": qy},
        initial=initial,
        fixed=fixed,
        q_window=(0.0, 1.3),
        auto_scale_initial=True,
        robust_loss="linear",
        max_nfev=150,
    )
    return result, truth


def test_auto_scaled_background_bounds_follow_intensity_without_using_masked_outliers():
    reference, truth = _fit_background_case(1.0)
    tiny, tiny_truth = _fit_background_case(1.0e-9)
    large, large_truth = _fit_background_case(1.0e6)
    masked_outlier, _ = _fit_background_case(1.0e6, masked_hot_pixel=True)

    assert reference.success
    assert tiny.success
    assert large.success
    assert masked_outlier.success
    assert reference.automatic_intensity_scale is not None
    assert 0 < reference.bound_flag_intensity_scale <= reference.automatic_intensity_scale
    assert tiny.automatic_intensity_scale == pytest.approx(
        1.0e-9 * reference.automatic_intensity_scale, rel=1e-12
    )
    assert tiny.bound_flag_intensity_scale == pytest.approx(1e-9 * reference.bound_flag_intensity_scale)
    assert large.automatic_intensity_scale == pytest.approx(
        1.0e6 * reference.automatic_intensity_scale, rel=1e-12
    )
    assert large.bound_flag_intensity_scale == pytest.approx(1e6 * reference.bound_flag_intensity_scale)
    assert masked_outlier.automatic_intensity_scale == pytest.approx(
        large.automatic_intensity_scale, rel=1e-12
    )
    assert masked_outlier.bound_flag_intensity_scale == pytest.approx(
        large.bound_flag_intensity_scale, rel=1e-12
    )
    for name in BACKGROUND_TERMS:
        assert reference.parameters[name] == pytest.approx(truth[name], rel=1e-4)
        assert tiny.parameters[name] == pytest.approx(1.0e-9 * reference.parameters[name], rel=1e-4)
        assert tiny.parameters[name] / 1.0e-9 == pytest.approx(tiny_truth[name] / 1.0e-9, rel=1e-4)
        assert large.parameters[name] == pytest.approx(1.0e6 * reference.parameters[name], rel=1e-4)
        assert large.parameters[name] / 1.0e6 == pytest.approx(large_truth[name] / 1.0e6, rel=1e-4)
        assert masked_outlier.parameters[name] == pytest.approx(large.parameters[name], rel=1e-8)
        assert reference.effective_bounds[name][0] == pytest.approx(0.0)
        assert reference.effective_bounds[name][1] > reference.automatic_intensity_scale
        assert masked_outlier.effective_bounds[name] == large.effective_bounds[name]
        assert not reference.bound_flags[name]
        assert not tiny.bound_flags[name]
        assert not large.bound_flags[name]
        assert not masked_outlier.bound_flags[name]


def test_auto_scaled_bound_flags_retain_a_true_lower_bound_hit():
    axis = np.linspace(-0.9, 0.9, 33)
    qx, qy = np.meshgrid(axis, axis)
    initial = dict(DEFAULT_PARAMETERS)
    fixed = {name: name != "background_slope" for name in DEFAULT_PARAMETERS}
    result = fit_intensity_model(
        {"data": -np.ones_like(qx)},
        {"qx": qx, "qy": qy},
        initial=initial,
        fixed=fixed,
        q_window=(0.0, 1.3),
        auto_scale_initial=True,
        robust_loss="linear",
        max_nfev=150,
    )

    assert result.success
    assert result.parameters["background_slope"] == pytest.approx(0.0, abs=1e-20)
    assert result.bound_flags["background_slope"]


@pytest.mark.parametrize("auto_scale", [True, False])
@pytest.mark.parametrize("background", [1.0, 1.0e8])
@pytest.mark.parametrize("bounds", [None, {"background_slope": (0.0, 1.0)}])
def test_large_fixed_pedestal_does_not_mark_an_interior_coefficient_at_bound(background, auto_scale, bounds):
    axis = np.linspace(-0.9, 0.9, 33)
    qx, qy = np.meshgrid(axis, axis)
    truth = dict(DEFAULT_PARAMETERS)
    truth.update(background=background, background_slope=0.5,
                 amplitude_plus=0.0, amplitude_minus=0.0)
    observed = double_ellipse_intensity(qx, qy, truth)
    result = fit_intensity_model(
        {"data": observed}, {"qx": qx, "qy": qy}, initial=truth,
        fixed={name: name != "background_slope" for name in truth},
        bounds=bounds, auto_scale_initial=auto_scale,
        robust_loss="linear", q_window=(0.0, 1.3), max_nfev=100,
    )
    assert result.success
    assert result.parameters["background_slope"] == pytest.approx(0.5)
    assert result.rmse == pytest.approx(0.0, abs=1e-12)
    assert not result.bound_flags["background_slope"]
    assert result.bound_flag_intensity_scale < 1.0


@pytest.mark.parametrize("background", [1e-9, 1.0, 1e8])
@pytest.mark.parametrize("auto_scale", [False, True])
@pytest.mark.parametrize("bounds", [None, {"background_slope": (0.0, 1.0)}])
def test_nearly_exact_zero_solution_is_still_reported_at_its_lower_bound(background, auto_scale, bounds):
    axis = np.linspace(-.9, .9, 33)
    qx, qy = np.meshgrid(axis, axis)
    truth = dict(DEFAULT_PARAMETERS)
    truth.update(background=background, background_slope=0.0,
                 amplitude_plus=0.0, amplitude_minus=0.0)
    result = fit_intensity_model(
        {"data": double_ellipse_intensity(qx, qy, truth)}, {"qx": qx, "qy": qy},
        initial=truth, fixed={name: name != "background_slope" for name in truth},
        bounds=bounds, auto_scale_initial=auto_scale, robust_loss="linear", max_nfev=150,
    )
    assert result.success
    assert result.parameters["background_slope"] < 1e-8
    assert result.bound_flags["background_slope"]


@pytest.mark.parametrize("loss", ["linear", "soft_l1"])
def test_small_weighted_interior_and_tied_coefficient_are_not_projected_to_a_bound(loss):
    axis = np.linspace(-.9, .9, 33)
    qx, qy = np.meshgrid(axis, axis)
    parameters = default_intensity_parameters()
    for name, spec in list(parameters.spec_items()):
        if not spec.is_tied:
            parameters[name] = spec.copy(vary=False)
    parameters["background"] = ParameterSpec(1.0, vary=False)
    parameters["amplitude_plus"] = ParameterSpec(0.0, vary=False)
    parameters["amplitude_minus"] = ParameterSpec(0.0, vary=False)
    parameters["background_slope"] = ParameterSpec(1e-10, min=0.0, max=1.0)
    parameters["background_curvature"] = ParameterSpec(expr="2*background_slope")
    observed = double_ellipse_intensity(qx, qy, parameters)
    result = fit_intensity_model(
        {"data": observed}, {"qx": qx, "qy": qy}, initial=parameters,
        weights=np.linspace(.5, 2., observed.size).reshape(observed.shape),
        robust_loss=loss, f_scale=1., auto_scale_initial=True,
    )
    assert result.success
    assert result.parameters.resolve()["background_slope"] == pytest.approx(1e-10, abs=1e-15)
    assert not result.bound_flags["background_slope"]
    assert result.parameters["background_curvature"].is_tied


def test_auto_scaled_background_bounds_preserve_explicit_fixed_and_tied_parameters():
    axis = np.linspace(-0.9, 0.9, 33)
    qx, qy = np.meshgrid(axis, axis)
    initial = default_intensity_parameters(a=0.85, axis_ratio=0.60 / 0.85)
    truth = parameter_values(initial)
    truth.update(
        {
            "amplitude_plus": 0.0,
            "amplitude_minus": 0.0,
            "background": 0.0,
            "background_width": 0.7,
            "background_slope": 5.0,
            "background_curvature": 10.0,
            "background_amplitude": 3.0,
        }
    )
    observed = double_ellipse_intensity(qx, qy, truth)

    for name, spec in list(initial.spec_items()):
        if spec.is_tied:
            continue
        if name == "background_slope":
            initial[name] = ParameterSpec(0.0, min=0.0, max=0.5, vary=True)
        else:
            value = truth.get(name, spec.value)
            initial[name] = spec.copy(value=value, vary=False)
    initial["background_curvature"] = ParameterSpec(expr="background_slope * 2")

    result = fit_intensity_model(
        {"data": observed},
        {"qx": qx, "qy": qy},
        initial=initial,
        q_window=(0.0, 1.3),
        auto_scale_initial=True,
        robust_loss="linear",
        max_nfev=100,
    )

    fitted = result.parameters.resolve()
    assert result.success
    assert result.parameters["background_slope"].max == pytest.approx(0.5)
    assert result.effective_bounds["background_slope"] == pytest.approx((0.0, 0.5))
    assert fitted["background_slope"] == pytest.approx(0.5, abs=1e-6)
    assert result.bound_flags["background_slope"]
    assert result.parameters["background_curvature"].is_tied
    assert fitted["background_curvature"] == pytest.approx(2.0 * fitted["background_slope"])
    assert result.parameters["background_amplitude"].is_fixed
    assert fitted["background_amplitude"] == pytest.approx(3.0)
    assert result.parameters["amplitude_plus"].is_fixed
    assert fitted["amplitude_plus"] == pytest.approx(0.0)


def test_auto_scaling_disabled_does_not_change_user_initial_parameters():
    axis = np.linspace(-0.5, 0.5, 9)
    qx, qy = np.meshgrid(axis, axis)
    observed = np.ones_like(qx)
    valid = np.ones_like(qx, dtype=bool)
    valid[:3, :3] = False
    observed[:3, :3] = 1.0e9
    initial = dict(DEFAULT_PARAMETERS)
    initial.update(
        {
            "background_slope": 0.25,
            "background_curvature": 0.50,
            "background_amplitude": 0.75,
        }
    )
    before = dict(initial)
    fixed = {name: True for name in initial}

    result = fit_intensity_model(
        {"data": observed, "valid_mask": valid},
        {"qx": qx, "qy": qy},
        initial=initial,
        fixed=fixed,
        auto_scale_initial=False,
    )

    assert initial == before
    assert result.initial_parameters == before
    assert result.automatic_intensity_scale is None
    assert result.bound_flag_intensity_scale == 0.0  # constant input; no free parameters
    assert result.effective_bounds == {}
    for name in BACKGROUND_TERMS:
        assert result.parameters[name] == pytest.approx(before[name])
