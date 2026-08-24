from __future__ import annotations

import numpy as np
import pytest

from butterfly_saxs.intensity import (
    DEFAULT_PARAMETERS,
    MODEL_FLAGS,
    deterministic_pixel_sample,
    default_intensity_parameters,
    double_ellipse_intensity,
    fit_intensity_model,
    parameter_values,
)
from butterfly_saxs.parameters import ParameterSet, ParameterSpec
from butterfly_saxs.synthetic import make_butterfly_sequence


def test_double_ellipse_has_four_envelopes_and_explicit_flags():
    q = np.linspace(-1.2, 1.2, 64)
    qx, qy = np.meshgrid(q, q)
    components = double_ellipse_intensity(
        qx,
        qy,
        {"a": 0.8, "b": 0.55, "theta": 0.2, "lobe_angle": 0.5, "angular_width": 0.1, "amplitude": 4.0},
        return_components=True,
    )
    assert components["intensity"].shape == qx.shape
    assert np.nanmax(components["branch_plus"]) > 0
    assert np.nanmax(components["branch_minus"]) > 0
    assert "empirical_model_only" in MODEL_FLAGS


def test_shorthand_parameters_expand_without_duplicate_model_degrees_of_freedom():
    q = np.linspace(-1.0, 1.0, 48)
    qx, qy = np.meshgrid(q, q)
    shorthand = double_ellipse_intensity(
        qx, qy, {"amplitude": 3.0, "radial_width": 0.025, "background": 0.0}
    )
    explicit = double_ellipse_intensity(
        qx,
        qy,
        {
            "amplitude_plus": 3.0,
            "amplitude_minus": 3.0,
            "radial_sigma": 0.025,
            "radial_gamma": 0.025,
            "background": 0.0,
        },
    )
    np.testing.assert_allclose(shorthand, explicit)


def test_plain_mapping_axis_ratio_remains_a_tied_geometry() -> None:
    values = parameter_values({"a": 0.7, "axis_ratio": 0.5})
    assert values["b"] == pytest.approx(0.35)
    # axis_ratio is the authoritative tie whenever both spellings are given.
    assert parameter_values({"a": 0.7, "b": 0.6, "axis_ratio": 0.5})["b"] == pytest.approx(0.35)

    q = np.linspace(-1.0, 1.0, 24)
    qx, qy = np.meshgrid(q, q)
    image = double_ellipse_intensity(qx, qy, values)
    initial = dict(DEFAULT_PARAMETERS)
    initial.update({"a": 0.7, "axis_ratio": 0.5})
    fixed = {name: name not in {"a", "b"} for name in DEFAULT_PARAMETERS}
    result = fit_intensity_model(
        image,
        {"qx": qx, "qy": qy},
        initial=initial,
        fixed=fixed,
        max_pixels=None,
        scales=(1.0,),
        max_nfev=2,
    )
    assert "axis_ratio" in result.covariance_names
    assert "b" not in result.covariance_names


def test_plain_mapping_rejects_duplicate_angle_units() -> None:
    with pytest.raises(ValueError, match="theta.*theta_deg"):
        parameter_values({"theta": 0.1, "theta_deg": 10.0})


def test_empirical_model_reference_axis_rotates_with_specimen_frame() -> None:
    axis = np.linspace(-1.1, 1.1, 41)
    qx, qy = np.meshgrid(axis, axis)
    parameters = {
        "a": 0.78,
        "axis_ratio": 0.61,
        "theta_deg": 14.0,
        "lobe_angle_deg": 58.0,
        "angular_width_deg": 7.0,
        "radial_sigma": 0.035,
        "radial_gamma": 0.035,
        "background": 0.02,
    }
    reference_deg = 37.0
    reference = np.deg2rad(reference_deg)
    qx_relative = np.cos(reference) * qx + np.sin(reference) * qy
    qy_relative = -np.sin(reference) * qx + np.cos(reference) * qy

    rotated_specimen = double_ellipse_intensity(
        qx,
        qy,
        parameters,
        reference_axis_deg=reference_deg,
    )
    canonical_specimen = double_ellipse_intensity(qx_relative, qy_relative, parameters)

    np.testing.assert_allclose(rotated_specimen, canonical_specimen, rtol=1e-12, atol=1e-12)


def test_background_components_are_monotonically_decreasing_in_q():
    q = np.linspace(0.0, 2.0, 300)
    background = double_ellipse_intensity(
        q,
        np.zeros_like(q),
        {
            "amplitude_plus": 0.0,
            "amplitude_minus": 0.0,
            "background": 0.2,
            "background_slope": 0.8,
            "background_curvature": 0.4,
            "background_amplitude": 0.5,
            "background_width": 0.7,
        },
    )
    assert np.all(np.diff(background) <= 1e-12)


def test_deterministic_sampling_is_reproducible():
    indices = np.arange(1000)
    first = deterministic_pixel_sample(indices, 80, seed=17)
    second = deterministic_pixel_sample(indices, 80, seed=17)
    different = deterministic_pixel_sample(indices, 80, seed=18)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, different)
    assert np.array_equal(first, np.sort(first))


def test_default_fit_uses_all_valid_pixels_once_and_reports_full_metrics():
    q = np.linspace(-1.0, 1.0, 16)
    qx, qy = np.meshgrid(q, q)
    truth = dict(DEFAULT_PARAMETERS)
    truth.update({"a": 0.82, "background": 0.08})
    image = double_ellipse_intensity(qx, qy, truth)
    image = np.asarray(image, dtype=float)
    image[0, 0] = np.nan

    initial = dict(truth)
    initial["a"] = 0.76
    fixed = {name: name != "a" for name in DEFAULT_PARAMETERS}
    result = fit_intensity_model(
        {"data": image},
        {"qx": qx, "qy": qy},
        initial=initial,
        fixed=fixed,
        scales=(0.25, 0.5, 1.0),
        max_nfev=8,
    )

    valid = np.isfinite(image) & np.isfinite(qx) & np.isfinite(qy)
    expected_residual = result.residual[valid.ravel()]
    expected_rmse = float(np.sqrt(np.mean(expected_residual**2)))
    expected_weighted_rmse = float(
        np.sqrt(np.mean((expected_residual / np.std(image[valid])) ** 2))
    )
    assert result.ndata == int(valid.sum())
    assert result.sampled_n == int(valid.sum())
    assert result.rmse == pytest.approx(expected_rmse)
    assert result.weighted_rmse == pytest.approx(expected_weighted_rmse)
    assert result.sample_rmse == pytest.approx(expected_rmse)
    assert len(result.scale_history) == 1
    assert result.scale_history[0]["n_pixels"] == int(valid.sum())


def test_explicit_sampling_keeps_full_pixel_metrics_and_sample_diagnostics():
    q = np.linspace(-1.0, 1.0, 20)
    qx, qy = np.meshgrid(q, q)
    truth = dict(DEFAULT_PARAMETERS)
    truth.update({"a": 0.82, "background": 0.08})
    image = np.asarray(double_ellipse_intensity(qx, qy, truth), dtype=float)
    image += np.linspace(-0.2, 0.2, image.size).reshape(image.shape)
    image[0, 0] = np.nan

    initial = dict(truth)
    initial["a"] = 0.70
    fixed = {name: name != "a" for name in DEFAULT_PARAMETERS}
    result = fit_intensity_model(
        {"data": image},
        {"qx": qx, "qy": qy},
        initial=initial,
        fixed=fixed,
        max_pixels=24,
        scales=(1.0,),
        seed=13,
        max_nfev=8,
    )

    valid = np.isfinite(image) & np.isfinite(qx) & np.isfinite(qy)
    full_residual = result.residual[valid.ravel()]
    sampled_residual = result.residual[result.sampled_indices]
    obs_scale = np.std(image[valid])
    assert result.ndata == int(valid.sum())
    assert result.sampled_n == len(result.sampled_indices) == 24
    assert result.rmse == pytest.approx(float(np.sqrt(np.mean(full_residual**2))))
    assert result.weighted_rmse == pytest.approx(
        float(np.sqrt(np.mean((full_residual / obs_scale) ** 2)))
    )
    assert result.sample_rmse == pytest.approx(float(np.sqrt(np.mean(sampled_residual**2))))
    assert not np.isclose(result.rmse, result.sample_rmse)


def test_fixed_nuisance_refinement_recovers_known_geometry():
    true = {
        "a": 0.78,
        "b": 0.53,
        "theta": 0.19,
        "lobe_angle": 0.47,
        "angular_width": 0.12,
        "radial_width": 0.035,
        "radial_sigma": 0.035,
        "radial_gamma": 0.035,
        "eta": 0.3,
        "amplitude": 6.0,
        "background": 0.08,
    }
    sequence = make_butterfly_sequence(1, shape=(64, 64), parameters=true, seed=5, noise_sigma=0.0)
    initial = dict(true)
    initial.update({"a": 0.72, "b": 0.60, "theta": 0.05, "lobe_angle": 0.40})
    variable = {"a", "b", "theta", "lobe_angle"}
    fixed = {name: name not in variable for name in DEFAULT_PARAMETERS}
    result = fit_intensity_model(
        sequence.frames[0],
        sequence.qmaps[0],
        initial,
        q_window=(0.25, 1.0),
        fixed=fixed,
        max_pixels=1800,
        seed=11,
        scales=(0.5, 1.0),
        max_nfev=250,
    )
    values = result.parameters
    assert result.success
    assert np.isclose(values["a"], true["a"], atol=0.05)
    assert np.isclose(values["b"], true["b"], atol=0.05)
    assert np.isclose(abs(values["theta"]), true["theta"], atol=0.06)
    assert np.isclose(values["lobe_angle"], true["lobe_angle"], atol=0.06)
    assert "empirical_model_only" in result.flags
    assert "nonunique_inverse_problem" in result.flags


def test_degree_input_remains_separate_from_app_arbitrary_metadata():
    qx = np.array([0.8])
    qy = np.array([0.0])
    values = {"a": 0.8, "b": 0.5, "theta_deg": 15.0, "phi_app_deg": 24.0, "alpha_candidate_deg": 11.0}
    intensity = double_ellipse_intensity(qx, qy, values)
    assert np.isfinite(intensity).all()


def test_per_pixel_sigma_produces_weighted_diagnostics_and_covariance():
    true = dict(DEFAULT_PARAMETERS)
    true.update({"a": 0.76, "b": 0.52, "theta": 0.17, "amplitude": 5.0, "background": 0.05})
    sequence = make_butterfly_sequence(1, shape=(56, 56), parameters=true, seed=9, noise_sigma=0.0)
    initial = dict(true)
    initial["a"] = 0.69
    fixed = {name: name != "a" for name in DEFAULT_PARAMETERS}
    sigma = np.full((56, 56), 0.02)
    result = fit_intensity_model(
        sequence.frames[0],
        sequence.qmaps[0],
        initial,
        q_window=(0.2, 1.0),
        fixed=fixed,
        sigma=sigma,
        max_pixels=1800,
        scales=(1.0,),
        max_nfev=160,
    )
    assert result.success
    assert result.weighting == "per_pixel_sigma"
    assert np.isfinite(result.weighted_rmse)
    assert result.covariance is not None and result.covariance.shape == (1, 1)
    assert result.covariance_names == ("a",)
    assert np.isfinite(result.stderr["a"])
    assert result.condition_number == pytest.approx(1.0)


def test_invalid_uncertainty_contract_fails_closed():
    sequence = make_butterfly_sequence(1, shape=(24, 24), seed=1)
    with pytest.raises(ValueError, match="either sigma or weights"):
        fit_intensity_model(
            sequence.frames[0], sequence.qmaps[0], sigma=np.ones((24, 24)), weights=np.ones((24, 24))
        )
    with pytest.raises(ValueError, match="sigma shape"):
        fit_intensity_model(sequence.frames[0], sequence.qmaps[0], sigma=np.ones((12, 12)))


def test_parameter_expression_is_recomputed_during_refinement():
    truth_params = default_intensity_parameters(a=0.78, axis_ratio=0.68, theta_deg=11.0)
    truth = parameter_values(truth_params)
    truth.update({"amplitude_plus": 5.0, "amplitude_minus": 5.0, "background": 0.04})
    sequence = make_butterfly_sequence(1, shape=(64, 64), parameters=truth, seed=12, noise_sigma=0.0)

    initial = default_intensity_parameters(a=0.70, axis_ratio=0.80, theta_deg=11.0)
    for name in initial.names:
        if name not in {"a", "axis_ratio", "b", "theta_deg", "lobe_angle_deg"}:
            initial[name].vary = False
    initial["amplitude_plus"].set_value(5.0)
    initial["amplitude_minus"].set_value(5.0)
    initial["background"].set_value(0.04)
    result = fit_intensity_model(
        sequence.frames[0],
        sequence.qmaps[0],
        initial,
        q_window=(0.2, 1.0),
        max_pixels=2600,
        scales=(0.5, 1.0),
        max_nfev=250,
    )
    resolved = result.parameters.resolve()
    assert result.success
    assert resolved["b"] == pytest.approx(resolved["a"] * resolved["axis_ratio"], rel=1e-12)
    assert resolved["a"] == pytest.approx(0.78, abs=0.04)
    assert resolved["axis_ratio"] == pytest.approx(0.68, abs=0.05)


def test_parameter_set_one_sided_bound_is_respected() -> None:
    q = np.linspace(-1.5, 1.5, 45)
    qx, qy = np.meshgrid(q, q)
    truth = dict(DEFAULT_PARAMETERS)
    truth.update({"a": 1.15, "b": 0.7, "amplitude_plus": 2.0, "amplitude_minus": 2.0})
    image = double_ellipse_intensity(qx, qy, truth)

    parameters = default_intensity_parameters(a=0.75, axis_ratio=0.7 / 1.15)
    for name, spec in list(parameters.spec_items()):
        if not spec.is_tied:
            parameters[name] = spec.copy(vary=False)
    parameters["a"] = ParameterSpec(0.75, max=0.9, vary=True)

    result = fit_intensity_model(
        image,
        {"qx": qx, "qy": qy},
        initial=parameters,
        max_pixels=None,
        scales=(1.0,),
        max_nfev=120,
    )
    assert isinstance(result.parameters, ParameterSet)
    assert result.parameters.resolve()["a"] <= 0.9 + 1e-8


def test_auto_scale_initial_handles_absolute_intensity_magnitude() -> None:
    q = np.linspace(-1.5, 1.5, 45)
    qx, qy = np.meshgrid(q, q)
    truth = parameter_values(default_intensity_parameters(a=1.0, axis_ratio=0.68, theta_deg=14.0))
    truth.update(
        {
            "amplitude_plus": 450.0,
            "amplitude_minus": 320.0,
            "background": 12.0,
        }
    )
    image = double_ellipse_intensity(qx, qy, truth)

    initial = default_intensity_parameters(a=1.0, axis_ratio=0.68, theta_deg=14.0)
    for name, spec in list(initial.spec_items()):
        if name not in {"amplitude_plus", "amplitude_minus", "background"} and not spec.is_tied:
            initial[name] = spec.copy(vary=False)

    result = fit_intensity_model(
        image,
        {"qx": qx, "qy": qy},
        initial,
        auto_scale_initial=True,
        max_pixels=None,
        max_nfev=120,
    )

    fitted = parameter_values(result.parameters)
    scaled_start = parameter_values(result.initial_parameters)
    assert result.success
    assert "initial_intensity_scale_estimated" in result.flags
    assert scaled_start["amplitude_plus"] > 100.0
    assert scaled_start["background"] > 5.0
    assert fitted["amplitude_plus"] == pytest.approx(450.0, rel=2e-3)
    assert fitted["amplitude_minus"] == pytest.approx(320.0, rel=2e-3)
    assert fitted["background"] == pytest.approx(12.0, rel=2e-3)
    assert result.rmse < 0.1
