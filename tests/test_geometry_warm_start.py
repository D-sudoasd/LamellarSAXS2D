from __future__ import annotations

import numpy as np
import pytest

from butterfly_saxs.analysis_config import ellipse_parameter_specs, validate_analysis_settings
from butterfly_saxs.batch import run_batch
from butterfly_saxs.pipeline import analyze_frame


def _ellipse_pattern() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    axis = np.linspace(-1.2, 1.2, 181)
    qx, qy = np.meshgrid(axis, axis)
    a, ratio, theta = 0.72, 0.18, np.deg2rad(18.0)
    image = np.full_like(qx, 0.05)
    for sign in (1.0, -1.0):
        angle = sign * theta
        u = np.cos(angle) * qx + np.sin(angle) * qy
        v = -np.sin(angle) * qx + np.cos(angle) * qy
        normalized_radius = np.sqrt((u / a) ** 2 + (v / (a * ratio)) ** 2)
        image += np.exp(-0.5 * ((normalized_radius - 1.0) / 0.018) ** 2)
    return image, qx, qy


def test_ellipse_warm_start_changes_only_free_initial_values() -> None:
    settings = {
        "ellipse": {
            "preset": "standard",
            "a": {"value": 0.5, "min": 0.2, "max": 0.8},
            "axis_ratio": {"value": 0.12, "min": 0.05, "max": 0.3, "vary": False},
            "fixed_center": True,
            "center_qx": 0.01,
            "center_qy": -0.02,
            "angle_deg": 5.0,
            "theta_min_deg": -20.0,
            "theta_max_deg": 20.0,
            "b_min": 0.05,
            "b_max": 0.3,
        }
    }
    specs = ellipse_parameter_specs(
        settings,
        q_window=(0.1, 1.0),
        initial_parameters={
            "a": 1.1,
            "b": 0.55,
            "axis_ratio": 0.5,
            "theta_deg": 45.0,
            "center_qx": 0.2,
            "center_qy": 0.3,
        },
    )

    assert specs is not None
    assert specs["a"] == {"value": 0.8, "min": 0.2, "max": 0.8, "vary": True}
    assert specs["axis_ratio"] == {"value": 0.12, "min": 0.05, "max": 0.3, "vary": False}
    assert specs["cx"] == {"value": 0.01, "vary": False}
    assert specs["cy"] == {"value": -0.02, "vary": False}
    assert specs["theta_deg"] == {"value": 20.0, "min": -20.0, "max": 20.0, "vary": True}
    assert specs["b"] == {
        "value": pytest.approx(0.096),
        "min": 0.05,
        "max": 0.3,
        "vary": False,
        "expr": "a*axis_ratio",
    }


def test_geometry_warm_start_reaches_real_batch_pipeline_and_refits_current_qmap(
    tmp_path,
) -> None:
    image, qx, qy = _ellipse_pattern()
    paths = [tmp_path / "frame_0.npz", tmp_path / "frame_1.npz"]
    np.savez_compressed(paths[0], image=image, qx=qx, qy=qy, q_unit=np.asarray("nm^-1"))
    # The observed intensity is unchanged, while this frame's calibrated q
    # coordinates move the fitted geometry.  Its fit must start from frame 0
    # and then follow the new frame's own measured ridge.
    np.savez_compressed(
        paths[1], image=image, qx=1.15 * qx, qy=1.15 * qy, q_unit=np.asarray("nm^-1")
    )
    # Feed the same complete canonical settings mapping that the service and
    # UI transport.  Its standard row is a display default, not a fit prior.
    config = validate_analysis_settings(
        {
            "q_window": [0.08, 0.95],
            "ridge_method": "radial_peak",
            "n_ridge_angles": 144,
            "n_radial_bins": 256,
            "n_angular_bins": 180,
            "ellipse_multistart": 1,
        }
    )
    received_initials = []

    def analyzer(path, initial_parameters=None):
        received_initials.append(initial_parameters)
        return analyze_frame(
            path,
            initial_parameters=initial_parameters,
            config=config,
            full2d=False,
        )

    batch = run_batch(paths, analyzer, mode="warm_start")

    assert all(item.status in {"ok", "warning"} for item in batch.frame_results)
    assert batch.frame_results[1].warm_start_from is not None
    first, second = [item.result for item in batch.frame_results]
    assert received_initials[0] is None
    assert received_initials[1]["a"] == pytest.approx(first.ellipse_fit["parameters"]["a"])
    assert len(second.ridges) > 100
    assert second.ellipse_fit["status"] == "ok"
    second_start = second.ellipse_fit["candidate_solutions"][0]["start_values"]
    first_values = first.ellipse_fit["parameters"]
    first_start = first.ellipse_fit["candidate_solutions"][0]["start_values"]
    assert first_start["a"] != pytest.approx(0.515)
    assert first_start["axis_ratio"] != pytest.approx(0.7)
    assert second_start["a"] == pytest.approx(first_values["a"])
    assert second_start["axis_ratio"] == pytest.approx(first_values["axis_ratio"])
    assert second_start["theta_deg"] == pytest.approx(first_values["theta_deg"])
    assert second.ellipse_fit["parameters"]["a"] > first_values["a"] * 1.05


def test_butterfly_analysis_keeps_standard_geometry_data_driven() -> None:
    settings = validate_analysis_settings({"ridge_method": "butterfly_curvature"})
    assert settings["ellipse_preset"] == "standard"
    assert settings["ellipse"]["preset"] == "standard"
    assert settings["ellipse"]["axis_ratio_min"] is None
    assert settings["ellipse"]["axis_ratio_max"] is None
    assert settings["ellipse"]["fixed_center"] is False
    assert ellipse_parameter_specs(
        {"ellipse": settings["ellipse"]}, q_window=(0.08, 0.95)
    ) is None

    explicit = validate_analysis_settings(
        {"ridge_method": "butterfly_curvature", "ellipse_preset": "flat_ellipse"}
    )
    assert explicit["ellipse_preset"] == "flat_ellipse"
    assert explicit["ellipse"]["axis_ratio_max"] == pytest.approx(0.35)


def test_standard_solver_controls_survive_validation_and_reach_real_pipeline(
    tmp_path, monkeypatch
) -> None:
    from butterfly_saxs import observables as observable_module

    image, qx, qy = _ellipse_pattern()
    path = tmp_path / "solver_controls.npz"
    np.savez_compressed(path, image=image, qx=qx, qy=qy, q_unit=np.asarray("nm^-1"))

    settings = validate_analysis_settings(
        {
            "q_window": [0.08, 0.95],
            "ridge_method": "radial_peak",
            "n_ridge_angles": 144,
            "n_radial_bins": 256,
            "n_angular_bins": 180,
            "ellipse": {
                "preset": "standard",
                "multistart": 2,
                "residual": "geometric",
            },
            "butterfly": {"sensitivity": False, "resamples": 0},
        }
    )
    assert settings["ellipse_multistart"] == 2
    assert settings["ellipse_residual"] == "geometric"
    assert settings["ellipse"]["multistart"] == 2
    assert settings["ellipse"]["residual"] == "geometric"
    assert ellipse_parameter_specs(
        {"ellipse": settings["ellipse"]}, q_window=(0.08, 0.95)
    ) is None

    actual_measure = observable_module.measure_observables
    received = {}

    def observe_real_solver(*args, **kwargs):
        received.update(
            ellipse_multistart=kwargs["ellipse_multistart"],
            ellipse_residual=kwargs["ellipse_residual"],
            ellipse_parameters=kwargs["ellipse_parameters"],
        )
        return actual_measure(*args, **kwargs)

    monkeypatch.setattr(observable_module, "measure_observables", observe_real_solver)
    result = analyze_frame(path, config=settings, full2d=False)

    assert received["ellipse_multistart"] == 2
    assert received["ellipse_residual"] == "geometric"
    assert received["ellipse_parameters"] is None
    assert result.ellipse_fit["multistart_count"] == 2
    assert result.analysis["ellipse_multistart"] == 2
    assert result.analysis["ellipse_residual"] == "geometric"
    assert "ellipse_constraints_active" not in result.ellipse_fit["flags"]


@pytest.mark.parametrize(
    ("ellipse", "root_controls", "expected_multistart", "expected_residual"),
    [
        (
            {"preset": "standard", "multistart": 2, "residual": "distance"},
            {},
            2,
            "geometric",
        ),
        (
            {"preset": "standard", "multistart": 2, "residual": "geometric"},
            {"ellipse_multistart": 5, "ellipse_residual": "sampson"},
            2,
            "geometric",
        ),
        (
            {"preset": "standard"},
            {"ellipse_multistart": 3, "ellipse_residual": "geometric"},
            3,
            "geometric",
        ),
    ],
)
def test_standard_solver_control_roundtrip_keeps_geometry_unconstrained(
    ellipse, root_controls, expected_multistart, expected_residual
) -> None:
    settings = validate_analysis_settings(
        {
            **root_controls,
            "ellipse": ellipse,
            "butterfly": {"sensitivity": False, "resamples": 0},
        }
    )

    assert settings["ellipse_multistart"] == expected_multistart
    assert settings["ellipse"]["multistart"] == expected_multistart
    assert settings["ellipse_residual"] == expected_residual
    assert settings["ellipse"]["residual"] == expected_residual
    assert ellipse_parameter_specs(
        {"ellipse": settings["ellipse"]}, q_window=(0.08, 0.95)
    ) is None


def test_standard_nested_multistart_reaches_actual_butterfly_arc_solver(
    monkeypatch,
) -> None:
    from butterfly_saxs import butterfly as butterfly_module
    from butterfly_saxs.benchmark_arcs import generate_arc_case

    case = generate_arc_case("ellipse_ratio_400", seed=123, shape=(64, 64))
    settings = validate_analysis_settings(
        {
            "ridge_method": "butterfly_curvature",
            "q_window": [0.05, 1.1],
            "n_ridge_angles": 72,
            "n_radial_bins": 128,
            "n_angular_bins": 96,
            "ellipse": {
                "preset": "standard",
                "multistart": 2,
                "residual": "geometric",
            },
            "butterfly": {
                "stage": "evaluate",
                "resamples": 0,
                "sensitivity": False,
            },
        }
    )
    actual_analyze_butterfly = butterfly_module.analyze_butterfly
    received = {}

    def fit_real_arcs(*args, **kwargs):
        received.update(
            multistart=kwargs["multistart"],
            parameters=kwargs["parameters"],
        )
        return actual_analyze_butterfly(*args, **kwargs)

    monkeypatch.setattr(butterfly_module, "analyze_butterfly", fit_real_arcs)
    result = analyze_frame(
        case["image"],
        qmap=case["qmap"],
        mask=case["mask"],
        config=settings,
        full2d=False,
    )

    assert received == {"multistart": 2, "parameters": None}
    assert result.butterfly is not None
    assert result.butterfly["candidate_fit"]["multistart_count"] == 2
