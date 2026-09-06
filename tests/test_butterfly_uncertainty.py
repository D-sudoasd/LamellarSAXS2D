from __future__ import annotations

import threading

import numpy as np
import pytest

from butterfly_saxs.benchmark_arcs import generate_arc_case
from butterfly_saxs.butterfly_uncertainty import (
    _copy_qmap_with_perturbation,
    evaluate_refit_sensitivity,
    resample_butterfly,
)
from butterfly_saxs.cancellation import AnalysisCancelled
from butterfly_saxs.geometry import GeometryMaps


def test_declared_calibration_without_qmap_fails_before_callback():
    calls = []
    with pytest.raises(ValueError, match="requires qmap"):
        resample_butterfly(np.ones((8, 8)), None,
            refit=lambda image, options: calls.append(options),
            options={"resamples": 1, "center_sigma_px": .2, "qcal_sigma": .01})
    assert calls == []


def test_outer_ok_cannot_override_failed_candidate_status():
    result = resample_butterfly(np.ones((8, 8)), None,
        refit=lambda image, options: {"status": "ok", "candidate_fit": {
            "success": False, "a": 1., "b": .2, "axis_ratio": .2, "theta_deg": 10.}},
        options={"resamples": 1})
    assert result["successes"] == 0
    assert result["failed_refits"] == 1
    assert "conflicting" in result["errors"][0]["error"]


def _variable_refit(image: np.ndarray, options: dict) -> dict:
    assert image.ndim == 2
    assert options["resamples"] == 0
    assert options["stage"] == "evaluate"
    qmap = options.get("qmap")
    assert qmap is not None
    mean = float(np.nanmean(image))
    return {
        "candidate_fit": {
            "a": 0.8 + mean,
            "b": 0.2 + 0.5 * mean,
            "axis_ratio": (0.2 + 0.5 * mean) / (0.8 + mean),
            "theta_deg": 5.0 + 10.0 * mean,
            "success": True,
        }
    }


def test_spatial_block_resampling_is_deterministic_and_preserves_image_shape() -> None:
    case = generate_arc_case("ellipse_ratio_100", seed=11, shape=(36, 40))
    seen: list[dict] = []

    def callback(image: np.ndarray, options: dict) -> dict:
        seen.append(options)
        return _variable_refit(image, options)

    options = {"resamples": 8, "seed": 91, "block_shape": (4, 5), "center_sigma_px": 0.2, "qcal_sigma": 0.01}
    first = resample_butterfly(case["image"], case["qmap"], mask=case["mask"], q_window=(0.1, 1.0), refit=callback, options=options)
    second = resample_butterfly(case["image"], case["qmap"], mask=case["mask"], q_window=(0.1, 1.0), refit=_variable_refit, options=options)
    assert first["successes"] == 8
    assert first["settings"]["residual_block_status"] == "provisional_empirical_model"
    assert first["settings"]["iid_pixel_bootstrap"] is False
    assert first["settings"]["image_shape_preserved"] is True
    assert first["settings"]["cropped_shape"] != list(case["image"].shape)
    assert first["intervals"] == second["intervals"]
    assert all(draw["fit"]["success"] for draw in first["draws"])
    assert all(item["resamples"] == 0 and item["stage"] == "evaluate" for item in seen)
    assert first["source"]["calibrated_confidence"] is False
    assert first["source"]["instrument_psf_uncertainty"] == "absent"


def test_failed_topology_and_refit_are_explicitly_tallied() -> None:
    case = generate_arc_case("missing_branch", shape=(28, 30))
    calls = 0

    def callback(_image: np.ndarray, _options: dict) -> dict:
        nonlocal calls
        calls += 1
        if calls % 2:
            return {"success": False, "failure_kind": "topology", "failure_code": "missing_branch", "message": "observed branch is absent"}
        raise RuntimeError("optimizer failed")

    result = resample_butterfly(case["image"], case["qmap"], refit=callback, options={"resamples": 6, "seed": 4})
    assert result["successes"] == 0
    assert result["failed_topology"] == 3
    assert result["failed_refits"] == 3
    assert result["failure_counts"]["topology"] == 3
    assert result["failure_counts"]["refit"] == 3
    assert all(bounds == [None, None] for bounds in result["intervals"].values())
    assert len(result["per_replicate_errors"]) == 6
    assert all(error is not None for error in result["per_replicate_errors"])
    assert result["draws"][0]["failure_code"] == "missing_branch"


def test_explicit_topology_flag_wins_over_a_numeric_candidate() -> None:
    case = generate_arc_case("ellipse_ratio_100", shape=(22, 24))

    def callback(_image: np.ndarray, _options: dict) -> dict:
        return {"candidate_fit": {"a": 1.0, "b": 0.2, "theta_deg": 2.0, "success": True}, "topology_failed": True}

    result = resample_butterfly(case["image"], case["qmap"], refit=callback, options={"resamples": 2})
    assert result["successes"] == 0
    assert result["failed_topology"] == 2


def test_production_geometry_maps_are_perturbed_without_mutation() -> None:
    rows, cols = 14, 16
    y = np.linspace(-0.5, 0.5, rows)
    x = np.linspace(-0.6, 0.6, cols)
    qx, qy = np.meshgrid(x, y)
    q = np.hypot(qx, qy)
    chi = np.arctan2(qy, qx)
    valid = np.ones((rows, cols), dtype=bool)
    valid[0, 0] = False
    geometry = GeometryMaps(
        q_nm_inv=q,
        chi_rad=chi,
        qx_nm_inv=qx,
        qy_nm_inv=qy,
        valid_mask=valid,
        metadata={"q_unit": "nm^-1", "test_marker": "production_geometry_maps"},
        fingerprint="geometry-test-fingerprint",
    )
    original_qx = geometry.qx.copy()
    original_qy = geometry.qy.copy()
    original_q = geometry.q.copy()
    received: list[object] = []

    def callback(_image: np.ndarray, options: dict) -> dict:
        received.append(options["qmap"])
        return {"candidate_fit": {"a": 1.0, "b": 0.2, "theta_deg": 2.0, "success": True}}

    result = resample_butterfly(
        np.ones((rows, cols), dtype=float),
        geometry,
        refit=callback,
        options={"resamples": 3, "seed": 13, "center_sigma_px": (0.4, 0.2), "qcal_sigma": 0.01},
    )
    assert result["successes"] == 3
    assert received
    for perturbed in received:
        assert isinstance(perturbed, dict)
        assert not np.array_equal(perturbed["qx"], original_qx)
        assert not np.array_equal(perturbed["qy"], original_qy)
        np.testing.assert_allclose(perturbed["q"], np.hypot(perturbed["qx"], perturbed["qy"]))
        np.testing.assert_allclose(perturbed["chi"], np.arctan2(perturbed["qy"], perturbed["qx"]))
        np.testing.assert_array_equal(perturbed["mask"], geometry.mask)
        np.testing.assert_array_equal(perturbed["valid_mask"], geometry.valid_mask)
        assert perturbed["metadata"] == geometry.metadata
        assert perturbed["q_unit"] == "nm^-1"
    np.testing.assert_array_equal(geometry.qx, original_qx)
    np.testing.assert_array_equal(geometry.qy, original_qy)
    np.testing.assert_array_equal(geometry.q, original_q)


@pytest.mark.parametrize(
    "callback_result",
    [
        {"candidate_fit": {"a": 1.0, "b": 0.2, "theta_deg": 2.0, "success": "false"}},
        {"candidate_fit": {"a": 1.0, "b": 0.2, "theta_deg": 2.0}},
        {"candidate_fit": {"a": 1.0, "b": 0.2, "theta_deg": 2.0, "success": False}, "message": "arc solver failed"},
    ],
)
def test_malformed_or_generic_solver_status_cannot_be_promoted_to_topology(callback_result: dict) -> None:
    case = generate_arc_case("ellipse_ratio_100", shape=(20, 22))
    result = resample_butterfly(case["image"], case["qmap"], refit=lambda _image, _options: callback_result, options={"resamples": 1})
    assert result["successes"] == 0
    assert result["failed_refits"] == 1
    assert result["failed_topology"] == 0
    assert result["draws"][0]["failure_kind"] == "refit"


def test_structured_topology_loss_is_counted_with_specific_code() -> None:
    case = generate_arc_case("missing_branch", shape=(20, 22))

    def callback(_image: np.ndarray, _options: dict) -> dict:
        return {
            "candidate_fit": {"a": 1.0, "b": 0.2, "theta_deg": 2.0, "success": False},
            "topology_failed": True,
            "topology_code": "missing_branch",
            "message": "one observed branch is absent",
        }

    result = resample_butterfly(case["image"], case["qmap"], refit=callback, options={"resamples": 1})
    assert result["successes"] == 0
    assert result["failed_topology"] == 1
    assert result["draws"][0]["failure_code"] == "missing_branch"


def test_count_validation_explicit_poisson_and_cancellation() -> None:
    case = generate_arc_case("ellipse_ratio_400", shape=(24, 26))
    with pytest.raises(ValueError, match="exceeds configured max_resamples"):
        resample_butterfly(case["image"], case["qmap"], refit=_variable_refit, options={"resamples": 3, "max_resamples": 2})
    with pytest.raises(TypeError, match="resamples must be an integer"):
        resample_butterfly(case["image"], case["qmap"], refit=_variable_refit, options={"resamples": 2.5})
    with pytest.raises(ValueError, match="actual_counts declaration"):
        resample_butterfly(case["image"], case["qmap"], refit=_variable_refit, options={"resamples": 1, "noise_model": "poisson"})
    poisson = resample_butterfly(case["image"], case["qmap"], refit=_variable_refit, options={"resamples": 2, "noise_model": "poisson", "actual_counts": True, "count_scale": 100.0, "seed": 9})
    assert poisson["settings"]["noise_model"] == "poisson"
    assert poisson["settings"]["actual_counts_declared"] is True

    event = threading.Event()
    event.set()
    with pytest.raises(AnalysisCancelled):
        resample_butterfly(case["image"], case["qmap"], refit=_variable_refit, options={"resamples": 2}, cancel_event=event)


def test_max_resamples_cannot_escape_shared_hard_cap() -> None:
    case = generate_arc_case("ellipse_ratio_100", shape=(12, 14))
    with pytest.raises(ValueError, match="max_resamples must be <= 10000"):
        resample_butterfly(
            case["image"],
            case["qmap"],
            refit=_variable_refit,
            options={"resamples": 1, "max_resamples": 10_001},
        )


def test_zero_qmap_perturbation_uses_readonly_alias_without_source_mutation() -> None:
    qx = np.arange(12, dtype=float).reshape(3, 4)
    qy = np.arange(12, dtype=float).reshape(3, 4) / 10.0
    qmap = {"qx_map": qx, "qy_map": qy, "q_map": np.hypot(qx, qy), "bad_mask": np.zeros((1, 4), dtype=bool)}
    alias = _copy_qmap_with_perturbation(qmap, (3, 4), (0.0, 0.0), 1.0)
    assert alias["qx"].shape == (3, 4)
    assert alias["qx"].flags.writeable is False
    assert alias["mask"].flags.writeable is False
    with pytest.raises(ValueError):
        alias["qx"][0, 0] = 99.0
    np.testing.assert_array_equal(qmap["qx_map"], qx)


def test_center_and_qcal_perturbations_are_passed_to_callback() -> None:
    case = generate_arc_case("warped_qmap", shape=(24, 28))
    seen: list[dict] = []

    def callback(image: np.ndarray, options: dict) -> dict:
        seen.append(options)
        return {"a": 1.0, "b": 0.2, "theta_deg": 2.0, "success": True}

    result = resample_butterfly(case["image"], case["qmap"], refit=callback, options={"resamples": 5, "seed": 8, "center_sigma_px": (0.4, 0.2), "qcal_sigma": 0.02})
    assert result["successes"] == 5
    assert result["settings"]["instrument_uncertainty"]["center_status"] == "declared"
    assert result["settings"]["instrument_uncertainty"]["qcal_status"] == "declared"
    assert any(option["qmap_perturbation"]["qcal_scale"] != 1.0 for option in seen)
    assert all(option["qmap"].qx.shape == case["image"].shape for option in seen)


def test_sensitivity_helper_runs_scale_smoothing_and_mask_refits() -> None:
    case = generate_arc_case("ellipse_ratio_020", shape=(26, 28))
    result = evaluate_refit_sensitivity(
        case["image"],
        case["qmap"],
        _variable_refit,
        mask=case["mask"],
        options={"sensitivity": {"image_scales": (0.9, 1.1), "smoothing_sigmas": (0.0,), "masks": (case["mask"],)}},
    )
    assert result["successes"] == 4
    assert {record["type"] for record in result["variants"]} == {"image_scale", "smoothing", "mask"}
    assert all(record["fit"]["success"] for record in result["variants"])
    assert result["calibrated"] is False
