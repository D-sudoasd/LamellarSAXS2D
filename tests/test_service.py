from __future__ import annotations

from pathlib import Path
import threading
from types import SimpleNamespace

import numpy as np
import pytest

import butterfly_saxs.service as service_module
from butterfly_saxs.batch import FrameRef
from butterfly_saxs.io import LoadedImage
from butterfly_saxs.service import ButterflyAnalysisService


class _FakeFit:
    flags = ()
    success = True
    nfev = 1
    weighted_rmse = 0.0
    condition_number = 1.0
    stderr = {}
    bound_flags = {}
    rmse = 0.0
    parameters = {}

    def __init__(self, shape: tuple[int, int]) -> None:
        self.model_image = np.ones(shape, dtype=float)


def test_service_marks_unusable_pixels_and_reports_q_units() -> None:
    image = np.ones((12, 14), dtype=float)
    yy, xx = np.indices(image.shape, dtype=float)
    valid = np.ones_like(image, dtype=bool)
    valid[0, 0] = False
    qmap = {
        "qx": xx,
        "qy": yy,
        "q": np.hypot(xx, yy),
        "q_unit": "pixel-q",
        "valid_mask": valid,
        "mask": ~valid,
    }
    service = ButterflyAnalysisService()
    state = service.set_observed(image, qmap=qmap)

    result = service.preview(parameters=service.parameters, payload=state)

    for name in ("a", "b", "radial_sigma", "radial_gamma", "background_width"):
        assert result["parameters"][name]["unit"] == "pixel-q"
    np.testing.assert_array_equal(result["valid_mask"], valid)
    assert np.isnan(result["model"][0, 0])
    assert np.isnan(result["residual"][0, 0])
    if result["ellipse_fit"] is not None:
        assert result["ellipse_fit"]["Ln_from_minor_axis_nm"] is None
        assert result["ellipse_fit"]["Lz_from_draw_axis_nm"] is None


def test_service_converts_explicit_angstrom_inverse_qmap_to_nm_inverse() -> None:
    image = np.ones((3, 4), dtype=float)
    qx = np.full(image.shape, 0.1)
    qy = np.zeros(image.shape)
    service = ButterflyAnalysisService()

    payload = service.set_observed(
        image,
        qmap={"qx": qx, "qy": qy, "q": np.hypot(qx, qy), "q_unit": "Å^-1"},
    )

    assert payload["qmap"]["q_unit"] == "nm^-1"
    assert payload["qmap"]["source_q_unit"] == "Å^-1"
    assert payload["qmap"]["q_conversion_factor_to_nm_inv"] == 10.0
    assert np.allclose(payload["qmap"]["q"], 1.0)


def test_service_rejects_q_only_partial_and_inconsistent_coordinates() -> None:
    image = np.ones((6, 7), dtype=float)
    original = np.zeros_like(image)
    qx = np.zeros_like(image)
    qy = np.ones_like(image)
    service = ButterflyAnalysisService()
    service.set_observed(original)

    with pytest.raises(ValueError, match="cannot use q alone"):
        service.set_observed(image, qmap={"q": np.ones_like(image)})

    with pytest.raises(ValueError, match="provided together"):
        service.set_observed(image, qx=qx)

    with pytest.raises(ValueError, match="coordinates are inconsistent"):
        service.set_observed(
            image,
            qmap={"qx": qx, "qy": qy, "q": np.full_like(image, 2.0)},
        )
    np.testing.assert_array_equal(service.observed, original)


def test_service_converts_direct_payload_angstrom_qmap_to_nm_inverse() -> None:
    image = np.ones((3, 4), dtype=float)
    qx = np.full(image.shape, 0.1)
    qy = np.zeros(image.shape)
    service = ButterflyAnalysisService()

    _, qmap, _, _, _, _ = service._state(
        {
            "observed": image,
            "qmap": {
                "qx": qx,
                "qy": qy,
                "q": np.hypot(qx, qy),
                "q_unit": "Å^-1",
            },
        }
    )

    assert qmap["q_unit"] == "nm^-1"
    assert qmap["source_q_unit"] == "Å^-1"
    assert qmap["q_conversion_factor_to_nm_inv"] == 10.0
    assert np.allclose(qmap["q"], 1.0)


def test_service_forwards_weight_arrays_and_paths_without_default_sampling(
    monkeypatch, tmp_path: Path
) -> None:
    image = np.ones((8, 9), dtype=float)
    service = ButterflyAnalysisService()
    state = service.set_observed(image)
    calls: list[dict[str, object]] = []

    def fake_fit(frame, qmap, **kwargs):
        del frame, qmap
        calls.append(kwargs)
        return _FakeFit(image.shape)

    monkeypatch.setattr(service_module, "fit_intensity_model", fake_fit)
    sigma = np.full(image.shape, 2.0, dtype=float)
    service.optimize(
        parameters=service.parameters,
        payload={**state, "sigma": sigma},
    )
    assert calls[-1]["max_pixels"] is None
    np.testing.assert_array_equal(calls[-1]["sigma"], sigma)
    assert "weights" not in calls[-1]

    weights = np.full(image.shape, 3.0, dtype=float)
    weights_path = tmp_path / "weights.npy"
    np.save(weights_path, weights)
    service.optimize(
        parameters=service.parameters,
        payload={**state, "weights": str(weights_path), "speed_cap": 7},
    )
    assert calls[-1]["max_pixels"] == 7
    np.testing.assert_array_equal(calls[-1]["weights"], weights)
    assert "sigma" not in calls[-1]


def test_service_measurement_payload_controls_the_observable_chain(monkeypatch) -> None:
    image = np.ones((6, 8), dtype=float)
    yy, xx = np.indices(image.shape, dtype=float)
    qmap = {
        "qx": (xx - 3.5) / 4.0,
        "qy": (yy - 2.5) / 4.0,
        "q_unit": "1/nm",
    }
    captured: dict[str, object] = {}

    def fake_measure(frame, supplied_qmap, q_window, **kwargs):
        captured.update(kwargs)
        captured["q_window"] = q_window
        assert frame.shape == image.shape
        assert supplied_qmap["q_unit"] == "nm^-1"
        np.testing.assert_allclose(supplied_qmap["qx"], qmap["qx"])
        return SimpleNamespace(flags=(), ridge=None, ellipse=None)

    monkeypatch.setattr(service_module, "measure_observables", fake_measure)
    service = ButterflyAnalysisService()
    state = service.set_observed(image, qmap=qmap)
    settings = {
        "q_min": 0.25,
        "q_max": 0.75,
        "draw_axis_deg": 123.0,
        "ridge_method": "surface_curvature",
        "n_angular_bins": 24,
        "n_ridge_angles": 18,
        "n_radial_bins": 37,
        "curvature_sigma": 1.5,
        "curvature_percentile": 40.0,
        "normal_step": 0.8,
    }
    result = service.preview(
        parameters=service.parameters,
        payload={**state, "analysis": settings},
    )

    assert result["analysis"]["q_min"] == 0.25
    assert captured["q_window"] == (0.25, 0.75)
    assert captured["draw_axis_deg"] == 123.0
    assert captured["ridge_method"] == "surface_curvature"
    assert captured["n_angular_bins"] == 24
    assert captured["n_ridge_angles"] == 18
    assert captured["n_radial_bins"] == 37
    assert captured["curvature_sigma"] == 1.5
    assert captured["curvature_percentile"] == 40.0
    assert captured["curvature_normal_step"] == 0.8


def test_draw_axis_controls_preview_and_fit_reference_axis(monkeypatch) -> None:
    image = np.ones((5, 6), dtype=float)
    service = ButterflyAnalysisService()
    state = service.set_observed(image)
    captured: dict[str, object] = {}

    def fake_model(qx, qy, parameters, **kwargs):
        del parameters
        captured["preview_reference_axis_deg"] = kwargs["reference_axis_deg"]
        return np.zeros_like(qx, dtype=float) + np.asarray(qy) * 0.0

    def fake_fit(frame, qmap, **kwargs):
        del frame, qmap
        captured["fit_reference_axis_deg"] = kwargs["reference_axis_deg"]
        return _FakeFit(image.shape)

    monkeypatch.setattr(service_module, "double_ellipse_intensity", fake_model)
    monkeypatch.setattr(service_module, "fit_intensity_model", fake_fit)
    settings = {"draw_axis_deg": 123.0, "n_angular_bins": 24, "n_ridge_angles": 12, "n_radial_bins": 24}
    service.preview(parameters=service.parameters, payload={**state, "analysis": settings})
    service.optimize(parameters=service.parameters, payload={**state, "analysis": settings})
    assert captured["preview_reference_axis_deg"] == 33.0
    assert captured["fit_reference_axis_deg"] == 33.0


def test_service_forwards_all_fit_controls_and_reports_shared_domain(monkeypatch) -> None:
    image = np.ones((5, 6), dtype=float)
    service = ButterflyAnalysisService()
    state = service.set_observed(image)
    captured: dict[str, object] = {}

    def fake_fit(frame, qmap, **kwargs):
        del frame, qmap
        captured.update(kwargs)
        return _FakeFit(image.shape)

    monkeypatch.setattr(service_module, "fit_intensity_model", fake_fit)
    result = service.optimize(
        parameters=service.parameters,
        payload={
            **state,
            "analysis": {
                "robust_loss": "huber",
                "f_scale": 2.5,
                "max_nfev": 123,
                "scales": [0.5, 1.0],
                "seed": 17,
            },
        },
    )

    assert captured["robust_loss"] == "huber"
    assert captured["f_scale"] == 2.5
    assert captured["max_nfev"] == 123
    assert captured["scales"] == (0.5, 1.0)
    assert captured["seed"] == 17
    assert result["analysis_domain"]["fit_pixel_count"] == image.size
    assert result["metrics"]["domain_counts"]["fit_pixel_count"] == image.size
    np.testing.assert_array_equal(result["fit_valid_mask"], result["valid_mask"])


def test_service_analysis_auto_window_and_zero_max_pixels_are_explicit(monkeypatch) -> None:
    image = np.ones((5, 5), dtype=float)
    yy, xx = np.indices(image.shape, dtype=float)
    qx, qy = xx - 2.0, yy - 2.0
    qmap = {"qx": qx, "qy": qy, "q_unit": "pixel-q"}
    service = ButterflyAnalysisService()
    state = service.set_observed(image, qmap=qmap)
    captured: dict[str, object] = {}

    def fake_measure(frame, supplied_qmap, q_window, **kwargs):
        del frame, supplied_qmap
        captured["q_window"] = q_window
        return SimpleNamespace(flags=(), ridge=None, ellipse=None)

    def fake_fit(frame, supplied_qmap, **kwargs):
        del frame, supplied_qmap
        captured["max_pixels"] = kwargs["max_pixels"]
        captured["q_window"] = kwargs["q_window"]
        captured["auto_scale_initial"] = kwargs["auto_scale_initial"]
        return _FakeFit(image.shape)

    monkeypatch.setattr(service_module, "measure_observables", fake_measure)
    monkeypatch.setattr(service_module, "fit_intensity_model", fake_fit)
    result = service.optimize(
        parameters=service.parameters,
        payload={
            **state,
            "analysis": {"q_min": "Auto", "q_max": "Auto", "max_pixels": 0},
        },
    )

    assert captured["q_window"] == (0.0, np.sqrt(8.0))
    assert captured["max_pixels"] is None
    assert captured["auto_scale_initial"] is False
    assert result["analysis"]["q_min"] is None
    assert result["analysis"]["max_pixels"] == 0


def test_service_rejects_an_invalid_q_window_before_measurement(monkeypatch) -> None:
    image = np.ones((5, 5), dtype=float)
    service = ButterflyAnalysisService()
    state = service.set_observed(image)
    called = False

    def should_not_measure(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("invalid q window reached the observable chain")

    monkeypatch.setattr(service_module, "measure_observables", should_not_measure)
    result = service.preview(
        parameters=service.parameters,
        payload={**state, "analysis": {"q_min": 1.0, "q_max": 0.5}},
    )
    assert not called
    assert any(flag.startswith("observables_failed:ValueError") for flag in result["flags"])
    assert any(flag.startswith("analysis_validation_failed:q window") for flag in result["flags"])


def test_service_rejects_wrong_shape_masks_instead_of_fitting(monkeypatch) -> None:
    image = np.ones((5, 6), dtype=float)
    service = ButterflyAnalysisService()
    state = service.set_observed(image)

    def should_not_fit(*args, **kwargs):
        raise AssertionError("wrong-shape mask reached the optimizer")

    monkeypatch.setattr(service_module, "fit_intensity_model", should_not_fit)
    with pytest.raises(ValueError, match="external_mask shape"):
        service.optimize(
            parameters=service.parameters,
            payload={**state, "external_mask": np.zeros((2, 3), dtype=bool)},
        )


def test_service_rejects_invalid_roi_instead_of_ignoring_it(monkeypatch) -> None:
    image = np.ones((5, 6), dtype=float)
    service = ButterflyAnalysisService()
    state = service.set_observed(image)

    def should_not_fit(*args, **kwargs):
        raise AssertionError("invalid ROI reached the optimizer")

    monkeypatch.setattr(service_module, "fit_intensity_model", should_not_fit)
    with pytest.raises(ValueError, match="invalid ROI exclusion specification"):
        service.optimize(
            parameters=service.parameters,
            payload={**state, "rois": [{"type": "rectangle", "x0": 0}]},
        )


def test_service_reports_detector_external_and_roi_counts_separately() -> None:
    image = np.ones((4, 4), dtype=float)
    service = ButterflyAnalysisService()
    state = service.set_observed(image)
    detector_valid = np.ones(image.shape, dtype=bool)
    detector_valid[0, 0] = False
    external_mask = np.zeros(image.shape, dtype=bool)
    external_mask[0, 1] = True

    result = service.preview(
        parameters=service.parameters,
        payload={
            **state,
            "valid_mask": detector_valid,
            "external_mask": external_mask,
            "rois": [
                {
                    "type": "rectangle",
                    "x0": 2,
                    "x1": 2,
                    "y0": 0,
                    "y1": 0,
                }
            ],
        },
    )

    counts = result["analysis_domain"]
    assert counts["detector_valid_count"] == 15
    assert counts["external_mask_excluded_count"] == 1
    assert counts["roi_excluded_count"] == 1
    assert counts["fit_pixel_count"] == 13


def test_service_optimize_can_defer_parameter_commit(monkeypatch) -> None:
    image = np.ones((5, 6), dtype=float)
    service = ButterflyAnalysisService()
    state = service.set_observed(image)
    before = service.parameters["amplitude_plus"]["value"]

    class Fitted(_FakeFit):
        parameters = {"amplitude_plus": 321.0}

    monkeypatch.setattr(
        service_module,
        "fit_intensity_model",
        lambda frame, qmap, **kwargs: Fitted(image.shape),
    )

    result = service.optimize(
        parameters=service.parameters,
        payload={**state, "commit_parameters": False},
    )
    assert result["parameters"]["amplitude_plus"]["value"] == 321.0
    assert service.parameters["amplitude_plus"]["value"] == before

    service.optimize(
        parameters=service.parameters,
        payload={**state, "commit_parameters": True},
    )
    assert service.parameters["amplitude_plus"]["value"] == 321.0


def test_service_analyze_frame_forwards_per_frame_selectors(monkeypatch, tmp_path: Path) -> None:
    service = ButterflyAnalysisService()
    calls: list[dict[str, object]] = []

    def fake_load(path, **kwargs):
        calls.append({"path": Path(path), **kwargs})
        return service.set_observed(np.ones((4, 5), dtype=float))

    monkeypatch.setattr(service, "load_image", fake_load)
    monkeypatch.setattr(
        service,
        "optimize",
        lambda **kwargs: {"parameters": {}, "metrics": {"success": True}},
    )
    frame = SimpleNamespace(
        path=tmp_path / "multi.npz",
        frame=2,
        dataset="entry/data",
        time=1.5,
        metadata={},
    )

    result = service.analyze_frame(frame)

    assert calls == [
        {"path": tmp_path / "multi.npz", "frame": 2, "dataset": "entry/data"}
    ]
    assert result["frame_selector"] == 2
    assert result["dataset"] == "entry/data"


def test_service_analyze_frame_forwards_config_window_and_fit_controls(
    monkeypatch, tmp_path: Path
) -> None:
    image = np.ones((5, 6), dtype=float)
    yy, xx = np.indices(image.shape, dtype=float)
    qmap = {"qx": xx / 4.0, "qy": yy / 4.0, "q_unit": "1/nm"}
    service = ButterflyAnalysisService()
    state = service.set_observed(image, qmap=qmap)
    captured: dict[str, object] = {}

    monkeypatch.setattr(service, "load_image", lambda *args, **kwargs: state)

    def fake_fit(frame, supplied_qmap, **kwargs):
        del frame, supplied_qmap
        captured.update(kwargs)
        return _FakeFit(image.shape)

    monkeypatch.setattr(service_module, "fit_intensity_model", fake_fit)
    result = service.analyze_frame(
        tmp_path / "frame.npy",
        config={
            "analysis": {
                "q_window": {"q_min": 0.5, "q_max": 1.0},
                "robust_loss": "huber",
                "f_scale": 2.5,
                "max_nfev": 17,
            }
        },
    )

    assert captured["q_window"] == (0.5, 1.0)
    assert captured["robust_loss"] == "huber"
    assert captured["f_scale"] == 2.5
    assert captured["max_nfev"] == 17
    assert result["analysis_domain"]["q_window"] == [0.5, 1.0]
    assert result["analysis_domain"]["q_window_pixel_count"] < image.size


def test_service_analyze_frame_accepts_q_range_and_existing_q_min_q_max(
    monkeypatch, tmp_path: Path
) -> None:
    image = np.ones((4, 5), dtype=float)
    yy, xx = np.indices(image.shape, dtype=float)
    qmap = {"qx": xx / 4.0, "qy": yy / 4.0, "q_unit": "1/nm"}
    service = ButterflyAnalysisService()
    state = service.set_observed(image, qmap=qmap)
    monkeypatch.setattr(service, "load_image", lambda *args, **kwargs: state)
    monkeypatch.setattr(
        service_module,
        "fit_intensity_model",
        lambda frame, supplied_qmap, **kwargs: _FakeFit(image.shape),
    )

    result = service.analyze_frame(
        tmp_path / "frame.npy",
        config={"q_range": [0.25, 0.75], "q_min": 0.5},
    )

    assert result["analysis_domain"]["q_window"] == [0.5, 0.75]


def test_stale_service_batch_does_not_overwrite_newer_parameter_state(
    monkeypatch, tmp_path: Path
) -> None:
    service = ButterflyAnalysisService()
    started = threading.Event()
    release = threading.Event()
    errors: list[Exception] = []

    def delayed_run(*args, **kwargs):
        del args, kwargs
        started.set()
        assert release.wait(5.0)
        return SimpleNamespace(
            frame_results=[], mode="independent", checkpoint=None
        )

    def invoke_batch() -> None:
        try:
            service.batch(payload={"frames": [tmp_path / "frame.npy"]})
        except Exception as exc:  # pragma: no cover - diagnostic thread handoff
            errors.append(exc)

    monkeypatch.setattr(service_module, "run_batch", delayed_run)
    worker = threading.Thread(target=invoke_batch)
    worker.start()
    assert started.wait(5.0)
    service.set_parameters(
        {**service.parameters, "amplitude_plus": {"value": 987.0}}
    )
    release.set()
    worker.join(5.0)

    assert not worker.is_alive()
    assert not errors
    assert service.parameters["amplitude_plus"]["value"] == 987.0


def test_service_batch_forwards_manifest_selectors_without_worker_commit(
    monkeypatch, tmp_path: Path
) -> None:
    service = ButterflyAnalysisService()
    loads: list[tuple[int | None, str | None]] = []
    commits: list[object] = []

    def fake_load(path, *, frame=None, dataset=None, **kwargs):
        del path, kwargs
        loads.append((frame, dataset))
        return service.set_observed(np.ones((4, 5), dtype=float))

    def fake_optimize(*, payload, **kwargs):
        del kwargs
        commits.append(payload.get("commit_parameters"))
        return {
            "parameters": {"amplitude_plus": {"value": 1.0}},
            "metrics": {"success": True, "rmse": 0.0},
        }

    monkeypatch.setattr(service, "load_image", fake_load)
    monkeypatch.setattr(service, "optimize", fake_optimize)
    source = tmp_path / "multi.npz"
    result = service.batch(
        payload={
            "frames": [
                FrameRef(source, frame=0, dataset="series", frame_id="f0"),
                FrameRef(source, frame=1, dataset="series", frame_id="f1"),
            ],
            "mode": "independent",
        }
    )

    assert loads == [(0, "series"), (1, "series")]
    assert commits == [False, False]
    assert [record["frame_selector"] for record in result["records"]] == [0, 1]
    assert [record["dataset"] for record in result["records"]] == ["series", "series"]


def test_service_batch_isolates_loaded_masks_across_concurrent_frames(
    monkeypatch, tmp_path: Path
) -> None:
    service = ButterflyAnalysisService()
    source_a = tmp_path / "a.npy"
    source_b = tmp_path / "b.npy"
    source_a.touch()
    source_b.touch()
    barrier = threading.Barrier(2)
    captured: list[tuple[str, np.ndarray | None]] = []
    payloads: dict[str, dict[str, object]] = {}
    errors: list[Exception] = []
    a_optimize_entered = threading.Event()
    original_optimize = service.optimize

    def fake_read(path, **kwargs):
        del kwargs
        name = Path(path).stem
        data = np.full((4, 5), 1.0 if name == "a" else 2.0)
        valid = None
        if name == "b":
            valid = np.ones(data.shape, dtype=bool)
            valid[0, 0] = False
        return LoadedImage(data, source=Path(path), valid_mask=valid)

    def delayed_optimize(*, payload, **kwargs):
        name = "a" if float(payload["observed"][0, 0]) == 1.0 else "b"
        payloads[name] = dict(payload)
        if name == "a":
            a_optimize_entered.set()
        barrier.wait(5.0)
        return original_optimize(payload=payload, **kwargs)

    def fake_fit(frame, qmap, **kwargs):
        del qmap
        name = "a" if float(frame.data[0, 0]) == 1.0 else "b"
        mask = kwargs["mask"]
        captured.append((name, None if mask is None else np.array(mask, copy=True)))
        return _FakeFit(frame.data.shape)

    monkeypatch.setattr(service_module, "read_image", fake_read)
    monkeypatch.setattr(service, "optimize", delayed_optimize)
    monkeypatch.setattr(service_module, "fit_intensity_model", fake_fit)

    def run_batch(path: Path) -> None:
        try:
            service.batch(payload={"frames": [path], "mode": "independent"})
        except Exception as exc:  # pragma: no cover - diagnostic thread handoff
            errors.append(exc)

    worker_a = threading.Thread(target=run_batch, args=(source_a,))
    worker_b = threading.Thread(target=run_batch, args=(source_b,))
    worker_a.start()
    assert a_optimize_entered.wait(5.0)
    worker_b.start()
    worker_a.join(5.0)
    worker_b.join(5.0)

    assert not worker_a.is_alive()
    assert not worker_b.is_alive()
    assert not errors
    assert {name for name, _ in captured} == {"a", "b"}
    masks = {name: mask for name, mask in captured}
    assert masks["a"] is None
    assert masks["b"][0, 0]
    assert float(payloads["a"]["observed"][0, 0]) == 1.0
    assert payloads["a"]["valid_mask"] is None
    assert float(payloads["b"]["observed"][0, 0]) == 2.0
    np.testing.assert_array_equal(
        payloads["b"]["valid_mask"],
        np.array(
            [
                [False, True, True, True, True],
                [True, True, True, True, True],
                [True, True, True, True, True],
                [True, True, True, True, True],
            ],
            dtype=bool,
        ),
    )
