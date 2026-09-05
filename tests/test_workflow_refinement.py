from __future__ import annotations

from pathlib import Path
import json
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest

import butterfly_saxs.pipeline as pipeline
import butterfly_saxs.service as service_module
import butterfly_saxs.export as export_module
import butterfly_saxs.observables as observables_module
from butterfly_saxs.batch import (
    BatchRunResult,
    FrameFitResult,
    FrameRef,
    build_frame_refs,
    config_fingerprint,
    run_batch,
    select_frame_refs,
)
from butterfly_saxs.pipeline import PipelineResult, export_result
from butterfly_saxs.pipeline import batch_analyze, run_project_bounded
from butterfly_saxs.service import ButterflyAnalysisService
from butterfly_saxs.export import StreamingBatchExporter
from butterfly_saxs.cancellation import AnalysisCancelled


def test_flat_ellipse_settings_reach_measured_ridge_solver(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_measure(frame, qmap, q_window, **kwargs):
        del frame, qmap
        captured.update(kwargs)
        return SimpleNamespace(flags=(), ridge=None, ellipse=None)

    monkeypatch.setattr(service_module, "measure_observables", fake_measure)
    image = np.ones((12, 14), dtype=float)
    yy, xx = np.indices(image.shape, dtype=float)
    service = ButterflyAnalysisService()
    state = service.set_observed(
        image,
        qmap={
            "qx": (xx - 6.5) / 10.0,
            "qy": (yy - 5.5) / 10.0,
            "q_unit": "nm^-1",
        },
    )
    result = service.preview(
        payload={
            **state,
            "analysis": {
                "ridge_method": "azimuthal_peak",
                "ridge_snr_threshold": 3.0,
                "ridge_min_peak_fraction": 0.3,
                "ellipse": {
                    "preset": "flat_ellipse",
                    "axis_ratio_min": 0.005,
                    "axis_ratio_max": 0.35,
                    "a_min": 0.2,
                    "a_max": 4.0,
                    "fixed_center": True,
                    "angle_deg": 40.0,
                    "residual": "geometric",
                    "multistart": 5,
                },
            },
        }
    )
    ellipse_parameters = captured["ellipse_parameters"]
    assert isinstance(ellipse_parameters, dict)
    assert ellipse_parameters["axis_ratio"]["min"] == pytest.approx(0.005)
    assert ellipse_parameters["axis_ratio"]["max"] == pytest.approx(0.35)
    assert ellipse_parameters["a"]["min"] == pytest.approx(0.2)
    assert ellipse_parameters["b"]["expr"] == "a*axis_ratio"
    assert captured["ellipse_residual"] == "geometric"
    assert captured["ellipse_multistart"] == 5
    assert captured["ridge_method"] == "azimuthal_peak"
    assert result["analysis"]["ellipse"]["preset"] == "flat_ellipse"


def test_geometry_action_uses_ellipse_rmse_and_marks_model_unfitted(monkeypatch) -> None:
    service = ButterflyAnalysisService()

    def fake_preview(**kwargs):
        del kwargs
        return {
            "observed": np.ones((2, 3)),
            "model": np.full((2, 3), 0.5),
            "residual": np.full((2, 3), 0.5),
            "parameters": {"amplitude_plus": {"value": 12.0}},
            "metrics": {"rmse": 395.0, "ndata": 6, "success": True},
            "ellipse_fit": {
                "rmse": 0.0045,
                "n_points": 109,
                "q_unit": "nm^-1",
                "success": True,
                "parameters": {"a": 0.8, "axis_ratio": 0.08, "b": 0.064},
            },
            "flags": ["apparent_geometry_only"],
        }

    monkeypatch.setattr(service, "preview", fake_preview)
    result = service.measure_geometry()
    assert result["model"] is None
    assert result["residual"] is None
    assert result["model_status"] == "unfitted_preview"
    assert result["metrics"]["rmse"] == pytest.approx(0.0045)
    assert result["metrics"]["geometry_rmse"] == pytest.approx(0.0045)
    assert result["metrics"]["intensity_model_rmse"] is None
    assert result["metrics"]["ndata"] == 109
    assert result["geometry_parameters"]["axis_ratio"] == pytest.approx(0.08)
    assert result["intensity_parameters"]["amplitude_plus"]["value"] == pytest.approx(12.0)
    assert "intensity_model_unfitted" in result["flags"]


def test_service_result_exposes_multistart_cost_audit_fields(monkeypatch) -> None:
    service = ButterflyAnalysisService()
    image = np.ones((2, 3), dtype=float)
    state = service.set_observed(image)

    class Fit:
        model_image = np.ones_like(image)
        model = model_image
        residual = np.zeros_like(image)
        parameters = {"a": 1.0}
        success = True
        rmse = 0.1
        ndata = image.size
        sample_cost = 3.0
        full_cost = 4.0
        selection_objective = "full_valid_weighted_robust_cost"
        candidate_solutions = ({"start_index": 0, "cost": 4.0},)
        selected_start_index = 0
        multistart_count = 2
        flags = ()
        stderr = {}
        bound_flags = {}
        condition_number = 1.0

    monkeypatch.setattr(service_module, "fit_intensity_model", lambda *args, **kwargs: Fit())
    monkeypatch.setattr(
        service,
        "_measure",
        lambda *args, **kwargs: SimpleNamespace(flags=(), ridge=None, ellipse=None),
    )
    result = service.optimize(payload=state)
    assert result["fit_audit"]["sample_cost"] == pytest.approx(3.0)
    assert result["fit_audit"]["full_cost"] == pytest.approx(4.0)
    assert result["fit_audit"]["selection_objective"] == "full_valid_weighted_robust_cost"
    assert result["metrics"]["multistart_count"] == 2


def test_pipeline_flat_settings_are_converted_before_core(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_measure(frame, qmap, q_window, **kwargs):
        del frame, qmap, q_window
        captured.update(kwargs)
        return {"flags": (), "ridge": None, "ellipse": None}

    monkeypatch.setattr("butterfly_saxs.observables.measure_observables", fake_measure)
    image = np.ones((10, 11), dtype=float)
    yy, xx = np.indices(image.shape, dtype=float)
    pipeline.measure_observables(
        image,
        {"qx": xx / 10.0, "qy": yy / 10.0, "q_unit": "nm^-1"},
        config={
            "analysis": {
                "q_window": [0.01, 0.8],
                "ellipse": {
                    "preset": "flat_ellipse",
                    "axis_ratio_min": 0.005,
                    "axis_ratio_max": 0.35,
                    "fixed_center": True,
                },
            }
        },
    )
    parameters = captured["ellipse_parameters"]
    assert isinstance(parameters, dict)
    assert "axis_ratio_min" not in parameters
    assert parameters["axis_ratio"]["max"] == pytest.approx(0.35)


def test_service_batch_geometry_stage_exports_measured_parameters(monkeypatch, tmp_path: Path) -> None:
    service = ButterflyAnalysisService()
    source = tmp_path / "frame.npy"
    source.write_bytes(b"frame")
    calls: list[str] = []

    monkeypatch.setattr(
        service,
        "load_image",
        lambda path, **kwargs: service.set_observed(np.ones((2, 3))),
    )

    def fake_geometry(**kwargs):
        del kwargs
        calls.append("geometry")
        return {
            "parameters": {"amplitude_plus": {"value": 99.0}},
            "geometry_parameters": {"a": 0.8, "b": 0.064, "axis_ratio": 0.08},
            "ellipse_fit": {"rmse": 0.01, "success": True, "q_unit": "nm^-1"},
            "metrics": {"rmse": 0.01, "success": True},
            "flags": [],
        }

    def fail_full2d(**kwargs):
        del kwargs
        raise AssertionError("full2d path used for geometry stage")

    monkeypatch.setattr(service, "refine_geometry", fake_geometry)
    monkeypatch.setattr(service, "optimize", fail_full2d)
    result = service.batch(
        payload={
            "frames": [source],
            "stage": "geometry",
            "mode": "independent",
        }
    )
    assert calls == ["geometry"]
    assert result["stage"] == "geometry"
    assert result["records"][0]["stage"] == "geometry"
    assert result["records"][0]["parameters"]["a"] == pytest.approx(0.8)
    assert result["records"][0]["intensity_parameters"]["amplitude_plus"]["value"] == pytest.approx(99.0)


def test_stream_export_includes_lobe_scalar_measurements(tmp_path: Path) -> None:
    source = tmp_path / "frame.npy"
    source.write_bytes(b"frame")
    output = tmp_path / "out"
    writer = StreamingBatchExporter(output)
    run = run_batch(
        [source],
        lambda frame: {
            "parameters": {"a": 1.0},
            "metrics": {"success": True},
            "lobe_radial_peaks": [
                {
                    "angle_deg": 25.0,
                    "q_star": 0.28,
                    "radial_fwhm": 0.02,
                    "snr": 8.0,
                    "area": 0.4,
                    "coverage": 0.9,
                    "valid": True,
                    "q_unit": "nm^-1",
                }
            ],
        },
        result_sink=writer.write,
        retain_results=False,
    )
    outputs = writer.finalize(run)
    assert outputs["lobe_measurements"].exists()
    text = outputs["lobe_measurements"].read_text(encoding="utf-8")
    assert "q_star" in text and "0.28" in text and "0.02" in text


def test_adapter_does_not_retry_a_body_type_error() -> None:
    calls = 0

    def broken(image):
        nonlocal calls
        calls += 1
        raise TypeError("numerical body failure")

    with pytest.raises(TypeError, match="numerical body failure"):
        pipeline._call_adapter(broken, image=np.ones((2, 2)))
    assert calls == 1


def test_selector_range_and_series_are_explicit() -> None:
    refs = [
        FrameRef("f0.npy", frame_id="f0", source="A", order=0),
        FrameRef("f1.npy", frame_id="f1", source="B", order=1),
        FrameRef("f2.npy", frame_id="f2", source="A", order=2),
        FrameRef("f3.npy", frame_id="f3", source="A", order=3),
    ]
    selected = select_frame_refs(refs, series="A", start=1, stop=3, stride=2)
    assert [ref.id for ref in selected] == ["f2"]


@pytest.mark.parametrize("manifest", ([], {"frames": []}))
def test_empty_explicit_manifest_fails_closed(manifest) -> None:
    with pytest.raises(ValueError, match="manifest contains no frame entries"):
        build_frame_refs(["frame.npy"], manifest=manifest)


def test_batch_cancel_reports_progress_and_checkpoint_state(tmp_path: Path) -> None:
    event = __import__("threading").Event()
    progress: list[dict[str, object]] = []
    paths = [tmp_path / f"frame_{index}.npy" for index in range(3)]

    def analyzer(frame, initial=None):
        del initial
        np.save(frame.path, np.ones((2, 2), dtype=float))
        event.set()
        return {"parameters": {"index": 1.0}, "metrics": {"success": True}}

    checkpoint = tmp_path / "checkpoint.json"
    run = run_batch(
        paths,
        analyzer,
        checkpoint=checkpoint,
        cancel_event=event,
        progress=progress.append,
    )
    assert run.cancelled is True
    assert run.processed_count == 1
    assert run.total_count == 3
    assert checkpoint.exists() and checkpoint.stat().st_size > 0
    assert progress and progress[-1]["cancelled"] is True


def test_config_fingerprint_changes_when_mask_contents_change(tmp_path: Path) -> None:
    mask = tmp_path / "mask.npy"
    mask.write_bytes(b"first")
    first = config_fingerprint({"mask_path": str(mask)})
    mask.write_bytes(b"second")
    second = config_fingerprint({"mask_path": str(mask)})
    assert first != second


def test_resume_rejects_changed_configured_mask_contents(tmp_path: Path) -> None:
    mask = tmp_path / "mask.npy"
    mask.write_bytes(b"first")
    source = tmp_path / "frame.npy"
    source.write_bytes(b"frame")
    checkpoint = tmp_path / "checkpoint.json"

    def analyzer(frame):
        return {"parameters": {"a": 1.0}, "metrics": {"success": True}}

    config = {"mask_path": str(mask), "analysis": {"ridge_method": "radial_peak"}}
    run_batch([source], analyzer, config=config, checkpoint=checkpoint)
    mask.write_bytes(b"changed")
    with pytest.raises(ValueError, match="config hash mismatch"):
        run_batch([source], analyzer, config=config, checkpoint=checkpoint, resume=True)


def test_selector_aware_directory_exports_do_not_collide(tmp_path: Path) -> None:
    image = np.ones((3, 4), dtype=float)
    qx = np.zeros_like(image)
    qy = np.zeros_like(image)

    def result(frame: int) -> PipelineResult:
        return PipelineResult(
            image=image,
            qmap={"qx": qx, "qy": qy, "q": np.hypot(qx, qy), "q_unit": "pixel-q"},
            observables={},
            ridges=[],
            ellipse_fit={},
            metadata={"path": "multi.npy", "frame_selector": frame},
        )

    first = export_result(result(0), tmp_path / "exports")
    second = export_result(result(1), tmp_path / "exports")
    assert first[0] != second[0]
    assert first[0].exists() and second[0].exists()


def test_stream_resume_preserves_verified_npz_and_parameter_rows(tmp_path: Path) -> None:
    paths = [tmp_path / f"frame_{index}.dat" for index in range(2)]
    for index, path in enumerate(paths):
        path.write_bytes(f"frame-{index}".encode())
    output = tmp_path / "stream"
    checkpoint = tmp_path / "checkpoint.json"
    config = {"ridge_method": "radial_peak"}

    def analyzer(frame):
        index = int(frame.frame_id.rsplit("_", 1)[-1])
        return {
            "image": np.full((4, 5), index + 1.0),
            "parameters": {"a": {"value": index + 1.0, "unit": "q"}},
            "metrics": {"success": True},
        }

    writer = StreamingBatchExporter(output)
    run = run_batch(
        paths,
        analyzer,
        config=config,
        checkpoint=checkpoint,
        result_sink=writer.write,
        retain_results=False,
    )
    writer.finalize(run)
    before_params = (output / "parameters_long.csv").read_bytes()
    with np.load(output / "results.npz") as bundle:
        before_keys = set(bundle.files)
        before_image = bundle["frame_0000__image"].copy()

    resumed_writer = StreamingBatchExporter(output, force=True, resume=True)
    resumed = run_batch(
        paths,
        analyzer,
        config=config,
        checkpoint=checkpoint,
        resume=True,
        result_sink=resumed_writer.write,
        retain_results=False,
    )
    resumed_writer.finalize(resumed)
    with np.load(output / "results.npz") as bundle:
        assert set(bundle.files) == before_keys
        np.testing.assert_array_equal(bundle["frame_0000__image"], before_image)
        metadata = json.loads(str(bundle["__metadata__"].item()))
        assert metadata["complete"] is True
    assert (output / "parameters_long.csv").read_bytes() == before_params


def test_stream_cancel_then_resume_completes_previous_partial_bundle(tmp_path: Path) -> None:
    paths = [tmp_path / f"frame_{index}.dat" for index in range(2)]
    for path in paths:
        path.write_bytes(b"frame")
    output = tmp_path / "stream"
    checkpoint = tmp_path / "checkpoint.json"
    cancel = __import__("threading").Event()

    def analyzer(frame):
        index = int(frame.frame_id.rsplit("_", 1)[-1])
        if index == 0 and not cancel.is_set():
            cancel.set()
        return {
            "image": np.full((2, 3), index + 1.0),
            "parameters": {"a": {"value": index + 1.0}},
            "metrics": {"success": True},
        }

    writer = StreamingBatchExporter(output)
    first = run_batch(
        paths,
        analyzer,
        config={"series": "test"},
        checkpoint=checkpoint,
        cancel_event=cancel,
        result_sink=writer.write,
        retain_results=False,
    )
    writer.finalize(first)
    with np.load(output / "results.npz") as bundle:
        assert json.loads(str(bundle["__metadata__"].item()))["complete"] is False

    cancel.clear()
    resumed_writer = StreamingBatchExporter(output, force=True, resume=True)
    resumed = run_batch(
        paths,
        analyzer,
        config={"series": "test"},
        checkpoint=checkpoint,
        resume=True,
        result_sink=resumed_writer.write,
        retain_results=False,
    )
    resumed_writer.finalize(resumed)
    with np.load(output / "results.npz") as bundle:
        metadata = json.loads(str(bundle["__metadata__"].item()))
        assert metadata["complete"] is True
        assert "frame_0000__image" in bundle.files
        assert "frame_0001__image" in bundle.files


def test_direct_batch_mapping_resolves_base_dir_before_file_validation(tmp_path: Path) -> None:
    source = tmp_path / "frame.dat"
    weights = tmp_path / "weights.npy"
    source.write_bytes(b"frame")
    np.save(weights, np.ones((2, 2)))
    seen: dict[str, object] = {}

    def analyzer(frame, config=None):
        del frame
        seen["weights"] = config["weights"]
        return {"parameters": {"a": 1.0}, "metrics": {"success": True}}

    run = run_batch(
        [source],
        analyzer,
        config={"base_dir": str(tmp_path), "weights": "weights.npy"},
    )
    assert run.successful
    assert Path(str(seen["weights"])).resolve() == weights.resolve()


def test_noop_stream_resume_does_not_recompress_verified_npz(tmp_path: Path, monkeypatch) -> None:
    paths = [tmp_path / f"frame_{index}.dat" for index in range(2)]
    for path in paths:
        path.write_bytes(b"frame")
    output = tmp_path / "stream"
    checkpoint = tmp_path / "checkpoint.json"

    def analyzer(frame):
        index = int(frame.frame_id.rsplit("_", 1)[-1])
        return {
            "image": np.full((2, 2), index + 1.0),
            "parameters": {"a": index + 1.0},
            "metrics": {"success": True},
        }

    first_writer = StreamingBatchExporter(output)
    first = run_batch(
        paths,
        analyzer,
        checkpoint=checkpoint,
        result_sink=first_writer.write,
        retain_results=False,
    )
    first_writer.finalize(first)
    before = (output / "results.npz").read_bytes()

    def fail_copyfileobj(*args, **kwargs):
        raise AssertionError("no-op resume must not recompress old NPZ members")

    monkeypatch.setattr(export_module.shutil, "copyfileobj", fail_copyfileobj)
    resumed_writer = StreamingBatchExporter(output, force=True, resume=True)
    resumed = run_batch(
        paths,
        lambda *_args, **_kwargs: pytest.fail("verified frames should be restored"),
        checkpoint=checkpoint,
        resume=True,
        result_sink=resumed_writer.write,
        retain_results=False,
    )
    resumed_writer.finalize(resumed)
    assert (output / "results.npz").read_bytes() == before


def test_continuity_ridge_polls_cancellation_between_sectors(monkeypatch) -> None:
    event = threading.Event()
    image = np.ones((16, 16), dtype=float)
    yy, xx = np.indices(image.shape, dtype=float)
    qx, qy = (xx - 7.5) * 0.1, (yy - 7.5) * 0.1
    qmap = {"qx": qx, "qy": qy, "q": np.hypot(qx, qy), "q_unit": "nm^-1"}
    original = observables_module.measure_radial_profile
    calls = 0

    def wrapped(*args, **kwargs):
        nonlocal calls
        calls += 1
        profile = original(*args, **kwargs)
        if calls == 1:
            event.set()
        return profile

    monkeypatch.setattr(observables_module, "measure_radial_profile", wrapped)
    with pytest.raises(AnalysisCancelled):
        observables_module.measure_radial_ridges(
            image,
            qmap,
            (0.1, 1.0),
            n_angles=24,
            n_bins=32,
            cancel_event=event,
        )


def test_lobe_radial_companion_polls_cancellation(monkeypatch) -> None:
    event = threading.Event()
    image = np.ones((12, 12), dtype=float)
    yy, xx = np.indices(image.shape, dtype=float)
    qx, qy = (xx - 5.5) * 0.1, (yy - 5.5) * 0.1
    qmap = {"qx": qx, "qy": qy, "q": np.hypot(qx, qy), "q_unit": "nm^-1"}
    lobe = observables_module.LobeMetrics(
        angle=0.2,
        intensity=2.0,
        baseline=1.0,
        snr=4.0,
        fwhm=0.1,
        area=1.0,
        index=0,
        coverage=1.0,
        n_pixels=10,
    )
    original = observables_module.measure_radial_profile

    def wrapped(*args, **kwargs):
        profile = original(*args, **kwargs)
        event.set()
        return profile

    monkeypatch.setattr(observables_module, "measure_radial_profile", wrapped)
    with pytest.raises(AnalysisCancelled):
        observables_module._measure_lobe_radial_observables(
            image,
            qmap,
            (0.1, 1.0),
            [lobe],
            n_radial_bins=24,
            snr_threshold=1.0,
            min_coverage=0.0,
            cancel_event=event,
        )


def test_csv_dynamic_parameter_and_observable_names_are_formula_safe(tmp_path: Path) -> None:
    result = {
        "image": np.ones((2, 2)),
        "observables": {"=observable": "\t=SUM(A1:A2)"},
        "ellipse_fit": {"parameters": {"=1+1": -1.0}},
    }
    output = tmp_path / "single.csv"
    export_result(result, output)
    text = output.read_text(encoding="utf-8")
    assert "'=observable" in text
    assert "'\t=SUM" in text
    assert "ellipse.=1+1" in text

    stream = StreamingBatchExporter(tmp_path / "stream")
    stream.write(
        FrameFitResult(
            FrameRef("frame.dat"),
            result={
                "image": np.ones((2, 2)),
                "parameters": {"=1+1": -1.0},
                "metrics": {"success": True},
            },
        )
    )
    stream.finalize(BatchRunResult([], total_count=1, processed_count=1))
    assert "'=1+1" in (tmp_path / "stream" / "parameters_long.csv").read_text(
        encoding="utf-8"
    )


def test_legacy_batch_analyze_delegates_common_runner_and_preserves_quality_results(
    tmp_path: Path, monkeypatch
) -> None:
    paths = [tmp_path / "ok.dat", tmp_path / "quality.dat"]
    for path in paths:
        path.write_bytes(b"frame")

    def fake_analyze(source, **kwargs):
        del kwargs
        quality_failed = Path(source).stem == "quality"
        return PipelineResult(
            image=np.ones((2, 2)),
            qmap={"qx": np.zeros((2, 2)), "qy": np.zeros((2, 2)), "q": np.zeros((2, 2))},
            observables={},
            ridges=[],
            ellipse_fit={"status": "failed" if quality_failed else "ok", "success": not quality_failed},
            metadata={"path": str(source)},
        )

    monkeypatch.setattr(pipeline, "analyze_frame", fake_analyze)
    results = batch_analyze(paths)
    assert len(results) == 2
    with pytest.raises(Exception):
        monkeypatch.setattr(
            pipeline,
            "analyze_frame",
            lambda source, **kwargs: (_ for _ in ()).throw(OSError("read failed")),
        )
        batch_analyze([paths[0]])


def test_bounded_project_stream_keeps_arrays_in_sink_not_frame_results(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "frame.dat"
    source.write_bytes(b"frame")
    output = tmp_path / "project-output"

    def fake_analyze(source, **kwargs):
        del kwargs
        return PipelineResult(
            image=np.ones((3, 3)),
            qmap={"qx": np.zeros((3, 3)), "qy": np.zeros((3, 3)), "q": np.zeros((3, 3))},
            observables={},
            ridges=[],
            ellipse_fit={"status": "ok", "success": True, "parameters": {"a": 1.0}},
            metadata={"path": str(source)},
        )

    monkeypatch.setattr(pipeline, "analyze_frame", fake_analyze)
    run = run_project_bounded(
        pipeline.ProjectConfig(
            inputs=[source],
            output=output,
            analysis={"stream": True},
        )
    )
    assert run.frame_results
    assert isinstance(run.frame_results[0].result, dict)
    assert run.frame_results[0].result["image"]["array_omitted"] is True
    assert (output / "results.npz").exists()


def test_service_and_pipeline_ellipse_payloads_share_compatibility_aliases() -> None:
    fit = SimpleNamespace(
        success=True,
        a=1.0,
        b=0.2,
        axes_ratio=0.2,
        theta=0.1,
        theta_deg=np.degrees(0.1),
        center=(0.0, 0.0),
        parameter_values={"a": 1.0, "b": 0.2, "axis_ratio": 0.2},
        ellipses=(),
        flags=(),
        n_points=8,
        rmse=0.01,
        L_N=float("nan"),
        L_z=float("nan"),
        Ln_from_minor_axis_nm=float("nan"),
        Lz_from_draw_axis_nm=float("nan"),
        eccentricity=float("nan"),
        ellipticity=float("nan"),
        q_unit="nm^-1",
        branch_assignment_indices=[3, 7],
    )
    service_payload = service_module._public_ellipse(fit)
    pipeline_payload = pipeline._public_ellipse_fit(
        fit,
        n_points=8,
        qmap={"q_unit": "nm^-1"},
    )
    for payload in (service_payload, pipeline_payload):
        assert payload["parameters"] == payload["parameter_values"]
        assert payload["a"] == pytest.approx(1.0)
        assert payload["b"] == pytest.approx(0.2)
        assert payload["axis_ratio"] == pytest.approx(0.2)
        for name in (
            "L_N", "L_z", "Ln_from_minor_axis_nm", "Lz_from_draw_axis_nm",
            "eccentricity", "ellipticity", "q_unit",
        ):
            assert name in payload["parameters"]
        assert payload["branch_assignment_indices"] == [3, 7]


def test_public_analysis_settings_boundary_does_not_import_service(monkeypatch) -> None:
    import butterfly_saxs.settings as settings

    monkeypatch.setitem(sys.modules, "butterfly_saxs.service", None)
    normalized = settings.resolve_analysis_settings(
        {"n_angular_bins": 24, "normal_step": 0.5}
    )
    assert normalized["n_angular_bins"] == 24
    assert normalized["normal_step"] == pytest.approx(0.5)


def test_pipeline_branch_descriptor_keeps_lossless_values_and_indices() -> None:
    fit = SimpleNamespace(
        success=True,
        parameter_values={"a": 1.0, "b": 0.2, "axis_ratio": 0.2},
        branch_assignment=np.asarray([0, 1, 1], dtype=int),
        branch_assignment_indices=np.asarray([4, 8, 9], dtype=int),
        ellipses=(),
    )
    payload = pipeline._public_ellipse_fit(fit, n_points=3, qmap={"q_unit": "nm^-1"})
    assert payload["branch_assignment"]["shape"] == [3]
    np.testing.assert_array_equal(payload["branch_assignment_values"], [0, 1, 1])
    np.testing.assert_array_equal(payload["branch_assignment_indices"], [4, 8, 9])
