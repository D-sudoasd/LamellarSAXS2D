from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from butterfly_saxs.cli import build_parser, main
from butterfly_saxs.intensity import default_intensity_parameters, double_ellipse_intensity
from butterfly_saxs.pipeline import (
    PipelineError,
    _loaded_frame,
    _coerce_qmap,
    _public_angles,
    analyze_frame,
    export_result,
    fit_full2d,
    inspect_frame,
    synthetic_butterfly,
)
from butterfly_saxs.project import ProjectConfig, load_project, save_project


def test_cli_help_has_public_vertical_slice() -> None:
    parser = build_parser()
    assert parser.description.startswith("LamellarSAXS2D")
    assert {"inspect", "analyze", "batch", "synthetic", "gui"}.issubset(
        parser._subparsers._group_actions[0].choices
    )
    with pytest.raises(SystemExit) as error:
        main(["--help"])
    assert error.value.code == 0


def test_cli_synthetic_writes_array(tmp_path: Path) -> None:
    destination = tmp_path / "synthetic.npz"
    assert main(["synthetic", "--shape", "32x40", "--output", str(destination)]) == 0
    with np.load(destination) as bundle:
        assert bundle["data"].shape == (32, 40)
        assert {"qx", "qy", "q"}.issubset(bundle.files)
        assert str(bundle["q_unit"].item()) == "pixel-q"
    report = inspect_frame(destination)
    assert report["q_unit"] == "pixel-q"
    assert report["flags"]["uncalibrated_pixel_q"] is True


def test_single_frame_pipeline_accepts_qmap_fixture(tmp_path: Path) -> None:
    image, qmap = synthetic_butterfly((48, 48), return_qmap=True, seed=3)
    result = analyze_frame(image, qmap=qmap)
    assert result.image.shape == (48, 48)
    assert "phi_app_deg" in result.observables
    assert "alpha_candidate_deg" in result.observables
    assert {"L_N", "L_z"}.issubset(result.observables["ellipse"])
    assert result.ellipse_fit["ellipse_axis_tilt_deg"] == pytest.approx(
        result.ellipse_fit["theta_deg"]
    )
    assert "eccentricity" in result.ellipse_fit
    assert "spacing_unavailable_unknown_q_unit" in result.ellipse_fit["flags"]
    assert "flags" in result.to_mapping()
    assert "parameters" in result.to_mapping()
    assert result.to_mapping()["parameters"]["a"] == pytest.approx(result.parameters["a"])

    output = tmp_path / "analysis"
    paths = export_result(result, output)
    assert {path.suffix for path in paths} == {".json", ".npz"}
    with paths[0].open(encoding="utf-8") as handle:
        summary = json.load(handle)
    assert summary["flags"]["nonunique_inverse_problem"] is True
    assert "NaN" not in paths[0].read_text(encoding="utf-8")


def test_analyze_reuses_measured_ridge_instead_of_extracting_twice(monkeypatch) -> None:
    from butterfly_saxs import pipeline

    image, qmap = synthetic_butterfly((40, 40), return_qmap=True, seed=23)

    def duplicate_extraction(*_args, **_kwargs):
        raise AssertionError("ridge extraction was repeated")

    monkeypatch.setattr(pipeline, "extract_ridges", duplicate_extraction)
    result = pipeline.analyze_frame(image, qmap=qmap)

    assert len(result.ridges) > 0
    assert result.ellipse_fit["n_points"] >= 5


def test_analyze_reuses_measured_ellipse_instead_of_fitting_twice(monkeypatch) -> None:
    """The observable bundle is the single authoritative ellipse fit."""

    from butterfly_saxs import observables
    from butterfly_saxs import pipeline

    image, qmap = synthetic_butterfly((40, 40), return_qmap=True, seed=24)
    original = observables.fit_symmetric_double_ellipse
    calls = 0

    def counted_fit(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(observables, "fit_symmetric_double_ellipse", counted_fit)
    result = pipeline.analyze_frame(image, qmap=qmap)

    assert calls == 1
    assert result.ellipse_fit["parameters"]["a"] == pytest.approx(
        result.observables["ellipse"]["a"]
    )
    assert result.ellipse_fit["theta_deg"] == pytest.approx(
        result.observables["ellipse"]["theta_deg"]
    )


def test_pipeline_does_not_fallback_after_ambiguous_npz_selection(tmp_path: Path) -> None:
    """An ambiguous NPZ must fail closed instead of silently choosing a key."""

    source = tmp_path / "ambiguous.npz"
    np.savez(source, first=np.ones((12, 12)), second=np.full((12, 12), 2.0))

    with pytest.raises(PipelineError, match="读取输入图像失败|dataset|数据集"):
        inspect_frame(source)


def test_pipeline_respects_explicit_npz_dataset_and_embedded_qmap(tmp_path: Path) -> None:
    first = np.ones((12, 12), dtype=float)
    selected = np.full((12, 12), 7.0, dtype=float)
    y, x = np.indices(first.shape, dtype=float)
    qx = x - 5.5
    qy = y - 5.5
    source = tmp_path / "multi.npz"
    np.savez(
        source,
        first=first,
        selected=selected,
        qx=qx,
        qy=qy,
        q=np.hypot(qx, qy),
        q_unit=np.asarray("pixel-q"),
    )

    report = inspect_frame(source, dataset="selected")

    assert report["intensity_min"] == pytest.approx(7.0)
    assert report["metadata"]["dataset"] == "selected"
    assert report["q_unit"] == "pixel-q"


def test_pipeline_fails_closed_for_shape_mismatched_mask(tmp_path: Path) -> None:
    source = tmp_path / "image.npy"
    np.save(source, np.ones((12, 12), dtype=float))

    with pytest.raises(PipelineError, match="掩膜|mask|形状|读取输入图像失败"):
        inspect_frame(source, mask=np.ones((4, 4), dtype=bool))


def test_loaded_frame_does_not_swallow_mask_or_shape_errors() -> None:
    with pytest.raises(Exception, match="形状|shape"):
        _loaded_frame(np.ones((4, 4)), valid_mask=np.ones((3, 3), dtype=bool))


def test_project_config_round_trip(tmp_path: Path) -> None:
    source = ProjectConfig(
        inputs=["data/a.cbf", "data/b.edf"],
        poni="geometry/sample.poni",
        output="results",
        q_unit="1/nm",
        full2d=True,
        analysis={"q_scale": 0.05, "ridge_bins": 90},
        metadata={"sample": "demo"},
    )
    path = tmp_path / "project.toml"
    save_project(source, path)
    loaded = load_project(path)
    assert loaded.input_paths == source.input_paths
    assert loaded.poni_path == source.poni_path
    assert loaded.output_dir == source.output_dir
    assert loaded.full2d is True
    assert loaded.analysis["q_scale"] == pytest.approx(0.05)
    assert loaded.metadata["sample"] == "demo"
    with pytest.raises(FileExistsError):
        save_project(source, path)


def test_project_resolves_analysis_paths_beside_toml(tmp_path: Path) -> None:
    source = ProjectConfig(
        inputs=["data/a.cbf"],
        poni="geometry/sample.poni",
        output="results",
        analysis={
            "mask": "geometry/mask.npy",
            "valid_mask": "geometry/valid.npy",
            "manifest": "data/manifest.csv",
            "checkpoint": "results/checkpoint.json",
            "sigma": "data/sigma.tif",
        },
    )
    resolved = source.resolve_paths(tmp_path)
    for key in ("mask", "valid_mask", "manifest", "checkpoint", "sigma"):
        assert Path(resolved.analysis[key]).is_absolute()
        assert Path(resolved.analysis[key]).is_relative_to(tmp_path)


def test_uncalibrated_fallback_is_labelled_pixel_q() -> None:
    report = inspect_frame(np.ones((20, 22), dtype=float))
    assert report["q_unit"] == "pixel-q"
    assert report["flags"]["uncalibrated_pixel_q"] is True


def test_full2d_exports_numeric_values_and_degree_adapters() -> None:
    q = np.linspace(-1.2, 1.2, 32)
    qx, qy = np.meshgrid(q, q)
    parameters = default_intensity_parameters(a=0.8, axis_ratio=0.65, theta_deg=13.0)
    for name, spec in list(parameters.spec_items()):
        if not spec.is_tied:
            parameters[name] = spec.copy(vary=False)
    image = double_ellipse_intensity(qx, qy, parameters)
    result = analyze_frame(
        image,
        qmap={"qx": qx, "qy": qy, "q": np.hypot(qx, qy), "q_unit": "1/nm"},
        full2d=True,
        initial_parameters=parameters,
    )
    values = result.full2d["parameters"]
    assert values["a"] == pytest.approx(0.8)
    assert values["theta_deg"] == pytest.approx(13.0)
    assert values["lobe_angle_deg"] == pytest.approx(np.degrees(values["lobe_angle"]))
    assert values["angular_width_deg"] == pytest.approx(np.degrees(values["angular_width"]))
    assert result.to_mapping()["parameters"]["angular_width_deg"] == pytest.approx(
        values["angular_width_deg"]
    )
    json.dumps(result.to_mapping(), allow_nan=False)


def test_public_angle_names_and_analysis_options_are_preserved() -> None:
    converted = _public_angles(
        {"theta": np.pi / 6, "angle": np.pi / 4, "angles": np.array([0.0, np.pi]), "chi": np.pi / 2}
    )
    assert converted["theta_deg"] == pytest.approx(30.0)
    assert converted["angle_deg"] == pytest.approx(45.0)
    assert np.allclose(converted["angles_deg"], [0.0, 180.0])
    assert converted["chi_deg"] == pytest.approx(90.0)

    image, qmap = synthetic_butterfly((40, 40), return_qmap=True, seed=5)
    valid = np.ones(image.shape, dtype=bool)
    valid[:2, :] = False
    qmap = {**qmap, "mask": np.zeros(image.shape, dtype=bool)}
    result = analyze_frame(
        {"data": image, "qmap": qmap, "valid_mask": valid},
        config={
            "analysis": {
                "q_window": [0.0, 30.0],
                "ridge_method": "radial_peak",
                "n_angles": 24,
            }
        },
    )
    assert int(result.valid_mask.sum()) == image.size - 2 * image.shape[1]
    assert int(result.qmap["mask"].sum()) == 0
    assert len(result.observables["ridge"]["angles_deg"]) == 24
    assert result.observables["ridge"]["points"][0]["method"] == "radial_peak"
    # Ellipse rotation remains theta; no mechanical alpha/phi alias is added.
    assert "theta_deg" in result.ellipse_fit["parameters"]
    assert "theta" not in result.ellipse_fit["parameters"]
    assert result.ellipse_fit.get("alpha_candidate_deg") is None


def test_pipeline_ellipse_uses_draw_axis_reference_and_exposes_spacing_aliases() -> None:
    from butterfly_saxs.pipeline import fit_symmetric_ellipses

    phi = np.linspace(0.0, 2.0 * np.pi, 120, endpoint=False)
    a, b, tilt_deg, draw_axis_deg = 0.82, 0.49, 17.0, 121.0
    reference = np.deg2rad(draw_axis_deg - 90.0)
    tilt = np.deg2rad(tilt_deg)

    def branch(sign: float) -> np.ndarray:
        c, s = np.cos(sign * tilt), np.sin(sign * tilt)
        x_local = c * a * np.cos(phi) - s * b * np.sin(phi)
        y_local = s * a * np.cos(phi) + c * b * np.sin(phi)
        cr, sr = np.cos(reference), np.sin(reference)
        return np.column_stack((cr * x_local - sr * y_local, sr * x_local + cr * y_local))

    points = np.vstack((branch(1.0), branch(-1.0)))
    fit = fit_symmetric_ellipses(
        [{"qx": float(x), "qy": float(y)} for x, y in points],
        qmap={"q_unit": "nm^-1"},
        config={"analysis": {"draw_axis_deg": draw_axis_deg}},
    )
    assert fit["parameters"]["reference_axis_deg"] == pytest.approx(draw_axis_deg - 90.0)
    assert fit["L_N"] == pytest.approx(2.0 * np.pi / b, rel=2e-3)
    expected_qz = a * b / np.sqrt(
        (b * np.cos(np.pi / 2 - tilt)) ** 2 + (a * np.sin(np.pi / 2 - tilt)) ** 2
    )
    assert fit["L_z"] == pytest.approx(2.0 * np.pi / expected_qz, rel=2e-3)


def test_coerce_qmap_skips_none_attributes_and_requires_exact_2d_arrays() -> None:
    shape = (4, 5)

    class OptionalQMap:
        qx = np.zeros(shape)
        qy = np.ones(shape)
        q = None
        mask = None
        valid_mask = None

    coerced = _coerce_qmap(OptionalQMap(), shape)
    assert set(coerced) >= {"qx", "qy", "object"}
    assert "q" not in coerced
    assert "mask" not in coerced
    assert "valid_mask" not in coerced

    with pytest.raises(PipelineError, match="形状"):
        _coerce_qmap({"qx": np.zeros((shape[0],)), "qy": np.zeros(shape)}, shape)


def test_pipeline_combines_source_config_and_embedded_qmap_masks() -> None:
    image, qmap = synthetic_butterfly((32, 32), return_qmap=True, seed=11)
    source_valid = np.ones(image.shape, dtype=bool)
    source_valid[0, 0] = False
    source_mask = np.zeros(image.shape, dtype=bool)
    source_mask[0, 1] = True
    config_valid = np.ones(image.shape, dtype=bool)
    config_valid[1, 0] = False
    config_mask = np.zeros(image.shape, dtype=bool)
    config_mask[1, 1] = True
    qmap_mask = np.zeros(image.shape, dtype=bool)
    qmap_mask[2, 2] = True
    qmap = {**qmap, "mask": qmap_mask}

    result = analyze_frame(
        {
            "data": image,
            "valid_mask": source_valid,
            "mask": source_mask,
            "qmap": qmap,
        },
        config={"analysis": {"valid_mask": config_valid, "mask": config_mask}},
    )
    expected = source_valid & config_valid & ~source_mask & ~config_mask & ~qmap_mask
    assert np.array_equal(result.valid_mask, expected)
    assert np.array_equal(result.qmap["valid_mask"], expected)
    assert bool(result.qmap["mask"][2, 2])


def test_fit_full2d_passes_configured_sigma_and_weights_and_ties_auto_b(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from butterfly_saxs import intensity
    from butterfly_saxs.parameters import ParameterSet
    from butterfly_saxs.project import ProjectConfig

    image, qmap = synthetic_butterfly((12, 12), return_qmap=True, seed=12)
    sigma = np.full(image.shape, 2.0)
    weights = np.full(image.shape, 3.0)
    weights_path = tmp_path / "weights.npy"
    np.save(weights_path, weights)
    calls: list[dict[str, object]] = []

    class FakeFit:
        parameters = {"a": 1.0, "axis_ratio": 0.5, "theta": 0.0}
        success = True
        message = "ok"
        flags = ()
        prediction = np.zeros_like(image)
        residual = np.zeros_like(image)

    def fake_fit(_frame, _qmap, initial, **kwargs):
        calls.append({"initial": initial, **kwargs})
        return FakeFit()

    monkeypatch.setattr(intensity, "fit_intensity_model", fake_fit)
    config = ProjectConfig(analysis={"sigma": sigma, "draw_axis_deg": 30.0})
    fit_full2d(
        image,
        qmap,
        {"parameters": {"a": 0.8, "b": 0.4, "axis_ratio": 0.5, "theta_deg": 12.0}},
        config=config,
    )
    assert len(calls) == 1
    # Both weighting modes are accepted independently; this call uses the
    # in-memory sigma and a second call below verifies the path-backed weight.
    assert np.array_equal(calls[0]["sigma"], sigma)
    assert calls[0].get("max_pixels") is None
    assert calls[0]["reference_axis_deg"] == pytest.approx(-60.0)
    assert calls[0]["auto_scale_initial"] is True
    initial = calls[0]["initial"]
    assert isinstance(initial, ParameterSet)
    assert initial["b"].is_tied
    assert initial["b"].expr == "a*axis_ratio"

    config = ProjectConfig(analysis={"weights": "weights.npy"}).resolve_paths(tmp_path)
    warm_start = {
        "amplitude_plus": 42.0,
        "background": 3.0,
        "theta": 0.2,
        "theta_deg": float(np.degrees(0.2)),
        "lobe_angle": 0.4,
        "lobe_angle_deg": float(np.degrees(0.4)),
        "angular_width": 0.1,
        "angular_width_deg": float(np.degrees(0.1)),
    }
    fit_full2d(
        image,
        qmap,
        {"parameters": {"a": 0.8, "axis_ratio": 0.5, "theta_deg": 12.0}},
        config=config,
        initial_parameters=warm_start,
    )
    assert np.array_equal(calls[1]["weights"], weights)
    assert calls[1]["auto_scale_initial"] is False
    assert calls[1]["initial"] == {
        "amplitude_plus": 42.0,
        "background": 3.0,
        "theta": 0.2,
        "lobe_angle": 0.4,
        "angular_width": 0.1,
    }
    assert "theta_deg" in warm_start


def test_cli_batch_uses_warm_start_checkpoint_and_exports(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    inputs = []
    for index in range(2):
        image = synthetic_butterfly(
            (32, 32),
            q0=14.0 + index,
            width=1.8,
            ellipticity=1.5,
            angle_deg=18.0 + index,
            seed=index,
        )
        path = tmp_path / f"frame_{index:04d}.npy"
        np.save(path, image)
        inputs.append(path)
    output = tmp_path / "batch-output"
    checkpoint = tmp_path / "checkpoint.json"
    assert main(
        [
            "batch",
            *(str(path) for path in inputs),
            "--output",
            str(output),
            "--mode",
            "warm_start",
            "--checkpoint",
            str(checkpoint),
            "--force",
        ]
    ) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["mode"] == "warm_start"
    assert report["n_frames"] == 2
    assert report["n_success"] == 2
    assert report["n_failed"] == 0
    assert checkpoint.exists()
    assert (output / "frame_summary.csv").exists()
    assert (output / "evolution.png").exists()
    # Every JSON boundary used by the CLI must reject NaN/Infinity.
    json.dumps(report, allow_nan=False)


def test_cli_batch_manifest_selects_distinct_npz_frames(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    images = np.stack(
        [
            synthetic_butterfly((48, 48), q0=12.0, angle_deg=16.0, seed=21),
            synthetic_butterfly((48, 48), q0=16.0, angle_deg=24.0, seed=22),
        ]
    )
    source = tmp_path / "多帧.npz"
    np.savez(source, series=images)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "frames": [
                    {"path": str(source), "dataset": "series", "frame": 0, "frame_id": "f0"},
                    {"path": str(source), "dataset": "series", "frame": 1, "frame_id": "f1"},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    output = tmp_path / "selected-output"

    assert main(
        [
            "batch",
            str(source),
            "--manifest",
            str(manifest),
            "--output",
            str(output),
            "--force",
        ]
    ) == 0
    report = json.loads(capsys.readouterr().out)

    assert report["n_frames"] == 2
    assert report["n_success"] == 2
    metadata = [record["result"]["metadata"] for record in report["frames"]]
    assert [item["frame"] for item in metadata] == [0, 1]
    assert [item["dataset"] for item in metadata] == ["series", "series"]
    assert metadata[0]["frame"] != metadata[1]["frame"]


def test_cli_batch_exports_partial_results_but_returns_nonzero_on_failed_frame(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    good = tmp_path / "good.npy"
    np.save(good, synthetic_butterfly((32, 32), seed=31))
    missing = tmp_path / "missing.npy"
    output = tmp_path / "partial-output"

    code = main(
        [
            "batch",
            str(good),
            str(missing),
            "--output",
            str(output),
            "--force",
        ]
    )
    report = json.loads(capsys.readouterr().out)

    assert code == 1
    assert report["n_success"] == 1
    assert report["n_failed"] == 1
    assert (output / "frame_summary.csv").exists()


def test_cli_analyze_returns_nonzero_for_an_explicit_fit_failure(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import butterfly_saxs.cli as cli_module

    class FailedResult:
        def to_mapping(self):
            return {
                "ellipse_fit": {"status": "ok", "success": True},
                "full2d": {"status": "failed", "success": False},
            }

    monkeypatch.setattr(cli_module, "analyze_frame", lambda *args, **kwargs: FailedResult())

    assert main(["analyze", "frame.npy", "--full2d"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["full2d"]["status"] == "failed"
