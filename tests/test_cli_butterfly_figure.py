"""The CLI exports the same measurement and arrays that were analysed."""

import json

import numpy as np
import pytest

from butterfly_saxs.cli import main


def test_figure_target_is_checked_before_fitting(tmp_path, monkeypatch, capsys):
    target = tmp_path / "figure"
    target.mkdir()
    sentinel = target / "keep.txt"
    sentinel.write_text("existing evidence", encoding="utf-8")

    def unexpected(*args, **kwargs):
        pytest.fail("existing figure target must be rejected before analysis")

    monkeypatch.setattr("butterfly_saxs.cli.analyze_frame", unexpected)
    assert main(["analyze", "missing.npy", "--figure-output", str(target), "--force"]) == 2
    assert sentinel.read_text(encoding="utf-8") == "existing evidence"
    assert "未覆盖" in capsys.readouterr().err


def test_figure_rejects_conflicting_analysis_method(tmp_path, capsys):
    assert main(["analyze", "missing.npy", "--figure-output", str(tmp_path / "figure"),
                 "--ridge-method", "radial_peak"]) == 2
    assert "butterfly_curvature" in capsys.readouterr().err


@pytest.mark.parametrize("method", ["butterfly-curvature", " BUTTERFLY_CURVATURE "])
def test_figure_accepts_the_shared_configured_method_aliases(tmp_path, monkeypatch, capsys, method):
    from butterfly_saxs import butterfly_figure
    from butterfly_saxs.pipeline import PipelineResult

    config = tmp_path / "project.toml"
    config.write_text(f'[analysis]\nridge_method = "{method}"\n', encoding="utf-8")
    yy, xx = np.indices((6, 7), dtype=float)
    result = PipelineResult(
        image=xx + yy + 2., qmap={"qx": xx, "qy": yy, "q_unit": "pixel-q"},
        observables={"butterfly": {"measurement_status": "traced", "points": []}},
        ridges=[], ellipse_fit={},
    )
    seen = []

    def analyze(*args, **kwargs):
        seen.append(kwargs["config"].analysis["ridge_method"])
        return result

    monkeypatch.setattr("butterfly_saxs.cli.analyze_frame", analyze)
    monkeypatch.setattr(butterfly_figure, "export_butterfly_figure", lambda *a, **kw: {})
    assert main(["analyze", "input.npy", "--config", str(config),
                 "--figure-output", str(tmp_path / "figure")]) in (0, 1)
    assert seen == ["butterfly_curvature"]
    assert "科研图导出需要" not in capsys.readouterr().err


@pytest.mark.parametrize("suffix", [".json", ".npz"])
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("force", [False, True])
def test_resolved_analysis_file_cannot_become_a_figure_directory(tmp_path, monkeypatch, capsys, suffix, nested, force):
    from butterfly_saxs import butterfly_figure
    from butterfly_saxs.pipeline import PipelineResult

    yy, xx = np.indices((6, 7), dtype=float)
    result = PipelineResult(
        image=xx + yy + 2., qmap={"qx": xx, "qy": yy, "q_unit": "pixel-q"},
        observables={"butterfly": {"measurement_status": "traced", "points": []}},
        ridges=[], ellipse_fit={}, metadata={"path": str(tmp_path / "same.edf")},
    )
    monkeypatch.setattr("butterfly_saxs.cli.analyze_frame", lambda *a, **kw: result)

    def no_write(*args, **kwargs):
        pytest.fail("neither exporter may run with conflicting resolved destinations")

    monkeypatch.setattr(butterfly_figure, "export_butterfly_figure", no_write)
    monkeypatch.setattr("butterfly_saxs.pipeline.export_result", no_write)
    output = tmp_path / "results"
    figure = output / f"same{suffix}"
    if nested:
        figure = figure / "figure"
    args = ["analyze", "same.edf", "--output", str(output), "--figure-output", str(figure)]
    assert main(args + (["--force"] if force else [])) == 2
    assert "独立目录" in capsys.readouterr().err
    assert not output.exists()


@pytest.mark.parametrize("conflict", ["generated_directory", "parent_file"])
@pytest.mark.parametrize("force", [False, True])
def test_existing_incompatible_analysis_target_type_blocks_figure_publish(tmp_path, monkeypatch, capsys, conflict, force):
    from butterfly_saxs import butterfly_figure
    from butterfly_saxs.pipeline import PipelineResult

    yy, xx = np.indices((6, 7), dtype=float)
    result = PipelineResult(
        image=xx + yy + 2., qmap={"qx": xx, "qy": yy, "q_unit": "pixel-q"},
        observables={"butterfly": {"measurement_status": "traced", "points": []}},
        ridges=[], ellipse_fit={}, metadata={"path": str(tmp_path / "same.edf")},
    )
    monkeypatch.setattr("butterfly_saxs.cli.analyze_frame", lambda *a, **kw: result)

    def no_write(*args, **kwargs):
        pytest.fail("a known target-type conflict must be rejected before any export")

    monkeypatch.setattr(butterfly_figure, "export_butterfly_figure", no_write)
    output = tmp_path / "results"
    if conflict == "generated_directory":
        (output / "same.json").mkdir(parents=True)
        sentinel = output / "same.json" / "keep.txt"
    else:
        sentinel = output
    sentinel.write_text("keep", encoding="utf-8")
    figure = tmp_path / "figure"
    args = ["analyze", "same.edf", "--output", str(output), "--figure-output", str(figure)]
    assert main(args + (["--force"] if force else [])) == 2
    assert "科研图未写入" in capsys.readouterr().err
    assert not figure.exists()
    assert sentinel.read_text(encoding="utf-8") == "keep"


@pytest.mark.parametrize("placement", ["same", "inside_figure", "inside_analysis_file"])
def test_conflicting_output_directories_fail_before_analysis(tmp_path, monkeypatch, capsys, placement):
    target = tmp_path / "figure"
    output = target / "analysis.json" if placement == "inside_figure" else target
    if placement == "inside_analysis_file":
        output = tmp_path / "analysis.json"
        target = output / "figure"

    def unexpected(*args, **kwargs):
        pytest.fail("overlapping destinations must fail before writing analysis output")

    monkeypatch.setattr("butterfly_saxs.cli.analyze_frame", unexpected)
    assert main(["analyze", "missing.npy", "--output", str(output),
                 "--figure-output", str(target)]) == 2
    assert not target.exists()
    assert "独立目录" in capsys.readouterr().err


def test_cli_passes_measured_arrays_mask_and_units_to_figure(tmp_path, monkeypatch, capsys):
    from butterfly_saxs import butterfly_figure

    yy, xx = np.indices((24, 28), dtype=float)
    data = 5.0 + np.exp(-((np.hypot(xx - 13.5, yy - 11.5) - 7.0) / 1.2) ** 2)
    qx, qy = (xx - 13.5) * 0.01, (yy - 11.5) * 0.01
    source = tmp_path / "input.npz"
    np.savez(source, data=data, qx=qx, qy=qy, q_unit="nm^-1")
    excluded = np.zeros(data.shape, dtype=bool)
    excluded[0:3] = True
    mask = tmp_path / "mask.npy"
    np.save(mask, excluded)
    captured = {}

    def capture(target, **kwargs):
        captured.update(kwargs)
        captured["target"] = target
        return {"svg": tmp_path / "figure" / "figure.svg"}

    monkeypatch.setattr(butterfly_figure, "export_butterfly_figure", capture)
    persisted_report = tmp_path / "analysis.json"
    status = main(["analyze", str(source), "--mask", str(mask),
                   "--butterfly-stage", "trace", "--figure-output", str(tmp_path / "figure"),
                   "--output", str(persisted_report),
                   "--figure-width", "89", "--figure-dpi", "300"])
    assert status in (0, 1)  # The scientific result retains its normal quality exit code.
    report = json.loads(capsys.readouterr().out)
    np.testing.assert_array_equal(captured["observed"], data)
    np.testing.assert_allclose(captured["qx"], qx)
    np.testing.assert_allclose(captured["qy"], qy)
    assert not np.any(captured["valid_mask"][excluded])
    assert captured["q_unit"] == "nm^-1"
    assert captured["width_mm"] == 89.0
    assert captured["dpi"] == 300
    assert captured["result"]["measurement_status"] == "traced"
    assert captured["result"]["points"] == report["butterfly"]["points"]
    assert str(tmp_path / "figure" / "figure.svg") in report["output_paths"]
    saved = json.loads(persisted_report.read_text(encoding="utf-8"))
    assert str(tmp_path / "figure" / "figure.svg") in saved["output_paths"]

    def must_not_publish(*args, **kwargs):
        pytest.fail("pre-existing analysis output must be checked before publishing a figure")

    monkeypatch.setattr(butterfly_figure, "export_butterfly_figure", must_not_publish)
    original_report = persisted_report.read_bytes()
    new_target = tmp_path / "new_figure"
    assert main(["analyze", str(source), "--butterfly-stage", "trace", "--output", str(persisted_report),
                 "--figure-output", str(new_target)]) == 2
    assert not new_target.exists()
    assert persisted_report.read_bytes() == original_report
    assert "科研图未写入" in capsys.readouterr().err


def test_cli_passes_only_explicit_full2d_prediction_for_image_comparison(tmp_path, monkeypatch, capsys):
    from butterfly_saxs import butterfly_figure
    from butterfly_saxs.pipeline import PipelineResult

    yy, xx = np.indices((6, 7), dtype=float)
    observed = xx + yy + 2.0
    model = observed * .9
    result = PipelineResult(
        image=observed, qmap={"qx": xx, "qy": yy, "q_unit": "pixel-q"},
        observables={"butterfly": {"measurement_status": "traced", "points": []}},
        ridges=[], ellipse_fit={}, full2d={"model_image": model, "status": "ok"},
    )
    captured = {}
    monkeypatch.setattr("butterfly_saxs.cli.analyze_frame", lambda *a, **kw: result)

    def capture(target, **kwargs):
        captured.update(kwargs)
        return {"svg": tmp_path / "figure.svg"}

    monkeypatch.setattr(butterfly_figure, "export_butterfly_figure", capture)
    assert main(["analyze", "explicit_input.npy", "--full2d", "--figure-output", str(tmp_path / "figure")]) in (0, 1)
    capsys.readouterr()
    np.testing.assert_array_equal(captured["model"], model)
    assert captured["context"]["pixel_model_source"] == "full2d fit output"
    assert captured["context"]["pixel_model_status"] == "ok"
