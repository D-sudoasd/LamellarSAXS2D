from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from butterfly_saxs import cli
from butterfly_saxs import batch as batch_module
from butterfly_saxs import export as export_module
from butterfly_saxs.service import ButterflyAnalysisService


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    package = tmp_path / "package"
    package.mkdir()
    np.save(package / "frame1.npy", np.ones((8, 8)))
    np.save(package / "frame2.npy", np.ones((8, 8)) * 2)
    poni = package / "geometry.poni"
    poni.write_text("test calibration", encoding="utf-8")
    mask = package / "mask.npy"
    np.save(mask, np.zeros((8, 8), dtype=bool))
    return package, poni, mask


def test_unattended_batch_preflights_selected_frames_and_resumes(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    package, poni, mask = _inputs(tmp_path)
    output = tmp_path / "results"
    preflight_calls = []
    analyzed = []
    recipes = []

    def preflight(self, source, **kwargs):
        preflight_calls.append((source, kwargs))
        return {"status": {"status_color": "yellow", "scientific_status": "WARN", "exit_code": 1}}

    def analyze(source, **kwargs):
        analyzed.append(Path(source).name)
        recipes.append(kwargs["config"].analysis)
        return {"success": True, "status": "ok", "parameters": {"a": 1.0}}

    monkeypatch.setattr(ButterflyAnalysisService, "preflight", preflight)
    monkeypatch.setattr(cli, "analyze_frame", analyze)
    command = [
        "batch", str(package / "frame*.npy"), "--poni", str(poni),
        "--mask", str(mask), "--unattended", str(package), "-o", str(output),
    ]

    assert cli.main(command) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["n_frames"] == 2
    assert report["n_failed"] == 0
    assert report["preflight"]["status_color"] == "yellow"
    assert report["agent"]["scientific_acceptance"] is False
    assert all("result" not in frame for frame in report["frames"])
    assert analyzed == ["frame1.npy", "frame2.npy"]
    assert all(recipe["ridge_method"] == "butterfly_curvature" for recipe in recipes)
    assert all(recipe["butterfly"]["trace_method"] == "annular_peak" for recipe in recipes)
    assert all(recipe["butterfly"]["stage"] == "evaluate" for recipe in recipes)
    assert (output / "checkpoint.json").exists()
    assert (output / "results.npz").exists()
    assert preflight_calls[0][1]["mask"] == str(mask)
    assert [Path(row["path"]).name for row in preflight_calls[0][1]["manifest"]] == analyzed

    assert cli.main([*command, "--resume"]) == 1
    resumed = json.loads(capsys.readouterr().out)
    assert resumed["n_frames"] == 2
    assert all(frame["resumed"] for frame in resumed["frames"])
    assert analyzed == ["frame1.npy", "frame2.npy"]


def test_unattended_batch_stops_on_red_preflight(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    package, poni, _ = _inputs(tmp_path)
    output = tmp_path / "results"

    def preflight(self, source, **kwargs):
        return {"status": {"status_color": "red", "scientific_status": "FAIL", "exit_code": 2}}

    def analyze(*args, **kwargs):
        raise AssertionError("red preflight must stop before fitting")

    monkeypatch.setattr(ButterflyAnalysisService, "preflight", preflight)
    monkeypatch.setattr(cli, "analyze_frame", analyze)
    assert cli.main([
        "batch", str(package / "frame*.npy"), "--poni", str(poni),
        "--unattended", str(package), "-o", str(output),
    ]) == 2
    report = json.loads(capsys.readouterr().out)
    assert report["blocked_stage"] == "preflight"
    assert report["preflight"]["status_color"] == "red"
    assert any("fitting did not start" in step for step in report["agent"]["next"])
    assert not (output / "results.npz").exists()


def test_unattended_series_selection_and_failed_frame_are_explicit(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    package, poni, _ = _inputs(tmp_path)
    manifest = package / "manifest.json"
    manifest.write_text(json.dumps([
        {"path": "frame1.npy", "source": "reference", "order": 0},
        {"path": "frame2.npy", "source": "hold", "order": 1},
    ]), encoding="utf-8")
    seen = []

    def preflight(self, source, **kwargs):
        seen.extend(kwargs["manifest"])
        return {"status": {"status_color": "green", "scientific_status": "PASS", "exit_code": 0}}

    def analyze(source, **kwargs):
        raise RuntimeError("detector read failed")

    monkeypatch.setattr(ButterflyAnalysisService, "preflight", preflight)
    monkeypatch.setattr(cli, "analyze_frame", analyze)
    output = tmp_path / "results"
    assert cli.main([
        "batch", "--manifest", str(manifest), "--series", "hold",
        "--poni", str(poni), "--unattended", str(package), "-o", str(output),
    ]) == 1
    report = json.loads(capsys.readouterr().out)
    assert [Path(row["path"]).name for row in seen] == ["frame2.npy"]
    assert report["n_frames"] == report["n_failed"] == 1
    assert report["frames"][0]["status"] == "failed"
    assert "detector read failed" in report["frames"][0]["error"]
    with np.load(output / "results.npz", allow_pickle=False) as arrays:
        metadata = json.loads(str(arrays["__metadata__"].item()))
    assert metadata["complete"] is False
    assert metadata["quality_complete"] is False


def test_unattended_relative_paths_and_frame_selector_keep_recipe(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    package, poni, mask = _inputs(tmp_path)
    monkeypatch.chdir(tmp_path)
    preflight_values = []
    analyzed_configs = []
    analyzed_masks = []

    def preflight(self, source, **kwargs):
        preflight_values.append(kwargs)
        return {"status": {"status_color": "green", "scientific_status": "PASS", "exit_code": 0}}

    def analyze(source, **kwargs):
        analyzed_configs.append(kwargs["config"])
        analyzed_masks.append(kwargs["mask"])
        return {"success": True, "status": "ok"}

    monkeypatch.setattr(ButterflyAnalysisService, "preflight", preflight)
    monkeypatch.setattr(cli, "analyze_frame", analyze)
    assert cli.main([
        "batch", "package/frame*.npy", "--unattended", "package",
        "--poni", "package/geometry.poni", "--mask", "mask.npy",
        "--frame", "0", "--dataset", "data", "-o", "results",
    ]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["n_frames"] == 2
    assert preflight_values[0]["poni"] == str(poni)
    assert preflight_values[0]["mask"] == str(mask)
    assert all(Path(row["path"]).is_absolute() for row in preflight_values[0]["manifest"])
    assert all(row["frame"] == 0 and row["dataset"] == "data" for row in preflight_values[0]["manifest"])
    assert all(config.poni_path == str(poni) for config in analyzed_configs)
    assert analyzed_masks == [str(mask), str(mask)]
    assert all(config.analysis["ridge_method"] == "butterfly_curvature" for config in analyzed_configs)
    assert all(config.analysis["butterfly"]["trace_method"] == "annular_peak" for config in analyzed_configs)


def test_unattended_relative_paths_reach_real_preflight(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    from pyFAI.azimuthalIntegrator import AzimuthalIntegrator

    package, poni, _ = _inputs(tmp_path)
    AzimuthalIntegrator(
        dist=0.12, poni1=0.0003, poni2=0.00035,
        pixel1=0.0001, pixel2=0.0001, wavelength=1.0e-10,
    ).save(str(poni))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        cli, "analyze_frame",
        lambda source, **kwargs: {"success": True, "status": "ok"},
    )

    assert cli.main([
        "batch", "package/frame*.npy", "--unattended", "package",
        "--poni", "package/geometry.poni", "--mask", "mask.npy",
        "-o", "results",
    ]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["n_frames"] == 2
    assert report["preflight"]["status_color"] == "yellow"
    evidence = json.loads((tmp_path / "results" / "preflight" / "preflight.json").read_text(encoding="utf-8"))
    assert evidence["geometry"]["q_unit"] == "nm^-1"
    assert len(evidence["input"]["images"]) == 2


def test_unattended_batch_requires_separate_output_and_calibration(
    tmp_path: Path, capsys
) -> None:
    package, poni, _ = _inputs(tmp_path)
    base = ["batch", str(package / "frame*.npy"), "--unattended", str(package)]
    assert cli.main([*base, "-o", str(tmp_path / "results")]) == 2
    assert "PONI" in json.loads(capsys.readouterr().out)["error"]["message"]
    assert cli.main([*base, "--poni", str(poni), "-o", str(package / "results")]) == 2
    assert "outside" in json.loads(capsys.readouterr().out)["error"]["message"]
    assert cli.main([
        *base, "--poni", str(poni), "-o", str(tmp_path / "results"),
        "--checkpoint", str(package / "checkpoint.json"),
    ]) == 2
    assert "checkpoint" in json.loads(capsys.readouterr().out)["error"]["message"]


def test_cancelled_batch_and_project_never_report_success(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    package, _, _ = _inputs(tmp_path)
    cancelled = batch_module.BatchRunResult([], cancelled=True, total_count=2)
    monkeypatch.setattr(batch_module, "run_batch", lambda *args, **kwargs: cancelled)
    monkeypatch.setattr(export_module, "export_batch", lambda *args, **kwargs: {})
    assert cli.main([
        "batch", str(package / "frame*.npy"), "-o", str(tmp_path / "results"),
    ]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["cancelled"] is True
    assert report["agent"]["exit_code"] == 1

    monkeypatch.setattr(cli, "run_project", lambda *args, **kwargs: cancelled)
    assert cli.main(["project", str(tmp_path / "project.toml")]) == 1
    project_report = json.loads(capsys.readouterr().out)
    assert project_report["cancelled"] is True
    assert project_report["agent"]["exit_code"] == 1
