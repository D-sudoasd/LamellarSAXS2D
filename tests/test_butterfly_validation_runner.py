"""Validate the scientific split guard, independently of fitting accuracy."""
import importlib.util
import hashlib
import json
from pathlib import Path

import pytest
import numpy as np


def test_same_case_seed_cannot_be_reused_as_holdout(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[1] / "scripts" / "validate_butterfly_arcs.py"
    spec = importlib.util.spec_from_file_location("arc_validation_runner", source)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    dev = tmp_path / "dev"
    base = ["--benchmark", "--cases", "ellipse_ratio_400", "--seeds", "1", "--shape", "24"]
    assert runner.main(base + ["--output", str(dev), "--freeze"]) == 0
    frozen = json.loads((dev / "frozen_recipe.json").read_text(encoding="utf-8"))
    assert frozen["job_ids"] == ["ellipse_ratio_400_1"]
    assert frozen["generator_sha256"] == runner.GENERATOR_HASH
    report = json.loads((dev / "report.json").read_text(encoding="utf-8"))
    assert report["context"]["method_version"] == runner.METHOD_VERSION

    def must_not_run(*args, **kwargs):
        raise AssertionError("holdout overlap must fail before any image generation")

    monkeypatch.setattr(runner, "generate_arc_case", must_not_run)
    with pytest.raises(ValueError, match="overlap"):
        runner.main(base + ["--output", str(tmp_path / "hold"), "--split", "holdout",
                           "--frozen-recipe", str(dev / "frozen_recipe.json")])
    assert not (tmp_path / "hold" / "report.json").exists()
    with pytest.raises(ValueError, match="ellipse_preset_override"):
        runner.main(base + ["--output", str(tmp_path / "changed_prior"), "--split", "holdout",
                           "--ellipse-preset", "standard", "--frozen-recipe", str(dev / "frozen_recipe.json")])


def test_disjoint_holdout_records_frozen_hash_and_real_summary(tmp_path):
    source = Path(__file__).resolve().parents[1] / "scripts" / "validate_butterfly_arcs.py"
    spec = importlib.util.spec_from_file_location("arc_validation_holdout_runner", source)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    dev = tmp_path / "dev"
    holdout = tmp_path / "holdout"
    base = ["--benchmark", "--cases", "ellipse_ratio_400", "--shape", "32"]

    assert runner.main(base + ["--seeds", "1", "--output", str(dev), "--freeze"]) == 0
    frozen_path = dev / "frozen_recipe.json"
    frozen_bytes = frozen_path.read_bytes()
    assert runner.main(base + ["--seeds", "2", "--output", str(holdout), "--split", "holdout",
                               "--frozen-recipe", str(frozen_path)]) == 0

    report = json.loads((holdout / "report.json").read_text(encoding="utf-8"))
    context = report["context"]
    frame = report["frames"][0]
    assert context["split"] == "holdout"
    assert context["job_ids"] == ["ellipse_ratio_400_2"]
    assert context["frozen_recipe_sha256"] == hashlib.sha256(frozen_bytes).hexdigest()
    assert frozen_path.read_bytes() == frozen_bytes
    assert context["code_unchanged"] is True
    assert frame["execution_status"] == "complete"
    assert frame["solver_success"] is True
    assert frame["candidate"]["a"] is not None
    assert frame["candidate"]["axis_ratio"] is not None
    assert frame["point_count"] > 0
    assert frame["side_counts"]
    assert frame["generator_sha256"] == runner.GENERATOR_HASH
    assert frame["case_identity"] == {"case_id": "ellipse_ratio_400", "seed": 2, "shape": [32, 32]}


def test_real_project_paths_resolve_against_toml_directory(tmp_path, monkeypatch):
    import tifffile
    from types import SimpleNamespace

    source = Path(__file__).resolve().parents[1] / "scripts" / "validate_butterfly_arcs.py"
    spec = importlib.util.spec_from_file_location("arc_path_runner", source)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    project_dir = tmp_path / "project"
    (project_dir / "data").mkdir(parents=True)
    (project_dir / "geom").mkdir()
    tifffile.imwrite(project_dir / "data/frame_00110.tif", np.ones((8, 8), dtype=np.uint16))
    np.save(project_dir / "mask.npy", np.zeros((8, 8), dtype=bool))
    (project_dir / "geom/cal.poni").write_text("hashable calibration fixture", encoding="utf-8")
    config = project_dir / "project.toml"
    config.write_text('[inputs]\nfiles=["data/frame_00110.tif"]\nponi="geom/cal.poni"\n'
                      '[analysis]\nmask="mask.npy"\nq_window=[0.1,0.5]\n', encoding="utf-8")
    seen = {}

    def geometry(shape, poni):
        seen["poni"] = Path(poni)
        return SimpleNamespace(q=np.ones(shape))

    def record(name, data, qmap, mask, *args, **kwargs):
        seen["mask"] = mask
        return {}, {"id": name, "execution_status": "complete"}

    monkeypatch.setattr(runner, "build_geometry", geometry)
    monkeypatch.setattr(runner, "record_fit", record)
    monkeypatch.chdir(tmp_path)
    assert runner.main(["--project", str(config), "--frames", "110", "--output", str(tmp_path / "out")]) == 0
    assert seen["poni"] == project_dir / "geom/cal.poni"
    np.testing.assert_array_equal(seen["mask"], np.zeros((8, 8), dtype=bool))


def test_real_npy_without_mask_preserves_suffix_width_and_prior(tmp_path, monkeypatch):
    from types import SimpleNamespace

    source = Path(__file__).resolve().parents[1] / "scripts" / "validate_butterfly_arcs.py"
    spec = importlib.util.spec_from_file_location("arc_npy_runner", source)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    data = np.arange(64, dtype=float).reshape(8, 8)
    frame = tmp_path / "frame_007.npy"
    np.save(frame, data)
    original = frame.read_bytes()
    (tmp_path / "cal.poni").write_text("test calibration", encoding="utf-8")
    project = tmp_path / "project.toml"
    project.write_text('[inputs]\nfiles=["frame_007.npy"]\nponi="cal.poni"\n', encoding="utf-8")
    captured = {}

    def record(name, image, qmap, mask, *args, **kwargs):
        captured.update(image=image, mask=mask, analysis=kwargs["analysis"])
        return {}, {"id": name, "execution_status": "complete"}

    monkeypatch.setattr(runner, "build_geometry", lambda shape, poni: SimpleNamespace(q=np.ones(shape)))
    monkeypatch.setattr(runner, "record_fit", record)
    assert runner.main(["--project", str(project), "--frames", "7", "--ellipse-preset", "standard",
                        "--output", str(tmp_path / "out")]) == 0
    np.testing.assert_array_equal(captured["image"], data)
    assert captured["mask"] is None or not captured["mask"].any()
    assert captured["analysis"]["ellipse_preset"] == "standard"
    assert captured["analysis"]["ellipse"]["preset"] == "standard"
    assert frame.read_bytes() == original
    report = json.loads((tmp_path / "out" / "report.json").read_text(encoding="utf-8"))
    assert report["frames"][0]["source_unchanged"] is True
    processed = tmp_path / "frame_007_abs2d_cm-1.npy"
    np.save(processed, data)
    named_project = tmp_path / "named_project.toml"
    named_project.write_text('[inputs]\nfiles=["frame_007_abs2d_cm-1.npy"]\nponi="cal.poni"\n', encoding="utf-8")
    assert runner.main(["--project", str(named_project), "--all-project-inputs",
                        "--output", str(tmp_path / "named_out")]) == 0
    named_report = json.loads((tmp_path / "named_out" / "report.json").read_text(encoding="utf-8"))
    assert named_report["context"]["job_ids"] == [processed.stem]
    assert named_report["frames"][0]["source"] == str(processed)


def test_holdout_rejects_changed_positive_mask_contents(tmp_path, monkeypatch):
    from types import SimpleNamespace

    source = Path(__file__).resolve().parents[1] / "scripts" / "validate_butterfly_arcs.py"
    spec = importlib.util.spec_from_file_location("arc_positive_mask_runner", source)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    np.save(tmp_path / "frame_001.npy", np.ones((8, 8)))
    valid = np.ones((8, 8), dtype=bool)
    mask = tmp_path / "valid.npy"
    np.save(mask, valid)
    (tmp_path / "cal.poni").write_text("test geometry", encoding="utf-8")
    project = tmp_path / "project.toml"
    project.write_text('[inputs]\nfiles=["frame_001.npy"]\nponi="cal.poni"\n'
                       '[analysis]\nvalid_mask="valid.npy"\n', encoding="utf-8")
    monkeypatch.setattr(runner, "build_geometry", lambda shape, poni: SimpleNamespace(q=np.ones(shape)))
    monkeypatch.setattr(runner, "record_fit", lambda name, *a, **kw: ({}, {"id": name, "execution_status": "complete"}))
    dev = tmp_path / "dev"
    args = ["--project", str(project), "--frames", "1"]
    assert runner.main(args + ["--output", str(dev), "--freeze"]) == 0
    frozen = dev / "frozen_recipe.json"
    original_recipe = frozen.read_bytes()
    valid[3, 4] = False
    np.save(mask, valid)
    with pytest.raises(ValueError, match="valid_mask_sha256"):
        runner.main(args + ["--output", str(tmp_path / "holdout"), "--split", "holdout",
                            "--frozen-recipe", str(frozen)])
    assert frozen.read_bytes() == original_recipe

    def change_mask_during_fit(name, *args, **kwargs):
        valid[2, 3] = False
        np.save(mask, valid)
        return {}, {"id": name, "execution_status": "complete"}

    monkeypatch.setattr(runner, "record_fit", change_mask_during_fit)
    target = tmp_path / "changed_during_run"
    assert runner.main(args + ["--output", str(target)]) == 1
    report = json.loads((target / "report.json").read_text(encoding="utf-8"))
    assert report["context"]["calibration_unchanged"] is False


def test_runner_refuses_partial_existing_evidence(tmp_path):
    source = Path(__file__).resolve().parents[1] / "scripts" / "validate_butterfly_arcs.py"
    spec = importlib.util.spec_from_file_location("arc_partial_runner", source)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    target = tmp_path / "partial"
    target.mkdir()
    frozen = target / "frozen_recipe.json"
    frozen.write_text("earlier interrupted run", encoding="utf-8")
    with pytest.raises(FileExistsError, match="new empty output"):
        runner.main(["--benchmark", "--output", str(target), "--freeze"])
    assert frozen.read_text(encoding="utf-8") == "earlier interrupted run"
