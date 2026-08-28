from __future__ import annotations

import csv
import json
from enum import IntEnum
from pathlib import Path

import numpy as np
import pytest

from butterfly_saxs.batch import FrameFitResult, FrameRef, build_frame_refs, run_batch
from butterfly_saxs.export import _contains_omitted_array, export_batch


def _touch_frames(root: Path, names: list[str]) -> list[Path]:
    paths = []
    for name in names:
        path = root / name
        path.write_bytes(name.encode("ascii"))
        paths.append(path)
    return paths


def test_natural_sort_and_manifest_time_take_priority(tmp_path: Path) -> None:
    paths = _touch_frames(tmp_path, ["frame10.tif", "frame2.tif", "frame1.tif"])
    refs = build_frame_refs(paths)
    assert [ref.path.name for ref in refs] == ["frame1.tif", "frame2.tif", "frame10.tif"]

    manifest = [
        {"path": str(paths[0]), "time": 20.0},
        {"path": str(paths[1]), "time": 10.0},
    ]
    refs = build_frame_refs(paths, manifest=manifest)
    assert [ref.path.name for ref in refs] == ["frame2.tif", "frame10.tif"]
    assert [ref.time for ref in refs] == [10.0, 20.0]


def test_manifest_file_resolves_relative_frame_paths_beside_it(tmp_path: Path) -> None:
    manifest_dir = tmp_path / "sequence"
    data_dir = manifest_dir / "data"
    data_dir.mkdir(parents=True)
    frame = data_dir / "frame.npy"
    frame.write_bytes(b"frame")
    manifest = manifest_dir / "manifest.json"
    manifest.write_text(
        json.dumps(
            {"frames": [{"path": "data/frame.npy", "frame": 0, "dataset": "series"}]}
        ),
        encoding="utf-8",
    )

    refs = build_frame_refs([], manifest=manifest)

    assert refs[0].path.resolve() == frame.resolve()
    assert refs[0].frame == 0
    assert refs[0].dataset == "series"


def test_analyzer_source_parameter_receives_path_not_frame_ref(tmp_path: Path) -> None:
    path = _touch_frames(tmp_path, ["frame1.tif"])[0]
    received: list[Path] = []

    def analyze(source):
        received.append(Path(source))
        return {"status": "ok"}

    run = run_batch([path], analyze)
    assert run[0].ok
    assert received == [path]


def test_warm_start_lineage_does_not_propagate_failed_frame(tmp_path: Path) -> None:
    paths = _touch_frames(tmp_path, ["frame1.tif", "frame2.tif", "frame3.tif"])
    calls: list[tuple[str, object]] = []

    def analyze(frame: FrameRef, initial=None):
        calls.append((frame.path.name, initial))
        if frame.path.name == "frame2.tif":
            raise RuntimeError("bad detector frame")
        return {"value": frame.path.stem, "initial": initial}

    run = run_batch(paths, analyze, mode="warm_start")
    assert [item.status for item in run] == ["ok", "failed", "ok"]
    assert calls[0][1] is None
    assert calls[1][1]["value"] == "frame1"
    assert calls[2][1]["value"] == "frame1"
    assert run[1].warm_start_from == FrameRef(paths[0]).key
    assert run[2].warm_start_from == FrameRef(paths[0]).key


def test_warm_start_quality_gate_rejects_explicit_and_nested_failures_without_rmse_threshold(
    tmp_path: Path,
) -> None:
    paths = _touch_frames(
        tmp_path,
        ["frame1.tif", "frame2.tif", "frame3.tif", "frame4.tif", "frame5.tif"],
    )
    calls: list[tuple[str, object]] = []

    def analyze(frame: FrameRef, initial=None):
        name = frame.path.name
        calls.append((name, initial))
        if name == "frame1.tif":
            # A large RMSE is not a batch-layer quality threshold.  The
            # explicit success flag is the only quality signal here.
            return {"success": True, "rmse": 1e9, "parameters": {"value": 1}}
        if name == "frame2.tif":
            return {"success": False, "rmse": 0.0, "parameters": {"value": 2}}
        if name == "frame3.tif":
            return {
                "parameters": {"value": 3},
                "full2d": {"success": False, "parameters": {"value": 3}},
            }
        if name == "frame4.tif":
            return {
                "parameters": {"value": 4},
                "full2d": {"status": "error", "parameters": {"value": 4}},
            }
        return {"success": True, "rmse": 1e9, "parameters": {"value": 5}}

    run = run_batch(paths, analyze, mode="warm_start")

    assert [item.status for item in run] == ["ok", "failed", "failed", "failed", "ok"]
    assert [initial["value"] if initial else None for _, initial in calls] == [
        None,
        1,
        1,
        1,
        1,
    ]
    assert "success=False" in (run[1].error or "")
    assert "full2d.success=False" in (run[2].error or "")
    assert "full2d.status=error" in (run[3].error or "")
    assert run[4].warm_start_from == FrameRef(paths[0]).key


def test_warm_start_quality_gate_rejects_metrics_ellipse_full2d_and_failure_flags(
    tmp_path: Path,
) -> None:
    paths = _touch_frames(
        tmp_path,
        [
            "frame1.tif",
            "frame2.tif",
            "frame3.tif",
            "frame4.tif",
            "frame5.tif",
            "frame6.tif",
            "frame7.tif",
            "frame8.tif",
        ],
    )
    calls: list[tuple[str, object]] = []

    def analyze(frame: FrameRef, initial=None):
        name = frame.path.name
        calls.append((name, initial))
        if name == "frame1.tif":
            return {"parameters": {"value": 1}}
        if name == "frame2.tif":
            return {
                "metrics": {"success": False},
                "parameters": {"value": 2},
            }
        if name == "frame3.tif":
            return {
                "ellipse_fit": {"status": "insufficient_data"},
                "parameters": {"value": 3},
            }
        if name == "frame4.tif":
            return {
                "full2d": {"status": "insufficient_data"},
                "parameters": {"value": 4},
            }
        if name == "frame5.tif":
            return {
                "flags": ["intensity_fit_failed:RuntimeError"],
                "parameters": {"value": 5},
            }
        if name == "frame6.tif":
            return {
                "metrics": {"flags": ["analysis_validation_failed:q window"]},
                "parameters": {"value": 6},
            }
        if name == "frame7.tif":
            return {
                "ellipse_fit": {
                    "status": "ok",
                    "success": True,
                    "quality_status": "FAIL",
                },
                "parameters": {"value": 7},
            }
        return {"parameters": {"value": 8}}

    run = run_batch(paths, analyze, mode="warm_start")

    assert [item.status for item in run] == [
        "ok",
        "failed",
        "failed",
        "failed",
        "failed",
        "failed",
        "failed",
        "ok",
    ]
    assert [initial["value"] if initial else None for _, initial in calls] == [
        None,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
    ]
    assert "metrics.success=False" in (run[1].error or "")
    assert "ellipse_fit.status=insufficient_data" in (run[2].error or "")
    assert "full2d.status=insufficient_data" in (run[3].error or "")
    assert "intensity_fit_failed:RuntimeError" in (run[4].error or "")
    assert "analysis_validation_failed:q window" in (run[5].error or "")
    assert "ellipse_fit.quality_status=FAIL" in (run[6].error or "")
    assert run[7].warm_start_from == FrameRef(paths[0]).key


def test_batch_rejects_top_level_fail_status(tmp_path: Path) -> None:
    path = _touch_frames(tmp_path, ["frame1.tif"])[0]

    run = run_batch([path], lambda _frame: {"status": "FAIL"})

    assert run[0].status == "failed"
    assert "status=FAIL" in (run[0].error or "")


def test_frame_ref_key_uses_canonical_path_frame_and_dataset_identity(tmp_path: Path) -> None:
    left = tmp_path / "left" / "frame.npy"
    right = tmp_path / "right" / "frame.npy"
    left.parent.mkdir()
    right.parent.mkdir()
    left.write_bytes(b"left")
    right.write_bytes(b"right")

    left_ref = FrameRef(left, frame_id="0", metadata={"dataset": "entry"})
    right_ref = FrameRef(right, frame_id="0", metadata={"dataset": "entry"})
    other_dataset = FrameRef(left, frame_id="0", metadata={"dataset": "other"})
    other_frame = FrameRef(left, frame_id="1", metadata={"dataset": "entry"})

    assert left_ref.key != right_ref.key
    assert left_ref.key != other_dataset.key
    assert left_ref.key != other_frame.key
    assert left_ref.key == FrameRef(left.resolve(), frame_id="0", metadata={"dataset": "entry"}).key
    assert json.loads(left_ref.key)["path"] == left_ref.path.resolve().as_posix().casefold()


def test_frame_refs_preserve_frame_and_dataset_selectors_for_generators(tmp_path: Path) -> None:
    source = tmp_path / "multi.npz"
    source.write_bytes(b"placeholder")
    inputs = (
        item
        for item in (
            FrameRef(source, frame=0, dataset="entry/data", frame_id="first"),
            {"path": source, "frame_index": 1, "dataset": "entry/data", "frame_id": "second"},
        )
    )

    refs = build_frame_refs(inputs)

    assert [(ref.frame, ref.dataset, ref.frame_id) for ref in refs] == [
        (0, "entry/data", "first"),
        (1, "entry/data", "second"),
    ]
    assert refs[0].key != refs[1].key
    assert refs[1].to_dict()["frame"] == 1
    with pytest.raises(ValueError, match="non-negative integer"):
        FrameRef(source, frame=1.5)


def test_checkpoint_resume_and_hash_guard(tmp_path: Path) -> None:
    paths = _touch_frames(tmp_path, ["frame1.tif", "frame2.tif"])
    checkpoint = tmp_path / "work" / "checkpoint.json"
    first_calls: list[str] = []

    def analyze(frame: FrameRef):
        first_calls.append(frame.path.name)
        return {"parameters": {"a": {"value": 1.0}}}

    run_batch(paths, analyze, config={"loss": "linear"}, checkpoint=checkpoint)
    assert len(first_calls) == 2
    assert checkpoint.exists()
    assert not list(checkpoint.parent.glob("*.tmp"))

    resumed_calls: list[str] = []

    def should_not_run(frame: FrameRef):
        resumed_calls.append(frame.path.name)
        return {}

    resumed = run_batch(
        paths,
        should_not_run,
        config={"loss": "linear"},
        checkpoint=checkpoint,
        resume=True,
    )
    assert resumed_calls == []
    assert all(item.resumed for item in resumed)
    with pytest.raises(ValueError, match="config hash mismatch"):
        run_batch(paths, should_not_run, config={"loss": "soft_l1"}, checkpoint=checkpoint, resume=True)


def test_checkpoint_omits_detector_sized_arrays_but_keeps_restart_parameters(tmp_path: Path) -> None:
    path = _touch_frames(tmp_path, ["frame1.tif"])[0]
    checkpoint = tmp_path / "checkpoint.json"

    def analyze(source):
        return {
            "parameters": {"a": 0.8, "theta_deg": 18.0},
            "image": np.ones((256, 256), dtype=np.float64),
            "residual": np.zeros((256, 256), dtype=np.float64),
        }

    run_batch([path], analyze, checkpoint=checkpoint)
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    result = payload["frames"][0]["result"]
    assert result["parameters"]["a"] == pytest.approx(0.8)
    assert result["image"]["array_omitted"] is True
    assert result["image"]["shape"] == [256, 256]
    assert checkpoint.stat().st_size < 20_000


def test_checkpoint_restores_top_level_and_nested_parameters_for_warm_start(tmp_path: Path) -> None:
    paths = _touch_frames(tmp_path, ["frame1.tif", "frame2.tif"])
    checkpoint = tmp_path / "checkpoint.json"

    class PipelineLikeResult:
        def __init__(self) -> None:
            self.parameters = {"top": 1.0}
            self.full2d = {"parameters": {"nested": 2.0}}
            self.image = np.ones((8, 8), dtype=np.float64)

        def to_mapping(self, *, include_arrays: bool = False):
            del include_arrays
            return {
                "full2d": self.full2d,
                "image": {"shape": [8, 8], "dtype": "float64"},
            }

    def first_run(frame: FrameRef, initial=None):
        del initial
        if frame.path.name == "frame2.tif":
            raise RuntimeError("stop after first frame")
        return PipelineLikeResult()

    first = run_batch(paths, first_run, mode="warm_start", checkpoint=checkpoint)
    assert first[0].ok and first[1].failed
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    saved = payload["frames"][0]["result"]
    assert saved["parameters"] == {"top": 1.0}
    assert saved["full2d"]["parameters"] == {"nested": 2.0}
    assert saved["image"]["array_omitted"] is True

    resumed_initials: list[object] = []

    def resumed_run(frame: FrameRef, initial=None):
        if frame.path.name == "frame2.tif":
            resumed_initials.append(initial)
        return {"parameters": {"top": 3.0}}

    resumed = run_batch(
        paths,
        resumed_run,
        mode="warm_start",
        checkpoint=checkpoint,
        resume=True,
    )
    assert resumed[0].resumed
    restored = resumed_initials[0]
    assert restored == {"top": 1.0}


def test_input_fingerprint_changes_when_content_changes_without_stat_change(tmp_path: Path) -> None:
    from butterfly_saxs.batch import input_fingerprint

    path = _touch_frames(tmp_path, ["frame1.tif"])[0]
    ref = FrameRef(path)
    before = input_fingerprint([ref])
    stat = path.stat()
    path.write_bytes(b"update.tif")
    # Preserve size and timestamps to ensure the content digest, not mtime/size,
    # is what invalidates a resume.
    import os

    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert input_fingerprint([ref]) != before


def test_export_files_round_trip_without_dropping_arrays_or_flags(tmp_path: Path) -> None:
    paths = _touch_frames(tmp_path, ["frame1.tif", "frame2.tif"])

    def analyze(frame: FrameRef):
        number = int(frame.path.stem[-1])
        return {
            "parameters": {
                "spacing": {"value": float(number), "uncertainty": 0.1, "fixed": False, "unit": "nm"},
                "axis_ratio": {"value": 0.6, "uncertainty": 0.0, "fixed": True},
            },
            "ridge_points": [{"qx": number, "qy": number + 0.5, "component": 0}],
            "ellipse_fit": {"a": number, "b": 0.6, "flags": ["symmetric"]},
            "full2d": {
                "status": "ok",
                "success": True,
                "rmse": 0.01 * number,
                "weighted_rmse": 0.005 * number,
                "ndata": 12,
                "nfev": 7,
                "condition_number": 9.0,
            },
            "scientific_flags": ["absolute_intensity", "calibrated"],
            "full_image": np.arange(12, dtype=float).reshape(3, 4),
        }

    run = run_batch(paths, analyze)
    outputs = export_batch(run, tmp_path / "exports")
    assert set(("frame_summary", "parameters_long", "ridge_points", "ellipse_fit", "ellipse_fit_jsonl", "manifest", "provenance", "npz", "evolution_png")) <= set(outputs)
    with outputs["frame_summary"].open(newline="", encoding="utf-8") as handle:
        summary = list(csv.DictReader(handle))
    assert len(summary) == 2
    assert summary[0]["scientific_flags"]
    assert summary[0]["full2d_status"] == "ok"
    assert summary[0]["full2d_success"] == "True"
    assert summary[0]["rmse"] == "0.01"
    assert summary[0]["weighted_rmse"] == "0.005"
    assert summary[0]["ndata"] == "12"
    assert summary[0]["condition_number"] == "9.0"
    with outputs["parameters_long"].open(newline="", encoding="utf-8") as handle:
        parameters = list(csv.DictReader(handle))
    assert {row["parameter"] for row in parameters} == {"spacing", "axis_ratio"}
    ellipse = json.loads(outputs["ellipse_fit"].read_text(encoding="utf-8"))
    assert ellipse["frames"][0]["ellipse_fit"]["flags"] == ["symmetric"]
    with np.load(outputs["npz"], allow_pickle=False) as arrays:
        assert arrays["frame_0000__full_image"].shape == (3, 4)
        assert arrays["frame_0001__full_image"].shape == (3, 4)
    assert outputs["evolution_png"].stat().st_size > 0


def test_export_provenance_records_versions_and_stays_strict_json(tmp_path: Path) -> None:
    frame = FrameFitResult(
        frame=FrameRef(tmp_path / "frame1.tif"),
        result={"parameters": {"a": float("nan")}},
    )

    outputs = export_batch([frame], tmp_path / "exports")

    provenance = json.loads(outputs["provenance"].read_text(encoding="utf-8"))
    versions = provenance["versions"]
    assert {
        "python",
        "ButterflySAXS",
        "numpy",
        "scipy",
        "fabio",
        "pyFAI",
    } <= set(versions)
    assert all(value is None or isinstance(value, str) for value in versions.values())
    assert "NaN" not in outputs["provenance"].read_text(encoding="utf-8")
    manifest = json.loads(outputs["manifest"].read_text(encoding="utf-8"))
    assert manifest["provenance"]["versions"] == versions


def test_npz_metadata_marks_checkpoint_omissions_incomplete(tmp_path: Path) -> None:
    frame = FrameFitResult(
        frame=FrameRef(tmp_path / "frame1.tif"),
        result={
            "parameters": {"a": 0.8},
            "image": {"array_omitted": True, "shape": [256, 256], "dtype": "float64"},
        },
        resumed=True,
    )

    outputs = export_batch([frame], tmp_path / "exports")

    with np.load(outputs["npz"], allow_pickle=False) as arrays:
        metadata = json.loads(str(arrays["__metadata__"]))
    assert metadata["complete"] is False
    assert metadata["missing_frames"] == [0]


def test_export_npz_treats_pyfai_like_orientation_enum_as_a_leaf(tmp_path: Path) -> None:
    """pyFAI's detector orientation enum must not recurse through ``__objclass__``."""

    class Orientation(IntEnum):
        BottomRight = 3

    class Geometry:
        def __init__(self) -> None:
            self.orientation = Orientation.BottomRight

    frame = FrameFitResult(
        frame=FrameRef(tmp_path / "frame1.tif"),
        result={"geometry": Geometry(), "parameters": {"a": 1.0}},
    )

    outputs = export_batch([frame], tmp_path / "exports")

    with np.load(outputs["npz"], allow_pickle=False) as arrays:
        metadata = json.loads(str(arrays["__metadata__"]))
    assert metadata["complete"] is True


def test_omitted_array_scan_terminates_on_cycles_and_ignores_private_links() -> None:
    class Cycle:
        def __init__(self) -> None:
            self.child = self
            self._private_marker = {"array_omitted": True}

    assert _contains_omitted_array(Cycle()) is False
    assert _contains_omitted_array({"array_omitted": np.bool_(True)}) is True
    assert _contains_omitted_array({"array_omitted": np.bool_(False)}) is False


def test_parameter_long_extracts_top_level_and_nested_pipeline_parameters(tmp_path: Path) -> None:
    class TopLevelResult:
        parameters = {
            "top": {
                "value": 1.25,
                "stderr": 0.2,
                "fixed": True,
                "unit": "nm",
                "flags": ["bound"],
            },
            "not_finite": {"value": float("nan"), "stderr": float("nan")},
        }
        full2d = {"parameters": {"should_not_replace_top_level": {"value": 9.0}}}

    nested_parameters = {
        "nested": {"value": 2.5, "stderr": 0.3, "unit": "deg"},
        "numpy_scalar": np.float64(0.25),
    }
    nested = {
        # Checkpoint JSON retains this public alias beside the authoritative
        # full2d block; resume exports must keep the nested diagnostics.
        "parameters": dict(nested_parameters),
        "full2d": {
            "parameters": nested_parameters,
            "flags": ["empirical_model_only"],
        }
    }
    frames = [
        FrameFitResult(frame=FrameRef(tmp_path / "top.tif"), result=TopLevelResult()),
        FrameFitResult(frame=FrameRef(tmp_path / "nested.tif"), result=nested),
    ]

    outputs = export_batch(frames, tmp_path / "exports")

    with outputs["parameters_long"].open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["parameter"] for row in rows} == {
        "top",
        "not_finite",
        "nested",
        "numpy_scalar",
    }
    top = next(row for row in rows if row["parameter"] == "top")
    assert top["stderr"] == "0.2"
    assert top["uncertainty"] == "0.2"
    assert top["fixed"] == "True"
    assert top["unit"] == "nm"
    assert json.loads(top["flags"]) == ["bound"]
    not_finite = next(row for row in rows if row["parameter"] == "not_finite")
    assert not_finite["value"] not in {"0", "0.0"}
    assert not_finite["stderr"] not in {"0", "0.0"}
    nested_row = next(row for row in rows if row["parameter"] == "nested")
    assert nested_row["stderr"] == "0.3"
    assert nested_row["unit"] == "deg"
    assert json.loads(nested_row["flags"]) == ["empirical_model_only"]
    numpy_scalar = next(row for row in rows if row["parameter"] == "numpy_scalar")
    assert numpy_scalar["value"] == "0.25"
    assert json.loads(numpy_scalar["flags"]) == ["empirical_model_only"]


def test_evolution_plot_uses_separate_panels_for_different_units(tmp_path: Path) -> None:
    frame = FrameFitResult(
        frame=FrameRef(tmp_path / "frame1.tif"),
        result={
            "parameters": {
                "spacing": {"value": 1.0, "unit": "nm"},
                "theta": {"value": 30.0, "unit": "deg"},
            },
        },
    )

    outputs = export_batch([frame], tmp_path / "exports")

    from matplotlib.image import imread

    image = imread(outputs["evolution_png"])
    assert image.shape[0] > 800
