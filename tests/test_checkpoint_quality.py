from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from butterfly_saxs import cli as cli_module
from butterfly_saxs.batch import FrameRef, run_batch


def test_frame_ref_key_includes_service_metadata_selectors_with_explicit_priority(
    tmp_path: Path,
) -> None:
    source = tmp_path / "multi.npz"
    source.write_bytes(b"placeholder")

    metadata_frame = FrameRef(
        source,
        frame_id="same",
        metadata={"frame": "0", "dataset": "series"},
    )
    metadata_frame_index = FrameRef(
        source,
        frame_id="same",
        metadata={"frame_index": 1, "dataset_name": "series"},
    )
    assert metadata_frame.key != metadata_frame_index.key
    assert json.loads(metadata_frame.key)["frame"] == 0
    assert json.loads(metadata_frame_index.key)["frame"] == 1
    assert json.loads(metadata_frame.key)["dataset"] == "series"

    explicit = FrameRef(
        source,
        frame=7,
        dataset="explicit",
        metadata={"frame": 2, "dataset": "metadata"},
    )
    identity = json.loads(explicit.key)
    assert identity["frame"] == 7
    assert identity["dataset"] == "explicit"


def test_cli_batch_selector_changes_invalidate_checkpoint(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    source = tmp_path / "multi.npy"
    source.write_bytes(b"placeholder")
    output = tmp_path / "output"
    checkpoint = tmp_path / "checkpoint.json"
    seen_selectors: list[tuple[int | None, str | None]] = []

    def fake_analyze(source_path, **kwargs):
        del source_path
        seen_selectors.append((kwargs.get("frame"), kwargs.get("dataset")))
        return {"parameters": {"value": 1}}

    monkeypatch.setattr(cli_module, "analyze_frame", fake_analyze)

    import butterfly_saxs.export as export_module

    monkeypatch.setattr(export_module, "export_batch", lambda *args, **kwargs: {})

    first_code = cli_module.main(
        [
            "batch",
            str(source),
            "--frame",
            "0",
            "--dataset",
            "series",
            "--output",
            str(output),
            "--checkpoint",
            str(checkpoint),
            "--force",
        ]
    )
    first_report = json.loads(capsys.readouterr().out)
    assert first_code == 0
    assert first_report["frames"][0]["frame"]["frame"] == 0
    assert first_report["frames"][0]["frame"]["dataset"] == "series"
    checkpoint_payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert checkpoint_payload["config"]["analysis"]["frame"] == 0
    assert checkpoint_payload["config"]["analysis"]["dataset"] == "series"
    assert seen_selectors == [(0, "series")]

    second_code = cli_module.main(
        [
            "batch",
            str(source),
            "--frame",
            "1",
            "--dataset",
            "series",
            "--output",
            str(output),
            "--checkpoint",
            str(checkpoint),
            "--resume",
            "--force",
        ]
    )
    captured = capsys.readouterr()
    assert second_code == 2
    assert "checkpoint input hash mismatch" in captured.err
    assert seen_selectors == [(0, "series")]


def test_cli_selector_override_preserves_resolved_manifest_order(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    source = tmp_path / "multi.npy"
    source.write_bytes(b"placeholder")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "frames": [
                    {"path": str(source), "time": 20.0, "frame": 0, "frame_id": "late"},
                    {"path": str(source), "time": 10.0, "frame": 1, "frame_id": "early"},
                ]
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "output"

    monkeypatch.setattr(
        cli_module,
        "analyze_frame",
        lambda source_path, **kwargs: {"parameters": {"value": kwargs["frame"]}},
    )
    import butterfly_saxs.export as export_module

    monkeypatch.setattr(export_module, "export_batch", lambda *args, **kwargs: {})

    assert (
        cli_module.main(
            [
                "batch",
                str(source),
                "--manifest",
                str(manifest),
                "--frame",
                "3",
                "--dataset",
                "series",
                "--output",
                str(output),
                "--force",
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert [item["frame"]["frame_id"] for item in report["frames"]] == [
        "early",
        "late",
    ]
    assert [item["frame"]["frame"] for item in report["frames"]] == [3, 3]
    assert [item["frame"]["dataset"] for item in report["frames"]] == [
        "series",
        "series",
    ]


def test_empty_observation_never_becomes_warm_start_seed(tmp_path: Path) -> None:
    paths = []
    for name in ("empty.tif", "next.tif"):
        path = tmp_path / name
        path.write_bytes(name.encode("ascii"))
        paths.append(path)
    calls: list[object] = []

    def analyze(frame: FrameRef, initial=None):
        calls.append(initial)
        if frame.path.name == "empty.tif":
            return {
                "observed": None,
                "metrics": {"ndata": 0, "flags": ["no_observed"]},
                "parameters": {"value": 1},
            }
        return {"metrics": {"ndata": 1}, "parameters": {"value": 2}}

    run = run_batch(paths, analyze, mode="warm_start")

    assert run[0].failed
    assert "no_observed" in (run[0].error or "")
    assert calls == [None, None]


def test_zero_ndata_without_flag_is_not_a_valid_warm_start_seed(tmp_path: Path) -> None:
    paths = []
    for name in ("empty.tif", "next.tif"):
        path = tmp_path / name
        path.write_bytes(name.encode("ascii"))
        paths.append(path)
    calls: list[object] = []

    def analyze(frame: FrameRef, initial=None):
        calls.append(initial)
        if frame.path.name == "empty.tif":
            return {"metrics": {"ndata": 0}, "parameters": {"value": 1}}
        return {"metrics": {"ndata": 1}, "parameters": {"value": 2}}

    run = run_batch(paths, analyze, mode="warm_start")

    assert run[0].failed
    assert "ndata=0" in (run[0].error or "")
    assert calls == [None, None]


@pytest.mark.parametrize(
    "bad_result_factory",
    [
        pytest.param(lambda: None, id="analyzer-none"),
        pytest.param(
            lambda: {
                "observed": np.empty(0, dtype=float),
                "parameters": {"value": 1},
            },
            id="empty-observed-array",
        ),
        pytest.param(
            lambda: {
                "observed": np.full(3, np.nan, dtype=float),
                "parameters": {"value": 1},
            },
            id="all-nan-observed-array",
        ),
    ],
)
def test_missing_or_empty_observed_cannot_seed_or_restore_checkpoint(
    tmp_path: Path,
    bad_result_factory,
) -> None:
    paths = []
    for name in ("bad.tif", "next.tif"):
        path = tmp_path / name
        path.write_bytes(name.encode("ascii"))
        paths.append(path)
    checkpoint = tmp_path / "checkpoint.json"
    calls: list[tuple[str, object]] = []

    def analyze(frame: FrameRef, initial=None):
        calls.append((frame.path.name, initial))
        if frame.path.name == "bad.tif":
            return bad_result_factory()
        return {"parameters": {"value": 2}}

    first = run_batch(paths, analyze, mode="warm_start", checkpoint=checkpoint)

    assert [item.status for item in first] == ["failed", "ok"]
    assert first[0].error
    assert calls == [("bad.tif", None), ("next.tif", None)]

    # Simulate a legacy checkpoint that mislabeled the invalid result as a
    # successful frame.  Resume must re-check the stored payload instead of
    # restoring it as a valid warm-start predecessor.
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    payload["frames"][0]["status"] = "ok"
    checkpoint.write_text(json.dumps(payload), encoding="utf-8")

    calls.clear()
    resumed = run_batch(
        paths,
        analyze,
        mode="warm_start",
        checkpoint=checkpoint,
        resume=True,
    )

    assert resumed[0].failed
    assert not resumed[0].resumed
    assert resumed[1].resumed
    assert calls == [("bad.tif", None)]
