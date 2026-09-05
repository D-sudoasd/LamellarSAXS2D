from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import butterfly_saxs.annotation_pack as annotation_pack_module
from butterfly_saxs.annotation_pack import AnnotationPackError, build_annotation_pack


def _write_frame(path: Path, value: float) -> None:
    array = np.full((8, 8), value, dtype=np.float32)
    array[2:4, 2:4] = value + 10.0
    np.save(path, array)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_build_annotation_pack_creates_eight_blind_unique_frames_and_prefilled_templates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = tmp_path / "package"
    package.mkdir()
    frames_dir = package / "frames"
    frames_dir.mkdir()
    rt_path = frames_dir / "rt.npy"
    _write_frame(rt_path, 1.0)
    hold_paths: list[Path] = []
    for index in range(8):
        path = frames_dir / f"hold_{index}.npy"
        _write_frame(path, float(index))
        hold_paths.append(path)

    rt_manifest = package / "rt_manifest.json"
    rt_manifest.write_text(
        json.dumps([{"path": "frames/rt.npy", "time": 0.0}]), encoding="utf-8"
    )
    hold_manifest = package / "hold_manifest.json"
    hold_manifest.write_text(
        json.dumps(
            [
                {"path": f"frames/{path.name}", "time": float(index)}
                for index, path in enumerate(hold_paths)
            ]
        ),
        encoding="utf-8",
    )
    preflight = package / "preflight.json"
    preflight.write_text(
        json.dumps(
            {
                "frames": [
                    {
                        "path": f"frames/{path.name}",
                        "summary": {
                            "negative_fraction": 0.5 if index == 4 else 0.0,
                            "robust_high_fraction": 0.25 if index == 5 else 0.0,
                        },
                    }
                    for index, path in enumerate(hold_paths)
                ],
                "fit": {"must_not_be_used": True},
                "model": {"must_not_be_used": True},
            }
        ),
        encoding="utf-8",
    )
    poni = package / "geometry.poni"
    poni.write_text("Distance = 0.1\n", encoding="utf-8")
    mask = package / "mask.npy"
    np.save(mask, np.zeros((8, 8), dtype=np.uint8))

    source_files = [rt_path, *hold_paths, rt_manifest, hold_manifest, preflight, poni, mask]
    before_hashes = {
        path: hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files
    }

    load_calls: list[Path] = []
    original_load_image = annotation_pack_module.load_image

    def counting_load_image(path: str | Path, **kwargs: object):
        load_calls.append(Path(path))
        return original_load_image(path, **kwargs)

    monkeypatch.setattr(annotation_pack_module, "load_image", counting_load_image)

    output = tmp_path / "annotation_pack"
    result = build_annotation_pack(
        package,
        rt_manifest,
        hold_manifest,
        output,
        preflight_json=preflight,
        poni=poni,
        mask=mask,
    )

    manifest_rows = _read_rows(output / "annotation_manifest.csv")
    assert result["candidate_count"] == 8
    assert len(load_calls) == 8
    assert len(manifest_rows) == 8
    assert len({row["blind_id"] for row in manifest_rows}) == 8
    assert len({(row["source_path"], row["selector"]) for row in manifest_rows}) == 8
    assert {row["role"] for row in manifest_rows} == {
        "RT",
        "hold_first",
        "hold_middle",
        "hold_last",
        "difficult_1",
        "difficult_2",
        "difficult_3",
        "difficult_4",
    }
    assert all(len(row["sha256"]) == 64 for row in manifest_rows)

    payload = output / "blind_payload"
    for name in ("annotator_a.csv", "annotator_b.csv"):
        rows = _read_rows(payload / name)
        assert [row["blind_id"] for row in rows] == [
            f"blind_{index:03d}" for index in range(1, 9)
        ]
        assert all(
            row["coordinate_system"]
            == "image_pixel_x_right_y_up_origin_lower_left"
            for row in rows
        )
        assert all(len(row["image_version"]) == 64 for row in rows)
        assert all(row["annotator"] == "" for row in rows)
    consensus_rows = _read_rows(output / "consensus_review.csv")
    assert len(consensus_rows) == 8
    assert all(row["consensus_status"] == "" for row in consensus_rows)
    assert len(list(payload.glob("blind_*.png"))) == 8
    assert all(path.stat().st_size > 0 for path in payload.glob("blind_*.png"))
    assert not (payload / "annotation_manifest.csv").exists()
    assert not (payload / "annotation_status.json").exists()

    status = json.loads((output / "annotation_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "awaiting_human_annotations"
    assert status["human_consensus"] is False
    assert status["input"]["read_only"] is True
    assert "fit_performed" not in status["input"]
    assert status["display"]["overlay_policy"] == "none"
    assert all(item["unchanged"] for item in status["input_hashes"])
    assert status["files"]["annotation_manifest"] == "annotation_manifest.csv"
    assert status["files"]["annotator_a"] == "blind_payload/annotator_a.csv"
    assert len(status["blind_image_hashes"]) == 8
    assert len(status["immutable_output_hashes"]) == 10
    assert all(
        hashlib.sha256((output / relative).read_bytes()).hexdigest() == digest
        for relative, digest in status["immutable_output_hashes"].items()
    )

    after_hashes = {
        path: hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files
    }
    assert after_hashes == before_hashes


def test_build_annotation_pack_refuses_existing_output(tmp_path: Path) -> None:
    package = tmp_path / "package"
    package.mkdir()
    frames_dir = package / "frames"
    frames_dir.mkdir()
    rt_path = frames_dir / "rt.npy"
    _write_frame(rt_path, 1.0)
    hold_paths: list[Path] = []
    for index in range(8):
        path = frames_dir / f"hold_{index}.npy"
        _write_frame(path, float(index))
        hold_paths.append(path)
    rt_manifest = package / "rt.json"
    rt_manifest.write_text(json.dumps([{"path": "frames/rt.npy"}]), encoding="utf-8")
    hold_manifest = package / "hold.json"
    hold_manifest.write_text(
        json.dumps([{"path": f"frames/{path.name}"} for path in hold_paths]),
        encoding="utf-8",
    )
    output = tmp_path / "pack"
    output.mkdir()
    sentinel = output / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="不覆盖"):
        build_annotation_pack(package, rt_manifest, hold_manifest, output)
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_build_annotation_pack_rejects_output_inside_read_only_package(
    tmp_path: Path,
) -> None:
    package = tmp_path / "package"
    frames = package / "frames"
    frames.mkdir(parents=True)
    _write_frame(frames / "rt.npy", 1.0)
    hold_rows = []
    for index in range(3):
        _write_frame(frames / f"hold_{index}.npy", float(index))
        hold_rows.append({"path": f"frames/hold_{index}.npy"})
    rt_manifest = package / "rt.json"
    rt_manifest.write_text(json.dumps([{"path": "frames/rt.npy"}]), encoding="utf-8")
    hold_manifest = package / "hold.json"
    hold_manifest.write_text(json.dumps(hold_rows), encoding="utf-8")

    with pytest.raises(AnnotationPackError, match="不得位于.*原始数据包内部"):
        build_annotation_pack(
            package,
            rt_manifest,
            hold_manifest,
            package / "derived_annotations",
        )


def test_build_annotation_pack_rejects_data_local_output(tmp_path: Path) -> None:
    package = tmp_path / "package"
    frames = package / "frames"
    frames.mkdir(parents=True)
    _write_frame(frames / "rt.npy", 1.0)
    hold_rows = []
    for index in range(3):
        _write_frame(frames / f"hold_{index}.npy", float(index))
        hold_rows.append({"path": f"frames/hold_{index}.npy"})
    rt_manifest = package / "rt.json"
    rt_manifest.write_text(json.dumps([{"path": "frames/rt.npy"}]), encoding="utf-8")
    hold_manifest = package / "hold.json"
    hold_manifest.write_text(json.dumps(hold_rows), encoding="utf-8")

    with pytest.raises(AnnotationPackError, match="data_local"):
        build_annotation_pack(
            package,
            rt_manifest,
            hold_manifest,
            tmp_path / "data_local" / "derived_annotations",
        )
