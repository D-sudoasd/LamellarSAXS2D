from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from butterfly_saxs.lamellar_export import (
    LamellarExportCancelled,
    export_lamellar_scene,
    export_lamellar_sequence,
)
from butterfly_saxs.lamellar import build_lamellar_scene


def _scene(*, frame_id: str = "frame-A", status: str = "candidate") -> SimpleNamespace:
    center = np.asarray([0.0, 0.0, 0.0])
    half = np.asarray([0.5, 1.0, 0.15])
    signs = [(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1), (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)]
    vertices = np.asarray([center + np.asarray(sign) * half for sign in signs])[None, :, :]
    return SimpleNamespace(
        centers=np.asarray([center]),
        sizes=np.asarray([half * 2.0]),
        orientations=np.eye(3)[None, :, :],
        vertices=vertices,
        stack_ids=np.asarray([0]),
        branch_ids=np.asarray([1]),
        colors=np.asarray([[0.92, 0.47, 0.18, 0.84]]),
        bounds=np.asarray([[-1.0, -1.5, -1.0], [1.0, 1.5, 1.0]]),
        length_unit="relative",
        metadata={"frame_id": frame_id, "frame_selector": 7},
        status=status,
        message="",
        source_identity="source-A",
        assumptions=["orientation is schematic"],
        settings={"period": 2.0},
        parameter_sources=[{"name": "period", "value": 2.0, "unit": "relative", "status": "candidate"}],
        populations=[(1, 2.0, 15.0, "candidate")],
        scientific_boundary="candidate only",
        draw_axis_deg=90.0,
        reference_period=2.0,
    )


def test_scene_export_is_complete_hashed_and_pickle_free(tmp_path: Path) -> None:
    output = export_lamellar_scene(_scene(), tmp_path / "scene")

    expected = {"2d_svg", "2d_pdf", "3d_png", "combined_png", "settings", "provenance", "parameter_sources", "scene_arrays", "manifest"}
    assert expected == set(output)
    assert all(path.is_file() and path.stat().st_size > 0 for path in output.values())
    manifest = json.loads(output["manifest"].read_text(encoding="utf-8"))
    assert manifest["scene_status"] == "candidate"
    assert manifest["file_hashes"]["scene_arrays.npz"]
    with np.load(output["scene_arrays"], allow_pickle=False) as arrays:
        np.testing.assert_allclose(arrays["vertices"], _scene().vertices)
        assert "parameter_sources" not in arrays.files
    assert "候选参数驱动示意" in output["provenance"].read_text(encoding="utf-8") or "candidate" in output["provenance"].read_text(encoding="utf-8")


def test_supplied_rgba_capture_keeps_native_png_dimensions(tmp_path: Path) -> None:
    capture = np.zeros((37, 53, 4), dtype=np.uint8)
    capture[..., 0] = 120
    capture[..., 3] = 255
    output = export_lamellar_scene(_scene(), tmp_path / "native-capture", image_3d=capture, language="en")

    from PIL import Image

    with Image.open(output["3d_png"]) as image:
        assert image.size == (53, 37)
        assert image.mode in {"RGB", "RGBA"}


def test_single_export_rejects_stale_and_does_not_overwrite(tmp_path: Path) -> None:
    target = tmp_path / "scene"
    target.mkdir()
    sentinel = target / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        export_lamellar_scene(_scene(), target)
    assert sentinel.read_text(encoding="utf-8") == "keep"
    with pytest.raises(ValueError, match="stale"):
        export_lamellar_scene(_scene(status="stale"), tmp_path / "stale")
    assert not (tmp_path / "stale").exists()


def test_sequence_preserves_missing_slot_and_cleans_cancelled_stage(tmp_path: Path) -> None:
    target = tmp_path / "sequence"
    outputs = export_lamellar_sequence([_scene(frame_id="A"), None, _scene(frame_id="C")], target, fps=4.0)

    manifest = json.loads(outputs["manifest"].read_text(encoding="utf-8"))
    assert manifest["frame_count"] == 3
    assert [item["frame_id"] for item in manifest["frames"]] == ["A", 1, "C"]
    assert manifest["frames"][1]["missing"] is True
    assert outputs["frame_0001"].is_file()
    from PIL import Image

    with Image.open(outputs["gif"]) as gif:
        assert gif.n_frames == 3

    event = SimpleNamespace(is_set=lambda: True)
    cancelled_target = tmp_path / "cancelled"
    with pytest.raises(LamellarExportCancelled):
        export_lamellar_sequence([_scene()], cancelled_target, cancel_event=event)
    assert not cancelled_target.exists()


def test_core_sequence_manifest_preserves_nested_identity_and_scene_provenance(tmp_path: Path) -> None:
    source = {
        "source_identity": {"source": "stack.cbf", "frame": 110, "dataset": "entry/data"},
        "q_unit": "nm^-1",
        "lobe_radial_peaks": [
            {"angle_deg": 30.0, "q_star": 0.2, "q_unit": "nm^-1", "valid": True, "branch_id": 0},
            {"angle_deg": 210.0, "q_star": 0.2, "q_unit": "nm^-1", "valid": True, "branch_id": 0},
            {"angle_deg": 150.0, "q_star": 0.1, "q_unit": "nm^-1", "valid": True, "branch_id": 1},
            {"angle_deg": 330.0, "q_star": 0.1, "q_unit": "nm^-1", "valid": True, "branch_id": 1},
        ],
        "ellipse_fit": {"b": {"candidate_value": 0.1, "status": "candidate"}, "center": [0.0, 0.0], "q_unit": "nm^-1"},
    }
    scene = build_lamellar_scene(source, {"period_source": "ellipse", "layer_count": 1, "stack_count": 1})
    camera = {"yaw": -35., "pitch": -25., "ortho_height": 35.}
    output = export_lamellar_sequence([scene], tmp_path / "core-sequence", images=[None], camera=camera)

    manifest = json.loads(output["manifest"].read_text(encoding="utf-8"))
    frame = manifest["frames"][0]
    assert manifest["camera"] == camera
    assert frame["sequence_index"] == 0
    assert frame["frame_id"] == 110
    assert frame["frame_selector"] == 110
    assert frame["source_identity"]["dataset"] == "entry/data"
    assert frame["known_missing_image"] is True
    payload = frame["scene"]
    assert payload["assumptions"]
    assert payload["settings"]["period_source"] == "ellipse"
    assert payload["scientific_boundary"]
    assert any(row.get("candidate_value") == pytest.approx(0.1) for row in payload["parameter_sources"])
    csv_text = output["parameter_sources"].read_text(encoding="utf-8")
    assert "sequence_index" in csv_text
    assert "candidate_value" in csv_text
    assert "unit" in csv_text


def test_sequence_loader_error_propagates_and_unpublished_stage_is_removed(tmp_path: Path) -> None:
    class FailingLoader:
        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: int) -> object:
            raise OSError(f"cannot read frame {index}")

    target = tmp_path / "loader-failure"
    with pytest.raises(OSError, match="cannot read frame 0"):
        export_lamellar_sequence([_scene()], target, images=FailingLoader())
    assert not target.exists()
