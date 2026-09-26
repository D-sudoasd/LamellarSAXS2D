from __future__ import annotations

import numpy as np

from scripts.validate_lamellar_sequence import (
    _expected_output_files,
    _validate_gui_demo_roundtrip,
)


def test_demo_output_collision_names_follow_masked_frame_selection() -> None:
    outputs = _expected_output_files(oblique_frames=3)

    assert "demo_frames/oblique_02.npz" in outputs
    assert "demo_frames/oblique_02_mask.npy" in outputs
    assert "demo_frames/oblique_02_truth.json" in outputs
    assert "demo_frames/oblique_04.npz" not in outputs


def test_demo_npz_roundtrip_preserves_image_mask_and_physical_q(tmp_path) -> None:
    shape = (16, 18)
    y, x = np.indices(shape, dtype=float)
    image = x + y
    qx = (x - (shape[1] - 1) / 2) * 0.02
    qy = (y - (shape[0] - 1) / 2) * 0.02
    mask = np.zeros(shape, dtype=bool)
    mask[3:5, 6:8] = True
    path = tmp_path / "oblique_00.npz"
    np.savez_compressed(
        path,
        image=image,
        qx=qx,
        qy=qy,
        q_unit=np.asarray("nm^-1"),
        mask=mask,
    )
    np.save(tmp_path / "oblique_00_mask.npy", mask, allow_pickle=False)

    result = _validate_gui_demo_roundtrip(path)

    assert result["status"] == "passed"
    assert result["q_unit"] == "nm^-1"
    assert result["masked_pixel_count"] == int(np.count_nonzero(mask))
