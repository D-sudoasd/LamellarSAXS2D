from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from butterfly_saxs.batch import FrameFitResult, FrameRef
from butterfly_saxs.export import export_batch


def test_npz_completeness_lists_missing_frames_and_keeps_success_arrays(
    tmp_path: Path,
) -> None:
    successful_path = tmp_path / "successful-frame.tif"
    failed_path = tmp_path / "failed-frame.tif"
    resultless_path = tmp_path / "resultless-frame.tif"
    successful = FrameFitResult(
        frame=FrameRef(successful_path, frame_id="ok"),
        result={"image": np.arange(6, dtype=float).reshape(2, 3)},
    )
    failed = FrameFitResult(
        frame=FrameRef(failed_path, frame_id="failed"),
        result={"partial_image": np.ones((2, 2), dtype=float)},
        status="failed",
        error="fit failed",
    )
    resultless = FrameFitResult(
        frame=FrameRef(resultless_path, frame_id="missing"),
        result=None,
    )

    outputs = export_batch([successful, failed, resultless], tmp_path / "exports")

    with np.load(outputs["npz"], allow_pickle=False) as arrays:
        metadata = json.loads(str(arrays["__metadata__"]))
        np.testing.assert_array_equal(
            arrays["frame_0000__image"],
            np.arange(6, dtype=float).reshape(2, 3),
        )
    assert metadata == {
        "arrays": [
            "frame_0000__image",
            "frame_0001__partial_image",
        ],
        "frame_count": 3,
        "complete": False,
        "missing_frames": [1, 2],
        "missing_frame_ids": ["failed", "missing"],
        "missing_frame_paths": [str(failed_path), str(resultless_path)],
    }
