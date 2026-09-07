from __future__ import annotations

import numpy as np
import pytest

from butterfly_saxs.ui.lamellar_state import FrameImageReader, compact_source, source_images


def test_native_array_keys_select_the_actual_physical_frame(tmp_path):
    path = tmp_path / "results.npz"
    np.savez(path, frame_0001__image=np.full((2, 3), 17),
             frame_0001__qx=np.ones((2, 3)), frame_0001__qy=np.zeros((2, 3)))
    source = {"npz_path": str(path), "array_keys": {"observed": "frame_0001__image", "qx": "frame_0001__qx", "qy": "frame_0001__qy"},
              "source_identity": {"source": "stack.h5", "frame": 8, "dataset": "saxs"}}
    restored = compact_source(source)
    image, qx, qy = source_images(restored)
    assert np.all(image == 17)
    assert qx.shape == qy.shape == image.shape
    assert restored["source_identity"]["frame"] == 8


def test_mismatched_calibrated_map_does_not_silently_become_pixel_q(tmp_path):
    path = tmp_path / "wrong.npz"
    np.savez(path, observed=np.ones((4, 5)), qx=np.ones((3, 5)), qy=np.ones((3, 5)))
    with pytest.raises(ValueError, match="q maps"):
        source_images({"npz_path": str(path)})
    reader = FrameImageReader([{"npz_path": str(path)}])
    with pytest.raises(ValueError, match="q maps"):
        _ = reader.view()[0]


def test_sequence_images_keep_bounded_cache():
    sources = [{"observed": np.full((4, 4), i)} for i in range(10)]
    reader = FrameImageReader(sources)
    images, qmaps = reader.view(), reader.view(qmaps=True)
    for index in range(10):
        assert np.all(images[index] == index)
        assert qmaps[index] == (None, None)
        assert len(reader.cache) <= 3
