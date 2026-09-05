from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from butterfly_saxs.batch import FrameRef, build_frame_refs
from butterfly_saxs.io import FrameSelectionError, load_image
from butterfly_saxs.pipeline import inspect_frame


def test_hdf5_image_does_not_forward_image_dataset_to_npy_mask(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    image_path = tmp_path / "image.h5"
    mask_path = tmp_path / "mask.npy"
    image = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    mask = np.zeros((3, 4), dtype=np.uint8)
    mask[0, 1] = 1
    with h5py.File(image_path, "w") as handle:
        handle.create_dataset("entry/data", data=image)
    np.save(mask_path, mask)

    loaded = load_image(
        image_path,
        frame=1,
        dataset="entry/data",
        external_mask=mask_path,
    )

    np.testing.assert_array_equal(loaded.data, image[1])
    assert loaded.valid_mask is not None
    assert not loaded.valid_mask[0, 1]


def test_hdf5_image_and_hdf5_mask_use_independent_selectors(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    image_path = tmp_path / "image.h5"
    mask_path = tmp_path / "mask.h5"
    image = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    masks = np.zeros((2, 3, 4), dtype=np.uint8)
    masks[0, 1, 2] = 1
    masks[1, 2, 3] = 1
    with h5py.File(image_path, "w") as handle:
        handle.create_dataset("entry/data", data=image)
    with h5py.File(mask_path, "w") as handle:
        handle.create_dataset("mask/series", data=masks)

    loaded = load_image(
        image_path,
        frame=1,
        dataset="entry/data",
        external_mask=mask_path,
        mask_frame=0,
        mask_dataset="mask/series",
    )

    np.testing.assert_array_equal(loaded.data, image[1])
    assert loaded.valid_mask is not None
    assert not loaded.valid_mask[1, 2]
    assert loaded.valid_mask[2, 3]


def test_pipeline_forwards_mask_selector_to_positive_valid_mask(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    image_path = tmp_path / "image.h5"
    mask_path = tmp_path / "valid.h5"
    images = np.ones((2, 16, 16), dtype=np.float32)
    valid_masks = np.ones((2, 16, 16), dtype=np.uint8)
    valid_masks[0, 1, 2] = 0
    valid_masks[1, 3, 4] = 0
    with h5py.File(image_path, "w") as handle:
        handle.create_dataset("entry/data", data=images)
    with h5py.File(mask_path, "w") as handle:
        handle.create_dataset("mask/series", data=valid_masks)

    report = inspect_frame(
        image_path,
        frame=1,
        dataset="entry/data",
        valid_mask=mask_path,
        mask_frame=0,
        mask_dataset="mask/series",
    )

    assert report["analysis_domain"]["detector_valid_count"] == 255


def test_multiframe_mask_requires_explicit_mask_frame(tmp_path: Path) -> None:
    image_path = tmp_path / "image.npy"
    mask_path = tmp_path / "mask.npy"
    np.save(image_path, np.ones((2, 3), dtype=np.float32))
    masks = np.zeros((2, 2, 3), dtype=np.uint8)
    masks[1, 0, 2] = 1
    np.save(mask_path, masks)

    with pytest.raises(FrameSelectionError, match="explicit frame"):
        load_image(image_path, external_mask=mask_path)

    loaded = load_image(image_path, external_mask=mask_path, mask_frame=1)
    assert loaded.valid_mask is not None
    assert not loaded.valid_mask[0, 2]


def test_manifest_order_is_numeric_for_integers_leading_zeros_and_decimals(
    tmp_path: Path,
) -> None:
    paths = [tmp_path / f"frame_{name}.npy" for name in ("one", "two", "ten", "fraction")]
    manifest = [
        {"path": paths[2], "order": "010"},
        {"path": paths[1], "order": "002"},
        {"path": paths[0], "order": "001"},
        {"path": paths[3], "order": "1.5"},
    ]

    refs = build_frame_refs([], manifest=manifest)

    assert [ref.path for ref in refs] == [paths[0], paths[3], paths[1], paths[2]]
    assert [ref.order for ref in refs] == [1, 1.5, 2, 10]
    assert all(isinstance(ref.order, (int, float)) for ref in refs)


@pytest.mark.parametrize("order", ["not-a-number", "NaN", "Inf", np.nan, np.inf])
def test_manifest_and_frameref_reject_nonfinite_or_nonnumeric_order(
    tmp_path: Path, order: object
) -> None:
    path = tmp_path / "frame.npy"

    with pytest.raises(ValueError, match="finite number"):
        FrameRef(path, order=order)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="finite number"):
        build_frame_refs([], manifest=[{"path": path, "order": order}])


def test_manifest_without_order_uses_time_then_original_manifest_order(tmp_path: Path) -> None:
    paths = [tmp_path / f"frame_{index}.npy" for index in range(3)]
    timed = [
        {"path": paths[0], "time": 20.0},
        {"path": paths[1], "time": 10.0},
    ]
    assert [ref.path for ref in build_frame_refs([], manifest=timed)] == [
        paths[1],
        paths[0],
    ]

    original = [{"path": paths[2]}, {"path": paths[0]}, {"path": paths[1]}]
    refs = build_frame_refs([], manifest=original)
    assert [ref.path for ref in refs] == [paths[2], paths[0], paths[1]]


def test_manifest_blank_order_is_missing_and_uses_time_or_original_order(
    tmp_path: Path,
) -> None:
    paths = [tmp_path / f"frame_{index}.npy" for index in range(3)]
    timed = [
        {"path": paths[0], "order": "", "time": 20.0},
        {"path": paths[1], "order": "  ", "time": 10.0},
    ]
    assert [ref.path for ref in build_frame_refs([], manifest=timed)] == [
        paths[1],
        paths[0],
    ]

    original = [
        {"path": paths[2], "order": ""},
        {"path": paths[0], "order": ""},
        {"path": paths[1], "order": ""},
    ]
    refs = build_frame_refs([], manifest=original)
    assert [ref.path for ref in refs] == [paths[2], paths[0], paths[1]]
    assert all(ref.order is None for ref in refs)
