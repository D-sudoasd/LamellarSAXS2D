from __future__ import annotations

import numpy as np
import pytest

from butterfly_saxs.geometry import build_geometry
from butterfly_saxs.io import (
    DataShapeError,
    FrameSelectionError,
    combine_masks,
    load_image,
)


def test_npy_preserves_float_dtype_and_values(tmp_path):
    source = tmp_path / "frame.npy"
    values = np.array([[0.25, 2.5], [100.0, -3.0]], dtype=np.float32)
    np.save(source, values)

    loaded = load_image(source)

    assert loaded.data.dtype == values.dtype
    np.testing.assert_array_equal(loaded.data, values)
    assert loaded.preserves_absolute_intensity
    assert loaded.metadata["intensity_semantics"].startswith("source values unchanged")


def test_tiff_preserves_float_dtype_and_values(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    source = tmp_path / "frame.tiff"
    values = np.arange(12, dtype=np.float64).reshape(3, 4) / 7.0
    tifffile.imwrite(source, values)

    loaded = load_image(source)

    assert loaded.data.dtype == values.dtype
    np.testing.assert_array_equal(loaded.data, values)
    assert loaded.metadata["format"] == "tiff"


def test_fabio_edf_is_read_with_header(tmp_path):
    pytest.importorskip("fabio")
    from fabio.edfimage import EdfImage

    source = tmp_path / "frame.edf"
    values = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32)
    EdfImage(data=values, header={"Exposure_time": "0.5"}).write(str(source))

    loaded = load_image(source)

    assert loaded.data.dtype == values.dtype
    np.testing.assert_array_equal(loaded.data, values)
    assert loaded.metadata["header"]["Exposure_time"] == "0.5"


def test_fabio_cbf_is_read_without_intensity_rescaling(tmp_path):
    pytest.importorskip("fabio")
    from fabio.cbfimage import CbfImage

    source = tmp_path / "frame.cbf"
    values = np.arange(30, dtype=np.int32).reshape(5, 6) * 17
    CbfImage(data=values, header={"Exposure_time": "1.25"}).write(str(source))

    loaded = load_image(source)

    np.testing.assert_array_equal(loaded.data, values)
    assert loaded.metadata["format"] == "cbf"
    assert loaded.metadata["absolute_intensity_preserved"] is True


def test_unicode_paths_work_for_cbf_and_poni(tmp_path):
    """Windows beamline data commonly lives below Chinese-named folders."""

    pytest.importorskip("fabio")
    pytest.importorskip("pyFAI")
    from fabio.cbfimage import CbfImage
    from pyFAI.azimuthalIntegrator import AzimuthalIntegrator

    folder = tmp_path / "原位实验 数据"
    folder.mkdir()
    image_path = folder / "帧 0001.cbf"
    poni_path = folder / "几何标定.poni"
    values = np.arange(42, dtype=np.int32).reshape(6, 7)
    # FabIO's CBF *writer* embeds the filename in an ASCII CIF header, so
    # create an ASCII fixture and then move it.  The application is a reader
    # of existing beamline CBF files; that is the Unicode path under test.
    ascii_image = tmp_path / "frame.cbf"
    CbfImage(data=values).write(str(ascii_image))
    ascii_image.replace(image_path)
    integrator = AzimuthalIntegrator(
        dist=0.12,
        poni1=0.0003,
        poni2=0.00035,
        pixel1=0.0001,
        pixel2=0.0001,
        wavelength=1.0e-10,
    )
    ascii_poni = tmp_path / "geometry.poni"
    integrator.save(str(ascii_poni))
    ascii_poni.replace(poni_path)

    loaded = load_image(image_path)
    maps = build_geometry(values.shape, poni_path, valid_mask=loaded.valid_mask)

    np.testing.assert_array_equal(loaded.data, values)
    assert maps.shape == values.shape
    assert maps.metadata["q_unit"] == "nm^-1"


def test_multiframe_npy_requires_explicit_frame(tmp_path):
    source = tmp_path / "stack.npy"
    values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    np.save(source, values)

    with pytest.raises(FrameSelectionError, match="explicit frame"):
        load_image(source)
    loaded = load_image(source, frame=1)
    np.testing.assert_array_equal(loaded.data, values[1])
    assert loaded.frame == 1


def test_shape_and_mask_polarity_are_strict(tmp_path):
    source = tmp_path / "frame.npy"
    np.save(source, np.ones((2, 3), dtype=np.float64))
    with pytest.raises(FrameSelectionError):
        load_image(source, frame=1)
    with pytest.raises(DataShapeError, match="does not exactly match"):
        combine_masks((2, 3), external_mask=np.ones((2, 2), dtype=bool))

    valid = np.array([[True, False, True], [True, True, False]])
    rejected = np.array([[False, True, False], [False, False, False]])
    combined = combine_masks((2, 3), valid_mask=valid, external_mask=rejected)
    np.testing.assert_array_equal(combined, valid & ~rejected)


def test_simple_poni_geometry_maps_match_pyfai_and_shape(tmp_path):
    pytest.importorskip("pyFAI")
    from pyFAI.azimuthalIntegrator import AzimuthalIntegrator

    source = tmp_path / "geometry.poni"
    integrator = AzimuthalIntegrator(
        dist=0.10,
        poni1=0.00015,
        poni2=0.00015,
        rot1=0.01,
        rot2=-0.02,
        rot3=0.03,
        pixel1=0.0001,
        pixel2=0.0001,
        wavelength=1.0e-10,
    )
    integrator.save(str(source))

    maps = build_geometry((3, 3), source)
    expected_q = integrator.qArray((3, 3))
    expected_chi = np.mod(integrator.center_array((3, 3), unit="chi_rad"), 2 * np.pi)
    np.testing.assert_allclose(maps.q_nm_inv, expected_q)
    np.testing.assert_allclose(maps.chi_rad, expected_chi)
    np.testing.assert_allclose(maps.qx_nm_inv, maps.q_nm_inv * np.cos(maps.chi_rad))
    np.testing.assert_allclose(maps.qy_nm_inv, maps.q_nm_inv * np.sin(maps.chi_rad))
    assert maps.q_nm_inv.shape == maps.chi_rad.shape == maps.qx_nm_inv.shape == (3, 3)
    assert len(maps.fingerprint) == 64
    assert maps.metadata["q_unit"] == "nm^-1"
    assert maps.metadata["chi_zero_deg"] == "+qx"
    assert maps.metadata["chi_ninety_deg"] == "+qy"
