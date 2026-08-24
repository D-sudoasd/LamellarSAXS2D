from __future__ import annotations

import numpy as np
import pytest

from butterfly_saxs.masking import (
    MaskSpecError,
    combine_exclusion_masks,
    ellipse_mask,
    q_sector_mask,
    rectangle_mask,
)


def test_pixel_rois_use_true_means_excluded_and_combine_by_union():
    rectangle = rectangle_mask((20, 30), x0=2, x1=6, y0=4, y1=9)
    ellipse = ellipse_mask((20, 30), cx=15, cy=10, rx=3, ry=5, angle_deg=20)
    combined = combine_exclusion_masks((20, 30), masks=[rectangle], rois=[
        {"type": "ellipse", "cx": 15, "cy": 10, "rx": 3, "ry": 5, "angle_deg": 20}
    ])
    np.testing.assert_array_equal(combined, rectangle | ellipse)
    assert combined.dtype == bool
    assert combined[4, 2]
    assert combined[10, 15]


def test_q_sector_handles_wraparound_and_shape_errors():
    angle = np.deg2rad(np.array([[175.0, -175.0, 0.0]]))
    qx, qy = np.cos(angle), np.sin(angle)
    mask = q_sector_mask(qx, qy, q_min=0.5, q_max=1.5, chi_min_deg=170, chi_max_deg=-170)
    np.testing.assert_array_equal(mask, np.array([[True, True, False]]))
    with pytest.raises(MaskSpecError, match="does not match"):
        combine_exclusion_masks((3, 3), masks=[np.zeros((2, 2), dtype=bool)])
