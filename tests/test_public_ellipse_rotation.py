"""Public periods and members use the same angular reference as the solver."""

import math

import pytest

from butterfly_saxs.public_ellipse import canonical_ellipse_payload


@pytest.mark.parametrize("reference", [-40.0, 30.0, 90.0])
def test_rotating_reference_and_draw_axes_preserves_draw_period(reference):
    fit = {"success": True, "a": 0.8, "b": 0.24, "axis_ratio": 0.3,
           "theta_deg": 17.0, "center": (0.0, 0.0), "q_unit": "nm^-1"}
    original = canonical_ellipse_payload({**fit, "reference_axis_deg": 0.0})
    rotated = canonical_ellipse_payload({**fit, "reference_axis_deg": reference})
    # In the reference frame the draw axis is always 90 degrees from +qx.
    angle = math.radians(90.0 - 17.0)
    q_draw = 0.8 * 0.24 / math.hypot(0.24 * math.cos(angle), 0.8 * math.sin(angle))
    assert rotated["Lz_from_draw_axis_nm"] == pytest.approx(2.0 * math.pi / q_draw)
    assert rotated["Lz_from_draw_axis_nm"] == pytest.approx(original["Lz_from_draw_axis_nm"])
    assert [member["theta_deg"] for member in rotated["ellipses"]] == pytest.approx(
        [reference + 17.0, reference - 17.0]
    )
