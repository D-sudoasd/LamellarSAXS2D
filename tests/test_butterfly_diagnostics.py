import math
import numpy as np
import pytest

from butterfly_saxs.butterfly_diagnostics import ellipse_local_views, profile_residuals


def test_local_coordinates_preserve_branch_mapping_and_physical_values():
    angle = math.radians(-25.)
    u, v = .6, .02
    x, y = math.cos(angle)*u - math.sin(angle)*v, math.sin(angle)*u + math.cos(angle)*v
    point = {"point_id": "one", "qx": x, "qy": y, "branch_id": 0, "side": "upper", "accepted": True}
    candidate = {"a": 1., "b": .03, "theta_deg": 30., "reference_axis_deg": 5., "branch_swap_applied": True}
    result = ellipse_local_views([point], candidate)["0"]
    assert result["points"][0]["u"] == pytest.approx(u)
    assert result["points"][0]["v"] == pytest.approx(v)
    assert result["source"] == "candidate_geometry"
    assert "magnification" not in result


def test_profile_residuals_are_raw_minus_fit_with_masked_gap():
    profiles = {"p": {"raw_intensity": [3., np.nan, 5.], "fit_intensity": [2., np.nan, 4.]}}
    profile_residuals(profiles)
    np.testing.assert_allclose(profiles["p"]["residual"], [1., np.nan, 1.], equal_nan=True)


def test_no_geometry_does_not_invent_local_axes():
    assert ellipse_local_views([], {"a": None}) == {}
