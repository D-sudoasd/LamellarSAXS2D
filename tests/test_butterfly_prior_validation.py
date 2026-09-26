"""The standard butterfly fit derives geometry from the observed data."""

import pytest

from butterfly_saxs.benchmark_arcs import generate_arc_case
from butterfly_saxs.pipeline import analyze_frame


def test_standard_butterfly_fit_can_recover_a_broader_ellipse():
    case = generate_arc_case("ellipse_ratio_400", seed=20260906, shape=(96, 96))
    result = analyze_frame(
        case["image"], qmap=case["qmap"], mask=case["mask"], full2d=False,
        config={"analysis": {
            "ridge_method": "butterfly_curvature", "q_window": [0.05, 1.1],
            "ellipse_multistart": 1,
            "butterfly": {"stage": "evaluate", "resamples": 0, "sensitivity": False},
        }},
    )
    assert result.analysis["ellipse"]["preset"] == "standard"
    assert result.ellipse_fit["success"] is True
    assert result.ellipse_fit["axis_ratio"] == pytest.approx(
        case["truth"]["axis_ratio"], rel=0.2
    )
    assert result.ellipse_fit["bound_flags"]["axis_ratio"] is False
