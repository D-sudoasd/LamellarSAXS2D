"""An incompatible shape prior remains a bound hit, not an accuracy claim."""

from butterfly_saxs.benchmark_arcs import generate_arc_case
from butterfly_saxs.pipeline import analyze_frame


def test_default_flat_prior_stays_bounded_for_a_broader_ellipse():
    case = generate_arc_case("ellipse_ratio_400", seed=20260906, shape=(96, 96))
    result = analyze_frame(
        case["image"], qmap=case["qmap"], mask=case["mask"], full2d=False,
        config={"analysis": {
            "ridge_method": "butterfly_curvature", "q_window": [0.05, 1.1],
            "ellipse_multistart": 1,
            "butterfly": {"stage": "evaluate", "resamples": 0, "sensitivity": False},
        }},
    )
    assert result.analysis["ellipse"]["preset"] == "flat_ellipse"
    assert result.ellipse_fit["axis_ratio"] <= 0.35
    assert result.ellipse_fit["bound_flags"]["axis_ratio"]
    for name in ("a", "b", "axis_ratio", "theta_deg"):
        assert result.butterfly["quantitative_parameters"][name]["value"] is None
