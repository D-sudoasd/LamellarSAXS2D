from __future__ import annotations

import math

import pytest

from butterfly_saxs.butterfly_quality import evaluate_arc_evidence


def _observed_long_wing_points() -> list[dict]:
    points = []
    for branch, sign in ((0, 1.0), (1, -1.0)):
        for side, side_sign in (("upper", 1.0), ("lower", -1.0)):
            arc_id = branch * 2 + (side == "lower")
            for index, radius in enumerate((0.36, 0.45, 0.55)):
                point_id = f"{branch}:{side}:{index}"
                points.append(
                    {
                        "point_id": point_id,
                        "arc_id": arc_id,
                        "accepted": True,
                        "valid": True,
                        "branch_id": branch,
                        "side": side,
                        "qx": sign * radius * math.cos(math.radians(6.0)),
                        "qy": side_sign * radius * math.sin(math.radians(6.0)),
                        "localization_sigma_q": 0.001,
                    }
                )
    return points


def _candidate(points: list[dict], *, a: float) -> dict:
    return {
        "success": True,
        "a": a,
        "b": 0.12,
        "axis_ratio": 0.12 / a,
        "theta_deg": 0.0,
        "center_qx": 0.0,
        "center_qy": 0.0,
        "condition": 2.0,
        "rmse": 0.001,
        "point_diagnostics": [
            {"point_id": point["point_id"], "used": True} for point in points
        ],
    }


def test_observed_long_wing_is_counted_beyond_first_order_hint() -> None:
    points = _observed_long_wing_points()
    trace = {
        "points": points,
        "diagnostics": {
            "first_order_q_hint": {
                "q_star": 0.092,
                "selection_status": "ambiguous",
                "candidate_peaks": [{"q": 0.092}, {"q": 0.40}],
            }
        },
    }

    result = evaluate_arc_evidence(trace, _candidate(points, a=0.55))

    assert result["quality"]["metrics"]["observed_q_extent"] == pytest.approx(0.55, abs=1e-12)
    assert "major_axis_exceeds_observed_extent" not in result["quality"]["flags"]
    assert "first_order_q_hint_ambiguous" in result["quality"]["flags"]
    assert result["quantitative_parameters"]["a"]["candidate_value"] == pytest.approx(0.55)


def test_genuinely_extrapolated_major_axis_warns_without_discarding_candidate() -> None:
    points = _observed_long_wing_points()
    trace = {
        "points": points,
        "diagnostics": {"first_order_q_hint": {"q_star": 0.092, "selection_status": "selected"}},
    }

    within_localization_scale = evaluate_arc_evidence(trace, _candidate(points, a=0.552))
    assert "major_axis_exceeds_observed_extent" not in within_localization_scale["quality"]["flags"]

    result = evaluate_arc_evidence(trace, _candidate(points, a=0.80))

    assert "major_axis_exceeds_observed_extent" in result["quality"]["flags"]
    assert result["quantitative_parameters"]["a"]["candidate_value"] == pytest.approx(0.80)


def test_extent_metrics_keep_origin_and_candidate_center_references_explicit() -> None:
    points = _observed_long_wing_points()
    center = (0.02, -0.01)
    for point in points:
        point["qx"] += center[0]
        point["qy"] += center[1]
    candidate = _candidate(points, a=0.55)
    candidate.update(center_qx=center[0], center_qy=center[1])

    result = evaluate_arc_evidence({"points": points}, candidate)
    metrics = result["quality"]["metrics"]

    assert metrics["observed_q_extent"] > 0.55
    assert metrics["candidate_centered_q_extent"] == pytest.approx(0.55, abs=1e-12)
    assert "major_axis_exceeds_observed_extent" not in result["quality"]["flags"]


def test_q_extent_uses_only_fit_used_points() -> None:
    points = _observed_long_wing_points()
    rejected_from_fit = dict(points[0], point_id="not-used", qx=1.2, qy=0.0)
    points.append(rejected_from_fit)
    candidate = _candidate(points[:-1], a=0.55)
    candidate["point_diagnostics"].append({"point_id": "not-used", "used": False})

    result = evaluate_arc_evidence({"points": points}, candidate)

    assert result["quality"]["metrics"]["observed_q_extent"] == pytest.approx(0.55, abs=1e-12)


def test_ambiguous_radial_populations_limit_quality_without_discarding_fit() -> None:
    points = _observed_long_wing_points()
    trace = {
        "points": points,
        "diagnostics": {
            "radial_population": {
                "split": True,
                "selection_status": "ambiguous",
                "reason": "q_hint_unavailable",
                "keep": "all_observed",
                "low_q_population": {"q_min": 0.10, "q_max": 0.12, "n_points": 8},
                "high_q_population": {"q_min": 0.40, "q_max": 0.55, "n_points": 10},
                "flags": ["mixed_radial_populations"],
            }
        },
    }

    result = evaluate_arc_evidence(trace, _candidate(points, a=0.55))

    assert "mixed_radial_populations" in result["quality"]["flags"]
    assert result["quantitative_parameters"]["a"]["candidate_value"] == pytest.approx(0.55)
    assert "mixed_radial_populations" in result["quantitative_parameters"]["a"]["reasons"]
