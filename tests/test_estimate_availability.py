"""Numerical estimates remain usable before optional confidence assessment."""

from __future__ import annotations

import pytest

from butterfly_saxs.butterfly_quality import classify_ellipse_publication, evaluate_arc_evidence


def _evidence():
    points = [
        {"accepted": True, "valid": True, "branch_id": branch, "side": side,
         "qx": (0.4 + index * 0.1) * (1 if branch == 0 else -1),
         "qy": 0.1 * (1 if side == "upper" else -1),
         "localization_sigma_q": 0.01}
        for branch in (0, 1) for side in ("upper", "lower") for index in range(4)
    ]
    arcs = [
        {"arc_id": index, "support_status": "frozen", "support_component_count": 1,
         "normal_residual_q_rms": 0.002, "mirror_symmetry_status": "observed"}
        for index in range(4)
    ]
    candidate = {"success": True, "a": 0.7, "b": 0.21, "axis_ratio": 0.3,
                 "theta_deg": 17.0, "condition": 3.0, "rmse": 0.002, "q_unit": "nm^-1"}
    return {"points": points, "arc_diagnostics": arcs}, candidate


def test_optional_uncertainty_does_not_erase_estimates_or_disable_initialization():
    trace, candidate = _evidence()
    result = evaluate_arc_evidence(trace, candidate)
    assert result["warm_start_eligible"] is True
    for name, row in result["quantitative_parameters"].items():
        assert row["value"] == pytest.approx(candidate[name])
        assert row["status"] == "estimate"
        assert row["publication_status"] == "not_assessed"
        assert row["confidence"] == "empirical"
        assert row["interval"] is None
        assert "image_resampling_interval_unavailable" in row["reasons"]


def test_partial_arcs_keep_candidate_values_and_explain_support():
    trace, candidate = _evidence()
    trace["points"] = [p for p in trace["points"] if p["branch_id"] == 0]
    result = evaluate_arc_evidence(trace, candidate)
    assert result["quantitative_parameters"]["a"]["value"] == pytest.approx(0.7)
    assert result["quantitative_parameters"]["a"]["status"] == "candidate"
    assert "insufficient_occupied_sides" in result["quality"]["flags"]
    assert result["warm_start_eligible"] is False


@pytest.mark.parametrize("failure", ["solver", "no_points", "invalid_points", "nonfinite_points", "degenerate"])
def test_unusable_fit_does_not_become_an_estimate_or_a_seed(failure):
    trace, candidate = _evidence()
    if failure == "solver":
        candidate["success"] = False
    elif failure == "no_points":
        trace["points"] = []
    elif failure == "invalid_points":
        for point in trace["points"]:
            point["valid"] = False
    elif failure == "nonfinite_points":
        for point in trace["points"]:
            point["qx"] = float("nan")
    else:
        candidate["b"] = 0.0
    result = evaluate_arc_evidence(trace, candidate)
    assert result["warm_start_eligible"] is False
    assert all(row["value"] is None for row in result["quantitative_parameters"].values())


def test_boundary_estimate_is_retained_without_certifying_it():
    trace, candidate = _evidence()
    candidate["bound_flags"] = {"axis_ratio": True}
    result = evaluate_arc_evidence(trace, candidate)
    ratio = result["quantitative_parameters"]["axis_ratio"]
    assert ratio["value"] == pytest.approx(0.3)
    assert ratio["status"] == "candidate"
    assert ratio["publication_status"] == "not_assessed"
    assert "parameter_at_explicit_bound" in ratio["reasons"]


def test_rejected_background_diagnostics_are_not_an_extra_manual_arc():
    trace, candidate = _evidence()
    candidate["arc_diagnostics"] = [*trace["arc_diagnostics"],
        {"arc_id": -1, "n_used": 0, "n_excluded": 120, "support_status": "manual_only"}]
    result = evaluate_arc_evidence(trace, candidate)
    assert result["quality"]["metrics"]["arc_support"]["arc_count"] == 4
    assert "manual_arc_bounds_not_experimental_evidence" not in result["quality"]["flags"]
    assert result["quantitative_parameters"]["a"]["status"] == "estimate"


def test_quality_uses_post_fit_residual_diagnostics_when_available():
    trace, candidate = _evidence()
    candidate["arc_diagnostics"] = [dict(arc) for arc in trace["arc_diagnostics"]]
    for arc in trace["arc_diagnostics"]:
        arc.pop("normal_residual_q_rms")
    result = evaluate_arc_evidence(trace, candidate)
    assert "per_arc_normal_residual_evidence_unavailable" not in result["quality"]["flags"]


def test_ring_shape_diagnosis_survives_insufficient_butterfly_side_support():
    # Failing a double-ellipse quality screen does not erase the ring diagnosis.
    assert classify_ellipse_publication(quality_status="FAIL", axis_ratio=0.99) == "ring"
    assert classify_ellipse_publication(quality_status="FAIL", axis_ratio=0.3) == "fail"
    assert classify_ellipse_publication(quality_status="FAIL") == "fail"


def test_unassigned_candidates_do_not_supply_missing_fitted_sides():
    trace, candidate = _evidence()
    for point in trace["points"]:
        point["arc_id"] = 0 if point["branch_id"] == 0 else -1
    result = evaluate_arc_evidence(trace, candidate)
    assert result["warm_start_eligible"] is False
    assert result["measurement_status"] == "undetermined"
    assert result["quantitative_parameters"]["a"]["confidence"] == "limited"
    assert result["quantitative_parameters"]["a"]["value"] == pytest.approx(0.7)
    assert "insufficient_occupied_sides" in result["quality"]["flags"]


def test_post_fit_excluded_points_cannot_inflate_side_support():
    trace, candidate = _evidence()
    candidate["point_diagnostics"] = [
        {"point_id": index, "used": point["branch_id"] == 0}
        for index, point in enumerate(trace["points"])
    ]
    result = evaluate_arc_evidence(trace, candidate)
    assert result["quality"]["metrics"]["occupied_sides"] == 2
    assert result["warm_start_eligible"] is False
    assert result["quantitative_parameters"]["a"]["value"] == pytest.approx(0.7)
    assert result["quantitative_parameters"]["a"]["confidence"] == "limited"


def test_ambiguous_trajectory_keeps_values_with_limited_confidence():
    trace, candidate = _evidence()
    trace["points"][0]["trajectory_confidence"] = "low"
    trace["points"][0]["trajectory_ambiguous"] = True
    result = evaluate_arc_evidence(trace, candidate)
    assert result["quantitative_parameters"]["a"]["value"] == pytest.approx(0.7)
    assert result["quantitative_parameters"]["a"]["status"] == "candidate"
    assert "annular_trajectory_support_limited" in result["quality"]["flags"]
    assert result["warm_start_eligible"] is False
