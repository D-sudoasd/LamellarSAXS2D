from __future__ import annotations

import json
import math

import numpy as np
import pytest

from butterfly_saxs.arc_geometry import fit_arc_ellipses, project_arc_points
from butterfly_saxs.arc_support import SUPPORT_VERSION, freeze_observed_support
from butterfly_saxs.butterfly_diagnostics import arc_diagnostic_rows
from butterfly_saxs.butterfly_quality import evaluate_arc_evidence
from butterfly_saxs.ellipse import EllipseGeometry


def _fixed_parameters(a=1.5, ratio=0.1, theta=0.35):
    return {
        "cx": {"value": 0.0, "vary": False},
        "cy": {"value": 0.0, "vary": False},
        "a": {"value": a, "vary": False},
        "axis_ratio": {"value": ratio, "vary": False},
        "theta": {"value": theta, "vary": False},
    }


def _observed_points(values=(0.4, 0.5, 0.6), *, branch=0, arc_id=3):
    geometry = EllipseGeometry(0.0, 0.0, 1.5, 0.1, (1.0 if branch == 0 else -1.0) * 0.35)
    points = []
    arc = {"arc_id": arc_id, "ordered_point_ids": [], "valid": True}
    for index, t in enumerate(values):
        qx, qy = geometry.point(t)
        point_id = f"p-{index}"
        arc["ordered_point_ids"].append(point_id)
        points.append(
            {
                "point_id": point_id,
                "qx": float(qx),
                "qy": float(qy),
                "branch_id": branch,
                "side": "upper",
                "arc_id": arc_id,
                "accepted": True,
                "localization_sigma_q": 0.002,
                "sampling_sigma_q": 0.002,
                "q_normal_step": 0.004,
                "tangent_qx": float(-math.sin(t)),
                "tangent_qy": float(math.cos(t)),
                "normal_qx": float(math.cos(t)),
                "normal_qy": float(math.sin(t)),
            }
        )
    return points, [arc]


def test_freeze_observed_support_is_json_safe_and_keeps_observed_endpoints():
    points, arcs = _observed_points()
    envelope = freeze_observed_support(points, arcs)

    assert envelope["support_version"] == SUPPORT_VERSION
    assert envelope["support_status"] == "frozen"
    assert arcs[0]["support_gap_count"] == 0
    component = arcs[0]["support_components"][0]
    assert component["point_ids"] == ["p-0", "p-1", "p-2"]
    rectangle = component["rectangles"][0]
    assert rectangle["endpoint_flags"] == {"start": True, "end": False}
    assert all(point["observed_support"]["status"] == "frozen" for point in points)
    json.dumps(envelope)


@pytest.mark.parametrize("as_callable", [False, True])
def test_mid_freeze_uses_shared_cancellation_contract(as_callable):
    from butterfly_saxs.cancellation import AnalysisCancelled

    class CancelOnSecondCheck:
        calls = 0

        def is_set(self):
            self.calls += 1
            return self.calls >= 2

    event = CancelOnSecondCheck()
    points, arcs = _observed_points()
    token = event.is_set if as_callable else event
    with pytest.raises(AnalysisCancelled) as caught:
        freeze_observed_support(points, arcs, summary_only=True, cancel_event=token)
    assert caught.value.stage == "observed-support-freeze"
    assert event.calls == 2


def test_skew_measured_tangent_rectangle_contains_both_source_endpoints():
    points, arcs = _observed_points((0.4, 0.9))
    for point in points:
        # Deliberately perturb the supplied local tangent away from the chord.
        tx, ty = point["tangent_qx"], point["tangent_qy"]
        point["tangent_qx"] = 0.96 * tx - 0.28 * ty
        point["tangent_qy"] = 0.28 * tx + 0.96 * ty
    freeze_observed_support(points, arcs)
    rectangle = arcs[0]["support_components"][0]["rectangles"][0]
    origin = np.asarray(rectangle["frame_origin_q"], dtype=float)
    tangent = np.asarray(rectangle["tangent_q"], dtype=float)
    normal = np.asarray(rectangle["normal_q"], dtype=float)
    for point in points:
        delta = np.asarray([point["qx"], point["qy"]]) - origin
        t_value, n_value = float(delta @ tangent), float(delta @ normal)
        assert rectangle["tangent_bounds"][0] - 1.0e-12 <= t_value <= rectangle["tangent_bounds"][1] + 1.0e-12
        assert rectangle["normal_bounds"][0] - 1.0e-12 <= n_value <= rectangle["normal_bounds"][1] + 1.0e-12


def test_support_projection_stays_inside_observed_corridor_and_reports_q_units():
    points, arcs = _observed_points()
    freeze_observed_support(points, arcs)
    result = fit_arc_ellipses(points, parameters=_fixed_parameters(), multistart=1)

    assert result["fit"] is not None
    assert all(row["projection_valid"] for row in result["point_diagnostics"])
    assert all(row["support_infeasible"] is False for row in result["point_diagnostics"])
    assert result["fit"].coverage.radial_rms_definition == "unweighted_q_space_projection_distance"
    assert result["fit"].coverage.radial_rms_units == "q"
    assert result["fit"].coverage.radial_rms < 1.0e-10


def test_mirror_evidence_uses_fixed_source_labels_and_q_space_reflection():
    first, first_arcs = _observed_points((0.4, 0.5, 0.6), branch=0, arc_id=0)
    second, second_arcs = _observed_points(tuple(2.0 * math.pi - value for value in (0.4, 0.5, 0.6)), branch=1, arc_id=1)
    for point in second:
        point["side"] = "lower"
    points = first + second
    arcs = first_arcs + second_arcs
    for point in points:
        point["center_qx"] = 0.0
        point["center_qy"] = 0.0
    freeze_observed_support(points, arcs)
    result = fit_arc_ellipses(points, parameters=_fixed_parameters(), multistart=1, reference_center=(0.0, 0.0))
    assert all(row["mirror_symmetry_status"] == "observed" for row in result["arc_diagnostics"])
    assert all(row["mirror_pair_fraction"] == pytest.approx(1.0) for row in result["arc_diagnostics"])


def test_disconnected_declared_components_are_not_bridged():
    points, arcs = _observed_points((0.3, 0.4, 1.1, 1.2))
    for point in points[:2]:
        point["support_component_id"] = 0
    for point in points[2:]:
        point["support_component_id"] = 1
    envelope = freeze_observed_support(points, arcs)
    assert envelope["summary"]["n_support_components"] == 2
    assert envelope["summary"]["n_support_gaps"] == 1
    assert len(arcs[0]["support_components"]) == 2
    assert arcs[0]["support_gaps"][0]["reason"] == "caller_declared_disconnected_support"


def test_rejected_interior_observation_forces_a_frozen_support_gap():
    points = []
    for index in range(5):
        points.append(
            {
                "point_id": str(index),
                "qx": index * 0.01,
                "qy": 0.0,
                "branch_id": 0,
                "side": "upper",
                "arc_id": 9,
                "accepted": index != 2,
                "localization_sigma_q": 0.001,
                "sampling_sigma_q": 0.001,
                "q_normal_step": 0.001,
                "tangent_qx": 1.0,
                "tangent_qy": 0.0,
                "normal_qx": 0.0,
                "normal_qy": 1.0,
            }
        )
    arcs = [{"arc_id": 9, "ordered_point_ids": [str(index) for index in range(5)]}]
    freeze_observed_support(points, arcs, summary_only=True)
    assert arcs[0]["support_component_count"] == 2
    assert arcs[0]["support_gap_count"] == 1
    assert arcs[0]["support_gaps"][0]["reason"] == "rejected_or_ambiguous_interior_observation"


def test_infeasible_candidate_has_finite_penalty_and_no_projected_coordinates():
    point = {
        "point_id": "far",
        "qx": 1.8,
        "qy": 0.4,
        "branch_id": 0,
        "side": "upper",
        "arc_id": 0,
        "accepted": True,
        "localization_sigma_q": 1.0e-4,
        "observed_support": {
            "version": SUPPORT_VERSION,
            "status": "frozen",
            "components": [
                {
                    "component_id": 0,
                    "rectangles": [
                        {
                            "component_id": 0,
                            "segment_index": 0,
                            "point_ids": ["far"],
                            "frame_origin_q": [1.8, 0.4],
                            "tangent_q": [1.0, 0.0],
                            "normal_q": [0.0, 1.0],
                            "tangent_bounds": [-1.0e-4, 1.0e-4],
                            "normal_bounds": [-1.0e-4, 1.0e-4],
                        }
                    ],
                }
            ],
        },
    }
    result = fit_arc_ellipses(point and [point], parameters=_fixed_parameters(a=1.0, ratio=0.2), multistart=1)
    row = result["point_diagnostics"][0]
    assert result["fit"] is not None
    assert np.isfinite(result["fit"].cost)
    assert np.all(np.isfinite(result["fit"].residuals))
    assert row["projection_valid"] is False
    assert row["projection_t"] is None
    assert row["projected_qx"] is None and row["projected_qy"] is None
    assert np.isfinite(row["support_violation_q"])


def test_projection_only_helper_rejects_held_out_point_outside_frozen_support():
    points, arcs = _observed_points((0.4, 0.5, 0.6))
    freeze_observed_support(points, arcs)
    held_out_geometry = EllipseGeometry(0.0, 0.0, 1.5, 0.1, 0.35)
    qx, qy = held_out_geometry.point(1.2)
    held_out = dict(points[0])
    held_out.update({"qx": float(qx), "qy": float(qy), "accepted": True})
    result = project_arc_points([held_out], _fixed_parameters(), reference_axis_deg=0.0)
    row = result["point_diagnostics"][0]
    assert result["n_valid"] == 0
    assert result["n_invalid"] == 1
    assert row["projection_valid"] is False
    assert row["projection_reason"] == "observed_support_point_outside_corridor"
    assert row["projection_t"] is None
    assert row["projected_qx"] is None and row["projected_qy"] is None
    assert np.isfinite(row["support_violation_q"])


def test_missing_support_is_excluded_instead_of_becoming_a_whole_side_interval():
    points, _ = _observed_points((0.4,))
    points[0].pop("localization_sigma_q")
    points[0].pop("sampling_sigma_q")
    points[0].pop("q_normal_step")
    result = fit_arc_ellipses(points, parameters=_fixed_parameters(), multistart=1)
    assert result["fit"] is None
    assert result["point_diagnostics"][0]["excluded_reason"] == "observed_support_unavailable"


def test_quality_retains_candidate_but_marks_short_endpoint_support_undetermined():
    trace = {
        "points": [
            {"accepted": True, "branch_id": branch, "side": side, "localization_sigma_q": 0.01}
            for branch in (0, 1)
            for side in ("upper", "lower")
            for _ in range(3)
        ],
        "arc_diagnostics": [
            {
                "arc_id": 1,
                "support_status": "frozen",
                "short_arc": True,
                "support_endpoint_fraction": 1.0,
                "normal_residual_q_rms": 0.001,
                "support_component_count": 1,
            }
        ],
    }
    candidate = {"success": True, "a": 1.0, "b": 0.1, "axis_ratio": 0.1,
                 "theta_deg": 20.0, "condition": 2.0, "rmse": 0.001,
                 "arc_diagnostics": trace["arc_diagnostics"]}
    result = evaluate_arc_evidence(trace, candidate)
    assert result["measurement_status"] == "undetermined"
    assert result["quantitative_parameters"]["a"]["candidate_value"] == 1.0
    assert result["quantitative_parameters"]["a"]["value"] is None
    assert "short_observed_arc" in result["quality"]["flags"]
    assert "arc_support" in result["quality"]["metrics"]


def test_arc_diagnostics_helper_preserves_support_penalty_distinction():
    row = {"arc_id": 1, "projection_distance_q": None, "support_violation_q": 0.2}
    view = arc_diagnostic_rows({"arc_diagnostics": [row]})
    assert view["rows"][0] == row
    assert "projection_distance_q" in view["field_provenance"]
    assert "support_violation_q" in view["field_provenance"]
