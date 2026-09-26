from __future__ import annotations

import math

from butterfly_saxs.arc_geometry import fit_arc_ellipses
from butterfly_saxs.arc_support import freeze_observed_support
from butterfly_saxs.ellipse import EllipseGeometry


def _supported_arc():
    geometry = EllipseGeometry(0.0, 0.0, 1.5, 0.1, 0.35)
    points = []
    point_ids = []
    for index, t in enumerate((0.4, 0.5, 0.6)):
        qx, qy = geometry.point(t)
        point_id = f"p-{index}"
        point_ids.append(point_id)
        points.append(
            {
                "point_id": point_id,
                "qx": float(qx),
                "qy": float(qy),
                "branch_id": 0,
                "side": "upper",
                "arc_id": 3,
                "accepted": True,
                "localization_sigma_q": 0.002,
                "sampling_sigma_q": 0.002,
                "q_normal_step": 0.004,
                "tangent_qx": -math.sin(t),
                "tangent_qy": math.cos(t),
                "normal_qx": math.cos(t),
                "normal_qy": math.sin(t),
            }
        )
    arc = {"arc_id": 3, "ordered_point_ids": point_ids}
    return points, [arc]


def _fixed_parameters():
    return {
        "cx": {"value": 0.0, "vary": False},
        "cy": {"value": 0.0, "vary": False},
        "a": {"value": 1.5, "vary": False},
        "axis_ratio": {"value": 0.1, "vary": False},
        "theta": {"value": 0.35, "vary": False},
    }


def test_unassigned_points_do_not_count_as_observed_support_arcs():
    points, arcs = _supported_arc()
    unassigned = {
        "point_id": "unassigned",
        "qx": 0.7,
        "qy": 0.2,
        "branch_id": 0,
        "side": "upper",
        "arc_id": -1,
        "accepted": False,
        "localization_sigma_q": 0.002,
        "sampling_sigma_q": 0.002,
        "q_normal_step": 0.004,
        "tangent_qx": 1.0,
        "tangent_qy": 0.0,
        "normal_qx": 0.0,
        "normal_qy": 1.0,
    }
    points.append(unassigned)

    envelope = freeze_observed_support(points, arcs)

    assert [arc["arc_id"] for arc in envelope["arcs"]] == [3]
    assert envelope["summary"]["n_arcs"] == 1
    assert envelope["summary"]["n_frozen_arcs"] == 1
    assert envelope["summary"]["n_unavailable_arcs"] == 0
    assert envelope["summary"]["n_unassigned_points"] == 1
    assert unassigned["support_status"] == "unavailable_no_arc"
    assert unassigned["support_frozen"] is False


def test_arc_fit_keeps_unassigned_point_diagnostic_without_pseudo_arc():
    points, arcs = _supported_arc()
    unassigned = {
        "point_id": "unassigned",
        "qx": 0.7,
        "qy": 0.2,
        "branch_id": 0,
        "side": "upper",
        "arc_id": -1,
        "accepted": True,
    }
    points.append(unassigned)
    freeze_observed_support(points, arcs)

    result = fit_arc_ellipses(points, parameters=_fixed_parameters(), multistart=1)

    diagnostic = result["point_diagnostics"][-1]
    assert diagnostic["accepted"] is True
    assert diagnostic["used"] is False
    assert diagnostic["excluded_reason"] == "unassigned_arc_id"
    assert diagnostic["support_status"] == "unavailable_no_arc"
    assert [arc["arc_id"] for arc in result["arc_diagnostics"]] == [3]


def test_missing_observed_support_is_not_labeled_manual_only():
    points, arcs = _supported_arc()
    unsupported = {
        "point_id": "unsupported",
        "qx": 0.8,
        "qy": 0.2,
        "branch_id": 0,
        "side": "upper",
        "arc_id": 9,
        "accepted": True,
    }
    points.append(unsupported)
    arcs.append({"arc_id": 9, "ordered_point_ids": ["unsupported"]})
    freeze_observed_support(points, arcs)

    result = fit_arc_ellipses(points, parameters=_fixed_parameters(), multistart=1)

    diagnostic = next(row for row in result["point_diagnostics"] if row["point_id"] == "unsupported")
    arc = next(row for row in result["arc_diagnostics"] if row["arc_id"] == 9)
    assert diagnostic["used"] is False
    assert diagnostic["excluded_reason"] == "observed_support_unavailable"
    assert arc["n_used"] == 0
    assert arc["support_status"] == "unavailable_missing_observed_scale"
