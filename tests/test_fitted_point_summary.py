from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from butterfly_saxs import arc_geometry, butterfly


def _point(point_id: str, branch: int, side: str, radius: float, arc_id: int) -> dict:
    sign_x = 1.0 if branch == 0 else -1.0
    sign_y = 1.0 if side == "upper" else -1.0
    return {
        "point_id": point_id,
        "qx": sign_x * radius * math.cos(math.radians(30.0)),
        "qy": sign_y * radius * math.sin(math.radians(30.0)),
        "branch_id": branch,
        "side": side,
        "arc_id": arc_id,
        "accepted": True,
        "valid": True,
    }


def _run_fit_trace(monkeypatch, *, first_order_q_hint=None, method_version=None):
    fitted = [
        _point("fit-0", 0, "upper", 1.00, 0),
        _point("fit-1", 0, "upper", 1.01, 0),
        _point("fit-2", 0, "lower", 0.99, 1),
        _point("fit-3", 1, "upper", 1.02, 2),
        _point("fit-4", 1, "lower", 0.98, 3),
    ]
    unassigned = [
        _point(f"unassigned-{index}", index % 2, "upper" if index % 2 else "lower", 0.5, -1)
        for index in range(10)
    ]
    invalid = _point("invalid-coordinate", 1, "upper", 0.5, 4)
    invalid["qx"] = float("nan")
    trace_points = [*fitted, *unassigned, invalid]
    if isinstance(first_order_q_hint, dict):
        first_order_diagnostic = first_order_q_hint
    elif first_order_q_hint is None:
        first_order_diagnostic = {
            "q_star": None,
            "selection_status": "no_hint",
            "reason": "no_supported_local_peak",
        }
    else:
        first_order_diagnostic = {
            "q_star": first_order_q_hint,
            "selection_status": "selected",
            "reason": "ok",
        }
    trace = {
        "points": trace_points,
        "diagnostics": {"first_order_q_hint": first_order_diagnostic},
    }
    if method_version is not None:
        trace["method_version"] = method_version
    captured = {}

    def fake_fit(points, **_kwargs):
        captured["input_points"] = list(points)
        return {
            "fit": SimpleNamespace(
                success=True,
                status="ok",
                a=1.1,
                b=0.4,
                axis_ratio=0.4 / 1.1,
                theta_deg=15.0,
                center=(0.0, 0.0),
                q_unit="nm^-1",
                message="mock fit",
                bound_flags={},
            ),
            "point_diagnostics": [
                {
                    "point_id": point["point_id"],
                    "used": bool(
                        point["arc_id"] >= 0 and np.isfinite(point["qx"])
                        and np.isfinite(point["qy"])
                    ),
                    "distance_q": 0.002 if point["arc_id"] >= 0 and np.isfinite(point["qx"])
                    and np.isfinite(point["qy"]) else None,
                }
                for point in points
            ],
            "arc_diagnostics": [],
            "branch_swap_applied": False,
        }

    monkeypatch.setattr(arc_geometry, "fit_arc_ellipses", fake_fit)
    result = butterfly._fit_trace(
        trace,
        parameters=None,
        reference=0.0,
        multistart=1,
        unit="nm^-1",
        cancel_event=None,
    )
    return result, captured, trace_points


@pytest.mark.parametrize(
    ("hint", "expected_q", "expected_source", "expected_status"),
    [
        (None, 1.0, "observed_arc_radius_not_order_assigned", "no_hint"),
        (2.0, 2.0, "first_order_iq", "selected"),
        (
            {"q_star": 2.0, "selection_status": "ambiguous", "reason": "ambiguous_lower_q_peak_family"},
            1.0,
            "observed_arc_radius_not_order_assigned",
            "ambiguous",
        ),
        (
            {"q_star": 2.0, "selection_status": "no_hint", "reason": "no_significant_peak"},
            1.0,
            "observed_arc_radius_not_order_assigned",
            "no_hint",
        ),
        (
            {"q_star": 2.0, "selection_status": None, "reason": "missing_selection_status"},
            1.0,
            "observed_arc_radius_not_order_assigned",
            "not_available",
        ),
    ],
)
def test_fit_summaries_use_only_solver_used_points_and_keep_ring_hint(
    monkeypatch, hint, expected_q, expected_source, expected_status
):
    result, captured, trace_points = _run_fit_trace(
        monkeypatch, first_order_q_hint=hint
    )

    # The solver still receives the complete trace so rejected observations
    # remain available to its point-level diagnostics.
    assert len(captured["input_points"]) == len(trace_points)
    assert any(point["arc_id"] == -1 for point in captured["input_points"])
    assert result["n_points"] == 5
    assert sum(result["symmetry"]["quadrant_counts"].values()) == 5
    assert result["symmetry"]["unassigned_count"] == 0
    assert result["q_star_from_arcs"] == pytest.approx(expected_q)
    assert result["L_from_observed_radius_nm"] == pytest.approx(2.0 * math.pi / expected_q)
    assert result["q_star_source"] == expected_source
    assert result["q_unit"] == "nm^-1"
    assert result["observed_arc_q_median"] == pytest.approx(1.0)
    expected_hint = hint.get("q_star") if isinstance(hint, dict) else hint
    assert result["radial_hint_q"] == expected_hint
    assert result["radial_hint_selection_status"] == expected_status
    assert result["radial_hint_reason"] == (
        hint.get("reason") if isinstance(hint, dict)
        else "no_supported_local_peak" if hint is None else "ok"
    )
    comparison = result["radial_arc_comparison"]
    assert comparison["q_unit"] == "nm^-1"
    assert comparison["observed_arc_q_median"] == pytest.approx(1.0)
    assert comparison["radial_hint_q"] == expected_hint
    if expected_hint is not None:
        assert comparison["radial_hint_to_observed_arc_ratio"] == pytest.approx(expected_hint)
        assert comparison["signed_relative_difference"] == pytest.approx(expected_hint - 1.0)
    else:
        assert comparison["radial_hint_to_observed_arc_ratio"] is None
        assert comparison["signed_relative_difference"] is None
    assert "observed_arc_q_median" in result["parameters"]
    assert "radial_hint_q" in result["parameters"]
    assert "radial_arc_comparison" not in result["parameters"]


def test_annular_point_radii_are_labeled_as_prescribed_coordinates(monkeypatch):
    result, _captured, _trace_points = _run_fit_trace(
        monkeypatch,
        first_order_q_hint={
            "q_star": None,
            "selection_status": "not_used",
            "reason": "prescribed_annuli",
        },
        method_version="butterfly-annular-test",
    )

    assert result["q_star_from_arcs"] is None
    assert result["L_from_observed_radius_nm"] is None
    assert result["q_star_source"] == "unavailable_prescribed_annuli"
    assert result["observed_arc_q_median"] == pytest.approx(1.0)
    assert result["observed_arc_q_source"] == "prescribed_annulus_coordinates"
    assert result["radial_arc_comparison"]["status"] == "prescribed_annulus_coordinates"
    assert (
        result["radial_arc_comparison"]["observed_arc_q_source"]
        == "prescribed_annulus_coordinates"
    )


def test_negative_ids_and_nonfinite_coordinates_do_not_pass_initial_fit_screen(monkeypatch):
    called = False

    def unexpected_fit(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("insufficient finite assigned points should return before fitting")

    monkeypatch.setattr(arc_geometry, "fit_arc_ellipses", unexpected_fit)
    points = [
        _point(f"valid-{index}", index % 2, "upper" if index % 2 else "lower", 1.0, index)
        for index in range(4)
    ]
    unassigned = _point("unassigned", 1, "upper", 0.5, -1)
    invalid = _point("invalid", 1, "lower", 0.5, 5)
    invalid["qy"] = 10**400

    result = butterfly._fit_trace(
        {"points": [*points, unassigned, invalid]},
        parameters=None,
        reference=0.0,
        multistart=1,
        unit="nm^-1",
        cancel_event=None,
    )

    assert called is False
    assert result["success"] is False
    assert result["status"] == "not_fitted"


@pytest.mark.parametrize("trace", [{}, {"points": []}])
def test_empty_trace_returns_not_fitted_without_calling_solver(monkeypatch, trace):
    def unexpected_fit(*_args, **_kwargs):
        raise AssertionError("an empty trace must not reach the fitter")

    monkeypatch.setattr(arc_geometry, "fit_arc_ellipses", unexpected_fit)
    result = butterfly._fit_trace(
        trace,
        parameters=None,
        reference=0.0,
        multistart=1,
        unit="nm^-1",
        cancel_event=None,
    )

    assert result["success"] is False
    assert result["status"] == "not_fitted"
