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


def _run_fit_trace(monkeypatch, *, first_order_q_hint=None):
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
    trace = {
        "points": trace_points,
        "diagnostics": {
            "first_order_q_hint": {"q_star": first_order_q_hint},
        },
    }
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
    ("hint", "expected_q", "expected_source"),
    [(None, 1.0, "observed_arc_radius"), (2.0, 2.0, "first_order_iq")],
)
def test_fit_summaries_use_only_solver_used_points_and_keep_ring_hint(
    monkeypatch, hint, expected_q, expected_source
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
