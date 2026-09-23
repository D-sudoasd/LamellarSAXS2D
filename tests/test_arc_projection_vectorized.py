from __future__ import annotations

import math

import numpy as np

from butterfly_saxs.arc_geometry import (
    _project_ellipse_arc,
    _project_point_to_support,
    _symmetric_arc_projection,
    project_arc_points,
)
from butterfly_saxs.ellipse import EllipseGeometry, ellipse_implicit


_PROJECTION_FIELDS = (
    "t",
    "distance",
    "global_distance",
    "at_endpoint",
    "endpoint_clipped",
    "support_endpoint_clipped",
    "manual_bound_clipped",
    "projection_valid",
    "support_infeasible",
    "support_violation_q",
    "objective_distance",
    "support_component_id",
    "support_interval_index",
)


def _rectangle_at(point: np.ndarray, *, component_id: int, segment_index: int, half_width: float = 0.02) -> dict[str, object]:
    return {
        "frame_origin_q": point.tolist(),
        "tangent_q": [1.0, 0.0],
        "normal_q": [0.0, 1.0],
        "tangent_bounds": [-half_width, half_width],
        "normal_bounds": [-half_width, half_width],
        "component_id": component_id,
        "segment_index": segment_index,
    }


def _projection_fixture() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[dict[str, object]],
    dict[str, float],
    float,
]:
    values = {
        "cx": 0.12,
        "cy": -0.08,
        "a": 1.4,
        "axis_ratio": 0.04,
        "theta": 0.37,
    }
    reference_axis = 0.13
    branch_geometry = {
        branch: EllipseGeometry(
            values["cx"],
            values["cy"],
            values["a"],
            values["axis_ratio"],
            reference_axis + (1.0 if branch == 0 else -1.0) * values["theta"],
        )
        for branch in (0, 1)
    }
    rows = [
        (0, "upper", 0.48, [(0.10, 0.24), (0.40, 0.60)], ()),
        (0, "lower", 3.62, None, ()),
        (0, "upper", 0.78, None, ()),
        (1, "lower", 3.28, [(3.30, 3.42), (3.55, 3.72)], ()),
        (1, "upper", 0.62, None, ()),
        (1, "lower", 5.70, None, ()),
        (0, "upper", 0.15, [(0.30, 0.50)], ()),
    ]
    points: list[list[float]] = []
    branch_ids: list[int] = []
    sides: list[str] = []
    interval_specs: list[dict[str, object]] = []
    records: list[dict[str, object]] = []
    for point_id, (branch, side, t, manual_intervals, _unused) in enumerate(rows):
        geometry = branch_geometry[branch]
        point = np.asarray(geometry.point(t), dtype=float).reshape(2)
        rectangles: tuple[dict[str, object], ...]
        if point_id == 1:
            rectangles = (
                _rectangle_at(point + np.asarray([3.0, 3.0]), component_id=7, segment_index=0),
                _rectangle_at(point, component_id=8, segment_index=4),
            )
        elif point_id == 2:
            rectangles = (_rectangle_at(np.asarray([3.0, 3.0]), component_id=9, segment_index=2),)
        elif point_id == 4:
            rectangles = (_rectangle_at(point, component_id=10, segment_index=5),)
        elif point_id == 5:
            rectangles = (_rectangle_at(np.asarray([-3.0, 2.0]), component_id=11, segment_index=1),)
        else:
            rectangles = ()
        points.append(point.tolist())
        branch_ids.append(branch)
        sides.append(side)
        interval_specs.append({"manual_intervals": manual_intervals, "rectangles": rectangles})
        record = {
            "point_id": point_id,
            "qx": float(point[0]),
            "qy": float(point[1]),
            "branch_id": branch,
            "side": side,
            "arc_id": point_id,
            "accepted": True,
            "arc_t_intervals": manual_intervals,
        }
        records.append(record)
    return (
        np.asarray(points, dtype=float),
        np.asarray(branch_ids, dtype=int),
        np.asarray(sides, dtype=object),
        interval_specs,
        values,
        reference_axis,
    )


def _scalar_oracle(
    points: np.ndarray,
    values: dict[str, float],
    branch_ids: np.ndarray,
    sides: np.ndarray,
    interval_specs: list[dict[str, object]],
    reference_axis: float,
) -> dict[str, object]:
    geometry = EllipseGeometry.from_values(values)
    result: dict[str, object] = {
        "t": np.full(points.shape[0], np.nan, dtype=float),
        "distance": np.full(points.shape[0], np.nan, dtype=float),
        "global_distance": np.full(points.shape[0], np.nan, dtype=float),
        "at_endpoint": np.zeros(points.shape[0], dtype=bool),
        "endpoint_clipped": np.zeros(points.shape[0], dtype=bool),
        "support_endpoint_clipped": np.zeros(points.shape[0], dtype=bool),
        "manual_bound_clipped": np.zeros(points.shape[0], dtype=bool),
        "projection_valid": np.zeros(points.shape[0], dtype=bool),
        "support_infeasible": np.zeros(points.shape[0], dtype=bool),
        "support_violation_q": np.zeros(points.shape[0], dtype=float),
        "objective_distance": np.full(points.shape[0], np.nan, dtype=float),
        "support_component_id": np.full(points.shape[0], None, dtype=object),
        "support_interval_index": np.full(points.shape[0], -1, dtype=int),
        "implicit": np.full(points.shape[0], np.nan, dtype=float),
    }
    for branch in (0, 1):
        mask = branch_ids == branch
        branch_geometry = EllipseGeometry(
            geometry.cx,
            geometry.cy,
            geometry.a,
            geometry.axis_ratio,
            reference_axis + (1.0 if branch == 0 else -1.0) * geometry.theta,
        )
        for row in np.flatnonzero(mask):
            spec = interval_specs[int(row)]
            projection = _project_point_to_support(
                points[row],
                branch_geometry,
                str(sides[row]),
                spec.get("manual_intervals"),
                spec.get("rectangles", ()),
                include_global_oracle=True,
            )
            for name in _PROJECTION_FIELDS:
                result[name][row] = projection[name]
        result["implicit"][mask] = ellipse_implicit(points[mask], branch_geometry)
    signed_distance = np.copysign(result["objective_distance"], result["implicit"])
    result["signed_distance"] = np.nan_to_num(
        signed_distance, nan=1.0e12, posinf=1.0e12, neginf=-1.0e12
    )
    result["objective_distance"] = np.nan_to_num(
        result["objective_distance"], nan=1.0e12, posinf=1.0e12, neginf=1.0e12
    )
    return result


def test_batched_projection_matches_scalar_oracle_for_support_and_manual_intervals() -> None:
    points, branch_ids, sides, interval_specs, values, reference_axis = _projection_fixture()

    batched = _symmetric_arc_projection(
        points,
        values,
        branch_ids,
        sides,
        interval_specs,
        reference_axis,
        include_global_oracle=True,
    )
    scalar = _scalar_oracle(points, values, branch_ids, sides, interval_specs, reference_axis)

    for name in _PROJECTION_FIELDS + ("implicit", "signed_distance"):
        actual = np.asarray(batched[name])
        expected = np.asarray(scalar[name])
        if actual.dtype == object:
            assert np.array_equal(actual, expected), name
        elif actual.dtype == bool:
            assert np.array_equal(actual, expected), name
        else:
            np.testing.assert_allclose(actual, expected, rtol=0.0, atol=3.0e-13, equal_nan=True)

    assert bool(batched["projection_valid"][1])
    assert int(batched["support_component_id"][1]) == 8
    assert not bool(batched["projection_valid"][2])
    assert bool(batched["support_infeasible"][2])
    assert bool(batched["manual_bound_clipped"][6])


def test_vectorized_flat_arc_projection_preserves_endpoint_and_global_distance() -> None:
    geometry = EllipseGeometry(0.11, -0.07, 1.6, 0.005, 0.41)
    t = np.asarray([0.20, 0.2000000001, 0.80, 3.30, 3.31], dtype=float)
    points = np.asarray(geometry.point(t), dtype=float).reshape(-1, 2)
    local_u, local_v = (
        points[:, 0] - geometry.cx,
        points[:, 1] - geometry.cy,
    )
    cosine = math.cos(geometry.theta)
    sine = math.sin(geometry.theta)
    local_u, local_v = cosine * local_u + sine * local_v, -sine * local_u + cosine * local_v
    vectorized = _project_ellipse_arc(
        local_u,
        local_v,
        geometry.a,
        geometry.b,
        np.asarray([0.25, 0.25, 0.25, 3.25, 3.25]),
        np.asarray([0.75, 0.75, 0.75, 3.35, 3.35]),
        include_global_oracle=True,
    )
    scalar = [
        _project_ellipse_arc(
            np.asarray([u]),
            np.asarray([v]),
            geometry.a,
            geometry.b,
            np.asarray([lo]),
            np.asarray([hi]),
            include_global_oracle=True,
        )
        for u, v, lo, hi in zip(
            local_u,
            local_v,
            np.asarray([0.25, 0.25, 0.25, 3.25, 3.25]),
            np.asarray([0.75, 0.75, 0.75, 3.35, 3.35]),
        )
    ]
    for name in ("t", "distance", "global_distance"):
        expected = np.asarray([item[name][0] for item in scalar])
        np.testing.assert_allclose(vectorized[name], expected, rtol=0.0, atol=3.0e-13, equal_nan=True)
    assert np.array_equal(vectorized["at_endpoint"], np.asarray([item["at_endpoint"][0] for item in scalar]))
    assert np.array_equal(vectorized["endpoint_clipped"], np.asarray([item["endpoint_clipped"][0] for item in scalar]))


def test_public_projection_branch_swap_reuses_same_bounded_geometry() -> None:
    points, branch_ids, _sides, interval_specs, values, _reference_axis = _projection_fixture()
    records: list[dict[str, object]] = []
    for index, point in enumerate(points):
        spec = interval_specs[index]
        if spec["manual_intervals"] is None or spec["rectangles"]:
            continue
        records.append(
            {
                "point_id": index,
                "qx": float(point[0]),
                "qy": float(point[1]),
                "branch_id": int(1 - branch_ids[index]),
                "side": "upper" if index in (0, 2, 6) else "lower",
                "arc_id": index,
                "accepted": True,
                "arc_t_intervals": spec["manual_intervals"],
            }
        )
    swapped = project_arc_points(records, values, branch_swap_applied=True)
    direct_records = [dict(record, branch_id=1 - int(record["branch_id"])) for record in records]
    direct = project_arc_points(direct_records, values, branch_swap_applied=False)
    swapped_rows = swapped["point_diagnostics"]
    direct_rows = direct["point_diagnostics"]
    assert len(swapped_rows) == len(direct_rows)
    for actual, expected in zip(swapped_rows, direct_rows):
        assert actual["projection_valid"] == expected["projection_valid"]
        assert actual["projection_t"] == expected["projection_t"]
        assert actual["distance_q"] == expected["distance_q"]
        assert actual["projected_qx"] == expected["projected_qx"]
        assert actual["projected_qy"] == expected["projected_qy"]
