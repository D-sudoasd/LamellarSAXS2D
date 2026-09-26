from __future__ import annotations

import math

from butterfly_saxs.annular_trace import _select_and_connect


def _peak(ring: int, angle: float, prominence: float, *, name: str) -> dict:
    radians = math.radians(angle)
    return {
        "point_id": name,
        "annulus_index": ring,
        "angular_bin_index": ring,
        "chi_deg": angle,
        "qx": math.cos(radians),
        "qy": math.sin(radians),
        "branch_id": 0,
        "quadrant": "QI",
        "prominence": prominence,
        "accepted": True,
        "valid": True,
        "reason": "accepted_annular_peak",
        "trajectory_id": None,
        "topology_flags": [],
    }


def test_continuous_weaker_peaks_beat_stronger_jumping_candidates():
    points = []
    weak_angles = (10.0, 15.0, 20.0, 25.0)
    strong_angles = (45.0, 70.0, 85.0, 85.0)
    for ring, (weak, strong) in enumerate(zip(weak_angles, strong_angles)):
        points.extend(
            (
                _peak(ring, weak, 6.0, name=f"weak-{ring}"),
                _peak(ring, strong, 10.0, name=f"strong-{ring}"),
            )
        )

    groups, edges, diagnostics = _select_and_connect(points, angle_step_deg=5.0)

    selected = [point for point in points if point["accepted"]]
    assert [point["chi_deg"] for point in selected] == list(weak_angles)
    assert len(groups) == 1
    assert len(edges) == 3
    assert all(not point["accepted"] for point in points if point["point_id"].startswith("strong-"))
    assert selected[0]["trajectory_confidence"] == "low"
    assert "weaker_local_prominence_path" in selected[0]["trajectory_low_confidence_reasons"]
    assert diagnostics["n_selected_points"] == 4


def test_fork_prefers_the_longer_observed_path_and_keeps_shorter_branch_visible():
    points = [
        _peak(0, 20.0, 8.0, name="shared-start"),
        _peak(1, 24.0, 8.0, name="long-1"),
        _peak(1, 35.0, 20.0, name="short-1"),
        _peak(2, 28.0, 8.0, name="long-2"),
        _peak(2, 45.0, 20.0, name="short-2"),
        _peak(3, 32.0, 8.0, name="long-3"),
        _peak(3, 55.0, 20.0, name="short-3"),
        _peak(4, 36.0, 8.0, name="long-4"),
    ]

    groups, edges, diagnostics = _select_and_connect(points, angle_step_deg=5.0)

    selected = [point for point in points if point["accepted"]]
    assert {point["point_id"] for point in selected} == {
        "shared-start", "long-1", "long-2", "long-3", "long-4"
    }
    assert len(groups) == 1
    assert len(edges) == 4
    assert all(not point["accepted"] for point in points if point["point_id"].startswith("short-"))
    assert all(point["reason"] for point in points if point["point_id"].startswith("short-"))


def test_alternating_local_prominence_does_not_switch_between_two_tracks():
    points = []
    track_a = (15.0, 20.0, 25.0, 30.0)
    track_b = (60.0, 63.0, 66.0, 69.0)
    a_strength = (10.0, 6.0, 10.0, 6.0)
    b_strength = (6.0, 10.0, 6.0, 10.0)
    for ring in range(4):
        points.extend(
            (
                _peak(ring, track_a[ring], a_strength[ring], name=f"a-{ring}"),
                _peak(ring, track_b[ring], b_strength[ring], name=f"b-{ring}"),
            )
        )

    groups, edges, diagnostics = _select_and_connect(points, angle_step_deg=5.0)

    selected = [point for point in points if point["accepted"]]
    selected_ids = {point["point_id"] for point in selected}
    assert selected_ids in (
        {f"a-{ring}" for ring in range(4)},
        {f"b-{ring}" for ring in range(4)},
    )
    assert len(groups) == 1
    assert len(edges) == 3
    assert all(point.get("trajectory_confidence") == "low" for point in selected)
    assert all("ambiguous_annular_trajectory" in point["topology_flags"] for point in selected)
    assert all(not point["accepted"] for point in points if point not in selected)
    assert all(
        point["reason"] == "ambiguous_trajectory_alternative"
        for point in points if point not in selected
    )
    assert diagnostics["n_ambiguous_trajectories"] == 1


def test_missing_annulus_splits_tracks_without_inserting_or_bridging_a_peak():
    points = [
        _peak(0, 20.0, 8.0, name="before-0"),
        _peak(1, 25.0, 8.0, name="before-1"),
        _peak(3, 35.0, 8.0, name="after-3"),
        _peak(4, 40.0, 8.0, name="after-4"),
    ]

    groups, edges, _diagnostics = _select_and_connect(points, angle_step_deg=5.0)

    assert len(groups) == 2
    assert len(edges) == 2
    assert {point["annulus_index"] for point in points} == {0, 1, 3, 4}
    assert all(point["accepted"] for point in points)
    assert len({point["trajectory_id"] for point in points}) == 2
    assert not any(
        points[left]["annulus_index"] < 2 < points[right]["annulus_index"]
        for left, right in edges
    )
    assert all(point["trajectory_confidence"] == "low" for point in points)
