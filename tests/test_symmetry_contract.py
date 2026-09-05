from __future__ import annotations

import numpy as np
import pytest

from butterfly_saxs.observables import fit_symmetric_double_ellipse
from butterfly_saxs.pipeline import analyze_frame, fit_symmetric_ellipses, synthetic_butterfly


def _quadrant_points(
    *,
    ratio: float = 0.02,
    reference_deg: float = 5.0,
    theta_deg: float = 17.0,
    center: tuple[float, float] = (0.0, 0.0),
    include_all: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    reference = np.deg2rad(reference_deg)
    theta = np.deg2rad(theta_deg)
    centers = {
        0: (35.0, 215.0),
        1: (145.0, 325.0),
    }
    points: list[list[float]] = []
    labels: list[int] = []
    for branch, angles_deg in centers.items():
        if not include_all and branch == 0:
            angles_deg = angles_deg[:1]
        for angle_deg in angles_deg:
            for angle in np.linspace(
                np.deg2rad(angle_deg - 12.0),
                np.deg2rad(angle_deg + 12.0),
                20,
            ):
                delta = angle - reference - (theta if branch == 0 else -theta)
                radius = 1.0 * ratio / np.sqrt(
                    (ratio * np.cos(delta)) ** 2 + np.sin(delta) ** 2
                )
                points.append(
                    [
                        center[0] + radius * np.cos(angle),
                        center[1] + radius * np.sin(angle),
                    ]
                )
                labels.append(branch)
    return np.asarray(points, dtype=float), np.asarray(labels, dtype=int)


@pytest.mark.parametrize("ratio", [0.02, 0.005])
def test_strict_butterfly_pairing_reports_all_quadrants_without_switching(ratio: float) -> None:
    points, labels = _quadrant_points(ratio=ratio)
    fit = fit_symmetric_double_ellipse(
        points,
        labels=labels,
        strict_symmetry=True,
        reference_axis_deg=5.0,
        parameters={
            "a": {"value": 1.0, "vary": False},
            "axis_ratio": {"value": ratio, "vary": False},
            "theta_deg": {"value": 17.0, "vary": False},
            "cx": {"value": 0.0, "vary": False},
            "cy": {"value": 0.0, "vary": False},
        },
        multistart=1,
    )
    symmetry = fit.symmetry
    assert fit.success
    assert symmetry["symmetry_status"] == "PASS"
    assert symmetry["quadrant_counts"] == {"QI": 20, "QII": 20, "QIII": 20, "QIV": 20}
    assert symmetry["branch_leaks"]["selected"] == 0
    assert symmetry["paired_support"]["0"]["missing_opposite_count"] == 0
    assert symmetry["paired_support"]["1"]["missing_opposite_count"] == 0
    assert not any(
        "synth" in str(flag).casefold() or "mirror" in str(flag).casefold()
        for flag in symmetry["flags"]
    )


def test_partial_invalid_labels_preserve_valid_subset_and_mark_unassigned() -> None:
    points, labels = _quadrant_points(ratio=0.02)
    partial = labels.astype(float)
    partial[3] = np.nan
    fit = fit_symmetric_double_ellipse(
        points,
        labels=partial,
        strict_symmetry=True,
        parameters={"a": 1.0, "axis_ratio": 0.02, "theta_deg": 17.0, "cx": {"value": 0.0, "vary": False}, "cy": {"value": 0.0, "vary": False}},
        multistart=1,
    )
    assert fit.n_points == points.shape[0] - 1
    assert fit.symmetry["unassigned_count"] == 1
    assert "partial_branch_labels_excluded" in fit.symmetry["flags"]
    assert fit.symmetry["symmetry_status"] == "WARN"


def test_strict_fit_excludes_reference_axis_points_but_keeps_original_indices() -> None:
    points = np.asarray(
        [[1.0, 0.0], [0.8, 0.3], [0.6, 0.4], [-0.8, -0.3], [-0.6, -0.4], [0.7, -0.4]],
        dtype=float,
    )
    labels = np.asarray([0, 0, 0, 0, 0, 1], dtype=int)
    fit = fit_symmetric_double_ellipse(
        points,
        labels=labels,
        strict_symmetry=True,
        parameters={"a": 1.0, "axis_ratio": 0.4, "theta_deg": 12.0, "cx": {"value": 0.0, "vary": False}, "cy": {"value": 0.0, "vary": False}},
        multistart=1,
    )
    assert fit.n_points == 5
    assert fit.branch_assignment_indices is not None
    assert 0 not in set(np.asarray(fit.branch_assignment_indices, dtype=int).tolist())
    assert fit.symmetry["unassigned_count"] >= 1


def test_missing_opposite_quadrant_is_diagnostic_only_and_no_points_are_added() -> None:
    points, labels = _quadrant_points(ratio=0.02, include_all=False)
    fit = fit_symmetric_double_ellipse(
        points,
        labels=labels,
        strict_symmetry=True,
        reference_axis_deg=5.0,
        parameters={"a": 1.0, "axis_ratio": 0.02, "theta_deg": 17.0, "cx": {"value": 0.0, "vary": False}, "cy": {"value": 0.0, "vary": False}},
        multistart=1,
    )
    assert fit.n_points == points.shape[0]
    assert fit.symmetry["symmetry_status"] in {"WARN", "FAIL"}
    assert sum(item["missing_opposite_count"] for item in fit.symmetry["paired_support"].values()) > 0


def test_nonzero_fixed_center_and_reference_rotation_are_classified_in_local_frame() -> None:
    center = (0.23, -0.17)
    points, labels = _quadrant_points(
        ratio=0.02,
        reference_deg=17.0,
        theta_deg=11.0,
        center=center,
    )
    fit = fit_symmetric_double_ellipse(
        points,
        labels=labels,
        strict_symmetry=True,
        reference_axis_deg=17.0,
        parameters={
            "a": {"value": 1.0, "vary": False},
            "axis_ratio": {"value": 0.02, "vary": False},
            "theta_deg": {"value": 11.0, "vary": False},
            "cx": {"value": center[0], "vary": False},
            "cy": {"value": center[1], "vary": False},
        },
        multistart=1,
    )
    assert fit.success
    assert fit.center == pytest.approx(center, abs=1e-12)
    assert fit.symmetry["center_verified"] is True
    assert fit.symmetry["branch_leaks"]["selected"] == 0


def test_global_branch_swap_is_equivalent_but_quadrant_pairing_remains_fixed() -> None:
    points, labels = _quadrant_points(ratio=0.02)
    fit = fit_symmetric_double_ellipse(
        points,
        labels=1 - labels,
        strict_symmetry=True,
        reference_axis_deg=5.0,
        parameters={"a": 1.0, "axis_ratio": 0.02, "theta_deg": 17.0, "cx": {"value": 0.0, "vary": False}, "cy": {"value": 0.0, "vary": False}},
        multistart=1,
    )
    assert fit.symmetry["branch_leaks"]["selected"] == 0
    assert fit.symmetry["branch_leaks"]["global_swap"] is True
    assert fit.symmetry["paired_support"]["0"]["quadrant_pair"] == "QI+QIII"
    assert fit.symmetry["paired_support"]["1"]["quadrant_pair"] == "QII+QIV"


def test_strict_labels_win_over_nearest_ellipse_at_a_crossing() -> None:
    points, labels = _quadrant_points(ratio=0.02)
    adversarial = points.copy()
    adversarial[0] = points[40]  # branch-1 geometry, retained branch-0 identity
    fit = fit_symmetric_double_ellipse(
        adversarial,
        labels=labels,
        strict_symmetry=True,
        reference_axis_deg=5.0,
        parameters={
            "a": {"value": 1.0, "vary": False},
            "axis_ratio": {"value": 0.02, "vary": False},
            "theta_deg": {"value": 17.0, "vary": False},
            "cx": {"value": 0.0, "vary": False},
            "cy": {"value": 0.0, "vary": False},
        },
        multistart=1,
    )
    assert fit.branch_assignment is not None
    assert int(fit.branch_assignment[0]) == 0
    assert "branch_quadrant_leak" in fit.symmetry["flags"]


def test_pipeline_wrapper_preserves_branch_labels_and_weights() -> None:
    points, labels = _quadrant_points(ratio=0.02)
    rows = [
        {"qx": float(point[0]), "qy": float(point[1]), "branch_id": int(label), "weight": 2.0}
        for point, label in zip(points, labels)
    ]
    result = fit_symmetric_ellipses(
        rows,
        qmap={"q_unit": "nm^-1"},
        config={"analysis": {"strict_symmetry": True}},
    )
    assert result["status"] == "ok"
    assert result["symmetry"]["branch_leaks"]["selected"] == 0
    assert result["symmetry"]["quadrant_counts"]["QI"] == 20
    assert result["branch_assignment"]["shape"] == [len(rows)]


def test_public_pipeline_exposes_symmetry_and_quadrant_point_metadata() -> None:
    image, qmap = synthetic_butterfly((32, 32), return_qmap=True, seed=41)
    result = analyze_frame(image, qmap=qmap)
    symmetry = result.ellipse_fit["symmetry"]
    assert "quadrant_counts" in symmetry
    assert "paired_support" in symmetry
    points = result.observables["ridge"]["points"]
    observed = [point for point in points if point.get("valid")]
    assert observed
    assert all("quadrant_pair" in point for point in observed)
    assert all("branch_assignment_source" in point for point in observed)
