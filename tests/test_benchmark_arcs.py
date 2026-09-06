from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

import numpy as np
import pytest

from butterfly_saxs import benchmark_arcs


def test_default_matrix_contains_requested_ratios_and_nonellipse_cases() -> None:
    ratios = {case.axis_ratio for case in benchmark_arcs.DEFAULT_CASES if case.axis_ratio is not None}
    assert {0.005, 0.02, 0.1, 0.4} <= ratios
    assert {"partial_arcs", "missing_branch", "overlapping_unresolved", "asymmetric_branches", "miscentered_branches", "warped_qmap"} <= set(benchmark_arcs.DEFAULT_CASE_IDS)
    assert any(case.non_elliptic for case in benchmark_arcs.DEFAULT_CASES)
    assert any(case.null for case in benchmark_arcs.DEFAULT_CASES)


def test_generator_is_seeded_independent_and_shape_consistent() -> None:
    first = benchmark_arcs.generate_arc_case("ellipse_ratio_020", seed=17, shape=(32, 36))
    second = benchmark_arcs.generate_arc_case("ellipse_ratio_020", seed=17, shape=(32, 36))
    np.testing.assert_array_equal(first["image"], second["image"])
    np.testing.assert_array_equal(first["mask"], second["mask"])
    np.testing.assert_array_equal(first["qmap"].qx, second["qmap"].qx)
    np.testing.assert_array_equal(first["qmap"].qy, second["qmap"].qy)
    assert first["image"].shape == first["qmap"].shape == first["mask"].shape == (32, 36)
    assert np.isfinite(first["image"]).all()
    assert np.all(first["image"] >= 0.0)
    assert first["mask"].dtype == bool
    assert first["truth"]["calibration_claim"] is False
    source_hash = hashlib.sha256(Path(benchmark_arcs.__file__).read_bytes()).hexdigest()
    assert first["truth"]["generator_sha256"] == source_hash
    assert first["truth"]["generator_sha256"] == benchmark_arcs.GENERATOR_HASH
    assert first["truth"]["case_identity"] == {
        "case_id": "ellipse_ratio_020",
        "seed": 17,
        "shape": [32, 36],
    }
    assert len(first["truth"]["case_identity_sha256"]) == 64

    source = inspect.getsource(benchmark_arcs)
    assert "from .intensity" not in source
    assert "from .synthetic" not in source
    assert "fit_intensity_model" not in source


def test_truth_retains_observable_arc_geometry_and_branch_side_labels() -> None:
    result = benchmark_arcs.generate_arc_case("ellipse_ratio_100", shape=(40, 42))
    truth = result["truth"]
    assert truth["axis_ratio"] == pytest.approx(0.1)
    assert truth["b_over_a"] == pytest.approx(truth["b"] / truth["a"])
    assert truth["actual_observable_arcs"]
    assert len(truth["actual_observable_arcs"]) == len(truth["branch_side_truth"])
    for arc in truth["actual_observable_arcs"]:
        assert arc["observable"] is True
        assert arc["branch"] in {"plus", "minus"}
        assert arc["side"] in {"upper", "lower"}
        assert len(arc["polyline_q"]) >= 12
        assert arc["b_over_a"] == pytest.approx(arc["b"] / arc["a"])

    missing = benchmark_arcs.generate_arc_case("missing_branch", shape=(40, 42))["truth"]
    assert missing["branches"]["minus"]["present"] is False
    assert all(arc["branch"] == "plus" for arc in missing["actual_observable_arcs"])


def test_every_ellipse_polyline_matches_declared_local_side_and_quadrant_pair() -> None:
    """Catch labels that cross the local ellipse-side boundary."""

    for spec in benchmark_arcs.DEFAULT_CASES:
        if spec.axis_ratio is None:
            continue
        truth = benchmark_arcs.generate_arc_case(spec, shape=(28, 32))["truth"]
        for arc in truth["actual_observable_arcs"]:
            branch = arc["branch"]
            theta = np.deg2rad(truth["branch_axes"][branch]["theta_deg"])
            center = np.asarray(arc["center"], dtype=float)
            points = np.asarray(arc["polyline_q"], dtype=float)
            delta = points - center
            local_v = -np.sin(theta) * delta[:, 0] + np.cos(theta) * delta[:, 1]
            tolerance = 1.0e-12
            if arc["side"] == "upper":
                assert np.min(local_v) >= -tolerance, (spec.case_id, arc["arc_id"])
            else:
                assert np.max(local_v) <= tolerance, (spec.case_id, arc["arc_id"])

            signs = np.sign(delta)
            if branch == "plus":
                expected_pair = "QI+QIII"
                expected_signs = (1, 1) if arc["side"] == "upper" else (-1, -1)
            else:
                expected_pair = "QII+QIV"
                expected_signs = (-1, 1) if arc["side"] == "upper" else (1, -1)
            assert arc["quadrant_pair"] == expected_pair
            assert np.all(signs[:, 0] == expected_signs[0])
            assert np.all(signs[:, 1] == expected_signs[1])


def test_nonelliptic_truth_does_not_invent_true_axes() -> None:
    curved = benchmark_arcs.generate_arc_case("smooth_nonellipse", shape=(32, 36))
    null = benchmark_arcs.generate_arc_case("null", shape=(32, 36))
    assert curved["truth"]["is_elliptic"] is False
    assert "axes" not in curved["truth"]
    assert "axis_ratio" not in curved["truth"]
    assert curved["truth"]["actual_observable_arcs"]
    assert null["truth"]["actual_observable_arcs"] == []
    assert "axes" not in null["truth"]


def test_warped_qmap_is_observable_but_not_identical_to_flat_map() -> None:
    flat = benchmark_arcs.generate_arc_case("ellipse_ratio_100", shape=(28, 30))
    warped = benchmark_arcs.generate_arc_case("warped_qmap", shape=(28, 30))
    assert warped["truth"]["qmap"]["warped"] is True
    assert not np.array_equal(flat["qmap"].qx, warped["qmap"].qx)
    assert not np.array_equal(flat["qmap"].qy, warped["qmap"].qy)
    assert warped["image"].shape == warped["qmap"].qx.shape
    assert warped["truth"]["detector_pixel_integration"]["cross_component_jacobian"] is True
    assert warped["truth"]["detector_pixel_integration"]["method"].endswith("local_jacobian")
    assert warped["truth"]["centers"]["plus"] == [0.0, 0.0]
    assert warped["truth"]["centers"]["minus"] == [0.0, 0.0]


def test_miscentered_negative_control_is_separate_from_warped_qmap() -> None:
    warped = benchmark_arcs.generate_arc_case("warped_qmap", shape=(24, 26))["truth"]
    miscentered = benchmark_arcs.generate_arc_case("miscentered_branches", shape=(24, 26))["truth"]
    assert warped["qmap"]["warped"] is True
    assert warped["centers"]["plus"] == warped["centers"]["minus"] == [0.0, 0.0]
    assert miscentered["centers"]["plus"] != miscentered["centers"]["minus"]
