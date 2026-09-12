from __future__ import annotations

from collections import Counter

import numpy as np

from butterfly_saxs.benchmark_arcs import generate_arc_case
from butterfly_saxs.butterfly_uncertainty import resample_butterfly
from butterfly_saxs.geometry import GeometryMaps
from butterfly_saxs.pipeline import analyze_frame


def _nearest_polyline_values(case: dict, polyline: list[list[float]]) -> np.ndarray:
    """Sample generated intensity at the detector pixels nearest to q points."""

    qx = np.asarray(case["qmap"].qx, dtype=float)
    qy = np.asarray(case["qmap"].qy, dtype=float)
    image = np.asarray(case["image"], dtype=float)
    points = np.asarray(polyline, dtype=float)
    stride = max(1, len(points) // 32)
    values = []
    for qx_point, qy_point in points[::stride]:
        distance = (qx - qx_point) ** 2 + (qy - qy_point) ** 2
        row, column = np.unravel_index(int(np.argmin(distance)), distance.shape)
        values.append(image[row, column])
    return np.asarray(values, dtype=float)


def test_rasterized_observable_polylines_have_signal_above_background() -> None:
    case_ids = (
        "ellipse_ratio_005",
        "ellipse_ratio_020",
        "ellipse_ratio_100",
        "ellipse_ratio_400",
        "partial_arcs",
        "missing_branch",
        "overlapping_unresolved",
        "asymmetric_branches",
        "miscentered_branches",
        "warped_qmap",
        "smooth_nonellipse",
    )
    for case_id in case_ids:
        case = generate_arc_case(case_id, seed=17, shape=(48, 52))
        truth = case["truth"]
        for arc in truth["actual_observable_arcs"]:
            values = _nearest_polyline_values(case, arc["polyline_q"])
            assert np.isfinite(values).all(), (case_id, arc["arc_id"])
            assert np.nanmedian(values) > 0.018 + 10.0 * truth["noise"]["sigma"], (
                case_id,
                arc["arc_id"],
                float(np.nanmedian(values)),
            )


def test_seeds_change_image_noise_without_changing_declared_geometry() -> None:
    first = generate_arc_case("ellipse_ratio_020", seed=17, shape=(48, 52))
    second = generate_arc_case("ellipse_ratio_020", seed=18, shape=(48, 52))

    np.testing.assert_array_equal(first["qmap"].qx, second["qmap"].qx)
    np.testing.assert_array_equal(first["qmap"].qy, second["qmap"].qy)
    np.testing.assert_array_equal(first["mask"], second["mask"])
    assert first["truth"]["actual_observable_arcs"] == second["truth"]["actual_observable_arcs"]
    assert first["truth"]["branch_side_truth"] == second["truth"]["branch_side_truth"]
    assert first["truth"]["case_identity"] != second["truth"]["case_identity"]
    assert first["truth"]["case_identity"]["shape"] == [48, 52]
    assert second["truth"]["case_identity"]["shape"] == [48, 52]
    assert not np.array_equal(first["image"], second["image"])
    assert float(np.std(first["image"] - second["image"])) > 1.0e-4


def _pipeline_config() -> dict:
    return {
        "analysis": {
            "ridge_method": "butterfly_curvature",
            "q_window": [0.05, 1.1],
            "ellipse_multistart": 3,
            "butterfly": {"stage": "evaluate", "resamples": 0, "sensitivity": False},
        }
    }


def test_well_resolved_known_truth_reaches_pipeline_with_geometry_and_support() -> None:
    case = generate_arc_case("ellipse_ratio_400", seed=20260906, shape=(96, 96))
    config = _pipeline_config()
    # This oracle has b/a=0.4, outside the explicit flat prior's 0.35 cap.
    # Test localization accuracy with a compatible prior; retain the 5% limit.
    config["analysis"]["ellipse_preset"] = "standard"
    result = analyze_frame(
        case["image"],
        qmap=case["qmap"],
        mask=case["mask"],
        config=config,
        full2d=False,
    )

    fit = result.ellipse_fit
    truth = case["truth"]
    assert result.butterfly is not None
    assert fit["success"] is True
    for key in ("a", "b", "axis_ratio"):
        assert abs(float(fit[key]) - float(truth[key])) / float(truth[key]) < 0.05, (key, fit[key], truth[key])
    angle_error = abs((float(fit["theta_deg"]) - 17.0 + 90.0) % 180.0 - 90.0)
    assert angle_error < 5.0

    accepted = [point for point in result.butterfly["points"] if point.get("accepted")]
    side_counts = Counter((point.get("branch_id"), point.get("side")) for point in accepted)
    assert set(side_counts) >= {
        (0, "upper"),
        (0, "lower"),
        (1, "upper"),
        (1, "lower"),
    }
    assert min(side_counts[key] for key in ((0, "upper"), (0, "lower"), (1, "upper"), (1, "lower"))) >= 3


def test_production_geometry_maps_and_nontrivial_mask_reach_analyze_frame() -> None:
    case = generate_arc_case("ellipse_ratio_400", seed=20260906, shape=(64, 68))
    source_qmap = case["qmap"]
    valid_mask = np.ones(source_qmap.shape, dtype=bool)
    valid_mask[4:9, 18:30] = False
    valid_mask[40:48, 42:50] = False
    geometry = GeometryMaps(
        q_nm_inv=source_qmap.q,
        chi_rad=np.arctan2(source_qmap.qy, source_qmap.qx),
        qx_nm_inv=source_qmap.qx,
        qy_nm_inv=source_qmap.qy,
        valid_mask=valid_mask,
        metadata={"test_marker": "review_oracle"},
        fingerprint="review-oracle-geometry",
    )

    result = analyze_frame(
        case["image"],
        qmap=geometry,
        config=_pipeline_config(),
        full2d=False,
    )

    assert result.ellipse_fit["success"] is True
    np.testing.assert_array_equal(result.analysis_domain.detector_valid_mask, valid_mask)
    assert not np.any(result.analysis_domain.fit_valid_mask & ~valid_mask)
    assert np.count_nonzero(~valid_mask) > 0


def test_resampling_callbacks_receive_changed_full_shape_images_and_drive_refits() -> None:
    case = generate_arc_case("ellipse_ratio_400", seed=11, shape=(36, 40))
    weights = np.arange(case["image"].size, dtype=float).reshape(case["image"].shape)
    seen: list[tuple[np.ndarray, float]] = []

    def callback(image: np.ndarray, options: dict) -> dict:
        array = np.asarray(image, dtype=float)
        assert array.shape == case["image"].shape
        assert tuple(options["image_shape"]) == array.shape
        checksum = float(np.sum(array * weights))
        seen.append((array.copy(), checksum))
        a = 0.7 + checksum * 1.0e-6
        b = 0.2 + checksum * 1.0e-6
        return {
            "candidate_fit": {
                "a": a,
                "b": b,
                "axis_ratio": b / a,
                "theta_deg": 10.0 + checksum * 1.0e-4,
                "success": True,
            }
        }

    result = resample_butterfly(
        case["image"],
        case["qmap"],
        mask=case["mask"],
        q_window=(0.1, 1.0),
        refit=callback,
        options={"resamples": 8, "seed": 91, "block_shape": (4, 5)},
    )

    assert result["successes"] == 8
    assert len(seen) == 8
    assert all(array.shape == case["image"].shape for array, _ in seen)
    assert len({array.tobytes() for array, _ in seen}) > 1
    checksums = np.asarray([checksum for _, checksum in seen], dtype=float)
    assert float(np.ptp(checksums)) > 1.0
    fitted_a = np.asarray([draw["fit"]["a"] for draw in result["draws"]], dtype=float)
    assert float(np.ptp(fitted_a)) > 1.0e-5
    np.testing.assert_allclose(fitted_a, 0.7 + checksums * 1.0e-6)
