"""P4 engineering runner for ridge, lobe and symmetric-ellipse evidence.

This runner uses fixed suite-level settings.  It never changes raw inputs and
never promotes provisional thresholds or unreviewed R0 frames to scientific
acceptance.  The purpose is to make the important P4 path executable and
auditable before the external P3 evidence is complete.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from .benchmark_t1 import DEFAULT_CASE_NAMES as T1_DEFAULT_CASE_NAMES
from .observables import ellipse_radius
from .pipeline import analyze_frame


SCHEMA_VERSION = "lamellarsaxs2d.p4_engineering.v1"
Progress = Callable[[str], None]
_T2_REQUIRED_CASES = ("2-point", "eyebrow", "butterfly", "non_elliptical")
_T1_ANALYSIS_CONFIG = {
    "q_window_nm_inv": [0.15, 1.25],
    "n_angular_bins": 90,
    "n_ridge_angles": 72,
    "n_radial_bins": 128,
}
_T2_ANALYSIS_CONFIG = {
    "q_window_nm_inv": [0.30, 0.80],
    "n_angular_bins": 90,
    "n_ridge_angles": 72,
    "n_radial_bins": 128,
}
_R0_ANALYSIS_CONFIG = {
    "q_window_nm_inv": [0.10, 2.00],
    "n_angular_bins": 90,
    "n_ridge_angles": 72,
    "n_radial_bins": 128,
    "full2d": False,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON root must be an object: {path}")
    return dict(value)


def _required_manifest_entries(
    manifest: Mapping[str, Any],
    expected_case_ids: Sequence[str],
    *,
    suite: str,
) -> list[dict[str, Any]]:
    raw_entries = manifest.get("cases")
    if not isinstance(raw_entries, list) or not all(
        isinstance(entry, Mapping) for entry in raw_entries
    ):
        raise ValueError(f"{suite} manifest must contain a cases list")
    entries = [dict(entry) for entry in raw_entries]
    case_ids = [str(entry.get("case_id", "")) for entry in entries]
    expected = list(expected_case_ids)
    if len(case_ids) != len(set(case_ids)):
        raise ValueError(f"{suite} manifest contains duplicate case IDs")
    if set(case_ids) != set(expected) or len(case_ids) != len(expected):
        raise ValueError(
            f"{suite} manifest case IDs must be exactly {sorted(expected)}; got {sorted(case_ids)}"
        )
    return entries


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _periodic_error_deg(left: float, right: float) -> float:
    return float(abs((left - right + 180.0) % 360.0 - 180.0))


def _assigned_periodic_errors(
    measured: Sequence[float], truth: Sequence[float]
) -> list[float]:
    if not measured or not truth:
        return []
    cost = np.asarray(
        [[_periodic_error_deg(float(left), float(right)) for right in truth] for left in measured],
        dtype=float,
    )
    rows, columns = linear_sum_assignment(cost)
    return [float(cost[row, column]) for row, column in zip(rows, columns)]


def _case_arrays(path: Path) -> dict[str, np.ndarray | str]:
    with np.load(path, allow_pickle=False) as archive:
        required = ("intensity", "qx", "qy", "q", "valid_mask", "q_unit")
        missing = [name for name in required if name not in archive.files]
        if missing:
            raise ValueError(f"P4 case {path} is missing arrays: {missing}")
        return {
            "intensity": np.asarray(archive["intensity"], dtype=float),
            "qx": np.asarray(archive["qx"], dtype=float),
            "qy": np.asarray(archive["qy"], dtype=float),
            "q": np.asarray(archive["q"], dtype=float),
            "valid_mask": np.asarray(archive["valid_mask"], dtype=bool),
            "q_unit": str(archive["q_unit"]),
            **(
                {"projection_reference": np.asarray(archive["projection_reference"], dtype=float)}
                if "projection_reference" in archive.files
                else {}
            ),
        }


def _analyze_case(
    arrays: Mapping[str, Any],
    *,
    q_window: tuple[float, float],
    ridge_method: str,
) -> Any:
    qmap = {
        "qx": arrays["qx"],
        "qy": arrays["qy"],
        "q": arrays["q"],
        "q_unit": arrays["q_unit"],
    }
    return analyze_frame(
        arrays["intensity"],
        qmap=qmap,
        valid_mask=arrays["valid_mask"],
        config={
            "q_window": q_window,
            "n_angular_bins": 90,
            "n_ridge_angles": 72,
            "n_radial_bins": 128,
            "ridge_method": ridge_method,
        },
        full2d=False,
    )


def _ridge_rows(result: Any) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in result.ridges
        if isinstance(row, Mapping) and bool(row.get("valid", row.get("accepted", True)))
    ]


def _t1_metrics(result: Any, truth: Mapping[str, Any]) -> dict[str, Any]:
    parameters = dict(truth.get("truth_parameters", truth.get("parameters", {})))
    a = float(parameters["a"])
    b = float(parameters["b"])
    theta = float(parameters["theta"])
    rows = _ridge_rows(result)
    errors_q: list[float] = []
    for row in rows:
        angle_deg = _finite(row.get("angle_deg", row.get("theta_deg")))
        observed_q = _finite(row.get("q"))
        if angle_deg is None or observed_q is None:
            continue
        angle = np.deg2rad(angle_deg)
        plus = float(ellipse_radius(angle, a, b, theta))
        minus = float(ellipse_radius(angle, a, b, -theta))
        errors_q.append(min(abs(observed_q - plus), abs(observed_q - minus)))
    spacing = np.asarray(truth.get("q_spacing", ()), dtype=float)
    q_step = float(np.median(spacing[np.isfinite(spacing) & (spacing > 0)])) if spacing.size else float("nan")
    errors_px = np.asarray(errors_q, dtype=float) / q_step if np.isfinite(q_step) and q_step > 0 else np.asarray([])

    lobes = result.observables.get("lobes", [])
    measured_lobes = [
        float(item["angle_deg"])
        for item in lobes
        if isinstance(item, Mapping)
        and bool(item.get("valid", True))
        and _finite(item.get("angle_deg")) is not None
    ]
    truth_lobes = [float(value) for value in truth.get("ridge_truth", {}).get("lobe_angles_deg", ())]
    lobe_errors = _assigned_periodic_errors(measured_lobes, truth_lobes)

    ellipse = result.ellipse_fit
    fit_a = _finite(ellipse.get("a"))
    fit_b = _finite(ellipse.get("b"))
    fit_theta = _finite(ellipse.get("theta_deg"))
    fit_cx = _finite(ellipse.get("parameters", {}).get("center_qx"))
    fit_cy = _finite(ellipse.get("parameters", {}).get("center_qy"))
    center_q = float(np.hypot(fit_cx, fit_cy)) if fit_cx is not None and fit_cy is not None else None
    matched = int(np.count_nonzero(errors_px <= 1.0)) if errors_px.size else 0
    denominator = 72
    return {
        "ridge_median_error_px": float(np.median(errors_px)) if errors_px.size else None,
        "ridge_p95_error_px": float(np.percentile(errors_px, 95.0)) if errors_px.size else None,
        "ridge_f1_at_1px": float(2.0 * matched / (len(rows) + denominator)) if rows else 0.0,
        "lobe_periodic_angle_error_deg": max(lobe_errors) if lobe_errors else None,
        "valid_lobe_count": len(measured_lobes),
        "truth_lobe_count": len(truth_lobes),
        "lobe_count_complete": len(measured_lobes) == len(truth_lobes),
        "ellipse_a_relative_error": abs(fit_a - a) / a if fit_a is not None else None,
        "ellipse_b_relative_error": abs(fit_b - b) / b if fit_b is not None else None,
        "ellipse_theta_periodic_error_deg": (
            min(abs(fit_theta - abs(np.degrees(theta))), abs(fit_theta + abs(np.degrees(theta))))
            if fit_theta is not None
            else None
        ),
        "ellipse_center_equivalent_pixel_error": (
            center_q / q_step
            if center_q is not None and np.isfinite(q_step) and q_step > 0
            else None
        ),
        "valid_ridge_count": len(rows),
    }


def _t1_expected_rejection(truth: Mapping[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if bool(truth.get("non_elliptic_negative")):
        reasons.append("non_elliptic_negative")
    if bool(truth.get("low_snr")):
        reasons.append("low_snr")
    parameters = truth.get("truth_parameters", truth.get("parameters", {}))
    if isinstance(parameters, Mapping):
        phi = _finite(parameters.get("lobe_angle"))
        sigma = _finite(parameters.get("angular_width"))
        if phi is not None and sigma is not None:
            lobe_separation = 2.0 * abs(phi)
            lobe_fwhm = 2.0 * np.sqrt(2.0 * np.log(2.0)) * abs(sigma)
            if lobe_separation <= lobe_fwhm:
                reasons.append("overlapping_lobes_unresolved")
    return bool(reasons), reasons


def _threshold_check(value: Any, *, maximum: float | None = None, minimum: float | None = None) -> bool:
    number = _finite(value)
    if number is None:
        return False
    if maximum is not None and number > maximum:
        return False
    if minimum is not None and number < minimum:
        return False
    return True


def _t1_case_passes(
    truth: Mapping[str, Any],
    metrics: Mapping[str, Any],
    quality_status: str,
    thresholds: Mapping[str, Any],
) -> bool:
    expected_rejection, _rejection_reasons = _t1_expected_rejection(truth)
    if expected_rejection:
        return quality_status == "FAIL" or int(metrics.get("valid_lobe_count", 0)) == 0
    return quality_status != "FAIL" and all(
        (
            _threshold_check(metrics.get("ridge_median_error_px"), maximum=float(thresholds["ridge_detector_median_error_px_max"])),
            _threshold_check(metrics.get("ridge_p95_error_px"), maximum=float(thresholds["ridge_detector_p95_error_px_max"])),
            _threshold_check(metrics.get("ridge_f1_at_1px"), minimum=float(thresholds["ridge_f1_min"])),
            bool(metrics.get("lobe_count_complete")),
            _threshold_check(metrics.get("lobe_periodic_angle_error_deg"), maximum=float(thresholds["lobe_periodic_angle_error_deg_max"])),
            _threshold_check(metrics.get("ellipse_a_relative_error"), maximum=float(thresholds["ellipse_a_relative_error_max"])),
            _threshold_check(metrics.get("ellipse_b_relative_error"), maximum=float(thresholds["ellipse_b_relative_error_max"])),
            _threshold_check(metrics.get("ellipse_theta_periodic_error_deg"), maximum=float(thresholds["ellipse_theta_periodic_error_deg_max"])),
            _threshold_check(metrics.get("ellipse_center_equivalent_pixel_error"), maximum=float(thresholds["ellipse_center_equivalent_pixel_error_max"])),
        )
    )


def _t2_metrics(result: Any, projection_reference: np.ndarray) -> dict[str, Any]:
    rows = _ridge_rows(result)
    detected = np.asarray(
        [
            (float(row["qx"]), float(row["qy"]))
            for row in rows
            if _finite(row.get("qx")) is not None and _finite(row.get("qy")) is not None
        ],
        dtype=float,
    )
    if detected.size == 0 or projection_reference.size == 0:
        return {
            "ridge_median_error_q": None,
            "ridge_p95_error_q": None,
            "ridge_error_local_fwhm_fraction": None,
            "truth_reference_median_error_q": None,
            "truth_reference_p95_error_q": None,
            "valid_ridge_count": int(len(detected)),
        }
    reference = np.asarray(projection_reference, dtype=float)
    distances, _ = cKDTree(reference).query(detected)
    reverse_distances, _ = cKDTree(detected).query(reference)
    widths = np.asarray(
        [
            float(row["radial_fwhm"])
            for row in rows
            if _finite(row.get("radial_fwhm")) is not None and float(row["radial_fwhm"]) > 0
        ],
        dtype=float,
    )
    median_width = float(np.median(widths)) if widths.size else None
    return {
        "ridge_median_error_q": float(np.median(distances)),
        "ridge_p95_error_q": float(np.percentile(distances, 95.0)),
        "ridge_error_local_fwhm_fraction": (
            float(np.median(distances) / median_width)
            if median_width is not None and median_width > 0
            else None
        ),
        "truth_reference_median_error_q": float(np.median(reverse_distances)),
        "truth_reference_p95_error_q": float(np.percentile(reverse_distances, 95.0)),
        "valid_ridge_count": int(detected.shape[0]),
    }


def _fit_summary(result: Any) -> dict[str, Any]:
    ellipse = result.ellipse_fit
    quality = dict(ellipse.get("quality", {}) or {})
    return {
        "status": ellipse.get("status"),
        "solver_status": ellipse.get("solver_status"),
        "quality_status": ellipse.get("quality_status"),
        "a": ellipse.get("a"),
        "b": ellipse.get("b"),
        "theta_deg": ellipse.get("theta_deg"),
        "axis_ratio": ellipse.get("axis_ratio"),
        "rmse": ellipse.get("rmse"),
        "condition": ellipse.get("condition"),
        "coverage": ellipse.get("coverage"),
        "bound_flags": ellipse.get("bound_flags"),
        "branch_counts": ellipse.get("branch_counts"),
        "multistart_count": ellipse.get("multistart_count"),
        "selected_start_index": ellipse.get("selected_start_index"),
        "quality": quality,
    }


def _run_t1(
    manifest_path: Path,
    thresholds: Mapping[str, Any],
    *,
    ridge_method: str,
    progress: Progress,
) -> dict[str, Any]:
    manifest = _read_json(manifest_path)
    cases: list[dict[str, Any]] = []
    entries = _required_manifest_entries(
        manifest,
        T1_DEFAULT_CASE_NAMES,
        suite="T1",
    )
    for entry in entries:
        case_id = str(entry["case_id"])
        progress(f"P4 T1: {case_id}")
        npz_path = manifest_path.parent / str(entry.get("npz_file", entry.get("npz")))
        truth_path = manifest_path.parent / str(entry.get("truth_json", entry.get("truth_file")))
        truth = _read_json(truth_path)
        arrays = _case_arrays(npz_path)
        result = _analyze_case(arrays, q_window=(0.15, 1.25), ridge_method=ridge_method)
        metrics = _t1_metrics(result, truth)
        quality_status = str(result.ellipse_fit.get("quality_status") or "FAIL")
        expected_rejection, rejection_reasons = _t1_expected_rejection(truth)
        passed = _t1_case_passes(truth, metrics, quality_status, thresholds)
        cases.append(
            {
                "case_id": case_id,
                "category": entry.get("category"),
                "expected_rejection": expected_rejection,
                "expected_rejection_reasons": rejection_reasons,
                "passed": passed,
                "metrics": metrics,
                "fit": _fit_summary(result),
                "input_sha256": _sha256(npz_path),
            }
        )
    passed_count = sum(bool(case["passed"]) for case in cases)
    return {
        "status": "PASS" if passed_count == len(cases) else "FAIL",
        "passed_count": passed_count,
        "case_count": len(cases),
        "cases": cases,
    }


def _run_t2(
    manifest_path: Path,
    thresholds: Mapping[str, Any],
    *,
    ridge_method: str,
    progress: Progress,
) -> dict[str, Any]:
    manifest = _read_json(manifest_path)
    cases: list[dict[str, Any]] = []
    maximum_fraction = float(thresholds["ridge_error_local_fwhm_fraction_max"])
    entries = _required_manifest_entries(
        manifest,
        _T2_REQUIRED_CASES,
        suite="T2",
    )
    for entry in entries:
        case_id = str(entry["case_id"])
        category = str(entry.get("category", case_id))
        progress(f"P4 T2: {case_id}")
        npz_path = manifest_path.parent / str(entry["npz_file"])
        arrays = _case_arrays(npz_path)
        result = _analyze_case(arrays, q_window=(0.30, 0.80), ridge_method=ridge_method)
        metrics = _t2_metrics(result, np.asarray(arrays.get("projection_reference", []), dtype=float))
        quality_status = str(result.ellipse_fit.get("quality_status") or "FAIL")
        expected_rejection = category in {"2-point", "non_elliptical"}
        classification_correct = (
            quality_status == "FAIL" if expected_rejection else quality_status != "FAIL"
        )
        ridge_pass = _threshold_check(
            metrics.get("ridge_error_local_fwhm_fraction"), maximum=maximum_fraction
        )
        projection_truth = entry.get("projection_truth", {})
        projection_targets = {
            "a": _finite(projection_truth.get("a")) if isinstance(projection_truth, Mapping) else None,
            "b": _finite(projection_truth.get("b")) if isinstance(projection_truth, Mapping) else None,
            "tilt_deg": (
                _finite(projection_truth.get("tilt_deg"))
                if isinstance(projection_truth, Mapping)
                else None
            ),
        }
        projection_evaluable = all(value is not None for value in projection_targets.values())
        projection_errors = {
            "a_relative_error": (
                abs(float(result.ellipse_fit.get("a")) - float(projection_targets["a"]))
                / abs(float(projection_targets["a"]))
                if projection_evaluable
                and _finite(result.ellipse_fit.get("a")) is not None
                and float(projection_targets["a"]) != 0.0
                else None
            ),
            "b_relative_error": (
                abs(float(result.ellipse_fit.get("b")) - float(projection_targets["b"]))
                / abs(float(projection_targets["b"]))
                if projection_evaluable
                and _finite(result.ellipse_fit.get("b")) is not None
                and float(projection_targets["b"]) != 0.0
                else None
            ),
            "tilt_error_deg": (
                _periodic_error_deg(
                    float(result.ellipse_fit.get("theta_deg")),
                    float(projection_targets["tilt_deg"]),
                )
                if projection_evaluable
                and _finite(result.ellipse_fit.get("theta_deg")) is not None
                else None
            ),
        }
        projection_pass = projection_evaluable and all(
            (
                _threshold_check(
                    projection_errors["a_relative_error"],
                    maximum=float(thresholds["projection_a_relative_error_max"]),
                ),
                _threshold_check(
                    projection_errors["b_relative_error"],
                    maximum=float(thresholds["projection_b_relative_error_max"]),
                ),
                _threshold_check(
                    projection_errors["tilt_error_deg"],
                    maximum=float(thresholds["projection_tilt_error_deg_max"]),
                ),
            )
        )
        passed = classification_correct and (
            expected_rejection or (ridge_pass and projection_pass)
        )
        cases.append(
            {
                "case_id": case_id,
                "category": category,
                "expected_rejection": expected_rejection,
                "classification_correct": classification_correct,
                "projection_thresholds_evaluable": projection_evaluable,
                "projection_targets": projection_targets,
                "projection_errors": projection_errors,
                "projection_thresholds_passed": projection_pass,
                "passed": passed,
                "metrics": metrics,
                "fit": _fit_summary(result),
                "input_sha256": _sha256(npz_path),
            }
        )
    passed_count = sum(bool(case["passed"]) for case in cases)
    classification_accuracy = (
        sum(bool(case["classification_correct"]) for case in cases) / len(cases)
        if cases
        else 0.0
    )
    classification_threshold_passed = _threshold_check(
        classification_accuracy,
        minimum=float(thresholds["pattern_class_accuracy_min"]),
    )
    projection_contract_complete = all(
        bool(case["expected_rejection"])
        or bool(case["projection_thresholds_evaluable"])
        for case in cases
    )
    suite_passed = (
        passed_count == len(cases)
        and classification_threshold_passed
        and projection_contract_complete
    )
    return {
        "status": "PASS" if suite_passed else "FAIL",
        "passed_count": passed_count,
        "case_count": len(cases),
        "pattern_class_accuracy": classification_accuracy,
        "pattern_class_accuracy_threshold_passed": classification_threshold_passed,
        "projection_threshold_contract_complete": projection_contract_complete,
        "projection_threshold_contract_reason": (
            None
            if projection_contract_complete
            else "current T2 projection truth has no explicit a/b/tilt targets"
        ),
        "cases": cases,
    }


def _load_r0_rows(manifest_path: Path) -> list[dict[str, str]]:
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    expected = [f"blind_{index:03d}" for index in range(1, 9)]
    if [row.get("blind_id") for row in rows] != expected:
        raise ValueError("R0 P4 manifest must contain blind_001 through blind_008 in order")
    return rows


def _run_r0(
    package: Path,
    manifest_path: Path,
    poni: Path,
    mask: Path,
    *,
    ridge_method: str,
    progress: Progress,
) -> dict[str, Any]:
    package = package.resolve()
    poni_before = _sha256(poni)
    mask_before = _sha256(mask)
    frames: list[dict[str, Any]] = []
    for row in _load_r0_rows(manifest_path):
        blind_id = str(row["blind_id"])
        relative = row.get("source_path_relative_package") or row.get("source_path")
        if not relative:
            raise ValueError(f"R0 row {blind_id} has no source path")
        source = (package / relative).resolve()
        if not source.is_relative_to(package):
            raise ValueError(f"R0 source escapes package root: {source}")
        before = _sha256(source)
        declared = str(row.get("sha256", "") or "").lower()
        if declared and declared != before.lower():
            raise ValueError(f"R0 manifest hash mismatch for {blind_id}: {source}")
        progress(f"P4 R0: {blind_id}")
        result = analyze_frame(
            source,
            poni=poni,
            mask=mask,
            config={
                "q_window": (0.10, 2.00),
                "n_angular_bins": 90,
                "n_ridge_angles": 72,
                "n_radial_bins": 128,
                "ridge_method": ridge_method,
            },
            full2d=False,
        )
        after = _sha256(source)
        if before != after:
            raise RuntimeError(f"raw R0 input changed during P4 analysis: {source}")
        lobe_rows = [
            {
                "angle_deg": item.get("angle_deg"),
                "valid": bool(item.get("valid", False)),
                "reason": item.get("reason"),
                "coverage": item.get("coverage"),
                "snr": item.get("snr"),
                "fwhm_deg": item.get("fwhm_deg"),
                "refinement": item.get("refinement"),
            }
            for item in result.observables.get("lobes", [])
            if isinstance(item, Mapping)
        ]
        frames.append(
            {
                "blind_id": blind_id,
                "role": row.get("role"),
                "source_path_relative_package": relative.replace("\\", "/"),
                "input_sha256_before": before,
                "input_sha256_after": after,
                "input_unchanged": True,
                "valid_ridge_count": len(_ridge_rows(result)),
                "valid_lobe_count": sum(bool(item["valid"]) for item in lobe_rows),
                "lobes": lobe_rows,
                "fit": _fit_summary(result),
            }
        )
    counts = {
        status: sum(frame["fit"].get("quality_status") == status for frame in frames)
        for status in ("PASS", "WARN", "FAIL")
    }
    poni_after = _sha256(poni)
    mask_after = _sha256(mask)
    if poni_before != poni_after or mask_before != mask_after:
        raise RuntimeError("PONI or mask changed during P4 analysis")
    engineering_status = (
        "FAIL"
        if counts["FAIL"]
        else ("WARN" if counts["WARN"] else "PASS")
    )
    return {
        "status": engineering_status,
        "scientific_status": "NOT_ACCEPTED",
        "frame_count": len(frames),
        "quality_counts": counts,
        "support_files_unchanged": True,
        "poni_sha256_before": poni_before,
        "poni_sha256_after": poni_after,
        "mask_sha256_before": mask_before,
        "mask_sha256_after": mask_after,
        "frames": frames,
    }


def _sensitivity(
    t1_manifest: Path,
    *,
    progress: Progress,
) -> dict[str, Any]:
    manifest = _read_json(t1_manifest)
    entry = next(
        (item for item in manifest.get("cases", []) if item.get("case_id") == "noiseless_default"),
        None,
    )
    if entry is None:
        return {"status": "NOT_RUN", "reason": "noiseless_default is absent"}
    arrays = _case_arrays(t1_manifest.parent / str(entry.get("npz_file", entry.get("npz"))))
    results: dict[str, Any] = {}
    for method in ("radial_peak", "surface_curvature"):
        progress(f"P4 sensitivity: {method}")
        result = _analyze_case(arrays, q_window=(0.15, 1.25), ridge_method=method)
        results[method] = _fit_summary(result)
    radial = results["radial_peak"]
    curvature = results["surface_curvature"]
    differences = {
        name: (
            abs(float(radial[name]) - float(curvature[name]))
            if _finite(radial.get(name)) is not None and _finite(curvature.get(name)) is not None
            else None
        )
        for name in ("a", "b", "theta_deg")
    }
    return {
        "status": "COMPARED",
        "case_id": "noiseless_default",
        "methods": results,
        "absolute_parameter_differences": differences,
        "note": "method comparison is engineering sensitivity evidence, not threshold calibration",
    }


def _write_summary_csv(report: Mapping[str, Any], destination: Path) -> None:
    rows: list[dict[str, Any]] = []
    for suite_name in ("t1", "t2"):
        for case in report[suite_name]["cases"]:
            rows.append(
                {
                    "suite": suite_name.upper(),
                    "case_id": case["case_id"],
                    "passed": case["passed"],
                    "quality_status": case["fit"].get("quality_status"),
                    "valid_ridge_count": case["metrics"].get("valid_ridge_count"),
                    "valid_lobe_count": case["metrics"].get("valid_lobe_count"),
                    "a": case["fit"].get("a"),
                    "b": case["fit"].get("b"),
                    "theta_deg": case["fit"].get("theta_deg"),
                }
            )
    for frame in report.get("r0", {}).get("frames", []):
        rows.append(
            {
                "suite": "R0",
                "case_id": frame["blind_id"],
                "passed": "NOT_ACCEPTED",
                "quality_status": frame["fit"].get("quality_status"),
                "valid_ridge_count": frame.get("valid_ridge_count"),
                "valid_lobe_count": frame.get("valid_lobe_count"),
                "a": frame["fit"].get("a"),
                "b": frame["fit"].get("b"),
                "theta_deg": frame["fit"].get("theta_deg"),
            }
        )
    fields = (
        "suite",
        "case_id",
        "passed",
        "quality_status",
        "valid_ridge_count",
        "valid_lobe_count",
        "a",
        "b",
        "theta_deg",
    )
    with destination.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_p4_engineering(
    *,
    t1_manifest: str | Path,
    t2_manifest: str | Path,
    thresholds: str | Path,
    output: str | Path,
    r0_package: str | Path | None = None,
    r0_manifest: str | Path | None = None,
    poni: str | Path | None = None,
    mask: str | Path | None = None,
    ridge_method: str = "radial_peak",
    run_sensitivity: bool = True,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Run fixed P4 engineering evidence and write JSON/CSV into a new directory."""

    progress = progress or (lambda _message: None)
    ridge_method = str(ridge_method).lower().replace("-", "_")
    if ridge_method not in {"radial_peak", "surface_curvature"}:
        raise ValueError("ridge_method must be 'radial_peak' or 'surface_curvature'")
    t1_path = Path(t1_manifest).resolve()
    t2_path = Path(t2_manifest).resolve()
    threshold_path = Path(thresholds).resolve()
    output_path = Path(output).resolve()
    for path in (t1_path, t2_path, threshold_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output_path.exists():
        raise FileExistsError(f"P4 output already exists; choose a new directory: {output_path}")

    real_arguments = (r0_package, r0_manifest, poni, mask)
    resolved_r0: tuple[Path, Path, Path, Path] | None = None
    if any(value is not None for value in real_arguments):
        if not all(value is not None for value in real_arguments):
            raise ValueError("R0 requires package, manifest, PONI and mask together")
        package_path = Path(r0_package).resolve()  # type: ignore[arg-type]
        manifest_path = Path(r0_manifest).resolve()  # type: ignore[arg-type]
        poni_path = Path(poni).resolve()  # type: ignore[arg-type]
        mask_path = Path(mask).resolve()  # type: ignore[arg-type]
        if output_path.is_relative_to(package_path):
            raise ValueError("P4 output must not be written inside the raw R0 package")
        if not package_path.is_dir():
            raise FileNotFoundError(package_path)
        for path in (manifest_path, poni_path, mask_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        resolved_r0 = (package_path, manifest_path, poni_path, mask_path)

    threshold_document = _read_json(threshold_path)
    _required_manifest_entries(
        _read_json(t1_path),
        T1_DEFAULT_CASE_NAMES,
        suite="T1",
    )
    _required_manifest_entries(
        _read_json(t2_path),
        _T2_REQUIRED_CASES,
        suite="T2",
    )
    t1_thresholds = threshold_document.get("t1_high_snr")
    t2_thresholds = threshold_document.get("t2_independent")
    if not isinstance(t1_thresholds, Mapping) or not isinstance(t2_thresholds, Mapping):
        raise ValueError("threshold document must contain t1_high_snr and t2_independent")

    output_path.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "stage": "P4",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "ridge_method": ridge_method,
        "analysis_config": {
            "t1": _T1_ANALYSIS_CONFIG,
            "t2": _T2_ANALYSIS_CONFIG,
            "r0": _R0_ANALYSIS_CONFIG,
            "ridge_method": ridge_method,
        },
        "source_provenance": {
            name: {
                "path": path.as_posix(),
                "sha256": _sha256(path),
            }
            for name, path in {
                "p4_validation": Path(__file__).resolve(),
                "observables": Path(__file__).with_name("observables.py").resolve(),
                "ellipse": Path(__file__).with_name("ellipse.py").resolve(),
                "pipeline": Path(__file__).with_name("pipeline.py").resolve(),
                "p4_quality": Path(__file__).with_name("p4_quality.py").resolve(),
            }.items()
        },
        "scientific_status": "NOT_ACCEPTED",
        "scientific_reason": "P3 human consensus and frozen evidence-backed thresholds are incomplete",
        "thresholds": {
            "path": threshold_path.as_posix(),
            "sha256": _sha256(threshold_path),
            "version": threshold_document.get("thresholds_version"),
            "status": threshold_document.get("status"),
            "frozen": bool(threshold_document.get("frozen", False)),
            "usable_for_final_pass_fail": bool(
                threshold_document.get("usable_for_final_pass_fail", False)
            ),
        },
        "inputs": {
            "t1_manifest": {"path": t1_path.as_posix(), "sha256": _sha256(t1_path)},
            "t2_manifest": {"path": t2_path.as_posix(), "sha256": _sha256(t2_path)},
        },
    }
    report["t1"] = _run_t1(
        t1_path, t1_thresholds, ridge_method=ridge_method, progress=progress
    )
    report["t2"] = _run_t2(
        t2_path, t2_thresholds, ridge_method=ridge_method, progress=progress
    )
    report["sensitivity"] = (
        _sensitivity(t1_path, progress=progress)
        if run_sensitivity
        else {"status": "NOT_RUN", "reason": "disabled by caller"}
    )

    if resolved_r0 is not None:
        package_path, manifest_path, poni_path, mask_path = resolved_r0
        report["inputs"]["r0"] = {
            "package": package_path.as_posix(),
            "manifest": {"path": manifest_path.as_posix(), "sha256": _sha256(manifest_path)},
            "poni": {"path": poni_path.as_posix(), "sha256": _sha256(poni_path)},
            "mask": {"path": mask_path.as_posix(), "sha256": _sha256(mask_path)},
        }
        report["r0"] = _run_r0(
            package_path,
            manifest_path,
            poni_path,
            mask_path,
            ridge_method=ridge_method,
            progress=progress,
        )
    else:
        report["r0"] = {
            "status": "NOT_RUN",
            "scientific_status": "NOT_ACCEPTED",
            "reason": "R0 arguments were not supplied",
        }

    r0_incomplete = report["r0"]["status"] in {"NOT_RUN", "FAIL"}
    algorithm_fail = (
        report["t1"]["status"] == "FAIL"
        or report["t2"]["status"] == "FAIL"
        or r0_incomplete
    )
    report["engineering_status"] = "FAIL" if algorithm_fail else "PASS"
    report["p4_go_no_go"] = "NO_GO"
    report["p4_go_no_go_reasons"] = [
        "P3 human consensus is incomplete",
        "acceptance thresholds are provisional and not frozen",
        *(
            ["T1 and/or T2 engineering thresholds are not met"]
            if report["t1"]["status"] == "FAIL" or report["t2"]["status"] == "FAIL"
            else []
        ),
        *(["R0 fixed eight-frame engineering pilot is incomplete"] if r0_incomplete else []),
    ]
    report_path = output_path / "p4_engineering_report.json"
    summary_path = output_path / "p4_summary.csv"
    report["outputs"] = {
        "report_json": report_path.as_posix(),
        "summary_csv": summary_path.as_posix(),
    }
    report_path.write_text(
        json.dumps(_jsonable(report), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_summary_csv(report, summary_path)
    return _jsonable(report)


__all__ = ["SCHEMA_VERSION", "run_p4_engineering"]
