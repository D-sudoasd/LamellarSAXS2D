"""P3 evidence gate for benchmark, human annotation, and frozen thresholds.

The gate is intentionally read-only.  It does not infer human consensus,
freeze provisional thresholds, or run the analysis pipeline.  A missing
scientific evidence source therefore produces a structured No-Go result.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .annotation_pack import (
    ANNOTATION_COORDINATE_SYSTEM,
    ANNOTATION_PACK_SCHEMA_VERSION,
)
from .benchmark_t1 import DEFAULT_CASE_NAMES as T1_DEFAULT_CASE_NAMES
from .benchmark_t1 import GENERATOR_DEPENDENCY_HASHES as T1_DEPENDENCY_HASHES
from .benchmark_t1 import GENERATOR_HASH as T1_GENERATOR_HASH
from .benchmark_t1 import GENERATOR_VERSION as T1_GENERATOR_VERSION
from .benchmark_t1 import T1_Q_UNIT, T1_SCHEMA_VERSION
from .benchmark_t2 import GENERATOR_HASH as T2_GENERATOR_HASH
from .benchmark_t2 import GENERATOR_VERSION as T2_GENERATOR_VERSION
from .benchmark_t2 import T2_Q_UNIT
from .benchmark_t2 import generate_case as generate_t2_case


P3_GATE_SCHEMA_VERSION = "lamellarsaxs2d.p3_gate.v3"
_T2_REQUIRED_CATEGORIES = {"2-point", "eyebrow", "butterfly", "non_elliptical"}
_COMPLETE_ANNOTATION_STATES = {"complete", "completed", "consensus_complete", "accepted"}
_BLIND_IDS = {f"blind_{index:03d}" for index in range(1, 9)}
_UNKNOWN_TOKENS = {"unknown", "not_visible", "not_applicable"}
_NONDECISIVE_TOKENS = _UNKNOWN_TOKENS | {"pending", "awaiting", "not_reviewed"}
_SUPPORTED_AGGREGATIONS = {"mean", "median", "p95", "max", "min"}
_ANNOTATION_CONTENT_FIELDS = (
    "valid_area",
    "beamstop",
    "streak",
    "overlap",
    "lobe_center_x",
    "lobe_center_y",
    "ridge_points",
    "software",
    "software_version",
    "coordinate_system",
    "image_version",
    "annotation_time",
    "annotator",
)
_CONSENSUS_CONTENT_FIELDS = (
    "consensus_status",
    "valid_area",
    "beamstop",
    "streak",
    "overlap",
    "lobe_center_x",
    "lobe_center_y",
    "ridge_points",
    "reviewer",
    "software",
    "software_version",
    "coordinate_system",
    "image_version",
    "review_time",
)

# These are the numeric contracts consumed by P4.  Keeping the schema here
# prevents a frozen metadata envelope from being mistaken for usable numeric
# acceptance evidence.  The values are deliberately conservative: ranges
# reject malformed documents but never rewrite or loosen a threshold.
_NUMERIC_THRESHOLD_FIELDS: dict[str, dict[str, tuple[float | None, float | None, str]]] = {
    "t1_high_snr": {
        "ridge_detector_median_error_px_max": (0.0, None, "number"),
        "ridge_detector_p95_error_px_max": (0.0, None, "number"),
        "ridge_f1_min": (0.0, 1.0, "number_open"),
        "lobe_periodic_angle_error_deg_max": (0.0, 180.0, "number"),
        "ellipse_a_relative_error_max": (0.0, None, "number"),
        "ellipse_b_relative_error_max": (0.0, None, "number"),
        "ellipse_theta_periodic_error_deg_max": (0.0, 180.0, "number"),
        "ellipse_center_equivalent_pixel_error_max": (0.0, None, "number"),
    },
    "t2_independent": {
        "ridge_error_local_fwhm_fraction_max": (0.0, None, "number"),
        "pattern_class_accuracy_min": (0.0, 1.0, "number_open"),
        "projection_a_relative_error_max": (0.0, None, "number"),
        "projection_b_relative_error_max": (0.0, None, "number"),
        "projection_tilt_error_deg_max": (0.0, 180.0, "number"),
    },
    "full2d_quality": {
        "scaled_condition_pass_lt": (0.0, None, "number"),
        "scaled_condition_warn_lt_or_equal": (0.0, None, "number"),
        "scaled_condition_fail_gt": (0.0, None, "number"),
    },
    "uncertainty": {
        "repeats_per_representative_condition_min": (1.0, None, "integer"),
        "repeats_per_representative_condition_max": (1.0, None, "integer"),
        "interval_level": (0.0, 1.0, "number_open"),
        "empirical_coverage_min": (0.0, 1.0, "number_open"),
        "empirical_coverage_max": (0.0, 1.0, "number"),
        "false_pass_rate_max": (0.0, 1.0, "number"),
    },
    "real_data": {
        "ridge_f1_min": (0.0, 1.0, "number_open"),
        "lobe_periodic_angle_error_deg_max": (0.0, 180.0, "number"),
        "repeat_frame_apparent_parameter_cv_max": (0.0, None, "number"),
        "pilot_frame_count": (1.0, None, "integer"),
        "pilot_difficult_or_negative_count_min": (0.0, None, "integer_open"),
        "holding_sequence_denominator": (1.0, None, "integer"),
        "usable_fraction_min": (0.0, 1.0, "number_open"),
    },
}
_BOOLEAN_THRESHOLD_FIELDS: dict[str, tuple[str, ...]] = {
    "t1_high_snr": ("same_seed_deterministic",),
    "t2_independent": ("structure_truth_is_not_empirical_inverse_truth",),
    "full2d_quality": (
        "nonfinite_condition_fails",
        "critical_bound_hit_fails",
        "withheld_failure_fails",
        "structured_residual_fails",
    ),
    "uncertainty": ("statistical_and_selection_uncertainty_separate",),
    "real_data": (
        "independent_warm_start_difference_within_combined_uncertainty",
        "forward_reverse_systematic_bias_allowed",
        "resume_matches_continuous",
    ),
}


def _numeric_threshold_contract(
    threshold: Mapping[str, Any],
    *,
    require_final_blocks: bool,
) -> dict[str, Any]:
    """Validate the complete typed P3 threshold content and its references."""

    required_blocks = tuple(_NUMERIC_THRESHOLD_FIELDS)
    evidence_sources = threshold.get("evidence_sources")
    evidence_names = set(evidence_sources) if isinstance(evidence_sources, Mapping) else set()
    content = {name: threshold.get(name) for name in required_blocks}
    content_digest = hashlib.sha256(
        json.dumps(
            content,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    recorded_digest = threshold.get("threshold_content_sha256")
    content_bound = isinstance(recorded_digest, str) and recorded_digest.casefold() == content_digest
    result: dict[str, Any] = {
        "required_blocks": list(required_blocks),
        "blocks": {},
        "content_sha256": content_digest,
        "recorded_content_sha256": recorded_digest,
        "content_bound": content_bound,
    }
    all_valid = True
    for block_name in required_blocks:
        block = threshold.get(block_name)
        errors: list[str] = []
        if not isinstance(block, Mapping):
            errors.append("missing_or_not_mapping")
            result["blocks"][block_name] = {"valid": False, "errors": errors}
            all_valid = False
            continue
        refs = block.get("evidence_refs")
        if not isinstance(refs, list) or not refs or not all(
            isinstance(item, str) and item.strip() for item in refs
        ):
            errors.append("evidence_refs_missing_or_empty")
        else:
            unknown_refs = sorted(
                set(str(item).strip() for item in refs).difference(
                    evidence_names | {"fixed_contracts", "evidence_record_contract"}
                )
            )
            if unknown_refs:
                errors.append(f"unknown_evidence_refs:{','.join(unknown_refs)}")
        for field, (minimum, maximum, kind) in _NUMERIC_THRESHOLD_FIELDS[block_name].items():
            value = block.get(field)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
                errors.append(f"{field}:not_numeric")
                continue
            number = float(value)
            if not math.isfinite(number):
                errors.append(f"{field}:nonfinite")
                continue
            if kind in {"integer", "integer_open"} and (
                not float(number).is_integer() or int(number) != number
            ):
                errors.append(f"{field}:not_integer")
            if minimum is not None and (
                number <= minimum
                if kind in {"number_open", "integer_open"}
                else number < minimum
            ):
                errors.append(f"{field}:below_minimum")
            if maximum is not None and number > maximum:
                errors.append(f"{field}:above_maximum")
        for field in _BOOLEAN_THRESHOLD_FIELDS[block_name]:
            if not isinstance(block.get(field), bool):
                errors.append(f"{field}:not_bool")
        # The full2d condition ladder and paired bounds must be ordered.  A
        # frozen but contradictory ladder cannot be consumed by P4.
        if block_name == "full2d_quality" and not errors:
            if not (
                float(block["scaled_condition_pass_lt"])
                <= float(block["scaled_condition_warn_lt_or_equal"])
                <= float(block["scaled_condition_fail_gt"])
            ):
                errors.append("condition_thresholds_not_ordered")
        if block_name == "uncertainty" and not errors:
            if int(block["repeats_per_representative_condition_max"]) < int(
                block["repeats_per_representative_condition_min"]
            ) or float(block["empirical_coverage_max"]) < float(
                block["empirical_coverage_min"]
            ):
                errors.append("uncertainty_ranges_not_ordered")
        result["blocks"][block_name] = {
            "valid": not errors,
            "errors": errors,
            "evidence_refs": list(refs) if isinstance(refs, list) else [],
        }
        if errors:
            all_valid = False
    result["content_valid"] = bool(all_valid)
    result["valid"] = bool(all_valid and (content_bound if require_final_blocks else True))
    result["required_for_final_pass_fail"] = bool(require_final_blocks)
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"JSON 包含非有限数值：{value}")


def _load_mapping(path: str | Path, label: str) -> tuple[Path, dict[str, Any]]:
    source = Path(path).expanduser().resolve(strict=False)
    if not source.is_file():
        raise FileNotFoundError(f"{label} 不存在：{source}")
    try:
        parsed = json.loads(
            source.read_text(encoding="utf-8-sig"),
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法解析 {label}：{source}: {exc}") from exc
    if not isinstance(parsed, Mapping):
        raise ValueError(f"{label} 顶层必须是 JSON object：{source}")
    return source, dict(parsed)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_hash_matches(path: Path, expected: Any) -> bool:
    if not isinstance(expected, str):
        return False
    expected = expected.strip().lower()
    if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
        return False
    try:
        return path.is_file() and _sha256_file(path) == expected
    except OSError:
        return False


def _scalar_text(archive: Any, key: str) -> str | None:
    if key not in archive.files:
        return None
    value = np.asarray(archive[key])
    if value.ndim != 0:
        return None
    return str(value.item())


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


def _nonempty_fields(row: Mapping[str, Any], fields: Sequence[str]) -> bool:
    return all(isinstance(row.get(field), str) and row[field].strip() for field in fields)


def _json_list(value: Any) -> list[Any] | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, list) else None


def _finite_xy_point(value: Any) -> bool:
    if not isinstance(value, list) or len(value) != 2:
        return False
    try:
        x, y = float(value[0]), float(value[1])
    except (TypeError, ValueError, OverflowError):
        return False
    return bool(np.isfinite(x) and np.isfinite(y))


def _annotation_geometry_valid(row: Mapping[str, Any], *, consensus: bool) -> bool:
    valid_area = _json_list(row.get("valid_area"))
    if valid_area is None or len(valid_area) < 3 or not all(
        _finite_xy_point(point) for point in valid_area
    ):
        return False
    polygon = np.asarray(valid_area, dtype=float)
    area_twice = abs(
        float(
            np.dot(polygon[:, 0], np.roll(polygon[:, 1], -1))
            - np.dot(polygon[:, 1], np.roll(polygon[:, 0], -1))
        )
    )
    if len(np.unique(polygon, axis=0)) < 3 or area_twice <= 0.0:
        return False
    for field in ("beamstop", "streak", "overlap"):
        value = str(row.get(field, "")).strip()
        if value.casefold() not in _UNKNOWN_TOKENS and _json_list(value) is None:
            return False
    ridge_text = str(row.get("ridge_points", "")).strip()
    if ridge_text.casefold() not in _UNKNOWN_TOKENS:
        ridge_points = _json_list(ridge_text)
        if ridge_points is None or not all(_finite_xy_point(point) for point in ridge_points):
            return False
    lobe_x = str(row.get("lobe_center_x", "")).strip()
    lobe_y = str(row.get("lobe_center_y", "")).strip()
    lobe_unknown = lobe_x.casefold() in _UNKNOWN_TOKENS and lobe_y.casefold() in _UNKNOWN_TOKENS
    if not lobe_unknown:
        try:
            coordinates = (float(lobe_x), float(lobe_y))
        except (TypeError, ValueError, OverflowError):
            return False
        if not all(np.isfinite(value) for value in coordinates):
            return False
    identity_fields = ("software", "software_version", "reviewer" if consensus else "annotator")
    if any(str(row.get(field, "")).strip().casefold() in _NONDECISIVE_TOKENS for field in identity_fields):
        return False
    if consensus and str(row.get("consensus_status", "")).strip().casefold() in _NONDECISIVE_TOKENS:
        return False
    return True


def _aggregate(values: Sequence[float], method: str) -> float | None:
    if not values or method not in _SUPPORTED_AGGREGATIONS:
        return None
    array = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(array)):
        return None
    operations = {
        "mean": np.mean,
        "median": np.median,
        "p95": lambda data: np.percentile(data, 95),
        "max": np.max,
        "min": np.min,
    }
    return float(operations[method](array))


def _metric_matches_samples(
    source: Mapping[str, Any],
    samples: Sequence[float],
    *,
    positive: bool,
) -> bool:
    metric = source.get("metric")
    if not isinstance(metric, Mapping):
        return False
    method = str(metric.get("aggregation", ""))
    calculated = _aggregate(samples, method)
    try:
        recorded = float(metric.get("value"))
    except (TypeError, ValueError, OverflowError):
        return False
    minimum_ok = all(value > 0.0 for value in samples) if positive else all(
        value >= 0.0 for value in samples
    )
    return bool(
        calculated is not None
        and minimum_ok
        and np.isclose(recorded, calculated, rtol=1e-12, atol=1e-12)
    )


def _case_files_exist(manifest_path: Path, cases: Sequence[Any], keys: Sequence[str]) -> bool:
    for item in cases:
        if not isinstance(item, Mapping):
            return False
        filename = next((item.get(key) for key in keys if item.get(key)), None)
        if not isinstance(filename, str) or not (manifest_path.parent / filename).is_file():
            return False
    return True


def _npz_arrays_valid(
    manifest_path: Path,
    cases: Sequence[Any],
    *,
    filename_keys: Sequence[str],
    required_arrays: set[str],
    nonnegative_arrays: set[str] = frozenset(),
) -> bool:
    """Check the small fixed scientific array contract for every evidence file."""

    for item in cases:
        if not isinstance(item, Mapping):
            return False
        filename = next((item.get(key) for key in filename_keys if item.get(key)), None)
        if not isinstance(filename, str):
            return False
        try:
            with np.load(manifest_path.parent / filename, allow_pickle=False) as archive:
                if not required_arrays <= set(archive.files):
                    return False
                shape = np.asarray(archive["intensity"]).shape
                for name in required_arrays:
                    array = np.asarray(archive[name])
                    if array.shape != shape or not np.all(np.isfinite(array)):
                        return False
                    if name in nonnegative_arrays and np.any(array < 0):
                        return False
                if np.asarray(archive["mask"]).dtype != bool:
                    return False
                if "valid_mask" in required_arrays:
                    valid_mask = np.asarray(archive["valid_mask"])
                    if valid_mask.dtype != bool or not np.array_equal(
                        valid_mask, ~np.asarray(archive["mask"], dtype=bool)
                    ):
                        return False
                if "truth_ridge_support" in required_arrays and np.asarray(
                    archive["truth_ridge_support"]
                ).dtype != bool:
                    return False
                qx = np.asarray(archive["qx"], dtype=float)
                qy = np.asarray(archive["qy"], dtype=float)
                q = np.asarray(archive["q"], dtype=float)
                if not np.allclose(q, np.hypot(qx, qy), rtol=1e-12, atol=1e-12):
                    return False
        except (OSError, ValueError, KeyError):
            return False
    return True


def _t1_truth_valid(manifest_path: Path, cases: Sequence[Any]) -> bool:
    for item in cases:
        if not isinstance(item, Mapping):
            return False
        case_name = str(item.get("name", item.get("case_id", "")))
        npz_name = item.get("npz", item.get("npz_file"))
        truth_name = item.get("truth_json", item.get("truth_file"))
        if (
            item.get("q_unit") != T1_Q_UNIT
            or item.get("generator_version") != T1_GENERATOR_VERSION
            or item.get("generator_hash") != T1_GENERATOR_HASH
            or not isinstance(npz_name, str)
            or not isinstance(truth_name, str)
        ):
            return False
        npz_path = manifest_path.parent / npz_name
        truth_path = manifest_path.parent / truth_name
        if not _artifact_hash_matches(npz_path, item.get("npz_sha256")) or not _artifact_hash_matches(
            truth_path, item.get("truth_json_sha256")
        ):
            return False
        try:
            _, truth = _load_mapping(truth_path, "T1 case truth")
            with np.load(npz_path, allow_pickle=False) as archive:
                npz_q_unit = _scalar_text(archive, "q_unit")
                npz_version = _scalar_text(archive, "generator_version")
                npz_hash = _scalar_text(archive, "generator_hash")
        except (OSError, ValueError, KeyError):
            return False
        generator = truth.get("generator")
        files = truth.get("files")
        if (
            truth.get("schema_version") != T1_SCHEMA_VERSION
            or truth.get("case_name") != case_name
            or truth.get("q_unit") != T1_Q_UNIT
            or not isinstance(generator, Mapping)
            or generator.get("version") != T1_GENERATOR_VERSION
            or generator.get("hash") != T1_GENERATOR_HASH
            or generator.get("dependency_sha256") != T1_DEPENDENCY_HASHES
            or not isinstance(files, Mapping)
            or files.get("npz") != npz_name
            or files.get("truth_json") != truth_name
            or npz_q_unit != T1_Q_UNIT
            or npz_version != T1_GENERATOR_VERSION
            or npz_hash != T1_GENERATOR_HASH
        ):
            return False
    return True


def _t2_fft_and_truth_valid(manifest_path: Path, cases: Sequence[Any]) -> bool:
    for item in cases:
        if not isinstance(item, Mapping):
            return False
        case_id = str(item.get("case_id", ""))
        category = str(item.get("category", ""))
        if case_id != category or category not in _T2_REQUIRED_CATEGORIES:
            return False
        projection = item.get("projection_truth")
        structure = item.get("structure_truth")
        if not isinstance(projection, Mapping) or not isinstance(structure, Mapping):
            return False
        if (
            projection.get("reference_method")
            != "analytic_bragg_vectors_from_generator_structure"
            or projection.get("independent_of_generated_fft_pixels") is not True
        ):
            return False
        if item.get("q_unit") != T2_Q_UNIT:
            return False
        if category == "non_elliptical" and (
            projection.get("quantitative_use") != "negative_classification_only"
        ):
            return False
        components = structure.get("realized_components")
        if not isinstance(components, list):
            return False
        for component in components:
            if not isinstance(component, Mapping):
                return False
            positions = np.asarray(component.get("layer_positions_nm"), dtype=float)
            if positions.ndim != 1 or len(positions) < 2 or np.any(np.diff(positions) <= 0):
                return False
        filename = item.get("npz_file", item.get("npz"))
        if not isinstance(filename, str) or not _artifact_hash_matches(
            manifest_path.parent / filename, item.get("npz_sha256")
        ):
            return False
        try:
            with np.load(manifest_path.parent / filename, allow_pickle=False) as archive:
                density = np.asarray(archive["real_space_density"], dtype=float)
                clean = np.asarray(archive["intensity_noiseless"], dtype=float)
                projection_reference = np.asarray(archive["projection_reference"], dtype=float)
                qx = np.asarray(archive["qx"], dtype=float)
                qy = np.asarray(archive["qy"], dtype=float)
                q = np.asarray(archive["q"], dtype=float)
                mask = np.asarray(archive["mask"])
                valid_mask = np.asarray(archive["valid_mask"])
                npz_hash = str(np.asarray(archive["generator_hash"]).item())
                npz_version = str(np.asarray(archive["generator_version"]).item())
                npz_q_unit = _scalar_text(archive, "q_unit")
                npz_case_id = _scalar_text(archive, "case_id")
                npz_category = _scalar_text(archive, "category")
                projection_json = json.loads(
                    str(np.asarray(archive["projection_truth_json"]).item()),
                    parse_constant=_reject_json_constant,
                )
                structure_json = json.loads(
                    str(np.asarray(archive["structure_truth_json"]).item()),
                    parse_constant=_reject_json_constant,
                )
        except (OSError, ValueError, KeyError):
            return False
        if (
            npz_hash != T2_GENERATOR_HASH
            or npz_version != T2_GENERATOR_VERSION
            or npz_q_unit != T2_Q_UNIT
            or npz_case_id != case_id
            or npz_category != category
            or projection_json != projection
            or structure_json != structure
            or projection_reference.ndim != 2
            or projection_reference.shape[1] != 2
            or not np.all(np.isfinite(projection_reference))
            or mask.dtype != bool
            or valid_mask.dtype != bool
            or not np.array_equal(valid_mask, ~mask)
        ):
            return False
        pixel_size_nm = float(structure.get("pixel_size_nm", 0.0))
        if pixel_size_nm <= 0 or not np.isfinite(pixel_size_nm):
            return False
        rows, cols = density.shape
        expected_qx_axis = np.fft.fftshift(
            2.0 * np.pi * np.fft.fftfreq(cols, d=pixel_size_nm)
        )
        expected_qy_axis = np.fft.fftshift(
            2.0 * np.pi * np.fft.fftfreq(rows, d=pixel_size_nm)
        )
        expected_qx, expected_qy = np.meshgrid(expected_qx_axis, expected_qy_axis)
        if (
            not np.allclose(qx, expected_qx, rtol=1e-12, atol=1e-12)
            or not np.allclose(qy, expected_qy, rtol=1e-12, atol=1e-12)
            or not np.allclose(q, np.hypot(expected_qx, expected_qy), rtol=1e-12, atol=1e-12)
        ):
            return False
        fourier = np.fft.fftshift(np.fft.fft2(density - float(np.mean(density))))
        rebuilt = np.abs(fourier) ** 2
        maximum = float(np.max(rebuilt))
        if maximum <= 0 or not np.allclose(clean, rebuilt / maximum, rtol=1e-12, atol=1e-12):
            return False
        try:
            expected = generate_t2_case(
                case_id,
                shape=tuple(int(value) for value in item["shape"]),
                seed=int(item["seed"]),
                noise_sigma=float(item["noise_sigma"]),
                pixel_size_nm=float(structure["pixel_size_nm"]),
            )
        except (KeyError, TypeError, ValueError):
            return False
        if not np.allclose(density, expected["real_space_density"], rtol=1e-12, atol=1e-12):
            return False
        if not np.allclose(projection_reference, expected["projection_reference"], rtol=1e-12, atol=1e-12):
            return False
        if projection != expected["projection_truth"] or structure != expected["structure_truth"]:
            return False
    return True


def _check(check_id: str, passed: bool, criterion: str, evidence: Any) -> dict[str, Any]:
    return {
        "id": check_id,
        "status": "PASS" if passed else "FAIL",
        "criterion": criterion,
        "evidence": evidence,
    }


def _human_evidence_complete(annotation: Mapping[str, Any]) -> bool:
    evidence = annotation.get("human_evidence")
    if not isinstance(evidence, Mapping) or evidence.get("blinded") is not True:
        return False
    mode = evidence.get("mode")
    if mode == "two_independent_annotators":
        return int(evidence.get("annotator_count", 0)) >= 2
    if mode == "one_expert_repeat":
        return (
            int(evidence.get("session_count", 0)) >= 2
            and float(evidence.get("interval_days", 0.0)) >= 7.0
            and evidence.get("lower_evidence") is True
        )
    return False


def _annotation_csv_evidence_complete(
    status_path: Path,
    annotation: Mapping[str, Any],
) -> bool:
    files = annotation.get("files")
    if not isinstance(files, Mapping):
        return False
    required = ("annotation_manifest", "annotator_a", "annotator_b", "consensus_review")
    rows_by_key: dict[str, list[dict[str, str]]] = {}
    try:
        for key in required:
            relative = files.get(key)
            if not isinstance(relative, str) or not relative:
                return False
            path = Path(relative)
            if not path.is_absolute():
                path = status_path.parent / path
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                rows_by_key[key] = list(csv.DictReader(handle))
    except (OSError, UnicodeError, csv.Error):
        return False

    for rows in rows_by_key.values():
        if len(rows) != 8 or {row.get("blind_id") for row in rows} != _BLIND_IDS:
            return False
    manifest_rows = rows_by_key["annotation_manifest"]
    if any(
        not _nonempty_fields(
            row,
            (
                "role",
                "source_path_relative_package",
                "selector",
                "sha256",
                "selection_reason",
            ),
        )
        or len(row["sha256"].strip()) != 64
        or any(char not in "0123456789abcdef" for char in row["sha256"].strip().lower())
        for row in manifest_rows
    ):
        return False

    image_hashes = annotation.get("blind_image_hashes")
    immutable_hashes = annotation.get("immutable_output_hashes")
    input_hashes = annotation.get("input_hashes")
    input_info = annotation.get("input")
    if (
        not isinstance(image_hashes, Mapping)
        or set(image_hashes) != _BLIND_IDS
        or not isinstance(immutable_hashes, Mapping)
        or not isinstance(input_hashes, list)
        or not input_hashes
        or not isinstance(input_info, Mapping)
        or input_info.get("read_only") is not True
    ):
        return False
    recorded_input_hashes: set[str] = set()
    for record in input_hashes:
        if not isinstance(record, Mapping):
            return False
        before = str(record.get("sha256_before", "")).lower()
        after = str(record.get("sha256_after", "")).lower()
        if (
            record.get("unchanged") is not True
            or len(before) != 64
            or before != after
            or any(char not in "0123456789abcdef" for char in before)
        ):
            return False
        recorded_input_hashes.add(after)
    if not {row["sha256"].strip().lower() for row in manifest_rows} <= recorded_input_hashes:
        return False

    for blind_id in sorted(_BLIND_IDS):
        relative = files.get(f"{blind_id}_png")
        expected_hash = str(image_hashes.get(blind_id, "")).lower()
        if not isinstance(relative, str) or len(expected_hash) != 64:
            return False
        image_path = Path(relative)
        if not image_path.is_absolute():
            image_path = status_path.parent / image_path
        try:
            actual_hash = _sha256_file(image_path)
        except OSError:
            return False
        if actual_hash != expected_hash:
            return False
    for relative, expected_hash in immutable_hashes.items():
        if not isinstance(relative, str) or not isinstance(expected_hash, str):
            return False
        immutable_path = Path(relative)
        if not immutable_path.is_absolute():
            immutable_path = status_path.parent / immutable_path
        try:
            actual_hash = _sha256_file(immutable_path)
        except OSError:
            return False
        if actual_hash != expected_hash.lower():
            return False

    for key in ("annotator_a", "annotator_b"):
        if any(
            not _nonempty_fields(row, _ANNOTATION_CONTENT_FIELDS)
            or not _annotation_geometry_valid(row, consensus=False)
            for row in rows_by_key[key]
        ):
            return False
    consensus_rows = rows_by_key["consensus_review"]
    if any(
        not _nonempty_fields(row, _CONSENSUS_CONTENT_FIELDS)
        or not _annotation_geometry_valid(row, consensus=True)
        for row in consensus_rows
    ):
        return False
    for rows, time_key in (
        (rows_by_key["annotator_a"], "annotation_time"),
        (rows_by_key["annotator_b"], "annotation_time"),
        (consensus_rows, "review_time"),
    ):
        if any(_parse_timestamp(row.get(time_key)) is None for row in rows):
            return False
        if any(
            row.get("coordinate_system") != ANNOTATION_COORDINATE_SYSTEM
            or row.get("image_version") != image_hashes.get(row.get("blind_id"))
            for row in rows
        ):
            return False

    evidence = annotation.get("human_evidence")
    if not isinstance(evidence, Mapping):
        return False
    annotators_a = {row["annotator"] for row in rows_by_key["annotator_a"]}
    annotators_b = {row["annotator"] for row in rows_by_key["annotator_b"]}
    if len(annotators_a) != 1 or len(annotators_b) != 1:
        return False
    mode = evidence.get("mode")
    if mode == "two_independent_annotators":
        if annotators_a == annotators_b:
            return False
    elif mode == "one_expert_repeat":
        if annotators_a != annotators_b:
            return False
        rows_a = {row["blind_id"]: row for row in rows_by_key["annotator_a"]}
        rows_b = {row["blind_id"]: row for row in rows_by_key["annotator_b"]}
        for blind_id in _BLIND_IDS:
            time_a = _parse_timestamp(rows_a[blind_id]["annotation_time"])
            time_b = _parse_timestamp(rows_b[blind_id]["annotation_time"])
            if time_a is None or time_b is None:
                return False
            if abs((time_b - time_a).total_seconds()) < 7 * 24 * 60 * 60:
                return False
    else:
        return False
    return True


def _source_file_matches_hash(threshold_path: Path, value: Any) -> bool:
    if not isinstance(value, Mapping) or not isinstance(value.get("source"), str):
        return False
    expected = str(value.get("sha256", "")).lower()
    if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
        return False
    path = Path(value["source"])
    if not path.is_absolute():
        path = threshold_path.parent / path
    if not path.is_file():
        return False
    try:
        actual = _sha256_file(path)
    except OSError:
        return False
    return actual == expected


def _source_mapping(threshold_path: Path, value: Any) -> tuple[Path, dict[str, Any]] | None:
    if not _source_file_matches_hash(threshold_path, value) or not isinstance(value, Mapping):
        return None
    path = Path(str(value["source"]))
    if not path.is_absolute():
        path = threshold_path.parent / path
    try:
        _, parsed = _load_mapping(path, "threshold evidence source")
    except (FileNotFoundError, ValueError):
        return None
    return path, parsed


def _recorded_file_complete(base_path: Path, record: Any) -> bool:
    if not isinstance(record, Mapping) or not isinstance(record.get("source"), str):
        return False
    expected = str(record.get("sha256", "")).lower()
    if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
        return False
    path = Path(record["source"])
    if not path.is_absolute():
        path = base_path.parent / path
    try:
        return path.is_file() and _sha256_file(path) == expected
    except OSError:
        return False


def _matching_metric(
    record: Any,
    source: Mapping[str, Any],
    *,
    positive: bool,
    required_unit: str,
) -> bool:
    if not isinstance(record, Mapping) or record.get("status") != "complete":
        return False
    record_metric = record.get("metric")
    source_metric = source.get("metric")
    if not isinstance(record_metric, Mapping) or not isinstance(source_metric, Mapping):
        return False
    try:
        record_value = float(record_metric.get("value"))
        source_value = float(source_metric.get("value"))
    except (TypeError, ValueError, OverflowError):
        return False
    minimum_ok = record_value > 0.0 if positive else record_value >= 0.0
    record_aggregation = record_metric.get("aggregation")
    source_aggregation = source_metric.get("aggregation")
    return bool(
        np.isfinite(record_value)
        and np.isfinite(source_value)
        and minimum_ok
        and record_value == source_value
        and record_metric.get("unit") == required_unit
        and source_metric.get("unit") == required_unit
        and record_aggregation in _SUPPORTED_AGGREGATIONS
        and record_aggregation == source_aggregation
    )


def _human_source_complete(
    threshold_path: Path,
    value: Any,
    annotation_path: Path,
) -> bool:
    loaded = _source_mapping(threshold_path, value)
    if loaded is None:
        return False
    _, source = loaded
    if not _matching_metric(value, source, positive=False, required_unit="px"):
        return False
    try:
        annotation_hash = _sha256_file(annotation_path)
    except OSError:
        return False
    per_frame = source.get("per_frame_error_px")
    if not isinstance(per_frame, Mapping) or set(per_frame) != _BLIND_IDS:
        return False
    try:
        samples = [float(per_frame[blind_id]) for blind_id in sorted(_BLIND_IDS)]
    except (TypeError, ValueError, OverflowError):
        return False
    return (
        source.get("schema_version") == "lamellarsaxs2d.human_repeatability.v1"
        and source.get("status") == "complete"
        and source.get("blinded") is True
        and source.get("mode") in {"two_independent_annotators", "one_expert_repeat"}
        and source.get("frame_count") == 8
        and source.get("annotation_status_sha256") == annotation_hash
        and _metric_matches_samples(source, samples, positive=False)
        and isinstance(source.get("reviewed_by"), str)
        and bool(source["reviewed_by"].strip())
        and _parse_timestamp(source.get("reviewed_at")) is not None
    )


def _instrument_source_complete(threshold_path: Path, value: Any) -> bool:
    loaded = _source_mapping(threshold_path, value)
    if loaded is None:
        return False
    source_path, source = loaded
    measurements = source.get("measurements_nm_inv")
    if not isinstance(measurements, list) or not measurements:
        return False
    try:
        samples = [float(item) for item in measurements]
    except (TypeError, ValueError, OverflowError):
        return False
    return bool(
        _matching_metric(value, source, positive=True, required_unit=T2_Q_UNIT)
        and _metric_matches_samples(source, samples, positive=True)
        and _recorded_file_complete(source_path, source.get("calibration_record"))
        and source.get("schema_version") == "lamellarsaxs2d.instrument_resolution.v1"
        and source.get("status") == "complete"
        and isinstance(source.get("method"), str)
        and bool(source["method"].strip())
        and isinstance(source.get("reviewed_by"), str)
        and bool(source["reviewed_by"].strip())
        and _parse_timestamp(source.get("reviewed_at")) is not None
    )


def _pilot_source_complete(
    threshold_path: Path,
    value: Any,
    annotation_path: Path,
) -> bool:
    loaded = _source_mapping(threshold_path, value)
    if loaded is None or not isinstance(value, Mapping):
        return False
    _, source = loaded
    try:
        _, annotation = _load_mapping(annotation_path, "annotation status")
        if source.get("annotation_status_sha256") != _sha256_file(annotation_path):
            return False
        files = annotation.get("files")
        if not isinstance(files, Mapping) or not isinstance(files.get("consensus_review"), str):
            return False
        consensus_path = Path(files["consensus_review"])
        if not consensus_path.is_absolute():
            consensus_path = annotation_path.parent / consensus_path
        if source.get("consensus_sha256") != _sha256_file(consensus_path):
            return False
        with consensus_path.open("r", encoding="utf-8-sig", newline="") as handle:
            consensus_rows = list(csv.DictReader(handle))
    except (OSError, UnicodeError, ValueError, csv.Error):
        return False
    frame_results = source.get("frame_results")
    if not isinstance(frame_results, list) or len(frame_results) != 8:
        return False
    source_statuses = {
        str(item.get("blind_id")): str(item.get("consensus_status", "")).strip()
        for item in frame_results
        if isinstance(item, Mapping)
    }
    consensus_statuses = {
        str(row.get("blind_id")): str(row.get("consensus_status", "")).strip()
        for row in consensus_rows
    }
    return (
        value.get("status") == "complete"
        and value.get("frame_count") == 8
        and source.get("schema_version") == "lamellarsaxs2d.pilot_evidence.v1"
        and source.get("status") == "complete"
        and source.get("frame_count") == 8
        and set(source.get("blind_ids", ())) == _BLIND_IDS
        and len(source.get("blind_ids", ())) == 8
        and set(source_statuses) == _BLIND_IDS
        and source_statuses == consensus_statuses
        and all(status.casefold() not in _NONDECISIVE_TOKENS for status in source_statuses.values())
        and isinstance(source.get("reviewed_by"), str)
        and bool(source["reviewed_by"].strip())
        and _parse_timestamp(source.get("reviewed_at")) is not None
    )


def evaluate_p3_gate(
    t1_manifest: str | Path,
    t2_manifest: str | Path,
    annotation_status: str | Path,
    thresholds: str | Path,
) -> dict[str, Any]:
    """Evaluate the fixed P3 Go/No-Go criteria without modifying inputs."""

    t1_path, t1 = _load_mapping(t1_manifest, "T1 truth manifest")
    t2_path, t2 = _load_mapping(t2_manifest, "T2 truth manifest")
    annotation_path, annotation = _load_mapping(annotation_status, "annotation status")
    threshold_path, threshold = _load_mapping(thresholds, "acceptance thresholds")

    t1_cases = t1.get("cases") if isinstance(t1.get("cases"), list) else []
    t1_names = {
        str(item.get("name", item.get("case_id")))
        for item in t1_cases
        if isinstance(item, Mapping)
    }
    t1_required = set(T1_DEFAULT_CASE_NAMES)
    t1_complete = (
        t1.get("schema") == "t1_truth_manifest_v1"
        and t1.get("same_model") is True
        and t1.get("generator_version") == T1_GENERATOR_VERSION
        and t1.get("generator_hash") == T1_GENERATOR_HASH
        and isinstance(t1.get("generator"), Mapping)
        and t1["generator"].get("dependency_sha256") == T1_DEPENDENCY_HASHES
        and isinstance(t1.get("array_contract"), Mapping)
        and t1["array_contract"].get("q_unit") == T1_Q_UNIT
        and len(t1_cases) == len(t1_required)
        and t1_names == t1_required
        and _case_files_exist(t1_path, t1_cases, ("npz", "npz_file"))
        and _case_files_exist(t1_path, t1_cases, ("truth_json", "truth_file"))
        and _npz_arrays_valid(
            t1_path,
            t1_cases,
            filename_keys=("npz", "npz_file"),
            required_arrays={
                "intensity",
                "qx",
                "qy",
                "q",
                "mask",
                "valid_mask",
                "truth_intensity",
                "noise",
                "truth_ridge_plus",
                "truth_ridge_minus",
                "truth_ridge_support",
            },
        )
        and _t1_truth_valid(t1_path, t1_cases)
    )

    t2_cases = t2.get("cases") if isinstance(t2.get("cases"), list) else []
    t2_categories = {
        str(item.get("category")) for item in t2_cases if isinstance(item, Mapping)
    }
    t2_complete = (
        t2.get("schema") == "t2_truth_manifest_v1"
        and t2.get("model_scope") == "independent_physical_synthetic"
        and t2.get("generator_hash") == T2_GENERATOR_HASH
        and t2.get("generator_version") == T2_GENERATOR_VERSION
        and isinstance(t2.get("array_contract"), Mapping)
        and t2["array_contract"].get("q_unit") == T2_Q_UNIT
        and len(t2_cases) == len(_T2_REQUIRED_CATEGORIES)
        and t2_categories == _T2_REQUIRED_CATEGORIES
        and _case_files_exist(t2_path, t2_cases, ("npz_file", "npz"))
        and _npz_arrays_valid(
            t2_path,
            t2_cases,
            filename_keys=("npz_file", "npz"),
            required_arrays={
                "intensity",
                "qx",
                "qy",
                "q",
                "mask",
                "intensity_noiseless",
                "noise",
                "real_space_density",
            },
            nonnegative_arrays={"intensity", "intensity_noiseless"},
        )
        and all(
            isinstance(item, Mapping)
            and isinstance(item.get("projection_truth"), Mapping)
            and isinstance(item.get("structure_truth"), Mapping)
            for item in t2_cases
        )
        and _t2_fft_and_truth_valid(t2_path, t2_cases)
    )

    annotation_state = str(annotation.get("status", ""))
    human_complete = (
        annotation.get("schema_version") == ANNOTATION_PACK_SCHEMA_VERSION
        and annotation.get("human_consensus") is True
        and annotation_state in _COMPLETE_ANNOTATION_STATES
        and int(annotation.get("candidate_count", 0)) == 8
        and int(annotation.get("consensus_records_count", 0)) == 8
        and _human_evidence_complete(annotation)
        and _annotation_csv_evidence_complete(annotation_path, annotation)
    )

    policy = threshold.get("policy") if isinstance(threshold.get("policy"), Mapping) else {}
    sources = (
        threshold.get("evidence_sources")
        if isinstance(threshold.get("evidence_sources"), Mapping)
        else {}
    )
    source_checks = {
        "human_annotation_repeatability": _human_source_complete(
            threshold_path,
            sources.get("human_annotation_repeatability"),
            annotation_path,
        ),
        "instrument_resolution": _instrument_source_complete(
            threshold_path, sources.get("instrument_resolution")
        ),
        "pilot_report": _pilot_source_complete(
            threshold_path,
            sources.get("pilot_report"),
            annotation_path,
        ),
        "human_source_file_hash": _source_file_matches_hash(
            threshold_path, sources.get("human_annotation_repeatability")
        ),
        "instrument_source_file_hash": _source_file_matches_hash(
            threshold_path, sources.get("instrument_resolution")
        ),
        "pilot_source_file_hash": _source_file_matches_hash(
            threshold_path, sources.get("pilot_report")
        ),
        "frozen_by": sources.get("frozen_by") not in (None, ""),
        "frozen_at": _parse_timestamp(sources.get("frozen_at")) is not None,
    }
    sources_complete = all(source_checks.values())
    numeric_contract = _numeric_threshold_contract(
        threshold,
        require_final_blocks=threshold.get("usable_for_final_pass_fail") is True,
    )
    thresholds_frozen = (
        threshold.get("schema_version") == "lamellarsaxs2d.acceptance_thresholds.v1"
        and threshold.get("thresholds_version") == "v1"
        and threshold.get("status") == "frozen"
        and threshold.get("frozen") is True
        and threshold.get("usable_for_final_pass_fail") is True
        and policy.get("algorithm_performance_may_change_thresholds") is False
        and policy.get("requires_human_repeatability") is True
        and policy.get("requires_instrument_resolution") is True
        and policy.get("requires_pilot_evidence") is True
        and sources_complete
        and numeric_contract["valid"]
    )

    checks = [
        _check(
            "t1_same_model_matrix",
            t1_complete,
            "T1 包含冻结的 15 类同模型矩阵和完整 NPZ/JSON 证据",
            {"case_count": len(t1_cases), "required_cases_present": sorted(t1_required & t1_names)},
        ),
        _check(
            "t2_independent_generator",
            t2_complete,
            "T2 与经验模型独立，并覆盖 2-point/eyebrow/butterfly/non_elliptical",
            {"categories": sorted(t2_categories), "model_scope": t2.get("model_scope")},
        ),
        _check(
            "r0_human_consensus",
            human_complete,
            "8 帧盲标已有两位独立标注者，或同一专家间隔至少 7 天复标，并完成 consensus",
            {
                "status": annotation.get("status"),
                "human_consensus": annotation.get("human_consensus"),
                "candidate_count": annotation.get("candidate_count"),
                "consensus_records_count": annotation.get("consensus_records_count"),
                "human_evidence": annotation.get("human_evidence"),
            },
        ),
        _check(
            "acceptance_thresholds_frozen",
            thresholds_frozen,
            "阈值由人工重复性、仪器分辨率和 pilot 证据冻结，且不随算法表现放宽",
            {
                "status": threshold.get("status"),
                "frozen": threshold.get("frozen"),
                "usable_for_final_pass_fail": threshold.get("usable_for_final_pass_fail"),
                "evidence_sources_complete": sources_complete,
                "evidence_source_checks": source_checks,
                "numeric_threshold_contract": numeric_contract,
            },
        ),
    ]
    blockers = [item["id"] for item in checks if item["status"] != "PASS"]
    go = not blockers
    input_paths = {
        "t1_manifest": t1_path,
        "t2_manifest": t2_path,
        "annotation_status": annotation_path,
        "thresholds": threshold_path,
    }
    input_records = {
        name: {"path": path.as_posix(), "sha256": _sha256_file(path)}
        for name, path in input_paths.items()
    }
    evidence_fingerprint = hashlib.sha256(
        json.dumps(
            {name: record["sha256"] for name, record in input_records.items()},
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": P3_GATE_SCHEMA_VERSION,
        "phase": "P3",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "go" if go else "no_go",
        "go": go,
        "exit_code": 0 if go else 1,
        "checks": checks,
        "blocking_checks": blockers,
        "inputs": input_records,
        "provenance": {
            "gate_code_sha256": _sha256_file(Path(__file__)),
            "evidence_fingerprint_sha256": evidence_fingerprint,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "t1_generator_version": T1_GENERATOR_VERSION,
            "t1_generator_hash": T1_GENERATOR_HASH,
            "t2_generator_version": T2_GENERATOR_VERSION,
            "t2_generator_hash": T2_GENERATOR_HASH,
            "numeric_threshold_contract_sha256": hashlib.sha256(
                json.dumps(
                    numeric_contract,
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
        },
        "next_phase": "P4" if go else None,
        "scope": "read_only_evidence_gate; does not run fitting or alter thresholds",
    }


def write_p3_gate_report(
    output: str | Path,
    t1_manifest: str | Path,
    t2_manifest: str | Path,
    annotation_status: str | Path,
    thresholds: str | Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Evaluate P3 and write one strict-JSON report without default overwrite."""

    destination = Path(output)
    if any(part.casefold() == "data_local" for part in destination.resolve(strict=False).parts):
        raise ValueError("P3 门禁报告不得写入 data_local 原始数据目录")
    if destination.exists() and not force:
        raise FileExistsError(f"P3 门禁报告已存在，未覆盖：{destination}")
    report = evaluate_p3_gate(t1_manifest, t2_manifest, annotation_status, thresholds)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


__all__ = ["P3_GATE_SCHEMA_VERSION", "evaluate_p3_gate", "write_p3_gate_report"]
