from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest

from butterfly_saxs.annotation_pack import (
    ANNOTATION_COORDINATE_SYSTEM,
    ANNOTATION_PACK_SCHEMA_VERSION,
)
from butterfly_saxs.benchmark_t1 import DEFAULT_CASE_NAMES
from butterfly_saxs.benchmark_t1 import GENERATOR_DEPENDENCY_HASHES as T1_DEPENDENCY_HASHES
from butterfly_saxs.benchmark_t1 import GENERATOR_HASH as T1_GENERATOR_HASH
from butterfly_saxs.benchmark_t1 import GENERATOR_VERSION as T1_GENERATOR_VERSION
from butterfly_saxs.benchmark_t1 import T1_Q_UNIT, T1_SCHEMA_VERSION
from butterfly_saxs.benchmark_t2 import GENERATOR_HASH as T2_GENERATOR_HASH
from butterfly_saxs.benchmark_t2 import GENERATOR_VERSION as T2_GENERATOR_VERSION
from butterfly_saxs.benchmark_t2 import T2_Q_UNIT
from butterfly_saxs.benchmark_t2 import generate_case as generate_t2_case
from butterfly_saxs.cli import main
from butterfly_saxs.p3_gate import evaluate_p3_gate, write_p3_gate_report


def _write_json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")
    return path


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> Path:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    path.write_text(buffer.getvalue(), encoding="utf-8")
    return path


def _rewrite_npz(path: Path, mutate) -> None:
    with np.load(path, allow_pickle=False) as archive:
        values = {name: np.array(archive[name], copy=True) for name in archive.files}
    mutate(values)
    np.savez_compressed(path, **values)


def _refresh_case_npz_hash(manifest: Path, *, t2: bool = False) -> None:
    value = json.loads(manifest.read_text(encoding="utf-8"))
    item = value["cases"][0]
    key = "npz_file" if t2 else "npz"
    filename = item[key]
    item["npz_sha256"] = hashlib.sha256(
        (manifest.parent / filename).read_bytes()
    ).hexdigest()
    _write_json(manifest, value)


def _complete_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    shape = (4, 5)
    qx, qy = np.meshgrid(np.linspace(-1.0, 1.0, shape[1]), np.linspace(-0.8, 0.8, shape[0]))
    q = np.hypot(qx, qy)
    intensity = np.ones(shape)
    mask = np.zeros(shape, dtype=bool)
    t1_dir = tmp_path / "t1"
    t1_cases = []
    for name in DEFAULT_CASE_NAMES:
        (t1_dir / f"{name}.npz").parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            t1_dir / f"{name}.npz",
            intensity=intensity,
            qx=qx,
            qy=qy,
            q=q,
            mask=mask,
            valid_mask=~mask,
            truth_intensity=intensity,
            noise=np.zeros(shape),
            truth_ridge_plus=np.ones(shape),
            truth_ridge_minus=np.ones(shape),
            truth_ridge_support=np.ones(shape, dtype=bool),
            q_unit=np.asarray(T1_Q_UNIT),
            generator_version=np.asarray(T1_GENERATOR_VERSION),
            generator_hash=np.asarray(T1_GENERATOR_HASH),
        )
        _write_json(
            t1_dir / f"{name}.json",
            {
                "schema_version": T1_SCHEMA_VERSION,
                "case_name": name,
                "q_unit": T1_Q_UNIT,
                "generator": {
                    "version": T1_GENERATOR_VERSION,
                    "hash": T1_GENERATOR_HASH,
                    "dependency_sha256": T1_DEPENDENCY_HASHES,
                },
                "files": {"npz": f"{name}.npz", "truth_json": f"{name}.json"},
            },
        )
        t1_cases.append(
            {
                "name": name,
                "npz": f"{name}.npz",
                "truth_json": f"{name}.json",
                "npz_sha256": hashlib.sha256(
                    (t1_dir / f"{name}.npz").read_bytes()
                ).hexdigest(),
                "truth_json_sha256": hashlib.sha256(
                    (t1_dir / f"{name}.json").read_bytes()
                ).hexdigest(),
                "q_unit": T1_Q_UNIT,
                "generator_version": T1_GENERATOR_VERSION,
                "generator_hash": T1_GENERATOR_HASH,
            }
        )
    t1 = _write_json(
        t1_dir / "truth_manifest.json",
        {
            "schema": "t1_truth_manifest_v1",
            "same_model": True,
            "generator_version": T1_GENERATOR_VERSION,
            "generator_hash": T1_GENERATOR_HASH,
            "generator": {"dependency_sha256": T1_DEPENDENCY_HASHES},
            "array_contract": {
                "q_unit": T1_Q_UNIT,
                "required_npz_keys": [
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
                ],
            },
            "cases": t1_cases,
        },
    )

    t2_dir = tmp_path / "t2"
    t2_cases = []
    for index, category in enumerate(("2-point", "eyebrow", "butterfly", "non_elliptical")):
        generated = generate_t2_case(
            category,
            shape=(16, 18),
            seed=100 + index,
            noise_sigma=0.0,
        )
        filename = f"{category}.npz"
        (t2_dir / filename).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            t2_dir / filename,
            real_space_density=generated["real_space_density"],
            intensity=generated["intensity"],
            qx=generated["qx"],
            qy=generated["qy"],
            q=generated["q"],
            q_unit=np.asarray(T2_Q_UNIT),
            mask=generated["mask"],
            valid_mask=generated["valid_mask"],
            intensity_noiseless=generated["intensity_noiseless"],
            noise=generated["noise"],
            projection_reference=generated["projection_reference"],
            projection_truth_json=np.asarray(
                json.dumps(generated["projection_truth"], sort_keys=True)
            ),
            structure_truth_json=np.asarray(
                json.dumps(generated["structure_truth"], sort_keys=True)
            ),
            case_id=np.asarray(category),
            category=np.asarray(category),
            generator_hash=np.asarray(T2_GENERATOR_HASH),
            generator_version=np.asarray(T2_GENERATOR_VERSION),
            model_scope=np.asarray("independent_physical_synthetic"),
        )
        t2_cases.append(
            {
                "case_id": category,
                "category": category,
                "npz_file": filename,
                "npz_sha256": hashlib.sha256((t2_dir / filename).read_bytes()).hexdigest(),
                "seed": 100 + index,
                "noise_sigma": 0.0,
                "shape": [16, 18],
                "q_unit": T2_Q_UNIT,
                "projection_truth": generated["projection_truth"],
                "structure_truth": generated["structure_truth"],
            }
        )
    t2 = _write_json(
        t2_dir / "truth_manifest.json",
        {
            "schema": "t2_truth_manifest_v1",
            "model_scope": "independent_physical_synthetic",
            "generator_hash": T2_GENERATOR_HASH,
            "generator_version": T2_GENERATOR_VERSION,
            "array_contract": {"q_unit": T2_Q_UNIT},
            "cases": t2_cases,
        },
    )
    annotation_dir = tmp_path / "annotation"
    blind_ids = [f"blind_{index:03d}" for index in range(1, 9)]
    payload_dir = annotation_dir / "blind_payload"
    payload_dir.mkdir(parents=True)
    source_hashes_by_id = {
        blind_id: hashlib.sha256(f"source-{blind_id}".encode()).hexdigest()
        for blind_id in blind_ids
    }
    _write_csv(
        annotation_dir / "annotation_manifest.csv",
        [
            "blind_id",
            "role",
            "source_path",
            "source_path_relative_package",
            "selector",
            "sha256",
            "selection_reason",
        ],
        [
            {
                "blind_id": blind_id,
                "role": f"role_{index}",
                "source_path": f"frames/{blind_id}.edf",
                "source_path_relative_package": f"frames/{blind_id}.edf",
                "selector": "default",
                "sha256": source_hashes_by_id[blind_id],
                "selection_reason": "fixed pilot frame",
            }
            for index, blind_id in enumerate(blind_ids)
        ],
    )
    (payload_dir / "annotation_protocol.md").write_text("blind protocol\n", encoding="utf-8")
    blind_image_hashes = {}
    for blind_id in blind_ids:
        image_path = payload_dir / f"{blind_id}.png"
        image_path.write_bytes(f"png-{blind_id}".encode())
        blind_image_hashes[blind_id] = hashlib.sha256(image_path.read_bytes()).hexdigest()
    annotation_fields = [
        "blind_id",
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
        "notes",
    ]
    for filename, annotator in (("annotator_a.csv", "expert_a"), ("annotator_b.csv", "expert_b")):
        _write_csv(
            payload_dir / filename,
            annotation_fields,
            [
                {
                    "blind_id": blind_id,
                    "valid_area": "[[0,0],[10,0],[10,10],[0,10]]",
                    "beamstop": "[]",
                    "streak": "[]",
                    "overlap": "[]",
                    "lobe_center_x": "5.0",
                    "lobe_center_y": "6.0",
                    "ridge_points": "[[4,5],[5,6],[6,5]]",
                    "software": "manual-tool",
                    "software_version": "1",
                    "coordinate_system": ANNOTATION_COORDINATE_SYSTEM,
                    "image_version": blind_image_hashes[blind_id],
                    "annotation_time": "2026-08-28T12:00:00+08:00",
                    "annotator": annotator,
                    "notes": "",
                }
                for blind_id in blind_ids
            ],
        )
    consensus_fields = [
        "blind_id",
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
        "notes",
    ]
    _write_csv(
        annotation_dir / "consensus_review.csv",
        consensus_fields,
        [
            {
                "blind_id": blind_id,
                "consensus_status": "accepted",
                "valid_area": "[[0,0],[10,0],[10,10],[0,10]]",
                "beamstop": "[]",
                "streak": "[]",
                "overlap": "[]",
                "lobe_center_x": "5.0",
                "lobe_center_y": "6.0",
                "ridge_points": "[[4,5],[5,6],[6,5]]",
                "reviewer": "reviewer",
                "software": "manual-tool",
                "software_version": "1",
                "coordinate_system": ANNOTATION_COORDINATE_SYSTEM,
                "image_version": blind_image_hashes[blind_id],
                "review_time": "2026-08-28T13:00:00+08:00",
                "notes": "",
            }
            for blind_id in blind_ids
        ],
    )
    immutable_relative = [
        "annotation_manifest.csv",
        "blind_payload/annotation_protocol.md",
        *(f"blind_payload/{blind_id}.png" for blind_id in blind_ids),
    ]
    immutable_hashes = {
        relative: hashlib.sha256((annotation_dir / relative).read_bytes()).hexdigest()
        for relative in immutable_relative
    }
    annotation = _write_json(
        annotation_dir / "annotation_status.json",
        {
            "schema_version": ANNOTATION_PACK_SCHEMA_VERSION,
            "status": "consensus_complete",
            "human_consensus": True,
            "candidate_count": 8,
            "consensus_records_count": 8,
            "human_evidence": {
                "mode": "two_independent_annotators",
                "annotator_count": 2,
                "blinded": True,
            },
            "input": {"read_only": True},
            "input_hashes": [
                {
                    "kind": "frame",
                    "sha256_before": source_hashes_by_id[blind_id],
                    "sha256_after": source_hashes_by_id[blind_id],
                    "unchanged": True,
                }
                for blind_id in blind_ids
            ],
            "blind_image_hashes": blind_image_hashes,
            "immutable_output_hashes": immutable_hashes,
            "files": {
                "annotation_manifest": "annotation_manifest.csv",
                "annotator_a": "blind_payload/annotator_a.csv",
                "annotator_b": "blind_payload/annotator_b.csv",
                "consensus_review": "consensus_review.csv",
                **{
                    f"{blind_id}_png": f"blind_payload/{blind_id}.png"
                    for blind_id in blind_ids
                },
            },
        },
    )
    _write_json(
        tmp_path / "repeatability.json",
        {
            "schema_version": "lamellarsaxs2d.human_repeatability.v1",
            "status": "complete",
            "blinded": True,
            "mode": "two_independent_annotators",
            "frame_count": 8,
            "annotation_status_sha256": hashlib.sha256(annotation.read_bytes()).hexdigest(),
            "per_frame_error_px": {blind_id: 0.35 for blind_id in blind_ids},
            "metric": {"value": 0.35, "unit": "px", "aggregation": "max"},
            "reviewed_by": "reviewer",
            "reviewed_at": "2026-08-28T14:00:00+08:00",
        },
    )
    calibration = tmp_path / "instrument_calibration.txt"
    calibration.write_text("measured q-resolution series\n", encoding="utf-8")
    _write_json(
        tmp_path / "instrument.json",
        {
            "schema_version": "lamellarsaxs2d.instrument_resolution.v1",
            "status": "complete",
            "measurements_nm_inv": [0.009, 0.01, 0.011],
            "metric": {"value": 0.01, "unit": T2_Q_UNIT, "aggregation": "median"},
            "method": "beamline calibration record",
            "calibration_record": {
                "source": calibration.name,
                "sha256": hashlib.sha256(calibration.read_bytes()).hexdigest(),
            },
            "reviewed_by": "beamline_scientist",
            "reviewed_at": "2026-08-28T14:00:00+08:00",
        },
    )
    _write_json(
        tmp_path / "pilot.json",
        {
            "schema_version": "lamellarsaxs2d.pilot_evidence.v1",
            "status": "complete",
            "frame_count": 8,
            "blind_ids": blind_ids,
            "annotation_status_sha256": hashlib.sha256(annotation.read_bytes()).hexdigest(),
            "consensus_sha256": hashlib.sha256(
                (annotation_dir / "consensus_review.csv").read_bytes()
            ).hexdigest(),
            "frame_results": [
                {"blind_id": blind_id, "consensus_status": "accepted"}
                for blind_id in blind_ids
            ],
            "reviewed_by": "reviewer",
            "reviewed_at": "2026-08-28T14:00:00+08:00",
        },
    )
    source_hashes = {
        name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
        for name in ("repeatability.json", "instrument.json", "pilot.json")
    }
    thresholds = _write_json(
        tmp_path / "acceptance_thresholds_v1.json",
        {
            "schema_version": "lamellarsaxs2d.acceptance_thresholds.v1",
            "thresholds_version": "v1",
            "status": "frozen",
            "frozen": True,
            "usable_for_final_pass_fail": True,
            "policy": {
                "algorithm_performance_may_change_thresholds": False,
                "requires_human_repeatability": True,
                "requires_instrument_resolution": True,
                "requires_pilot_evidence": True,
            },
            "evidence_sources": {
                "human_annotation_repeatability": {
                    "status": "complete",
                    "source": "repeatability.json",
                    "sha256": source_hashes["repeatability.json"],
                    "metric": {"value": 0.35, "unit": "px", "aggregation": "max"},
                },
                "instrument_resolution": {
                    "status": "complete",
                    "source": "instrument.json",
                    "sha256": source_hashes["instrument.json"],
                    "metric": {
                        "value": 0.01,
                        "unit": T2_Q_UNIT,
                        "aggregation": "median",
                    },
                },
                "pilot_report": {
                    "status": "complete",
                    "source": "pilot.json",
                    "sha256": source_hashes["pilot.json"],
                    "frame_count": 8,
                },
                "frozen_by": "reviewer",
                "frozen_at": "2026-08-28T12:00:00+08:00",
            },
            "t1_high_snr": {
                "ridge_detector_median_error_px_max": 0.5,
                "ridge_detector_p95_error_px_max": 1.0,
                "ridge_f1_min": 0.9,
                "lobe_periodic_angle_error_deg_max": 1.0,
                "ellipse_a_relative_error_max": 0.03,
                "ellipse_b_relative_error_max": 0.03,
                "ellipse_theta_periodic_error_deg_max": 1.0,
                "ellipse_center_equivalent_pixel_error_max": 0.5,
                "same_seed_deterministic": True,
                "evidence_refs": ["human_annotation_repeatability", "pilot_report"],
            },
            "t2_independent": {
                "ridge_error_local_fwhm_fraction_max": 0.25,
                "pattern_class_accuracy_min": 0.9,
                "projection_a_relative_error_max": 0.05,
                "projection_b_relative_error_max": 0.05,
                "projection_tilt_error_deg_max": 1.0,
                "structure_truth_is_not_empirical_inverse_truth": True,
                "evidence_refs": ["instrument_resolution", "pilot_report"],
            },
            "full2d_quality": {
                "scaled_condition_pass_lt": 1e8,
                "scaled_condition_warn_lt_or_equal": 1e12,
                "scaled_condition_fail_gt": 1e12,
                "nonfinite_condition_fails": True,
                "critical_bound_hit_fails": True,
                "withheld_failure_fails": True,
                "structured_residual_fails": True,
                "evidence_refs": ["pilot_report"],
            },
            "uncertainty": {
                "repeats_per_representative_condition_min": 100,
                "repeats_per_representative_condition_max": 200,
                "interval_level": 0.95,
                "empirical_coverage_min": 0.9,
                "empirical_coverage_max": 0.98,
                "false_pass_rate_max": 0.05,
                "statistical_and_selection_uncertainty_separate": True,
                "evidence_refs": ["human_annotation_repeatability", "pilot_report"],
            },
            "real_data": {
                "ridge_f1_min": 0.9,
                "lobe_periodic_angle_error_deg_max": 1.0,
                "repeat_frame_apparent_parameter_cv_max": 0.05,
                "pilot_frame_count": 8,
                "pilot_difficult_or_negative_count_min": 4,
                "holding_sequence_denominator": 120,
                "usable_fraction_min": 0.9,
                "independent_warm_start_difference_within_combined_uncertainty": True,
                "forward_reverse_systematic_bias_allowed": False,
                "resume_matches_continuous": True,
                "evidence_refs": ["pilot_report"],
            },
        },
    )
    threshold_value = json.loads(thresholds.read_text(encoding="utf-8"))
    threshold_content = {
        name: threshold_value.get(name)
        for name in ("t1_high_snr", "t2_independent", "full2d_quality", "uncertainty", "real_data")
    }
    threshold_value["threshold_content_sha256"] = hashlib.sha256(
        json.dumps(
            threshold_content,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    _write_json(thresholds, threshold_value)
    return t1, t2, annotation, thresholds


def test_complete_independent_and_human_evidence_is_go(tmp_path: Path) -> None:
    report = evaluate_p3_gate(*_complete_inputs(tmp_path))
    assert report["go"] is True
    assert report["status"] == "go"
    assert report["next_phase"] == "P4"
    assert report["blocking_checks"] == []
    assert all(item["status"] == "PASS" for item in report["checks"])
    assert all(len(record["sha256"]) == 64 for record in report["inputs"].values())
    assert len(report["provenance"]["evidence_fingerprint_sha256"]) == 64


@pytest.mark.parametrize("mutation", ["missing_block", "invalid_value", "altered_content"])
def test_final_threshold_gate_requires_bound_numeric_content(
    tmp_path: Path,
    mutation: str,
) -> None:
    _t1, _t2, _annotation, thresholds = _complete_inputs(tmp_path)
    value = json.loads(thresholds.read_text(encoding="utf-8"))
    if mutation == "missing_block":
        value.pop("t2_independent")
    elif mutation == "invalid_value":
        value["real_data"]["usable_fraction_min"] = 1.5
    else:
        value["t1_high_snr"]["ridge_f1_min"] = 0.95
    _write_json(thresholds, value)
    report = evaluate_p3_gate(_t1, _t2, _annotation, thresholds)
    assert report["go"] is False
    assert "acceptance_thresholds_frozen" in report["blocking_checks"]
    contract = next(
        check for check in report["checks"] if check["id"] == "acceptance_thresholds_frozen"
    )["evidence"]["numeric_threshold_contract"]
    assert contract["valid"] is False


def test_final_threshold_gate_rejects_zero_quality_floors_even_with_digest(
    tmp_path: Path,
) -> None:
    t1, t2, annotation, thresholds = _complete_inputs(tmp_path)
    value = json.loads(thresholds.read_text(encoding="utf-8"))
    value["t1_high_snr"]["ridge_f1_min"] = 0.0
    value["t2_independent"]["pattern_class_accuracy_min"] = 0.0
    value["uncertainty"]["empirical_coverage_min"] = 0.0
    value["real_data"]["ridge_f1_min"] = 0.0
    value["real_data"]["usable_fraction_min"] = 0.0
    value["real_data"]["pilot_difficult_or_negative_count_min"] = 0
    content = {
        name: value.get(name)
        for name in ("t1_high_snr", "t2_independent", "full2d_quality", "uncertainty", "real_data")
    }
    value["threshold_content_sha256"] = hashlib.sha256(
        json.dumps(
            content,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    _write_json(thresholds, value)

    report = evaluate_p3_gate(t1, t2, annotation, thresholds)

    assert report["go"] is False
    assert "acceptance_thresholds_frozen" in report["blocking_checks"]


@pytest.mark.parametrize(
    ("suite", "mutation"),
    [
        ("t1", "missing_required"),
        ("t1", "nonfinite"),
        ("t1", "mask_dtype"),
        ("t1", "q_mismatch"),
        ("t2", "missing_required"),
        ("t2", "nonfinite"),
        ("t2", "mask_dtype"),
        ("t2", "q_mismatch"),
    ],
)
def test_gate_rejects_corrupted_t1_and_t2_archives(
    tmp_path: Path,
    suite: str,
    mutation: str,
) -> None:
    t1, t2, annotation, thresholds = _complete_inputs(tmp_path)
    if suite == "t1":
        manifest = t1
        filename = json.loads(t1.read_text(encoding="utf-8"))["cases"][0]["npz"]

        def mutate(values: dict[str, np.ndarray]) -> None:
            if mutation == "missing_required":
                values.pop("truth_ridge_support")
            elif mutation == "nonfinite":
                values["truth_intensity"][0, 0] = np.nan
            elif mutation == "mask_dtype":
                values["mask"] = values["mask"].astype(np.uint8)
            else:
                values["q"][0, 0] += 1.0

        blocker = "t1_same_model_matrix"
    else:
        manifest = t2
        filename = json.loads(t2.read_text(encoding="utf-8"))["cases"][0]["npz_file"]

        def mutate(values: dict[str, np.ndarray]) -> None:
            if mutation == "missing_required":
                values.pop("projection_reference")
            elif mutation == "nonfinite":
                values["intensity_noiseless"][0, 0] = np.nan
            elif mutation == "mask_dtype":
                values["mask"] = values["mask"].astype(np.uint8)
            else:
                values["q"][0, 0] += 1.0

        blocker = "t2_independent_generator"
    _rewrite_npz(manifest.parent / filename, mutate)
    _refresh_case_npz_hash(manifest, t2=suite == "t2")

    report = evaluate_p3_gate(t1, t2, annotation, thresholds)

    assert report["go"] is False
    assert blocker in report["blocking_checks"]


def test_provisional_thresholds_and_empty_annotation_are_no_go(tmp_path: Path) -> None:
    t1, t2, annotation, thresholds = _complete_inputs(tmp_path)
    _write_json(
        annotation,
        {"status": "awaiting_human_annotations", "human_consensus": False, "candidate_count": 8},
    )
    _write_json(
        thresholds,
        {
            "schema_version": "lamellarsaxs2d.acceptance_thresholds.v1",
            "status": "draft_provisional",
            "frozen": False,
            "usable_for_final_pass_fail": False,
            "policy": {
                "algorithm_performance_may_change_thresholds": False,
                "requires_human_repeatability": True,
                "requires_instrument_resolution": True,
                "requires_pilot_evidence": True,
            },
            "evidence_sources": {
                "human_annotation_repeatability": None,
                "instrument_resolution": None,
                "pilot_report": None,
            },
        },
    )
    report = evaluate_p3_gate(t1, t2, annotation, thresholds)
    assert report["go"] is False
    assert report["next_phase"] is None
    assert report["blocking_checks"] == [
        "r0_human_consensus",
        "acceptance_thresholds_frozen",
    ]


def test_report_refuses_default_overwrite(tmp_path: Path) -> None:
    inputs = _complete_inputs(tmp_path)
    output = tmp_path / "gate.json"
    report = write_p3_gate_report(output, *inputs)
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == report["status"]
    before = output.read_bytes()
    with pytest.raises(FileExistsError, match="未覆盖"):
        write_p3_gate_report(output, *inputs)
    assert output.read_bytes() == before


def test_report_refuses_data_local_output(tmp_path: Path) -> None:
    inputs = _complete_inputs(tmp_path)

    with pytest.raises(ValueError, match="data_local"):
        write_p3_gate_report(tmp_path / "data_local" / "gate.json", *inputs)


def test_gate_rejects_metadata_only_human_annotation(tmp_path: Path) -> None:
    t1, t2, annotation, thresholds = _complete_inputs(tmp_path)
    annotation_csv = annotation.parent / "blind_payload" / "annotator_a.csv"
    with annotation_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fields = list(rows[0])
    rows[0]["valid_area"] = ""
    _write_csv(annotation_csv, fields, rows)

    report = evaluate_p3_gate(t1, t2, annotation, thresholds)

    assert "r0_human_consensus" in report["blocking_checks"]


def test_gate_rejects_placeholder_only_human_geometry(tmp_path: Path) -> None:
    t1, t2, annotation, thresholds = _complete_inputs(tmp_path)
    for relative in (
        "blind_payload/annotator_a.csv",
        "blind_payload/annotator_b.csv",
        "consensus_review.csv",
    ):
        path = annotation.parent / relative
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
            fields = list(rows[0])
        for row in rows:
            row["valid_area"] = "[]"
            row["beamstop"] = "unknown"
            row["streak"] = "unknown"
            row["overlap"] = "unknown"
            row["lobe_center_x"] = "unknown"
            row["lobe_center_y"] = "unknown"
            row["ridge_points"] = "[]"
        _write_csv(path, fields, rows)

    report = evaluate_p3_gate(t1, t2, annotation, thresholds)

    assert "r0_human_consensus" in report["blocking_checks"]


def test_gate_rejects_zero_area_valid_polygon(tmp_path: Path) -> None:
    t1, t2, annotation, thresholds = _complete_inputs(tmp_path)
    annotation_csv = annotation.parent / "blind_payload" / "annotator_a.csv"
    with annotation_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fields = list(rows[0])
    rows[0]["valid_area"] = "[[0,0],[1,1],[2,2]]"
    _write_csv(annotation_csv, fields, rows)

    report = evaluate_p3_gate(t1, t2, annotation, thresholds)

    assert "r0_human_consensus" in report["blocking_checks"]


def test_gate_rejects_semantically_empty_threshold_source_even_with_matching_hash(
    tmp_path: Path,
) -> None:
    t1, t2, annotation, thresholds = _complete_inputs(tmp_path)
    instrument = tmp_path / "instrument.json"
    _write_json(
        instrument,
        {
            "schema_version": "lamellarsaxs2d.instrument_resolution.v1",
            "status": "complete",
            "metric": {"value": 0.01, "unit": T2_Q_UNIT},
            "method": "",
            "reviewed_by": "beamline_scientist",
            "reviewed_at": "2026-08-28T14:00:00+08:00",
        },
    )
    threshold_value = json.loads(thresholds.read_text(encoding="utf-8"))
    threshold_value["evidence_sources"]["instrument_resolution"]["sha256"] = (
        hashlib.sha256(instrument.read_bytes()).hexdigest()
    )
    _write_json(thresholds, threshold_value)

    report = evaluate_p3_gate(t1, t2, annotation, thresholds)

    assert "acceptance_thresholds_frozen" in report["blocking_checks"]


def test_gate_rejects_pilot_without_consensus_binding(tmp_path: Path) -> None:
    t1, t2, annotation, thresholds = _complete_inputs(tmp_path)
    pilot = tmp_path / "pilot.json"
    pilot_value = json.loads(pilot.read_text(encoding="utf-8"))
    pilot_value.pop("frame_results")
    pilot_value.pop("consensus_sha256")
    _write_json(pilot, pilot_value)
    threshold_value = json.loads(thresholds.read_text(encoding="utf-8"))
    threshold_value["evidence_sources"]["pilot_report"]["sha256"] = hashlib.sha256(
        pilot.read_bytes()
    ).hexdigest()
    _write_json(thresholds, threshold_value)

    report = evaluate_p3_gate(t1, t2, annotation, thresholds)

    assert "acceptance_thresholds_frozen" in report["blocking_checks"]


def test_gate_rejects_changed_instrument_calibration_record(tmp_path: Path) -> None:
    t1, t2, annotation, thresholds = _complete_inputs(tmp_path)
    (tmp_path / "instrument_calibration.txt").write_text(
        "changed after review\n",
        encoding="utf-8",
    )

    report = evaluate_p3_gate(t1, t2, annotation, thresholds)

    assert "acceptance_thresholds_frozen" in report["blocking_checks"]


def test_gate_rejects_mismatched_t1_truth_and_t2_case_identity(tmp_path: Path) -> None:
    t1, t2, annotation, thresholds = _complete_inputs(tmp_path)
    t1_value = json.loads(t1.read_text(encoding="utf-8"))
    _write_json(t1.parent / t1_value["cases"][0]["truth_json"], {})
    t2_value = json.loads(t2.read_text(encoding="utf-8"))
    t2_value["cases"][0]["case_id"] = "butterfly"
    _write_json(t2, t2_value)

    report = evaluate_p3_gate(t1, t2, annotation, thresholds)

    assert "t1_same_model_matrix" in report["blocking_checks"]
    assert "t2_independent_generator" in report["blocking_checks"]


def test_gate_rejects_stale_generator_and_changed_threshold_source(tmp_path: Path) -> None:
    t1, t2, annotation, thresholds = _complete_inputs(tmp_path)
    t2_value = json.loads(t2.read_text(encoding="utf-8"))
    t2_value["generator_hash"] = "0" * 64
    _write_json(t2, t2_value)
    (tmp_path / "repeatability.json").write_text("{}\n", encoding="utf-8")
    report = evaluate_p3_gate(t1, t2, annotation, thresholds)
    assert "t2_independent_generator" in report["blocking_checks"]
    assert "acceptance_thresholds_frozen" in report["blocking_checks"]


def test_cli_returns_gate_exit_code_and_writes_report(tmp_path: Path) -> None:
    t1, t2, annotation, thresholds = _complete_inputs(tmp_path)
    output = tmp_path / "p3_gate_report.json"
    exit_code = main(
        [
            "p3-status",
            "--t1-manifest",
            str(t1),
            "--t2-manifest",
            str(t2),
            "--annotation-status",
            str(annotation),
            "--thresholds",
            str(thresholds),
            "--output",
            str(output),
        ]
    )
    assert exit_code == 0
    assert json.loads(output.read_text(encoding="utf-8"))["go"] is True
