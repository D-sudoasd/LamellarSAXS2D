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
            truth_intensity=intensity,
            noise=np.zeros(shape),
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
            "array_contract": {"q_unit": T1_Q_UNIT},
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
        },
    )
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
