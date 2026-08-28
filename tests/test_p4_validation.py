from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from butterfly_saxs.benchmark_t1 import generate_case
from butterfly_saxs.cli import main
from butterfly_saxs.p4_validation import (
    _assigned_periodic_errors,
    _load_r0_rows,
    _r0_quality_summary,
    _t2_expected_outcome,
    _t2_projection_contract_complete,
    _t1_visible_ridge_angles,
    run_p4_engineering,
)


def test_p4_periodic_lobe_assignment_wraps_at_180_degrees() -> None:
    errors = _assigned_periodic_errors([-179.5, 30.0], [179.5, 31.0])

    assert sorted(errors) == pytest.approx([1.0, 1.0])


def test_t1_f1_reference_uses_visible_truth_sectors_not_latent_full_ellipse() -> None:
    sample = generate_case("noiseless_default")
    arrays = {
        "truth_intensity": sample.truth_intensity,
        "valid_mask": ~sample.mask,
        "qx": sample.qx,
        "qy": sample.qy,
        "q": sample.q,
        "q_unit": sample.truth["q_unit"],
    }

    angles = _t1_visible_ridge_angles(arrays)

    assert angles is not None
    assert 0 < len(angles) < 72
    assert all(np.isfinite(value) for value in angles)


@pytest.mark.parametrize(
    ("category", "projection_evaluable", "expected"),
    [
        ("2-point", False, "reject_nonellipse_or_insufficient"),
        ("non_elliptical", True, "reject_nonellipse_or_insufficient"),
        ("eyebrow", False, "reject_information_insufficient_for_ellipse"),
        ("butterfly", False, "reject_information_insufficient_for_ellipse"),
        ("butterfly", True, "fit_projection_ellipse"),
    ],
)
def test_t2_expected_outcome_respects_available_projection_truth(
    category: str,
    projection_evaluable: bool,
    expected: str,
) -> None:
    assert (
        _t2_expected_outcome(category, projection_evaluable=projection_evaluable)
        == expected
    )


def test_t2_information_insufficient_cases_do_not_complete_ellipse_contract() -> None:
    cases = [
        {
            "expected_outcome": "reject_nonellipse_or_insufficient",
            "projection_thresholds_evaluable": False,
        },
        {
            "expected_outcome": "reject_information_insufficient_for_ellipse",
            "projection_thresholds_evaluable": False,
        },
    ]

    assert not _t2_projection_contract_complete(cases)

    cases[1] = {
        "expected_outcome": "fit_projection_ellipse",
        "projection_thresholds_evaluable": True,
    }
    assert _t2_projection_contract_complete(cases)


def test_p4_cli_help_is_available() -> None:
    with pytest.raises(SystemExit) as error:
        main(["p4-evaluate", "--help"])

    assert error.value.code == 0


def test_r0_manifest_requires_eight_complete_sha256_values(tmp_path: Path) -> None:
    manifest = tmp_path / "annotation_manifest.csv"
    rows = ["blind_id,source_path,sha256"]
    rows.extend(
        f"blind_{index:03d},frame_{index:03d}.edf,"
        for index in range(1, 9)
    )
    manifest.write_text("\n".join(rows), encoding="utf-8")

    with pytest.raises(ValueError, match="complete 64-character SHA-256"):
        _load_r0_rows(manifest)


def test_r0_quality_summary_fails_closed_for_unknown_or_missing_frames() -> None:
    valid = [{"fit": {"quality_status": "PASS"}} for _ in range(8)]
    status, counts = _r0_quality_summary(valid)
    assert status == "PASS"
    assert counts == {"PASS": 8, "WARN": 0, "FAIL": 0, "UNKNOWN": 0}

    valid[-1] = {"fit": {"quality_status": "unexpected"}}
    status, counts = _r0_quality_summary(valid)
    assert status == "FAIL"
    assert counts["UNKNOWN"] == 1

    status, _counts = _r0_quality_summary(valid[:-1])
    assert status == "FAIL"


def test_p4_rejects_output_inside_raw_package_before_creating_it(tmp_path: Path) -> None:
    t1 = tmp_path / "t1.json"
    t2 = tmp_path / "t2.json"
    thresholds = tmp_path / "thresholds.json"
    t1.write_text(json.dumps({"cases": []}), encoding="utf-8")
    t2.write_text(json.dumps({"cases": []}), encoding="utf-8")
    thresholds.write_text(
        json.dumps({"t1_high_snr": {}, "t2_independent": {}}),
        encoding="utf-8",
    )
    package = tmp_path / "raw_package"
    package.mkdir()
    manifest = tmp_path / "annotation_manifest.csv"
    poni = tmp_path / "geometry.poni"
    mask = tmp_path / "mask.npy"
    manifest.write_text("blind_id,source_path\n", encoding="utf-8")
    poni.write_text("poni", encoding="utf-8")
    mask.write_bytes(b"mask")
    output = package / "derived_results"

    with pytest.raises(ValueError, match="must not be written inside"):
        run_p4_engineering(
            t1_manifest=t1,
            t2_manifest=t2,
            thresholds=thresholds,
            output=output,
            r0_package=package,
            r0_manifest=manifest,
            poni=poni,
            mask=mask,
        )

    assert not output.exists()


def test_p4_rejects_incomplete_manifests_before_creating_output(tmp_path: Path) -> None:
    t1 = tmp_path / "t1.json"
    t2 = tmp_path / "t2.json"
    thresholds = tmp_path / "thresholds.json"
    output = tmp_path / "p4_output"
    t1.write_text(json.dumps({"cases": []}), encoding="utf-8")
    t2.write_text(json.dumps({"cases": []}), encoding="utf-8")
    thresholds.write_text(
        json.dumps({"t1_high_snr": {}, "t2_independent": {}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="T1 manifest case IDs must be exactly"):
        run_p4_engineering(
            t1_manifest=t1,
            t2_manifest=t2,
            thresholds=thresholds,
            output=output,
        )

    assert not output.exists()
