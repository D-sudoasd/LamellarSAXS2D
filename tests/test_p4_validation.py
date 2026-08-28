from __future__ import annotations

import json
from pathlib import Path

import pytest

from butterfly_saxs.cli import main
from butterfly_saxs.p4_validation import (
    _assigned_periodic_errors,
    run_p4_engineering,
)


def test_p4_periodic_lobe_assignment_wraps_at_180_degrees() -> None:
    errors = _assigned_periodic_errors([-179.5, 30.0], [179.5, 31.0])

    assert sorted(errors) == pytest.approx([1.0, 1.0])


def test_p4_cli_help_is_available() -> None:
    with pytest.raises(SystemExit) as error:
        main(["p4-evaluate", "--help"])

    assert error.value.code == 0


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
