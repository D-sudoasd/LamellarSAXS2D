from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from butterfly_saxs.benchmark_t1 import (
    DEFAULT_CASES,
    T1_Q_UNIT,
    default_cases,
    generate_case,
    write_evidence_directory,
)


def test_default_matrix_covers_requested_t1_categories_and_parameters() -> None:
    cases = default_cases()
    assert len(cases) >= 12
    assert {case.noise_model for case in cases} >= {"none", "gaussian", "poisson"}
    assert {artifact for case in cases for artifact in case.artifacts} >= {
        "beamstop",
        "streak",
        "gap",
        "bad_points",
        "missing_sector",
    }
    assert any(case.low_snr for case in cases)
    assert any(case.overlap for case in cases)
    assert any(case.non_elliptic for case in cases)
    assert any(case.center_offset != (0.0, 0.0) for case in cases)
    assert any(case.shape[0] != case.shape[1] for case in cases)
    assert any(case.q_range != (-1.25, 1.25) for case in cases)
    all_parameters = {key for case in cases for key in case.parameters}
    assert {"a", "b", "theta_deg", "lobe_angle_deg", "angular_width_deg", "amplitude", "background"} <= all_parameters
    assert tuple(case.name for case in cases) == tuple(case.name for case in DEFAULT_CASES)


def test_single_case_is_deterministic_and_obeys_array_contract() -> None:
    first = generate_case("gaussian_parameter_sweep")
    second = generate_case("gaussian_parameter_sweep")
    for key in first.arrays():
        np.testing.assert_array_equal(first[key], second[key])
    assert first.intensity.shape == first.qx.shape == first.qy.shape == first.q.shape == first.mask.shape
    assert first.mask.dtype == bool
    assert first.truth["q_unit"] == T1_Q_UNIT
    assert first.truth["mask"]["shape"] == list(first.shape)
    assert np.isfinite(first.intensity).all()
    assert np.isfinite(first.truth_intensity).all()


def test_case_input_accepts_small_explicit_spec_and_center_q_range() -> None:
    sample = generate_case(
        {
            "name": "explicit",
            "shape": (24, 30),
            "q_range": ((-0.7, 1.1), (-0.4, 0.9)),
            "center_offset": (1.25, -2.0),
            "parameters": {
                "a": 0.6,
                "b": 0.4,
                "theta_deg": 9.0,
                "lobe_angle_deg": 32.0,
                "width": 0.12,
                "amplitude": 1.8,
                "background": 0.02,
            },
        },
        seed=17,
    )
    assert sample.shape == (24, 30)
    assert sample.truth["center_offset_px_dy_dx"] == [1.25, -2.0]
    assert sample.truth["q_range"] == [[-0.7, 1.1], [-0.4, 0.9]]


def test_evidence_npz_truth_json_and_no_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "evidence"
    manifest_result = write_evidence_directory(output, cases=["noiseless_default", "poisson_counting"], seed=8)
    manifest_path = output / "truth_manifest.json"
    assert manifest_result == manifest_path
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["same_model"] is True
    assert manifest["scientific_scope"]
    assert manifest["case_count"] == 2
    for record in manifest["cases"]:
        npz_path = output / record["npz"]
        truth_path = output / record["truth_json"]
        assert npz_path.exists() and truth_path.exists()
        with np.load(npz_path) as arrays:
            required = {"intensity", "qx", "qy", "q", "mask", "truth_intensity", "noise"}
            assert required <= set(arrays.files)
            assert arrays["intensity"].shape == tuple(record["shape"])
            assert arrays["mask"].dtype == bool
        truth = json.loads(truth_path.read_text(encoding="utf-8"))
        assert truth["q_unit"] == T1_Q_UNIT
        assert truth["files"]["npz"] == npz_path.name
        assert "NaN" not in truth_path.read_text(encoding="utf-8")
        assert "Infinity" not in truth_path.read_text(encoding="utf-8")

    before = manifest_path.read_bytes()
    with pytest.raises(FileExistsError):
        write_evidence_directory(output, cases=["noiseless_default", "poisson_counting"], seed=8)
    assert manifest_path.read_bytes() == before

    unrelated = output / "keep.me"
    unrelated.write_text("leave me", encoding="utf-8")
    write_evidence_directory(output, cases=["noiseless_default"], seed=9, force=True)
    assert unrelated.read_text(encoding="utf-8") == "leave me"


def test_evidence_refuses_data_local_output(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="data_local"):
        write_evidence_directory(
            tmp_path / "data_local" / "t1",
            cases=["noiseless_default"],
        )
