from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from butterfly_saxs import benchmark_t2


def test_default_cases_cover_required_categories() -> None:
    assert {case.category for case in benchmark_t2.DEFAULT_CASES} == {
        "2-point",
        "eyebrow",
        "butterfly",
        "non_elliptical",
    }
    by_category = {case.category: case for case in benchmark_t2.DEFAULT_CASES}
    assert len(by_category["eyebrow"].orientation_offsets_deg) >= 30
    assert len(by_category["butterfly"].orientation_offsets_deg) >= 40


def test_generation_is_deterministic_and_independent() -> None:
    first = benchmark_t2.generate_case("butterfly", shape=(48, 52))
    second = benchmark_t2.generate_case("butterfly", shape=(48, 52))
    for key in (
        "real_space_density",
        "intensity_noiseless",
        "intensity_noisy",
        "qx",
        "qy",
        "q",
        "mask",
        "projection_reference",
    ):
        assert np.array_equal(first[key], second[key]), key
    assert first["projection_truth"] == second["projection_truth"]
    assert first["structure_truth"] == second["structure_truth"]

    source = inspect.getsource(benchmark_t2)
    assert "from .intensity" not in source
    assert "from .synthetic" not in source


def test_categories_produce_distinct_clean_images() -> None:
    results = [benchmark_t2.generate_case(case, shape=(40, 44)) for case in benchmark_t2.DEFAULT_CASES]
    images = [result["intensity_noiseless"] for result in results]
    assert all(not np.array_equal(images[0], image) for image in images[1:])


def test_fft_outputs_are_finite_nonnegative_and_shape_consistent() -> None:
    result = benchmark_t2.generate_case("non_elliptical", shape=(36, 40), noise_sigma=0.01)
    shape = (36, 40)
    for key in (
        "real_space_density",
        "intensity_noiseless",
        "intensity_noisy",
        "noise",
        "qx",
        "qy",
        "q",
        "mask",
        "valid_mask",
    ):
        assert result[key].shape == shape, key
        assert np.isfinite(result[key]).all(), key
    assert np.all(result["intensity_noiseless"] >= 0)
    assert np.all(result["intensity_noisy"] >= 0)
    assert result["mask"].dtype == bool
    assert np.isclose(result["intensity_noiseless"].max(), 1.0)


def test_projection_and_structure_truth_are_separate_scopes() -> None:
    result = benchmark_t2.generate_case("eyebrow", shape=(32, 32))
    projection = result["projection_truth"]
    structure = result["structure_truth"]
    assert projection is not structure
    assert projection["truth_scope"] == "projection_only_for_empirical_pipeline_validation"
    assert structure["truth_scope"] == "generator_only_for_physical_forward_validation"
    assert "ridges" in projection
    assert "layer_spacing_nm" in structure
    assert "orientation_distribution" in structure
    assert "curvature" in structure
    assert "layer_spacing_nm" not in projection
    assert "ridges" not in structure


def test_analytic_projection_references_are_independent_and_near_fft_features() -> None:
    for case in benchmark_t2.DEFAULT_CASES:
        result = benchmark_t2.generate_case(case, shape=(96, 96), noise_sigma=0.0)
        reference = result["projection_reference"]
        qx = result["qx"]
        qy = result["qy"]
        intensity = result["intensity_noiseless"]
        nearby_maxima = []
        q_resolution = result["projection_truth"]["q_grid_resolution_nm_inv"]
        for expected_qx, expected_qy in reference:
            distance = np.hypot(qx - expected_qx, qy - expected_qy)
            nearby_maxima.append(float(np.max(intensity[distance <= 2.5 * q_resolution])))
        assert np.median(nearby_maxima) > np.percentile(intensity, 99.0)
        assert result["projection_truth"]["reference_method"] == (
            "analytic_bragg_vectors_from_generator_structure"
        )
        assert result["projection_truth"]["independent_of_generated_fft_pixels"] is True


def test_spacing_jitter_preserves_layer_order() -> None:
    result = benchmark_t2.generate_case("non_elliptical", shape=(64, 64))
    for component in result["structure_truth"]["realized_components"]:
        assert np.all(np.diff(component["layer_positions_nm"]) > 0)


def test_evidence_manifest_is_finite_and_no_overwrite(tmp_path: Path) -> None:
    output_dir = tmp_path / "t2"
    manifest_path = benchmark_t2.write_evidence_directory(
        output_dir,
        cases=("2-point", "butterfly"),
        shape=(24, 28),
        seed=99,
        noise_sigma=0.0,
    )
    assert manifest_path == output_dir / "truth_manifest.json"
    assert (output_dir / "2-point.npz").exists()
    assert (output_dir / "butterfly.npz").exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["model_scope"] == "independent_physical_synthetic"
    assert manifest["generator_version"] == benchmark_t2.GENERATOR_VERSION
    assert manifest["generator_hash"] == benchmark_t2.GENERATOR_HASH
    assert manifest["cases"][0]["seed"] == 99
    assert manifest["cases"][1]["seed"] == 100
    for record in manifest["cases"]:
        artifact = output_dir / record["npz_file"]
        assert record["npz_sha256"] == __import__("hashlib").sha256(
            artifact.read_bytes()
        ).hexdigest()
    assert "intensity_noisy" not in json.dumps(manifest)

    keep = output_dir / "unrelated.txt"
    keep.write_text("keep", encoding="utf-8")
    original = (output_dir / "2-point.npz").read_bytes()
    with pytest.raises(FileExistsError):
        benchmark_t2.write_evidence_directory(output_dir, cases=("2-point",), shape=(24, 28))
    assert (output_dir / "2-point.npz").read_bytes() == original
    assert keep.read_text(encoding="utf-8") == "keep"

    benchmark_t2.write_evidence_directory(output_dir, cases=("2-point",), shape=(24, 28), force=True)
    assert keep.read_text(encoding="utf-8") == "keep"


def test_npz_contains_projection_reference_and_truth_sidecars(tmp_path: Path) -> None:
    output_dir = tmp_path / "evidence"
    benchmark_t2.write_evidence_directory(output_dir, cases=("2-point",), shape=(24, 24), noise_sigma=0.0)
    with np.load(output_dir / "2-point.npz", allow_pickle=False) as archive:
        required = {
            "real_space_density",
            "intensity_noiseless",
            "intensity_noisy",
            "qx",
            "qy",
            "q",
            "mask",
            "projection_reference",
            "projection_truth_json",
            "structure_truth_json",
        }
        assert required.issubset(set(archive.files))
        projection = json.loads(str(archive["projection_truth_json"].item()))
        structure = json.loads(str(archive["structure_truth_json"].item()))
        assert projection["truth_scope"].startswith("projection_only")
        assert structure["truth_scope"].startswith("generator_only")


def test_evidence_refuses_data_local_output(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="data_local"):
        benchmark_t2.write_evidence_directory(
            tmp_path / "data_local" / "t2",
            cases=("2-point",),
            shape=(24, 24),
        )
