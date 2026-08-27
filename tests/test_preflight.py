from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import butterfly_saxs.preflight as preflight_module
from butterfly_saxs.preflight import PreflightError, run_preflight
from butterfly_saxs.validation import (
    RESULT_SCHEMA_VERSION,
    ResultSchemaError,
    validate_result_schema,
)


def _write_image(package: Path, name: str = "image.npy", shape: tuple[int, int] = (3, 4)) -> Path:
    path = package / name
    np.save(path, np.arange(np.prod(shape), dtype=np.float32).reshape(shape))
    return path


def _preflight_extension(report: dict[str, object]) -> dict[str, object]:
    return report["extensions"]["preflight"]  # type: ignore[index,return-value]


def _check(report: dict[str, object], check_id: str) -> dict[str, object]:
    checks = report["quality"]["checks"]  # type: ignore[index]
    return next(item for item in checks if item["id"] == check_id)  # type: ignore[union-attr,return-value]


def test_minimal_npy_pixel_q_domain_and_strict_json(tmp_path: Path) -> None:
    image = _write_image(tmp_path)
    mask = tmp_path / "mask.npy"
    mask_values = np.zeros((3, 4), dtype=np.uint8)
    mask_values[0, 0] = 1
    np.save(mask, mask_values)

    report = run_preflight(
        tmp_path,
        image_glob=image.name,
        mask=mask.name,
        q_window=(0.0, 2.0),
    )

    assert report["status"]["status_color"] == "yellow"
    assert report["geometry"]["q_unit"] == "pixel-q"
    counts = report["analysis_domain"]["counts"]
    assert counts["external_mask_excluded_count"] == 1
    assert counts["fit_pixel_count"] == 11
    frames = _preflight_extension(report)["frames"]
    assert frames[0]["shape"] == [3, 4]
    assert frames[0]["dtype"] == "float32"
    assert report["result_type"] == "preflight"
    assert report["status"]["solver_status"] == "not_run"
    assert report["status"]["numerical_status"] == "NOT_TESTED"
    assert report["measurements"] is None
    assert report["fit"] is None
    assert report["provenance"]["arguments"]["mask"] == "mask.npy"
    assert report["provenance"]["arguments"]["q_window"] == [0.0, 2.0]
    assert len(report["provenance"]["hashes"]["source_tree_sha256"]) == 64
    assert report["provenance"]["dependencies"]["numpy"] == np.__version__
    assert set(_preflight_extension(report)["analysis_stages"].values()) == {"NOT_TESTED"}
    validate_result_schema(report)
    json.dumps(report, allow_nan=False)


def test_result_schema_validator_rejects_unknown_missing_and_nonfinite_fields(
    tmp_path: Path,
) -> None:
    image = _write_image(tmp_path)
    report = run_preflight(tmp_path, image_glob=image.name, q_window=(0.0, 2.0))

    unknown = {**report, "legacy_status": "yellow"}
    with pytest.raises(ResultSchemaError, match="unknown top-level"):
        validate_result_schema(unknown)
    missing = dict(report)
    missing.pop("geometry")
    with pytest.raises(ResultSchemaError, match="missing fields"):
        validate_result_schema(missing)
    nonfinite = json.loads(json.dumps(report))
    nonfinite["quality"]["metrics"]["coverage"] = float("nan")
    with pytest.raises(ResultSchemaError, match="non-finite"):
        validate_result_schema(nonfinite)
    string_nonfinite = json.loads(json.dumps(report))
    string_nonfinite["quality"]["metrics"]["coverage"] = "NaN"
    with pytest.raises(ResultSchemaError, match="string-form non-finite"):
        validate_result_schema(string_nonfinite)


def test_one_valid_zero_invalid_convention_is_applied(tmp_path: Path) -> None:
    image = _write_image(tmp_path)
    mask = tmp_path / "valid.npy"
    valid = np.ones((3, 4), dtype=np.uint8)
    valid[1, 2] = 0
    np.save(mask, valid)

    report = run_preflight(
        tmp_path,
        image_glob=image.name,
        mask=mask.name,
        mask_convention="1_valid_0_invalid",
        q_window=(0.0, 2.0),
    )

    assert report["analysis_domain"]["counts"]["external_mask_excluded_count"] == 1
    assert report["mask"]["source"]["raw_polarity"] == "1_valid_0_invalid"


def test_manifest_order_and_independent_hdf5_mask_selector(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    image_path = tmp_path / "image.h5"
    mask_path = tmp_path / "mask.h5"
    image = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    masks = np.zeros((2, 3, 4), dtype=np.uint8)
    masks[0, 0, 1] = 1
    masks[1, 2, 3] = 1
    with h5py.File(image_path, "w") as handle:
        handle.create_dataset("entry/data", data=image)
    with h5py.File(mask_path, "w") as handle:
        handle.create_dataset("mask/series", data=masks)

    manifest = [
        {"path": image_path.name, "order": "10", "frame": 1, "dataset": "entry/data"},
        {"path": image_path.name, "order": "2", "frame": 0, "dataset": "entry/data"},
    ]
    report = run_preflight(
        tmp_path,
        manifest=manifest,
        mask=mask_path.name,
        mask_frame=1,
        mask_dataset="mask/series",
        q_window=(0.0, 2.0),
    )

    frames = _preflight_extension(report)["frames"]
    assert [item["manifest_frame"]["order"] for item in frames] == [2, 10]
    assert frames[0]["frame"] == 0
    assert frames[1]["frame"] == 1
    assert report["analysis_domain"]["counts"]["external_mask_excluded_count"] == 1
    manifest_check = _check(report, "manifest")
    assert manifest_check["status"] == "WARN"  # time is intentionally absent
    assert manifest_check["observed"]["path_unique"] is False
    assert manifest_check["observed"]["selector_unique"] is True


def test_hdf5_image_and_npy_mask_do_not_share_dataset_selector(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    image_path = tmp_path / "image.h5"
    mask_path = tmp_path / "mask.npy"
    with h5py.File(image_path, "w") as handle:
        handle.create_dataset("entry/data", data=np.ones((2, 3, 4), dtype=np.float32))
    mask_values = np.zeros((3, 4), dtype=np.uint8)
    mask_values[0, 2] = 1
    np.save(mask_path, mask_values)

    report = run_preflight(
        tmp_path,
        manifest=[{"path": image_path.name, "frame": 1, "dataset": "entry/data"}],
        mask=mask_path.name,
        q_window=(0.0, 2.0),
    )

    first_frame = _preflight_extension(report)["frames"][0]
    assert first_frame["frame"] == 1
    assert first_frame["dataset"] == "entry/data"
    assert report["analysis_domain"]["counts"]["external_mask_excluded_count"] == 1


def test_mask_shape_mismatch_fails_closed(tmp_path: Path) -> None:
    _write_image(tmp_path)
    bad_mask = tmp_path / "bad_mask.npy"
    np.save(bad_mask, np.zeros((2, 2), dtype=np.uint8))
    with pytest.raises(PreflightError, match="mask shape"):
        run_preflight(tmp_path, image_glob="image.npy", mask=bad_mask.name)


def test_output_no_overwrite_and_force(tmp_path: Path) -> None:
    _write_image(tmp_path)
    output = tmp_path / "output"
    first = run_preflight(tmp_path, image_glob="image.npy", q_window=(0.0, 2.0), output=output)
    before = (output / "preflight.json").read_bytes()
    with pytest.raises(FileExistsError, match="force=True"):
        run_preflight(tmp_path, image_glob="image.npy", q_window=(0.0, 2.0), output=output)
    assert (output / "preflight.json").read_bytes() == before

    second = run_preflight(
        tmp_path,
        image_glob="image.npy",
        q_window=(0.0, 2.0),
        output=output,
        force=True,
    )
    assert second["outputs"]["directory"] == output.as_posix()
    assert second["outputs"]["files"]["preflight_json"] == "preflight.json"
    assert second["outputs"]["force"] is True
    assert set(second["outputs"]["overwritten_paths"]) == {
        "arrays.npz",
        "preflight.json",
        "run_report.md",
    }
    json.loads((output / "preflight.json").read_text(encoding="utf-8"))
    with np.load(output / "arrays.npz") as archive:
        expected_mask_keys = set(second["analysis_domain"]["arrays_npz"]["keys"].values())
        assert expected_mask_keys <= set(archive.files)
        for key in expected_mask_keys:
            assert archive[key].shape == (3, 4)
            assert archive[key].dtype == np.bool_
        assert int(np.count_nonzero(archive["fit_valid_mask"])) == second[
            "analysis_domain"
        ]["counts"]["fit_pixel_count"]
    assert first["analysis_domain"]["counts"] == second["analysis_domain"]["counts"]


def test_preflight_refuses_nonempty_output_directory_without_force(tmp_path: Path) -> None:
    _write_image(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    unrelated = output / "keep.txt"
    unrelated.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="force=True"):
        run_preflight(
            tmp_path,
            image_glob="image.npy",
            q_window=(0.0, 2.0),
            output=output,
        )

    assert unrelated.read_text(encoding="utf-8") == "keep"


def test_relative_output_is_resolved_from_current_working_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    package = tmp_path / "package"
    package.mkdir()
    _write_image(package)
    monkeypatch.chdir(tmp_path)

    report = run_preflight(package, image_glob="image.npy", q_window=(0.0, 2.0), output="relative-results")

    expected_directory = (tmp_path / "relative-results").as_posix()
    assert report["outputs"]["directory"] == expected_directory
    assert report["outputs"]["files"]["preflight_json"] == "preflight.json"
    assert (tmp_path / "relative-results" / "preflight.json").is_file()
    assert not (package / "relative-results").exists()


def test_source_intensity_derives_external_recipe_correction_state(tmp_path: Path) -> None:
    image = _write_image(tmp_path)
    context = {
        "preferred_project_input": {"image_glob": image.name},
        "source_intensity": {
            "already_applied": ["background_subtraction"],
            "not_burned_into_2d_values": ["solid_angle", "polarization"],
            "uncertainty_status": "partial",
        },
    }

    report = run_preflight(tmp_path, context=context, q_window=(0.0, 2.0))

    assert report["correction_state"] == "external_recipe_declared"
    assert report["uncertainty_state"] == "partial"
    correction_check = _check(report, "correction_state")
    assert correction_check["status"] == "WARN"
    assert "solid-angle/polarization" in correction_check["message"]


def test_uncertainty_hdf5_source_and_components_are_reported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    h5py = pytest.importorskip("h5py")
    image = _write_image(tmp_path)
    uncertainty_dir = tmp_path / "images_h5"
    uncertainty_dir.mkdir()
    uncertainty_file = uncertainty_dir / "image_uncertainty.h5"
    with h5py.File(uncertainty_file, "w") as handle:
        group = handle.create_group("entry/data/uncertainty")
        dataset = group.create_dataset("statistical", data=np.ones((3, 4)))
        dataset.attrs["units"] = "cm^-1"

    original_load_image = preflight_module.load_image

    def load_with_uncertainty_header(path, **kwargs):
        loaded = original_load_image(path, **kwargs)
        loaded.metadata["header"] = {
            "UncertaintyStatus": "partial",
            "UncertaintyHDF5": rf"H:\archive\images_h5\{uncertainty_file.name}",
        }
        return loaded

    monkeypatch.setattr(preflight_module, "load_image", load_with_uncertainty_header)
    report = run_preflight(
        tmp_path,
        image_glob=image.name,
        q_window=(0.0, 2.0),
        uncertainty_state="partial",
    )

    uncertainty = report["uncertainty"]
    assert uncertainty["resolved_file_count"] == 1
    assert uncertainty["sources"] == ["images_h5/image_uncertainty.h5"]
    assert set(uncertainty["components"]) == {"statistical"}
    assert uncertainty["units"] == ["cm^-1"]
    assert uncertainty["components"]["statistical"]["dataset"] == "entry/data/uncertainty/statistical"
    uncertainty_hash = next(
        item for item in report["provenance"]["hashes"]["files"]
        if item["role"] == "uncertainty"
    )
    assert uncertainty_hash["unchanged"] is True


def test_complete_uncertainty_without_source_inventory_is_red(tmp_path: Path) -> None:
    image = _write_image(tmp_path)

    report = run_preflight(
        tmp_path,
        image_glob=image.name,
        q_window=(0.0, 2.0),
        uncertainty_state="complete",
    )

    assert report["status"]["status_color"] == "red"
    assert report["status"]["scientific_status"] == "FAIL"
    assert report["status"]["exit_code"] == 1
    uncertainty_check = _check(report, "uncertainty_state")
    assert uncertainty_check["status"] == "FAIL"
    assert "file/dataset/unit inventory" in uncertainty_check["message"]


def test_manifest_duplicate_order_and_path_are_red(tmp_path: Path) -> None:
    image = _write_image(tmp_path)
    manifest = [
        {"path": image.name, "order": 1, "time": 0},
        {"path": image.name, "order": 1, "time": 1},
    ]

    report = run_preflight(tmp_path, manifest=manifest, q_window=(0.0, 2.0))

    assert report["status"]["status_color"] == "red"
    assert report["status"]["scientific_status"] == "FAIL"
    assert report["status"]["exit_code"] == 2
    manifest_check = _check(report, "manifest")
    assert manifest_check["status"] == "FAIL"
    assert manifest_check["observed"]["duplicate_orders"] == [1.0]
    assert manifest_check["observed"]["path_unique"] is False
    assert manifest_check["observed"]["selector_unique"] is False


def test_manifest_nonmonotonic_time_is_red_and_result_schema_is_reported(tmp_path: Path) -> None:
    first = _write_image(tmp_path, "frame_1.npy")
    second = _write_image(tmp_path, "frame_2.npy")
    report = run_preflight(
        tmp_path,
        manifest=[
            {"path": first.name, "order": 1, "time": 10},
            {"path": second.name, "order": 2, "time": 5},
        ],
        q_window=(0.0, 2.0),
    )

    assert report["status"]["status_color"] == "red"
    assert report["status"]["scientific_status"] == "FAIL"
    assert report["status"]["exit_code"] == 2
    assert report["schema_version"] == RESULT_SCHEMA_VERSION
    manifest_check = _check(report, "manifest")
    assert manifest_check["observed"]["time_monotonic"] is False


def test_manifest_missing_time_is_yellow_with_structured_evidence(tmp_path: Path) -> None:
    first = _write_image(tmp_path, "frame_1.npy")
    second = _write_image(tmp_path, "frame_2.npy")
    report = run_preflight(
        tmp_path,
        manifest=[{"path": first.name, "order": 1}, {"path": second.name, "order": 2}],
        q_window=(0.0, 2.0),
    )

    manifest_check = _check(report, "manifest")
    assert manifest_check["status"] == "WARN"
    assert manifest_check["observed"]["missing_time_indices"] == [0, 1]


def test_csv_blank_order_falls_back_to_numeric_time(tmp_path: Path) -> None:
    first = _write_image(tmp_path, "frame_1.npy")
    second = _write_image(tmp_path, "frame_2.npy")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "path,frame_id,order,time\n"
        f"{first.name},late,,10\n"
        f"{second.name},early,,2\n",
        encoding="utf-8",
    )

    report = run_preflight(tmp_path, manifest=manifest.name, q_window=(0.0, 2.0))

    frames = _preflight_extension(report)["frames"]
    assert [item["id"] for item in frames] == ["early", "late"]
    manifest_check = _check(report, "manifest")
    assert manifest_check["observed"]["explicit_order"] is False
    assert manifest_check["observed"]["time_values"] == [2.0, 10.0]
    assert manifest_check["observed"]["time_monotonic"] is True


def test_csv_blank_order_and_time_preserve_original_row_order(tmp_path: Path) -> None:
    first = _write_image(tmp_path, "frame_10.npy")
    second = _write_image(tmp_path, "frame_2.npy")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "path,frame_id,order,time\n"
        f"{first.name},first,,\n"
        f"{second.name},second,,\n",
        encoding="utf-8",
    )

    report = run_preflight(tmp_path, manifest=manifest.name, q_window=(0.0, 2.0))

    frames = _preflight_extension(report)["frames"]
    assert [item["id"] for item in frames] == ["first", "second"]
    assert _check(report, "manifest")["observed"]["explicit_order"] is False


def test_explicit_numeric_order_controls_time_monotonicity_check(tmp_path: Path) -> None:
    first = _write_image(tmp_path, "frame_1.npy")
    second = _write_image(tmp_path, "frame_2.npy")

    report = run_preflight(
        tmp_path,
        manifest=[
            {"path": second.name, "frame_id": "second", "order": "2", "time": "20"},
            {"path": first.name, "frame_id": "first", "order": "1", "time": "10"},
        ],
        q_window=(0.0, 2.0),
    )

    frames = _preflight_extension(report)["frames"]
    assert [item["id"] for item in frames] == ["first", "second"]
    manifest_check = _check(report, "manifest")
    assert manifest_check["observed"]["time_values"] == [10.0, 20.0]
    assert manifest_check["observed"]["time_monotonic"] is True


def test_angstrom_inverse_qmap_is_converted_to_nm_inverse(tmp_path: Path) -> None:
    image = _write_image(tmp_path)
    qx = np.full((3, 4), 0.1)
    qy = np.zeros((3, 4))
    output = tmp_path / "evidence"

    report = run_preflight(
        tmp_path,
        image_glob=image.name,
        poni={"qx": qx, "qy": qy, "q": np.hypot(qx, qy), "q_unit": "Å^-1"},
        q_window=(0.5, 1.5),
        output=output,
    )

    assert report["geometry"]["q_unit"] == "nm^-1"
    assert report["geometry"]["source_q_unit"] == "Å^-1"
    assert report["geometry"]["q_conversion_factor_to_nm_inv"] == 10.0
    assert report["geometry"]["q_range"] == {"min": 1.0, "max": 1.0, "unit": "nm^-1"}
    assert report["analysis_domain"]["counts"]["fit_pixel_count"] == 12
    with np.load(output / "arrays.npz") as archive:
        assert np.allclose(archive["qx_nm_inv"], 1.0)
        assert np.allclose(archive["q_nm_inv"], 1.0)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"mask_convention": "bad"}, "mask_convention"),
        ({"q_window": (1.0, 1.0)}, "q_window"),
    ],
)
def test_invalid_preflight_parameters_fail_closed(
    tmp_path: Path, kwargs: dict[str, object], message: str
) -> None:
    _write_image(tmp_path)
    with pytest.raises(PreflightError, match=message):
        run_preflight(tmp_path, image_glob="image.npy", **kwargs)


def test_missing_manifest_path_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(PreflightError, match="manifest file does not exist"):
        run_preflight(tmp_path, manifest="missing.json")


def test_input_hash_matches_sha256_and_fit_mask_count(tmp_path: Path) -> None:
    image = _write_image(tmp_path)
    report = run_preflight(tmp_path, image_glob=image.name, q_window=(0.0, 2.0))
    expected = hashlib.sha256(image.read_bytes()).hexdigest()
    entry = next(
        item for item in report["provenance"]["hashes"]["files"]
        if item["role"] == "image"
    )
    assert entry["before"] == expected
    assert entry["after"] == expected
    assert entry["unchanged"] is True
    with pytest.raises(KeyError):
        entry["not-a-real-field"]
