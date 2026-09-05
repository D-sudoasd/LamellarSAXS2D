from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import butterfly_saxs.annotation_pack as annotation_pack
from butterfly_saxs.annotation_pack import AnnotationPackError, build_annotation_pack
from butterfly_saxs.benchmark_t1 import DEFAULT_CASE_NAMES, generate_case as generate_t1_case
from butterfly_saxs.benchmark_t2 import generate_case
from butterfly_saxs.p4_validation import _t1_visible_ridge_angles
from butterfly_saxs.preflight import PreflightError, run_preflight


def _write_image(path: Path, value: float = 1.0) -> Path:
    np.save(path, np.full((3, 4), value, dtype=np.float32))
    return path


def test_manifest_rows_are_relative_to_manifest_parent(tmp_path: Path) -> None:
    package = tmp_path / "package"
    nested = package / "manifests" / "frames"
    nested.mkdir(parents=True)
    image = _write_image(nested / "frame.npy")
    manifest = package / "manifests" / "frames.json"
    manifest.write_text(json.dumps([{"path": "frames/frame.npy"}]), encoding="utf-8")

    report = run_preflight(package, manifest=manifest.relative_to(package), q_window=(0.0, 2.0))

    assert report["input"]["images"][0]["path"] == "manifests/frames/frame.npy"
    assert image.is_file()


def test_package_symlink_escape_is_rejected_but_explicit_external_root_is_allowed(
    tmp_path: Path,
) -> None:
    package = tmp_path / "package"
    external = tmp_path / "external"
    package.mkdir()
    external.mkdir()
    frame = _write_image(external / "frame.npy")
    with pytest.raises(PreflightError, match="authorized package/external roots"):
        run_preflight(package, manifest=[{"path": "../external/frame.npy"}], q_window=(0.0, 2.0))
    link = package / "linked"
    try:
        link.symlink_to(external, target_is_directory=True)
    except (OSError, NotImplementedError):  # pragma: no cover - Windows policy dependent
        link = None
    if link is not None:
        with pytest.raises(PreflightError, match="authorized package/external roots"):
            run_preflight(package, manifest=[{"path": "linked/frame.npy"}], q_window=(0.0, 2.0))

    report = run_preflight(
        package,
        manifest=[{"path": str(frame)}],
        external_roots=[external],
        q_window=(0.0, 2.0),
    )
    assert report["input"]["images"][0]["path"] == frame.resolve().as_posix()


def test_explicit_qmap_combines_positive_and_negative_masks_and_rejects_bad_q(
    tmp_path: Path,
) -> None:
    image = _write_image(tmp_path / "frame.npy")
    qx, qy = np.meshgrid(np.linspace(-0.5, 0.5, 4), np.linspace(-0.4, 0.4, 3))
    valid = np.ones(qx.shape, dtype=bool)
    valid[0, 0] = False
    invalid = np.zeros(qx.shape, dtype=bool)
    invalid[1, 1] = True
    report = run_preflight(
        tmp_path,
        manifest=[{"path": image.name}],
        poni={
            "qx": qx,
            "qy": qy,
            "q": np.hypot(qx, qy),
            "q_unit": "nm^-1",
            "valid_mask": valid,
            "mask": invalid,
        },
        q_window=(0.0, 2.0),
    )
    assert report["analysis_domain"]["counts"]["fit_pixel_count"] == 10

    with pytest.raises(PreflightError, match="coordinates are inconsistent"):
        run_preflight(
            tmp_path,
            manifest=[{"path": image.name}],
            poni={"qx": qx, "qy": qy, "q": np.full(qx.shape, 99.0), "q_unit": "nm^-1"},
            q_window=(0.0, 100.0),
        )


def test_uncertainty_inventory_hashes_every_declared_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    h5py = pytest.importorskip("h5py")
    image_a = _write_image(tmp_path / "a.npy", 1.0)
    image_b = _write_image(tmp_path / "b.npy", 2.0)
    first = tmp_path / "u1.h5"
    second = tmp_path / "u2.h5"
    with h5py.File(first, "w") as handle:
        dataset = handle.create_dataset("entry/data/uncertainty/sigma", data=np.ones((3, 4)))
        dataset.attrs["units"] = "counts"
    with h5py.File(second, "w") as handle:
        handle.create_group("entry/data/other")

    import butterfly_saxs.preflight as preflight

    original = preflight.load_image

    def with_header(path: str | Path, **kwargs: object):
        loaded = original(path, **kwargs)
        loaded.metadata["header"] = {
            "UncertaintyStatus": "complete",
            "UncertaintyHDF5": "u1.h5" if Path(path).name == image_a.name else "u2.h5",
        }
        return loaded

    monkeypatch.setattr(preflight, "load_image", with_header)
    report = run_preflight(
        tmp_path,
        manifest=[{"path": image_a.name}, {"path": image_b.name}],
        uncertainty_state="complete",
        q_window=(0.0, 2.0),
    )
    uncertainty = report["uncertainty"]
    assert uncertainty["declared_file_count"] == 2
    assert uncertainty["resolved_file_count"] == 2
    assert uncertainty["dataset_inventory_status"] == "partial_invalid"
    uncertainty_check = next(
        item for item in report["quality"]["checks"] if item["id"] == "uncertainty_state"
    )
    assert uncertainty_check["status"] == "FAIL"
    assert report["status"]["status_color"] == "red"
    assert report["status"]["scientific_status"] == "FAIL"
    assert report["status"]["exit_code"] == 1
    assert {item["status"] for item in uncertainty["file_inventory"]} == {
        "reference_schema_read",
        "uncertainty_group_missing",
    }
    records = [item for item in report["provenance"]["hashes"]["files"] if item["role"] == "uncertainty"]
    assert len(records) == 2
    assert all(item["before"] == item["after"] == item["sha256"] for item in records)


def test_unreadable_uncertainty_file_keeps_before_after_hash_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("h5py")
    image = _write_image(tmp_path / "frame.npy")
    bad = tmp_path / "bad.h5"
    bad.write_bytes(b"not an HDF5 file")

    import butterfly_saxs.preflight as preflight

    original = preflight.load_image

    def with_header(path: str | Path, **kwargs: object):
        loaded = original(path, **kwargs)
        loaded.metadata["header"] = {
            "UncertaintyStatus": "partial",
            "UncertaintyHDF5": bad.name,
        }
        return loaded

    monkeypatch.setattr(preflight, "load_image", with_header)
    report = run_preflight(
        tmp_path,
        manifest=[{"path": image.name}],
        uncertainty_state="partial",
        q_window=(0.0, 2.0),
    )
    inventory = report["uncertainty"]["file_inventory"]
    assert inventory[0]["status"] == "reference_hdf5_unreadable"
    assert inventory[0]["reason"]
    records = [item for item in report["provenance"]["hashes"]["files"] if item["role"] == "uncertainty"]
    assert len(records) == 1
    assert records[0]["before"] == records[0]["after"] == records[0]["sha256"]
    assert records[0]["read_status"] == "error"
    assert records[0]["read_error"]


def test_annotation_csv_escapes_formula_text_without_changing_numbers(tmp_path: Path) -> None:
    output = tmp_path / "annotation.csv"
    annotation_pack._write_csv(
        output,
        ("notes", "value"),
        ({"notes": "=HYPERLINK(\"secret\")", "value": 4.5},),
    )
    raw = output.read_text(encoding="utf-8")
    assert "'=HYPERLINK" in raw
    with output.open("r", encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["value"] == "4.5"


def test_annotation_pack_external_roots_are_explicit_and_auditable(tmp_path: Path) -> None:
    package = tmp_path / "package"
    external = tmp_path / "external"
    package.mkdir()
    external.mkdir()
    paths = []
    for index in range(8):
        paths.append(_write_image(external / f"frame_{index}.npy", float(index)))
    rt_manifest = package / "rt.json"
    hold_manifest = package / "hold.json"
    rt_manifest.write_text(json.dumps([{"path": str(paths[0])}]), encoding="utf-8")
    hold_manifest.write_text(
        json.dumps([{"path": str(path)} for path in paths]), encoding="utf-8"
    )
    with pytest.raises(AnnotationPackError, match="authorized package/external roots"):
        build_annotation_pack(package, rt_manifest, hold_manifest, tmp_path / "blocked")

    output = tmp_path / "allowed"
    result = build_annotation_pack(
        package,
        rt_manifest,
        hold_manifest,
        output,
        external_roots=[external],
    )
    with (output / "annotation_manifest.csv").open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert result["candidate_count"] == 8
    assert all(Path(row["source_path"]).is_absolute() for row in rows)


def test_t2_mapping_rejects_unknown_keys() -> None:
    with pytest.raises(ValueError, match="unknown T2 case mapping key"):
        generate_case({"name": "butterfly", "layer_spacng_nm": 99})


def test_t1_truth_matrix_is_frozen_and_poisson_support_is_analytic() -> None:
    assert len(DEFAULT_CASE_NAMES) == 15
    sample = generate_t1_case("poisson_counting")
    angles = _t1_visible_ridge_angles(sample.arrays())
    assert angles is not None
    assert 0 < len(angles) <= 72
    assert sample.truth_ridge_support.dtype == bool
    parameters = sample.truth["truth_parameters"]
    angle = np.arctan2(sample.qy, sample.qx)

    def explicit_radius(theta: float) -> np.ndarray:
        relative = angle - theta
        denominator = (np.cos(relative) / float(parameters["a"])) ** 2 + (
            np.sin(relative) / float(parameters["b"])
        ) ** 2
        return 1.0 / np.sqrt(denominator)

    np.testing.assert_allclose(sample.truth_ridge_plus, explicit_radius(float(parameters["theta"])))
    np.testing.assert_allclose(
        sample.truth_ridge_minus,
        explicit_radius(-float(parameters["theta"])),
    )
    expected_support = (
        np.isfinite(sample.truth_ridge_plus)
        & np.isfinite(sample.truth_ridge_minus)
        & np.isfinite(sample.q)
        & (sample.q >= 0.15)
        & (sample.q <= 1.25)
    )
    np.testing.assert_array_equal(sample.truth_ridge_support, expected_support)
    assert sample.poisson_counts is not None
    assert np.issubdtype(sample.poisson_counts.dtype, np.integer)
    np.testing.assert_allclose(
        sample.intensity,
        sample.poisson_counts / float(sample.truth["poisson_scale"]),
    )


def test_doctor_rejects_fake_zero_versions_and_package_root_is_lazy(tmp_path: Path) -> None:
    from butterfly_saxs import doctor

    report = doctor.collect_diagnostics(
        cwd=Path.cwd(),
        importer=lambda _module: object(),
        version_getter=lambda _distribution: "0.0.0",
        version_info=(3, 12, 0),
    )
    assert report["ready"] is False
    assert "NumPy" in report["required_failures"]

    code = (
        "import sys; import butterfly_saxs; "
        "assert 'numpy' not in sys.modules; "
        "from butterfly_saxs import doctor; "
        "assert doctor.collect_diagnostics(version_info=(3,12,0), "
        "importer=lambda _: object(), version_getter=lambda _: '0.0.0')['ready'] is False"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path.cwd(),
        env={
            **os.environ,
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(Path.cwd() / "src"),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
