from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from butterfly_saxs.lamellar import (
    LamellarScene,
    LamellarSettings,
    build_lamellar_scene,
    load_lamellar_sources,
)


def _radial_source(*, q_unit: str = "nm^-1") -> dict:
    return {
        "source_identity": {
            "source": "frame_001.cbf",
            "frame": 3,
            "dataset": "entry/data",
        },
        "q_unit": q_unit,
        "draw_axis_deg": 90.0,
        "lobe_radial_peaks": [
            {
                "angle_deg": 30.0,
                "q_star": 0.2,
                "q_unit": q_unit,
                "valid": True,
                "branch_id": 0,
            },
            {
                "angle_deg": 210.0,
                "q_star": 0.2,
                "q_unit": q_unit,
                "valid": True,
                "branch_id": 0,
            },
            {
                "angle_deg": 150.0,
                "q_star": 0.1,
                "q_unit": q_unit,
                "valid": True,
                "branch_id": 1,
            },
            {
                "angle_deg": 330.0,
                "q_star": 0.1,
                "q_unit": q_unit,
                "valid": True,
                "branch_id": 1,
            },
        ],
    }


def test_settings_round_trip_and_validation() -> None:
    settings = LamellarSettings.from_mapping(
        {"layer_count": 3, "stack_count": 2, "mode": "multi"}
    )
    assert settings.to_dict()["layer_count"] == 3
    assert settings.to_dict()["mode"] == "multi"
    with pytest.raises(ValueError, match="thickness_ratio"):
        LamellarSettings(thickness_ratio=1.0)
    with pytest.raises(ValueError, match="selected_branch"):
        LamellarSettings(selected_branch=2)


def test_radial_scene_uses_physical_period_and_contract_shapes() -> None:
    scene = build_lamellar_scene(_radial_source(), {"layer_count": 2, "stack_count": 4})
    assert isinstance(scene, LamellarScene)
    assert scene.status == "schematic"
    assert scene.length_unit == "nm"
    assert scene.centers.shape == (4, 3)
    assert scene.sizes.shape == (4, 3)
    assert scene.orientations.shape == (4, 3, 3)
    assert scene.vertices.shape == (4, 8, 3)
    assert scene.stack_ids.shape == (4,)
    assert scene.branch_ids.shape == (4,)
    assert scene.colors.shape == (4, 4)
    assert scene.metadata["source_identity"]["frame"] == 3
    assert scene.metadata["populations"][0]["period"] == pytest.approx(
        2.0 * np.pi / 0.2
    )
    assert np.all(np.isfinite(scene.vertices))
    np.testing.assert_allclose(
        scene.vertices[0, 0],
        scene.centers[0]
        + scene.orientations[0]
        @ np.array(
            [-scene.sizes[0, 0] / 2, -scene.sizes[0, 1] / 2, -scene.sizes[0, 2] / 2]
        ),
    )


def test_draw_axis_rotation_and_opposite_support_are_explicit() -> None:
    source = _radial_source()
    source["draw_axis_deg"] = 0.0
    scene = build_lamellar_scene(source, {"layer_count": 1, "stack_count": 1})
    # 30° measured q plus (90 - draw_axis) display rotation gives 120°.
    normal = scene.orientations[0][:, 2]
    assert np.degrees(np.arctan2(normal[1], normal[0])) == pytest.approx(120.0)
    assert "inconsistent_opposite_support" not in scene.metadata["flags"]
    assert scene.metadata["populations"][0]["support_indices"] == [0, 1]


def test_unknown_q_is_relative_and_annulus_or_invalid_peaks_are_rejected() -> None:
    source = {
        "lobe_radial_peaks": [
            {
                "angle_deg": 20,
                "q": 2.0,
                "q_unit": "unknown",
                "method": "azimuthal_peak",
                "valid": True,
            },
            {"angle_deg": 20, "q_star": 0.0, "q_unit": "unknown", "valid": True},
            {"angle_deg": 20, "q_star": 1.0, "q_unit": "unknown", "valid": False},
        ]
    }
    scene = build_lamellar_scene(source, {"layer_count": 1, "stack_count": 1})
    assert scene.status == "unavailable"
    assert scene.centers.shape == (0, 3)
    assert scene.length_unit == "relative"
    assert "no_valid_lobe_radial_peaks" in scene.metadata["flags"]

    relative = build_lamellar_scene(
        {"lobe_radial_peaks": [{"angle_deg": 20, "q_star": 2.0, "valid": True}]},
        {"layer_count": 1, "stack_count": 1},
    )
    assert relative.status == "schematic"
    assert relative.length_unit == "relative"
    assert relative.metadata["parameter_sources"][0]["value"] == pytest.approx(1.0)


def test_historic_json_safe_ridgepoint_uses_only_radial_lobe_metadata() -> None:
    source = {
        "q_unit": "nm^-1",
        "lobe_radial_peaks": [
            {
                "qx": 0.2 * np.cos(np.deg2rad(30)),
                "qy": 0.2 * np.sin(np.deg2rad(30)),
                "q_unit": "nm^-1",
                "component": 0,
                "metadata": {
                    "accepted": True,
                    "angle": np.deg2rad(30),
                    "method": "radial_peak",
                    "flags": ["lobe_radial_peak"],
                },
            },
            {
                "qx": 0.2 * np.cos(np.deg2rad(210)),
                "qy": 0.2 * np.sin(np.deg2rad(210)),
                "q_unit": "nm^-1",
                "component": 0,
                "metadata": {
                    "accepted": True,
                    "angle": np.deg2rad(210),
                    "source_method": "radial_peak_in_observed_lobe_sector",
                    "flags": ["lobe_radial_peak"],
                },
            },
        ],
    }
    scene = build_lamellar_scene(source, {"layer_count": 1, "stack_count": 1})
    assert scene.metadata["available"]
    assert scene.metadata["populations"][0]["period"] == pytest.approx(
        2.0 * np.pi / 0.2, rel=1e-6
    )


def test_failed_or_nested_cancelled_sources_are_blank_but_manual_mode_survives() -> (
    None
):
    failed = _radial_source()
    failed["status"] = "failed"
    scene = build_lamellar_scene(failed)
    assert scene.status == "unavailable"
    assert not scene.metadata["available"]
    nested = {"result": {"status": "cancelled", **_radial_source()}}
    nested_scene = build_lamellar_scene(nested)
    assert nested_scene.status == "unavailable"
    manual = build_lamellar_scene(
        {"status": "failed"},
        {"period_source": "manual", "layer_count": 1, "stack_count": 1},
    )
    assert manual.status == "manual"
    assert manual.metadata["available"]


def test_unknown_native_q_mismatch_is_candidate_with_both_support_values() -> None:
    source = {
        "q_unit": "unknown",
        "lobe_radial_peaks": [
            {
                "angle_deg": 30.0,
                "q_star": 0.2,
                "q_unit": "unknown",
                "branch_id": 0,
                "valid": True,
            },
            {
                "angle_deg": 210.0,
                "q_star": 0.4,
                "q_unit": "unknown",
                "branch_id": 0,
                "valid": True,
            },
        ],
    }
    scene = build_lamellar_scene(source, {"layer_count": 1, "stack_count": 1})
    assert scene.status == "candidate"
    assert "inconsistent_opposite_support" in scene.metadata["flags"]
    assert scene.metadata["populations"][0]["support_q_values"] == pytest.approx(
        [0.2, 0.4]
    )
    mixed = {
        "lobe_radial_peaks": [
            {
                "angle_deg": 30.0,
                "q_star": 0.2,
                "q_unit": "unknown",
                "branch_id": 0,
                "valid": True,
            },
            {
                "angle_deg": 210.0,
                "q_star": 0.4,
                "q_unit": "pixel-q",
                "branch_id": 0,
                "valid": True,
            },
        ]
    }
    mixed_scene = build_lamellar_scene(mixed, {"layer_count": 1, "stack_count": 1})
    assert mixed_scene.status == "candidate"
    assert "mixed_q_unit_support" in mixed_scene.metadata["flags"]


def test_manual_mode_is_available_without_source() -> None:
    scene = build_lamellar_scene(
        {},
        {
            "period_source": "manual",
            "manual_period": 2.5,
            "manual_angle_deg": 15.0,
            "manual_unit": "relative",
            "layer_count": 2,
            "stack_count": 1,
        },
    )
    assert scene.status == "manual"
    assert scene.length_unit == "relative"
    assert scene.metadata["populations"][0]["status"] == "manual"
    assert scene.centers.shape == (2, 3)


def test_ellipse_requires_origin_and_keeps_candidate_formal_value_null() -> None:
    base = _radial_source()
    base["ellipse_fit"] = {"b": 0.1, "center": [0.01, 0.0], "q_unit": "nm^-1"}
    unavailable = build_lamellar_scene(
        base, {"period_source": "ellipse", "layer_count": 1, "stack_count": 1}
    )
    assert unavailable.status == "unavailable"
    assert "spacing_unavailable_nonzero_center" in unavailable.metadata["flags"]

    candidate = _radial_source()
    candidate["ellipse_fit"] = {
        "b": {"candidate_value": 0.1, "status": "candidate"},
        "center": [0.0, 0.0],
        "q_unit": "nm^-1",
    }
    scene = build_lamellar_scene(
        candidate, {"period_source": "ellipse", "layer_count": 1, "stack_count": 1}
    )
    assert scene.status == "candidate"
    source_row = next(
        item
        for item in scene.metadata["parameter_sources"]
        if item["name"] == "period_from_ellipse_minor_axis"
    )
    assert source_row["value"] is None
    assert source_row["status"] == "candidate"
    b_row = next(
        item for item in scene.metadata["parameter_sources"] if item["name"] == "b"
    )
    assert b_row["value"] is None
    assert b_row["candidate_value"] == pytest.approx(0.1)


def test_scene_is_deterministic_without_mutating_source() -> None:
    source = _radial_source()
    before = json.loads(json.dumps(source))
    settings = {
        "layer_count": 3,
        "stack_count": 2,
        "mode": "multi",
        "spread_deg": 8.0,
        "seed": 42,
    }
    first = build_lamellar_scene(source, settings)
    second = build_lamellar_scene(source, settings)
    np.testing.assert_array_equal(first.centers, second.centers)
    np.testing.assert_array_equal(first.orientations, second.orientations)
    np.testing.assert_array_equal(first.vertices, second.vertices)
    assert source == before


def test_multi_stack_count_is_total_and_fixed_slots_survive_missing_branch() -> None:
    both = build_lamellar_scene(
        _radial_source(),
        {
            "mode": "multi",
            "layer_count": 1,
            "stack_count": 12,
            "reference_period": 10.0,
            "seed": 7,
        },
    )
    assert len(both.centers) == 12
    assert set(both.stack_ids.tolist()) == set(range(12))
    assert (
        len(
            {(round(float(row[0]), 6), round(float(row[1]), 6)) for row in both.centers}
        )
        > 4
    )
    only_a = _radial_source()
    only_a["lobe_radial_peaks"] = only_a["lobe_radial_peaks"][:2]
    one = build_lamellar_scene(
        only_a,
        {
            "mode": "multi",
            "layer_count": 1,
            "stack_count": 12,
            "reference_period": 10.0,
            "seed": 7,
        },
    )
    assert len(one.centers) == 6
    assert set(one.stack_ids.tolist()) == {0, 2, 4, 6, 8, 10}
    both_by_id = {
        int(stack): center
        for stack, center in zip(both.stack_ids, both.centers)
        if int(stack) % 2 == 0
    }
    one_by_id = {
        int(stack): center for stack, center in zip(one.stack_ids, one.centers)
    }
    for stack in one_by_id:
        np.testing.assert_allclose(one_by_id[stack], both_by_id[stack])


def test_native_single_loader_embeds_arrays_and_batch_loader_is_lazy(
    tmp_path: Path,
) -> None:
    observed = np.arange(12, dtype=float).reshape(3, 4)
    qx = np.tile(np.arange(4, dtype=float), (3, 1))
    qy = np.tile(np.arange(3, dtype=float)[:, None], (1, 4))
    summary = {
        "q_unit": "nm^-1",
        "observables": {"lobe_radial_peaks": []},
        "metadata": {"path": "frame.npy"},
        "flags": {},
    }
    summary_path = tmp_path / "frame.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    np.savez_compressed(tmp_path / "frame.npz", image=observed, qx=qx, qy=qy)
    loaded = load_lamellar_sources(summary_path)
    assert len(loaded) == 1
    np.testing.assert_array_equal(loaded[0]["observed"], observed)
    np.testing.assert_array_equal(loaded[0]["qx"], qx)
    assert loaded[0]["source_identity"]["source"] == "frame.npy"

    manifest = {
        "schema_version": "lamellarsaxs2d.batch.v1",
        "frames": [
            {"frame_index": 0, "frame_id": "a"},
            {"frame_index": 1, "frame_id": "b"},
            {"frame_index": 2, "frame_id": "failed", "status": "failed"},
        ],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "ellipse_fit.json").write_text(
        json.dumps(
            {
                "frames": [
                    {
                        "frame_index": 0,
                        "ellipse_fit": {"b": 0.1},
                        "lobe_radial_peaks": [
                            {"angle_deg": 30, "q_star": 0.2, "valid": True}
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    np.savez_compressed(
        tmp_path / "results.npz",
        **{
            "frame_0000__image": observed,
            "frame_0000__qmap__qx": qx,
            "frame_0000__qmap__qy": qy,
            "frame_0001__image": observed + 1,
            "frame_0001__qmap__qx": qx,
            "frame_0001__qmap__qy": qy,
        },
    )
    batch = load_lamellar_sources(tmp_path)
    assert len(batch) == 3
    assert "observed" not in batch[0]
    assert batch[0]["array_keys"]["observed"] == "frame_0000__image"
    assert batch[0]["lobe_radial_peaks"][0]["q_star"] == pytest.approx(0.2)
    assert Path(batch[0]["npz_path"]).resolve() == (tmp_path / "results.npz").resolve()
    assert batch[2]["frame_id"] == "failed"
    assert "array_error" in batch[2]


def test_loader_rejects_csv_and_unrecognized_json(tmp_path: Path) -> None:
    csv_path = tmp_path / "values.csv"
    csv_path.write_text("q,value\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="native"):
        load_lamellar_sources(csv_path)
    json_path = tmp_path / "random.json"
    json_path.write_text(json.dumps({"foo": "bar"}), encoding="utf-8")
    with pytest.raises(ValueError, match="native"):
        load_lamellar_sources(json_path)


def test_existing_native_batch_bundle_contract_if_available() -> None:
    bundle = Path(
        r"G:\LamellarSAXS2D_butterfly_upgrade_20260906\results\batch_resume_release"
    )
    if not bundle.is_dir():
        pytest.skip("recorded native batch fixture is not mounted")
    sources = load_lamellar_sources(bundle)
    assert len(sources) >= 2
    assert all(source.get("array_keys", {}).get("observed") for source in sources)
    assert all(source.get("lobe_radial_peaks") is not None for source in sources)


def test_manual_stack_count_is_the_requested_total():
    scene = build_lamellar_scene(
        {}, {"period_source": "manual", "mode": "multi", "stack_count": 12}
    )
    assert len(np.unique(scene.stack_ids)) == 12
    assert len(scene.centers) == 12 * 8
