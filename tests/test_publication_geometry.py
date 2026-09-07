from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from butterfly_saxs.lamellar import build_lamellar_scene
from butterfly_saxs.publication_geometry import (
    PublicationMesh,
    build_publication_mesh,
    camera_basis,
    fit_publication_camera,
    project_points,
)
from butterfly_saxs.publication_models import PublicationStyle


def _scene(*, mode: str = "single", layer_count: int = 2, stack_count: int = 2):
    source = {
        "source_identity": {"source": "publication-fixture", "frame": 4},
        "q_unit": "nm^-1",
        "draw_axis_deg": 90.0,
        "lobe_radial_peaks": [
            {
                "angle_deg": 30.0,
                "q_star": 0.20,
                "q_unit": "nm^-1",
                "branch_id": 0,
                "valid": True,
            },
            {
                "angle_deg": 210.0,
                "q_star": 0.20,
                "q_unit": "nm^-1",
                "branch_id": 0,
                "valid": True,
            },
            {
                "angle_deg": 150.0,
                "q_star": 0.20,
                "q_unit": "nm^-1",
                "branch_id": 1,
                "valid": True,
            },
            {
                "angle_deg": 330.0,
                "q_star": 0.20,
                "q_unit": "nm^-1",
                "branch_id": 1,
                "valid": True,
            },
        ],
    }
    return build_lamellar_scene(
        source,
        {
            "mode": mode,
            "layer_count": layer_count,
            "stack_count": stack_count,
            "reference_period": 30.0,
            "seed": 12,
        },
    )


def _edge_multiplicity(mesh: PublicationMesh) -> Counter:
    triangles = mesh.positions[mesh.indices]
    edges = []
    for triangle in triangles:
        for first, second in (
            (triangle[0], triangle[1]),
            (triangle[1], triangle[2]),
            (triangle[2], triangle[0]),
        ):
            a = tuple(np.round(first, 6))
            b = tuple(np.round(second, 6))
            edges.append(tuple(sorted((a, b))))
    return Counter(edges)


def test_publication_mesh_preserves_source_envelope_and_has_closed_winding() -> None:
    scene = _scene(layer_count=1, stack_count=2)
    source_arrays = (
        scene.centers.copy(),
        scene.sizes.copy(),
        scene.orientations.copy(),
        scene.vertices.copy(),
    )
    mesh = build_publication_mesh(scene)
    assert mesh.positions.dtype == np.float32
    assert mesh.normals.dtype == np.float32
    assert mesh.indices.dtype == np.uint32
    assert mesh.colors.dtype == np.float32
    np.testing.assert_allclose(mesh.bounds, scene.bounds)
    np.testing.assert_array_equal(scene.centers, source_arrays[0])
    np.testing.assert_array_equal(scene.sizes, source_arrays[1])
    np.testing.assert_array_equal(scene.orientations, source_arrays[2])
    np.testing.assert_array_equal(scene.vertices, source_arrays[3])
    assert np.allclose(np.linalg.norm(mesh.normals, axis=1), 1.0, atol=2e-6)
    triangles = mesh.positions[mesh.indices]
    winding = np.einsum(
        "ij,ij->i",
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        mesh.normals[mesh.indices[:, 0]],
    )
    assert np.all(winding > 0.0)
    assert set(_edge_multiplicity(mesh).values()) == {2}
    assert mesh.metadata["source_geometry_status"] == "validated"
    assert mesh.metadata["style_visual_only"] is True


def test_zero_bevel_is_supported_and_mesh_hash_is_deterministic() -> None:
    scene = _scene(layer_count=1, stack_count=1)
    style = PublicationStyle(bevel_fraction=0.0)
    first = build_publication_mesh(scene, style)
    second = build_publication_mesh(scene, style)
    assert len(first.indices) == 12 * len(scene.centers)
    assert first.geometry_hash == second.geometry_hash
    np.testing.assert_array_equal(first.positions, second.positions)
    assert all(value == 0.0 for value in first.metadata["bevel_radii"])


def test_source_consistency_mismatch_is_rejected() -> None:
    scene = _scene(layer_count=1, stack_count=1)
    vertices = scene.vertices.copy()
    vertices[0, 0, 0] += 0.1
    broken = SimpleNamespace(
        centers=scene.centers,
        sizes=scene.sizes,
        orientations=scene.orientations,
        vertices=vertices,
        stack_ids=scene.stack_ids,
        branch_ids=scene.branch_ids,
        bounds=scene.bounds,
    )
    with pytest.raises(ValueError, match="disagree"):
        build_publication_mesh(broken)


def test_camera_basis_and_projection_are_qt_y_up_and_unstretched() -> None:
    right, up, direction = camera_basis({"yaw": 0.0, "pitch": 0.0})
    np.testing.assert_allclose(right, (1.0, 0.0, 0.0))
    np.testing.assert_allclose(up, (0.0, 1.0, 0.0))
    np.testing.assert_allclose(direction, (0.0, 0.0, -1.0))
    points = np.asarray(
        ((-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (1.0, 1.0, 0.0), (-1.0, 1.0, 0.0))
    )
    projected = project_points(
        points,
        {"yaw": 0.0, "pitch": 0.0, "ortho_height": 35.0, "target": (0.0, 0.0, 0.0)},
        ((-2.0, -2.0, -1.0), (2.0, 2.0, 1.0)),
        800,
        600,
    )
    pixel_span = np.ptp(projected, axis=0) * (800.0, 600.0)
    assert pixel_span[0] == pytest.approx(pixel_span[1])
    assert projected[0, 1] > projected[2, 1]


def test_fit_camera_targets_balanced_occupancy_and_branch_gap() -> None:
    scene = _scene(layer_count=1, stack_count=1)
    camera = fit_publication_camera(scene, width=800, height=600)
    projected = project_points(
        scene.vertices.reshape(-1, 3), camera, scene.bounds, 800, 600
    )
    occupancy = float(np.max(np.ptp(projected, axis=0)))
    assert 0.60 <= occupancy <= 0.75
    branch_centers = []
    for branch in sorted(set(int(value) for value in scene.branch_ids)):
        centers = scene.centers[scene.branch_ids == branch]
        branch_centers.append(
            np.mean(project_points(centers, camera, scene.bounds, 800, 600), axis=0)
        )
    assert len(branch_centers) == 2
    assert np.linalg.norm(branch_centers[0] - branch_centers[1]) > 0.05


def test_fit_camera_is_resolution_invariant_for_same_aspect_ratio() -> None:
    scene = _scene(layer_count=1, stack_count=1)
    first = fit_publication_camera(scene, width=800, height=600)
    second = fit_publication_camera(scene, width=1600, height=1200)
    assert first["yaw"] == second["yaw"]
    assert first["pitch"] == second["pitch"]
    assert first["ortho_height"] == pytest.approx(second["ortho_height"], abs=1e-12)
    assert first["target"] == second["target"]


def test_real110_prefers_balanced_shallow_view_when_fixture_is_mounted() -> None:
    path = Path(r"G:\LamellarSAXS2D_schematic_20260906\real110\110.json")
    if not path.is_file():
        pytest.skip("real110 comparison fixture is not mounted")
    scene = build_lamellar_scene(
        json.loads(path.read_text(encoding="utf-8")),
        {"layer_count": 8, "stack_count": 12, "reference_period": 30.0, "seed": 12},
    )
    camera = fit_publication_camera(scene, width=800, height=600)
    assert (camera["yaw"], camera["pitch"]) == (0.0, -18.0)


def test_publication_mesh_cancellation_before_validation() -> None:
    scene = _scene(layer_count=1, stack_count=1)
    event = type("Cancelled", (), {"is_set": lambda self: True})()
    with pytest.raises(RuntimeError, match="publication mesh cancelled"):
        build_publication_mesh(scene, cancel_event=event)


def test_publication_mesh_cancellation_between_plates() -> None:
    scene = _scene(layer_count=1, stack_count=8)
    calls = 0

    def cancel_after_one_plate() -> bool:
        nonlocal calls
        calls += 1
        return calls >= 3

    with pytest.raises(RuntimeError, match="publication mesh cancelled"):
        build_publication_mesh(scene, cancel_event=cancel_after_one_plate)
    assert calls >= 3


def test_uncancelled_callback_keeps_geometry_and_hash_identical() -> None:
    scene = _scene(layer_count=1, stack_count=2)
    expected = build_publication_mesh(scene)
    checks = 0

    def never_cancel() -> bool:
        nonlocal checks
        checks += 1
        return False

    actual = build_publication_mesh(scene, cancel_event=never_cancel)
    assert checks == len(scene.centers) + 1
    assert actual.geometry_hash == expected.geometry_hash
    np.testing.assert_array_equal(actual.positions, expected.positions)
    np.testing.assert_array_equal(actual.indices, expected.indices)
    np.testing.assert_array_equal(actual.colors, expected.colors)
