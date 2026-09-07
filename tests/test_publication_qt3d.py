from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from butterfly_saxs.publication_geometry import build_publication_mesh
from butterfly_saxs.publication_models import PublicationStyle


def test_gui_cancel_event_is_serviced_during_mesh_generation(qtbot):
    import threading
    from PySide6 import QtCore
    from butterfly_saxs.lamellar import build_lamellar_scene
    from butterfly_saxs.ui.lamellar_3d import _ResponsiveCancellation

    scene = build_lamellar_scene({}, {"period_source": "manual", "mode": "multi", "stack_count": 32, "layer_count": 32})
    event = threading.Event()
    QtCore.QTimer.singleShot(20, event.set)
    with pytest.raises(RuntimeError, match="cancelled"):
        build_publication_mesh(scene, cancel_event=_ResponsiveCancellation(event))
    assert event.is_set()


_SIGNS = np.asarray(
    (
        (-1.0, -1.0, -1.0),
        (1.0, -1.0, -1.0),
        (1.0, 1.0, -1.0),
        (-1.0, 1.0, -1.0),
        (-1.0, -1.0, 1.0),
        (1.0, -1.0, 1.0),
        (1.0, 1.0, 1.0),
        (-1.0, 1.0, 1.0),
    ),
    dtype=float,
)


def _box(center: tuple[float, float, float], size: tuple[float, float, float]) -> np.ndarray:
    return np.asarray(center, dtype=float) + _SIGNS * np.asarray(size, dtype=float) / 2.0


def _scene() -> SimpleNamespace:
    centers = np.asarray([[-1.0, 0.0, 0.0], [1.0, 0.1, 0.12]], dtype=float)
    sizes = np.asarray([[1.2, 0.7, 0.22], [1.2, 0.7, 0.22]], dtype=float)
    orientations = np.asarray([np.eye(3), np.eye(3)], dtype=float)
    vertices = np.asarray([_box(tuple(center), tuple(size)) for center, size in zip(centers, sizes)])
    return SimpleNamespace(
        centers=centers,
        sizes=sizes,
        orientations=orientations,
        vertices=vertices,
        stack_ids=np.asarray([0, 1]),
        branch_ids=np.asarray([0, 1]),
        bounds=np.asarray([[-2.0, -1.0, -0.5], [2.0, 1.0, 0.5]]),
        metadata={"available": True},
        length_unit="relative",
    )


def test_publication_mesh_is_finite_opaque_and_envelope_preserving() -> None:
    scene = _scene()
    mesh = build_publication_mesh(scene, PublicationStyle(), quality="publication")

    assert mesh.positions.dtype == np.float32
    assert mesh.normals.shape == mesh.positions.shape
    assert mesh.indices.ndim == 2 and mesh.indices.shape[1] == 3
    assert mesh.branch_ids.shape == (len(mesh.indices),)
    assert mesh.colors.shape == (len(mesh.indices), 4)
    assert np.all(np.isfinite(mesh.positions))
    assert np.all(np.isfinite(mesh.normals))
    assert np.allclose(np.linalg.norm(mesh.normals, axis=1), 1.0, atol=1e-5)
    assert np.all(mesh.colors[:, 3] == 1.0)
    assert np.min(mesh.positions, axis=0).tolist() == pytest.approx(scene.vertices.min(axis=(0, 1)).tolist())
    assert np.max(mesh.positions, axis=0).tolist() == pytest.approx(scene.vertices.max(axis=(0, 1)).tolist())
    assert mesh.metadata["source_geometry_status"] == "validated"
    assert mesh.geometry_hash


def test_qquick_geometry_groups_are_merged_per_branch() -> None:
    pytest.importorskip("PySide6")
    from butterfly_saxs.ui.publication_qt_geometry import build_publication_geometry_groups

    mesh = build_publication_mesh(_scene(), PublicationStyle(), quality="publication")
    groups, normalized_bounds = build_publication_geometry_groups(mesh, bounds=mesh.bounds)
    assert {group.branch_id for group in groups} == {0, 1}
    assert normalized_bounds.shape == (2, 3)
    assert all(group.triangle_count > 0 for group in groups)
    assert all(group.geometry.vertexData().size() > 0 for group in groups)
    assert all(group.geometry.attributeCount() == 3 for group in groups)


def test_publication_capture_is_controls_free_or_fails_truthfully(qtbot) -> None:
    pytest.importorskip("PySide6")
    from butterfly_saxs.ui.lamellar_3d import Lamellar3DView

    view = Lamellar3DView()
    qtbot.addWidget(view)
    view.resize(520, 360)
    view.set_scene(_scene(), reset_camera=True)
    view.show()
    qtbot.wait(180)
    if not view.available:
        with pytest.raises(RuntimeError):
            view.capture_publication(_scene(), width=320, height=240)
        return

    result = view.capture_publication(_scene(), width=320, height=240, transparent=True)
    assert result.rgba.shape == (240, 320, 4)
    assert result.rgba.dtype == np.uint8
    assert result.provenance["controls_burned_in"] is False
    assert result.provenance["resolution"] == [320, 240]
    assert result.provenance["geometry_hash"]
    assert np.any(result.rgba[:, :, 3] > 0)


def test_publication_square_plate_has_equal_aspect_landscape_and_portrait(qtbot) -> None:
    pytest.importorskip("PySide6")
    from butterfly_saxs.ui.lamellar_3d import Lamellar3DView

    base = _scene()
    plate = SimpleNamespace(
        centers=np.zeros((1, 3)),
        sizes=np.asarray([[1.0, 1.0, 0.2]]),
        orientations=base.orientations[:1].copy(),
        vertices=np.asarray([_box((0.0, 0.0, 0.0), (1.0, 1.0, 0.2))]),
        stack_ids=np.asarray([0]),
        branch_ids=np.asarray([0]),
        bounds=np.asarray([[-0.6, -0.6, -0.2], [0.6, 0.6, 0.2]]),
        metadata={"available": True},
        length_unit="relative",
    )
    view = Lamellar3DView()
    qtbot.addWidget(view)
    view.resize(520, 360)
    view.set_scene(plate, reset_camera=True)
    view.show()
    qtbot.wait(180)
    if not view.available:
        pytest.skip(view.error_message)

    # Use an explicit front camera so the measured object is the known
    # rectangular source face rather than a family-separation fit choice.
    camera = {"yaw": 0.0, "pitch": -12.0, "distance": 5.0, "ortho_height": 35.0, "target": [0.0, 0.0, 0.0]}
    ratios: list[float] = []
    for width, height in ((1200, 900), (900, 1200)):
        result = view.capture_publication(plate, camera=camera, width=width, height=height, transparent=False)
        background = np.asarray([255, 255, 255], dtype=np.uint8)
        mask = np.any(result.rgba[:, :, :3] != background, axis=2)
        yy, xx = np.where(mask)
        assert len(xx) > 1000
        ratios.append(float((xx.max() - xx.min() + 1) / (yy.max() - yy.min() + 1)))
    assert ratios[0] == pytest.approx(1.0, abs=0.04)
    assert ratios[1] == pytest.approx(1.0, abs=0.04)


def test_publication_camera_getter_tracks_user_camera_in_source_units(qtbot) -> None:
    pytest.importorskip("PySide6")
    from butterfly_saxs.ui.lamellar_3d import Lamellar3DView

    view = Lamellar3DView()
    qtbot.addWidget(view)
    view.resize(520, 360)
    view.set_scene(_scene(), reset_camera=True)
    view.show()
    qtbot.wait(180)
    if not view.available:
        pytest.skip(view.error_message)

    view.set_camera_state({"yaw": 27.0, "pitch": -11.0, "distance": 50.0, "ortho_height": 41.0, "target": (0.5, -0.2, 0.0)})
    camera = view.publication_camera_state()
    assert camera["yaw"] == pytest.approx(27.0)
    assert camera["pitch"] == pytest.approx(-11.0)
    assert camera["target"] == pytest.approx((0.25, -0.1, 0.0), abs=1e-6)
    view.set_standard_view("front")
    assert view.publication_camera_state()["yaw"] == pytest.approx(0.0)


def test_fixed_sequence_bounds_round_trip_through_standalone_camera() -> None:
    pytest.importorskip("PySide6")
    from butterfly_saxs.ui.lamellar_3d import render_publication_scene

    scene = _scene()
    scene.bounds = np.asarray([[-10.0, -8.0, -2.0], [10.0, 8.0, 2.0]])
    try:
        first = render_publication_scene(scene, width=320, height=240, transparent=True)
        second = render_publication_scene(scene, camera=first.camera, width=320, height=240, transparent=True)
    except RuntimeError as exc:
        if "native RHI" in str(exc) or "Software" in str(exc):
            pytest.skip(str(exc))
        raise

    assert np.allclose(first.bounds, scene.bounds)
    np.testing.assert_allclose(first.camera["viewport_bounds"], scene.bounds)
    np.testing.assert_allclose(second.camera["viewport_bounds"], scene.bounds)
    first_mask = first.rgba[:, :, 3] > 0
    second_mask = second.rgba[:, :, 3] > 0
    assert np.array_equal(first_mask, second_mask)
