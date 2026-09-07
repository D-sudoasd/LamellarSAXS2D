from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6 import QtCore

from butterfly_saxs.ui.lamellar_3d import Lamellar3DView, _scene_payload


def test_native_qml_camera_signal_updates_saved_state(qtbot):
    from PySide6.QtQml import QJSEngine

    viewer = Lamellar3DView()
    qtbot.addWidget(viewer)
    engine = QJSEngine()
    state = engine.evaluate("({yaw: -67.25, pitch: -37.25, ortho_height: 40, target: {x: 1, y: 2, z: 3}})")
    viewer._on_qml_camera_changed(state)
    assert viewer.camera_state()["yaw"] == -67.25
    assert viewer.camera_state()["pitch"] == -37.25
    assert viewer.camera_state()["target"] == (1., 2., 3.)
    viewer.set_scene(_scene())
    baseline = viewer._fit_camera()["ortho_height"]
    viewer.set_camera_state({"zoom": 2.})
    assert viewer.camera_state()["ortho_height"] == baseline * 2.


def _scene(offset: float = 0.0) -> SimpleNamespace:
    orientations = np.asarray(
        [
            np.eye(3),
            [
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
        ],
        dtype=float,
    )
    return SimpleNamespace(
        centers=np.asarray([[-1.0 + offset, 0.0, 0.0], [1.0 + offset, 0.0, 0.2]]),
        sizes=np.asarray([[1.2, 0.55, 0.18], [0.9, 0.65, 0.24]]),
        orientations=orientations,
        vertices=np.empty((2, 8, 3), dtype=float),
        stack_ids=np.asarray([0, 1]),
        branch_ids=np.asarray([0, 1]),
        colors=np.asarray([[0.20, 0.45, 0.85, 1.0], [0.95, 0.42, 0.18, 0.9]]),
        bounds=np.asarray([[-2.5, -2.0, -1.0], [2.5, 2.0, 1.0]]),
        length_unit="relative",
        metadata={"available": True},
    )


def test_scene_payload_preserves_contract_geometry_and_branch_metadata() -> None:
    payload = _scene_payload(_scene())

    assert len(payload.items) == 2
    assert payload.items[0]["stack"] == 0
    assert payload.items[1]["branch"] == 1
    assert payload.items[0]["sx"] == pytest.approx(1.92)
    assert payload.items[0]["sy"] == pytest.approx(0.88)
    # The second orientation is a +90 degree in-plane rotation, represented
    # by a unit quaternion rather than an inferred Euler approximation.
    assert payload.items[1]["qw"] == pytest.approx(2**-0.5)
    assert payload.items[1]["qz"] == pytest.approx(2**-0.5)
    assert np.allclose(payload.bounds, _scene().bounds)


def test_view_preserves_camera_between_frames_and_maps_model_scale(qtbot) -> None:
    view = Lamellar3DView()
    qtbot.addWidget(view)
    view.resize(620, 420)
    view.set_scene(_scene(), reset_camera=True)
    view.set_camera_state(
        {
            "yaw": 14.0,
            "pitch": -17.0,
            "distance": 10.0,
            "ortho_height": 6.0,
            "target": (0.2, -0.1, 0.0),
        }
    )
    before = view.camera_state()
    assert view._quick is None
    view.set_scene(_scene(offset=0.3), reset_camera=False)
    after = view.camera_state()
    assert after == before
    assert view.bounds is not None

    view.show()
    qtbot.wait(120)
    if view.available:
        models = [
            item
            for item in view._qml_root.findChildren(QtCore.QObject)
            if "QQuick3DModel" in item.metaObject().className()
        ]
        assert models
        scale = models[0].property("scale")
        assert scale.x() == pytest.approx(view._payload.items[0]["sx"] * 0.01)
        assert scale.y() == pytest.approx(view._payload.items[0]["sy"] * 0.01)
    else:
        assert view.error_message


def test_capture_has_requested_pixels_or_truthful_renderer_error(qtbot) -> None:
    view = Lamellar3DView()
    qtbot.addWidget(view)
    view.resize(500, 360)
    view.set_scene(_scene(), reset_camera=True)
    view.show()
    qtbot.wait(180)

    if view.available:
        image = view.capture_image(320, 240)
        assert image is not None
        assert image.size() == QtCore.QSize(320, 240)
        assert not image.isNull()
        assert view._quick.status() == view._quick.Status.Ready
    else:
        assert view.error_message
        with pytest.raises(RuntimeError, match="3-D image was rendered|renderer"):
            view.capture_image(320, 240)


def test_existing_qml_renderer_reappears_after_clear_message(qtbot) -> None:
    view = Lamellar3DView()
    qtbot.addWidget(view)
    view.resize(520, 360)
    view.show()
    view.set_scene(_scene(), reset_camera=True)
    qtbot.wait(180)
    if not view.available:
        pytest.skip(view.error_message)

    assert view._stack.currentWidget() is view._quick
    view.clear("updating")
    assert view._stack.currentWidget() is view._message
    view.set_scene(_scene(offset=0.2), reset_camera=False)
    assert view._stack.currentWidget() is view._quick


def test_square_plate_keeps_equal_pixel_aspect_on_native_rhi(qtbot) -> None:
    """The orthographic camera must not stretch a square with viewport aspect."""

    plate = SimpleNamespace(
        centers=np.asarray([[0.0, 0.0, 0.0]]),
        sizes=np.asarray([[1.0, 1.0, 0.12]]),
        orientations=np.asarray([np.eye(3)]),
        vertices=np.empty((1, 8, 3), dtype=float),
        stack_ids=np.asarray([0]),
        branch_ids=np.asarray([0]),
        colors=np.asarray([[0.95, 0.42, 0.18, 1.0]]),
        bounds=np.asarray([[-1.5, -1.5, -1.0], [1.5, 1.5, 1.0]]),
        length_unit="relative",
        metadata={"available": True},
    )
    view = Lamellar3DView()
    qtbot.addWidget(view)
    view.resize(520, 360)
    view.show()
    view.set_scene(plate)
    qtbot.wait(180)
    if not view.available:
        pytest.skip(view.error_message)
    view.set_standard_view("front")

    ratios: list[float] = []
    for width, height in ((1200, 900), (900, 1200)):
        image = view.capture_image(width, height)
        rgba = image.convertToFormat(image.Format.Format_RGBA8888)
        pixels = np.frombuffer(bytes(rgba.constBits()), dtype=np.uint8)
        rows = pixels.reshape(rgba.height(), rgba.bytesPerLine())
        rgb = rows[:, : rgba.width() * 4].reshape(rgba.height(), rgba.width(), 4)[:, :, :3]
        # The test slab is orange; axis-helper red/green/blue pixels are
        # excluded by requiring a strong red-over-green and green-over-blue
        # separation.
        yy_grid, xx_grid = np.indices(rgb.shape[:2])
        orange = (
            (rgb[:, :, 0] > 120)
            & (rgb[:, :, 0] > rgb[:, :, 1] * 1.25)
            & (rgb[:, :, 1] > rgb[:, :, 2] * 1.4)
            & (xx_grid > 100)
            & (xx_grid < width - 100)
            & (yy_grid > 100)
            & (yy_grid < height - 100)
        )
        yy, xx = np.where(orange)
        assert len(xx) > 1000
        ratios.append(float((xx.max() - xx.min() + 1) / (yy.max() - yy.min() + 1)))
    assert ratios[0] == pytest.approx(1.0, abs=0.03)
    assert ratios[1] == pytest.approx(1.0, abs=0.03)
