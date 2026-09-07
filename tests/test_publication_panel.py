from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("PySide6")
from PySide6 import QtCore, QtWidgets  # noqa: E402

from butterfly_saxs.lamellar import build_lamellar_scene
from butterfly_saxs.publication_models import PublicationRenderResult
from butterfly_saxs.ui import lamellar_page as page_module
from butterfly_saxs.ui.publication_panel import PublicationDialog


def test_replacing_artboard_does_not_accumulate_canvas_callbacks(dialog):
    from matplotlib.figure import Figure

    counts = []
    for _ in range(12):
        dialog._replace_figure(Figure())
        counts.append(tuple(len(dialog.canvas.callbacks.callbacks.get(name, {})) for name in
                            ("button_press_event", "motion_notify_event", "button_release_event")))
    assert len(set(counts)) == 1


def test_artboard_resize_preserves_mm_and_matches_mouse_pixel_coordinates(dialog):
    from matplotlib.figure import Figure
    from PySide6 import QtGui

    size = (183. / 25.4, 95. / 25.4)
    dialog._replace_figure(Figure(figsize=size, dpi=140))
    for width in (400, 760):
        height = round(width * 95 / 183)
        dialog.canvas.resize(width, height)
        dialog.canvas.resizeEvent(QtGui.QResizeEvent(dialog.canvas.size(), dialog.canvas.size()))
        np.testing.assert_allclose(dialog._figure.get_size_inches(), size)
        assert dialog._figure.bbox.width == pytest.approx(width * dialog.canvas.device_pixel_ratio)


def test_native_cancel_keeps_cancel_status(dialog, monkeypatch, tmp_path):
    import butterfly_saxs.ui.lamellar_3d as module

    def cancel_render(*args, **kwargs):
        dialog.cancel()
        raise RuntimeError("publication mesh cancelled")

    monkeypatch.setattr(module, "render_publication_scene", cancel_render)
    dialog.export_to(tmp_path / "never-published")
    assert dialog.status.text() == "已取消"
    assert not (tmp_path / "never-published").exists()


def _source() -> dict:
    return {
        "source_identity": {"source": "panel-frame.cbf", "frame": 7, "id": "panel-7"},
        "q_unit": "nm^-1",
        "draw_axis_deg": 90.0,
        "lobe_radial_peaks": [
            {
                "angle_deg": 30.0,
                "q_star": 0.2,
                "q_unit": "nm^-1",
                "branch_id": 0,
                "valid": True,
            },
            {
                "angle_deg": 210.0,
                "q_star": 0.2,
                "q_unit": "nm^-1",
                "branch_id": 0,
                "valid": True,
            },
            {
                "angle_deg": 150.0,
                "q_star": 0.1,
                "q_unit": "nm^-1",
                "branch_id": 1,
                "valid": True,
            },
            {
                "angle_deg": 330.0,
                "q_star": 0.1,
                "q_unit": "nm^-1",
                "branch_id": 1,
                "valid": True,
            },
        ],
    }


def _scene(*, layer_count: int = 2, stack_count: int = 2):
    return build_lamellar_scene(
        _source(), {"layer_count": layer_count, "stack_count": stack_count}
    )


def _context(scene):
    return {
        "scene": scene,
        "observed": np.ones((8, 8), dtype=float),
        "qx": np.zeros((8, 8), dtype=float),
        "qy": np.zeros((8, 8), dtype=float),
        "camera": {
            "yaw": 12.0,
            "pitch": -10.0,
            "distance": 40.0,
            "ortho_height": 35.0,
            "target": (0.0, 0.0, 0.0),
        },
    }


class _Stub3D(QtWidgets.QWidget):
    cameraChanged = QtCore.Signal(object)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._camera = {
            "yaw": 12.0,
            "pitch": -10.0,
            "distance": 40.0,
            "ortho_height": 35.0,
            "target": (10.0, 20.0, 30.0),
        }

    def set_scene(self, scene, **kwargs):
        self.scene = scene

    def clear(self, message=""):
        self.scene = None

    def camera_state(self):
        return dict(self._camera)

    def publication_camera_state(self):
        return dict(self._camera)

    def set_camera_state(self, value):
        self._camera.update(value)

    def set_publication_style(self, style):
        self.style = style

    def set_language(self, language):
        self.language = language

    def set_selected_branch(self, branch):
        self.selected_branch = branch

    def set_standard_view(self, name):
        self.view = name

    def reset_camera(self):
        self._camera = {
            "yaw": 12.0,
            "pitch": -10.0,
            "distance": 40.0,
            "ortho_height": 35.0,
            "target": (10.0, 20.0, 30.0),
        }


@pytest.fixture
def dialog(qtbot):
    context = _context(_scene())
    widget = PublicationDialog(document=None, context_provider=lambda: context)
    qtbot.addWidget(widget)
    yield widget
    widget.cancel()
    widget.close()


def test_document_settings_camera_and_annotations_roundtrip(dialog):
    document = {
        "version": 1,
        "style": {
            "palette": "grayscale",
            "bevel_fraction": 0.0,
            "roughness": 0.8,
            "ambient_occlusion": False,
        },
        "figure": {
            "template": "evidence",
            "width_mm": 123.0,
            "height_mm": 97.0,
            "dpi": 450,
            "language": "en",
            "background": "transparent",
            "quality": "publication",
            "annotation_positions": {"a": [0.22, 0.33], "b": [0.71, 0.64]},
        },
        "camera": {
            "yaw": 24.0,
            "pitch": -12.0,
            "distance": 44.0,
            "ortho_height": 31.0,
            "target": (1.0, 2.0, 3.0),
        },
        "camera_auto": False,
    }
    dialog.restore_document(document)
    restored = dialog.document()
    assert restored["style"] == document["style"]
    assert restored["figure"]["width_mm"] == pytest.approx(123.0)
    assert restored["figure"]["height_mm"] == pytest.approx(97.0)
    assert (
        restored["figure"]["annotation_positions"]
        == document["figure"]["annotation_positions"]
    )
    assert restored["camera"] == document["camera"]
    assert restored["camera_auto"] is False


def test_width_preset_custom_width_and_artboard_aspect(dialog, qtbot):
    dialog.width_preset.setCurrentIndex(dialog.width_preset.findData(89.0))
    qtbot.wait(20)
    assert dialog.width_spin.value() == pytest.approx(89.0)
    assert dialog.height_spin.value() == pytest.approx(70.0)
    dialog.width_preset.setCurrentIndex(dialog.width_preset.findData(0.0))
    dialog.width_spin.setValue(123.0)
    dialog.height_spin.setValue(97.0)
    dialog._controls_changed()
    assert dialog.document()["figure"]["width_mm"] == pytest.approx(123.0)
    assert dialog.document()["figure"]["height_mm"] == pytest.approx(97.0)
    dialog.restore_document({})
    dialog.artboard.resize(980, 680)
    qtbot.wait(20)
    geometry = dialog.canvas.geometry()
    assert geometry.width() / geometry.height() == pytest.approx(
        183.0 / 95.0, rel=0.02
    )


def test_refresh_preview_uses_explicit_mocked_native_result(dialog, monkeypatch):
    calls = []

    def fake_native(
        scene, *, style, width, height, camera, quality, transparent, cancel_event
    ):
        calls.append(
            {
                "width": width,
                "height": height,
                "camera": camera,
                "quality": quality,
                "transparent": transparent,
            }
        )
        rgba = np.zeros((120, 160, 4), dtype=np.uint8)
        rgba[..., 3] = 255
        return PublicationRenderResult(
            rgba=rgba,
            camera={
                "yaw": 3.0,
                "pitch": -9.0,
                "distance": 41.0,
                "ortho_height": 34.0,
                "target": (1.0, 2.0, 3.0),
            },
            bounds=scene.bounds,
            provenance={"test_fixture": True, "renderer_backend": "mock-native"},
        )

    def fake_figure(*args, **kwargs):
        from matplotlib.figure import Figure

        figure = Figure(figsize=(4, 3), facecolor="white")
        figure.publication_annotation_artists = {}
        return figure

    import butterfly_saxs.publication as publication_module
    import butterfly_saxs.ui.lamellar_3d as lamellar_3d_module

    monkeypatch.setattr(lamellar_3d_module, "render_publication_scene", fake_native)
    monkeypatch.setattr(publication_module, "render_publication_figure", fake_figure)
    dialog.refresh_preview()
    assert calls and calls[0]["quality"] == "preview"
    assert calls[0]["transparent"] is True
    assert dialog._rendered is not None
    assert dialog._camera["ortho_height"] == pytest.approx(34.0)
    assert dialog.export_button.isEnabled()
    assert dialog._rendered.provenance["test_fixture"] is True


def test_invalidation_cancellation_and_pending_preview_state(dialog, qtbot):
    dialog._rendered = SimpleNamespace(camera={"yaw": 1.0})
    dialog._context = _context(_scene())
    dialog._set_busy(False)
    assert dialog.export_button.isEnabled()
    dialog._rendering = True
    dialog.request_preview()
    assert dialog._pending is True
    dialog.refresh_preview()
    assert dialog._pending is True
    dialog.cancel()
    assert dialog._pending is False
    assert dialog.jobs_running() is True
    dialog._rendering = False
    dialog.invalidate_source()
    assert dialog._context is None
    assert dialog._rendered is None
    assert not dialog.export_button.isEnabled()
    assert not dialog.jobs_running()
    dialog.close()
    qtbot.wait(20)


def test_page_publication_context_uses_actual_publication_camera_units(
    qtbot, monkeypatch
):
    monkeypatch.setattr(page_module, "Lamellar3DView", _Stub3D)
    page = page_module.LamellarPage(language="en")
    qtbot.addWidget(page)
    scene = _scene(layer_count=1, stack_count=1)
    page.current_scene = scene
    page.scenes = [scene]
    page._images = (np.ones((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)))
    page._image_error = None
    page._active.clear()
    context = page.publication_context()
    assert context is not None
    assert context["camera"]["target"] == (10.0, 20.0, 30.0)
    assert context["camera"]["ortho_height"] == pytest.approx(35.0)
    page.shutdown()
