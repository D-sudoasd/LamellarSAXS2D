from __future__ import annotations

from copy import deepcopy
import threading

import numpy as np
import pytest

pytest.importorskip("PySide6")
from PySide6 import QtCore, QtGui, QtWidgets  # noqa: E402

from butterfly_saxs.ui import lamellar_page as page_module  # noqa: E402


class Stub3D(QtWidgets.QWidget):
    """Page tests isolate the UI controller; actual RHI has separate tests."""

    cameraChanged = QtCore.Signal(object)
    available = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.camera = {"yaw": 25., "pitch": 15., "zoom": 1.}

    def set_scene(self, scene, **kwargs):
        self.scene = scene

    def clear(self, message=""):
        self.scene = None

    def camera_state(self):
        return dict(self.camera)

    def set_camera_state(self, value):
        self.camera.update(value)

    def set_standard_view(self, name):
        self.camera["view"] = name

    def reset_camera(self):
        self.camera = {"yaw": 25., "pitch": 15., "zoom": 1.}

    def set_language(self, language):
        pass

    def set_selected_branch(self, branch):
        self.selected_branch = branch

    def capture_image(self, width=2400, height=1800):
        image = QtGui.QImage(width, height, QtGui.QImage.Format.Format_RGBA8888)
        image.fill(QtGui.QColor("#90bddd"))
        return image


def source(q=.2, frame=0, *, status="ok", image=True):
    data = {
        "source_identity": {"source": "in-memory", "frame": frame, "dataset": "detector", "id": f"frame-{frame}"},
        "q_unit": "nm^-1", "draw_axis_deg": 90., "status": status,
        "lobe_radial_peaks": [
            {"q_star": q, "angle": np.pi / 3, "valid": True, "q_unit": "nm^-1", "branch_id": 0},
            {"q_star": q, "angle": 4 * np.pi / 3, "valid": True, "q_unit": "nm^-1", "branch_id": 0},
        ],
    }
    if image:
        x, y = np.meshgrid(np.linspace(-.5, .5, 16), np.linspace(-.5, .5, 16))
        data.update(observed=np.ones((16, 16)) * (frame + 1), qx=x, qy=y)
    return data


@pytest.fixture
def page(qtbot, monkeypatch):
    monkeypatch.setattr(page_module, "Lamellar3DView", Stub3D)
    widget = page_module.LamellarPage(language="en")
    qtbot.addWidget(widget)
    yield widget
    widget.shutdown()
    qtbot.waitUntil(lambda: not widget.jobs_running(), timeout=15000)


def ready(qtbot, page):
    qtbot.waitUntil(lambda: page.current_scene is not None and not page.jobs_running(), timeout=15000)


def test_settings_are_independent_and_undo_restores_geometry(page, qtbot):
    original = source()
    before = deepcopy(original["lobe_radial_peaks"])
    page.set_source(original, context_signature="a")
    ready(qtbot, page)
    vertices = page.current_scene.vertices.copy()
    page.update_settings({**page.settings, "thickness_ratio": .5})
    qtbot.waitUntil(lambda: bool(page.scenes) and page.settings["thickness_ratio"] == .5 and not page.jobs_running(), timeout=15000)
    assert not np.array_equal(vertices, page.current_scene.vertices)
    assert original["lobe_radial_peaks"] == before
    page.undo()
    ready(qtbot, page)
    np.testing.assert_allclose(page.current_scene.vertices, vertices)
    page.redo()
    ready(qtbot, page)
    assert page.current_scene.metadata["settings"]["thickness_ratio"] == .5


def test_invalidation_cannot_be_bypassed_by_cosmetic_change(page, qtbot):
    page.set_source(source())
    ready(qtbot, page)
    page.invalidate()
    page.update_settings({**page.settings, "width_ratio": 5.})
    qtbot.wait(250)
    assert page.current_scene is None
    assert not page.export_button.isEnabled()
    with pytest.raises(ValueError, match="current scene"):
        page.export_to("never-written")
    page.update_settings({**page.settings, "period_source": "manual"})
    ready(qtbot, page)
    assert page.current_scene.metadata["status"] == "manual"


def test_palette_undo_preserves_geometry_and_source(page, qtbot):
    data = source()
    page.set_source(data)
    ready(qtbot, page)
    geometry = page.current_scene.vertices.copy()
    colors = page.current_scene.colors.copy()
    page.set_palette("grayscale")
    np.testing.assert_array_equal(page.current_scene.vertices, geometry)
    assert not np.array_equal(colors, page.current_scene.colors)
    assert page.document()["presentation"]["palette"] == "grayscale"
    page.undo()
    np.testing.assert_array_equal(page.current_scene.colors, colors)
    assert page.sources[0] is data


def test_failed_frame_is_blank_and_does_not_reuse_image(page, qtbot):
    bad = source(.2, 1, status="failed", image=False)
    bad["lobe_radial_peaks"] = []
    page.set_series([source(.2, 0), bad, source(.4, 2)])
    ready(qtbot, page)
    reference = page._bounds.copy()
    camera = page.view3d.camera_state()
    assert page.q_view.observed is not None
    page.select_frame(1)
    ready(qtbot, page)
    assert not page.current_scene.metadata["available"]
    assert not len(page.current_scene.vertices)
    assert page.q_view.observed is None
    assert "frame 1" in page.frame_text.text()
    page.select_frame(2)
    ready(qtbot, page)
    assert page.current_scene.metadata["available"]
    np.testing.assert_allclose(reference, page._bounds)
    assert page.view3d.camera_state() == camera
    assert np.all(page.q_view.observed == 3)


def test_document_roundtrip_and_context_mismatch(page, qtbot):
    page.set_source(source(), context_signature="original input hashes")
    ready(qtbot, page)
    page.view3d.set_camera_state({"yaw": 70.})
    saved = page.document()
    vertices = page.current_scene.vertices.copy()
    assert "observed" not in saved["sources"][0]
    page.restore_document(saved, context_signature="original input hashes")
    ready(qtbot, page)
    np.testing.assert_array_equal(vertices, page.current_scene.vertices)
    assert page.view3d.camera_state()["yaw"] == 70.
    page.restore_document(saved, context_signature="changed input hashes")
    qtbot.wait(250)
    assert page.current_scene is None
    assert not page.export_button.isEnabled()


def test_unit_display_spelling_is_not_a_new_physical_context():
    import json

    first = {"parameters": {"a": {"value": .3, "unit": "nm⁻¹"}}, "input": {"frame": 7}}
    second = deepcopy(first)
    second["parameters"]["a"]["unit"] = "nm^-1"
    assert page_module.LamellarPage._context_hash(json.dumps(first)) == page_module.LamellarPage._context_hash(json.dumps(second))
    second["parameters"]["a"]["unit"] = "Å^-1"
    assert page_module.LamellarPage._context_hash(json.dumps(first)) != page_module.LamellarPage._context_hash(json.dumps(second))


def test_old_background_result_cannot_replace_new_frame(page, qtbot, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    original = page_module._build_scenes

    def delayed(sources, values, cancel):
        if sources[0]["source_identity"]["frame"] == 1:
            entered.set()
            release.wait(5)
            # Ignore cooperative cancellation here to exercise the generation guard.
            return original(sources, values, threading.Event())
        return original(sources, values, cancel)

    monkeypatch.setattr(page_module, "_build_scenes", delayed)
    page.set_source(source(.2, 1))
    qtbot.waitUntil(entered.is_set, timeout=5000)
    page.set_source(source(.4, 2))
    qtbot.waitUntil(lambda: page.current_scene is not None, timeout=10000)
    release.set()
    ready(qtbot, page)
    assert page.current_scene.metadata["source_identity"]["frame"] == 2
    assert page.sources[0]["lobe_radial_peaks"][0]["q_star"] == .4


def test_user_cancel_reports_cancelled_not_permanent_updating(page, qtbot, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    original = page_module._build_scenes

    def delayed(sources, values, cancel):
        entered.set()
        release.wait(5)
        return original(sources, values, cancel)

    monkeypatch.setattr(page_module, "_build_scenes", delayed)
    page.set_source(source())
    qtbot.waitUntil(entered.is_set, timeout=5000)
    page.cancel_jobs()
    release.set()
    qtbot.waitUntil(lambda: not page.jobs_running(), timeout=10000)
    assert "cancelled" in page.status.text().lower()


def test_main_window_registers_page_and_carries_current_source(qtbot, monkeypatch):
    monkeypatch.setattr(page_module, "Lamellar3DView", Stub3D)
    from butterfly_saxs.ui.main_window import RefinementMainWindow

    window = RefinementMainWindow(engine={}, auto_preview=False, language="en")
    qtbot.addWidget(window)
    result = source()
    window._source_path = "in-memory"
    window._frame = 7
    window._apply_result(result)
    ready(qtbot, window.lamellar_page)
    assert window.lamellar_page.current_scene.metadata["source_identity"]["frame"] == 7
    assert window.pages.indexOf(window.lamellar_page) >= 0
    assert "lamellar_view" in window.project_to_dict()
    assert window.set_parameter("q_major", .42)
    assert window.lamellar_page.current_scene is None
    window._invalidate_pending_work(clear_fit=True)
    assert window.lamellar_page.current_scene is None
    window.lamellar_page.shutdown()
    qtbot.waitUntil(lambda: not window.lamellar_page.jobs_running(), timeout=15000)
