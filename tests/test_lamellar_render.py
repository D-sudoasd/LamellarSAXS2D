from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from matplotlib.figure import Figure

from butterfly_saxs.lamellar_render import (
    _camera_orientation,
    render_lamellar_2d,
    render_lamellar_3d,
    render_lamellar_combined,
)


def _scene(*, status: str = "schematic", length_unit: str = "nm", message: str = "") -> SimpleNamespace:
    centers = np.asarray([[0.0, 0.0, 0.0], [1.8, 0.15, 0.0]])
    sizes = np.asarray([[1.0, 2.2, 0.25], [1.0, 2.2, 0.25]])
    vertices = []
    signs = [(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1), (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)]
    for center, size in zip(centers, sizes):
        half = size / 2.0
        vertices.append(np.asarray([center + np.asarray(sign) * half for sign in signs]))
    return SimpleNamespace(
        centers=centers,
        sizes=sizes,
        orientations=np.repeat(np.eye(3)[None, :, :], 2, axis=0),
        vertices=np.asarray(vertices),
        stack_ids=np.asarray([0, 1]),
        branch_ids=np.asarray([0, 1]),
        colors=np.asarray([[0.20, 0.45, 0.72, 0.84], [0.92, 0.47, 0.18, 0.84]]),
        bounds=np.asarray([[-1.0, -1.5, -1.0], [2.5, 1.5, 1.0]]),
        length_unit=length_unit,
        metadata={"frame_id": "frame-A"},
        status=status,
        message=message,
        source_identity="source-A",
        assumptions=[],
        settings={},
        parameter_sources=[],
        populations=[],
        scientific_boundary="schematic only",
        draw_axis_deg=90.0,
        reference_period=None,
    )


def test_2d_uses_actual_vertices_and_physical_unit_labels() -> None:
    scene = _scene()
    figure = render_lamellar_2d(scene)
    axis = figure.axes[0]

    assert len(axis.patches) == len(scene.vertices)
    assert axis.get_aspect() == 1.0
    assert "nm" in axis.get_xlabel()
    assert "+y" in "\n".join(text.get_text() for text in axis.texts)
    assert "参数驱动示意" in "\n".join(text.get_text() for text in figure.texts)


def test_3d_and_combined_views_keep_source_status_visible() -> None:
    scene = _scene(status="candidate", length_unit="relative")
    figure_3d = render_lamellar_3d(scene, camera={"elev": 25, "azim": 35})
    figure_combined = render_lamellar_combined(scene)

    assert len(figure_3d.axes[0].collections) >= len(scene.vertices)
    labels = "\n".join(text.get_text() for text in figure_combined.texts)
    assert "候选参数驱动示意" in labels
    assert "相对尺度" in labels


def test_qt_camera_uses_world_y_as_matplotlib_vertical_axis() -> None:
    first = _camera_orientation({"yaw": 0.0, "pitch": 0.0})
    second = _camera_orientation({"yaw": 90.0, "pitch": 0.0})
    assert first is not None and first[:3] == pytest.approx((0.0, 0.0, 0.0)) and first[3] == "y"
    assert second is not None and second[:3] == pytest.approx((0.0, 90.0, 0.0)) and second[3] == "y"
    elev, azim, _, vertical = _camera_orientation({"yaw": -38.0, "pitch": -26.0})
    assert elev == pytest.approx(26.0)
    assert azim == pytest.approx(-38.0)
    assert vertical == "y"


def test_combined_3d_has_one_banner_and_no_tick_or_normal_clutter() -> None:
    scene = _scene()
    figure = render_lamellar_combined(scene, camera={"yaw": 0.0, "pitch": 0.0})
    three_d = figure.axes[-1]

    assert len(figure.texts) == 1
    assert three_d.get_title() == ""
    assert len(three_d.collections) == len(scene.vertices)
    assert three_d._vertical_axis == 1
    assert len(three_d.get_xticks()) == len(three_d.get_yticks()) == len(three_d.get_zticks()) == 0


def test_qt_front_camera_projects_xy_with_positive_y_up() -> None:
    scene = _scene()
    figure = render_lamellar_3d(scene, camera={"yaw": 0.0, "pitch": 0.0}, decorate=False)
    axis = figure.axes[0]
    figure.canvas.draw()

    np.testing.assert_allclose(axis._view_w, np.asarray([0.0, 0.0, 1.0]), atol=1.0e-12)
    np.testing.assert_allclose(axis._view_u, np.asarray([1.0, 0.0, 0.0]), atol=1.0e-12)
    np.testing.assert_allclose(axis._view_v, np.asarray([0.0, 1.0, 0.0]), atol=1.0e-12)


def test_q_map_axes_keep_source_q_unit() -> None:
    scene = _scene()
    scene.metadata["q_unit"] = "nm^-1"
    values = np.linspace(-0.5, 0.5, 9)
    qx, qy = np.meshgrid(values, values)
    figure = render_lamellar_combined(scene, observed=np.ones_like(qx), qx=qx, qy=qy)

    observed_axis = figure.axes[0]
    assert "nm^-1" in observed_axis.get_xlabel()
    assert "nm^-1" in observed_axis.get_ylabel()


def test_stale_scene_is_an_explicit_blank_reason() -> None:
    scene = _scene(status="stale", message="fit is older than the current frame")
    figure = render_lamellar_2d(scene)
    axis = figure.axes[0]

    assert not axis.patches
    assert "fit is older" in "\n".join(text.get_text() for text in axis.texts)
    assert "结果已过期" in "\n".join(text.get_text() for text in figure.texts)


def test_embedded_2d_mode_omits_standalone_figure_caption() -> None:
    scene = _scene()
    figure = Figure(figsize=(4.0, 3.0))
    axis = figure.add_subplot(111)
    render_lamellar_2d(scene, ax=axis, decorate=False)

    assert figure.texts == []
    assert axis.get_title() == ""
    assert "+y" in "\n".join(text.get_text() for text in axis.texts)


def test_combined_q_map_masks_and_crops_to_display_window_without_mutating_source() -> None:
    scene = _scene()
    scene.metadata["display_q_window"] = {"min": 0.1, "max": 0.5, "unit": "nm^-1"}
    axis_values = np.linspace(-8.0, 8.0, 81)
    qx, qy = np.meshgrid(axis_values, axis_values)
    observed = np.ones_like(qx)
    observed[40, 40] = 9.0
    original = observed.copy()

    figure = render_lamellar_combined(scene, observed=observed, qx=qx, qy=qy)
    image_axis = figure.axes[0]
    shown = image_axis.images[0].get_array()

    np.testing.assert_array_equal(observed, original)
    assert np.ma.isMaskedArray(shown)
    assert np.count_nonzero(np.ma.getmaskarray(shown)) > 0
    assert image_axis.get_xlim() == pytest.approx((-0.5, 0.5))
    assert image_axis.get_ylim() == pytest.approx((-0.5, 0.5))
    assert "0.1" in image_axis.get_title()
    assert "0.5" in image_axis.get_title()


def test_qt_magnification_enlarges_export_without_changing_metric_aspect():
    normal = render_lamellar_3d(_scene(), camera={"yaw": 0., "pitch": 0., "ortho_height": 35.})
    zoomed = render_lamellar_3d(_scene(), camera={"yaw": 0., "pitch": 0., "ortho_height": 70.})
    np.testing.assert_allclose(zoomed.axes[0].get_box_aspect(), normal.axes[0].get_box_aspect() * 2.)
