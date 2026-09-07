from __future__ import annotations

import numpy as np
import pytest

from butterfly_saxs.lamellar import build_lamellar_scene
from butterfly_saxs.publication_models import PublicationFigureSpec, PublicationRenderResult, PublicationStyle
from butterfly_saxs.publication_render import render_publication_figure


def _source() -> dict:
    return {
        "source_identity": {"source": "frame_061.cbf", "frame": 61, "id": "61"},
        "q_unit": "nm^-1",
        "lobe_radial_peaks": [
            {"angle_deg": 30.0, "q_star": 0.2, "q_unit": "nm^-1", "valid": True, "branch_id": 0},
            {"angle_deg": 210.0, "q_star": 0.2, "q_unit": "nm^-1", "valid": True, "branch_id": 0},
            {"angle_deg": 150.0, "q_star": 0.1, "q_unit": "nm^-1", "valid": True, "branch_id": 1},
            {"angle_deg": 330.0, "q_star": 0.1, "q_unit": "nm^-1", "valid": True, "branch_id": 1},
        ],
    }


def _scene(*, manual: bool = False):
    settings = {"period_source": "manual", "manual_period": 2.5, "manual_unit": "relative", "layer_count": 3, "stack_count": 2} if manual else {"layer_count": 3, "stack_count": 2}
    scene = build_lamellar_scene({} if manual else _source(), settings)
    scene.metadata["display_q_window"] = [0.1, 0.5]
    return scene


def _rendered(scene) -> PublicationRenderResult:
    rgba = np.zeros((420, 640, 4), dtype=np.uint8)
    rgba[..., :3] = (244, 241, 234)
    rgba[..., 3] = 255
    return PublicationRenderResult(rgba=rgba, camera={"yaw": -38.0, "pitch": -26.0}, bounds=scene.bounds, provenance={"renderer": "test-native-main-thread"})


def _q_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.linspace(-0.6, 0.6, 61)
    qx, qy = np.meshgrid(values, values)
    observed = 100.0 + 20.0 * np.exp(-((qx - .2) ** 2 + (qy - .1) ** 2) / .01)
    return observed, qx, qy


def test_structure_template_has_fixed_artboard_and_editable_annotations() -> None:
    scene = _scene()
    spec = PublicationFigureSpec(width_mm=89.0, height_mm=70.0, dpi=120, template="structure", language="en")
    observed, qx, qy = _q_data()
    figure = render_publication_figure(scene, spec=spec, style=PublicationStyle(), rendered_scene=_rendered(scene), observed=observed, qx=qx, qy=qy)

    assert figure.get_size_inches()[0] == pytest.approx(89.0 / 25.4)
    assert figure.publication_annotation_artists.keys() >= {"a", "b", "family_0", "family_1", "normal_0", "scale"}
    assert len(figure.axes) == 2  # structure hero + local detail, no redundant SAXS
    assert all(text.get_text() != "61" for text in figure.texts if text is not figure.publication_annotation_artists.get("a"))
    assert any("Parameter-driven schematic" in text.get_text() for text in figure.texts)


def test_evidence_template_uses_left_source_panels_and_main_hero() -> None:
    scene = _scene()
    spec = PublicationFigureSpec(width_mm=183.0, height_mm=126.0, dpi=120, template="evidence", language="en")
    observed, qx, qy = _q_data()
    figure = render_publication_figure(scene, spec=spec, rendered_scene=_rendered(scene), observed=observed, qx=qx, qy=qy)

    positions = [axis.get_position().bounds for axis in figure.axes]
    assert any(position[0] < .1 and position[1] > .5 for position in positions)
    assert any(position[0] < .1 and position[1] < .5 for position in positions)
    assert any(position[0] > .3 and position[2] > .5 for position in positions)
    assert "nm⁻¹" in figure.axes[1].get_xlabel()


def test_manual_relative_scene_keeps_honest_manual_caption() -> None:
    scene = _scene(manual=True)
    figure = render_publication_figure(scene, spec=PublicationFigureSpec(width_mm=89.0, height_mm=70.0, dpi=120), rendered_scene=_rendered(scene))

    assert any("Manual-assumption schematic" in text.get_text() for text in figure.texts)
    assert scene.length_unit == "relative"
    assert figure.publication_scale["unit"] == "rel. u."


@pytest.mark.parametrize("magnification", [25., 70.])
def test_scale_bar_uses_real_camera_magnification(magnification):
    scene = _scene()
    result = _rendered(scene)
    result.camera = {"yaw": 0., "pitch": 0., "ortho_height": magnification}
    spec = PublicationFigureSpec()
    figure = render_publication_figure(scene, spec=spec, rendered_scene=result)
    scale = figure.publication_scale
    endpoints = np.array(scale["figure_endpoints"])
    # Independent pixel/magnification calculation, including panel width.
    height, width = result.rgba.shape[:2]
    expected_panel_fraction = scale["length"] * (8 / np.ptp(scene.bounds, axis=0).max()) * magnification * min(width, height) / 480 / width
    assert endpoints[1, 0] - endpoints[0, 0] == pytest.approx(expected_panel_fraction * spec.main_panel[2])
    assert endpoints[1, 1] == endpoints[0, 1]


def test_projection_and_detail_keep_source_geometry_and_measured_spacing():
    scene = _scene()
    before = scene.vertices.copy()
    spec = PublicationFigureSpec(template="evidence", height_mm=126.)
    figure = render_publication_figure(scene, spec=spec, rendered_scene=_rendered(scene))
    np.testing.assert_array_equal(scene.vertices, before)
    projection = next(axis for axis in figure.axes if axis.get_gid() == "vector-projection-layer")
    assert len(projection.patches) == len(scene.vertices)
    metrics = figure.publication_local_metrics
    assert metrics["l_app"] == pytest.approx(2 * np.pi / .2)
    assert metrics["t_schem"] == pytest.approx(.25 * metrics["l_app"])
    assert len(metrics["shown_plate_indices"]) == 3


def test_malformed_q_coordinates_cannot_silently_become_detector_pixels():
    scene = _scene()
    spec = PublicationFigureSpec(template="evidence")
    with pytest.raises(ValueError, match="shapes must match"):
        render_publication_figure(scene, spec=spec, rendered_scene=_rendered(scene),
            observed=np.ones((8, 8)), qx=np.ones((7, 8)), qy=np.ones((7, 8)))


def test_multi_stack_family_labels_are_above_the_whole_scene():
    from butterfly_saxs.publication_geometry import fit_publication_camera
    from butterfly_saxs.publication_render import _project_figure

    scene = build_lamellar_scene(_source(), {"mode": "multi", "stack_count": 12, "spread_deg": 6.})
    result = _rendered(scene)
    result.camera = fit_publication_camera(scene, width=640, height=420)
    spec = PublicationFigureSpec(height_mm=110.)
    figure = render_publication_figure(scene, spec=spec, rendered_scene=result)
    top = _project_figure(scene.vertices.reshape(-1, 3), result, spec)[:, 1].max()
    for name in ("family_0", "family_1"):
        assert figure.publication_annotation_artists[name].xyann[1] > top
