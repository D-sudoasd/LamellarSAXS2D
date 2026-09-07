from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pytest

from butterfly_saxs.lamellar import build_lamellar_scene
from butterfly_saxs.publication_export import PublicationExportCancelled, export_publication_figure
from butterfly_saxs.publication_geometry import build_publication_mesh
from butterfly_saxs.publication_models import PublicationFigureSpec, PublicationRenderResult, PublicationStyle


def _scene():
    source = {
        "source_identity": {"source": "frame_061.cbf", "frame": 61, "id": "61"},
        "q_unit": "nm^-1",
        "lobe_radial_peaks": [
            {"angle_deg": 30.0, "q_star": 0.2, "q_unit": "nm^-1", "valid": True, "branch_id": 0},
            {"angle_deg": 210.0, "q_star": 0.2, "q_unit": "nm^-1", "valid": True, "branch_id": 0},
            {"angle_deg": 150.0, "q_star": 0.1, "q_unit": "nm^-1", "valid": True, "branch_id": 1},
            {"angle_deg": 330.0, "q_star": 0.1, "q_unit": "nm^-1", "valid": True, "branch_id": 1},
        ],
    }
    scene = build_lamellar_scene(source, {"layer_count": 2, "stack_count": 2})
    scene.metadata["display_q_window"] = [0.1, 0.5]
    return scene


def _rendered(scene) -> PublicationRenderResult:
    rgba = np.full((120, 160, 4), 238, dtype=np.uint8)
    rgba[..., 3] = 255
    return PublicationRenderResult(rgba=rgba, camera={"yaw": -38.0, "pitch": -26.0}, bounds=scene.bounds, provenance={"renderer": "test-native"})


def test_publication_export_writes_fixed_layers_and_hash_manifest(tmp_path: Path) -> None:
    scene = _scene()
    spec = PublicationFigureSpec(width_mm=89.0, height_mm=70.0, dpi=120, language="en", quality="preview")
    calls = []
    output = export_publication_figure(scene, tmp_path / "figure", spec=spec, rendered_scene=_rendered(scene), progress=lambda current, total: calls.append((current, total)))

    assert {"png", "tiff", "svg", "pdf", "3d_png", "source_settings", "caption", "provenance", "parameter_sources", "scene_arrays", "manifest"} == set(output)
    assert all(path.is_file() and path.stat().st_size > 0 for path in output.values())
    manifest = json.loads(output["manifest"].read_text(encoding="utf-8"))
    assert manifest["native_render_layer"] is True
    assert manifest["file_hashes"]["figure.svg"]
    with np.load(output["scene_arrays"], allow_pickle=False) as arrays:
        np.testing.assert_allclose(arrays["vertices"], scene.vertices)
    from PIL import Image

    with Image.open(output["png"]) as image:
        assert image.size == spec.pixel_size
        assert image.mode == "RGB"
        # PNG pHYs stores integer pixels/metre, not a floating-point DPI.
        encoded_dpi = int(spec.dpi / .0254 + .5) * .0254
        assert image.info["dpi"][0] == pytest.approx(encoded_dpi, abs=1e-6)
    with Image.open(output["tiff"]) as image:
        assert image.size == spec.pixel_size
        assert image.mode == "RGB"
    assert b"<text" in output["svg"].read_bytes()
    assert "Candidate" not in output["caption"].read_text(encoding="utf-8")
    assert len(calls) >= 11 and calls[-1] == (11, 11)


def test_publication_export_rejects_stale_and_existing_targets(tmp_path: Path) -> None:
    scene = _scene()
    target = tmp_path / "existing"
    target.mkdir()
    with pytest.raises(FileExistsError):
        export_publication_figure(scene, target, spec=PublicationFigureSpec(quality="preview"), rendered_scene=_rendered(scene))
    scene.metadata["status"] = "stale"
    with pytest.raises(ValueError, match="unavailable/stale"):
        export_publication_figure(scene, tmp_path / "stale", spec=PublicationFigureSpec(quality="preview"), rendered_scene=_rendered(scene))


def test_publication_export_cancellation_does_not_publish(tmp_path: Path) -> None:
    scene = _scene()
    event = type("Event", (), {"is_set": lambda self: True})()
    with pytest.raises(PublicationExportCancelled):
        export_publication_figure(scene, tmp_path / "cancelled", rendered_scene=_rendered(scene), cancel_event=event)
    assert not (tmp_path / "cancelled").exists()


def test_cancel_from_final_progress_does_not_publish(tmp_path):
    import threading

    event = threading.Event()
    scene = _scene()
    with pytest.raises(PublicationExportCancelled):
        export_publication_figure(scene, tmp_path / "cancelled-final", rendered_scene=_rendered(scene),
            spec=PublicationFigureSpec(width_mm=89., height_mm=70., dpi=120, quality="preview"),
            cancel_event=event, progress=lambda current, total: event.set() if current == total else None)
    assert not (tmp_path / "cancelled-final").exists()


def _valid_publication_rendered(scene, spec: PublicationFigureSpec) -> PublicationRenderResult:
    style = PublicationStyle()
    mesh = build_publication_mesh(scene, style=style, quality="publication")
    width, height = spec.main_pixel_size
    return PublicationRenderResult(
        rgba=np.full((height, width, 4), 238, dtype=np.uint8),
        camera={"yaw": -38.0, "pitch": -26.0, "ortho_height": 35.0},
        bounds=scene.bounds,
        provenance={
            "renderer_backend": "test_fixture",
            "rhi_backend": "test_instrument",
            "test_fixture": True,
            "quality": "publication",
            "resolution": [width, height],
            "geometry_hash": mesh.geometry_hash,
            "source_geometry_hash": mesh.metadata["source_geometry_hash"],
            "style": style.to_dict(),
        },
    )


def test_publication_contract_accepts_explicit_fixture_and_preserves_identity(tmp_path: Path) -> None:
    scene = _scene()
    spec = PublicationFigureSpec(width_mm=50.0, height_mm=40.0, dpi=72, quality="publication")
    rendered = _valid_publication_rendered(scene, spec)
    output = export_publication_figure(scene, tmp_path / "valid", spec=spec, rendered_scene=rendered, camera={"yaw": 999})

    provenance = json.loads(output["provenance"].read_text(encoding="utf-8"))
    assert provenance["source_identity"]["frame"] == 61
    assert provenance["camera"] == rendered.camera
    assert provenance["renderer_declaration"]["test_fixture"] is True
    assert provenance["renderer_declaration"]["publication_quality"] is False


def test_qt3d_renderer_pair_is_recognized_as_real_publication_backend(tmp_path: Path) -> None:
    scene = _scene()
    spec = PublicationFigureSpec(width_mm=50.0, height_mm=40.0, dpi=72, quality="publication")
    rendered = _valid_publication_rendered(scene, spec)
    rendered.provenance.update({
        "test_fixture": False,
        "renderer": "qtquick3d_publication",
        "renderer_backend": "qtquick3d",
        "rhi_backend": "GraphicsApi.Direct3D11",
    })
    output = export_publication_figure(scene, tmp_path / "qt3d-contract", spec=spec, rendered_scene=rendered)
    provenance = json.loads(output["provenance"].read_text(encoding="utf-8"))
    assert provenance["renderer_declaration"]["publication_quality"] is True
    assert provenance["renderer_declaration"]["renderer_backend"] == "qtquick3d"


def test_publication_rejects_preview_sized_layer_for_publication(tmp_path: Path) -> None:
    scene = _scene()
    spec = PublicationFigureSpec(width_mm=89.0, height_mm=70.0, dpi=600, quality="publication")
    with pytest.raises(ValueError, match="resolution"):
        export_publication_figure(scene, tmp_path / "wrong-resolution", spec=spec, rendered_scene=_rendered(scene))


def test_publication_rejects_wrong_geometry_hash(tmp_path: Path) -> None:
    scene = _scene()
    spec = PublicationFigureSpec(width_mm=50.0, height_mm=40.0, dpi=72, quality="publication")
    rendered = _valid_publication_rendered(scene, spec)
    rendered.provenance["geometry_hash"] = "wrong-geometry"
    with pytest.raises(ValueError, match="geometry_hash"):
        export_publication_figure(scene, tmp_path / "wrong-hash", spec=spec, rendered_scene=rendered)


def test_chinese_publication_text_has_no_missing_glyph_warning(tmp_path: Path) -> None:
    scene = _scene()
    spec = PublicationFigureSpec(width_mm=50.0, height_mm=40.0, dpi=72, quality="preview", language="zh_CN", title="层状结构示意")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        output = export_publication_figure(scene, tmp_path / "chinese", spec=spec, rendered_scene=_rendered(scene))
    assert not any("Glyph" in str(item.message) and "missing" in str(item.message) for item in caught)
    try:
        import fitz
    except ImportError:
        return
    document = fitz.open(output["pdf"])
    assert "层状结构示意" in document[0].get_text()
