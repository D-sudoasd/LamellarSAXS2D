"""Atomic layered export for LamellarSAXS2D publication figures."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import shutil
import tempfile
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib import rc_context

from .lamellar_export import _jsonable, _parameter_rows, _scene_arrays
from .publication_models import PublicationFigureSpec, PublicationRenderResult, PublicationStyle
from .publication_render import _as_render_result, _scene_status, render_publication_figure


PUBLICATION_EXPORT_FORMAT = "lamellarsaxs2d.publication-export.v1"


class PublicationExportCancelled(RuntimeError):
    """Raised when publication export is cancelled before atomic publication."""


def _get(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _check_cancel(event: Any) -> None:
    if event is None:
        return
    try:
        method = getattr(event, "is_set", None)
        active = bool(method()) if callable(method) else bool(event() if callable(event) else event)
    except Exception:
        active = False
    if active:
        raise PublicationExportCancelled("publication export cancelled")


def _report_progress(callback: Any, current: int, total: int) -> None:
    if callback is None:
        return
    try:
        callback(current, total)
    except TypeError:
        callback(current)


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _write_parameter_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    columns = {"source_index"}
    for row in rows:
        columns.update(str(key) for key in row)
    ordered = ["source_index"] + sorted(item for item in columns if item != "source_index")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            output = {}
            for name in ordered:
                value = row.get(name, "")
                output[name] = json.dumps(_jsonable(value), ensure_ascii=False, separators=(",", ":")) if isinstance(value, (Mapping, list, tuple, np.ndarray)) else "" if value is None else str(_jsonable(value))
            writer.writerow(output)


def _native_render(
    scene: Any,
    style: PublicationStyle,
    spec: PublicationFigureSpec,
    *,
    camera: Any,
    cancel_event: Any,
) -> PublicationRenderResult:
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("publication-quality native rendering must be requested on the Qt main thread")
    try:
        from .ui.lamellar_3d import render_publication_scene
    except Exception as exc:  # pragma: no cover - optional UI host
        raise RuntimeError("native publication renderer is unavailable; supply rendered_scene from the UI main thread") from exc
    rendered = render_publication_scene(
        scene,
        style=style,
        width=spec.main_pixel_size[0],
        height=spec.main_pixel_size[1],
        camera=camera,
        quality=spec.quality,
        # Keep the native layer transparent in all modes; paper/background
        # appearance belongs to the Matplotlib artboard, not the Qt capture.
        transparent=True,
        cancel_event=cancel_event,
    )
    return _as_render_result(rendered)


def _new_stage(destination: str | os.PathLike[str] | Path) -> tuple[Path, Path]:
    raw = Path(destination).expanduser()
    target = raw.resolve()
    if os.path.lexists(raw) or os.path.lexists(target):
        raise FileExistsError(f"publication export target already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    return target, stage


def _publish(stage: Path, target: Path) -> None:
    if os.path.lexists(target):
        raise FileExistsError(f"publication export target already exists: {target}")
    os.rename(stage, target)


def _geometry_metrics(scene: Any) -> dict[str, Any]:
    centers = np.asarray(_get(scene, "centers", np.empty((0, 3))), dtype=float)
    sizes = np.asarray(_get(scene, "sizes", np.empty((0, 3))), dtype=float)
    stack_ids = np.asarray(_get(scene, "stack_ids", np.empty((0,))), dtype=int)
    return {
        "slab_count": int(len(centers)) if centers.ndim == 2 else 0,
        "stack_count": int(len(np.unique(stack_ids))) if stack_ids.size else 0,
        "centers_finite": bool(np.all(np.isfinite(centers))) if centers.size else True,
        "sizes_finite": bool(np.all(np.isfinite(sizes))) if sizes.size else True,
    }


def _source_identity(scene: Any) -> Any:
    value = _get(scene, "source_identity", None)
    if isinstance(value, Mapping):
        return _jsonable(value)
    metadata = _get(scene, "metadata", {})
    if isinstance(metadata, Mapping) and isinstance(metadata.get("source_identity"), Mapping):
        return _jsonable(metadata["source_identity"])
    return _jsonable(value if value not in (None, "") else {})


def _scene_field(scene: Any, name: str, default: Any = None) -> Any:
    value = _get(scene, name, default)
    if value not in (None, ""):
        return value
    metadata = _get(scene, "metadata", {})
    if isinstance(metadata, Mapping) and name in metadata:
        return metadata[name]
    return default


def _native_contract(scene: Any, style: PublicationStyle, spec: PublicationFigureSpec, rendered: PublicationRenderResult) -> dict[str, Any]:
    """Validate native provenance without inventing GPU/backend evidence."""

    provenance = rendered.provenance if isinstance(rendered.provenance, Mapping) else {}
    renderer_name = str(provenance.get("renderer", "")).strip().lower()
    backend = provenance.get("renderer_backend", provenance.get("renderer"))
    backend_text = str(backend or "").strip().lower()
    test_fixture = bool(provenance.get("test_fixture")) or backend_text.startswith("test_")
    native_qt3d = renderer_name == "qtquick3d_publication" and backend_text == "qtquick3d"
    if spec.quality == "publication":
        expected_width, expected_height = spec.main_pixel_size
        actual_height, actual_width = rendered.rgba.shape[:2]
        if abs(int(actual_width) - expected_width) > 1 or abs(int(actual_height) - expected_height) > 1:
            raise ValueError(f"publication native layer resolution {actual_width}x{actual_height} does not match main panel {expected_width}x{expected_height}")
        if not test_fixture and not native_qt3d:
            raise ValueError("publication export requires renderer=qtquick3d_publication and renderer_backend=qtquick3d, or an explicit test_fixture backend")
        if not test_fixture and not str(provenance.get("rhi_backend", "")).strip():
            raise ValueError("publication export requires a declared native RHI backend")
        if not test_fixture and str(provenance.get("quality", "")) != "publication":
            raise ValueError("native publication layer quality declaration is missing or mismatched")
    mesh = None
    geometry_hash = provenance.get("geometry_hash")
    source_geometry_hash = provenance.get("source_geometry_hash")
    if geometry_hash is not None or source_geometry_hash is not None or spec.quality == "publication":
        try:
            from .publication_geometry import build_publication_mesh

            mesh = build_publication_mesh(scene, style=style, quality=spec.quality)
        except Exception as exc:
            raise ValueError(f"could not validate publication mesh provenance: {exc}") from exc
        if geometry_hash is not None and str(geometry_hash) != str(mesh.geometry_hash):
            raise ValueError("native publication geometry_hash does not match the current scene/style")
        expected_source_hash = mesh.metadata.get("source_geometry_hash")
        if source_geometry_hash is not None and str(source_geometry_hash) != str(expected_source_hash):
            raise ValueError("native publication source_geometry_hash does not match the current scene")
        if spec.quality == "publication" and geometry_hash is None and source_geometry_hash is None:
            raise ValueError("publication native provenance must declare geometry_hash or source_geometry_hash")
        declared_style = provenance.get("style")
        if declared_style is not None and _jsonable(declared_style) != _jsonable(style.to_dict()):
            raise ValueError("native publication style declaration does not match the requested style")
    resolution = provenance.get("resolution")
    return {
        "renderer": provenance.get("renderer"),
        "renderer_backend": backend,
        "rhi_backend": provenance.get("rhi_backend"),
        "test_fixture": test_fixture,
        "publication_quality": bool(spec.quality == "publication" and not test_fixture and native_qt3d),
        "native_layer_supplied": True,
        "fallback_used": False,
        "declared_quality": provenance.get("quality", spec.quality),
        "declared_resolution": resolution,
        "geometry_hash": geometry_hash if geometry_hash is not None else (mesh.geometry_hash if mesh is not None else None),
        "source_geometry_hash": source_geometry_hash if source_geometry_hash is not None else (mesh.metadata.get("source_geometry_hash") if mesh is not None else None),
    }


def _caption(scene: Any, spec: PublicationFigureSpec) -> str:
    status = _scene_status(scene)
    labels = {
        "candidate": "Candidate parameter-driven schematic",
        "manual": "Manual-assumption schematic",
        "schematic": "Parameter-driven schematic",
    }
    if spec.language.startswith("zh"):
        labels = {"candidate": "候选参数驱动示意", "manual": "手动假设示意", "schematic": "参数驱动示意"}
    title = spec.title.strip() or "LamellarSAXS2D publication figure"
    boundary = str(_scene_field(scene, "scientific_boundary", "") or "")
    settings = _scene_field(scene, "settings", {})
    unit = "nm" if _scene_field(scene, "length_unit") == "nm" else "relative units"
    periods = "; ".join(f"{chr(65 + int(p['branch_id']))}: {float(p['period']):.4g} {unit}"
                        for p in _scene_field(scene, "populations", ()) if p.get("period") is not None)
    source = settings.get("period_source", "radial")
    mapping = {"radial": "Apparent periods follow L_app = 2π/q* from the recorded radial peaks.",
               "ellipse": "Periods use the existing ellipse-derived values and their recorded applicability checks.",
               "manual": "Periods are explicit manual schematic settings."}[source]
    identity = _source_identity(scene)
    frame = identity.get("id", identity.get("frame", "unspecified")) if isinstance(identity, Mapping) else "unspecified"
    body = (f"Source frame: {frame}. {mapping} {periods}. "
            f"The drawing contains {len(np.unique(scene.stack_ids))} stacks and {len(scene.vertices)} planar lamellae. "
            f"Thickness/period = {settings.get('thickness_ratio', .25):g}; width/period = {settings.get('width_ratio', 4):g}; "
            f"depth/period = {settings.get('depth_ratio', 4):g}. These dimensions, stack counts, positions and any morphological spread are drawing assumptions. "
            "Normals follow the scattering-peak directions by assumption. The reference arrow follows the recorded reference direction; identifying it with physical loading requires source documentation. ")
    if spec.show_inset:
        body += "The local detail shows up to three existing lamellae from one stack; all lamellae are retained in the main scene. "
    if spec.template == "evidence":
        body += "The SAXS panel displays supplied measured intensity with logarithmic contrast (1st–99.5th positive percentiles); the xy panel projects the same scene vertices. "
    return f"{title}\n{labels.get(status, 'Parameter-driven schematic')}\n\n{body}\n\n{boundary}".strip() + "\n"


def _manifest(stage: Path, *, scene: Any, spec: PublicationFigureSpec, rendered: PublicationRenderResult) -> dict[str, Any]:
    artifacts = []
    for path in sorted(item for item in stage.iterdir() if item.is_file() and item.name != "manifest.json"):
        artifacts.append({"name": path.name, "size": path.stat().st_size, "sha256": _hash(path)})
    return {
        "format_version": PUBLICATION_EXPORT_FORMAT,
        "scene_status": _scene_status(scene),
        "source_identity": _source_identity(scene),
        "spec": spec.to_dict(),
        "native_render_layer": True,
        "artifacts": artifacts,
        "files": [item["name"] for item in artifacts],
        "file_hashes": {item["name"]: item["sha256"] for item in artifacts},
        "render_camera": _jsonable(rendered.camera),
    }


def _verify_manifest(stage: Path, manifest: Mapping[str, Any]) -> None:
    for item in manifest.get("artifacts", ()):
        path = stage / str(item["name"])
        if not path.is_file() or _hash(path) != str(item["sha256"]):
            raise OSError(f"publication manifest hash mismatch: {path.name}")
    readback = json.loads((stage / "manifest.json").read_text(encoding="utf-8"))
    if readback.get("file_hashes") != manifest.get("file_hashes"):
        raise OSError("publication manifest readback mismatch")


def _save_figure(fig: Any, path: Path, spec: PublicationFigureSpec, fmt: str) -> None:
    facecolor = "none" if spec.background == "transparent" else "white"
    original_size = fig.get_size_inches().copy()
    raster = fmt in {"png", "tiff"}
    try:
        if raster:
            # Raster pixels must be integers. PDF/SVG retain the exact mm box.
            fig.set_size_inches(np.asarray(spec.pixel_size) / spec.dpi)
        destination = io.BytesIO() if raster else path
        with rc_context({"svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42}):
            fig.savefig(destination, format=fmt, dpi=spec.dpi, bbox_inches=None, pad_inches=0, facecolor=facecolor, transparent=spec.background == "transparent")
        if raster:
            from PIL import Image

            destination.seek(0)
            with Image.open(destination) as saved:
                pixels = saved.convert("RGB") if spec.background == "white" else saved.copy()
            options = {"compression": "tiff_lzw"} if fmt == "tiff" else {}
            pixels.save(path, dpi=(spec.dpi, spec.dpi), **options)
    finally:
        fig.set_size_inches(original_size)


def _output_paths(target: Path) -> dict[str, Path]:
    return {
        "png": target / "figure.png",
        "tiff": target / "figure.tiff",
        "svg": target / "figure.svg",
        "pdf": target / "figure.pdf",
        "3d_png": target / "native_3d.png",
        "source_settings": target / "source_settings.json",
        "caption": target / "figure_caption.txt",
        "provenance": target / "provenance.json",
        "parameter_sources": target / "parameter_sources.csv",
        "scene_arrays": target / "scene_arrays.npz",
        "manifest": target / "manifest.json",
    }


def export_publication_figure(
    scene: Any,
    destination: str | os.PathLike[str] | Path,
    *,
    spec: PublicationFigureSpec | Mapping[str, Any] | None = None,
    style: PublicationStyle | Mapping[str, Any] | None = None,
    rendered_scene: PublicationRenderResult | Any | None = None,
    camera: Any = None,
    observed: Any | None = None,
    qx: Any | None = None,
    qy: Any | None = None,
    cancel_event: Any | None = None,
    progress: Any | None = None,
) -> dict[str, Path]:
    """Export fixed-size publication layers without overwriting a directory."""

    resolved_spec = PublicationFigureSpec.from_mapping(spec)
    resolved_style = PublicationStyle.from_mapping(style)
    if _scene_status(scene) in {"unavailable", "stale"}:
        raise ValueError(f"cannot export {_scene_status(scene)} publication figure from unavailable/stale scene")
    _check_cancel(cancel_event)
    progress_total = 11
    progress_current = 0
    target, stage = _new_stage(destination)
    try:
        native = _as_render_result(rendered_scene) if rendered_scene is not None else _native_render(scene, resolved_style, resolved_spec, camera=camera, cancel_event=cancel_event)
        renderer_declaration = _native_contract(scene, resolved_style, resolved_spec, native)
        _check_cancel(cancel_event)
        progress_current += 1
        _report_progress(progress, progress_current, progress_total)
        figure = render_publication_figure(scene, spec=resolved_spec, style=resolved_style, rendered_scene=native, observed=observed, qx=qx, qy=qy)
        for name, fmt in (("figure.png", "png"), ("figure.tiff", "tiff"), ("figure.svg", "svg"), ("figure.pdf", "pdf")):
            _check_cancel(cancel_event)
            _save_figure(figure, stage / name, resolved_spec, fmt)
            _check_cancel(cancel_event)
            progress_current += 1
            _report_progress(progress, progress_current, progress_total)
        from PIL import Image

        _check_cancel(cancel_event)
        Image.fromarray(native.rgba, mode="RGBA").save(stage / "native_3d.png", format="PNG", dpi=(resolved_spec.dpi, resolved_spec.dpi))
        _check_cancel(cancel_event)
        progress_current += 1
        _report_progress(progress, progress_current, progress_total)
        _write_json(stage / "source_settings.json", {
            "format_version": PUBLICATION_EXPORT_FORMAT,
            "figure_spec": resolved_spec.to_dict(),
            "style": resolved_style.to_dict(),
            "camera": _jsonable(native.camera),
            "native_render_provenance": native.provenance,
        })
        _check_cancel(cancel_event)
        progress_current += 1
        _report_progress(progress, progress_current, progress_total)
        (stage / "figure_caption.txt").write_text(_caption(scene, resolved_spec), encoding="utf-8")
        _check_cancel(cancel_event)
        progress_current += 1
        _report_progress(progress, progress_current, progress_total)
        metadata = _get(scene, "metadata", {})
        metadata_map = metadata if isinstance(metadata, Mapping) else {}
        provenance = {
            "format_version": PUBLICATION_EXPORT_FORMAT,
            "revision": metadata_map.get("revision", metadata_map.get("source_revision")),
            "renderer_declaration": {
                **renderer_declaration,
                "native_layer_provenance": native.provenance,
            },
            "source_identity": _source_identity(scene),
            "scene_status": _scene_status(scene),
            "length_unit": _scene_field(scene, "length_unit", "relative"),
            "metadata": metadata,
            "settings": _scene_field(scene, "settings", {}),
            "assumptions": _scene_field(scene, "assumptions", ()),
            "scientific_boundary": _scene_field(scene, "scientific_boundary", ""),
            "geometry_metrics": _geometry_metrics(scene),
            "core_metrics": metadata_map.get("core_metrics", metadata_map.get("metrics", {})),
            "figure_spec": resolved_spec.to_dict(),
            "style": resolved_style.to_dict(),
            "camera": native.camera,
            "bounds": native.bounds,
        }
        _write_json(stage / "provenance.json", provenance)
        _check_cancel(cancel_event)
        progress_current += 1
        _report_progress(progress, progress_current, progress_total)
        _write_parameter_csv(stage / "parameter_sources.csv", _parameter_rows(scene))
        np.savez_compressed(stage / "scene_arrays.npz", **_scene_arrays(scene))
        _check_cancel(cancel_event)
        progress_current += 1
        _report_progress(progress, progress_current, progress_total)
        manifest = _manifest(stage, scene=scene, spec=resolved_spec, rendered=native)
        _write_json(stage / "manifest.json", manifest)
        _verify_manifest(stage, manifest)
        _check_cancel(cancel_event)
        progress_current += 1
        _report_progress(progress, progress_current, progress_total)
        _check_cancel(cancel_event)
        _publish(stage, target)
        output = _output_paths(target)
        return output
    except Exception:
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)
        raise


__all__ = ["PUBLICATION_EXPORT_FORMAT", "PublicationExportCancelled", "export_publication_figure"]
