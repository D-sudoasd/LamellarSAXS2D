"""Audited, non-overwriting exports for LamellarSAXS2D scenes.

The exporter is intentionally Qt-free.  It writes a fresh directory through a
private staging directory, records provenance and SHA-256 hashes, and stores
numeric scene arrays in a pickle-free NPZ sidecar.  A sequence export keeps one
PNG/GIF slot per input scene, including explicit blank frames for missing or
stale scenes.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from .lamellar_render import (
    _FORMAT_VERSION as RENDER_FORMAT_VERSION,
    _image_to_array,
    render_lamellar_2d,
    render_lamellar_3d,
    render_lamellar_combined,
)


EXPORT_FORMAT_VERSION = "lamellarsaxs2d.lamellar-export.v1"


class LamellarExportCancelled(RuntimeError):
    """Raised when a scene or sequence export is cancelled before publish."""


def _get(scene: Any, name: str, default: Any = None) -> Any:
    if isinstance(scene, Mapping):
        return scene.get(name, default)
    return getattr(scene, name, default)


def _status(scene: Any) -> str:
    return str(_get(scene, "status", "schematic") or "schematic").strip().lower()


def _source_identity(scene: Any) -> Any:
    value = _get(scene, "source_identity", None)
    if value not in (None, ""):
        return value
    metadata = _get(scene, "metadata", {})
    if isinstance(metadata, Mapping):
        for key in ("frame_id", "source_id", "id", "path", "source"):
            if metadata.get(key) not in (None, ""):
                return metadata[key]
    return "Lamellar scene"


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (Path, os.PathLike)):
        return str(value)
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if is_dataclass(value):
        try:
            return _jsonable(asdict(value))
        except (TypeError, ValueError):
            return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    enum_value = getattr(value, "value", None)
    if enum_value is not None and enum_value is not value:
        return _jsonable(enum_value)
    return str(value)


def _json_text(value: Any) -> str:
    return json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _scene_arrays(scene: Any) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    specs = {
        "centers": (float, (0, 3)),
        "sizes": (float, (0, 3)),
        "orientations": (float, (0, 3, 3)),
        "vertices": (float, (0, 8, 3)),
        "stack_ids": (int, (0,)),
        "branch_ids": (int, (0,)),
        "colors": (float, (0, 4)),
        "bounds": (float, (2, 3)),
    }
    for name, (dtype, empty_shape) in specs.items():
        value = _get(scene, name, None)
        try:
            array = np.asarray(value, dtype=dtype)
        except (TypeError, ValueError):
            array = np.empty(empty_shape, dtype=dtype)
        if array.shape != empty_shape and (
            name in {"centers", "sizes", "orientations", "vertices", "colors"} and (array.ndim != len(empty_shape) or array.shape[1:] != empty_shape[1:])
            or name in {"stack_ids", "branch_ids"} and array.ndim != 1
            or name == "bounds" and array.shape != (2, 3)
        ):
            array = np.empty(empty_shape, dtype=dtype)
        if array.size == 0:
            array = np.empty(empty_shape, dtype=dtype)
        arrays[name] = array
    populations = _get(scene, "populations", ())
    population_rows: list[tuple[Any, Any, Any, Any]] = []
    try:
        for row in populations or ():
            if isinstance(row, Mapping):
                population_rows.append((row.get("branch_id"), row.get("period"), row.get("angle_deg"), row.get("status", "")))
            else:
                values = list(row)
                values += [""] * max(0, 4 - len(values))
                population_rows.append(tuple(values[:4]))
    except (TypeError, ValueError):
        population_rows = []
    arrays["population_branch_id"] = np.asarray([row[0] for row in population_rows], dtype=float) if population_rows else np.empty((0,), dtype=float)
    arrays["population_period"] = np.asarray([row[1] for row in population_rows], dtype=float) if population_rows else np.empty((0,), dtype=float)
    arrays["population_angle_deg"] = np.asarray([row[2] for row in population_rows], dtype=float) if population_rows else np.empty((0,), dtype=float)
    arrays["population_status"] = np.asarray([str(row[3]) for row in population_rows], dtype="U64") if population_rows else np.empty((0,), dtype="U1")
    arrays["scene_metadata_json"] = np.asarray(_json_text({
        "metadata": _get(scene, "metadata", {}),
        "assumptions": _get(scene, "assumptions", ()),
        "scientific_boundary": _get(scene, "scientific_boundary", ""),
        "message": _get(scene, "message", ""),
    }))
    arrays["export_format_version"] = np.asarray(EXPORT_FORMAT_VERSION)
    arrays["scene_status"] = np.asarray(_status(scene))
    arrays["length_unit"] = np.asarray(str(_get(scene, "length_unit", "relative")))
    return arrays


def _parameter_rows(scene: Any) -> list[dict[str, Any]]:
    sources = _get(scene, "parameter_sources", ())
    if sources is None:
        return []
    if isinstance(sources, Mapping):
        sources = [sources]
    rows: list[dict[str, Any]] = []
    try:
        iterable = list(sources)
    except TypeError:
        iterable = [sources]
    for index, value in enumerate(iterable):
        if isinstance(value, Mapping):
            row = {str(key): item for key, item in value.items()}
        elif is_dataclass(value):
            row = {str(key): item for key, item in asdict(value).items()}
        elif hasattr(value, "__dict__"):
            row = {str(key): item for key, item in vars(value).items()}
        else:
            row = {"value": value}
        row.setdefault("source_index", index)
        rows.append(row)
    return rows


def _write_parameter_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    columns = {"source_index"}
    for row in rows:
        columns.update(str(key) for key in row)
    ordered = ["source_index"] + sorted(column for column in columns if column != "source_index")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: _json_text(row.get(name, "")) if isinstance(row.get(name, ""), (Mapping, list, tuple, np.ndarray)) else str(_jsonable(row.get(name, "")) if row.get(name, "") is not None else "") for name in ordered})


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cancelled(event: Any) -> bool:
    if event is None:
        return False
    try:
        method = getattr(event, "is_set", None)
        if callable(method):
            return bool(method())
        if callable(event):
            return bool(event())
        return bool(event)
    except Exception:
        return False


def _check_cancel(event: Any) -> None:
    if _cancelled(event):
        raise LamellarExportCancelled("lamellar export cancelled")


def _new_stage(destination: str | os.PathLike[str] | Path) -> tuple[Path, Path]:
    raw_target = Path(destination).expanduser()
    target = raw_target.resolve()
    if os.path.lexists(raw_target) or os.path.lexists(target):
        raise FileExistsError(f"export target already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    return target, stage


def _publish(stage: Path, target: Path) -> None:
    if os.path.lexists(target):
        raise FileExistsError(f"export target already exists: {target}")
    # os.rename is a same-volume directory transaction and does not overwrite
    # an existing directory on Windows.  The existence check above also makes
    # the common race explicit before publishing.
    try:
        os.rename(stage, target)
    except FileExistsError as exc:
        raise FileExistsError(f"export target already exists: {target}") from exc


def _scene_payload(scene: Any) -> dict[str, Any]:
    metadata = _get(scene, "metadata", {})
    metadata = metadata if isinstance(metadata, Mapping) else {}
    colors = _get(scene, "colors", ())
    branch_ids = _get(scene, "branch_ids", ())
    return {
        "status": _status(scene),
        "message": _get(scene, "message", ""),
        "source_identity": _source_identity(scene),
        "length_unit": _get(scene, "length_unit", "relative"),
        "source_units": {
            "length_unit": _get(scene, "length_unit", "relative"),
            "q_unit": metadata.get("q_unit"),
        },
        "metadata": metadata,
        "assumptions": _get(scene, "assumptions", ()),
        "settings": _get(scene, "settings", {}),
        "parameter_sources": _parameter_rows(scene),
        "scientific_boundary": _get(scene, "scientific_boundary", ""),
        "draw_axis_deg": _get(scene, "draw_axis_deg", None),
        "reference_period": _get(scene, "reference_period", None),
        "populations": _get(scene, "populations", ()),
        "appearance": {
            "branch_ids": branch_ids,
            "colors": colors,
        },
    }


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _capture_uint8(value: Any) -> np.ndarray | None:
    """Normalize a QImage-derived array without changing its pixel extent."""

    array = np.asarray(value)
    if array.ndim not in {2, 3} or array.size == 0:
        return None
    if array.ndim == 3 and array.shape[-1] not in {3, 4}:
        return None
    if array.dtype.kind in "fc":
        finite = np.isfinite(array)
        if not np.any(finite):
            return None
        high = float(np.nanmax(array[finite]))
        if high <= 1.001:
            array = array * 255.0
    elif array.dtype.kind not in "uib":
        return None
    return np.clip(np.nan_to_num(array, nan=0.0, posinf=255.0, neginf=0.0), 0, 255).astype(np.uint8, copy=False)


def _capture_png(path: Path, capture: Any, *, scene: Any | None = None, language: str = "zh") -> bool:
    raw_array = _image_to_array(capture)
    array = _capture_uint8(raw_array) if raw_array is not None else None
    if array is None:
        return False
    try:
        from PIL import Image, ImageDraw, ImageFont

        image = Image.fromarray(array, mode="L" if array.ndim == 2 else "RGBA" if array.shape[-1] == 4 else "RGB")
        if scene is not None:
            from .lamellar_render import _font, _title_suffix

            caption = _title_suffix(scene, language)
            size = max(18, min(48, image.width // 55))
            try:
                font_path = _font().get_file()
                font = ImageFont.truetype(str(font_path), size=size)
            except (OSError, TypeError, ValueError):
                font = ImageFont.load_default()
                size = 18
            max_chars = max(24, image.width // max(size // 2, 1))
            if len(caption) > max_chars:
                caption = caption[: max_chars - 1] + "…"
            overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            bbox = draw.textbbox((0, 0), caption, font=font)
            pad = max(8, size // 3)
            bar_height = bbox[3] - bbox[1] + 2 * pad
            draw.rectangle((0, 0, image.width, bar_height), fill=(250, 248, 244, 230))
            draw.text((pad, pad - bbox[1]), caption, font=font, fill=(37, 50, 61, 255))
            image = Image.alpha_composite(image.convert("RGBA"), overlay)
            if array.ndim == 2:
                image = image.convert("RGB")
            elif array.shape[-1] == 3:
                image = image.convert("RGB")
        image.save(path, format="PNG")
        return path.is_file() and path.stat().st_size > 0
    except ImportError:  # pragma: no cover - Pillow ships with normal Matplotlib installs
        # Retain the exact native extent with Agg if Pillow is unavailable.
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        height, width = array.shape[:2]
        fig = Figure(figsize=(width / 240.0, height / 240.0), dpi=240, facecolor="white")
        FigureCanvasAgg(fig)
        axis = fig.add_axes((0.0, 0.0, 1.0, 1.0))
        axis.imshow(array, interpolation="nearest")
        axis.set_axis_off()
        if scene is not None:
            from .lamellar_render import _font, _title_suffix

            fig.text(0.012, 0.988, _title_suffix(scene, language), ha="left", va="top", color="#25323d", fontproperties=_font(), fontsize=max(8, width / 220.0), weight="semibold")
        fig.savefig(path, dpi=240, bbox_inches=None, pad_inches=0)
        return path.is_file() and path.stat().st_size > 0


def _write_scene_artifacts(
    scene: Any,
    stage: Path,
    *,
    observed: Any | None,
    qx: Any | None,
    qy: Any | None,
    image_3d: Any | None,
    camera: Any | None,
    language: str,
    bounds: Any | None = None,
    cancel_event: Any | None = None,
) -> None:
    _check_cancel(cancel_event)
    figure_2d = render_lamellar_2d(scene, language=language, bounds=bounds)
    from matplotlib import rc_context

    with rc_context({"svg.fonttype": "none", "pdf.fonttype": 42}):
        figure_2d.savefig(stage / "lamellar_2d.svg", format="svg", bbox_inches="tight")
        figure_2d.savefig(stage / "lamellar_2d.pdf", format="pdf", bbox_inches="tight")
    _check_cancel(cancel_event)
    provided_capture = _capture_png(stage / "lamellar_3d.png", image_3d, scene=scene, language=language)
    if not provided_capture:
        figure_3d = render_lamellar_3d(scene, camera=camera, language=language, bounds=bounds)
        figure_3d.savefig(stage / "lamellar_3d.png", format="png", dpi=320, bbox_inches="tight")
    _check_cancel(cancel_event)
    # Paper panels render the same geometry directly, keeping labels readable
    # instead of shrinking the interactive viewport's corner controls.  The
    # supplied high-resolution capture remains available as lamellar_3d.png.
    figure_combined = render_lamellar_combined(scene, observed=observed, qx=qx, qy=qy, camera=camera, language=language, bounds=bounds)
    figure_combined.savefig(stage / "lamellar_combined.png", format="png", dpi=240, bbox_inches="tight")
    _check_cancel(cancel_event)
    _write_json(stage / "settings.json", {
        "format_version": EXPORT_FORMAT_VERSION,
        "render_format_version": RENDER_FORMAT_VERSION,
        "scene_status": _status(scene),
        "length_unit": _get(scene, "length_unit", "relative"),
        "scene_settings": _get(scene, "settings", {}),
        "render": {"language": language, "camera": camera, "bounds": bounds,
                   "three_d_png": "provided_capture" if provided_capture else "matplotlib_shared_geometry",
                   "combined_3d": "matplotlib_shared_geometry"},
    })
    _write_json(stage / "provenance.json", {
        "format_version": EXPORT_FORMAT_VERSION,
        "source_identity": _source_identity(scene),
        "scene": _scene_payload(scene),
        "scientific_boundary": _get(scene, "scientific_boundary", ""),
        "parameter_sources": _parameter_rows(scene),
    })
    _write_parameter_csv(stage / "parameter_sources.csv", _parameter_rows(scene))
    np.savez_compressed(stage / "scene_arrays.npz", **_scene_arrays(scene))


def _artifact_manifest(stage: Path, *, scene: Any, frame_entries: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    paths = sorted(path for path in stage.iterdir() if path.is_file() and path.name != "manifest.json")
    artifacts = [{"name": path.name, "size": path.stat().st_size, "sha256": _hash(path)} for path in paths]
    return {
        "format_version": EXPORT_FORMAT_VERSION,
        "scene_status": _status(scene),
        "source_identity": _source_identity(scene),
        "files": [item["name"] for item in artifacts],
        "artifacts": artifacts,
        "file_hashes": {item["name"]: item["sha256"] for item in artifacts},
        "frames": list(frame_entries or []),
    }


def _verify_manifest(stage: Path, manifest: Mapping[str, Any]) -> None:
    for item in manifest.get("artifacts", ()):
        path = stage / str(item["name"])
        if not path.is_file() or _hash(path) != str(item["sha256"]):
            raise OSError(f"manifest readback hash mismatch: {path.name}")
    manifest_path = stage / "manifest.json"
    readback = json.loads(manifest_path.read_text(encoding="utf-8"))
    if readback.get("file_hashes") != manifest.get("file_hashes"):
        raise OSError("manifest readback mismatch")


def _single_output_paths(target: Path) -> dict[str, Path]:
    return {
        "2d_svg": target / "lamellar_2d.svg",
        "2d_pdf": target / "lamellar_2d.pdf",
        "3d_png": target / "lamellar_3d.png",
        "combined_png": target / "lamellar_combined.png",
        "settings": target / "settings.json",
        "provenance": target / "provenance.json",
        "parameter_sources": target / "parameter_sources.csv",
        "scene_arrays": target / "scene_arrays.npz",
        "manifest": target / "manifest.json",
    }


def export_lamellar_scene(
    scene: Any,
    destination: str | os.PathLike[str] | Path,
    *,
    observed: Any | None = None,
    qx: Any | None = None,
    qy: Any | None = None,
    image_3d: Any | None = None,
    camera: Any | None = None,
    language: str = "zh",
    cancel_event: Any | None = None,
) -> dict[str, Path]:
    """Export one scene to a fresh, auditable directory."""

    if _status(scene) in {"unavailable", "stale"}:
        raise ValueError(f"cannot export {_status(scene)} lamellar scene: {_get(scene, 'message', '') or 'scene is not current'}")
    _check_cancel(cancel_event)
    target, stage = _new_stage(destination)
    try:
        _write_scene_artifacts(scene, stage, observed=observed, qx=qx, qy=qy, image_3d=image_3d, camera=camera, language=language, cancel_event=cancel_event)
        _check_cancel(cancel_event)
        manifest = _artifact_manifest(stage, scene=scene)
        _write_json(stage / "manifest.json", manifest)
        _verify_manifest(stage, manifest)
        _check_cancel(cancel_event)
        _publish(stage, target)
        return _single_output_paths(target)
    except Exception:
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)
        raise


def _sequence_item(value: Any, index: int) -> Any:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        if value.ndim >= 3:
            try:
                return value[index]
            except IndexError:
                return None
        return value if index == 0 else None
    if isinstance(value, (str, bytes)):
        return value if index == 0 else None
    try:
        return value[index]
    except IndexError:
        return None


def _qmap_item(value: Any, index: int) -> tuple[Any, Any, Any]:
    item = _sequence_item(value, index)
    if isinstance(item, Mapping):
        return item.get("observed", item.get("image")), item.get("qx"), item.get("qy")
    if isinstance(item, Sequence) and not isinstance(item, (str, bytes, np.ndarray)):
        values = list(item)
        if len(values) == 2:
            # Sequence qmaps supplied by the page are coordinate pairs; the
            # observed image is carried by the parallel ``images`` loader.
            return None, values[0], values[1]
        return (values + [None, None, None])[:3]  # type: ignore[return-value]
    return item, None, None


def _is_rgb_capture(value: Any) -> bool:
    array = _image_to_array(value)
    return array is not None and array.ndim == 3 and array.shape[-1] in {3, 4}


def _frame_identity(scene: Any, index: int) -> dict[str, Any]:
    metadata = _get(scene, "metadata", {})
    metadata = metadata if isinstance(metadata, Mapping) else {}
    identity = _source_identity(scene)
    identity = dict(identity) if isinstance(identity, Mapping) else {}

    def _first(names: Sequence[str], default: Any = None) -> Any:
        for container in (identity, metadata):
            for name in names:
                value = container.get(name)
                if value not in (None, ""):
                    return value
        for name in names:
            value = _get(scene, name, None)
            if value not in (None, ""):
                return value
        return default

    # Native sources put the physical frame selector inside
    # metadata.source_identity.  Keep sequence_index independent so a source
    # frame such as 110 is never rewritten to sequence slot 0.
    frame_id = _first(("frame_id", "id", "frame"), index)
    selector = _first(("frame_selector", "selector", "frame"))
    return {
        "sequence_index": index,
        "frame_index": index,
        "frame_id": _jsonable(frame_id),
        "frame_selector": _jsonable(selector),
        "source": _jsonable(_first(("source", "path"))),
        "dataset": _jsonable(_first(("dataset", "dataset_selector"))),
        "time": _jsonable(_first(("time", "timestamp"))),
        "source_identity": _jsonable(_source_identity(scene)),
        "status": _status(scene),
    }


def _sequence_parameter_rows(frame_entries: Sequence[Mapping[str, Any]], scenes: Sequence[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry, scene in zip(frame_entries, scenes, strict=False):
        parameters = _parameter_rows(scene)
        if not parameters:
            # Keep a row for a formal unavailable/missing frame so the CSV
            # preserves the frame/source identity even when no parameter was
            # produced.
            parameters = [{"source_index": "", "status": "unavailable", "reason": _get(scene, "message", "")}]
        for parameter_index, row in enumerate(parameters):
            enriched = dict(row)
            enriched["sequence_index"] = entry.get("sequence_index")
            enriched["frame_index"] = entry.get("frame_index")
            enriched["frame_id"] = entry.get("frame_id")
            enriched["frame_selector"] = entry.get("frame_selector")
            enriched["source_identity"] = entry.get("source_identity")
            enriched["parameter_index"] = parameter_index
            rows.append(enriched)
    return rows


def _blank_scene(index: int, reason: str) -> Any:
    return SimpleNamespace(
        status="unavailable",
        message=reason,
        source_identity=f"frame_{index:04d}",
        length_unit="relative",
        metadata={"frame_index": index},
        vertices=np.empty((0, 8, 3), dtype=float),
        bounds=np.asarray(((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)), dtype=float),
    )


def _sequence_bounds(scenes: Sequence[Any]) -> np.ndarray:
    points: list[np.ndarray] = []
    for scene in scenes:
        try:
            vertices = np.asarray(_get(scene, "vertices", np.empty((0, 8, 3))), dtype=float)
        except (TypeError, ValueError):
            vertices = np.empty((0, 8, 3), dtype=float)
        if vertices.ndim == 3 and vertices.shape[1:] == (8, 3) and vertices.size:
            points.append(vertices.reshape(-1, 3))
    if not points:
        return np.asarray(((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)), dtype=float)
    flat = np.concatenate(points, axis=0)
    low, high = np.nanmin(flat, axis=0), np.nanmax(flat, axis=0)
    span = np.maximum(high - low, 1.0)
    return np.vstack((low - 0.08 * span, high + 0.08 * span))


def _has_geometry(scene: Any) -> bool:
    try:
        vertices = np.asarray(_get(scene, "vertices", np.empty((0, 8, 3))), dtype=float)
    except (TypeError, ValueError):
        return False
    return vertices.ndim == 3 and vertices.shape[1:] == (8, 3) and vertices.shape[0] > 0 and bool(np.all(np.isfinite(vertices)))


def _progress(callback: Any, index: int, total: int) -> None:
    if callback is None:
        return
    try:
        callback(index, total)
    except TypeError:
        callback(index)


def export_lamellar_sequence(
    scenes: Iterable[Any],
    destination: str | os.PathLike[str] | Path,
    *,
    images: Any | None = None,
    qmaps: Any | None = None,
    camera: Any | None = None,
    language: str = "zh",
    fps: float = 5.0,
    cancel_event: Any | None = None,
    progress: Any | None = None,
) -> dict[str, Path]:
    """Export a fixed-viewport scene sequence with explicit missing slots."""

    try:
        fps_value = float(fps)
    except (TypeError, ValueError) as exc:
        raise ValueError("fps must be a finite positive number") from exc
    if not np.isfinite(fps_value) or fps_value <= 0:
        raise ValueError("fps must be a finite positive number")
    scene_list = list(scenes)
    if not scene_list:
        raise ValueError("at least one scene is required")
    _check_cancel(cancel_event)
    target, stage = _new_stage(destination)
    viewport = _sequence_bounds(scene_list)
    frame_entries: list[dict[str, Any]] = []
    provenance_scenes: list[Any] = []
    frame_paths: list[Path] = []
    try:
        for index, original in enumerate(scene_list):
            _check_cancel(cancel_event)
            source_scene = original
            if source_scene is None:
                source_scene = _blank_scene(index, "Frame is missing; no geometry was supplied" if str(language).lower().startswith("en") else "帧缺失，未提供几何数据")
            scene = source_scene
            status = _status(scene)
            missing = original is None or status in {"unavailable", "stale"} or not _has_geometry(scene)
            if missing and original is not None and status == "stale" and not _get(scene, "message", ""):
                scene = _blank_scene(index, "Frame is stale; no current geometry was exported" if str(language).lower().startswith("en") else "帧已过期，未导出当前几何数据")
            qmap_observed, qx, qy = _qmap_item(qmaps, index)
            image_item = _sequence_item(images, index)
            image_missing = images is not None and image_item is None and qmap_observed is None
            if _is_rgb_capture(image_item):
                capture = image_item
                observed = qmap_observed
            else:
                # The page's sequence ``images`` input is normally the raw
                # observed detector frame.  A colour/RGBA item is reserved
                # for an optional 3-D capture, so 2-D arrays stay in the
                # observed panel with their q-map coordinates.
                capture = None
                observed = qmap_observed if qmap_observed is not None else image_item
            if observed is not None or capture is not None:
                figure = render_lamellar_combined(scene, observed=observed, qx=qx, qy=qy, image_3d=capture, camera=camera, language=language, bounds=viewport)
            else:
                figure = render_lamellar_2d(scene, language=language, bounds=viewport)
            frame_path = stage / f"frame_{index:04d}.png"
            figure.savefig(frame_path, format="png", dpi=220, bbox_inches="tight")
            frame_paths.append(frame_path)
            provenance_scenes.append(source_scene)
            entry = _frame_identity(source_scene, index)
            entry.update({
                "file": frame_path.name,
                "missing": bool(missing),
                "geometry_missing": bool(missing),
                "image_missing": bool(image_missing),
                "known_missing_image": bool(image_missing),
                "image_status": "missing" if image_missing else "available" if (image_item is not None or qmap_observed is not None) else "not_supplied",
                "message": _jsonable(_get(scene, "message", "")),
                "render_status": _status(scene),
                "scene": _scene_payload(source_scene),
            })
            frame_entries.append(entry)
            _progress(progress, index + 1, len(scene_list))
        _check_cancel(cancel_event)
        parameter_sources_path = stage / "parameter_sources.csv"
        _write_parameter_csv(parameter_sources_path, _sequence_parameter_rows(frame_entries, provenance_scenes))
        try:
            from PIL import Image
        except ImportError as exc:  # pragma: no cover - Pillow is matplotlib's normal dependency
            raise RuntimeError("Pillow is required for GIF sequence export") from exc
        pil_frames = []
        for path in frame_paths:
            with Image.open(path) as image:
                pil_frames.append(image.convert("RGB"))
        # GIF encoders are allowed to coalesce identical consecutive images;
        # that would erase missing/valid frame slots from a scientific
        # sequence.  Encode the slot number in one corner pixel.  At the
        # export resolution this is sub-pixel to the reader, but guarantees
        # that every input frame remains a real GIF frame.
        for index, image in enumerate(pil_frames):
            red = index % 256
            green = (index // 256) % 256
            blue = (index // (256 * 256)) % 256
            image.putpixel((0, 0), (red, green, blue))
        gif_path = stage / "lamellar_sequence.gif"
        pil_frames[0].save(gif_path, save_all=True, append_images=pil_frames[1:], duration=max(1, int(round(1000.0 / fps_value))), loop=0, optimize=False, disposal=2)
        for image in pil_frames:
            image.close()
        manifest = {
            "format_version": EXPORT_FORMAT_VERSION,
            "render_format_version": RENDER_FORMAT_VERSION,
            "sequence": True,
            "camera": _jsonable(camera),
            "language": language,
            "fps": fps_value,
            "frame_count": len(scene_list),
            "frames": frame_entries,
            "viewport_bounds": _jsonable(viewport),
            "files": [path.name for path in frame_paths] + [gif_path.name, parameter_sources_path.name],
        }
        _write_json(stage / "manifest.json", manifest)
        # Add file hashes after the frame identity records are written.  The
        # manifest itself is deliberately excluded to avoid a circular hash.
        artifacts = [{"name": path.name, "size": path.stat().st_size, "sha256": _hash(path)} for path in sorted(stage.iterdir()) if path.is_file() and path.name != "manifest.json"]
        manifest["artifacts"] = artifacts
        manifest["file_hashes"] = {item["name"]: item["sha256"] for item in artifacts}
        _write_json(stage / "manifest.json", manifest)
        _verify_manifest(stage, manifest)
        _check_cancel(cancel_event)
        _publish(stage, target)
        result = {f"frame_{index:04d}": target / f"frame_{index:04d}.png" for index in range(len(scene_list))}
        result["gif"] = target / "lamellar_sequence.gif"
        result["parameter_sources"] = target / "parameter_sources.csv"
        result["manifest"] = target / "manifest.json"
        return result
    except Exception:
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)
        raise


__all__ = [
    "EXPORT_FORMAT_VERSION",
    "LamellarExportCancelled",
    "export_lamellar_scene",
    "export_lamellar_sequence",
]
