"""Small, Qt-free document and frame helpers for the schematic workbench."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..serialization import json_safe


def field(value: Any, name: str, default: Any = None) -> Any:
    return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)


def compact_source(source: Any) -> dict[str, Any]:
    """Persist measurement evidence, not full detector/profile arrays.

    The raw image is loaded independently with its container selectors.  In
    particular, saving this document cannot turn an azimuthal annulus into a
    radial measurement or discard an explicit unavailable quantitative value.
    """

    keys = (
        "source_identity", "source", "path", "frame", "dataset", "id", "time",
        "timestamp", "q_unit", "draw_axis_deg", "flags", "status", "error",
        "measurement_status", "quantitative_parameters", "quality", "analysis",
        "parameters", "lobe_radial_peaks", "lobes", "ellipse_fit", "ellipse",
        "candidate_fit", "array_path", "npz_path", "arrays_path", "array_keys", "stale",
    )
    result = {key: field(source, key) for key in keys if field(source, key) is not None}
    observables = field(source, "observables", {})
    result["observables"] = {
        key: field(observables, key)
        for key in ("q_unit", "draw_axis_deg", "flags", "lobes", "lobe_radial_peaks", "ellipse")
        if field(observables, key) is not None
    }
    butterfly = field(source, "butterfly", field(observables, "butterfly", {}))
    if butterfly:
        result["butterfly"] = {
            key: field(butterfly, key)
            for key in ("candidate_fit", "quantitative_parameters", "measurement_status", "quality")
            if field(butterfly, key) is not None
        }
    return json_safe(result)


def frame_label(source: Any, index: int) -> str:
    identity = field(source, "source_identity", {}) or {}
    path = field(identity, "source", field(source, "source", field(source, "path", "")))
    number = field(identity, "frame", field(source, "frame"))
    name = Path(str(path)).name if path else str(field(identity, "id", field(source, "id", index + 1)))
    parts = [name]
    if number is not None:
        parts.append(f"frame {number}")
    dataset = field(identity, "dataset", field(source, "dataset"))
    if dataset:
        parts.append(str(dataset))
    stamp = field(identity, "time", field(source, "time", field(source, "timestamp")))
    if stamp is not None:
        parts.append(f"t={stamp}")
    return " · ".join(parts)


def source_images(source: Any) -> tuple[Any, Any, Any]:
    """Load only the selected native frame; never guess a container selector."""

    observed = field(source, "observed", field(source, "image", field(source, "I")))
    qx, qy = field(source, "qx"), field(source, "qy")
    array_path = next((field(source, key) for key in ("array_path", "npz_path", "arrays_path")
                       if field(source, key)), None)
    if observed is None and array_path:
        with np.load(Path(array_path), allow_pickle=False) as data:
            keys = field(source, "array_keys", {}) or {}
            image_key = keys.get("observed")
            observed = (data[image_key] if image_key else
                        next((data[key] for key in ("observed", "I", "image", "data") if key in data), None))
            qx_key, qy_key = keys.get("qx", "qx"), keys.get("qy", "qy")
            qx = data[qx_key] if qx_key in data else qx
            qy = data[qy_key] if qy_key in data else qy
    if observed is None:
        identity = field(source, "source_identity", {}) or {}
        path = field(identity, "source", field(source, "source", field(source, "path")))
        if path and Path(str(path)).is_file():
            from ..io import load_image

            frame = field(identity, "frame", field(source, "frame"))
            dataset = field(identity, "dataset", field(source, "dataset"))
            loaded = load_image(path, frame=frame, dataset=dataset)
            observed = field(loaded, "data")
    if observed is not None:
        observed = np.asarray(observed)
        if observed.ndim != 2:
            raise ValueError("Selected native image must be two-dimensional")
        if (qx is None) != (qy is None):
            raise ValueError("Both qx and qy are required for a calibrated display")
        if qx is not None:
            qx, qy = np.asarray(qx), np.asarray(qy)
            vector_grid = qx.ndim == qy.ndim == 1 and (qy.size, qx.size) == observed.shape
            if not vector_grid and (qx.shape != observed.shape or qy.shape != observed.shape):
                raise ValueError("Native q maps do not match the selected image")
    return observed, qx, qy


def rgba_image(image: Any) -> np.ndarray:
    """Copy QImage pixels before a background exporter outlives its buffer."""

    converted = image.convertToFormat(image.Format.Format_RGBA8888)
    pixels = np.frombuffer(converted.bits(), dtype=np.uint8)
    rows = pixels.reshape(converted.height(), converted.bytesPerLine())
    return rows[:, :converted.width() * 4].reshape(converted.height(), converted.width(), 4).copy()


def padded_bounds(scenes: list[Any]) -> np.ndarray | None:
    available = [np.asarray(scene.bounds, dtype=float) for scene in scenes
                 if scene.metadata.get("available")]
    if not available:
        return None
    bounds = np.array([np.min([b[0] for b in available], axis=0),
                       np.max([b[1] for b in available], axis=0)])
    span = max(float(np.max(bounds[1] - bounds[0])), 1e-9)
    return bounds + np.array([[-1], [1]]) * span * 0.08


class FrameImageReader:
    """Bounded native-image cache shared by image and q-map sequence views."""

    def __init__(self, sources: list[Any]) -> None:
        self.sources = sources
        self.cache: OrderedDict[int, tuple[Any, Any, Any]] = OrderedDict()

    def read(self, index: int) -> tuple[Any, Any, Any]:
        if index not in self.cache:
            self.cache[index] = source_images(self.sources[index])
            while len(self.cache) > 3:
                self.cache.popitem(last=False)
        self.cache.move_to_end(index)
        return self.cache[index]

    def view(self, *, qmaps: bool = False) -> Sequence:
        reader = self

        class View(Sequence):
            def __len__(self) -> int:
                return len(reader.sources)

            def __getitem__(self, index: int) -> Any:
                if not isinstance(index, int) or not 0 <= index < len(self):
                    raise IndexError(index)
                values = reader.read(index)
                return values[1:] if qmaps else values[0]

        return View()
