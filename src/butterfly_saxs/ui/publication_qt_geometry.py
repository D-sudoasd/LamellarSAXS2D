"""Qt Quick 3D adapter for publication geometry meshes.

The publication mesh builder is deliberately Qt-free.  This module only
turns its validated positions, normals, triangle indices and per-triangle
colours into GUI-thread ``QQuick3DGeometry`` objects.  The adapter applies
the same uniform eight-unit world normalization as the interactive viewer;
the mesh contract and its geometry hash remain in the original coordinates.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

try:
    import numpy as np
except Exception:  # pragma: no cover - NumPy is a project dependency
    np = None  # type: ignore[assignment]

from .qt_compat import QT_AVAILABLE, QtCore, QtGui


def _read(source: Any, name: str, default: Any = None) -> Any:
    if isinstance(source, Mapping):
        return source.get(name, default)
    return getattr(source, name, default)


@dataclass
class PublicationGeometryGroup:
    """One merged branch/material geometry used by a QML ``Model``."""

    branch_id: int
    geometry: Any
    triangle_count: int
    color: tuple[float, float, float, float]


def _array(value: Any, *, dtype: Any, ndim: int | None = None) -> Any:
    if np is None or value is None:
        return None
    try:
        result = np.asarray(value, dtype=dtype)
    except (TypeError, ValueError):
        return None
    if ndim is not None and result.ndim != ndim:
        return None
    return result


def _mesh_arrays(mesh: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return validated mesh arrays from either a mapping or dataclass."""

    if np is None:
        raise RuntimeError("NumPy is required for publication Qt geometry")
    positions = _array(_read(mesh, "positions"), dtype=np.float32, ndim=2)
    normals = _array(_read(mesh, "normals"), dtype=np.float32, ndim=2)
    indices = _array(_read(mesh, "indices"), dtype=np.uint32, ndim=2)
    branch_ids = _array(_read(mesh, "branch_ids"), dtype=np.int32, ndim=1)
    colors = _array(_read(mesh, "colors"), dtype=np.float32, ndim=2)
    if positions is None or positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("publication positions must have shape (V, 3)")
    if normals is None or normals.shape != positions.shape:
        raise ValueError("publication normals must match positions with shape (V, 3)")
    if indices is None or indices.ndim != 2 or indices.shape[1] != 3:
        raise ValueError("publication indices must have shape (T, 3)")
    if branch_ids is None or branch_ids.shape != (len(indices),):
        raise ValueError("publication branch_ids must have shape (T,)")
    if colors is None or colors.shape != (len(indices), 4):
        raise ValueError("publication colors must have shape (T, 4)")
    if len(indices) and (int(indices.min()) < 0 or int(indices.max()) >= len(positions)):
        raise ValueError("publication triangle indices are out of range")
    if not all(np.all(np.isfinite(array)) for array in (positions, normals, colors)):
        raise ValueError("publication mesh arrays must be finite")
    if len(positions):
        norms = np.linalg.norm(normals, axis=1)
        if np.any(norms <= 1.0e-10):
            raise ValueError("publication normals must be nonzero")
        normals = normals / norms[:, None]
    return positions, normals, indices, branch_ids, np.clip(colors, 0.0, 1.0)


def _normalization_bounds(mesh: Any, bounds: Any = None) -> np.ndarray:
    if np is None:
        raise RuntimeError("NumPy is required for publication Qt geometry")
    value = bounds if bounds is not None else _read(mesh, "bounds")
    result = _array(value, dtype=float, ndim=2)
    if result is None or result.shape != (2, 3) or not np.all(np.isfinite(result)):
        positions = _array(_read(mesh, "positions"), dtype=float, ndim=2)
        if positions is None or positions.size == 0 or positions.shape[1] != 3:
            raise ValueError("publication mesh has no finite bounds")
        result = np.stack((positions.min(axis=0), positions.max(axis=0)))
    low = np.minimum(result[0], result[1])
    high = np.maximum(result[0], result[1])
    span = high - low
    nonzero = span[span > 0]
    pad = float(nonzero.max()) * 1.0e-6 if nonzero.size else 1.0
    for axis in range(3):
        if high[axis] <= low[axis]:
            low[axis] -= pad / 2.0
            high[axis] += pad / 2.0
    return np.stack((low, high)).astype(float, copy=False)


def normalize_publication_positions(mesh: Any, *, bounds: Any = None) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized positions and normalized bounds for the QML scene."""

    if np is None:
        raise RuntimeError("NumPy is required for publication Qt geometry")
    positions = _array(_read(mesh, "positions"), dtype=np.float32, ndim=2)
    if positions is None or positions.shape[1:] != (3,):
        raise ValueError("publication positions must have shape (V, 3)")
    resolved = _normalization_bounds(mesh, bounds)
    span = resolved[1] - resolved[0]
    world_scale = 8.0 / max(float(np.max(span)), 1.0e-12)
    origin = (resolved[0] + resolved[1]) / 2.0
    normalized = ((positions.astype(float) - origin) * world_scale).astype(np.float32)
    normalized_bounds = ((resolved - origin) * world_scale).astype(np.float32)
    return normalized, normalized_bounds


def _geometry_for_triangles(
    positions: np.ndarray,
    normals: np.ndarray,
    indices: np.ndarray,
    colors: np.ndarray,
    triangle_indices: np.ndarray,
    bounds: np.ndarray,
    *,
    object_name: str,
    cancel_event: Any = None,
) -> Any:
    """Create a flat-shaded, colour-per-triangle QQuick3DGeometry object."""

    if not QT_AVAILABLE or QtCore is None or QtGui is None:
        raise RuntimeError("PySide6 is required for publication Qt geometry")
    from PySide6 import QtQuick3D

    if _cancelled(cancel_event):
        raise RuntimeError("publication Qt geometry build cancelled")
    selected = indices[triangle_indices].reshape(-1)
    face_positions = positions[selected]
    face_normals = normals[selected]
    face_colors = np.repeat(colors[triangle_indices], 3, axis=0)
    # The publication path is intentionally opaque.  Alpha remains part of
    # the source mesh contract/provenance but cannot create depth-ordering
    # stripes in a stacked scientific figure.
    face_colors[:, 3] = 1.0
    interleaved = np.concatenate((face_positions, face_normals, face_colors), axis=1).astype(np.float32, copy=False)
    geometry = QtQuick3D.QQuick3DGeometry()
    geometry.setObjectName(object_name)
    geometry.setVertexData(QtCore.QByteArray(interleaved.tobytes(order="C")))
    geometry.setStride(10 * 4)
    geometry.setPrimitiveType(QtQuick3D.QQuick3DGeometry.PrimitiveType.Triangles)
    geometry.addAttribute(QtQuick3D.QQuick3DGeometry.Attribute.PositionSemantic, 0, QtQuick3D.QQuick3DGeometry.Attribute.F32Type)
    geometry.addAttribute(QtQuick3D.QQuick3DGeometry.Attribute.NormalSemantic, 12, QtQuick3D.QQuick3DGeometry.Attribute.F32Type)
    geometry.addAttribute(QtQuick3D.QQuick3DGeometry.Attribute.ColorSemantic, 24, QtQuick3D.QQuick3DGeometry.Attribute.F32Type)
    geometry.setBounds(
        QtGui.QVector3D(float(bounds[0, 0]), float(bounds[0, 1]), float(bounds[0, 2])),
        QtGui.QVector3D(float(bounds[1, 0]), float(bounds[1, 1]), float(bounds[1, 2])),
    )
    if _cancelled(cancel_event):
        raise RuntimeError("publication Qt geometry build cancelled")
    return geometry


def _cancelled(cancel_event: Any) -> bool:
    if cancel_event is None:
        return False
    checker = getattr(cancel_event, "is_set", None)
    if callable(checker):
        return bool(checker())
    if callable(cancel_event):
        return bool(cancel_event())
    return bool(cancel_event)


def build_publication_geometry_groups(
    mesh: Any,
    *,
    bounds: Any = None,
    cancel_event: Any = None,
) -> tuple[list[PublicationGeometryGroup], np.ndarray]:
    """Build one merged opaque geometry per branch/material group."""

    if np is None:
        raise RuntimeError("NumPy is required for publication Qt geometry")
    positions, normals, indices, branch_ids, colors = _mesh_arrays(mesh)
    normalized, normalized_bounds = normalize_publication_positions(mesh, bounds=bounds)
    groups: list[PublicationGeometryGroup] = []
    for branch in sorted({int(value) for value in branch_ids.tolist()}):
        if _cancelled(cancel_event):
            raise RuntimeError("publication Qt geometry build cancelled")
        triangle_indices = np.flatnonzero(branch_ids == branch).astype(np.int64)
        if not len(triangle_indices):
            continue
        geometry = _geometry_for_triangles(
            normalized,
            normals,
            indices,
            colors,
            triangle_indices,
            normalized_bounds,
            object_name=f"publicationGeometryBranch{branch}",
            cancel_event=cancel_event,
        )
        mean_color = np.mean(colors[triangle_indices], axis=0)
        color = tuple(float(value) for value in (*mean_color[:3], 1.0))
        groups.append(PublicationGeometryGroup(branch, geometry, len(triangle_indices), color))
    return groups, normalized_bounds


__all__ = [
    "PublicationGeometryGroup",
    "build_publication_geometry_groups",
    "normalize_publication_positions",
]
