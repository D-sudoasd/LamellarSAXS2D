"""Publication-safe mesh and camera geometry for lamellar scenes.

The module owns presentation geometry only.  It validates the supplied scene,
keeps all source coordinates in their original units, and constructs a small
analytical chamfered-box mesh without changing any scientific dimensions.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import hashlib
import math
from typing import Any

import numpy as np

from .publication_models import PublicationStyle


_CORNER_SIGNS = np.asarray(
    (
        (-1.0, -1.0, -1.0),
        (1.0, -1.0, -1.0),
        (1.0, 1.0, -1.0),
        (-1.0, 1.0, -1.0),
        (-1.0, -1.0, 1.0),
        (1.0, -1.0, 1.0),
        (1.0, 1.0, 1.0),
        (-1.0, 1.0, 1.0),
    ),
    dtype=float,
)


@dataclass
class PublicationMesh:
    """Validated triangle mesh in the source scene coordinate system."""

    positions: np.ndarray
    normals: np.ndarray
    indices: np.ndarray
    branch_ids: np.ndarray
    colors: np.ndarray
    bounds: np.ndarray
    geometry_hash: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.positions = np.asarray(self.positions, dtype=np.float32)
        self.normals = np.asarray(self.normals, dtype=np.float32)
        self.indices = np.asarray(self.indices, dtype=np.uint32)
        self.branch_ids = np.asarray(self.branch_ids, dtype=int)
        self.colors = np.asarray(self.colors, dtype=np.float32)
        self.bounds = np.asarray(self.bounds, dtype=float)
        if self.positions.ndim != 2 or self.positions.shape[1:] != (3,):
            raise ValueError("PublicationMesh.positions must have shape (V, 3)")
        if self.normals.shape != self.positions.shape:
            raise ValueError("PublicationMesh.normals must match positions")
        if self.indices.ndim != 2 or self.indices.shape[1:] != (3,):
            raise ValueError("PublicationMesh.indices must have shape (T, 3)")
        if self.branch_ids.shape != (len(self.indices),):
            raise ValueError(
                "PublicationMesh.branch_ids must have one value per triangle"
            )
        if self.colors.shape != (len(self.indices), 4):
            raise ValueError(
                "PublicationMesh.colors must have one RGBA value per triangle"
            )
        if self.bounds.shape != (2, 3) or not np.all(np.isfinite(self.bounds)):
            raise ValueError("PublicationMesh.bounds must have finite shape (2, 3)")
        if len(self.indices) and np.max(self.indices) >= len(self.positions):
            raise ValueError("PublicationMesh.indices reference missing vertices")
        if not all(
            np.all(np.isfinite(value))
            for value in (self.positions, self.normals, self.colors)
        ):
            raise ValueError("PublicationMesh arrays must be finite")
        self.metadata = dict(self.metadata or {})


def _read(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _as_array(
    value: Any, dtype: Any, shape: tuple[int, ...] | None = None
) -> np.ndarray | None:
    if value is None:
        return None
    try:
        array = np.asarray(value, dtype=dtype)
    except (TypeError, ValueError):
        return None
    if shape is not None and array.shape != shape:
        return None
    return array


def _source_geometry(scene: Any) -> dict[str, Any]:
    centers = _as_array(_read(scene, "centers"), float)
    sizes = _as_array(_read(scene, "sizes"), float)
    orientations = _as_array(_read(scene, "orientations"), float)
    vertices = _as_array(_read(scene, "vertices"), float)
    if centers is None or centers.ndim != 2 or centers.shape[1:] != (3,):
        raise ValueError("scene.centers must have shape (N, 3)")
    count = len(centers)
    if sizes is None or sizes.shape != (count, 3):
        raise ValueError("scene.sizes must have shape (N, 3)")
    if orientations is None or orientations.shape != (count, 3, 3):
        raise ValueError("scene.orientations must have shape (N, 3, 3)")
    if vertices is None or vertices.shape != (count, 8, 3):
        raise ValueError("scene.vertices must have shape (N, 8, 3)")
    if not all(
        np.all(np.isfinite(value)) for value in (centers, sizes, orientations, vertices)
    ):
        raise ValueError("publication source geometry must be finite")
    if np.any(sizes <= 0.0):
        raise ValueError("publication source sizes must be positive")
    gram = np.einsum("nij,nkj->nik", orientations, orientations)
    if not np.allclose(gram, np.eye(3)[None, :, :], atol=2e-6, rtol=2e-6):
        raise ValueError("publication source orientations must be orthonormal")
    determinant = np.linalg.det(orientations)
    if not np.all(determinant > 0.0):
        raise ValueError("publication source orientations must be right-handed")
    expected = centers[:, None, :] + (
        _CORNER_SIGNS[None, :, :] * (sizes[:, None, :] / 2.0)
    ) @ np.transpose(orientations, (0, 2, 1))
    tolerance = max(1e-7, 3e-6 * float(np.max(np.abs(sizes))))
    if not np.allclose(vertices, expected, atol=tolerance, rtol=3e-6):
        raise ValueError("scene centers/sizes/orientations disagree with vertices")
    bounds = _as_array(_read(scene, "bounds"), float, (2, 3))
    if bounds is None:
        bounds = (
            np.stack((vertices.min(axis=(0, 1)), vertices.max(axis=(0, 1))))
            if count
            else np.asarray(((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)))
        )
    if not np.all(np.isfinite(bounds)) or np.any(bounds[1] < bounds[0]):
        raise ValueError("scene.bounds must be finite with nonnegative spans")
    if count and (
        np.any(vertices.min(axis=(0, 1)) < bounds[0] - tolerance)
        or np.any(vertices.max(axis=(0, 1)) > bounds[1] + tolerance)
    ):
        raise ValueError("scene.bounds do not enclose the source vertices")
    stack_ids = _as_array(_read(scene, "stack_ids"), int)
    branch_ids = _as_array(_read(scene, "branch_ids"), int)
    if stack_ids is None or stack_ids.shape != (count,):
        stack_ids = np.arange(count, dtype=int)
    if branch_ids is None or branch_ids.shape != (count,):
        branch_ids = np.zeros(count, dtype=int)
    geometry_hash = _hash_arrays(
        centers, sizes, orientations, vertices, stack_ids, branch_ids, bounds
    )
    return {
        "centers": centers.copy(),
        "sizes": sizes.copy(),
        "orientations": orientations.copy(),
        "vertices": vertices.copy(),
        "stack_ids": stack_ids.copy(),
        "branch_ids": branch_ids.copy(),
        "bounds": bounds.copy(),
        "geometry_hash": geometry_hash,
    }


def _hash_arrays(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _style_color(style: PublicationStyle, branch: int) -> np.ndarray:
    return np.asarray(style.colors[int(branch) % 2], dtype=np.float32)


def _raise_if_cancelled(cancel_event: Any) -> None:
    if cancel_event is None:
        return
    checker = getattr(cancel_event, "is_set", None)
    if callable(checker):
        cancelled = bool(checker())
    elif callable(cancel_event):
        cancelled = bool(cancel_event())
    else:
        raise TypeError(
            "cancel_event must be an Event-like object or zero-argument callable"
        )
    if cancelled:
        raise RuntimeError("publication mesh cancelled")


def _polygons_for_box(
    half: np.ndarray, radius: float
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return local surface polygons and their outward normals."""

    polygons: list[tuple[np.ndarray, np.ndarray]] = []
    if radius <= 0.0:
        for axis in range(3):
            others = [item for item in range(3) if item != axis]
            for sign in (-1.0, 1.0):
                points = []
                for first, second in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                    point = np.zeros(3, dtype=float)
                    point[axis] = sign * half[axis]
                    point[others[0]] = first * half[others[0]]
                    point[others[1]] = second * half[others[1]]
                    points.append(point)
                normal = np.zeros(3, dtype=float)
                normal[axis] = sign
                polygons.append((np.asarray(points), normal))
        return polygons
    inset = half - radius
    for axis in range(3):
        others = [item for item in range(3) if item != axis]
        for sign in (-1.0, 1.0):
            points = []
            for first, second in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                point = np.zeros(3, dtype=float)
                point[axis] = sign * half[axis]
                point[others[0]] = first * inset[others[0]]
                point[others[1]] = second * inset[others[1]]
                points.append(point)
            normal = np.zeros(3, dtype=float)
            normal[axis] = sign
            polygons.append((np.asarray(points), normal))
    for ia, ib, ik in ((0, 1, 2), (0, 2, 1), (1, 2, 0)):
        for sa in (-1.0, 1.0):
            for sb in (-1.0, 1.0):
                low, high = -half[ik] + radius, half[ik] - radius
                p_a0 = np.zeros(3, dtype=float)
                p_b0 = np.zeros(3, dtype=float)
                p_b1 = np.zeros(3, dtype=float)
                p_a1 = np.zeros(3, dtype=float)
                for point, va, vb, vk in (
                    (p_a0, half[ia], half[ib] - radius, low),
                    (p_b0, half[ia] - radius, half[ib], low),
                    (p_b1, half[ia] - radius, half[ib], high),
                    (p_a1, half[ia], half[ib] - radius, high),
                ):
                    point[ia], point[ib], point[ik] = sa * va, sb * vb, vk
                normal = np.zeros(3, dtype=float)
                normal[ia], normal[ib] = sa, sb
                polygons.append((np.asarray((p_a0, p_b0, p_b1, p_a1)), normal))
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                points = np.asarray(
                    (
                        (sx * half[0], sy * inset[1], sz * inset[2]),
                        (sx * inset[0], sy * half[1], sz * inset[2]),
                        (sx * inset[0], sy * inset[1], sz * half[2]),
                    ),
                    dtype=float,
                )
                polygons.append((points, np.asarray((sx, sy, sz), dtype=float)))
    return polygons


def _mesh_hash(
    mesh_positions: np.ndarray,
    mesh_normals: np.ndarray,
    indices: np.ndarray,
    branch_ids: np.ndarray,
    colors: np.ndarray,
) -> str:
    return _hash_arrays(mesh_positions, mesh_normals, indices, branch_ids, colors)


def build_publication_mesh(
    scene: Any,
    style: PublicationStyle | Mapping[str, Any] | None = None,
    *,
    quality: str = "publication",
    cancel_event: Any = None,
) -> PublicationMesh:
    """Create an opaque, publication-safe mesh without altering source geometry."""

    _raise_if_cancelled(cancel_event)
    if quality not in {"preview", "publication"}:
        raise ValueError("quality must be 'preview' or 'publication'")
    resolved_style = PublicationStyle.from_mapping(style)
    source = _source_geometry(scene)
    positions: list[np.ndarray] = []
    normals: list[np.ndarray] = []
    triangles: list[tuple[int, int, int]] = []
    triangle_branches: list[int] = []
    triangle_colors: list[np.ndarray] = []
    triangle_plate_ids: list[int] = []
    triangle_stack_ids: list[int] = []
    radii: list[float] = []
    for index, (center, size, orientation, branch) in enumerate(
        zip(
            source["centers"],
            source["sizes"],
            source["orientations"],
            source["branch_ids"],
            strict=True,
        )
    ):
        _raise_if_cancelled(cancel_event)
        half = size / 2.0
        radius = min(
            float(resolved_style.bevel_fraction) * float(size[2]),
            0.02 * float(min(size[0], size[1])),
        )
        if radius < 0.0 or radius >= float(np.min(half)):
            raise ValueError(
                "publication bevel radius does not fit inside the source box"
            )
        radii.append(radius)
        for local_polygon, local_normal in _polygons_for_box(half, radius):
            world_polygon = center + local_polygon @ orientation.T
            world_normal = local_normal @ orientation.T
            norm = float(np.linalg.norm(world_normal))
            if norm <= 0.0 or not np.isfinite(norm):
                raise ValueError("publication polygon normal is invalid")
            world_normal /= norm
            cross = np.cross(
                world_polygon[1] - world_polygon[0], world_polygon[2] - world_polygon[0]
            )
            if float(np.dot(cross, world_normal)) < 0.0:
                world_polygon = world_polygon[::-1]
            base = len(positions)
            positions.extend(np.asarray(world_polygon, dtype=np.float32))
            normals.extend(
                np.tile(world_normal.astype(np.float32), (len(world_polygon), 1))
            )
            color = _style_color(resolved_style, int(branch))
            for local_index in range(1, len(world_polygon) - 1):
                triangles.append((base, base + local_index, base + local_index + 1))
                triangle_branches.append(int(branch))
                triangle_colors.append(color.copy())
                triangle_plate_ids.append(index)
                triangle_stack_ids.append(int(source["stack_ids"][index]))
    position_array = (
        np.asarray(positions, dtype=np.float32).reshape((-1, 3))
        if positions
        else np.empty((0, 3), dtype=np.float32)
    )
    normal_array = (
        np.asarray(normals, dtype=np.float32).reshape((-1, 3))
        if normals
        else np.empty((0, 3), dtype=np.float32)
    )
    index_array = (
        np.asarray(triangles, dtype=np.uint32).reshape((-1, 3))
        if triangles
        else np.empty((0, 3), dtype=np.uint32)
    )
    branch_array = np.asarray(triangle_branches, dtype=int)
    color_array = (
        np.asarray(triangle_colors, dtype=np.float32).reshape((-1, 4))
        if triangle_colors
        else np.empty((0, 4), dtype=np.float32)
    )
    mesh_hash = _mesh_hash(
        position_array, normal_array, index_array, branch_array, color_array
    )
    metadata = {
        "source_geometry_hash": source["geometry_hash"],
        "geometry_hash": mesh_hash,
        "source_geometry_status": "validated",
        "status": "publication_mesh",
        "quality": quality,
        "style_visual_only": True,
        "style": resolved_style.to_dict(),
        "triangle_count": int(len(index_array)),
        "vertex_count": int(len(position_array)),
        "plate_count": int(len(source["centers"])),
        "stack_ids": [int(value) for value in source["stack_ids"]],
        "branch_ids": [int(value) for value in source["branch_ids"]],
        "triangle_plate_ids": triangle_plate_ids,
        "triangle_stack_ids": triangle_stack_ids,
        "bevel_radii": radii,
        "bevel_radius_by_plate": radii,
        "source_envelope": source["bounds"].tolist(),
    }
    return PublicationMesh(
        position_array,
        normal_array,
        index_array,
        branch_array,
        color_array,
        source["bounds"],
        mesh_hash,
        metadata,
    )


def _camera_field(camera: Mapping[str, Any] | Any, name: str, default: Any) -> Any:
    if isinstance(camera, Mapping):
        return camera.get(name, default)
    return getattr(camera, name, default)


def camera_basis(
    camera: Mapping[str, Any] | Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return right, up, and camera-to-target direction for Qt's Y-up camera."""

    yaw = math.radians(float(_camera_field(camera, "yaw", 0.0)))
    pitch = math.radians(float(_camera_field(camera, "pitch", -10.0)))
    cy, sy = math.cos(yaw), math.sin(yaw)
    cx, sx = math.cos(pitch), math.sin(pitch)
    rotation = np.asarray(
        ((cy, sy * sx, sy * cx), (0.0, cx, -sx), (-sy, cy * sx, cy * cx)), dtype=float
    )
    right = rotation[:, 0]
    up = rotation[:, 1]
    direction = -rotation[:, 2]
    return (
        right / np.linalg.norm(right),
        up / np.linalg.norm(up),
        direction / np.linalg.norm(direction),
    )


def _camera_bounds(bounds: Any) -> np.ndarray:
    array = np.asarray(bounds, dtype=float)
    if (
        array.shape != (2, 3)
        or not np.all(np.isfinite(array))
        or np.any(array[1] < array[0])
    ):
        raise ValueError(
            "camera bounds must have finite shape (2, 3) with nonnegative spans"
        )
    return array.copy()


def project_points(
    points: Any, camera: Mapping[str, Any] | Any, bounds: Any, width: int, height: int
) -> np.ndarray:
    """Project original-coordinate points into normalized top-left screen coordinates."""

    values = np.asarray(points, dtype=float)
    if values.ndim != 2 or values.shape[1:] != (3,) or not np.all(np.isfinite(values)):
        raise ValueError("points must have finite shape (N, 3)")
    if (
        isinstance(width, bool)
        or isinstance(height, bool)
        or int(width) <= 0
        or int(height) <= 0
    ):
        raise ValueError("width and height must be positive")
    limits = _camera_bounds(bounds)
    span = limits[1] - limits[0]
    largest = max(float(np.max(span)), np.finfo(float).eps)
    target_value = _camera_field(
        camera,
        "target",
        (
            float(np.mean(limits[:, 0])),
            float(np.mean(limits[:, 1])),
            float(np.mean(limits[:, 2])),
        ),
    )
    target = np.asarray(target_value, dtype=float).reshape(3)
    ortho_height = float(_camera_field(camera, "ortho_height", 35.0))
    if not np.isfinite(ortho_height) or ortho_height <= 0.0:
        raise ValueError("camera ortho_height must be positive and finite")
    right, up, _ = camera_basis(camera)
    world_scale = 8.0 / largest
    relative = (values - target[None, :]) * world_scale
    magnification = ortho_height * min(int(width), int(height)) / 480.0
    return np.column_stack(
        (
            0.5 + relative @ right * magnification / float(width),
            0.5 - relative @ up * magnification / float(height),
        )
    )


def _camera_occupancy(
    vertices: np.ndarray,
    camera: Mapping[str, Any],
    bounds: np.ndarray,
    width: int,
    height: int,
) -> tuple[float, np.ndarray]:
    projected = project_points(vertices, camera, bounds, width, height)
    return float(np.max(np.ptp(projected, axis=0))) if len(
        projected
    ) else 0.0, projected


def fit_publication_camera(
    scene: Any, *, bounds: Any = None, width: int = 800, height: int = 600
) -> dict[str, Any]:
    """Fit a deterministic shallow orthographic camera for a publication panel."""

    source = _source_geometry(scene)
    limits = _camera_bounds(source["bounds"] if bounds is None else bounds)
    target = tuple(float(value) for value in np.mean(limits, axis=0))
    if len(source["vertices"]) == 0:
        return {
            "yaw": 0.0,
            "pitch": -10.0,
            "distance": 5.0,
            "ortho_height": 35.0,
            "target": target,
        }
    candidates = [
        (yaw, pitch)
        for yaw in (0.0, 8.0, -8.0, 18.0, -18.0, 35.0, -35.0)
        for pitch in (-12.0, -18.0, -24.0)
    ]
    best: tuple[float, dict[str, Any]] | None = None
    desired = 0.68
    tie_tolerance = 2.0e-3
    for yaw, pitch in candidates:
        provisional = {
            "yaw": yaw,
            "pitch": pitch,
            "ortho_height": 1.0,
            "target": target,
        }
        occupancy_at_one, _ = _camera_occupancy(
            source["vertices"].reshape((-1, 3)), provisional, limits, width, height
        )
        if occupancy_at_one <= 0.0:
            continue
        ortho = max(0.1, desired / occupancy_at_one)
        camera = {"yaw": yaw, "pitch": pitch, "ortho_height": ortho, "target": target}
        occupancy, projected = _camera_occupancy(
            source["vertices"].reshape((-1, 3)), camera, limits, width, height
        )
        score = abs(occupancy - desired)
        _, _, direction = camera_basis(camera)
        view_to_camera = -direction
        axis_cosines = np.abs(
            np.einsum("nij,j->ni", source["orientations"], view_to_camera)
        )
        ordered_cosines = np.sort(axis_cosines, axis=1)
        second_face_visibility = float(np.mean(ordered_cosines[:, -2]))
        solid_block_penalty = max(0.0, float(np.mean(ordered_cosines[:, -1])) - 0.84)
        score += 0.08 * max(0.0, 0.48 - second_face_visibility)
        score += 0.12 * solid_block_penalty
        plate_normal_visibility = np.abs(
            np.einsum("ni,i->n", source["orientations"][:, :, 2], view_to_camera)
        )
        normal_mean = float(np.mean(plate_normal_visibility))
        normal_max = float(np.max(plate_normal_visibility))
        score += 0.18 * abs(normal_mean - 0.225)
        score += 0.08 * max(0.0, normal_max - 0.65)
        score += 0.002 * abs(pitch + 18.0)
        branches = source["branch_ids"]
        if len(np.unique(branches)) > 1:
            centroids = [
                np.mean(projected[np.repeat(branches == branch, 8)], axis=0)
                for branch in np.unique(branches)
            ]
            if len(centroids) >= 2:
                gap = float(np.linalg.norm(centroids[0] - centroids[1]))
                score += max(0.0, 0.08 - gap) * 2.0
        projected_centers = project_points(
            source["centers"], camera, limits, width, height
        )
        stack_ids = source["stack_ids"]
        gaps: list[float] = []
        for index, center in enumerate(projected_centers):
            candidates_gap = [
                float(np.linalg.norm(center - projected_centers[other]))
                for other in range(len(projected_centers))
                if other != index and stack_ids[other] != stack_ids[index]
            ]
            if candidates_gap:
                gaps.append(min(candidates_gap))
        if gaps:
            score += 0.12 * abs(float(np.median(gaps)) - 0.075)
        candidate = {
            "yaw": yaw,
            "pitch": pitch,
            "distance": max(5.0, 5.0 * float(np.max(limits[1] - limits[0]))),
            "ortho_height": ortho,
            "target": target,
        }
        if best is None or score < best[0] - tie_tolerance:
            best = (score, candidate)
    if best is None:
        return {
            "yaw": 0.0,
            "pitch": -10.0,
            "distance": 5.0,
            "ortho_height": 35.0,
            "target": target,
        }
    return best[1]


__all__ = [
    "PublicationMesh",
    "build_publication_mesh",
    "camera_basis",
    "fit_publication_camera",
    "project_points",
]
