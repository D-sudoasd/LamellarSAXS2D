"""Bounded geometric fitting for labelled butterfly SAXS arcs.

The ordinary ellipse fitter is intentionally permissive about branch and
quadrant assignment.  Arc fitting is a different contract: each observed
point arrives with an already reviewed branch, side, and arc interval.  The
labels are therefore treated as supervision.  A malformed or unknown label is
excluded and is reported in the point diagnostics; it is never replaced by a
nearest-branch guess.

The optimizer and parameter representation are the canonical ones from
``butterfly_saxs.ellipse``.  Only the residual adapter is specific to this
module: it projects each point onto its assigned, bounded half ellipse and
keeps both endpoints in the candidate set.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from numbers import Integral, Real
from typing import Any

import numpy as np

from .cancellation import raise_if_cancelled
from .ellipse import (
    EllipseFitResult,
    EllipseGeometry,
    _closest_ellipse_distances,
    _coverage,
    _make_parameter_set,
    _multistart_parameter_sets,
    _run_fit,
    _validate_multistart_count,
    ellipse_implicit,
    ellipse_sampson_residuals,
)
from .parameters import ParameterSet


_TAU = 2.0 * math.pi
_HALF_PI = 0.5 * math.pi
_PROJECTION_GRID = 64
_FLAT_PROJECTION_GRID = 128


def _as_finite(value: Any) -> float | None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def _is_bool(value: Any) -> bool:
    return isinstance(value, (bool, np.bool_))


def _arc_interval(side: str, record: Mapping[str, Any]) -> tuple[float, float] | None:
    """Return the legacy single side-intersected manual interval."""

    intervals = _arc_intervals(side, record)
    return intervals[0] if intervals else None


def _arc_intervals(side: str, record: Mapping[str, Any]) -> list[tuple[float, float]] | None:
    """Return side-intersected explicit manual intervals.

    An absent interval is represented by ``None`` so the caller can require a
    frozen q-space support record.  This distinction is essential: an absent
    manual interval must never become the whole side ellipse.
    """

    side_lower, side_upper = (0.0, math.pi) if side == "upper" else (math.pi, _TAU)
    raw_intervals = record.get("arc_t_intervals", record.get("manual_t_intervals"))
    if raw_intervals is None:
        has_min = "arc_t_min" in record
        has_max = "arc_t_max" in record
        if has_min != has_max:
            return None
        if not has_min:
            return None
        raw_intervals = [(record.get("arc_t_min"), record.get("arc_t_max"))]
    elif isinstance(raw_intervals, Mapping):
        raw_intervals = [raw_intervals]
    if isinstance(raw_intervals, (str, bytes)) or not isinstance(raw_intervals, Sequence):
        return None
    normalized: list[tuple[float, float]] = []
    for raw_interval in raw_intervals:
        if isinstance(raw_interval, Mapping):
            raw_min = raw_interval.get("min", raw_interval.get("low", raw_interval.get("lower")))
            raw_max = raw_interval.get("max", raw_interval.get("high", raw_interval.get("upper")))
        elif isinstance(raw_interval, Sequence) and not isinstance(raw_interval, (str, bytes)) and len(raw_interval) == 2:
            raw_min, raw_max = raw_interval
        else:
            return None
        raw_min = _as_finite(raw_min)
        raw_max = _as_finite(raw_max)
        if raw_min is None or raw_max is None or raw_max < raw_min:
            return None
        midpoint = 0.5 * (raw_min + raw_max)
        side_midpoint = 0.5 * (side_lower + side_upper)
        shift = int(round((side_midpoint - midpoint) / _TAU))
        candidates = []
        for offset in (shift - 1, shift, shift + 1):
            lo = raw_min + offset * _TAU
            hi = raw_max + offset * _TAU
            overlap_lo = max(side_lower, lo)
            overlap_hi = min(side_upper, hi)
            if overlap_hi >= overlap_lo:
                candidates.append((overlap_lo, overlap_hi))
        if candidates:
            normalized.append(max(candidates, key=lambda item: (item[1] - item[0], -item[0])))
    if not normalized:
        return None
    normalized.sort(key=lambda item: (item[0], item[1]))
    merged: list[list[float]] = []
    for lo, hi in normalized:
        if merged and lo <= merged[-1][1] + 1.0e-12:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    return [(float(item[0]), float(item[1])) for item in merged]


def _support_records(record: Mapping[str, Any], support_registry: Mapping[Any, Any] | None = None) -> list[dict[str, Any]]:
    """Extract observed q-space rectangles from point-local JSON records."""

    candidates: Any = record.get("support_components", record.get("support_rectangles"))
    observed = record.get("observed_support", record.get("support"))
    if not candidates and isinstance(observed, Mapping):
        candidates = observed.get("components", observed.get("support_components"))
    if not candidates and support_registry is not None:
        arc_id = record.get("arc_id")
        source = support_registry.get(arc_id, support_registry.get(str(arc_id)))
        if source is None:
            nested_registry = support_registry.get("arc_support", support_registry.get("support_registry"))
            if isinstance(nested_registry, Mapping):
                source = nested_registry.get(arc_id, nested_registry.get(str(arc_id)))
        if isinstance(source, Mapping):
            candidates = source.get("support_components", source.get("components"))
            if not observed:
                observed = source
    if isinstance(observed, Mapping):
        status = observed.get("status", observed.get("support_status"))
        if status not in (None, "frozen"):
            return []
    if isinstance(candidates, Mapping):
        candidates = [candidates]
    if isinstance(candidates, (str, bytes)) or not isinstance(candidates, Sequence):
        return []
    rectangles: list[dict[str, Any]] = []
    for component_index, component in enumerate(candidates):
        if not isinstance(component, Mapping):
            continue
        nested = component.get("rectangles", component.get("segments"))
        if isinstance(nested, Mapping):
            nested = [nested]
        if isinstance(nested, Sequence) and not isinstance(nested, (str, bytes)):
            values = nested
        else:
            values = [component]
        for segment_index, raw in enumerate(values):
            if not isinstance(raw, Mapping):
                continue
            origin = raw.get("frame_origin_q", raw.get("origin_q"))
            tangent = raw.get("tangent_q")
            normal = raw.get("normal_q")
            t_bounds = raw.get("tangent_bounds")
            n_bounds = raw.get("normal_bounds")
            if origin is None or tangent is None or normal is None:
                continue
            if t_bounds is None:
                t_bounds = (raw.get("tangent_min"), raw.get("tangent_max"))
            if n_bounds is None:
                n_bounds = (raw.get("normal_min"), raw.get("normal_max"))
            if isinstance(t_bounds, Mapping):
                t_bounds = (t_bounds.get("min", t_bounds.get("low", t_bounds.get("lower"))),
                            t_bounds.get("max", t_bounds.get("high", t_bounds.get("upper"))))
            if isinstance(n_bounds, Mapping):
                n_bounds = (n_bounds.get("min", n_bounds.get("low", n_bounds.get("lower"))),
                            n_bounds.get("max", n_bounds.get("high", n_bounds.get("upper"))))
            try:
                origin = [float(origin[0]), float(origin[1])]
                tangent = np.asarray([float(tangent[0]), float(tangent[1])], dtype=float)
                normal = np.asarray([float(normal[0]), float(normal[1])], dtype=float)
                t_bounds = [float(t_bounds[0]), float(t_bounds[1])]
                n_bounds = [float(n_bounds[0]), float(n_bounds[1])]
            except (TypeError, ValueError, IndexError):
                continue
            if not all(np.isfinite(origin)) or not np.all(np.isfinite(tangent)) or not np.all(np.isfinite(normal)):
                continue
            if not all(np.isfinite(t_bounds)) or not all(np.isfinite(n_bounds)) or t_bounds[1] < t_bounds[0] or n_bounds[1] < n_bounds[0]:
                continue
            tangent_norm = float(np.linalg.norm(tangent))
            normal_norm = float(np.linalg.norm(normal))
            if tangent_norm <= 1.0e-14 or normal_norm <= 1.0e-14:
                continue
            tangent /= tangent_norm
            normal /= normal_norm
            rectangles.append(
                {
                    "component_id": raw.get("component_id", component.get("component_id", component_index)),
                    "segment_index": raw.get("segment_index", segment_index),
                    "frame_origin_q": origin,
                    "tangent_q": tangent.tolist(),
                    "normal_q": normal.tolist(),
                    "tangent_bounds": t_bounds,
                    "normal_bounds": n_bounds,
                    "point_ids": list(raw.get("point_ids", component.get("point_ids", [])))
                    if isinstance(raw.get("point_ids", component.get("point_ids", [])), Sequence)
                    and not isinstance(raw.get("point_ids", component.get("point_ids", [])), (str, bytes))
                    else [],
                    "endpoint_flags": deepcopy(raw.get("endpoint_flags", component.get("endpoint_flags", {}))),
                }
            )
    return rectangles


def _q_inside_rectangle(point: Sequence[float], rectangle: Mapping[str, Any]) -> bool:
    try:
        q = np.asarray([float(point[0]), float(point[1])], dtype=float)
        origin = np.asarray(rectangle["frame_origin_q"], dtype=float)
        tangent = np.asarray(rectangle["tangent_q"], dtype=float)
        normal = np.asarray(rectangle["normal_q"], dtype=float)
        values = (float(np.dot(tangent, q - origin)), float(np.dot(normal, q - origin)))
        scale = max(1.0, float(np.linalg.norm(q)), *(abs(value) for value in values))
        tolerance = 2.0e-10 * scale
        return (
            rectangle["tangent_bounds"][0] - tolerance <= values[0] <= rectangle["tangent_bounds"][1] + tolerance
            and rectangle["normal_bounds"][0] - tolerance <= values[1] <= rectangle["normal_bounds"][1] + tolerance
        )
    except (TypeError, ValueError, KeyError, IndexError):
        return False


def _q_rectangle_violation(point: Sequence[float], rectangles: Sequence[Mapping[str, Any]]) -> float | None:
    distances: list[float] = []
    try:
        q = np.asarray([float(point[0]), float(point[1])], dtype=float)
        for rectangle in rectangles:
            origin = np.asarray(rectangle["frame_origin_q"], dtype=float)
            tangent = np.asarray(rectangle["tangent_q"], dtype=float)
            normal = np.asarray(rectangle["normal_q"], dtype=float)
            values = (float(np.dot(tangent, q - origin)), float(np.dot(normal, q - origin)))
            delta_t = max(rectangle["tangent_bounds"][0] - values[0], 0.0, values[0] - rectangle["tangent_bounds"][1])
            delta_n = max(rectangle["normal_bounds"][0] - values[1], 0.0, values[1] - rectangle["normal_bounds"][1])
            distances.append(float(math.hypot(delta_t, delta_n)))
    except (TypeError, ValueError, KeyError, IndexError):
        return None
    finite = [distance for distance in distances if np.isfinite(distance)]
    return float(min(finite)) if finite else None


def _parse_points(
    points: Sequence[Mapping[str, Any]],
    *,
    support_registry: Mapping[Any, Any] | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[dict[str, Any]],
    dict[int, dict[str, Any]],
    list[dict[str, Any]],
]:
    """Validate labels and require manual bounds or frozen observed support."""

    if isinstance(points, (str, bytes)) or not isinstance(points, Sequence):
        raise TypeError("points must be a sequence of point mappings")

    point_diagnostics: list[dict[str, Any]] = []
    parsed_rows: list[dict[str, Any]] = []
    arc_records: dict[int, dict[str, Any]] = {}
    for index, raw in enumerate(points):
        diagnostic: dict[str, Any] = {
            "index": int(index),
            "point_id": int(index),
            "used": False,
            "accepted": False,
            "excluded": True,
            "valid": False,
            "excluded_reason": None,
        }
        reason: str | None = None
        row: dict[str, Any] | None = None
        if not isinstance(raw, Mapping):
            reason = "malformed_point"
        else:
            diagnostic["point_id"] = raw.get("point_id", index)
            qx = _as_finite(raw.get("qx"))
            qy = _as_finite(raw.get("qy"))
            if qx is not None:
                diagnostic["qx"] = qx
            if qy is not None:
                diagnostic["qy"] = qy
            branch = raw.get("branch_id")
            side = raw.get("side")
            arc_id = raw.get("arc_id")
            accepted = raw.get("accepted")
            sigma_raw = raw.get("localization_sigma_q")
            sigma = _as_finite(sigma_raw)
            if sigma is None:
                # Profile refinement may have a finite sampling scale while
                # the curve-fit localization estimate is unavailable.  Use
                # that measured q scale for optimizer weighting; support
                # construction records the same provenance independently.
                sigma = _as_finite(raw.get("sampling_sigma_q"))
                if sigma is None:
                    step = _as_finite(raw.get("q_normal_step", raw.get("normal_step_q")))
                    sigma = step / math.sqrt(12.0) if step is not None and step > 0 else None
                if sigma is None and "localization_sigma_q" not in raw:
                    sigma = 1.0
            if qx is None or qy is None:
                reason = "malformed_coordinates"
            elif not _is_bool(accepted):
                reason = "malformed_accepted"
            elif not bool(accepted):
                reason = "not_accepted"
            elif isinstance(branch, (bool, np.bool_)) or not isinstance(branch, Integral) or int(branch) not in (0, 1):
                reason = "malformed_branch_id"
            elif side not in ("upper", "lower"):
                reason = "unknown_side"
            elif isinstance(arc_id, (bool, np.bool_)) or not isinstance(arc_id, Integral):
                reason = "malformed_arc_id"
            elif sigma is None or sigma <= 0.0:
                reason = "malformed_localization_sigma_q"
            else:
                manual_intervals = _arc_intervals(str(side), raw)
                manual_keys = {"arc_t_min", "arc_t_max", "arc_t_intervals", "manual_t_intervals"}
                manual_supplied = bool(manual_keys.intersection(raw))
                if manual_supplied and manual_intervals is None:
                    reason = "malformed_or_disjoint_arc_interval"
                else:
                    rectangles = _support_records(raw, support_registry)
                    support_point_id = raw.get("point_id")
                    support_point_ids = {
                        str(point_id)
                        for rectangle in rectangles
                        for point_id in rectangle.get("point_ids", ())
                    }
                    support_status = raw.get("support_status")
                    if support_status is None and isinstance(raw.get("observed_support"), Mapping):
                        support_status = raw["observed_support"].get("status", raw["observed_support"].get("support_status"))
                    if support_status is None and rectangles:
                        support_status = "frozen"
                    source_q = (qx, qy)
                    if not manual_supplied and rectangles and support_point_ids and str(support_point_id) not in support_point_ids:
                        reason = "observed_support_point_identity_mismatch"
                    elif rectangles and not any(_q_inside_rectangle(source_q, rectangle) for rectangle in rectangles):
                        reason = "observed_support_point_outside_corridor"
                        diagnostic["support_violation_q"] = _q_rectangle_violation(source_q, rectangles)
                    elif not manual_supplied and not rectangles:
                        reason = "observed_support_unavailable"
                    elif support_status not in (None, "frozen") and not manual_supplied:
                        reason = f"observed_support_{support_status}"
                    else:
                        row = {
                            "qx": float(qx),
                            "qy": float(qy),
                            "branch": int(branch),
                            "side": str(side),
                            "arc_id": int(arc_id),
                            "sigma": float(sigma),
                            "manual_intervals": manual_intervals,
                            "manual_supplied": manual_supplied,
                            "rectangles": rectangles,
                            "support_status": support_status or ("manual_only" if manual_supplied else "frozen"),
                            "support_component_count": int(raw.get("support_component_count", len({str(rectangle.get("component_id")) for rectangle in rectangles}))),
                            "support_gap_count": int(raw.get("support_gap_count", (raw.get("observed_support", {}) or {}).get("gap_count", 0) if isinstance(raw.get("observed_support"), Mapping) else 0)),
                            "original_index": index,
                        }
        if row is None:
            diagnostic["excluded_reason"] = reason or "malformed_point"
            diagnostic["reason"] = diagnostic["excluded_reason"]
            if isinstance(raw, Mapping):
                if _is_bool(raw.get("accepted")):
                    diagnostic["accepted"] = bool(raw["accepted"])
                if isinstance(raw.get("branch_id"), Integral) and not isinstance(raw.get("branch_id"), (bool, np.bool_)):
                    diagnostic["branch_id"] = int(raw["branch_id"])
                if raw.get("side") in ("upper", "lower", "unknown"):
                    diagnostic["side"] = raw.get("side")
                if isinstance(raw.get("arc_id"), Integral) and not isinstance(raw.get("arc_id"), (bool, np.bool_)):
                    diagnostic["arc_id"] = int(raw["arc_id"])
            point_diagnostics.append(diagnostic)
            continue
        diagnostic.update(
            {
                "accepted": True,
                "used": True,
                "excluded": False,
                "valid": True,
                "qx": row["qx"],
                "qy": row["qy"],
                "branch_id": row["branch"],
                "side": row["side"],
                "arc_id": row["arc_id"],
                "localization_sigma_q": row["sigma"],
                "support_status": row["support_status"],
                "support_component_count": row["support_component_count"],
                "support_gap_count": row["support_gap_count"],
                "support_padding_q": _as_finite(raw.get("support_padding_q")) if isinstance(raw, Mapping) else None,
            }
        )
        if isinstance(raw, Mapping):
            center = raw.get("center_q")
            cx = _as_finite(raw.get("center_qx"))
            cy = _as_finite(raw.get("center_qy"))
            if isinstance(center, Sequence) and not isinstance(center, (str, bytes)) and len(center) == 2:
                cx = _as_finite(center[0])
                cy = _as_finite(center[1])
            if cx is not None and cy is not None:
                diagnostic["center_qx"] = cx
                diagnostic["center_qy"] = cy
        if row["manual_intervals"]:
            diagnostic["t_intervals"] = [tuple(map(float, interval)) for interval in row["manual_intervals"]]
            diagnostic["t_interval"] = diagnostic["t_intervals"][0] if len(diagnostic["t_intervals"]) == 1 else None
        else:
            diagnostic["t_intervals"] = None
            diagnostic["t_interval"] = None
        point_diagnostics.append(diagnostic)
        parsed_rows.append(row)
        arc = arc_records.setdefault(
            row["arc_id"],
            {
                "arc_id": row["arc_id"],
                "branch_id": row["branch"],
                "side": row["side"],
                "n_points": 0,
                "n_used": 0,
                "n_excluded": 0,
                "metadata_consistent": True,
                "support_status": row["support_status"],
                "support_components": deepcopy(row["rectangles"]),
                "support_component_count": row["support_component_count"],
                "support_gap_count": row["support_gap_count"],
                "support_gaps": [],
            },
        )
        if arc["branch_id"] != row["branch"] or arc["side"] != row["side"]:
            arc["metadata_consistent"] = False
            arc["support_status"] = "ineligible_mixed_labels"
        arc["n_points"] += 1
        arc["n_used"] += 1

    # A source arc is one fixed branch/side identity.  Do not let a mixed
    # component contaminate a labelled fit by dropping individual labels.
    # Include accepted rows that were rejected for missing support: their
    # topology is still evidence and must be able to invalidate the arc.
    for diagnostic in point_diagnostics:
        if not diagnostic.get("accepted"):
            continue
        arc_id = diagnostic.get("arc_id")
        branch = diagnostic.get("branch_id")
        side = diagnostic.get("side")
        if not isinstance(arc_id, Integral) or isinstance(arc_id, (bool, np.bool_)):
            continue
        arc = arc_records.setdefault(
            int(arc_id),
            {
                "arc_id": int(arc_id),
                "branch_id": branch,
                "side": side,
                "n_points": 0,
                "n_used": 0,
                "n_excluded": 0,
                "metadata_consistent": True,
            },
        )
        if arc.get("branch_id") != branch or arc.get("side") != side:
            arc["metadata_consistent"] = False
            arc["support_status"] = "ineligible_mixed_labels"
    mixed_arc_ids = {
        int(arc_id)
        for arc_id, arc in arc_records.items()
        if not bool(arc.get("metadata_consistent", True))
    }
    if mixed_arc_ids:
        kept_rows: list[dict[str, Any]] = []
        for row in parsed_rows:
            if int(row["arc_id"]) not in mixed_arc_ids:
                kept_rows.append(row)
                continue
            diagnostic = point_diagnostics[int(row["original_index"])]
            diagnostic.update(
                {
                    "used": False,
                    "excluded": True,
                    "valid": False,
                    "excluded_reason": "mixed_arc_metadata",
                    "reason": "mixed_arc_metadata",
                }
            )
        parsed_rows = kept_rows
    for diagnostic in point_diagnostics:
        arc_id = diagnostic.get("arc_id")
        if isinstance(arc_id, Integral) and not isinstance(arc_id, (bool, np.bool_)) and int(arc_id) in mixed_arc_ids and diagnostic.get("accepted"):
            diagnostic.update(
                {
                    "used": False,
                    "excluded": True,
                    "valid": False,
                    "excluded_reason": "mixed_arc_metadata",
                    "reason": "mixed_arc_metadata",
                }
            )
        if diagnostic.get("excluded") and isinstance(arc_id, Integral) and not isinstance(arc_id, (bool, np.bool_)):
            arc = arc_records.setdefault(
                int(arc_id),
                {
                    "arc_id": int(arc_id),
                    "branch_id": diagnostic.get("branch_id"),
                    "side": diagnostic.get("side"),
                    "n_points": 0,
                    "n_used": 0,
                    "n_excluded": 0,
                    "metadata_consistent": True,
                },
            )
            arc["n_points"] += 1
            arc["n_excluded"] += 1
    for arc_id, arc in arc_records.items():
        members = [item for item in point_diagnostics if item.get("arc_id") == arc_id]
        arc["n_points"] = len(members)
        arc["n_used"] = sum(bool(item.get("used")) for item in members)
        arc["n_excluded"] = sum(bool(item.get("excluded")) for item in members)
    for row_index, row in enumerate(parsed_rows):
        diagnostic = point_diagnostics[int(row["original_index"])]
        diagnostic["valid_row"] = row_index
        diagnostic["used"] = True
        diagnostic["excluded"] = False
        diagnostic["valid"] = True

    xy = np.asarray([(row["qx"], row["qy"]) for row in parsed_rows], dtype=float) if parsed_rows else np.empty((0, 2), dtype=float)
    return (
        xy,
        np.asarray([row["branch"] for row in parsed_rows], dtype=int),
        np.asarray([row["side"] for row in parsed_rows], dtype=object),
        np.asarray([row["arc_id"] for row in parsed_rows], dtype=int),
        np.asarray([row["sigma"] for row in parsed_rows], dtype=float),
        point_diagnostics,
        arc_records,
        parsed_rows,
    )


def _local_coordinates(points: np.ndarray, geometry: EllipseGeometry) -> tuple[np.ndarray, np.ndarray]:
    dx = points[:, 0] - geometry.cx
    dy = points[:, 1] - geometry.cy
    cosine = math.cos(geometry.theta)
    sine = math.sin(geometry.theta)
    return cosine * dx + sine * dy, -sine * dx + cosine * dy


def _project_ellipse_arc(
    u: np.ndarray,
    v: np.ndarray,
    a: float,
    b: float,
    t_min: np.ndarray,
    t_max: np.ndarray,
    *,
    include_global_oracle: bool = False,
) -> dict[str, np.ndarray]:
    """Vectorized closest-point projection onto independent bounded arcs.

    Coarse candidates and both interval endpoints are evaluated in one array
    operation.  A clipped Newton refinement then improves the selected basin;
    no per-point scalar minimizer is used.  The optional global vectorized
    ellipse distance is recorded only for final diagnostics as an independent
    oracle for very flat curves, keeping repeated objective evaluations fast.
    """

    u, v, t_min, t_max = np.broadcast_arrays(
        np.asarray(u, dtype=float),
        np.asarray(v, dtype=float),
        np.asarray(t_min, dtype=float),
        np.asarray(t_max, dtype=float),
    )
    original_shape = u.shape
    u = u.ravel()
    v = v.ravel()
    t_min = t_min.ravel()
    t_max = t_max.ravel()
    width = np.maximum(0.0, t_max - t_min)
    ratio = float(b / a)
    grid_size = _FLAT_PROJECTION_GRID if ratio < 0.08 else _PROJECTION_GRID
    fraction = np.linspace(0.0, 1.0, grid_size + 1, dtype=float)
    grid = t_min[:, None] + width[:, None] * fraction[None, :]
    cosine = np.cos(grid)
    sine = np.sin(grid)
    squared = (a * cosine - u[:, None]) ** 2 + (b * sine - v[:, None]) ** 2
    selected = np.argmin(squared, axis=1)
    rows = np.arange(u.size)
    t = grid[rows, selected].copy()

    # Include endpoints explicitly even after refinement.  This is essential
    # when the unconstrained closest point lies outside a short observed arc.
    endpoint_left = (a * np.cos(t_min) - u) ** 2 + (b * np.sin(t_min) - v) ** 2
    endpoint_right = (a * np.cos(t_max) - u) ** 2 + (b * np.sin(t_max) - v) ** 2

    for _ in range(16):
        sine_t = np.sin(t)
        cosine_t = np.cos(t)
        derivative = (b * b - a * a) * sine_t * cosine_t + a * u * sine_t - b * v * cosine_t
        second = (b * b - a * a) * (cosine_t * cosine_t - sine_t * sine_t) + a * u * cosine_t + b * v * sine_t
        step = np.divide(derivative, second, out=np.zeros_like(derivative), where=np.abs(second) > 1.0e-14)
        candidate = np.clip(t - step, t_min, t_max)
        current = (a * cosine_t - u) ** 2 + (b * sine_t - v) ** 2
        candidate_squared = (a * np.cos(candidate) - u) ** 2 + (b * np.sin(candidate) - v) ** 2
        improve = np.isfinite(candidate_squared) & (candidate_squared <= current)
        new_t = np.where(improve, candidate, t)
        if np.all(np.abs(new_t - t) <= 1.0e-13):
            t = new_t
            break
        t = new_t

    squared_refined = (a * np.cos(t) - u) ** 2 + (b * np.sin(t) - v) ** 2
    choose_left = endpoint_left < squared_refined
    t = np.where(choose_left, t_min, t)
    squared_refined = np.where(choose_left, endpoint_left, squared_refined)
    choose_right = endpoint_right < squared_refined
    t = np.where(choose_right, t_max, t)
    squared_refined = np.where(choose_right, endpoint_right, squared_refined)
    distance = np.sqrt(np.maximum(0.0, squared_refined))
    if include_global_oracle:
        global_distance = _closest_ellipse_distances(u, v, a, b).ravel()
    else:
        global_distance = np.full(distance.shape, np.nan, dtype=float)
    at_endpoint = np.minimum(np.abs(t - t_min), np.abs(t - t_max)) <= 2.0e-9
    endpoint_clipped = np.zeros(distance.shape, dtype=bool)
    if include_global_oracle:
        endpoint_clipped = at_endpoint & (
            distance > global_distance + 1.0e-9 * np.maximum(1.0, np.maximum(distance, global_distance))
        )
    return {
        "t": t.reshape(original_shape),
        "distance": distance.reshape(original_shape),
        "global_distance": global_distance.reshape(original_shape),
        "at_endpoint": at_endpoint.reshape(original_shape),
        "endpoint_clipped": endpoint_clipped.reshape(original_shape),
    }


def _side_domain(side: str) -> tuple[float, float]:
    return (0.0, math.pi) if side == "upper" else (math.pi, _TAU)


def _ellipse_linear_coefficients(
    geometry: EllipseGeometry,
    tangent: np.ndarray,
    origin: np.ndarray,
) -> tuple[float, float, float]:
    """Return ``base, A, B`` for ``w·(P(t)-origin)=base+A cos(t)+B sin(t)``."""

    cosine, sine = math.cos(geometry.theta), math.sin(geometry.theta)
    base = float(np.dot(tangent, np.asarray([geometry.cx, geometry.cy], dtype=float) - origin))
    coefficient_cos = float(geometry.a * (tangent[0] * cosine + tangent[1] * sine))
    coefficient_sin = float(geometry.b * (-tangent[0] * sine + tangent[1] * cosine))
    return base, coefficient_cos, coefficient_sin


def _sinusoid_roots(a_coefficient: float, b_coefficient: float, target: float, lo: float, hi: float) -> list[float]:
    """Solve ``A*cos(t)+B*sin(t)=target`` on a bounded interval."""

    rho = float(math.hypot(a_coefficient, b_coefficient))
    scale = max(1.0, abs(target), rho)
    tolerance = 2.0e-12 * scale
    if rho <= 1.0e-15:
        return []
    ratio = float(target / rho)
    if ratio > 1.0 + tolerance or ratio < -1.0 - tolerance:
        return []
    ratio = float(np.clip(ratio, -1.0, 1.0))
    phase = math.atan2(b_coefficient, a_coefficient)
    offset = math.acos(ratio)
    roots: list[float] = []
    # The side domain is at most one full turn, but using a small explicit
    # integer range also handles manually unwrapped intervals safely.
    k_min = int(math.floor((lo - phase - offset) / _TAU)) - 1
    k_max = int(math.ceil((hi - phase + offset) / _TAU)) + 1
    for k in range(k_min, k_max + 1):
        for root in (phase + offset + k * _TAU, phase - offset + k * _TAU):
            if lo - 2.0e-12 <= root <= hi + 2.0e-12:
                roots.append(float(np.clip(root, lo, hi)))
    roots.sort()
    unique: list[float] = []
    for root in roots:
        if not unique or abs(root - unique[-1]) > 2.0e-11:
            unique.append(root)
    return unique


def _rectangle_value(
    geometry: EllipseGeometry,
    t: float,
    vector: np.ndarray,
    origin: np.ndarray,
) -> float:
    return float(np.dot(vector, geometry.point(float(t)) - origin))


def _rectangle_contains(
    geometry: EllipseGeometry,
    t: float,
    rectangle: Mapping[str, Any],
) -> bool:
    origin = np.asarray(rectangle["frame_origin_q"], dtype=float)
    tangent = np.asarray(rectangle["tangent_q"], dtype=float)
    normal = np.asarray(rectangle["normal_q"], dtype=float)
    tangent_bounds = rectangle["tangent_bounds"]
    normal_bounds = rectangle["normal_bounds"]
    values = (
        _rectangle_value(geometry, t, tangent, origin),
        _rectangle_value(geometry, t, normal, origin),
    )
    scale = max(1.0, abs(geometry.a), abs(geometry.b), *(abs(value) for value in values))
    tolerance = 2.0e-10 * scale
    return (
        tangent_bounds[0] - tolerance <= values[0] <= tangent_bounds[1] + tolerance
        and normal_bounds[0] - tolerance <= values[1] <= normal_bounds[1] + tolerance
    )


def _rectangle_at_boundary(
    geometry: EllipseGeometry,
    t: float,
    rectangle: Mapping[str, Any],
) -> bool:
    origin = np.asarray(rectangle["frame_origin_q"], dtype=float)
    scale = max(1.0, abs(geometry.a), abs(geometry.b))
    tolerance = 2.0e-9 * scale
    for vector, bounds in (
        (np.asarray(rectangle["tangent_q"], dtype=float), rectangle["tangent_bounds"]),
        (np.asarray(rectangle["normal_q"], dtype=float), rectangle["normal_bounds"]),
    ):
        value = _rectangle_value(geometry, t, vector, origin)
        if abs(value - bounds[0]) <= tolerance or abs(value - bounds[1]) <= tolerance:
            return True
    return False


def _rectangle_intervals(
    geometry: EllipseGeometry,
    rectangle: Mapping[str, Any],
    domain_intervals: Sequence[tuple[float, float]],
) -> list[dict[str, Any]]:
    """Intersect one observed rectangle and manual/domain intervals analytically."""

    origin = np.asarray(rectangle["frame_origin_q"], dtype=float)
    axes = (
        (np.asarray(rectangle["tangent_q"], dtype=float), rectangle["tangent_bounds"]),
        (np.asarray(rectangle["normal_q"], dtype=float), rectangle["normal_bounds"]),
    )
    intervals: list[dict[str, Any]] = []
    for domain_lo, domain_hi in domain_intervals:
        if domain_hi < domain_lo:
            continue
        boundaries = [float(domain_lo), float(domain_hi)]
        boundary_tags: dict[float, set[str]] = {
            float(domain_lo): {"manual"},
            float(domain_hi): {"manual"},
        }
        for axis_index, (axis, bounds) in enumerate(axes):
            base, coefficient_cos, coefficient_sin = _ellipse_linear_coefficients(geometry, axis, origin)
            for bound_index, bound in enumerate(bounds):
                for root in _sinusoid_roots(coefficient_cos, coefficient_sin, float(bound) - base, domain_lo, domain_hi):
                    boundaries.append(root)
                    boundary_tags.setdefault(root, set()).add(f"support_{axis_index}_{bound_index}")
        boundaries = sorted(boundaries)
        unique: list[float] = []
        for value in boundaries:
            if not unique or abs(value - unique[-1]) > 2.0e-11:
                unique.append(value)
            else:
                boundary_tags[unique[-1]].update(boundary_tags.get(value, set()))
        for left, right in zip(unique[:-1], unique[1:]):
            midpoint = 0.5 * (left + right)
            if _rectangle_contains(geometry, midpoint, rectangle):
                tags_left = boundary_tags.get(left, set())
                tags_right = boundary_tags.get(right, set())
                intervals.append(
                    {
                        "lo": float(left),
                        "hi": float(right),
                        "component_id": rectangle.get("component_id"),
                        "support_interval_index": rectangle.get("segment_index", 0),
                        "lo_support": any(tag.startswith("support_") for tag in tags_left),
                        "hi_support": any(tag.startswith("support_") for tag in tags_right),
                        "lo_manual": "manual" in tags_left,
                        "hi_manual": "manual" in tags_right,
                    }
                )
        # Preserve an isolated tangential contact, which has no positive-width
        # midpoint basin but is still an observed endpoint/root.
        for value in unique:
            if _rectangle_contains(geometry, value, rectangle):
                tags = boundary_tags.get(value, set())
                intervals.append(
                    {
                        "lo": float(value),
                        "hi": float(value),
                        "component_id": rectangle.get("component_id"),
                        "support_interval_index": rectangle.get("segment_index", 0),
                        "lo_support": any(tag.startswith("support_") for tag in tags),
                        "hi_support": any(tag.startswith("support_") for tag in tags),
                        "lo_manual": "manual" in tags,
                        "hi_manual": "manual" in tags,
                    }
                )
    # Merge only overlapping intervals from this same rectangle.  The caller
    # never merges intervals from distinct support components, so a measured
    # q-space gap cannot be bridged accidentally.
    intervals.sort(key=lambda item: (item["lo"], item["hi"]))
    merged: list[dict[str, Any]] = []
    for item in intervals:
        if merged and item["lo"] <= merged[-1]["hi"] + 2.0e-11:
            merged[-1]["hi"] = max(merged[-1]["hi"], item["hi"])
            merged[-1]["hi_support"] = bool(merged[-1]["hi_support"] or item["hi_support"])
            merged[-1]["hi_manual"] = bool(merged[-1]["hi_manual"] or item["hi_manual"])
        else:
            merged.append(dict(item))
    return merged


def _support_violation_q(
    point: np.ndarray,
    geometry: EllipseGeometry,
    rectangles: Sequence[Mapping[str, Any]],
    domain_intervals: Sequence[tuple[float, float]],
) -> float:
    """Finite deterministic penalty when no curve/support intersection exists."""

    candidates: list[np.ndarray] = []
    # Rectangle-boundary sinusoid roots are the primary penalty candidates.
    for rectangle in rectangles:
        origin = np.asarray(rectangle["frame_origin_q"], dtype=float)
        for axis_name, axis, bounds in (
            ("tangent", np.asarray(rectangle["tangent_q"], dtype=float), rectangle["tangent_bounds"]),
            ("normal", np.asarray(rectangle["normal_q"], dtype=float), rectangle["normal_bounds"]),
        ):
            base, coefficient_cos, coefficient_sin = _ellipse_linear_coefficients(geometry, axis, origin)
            for bound in bounds:
                for domain_lo, domain_hi in domain_intervals:
                    for root in _sinusoid_roots(coefficient_cos, coefficient_sin, float(bound) - base, domain_lo, domain_hi):
                        candidates.append(np.asarray(geometry.point(root), dtype=float).reshape(2))
    # Domain endpoints and a fixed fallback grid keep the penalty finite when
    # an ellipse misses every rectangle boundary entirely.
    for domain_lo, domain_hi in domain_intervals:
        candidates.extend(
            [
                np.asarray(geometry.point(domain_lo), dtype=float).reshape(2),
                np.asarray(geometry.point(domain_hi), dtype=float).reshape(2),
            ]
        )
        if domain_hi > domain_lo:
            for value in np.linspace(domain_lo, domain_hi, 33):
                candidates.append(np.asarray(geometry.point(value), dtype=float).reshape(2))
    if not candidates:
        return float(max(geometry.a, geometry.b, 1.0))
    distances = np.asarray([np.linalg.norm(candidate - point) for candidate in candidates], dtype=float)
    finite = distances[np.isfinite(distances)]
    return float(max(np.min(finite) if finite.size else max(geometry.a, geometry.b, 1.0), 1.0e-12))


def _project_point_to_support(
    point: np.ndarray,
    geometry: EllipseGeometry,
    side: str,
    manual_intervals: Sequence[tuple[float, float]] | None,
    rectangles: Sequence[Mapping[str, Any]],
    *,
    include_global_oracle: bool = False,
) -> dict[str, Any]:
    domain = list(manual_intervals) if manual_intervals is not None else [ _side_domain(side) ]
    if manual_intervals is not None and not manual_intervals:
        domain = []
    candidates: list[dict[str, Any]] = []
    local_candidates: list[tuple[float, float, dict[str, Any]]] = []
    if rectangles:
        # Fast path for the common case: the unconstrained closest point in a
        # manual/side domain already lies inside an observed rectangle.  A
        # point that is feasible for the global domain is necessarily the
        # constrained optimum, so no sinusoid boundary enumeration is needed.
        u, v = _local_coordinates(np.asarray([point], dtype=float), geometry)
        unconstrained: list[dict[str, Any]] = []
        for domain_index, (domain_lo, domain_hi) in enumerate(domain):
            raw_projection = _project_ellipse_arc(
                u,
                v,
                geometry.a,
                geometry.b,
                np.asarray([domain_lo]),
                np.asarray([domain_hi]),
                include_global_oracle=include_global_oracle,
            )
            unconstrained.append(
                {
                    "t": float(raw_projection["t"][0]),
                    "distance": float(raw_projection["distance"][0]),
                    "global_distance": float(raw_projection["global_distance"][0]),
                    "at_endpoint": bool(raw_projection["at_endpoint"][0]),
                    "endpoint_clipped": bool(raw_projection["endpoint_clipped"][0]),
                    "domain_index": domain_index,
                }
            )
        for candidate in sorted(unconstrained, key=lambda item: (item["distance"], item["t"])):
            for rectangle in rectangles:
                if _rectangle_contains(geometry, candidate["t"], rectangle):
                    return {
                        "projection_valid": True,
                        "t": candidate["t"],
                        "distance": candidate["distance"],
                        "global_distance": candidate["global_distance"],
                        "at_endpoint": candidate["at_endpoint"],
                        "endpoint_clipped": candidate["endpoint_clipped"],
                        "support_component_id": rectangle.get("component_id"),
                        "support_interval_index": int(rectangle.get("segment_index", 0)),
                        "support_endpoint_clipped": _rectangle_at_boundary(geometry, candidate["t"], rectangle),
                        "manual_bound_clipped": candidate["at_endpoint"],
                        "support_infeasible": False,
                        "support_violation_q": 0.0,
                        "objective_distance": candidate["distance"],
                    }
        for rectangle in rectangles:
            for item in _rectangle_intervals(geometry, rectangle, domain):
                local_candidates.append((float(item["lo"]), float(item["hi"]), item))
    elif manual_intervals is not None:
        for index, (lo, hi) in enumerate(domain):
            local_candidates.append(
                (
                    float(lo),
                    float(hi),
                    {
                        "component_id": None,
                        "support_interval_index": index,
                        "lo_support": False,
                        "hi_support": False,
                        "lo_manual": True,
                        "hi_manual": True,
                    },
                )
            )
    for lo, hi, metadata in local_candidates:
        # The helper above works in a branch-local frame.  Convert the observed
        # q point to that frame before evaluating it.
        u, v = _local_coordinates(np.asarray([point], dtype=float), geometry)
        projection = _project_ellipse_arc(
            u,
            v,
            geometry.a,
            geometry.b,
            np.asarray([lo]),
            np.asarray([hi]),
            include_global_oracle=include_global_oracle,
        )
        candidates.append(
            {
                "t": float(projection["t"][0]),
                "distance": float(projection["distance"][0]),
                "global_distance": float(projection["global_distance"][0]),
                "at_endpoint": bool(projection["at_endpoint"][0]),
                "endpoint_clipped": bool(projection["endpoint_clipped"][0]),
                "metadata": metadata,
            }
        )
    if not candidates:
        penalty = _support_violation_q(point, geometry, rectangles, domain)
        return {
            "projection_valid": False,
            "t": float("nan"),
            "distance": float("nan"),
            "global_distance": float("nan"),
            "at_endpoint": False,
            "endpoint_clipped": False,
            "support_component_id": None,
            "support_interval_index": -1,
            "support_endpoint_clipped": False,
            "manual_bound_clipped": False,
            "support_infeasible": True,
            "support_violation_q": float(penalty),
            "objective_distance": float(penalty),
        }
    selected = min(candidates, key=lambda item: (item["distance"], item["t"], str(item["metadata"].get("component_id"))))
    metadata = selected["metadata"]
    support_endpoint = bool(selected["at_endpoint"] and (metadata.get("lo_support") or metadata.get("hi_support")))
    manual_endpoint = bool(selected["at_endpoint"] and (metadata.get("lo_manual") or metadata.get("hi_manual")))
    return {
        "projection_valid": True,
        "t": selected["t"],
        "distance": selected["distance"],
        "global_distance": selected["global_distance"],
        "at_endpoint": selected["at_endpoint"],
        "endpoint_clipped": selected["endpoint_clipped"],
        "support_component_id": metadata.get("component_id"),
        "support_interval_index": int(metadata.get("support_interval_index", -1)),
        "support_endpoint_clipped": support_endpoint,
        "manual_bound_clipped": manual_endpoint,
        "support_infeasible": False,
        "support_violation_q": 0.0,
        "objective_distance": selected["distance"],
    }


def _symmetric_arc_projection(
    points: np.ndarray,
    values: Mapping[str, float],
    branch_ids: np.ndarray,
    sides: np.ndarray,
    interval_specs: Sequence[Mapping[str, Any]],
    reference_axis: float,
    *,
    include_global_oracle: bool = False,
) -> dict[str, Any]:
    geometry = EllipseGeometry.from_values(values)
    n_points = points.shape[0]
    result: dict[str, Any] = {
        "t": np.full(n_points, np.nan, dtype=float),
        "distance": np.full(n_points, np.nan, dtype=float),
        "global_distance": np.full(n_points, np.nan, dtype=float),
        "at_endpoint": np.zeros(n_points, dtype=bool),
        "endpoint_clipped": np.zeros(n_points, dtype=bool),
        "support_endpoint_clipped": np.zeros(n_points, dtype=bool),
        "manual_bound_clipped": np.zeros(n_points, dtype=bool),
        "projection_valid": np.zeros(n_points, dtype=bool),
        "support_infeasible": np.zeros(n_points, dtype=bool),
        "support_violation_q": np.zeros(n_points, dtype=float),
        "objective_distance": np.full(n_points, np.nan, dtype=float),
        "support_component_id": np.full(n_points, None, dtype=object),
        "support_interval_index": np.full(n_points, -1, dtype=int),
        "implicit": np.full(n_points, np.nan, dtype=float),
    }
    for branch in (0, 1):
        mask = branch_ids == branch
        if not np.any(mask):
            continue
        sign = 1.0 if branch == 0 else -1.0
        branch_geometry = EllipseGeometry(
            geometry.cx,
            geometry.cy,
            geometry.a,
            geometry.axis_ratio,
            reference_axis + sign * geometry.theta,
        )
        for row in np.flatnonzero(mask):
            spec = interval_specs[int(row)]
            projection = _project_point_to_support(
                np.asarray(points[row], dtype=float),
                branch_geometry,
                str(sides[row]),
                spec.get("manual_intervals"),
                spec.get("rectangles", ()),
                include_global_oracle=include_global_oracle,
            )
            for name in (
                "t", "distance", "global_distance", "at_endpoint", "endpoint_clipped",
                "support_endpoint_clipped", "manual_bound_clipped", "projection_valid",
                "support_infeasible", "support_violation_q", "objective_distance",
                "support_component_id", "support_interval_index",
            ):
                result[name][row] = projection[name]
        result["implicit"][mask] = ellipse_implicit(points[mask], branch_geometry)
    result["signed_distance"] = np.copysign(result["objective_distance"], result["implicit"])
    result["signed_distance"] = np.nan_to_num(result["signed_distance"], nan=1.0e12, posinf=1.0e12, neginf=-1.0e12)
    result["objective_distance"] = np.nan_to_num(result["objective_distance"], nan=1.0e12, posinf=1.0e12, neginf=1.0e12)
    return result


def _parameter_set_for_arcs(
    points: np.ndarray,
    parameters: ParameterSet | Mapping[str, Any] | None,
    reference_axis: float,
) -> ParameterSet:
    result = _make_parameter_set(points, parameters)
    if parameters is None:
        # The arc API is designed for q-space data already centred at the
        # calibrated origin.  Keep this as an editable explicit choice rather
        # than allowing the point median to become a hidden fitted centre.
        result["cx"] = result["cx"].copy(value=0.0, vary=False, name="cx")
        result["cy"] = result["cy"].copy(value=0.0, vary=False, name="cy")
        relative = (result["theta"].value - reference_axis + _HALF_PI) % math.pi - _HALF_PI
        result["theta"].set_value(relative)

    theta_spec = result["theta"]
    lower = max(0.0, theta_spec.min if theta_spec.min is not None else 0.0)
    upper = min(_HALF_PI, theta_spec.max if theta_spec.max is not None else _HALF_PI)
    if lower > upper:
        raise ValueError("symmetric arc theta bounds must overlap [0, pi/2]")
    if theta_spec.expr is None:
        canonical = abs((float(theta_spec.value) + _HALF_PI) % math.pi - _HALF_PI)
        result["theta"] = theta_spec.copy(
            value=float(np.clip(canonical, lower, upper)),
            min=lower,
            max=upper,
            name="theta",
        )
    else:
        result["theta"] = theta_spec.copy(min=lower, max=upper, name="theta")
    return result


def _wrap_half_turn(value: float) -> float:
    return (float(value) + _HALF_PI) % math.pi - _HALF_PI


def _algebraic_arc_seed(
    points: np.ndarray,
    labels: np.ndarray,
    parameters: ParameterSet,
    reference_axis: float,
) -> ParameterSet:
    """Seed free axes from the fixed-centre quadratic ellipse equation.

    Very flat, short arcs can make the covariance initial guess follow the
    local tangent rather than the ellipse's major axis.  Each labelled branch
    supplies a small linear conic problem; with a free centre the linear and
    quadratic terms also recover a centre seed.  Positive eigenvalues provide
    deterministic axis and orientation seeds.  This is an initialization only.
    The bounded closest-point fit remains the sole nonlinear optimizer and all
    user bounds/fixed/tied states are preserved.
    """

    if points.shape[0] < 10 or any(np.count_nonzero(labels == branch) < 5 for branch in (0, 1)):
        return parameters
    values = parameters.resolve()
    centre = np.asarray((values["cx"], values["cy"]), dtype=float)
    fixed_center = parameters["cx"].is_fixed and parameters["cy"].is_fixed
    branch_axes: list[tuple[np.ndarray, float, float, float]] = []
    for branch in (0, 1):
        centered = points[labels == branch] - centre
        if fixed_center:
            design = np.column_stack(
                (centered[:, 0] ** 2, 2.0 * centered[:, 0] * centered[:, 1], centered[:, 1] ** 2)
            )
        else:
            design = np.column_stack(
                (
                    centered[:, 0] ** 2,
                    2.0 * centered[:, 0] * centered[:, 1],
                    centered[:, 1] ** 2,
                    centered[:, 0],
                    centered[:, 1],
                )
            )
        required_rank = 3 if fixed_center else 5
        if np.linalg.matrix_rank(design) < required_rank:
            return parameters
        try:
            coefficients, _, _, singular_values = np.linalg.lstsq(
                design, np.ones(centered.shape[0], dtype=float), rcond=None
            )
        except np.linalg.LinAlgError:
            return parameters
        if singular_values.size < 3 or singular_values[-1] <= 0.0:
            return parameters
        condition = float(singular_values[0] / singular_values[-1])
        if not np.isfinite(condition) or condition > 1.0e12:
            return parameters
        matrix = np.asarray(
            ((coefficients[0], coefficients[1]), (coefficients[1], coefficients[2])),
            dtype=float,
        )
        if fixed_center:
            branch_centre = centre.copy()
            normalization = 1.0
        else:
            linear = np.asarray(coefficients[3:5], dtype=float)
            try:
                branch_offset = -0.5 * np.linalg.solve(matrix, linear)
            except np.linalg.LinAlgError:
                return parameters
            branch_centre = centre + branch_offset
            normalization = 1.0 + float(branch_offset @ matrix @ branch_offset)
        if not np.isfinite(normalization) or abs(normalization) <= np.finfo(float).tiny:
            return parameters
        matrix = matrix / normalization
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        if (
            not np.all(np.isfinite(eigenvalues))
            or np.any(eigenvalues <= 0.0)
            or not np.all(np.isfinite(eigenvectors))
        ):
            return parameters
        major_index = int(np.argmin(eigenvalues))
        axes = 1.0 / np.sqrt(eigenvalues)
        a_seed = float(axes[major_index])
        ratio_seed = float(np.min(axes) / np.max(axes))
        angle = math.atan2(
            float(eigenvectors[1, major_index]),
            float(eigenvectors[0, major_index]),
        )
        branch_axes.append((np.asarray(branch_centre, dtype=float), a_seed, ratio_seed, angle))

    centre_seed = np.mean([item[0] for item in branch_axes], axis=0)
    a_seed = float(np.mean([item[1] for item in branch_axes]))
    ratio_seed = float(np.mean([item[2] for item in branch_axes]))
    relative_angles = [
        _wrap_half_turn(branch_axes[0][3] - reference_axis),
        _wrap_half_turn(branch_axes[1][3] - reference_axis),
    ]
    theta_seed = float(np.mean(np.abs(relative_angles)))
    if not all(np.isfinite((a_seed, ratio_seed, theta_seed))):
        return parameters
    seeded = parameters.copy()
    for name, value in (
        ("cx", float(centre_seed[0])),
        ("cy", float(centre_seed[1])),
        ("a", a_seed),
        ("axis_ratio", ratio_seed),
        ("theta", theta_seed),
    ):
        spec = seeded[name]
        if spec.expr is not None or not spec.vary:
            continue
        if spec.min is not None:
            value = max(float(spec.min), value)
        if spec.max is not None:
            value = min(float(spec.max), value)
        try:
            spec.set_value(value)
        except ValueError:
            return parameters
    return seeded


def _select_candidate(
    candidates: list[tuple[int, EllipseFitResult, dict[str, Any]]],
) -> tuple[int, EllipseFitResult, tuple[dict[str, Any], ...]]:
    finite_candidates = [item for item in candidates if np.isfinite(item[1].cost)]
    pool = finite_candidates if finite_candidates else candidates
    selected_index, selected, _ = min(
        pool,
        key=lambda item: (
            float(item[1].cost) if np.isfinite(item[1].cost) else float("inf"),
            not bool(item[1].success),
            item[0],
        ),
    )
    return selected_index, selected, tuple(item[2] for item in candidates)


def _mirror_point(point: np.ndarray, center: np.ndarray, reference_axis: float) -> np.ndarray:
    axis = np.asarray([math.cos(reference_axis), math.sin(reference_axis)], dtype=float)
    normal = np.asarray([-axis[1], axis[0]], dtype=float)
    delta = point - center
    return center + axis * float(np.dot(delta, axis)) - normal * float(np.dot(delta, normal))


def _update_mirror_evidence(
    point_diagnostics: list[dict[str, Any]],
    arc_records: dict[int, dict[str, Any]],
    reference_axis: float,
    *,
    reference_center: Sequence[float] | None = None,
) -> None:
    """Pair observed source arcs using fixed labels and q-space reflection."""

    center: np.ndarray | None = None
    if reference_center is not None:
        try:
            candidate = np.asarray([float(reference_center[0]), float(reference_center[1])], dtype=float)
            if np.all(np.isfinite(candidate)):
                center = candidate
        except (TypeError, ValueError, IndexError):
            center = None
    if center is None:
        centers = [
            (float(item["center_qx"]), float(item["center_qy"]))
            for item in point_diagnostics
            if item.get("center_qx") is not None and item.get("center_qy") is not None
        ]
        if centers:
            candidate = np.asarray(centers, dtype=float)
            if candidate.size and np.all(np.isfinite(candidate)):
                center = np.median(candidate, axis=0)
    by_arc: dict[int, list[dict[str, Any]]] = {}
    for item in point_diagnostics:
        if not item.get("accepted") or not np.isfinite(item.get("qx", np.nan)) or not np.isfinite(item.get("qy", np.nan)):
            continue
        arc_id = item.get("arc_id")
        if isinstance(arc_id, Integral) and not isinstance(arc_id, (bool, np.bool_)):
            by_arc.setdefault(int(arc_id), []).append(item)
    for arc_id, arc in arc_records.items():
        if center is None:
            arc.update(
                {
                    "mirror_symmetry_status": "unavailable",
                    "mirror_symmetry_reason": "tracer_center_unavailable",
                    "mirror_partner_arc_ids": [],
                    "mirror_pair_fraction": None,
                    "mirror_deviation_q_median": None,
                    "mirror_deviation_q_rms": None,
                    "mirror_deviation_q_p95": None,
                }
            )
            continue
        branch = arc.get("branch_id")
        side = arc.get("side")
        if branch not in (0, 1) or side not in ("upper", "lower"):
            arc.update(
                {
                    "mirror_symmetry_status": "unavailable",
                    "mirror_symmetry_reason": "source_branch_or_side_unresolved",
                    "mirror_partner_arc_ids": [],
                    "mirror_pair_fraction": None,
                    "mirror_deviation_q_median": None,
                    "mirror_deviation_q_rms": None,
                    "mirror_deviation_q_p95": None,
                }
            )
            continue
        source = by_arc.get(int(arc_id), [])
        partner_side = "lower" if side == "upper" else "upper"
        partner_arcs = [
            other_id
            for other_id, other in arc_records.items()
            if other_id != arc_id and other.get("branch_id") == 1 - branch and other.get("side") == partner_side
        ]
        partners = [item for other_id in partner_arcs for item in by_arc.get(other_id, [])]
        if not source or not partners:
            arc.update(
                {
                    "mirror_symmetry_status": "unpaired",
                    "mirror_symmetry_reason": "opposite_fixed_label_arc_unavailable",
                    "mirror_partner_arc_ids": sorted(int(item) for item in partner_arcs),
                    "mirror_pair_fraction": 0.0,
                    "mirror_deviation_q_median": None,
                    "mirror_deviation_q_rms": None,
                    "mirror_deviation_q_p95": None,
                }
            )
            continue
        deviations: list[float] = []
        ambiguous = False
        for item in source:
            source_q = np.asarray([float(item["qx"]), float(item["qy"])], dtype=float)
            target = _mirror_point(source_q, center, reference_axis)
            distances = []
            for partner in partners:
                partner_q = np.asarray([float(partner["qx"]), float(partner["qy"])], dtype=float)
                distance = float(np.linalg.norm(partner_q - target))
                scale = max(
                    _as_finite(item.get("support_padding_q")) or 0.0,
                    _as_finite(partner.get("support_padding_q")) or 0.0,
                    1.0e-12,
                )
                if distance <= 2.0 * scale:
                    distances.append((distance, partner))
            distances.sort(key=lambda pair: (pair[0], str(pair[1].get("point_id", ""))))
            if not distances:
                continue
            if len(distances) > 1 and abs(distances[1][0] - distances[0][0]) <= 1.0e-12:
                ambiguous = True
                continue
            deviations.append(float(distances[0][0]))
        fraction = float(len(deviations) / len(source)) if source else 0.0
        status = "ambiguous" if ambiguous else "observed" if deviations else "unpaired"
        arc.update(
            {
                "mirror_symmetry_status": status,
                "mirror_symmetry_reason": "multiple_equal_nearest_pairs" if ambiguous else None if deviations else "no_partner_within_observed_padding",
                "mirror_partner_arc_ids": sorted(int(item) for item in partner_arcs),
                "mirror_pair_fraction": fraction,
                "mirror_deviation_q_median": float(np.median(deviations)) if deviations else None,
                "mirror_deviation_q_rms": float(np.sqrt(np.mean(np.square(deviations)))) if deviations else None,
                "mirror_deviation_q_p95": float(np.percentile(deviations, 95.0)) if deviations else None,
            }
        )


def _update_diagnostics(
    point_diagnostics: list[dict[str, Any]],
    arc_records: dict[int, dict[str, Any]],
    valid_original_indices: list[int],
    projection: Mapping[str, np.ndarray],
    signed_residuals: np.ndarray,
    global_labels: np.ndarray,
    values: Mapping[str, float],
    reference_axis: float,
    reference_center: Sequence[float] | None = None,
) -> list[dict[str, Any]]:
    base_geometry = EllipseGeometry.from_values(values)
    for row, original_index in enumerate(valid_original_indices):
        diagnostic = point_diagnostics[original_index]
        branch = int(global_labels[row])
        valid = bool(projection["projection_valid"][row])
        t_raw = projection["t"][row]
        distance_raw = projection["distance"][row]
        sign = 1.0 if branch == 0 else -1.0
        branch_geometry = EllipseGeometry(
            base_geometry.cx,
            base_geometry.cy,
            base_geometry.a,
            base_geometry.axis_ratio,
            reference_axis + sign * base_geometry.theta,
        )
        t_value = float(t_raw) if valid and np.isfinite(t_raw) else None
        distance = float(distance_raw) if valid and np.isfinite(distance_raw) else None
        projected = np.asarray(branch_geometry.point(t_value), dtype=float).reshape(2) if t_value is not None else None
        normal_residual = float(
            ellipse_sampson_residuals(
                np.asarray([[diagnostic.get("qx", np.nan), diagnostic.get("qy", np.nan)]], dtype=float),
                branch_geometry,
            )[0]
        ) if valid and "qx" in diagnostic and "qy" in diagnostic else None
        support_violation = float(projection["support_violation_q"][row])
        diagnostic.update(
            {
                "fit_branch_id": branch,
                "projection_valid": valid,
                "projection_t": t_value,
                "projected_qx": float(projected[0]) if projected is not None else None,
                "projected_qy": float(projected[1]) if projected is not None else None,
                "projection_distance_q": distance,
                "distance_q": distance,
                "projection_residual_q": float(signed_residuals[row]) if valid else None,
                "normal_residual_q": normal_residual,
                "global_distance_oracle_q": float(projection["global_distance"][row]) if valid and np.isfinite(projection["global_distance"][row]) else None,
                "projection_at_endpoint": bool(projection["at_endpoint"][row]) if valid else False,
                "endpoint_clipped": bool(projection["endpoint_clipped"][row]) if valid else False,
                "support_component_id": projection["support_component_id"][row],
                "support_interval_index": int(projection["support_interval_index"][row]),
                "support_endpoint_clipped": bool(projection["support_endpoint_clipped"][row]),
                "manual_bound_clipped": bool(projection["manual_bound_clipped"][row]),
                "support_infeasible": bool(projection["support_infeasible"][row]),
                "support_violation_q": support_violation if np.isfinite(support_violation) else None,
                "projection_reason": "valid_observed_support_projection" if valid else "observed_support_infeasible",
            }
        )
        arc_id = int(diagnostic["arc_id"])
        arc = arc_records[arc_id]
        arc.setdefault("_projection_t", [])
        arc.setdefault("_distances", [])
        arc.setdefault("_normal_residuals", [])
        arc.setdefault("_normal_ratios", [])
        arc.setdefault("_valid_count", 0)
        arc.setdefault("_invalid_count", 0)
        arc.setdefault("_endpoint_count", 0)
        arc.setdefault("_support_endpoint_count", 0)
        arc.setdefault("_manual_endpoint_count", 0)
        arc.setdefault("_support_violations", [])
        if valid:
            arc["_projection_t"].append(float(t_value))
            arc["_distances"].append(float(distance))
            if normal_residual is not None and np.isfinite(normal_residual):
                arc["_normal_residuals"].append(float(normal_residual))
                sigma = _as_finite(diagnostic.get("localization_sigma_q"))
                if sigma is not None and sigma > 0.0:
                    arc["_normal_ratios"].append(float(abs(normal_residual) / sigma))
            arc["_valid_count"] += 1
            if bool(projection["at_endpoint"][row]):
                arc["_endpoint_count"] += 1
            if bool(projection["support_endpoint_clipped"][row]):
                arc["_support_endpoint_count"] += 1
            if bool(projection["manual_bound_clipped"][row]):
                arc["_manual_endpoint_count"] += 1
        else:
            arc["_invalid_count"] += 1
            if np.isfinite(support_violation):
                arc["_support_violations"].append(support_violation)

    for arc in arc_records.values():
        t_values = np.asarray(arc.pop("_projection_t", ()), dtype=float)
        distances = np.asarray(arc.pop("_distances", ()), dtype=float)
        normal_residuals = np.asarray(arc.pop("_normal_residuals", ()), dtype=float)
        normal_ratios = np.asarray(arc.pop("_normal_ratios", ()), dtype=float)
        support_violations = np.asarray(arc.pop("_support_violations", ()), dtype=float)
        valid_count = int(arc.pop("_valid_count", 0))
        invalid_count = int(arc.pop("_invalid_count", 0))
        endpoint_count = int(arc.pop("_endpoint_count", 0))
        support_endpoint_count = int(arc.pop("_support_endpoint_count", 0))
        manual_endpoint_count = int(arc.pop("_manual_endpoint_count", 0))
        arc["projection_t_min"] = float(np.min(t_values)) if t_values.size else None
        arc["projection_t_max"] = float(np.max(t_values)) if t_values.size else None
        arc["projection_span"] = float(np.ptp(t_values)) if t_values.size else 0.0
        arc["endpoint_fraction"] = float(endpoint_count / t_values.size) if t_values.size else 0.0
        arc["support_endpoint_fraction"] = float(support_endpoint_count / t_values.size) if t_values.size else 0.0
        arc["manual_endpoint_fraction"] = float(manual_endpoint_count / t_values.size) if t_values.size else 0.0
        arc["valid_projection_count"] = valid_count
        arc["infeasible_projection_count"] = invalid_count
        arc["projection_rmse_q"] = (
            float(np.sqrt(np.mean(np.square(distances)))) if distances.size else None
        )
        arc["normal_residual_q_median"] = float(np.median(normal_residuals)) if normal_residuals.size else None
        arc["normal_residual_q_rms"] = float(np.sqrt(np.mean(np.square(normal_residuals)))) if normal_residuals.size else None
        arc["normal_residual_q_p95"] = float(np.percentile(np.abs(normal_residuals), 95.0)) if normal_residuals.size else None
        arc["normal_residual_localization_ratio_median"] = float(np.median(normal_ratios)) if normal_ratios.size else None
        arc["normal_residual_localization_ratio_rms"] = float(np.sqrt(np.mean(np.square(normal_ratios)))) if normal_ratios.size else None
        arc["normal_residual_localization_ratio_p95"] = float(np.percentile(normal_ratios, 95.0)) if normal_ratios.size else None
        arc["support_violation_q_median"] = float(np.median(support_violations)) if support_violations.size else None
        arc["support_violation_q_max"] = float(np.max(support_violations)) if support_violations.size else None
        arc["support_infeasible"] = bool(invalid_count > 0)
        arc.setdefault("support_gap_count", 0)
        arc.setdefault("support_status", "manual_only")
        arc["evidence_status"] = "incomplete" if invalid_count or not normal_residuals.size else "observed"
    _update_mirror_evidence(
        point_diagnostics,
        arc_records,
        reference_axis,
        reference_center=reference_center,
    )
    return point_diagnostics


def fit_arc_ellipses(
    points: list[dict[str, Any]],
    *,
    parameters: ParameterSet | Mapping[str, Any] | None = None,
    reference_axis_deg: float = 0.0,
    multistart: int = 7,
    max_nfev: int = 800,
    cancel_event: Any = None,
    observed_support: Mapping[Any, Any] | None = None,
    reference_center: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Fit a shared-centre mirror pair to labelled bounded ellipse arcs.

    ``branch_id=0`` is the ellipse at ``reference_axis + theta`` and
    ``branch_id=1`` is the ellipse at ``reference_axis - theta``.  Within
    either ellipse, ``upper`` means local parameter ``t`` in ``[0, pi]`` and
    ``lower`` means ``t`` in ``[pi, 2*pi]``.  Manual arc bounds are intersected
    with that side interval; points without manual bounds require a frozen
    observed q-space support corridor.
    """

    reference_axis = math.radians(float(reference_axis_deg))
    if not np.isfinite(reference_axis):
        raise ValueError("reference_axis_deg must be finite")
    if isinstance(max_nfev, (bool, np.bool_)) or not isinstance(max_nfev, Integral) or int(max_nfev) < 1:
        raise ValueError("max_nfev must be an integer >= 1")
    multistart_count = _validate_multistart_count(multistart)
    (
        xy,
        source_labels,
        sides,
        _arc_ids,
        sigmas,
        point_diagnostics,
        arc_records,
        parsed_rows,
    ) = _parse_points(points, support_registry=observed_support)
    if xy.shape[0] == 0:
        for diagnostic in point_diagnostics:
            arc_id = diagnostic.get("arc_id")
            if diagnostic.get("excluded") and isinstance(arc_id, int) and arc_id in arc_records:
                arc_records[arc_id].setdefault("excluded_reasons", []).append(
                    diagnostic.get("excluded_reason")
                )
        return {
            "fit": None,
            "point_diagnostics": point_diagnostics,
            "points": point_diagnostics,
            "rows": point_diagnostics,
            "arc_diagnostics": [arc_records[key] for key in sorted(arc_records)],
            "error": "no accepted points with valid branch, side, and arc labels",
            "message": "no accepted points with valid branch, side, and arc labels",
            "branch_swap_applied": False,
        }

    valid_original_indices = [
        index for index, diagnostic in enumerate(point_diagnostics) if diagnostic.get("used")
    ]
    interval_specs = [
        {
            "manual_intervals": row.get("manual_intervals"),
            "rectangles": row.get("rectangles", ()),
            "support_status": row.get("support_status"),
        }
        for row in parsed_rows
    ]
    parameter_set = _parameter_set_for_arcs(xy, parameters, reference_axis)
    parameter_set = _algebraic_arc_seed(xy, source_labels, parameter_set, reference_axis)
    sigma_weights = 1.0 / sigmas

    def objective(candidate: ParameterSet, labels: np.ndarray) -> np.ndarray:
        raise_if_cancelled(cancel_event, "ellipse:arc:residual")
        projection = _symmetric_arc_projection(
            xy,
            candidate.resolve(),
            labels,
            sides,
            interval_specs,
            reference_axis,
            include_global_oracle=False,
        )
        return projection["signed_distance"] * sigma_weights

    label_options = [source_labels]
    if np.any(source_labels == 0) and np.any(source_labels == 1):
        # Branch IDs identify the observed components, while the mirror pair
        # itself is invariant under a global component swap.  Trying this one
        # deterministic alternative handles that equivalence without ever
        # inferring a label for an individual point.
        label_options.append(1 - source_labels)

    all_candidates: list[tuple[int, EllipseFitResult, dict[str, Any], np.ndarray]] = []
    candidate_index = 0
    for option_index, labels in enumerate(label_options):
        starts = _multistart_parameter_sets(parameter_set, multistart_count)
        for start_index, start in enumerate(starts):
            raise_if_cancelled(cancel_event, "ellipse:arc:multistart")
            result = _run_fit(
                xy,
                start,
                lambda candidate, labels=labels: objective(candidate, labels),
                loss="soft_l1",
                f_scale=1.0,
                max_nfev=int(max_nfev),
                model="bounded_arc_symmetric_double_ellipse",
                components=2,
                labels=labels,
                reference_axis=reference_axis,
                residual="geometric",
                cancel_event=cancel_event,
            )
            record = {
                "start_index": int(start_index),
                "label_option": int(option_index),
                "start_values": dict(start.resolve()),
                "values": dict(result.values),
                "success": bool(result.success),
                "finite_cost": bool(np.isfinite(result.cost)),
                "cost": float(result.cost),
                "message": str(result.message),
            }
            all_candidates.append((candidate_index, result, record, labels))
            candidate_index += 1

    selected_index, selected, _ = _select_candidate(
        [(index, result, record) for index, result, record, _ in all_candidates]
    )
    selected_item = next(item for item in all_candidates if item[0] == selected_index)
    selected_labels = selected_item[3]
    # Preserve the canonical result type and attach the complete deterministic
    # audit trail, including the optional global branch swap.
    fit = EllipseFitResult(
        **{
            field_name: getattr(selected, field_name)
            for field_name in EllipseFitResult.__dataclass_fields__
            if field_name not in {"candidate_solutions", "selected_start_index", "multistart_count"}
        },
        candidate_solutions=tuple(item[2] for item in all_candidates),
        selected_start_index=int(selected_item[2]["start_index"]),
        multistart_count=multistart_count,
    )

    projection = _symmetric_arc_projection(
        xy,
        fit.values,
        selected_labels,
        sides,
        interval_specs,
        reference_axis,
        include_global_oracle=True,
    )
    signed_residuals = np.asarray(projection["signed_distance"], dtype=float)
    _update_diagnostics(
        point_diagnostics,
        arc_records,
        valid_original_indices,
        projection,
        signed_residuals,
        selected_labels,
        fit.values,
        reference_axis,
        reference_center,
    )
    valid_projection = np.asarray(projection["projection_valid"], dtype=bool)
    # ``fit.residuals`` remains the sigma-weighted optimizer objective.  The
    # public coverage metric is deliberately rebuilt from valid, unweighted
    # q-space projection distances so its units cannot be mistaken for sigma
    # units.  Invalid support rows never enter the RMS.
    coverage_points = xy[valid_projection]
    coverage_distances = np.asarray(projection["distance"], dtype=float)[valid_projection]
    coverage_labels = np.asarray(selected_labels, dtype=int)[valid_projection]
    fit.coverage = _coverage(
        coverage_points,
        coverage_distances,
        fit.parameters,
        components=2,
        labels=coverage_labels,
        reference_axis=reference_axis,
        radial_rms_definition="unweighted_q_space_projection_distance",
        radial_rms_units="q",
    )
    for diagnostic in point_diagnostics:
        if diagnostic.get("excluded"):
            arc_id = diagnostic.get("arc_id")
            if isinstance(arc_id, int) and arc_id in arc_records:
                arc_records[arc_id].setdefault("excluded_reasons", []).append(
                    diagnostic.get("excluded_reason")
                )
    arc_diagnostics = [arc_records[key] for key in sorted(arc_records)]
    return {
        "fit": fit,
        "point_diagnostics": point_diagnostics,
        "points": point_diagnostics,
        "rows": point_diagnostics,
        "arc_diagnostics": arc_diagnostics,
        "branch_swap_applied": bool(selected_item[2]["label_option"] == 1),
    }


def _projection_values(values: Mapping[str, Any] | ParameterSet) -> dict[str, float]:
    if isinstance(values, ParameterSet):
        source = dict(values.resolve())
    elif isinstance(values, Mapping):
        source = dict(values)
    else:
        raise TypeError("values must be an ellipse mapping or ParameterSet")
    def scalar(value: Any) -> Any:
        return value.get("value", value.get("initial")) if isinstance(value, Mapping) else value

    if "axis_ratio" not in source and source.get("a") is not None and source.get("b") is not None:
        source["axis_ratio"] = float(scalar(source["b"])) / float(scalar(source["a"]))
    if "theta" not in source and source.get("theta_deg") is not None:
        source["theta"] = math.radians(float(scalar(source["theta_deg"])))
    source.setdefault("theta", 0.0)
    required = ("cx", "cy", "a", "axis_ratio", "theta")
    missing = [name for name in required if name not in source]
    if missing:
        raise ValueError("missing ellipse projection values: " + ", ".join(missing))
    normalized = {}
    for name in required:
        normalized[name] = float(scalar(source[name]))
    return normalized


def project_arc_points(
    records: Sequence[Mapping[str, Any]],
    values: Mapping[str, Any] | ParameterSet,
    *,
    reference_axis_deg: float = 0.0,
    branch_swap_applied: bool = False,
) -> dict[str, Any]:
    """Project labelled records with frozen support without refitting.

    The return envelope is source-order and JSON-safe: ``point_diagnostics``
    contains one row per input record, valid rows expose a bounded
    ``projection_t`` and projected q coordinates, and infeasible or malformed
    rows expose ``projection_valid=False``, missing projection coordinates, a
    finite ``support_violation_q`` when applicable, and an explicit
    ``projection_reason``/``excluded_reason``.  This is intended for held-out
    prediction diagnostics; it never enforces the two-branch fit requirement.
    """

    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise TypeError("records must be a sequence of point mappings")
    reference_axis = math.radians(float(reference_axis_deg))
    if not np.isfinite(reference_axis):
        raise ValueError("reference_axis_deg must be finite")
    projection_values = _projection_values(values)
    rows = list(records)
    (
        xy,
        source_labels,
        sides,
        _arc_ids,
        _sigmas,
        point_diagnostics,
        arc_records,
        parsed_rows,
    ) = _parse_points(rows)
    if xy.shape[0] == 0:
        for diagnostic in point_diagnostics:
            diagnostic.setdefault("projection_valid", False)
            diagnostic.setdefault("projection_reason", diagnostic.get("excluded_reason", "observed_support_unavailable"))
            diagnostic.setdefault("projection_t", None)
            diagnostic.setdefault("projected_qx", None)
            diagnostic.setdefault("projected_qy", None)
            diagnostic.setdefault("support_violation_q", None)
        return {
            "projection_version": "observed_arc_projection_v1",
            "values": projection_values,
            "reference_axis_deg": float(reference_axis_deg),
            "branch_swap_applied": bool(branch_swap_applied),
            "branch_label_mapping": {
                "fit_branch_to_observed": {
                    "0": 1 if branch_swap_applied else 0,
                    "1": 0 if branch_swap_applied else 1,
                }
            },
            "point_diagnostics": point_diagnostics,
            "points": point_diagnostics,
            "rows": point_diagnostics,
            "arc_diagnostics": [arc_records[key] for key in sorted(arc_records)],
            "n_input": len(rows),
            "n_valid": 0,
            "n_invalid": len(rows),
            "error": "no projectable observed arc points",
        }
    interval_specs = [
        {
            "manual_intervals": row.get("manual_intervals"),
            "rectangles": row.get("rectangles", ()),
            "support_status": row.get("support_status"),
        }
        for row in parsed_rows
    ]
    labels = 1 - source_labels if branch_swap_applied else source_labels
    projection = _symmetric_arc_projection(
        xy,
        projection_values,
        labels,
        sides,
        interval_specs,
        reference_axis,
        include_global_oracle=True,
    )
    signed_residuals = np.asarray(projection["signed_distance"], dtype=float)
    valid_original_indices = [
        index for index, diagnostic in enumerate(point_diagnostics) if diagnostic.get("used")
    ]
    _update_diagnostics(
        point_diagnostics,
        arc_records,
        valid_original_indices,
        projection,
        signed_residuals,
        labels,
        projection_values,
        reference_axis,
        reference_center=None,
    )
    for diagnostic in point_diagnostics:
        if diagnostic.get("excluded"):
            diagnostic.setdefault("projection_valid", False)
            diagnostic.setdefault("projection_reason", diagnostic.get("excluded_reason", "observed_support_unavailable"))
            diagnostic.setdefault("projection_t", None)
            diagnostic.setdefault("projected_qx", None)
            diagnostic.setdefault("projected_qy", None)
            diagnostic.setdefault("support_violation_q", None)
    valid_count = sum(bool(item.get("projection_valid")) for item in point_diagnostics)
    return {
        "projection_version": "observed_arc_projection_v1",
        "values": projection_values,
        "reference_axis_deg": float(reference_axis_deg),
        "branch_swap_applied": bool(branch_swap_applied),
        "branch_label_mapping": {
            "fit_branch_to_observed": {
                "0": 1 if branch_swap_applied else 0,
                "1": 0 if branch_swap_applied else 1,
            }
        },
        "point_diagnostics": point_diagnostics,
        "points": point_diagnostics,
        "rows": point_diagnostics,
        "arc_diagnostics": [arc_records[key] for key in sorted(arc_records)],
        "n_input": len(rows),
        "n_valid": int(valid_count),
        "n_invalid": int(len(rows) - valid_count),
    }


__all__ = ["fit_arc_ellipses", "project_arc_points"]
