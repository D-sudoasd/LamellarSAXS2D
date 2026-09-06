"""Freeze observed q-space support for traced butterfly arcs.

The ellipse fitter must receive the support that was actually observed by the
ridge tracer.  This module deliberately knows nothing about ellipse
parameters: it builds a finite collection of q-space rectangles from accepted
neighbouring points and records disconnected gaps instead of filling them.

``freeze_observed_support`` mutates the supplied public point and arc records
so the support survives the existing trace -> fit -> JSON path.  The returned
envelope is useful to callers that keep a separate trace diagnostics object;
``summary_only=True`` keeps only its scalar summary when those records are
already retained on the points/arcs.  The representation is JSON-safe and
versioned because it is an evidence boundary, rather than an internal
numerical cache.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from numbers import Integral, Real
from typing import Any

import numpy as np

from .cancellation import raise_if_cancelled


SUPPORT_VERSION = "observed_q_corridor_v1"
_EPS = np.finfo(float).eps
_PAD_FORMULA = "max(localization_sigma_q, sampling_sigma_q, q_normal_step/sqrt(12))"
_PAD_ROLE = "finite_observed_support_corridor_not_confidence_interval"
_PAD_MULTIPLIER = 2.0
_GAP_THRESHOLD_MULTIPLIER = 4.0


def _finite(value: Any) -> float | None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _point_id(point: Mapping[str, Any], fallback: int) -> Any:
    return _json_scalar(point.get("point_id", point.get("id", fallback)))


def _same_id(left: Any, right: Any) -> bool:
    return left == right or str(left) == str(right)


def _q(point: Mapping[str, Any]) -> tuple[float, float] | None:
    x = _finite(point.get("qx"))
    y = _finite(point.get("qy"))
    if x is None or y is None:
        return None
    return float(x), float(y)


def _positive_scale(point: Mapping[str, Any], arc: Mapping[str, Any]) -> tuple[float | None, dict[str, float | None]]:
    """Return the prescribed finite q-space padding and its provenance."""

    values: dict[str, float | None] = {}
    for name in ("localization_sigma_q", "sampling_sigma_q"):
        value = _finite(point.get(name))
        if value is None:
            value = _finite(arc.get(name))
        values[name] = value if value is not None and value > 0 else None
    step = _finite(point.get("q_normal_step"))
    if step is None:
        step = _finite(point.get("normal_step_q"))
    if step is None:
        step = _finite(arc.get("q_normal_step"))
    values["q_normal_step"] = step if step is not None and step > 0 else None
    step_sigma = step / math.sqrt(12.0) if step is not None and step > 0 else None
    values["q_normal_step_over_sqrt12"] = step_sigma
    finite = [
        value
        for value in (values["localization_sigma_q"], values["sampling_sigma_q"], step_sigma)
        if value is not None and np.isfinite(value) and value > 0
    ]
    return (max(finite) if finite else None), values


def _unit_vector(point: Mapping[str, Any], names: tuple[str, str]) -> np.ndarray | None:
    x = _finite(point.get(names[0]))
    y = _finite(point.get(names[1]))
    if x is None or y is None:
        return None
    vector = np.asarray([float(x), float(y)], dtype=float)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= _EPS:
        return None
    return vector / norm


def _ordered_members(
    members: list[tuple[int, Mapping[str, Any]]],
    arc: Mapping[str, Any],
) -> list[tuple[int, Mapping[str, Any]]]:
    """Follow the tracer's explicit ordering whenever it is available."""

    ordered_ids = arc.get("ordered_point_ids", arc.get("point_ids"))
    if isinstance(ordered_ids, Sequence) and not isinstance(ordered_ids, (str, bytes)):
        remaining = list(members)
        by_id = {
            str(_point_id(point, index)): (index, point)
            for index, point in remaining
        }
        ordered: list[tuple[int, Mapping[str, Any]]] = []
        for wanted in ordered_ids:
            item = by_id.pop(str(wanted), None)
            if item is not None:
                ordered.append(item)
                try:
                    remaining.remove(item)
                except ValueError:
                    pass
        # Preserve any points omitted by a stale edit record at the end.  They
        # remain observed data, but never reorder the explicit trace path.
        ordered.extend(sorted(remaining, key=lambda item: (int(item[1].get("arc_order", item[0])), item[0])))
        return ordered
    return sorted(members, key=lambda item: (int(item[1].get("arc_order", item[0])), item[0]))


def _explicit_component_groups(
    ordered: list[tuple[int, Mapping[str, Any]]],
    arc: Mapping[str, Any],
) -> list[list[tuple[int, Mapping[str, Any]]]] | None:
    """Use caller-labelled fragments when present; never infer a bridge."""

    grouped: dict[str, list[tuple[int, Mapping[str, Any]]]] = {}
    has_explicit = False
    for item in ordered:
        point = item[1]
        raw = point.get("support_component_id", point.get("component_id"))
        if raw is None:
            continue
        has_explicit = True
        grouped.setdefault(str(raw), []).append(item)
    raw_components = arc.get("support_components", arc.get("components"))
    if isinstance(raw_components, Sequence) and not isinstance(raw_components, (str, bytes)):
        by_id = {
            str(_point_id(point, index)): (index, point)
            for index, point in ordered
        }
        for component_index, component in enumerate(raw_components):
            if not isinstance(component, Mapping):
                continue
            ids = component.get("point_ids", component.get("ordered_point_ids", ()))
            if not isinstance(ids, Sequence) or isinstance(ids, (str, bytes)):
                continue
            has_explicit = True
            key = str(component.get("component_id", component_index))
            selected = [by_id[str(wanted)] for wanted in ids if str(wanted) in by_id]
            if selected:
                grouped[key] = selected
    if not has_explicit:
        return None
    return [grouped[key] for key in sorted(grouped, key=str) if grouped[key]]


def _gap_threshold(left_pad: float, right_pad: float) -> float:
    return _GAP_THRESHOLD_MULTIPLIER * max(left_pad, right_pad, _EPS)


def _forced_rejected_gaps(
    ordered_members: list[tuple[int, Mapping[str, Any]]],
    arc: Mapping[str, Any],
) -> set[tuple[str, str]]:
    """Return accepted-neighbour pairs separated by excluded trace members."""

    by_id = {
        str(_point_id(point, index)): point
        for index, point in ordered_members
    }
    ordered_ids = arc.get("ordered_point_ids", arc.get("point_ids"))
    if not isinstance(ordered_ids, Sequence) or isinstance(ordered_ids, (str, bytes)):
        ordered_ids = [_point_id(point, index) for index, point in ordered_members]
    forced: set[tuple[str, str]] = set()
    previous_accepted: str | None = None
    excluded_between = False
    for wanted in ordered_ids:
        key = str(wanted)
        point = by_id.get(key)
        accepted = bool(point is not None and point.get("accepted", point.get("valid", False)))
        if accepted:
            if previous_accepted is not None and excluded_between:
                forced.add((previous_accepted, key))
            previous_accepted = key
            excluded_between = False
        elif previous_accepted is not None:
            # A rejected, ambiguous, malformed, or missing ordered ID is an
            # observed topology break even when q spacing alone is small.
            excluded_between = True
    return forced


def _split_by_observed_gaps(
    ordered: list[tuple[int, Mapping[str, Any]]],
    pads: Mapping[Any, float],
    forced_breaks: set[tuple[str, str]] | None = None,
) -> tuple[list[list[tuple[int, Mapping[str, Any]]]], list[dict[str, Any]]]:
    if not ordered:
        return [], []
    distances: list[float] = []
    forced_breaks = forced_breaks or set()
    for left, right in zip(ordered[:-1], ordered[1:]):
        left_q, right_q = _q(left[1]), _q(right[1])
        if left_q is not None and right_q is not None:
            distance = float(math.hypot(right_q[0] - left_q[0], right_q[1] - left_q[1]))
            if np.isfinite(distance) and distance > _EPS:
                distances.append(distance)
    # Tracers commonly retain only profile-supported ridge samples, so their
    # tangential spacing can be much larger than the normal pixel step.  Use a
    # robust observed-neighbour scale to avoid declaring every sparse but
    # connected arc a gap; a true masked gap remains visible as a large outlier.
    typical_spacing = float(np.median(distances)) if distances else 0.0
    spacing_threshold = max(2.5 * typical_spacing, 4.0 * max(pads.values(), default=_EPS))
    groups: list[list[tuple[int, Mapping[str, Any]]]] = [[ordered[0]]]
    gaps: list[dict[str, Any]] = []
    for left, right in zip(ordered[:-1], ordered[1:]):
        left_q, right_q = _q(left[1]), _q(right[1])
        if left_q is None or right_q is None:
            distance = float("nan")
        else:
            distance = float(math.hypot(right_q[0] - left_q[0], right_q[1] - left_q[1]))
        left_id, right_id = _point_id(left[1], left[0]), _point_id(right[1], right[0])
        threshold = max(
            _gap_threshold(float(pads[left_id]), float(pads[right_id])),
            spacing_threshold,
        )
        forced = (str(left_id), str(right_id)) in forced_breaks
        disconnected = forced or not np.isfinite(distance) or distance > threshold
        if disconnected:
            gaps.append(
                {
                    "before_point_id": left_id,
                    "after_point_id": right_id,
                    "distance_q": distance if np.isfinite(distance) else None,
                    "threshold_q": float(threshold),
                    "reason": "rejected_or_ambiguous_interior_observation" if forced else "observed_neighbor_gap",
                }
            )
            groups.append([right])
        else:
            groups[-1].append(right)
    return groups, gaps


def _split_by_declared_gaps(
    ordered: list[tuple[int, Mapping[str, Any]]],
    arc: Mapping[str, Any],
) -> tuple[list[list[tuple[int, Mapping[str, Any]]]], list[dict[str, Any]]] | None:
    raw_gaps = arc.get("support_gaps", arc.get("gaps"))
    if not isinstance(raw_gaps, Sequence) or isinstance(raw_gaps, (str, bytes)):
        return None
    break_after = {
        str(item.get("before_point_id"))
        for item in raw_gaps
        if isinstance(item, Mapping) and item.get("before_point_id") is not None
    }
    if not break_after:
        return None
    groups: list[list[tuple[int, Mapping[str, Any]]]] = [[ordered[0]]] if ordered else []
    gaps: list[dict[str, Any]] = []
    for left, right in zip(ordered[:-1], ordered[1:]):
        left_id = _point_id(left[1], left[0])
        if str(left_id) in break_after:
            groups.append([right])
            matched = next(
                item for item in raw_gaps
                if isinstance(item, Mapping) and str(item.get("before_point_id")) == str(left_id)
            )
            gaps.append(dict(matched))
        else:
            groups[-1].append(right)
    return groups, gaps


def _split_groups_by_forced_gaps(
    groups: list[list[tuple[int, Mapping[str, Any]]]],
    pads: Mapping[Any, float],
    forced_breaks: set[tuple[str, str]],
) -> tuple[list[list[tuple[int, Mapping[str, Any]]]], list[dict[str, Any]]]:
    if not forced_breaks:
        return groups, []
    split: list[list[tuple[int, Mapping[str, Any]]]] = []
    gaps: list[dict[str, Any]] = []
    for group in groups:
        if not group:
            continue
        current = [group[0]]
        for left, right in zip(group[:-1], group[1:]):
            left_id = _point_id(left[1], left[0])
            right_id = _point_id(right[1], right[0])
            if (str(left_id), str(right_id)) in forced_breaks:
                left_q, right_q = _q(left[1]), _q(right[1])
                distance = math.hypot(right_q[0] - left_q[0], right_q[1] - left_q[1]) if left_q and right_q else float("nan")
                gaps.append(
                    {
                        "before_point_id": left_id,
                        "after_point_id": right_id,
                        "distance_q": float(distance) if np.isfinite(distance) else None,
                        "threshold_q": _gap_threshold(pads[left_id], pads[right_id]),
                        "reason": "rejected_or_ambiguous_interior_observation",
                    }
                )
                split.append(current)
                current = [right]
            else:
                current.append(right)
        split.append(current)
    return split, gaps


def _tangent_for_segment(
    left: Mapping[str, Any], right: Mapping[str, Any],
    displacement: np.ndarray,
) -> np.ndarray:
    norm = float(np.linalg.norm(displacement))
    left_tangent = _unit_vector(left, ("tangent_qx", "tangent_qy"))
    right_tangent = _unit_vector(right, ("tangent_qx", "tangent_qy"))
    if left_tangent is not None and right_tangent is not None:
        tangent = left_tangent + right_tangent
        tangent_norm = float(np.linalg.norm(tangent))
        tangent = tangent / tangent_norm if tangent_norm > _EPS else left_tangent
    elif left_tangent is not None:
        tangent = left_tangent
    elif right_tangent is not None:
        tangent = right_tangent
    elif np.isfinite(norm) and norm > _EPS:
        tangent = displacement / norm
    else:
        tangent = np.asarray([1.0, 0.0], dtype=float)
    # Keep the orientation along the ordered observed path.  A rectangle is
    # invariant to a sign flip, but deterministic orientation matters for
    # audit and for interpreting tangent bounds.
    if np.isfinite(norm) and norm > _EPS and float(np.dot(tangent, displacement)) < 0:
        tangent = -tangent
    return tangent


def _segment_rectangle(
    left_item: tuple[int, Mapping[str, Any]],
    right_item: tuple[int, Mapping[str, Any]],
    pads: Mapping[Any, float],
    component_id: int,
    segment_index: int,
    *,
    endpoint_start: bool,
    endpoint_end: bool,
) -> dict[str, Any] | None:
    left_index, left = left_item
    right_index, right = right_item
    left_q, right_q = _q(left), _q(right)
    if left_q is None or right_q is None:
        return None
    origin = np.asarray(left_q, dtype=float)
    displacement = np.asarray(right_q, dtype=float) - origin
    tangent = _tangent_for_segment(left, right, displacement)
    normal = np.asarray([-tangent[1], tangent[0]], dtype=float)
    measured_normal = _unit_vector(left, ("normal_qx", "normal_qy"))
    if measured_normal is not None and float(np.dot(normal, measured_normal)) < 0:
        normal = -normal
    distance = float(np.dot(displacement, tangent))
    if not np.isfinite(distance) or distance < 0:
        tangent = -tangent
        normal = -normal
        distance = float(np.dot(displacement, tangent))
    left_pad = float(pads[_point_id(left, left_index)])
    right_pad = float(pads[_point_id(right, right_index)])
    normal_coordinates = (0.0, float(np.dot(displacement, normal)))
    normal_pad = max(left_pad, right_pad)
    normal_min = min(normal_coordinates) - normal_pad
    normal_max = max(normal_coordinates) + normal_pad
    return {
        "component_id": int(component_id),
        "segment_index": int(segment_index),
        "point_ids": [_point_id(left, left_index), _point_id(right, right_index)],
        "frame_origin_q": [float(origin[0]), float(origin[1])],
        "tangent_q": [float(tangent[0]), float(tangent[1])],
        "normal_q": [float(normal[0]), float(normal[1])],
        "tangent_bounds": [-left_pad, float(distance + right_pad)],
        "normal_bounds": [float(normal_min), float(normal_max)],
        "tangent_min": float(-left_pad),
        "tangent_max": float(distance + right_pad),
        "normal_min": float(normal_min),
        "normal_max": float(normal_max),
        "support_padding_q": float(max(left_pad, right_pad)),
        "endpoint_flags": {"start": bool(endpoint_start), "end": bool(endpoint_end)},
        "is_endpoint_start": bool(endpoint_start),
        "is_endpoint_end": bool(endpoint_end),
    }


def _point_rectangle(
    item: tuple[int, Mapping[str, Any]], pad: float, component_id: int,
    *, endpoint_start: bool = True, endpoint_end: bool = True,
) -> dict[str, Any] | None:
    index, point = item
    q = _q(point)
    if q is None:
        return None
    tangent = _unit_vector(point, ("tangent_qx", "tangent_qy"))
    if tangent is None:
        tangent = np.asarray([1.0, 0.0], dtype=float)
    normal = np.asarray([-tangent[1], tangent[0]], dtype=float)
    measured_normal = _unit_vector(point, ("normal_qx", "normal_qy"))
    if measured_normal is not None and float(np.dot(normal, measured_normal)) < 0:
        normal = -normal
    point_id = _point_id(point, index)
    return {
        "component_id": int(component_id),
        "segment_index": 0,
        "point_ids": [point_id],
        "frame_origin_q": [float(q[0]), float(q[1])],
        "tangent_q": [float(tangent[0]), float(tangent[1])],
        "normal_q": [float(normal[0]), float(normal[1])],
        "tangent_bounds": [-float(pad), float(pad)],
        "normal_bounds": [-float(pad), float(pad)],
        "tangent_min": -float(pad),
        "tangent_max": float(pad),
        "normal_min": -float(pad),
        "normal_max": float(pad),
        "support_padding_q": float(pad),
        "endpoint_flags": {"start": bool(endpoint_start), "end": bool(endpoint_end)},
        "is_endpoint_start": bool(endpoint_start),
        "is_endpoint_end": bool(endpoint_end),
    }


def _component_record(
    component_id: int,
    members: list[tuple[int, Mapping[str, Any]]],
    pads: Mapping[Any, float],
) -> dict[str, Any] | None:
    if not members:
        return None
    rectangles: list[dict[str, Any]] = []
    if len(members) == 1:
        rectangle = _point_rectangle(members[0], float(pads[_point_id(members[0][1], members[0][0])]), component_id)
        if rectangle is not None:
            rectangles.append(rectangle)
    else:
        for segment_index, (left, right) in enumerate(zip(members[:-1], members[1:])):
            rectangle = _segment_rectangle(
                left,
                right,
                pads,
                component_id,
                segment_index,
                endpoint_start=segment_index == 0,
                endpoint_end=segment_index == len(members) - 2,
            )
            if rectangle is not None:
                rectangles.append(rectangle)
        # Explicitly retain both measured endpoints when a degenerate or
        # duplicate neighbour made a segment unavailable.
        if not rectangles:
            for index, item in enumerate((members[0], members[-1])):
                rectangle = _point_rectangle(
                    item,
                    float(pads[_point_id(item[1], item[0])]),
                    component_id,
                    endpoint_start=index == 0,
                    endpoint_end=index == 1,
                )
                if rectangle is not None:
                    rectangles.append(rectangle)
    if not rectangles:
        return None
    point_ids = [_point_id(point, index) for index, point in members]
    pad_values = [float(pads[point_id]) for point_id in point_ids]
    return {
        "component_id": int(component_id),
        "point_ids": point_ids,
        "rectangles": rectangles,
        "n_points": int(len(point_ids)),
        "support_padding_q": float(max(pad_values)),
        "endpoint_flags": {
            "start": True,
            "end": True,
            "point_ids": [point_ids[0], point_ids[-1]],
        },
    }


def _arc_support_record(
    arc: Mapping[str, Any],
    members: list[tuple[int, Mapping[str, Any]]],
) -> dict[str, Any]:
    arc_id = _json_scalar(arc.get("arc_id", -1))
    ordered_members = _ordered_members(members, arc)
    accepted = [item for item in ordered_members if bool(item[1].get("accepted", item[1].get("valid", False)))]
    branch_values = sorted({int(point.get("branch_id", -1)) for _, point in accepted if isinstance(point.get("branch_id"), Integral) and not isinstance(point.get("branch_id"), (bool, np.bool_))})
    side_values = sorted({str(point.get("side", "unknown")) for _, point in accepted})
    identity_ok = len(branch_values) == 1 and branch_values[0] in (0, 1) and len(side_values) == 1 and side_values[0] in {"upper", "lower"}
    record: dict[str, Any] = {
        "arc_id": arc_id,
        "support_version": SUPPORT_VERSION,
        "support_frozen": False,
        "support_status": "unavailable",
        "branch_id": branch_values[0] if len(branch_values) == 1 else None,
        "side": side_values[0] if len(side_values) == 1 else None,
        "point_ids": [_point_id(point, index) for index, point in accepted],
        "support_components": [],
        "support_component_count": 0,
        "support_gaps": [],
        "support_gap_count": 0,
        "endpoint_flags": {"start": None, "end": None, "point_ids": []},
        "short_arc": bool(arc.get("short_arc", "short_arc" in arc.get("topology_flags", []))),
        "padding_provenance": {
            "formula": _PAD_FORMULA,
            "role": _PAD_ROLE,
            "multiplier": _PAD_MULTIPLIER,
            "units": "q",
            "values_by_point": {},
        },
    }
    if not accepted:
        record["support_status"] = "unavailable_no_accepted_observed_points"
        return record
    if not identity_ok:
        record["support_status"] = "ineligible_mixed_labels"
        record["reason"] = "one_resolved_branch_and_side_required"
        return record
    ordered = accepted
    forced_breaks = _forced_rejected_gaps(ordered_members, arc)
    pads: dict[Any, float] = {}
    for index, point in ordered:
        point_id = _point_id(point, index)
        pad, provenance = _positive_scale(point, arc)
        if pad is None:
            record["support_status"] = "unavailable_missing_observed_scale"
            record["reason"] = "finite_localization_or_sampling_scale_required"
            record["padding_provenance"]["values_by_point"][str(point_id)] = provenance
            return record
        # The measured scale is a centre-localization/sampling scale, not a
        # 68% confidence interval.  A fixed two-scale corridor keeps noisy
        # observed ridges inside the finite support while remaining tied to
        # the source measurement and independent of any fitted ellipse.
        corridor_pad = float(_PAD_MULTIPLIER * pad)
        pads[point_id] = corridor_pad
        record["padding_provenance"]["values_by_point"][str(point_id)] = {
            **provenance,
            "measurement_scale_q": float(pad),
            "support_padding_q": corridor_pad,
            "support_padding_multiplier": _PAD_MULTIPLIER,
        }
    explicit_groups = _explicit_component_groups(ordered, arc)
    declared_gap_split = _split_by_declared_gaps(ordered, arc) if explicit_groups is None else None
    if explicit_groups is None and declared_gap_split is not None:
        groups, gaps = declared_gap_split
        groups, forced_gaps = _split_groups_by_forced_gaps(groups, pads, forced_breaks)
        gaps.extend(forced_gaps)
    elif explicit_groups is None:
        groups, gaps = _split_by_observed_gaps(ordered, pads, forced_breaks)
    else:
        groups = explicit_groups
        gaps = []
        for left_group, right_group in zip(groups[:-1], groups[1:]):
            if left_group and right_group:
                left_id = _point_id(left_group[-1][1], left_group[-1][0])
                right_id = _point_id(right_group[0][1], right_group[0][0])
                left_q, right_q = _q(left_group[-1][1]), _q(right_group[0][1])
                distance = math.hypot(right_q[0] - left_q[0], right_q[1] - left_q[1]) if left_q and right_q else float("nan")
                gaps.append({
                    "before_point_id": left_id,
                    "after_point_id": right_id,
                    "distance_q": float(distance) if np.isfinite(distance) else None,
                    "threshold_q": _gap_threshold(pads[left_id], pads[right_id]),
                    "reason": "caller_declared_disconnected_support",
                })
        groups, forced_gaps = _split_groups_by_forced_gaps(groups, pads, forced_breaks)
        gaps.extend(forced_gaps)
    components: list[dict[str, Any]] = []
    for component_id, group in enumerate(groups):
        component = _component_record(component_id, group, pads)
        if component is not None:
            components.append(component)
    if not components:
        record["support_status"] = "unavailable_no_finite_observed_segments"
        record["reason"] = "finite_observed_neighbour_segments_required"
        return record
    record.update(
        {
            "support_frozen": True,
            "support_status": "frozen",
            "support_components": components,
            "support_gaps": gaps,
            "support_gap_count": len(gaps),
            "component_count": len(components),
            "support_component_count": len(components),
            "support_padding_q": float(max(pads.values())),
            "endpoint_flags": {
                "start": True,
                "end": True,
                "point_ids": [record["point_ids"][0], record["point_ids"][-1]],
            },
        }
    )
    return record


def _arc_lookup(arcs: Any) -> tuple[list[Mapping[str, Any]], dict[Any, Mapping[str, Any]]]:
    if isinstance(arcs, Mapping):
        values = list(arcs.values())
    elif isinstance(arcs, Sequence) and not isinstance(arcs, (str, bytes)):
        values = [item for item in arcs if isinstance(item, Mapping)]
    else:
        values = []
    lookup: dict[Any, Mapping[str, Any]] = {}
    for arc in values:
        arc_id = arc.get("arc_id", arc.get("id"))
        if arc_id is not None:
            lookup[arc_id] = arc
            lookup[str(arc_id)] = arc
    return values, lookup


def freeze_observed_support(
    points: list[dict[str, Any]],
    arcs: Any,
    *,
    summary_only: bool = False,
    cancel_event: Any = None,
) -> dict[str, Any]:
    """Freeze finite observed support and attach it to point/arc records.

    The helper is intentionally called after profile refinement, seed edits,
    and point exclusions.  Only points that are currently accepted and have a
    resolved branch/side contribute rectangles.  Rejected and mixed topology
    records remain in the trace with explicit support status.
    """

    if not isinstance(points, list):
        raise TypeError("points must be a mutable list of mappings")

    def check_cancelled() -> None:
        raise_if_cancelled(cancel_event, "observed-support-freeze")

    arc_values, _ = _arc_lookup(arcs)
    points_by_arc: dict[Any, list[tuple[int, Mapping[str, Any]]]] = {}
    for index, point in enumerate(points):
        if not isinstance(point, Mapping):
            continue
        arc_id = point.get("arc_id", -1)
        points_by_arc.setdefault(arc_id, []).append((index, point))
        points_by_arc.setdefault(str(arc_id), points_by_arc[arc_id])
    records: list[dict[str, Any]] = []
    seen_arc_keys: set[str] = set()
    for raw_arc in arc_values:
        check_cancelled()
        arc_id = raw_arc.get("arc_id", raw_arc.get("id", -1))
        key = str(arc_id)
        if key in seen_arc_keys:
            continue
        seen_arc_keys.add(key)
        members = points_by_arc.get(arc_id, points_by_arc.get(key, []))
        record = _arc_support_record(raw_arc, members)
        if isinstance(raw_arc, dict):
            raw_arc.update(record)
        records.append(record)
    # Preserve point records even if a caller forgot to include an arc object.
    for arc_id, members in list(points_by_arc.items()):
        key = str(arc_id)
        if key in seen_arc_keys:
            continue
        raw_arc = {"arc_id": arc_id}
        record = _arc_support_record(raw_arc, members)
        records.append(record)
        seen_arc_keys.add(key)
    by_arc = {str(record["arc_id"]): record for record in records}
    rectangles_by_arc_point: dict[str, dict[str, list[tuple[Any, dict[str, Any]]]]] = {}
    for record in records:
        if record.get("support_status") != "frozen":
            continue
        by_point: dict[str, list[tuple[Any, dict[str, Any]]]] = {}
        for component in record.get("support_components", []):
            component_id = component.get("component_id")
            for rectangle in component.get("rectangles", []):
                for point_id in rectangle.get("point_ids", []):
                    by_point.setdefault(str(point_id), []).append((component_id, rectangle))
        rectangles_by_arc_point[str(record["arc_id"])] = by_point
    for index, point in enumerate(points):
        if index % 512 == 0:
            check_cancelled()
        if not isinstance(point, dict):
            continue
        arc_record = by_arc.get(str(point.get("arc_id", -1)))
        if arc_record is None:
            point.update({"support_version": SUPPORT_VERSION, "support_frozen": False, "support_status": "unavailable_no_arc"})
            continue
        point["support_version"] = SUPPORT_VERSION
        point["support_frozen"] = bool(arc_record.get("support_frozen", False))
        point["support_status"] = str(arc_record.get("support_status", "unavailable"))
        point["support_arc_id"] = _json_scalar(arc_record.get("arc_id"))
        point["support_component_count"] = int(len(arc_record.get("support_components", [])))
        point["support_gap_count"] = int(arc_record.get("support_gap_count", 0))
        point["support_padding_q"] = arc_record.get("support_padding_q")
        if point["support_frozen"] and bool(point.get("accepted", point.get("valid", False))):
            point_id = _point_id(point, index)
            # Keep point-local support small.  The arc record retains the
            # canonical complete corridor; each fit row only needs the one or
            # two observed-neighbour rectangles touching its own point.
            grouped_local: dict[str, dict[str, Any]] = {}
            for component_id, rectangle in rectangles_by_arc_point.get(str(arc_record["arc_id"]), {}).get(str(point_id), []):
                component = grouped_local.setdefault(
                    str(component_id),
                    {"component_id": component_id, "rectangles": [], "point_ids": [point_id]},
                )
                component["rectangles"].append(rectangle)
            local_components = list(grouped_local.values())
            point["support_components"] = local_components
            point["support_rectangles"] = [
                rectangle
                for component in local_components
                for rectangle in component.get("rectangles", [])
            ]
            point["support_component_ids"] = [component.get("component_id") for component in local_components]
            point["observed_support"] = {
                "version": SUPPORT_VERSION,
                "arc_id": _json_scalar(arc_record.get("arc_id")),
                "status": point["support_status"],
                "components": local_components,
                "gap_count": int(arc_record.get("support_gap_count", 0)),
            }
        else:
            point.setdefault("support_components", [])
            point["support_component_ids"] = []
            point["observed_support"] = {
                "version": SUPPORT_VERSION,
                "arc_id": _json_scalar(arc_record.get("arc_id")),
                "status": point["support_status"],
                "components": [],
                "gap_count": int(arc_record.get("support_gap_count", 0)),
            }
    statuses = [str(record.get("support_status", "unavailable")) for record in records]
    all_frozen = bool(records) and all(status == "frozen" for status in statuses)
    any_frozen = any(status == "frozen" for status in statuses)
    overall = "frozen" if all_frozen else "partial" if any_frozen else "unavailable"
    envelope = {
        "support_version": SUPPORT_VERSION,
        "version": SUPPORT_VERSION,
        "support_frozen": all_frozen,
        "frozen": all_frozen,
        "support_status": overall,
        "status": overall,
        "arcs": records,
        "arc_support": {str(record["arc_id"]): record for record in records},
        "support_registry": {str(record["arc_id"]): record for record in records},
        "summary": {
            "n_arcs": len(records),
            "n_frozen_arcs": sum(status == "frozen" for status in statuses),
            "n_unavailable_arcs": sum(status != "frozen" for status in statuses),
            "n_support_components": sum(len(record.get("support_components", [])) for record in records),
            "n_support_gaps": sum(len(record.get("support_gaps", [])) for record in records),
        },
    }
    if summary_only:
        envelope["arcs"] = []
        envelope["arc_support"] = {}
        envelope["support_registry"] = {}
    return envelope


__all__ = ["SUPPORT_VERSION", "freeze_observed_support"]
