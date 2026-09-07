"""Scientific source adaptation and deterministic lamellar schematic geometry."""

from __future__ import annotations
import os
from collections.abc import Mapping
from typing import Any
import numpy as np

from .lamellar_models import LamellarScene, LamellarSettings, _empty_scene
from .lamellar_utils import (
    _finite,
    _finite_positive,
    _is_physical_unit,
    _normalise_q_unit,
    _q_scale_to_nm,
    _read,
)
from .lamellar_adapter import (
    _ellipse_components,
    _ellipse_parameter_records,
    _ellipse_payload,
    _direction_for_group,
    _execution_failure,
    _group_populations,
    _metadata_base,
    _period_record,
    _q_unit_from_source,
    _radial_peaks,
    _source_identity,
    _source_mapping,
    _stable_seed,
)

_MISSING = object()
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
_BLUE = np.asarray((0.20, 0.45, 0.72, 0.82), dtype=float)
_ORANGE = np.asarray((0.92, 0.47, 0.18, 0.82), dtype=float)


def build_lamellar_scene(
    source: Any, settings: Mapping[str, Any] | LamellarSettings | None = None
) -> LamellarScene:
    resolved = LamellarSettings.from_mapping(settings)
    root = _source_mapping(source)
    identity = _source_identity(source, root)
    raw_peaks = _radial_peaks(root)
    q_unit = _q_unit_from_source(root, raw_peaks)
    draw_axis = _finite(_read(root, ("draw_axis_deg",), _MISSING))
    if draw_axis is None:
        observables = _read(root, ("observables", "measurements"), None)
        draw_axis = (
            _finite(
                _read(
                    observables,
                    ("draw_axis_deg",),
                    _read(_read(root, ("analysis",), None), ("draw_axis_deg",), 90.0),
                )
            )
            or 90.0
        )
    source_status = str(_read(root, ("status", "measurement_status"), "") or "").lower()
    metadata_root = _read(root, ("metadata",), {})
    if not source_status and isinstance(metadata_root, Mapping):
        source_status = str(
            metadata_root.get("status", metadata_root.get("measurement_status", ""))
            or ""
        ).lower()
    execution_failure = _execution_failure(root)
    stale = source_status == "stale" or bool(_read(root, ("stale",), False))
    assumptions = [
        "Measured q-angle is used as the schematic lamellar-normal direction after draw-axis display rotation.",
        "Opposite q directions represent one unoriented normal; original branch identifiers are retained.",
        "Box dimensions are explicit ratios and do not identify a unique three-dimensional structure.",
    ]
    flags: list[str] = []
    if execution_failure is not None and resolved.period_source != "manual":
        metadata = _metadata_base(
            resolved,
            identity,
            status="unavailable",
            available=False,
            message=f"Source execution status is {execution_failure}; measured peaks are not treated as current.",
            draw_axis_deg=draw_axis,
            reference_period=resolved.reference_period or 1.0,
            assumptions=assumptions,
            q_unit=q_unit,
            flags=("source_execution_failed",),
        )
        return _empty_scene(resolved, metadata)
    if stale and resolved.period_source != "manual":
        metadata = _metadata_base(
            resolved,
            identity,
            status="stale",
            available=False,
            message="The source result is marked stale; no current lamellar scene is available.",
            draw_axis_deg=draw_axis,
            reference_period=resolved.reference_period or 1.0,
            assumptions=assumptions,
            q_unit=q_unit,
            flags=("stale_source",),
        )
        return _empty_scene(resolved, metadata)
    populations: list[dict[str, Any]] = []
    parameter_sources: list[dict[str, Any]] = []
    period_unit = "relative"
    length_unit = "relative"
    ellipse_for_metadata = _ellipse_payload(root)
    parameter_sources.extend(_ellipse_parameter_records(ellipse_for_metadata))
    if resolved.period_source == "manual":
        period = float(resolved.manual_period)
        angle = float(resolved.manual_angle_deg)
        branch = 0 if resolved.selected_branch < 0 else int(resolved.selected_branch)
        populations = [
            {
                "branch_id": branch,
                "period": period,
                "angle_deg": angle,
                "status": "manual",
            }
        ]
        parameter_sources.append(
            _period_record(
                "period",
                period,
                resolved.manual_unit,
                "manual",
                "manual",
                "explicit manual assumption",
            )
        )
        period_unit = resolved.manual_unit
        length_unit = resolved.manual_unit
        assumptions.append(
            "Period and angle are explicit manual assumptions supplied by the caller."
        )
    elif not raw_peaks:
        parameter_sources.append(
            _period_record(
                "period",
                None,
                "relative",
                "lobe_radial_peaks",
                "unavailable",
                "no valid radial lobe peaks; annulus q is not accepted",
            )
        )
        metadata = _metadata_base(
            resolved,
            identity,
            status="unavailable",
            available=False,
            message="No valid lobe_radial_peaks with a positive radial q_star were found; annulus/azimuthal q is not used.",
            draw_axis_deg=draw_axis,
            reference_period=resolved.reference_period or 1.0,
            assumptions=assumptions,
            parameter_sources=parameter_sources,
            q_unit=q_unit,
            flags=("no_valid_lobe_radial_peaks", "spacing_unavailable_unknown_q_unit"),
        )
        return _empty_scene(resolved, metadata)
    else:
        groups = _group_populations(raw_peaks)
        reference_q_nm = next(
            (
                float(item["q_nm_inv"])
                for item in raw_peaks
                if item["q_nm_inv"] is not None
            ),
            None,
        )
        reference_q_native = float(
            next(
                (item["q"] for item in raw_peaks if item["q_nm_inv"] is None),
                raw_peaks[0]["q"],
            )
        )
        for branch, group in groups:
            angle, inconsistent_angle = _direction_for_group(group)
            first = group[0]
            q_nm_values = [
                item["q_nm_inv"] for item in group if item["q_nm_inv"] is not None
            ]
            q_nm = float(q_nm_values[0]) if q_nm_values else None
            inconsistent_q = False
            uncertain_q = False
            if len(q_nm_values) == len(group):
                inconsistent_q = any(
                    abs(float(item) - q_nm) > max(1e-12, 0.05 * q_nm)
                    for item in q_nm_values[1:]
                )
            else:
                native_units = {_normalise_q_unit(item["q_unit"]) for item in group}
                if len(native_units) == 1:
                    native_q = float(first["q"])
                    inconsistent_q = any(
                        abs(float(item["q"]) - native_q) > max(1e-12, 0.05 * native_q)
                        for item in group[1:]
                    )
                else:
                    uncertain_q = len(group) > 1
            if q_nm is not None:
                period = float(2.0 * np.pi / q_nm)
                unit = "nm"
                period_status = (
                    "candidate"
                    if any(
                        item["status"] in {"candidate", "provisional", "undetermined"}
                        for item in group
                    )
                    else "available"
                )
                period_source_name = "lobe_radial_peaks"
                period_reason = (
                    "2π/q from the first valid radial lobe peak in this branch"
                )
                length_unit = "nm"
                period_unit = "nm"
            else:
                q_value = float(first["q"])
                base = (
                    float(resolved.reference_period)
                    if resolved.reference_period is not None
                    else 1.0
                )
                period = float(base * reference_q_native / q_value)
                unit = "relative"
                period_status = (
                    "candidate"
                    if any(
                        item["status"] in {"candidate", "provisional", "undetermined"}
                        for item in group
                    )
                    else "available"
                )
                period_source_name = "lobe_radial_peaks"
                period_reason = f"normalized to explicit reference q={reference_q_native:g} ({first['q_unit'] or 'unknown'})"
                length_unit = "relative"
                flags.append("spacing_unavailable_unknown_q_unit")
                assumptions.append(
                    "Unknown q units are displayed on an explicit relative scale normalized to the retained reference q."
                )
            if uncertain_q:
                flags.append("mixed_q_unit_support")
                period_reason += "; opposite q units are mixed and cannot be compared"
                period_status = "candidate"
            if inconsistent_angle or inconsistent_q:
                flags.append("inconsistent_opposite_support")
                period_reason += f"; opposite q values={[float(item['q']) for item in group]} are inconsistent and were not mirrored or fabricated"
                period_status = (
                    "candidate" if period_status == "available" else period_status
                )
            if any(
                item["status"] in {"candidate", "provisional", "undetermined"}
                for item in group
            ):
                flags.append("candidate_radial_support")
            populations.append(
                {
                    "branch_id": int(branch),
                    "period": period,
                    "angle_deg": angle,
                    "status": "candidate"
                    if period_status == "candidate"
                    else "schematic",
                    "support_indices": [int(item["index"]) for item in group],
                    "support_q_values": [float(item["q"]) for item in group],
                    "support_q_units": [str(item["q_unit"]) for item in group],
                }
            )
            parameter_sources.append(
                _period_record(
                    "period",
                    None if period_status == "candidate" else period,
                    unit,
                    period_source_name,
                    period_status,
                    period_reason,
                    candidate_value=period if period_status == "candidate" else None,
                )
            )
            q_record = q_nm if q_nm is not None else float(first["q"])
            parameter_sources.append(
                _period_record(
                    "q_star",
                    None if period_status == "candidate" else q_record,
                    "nm^-1" if q_nm is not None else str(first["q_unit"]),
                    "lobe_radial_peaks",
                    "available" if period_status == "available" else period_status,
                    "retained radial peak",
                    candidate_value=q_record if period_status == "candidate" else None,
                )
            )
        if resolved.period_source == "ellipse":
            ellipse = _ellipse_payload(root)
            b, candidate_b, b_status, center, ellipse_unit, existing_period = (
                _ellipse_components(ellipse)
            )
            a_hint = _finite(
                _read(
                    ellipse,
                    ("a", "semi_major", "major_axis"),
                    _read(
                        _read(ellipse, ("parameters", "quantitative_parameters"), {}),
                        ("a", "semi_major", "major_axis"),
                        0.0,
                    ),
                )
            )
            if center is None:
                ellipse_status = "unavailable"
                ellipse_reason = "ellipse centre is not available, so the canonical origin assumption cannot be checked"
                ellipse_period = None
            elif float(np.hypot(*center)) > 1e-8 * max(
                1.0, abs(float(a_hint or 0.0)), abs(float(b or 0.0))
            ):
                ellipse_status = "unavailable"
                ellipse_reason = (
                    "nonzero ellipse centre makes the origin-centred period unavailable"
                )
                ellipse_period = None
                flags.append("spacing_unavailable_nonzero_center")
            elif existing_period is not None and _is_physical_unit(ellipse_unit):
                ellipse_status = "candidate" if candidate_b else "available"
                ellipse_period = float(existing_period)
                ellipse_reason = "existing canonical Ln_from_minor_axis_nm from an origin-centred ellipse"
            elif b is None or b <= 0.0:
                ellipse_status = "unavailable"
                ellipse_reason = "ellipse minor axis b is missing or nonpositive"
                ellipse_period = None
            elif _q_scale_to_nm(ellipse_unit) is not None:
                ellipse_status = "candidate" if candidate_b else "available"
                ellipse_period = float(
                    2.0 * np.pi / (float(b) * float(_q_scale_to_nm(ellipse_unit)))
                )
                ellipse_reason = (
                    "2π/b from declared physical q unit and origin-centred ellipse"
                )
            else:
                reference_for_ellipse = (
                    reference_q_nm if reference_q_nm is not None else reference_q_native
                )
                base = (
                    float(resolved.reference_period)
                    if resolved.reference_period is not None
                    else 1.0
                )
                ellipse_period = float(base * reference_for_ellipse / float(b))
                ellipse_status = "candidate" if candidate_b else "available"
                ellipse_reason = "ellipse b normalized to explicit reference q because q unit is unknown"
                flags.append("spacing_unavailable_unknown_q_unit")
                assumptions.append(
                    "Ellipse period remains relative when q units are unknown; no nm value is assigned."
                )
            parameter_sources.append(
                _period_record(
                    "period_from_ellipse_minor_axis",
                    None if candidate_b or ellipse_period is None else ellipse_period,
                    "nm" if _q_scale_to_nm(ellipse_unit) is not None else "relative",
                    "ellipse_fit",
                    ellipse_status,
                    ellipse_reason,
                    candidate_value=ellipse_period if candidate_b else None,
                )
            )
            if ellipse_period is None:
                for population in populations:
                    population["period"] = None
                    population["status"] = "unavailable"
                metadata = _metadata_base(
                    resolved,
                    identity,
                    status="unavailable",
                    available=False,
                    message=ellipse_reason,
                    draw_axis_deg=draw_axis,
                    reference_period=resolved.reference_period or 1.0,
                    assumptions=assumptions,
                    parameter_sources=parameter_sources,
                    populations=populations,
                    q_unit=q_unit,
                    flags=flags,
                )
                return _empty_scene(resolved, metadata)
            for population in populations:
                population["period"] = float(ellipse_period)
                population["status"] = "candidate" if candidate_b else "schematic"
            period_unit = (
                "nm" if _q_scale_to_nm(ellipse_unit) is not None else "relative"
            )
            length_unit = period_unit
            if candidate_b:
                flags.append("candidate_ellipse_parameter")
                assumptions.append(
                    "Candidate ellipse b may drive a provisional schematic, but formal metadata values remain null."
                )
    selected = [
        population
        for population in populations
        if resolved.selected_branch < 0
        or int(population["branch_id"]) == int(resolved.selected_branch)
    ]
    if not selected:
        metadata = _metadata_base(
            resolved,
            identity,
            status="unavailable",
            available=False,
            message=f"Selected branch {resolved.selected_branch} has no supported lamellar population.",
            draw_axis_deg=draw_axis,
            reference_period=resolved.reference_period or 1.0,
            assumptions=assumptions,
            parameter_sources=parameter_sources,
            populations=populations,
            q_unit=q_unit,
            flags=(*flags, "selected_branch_unavailable"),
        )
        return _empty_scene(resolved, metadata)
    if (
        len(selected) < 2
        and resolved.selected_branch < 0
        and resolved.period_source != "manual"
    ):
        flags.append("single_branch_supported")
    periods = [
        float(item["period"])
        for item in selected
        if _finite_positive(item.get("period")) is not None
    ]
    if not periods:
        metadata = _metadata_base(
            resolved,
            identity,
            status="unavailable",
            available=False,
            message="No finite positive period remained after source adaptation.",
            draw_axis_deg=draw_axis,
            reference_period=resolved.reference_period or 1.0,
            assumptions=assumptions,
            parameter_sources=parameter_sources,
            populations=populations,
            q_unit=q_unit,
            flags=flags,
        )
        return _empty_scene(resolved, metadata)
    layout_period = (
        float(resolved.reference_period)
        if resolved.reference_period is not None
        else float(max(periods))
    )
    n_layers = int(resolved.layer_count)
    centres: list[np.ndarray] = []
    sizes: list[np.ndarray] = []
    orientations: list[np.ndarray] = []
    vertices: list[np.ndarray] = []
    stack_ids: list[int] = []
    branch_ids: list[int] = []
    colors: list[np.ndarray] = []
    # Separate whole packets, including their stack length and optional slip.
    packet_width = resolved.width_ratio + (n_layers - 1) * abs(
        resolved.lateral_shift_ratio
    )
    packet_length = n_layers - 1 + resolved.thickness_ratio
    spacing = layout_period * (
        float(np.linalg.norm((packet_width, resolved.depth_ratio, packet_length))) + 1.0
    )
    slot_entries: list[tuple[int, int, int]] = []
    if resolved.mode == "single":
        if resolved.stack_count > 0:
            for population_index, population in enumerate(selected):
                branch = int(population["branch_id"])
                slot_entries.append(
                    (
                        branch if branch in {0, 1} else population_index,
                        population_index,
                        0,
                    )
                )
    elif resolved.stack_count > 0:
        has_two = len(selected) >= 2
        only_branch = int(selected[0]["branch_id"]) if len(selected) == 1 else None
        for slot in range(int(resolved.stack_count)):
            if has_two:
                population_index = slot % len(selected)
            elif only_branch in {0, 1} and resolved.period_source != "manual":
                if slot % 2 != only_branch:
                    continue
                population_index = 0
            else:
                population_index = 0
            slot_entries.append((slot, population_index, slot))
        assumptions.append(
            "Multi-stack slots use a fixed compact grid; missing branch families leave their slots empty."
        )
    total_slots = max(1, int(resolved.stack_count))
    grid_columns = max(1, int(np.ceil(np.sqrt(total_slots))))
    grid_rows = int(np.ceil(total_slots / grid_columns))
    for stack_id, population_index, slot in slot_entries:
        population = selected[population_index]
        branch = int(population["branch_id"])
        period = float(population["period"])
        angle = float(population["angle_deg"])
        branch_slot = branch if branch in {0, 1} else population_index
        rng = np.random.default_rng(_stable_seed(resolved.seed, 0, slot))
        deviation = (
            float(rng.uniform(-float(resolved.spread_deg), float(resolved.spread_deg)))
            if resolved.spread_deg
            else 0.0
        )
        if resolved.mode == "single":
            population_offset = (
                (float(branch_slot) - 0.5) * spacing
                if len(selected) > 1 or branch_slot in {0, 1}
                else 0.0
            )
            base_center = np.asarray((population_offset, 0.0, 0.0), dtype=float)
        else:
            column = slot % grid_columns
            row = slot // grid_columns
            base_center = np.asarray(
                (
                    (column - (grid_columns - 1) / 2.0) * spacing,
                    (row - (grid_rows - 1) / 2.0) * spacing,
                    0.0,
                ),
                dtype=float,
            )
            jitter = 0.12 * layout_period
            base_center[:2] += rng.uniform(-jitter, jitter, size=2)
        display_angle = np.radians(angle + deviation + (90.0 - float(draw_axis)))
        tilt = np.radians(float(resolved.out_of_plane_deg))
        normal = np.asarray(
            (
                np.cos(tilt) * np.cos(display_angle),
                np.cos(tilt) * np.sin(display_angle),
                np.sin(tilt),
            ),
            dtype=float,
        )
        width_basis = np.asarray(
            (-np.sin(display_angle), np.cos(display_angle), 0.0), dtype=float
        )
        depth_basis = np.cross(normal, width_basis)
        depth_norm = float(np.linalg.norm(depth_basis))
        if depth_norm <= 0.0 or not np.isfinite(depth_norm):
            depth_basis = np.asarray((0.0, 0.0, 1.0), dtype=float)
        else:
            depth_basis /= depth_norm
        orientation = np.column_stack((width_basis, depth_basis, normal))
        for layer_index in range(n_layers):
            centred_index = float(layer_index) - (n_layers - 1) / 2.0
            centre = (
                base_center
                + normal * (centred_index * period)
                + width_basis
                * (centred_index * float(resolved.lateral_shift_ratio) * period)
            )
            size = np.asarray(
                (
                    float(resolved.width_ratio) * period,
                    float(resolved.depth_ratio) * period,
                    float(resolved.thickness_ratio) * period,
                ),
                dtype=float,
            )
            local_corners = _CORNER_SIGNS * (0.5 * size)
            corners = local_corners @ orientation.T + centre
            centres.append(centre)
            sizes.append(size)
            orientations.append(orientation)
            vertices.append(corners)
            stack_ids.append(stack_id)
            branch_ids.append(branch)
            colors.append(_BLUE.copy() if branch_slot % 2 == 0 else _ORANGE.copy())
    centre_array = (
        np.asarray(centres, dtype=float).reshape((-1, 3))
        if centres
        else np.empty((0, 3), dtype=float)
    )
    size_array = (
        np.asarray(sizes, dtype=float).reshape((-1, 3))
        if sizes
        else np.empty((0, 3), dtype=float)
    )
    orientation_array = (
        np.asarray(orientations, dtype=float).reshape((-1, 3, 3))
        if orientations
        else np.empty((0, 3, 3), dtype=float)
    )
    vertex_array = (
        np.asarray(vertices, dtype=float).reshape((-1, 8, 3))
        if vertices
        else np.empty((0, 8, 3), dtype=float)
    )
    color_array = (
        np.asarray(colors, dtype=float).reshape((-1, 4))
        if colors
        else np.empty((0, 4), dtype=float)
    )
    if len(vertex_array):
        bounds = np.vstack(
            (
                np.min(vertex_array.reshape(-1, 3), axis=0),
                np.max(vertex_array.reshape(-1, 3), axis=0),
            )
        )
    else:
        bounds = np.asarray(((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)), dtype=float)
    scene_status = (
        "candidate"
        if any(item.get("status") == "candidate" for item in selected)
        else ("manual" if resolved.period_source == "manual" else "schematic")
    )
    message = (
        "Manual-assumption lamellar schematic."
        if scene_status == "manual"
        else (
            "Candidate parameters drive this schematic; formal values remain provisional."
            if scene_status == "candidate"
            else "Schematic generated from measured radial lobe peaks."
        )
    )
    metadata = _metadata_base(
        resolved,
        identity,
        status=scene_status,
        available=True,
        message=message,
        draw_axis_deg=draw_axis,
        reference_period=layout_period,
        assumptions=assumptions,
        parameter_sources=parameter_sources,
        populations=populations,
        q_unit=q_unit,
        flags=flags,
    )
    metadata["length_unit"] = length_unit
    metadata["period_source"] = resolved.period_source
    metadata["selected_branch"] = int(resolved.selected_branch)
    return LamellarScene(
        centers=centre_array,
        sizes=size_array,
        orientations=orientation_array,
        vertices=vertex_array,
        stack_ids=np.asarray(stack_ids, dtype=int),
        branch_ids=np.asarray(branch_ids, dtype=int),
        colors=color_array,
        bounds=bounds,
        length_unit=length_unit,
        metadata=metadata,
    )


def load_lamellar_sources(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    from .lamellar_sources import load_lamellar_sources as _load

    return _load(path)


__all__ = (
    "LamellarScene",
    "LamellarSettings",
    "build_lamellar_scene",
    "load_lamellar_sources",
)
