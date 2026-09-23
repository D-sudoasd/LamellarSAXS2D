"""Adapt measured sector-profile peaks to the shared observed-arc workflow.

One angular sector contributes at most one measured peak. Missing, ambiguous
or excluded sectors are never joined across, and no ellipse is used to choose
or relocate the measured peaks. The finite sector footprint and bin scale are
recorded; neither is a calibrated confidence interval.
"""

from __future__ import annotations

from collections.abc import Mapping
import math
import time
from typing import Any

import numpy as np

from .arc_support import freeze_observed_support
from .butterfly_ridge import (
    _apply_edits,
    _apply_seeds,
    _arc_topology,
    _assign_reference_branches,
    _normalise_options,
    _parse_q_window,
    _point_signature,
    _public_point,
    _refresh_arc_identity,
)
from .cancellation import raise_if_cancelled
from .ridge_inputs import canonical_inputs
from .sector_peaks import measure_sector_peaks
from .settings import strict_int

METHOD_VERSION = "butterfly-radial-sector-v1.0"


def _sector_components(points, sector_count, *, maximum_jump):
    """Return observed angular chains; a missing sector is an explicit gap."""

    lookup = {p["sector_index"]: i for i, p in enumerate(points) if p["accepted"]}
    neighbours = {i: [] for i in lookup.values()}
    edges = []
    for sector, index in lookup.items():
        other = lookup.get((sector + 1) % sector_count)
        if other is None or other == index:
            continue
        a, b = points[index], points[other]
        if a["branch_id"] != b["branch_id"] or a["branch_id"] not in (0, 1):
            continue
        gap = (b["sector_center_deg"] - a["sector_center_deg"]) % 360.0
        if gap > 0.5 * (a["sector_width_deg"] + b["sector_width_deg"]) + 1e-10:
            for point in (a, b):
                point["topology_flags"].append("unmeasured_angular_gap")
            continue
        if abs(a["q_star"] - b["q_star"]) > maximum_jump:
            for point in (a, b):
                point["topology_flags"].append("adjacent_sector_peak_jump")
            continue
        edges.append((index, other))
        neighbours[index].append(other)
        neighbours[other].append(index)
    groups, visited = [], set()
    for start in neighbours:
        if start in visited:
            continue
        pending, component = [start], []
        while pending:
            current = pending.pop()
            if current in visited:
                continue
            visited.add(current)
            component.append(current)
            pending.extend(neighbours[current])
        groups.append(component)
        # Estimate geometric tangents solely from adjacent measured peaks.
        for current in component:
            linked = neighbours[current]
            if not linked:
                continue
            first = points[linked[0]]
            last = points[linked[-1]] if len(linked) > 1 else points[current]
            dx, dy = last["qx"] - first["qx"], last["qy"] - first["qy"]
            norm = math.hypot(dx, dy)
            if norm > np.finfo(float).eps:
                points[current]["tangent_qx"] = dx / norm
                points[current]["tangent_qy"] = dy / norm
    return groups, edges


def trace_butterfly_sector_peaks(
    image: Any,
    qmap: Any,
    q_window: Any,
    *,
    mask: Any = None,
    reference_axis_deg: float = 0.0,
    options: Mapping[str, Any] | None = None,
    edits: Any = None,
    cancel_event: Any = None,
) -> dict[str, Any]:
    """Measure profiles first, then freeze supported peak-trajectory fragments."""

    started = time.perf_counter()
    raise_if_cancelled(cancel_event, "sector-trace:inputs")
    data, qx, qy, q, invalid = canonical_inputs(image, qmap, mask=mask)
    window = _parse_q_window(q_window, q)
    options = dict(options or {})
    edits = list(options.get("edits", []) if edits is None else edits)
    geometry = np.isfinite(qx) & np.isfinite(qy) & np.isfinite(q)
    geometry &= (q >= window[0]) & (q <= window[1])
    valid = geometry & np.isfinite(data) & ~invalid
    valid, applied_edits, seeds = _apply_edits(valid, qx, qy, edits)
    measured = measure_sector_peaks(
        data, qmap, window, mask=~valid, options=options, cancel_event=cancel_event
    )
    bin_width = float(measured["sampling"]["effective_radial_bin_width"])
    sigma = float(measured["sampling"]["sampling_sigma_q"])
    topology_options = _normalise_options(options)
    topology_options["reference_axis_deg"] = float(reference_axis_deg)
    topology_options["min_arc_points"] = strict_int(
        options.get("sector_min_arc_points", 3), "sector_min_arc_points", minimum=2
    )
    jump_bins = float(options.get("sector_max_jump_bins", 3.0))
    if not math.isfinite(jump_bins) or jump_bins <= 0:
        raise ValueError("sector_max_jump_bins must be positive and finite")
    signature = _point_signature(data, {**options, "trace_method": "radial_sector"}, window)
    excluded = {
        str(edit.get("point_id")) for edit in edits
        if isinstance(edit, Mapping) and edit.get("type") == "exclude_point"
    }
    points, profiles = [], {}
    for sector in measured["sectors"]:
        raise_if_cancelled(cancel_event, "sector-trace:profiles")
        index = int(sector["sector_index"])
        point_id = f"sector-{signature}-{index:03d}"
        sector["point_id"] = point_id
        peak = sector["selected_peak"]
        sector.update(
            sector_center_deg=sector["angle_deg"], sector_width_deg=sector["width_deg"],
            selected_peak_q=peak["q_star"] if peak else None,
            q_star=peak["q_star"] if peak else None,
            profile_only=peak is None, accepted=bool(peak), valid=bool(peak),
            source_method="radial_sector",
        )
        profiles[point_id] = {
            "point_id": point_id,
            "profile_axis": "radial",
            "method": "observed_sector_pixel_mean",
            "q_unit": measured["q_unit"],
            "sector_center_deg": sector["angle_deg"],
            "sector_width_deg": sector["width_deg"],
            "q": sector["q_centers"],
            "raw_intensity": sector["raw_mean"],
            "smoothed_intensity": sector["smoothed_intensity"],
            "counts": sector["counts"],
            "coverage": sector["coverage"],
            "fit_intensity": [],
            "selected_peak_q": peak["q_star"] if peak else None,
            "radial_fwhm_q": peak["radial_fwhm"] if peak else None,
            "sampling_sigma_q": sigma,
            "uncertainty_source": measured["sampling"]["sigma_basis"],
            "valid": bool(peak),
            "reason": sector["reason"],
            "candidates": sector["candidates"],
        }
        if peak is None:
            continue
        angle = math.radians(float(sector["angle_deg"]))
        radius = float(peak["q_star"])
        accepted = point_id not in excluded
        point = {
            "point_id": point_id, "sector_index": index,
            "sector_center_deg": sector["angle_deg"], "sector_width_deg": sector["width_deg"],
            "qx": radius * math.cos(angle), "qy": radius * math.sin(angle), "q_star": radius,
            "pixel_x": peak["pixel_x"], "pixel_y": peak["pixel_y"],
            "source_pixel_role": "representative supporting pixel; q_star is a sector-profile statistic",
            "source_method": "radial_sector", "q_unit": measured["q_unit"],
            "intensity": peak["raw_intensity"], "snr": peak["snr"],
            "prominence": peak["prominence"], "radial_fwhm": peak["radial_fwhm"],
            "coverage": float(sector["coverage"][int(peak["peak_bin_index"])]),
            "n_pixels": peak["source_pixel_count"],
            "normal_qx": math.cos(angle), "normal_qy": math.sin(angle),
            "normal_basis": "radial measurement direction, not a fitted ellipse normal",
            "tangent_qx": -math.sin(angle), "tangent_qy": math.cos(angle),
            "q_normal_step": bin_width, "sampling_sigma_q": sigma,
            "uncertainty_source": measured["sampling"]["sigma_basis"],
            "normal_fwhm_q": float("nan"), "localization_sigma_q": float("nan"),
            "scale": 1.0,  # Temporary compatibility input for the topology helper.
            "branch_id": -1, "side": "unknown", "arc_id": -1,
            "accepted": accepted, "valid": accepted,
            "reason": "accepted_sector_peak" if accepted else "excluded_point_edit",
            "score": peak["snr"],
            "topology_flags": ["sector_profile_peak", "peak_order_unassigned", "finite_sector_footprint"],
        }
        points.append(point)
    _assign_reference_branches(points, topology_options)
    groups, edges = _sector_components(
        points, len(measured["sectors"]), maximum_jump=jump_bins * bin_width
    )
    arcs = _arc_topology(groups, edges, points, topology_options, bin_width)
    seed_records = _apply_seeds(points, arcs, seeds, topology_options, bin_width)
    _refresh_arc_identity(arcs, points)
    # The graph helper's scale consistency is not multiscale evidence for a
    # single-resolution sector measurement. Keep it explicitly unavailable.
    for item in [*points, *arcs]:
        item.pop("scale", None)
        item["scale_stability"] = float("nan")
        item["scale_stable"] = False
        item["source_method"] = "radial_sector"
    by_id = {point["point_id"]: point for point in points}
    for sector in measured["sectors"]:
        point = by_id.get(sector["point_id"])
        if point is not None:
            for key in ("qx", "qy", "pixel_x", "pixel_y", "branch_id", "side", "arc_id", "accepted", "valid"):
                sector[key] = point[key]
            sector["geometry_reason"] = point["reason"]
    support = freeze_observed_support(points, arcs, summary_only=True, cancel_event=cancel_event)
    measured["interpretation"] += "; finite sector footprint can shift or broaden a peak"
    return {
        "points": [_public_point(point) for point in points],
        "arcs": arcs, "profiles": profiles, "sector_peaks": measured,
        "method_version": METHOD_VERSION,
        "diagnostics": {
            "method": "fixed_azimuth_sector_radial_peak",
            "topology_before_ellipse_fit": True, "ellipse_fit": None,
            "q_window": list(window), "reference_axis_deg": float(reference_axis_deg),
            "center_q": [float(topology_options["center_qx"]), float(topology_options["center_qy"])],
            "sector_origin_q": [0.0, 0.0],
            "n_sectors": len(measured["sectors"]), "n_points": len(points),
            "n_accepted_points": sum(bool(p["accepted"]) for p in points),
            "n_arcs": len(arcs), "n_raw_candidates": sum(len(s["candidates"]) for s in measured["sectors"]),
            "mask_fraction_in_q_window": 1.0 - float(valid.sum()) / max(1, int(geometry.sum())),
            "observed_support": support, "applied_edits": applied_edits,
            "seed_actions": seed_records, "maximum_adjacent_sector_jump_q": jump_bins * bin_width,
            "first_order_q_hint": {"selection_status": "not_used", "q_star": None,
                                   "reason": "radial_peak_order_is_not_assigned"},
            "sampling_sigma_basis": measured["sampling"]["sigma_basis"],
            "sector_overlap": measured["sector_overlap"],
            "elapsed_s": time.perf_counter() - started,
        },
    }
