"""Track observed azimuthal maxima outwards through calibrated q annuli.

The rings are prescribed sampling coordinates, not radial reflection peaks.
No fitted ellipse selects the points and missing lobes remain missing.
"""
from __future__ import annotations

from collections import defaultdict
import math
import time
from typing import Any

import numpy as np

from .arc_support import freeze_observed_support
from .butterfly_ridge import (
    _apply_edits, _apply_seeds, _arc_topology, _assign_reference_branches,
    _normalise_options, _parse_q_window, _point_signature, _public_point,
    _refresh_arc_identity,
)
from .cancellation import raise_if_cancelled
from .ridge_inputs import canonical_inputs
from .sector_peaks import _local_representative_q_step
from .settings import strict_int

METHOD_VERSION = "butterfly-annular-trajectory-v1.0"


def _select_and_connect(points, *, angle_step_deg, min_track_points=3):
    """One supported, unambiguous peak per observed quadrant and annulus.

    Ambiguous competitors remain visible as rejected candidates. Chains join
    only consecutive annuli with a bounded angular displacement, so neither
    masks nor a missing ring are silently bridged.
    """
    choices = defaultdict(list)
    for i, point in enumerate(points):
        if point["accepted"] and point["branch_id"] in (0, 1):
            choices[(point["annulus_index"], point["quadrant"])].append(i)
        elif point["accepted"]:
            point.update(accepted=False, valid=False, reason="reference_axis_or_streak_region")
    chosen = {}
    for key, indices in choices.items():
        ordered = sorted(indices, key=lambda i: points[i]["prominence"], reverse=True)
        ambiguous = len(ordered) > 1 and points[ordered[0]]["prominence"] < 1.35 * points[ordered[1]]["prominence"]
        for rank, index in enumerate(ordered):
            point = points[index]
            point["candidate_rank_in_quadrant"] = rank + 1
            if ambiguous or rank:
                point.update(accepted=False, valid=False,
                             reason="ambiguous_angular_peaks" if ambiguous else "weaker_angular_competitor")
            else:
                chosen[key] = index
    groups, edges, used = [], [], set()
    for key, start in sorted(chosen.items()):
        if start in used:
            continue
        component = [start]
        used.add(start)
        ring, quadrant = key
        while (ring + 1, quadrant) in chosen:
            next_index = chosen[(ring + 1, quadrant)]
            a, b = points[component[-1]], points[next_index]
            angular_step = abs((b["chi_deg"] - a["chi_deg"] + 180) % 360 - 180)
            if angular_step > 2.5 * angle_step_deg:
                b["topology_flags"].append("angular_track_jump")
                break
            edges.append((component[-1], next_index))
            component.append(next_index)
            used.add(next_index)
            ring += 1
        track_id = len(groups)
        for index in component:
            point = points[index]
            point["trajectory_id"] = track_id
            if len(component) < min_track_points:
                point.update(accepted=False, valid=False, reason="short_annular_track")
        for offset, index in enumerate(component):
            left = points[component[max(0, offset - 1)]]
            right = points[component[min(len(component) - 1, offset + 1)]]
            dx, dy = right["qx"] - left["qx"], right["qy"] - left["qy"]
            norm = math.hypot(dx, dy)
            if norm:
                points[index].update(tangent_qx=dx / norm, tangent_qy=dy / norm)
        groups.append(component)
    return groups, edges


def trace_butterfly_annuli(image, qmap, q_window, *, mask=None, reference_axis_deg=0.,
                          options=None, edits=None, cancel_event=None) -> dict[str, Any]:
    from .observables import _azimuthal_peak_ridges, _q_unit

    started = time.perf_counter()
    data, qx, qy, q, invalid = canonical_inputs(image, qmap, mask=mask)
    window = _parse_q_window(q_window, q)
    options = dict(options or {})
    edits = list(options.get("edits", []) if edits is None else edits)
    finite = np.isfinite(qx) & np.isfinite(qy) & np.isfinite(q)
    geometry = finite & (q >= window[0]) & (q <= window[1])
    valid, applied, seeds = _apply_edits(geometry & ~invalid & np.isfinite(data), qx, qy, edits)
    step, step_source, step_details = _local_representative_q_step(
        qx, qy, q, finite, *window, q_is_supplied=True)
    requested = strict_int(options.get("annular_radial_bins", 40), "annular_radial_bins", minimum=4)
    requested_angles = strict_int(options.get("annular_angle_bins", 72), "annular_angle_bins", minimum=16)
    angle_bins = requested_angles
    if step and np.any(valid):
        smallest_radius = float(np.min(q[valid]))
        angular_pixel_step = math.degrees(math.atan2(step, smallest_radius))
        angle_bins = min(requested_angles, int(math.floor(360 / angular_pixel_step)))
        if angle_bins < 16:
            raise ValueError("q domain is too close to the origin for resolved angular profiles")
    n_annuli = min(requested, max(1, int((window[1] - window[0]) / step))) if step else requested
    if n_annuli < 4:
        raise ValueError("q window is too narrow to resolve at least four annuli; widen the analysis window")
    accumulator = {}
    measured, *_ = _azimuthal_peak_ridges(
        data, qmap, window, mask=~valid, n_annuli=n_annuli, n_angle_bins=angle_bins,
        snr_threshold=float(options.get("annular_snr_threshold", 3.)),
        min_peak_fraction=float(options.get("annular_min_peak_fraction", .2)),
        min_coverage=float(options.get("annular_min_coverage", .5)),
        min_bin_count=2, diagnostics=accumulator, cancel_event=cancel_event,
    )
    signature = _point_signature(data, {**options, "trace_method": "annular_peak"}, window)
    excluded = {edit.get("point_id") for edit in edits if edit.get("type") == "exclude_point"}
    unit = _q_unit(qmap)
    edges_q = np.asarray(accumulator["q_edges"])
    centers_q = np.asarray(accumulator["q_centers"])
    angle_order = np.argsort(np.asarray(accumulator["angle_centers_deg"]) % 360)
    display_bin = np.argsort(angle_order)
    angles_deg = (np.asarray(accumulator["angle_centers_deg"]) % 360)[angle_order]
    dq = float(edges_q[1] - edges_q[0])
    angle_step = 360. / angle_bins
    points = []
    for source in measured:
        raise_if_cancelled(cancel_event, "annular-trace:points")
        ring = int(source.metadata["annulus_index"])
        point_id = f"annular-{signature}-{ring:03d}-{source.metadata['angular_bin_index']:03d}"
        radius, chi = float(source.q), float(source.angle)
        accepted = point_id not in excluded
        points.append({
            "point_id": point_id, "profile_id": f"annulus-{signature}-{ring:03d}",
            "annulus_index": ring, "angular_bin_index": int(source.metadata["angular_bin_index"]),
            "q_annulus": radius, "chi_deg": math.degrees(chi) % 360,
            "qx": radius * math.cos(chi), "qy": radius * math.sin(chi),
            "pixel_x": source.metadata["pixel_x"], "pixel_y": source.metadata["pixel_y"],
            "source_pixel_role": "representative supporting pixel; peak angle comes from annular I(chi)",
            "source_method": "annular_peak", "q_unit": unit,
            "intensity": source.metadata.get("raw_intensity", source.intensity),
            "snr": float(source.snr), "prominence": float(source.metadata["prominence"]),
            "azimuthal_fwhm_deg": math.degrees(source.azimuthal_fwhm),
            "n_pixels": source.n_pixels, "coverage": source.metadata["angular_bin_coverage"],
            "normal_qx": -math.sin(chi), "normal_qy": math.cos(chi),
            "normal_basis": "azimuthal scan direction, not necessarily the intensity ridge normal",
            "tangent_qx": math.cos(chi), "tangent_qy": math.sin(chi),
            "q_normal_step": radius * math.radians(angle_step),
            "sampling_sigma_q": radius * math.radians(angle_step) / math.sqrt(12),
            "uncertainty_source": "angular bin resolution, not a confidence interval",
            "normal_fwhm_q": float("nan"), "localization_sigma_q": float("nan"),
            "radial_bin_width": dq, "scale": 1., "branch_id": -1, "side": "unknown", "arc_id": -1,
            "accepted": accepted, "valid": accepted,
            "reason": "accepted_annular_peak" if accepted else "excluded_point_edit",
            "score": float(source.score), "trajectory_id": None,
            "topology_flags": ["annular_angular_maximum", "prescribed_q_not_radial_peak"],
        })
    topology = _normalise_options(options)
    topology.update(reference_axis_deg=float(reference_axis_deg), min_arc_points=3)
    _assign_reference_branches(points, topology)
    groups, graph_edges = _select_and_connect(points, angle_step_deg=angle_step)
    arcs = _arc_topology(groups, graph_edges, points, topology, dq)
    seed_records = _apply_seeds(points, arcs, seeds, topology, dq)
    _refresh_arc_identity(arcs, points)
    for item in [*points, *arcs]:
        item.pop("scale", None)
        item.update(scale_stability=float("nan"), scale_stable=False, source_method="annular_peak")
    profiles, annuli = {}, []
    for index, radius in enumerate(centers_q):
        profile_id = f"annulus-{signature}-{index:03d}"
        selected = [p for p in points if p["annulus_index"] == index and p["accepted"]]
        candidates = accumulator["annuli"][index]["candidates"]
        by_bin = {p["angular_bin_index"]: p for p in points if p["annulus_index"] == index}
        for candidate in candidates:
            point = by_bin.get(candidate.get("angular_bin_index"))
            candidate["trace_accepted"] = bool(point and point["accepted"])
            candidate["trace_reason"] = point["reason"] if point else candidate["reason"]
            candidate["point_id"] = point["point_id"] if point else None
            candidate["source_angular_bin_index"] = candidate["angular_bin_index"]
            candidate["angular_bin_index"] = int(display_bin[candidate["angular_bin_index"]])
            candidate["chi_deg"] = float(candidate["chi_deg"]) % 360
        for point in by_bin.values():
            point["source_angular_bin_index"] = point["angular_bin_index"]
            point["angular_bin_index"] = int(display_bin[point["angular_bin_index"]])
        row = {
            "annulus_index": index, "profile_id": profile_id, "point_id": profile_id,
            "q_center": float(radius), "q_min": float(edges_q[index]), "q_max": float(edges_q[index + 1]),
            "raw_mean": accumulator["raw_mean"][index][angle_order], "raw_sum": accumulator["raw_sum"][index][angle_order],
            "counts": accumulator["counts"][index][angle_order], "geometry_counts": accumulator["geometry_counts"][index][angle_order],
            "coverage": accumulator["coverage"][index][angle_order], "smoothed_intensity": accumulator["smoothed"][index][angle_order],
            "candidates": candidates, "selected_peaks": [_public_point(p) for p in selected],
            "status": "tracked" if selected else "no_tracked_peak",
            "reason": "four_observed_lobes" if len(selected) == 4 else "partial_or_missing_lobe_support",
            "profile_only": True, "source_method": "annular_peak",
        }
        annuli.append(row)
        profiles[profile_id] = {
            "profile_axis": "azimuthal", "angle_deg": angles_deg,
            "raw_intensity": row["raw_mean"], "smoothed_intensity": row["smoothed_intensity"],
            "counts": row["counts"], "coverage": row["coverage"],
            "peak_angles_deg": [p["chi_deg"] for p in selected], "q_center": row["q_center"],
            "q_min": row["q_min"], "q_max": row["q_max"], "q_unit": unit, "reason": row["reason"],
        }
    support = freeze_observed_support(points, arcs, summary_only=True, cancel_event=cancel_event)
    settings = {"requested_radial_bins": requested, "effective_radial_bins": n_annuli,
                "requested_angle_bins": requested_angles, "angle_bins": angle_bins,
                "q_bin_width": dq, "angle_bin_width_deg": angle_step,
                "minimum_track_rings": 3, "competing_peak_ratio": 1.35,
                "maximum_track_angle_jump_deg": 2.5 * angle_step,
                "q_step": step, "q_step_source": step_source, "q_step_details": step_details,
                **accumulator.get("settings", {})}
    return {
        "method_version": METHOD_VERSION, "q_unit": unit,
        "branch_pairing": {"reference_axis_deg": reference_axis_deg,
                           "families": {"0": ["QI", "QIII"], "1": ["QII", "QIV"]},
                           "memberships_frozen_before_fit": True},
        "points": [_public_point(p) for p in points], "arcs": arcs, "profiles": profiles,
        "annular_peaks": {"method_version": METHOD_VERSION, "q_unit": unit, "q_window": list(window),
                          "q_edges": edges_q, "angle_centers_deg": angles_deg,
                          "settings": settings, "annuli": annuli},
        "diagnostics": {"method": "fixed_q_annulus_angular_peak_trajectory", "q_window": list(window),
                        "reference_axis_deg": reference_axis_deg, "n_annuli": n_annuli,
                        "n_points": len(points), "n_accepted_points": sum(p["accepted"] for p in points),
                        "n_arcs": len(arcs), "n_raw_candidates": sum(len(r["candidates"]) for r in annuli),
                        "observed_support": support, "applied_edits": applied, "seed_actions": seed_records,
                        "first_order_q_hint": {"selection_status": "not_used", "q_star": None,
                                               "reason": "annuli_are_sampling_coordinates_not_radial_peaks"},
                        "elapsed_s": time.perf_counter() - started},
    }
