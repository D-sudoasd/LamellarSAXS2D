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

METHOD_VERSION = "butterfly-annular-trajectory-v1.1"


def _select_and_connect(points, *, angle_step_deg, min_track_points=3):
    """Select observed annular peaks by joint support and trajectory continuity.

    Candidate detection and its intensity/coverage thresholds happen upstream.
    This selector never lowers those thresholds or creates points. It builds a
    directed graph from observed peaks in adjacent q annuli, then uses a small
    dynamic program to select the best path in each uninterrupted graph
    window. A missing annulus or a boundary with no feasible angular link
    splits the window, so paths cannot jump over an observed gap. Competing
    paths remain as rejected point records with explicit selection reasons.
    """

    max_jump_deg = 2.5 * float(angle_step_deg)
    ambiguity_score_margin = math.log(1.35)
    groups: list[list[int]] = []
    edges: list[tuple[int, int]] = []
    choices: dict[tuple[int, str], list[int]] = defaultdict(list)

    def add_flag(point, flag):
        flags = list(point.get("topology_flags", []))
        if flag not in flags:
            flags.append(flag)
        point["topology_flags"] = flags

    def prominence(index):
        try:
            value = float(points[index].get("prominence", 0.0))
        except (TypeError, ValueError):
            value = 0.0
        return value if np.isfinite(value) and value > 0.0 else float(np.finfo(float).eps)

    def signed_delta(left_index, right_index):
        left = float(points[left_index]["chi_deg"])
        right = float(points[right_index]["chi_deg"])
        return (right - left + 180.0) % 360.0 - 180.0

    def can_link(left_index, right_index):
        return abs(signed_delta(left_index, right_index)) <= max_jump_deg + 1e-12

    def record_metrics(record):
        path, strength_sum, turn_sum = record
        support = len(path)
        mean_strength = strength_sum / support
        mean_turn = turn_sum / max(support - 2, 1) if support > 2 else 0.0
        turn_penalty_per_ring = turn_sum / support
        score = mean_strength - 0.5 * turn_penalty_per_ring
        return {
            "support_rings": support,
            "mean_log_relative_prominence": mean_strength,
            "mean_squared_turn_bins": mean_turn,
            "turn_penalty_per_ring": turn_penalty_per_ring,
            "selection_score": score,
        }

    def record_rank(record):
        metrics = record_metrics(record)
        return (
            -metrics["support_rings"],
            -metrics["selection_score"],
            -metrics["mean_log_relative_prominence"],
            metrics["mean_squared_turn_bins"],
            record[0],
        )

    def retain_best_two(records):
        unique = {record[0]: record for record in records}
        return sorted(unique.values(), key=record_rank)[:2]

    def solve_window(ring_groups, window_rings):
        """Return best and runner-up observed paths for one connected window."""

        relative_log_prominence = {}
        for ring in window_rings:
            indices = ring_groups[ring]
            strongest = max(prominence(index) for index in indices)
            for index in indices:
                relative_log_prominence[index] = math.log(prominence(index) / strongest)

        # Each state is keyed by (previous_point, current_point). Keeping the
        # best two records per state is sufficient because every continuation
        # adds the same local strength and turn terms to that state.
        paths_by_end: dict[int, dict[int | None, list[tuple[tuple[int, ...], float, float]]]] = {}
        for ring_position, ring in enumerate(window_rings):
            prior_ring = window_rings[ring_position - 1] if ring_position else None
            for current in ring_groups[ring]:
                states: dict[int | None, list[tuple[tuple[int, ...], float, float]]] = {
                    None: [((current,), relative_log_prominence[current], 0.0)]
                }
                if prior_ring is not None:
                    for previous in ring_groups[prior_ring]:
                        if not can_link(previous, current):
                            continue
                        previous_states = paths_by_end.get(previous, {})
                        signed_step = signed_delta(previous, current)
                        for records in previous_states.values():
                            for record in records:
                                path, strength_sum, turn_sum = record
                                turn = 0.0
                                if len(path) >= 2:
                                    prior_step = signed_delta(path[-2], path[-1])
                                    turn = ((signed_step - prior_step) / float(angle_step_deg)) ** 2
                                states.setdefault(previous, []).append(
                                    (path + (current,), strength_sum + relative_log_prominence[current], turn_sum + turn)
                                )
                for previous, records in states.items():
                    states[previous] = retain_best_two(records)
                paths_by_end[current] = states

        all_records = []
        for states in paths_by_end.values():
            for records in states.values():
                all_records.extend(records)
        best_records = retain_best_two(all_records)
        return best_records[0], best_records[1] if len(best_records) > 1 else None

    for index, point in enumerate(points):
        point.update(
            trajectory_id=None,
            trajectory_status="not_selected",
            trajectory_confidence="unassessed",
            trajectory_ambiguous=False,
            trajectory_support_rings=0,
            trajectory_selection_reason="not_selected",
            trajectory_alternative_point_ids=[],
        )
        if point["accepted"] and point["branch_id"] in (0, 1):
            choices[(int(point["annulus_index"]), str(point["quadrant"]))].append(index)
        elif point["accepted"]:
            point.update(accepted=False, valid=False, reason="reference_axis_or_streak_region")
            point["trajectory_selection_reason"] = "reference_axis_or_streak_region"

    for (ring, quadrant), indices in choices.items():
        ordered = sorted(indices, key=lambda item: (-prominence(item), item))
        for rank, index in enumerate(ordered, start=1):
            points[index]["candidate_rank_in_quadrant"] = rank

    diagnostics = {
        "selection_method": "adjacent_annulus_dynamic_path",
        "objective": "maximize observed annuli, then mean log-relative prominence minus 0.5 times angular-turn penalty per ring",
        "max_angular_step_deg": max_jump_deg,
        "turn_penalty_weight": 0.5,
        "ambiguity_score_margin": ambiguity_score_margin,
        "windows": [],
        "trajectories": [],
        "n_candidate_points": sum(len(indices) for indices in choices.values()),
        "n_selected_points": 0,
        "n_rejected_candidates": 0,
        "n_candidate_graph_edges": 0,
        "n_selected_edges": 0,
        "n_ambiguous_trajectories": 0,
        "n_low_confidence_trajectories": 0,
    }

    quadrants = sorted({quadrant for _, quadrant in choices})
    for quadrant in quadrants:
        ring_groups = {
            ring: indices
            for (ring, candidate_quadrant), indices in choices.items()
            if candidate_quadrant == quadrant
        }
        rings = sorted(ring_groups)
        if not rings:
            continue
        windows: list[list[int]] = []
        current_window = [rings[0]]
        for ring in rings[1:]:
            previous_ring = current_window[-1]
            if ring == previous_ring + 1:
                has_link = any(
                    can_link(left, right)
                    for left in ring_groups[previous_ring]
                    for right in ring_groups[ring]
                )
                if has_link:
                    diagnostics["n_candidate_graph_edges"] += sum(
                        can_link(left, right)
                        for left in ring_groups[previous_ring]
                        for right in ring_groups[ring]
                    )
                    current_window.append(ring)
                    continue
            windows.append(current_window)
            current_window = [ring]
        windows.append(current_window)

        for window_rings in windows:
            best, runner_up = solve_window(ring_groups, window_rings)
            best_metrics = record_metrics(best)
            runner_metrics = record_metrics(runner_up) if runner_up else None
            ambiguous = False
            score_margin = None
            if runner_up and best_metrics["support_rings"] == runner_metrics["support_rings"]:
                score_margin = best_metrics["selection_score"] - runner_metrics["selection_score"]
                ambiguous = abs(score_margin) <= ambiguity_score_margin

            trajectory_id = len(groups)
            path = list(best[0])
            alternative_path = list(runner_up[0]) if runner_up else []
            alternative_ids = [
                str(points[index].get("point_id", index))
                for index in alternative_path
                if index not in path
            ]
            short = len(path) < int(min_track_points)
            low_confidence_reasons = []
            if ambiguous:
                low_confidence_reasons.append("similar_supported_trajectory")
            if short:
                low_confidence_reasons.append("short_observed_support")
            if best_metrics["mean_log_relative_prominence"] <= -ambiguity_score_margin:
                low_confidence_reasons.append("weaker_local_prominence_path")
            confidence = "low" if low_confidence_reasons else "provisional"
            status = "ambiguous_selected" if ambiguous else ("short_observed_track" if short else "selected")
            for index in path:
                point = points[index]
                point.update(
                    trajectory_id=trajectory_id,
                    trajectory_status=status,
                    trajectory_confidence=confidence,
                    trajectory_ambiguous=ambiguous,
                    trajectory_support_rings=len(path),
                    trajectory_selection_reason=status,
                    trajectory_alternative_point_ids=alternative_ids,
                    trajectory_low_confidence_reasons=low_confidence_reasons,
                    trajectory_mean_log_relative_prominence=best_metrics["mean_log_relative_prominence"],
                    trajectory_mean_squared_turn_bins=best_metrics["mean_squared_turn_bins"],
                    trajectory_selection_score=best_metrics["selection_score"],
                )
                point["accepted"] = True
                point["valid"] = True
                if ambiguous:
                    add_flag(point, "ambiguous_annular_trajectory")
                if short:
                    add_flag(point, "short_annular_trajectory")

            path_set = set(path)
            alternative_set = set(alternative_path)
            for ring in window_rings:
                for index in ring_groups[ring]:
                    if index in path_set:
                        continue
                    point = points[index]
                    is_ambiguous_alternative = ambiguous and index in alternative_set
                    reason = "ambiguous_trajectory_alternative" if is_ambiguous_alternative else "weaker_trajectory_candidate"
                    point.update(
                        accepted=False,
                        valid=False,
                        reason=reason,
                        trajectory_status="ambiguous_alternative" if is_ambiguous_alternative else "not_selected",
                        trajectory_confidence="low" if is_ambiguous_alternative else "unassessed",
                        trajectory_ambiguous=bool(is_ambiguous_alternative),
                        trajectory_support_rings=(runner_metrics["support_rings"] if is_ambiguous_alternative else 0),
                        trajectory_selection_reason=reason,
                        trajectory_alternative_of=trajectory_id,
                    )
                    if is_ambiguous_alternative:
                        add_flag(point, "ambiguous_annular_trajectory")
                    else:
                        add_flag(point, "unselected_annular_trajectory")

            for left, right in zip(path, path[1:]):
                edges.append((left, right))
            for offset, index in enumerate(path):
                left = points[path[max(0, offset - 1)]]
                right = points[path[min(len(path) - 1, offset + 1)]]
                dx, dy = right["qx"] - left["qx"], right["qy"] - left["qy"]
                norm = math.hypot(dx, dy)
                if norm:
                    points[index].update(tangent_qx=dx / norm, tangent_qy=dy / norm)
            groups.append(path)
            trajectory = {
                "trajectory_id": trajectory_id,
                "quadrant": quadrant,
                "annulus_indices": [int(points[index]["annulus_index"]) for index in path],
                "point_ids": [str(points[index].get("point_id", index)) for index in path],
                "confidence": confidence,
                "status": status,
                "ambiguous": ambiguous,
                "low_confidence_reasons": low_confidence_reasons,
                **best_metrics,
                "runner_up": ({
                    "point_ids": [str(points[index].get("point_id", index)) for index in alternative_path],
                    **runner_metrics,
                } if runner_up else None),
                "score_margin_to_runner_up": score_margin,
            }
            diagnostics["trajectories"].append(trajectory)
            diagnostics["windows"].append({
                "quadrant": quadrant,
                "annulus_indices": list(window_rings),
                "selected_trajectory_id": trajectory_id,
                "candidate_point_count": sum(len(ring_groups[ring]) for ring in window_rings),
                "selected_point_count": len(path),
                "ambiguous": ambiguous,
                "reason": "contiguous_candidate_rings_with_feasible_cross_ring_links",
            })
            diagnostics["n_selected_points"] += len(path)
            diagnostics["n_rejected_candidates"] += sum(
                not points[index]["accepted"]
                for ring in window_rings
                for index in ring_groups[ring]
            )
            diagnostics["n_selected_edges"] += max(0, len(path) - 1)
            diagnostics["n_ambiguous_trajectories"] += int(ambiguous)
            diagnostics["n_low_confidence_trajectories"] += int(confidence == "low")

    return groups, edges, diagnostics


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
    groups, graph_edges, trajectory_selection = _select_and_connect(points, angle_step_deg=angle_step)
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
            candidate["trajectory_id"] = point.get("trajectory_id") if point else None
            candidate["trajectory_status"] = point.get("trajectory_status") if point else "not_selected"
            candidate["trajectory_confidence"] = point.get("trajectory_confidence") if point else "unassessed"
            candidate["trajectory_ambiguous"] = bool(point.get("trajectory_ambiguous", False)) if point else False
            candidate["trajectory_selection_reason"] = (
                point.get("trajectory_selection_reason") if point else candidate["trace_reason"]
            )
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
            "trajectory_selection": {
                "selected_count": len(selected),
                "low_confidence_count": sum(p.get("trajectory_confidence") == "low" for p in selected),
                "ambiguous_count": sum(bool(p.get("trajectory_ambiguous")) for p in selected),
            },
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
    outer_sides = sorted({
        f"{point['branch_id']}:{point['side']}"
        for point in points
        if point["accepted"] and point["annulus_index"] == n_annuli - 1
        and point["branch_id"] in (0, 1) and point["side"] in ("upper", "lower")
    })
    settings = {"requested_radial_bins": requested, "effective_radial_bins": n_annuli,
                "requested_angle_bins": requested_angles, "angle_bins": angle_bins,
                "q_bin_width": dq, "angle_bin_width_deg": angle_step,
                "minimum_track_rings": 3, "minimum_track_rings_is_confidence_only": True,
                "competing_peak_ratio": 1.35,
                "trajectory_selection_method": "adjacent_annulus_dynamic_path",
                "maximum_track_angle_jump_deg": 2.5 * angle_step,
                "q_step": step, "q_step_source": step_source, "q_step_details": step_details,
                **accumulator.get("settings", {})}
    explicit_center = any(key in options for key in ("center_q", "center_hint", "center_qx", "center_qy"))
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
                        "center_q": ([float(topology["center_qx"]), float(topology["center_qy"])]
                                     if explicit_center else None),
                        "center_q_source": "explicit_option" if explicit_center else "not_supplied",
                        "n_points": len(points), "n_accepted_points": sum(p["accepted"] for p in points),
                        "n_arcs": len(arcs), "n_raw_candidates": sum(len(r["candidates"]) for r in annuli),
                        "trajectory_selection": trajectory_selection,
                        "outer_window_accepted_sides": outer_sides,
                        "outer_window_truncated": len(outer_sides) >= 2,
                        "observed_support": support, "applied_edits": applied, "seed_actions": seed_records,
                        "first_order_q_hint": {"selection_status": "not_used", "q_star": None,
                                               "reason": "annuli_are_sampling_coordinates_not_radial_peaks"},
                        "elapsed_s": time.perf_counter() - started},
    }
