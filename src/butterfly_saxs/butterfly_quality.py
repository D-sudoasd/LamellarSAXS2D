"""Parameter-level evidence checks for observed arcs, not optimizer acceptance.

Limits here are explicitly provisional engineering screens. Statistical
confidence and material interpretation require independent calibration evidence.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
import numpy as np

from .butterfly_ridge import FIRST_ORDER_FAMILY_SPAN

# A first-order Wang ellipse can extend past q*, but a major axis more than
# twice q* is fitting a larger-q continuum, not the Bragg ellipse.
MAJOR_AXIS_TO_QSTAR_MAX = 2.0


PARAMETERS = ("a", "b", "axis_ratio", "theta_deg")


def classify_ellipse_publication(
    *,
    quality_status=None,
    axis_ratio=None,
    flags=None,
) -> str:
    """Say whether the published row is a ring period, an interior ellipse, or a fail.

    A cap/floor ``b/a`` is not a measured ellipticity.  Those frames still have
    a first-order ring period; they are not a quantitative Wang ellipse.
    """

    quality = str(quality_status or "").strip().upper()
    if isinstance(flags, Mapping):
        flag_tokens = [str(item) for item in (flags.get("flags") or ()) if item]
        if flags.get("axis_ratio"):
            flag_tokens.append("axis_ratio_at_bound")
    elif isinstance(flags, str):
        flag_tokens = [part.strip() for part in flags.split(",") if part.strip()]
    else:
        flag_tokens = [str(item) for item in (flags or ()) if item]
    flag_text = ",".join(flag_tokens)
    if quality in {"FAIL", "FAILED", "INVALID"}:
        return "fail"
    if any(
        token in flag_text
        for token in (
            "axis_ratio_at_bound",
            "axis_ratio_collapsed_to_line",
            "major_axis_exceeds_observed_extent",
        )
    ):
        return "ring"
    ratio = _finite(axis_ratio)
    if ratio is not None:
        return "ellipse"
    return "undetermined"


def unpublished_ellipse_shape(*, quality_status=None, axis_ratio=None, flags=None) -> bool:
    """True when the canvas/table should show a first-order ring, not an ellipse."""

    return classify_ellipse_publication(
        quality_status=quality_status,
        axis_ratio=axis_ratio,
        flags=flags,
    ) == "ring"


def _finite(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _arc_rows(trace, candidate):
    """Read lossless per-arc evidence from either trace or fit payload."""

    candidates = []
    if isinstance(trace, Mapping):
        candidates.extend((trace.get("arc_diagnostics"), trace.get("arc_support")))
        diagnostics = trace.get("diagnostics")
        if isinstance(diagnostics, Mapping):
            candidates.extend((diagnostics.get("arc_diagnostics"), diagnostics.get("arc_support")))
    if isinstance(candidate, Mapping):
        candidates.extend((candidate.get("arc_diagnostics"), candidate.get("arc_support")))
    elif isinstance(trace, Sequence) and not isinstance(trace, (str, bytes)):
        candidates.append(trace)
    for value in candidates:
        if isinstance(value, Mapping):
            value = value.get("rows", value.get("arcs", value.get("records")))
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return [dict(item) for item in value if isinstance(item, Mapping)]
    return []


def _dedupe_reasons(reasons):
    return list(dict.fromkeys(str(reason) for reason in reasons if reason))


def _arc_support_quality(arc_rows):
    """Aggregate support/evidence without promoting a fit to acceptance."""

    support_statuses = [str(row.get("support_status", "unavailable")) for row in arc_rows]
    support_frozen = sum(status in {"frozen", "manual_only"} for status in support_statuses)
    support_missing = sum(status not in {"frozen", "manual_only"} for status in support_statuses)
    manual_only = sum(status == "manual_only" for status in support_statuses)
    invalid = [_finite(row.get("infeasible_projection_count", row.get("support_infeasible_count"))) or 0.0 for row in arc_rows]
    endpoint = [value for row in arc_rows if (value := _finite(row.get("support_endpoint_fraction", row.get("endpoint_fraction")))) is not None]
    manual = [value for row in arc_rows if (value := _finite(row.get("manual_endpoint_fraction"))) is not None]
    normal = [value for row in arc_rows if (value := _finite(row.get("normal_residual_q_rms"))) is not None]
    normal_ratio = [value for row in arc_rows if (value := _finite(row.get("normal_residual_localization_ratio_rms"))) is not None]
    coverage = [value for row in arc_rows if (value := _finite(row.get("projection_rmse_q"))) is not None]
    gaps = [_finite(row.get("support_gap_count")) or 0.0 for row in arc_rows]
    short = sum(bool(row.get("short_arc")) or "short_arc" in row.get("topology_flags", ()) for row in arc_rows)
    return {
        "arc_count": len(arc_rows),
        "support_frozen_count": int(support_frozen),
        "support_unavailable_count": int(support_missing),
        "manual_only_arc_count": int(manual_only),
        "support_component_count": int(sum(int(row.get("support_component_count", len(row.get("support_components", ())))) for row in arc_rows)),
        "support_gap_count": int(sum(int(value) for value in gaps)),
        "short_arc_count": int(short),
        "infeasible_projection_count": int(sum(int(value) for value in invalid)),
        "projection_rmse_q_median": float(np.median(coverage)) if coverage else None,
        "normal_residual_q_rms_median": float(np.median(normal)) if normal else None,
        "normal_residual_localization_ratio_rms_median": float(np.median(normal_ratio)) if normal_ratio else None,
        "support_endpoint_fraction_median": float(np.median(endpoint)) if endpoint else None,
        "manual_endpoint_fraction_median": float(np.median(manual)) if manual else None,
    }


def evaluate_arc_evidence(trace, candidate, *, uncertainty=None, sensitivity=None):
    """Keep a candidate, its empirical support, and publishability separate."""
    trace = trace if isinstance(trace, Mapping) else {}
    candidate = candidate if isinstance(candidate, Mapping) else {}
    points = [p for p in trace.get("points", []) if p.get("accepted", p.get("valid", False))
              and p.get("side") in ("upper", "lower") and p.get("branch_id") in (0, 1)]
    groups = {f"{branch}:{side}": sum(p.get("branch_id") == branch and p.get("side") == side
                                    for p in points)
              for branch in (0, 1) for side in ("upper", "lower")}
    common = []
    occupied_sides = sum(
        1 for count in groups.values() if isinstance(count, (int, float)) and count > 0
    )
    if not candidate or not candidate.get("success"):
        common.append("solver_or_arc_support_unavailable")
    if occupied_sides < 3:
        common.append("insufficient_occupied_sides")
    if any(count < 3 for count in groups.values()):
        common.append("insufficient_independent_side_support")
    ratio = _finite(candidate.get("axis_ratio"))
    bound_flags = candidate.get("bound_flags", {}) or {}
    flag_names = {str(item) for item in (candidate.get("flags") or ())}
    line_collapsed = (
        ratio is not None
        and ratio <= 0.02
        and (
            bool(bound_flags.get("axis_ratio"))
            or "axis_ratio_at_bound" in flag_names
        )
    )
    if line_collapsed:
        common.append("axis_ratio_collapsed_to_line")
    qs = []
    for point in points:
        qx = _finite(point.get("qx"))
        qy = _finite(point.get("qy"))
        if qx is not None and qy is not None:
            qs.append(math.hypot(qx, qy))
    hint = None
    diagnostics = trace.get("diagnostics")
    if isinstance(diagnostics, Mapping):
        first_order = diagnostics.get("first_order_q_hint")
        if isinstance(first_order, Mapping):
            hint = _finite(first_order.get("q_star"))
    if hint is not None and hint > 0.0:
        family = [radius for radius in qs if radius <= FIRST_ORDER_FAMILY_SPAN * hint]
        if family:
            qs = family
    q_extent = max(qs) if qs else None
    a_value = _finite(candidate.get("a"))
    major_exceeds_extent = (
        a_value is not None
        and q_extent is not None
        and q_extent > 0
        and a_value > 1.2 * q_extent
    )
    if (
        a_value is not None
        and hint is not None
        and hint > 0.0
        and a_value > MAJOR_AXIS_TO_QSTAR_MAX * hint
    ):
        major_exceeds_extent = True
    if major_exceeds_extent:
        common.append("major_axis_exceeds_observed_extent")
    arc_rows = _arc_rows(trace, candidate)
    arc_metrics = _arc_support_quality(arc_rows)
    if not arc_rows:
        common.append("per_arc_support_evidence_unavailable")
    else:
        if arc_metrics["support_unavailable_count"]:
            common.append("observed_support_unavailable")
        if arc_metrics["manual_only_arc_count"]:
            common.append("manual_arc_bounds_not_experimental_evidence")
        if arc_metrics["short_arc_count"]:
            common.append("short_observed_arc")
        if arc_metrics["infeasible_projection_count"]:
            common.append("observed_support_infeasible")
        if arc_metrics["support_gap_count"]:
            common.append("disconnected_observed_support")
        endpoint_dependence = any(
            (_finite(row.get("support_endpoint_fraction", row.get("endpoint_fraction"))) or 0.0) > 0.0
            or (_finite(row.get("manual_endpoint_fraction")) or 0.0) > 0.0
            or bool(row.get("support_endpoint_clipped"))
            or bool(row.get("manual_bound_clipped"))
            for row in arc_rows
        )
        if endpoint_dependence:
            common.append("arc_endpoint_or_manual_bound_dependent")
        mirror_available = [row for row in arc_rows if row.get("mirror_symmetry_status", row.get("mirror_status")) in {"available", "observed", "supported"}]
        mirror_missing = [row for row in arc_rows if row.get("mirror_symmetry_status", row.get("mirror_status")) in {None, "unavailable", "ambiguous", "unpaired"}]
        if mirror_missing or (arc_rows and not mirror_available):
            common.append("mirror_symmetry_evidence_unavailable_or_ambiguous")
        normal_missing = [row for row in arc_rows if _finite(row.get("normal_residual_q_rms")) is None and _finite(row.get("normal_residual_q_p95")) is None]
        if normal_missing:
            common.append("per_arc_normal_residual_evidence_unavailable")
        normal_screen = []
        for row in arc_rows:
            value = _finite(row.get("normal_residual_q_rms", row.get("normal_residual_q_p95")))
            scale = _finite(row.get("median_localization_sigma_q", row.get("localization_sigma_q")))
            if value is not None and scale is not None and scale > 0:
                normal_screen.append(value > 3.0 * scale)
        if any(normal_screen):
            common.append("per_arc_normal_residual_exceeds_localization_scale")
    trace_diagnostics = trace.get("diagnostics")
    seed_matches = trace_diagnostics.get("seed_matches", []) if isinstance(trace_diagnostics, Mapping) else []
    if any(not match.get("matched", False) for match in seed_matches if isinstance(match, Mapping)):
        common.append("seed_not_supported_by_observed_arc")
    sigmas = np.asarray([v for p in points if (v := _finite(p.get("localization_sigma_q")))
                         is not None and v > 0], dtype=float)
    median_sigma = float(np.median(sigmas)) if sigmas.size else None
    if median_sigma is not None:
        if any(
            (value := _finite(row.get("normal_residual_q_rms", row.get("normal_residual_q_p95")))) is not None
            and value > 3.0 * median_sigma
            for row in arc_rows
        ):
            common.append("per_arc_normal_residual_exceeds_localization_scale")
    rmse = _finite(candidate.get("rmse"))
    if median_sigma is None:
        common.append("localization_uncertainty_unavailable")
    elif rmse is not None and rmse > 3 * median_sigma:
        common.append("residual_exceeds_localization_scale")
    condition = _finite(candidate.get("condition"))
    if condition is None or condition > 1e8:
        common.append("ill_conditioned_geometry")
    uncertainty = dict(uncertainty or {})
    intervals = uncertainty.get("intervals", {})
    sensitivity = dict(sensitivity or {})
    for holdout in sensitivity.get("held_out_arcs", []):
        if not holdout.get("success"):
            common.append("held_out_arc_not_identifiable")
        elif median_sigma is not None:
            error = _finite(holdout.get("predictive_rmse_q"))
            if error is not None and error > 3 * median_sigma:
                common.append("held_out_arc_prediction_exceeds_localization")
    rows = {}
    b = _finite(candidate.get("b"))
    width_scales = [v for p in points if (v := _finite(p.get("normal_fwhm_q")))
                    is not None and v > 0]
    bound_flags = candidate.get("bound_flags", {})
    for name in PARAMETERS:
        reasons = list(common)
        value = _finite(candidate.get(name))
        if value is None:
            reasons.append("parameter_unavailable")
        bound_name = "theta" if name == "theta_deg" else name
        if bound_flags.get(bound_name) or bound_flags.get(name) or (
            name == "b" and (bound_flags.get("a") or bound_flags.get("axis_ratio"))
        ):
            reasons.append("parameter_at_explicit_bound")
        if name in ("b", "axis_ratio") and b is not None and median_sigma is not None and b <= 2 * median_sigma:
            reasons.append("minor_axis_not_resolved_against_localization")
        interval = intervals.get(name)
        if isinstance(interval, Mapping):
            interval = [interval.get("low", interval.get("lower")), interval.get("high", interval.get("upper"))]
        finite_interval = None
        if isinstance(interval, (list, tuple)) and len(interval) == 2:
            lo, hi = (_finite(x) for x in interval)
            if lo is not None and hi is not None and hi >= lo:
                finite_interval = [lo, hi]
                if name != "theta_deg" and lo <= 0:
                    reasons.append("interval_reaches_degenerate_geometry")
        if finite_interval is None:
            reasons.append("image_resampling_interval_unavailable")
        success_fraction = _finite(uncertainty.get("success_fraction"))
        if success_fraction is not None and success_fraction < 0.9:
            reasons.append("resampling_topology_or_fit_unstable")
        shifts = sensitivity.get("parameter_ranges", {}).get(name)
        if shifts is not None and finite_interval is not None:
            if shifts[0] < finite_interval[0] or shifts[1] > finite_interval[1]:
                reasons.append("analysis_choices_exceed_resampling_interval")
        if sensitivity.get("failed_variants"):
            reasons.append("analysis_choice_topology_unstable")
        if not sensitivity.get("completed", False):
            reasons.append("analysis_choice_sensitivity_unassessed")
        # No machine setting or human Accept click can self-certify coverage.
        # Candidate/empirical information remains available even before the
        # external evidence gate has closed.
        reasons = _dedupe_reasons(reasons)
        support = "supported" if not reasons else "undetermined"
        calibrated = bool(uncertainty.get("coverage_calibrated", False))
        publication_reasons = list(reasons)
        if not calibrated:
            publication_reasons.append("interval_coverage_not_calibrated")
        rows[name] = {
            "value": value if not publication_reasons else None,
            "candidate_value": value,
            "status": "available" if not publication_reasons else "undetermined",
            "empirical_status": support,
            "reason": "; ".join(_dedupe_reasons(publication_reasons)),
            "reasons": _dedupe_reasons(publication_reasons),
            "interval": finite_interval,
            "interval_kind": "calibrated_confidence_interval" if calibrated else "empirical_central_95_percent",
            "unit": "degree" if name == "theta_deg" else "dimensionless" if name == "axis_ratio" else candidate.get("q_unit", "unknown"),
        }
    status = "supported" if all(r["empirical_status"] == "supported" for r in rows.values()) else "undetermined"
    return {
        "quantitative_parameters": rows,
        "measurement_status": status,
        "warm_start_eligible": status == "supported",
        "quality": {
            "status": (
                "FAIL"
                if not candidate.get("success") or occupied_sides < 3
                else "WARN"
            ),
            "engineering_status": "WARN" if status != "supported" else "PASS",
            "scientific_status": "NOT_ACCEPTED",
            "scientific_reason": "Independent human/instrument evidence and a frozen dataset-specific gate are required",
            "thresholds_version": "butterfly-arcs-engineering-provisional-v1",
            "thresholds_frozen": False,
            "metrics": {"side_counts": groups, "occupied_sides": occupied_sides,
                        "observed_q_extent": q_extent,
                        "a_over_observed_extent": (
                            a_value / q_extent
                            if a_value is not None and q_extent not in (None, 0)
                            else None
                        ),
                        "median_localization_sigma_q": median_sigma,
                        "median_normal_fwhm_q": float(np.median(width_scales)) if width_scales else None,
                        "residual_sigma_ratio": rmse / median_sigma if rmse is not None and median_sigma else None,
                        "arc_support": arc_metrics},
            "arc_support": arc_metrics,
            "provisional_limits": {"side_points_min": 3, "residual_sigma_ratio_max": 3,
                                   "minor_axis_sigma_ratio_min": 2, "condition_max": 1e8,
                                   "resampling_success_fraction_min": 0.9},
            "flags": sorted({reason for row in rows.values() for reason in row["reasons"]}),
        },
    }
