"""Shared observed-arc workflow for Qt, Python, CLI and streaming batches.

Tracing owns branch/side identities. The ellipse solver may parameterize those
observations, but may never repair their topology by relabelling them.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
from typing import Any

import numpy as np

from .butterfly_settings import METHOD_VERSION, normalize_butterfly_settings
from .butterfly_quality import PARAMETERS, evaluate_arc_evidence
from .cancellation import raise_if_cancelled
from .public_ellipse import canonical_ellipse_payload, observed_arc_radius_period
from .serialization import strict_jsonable


def _read(value, name, default=None):
    return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)


def _qmap_unit(qmap: Any) -> str:
    """Read a declared q unit without inferring one from numeric arrays.

    PONI ``GeometryMaps`` keep the unit in ``metadata`` rather than a public
    constructor field.  Treating that as missing made calibrated frames report
    ``unknown`` and drop Ln/Lz.
    """

    unit = _read(qmap, "q_unit", None)
    if unit in (None, "", "unknown"):
        metadata = _read(qmap, "metadata", {}) or {}
        if isinstance(metadata, Mapping):
            nested = metadata.get("q_unit", metadata.get("unit"))
            if nested not in (None, ""):
                unit = nested
    return str(unit or "unknown")


def _image(frame):
    if isinstance(frame, np.ndarray):
        return np.asarray(frame, dtype=float)
    value = _read(frame, "data", _read(frame, "image", None))
    if value is None:
        raise ValueError("butterfly analysis requires a two-dimensional image")
    return np.asarray(value, dtype=float)


def _attach_candidate_diagnostics(candidate, source):
    """Retain observed/arc diagnostics when no optimizer result exists."""

    if not isinstance(source, Mapping):
        return candidate
    for target, names in (
        ("point_diagnostics", ("point_diagnostics", "points")),
        ("arc_diagnostics", ("arc_diagnostics", "arcs")),
    ):
        for name in names:
            if name in source and source[name] is not None:
                candidate[target] = source[name]
                break
    if "branch_swap_applied" in source:
        candidate["branch_swap_applied"] = bool(source["branch_swap_applied"])
    if "diagnostics" in source and source["diagnostics"] is not None:
        candidate["trace_diagnostics"] = source["diagnostics"]
    return candidate


def _empty_candidate(unit, reference, message="No fitted candidate", *, diagnostics=None):
    candidate = {"success": False, "status": "not_fitted", "solver_status": "not_run",
                 "message": message, "q_unit": unit, "reference_axis_deg": reference,
                 "a": None, "b": None, "axis_ratio": None, "theta_deg": None,
                 "center_qx": 0.0, "center_qy": 0.0, "parameters": {}, "ellipses": [],
                 "n_points": 0, "rmse": None, "condition": None, "flags": ["arc_fit_unavailable"]}
    return _attach_candidate_diagnostics(candidate, diagnostics)


def _trace_reference_center(trace: Mapping[str, Any]) -> tuple[float, float] | None:
    """Read the tracer's finite observed center without inventing one."""

    diagnostics = trace.get("diagnostics")
    sources = [diagnostics, trace]
    for source in sources:
        if not isinstance(source, Mapping):
            continue
        candidates = [source.get("center_q"), source.get("center")]
        if source.get("center_qx") is not None or source.get("center_qy") is not None:
            candidates.append((source.get("center_qx"), source.get("center_qy")))
        for candidate in candidates:
            if isinstance(candidate, Mapping):
                candidate = (
                    candidate.get("qx", candidate.get("x")),
                    candidate.get("qy", candidate.get("y")),
                )
            if isinstance(candidate, (str, bytes)):
                continue
            try:
                if len(candidate) != 2:
                    continue
                center = (float(candidate[0]), float(candidate[1]))
            except (TypeError, ValueError, IndexError):
                continue
            if all(np.isfinite(value) for value in center):
                return center
    return None


def _fit_trace(trace, *, parameters, reference, multistart, unit, cancel_event, max_nfev=800):
    from .arc_geometry import fit_arc_ellipses

    supported = [p for p in trace.get("points", [])
                 if p.get("accepted", p.get("valid", False))
                 and p.get("branch_id") in (0, 1) and p.get("side") in ("upper", "lower")]
    if len(supported) < 5 or len({p["branch_id"] for p in supported}) < 2:
        return _empty_candidate(
            unit,
            reference,
            "Insufficient independently assigned arcs",
            diagnostics=trace,
        )
    try:
        reference_center = _trace_reference_center(trace)
        result = fit_arc_ellipses(trace["points"], parameters=parameters,
                                  reference_axis_deg=reference, multistart=multistart,
                                  max_nfev=max_nfev, cancel_event=cancel_event,
                                  reference_center=reference_center)
    except ValueError as exc:
        # Keep the observed trace and candidate diagnostics available to the
        # correction/export boundary when the optimizer rejects its input.
        return _empty_candidate(unit, reference, str(exc), diagnostics=trace)
    fit = result.get("fit")
    if fit is None:
        return _empty_candidate(
            unit,
            reference,
            result.get("message", "Arc fit unavailable"),
            diagnostics=result,
        )
    point_diagnostics = result.get("point_diagnostics", [])
    if isinstance(point_diagnostics, Mapping):
        point_diagnostics = list(point_diagnostics.values())
    distances = [p["distance_q"] for p in point_diagnostics
                 if p.get("distance_q") is not None and np.isfinite(p["distance_q"])]
    rmse = float(np.sqrt(np.mean(np.square(distances)))) if distances else None
    swap = bool(result.get("branch_swap_applied", False))
    payload = canonical_ellipse_payload(
        fit,
        n_points=len(supported),
        qmap={"q_unit": unit},
        config={"analysis": {"draw_axis_deg": float(reference) + 90.0}},
    )
    theta_deg = payload.get("theta_deg")
    try:
        theta_deg = float(theta_deg)
    except (TypeError, ValueError):
        theta_deg = float("nan")
    values = payload.get("parameters", {})
    values = dict(values) if isinstance(values, Mapping) else {}
    members = []
    for branch, sign in enumerate((1, -1)):
        observed_branch = 1 - branch if swap else branch
        angle = float(reference) + sign * theta_deg
        member = {
            key: payload[key]
            for key in (
                "a", "b", "semi_major", "semi_minor", "axis_ratio",
                "center_qx", "center_qy", "q_unit", "eccentricity",
                "ellipticity", "reference_axis_deg",
                "L_N", "L_z", "Ln_from_minor_axis_nm", "Lz_from_draw_axis_nm",
                "L_from_major_axis_nm",
                "q_star_from_arcs", "L_from_observed_radius_nm",
            )
            if key in payload
        }
        member.update(
            {
                "center": [payload.get("center_qx"), payload.get("center_qy")],
                "angle_deg": angle,
                "theta_deg": angle,
                "branch_id": observed_branch,
                "fit_branch_id": branch,
            }
        )
        members.append(member)
    diagnostics = trace.get("diagnostics") if isinstance(trace, Mapping) else {}
    first_order = diagnostics.get("first_order_q_hint") if isinstance(diagnostics, Mapping) else {}
    hint = first_order.get("q_star") if isinstance(first_order, Mapping) else None
    q_star, radius_period, radius_flags = observed_arc_radius_period(
        supported, unit, first_order_q=hint
    )
    bound_flags = getattr(fit, "bound_flags", {}) or {}
    extra_flags = list(radius_flags)
    if bound_flags.get("axis_ratio"):
        extra_flags.append("axis_ratio_at_bound")
    payload["q_star_from_arcs"] = q_star
    payload["L_from_observed_radius_nm"] = radius_period
    payload["q_star_source"] = (
        "first_order_iq"
        if "spacing_from_first_order_iq" in radius_flags
        else "observed_arc_radius"
    )
    values["q_star_from_arcs"] = q_star
    values["L_from_observed_radius_nm"] = radius_period
    values["q_star_source"] = payload["q_star_source"]
    for member in members:
        member["q_star_from_arcs"] = q_star
        member["L_from_observed_radius_nm"] = radius_period
        member["q_star_source"] = payload["q_star_source"]
    payload.update(
        {
            "success": bool(fit.success),
            "status": "ok" if fit.success else "failed",
            "solver_status": "ok" if fit.success else "failed",
            "message": fit.message,
            "parameters": values,
            "parameter_values": dict(values),
            "ellipses": members,
            "q_unit": unit,
            "n_points": len(supported),
            "rmse": rmse,
            "residual_rms": rmse,
            "rss": float(np.sum(np.square(distances))) if distances else None,
            "point_diagnostics": point_diagnostics,
            "branch_swap_applied": swap,
            "branch_label_mapping": {
                "fit_branch_to_observed": {
                    str(branch): 1 - branch if swap else branch
                    for branch in (0, 1)
                }
            },
            "arc_diagnostics": result.get("arc_diagnostics", []),
            "bound_flags": dict(bound_flags),
            "flags": list(
                dict.fromkeys(
                    tuple(payload.get("flags", ()))
                    + (
                        "fixed_observed_arc_topology",
                        "local_covariance_not_complete_uncertainty",
                    )
                    + tuple(extra_flags)
                )
            ),
        }
    )
    # Normalize only the returned public copy.  The image/q-map and all
    # intermediate arrays above remain NumPy values during numerical work.
    return strict_jsonable(payload)


def _q_step(qmap):
    try:
        from .ridge_inputs import representative_q_step

        return representative_q_step(qmap)
    except (ImportError, TypeError, ValueError):
        # Keep a compatibility fallback for source checkouts where the shared
        # input adapter is not installed yet.  The normal package path above
        # uses both detector axes.
        pass
    x = np.asarray(_read(qmap, "qx"), dtype=float)
    y = np.asarray(_read(qmap, "qy"), dtype=float)
    steps = np.concatenate(
        (
            np.hypot(np.diff(x, axis=1), np.diff(y, axis=1)).ravel(),
            np.hypot(np.diff(x, axis=0), np.diff(y, axis=0)).ravel(),
        )
    )
    finite = steps[np.isfinite(steps) & (steps > 0)]
    return float(np.median(finite)) if finite.size else None


_GEOMETRY_BOUND_ALIASES = {
    "center_x": "cx",
    "center_y": "cy",
    "x0": "cx",
    "y0": "cy",
    "ratio": "axis_ratio",
    "theta_deg": "theta",
}
_GEOMETRY_BOUND_NAMES = {"cx", "cy", "a", "axis_ratio", "theta", "b"}
_BOUND_SHIFT_FRACTION = 0.5


def _finite_bound(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _prior_parameter_entries(parameters):
    """Return caller-supplied geometry specs and explicit-bound metadata."""

    if parameters is None:
        return [], [{"reason": "no_explicit_parameters"}]
    if hasattr(parameters, "spec_items"):
        source = list(parameters.spec_items())
    elif isinstance(parameters, Mapping):
        source = list(parameters.items())
    else:
        return [], [{"reason": "unsupported_parameter_container"}]

    entries = []
    skipped = []
    seen = set()
    for raw_name, raw_spec in source:
        raw_name = str(raw_name)
        canonical = _GEOMETRY_BOUND_ALIASES.get(raw_name, raw_name)
        if canonical not in _GEOMETRY_BOUND_NAMES:
            continue
        if canonical in seen:
            skipped.append({"parameter": canonical, "reason": "duplicate_parameter_alias"})
            continue
        seen.add(canonical)
        if isinstance(raw_spec, Mapping):
            has_min = "min" in raw_spec
            has_max = "max" in raw_spec
            value = raw_spec.get("value")
            vary = raw_spec.get("vary", True)
            expr = raw_spec.get("expr")
            lower = raw_spec.get("min")
            upper = raw_spec.get("max")
        else:
            has_min = hasattr(raw_spec, "min")
            has_max = hasattr(raw_spec, "max")
            value = getattr(raw_spec, "value", raw_spec)
            vary = getattr(raw_spec, "vary", True)
            expr = getattr(raw_spec, "expr", None)
            lower = getattr(raw_spec, "min", None)
            upper = getattr(raw_spec, "max", None)
        finite_min = _finite_bound(lower) if has_min else None
        finite_max = _finite_bound(upper) if has_max else None
        if finite_min is None and finite_max is None:
            continue
        if canonical == "b":
            skipped.append({"parameter": canonical, "reason": "derived_parameter_not_applicable"})
            continue
        current = _finite_bound(value)
        entries.append(
            {
                "raw_name": raw_name,
                "parameter": canonical,
                "value": current,
                "vary": bool(vary),
                "expr": expr,
                "min": finite_min,
                "max": finite_max,
                "has_min": has_min,
                "has_max": has_max,
                "theta_degrees": raw_name == "theta_deg",
            }
        )
    if not entries and not skipped:
        skipped.append({"reason": "no_finite_explicit_bounds"})
    return entries, skipped


def _prior_physical_bounds(entry):
    parameter = entry["parameter"]
    if parameter == "a":
        return np.finfo(float).eps, None
    if parameter == "axis_ratio":
        return np.finfo(float).eps, 1.0
    if parameter == "theta":
        return (0.0, 90.0) if entry["theta_degrees"] else (0.0, 0.5 * np.pi)
    return None, None


def _copy_with_prior_bounds(parameters, entry, lower, upper):
    """Clone a mapping/ParameterSet while changing only the selected bounds."""

    clone = deepcopy(parameters)
    raw_name = entry["raw_name"]
    if isinstance(clone, Mapping):
        raw_spec = clone[raw_name]
        if isinstance(raw_spec, Mapping):
            updated = dict(raw_spec)
            if entry["has_min"]:
                updated["min"] = lower
            if entry["has_max"]:
                updated["max"] = upper
            clone[raw_name] = updated
        elif hasattr(raw_spec, "copy"):
            clone[raw_name] = raw_spec.copy(min=lower, max=upper)
        else:
            return None
        return clone
    try:
        spec = clone[raw_name]
        clone[raw_name] = spec.copy(min=lower, max=upper)
    except (KeyError, TypeError, AttributeError):
        return None
    return clone


def _explicit_prior_bound_variants(parameters):
    """Build modest contracted/expanded variants for explicit free bounds."""

    entries, skipped = _prior_parameter_entries(parameters)
    variants = []
    for entry in entries:
        parameter = entry["parameter"]
        if entry["expr"] is not None or not entry["vary"]:
            skipped.append({"parameter": parameter, "reason": "fixed_or_tied"})
            continue
        current = entry["value"]
        lower = entry["min"]
        upper = entry["max"]
        if current is None:
            skipped.append({"parameter": parameter, "reason": "nonfinite_value"})
            continue
        if lower is not None and current < lower or upper is not None and current > upper:
            skipped.append({"parameter": parameter, "reason": "value_outside_explicit_bounds"})
            continue
        physical_lower, physical_upper = _prior_physical_bounds(entry)
        if lower is not None and upper is not None:
            expansion_scale = upper - lower
        elif lower is not None:
            expansion_scale = current - lower
        elif upper is not None:
            expansion_scale = upper - current
        else:
            expansion_scale = None
        if expansion_scale is None or not np.isfinite(expansion_scale) or expansion_scale <= 0.0:
            expansion_scale = abs(current)
        if not np.isfinite(expansion_scale) or expansion_scale <= 0.0:
            expansion_scale = None
        for mode in ("contract", "expand"):
            new_lower, new_upper = lower, upper
            if lower is not None:
                distance = current - lower
                if mode == "contract":
                    new_lower = lower + _BOUND_SHIFT_FRACTION * distance
                else:
                    if expansion_scale is None:
                        new_lower = None
                    else:
                        new_lower = lower - _BOUND_SHIFT_FRACTION * expansion_scale
                if physical_lower is not None and new_lower is not None:
                    new_lower = max(physical_lower, new_lower)
            if upper is not None:
                distance = upper - current
                if mode == "contract":
                    new_upper = upper - _BOUND_SHIFT_FRACTION * distance
                else:
                    if expansion_scale is None:
                        new_upper = None
                    else:
                        new_upper = upper + _BOUND_SHIFT_FRACTION * expansion_scale
                if physical_upper is not None and new_upper is not None:
                    new_upper = min(physical_upper, new_upper)
            if (
                new_lower is not None
                and new_upper is not None
                and new_lower >= new_upper
            ):
                skipped.append({"parameter": parameter, "mode": mode, "reason": "empty_changed_interval"})
                continue
            if new_lower is not None and current < new_lower:
                skipped.append({"parameter": parameter, "mode": mode, "reason": "value_below_changed_bound"})
                continue
            if new_upper is not None and current > new_upper:
                skipped.append({"parameter": parameter, "mode": mode, "reason": "value_above_changed_bound"})
                continue
            if mode == "expand" and expansion_scale is None:
                skipped.append({"parameter": parameter, "mode": mode, "reason": "no_dimensionally_consistent_scale"})
                continue
            if new_lower == lower and new_upper == upper:
                skipped.append({"parameter": parameter, "mode": mode, "reason": "physical_domain_clipped"})
                continue
            if new_lower is not None:
                new_lower = float(new_lower)
            if new_upper is not None:
                new_upper = float(new_upper)
            clone = _copy_with_prior_bounds(parameters, entry, new_lower, new_upper)
            if clone is None:
                skipped.append({"parameter": parameter, "mode": mode, "reason": "cannot_clone_parameter_container"})
                continue
            variants.append(
                {
                    "variant": f"prior_{parameter}_{mode}",
                    "parameter": parameter,
                    "mode": mode,
                    "parameters": clone,
                    "bound_changes": {
                        "parameter": parameter,
                        "source_name": entry["raw_name"],
                        "unit": "deg" if entry["theta_degrees"] else "native",
                        "before": {"min": lower, "max": upper},
                        "after": {"min": new_lower, "max": new_upper},
                        "value": current,
                    },
                }
            )
    return variants, skipped


def _holdout_prediction(
    records: list[Mapping[str, Any]],
    values: Mapping[str, Any],
    *,
    reference_axis_deg: float,
    branch_swap_applied: bool,
) -> dict[str, Any]:
    """Project held-out records through the frozen observed-support helper."""

    from .arc_geometry import project_arc_points

    projector = project_arc_points
    total = len(records)
    try:
        raw = projector(
            records,
            values,
            reference_axis_deg=reference_axis_deg,
            branch_swap_applied=branch_swap_applied,
        )
    except (TypeError, ValueError) as exc:
        return {
            "success": False,
            "reason": f"observed_support_projection_error:{exc}",
            "n_points": total,
            "n_valid": 0,
            "n_invalid": total,
            "invalid_point_ids": [point.get("point_id", index) for index, point in enumerate(records)],
            "prediction_diagnostics": [],
        }

    if isinstance(raw, Mapping):
        rows = None
        row_sources = [raw]
        if isinstance(raw.get("projection"), Mapping):
            row_sources.append(raw["projection"])
        for source in row_sources:
            for name in (
                "point_diagnostics", "prediction_diagnostics", "predictions",
                "projected_points", "records", "points",
            ):
                candidate = source.get(name)
                if isinstance(candidate, Mapping):
                    rows = list(candidate.values())
                    break
                if isinstance(candidate, (list, tuple)):
                    rows = list(candidate)
                    break
            if rows is not None:
                break
        if rows is None:
            projection = raw.get("projection")
            source = projection if isinstance(projection, Mapping) else raw
            distances = source.get(
                "distance",
                source.get("distances", source.get("projection_distance_q")),
            )
            valid = source.get("projection_valid", source.get("valid"))
            try:
                distance_values = np.asarray(distances, dtype=float).reshape(-1)
            except (TypeError, ValueError):
                distance_values = np.asarray([], dtype=float)
            try:
                valid_values = np.asarray(valid, dtype=bool).reshape(-1) if valid is not None else None
            except (TypeError, ValueError):
                valid_values = None
            rows = [
                {
                    "distance_q": distance_values[index] if index < distance_values.size else None,
                    "projection_valid": bool(valid_values[index])
                    if valid_values is not None and index < valid_values.size
                    else None,
                }
                for index in range(max(total, int(distance_values.size)))
            ]
    elif isinstance(raw, (list, tuple)):
        rows = list(raw)
    else:
        rows = []

    rows = [
        {"distance_q": row} if isinstance(row, (int, float, np.number)) else row
        for row in rows
    ]

    # Match by source point id when supplied; otherwise preserve the helper's
    # documented source order.  Missing/duplicate rows are explicit failures.
    expected_ids = [point.get("point_id", index) for index, point in enumerate(records)]
    by_id: dict[str, Mapping[str, Any]] = {}
    ordered_rows: list[Mapping[str, Any] | None] = [None] * total
    has_ids = any(isinstance(row, Mapping) and row.get("point_id") is not None for row in rows)
    if has_ids:
        for row in rows:
            if not isinstance(row, Mapping) or row.get("point_id") is None:
                continue
            by_id.setdefault(str(row.get("point_id")), row)
        for index, point_id in enumerate(expected_ids):
            ordered_rows[index] = by_id.get(str(point_id))
    else:
        for index, row in enumerate(rows[:total]):
            ordered_rows[index] = row if isinstance(row, Mapping) else None

    diagnostics: list[dict[str, Any]] = []
    distances: list[float] = []
    invalid_ids: list[Any] = []
    for index, (point_id, row) in enumerate(zip(expected_ids, ordered_rows)):
        normalized = dict(row) if isinstance(row, Mapping) else {}
        normalized["point_id"] = point_id
        reasons: list[str] = []
        if row is None:
            reasons.append("missing_projection_record")
        projection_valid = normalized.get(
            "projection_valid",
            normalized.get("valid_projection", normalized.get("valid")),
        )
        if projection_valid is False:
            reasons.append("projection_invalid")
        if normalized.get("support_valid") is False or normalized.get("support_infeasible") is True:
            reasons.append("support_invalid")
        support_status = normalized.get("support_status")
        if support_status not in (None, "frozen", "manual_only"):
            reasons.append(f"support_{support_status}")
        distance = normalized.get(
            "distance_q",
            normalized.get(
                "projection_distance_q",
                normalized.get(
                    "projection_residual_q",
                    normalized.get("distance", normalized.get("residual_q")),
                ),
            ),
        )
        try:
            distance_value = float(distance)
        except (TypeError, ValueError):
            distance_value = float("nan")
        if not np.isfinite(distance_value):
            reasons.append("projection_distance_unavailable")
        if reasons:
            normalized["prediction_valid"] = False
            normalized["prediction_reasons"] = list(dict.fromkeys(reasons))
            invalid_ids.append(point_id)
        else:
            normalized["prediction_valid"] = True
            normalized["distance_q"] = distance_value
            distances.append(distance_value)
        diagnostics.append(normalized)

    if isinstance(raw, Mapping) and (
        raw.get("success") is False
        or str(raw.get("status", "")).casefold() in {"failed", "error"}
    ):
        for normalized in diagnostics:
            if normalized.get("prediction_valid"):
                normalized["prediction_valid"] = False
                normalized["prediction_reasons"] = ["projector_reported_failure"]
                invalid_ids.append(normalized["point_id"])
        distances = []

    invalid_ids = list(dict.fromkeys(invalid_ids))
    return {
        "success": bool(total > 0 and not invalid_ids),
        "predictive_rmse_q": (
            float(np.sqrt(np.mean(np.square(distances))))
            if total > 0 and len(distances) == total and not invalid_ids
            else None
        ),
        "n_points": total,
        "n_valid": total - len(invalid_ids),
        "n_invalid": len(invalid_ids),
        "invalid_point_ids": invalid_ids,
        "prediction_diagnostics": diagnostics,
        "reason": None if not invalid_ids else "held_out_point_support_or_projection_invalid",
    }


def _sensitivity(image, qmap, window, *, mask, options, parameters, reference,
                 multistart, cancel_event, trace):
    """Rerun image-to-arcs after choices change; never bootstrap point labels."""
    variants = []
    qstep = _q_step(qmap)
    if qstep is not None and window[1] - window[0] > 4 * qstep:
        variants.extend([("q_contract", [window[0] + qstep, window[1] - qstep], {}),
                         ("q_expand", [max(0., window[0] - qstep), window[1] + qstep], {})])
    scales = options.get("smoothing_scales", options.get("scales_px", [1.2, 2.2, 3.6]))
    variants.extend((f"smoothing_{factor}", window,
                     {"smoothing_scales": [float(x) * factor for x in scales]})
                    for factor in (0.75, 1.25))
    if qstep is not None:
        for axis in ("center_qx", "center_qy"):
            for sign in (-1, 1):
                variants.append((f"{axis}_{sign}", window,
                                 {axis: float(options.get(axis, 0.)) + sign * qstep * .5}))
    variants.extend([("mask_dilate_1", window, {}), ("mask_dilate_2", window, {})])
    prior_variants, prior_skipped = _explicit_prior_bound_variants(parameters)
    records = []
    for name, domain, changed in variants:
        raise_if_cancelled(cancel_event, "butterfly:sensitivity")
        setting = {**options, **changed, "resamples": 0, "sensitivity": False}
        variant_mask = mask
        if name.startswith("mask_dilate_"):
            from scipy.ndimage import binary_dilation

            base_mask = np.zeros(image.shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool).copy()
            qmask = _read(qmap, "mask", None)
            if qmask is not None:
                base_mask |= np.asarray(qmask, dtype=bool)
            valid = _read(qmap, "valid_mask", None)
            if valid is not None:
                base_mask |= ~np.asarray(valid, dtype=bool)
            variant_mask = binary_dilation(base_mask, iterations=int(name.rsplit("_", 1)[1]))
        shifted_parameters = deepcopy(parameters)
        if shifted_parameters is None and any(key in changed for key in ("center_qx", "center_qy")):
            shifted_parameters = {pname: {"value": float(setting.get(qname, 0.)), "vary": False}
                                  for qname, pname in (("center_qx", "cx"), ("center_qy", "cy"))}
        if shifted_parameters is not None:
            for qname, pname in (("center_qx", "cx"), ("center_qy", "cy")):
                if qname in changed:
                    shifted_parameters[pname] = {"value": changed[qname], "vary": False}
        result = analyze_butterfly(image, qmap, domain, mask=variant_mask, options=setting,
                                   parameters=shifted_parameters, reference_axis_deg=reference,
                                   multistart=multistart, cancel_event=cancel_event)
        fit = result["candidate_fit"]
        records.append({"variant": name, "success": bool(fit.get("success")),
                        "parameters": {key: fit.get(key) for key in PARAMETERS}})

    prior_records = []
    for prior in prior_variants:
        raise_if_cancelled(cancel_event, "butterfly:sensitivity:prior_bounds")
        setting = {**options, "resamples": 0, "sensitivity": False}
        try:
            result = analyze_butterfly(
                image,
                qmap,
                window,
                mask=mask,
                options=setting,
                parameters=prior["parameters"],
                reference_axis_deg=reference,
                multistart=multistart,
                cancel_event=cancel_event,
            )
            fit = result["candidate_fit"]
            record = {
                "variant": prior["variant"],
                "kind": "prior_bound",
                "success": bool(fit.get("success")),
                "parameters": {key: fit.get(key) for key in PARAMETERS},
                "bound_changes": prior["bound_changes"],
            }
        except ValueError as exc:
            record = {
                "variant": prior["variant"],
                "kind": "prior_bound",
                "success": False,
                "parameters": {key: None for key in PARAMETERS},
                "bound_changes": prior["bound_changes"],
                "error": str(exc),
            }
        records.append(record)
        prior_records.append(record)
    # Held-out arcs are not relabelled or used to create the training geometry.
    from .arc_geometry import fit_arc_ellipses

    held_out = []
    points = trace.get("points", [])
    reference_center = _trace_reference_center(trace)
    arc_ids = sorted({p.get("arc_id") for p in points if p.get("accepted") and p.get("arc_id") is not None})
    # Every observed arc is part of the identifiability check.  A previous
    # first-eight shortcut silently converted untested arcs into apparent
    # sensitivity coverage, which made the evidence status depend on point
    # ordering rather than observed topology.
    for arc_id in arc_ids:
        raise_if_cancelled(cancel_event, "butterfly:holdout_arc")
        training = [p for p in points if p.get("arc_id") != arc_id]
        held = [p for p in points if p.get("arc_id") == arc_id and p.get("accepted")
                and p.get("branch_id") in (0, 1) and p.get("side") in ("upper", "lower")]
        if not held:
            held_out.append({"arc_id": arc_id, "success": False,
                             "reason": "no_eligible_holdout_points"})
            continue
        if len(training) < 5:
            held_out.append({"arc_id": arc_id, "success": False,
                             "reason": "insufficient_training_arcs"})
            continue
        try:
            trial_bundle = fit_arc_ellipses(
                training,
                parameters=parameters,
                reference_axis_deg=reference,
                multistart=1,
                max_nfev=options.get("max_nfev", 800),
                cancel_event=cancel_event,
                reference_center=reference_center,
            )
            trial = trial_bundle.get("fit")
        except ValueError as exc:
            held_out.append({"arc_id": arc_id, "success": False, "reason": str(exc)})
            continue
        if trial is None or not trial.success:
            held_out.append({"arc_id": arc_id, "success": False, "reason": "training_arcs_not_identified"})
            continue
        prediction = _holdout_prediction(
            held,
            trial.values,
            reference_axis_deg=reference,
            branch_swap_applied=bool(trial_bundle.get("branch_swap_applied", False)),
        )
        raise_if_cancelled(cancel_event, "butterfly:holdout_prediction")
        held_out.append({"arc_id": arc_id, **prediction})
    ranges = {}
    for key in PARAMETERS:
        values = [r["parameters"][key] for r in records if r["success"]
                  and r["parameters"][key] is not None and np.isfinite(r["parameters"][key])]
        if values:
            ranges[key] = [min(values), max(values)]
    if parameters is None:
        prior_status = "not_applicable"
    elif prior_records:
        prior_status = "completed"
    else:
        prior_status = "skipped"
    explicit_prior_request = bool(
        prior_variants
        or any(
            item.get("reason") not in {"no_explicit_parameters", "no_finite_explicit_bounds"}
            for item in prior_skipped
        )
    )
    return {"completed": True, "records": records, "parameter_ranges": ranges,
            "failed_variants": [r["variant"] for r in records if not r["success"]],
            "held_out_arcs": held_out,
            "prior_bound_sensitivity": {
                "status": prior_status,
                "variants": prior_records,
                "skipped": prior_skipped,
                "explicit_bounds_requested": explicit_prior_request,
            },
            "center_perturbation_kind": "half_pixel_sensitivity_not_instrument_uncertainty"}


def analyze_butterfly(image, qmap, q_window, *, mask=None, options=None,
                      parameters=None, reference_axis_deg=0., multistart=7, cancel_event=None,
                      _radial_hint_cache=None):
    """Return serializable observed arcs, candidate geometry and evidence."""
    from .butterfly_ridge import _RadialHintGeometryCache, trace_butterfly_ridges
    from .analysis_config import DEFAULT_ANALYSIS_SETTINGS

    settings = normalize_butterfly_settings(options)
    settings.setdefault("snr_threshold", DEFAULT_ANALYSIS_SETTINGS["ridge_snr_threshold"])
    settings.setdefault("max_nfev", DEFAULT_ANALYSIS_SETTINGS["max_nfev"])
    image = _image(image)
    if image.ndim != 2:
        raise ValueError("butterfly analysis requires a 2D image")
    q_unit = _qmap_unit(qmap)
    for qname, pname in (("center_qx", "cx"), ("center_qy", "cy")):
        if isinstance(parameters, Mapping) and pname in parameters:
            spec = parameters[pname]
            settings.setdefault(qname, float(_read(spec, "value", spec)))
    radial_hint_cache = _radial_hint_cache
    if (
        radial_hint_cache is None
        and settings["stage"] == "evaluate"
        and settings["resamples"] > 0
    ):
        radial_hint_cache = _RadialHintGeometryCache()
    raise_if_cancelled(cancel_event, "butterfly:trace")
    trace = trace_butterfly_ridges(image, qmap, q_window, mask=mask,
                                  reference_axis_deg=reference_axis_deg, options=settings,
                                  edits=settings["edits"], cancel_event=cancel_event,
                                  radial_hint_cache=radial_hint_cache)
    candidate = _empty_candidate(q_unit, reference_axis_deg, diagnostics=trace)
    uncertainty = {"intervals": {}, "coverage_calibrated": False, "status": "not_run"}
    sensitivity = {"completed": False, "records": [], "held_out_arcs": []}
    if settings["stage"] == "evaluate":
        candidate = _fit_trace(trace, parameters=parameters, reference=reference_axis_deg,
                               multistart=multistart, unit=q_unit, cancel_event=cancel_event,
                               max_nfev=settings["max_nfev"])
        if candidate.get("success") and settings["resamples"] > 0:
            from .butterfly_uncertainty import resample_butterfly

            def refit(perturbed, overrides):
                overrides = dict(overrides)
                perturbed_qmap = overrides.pop("qmap", qmap)
                perturbed_mask = overrides.pop("mask", mask)
                replicate_radial_cache = radial_hint_cache
                qmap_perturbation = overrides.get("qmap_perturbation")
                if isinstance(qmap_perturbation, Mapping):
                    center_delta = qmap_perturbation.get("center_delta_px", (0.0, 0.0))
                    qcal_scale = float(qmap_perturbation.get("qcal_scale", 1.0))
                    if any(float(value) != 0.0 for value in center_delta) or qcal_scale != 1.0:
                        replicate_radial_cache = None
                replicate = analyze_butterfly(perturbed, perturbed_qmap, q_window, mask=perturbed_mask,
                    options={**settings, **overrides, "resamples": 0, "sensitivity": False},
                    parameters=parameters, reference_axis_deg=reference_axis_deg,
                    multistart=1, cancel_event=cancel_event,
                    _radial_hint_cache=replicate_radial_cache)
                groups = {(p.get("branch_id"), p.get("side")) for p in replicate.get("points", [])
                          if p.get("accepted") and p.get("side") in ("upper", "lower")
                          and p.get("branch_id") in (0, 1)}
                missing = {(0, "upper"), (0, "lower"), (1, "upper"), (1, "lower")} - groups
                replicate["topology_failed"] = bool(missing)
                if missing:
                    replicate["message"] = "resampling lost observed sides: " + repr(sorted(missing))
                return replicate

            uncertainty = resample_butterfly(image, qmap, mask=mask, q_window=q_window,
                refit=refit, options=settings, cancel_event=cancel_event)
        if candidate.get("success") and settings["sensitivity"]:
            sensitivity = _sensitivity(image, qmap, q_window, mask=mask, options=settings,
                parameters=parameters, reference=reference_axis_deg, multistart=1,
                cancel_event=cancel_event, trace=trace)
    evidence = evaluate_arc_evidence(trace, candidate, uncertainty=uncertainty, sensitivity=sensitivity)
    from .butterfly_diagnostics import ellipse_local_views, profile_residuals

    profile_residuals(trace.get("profiles", {}))
    if settings["stage"] == "trace":
        evidence["measurement_status"] = "traced"
        evidence["quality"].update(status="NOT_EVALUATED", engineering_status="NOT_EVALUATED",
                                  flags=["geometry_not_evaluated"])
        for row in evidence["quantitative_parameters"].values():
            row.update(status="not_evaluated", empirical_status="not_evaluated",
                       reason="geometry_not_evaluated", reasons=["geometry_not_evaluated"])
    payload = {**trace, **evidence, "candidate_fit": candidate, "uncertainty": uncertainty,
               "sensitivity": sensitivity, "edits": settings["edits"], "settings": settings,
               "method_version": METHOD_VERSION}
    payload["ellipse_local"] = ellipse_local_views(trace.get("points", []), candidate)
    payload["recipe_sha256"] = hashlib.sha256(json.dumps(settings, sort_keys=True,
        ensure_ascii=True, allow_nan=False).encode()).hexdigest()
    candidate.update({**evidence, "method_version": METHOD_VERSION,
                      "uncertainty": uncertainty, "sensitivity": sensitivity})
    return strict_jsonable(payload)


def measure_butterfly_observables(frame, qmap, q_window, *, mask=None, options=None,
                                 draw_axis_deg=90., ellipse_parameters=None, multistart=7,
                                 fit_ellipse=True, n_angular_bins=360, n_radial_bins=256,
                                 snr_threshold=None, cancel_event=None):
    """Adapter into the existing observable bundle without changing old methods."""
    from .observables import (AngularSpectrum, ObservableSet, _measure_lobe_radial_observables,
                              apparent_lamellar_tilt, measure_angular_spectrum,
                              measure_four_lobe_peaks)
    settings = dict(options or {})
    if not fit_ellipse:
        settings["stage"] = "trace"
    if snr_threshold is not None:
        settings.setdefault("snr_threshold", snr_threshold)
    result = analyze_butterfly(frame, qmap, q_window, mask=mask, options=settings,
                              parameters=ellipse_parameters, reference_axis_deg=draw_axis_deg - 90.,
                              multistart=multistart, cancel_event=cancel_event)
    # Landmarks are independent display/measurement diagnostics. Calculate
    # them once at this public adapter, never in each uncertainty refit.
    from .peak_landmarks import compute_peak_landmarks
    from .ridge_inputs import canonical_inputs
    from .butterfly_ridge import _parse_q_window, _apply_edits

    peak_image, peak_qx, peak_qy, peak_q, peak_invalid = canonical_inputs(frame, qmap, mask=mask)
    peak_window = _parse_q_window(q_window, peak_q)
    peak_valid = (~peak_invalid & np.isfinite(peak_image) & np.isfinite(peak_qx)
                  & np.isfinite(peak_qy) & np.isfinite(peak_q)
                  & (peak_q >= peak_window[0]) & (peak_q <= peak_window[1]))
    peak_valid, peak_edits, _ = _apply_edits(peak_valid, peak_qx, peak_qy, result.get("edits", []))
    peak_hint = result.get("diagnostics", {}).get("first_order_q_hint", {})
    signal_window = peak_hint.get("band") if peak_hint.get("selection_status") == "selected" else None
    peak_options = dict(settings.get("peak_landmark_options") or {})
    if signal_window is not None:
        peak_options["signal_q_window_origin"] = (
            f"first_order_q_hint:{peak_hint.get('q_star')}; method={peak_hint.get('selection_method')}"
        )
    result["peak_landmarks"] = compute_peak_landmarks(
        peak_image, peak_qx, peak_qy, valid_mask=peak_valid, q_window=peak_window,
        signal_q_window=signal_window, q_unit=_qmap_unit(qmap),
        options=peak_options, cancel_event=cancel_event,
    )
    result["peak_landmarks"]["domain"]["applied_polygon_edits"] = peak_edits
    unit = str(_read(qmap, "q_unit", "unknown"))
    if settings.get("companion_observables", True):
        raise_if_cancelled(cancel_event, "butterfly:angular-spectrum")
        angular = measure_angular_spectrum(frame, qmap, q_window, n_bins=n_angular_bins, mask=mask)
        raise_if_cancelled(cancel_event, "butterfly:angular-spectrum")
        lobes = measure_four_lobe_peaks(angular, symmetric_refine=True,
                                       reference_axis_deg=draw_axis_deg - 90.)
        raise_if_cancelled(cancel_event, "butterfly:lobe-peaks")
        profiles, radial_peaks = _measure_lobe_radial_observables(frame, qmap, q_window, lobes,
            n_radial_bins=n_radial_bins, snr_threshold=float(result["settings"]["snr_threshold"]),
            min_coverage=0., mask=mask, cancel_event=cancel_event)
        raise_if_cancelled(cancel_event, "butterfly:lobe-profiles")
        tilt, spread = apparent_lamellar_tilt(lobes, draw_axis_deg=draw_axis_deg)
        raise_if_cancelled(cancel_event, "butterfly:lobe-summary")
    else:
        angular = AngularSpectrum(
            angle=np.zeros(0, dtype=float),
            intensity=np.zeros(0, dtype=float),
            counts=np.zeros(0, dtype=int),
            candidate_counts=np.zeros(0, dtype=int),
            coverage=np.zeros(0, dtype=float),
            q_min=float("nan"),
            q_max=float("nan"),
            q_center=float("nan"),
            flags=(),
            q_unit=unit,
        )
        lobes = []
        profiles, radial_peaks = [], []
        tilt, spread = float("nan"), float("nan")
    points = result.get("points", [])
    ridge = {"points": points, "q_unit": unit, "flags": [METHOD_VERSION],
             "valid_fraction": sum(bool(p.get("accepted")) for p in points) / max(1, len(points)),
             "method": "butterfly_curvature"}
    return ObservableSet(angular=angular, lobes=lobes, ridge=ridge,
        ellipse=result["candidate_fit"], phi_app_deg=tilt, phi_app_std_deg=spread,
        draw_axis_deg=draw_axis_deg, q_unit=unit, lobe_radial_profiles=profiles,
        lobe_radial_peaks=radial_peaks, butterfly=result)
