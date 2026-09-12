"""Separate observed-ridge fits from the geometry of a full-pixel model.

Neither layer is a structural validation. The model ellipse is evaluated with
the same polar-radius kernel as the intensity model, including its reference
axis; angular envelopes and broadening can move the model's intensity maxima
away from that nominal ellipse.
"""
from __future__ import annotations

from collections.abc import Mapping
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .cancellation import raise_if_cancelled
from .intensity import ellipse_polar_radius, parameter_values, _parameter_mapping
from .serialization import strict_jsonable


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def assess_geometry_fit(result: Mapping[str, Any]) -> dict[str, Any]:
    """Explain an existing screen without inventing a new acceptance limit."""
    candidate = result.get("candidate_fit") or {}
    quality = result.get("quality") or {}
    if not isinstance(candidate, Mapping):
        candidate = {}
    if not isinstance(quality, Mapping):
        quality = {}
    metrics = quality.get("metrics") or {}
    limits = quality.get("provisional_limits") or {}
    ratio = _finite(metrics.get("residual_sigma_ratio")) if isinstance(metrics, Mapping) else None
    limit = _finite(limits.get("residual_sigma_ratio_max")) if isinstance(limits, Mapping) else None
    stage = str(result.get("measurement_status", "")).lower()
    flags = list(quality.get("flags") or [])
    bounds = candidate.get("bound_flags") or {}
    at_bound = any(value is True for value in bounds.values()) if isinstance(bounds, Mapping) else False
    if stage == "traced":
        status, reason = "not_evaluated", "trace_only"
    elif not candidate or candidate.get("success") is not True:
        status, reason = "unavailable", "geometry_not_converged_or_missing"
    elif ratio is not None and limit is not None and ratio > limit:
        status, reason = "poor_match", "residual_exceeds_localization_scale"
    elif at_bound or any("at_bound" in str(flag) or "at_explicit_bound" in str(flag) for flag in flags):
        status, reason = "bound_limited", "parameter_bound_active"
    else:
        status, reason = "diagnostic_candidate", "independent_validation_required"
    return {
        "geometry_status": status, "reason": reason,
        "residual_sigma_ratio": ratio, "residual_limit": limit,
        "rmse_q": _finite(candidate.get("rmse")),
        "median_localization_sigma_q": _finite(metrics.get("median_localization_sigma_q")) if isinstance(metrics, Mapping) else None,
        "solver_success": candidate.get("success"), "bound_flags": dict(bounds) if isinstance(bounds, Mapping) else {},
        "scientific_status": quality.get("scientific_status", "NOT_ACCEPTED"),
        "scientific_acceptance": False,
    }


def fit_geometry_layers(
    result: Mapping[str, Any], *, model_parameters: Any = None,
    model_reference_axis_deg: float | None = None, model_status: str | None = None,
    model_diagnostics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return separately named geometry/model curves in the supplied q units."""
    from .candidate_geometry import _candidate_curves

    if not isinstance(result, Mapping):
        raise ValueError("result must be a butterfly result mapping")
    assessment = assess_geometry_fit(result)
    state, curves, metadata = _candidate_curves(result)
    geometry_curves = [
        {**curve, "source": "geometry", "role": "observed_ridge_fit_candidate", "label": f"Geometry candidate {index + 1}"}
        for index, curve in enumerate(curves)
    ]
    candidate = result.get("candidate_fit") or {}
    geometry = {
        "curves": geometry_curves, "status": assessment["geometry_status"],
        "reason": assessment["reason"], "candidate_state": state,
        "parameters": {name: _finite(candidate.get(name)) for name in ("a", "b", "axis_ratio", "theta_deg", "center_qx", "center_qy", "reference_axis_deg")} if isinstance(candidate, Mapping) else {},
        "metadata": metadata,
    }
    model_layer: dict[str, Any] = {"curves": [], "status": "unavailable", "reason": "model_parameters_or_reference_missing", "parameters": {}}
    if model_parameters is not None:
        # The live GUI service returns parameter specifications with a resolved
        # value, while saved pipeline JSON carries plain scalars.
        source = {
            name: value.get("value") if isinstance(value, Mapping) else value
            for name, value in model_parameters.items()
        } if isinstance(model_parameters, Mapping) else model_parameters
        # Pipeline output carries verified degree aliases alongside radians.
        # Accept redundant output units only when they agree; do not choose an
        # arbitrary angle from conflicting user/model records.
        units_consistent = True
        if isinstance(source, dict):
            for radians_name in ("theta", "lobe_angle", "angular_width"):
                degree_name = radians_name + "_deg"
                if radians_name in source and degree_name in source:
                    rad, deg = _finite(source[radians_name]), _finite(source[degree_name])
                    if rad is None or deg is None or not math.isclose(rad, math.radians(deg), rel_tol=1e-10, abs_tol=1e-12):
                        units_consistent = False
                        break
                    source.pop(degree_name)
        try:
            supplied = _parameter_mapping(source) if units_consistent else {}
            required_present = "a" in supplied and ("b" in supplied or "axis_ratio" in supplied) and "theta" in supplied
            values = parameter_values(source) if required_present else {}
        except (TypeError, ValueError, OverflowError):
            values = {}
        a, b = _finite(values.get("a")), _finite(values.get("b"))
        theta = _finite(values.get("theta"))
        reference = _finite(model_reference_axis_deg)
        if theta is None and _finite(values.get("theta_deg")) is not None:
            theta = math.radians(float(values["theta_deg"]))
        if a is not None and b is not None and a > 0.0 and b > 0.0 and theta is not None and reference is not None:
            azimuth = np.linspace(-np.pi, np.pi, 721)
            relative = azimuth - math.radians(reference)
            model_curves = []
            for index, sign in enumerate((1.0, -1.0)):
                radius = ellipse_polar_radius(relative, a, b, sign * theta)
                xy = np.column_stack((radius * np.cos(azimuth), radius * np.sin(azimuth)))
                model_curves.append({
                    "points": xy, "label": f"Intensity-model ellipse {index + 1}",
                    "branch_id": index, "source": "intensity_model",
                    "role": "nominal_intensity_model_ellipse_not_observed_ridge",
                })
            model_layer = {
                "curves": model_curves, "status": str(model_status or "unknown"),
                "reason": "empirical_model_geometry_not_measured_ridge",
                "parameters": {"a": a, "b": b, "axis_ratio": b / a, "theta_deg": math.degrees(theta), "reference_axis_deg": reference},
            }
    diagnostics = dict(model_diagnostics or {})
    model_layer["diagnostics"] = {
        "condition_number": _finite(diagnostics.get("condition_number")),
        "rmse": _finite(diagnostics.get("rmse")),
        "bound_flags": dict(diagnostics.get("bound_flags") or {}),
        "effective_bounds": dict(diagnostics.get("effective_bounds") or {}),
        "scientific_acceptance": False,
    }
    return {"geometry": geometry, "intensity_model": model_layer, "assessment": assessment}


def _reference_from_context(context: Mapping[str, Any]) -> float | None:
    explicit = _finite(context.get("pixel_model_reference_axis_deg"))
    if explicit is not None:
        return explicit
    analysis = context.get("analysis")
    if isinstance(analysis, Mapping):
        draw = _finite(analysis.get("draw_axis_deg"))
        if draw is not None:
            return draw - 90.0
    return None


def export_fit_overlays(stage: str | Path, *, data: Mapping[str, Any], cancel_event: Any = None) -> tuple[dict[str, Path], dict[str, Any]]:
    """Write uncluttered, source-labelled overlays into the parent transaction."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    import matplotlib as mpl
    from .figure_support import _draw_map, _EXPORT_LOCK, _FIGURE_RC

    stage = Path(stage)
    context = data.get("context") or {}
    if not isinstance(context, Mapping):
        context = {}
    layers = fit_geometry_layers(
        data["result_safe"], model_parameters=context.get("pixel_model_parameters"),
        model_reference_axis_deg=_reference_from_context(context), model_status=context.get("pixel_model_status"),
        model_diagnostics={
            "condition_number": context.get("pixel_model_condition_number"),
            "rmse": context.get("pixel_model_rmse"),
            "bound_flags": context.get("pixel_model_bound_flags"),
            "effective_bounds": context.get("pixel_model_effective_bounds"),
        },
    )
    outputs: dict[str, Path] = {}
    width = float(data["width_mm"])
    height = 89.0 if width == 89.0 else 125.0
    colors = {"geometry": ("#CC79A7", "#D55E00"), "intensity_model": ("#0072B2", "#009E73")}
    captions = {
        "measured_only": "Observed intensity on the supplied q map and valid domain. Display transforms do not change source intensities.",
        "geometry_overlay": "Observed intensity with the observed-ridge geometry candidate. Full dashed ellipses include unobserved extrapolation. A poor or bound-limited fit is not a quantitatively accepted geometry.",
        "intensity_model_overlay": "Observed intensity with nominal ellipses from the actual full2d model parameters, evaluated by its polar-radius kernel. Angular envelopes, radial broadening and overlapping components can displace intensity maxima from these ellipses. These are not measured ridge ellipses.",
    }
    for name, source in (("measured_only", None), ("geometry_overlay", "geometry"), ("intensity_model_overlay", "intensity_model")):
        layer = layers[source] if source is not None else None
        if source == "intensity_model" and not layer["curves"]:
            continue
        raise_if_cancelled(cancel_event, f"fit-overlay:{name}")
        with _EXPORT_LOCK, mpl.rc_context(_FIGURE_RC):
            fig = Figure(figsize=(width / 25.4, height / 25.4), dpi=data["dpi"])
            FigureCanvasAgg(fig)
            ax = fig.add_axes([.17, .19, .63, .69])
            cax = fig.add_axes([.83, .24, .025, .56])
            _draw_map(fig, ax, cax, data, draw_overlays=False)
            ax.set_gid(name)
            if layer is None:
                title = "Measured intensity"
            elif layer["curves"]:
                for index, curve in enumerate(layer["curves"]):
                    xy = curve["points"]
                    ax.plot(xy[:, 0], xy[:, 1], color=colors[source][index % 2], lw=.9, ls="--", label=curve["label"])
                title = "Geometry candidate · " + layer["status"].replace("_", " ") if source == "geometry" else "Intensity-model geometry · " + layer["status"].replace("_", " ")
                ax.legend(loc="upper center", bbox_to_anchor=(.5, -.18), ncol=1 if width == 89 else 2, frameon=False, fontsize=5.5)
            else:
                title = "Geometry fit unavailable"
            ax.set_title(title, fontsize=6.5, pad=5)
            if source == "geometry":
                ratio, limit = layers["assessment"]["residual_sigma_ratio"], layers["assessment"]["residual_limit"]
                if ratio is not None:
                    text = f"Residual / localization scale = {ratio:.2g}"
                    if limit is not None:
                        text += f"; existing screen {limit:g}"
                    fig.text(.17, .94, text, fontsize=5.5)
            elif source == "intensity_model":
                diagnostics = layer["diagnostics"]
                condition = diagnostics["condition_number"]
                flags = [name for name, active in diagnostics["bound_flags"].items() if active is True]
                details = []
                if condition is not None:
                    details.append(f"Condition number {condition:.2g}")
                if flags:
                    details.append("Parameter bounds active")
                if details:
                    fig.text(.17, .94, "; ".join(details), fontsize=5.5)
            for extension in ("svg", "pdf", "png", "tiff"):
                raise_if_cancelled(cancel_event, f"fit-overlay:{name}:{extension}")
                path = stage / f"{name}.{extension}"
                kwargs = {"dpi": data["dpi"]} if extension in {"png", "tiff"} else {}
                fig.savefig(path, format=extension, facecolor="white", **kwargs)
                outputs[f"{name}_{extension}"] = path
            fig.clear()
        caption_path = stage / f"{name}_caption.txt"
        caption_path.write_text(captions[name] + "\n", encoding="utf-8")
        outputs[f"{name}_caption"] = caption_path
    assessment_path = stage / "fit_assessment.json"
    compact = {"assessment": layers["assessment"], **{key: {k: v for k, v in layers[key].items() if k != "curves"} for key in ("geometry", "intensity_model")}}
    assessment_path.write_text(json.dumps(strict_jsonable(compact), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    outputs["fit_assessment"] = assessment_path
    csv_path = stage / "fit_source_parameters.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("fit_source", "parameter", "value", "unit", "status"))
        for source in ("geometry", "intensity_model"):
            for name, value in layers[source]["parameters"].items():
                unit = "deg" if name.endswith("deg") else "dimensionless" if name == "axis_ratio" else data["q_unit"]
                writer.writerow((source, name, "" if value is None else value, unit, layers[source]["status"]))
    outputs["fit_source_parameters"] = csv_path
    curve_path = stage / "fit_overlay_curves.npz"
    arrays = {f"{source}_{i}": curve["points"] for source in ("geometry", "intensity_model") for i, curve in enumerate(layers[source]["curves"])}
    np.savez_compressed(curve_path, **arrays)
    outputs["fit_overlay_curves"] = curve_path
    return outputs, strict_jsonable(compact)
