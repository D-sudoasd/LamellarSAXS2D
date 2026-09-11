"""Independent observed ridge and arc extraction for butterfly SAXS images.

This module intentionally stops at the observed topology.  It finds connected
intensity ridges in the measured image, records uncertain/rejected candidates,
and supplies local normal profiles for review.  No ellipse is fitted and no
missing quadrant or side is manufactured here.

The detector image is treated as a height surface over *scaled physical q*
coordinates.  Principal curvatures are obtained from the first and second
fundamental forms of the graph surface.  The smallest principal curvature,
its principal direction, and a zero of the directional slope define a ridge
candidate.  This is deliberately more explicit than using a Hessian
eigenvalue as a proxy for curvature.
"""

from __future__ import annotations

import time as _time

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import time
from typing import Any

import numpy as np

try:  # scipy is a declared project dependency.
    from scipy.ndimage import gaussian_filter
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover - source inspection in partial installs
    gaussian_filter = None
    cKDTree = None

from .ridge_inputs import (
    as_image as _shared_as_image,
    as_qmap as _shared_as_qmap,
    coordinate_derivatives as _shared_coordinate_derivatives,
    field as _field,
    robust_noise as _robust_noise,
    sample as _sample,
)


METHOD_VERSION = "butterfly-observed-ridges-v1.0"


def _check_cancelled(cancel_event: Any, stage: str) -> None:
    if cancel_event is None:
        return
    from .cancellation import raise_if_cancelled

    raise_if_cancelled(cancel_event, stage)


def _safe_gradient(array: np.ndarray, axis: int) -> np.ndarray:
    """Finite-difference gradient that remains defined on degenerate axes."""

    if array.shape[axis] < 2:
        return np.zeros_like(array, dtype=float)
    return np.asarray(np.gradient(array, axis=axis), dtype=float)


def _offset_nearest(array: np.ndarray, drow: np.ndarray, dcol: np.ndarray) -> np.ndarray:
    """Sample ``array`` at each pixel plus a local (row, col) offset.

    The curvature stencil only needs a one-pixel sign/height check.  Full-image
    ``map_coordinates`` on a 2.5 Mpx detector was the dominant evaluate cost.
    """

    rows, cols = array.shape
    row_index = np.arange(rows, dtype=np.float64)[:, None]
    col_index = np.arange(cols, dtype=np.float64)[None, :]
    yi = np.clip(
        np.rint(np.nan_to_num(row_index - drow, nan=0.0, posinf=0.0, neginf=0.0)).astype(np.intp),
        0,
        rows - 1,
    )
    xi = np.clip(
        np.rint(np.nan_to_num(col_index - dcol, nan=0.0, posinf=0.0, neginf=0.0)).astype(np.intp),
        0,
        cols - 1,
    )
    return np.asarray(array[yi, xi], dtype=array.dtype)


def _as_image(image: Any) -> tuple[np.ndarray, np.ndarray | None]:
    return _shared_as_image(image)


def _as_qmap(qmap: Any, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    return _shared_as_qmap(qmap, shape)


def _normalise_options(options: Any) -> dict[str, Any]:
    if options is None:
        values: dict[str, Any] = {}
    elif isinstance(options, Mapping):
        values = dict(options)
    else:
        try:
            values = dict(vars(options))
        except TypeError:
            values = {}
    defaults: dict[str, Any] = {
        "smoothing_scales": (1.2, 2.2, 3.6),
        "support_min": 0.78,
        "curvature_percentile": 65.0,
        "candidate_snr_min": 3.0,
        "axis_separation_min": 0.12,
        "q_scale_anisotropy_max": 4.0,
        "normal_step_factor": 1.0,
        "max_offset_fraction": 0.75,
        "nms_radius_factor": 0.45,
        "graph_radius_factor": 3.5,
        "graph_normal_factor": 1.8,
        "graph_tangent_angle_deg": 65.0,
        "max_arc_tangent_error_deg": 45.0,
        "min_arc_points": 6,
        "max_candidates": 30000,
        "max_points": None,
        "snr_threshold": 1.5,
        "profile_half_width_q": None,
        "profile_samples": 33,
        "profile_delta_bic": 4.0,
        "profile_refinement": True,
        "profile_refinement_max_q_step": 1.5,
        "profile_refinement_min_snr": 3.0,
        "run_wang_check": True,
        "quadrant_axis_tol_deg": 4.0,
        "seed_snap_factor": 3.0,
        "center_qx": 0.0,
        "center_qy": 0.0,
        "first_order_prefer": True,
        "fill_sparse_first_order_ring": True,
    }
    merged = {**defaults, **values}
    if "scales" in values and "smoothing_scales" not in values:
        merged["smoothing_scales"] = values["scales"]
    if "smooth_scales" in values and "smoothing_scales" not in values:
        merged["smoothing_scales"] = values["smooth_scales"]
    if "scales_px" in values and "smoothing_scales" not in values:
        merged["smoothing_scales"] = values["scales_px"]
    if "min_candidate_snr" in values and "candidate_snr_min" not in values:
        merged["candidate_snr_min"] = values["min_candidate_snr"]
    if "refine_from_profile" in values and "profile_refinement" not in values:
        merged["profile_refinement"] = values["refine_from_profile"]
    if "max_points" in values and values["max_points"] is not None:
        merged["max_candidates"] = max(int(values["max_points"]), int(merged["max_candidates"]))
    center = values.get("center_hint", values.get("center_q", None))
    if center is not None:
        try:
            merged["center_qx"], merged["center_qy"] = float(center[0]), float(center[1])
        except Exception as exc:
            raise ValueError("center_hint/center_q must be a two-element sequence") from exc
    try:
        scales = tuple(sorted({float(scale) for scale in merged["smoothing_scales"]}))
    except (TypeError, ValueError) as exc:
        raise ValueError("smoothing_scales must contain finite positive values") from exc
    if not scales or any(not np.isfinite(scale) or scale <= 0 for scale in scales):
        raise ValueError("smoothing_scales must contain finite positive values")
    merged["smoothing_scales"] = scales
    for name, lower, upper in (
        ("support_min", 0.0, 1.0),
        ("axis_separation_min", 0.0, 1.0),
        ("curvature_percentile", 0.0, 100.0),
        ("nms_radius_factor", 0.05, 10.0),
        ("graph_radius_factor", 0.5, 20.0),
        ("graph_normal_factor", 0.1, 20.0),
        ("graph_tangent_angle_deg", 1.0, 90.0),
        ("max_arc_tangent_error_deg", 1.0, 90.0),
        ("q_scale_anisotropy_max", 1.0, 1e6),
        ("snr_threshold", 0.0, 1e9),
        ("candidate_snr_min", 0.0, 1e9),
        ("profile_refinement_max_q_step", 1e-6, 10.0),
        ("profile_refinement_min_snr", 0.0, 1e9),
        ("quadrant_axis_tol_deg", 0.0, 45.0),
        ("seed_snap_factor", 0.1, 100.0),
    ):
        number = float(merged[name])
        if not np.isfinite(number) or number < lower or number > upper:
            raise ValueError(f"{name} must be finite and in [{lower}, {upper}]")
        merged[name] = number
    merged["min_arc_points"] = max(2, int(merged["min_arc_points"]))
    merged["max_candidates"] = max(1, int(merged["max_candidates"]))
    if merged["max_points"] is not None:
        merged["max_points"] = max(1, int(merged["max_points"]))
    merged["profile_samples"] = max(9, int(merged["profile_samples"]))
    if not isinstance(merged["profile_refinement"], (bool, np.bool_)):
        raise ValueError("profile_refinement must be boolean")
    merged["profile_refinement"] = bool(merged["profile_refinement"])
    if "first_order_band" in values and "first_order_prefer" not in values:
        merged["first_order_prefer"] = values["first_order_band"]
    if not isinstance(merged["first_order_prefer"], (bool, np.bool_)):
        raise ValueError("first_order_prefer must be boolean")
    merged["first_order_prefer"] = bool(merged["first_order_prefer"])
    if not isinstance(merged["fill_sparse_first_order_ring"], (bool, np.bool_)):
        raise ValueError("fill_sparse_first_order_ring must be boolean")
    merged["fill_sparse_first_order_ring"] = bool(merged["fill_sparse_first_order_ring"])
    return merged


def _parse_q_window(q_window: Any, q: np.ndarray) -> tuple[float, float]:
    if q_window is None:
        finite = q[np.isfinite(q)]
        return (float(np.min(finite)), float(np.max(finite))) if finite.size else (float("nan"), float("nan"))
    if isinstance(q_window, Mapping):
        low = _field(q_window, ("q_min", "min", "low", "start"), None)
        high = _field(q_window, ("q_max", "max", "high", "stop"), None)
    else:
        try:
            low, high = q_window
        except Exception as exc:
            raise ValueError("q_window must be a (q_min, q_max) pair or mapping") from exc
    low, high = float(low), float(high)
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        raise ValueError("q_window must contain finite q_min < q_max")
    return low, high


def _polygon_mask(qx: np.ndarray, qy: np.ndarray, vertices: Any) -> np.ndarray:
    points = np.asarray(vertices, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 3 or not np.all(np.isfinite(points)):
        raise ValueError("polygon points must be a finite (n>=3, 2) array")
    inside = np.zeros(qx.shape, dtype=bool)
    x, y = qx, qy
    x0, y0 = points[-1]
    for x1, y1 in points:
        crosses = (y1 > y) != (y0 > y)
        denominator = y0 - y1
        with np.errstate(divide="ignore", invalid="ignore"):
            crossing_x = (x0 - x1) * (y - y1) / denominator + x1
        inside ^= crosses & (x < crossing_x)
        x0, y0 = x1, y1
    return inside


def _apply_edits(
    valid: np.ndarray,
    qx: np.ndarray,
    qy: np.ndarray,
    edits: Any,
) -> tuple[np.ndarray, list[dict[str, Any]], list[dict[str, Any]]]:
    if edits is None:
        return valid, [], []
    if isinstance(edits, Mapping):
        actions = edits.get("actions", edits.get("edits", []))
    else:
        actions = edits
    if isinstance(actions, Mapping) or isinstance(actions, (str, bytes)):
        actions = [actions]
    edited = np.asarray(valid, dtype=bool).copy()
    applied: list[dict[str, Any]] = []
    seed_actions: list[dict[str, Any]] = []
    for action in list(actions or []):
        if not isinstance(action, Mapping):
            continue
        kind = str(action.get("type", action.get("kind", ""))).strip().lower().replace("-", "_")
        if kind == "seed":
            seed_actions.append(dict(action))
            continue
        if kind not in {"exclude_polygon", "include_polygon"}:
            continue
        polygon = _polygon_mask(qx, qy, action.get("points", []))
        if kind == "exclude_polygon":
            edited[polygon] = False
        else:
            # Include edits override the editable exclusion state while still
            # respecting non-finite image/q coordinates at the caller.
            edited[polygon] = True
        applied.append({"type": kind, "n_pixels": int(np.count_nonzero(polygon))})
    return edited, applied, seed_actions


@dataclass
class _CurvatureField:
    sigma: float
    smooth: np.ndarray
    support: np.ndarray
    qx: np.ndarray
    qy: np.ndarray
    valid: np.ndarray
    gradient_x: np.ndarray
    gradient_y: np.ndarray
    lambda_min: np.ndarray
    lambda_max: np.ndarray
    normal_x: np.ndarray
    normal_y: np.ndarray
    tangent_x: np.ndarray
    tangent_y: np.ndarray
    normal_slope: np.ndarray
    slope_minus: np.ndarray
    slope_plus: np.ndarray
    intensity_minus: np.ndarray
    intensity_plus: np.ndarray
    offset_fraction: np.ndarray
    q_normal_step: np.ndarray
    q_scale_anisotropy: np.ndarray
    axis_separation: np.ndarray
    candidate: np.ndarray
    accepted: np.ndarray
    score: np.ndarray
    curvature_threshold: float
    baseline: float
    noise: float
    q_step: float
    height_scale: float
    coordinate_derivatives: Mapping[str, np.ndarray] | None = None


def _coordinate_derivatives(qx: np.ndarray, qy: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray, float]:
    """Return detector-to-q derivatives, local inverse Jacobian, q step."""

    return _shared_coordinate_derivatives(qx, qy)


def _scaled_surface_field(
    image: np.ndarray,
    qx: np.ndarray,
    qy: np.ndarray,
    valid: np.ndarray,
    *,
    sigma: float,
    options: Mapping[str, Any],
    coordinate_derivatives: tuple[dict[str, np.ndarray], np.ndarray, float] | None = None,
) -> _CurvatureField:
    if gaussian_filter is None:  # pragma: no cover
        raise RuntimeError("ridge tracing requires scipy.ndimage")
    weights = valid.astype(float)
    support = gaussian_filter(weights, sigma=float(sigma), mode="nearest")
    numerator = gaussian_filter(np.where(valid, image, 0.0), sigma=float(sigma), mode="nearest")
    smooth = np.divide(numerator, support, out=np.full_like(image, np.nan, dtype=float), where=support > 1e-8)
    finite_smooth = np.isfinite(smooth) & (support > 1e-8)
    fill = float(np.nanmedian(smooth[finite_smooth])) if np.any(finite_smooth) else 0.0
    surface = np.where(finite_smooth, smooth, fill)
    finite_values = surface[valid & np.isfinite(surface)]
    baseline = float(np.nanpercentile(finite_values, 10.0)) if finite_values.size else 0.0
    height_scale = float(np.nanpercentile(finite_values, 90.0) - np.nanpercentile(finite_values, 10.0)) if finite_values.size else 1.0
    noise = _robust_noise(finite_values - baseline)
    if not np.isfinite(noise) or noise <= np.finfo(float).eps:
        noise = max(float(np.nanstd(finite_values)), np.finfo(float).eps) if finite_values.size else 1.0
    height_scale = max(height_scale, 3.0 * noise, np.finfo(float).eps)
    z = (surface - baseline) / height_scale
    if coordinate_derivatives is None:
        derivatives, _inverse, q_step = _coordinate_derivatives(qx, qy)
    else:
        derivatives, _inverse, q_step = coordinate_derivatives
    inv00, inv01 = derivatives["inv00"], derivatives["inv01"]
    inv10, inv11 = derivatives["inv10"], derivatives["inv11"]
    z_y, z_x = _safe_gradient(z, 0), _safe_gradient(z, 1)
    z_yy, z_yx = _safe_gradient(z_y, 0), _safe_gradient(z_y, 1)
    _z_xy, z_xx = _safe_gradient(z_x, 0), _safe_gradient(z_x, 1)
    z_xy = 0.5 * (z_yx + _z_xy)
    with np.errstate(divide="ignore", invalid="ignore"):
        gradient_x = inv00 * z_x + inv10 * z_y
        gradient_y = inv01 * z_x + inv11 * z_y
        # Remove the coordinate-map second derivatives before transforming
        # the detector Hessian.  This is the chain rule for a graph surface,
        # rather than a pixel-grid Hessian used as a curvature surrogate.
        residual_xx = z_xx - gradient_x * derivatives["qx_xx"] - gradient_y * derivatives["qy_xx"]
        residual_xy = z_xy - gradient_x * derivatives["qx_xy"] - gradient_y * derivatives["qy_xy"]
        residual_yy = z_yy - gradient_x * derivatives["qx_yy"] - gradient_y * derivatives["qy_yy"]
        hxx = inv00 * (residual_xx * inv00 + residual_xy * inv10) + inv10 * (residual_xy * inv00 + residual_yy * inv10)
        hxy = inv00 * (residual_xx * inv01 + residual_xy * inv11) + inv10 * (residual_xy * inv01 + residual_yy * inv11)
        hyy = inv01 * (residual_xx * inv01 + residual_xy * inv11) + inv11 * (residual_xy * inv01 + residual_yy * inv11)
    # q coordinates are scaled so one typical detector q step is one unit.
    gradient_x *= q_step
    gradient_y *= q_step
    hxx *= q_step * q_step
    hxy *= q_step * q_step
    hyy *= q_step * q_step
    with np.errstate(divide="ignore", invalid="ignore"):
        graph_metric_det = 1.0 + gradient_x * gradient_x + gradient_y * gradient_y
        graph_normal = np.sqrt(graph_metric_det)
        first_e = 1.0 + gradient_x * gradient_x
        first_f = gradient_x * gradient_y
        first_g = 1.0 + gradient_y * gradient_y
        second_e = hxx / graph_normal
        second_f = hxy / graph_normal
        second_g = hyy / graph_normal
        mean_curvature = (first_e * second_g - 2.0 * first_f * second_f + first_g * second_e) / (2.0 * graph_metric_det)
        gaussian_curvature = (second_e * second_g - second_f * second_f) / graph_metric_det
        discriminant = np.sqrt(np.maximum(mean_curvature * mean_curvature - gaussian_curvature, 0.0))
        lambda_min = mean_curvature - discriminant
        lambda_max = mean_curvature + discriminant
        # Shape operator = metric^{-1} * second fundamental form.  The
        # selected eigenvector is a direction in the scaled q tangent plane.
        shape_a11 = (first_g * second_e - first_f * second_f) / graph_metric_det
        shape_a12 = (first_g * second_f - first_f * second_g) / graph_metric_det
        shape_a21 = (-first_f * second_e + first_e * second_f) / graph_metric_det
        shape_a22 = (-first_f * second_f + first_e * second_g) / graph_metric_det
        normal_x = shape_a12
        normal_y = lambda_min - shape_a11
        alternative = np.hypot(normal_x, normal_y) <= 1e-10
        normal_x = np.where(alternative, lambda_min - shape_a22, normal_x)
        normal_y = np.where(alternative, shape_a21, normal_y)
        normal_norm = np.hypot(normal_x, normal_y)
        normal_x = np.divide(normal_x, normal_norm, out=np.full_like(normal_x, np.nan), where=normal_norm > 1e-10)
        normal_y = np.divide(normal_y, normal_norm, out=np.full_like(normal_y, np.nan), where=normal_norm > 1e-10)
    tangent_x, tangent_y = -normal_y, normal_x
    with np.errstate(divide="ignore", invalid="ignore"):
        axis_separation = (lambda_max - lambda_min) / np.maximum(np.abs(lambda_max), np.finfo(float).eps)
        # One q step along the local principal normal, expressed in detector
        # pixels using the physical q-map Jacobian.
        displacement_x = q_step * (inv00 * normal_x + inv01 * normal_y)
        displacement_y = q_step * (inv10 * normal_x + inv11 * normal_y)
        q_normal_step = np.hypot(
            derivatives["qx_x"] * displacement_x + derivatives["qx_y"] * displacement_y,
            derivatives["qy_x"] * displacement_x + derivatives["qy_y"] * displacement_y,
        )
        singular_trace = derivatives["step_x"] ** 2 + derivatives["step_y"] ** 2
        singular_disc = np.sqrt(np.maximum((derivatives["step_x"] ** 2 - derivatives["step_y"] ** 2) ** 2 + 4.0 * (derivatives["qx_x"] * derivatives["qx_y"] + derivatives["qy_x"] * derivatives["qy_y"]) ** 2, 0.0))
        singular_max = np.sqrt(np.maximum((singular_trace + singular_disc) / 2.0, 0.0))
        singular_min = np.sqrt(np.maximum((singular_trace - singular_disc) / 2.0, 0.0))
        q_scale_anisotropy = singular_max / np.maximum(singular_min, np.finfo(float).eps)
        normal_slope = gradient_x * normal_x + gradient_y * normal_y
    step_factor = float(options.get("normal_step_factor", 1.0))
    sample_dx = displacement_x * step_factor
    sample_dy = displacement_y * step_factor
    slope_minus = _offset_nearest(normal_slope, sample_dy, sample_dx)
    slope_plus = _offset_nearest(normal_slope, -sample_dy, -sample_dx)
    intensity_minus = _offset_nearest(z, sample_dy, sample_dx)
    intensity_plus = _offset_nearest(z, -sample_dy, -sample_dx)
    slope_derivative = (slope_plus - slope_minus) / np.maximum(2.0 * q_step * step_factor, np.finfo(float).eps)
    offset_q = np.divide(-normal_slope, slope_derivative, out=np.full_like(normal_slope, np.nan), where=np.abs(slope_derivative) > 1e-10)
    offset_fraction = np.divide(offset_q, max(q_step * step_factor, np.finfo(float).eps))
    max_offset = float(options.get("max_offset_fraction", 0.75))
    offset_fraction = np.clip(offset_fraction, -max_offset, max_offset)
    # Mask and crop boundaries must have one complete local stencil.
    edge = np.ones(image.shape, dtype=bool)
    edge[[0, -1], :] = False
    edge[:, [0, -1]] = False
    support_ok = support >= float(options["support_min"])
    finite = (
        valid
        & support_ok
        & edge
        & np.isfinite(lambda_min)
        & np.isfinite(normal_slope)
        & np.isfinite(offset_fraction)
        & np.isfinite(q_scale_anisotropy)
    )
    concave = lambda_min < 0.0
    local_maximum = (intensity_minus <= z) & (intensity_plus <= z)
    sign_change = ((slope_minus <= 0.0) & (slope_plus >= 0.0)) | ((slope_minus >= 0.0) & (slope_plus <= 0.0))
    slope_small = np.abs(normal_slope) <= np.nanpercentile(np.abs(normal_slope[finite]), 35.0) if np.any(finite) else np.zeros_like(finite)
    signal_ok = surface >= baseline + float(options["candidate_snr_min"]) * noise
    potential = finite & concave & local_maximum & signal_ok & (sign_change | slope_small)
    strengths = -lambda_min[potential]
    threshold = float(np.nanpercentile(strengths, float(options["curvature_percentile"]))) if strengths.size else float("inf")
    threshold = max(threshold, np.finfo(float).eps)
    score = np.maximum(-lambda_min, 0.0) * np.maximum(surface - baseline, 0.0) / max(noise / height_scale, np.finfo(float).eps)
    accepted = potential & (-lambda_min >= threshold)
    accepted &= axis_separation >= float(options["axis_separation_min"])
    accepted &= q_scale_anisotropy <= float(options["q_scale_anisotropy_max"])
    return _CurvatureField(
        sigma=float(sigma),
        # ``surface`` is already the mask-normalized Gaussian signal in the
        # image's original intensity units.  ``z`` is the separately scaled
        # surface used only for derivatives/curvature; do not scale the
        # metadata surface a second time.
        smooth=surface,
        support=support,
        qx=qx,
        qy=qy,
        valid=valid,
        gradient_x=gradient_x,
        gradient_y=gradient_y,
        lambda_min=lambda_min,
        lambda_max=lambda_max,
        normal_x=normal_x,
        normal_y=normal_y,
        tangent_x=tangent_x,
        tangent_y=tangent_y,
        normal_slope=normal_slope,
        slope_minus=slope_minus,
        slope_plus=slope_plus,
        intensity_minus=intensity_minus,
        intensity_plus=intensity_plus,
        offset_fraction=offset_fraction,
        q_normal_step=q_normal_step,
        q_scale_anisotropy=q_scale_anisotropy,
        axis_separation=axis_separation,
        candidate=potential,
        accepted=accepted,
        score=score,
        curvature_threshold=threshold,
        baseline=baseline,
        noise=noise,
        q_step=q_step,
        height_scale=height_scale,
        coordinate_derivatives=derivatives,
    )


def _raw_candidates(
    field: _CurvatureField,
    *,
    row_offset: int,
    col_offset: int,
    max_candidates: int,
    options: Mapping[str, Any],
) -> list[dict[str, Any]]:
    flat = np.flatnonzero(field.candidate.ravel())
    if flat.size > max_candidates:
        ranking = np.argsort(field.score.ravel()[flat])[::-1][:max_candidates]
        flat = flat[ranking]
    if flat.size == 0:
        return []
    rows, cols = np.unravel_index(flat, field.candidate.shape)
    fraction = field.offset_fraction[rows, cols]
    # The exact detector displacement is reconstructed from q-normal direction
    # and the local q-map Jacobian.  A local finite difference is sufficient
    # here because the q map is re-sampled at the resulting coordinates.
    derivatives = field.coordinate_derivatives
    if derivatives is None:  # Compatibility for manually constructed fields.
        derivatives, _inverse, _q_step = _coordinate_derivatives(field.qx, field.qy)
    inv00 = derivatives["inv00"]
    inv01 = derivatives["inv01"]
    inv10 = derivatives["inv10"]
    inv11 = derivatives["inv11"]
    displacement_x = field.q_step * (inv00[rows, cols] * field.normal_x[rows, cols] + inv01[rows, cols] * field.normal_y[rows, cols])
    displacement_y = field.q_step * (inv10[rows, cols] * field.normal_x[rows, cols] + inv11[rows, cols] * field.normal_y[rows, cols])
    sub_x = cols.astype(float) + fraction * displacement_x
    sub_y = rows.astype(float) + fraction * displacement_y
    qx_sub = _sample(field.qx, sub_y, sub_x)
    qy_sub = _sample(field.qy, sub_y, sub_x)
    intensity = _sample(field.smooth, sub_y, sub_x)
    local_noise = max(field.noise, np.finfo(float).eps)
    local_snr = (intensity - field.baseline) / local_noise
    results: list[dict[str, Any]] = []
    for index in range(flat.size):
        row, col = int(rows[index]), int(cols[index])
        accepted = bool(field.accepted[row, col])
        reason = "accepted"
        if not accepted:
            if float(field.axis_separation[row, col]) < float(options["axis_separation_min"]):
                reason = "ambiguous_local_axes"
            elif float(field.q_scale_anisotropy[row, col]) > float(options["q_scale_anisotropy_max"]):
                reason = "unstable_q_scale"
            elif float(-field.lambda_min[row, col]) < field.curvature_threshold:
                reason = "weak_curvature"
            elif not np.isfinite(local_snr[index]) or local_snr[index] < float(options["snr_threshold"]):
                reason = "low_snr"
            else:
                reason = "ridge_candidate_rejected"
        results.append(
            {
                "qx": float(qx_sub[index]),
                "qy": float(qy_sub[index]),
                "pixel_x": float(sub_x[index] + col_offset),
                "pixel_y": float(sub_y[index] + row_offset),
                "_local_pixel_x": float(sub_x[index]),
                "_local_pixel_y": float(sub_y[index]),
                "intensity": float(intensity[index]),
                "snr": float(local_snr[index]),
                "normal_qx": float(field.normal_x[row, col]),
                "normal_qy": float(field.normal_y[row, col]),
                "tangent_qx": float(field.tangent_x[row, col]),
                "tangent_qy": float(field.tangent_y[row, col]),
                "curvature": float(field.lambda_min[row, col]),
                "curvature_max": float(field.lambda_max[row, col]),
                "principal_curvature_min": float(field.lambda_min[row, col]),
                "principal_curvature_max": float(field.lambda_max[row, col]),
                "curvature_method": "scaled_graph_shape_operator",
                "axis_separation": float(field.axis_separation[row, col]),
                "q_normal_step": float(field.q_normal_step[row, col]),
                "q_scale_anisotropy": float(field.q_scale_anisotropy[row, col]),
                "scale": float(field.sigma),
                "scale_stability": float("nan"),
                "scale_stable": False,
                "topology_flag": "candidate",
                "topology": "candidate",
                "topology_flags": ["candidate"],
                "accepted": accepted,
                "valid": True,
                "reason": reason,
                "arc_id": -1,
                "branch_id": -1,
                "quadrant": "unknown",
                "side": "unknown",
                "score": float(field.score[row, col]),
                "_scale_index": int(round(field.sigma * 1000.0)),
            }
        )
    return results


def _point_signature(image: np.ndarray, options: Mapping[str, Any], q_window: Any = None) -> str:
    digest = hashlib.sha1(np.ascontiguousarray(image).view(np.uint8)).hexdigest()[:16]
    stable_options = {
        key: value
        for key, value in options.items()
        if key not in {"edits", "seed", "resamples", "sensitivity", "stage"}
    }
    try:
        option_text = json.dumps(
            {"options": stable_options, "q_window": q_window},
            sort_keys=True,
            default=str,
            separators=(",", ":"),
        )
    except TypeError:
        option_text = repr((sorted((str(key), repr(value)) for key, value in stable_options.items()), repr(q_window)))
    option_digest = hashlib.sha1(option_text.encode("utf-8")).hexdigest()[:12]
    return f"{digest}-{option_digest}"


def _assign_point_ids(candidates: list[dict[str, Any]], signature: str) -> None:
    for point in candidates:
        text = f"{signature}|{point['qx']:.8g}|{point['qy']:.8g}|{point['pixel_x']:.5f}|{point['pixel_y']:.5f}"
        point["point_id"] = "ridge-" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _nms(candidates: list[dict[str, Any]], options: Mapping[str, Any], q_step: float) -> list[dict[str, Any]]:
    if not candidates:
        return []
    ordered = sorted(
        candidates,
        key=lambda point: (
            not bool(point["accepted"]),
            -float(point.get("score", 0.0)),
            float(point["qx"]),
            float(point["qy"]),
            float(point["scale"]),
        ),
    )
    base = max(0.25 * q_step, float(options["nms_radius_factor"]) * q_step)
    bins: dict[tuple[int, int], list[int]] = defaultdict(list)
    kept: list[dict[str, Any]] = []
    for point in ordered:
        px, py = float(point["qx"]), float(point["qy"])
        radius = max(base, float(options["nms_radius_factor"]) * float(point["scale"]) * q_step)
        bx, by = int(math.floor(px / base)), int(math.floor(py / base))
        blocked = False
        for ix in range(bx - 2, bx + 3):
            for iy in range(by - 2, by + 3):
                for kept_index in bins.get((ix, iy), []):
                    other = kept[kept_index]
                    distance = math.hypot(px - float(other["qx"]), py - float(other["qy"]))
                    if distance >= max(radius, float(options["nms_radius_factor"]) * float(other["scale"]) * q_step):
                        continue
                    tangent_dot = abs(float(point["tangent_qx"]) * float(other["tangent_qx"]) + float(point["tangent_qy"]) * float(other["tangent_qy"]))
                    if tangent_dot >= 0.35:
                        blocked = True
                        break
                if blocked:
                    break
            if blocked:
                break
        if not blocked:
            index = len(kept)
            kept.append(point)
            bins[(bx, by)].append(index)
    for point in kept:
        point["_nms_kept"] = True
    return kept


def _annotate_scale_stability(
    candidates: list[dict[str, Any]],
    raw_candidates: list[dict[str, Any]],
    q_step: float,
    scales: Sequence[float],
    *,
    cancel_event: Any = None,
) -> None:
    """Annotate NMS points from an exact physical-q neighborhood query.

    The previous implementation compared every surviving point with every raw
    candidate.  A ``cKDTree`` gives the same closed-ball support set without
    silently dropping or downsampling raw candidates.  The small binned
    fallback remains exact when scipy.spatial is unavailable.
    """

    if not candidates:
        return
    radius = 1.8 * max(float(q_step), np.finfo(float).eps)
    raw_xy = np.asarray([[float(point["qx"]), float(point["qy"])] for point in raw_candidates], dtype=float)
    finite_raw = np.flatnonzero(np.all(np.isfinite(raw_xy), axis=1)) if raw_xy.size else np.asarray([], dtype=int)
    tree = cKDTree(raw_xy[finite_raw]) if finite_raw.size and cKDTree is not None else None
    bins: dict[tuple[int, int], list[int]] = {}
    if tree is None and raw_xy.size:
        cell = radius
        for raw_index in finite_raw.tolist():
            point = raw_candidates[raw_index]
            if raw_index % 2048 == 0:
                _check_cancelled(cancel_event, "ridge-trace:scale-stability-index")
            key = (int(math.floor(float(point["qx"]) / cell)), int(math.floor(float(point["qy"]) / cell)))
            bins.setdefault(key, []).append(raw_index)
    n_scales = max(len(scales), 1)
    for index, point in enumerate(candidates):
        if index % 128 == 0:
            _check_cancelled(cancel_event, "ridge-trace:scale-stability-query")
        xy = [float(point["qx"]), float(point["qy"])]
        if tree is not None:
            support_indices = finite_raw[np.asarray(tree.query_ball_point(xy, radius), dtype=int)].tolist()
        elif raw_xy.size:
            cell = radius
            bx = int(math.floor(xy[0] / cell))
            by = int(math.floor(xy[1] / cell))
            support_indices = [
                raw_index
                for ix in range(bx - 1, bx + 2)
                for iy in range(by - 1, by + 2)
                for raw_index in bins.get((ix, iy), ())
                if math.hypot(raw_xy[raw_index, 0] - xy[0], raw_xy[raw_index, 1] - xy[1]) <= radius
            ]
        else:
            support_indices = []
        _check_cancelled(cancel_event, "ridge-trace:scale-stability-query")
        unique_scales = len({round(float(raw_candidates[raw_index]["scale"]), 5) for raw_index in support_indices})
        point["scale_stability"] = float(unique_scales / n_scales)
        point["scale_stable"] = bool(point["scale_stability"] >= 0.34)
        if point["scale_stability"] < 0.34:
            point["topology_flags"] = list(dict.fromkeys(point.get("topology_flags", []) + ["single_scale_support"]))


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left, right = self.find(left), self.find(right)
        if left == right:
            return
        if self.rank[left] < self.rank[right]:
            left, right = right, left
        self.parent[right] = left
        if self.rank[left] == self.rank[right]:
            self.rank[left] += 1


def _graph_arcs(candidates: list[dict[str, Any]], options: Mapping[str, Any], q_step: float) -> tuple[list[list[int]], list[tuple[int, int]]]:
    if not candidates:
        return [], []
    radius = float(options["graph_radius_factor"]) * max(q_step, np.finfo(float).eps)
    base = max(radius / 2.0, np.finfo(float).eps)
    bins: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, point in enumerate(candidates):
        bins[(int(math.floor(float(point["qx"]) / base)), int(math.floor(float(point["qy"]) / base)))].append(index)
    union = _UnionFind(len(candidates))
    edges: list[tuple[int, int]] = []
    tangent_cos = math.cos(math.radians(float(options["graph_tangent_angle_deg"])))
    for index, point in enumerate(candidates):
        px, py = float(point["qx"]), float(point["qy"])
        bx, by = int(math.floor(px / base)), int(math.floor(py / base))
        scale = max(float(point["scale"]) * q_step, q_step)
        for ix in range(bx - 2, bx + 3):
            for iy in range(by - 2, by + 3):
                for other_index in bins.get((ix, iy), []):
                    if other_index <= index:
                        continue
                    other = candidates[other_index]
                    # Topology is built from candidates that pass the local
                    # curvature/axis gates.  Rejected candidates remain in
                    # ``points`` for UI review but must not act as bridges
                    # that stitch unrelated arcs through noise or a mask.
                    if not bool(point.get("accepted", False)) or not bool(other.get("accepted", False)):
                        continue
                    # Reference-axis branch identity is assigned before graph
                    # connectivity in the production trace.  Do not let an
                    # unresolved axis-boundary point bridge two branches.
                    branch = int(point.get("branch_id", -1))
                    other_branch = int(other.get("branch_id", -1))
                    if branch not in (0, 1) or other_branch not in (0, 1) or branch != other_branch:
                        continue
                    dx, dy = float(other["qx"]) - px, float(other["qy"]) - py
                    distance = math.hypot(dx, dy)
                    max_distance = max(radius, float(other["scale"]) * q_step * float(options["graph_radius_factor"]))
                    if distance <= np.finfo(float).eps or distance > max_distance:
                        continue
                    dot_tangent = abs(float(point["tangent_qx"]) * float(other["tangent_qx"]) + float(point["tangent_qy"]) * float(other["tangent_qy"]))
                    if not np.isfinite(dot_tangent) or dot_tangent < tangent_cos:
                        continue
                    normal_projection = min(
                        abs(dx * float(point["normal_qx"]) + dy * float(point["normal_qy"])),
                        abs(dx * float(other["normal_qx"]) + dy * float(other["normal_qy"])),
                    )
                    if normal_projection > float(options["graph_normal_factor"]) * max(scale, float(other["scale"]) * q_step):
                        continue
                    union.union(index, other_index)
                    edges.append((index, other_index))
    components: dict[int, list[int]] = defaultdict(list)
    for index in range(len(candidates)):
        components[union.find(index)].append(index)
    groups = sorted(components.values(), key=lambda values: min(values))
    return groups, edges


def _ordered_component(component: list[int], edges: list[tuple[int, int]], candidates: list[dict[str, Any]]) -> list[int]:
    if len(component) <= 2:
        return sorted(component)
    members = set(component)
    adjacency: dict[int, list[int]] = defaultdict(list)
    for left, right in edges:
        if left in members and right in members:
            adjacency[left].append(right)
            adjacency[right].append(left)
    endpoints = [index for index in component if len(adjacency[index]) == 1]
    if endpoints:
        start = min(endpoints, key=lambda index: (float(candidates[index]["qx"]), float(candidates[index]["qy"])))
    else:
        start = min(component, key=lambda index: (float(candidates[index]["qx"]), float(candidates[index]["qy"])))
    ordered = [start]
    previous = -1
    current = start
    while True:
        neighbours = [value for value in adjacency[current] if value != previous and value not in ordered]
        if not neighbours:
            break
        if previous >= 0:
            vx = float(candidates[current]["qx"]) - float(candidates[previous]["qx"])
            vy = float(candidates[current]["qy"]) - float(candidates[previous]["qy"])
            norm = math.hypot(vx, vy) or 1.0
            vx, vy = vx / norm, vy / norm
            current = max(neighbours, key=lambda index: abs(vx * (float(candidates[index]["qx"]) - float(candidates[current]["qx"])) + vy * (float(candidates[index]["qy"]) - float(candidates[current]["qy"]))))
        else:
            current = neighbours[0]
        ordered.append(current)
        previous, current = ordered[-2], ordered[-1]
        if len(ordered) == len(component):
            break
    if len(ordered) != len(component):
        remaining = [index for index in component if index not in ordered]
        ordered.extend(sorted(remaining))
    return ordered


def _quadrant_branch(qx: float, qy: float, *, center_x: float, center_y: float, reference_axis_deg: float, tolerance_deg: float) -> tuple[int, str]:
    theta = np.deg2rad(float(reference_axis_deg))
    dx, dy = qx - center_x, qy - center_y
    along = dx * np.cos(theta) + dy * np.sin(theta)
    perpendicular = -dx * np.sin(theta) + dy * np.cos(theta)
    angle = float(np.degrees(np.arctan2(perpendicular, along)))
    if abs(abs(angle) - 90.0) <= tolerance_deg or abs(abs(angle) - 180.0) <= tolerance_deg or abs(angle) <= tolerance_deg:
        quadrant = "axis_boundary"
        return -1, quadrant
    if along >= 0 and perpendicular >= 0:
        return 0, "QI"
    if along < 0 and perpendicular >= 0:
        return 1, "QII"
    if along < 0 and perpendicular < 0:
        return 0, "QIII"
    return 1, "QIV"


def _assign_reference_branches(candidates: list[dict[str, Any]], options: Mapping[str, Any]) -> None:
    """Assign reference-axis identities before graph connectivity is formed."""

    center_x, center_y = float(options["center_qx"]), float(options["center_qy"])
    reference = float(options.get("reference_axis_deg", 0.0))
    tolerance = float(options["quadrant_axis_tol_deg"])
    for point in candidates:
        branch, quadrant = _quadrant_branch(
            float(point["qx"]),
            float(point["qy"]),
            center_x=center_x,
            center_y=center_y,
            reference_axis_deg=reference,
            tolerance_deg=tolerance,
        )
        point["branch_id"], point["quadrant"] = int(branch), quadrant
        if branch not in (0, 1):
            point["topology_flags"] = list(dict.fromkeys(point.get("topology_flags", []) + ["unresolved_reference_branch"]))


def _side_from_reference_axis(
    qx: float,
    qy: float,
    *,
    center_x: float,
    center_y: float,
    reference_axis_deg: float,
    q_step: float,
) -> str:
    """Upper/lower from the declared draw axis when the local major is unusable."""

    theta = math.radians(float(reference_axis_deg))
    dx = float(qx) - float(center_x)
    dy = float(qy) - float(center_y)
    perpendicular = -dx * math.sin(theta) + dy * math.cos(theta)
    scale = _side_axis_threshold(math.hypot(dx, dy), q_step)
    if perpendicular > scale:
        return "upper"
    if perpendicular < -scale:
        return "lower"
    return "unknown"


def _side_axis_threshold(radius: float, q_step: float) -> float:
    """Perpendicular band that is still 'on' the local major axis.

    This is an angular criterion, not a detector-pixel floor.  A one-pixel
    band swallows an entire flat butterfly wing once ``b`` is smaller than
    the q-grid step, leaving every point ``side=unknown``.
    """

    del q_step
    finite_radius = abs(float(radius)) if np.isfinite(radius) else 0.0
    return max(finite_radius * math.sin(math.radians(0.20)), float(np.finfo(float).eps))


def _point_identity_key(point: Mapping[str, Any]) -> tuple[int, str] | None:
    """Return a quantitative (branch, side) key, or None when it is unresolved."""

    try:
        branch = int(point.get("branch_id", -1))
    except (TypeError, ValueError):
        return None
    side = str(point.get("side", "unknown") or "unknown")
    if branch in (0, 1) and side in {"upper", "lower"}:
        return branch, side
    return None


def _demote_unresolved_identity_point(point: dict[str, Any], reason: str, *flags: str) -> None:
    """Keep a tip/axis candidate visible without letting it poison siblings."""

    extra = list(flags) + [reason]
    point["topology_flags"] = list(dict.fromkeys(list(point.get("topology_flags", [])) + extra))
    if bool(point.get("accepted", False)):
        point["accepted"] = False
        point["reason"] = reason


def _first_order_q_hint(
    q: np.ndarray,
    intensity: np.ndarray,
    valid: np.ndarray,
    q_min: float,
    q_max: float,
    *,
    n_bins: int = 96,
) -> dict[str, Any]:
    """Locate the lowest-q significant ring inside the user window.

    A wide experimental window often contains the first-order butterfly and a
    growing harmonic.  The Wang/Grubb ellipse is the innermost ring, so the
    hint is the lowest-q local maximum that is still a substantial fraction
    of the global I(q) peak.  The caller may tighten an analysis band without
    rewriting the user's configured q_window.
    """

    summary: dict[str, Any] = {
        "q_star": None,
        "band": None,
        "n_bins": int(n_bins),
        "reason": None,
    }
    selected = (
        np.asarray(valid, dtype=bool)
        & np.isfinite(q)
        & np.isfinite(intensity)
        & (q >= q_min)
        & (q <= q_max)
    )
    if int(np.count_nonzero(selected)) < 200:
        summary["reason"] = "insufficient_pixels"
        return summary
    radii = np.asarray(q[selected], dtype=float)
    weights = np.clip(np.asarray(intensity[selected], dtype=float), 0.0, None)
    edges = np.linspace(float(q_min), float(q_max), int(n_bins) + 1)
    summed, _ = np.histogram(radii, bins=edges, weights=weights)
    counts, _ = np.histogram(radii, bins=edges)
    profile = np.divide(
        summed,
        counts,
        out=np.zeros_like(summed, dtype=float),
        where=counts > 0,
    )
    centres = 0.5 * (edges[:-1] + edges[1:])
    usable = np.isfinite(profile) & (counts >= 8)
    if int(np.count_nonzero(usable)) < 8:
        summary["reason"] = "sparse_radial_profile"
        return summary
    intensity_q = np.where(usable, profile, 0.0)
    peak = float(np.max(intensity_q))
    if not np.isfinite(peak) or peak <= 0.0:
        summary["reason"] = "empty_radial_profile"
        return summary
    interior = (
        (intensity_q[1:-1] >= intensity_q[:-2])
        & (intensity_q[1:-1] >= intensity_q[2:])
        & (intensity_q[1:-1] >= 0.25 * peak)
    )
    peak_indices = np.where(interior)[0] + 1
    if peak_indices.size == 0:
        peak_indices = np.asarray([int(np.argmax(intensity_q))], dtype=int)
    # Adjacent bins of one Bragg ring can each look like a local max.
    # Harmonics sit at ≳2× q*, so keep the strongest member of the lowest-q
    # family instead of the first noisy shoulder.
    peak_index = _select_first_order_peak_index(
        centres[peak_indices],
        intensity_q[peak_indices],
        peak_indices,
    )
    q_bin = float(centres[peak_index])
    q_star = _refine_radial_peak_q(
        radii, weights, q_bin, q_min, q_max, coarse_bins=int(n_bins)
    )
    if not np.isfinite(q_star) or q_star <= 0.0:
        summary["reason"] = "invalid_peak"
        return summary
    low = max(float(q_min), 0.62 * q_star)
    high = min(float(q_max), 1.45 * q_star)
    if high <= low:
        summary["q_star"] = q_star
        summary["q_star_bin"] = q_bin
        summary["reason"] = "degenerate_band"
        return summary
    summary.update(
        {
            "q_star": q_star,
            "q_star_bin": q_bin,
            "band": [low, high],
            "reason": "ok",
        }
    )
    return summary


def _select_first_order_peak_index(
    peak_q: np.ndarray,
    peak_intensity: np.ndarray,
    peak_indices: np.ndarray,
    *,
    family_span: float = 1.45,
) -> int:
    """Pick the strongest bin in the lowest-q peak family."""

    if peak_indices.size == 1:
        return int(peak_indices[0])
    families: list[list[int]] = [[0]]
    for offset in range(1, int(peak_q.size)):
        root = float(peak_q[families[-1][0]])
        value = float(peak_q[offset])
        if np.isfinite(root) and np.isfinite(value) and value <= family_span * root:
            families[-1].append(offset)
        else:
            families.append([offset])
    family = families[0]
    best = max(family, key=lambda item: float(peak_intensity[item]))
    return int(peak_indices[best])


def _refine_radial_peak_q(
    radii: np.ndarray,
    weights: np.ndarray,
    q_bin: float,
    q_min: float,
    q_max: float,
    *,
    n_bins: int = 25,
    coarse_bins: int = 96,
) -> float:
    """Parabolic sub-bin refinement of I(q)* around a coarse histogram peak."""

    try:
        centre = float(q_bin)
    except (TypeError, ValueError):
        return float("nan")
    if not (np.isfinite(centre) and centre > 0.0):
        return float("nan")
    coarse_step = (float(q_max) - float(q_min)) / max(int(coarse_bins), 1)
    if not np.isfinite(coarse_step) or coarse_step <= 0.0:
        coarse_step = 0.08 * centre
    # Stay inside the coarse peak's own bin.  A 0.75–1.25 window plus a
    # grid that misses q_bin makes the 3-point interpolate hit its clamp
    # and snap every frame to one of two discrete q* values.
    span = max(float(coarse_step), 0.06 * centre)
    count = max(7, int(n_bins) | 1)
    mid = count // 2
    step = 2.0 * span / count
    low = centre - (mid + 0.5) * step
    high = centre + (mid + 0.5) * step
    low = max(float(q_min), low)
    high = min(float(q_max), high)
    if high <= low:
        return centre
    edges = np.linspace(low, high, count + 1)
    summed, _ = np.histogram(radii, bins=edges, weights=weights)
    counts, _ = np.histogram(radii, bins=edges)
    profile = np.divide(
        summed,
        counts,
        out=np.zeros_like(summed, dtype=float),
        where=counts > 0,
    )
    centres = 0.5 * (edges[:-1] + edges[1:])
    usable = np.isfinite(profile) & (counts >= 4)
    if int(np.count_nonzero(usable)) < 5:
        return centre
    intensity_q = np.where(usable, profile, 0.0)
    index = int(np.argmin(np.abs(centres - centre)))
    if not usable[index]:
        return centre
    if 0 < index < intensity_q.size - 1 and (
        intensity_q[index] < intensity_q[index - 1]
        or intensity_q[index] < intensity_q[index + 1]
    ):
        # One step toward the higher neighbour keeps the interpolate on
        # this ring without a second argmax over the whole window.
        index = index - 1 if intensity_q[index - 1] > intensity_q[index + 1] else index + 1
        if not usable[index]:
            return centre
    if index <= 0 or index >= intensity_q.size - 1:
        return float(centres[index])
    y0, y1, y2 = float(intensity_q[index - 1]), float(intensity_q[index]), float(intensity_q[index + 1])
    denom = y0 - 2.0 * y1 + y2
    if abs(denom) < 1e-12 or y1 + 1e-12 < max(y0, y2):
        return float(centres[index])
    delta = 0.5 * (y0 - y2) / denom
    delta = max(-0.5, min(0.5, float(delta)))
    bin_step = float(centres[index] - centres[index - 1])
    refined = float(centres[index] + delta * bin_step)
    if not (np.isfinite(refined) and low <= refined <= high):
        return float(centres[index])
    return refined


def _prefer_first_order_scores(
    candidates: Sequence[dict[str, Any]],
    q_star: float | None,
    *,
    sigma_log: float = 0.45,
) -> int:
    """Down-weight harmonic-scale candidates without clipping ellipse tips.

    A hard |q| crop around I(q)* removes the major-axis ends of a flat
    Wang ellipse.  A log-q score weight keeps those tips while making a
    3–4× harmonic lose NMS / max_points competition.
    """

    try:
        hint = float(q_star) if q_star is not None else float("nan")
    except (TypeError, ValueError):
        return 0
    if not np.isfinite(hint) or hint <= 0.0 or not np.isfinite(sigma_log) or sigma_log <= 0.0:
        return 0
    updated = 0
    for point in candidates:
        try:
            radius = float(np.hypot(float(point["qx"]), float(point["qy"])))
            score = float(point.get("score", 0.0))
        except (KeyError, TypeError, ValueError):
            continue
        if not (np.isfinite(radius) and radius > 0.0 and np.isfinite(score)):
            continue
        weight = math.exp(-0.5 * (math.log(radius / hint) / float(sigma_log)) ** 2)
        point["score"] = score * weight
        point["first_order_weight"] = float(weight)
        updated += 1
    return updated


def _axis_deviation_deg(
    qx: float,
    qy: float,
    *,
    center_x: float,
    center_y: float,
    reference_axis_deg: float,
) -> float:
    """Absolute angle from the declared draw-axis line, in degrees."""

    theta = math.radians(float(reference_axis_deg))
    dx = float(qx) - float(center_x)
    dy = float(qy) - float(center_y)
    along = dx * math.cos(theta) + dy * math.sin(theta)
    perpendicular = -dx * math.sin(theta) + dy * math.cos(theta)
    angle = math.degrees(math.atan2(perpendicular, along))
    return min(abs(angle), abs(abs(angle) - 180.0))


def _angular_delta_deg(first: float, second: float) -> float:
    return abs(((float(first) - float(second) + 180.0) % 360.0) - 180.0)


def _sparse_first_order_coverage(
    points: Sequence[Mapping[str, Any]],
    *,
    center_x: float,
    center_y: float,
    reference_axis_deg: float,
) -> dict[str, Any]:
    """True when accepted identity points are too axial to define an ellipse."""

    identity = [
        point
        for point in points
        if bool(point.get("accepted", False)) and _point_identity_key(point) is not None
    ]
    if not identity:
        return {"sparse": True, "n": 0, "max_dev_deg": None}
    deviations = [
        _axis_deviation_deg(
            float(point["qx"]),
            float(point["qy"]),
            center_x=center_x,
            center_y=center_y,
            reference_axis_deg=reference_axis_deg,
        )
        for point in identity
    ]
    max_dev = max(deviations)
    sparse = max_dev < 18.0 or (len(identity) < 8 and max_dev < 25.0)
    return {"sparse": sparse, "n": len(identity), "max_dev_deg": float(max_dev)}


def _sector_first_order_peak(
    qx: np.ndarray,
    qy: np.ndarray,
    q: np.ndarray,
    intensity: np.ndarray,
    valid: np.ndarray,
    *,
    sector_deg: float,
    halfwidth_deg: float,
    hint: float,
) -> dict[str, Any] | None:
    """Return one observed radial peak in an azimuth sector, or None."""

    if not (np.isfinite(hint) and hint > 0.0):
        return None
    ang = np.degrees(np.arctan2(qy, qx))
    delta = np.abs(((ang - float(sector_deg) + 180.0) % 360.0) - 180.0)
    q_lo, q_hi = 0.70 * hint, 1.45 * hint
    selected = (
        np.asarray(valid, dtype=bool)
        & np.isfinite(q)
        & np.isfinite(intensity)
        & (delta <= float(halfwidth_deg))
        & (q >= q_lo)
        & (q <= q_hi)
    )
    if int(np.count_nonzero(selected)) < 12:
        return None
    radii = np.asarray(q[selected], dtype=float)
    values = np.asarray(intensity[selected], dtype=float)
    edges = np.linspace(q_lo, q_hi, 9)
    profile = np.full(edges.size - 1, np.nan, dtype=float)
    counts = np.zeros(edges.size - 1, dtype=int)
    idx = np.digitize(radii, edges) - 1
    for bin_i in range(edges.size - 1):
        in_bin = idx == bin_i
        counts[bin_i] = int(np.count_nonzero(in_bin))
        if counts[bin_i] >= 2:
            profile[bin_i] = float(np.nanmedian(values[in_bin]))
    usable = np.isfinite(profile)
    if int(np.count_nonzero(usable)) < 3:
        return None
    peak_i = int(np.nanargmax(np.where(usable, profile, -np.inf)))
    peak = float(profile[peak_i])
    baseline = float(np.nanmedian(profile[usable]))
    if not (np.isfinite(peak) and np.isfinite(baseline) and baseline > 0 and peak >= 1.30 * baseline):
        return None
    q_star = float(0.5 * (edges[peak_i] + edges[peak_i + 1]))
    in_bin = selected & (q >= edges[peak_i]) & (q < edges[peak_i + 1])
    if int(np.count_nonzero(in_bin)) < 3:
        in_bin = selected
    sample_qx = float(np.nanmedian(qx[in_bin]))
    sample_qy = float(np.nanmedian(qy[in_bin]))
    if not (np.isfinite(sample_qx) and np.isfinite(sample_qy)):
        return None
    rows, cols = np.nonzero(in_bin)
    nearest = int(np.argmin((qx[in_bin] - sample_qx) ** 2 + (qy[in_bin] - sample_qy) ** 2))
    return {
        "qx": sample_qx,
        "qy": sample_qy,
        "q_star": q_star,
        "intensity": peak,
        "contrast": peak / baseline,
        "pixel_x": float(cols[nearest]),
        "pixel_y": float(rows[nearest]),
        "sector_deg": float(sector_deg),
    }


def _next_arc_id(points: Sequence[Mapping[str, Any]], arcs: Sequence[Mapping[str, Any]] | None = None) -> int:
    used = {-1}
    for record in list(points) + list(arcs or ()):
        try:
            used.add(int(record.get("arc_id", -1)))
        except (TypeError, ValueError):
            continue
    return max(used) + 1


def _fill_sparse_first_order_ring(
    points: list[dict[str, Any]],
    *,
    qx: np.ndarray,
    qy: np.ndarray,
    q: np.ndarray,
    intensity: np.ndarray,
    valid: np.ndarray,
    hint: float | None,
    options: Mapping[str, Any],
    q_step: float,
    signature: str,
    arcs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Add observed first-order ring samples when curvature kept only the streak.

    This does not invent quadrants.  A sector is filled only when I(q) still
    has a peak near the first-order hint and no accepted identity point is
    already there.
    """

    summary: dict[str, Any] = {"applied": False, "added": 0, "reason": None, "sectors": []}
    if not bool(options.get("fill_sparse_first_order_ring", True)):
        summary["reason"] = "disabled"
        return summary
    try:
        hint_q = float(hint) if hint is not None else float("nan")
    except (TypeError, ValueError):
        hint_q = float("nan")
    if not (np.isfinite(hint_q) and hint_q > 0.0):
        summary["reason"] = "no_first_order_hint"
        return summary
    center_x = float(options.get("center_qx", 0.0))
    center_y = float(options.get("center_qy", 0.0))
    reference = float(options.get("reference_axis_deg", 0.0))
    coverage = _sparse_first_order_coverage(
        points, center_x=center_x, center_y=center_y, reference_axis_deg=reference
    )
    summary.update(coverage)
    if not coverage["sparse"]:
        summary["reason"] = "identity_coverage_sufficient"
        return summary
    existing_angles = []
    for point in points:
        if not bool(point.get("accepted", False)):
            continue
        existing_angles.append(math.degrees(math.atan2(float(point["qy"]), float(point["qx"]))))
    added: list[dict[str, Any]] = []
    for quadrant in (0.0, 90.0, 180.0, 270.0):
        for offset in (30.0, 45.0, 60.0, 75.0):
            sector = reference + quadrant + offset
            if any(_angular_delta_deg(sector, angle) < 12.0 for angle in existing_angles):
                continue
            peak = _sector_first_order_peak(
                qx,
                qy,
                q,
                intensity,
                valid,
                sector_deg=sector,
                halfwidth_deg=7.5,
                hint=hint_q,
            )
            if peak is None:
                continue
            point = {
                "qx": peak["qx"],
                "qy": peak["qy"],
                "pixel_x": peak["pixel_x"],
                "pixel_y": peak["pixel_y"],
                "intensity": peak["intensity"],
                "snr": peak["contrast"],
                "normal_qx": peak["qx"] / max(hint_q, 1e-12),
                "normal_qy": peak["qy"] / max(hint_q, 1e-12),
                "tangent_qx": -peak["qy"] / max(hint_q, 1e-12),
                "tangent_qy": peak["qx"] / max(hint_q, 1e-12),
                "curvature": float("nan"),
                "scale": 1.0,
                "scale_stability": float("nan"),
                "scale_stable": False,
                "topology_flag": "first_order_ring_sample",
                "topology": "first_order_ring_sample",
                "topology_flags": ["first_order_ring_sample", "side_from_reference_axis"],
                "accepted": True,
                "valid": True,
                "reason": "first_order_ring_sample",
                "arc_id": -1,
                "score": float(peak["contrast"]),
                "q_normal_step": float(q_step),
                "localization_sigma_q": float(q_step),
                "sampling_sigma_q": float(q_step),
                "uncertainty_source": "azimuthal_radial_peak",
            }
            _assign_reference_branches([point], options)
            point["side"] = _side_from_reference_axis(
                float(point["qx"]),
                float(point["qy"]),
                center_x=center_x,
                center_y=center_y,
                reference_axis_deg=reference,
                q_step=q_step,
            )
            if _point_identity_key(point) is None:
                continue
            added.append(point)
            existing_angles.append(math.degrees(math.atan2(float(point["qy"]), float(point["qx"]))))
            summary["sectors"].append(
                {
                    "sector_deg": peak["sector_deg"],
                    "q_star": peak["q_star"],
                    "contrast": peak["contrast"],
                }
            )
    if not added:
        summary["reason"] = "no_observed_sector_peak"
        return summary
    _assign_point_ids(added, signature)
    next_arc = _next_arc_id(points, arcs)
    for point in added:
        point["arc_id"] = int(next_arc)
        if arcs is not None:
            arcs.append(
                {
                    "arc_id": int(next_arc),
                    "valid": True,
                    "reason": "first_order_ring_sample",
                    "topology_flags": ["first_order_ring_sample"],
                    "point_ids": [str(point["point_id"])],
                    "identity_resolved": _point_identity_key(point) is not None,
                }
            )
        next_arc += 1
    points.extend(added)
    summary["applied"] = True
    summary["added"] = len(added)
    summary["reason"] = "filled_observed_ring_sectors"
    return summary


# First-order tips of a flat origin-centred ellipse can reach ~3× q*.
# A 3–4× harmonic sits beyond that and must not remain in the same family.
FIRST_ORDER_FAMILY_SPAN = 3.5


def _demote_secondary_radial_population(
    points: Sequence[dict[str, Any]],
    prefer_radius: float | None = None,
) -> dict[str, Any]:
    """Keep one radial family when a late frame still has an earlier ring.

    A single origin-centred ellipse cannot carry the first-order butterfly and
    a harmonic at once.  A clear |q| gap demotes the family farther from the
    first-order hint; without a hint the inner ring is kept.  Unimodal wings
    are unchanged.
    """

    accepted = [
        point
        for point in points
        if bool(point.get("accepted", False)) and _point_identity_key(point) is not None
    ]
    radii = []
    for point in accepted:
        try:
            radius = float(np.hypot(float(point["qx"]), float(point["qy"])))
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite(radius) and radius > 0.0:
            radii.append(radius)
    summary = {"split": False, "kept": 0, "demoted": 0, "threshold": None, "keep": None}
    if len(radii) >= 12:
        ordered = np.sort(np.asarray(radii, dtype=float))
        gaps = np.diff(ordered)
        if gaps.size > 0:
            split_at = int(np.argmax(gaps))
            low = ordered[: split_at + 1]
            high = ordered[split_at + 1 :]
            gap = float(gaps[split_at])
            median = float(np.median(ordered))
            can_split = (
                low.size >= 4
                and high.size >= 4
                and median > 0.0
                and gap >= 0.35 * median
                and float(high[0] / max(float(low[-1]), np.finfo(float).eps)) >= 1.8
            )
            if can_split:
                if prefer_radius is not None:
                    try:
                        hint_radius = float(prefer_radius)
                    except (TypeError, ValueError):
                        hint_radius = float("nan")
                    if np.isfinite(hint_radius) and hint_radius > 0.0:
                        keep_high = abs(float(np.median(high)) - hint_radius) < abs(
                            float(np.median(low)) - hint_radius
                        )
                    else:
                        keep_high = False
                else:
                    keep_high = False
                threshold = 0.5 * (float(low[-1]) + float(high[0]))
                demoted = 0
                for point in accepted:
                    try:
                        radius = float(np.hypot(float(point["qx"]), float(point["qy"])))
                    except (KeyError, TypeError, ValueError):
                        continue
                    drop = radius < threshold if keep_high else radius >= threshold
                    if drop:
                        _demote_unresolved_identity_point(
                            point, "secondary_radial_population", "mixed_radial_populations"
                        )
                        demoted += 1
                summary.update(
                    {
                        "split": True,
                        "kept": int(len(accepted) - demoted),
                        "demoted": int(demoted),
                        "threshold": threshold,
                        "keep": "high_q" if keep_high else "low_q",
                    }
                )
    extra = _demote_beyond_first_order_family(accepted, prefer_radius)
    if extra:
        summary["demoted"] = int(summary.get("demoted") or 0) + extra
        summary["hint_ceiling"] = _first_order_family_ceiling(prefer_radius)
        if not summary["split"]:
            summary["keep"] = "first_order_hint"
    summary["kept"] = int(sum(1 for point in accepted if point.get("accepted")))
    return summary


def _first_order_family_ceiling(prefer_radius: Any) -> float | None:
    try:
        hint = float(prefer_radius) if prefer_radius is not None else float("nan")
    except (TypeError, ValueError):
        return None
    if not (np.isfinite(hint) and hint > 0.0):
        return None
    return float(FIRST_ORDER_FAMILY_SPAN) * hint


def _demote_beyond_first_order_family(
    accepted: Sequence[dict[str, Any]],
    prefer_radius: Any,
) -> int:
    ceiling = _first_order_family_ceiling(prefer_radius)
    if ceiling is None:
        return 0
    demoted = 0
    for point in accepted:
        if not bool(point.get("accepted", False)):
            continue
        try:
            radius = float(np.hypot(float(point["qx"]), float(point["qy"])))
        except (KeyError, TypeError, ValueError):
            continue
        if radius > ceiling:
            _demote_unresolved_identity_point(
                point, "secondary_radial_population", "beyond_first_order_family"
            )
            demoted += 1
    return demoted


def _component_major_direction(
    ordered: Sequence[int],
    candidates: Sequence[Mapping[str, Any]],
    *,
    center_x: float,
    center_y: float,
    reference_vector: np.ndarray,
) -> tuple[np.ndarray, float, bool]:
    positions = np.asarray(
        [[candidates[index]["qx"], candidates[index]["qy"]] for index in ordered],
        dtype=float,
    )
    centered = positions - np.asarray([center_x, center_y])
    if len(positions) >= 3 and np.linalg.matrix_rank(np.cov(centered.T)) > 0:
        covariance = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        major = np.asarray(eigenvectors[:, int(np.argmax(eigenvalues))], dtype=float)
    else:
        tangents = np.asarray(
            [[candidates[index]["tangent_qx"], candidates[index]["tangent_qy"]] for index in ordered],
            dtype=float,
        )
        major = np.nanmedian(tangents, axis=0)
    major_norm = float(np.hypot(*major))
    direction_confidence = 0.0
    if major_norm > 1e-10 and np.all(np.isfinite(major)):
        major = major / major_norm
        alignment = abs(float(np.dot(major, reference_vector)))
        direction_confidence = float(min(1.0, alignment))
        if float(np.dot(major, reference_vector)) < 0:
            major = -major
    else:
        major = np.asarray([np.nan, np.nan], dtype=float)
    resolved = bool(np.all(np.isfinite(major)) and direction_confidence >= 0.20)
    return major, direction_confidence, resolved


def _assign_component_identities(
    ordered: Sequence[int],
    candidates: list[dict[str, Any]],
    options: Mapping[str, Any],
    q_step: float,
    *,
    center_x: float,
    center_y: float,
    major: np.ndarray,
    resolved: bool,
) -> None:
    positions = np.asarray(
        [[candidates[index]["qx"], candidates[index]["qy"]] for index in ordered],
        dtype=float,
    )
    centered = positions - np.asarray([center_x, center_y])
    side_normal = np.asarray([-major[1], major[0]]) if resolved else np.asarray([np.nan, np.nan])
    median_radius = float(np.nanmedian(np.linalg.norm(centered, axis=1)))
    side_scale = _side_axis_threshold(median_radius, q_step)
    for order_index, index in enumerate(ordered):
        point = candidates[index]
        point["arc_order"] = int(order_index)
        branch, quadrant = _quadrant_branch(
            float(point["qx"]),
            float(point["qy"]),
            center_x=center_x,
            center_y=center_y,
            reference_axis_deg=float(options.get("reference_axis_deg", 0.0)),
            tolerance_deg=float(options["quadrant_axis_tol_deg"]),
        )
        point["branch_id"], point["quadrant"] = int(branch), quadrant
        short_arc = len(ordered) < int(options["min_arc_points"])
        use_reference_side = (not resolved) or short_arc
        if use_reference_side:
            point["side"] = _side_from_reference_axis(
                float(point["qx"]),
                float(point["qy"]),
                center_x=center_x,
                center_y=center_y,
                reference_axis_deg=float(options.get("reference_axis_deg", 0.0)),
                q_step=q_step,
            )
        else:
            radial = np.asarray([float(point["qx"]) - center_x, float(point["qy"]) - center_y])
            side_value = float(np.dot(radial, side_normal))
            point["side"] = (
                "upper" if side_value > side_scale else "lower" if side_value < -side_scale else "unknown"
            )
        flags = list(point.get("topology_flags", []))
        flags.append("connected_arc")
        point["topology_flag"] = "connected_observed_arc"
        point["topology"] = "connected_observed_arc"
        if short_arc:
            flags.append("short_arc")
        if not resolved:
            flags.append("major_direction_ambiguous")
        if use_reference_side and point["side"] in {"upper", "lower"}:
            flags.append("side_from_reference_axis")
        elif not resolved and bool(point.get("accepted", False)):
            point["reason"] = "ambiguous_major_direction"
        if _point_identity_key(point) is None:
            reason = "side_axis_boundary" if int(point["branch_id"]) in (0, 1) else "unresolved_reference_branch"
            flags.append(reason)
            if bool(point.get("accepted", False)):
                point["accepted"] = False
                point["reason"] = reason
        point["topology_flags"] = list(dict.fromkeys(flags))


def _component_tangent_errors(
    ordered: Sequence[int],
    candidates: list[dict[str, Any]],
) -> list[float]:
    tangent_errors: list[float] = []
    for order_index, index in enumerate(ordered):
        if len(ordered) == 1:
            tangent_errors.append(float("nan"))
            continue
        neighbour = ordered[min(order_index + 1, len(ordered) - 1)] if order_index == 0 else ordered[order_index - 1]
        dx = float(candidates[neighbour]["qx"]) - float(candidates[index]["qx"])
        dy = float(candidates[neighbour]["qy"]) - float(candidates[index]["qy"])
        norm = math.hypot(dx, dy)
        if norm <= np.finfo(float).eps:
            tangent_errors.append(float("nan"))
            continue
        observed = np.asarray([dx / norm, dy / norm])
        tangent = np.asarray(
            [float(candidates[index]["tangent_qx"]), float(candidates[index]["tangent_qy"])]
        )
        dot = float(np.clip(abs(np.dot(observed, tangent)), -1.0, 1.0))
        error = float(np.degrees(np.arccos(dot)))
        tangent_errors.append(error)
        candidates[index]["tangent_error_deg"] = error
    return tangent_errors


def _split_component_identities(ordered: Sequence[int], candidates: Sequence[Mapping[str, Any]]) -> dict[tuple[int, str], list[int]]:
    """Split one connected wing into identity-pure (branch, side) groups.

    A continuous ellipse wing often crosses the local major axis at the tip.
    Those two sides belong to the same observed component but must not share
    one arc identity, or the later refresh would discard the whole wing.
    """

    groups: dict[tuple[int, str], list[int]] = {}
    for index in ordered:
        key = _point_identity_key(candidates[index])
        if key is None:
            continue
        groups.setdefault(key, []).append(index)
    return groups


def _build_identity_arc(
    *,
    arc_id: int,
    ordered: Sequence[int],
    candidates: list[dict[str, Any]],
    options: Mapping[str, Any],
    major: np.ndarray,
    direction_confidence: float,
    resolved: bool,
    parent_split: bool,
) -> dict[str, Any]:
    for order_index, index in enumerate(ordered):
        candidates[index]["arc_id"] = int(arc_id)
        candidates[index]["arc_order"] = int(order_index)
        if parent_split:
            candidates[index]["topology_flags"] = list(
                dict.fromkeys(candidates[index].get("topology_flags", []) + ["identity_split_from_connected_wing"])
            )
    tangent_errors = _component_tangent_errors(ordered, candidates)
    accepted_count = int(sum(bool(candidates[index]["accepted"]) for index in ordered))
    scale_values = np.asarray([float(candidates[index]["scale"]) for index in ordered], dtype=float)
    scale_stability = (
        float(1.0 / (1.0 + np.nanstd(scale_values) / max(np.nanmean(scale_values), np.finfo(float).eps)))
        if scale_values.size
        else float("nan")
    )
    for index in ordered:
        candidates[index]["scale_stability"] = scale_stability
        candidates[index]["scale_stable"] = bool(scale_stability >= 0.5)
        if scale_stability < 0.5:
            candidates[index]["topology_flags"] = list(
                dict.fromkeys(candidates[index]["topology_flags"] + ["scale_unstable"])
            )
    tangent_error_median = (
        float(np.nanmedian(tangent_errors)) if np.any(np.isfinite(tangent_errors)) else float("nan")
    )
    branch_ids = sorted({int(candidates[index]["branch_id"]) for index in ordered})
    sides = sorted({str(candidates[index]["side"]) for index in ordered if _point_identity_key(candidates[index])})
    identity_resolved = bool(len(branch_ids) == 1 and branch_ids[0] in (0, 1) and len(sides) == 1 and sides[0] in {"upper", "lower"})
    identity_flags: list[str] = []
    if not (len(branch_ids) == 1 and branch_ids[0] in (0, 1)):
        identity_flags.append("unresolved_or_mixed_branch")
    if not (len(sides) == 1 and sides[0] in {"upper", "lower"}):
        identity_flags.append("unresolved_or_mixed_side")
    if parent_split:
        identity_flags.append("identity_split_from_connected_wing")
    arc_valid = bool(
        accepted_count >= int(options["min_arc_points"])
        and np.isfinite(tangent_error_median)
        and tangent_error_median <= float(options.get("max_arc_tangent_error_deg", 45.0))
        and scale_stability >= 0.4
        and identity_resolved
    )
    return {
        "arc_id": int(arc_id),
        "point_ids": [str(candidates[index]["point_id"]) for index in ordered],
        "ordered_point_ids": [str(candidates[index]["point_id"]) for index in ordered],
        "n_points": int(len(ordered)),
        "accepted_points": accepted_count,
        "valid": arc_valid,
        "reason": "accepted" if arc_valid else "short_or_rejected_arc",
        "major_direction_qx": float(major[0]),
        "major_direction_qy": float(major[1]),
        "major_direction_resolved": resolved,
        "major_direction_confidence": direction_confidence,
        "side_reference": "observed_arc_major_direction_and_center",
        "reference_axis_deg": float(options.get("reference_axis_deg", 0.0)),
        "branch_ids": branch_ids,
        "quadrants": sorted({str(candidates[index]["quadrant"]) for index in ordered}),
        "sides": sides,
        "accepted_branch_ids": branch_ids,
        "accepted_sides": sides,
        "identity_resolved": identity_resolved,
        "identity_flags": identity_flags,
        "tangent_error_deg": tangent_errors,
        "tangent_error_median_deg": tangent_error_median,
        "scale_stability": scale_stability,
        "topology_flag": "connected_observed_arc",
        "topology_flags": ["topology_before_ellipse_fit", *identity_flags],
        "parent_component_split": bool(parent_split),
    }


def _attach_unresolved_to_nearest_arc(
    unresolved: Sequence[int],
    arcs: Sequence[Mapping[str, Any]],
    candidates: list[dict[str, Any]],
) -> None:
    """Keep tip/axis points on a same-branch sibling arc for review only."""

    if not unresolved or not arcs:
        return
    by_id = {str(item.get("point_id")): item for item in candidates}
    for index in unresolved:
        point = candidates[index]
        if int(point.get("arc_id", -1)) >= 0:
            continue
        branch = int(point.get("branch_id", -1))
        qx, qy = float(point["qx"]), float(point["qy"])
        best_id = None
        best_distance = float("inf")
        for arc in arcs:
            if branch in (0, 1) and list(arc.get("branch_ids", ())) != [branch]:
                continue
            for point_id in arc.get("ordered_point_ids", ()):
                sibling = by_id.get(str(point_id))
                if sibling is None:
                    continue
                distance = math.hypot(qx - float(sibling["qx"]), qy - float(sibling["qy"]))
                if distance < best_distance:
                    best_distance = distance
                    best_id = int(arc["arc_id"])
        if best_id is None:
            continue
        point["arc_id"] = best_id
        point["topology_flags"] = list(
            dict.fromkeys(point.get("topology_flags", []) + ["attached_unresolved_identity"])
        )


def _arc_topology(
    groups: list[list[int]],
    edges: list[tuple[int, int]],
    candidates: list[dict[str, Any]],
    options: Mapping[str, Any],
    q_step: float,
) -> list[dict[str, Any]]:
    center_x, center_y = float(options["center_qx"]), float(options["center_qy"])
    reference = np.deg2rad(float(options.get("reference_axis_deg", 0.0)))
    reference_vector = np.asarray([np.cos(reference), np.sin(reference)], dtype=float)
    arcs: list[dict[str, Any]] = []
    arc_id = 0
    for component in groups:
        if len(component) < 2:
            continue
        ordered = _ordered_component(component, edges, candidates)
        major, direction_confidence, resolved = _component_major_direction(
            ordered,
            candidates,
            center_x=center_x,
            center_y=center_y,
            reference_vector=reference_vector,
        )
        _assign_component_identities(
            ordered,
            candidates,
            options,
            q_step,
            center_x=center_x,
            center_y=center_y,
            major=major,
            resolved=resolved,
        )
        identity_groups = _split_component_identities(ordered, candidates)
        parent_split = len(identity_groups) > 1
        unresolved = [index for index in ordered if _point_identity_key(candidates[index]) is None]
        created: list[dict[str, Any]] = []
        for key in sorted(identity_groups):
            group = identity_groups[key]
            arc = _build_identity_arc(
                arc_id=arc_id,
                ordered=group,
                candidates=candidates,
                options=options,
                major=major,
                direction_confidence=direction_confidence,
                resolved=resolved,
                parent_split=parent_split,
            )
            arcs.append(arc)
            created.append(arc)
            arc_id += 1
        _attach_unresolved_to_nearest_arc(unresolved, created, candidates)
    for point in candidates:
        if int(point.get("arc_id", -1)) < 0:
            branch, quadrant = _quadrant_branch(
                float(point["qx"]),
                float(point["qy"]),
                center_x=center_x,
                center_y=center_y,
                reference_axis_deg=float(options.get("reference_axis_deg", 0.0)),
                tolerance_deg=float(options["quadrant_axis_tol_deg"]),
            )
            point["branch_id"], point["quadrant"] = int(branch), quadrant
            if str(point.get("side", "unknown")) not in {"upper", "lower"}:
                point["side"] = _side_from_reference_axis(
                    float(point["qx"]),
                    float(point["qy"]),
                    center_x=center_x,
                    center_y=center_y,
                    reference_axis_deg=float(options.get("reference_axis_deg", 0.0)),
                    q_step=q_step,
                )
            extra_flags = ["unconnected_candidate"]
            if point["side"] in {"upper", "lower"}:
                extra_flags.append("side_from_reference_axis")
            point["topology_flag"] = "unconnected_candidate"
            point["topology"] = "unconnected_candidate"
            point["topology_flags"] = list(dict.fromkeys(point.get("topology_flags", []) + extra_flags))
    return arcs


def _refresh_arc_identity(arcs: list[dict[str, Any]], points: list[dict[str, Any]]) -> None:
    """Recheck post-profile/edit identities without dropping observed records."""

    points_by_arc: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for point in points:
        arc_id = int(point.get("arc_id", -1))
        if arc_id >= 0:
            points_by_arc[arc_id].append(point)
    for arc in arcs:
        members = points_by_arc.get(int(arc.get("arc_id", -1)), [])
        for point in members:
            if _point_identity_key(point) is not None:
                continue
            reason = (
                "side_axis_boundary"
                if int(point.get("branch_id", -1)) in (0, 1)
                else "unresolved_point_identity"
            )
            _demote_unresolved_identity_point(point, reason, "unresolved_point_identity")
        branch_ids = sorted({int(point.get("branch_id", -1)) for point in members})
        sides = sorted({str(point.get("side", "unknown")) for point in members})
        accepted_members = [
            point
            for point in members
            if bool(point.get("accepted", False)) and bool(point.get("valid", False))
        ]
        accepted_keys = [_point_identity_key(point) for point in accepted_members]
        known_accepted = [key for key in accepted_keys if key is not None]
        accepted_branch_ids = sorted({key[0] for key in known_accepted})
        accepted_sides = sorted({key[1] for key in known_accepted})
        branch_resolved = len(accepted_branch_ids) == 1 and accepted_branch_ids[0] in (0, 1)
        side_resolved = len(accepted_sides) == 1 and accepted_sides[0] in {"upper", "lower"}
        identity_resolved = bool(branch_resolved and side_resolved and known_accepted)
        identity_flags: list[str] = []
        if not branch_resolved:
            identity_flags.append("unresolved_or_mixed_branch")
        if not side_resolved:
            identity_flags.append("unresolved_or_mixed_side")
        arc["branch_ids"] = branch_ids
        arc["sides"] = sides
        arc["accepted_branch_ids"] = accepted_branch_ids
        arc["accepted_sides"] = accepted_sides
        arc["identity_resolved"] = identity_resolved
        arc["identity_flags"] = identity_flags
        arc["topology_flags"] = list(dict.fromkeys(list(arc.get("topology_flags", [])) + identity_flags))
        if identity_resolved:
            continue
        # Only mixed *known* identities reject the remaining accepted points.
        # Unresolved tip/axis points were already demoted above.
        if known_accepted and not identity_resolved:
            arc["valid"] = False
            arc["reason"] = "unresolved_or_mixed_branch_or_side"
            for point in accepted_members:
                if _point_identity_key(point) is None:
                    continue
                point["accepted"] = False
                point["reason"] = "unresolved_arc_identity"
                point["topology_flags"] = list(
                    dict.fromkeys(list(point.get("topology_flags", [])) + identity_flags + ["unresolved_arc_identity"])
                )
        elif not known_accepted:
            arc["valid"] = False
            if not arc.get("reason"):
                arc["reason"] = "unresolved_or_mixed_branch_or_side"


def _apply_seeds(
    points: list[dict[str, Any]],
    arcs: list[dict[str, Any]],
    seed_actions: Sequence[Mapping[str, Any]],
    options: Mapping[str, Any],
    q_step: float,
) -> list[dict[str, Any]]:
    """Use seeds to select coherent supported graph components.

    Seeds are UI priors over *observed* graph components.  They cannot create
    points or promote a candidate whose local profile/validity gate failed.
    Auto seeds choose the nearest strongest supported component; competing
    components remain in the output with an explicit rejection reason.
    """

    arc_by_id = {int(arc["arc_id"]): arc for arc in arcs}
    points_by_arc: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for point in points:
        arc_id = int(point.get("arc_id", -1))
        if arc_id >= 0:
            points_by_arc[arc_id].append(point)

    def prior_values(action: Mapping[str, Any]) -> tuple[int, str, str]:
        raw_branch = action.get("branch_id", -1)
        try:
            branch = int(raw_branch) if raw_branch is not None else -1
        except (TypeError, ValueError):
            branch = -1
        if branch not in {0, 1}:
            branch = -1
        raw_side = str(action.get("side", "unknown") or "unknown").strip().lower()
        side = raw_side if raw_side in {"upper", "lower"} else "unknown"
        mode = "explicit_prior" if branch in {0, 1} or side != "unknown" else "auto"
        return branch, side, mode

    def arc_conflict(
        arc_points: Sequence[Mapping[str, Any]],
        branch_prior: int,
        side_prior: str,
    ) -> str | None:
        accepted_points = [point for point in arc_points if bool(point.get("accepted", False)) and bool(point.get("valid", False))]
        known_branches = {int(point.get("branch_id", -1)) for point in accepted_points if int(point.get("branch_id", -1)) in {0, 1}}
        known_sides = {str(point.get("side", "unknown")) for point in accepted_points if str(point.get("side", "unknown")) in {"upper", "lower"}}
        if branch_prior in {0, 1} and known_branches and known_branches != {branch_prior}:
            return "seed_branch_conflict"
        if side_prior in {"upper", "lower"} and known_sides and known_sides != {side_prior}:
            return "seed_side_conflict"
        return None

    records: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    for seed_index, action in enumerate(seed_actions):
        try:
            qx, qy = float(action["qx"]), float(action["qy"])
        except (KeyError, TypeError, ValueError):
            records.append({"matched": False, "seed_index": seed_index, "reason": "invalid_seed"})
            continue
        branch_prior, side_prior, mode = prior_values(action)
        nearby_by_arc: dict[int, list[tuple[float, dict[str, Any]]]] = defaultdict(list)
        rejected_nearby: list[dict[str, Any]] = []
        for point in points:
            if not np.isfinite(point.get("qx", np.nan)) or not np.isfinite(point.get("qy", np.nan)):
                continue
            distance = math.hypot(float(point["qx"]) - qx, float(point["qy"]) - qy)
            tolerance = float(options["seed_snap_factor"]) * max(q_step, float(point.get("q_normal_step", q_step)))
            if not np.isfinite(distance) or distance > tolerance:
                continue
            if not bool(point.get("valid", False)) or not bool(point.get("accepted", False)) or int(point.get("arc_id", -1)) < 0:
                rejected_nearby.append({"point_id": str(point.get("point_id", "")), "arc_id": int(point.get("arc_id", -1)), "reason": "seed_candidate_failed_profile_or_validity"})
                continue
            if side_prior == "unknown":
                arc_state = arc_by_id.get(int(point.get("arc_id", -1)), {})
                known_sides = {
                    str(value)
                    for value in arc_state.get("accepted_sides", arc_state.get("sides", []))
                    if str(value) in {"upper", "lower"}
                }
                if len(known_sides) != 1:
                    rejected_nearby.append({"point_id": str(point.get("point_id", "")), "arc_id": int(point.get("arc_id", -1)), "reason": "seed_candidate_unresolved_arc_identity"})
                    continue
            nearby_by_arc[int(point["arc_id"])].append((distance, point))
        if not nearby_by_arc:
            reason = "seed_no_supported_observation" if rejected_nearby else "seed_outside_snap_scale"
            records.append({"matched": False, "seed_index": seed_index, "qx": qx, "qy": qy, "reason": reason, "rejected_candidates": rejected_nearby})
            continue
        compatible: list[tuple[tuple[float, float, float, float], int, str | None]] = []
        conflicts: list[dict[str, Any]] = []
        for arc_id, candidates in nearby_by_arc.items():
            arc_points = points_by_arc.get(arc_id, [])
            conflict = arc_conflict(arc_points, branch_prior, side_prior)
            if conflict is not None:
                conflicts.append({"arc_id": arc_id, "reason": conflict})
                continue
            nearest_distance, nearest_point = min(candidates, key=lambda item: (item[0], -float(item[1].get("snr", 0.0))))
            arc = arc_by_id.get(arc_id, {})
            support = float(arc.get("accepted_points", 0))
            stability = float(arc.get("scale_stability", 0.0))
            compatible.append(((nearest_distance, -support, -stability, -float(nearest_point.get("snr", 0.0))), arc_id, None))
        if not compatible:
            records.append({"matched": False, "seed_index": seed_index, "qx": qx, "qy": qy, "reason": conflicts[0]["reason"] if conflicts else "seed_prior_conflict", "conflicts": conflicts, "rejected_candidates": rejected_nearby})
            continue
        _rank, selected_arc_id, _ = min(compatible, key=lambda item: item[0])
        selected_candidates = nearby_by_arc[selected_arc_id]
        distance, snapped = min(selected_candidates, key=lambda item: (item[0], -float(item[1].get("snr", 0.0))))
        selected_points = points_by_arc.get(selected_arc_id, [])
        competing_arc_ids = sorted(arc_id for _score, arc_id, _reason in compatible if arc_id != selected_arc_id)
        selection = {
            "seed_index": seed_index,
            "selected_arc_id": int(selected_arc_id),
            "selected_points": selected_points,
            "competing_arc_ids": competing_arc_ids,
            "branch_prior": branch_prior,
            "side_prior": side_prior,
            "mode": mode,
        }
        selections.append(selection)
        records.append({
            "matched": True,
            "seed_index": seed_index,
            "qx": qx,
            "qy": qy,
            "point_id": str(snapped["point_id"]),
            "distance_q": float(distance),
            "arc_id": int(selected_arc_id),
            "selected_arc_id": int(selected_arc_id),
            "affected_point_ids": [str(point["point_id"]) for point in selected_points],
            "competing_arc_ids": competing_arc_ids,
            "competing_point_ids": [
                str(point["point_id"])
                for competing_arc_id in competing_arc_ids
                for point in points_by_arc.get(int(competing_arc_id), [])
            ],
            "branch_prior": branch_prior,
            "side_prior": side_prior,
            "selection_mode": mode,
            "manual_prior_provenance": "manual_seed_branch_side_prior" if mode == "explicit_prior" else "manual_seed_auto_component",
            "rejected_candidates": rejected_nearby,
            "reason": "selected_supported_observed_arc",
        })

    selected_arc_ids = {int(selection["selected_arc_id"]) for selection in selections}
    for selection in selections:
        arc_id = int(selection["selected_arc_id"])
        arc = arc_by_id.get(arc_id)
        if arc is not None:
            arc["seed_selected"] = True
            arc["seed_selection_mode"] = selection["mode"]
            arc["seed_provenance"] = "manual_seed_branch_side_prior" if selection["mode"] == "explicit_prior" else "manual_seed_auto_component"
            arc["seed_prior_branch_id"] = selection["branch_prior"]
            arc["seed_prior_side"] = selection["side_prior"]
        for point in selection["selected_points"]:
            point["seeded"] = True
            point["seed_selected"] = True
            point["seed_selection_status"] = "selected_observed_arc"
            point["seed_provenance"] = "manual_seed_branch_side_prior" if selection["mode"] == "explicit_prior" else "manual_seed_auto_component"
            point["seed_ids"] = list(dict.fromkeys(list(point.get("seed_ids", [])) + [selection["seed_index"]]))
            point["topology_flags"] = list(dict.fromkeys(point.get("topology_flags", []) + ["seed_selected_observed_arc"]))
            if not bool(point.get("accepted", False)) or not bool(point.get("valid", False)):
                continue
            branch_prior = int(selection["branch_prior"])
            if branch_prior in {0, 1}:
                point["branch_id"] = branch_prior
                point["seed_branch_prior_applied"] = True
            side_prior = str(selection["side_prior"])
            if side_prior in {"upper", "lower"}:
                point["side"] = side_prior
                point["seed_side_prior_applied"] = True
        if arc is not None:
            accepted_arc_points = [point for point in selection["selected_points"] if bool(point.get("accepted", False)) and bool(point.get("valid", False))]
            arc["branch_ids"] = sorted({int(point.get("branch_id", -1)) for point in accepted_arc_points})
            arc["sides"] = sorted({str(point.get("side", "unknown")) for point in accepted_arc_points})
            arc["seed_coherent"] = bool(
                (selection["branch_prior"] in {-1, 0, 1})
                and (selection["side_prior"] in {"unknown", "upper", "lower"})
                and (selection["side_prior"] == "unknown" or set(arc["sides"]) <= {selection["side_prior"]})
            )
            arc["seed_affected_point_ids"] = [str(point["point_id"]) for point in selection["selected_points"]]

    # A selected seed makes a competing supported fragment explicitly
    # ineligible for this seeded trajectory while retaining every point and
    # its original acceptance state for review/correction.
    for selection in selections:
        for competing_arc_id in selection["competing_arc_ids"]:
            if int(competing_arc_id) in selected_arc_ids:
                continue
            arc = arc_by_id.get(int(competing_arc_id))
            if arc is not None:
                arc["seed_competing"] = True
                arc["seed_selection_reason"] = "competing_seed_component"
            for point in points_by_arc.get(int(competing_arc_id), []):
                point["seed_competing"] = True
                point["seed_selection_status"] = "competing_seed_component"
                point["seed_selection_reason"] = "competing_seed_component"
                point["topology_flags"] = list(dict.fromkeys(point.get("topology_flags", []) + ["seed_competing_component"]))
                if bool(point.get("accepted", False)):
                    point["seed_original_reason"] = point.get("reason", "accepted")
                    point["accepted"] = False
                    point["reason"] = "competing_seed_component"
    for arc in arcs:
        arc_points = points_by_arc.get(int(arc["arc_id"]), [])
        arc["accepted_points_after_seed"] = int(sum(bool(point.get("accepted", False)) for point in arc_points))
        arc["valid_after_seed"] = bool(arc["accepted_points_after_seed"] >= int(options.get("min_arc_points", 2)))
    return records


def _apply_profile_refinement(
    point: dict[str, Any],
    profile: Mapping[str, Any],
    arcs_by_id: Mapping[int, Mapping[str, Any]],
    options: Mapping[str, Any],
    q_step: float,
) -> None:
    """Apply an observed single-peak normal shift with consistency gates.

    The curvature zero is retained as ``curvature_seed_*``.  A profile fit is
    allowed to move the public point only when it is a supported, unambiguous
    nearby observation and the shift preserves the pre-fit branch/quadrant
    and side assignment.  Failed or ambiguous profiles remain visible without
    promoting a seed to an accepted observation.
    """

    point.setdefault("curvature_seed_qx", float(point.get("qx", float("nan"))))
    point.setdefault("curvature_seed_qy", float(point.get("qy", float("nan"))))
    point.setdefault("curvature_seed_pixel_x", float(point.get("pixel_x", float("nan"))))
    point.setdefault("curvature_seed_pixel_y", float(point.get("pixel_y", float("nan"))))
    point.setdefault("profile_refinement_applied", False)
    point.setdefault("profile_shift_q", 0.0)
    point.setdefault("profile_shift_pixel", 0.0)
    point.setdefault("profile_refinement_reason", "not_attempted")
    point.setdefault("profile_center_offset_q", float("nan"))
    if not bool(options.get("profile_refinement", True)):
        point["profile_refinement_reason"] = "disabled_by_options"
        return
    if not bool(point.get("accepted", False)):
        point["profile_refinement_reason"] = "curvature_candidate_not_accepted"
        return
    if not bool(profile.get("valid", False)):
        point["profile_refinement_reason"] = "profile_unavailable"
        return
    if bool(profile.get("ambiguous", False)) or int(profile.get("peak_count", 0)) != 1:
        point["profile_refinement_reason"] = "ambiguous_or_multiple_normal_peaks"
        point["topology_flags"] = list(dict.fromkeys(point.get("topology_flags", []) + ["profile_refinement_not_applied_ambiguous"]))
        return
    profile_snr = float(profile.get("snr", float("nan")))
    if not np.isfinite(profile_snr) or profile_snr < float(options["profile_refinement_min_snr"]):
        point["profile_refinement_reason"] = "profile_snr_below_refinement_floor"
        return
    support_fraction = float(profile.get("support_fraction", 0.0))
    if not np.isfinite(support_fraction) or support_fraction < float(options["support_min"]):
        point["profile_refinement_reason"] = "profile_support_below_refinement_floor"
        return
    offset_q = float(profile.get("peak_center_q", float("nan")))
    step = float(point.get("q_normal_step", q_step))
    max_shift = float(options["profile_refinement_max_q_step"]) * max(step, q_step, np.finfo(float).eps)
    if not np.isfinite(offset_q) or abs(offset_q) > max_shift:
        point["profile_refinement_reason"] = "profile_shift_exceeds_local_scale"
        point["profile_center_offset_q"] = offset_q
        return
    normal_x = float(point.get("normal_qx", float("nan")))
    normal_y = float(point.get("normal_qy", float("nan")))
    normal_norm = float(np.hypot(normal_x, normal_y))
    if not np.isfinite(normal_norm) or normal_norm <= np.finfo(float).eps:
        point["profile_refinement_reason"] = "profile_shift_missing_unit_normal"
        return
    normal_x, normal_y = normal_x / normal_norm, normal_y / normal_norm
    old_qx, old_qy = float(point["qx"]), float(point["qy"])
    new_qx = float(profile.get("peak_center_qx", float("nan")))
    new_qy = float(profile.get("peak_center_qy", float("nan")))
    if not np.isfinite(new_qx) or not np.isfinite(new_qy):
        new_qx, new_qy = old_qx + offset_q * normal_x, old_qy + offset_q * normal_y
    new_pixel_x = float(profile.get("peak_center_pixel_x", float("nan")))
    new_pixel_y = float(profile.get("peak_center_pixel_y", float("nan")))
    if not np.isfinite(new_pixel_x) or not np.isfinite(new_pixel_y):
        new_pixel_x, new_pixel_y = float(point["pixel_x"]), float(point["pixel_y"])
    center_x, center_y = float(options["center_qx"]), float(options["center_qy"])
    branch, quadrant = _quadrant_branch(
        new_qx,
        new_qy,
        center_x=center_x,
        center_y=center_y,
        reference_axis_deg=float(options.get("reference_axis_deg", 0.0)),
        tolerance_deg=float(options["quadrant_axis_tol_deg"]),
    )
    old_branch = int(point.get("branch_id", -1))
    old_quadrant = str(point.get("quadrant", "unknown"))
    if old_branch in {0, 1} and branch != old_branch:
        point["profile_refinement_reason"] = "profile_shift_crosses_branch_boundary"
        return
    if old_quadrant not in {"unknown", "axis_boundary"} and quadrant not in {old_quadrant, "axis_boundary"}:
        point["profile_refinement_reason"] = "profile_shift_crosses_quadrant_boundary"
        return
    arc = arcs_by_id.get(int(point.get("arc_id", -1)))
    new_side = str(point.get("side", "unknown"))
    if arc is not None and bool(arc.get("major_direction_resolved", False)):
        major_x = float(arc.get("major_direction_qx", float("nan")))
        major_y = float(arc.get("major_direction_qy", float("nan")))
        radial_x, radial_y = new_qx - center_x, new_qy - center_y
        side_value = radial_x * (-major_y) + radial_y * major_x
        side_scale = _side_axis_threshold(float(np.hypot(radial_x, radial_y)), q_step)
        new_side = "upper" if side_value > side_scale else "lower" if side_value < -side_scale else "unknown"
        if str(point.get("side", "unknown")) in {"upper", "lower"} and new_side != str(point["side"]):
            point["profile_refinement_reason"] = "profile_shift_crosses_side_boundary"
            return
    point["profile_center_offset_q"] = offset_q
    point["profile_shift_q"] = float(np.hypot(new_qx - old_qx, new_qy - old_qy))
    point["profile_shift_pixel"] = float(np.hypot(new_pixel_x - float(point["pixel_x"]), new_pixel_y - float(point["pixel_y"])))
    point["profile_refined_qx"] = new_qx
    point["profile_refined_qy"] = new_qy
    point["profile_refined_pixel_x"] = new_pixel_x
    point["profile_refined_pixel_y"] = new_pixel_y
    point["qx"], point["qy"] = new_qx, new_qy
    point["pixel_x"], point["pixel_y"] = new_pixel_x, new_pixel_y
    point["branch_id"], point["quadrant"], point["side"] = int(branch), quadrant, new_side
    point["profile_refinement_applied"] = True
    point["profile_refinement_reason"] = "observed_single_normal_peak"
    point["topology_flags"] = list(dict.fromkeys(point.get("topology_flags", []) + ["profile_refined_observed_peak"]))


def _public_point(point: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in point.items():
        if key.startswith("_"):
            continue
        if isinstance(value, np.generic):
            value = value.item()
        result[key] = value
    result.setdefault("branch_id", -1)
    result.setdefault("quadrant", "unknown")
    result.setdefault("arc_id", -1)
    result.setdefault("side", "unknown")
    result.setdefault("accepted", False)
    result.setdefault("valid", False)
    result.setdefault("reason", "unknown")
    result.setdefault("topology_flags", [])
    result.setdefault("topology_flag", "unknown")
    result.setdefault("topology", result["topology_flag"])
    result.setdefault("scale_stability", float("nan"))
    result.setdefault("scale_stable", False)
    result.setdefault("normal_fwhm_q", float("nan"))
    result.setdefault("localization_sigma_q", float("nan"))
    result.setdefault("sampling_sigma_q", float("nan"))
    result.setdefault("profile_refinement_applied", False)
    result.setdefault("profile_shift_q", 0.0)
    result.setdefault("profile_shift_pixel", 0.0)
    result.setdefault("profile_refinement_reason", "not_attempted")
    return result


def _deferred_rejected_profile(point: Mapping[str, Any]) -> dict[str, Any]:
    """Keep a review stub without sampling a detector-sized profile."""

    return {
        "point_id": str(point.get("point_id", "")),
        "valid": True,
        "model": "deferred_candidate_profile",
        "reason": "fit_deferred_rejected_candidate",
        "fit_deferred": True,
        "ambiguous": False,
        "offset_q": [],
        "raw_intensity": [],
        "fit_intensity": [],
        "peaks": [],
        "peak_count": 0,
        "snr": float("nan"),
        "normal_fwhm_q": float("nan"),
        "localization_sigma_q": float("nan"),
        "sampling_sigma_q": float("nan"),
        "uncertainty_source": "not_fit_rejected_candidate",
        "support_fraction": 0.0,
    }


def trace_butterfly_ridges(
    image: np.ndarray | Mapping[str, Any] | Any,
    qmap: Mapping[str, Any] | Any,
    q_window: Any,
    *,
    mask: Any = None,
    reference_axis_deg: float = 0.0,
    options: Any = None,
    edits: Any = None,
    cancel_event: Any = None,
) -> dict[str, Any]:
    """Trace observed butterfly ridges and return points, arcs and profiles.

    The output is intentionally a plain mapping so the GUI and JSON export
    can preserve rejected candidates without depending on the legacy
    ``observables`` dataclasses.  ``branch_id`` is 0 for QI/QIII, 1 for
    QII/QIV, and -1 when the reference-axis quadrant is ambiguous.  The
    result contains no ellipse fit and makes that topology-first boundary
    explicit in ``diagnostics``.
    """

    _check_cancelled(cancel_event, "ridge-trace:input")
    started = time.perf_counter()
    image_array, image_mask = _as_image(image)
    qx, qy, q, qmap_mask = _as_qmap(qmap, image_array.shape)
    q_min, q_max = _parse_q_window(q_window, q)
    opts = _normalise_options(options)
    if edits is None and opts.get("edits") is not None:
        edits = opts.get("edits")
    opts["reference_axis_deg"] = float(reference_axis_deg)
    if not np.isfinite(opts["reference_axis_deg"]):
        raise ValueError("reference_axis_deg must be finite")
    center_hint = opts.get("center_hint", opts.get("center_q", None))
    if center_hint is not None:
        opts["center_qx"], opts["center_qy"] = float(center_hint[0]), float(center_hint[1])
    valid = np.isfinite(image_array) & np.isfinite(qx) & np.isfinite(qy) & np.isfinite(q) & (q >= q_min) & (q <= q_max)
    if image_mask is not None:
        valid &= ~image_mask
    if qmap_mask is not None:
        valid &= ~qmap_mask
    if mask is not None:
        try:
            explicit_mask = np.asarray(np.broadcast_to(np.asarray(mask, dtype=bool), image_array.shape), dtype=bool)
        except ValueError as exc:
            raise ValueError("mask must broadcast to image shape") from exc
        valid &= ~explicit_mask
    valid, applied_edits, seed_actions = _apply_edits(valid, qx, qy, edits)
    finite_domain = np.isfinite(qx) & np.isfinite(qy) & np.isfinite(q) & (q >= q_min) & (q <= q_max)
    first_order = _first_order_q_hint(q, image_array, valid, q_min, q_max)
    first_order["applied"] = False
    if not np.any(finite_domain):
        return {
            "points": [],
            "arcs": [],
            "profiles": {},
            "diagnostics": {"topology_before_ellipse_fit": True, "ellipse_fit": None, "reason": "empty_q_window"},
            "method_version": METHOD_VERSION,
        }
    ys, xs = np.where(finite_domain)
    pad = int(np.ceil(5.0 * max(opts["smoothing_scales"]))) + 3
    row0, row1 = max(0, int(ys.min()) - pad), min(image_array.shape[0], int(ys.max()) + pad + 1)
    col0, col1 = max(0, int(xs.min()) - pad), min(image_array.shape[1], int(xs.max()) + pad + 1)
    crop_image = image_array[row0:row1, col0:col1]
    crop_qx, crop_qy = qx[row0:row1, col0:col1], qy[row0:row1, col0:col1]
    crop_valid = valid[row0:row1, col0:col1]
    _check_cancelled(cancel_event, "ridge-trace:crop")
    signature = _point_signature(image_array, opts, (q_min, q_max))
    raw_candidates: list[dict[str, Any]] = []
    coordinate_cache = _coordinate_derivatives(crop_qx, crop_qy)
    for scale in opts["smoothing_scales"]:
        _check_cancelled(cancel_event, f"ridge-trace:curvature:{scale:g}")
        field = _scaled_surface_field(
            crop_image,
            crop_qx,
            crop_qy,
            crop_valid,
            sigma=float(scale),
            options=opts,
            coordinate_derivatives=coordinate_cache,
        )
        raw_candidates.extend(
            _raw_candidates(
                field,
                row_offset=row0,
                col_offset=col0,
                max_candidates=int(opts["max_candidates"]),
                options=opts,
            )
        )
    raw_steps = [
        float(candidate["q_normal_step"])
        for candidate in raw_candidates
        if np.isfinite(candidate["q_normal_step"]) and candidate["q_normal_step"] > 0
    ]
    q_step = float(np.nanmedian(raw_steps)) if raw_steps else float(coordinate_cache[2])
    if not np.isfinite(q_step) or q_step <= 0:
        q_step = 1.0
    if bool(opts.get("first_order_prefer", True)):
        n_weighted = _prefer_first_order_scores(raw_candidates, first_order.get("q_star"))
        first_order["applied"] = bool(n_weighted)
        first_order["n_weighted"] = int(n_weighted)
    candidates = _nms(raw_candidates, opts, q_step)
    if opts["max_points"] is not None and len(candidates) > int(opts["max_points"]):
        candidates = sorted(candidates, key=lambda point: (-float(point.get("score", 0.0)), float(point["qx"]), float(point["qy"])))[: int(opts["max_points"])]
    _assign_point_ids(candidates, signature)
    # Scale support is measured after NMS so the point records can expose it
    # without carrying duplicate candidate pixels into the graph.
    _annotate_scale_stability(
        candidates,
        raw_candidates,
        q_step,
        opts["smoothing_scales"],
        cancel_event=cancel_event,
    )
    _check_cancelled(cancel_event, "ridge-trace:nms")
    _assign_reference_branches(candidates, opts)
    groups, edges = _graph_arcs(candidates, opts, q_step)
    arcs = _arc_topology(groups, edges, candidates, opts, q_step)
    _check_cancelled(cancel_event, "ridge-trace:topology")
    # Local profiles are intentionally imported lazily: the legacy
    # observables module can call this module without creating an import loop.
    from .ridge_profiles import build_profile_context, extract_normal_profile, wang2007_vertical_slice_check

    profile_context = (
        build_profile_context(image_array, {"qx": qx, "qy": qy, "q": q}, mask=~valid)
        if candidates
        else None
    )
    arcs_by_id = {int(arc["arc_id"]): arc for arc in arcs}
    profiles: dict[str, dict[str, Any]] = {}
    n_profiled = 0
    for point in candidates:
        _check_cancelled(cancel_event, "ridge-trace:profiles")
        if not bool(point.get("accepted", False)):
            profile = _deferred_rejected_profile(point)
        else:
            n_profiled += 1
            profile = extract_normal_profile(
                image_array,
                {"qx": qx, "qy": qy, "q": q},
                point,
                mask=~valid,
                options={
                    **opts,
                    "_profile_context": profile_context,
                    "fit_gaussian": True,
                    "profile_half_width_q": opts["profile_half_width_q"] or 5.0 * max(float(point.get("q_normal_step", q_step)), q_step),
                },
                cancel_event=cancel_event,
            )
        point_id = str(point["point_id"])
        profiles[point_id] = profile
        point["normal_fwhm_q"] = float(profile.get("normal_fwhm_q", float("nan")))
        point["localization_sigma_q"] = float(profile.get("localization_sigma_q", float("nan")))
        point["sampling_sigma_q"] = float(profile.get("sampling_sigma_q", float("nan")))
        point["uncertainty_source"] = str(profile.get("uncertainty_source", "unknown"))
        profile_snr = float(profile.get("snr", float("nan")))
        if np.isfinite(profile_snr):
            point["snr"] = profile_snr
        if not bool(profile.get("valid", False)):
            point["accepted"] = False
            point["reason"] = "profile_unavailable"
            point["topology_flags"] = list(dict.fromkeys(point["topology_flags"] + ["profile_unavailable"]))
        elif bool(profile.get("fit_deferred", False)):
            point["topology_flags"] = list(dict.fromkeys(point["topology_flags"] + ["profile_fit_deferred_for_rejected_candidate"]))
        elif bool(profile.get("ambiguous", False)):
            point["accepted"] = False
            point["reason"] = "ambiguous_normal_profile"
            point["topology_flags"] = list(dict.fromkeys(point["topology_flags"] + ["ambiguous_normal_profile"]))
        elif not profile.get("peaks"):
            point["accepted"] = False
            point["reason"] = "no_normal_peak"
            point["topology_flags"] = list(dict.fromkeys(point["topology_flags"] + ["no_normal_peak"]))
        _apply_profile_refinement(point, profile, arcs_by_id, opts, q_step)
    seed_records = _apply_seeds(candidates, arcs, seed_actions, opts, q_step)
    excluded_ids: set[str] = set()
    for action in list(edits or [] if not isinstance(edits, Mapping) else edits.get("actions", edits.get("edits", []))):
        if isinstance(action, Mapping) and str(action.get("type", "")).lower().replace("-", "_") == "exclude_point":
            point_id = str(action.get("point_id", ""))
            if point_id:
                excluded_ids.add(point_id)
    for point in candidates:
        if str(point["point_id"]) in excluded_ids:
            point["accepted"] = False
            point["reason"] = "excluded_point_edit"
            point["topology_flags"] = list(dict.fromkeys(point["topology_flags"] + ["excluded_point_edit"]))
    _refresh_arc_identity(arcs, candidates)
    radial_split = _demote_secondary_radial_population(
        candidates, prefer_radius=first_order.get("q_star")
    )
    ring_fill = _fill_sparse_first_order_ring(
        candidates,
        qx=qx,
        qy=qy,
        q=q,
        intensity=image_array,
        valid=valid,
        hint=first_order.get("q_star"),
        options=opts,
        q_step=q_step,
        signature=signature,
        arcs=arcs,
    )
    for point in candidates:
        point_id = str(point.get("point_id", ""))
        if point_id and point_id not in profiles:
            profiles[point_id] = {
                "point_id": point_id,
                "valid": True,
                "model": "azimuthal_radial_peak",
                "reason": "first_order_ring_sample",
                "fit_deferred": False,
                "ambiguous": False,
                "offset_q": [],
                "raw_intensity": [],
                "fit_intensity": [],
                "peaks": [{"q": float(np.hypot(float(point["qx"]), float(point["qy"])))}],
                "peak_count": 1,
                "snr": float(point.get("snr", float("nan"))),
                "normal_fwhm_q": float("nan"),
                "localization_sigma_q": float(point.get("localization_sigma_q", q_step)),
                "sampling_sigma_q": float(q_step),
                "uncertainty_source": "azimuthal_radial_peak",
                "support_fraction": 1.0,
            }
    _check_cancelled(cancel_event, "ridge-trace:edits")
    # Freeze finite observed support only after profile refinement and all
    # seed/exclude edits.  The freezer mutates the working records so its
    # point-local/arc-local JSON fields survive public serialization; retain
    # only its compact summary in tracer diagnostics.
    from .arc_support import freeze_observed_support

    _check_cancelled(cancel_event, "ridge-trace:support")
    support_envelope = freeze_observed_support(
        candidates, arcs, summary_only=True, cancel_event=cancel_event
    )
    support_summary = dict(support_envelope.get("summary", {}))
    support_summary.update(
        {
            "support_version": support_envelope.get("support_version", support_envelope.get("version")),
            "support_status": support_envelope.get("support_status", support_envelope.get("status", "unavailable")),
            "support_frozen": bool(support_envelope.get("support_frozen", support_envelope.get("frozen", False))),
        }
    )
    if bool(opts.get("run_wang_check", True)):
        wang = wang2007_vertical_slice_check(
            image_array,
            {"qx": qx, "qy": qy, "q": q},
            (q_min, q_max),
            mask=~valid,
            reference_axis_deg=float(reference_axis_deg),
            options=opts,
            cancel_event=cancel_event,
        )
    else:
        wang = {"applicable": False, "diagnostic_only": True, "used_for_acceptance": False, "reason": "disabled_by_options"}
    public_points = [_public_point(point) for point in candidates]
    diagnostics: dict[str, Any] = {
        "topology_before_ellipse_fit": True,
        "ellipse_fit": None,
        "curvature_definition": "minimum principal curvature of the scaled q-space graph surface from first/second fundamental forms",
        "ridge_condition": "directional slope zero with negative minimum principal curvature",
        "method": "mask_normalized_multiscale_principal_curvature_zero_slope",
        "reference_axis_deg": float(reference_axis_deg),
        "center_q": [float(opts["center_qx"]), float(opts["center_qy"])],
        "q_window": [float(q_min), float(q_max)],
        "first_order_q_hint": first_order,
        "first_order_ring_fill": ring_fill,
        "smoothing_scales": [float(value) for value in opts["smoothing_scales"]],
        "crop": {"row_start": row0, "row_stop": row1, "col_start": col0, "col_stop": col1, "shape": [row1 - row0, col1 - col0], "original_shape": list(image_array.shape)},
        "n_raw_candidates": int(len(raw_candidates)),
        "n_points": int(len(public_points)),
        "n_profiled_points": int(n_profiled),
        "n_accepted_points": int(sum(bool(point["accepted"]) for point in public_points)),
        "elapsed_s": float(time.perf_counter() - started),
        "n_arcs": int(len(arcs)),
        "n_valid_arcs": int(sum(bool(arc["valid"]) for arc in arcs)),
        "scale_stability": {"mean": float(np.nanmean([point["scale_stability"] for point in public_points])) if public_points else float("nan"), "min": float(np.nanmin([point["scale_stability"] for point in public_points])) if public_points else float("nan")},
        "mask_fraction_in_q_window": float(1.0 - np.count_nonzero(crop_valid) / max(1, np.count_nonzero(finite_domain[row0:row1, col0:col1]))),
        "edits": applied_edits,
        "seed_matches": seed_records,
        "excluded_point_ids": sorted(excluded_ids),
        "observed_support": support_summary,
        "radial_population": radial_split,
        "wang2007_vertical_slice": wang,
        "flags": ["observed_only", "no_ellipse_fit", "rejected_candidates_retained", "topology_before_ellipse_fit"]
        + (["mixed_radial_populations"] if radial_split.get("split") else []),
    }
    return {"points": public_points, "arcs": arcs, "profiles": profiles, "diagnostics": diagnostics, "method_version": METHOD_VERSION}


__all__ = ["METHOD_VERSION", "trace_butterfly_ridges"]
