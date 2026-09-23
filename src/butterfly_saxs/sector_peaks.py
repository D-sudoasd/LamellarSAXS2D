"""Observed radial peaks from overlapping azimuthal sector integrals.

This module is deliberately independent from the butterfly ridge and ellipse
fit paths.  It answers a narrower measurement question: for each angular
sector, what radial features are present in the measured intensity profile?
The result is an audit-friendly collection of raw profiles and candidates.
It does not index a reflection, impose four lobes, fill masked quadrants, or
choose a peak from an empty/ambiguous profile.

The sector profiles are formed from actual detector pixels.  Smoothing is
performed separately on each contiguous supported radial run, so a detector
hole cannot be bridged into a synthetic peak.  A selected peak reports a
representative source pixel from the measured support bin; its sub-bin q
coordinate is a localisation aid, not a fitted physical model.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .cancellation import raise_if_cancelled
from .observables import _q_unit
from .ridge_inputs import Q_ALIASES, array_field, canonical_inputs, robust_noise

try:  # scipy is a declared project dependency, but retain a small fallback.
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks
except Exception:  # pragma: no cover - only useful in partial installs
    gaussian_filter1d = None
    find_peaks = None


SECTOR_PEAKS_SCHEMA_VERSION = "sector-peaks-v1"
SECTOR_PEAKS_METHOD_VERSION = "sector-peaks-v1.0"

_EPS = np.finfo(float).eps


def _options_mapping(options: Any) -> dict[str, Any]:
    if options is None:
        return {}
    if isinstance(options, Mapping):
        return dict(options)
    try:
        return dict(vars(options))
    except TypeError as exc:
        raise TypeError("options must be a mapping or expose attributes") from exc


def _option(values: Mapping[str, Any], name: str, default: Any, *aliases: str) -> Any:
    for key in (name, *aliases):
        if key in values:
            return values[key]
    return default


def _finite_float(value: Any, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite number, not bool")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum:g}")
    return result


def _strict_int(value: Any, name: str, *, minimum: int = 0) -> int:
    """Parse an integer option without silently truncating floats or bools."""

    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer, not bool")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if not np.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(f"{name} must be an integer")
    result = int(numeric)
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _q_window(value: Any) -> tuple[float, float]:
    if isinstance(value, Mapping):
        low = value.get("q_min", value.get("min", value.get("low", value.get("start"))))
        high = value.get("q_max", value.get("max", value.get("high", value.get("stop"))))
    else:
        try:
            low, high = value
        except (TypeError, ValueError) as exc:
            raise ValueError("q_window must be a finite (q_min, q_max) pair") from exc
    low = _finite_float(low, "q_window[0]")
    high = _finite_float(high, "q_window[1]")
    if high <= low:
        raise ValueError("q_window must have q_max > q_min")
    return low, high


def _contiguous_runs(supported: np.ndarray) -> list[tuple[int, int]]:
    """Return half-open runs of True values without bridging holes."""

    indices = np.flatnonzero(np.asarray(supported, dtype=bool))
    if not indices.size:
        return []
    breaks = np.flatnonzero(np.diff(indices) > 1)
    starts = np.r_[0, breaks + 1]
    stops = np.r_[breaks + 1, indices.size]
    return [(int(indices[start]), int(indices[stop - 1]) + 1) for start, stop in zip(starts, stops)]


def _angle_delta_deg(angle: np.ndarray, centre: float) -> np.ndarray:
    return np.abs((np.asarray(angle, dtype=float) - float(centre) + 180.0) % 360.0 - 180.0)


def _smooth_supported(profile: np.ndarray, supported: np.ndarray, sigma: float) -> np.ndarray:
    """Smooth each supported run independently and leave holes as NaN."""

    result = np.full(np.asarray(profile).shape, np.nan, dtype=float)
    sigma = float(sigma)
    for left, right in _contiguous_runs(supported):
        values = np.asarray(profile[left:right], dtype=float)
        if gaussian_filter1d is not None and sigma > 0.0 and values.size >= 3:
            result[left:right] = gaussian_filter1d(values, sigma=sigma, mode="nearest")
        else:
            result[left:right] = values
    return result


def _profile_noise(profile: np.ndarray, smooth: np.ndarray, supported: np.ndarray) -> float:
    """Estimate profile noise from raw-minus-smooth and adjacent raw bins.

    The estimate is a profile-local robust scale, not an experimental
    uncertainty or detector calibration sigma.  Only contiguous supported
    bins contribute, so masked gaps do not become a noise window bridge.
    """

    residuals: list[np.ndarray] = []
    differences: list[np.ndarray] = []
    for left, right in _contiguous_runs(supported):
        raw = np.asarray(profile[left:right], dtype=float)
        filtered = np.asarray(smooth[left:right], dtype=float)
        finite = np.isfinite(raw) & np.isfinite(filtered)
        if np.count_nonzero(finite) >= 2:
            residuals.append(raw[finite] - filtered[finite])
        if raw.size >= 2:
            adjacent = np.isfinite(raw[:-1]) & np.isfinite(raw[1:])
            if np.any(adjacent):
                differences.append(np.diff(raw)[adjacent])
    candidates: list[float] = []
    if residuals:
        value = robust_noise(np.concatenate(residuals))
        if np.isfinite(value):
            candidates.append(float(value))
    if differences:
        value = robust_noise(np.concatenate(differences)) / np.sqrt(2.0)
        if np.isfinite(value):
            candidates.append(float(value))
    if not candidates:
        finite_values = np.asarray(profile, dtype=float)[np.asarray(supported, dtype=bool)]
        value = robust_noise(finite_values)
        if np.isfinite(value):
            candidates.append(float(value))
    finite_values = np.asarray(profile, dtype=float)[np.asarray(supported, dtype=bool)]
    scale = max(float(np.ptp(finite_values)) if finite_values.size else 0.0, 1.0)
    return max(float(max(candidates, default=0.0)), _EPS * scale)


def _robust_raw_baseline(values: np.ndarray) -> float:
    """Estimate a raw-profile baseline while trimming isolated outliers."""

    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return float("nan")
    centre = float(np.median(finite))
    mad = robust_noise(finite - centre)
    if np.isfinite(mad) and mad > _EPS:
        inlier = np.abs(finite - centre) <= 6.0 * mad
        trimmed = finite[inlier]
        if trimmed.size >= max(3, int(np.ceil(0.5 * finite.size))):
            finite = trimmed
    return float(np.percentile(finite, 10.0))


def _quadratic_localisation(q: np.ndarray, profile: np.ndarray, index: int) -> tuple[float, float]:
    """Local parabolic interpolation on three supported bins only."""

    index = int(index)
    q_value = float(q[index])
    intensity = float(profile[index])
    if index <= 0 or index >= len(profile) - 1:
        return q_value, intensity
    left, centre, right = (float(profile[index - 1]), float(profile[index]), float(profile[index + 1]))
    denominator = left - 2.0 * centre + right
    if not np.isfinite(denominator) or abs(denominator) <= _EPS:
        return q_value, intensity
    offset = 0.5 * (left - right) / denominator
    if not np.isfinite(offset):
        return q_value, intensity
    offset = float(np.clip(offset, -0.5, 0.5))
    step = float(q[index + 1] - q[index])
    if not np.isfinite(step) or step <= 0.0:
        return q_value, intensity
    q_value += offset * step
    intensity -= 0.25 * (left - right) * offset
    return q_value, intensity


def _manual_peak_indices(values: np.ndarray, prominence: float, distance: int) -> list[int]:
    """Small fallback for environments without scipy.signal."""

    if values.size < 3:
        return []
    result: list[int] = []
    for index in range(1, values.size - 1):
        if not (values[index] > values[index - 1] and values[index] >= values[index + 1]):
            continue
        left_min = float(np.min(values[: index + 1]))
        right_min = float(np.min(values[index:]))
        if values[index] - max(left_min, right_min) < prominence:
            continue
        if result and index - result[-1] < distance:
            if values[index] > values[result[-1]]:
                result[-1] = index
            continue
        result.append(index)
    return result


def _side_background_return(
    values: np.ndarray,
    local_index: int,
    baseline: float,
    *,
    fraction: float,
) -> tuple[bool, bool]:
    """Check that each side returns toward its local background."""

    height = float(values[local_index] - baseline)
    if not np.isfinite(height) or height <= 0.0:
        return False, False
    threshold = float(values[local_index] - fraction * height)
    left_return = bool(np.nanmin(values[:local_index]) <= threshold) if local_index else False
    right_return = bool(np.nanmin(values[local_index + 1 :]) <= threshold) if local_index + 1 < values.size else False
    return left_return, right_return


def _flat_top_metrics(
    values: np.ndarray,
    local_index: int,
    baseline: float,
    width_bins: float,
    fraction: float,
) -> tuple[int, float]:
    """Measure a flat top relative to the candidate's half-height width."""

    peak = float(values[local_index])
    height = peak - float(baseline)
    if not np.isfinite(height) or height <= 0.0:
        return 0, 0.0
    near_peak_threshold = float(baseline + float(fraction) * height)
    left = local_index
    right = local_index
    while left > 0 and float(values[left - 1]) >= near_peak_threshold:
        left -= 1
    while right + 1 < values.size and float(values[right + 1]) >= near_peak_threshold:
        right += 1
    top_width = int(right - left + 1)
    ratio = float(top_width / max(float(width_bins), 1.0))
    return top_width, ratio


def _source_record(
    *,
    candidate: dict[str, Any],
    q_edges: np.ndarray,
    q_values: np.ndarray,
    intensity_values: np.ndarray,
    qx_values: np.ndarray,
    qy_values: np.ndarray,
    original_indices: np.ndarray,
    image_shape: tuple[int, int],
    baseline: float,
    neighbourhood_bins: int,
) -> dict[str, Any]:
    """Attach measured source-pixel coordinates and support statistics."""

    bin_index = int(candidate["peak_bin_index"])
    in_bin = (
        np.isfinite(q_values)
        & np.isfinite(intensity_values)
        & (q_values >= q_edges[bin_index])
        & ((q_values < q_edges[bin_index + 1]) | (bin_index == len(q_edges) - 2))
    )
    indices = np.flatnonzero(in_bin)
    if not indices.size:
        candidate["source_pixel_count"] = 0
        candidate["pixel_x"] = None
        candidate["pixel_y"] = None
        candidate["representative_pixel"] = None
        return candidate
    q_subset = np.asarray(q_values[indices], dtype=float)
    i_subset = np.asarray(intensity_values[indices], dtype=float)
    target_q = float(candidate["q_star"])
    distance = np.abs(q_subset - target_q)
    # A real source pixel is chosen by q proximity, then by observed intensity.
    order = np.lexsort((-i_subset, distance))
    selected_local = int(indices[int(order[0])])
    source_flat = int(original_indices[selected_local])
    pixel_y, pixel_x = np.unravel_index(source_flat, image_shape)
    source_angles = np.arctan2(qy_values[indices], qx_values[indices])
    source_angle = float(np.degrees(np.arctan2(np.mean(np.sin(source_angles)), np.mean(np.cos(source_angles)))) % 360.0)
    neighbourhood_left = max(0, bin_index - int(neighbourhood_bins))
    neighbourhood_right = min(len(q_edges) - 2, bin_index + int(neighbourhood_bins))
    support = (
        np.isfinite(q_values)
        & np.isfinite(intensity_values)
        & (q_values >= q_edges[neighbourhood_left])
        & ((q_values < q_edges[neighbourhood_right + 1]) | (neighbourhood_right == len(q_edges) - 2))
    )
    positive_excess = np.maximum(np.asarray(intensity_values[support], dtype=float) - float(baseline), 0.0)
    weight_sum = float(np.sum(positive_excess))
    weight_square_sum = float(np.sum(np.square(positive_excess)))
    if weight_sum > _EPS and weight_square_sum > _EPS:
        effective_pixels = float(weight_sum * weight_sum / weight_square_sum)
        max_fraction = float(np.max(positive_excess) / weight_sum)
    else:
        effective_pixels = 0.0
        max_fraction = float("nan")
    candidate.update(
        {
            "source_pixel_count": int(indices.size),
            "pixel_x": int(pixel_x),
            "pixel_y": int(pixel_y),
            "pixel_q": float(q_values[selected_local]),
            "pixel_intensity": float(intensity_values[selected_local]),
            "source_q_mean": float(np.mean(q_subset)),
            "source_q_std": float(np.std(q_subset)),
            "source_angle_deg": source_angle,
            "source_intensity_mean": float(np.mean(i_subset)),
            "source_intensity_max": float(np.max(i_subset)),
            "effective_pixel_count": effective_pixels,
            "effective_sample_size": effective_pixels,
            "n_eff": effective_pixels,
            "max_contribution_fraction": max_fraction,
            "positive_excess_pixel_count": int(np.count_nonzero(positive_excess > 0.0)),
            "effective_support_bin_range": [int(neighbourhood_left), int(neighbourhood_right)],
            "representative_pixel": {"pixel_x": int(pixel_x), "pixel_y": int(pixel_y)},
        }
    )
    return candidate


def _detect_candidates(
    *,
    q_edges: np.ndarray,
    q_centers: np.ndarray,
    raw_mean: np.ndarray,
    raw_count: np.ndarray,
    coverage: np.ndarray,
    smooth: np.ndarray,
    options: Mapping[str, Any],
    q_values: np.ndarray,
    intensity_values: np.ndarray,
    qx_values: np.ndarray,
    qy_values: np.ndarray,
    original_indices: np.ndarray,
    image_shape: tuple[int, int],
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, str, float]:
    supported = (
        (np.asarray(raw_count, dtype=int) >= int(options["min_bin_count"]))
        & (np.asarray(coverage, dtype=float) >= float(options["min_coverage"]))
        & np.isfinite(raw_mean)
        & np.isfinite(smooth)
    )
    noise = _profile_noise(raw_mean, smooth, supported)
    finite_values = raw_mean[supported]
    if not finite_values.size:
        raw_values = raw_mean[(np.asarray(raw_count, dtype=int) > 0) & np.isfinite(raw_mean)]
        reason = "no_supported_bins_after_count_coverage_gate" if raw_values.size else "no_finite_intensity_support"
        return [], None, reason, noise

    span = max(float(np.ptp(finite_values)), _EPS)
    min_prominence = max(
        float(options["min_prominence_sigma"]) * noise,
        float(options["min_prominence_fraction"]) * span,
    )
    min_side_bins = int(options["min_side_bins"])
    min_width_bins = float(options["min_width_bins"])
    max_width_fraction = options["max_width_fraction"]
    min_effective_pixels = int(options["min_effective_pixels"])
    max_contribution_fraction = float(options["max_contribution_fraction"])
    effective_support_neighbourhood = int(options["effective_support_neighbourhood_bins"])
    min_distance = max(1, int(np.ceil(float(options["min_peak_separation_bins"]))))
    baseline = _robust_raw_baseline(finite_values)
    height_threshold = float(options["min_prominence_sigma"]) * noise
    candidates: list[dict[str, Any]] = []

    for left, right in _contiguous_runs(supported):
        segment = np.asarray(smooth[left:right], dtype=float)
        if segment.size < (2 * min_side_bins + 3):
            continue
        if find_peaks is not None:
            local_indices, properties = find_peaks(
                segment,
                prominence=min_prominence,
                distance=min_distance,
                width=min_width_bins,
            )
            prominences = np.asarray(properties.get("prominences", np.zeros(len(local_indices))), dtype=float)
            widths = np.asarray(properties.get("widths", np.ones(len(local_indices))), dtype=float)
        else:  # pragma: no cover
            local_indices = np.asarray(_manual_peak_indices(segment, min_prominence, min_distance), dtype=int)
            prominences = np.full(local_indices.size, np.nan, dtype=float)
            widths = np.ones(local_indices.size, dtype=float)
        for local_pos, prominence, width in zip(local_indices, prominences, widths):
            local_pos = int(local_pos)
            global_pos = left + local_pos
            record: dict[str, Any] = {
                "peak_bin_index": global_pos,
                "q_bin_center": float(q_centers[global_pos]),
                "q_star": float(q_centers[global_pos]),
                "intensity": float(raw_mean[global_pos]),
                "raw_intensity": float(raw_mean[global_pos]),
                "smoothed_intensity": float(smooth[global_pos]),
                "prominence": float(prominence),
                "snr": float(prominence / max(noise, _EPS)) if np.isfinite(prominence) else float("nan"),
                "prominence_snr": float(prominence / max(noise, _EPS)) if np.isfinite(prominence) else float("nan"),
                "prominence_threshold": float(min_prominence),
                "raw_baseline": float(baseline),
                "baseline_method": "10th percentile of robust-MAD-trimmed raw profile",
                "height": float(smooth[global_pos] - baseline),
                "height_threshold": height_threshold,
                "height_snr": float((smooth[global_pos] - baseline) / max(noise, _EPS)),
                "radial_fwhm_bins": float(width),
                "radial_fwhm": float(width * (q_edges[1] - q_edges[0])),
                "sampling_sigma_q": float((q_edges[1] - q_edges[0]) / np.sqrt(12.0)),
                "half_bin_resolution_q": float(0.5 * (q_edges[1] - q_edges[0])),
                "sampling_sigma_basis": "uniform-within-bin sampling resolution; sampling_resolution_not_CI",
                "selected": False,
                "status": "candidate",
                "reason": None,
            }
            _source_record(
                candidate=record,
                q_edges=q_edges,
                q_values=q_values,
                intensity_values=intensity_values,
                qx_values=qx_values,
                qy_values=qy_values,
                original_indices=original_indices,
                image_shape=image_shape,
                baseline=baseline,
                neighbourhood_bins=effective_support_neighbourhood,
            )
            if local_pos < min_side_bins or local_pos >= segment.size - min_side_bins:
                record["status"] = "rejected"
                record["reason"] = "insufficient_two_sided_support"
            else:
                left_return, right_return = _side_background_return(
                    segment,
                    local_pos,
                    baseline,
                    fraction=float(options["side_return_fraction"]),
                )
                record["left_background_return"] = left_return
                record["right_background_return"] = right_return
                if not (left_return and right_return):
                    record["status"] = "rejected"
                    record["reason"] = "insufficient_two_sided_background_return"
                else:
                    top_width, flat_ratio = _flat_top_metrics(
                        segment,
                        local_pos,
                        baseline,
                        float(width),
                        float(options["flat_top_fraction"]),
                    )
                    record["flat_top_bins"] = top_width
                    record["flat_top_ratio"] = flat_ratio
                    record["flat_top_fraction"] = float(options["flat_top_fraction"])
                    if (
                        top_width >= int(options["flat_top_min_bins"])
                        and flat_ratio >= float(options["flat_top_ratio_threshold"])
                    ):
                        record["status"] = "rejected"
                        record["reason"] = "flat_top_unresolved_peak"
                    elif max_width_fraction is not None and width > max_width_fraction * segment.size:
                        # Optional user opt-in only.  The default is None:
                        # broad but returned-to-background profiles remain
                        # measurable; width is otherwise a diagnostic.
                        record["status"] = "rejected"
                        record["reason"] = "width_gate_rejected_by_explicit_option"
            if record["status"] == "candidate":
                q_peak, smooth_peak = _quadratic_localisation(q_centers, smooth, global_pos)
                record["q_star"] = float(q_peak)
                record["smoothed_intensity"] = float(smooth_peak)
                record["height"] = float(smooth_peak - baseline)
                record["height_snr"] = float((smooth_peak - baseline) / max(noise, _EPS))
                if not record.get("source_pixel_count"):
                    record["status"] = "rejected"
                    record["reason"] = "no_source_pixels_in_peak_bin"
                elif float(record["height"]) < height_threshold:
                    record["status"] = "rejected"
                    record["reason"] = "height_below_baseline_noise"
                elif (
                    float(record.get("n_eff", 0.0)) < min_effective_pixels
                    or float(record.get("max_contribution_fraction", np.inf)) > max_contribution_fraction
                ):
                    record["status"] = "rejected"
                    record["reason"] = (
                        "single_pixel_dominated_support"
                        if float(record.get("max_contribution_fraction", np.inf)) > max_contribution_fraction
                        else "insufficient_effective_pixel_support"
                    )
            candidates.append(record)

    usable = [candidate for candidate in candidates if candidate["status"] == "candidate"]
    if not usable:
        rejected_reasons = [str(candidate.get("reason")) for candidate in candidates]
        if rejected_reasons and all(reason == "flat_top_unresolved_peak" for reason in rejected_reasons):
            return candidates, None, "flat_top_unresolved_peak", noise
        if rejected_reasons and all(reason == "width_gate_rejected_by_explicit_option" for reason in rejected_reasons):
            return candidates, None, "width_gate_rejected_by_explicit_option", noise
        if rejected_reasons and all(reason == "single_pixel_dominated_support" for reason in rejected_reasons):
            return candidates, None, "single_pixel_dominated_support", noise
        if rejected_reasons and all(reason == "insufficient_effective_pixel_support" for reason in rejected_reasons):
            return candidates, None, "insufficient_effective_pixel_support", noise
        if rejected_reasons and all(reason == "height_below_baseline_noise" for reason in rejected_reasons):
            return candidates, None, "height_below_baseline_noise", noise
        if rejected_reasons and all(reason == "insufficient_two_sided_background_return" for reason in rejected_reasons):
            return candidates, None, "insufficient_two_sided_background_return", noise
        if finite_values.size <= 4 * min_side_bins:
            reason = "insufficient_two_sided_support"
        else:
            first = int(np.flatnonzero(supported)[0])
            last = int(np.flatnonzero(supported)[-1])
            max_index = int(np.nanargmax(np.where(supported, smooth, -np.inf)))
            if max_index == first:
                reason = "low_q_boundary_without_two_sided_support"
            elif max_index == last:
                reason = "q_window_boundary_without_two_sided_support"
            elif span <= float(options["min_prominence_sigma"]) * noise:
                reason = "background_or_noise_below_prominence"
            else:
                differences: list[np.ndarray] = []
                for left, right in _contiguous_runs(supported):
                    if right - left >= 2:
                        differences.append(np.diff(smooth[left:right]))
                monotonic = False
                if differences:
                    delta = np.concatenate(differences)
                    tolerance = max(_EPS, 0.01 * span)
                    monotonic = bool(np.all(delta >= -tolerance) or np.all(delta <= tolerance))
                reason = "monotonic_profile_no_interior_peak" if monotonic else "no_supported_local_peak"
        return candidates, None, reason, noise

    usable.sort(key=lambda candidate: float(candidate["prominence"]), reverse=True)
    selected: dict[str, Any] | None = None
    reason = "selected"
    if len(usable) > 1:
        strongest, second = usable[0], usable[1]
        ratio = float(strongest["prominence"]) / max(float(second["prominence"]), _EPS)
        strongest["competitor_prominence_ratio"] = ratio
        if ratio < float(options["selection_prominence_ratio"]):
            reason = "ambiguous_multiple_peaks"
            for candidate in usable:
                candidate["status"] = "ambiguous"
                candidate["reason"] = "no_clear_prominence_winner"
            return candidates, None, reason, noise
        selected = strongest
        for candidate in usable[1:]:
            candidate["status"] = "candidate_not_selected"
            candidate["reason"] = "weaker_competitor"
    else:
        selected = usable[0]
    if selected is not None:
        selected["selected"] = True
        selected["status"] = "selected"
        selected["reason"] = "selected_dominant_local_peak"
    return candidates, selected, reason, noise


def _resolve_settings(options: Mapping[str, Any], q_range: float, q_step: float | None) -> dict[str, Any]:
    width = _finite_float(_option(options, "sector_width_deg", 10.0, "width_deg", "sector_width"), "sector_width_deg", minimum=_EPS)
    step = _finite_float(_option(options, "sector_step_deg", 5.0, "step_deg", "sector_step"), "sector_step_deg", minimum=_EPS)
    if width > 360.0 or step > 360.0:
        raise ValueError("sector_width_deg and sector_step_deg must not exceed 360")
    smoothing = _finite_float(_option(options, "smoothing_sigma_bins", 1.0, "smoothing_sigma"), "smoothing_sigma_bins", minimum=0.0)
    min_prominence_sigma = _finite_float(_option(options, "min_prominence_sigma", 4.0, "prominence_sigma"), "min_prominence_sigma", minimum=0.0)
    min_prominence_fraction = _finite_float(_option(options, "min_prominence_fraction", 0.02, "prominence_fraction"), "min_prominence_fraction", minimum=0.0)
    min_side_bins = _strict_int(_option(options, "min_side_bins", 2), "min_side_bins", minimum=1)
    min_bin_count = _strict_int(_option(options, "min_bin_count", 2), "min_bin_count", minimum=1)
    min_coverage = _finite_float(_option(options, "min_coverage", 0.5), "min_coverage", minimum=0.0)
    if min_coverage > 1.0:
        raise ValueError("min_coverage must be <= 1")
    min_width_bins = _finite_float(_option(options, "min_width_bins", 0.8), "min_width_bins", minimum=0.0)
    max_width_raw = _option(options, "max_width_fraction", None)
    max_width_fraction = None if max_width_raw is None else _finite_float(max_width_raw, "max_width_fraction", minimum=0.0)
    if max_width_fraction is not None and max_width_fraction > 1.0:
        raise ValueError("max_width_fraction must be <= 1")
    flat_top_min_bins = _strict_int(_option(options, "flat_top_min_bins", 3), "flat_top_min_bins", minimum=1)
    flat_top_ratio_threshold = _finite_float(
        _option(options, "flat_top_ratio_threshold", 0.45),
        "flat_top_ratio_threshold",
        minimum=0.0,
    )
    flat_top_fraction = _finite_float(
        _option(options, "flat_top_fraction", 0.98),
        "flat_top_fraction",
        minimum=0.0,
    )
    if flat_top_fraction > 1.0:
        raise ValueError("flat_top_fraction must be <= 1")
    side_return_fraction = _finite_float(_option(options, "side_return_fraction", 0.20), "side_return_fraction", minimum=0.0)
    if side_return_fraction >= 1.0:
        raise ValueError("side_return_fraction must be < 1")
    min_peak_separation_bins = _finite_float(_option(options, "min_peak_separation_bins", 2.0), "min_peak_separation_bins", minimum=1.0)
    selection_ratio = _finite_float(_option(options, "selection_prominence_ratio", 1.35), "selection_prominence_ratio", minimum=1.0)
    min_effective_pixels = _strict_int(
        _option(options, "min_effective_pixels", 2),
        "min_effective_pixels",
        minimum=2,
    )
    max_contribution_fraction = _finite_float(
        _option(options, "max_contribution_fraction", 0.75),
        "max_contribution_fraction",
        minimum=0.0,
    )
    if max_contribution_fraction > 1.0:
        raise ValueError("max_contribution_fraction must be <= 1")
    effective_support_neighbourhood = _strict_int(
        _option(options, "effective_support_neighbourhood_bins", 1),
        "effective_support_neighbourhood_bins",
        minimum=0,
    )
    requested_bins_raw = _option(options, "radial_bins", None, "n_radial_bins")
    requested_bins = None if requested_bins_raw is None else _strict_int(requested_bins_raw, "radial_bins", minimum=1)
    requested_max = _strict_int(_option(options, "max_radial_bins", 192), "max_radial_bins", minimum=1)
    max_radial_bins = min(requested_max, 192)
    if q_step is not None and np.isfinite(q_step) and q_step > 0.0:
        resolution_limited = max(1, int(np.floor(q_range / q_step + 1e-12)))
        q_step_source = "ridge_inputs.representative_q_step"
    else:
        resolution_limited = None
        q_step_source = "unavailable; explicit/default radial-bin cap only"
    radial_limit = max_radial_bins if resolution_limited is None else min(max_radial_bins, resolution_limited)
    effective_bins = radial_limit if requested_bins is None else min(requested_bins, radial_limit)
    effective_bins = max(1, int(effective_bins))
    angle_offset = _finite_float(_option(options, "angle_offset_deg", 0.0), "angle_offset_deg")
    return {
        "sector_width_deg": width,
        "sector_step_deg": step,
        "angle_offset_deg": angle_offset,
        "smoothing_sigma_bins": smoothing,
        "min_prominence_sigma": min_prominence_sigma,
        "min_prominence_fraction": min_prominence_fraction,
        "min_side_bins": min_side_bins,
        "min_bin_count": min_bin_count,
        "min_coverage": min_coverage,
        "min_width_bins": min_width_bins,
        "max_width_fraction": max_width_fraction,
        "flat_top_min_bins": flat_top_min_bins,
        "flat_top_fraction": flat_top_fraction,
        "flat_top_ratio_threshold": flat_top_ratio_threshold,
        "side_return_fraction": side_return_fraction,
        "min_peak_separation_bins": min_peak_separation_bins,
        "selection_prominence_ratio": selection_ratio,
        "min_effective_pixels": min_effective_pixels,
        "max_contribution_fraction": max_contribution_fraction,
        "effective_support_neighbourhood_bins": effective_support_neighbourhood,
        "requested_radial_bins": requested_bins,
        "requested_max_radial_bins": requested_max,
        "max_radial_bins": max_radial_bins,
        "effective_radial_bins": effective_bins,
        "q_resolution_limited_bins": resolution_limited,
        "representative_q_step": None if q_step is None else float(q_step),
        "q_step_source": q_step_source,
    }


def _local_representative_q_step(
    qx: np.ndarray,
    qy: np.ndarray,
    q: np.ndarray,
    finite_geometry: np.ndarray,
    q_min: float,
    q_max: float,
    *,
    q_is_supplied: bool,
) -> tuple[float | None, str, dict[str, Any]]:
    """Estimate a conservative local q step from finite neighbor pairs.

    The old bbox-median estimate could hide a coarse detector region behind a
    much larger fine region.  Here the q-window is used only to select a local
    bbox; within that bbox, each horizontal/vertical adjacent pair is kept
    when both endpoints have finite geometry and at least one endpoint lies in
    the requested q window.  The maximum finite qx/qy vector step and, when a
    q array was explicitly supplied, the maximum radial-q difference are
    combined as a conservative upper bound.  The maximum is intentionally
    conservative: a bin is never presented as finer than the coarsest local
    detector neighbor represented in the q-window domain.
    """

    window = finite_geometry & (q >= q_min) & (q <= q_max)
    locations = np.argwhere(window)
    if locations.size == 0:
        return None, "no finite q-map pixels in q window", {
            "neighbor_pairs": 0,
            "vector_pair_count": 0,
            "radial_pair_count": 0,
            "vector_step_max": None,
            "radial_step_max": None,
            "statistic": "maximum finite local adjacent step",
        }
    row_min, col_min = np.min(locations, axis=0).astype(int)
    row_max, col_max = np.max(locations, axis=0).astype(int)
    row_min = max(0, row_min - 1)
    col_min = max(0, col_min - 1)
    row_max = min(qx.shape[0] - 1, row_max + 1)
    col_max = min(qx.shape[1] - 1, col_max + 1)
    qx_local = qx[row_min : row_max + 1, col_min : col_max + 1]
    qy_local = qy[row_min : row_max + 1, col_min : col_max + 1]
    q_local = q[row_min : row_max + 1, col_min : col_max + 1]
    finite_local = finite_geometry[row_min : row_max + 1, col_min : col_max + 1]
    window_local = window[row_min : row_max + 1, col_min : col_max + 1]
    vector_steps: list[np.ndarray] = []
    radial_steps: list[np.ndarray] = []
    pair_count = 0
    radial_pair_count = 0
    for axis in (0, 1):
        slicer_a: list[slice] = [slice(None), slice(None)]
        slicer_b: list[slice] = [slice(None), slice(None)]
        slicer_a[axis] = slice(0, -1)
        slicer_b[axis] = slice(1, None)
        finite_pair = finite_local[tuple(slicer_a)] & finite_local[tuple(slicer_b)]
        in_domain_pair = window_local[tuple(slicer_a)] | window_local[tuple(slicer_b)]
        pair = finite_pair & in_domain_pair
        if not np.any(pair):
            continue
        qx_a, qx_b = qx_local[tuple(slicer_a)], qx_local[tuple(slicer_b)]
        qy_a, qy_b = qy_local[tuple(slicer_a)], qy_local[tuple(slicer_b)]
        vector = np.hypot(qx_b - qx_a, qy_b - qy_a)[pair]
        vector = vector[np.isfinite(vector) & (vector > _EPS)]
        if vector.size:
            vector_steps.append(np.asarray(vector, dtype=float))
            pair_count += int(vector.size)
        # q is always available after canonicalisation.  A radial difference
        # is explicitly reported as a second evidence family when qmap
        # supplied a q/radius field; for derived q, the vector step remains
        # the calibration-resolution family.
        if q_is_supplied:
            q_a, q_b = q_local[tuple(slicer_a)], q_local[tuple(slicer_b)]
            radial = np.abs(q_b - q_a)[pair]
            radial = radial[np.isfinite(radial) & (radial > _EPS)]
            if radial.size:
                radial_steps.append(np.asarray(radial, dtype=float))
                radial_pair_count += int(radial.size)
    details = {
        "neighbor_pairs": pair_count,
        "vector_pair_count": pair_count,
        "radial_pair_count": radial_pair_count,
        "vector_step_max": None,
        "radial_step_max": None,
        "statistic": "maximum finite local adjacent step",
    }
    if vector_steps:
        details["vector_step_max"] = float(np.max(np.concatenate(vector_steps)))
    if radial_steps:
        details["radial_step_max"] = float(np.max(np.concatenate(radial_steps)))
    finite_maxima = [
        value
        for value in (details["vector_step_max"], details["radial_step_max"])
        if value is not None and np.isfinite(value) and value > _EPS
    ]
    local_q_step = max(finite_maxima) if finite_maxima else None
    source = "finite q-window adjacent geometry pairs; conservative maximum qx/qy vector step"
    if q_is_supplied:
        source += " and supplied radial-q difference"
    if local_q_step is None:
        source = "unavailable: no finite q-window adjacent geometry step"
    return (
        local_q_step,
        source,
        details,
    )


def measure_sector_peaks(
    image: Any,
    qmap: Any,
    q_window: Any,
    *,
    mask: Any = None,
    options: Any = None,
    cancel_event: Any = None,
) -> dict[str, Any]:
    """Measure radial intensity profiles and dominant local peaks by angle.

    Parameters
    ----------
    image, qmap:
        The same image/q-map seam used by the main analysis.  ``qmap`` must
        expose qx and qy; q is taken from the map when supplied, otherwise it
        is derived by the shared canonical adapter.
    q_window:
        Inclusive physical or declared q-unit interval.  The unit is copied
        from qmap metadata and never inferred here.
    mask:
        Detector-style invalid mask: ``True`` excludes a pixel from measured
        sums and means.  Geometry coverage still counts finite q-map pixels.
    options:
        Optional mapping.  Defaults are 10 degree sector width, 5 degree
        sector step, mask-aware smoothing sigma 1 bin, and a maximum of 192
        radial bins subject to the measured pixel q step.

    Returns
    -------
    dict
        A mapping containing all sector profiles, candidate diagnostics,
        selected measured peaks (or ``None``), and explicit sampling/overlap
        metadata.  ``sampling_sigma_q`` is a bin-resolution indicator rather
        than a statistical confidence interval.
    """

    raise_if_cancelled(cancel_event, "sector-peaks:inputs")
    q_min, q_max = _q_window(q_window)
    data, qx, qy, q, invalid = canonical_inputs(image, qmap, mask=mask)
    q_unit = _q_unit(qmap)
    shape = tuple(int(value) for value in data.shape)

    angle = np.mod(np.degrees(np.arctan2(qy, qx)), 360.0)
    finite_geometry = np.isfinite(qx) & np.isfinite(qy) & np.isfinite(q) & np.isfinite(angle)
    q_is_supplied = array_field(qmap, Q_ALIASES) is not None
    q_step, q_step_source, q_step_details = _local_representative_q_step(
        qx,
        qy,
        q,
        finite_geometry,
        q_min,
        q_max,
        q_is_supplied=q_is_supplied,
    )
    settings = _resolve_settings(_options_mapping(options), q_max - q_min, q_step)
    settings["q_step_source"] = q_step_source
    settings["q_step_details"] = q_step_details
    in_window = finite_geometry & (q >= q_min) & (q <= q_max)
    window_indices = np.flatnonzero(in_window.ravel())
    q_values = np.asarray(q.ravel()[window_indices], dtype=float)
    qx_values = np.asarray(qx.ravel()[window_indices], dtype=float)
    qy_values = np.asarray(qy.ravel()[window_indices], dtype=float)
    angle_values = np.asarray(angle.ravel()[window_indices], dtype=float)
    image_values = np.asarray(data.ravel()[window_indices], dtype=float)
    invalid_values = np.asarray(invalid.ravel()[window_indices], dtype=bool)
    finite_intensity = np.isfinite(image_values) & ~invalid_values

    n_bins = int(settings["effective_radial_bins"])
    q_edges = np.linspace(q_min, q_max, n_bins + 1, dtype=float)
    q_centers = 0.5 * (q_edges[:-1] + q_edges[1:])
    q_bin_width = float(q_edges[1] - q_edges[0]) if n_bins else float("nan")
    bin_values = np.searchsorted(q_edges, q_values, side="right") - 1
    bin_values = np.clip(bin_values, 0, n_bins - 1).astype(np.int64, copy=False)

    width = float(settings["sector_width_deg"])
    step = float(settings["sector_step_deg"])
    offset = float(settings["angle_offset_deg"])
    centres = np.arange(0.0, 360.0, step, dtype=float) + offset
    centres = np.mod(centres, 360.0)
    # Avoid accidental duplicate sectors when a floating step lands on 360.
    unique_centres, unique_indices = np.unique(np.round(centres, decimals=12), return_index=True)
    centres = centres[np.sort(unique_indices)]

    sectors: list[dict[str, Any]] = []
    for sector_index, centre in enumerate(centres):
        raise_if_cancelled(cancel_event, f"sector-peaks:sector-{sector_index}")
        geometric = _angle_delta_deg(angle_values, float(centre)) <= 0.5 * width + 1e-12
        geometry_q = bin_values[geometric]
        geometry_count = np.bincount(geometry_q, minlength=n_bins).astype(np.int64)
        measured = geometric & finite_intensity
        measured_q = bin_values[measured]
        measured_i = image_values[measured]
        raw_sum = np.bincount(measured_q, weights=measured_i, minlength=n_bins).astype(float)
        raw_count = np.bincount(measured_q, minlength=n_bins).astype(np.int64)
        raw_mean = np.full(n_bins, np.nan, dtype=float)
        np.divide(raw_sum, raw_count, out=raw_mean, where=raw_count > 0)
        coverage = np.divide(raw_count, geometry_count, out=np.zeros(n_bins, dtype=float), where=geometry_count > 0)
        supported_bins = (
            (raw_count >= int(settings["min_bin_count"]))
            & (coverage >= float(settings["min_coverage"]))
            & np.isfinite(raw_mean)
        )
        smooth = _smooth_supported(raw_mean, supported_bins, float(settings["smoothing_sigma_bins"]))

        candidates, selected, reason, noise = _detect_candidates(
            q_edges=q_edges,
            q_centers=q_centers,
            raw_mean=raw_mean,
            raw_count=raw_count,
            coverage=coverage,
            smooth=smooth,
            options=settings,
            q_values=q_values[measured],
            intensity_values=image_values[measured],
            qx_values=qx_values[measured],
            qy_values=qy_values[measured],
            original_indices=window_indices[measured],
            image_shape=shape,
        )
        if selected is not None:
            selected = dict(selected)
            selected["selected"] = True
        geometry_total = int(np.sum(geometry_count))
        measured_total = int(np.sum(raw_count))
        sector_coverage = float(measured_total / geometry_total) if geometry_total else 0.0
        status = "selected" if selected is not None else ("ambiguous" if reason == "ambiguous_multiple_peaks" else "no_peak")
        sector = {
            "sector_index": int(sector_index),
            "angle_deg": float(centre),
            "center_angle_deg": float(centre),
            "start_angle_deg": float((centre - 0.5 * width) % 360.0),
            "end_angle_deg": float((centre + 0.5 * width) % 360.0),
            "width_deg": width,
            "q": q_centers.copy(),
            "q_edges": q_edges.copy(),
            "q_centers": q_centers.copy(),
            "raw": {"mean": raw_mean, "sum": raw_sum, "count": raw_count},
            "raw_mean": raw_mean,
            "raw_sum": raw_sum,
            "raw_count": raw_count,
            "counts": raw_count,
            "intensity": raw_mean,
            "smoothed": smooth,
            "smoothed_intensity": smooth,
            "geometry_count": geometry_count,
            "geometric_counts": geometry_count,
            "coverage": coverage,
            "geometry_coverage": coverage,
            "supported_bin_mask": supported_bins,
            "geometry_total_count": geometry_total,
            "measured_total_count": measured_total,
            "sector_coverage": sector_coverage,
            "profile_noise_sigma": float(noise),
            "noise_provenance": {
                "method": "robust MAD of raw-minus-within-run-smooth and adjacent raw-bin differences",
                "window": "each contiguous supported radial run in this sector; masked gaps excluded",
                "experimental_sigma": False,
                "interpretation": "profile-local noise scale for detection gates, not a measurement uncertainty",
            },
            "candidates": candidates,
            "selected_peak": selected,
            "status": status,
            "reason": reason,
        }
        sectors.append(sector)

    overlap_fraction = max(0.0, min(1.0, 1.0 - step / width))
    result = {
        "schema_version": SECTOR_PEAKS_SCHEMA_VERSION,
        "method_version": SECTOR_PEAKS_METHOD_VERSION,
        "interpretation": "unindexed dominant radial peak measured from observed sector-integrated intensity; no reflection assignment or four-lobe completion",
        "q_unit": str(q_unit or "unknown"),
        "angle_unit": "degree",
        "intensity_unit": "input intensity units",
        "q_window": [float(q_min), float(q_max)],
        "q_edges": q_edges,
        "q_centers": q_centers,
        "sector_centers_deg": np.asarray(centres, dtype=float),
        "settings": settings,
        "effective_settings": settings,
        "units": {"q": str(q_unit or "unknown"), "angle": "degree", "intensity": "input intensity units"},
        "sampling": {
            "representative_q_step": None if q_step is None else float(q_step),
            "radial_q_supplied": bool(q_is_supplied),
            "q_step_details": q_step_details,
            "effective_radial_bin_width": q_bin_width,
            "sampling_sigma_q": q_bin_width / np.sqrt(12.0),
            "half_bin_resolution_q": 0.5 * q_bin_width,
            "sigma_basis": "uniform-within-bin sampling resolution; sampling_resolution_not_CI",
            "precision_statement": "radial bins are not finer than the local representative detector q step when that step is available",
        },
        "noise_provenance": {
            "method": "robust MAD of raw-minus-within-run-smooth and adjacent raw-bin differences",
            "window": "each contiguous supported radial run per sector",
            "experimental_sigma": False,
            "height_gate": "smoothed local peak minus robust raw-profile baseline must exceed min_prominence_sigma times this scale",
            "baseline": "10th percentile of a robust-MAD-trimmed raw profile; smoothing valleys are not used as the baseline",
        },
        "sector_overlap": {
            "width_deg": width,
            "step_deg": step,
            "overlap_fraction": overlap_fraction,
            "overlapping": bool(width > step),
            "correlation_statement": "overlapping sectors share detector pixels and are correlated; sector-to-sector peaks are not independent samples",
        },
        "coverage_definition": "coverage is measured finite intensity count divided by finite geometry count per q bin; masked or non-finite intensity is excluded only from the numerator",
        "input": {
            "shape": list(shape),
            "geometry_pixel_count_in_q_window": int(window_indices.size),
            "measured_pixel_count_in_q_window": int(np.count_nonzero(finite_intensity)),
            "mask_true_is_excluded": True,
            "input_arrays_mutated": False,
        },
        "sectors": sectors,
    }
    raise_if_cancelled(cancel_event, "sector-peaks:complete")
    return result


__all__ = [
    "SECTOR_PEAKS_METHOD_VERSION",
    "SECTOR_PEAKS_SCHEMA_VERSION",
    "measure_sector_peaks",
]
