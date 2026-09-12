"""Measured strongest-pixel and supported angular-lobe diagnostics.

The raw maximum is deliberately kept separate from the smoothed, supported
angular lobes. Neither record identifies a crystallographic reflection or
establishes a structural interpretation.
"""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.signal import find_peaks, peak_prominences, peak_widths

from .cancellation import raise_if_cancelled
from .serialization import json_safe


PEAK_LANDMARK_METHOD_VERSION = "supported-angular-lobes-v2"
PEAK_LANDMARK_SCHEMA_VERSION = "peak-landmarks-v2"

_DEFAULT_OPTIONS: dict[str, Any] = {
    "smoothing_sigma_px": 1.2,
    "minimum_smoothing_support": 0.70,
    "angular_bins": 180,
    "angular_profile_sigma_bins": 1.0,
    "minimum_angular_coverage_fraction": 0.25,
    "minimum_angular_bin_pixels": 1,
    "minimum_prominence_sigma": 4.0,
    "minimum_prominence": 0.0,
    "minimum_peak_separation_deg": 24.0,
    "minimum_lobe_width_deg": 4.0,
    "raw_support_sigma": 2.0,
    "minimum_lobe_support_pixels": 5,
    "maximum_lobes": 4,
    "radial_bins": 96,
    # A six-bin smoothing scale mixed broad (e.g. 22-degree sigma) lobes into noise.
    "angular_noise_smoothing_sigma_bins": 1.0,
}


def _numeric_array(value: Any, name: str) -> tuple[np.ndarray, np.ndarray]:
    try:
        array = np.asanyarray(value)
    except Exception as exc:
        raise TypeError(f"{name} must be a numeric array") from exc
    raw = np.asarray(np.ma.getdata(array))
    if raw.dtype.kind not in "iuf":
        raise TypeError(f"{name} must contain real numeric values")
    return raw, np.asarray(np.ma.getmaskarray(array), dtype=bool)


def _parse_window(value: Any, name: str) -> tuple[float, float] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        low = value.get("q_min", value.get("min", value.get("low", value.get("start"))))
        high = value.get("q_max", value.get("max", value.get("high", value.get("stop"))))
    else:
        if isinstance(value, (str, bytes)):
            raise TypeError(f"{name} must contain two finite q values")
        try:
            if len(value) != 2:
                raise ValueError
            low, high = value[0], value[1]
        except (TypeError, KeyError, IndexError, ValueError) as exc:
            raise TypeError(f"{name} must contain two finite q values") from exc
    try:
        pair = float(low), float(high)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain two finite q values") from exc
    if not all(math.isfinite(item) for item in pair) or pair[0] >= pair[1]:
        raise ValueError(f"{name} must be finite and strictly increasing")
    return pair


def _options_from(value: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    if value is None:
        supplied: Mapping[str, Any] = {}
    elif isinstance(value, Mapping):
        supplied = value
    else:
        raise TypeError("options must be a mapping or None")
    out = dict(_DEFAULT_OPTIONS)
    ignored: dict[str, Any] = {}
    for key, item in supplied.items():
        if key in out:
            out[key] = item
        else:
            ignored[str(key)] = json_safe(item)

    for key in (
        "smoothing_sigma_px",
        "minimum_smoothing_support",
        "angular_profile_sigma_bins",
        "minimum_angular_coverage_fraction",
        "minimum_prominence_sigma",
        "minimum_prominence",
        "minimum_peak_separation_deg",
        "minimum_lobe_width_deg",
        "raw_support_sigma",
        "angular_noise_smoothing_sigma_bins",
    ):
        try:
            out[key] = float(out[key])
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"options[{key!r}] must be numeric") from exc
        if not math.isfinite(out[key]):
            raise ValueError(f"options[{key!r}] must be finite")

    for key in (
        "angular_bins",
        "minimum_angular_bin_pixels",
        "minimum_lobe_support_pixels",
        "maximum_lobes",
        "radial_bins",
    ):
        item = out[key]
        if isinstance(item, bool):
            raise ValueError(f"options[{key!r}] must be an integer")
        try:
            cast = int(item)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"options[{key!r}] must be an integer") from exc
        if cast != item:
            raise ValueError(f"options[{key!r}] must be an integer")
        out[key] = cast

    if out["smoothing_sigma_px"] < 0.0:
        raise ValueError("smoothing_sigma_px must be non-negative")
    if not 0.0 < out["minimum_smoothing_support"] <= 1.0:
        raise ValueError("minimum_smoothing_support must be in (0, 1]")
    if out["angular_bins"] < 36 or out["angular_bins"] > 1440:
        raise ValueError("angular_bins must be between 36 and 1440")
    if out["angular_profile_sigma_bins"] < 0.0:
        raise ValueError("angular_profile_sigma_bins must be non-negative")
    if not 0.0 <= out["minimum_angular_coverage_fraction"] <= 1.0:
        raise ValueError("minimum_angular_coverage_fraction must be in [0, 1]")
    if out["minimum_angular_bin_pixels"] < 1:
        raise ValueError("minimum_angular_bin_pixels must be at least 1")
    if out["minimum_prominence_sigma"] < 0.0 or out["minimum_prominence"] < 0.0:
        raise ValueError("prominence thresholds must be non-negative")
    if not 0.0 <= out["minimum_peak_separation_deg"] < 180.0:
        raise ValueError("minimum_peak_separation_deg must be in [0, 180)")
    if out["minimum_lobe_width_deg"] < 0.0:
        raise ValueError("minimum_lobe_width_deg must be non-negative")
    if out["raw_support_sigma"] < 0.0:
        raise ValueError("raw_support_sigma must be non-negative")
    if out["minimum_lobe_support_pixels"] < 1:
        raise ValueError("minimum_lobe_support_pixels must be at least 1")
    if not 1 <= out["maximum_lobes"] <= 4:
        raise ValueError("maximum_lobes must be between 1 and 4")
    if not 16 <= out["radial_bins"] <= 512:
        raise ValueError("radial_bins must be between 16 and 512")
    if out["angular_noise_smoothing_sigma_bins"] <= 0.0:
        raise ValueError("angular_noise_smoothing_sigma_bins must be positive")
    return out, ignored


def _robust_sigma(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0
    median = float(np.median(finite))
    mad_sigma = 1.4826 * float(np.median(np.abs(finite - median)))
    if math.isfinite(mad_sigma) and mad_sigma > 0.0:
        return mad_sigma
    q25, q75 = np.percentile(finite, [25.0, 75.0])
    iqr_sigma = float((q75 - q25) / 1.349)
    if math.isfinite(iqr_sigma) and iqr_sigma > 0.0:
        return iqr_sigma
    std = float(np.std(finite))
    return std if math.isfinite(std) and std > 0.0 else 0.0


def _pixel_q_resolution(
    qx: np.ndarray, qy: np.ndarray, valid: np.ndarray
) -> float:
    differences: list[np.ndarray] = []
    for axis in (0, 1):
        if qx.shape[axis] <= 1:
            continue
        sl_first = [slice(None), slice(None)]
        sl_second = [slice(None), slice(None)]
        sl_first[axis] = slice(0, -1)
        sl_second[axis] = slice(1, None)
        first = tuple(sl_first)
        second = tuple(sl_second)
        pair_valid = valid[first] & valid[second]
        if not np.any(pair_valid):
            continue
        delta = np.hypot(qx[second] - qx[first], qy[second] - qy[first])
        selected = delta[pair_valid]
        selected = selected[np.isfinite(selected) & (selected > 0.0)]
        if selected.size:
            differences.append(selected)
    if not differences:
        return 0.0
    values = np.concatenate(differences)
    if values.size > 200_000:
        stride = int(math.ceil(values.size / 200_000))
        values = values[::stride]
    return float(np.median(values))


def _smooth_masked(
    values: np.ndarray,
    valid: np.ndarray,
    sigma: float,
    min_support: float,
) -> tuple[np.ndarray, np.ndarray]:
    if sigma <= 0.0:
        weights = valid.astype(np.float64, copy=False)
        smoothed = np.where(valid, values, np.nan).astype(np.float64, copy=False)
        return smoothed, weights
    weights = gaussian_filter(
        valid.astype(np.float64), sigma=sigma, mode="constant", cval=0.0
    )
    numerator = gaussian_filter(
        np.where(valid, values, 0.0), sigma=sigma, mode="constant", cval=0.0
    )
    smoothed = np.full(values.shape, np.nan, dtype=np.float64)
    supported = weights >= min_support
    np.divide(numerator, weights, out=smoothed, where=supported)
    return smoothed, weights


def _window_mask(
    radius: np.ndarray, finite_q: np.ndarray, window: tuple[float, float] | None
) -> np.ndarray:
    if window is None:
        return finite_q.copy()
    return finite_q & (radius >= window[0]) & (radius <= window[1])


def _circular_distance_deg(first: float, second: np.ndarray | float) -> np.ndarray:
    return np.abs((np.asarray(second, dtype=np.float64) - first + 180.0) % 360.0 - 180.0)


def _interval_mask_deg(
    angle: np.ndarray, left: float, right: float, *, all_angles: bool = False
) -> np.ndarray:
    if all_angles:
        return np.ones(angle.shape, dtype=bool)
    span = (right - left) % 360.0
    distance = (angle - left) % 360.0
    return distance <= span + 1e-10


def _window_edge_flags(
    radius: float,
    q_window: tuple[float, float] | None,
    signal_window: tuple[float, float] | None,
    q_resolution: float,
) -> list[str]:
    flags: list[str] = []
    tolerance = max(q_resolution * 0.5, np.finfo(np.float64).eps * max(abs(radius), 1.0) * 8.0)
    if q_window is not None and min(abs(radius - q_window[0]), abs(radius - q_window[1])) <= tolerance:
        flags.append("user_q_window_boundary")
    if signal_window is not None and min(
        abs(radius - signal_window[0]), abs(radius - signal_window[1])
    ) <= tolerance:
        flags.append("signal_q_window_boundary")
    return flags


def _profile_arrays(
    *,
    radius: np.ndarray,
    angle_deg: np.ndarray,
    raw: np.ndarray,
    smooth: np.ndarray,
    valid: np.ndarray,
    geometry: np.ndarray,
    angular_bins: int,
    radial_bins: int,
    min_coverage: float,
    min_bin_pixels: int,
) -> dict[str, dict[str, np.ndarray]]:
    angular_edges = np.linspace(0.0, 360.0, angular_bins + 1, dtype=np.float64)
    angular_centers = 0.5 * (angular_edges[:-1] + angular_edges[1:])
    safe_angle = np.where(np.isfinite(angle_deg), np.mod(angle_deg, 360.0), 0.0)
    angular_index = np.minimum(
        angular_bins - 1,
        np.floor(safe_angle / 360.0 * angular_bins).astype(np.int64),
    )
    geometry_angular = np.bincount(
        angular_index[geometry], minlength=angular_bins
    ).astype(np.int64)
    valid_angular = np.bincount(angular_index[valid], minlength=angular_bins).astype(
        np.int64
    )
    raw_sum_angular = np.bincount(
        angular_index[valid], weights=raw[valid], minlength=angular_bins
    )
    smooth_valid = valid & np.isfinite(smooth)
    smooth_angular = np.bincount(
        angular_index[smooth_valid], weights=smooth[smooth_valid], minlength=angular_bins
    )
    smooth_count_angular = np.bincount(
        angular_index[smooth_valid], minlength=angular_bins
    ).astype(np.int64)
    raw_mean_angular = np.full(angular_bins, np.nan, dtype=np.float64)
    smooth_mean_angular = np.full(angular_bins, np.nan, dtype=np.float64)
    np.divide(
        raw_sum_angular,
        valid_angular,
        out=raw_mean_angular,
        where=valid_angular > 0,
    )
    np.divide(
        smooth_angular,
        smooth_count_angular,
        out=smooth_mean_angular,
        where=smooth_count_angular > 0,
    )
    coverage_angular = np.zeros(angular_bins, dtype=np.float64)
    np.divide(
        valid_angular,
        geometry_angular,
        out=coverage_angular,
        where=geometry_angular > 0,
    )
    supported_angular = (
        (geometry_angular >= min_bin_pixels)
        & (valid_angular >= min_bin_pixels)
        & (coverage_angular >= min_coverage)
    )

    finite_radius = radius[geometry]
    if finite_radius.size:
        radial_low = float(np.min(finite_radius))
        radial_high = float(np.max(finite_radius))
    else:
        radial_low, radial_high = 0.0, 1.0
    if radial_high <= radial_low:
        radial_high = radial_low + max(abs(radial_low) * 1e-6, 1e-12)
    radial_edges = np.linspace(radial_low, radial_high, radial_bins + 1, dtype=np.float64)
    radial_centers = 0.5 * (radial_edges[:-1] + radial_edges[1:])
    radial_index = np.searchsorted(radial_edges, radius, side="right") - 1
    radial_index = np.clip(radial_index, 0, radial_bins - 1)
    geometry_radial = np.bincount(radial_index[geometry], minlength=radial_bins).astype(
        np.int64
    )
    valid_radial = np.bincount(radial_index[valid], minlength=radial_bins).astype(np.int64)
    raw_sum_radial = np.bincount(
        radial_index[valid], weights=raw[valid], minlength=radial_bins
    )
    smooth_radial_valid = valid & np.isfinite(smooth)
    smooth_sum_radial = np.bincount(
        radial_index[smooth_radial_valid],
        weights=smooth[smooth_radial_valid],
        minlength=radial_bins,
    )
    smooth_count_radial = np.bincount(
        radial_index[smooth_radial_valid], minlength=radial_bins
    ).astype(np.int64)
    raw_mean_radial = np.full(radial_bins, np.nan, dtype=np.float64)
    smooth_mean_radial = np.full(radial_bins, np.nan, dtype=np.float64)
    np.divide(raw_sum_radial, valid_radial, out=raw_mean_radial, where=valid_radial > 0)
    np.divide(
        smooth_sum_radial,
        smooth_count_radial,
        out=smooth_mean_radial,
        where=smooth_count_radial > 0,
    )
    coverage_radial = np.zeros(radial_bins, dtype=np.float64)
    np.divide(
        valid_radial,
        geometry_radial,
        out=coverage_radial,
        where=geometry_radial > 0,
    )
    return {
        "angular": {
            "angle_deg": angular_centers,
            "intensity_raw": raw_mean_angular,
            "intensity_smoothed": smooth_mean_angular,
            "valid_pixel_counts": valid_angular,
            "smoothed_pixel_counts": smooth_count_angular,
            "geometric_pixel_counts": geometry_angular,
            "coverage_fraction": coverage_angular,
            "supported": supported_angular,
        },
        "radial": {
            "q": radial_centers,
            "mean_intensity_raw": raw_mean_radial,
            "mean_intensity_smoothed": smooth_mean_radial,
            "valid_pixel_counts": valid_radial,
            "smoothed_pixel_counts": smooth_count_radial,
            "geometric_pixel_counts": geometry_radial,
            "coverage_fraction": coverage_radial,
        },
    }


def _isotropic_reference_map(
    *,
    raw: np.ndarray,
    radius: np.ndarray,
    valid: np.ndarray,
    geometry: np.ndarray,
    pixel_q_resolution: float,
    maximum_bins: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Estimate a robust radial intensity reference on the detector q map.

    The reference is estimated from observed raw pixels, interpolated at each
    pixel's actual q radius, and then passed through the same masked spatial and
    angular profile operations as the observation. This removes angular bias
    from unequal radial sampling without changing the measured profile.
    """

    reference_map = np.full(raw.shape, np.nan, dtype=np.float64)
    finite_geometry = geometry & np.isfinite(radius)
    if not np.any(finite_geometry) or not np.any(valid):
        return reference_map, {
            "estimator": "median_raw_intensity_per_radial_bin",
            "bin_count": 0,
            "populated_bin_count": 0,
            "q_bin_width": None,
            "pixel_q_resolution": float(pixel_q_resolution),
        }

    radial_low = float(np.min(radius[finite_geometry]))
    radial_high = float(np.max(radius[finite_geometry]))
    radial_span = radial_high - radial_low
    if radial_span <= 0.0:
        reference_map[finite_geometry] = float(np.median(raw[valid]))
        return reference_map, {
            "estimator": "median_raw_intensity_per_radial_bin",
            "bin_count": 1,
            "populated_bin_count": 1,
            "q_bin_width": 0.0,
            "pixel_q_resolution": float(pixel_q_resolution),
        }

    if pixel_q_resolution > 0.0 and math.isfinite(pixel_q_resolution):
        # Avoid estimating radial detail finer than the map's median pixel-q step.
        resolution_limited_bins = int(math.floor(radial_span / pixel_q_resolution))
    else:
        resolution_limited_bins = maximum_bins
    bin_count = max(4, min(int(maximum_bins), resolution_limited_bins))
    edges = np.linspace(radial_low, radial_high, bin_count + 1, dtype=np.float64)
    centers = 0.5 * (edges[:-1] + edges[1:])
    radial_index = np.searchsorted(edges, radius[valid], side="right") - 1
    radial_index = np.clip(radial_index, 0, bin_count - 1).astype(np.int32, copy=False)
    values = np.asarray(raw[valid], dtype=np.float64)
    counts = np.bincount(radial_index, minlength=bin_count)
    order = np.argsort(radial_index)
    offsets = np.concatenate(([0], np.cumsum(counts)))
    radial_reference = np.full(bin_count, np.nan, dtype=np.float64)
    for index, count in enumerate(counts):
        if count:
            start, stop = int(offsets[index]), int(offsets[index + 1])
            radial_reference[index] = float(np.median(values[order[start:stop]]))

    populated = np.isfinite(radial_reference)
    if not np.any(populated):
        return reference_map, {
            "estimator": "median_raw_intensity_per_radial_bin",
            "bin_count": int(bin_count),
            "populated_bin_count": 0,
            "q_bin_width": float(radial_span / bin_count),
            "pixel_q_resolution": float(pixel_q_resolution),
        }
    radial_reference = np.interp(
        centers,
        centers[populated],
        radial_reference[populated],
    )
    reference_map = np.interp(
        radius.reshape(-1), centers, radial_reference
    ).reshape(raw.shape)
    return reference_map, {
        "estimator": "median_raw_intensity_per_radial_bin",
        "bin_count": int(bin_count),
        "populated_bin_count": int(np.count_nonzero(populated)),
        "q_bin_width": float(radial_span / bin_count),
        "pixel_q_resolution": float(pixel_q_resolution),
    }


def _angular_candidates(
    profile: dict[str, np.ndarray],
    *,
    options: Mapping[str, Any],
    pixel_noise: float,
) -> tuple[list[dict[str, Any]], float]:
    values = np.asarray(profile["intensity_smoothed"], dtype=np.float64)
    supported = np.asarray(profile["supported"], dtype=bool) & np.isfinite(values)
    n_bins = values.size
    if n_bins == 0 or np.count_nonzero(supported) < 5:
        return [], 0.0
    bin_width = 360.0 / n_bins
    centers = np.asarray(profile["angle_deg"], dtype=np.float64)
    baseline = float(np.median(values[supported]))
    valid_values = np.where(supported, values, baseline)
    broad_sigma = float(options["angular_noise_smoothing_sigma_bins"])
    broad = gaussian_filter1d(valid_values, sigma=broad_sigma, mode="wrap")
    profile_noise = _robust_sigma((values - broad)[supported])
    count_median = max(1.0, float(np.median(profile["valid_pixel_counts"][supported])))
    noise_sigma = max(profile_noise, pixel_noise / math.sqrt(count_median))
    noise_floor = np.finfo(np.float64).eps * max(float(np.max(np.abs(values[supported]))), 1.0)
    noise_sigma = max(noise_sigma, noise_floor)
    dynamic_range = float(np.ptp(values[supported]))
    if dynamic_range <= noise_floor:
        return [], noise_sigma
    required_prominence = max(
        float(options["minimum_prominence"]),
        float(options["minimum_prominence_sigma"]) * noise_sigma,
    )
    fill_floor = baseline - max(dynamic_range, abs(baseline), 1.0) * 4.0
    filled = np.where(supported, values, fill_floor)
    extended = np.tile(filled, 3)
    repeated_peaks, _ = find_peaks(extended)
    middle = repeated_peaks[(repeated_peaks >= n_bins) & (repeated_peaks < 2 * n_bins)]
    wlen = int(
        max(
            5,
            round(2.0 * float(options["minimum_peak_separation_deg"]) / bin_width) + 1,
        )
    )
    if wlen % 2 == 0:
        wlen += 1
    wlen = min(wlen, 2 * n_bins - 1)
    if wlen % 2 == 0:
        wlen -= 1
    records: list[dict[str, Any]] = []
    for peak_extended in middle:
        index = int(peak_extended - n_bins)
        if not supported[index]:
            continue
        prominence_data = peak_prominences(
            extended, np.asarray([peak_extended]), wlen=max(3, wlen)
        )
        prominence = float(prominence_data[0][0])
        width_data = peak_widths(
            extended,
            np.asarray([peak_extended]),
            rel_height=0.5,
            prominence_data=prominence_data,
        )
        width_deg = float(width_data[0][0] * bin_width)
        angle = float(centers[index])
        coverage = float(profile["coverage_fraction"][index])
        count = int(profile["valid_pixel_counts"][index])
        reasons: list[str] = []
        if prominence < required_prominence:
            reasons.append("prominence_below_highpass_noise_threshold")
        if width_deg < float(options["minimum_lobe_width_deg"]):
            reasons.append("angular_width_below_minimum")
        if coverage < float(options["minimum_angular_coverage_fraction"]):
            reasons.append("insufficient_angular_coverage")
        if count < int(options["minimum_angular_bin_pixels"]):
            reasons.append("insufficient_angular_bin_pixels")
        support_half_width = max(width_deg * 0.75, 2.0 * bin_width)
        records.append(
            {
                "candidate_index": index,
                "angular_peak_deg": angle,
                "profile_intensity": float(values[index]),
                "angular_prominence": prominence,
                "angular_noise_sigma": noise_sigma,
                "angular_snr": prominence / noise_sigma,
                "angular_fwhm_deg": width_deg,
                "angular_coverage_fraction": coverage,
                "angular_bin_pixel_count": count,
                "support_half_width_deg": support_half_width,
                "accepted": not reasons,
                "reason": reasons[0] if reasons else None,
                "reasons": reasons,
            }
        )
    records.sort(key=lambda item: item["angular_prominence"], reverse=True)
    return records, noise_sigma


def _model_array(model: Any, expected_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray] | None:
    if model is None:
        return None
    value = model
    if isinstance(model, Mapping):
        for key in ("model", "full2d", "predicted_intensity", "intensity", "prediction"):
            if key in model:
                value = model[key]
                break
        else:
            raise ValueError(
                "model mapping must contain one of: model, full2d, predicted_intensity, intensity, prediction"
            )
    raw, mask = _numeric_array(value, "model")
    if raw.shape != expected_shape:
        raise ValueError("model must have the same 2D shape as observed")
    try:
        data = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError("model must be convertible to float64") from exc
    return data, (~mask) & np.isfinite(data)


def _point_from_index(
    row: int,
    col: int,
    *,
    qx: np.ndarray,
    qy: np.ndarray,
    radius: np.ndarray,
    raw: np.ndarray,
    smooth: np.ndarray,
    q_window: tuple[float, float] | None,
    signal_window: tuple[float, float] | None,
    q_resolution: float,
    flags: list[str] | None = None,
) -> dict[str, Any]:
    point_flags = list(flags or ())
    if row in {0, raw.shape[0] - 1} or col in {0, raw.shape[1] - 1}:
        point_flags.append("detector_boundary")
    point_flags.extend(
        _window_edge_flags(float(radius[row, col]), q_window, signal_window, q_resolution)
    )
    return {
        "pixel_x": int(col),
        "pixel_y": int(row),
        "qx": float(qx[row, col]),
        "qy": float(qy[row, col]),
        "q": float(radius[row, col]),
        "chi_deg": float(np.mod(np.degrees(np.arctan2(qy[row, col], qx[row, col])), 360.0)),
        "raw_intensity": float(raw[row, col]),
        "smoothed_intensity": (
            float(smooth[row, col]) if math.isfinite(float(smooth[row, col])) else None
        ),
        "flags": list(dict.fromkeys(point_flags)),
    }


def _select_local_maximum(
    *,
    angle_map: np.ndarray,
    radius: np.ndarray,
    qx: np.ndarray,
    qy: np.ndarray,
    raw: np.ndarray,
    smoothed: np.ndarray,
    smooth_support: np.ndarray,
    pixel_valid: np.ndarray,
    candidate_angle: float,
    basin_left: float,
    basin_right: float,
    basin_all: bool,
    q_window: tuple[float, float] | None,
    signal_window: tuple[float, float] | None,
    q_resolution: float,
    minimum_smoothing_support: float,
) -> tuple[dict[str, Any] | None, str | None]:
    basin = _interval_mask_deg(angle_map, basin_left, basin_right, all_angles=basin_all)
    coarse_valid = pixel_valid & basin & np.isfinite(smoothed) & (smooth_support > 0.20)
    if not np.any(coarse_valid):
        return None, "no_supported_pixel_in_angular_basin"
    coarse = np.where(coarse_valid, smoothed, -np.inf)
    coarse_flat = int(np.argmax(coarse))
    coarse_row, coarse_col = np.unravel_index(coarse_flat, raw.shape)
    if coarse_row in {0, raw.shape[0] - 1} or coarse_col in {0, raw.shape[1] - 1}:
        return None, "maximum_at_detector_boundary"
    coarse_radius = float(radius[coarse_row, coarse_col])
    edge_flags = _window_edge_flags(coarse_radius, q_window, signal_window, q_resolution)
    if edge_flags:
        return None, "maximum_at_q_window_boundary"
    usable = coarse_valid & (smooth_support >= minimum_smoothing_support)
    usable &= np.isfinite(smoothed)
    if not np.any(usable):
        return None, "insufficient_local_smoothing_support"
    scores = np.where(usable, smoothed, -np.inf)
    flat = int(np.argmax(scores))
    row, col = np.unravel_index(flat, raw.shape)
    if row in {0, raw.shape[0] - 1} or col in {0, raw.shape[1] - 1}:
        return None, "maximum_at_detector_boundary"
    radius_here = float(radius[row, col])
    edge_flags = _window_edge_flags(radius_here, q_window, signal_window, q_resolution)
    if edge_flags:
        return None, "maximum_at_q_window_boundary"
    point = _point_from_index(
        int(row),
        int(col),
        qx=qx,
        qy=qy,
        radius=radius,
        raw=raw,
        smooth=smoothed,
        q_window=q_window,
        signal_window=signal_window,
        q_resolution=q_resolution,
    )
    point["profile_angle_deg"] = float(candidate_angle)
    point["basin_angle_deg"] = [float(basin_left % 360.0), float(basin_right % 360.0)]
    return point, None


def _assign_basins(candidates: list[dict[str, Any]]) -> None:
    ordered = sorted(candidates, key=lambda item: item["angular_peak_deg"])
    count = len(ordered)
    for index, candidate in enumerate(ordered):
        angle = float(candidate["angular_peak_deg"])
        if count == 1:
            candidate["basin_left"] = (angle - 180.0) % 360.0
            candidate["basin_right"] = (angle + 180.0) % 360.0
            candidate["basin_all"] = True
            continue
        previous = float(ordered[(index - 1) % count]["angular_peak_deg"])
        following = float(ordered[(index + 1) % count]["angular_peak_deg"])
        previous_gap = (angle - previous) % 360.0
        following_gap = (following - angle) % 360.0
        candidate["basin_left"] = (previous + 0.5 * previous_gap) % 360.0
        candidate["basin_right"] = (angle + 0.5 * following_gap) % 360.0
        candidate["basin_all"] = False


def _profile_safe(profile: Mapping[str, np.ndarray]) -> dict[str, Any]:
    return {key: json_safe(value) for key, value in profile.items()}


def compute_peak_landmarks(
    observed: Any,
    qx: Any,
    qy: Any,
    *,
    valid_mask: Any = None,
    q_window: Any = None,
    signal_q_window: Any = None,
    q_unit: str = "unknown",
    model: Any = None,
    options: Mapping[str, Any] | None = None,
    cancel_event: Any = None,
) -> dict[str, Any]:
    """Return an unfiltered raw maximum and supported angular scattering lobes.

    Pixel coordinates use detector indexing: ``pixel_x`` is column (zero-based)
    and ``pixel_y`` is row (zero-based). A supplied ``signal_q_window`` restricts
    lobe-profile support; the raw maximum searches the full effective user domain
    defined by ``valid_mask`` and ``q_window``. The raw maximum is never called a
    reflection, and the lobe count is data-driven up to four.
    """

    raise_if_cancelled(cancel_event, "peak-landmarks:validate")
    observed_raw, observed_mask = _numeric_array(observed, "observed")
    qx_raw, qx_mask = _numeric_array(qx, "qx")
    qy_raw, qy_mask = _numeric_array(qy, "qy")
    if observed_raw.ndim != 2 or 0 in observed_raw.shape:
        raise ValueError("observed must be a non-empty 2D array")
    shape = observed_raw.shape
    if qx_raw.shape != shape or qy_raw.shape != shape:
        raise ValueError("qx and qy must have the same shape as observed")
    q_window_pair = _parse_window(q_window, "q_window")
    signal_window_pair = _parse_window(signal_q_window, "signal_q_window")
    if options is not None and not isinstance(options, Mapping):
        raise TypeError("options must be a mapping or None")
    option_values = dict(options or {})
    signal_window_origin = option_values.pop(
        "signal_q_window_origin", option_values.pop("signal_q_window_source", None)
    )
    if signal_window_origin is not None:
        signal_window_origin = str(signal_window_origin)
    thresholds, ignored_options = _options_from(option_values)
    try:
        raw = np.asarray(observed_raw, dtype=np.float64)
        qx_float = np.asarray(qx_raw, dtype=np.float64)
        qy_float = np.asarray(qy_raw, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError("observed, qx, and qy must be convertible to float64") from exc
    finite_q = np.isfinite(qx_float) & np.isfinite(qy_float) & ~qx_mask & ~qy_mask
    finite_observed = np.isfinite(raw) & ~observed_mask
    if valid_mask is None:
        user_valid = np.ones(shape, dtype=bool)
        user_mask_supplied = False
        user_mask_mask = np.zeros(shape, dtype=bool)
    else:
        mask_array = np.asanyarray(valid_mask)
        user_mask_mask = np.asarray(np.ma.getmaskarray(mask_array), dtype=bool)
        mask_values = np.asarray(np.ma.getdata(mask_array))
        if mask_values.shape != shape:
            raise ValueError("valid_mask must have the same shape as observed")
        if mask_values.dtype.kind != "b":
            raise TypeError("valid_mask must contain boolean values")
        user_valid = np.asarray(mask_values, dtype=bool)
        user_mask_supplied = True
    effective = (
        user_valid
        & ~user_mask_mask
        & finite_observed
        & finite_q
    )
    radius = np.hypot(qx_float, qy_float)
    raw_domain = _window_mask(radius, effective, q_window_pair)
    raw_count = int(np.count_nonzero(raw_domain))
    q_resolution = _pixel_q_resolution(qx_float, qy_float, effective)
    empty_signal_intersection = False
    if signal_window_pair is not None:
        effective_signal_window = signal_window_pair
        if q_window_pair is not None:
            effective_signal_window = (
                max(signal_window_pair[0], q_window_pair[0]),
                min(signal_window_pair[1], q_window_pair[1]),
            )
            if effective_signal_window[0] >= effective_signal_window[1]:
                effective_signal_window = None
                empty_signal_intersection = True
        signal_window_source = "explicit_signal_q_window"
        if signal_window_origin is None:
            signal_window_origin = "caller_supplied_signal_q_window"
    elif q_window_pair is not None:
        effective_signal_window = q_window_pair
        signal_window_source = "caller_q_window"
        if signal_window_origin is None:
            signal_window_origin = "inherited_from_user_q_window"
    else:
        effective_signal_window = None
        signal_window_source = "effective_valid_domain"
        if signal_window_origin is None:
            signal_window_origin = "effective_valid_q_domain"

    if empty_signal_intersection:
        signal_geometry = np.zeros(shape, dtype=bool)
        signal_valid = np.zeros(shape, dtype=bool)
    else:
        signal_geometry = _window_mask(radius, finite_q, effective_signal_window)
        signal_valid = raw_domain & _window_mask(radius, finite_q, effective_signal_window)
    if q_resolution > 0.0:
        angular_q_floor = 0.5 * q_resolution
    else:
        angular_q_floor = 0.0
    profile_geometry = signal_geometry & (radius >= angular_q_floor)
    profile_valid = signal_valid & (radius >= angular_q_floor)
    chi = np.mod(np.degrees(np.arctan2(qy_float, qx_float)), 360.0)

    if raw_count:
        raw_values = np.where(raw_domain, raw, -np.inf)
        raw_flat = int(np.argmax(raw_values))
        raw_row, raw_col = np.unravel_index(raw_flat, shape)
        raw_point = _point_from_index(
            int(raw_row),
            int(raw_col),
            qx=qx_float,
            qy=qy_float,
            radius=radius,
            raw=raw,
            smooth=np.full(shape, np.nan, dtype=np.float64),
            q_window=q_window_pair,
            signal_window=None,
            q_resolution=q_resolution,
        )
        raw_point["interpretation"] = "raw_maximum_only_not_an_identified_reflection"
        raw_point["label"] = "G"
        if radius[raw_row, raw_col] <= 1.5 * q_resolution and q_resolution > 0.0:
            raw_point["flags"].append("near_q_origin_inspect_beamstop_or_direct_beam")
        raw_point["flags"] = list(dict.fromkeys(raw_point["flags"]))
        raw_point["smoothed_intensity"] = None
    else:
        raw_point = None

    raise_if_cancelled(cancel_event, "peak-landmarks:profile-preparation")
    if effective_signal_window is not None and effective_signal_window[0] < effective_signal_window[1]:
        profile_valid &= (radius >= effective_signal_window[0]) & (
            radius <= effective_signal_window[1]
        )
        profile_geometry &= (radius >= effective_signal_window[0]) & (
            radius <= effective_signal_window[1]
        )
    smooth, smooth_support = _smooth_masked(
        raw,
        profile_valid,
        float(thresholds["smoothing_sigma_px"]),
        float(thresholds["minimum_smoothing_support"]),
    )
    usable_smooth = np.isfinite(smooth)
    hp_values = raw[profile_valid & usable_smooth] - smooth[profile_valid & usable_smooth]
    pixel_noise = _robust_sigma(hp_values)
    if pixel_noise == 0.0 and np.count_nonzero(profile_valid) > 1:
        # Constant or noiseless frames retain zero empirical noise; only a
        # machine-scale floor is used later when converting prominence to SNR.
        pixel_noise = 0.0

    profile_data = _profile_arrays(
        radius=radius,
        angle_deg=chi,
        raw=raw,
        smooth=smooth,
        valid=profile_valid,
        geometry=profile_geometry,
        angular_bins=int(thresholds["angular_bins"]),
        radial_bins=int(thresholds["radial_bins"]),
        min_coverage=float(thresholds["minimum_angular_coverage_fraction"]),
        min_bin_pixels=int(thresholds["minimum_angular_bin_pixels"]),
    )
    isotropic_reference_map, radial_reference = _isotropic_reference_map(
        raw=raw,
        radius=radius,
        valid=profile_valid,
        geometry=profile_geometry,
        pixel_q_resolution=q_resolution,
        maximum_bins=int(thresholds["radial_bins"]),
    )
    isotropic_reference_smooth, _ = _smooth_masked(
        isotropic_reference_map,
        profile_valid,
        float(thresholds["smoothing_sigma_px"]),
        float(thresholds["minimum_smoothing_support"]),
    )
    isotropic_profile_data = _profile_arrays(
        radius=radius,
        angle_deg=chi,
        raw=isotropic_reference_map,
        smooth=isotropic_reference_smooth,
        valid=profile_valid,
        geometry=profile_geometry,
        angular_bins=int(thresholds["angular_bins"]),
        radial_bins=int(thresholds["radial_bins"]),
        min_coverage=float(thresholds["minimum_angular_coverage_fraction"]),
        min_bin_pixels=int(thresholds["minimum_angular_bin_pixels"]),
    )
    angular = profile_data["angular"]
    isotropic_angular = isotropic_profile_data["angular"]
    profile_sigma = float(thresholds["angular_profile_sigma_bins"])
    angular_supported = np.asarray(angular["supported"], dtype=bool)
    measured_angular = np.asarray(angular["intensity_smoothed"], dtype=np.float64)
    isotropic_angular_reference = np.asarray(
        isotropic_angular["intensity_smoothed"], dtype=np.float64
    )
    common_angular_support = (
        angular_supported
        & np.isfinite(measured_angular)
        & np.isfinite(isotropic_angular_reference)
    )
    if profile_sigma > 0.0:
        angular_counts = np.asarray(angular["valid_pixel_counts"], dtype=np.float64)
        denominator = gaussian_filter1d(
            common_angular_support.astype(np.float64),
            sigma=profile_sigma,
            mode="wrap",
        )

        def filter_angular(values: np.ndarray) -> np.ndarray:
            numerator = gaussian_filter1d(
                np.where(common_angular_support, values, 0.0),
                sigma=profile_sigma,
                mode="wrap",
            )
            filtered = np.full(values.shape, np.nan, dtype=np.float64)
            np.divide(numerator, denominator, out=filtered, where=denominator > 0.05)
            return filtered

        measured_angular = filter_angular(measured_angular)
        isotropic_angular_reference = filter_angular(isotropic_angular_reference)
        angular["intensity_smoothed"] = measured_angular
        counts_numerator = gaussian_filter1d(
            np.where(common_angular_support, angular_counts, 0.0),
            sigma=profile_sigma,
            mode="wrap",
        )
        angular["smoothed_pixel_counts"] = np.rint(counts_numerator).astype(np.int64)
    angular["intensity_isotropic_reference"] = isotropic_angular_reference
    angular["intensity_detection"] = measured_angular - isotropic_angular_reference

    raise_if_cancelled(cancel_event, "peak-landmarks:angular-detection")
    detection_angular = dict(angular)
    detection_angular["intensity_smoothed"] = np.asarray(
        angular["intensity_detection"], dtype=np.float64
    )
    candidates, profile_noise = _angular_candidates(
        detection_angular, options=thresholds, pixel_noise=pixel_noise
    )
    for candidate in candidates:
        candidate_index = int(candidate["candidate_index"])
        candidate["measured_profile_intensity"] = float(
            angular["intensity_smoothed"][candidate_index]
        )
        candidate["isotropic_reference_profile_intensity"] = float(
            angular["intensity_isotropic_reference"][candidate_index]
        )
    if profile_valid.any():
        raw_reference_residual = raw - isotropic_reference_map
        raw_baseline = float(np.median(raw_reference_residual[profile_valid]))
    else:
        raw_reference_residual = np.full(shape, np.nan, dtype=np.float64)
        raw_baseline = 0.0
    support_threshold = raw_baseline + float(thresholds["raw_support_sigma"]) * pixel_noise
    angular_half_width = 180.0 / int(thresholds["angular_bins"])
    support_selection = profile_valid & (raw_reference_residual > support_threshold)
    safe_chi = np.where(np.isfinite(chi), np.mod(chi, 360.0), 0.0)
    support_bin_index = np.minimum(
        int(thresholds["angular_bins"]) - 1,
        np.floor(safe_chi / 360.0 * int(thresholds["angular_bins"])).astype(np.int64),
    )
    support_by_angle = np.bincount(
        support_bin_index[support_selection],
        minlength=int(thresholds["angular_bins"]),
    )
    detector_border = np.zeros(shape, dtype=bool)
    detector_border[0, :] = True
    detector_border[-1, :] = True
    detector_border[:, 0] = True
    detector_border[:, -1] = True
    border_support_by_angle = np.bincount(
        support_bin_index[support_selection & detector_border],
        minlength=int(thresholds["angular_bins"]),
    )
    signal_boundary_support = np.zeros(shape, dtype=bool)
    if effective_signal_window is not None and q_resolution > 0.0:
        boundary_tolerance = 0.5 * q_resolution
        signal_boundary_support = (
            np.minimum(
                np.abs(radius - effective_signal_window[0]),
                np.abs(radius - effective_signal_window[1]),
            )
            <= boundary_tolerance
        )
    signal_edge_support_by_angle = np.bincount(
        support_bin_index[support_selection & signal_boundary_support],
        minlength=int(thresholds["angular_bins"]),
    )
    angular_centers = np.asarray(angular["angle_deg"], dtype=np.float64)
    preliminarily_supported: list[dict[str, Any]] = []
    for candidate in candidates:
        if not candidate["accepted"]:
            continue
        nearby_angle_bins = _circular_distance_deg(
            float(candidate["angular_peak_deg"]), angular_centers
        ) <= max(float(candidate["support_half_width_deg"]), angular_half_width)
        support_count = int(np.sum(support_by_angle[nearby_angle_bins]))
        border_support_count = int(np.sum(border_support_by_angle[nearby_angle_bins]))
        signal_edge_support_count = int(np.sum(signal_edge_support_by_angle[nearby_angle_bins]))
        candidate["support_pixel_count"] = support_count
        candidate["detector_border_support_pixel_count"] = border_support_count
        candidate["signal_window_boundary_support_pixel_count"] = signal_edge_support_count
        candidate["support_flags"] = []
        if support_count < int(thresholds["minimum_lobe_support_pixels"]):
            candidate["accepted"] = False
            candidate["reason"] = "insufficient_unfiltered_isotropic_residual_support"
            candidate["reasons"].append("insufficient_unfiltered_isotropic_residual_support")
        else:
            if border_support_count:
                candidate["support_flags"].append("lobe_signal_reaches_detector_boundary")
            if signal_edge_support_count:
                candidate["support_flags"].append("lobe_signal_reaches_signal_window_boundary")
            preliminarily_supported.append(candidate)

    separation = float(thresholds["minimum_peak_separation_deg"])
    independent: list[dict[str, Any]] = []
    for candidate in preliminarily_supported:
        too_close = any(
            float(_circular_distance_deg(candidate["angular_peak_deg"], item["angular_peak_deg"]))
            < separation
            for item in independent
        )
        if too_close:
            candidate["accepted"] = False
            candidate["reason"] = "within_minimum_separation_of_stronger_candidate"
            candidate["reasons"].append("within_minimum_separation_of_stronger_candidate")
            continue
        if len(independent) >= int(thresholds["maximum_lobes"]):
            candidate["accepted"] = False
            candidate["reason"] = "maximum_supported_lobe_count_reached"
            candidate["reasons"].append("maximum_supported_lobe_count_reached")
            continue
        independent.append(candidate)

    _assign_basins(independent)
    accepted_candidates: list[dict[str, Any]] = []
    for candidate in independent:
        point, reason = _select_local_maximum(
            angle_map=chi,
            radius=radius,
            qx=qx_float,
            qy=qy_float,
            raw=raw,
            smoothed=smooth,
            smooth_support=smooth_support,
            pixel_valid=profile_valid,
            candidate_angle=float(candidate["angular_peak_deg"]),
            basin_left=float(candidate["basin_left"]),
            basin_right=float(candidate["basin_right"]),
            basin_all=bool(candidate["basin_all"]),
            q_window=q_window_pair,
            signal_window=signal_window_pair,
            q_resolution=q_resolution,
            minimum_smoothing_support=float(thresholds["minimum_smoothing_support"]),
        )
        if point is None:
            candidate["accepted"] = False
            candidate["reason"] = reason
            candidate["reasons"].append(str(reason))
            continue
        candidate["point"] = point
        accepted_candidates.append(candidate)

    model_data = _model_array(model, shape)
    if model_data is not None and accepted_candidates:
        model_raw, model_valid = model_data
        model_profile_valid = profile_valid & model_valid
        model_smooth, model_support = _smooth_masked(
            model_raw,
            model_profile_valid,
            float(thresholds["smoothing_sigma_px"]),
            float(thresholds["minimum_smoothing_support"]),
        )
        for candidate in accepted_candidates:
            model_peak, reason = _select_local_maximum(
                angle_map=chi,
                radius=radius,
                qx=qx_float,
                qy=qy_float,
                raw=model_raw,
                smoothed=model_smooth,
                smooth_support=model_support,
                pixel_valid=model_profile_valid,
                candidate_angle=float(candidate["angular_peak_deg"]),
                basin_left=float(candidate["basin_left"]),
                basin_right=float(candidate["basin_right"]),
                basin_all=bool(candidate["basin_all"]),
                q_window=q_window_pair,
                signal_window=signal_window_pair,
                q_resolution=q_resolution,
                minimum_smoothing_support=float(thresholds["minimum_smoothing_support"]),
            )
            candidate["model_peak"] = model_peak
            candidate["model_peak_unavailable_reason"] = reason
            if model_peak is not None:
                observed_point = candidate["point"]
                candidate["delta_qx"] = float(model_peak["qx"] - observed_point["qx"])
                candidate["delta_qy"] = float(model_peak["qy"] - observed_point["qy"])
                candidate["delta_q"] = float(
                    math.hypot(candidate["delta_qx"], candidate["delta_qy"])
                )
    else:
        for candidate in accepted_candidates:
            candidate["model_peak"] = None
            candidate["model_peak_unavailable_reason"] = None
            candidate["delta_qx"] = None
            candidate["delta_qy"] = None
            candidate["delta_q"] = None

    accepted_candidates.sort(key=lambda item: item["angular_peak_deg"])
    peaks: list[dict[str, Any]] = []
    for index, candidate in enumerate(accepted_candidates, start=1):
        point = candidate["point"]
        peak_flags = list(point["flags"]) + list(candidate.get("support_flags", ()))
        if candidate.get("model_peak_unavailable_reason"):
            peak_flags.append("model_maximum_unavailable_in_basin")
        peaks.append(
            {
                "peak_id": f"P{index}",
                "angular_peak_deg": float(candidate["angular_peak_deg"]),
                "chi_deg": float(point["chi_deg"]),
                "pixel_x": int(point["pixel_x"]),
                "pixel_y": int(point["pixel_y"]),
                "qx": float(point["qx"]),
                "qy": float(point["qy"]),
                "q": float(point["q"]),
                "raw_intensity": float(point["raw_intensity"]),
                "smoothed_intensity": float(point["smoothed_intensity"]),
                "angular_detection_intensity": float(candidate["profile_intensity"]),
                "angular_isotropic_reference_intensity": float(
                    candidate["isotropic_reference_profile_intensity"]
                ),
                "angular_prominence": float(candidate["angular_prominence"]),
                "angular_noise_sigma": float(candidate["angular_noise_sigma"]),
                "angular_snr": float(candidate["angular_snr"]),
                "angular_fwhm_deg": float(candidate["angular_fwhm_deg"]),
                "angular_coverage_fraction": float(candidate["angular_coverage_fraction"]),
                "support_pixel_count": int(candidate["support_pixel_count"]),
                "basin_angle_deg": [
                    float(candidate["basin_left"] % 360.0),
                    float(candidate["basin_right"] % 360.0),
                ],
                "flags": list(dict.fromkeys(peak_flags)),
                "model_peak": candidate.get("model_peak"),
                "model_peak_unavailable_reason": candidate.get(
                    "model_peak_unavailable_reason"
                ),
                "delta_qx": candidate.get("delta_qx"),
                "delta_qy": candidate.get("delta_qy"),
                "delta_q": candidate.get("delta_q"),
            }
        )

    final_candidates: list[dict[str, Any]] = []
    for candidate in candidates:
        clean = {key: value for key, value in candidate.items() if key not in {"point", "basin_left", "basin_right", "basin_all"}}
        if candidate.get("accepted"):
            clean["reason"] = None
        final_candidates.append(clean)
    final_flags: list[str] = []
    if raw_point is None:
        final_flags.append("no_valid_pixels_in_raw_search_domain")
    if not peaks:
        final_flags.append("no_supported_angular_lobes_found")
    if empty_signal_intersection:
        final_flags.append("signal_q_window_does_not_overlap_user_q_window")
    if raw_point is not None and "detector_boundary" in raw_point["flags"]:
        final_flags.append("raw_global_maximum_at_detector_boundary")
    if raw_point is not None and "near_q_origin_inspect_beamstop_or_direct_beam" in raw_point["flags"]:
        final_flags.append("raw_global_maximum_near_q_origin_requires_inspection")

    if raw_count == 0:
        q_range = None
    else:
        q_values = radius[raw_domain]
        q_range = [float(np.min(q_values)), float(np.max(q_values))]
    signal_count = int(np.count_nonzero(signal_valid))
    domain = {
        "search_q_window": q_window_pair,
        "signal_q_window": signal_window_pair,
        "effective_signal_q_window": effective_signal_window,
        "signal_q_window_source": signal_window_source,
        "signal_q_window_origin": signal_window_origin,
        "q_unit": str(q_unit or "unknown"),
        "valid_mask_supplied": bool(user_mask_supplied),
        "valid_mask_semantics": (
            "caller-supplied boolean selection mask; True means valid"
            if user_mask_supplied
            else "no caller selection mask supplied; masked/non-finite observations and q coordinates define validity"
        ),
        "raw_search_pixel_count": raw_count,
        "signal_pixel_count": signal_count,
        "angular_profile_pixel_count": int(np.count_nonzero(profile_valid)),
        "raw_search_radius_range": q_range,
        "pixel_q_resolution": q_resolution,
        "angular_q_floor": angular_q_floor,
        "raw_search_empty": raw_point is None,
    }
    status = "ok" if peaks else ("no_valid_pixels" if raw_point is None else "no_supported_lobes")
    output = {
        "schema_version": PEAK_LANDMARK_SCHEMA_VERSION,
        "method_version": PEAK_LANDMARK_METHOD_VERSION,
        "status": status,
        "interpretation": "Observed intensity landmarks only; not reflection indexing, structure assignment, or fit acceptance.",
        "peak_selection_method": "maximum mask-normalized smoothed observed pixel within each independent supported angular basin and signal q band, subject to minimum smoothing support",
        "angular_detection": {
            "method": "observed_angular_profile_minus_isotropic_reference",
            "reference_method": "median raw intensity per adaptive radial bin, interpolated at each pixel's actual q radius",
            "reference_estimator": "median_raw_intensity_per_radial_bin",
            "reference_profile_key": "profiles.angular.intensity_isotropic_reference",
            "detection_profile_key": "profiles.angular.intensity_detection",
            "support_method": "raw intensity minus isotropic reference map within the signal q window",
            "reference_spatial_smoothing": "same mask-normalized spatial smoothing and angular binning as the observed frame",
            "noise_estimator": {
                "method": "robust MAD of the supported corrected angular profile minus its circular Gaussian-smoothed baseline",
                "smoothing_sigma_bins": float(thresholds["angular_noise_smoothing_sigma_bins"]),
                "propagated_pixel_floor": "pixel high-pass sigma divided by the square root of the median valid pixels per supported angular bin; assumes independent pixel residuals",
                "combination": "maximum of angular high-pass MAD, propagated pixel floor, and float64 noise floor",
                "prominence_rule": "minimum_prominence_sigma times the combined angular noise estimate; default multiplier remains 4",
            },
            "radial_reference": radial_reference,
        },
        "model_comparison": {
            "supplied": model_data is not None,
            "selection_method": (
                "maximum mask-normalized smoothed supplied-model pixel within the same observed angular basin and signal q band, subject to the same minimum smoothing support"
                if model_data is not None
                else None
            ),
            "paired_lobe_count": sum(peak.get("model_peak") is not None for peak in peaks),
            "delta_q_definition": "hypot(model_qx - observed_qx, model_qy - observed_qy); 2D q-vector displacement magnitude, not a difference between q radii",
        },
        "q_unit": str(q_unit or "unknown"),
        "domain": domain,
        "raw_global_max": raw_point,
        "peak_count": len(peaks),
        "peaks": peaks,
        "profiles": {key: _profile_safe(value) for key, value in profile_data.items()},
        "candidate_diagnostics": final_candidates,
        "noise": {
            "pixel_highpass_sigma": pixel_noise,
            "angular_profile_highpass_sigma": profile_noise,
            "raw_support_baseline": raw_baseline,
            "raw_support_threshold": support_threshold,
            "raw_support_pixel_count_in_signal_band": int(
                np.count_nonzero(support_selection)
            ),
            "raw_support_basis": "raw intensity minus isotropic reference map",
            "noise_estimators": "robust MAD of valid-pixel raw-minus-mask-normalized-smooth; angular estimate uses the supported corrected profile high-pass and the propagated pixel floor documented in angular_detection.noise_estimator",
        },
        "options": thresholds,
        "ignored_options": ignored_options,
        "flags": final_flags,
    }
    raise_if_cancelled(cancel_event, "peak-landmarks:serialize")
    return json_safe(output)


__all__ = [
    "PEAK_LANDMARK_METHOD_VERSION",
    "PEAK_LANDMARK_SCHEMA_VERSION",
    "compute_peak_landmarks",
]
