"""Observed normal profiles for the independent butterfly ridge tracer.

The profile helpers in this module are deliberately small and descriptive.  A
profile is an observation attached to an already detected candidate; a
Gaussian fit is a local measurement aid and is never a replacement for the
observed intensity samples.  In particular, the optional Wang (2007)-style
slice check is reported as a diagnostic with an explicit applicability flag.
It is not used as a truth source or as an automatic four-lobe constraint.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .ridge_inputs import (
    as_image as _shared_as_image,
    as_qmap as _shared_as_qmap,
    coordinate_derivatives as _coordinate_derivatives,
    field as _field,
    robust_noise as _robust_noise,
    sample as _sample,
)

try:  # scipy is a declared core dependency of butterfly-saxs.
    from scipy.ndimage import gaussian_filter1d
    from scipy.optimize import curve_fit
    from scipy.signal import find_peaks
except Exception:  # pragma: no cover - useful for source inspection only
    gaussian_filter1d = None
    curve_fit = None
    find_peaks = None


METHOD_VERSION = "butterfly-ridge-profiles-v1.0"


def _as_image(image: Any) -> tuple[np.ndarray, np.ndarray | None]:
    return _shared_as_image(image)


def _as_qmap(qmap: Any, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    return _shared_as_qmap(qmap, shape)


def _normalise_options(options: Any) -> dict[str, Any]:
    if options is None:
        return {}
    if isinstance(options, Mapping):
        return dict(options)
    try:
        return dict(vars(options))
    except TypeError:
        return {}


def _q_step(qx: np.ndarray, qy: np.ndarray, py: int, px: int) -> float:
    """Estimate a local q step from the physical q map."""

    derivatives, _inverse, representative = _coordinate_derivatives(qx, qy)
    local = np.asarray(
        [
            np.hypot(derivatives["qx_x"][py, px], derivatives["qy_x"][py, px]),
            np.hypot(derivatives["qx_y"][py, px], derivatives["qy_y"][py, px]),
        ],
        dtype=float,
    )
    finite = local[np.isfinite(local) & (local > np.finfo(float).eps)]
    if finite.size:
        return float(np.median(finite))
    # The full map is only used as a fallback for a singular local Jacobian.
    return float(representative) if np.isfinite(representative) and representative > 0 else 1.0


def _local_inverse_jacobian(qx: np.ndarray, qy: np.ndarray, py: int, px: int) -> tuple[np.ndarray, float]:
    derivatives, _inverse, _q_step_value = _coordinate_derivatives(qx, qy)
    j = np.asarray(
        [
            [derivatives["qx_x"][py, px], derivatives["qx_y"][py, px]],
            [derivatives["qy_x"][py, px], derivatives["qy_y"][py, px]],
        ],
        dtype=float,
    )
    determinant = float(np.linalg.det(j))
    if not np.isfinite(determinant) or abs(determinant) <= 1e-14:
        return np.full((2, 2), np.nan, dtype=float), determinant
    return np.linalg.inv(j), determinant


def build_profile_context(image: Any, qmap: Any, *, mask: Any = None) -> dict[str, Any]:
    """Precompute maps reused by a batch of local normal profiles.

    ``trace_butterfly_ridges`` can attach this context through the private
    ``_profile_context`` option.  Computing q-map gradients once is essential
    for large detector images: one profile should not differentiate a 2.4 Mpx
    q map again for every candidate point.
    """

    data, image_mask = _as_image(image)
    qx, qy, q, qmap_mask = _as_qmap(qmap, data.shape)
    invalid = np.zeros(data.shape, dtype=bool)
    if image_mask is not None:
        invalid |= image_mask
    if qmap_mask is not None:
        invalid |= qmap_mask
    if mask is not None:
        try:
            explicit = np.asarray(np.broadcast_to(np.asarray(mask, dtype=bool), data.shape), dtype=bool)
        except ValueError as exc:
            raise ValueError("profile context mask must broadcast to image shape") from exc
        invalid |= explicit
    valid = np.isfinite(data) & ~invalid & np.isfinite(qx) & np.isfinite(qy)
    derivatives, inverse, q_step = _coordinate_derivatives(qx, qy)
    determinant = derivatives["determinant"]
    return {
        "data": data,
        "qx": qx,
        "qy": qy,
        "q": q,
        "valid": valid,
        "inverse_jacobian": inverse,
        "jacobian_determinant": determinant,
        "q_step": q_step,
    }


def _gaussian_sum(x: np.ndarray, *parameters: float) -> np.ndarray:
    """Constant-plus-Gaussian model used for local profile diagnostics."""

    if len(parameters) < 3:
        return np.full_like(x, np.nan, dtype=float)
    baseline, slope = float(parameters[0]), float(parameters[1])
    result = baseline + slope * x
    for offset in range(2, len(parameters), 3):
        if offset + 2 >= len(parameters):
            break
        amplitude, centre, sigma = (float(parameters[offset + item]) for item in range(3))
        sigma = max(abs(sigma), np.finfo(float).eps)
        result = result + amplitude * np.exp(-0.5 * ((x - centre) / sigma) ** 2)
    return np.asarray(result, dtype=float)


def _initial_peak_centres(x: np.ndarray, y: np.ndarray, noise: float, step: float) -> list[int]:
    finite = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(finite) < 5:
        return []
    smooth = y.copy()
    if gaussian_filter1d is not None and np.count_nonzero(finite) == y.size:
        smooth = gaussian_filter1d(y, sigma=0.8, mode="nearest")
    prominence = max(2.0 * noise if np.isfinite(noise) else 0.0, 0.03 * max(float(np.nanmax(smooth) - np.nanmin(smooth)), 0.0))
    distance = max(1, int(np.ceil(1.25 * max(step, np.finfo(float).eps) / max(np.nanmedian(np.diff(x)), np.finfo(float).eps))))
    if find_peaks is None:  # pragma: no cover
        candidates = [int(np.nanargmax(smooth))]
    else:
        candidates, properties = find_peaks(
            np.nan_to_num(smooth, nan=float(np.nanmedian(smooth[finite]))),
            prominence=prominence,
            distance=distance,
        )
        if not len(candidates):
            candidates = np.asarray([int(np.nanargmax(smooth))], dtype=int)
        else:
            prominences = np.asarray(properties.get("prominences", np.zeros(len(candidates))), dtype=float)
            candidates = np.asarray(candidates, dtype=int)[np.argsort(prominences)[::-1]]
    selected: list[int] = []
    for index in np.asarray(candidates, dtype=int).tolist():
        if not finite[index]:
            continue
        if all(abs(float(x[index] - x[other])) >= 1.25 * max(step, np.finfo(float).eps) for other in selected):
            selected.append(int(index))
        if len(selected) == 2:
            break
    return selected


def _fit_profile_models(
    x: np.ndarray,
    y: np.ndarray,
    *,
    noise: float,
    step: float,
    delta_bic: float,
) -> dict[str, Any]:
    """Fit one and, where supported by the samples, two local peaks."""

    finite = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(finite) < 5:
        return {
            "model": "none",
            "fit": np.full_like(x, np.nan, dtype=float),
            "peaks": [],
            "ambiguous": False,
            "uncertainty_source": "unavailable_insufficient_samples",
        }
    xx, yy = np.asarray(x[finite], dtype=float), np.asarray(y[finite], dtype=float)
    baseline = float(np.nanpercentile(yy, 10.0))
    amplitude = max(float(np.nanmax(yy) - baseline), np.finfo(float).eps)
    sigma_min = max(0.35 * step, np.finfo(float).eps)
    sigma_max = max(float(np.ptp(xx)), 2.0 * sigma_min)
    centre_indices = _initial_peak_centres(xx, yy, noise, step)

    fits: dict[int, tuple[np.ndarray, np.ndarray | None, float, float]] = {}
    for count in (1, 2):
        if count == 2 and len(centre_indices) < 2:
            continue
        centres = [float(xx[index]) for index in centre_indices[:count]]
        heights = [max(float(yy[index] - baseline), 0.25 * amplitude) for index in centre_indices[:count]]
        initial = [baseline, 0.0]
        lower = [float(np.nanmin(yy) - amplitude), -amplitude / max(sigma_max, 1.0)]
        upper = [float(np.nanmax(yy) + amplitude), amplitude / max(sigma_min, np.finfo(float).eps)]
        for height, centre in zip(heights, centres):
            initial.extend((height, centre, max(1.5 * sigma_min, min(0.35 * sigma_max, sigma_max))))
            lower.extend((0.0, float(np.nanmin(xx)), sigma_min))
            upper.extend((4.0 * amplitude, float(np.nanmax(xx)), sigma_max))
        try:
            if curve_fit is None:  # pragma: no cover
                raise RuntimeError("scipy.optimize.curve_fit is unavailable")
            parameters, covariance = curve_fit(
                _gaussian_sum,
                xx,
                yy,
                p0=np.asarray(initial, dtype=float),
                bounds=(np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)),
                maxfev=5000,
            )
            fitted = _gaussian_sum(xx, *parameters)
            residual = yy - fitted
            rss = float(np.sum(np.square(residual)))
            n = int(xx.size)
            k = int(parameters.size)
            bic = float(n * np.log(max(rss / max(n, 1), np.finfo(float).eps)) + k * np.log(max(n, 2)))
            fits[count] = (np.asarray(parameters, dtype=float), np.asarray(covariance, dtype=float), rss, bic)
        except Exception:
            continue

    if not fits:
        fallback = np.full_like(x, np.nan, dtype=float)
        fallback[finite] = baseline + amplitude * np.exp(-0.5 * ((x[finite] - float(x[int(np.nanargmax(y))])) / max(2.0 * sigma_min, sigma_min)) ** 2)
        return {
            "model": "single_gaussian_fallback",
            "fit": fallback,
            "peaks": [],
            "ambiguous": True,
            "uncertainty_source": "robust_profile_fallback",
        }

    one = fits.get(1)
    two = fits.get(2)
    ambiguous = False
    if one is None:
        selected_count = 2
    elif two is None:
        selected_count = 1
    else:
        # A second component is retained only when it earns a material BIC
        # improvement.  Close scores are reported as ambiguity rather than
        # silently making a structural decision.
        bic_delta = float(one[3] - two[3])
        if bic_delta > float(delta_bic):
            selected_count = 2
        elif abs(bic_delta) <= float(delta_bic):
            selected_count = 1
            ambiguous = True
        else:
            selected_count = 1
    parameters, covariance, _rss, _bic = fits[selected_count]
    fitted_all = np.full_like(x, np.nan, dtype=float)
    fitted_all[finite] = _gaussian_sum(xx, *parameters)
    peaks: list[dict[str, float]] = []
    for offset in range(2, len(parameters), 3):
        if offset + 2 >= len(parameters):
            break
        height, centre, sigma = (float(parameters[offset + item]) for item in range(3))
        sigma = abs(sigma)
        centre_stderr = float("nan")
        if covariance is not None and covariance.ndim == 2 and offset + 1 < covariance.shape[0]:
            variance = float(covariance[offset + 1, offset + 1])
            if np.isfinite(variance) and variance >= 0:
                centre_stderr = float(np.sqrt(variance))
        snr = float(height / max(noise, np.finfo(float).eps)) if np.isfinite(noise) else float("nan")
        peaks.append(
            {
                "height": height,
                "center_q": centre,
                "sigma_q": sigma,
                "fwhm_q": float(2.354820045 * sigma),
                "area": float(height * sigma * np.sqrt(2.0 * np.pi)),
                "snr": snr,
                "center_sigma_q": centre_stderr,
            }
        )
    peaks.sort(key=lambda peak: (-float(peak.get("height", 0.0)), abs(float(peak.get("center_q", 0.0)))))
    if selected_count == 2:
        model = "double_gaussian"
    else:
        model = "ambiguous_single_double" if ambiguous else "single_gaussian"
    uncertainty_source = "curve_fit_covariance" if covariance is not None and np.all(np.isfinite(covariance)) else "robust_profile_residual"
    return {
        "model": model,
        "fit": fitted_all,
        "peaks": peaks,
        "ambiguous": bool(ambiguous),
        "uncertainty_source": uncertainty_source,
    }


def extract_normal_profile(
    image: Any,
    qmap: Any,
    point: Mapping[str, Any] | Any,
    *,
    mask: Any = None,
    options: Any = None,
    cancel_event: Any = None,
) -> dict[str, Any]:
    """Sample and fit one observed normal profile at a ridge candidate.

    ``raw_intensity`` is always retained.  ``fit_intensity`` and the Gaussian
    peak records are local descriptive fits; a failed fit leaves the raw
    samples available and reports the uncertainty source explicitly.
    """

    if cancel_event is not None:
        from .cancellation import raise_if_cancelled

        raise_if_cancelled(cancel_event, "ridge-profile:start")
    opts = _normalise_options(options)
    context = opts.get("_profile_context")
    if isinstance(context, Mapping):
        # Reuse the precomputed buffers.  Recasting 2.5 Mpx arrays on every
        # candidate was the dominant cost on calibrated detector frames.
        data = context["data"]
        qx = context["qx"]
        qy = context["qy"]
        valid = context["valid"]
        inverse_context = context.get("inverse_jacobian")
        determinant_context = context.get("jacobian_determinant")
        context_q_step = float(context.get("q_step", float("nan")))
    else:
        data, _image_mask = _as_image(image)
        qx, qy, _q, _qmap_mask = _as_qmap(qmap, data.shape)
        invalid = np.zeros(data.shape, dtype=bool)
        if _image_mask is not None:
            invalid |= _image_mask
        if _qmap_mask is not None:
            invalid |= _qmap_mask
        if mask is not None:
            try:
                explicit = np.asarray(np.broadcast_to(np.asarray(mask, dtype=bool), data.shape), dtype=bool)
            except ValueError as exc:
                raise ValueError("profile mask must broadcast to image shape") from exc
            invalid |= explicit
        valid = np.isfinite(data) & ~invalid & np.isfinite(qx) & np.isfinite(qy)
        inverse_context = np.full((4, *data.shape), np.nan, dtype=float)
        determinant_context = np.full(data.shape, np.nan, dtype=float)
        context_q_step = float("nan")

    py = float(_field(point, ("pixel_y", "y_pixel", "row"), float("nan")))
    px = float(_field(point, ("pixel_x", "x_pixel", "column", "col"), float("nan")))
    qx0 = float(_field(point, ("qx", "q_x"), float("nan")))
    qy0 = float(_field(point, ("qy", "q_y"), float("nan")))
    if not (np.isfinite(py) and np.isfinite(px)):
        if np.isfinite(qx0) and np.isfinite(qy0):
            distance = np.square(qx - qx0) + np.square(qy - qy0)
            distance[~np.isfinite(distance)] = np.inf
            index = int(np.argmin(distance))
            py, px = (float(value) for value in np.unravel_index(index, data.shape))
        else:
            return {
                "point_id": _field(point, ("point_id", "id"), ""),
                "valid": False,
                "model": "none",
                "reason": "missing_point_coordinate",
                "uncertainty_source": "unavailable_missing_point_coordinate",
                "offset_q": [],
                "raw_intensity": [],
                "fit_intensity": [],
                "background": float("nan"),
                "peaks": [],
                "snr": float("nan"),
            }
    py_clamped, px_clamped = int(np.clip(np.rint(py), 0, data.shape[0] - 1)), int(np.clip(np.rint(px), 0, data.shape[1] - 1))
    normal_x = float(_field(point, ("normal_qx", "normal_x"), float("nan")))
    normal_y = float(_field(point, ("normal_qy", "normal_y"), float("nan")))
    if not (np.isfinite(normal_x) and np.isfinite(normal_y)) or np.hypot(normal_x, normal_y) <= np.finfo(float).eps:
        radial_x, radial_y = qx0, qy0
        norm = float(np.hypot(radial_x, radial_y))
        if norm <= np.finfo(float).eps or not np.isfinite(norm):
            normal_x, normal_y = 0.0, 1.0
        else:
            normal_x, normal_y = radial_x / norm, radial_y / norm
    else:
        norm = float(np.hypot(normal_x, normal_y))
        normal_x, normal_y = normal_x / norm, normal_y / norm
    if isinstance(context, Mapping):
        inverse_j = inverse_context[:, py_clamped, px_clamped].reshape(2, 2)
        determinant = float(determinant_context[py_clamped, px_clamped])
    else:
        inverse_j, determinant = _local_inverse_jacobian(qx, qy, py_clamped, px_clamped)
    step = float(_field(point, ("q_normal_step", "normal_step_q"), float("nan")))
    if not np.isfinite(step) or step <= 0:
        step = context_q_step if np.isfinite(context_q_step) else _q_step(qx, qy, py_clamped, px_clamped)
    if not np.isfinite(step) or step <= 0:
        step = 1.0
    half_width = float(opts.get("profile_half_width_q", opts.get("normal_profile_half_width_q", 5.0 * step)))
    if not np.isfinite(half_width) or half_width <= 0:
        half_width = 5.0 * step
    n_samples = int(opts.get("profile_samples", opts.get("normal_profile_samples", 33)))
    n_samples = max(9, n_samples + (n_samples + 1) % 2)
    offsets_requested = np.linspace(-half_width, half_width, n_samples, dtype=float)
    if np.all(np.isfinite(inverse_j)):
        displacement = inverse_j @ np.asarray([normal_x, normal_y], dtype=float)
        sample_x = px + displacement[0] * offsets_requested
        sample_y = py + displacement[1] * offsets_requested
    else:
        sample_x = np.full_like(offsets_requested, px)
        sample_y = np.full_like(offsets_requested, py)
    raw = _sample(data, sample_y, sample_x, order=1)
    support = _sample(valid.astype(float), sample_y, sample_x, order=1)
    raw[(support < 0.5) | (sample_y < 0) | (sample_y > data.shape[0] - 1) | (sample_x < 0) | (sample_x > data.shape[1] - 1)] = np.nan
    sampled_qx = _sample(qx, sample_y, sample_x, order=1)
    sampled_qy = _sample(qy, sample_y, sample_x, order=1)
    if np.isfinite(qx0) and np.isfinite(qy0):
        offsets_q = (sampled_qx - qx0) * normal_x + (sampled_qy - qy0) * normal_y
        offsets_q[~np.isfinite(offsets_q)] = offsets_requested[~np.isfinite(offsets_q)]
    else:
        offsets_q = offsets_requested.copy()
    order = np.argsort(offsets_q)
    offsets_q = np.asarray(offsets_q[order], dtype=float)
    raw = np.asarray(raw[order], dtype=float)
    support = np.asarray(support[order], dtype=float)
    sample_x = np.asarray(sample_x[order], dtype=float)
    sample_y = np.asarray(sample_y[order], dtype=float)
    finite = np.isfinite(offsets_q) & np.isfinite(raw)
    if np.count_nonzero(finite) < 5:
        return {
            "point_id": _field(point, ("point_id", "id"), ""),
            "valid": False,
            "model": "none",
            "reason": "masked_or_outside_profile",
            "uncertainty_source": "unavailable_masked_profile",
            "offset_q": offsets_q.tolist(),
            "offset_pixel_x": sample_x.tolist(),
            "offset_pixel_y": sample_y.tolist(),
            "raw_intensity": raw.tolist(),
            "fit_intensity": [float("nan")] * n_samples,
            "background": float("nan"),
            "peaks": [],
            "snr": float("nan"),
            "normal_fwhm_q": float("nan"),
            "localization_sigma_q": float("nan"),
            "normal_qx": normal_x,
            "normal_qy": normal_y,
            "q_normal_step": step,
            "jacobian_determinant": determinant,
        }
    finite_values = raw[finite]
    edge_count = max(2, int(np.ceil(0.2 * finite_values.size)))
    edge_values = np.r_[finite_values[:edge_count], finite_values[-edge_count:]]
    background = float(np.nanpercentile(edge_values, 20.0))
    noise = _robust_noise(edge_values - background)
    if not np.isfinite(noise) or noise <= np.finfo(float).eps:
        noise = _robust_noise(finite_values - background)
    if not np.isfinite(noise) or noise <= np.finfo(float).eps:
        noise = max(float(np.nanstd(finite_values)), np.finfo(float).eps)
    if not bool(opts.get("fit_gaussian", True)):
        signal = np.maximum(finite_values - background, 0.0)
        snr = float(np.nanmax(signal) / max(noise, np.finfo(float).eps)) if signal.size else float("nan")
        fit_values = np.full_like(raw, background, dtype=float)
        fit_values[~finite] = np.nan
        if cancel_event is not None:
            from .cancellation import raise_if_cancelled

            raise_if_cancelled(cancel_event, "ridge-profile:deferred")
        return {
            "point_id": _field(point, ("point_id", "id"), ""),
            "valid": True,
            "model": "deferred_candidate_profile",
            "reason": "fit_deferred_rejected_candidate",
            "fit_deferred": True,
            "ambiguous": False,
            "offset_q": offsets_q.tolist(),
            "offset_pixel_x": sample_x.tolist(),
            "offset_pixel_y": sample_y.tolist(),
            "raw_intensity": raw.tolist(),
            "fit_intensity": fit_values.tolist(),
            "background": background,
            "noise": float(noise),
            "peaks": [],
            "peak_count": 0,
            "snr": snr,
            "normal_fwhm_q": float("nan"),
            "localization_sigma_q": float("nan"),
            "uncertainty_source": "not_fit_rejected_candidate",
            "normal_qx": normal_x,
            "normal_qy": normal_y,
            "q_normal_step": step,
            "jacobian_determinant": determinant,
            "support_fraction": float(np.mean(support[finite])) if np.any(finite) else 0.0,
        }
    fit_result = _fit_profile_models(
        offsets_q,
        raw,
        noise=noise,
        step=step,
        delta_bic=float(opts.get("profile_delta_bic", opts.get("delta_bic", 4.0))),
    )
    fit_values = np.asarray(fit_result.get("fit", np.full_like(offsets_q, np.nan)), dtype=float)
    fit_values = fit_values.tolist()
    peaks = [dict(peak) for peak in fit_result.get("peaks", [])]
    primary = None
    if peaks:
        primary = min(peaks, key=lambda item: abs(float(item.get("center_q", 0.0))))
    snr = float(primary.get("snr", float("nan"))) if primary else float("nan")
    fwhm = float(primary.get("fwhm_q", float("nan"))) if primary else float("nan")
    localization_sigma = float(primary.get("center_sigma_q", float("nan"))) if primary else float("nan")
    if not np.isfinite(localization_sigma):
        localization_sigma = float(fwhm / max(2.354820045 * np.sqrt(max(snr, 1.0)), 1.0)) if np.isfinite(fwhm) else float("nan")
    sampling_sigma = float(step / np.sqrt(12.0)) if np.isfinite(step) and step > 0 else float("nan")
    if np.isfinite(sampling_sigma) and np.isfinite(snr) and snr > 0:
        sampling_sigma *= float(np.sqrt(max(1.0, 5.0 / snr)))
    empirical_sigma = max(
        value
        for value in (localization_sigma, sampling_sigma)
        if np.isfinite(value)
    ) if any(np.isfinite(value) for value in (localization_sigma, sampling_sigma)) else float("nan")
    peak_center_q = float(primary.get("center_q", float("nan"))) if primary else float("nan")
    peak_center_pixel_x = float(np.interp(peak_center_q, offsets_q, sample_x)) if np.isfinite(peak_center_q) else float("nan")
    peak_center_pixel_y = float(np.interp(peak_center_q, offsets_q, sample_y)) if np.isfinite(peak_center_q) else float("nan")
    peak_center_qx = float(np.interp(peak_center_q, offsets_q, sampled_qx[order])) if np.isfinite(peak_center_q) else float("nan")
    peak_center_qy = float(np.interp(peak_center_q, offsets_q, sampled_qy[order])) if np.isfinite(peak_center_q) else float("nan")
    if cancel_event is not None:
        from .cancellation import raise_if_cancelled

        raise_if_cancelled(cancel_event, "ridge-profile:complete")
    return {
        "point_id": _field(point, ("point_id", "id"), ""),
        "valid": True,
        "model": str(fit_result.get("model", "none")),
        "reason": "accepted" if peaks else "no_gaussian_peak",
        "ambiguous": bool(fit_result.get("ambiguous", False)),
        "offset_q": offsets_q.tolist(),
        "offset_pixel_x": sample_x.tolist(),
        "offset_pixel_y": sample_y.tolist(),
        "raw_intensity": raw.tolist(),
        "fit_intensity": fit_values,
        "background": background,
        "noise": float(noise),
        "peaks": peaks,
        "peak_count": int(len(peaks)),
        "snr": snr,
        "normal_fwhm_q": fwhm,
        "uncertainty_source": (
            "empirical_profile_sampling_plus_curve_fit"
            if str(fit_result.get("uncertainty_source", "")) == "curve_fit_covariance"
            else "empirical_profile_sampling_plus_residual"
        ),
        "fit_uncertainty_source": str(fit_result.get("uncertainty_source", "unknown")),
        "fit_center_sigma_q": localization_sigma,
        "sampling_sigma_q": sampling_sigma,
        "peak_center_q": peak_center_q,
        "peak_center_pixel_x": peak_center_pixel_x,
        "peak_center_pixel_y": peak_center_pixel_y,
        "peak_center_qx": peak_center_qx,
        "peak_center_qy": peak_center_qy,
        "normal_qx": normal_x,
        "normal_qy": normal_y,
        "q_normal_step": step,
        "jacobian_determinant": determinant,
        "support_fraction": float(np.mean(support[finite])) if np.any(finite) else 0.0,
        "localization_sigma_q": empirical_sigma,
    }


def wang2007_vertical_slice_check(
    image: Any,
    qmap: Any,
    q_window: Any = None,
    *,
    mask: Any = None,
    reference_axis_deg: float = 0.0,
    options: Any = None,
    cancel_event: Any = None,
) -> dict[str, Any]:
    """Run a reference-aligned vertical slice double-Gaussian diagnostic.

    The check follows the *style* of a vertical/reference-aligned slice in
    Wang et al. (2007): intensities are aggregated in a narrow strip and a
    one- or two-Gaussian description is compared.  It is applicable only when
    the calibrated q map and strip have enough finite support.  The result is
    explicitly diagnostic-only and is never promoted to a ridge truth label.
    """

    if cancel_event is not None:
        from .cancellation import raise_if_cancelled

        raise_if_cancelled(cancel_event, "wang2007-slice:start")
    opts = _normalise_options(options)
    data, image_mask = _as_image(image)
    qx, qy, q, qmap_mask = _as_qmap(qmap, data.shape)
    invalid = np.zeros(data.shape, dtype=bool)
    if image_mask is not None:
        invalid |= image_mask
    if qmap_mask is not None:
        invalid |= qmap_mask
    if mask is not None:
        try:
            explicit = np.asarray(np.broadcast_to(np.asarray(mask, dtype=bool), data.shape), dtype=bool)
        except ValueError as exc:
            raise ValueError("slice mask must broadcast to image shape") from exc
        invalid |= explicit
    valid = np.isfinite(data) & np.isfinite(qx) & np.isfinite(qy) & ~invalid
    q_min, q_max = _parse_q_window(q_window, q)
    theta = np.deg2rad(float(reference_axis_deg))
    axis_x, axis_y = np.cos(theta), np.sin(theta)
    vertical_x, vertical_y = -axis_y, axis_x
    center = opts.get("center_q", opts.get("center_hint", (0.0, 0.0)))
    try:
        center_x, center_y = float(center[0]), float(center[1])
    except Exception:
        center_x = center_y = 0.0
    rel_x, rel_y = qx - center_x, qy - center_y
    along = rel_x * vertical_x + rel_y * vertical_y
    across = rel_x * axis_x + rel_y * axis_y
    finite_steps = np.r_[np.abs(np.diff(np.nanmedian(qx, axis=0))), np.abs(np.diff(np.nanmedian(qy, axis=1)))]
    finite_steps = finite_steps[np.isfinite(finite_steps) & (finite_steps > np.finfo(float).eps)]
    step = float(np.median(finite_steps)) if finite_steps.size else 1.0
    strip_half = float(opts.get("wang_strip_half_width_q", 2.0 * step))
    n_bins = max(17, int(opts.get("wang_slice_bins", 96)))
    domain = valid & (q >= q_min) & (q <= q_max) & (np.abs(across) <= strip_half)
    reasons: list[str] = []
    if not np.any(domain):
        reasons.append("no_finite_reference_aligned_strip")
    if np.count_nonzero(domain) < int(opts.get("wang_min_pixels", 32)):
        reasons.append("insufficient_strip_support")
    if not np.isfinite(q_min) or not np.isfinite(q_max) or q_max <= q_min:
        reasons.append("invalid_q_window")
    if reasons:
        return {
            "method": "wang2007_vertical_reference_slice",
            "method_version": METHOD_VERSION,
            "applicable": False,
            "applicability": {"applicable": False, "reasons": reasons},
            "diagnostic_only": True,
            "used_for_acceptance": False,
            "reference_axis_deg": float(reference_axis_deg),
            "strip_half_width_q": strip_half,
            "peaks": [],
            "model": "none",
            "offset_q": [],
            "raw_intensity": [],
            "scope_note": "single reference-aligned diagnostic slice; not a series-parallel-z reconstruction and not a ridge truth source",
        }
    edges = np.linspace(float(np.nanmin(along[domain])), float(np.nanmax(along[domain])), n_bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    sums = np.zeros(n_bins, dtype=float)
    counts = np.zeros(n_bins, dtype=int)
    indices = np.searchsorted(edges, along[domain], side="right") - 1
    values = data[domain]
    for index, value in zip(indices.tolist(), values.tolist()):
        if 0 <= int(index) < n_bins and np.isfinite(value):
            sums[int(index)] += float(value)
            counts[int(index)] += 1
    profile = np.divide(sums, counts, out=np.full(n_bins, np.nan, dtype=float), where=counts > 0)
    geometry_domain = np.isfinite(q) & np.isfinite(qx) & np.isfinite(qy) & (q >= q_min) & (q <= q_max) & (np.abs(across) <= strip_half)
    geometry_indices = np.searchsorted(edges, along[geometry_domain], side="right") - 1
    geometry_counts = np.bincount(geometry_indices[(geometry_indices >= 0) & (geometry_indices < n_bins)], minlength=n_bins)
    noise = _robust_noise(profile[counts > 0])
    if not np.isfinite(noise) or noise <= np.finfo(float).eps:
        noise = max(float(np.nanstd(profile[counts > 0])), np.finfo(float).eps)
    fit = _fit_profile_models(
        centres,
        profile,
        noise=noise,
        step=max(float(np.median(np.diff(centres))), step),
        delta_bic=float(opts.get("wang_delta_bic", opts.get("profile_delta_bic", 4.0))),
    )
    finite_bins = counts > 0
    occupied = np.flatnonzero(finite_bins)
    if occupied.size:
        occupied_span = finite_bins[int(occupied[0]) : int(occupied[-1]) + 1]
        interior_gap_fraction = float(np.count_nonzero(~occupied_span) / max(1, occupied_span.size))
    else:
        interior_gap_fraction = 1.0
    mask_gap_bins = (geometry_counts > 0) & (counts == 0)
    mask_gap_fraction = float(np.count_nonzero(mask_gap_bins) / max(1, np.count_nonzero(geometry_counts > 0)))
    if mask_gap_fraction > float(opts.get("wang_max_gap_fraction", 0.15)):
        reasons.append("reference_slice_support_gap")
    if cancel_event is not None:
        from .cancellation import raise_if_cancelled

        raise_if_cancelled(cancel_event, "wang2007-slice:complete")
    applicable = bool(np.count_nonzero(np.isfinite(profile)) >= 5 and mask_gap_fraction <= float(opts.get("wang_max_gap_fraction", 0.15)))
    if not applicable:
        reasons.append("insufficient_binned_profile")
    return {
        "method": "wang2007_vertical_reference_slice",
        "method_version": METHOD_VERSION,
        "applicable": applicable,
        "applicability": {"applicable": applicable, "reasons": reasons},
        "diagnostic_only": True,
        "used_for_acceptance": False,
        "reference_axis_deg": float(reference_axis_deg),
        "center_q": [center_x, center_y],
        "strip_half_width_q": strip_half,
        "q_window": [q_min, q_max],
        "n_pixels": int(np.count_nonzero(domain)),
        "offset_q": centres.tolist(),
        "raw_intensity": profile.tolist(),
        "counts": counts.tolist(),
        "fit_intensity": np.asarray(fit.get("fit", np.full_like(centres, np.nan)), dtype=float).tolist(),
        "background": float(np.nanpercentile(profile[np.isfinite(profile)], 10.0)) if np.any(np.isfinite(profile)) else float("nan"),
        "noise": float(noise),
        "model": str(fit.get("model", "none")),
        "peaks": [dict(peak) for peak in fit.get("peaks", [])],
        "ambiguous": bool(fit.get("ambiguous", False)),
        "uncertainty_source": str(fit.get("uncertainty_source", "unknown")),
        "support_gap_fraction": interior_gap_fraction,
        "mask_gap_fraction": mask_gap_fraction,
        "scope_note": "single reference-aligned diagnostic slice; not a series-parallel-z reconstruction and not a ridge truth source",
    }


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


# Friendly aliases for callers that use the descriptive names from the
# protocol.  All aliases keep the explicit diagnostic-only semantics.
normal_profile = extract_normal_profile
measure_normal_profile = extract_normal_profile
wang2007_style_double_gaussian_check = wang2007_vertical_slice_check
vertical_reference_slice_check = wang2007_vertical_slice_check


__all__ = [
    "METHOD_VERSION",
    "build_profile_context",
    "extract_normal_profile",
    "normal_profile",
    "measure_normal_profile",
    "wang2007_vertical_slice_check",
    "wang2007_style_double_gaussian_check",
    "vertical_reference_slice_check",
]
