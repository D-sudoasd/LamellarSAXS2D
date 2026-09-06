from __future__ import annotations

import numpy as np
import pytest

from butterfly_saxs.ridge_profiles import (
    extract_normal_profile,
    wang2007_style_double_gaussian_check,
    wang2007_vertical_slice_check,
)


def _qmap(size: int = 121):
    axis = np.linspace(-1.0, 1.0, size)
    qx, qy = np.meshgrid(axis, axis)
    return qx, qy, np.hypot(qx, qy)


def _point(pixel: float, *, point_id: str = "p") -> dict[str, float | str]:
    return {
        "point_id": point_id,
        "qx": pixel,
        "qy": 0.0,
        "pixel_x": 60.0 + pixel * 60.0,
        "pixel_y": 60.0,
        "normal_qx": 1.0,
        "normal_qy": 0.0,
        "q_normal_step": 2.0 / 120.0,
    }


def test_normal_profile_keeps_raw_samples_and_selects_single_gaussian() -> None:
    qx, qy, q = _qmap()
    image = 0.12 + 6.0 * np.exp(-0.5 * ((qx - 0.50) / 0.032) ** 2)
    image += np.random.default_rng(7).normal(0.0, 0.015, image.shape)
    profile = extract_normal_profile(
        image,
        {"qx": qx, "qy": qy, "q": q},
        _point(0.50),
        options={"profile_half_width_q": 0.16, "profile_samples": 41},
    )
    assert profile["valid"] is True
    assert profile["peak_count"] == 1
    assert profile["model"] in {"single_gaussian", "single_gaussian_fallback", "ambiguous_single_double"}
    assert len(profile["offset_q"]) == len(profile["raw_intensity"]) == len(profile["fit_intensity"])
    assert profile["uncertainty_source"] in {
        "empirical_profile_sampling_plus_curve_fit",
        "empirical_profile_sampling_plus_residual",
        "robust_profile_fallback",
    }
    assert profile["fit_uncertainty_source"] in {"curve_fit_covariance", "robust_profile_residual", "robust_profile_fallback"}
    assert np.isfinite(profile["normal_fwhm_q"])
    assert np.isfinite(profile["localization_sigma_q"])


def test_normal_profile_detects_data_driven_double_peak_without_forcing_it() -> None:
    qx, qy, q = _qmap()
    image = (
        0.10
        + 4.0 * np.exp(-0.5 * ((qx - 0.49) / 0.025) ** 2)
        + 3.0 * np.exp(-0.5 * ((qx - 0.57) / 0.022) ** 2)
    )
    profile = extract_normal_profile(
        image,
        {"qx": qx, "qy": qy, "q": q},
        _point(0.49, point_id="double"),
        options={"profile_half_width_q": 0.18, "profile_samples": 51, "profile_delta_bic": 2.0},
    )
    assert profile["valid"] is True
    assert profile["peak_count"] == 2
    assert profile["model"] == "double_gaussian"
    assert len(profile["peaks"]) == 2
    assert profile["ambiguous"] is False


def test_wang_slice_is_explicitly_diagnostic_only_and_has_alias() -> None:
    qx, qy, q = _qmap()
    image = 0.10 + 4.0 * np.exp(-0.5 * ((qy - 0.48) / 0.03) ** 2) + 3.0 * np.exp(-0.5 * ((qy - 0.57) / 0.025) ** 2)
    kwargs = {"mask": None, "reference_axis_deg": 0.0, "options": {"wang_slice_bins": 48}}
    result = wang2007_vertical_slice_check(image, {"qx": qx, "qy": qy, "q": q}, (0.30, 0.80), **kwargs)
    alias = wang2007_style_double_gaussian_check(image, {"qx": qx, "qy": qy, "q": q}, (0.30, 0.80), **kwargs)
    assert result["diagnostic_only"] is True
    assert result["used_for_acceptance"] is False
    assert result["applicability"]["applicable"] is True
    assert result["model"] in {"single_gaussian", "double_gaussian", "ambiguous_single_double", "single_gaussian_fallback"}
    assert alias["method"] == result["method"]


def test_masked_normal_profile_retains_unavailable_evidence() -> None:
    qx, qy, q = _qmap()
    image = 0.1 + 5.0 * np.exp(-0.5 * ((qx - 0.5) / 0.03) ** 2)
    mask = np.ones_like(image, dtype=bool)
    profile = extract_normal_profile(
        image,
        {"qx": qx, "qy": qy, "q": q},
        _point(0.50, point_id="masked"),
        mask=mask,
    )
    assert profile["valid"] is False
    assert profile["peaks"] == []
    assert profile["uncertainty_source"].startswith("unavailable_")


def test_profile_center_reports_observed_shift_and_empirical_uncertainty() -> None:
    qx, qy, q = _qmap()
    step = 2.0 / 120.0
    image = 0.10 + 6.0 * np.exp(-0.5 * ((qx - 0.515) / 0.028) ** 2)
    profile = extract_normal_profile(
        image,
        {"qx": qx, "qy": qy, "q": q},
        _point(0.50, point_id="shifted"),
        options={"profile_half_width_q": 0.16, "profile_samples": 41},
    )
    assert profile["model"] == "single_gaussian"
    assert profile["peak_center_q"] == pytest.approx(0.015, abs=0.004)
    assert profile["peak_center_qx"] == pytest.approx(0.515, abs=0.004)
    assert profile["localization_sigma_q"] >= step / np.sqrt(12.0)
