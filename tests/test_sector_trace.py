from __future__ import annotations

import numpy as np
import pytest

from butterfly_saxs.butterfly import analyze_butterfly, measure_butterfly_observables
from butterfly_saxs.butterfly_settings import normalize_butterfly_settings
from butterfly_saxs.sector_trace import trace_butterfly_sector_peaks


def _ring():
    axis = np.linspace(-0.85, 0.85, 161)
    qx, qy = np.meshgrid(axis, axis)
    q = np.hypot(qx, qy)
    image = 3.0 + 50.0 * np.exp(-0.5 * ((q - 0.42) / 0.04) ** 2)
    return image, {"qx": qx, "qy": qy, "q": q, "q_unit": "nm^-1"}


def test_sector_trace_is_profile_first_and_keeps_missing_sectors():
    image, qmap = _ring()
    chi = np.mod(np.degrees(np.arctan2(qmap["qy"], qmap["qx"])), 360)
    mask = (chi >= 50) & (chi <= 110)
    original = image.copy()
    result = trace_butterfly_sector_peaks(
        image, qmap, (.15, .7), mask=mask,
        options={"sector_width_deg": 10, "sector_step_deg": 10},
    )
    measurement = result["sector_peaks"]
    assert len(measurement["sectors"]) == 36
    assert 8 < len(result["points"]) <= 36
    assert len({p["sector_index"] for p in result["points"]}) == len(result["points"])
    assert all(abs(p["q_star"] - .42) < .025 for p in result["points"])
    assert not any(60 <= p["sector_center_deg"] <= 100 for p in result["points"])
    missing = next(s for s in measurement["sectors"] if s["angle_deg"] == 80)
    assert missing["selected_peak"] is None
    profile = result["profiles"][missing["point_id"]]
    assert profile["profile_axis"] == "radial"
    assert profile["fit_intensity"] == []
    assert np.count_nonzero(profile["counts"]) == 0
    for point in result["points"]:
        assert not mask[int(point["pixel_y"]), int(point["pixel_x"])]
        assert np.isnan(point["normal_fwhm_q"])
        assert np.isnan(point["localization_sigma_q"])
        assert not point["scale_stable"]
    np.testing.assert_array_equal(image, original)


def test_sector_pipeline_trace_is_separately_versioned_and_does_not_fit():
    image, qmap = _ring()
    result = analyze_butterfly(image, qmap, (.15, .7), options={
        "stage": "trace", "trace_method": "radial_sector", "sector_step_deg": 10,
    })
    assert result["method_version"].startswith("butterfly-radial-sector-")
    assert result["quality"]["status"] == "NOT_EVALUATED"
    assert result["candidate_fit"]["success"] is False
    assert result["diagnostics"]["first_order_q_hint"]["q_star"] is None
    assert "sector_peaks" in result


def test_sector_median_summary_is_explicit_for_trace_and_evaluate():
    image, qmap = _ring()
    for stage in ("trace", "evaluate"):
        result = analyze_butterfly(image, qmap, (.15, .7), options={
            "stage": stage,
            "trace_method": "radial_sector",
            "sector_step_deg": 10,
            "resamples": 0,
            "sensitivity": False,
        })
        summary = result["measurement_summary"]
        assert result["q_star_sector_median"] == pytest.approx(
            summary["q_star_sector_median"]
        )
        assert summary["q_star_sector_median"] == pytest.approx(.42, abs=.025)
        assert summary["apparent_period_from_sector_median_nm"] == pytest.approx(
            2 * np.pi / summary["q_star_sector_median"]
        )
        assert summary["q_star_source"] == "selected_sector_peak_median"
        assert summary["peak_order"] == "unassigned"
        assert "overlapping sectors are correlated" in summary["interpretation"]
        assert result["candidate_fit"].get("q_star_source") != "unindexed_sector_peak_median"

    observables = measure_butterfly_observables(
        image,
        qmap,
        (.15, .7),
        options={
            "stage": "trace",
            "trace_method": "radial_sector",
            "sector_step_deg": 10,
            "companion_observables": False,
        },
    )
    assert observables.ridge["flags"] == ["butterfly-radial-sector-v1.0"]


def test_sector_median_does_not_report_nm_for_pixel_q():
    image, qmap = _ring()
    qmap["q_unit"] = "pixel-q"
    result = analyze_butterfly(image, qmap, (.15, .7), options={
        "stage": "trace", "trace_method": "radial_sector", "sector_step_deg": 10,
    })
    summary = result["measurement_summary"]
    assert summary["q_star_sector_median"] is not None
    assert summary["q_star_sector_median_unit"] == "pixel-q"
    assert summary["apparent_period_from_sector_median_nm"] is None


def test_excluding_one_sector_does_not_create_or_relocate_peaks():
    image, qmap = _ring()
    options = {"sector_step_deg": 10}
    baseline = trace_butterfly_sector_peaks(image, qmap, (.15, .7), options=options)
    chosen = next(p for p in baseline["points"] if p["accepted"])
    edited = trace_butterfly_sector_peaks(image, qmap, (.15, .7), options=options,
                                         edits=[{"type": "exclude_point", "point_id": chosen["point_id"]}])
    assert [(p["point_id"], p["qx"], p["qy"]) for p in baseline["points"]] == [
        (p["point_id"], p["qx"], p["qy"]) for p in edited["points"]]
    point = next(p for p in edited["points"] if p["point_id"] == chosen["point_id"])
    assert not point["accepted"] and not point["valid"]


def test_existing_recipe_keeps_curvature_and_invalid_method_is_rejected():
    assert normalize_butterfly_settings({})["trace_method"] == "curvature"
    with pytest.raises(ValueError, match="trace_method"):
        normalize_butterfly_settings({"trace_method": "brightest_pixel"})
    for value in (0, float("nan"), True):
        with pytest.raises(ValueError, match="sector_width_deg"):
            normalize_butterfly_settings({"sector_width_deg": value})


def test_cli_explicit_sector_method_selects_shared_workflow_and_rejects_conflict():
    from butterfly_saxs.cli import _analysis_overrides, build_parser

    parser = build_parser()
    args = parser.parse_args(["analyze", "frame.edf", "--butterfly-trace-method", "radial_sector",
                              "--sector-width", "10", "--sector-step", "5"])
    analysis = _analysis_overrides(args)
    assert analysis["ridge_method"] == "butterfly_curvature"
    assert analysis["butterfly"]["trace_method"] == "radial_sector"
    assert analysis["butterfly"]["sector_width_deg"] == 10
    args = parser.parse_args(["analyze", "frame.edf", "--butterfly-trace-method", "radial_sector",
                              "--ridge-method", "radial_peak"])
    with pytest.raises(ValueError, match="butterfly workflow"):
        _analysis_overrides(args)
