from __future__ import annotations

import numpy as np
import pytest

from butterfly_saxs.annular_trace import trace_butterfly_annuli
from butterfly_saxs.butterfly import analyze_butterfly


def _curved_petals():
    axis = np.linspace(-.85, .85, 241)
    qx, qy = np.meshgrid(axis, axis)
    q = np.hypot(qx, qy)
    chi = np.degrees(np.arctan2(qy, qx)) % 360
    # Four observed lobes following outward-moving angular maxima.
    centre = 62 - 36 * q
    image = np.full(q.shape, 2.)
    for angle in (centre, 180 - centre, 180 + centre, 360 - centre):
        delta = (chi - angle + 180) % 360 - 180
        image += 30 * np.exp(-.5 * (delta / 9) ** 2)
    image += np.random.default_rng(19).normal(0, .04, q.shape)
    return image, {"qx": qx, "qy": qy, "q": q, "q_unit": "nm^-1"}, chi


def test_annuli_trace_four_extended_petals_instead_of_radial_peak_positions():
    image, qmap, _ = _curved_petals()
    result = trace_butterfly_annuli(image, qmap, (.25, .75), options={"annular_radial_bins": 20})
    accepted = [p for p in result["points"] if p["accepted"]]
    assert len(accepted) >= 72
    assert len(result["arcs"]) == 4
    assert len({p["trajectory_id"] for p in accepted}) == 4
    assert all(len(row["selected_peaks"]) == 4 for row in result["annular_peaks"]["annuli"])
    for point in accepted:
        q, angle = point["q_annulus"], point["chi_deg"]
        expected = [62 - 36 * q, 180 - (62 - 36 * q), 180 + (62 - 36 * q), 360 - (62 - 36 * q)]
        assert min(abs((angle - x + 180) % 360 - 180) for x in expected) < 2.
        assert "q_star" not in point
    assert np.ptp([p["q_annulus"] for p in accepted]) > .45
    assert all(profile["profile_axis"] == "azimuthal" for profile in result["profiles"].values())


def test_missing_lobe_and_masked_ring_are_not_synthesized_or_bridged():
    image, qmap, chi = _curved_petals()
    q = qmap["q"]
    mask = ((chi > 0) & (chi < 90)) | ((q > .45) & (q < .55))
    result = trace_butterfly_annuli(image, qmap, (.25, .75), mask=mask,
                                  options={"annular_radial_bins": 20})
    accepted = [p for p in result["points"] if p["accepted"]]
    assert accepted
    assert not any(0 < p["chi_deg"] < 90 or .45 < p["q_annulus"] < .55 for p in accepted)
    for track in {p["trajectory_id"] for p in accepted}:
        positions = [p["q_annulus"] for p in accepted if p["trajectory_id"] == track]
        assert not (min(positions) < .45 and max(positions) > .55)
    for point in accepted:
        assert not mask[int(point["pixel_y"]), int(point["pixel_x"])]


def test_annular_recipe_dispatch_and_sampling_coordinates_do_not_become_periods():
    image, qmap, _ = _curved_petals()
    result = analyze_butterfly(image, qmap, (.25, .75), options={
        "trace_method": "annular_peak", "stage": "evaluate", "resamples": 0,
        "sensitivity": False, "annular_radial_bins": 20,
    }, multistart=1)
    assert result["method_version"] == "butterfly-annular-trajectory-v1.0"
    assert result["candidate_fit"]["q_star_from_arcs"] is None
    assert result["candidate_fit"]["L_from_observed_radius_nm"] is None
    assert result["candidate_fit"]["q_star_source"] == "unavailable_prescribed_annuli"
    assert result["quality"]["scientific_status"] == "NOT_ACCEPTED"
    assert "annular_peaks" in result and "sector_peaks" not in result
    with pytest.raises(ValueError, match="annular_radial_bins"):
        analyze_butterfly(image, qmap, (.25, .75), options={"trace_method":"annular_peak", "annular_radial_bins": True})
