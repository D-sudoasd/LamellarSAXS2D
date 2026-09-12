from __future__ import annotations

import numpy as np
import pytest

import butterfly_saxs.butterfly as butterfly_module
import butterfly_saxs.butterfly_diagnostics as butterfly_diagnostics
import butterfly_saxs.butterfly_ridge as butterfly_ridge
import butterfly_saxs.butterfly_uncertainty as butterfly_uncertainty
from butterfly_saxs.butterfly_ridge import _RadialHintGeometryCache, _first_order_q_hint


def _case() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    yy, xx = np.indices((121, 135), dtype=float)
    qx = (xx - 67.0) * 0.006
    qy = (yy - 60.0) * 0.007
    q = np.hypot(qx, qy)
    intensity = (
        0.2
        + 10.0 * np.exp(-0.5 * ((q - 0.19) / 0.012) ** 2)
        + 3.0 * np.exp(-0.5 * ((q - 0.39) / 0.018) ** 2)
    )
    intensity += 0.02 * np.sin(xx * 0.31) * np.cos(yy * 0.27)
    return qx, qy, q, intensity


def _assert_same_summary(left: dict, right: dict) -> None:
    assert left.keys() == right.keys()
    for key in left:
        if isinstance(left[key], float) and isinstance(right[key], float):
            assert left[key] == pytest.approx(right[key], rel=1e-12, abs=1e-12)
        else:
            assert left[key] == right[key]


def test_cache_reuses_only_geometry_while_recomputing_intensity_statistics() -> None:
    _qx, _qy, q, intensity = _case()
    valid = np.ones(q.shape, dtype=bool)
    cache = _RadialHintGeometryCache()

    baseline = _first_order_q_hint(q, intensity, valid, 0.05, 0.50, geometry_cache=cache)
    cached_labels = cache._bin_indices
    hot_pixel = np.unravel_index(np.argmin(np.abs(q - 0.19)), q.shape)
    changed_intensity = intensity.copy()
    changed_intensity[hot_pixel] += 500.0

    cached = _first_order_q_hint(q, changed_intensity, valid, 0.05, 0.50, geometry_cache=cache)
    uncached = _first_order_q_hint(q, changed_intensity, valid, 0.05, 0.50)

    assert cache._bin_indices is cached_labels
    _assert_same_summary(cached, uncached)
    assert cached != baseline


def test_cached_bin_profile_matches_histogram_reference() -> None:
    _qx, _qy, q, intensity = _case()
    q_min, q_max, n_bins = 0.05, 0.50, 96
    valid = np.isfinite(intensity)
    selected = valid & np.isfinite(q) & (q >= q_min) & (q <= q_max)
    radii = q[selected]
    weights = np.clip(intensity[selected], 0.0, None)
    edges = np.linspace(q_min, q_max, n_bins + 1)

    labels, counts = _RadialHintGeometryCache().get(radii, selected, edges)
    cached_sums = np.bincount(labels, weights=weights, minlength=n_bins)
    reference_sums, _ = np.histogram(radii, bins=edges, weights=weights)
    reference_counts, _ = np.histogram(radii, bins=edges)

    np.testing.assert_array_equal(counts, reference_counts)
    np.testing.assert_allclose(cached_sums, reference_sums, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("change", ["nan", "mask", "qmap", "center", "qcal", "window"])
def test_cache_rebuilds_for_changed_radial_support_or_calibration(change: str) -> None:
    qx, qy, q, intensity = _case()
    valid = np.ones(q.shape, dtype=bool)
    q_min, q_max = 0.05, 0.50
    changed_q = q
    changed_image = intensity
    changed_valid = valid

    if change == "nan":
        changed_image = intensity.copy()
        changed_image[60, 80] = np.nan
    elif change == "mask":
        changed_valid = valid.copy()
        changed_valid[20:35, 85:100] = False
    elif change == "qmap":
        changed_q = q * 1.013
    elif change == "center":
        changed_q = np.hypot(qx - 0.0015, qy + 0.0008)
    elif change == "qcal":
        changed_q = q * 1.02
    elif change == "window":
        q_min = 0.06

    cache = _RadialHintGeometryCache()
    _first_order_q_hint(q, intensity, valid, 0.05, 0.50, geometry_cache=cache)
    previous_labels = cache._bin_indices

    cached = _first_order_q_hint(
        changed_q,
        changed_image,
        changed_valid,
        q_min,
        q_max,
        geometry_cache=cache,
    )
    uncached = _first_order_q_hint(
        changed_q, changed_image, changed_valid, q_min, q_max
    )

    assert cache._bin_indices is not previous_labels
    _assert_same_summary(cached, uncached)


def test_cache_does_not_retain_geometry_over_its_memory_budget() -> None:
    _qx, _qy, q, intensity = _case()
    cache = _RadialHintGeometryCache(max_bytes=0)

    _first_order_q_hint(q, intensity, np.ones(q.shape, dtype=bool), 0.05, 0.50, geometry_cache=cache)

    assert cache._bin_indices is None
    assert cache._radii is None
    assert not any(isinstance(value, np.ndarray) for value in vars(cache).values())


def test_cache_budget_boundary_accounts_for_all_retained_arrays() -> None:
    _qx, _qy, q, intensity = _case()
    valid = np.ones(q.shape, dtype=bool)
    cache = _RadialHintGeometryCache()
    _first_order_q_hint(q, intensity, valid, .05, .50, geometry_cache=cache)
    retained = [value for value in vars(cache).values() if isinstance(value, np.ndarray)]
    assert retained and all(not value.flags.writeable for value in retained)
    required = sum(value.nbytes for value in retained)
    exact = _RadialHintGeometryCache(max_bytes=required)
    too_small = _RadialHintGeometryCache(max_bytes=required - 1)
    _first_order_q_hint(q, intensity, valid, .05, .50, geometry_cache=exact)
    _first_order_q_hint(q, intensity, valid, .05, .50, geometry_cache=too_small)
    assert sum(value.nbytes for value in vars(exact).values() if isinstance(value, np.ndarray)) == required
    assert not any(isinstance(value, np.ndarray) for value in vars(too_small).values())


@pytest.mark.parametrize(
    ("perturbation", "reuse_cache"),
    [
        ({"center_delta_px": [0.0, 0.0], "qcal_scale": 1.0}, True),
        ({"center_delta_px": [0.25, 0.0], "qcal_scale": 1.0}, False),
        ({"center_delta_px": [0.0, 0.0], "qcal_scale": 1.02}, False),
    ],
)
def test_analysis_threads_cache_only_when_resample_qmap_is_unchanged(
    monkeypatch: pytest.MonkeyPatch,
    perturbation: dict[str, object],
    reuse_cache: bool,
) -> None:
    cache_args: list[object] = []

    def trace_stub(*_args, **kwargs):
        cache_args.append(kwargs.get("radial_hint_cache"))
        return {"points": [], "arcs": [], "profiles": {}, "diagnostics": {}}

    def resample_stub(image, _qmap, *, refit, **_kwargs):
        refit(
            image,
            {
                "qmap": {"q_unit": "A^-1"},
                "mask": None,
                "qmap_perturbation": perturbation,
            },
        )
        return {"status": "completed", "intervals": {}}

    monkeypatch.setattr(butterfly_ridge, "trace_butterfly_ridges", trace_stub)
    monkeypatch.setattr(butterfly_uncertainty, "resample_butterfly", resample_stub)
    monkeypatch.setattr(butterfly_module, "_fit_trace", lambda *_args, **_kwargs: {"success": True})
    monkeypatch.setattr(butterfly_module, "evaluate_arc_evidence", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(butterfly_diagnostics, "profile_residuals", lambda _profiles: [])
    monkeypatch.setattr(butterfly_diagnostics, "ellipse_local_views", lambda *_args: [])

    butterfly_module.analyze_butterfly(
        np.ones((8, 8)),
        {"qx": np.zeros((8, 8)), "qy": np.zeros((8, 8)), "q_unit": "A^-1"},
        (0.0, 1.0),
        options={"stage": "evaluate", "resamples": 1, "sensitivity": False},
    )

    assert len(cache_args) == 2
    assert cache_args[0] is not None
    assert (cache_args[1] is cache_args[0]) is reuse_cache
