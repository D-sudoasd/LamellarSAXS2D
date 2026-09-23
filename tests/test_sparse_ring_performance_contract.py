from __future__ import annotations

import numpy as np

from butterfly_saxs import butterfly_ridge as ridge
from butterfly_saxs.butterfly_ridge import (
    _fill_sparse_first_order_ring,
    _normalise_options,
    _prepare_sparse_first_order_samples,
    _sector_first_order_peak,
)


_SECTORS = tuple(
    30.0 + quadrant + offset
    for quadrant in (0.0, 90.0, 180.0, 270.0)
    for offset in (30.0, 45.0, 60.0, 75.0)
)


def _legacy_sector_first_order_peak(
    qx: np.ndarray,
    qy: np.ndarray,
    q: np.ndarray,
    intensity: np.ndarray,
    valid: np.ndarray,
    *,
    sector_deg: float,
    halfwidth_deg: float,
    hint: float,
) -> dict[str, float] | None:
    """Reference implementation of the pre-optimization sector scan."""

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
    median_qx = float(np.nanmedian(qx[in_bin]))
    median_qy = float(np.nanmedian(qy[in_bin]))
    if not (np.isfinite(median_qx) and np.isfinite(median_qy)):
        return None
    rows, cols = np.nonzero(in_bin)
    nearest = int(np.argmin((qx[in_bin] - median_qx) ** 2 + (qy[in_bin] - median_qy) ** 2))
    return {
        "qx": median_qx,
        "qy": median_qy,
        "q_star": q_star,
        "intensity": peak,
        "contrast": peak / baseline,
        "pixel_x": float(cols[nearest]),
        "pixel_y": float(rows[nearest]),
        "sector_deg": float(sector_deg),
    }


def _ring_case(size: int = 192) -> tuple[np.ndarray, ...]:
    axis = np.linspace(-1.0, 1.0, size)
    qx, qy = np.meshgrid(axis, axis)
    q = np.hypot(qx, qy)
    image = 0.2 + 4.0 * np.exp(-0.5 * ((q - 0.50) / 0.03) ** 2)
    image += 0.02 * np.sin(5.0 * qx) * np.cos(3.0 * qy)
    valid = np.isfinite(image)
    valid[::17, 3::19] = False
    image[5::23, 7::29] = np.nan
    q[11::31, 13::27] = np.nan
    return qx, qy, q, image, valid


def _sector_result(result: dict[str, float] | None) -> tuple[tuple[str, float], ...] | None:
    if result is None:
        return None
    return tuple(sorted((key, float(value)) for key, value in result.items()))


def test_prepared_sector_scan_is_numerically_equivalent_to_legacy_scan() -> None:
    qx, qy, q, image, valid = _ring_case()
    prepared = _prepare_sparse_first_order_samples(qx, qy, q, image, valid, hint=0.50)
    for sector in _SECTORS:
        legacy = _legacy_sector_first_order_peak(
            qx, qy, q, image, valid, sector_deg=sector, halfwidth_deg=7.5, hint=0.50
        )
        optimized = _sector_first_order_peak(
            qx,
            qy,
            q,
            image,
            valid,
            sector_deg=sector,
            halfwidth_deg=7.5,
            hint=0.50,
            prepared=prepared,
        )
        assert _sector_result(optimized) == _sector_result(legacy)
        direct = _sector_first_order_peak(
            qx,
            qy,
            q,
            image,
            valid,
            sector_deg=sector,
            halfwidth_deg=7.5,
            hint=0.50,
        )
        assert _sector_result(direct) == _sector_result(legacy)


def test_prepared_scan_excludes_masked_and_nonfinite_pixels_without_source_mutation() -> None:
    qx, qy, q, image, valid = _ring_case()
    originals = tuple(array.copy() for array in (qx, qy, q, image, valid))
    prepared = _prepare_sparse_first_order_samples(qx, qy, q, image, valid, hint=0.50)

    assert np.all(np.isfinite(prepared["q"]))
    assert np.all(np.isfinite(prepared["intensity"]))
    assert prepared["q"].size < np.count_nonzero(np.isfinite(q))
    for current, original in zip((qx, qy, q, image, valid), originals):
        assert np.array_equal(current, original, equal_nan=True)


def test_highest_q_edge_is_kept_in_last_profile_bin() -> None:
    hint = 0.50
    q_lo, q_hi = 0.70 * hint, 1.45 * hint
    edges = np.linspace(q_lo, q_hi, 9)
    radial_values = np.concatenate(
        [
            np.full(3, edges[0] + 0.005),
            np.full(3, edges[1] + 0.005),
            np.full(3, edges[2] + 0.005),
            np.full(12, q_hi),
        ]
    )
    angle = np.deg2rad(45.0)
    qx = (radial_values * np.cos(angle))[:, None]
    qy = (radial_values * np.sin(angle))[:, None]
    q = radial_values[:, None]
    image = np.concatenate([np.ones(9), np.full(12, 10.0)])[:, None]
    valid = np.ones_like(q, dtype=bool)

    result = _sector_first_order_peak(
        qx,
        qy,
        q,
        image,
        valid,
        sector_deg=45.0,
        halfwidth_deg=7.5,
        hint=hint,
    )

    assert result is not None
    assert result["intensity"] == 10.0
    assert result["q_star"] == (edges[-2] + edges[-1]) / 2.0
    assert result["qx"] == q_hi * np.cos(angle)
    assert result["qy"] == q_hi * np.sin(angle)


def test_fill_path_keeps_original_arrays_unchanged() -> None:
    qx, qy, q, image, valid = _ring_case()
    originals = tuple(array.copy() for array in (qx, qy, q, image, valid))
    points = [
        {"qx": 0.50, "qy": 0.02, "accepted": True, "branch_id": 0, "side": "upper"},
        {"qx": -0.50, "qy": -0.02, "accepted": True, "branch_id": 1, "side": "lower"},
    ]
    _fill_sparse_first_order_ring(
        points,
        qx=qx,
        qy=qy,
        q=q,
        intensity=image,
        valid=valid,
        hint=0.50,
        options=_normalise_options({"reference_axis_deg": 0.0, "run_wang_check": False}),
        q_step=2.0 / q.shape[0],
        signature="no-source-mutation",
    )
    for current, original in zip((qx, qy, q, image, valid), originals):
        assert np.array_equal(current, original, equal_nan=True)


def test_fill_path_prepares_sparse_samples_once(monkeypatch) -> None:
    qx, qy, q, image, valid = _ring_case()
    original_prepare = ridge._prepare_sparse_first_order_samples
    calls = 0

    def counted_prepare(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_prepare(*args, **kwargs)

    monkeypatch.setattr(ridge, "_prepare_sparse_first_order_samples", counted_prepare)
    points = [
        {"qx": 0.50, "qy": 0.02, "accepted": True, "branch_id": 0, "side": "upper"},
        {"qx": -0.50, "qy": -0.02, "accepted": True, "branch_id": 1, "side": "lower"},
    ]
    _fill_sparse_first_order_ring(
        points,
        qx=qx,
        qy=qy,
        q=q,
        intensity=image,
        valid=valid,
        hint=0.50,
        options=_normalise_options({"reference_axis_deg": 0.0, "run_wang_check": False}),
        q_step=2.0 / q.shape[0],
        signature="one-preparation",
    )

    assert calls == 1
