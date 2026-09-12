from __future__ import annotations

import numpy as np
import pytest

import butterfly_saxs.ridge_inputs as ridge_inputs
from butterfly_saxs.butterfly_ridge import _apply_edits, trace_butterfly_ridges


def _row_polygon(x_limit: float) -> list[list[float]]:
    return [[-x_limit, -0.2], [x_limit, -0.2], [x_limit, 0.2], [-x_limit, 0.2]]


def test_include_restores_prior_polygon_exclusion_but_keeps_input_validity() -> None:
    yy, xx = np.indices((9, 9), dtype=float)
    qx, qy = xx - 4.0, yy - 4.0
    q = np.hypot(qx, qy)
    image = np.ones(q.shape, dtype=float)
    image[4, 2] = np.nan
    image_mask = np.zeros(q.shape, dtype=bool)
    image_mask[4, 3] = True
    qmap_mask = np.zeros(q.shape, dtype=bool)
    qmap_mask[4, 1] = True
    roi_mask = np.zeros(q.shape, dtype=bool)
    roi_mask[4, 7] = True
    external_mask = np.zeros(q.shape, dtype=bool)
    external_mask[4, 5] = True
    q_min, q_max = 0.1, 3.1
    base_valid = (
        np.isfinite(image)
        & np.isfinite(qx)
        & np.isfinite(qy)
        & np.isfinite(q)
        & (q >= q_min)
        & (q <= q_max)
        & ~image_mask
        & ~qmap_mask
        & ~roi_mask
        & ~external_mask
    )
    polygon = _row_polygon(4.2)
    edits = [
        {"type": "exclude_polygon", "points": polygon},
        {"type": "include_polygon", "points": polygon},
    ]

    edited, applied, seeds = _apply_edits(base_valid, qx, qy, edits)

    assert [item["type"] for item in applied] == ["exclude_polygon", "include_polygon"]
    assert seeds == []
    assert edited[4, 6]  # q=(2, 0): include restores the editable exclusion.
    assert not edited[4, 2]  # non-finite image intensity
    assert not edited[4, 3]  # image/detector mask
    assert not edited[4, 1]  # q-map mask
    assert not edited[4, 5]  # explicit external mask
    assert not edited[4, 7]  # user ROI mask
    assert not edited[4, 8]  # q=(4, 0), outside the q-window


def test_trace_include_does_not_hide_external_mask_in_q_window_diagnostic() -> None:
    shape = (9, 9)
    yy, xx = np.indices(shape)
    qx = xx.astype(float) - 4.0
    qy = yy.astype(float) - 4.0
    q = np.hypot(qx, qy)
    image = np.ones(shape, dtype=float)
    external_mask = np.zeros(shape, dtype=bool)
    external_mask[4, 3] = True
    polygon = [[-1.2, -0.2], [-0.8, -0.2], [-0.8, 0.2], [-1.2, 0.2]]
    edits = [
        {"type": "exclude_polygon", "points": polygon},
        {"type": "include_polygon", "points": polygon},
    ]
    original_arrays = [image.copy(), qx.copy(), qy.copy(), q.copy(), external_mask.copy()]

    baseline = trace_butterfly_ridges(
        image,
        {"qx": qx, "qy": qy, "q": q},
        (0.1, 6.0),
        options={"smoothing_scales": (1.0,), "run_wang_check": False},
        mask=external_mask,
    )
    edited = trace_butterfly_ridges(
        image,
        {"qx": qx, "qy": qy, "q": q},
        (0.1, 6.0),
        options={"smoothing_scales": (1.0,), "run_wang_check": False},
        mask=external_mask,
        edits=edits,
    )

    assert baseline["diagnostics"]["mask_fraction_in_q_window"] == pytest.approx(0.0125)
    assert edited["diagnostics"]["mask_fraction_in_q_window"] == pytest.approx(0.0125)
    for original, current in zip(original_arrays, (image, qx, qy, q, external_mask)):
        np.testing.assert_array_equal(current, original)


def _coordinate_maps(kind: str) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[-3.0:4.0, -4.0:5.0]
    if kind == "regular":
        return 0.012 * xx + 0.001 * yy, 0.002 * xx + 0.021 * yy
    if kind == "warped":
        qx = 0.012 * xx + 0.0004 * xx * yy + 0.0002 * yy**2
        qy = 0.021 * yy + 0.0007 * xx * yy - 0.0001 * xx**2
        return qx, qy
    if kind == "reversed":
        return -0.012 * xx + 0.001 * yy, 0.002 * xx - 0.021 * yy
    if kind == "degenerate":
        return np.zeros_like(xx), np.ones_like(yy)
    if kind == "single_pixel_axis":
        x = np.arange(9, dtype=float)[None, :]
        return 0.012 * x, -0.021 * x
    raise AssertionError(f"unexpected map kind: {kind}")


@pytest.mark.parametrize("kind", ["regular", "warped", "reversed", "degenerate", "single_pixel_axis"])
def test_coordinate_derivative_q_step_matches_representative_oracle_without_mutation(kind: str) -> None:
    qx, qy = _coordinate_maps(kind)
    qx_before, qy_before = qx.copy(), qy.copy()
    oracle = ridge_inputs.representative_q_step(qx, qy)

    _derivatives, _inverse, q_step = ridge_inputs.coordinate_derivatives(qx, qy)

    assert q_step == (oracle if oracle is not None else 1.0)
    assert np.array_equal(qx, qx_before, equal_nan=True)
    assert np.array_equal(qy, qy_before, equal_nan=True)


def test_coordinate_derivatives_reuses_gradients_for_representative_step(monkeypatch: pytest.MonkeyPatch) -> None:
    qx, qy = _coordinate_maps("warped")
    oracle = ridge_inputs.representative_q_step(qx, qy)
    original_gradient = ridge_inputs._gradient
    calls = 0

    def counted_gradient(array: np.ndarray, axis: int) -> np.ndarray:
        nonlocal calls
        calls += 1
        return original_gradient(array, axis)

    monkeypatch.setattr(ridge_inputs, "_gradient", counted_gradient)
    _derivatives, _inverse, q_step = ridge_inputs.coordinate_derivatives(qx, qy)

    assert q_step == oracle
    assert calls == 12
