from __future__ import annotations

import numpy as np
import pytest

from butterfly_saxs.io import LoadedImage


def _input_case() -> tuple[dict[str, object], LoadedImage, dict[str, object], np.ndarray, np.ndarray]:
    shape = (17, 17)
    qy, qx = np.mgrid[-8.0:9.0, -8.0:9.0]
    q = np.hypot(qx, qy)
    valid = np.ones(shape, dtype=bool)
    valid[:, 7:10] = False
    data = np.ones(shape, dtype=float)
    frame_mapping = {"data": data, "valid_mask": valid}
    frame_object = LoadedImage(data.copy(), valid_mask=valid.copy())
    qmap = {"qx": qx, "qy": qy, "q": q, "metadata": {"q_unit": "nm^-1"}}
    return frame_mapping, frame_object, qmap, data, valid


def _expected_mask_fraction(valid: np.ndarray, qmap: dict[str, object], explicit: np.ndarray | None = None) -> float:
    q = np.asarray(qmap["q"], dtype=float)
    in_window = (q >= 1.0) & (q <= 9.0)
    invalid = ~valid
    if explicit is not None:
        invalid |= explicit
    return float(np.count_nonzero(invalid & in_window) / np.count_nonzero(in_window))


@pytest.mark.parametrize("frame_kind", ["mapping", "object"])
@pytest.mark.parametrize("stage", ["trace", "evaluate"])
def test_analyze_butterfly_preserves_frame_valid_mask_for_trace_and_evaluate(
    frame_kind: str, stage: str
) -> None:
    from butterfly_saxs.butterfly import analyze_butterfly

    frame_mapping, frame_object, qmap, _data, valid = _input_case()
    frame = frame_mapping if frame_kind == "mapping" else frame_object
    result = analyze_butterfly(
        frame,
        qmap,
        (1.0, 9.0),
        options={
            "stage": stage,
            "resamples": 0,
            "sensitivity": False,
            "smoothing_scales": (1.0,),
            "run_wang_check": False,
        },
    )

    assert result["diagnostics"]["mask_fraction_in_q_window"] == pytest.approx(
        _expected_mask_fraction(valid, qmap)
    )


def test_explicit_mask_is_unioned_with_frame_mask_and_include_edit_cannot_restore_it() -> None:
    from butterfly_saxs.butterfly import analyze_butterfly

    frame_mapping, _frame_object, qmap, _data, _valid = _input_case()
    data_before = np.asarray(frame_mapping["data"]).copy()
    valid_before = np.asarray(frame_mapping["valid_mask"]).copy()
    explicit = np.zeros_like(valid_before)
    explicit[8, 6] = True
    qmap_before = {name: np.asarray(value).copy() for name, value in qmap.items() if name in {"qx", "qy", "q"}}
    result = analyze_butterfly(
        frame_mapping,
        qmap,
        (1.0, 9.0),
        mask=explicit,
        options={
            "stage": "trace",
            "smoothing_scales": (1.0,),
            "run_wang_check": False,
            "edits": [
                {
                    "type": "include_polygon",
                    "points": [[-2.5, -8.0], [2.5, -8.0], [2.5, 8.0], [-2.5, 8.0]],
                }
            ],
        },
    )

    assert result["diagnostics"]["mask_fraction_in_q_window"] == pytest.approx(
        _expected_mask_fraction(valid_before, qmap, explicit)
    )
    np.testing.assert_array_equal(frame_mapping["data"], data_before)
    np.testing.assert_array_equal(frame_mapping["valid_mask"], valid_before)
    for name, original in qmap_before.items():
        np.testing.assert_array_equal(qmap[name], original)


def test_qmap_valid_mask_is_unioned_with_frame_domain() -> None:
    from butterfly_saxs.butterfly import analyze_butterfly

    frame_mapping, _frame_object, qmap, _data, valid = _input_case()
    qmap_valid = np.ones_like(valid)
    qmap_valid[8, 5] = False
    qmap_with_mask = {**qmap, "valid_mask": qmap_valid}
    result = analyze_butterfly(
        frame_mapping,
        qmap_with_mask,
        (1.0, 9.0),
        options={
            "stage": "trace",
            "smoothing_scales": (1.0,),
            "run_wang_check": False,
        },
    )

    expected_valid = valid & qmap_valid
    assert result["diagnostics"]["mask_fraction_in_q_window"] == pytest.approx(
        _expected_mask_fraction(expected_valid, qmap)
    )


def test_companions_receive_the_same_masked_and_edited_analysis_domain(monkeypatch: pytest.MonkeyPatch) -> None:
    import butterfly_saxs.observables as observables
    from butterfly_saxs.butterfly import measure_butterfly_observables
    from butterfly_saxs.butterfly_ridge import _apply_edits
    from butterfly_saxs.ridge_inputs import canonical_inputs

    frame_mapping, _frame_object, qmap, _data, _valid = _input_case()
    explicit = np.zeros_like(frame_mapping["valid_mask"])
    explicit[8, 2] = True
    edits = [{"type": "exclude_polygon", "points": [[-7.5, -0.5], [-3.5, -0.5], [-3.5, 0.5], [-7.5, 0.5]]}]
    captured: dict[str, np.ndarray] = {}
    original_angular = observables.measure_angular_spectrum

    def capture_angular(*args: object, **kwargs: object):
        captured["angular"] = np.asarray(kwargs["mask"], dtype=bool).copy()
        return original_angular(*args, **kwargs)

    def capture_radial(*args: object, **kwargs: object):
        captured["radial"] = np.asarray(kwargs["mask"], dtype=bool).copy()
        return [], []

    monkeypatch.setattr(observables, "measure_angular_spectrum", capture_angular)
    monkeypatch.setattr(observables, "_measure_lobe_radial_observables", capture_radial)
    measure_butterfly_observables(
        frame_mapping,
        qmap,
        (1.0, 9.0),
        mask=explicit,
        fit_ellipse=False,
        options={
            "stage": "trace",
            "smoothing_scales": (1.0,),
            "run_wang_check": False,
            "edits": edits,
        },
    )

    image, qx, qy, q, invalid = canonical_inputs(frame_mapping, qmap, mask=explicit)
    expected_valid = (
        ~invalid
        & np.isfinite(image)
        & np.isfinite(qx)
        & np.isfinite(qy)
        & np.isfinite(q)
        & (q >= 1.0)
        & (q <= 9.0)
    )
    expected_valid, _, _ = _apply_edits(expected_valid, qx, qy, edits)
    np.testing.assert_array_equal(captured["angular"], ~expected_valid)
    np.testing.assert_array_equal(captured["radial"], ~expected_valid)


def test_metadata_only_q_unit_reaches_observable_bundle() -> None:
    from butterfly_saxs.butterfly import measure_butterfly_observables

    frame_mapping, _frame_object, qmap, _data, _valid = _input_case()
    result = measure_butterfly_observables(
        frame_mapping,
        qmap,
        (1.0, 9.0),
        fit_ellipse=False,
        options={
            "stage": "trace",
            "companion_observables": False,
            "smoothing_scales": (1.0,),
            "run_wang_check": False,
        },
    )

    assert result.q_unit == "nm^-1"
    assert result.ridge["q_unit"] == "nm^-1"
    assert result.butterfly["candidate_fit"]["q_unit"] == "nm^-1"
    assert result.butterfly["peak_landmarks"]["q_unit"] == "nm^-1"


def _patch_minimal_evaluate(monkeypatch: pytest.MonkeyPatch, captured: dict[str, object]) -> None:
    import butterfly_saxs.butterfly as butterfly
    import butterfly_saxs.butterfly_ridge as butterfly_ridge

    def fake_trace(*args: object, **kwargs: object) -> dict[str, object]:
        captured.setdefault("trace_masks", []).append(np.asarray(kwargs["mask"], dtype=bool).copy())
        return {"points": [], "arcs": [], "profiles": {}, "diagnostics": {}, "method_version": "test"}

    monkeypatch.setattr(butterfly_ridge, "trace_butterfly_ridges", fake_trace)
    monkeypatch.setattr(
        butterfly,
        "_fit_trace",
        lambda *_args, **_kwargs: {"success": True, "q_unit": "nm^-1"},
    )
    monkeypatch.setattr(
        butterfly,
        "evaluate_arc_evidence",
        lambda *_args, **_kwargs: {"quality": {}, "quantitative_parameters": {}},
    )


def test_resample_mask_override_cannot_restore_frame_invalid_pixels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import butterfly_saxs.butterfly as butterfly

    frame_mapping, _frame_object, qmap, _data, valid = _input_case()
    captured: dict[str, object] = {}
    _patch_minimal_evaluate(monkeypatch, captured)

    def fake_resample(image: np.ndarray, qmap: object, *, mask: np.ndarray, q_window: object,
                      refit: object, options: dict[str, object], cancel_event: object) -> dict[str, object]:
        del qmap, q_window, options, cancel_event
        captured["resample_mask"] = np.asarray(mask, dtype=bool).copy()
        refit(np.asarray(image, dtype=float).copy(), {"mask": np.zeros_like(mask, dtype=bool)})
        return {"intervals": {}, "coverage_calibrated": False, "status": "mock"}

    import butterfly_saxs.butterfly_uncertainty as uncertainty

    monkeypatch.setattr(uncertainty, "resample_butterfly", fake_resample)
    butterfly.analyze_butterfly(
        frame_mapping,
        qmap,
        (1.0, 9.0),
        options={
            "stage": "evaluate",
            "resamples": 1,
            "sensitivity": False,
        },
    )

    expected_invalid = ~valid
    np.testing.assert_array_equal(captured["resample_mask"], expected_invalid)
    assert len(captured["trace_masks"]) == 2
    for trace_mask in captured["trace_masks"]:
        np.testing.assert_array_equal(trace_mask, expected_invalid)


def test_sensitivity_receives_canonical_base_invalid_mask(monkeypatch: pytest.MonkeyPatch) -> None:
    import butterfly_saxs.butterfly as butterfly

    frame_mapping, _frame_object, qmap, _data, valid = _input_case()
    captured: dict[str, object] = {}
    _patch_minimal_evaluate(monkeypatch, captured)

    def fake_sensitivity(*args: object, **kwargs: object) -> dict[str, object]:
        del args
        captured["sensitivity_mask"] = np.asarray(kwargs["mask"], dtype=bool).copy()
        return {"completed": True, "records": [], "held_out_arcs": []}

    monkeypatch.setattr(butterfly, "_sensitivity", fake_sensitivity)
    butterfly.analyze_butterfly(
        frame_mapping,
        qmap,
        (1.0, 9.0),
        options={
            "stage": "evaluate",
            "resamples": 0,
            "sensitivity": True,
        },
    )

    np.testing.assert_array_equal(captured["sensitivity_mask"], ~valid)
