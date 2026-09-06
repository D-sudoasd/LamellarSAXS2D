from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

import butterfly_saxs.butterfly as butterfly
from butterfly_saxs.parameters import ParameterSet, ParameterSpec


def _sensitivity_inputs():
    image = np.ones((1, 1), dtype=float)
    qmap = SimpleNamespace(qx=np.zeros((1, 1)), qy=np.zeros((1, 1)))
    trace = {"points": []}
    options = {"smoothing_scales": [1.0]}
    return image, qmap, trace, options


def _fit_payload():
    return {
        "success": True,
        "a": 1.0,
        "b": 0.2,
        "axis_ratio": 0.2,
        "theta_deg": 20.0,
    }


def test_explicit_geometry_bounds_create_real_contract_expand_refits(monkeypatch) -> None:
    image, qmap, trace, options = _sensitivity_inputs()
    parameters = {
        "cx": {"value": 0.0, "vary": False, "min": -0.5, "max": 0.5},
        "a": {"value": 1.0, "min": 0.5, "max": 1.5},
        "axis_ratio": {"value": 0.2, "min": 0.1, "max": 0.4},
        "theta_deg": {"value": 20.0, "min": 10.0, "max": 40.0},
        "b": {"value": 0.2, "min": 0.1, "max": 0.4, "expr": "a*axis_ratio"},
    }
    seen: list[dict[str, object]] = []

    def fake_analyze(*args, **kwargs):
        variant_parameters = kwargs["parameters"]
        seen.append({"parameters": deepcopy(variant_parameters), "options": kwargs["options"]})
        payload = _fit_payload()
        payload["a"] = float(variant_parameters["a"].get("min", 0.0))
        payload["axis_ratio"] = float(variant_parameters["axis_ratio"].get("min", 0.0))
        payload["theta_deg"] = float(variant_parameters["theta_deg"].get("min", 0.0))
        return {"candidate_fit": payload}

    monkeypatch.setattr(butterfly, "analyze_butterfly", fake_analyze)
    result = butterfly._sensitivity(
        image,
        qmap,
        [0.0, 1.0],
        mask=None,
        options=options,
        parameters=parameters,
        reference=0.0,
        multistart=1,
        cancel_event=None,
        trace=trace,
    )

    prior = result["prior_bound_sensitivity"]
    assert prior["status"] == "completed"
    assert {record["variant"] for record in prior["variants"]} == {
        "prior_a_contract",
        "prior_a_expand",
        "prior_axis_ratio_contract",
        "prior_axis_ratio_expand",
        "prior_theta_contract",
        "prior_theta_expand",
    }
    assert len(seen) == 10  # four established variants plus six prior variants
    changes = {record["variant"]: record["bound_changes"] for record in prior["variants"]}
    assert changes["prior_a_contract"]["after"] == {"min": 0.75, "max": 1.25}
    assert changes["prior_theta_expand"]["source_name"] == "theta_deg"
    assert changes["prior_theta_expand"]["unit"] == "deg"
    assert result["parameter_ranges"]["a"][0] < result["parameter_ranges"]["a"][1]
    assert result["parameter_ranges"]["axis_ratio"][0] < result["parameter_ranges"]["axis_ratio"][1]
    assert result["parameter_ranges"]["theta_deg"][0] < result["parameter_ranges"]["theta_deg"][1]
    assert any(item["parameter"] == "cx" and item["reason"] == "fixed_or_tied" for item in prior["skipped"])
    assert any(item["parameter"] == "b" and item["reason"] == "derived_parameter_not_applicable" for item in prior["skipped"])
    for call in seen[4:]:
        variant_parameters = call["parameters"]
        assert variant_parameters["cx"]["vary"] is False
        assert variant_parameters["b"]["expr"] == "a*axis_ratio"


def test_parameter_set_bounds_are_preserved_and_changed_only_for_free_geometry(monkeypatch) -> None:
    image, qmap, trace, options = _sensitivity_inputs()
    parameters = ParameterSet(
        {
            "cx": ParameterSpec(0.0, vary=False, min=-0.5, max=0.5),
            "a": ParameterSpec(1.0, min=0.5, max=1.5),
            "axis_ratio": ParameterSpec(0.2, min=0.1, max=0.4),
            "theta": ParameterSpec(0.3, min=0.1, max=0.5),
            "b": ParameterSpec(0.2, min=0.1, max=0.4, expr="a*axis_ratio"),
        }
    )
    seen: list[ParameterSet] = []

    def fake_analyze(*args, **kwargs):
        seen.append(kwargs["parameters"])
        return {"candidate_fit": _fit_payload()}

    monkeypatch.setattr(butterfly, "analyze_butterfly", fake_analyze)
    result = butterfly._sensitivity(
        image,
        qmap,
        [0.0, 1.0],
        mask=None,
        options=options,
        parameters=parameters,
        reference=0.0,
        multistart=1,
        cancel_event=None,
        trace=trace,
    )

    assert len(result["prior_bound_sensitivity"]["variants"]) == 6
    assert all(isinstance(value, ParameterSet) for value in seen[4:])
    for variant in seen[4:]:
        assert variant["cx"].is_fixed
        assert variant["b"].is_tied
        assert variant["b"].expr == "a*axis_ratio"


def test_no_explicit_bounds_adds_no_prior_refits(monkeypatch) -> None:
    image, qmap, trace, options = _sensitivity_inputs()
    calls = []

    def fake_analyze(*args, **kwargs):
        calls.append(kwargs["parameters"])
        return {"candidate_fit": _fit_payload()}

    monkeypatch.setattr(butterfly, "analyze_butterfly", fake_analyze)
    baseline = butterfly._sensitivity(
        image,
        qmap,
        [0.0, 1.0],
        mask=None,
        options=options,
        parameters=None,
        reference=0.0,
        multistart=1,
        cancel_event=None,
        trace=trace,
    )
    no_bounds = butterfly._sensitivity(
        image,
        qmap,
        [0.0, 1.0],
        mask=None,
        options=options,
        parameters={"a": 1.0, "axis_ratio": 0.2, "theta": 0.3},
        reference=0.0,
        multistart=1,
        cancel_event=None,
        trace=trace,
    )

    assert len(calls) == 8
    assert baseline["prior_bound_sensitivity"]["status"] == "not_applicable"
    assert no_bounds["prior_bound_sensitivity"]["status"] == "skipped"
    assert baseline["prior_bound_sensitivity"]["explicit_bounds_requested"] is False
    assert no_bounds["prior_bound_sensitivity"]["variants"] == []
    assert baseline["parameter_ranges"] == no_bounds["parameter_ranges"]
    assert [record["variant"] for record in baseline["records"]] == [
        record["variant"] for record in no_bounds["records"]
    ]


def test_expansion_uses_small_q_and_center_scales(monkeypatch) -> None:
    image, qmap, trace, options = _sensitivity_inputs()
    parameters = {
        "a": {"value": 0.04, "min": 0.035, "max": 0.045},
        "cx": {"value": 0.0, "min": -0.001, "max": 0.001},
    }

    monkeypatch.setattr(
        butterfly,
        "analyze_butterfly",
        lambda *args, **kwargs: {"candidate_fit": _fit_payload()},
    )
    result = butterfly._sensitivity(
        image,
        qmap,
        [0.0, 1.0],
        mask=None,
        options=options,
        parameters=parameters,
        reference=0.0,
        multistart=1,
        cancel_event=None,
        trace=trace,
    )
    changes = {
        record["variant"]: record["bound_changes"]["after"]
        for record in result["prior_bound_sensitivity"]["variants"]
    }
    assert changes["prior_a_expand"]["min"] == pytest.approx(0.03)
    assert changes["prior_a_expand"]["max"] == pytest.approx(0.05)
    assert changes["prior_cx_expand"]["min"] == pytest.approx(-0.002)
    assert changes["prior_cx_expand"]["max"] == pytest.approx(0.002)


def test_inclusive_boundary_values_still_receive_valid_bound_variants() -> None:
    parameters = {
        "a": {"value": 0.04, "min": 0.04, "max": 0.045},
        "axis_ratio": {"value": 0.005, "min": 0.005, "max": 0.01},
        "theta": {"value": 0.0, "min": 0.0, "max": 0.2},
    }
    variants, skipped = butterfly._explicit_prior_bound_variants(parameters)

    assert {variant["variant"] for variant in variants} == {
        "prior_a_contract",
        "prior_a_expand",
        "prior_axis_ratio_contract",
        "prior_axis_ratio_expand",
        "prior_theta_contract",
        "prior_theta_expand",
    }
    assert not any(item.get("reason") in {"value_at_lower_bound", "value_at_upper_bound"} for item in skipped)
def test_standard_application_specs_do_not_report_an_explicit_prior():
    from butterfly_saxs.analysis_config import ellipse_parameter_specs
    from butterfly_saxs.butterfly import _explicit_prior_bound_variants

    specs = ellipse_parameter_specs({"ellipse_multistart": 3}, q_window=(0.1, 0.5))
    assert specs["b"]["expr"] == "a*axis_ratio"
    variants, skipped = _explicit_prior_bound_variants(specs)
    assert variants == []
    assert skipped == [{"reason": "no_finite_explicit_bounds"}]
