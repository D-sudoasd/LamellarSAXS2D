from __future__ import annotations

import math

import pytest

from butterfly_saxs.parameters import ParameterSet, ParameterSpec, default_ellipse_parameters


def test_expression_resolution_and_default_physical_parameterization() -> None:
    parameters = default_ellipse_parameters(a=2.0, axis_ratio=0.4, theta=math.radians(18.0))
    values = parameters.resolve()

    assert values["b"] == pytest.approx(0.8)
    assert values["theta_deg"] == pytest.approx(18.0)
    assert parameters["b"].is_tied
    assert "b" not in parameters.free_names


def test_parameter_expression_unknown_name_and_cycle_fail() -> None:
    with pytest.raises(ValueError, match="unknown"):
        ParameterSet({"x": ParameterSpec(1.0), "y": ParameterSpec(expr="missing")})
    with pytest.raises(ValueError, match="cyclic"):
        ParameterSet({"x": ParameterSpec(expr="y"), "y": ParameterSpec(expr="x")})


def test_fixed_and_bound_values_are_checked() -> None:
    parameters = ParameterSet(
        {
            "x": ParameterSpec(2.0, min=0.0, max=3.0, vary=False),
            "y": {"value": 3.0, "expr": "x+1"},
        }
    )
    assert parameters.fixed_names == ("x",)
    assert parameters.resolve()["y"] == pytest.approx(3.0)
    with pytest.raises(ValueError):
        parameters.update_values({"y": 1.0})
