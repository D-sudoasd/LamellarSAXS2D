from __future__ import annotations

import numpy as np
import pytest

from butterfly_saxs.intensity import default_intensity_parameters, parameter_values
from butterfly_saxs.parameters import ParameterSet, ParameterSpec
from butterfly_saxs.pipeline import fit_full2d, inspect_frame
from butterfly_saxs.project import ProjectConfig


def _qmap(shape: tuple[int, int], q_unit: str) -> dict[str, np.ndarray | str]:
    height, width = shape
    y, x = np.indices(shape, dtype=float)
    qx = x - (width - 1.0) / 2.0
    qy = y - (height - 1.0) / 2.0
    return {"qx": qx, "qy": qy, "q": np.hypot(qx, qy), "q_unit": q_unit}


def test_fit_full2d_preserves_parameter_set_warm_start_specs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from butterfly_saxs import intensity

    image = np.ones((8, 8), dtype=float)
    qmap = _qmap(image.shape, "pixel-q")
    warm_start = ParameterSet(
        {
            "a": ParameterSpec(0.81, min=0.4, max=1.2, vary=True),
            "axis_ratio": ParameterSpec(0.56, min=0.2, max=0.9, vary=False),
            "b": ParameterSpec(
                0.45,
                min=0.1,
                max=0.8,
                vary=False,
                expr="a*axis_ratio",
            ),
            "theta": ParameterSpec(0.18, min=-1.0, max=1.0, vary=False),
            "theta_deg": ParameterSpec(0.0, vary=False, expr="theta*180/pi"),
        }
    )
    calls: list[ParameterSet] = []

    class FakeFit:
        parameters = warm_start
        success = True
        message = "ok"
        flags: tuple[str, ...] = ()
        prediction = np.zeros_like(image)
        residual = np.zeros_like(image)

    def fake_fit(_frame: object, _qmap: object, initial: ParameterSet, **_kwargs: object) -> FakeFit:
        calls.append(initial)
        return FakeFit()

    monkeypatch.setattr(intensity, "fit_intensity_model", fake_fit)
    result = fit_full2d(image, qmap, {}, initial_parameters=warm_start)

    assert result["status"] == "ok"
    assert len(calls) == 1
    received = calls[0]
    assert isinstance(received, ParameterSet)
    for name in ("a", "axis_ratio", "b", "theta", "theta_deg"):
        assert received[name].value == pytest.approx(warm_start[name].value)
        assert received[name].min == warm_start[name].min
        assert received[name].max == warm_start[name].max
        assert received[name].vary == warm_start[name].vary
        assert received[name].expr == warm_start[name].expr
    assert received["b"].expr == "a*axis_ratio"


@pytest.mark.parametrize("q_unit", ["pixel-q", "Å^-1"])
def test_inspect_report_uses_qmap_unit_over_project_default(q_unit: str) -> None:
    image = np.ones((10, 10), dtype=float)
    report = inspect_frame(
        image,
        qmap=_qmap(image.shape, q_unit),
        config=ProjectConfig(q_unit="1/nm"),
    )

    assert report["q_unit"] == q_unit


def test_parameter_set_rejects_independent_radian_and_degree_fields() -> None:
    malformed = ParameterSet(
        {
            "theta": ParameterSpec(0.2),
            "theta_deg": ParameterSpec(20.0, vary=False),
        }
    )

    with pytest.raises(ValueError, match="theta.*theta_deg"):
        parameter_values(malformed)


def test_default_parameter_set_tied_degree_adapter_remains_valid() -> None:
    parameters = default_intensity_parameters(theta_deg=17.0)

    values = parameter_values(parameters)

    assert values["theta"] == pytest.approx(np.deg2rad(17.0))
    assert parameters["theta_deg"].is_tied
