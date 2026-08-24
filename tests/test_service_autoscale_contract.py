from __future__ import annotations

import numpy as np

import butterfly_saxs.service as service_module
from butterfly_saxs.intensity import DEFAULT_PARAMETERS
from butterfly_saxs.service import ButterflyAnalysisService


class _FakeFit:
    flags = ()
    success = True
    nfev = 1
    weighted_rmse = 0.0
    condition_number = 1.0
    stderr = {}
    bound_flags = {}
    rmse = 0.0
    parameters = {}

    def __init__(self, shape: tuple[int, int]) -> None:
        self.model_image = np.ones(shape, dtype=float)


def _capture_auto_scale(
    monkeypatch,
) -> tuple[ButterflyAnalysisService, dict[str, object], dict[str, object]]:
    image = np.ones((5, 6), dtype=float)
    service = ButterflyAnalysisService()
    state = service.set_observed(image)
    captured: dict[str, object] = {}

    def fake_measure(*args, **kwargs):
        del args, kwargs
        return None

    def fake_fit(frame, qmap, **kwargs):
        del frame, qmap
        captured["auto_scale_initial"] = kwargs["auto_scale_initial"]
        return _FakeFit(image.shape)

    monkeypatch.setattr(service_module, "measure_observables", fake_measure)
    monkeypatch.setattr(service_module, "fit_intensity_model", fake_fit)
    return service, state, captured


def test_service_enables_auto_scale_for_service_generated_scale_defaults(
    monkeypatch,
) -> None:
    service, state, captured = _capture_auto_scale(monkeypatch)

    service.optimize(payload=state)

    assert captured["auto_scale_initial"] is True


def test_service_does_not_auto_scale_explicit_default_scale_values(monkeypatch) -> None:
    service, state, captured = _capture_auto_scale(monkeypatch)
    explicit = {
        name: DEFAULT_PARAMETERS[name]
        for name in ("amplitude_plus", "amplitude_minus", "background")
    }

    service.optimize(parameters=explicit, payload=state)

    assert captured["auto_scale_initial"] is False
