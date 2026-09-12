from types import SimpleNamespace

import numpy as np

from butterfly_saxs.service import ButterflyAnalysisService


def test_service_retains_fitted_bound_provenance_for_ui_and_export():
    qx, qy = np.meshgrid(np.linspace(-.5, .5, 12), np.linspace(-.5, .5, 12))
    observed = np.ones(qx.shape)
    service = ButterflyAnalysisService()
    state = service.set_observed(observed, qmap={"qx": qx, "qy": qy, "q_unit": "nm^-1"})
    fit = SimpleNamespace(success=True, rmse=.3, condition_number=1e11,
                          bound_flags={"background": True},
                          effective_bounds={"background": (0., 50.)},
                          bound_flag_intensity_scale=2.5)
    result = service._result_mapping(observed, state["qmap"], observed * .7,
                                     parameters=service.parameters, fit=fit)
    metrics = result["metrics"]
    assert metrics["success"] is True
    assert metrics["condition_number"] == 1e11
    assert metrics["bound_flags"] == {"background": True}
    assert list(metrics["effective_bounds"]["background"]) == [0., 50.]
    assert metrics["bound_flag_intensity_scale"] == 2.5
    # A model preview has no optimizer-bound evidence to attribute to a fit.
    preview = service._result_mapping(observed, state["qmap"], observed * .7)
    assert "effective_bounds" not in preview["metrics"]
