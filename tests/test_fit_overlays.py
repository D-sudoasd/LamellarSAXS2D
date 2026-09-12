from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path
import numpy as np

from butterfly_saxs.fit_overlays import assess_geometry_fit, fit_geometry_layers, export_fit_overlays
from butterfly_saxs.intensity import ellipse_polar_radius
from butterfly_saxs.butterfly_figure import _prepare_inputs


def _result():
    return {
        "measurement_status": "undetermined",
        "candidate_fit": {"success": True, "a": .45, "b": .02, "axis_ratio": .02/.45,
                          "theta_deg": 36., "reference_axis_deg": 0., "center_qx": 0., "center_qy": 0., "rmse": .046},
        "quality": {"status": "WARN", "metrics": {"residual_sigma_ratio": 10.2, "median_localization_sigma_q": .0045},
                    "provisional_limits": {"residual_sigma_ratio_max": 3.0}, "scientific_status": "NOT_ACCEPTED"},
    }




def test_geometry_helper_stays_independent_of_export_and_matplotlib_modules():
    script = "import sys\nsys.path.insert(0, 'src')\nfrom butterfly_saxs.fit_overlays import fit_geometry_layers\nfit_geometry_layers({})\nassert 'butterfly_saxs.candidate_geometry' in sys.modules\nassert 'butterfly_saxs.butterfly_comparison' not in sys.modules\nassert 'butterfly_saxs.butterfly_figure' not in sys.modules\nassert not any(name == 'matplotlib' or name.startswith('matplotlib.') for name in sys.modules)"
    completed = subprocess.run([sys.executable, '-c', script], cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
def test_existing_residual_screen_explains_poor_fit_without_promoting_it():
    assessment = assess_geometry_fit(_result())
    assert assessment["geometry_status"] == "poor_match"
    assert assessment["residual_sigma_ratio"] > assessment["residual_limit"]
    assert assessment["scientific_acceptance"] is False
    result = _result()
    result["quality"].pop("provisional_limits")
    assert assess_geometry_fit(result)["geometry_status"] == "diagnostic_candidate"
    result["measurement_status"] = "traced"
    assert assess_geometry_fit(result)["geometry_status"] == "not_evaluated"


def test_geometry_and_intensity_model_curves_are_distinct_and_match_actual_kernel():
    theta = math.radians(42)
    params = {"a": .4, "b": .18, "theta": theta, "theta_deg": 42.0}
    layers = fit_geometry_layers(_result(), model_parameters=params, model_reference_axis_deg=17., model_status="ok")
    assert layers["geometry"]["parameters"]["axis_ratio"] != layers["intensity_model"]["parameters"]["axis_ratio"]
    assert layers["geometry"]["status"] == "poor_match"
    for index, curve in enumerate(layers["intensity_model"]["curves"]):
        xy = curve["points"]
        actual_q = np.hypot(xy[:, 0], xy[:, 1])
        chi = np.arctan2(xy[:, 1], xy[:, 0]) - math.radians(17)
        expected = ellipse_polar_radius(chi, .4, .18, theta if index == 0 else -theta)
        np.testing.assert_allclose(actual_q, expected, rtol=1e-13, atol=1e-15)
        assert curve["source"] == "intensity_model"


def test_missing_or_conflicting_model_geometry_never_uses_default_axes():
    for params, ref in (({"a": .4}, 0.), ({"a": .4, "b": .18, "theta": .5}, None),
                        ({"a": .4, "b": .18, "theta": .5, "theta_deg": 80.}, 0.)):
        layer = fit_geometry_layers(_result(), model_parameters=params, model_reference_axis_deg=ref)
        assert not layer["intensity_model"]["curves"]


def test_live_service_parameter_specifications_render_their_resolved_values():
    from butterfly_saxs.service import ButterflyAnalysisService

    specs = ButterflyAnalysisService().parameters
    assert isinstance(specs["a"], dict) and "value" in specs["a"]
    specs["a"]["value"] = .6
    specs["axis_ratio"]["value"] = .25
    specs["theta_deg"]["value"] = 23.
    layers = fit_geometry_layers(_result(), model_parameters=specs, model_reference_axis_deg=17.)
    model = layers["intensity_model"]
    assert len(model["curves"]) == 2
    assert model["parameters"]["a"] == .6
    assert model["parameters"]["b"] == .15
    assert math.isclose(model["parameters"]["theta_deg"], 23.)


def test_clean_standalone_overlays_export_both_sources_and_original_values(tmp_path):
    qx, qy = np.meshgrid(np.linspace(-.5, .5, 24), np.linspace(-.5, .5, 24))
    observed = 1 + np.exp(-(qx**2+qy**2)/.1)
    original = observed.copy()
    data = _prepare_inputs(observed=observed, qx=qx, qy=qy, valid_mask=None, result=_result(), q_unit="nm^-1",
                           context={"pixel_model_parameters": {"a": .4, "b": .18, "theta": .7}, "pixel_model_reference_axis_deg": 0., "pixel_model_status": "ok"},
                           display_scale="log1p", width_mm=89., dpi=72)
    files, assessment = export_fit_overlays(tmp_path, data=data)
    assert {"measured_only_svg", "geometry_overlay_svg", "intensity_model_overlay_svg", "fit_overlay_curves"} <= files.keys()
    assert all(path.is_file() for path in files.values())
    assert "poor match" in files["geometry_overlay_svg"].read_text(encoding="utf-8").replace("_", " ")
    assert assessment["assessment"]["scientific_acceptance"] is False
    np.testing.assert_array_equal(observed, original)
    with np.load(files["fit_overlay_curves"], allow_pickle=False) as curves:
        assert {"geometry_0", "geometry_1", "intensity_model_0", "intensity_model_1"} <= set(curves.files)
