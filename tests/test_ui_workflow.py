from __future__ import annotations

import json
import time

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

from butterfly_saxs.ui import MainWindow
from butterfly_saxs.ui.views import ViewGrid


class _BatchEngine:
    def __init__(self) -> None:
        self.payloads: list[dict] = []

    def batch(self, *, parameters, payload):
        del parameters
        self.payloads.append(payload)
        return {
            "records": [
                {"time": 0.0, "parameters": {"theta_deg": {"value": 4.0}, "a": {"value": 10.0}}},
                {"time": 1.0, "parameters": {"theta_deg": {"value": 6.0}, "a": {"value": 12.0}}},
            ]
        }


class _StateEngine:
    def __init__(self) -> None:
        self.parameters = {
            "theta_deg": {
                "value": 10.0,
                "min": -90.0,
                "max": 90.0,
                "vary": True,
                "expr": "",
                "unit": "degree",
                "stderr": None,
            }
        }
        self.commits: list[dict] = []

    def set_parameters(self, parameters) -> None:
        self.commits.append(parameters)


class _PoniEngine(_StateEngine):
    def set_poni(self, path):
        del path
        yy, xx = np.indices((4, 5), dtype=float)
        return {"qx": xx / 10.0, "qy": yy / 10.0, "q_unit": "nm^-1"}


def test_overlay_has_observed_q_background_and_extent(qtbot) -> None:
    grid = ViewGrid()
    qtbot.addWidget(grid)
    observed = np.arange(36, dtype=float).reshape(6, 6)
    yy, xx = np.indices(observed.shape, dtype=float)
    qx = (xx - 2.5) / 10.0
    qy = (yy - 2.5) / 20.0
    grid.set_images(observed, qx=qx, qy=qy)
    ridge = [{"qx": 0.12, "qy": -0.08}]
    grid.set_overlay(ridge, [{"cx": 0.0, "cy": 0.0, "a": 0.2, "b": 0.1, "angle_deg": 15.0}])

    overlay = grid.overlay
    assert overlay.image_data is not None
    assert overlay.image_extent is not None
    xmin, xmax, ymin, ymax = overlay.image_extent
    assert xmin <= ridge[0]["qx"] <= xmax
    assert ymin <= ridge[0]["qy"] <= ymax
    if overlay.plot is not None:
        view_range = overlay.plot.viewRange()
        assert view_range[0][0] <= ridge[0]["qx"] <= view_range[0][1]
        assert view_range[1][0] <= ridge[0]["qy"] <= view_range[1][1]


def test_pattern_views_force_row_major_for_rectangular_detector_arrays(qtbot) -> None:
    grid = ViewGrid()
    qtbot.addWidget(grid)
    observed = np.arange(21, dtype=float).reshape(3, 7)
    model = observed + 10.0
    residual = observed - model
    yy, xx = np.indices(observed.shape, dtype=float)
    qx = (xx - 3.0) / 10.0
    qy = (yy - 1.0) / 20.0

    grid.set_images(observed, model, residual, qx=qx, qy=qy, q_unit="nm^-1")

    expected = {
        "observed": observed,
        "model": model,
        "residual": residual,
        "overlay": observed,
    }
    for name, view in grid.views.items():
        assert view.image_item is not None
        assert view.image_item.axisOrder == "row-major"
        np.testing.assert_array_equal(view.image_data, expected[name])

    for name in ("observed", "model", "residual"):
        rect = grid.views[name].image_item.boundingRect()
        assert rect.width() == pytest.approx(observed.shape[1])
        assert rect.height() == pytest.approx(observed.shape[0])
    grid.close()


def test_parameter_edit_invalidates_an_inflight_worker_result(qtbot) -> None:
    engine = _StateEngine()
    window = MainWindow(engine=engine, auto_preview=False)
    qtbot.addWidget(window)
    old_generation = window._generation.next()

    assert window.set_parameter("theta_deg", 20.0)
    window._on_worker_finished(
        old_generation,
        "optimize",
        {"parameters": {"theta_deg": {"value": 5.0, "unit": "degree"}}},
    )

    assert window.parameters["theta_deg"] == pytest.approx(20.0)
    assert engine.commits[-1]["theta_deg"]["value"] == pytest.approx(20.0)
    assert all(commit["theta_deg"]["value"] != 5.0 for commit in engine.commits)
    window.close()


def test_bulk_parameter_commit_invalidates_an_inflight_worker_result(qtbot) -> None:
    engine = _StateEngine()
    window = MainWindow(engine=engine, auto_preview=False)
    qtbot.addWidget(window)
    old_generation = window._generation.next()

    window.set_parameters({"theta_deg": {"value": 20.0, "unit": "degree"}})
    window._on_worker_finished(
        old_generation,
        "optimize",
        {"parameters": {"theta_deg": {"value": 5.0, "unit": "degree"}}},
    )

    assert window.parameters["theta_deg"] == pytest.approx(20.0)
    assert engine.commits[-1]["theta_deg"]["value"] == pytest.approx(20.0)
    assert window._payload()["analysis"]["auto_scale_initial"] is False
    window.close()


def test_ui_auto_scale_is_one_shot_for_untouched_generated_scale(qtbot) -> None:
    engine = _StateEngine()
    engine.parameters["amplitude_plus"] = {
        "value": 1.0,
        "min": 0.0,
        "max": None,
        "vary": True,
        "expr": "",
        "unit": "a.u.",
        "stderr": None,
    }
    window = MainWindow(engine=engine, auto_preview=False)
    qtbot.addWidget(window)

    assert window._payload()["analysis"]["auto_scale_initial"] is True
    assert window.set_parameter("theta_deg", 12.0)
    assert window._payload()["analysis"]["auto_scale_initial"] is True
    assert window.set_parameter("amplitude_plus", 2.0)
    assert window._payload()["analysis"]["auto_scale_initial"] is False
    window.close()


def test_poni_and_roi_changes_invalidate_old_fit_result(qtbot) -> None:
    engine = _PoniEngine()
    window = MainWindow(engine=engine, auto_preview=False)
    qtbot.addWidget(window)
    observed = np.ones((4, 5), dtype=float)
    window.set_observed_data(observed)

    poni_generation = window._generation.next()
    assert window.set_poni("new-geometry.poni")
    window._on_worker_finished(
        poni_generation,
        "preview",
        {"model": observed * 3.0, "residual": -observed},
    )
    assert window.views.model.image_data is None

    roi_generation = window._generation.next()
    assert window.set_exclusion_roi(
        {"type": "rectangle", "x0": 0.0, "y0": 0.0, "x1": 2.0, "y1": 2.0}
    )
    window._on_worker_finished(
        roi_generation,
        "preview",
        {"model": observed * 4.0, "residual": -observed},
    )
    assert window.views.model.image_data is None
    window.close()


def test_select_mask_without_observed_invalidates_old_worker(qtbot, tmp_path) -> None:
    mask_path = tmp_path / "mask.npy"
    np.save(mask_path, np.zeros((4, 5), dtype=np.uint8))

    window = MainWindow(engine=_StateEngine(), auto_preview=False)
    qtbot.addWidget(window)
    old_generation = window._generation.next()

    assert window.select_mask(mask_path)
    assert not window._generation.is_current(old_generation)
    window.close()


def test_image_and_mask_selectors_are_independent(qtbot, tmp_path) -> None:
    image_path = tmp_path / "images.npz"
    mask_path = tmp_path / "masks.npz"
    image_frames = np.stack(
        [
            np.ones((3, 4), dtype=np.float32),
            np.full((3, 4), 2.0, dtype=np.float32),
        ]
    )
    mask_frames = np.zeros((2, 3, 4), dtype=np.uint8)
    mask_frames[0, 0, 0] = 1
    mask_frames[1, 0, 1] = 1
    np.savez(image_path, image_series=image_frames)
    np.savez(mask_path, mask_series=mask_frames)

    window = MainWindow(mask_frame=0, mask_dataset="mask_series", auto_preview=False)
    qtbot.addWidget(window)
    assert window.open_image(
        image_path,
        frame=1,
        dataset="image_series",
        external_mask=mask_path,
    )

    assert window._frame == 1
    assert window._dataset == "image_series"
    assert window._mask_frame == 0
    assert window._mask_dataset == "mask_series"
    assert np.asarray(window._observed)[0, 0] == pytest.approx(2.0)
    assert bool(window._external_mask[0, 0])
    assert not bool(window._external_mask[0, 1])
    window.close()


def test_set_exclusion_roi_without_observed_invalidates_old_worker(qtbot) -> None:
    window = MainWindow(engine=_StateEngine(), auto_preview=False)
    qtbot.addWidget(window)
    old_generation = window._generation.next()

    assert not window.set_exclusion_roi(
        {"type": "rectangle", "x0": 0.0, "y0": 0.0, "x1": 2.0, "y1": 2.0}
    )
    assert not window._generation.is_current(old_generation)
    window.close()


def test_different_shape_image_without_new_mask_clears_previous_mask(qtbot, tmp_path) -> None:
    old_image_path = tmp_path / "old.npy"
    new_image_path = tmp_path / "new.npy"
    mask_path = tmp_path / "mask.npy"
    np.save(old_image_path, np.ones((4, 5), dtype=np.float32))
    np.save(new_image_path, np.full((3, 6), 2.0, dtype=np.float32))
    mask = np.zeros((4, 5), dtype=np.uint8)
    mask[0, 0] = 1
    np.save(mask_path, mask)

    window = MainWindow(auto_preview=False)
    qtbot.addWidget(window)
    assert window.open_image(old_image_path)
    assert window.select_mask(mask_path)
    assert window._file_mask is not None
    old_generation = window._generation.next()

    assert window.open_image(new_image_path)
    assert not window._generation.is_current(old_generation)
    assert np.asarray(window._observed).shape == (3, 6)
    assert window._mask_path is None
    assert window._file_mask is None
    assert window._external_mask is None
    assert "mask_cleared_shape_changed" in window.flags_label.text()
    window.close()


def test_new_observed_data_invalidates_old_result_and_clears_fit_views(qtbot) -> None:
    engine = _StateEngine()
    window = MainWindow(engine=engine, auto_preview=False)
    qtbot.addWidget(window)
    old_observed = np.ones((4, 5), dtype=float)
    window.set_observed_data(old_observed)
    old_generation = window._generation.next()
    window._apply_result(
        {
            "observed": old_observed,
            "model": old_observed * 2.0,
            "residual": -old_observed,
            "ridge_points": [{"qx": 0.1, "qy": 0.2}],
            "ellipses": [{"cx": 0.0, "cy": 0.0, "a": 0.4, "b": 0.2}],
        }
    )

    new_observed = np.full((4, 5), 7.0)
    window.set_observed_data(new_observed)
    window._on_worker_finished(
        old_generation,
        "optimize",
        {
            "observed": old_observed,
            "model": old_observed,
            "residual": np.zeros_like(old_observed),
            "parameters": {"theta_deg": {"value": 5.0, "unit": "degree"}},
        },
    )

    assert np.array_equal(window.views.observed.image_data, new_observed)
    assert window.views.model.image_data is None
    assert window.views.residual.image_data is None
    assert window.views.overlay.ridge_points == []
    assert window.views.overlay.ellipses == []
    assert window.parameters["theta_deg"] == pytest.approx(10.0)
    window.close()


def test_evolution_flattens_parameter_specs_and_selects_series(qtbot) -> None:
    window = MainWindow(engine=_BatchEngine(), auto_preview=False)
    qtbot.addWidget(window)
    window.plot_evolution(
        [
            {"time": 0.0, "parameters": {"theta_deg": {"value": 4.0}, "a": {"value": 10.0}}},
            {"time": 1.0, "parameters": {"theta_deg": {"value": 6.0}, "a": {"value": 12.0}}},
        ]
    )
    labels = [window.evolution_parameter_combo.itemText(i) for i in range(window.evolution_parameter_combo.count())]
    assert {"theta_deg", "a"} <= set(labels)
    window.evolution_parameter_combo.setCurrentText("theta_deg")
    assert window.evolution_y_key == "theta_deg"
    assert [row["theta_deg"] for row in window._evolution_rows] == [4.0, 6.0]
    if window.evolution_plot is not None:
        assert window.evolution_plot.listDataItems()
    window.close()


def test_external_mask_ellipse_roi_batch_controls_and_project_roundtrip(qtbot, tmp_path) -> None:
    image = np.ones((12, 14), dtype=np.float32)
    image_path = tmp_path / "frame.npy"
    mask_path = tmp_path / "mask.npy"
    np.save(image_path, image)
    file_mask = np.zeros_like(image, dtype=np.uint8)
    file_mask[0, 0] = 1
    np.save(mask_path, file_mask)

    engine = _BatchEngine()
    window = MainWindow(auto_preview=False)
    qtbot.addWidget(window)
    assert window.open_image(image_path)
    assert window.select_mask(mask_path)
    assert window.set_exclusion_roi(
        {"type": "ellipse", "cx": 6.0, "cy": 5.0, "rx": 2.0, "ry": 3.0, "angle_deg": 20.0}
    )
    assert window._mask_path == str(mask_path)
    assert bool(window._external_mask[0, 0])
    assert window._roi_specs[0]["type"] == "ellipse"
    assert window.views.overlay.roi_specs[0]["type"] == "ellipse"

    window.batch_mode_combo.setCurrentText("Warm start")
    window.batch_manifest_edit.setText("manifest.csv")
    window.batch_checkpoint_edit.setText("checkpoint.json")
    window.batch_resume_check.setChecked(True)
    window.batch_output_edit.setText("results")
    window.set_batch_frames([image_path])
    project_path = tmp_path / "workflow.json"
    assert window.save_project(project_path)
    project = json.loads(project_path.read_text(encoding="utf-8"))
    assert project["mask"] == str(mask_path)
    assert project["rois"][0]["type"] == "ellipse"
    assert project["batch"]["mode"] == "warm_start"
    assert project["batch"]["resume"] is True
    assert project["batch"]["frames"] == [str(image_path)]

    window.engine = engine
    window.run_batch()
    qtbot.waitUntil(lambda: bool(engine.payloads), timeout=2_000)
    payload = engine.payloads[0]
    assert payload["mode"] == "warm_start"
    assert payload["manifest"] == "manifest.csv"
    assert payload["checkpoint"] == "checkpoint.json"
    assert payload["resume"] is True
    assert payload["output"] == "results"
    window.close()


def test_project_relative_paths_resolve_from_project_directory(qtbot, tmp_path, monkeypatch) -> None:
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    image_path = project_dir / "frame.npy"
    second_path = project_dir / "frame-2.npy"
    np.save(image_path, np.ones((5, 6), dtype=np.float32))
    np.save(second_path, np.full((5, 6), 2.0, dtype=np.float32))
    project_path = project_dir / "relative.json"
    project_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "input": "frame.npy",
                "parameters": {},
                "analysis": {},
                "batch": {
                    "mode": "independent",
                    "frames": ["frame.npy", "frame-2.npy"],
                    "manifest": "manifest.csv",
                    "checkpoint": "checkpoint.json",
                    "output": "results",
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    window = MainWindow(auto_preview=False)
    qtbot.addWidget(window)
    assert window.load_project(project_path)
    assert window._source_path == str(image_path.resolve())
    assert window.batch_frames == [str(image_path.resolve()), str(second_path.resolve())]
    assert window.batch_manifest_edit.text() == str((project_dir / "manifest.csv").resolve())
    assert window.batch_checkpoint_edit.text() == str((project_dir / "checkpoint.json").resolve())
    assert window.batch_output_edit.text() == str((project_dir / "results").resolve())
    window.close()


def test_loading_project_invalidates_old_parameter_result(qtbot, tmp_path) -> None:
    engine = _StateEngine()
    window = MainWindow(engine=engine, auto_preview=False)
    qtbot.addWidget(window)
    old_generation = window._generation.next()
    project_path = tmp_path / "parameters.json"
    project_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "parameters": {"theta_deg": {"value": 33.0, "unit": "degree"}},
                "analysis": {},
            }
        ),
        encoding="utf-8",
    )

    assert window.load_project(project_path)
    window._on_worker_finished(
        old_generation,
        "optimize",
        {"parameters": {"theta_deg": {"value": 5.0, "unit": "degree"}}},
    )

    assert window.parameters["theta_deg"] == pytest.approx(33.0)
    assert engine.commits[-1]["theta_deg"]["value"] == pytest.approx(33.0)
    window.close()


class _MeasurementPageEngine:
    def __init__(self) -> None:
        self.payloads: list[dict] = []

    def preview(self, *, parameters, payload):
        del parameters
        self.payloads.append(payload)
        observed = np.asarray(payload["observed"], dtype=float)
        return {
            "observed": observed,
            "model": observed,
            "residual": np.zeros_like(observed),
            "observables": {
                "angular": {
                    "angle": np.array([-0.5, 0.5]),
                    "intensity": np.array([1.0, 2.0]),
                    "coverage": np.array([0.8, 1.0]),
                },
                "ridge": {
                    "points": [
                        {"angle": 0.1, "q": 0.4, "accepted": True, "valid": True},
                        {"angle": 1.1, "q": 0.6, "accepted": False, "valid": False},
                    ]
                },
                "lobes": [
                    {"angle": 0.1, "intensity": 3.0, "baseline": 1.0, "snr": 2.0, "fwhm_deg": 5.0, "coverage": 0.9, "valid": True},
                ],
                "ellipse": {
                    "a": 0.8,
                    "b": 0.4,
                    "theta_deg": 17.0,
                    "axis_ratio": 0.5,
                    "ellipticity": 0.866,
                    "Ln_from_minor_axis_nm": 15.7,
                    "Lz_from_draw_axis_nm": 7.8,
                    "rmse": 0.02,
                    "rss": 0.04,
                    "n_points": 1,
                    "success": True,
                    "flags": ["apparent_geometry_only"],
                },
                "phi_app_deg": 12.0,
                "alpha_candidate_deg": None,
                "psi_candidate_deg": None,
            },
            "metrics": {"rmse": 0.0, "ndata": int(observed.size), "flags": []},
        }


def test_measurement_controls_payload_profiles_page_and_project_roundtrip(qtbot, tmp_path) -> None:
    engine = _MeasurementPageEngine()
    window = MainWindow(engine=engine, auto_preview=False)
    qtbot.addWidget(window)
    window.set_observed_data(np.ones((4, 5)))

    assert window.analysis_group.title() == "Analysis / Measurement"
    assert window.measurements_page is not None
    assert window.pages.tabText(window.pages.indexOf(window.measurements_page)) == "Measurements / Profiles"
    window.q_min_edit.setText("0.2")
    window.q_max_edit.setText("Auto")
    window.draw_axis_deg_spin.setValue(123.0)
    window.ridge_method_combo.setCurrentIndex(window.ridge_method_combo.findData("surface_curvature"))
    window.n_angular_bins_spin.setValue(24)
    window.n_ridge_angles_spin.setValue(18)
    window.n_radial_bins_spin.setValue(37)
    window.curvature_sigma_spin.setValue(1.5)
    window.curvature_percentile_spin.setValue(40.0)
    window.normal_step_spin.setValue(0.8)
    window.max_pixels_spin.setValue(0)

    window.request_preview()
    qtbot.waitUntil(lambda: bool(engine.payloads), timeout=2_000)
    analysis = engine.payloads[-1]["analysis"]
    assert analysis["q_min"] == pytest.approx(0.2)
    assert analysis["q_max"] is None
    assert analysis["draw_axis_deg"] == pytest.approx(123.0)
    assert analysis["ridge_method"] == "surface_curvature"
    assert analysis["max_pixels"] == 0

    window._apply_result(engine.preview(parameters={}, payload=engine.payloads[-1]))
    assert window.angular_plot is not None
    assert window.lobe_table.rowCount() == 1
    assert window.ridge_table.rowCount() == 2
    assert window.ellipse_table.rowCount() > 0
    assert window.cancel_button.text() == "Cancel"
    assert window.ignore_late_result_button.text() == "Ignore late result"

    project = tmp_path / "analysis.json"
    assert window.save_project(project)
    window.q_min_edit.setText("0.9")
    assert window.load_project(project)
    assert window.q_min_edit.text() == "0.2"
    assert window.q_max_edit.text() == "Auto"
    assert window.draw_axis_deg_spin.value() == pytest.approx(123.0)
    window.close()


def test_q_overlay_keeps_pixel_roi_out_of_q_space_and_shows_mask(qtbot) -> None:
    grid = ViewGrid()
    qtbot.addWidget(grid)
    observed = np.ones((6, 6), dtype=float)
    yy, xx = np.indices(observed.shape, dtype=float)
    qx = (xx - 2.5) / 10.0
    qy = (yy - 2.5) / 10.0
    valid = np.ones_like(observed, dtype=bool)
    valid[0, 0] = False
    grid.set_images(observed, qx=qx, qy=qy, q_unit="1/nm", valid_mask=valid)
    grid.set_roi({"type": "rectangle", "x0": 0, "y0": 0, "x1": 5, "y1": 5})

    assert np.isnan(grid.observed.image_data[0, 0])
    assert grid.overlay.roi_specs
    if grid.overlay.plot is not None:
        assert not grid.overlay._roi_items
        assert grid.overlay.plot.getAxis("bottom").labelText == "qx (1/nm)"
        assert grid.overlay.plot.getAxis("left").labelText == "qy (1/nm)"
        assert grid.residual.image_item.getColorMap() is not None
    grid.close()


def test_editable_model_ellipses_are_separate_and_follow_parameters(qtbot) -> None:
    engine = _StateEngine()
    parameters = {
        "a": {"value": 0.8, "unit": "unknown"},
        "axis_ratio": {"value": 0.5, "unit": ""},
        "theta_deg": {"value": 15.0, "unit": "degree"},
    }
    window = MainWindow(engine=engine, parameters=parameters, auto_preview=False)
    qtbot.addWidget(window)
    window.set_observed_data(np.ones((4, 4)))
    measured = {"a": 0.7, "b": 0.4, "angle_deg": 12.0, "source": "measured"}
    window.set_fit_overlay([], [measured])

    assert window.views.overlay.ellipses == [measured]
    assert len(window.views.overlay.model_ellipses) == 2
    assert {curve["source"] for curve in window.views.overlay.model_ellipses} == {"model"}
    old_angles = [curve["angle_deg"] for curve in window.views.overlay.model_ellipses]
    assert window.set_parameter("a", 1.0)
    assert window.set_parameter("axis_ratio", 0.25)
    assert window.set_parameter("theta_deg", 30.0)
    updated = window.views.overlay.model_ellipses
    assert [curve["a"] for curve in updated] == [1.0, 1.0]
    assert [curve["b"] for curve in updated] == [0.25, 0.25]
    assert [curve["angle_deg"] for curve in updated] != old_angles
    assert window.views.overlay.ellipses == [measured]
    window.close()


def test_poni_refreshes_physical_q_units_and_partial_failure_clears_images(qtbot) -> None:
    engine = _PoniEngine()
    engine.parameters.update({
        "a": {"value": 0.8, "unit": "unknown"},
        "b": {"value": 0.4, "unit": "unknown"},
        "radial_sigma": {"value": 0.04, "unit": "unknown"},
    })
    window = MainWindow(engine=engine, parameters=engine.parameters, auto_preview=False)
    qtbot.addWidget(window)
    observed = np.ones((4, 5))
    window.set_observed_data(observed)
    window._apply_result({"observed": observed, "model": observed, "residual": np.zeros_like(observed)})
    assert window.views.model.image_data is not None
    assert window.views.residual.image_data is not None
    assert window.set_poni("example.poni")
    units = {row.name: row.unit for row in window.parameter_model.rows}
    assert units["a"] == "nm^-1"
    assert units["b"] == "nm^-1"
    window._apply_result({"observed": observed, "flags": ["analysis_failed:ValueError"]})
    assert window.views.model.image_data is None
    assert window.views.residual.image_data is None
    assert window.status_message.text() != "Optimize complete"
    window.close()


def test_invalid_q_bounds_do_not_start_worker_or_fall_back_to_auto(qtbot) -> None:
    engine = _MeasurementPageEngine()
    window = MainWindow(engine=engine, auto_preview=False)
    qtbot.addWidget(window)
    window.set_observed_data(np.ones((4, 5)))
    window.q_min_edit.setText("not-a-number")
    generation = window.request_preview()
    assert generation > 0
    assert engine.payloads == []
    assert "invalid_analysis" in window.flags_label.text()
    window.close()


def test_programmatic_analysis_change_rejects_an_older_worker_result(qtbot) -> None:
    window = MainWindow(engine=_StateEngine(), auto_preview=False)
    qtbot.addWidget(window)
    observed = np.ones((4, 5), dtype=float)
    model = observed * 0.8
    window.set_observed_data(observed)
    old_generation = window._generation.next()

    window.set_analysis_settings({"q_min": 0.2}, trigger_preview=False)
    assert not window._generation.is_current(old_generation)
    window._on_worker_finished(
        old_generation,
        "preview",
        {
            "observed": observed,
            "model": model,
            "residual": observed - model,
            "parameters": window.parameter_model.parameter_dict(),
        },
    )

    assert window._last_result is None
    assert not window._current_result_is_reviewable()
    window.close()


def test_optimize_pre_snapshot_is_detached_and_stale_result_cannot_change_candidate(qtbot) -> None:
    class _SlowOptimizeEngine(_StateEngine):
        def optimize(self, *, parameters, payload):
            del parameters, payload
            time.sleep(0.1)
            return {"parameters": {"theta_deg": {"value": 5.0, "unit": "degree"}}}

    engine = _SlowOptimizeEngine()
    window = MainWindow(engine=engine, auto_preview=False)
    qtbot.addWidget(window)
    observed = np.arange(12, dtype=float).reshape(3, 4)
    window.set_observed_data(observed)
    generation = window.request_optimize()
    before = window._fit_session["optimize_before"]

    assert before["parameters"]["theta_deg"]["value"] == pytest.approx(10.0)
    assert np.array_equal(before["input"]["data"], observed)
    observed[0, 0] = 999.0
    assert before["input"]["data"][0, 0] != pytest.approx(999.0)

    window.cancel_jobs()
    window._on_worker_finished(
        generation,
        "optimize",
        {"parameters": {"theta_deg": {"value": 5.0, "unit": "degree"}}},
    )
    assert window.parameters["theta_deg"] == pytest.approx(10.0)
    assert window._fit_session["optimize_after"] is None
    window.close()


def test_optimize_review_requires_explicit_reviewer_and_edit_invalidates_status(qtbot) -> None:
    window = MainWindow(engine=_StateEngine(), auto_preview=False)
    qtbot.addWidget(window)
    observed = np.ones((4, 5), dtype=float)
    model = observed * 0.8
    window.set_observed_data(observed)
    window._fit_session["optimize_before"] = window._capture_fit_context()
    generation = window._generation.next()
    window._on_worker_finished(
        generation,
        "optimize",
        {
            "observed": observed,
            "model": model,
            "residual": observed - model,
            "parameters": {"theta_deg": {"value": 15.0, "unit": "degree"}},
        },
    )

    assert window.fit_session["manual_status"] == "unreviewed"
    assert not window.accept_current()
    window.reviewer_edit.setText("Dr. Reviewer")
    window.review_notes_edit.setText("clear four-lobe frame")
    assert window.accept_current()
    assert window.fit_session["manual_status"] == "accepted"
    assert window.fit_session["reviewed_by"] == "Dr. Reviewer"
    assert window.fit_session["reviewed_at"]
    assert window.fit_session["accepted_parameters"]["theta_deg"]["value"] == pytest.approx(15.0)

    assert window.set_parameter("theta_deg", 20.0)
    assert window.fit_session["manual_status"] == "unreviewed"
    assert window.fit_session["accepted_parameters"] is None
    window.close()


def test_multiple_snapshots_preserve_notes_order_and_exact_parameter_fields(qtbot) -> None:
    parameters = {
        "a": {"value": 0.8, "min": 0.2, "max": 1.2, "vary": True, "expr": "", "unit": "nm^-1", "stderr": 0.01},
        "axis_ratio": {"value": 0.5, "min": 0.1, "max": 1.0, "vary": False, "expr": "", "unit": "", "stderr": None},
        "theta_deg": {"value": 15.0, "min": -90.0, "max": 90.0, "vary": True, "expr": "", "unit": "degree", "stderr": 0.2},
    }
    window = MainWindow(engine=_StateEngine(), parameters=parameters, auto_preview=False)
    qtbot.addWidget(window)
    window.snapshot_note_edit.setText("initial manual state")
    assert window.save_snapshot()
    first = window.fit_session["snapshots"][0]

    assert window.set_parameter("a", 1.1)
    window.snapshot_note_edit.setText("wider ellipse")
    assert window.save_snapshot()
    assert [item["note"] for item in window.fit_session["snapshots"]] == ["initial manual state", "wider ellipse"]
    assert [window.snapshot_combo.itemData(i) for i in range(window.snapshot_combo.count())] == [0, 1]

    window._apply_result({"model": np.ones((3, 3)), "residual": np.zeros((3, 3)), "ellipses": [{"a": 1.0, "b": 0.5}]})
    assert window.restore_snapshot(0)
    assert window.parameter_model.parameter_dict() == first["parameters"]
    assert window.views.model.image_data is None
    assert window.views.residual.image_data is None
    assert window.views.overlay.ellipses == []
    window.close()


def test_project_schema_two_roundtrip_restores_fit_session_without_detector_arrays(qtbot, tmp_path) -> None:
    window = MainWindow(engine=_StateEngine(), auto_preview=False)
    qtbot.addWidget(window)
    window.set_observed_data(np.ones((5, 6), dtype=float))
    window._fit_session["optimize_before"] = window._capture_fit_context()
    generation = window._generation.next()
    window._on_worker_finished(
        generation,
        "optimize",
        {
            "observed": np.ones((5, 6), dtype=float),
            "model": np.full((5, 6), 0.9, dtype=float),
            "residual": np.full((5, 6), 0.1, dtype=float),
            "parameters": {"theta_deg": {"value": 17.0, "unit": "degree"}},
        },
    )
    window.reviewer_edit.setText("Reviewer A")
    window.review_notes_edit.setText("accepted for manual comparison")
    assert window.accept_current()
    window.snapshot_note_edit.setText("accepted candidate")
    assert window.save_snapshot()

    project_path = tmp_path / "fit-session-v2.json"
    assert window.save_project(project_path)
    project = json.loads(project_path.read_text(encoding="utf-8"))
    assert project["schema_version"] == 2
    assert project["fit_session"]["manual_status"] == "accepted"
    assert project["fit_session"]["snapshots"][0]["parameters"]["theta_deg"]["value"] == pytest.approx(17.0)

    def _contains_detector_array(value):
        if isinstance(value, dict):
            return any(_contains_detector_array(item) for item in value.values())
        if isinstance(value, list):
            return any(_contains_detector_array(item) for item in value)
        return False

    # The persisted fit context keeps selectors and settings, not observed/q/mask arrays.
    assert "data" not in project["fit_session"]["optimize_before"]["input"]
    assert "qmap" not in project["fit_session"]["optimize_before"]
    assert "external" not in project["fit_session"]["optimize_before"].get("mask", {})
    assert not _contains_detector_array(project["fit_session"]["optimize_before"].get("input", {}).get("data"))

    restored = MainWindow(engine=_StateEngine(), auto_preview=False)
    qtbot.addWidget(restored)
    assert restored.load_project(project_path)
    assert restored.fit_session["manual_status"] == "accepted"
    assert restored.fit_session["reviewed_by"] == "Reviewer A"
    assert len(restored.fit_session["snapshots"]) == 1
    assert restored.parameter_model.parameter_dict()["theta_deg"]["value"] == pytest.approx(17.0)
    restored.close()
    window.close()


def test_preview_can_be_reviewed_and_exported_but_parameter_edit_blocks_stale_result(
    qtbot,
    tmp_path,
) -> None:
    window = MainWindow(engine=_StateEngine(), auto_preview=False)
    qtbot.addWidget(window)
    observed = np.arange(20, dtype=float).reshape(4, 5) + 1.0
    model = observed * 0.75
    yy, xx = np.indices(observed.shape, dtype=float)
    window.set_observed_data(
        observed,
        qmap={
            "qx": xx / 10.0,
            "qy": yy / 10.0,
            "metadata": {"q_unit": "nm^-1"},
        },
    )

    generation = window._generation.next()
    window._on_worker_finished(
        generation,
        "preview",
        {
            "observed": observed,
            "model": model,
            "residual": observed - model,
            "parameters": window.parameter_model.parameter_dict(),
        },
    )

    assert window.accept_current_button.isEnabled()
    assert window.export_evidence_action.isEnabled()
    assert not window.accept_current()
    window.reviewer_edit.setText("Reviewer Preview")
    window.review_notes_edit.setText("manual parameters match the visible lobes")
    assert window.accept_current()

    evidence_dir = tmp_path / "preview-evidence"
    assert window.export_manual_evidence(evidence_dir)
    assert {path.name for path in evidence_dir.iterdir()} == {
        "observed.png",
        "model.png",
        "residual.png",
        "overlay.png",
        "parameters.csv",
        "fit_session.json",
        "provenance.json",
    }
    exported_session = json.loads((evidence_dir / "fit_session.json").read_text(encoding="utf-8"))
    assert exported_session["manual_status"] == "accepted"
    assert exported_session["reviewed_by"] == "Reviewer Preview"
    assert exported_session["analysis"]["q_unit"] == "nm^-1"

    assert window.set_parameter("theta_deg", 18.0)
    assert not window.export_manual_evidence(tmp_path / "stale-evidence")
    assert not (tmp_path / "stale-evidence").exists()
    assert "evidence_stale" in window.flags_label.text()
    window.close()


def test_current_preview_can_be_explicitly_rejected_and_exported(qtbot, tmp_path) -> None:
    window = MainWindow(engine=_StateEngine(), auto_preview=False)
    qtbot.addWidget(window)
    observed = np.ones((3, 4), dtype=float)
    model = np.zeros_like(observed)
    window.set_observed_data(observed)
    generation = window._generation.next()
    window._on_worker_finished(
        generation,
        "preview",
        {
            "observed": observed,
            "model": model,
            "residual": observed - model,
            "parameters": window.parameter_model.parameter_dict(),
            "flags": ["apparent_geometry_only", "nonunique_inverse_problem"],
        },
    )

    window.reviewer_edit.setText("Reviewer Negative")
    window.review_notes_edit.setText("current empirical model is not suitable for this frame")
    assert window.reject_current()
    assert window.fit_session["manual_status"] == "rejected"
    assert window.fit_session["accepted_parameters"] is None

    evidence_dir = tmp_path / "rejected-evidence"
    assert window.export_manual_evidence(evidence_dir)
    exported_session = json.loads((evidence_dir / "fit_session.json").read_text(encoding="utf-8"))
    assert exported_session["manual_status"] == "rejected"
    assert exported_session["reviewed_by"] == "Reviewer Negative"
    window.close()
