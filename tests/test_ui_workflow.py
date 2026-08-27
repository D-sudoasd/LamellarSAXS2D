from __future__ import annotations

import json

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
