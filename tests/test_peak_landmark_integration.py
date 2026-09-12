import csv
import json
import numpy as np

from butterfly_saxs.batch import FrameFitResult, FrameRef
from butterfly_saxs.export import export_batch
from butterfly_saxs.pipeline import analyze_frame
from butterfly_saxs.butterfly_figure import export_butterfly_figure


def test_public_analysis_keeps_landmarks_separate_from_ridge_points_and_masks():
    axis = np.linspace(-.5, .5, 80)
    qx, qy = np.meshgrid(axis, axis)
    image = np.ones(qx.shape)
    for x, y in ((.25, .18), (-.25, .18), (-.25, -.18), (.25, -.18)):
        image += 15 * np.exp(-((qx-x)**2+(qy-y)**2)/.006)
    mask = np.zeros(image.shape, dtype=bool)
    hot_y, hot_x = 50, 52
    assert .1 < np.hypot(qx[hot_y, hot_x], qy[hot_y, hot_x]) < .45
    mask[hot_y, hot_x] = True
    image[hot_y, hot_x] = 1e9
    result = analyze_frame(image, qmap={"qx": qx, "qy": qy, "q_unit": "nm^-1"}, mask=mask,
                           config={"analysis": {"ridge_method": "butterfly_curvature", "q_window": [.1, .45],
                                                "butterfly": {"stage": "trace", "resamples": 0, "sensitivity": False}}})
    landmarks = result.butterfly["peak_landmarks"]
    raw = landmarks["raw_global_max"]
    assert raw["raw_intensity"] < 1e9
    assert not mask[raw["pixel_y"], raw["pixel_x"]]
    assert raw["qx"] == qx[raw["pixel_y"], raw["pixel_x"]]
    assert result.butterfly["measurement_status"] == "traced"
    assert all(point.get("point_id") != "G" for point in result.butterfly["points"])
    assert "peak_landmarks" in result.to_mapping()["butterfly"]
    unmasked = analyze_frame(image, qmap={"qx": qx, "qy": qy, "q_unit": "nm^-1"},
                            config={"analysis": {"ridge_method": "butterfly_curvature", "q_window": [.1, .45],
                                                 "butterfly": {"stage": "trace", "resamples": 0, "sensitivity": False}}})
    assert unmasked.butterfly["peak_landmarks"]["raw_global_max"]["raw_intensity"] == 1e9


def test_batch_csv_keeps_pixel_maximum_distinct_from_radial_reflection(tmp_path):
    frame = FrameFitResult(frame=FrameRef(tmp_path / "frame.edf"), result={
        "ridge_points": [],
        "butterfly": {"peak_landmarks": {
            "method_version": "peak-landmarks-v1", "q_unit": "nm^-1",
            "raw_global_max": {"pixel_x": 4, "pixel_y": 3, "qx": .1, "qy": .2,
                               "q": .2236, "raw_intensity": 100., "flags": ["user_q_window_boundary"]},
            "peaks": [{"peak_id": "P1", "pixel_x": 8, "pixel_y": 7, "qx": .3, "qy": .2,
                       "q": .3606, "raw_intensity": 50., "smoothed_intensity": 45., "chi_deg": 33.69}],
        }},
    })
    files = export_batch([frame], tmp_path / "batch")
    with files["lobe_measurements"].open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert {row["measurement_kind"] for row in rows} == {"raw_pixel_maximum", "supported_lobe_pixel"}
    assert all(row["q_star"] == "" for row in rows)
    peak = next(row for row in rows if row["point_id"] == "P1")
    assert float(peak["qx"]) == .3
    assert float(peak["q"]) == .3606
    assert json.loads(peak["landmark_json"])["pixel_y"] == 7


def test_figure_export_replays_measurement_polygon_exclusions(tmp_path):
    axis = np.linspace(-.5, .5, 41)
    qx, qy = np.meshgrid(axis, axis)
    image = np.ones(qx.shape)
    image[20, 30] = 1000.0
    edits = [{"type": "exclude_polygon", "points": [[.21, -.04], [.29, -.04], [.29, .04], [.21, .04]]}]
    result = analyze_frame(image, qmap={"qx": qx, "qy": qy, "q_unit": "nm^-1"},
                           config={"analysis": {"ridge_method": "butterfly_curvature", "q_window": [.05, .45],
                                                "butterfly": {"stage": "trace", "resamples": 0, "sensitivity": False,
                                                              "edits": edits}}})
    measured = result.butterfly["peak_landmarks"]
    assert measured["raw_global_max"]["raw_intensity"] == 1.0
    paths = export_butterfly_figure(tmp_path / "edited_figure", observed=image, qx=qx, qy=qy,
                                   result=result.butterfly, valid_mask=np.ones(image.shape, dtype=bool),
                                   q_unit="nm^-1", width_mm=89, dpi=72)
    exported = json.loads(paths["peak_landmarks_json"].read_text(encoding="utf-8"))
    assert exported["raw_global_max"] == measured["raw_global_max"]
    assert exported["domain"]["raw_search_pixel_count"] == measured["domain"]["raw_search_pixel_count"]
    assert exported["domain"]["applied_polygon_edits"] == measured["domain"]["applied_polygon_edits"]
    with np.load(paths["source_data"], allow_pickle=False) as source:
        assert source["observed"][20, 30] == 1000.0
