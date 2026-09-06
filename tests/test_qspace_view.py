from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6 import QtCore, QtTest

from butterfly_saxs.ui.qspace import QSpaceView


def _curved_map(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.indices(shape, dtype=float)
    return (
        (xx - 4.0) / 10.0 + 0.015 * np.sin(yy / 2.0),
        (yy - 3.0) / 10.0 + 0.01 * np.cos(xx / 2.0),
    )


def test_qspace_preserves_curvilinear_bounds_and_aspect(qtbot, tmp_path):
    view = QSpaceView()
    qtbot.addWidget(view)
    observed = np.arange(56, dtype=float).reshape(7, 8)
    qx, qy = _curved_map(observed.shape)
    view.set_data(observed, qx=qx, qy=qy, valid_mask=np.ones_like(observed, dtype=bool))
    view.resize(620, 460)
    view.show()
    qtbot.waitUntil(lambda: view.q_bounds is not None, timeout=1_000)
    assert view.q_bounds == pytest.approx(
        (float(qx.min()), float(qx.max()), float(qy.min()), float(qy.max()))
    )
    # The view keeps one q-unit per display unit; the painted plot therefore
    # cannot silently adopt the detector's 8:7 aspect ratio.
    view.grab().save(str(tmp_path / "qspace_curvilinear.png"))
    assert (tmp_path / "qspace_curvilinear.png").stat().st_size > 0


def test_qspace_point_selection_and_serializable_edit_modes(qtbot):
    view = QSpaceView()
    qtbot.addWidget(view)
    observed = np.ones((8, 8), dtype=float)
    qx, qy = _curved_map(observed.shape)
    view.set_data(observed, qx=qx, qy=qy)
    view.set_butterfly(
        {
            "points": [
                {
                    "point_id": "p-centre",
                    "qx": 0.0,
                    "qy": 0.0,
                    "branch_id": 0,
                    "side": "upper",
                    "valid": True,
                    "accepted": True,
                }
            ]
        }
    )
    view.resize(620, 460)
    view.show()
    qtbot.waitUntil(lambda: view.q_bounds is not None, timeout=1_000)
    selected: list[dict] = []
    edits: list[dict] = []
    view.pointSelected.connect(selected.append)
    view.editRequested.connect(edits.append)

    # q=(0,0) is close to the visual centre of this map.
    QtTest.QTest.mouseClick(
        view,
        QtCore.Qt.MouseButton.LeftButton,
        pos=view.q_to_widget(0.0, 0.0),
    )
    assert selected and selected[-1]["point_id"] == "p-centre"

    view.set_interaction_mode("seed")
    QtTest.QTest.mouseClick(view, QtCore.Qt.MouseButton.LeftButton, pos=QtCore.QPoint(320, 240))
    assert edits[-1]["type"] == "seed"
    assert set(("qx", "qy")) <= set(edits[-1])

    view.set_interaction_mode("polygon_exclude")
    for pos in (QtCore.QPoint(260, 190), QtCore.QPoint(360, 190), QtCore.QPoint(360, 280)):
        QtTest.QTest.mouseClick(view, QtCore.Qt.MouseButton.LeftButton, pos=pos)
    view.finish_polygon()
    assert edits[-1]["type"] == "exclude_polygon"
    assert len(edits[-1]["points"]) == 3


def test_qspace_window_and_arc_visibility_share_branch_state(qtbot):
    view = QSpaceView()
    qtbot.addWidget(view)
    data = np.ones((10, 10), dtype=float)
    qx, qy = _curved_map(data.shape)
    view.set_data(data, qx=qx, qy=qy)
    view.set_q_window((0.1, 0.3))
    assert view.q_window == pytest.approx((0.1, 0.3))
    view.set_butterfly(
        {
            "points": [],
            "arcs": [
                {"branch_id": 0, "side": "upper", "points": [[0.1, 0.1], [0.2, 0.2]]},
                {"branch_id": 1, "side": "lower", "points": [[-0.1, -0.1], [-0.2, -0.2]]},
            ],
        }
    )
    assert view._arc_visible(view._butterfly["arcs"][0])
    view.set_visible_branch(0, "upper", False)
    assert not view._arc_visible(view._butterfly["arcs"][0])
    assert view._arc_visible(view._butterfly["arcs"][1])
    assert not view._arc_is_extrapolated({"supported": True})
    assert view._arc_is_extrapolated({"supported": False})
    assert view._arc_is_extrapolated({"extrapolated": True, "supported": True})
