from __future__ import annotations

import numpy as np
import pytest


def test_parameter_dock_is_scrollable_and_workflow_guide_is_actionable(qtbot) -> None:
    pytest.importorskip("PySide6")
    pytest.importorskip("pyqtgraph")
    from PySide6 import QtWidgets

    from butterfly_saxs.ui import MainWindow

    window = MainWindow(auto_preview=False, language="en")
    qtbot.addWidget(window)
    window.resize(980, 680)
    window.show()
    qtbot.wait(100)

    scroll = window.parameters_dock.widget()
    assert isinstance(scroll, QtWidgets.QScrollArea)
    assert scroll.objectName() == "parametersScrollArea"
    assert scroll.widgetResizable()
    assert window.workflow_status_group.title() == "Workflow status"
    assert "Open a two-dimensional SAXS image" in window.workflow_status_label.text()
    assert scroll.verticalScrollBar().maximum() > 0

    window.set_observed_data(np.ones((32, 36), dtype=float))
    qtbot.wait(700)
    assert "Select the matching PONI" in window.workflow_status_label.text()
    window.close()
