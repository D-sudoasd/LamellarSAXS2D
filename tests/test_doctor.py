from __future__ import annotations

import json

import pytest

from butterfly_saxs import doctor


def _version(distribution: str) -> str:
    versions = {
        "numpy": "2.1.0",
        "scipy": "1.13.0",
        "matplotlib": "3.8.0",
        "fabio": "2024.4.0",
        "pyFAI": "2024.5.0",
        "tifffile": "2024.5.0",
        "PyYAML": "6.0.0",
        "PySide6": "6.8.0",
        "pyqtgraph": "0.13.7",
        "h5py": "3.10.0",
    }
    return versions.get(distribution, "1.2.3")


def test_doctor_requires_only_requested_ui_dependencies() -> None:
    def importer(module: str):
        if module == "PySide6.QtWidgets":
            raise ImportError("missing Qt runtime")
        return object()

    core_report = doctor.collect_diagnostics(
        require_ui=False,
        importer=importer,
        version_getter=_version,
        version_info=(3, 12, 0),
    )
    ui_report = doctor.collect_diagnostics(
        require_ui=True,
        importer=importer,
        version_getter=_version,
        version_info=(3, 12, 0),
    )

    assert core_report["ready"] is True
    assert ui_report["ready"] is False
    assert ui_report["required_failures"] == ["PySide6"]


@pytest.mark.parametrize("version", [(3, 10, 9), (3, 14, 0)])
def test_doctor_rejects_python_outside_supported_range(version) -> None:
    report = doctor.collect_diagnostics(
        importer=lambda _module: object(),
        version_getter=_version,
        version_info=version,
    )

    assert report["ready"] is False
    assert "Python" in report["required_failures"]


def test_doctor_json_is_strict_and_exit_code_tracks_readiness(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report = {
        "ready": False,
        "python": {"supported_range": ">=3.11,<3.14"},
        "checks": [],
        "repair_command": "repair",
    }
    monkeypatch.setattr(doctor, "collect_diagnostics", lambda **_kwargs: report)

    assert doctor.main(["--json", "--require-ui"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ready"] is False
    json.dumps(payload, allow_nan=False)
