from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from butterfly_saxs.cli import main
from butterfly_saxs.service import ButterflyAnalysisService


def _package(tmp_path: Path) -> Path:
    package = tmp_path / "package"
    package.mkdir()
    np.save(package / "image.npy", np.arange(12, dtype=np.float32).reshape(3, 4))
    return package


def test_preflight_cli_warning_returns_one_and_strict_json(
    tmp_path: Path, capsys
) -> None:
    package = _package(tmp_path)

    code = main(
        [
            "preflight",
            str(package),
            "--image-glob",
            "image.npy",
            "--q-window",
            "0",
            "2",
        ]
    )

    assert code == 1
    report = json.loads(capsys.readouterr().out)
    assert report["status"]["status_color"] == "yellow"
    assert report["status"]["scientific_status"] == "WARN"
    assert report["status"]["exit_code"] == 1


def test_preflight_cli_green_report_returns_zero(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    package = _package(tmp_path)
    monkeypatch.setattr(
        ButterflyAnalysisService,
        "preflight",
        lambda self, package, **kwargs: {
            "status": {
                "status_color": "green",
                "scientific_status": "PASS",
                "exit_code": 0,
            },
        },
    )

    assert main(["preflight", str(package)]) == 0
    assert json.loads(capsys.readouterr().out)["status"]["status_color"] == "green"


def test_preflight_cli_input_error_returns_two(tmp_path: Path, capsys) -> None:
    code = main(["preflight", str(tmp_path / "missing")])

    assert code == 2
    assert "package must be an existing directory" in capsys.readouterr().err


def test_preflight_cli_refuses_existing_output_without_force(
    tmp_path: Path, capsys
) -> None:
    package = _package(tmp_path)
    output = tmp_path / "evidence"
    command = [
        "preflight",
        str(package),
        "--image-glob",
        "image.npy",
        "--q-window",
        "0",
        "2",
        "--output",
        str(output),
    ]

    assert main(command) == 1
    capsys.readouterr()
    assert main(command) == 2
    assert "force=True" in capsys.readouterr().err


def test_preflight_cli_manifest_structure_failure_returns_two(
    tmp_path: Path,
    capsys,
) -> None:
    package = _package(tmp_path)
    manifest = package / "manifest.json"
    manifest.write_text(
        json.dumps(
            [
                {"path": "image.npy", "order": 1, "time": 0},
                {"path": "image.npy", "order": 1, "time": 1},
            ]
        ),
        encoding="utf-8",
    )

    code = main(
        [
            "preflight",
            str(package),
            "--manifest",
            manifest.name,
            "--q-window",
            "0",
            "2",
        ]
    )

    assert code == 2
    report = json.loads(capsys.readouterr().out)
    assert report["status"]["status_color"] == "red"
    assert report["status"]["exit_code"] == 2
