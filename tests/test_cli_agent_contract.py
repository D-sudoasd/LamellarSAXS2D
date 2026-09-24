from __future__ import annotations

import ast
import json
from pathlib import Path

from butterfly_saxs.cli import build_parser, main
from butterfly_saxs.cli_contract import (
    AGENT_MANIFEST_SCHEMA,
    CLI_ERROR_SCHEMA,
    INSPECT_SCHEMA,
    agent_manifest,
)
from butterfly_saxs.errors import PipelineError


def test_cli_module_keeps_numpy_off_the_import_path() -> None:
    source = (Path(__file__).resolve().parents[1] / "src" / "butterfly_saxs" / "cli.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] != "numpy"
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            assert not module.startswith("numpy")
            assert module != "pipeline"


def test_describe_is_the_default_agent_catalog(
    capsys,
) -> None:
    assert main([]) == 0
    default_report = json.loads(capsys.readouterr().out)
    assert main(["describe"]) == 0
    named_report = json.loads(capsys.readouterr().out)

    assert default_report == named_report == agent_manifest()
    assert default_report["schema_version"] == AGENT_MANIFEST_SCHEMA
    assert default_report["ok"] is True
    names = {item["name"] for item in default_report["commands"]}
    assert {"describe", "doctor", "inspect", "analyze", "batch", "preflight"} <= names
    assert default_report["defaults"]["recommended_ridge_method"] == "butterfly_curvature"
    assert any("scientific acceptance" in item.lower() for item in default_report["invariants"])
    json.dumps(default_report, allow_nan=False)


def test_cli_help_includes_describe_and_doctor() -> None:
    parser = build_parser()
    choices = parser._subparsers._group_actions[0].choices
    assert {"describe", "doctor", "inspect", "analyze", "batch", "synthetic", "gui"} <= set(choices)


def test_inspect_without_input_emits_json_error_envelope(capsys) -> None:
    assert main(["inspect"]) == 2
    captured = capsys.readouterr()
    assert "错误：" in captured.err
    payload = json.loads(captured.out)
    assert payload["schema_version"] == CLI_ERROR_SCHEMA
    assert payload["ok"] is False
    assert payload["exit_code"] == 2
    assert payload["error"]["type"] == "PipelineError"
    assert payload["error"]["code"] == "pipeline_error"
    assert payload["agent"]["scientific_acceptance"] is False
    json.dumps(payload, allow_nan=False)


def test_usage_error_is_json_on_stdout(capsys) -> None:
    assert main(["inspect", "--definitely-not-a-flag"]) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == CLI_ERROR_SCHEMA
    assert payload["error"]["code"] == "usage_error"
    assert payload["ok"] is False


def test_inspect_report_carries_agent_contract(tmp_path, capsys) -> None:
    import numpy as np

    source = tmp_path / "frame.npy"
    np.save(source, np.ones((8, 8), dtype=float))
    output = tmp_path / "inspect.json"
    assert main(["inspect", str(source), "-o", str(output)]) == 0
    report = json.loads(capsys.readouterr().out)
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert report == saved
    assert report["schema_version"] == INSPECT_SCHEMA
    assert report["result_type"] == "inspect"
    assert report["agent"]["scientific_acceptance"] is False
    assert report["agent"]["pixel_q"] is True
    assert any("poni" in item.lower() for item in report["agent"]["next"])
    assert "butterfly_curvature" in " ".join(report["agent"]["next"])


def test_analyze_annotation_keeps_payload_and_denies_acceptance(
    monkeypatch, capsys
) -> None:
    import butterfly_saxs.cli as cli_module

    class Result:
        def to_mapping(self):
            return {
                "ellipse_fit": {"status": "ok", "success": True},
                "full2d": {"status": "failed", "success": False},
                "flags": {"uncalibrated_pixel_q": True},
            }

    monkeypatch.setattr(cli_module, "analyze_frame", lambda *args, **kwargs: Result())
    assert main(["analyze", "frame.npy", "--full2d"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["full2d"]["status"] == "failed"
    assert report["schema_version"] == "lamellarsaxs2d.analysis_summary.v1"
    assert report["agent"]["scientific_acceptance"] is False
    assert report["agent"]["quality_gate_reason"]
    assert report["agent"]["pixel_q"] is True


def test_analyze_warn_exits_one_without_discarding_measurement(monkeypatch, capsys) -> None:
    import butterfly_saxs.cli as cli_module

    class Result:
        def to_mapping(self):
            return {
                "butterfly": {
                    "settings": {"stage": "evaluate"},
                    "candidate_fit": {"success": True, "status": "ok"},
                    "quality": {"status": "WARN", "scientific_status": "NOT_ACCEPTED"},
                    "points": [{"accepted": True}],
                },
                "ellipse_fit": {"success": True, "status": "ok"},
            }

    monkeypatch.setattr(cli_module, "analyze_frame", lambda *args, **kwargs: Result())
    assert main(["analyze", "frame.npy"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["butterfly"]["candidate_fit"]["success"] is True
    assert report["agent"]["quality_gate_reason"] == "butterfly.quality.status=WARN"
    assert report["agent"]["exit_code"] == 1


def test_pipeline_error_is_stdlib_importable() -> None:
    assert issubclass(PipelineError, RuntimeError)
    assert PipelineError("x").args == ("x",)
