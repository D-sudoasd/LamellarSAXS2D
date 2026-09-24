"""Machine-readable CLI contract for agents and automation.

Scientific payloads keep their existing field names.  This module only adds
stable envelopes, error objects, and operator guidance.  It is standard-library
only so ``bsaxs describe`` and ``bsaxs doctor`` can start when NumPy/SciPy are
missing.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from . import __version__

INSPECT_SCHEMA = "lamellarsaxs2d.inspect.v1"
ANALYSIS_SUMMARY_SCHEMA = "lamellarsaxs2d.analysis_summary.v1"
BATCH_RUN_SCHEMA = "lamellarsaxs2d.batch_run.v1"
SYNTHETIC_SCHEMA = "lamellarsaxs2d.synthetic.v1"
PROJECT_RUN_SCHEMA = "lamellarsaxs2d.project_run.v2"
AGENT_MANIFEST_SCHEMA = "lamellarsaxs2d.agent_manifest.v1"
CLI_ERROR_SCHEMA = "lamellarsaxs2d.cli_error.v1"

EXIT_CODES = {
    "0": "Completed; required quality gates passed (or the command has no quality gate).",
    "1": "Completed with WARN, quality FAIL, partial batch failure, or cancellation; evidence kept.",
    "2": "Input, usage, selector, unit, mask, PONI, or output-overwrite error; do not treat as a result.",
}

INVARIANTS = (
    "success=True and solver_status=success are not scientific acceptance.",
    "pixel-q never becomes a physical period; load a PONI before reporting L in nm.",
    "Do not fabricate missing butterfly quadrants or opposite-side arcs.",
    "A flat_ellipse bound hit or a runaway major axis is ring-only; do not copy ring L into Ln.",
    "Do not overwrite outputs without an explicit --force; --force never applies to raw inputs.",
    "Identify (trace) then Evaluate; --full2d is a separate empirical intensity model.",
    "geometry-only analysis must not silently start full2d.",
    "JSON stdout is strict (no NaN/Infinity). Human diagnostics stay on stderr.",
)

_SCHEMA_BY_COMMAND = {
    "inspect": INSPECT_SCHEMA,
    "analyze": ANALYSIS_SUMMARY_SCHEMA,
    "batch": BATCH_RUN_SCHEMA,
    "synthetic": SYNTHETIC_SCHEMA,
    "project": PROJECT_RUN_SCHEMA,
}


def tool_info() -> dict[str, str]:
    return {
        "name": "WingSAXS",
        "distribution": "butterfly-saxs",
        "import": "butterfly_saxs",
        "version": str(__version__),
        "cli": "bsaxs",
    }


def error_code_for(exc: BaseException) -> str:
    name = type(exc).__name__
    mapping = {
        "FileExistsError": "output_exists",
        "FileNotFoundError": "input_not_found",
        "IsADirectoryError": "input_not_found",
        "NotADirectoryError": "input_not_found",
        "PermissionError": "os_error",
        "PipelineError": "pipeline_error",
        "ProjectConfigError": "project_config",
        "PreflightError": "preflight_error",
        "PathContractError": "path_contract",
        "AnalysisCancelled": "cancelled",
        "ValueError": "value_error",
        "OSError": "os_error",
    }
    return mapping.get(name, "error")


def cli_error_payload(
    exc: BaseException,
    *,
    exit_code: int,
    command: str | None = None,
    code: str | None = None,
) -> dict[str, Any]:
    """Strict JSON error envelope for stdout. Human text still goes to stderr."""

    return {
        "schema_version": CLI_ERROR_SCHEMA,
        "ok": False,
        "exit_code": int(exit_code),
        "command": command,
        "tool": tool_info(),
        "error": {
            "type": type(exc).__name__,
            "code": code or error_code_for(exc),
            "message": str(exc),
            "stage": getattr(exc, "stage", None),
        },
        "agent": {
            "scientific_acceptance": False,
            "next": [
                "Read error.code and error.message; do not retry with --force unless code is output_exists and the target is generated output.",
                "Run `bsaxs describe` for commands, exit codes, and invariants.",
                "If the CLI cannot import, run `bsaxs-doctor --json` (or `bsaxs doctor --json`).",
            ],
        },
    }


def usage_error_payload(message: str) -> dict[str, Any]:
    return cli_error_payload(
        RuntimeError(message),
        exit_code=2,
        code="usage_error",
    )


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _flag_map(report: Mapping[str, Any]) -> dict[str, Any]:
    flags = report.get("flags")
    return _mapping(flags)


def _pixel_q(report: Mapping[str, Any]) -> bool:
    flags = _flag_map(report)
    if flags.get("uncalibrated_pixel_q") is True:
        return True
    unit = str(report.get("q_unit") or "")
    if unit == "pixel-q":
        return True
    geometry = report.get("geometry")
    if isinstance(geometry, Mapping) and str(geometry.get("q_unit") or "") == "pixel-q":
        return True
    qmap = report.get("qmap")
    if isinstance(qmap, Mapping) and str(qmap.get("q_unit") or "") == "pixel-q":
        return True
    return False


def _publication_state(report: Mapping[str, Any]) -> str | None:
    for source in (
        report.get("butterfly"),
        report.get("ellipse_fit"),
        _mapping(report.get("observables")).get("butterfly"),
        _mapping(report.get("observables")).get("ellipse"),
    ):
        payload = _mapping(source)
        quality = payload.get("quality")
        if isinstance(quality, Mapping):
            publication = quality.get("publication") or quality.get("publication_state")
            if publication:
                return str(publication)
            status = str(quality.get("status") or "").upper()
            if status in {"FAIL", "FAILED", "INVALID"}:
                return "fail"
        status = str(payload.get("measurement_status") or payload.get("publication") or "")
        if status:
            return status
    return None


def agent_guidance(
    command: str,
    report: Mapping[str, Any],
    *,
    exit_code: int | None = None,
    quality_gate_reason: str | None = None,
) -> dict[str, Any]:
    """Operator hints derived from a completed JSON report. Never a scientific verdict."""

    pixel_q = _pixel_q(report)
    publication = _publication_state(report)
    next_steps: list[str] = []
    if command == "inspect":
        if pixel_q:
            next_steps.append("Provide --poni before interpreting q*, L ring, or any spacing as physical.")
        next_steps.append(
            "bsaxs analyze INPUT --ridge-method butterfly_curvature "
            "--ellipse-preset flat_ellipse --butterfly-stage evaluate"
        )
        next_steps.append("Do not pass --full2d unless an empirical whole-pixel intensity model is explicitly requested.")
    elif command == "analyze":
        if quality_gate_reason:
            next_steps.append(f"Quality gate reported ({quality_gate_reason}); keep the JSON evidence, do not publish ellipse shape.")
        if publication == "ring":
            next_steps.append("Publish first-order ring L only; leave Ln/Lz/major-axis unpublished.")
        if pixel_q:
            next_steps.append("Result is pixel-q; do not convert to nm without a PONI.")
        next_steps.append("Treat solver success as an engineering status, not scientific acceptance.")
    elif command == "batch":
        failed = report.get("n_failed")
        if isinstance(failed, int) and failed > 0:
            next_steps.append("Inspect failed frames in stdout JSON and the output directory; remaining frames were isolated.")
        if report.get("blocked_stage") == "preflight":
            next_steps.append("Read preflight/preflight.json; fitting did not start.")
        elif report.get("unattended"):
            next_steps.append("Read preflight/preflight.json and frame_summary.csv; require complete NPZ metadata before using the batch evidence.")
        if report.get("cancelled"):
            next_steps.append("The batch was cancelled; resume only with the same inputs and scientific configuration.")
        next_steps.append("Keep PONI, mask, q-window, and config identical across the series.")
    elif command == "synthetic":
        next_steps.append("Synthetic arrays are empirical fixtures with pixel-q unless a PONI is supplied later.")
        next_steps.append("bsaxs inspect OUTPUT.npz && bsaxs analyze OUTPUT.npz --ridge-method butterfly_curvature")
    elif command == "project":
        next_steps.append("Prefer `bsaxs batch` when you need streaming CSV/JSON/NPZ longitudinal exports.")
    else:
        next_steps.append("Run `bsaxs describe` for the supported command catalog.")

    return {
        "scientific_acceptance": False,
        "pixel_q": pixel_q,
        "publication": publication,
        "quality_gate_reason": quality_gate_reason,
        "exit_code": exit_code,
        "next": next_steps,
        "do_not": [
            "Treat success=True as scientific acceptance.",
            "Invent a physical period from pixel-q.",
            "Overwrite outputs without --force.",
            "Mirror missing quadrants into observed arcs.",
            "Copy ring L into the Ln column.",
        ],
    }


def annotate_report(
    report: Mapping[str, Any],
    *,
    command: str,
    exit_code: int | None = None,
    quality_gate_reason: str | None = None,
) -> dict[str, Any]:
    """Copy a scientific payload and add contract fields without dropping keys."""

    annotated = dict(report)
    annotated.setdefault("schema_version", _SCHEMA_BY_COMMAND.get(command, AGENT_MANIFEST_SCHEMA))
    annotated.setdefault("result_type", command)
    annotated.setdefault("tool", tool_info())
    annotated["agent"] = agent_guidance(
        command,
        annotated,
        exit_code=exit_code,
        quality_gate_reason=quality_gate_reason,
    )
    return annotated


def agent_manifest() -> dict[str, Any]:
    """Catalog an agent can parse before touching experimental data."""

    return {
        "schema_version": AGENT_MANIFEST_SCHEMA,
        "ok": True,
        "tool": tool_info(),
        "python": {">=": "3.11", "<": "3.14", "supported_range": ">=3.11,<3.14"},
        "entry_points": [
            "bsaxs",
            "python -m butterfly_saxs",
            "bsaxs-doctor",
            "bsaxs-gui",
        ],
        "exit_codes": EXIT_CODES,
        "invariants": list(INVARIANTS),
        "defaults": {
            "ridge_method": "radial_peak",
            "recommended_ridge_method": "butterfly_curvature",
            "recommended_ellipse_preset": "flat_ellipse",
            "recommended_butterfly_stage": "evaluate",
            "overwrite": False,
        },
        "recommended_agent_workflow": [
            "bsaxs-doctor --json   # or: bsaxs doctor --json",
            "bsaxs describe",
            "bsaxs synthetic --shape 128x128 -o synthetic.npz",
            "bsaxs inspect synthetic.npz",
            "bsaxs analyze synthetic.npz --ridge-method butterfly_curvature "
            "--ellipse-preset flat_ellipse --butterfly-stage evaluate --butterfly-resamples 0",
            "bsaxs preflight PACKAGE --manifest MANIFEST --poni PONI --mask MASK -o results/preflight",
            "bsaxs batch 'PACKAGE/images/*.edf' --unattended PACKAGE --manifest PACKAGE/manifest.csv --poni PACKAGE/geometry.poni --mask PACKAGE/mask.npy -o results/unattended_001",
        ],
        "commands": [
            {
                "name": "describe",
                "purpose": "Print this machine-readable catalog.",
                "stdout": AGENT_MANIFEST_SCHEMA,
                "exit_codes": {"0": EXIT_CODES["0"]},
            },
            {
                "name": "doctor",
                "purpose": "Stdlib environment diagnostics; same as bsaxs-doctor.",
                "stdout": "doctor JSON when --json, otherwise text",
                "exit_codes": {
                    "0": "Environment ready.",
                    "1": "Required dependency or Python version missing.",
                },
            },
            {
                "name": "inspect",
                "purpose": "Read-only frame/q-space diagnostics. No ellipse publication.",
                "stdout": INSPECT_SCHEMA,
                "exit_codes": {"0": EXIT_CODES["0"], "2": EXIT_CODES["2"]},
            },
            {
                "name": "analyze",
                "purpose": "Single-frame geometry measurement; --full2d is optional and separate.",
                "stdout": ANALYSIS_SUMMARY_SCHEMA,
                "exit_codes": EXIT_CODES,
            },
            {
                "name": "batch",
                "purpose": "Independent or quality-gated warm-start series; --unattended adds preflight, streaming evidence and a checkpoint.",
                "stdout": BATCH_RUN_SCHEMA,
                "exit_codes": EXIT_CODES,
            },
            {
                "name": "synthetic",
                "purpose": "Write a reproducible empirical butterfly fixture (pixel-q).",
                "stdout": SYNTHETIC_SCHEMA,
                "exit_codes": {"0": EXIT_CODES["0"], "2": EXIT_CODES["2"]},
            },
            {
                "name": "preflight",
                "purpose": "Read-only package/PONI/mask/unit gate. Does not fit.",
                "stdout": "lamellarsaxs2d.result.v1",
                "exit_codes": EXIT_CODES,
            },
            {
                "name": "project",
                "purpose": "Run a TOML project; default envelope is project_run.v2.",
                "stdout": PROJECT_RUN_SCHEMA,
                "exit_codes": EXIT_CODES,
            },
            {
                "name": "gui",
                "purpose": "Open the workbench. Prefer bsaxs-gui on Windows pythonw.",
                "stdout": "none",
                "exit_codes": {"0": "Window closed normally.", "1": "Start-up failure; see launcher.log."},
            },
        ],
        "documentation": {
            "agents": "AGENTS.md",
            "user_guide": "docs/user_guide_zh.md",
            "first_run": "docs/first_run_zh.md",
            "result_schema": "docs/validation/result_schema_v1.md",
            "scientific_scope": "docs/scientific_basis_zh.md",
        },
    }


__all__ = [
    "AGENT_MANIFEST_SCHEMA",
    "ANALYSIS_SUMMARY_SCHEMA",
    "BATCH_RUN_SCHEMA",
    "CLI_ERROR_SCHEMA",
    "EXIT_CODES",
    "INSPECT_SCHEMA",
    "INVARIANTS",
    "PROJECT_RUN_SCHEMA",
    "SYNTHETIC_SCHEMA",
    "agent_guidance",
    "agent_manifest",
    "annotate_report",
    "cli_error_payload",
    "error_code_for",
    "tool_info",
    "usage_error_payload",
]
