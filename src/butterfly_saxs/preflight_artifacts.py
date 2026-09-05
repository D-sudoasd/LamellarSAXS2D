"""Artifact publication seam for preflight reports."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np


def prepare_outputs(
    output: Any,
    force: bool,
    *,
    resolve_output_path: Callable[[str | os.PathLike[str] | Path], Path],
    error_type: type[Exception] = ValueError,
) -> tuple[dict[str, Path] | None, list[Path]]:
    """Select the three publication targets and enforce overwrite policy."""

    if output is None:
        return None, []
    if not isinstance(output, (str, os.PathLike, Path)):
        raise error_type("output must be a directory path or None")
    target = resolve_output_path(output)
    if target.exists() and not target.is_dir():
        raise error_type(f"output must be a directory: {target}")
    paths = {
        "preflight_json": target / "preflight.json",
        "arrays_npz": target / "arrays.npz",
        "run_report": target / "run_report.md",
    }
    existing = [path for path in paths.values() if path.exists()]
    directory_contents = list(target.iterdir()) if target.exists() else []
    if not force and directory_contents:
        names = ", ".join(str(path) for path in directory_contents)
        raise FileExistsError(
            f"preflight output exists; pass force=True to overwrite: {names}"
        )
    return paths, existing


def atomic_text(path: Path, text: str) -> None:
    """Write text through a same-directory replace."""

    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(text, encoding="utf-8", newline="\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_npz(path: Path, arrays: Mapping[str, Any]) -> None:
    """Write compressed arrays through a same-directory replace."""

    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".npz", dir=path.parent
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def markdown_report(report: Mapping[str, Any]) -> str:
    """Render the human-readable companion without changing JSON data."""

    status = report.get("status", {})
    quality = report.get("quality", {})
    checks = quality.get("checks", []) if isinstance(quality, Mapping) else []
    extensions = report.get("extensions", {})
    preflight_extension = (
        extensions.get("preflight", {}) if isinstance(extensions, Mapping) else {}
    )
    frames = preflight_extension.get("frames", [])
    lines = [
        "# SAXS package preflight",
        "",
        f"- Status: `{status.get('status_color')}` / scientific status `{status.get('scientific_status')}`",
        f"- Exit code: `{status.get('exit_code')}`",
        f"- Frames: `{len(frames)}`",
        f"- q unit: `{report.get('geometry', {}).get('q_unit', 'unknown')}`",
        f"- Fit-valid pixels (reference frame): `{report.get('analysis_domain', {}).get('counts', {}).get('fit_pixel_count', 'n/a')}`",
        f"- Solver status: `{status.get('solver_status', 'not_run')}` (preflight does not fit data)",
        "",
        "## Checks",
        "",
        "| id | status | reason |",
        "|---|---|---|",
    ]
    for item in checks:
        reason = str(item.get("message", item.get("reason", ""))).replace(
            "|", "\\|"
        ).replace("\n", " ")
        lines.append(
            f"| {item.get('id', '')} | {item.get('status', '')} | {reason} |"
        )
    warnings = preflight_extension.get("warnings", [])
    errors = preflight_extension.get("errors", [])
    if warnings:
        lines.extend(["", "## Warnings", "", *[f"- {item}" for item in warnings]])
    if errors:
        lines.extend(["", "## Errors", "", *[f"- {item}" for item in errors]])
    return "\n".join(lines) + "\n"


class PreflightArtifactWriter:
    """Publish the three preflight artifacts through injected legacy seams."""

    def __init__(
        self,
        *,
        prepare_outputs: Callable[[Any, bool], tuple[dict[str, Path] | None, list[Path]]],
        atomic_text: Callable[[Path, str], None],
        atomic_npz: Callable[[Path, Mapping[str, Any]], None],
        markdown_report: Callable[[Mapping[str, Any]], str],
    ) -> None:
        self._prepare_outputs = prepare_outputs
        self._atomic_text = atomic_text
        self._atomic_npz = atomic_npz
        self._markdown_report = markdown_report

    def prepare(self, output: Any, force: bool) -> tuple[dict[str, Path] | None, list[Path]]:
        return self._prepare_outputs(output, force)

    def write(
        self,
        output_paths: Mapping[str, Path],
        report: Mapping[str, Any],
        arrays: Mapping[str, Any],
    ) -> None:
        output_paths["preflight_json"].parent.mkdir(parents=True, exist_ok=True)
        self._atomic_npz(output_paths["arrays_npz"], arrays)
        self._atomic_text(
            output_paths["preflight_json"],
            json.dumps(
                report,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
        )
        self._atomic_text(
            output_paths["run_report"],
            self._markdown_report(report),
        )


__all__ = ["PreflightArtifactWriter"]
