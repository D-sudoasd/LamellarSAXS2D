"""Environment diagnostics for first-run and support workflows.

The module intentionally depends only on the Python standard library so it can
explain a broken scientific/UI environment instead of failing while importing
the package's optional stacks.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

SUPPORTED_MIN = (3, 11)
SUPPORTED_MAX_EXCLUSIVE = (3, 14)

_CORE_MODULES: tuple[tuple[str, str, str], ...] = (
    ("NumPy", "numpy", "numpy"),
    ("SciPy", "scipy", "scipy"),
    ("Matplotlib", "matplotlib", "matplotlib"),
    ("FabIO", "fabio", "fabio"),
    ("pyFAI", "pyFAI", "pyFAI"),
    ("tifffile", "tifffile", "tifffile"),
    ("PyYAML", "yaml", "PyYAML"),
)
_UI_MODULES: tuple[tuple[str, str, str], ...] = (
    ("PySide6", "PySide6.QtWidgets", "PySide6"),
    ("pyqtgraph", "pyqtgraph", "pyqtgraph"),
)
_OPTIONAL_MODULES: tuple[tuple[str, str, str], ...] = (
    ("h5py (HDF5)", "h5py", "h5py"),
)


@dataclass(frozen=True)
class DiagnosticCheck:
    """One machine-readable environment check."""

    name: str
    category: str
    required: bool
    ok: bool
    detail: str
    remediation: str = ""


def _version_text(
    distribution: str,
    *,
    version_getter: Callable[[str], str],
) -> str:
    try:
        return str(version_getter(distribution))
    except importlib.metadata.PackageNotFoundError:
        return "installed (version metadata unavailable)"
    except Exception as exc:  # pragma: no cover - unusual metadata backends
        return f"installed (version lookup failed: {type(exc).__name__})"


def _dependency_check(
    display_name: str,
    module_name: str,
    distribution: str,
    *,
    category: str,
    required: bool,
    importer: Callable[[str], Any],
    version_getter: Callable[[str], str],
) -> DiagnosticCheck:
    try:
        importer(module_name)
    except Exception as exc:  # noqa: BLE001 - diagnostics must report import/DLL errors
        return DiagnosticCheck(
            name=display_name,
            category=category,
            required=required,
            ok=False,
            detail=f"{type(exc).__name__}: {exc}",
            remediation=_repair_command(),
        )
    return DiagnosticCheck(
        name=display_name,
        category=category,
        required=required,
        ok=True,
        detail=_version_text(distribution, version_getter=version_getter),
    )


def _repair_command(cwd: Path | None = None) -> str:
    root = Path.cwd() if cwd is None else Path(cwd)
    constraint = root / "constraints" / "validation-py311-313.txt"
    if (root / "pyproject.toml").exists():
        if constraint.exists():
            return (
                f'"{sys.executable}" -m pip install '
                '-c constraints/validation-py311-313.txt -e ".[all]"'
            )
        return f'"{sys.executable}" -m pip install -e ".[all]"'
    return (
        "Run from the LamellarSAXS2D project directory, then install "
        'with: python -m pip install -e ".[all]"'
    )


def collect_diagnostics(
    *,
    require_ui: bool = False,
    importer: Callable[[str], Any] = importlib.import_module,
    version_getter: Callable[[str], str] = importlib.metadata.version,
    version_info: Sequence[int] | None = None,
    cwd: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Collect deterministic environment checks without starting Qt."""

    version = tuple(version_info or sys.version_info[:3])
    python_ok = SUPPORTED_MIN <= version[:2] < SUPPORTED_MAX_EXCLUSIVE
    checks: list[DiagnosticCheck] = [
        DiagnosticCheck(
            name="Python",
            category="runtime",
            required=True,
            ok=python_ok,
            detail=(
                f"{version[0]}.{version[1]}.{version[2]} "
                f"({sys.executable})"
            ),
            remediation=(
                ""
                if python_ok
                else "Use Python 3.11, 3.12, or 3.13; Python 3.14+ is unsupported."
            ),
        )
    ]
    for display_name, module_name, distribution in _CORE_MODULES:
        checks.append(
            _dependency_check(
                display_name,
                module_name,
                distribution,
                category="core",
                required=True,
                importer=importer,
                version_getter=version_getter,
            )
        )
    for display_name, module_name, distribution in _UI_MODULES:
        checks.append(
            _dependency_check(
                display_name,
                module_name,
                distribution,
                category="ui",
                required=bool(require_ui),
                importer=importer,
                version_getter=version_getter,
            )
        )
    for display_name, module_name, distribution in _OPTIONAL_MODULES:
        checks.append(
            _dependency_check(
                display_name,
                module_name,
                distribution,
                category="optional",
                required=False,
                importer=importer,
                version_getter=version_getter,
            )
        )

    root = Path.cwd() if cwd is None else Path(cwd)
    required_failures = [check for check in checks if check.required and not check.ok]
    return {
        "schema_version": 1,
        "ready": not required_failures,
        "require_ui": bool(require_ui),
        "application": "LamellarSAXS2D",
        "distribution": "butterfly-saxs",
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
            "supported": python_ok,
            "supported_range": ">=3.11,<3.14",
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "working_directory": str(root.resolve()),
        "checks": [asdict(check) for check in checks],
        "required_failures": [check.name for check in required_failures],
        "repair_command": _repair_command(root),
    }


def _format_text(report: dict[str, Any]) -> str:
    status = "READY" if report["ready"] else "NOT READY"
    lines = [
        "LamellarSAXS2D environment diagnostics / 环境诊断",
        f"Status / 状态: {status}",
        f"Python support: {report['python']['supported_range']}",
        "",
    ]
    for check in report["checks"]:
        if check["ok"]:
            marker = "OK"
        elif check["required"]:
            marker = "FAIL"
        else:
            marker = "OPTIONAL-MISSING"
        requirement = "required" if check["required"] else "optional"
        lines.append(
            f"[{marker}] {check['name']} ({check['category']}, {requirement}): "
            f"{check['detail']}"
        )
    if not report["ready"]:
        lines.extend(
            [
                "",
                "Repair / 修复建议:",
                report["repair_command"],
            ]
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bsaxs-doctor",
        description=(
            "Check the supported Python version and LamellarSAXS2D dependencies "
            "without opening the GUI."
        ),
    )
    parser.add_argument(
        "--require-ui",
        action="store_true",
        help="Treat PySide6 and pyqtgraph as required.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print a strict machine-readable JSON report.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Print nothing; communicate readiness through the exit code.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    options = build_parser().parse_args(list(argv) if argv is not None else None)
    report = collect_diagnostics(require_ui=options.require_ui)
    if not options.quiet:
        if options.json:
            print(json.dumps(report, ensure_ascii=True, indent=2, allow_nan=False))
        else:
            print(_format_text(report))
    return 0 if report["ready"] else 1


if __name__ == "__main__":  # pragma: no cover - exercised through console script
    raise SystemExit(main())
