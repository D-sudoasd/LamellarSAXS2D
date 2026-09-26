from __future__ import annotations

import os
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHERS = ("启动_WingSAXS.cmd", "启动_LamellarSAXS2D.cmd")


def test_cmd_launchers_keep_windows_line_endings_across_checkouts():
    attributes = (ROOT / ".gitattributes").read_text(encoding="utf-8").splitlines()
    assert "*.cmd text eol=crlf" in attributes

    for launcher in LAUNCHERS:
        content = (ROOT / launcher).read_bytes()
        assert content
        assert b"\n" not in content.replace(b"\r\n", b"")
        assert b"\r" not in content.replace(b"\r\n", b"")


@pytest.mark.skipif(os.name != "nt", reason="Windows CMD integration test")
def test_both_cmd_launchers_pass_their_environment_check():
    for launcher in LAUNCHERS:
        result = subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-Command",
                f"& '.\\{launcher}' --check; exit $LASTEXITCODE",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "environment check passed" in result.stdout.lower()
