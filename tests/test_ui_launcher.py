from __future__ import annotations

from pathlib import Path

from butterfly_saxs.ui import launcher


def test_gui_launcher_returns_wrapped_exit_code_without_log(tmp_path: Path) -> None:
    log_path = tmp_path / "launcher.log"

    assert (
        launcher.run(
            ["frame.npy"],
            launch_fn=lambda argv: 0 if argv == ["frame.npy"] else 2,
            log_path=log_path,
        )
        == 0
    )
    assert not log_path.exists()


def test_gui_launcher_records_and_presents_startup_failure(tmp_path: Path) -> None:
    log_path = tmp_path / "launcher.log"
    presented: list[tuple[str, str]] = []

    def fail(_argv):
        raise TypeError("simulated Qt plug-in failure")

    code = launcher.run(
        ["--poni", "geometry.poni"],
        launch_fn=fail,
        log_path=log_path,
        presenter=lambda title, message: presented.append((title, message)),
    )

    assert code == 1
    assert log_path.exists()
    log = log_path.read_text(encoding="utf-8")
    assert "TypeError: simulated Qt plug-in failure" in log
    assert "geometry.poni" in log
    assert presented
    assert str(log_path) in presented[0][1]
