from __future__ import annotations

import json

import pytest

from butterfly_saxs.ui.project_document import ProjectDocumentController


def test_controller_normalizes_paths_and_restores_on_apply_failure(tmp_path) -> None:
    source = tmp_path / "frame.npy"
    source.write_bytes(b"source")
    state = {"value": "old"}
    applied: list[tuple[dict, str]] = []

    def snapshot() -> dict[str, str]:
        return dict(state)

    def restore(value: dict[str, str]) -> None:
        state.clear()
        state.update(value)

    def apply(document, target) -> None:
        state["value"] = "new"
        applied.append((dict(document), str(target)))
        raise RuntimeError("apply failed")

    controller = ProjectDocumentController(
        snapshot=snapshot,
        restore=restore,
        apply=apply,
    )
    project = tmp_path / "project.json"
    project.write_text(
        json.dumps({"input": "frame.npy", "batch": {"frames": ["frame.npy"]}}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="apply failed"):
        controller.load(project)

    assert state == {"value": "old"}
    assert applied[0][0]["input"] == str(source.resolve())
    assert applied[0][0]["batch"]["frames"] == [str(source.resolve())]


def test_controller_atomic_save_writes_strict_json(tmp_path) -> None:
    controller = ProjectDocumentController(
        snapshot=lambda: {},
        restore=lambda _: None,
        apply=lambda _document, _target: None,
        serialize=lambda value: value,
    )
    target = controller.save(tmp_path / "project.json", {"value": 3})
    assert target == (tmp_path / "project.json").resolve()
    assert json.loads(target.read_text(encoding="utf-8")) == {"value": 3}
    assert not list(tmp_path.glob(".project.json.*.tmp"))
