"""Qt-free project document persistence and transaction boundary.

The workbench supplies callbacks for applying/restoring a document.  This
module owns JSON parsing, project-relative path resolution, atomic publication,
and rollback orchestration without importing Qt or holding a window object.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any


class ProjectDocumentController:
    """Persist and transactionally apply one project document."""

    def __init__(
        self,
        *,
        snapshot: Callable[[], Any],
        restore: Callable[[Any], None],
        apply: Callable[[Mapping[str, Any], Path], None],
        serialize: Callable[[Any], Any] | None = None,
    ) -> None:
        self._snapshot = snapshot
        self._restore = restore
        self._apply = apply
        self._serialize = serialize or (lambda value: value)

    @staticmethod
    def resolve_path(value: Any, base: Path) -> Any:
        """Resolve one persisted path relative to its project file."""

        if value is None or value == "" or not isinstance(value, (str, Path)):
            return value
        if str(value) == "in-memory":
            return value
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            candidate = base / candidate
        return str(candidate.resolve())

    @classmethod
    def resolve_frame(cls, value: Any, base: Path) -> Any:
        """Resolve a path-like batch frame while preserving selectors."""

        if isinstance(value, Mapping):
            resolved = dict(value)
            for key in ("path", "file", "source"):
                if key in resolved:
                    resolved[key] = cls.resolve_path(resolved[key], base)
                    break
            return resolved
        return cls.resolve_path(value, base)

    @staticmethod
    def _read_json(target: Path) -> Mapping[str, Any]:
        data = json.loads(
            target.read_text(encoding="utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant is not allowed: {value}")
            ),
        )
        if not isinstance(data, Mapping):
            raise ValueError("project root must be an object")
        return dict(data)

    @classmethod
    def normalize(cls, data: Mapping[str, Any], target: Path) -> dict[str, Any]:
        """Resolve project-relative paths and validate referenced files."""

        normalized = dict(data)
        base = target.parent
        for key, alias in (("input", "input_path"), ("poni", "poni_path"), ("mask", "mask_path")):
            value = data.get(key, data.get(alias))
            resolved = cls.resolve_path(value, base)
            if value is not None:
                normalized[key] = resolved
            if resolved not in (None, "", "in-memory") and not Path(resolved).is_file():
                raise FileNotFoundError(f"project {key} does not exist: {resolved}")

        batch = data.get("batch")
        if isinstance(batch, Mapping):
            batch_copy = dict(batch)
            for key in ("manifest", "checkpoint", "output"):
                if key in batch_copy:
                    batch_copy[key] = cls.resolve_path(batch_copy[key], base)
            if "frames" in batch_copy:
                batch_copy["frames"] = [
                    cls.resolve_frame(item, base)
                    for item in (batch_copy.get("frames") or ())
                ]
            normalized["batch"] = batch_copy
        return normalized

    def save(self, path: str | Path, document: Any) -> Path:
        """Atomically publish strict JSON and return the resolved target."""

        target = Path(path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = self._serialize(document)
        text = json.dumps(payload, indent=2, allow_nan=False)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=target.parent,
                prefix=f".{target.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
        return target

    def load(self, path: str | Path) -> Path:
        """Read, normalize, and apply a project with rollback on failure."""

        target = Path(path).expanduser().resolve()
        normalized = self.normalize(self._read_json(target), target)
        snapshot = self._snapshot()
        try:
            self._apply(normalized, target)
        except Exception:
            self._restore(snapshot)
            raise
        return target


__all__ = ["ProjectDocumentController"]
