"""Explicit package/external-root path resolution for evidence workflows."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


class PathContractError(ValueError):
    """Raised when an input path is outside its authorized roots."""


def authorized_roots(
    package_root: str | os.PathLike[str] | Path,
    external_roots: Iterable[str | os.PathLike[str] | Path] | None = None,
) -> tuple[Path, ...]:
    package = Path(package_root).expanduser().resolve(strict=False)
    roots = [package]
    if isinstance(external_roots, (str, os.PathLike, Path)):
        configured = (external_roots,)
    else:
        configured = external_roots or ()
    for value in configured:
        root_value = Path(value).expanduser()
        if not root_value.is_absolute():
            root_value = package / root_value
        root = root_value.resolve(strict=False)
        if not root.exists() or not root.is_dir():
            raise PathContractError(f"external root must be an existing directory: {root}")
        if root not in roots:
            roots.append(root)
    return tuple(roots)


def resolve_authorized_path(
    value: str | os.PathLike[str] | Path,
    *,
    package_root: str | os.PathLike[str] | Path,
    base_dir: str | os.PathLike[str] | Path | None = None,
    external_roots: Iterable[str | os.PathLike[str] | Path] | None = None,
    label: str = "input",
) -> Path:
    """Resolve a path and require containment in package or explicit roots.

    ``base_dir`` is the source manifest directory for file-backed manifests;
    otherwise it defaults to the selected package root.  Resolution follows
    symlinks before containment, so a package-relative symlink cannot escape
    into an unapproved directory.  Absolute external paths are accepted only
    when their root is explicitly listed in ``external_roots``.
    """

    package = Path(package_root).expanduser().resolve(strict=False)
    anchor = package if base_dir is None else Path(base_dir).expanduser().resolve(strict=False)
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = anchor / candidate
    resolved = candidate.resolve(strict=False)
    roots = authorized_roots(package, external_roots)
    if not any(resolved == root or resolved.is_relative_to(root) for root in roots):
        raise PathContractError(
            f"{label} escapes authorized package/external roots: {value!s} -> {resolved}"
        )
    return resolved


def display_path(path: Path, package_root: Path) -> str:
    """Use package-relative paths while preserving explicit external paths."""

    try:
        return path.resolve(strict=False).relative_to(package_root.resolve(strict=False)).as_posix()
    except ValueError:
        return path.resolve(strict=False).as_posix()


__all__ = ["PathContractError", "authorized_roots", "display_path", "resolve_authorized_path"]
