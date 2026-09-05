"""Shared path/discovery rules for image inputs and manifests."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Any


IMAGE_SUFFIXES = frozenset(
    {
        ".cbf", ".edf", ".tif", ".tiff", ".npy", ".npz", ".h5", ".hdf5",
        ".hdf", ".csv", ".txt",
    }
)


def is_supported_image_path(value: Any) -> bool:
    try:
        return Path(value).suffix.casefold() in IMAGE_SUFFIXES
    except (TypeError, ValueError):
        return False


def filter_supported_image_paths(values: Iterable[Any]) -> list[Path]:
    """Return files only, excluding detector sidecars such as ``.dat``."""

    result: list[Path] = []
    for value in values:
        try:
            path = Path(value)
        except (TypeError, ValueError):
            continue
        if path.is_file() and is_supported_image_path(path):
            result.append(path)
    return result


def canonical_path(value: str | os.PathLike[str] | Path) -> str:
    try:
        path = Path(value).expanduser().resolve(strict=False)
    except (OSError, RuntimeError):
        path = Path(os.path.abspath(os.fspath(value)))
    return os.path.normcase(path.as_posix()).replace("\\", "/")


__all__ = [
    "IMAGE_SUFFIXES",
    "canonical_path",
    "filter_supported_image_paths",
    "is_supported_image_path",
]
