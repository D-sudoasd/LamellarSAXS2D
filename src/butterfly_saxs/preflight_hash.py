"""Content hashing and read-time provenance for preflight inputs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, *, error_type: type[Exception] = ValueError) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise error_type(f"cannot read input for SHA-256: {path}: {exc}") from exc
    return digest.hexdigest()


def hashable_inline(value: Any, *, json_safe: Callable[[Any], Any]) -> Any:
    """Describe inline arrays without expanding detector-sized data to JSON."""

    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return {
            "__ndarray__": True,
            "dtype": str(array.dtype),
            "shape": [int(item) for item in array.shape],
            "sha256": sha256_bytes(array.tobytes()),
        }
    if isinstance(value, Mapping):
        return {
            str(key): hashable_inline(item, json_safe=json_safe)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [hashable_inline(item, json_safe=json_safe) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return hashable_inline(value.item(), json_safe=json_safe)
    return json_safe(value)


def inline_hash(value: Any, *, json_safe: Callable[[Any], Any]) -> str:
    canonical = json.dumps(
        hashable_inline(value, json_safe=json_safe),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return sha256_bytes(canonical.encode("utf-8"))


def read_file_record(
    path: Path,
    package: Path,
    reader: Callable[[], Any],
    records: list[dict[str, Any]],
    *,
    display_path: Callable[[Path, Path], str],
    error_type: type[Exception] = ValueError,
) -> Any:
    """Run a reader bracketed by SHA-256 checks and record both digests."""

    before = sha256_file(path, error_type=error_type)
    try:
        value = reader()
    except Exception as exc:
        after: str | None = None
        post_hash_error: str | None = None
        try:
            after = sha256_file(path, error_type=error_type)
        except Exception as hash_exc:  # pragma: no cover - damaged/removed source
            post_hash_error = f"{type(hash_exc).__name__}: {hash_exc}"
        record = {
            "path": display_path(path, package),
            "algorithm": "sha256",
            "sha256_before": before,
            "sha256_after": after,
            "before": before,
            "after": after,
            "unchanged": bool(after is not None and before == after),
            "read_status": "error",
            "read_error": f"{type(exc).__name__}: {exc}",
        }
        if post_hash_error is not None:
            record["post_read_hash_error"] = post_hash_error
        records.append(record)
        if after is not None and before != after:
            raise error_type(f"input changed while being read: {path}") from exc
        raise
    after = sha256_file(path, error_type=error_type)
    record = {
        "path": display_path(path, package),
        "algorithm": "sha256",
        "sha256_before": before,
        "sha256_after": after,
        "before": before,
        "after": after,
        "unchanged": before == after,
    }
    records.append(record)
    if before != after:
        raise error_type(f"input changed while being read: {path}")
    return value


__all__ = [
    "hashable_inline",
    "inline_hash",
    "read_file_record",
    "sha256_bytes",
    "sha256_file",
]
