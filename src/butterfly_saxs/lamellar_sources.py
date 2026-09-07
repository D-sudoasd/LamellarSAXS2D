"""Native LamellarSAXS2D single and batch result bundle loading.

The loader is deliberately strict: it recognizes the package's JSON/NPZ
sidecars and never guesses from arbitrary CSV files or array names.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .lamellar import _q_unit_from_source, _source_identity
from .lamellar_utils import _copy_metadata, _read

_NATIVE_SINGLE_HINTS = {
    "observed",
    "image",
    "data",
    "qmap",
    "qx",
    "qy",
    "observables",
    "ellipse_fit",
    "lobe_radial_peaks",
    "metadata",
    "flags",
}
_NATIVE_SCHEMA_PREFIXES = ("lamellarsaxs2d.", "butterflysaxs.")


def _native_json_mapping(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    schema = str(value.get("schema_version", "") or "")
    if schema.startswith(_NATIVE_SCHEMA_PREFIXES):
        return True
    keys = set(str(key) for key in value)
    return len(keys & _NATIVE_SINGLE_HINTS) >= 3 and bool(
        keys
        & {
            "observed",
            "image",
            "data",
            "observables",
            "ellipse_fit",
            "lobe_radial_peaks",
        }
    )


def _native_batch_manifest(path: Path, value: Any, npz_path: Path | None) -> bool:
    """Recognize the package's legacy batch manifest by its full companion set."""

    if (
        path.name != "manifest.json"
        or not isinstance(value, Mapping)
        or not isinstance(value.get("frames"), list)
    ):
        return False
    if npz_path is None or npz_path.name != "results.npz" or not npz_path.is_file():
        return False
    ellipse_path = path.parent / "ellipse_fit.json"
    provenance_path = path.parent / "provenance.json"
    if not ellipse_path.is_file() or not provenance_path.is_file():
        return False
    provenance = value.get("provenance")
    if not isinstance(provenance, Mapping):
        try:
            provenance = _load_json(provenance_path)
        except ValueError:
            return False
    tool = str(provenance.get("tool", "") or "").casefold()
    return tool in {"butterflysaxs", "butterflysaxs2d", "lamellarsaxs2d"} and bool(
        provenance.get("versions")
    )


def _load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid native JSON bundle: {path}: {exc}") from exc


def _npz_keys(path: Path) -> list[str]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            return list(archive.files)
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid native NPZ bundle: {path}: {exc}") from exc


def _first_key(keys: Sequence[str], names: Sequence[str]) -> str | None:
    for name in names:
        if name in keys:
            return name
    return None


def _validate_array_shapes(
    archive: Any, key_map: Mapping[str, str], *, path: Path
) -> None:
    image_key = key_map.get("observed")
    qx_key = key_map.get("qx")
    qy_key = key_map.get("qy")
    if image_key is None:
        raise ValueError(f"native NPZ bundle has no observed/image array: {path}")
    image = np.asarray(archive[image_key])
    if image.ndim != 2:
        raise ValueError(f"native observed array must be two-dimensional: {path}")
    for label, key in (("qx", qx_key), ("qy", qy_key)):
        if key is not None and np.asarray(archive[key]).shape != image.shape:
            raise ValueError(
                f"native {label} shape does not match observed array: {path}"
            )


def _attach_single_npz(
    source: dict[str, Any], path: Path, *, embed: bool = True
) -> dict[str, Any]:
    keys = _npz_keys(path)
    key_map = {
        "observed": _first_key(keys, ("observed", "image", "data", "I")),
        "qx": _first_key(keys, ("qx", "qx_nm_inv")),
        "qy": _first_key(keys, ("qy", "qy_nm_inv")),
        "q": _first_key(keys, ("q", "q_nm_inv")),
    }
    if key_map["observed"] is None:
        raise ValueError(f"native NPZ bundle has no observed/image array: {path}")
    with np.load(path, allow_pickle=False) as archive:
        _validate_array_shapes(
            archive, {k: v for k, v in key_map.items() if v is not None}, path=path
        )
        if embed:
            source["observed"] = np.array(archive[key_map["observed"]], copy=True)
            if key_map["qx"] is not None:
                source["qx"] = np.array(archive[key_map["qx"]], copy=True)
            if key_map["qy"] is not None:
                source["qy"] = np.array(archive[key_map["qy"]], copy=True)
            if key_map["q"] is not None:
                source.setdefault("q", np.array(archive[key_map["q"]], copy=True))
            q_unit_key = _first_key(keys, ("q_unit", "q_unit_name"))
            if q_unit_key is not None:
                try:
                    source.setdefault(
                        "q_unit", str(np.asarray(archive[q_unit_key]).item())
                    )
                except Exception:
                    pass
        source["array_path"] = os.fspath(path)
        source["npz_path"] = os.fspath(path)
        source["array_keys"] = dict(key_map)
    return source


def _batch_npz_sources(
    frames: Sequence[Any], npz_path: Path, base: Path
) -> list[dict[str, Any]]:
    keys = _npz_keys(npz_path)
    result: list[dict[str, Any]] = []
    for index, item in enumerate(frames):
        frame = dict(item) if isinstance(item, Mapping) else {"frame_index": index}
        frame_index = int(_read(frame, ("frame_index", "index"), index) or index)
        prefix = f"frame_{frame_index:04d}__"
        key_map = {
            "observed": _first_key(
                keys,
                (prefix + "image", prefix + "observed", prefix + "data", prefix + "I"),
            ),
            "qx": _first_key(
                keys,
                (
                    prefix + "qmap__qx",
                    prefix + "qmap__qx_nm_inv",
                    prefix + "qx",
                    prefix + "qx_nm_inv",
                ),
            ),
            "qy": _first_key(
                keys,
                (
                    prefix + "qmap__qy",
                    prefix + "qmap__qy_nm_inv",
                    prefix + "qy",
                    prefix + "qy_nm_inv",
                ),
            ),
            "q": _first_key(
                keys,
                (
                    prefix + "qmap__q",
                    prefix + "qmap__q_nm_inv",
                    prefix + "q",
                    prefix + "q_nm_inv",
                ),
            ),
        }
        source = dict(frame)
        source.setdefault("frame_index", frame_index)
        if key_map["observed"] is not None:
            source["array_path"] = os.fspath(npz_path)
            source["npz_path"] = os.fspath(npz_path)
            source["array_keys"] = dict(key_map)
            source["array_validation"] = "deferred_to_selected_frame"
        else:
            # Failed/skipped frames are still sources.  Their selector and
            # status remain inspectable even though no detector array exists.
            source["array_error"] = "missing observed array in native NPZ"
        result.append(source)
    return result


def _resolve_native_pair(path: Path) -> tuple[Path | None, Path | None]:
    if path.is_dir():
        manifest = path / "manifest.json"
        if manifest.is_file():
            npz = path / "results.npz"
            return manifest, npz if npz.is_file() else None
        json_candidates = sorted(path.glob("*.json"))
        for candidate in json_candidates:
            if candidate.name in {
                "provenance.json",
                "frame_summary.json",
                "ellipse_fit.json",
                ".bundle.commit.json",
            }:
                continue
            npz = candidate.with_suffix(".npz")
            if npz.is_file():
                return candidate, npz
        return None, None
    if path.suffix.lower() == ".json":
        npz = path.with_suffix(".npz")
        return path, npz if npz.is_file() else None
    if path.suffix.lower() == ".npz":
        json_path = path.with_suffix(".json")
        return json_path if json_path.is_file() else None, path
    return None, None


def _merge_batch_measurements(
    sources: list[dict[str, Any]], manifest_path: Path
) -> None:
    """Attach exported lobe/ellipse summaries without touching detector arrays."""

    sidecar = manifest_path.parent / "ellipse_fit.json"
    if not sidecar.is_file():
        return
    try:
        payload = _load_json(sidecar)
    except ValueError:
        return
    rows = payload.get("frames") if isinstance(payload, Mapping) else None
    if not isinstance(rows, list):
        return
    by_index = {
        int(row.get("frame_index")): row
        for row in rows
        if isinstance(row, Mapping) and row.get("frame_index") is not None
    }
    for source in sources:
        try:
            index = int(source.get("frame_index"))
        except (TypeError, ValueError):
            continue
        row = by_index.get(index)
        if not row:
            continue
        for name in (
            "ellipse_fit",
            "lobe_radial_peaks",
            "lobe_radial_profiles",
            "lobe_angular",
        ):
            if row.get(name) is not None:
                target_name = "lobes" if name == "lobe_angular" else name
                source.setdefault(target_name, row[name])


def load_lamellar_sources(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """Load only native single/batch result bundles.

    Single-frame exports embed the selected arrays.  Batch exports retain a
    validated ``npz_path`` and logical ``array_keys`` mapping so callers can
    load one selected frame lazily without materialising every detector frame.
    CSV files and arbitrary JSON/NPZ files are rejected instead of guessed.
    """

    target = Path(path)
    json_path, npz_path = _resolve_native_pair(target)
    if json_path is None and npz_path is None:
        raise ValueError(f"not a native LamellarSAXS2D result bundle: {target}")
    summary: Any = _load_json(json_path) if json_path is not None else None
    if summary is not None and not (
        _native_json_mapping(summary)
        or _native_batch_manifest(json_path, summary, npz_path)
    ):
        raise ValueError(f"JSON is not a recognized native result bundle: {json_path}")
    if isinstance(summary, Mapping) and isinstance(summary.get("frames"), list):
        frames = list(summary["frames"])
        # Exported batch manifest frames carry selectors; ellipse_fit.json and
        # project-run records may carry the actual result under ``result``.
        if json_path is not None and json_path.name == "manifest.json":
            sources = (
                _batch_npz_sources(frames, npz_path, json_path.parent)
                if npz_path is not None
                else [dict(frame) for frame in frames]
            )
            if npz_path is None:
                for source in sources:
                    source.setdefault("array_path", None)
            _merge_batch_measurements(sources, json_path)
            return [
                _normalise_loaded_source(source, json_path.parent) for source in sources
            ]
        sources: list[dict[str, Any]] = []
        for index, item in enumerate(frames):
            if isinstance(item, Mapping):
                nested = (
                    item.get("result")
                    if isinstance(item.get("result"), Mapping)
                    else item
                )
                source = dict(nested)
                for key in (
                    "frame_index",
                    "frame_id",
                    "path",
                    "frame_selector",
                    "dataset",
                    "time",
                ):
                    if key in item and key not in source:
                        source[key] = item[key]
            else:
                source = {"frame_index": index}
            sources.append(source)
        if npz_path is not None:
            # A sidecar results.npz is preferred for a directory-level bundle.
            array_sources = _batch_npz_sources(sources, npz_path, json_path.parent)
            for source, arrays in zip(sources, array_sources, strict=False):
                source.update(
                    {
                        key: value
                        for key, value in arrays.items()
                        if key in {"array_path", "npz_path", "array_keys"}
                    }
                )
        return [
            _normalise_loaded_source(source, json_path.parent) for source in sources
        ]
    source = dict(summary) if isinstance(summary, Mapping) else {}
    if npz_path is not None:
        source = _attach_single_npz(source, npz_path, embed=True)
    elif not _native_json_mapping(source):
        raise ValueError(f"native single bundle has no usable summary: {target}")
    return [
        _normalise_loaded_source(
            source, json_path.parent if json_path is not None else target.parent
        )
    ]


def _normalise_loaded_source(source: Mapping[str, Any], base: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    array_names = {
        "observed",
        "image",
        "data",
        "I",
        "qx",
        "qy",
        "q",
        "qx_nm_inv",
        "qy_nm_inv",
        "q_nm_inv",
    }
    for key, value in source.items():
        if key in array_names and isinstance(value, np.ndarray):
            result[str(key)] = np.array(value, copy=True)
        else:
            result[str(key)] = _copy_metadata(value)
    if result.get("array_path") and not Path(str(result["array_path"])).is_absolute():
        result["array_path"] = os.fspath((base / str(result["array_path"])).resolve())
    if result.get("npz_path") and not Path(str(result["npz_path"])).is_absolute():
        result["npz_path"] = os.fspath((base / str(result["npz_path"])).resolve())
    if "source_identity" not in result:
        result["source_identity"] = _source_identity(result, result)
    if "q_unit" not in result:
        result["q_unit"] = _q_unit_from_source(result)
    return result


__all__ = ["load_lamellar_sources"]
