"""Create a read-only, blinded annotation pack for SAXS image frames.

The pack is deliberately independent of the fitting pipeline.  It contains
blind PNGs and identity-prefilled human-annotation templates, while the manifest and
status file preserve enough provenance to audit exactly which input frame was
shown.  PONI and mask inputs are recorded only; they are never used to fit a
model or to select a candidate.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .batch import FrameRef, build_frame_refs
from .io import load_image


ANNOTATION_PACK_SCHEMA_VERSION = "lamellarsaxs2d.annotation_pack.v2"
ANNOTATION_COORDINATE_SYSTEM = "image_pixel_x_right_y_up_origin_lower_left"
_MANIFEST_KEYS = ("frames", "frame_manifest", "manifest", "data", "items")
_PATH_KEYS = ("path", "input_path", "file", "filename", "source_path")
_MANIFEST_COLUMNS = (
    "blind_id",
    "role",
    "source_path",
    "source_path_relative_package",
    "selector",
    "sha256",
    "selection_reason",
)
_ANNOTATION_COLUMNS = (
    "blind_id",
    "valid_area",
    "beamstop",
    "streak",
    "overlap",
    "lobe_center_x",
    "lobe_center_y",
    "ridge_points",
    "software",
    "software_version",
    "coordinate_system",
    "image_version",
    "annotation_time",
    "annotator",
    "notes",
)
_CONSENSUS_COLUMNS = (
    "blind_id",
    "consensus_status",
    "valid_area",
    "beamstop",
    "streak",
    "overlap",
    "lobe_center_x",
    "lobe_center_y",
    "ridge_points",
    "reviewer",
    "software",
    "software_version",
    "coordinate_system",
    "image_version",
    "review_time",
    "notes",
)


class AnnotationPackError(ValueError):
    """Raised when the inputs cannot produce the fixed eight-frame pack."""


@dataclass
class _InputHash:
    """A read-before/read-after digest for one input source."""

    kind: str
    path: str
    before: str
    after: str
    selector: str = ""
    source: str = "file"

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "path": self.path,
            "selector": self.selector,
            "sha256_before": self.before,
            "sha256_after": self.after,
            "unchanged": self.before == self.after,
            "source": self.source,
        }


@dataclass
class _Candidate:
    ref: FrameRef
    group: str
    source: Path
    selector: str
    summary: dict[str, float]
    missing_metadata: bool
    source_index: int
    array: np.ndarray | None = None
    sha256: str = ""
    role: str = ""
    reason: str = ""
    blind_id: str = ""

    @property
    def identity(self) -> tuple[str, str, int | None, str]:
        return (
            _canonical_path(self.source),
            self.selector,
            self.ref.frame_selector if isinstance(self.ref.frame_selector, int) else None,
            self.ref.dataset_id,
        )


def _canonical_path(path: str | os.PathLike[str] | Path) -> str:
    resolved = Path(path).expanduser().resolve(strict=False)
    return os.path.normcase(resolved.as_posix()).replace("\\", "/")


def _display_path(path: Path, package: Path) -> str:
    """Return a package-relative POSIX path, including ``..`` when needed."""

    try:
        return path.resolve(strict=False).relative_to(package).as_posix()
    except ValueError:
        return Path(os.path.relpath(path.resolve(strict=False), package)).as_posix()


def _resolve_input_path(package: Path, value: str | os.PathLike[str] | Path) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = package / candidate
    return candidate.resolve(strict=False)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise AnnotationPackError(f"无法计算输入文件 SHA-256：{path}: {exc}") from exc
    return digest.hexdigest()


def _hashable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return str(value)
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _hashable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_hashable(item) for item in value]
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return {
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "sha256": _sha256_bytes(array.tobytes()),
        }
    if isinstance(value, np.generic):
        return _hashable(value.item())
    if isinstance(value, FrameRef):
        return _hashable(value.to_dict())
    return str(value)


def _hash_inline(value: Any) -> str:
    encoded = json.dumps(
        _hashable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _relative_or_inline_path(value: Any, package: Path, label: str) -> str:
    if isinstance(value, (str, os.PathLike, Path)):
        return _display_path(_resolve_input_path(package, value), package)
    return label


def _read_manifest_file(path: Path) -> Any:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise AnnotationPackError(f"无法读取 manifest：{path}: {exc}") from exc
    try:
        suffix = path.suffix.casefold()
        if suffix == ".csv":
            return list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
        if suffix == ".json":
            return json.loads(raw.decode("utf-8-sig"))
        if suffix == ".toml":
            return tomllib.loads(raw.decode("utf-8-sig"))
    except (UnicodeError, json.JSONDecodeError, ValueError, tomllib.TOMLDecodeError) as exc:
        raise AnnotationPackError(f"无法解析 manifest：{path}: {exc}") from exc
    raise AnnotationPackError("manifest 必须是 CSV、JSON 或 TOML 文件")


def _manifest_rows(value: Any) -> list[Any]:
    if isinstance(value, FrameRef):
        return [value]
    if isinstance(value, Mapping):
        for key in _MANIFEST_KEYS:
            candidate = value.get(key)
            if isinstance(candidate, Sequence) and not isinstance(candidate, (str, bytes)):
                return list(candidate)
        if any(key in value for key in _PATH_KEYS):
            return [value]
        rows: list[dict[str, Any]] = []
        for path, metadata in value.items():
            row = dict(metadata) if isinstance(metadata, Mapping) else {"time": metadata}
            row.setdefault("path", path)
            rows.append(row)
        return rows
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return list(value)
    raise AnnotationPackError("manifest 必须包含 frame 行序列")


def _row_with_resolved_path(row: Any, base_dir: Path) -> Any:
    if isinstance(row, FrameRef):
        path = Path(row.path).expanduser()
        if not path.is_absolute():
            path = base_dir / path
        return FrameRef(
            path.resolve(strict=False),
            time=row.time,
            frame_id=row.frame_id,
            metadata=row.metadata,
            order=row.order,
            source=row.source,
            dataset=row.dataset,
            frame=row.frame,
        )
    if isinstance(row, Mapping):
        result = dict(row)
        path_key = next((key for key in _PATH_KEYS if result.get(key) is not None), None)
        if path_key is None:
            raise AnnotationPackError(f"manifest 行缺少 path：{row!r}")
        path = Path(result[path_key]).expanduser()
        if not path.is_absolute():
            path = base_dir / path
        resolved_path = str(path.resolve(strict=False))
        result[path_key] = resolved_path
        # ``FrameRef`` accepts path/input_path/file/filename.  Keep a
        # source_path alias usable in a manifest while giving the shared
        # resolver its canonical ``path`` field.
        result["path"] = resolved_path
        for key in ("frame", "frame_index", "dataset", "dataset_id", "dataset_name", "order", "time", "timestamp"):
            if key in result and isinstance(result[key], str) and not result[key].strip():
                result[key] = None
        return result
    if isinstance(row, (str, os.PathLike, Path)):
        path = Path(row).expanduser()
        if not path.is_absolute():
            path = base_dir / path
        return {"path": str(path.resolve(strict=False))}
    raise AnnotationPackError(f"manifest 行不是路径或 mapping：{row!r}")


def _load_manifest(
    package: Path,
    manifest: Any,
    *,
    kind: str,
    hashes: list[_InputHash],
) -> list[FrameRef]:
    if manifest is None:
        raise AnnotationPackError(f"{kind} manifest 不能为空")
    base_dir = package
    if isinstance(manifest, (str, os.PathLike, Path)):
        source = _resolve_input_path(package, manifest)
        if not source.is_file():
            raise AnnotationPackError(f"{kind} manifest 不存在：{source}")
        before = _sha256_file(source)
        parsed = _read_manifest_file(source)
        after = _sha256_file(source)
        hashes.append(
            _InputHash(
                kind=f"{kind}_manifest",
                path=_display_path(source, package),
                before=before,
                after=after,
            )
        )
        base_dir = source.parent
    else:
        parsed = manifest
        digest = _hash_inline(parsed)
        hashes.append(
            _InputHash(
                kind=f"{kind}_manifest",
                path=f"{kind}_manifest:in-memory",
                before=digest,
                after=digest,
                source="in-memory",
            )
        )
    rows = [_row_with_resolved_path(row, base_dir) for row in _manifest_rows(parsed)]
    if not rows:
        raise AnnotationPackError(f"{kind} manifest 没有 frame")
    try:
        return build_frame_refs([], manifest=rows)
    except (TypeError, ValueError) as exc:
        raise AnnotationPackError(f"无法解析 {kind} manifest：{exc}") from exc


def _selector(ref: FrameRef) -> str:
    parts: list[str] = []
    if ref.frame_selector is not None:
        parts.append(f"frame={ref.frame_selector}")
    if ref.dataset_id:
        parts.append(f"dataset={ref.dataset_id}")
    return ",".join(parts) if parts else "default"


def _file_hash_for_frame(
    path: Path,
    *,
    package: Path,
    selector: str,
    hashes: list[_InputHash],
    ref: FrameRef,
) -> tuple[np.ndarray, str]:
    if not path.is_file():
        raise AnnotationPackError(f"manifest frame 不存在：{path}")
    before = _sha256_file(path)
    try:
        loaded = load_image(
            path,
            frame=ref.frame_selector,
            dataset=ref.dataset_id or None,
        )
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        raise AnnotationPackError(f"无法读取 frame {path} ({selector})：{exc}") from exc
    after = _sha256_file(path)
    hashes.append(
        _InputHash(
            kind="frame",
            path=_display_path(path, package),
            selector=selector,
            before=before,
            after=after,
        )
    )
    if before != after:
        raise AnnotationPackError(f"frame 在读取期间发生变化：{path}")
    array = np.asarray(loaded.data)
    if array.ndim != 2 or array.size == 0:
        raise AnnotationPackError(f"frame 必须是非空二维数组：{path}，shape={array.shape!r}")
    if array.dtype.kind not in "biuf":
        raise AnnotationPackError(f"frame 必须是实数数值数组：{path}，dtype={array.dtype}")
    return np.array(array, copy=True), before


def _candidate_shell(
    ref: FrameRef,
    *,
    group: str,
    source_index: int,
    role: str = "",
    reason: str = "",
) -> _Candidate:
    """Create a candidate descriptor without reading its detector array."""

    metadata = dict(ref.metadata or {})
    missing_metadata = ref.time is None and not any(
        key in metadata for key in ("time", "timestamp", "temperature")
    )
    return _Candidate(
        ref=ref,
        group=group,
        source=Path(ref.path).resolve(strict=False),
        selector=_selector(ref),
        summary={
            "finite_fraction": 0.0,
            "negative_fraction": 0.0,
            "robust_high_fraction": 0.0,
            "nonfinite_fraction": 0.0,
        },
        missing_metadata=missing_metadata,
        source_index=source_index,
        role=role,
        reason=reason,
    )


def _intensity_summary(array: np.ndarray) -> dict[str, float]:
    values = np.asarray(array, dtype=float)
    finite = values[np.isfinite(values)]
    total = float(values.size)
    if finite.size == 0:
        return {
            "finite_fraction": 0.0,
            "negative_fraction": 0.0,
            "robust_high_fraction": 0.0,
            "nonfinite_fraction": 1.0,
        }
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    threshold = median + 6.0 * 1.4826 * mad
    if not math.isfinite(threshold):
        threshold = float(np.max(finite))
    return {
        "finite_fraction": float(finite.size / total) if total else 0.0,
        "negative_fraction": float(np.count_nonzero(finite < 0) / finite.size),
        "robust_high_fraction": float(np.count_nonzero(finite > threshold) / finite.size),
        "nonfinite_fraction": float(1.0 - finite.size / total) if total else 1.0,
    }


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _preflight_rows(value: Any) -> list[Mapping[str, Any]]:
    """Extract only raw frame records; fitting-related sections are ignored."""

    if not isinstance(value, Mapping):
        return []
    extension = value.get("extensions")
    if isinstance(extension, Mapping):
        preflight = extension.get("preflight")
        if isinstance(preflight, Mapping) and isinstance(preflight.get("frames"), Sequence):
            return [row for row in preflight["frames"] if isinstance(row, Mapping)]
    frames = value.get("frames")
    if isinstance(frames, Sequence) and not isinstance(frames, (str, bytes)):
        return [row for row in frames if isinstance(row, Mapping)]
    return []


def _preflight_quality_index(value: Any, package: Path) -> dict[str, list[dict[str, Any]]]:
    index: dict[str, list[dict[str, Any]]] = {}
    for row in _preflight_rows(value):
        nested = row.get("manifest_frame")
        nested = nested if isinstance(nested, Mapping) else {}
        raw_path = row.get("path")
        if raw_path is None:
            raw_path = nested.get("path")
        if raw_path is None:
            continue
        path = _resolve_input_path(package, str(raw_path))
        frame = row.get("frame", nested.get("frame", nested.get("frame_index")))
        dataset = row.get("dataset", nested.get("dataset", nested.get("dataset_id")))
        try:
            frame_key: int | None = int(frame) if frame is not None and str(frame).strip() else None
        except (TypeError, ValueError):
            frame_key = None
        dataset_key = "" if dataset is None else str(dataset)
        summary = row.get("summary")
        summary = summary if isinstance(summary, Mapping) else row
        quality = {
            "negative_fraction": _finite_float(
                summary.get("negative_fraction", summary.get("negativeFraction"))
            ),
            "robust_high_fraction": _finite_float(
                summary.get("robust_high_fraction", summary.get("robustHighFraction"))
            ),
            "finite_fraction": _finite_float(summary.get("finite_fraction")),
        }
        metadata = nested.get("metadata")
        if not isinstance(metadata, Mapping):
            metadata = row.get("metadata") if isinstance(row.get("metadata"), Mapping) else {}
        quality["missing_metadata"] = bool(
            nested.get("time") is None
            and nested.get("timestamp") is None
            and not metadata
        )
        item = {key: val for key, val in quality.items() if val is not None}
        item["frame"] = frame_key
        item["dataset"] = dataset_key
        key = f"{_canonical_path(path)}|{frame_key}|{dataset_key}"
        index.setdefault(key, []).append(item)
        index.setdefault(f"{_canonical_path(path)}|path-only", []).append(item)
    return index


def _quality_for_candidate(
    candidate: _Candidate,
    preflight_index: Mapping[str, list[dict[str, Any]]],
) -> dict[str, float | bool]:
    path_key = _canonical_path(candidate.source)
    frame = candidate.ref.frame_selector
    frame_key = frame if isinstance(frame, int) else None
    dataset = candidate.ref.dataset_id
    matches = preflight_index.get(f"{path_key}|{frame_key}|{dataset}", [])
    if not matches:
        path_matches = preflight_index.get(f"{path_key}|path-only", [])
        matches = path_matches if len(path_matches) == 1 else []
    quality = dict(candidate.summary)
    quality["missing_metadata"] = candidate.missing_metadata
    if matches:
        supplied = matches[0]
        for key in ("negative_fraction", "robust_high_fraction", "finite_fraction"):
            if supplied.get(key) is not None:
                quality[key] = float(supplied[key])
        if supplied.get("missing_metadata"):
            quality["missing_metadata"] = True
    return quality


def _difficulty_reason(quality: Mapping[str, Any]) -> str:
    signals: list[str] = []
    negative = float(quality.get("negative_fraction", 0.0))
    robust_high = float(quality.get("robust_high_fraction", 0.0))
    nonfinite = float(quality.get("nonfinite_fraction", 0.0))
    if negative > 0:
        signals.append(f"negative_fraction={negative:.6g}")
    if robust_high > 0:
        signals.append(f"robust_high_fraction={robust_high:.6g}")
    if bool(quality.get("missing_metadata")):
        signals.append("missing_metadata")
    if nonfinite > 0:
        signals.append(f"nonfinite_fraction={nonfinite:.6g}")
    return "; ".join(signals) if signals else "deterministic fallback: no raw quality signal"


def _difficulty_key(quality: Mapping[str, Any], source_index: int) -> tuple[float, ...]:
    return (
        float(quality.get("negative_fraction", 0.0)),
        float(quality.get("robust_high_fraction", 0.0)),
        float(bool(quality.get("missing_metadata"))),
        float(quality.get("nonfinite_fraction", 0.0)),
        -float(source_index),
    )


def _unique_refs(refs: Sequence[FrameRef]) -> list[FrameRef]:
    result: list[FrameRef] = []
    seen: set[tuple[str, str, int | None, str]] = set()
    for ref in refs:
        path = Path(ref.path).resolve(strict=False)
        frame = ref.frame_selector
        identity = (
            _canonical_path(path),
            _selector(ref),
            frame if isinstance(frame, int) else None,
            ref.dataset_id,
        )
        if identity in seen:
            continue
        seen.add(identity)
        result.append(ref)
    return result


def _pick_unique(
    refs: Sequence[FrameRef], preferred: int, used: set[tuple[str, str, int | None, str]]
) -> FrameRef | None:
    order = [preferred]
    for distance in range(1, len(refs)):
        left = preferred - distance
        right = preferred + distance
        if left >= 0:
            order.append(left)
        if right < len(refs):
            order.append(right)
    for index in order:
        ref = refs[index]
        frame = ref.frame_selector
        identity = (
            _canonical_path(Path(ref.path)),
            _selector(ref),
            frame if isinstance(frame, int) else None,
            ref.dataset_id,
        )
        if identity not in used:
            used.add(identity)
            return ref
    return None


def _make_protocol() -> str:
    return """# R0 盲标协议

本包中的 `blind_001.png`–`blind_008.png` 是仅供人工标注的二维强度图。文件名使用盲化编号；图像不叠加拟合、模型、ridge、椭圆或其他算法输出。`PONI` 和 mask 仅作为显示/记录输入，不用于拟合。

协调者只向标注者分发 `blind_payload/` 目录，不得分发其上级目录中的 `annotation_manifest.csv` 或 `annotation_status.json`。后两者含源帧角色与选择理由，只供标注结束后的审计。

## 标注设计

1. 两名标注者分别在不知道算法结果、候选选择理由和对方标注的条件下独立完成 `annotator_a.csv` 与 `annotator_b.csv`。
2. 如果暂时只有一名领域专家，应在间隔至少一周的两次独立盲化会话中完成两份标注；该证据标记为“单人重复”，不能等同于双人独立标注。
3. 分歧由第三次复核或预先指定的复核者形成 `consensus_review.csv`。共识形成前不得回看算法结果。
4. 记录每次标注的 `software`、`software_version`、`coordinate_system`、`image_version` 和时间字段。
5. 本包不预设 acceptance threshold；阈值必须依据重复标注误差和仪器分辨率另行冻结。

## 坐标和记录

PNG 采用图像像素坐标：`x` 为列方向向右，`y` 为行方向向上，原点为左下角显示像素中心；图像显示使用 `origin=lower`。模板已预填 `blind_id`、固定坐标系和对应 PNG 的 SHA-256，三列不得修改。`valid_area` 必须是至少 3 个不同有限 `[x,y]` 点组成、面积大于 0 的 JSON 多边形；beamstop、streak、overlap 和 ridge 点使用 JSON 数组文本记录，明确不存在时写 `[]`。lobe 中心使用数值像素坐标；其他确实无法判断的项目可写 `unknown`，但不能把 `valid_area` 留空或用 `[]`/`unknown` 代替。除 `notes` 外，每个字段都必须填写。

## 版本与审计

上级目录的 `annotation_manifest.csv` 保存盲化编号、角色、包内相对源路径、selector、源文件 SHA-256 和选择理由，仅供协调者审计，不应在 consensus 前展示给标注者。`annotation_status.json` 在人工填写前固定为 `awaiting_human_annotations` 且 `human_consensus=false`。原始输入只读，若读前后 SHA-256 不一致，应停止使用该包并重新生成。
"""


def _write_csv(path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, Any]] = ()) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def _write_blind_png(path: Path, array: np.ndarray, vmin: float, vmax: float) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("magma").with_extremes(bad="black")
    figure, axis = plt.subplots(figsize=(4.0, 4.0), dpi=150)
    axis.imshow(
        np.ma.masked_invalid(np.asarray(array, dtype=float)),
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        aspect="equal",
    )
    axis.axis("off")
    figure.subplots_adjust(0, 0, 1, 1)
    figure.savefig(path, format="png", dpi=150, pad_inches=0)
    plt.close(figure)


def _global_display_scale(candidates: Sequence[_Candidate]) -> tuple[float, float]:
    finite_arrays = [
        np.asarray(candidate.array, dtype=float)[np.isfinite(candidate.array)]
        for candidate in candidates
    ]
    finite_parts = [array for array in finite_arrays if array.size]
    if not finite_parts:
        return -0.5, 0.5
    finite = np.concatenate(finite_parts)
    vmin, vmax = np.percentile(finite, [1.0, 99.5])
    if not (math.isfinite(float(vmin)) and math.isfinite(float(vmax))) or vmax <= vmin:
        vmin, vmax = float(np.min(finite)), float(np.max(finite))
    if vmax <= vmin:
        center = float(vmin)
        delta = max(abs(center) * 0.01, 0.5)
        vmin, vmax = center - delta, center + delta
    return float(vmin), float(vmax)


def _record_auxiliary_input(
    value: Any,
    *,
    package: Path,
    kind: str,
    hashes: list[_InputHash],
) -> str | None:
    if value is None:
        return None
    if isinstance(value, (str, os.PathLike, Path)):
        path = _resolve_input_path(package, value)
        if not path.is_file():
            raise AnnotationPackError(f"{kind} 文件不存在：{path}")
        before = _sha256_file(path)
        after = _sha256_file(path)
        hashes.append(
            _InputHash(
                kind=kind,
                path=_display_path(path, package),
                before=before,
                after=after,
            )
        )
        return _display_path(path, package)
    digest = _hash_inline(value)
    hashes.append(
        _InputHash(
            kind=kind,
            path=f"{kind}:in-memory",
            before=digest,
            after=digest,
            source="in-memory",
        )
    )
    return f"{kind}:in-memory"


def _read_preflight(
    value: Any,
    *,
    package: Path,
    hashes: list[_InputHash],
) -> tuple[dict[str, list[dict[str, Any]]], str | None]:
    if value is None:
        return {}, None
    if isinstance(value, (str, os.PathLike, Path)):
        path = _resolve_input_path(package, value)
        if not path.is_file():
            raise AnnotationPackError(f"preflight JSON 不存在：{path}")
        before = _sha256_file(path)
        try:
            parsed = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise AnnotationPackError(f"无法解析 preflight JSON：{path}: {exc}") from exc
        after = _sha256_file(path)
        hashes.append(
            _InputHash(
                kind="preflight_json",
                path=_display_path(path, package),
                before=before,
                after=after,
            )
        )
        return _preflight_quality_index(parsed, package), _display_path(path, package)
    if not isinstance(value, Mapping):
        raise AnnotationPackError("preflight_json 必须是 JSON 路径或 mapping")
    digest = _hash_inline(value)
    hashes.append(
        _InputHash(
            kind="preflight_json",
            path="preflight_json:in-memory",
            before=digest,
            after=digest,
            source="in-memory",
        )
    )
    return _preflight_quality_index(value, package), "preflight_json:in-memory"


def _verify_input_hashes(hashes: Sequence[_InputHash], package: Path) -> None:
    for item in hashes:
        if item.source == "in-memory":
            continue
        path = _resolve_input_path(package, item.path)
        item.after = _sha256_file(path)
        if item.before != item.after:
            raise AnnotationPackError(f"输入文件在生成期间发生变化：{path}")


def _output_hashes(output: Path, names: Sequence[str]) -> dict[str, str]:
    return {name: _sha256_file(output / name) for name in names}


def build_annotation_pack(
    package: str | os.PathLike[str] | Path,
    rt_manifest: Any,
    hold_manifest: Any,
    output: str | os.PathLike[str] | Path,
    *,
    preflight_json: Any = None,
    poni: Any = None,
    mask: Any = None,
) -> dict[str, Any]:
    """Build a fixed eight-frame, human-only annotation pack.

    Parameters are read-only inputs.  ``rt_manifest`` and ``hold_manifest``
    may be CSV/JSON/TOML paths, manifest mappings, or sequences of frame rows.
    Relative manifest-row paths are resolved beside a manifest file and
    relative to ``package`` for in-memory manifests.  ``output`` must name a
    new directory; an existing directory is never overwritten.

    The returned mapping contains absolute output paths and a compact status
    summary.  No fitting or algorithm-result field is copied into the pack.
    """

    package_root = Path(package).expanduser().resolve(strict=False)
    if not package_root.is_dir():
        raise AnnotationPackError(f"package 必须是已存在的目录：{package_root}")
    output_root = Path(output).expanduser().resolve(strict=False)
    if any(part.casefold() == "data_local" for part in output_root.parts):
        raise AnnotationPackError("标注包派生证据不得写入 data_local 原始数据目录")
    if output_root == package_root or output_root.is_relative_to(package_root):
        raise AnnotationPackError(
            "标注包输出目录不得位于只读原始数据包内部；请使用 results/validation/annotations 等新目录"
        )
    if output_root.exists():
        raise FileExistsError(f"输出目录已存在，默认不覆盖：{output_root}")

    hashes: list[_InputHash] = []
    rt_refs = _load_manifest(package_root, rt_manifest, kind="RT", hashes=hashes)
    hold_refs = _load_manifest(package_root, hold_manifest, kind="hold", hashes=hashes)
    if not rt_refs:
        raise AnnotationPackError("RT manifest 没有可用 frame")
    if len(_unique_refs(hold_refs)) < 3:
        raise AnnotationPackError("hold manifest 至少需要首帧、中间帧和末帧三个唯一 frame")

    preflight_index, preflight_path = _read_preflight(
        preflight_json, package=package_root, hashes=hashes
    )
    poni_path = _record_auxiliary_input(
        poni, package=package_root, kind="poni", hashes=hashes
    )
    mask_path = _record_auxiliary_input(
        mask, package=package_root, kind="mask", hashes=hashes
    )

    all_refs = _unique_refs([*rt_refs, *hold_refs])
    if len(all_refs) < 8:
        raise AnnotationPackError(
            f"RT/hold manifest 合计只有 {len(all_refs)} 个唯一 frame，固定盲标包需要 8 个"
        )
    used: set[tuple[str, str, int | None, str]] = set()
    selected_specs: list[tuple[str, FrameRef, str]] = []
    rt_ref = _pick_unique(rt_refs, 0, used)
    if rt_ref is None:
        raise AnnotationPackError("无法选择唯一 RT frame")
    selected_specs.append(("RT", rt_ref, "RT manifest 的参考 frame"))

    hold_unique = _unique_refs(hold_refs)
    hold_positions = (
        ("hold_first", 0, "hold 序列首帧"),
        ("hold_middle", len(hold_unique) // 2, "hold 序列中间帧"),
        ("hold_last", len(hold_unique) - 1, "hold 序列末帧"),
    )
    for role, position, reason in hold_positions:
        ref = _pick_unique(hold_unique, position, used)
        if ref is None:
            raise AnnotationPackError(f"无法选择唯一 {role}")
        selected_specs.append((role, ref, reason))

    selected_identities = {
        (
            _canonical_path(Path(ref.path)),
            _selector(ref),
            ref.frame_selector if isinstance(ref.frame_selector, int) else None,
            ref.dataset_id,
        )
        for _, ref, _ in selected_specs
    }
    remaining_refs = [
        ref
        for ref in all_refs
        if (
            _canonical_path(Path(ref.path)),
            _selector(ref),
            ref.frame_selector if isinstance(ref.frame_selector, int) else None,
            ref.dataset_id,
        )
        not in selected_identities
    ]
    if len(remaining_refs) < 4:
        raise AnnotationPackError("可用于困难候选的唯一 frame 少于 4 个")

    candidates: list[_Candidate] = []
    for source_index, (role, ref, reason) in enumerate(
        selected_specs + [("", ref, "") for ref in remaining_refs]
    ):
        candidates.append(
            _candidate_shell(
                ref,
                group="RT" if ref in rt_refs else "hold",
                source_index=source_index,
                role=role,
                reason=reason,
            )
        )

    # A supplied preflight already contains the raw intensity summaries needed
    # to rank difficult candidates.  Do not load unselected detector arrays in
    # that case.  Without preflight, stream one array at a time and retain only
    # the small summary for ranking; selected arrays are read again below.
    if preflight_json is None:
        for candidate in candidates:
            array, _ = _file_hash_for_frame(
                candidate.source,
                package=package_root,
                selector=candidate.selector,
                hashes=hashes,
                ref=candidate.ref,
            )
            candidate.summary = _intensity_summary(array)
            del array

    selected = candidates[:4]
    remaining_candidates = candidates[4:]
    scored: list[tuple[tuple[float, ...], _Candidate, dict[str, Any]]] = []
    for candidate in remaining_candidates:
        quality = _quality_for_candidate(candidate, preflight_index)
        scored.append((_difficulty_key(quality, candidate.source_index), candidate, quality))
    scored.sort(key=lambda item: item[0], reverse=True)
    for index, (_, candidate, quality) in enumerate(scored[:4], start=1):
        candidate.role = f"difficult_{index}"
        candidate.reason = _difficulty_reason(quality)
        selected.append(candidate)

    if len(selected) != 8 or len({item.identity for item in selected}) != 8:
        raise AnnotationPackError("内部选择结果不是 8 个唯一 frame")
    for index, candidate in enumerate(selected, start=1):
        candidate.blind_id = f"blind_{index:03d}"

    # Only the final eight selected frames are retained for the common display
    # scale and PNG rendering.  This is the only load path when preflight is
    # available, so a large sequence remains memory-bounded.
    for candidate in selected:
        array, digest = _file_hash_for_frame(
            candidate.source,
            package=package_root,
            selector=candidate.selector,
            hashes=hashes,
            ref=candidate.ref,
        )
        if candidate.sha256 and candidate.sha256 != digest:
            raise AnnotationPackError(f"frame 在两次读取之间发生变化：{candidate.source}")
        candidate.array = array
        candidate.sha256 = digest
    vmin, vmax = _global_display_scale(selected)

    # Refuse to create a partial output if a manifest, preflight, geometry,
    # mask, or source frame changed during input preparation.
    _verify_input_hashes(hashes, package_root)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    output_root.mkdir()
    payload_root = output_root / "blind_payload"
    payload_root.mkdir()
    for candidate in selected:
        if candidate.array is None:
            raise AnnotationPackError(f"缺少入选 frame 数组：{candidate.blind_id}")
        _write_blind_png(
            payload_root / f"{candidate.blind_id}.png", candidate.array, vmin, vmax
        )
        candidate.array = None
    manifest_rows = [
        {
            "blind_id": candidate.blind_id,
            "role": candidate.role,
            "source_path": _display_path(candidate.source, package_root),
            "source_path_relative_package": _display_path(candidate.source, package_root),
            "selector": candidate.selector,
            "sha256": candidate.sha256,
            "selection_reason": candidate.reason,
        }
        for candidate in selected
    ]
    _write_csv(output_root / "annotation_manifest.csv", _MANIFEST_COLUMNS, manifest_rows)
    (payload_root / "annotation_protocol.md").write_text(_make_protocol(), encoding="utf-8")
    blind_image_hashes = {
        candidate.blind_id: _sha256_file(payload_root / f"{candidate.blind_id}.png")
        for candidate in selected
    }
    annotation_rows = [
        {
            "blind_id": candidate.blind_id,
            "coordinate_system": ANNOTATION_COORDINATE_SYSTEM,
            "image_version": blind_image_hashes[candidate.blind_id],
        }
        for candidate in selected
    ]
    consensus_rows = [
        {
            "blind_id": candidate.blind_id,
            "coordinate_system": ANNOTATION_COORDINATE_SYSTEM,
            "image_version": blind_image_hashes[candidate.blind_id],
        }
        for candidate in selected
    ]
    _write_csv(payload_root / "annotator_a.csv", _ANNOTATION_COLUMNS, annotation_rows)
    _write_csv(payload_root / "annotator_b.csv", _ANNOTATION_COLUMNS, annotation_rows)
    _write_csv(output_root / "consensus_review.csv", _CONSENSUS_COLUMNS, consensus_rows)

    _verify_input_hashes(hashes, package_root)
    status = {
        "schema_version": ANNOTATION_PACK_SCHEMA_VERSION,
        "status": "awaiting_human_annotations",
        "human_consensus": False,
        "candidate_count": 8,
        "input": {
            "package": package_root.as_posix(),
            "preflight_json": preflight_path,
            "poni": poni_path,
            "mask": mask_path,
            "read_only": True,
        },
        "display": {
            "scale_method": "one_global_1st_to_99.5th_percentile",
            "vmin": vmin,
            "vmax": vmax,
            "colormap": "magma",
            "overlay_policy": "none",
        },
        "input_hashes": [item.as_dict() for item in hashes],
        "blind_image_hashes": blind_image_hashes,
        "files": {
            "annotation_manifest": "annotation_manifest.csv",
            "annotation_protocol": "blind_payload/annotation_protocol.md",
            "annotator_a": "blind_payload/annotator_a.csv",
            "annotator_b": "blind_payload/annotator_b.csv",
            "consensus_review": "consensus_review.csv",
            **{
                f"{candidate.blind_id}_png": f"blind_payload/{candidate.blind_id}.png"
                for candidate in selected
            },
        },
    }
    immutable_names = [
        "annotation_manifest.csv",
        "blind_payload/annotation_protocol.md",
        *(f"blind_payload/{candidate.blind_id}.png" for candidate in selected),
    ]
    status["immutable_output_hashes"] = _output_hashes(output_root, immutable_names)
    (output_root / "annotation_status.json").write_text(
            json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
    output_names = [
        *immutable_names,
        "blind_payload/annotator_a.csv",
        "blind_payload/annotator_b.csv",
        "consensus_review.csv",
        "annotation_status.json",
    ]
    result_files = {name: output_root / name for name in output_names}
    return {
        "output_directory": output_root,
        "files": result_files,
        "candidate_count": 8,
        "status": status,
        "manifest_rows": manifest_rows,
    }


__all__ = [
    "ANNOTATION_COORDINATE_SYSTEM",
    "ANNOTATION_PACK_SCHEMA_VERSION",
    "AnnotationPackError",
    "build_annotation_pack",
]
