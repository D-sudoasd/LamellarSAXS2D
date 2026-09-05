"""Streaming frame, mask, geometry, and analysis-domain resolution.

This module owns the detector-sized part of preflight.  It deliberately keeps
only the reference arrays needed for the final mask export and turns all
other frames into small summaries, so the public facade does not also become
the image reader.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass
import math
import os
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .batch import FrameRef
from .preflight_context import PreflightContext


@dataclass(frozen=True)
class PreflightReadState:
    """All read/domain values consumed by checks and report construction."""

    image_metadata: list[Mapping[str, Any]]
    frames: list[dict[str, Any]]
    reference_array: np.ndarray
    reference_loaded: Any
    reference_source: Path
    first_shape: tuple[int, ...]
    mask_array: np.ndarray | None
    mask_source: Path | None
    external_mask: np.ndarray | None
    valid_mask: np.ndarray | None
    qmap: dict[str, Any]
    domain_q_window: Any
    reference_domain: Any
    domain_records: list[dict[str, Any]]


def read_preflight_inputs(
    context: PreflightContext,
    *,
    frame: int | None,
    dataset: str | None,
    mask_frame: int | None,
    mask_dataset: str | None,
    hash_groups: MutableMapping[str, list[dict[str, Any]]],
    read_one_frame: Callable[..., tuple[np.ndarray, Any, Path]],
    frame_record: Callable[..., dict[str, Any]],
    load_mask: Callable[..., tuple[np.ndarray | None, Path | None]],
    build_qmap: Callable[..., dict[str, Any]],
    build_analysis_domain: Callable[..., Any],
    preflight_error: type[Exception],
    analysis_domain_error: type[Exception],
) -> PreflightReadState:
    """Read all selected frames and build one shared valid pixel domain.

    The callbacks keep the historical loader seams in ``preflight.py``.  In
    particular, tests and clients can still monkeypatch its ``load_image``
    symbol while this module remains independent of the facade.
    """

    package_root = context.package_root
    refs = context.refs
    external_roots: Sequence[Path] = context.external_roots
    image_metadata: list[Mapping[str, Any]] = []
    frames: list[dict[str, Any]] = []
    reference_array, reference_loaded, reference_source = read_one_frame(
        refs[0],
        package_root,
        frame,
        dataset,
        hash_groups["inputs"],
        external_roots,
    )
    first_shape = tuple(int(item) for item in reference_array.shape)
    frames.append(
        frame_record(
            0,
            refs[0],
            reference_source,
            reference_loaded,
            reference_array,
            package_root,
            image_metadata,
        )
    )

    mask_array, mask_source = load_mask(
        context.selected_mask,
        package_root,
        first_shape,
        mask_frame,
        mask_dataset,
        hash_groups["mask"],
        external_roots,
    )
    external_mask: np.ndarray | None = None
    valid_mask: np.ndarray | None = None
    if mask_array is not None:
        if context.selected_convention == "0_valid_1_invalid":
            external_mask = mask_array
        else:
            valid_mask = mask_array
            external_mask = ~valid_mask

    qmap = build_qmap(
        first_shape,
        context.selected_poni,
        package_root,
        hash_groups["poni"],
        external_roots,
    )
    if qmap["qx"].shape != first_shape or qmap["qy"].shape != first_shape:
        raise preflight_error("qmap shape does not match input image shape")

    domain_q_window = context.selected_q_window
    if domain_q_window is None:
        finite_q = np.asarray(qmap["q"])[np.isfinite(qmap["q"])]
        if finite_q.size and float(np.min(finite_q)) == float(np.max(finite_q)):
            # Preserve the original implicit full-range accommodation for
            # tiny detectors.  An explicitly degenerate window is still
            # validated by build_analysis_domain.
            value = float(finite_q[0])
            delta = max(1.0, abs(value)) * 1e-12
            domain_q_window = (value - delta, value + delta)

    def build_domain(index: int, array: np.ndarray) -> Any:
        try:
            return build_analysis_domain(
                array,
                qmap["qx"],
                qmap["qy"],
                q=qmap["q"],
                detector_valid=qmap["detector_valid"],
                external_mask=external_mask,
                q_window=domain_q_window,
            )
        except (analysis_domain_error, TypeError, ValueError) as exc:
            raise preflight_error(
                f"could not build analysis domain for frame {index}: {exc}"
            ) from exc

    reference_domain = build_domain(0, reference_array)
    domain_records: list[dict[str, Any]] = [
        {
            "index": 0,
            "frame_id": refs[0].id,
            "summary": reference_domain.to_summary(),
        }
    ]
    for index, ref in enumerate(refs[1:], start=1):
        array, loaded, source = read_one_frame(
            ref,
            package_root,
            frame,
            dataset,
            hash_groups["inputs"],
            external_roots,
        )
        if tuple(array.shape) != first_shape:
            raise preflight_error(
                f"all frames must share one shape; frame {index} has {array.shape!r}, expected {first_shape!r}"
            )
        frames.append(
            frame_record(
                index,
                ref,
                source,
                loaded,
                array,
                package_root,
                image_metadata,
            )
        )
        domain = build_domain(index, array)
        domain_records.append(
            {
                "index": index,
                "frame_id": ref.id,
                "summary": domain.to_summary(),
            }
        )
        del array, loaded

    return PreflightReadState(
        image_metadata=image_metadata,
        frames=frames,
        reference_array=reference_array,
        reference_loaded=reference_loaded,
        reference_source=reference_source,
        first_shape=first_shape,
        mask_array=mask_array,
        mask_source=mask_source,
        external_mask=external_mask,
        valid_mask=valid_mask,
        qmap=qmap,
        domain_q_window=domain_q_window,
        reference_domain=reference_domain,
        domain_records=domain_records,
    )


def image_summary(array: np.ndarray, *, error_type: type[Exception] = ValueError) -> dict[str, Any]:
    if array.dtype.kind not in "biufc":
        raise error_type(f"image dtype {array.dtype!s} is not numeric")
    finite = np.isfinite(array)
    finite_count = int(np.count_nonzero(finite))
    image_count = int(array.size)
    result: dict[str, Any] = {
        "shape": [int(item) for item in array.shape],
        "dtype": str(array.dtype),
        "pixel_count": image_count,
        "finite_count": finite_count,
        "finite_fraction": finite_count / image_count if image_count else 0.0,
        "negative_count": 0,
        "negative_fraction": 0.0,
        "robust_high_count": 0,
        "robust_high_fraction": 0.0,
        "robust_high": {
            "method": "median_plus_6_mad",
            "median": None,
            "mad": None,
            "threshold": None,
            "count": 0,
            "fraction": 0.0,
        },
    }
    if not finite_count:
        return result
    values = np.asarray(array[finite], dtype=float)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    threshold = median + 6.0 * 1.4826 * mad
    if not math.isfinite(threshold):
        threshold = float(np.max(values))
    negative_count = int(np.count_nonzero(values < 0))
    high_count = int(np.count_nonzero(values > threshold))
    result.update(
        {
            "negative_count": negative_count,
            "negative_fraction": negative_count / finite_count,
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "median": median,
            "p95": float(np.percentile(values, 95)),
            "p99": float(np.percentile(values, 99)),
            "robust_high_count": high_count,
            "robust_high_fraction": high_count / finite_count,
            "robust_high": {
                "method": "median_plus_6_mad",
                "median": median,
                "mad": mad,
                "threshold": threshold,
                "count": high_count,
                "fraction": high_count / finite_count,
            },
        }
    )
    return result

def read_one_frame(
    ref: FrameRef,
    package: Path,
    frame_override: int | None,
    dataset_override: str | None,
    records: list[dict[str, Any]],
    external_roots: Sequence[str | os.PathLike[str] | Path] | None = None,
    *,
    resolve_path: Callable[..., Path],
    read_file_record: Callable[..., Any],
    load_image: Callable[..., Any],
    error_type: type[Exception] = ValueError,
) -> tuple[np.ndarray, Any, Path]:
    """Read one frame and immediately release the loader object to the caller."""

    source = resolve_path(
        package,
        ref.path,
        external_roots=external_roots,
        label="manifest frame",
    )
    if not source.exists() or not source.is_file():
        raise error_type(f"manifest frame path does not exist: {source}")
    selected_frame = frame_override if frame_override is not None else ref.frame_selector
    selected_dataset = dataset_override if dataset_override is not None else (ref.dataset_id or None)
    try:
        loaded = read_file_record(
            source,
            package,
            lambda source=source, selected_frame=selected_frame, selected_dataset=selected_dataset: load_image(
                source,
                frame=selected_frame,
                dataset=selected_dataset,
            ),
            records,
        )
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        if isinstance(exc, error_type):
            raise
        raise error_type(f"could not read frame {source}: {exc}") from exc
    array = np.asarray(loaded.data)
    if array.ndim != 2 or array.size == 0:
        raise error_type(f"frame {source} must be a non-empty 2-D image; got {array.shape!r}")
    return array, loaded, source

def frame_record(
    index: int,
    ref: FrameRef,
    source: Path,
    loaded: Any,
    array: np.ndarray,
    package: Path,
    image_metadata: list[Mapping[str, Any]],
    *,
    image_summary: Callable[..., dict[str, Any]],
    display_path: Callable[[Path, Path], str],
) -> dict[str, Any]:
    summary = image_summary(array)
    metadata = loaded.metadata if isinstance(loaded.metadata, Mapping) else {}
    image_metadata.append(metadata)
    return {
        "index": index,
        "id": ref.id,
        "path": display_path(source, package),
        "frame": loaded.frame,
        "dataset": loaded.dataset,
        "manifest_frame": ref.to_dict(),
        "summary": summary,
        "shape": summary["shape"],
        "dtype": summary["dtype"],
        "finite_fraction": summary["finite_fraction"],
        "negative_fraction": summary["negative_fraction"],
        "robust_high": summary["robust_high"],
    }


__all__ = ["PreflightReadState", "frame_record", "image_summary", "read_one_frame", "read_preflight_inputs"]
