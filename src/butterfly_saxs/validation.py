"""Pure validation contracts shared by CLI, service, GUI, and preflight.

The central object in this module is :class:`AnalysisDomain`.  It makes the
pixel population used by measurements and refinement explicit instead of
letting each stage silently rebuild a slightly different mask.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from datetime import datetime
from typing import Any, Mapping, Sequence

import numpy as np


ANALYSIS_DOMAIN_SCHEMA_VERSION = "lamellarsaxs2d.analysis_domain.v1"
RESULT_SCHEMA_VERSION = "lamellarsaxs2d.result.v1"

RESULT_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "result_type",
        "run_id",
        "created_at",
        "tool",
        "status",
        "input",
        "selector",
        "geometry",
        "mask",
        "correction_state",
        "correction",
        "uncertainty_state",
        "uncertainty",
        "analysis_domain",
        "quality",
        "measurements",
        "fit",
        "interpretation",
        "outputs",
        "provenance",
        "extensions",
    }
)

_NM_INVERSE_UNITS = {
    "nm^-1",
    "nm-1",
    "1/nm",
    "nm⁻¹",
}
_ANGSTROM_INVERSE_UNITS = {
    "å^-1",
    "å-1",
    "å⁻¹",
    "a^-1",
    "a-1",
    "1/å",
    "1/a",
    "angstrom^-1",
    "angstrom-1",
    "1/angstrom",
}
_PIXEL_Q_UNITS = {"pixel-q", "pixel_q"}


class AnalysisDomainError(ValueError):
    """The requested pixel domain is malformed or contains no usable pixels."""


class ResultSchemaError(ValueError):
    """A public result does not satisfy ``lamellarsaxs2d.result.v1``."""


def normalise_q_arrays(
    qx: Any,
    qy: Any,
    q: Any,
    q_unit: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Return q arrays in the v1 canonical unit and record the conversion.

    Physical reciprocal-space arrays use ``nm^-1`` internally.  Explicit
    angstrom-inverse input is multiplied by 10 exactly once.  Pixel-q and
    unknown units keep their numeric values and cannot be interpreted as a
    physical spacing.
    """

    qx_array = np.asarray(qx, dtype=float)
    qy_array = np.asarray(qy, dtype=float)
    q_array = np.asarray(q, dtype=float)
    source = str(q_unit or "unknown").strip()
    token = source.casefold().replace(" ", "")
    if token in _ANGSTROM_INVERSE_UNITS:
        canonical = "nm^-1"
        factor: float | None = 10.0
    elif token in _NM_INVERSE_UNITS:
        canonical = "nm^-1"
        factor = 1.0
    elif token in _PIXEL_Q_UNITS:
        canonical = "pixel-q"
        factor = None
    else:
        canonical = "unknown"
        factor = None
    if factor is not None and factor != 1.0:
        qx_array = qx_array * factor
        qy_array = qy_array * factor
        q_array = q_array * factor
    return qx_array, qy_array, q_array, {
        "q_unit": canonical,
        "source_q_unit": source if source != canonical else None,
        "q_conversion_factor_to_nm_inv": factor,
    }


def validate_q_coordinates(
    qx: Any,
    qy: Any,
    q: Any,
    *,
    rtol: float = 1e-7,
    atol: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate the reciprocal-space identity ``q = hypot(qx, qy)``.

    A radial magnitude alone cannot determine a direction in a two-dimensional
    detector plane.  Callers must therefore supply all three arrays (or derive
    ``q`` from ``qx`` and ``qy`` before calling this function).
    """

    qx_array = np.asarray(qx, dtype=float)
    qy_array = np.asarray(qy, dtype=float)
    q_array = np.asarray(q, dtype=float)
    if qx_array.shape != qy_array.shape or qx_array.shape != q_array.shape:
        raise AnalysisDomainError(
            "qx, qy, and q must have identical shapes; "
            f"got {qx_array.shape}, {qy_array.shape}, and {q_array.shape}"
        )

    expected = np.hypot(qx_array, qy_array)
    expected_finite = np.isfinite(expected)
    supplied_finite = np.isfinite(q_array)
    if not np.array_equal(expected_finite, supplied_finite):
        raise AnalysisDomainError(
            "q finite-value domain is inconsistent with hypot(qx, qy)"
        )
    if np.any(expected_finite) and not np.allclose(
        q_array[expected_finite],
        expected[expected_finite],
        rtol=rtol,
        atol=atol,
    ):
        max_difference = float(
            np.max(np.abs(q_array[expected_finite] - expected[expected_finite]))
        )
        raise AnalysisDomainError(
            "q is inconsistent with hypot(qx, qy); "
            f"maximum absolute difference is {max_difference:.6g}"
        )
    return qx_array, qy_array, q_array


def _validate_finite_json(value: Any, path: str = "result") -> None:
    if isinstance(value, str) and value.strip().casefold() in {
        "nan",
        "inf",
        "+inf",
        "-inf",
        "infinity",
        "+infinity",
        "-infinity",
    }:
        raise ResultSchemaError(f"{path} contains a string-form non-finite number")
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            raise ResultSchemaError(f"{path} contains a non-finite number")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _validate_finite_json(item, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_finite_json(item, f"{path}[{index}]")


def validate_result_schema(result: Mapping[str, Any]) -> None:
    """Validate the strict public v1 envelope used by P0--P2 preflight.

    This deliberately checks the stable boundary rather than every nested
    scientific extension.  Stage-specific evidence belongs in
    ``extensions`` and is validated by its producer.
    """

    if not isinstance(result, Mapping):
        raise ResultSchemaError("result must be a mapping")
    keys = set(result)
    missing = (RESULT_TOP_LEVEL_FIELDS - {"extensions"}) - keys
    unknown = keys - RESULT_TOP_LEVEL_FIELDS
    if missing:
        raise ResultSchemaError(f"result is missing fields: {sorted(missing)!r}")
    if unknown:
        raise ResultSchemaError(f"result has unknown top-level fields: {sorted(unknown)!r}")
    if result.get("schema_version") != RESULT_SCHEMA_VERSION:
        raise ResultSchemaError(f"schema_version must be {RESULT_SCHEMA_VERSION!r}")
    if result.get("result_type") not in {"preflight", "inspect", "analysis", "batch_frame"}:
        raise ResultSchemaError("result_type is not a v1 enum value")
    if not isinstance(result.get("run_id"), str) or not result["run_id"].strip():
        raise ResultSchemaError("run_id must be a non-empty string")
    created_at = result.get("created_at")
    if not isinstance(created_at, str):
        raise ResultSchemaError("created_at must be an ISO-8601 string")
    try:
        timestamp = datetime.fromisoformat(created_at)
    except ValueError as exc:
        raise ResultSchemaError("created_at must be a valid ISO-8601 timestamp") from exc
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ResultSchemaError("created_at must include a timezone")
    tool = result.get("tool")
    if not isinstance(tool, Mapping) or not all(
        isinstance(tool.get(key), str) and tool.get(key) for key in ("name", "version")
    ):
        raise ResultSchemaError("tool must contain non-empty name/version strings")
    status = result.get("status")
    if not isinstance(status, Mapping):
        raise ResultSchemaError("status must be an object")
    if status.get("status_color") not in {"green", "yellow", "red"}:
        raise ResultSchemaError("status.status_color is invalid")
    if status.get("scientific_status") not in {"PASS", "WARN", "FAIL"}:
        raise ResultSchemaError("status.scientific_status is invalid")
    if status.get("solver_status") not in {"not_run", "success", "failed", "cancelled"}:
        raise ResultSchemaError("status.solver_status is invalid")
    if status.get("numerical_status") not in {"PASS", "WARN", "FAIL", "NOT_TESTED"}:
        raise ResultSchemaError("status.numerical_status is invalid")
    if status.get("exit_code") not in {0, 1, 2}:
        raise ResultSchemaError("status.exit_code must be 0, 1, or 2")
    for name in ("flags", "failure_reasons"):
        entries = status.get(name)
        if not isinstance(entries, list):
            raise ResultSchemaError(f"status.{name} must be a list")
        for index, entry in enumerate(entries):
            if not isinstance(entry, Mapping) or not all(
                isinstance(entry.get(key), str) and entry.get(key)
                for key in ("code", "severity", "message")
            ):
                raise ResultSchemaError(
                    f"status.{name}[{index}] must contain code/severity/message"
                )
            if entry.get("severity") not in {"green", "yellow", "red"}:
                raise ResultSchemaError(f"status.{name}[{index}].severity is invalid")
    expected_scientific = {
        "green": "PASS",
        "yellow": "WARN",
        "red": "FAIL",
    }[status["status_color"]]
    if status["scientific_status"] != expected_scientific:
        raise ResultSchemaError("status color/scientific status are inconsistent")
    if status["status_color"] == "green" and status["exit_code"] != 0:
        raise ResultSchemaError("green status must use exit_code 0")
    if status["status_color"] == "yellow" and status["exit_code"] != 1:
        raise ResultSchemaError("yellow status must use exit_code 1")
    if status["status_color"] == "red" and status["exit_code"] not in {1, 2}:
        raise ResultSchemaError("red status must use exit_code 1 or 2")
    if status["scientific_status"] == "PASS" and status["failure_reasons"]:
        raise ResultSchemaError("PASS status cannot contain failure_reasons")
    if status["scientific_status"] == "FAIL" and not status["failure_reasons"]:
        raise ResultSchemaError("FAIL status must contain failure_reasons")
    if result.get("measurements") is not None or result.get("fit") is not None:
        if result.get("result_type") == "preflight":
            raise ResultSchemaError("preflight measurements and fit must be null")
    for name in (
        "input",
        "selector",
        "geometry",
        "mask",
        "correction",
        "uncertainty",
        "analysis_domain",
        "quality",
        "interpretation",
        "outputs",
        "provenance",
    ):
        if not isinstance(result.get(name), Mapping):
            raise ResultSchemaError(f"{name} must be an object")
    if "extensions" in result and not isinstance(result.get("extensions"), Mapping):
        raise ResultSchemaError("extensions must be an object when present")

    def require_fields(name: str, fields: set[str]) -> Mapping[str, Any]:
        value = result[name]
        missing_fields = fields - set(value)
        if missing_fields:
            raise ResultSchemaError(f"{name} is missing fields: {sorted(missing_fields)!r}")
        return value

    input_object = require_fields(
        "input",
        {"source_kind", "images", "manifest_path", "intensity_unit", "read_only"},
    )
    if result["result_type"] == "preflight" and input_object.get("read_only") is not True:
        raise ResultSchemaError("preflight input.read_only must be true")
    if not isinstance(input_object.get("images"), list) or not input_object["images"]:
        raise ResultSchemaError("input.images must be a non-empty list")
    selector = require_fields("selector", {"image", "mask", "manifest"})
    for selector_name in ("image", "mask", "manifest"):
        if not isinstance(selector.get(selector_name), Mapping):
            raise ResultSchemaError(f"selector.{selector_name} must be an object")
    require_fields(
        "geometry",
        {
            "poni",
            "q_unit",
            "source_q_unit",
            "q_conversion_factor_to_nm_inv",
            "coordinate_system",
            "q_window",
            "axis_labels",
            "arrays_npz",
        },
    )
    require_fields(
        "mask",
        {
            "source",
            "shape",
            "valid_mask_polarity",
            "external_mask_polarity",
            "roi_exclusion_polarity",
        },
    )
    if result.get("correction_state") not in {
        "raw_counts",
        "external_recipe_declared",
        "partially_corrected",
        "fully_corrected_external",
        "unknown",
    }:
        raise ResultSchemaError("correction_state is not a v1 enum value")
    require_fields(
        "correction",
        {
            "source_files",
            "declared_steps",
            "not_applied_steps",
            "software_reapply_prohibited",
            "absolute_intensity_comparable",
        },
    )
    if result.get("uncertainty_state") not in {"none", "partial", "complete", "unknown"}:
        raise ResultSchemaError("uncertainty_state is not a v1 enum value")
    require_fields(
        "uncertainty",
        {
            "sources",
            "components",
            "units",
            "stderr_scope",
            "separate_from_selection_uncertainty",
        },
    )
    analysis_domain = require_fields(
        "analysis_domain",
        {
            "schema_version",
            "status",
            "image_shape",
            "q_window",
            "weight_kind",
            "counts",
            "arrays_npz",
        },
    )
    if analysis_domain.get("schema_version") != ANALYSIS_DOMAIN_SCHEMA_VERSION:
        raise ResultSchemaError("analysis_domain.schema_version is invalid")
    if analysis_domain.get("status") == "computed":
        arrays_npz = analysis_domain.get("arrays_npz")
        if not isinstance(arrays_npz, Mapping) or not isinstance(arrays_npz.get("keys"), Mapping):
            raise ResultSchemaError("computed analysis_domain must declare arrays_npz.keys")
        expected_masks = {
            "finite_mask",
            "detector_valid_mask",
            "external_valid_mask",
            "q_window_mask",
            "roi_exclusion_mask",
            "weight_valid_mask",
            "fit_valid_mask",
            "sampled_valid_mask",
        }
        if set(arrays_npz["keys"]) != expected_masks:
            raise ResultSchemaError("analysis_domain arrays_npz.keys must declare all 8 masks")
    quality = require_fields(
        "quality",
        {"status", "status_color", "thresholds_version", "checks", "flags", "metrics"},
    )
    if quality.get("status") != status["scientific_status"] or quality.get(
        "status_color"
    ) != status["status_color"]:
        raise ResultSchemaError("quality and status summaries are inconsistent")
    require_fields(
        "interpretation",
        {
            "model_scope",
            "interpretation_limit",
            "claims_allowed",
            "claims_forbidden",
            "flags",
        },
    )
    require_fields(
        "outputs",
        {"directory", "paths_relative", "files", "overwrite", "force", "overwritten_paths"},
    )
    require_fields(
        "provenance",
        {
            "command",
            "working_directory",
            "git_commit",
            "dependencies",
            "hashes",
            "input_unchanged",
            "privacy",
        },
    )
    _validate_finite_json(result)


def _shape(array: Any, expected: tuple[int, int], name: str, *, dtype: Any) -> np.ndarray:
    value = np.asarray(array, dtype=dtype)
    if value.shape != expected:
        raise AnalysisDomainError(
            f"{name} shape {value.shape!r} does not match image shape {expected!r}"
        )
    return value


def _q_window(q: np.ndarray, value: Any) -> tuple[float, float]:
    finite = q[np.isfinite(q)]
    if not finite.size:
        raise AnalysisDomainError("q map contains no finite values")
    if value is None:
        lo, hi = float(np.min(finite)), float(np.max(finite))
    elif isinstance(value, Mapping):
        lo = value.get("min", value.get("q_min", value.get("low")))
        hi = value.get("max", value.get("q_max", value.get("high")))
        if lo is None or hi is None:
            raise AnalysisDomainError("q_window mapping must provide min/max")
        lo, hi = float(lo), float(hi)
    else:
        try:
            lo, hi = value
        except (TypeError, ValueError) as exc:
            raise AnalysisDomainError("q_window must be a (min, max) pair") from exc
        lo, hi = float(lo), float(hi)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        raise AnalysisDomainError("q_window must contain finite max > min")
    return lo, hi


@dataclass(frozen=True)
class AnalysisDomain:
    """Auditable decomposition of the pixels admitted to analysis.

    All masks use positive polarity: ``True`` means that the pixel survives
    the named stage.  Counts are cumulative, so the final count can be checked
    against every preceding exclusion without guessing mask polarity.
    """

    image_shape: tuple[int, int]
    q_window: tuple[float, float]
    finite_mask: np.ndarray
    detector_valid_mask: np.ndarray
    external_valid_mask: np.ndarray
    q_window_mask: np.ndarray
    roi_exclusion_mask: np.ndarray
    weight_valid_mask: np.ndarray
    fit_valid_mask: np.ndarray
    sampled_valid_mask: np.ndarray
    weight_kind: str = "none"
    schema_version: str = ANALYSIS_DOMAIN_SCHEMA_VERSION

    @property
    def counts(self) -> dict[str, int]:
        image_count = int(np.prod(self.image_shape))
        finite_count = int(np.count_nonzero(self.finite_mask))
        detector_count = int(np.count_nonzero(self.detector_valid_mask))
        external_count = int(np.count_nonzero(self.external_valid_mask))
        q_count = int(np.count_nonzero(self.q_window_mask))
        roi_excluded = int(np.count_nonzero(self.q_window_mask & self.roi_exclusion_mask))
        after_roi = self.q_window_mask & ~self.roi_exclusion_mask
        weight_invalid = int(np.count_nonzero(after_roi & ~self.weight_valid_mask))
        fit_count = int(np.count_nonzero(self.fit_valid_mask))
        sampled_count = int(np.count_nonzero(self.sampled_valid_mask))
        return {
            "image_pixel_count": image_count,
            "finite_pixel_count": finite_count,
            "detector_valid_count": detector_count,
            "external_mask_excluded_count": detector_count - external_count,
            "external_valid_count": external_count,
            "q_window_pixel_count": q_count,
            "roi_excluded_count": roi_excluded,
            "weight_invalid_count": weight_invalid,
            "fit_pixel_count": fit_count,
            "sampled_pixel_count": sampled_count,
        }

    def to_summary(self) -> dict[str, Any]:
        counts = self.counts
        counts.update(
            {
                "schema_version": self.schema_version,
                "image_shape": list(self.image_shape),
                "q_window": [float(self.q_window[0]), float(self.q_window[1])],
                "weight_kind": self.weight_kind,
                "fit_fraction": (
                    counts["fit_pixel_count"] / counts["image_pixel_count"]
                    if counts["image_pixel_count"]
                    else 0.0
                ),
                "sampling_fraction": (
                    counts["sampled_pixel_count"] / counts["fit_pixel_count"]
                    if counts["fit_pixel_count"]
                    else 0.0
                ),
            }
        )
        return counts

    def with_sampled_indices(self, indices: Sequence[int] | np.ndarray | None) -> "AnalysisDomain":
        """Return the same domain with an explicit optimizer sample recorded."""

        if indices is None:
            sampled = np.array(self.fit_valid_mask, copy=True)
        else:
            raw = np.asarray(indices)
            if raw.dtype.kind not in "iu":
                if raw.size and not np.all(np.isfinite(raw.astype(float))):
                    raise AnalysisDomainError("sampled_indices must contain finite integers")
                converted = raw.astype(np.int64)
                if raw.size and not np.array_equal(raw.astype(float), converted.astype(float)):
                    raise AnalysisDomainError("sampled_indices must contain integers")
                raw = converted
            flat = np.asarray(raw, dtype=np.int64).ravel()
            if flat.size and (int(np.min(flat)) < 0 or int(np.max(flat)) >= self.fit_valid_mask.size):
                raise AnalysisDomainError("sampled_indices contains an out-of-range pixel index")
            sampled = np.zeros(self.fit_valid_mask.size, dtype=bool)
            sampled[np.unique(flat)] = True
            sampled = sampled.reshape(self.image_shape)
            if np.any(sampled & ~self.fit_valid_mask):
                raise AnalysisDomainError("sampled_indices contains pixels outside fit_valid_mask")
        return replace(self, sampled_valid_mask=sampled)


def build_analysis_domain(
    image: Any,
    qx: Any,
    qy: Any,
    *,
    q: Any | None = None,
    detector_valid: Any | None = None,
    external_mask: Any | None = None,
    roi_exclusion: Any | None = None,
    q_window: Any = None,
    sigma: Any | None = None,
    weights: Any | None = None,
    sampled_indices: Sequence[int] | np.ndarray | None = None,
) -> AnalysisDomain:
    """Build the single analysis-domain contract used by all fit stages.

    The final domain is exactly::

        finite(I, qx, qy, q)
        & detector_valid
        & ~external_mask
        & ~roi_exclusion
        & q_window
        & weight_valid

    Invalid weight values inside the otherwise usable domain fail fast.  They
    are not silently discarded because doing so would change a scientific fit
    population without an explicit decision.
    """

    values = np.asarray(image)
    if values.ndim != 2 or values.size == 0:
        raise AnalysisDomainError(f"image must be a non-empty 2-D array, got {values.shape!r}")
    if values.dtype.kind not in "biufc":
        raise AnalysisDomainError(f"image must contain numeric values, got dtype {values.dtype}")
    shape = tuple(values.shape)
    qx_array = _shape(qx, shape, "qx", dtype=float)
    qy_array = _shape(qy, shape, "qy", dtype=float)
    q_array = np.hypot(qx_array, qy_array) if q is None else _shape(q, shape, "q", dtype=float)
    lo, hi = _q_window(q_array, q_window)

    finite = (
        np.isfinite(values)
        & np.isfinite(qx_array)
        & np.isfinite(qy_array)
        & np.isfinite(q_array)
    )
    detector = (
        np.ones(shape, dtype=bool)
        if detector_valid is None
        else _shape(detector_valid, shape, "detector_valid", dtype=bool)
    )
    external = (
        np.zeros(shape, dtype=bool)
        if external_mask is None
        else _shape(external_mask, shape, "external_mask", dtype=bool)
    )
    roi = (
        np.zeros(shape, dtype=bool)
        if roi_exclusion is None
        else _shape(roi_exclusion, shape, "roi_exclusion", dtype=bool)
    )

    finite_stage = finite
    detector_stage = finite_stage & detector
    external_stage = detector_stage & ~external
    q_stage = external_stage & (q_array >= lo) & (q_array <= hi)
    before_weight = q_stage & ~roi

    if sigma is not None and weights is not None:
        raise AnalysisDomainError("provide either sigma or weights, not both")
    weight_kind = "none"
    weight_valid = np.ones(shape, dtype=bool)
    if sigma is not None:
        sigma_array = _shape(sigma, shape, "sigma", dtype=float)
        weight_valid = np.isfinite(sigma_array) & (sigma_array > 0)
        weight_kind = "sigma"
    elif weights is not None:
        weight_array = _shape(weights, shape, "weights", dtype=float)
        weight_valid = np.isfinite(weight_array) & (weight_array > 0)
        weight_kind = "weights"
    invalid_weight_count = int(np.count_nonzero(before_weight & ~weight_valid))
    if invalid_weight_count:
        raise AnalysisDomainError(
            f"{weight_kind} contains {invalid_weight_count} non-finite or non-positive "
            "values inside the analysis domain"
        )

    fit_valid = before_weight & weight_valid
    if not np.any(q_stage):
        raise AnalysisDomainError("q_window and detector masks leave no pixels")
    if not np.any(fit_valid):
        raise AnalysisDomainError("analysis domain contains no fit-valid pixels")
    domain = AnalysisDomain(
        image_shape=shape,
        q_window=(lo, hi),
        finite_mask=finite_stage,
        detector_valid_mask=detector_stage,
        external_valid_mask=external_stage,
        q_window_mask=q_stage,
        roi_exclusion_mask=roi,
        weight_valid_mask=weight_valid,
        fit_valid_mask=fit_valid,
        sampled_valid_mask=np.array(fit_valid, copy=True),
        weight_kind=weight_kind,
    )
    return domain.with_sampled_indices(sampled_indices)


__all__ = [
    "ANALYSIS_DOMAIN_SCHEMA_VERSION",
    "RESULT_SCHEMA_VERSION",
    "AnalysisDomain",
    "AnalysisDomainError",
    "ResultSchemaError",
    "build_analysis_domain",
    "normalise_q_arrays",
    "validate_q_coordinates",
    "validate_result_schema",
]
