"""P3/T1 same-model synthetic benchmark matrix.

The T1 benchmark intentionally uses the public empirical intensity model from
``butterfly_saxs.intensity`` as its generator.  It is therefore useful for
checking deterministic data handling, mask propagation, serialization, and
parameter/ridge recovery inside the same model family.  It is *not* an
independent physical validation of lamellar structure or mechanism.

The module has no dependencies beyond the project's existing NumPy runtime
and the shared intensity model.  Generated arrays use detector-mask polarity:
``mask=True`` means that a pixel is excluded/invalid.  q coordinates are in
``nm^-1`` and are declared in every truth record.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

import numpy as np

from .intensity import double_ellipse_intensity, ellipse_polar_radius, parameter_values


T1_SCHEMA_VERSION = "t1.same_model.v1"
T1_BENCHMARK_ID = "P3.1/T1"
T1_Q_UNIT = "nm^-1"
DEFAULT_T1_SEED = 20260827
GENERATOR_VERSION = "t1-same-model-v2"
MODEL_SCOPE = "same_model_empirical_synthetic"

_MODEL_FLAGS = ("synthetic_data", "empirical_model_only", "nonunique_inverse_problem")
_ARTIFACT_NAMES = ("beamstop", "streak", "gap", "bad_points", "missing_sector")


def _source_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


GENERATOR_DEPENDENCY_HASHES = {
    "benchmark_t1.py": _source_hash(Path(__file__)),
    "intensity.py": _source_hash(Path(__file__).with_name("intensity.py")),
}
GENERATOR_HASH = hashlib.sha256(
    json.dumps(
        GENERATOR_DEPENDENCY_HASHES,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()


def _copy_value(value: Any) -> Any:
    """Copy a small case specification without sharing mutable mappings."""

    if isinstance(value, Mapping):
        return {str(key): _copy_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_copy_value(item) for item in value)
    if isinstance(value, list):
        return [_copy_value(item) for item in value]
    return value


def _json_safe(value: Any, *, path: str = "value") -> Any:
    """Return strict JSON-compatible values and reject non-finite numbers."""

    if isinstance(value, np.generic):
        return _json_safe(value.item(), path=path)
    if isinstance(value, np.ndarray):
        return [_json_safe(item, path=f"{path}[]") for item in value.tolist()]
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item, path=f"{path}.{key}") for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item, path=f"{path}[]") for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"truth value at {path} must be finite, got {value!r}")
        return value
    raise TypeError(f"truth value at {path} is not JSON-compatible: {type(value).__name__}")


def _strict_json_text(value: Mapping[str, Any]) -> str:
    """Serialize a mapping while rejecting NaN and Infinity."""

    return json.dumps(
        _json_safe(value),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"


def _validate_shape(shape: Sequence[int]) -> tuple[int, int]:
    if isinstance(shape, (str, bytes)):
        raise ValueError("shape must contain two positive integer dimensions")
    try:
        values = tuple(shape)
    except TypeError as exc:
        raise ValueError("shape must contain two positive integer dimensions") from exc
    if len(values) != 2 or any(isinstance(item, bool) for item in values):
        raise ValueError("shape must contain two positive integer dimensions")
    if any(not isinstance(item, (int, np.integer)) or int(item) < 8 for item in values):
        raise ValueError("shape dimensions must be integers >= 8")
    return int(values[0]), int(values[1])


def _finite_float(value: Any, name: str, *, positive: bool = False) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result) or (positive and result <= 0):
        qualifier = "finite and positive" if positive else "finite"
        raise ValueError(f"{name} must be {qualifier}")
    return result


def _parse_q_range(value: Any) -> tuple[tuple[float, float], tuple[float, float]]:
    """Parse a shared ``(min, max)`` or ``(qx_range, qy_range)`` value."""

    if value is None:
        value = (-1.25, 1.25)
    try:
        outer = tuple(value)
    except TypeError as exc:
        raise ValueError("q_range must be (min, max) or ((qx_min, qx_max), (qy_min, qy_max))") from exc
    if len(outer) != 2:
        raise ValueError("q_range must contain two limits")
    nested = all(not isinstance(item, (str, bytes)) and isinstance(item, Sequence) for item in outer)
    if nested:
        ranges: list[tuple[float, float]] = []
        for index, item in enumerate(outer):
            limits = tuple(item)
            if len(limits) != 2:
                raise ValueError(f"q_range[{index}] must contain two limits")
            lo = _finite_float(limits[0], f"q_range[{index}][0]")
            hi = _finite_float(limits[1], f"q_range[{index}][1]")
            if hi <= lo:
                raise ValueError("q_range upper limit must exceed lower limit")
            ranges.append((lo, hi))
        return ranges[0], ranges[1]
    lo = _finite_float(outer[0], "q_range[0]")
    hi = _finite_float(outer[1], "q_range[1]")
    if hi <= lo:
        raise ValueError("q_range upper limit must exceed lower limit")
    return (lo, hi), (lo, hi)


def _parse_offset(value: Any) -> tuple[float, float]:
    """Return detector-center offset as ``(dy, dx)`` pixels."""

    if value is None:
        return 0.0, 0.0
    try:
        values = tuple(value)
    except TypeError as exc:
        raise ValueError("center_offset must be (dy, dx) in detector pixels") from exc
    if len(values) != 2:
        raise ValueError("center_offset must be (dy, dx) in detector pixels")
    return (_finite_float(values[0], "center_offset[0]"), _finite_float(values[1], "center_offset[1]"))


@dataclass(frozen=True)
class T1CaseSpec(Mapping[str, Any]):
    """Serializable description of one T1 synthetic case.

    ``center_offset`` is in detector pixels and uses ``(dy, dx)`` order.  The
    model's geometric angles are radians unless an explicit ``*_deg`` key is
    used in ``parameters``.  ``artifacts`` names are represented in the output
    truth record and have detector-mask polarity (True means excluded).
    """

    name: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    shape: tuple[int, int] = (64, 64)
    q_range: Any = (-1.25, 1.25)
    center_offset: tuple[float, float] = (0.0, 0.0)
    noise_model: str = "none"
    noise_sigma: float = 0.0
    poisson_scale: float = 500.0
    artifacts: tuple[str, ...] = ()
    low_snr: bool = False
    overlap: bool = False
    non_elliptic: bool = False
    reference_axis_deg: float = 0.0
    seed_offset: int = 0
    description: str = ""

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise ValueError("case name cannot be empty")
        object.__setattr__(self, "name", str(self.name).strip())
        object.__setattr__(self, "parameters", dict(self.parameters))
        object.__setattr__(self, "shape", _validate_shape(self.shape))
        object.__setattr__(self, "center_offset", _parse_offset(self.center_offset))
        normalized_noise = str(self.noise_model).strip().lower()
        if normalized_noise == "gaussian_noise":
            normalized_noise = "gaussian"
        if normalized_noise not in {"none", "gaussian", "poisson"}:
            raise ValueError("noise_model must be 'none', 'gaussian', or 'poisson'")
        object.__setattr__(self, "noise_model", normalized_noise)
        sigma = _finite_float(self.noise_sigma, "noise_sigma")
        if sigma < 0:
            raise ValueError("noise_sigma must be non-negative")
        object.__setattr__(self, "noise_sigma", sigma)
        object.__setattr__(self, "poisson_scale", _finite_float(self.poisson_scale, "poisson_scale", positive=True))
        parsed_artifacts = tuple(str(item).strip().lower().replace("-", "_") for item in self.artifacts)
        aliases = {"bad_point": "bad_points", "bad_pixel": "bad_points", "bad_pixels": "bad_points", "sector": "missing_sector"}
        parsed_artifacts = tuple(aliases.get(item, item) for item in parsed_artifacts)
        unknown = sorted(set(parsed_artifacts).difference(_ARTIFACT_NAMES))
        if unknown:
            raise ValueError(f"unknown artifact(s): {', '.join(unknown)}")
        object.__setattr__(self, "artifacts", tuple(dict.fromkeys(parsed_artifacts)))
        object.__setattr__(self, "reference_axis_deg", _finite_float(self.reference_axis_deg, "reference_axis_deg"))
        try:
            seed_offset = int(self.seed_offset)
        except (TypeError, ValueError) as exc:
            raise ValueError("seed_offset must be an integer") from exc
        object.__setattr__(self, "seed_offset", seed_offset)
        object.__setattr__(self, "description", str(self.description))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "parameters": _copy_value(self.parameters),
            "shape": list(self.shape),
            "q_range": _copy_value(self.q_range),
            "center_offset": list(self.center_offset),
            "noise_model": self.noise_model,
            "noise_sigma": self.noise_sigma,
            "poisson_scale": self.poisson_scale,
            "artifacts": list(self.artifacts),
            "low_snr": self.low_snr,
            "overlap": self.overlap,
            "non_elliptic": self.non_elliptic,
            "reference_axis_deg": self.reference_axis_deg,
            "seed_offset": self.seed_offset,
            "description": self.description,
        }

    @property
    def case_id(self) -> str:
        """Alias matching the case identifier used by the T2 benchmark."""

        return self.name

    @property
    def seed(self) -> int:
        """The default effective seed for this case."""

        return int(DEFAULT_T1_SEED + self.seed_offset)

    @property
    def category(self) -> str:
        """Compact category label useful when tabulating the matrix."""

        if self.non_elliptic:
            return "non_elliptic"
        if self.low_snr:
            return "low_snr"
        if self.overlap:
            return "overlap"
        if self.artifacts:
            return "artifact_" + self.artifacts[0]
        if self.noise_model != "none":
            return self.noise_model
        if self.center_offset != (0.0, 0.0):
            return "center_offset"
        return "geometry"

    @property
    def model_parameters(self) -> Mapping[str, Any]:
        """Alias used by callers that distinguish case metadata from truth."""

        return self.parameters

    def __getitem__(self, key: str) -> Any:
        aliases = {"id": "name", "case_id": "name"}
        if key == "seed":
            return self.seed
        if key == "category":
            return self.category
        return self.to_dict()[aliases.get(key, key)]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())


@dataclass
class T1Sample(Mapping[str, Any]):
    """One generated T1 frame and its in-memory truth record."""

    case: T1CaseSpec
    intensity: np.ndarray
    qx: np.ndarray
    qy: np.ndarray
    q: np.ndarray
    mask: np.ndarray
    truth_intensity: np.ndarray
    noise: np.ndarray
    truth_signal: np.ndarray
    truth_background: np.ndarray
    truth_ridge_plus: np.ndarray
    truth_ridge_minus: np.ndarray
    mask_components: Mapping[str, np.ndarray]
    truth: dict[str, Any]
    poisson_counts: np.ndarray | None = None

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(int(item) for item in self.intensity.shape)

    @property
    def valid_mask(self) -> np.ndarray:
        return ~self.mask

    @property
    def metadata(self) -> dict[str, Any]:
        return self.truth

    def arrays(self) -> dict[str, np.ndarray]:
        arrays = {
            "intensity": self.intensity,
            "intensity_noisy": self.intensity,
            "intensity_clean": self.truth_intensity,
            "intensity_noiseless": self.truth_intensity,
            "qx": self.qx,
            "qy": self.qy,
            "q": self.q,
            "mask": self.mask,
            "valid_mask": ~self.mask,
            "truth_intensity": self.truth_intensity,
            "truth_model_intensity": self.truth_intensity,
            "truth_signal": self.truth_signal,
            "truth_background": self.truth_background,
            "noise": self.noise,
            "truth_ridge_plus": self.truth_ridge_plus,
            "truth_ridge_minus": self.truth_ridge_minus,
        }
        for name in _ARTIFACT_NAMES:
            arrays[f"mask_{name}"] = np.asarray(
                self.mask_components.get(name, np.zeros(self.shape, dtype=bool)), dtype=bool
            )
        parameter_names = tuple(sorted(str(key) for key in self.truth.get("parameters", {})))
        arrays["truth_parameter_names"] = np.asarray(parameter_names, dtype="U64")
        arrays["truth_parameter_values"] = np.asarray(
            [float(self.truth["parameters"][name]) for name in parameter_names], dtype=np.float64
        )
        if self.poisson_counts is not None:
            arrays["poisson_counts"] = self.poisson_counts
        return arrays

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"case": self.case, "truth": self.truth}
        result.update(self.arrays())
        return result

    def __getitem__(self, key: str) -> Any:
        if key == "case":
            return self.case
        if key in {"truth", "metadata"}:
            return self.truth
        if key in {"parameters", "truth_parameters"}:
            return self.truth["parameters"]
        if key == "case_id":
            return self.case.name
        if key == "category":
            return self.case.category
        if key == "seed":
            return self.truth["seed"]
        if key == "valid_mask":
            return self.valid_mask
        arrays = self.arrays()
        if key in arrays:
            return arrays[key]
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return iter(("case", "truth", "metadata", "parameters", "case_id", "category", "seed", *self.arrays().keys()))

    def __len__(self) -> int:
        return 7 + len(self.arrays())


def _parameters_for_case(raw_parameters: Mapping[str, Any]) -> dict[str, Any]:
    supplied = dict(raw_parameters)
    # ``width`` is intentionally a concise case-spec spelling for angular
    # width.  Explicit ``angular_width`` remains in model radians; values of
    # ``width`` above one turn are almost certainly degrees, but default cases
    # use the unambiguous ``angular_width_deg`` spelling.
    if "width" in supplied and "angular_width" not in supplied and "angular_width_deg" not in supplied:
        width = _finite_float(supplied.pop("width"), "width")
        supplied["angular_width"] = math.radians(width) if abs(width) > 2.0 * math.pi else width
    return parameter_values(supplied)


def _case_from_input(case: T1CaseSpec | Mapping[str, Any] | str) -> T1CaseSpec:
    if isinstance(case, T1CaseSpec):
        return case
    if isinstance(case, str):
        normalized = str(case).strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "noiseless": "noiseless_default",
            "clean": "noiseless_default",
            "reference": "noiseless_default",
            "gaussian": "gaussian_parameter_sweep",
            "gaussian_noise": "gaussian_parameter_sweep",
            "poisson": "poisson_counting",
            "poisson_noise": "poisson_counting",
            "center": "center_offset",
            "offset": "center_offset",
            "artifacts": "combined_detector_artifacts",
            "combined_artifacts": "combined_detector_artifacts",
            "beamstop": "beamstop",
            "streak": "streak",
            "gap": "gap",
            "bad_point": "bad_points",
            "bad_pixel": "bad_points",
            "bad_pixels": "bad_points",
            "missing_sector": "missing_sector",
            "sector": "missing_sector",
            "non_elliptic": "negative_non_elliptic",
            "nonelliptic": "negative_non_elliptic",
        }
        normalized = aliases.get(normalized, normalized)
        for candidate in DEFAULT_CASES:
            candidate_name = candidate.name.lower().replace("-", "_").replace(" ", "_")
            if candidate_name == normalized or candidate.category == normalized:
                return candidate
        raise KeyError(f"unknown T1 case: {case}")
    if not isinstance(case, Mapping):
        raise TypeError("case must be a T1CaseSpec, case name, or mapping")
    values = dict(case)
    name = values.pop("name", values.pop("case", values.pop("case_id", values.pop("id", None))))
    if name is None:
        raise ValueError("case mapping requires a name")
    raw_parameters = values.pop("parameters", values.pop("params", None))
    if raw_parameters is None:
        parameter_keys = {
            "a", "b", "axis_ratio", "theta", "theta_deg", "lobe_angle", "lobe_angle_deg",
            "angular_width", "angular_width_deg", "width", "radial_sigma", "radial_gamma",
            "radial_width", "eta", "amplitude", "amplitude_plus", "amplitude_minus",
            "background", "background_slope", "background_curvature", "background_amplitude",
            "background_width",
        }
        raw_parameters = {key: values.pop(key) for key in tuple(values) if key in parameter_keys}
    # A mapping named after a built-in case is treated as a light override,
    # matching the T2 benchmark's convenient case-selection contract.  A new
    # name remains a fully explicit custom case.
    normalized = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "noiseless": "noiseless_default",
        "gaussian": "gaussian_parameter_sweep",
        "poisson": "poisson_counting",
        "non_elliptic": "negative_non_elliptic",
        "nonelliptic": "negative_non_elliptic",
    }
    normalized = aliases.get(normalized, normalized)
    base = next(
        (
            candidate
            for candidate in DEFAULT_CASES
            if candidate.name.lower().replace("-", "_").replace(" ", "_") == normalized
        ),
        None,
    )
    if base is not None:
        merged = base.to_dict()
        merged.pop("name", None)
        merged.update(values)
        merged_parameters = dict(base.parameters)
        merged_parameters.update(dict(raw_parameters or {}))
        merged["parameters"] = merged_parameters
        return T1CaseSpec(name=base.name, **merged)
    values["parameters"] = dict(raw_parameters or {})
    return T1CaseSpec(name=str(name), **values)


def _make_q_grid(
    shape: tuple[int, int],
    q_range: Any,
    center_offset: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[tuple[float, float], tuple[float, float]], tuple[float, float]]:
    qx_limits, qy_limits = _parse_q_range(q_range)
    rows, cols = shape
    qx_axis = np.linspace(qx_limits[0], qx_limits[1], cols, dtype=float)
    qy_axis = np.linspace(qy_limits[0], qy_limits[1], rows, dtype=float)
    # The offset is a shift of the physical q=0 position relative to the
    # detector midpoint.  The range and q spacing stay explicit and stable.
    dy, dx = center_offset
    dq_y = (qy_limits[1] - qy_limits[0]) / max(rows - 1, 1)
    dq_x = (qx_limits[1] - qx_limits[0]) / max(cols - 1, 1)
    qy_axis = qy_axis - dy * dq_y
    qx_axis = qx_axis - dx * dq_x
    qx, qy = np.meshgrid(qx_axis, qy_axis)
    q = np.hypot(qx, qy)
    return qx, qy, q, (qx_limits, qy_limits), (dq_y, dq_x)


def _artifact_masks(
    spec: T1CaseSpec,
    qx: np.ndarray,
    qy: np.ndarray,
    q: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    shape = q.shape
    masks = {name: np.zeros(shape, dtype=bool) for name in _ARTIFACT_NAMES}
    q_span = max(float(np.nanmax(np.abs(np.concatenate((qx.ravel(), qy.ravel()))))), 1e-6)
    if "beamstop" in spec.artifacts:
        masks["beamstop"] = q <= max(0.10 * q_span, 0.10)
    if "streak" in spec.artifacts:
        # Equatorial detector streak, excluding the beam centre itself so the
        # two artifact types remain distinguishable in the component masks.
        masks["streak"] = (np.abs(qy) <= max(0.018 * q_span, 0.018)) & (np.abs(qx) >= 0.12 * q_span)
    if "gap" in spec.artifacts:
        _, cols = shape
        yy, xx = np.indices(shape)
        gap_center = 0.62 * (cols - 1)
        masks["gap"] = (np.abs(xx - gap_center) <= max(1.0, 0.018 * cols)) & (yy >= 0.15 * shape[0]) & (yy <= 0.86 * shape[0])
    if "bad_points" in spec.artifacts:
        count = min(13, max(5, shape[0] * shape[1] // 500))
        selected = rng.choice(shape[0] * shape[1], size=count, replace=False)
        masks["bad_points"].ravel()[selected] = True
    if "missing_sector" in spec.artifacts:
        angle_deg = np.degrees(np.arctan2(qy, qx))
        masks["missing_sector"] = (q > 0.18 * q_span) & (angle_deg >= 112.0) & (angle_deg <= 148.0)
    return masks


def _parameter_truth(values: Mapping[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for key, value in values.items():
        if isinstance(value, (bool, str)) or not np.isscalar(value):
            continue
        numeric = _finite_float(value, f"parameters.{key}")
        result[str(key)] = numeric
    result["amplitude"] = 0.5 * (result.get("amplitude_plus", 0.0) + result.get("amplitude_minus", 0.0))
    result["theta_deg"] = math.degrees(result["theta"])
    result["lobe_angle_deg"] = math.degrees(result["lobe_angle"])
    result["angular_width_deg"] = math.degrees(result["angular_width"])
    return result


def _mask_truth(masks: Mapping[str, np.ndarray], combined: np.ndarray) -> dict[str, Any]:
    return {
        "polarity": "true means excluded/invalid",
        "shape": list(combined.shape),
        "invalid_pixel_count": int(np.count_nonzero(combined)),
        "valid_pixel_count": int(combined.size - np.count_nonzero(combined)),
        "components": {
            name: {
                "invalid_pixel_count": int(np.count_nonzero(value)),
                "overlaps_combined": int(np.count_nonzero(value & combined)),
            }
            for name, value in masks.items()
        },
    }


def generate_case(
    case: T1CaseSpec | Mapping[str, Any] | str,
    *,
    seed: int | None = None,
    shape: Sequence[int] | None = None,
    q_range: Any = None,
    center_offset: Sequence[float] | None = None,
    noise_model: str | None = None,
    noise_sigma: float | None = None,
    poisson_scale: float | None = None,
) -> T1Sample:
    """Generate one deterministic T1 case in memory.

    Parameters supplied as keyword overrides affect only this generated sample;
    the case specification itself is never mutated.  The result contains the
    observed ``intensity``, q-map, detector-style ``mask``, and exact/noisy
    truth arrays needed by downstream benchmark tests.
    """

    spec = _case_from_input(case)
    selected_shape = _validate_shape(spec.shape if shape is None else shape)
    selected_q_range = spec.q_range if q_range is None else q_range
    selected_offset = spec.center_offset if center_offset is None else _parse_offset(center_offset)
    actual_seed = int(spec.seed_offset + (DEFAULT_T1_SEED if seed is None else int(seed)))
    rng = np.random.default_rng(actual_seed)
    selected_noise_model = spec.noise_model if noise_model is None else str(noise_model).strip().lower()
    if selected_noise_model == "gaussian_noise":
        selected_noise_model = "gaussian"
    if selected_noise_model not in {"none", "gaussian", "poisson"}:
        raise ValueError("noise_model must be 'none', 'gaussian', or 'poisson'")
    selected_noise_sigma = spec.noise_sigma if noise_sigma is None else _finite_float(noise_sigma, "noise_sigma")
    if selected_noise_sigma < 0:
        raise ValueError("noise_sigma must be non-negative")
    selected_poisson_scale = spec.poisson_scale if poisson_scale is None else _finite_float(
        poisson_scale, "poisson_scale", positive=True
    )

    values = _parameters_for_case(spec.parameters)
    qx, qy, q, q_limits, dq = _make_q_grid(selected_shape, selected_q_range, selected_offset)
    components = double_ellipse_intensity(
        qx,
        qy,
        values,
        return_components=True,
        reference_axis_deg=spec.reference_axis_deg,
    )
    model_intensity = np.asarray(components["intensity"], dtype=np.float64)
    truth_signal = np.asarray(components["signal"], dtype=np.float64)
    truth_background = np.asarray(components["background"], dtype=np.float64)

    # A negative case is still generated through the same public model, but is
    # deliberately a superposition of two parameter sets.  It tests whether a
    # single-ellipse quality gate can identify model mismatch; it is not a
    # claim about an independent physical generator.
    if spec.non_elliptic:
        secondary = dict(values)
        secondary["a"] = 0.82 * float(values["a"])
        secondary["b"] = min(float(values["a"]), 1.22 * float(values["b"]))
        secondary["theta"] = float(values["theta"]) + math.radians(27.0)
        secondary["lobe_angle"] = max(0.05, 0.65 * float(values["lobe_angle"]))
        secondary["amplitude_plus"] = 0.58 * float(values["amplitude_plus"])
        secondary["amplitude_minus"] = 0.42 * float(values["amplitude_minus"])
        secondary["background"] = 0.0
        secondary["background_slope"] = 0.0
        secondary["background_curvature"] = 0.0
        secondary["background_amplitude"] = 0.0
        extra = np.asarray(
            double_ellipse_intensity(
                qx,
                qy,
                secondary,
                reference_axis_deg=spec.reference_axis_deg,
            ),
            dtype=np.float64,
        )
        model_intensity = model_intensity + extra
        truth_signal = truth_signal + extra

    masks = _artifact_masks(spec, qx, qy, q, rng)
    mask = np.zeros(selected_shape, dtype=bool)
    for component_mask in masks.values():
        mask |= component_mask

    if selected_noise_model == "none":
        observed = model_intensity.copy()
        poisson_counts = None
    elif selected_noise_model == "gaussian":
        sigma = float(selected_noise_sigma)
        observed = model_intensity + rng.normal(0.0, sigma, size=selected_shape)
        poisson_counts = None
    else:
        poisson_counts = rng.poisson(np.clip(model_intensity, 0.0, None) * float(selected_poisson_scale)).astype(np.int64)
        observed = poisson_counts.astype(np.float64) / float(selected_poisson_scale)

    observed = np.asarray(observed, dtype=np.float64)
    noise = observed - model_intensity
    angle = np.arctan2(qy, qx)
    ridge_plus = np.asarray(
        ellipse_polar_radius(angle, values["a"], values["b"], values["theta"]), dtype=np.float64
    )
    ridge_minus = np.asarray(
        ellipse_polar_radius(angle, values["a"], values["b"], -float(values["theta"])), dtype=np.float64
    )

    if not all(np.all(np.isfinite(array)) for array in (observed, model_intensity, noise, qx, qy, q, ridge_plus, ridge_minus)):
        raise ValueError("T1 generator produced a non-finite array")

    center_px = (
        (selected_shape[0] - 1) / 2.0 + selected_offset[0],
        (selected_shape[1] - 1) / 2.0 + selected_offset[1],
    )
    parameters_truth = _parameter_truth(values)
    truth: dict[str, Any] = {
        "schema_version": T1_SCHEMA_VERSION,
        "benchmark": T1_BENCHMARK_ID,
        "case_name": spec.name,
        "generator": {
            "module": "butterfly_saxs.intensity",
            "function": "double_ellipse_intensity",
            "same_model": True,
            "model_family": "empirical_symmetric_double_ellipse",
            "version": GENERATOR_VERSION,
            "hash": GENERATOR_HASH,
            "dependency_sha256": dict(GENERATOR_DEPENDENCY_HASHES),
        },
        "same_model": True,
        "within_single_model": not spec.non_elliptic,
        "scientific_scope": (
            "same-model implementation and parameter/ridge recovery; not an independent physical validation"
            if not spec.non_elliptic
            else "intentional same-model superposition negative; validates model-mismatch gating, not structure"
        ),
        "flags": list(_MODEL_FLAGS),
        "seed": actual_seed,
        "shape": list(selected_shape),
        "q_unit": T1_Q_UNIT,
        "q_range": [list(q_limits[0]), list(q_limits[1])],
        "q_spacing": [dq[0], dq[1]],
        "center_offset_px_dy_dx": list(selected_offset),
        "center_px_yx": list(center_px),
        "noise_model": selected_noise_model,
        "noise_sigma": float(selected_noise_sigma),
        "poisson_scale": float(selected_poisson_scale),
        "parameters": parameters_truth,
        "truth_parameters": parameters_truth,
        "ridge_truth": {
            "q_unit": T1_Q_UNIT,
            "plus_array": "truth_ridge_plus",
            "minus_array": "truth_ridge_minus",
            "lobe_angles_deg": [
                parameters_truth["lobe_angle_deg"],
                -parameters_truth["lobe_angle_deg"],
                180.0 + parameters_truth["lobe_angle_deg"],
                180.0 - parameters_truth["lobe_angle_deg"],
            ],
        },
        "noise": {
            "model": selected_noise_model,
            "sigma": float(selected_noise_sigma),
            "poisson_scale": float(selected_poisson_scale),
            "counting_semantics": selected_noise_model == "poisson",
        },
        "artifacts": list(spec.artifacts),
        "mask": _mask_truth(masks, mask),
        "low_snr": bool(spec.low_snr),
        "overlap": bool(spec.overlap),
        "non_elliptic_negative": bool(spec.non_elliptic),
        "reference_axis_deg": float(spec.reference_axis_deg),
        "array_contract": {
            "intensity": "observed intensity, float64, finite",
            "truth_intensity": "noiseless generated intensity, float64, finite",
            "qx_qy_q": "reciprocal-space coordinates in nm^-1, float64, finite",
            "mask": "detector polarity: True means excluded/invalid",
            "truth_ridge_plus_minus": "analytic model radial trajectories in q units",
        },
    }

    return T1Sample(
        case=spec,
        intensity=observed,
        qx=qx,
        qy=qy,
        q=q,
        mask=mask,
        truth_intensity=model_intensity,
        noise=noise,
        truth_signal=truth_signal,
        truth_background=truth_background,
        truth_ridge_plus=ridge_plus,
        truth_ridge_minus=ridge_minus,
        mask_components=masks,
        truth=_json_safe(truth),
        poisson_counts=poisson_counts,
    )


def _case_names() -> tuple[str, ...]:
    return tuple(case.name for case in DEFAULT_CASES)


def _default_case_specs() -> tuple[T1CaseSpec, ...]:
    # Keep dimensions moderate so the complete evidence directory is quick to
    # generate while still exercising non-square and q-range handling.
    return (
        T1CaseSpec(
            "noiseless_default",
            parameters={"a": 0.86, "b": 0.54, "theta_deg": 12.0, "lobe_angle_deg": 31.0, "angular_width_deg": 8.0, "amplitude": 2.4, "background": 0.04},
            shape=(56, 56),
            q_range=(-1.25, 1.25),
            description="Exact finite reference frame.",
        ),
        T1CaseSpec(
            "gaussian_parameter_sweep",
            parameters={"a": 1.08, "b": 0.43, "theta_deg": -21.0, "lobe_angle_deg": 48.0, "angular_width_deg": 13.0, "radial_width": 0.035, "amplitude_plus": 3.6, "amplitude_minus": 1.7, "background": 0.16},
            shape=(64, 64),
            q_range=(-1.5, 1.5),
            noise_model="gaussian",
            noise_sigma=0.025,
            description="Gaussian noise plus changed geometry, widths, amplitudes, and background.",
        ),
        T1CaseSpec(
            "poisson_counting",
            parameters={"a": 0.72, "b": 0.61, "theta_deg": 7.0, "lobe_angle_deg": 24.0, "angular_width_deg": 6.0, "amplitude": 8.0, "background": 0.08},
            shape=(60, 72),
            q_range=(-1.0, 1.0),
            noise_model="poisson",
            poisson_scale=350.0,
            description="Poisson noise with an explicit expected-count scale.",
        ),
        T1CaseSpec(
            "center_offset",
            parameters={"a": 0.94, "b": 0.49, "theta_deg": 19.0, "lobe_angle_deg": 37.0, "angular_width_deg": 9.0, "amplitude": 2.8, "background": 0.03},
            shape=(64, 64),
            q_range=(-1.3, 1.3),
            center_offset=(2.35, -3.10),
            description="Physical q=0 shifted from the detector midpoint by (dy, dx) pixels.",
        ),
        T1CaseSpec(
            "q_range_narrow",
            parameters={"a": 0.58, "b": 0.39, "theta_deg": -8.0, "lobe_angle_deg": 62.0, "angular_width_deg": 5.0, "amplitude": 1.9, "background": 0.02},
            shape=(48, 48),
            q_range=(-0.75, 0.75),
            description="Narrow reciprocal-space extent.",
        ),
        T1CaseSpec(
            "shape_rectangular",
            parameters={"a": 1.12, "b": 0.66, "theta_deg": 27.0, "lobe_angle_deg": 40.0, "angular_width_deg": 11.0, "amplitude": 1.4, "background": 0.10},
            shape=(48, 80),
            q_range=((-1.6, 1.4), (-1.0, 1.2)),
            description="Non-square detector shape and asymmetric q limits.",
        ),
        T1CaseSpec(
            "beamstop",
            parameters={"a": 0.88, "b": 0.51, "theta_deg": 14.0, "lobe_angle_deg": 34.0, "angular_width_deg": 8.0, "amplitude": 2.2, "background": 0.05},
            artifacts=("beamstop",),
            description="Central circular beamstop exclusion.",
        ),
        T1CaseSpec(
            "streak",
            parameters={"a": 0.90, "b": 0.58, "theta_deg": -16.0, "lobe_angle_deg": 42.0, "angular_width_deg": 10.0, "amplitude": 2.1, "background": 0.05},
            artifacts=("streak",),
            description="Equatorial detector streak exclusion.",
        ),
        T1CaseSpec(
            "gap",
            parameters={"a": 0.78, "b": 0.50, "theta_deg": 10.0, "lobe_angle_deg": 53.0, "angular_width_deg": 7.0, "amplitude": 2.0, "background": 0.04},
            artifacts=("gap",),
            description="Vertical detector gap exclusion.",
        ),
        T1CaseSpec(
            "bad_points",
            parameters={"a": 0.82, "b": 0.47, "theta_deg": 3.0, "lobe_angle_deg": 29.0, "angular_width_deg": 8.0, "amplitude": 2.7, "background": 0.06},
            artifacts=("bad_points",),
            description="Deterministically selected isolated bad pixels.",
        ),
        T1CaseSpec(
            "missing_sector",
            parameters={"a": 1.00, "b": 0.56, "theta_deg": 23.0, "lobe_angle_deg": 45.0, "angular_width_deg": 8.0, "amplitude": 2.5, "background": 0.05},
            artifacts=("missing_sector",),
            description="A missing q-sector, retained as an explicit mask.",
        ),
        T1CaseSpec(
            "combined_detector_artifacts",
            parameters={"a": 0.96, "b": 0.48, "theta_deg": -25.0, "lobe_angle_deg": 57.0, "angular_width_deg": 12.0, "amplitude_plus": 2.9, "amplitude_minus": 1.8, "background": 0.12},
            shape=(72, 64),
            q_range=(-1.35, 1.35),
            center_offset=(-1.4, 2.1),
            artifacts=_ARTIFACT_NAMES,
            description="Combined beamstop, streak, gap, bad-point, and sector mask.",
        ),
        T1CaseSpec(
            "low_snr",
            parameters={"a": 0.86, "b": 0.54, "theta_deg": 11.0, "lobe_angle_deg": 31.0, "angular_width_deg": 9.0, "amplitude": 0.16, "background": 0.10},
            noise_model="gaussian",
            noise_sigma=0.22,
            low_snr=True,
            description="Low signal-to-noise case; precision claims should be conservative.",
        ),
        T1CaseSpec(
            "overlap",
            parameters={"a": 0.84, "b": 0.62, "theta_deg": 1.5, "lobe_angle_deg": 18.0, "angular_width_deg": 19.0, "amplitude": 2.6, "background": 0.06},
            overlap=True,
            description="Strongly overlapping mirrored branches within the empirical model.",
        ),
        T1CaseSpec(
            "negative_non_elliptic",
            parameters={"a": 0.90, "b": 0.52, "theta_deg": 15.0, "lobe_angle_deg": 35.0, "angular_width_deg": 8.0, "amplitude": 2.4, "background": 0.04},
            shape=(64, 64),
            non_elliptic=True,
            description="Intentional same-model superposition that is not one ellipse pair.",
        ),
    )


DEFAULT_CASES: tuple[T1CaseSpec, ...] = _default_case_specs()
DEFAULT_T1_CASES = DEFAULT_CASES
DEFAULT_CASE_NAMES = tuple(case.name for case in DEFAULT_CASES)
DEFAULT_SHAPE = (64, 64)


def default_cases() -> tuple[T1CaseSpec, ...]:
    """Return independent copies of the built-in T1 case specifications."""

    return tuple(
        T1CaseSpec(
            name=case.name,
            parameters=_copy_value(case.parameters),
            shape=case.shape,
            q_range=_copy_value(case.q_range),
            center_offset=case.center_offset,
            noise_model=case.noise_model,
            noise_sigma=case.noise_sigma,
            poisson_scale=case.poisson_scale,
            artifacts=case.artifacts,
            low_snr=case.low_snr,
            overlap=case.overlap,
            non_elliptic=case.non_elliptic,
            reference_axis_deg=case.reference_axis_deg,
            seed_offset=case.seed_offset,
            description=case.description,
        )
        for case in DEFAULT_CASES
    )


def generate_t1_matrix(
    cases: Sequence[T1CaseSpec | Mapping[str, Any] | str] | None = None,
    *,
    seed: int = DEFAULT_T1_SEED,
    shape: Sequence[int] | None = None,
    q_range: Any = None,
    noise_sigma: float | None = None,
) -> tuple[T1Sample, ...]:
    """Generate all selected T1 cases in deterministic input order."""

    selected = default_cases() if cases is None else tuple(cases)
    return tuple(
        generate_case(
            case,
            seed=int(seed) + index,
            shape=shape,
            q_range=q_range,
            noise_sigma=noise_sigma,
        )
        for index, case in enumerate(selected)
    )


def _safe_case_filename(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name).strip()).strip("._")
    if not cleaned:
        raise ValueError("case name does not produce a usable filename")
    return cleaned


def _write_case_files(sample: T1Sample, output_dir: Path, *, force: bool) -> dict[str, Path]:
    stem = _safe_case_filename(sample.case.name)
    npz_path = output_dir / f"{stem}.npz"
    truth_path = output_dir / f"{stem}.json"
    targets = (npz_path, truth_path)
    if not force:
        existing = [str(path) for path in targets if path.exists()]
        if existing:
            raise FileExistsError("refusing to overwrite existing T1 target(s): " + ", ".join(existing))

    case_truth = dict(sample.truth)
    case_truth["files"] = {"npz": npz_path.name, "truth_json": truth_path.name}
    case_truth = _json_safe(case_truth)
    arrays = sample.arrays()
    arrays.update(
        {
            "q_unit": np.asarray(T1_Q_UNIT),
            "generator_version": np.asarray(GENERATOR_VERSION),
            "generator_hash": np.asarray(GENERATOR_HASH),
        }
    )
    np.savez_compressed(npz_path, **arrays)
    truth_path.write_text(_strict_json_text(case_truth), encoding="utf-8")
    return {"npz": npz_path, "truth_json": truth_path}


def write_case_evidence(
    case: T1CaseSpec | Mapping[str, Any] | str,
    output_dir: str | Path,
    *,
    seed: int | None = None,
    force: bool = False,
) -> dict[str, Path]:
    """Generate and write one case's compressed NPZ and strict truth JSON."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    sample = generate_case(case, seed=seed)
    return _write_case_files(sample, directory, force=bool(force))


def write_evidence_directory(
    output_dir: str | Path,
    cases: Sequence[T1CaseSpec | Mapping[str, Any] | str] | None = None,
    *,
    shape: Sequence[int] | None = None,
    q_range: Any = None,
    seed: int | None = None,
    noise_model: str | None = None,
    noise_sigma: float | None = None,
    poisson_scale: float | None = None,
    force: bool = False,
) -> Path:
    """Write the complete T1 evidence directory and global truth manifest.

    Existing targets are rejected by default.  With ``force=True`` only the
    case NPZ/JSON files and the global ``truth_manifest.json`` selected by this
    call are overwritten; unrelated files in ``output_dir`` are untouched.
    """

    directory = Path(output_dir)
    if any(part.casefold() == "data_local" for part in directory.resolve(strict=False).parts):
        raise ValueError("T1 派生证据不得写入 data_local 原始数据目录")
    if directory.exists() and not directory.is_dir():
        raise NotADirectoryError(f"output path is not a directory: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    selected = default_cases() if cases is None else tuple(_case_from_input(case) for case in cases)
    names = [case.name for case in selected]
    if len(set(names)) != len(names):
        raise ValueError("T1 case names must be unique")
    samples = tuple(
        generate_case(
            case,
            shape=shape,
            q_range=q_range,
            seed=None if seed is None else int(seed) + index,
            noise_model=noise_model,
            noise_sigma=noise_sigma,
            poisson_scale=poisson_scale,
        )
        for index, case in enumerate(selected)
    )
    manifest_path = directory / "truth_manifest.json"
    all_targets = [manifest_path]
    for sample in samples:
        stem = _safe_case_filename(sample.case.name)
        all_targets.extend((directory / f"{stem}.npz", directory / f"{stem}.json"))
    if not force:
        existing = [str(path) for path in all_targets if path.exists()]
        if existing:
            raise FileExistsError("refusing to overwrite existing T1 target(s): " + ", ".join(existing))

    case_records: list[dict[str, Any]] = []
    for sample in samples:
        files = _write_case_files(sample, directory, force=bool(force))
        case_records.append(
            {
                "name": sample.case.name,
                "case_id": sample.case.name,
                "category": sample.case.category,
                "npz": files["npz"].name,
                "npz_file": files["npz"].name,
                "truth_json": files["truth_json"].name,
                "truth_file": files["truth_json"].name,
                "same_model": True,
                "within_single_model": not sample.case.non_elliptic,
                "scientific_scope": sample.truth["scientific_scope"],
                "noise_model": sample.truth["noise"]["model"],
                "noise_sigma": float(sample.truth["noise"]["sigma"]),
                "artifacts": list(sample.case.artifacts),
                "shape": list(sample.shape),
                "seed": int(sample.truth["seed"]),
                "q_unit": T1_Q_UNIT,
                "generator_version": GENERATOR_VERSION,
                "generator_hash": GENERATOR_HASH,
            }
        )
    manifest: dict[str, Any] = {
        "schema": "t1_truth_manifest_v1",
        "schema_version": T1_SCHEMA_VERSION,
        "benchmark": T1_BENCHMARK_ID,
        "title": "P3/T1 same-model synthetic matrix",
        "same_model": True,
        "model_scope": MODEL_SCOPE,
        "generator_version": GENERATOR_VERSION,
        "generator_hash": GENERATOR_HASH,
        "generator": {
            "module": "butterfly_saxs.intensity",
            "function": "double_ellipse_intensity",
            "model_family": "empirical_symmetric_double_ellipse",
            "dependency_sha256": dict(GENERATOR_DEPENDENCY_HASHES),
        },
        "scientific_scope": [
            "Validates deterministic implementation, parameter truth, ridge truth, q-map, mask, and evidence serialization.",
            "Does not validate independent physical realism, material mechanism, or generalization beyond this empirical model family.",
            "The negative_non_elliptic case is an intentional same-model superposition used to test model-mismatch gating.",
        ],
        "seed": None if seed is None else int(seed),
        "case_count": len(case_records),
        "cases": case_records,
        "array_contract": {
            "required_npz_keys": [
                "intensity",
                "qx",
                "qy",
                "q",
                "mask",
                "truth_intensity",
                "noise",
                "q_unit",
                "generator_version",
                "generator_hash",
            ],
            "q_unit": T1_Q_UNIT,
            "mask_polarity": "True means excluded/invalid",
            "shape_contract": "all image and q/mask arrays share (rows, cols)",
        },
        "strict_json": {"allow_nan": False, "finite_numeric_truth": True},
    }
    manifest_path.write_text(_strict_json_text(manifest), encoding="utf-8")
    return manifest_path


# Explicit aliases keep the public entry points discoverable for small scripts
# and for benchmark tests written before the final function naming settled.
default_t1_cases = default_cases
CaseSpec = T1CaseSpec
generate_t1_case = generate_case
generate_butterfly_t1_case = generate_case
generate_t1_cases = generate_t1_matrix
generate_cases = generate_t1_matrix
generate_default_cases = generate_t1_matrix
generate_matrix = generate_t1_matrix
write_t1_case = write_case_evidence
write_t1_evidence = write_evidence_directory
write_t1_evidence_directory = write_evidence_directory
write_benchmark_evidence = write_evidence_directory


__all__ = [
    "DEFAULT_CASES",
    "DEFAULT_CASE_NAMES",
    "DEFAULT_SHAPE",
    "DEFAULT_T1_CASES",
    "DEFAULT_T1_SEED",
    "GENERATOR_DEPENDENCY_HASHES",
    "GENERATOR_HASH",
    "GENERATOR_VERSION",
    "MODEL_SCOPE",
    "T1_BENCHMARK_ID",
    "T1CaseSpec",
    "T1Sample",
    "T1_SCHEMA_VERSION",
    "T1_Q_UNIT",
    "CaseSpec",
    "default_cases",
    "default_t1_cases",
    "generate_butterfly_t1_case",
    "generate_case",
    "generate_matrix",
    "generate_t1_case",
    "generate_t1_cases",
    "generate_t1_matrix",
    "generate_cases",
    "generate_default_cases",
    "write_case_evidence",
    "write_evidence_directory",
    "write_t1_case",
    "write_t1_evidence",
    "write_t1_evidence_directory",
    "write_benchmark_evidence",
]
