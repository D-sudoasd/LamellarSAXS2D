"""Independent physical-style benchmark data for the T2 validation stage.

This module deliberately has a small boundary.  It builds finite stacks of
curved, oriented layers in real space and obtains a reciprocal-space image
with a two-dimensional FFT.  The generator is an external test source for
the empirical analysis pipeline; its structural parameters are therefore
kept separate from the projection truth used to assess apparent ridges.

Only :mod:`numpy` is needed here.  In particular, this module does not import
the application's empirical intensity or synthetic-data modules.

``True`` in ``mask`` means an excluded detector pixel.  All q coordinates are
in ``nm^-1`` because the real-space pixel size is expressed in nanometres.
The FFT is a simple finite-window construction, not a complete 3-D physical
forward model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


GENERATOR_VERSION = "t2-independent-physical-v3"
MODEL_SCOPE = "independent_physical_synthetic"
T2_Q_UNIT = "nm^-1"
DEFAULT_SHAPE = (256, 256)
DEFAULT_PIXEL_SIZE_NM = 1.0


def _module_hash() -> str:
    """Return a stable hash of the generator source used for provenance."""

    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


GENERATOR_HASH = _module_hash()


@dataclass(frozen=True)
class CaseSpec(Mapping[str, Any]):
    """Finite layer-stack settings for one benchmark case.

    The fields describe how the real-space density is made.  They are not
    empirical ellipse parameters and must not be used as direct fit targets.
    ``orientation_offsets_deg`` and ``orientation_weights`` describe a frozen
    discrete quadrature of the orientation distribution around
    ``orientation_deg``.
    """

    case_id: str
    category: str
    seed: int
    layer_spacing_nm: float
    layer_count: int
    layer_width_nm: float
    orientation_deg: float
    orientation_offsets_deg: tuple[float, ...]
    orientation_weights: tuple[float, ...]
    curvature_nm: float
    waviness_nm: float
    spacing_jitter_fraction: float
    asymmetry: float
    noise_sigma: float

    @property
    def name(self) -> str:
        """Human-readable alias used by callers selecting a case by name."""

        return self.case_id

    def __getitem__(self, key: str) -> Any:
        aliases = {"name": "case_id", "id": "case_id"}
        field_name = aliases.get(key, key)
        if field_name not in self.__dataclass_fields__:
            raise KeyError(key)
        return getattr(self, field_name)

    def __iter__(self):
        return iter(self.__dataclass_fields__)

    def __len__(self) -> int:
        return len(self.__dataclass_fields__)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly mapping of the case settings."""

        values = asdict(self)
        values["orientation_offsets_deg"] = list(self.orientation_offsets_deg)
        values["orientation_weights"] = list(self.orientation_weights)
        values["name"] = self.name
        return values


# Dense one-degree quadratures make the eyebrow and wing arcs spatially
# resolved on the default FFT grid, instead of reducing them to a handful of
# disconnected orientation spots.
_EYEBROW_OFFSETS = tuple(float(value) for value in range(-18, 19))
_EYEBROW_WEIGHTS = tuple(float(np.exp(-0.5 * (value / 8.0) ** 2)) for value in _EYEBROW_OFFSETS)
_BUTTERFLY_OFFSETS = tuple(
    float(value) for value in (*range(-36, -9), *range(10, 37))
)
_BUTTERFLY_WEIGHTS = tuple(
    float(np.exp(-0.5 * ((abs(value) - 23.0) / 7.0) ** 2))
    for value in _BUTTERFLY_OFFSETS
)


# The four defaults intentionally have visibly different reciprocal-space
# topology.  Seeds are part of the public benchmark definition.
DEFAULT_CASES: tuple[CaseSpec, ...] = (
    CaseSpec(
        case_id="2-point",
        category="2-point",
        seed=2101,
        layer_spacing_nm=12.0,
        layer_count=14,
        layer_width_nm=1.25,
        orientation_deg=0.0,
        orientation_offsets_deg=(0.0,),
        orientation_weights=(1.0,),
        curvature_nm=0.0,
        waviness_nm=0.0,
        spacing_jitter_fraction=0.0,
        asymmetry=0.0,
        noise_sigma=0.015,
    ),
    CaseSpec(
        case_id="eyebrow",
        category="eyebrow",
        seed=2102,
        layer_spacing_nm=12.0,
        layer_count=14,
        layer_width_nm=1.25,
        orientation_deg=0.0,
        orientation_offsets_deg=_EYEBROW_OFFSETS,
        orientation_weights=_EYEBROW_WEIGHTS,
        curvature_nm=1.4,
        waviness_nm=0.0,
        spacing_jitter_fraction=0.0,
        asymmetry=0.0,
        noise_sigma=0.02,
    ),
    CaseSpec(
        case_id="butterfly",
        category="butterfly",
        seed=2103,
        layer_spacing_nm=12.0,
        layer_count=14,
        layer_width_nm=1.25,
        orientation_deg=0.0,
        orientation_offsets_deg=_BUTTERFLY_OFFSETS,
        orientation_weights=_BUTTERFLY_WEIGHTS,
        curvature_nm=1.8,
        waviness_nm=0.4,
        spacing_jitter_fraction=0.0,
        asymmetry=0.06,
        noise_sigma=0.025,
    ),
    CaseSpec(
        case_id="non_elliptical",
        category="non_elliptical",
        seed=2104,
        layer_spacing_nm=11.5,
        layer_count=12,
        layer_width_nm=1.45,
        orientation_deg=8.0,
        orientation_offsets_deg=(-25.0, -9.0, 8.0, 25.0),
        orientation_weights=(0.52, 0.88, 1.0, 0.68),
        curvature_nm=4.6,
        waviness_nm=2.0,
        spacing_jitter_fraction=0.08,
        asymmetry=0.24,
        noise_sigma=0.03,
    ),
)

DEFAULT_CASE_NAMES = tuple(case.case_id for case in DEFAULT_CASES)


def default_cases() -> tuple[CaseSpec, ...]:
    """Return the immutable default case definitions."""

    return DEFAULT_CASES


def _normalise_name(value: str) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def _case_from_mapping(values: Mapping[str, Any]) -> CaseSpec:
    """Resolve a mapping, allowing a default case to be lightly overridden."""

    raw_name = values.get("case_id", values.get("name", values.get("category")))
    if raw_name is None:
        raise ValueError("case mapping must contain case_id, name, or category")
    base = _resolve_case(str(raw_name))
    merged = asdict(base)
    aliases = {"name": "case_id", "id": "case_id"}
    allowed = set(merged) | set(aliases)
    unknown = sorted(str(key) for key in values if key not in allowed)
    if unknown:
        raise ValueError(
            "unknown T2 case mapping key(s): " + ", ".join(unknown)
        )
    for key, value in values.items():
        field_name = aliases.get(key, key)
        merged[field_name] = value
    merged["orientation_offsets_deg"] = tuple(float(v) for v in merged["orientation_offsets_deg"])
    merged["orientation_weights"] = tuple(float(v) for v in merged["orientation_weights"])
    return CaseSpec(**merged)


def _resolve_case(case: str | CaseSpec | Mapping[str, Any]) -> CaseSpec:
    if isinstance(case, CaseSpec):
        return case
    if isinstance(case, Mapping):
        return _case_from_mapping(case)
    name = _normalise_name(str(case))
    aliases = {
        "twopoint": "2_point",
        "two_point": "2_point",
        "2point": "2_point",
        "nonelliptical": "non_elliptical",
    }
    name = aliases.get(name, name)
    for candidate in DEFAULT_CASES:
        if _normalise_name(candidate.case_id) == name or _normalise_name(candidate.category) == name:
            return candidate
    choices = ", ".join(DEFAULT_CASE_NAMES)
    raise ValueError(f"unknown T2 case {case!r}; choose one of: {choices}")


def _validate_shape(shape: Sequence[int]) -> tuple[int, int]:
    if len(shape) != 2:
        raise ValueError("shape must contain exactly two dimensions")
    rows, cols = (int(shape[0]), int(shape[1]))
    if rows < 8 or cols < 8:
        raise ValueError("shape dimensions must each be at least 8")
    return rows, cols


def _validate_spec(spec: CaseSpec) -> None:
    if spec.layer_spacing_nm <= 0 or not np.isfinite(spec.layer_spacing_nm):
        raise ValueError("layer_spacing_nm must be finite and positive")
    if int(spec.layer_count) < 2:
        raise ValueError("layer_count must be at least 2")
    if spec.layer_width_nm <= 0 or not np.isfinite(spec.layer_width_nm):
        raise ValueError("layer_width_nm must be finite and positive")
    if len(spec.orientation_offsets_deg) != len(spec.orientation_weights):
        raise ValueError("orientation offsets and weights must have the same length")
    if not spec.orientation_offsets_deg:
        raise ValueError("at least one orientation component is required")
    weights = np.asarray(spec.orientation_weights, dtype=float)
    if np.any(~np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("orientation weights must be finite and positive")
    numeric = (
        spec.orientation_deg,
        spec.curvature_nm,
        spec.waviness_nm,
        spec.spacing_jitter_fraction,
        spec.asymmetry,
        spec.noise_sigma,
    )
    if not np.all(np.isfinite(np.asarray(numeric, dtype=float))):
        raise ValueError("case numeric settings must be finite")
    if spec.spacing_jitter_fraction < 0 or spec.spacing_jitter_fraction >= 1:
        raise ValueError("spacing_jitter_fraction must be in [0, 1)")
    if spec.noise_sigma < 0:
        raise ValueError("noise_sigma must be non-negative")


def _build_density(
    spec: CaseSpec,
    shape: tuple[int, int],
    pixel_size_nm: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Build a finite collection of Gaussian layers and record its stack."""

    rows, cols = shape
    y_axis = (np.arange(rows, dtype=float) - (rows - 1) / 2.0) * pixel_size_nm
    x_axis = (np.arange(cols, dtype=float) - (cols - 1) / 2.0) * pixel_size_nm
    x, y = np.meshgrid(x_axis, y_axis)
    span = max(float(np.max(np.abs(x))), float(np.max(np.abs(y))), pixel_size_nm)
    density = np.zeros(shape, dtype=float)
    weights = np.asarray(spec.orientation_weights, dtype=float)
    weights = weights / np.max(weights)
    records: list[dict[str, Any]] = []

    for component_index, (offset_deg, component_weight) in enumerate(
        zip(spec.orientation_offsets_deg, weights)
    ):
        angle_deg = float(spec.orientation_deg) + float(offset_deg)
        angle = np.deg2rad(angle_deg)
        normal_x, normal_y = np.cos(angle), np.sin(angle)
        tangent_x, tangent_y = -normal_y, normal_x
        normal_coordinate = x * normal_x + y * normal_y
        tangent_coordinate = x * tangent_x + y * tangent_y
        envelope = np.exp(-0.5 * (tangent_coordinate / (0.70 * span)) ** 2)

        spacing_jitter = (
            rng.normal(0.0, float(spec.spacing_jitter_fraction), int(spec.layer_count) - 1)
            if spec.spacing_jitter_fraction
            else np.zeros(int(spec.layer_count) - 1, dtype=float)
        )
        layer_spacings = float(spec.layer_spacing_nm) * (1.0 + spacing_jitter)
        if np.any(layer_spacings <= 0):
            raise ValueError("spacing jitter produced a non-positive layer spacing")
        layer_positions = np.concatenate(([0.0], np.cumsum(layer_spacings)))
        layer_positions -= float(np.mean(layer_positions))

        sign = 0.0 if abs(float(offset_deg)) < 1e-12 else np.sign(float(offset_deg))
        component_amplitude = float(component_weight) * (1.0 + float(spec.asymmetry) * sign)
        phase = 0.37 * component_index
        centerline_shift = float(spec.curvature_nm) * (
            (tangent_coordinate / span) ** 2 - 0.25
        )
        centerline_shift += float(spec.waviness_nm) * np.sin(
            np.pi * tangent_coordinate / span + phase
        )
        for layer_position in layer_positions:
            distance = normal_coordinate - float(layer_position) - centerline_shift
            density += component_amplitude * np.exp(
                -0.5 * (distance / float(spec.layer_width_nm)) ** 2
            ) * envelope
        records.append(
            {
                "component": int(component_index),
                "orientation_deg": angle_deg,
                "weight": component_amplitude,
                "layer_positions_nm": [float(v) for v in layer_positions],
            }
        )

    # Keep the density scale comparable between case definitions while
    # preserving the relative amplitudes created by the stack.
    maximum = float(np.max(density))
    if maximum > 0:
        density = density / maximum
    return np.asarray(density, dtype=float), records


def _q_grid(shape: tuple[int, int], pixel_size_nm: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows, cols = shape
    qx_axis = np.fft.fftshift(2.0 * np.pi * np.fft.fftfreq(cols, d=pixel_size_nm))
    qy_axis = np.fft.fftshift(2.0 * np.pi * np.fft.fftfreq(rows, d=pixel_size_nm))
    qx, qy = np.meshgrid(qx_axis, qy_axis)
    return qx, qy, np.hypot(qx, qy)


def _projection_reference(
    spec: CaseSpec,
    q_resolution_nm_inv: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Create analytic Bragg-vector truth from generator structure only."""

    q0 = 2.0 * np.pi / float(spec.layer_spacing_nm)
    theta = np.deg2rad(float(spec.orientation_deg))
    offsets = np.asarray(spec.orientation_offsets_deg, dtype=float)
    points_by_branch: list[tuple[str, np.ndarray]] = []

    if spec.category == "2-point":
        angles = np.array([theta, theta], dtype=float)
        radii = np.array([q0, q0], dtype=float)
        positive = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
        points_by_branch = [("positive", positive[:1]), ("negative", -positive[:1])]
    elif spec.category == "eyebrow":
        arc_parameter = np.linspace(-1.0, 1.0, 25)
        spread = np.deg2rad(max(abs(float(offsets.min())), abs(float(offsets.max()))))
        radial = np.full_like(arc_parameter, q0)
        positive_angle = theta + spread * arc_parameter
        positive = np.column_stack((radial * np.cos(positive_angle), radial * np.sin(positive_angle)))
        points_by_branch = [("upper", positive), ("lower", -positive)]
    elif spec.category == "butterfly":
        arc_parameter = np.linspace(-1.0, 1.0, 25)
        radial = np.full_like(arc_parameter, q0)
        offset_groups = (offsets[offsets < 0], offsets[offsets > 0])
        for branch_index, group in enumerate(offset_groups):
            branch_angle = theta + np.deg2rad(
                np.linspace(float(group.min()), float(group.max()), len(arc_parameter))
            )
            branch = np.column_stack((radial * np.cos(branch_angle), radial * np.sin(branch_angle)))
            points_by_branch.append((f"wing_{branch_index + 1}_positive", branch))
            points_by_branch.append((f"wing_{branch_index + 1}_negative", -branch))
    else:
        arc_parameter = np.linspace(-1.0, 1.0, 31)
        radial = q0 * (1.0 + 0.28 * np.sin(1.5 * np.pi * arc_parameter))
        angle_curve = theta + np.deg2rad(18.0) * np.sin(np.pi * arc_parameter)
        positive = np.column_stack((radial * np.cos(angle_curve), radial * np.sin(angle_curve)))
        points_by_branch = [("irregular_positive", positive), ("irregular_negative", -positive)]

    reference = np.concatenate([points for _, points in points_by_branch], axis=0).astype(
        float, copy=False
    )
    branch_ids = np.concatenate(
        [np.full(len(points), index, dtype=np.int16) for index, (_, points) in enumerate(points_by_branch)]
    )
    ridges: list[dict[str, Any]] = []
    cursor = 0
    for index, (label, points) in enumerate(points_by_branch):
        reference_points = reference[cursor : cursor + len(points)]
        ridges.append(
            {
                "branch_id": int(index),
                "label": label,
                "points_q": [
                    [float(qx_value), float(qy_value)]
                    for qx_value, qy_value in reference_points
                ],
            }
        )
        cursor += len(points)
    projection_truth = {
        "truth_scope": "projection_only_for_empirical_pipeline_validation",
        "reference_method": "analytic_bragg_vectors_from_generator_structure",
        "independent_of_generated_fft_pixels": True,
        "q_grid_resolution_nm_inv": float(q_resolution_nm_inv),
        "quantitative_use": (
            "analytic_projection_reference"
            if spec.category != "non_elliptical"
            else "negative_classification_only"
        ),
        "category": spec.category,
        "q_unit": T2_Q_UNIT,
        "ridges": ridges,
        "branch_count": len(ridges),
        "ridge_points_q": [[float(qx), float(qy)] for qx, qy in reference],
    }
    return reference, branch_ids, projection_truth


def _structure_truth(
    spec: CaseSpec,
    pixel_size_nm: float,
    realized_components: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "truth_scope": "generator_only_for_physical_forward_validation",
        "layer_spacing_nm": float(spec.layer_spacing_nm),
        "layer_count": int(spec.layer_count),
        "layer_width_nm": float(spec.layer_width_nm),
        "nominal_orientation_deg": float(spec.orientation_deg),
        "orientation_distribution": {
            "kind": "discrete_components",
            "offsets_deg": [float(v) for v in spec.orientation_offsets_deg],
            "weights": [float(v) for v in spec.orientation_weights],
        },
        "curvature": {
            "quadratic_shift_nm": float(spec.curvature_nm),
            "sinusoidal_shift_nm": float(spec.waviness_nm),
        },
        "spacing_jitter_fraction": float(spec.spacing_jitter_fraction),
        "asymmetry": float(spec.asymmetry),
        "pixel_size_nm": float(pixel_size_nm),
        "realized_components": realized_components,
    }


def _json_text(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))


def generate_case(
    case: str | CaseSpec | Mapping[str, Any],
    *,
    shape: Sequence[int] = DEFAULT_SHAPE,
    seed: int | None = None,
    noise_sigma: float | None = None,
    pixel_size_nm: float = DEFAULT_PIXEL_SIZE_NM,
) -> dict[str, Any]:
    """Generate one deterministic independent physical-style T2 case.

    Parameters
    ----------
    case:
        A default case name, :class:`CaseSpec`, or mapping based on one of
        the defaults.
    shape:
        ``(rows, columns)`` for both the real-space density and FFT image.
    seed:
        Optional replacement seed.  If omitted, the case's frozen seed is
        used.
    noise_sigma:
        Additive Gaussian standard deviation after clean intensity
        normalisation.  The observed image is clipped at zero.
    pixel_size_nm:
        Real-space grid spacing; it determines the q-map units.
    """

    spec = _resolve_case(case)
    _validate_spec(spec)
    image_shape = _validate_shape(shape)
    pixel_size_nm = float(pixel_size_nm)
    if pixel_size_nm <= 0 or not np.isfinite(pixel_size_nm):
        raise ValueError("pixel_size_nm must be finite and positive")
    effective_seed = int(spec.seed if seed is None else seed)
    sigma = float(spec.noise_sigma if noise_sigma is None else noise_sigma)
    if sigma < 0 or not np.isfinite(sigma):
        raise ValueError("noise_sigma must be finite and non-negative")

    rng = np.random.default_rng(effective_seed)
    density, realized_components = _build_density(spec, image_shape, pixel_size_nm, rng)
    centered_density = density - float(np.mean(density))
    fourier = np.fft.fftshift(np.fft.fft2(centered_density))
    clean_power = np.abs(fourier) ** 2
    clean_max = float(np.max(clean_power))
    if clean_max <= 0 or not np.isfinite(clean_max):
        raise ValueError("real-space stack produced an unusable FFT")
    intensity_noiseless = np.asarray(clean_power / clean_max, dtype=float)
    noise = rng.normal(0.0, sigma, size=image_shape) if sigma else np.zeros(image_shape, dtype=float)
    intensity_noisy = np.clip(intensity_noiseless + noise, 0.0, None)
    qx, qy, q = _q_grid(image_shape, pixel_size_nm)
    q0 = 2.0 * np.pi / float(spec.layer_spacing_nm)
    mask = np.asarray(q <= max(0.15 * q0, np.finfo(float).eps), dtype=bool)
    q_resolution = max(
        abs(float(qx[0, 1] - qx[0, 0])),
        abs(float(qy[1, 0] - qy[0, 0])),
    )
    projection_reference, projection_branch_id, projection_truth = _projection_reference(
        spec, q_resolution
    )
    structure_truth = _structure_truth(spec, pixel_size_nm, realized_components)
    metadata = {
        "generator_version": GENERATOR_VERSION,
        "generator_hash": GENERATOR_HASH,
        "model_scope": MODEL_SCOPE,
        "q_unit": T2_Q_UNIT,
        "seed": effective_seed,
        "noise_model": "gaussian_additive_clipped_at_zero",
        "noise_sigma": sigma,
        "shape": [int(image_shape[0]), int(image_shape[1])],
    }
    return {
        "case_id": spec.case_id,
        "category": spec.category,
        "seed": effective_seed,
        "noise_sigma": sigma,
        "generator_version": GENERATOR_VERSION,
        "generator_hash": GENERATOR_HASH,
        "model_scope": MODEL_SCOPE,
        "q_unit": T2_Q_UNIT,
        "real_space_density": density,
        "intensity_noiseless": intensity_noiseless,
        "intensity_clean": intensity_noiseless.copy(),
        "intensity_noisy": intensity_noisy,
        "intensity": intensity_noisy.copy(),
        "noise": noise,
        "qx": qx,
        "qy": qy,
        "q": q,
        "mask": mask,
        "valid_mask": ~mask,
        "projection_reference": projection_reference,
        "projection_reference_q": projection_reference.copy(),
        "projection_branch_id": projection_branch_id,
        "projection_truth": projection_truth,
        "structure_truth": structure_truth,
        "metadata": metadata,
    }


generate_t2_case = generate_case


def generate_cases(
    cases: Iterable[str | CaseSpec | Mapping[str, Any]] | None = None,
    *,
    shape: Sequence[int] = DEFAULT_SHAPE,
    seed: int | None = None,
    noise_sigma: float | None = None,
    pixel_size_nm: float = DEFAULT_PIXEL_SIZE_NM,
) -> tuple[dict[str, Any], ...]:
    """Generate a sequence of cases, deriving overridden seeds by index."""

    definitions = tuple(DEFAULT_CASES if cases is None else cases)
    generated: list[dict[str, Any]] = []
    for index, case in enumerate(definitions):
        case_seed = None if seed is None else int(seed) + index
        generated.append(
            generate_case(
                case,
                shape=shape,
                seed=case_seed,
                noise_sigma=noise_sigma,
                pixel_size_nm=pixel_size_nm,
            )
        )
    return tuple(generated)


generate_default_cases = generate_cases


def _case_npz_payload(result: Mapping[str, Any]) -> dict[str, Any]:
    """Select arrays and finite scalar metadata for one NPZ evidence file."""

    projection_truth = result["projection_truth"]
    structure_truth = result["structure_truth"]
    return {
        "real_space_density": result["real_space_density"],
        "intensity_noiseless": result["intensity_noiseless"],
        "intensity_clean": result["intensity_clean"],
        "intensity_noisy": result["intensity_noisy"],
        "intensity": result["intensity"],
        "noise": result["noise"],
        "qx": result["qx"],
        "qy": result["qy"],
        "q": result["q"],
        "q_unit": np.asarray(T2_Q_UNIT),
        "mask": result["mask"],
        "valid_mask": result["valid_mask"],
        "projection_reference": result["projection_reference"],
        "projection_reference_q": result["projection_reference_q"],
        "projection_branch_id": result["projection_branch_id"],
        "projection_truth_json": np.asarray(_json_text(projection_truth)),
        "structure_truth_json": np.asarray(_json_text(structure_truth)),
        "case_id": np.asarray(str(result["case_id"])),
        "category": np.asarray(str(result["category"])),
        "seed": np.asarray(int(result["seed"]), dtype=np.int64),
        "noise_sigma": np.asarray(float(result["noise_sigma"]), dtype=np.float64),
        "generator_version": np.asarray(GENERATOR_VERSION),
        "generator_hash": np.asarray(GENERATOR_HASH),
        "model_scope": np.asarray(MODEL_SCOPE),
    }


def _manifest_record(result: Mapping[str, Any], filename: str) -> dict[str, Any]:
    """Build the finite, compact truth entry for one case."""

    return {
        "case_id": str(result["case_id"]),
        "category": str(result["category"]),
        "seed": int(result["seed"]),
        "noise_sigma": float(result["noise_sigma"]),
        "shape": [int(v) for v in np.asarray(result["real_space_density"]).shape],
        "q_unit": T2_Q_UNIT,
        "npz_file": filename,
        "projection_truth": result["projection_truth"],
        "structure_truth": result["structure_truth"],
    }


def write_evidence_directory(
    output_dir: str | Path,
    cases: Iterable[str | CaseSpec | Mapping[str, Any]] | None = None,
    *,
    shape: Sequence[int] = DEFAULT_SHAPE,
    seed: int | None = None,
    noise_sigma: float | None = None,
    pixel_size_nm: float = DEFAULT_PIXEL_SIZE_NM,
    force: bool = False,
) -> Path:
    """Write NPZ cases and a compact ``truth_manifest.json``.

    Existing target files cause ``FileExistsError`` unless ``force=True``.
    With force enabled, only the target NPZ files and the manifest are
    replaced; unrelated files in ``output_dir`` are never removed.
    """

    destination = Path(output_dir)
    if any(part.casefold() == "data_local" for part in destination.resolve(strict=False).parts):
        raise ValueError("T2 派生证据不得写入 data_local 原始数据目录")
    if destination.exists() and not destination.is_dir():
        raise NotADirectoryError(f"output path is not a directory: {destination}")
    definitions = tuple(DEFAULT_CASES if cases is None else cases)
    resolved = tuple(_resolve_case(case) for case in definitions)
    target_names = [f"{_safe_filename(case.case_id)}.npz" for case in resolved]
    target_names.append("truth_manifest.json")
    if not force:
        existing = [destination / name for name in target_names if (destination / name).exists()]
        if existing:
            names = ", ".join(path.name for path in existing)
            raise FileExistsError(f"refusing to overwrite existing T2 evidence target(s): {names}")
    destination.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for index, spec in enumerate(resolved):
        case_seed = None if seed is None else int(seed) + index
        result = generate_case(
            spec,
            shape=shape,
            seed=case_seed,
            noise_sigma=noise_sigma,
            pixel_size_nm=pixel_size_nm,
        )
        filename = f"{_safe_filename(spec.case_id)}.npz"
        np.savez_compressed(destination / filename, **_case_npz_payload(result))
        record = _manifest_record(result, filename)
        record["npz_sha256"] = _sha256_file(destination / filename)
        records.append(record)

    manifest = {
        "schema": "t2_truth_manifest_v1",
        "generator_version": GENERATOR_VERSION,
        "generator_hash": GENERATOR_HASH,
        "model_scope": MODEL_SCOPE,
        "array_contract": {
            "q_unit": T2_Q_UNIT,
            "q_definition": "q = hypot(qx, qy); axes from 2*pi*fftfreq(pixel_size_nm)",
            "mask_polarity": "True means excluded/invalid",
        },
        "seed": None if seed is None else int(seed),
        "noise": {
            "model": "gaussian_additive_clipped_at_zero",
            "override_sigma": None if noise_sigma is None else float(noise_sigma),
        },
        "cases": records,
    }
    manifest_path = destination / "truth_manifest.json"
    manifest_path.write_text(_json_text(manifest) + "\n", encoding="utf-8")
    return manifest_path


def _safe_filename(value: str) -> str:
    """Use stable case names while keeping the allowed defaults readable."""

    cleaned = "".join(char if char.isalnum() or char in "-_" else "_" for char in str(value))
    return cleaned.strip("._") or "case"


write_t2_evidence = write_evidence_directory
write_benchmark_evidence = write_evidence_directory


__all__ = [
    "GENERATOR_HASH",
    "GENERATOR_VERSION",
    "MODEL_SCOPE",
    "T2_Q_UNIT",
    "DEFAULT_SHAPE",
    "CaseSpec",
    "DEFAULT_CASES",
    "DEFAULT_CASE_NAMES",
    "default_cases",
    "generate_case",
    "generate_t2_case",
    "generate_cases",
    "generate_default_cases",
    "write_evidence_directory",
    "write_t2_evidence",
    "write_benchmark_evidence",
]
