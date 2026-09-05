"""Empirical two-dimensional intensity model and deterministic refinement.

The model is deliberately a measurement model.  It describes the observed
four-lobe butterfly pattern as two origin-centred ellipses with shared axes,
tilts ``+theta`` and ``-theta``, angular envelopes at four Friedel-related
directions, a radial pseudo-Voigt/Gaussian line shape, and a smooth radial
background.  Its parameters should not be promoted to a unique microscopic
interpretation: every fit is marked ``empirical_model_only`` and
``nonunique_inverse_problem``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import dataclasses
import math
from typing import Any, Mapping, Sequence

import numpy as np

try:
    from scipy.optimize import least_squares
except Exception:  # pragma: no cover - only for incomplete installations
    least_squares = None

from .observables import APPARENT_FLAGS, _extract_maps, _get_field, _q_limits
from .parameters import ParameterSet, ParameterSpec
from .cancellation import raise_if_cancelled


MODEL_FLAGS = APPARENT_FLAGS + ("empirical_model_only",)


class _MappingResult:
    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def keys(self) -> tuple[str, ...]:
        return tuple(self.__dataclass_fields__)  # type: ignore[attr-defined]

    def as_dict(self) -> dict[str, Any]:
        return {key: getattr(self, key) for key in self.keys()}


@dataclass
class IntensityFitResult(_MappingResult):
    """Result of multi-scale least-squares refinement."""

    parameters: Any
    initial_parameters: Any
    success: bool
    message: str
    cost: float
    rmse: float
    nfev: int
    sampled_indices: np.ndarray
    prediction: np.ndarray
    residual: np.ndarray
    image_shape: tuple[int, ...]
    scale_history: tuple[dict[str, Any], ...]
    flags: tuple[str, ...] = MODEL_FLAGS
    weighting: str = "equal_robust_scaled_by_global_std"
    weighted_rmse: float = float("nan")
    stderr: dict[str, float] = field(default_factory=dict)
    covariance: np.ndarray | None = None
    covariance_names: tuple[str, ...] = ()
    condition_number: float = float("nan")
    bound_flags: dict[str, bool] = field(default_factory=dict)
    # Primary fit metrics are evaluated on every finite pixel that survived
    # the q-window/mask/uncertainty filters.  The sampled fields describe only
    # the pixel set used by the final optimizer call.
    ndata: int = 0
    sampled_n: int = 0
    sample_rmse: float = float("nan")
    reference_axis_deg: float = 0.0
    candidate_solutions: tuple[dict[str, Any], ...] = ()
    selected_start_index: int = 0
    multistart_count: int = 1
    sample_cost: float = float("nan")
    full_cost: float = float("nan")
    selection_objective: str = "full_valid_weighted_robust_cost"

    @property
    def values(self) -> Any:
        return self.parameters

    @property
    def model_image(self) -> np.ndarray:
        return np.asarray(self.prediction).reshape(self.image_shape)

    @property
    def residual_image(self) -> np.ndarray:
        return np.asarray(self.residual).reshape(self.image_shape)

    @property
    def used_pixels(self) -> int:
        return int(len(self.sampled_indices))

    @property
    def theta_deg(self) -> float:
        values = parameter_values(self.parameters)
        return float(np.degrees(float(values.get("theta", np.nan))))

    @property
    def phi_app_deg(self) -> float:
        # This is intentionally not derived from theta.  A caller may attach
        # an independently measured lobe/tilt value to a ParameterSet.
        value = _get_field(self.parameters, ("phi_app_deg", "phi_app"), np.nan)
        try:
            return float(value)
        except Exception:
            return float("nan")

    @property
    def alpha_candidate_deg(self) -> float:
        value = _get_field(self.parameters, ("alpha_candidate_deg", "alpha_candidate"), np.nan)
        try:
            return float(value)
        except Exception:
            return float("nan")


DEFAULT_PARAMETERS: dict[str, float] = {
    # q-space ellipse geometry
    "a": 1.0,
    "b": 0.7,
    "theta": 0.0,
    # four angular lobe centres are +/- lobe_angle and their opposites
    "lobe_angle": float(np.deg2rad(30.0)),
    "angular_width": float(np.deg2rad(8.0)),
    # radial pseudo-Voigt parameters
    "radial_sigma": 0.04,
    "radial_gamma": 0.04,
    "eta": 0.35,
    # signal and smooth radial background
    "amplitude_plus": 1.0,
    "amplitude_minus": 1.0,
    "background": 0.0,
    "background_slope": 0.0,
    "background_curvature": 0.0,
    "background_amplitude": 0.0,
    "background_width": 1.0,
}


def default_intensity_parameters(
    *,
    a: float = 1.0,
    axis_ratio: float = 0.7,
    theta_deg: float = 0.0,
) -> ParameterSet:
    """Canonical editable parameter set for full-pixel empirical refinement."""

    eps = float(np.finfo(float).eps)
    theta = float(np.deg2rad(theta_deg))
    return ParameterSet(
        {
            "a": ParameterSpec(float(a), min=eps),
            "axis_ratio": ParameterSpec(float(axis_ratio), min=eps, max=1.0),
            "b": ParameterSpec(float(a) * float(axis_ratio), vary=False, expr="a*axis_ratio"),
            "theta": ParameterSpec(theta, min=-np.pi / 2.0, max=np.pi / 2.0),
            "theta_deg": ParameterSpec(0.0, vary=False, expr="theta*180/pi"),
            "lobe_angle": ParameterSpec(float(np.deg2rad(30.0)), min=0.0, max=np.pi / 2.0),
            "lobe_angle_deg": ParameterSpec(0.0, vary=False, expr="lobe_angle*180/pi"),
            "angular_width": ParameterSpec(float(np.deg2rad(8.0)), min=eps, max=np.pi / 2.0),
            "radial_sigma": ParameterSpec(0.04, min=eps),
            "radial_gamma": ParameterSpec(0.04, min=eps),
            "eta": ParameterSpec(0.35, min=0.0, max=1.0),
            "amplitude_plus": ParameterSpec(1.0, min=0.0),
            "amplitude_minus": ParameterSpec(1.0, min=0.0),
            "background": ParameterSpec(0.0, min=0.0),
            "background_slope": ParameterSpec(0.0, min=0.0),
            "background_curvature": ParameterSpec(0.0, min=0.0),
            "background_amplitude": ParameterSpec(0.0, min=0.0),
            "background_width": ParameterSpec(1.0, min=eps),
        }
    )


_ALIASES: dict[str, str] = {
    "semi_major": "a",
    "major_axis": "a",
    "ellipse_a": "a",
    "axis_a": "a",
    "semi_minor": "b",
    "minor_axis": "b",
    "ellipse_b": "b",
    "axis_b": "b",
    "tilt": "theta",
    "ellipse_theta": "theta",
    "ellipse_angle": "theta",
    "theta_degrees": "theta_deg",
    "lobe_angle_degrees": "lobe_angle_deg",
    "phi0": "lobe_angle",
    "lobe_phi": "lobe_angle",
    "angular_sigma": "angular_width",
    "azimuthal_width": "angular_width",
    "azimuth_width": "angular_width",
    "radial_sigma_q": "radial_sigma",
    "radial_fwhm": "radial_fwhm",
    "line_width": "radial_width",
    "scale": "amplitude",
    "intensity_scale": "amplitude",
    "baseline": "background",
    "background0": "background",
    "bg": "background",
    "bg_slope": "background_slope",
    "bg_curvature": "background_curvature",
    "pv_fraction": "eta",
    "lorentzian_fraction": "eta",
}


def _normal_key(key: Any) -> str:
    return str(key).strip().lower().replace("-", "_").replace(" ", "_")


def canonical_parameter_name(key: Any) -> str:
    name = _normal_key(key)
    return _ALIASES.get(name, name)


def _parameter_mapping(parameters: Any) -> dict[str, Any]:
    """Extract values from a shared ParameterSet or ordinary mapping."""

    if parameters is None:
        return {}
    candidates: Any = parameters
    if not isinstance(candidates, Mapping):
        for name in ("values", "parameters", "params", "as_dict", "to_dict"):
            value = _get_field(parameters, (name,))
            if isinstance(value, Mapping):
                candidates = value
                break
    if isinstance(candidates, Mapping):
        if isinstance(candidates, ParameterSet):
            # ParameterSet.items() yields resolved numeric values, but its
            # specs remain the source of truth for unit adapters.  An
            # independent degree field alongside a radian field is
            # ambiguous; only a tied degree adapter (as in the default set)
            # may coexist with its corresponding radian parameter.
            for radians_name, degree_names in (
                ("theta", {"theta_deg", "theta_degrees"}),
                ("lobe_angle", {"lobe_angle_deg", "lobe_angle_degrees"}),
                ("angular_width", {"angular_width_deg", "angular_width_degrees"}),
            ):
                if radians_name not in candidates.names:
                    continue
                for degree_name in degree_names.intersection(candidates.names):
                    if not candidates[degree_name].is_tied:
                        raise ValueError(
                            f"supply either {radians_name} (radians) or {degree_name} (degrees), not both"
                        )
        else:
            normalized = {_normal_key(key) for key in candidates}
            for radians_name, degree_names in (
                ("theta", {"theta_deg", "theta_degrees"}),
                ("lobe_angle", {"lobe_angle_deg", "lobe_angle_degrees"}),
                ("angular_width", {"angular_width_deg", "angular_width_degrees"}),
            ):
                if radians_name in normalized and normalized.intersection(degree_names):
                    degree_name = sorted(normalized.intersection(degree_names))[0]
                    raise ValueError(
                        f"supply either {radians_name} (radians) or {degree_name} (degrees), not both"
                    )
        out: dict[str, Any] = {}
        for key, value in candidates.items():
            raw_name = _normal_key(key)
            name = canonical_parameter_name(key)
            if np.isscalar(value) or isinstance(value, str):
                # Public/UI degree fields are accepted, but the core model
                # always stores geometric angles in radians.  ``phi_app`` and
                # ``alpha_candidate`` are carried as metadata and never
                # silently collapsed into the ellipse ``theta`` parameter.
                if raw_name in {"theta_deg", "theta_degrees"}:
                    out["theta"] = np.deg2rad(float(value))
                elif raw_name in {"lobe_angle_deg", "lobe_angle_degrees"}:
                    out["lobe_angle"] = np.deg2rad(float(value))
                elif raw_name in {"angular_width_deg", "angular_width_degrees"}:
                    out["angular_width"] = np.deg2rad(float(value))
                elif raw_name in {"phi_app_deg", "phi_app"}:
                    out["phi_app_deg"] = value
                elif raw_name in {"alpha_candidate_deg", "alpha_candidate"}:
                    out["alpha_candidate_deg"] = value
                else:
                    out[name] = value
        return out
    out = {}
    for key in DEFAULT_PARAMETERS:
        if hasattr(parameters, key):
            value = getattr(parameters, key)
            if np.isscalar(value):
                out[key] = value
    return out


def parameter_values(parameters: Any = None) -> dict[str, Any]:
    """Return canonical values with defaults filled in."""

    supplied = _parameter_mapping(parameters)
    values = dict(DEFAULT_PARAMETERS)
    values.update(supplied)
    # ``axis_ratio`` is an explicit parameterisation choice.  When present it
    # owns the minor axis, so an inherited/default ``b`` cannot become a
    # second independent degree of freedom.
    if "axis_ratio" in supplied:
        ratio = float(supplied["axis_ratio"])
        if not np.isfinite(ratio) or not 0 < ratio <= 1:
            raise ValueError("axis_ratio must be in (0, 1]")
        values["b"] = float(values["a"]) * ratio
    # ``amplitude`` and ``radial_width`` are input shorthands only.  They are
    # expanded into effective model parameters and removed so the optimizer
    # never carries zero-Jacobian duplicate variables.
    if "amplitude" in supplied:
        if "amplitude_plus" not in supplied:
            values["amplitude_plus"] = values["amplitude"]
        if "amplitude_minus" not in supplied:
            values["amplitude_minus"] = values["amplitude"]
        values.pop("amplitude", None)
    if "radial_width" in supplied:
        if "radial_sigma" not in supplied:
            values["radial_sigma"] = values["radial_width"]
        if "radial_gamma" not in supplied:
            values["radial_gamma"] = values["radial_width"]
        values.pop("radial_width", None)
    return values


def _positive(value: Any, fallback: float = 1.0) -> float:
    try:
        value = float(value)
    except Exception:
        return float(fallback)
    return value if np.isfinite(value) else float(fallback)


def _periodic_distance(angle: np.ndarray, centre: float) -> np.ndarray:
    return np.angle(np.exp(1j * (np.asarray(angle, dtype=float) - float(centre))))


def _ellipse_radius(angle: np.ndarray, a: float, b: float, theta: float) -> np.ndarray:
    delta = np.asarray(angle, dtype=float) - float(theta)
    a = max(abs(float(a)), np.finfo(float).eps)
    b = max(abs(float(b)), np.finfo(float).eps)
    return a * b / np.sqrt((b * np.cos(delta)) ** 2 + (a * np.sin(delta)) ** 2)


def ellipse_polar_radius(angle: np.ndarray | float, a: float, b: float, theta: float) -> np.ndarray:
    """Public polar ellipse radius helper (q units)."""

    return _ellipse_radius(np.asarray(angle, dtype=float), a, b, theta)


def _angular_envelope(angle: np.ndarray, centre: float, width: float) -> np.ndarray:
    width = max(abs(float(width)), np.finfo(float).eps)
    delta = _periodic_distance(angle, centre)
    return np.exp(-0.5 * (delta / width) ** 2)


def _radial_line_shape(delta: np.ndarray, values: Mapping[str, Any]) -> np.ndarray:
    sigma = abs(_positive(values.get("radial_sigma", 0.04), 0.04))
    gamma = abs(_positive(values.get("radial_gamma", sigma), sigma))
    if "radial_fwhm" in values:
        fwhm = abs(_positive(values.get("radial_fwhm"), 0.0))
        if fwhm > 0:
            sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
            gamma = fwhm / 2.0
    gaussian = np.exp(-0.5 * (np.asarray(delta, dtype=float) / max(sigma, np.finfo(float).eps)) ** 2)
    lorentzian = 1.0 / (1.0 + (np.asarray(delta, dtype=float) / max(gamma, np.finfo(float).eps)) ** 2)
    eta = float(np.clip(_positive(values.get("eta", 0.0), 0.0), 0.0, 1.0))
    return (1.0 - eta) * gaussian + eta * lorentzian


def _smooth_background(q: np.ndarray, values: Mapping[str, Any]) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    result = float(_positive(values.get("background", 0.0), 0.0))
    bg_width = abs(_positive(values.get("background_width", 1.0), 1.0))
    scaled_q = np.maximum(q, 0.0) / max(bg_width, np.finfo(float).eps)
    # Non-negative coefficients make every component monotonically decreasing
    # with q, matching the background constraint used in the 2016 workflow.
    result = result + max(0.0, float(values.get("background_slope", 0.0))) / (1.0 + scaled_q)
    result = result + max(0.0, float(values.get("background_curvature", 0.0))) / (1.0 + scaled_q * scaled_q)
    bg_amp = max(0.0, float(values.get("background_amplitude", 0.0)))
    if bg_amp:
        result = result + bg_amp * np.exp(-0.5 * (q / max(bg_width, np.finfo(float).eps)) ** 2)
    return result


def double_ellipse_intensity(
    qx: Any,
    qy: Any,
    parameters: Any = None,
    *,
    return_components: bool = False,
    reference_axis_deg: float = 0.0,
) -> np.ndarray | dict[str, np.ndarray]:
    """Evaluate the empirical symmetric double-ellipse intensity model.

    Parameters ``a``/``b`` are semi-axis q values and ``theta`` is in radians.
    ``reference_axis_deg`` is the laboratory direction of the model's local
    +qx/equatorial axis; theta and lobe_angle are measured in that specimen
    frame.  Four angular envelopes are centred at
    ``(+/-lobe_angle, pi+/-lobe_angle)`` in the local frame.
    The two opposite envelopes belonging to each ellipse share its radial
    trajectory, producing a direct, symmetric butterfly pattern.
    """

    values = parameter_values(parameters)
    qx, qy = np.broadcast_arrays(np.asarray(qx, dtype=float), np.asarray(qy, dtype=float))
    q = np.hypot(qx, qy)
    angle = np.arctan2(qy, qx)
    reference_axis = float(np.deg2rad(reference_axis_deg))
    relative_angle = _periodic_distance(angle, reference_axis)
    # Azimuth is undefined at q=0.  Use one specimen-frame convention there
    # so a rotated coordinate system cannot change the central-pixel value.
    relative_angle = np.where(q <= np.finfo(float).eps, 0.0, relative_angle)
    a = abs(_positive(values.get("a", 1.0), 1.0))
    b = abs(_positive(values.get("b", 0.7), 0.7))
    theta = float(values.get("theta", 0.0))
    phi0 = float(values.get("lobe_angle", np.deg2rad(30.0)))
    angular_width = abs(_positive(values.get("angular_width", np.deg2rad(8.0)), np.deg2rad(8.0)))
    amp = float(values.get("amplitude", 1.0))
    amp_plus = float(values.get("amplitude_plus", amp))
    amp_minus = float(values.get("amplitude_minus", amp))

    # Each branch has two opposite lobes.  This is the four-envelope form,
    # and not a mirror completion of observed data: it is only the model.
    plus_envelope = _angular_envelope(relative_angle, phi0, angular_width) + _angular_envelope(relative_angle, np.pi + phi0, angular_width)
    minus_envelope = _angular_envelope(relative_angle, -phi0, angular_width) + _angular_envelope(relative_angle, np.pi - phi0, angular_width)
    plus_radial = _radial_line_shape(q - _ellipse_radius(relative_angle, a, b, theta), values)
    minus_radial = _radial_line_shape(q - _ellipse_radius(relative_angle, a, b, -theta), values)
    branch_plus = amp_plus * plus_envelope * plus_radial
    branch_minus = amp_minus * minus_envelope * minus_radial
    signal = branch_plus + branch_minus
    background = _smooth_background(q, values)
    total = signal + background
    if not return_components:
        return total
    return {
        "intensity": total,
        "signal": signal,
        "background": background,
        "branch_plus": branch_plus,
        "branch_minus": branch_minus,
        "q": q,
        "angle": angle,
        "angle_relative": relative_angle,
        "reference_axis_deg": np.full_like(q, float(reference_axis_deg)),
    }


evaluate_double_ellipse = double_ellipse_intensity
double_ellipse_model = double_ellipse_intensity
model_intensity = double_ellipse_intensity


class DoubleEllipseIntensityModel:
    """Callable object convenient for GUI model-preview code."""

    flags = MODEL_FLAGS

    def __init__(self, parameters: Any = None, *, reference_axis_deg: float = 0.0):
        self.parameters = parameters if parameters is not None else dict(DEFAULT_PARAMETERS)
        self.reference_axis_deg = float(reference_axis_deg)

    def __call__(self, qx: Any, qy: Any) -> np.ndarray:
        return np.asarray(
            double_ellipse_intensity(
                qx,
                qy,
                self.parameters,
                reference_axis_deg=self.reference_axis_deg,
            )
        )

    def evaluate(self, qx: Any, qy: Any, parameters: Any = None, *, return_components: bool = False,
                 reference_axis_deg: float | None = None) -> Any:
        reference = self.reference_axis_deg if reference_axis_deg is None else float(reference_axis_deg)
        return double_ellipse_intensity(
            qx,
            qy,
            self.parameters if parameters is None else parameters,
            return_components=return_components,
            reference_axis_deg=reference,
        )


def _fixed_names(parameters: Any, explicit: Any = None) -> set[str]:
    fixed: set[str] = set()
    if isinstance(explicit, Mapping):
        fixed.update(canonical_parameter_name(k) for k, v in explicit.items() if bool(v))
    elif explicit is not None:
        fixed.update(canonical_parameter_name(k) for k in explicit)
    source = parameters
    attr = _get_field(source, ("fixed", "fixed_names"))
    if isinstance(attr, Mapping):
        fixed.update(canonical_parameter_name(k) for k, v in attr.items() if bool(v))
    elif attr is not None and not isinstance(attr, (str, bytes)):
        try:
            fixed.update(canonical_parameter_name(k) for k in attr)
        except TypeError:
            pass
    specs = _get_field(source, ("specs", "parameters_spec", "parameter_specs"))
    if isinstance(specs, Mapping):
        for key, spec in specs.items():
            if bool(_get_field(spec, ("fixed", "is_fixed"), False)) or bool(
                _get_field(spec, ("is_tied",), False)
            ):
                fixed.add(canonical_parameter_name(key))
    return fixed


def _bounds_for(name: str, value: float, parameters: Any, explicit: Any = None) -> tuple[float, float]:
    eps = np.finfo(float).eps
    if name in {"a", "b", "radial_sigma", "radial_gamma", "angular_width", "background_width"}:
        default = (eps, max(abs(value) * 10.0, 10.0 * eps))
    elif name == "axis_ratio":
        default = (eps, 1.0)
    elif name == "eta":
        default = (0.0, 1.0)
    elif name in {"theta", "lobe_angle"}:
        default = (-np.pi, np.pi)
    elif name in {
        "amplitude_plus", "amplitude_minus", "background", "background_slope",
        "background_curvature", "background_amplitude",
    }:
        default = (0.0, max(abs(value) * 20.0 + 1.0, 1.0))
    else:
        default = (-np.inf, np.inf)

    def merge(candidate: Any) -> tuple[float, float] | None:
        if candidate is None:
            return None
        try:
            lo, hi = candidate
            return (
                default[0] if lo is None else float(lo),
                default[1] if hi is None else float(hi),
            )
        except Exception:
            return None

    if isinstance(explicit, Mapping):
        candidate = explicit.get(name, explicit.get(canonical_parameter_name(name)))
        merged = merge(candidate)
        if merged is not None:
            return merged
    specs = _get_field(parameters, ("specs", "parameter_specs", "parameters_spec"))
    if isinstance(specs, Mapping):
        spec = specs.get(name)
        candidate = _get_field(spec, ("bounds", "limit", "limits")) if spec is not None else None
        merged = merge(candidate)
        if merged is not None:
            return merged
    return default


def _restore_parameters(original: Any, values: Mapping[str, Any]) -> Any:
    """Best-effort preservation of the shared ParameterSet API."""

    if original is None:
        return dict(values)
    # ParameterSet is a MutableMapping, but preserving its fixed/tied specs
    # is more important than treating it as a plain dict.
    if hasattr(original, "update_values") and hasattr(original, "copy"):
        try:
            restored = original.copy()
            known: dict[str, Any] = {}
            names = set(getattr(restored, "names", tuple(restored)))
            for name, value in values.items():
                if name in names:
                    spec = restored[name]
                    if not bool(_get_field(spec, ("is_tied",), False)):
                        known[name] = value
            if known:
                restored.update_values(known)
            return restored
        except Exception:
            pass
    if isinstance(original, Mapping):
        return dict(values)
    for method_name in ("with_values", "replace_values", "updated", "copy_with"):
        method = getattr(original, method_name, None)
        if callable(method):
            try:
                return method(dict(values))
            except Exception:
                try:
                    return method(**dict(values))
                except Exception:
                    pass
    if dataclasses.is_dataclass(original):
        try:
            return dataclasses.replace(original, values=dict(values))
        except Exception:
            pass
    cls = type(original)
    for kwargs in ({"values": dict(values)}, {"parameters": dict(values)}):
        try:
            return cls(**kwargs)
        except Exception:
            pass
    return dict(values)


def deterministic_pixel_sample(indices: Sequence[int] | np.ndarray, max_pixels: int | None, seed: int = 0) -> np.ndarray:
    """Deterministically choose a sorted subset of flattened pixel indices."""

    values = np.asarray(indices, dtype=int).ravel()
    values = values[np.unique(values, return_index=True)[1]] if values.size else values
    if max_pixels is None or len(values) <= int(max_pixels):
        return np.sort(values)
    if int(max_pixels) <= 0:
        raise ValueError("max_pixels must be positive")
    rng = np.random.default_rng(int(seed))
    selected = rng.choice(values, size=int(max_pixels), replace=False)
    return np.sort(np.asarray(selected, dtype=int))


def _validate_multistart_count(multistart: Any) -> int:
    if isinstance(multistart, (bool, np.bool_)) or not isinstance(multistart, (int, np.integer)):
        raise TypeError("multistart must be an integer >= 1")
    count = int(multistart)
    if count < 1:
        raise ValueError("multistart must be an integer >= 1")
    return count


def _multistart_vectors(
    names: Sequence[str],
    base: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    count: int,
) -> tuple[np.ndarray, ...]:
    """Build deterministic bounded starts for the full-pixel optimizer."""

    if count == 1:
        return (np.asarray(base, dtype=float).copy(),)
    fractions = (0.20, 0.80, 0.35, 0.65, 0.50, 0.10, 0.90)
    offsets = (-1.5, 1.5, -0.75, 0.75, 0.0, -2.0, 2.0)
    # Geometry-only starts are deliberately first.  A poor full-pixel start
    # can be trapped in a broad, nearly circular intensity basin; changing
    # only the ellipse scale/ratio/orientation gives that basin a physically
    # meaningful alternative while retaining the user's widths, amplitudes,
    # eta, and background values.  The original start remains index 0.
    geometry_seeds = (
        {"theta": np.pi / 4.0, "axis_ratio": 0.05},
        {"theta": -np.pi / 4.0, "axis_ratio": 0.05},
        {"theta": 0.0, "axis_ratio": 0.10},
        {"theta": np.pi / 4.0, "axis_ratio": 0.20},
        {"theta": -np.pi / 4.0, "axis_ratio": 0.35},
    )

    def bounded_value(name: str, value: float, lo: float, hi: float) -> float:
        if name == "axis_ratio" and lo > 0.0 and hi > 0.0:
            value = max(float(value), 0.005 * float(hi))
        if np.isfinite(lo):
            value = max(float(value), float(np.nextafter(lo, hi)))
        if np.isfinite(hi):
            value = min(float(value), float(np.nextafter(hi, lo)))
        return float(value)

    starts: list[np.ndarray] = [np.asarray(base, dtype=float).copy()]
    for start_index in range(1, count):
        if start_index <= len(geometry_seeds):
            vector = np.asarray(base, dtype=float).copy()
            seed = geometry_seeds[start_index - 1]
            for index, (name, _value, lo, hi) in enumerate(zip(names, base, lower, upper)):
                if name in seed:
                    vector[index] = bounded_value(name, float(seed[name]), lo, hi)
            starts.append(vector)
            continue
        fraction = fractions[(start_index - 1) % len(fractions)]
        offset = offsets[(start_index - 1) % len(offsets)]
        vector = np.asarray(base, dtype=float).copy()
        for index, (name, value, lo, hi) in enumerate(zip(names, base, lower, upper)):
            # Bounds that came from the generic +/-1e12 fallback are
            # effectively unbounded.  Keep those starts local to the current
            # estimate rather than placing them at absurd absolute values.
            bounded = np.isfinite(lo) and np.isfinite(hi) and abs(lo) < 1.0e6 and abs(hi) < 1.0e6
            if bounded:
                if name in {"a", "axis_ratio", "radial_sigma", "radial_gamma", "angular_width", "background_width"} and lo > 0.0 and hi > 0.0:
                    log_lo = math.log(max(float(lo), 0.005 * float(hi)))
                    log_hi = math.log(float(hi))
                    vector[index] = math.exp(log_lo + fraction * (log_hi - log_lo))
                else:
                    vector[index] = lo + fraction * (hi - lo)
            else:
                scale = max(abs(float(value)), 1.0)
                vector[index] = float(value) + offset * scale
            vector[index] = bounded_value(name, float(vector[index]), lo, hi)
        starts.append(vector)
    return tuple(starts)


def _robust_cost(residual: np.ndarray, loss: str, f_scale: float) -> float:
    """Return scipy least-squares' robust objective for weighted residuals."""

    residual = np.asarray(residual, dtype=float)
    if not np.all(np.isfinite(residual)):
        return float("inf")
    scale = max(float(f_scale), np.finfo(float).eps)
    z = np.square(residual / scale)
    kind = str(loss).lower()
    if kind == "linear":
        rho = z
    elif kind == "soft_l1":
        rho = 2.0 * (np.sqrt(1.0 + z) - 1.0)
    elif kind == "huber":
        rho = np.where(z <= 1.0, z, 2.0 * np.sqrt(z) - 1.0)
    elif kind == "cauchy":
        rho = np.log1p(z)
    elif kind == "arctan":
        rho = np.arctan(z)
    else:
        raise ValueError("robust_loss must be one of linear, soft_l1, huber, cauchy, arctan")
    return float(0.5 * scale * scale * np.sum(rho))


def _fit_arrays(frame: Any, qmap: Any, q_window: Any = None, q_range: Any = None, mask: Any = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[int, ...]]:
    values, q, angle, valid = _extract_maps(frame, qmap, mask)
    shape = np.asarray(_get_field(frame, ("data", "intensity", "image", "values"))).shape
    q_min, q_max = _q_limits(q, q_window, q_range)
    valid &= q >= q_min
    valid &= q <= q_max
    qx, qy = q * np.cos(angle), q * np.sin(angle)
    return values, qx, qy, valid, tuple(shape)


def _parameter_key_order(values: Mapping[str, Any], fixed: set[str]) -> list[str]:
    names: list[str] = []
    for name, value in values.items():
        if name not in DEFAULT_PARAMETERS and name != "axis_ratio":
            continue
        if (
            name in fixed
            or name in {"theta_deg", "lobe_angle_deg"}
            or (name == "b" and "axis_ratio" in values)
            or isinstance(value, (str, bytes, bool))
        ):
            continue
        try:
            number = float(value)
        except Exception:
            continue
        if np.isfinite(number):
            names.append(name)
    return names


def _candidate_parameter_values(
    original: Any,
    base_values: Mapping[str, Any],
    names: Sequence[str],
    vector: Sequence[float],
) -> dict[str, Any]:
    updates = {name: float(value) for name, value in zip(names, vector)}
    if isinstance(original, ParameterSet):
        candidate = original.copy()
        allowed = {
            name: value
            for name, value in updates.items()
            if name in candidate.names and not candidate[name].is_tied
        }
        if allowed:
            candidate.update_values(allowed)
        return parameter_values(candidate)
    candidate = dict(base_values)
    candidate.update(updates)
    if "axis_ratio" in candidate and "b" not in updates:
        candidate["b"] = float(candidate["a"]) * float(candidate["axis_ratio"])
    return candidate


def _estimate_intensity_scale(
    initial: Any,
    observed: np.ndarray,
    fixed_names: set[str],
) -> tuple[Any, bool]:
    """Estimate only the linear intensity scale of an initial model.

    Geometry remains untouched.  The estimate is deliberately robust to a
    small number of hot pixels and gives absolute-intensity data a feasible
    starting point (and therefore feasible automatic bounds) without
    overwriting parameters the user fixed explicitly.
    """

    finite = np.asarray(observed, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return initial, False
    baseline = max(0.0, float(np.percentile(finite, 20.0)))
    upper = float(np.percentile(finite, 99.5))
    signal = max(upper - baseline, float(np.nanstd(finite)), np.finfo(float).eps)
    updates: dict[str, float] = {}
    for name in ("amplitude_plus", "amplitude_minus"):
        if name not in fixed_names:
            updates[name] = signal
    if "background" not in fixed_names:
        updates["background"] = baseline
    if not updates:
        return initial, False
    if isinstance(initial, ParameterSet):
        scaled = initial.copy()
        allowed = {
            name: value
            for name, value in updates.items()
            if name in scaled.names and not scaled[name].is_tied
        }
        if allowed:
            scaled.update_values(allowed)
        return scaled, bool(allowed)
    values = parameter_values(initial)
    values.update(updates)
    return values, True


def fit_intensity_model(
    frame: Any,
    qmap: Any,
    initial: Any = None,
    *,
    q_window: Any = None,
    q_range: Any = None,
    fixed: Any = None,
    bounds: Mapping[str, Sequence[float]] | None = None,
    max_pixels: int | None = None,
    seed: int = 0,
    scales: Sequence[float] = (0.25, 0.5, 1.0),
    robust_loss: str = "soft_l1",
    f_scale: float = 1.0,
    max_nfev: int = 800,
    mask: Any = None,
    sigma: Any = None,
    weights: Any = None,
    reference_axis_deg: float = 0.0,
    auto_scale_initial: bool = False,
    multistart: int = 1,
    full2d_multistart: int | None = None,
    cancel_event: Any = None,
) -> IntensityFitResult:
    """Fit the empirical model on all valid pixels by default.

    ``ParameterSet`` instances are accepted directly.  Fixed flags and bounds
    are read from their public ``fixed``/``specs`` fields when available; the
    explicit arguments take precedence.  ``max_pixels`` enables deterministic
    per-scale subsets when explicitly supplied; otherwise one full-pixel
    optimization is performed even when multiple scales were requested.
    ``multistart`` (or its ``full2d_multistart`` alias) adds deterministic
    bounded candidate fits; one start remains the backwards-compatible
    default.
    """

    raise_if_cancelled(cancel_event, "intensity:start")
    if least_squares is None:
        raise RuntimeError("scipy.optimize.least_squares is required for refinement")
    if full2d_multistart is not None:
        if multistart != 1:
            raise ValueError("supply only one of multistart or full2d_multistart")
        multistart = full2d_multistart
    multistart = _validate_multistart_count(multistart)
    if sigma is not None and weights is not None:
        raise ValueError("supply either sigma or weights, not both")
    values_obs, qx, qy, valid, shape = _fit_arrays(frame, qmap, q_window, q_range, mask)
    finite = valid & np.isfinite(values_obs) & np.isfinite(qx) & np.isfinite(qy)
    sigma_flat: np.ndarray | None = None
    weight_flat: np.ndarray | None = None
    if sigma is not None:
        sigma_array = np.asarray(sigma, dtype=float)
        if sigma_array.shape != shape:
            raise ValueError(f"sigma shape {sigma_array.shape} does not match image shape {shape}")
        sigma_flat = sigma_array.ravel()
        finite &= np.isfinite(sigma_flat) & (sigma_flat > 0)
        weighting = "per_pixel_sigma"
    elif weights is not None:
        weight_array = np.asarray(weights, dtype=float)
        if weight_array.shape != shape:
            raise ValueError(f"weights shape {weight_array.shape} does not match image shape {shape}")
        weight_flat = weight_array.ravel()
        finite &= np.isfinite(weight_flat) & (weight_flat > 0)
        weighting = "per_pixel_weight"
    else:
        weighting = "equal_robust_scaled_by_global_std"
    all_indices = np.flatnonzero(finite)
    if not all_indices.size:
        raise ValueError("q window and mask leave no finite pixels for fitting")
    fixed_names = _fixed_names(initial, fixed)
    scale_estimated = False
    if auto_scale_initial:
        initial, scale_estimated = _estimate_intensity_scale(
            initial,
            values_obs[all_indices],
            fixed_names,
        )
    values = parameter_values(initial)
    names = _parameter_key_order(values, fixed_names)
    obs_scale = max(float(np.nanstd(values_obs[all_indices])), np.finfo(float).eps)

    def scaled_residual(residual_values: np.ndarray, indices: np.ndarray) -> np.ndarray:
        if sigma_flat is not None:
            return residual_values / sigma_flat[indices]
        if weight_flat is not None:
            return residual_values * np.sqrt(weight_flat[indices])
        return residual_values / obs_scale

    if not names:
        prediction = np.asarray(
            double_ellipse_intensity(qx, qy, values, reference_axis_deg=reference_axis_deg),
            dtype=float,
        )
        residual = prediction - values_obs
        weighted = scaled_residual(residual[all_indices], all_indices)
        no_free_cost = 0.5 * float(np.sum(weighted**2))
        return IntensityFitResult(
            _restore_parameters(initial, values), initial, True, "no free parameters",
            no_free_cost,
            float(np.sqrt(np.mean(residual[all_indices] ** 2))),
            0, all_indices, prediction, residual, shape, tuple(),
            flags=MODEL_FLAGS
            + (("initial_intensity_scale_estimated",) if scale_estimated else ())
            + ("no_free_parameters",),
            weighting=weighting,
            weighted_rmse=float(np.sqrt(np.mean(weighted**2))),
            stderr={name: float("nan") for name in values},
            bound_flags={name: False for name in values},
            ndata=int(len(all_indices)),
            sampled_n=int(len(all_indices)),
            sample_rmse=float(np.sqrt(np.mean(residual[all_indices] ** 2))),
            reference_axis_deg=float(reference_axis_deg),
            sample_cost=no_free_cost,
            full_cost=no_free_cost,
        )

    lower = []
    upper = []
    for name in names:
        lo, hi = _bounds_for(name, float(values[name]), initial, bounds)
        if not np.isfinite(lo):
            lo = -1e12
        if not np.isfinite(hi):
            hi = 1e12
        if hi <= lo:
            raise ValueError(f"invalid bounds for parameter {name!r}")
        lower.append(lo)
        upper.append(hi)
    lower_arr, upper_arr = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    current = np.asarray([float(values[name]) for name in names], dtype=float)
    current = np.clip(current, lower_arr + 1e-12, upper_arr - 1e-12)
    history: list[dict[str, Any]] = []
    candidate_solutions: list[dict[str, Any]] = []
    best_result = None
    best_indices = all_indices
    selected_start_index = 0
    scales = tuple(float(scale) for scale in scales)
    if not scales or any(scale <= 0 for scale in scales):
        raise ValueError("scales must contain positive values")
    fit_scales = (1.0,) if max_pixels is None else scales

    for level, scale in enumerate(fit_scales):
        target = None if max_pixels is None else max(1, int(round(float(max_pixels) * scale)))
        selected = deterministic_pixel_sample(all_indices, target, seed=int(seed) + level * 1_000_003)
        best_indices = selected
        obs = values_obs[selected]
        xsel, ysel = qx[selected], qy[selected]

        def residual_fn(vector: np.ndarray) -> np.ndarray:
            raise_if_cancelled(cancel_event, "intensity:residual")
            candidate = _candidate_parameter_values(initial, values, names, vector)
            predicted = np.asarray(
                double_ellipse_intensity(
                    xsel,
                    ysel,
                    candidate,
                    reference_axis_deg=reference_axis_deg,
                ),
                dtype=float,
            )
            # Robust loss handles occasional beam-stop/hot-pixel remnants;
            # deterministic weighting remains stable across repeated runs.
            return scaled_residual(predicted - obs, selected)

        starts = _multistart_vectors(names, current, lower_arr, upper_arr, multistart)
        scale_candidates: list[tuple[int, Any, np.ndarray, dict[str, Any], float]] = []
        for start_index, start_vector in enumerate(starts):
            raise_if_cancelled(cancel_event, "intensity:multistart")
            result = least_squares(
                residual_fn,
                start_vector,
                bounds=(lower_arr, upper_arr),
                loss=robust_loss,
                f_scale=max(float(f_scale), np.finfo(float).eps),
                # Geometry, angular widths, and flat axis ratios live on
                # different numerical scales.  Jacobian scaling is safe for
                # both the default single start and explicit multistart.
                x_scale="jac",
                max_nfev=int(max_nfev),
            )
            raise_if_cancelled(cancel_event, "intensity:complete")
            candidate_values = _candidate_parameter_values(initial, values, names, result.x)
            candidate_full_prediction = np.asarray(
                double_ellipse_intensity(
                    qx[all_indices],
                    qy[all_indices],
                    candidate_values,
                    reference_axis_deg=reference_axis_deg,
                ),
                dtype=float,
            )
            candidate_full_weighted = scaled_residual(
                candidate_full_prediction - values_obs[all_indices],
                all_indices,
            )
            full_cost = _robust_cost(candidate_full_weighted, robust_loss, f_scale)
            record = {
                "scale": float(scale),
                "start_index": int(start_index),
                "start_values": {name: float(value) for name, value in zip(names, start_vector)},
                "values": {
                    name: float(value)
                    for name, value in candidate_values.items()
                    if np.isscalar(value) and not isinstance(value, (str, bytes))
                },
                "success": bool(result.success),
                "finite_cost": bool(np.isfinite(result.cost)),
                "cost": float(result.cost),
                "sample_cost": float(result.cost),
                "full_cost": float(full_cost),
                "nfev": int(result.nfev),
                "message": str(result.message),
            }
            scale_candidates.append(
                (start_index, result, np.asarray(result.x, dtype=float), record, full_cost)
            )
            candidate_solutions.append(record)
        selected_start_index, result, result_vector, selected_record, selected_full_cost = min(
            scale_candidates,
            key=lambda item: (
                not bool(item[1].success),
                not bool(np.isfinite(item[4])),
                float(item[4]) if np.isfinite(item[4]) else float("inf"),
                item[0],
            ),
        )
        selected_record["selected_for_full_objective"] = True
        selected_sample_cost = float(result.cost)
        current = result_vector
        values = _candidate_parameter_values(initial, values, names, current)
        history.append({
            "scale": scale,
            "n_pixels": int(len(selected)),
            "cost": float(selected_full_cost),
            "sample_cost": selected_sample_cost,
            "full_cost": float(selected_full_cost),
            "nfev": int(sum(item[1].nfev for item in scale_candidates)),
            "seed": int(seed) + level * 1_000_003,
            "selected_start_index": int(selected_start_index),
            "multistart_count": int(multistart),
        })
        best_result = result

    prediction = np.asarray(
        double_ellipse_intensity(qx, qy, values, reference_axis_deg=reference_axis_deg),
        dtype=float,
    )
    residual = prediction - values_obs
    full_residual = residual[all_indices]
    full_weighted = scaled_residual(full_residual, all_indices)
    sampled_residual = residual[best_indices]
    sampled_weighted = scaled_residual(sampled_residual, best_indices)
    sample_cost = _robust_cost(sampled_weighted, robust_loss, f_scale)
    full_cost = _robust_cost(full_weighted, robust_loss, f_scale)
    cost = full_cost
    covariance: np.ndarray | None = None
    condition = float("nan")
    stderr = {name: float("nan") for name in values}
    if best_result is not None and getattr(best_result, "jac", None) is not None:
        jacobian = np.asarray(best_result.jac, dtype=float)
        if jacobian.ndim == 2 and jacobian.shape[1] == len(names) and jacobian.size:
            normal_matrix = jacobian.T @ jacobian
            try:
                condition = float(np.linalg.cond(normal_matrix))
                dof = max(jacobian.shape[0] - jacobian.shape[1], 1)
                variance = float(2.0 * best_result.cost / dof)
                covariance = np.linalg.pinv(normal_matrix) * variance
                diagonal = np.diag(covariance)
                errors = np.sqrt(np.where(diagonal >= 0, diagonal, np.nan))
                stderr.update({name: float(error) for name, error in zip(names, errors)})
            except np.linalg.LinAlgError:
                covariance = None
    bound_flags = {name: False for name in values}
    tolerance = 1e-7
    for name, value, lo, hi in zip(names, current, lower_arr, upper_arr):
        scale_value = max(1.0, abs(float(value)))
        bound_flags[name] = bool(
            (np.isfinite(lo) and abs(float(value) - float(lo)) <= tolerance * scale_value)
            or (np.isfinite(hi) and abs(float(value) - float(hi)) <= tolerance * scale_value)
        )
    candidate_success = [bool(record["success"]) for record in candidate_solutions]
    candidate_failure_flags: tuple[str, ...] = ()
    if multistart > 1 and any(not success for success in candidate_success):
        candidate_failure_flags = (
            ("all_candidates_failed",)
            if not any(candidate_success)
            else ("multistart_candidate_failures",)
        )
    result_flags = (
        MODEL_FLAGS
        + (("initial_intensity_scale_estimated",) if scale_estimated else ())
        + ("covariance_local_linear_approximation",)
        + (("deterministic_multistart",) if multistart > 1 else ())
        + candidate_failure_flags
        + (
            ("solver_failed",)
            if best_result is None or not bool(best_result.success)
            else ()
        )
        + (("full_objective_candidate_selection",) if multistart > 1 else ())
    )
    return IntensityFitResult(
        parameters=_restore_parameters(initial, values),
        initial_parameters=initial,
        success=bool(best_result.success) if best_result is not None else False,
        message=str(best_result.message) if best_result is not None else "solver did not run",
        cost=cost,
        rmse=float(np.sqrt(np.mean(full_residual * full_residual))) if len(full_residual) else float("nan"),
        nfev=int(sum(item["nfev"] for item in history)),
        sampled_indices=np.asarray(best_indices, dtype=int),
        prediction=prediction,
        residual=residual,
        image_shape=shape,
        scale_history=tuple(history),
        flags=result_flags,
        weighting=weighting,
        weighted_rmse=float(np.sqrt(np.mean(full_weighted**2))) if len(full_weighted) else float("nan"),
        stderr=stderr,
        covariance=covariance,
        covariance_names=tuple(names),
        condition_number=condition,
        bound_flags=bound_flags,
        ndata=int(len(all_indices)),
        sampled_n=int(len(best_indices)),
        sample_rmse=float(np.sqrt(np.mean(sampled_residual * sampled_residual))) if len(sampled_residual) else float("nan"),
        reference_axis_deg=float(reference_axis_deg),
        candidate_solutions=tuple(candidate_solutions),
        selected_start_index=int(selected_start_index),
        multistart_count=int(multistart),
        sample_cost=float(sample_cost),
        full_cost=float(full_cost),
        selection_objective="full_valid_weighted_robust_cost",
    )


refine_intensity = fit_intensity_model
fit_double_ellipse_intensity = fit_intensity_model
least_squares_refine = fit_intensity_model


__all__ = [
    "MODEL_FLAGS",
    "DEFAULT_PARAMETERS",
    "default_intensity_parameters",
    "IntensityFitResult",
    "DoubleEllipseIntensityModel",
    "canonical_parameter_name",
    "parameter_values",
    "ellipse_polar_radius",
    "double_ellipse_intensity",
    "evaluate_double_ellipse",
    "double_ellipse_model",
    "model_intensity",
    "deterministic_pixel_sample",
    "fit_intensity_model",
    "refine_intensity",
    "fit_double_ellipse_intensity",
    "least_squares_refine",
]
