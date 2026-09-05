"""Measurements for anisotropic two-dimensional SAXS patterns.

The functions in this module deliberately operate on the small public seam
used by the rest of the project: an :class:`ImageFrame`, a :class:`QMap`, and
a q window.  The concrete model classes are imported lazily by the caller;
the adapters below also accept mappings and light-weight test doubles.  This
keeps the measurement code usable while the data-model module is being wired
up by the application layer.

The output is intentionally descriptive rather than structural.  A fitted
track is an *apparent geometry* of the measured intensity surface; it is not
an identification of a unique lamellar microstructure.  The two flags
``apparent_geometry_only`` and ``nonunique_inverse_problem`` are attached to
all geometry-bearing results for that reason.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from .cancellation import raise_if_cancelled

try:  # scipy is a declared project dependency, but keep imports friendly.
    from scipy.ndimage import gaussian_filter, gaussian_filter1d, map_coordinates
    from scipy.optimize import least_squares
    from scipy.signal import find_peaks, peak_prominences
except Exception:  # pragma: no cover - exercised only in a partial install
    gaussian_filter = None
    gaussian_filter1d = None
    least_squares = None
    map_coordinates = None
    find_peaks = None
    peak_prominences = None

try:
    # The shared model makes a measured ridge point consumable by
    # butterfly_saxs.ellipse.fit_symmetric_ellipses without conversion.
    from .models import RidgePoint as _SharedRidgePoint
except Exception:  # pragma: no cover - allows this module to bootstrap first
    class _SharedRidgePoint:  # type: ignore[no-redef]
        def __init__(self, qx: float, qy: float, intensity: float | None = None,
                     weight: float = 1.0, component: int | None = None,
                     metadata: Mapping[str, Any] | None = None) -> None:
            self.qx, self.qy = float(qx), float(qy)
            self.intensity = intensity
            self.weight = weight
            self.component = component
            self.metadata = dict(metadata or {})

try:
    # Keep one numerical implementation of the constrained pair fit.  The
    # observables module owns the user-facing, legacy-shaped result adapter;
    # the q-space optimizer and its diagnostics remain in ``ellipse.py``.
    from .ellipse import (
        EllipseGeometry as _CanonicalEllipseGeometry,
        ellipse_sampson_residuals as _canonical_sampson_residuals,
        fit_symmetric_ellipses as _fit_canonical_symmetric_ellipses,
        symmetric_ellipse_residuals as _canonical_symmetric_residuals,
    )
except Exception:  # pragma: no cover - only useful during partial bootstrap
    _CanonicalEllipseGeometry = None
    _canonical_sampson_residuals = None
    _fit_canonical_symmetric_ellipses = None
    _canonical_symmetric_residuals = None


APPARENT_FLAGS = (
    "apparent_geometry_only",
    "nonunique_inverse_problem",
)
# Relative tolerance for treating a fitted q-space ellipse as origin-centred
# before deriving the optional physical spacing proxies.
_SPACING_CENTER_REL_TOL = 1e-8


class _MappingResult:
    """Small common adapter: results work as both objects and mappings."""

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def keys(self) -> tuple[str, ...]:
        return tuple(self.__dataclass_fields__)  # type: ignore[attr-defined]

    def as_dict(self) -> dict[str, Any]:
        return {key: getattr(self, key) for key in self.keys()}


@dataclass
class AngularSpectrum(_MappingResult):
    """Azimuthal intensity profile in a q window.

    ``angle`` is in radians and is centred on ``[-pi, pi)``.  ``coverage`` is
    the fraction of finite candidate pixels in each angular bin that survived
    the detector mask; it is therefore useful for deciding whether an
    apparent lobe is supported by data rather than by an empty sector.
    """

    angle: np.ndarray
    intensity: np.ndarray
    counts: np.ndarray
    candidate_counts: np.ndarray
    coverage: np.ndarray
    q_min: float
    q_max: float
    q_center: float
    statistic: str = "mean"
    flags: tuple[str, ...] = APPARENT_FLAGS
    q_unit: str = "unknown"

    @property
    def azimuth(self) -> np.ndarray:
        return self.angle

    @property
    def profile(self) -> np.ndarray:
        return self.intensity

    @property
    def global_coverage(self) -> float:
        denom = float(np.sum(self.candidate_counts))
        return float(np.sum(self.counts) / denom) if denom else 0.0


@dataclass
class LobeMetrics(_MappingResult):
    """One measured angular lobe."""

    angle: float
    intensity: float
    baseline: float
    snr: float
    fwhm: float
    area: float
    index: int
    coverage: float
    n_pixels: int
    valid: bool = True
    flags: tuple[str, ...] = APPARENT_FLAGS
    reason: str = "accepted"
    refinement: str = "quadratic"

    @property
    def azimuth(self) -> float:
        return self.angle

    @property
    def azimuthal_fwhm(self) -> float:
        return self.fwhm

    @property
    def fwhm_deg(self) -> float:
        """Full width at half maximum in display-friendly degrees."""

        return float(np.degrees(self.fwhm)) if np.isfinite(self.fwhm) else float("nan")

    def keys(self) -> tuple[str, ...]:
        return super().keys() + ("fwhm_deg",)


@dataclass
class RadialProfile(_MappingResult):
    """Radial profile for one *observed* angular sector."""

    angle: float
    q: np.ndarray
    intensity: np.ndarray
    counts: np.ndarray
    candidate_counts: np.ndarray
    coverage: np.ndarray
    q_min: float
    q_max: float
    flags: tuple[str, ...] = APPARENT_FLAGS
    q_unit: str = "unknown"

    @property
    def profile(self) -> np.ndarray:
        return self.intensity


class RidgePoint(_MappingResult, _SharedRidgePoint):
    """A radial peak/ridge measured from one real angular sector.

    No counterpart mirrored from another quadrant is inserted.  ``source``
    records the number of pixels used for this point, and ``valid`` is false
    when the sector did not contain enough finite, unmasked signal.  The
    ``q_unit`` is deliberately fail-closed: direct callers must state a
    physical reciprocal-length unit before spacing fields are derived.
    Measured points receive the declaration propagated from their q map.
    Points from ``azimuthal_peak`` carry an annulus coordinate in ``q``; their
    ``q_star`` and lamellar-spacing aliases remain NaN because an angular
    maximum is not a measured radial reflection position.
    """

    def __init__(
        self,
        angle: float,
        q: float,
        q_star: float | None = None,
        lamellar_spacing: float | None = None,
        intensity: float = float("nan"),
        baseline: float = float("nan"),
        snr: float = float("nan"),
        radial_fwhm: float = float("nan"),
        azimuthal_fwhm: float = float("nan"),
        area: float = float("nan"),
        coverage: float = 0.0,
        n_pixels: int = 0,
        valid: bool = True,
        source: str = "observed",
        flags: tuple[str, ...] = APPARENT_FLAGS,
        method: str = "radial_peak",
        curvature: float = float("nan"),
        normal_slope: float = float("nan"),
        support: float = float("nan"),
        pixel_y: float = float("nan"),
        pixel_x: float = float("nan"),
        accepted: bool | None = None,
        reason: str = "",
        q_unit: str = "unknown",
        score: float = float("nan"),
        continuity_score: float = float("nan"),
        trajectory_id: int | None = None,
        branch_id: int | None = None,
        local_q_step: float = float("nan"),
        q_normal_step: float = float("nan"),
        q_scale_anisotropy: float = float("nan"),
    ) -> None:
        q = float(q)
        angle = float(angle)
        q_unit = str(q_unit or "unknown")
        q_for_coordinates = q if np.isfinite(q) else 0.0
        super().__init__(
            qx=q_for_coordinates * np.cos(angle),
            qy=q_for_coordinates * np.sin(angle),
            intensity=float(intensity) if np.isfinite(intensity) else None,
            weight=1.0,
            metadata={
                "angle": angle,
                "source": source,
                "flags": tuple(flags),
                "method": str(method),
                "q_unit": q_unit,
            },
        )
        self.angle = angle
        self.q = q
        self.q_star = q if q_star is None else float(q_star)
        q_scale = _q_to_nm_inverse_scale(q_unit)
        if lamellar_spacing is None:
            self.lamellar_spacing = (
                float(2.0 * np.pi / (q * q_scale))
                if q_scale is not None and q > 0
                else float("nan")
            )
        elif q_scale is None:
            # Do not retain an apparently physical spacing from an uncalibrated
            # q value merely because a caller supplied a legacy positional
            # value.
            self.lamellar_spacing = float("nan")
        else:
            self.lamellar_spacing = float(lamellar_spacing)
        self.baseline = float(baseline)
        self.snr = float(snr)
        self.radial_fwhm = float(radial_fwhm)
        self.azimuthal_fwhm = float(azimuthal_fwhm)
        self.area = float(area)
        self.coverage = float(coverage)
        self.n_pixels = int(n_pixels)
        self.valid = bool(valid)
        self.source = str(source)
        self.flags = tuple(flags)
        self.method = str(method)
        self.curvature = float(curvature)
        self.normal_slope = float(normal_slope)
        self.support = float(support)
        self.pixel_y = float(pixel_y)
        self.pixel_x = float(pixel_x)
        self.accepted = bool(valid if accepted is None else accepted)
        self.reason = str(reason or ("accepted" if self.accepted else "rejected"))
        self.q_unit = q_unit
        self.score = float(score)
        self.continuity_score = float(continuity_score)
        self.trajectory_id = None if trajectory_id is None else int(trajectory_id)
        self.branch_id = None if branch_id is None else int(branch_id)
        self.local_q_step = float(local_q_step)
        self.q_normal_step = float(q_normal_step)
        self.q_scale_anisotropy = float(q_scale_anisotropy)
        self.flags = _flags_with_q_unit(self.flags, q_unit)
        self.metadata.update(
            flags=self.flags,
            curvature=self.curvature,
            normal_slope=self.normal_slope,
            support=self.support,
            pixel_y=self.pixel_y,
            pixel_x=self.pixel_x,
            accepted=self.accepted,
            reason=self.reason,
            q_unit=self.q_unit,
            score=self.score,
            continuity_score=self.continuity_score,
            trajectory_id=self.trajectory_id,
            branch_id=self.branch_id,
            local_q_step=self.local_q_step,
            q_normal_step=self.q_normal_step,
            q_scale_anisotropy=self.q_scale_anisotropy,
        )

    def keys(self) -> tuple[str, ...]:
        return (
            "angle", "q", "q_star", "lamellar_spacing", "intensity", "baseline", "snr",
            "radial_fwhm", "azimuthal_fwhm", "area", "coverage", "n_pixels", "valid", "source", "flags",
            "qx", "qy", "method", "curvature", "normal_slope", "support",
            "pixel_y", "pixel_x", "accepted", "reason", "q_unit", "q_star_Ainv", "q_star_nm_inv", "Ln", "Ln_nm",
            "score", "point_score", "continuity_score", "trajectory_id", "branch_id", "local_q_step",
            "q_normal_step", "q_scale_anisotropy",
        )

    @property
    def point_score(self) -> float:
        """Explicit alias for the point-level quality score."""

        return self.score

    @property
    def q_star_Ainv(self) -> float:
        """Peak position in inverse angstrom, when q has a physical unit."""

        scale = _q_to_nm_inverse_scale(self.q_unit)
        if scale is None or not np.isfinite(self.q_star):
            return float("nan")
        return float(self.q_star * scale / 10.0)

    @property
    def q_star_nm_inv(self) -> float:
        scale = _q_to_nm_inverse_scale(self.q_unit)
        if scale is None or not np.isfinite(self.q_star):
            return float("nan")
        return float(self.q_star * scale)

    @property
    def Ln(self) -> float:
        """Compatibility alias for :attr:`Ln_nm`."""

        return self.Ln_nm

    @property
    def Ln_nm(self) -> float:
        q_nm_inv = self.q_star_nm_inv
        if not np.isfinite(q_nm_inv) or q_nm_inv <= 0:
            return float("nan")
        return float(2.0 * np.pi / q_nm_inv)

    @property
    def q_position(self) -> float:
        return self.q


@dataclass
class RidgeTrack(_MappingResult):
    """Collection of radial ridge points and quality metadata."""

    points: list[RidgePoint]
    angles: np.ndarray
    q: np.ndarray
    valid: np.ndarray
    coverage: np.ndarray
    flags: tuple[str, ...] = APPARENT_FLAGS
    q_unit: str = "unknown"
    valid_fraction: float = float("nan")
    continuity_fraction: float = float("nan")
    continuity_score: float = float("nan")

    @property
    def observed_points(self) -> list[RidgePoint]:
        return [point for point in self.points if point.valid]


@dataclass
class EllipseGeometry(_MappingResult):
    """One apparent ellipse in q-space.

    ``a``, ``b`` and ``theta`` retain the original observables API.  The
    canonical solver also estimates a shared centre, so the adapter carries
    it without changing the positional constructor used by older notebooks.
    ``theta`` is always radians; :attr:`theta_deg` is the display value.
    """

    a: float
    b: float
    theta: float
    center: tuple[float, float] = (0.0, 0.0)

    @property
    def axes_ratio(self) -> float:
        return float(self.b / self.a) if self.a else float("nan")

    @property
    def ellipticity(self) -> float:
        """Grubb's eccentricity-valued ellipticity."""

        ratio = self.axes_ratio
        return float(np.sqrt(max(0.0, 1.0 - ratio * ratio))) if np.isfinite(ratio) else float("nan")

    @property
    def eccentricity(self) -> float:
        return self.ellipticity

    @property
    def theta_deg(self) -> float:
        return float(np.degrees(self.theta))

    @property
    def cx(self) -> float:
        return float(self.center[0])

    @property
    def cy(self) -> float:
        return float(self.center[1])


@dataclass
class DoubleEllipseFit(_MappingResult):
    """Symmetric pair of apparent ellipses fitted to ridge points."""

    a: float
    b: float
    theta: float
    ellipses: tuple[EllipseGeometry, EllipseGeometry]
    ellipticity: float
    axes_ratio: float
    rmse: float
    rss: float
    n_points: int
    branch_counts: tuple[int, int]
    success: bool
    message: str
    flags: tuple[str, ...] = APPARENT_FLAGS
    center: tuple[float, float] = (float("nan"), float("nan"))
    stderr: dict[str, float] = field(default_factory=dict)
    covariance: np.ndarray | None = None
    condition_number: float = float("nan")
    coverage: Any = None
    bound_flags: dict[str, bool] = field(default_factory=dict)
    bound_status: dict[str, str | None] = field(default_factory=dict)
    free_names: tuple[str, ...] = ()
    parameter_values: dict[str, float] = field(default_factory=dict)
    reference_axis_deg: float = 0.0
    q_unit: str = "unknown"
    Ln_from_minor_axis_nm: float = float("nan")
    Lz_from_draw_axis_nm: float = float("nan")
    branch_assignment: np.ndarray | None = None
    candidate_solutions: tuple[dict[str, Any], ...] = ()
    selected_start_index: int = 0
    multistart_count: int = 1
    quality: dict[str, Any] = field(default_factory=dict)
    symmetry: dict[str, Any] = field(default_factory=dict)
    branch_assignment_indices: np.ndarray | None = None

    @property
    def theta_deg(self) -> float:
        return float(np.degrees(self.theta))

    @property
    def ellipse_axis_tilt_deg(self) -> float:
        """Apparent ellipse-axis tilt relative to ``reference_axis_deg``."""

        return self.theta_deg

    @property
    def eccentricity(self) -> float:
        """Alias for the paper's eccentricity-valued ``ellipticity``."""

        return self.ellipticity

    @property
    def coverage_fraction(self) -> float:
        value = _get_field(self.coverage, ("angular_coverage", "coverage_fraction"), float("nan"))
        return float(value)

    @property
    def phi_app_deg(self) -> float:
        """Placeholder for an independently measured lamellar-tilt angle.

        The ellipse-axis tilt is not the same quantity as the apparent lobe
        angle (and neither is a direct measurement of the structural
        ``alpha``).  Returning NaN makes that distinction explicit for UI and
        export code until a lobe-aware inference step supplies it.
        """

        return float("nan")

    @property
    def alpha_candidate_deg(self) -> float:
        """No structural alpha is inferred by an apparent ellipse fit."""

        return float("nan")

    @property
    def ellipse_plus(self) -> EllipseGeometry:
        return self.ellipses[0]

    @property
    def ellipse_minus(self) -> EllipseGeometry:
        return self.ellipses[1]


@dataclass
class ObservableSet(_MappingResult):
    """High-level bundle used by a GUI or batch pipeline."""

    angular: AngularSpectrum
    lobes: list[LobeMetrics]
    ridge: RidgeTrack
    ellipse: DoubleEllipseFit | None
    phi_app_deg: float = float("nan")
    phi_app_std_deg: float = float("nan")
    draw_axis_deg: float = 90.0
    alpha_candidate_deg: float = float("nan")
    psi_candidate_deg: float = float("nan")
    flags: tuple[str, ...] = APPARENT_FLAGS
    q_unit: str = "unknown"
    # ``phi_app_std_deg`` is retained for compatibility; the estimator is a
    # scaled median absolute deviation, so expose the precise name as well.
    # Keep this new field at the end so legacy positional construction retains
    # its original argument order.
    phi_app_mad_deg: float = float("nan")
    # Azimuthal-peak tracks intentionally do not expose their annulus q as a
    # radial q_star.  These optional, directly observed lobe-sector profiles
    # retain the radial reflection position and width needed by callers that
    # want both measurements in one observable bundle.
    lobe_radial_profiles: list[RadialProfile] = field(default_factory=list)
    lobe_radial_peaks: list[RidgePoint] = field(default_factory=list)

    def __post_init__(self) -> None:
        std = float(self.phi_app_std_deg)
        mad = float(self.phi_app_mad_deg)
        if not np.isfinite(mad) and np.isfinite(std):
            mad = std
        elif not np.isfinite(std) and np.isfinite(mad):
            std = mad
        self.phi_app_std_deg = std
        self.phi_app_mad_deg = mad
        self.q_unit = str(self.q_unit or "unknown")


def _get_field(obj: Any, names: Sequence[str], default: Any = None) -> Any:
    """Read a field from either a mapping, an object, or a zero-arg property."""

    for name in names:
        if isinstance(obj, Mapping) and name in obj:
            value = obj[name]
        elif hasattr(obj, name):
            value = getattr(obj, name)
        else:
            continue
        if callable(value):
            try:
                value = value()
            except TypeError:
                pass
        if value is not None:
            return value
    return default


def _q_unit(qmap: Any) -> str:
    unit = _get_field(qmap, ("q_unit", "unit"), None)
    if unit is None:
        metadata = _get_field(qmap, ("metadata",), {})
        unit = _get_field(metadata, ("q_unit", "unit"), "unknown")
    return str(unit or "unknown")


def _normalized_q_unit(unit: Any) -> str:
    """Normalize spelling only; never infer a missing physical unit."""

    return (
        str(unit or "unknown")
        .strip()
        .casefold()
        .replace(" ", "")
        .replace("⁻¹", "^-1")
        .replace("⁻", "^")
        .replace("−", "-")
        .replace("å", "a")
        .replace("Å", "a")
    )


def _q_to_nm_inverse_scale(unit: Any) -> float | None:
    normalized = _normalized_q_unit(unit)
    if normalized in {"1/nm", "nm^-1", "nm-1", "nm**-1"}:
        return 1.0
    if normalized in {
        "1/a",
        "a^-1",
        "a-1",
        "a**-1",
        "1/angstrom",
        "angstrom^-1",
        "angstrom-1",
        "angstrom**-1",
    }:
        return 10.0
    return None


def _q_unit_spacing_flag(unit: Any) -> str | None:
    """Return a stable diagnostic when spacing cannot be physicalized."""

    if _q_to_nm_inverse_scale(unit) is not None:
        return None
    # Keep the public diagnostic stable for both uncalibrated pixel-q and any
    # other undeclared unit.  The q_unit field itself retains the more specific
    # provenance for callers that need to distinguish them.
    return "spacing_unavailable_unknown_q_unit"


def _flags_with_q_unit(flags: Sequence[str], unit: Any) -> tuple[str, ...]:
    """Attach the spacing diagnostic once to each q-bearing result layer."""

    result = tuple(flags)
    diagnostic = _q_unit_spacing_flag(unit)
    if diagnostic is not None and diagnostic not in result:
        result += (diagnostic,)
    if _normalized_q_unit(unit) in {"pixel-q", "pixel_q", "pixelq"} and "uncalibrated_pixel_q" not in result:
        result += ("uncalibrated_pixel_q",)
    return result


def _array_field(obj: Any, names: Sequence[str], default: Any = None) -> np.ndarray | None:
    value = _get_field(obj, names, default)
    if value is None:
        return None
    try:
        return np.asarray(value)
    except Exception:
        return None


def _extract_maps(frame: Any, qmap: Any, extra_mask: Any = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return flattened intensity, q, angle and validity arrays.

    ``mask=True`` follows detector conventions and means invalid.  A field
    named ``valid_mask`` follows its literal meaning and means valid.  This
    distinction avoids silently inverting the common pyFAI mask while still
    allowing simple test doubles to use a valid mask.
    """

    data = _array_field(frame, ("data", "intensity", "image", "values"))
    if data is None:
        raise ValueError("ImageFrame must expose data/intensity/image/values")
    data = np.asarray(data, dtype=float)
    shape = data.shape

    qx = _array_field(qmap, ("qx", "q_x", "x", "qx_map"))
    qy = _array_field(qmap, ("qy", "q_y", "y", "qy_map"))
    q = _array_field(qmap, ("q", "q_map", "q_abs", "radius"))
    angle = _array_field(qmap, ("angle", "azimuth", "chi", "phi", "azimuth_map"))
    if q is None and qx is not None and qy is not None:
        q = np.hypot(qx, qy)
    if angle is None and qx is not None and qy is not None:
        angle = np.arctan2(qy, qx)
    if q is None:
        raise ValueError("QMap must expose q or qx/qy arrays")
    if angle is None:
        raise ValueError("QMap must expose azimuth/angle or qx/qy arrays")

    q = np.broadcast_to(np.asarray(q, dtype=float), shape)
    angle = np.broadcast_to(np.asarray(angle, dtype=float), shape)
    data = np.broadcast_to(data, shape)
    valid = np.isfinite(data) & np.isfinite(q) & np.isfinite(angle)

    # A detector mask is normally True for bad pixels.
    detector_mask = _array_field(frame, ("mask", "bad_mask", "invalid_mask"))
    if detector_mask is not None:
        valid &= ~np.broadcast_to(np.asarray(detector_mask, dtype=bool), shape)
    qmask = _array_field(qmap, ("mask", "bad_mask", "invalid_mask"))
    if qmask is not None:
        valid &= ~np.broadcast_to(np.asarray(qmask, dtype=bool), shape)
    valid_mask = _array_field(frame, ("valid_mask", "valid"))
    if valid_mask is not None and np.asarray(valid_mask).shape == shape:
        valid &= np.broadcast_to(np.asarray(valid_mask, dtype=bool), shape)
    q_valid_mask = _array_field(qmap, ("valid_mask", "valid"))
    if q_valid_mask is not None and np.asarray(q_valid_mask).shape == shape:
        valid &= np.broadcast_to(np.asarray(q_valid_mask, dtype=bool), shape)
    if extra_mask is not None:
        valid &= ~np.broadcast_to(np.asarray(extra_mask, dtype=bool), shape)

    return data.ravel(), q.ravel(), np.mod(angle.ravel() + np.pi, 2.0 * np.pi) - np.pi, valid.ravel()


def _q_limits(q: np.ndarray, q_window: Any = None, q_range: Any = None) -> tuple[float, float]:
    window = q_window if q_window is not None else q_range
    if window is None:
        finite = q[np.isfinite(q)]
        if not finite.size:
            raise ValueError("q map contains no finite values")
        return float(np.min(finite)), float(np.max(finite))
    if isinstance(window, Mapping):
        lo = _get_field(window, ("min", "q_min", "low", "start"))
        hi = _get_field(window, ("max", "q_max", "high", "stop"))
        if lo is None or hi is None:
            raise ValueError("q window mapping must provide min/max")
    else:
        try:
            lo, hi = window
        except Exception as exc:
            raise ValueError("q window must be a (min, max) pair") from exc
    lo, hi = float(lo), float(hi)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        raise ValueError("q window must contain finite max > min")
    return lo, hi


def _bin_mean(values: np.ndarray, indices: np.ndarray, n_bins: int, statistic: str) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(values) & (indices >= 0) & (indices < n_bins)
    indices = indices[finite]
    values = values[finite]
    counts = np.bincount(indices, minlength=n_bins).astype(int)
    if statistic == "sum":
        profile = np.bincount(indices, weights=values, minlength=n_bins).astype(float)
        return profile, counts
    sums = np.bincount(indices, weights=values, minlength=n_bins).astype(float)
    profile = np.full(n_bins, np.nan, dtype=float)
    np.divide(sums, counts, out=profile, where=counts > 0)
    return profile, counts


def measure_angular_spectrum(
    frame: Any,
    qmap: Any,
    q_window: Any = None,
    *,
    q_range: Any = None,
    n_bins: int = 360,
    statistic: str = "mean",
    mask: Any = None,
    smooth_sigma: float = 0.0,
) -> AngularSpectrum:
    """Measure a masked angular spectrum over a q window.

    The profile is a per-pixel mean by default, so a wedge with fewer pixels
    does not look artificially weak.  Use ``statistic='sum'`` for integrated
    intensity.  No symmetry or mirror completion is applied.
    """

    if int(n_bins) < 8:
        raise ValueError("n_bins must be at least 8")
    if statistic not in {"mean", "sum"}:
        raise ValueError("statistic must be 'mean' or 'sum'")
    values, q, angle, valid = _extract_maps(frame, qmap, mask)
    q_unit = _q_unit(qmap)
    q_min, q_max = _q_limits(q, q_window, q_range)
    candidate = np.isfinite(values) & np.isfinite(q) & (q >= q_min) & (q <= q_max)
    selected = candidate & valid
    n_bins = int(n_bins)
    edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    idx_all = np.digitize(angle[candidate], edges, right=False) - 1
    idx_valid = np.digitize(angle[selected], edges, right=False) - 1
    candidate_counts = np.bincount(np.clip(idx_all, 0, n_bins - 1), minlength=n_bins).astype(int)
    profile, counts = _bin_mean(values[selected], idx_valid, n_bins, statistic)
    if smooth_sigma and gaussian_filter1d is not None:
        finite_profile = np.where(np.isfinite(profile), profile, 0.0)
        normalizer = gaussian_filter1d((counts > 0).astype(float), float(smooth_sigma), mode="wrap")
        smoothed = gaussian_filter1d(finite_profile, float(smooth_sigma), mode="wrap")
        profile = np.divide(smoothed, normalizer, out=np.full_like(smoothed, np.nan), where=normalizer > 1e-9)
    coverage = np.divide(counts, candidate_counts, out=np.zeros(n_bins, dtype=float), where=candidate_counts > 0)
    return AngularSpectrum(
        angle=centres,
        intensity=profile,
        counts=counts,
        candidate_counts=candidate_counts,
        coverage=coverage,
        q_min=q_min,
        q_max=q_max,
        q_center=0.5 * (q_min + q_max),
        statistic=statistic,
        flags=_flags_with_q_unit(APPARENT_FLAGS, q_unit),
        q_unit=q_unit,
    )


angular_spectrum = measure_angular_spectrum
azimuthal_profile = measure_angular_spectrum
extract_azimuthal_spectrum = measure_angular_spectrum


def _robust_noise(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size < 2:
        return float("nan")
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    noise = 1.4826 * mad
    if not np.isfinite(noise) or noise <= np.finfo(float).eps:
        noise = float(np.std(finite, ddof=1))
    return noise


def _wrap_distance(angle: np.ndarray | float, centre: float) -> np.ndarray:
    return np.abs(np.angle(np.exp(1j * (np.asarray(angle) - float(centre)))))


def _quadratic_peak(x: np.ndarray, y: np.ndarray, index: int, period: float | None = None) -> tuple[float, float]:
    """Sub-bin parabolic peak estimate; returns x and y."""

    index = int(index)
    if len(y) < 3 or index <= 0 or index >= len(y) - 1:
        return float(x[index]), float(y[index])
    ym, y0, yp = float(y[index - 1]), float(y[index]), float(y[index + 1])
    denom = ym - 2.0 * y0 + yp
    if not np.isfinite(denom) or abs(denom) <= np.finfo(float).eps:
        return float(x[index]), y0
    offset = 0.5 * (ym - yp) / denom
    offset = float(np.clip(offset, -0.5, 0.5))
    dx = float(x[index] - x[index - 1])
    xp = float(x[index] + offset * dx)
    ypval = float(y0 - 0.25 * (ym - yp) * offset)
    if period is not None:
        xp = float(np.angle(np.exp(1j * xp)))
    return xp, ypval


def _periodic_fwhm(x: np.ndarray, y: np.ndarray, index: int, baseline: float, period: float = 2.0 * np.pi) -> float:
    finite = np.isfinite(y)
    if not np.any(finite) or not np.isfinite(y[index]):
        return float("nan")
    y0 = float(y[index])
    half = baseline + 0.5 * (y0 - baseline)
    n = len(y)
    ext_y = np.tile(np.asarray(y, dtype=float), 3)
    ext_x = np.linspace(x[0] - period, x[-1] + period, 3 * n)
    centre = n + index
    if ext_y[centre] < half:
        return float("nan")
    left = centre
    while left > 0 and np.isfinite(ext_y[left - 1]) and ext_y[left - 1] >= half:
        left -= 1
    right = centre
    while right < len(ext_y) - 1 and np.isfinite(ext_y[right + 1]) and ext_y[right + 1] >= half:
        right += 1
    if left == 0 or right == len(ext_y) - 1:
        return float("nan")
    # Linear crossing at each half-height boundary.
    def crossing(i0: int, i1: int) -> float:
        y0_, y1_ = ext_y[i0], ext_y[i1]
        if y1_ == y0_:
            return float(ext_x[i0])
        return float(ext_x[i0] + (half - y0_) * (ext_x[i1] - ext_x[i0]) / (y1_ - y0_))

    return max(0.0, crossing(right, right + 1) - crossing(left - 1, left))


def _linear_fwhm(x: np.ndarray, y: np.ndarray, index: int, baseline: float) -> float:
    finite = np.isfinite(y)
    if not np.any(finite) or not np.isfinite(y[index]):
        return float("nan")
    half = baseline + 0.5 * (float(y[index]) - baseline)
    if y[index] < half:
        return float("nan")
    left = int(index)
    while left > 0 and np.isfinite(y[left - 1]) and y[left - 1] >= half:
        left -= 1
    right = int(index)
    while right < len(y) - 1 and np.isfinite(y[right + 1]) and y[right + 1] >= half:
        right += 1

    def cross(i0: int, i1: int) -> float:
        if y[i1] == y[i0]:
            return float(x[i0])
        return float(x[i0] + (half - y[i0]) * (x[i1] - x[i0]) / (y[i1] - y[i0]))

    left_x = cross(left - 1, left) if left else float(x[left])
    right_x = cross(right, right + 1) if right < len(y) - 1 else float(x[right])
    return max(0.0, right_x - left_x)


def _refine_symmetric_lobes(
    spectrum: AngularSpectrum | Mapping[str, Any],
    lobes: Sequence[LobeMetrics],
    *,
    reference_axis_deg: float = 0.0,
) -> list[LobeMetrics]:
    """Refine observed candidates with one robust four-envelope angle model.

    The fit estimates a common apparent lobe angle and width from the finite,
    actually observed angular bins.  It never inserts a missing lobe: each
    returned record still corresponds to one detected candidate and retains
    that candidate's coverage and pixel count.
    """

    if least_squares is None or len(lobes) < 2:
        return list(lobes)
    angle = np.asarray(_get_field(spectrum, ("angle", "azimuth")), dtype=float)
    intensity = np.asarray(_get_field(spectrum, ("intensity", "profile")), dtype=float)
    counts = np.asarray(_get_field(spectrum, ("counts",)), dtype=float)
    coverage = np.asarray(_get_field(spectrum, ("coverage",)), dtype=float)
    finite = np.isfinite(angle) & np.isfinite(intensity) & (counts > 0)
    if np.count_nonzero(finite) < 12:
        return list(lobes)

    x = angle[finite]
    y = intensity[finite]
    weights = np.clip(coverage[finite], 0.0, 1.0)
    baseline = float(np.percentile(y, 10.0))
    amplitude = max(float(np.percentile(y, 99.0) - baseline), np.finfo(float).eps)
    scale = max(float(np.std(y)), np.finfo(float).eps)
    reference = np.deg2rad(float(reference_axis_deg))
    acute = [
        abs((float(lobe.angle) - reference + np.pi / 2.0) % np.pi - np.pi / 2.0)
        for lobe in lobes
    ]
    phi0 = float(np.clip(np.median(acute), 0.0, np.pi / 2.0))

    def model(candidate: np.ndarray) -> np.ndarray:
        offset, amplitude_plus, amplitude_minus, phi, sigma = candidate

        def envelope(centre: float) -> np.ndarray:
            delta = np.angle(np.exp(1j * (x - centre)))
            return np.exp(-0.5 * (delta / sigma) ** 2)

        return (
            offset
            + amplitude_plus * (envelope(reference + phi) + envelope(reference + np.pi + phi))
            + amplitude_minus * (envelope(reference - phi) + envelope(reference + np.pi - phi))
        )

    lower = np.asarray(
        [float(np.min(y) - amplitude), 0.0, 0.0, 0.0, np.deg2rad(1.0)],
        dtype=float,
    )
    upper = np.asarray(
        [float(np.max(y)), 10.0 * amplitude, 10.0 * amplitude, np.pi / 2.0, np.deg2rad(40.0)],
        dtype=float,
    )
    initial = np.asarray(
        [baseline, amplitude, amplitude, phi0, np.deg2rad(10.0)],
        dtype=float,
    )
    fit_candidates = []
    for phi_start in (phi0, 0.5 * phi0, min(np.pi / 2.0, 1.5 * phi0)):
        candidate_initial = initial.copy()
        candidate_initial[3] = max(lower[3], min(upper[3], phi_start))
        try:
            candidate_fit = least_squares(
                lambda candidate: (model(candidate) - y) * weights / scale,
                candidate_initial,
                bounds=(lower, upper),
                loss="cauchy",
                f_scale=0.5,
                max_nfev=500,
            )
        except (FloatingPointError, ValueError):
            continue
        if candidate_fit.success and np.all(np.isfinite(candidate_fit.x)):
            fit_candidates.append(candidate_fit)
    if not fit_candidates:
        return list(lobes)
    fit = min(fit_candidates, key=lambda candidate: (float(candidate.cost), tuple(candidate.x)))

    phi = float(fit.x[3])
    sigma = float(fit.x[4])
    centres = np.asarray(
        (
            reference + phi,
            reference - phi,
            reference + np.pi + phi,
            reference + np.pi - phi,
        ),
        dtype=float,
    )
    centres = np.angle(np.exp(1j * centres))
    assignments: dict[int, list[LobeMetrics]] = {}
    for lobe in lobes:
        index = int(np.argmin(_wrap_distance(centres, lobe.angle)))
        assignments.setdefault(index, []).append(lobe)

    overlap = bool(2.0 * phi <= 2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma)
    refined: list[LobeMetrics] = []
    for centre_index, candidates in assignments.items():
        candidates = sorted(candidates, key=lambda item: (item.valid, item.snr), reverse=True)
        for rank, lobe in enumerate(candidates):
            lobe.angle = float(centres[centre_index])
            lobe.fwhm = float(2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma)
            lobe.refinement = "symmetric_cauchy"
            if rank > 0:
                lobe.valid = False
                lobe.reason = "duplicate_symmetric_lobe_candidate"
                lobe.flags = tuple(dict.fromkeys(lobe.flags + (lobe.reason,)))
            elif overlap:
                lobe.valid = False
                lobe.reason = "overlapping_lobes_unresolved"
                lobe.flags = tuple(dict.fromkeys(lobe.flags + (lobe.reason,)))
            elif lobe.valid:
                lobe.reason = "accepted"
            refined.append(lobe)
    return sorted(refined, key=lambda item: item.angle)


def measure_four_lobe_peaks(
    spectrum: AngularSpectrum | Mapping[str, Any],
    *,
    expected: int = 4,
    min_prominence: float | None = None,
    snr_threshold: float = 2.0,
    min_distance_fraction: float = 0.12,
    symmetric_refine: bool = False,
    reference_axis_deg: float = 0.0,
) -> list[LobeMetrics]:
    """Find up to four angular lobes without completing missing quadrants."""

    angle = np.asarray(_get_field(spectrum, ("angle", "azimuth")), dtype=float)
    y = np.asarray(_get_field(spectrum, ("intensity", "profile")), dtype=float)
    counts = np.asarray(_get_field(spectrum, ("counts",)), dtype=float)
    coverage = np.asarray(_get_field(spectrum, ("coverage",)), dtype=float)
    finite = np.isfinite(y)
    if not np.any(finite):
        return []
    baseline = float(np.nanpercentile(y, 10.0))
    noise = _robust_noise(y - baseline)
    if not np.isfinite(noise) or noise <= np.finfo(float).eps:
        noise = max(float(np.nanstd(y)), np.finfo(float).eps)
    prominence = float(min_prominence) if min_prominence is not None else float(snr_threshold * noise)
    # Circular extension means lobes crossing -pi/pi are not discarded.
    if find_peaks is None:  # pragma: no cover
        candidates = np.asarray([int(np.nanargmax(y))])
    else:
        n = len(y)
        extended = np.tile(np.where(finite, y, baseline), 3)
        distance = max(1, int(round(n * float(min_distance_fraction))))
        candidates, props = find_peaks(extended, prominence=prominence, distance=distance)
        candidates = candidates[(candidates >= n) & (candidates < 2 * n)] - n
        # Deduplicate modulo n if a very broad peak is reported twice.
        unique: list[int] = []
        for candidate in sorted(candidates.tolist(), key=lambda i: float(y[i]), reverse=True):
            if all(_wrap_distance(angle[candidate], angle[i]) > (2.0 * np.pi / n) for i in unique):
                unique.append(int(candidate))
        candidates = np.asarray(unique, dtype=int)
    candidates = candidates[np.argsort(y[candidates])[::-1]] if len(candidates) else candidates
    candidates = candidates[: int(expected)]
    out: list[LobeMetrics] = []
    n = len(y)
    for index in candidates:
        index = int(index)
        peak_angle, peak_intensity = _quadratic_peak(angle, y, index, period=2.0 * np.pi)
        snr = (peak_intensity - baseline) / noise if noise else float("inf")
        fwhm = _periodic_fwhm(angle, y, index, baseline)
        # Integrate the positive lobe area over one angular-bin period.
        half = baseline + 0.5 * (peak_intensity - baseline)
        dist = _wrap_distance(angle, peak_angle)
        selected = np.isfinite(y) & (y >= half) & (dist <= max(fwhm / 2.0 if np.isfinite(fwhm) else np.inf, np.pi / n))
        if np.count_nonzero(selected) > 1:
            relative = np.angle(np.exp(1j * (angle[selected] - peak_angle)))
            order = np.argsort(relative)
            area = float(
                np.trapezoid(
                    np.maximum(y[selected][order] - baseline, 0.0),
                    relative[order],
                )
            )
        else:
            area = 0.0
        is_valid = bool(np.isfinite(snr) and snr >= snr_threshold)
        out.append(
            LobeMetrics(
                angle=float(peak_angle),
                intensity=float(peak_intensity),
                baseline=baseline,
                snr=float(snr),
                fwhm=float(fwhm),
                area=area,
                index=index,
                coverage=float(coverage[index]) if index < len(coverage) else 0.0,
                n_pixels=int(counts[index]) if index < len(counts) else 0,
                valid=is_valid,
                reason="accepted" if is_valid else "low_snr",
            )
        )
    out = sorted(out, key=lambda item: item.angle)
    return (
        _refine_symmetric_lobes(
            spectrum,
            out,
            reference_axis_deg=reference_axis_deg,
        )
        if symmetric_refine and int(expected) == 4
        else out
    )


find_four_lobe_peaks = measure_four_lobe_peaks
measure_four_peaks = measure_four_lobe_peaks


def apparent_lamellar_tilt(
    lobes: Sequence[LobeMetrics | Mapping[str, Any]],
    *,
    draw_axis_deg: float = 90.0,
) -> tuple[float, float]:
    """Return the directly observed lobe tilt ``phi`` and its robust spread.

    A SAXS reflection and its Friedel counterpart describe the same unoriented
    line, so every lobe angle is reduced modulo 180 degrees before measuring
    its absolute deviation from the draw axis.  In the ideal four-point case
    this equals half the peak-to-peak separation across the meridian described
    by Grubb et al.; no ellipse-axis angle or structural ``alpha`` is used.
    """

    draw_axis = np.deg2rad(float(draw_axis_deg))
    values: list[float] = []
    for lobe in lobes:
        if _get_field(lobe, ("valid",), True) is False:
            continue
        angle = _get_field(lobe, ("angle", "azimuth"))
        if angle is None or not np.isfinite(float(angle)):
            continue
        # Wrap an unoriented line difference onto [-pi/2, pi/2).
        delta = (float(angle) - draw_axis + np.pi / 2.0) % np.pi - np.pi / 2.0
        values.append(abs(delta))
    if not values:
        return float("nan"), float("nan")
    array = np.asarray(values, dtype=float)
    centre = float(np.median(array))
    mad = float(np.median(np.abs(array - centre)))
    return float(np.degrees(centre)), float(np.degrees(1.4826 * mad))


def measure_radial_profile(
    frame: Any,
    qmap: Any,
    angle: float,
    q_window: Any = None,
    *,
    q_range: Any = None,
    n_bins: int = 256,
    sector_width: float = np.deg2rad(4.0),
    statistic: str = "mean",
    mask: Any = None,
) -> RadialProfile:
    """Measure a radial profile in one observed angular sector."""

    if int(n_bins) < 8:
        raise ValueError("n_bins must be at least 8")
    values, q, phi, valid = _extract_maps(frame, qmap, mask)
    q_unit = _q_unit(qmap)
    q_min, q_max = _q_limits(q, q_window, q_range)
    candidate = np.isfinite(values) & np.isfinite(q) & (q >= q_min) & (q <= q_max) & (_wrap_distance(phi, angle) <= float(sector_width) / 2.0)
    selected = candidate & valid
    edges = np.linspace(q_min, q_max, int(n_bins) + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    all_idx = np.digitize(q[candidate], edges, right=False) - 1
    valid_idx = np.digitize(q[selected], edges, right=False) - 1
    candidate_counts = np.bincount(np.clip(all_idx, 0, int(n_bins) - 1), minlength=int(n_bins)).astype(int)
    profile, counts = _bin_mean(values[selected], valid_idx, int(n_bins), statistic)
    coverage = np.divide(counts, candidate_counts, out=np.zeros_like(profile), where=candidate_counts > 0)
    return RadialProfile(
        angle=float(angle),
        q=centres,
        intensity=profile,
        counts=counts,
        candidate_counts=candidate_counts,
        coverage=coverage,
        q_min=q_min,
        q_max=q_max,
        flags=_flags_with_q_unit(APPARENT_FLAGS, q_unit),
        q_unit=q_unit,
    )


radial_profile = measure_radial_profile


def _radial_profile_baseline_noise(
    q: np.ndarray,
    intensity: np.ndarray,
    finite: np.ndarray,
) -> tuple[float, float]:
    """Estimate baseline and high-frequency profile noise consistently.

    The MAD of all profile values is inflated by a broad physical radial
    peak, which made the same peak's SNR depend on how much of the feature was
    included in the q window.  Adjacent differences on genuinely observed
    neighbouring bins estimate the local high-frequency noise instead; a
    smoothed detrend and finally the profile standard deviation are defensive
    fallbacks for sparse/constant sectors.  Both the continuity candidate and
    single-sector fallback use this exact estimator.
    """

    del q
    values = np.asarray(intensity, dtype=float)
    finite = np.asarray(finite, dtype=bool)
    observed = values[finite]
    if observed.size == 0:
        return float("nan"), float("nan")
    baseline = float(np.nanpercentile(observed, 10.0))
    noise = float("nan")
    adjacent = finite[:-1] & finite[1:]
    if np.any(adjacent):
        differences = values[1:][adjacent] - values[:-1][adjacent]
        noise = _robust_noise(differences) / np.sqrt(2.0)
    if not np.isfinite(noise) or noise <= np.finfo(float).eps:
        if gaussian_filter1d is not None and observed.size > 8:
            filled = np.interp(np.arange(values.size), np.flatnonzero(finite), observed)
            trend = gaussian_filter1d(filled, 2.0, mode="nearest")
            noise = _robust_noise((values - trend)[finite])
    if not np.isfinite(noise) or noise <= np.finfo(float).eps:
        noise = float(np.nanstd(observed - baseline))
    if not np.isfinite(noise) or noise <= np.finfo(float).eps:
        noise = max(float(np.finfo(float).eps * max(1.0, np.nanmax(np.abs(observed)))), np.finfo(float).eps)
    return baseline, noise


def _radial_peak(
    profile: RadialProfile,
    snr_threshold: float = 2.0,
    min_coverage: float = 0.0,
) -> RidgePoint:
    q = np.asarray(profile.q, dtype=float)
    y = np.asarray(profile.intensity, dtype=float)
    finite = np.isfinite(y) & (np.asarray(profile.counts) > 0)
    if np.count_nonzero(finite) < 3:
        reason = "no_radial_candidate" if not np.any(finite) else "low_coverage"
        flags = list(_flags_with_q_unit(APPARENT_FLAGS, profile.q_unit))
        flags.append(reason)
        return RidgePoint(
            angle=profile.angle,
            q=float("nan"),
            q_star=float("nan"),
            lamellar_spacing=float("nan"),
            intensity=float("nan"),
            baseline=float("nan"),
            snr=float("nan"),
            radial_fwhm=float("nan"),
            azimuthal_fwhm=float("nan"),
            area=float("nan"),
            coverage=float(np.nanmean(profile.coverage)) if profile.coverage.size else 0.0,
            n_pixels=int(np.sum(profile.counts)),
            valid=False,
            flags=tuple(flags),
            score=0.0,
            reason=reason,
            q_unit=profile.q_unit,
        )
    global_coverage = float(np.sum(profile.counts) / max(1, np.sum(profile.candidate_counts)))
    baseline, noise = _radial_profile_baseline_noise(q, y, finite)
    smooth = y.copy()
    if gaussian_filter1d is not None and np.count_nonzero(finite) > 8:
        fill = np.interp(q, q[finite], y[finite])
        smooth = gaussian_filter1d(fill, 1.0, mode="nearest")
    # Interpolation is useful for smoothing across small gaps, but an empty
    # bin must never win the peak search.  Require measured support on both
    # sides before accepting a q location; if the strongest apparent feature
    # sits at a mask edge, the next supported maximum is considered instead.
    supported = finite.copy()
    if supported.size:
        supported[0] = False
        supported[-1] = False
    # Sparse detector sectors (for example small synthetic/pixel-q fixtures)
    # legitimately contain isolated sampled bins.  In that regime requiring
    # two finite neighbours would erase the real peak.  Dense profiles retain
    # the stricter two-sided support guard that rejects interpolation across a
    # masked q gap.
    dense_profile = np.count_nonzero(finite) >= max(12, int(np.ceil(0.5 * len(finite))))
    if supported.size > 2 and dense_profile:
        supported[1:-1] &= finite[:-2] & finite[2:]
    if not np.any(supported):
        flags = list(_flags_with_q_unit(APPARENT_FLAGS, profile.q_unit))
        flags.append("unsupported_radial_boundary")
        return RidgePoint(
            angle=profile.angle,
            q=float("nan"),
            q_star=float("nan"),
            lamellar_spacing=float("nan"),
            intensity=float("nan"),
            baseline=baseline,
            snr=float("nan"),
            radial_fwhm=float("nan"),
            azimuthal_fwhm=float("nan"),
            area=0.0,
            coverage=global_coverage,
            n_pixels=int(np.sum(profile.counts)),
            valid=False,
            flags=tuple(flags),
            score=0.0,
            reason="unsupported_radial_boundary",
            q_unit=profile.q_unit,
        )
    index = int(np.nanargmax(np.where(supported, smooth, np.nan)))
    peak_q, peak_intensity = _quadratic_peak(q, smooth, index)
    snr = (peak_intensity - baseline) / noise if noise else float("inf")
    local_indices = [index]
    for neighbour in (index - 1, index + 1):
        if 0 <= neighbour < len(y) and finite[neighbour]:
            local_indices.append(neighbour)
    local_coverage = float(
        np.nanmin(np.asarray(profile.coverage, dtype=float)[local_indices])
    )
    if local_coverage < float(min_coverage):
        flags = list(_flags_with_q_unit(APPARENT_FLAGS, profile.q_unit))
        flags.append("low_peak_support")
        point = RidgePoint(
            angle=profile.angle,
            q=float("nan"),
            q_star=float("nan"),
            lamellar_spacing=float("nan"),
            intensity=float(peak_intensity),
            baseline=baseline,
            snr=float(snr),
            radial_fwhm=float("nan"),
            azimuthal_fwhm=float("nan"),
            area=float("nan"),
            coverage=local_coverage,
            n_pixels=int(np.sum(profile.counts[local_indices])),
            valid=False,
            flags=tuple(flags),
            score=0.0,
            reason="low_peak_support",
            q_unit=profile.q_unit,
        )
        point.metadata.update(
            profile_global_coverage=global_coverage,
            peak_coverage=local_coverage,
            peak_bin_index=int(index),
        )
        return point
    fwhm = _linear_fwhm(q, smooth, index, baseline)
    area = float(np.trapezoid(np.maximum(y[finite] - baseline, 0.0), q[finite])) if np.count_nonzero(finite) > 1 else 0.0
    flags = list(_flags_with_q_unit(APPARENT_FLAGS, profile.q_unit))
    valid = bool(np.isfinite(snr) and snr >= snr_threshold and np.isfinite(peak_q))
    if not valid:
        flags.append("low_snr")
    reason = "accepted" if valid else "low_snr"
    # A low-SNR maximum is a diagnostic candidate, not an accepted radial
    # reflection.  Keep its intensity/SNR for audit but suppress q_star and
    # spacing aliases so downstream exports cannot mistake it for measured
    # structure.
    q_star = float(peak_q) if valid else float("nan")
    score = float(max(0.0, snr) * max(0.0, min(1.0, local_coverage))) if np.isfinite(snr) and valid else 0.0
    point = RidgePoint(
        angle=profile.angle,
        q=q_star,
        q_star=q_star,
        lamellar_spacing=None,
        intensity=float(peak_intensity),
        baseline=baseline,
        snr=float(snr),
        radial_fwhm=float(fwhm),
        azimuthal_fwhm=float("nan"),
        area=area,
        coverage=local_coverage,
        n_pixels=int(np.sum(profile.counts)),
        valid=valid,
        flags=tuple(flags),
        score=score,
        reason=reason,
        q_unit=profile.q_unit,
    )
    point.metadata.update(
        profile_global_coverage=global_coverage,
        peak_coverage=local_coverage,
        peak_bin_index=int(index),
    )
    return point


def _radial_noise_floor(
    frame: Any,
    qmap: Any,
    q_window: Any,
    *,
    q_range: Any = None,
    mask: Any = None,
) -> float:
    """Estimate one detector-domain noise scale for continuity candidates."""

    values, q, _angle, valid = _extract_maps(frame, qmap, mask)
    q_min, q_max = _q_limits(q, q_window, q_range)
    selected = valid & np.isfinite(values) & (q >= q_min) & (q <= q_max)
    observed = values[selected]
    if observed.size < 8:
        return float("nan")
    median = float(np.median(observed))
    noise = float(np.percentile(observed, 84.0) - median)
    if not np.isfinite(noise) or noise <= np.finfo(float).eps:
        scale = max(float(np.max(np.abs(observed))), 1.0)
        noise = float(np.finfo(float).eps * scale)
    return noise


def _radial_peak_candidates(
    profile: RadialProfile,
    *,
    noise_floor: float,
    top_n: int = 6,
    minimum_score: float = 2.0,
    min_coverage: float = 0.0,
) -> list[RidgePoint]:
    """Return observed local radial peaks eligible for continuity tracking."""

    if find_peaks is None or peak_prominences is None or gaussian_filter1d is None:
        return []
    q = np.asarray(profile.q, dtype=float)
    y = np.asarray(profile.intensity, dtype=float)
    counts = np.asarray(profile.counts)
    finite = np.isfinite(y) & (counts > 0)
    if np.count_nonzero(finite) < 3:
        return []
    fill = np.interp(q, q[finite], y[finite])
    smooth = gaussian_filter1d(fill, 1.0, mode="nearest")
    baseline, profile_noise = _radial_profile_baseline_noise(q, y, finite)
    noise = profile_noise if np.isfinite(profile_noise) and profile_noise > 0.0 else float(noise_floor)
    if not np.isfinite(noise) or noise <= 0.0:
        return []
    peak_indices, _ = find_peaks(smooth, distance=4)
    if peak_indices.size == 0:
        return []
    prominences = peak_prominences(smooth, peak_indices)[0]
    order = np.argsort(prominences, kind="mergesort")[::-1][: int(top_n)]
    global_coverage = float(np.sum(profile.counts) / max(1, np.sum(profile.candidate_counts)))
    area = (
        float(np.trapezoid(np.maximum(y[finite] - baseline, 0.0), q[finite]))
        if np.count_nonzero(finite) > 1
        else 0.0
    )
    candidates: list[RidgePoint] = []
    for position in order:
        index = int(peak_indices[int(position)])
        # Interpolation over a detector gap can manufacture a local maximum at
        # the edge of the observed support.  Such a candidate has no measured
        # two-sided peak shape and must not enter continuity tracking.
        dense_profile = np.count_nonzero(finite) >= max(12, int(np.ceil(0.5 * len(finite))))
        if (
            index <= 0
            or index >= len(y) - 1
            or not finite[index - 1]
            or not finite[index + 1]
        ) and dense_profile:
            continue
        score = float(prominences[int(position)] / noise)
        if not np.isfinite(score) or score < float(minimum_score):
            continue
        peak_q, peak_intensity = _quadratic_peak(q, smooth, index)
        fwhm = _linear_fwhm(q, smooth, index, baseline)
        local_indices = [index]
        for neighbour in (index - 1, index + 1):
            if 0 <= neighbour < len(y) and finite[neighbour]:
                local_indices.append(neighbour)
        local_coverage = float(
            np.nanmin(np.asarray(profile.coverage, dtype=float)[local_indices])
        )
        if local_coverage < float(min_coverage):
            continue
        candidates.append(
            RidgePoint(
                angle=profile.angle,
                q=peak_q,
                q_star=peak_q,
                intensity=peak_intensity,
                baseline=baseline,
                snr=score,
                radial_fwhm=fwhm,
                area=area,
                coverage=local_coverage,
                n_pixels=int(np.sum(profile.counts[local_indices])),
                valid=True,
                flags=_flags_with_q_unit(
                    APPARENT_FLAGS + ("radial_continuity_candidate",),
                    profile.q_unit,
                ),
                score=score,
                reason="accepted",
                q_unit=profile.q_unit,
            )
        )
        candidates[-1].metadata.update(
            profile_global_coverage=global_coverage,
            peak_coverage=local_coverage,
            peak_bin_index=int(index),
        )
    return candidates


def _reject_ridge_point(point: RidgePoint, reason: str) -> RidgePoint:
    point.valid = False
    point.accepted = False
    point.reason = str(reason)
    point.flags = tuple(dict.fromkeys(point.flags + (point.reason,)))
    point.metadata.update(
        flags=point.flags,
        accepted=False,
        reason=point.reason,
    )
    return point


def _continuous_radial_points(
    profiles: Sequence[RadialProfile],
    *,
    noise_floor: float,
    snr_threshold: float = 2.0,
    minimum_track_support: int = 3,
    min_coverage: float = 0.0,
) -> list[RidgePoint]:
    """Select one point per angle from real multi-peak continuity tracks."""

    candidates_by_sector = [
        _radial_peak_candidates(
            profile,
            noise_floor=noise_floor,
            minimum_score=float(snr_threshold),
            min_coverage=min_coverage,
        )
        for profile in profiles
    ]
    nodes: list[tuple[int, RidgePoint]] = [
        (sector, point)
        for sector, candidates in enumerate(candidates_by_sector)
        for point in candidates
    ]
    if not nodes:
        return [
            _reject_ridge_point(
                _radial_peak(
                    profile,
                    snr_threshold=snr_threshold,
                    min_coverage=min_coverage,
                ),
                "no_continuous_candidate",
            )
            for profile in profiles
        ]

    parent = list(range(len(nodes)))
    lookup: dict[int, list[int]] = {}
    for node_index, (sector, _point) in enumerate(nodes):
        lookup.setdefault(sector, []).append(node_index)

    def find(node_index: int) -> int:
        while parent[node_index] != node_index:
            parent[node_index] = parent[parent[node_index]]
            node_index = parent[node_index]
        return node_index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    n_sectors = len(profiles)
    q_step = float(
        np.median(
            [
                np.median(np.diff(profile.q))
                for profile in profiles
                if len(profile.q) > 1
            ]
        )
    )
    for sector in range(n_sectors):
        for left in lookup.get(sector, []):
            for right in lookup.get((sector + 1) % n_sectors, []):
                left_point, right_point = nodes[left][1], nodes[right][1]
                widths = [
                    value
                    for value in (left_point.radial_fwhm, right_point.radial_fwhm)
                    if np.isfinite(value) and value > 0.0
                ]
                width_scale = max(widths) if widths else 0.0
                maximum_jump = max(6.0 * q_step, 0.75 * width_scale)
                if abs(left_point.q - right_point.q) <= maximum_jump:
                    union(left, right)

    components: dict[int, list[int]] = {}
    for node_index in range(len(nodes)):
        components.setdefault(find(node_index), []).append(node_index)
    support = {
        root: len({nodes[node_index][0] for node_index in component})
        for root, component in components.items()
    }

    selected: list[RidgePoint] = []
    for sector, profile in enumerate(profiles):
        eligible = [
            node_index
            for node_index in lookup.get(sector, [])
            if support.get(find(node_index), 0) >= int(minimum_track_support)
        ]
        if eligible:
            chosen = max(
                eligible,
                key=lambda node_index: (
                    support[find(node_index)],
                    nodes[node_index][1].score,
                    -nodes[node_index][1].q,
                ),
            )
            selected.append(nodes[chosen][1])
            continue
        candidates = candidates_by_sector[sector]
        if candidates:
            selected.append(
                _reject_ridge_point(candidates[0], "short_disconnected_track")
            )
        else:
            selected.append(
                _reject_ridge_point(
                    _radial_peak(
                        profile,
                        snr_threshold=snr_threshold,
                        min_coverage=min_coverage,
                    ),
                    "no_continuous_candidate",
                )
            )
    return selected


@dataclass
class _CurvatureField:
    """Detector-grid principal-curvature diagnostics used for ridge tracing."""

    image: np.ndarray
    smooth: np.ndarray
    q: np.ndarray
    angle: np.ndarray
    qx: np.ndarray
    qy: np.ndarray
    valid: np.ndarray
    candidate_domain: np.ndarray
    ridge_candidate: np.ndarray
    score: np.ndarray
    lambda_min: np.ndarray
    normal_slope: np.ndarray
    normal_y: np.ndarray
    normal_x: np.ndarray
    offset: np.ndarray
    support: np.ndarray
    q_normal_step: np.ndarray
    q_scale_anisotropy: np.ndarray
    baseline: float
    noise: float
    q_unit: str = "unknown"


def _surface_curvature_field(
    frame: Any,
    qmap: Any,
    q_window: Any,
    *,
    q_range: Any = None,
    mask: Any = None,
    smooth_sigma: float = 2.0,
    curvature_percentile: float = 25.0,
    normal_step: float = 1.0,
) -> _CurvatureField:
    """Calculate a strongly smoothed, mask-normalized 2-D ridge field.

    This follows the topographic construction used by Grubb et al. (2021):
    the ridge normal is the eigenvector of the most negative Hessian
    eigenvalue, and the subpixel ridge position is the zero of the slope in
    that direction.  Derivatives are evaluated on the detector pixel grid;
    accepted points are converted to q coordinates only after localization.
    """

    if gaussian_filter is None or map_coordinates is None:  # pragma: no cover
        raise RuntimeError("surface_curvature requires scipy.ndimage")
    sigma = float(smooth_sigma)
    step = float(normal_step)
    percentile = float(curvature_percentile)
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("curvature_sigma must be finite and positive")
    if not np.isfinite(step) or step <= 0 or step > 2.0:
        raise ValueError("curvature_normal_step must be in (0, 2]")
    if not np.isfinite(percentile) or not 0 <= percentile <= 100:
        raise ValueError("curvature_percentile must be in [0, 100]")

    raw = _array_field(frame, ("data", "intensity", "image", "values"))
    if raw is None:
        raise ValueError("ImageFrame must expose data/intensity/image/values")
    image = np.asarray(raw, dtype=float)
    if image.ndim != 2:
        raise ValueError("surface_curvature requires a two-dimensional image")
    values, q_flat, angle_flat, valid_flat = _extract_maps(frame, qmap, mask)
    q_min, q_max = _q_limits(q_flat, q_window, q_range)
    q = q_flat.reshape(image.shape)
    angle = angle_flat.reshape(image.shape)
    valid = valid_flat.reshape(image.shape)
    candidate_domain = (
        np.isfinite(image)
        & np.isfinite(q)
        & np.isfinite(angle)
        & (q >= q_min)
        & (q <= q_max)
    )
    valid_domain = candidate_domain & valid
    if not np.any(valid_domain):
        raise ValueError("q window and mask leave no finite pixels for curvature ridge extraction")

    # Normalized convolution prevents masked pixels from being interpreted as
    # zero intensity.  The support field is retained so mask edges cannot be
    # mistaken for physical ridges after smoothing.
    weights = valid_domain.astype(float)
    support = gaussian_filter(weights, sigma=sigma, mode="nearest")
    numerator = gaussian_filter(np.where(valid_domain, image, 0.0), sigma=sigma, mode="nearest")
    smooth = np.divide(
        numerator,
        support,
        out=np.full_like(numerator, np.nan, dtype=float),
        where=support > 1e-8,
    )
    finite_smooth = np.isfinite(smooth) & (support > 1e-8)
    fill = float(np.nanmedian(smooth[finite_smooth])) if np.any(finite_smooth) else 0.0
    derivative_surface = np.where(finite_smooth, smooth, fill)

    grad_y, grad_x = np.gradient(derivative_surface)
    hyy, hyx = np.gradient(grad_y)
    hxy, hxx = np.gradient(grad_x)
    hxy = 0.5 * (hxy + hyx)
    trace = hxx + hyy
    discriminant = np.sqrt(np.maximum((hxx - hyy) ** 2 + 4.0 * hxy**2, 0.0))
    lambda_min = 0.5 * (trace - discriminant)

    # Half-angle gives the largest-eigenvalue direction; its perpendicular is
    # the minimum-eigenvalue direction.  Eigenvector sign is irrelevant to the
    # zero-slope and subpixel equations.
    tangent_angle = 0.5 * np.arctan2(2.0 * hxy, hxx - hyy)
    normal_x = -np.sin(tangent_angle)
    normal_y = np.cos(tangent_angle)
    normal_slope = grad_x * normal_x + grad_y * normal_y

    # Derivatives above intentionally remain on the detector pixel grid.  The
    # local q-map Jacobian below records how one pixel step maps into q space,
    # so curvature results can be compared across detector geometry/scales
    # without pretending that pixel curvature already has reciprocal units.
    qx_grid = q * np.cos(angle)
    qy_grid = q * np.sin(angle)
    dqx_dy, dqx_dx = np.gradient(qx_grid)
    dqy_dy, dqy_dx = np.gradient(qy_grid)
    normal_qx = dqx_dx * normal_x + dqx_dy * normal_y
    normal_qy = dqy_dx * normal_x + dqy_dy * normal_y
    tangent_x = -normal_y
    tangent_y = normal_x
    tangent_qx = dqx_dx * tangent_x + dqx_dy * tangent_y
    tangent_qy = dqy_dx * tangent_x + dqy_dy * tangent_y
    q_normal_step = np.hypot(normal_qx, normal_qy)
    q_tangent_step = np.hypot(tangent_qx, tangent_qy)
    q_scale_min = np.minimum(q_normal_step, q_tangent_step)
    q_scale_max = np.maximum(q_normal_step, q_tangent_step)
    q_scale_anisotropy = np.divide(
        q_scale_max,
        q_scale_min,
        out=np.full_like(q_scale_max, np.nan),
        where=q_scale_min > np.finfo(float).eps,
    )

    yy, xx = np.indices(image.shape, dtype=float)
    minus = map_coordinates(
        derivative_surface,
        [yy - step * normal_y, xx - step * normal_x],
        order=1,
        mode="nearest",
        prefilter=False,
    )
    plus = map_coordinates(
        derivative_surface,
        [yy + step * normal_y, xx + step * normal_x],
        order=1,
        mode="nearest",
        prefilter=False,
    )
    second_along_normal = (minus - 2.0 * derivative_surface + plus) / (step * step)
    with np.errstate(divide="ignore", invalid="ignore"):
        offset = 0.5 * step * (minus - plus) / (minus - 2.0 * derivative_surface + plus)

    # Strong smoothing creates a partial-support halo around masks.  Require
    # nearly complete support and one-pixel distance from the image boundary.
    edge_ok = np.ones(image.shape, dtype=bool)
    margin = max(1, int(np.ceil(step)))
    edge_ok[:margin, :] = edge_ok[-margin:, :] = False
    edge_ok[:, :margin] = edge_ok[:, -margin:] = False
    support_ok = support >= 0.90
    concave_down = (lambda_min < 0.0) & (second_along_normal < 0.0)
    subpixel_ok = np.isfinite(offset) & (np.abs(offset) <= 0.75 * step)
    strength = np.maximum(-lambda_min, 0.0)
    dynamic = float(np.nanmax(derivative_surface[valid_domain]) - np.nanmin(derivative_surface[valid_domain]))
    curvature_floor = max(np.finfo(float).eps, 1e-6 * max(dynamic, np.finfo(float).eps))
    usable_strength = strength[valid_domain & support_ok & concave_down & subpixel_ok]
    adaptive_floor = (
        float(np.nanpercentile(usable_strength, percentile))
        if usable_strength.size
        else float("inf")
    )
    threshold = max(curvature_floor, adaptive_floor)

    baseline = float(np.nanpercentile(derivative_surface[valid_domain], 10.0))
    noise = _robust_noise(derivative_surface[valid_domain] - baseline)
    if not np.isfinite(noise) or noise <= np.finfo(float).eps:
        noise = max(float(np.nanstd(derivative_surface[valid_domain])), np.finfo(float).eps)
    signal = np.maximum(derivative_surface - baseline, 0.0)
    score = strength * signal / max(noise, np.finfo(float).eps)
    ridge_candidate = (
        valid_domain
        & support_ok
        & edge_ok
        & concave_down
        & subpixel_ok
        & (strength >= threshold)
        & np.isfinite(score)
    )

    return _CurvatureField(
        image=image,
        smooth=derivative_surface,
        q=q,
        angle=angle,
        qx=qx_grid,
        qy=qy_grid,
        valid=valid,
        candidate_domain=candidate_domain,
        ridge_candidate=ridge_candidate,
        score=score,
        lambda_min=lambda_min,
        normal_slope=normal_slope,
        normal_y=normal_y,
        normal_x=normal_x,
        offset=offset,
        support=support,
        q_normal_step=q_normal_step,
        q_scale_anisotropy=q_scale_anisotropy,
        baseline=baseline,
        noise=noise,
        q_unit=_q_unit(qmap),
    )


def _curvature_point_for_sector(
    field: _CurvatureField,
    requested_angle: float,
    sector_width: float,
    snr_threshold: float,
    min_coverage: float = 0.0,
) -> RidgePoint:
    """Select and refine the strongest principal-curvature ridge in a sector."""

    sector = _wrap_distance(field.angle, requested_angle) <= float(sector_width) / 2.0
    candidate_pixels = field.candidate_domain & sector
    valid_pixels = candidate_pixels & field.valid
    coverage = float(np.count_nonzero(valid_pixels) / max(1, np.count_nonzero(candidate_pixels)))
    accepted = field.ridge_candidate & sector
    if coverage < float(min_coverage):
        reason = "low_coverage"
        flags = list(APPARENT_FLAGS + ("detector_pixel_principal_curvature", reason))
        return RidgePoint(
            angle=float(requested_angle),
            q=float("nan"),
            q_star=float("nan"),
            lamellar_spacing=float("nan"),
            coverage=coverage,
            n_pixels=int(np.count_nonzero(valid_pixels)),
            valid=False,
            flags=tuple(flags),
            method="surface_curvature",
            accepted=False,
            reason=reason,
            score=0.0,
            q_unit=field.q_unit,
        )
    if not np.any(accepted):
        reason = "low_coverage" if coverage < 0.5 else "no_curvature_candidate"
        flags = list(APPARENT_FLAGS + ("detector_pixel_principal_curvature", reason))
        return RidgePoint(
            angle=float(requested_angle),
            q=float("nan"),
            q_star=float("nan"),
            lamellar_spacing=float("nan"),
            coverage=coverage,
            n_pixels=int(np.count_nonzero(valid_pixels)),
            valid=False,
            flags=tuple(flags),
            method="surface_curvature",
            accepted=False,
            reason=reason,
            score=0.0,
            q_unit=field.q_unit,
        )

    flat_candidates = np.flatnonzero(accepted)
    chosen_flat = flat_candidates[int(np.argmax(field.score.flat[flat_candidates]))]
    py, px = np.unravel_index(chosen_flat, field.image.shape)
    delta = float(field.offset[py, px])
    sub_y = float(py + delta * field.normal_y[py, px])
    sub_x = float(px + delta * field.normal_x[py, px])
    coordinates = np.asarray([[sub_y], [sub_x]], dtype=float)
    qx = float(map_coordinates(field.qx, coordinates, order=1, mode="nearest", prefilter=False)[0])
    qy = float(map_coordinates(field.qy, coordinates, order=1, mode="nearest", prefilter=False)[0])
    q_star = float(np.hypot(qx, qy))
    actual_angle = float(np.arctan2(qy, qx))
    peak_intensity = float(map_coordinates(field.smooth, coordinates, order=1, mode="nearest", prefilter=False)[0])
    snr = float((peak_intensity - field.baseline) / max(field.noise, np.finfo(float).eps))
    is_valid = bool(np.isfinite(q_star) and q_star > 0 and np.isfinite(snr) and snr >= snr_threshold)
    reason = "accepted" if is_valid else "low_snr"
    flags = APPARENT_FLAGS + ("detector_pixel_principal_curvature",)
    if not is_valid:
        flags += ("low_snr",)
    return RidgePoint(
        angle=actual_angle,
        q=q_star,
        q_star=q_star,
        lamellar_spacing=None,
        intensity=peak_intensity,
        baseline=field.baseline,
        snr=snr,
        radial_fwhm=float("nan"),
        azimuthal_fwhm=float("nan"),
        area=float("nan"),
        coverage=coverage,
        n_pixels=int(np.count_nonzero(valid_pixels)),
        valid=is_valid,
        flags=flags,
        method="surface_curvature",
        curvature=float(field.lambda_min[py, px]),
        normal_slope=float(field.normal_slope[py, px]),
        support=float(field.support[py, px]),
        pixel_y=sub_y,
        pixel_x=sub_x,
        accepted=is_valid,
        reason=reason,
        score=float(max(0.0, field.score[py, px]) * max(0.0, min(1.0, coverage))) if np.isfinite(field.score[py, px]) else 0.0,
        q_normal_step=float(field.q_normal_step[py, px]),
        q_scale_anisotropy=float(field.q_scale_anisotropy[py, px]),
        q_unit=field.q_unit,
    )


def _annotate_ridge_continuity(
    points: Sequence[RidgePoint],
    requested_angles: np.ndarray,
) -> tuple[float, float, float, tuple[str, ...]]:
    """Attach deterministic circular continuity diagnostics to real points.

    This is intentionally a small diagnostic, not a multi-track tracker.  A
    valid point is connected to the preceding valid angular sample; missing
    sectors create a gap and an unusually large local q change creates a
    jump.  No point is added or mirrored when either condition is observed.
    """

    n_points = len(points)
    if n_points == 0:
        return 0.0, float("nan"), float("nan"), tuple()
    angles = np.asarray(requested_angles, dtype=float).ravel()
    if angles.size != n_points:
        raise ValueError("requested_angles must contain one value per ridge point")
    valid = np.asarray([bool(point.valid) and np.isfinite(point.q) for point in points], dtype=bool)
    order = np.argsort(np.mod(angles, 2.0 * np.pi), kind="mergesort")
    sorted_angles = np.mod(angles[order], 2.0 * np.pi)
    full_deltas = np.mod(np.roll(sorted_angles, -1) - sorted_angles, 2.0 * np.pi)
    positive_deltas = full_deltas[full_deltas > 1e-12]
    expected_step = float(np.median(positive_deltas)) if positive_deltas.size else float("nan")
    valid_positions = [int(pos) for pos, index in enumerate(order) if valid[index]]
    if not valid_positions:
        for point in points:
            point.continuity_score = 0.0
            point.trajectory_id = None
            point.local_q_step = float("nan")
            if "no_valid_ridge" not in point.flags:
                point.flags = tuple(point.flags) + ("no_valid_ridge",)
                point.metadata["flags"] = point.flags
        return 0.0, 0.0, 0.0, ("no_valid_ridge",)

    q_valid = np.asarray([points[index].q for index in order if valid[index]], dtype=float)
    q_scale = float(np.nanmedian(np.abs(q_valid))) if q_valid.size else 0.0
    q_mad = float(np.nanmedian(np.abs(q_valid - np.nanmedian(q_valid)))) if q_valid.size else 0.0
    jump_threshold = max(0.10 * q_scale, 6.0 * q_mad, 1e-12)
    incoming: dict[int, tuple[float, bool, bool]] = {}
    for offset, current_pos in enumerate(valid_positions):
        if len(valid_positions) == 1:
            incoming[current_pos] = (float("nan"), False, False)
            continue
        previous_pos = valid_positions[offset - 1]
        previous_index = int(order[previous_pos])
        current_index = int(order[current_pos])
        angle_gap = float(np.mod(sorted_angles[current_pos] - sorted_angles[previous_pos], 2.0 * np.pi))
        if current_pos <= previous_pos:
            angle_gap = float(
                np.mod(
                    sorted_angles[current_pos] + 2.0 * np.pi - sorted_angles[previous_pos],
                    2.0 * np.pi,
                )
            )
        between_positions = (current_pos - previous_pos - 1) % n_points
        if current_pos <= previous_pos:
            between_positions = n_points - previous_pos - 1 + current_pos
        missing = any(not valid[int(order[(previous_pos + step) % n_points])] for step in range(1, between_positions + 1))
        local_q_step = float(abs(points[current_index].q - points[previous_index].q))
        angle_gap_flag = bool(np.isfinite(expected_step) and angle_gap > 1.75 * expected_step)
        jump_flag = bool(np.isfinite(local_q_step) and local_q_step > jump_threshold)
        incoming[current_pos] = (local_q_step, bool(missing or angle_gap_flag), jump_flag)

    trajectory_id = 0
    for offset, current_pos in enumerate(valid_positions):
        index = int(order[current_pos])
        local_q_step, gap_flag, jump_flag = incoming[current_pos]
        point = points[index]
        point.local_q_step = local_q_step
        point.continuity_score = 0.0 if gap_flag or jump_flag else 1.0
        point.trajectory_id = trajectory_id
        extra_flags = []
        if gap_flag:
            extra_flags.append("continuity_gap")
        if jump_flag:
            extra_flags.append("continuity_jump")
        if extra_flags:
            point.flags = tuple(dict.fromkeys(tuple(point.flags) + tuple(extra_flags)))
            point.metadata["flags"] = point.flags
        if offset > 0 and (gap_flag or jump_flag):
            trajectory_id += 1
            point.trajectory_id = trajectory_id
    for index, point in enumerate(points):
        if not valid[index]:
            point.continuity_score = 0.0
            point.trajectory_id = None
            point.local_q_step = float("nan")

    valid_count = int(np.count_nonzero(valid))
    continuity_values = np.asarray(
        [points[int(order[pos])].continuity_score for pos in valid_positions],
        dtype=float,
    )
    continuity_fraction = float(np.mean(continuity_values > 0.0))
    continuity_score = float(np.mean(continuity_values))
    diagnostics = []
    if any("continuity_gap" in point.flags for point in points):
        diagnostics.append("continuity_gap")
    if any("continuity_jump" in point.flags for point in points):
        diagnostics.append("continuity_jump")
    return float(valid_count / n_points), continuity_fraction, continuity_score, tuple(diagnostics)


def _annotate_azimuthal_tracks(
    points: Sequence[RidgePoint],
    annulus_indices: Mapping[int, Sequence[int]],
    *,
    n_annuli: int,
    q_step: float,
    angle_step: float,
) -> tuple[float, float, float, tuple[str, ...]]:
    """Attach q-wise trajectories and observed branch identities.

    Azimuthal maxima are sampled independently in each q annulus, so the
    radial-sector continuity diagnostic cannot be reused.  A point is linked
    only to a maximum in the immediately preceding *observed* annulus when
    its angular displacement is compatible with one angular bin (or the
    measured peak width).  Branch labels are then obtained from the two
    circular clusters that are actually present modulo pi; no missing
    quadrant is created to complete a pair.
    """

    if not points:
        return 0.0, 0.0, 0.0, ("no_valid_ridge",)
    previous: list[int] = []
    next_trajectory = 0
    ever_observed = False
    continuity_values: list[float] = []
    supported_annuli: set[int] = set()
    for annulus in range(int(n_annuli)):
        current = list(annulus_indices.get(annulus, ()))
        if current:
            supported_annuli.add(annulus)
        used_previous: set[int] = set()
        for point_index in current:
            point = points[int(point_index)]
            best: tuple[float, int] | None = None
            for previous_index in previous:
                if previous_index in used_previous:
                    continue
                prior = points[previous_index]
                widths = [
                    0.5 * float(width)
                    for width in (point.azimuthal_fwhm, prior.azimuthal_fwhm)
                    if np.isfinite(width) and width > 0.0
                ]
                allowed = max([2.5 * float(angle_step), *widths])
                distance = float(_wrap_distance(point.angle, prior.angle))
                candidate = (distance, previous_index)
                if distance <= allowed and (best is None or candidate < best):
                    best = candidate
            if best is None:
                point.trajectory_id = next_trajectory
                next_trajectory += 1
                point.continuity_score = 0.0 if ever_observed else 1.0
                if ever_observed:
                    point.flags = tuple(dict.fromkeys(point.flags + ("continuity_gap",)))
                    point.metadata["flags"] = point.flags
            else:
                point.trajectory_id = points[best[1]].trajectory_id
                point.continuity_score = 1.0
                used_previous.add(best[1])
            point.local_q_step = float(q_step)
            continuity_values.append(float(point.continuity_score))
        if current:
            ever_observed = True
        previous = current

    # Circular two-means in the unoriented angle domain.  A pair at +phi and
    # -phi has distinct centres modulo pi; Friedel counterparts share the
    # same centre.  Degenerate one-cluster data receives branch 0 only.
    angles = np.mod(np.asarray([point.angle for point in points], dtype=float), np.pi)
    if angles.size == 1:
        labels = np.zeros(1, dtype=int)
    else:
        sorted_angles = np.sort(angles)
        centres = np.asarray(
            (sorted_angles[angles.size // 4], sorted_angles[(3 * angles.size) // 4]),
            dtype=float,
        )
        for _ in range(24):
            distance = np.abs(angles[:, None] - centres[None, :])
            distance = np.minimum(distance, np.pi - distance)
            labels = np.argmin(distance, axis=1)
            updated = centres.copy()
            for cluster in (0, 1):
                values = angles[labels == cluster]
                if values.size:
                    updated[cluster] = float(
                        0.5 * np.mod(np.arctan2(np.mean(np.sin(2.0 * values)),
                                                  np.mean(np.cos(2.0 * values))), 2.0 * np.pi)
                    )
            if np.max(np.minimum(np.abs(updated - centres), np.pi - np.abs(updated - centres))) < 1e-10:
                centres = updated
                break
            centres = updated
        separation = float(min(abs(centres[0] - centres[1]), np.pi - abs(centres[0] - centres[1])))
        if separation < max(1.5 * float(angle_step), 1e-6) or np.count_nonzero(labels == 1) == 0:
            labels = np.zeros(angles.size, dtype=int)
        elif centres[1] < centres[0]:
            labels = 1 - labels
    for point, label in zip(points, labels):
        point.branch_id = int(label)
        point.metadata["branch_id"] = int(label)

    valid_annulus_fraction = float(len(supported_annuli) / max(1, int(n_annuli)))
    continuity_fraction = float(np.mean(np.asarray(continuity_values) > 0.0)) if continuity_values else 0.0
    continuity_score = float(np.mean(continuity_values)) if continuity_values else 0.0
    flags: list[str] = []
    if any("continuity_gap" in point.flags for point in points):
        flags.append("continuity_gap")
    if len(supported_annuli) < int(n_annuli):
        flags.append("short_q_annulus_support")
    return valid_annulus_fraction, continuity_fraction, continuity_score, tuple(flags)


def _azimuthal_peak_ridges(
    frame: Any,
    qmap: Any,
    q_window: Any,
    *,
    q_range: Any = None,
    n_annuli: int = 256,
    n_angle_bins: int = 72,
    snr_threshold: float = 2.0,
    min_peak_fraction: float = 0.0,
    min_coverage: float = 0.0,
    mask: Any = None,
) -> tuple[list[RidgePoint], np.ndarray, np.ndarray, float, float, float, tuple[str, ...]]:
    """Extract directly observed angular maxima in sampled q annuli.

    The image is binned once into a q-by-angle accumulator.  This keeps the
    method practical for detector-sized frames while preserving the distinction
    from ``radial_peak``: each point is an angular maximum at a fixed q annulus,
    not a radial maximum in a fixed angular sector.
    """

    n_annuli = int(n_annuli)
    n_angle_bins = int(n_angle_bins)
    if n_annuli < 4:
        raise ValueError("n_bins must be at least 4 for azimuthal_peak")
    if n_angle_bins < 16:
        raise ValueError("n_angles must be at least 16 for azimuthal_peak")
    if not np.isfinite(float(snr_threshold)) or float(snr_threshold) < 0.0:
        raise ValueError("ridge_snr_threshold must be finite and non-negative")
    if not np.isfinite(float(min_peak_fraction)) or not 0.0 <= float(min_peak_fraction) <= 1.0:
        raise ValueError("ridge_min_peak_fraction must be in [0, 1]")
    if not np.isfinite(float(min_coverage)) or not 0.0 <= float(min_coverage) <= 1.0:
        raise ValueError("ridge_min_coverage must be in [0, 1]")

    values, q, angle, valid = _extract_maps(frame, qmap, mask)
    q_unit = _q_unit(qmap)
    q_min, q_max = _q_limits(q, q_window, q_range)
    q_edges = np.linspace(q_min, q_max, n_annuli + 1, dtype=float)
    q_centres = 0.5 * (q_edges[:-1] + q_edges[1:])
    q_step = float(q_edges[1] - q_edges[0])
    angle_edges = np.linspace(-np.pi, np.pi, n_angle_bins + 1, dtype=float)
    angle_centres = 0.5 * (angle_edges[:-1] + angle_edges[1:])
    angle_step = float(angle_edges[1] - angle_edges[0])

    q_index_float = (
        (q - q_min) / max(q_max - q_min, np.finfo(float).eps) * n_annuli
    )
    q_index = np.floor(np.where(np.isfinite(q_index_float), q_index_float, 0.0)).astype(np.int64)
    q_index = np.clip(q_index, 0, n_annuli - 1)
    angle_index_float = (angle + np.pi) / (2.0 * np.pi) * n_angle_bins
    angle_index = np.floor(
        np.where(np.isfinite(angle_index_float), angle_index_float, 0.0)
    ).astype(np.int64)
    angle_index = np.clip(angle_index, 0, n_angle_bins - 1)
    candidate = (
        np.isfinite(values)
        & np.isfinite(q)
        & np.isfinite(angle)
        & (q >= q_min)
        & (q <= q_max)
    )
    selected = candidate & valid
    flat_candidate = q_index[candidate] * n_angle_bins + angle_index[candidate]
    flat_selected = q_index[selected] * n_angle_bins + angle_index[selected]
    total = n_annuli * n_angle_bins
    candidate_counts = np.bincount(flat_candidate, minlength=total).reshape(n_annuli, n_angle_bins)
    counts = np.bincount(flat_selected, minlength=total).reshape(n_annuli, n_angle_bins)
    sums = np.bincount(
        flat_selected,
        weights=values[selected],
        minlength=total,
    ).reshape(n_annuli, n_angle_bins)
    profile = np.divide(
        sums,
        counts,
        out=np.full((n_annuli, n_angle_bins), np.nan, dtype=float),
        where=counts > 0,
    )
    coverage = np.divide(
        counts,
        candidate_counts,
        out=np.zeros_like(profile),
        where=candidate_counts > 0,
    )
    annulus_candidate = np.sum(candidate_counts, axis=1)
    annulus_valid = np.sum(counts, axis=1)
    annulus_coverage = np.divide(
        annulus_valid,
        annulus_candidate,
        out=np.zeros(n_annuli, dtype=float),
        where=annulus_candidate > 0,
    )
    points: list[RidgePoint] = []
    annulus_indices: dict[int, list[int]] = {}
    rejected_boundary = False
    rejected_support = False
    for annulus in range(n_annuli):
        if annulus_candidate[annulus] <= 0 or annulus_coverage[annulus] < float(min_coverage):
            if annulus_candidate[annulus] > 0:
                rejected_support = True
            continue
        values_angular = profile[annulus]
        finite_profile = np.isfinite(values_angular) & (counts[annulus] > 0)
        if np.count_nonzero(finite_profile) < 3:
            continue
        baseline = float(np.nanpercentile(values_angular[finite_profile], 10.0))
        noise = _robust_noise(values_angular[finite_profile] - baseline)
        if not np.isfinite(noise) or noise <= np.finfo(float).eps:
            noise = max(float(np.nanstd(values_angular[finite_profile])), np.finfo(float).eps)
        if gaussian_filter1d is not None:
            support_weights = finite_profile.astype(float)
            numerator = gaussian_filter1d(
                np.where(finite_profile, values_angular, 0.0),
                1.0,
                mode="wrap",
            )
            denominator = gaussian_filter1d(support_weights, 1.0, mode="wrap")
            smoothed = np.divide(
                numerator,
                denominator,
                out=np.full(n_angle_bins, np.nan, dtype=float),
                where=denominator > 1e-9,
            )
        else:  # pragma: no cover - scipy is a declared dependency
            smoothed = values_angular.copy()
        peak_input = np.where(np.isfinite(smoothed), smoothed, baseline)
        if find_peaks is None:  # pragma: no cover
            peak_indices = np.asarray([int(np.nanargmax(peak_input))])
            peak_prominence = np.asarray([float(np.nanmax(peak_input) - baseline)])
        else:
            extended = np.tile(peak_input, 3)
            peak_indices_extended, properties = find_peaks(
                extended,
                prominence=max(0.0, float(snr_threshold) * noise),
                distance=max(1, int(round(n_angle_bins * 0.02))),
            )
            # Keep the prominence from the corresponding central copy and
            # deduplicate edge peaks modulo the angular period.
            peak_indices_list: list[int] = []
            peak_prominence_list: list[float] = []
            all_prominences = np.asarray(properties.get("prominences", ()), dtype=float)
            for property_index, peak_position_extended in enumerate(peak_indices_extended.tolist()):
                if not (n_angle_bins <= peak_position_extended < 2 * n_angle_bins):
                    continue
                peak = int(peak_position_extended - n_angle_bins)
                if all(
                    _wrap_distance(angle_centres[int(peak)], angle_centres[other]) > angle_step
                    for other in peak_indices_list
                ):
                    peak_indices_list.append(int(peak))
                    if property_index < all_prominences.size:
                        peak_prominence_list.append(float(all_prominences[property_index]))
                    else:
                        peak_prominence_list.append(float(peak_input[int(peak)] - baseline))
            peak_indices = np.asarray(peak_indices_list, dtype=int)
            peak_prominence = np.asarray(peak_prominence_list, dtype=float)
        support_mask = finite_profile & (coverage[annulus] >= float(min_coverage))
        finite_prominence = peak_prominence[np.isfinite(peak_prominence) & (peak_prominence > 0.0)]
        strongest_prominence = float(np.max(finite_prominence)) if finite_prominence.size else float("nan")
        for peak_position, index in enumerate(peak_indices.tolist()):
            index = int(index)
            neighbours = support_mask[(index - 1) % n_angle_bins] and support_mask[(index + 1) % n_angle_bins]
            if not support_mask[index] or not neighbours:
                rejected_boundary = True
                continue
            peak_angle, peak_intensity = _quadratic_peak(
                angle_centres,
                smoothed,
                index,
                period=2.0 * np.pi,
            )
            prominence = float(peak_prominence[peak_position]) if peak_position < len(peak_prominence) else float(peak_intensity - baseline)
            if (
                np.isfinite(strongest_prominence)
                and prominence < float(min_peak_fraction) * strongest_prominence
            ):
                continue
            snr = float(max(prominence, peak_intensity - baseline) / max(noise, np.finfo(float).eps))
            if not np.isfinite(snr) or snr < float(snr_threshold):
                continue
            fwhm = _periodic_fwhm(angle_centres, smoothed, index, baseline)
            point = RidgePoint(
                angle=float(peak_angle),
                q=float(q_centres[annulus]),
                # An angular maximum is sampled at a prescribed q annulus;
                # it is not a radial reflection position.  Keep ``q`` for
                # the Cartesian track coordinate, but fail closed for the
                # legacy q_star/lamellar-spacing aliases.
                q_star=float("nan"),
                lamellar_spacing=float("nan"),
                intensity=float(peak_intensity),
                baseline=baseline,
                snr=snr,
                radial_fwhm=float("nan"),
                azimuthal_fwhm=float(fwhm),
                area=float(max(0.0, peak_intensity - baseline) * (fwhm if np.isfinite(fwhm) else angle_step)),
                coverage=float(annulus_coverage[annulus]),
                n_pixels=int(counts[annulus, index]),
                valid=True,
                source="observed",
                flags=_flags_with_q_unit(
                    APPARENT_FLAGS + (
                        "azimuthal_peak",
                        "observed_angular_maximum",
                        "azimuthal_area_approximation",
                        "spacing_unavailable_azimuthal_trajectory",
                    ),
                    q_unit,
                ),
                method="azimuthal_peak",
                support=float(annulus_coverage[annulus]),
                accepted=True,
                reason="accepted",
                q_unit=q_unit,
                score=float(max(0.0, snr) * max(0.0, min(1.0, annulus_coverage[annulus]))),
                q_normal_step=q_step,
            )
            point.metadata.update(
                annulus_index=int(annulus),
                annulus_q_min=float(q_edges[annulus]),
                annulus_q_max=float(q_edges[annulus + 1]),
                angular_bin_index=index,
                angular_bin_coverage=float(coverage[annulus, index]),
                area_definition="peak_height_times_angular_fwhm_approximation",
            )
            point_index = len(points)
            points.append(point)
            annulus_indices.setdefault(annulus, []).append(point_index)

    valid_annulus_fraction, continuity_fraction, continuity_score, continuity_flags = _annotate_azimuthal_tracks(
        points,
        annulus_indices,
        n_annuli=n_annuli,
        q_step=q_step,
        angle_step=angle_step,
    )
    flags: list[str] = [
        "azimuthal_peak_ridge",
        "observed_angular_maxima",
        "azimuthal_area_approximation",
    ]
    flags.extend(continuity_flags)
    if rejected_boundary:
        flags.append("masked_gap_or_boundary_peak_rejected")
    if rejected_support:
        flags.append("low_peak_support")
    if not points:
        flags.append("no_azimuthal_peak")
    return points, np.asarray([point.angle for point in points], dtype=float), np.asarray(
        [point.q for point in points], dtype=float
    ), valid_annulus_fraction, continuity_fraction, continuity_score, tuple(dict.fromkeys(flags))


def measure_radial_ridges(
    frame: Any,
    qmap: Any,
    q_window: Any = None,
    *,
    q_range: Any = None,
    angles: Sequence[float] | None = None,
    n_angles: int = 72,
    sector_width: float | None = None,
    n_bins: int = 256,
    lobe_metrics: Sequence[LobeMetrics] | None = None,
    snr_threshold: float = 2.0,
    ridge_snr_threshold: float | None = None,
    ridge_min_peak_fraction: float = 0.0,
    ridge_min_coverage: float = 0.0,
    ridge_method: str = "radial_peak",
    method: str | None = None,
    mask: Any = None,
    curvature_sigma: float = 2.0,
    curvature_percentile: float = 25.0,
    curvature_normal_step: float = 1.0,
    cancel_event: Any = None,
) -> RidgeTrack:
    """Measure directly observed radial or annular angular maxima.

    If ``angles`` is omitted, uniformly spaced sectors cover the complete
    angular range.  Crucially, this routine never copies or mirrors a point
    from a counterpart quadrant; masked and low-SNR sectors stay invalid.
    With ``ridge_method='azimuthal_peak'``, ``n_bins`` is the number of q
    annuli and ``n_angles`` is the angular bin count.  In that mode
    ``ridge_min_peak_fraction`` retains only angular maxima whose measured
    prominence is at least that fraction of the strongest prominence in the
    same annulus; ``ridge_min_coverage`` independently gates detector
    support.
    """

    raise_if_cancelled(cancel_event, "ridges:start")
    ridge_method = str(method if method is not None else ridge_method).lower().replace("-", "_")
    if ridge_method not in {"radial_peak", "surface_curvature", "curvature", "azimuthal_peak"}:
        raise ValueError("ridge_method must be 'radial_peak', 'surface_curvature' or 'azimuthal_peak'")
    if ridge_snr_threshold is not None:
        snr_threshold = float(ridge_snr_threshold)
    snr_threshold = float(snr_threshold)
    if not np.isfinite(snr_threshold) or snr_threshold < 0.0:
        raise ValueError("ridge_snr_threshold must be finite and non-negative")
    ridge_min_peak_fraction = float(ridge_min_peak_fraction)
    if not np.isfinite(ridge_min_peak_fraction) or not 0.0 <= ridge_min_peak_fraction <= 1.0:
        raise ValueError("ridge_min_peak_fraction must be in [0, 1]")
    ridge_min_coverage = float(ridge_min_coverage)
    if not np.isfinite(ridge_min_coverage) or not 0.0 <= ridge_min_coverage <= 1.0:
        raise ValueError("ridge_min_coverage must be in [0, 1]")
    azimuthal_mode = ridge_method == "azimuthal_peak"
    curvature_mode = ridge_method in {"surface_curvature", "curvature"}
    automatic_angles = angles is None
    if angles is None:
        # Lobe locations are annotations, not a substitute for a densely
        # sampled trajectory.  A four-point-only track cannot constrain the
        # five geometric parameters of the shared-centre double ellipse.
        angles = np.linspace(-np.pi, np.pi, int(n_angles), endpoint=False).tolist()
    angle_array = np.asarray(angles, dtype=float).ravel()
    q_unit = _q_unit(qmap)
    continuity_mode = bool(
        not curvature_mode
        and automatic_angles
        and len(angle_array) >= 12
        and _q_to_nm_inverse_scale(q_unit) is not None
    )
    if not angle_array.size:
        return RidgeTrack(
            points=[],
            angles=angle_array,
            q=np.array([]),
            valid=np.array([], dtype=bool),
            coverage=np.array([]),
            q_unit=q_unit,
        )
    if azimuthal_mode:
        points, observed_angles, q_values, valid_fraction, continuity_fraction, continuity_score, azimuthal_flags = _azimuthal_peak_ridges(
            frame,
            qmap,
            q_window,
            q_range=q_range,
            n_annuli=n_bins,
            n_angle_bins=len(angle_array),
            snr_threshold=snr_threshold,
            min_peak_fraction=ridge_min_peak_fraction,
            min_coverage=ridge_min_coverage,
            mask=mask,
        )
        valid = np.asarray([point.valid for point in points], dtype=bool)
        coverage = np.asarray([point.coverage for point in points], dtype=float)
        track_flags = _flags_with_q_unit(
            APPARENT_FLAGS + azimuthal_flags,
            q_unit,
        )
        if lobe_metrics and points:
            # Keep the annulus-local angular width measured from the direct
            # profile.  A nearest lobe is a separate global annotation and
            # must not overwrite this method's observable.
            for point in points:
                raise_if_cancelled(cancel_event, "ridges:azimuthal-annotation")
                distances = [_wrap_distance(point.angle, lobe.angle) for lobe in lobe_metrics]
                nearest_lobe = lobe_metrics[int(np.argmin(distances))]
                point.metadata["nearest_global_lobe_angle"] = float(nearest_lobe.angle)
                point.metadata["nearest_global_lobe_fwhm"] = float(nearest_lobe.fwhm)
        return RidgeTrack(
            points=points,
            angles=observed_angles,
            q=q_values,
            valid=valid,
            coverage=coverage,
            flags=track_flags,
            q_unit=q_unit,
            valid_fraction=float(valid_fraction),
            continuity_fraction=float(continuity_fraction),
            continuity_score=float(continuity_score),
        )
    if sector_width is None:
        sector_width = 2.0 * np.pi / max(1, len(angle_array)) * 1.25
    curvature_field = None
    if curvature_mode:
        curvature_field = _surface_curvature_field(
            frame,
            qmap,
            q_window,
            q_range=q_range,
            mask=mask,
            smooth_sigma=curvature_sigma,
            curvature_percentile=curvature_percentile,
            normal_step=curvature_normal_step,
        )
    points: list[RidgePoint]
    if continuity_mode:
        profiles: list[RadialProfile] = []
        for angle in angle_array:
            raise_if_cancelled(cancel_event, "ridges:continuity-sector")
            profiles.append(measure_radial_profile(
                frame,
                qmap,
                float(angle),
                q_window,
                q_range=q_range,
                n_bins=n_bins,
                sector_width=sector_width,
                mask=mask,
            ))
        raise_if_cancelled(cancel_event, "ridges:continuity-noise")
        noise_floor = _radial_noise_floor(
            frame,
            qmap,
            q_window,
            q_range=q_range,
            mask=mask,
        )
        points = _continuous_radial_points(
            profiles,
            noise_floor=noise_floor,
            snr_threshold=snr_threshold,
            min_coverage=ridge_min_coverage,
        )
        raise_if_cancelled(cancel_event, "ridges:continuity-complete")
    else:
        points = []
        for angle in angle_array:
            raise_if_cancelled(cancel_event, "ridges:sector")
            if curvature_mode:
                assert curvature_field is not None
                point = _curvature_point_for_sector(
                    curvature_field,
                    float(angle),
                    float(sector_width),
                    float(snr_threshold),
                    min_coverage=float(ridge_min_coverage),
                )
            else:
                radial = measure_radial_profile(
                    frame,
                    qmap,
                    float(angle),
                    q_window,
                    q_range=q_range,
                    n_bins=n_bins,
                    sector_width=sector_width,
                    mask=mask,
                )
                point = _radial_peak(
                    radial,
                    snr_threshold=snr_threshold,
                    min_coverage=ridge_min_coverage,
                )
            points.append(point)

    if lobe_metrics:
        for angle, point in zip(angle_array, points):
            distances = [_wrap_distance(float(angle), lobe.angle) for lobe in lobe_metrics]
            nearest = int(np.argmin(distances))
            nearest_lobe = lobe_metrics[nearest]
            point.azimuthal_fwhm = float(nearest_lobe.fwhm)  # type: ignore[misc]
    q_values = np.asarray([point.q for point in points], dtype=float)
    valid = np.asarray([point.valid for point in points], dtype=bool)
    coverage = np.asarray([point.coverage for point in points], dtype=float)
    valid_fraction, continuity_fraction, continuity_score, continuity_flags = _annotate_ridge_continuity(
        points,
        angle_array,
    )
    track_flags = APPARENT_FLAGS + (("detector_pixel_principal_curvature",) if curvature_mode else tuple())
    if continuity_mode:
        track_flags += ("radial_continuity_tracking",)
    track_flags += continuity_flags
    track_flags = _flags_with_q_unit(track_flags, q_unit)
    return RidgeTrack(
        points=points,
        angles=angle_array,
        q=q_values,
        valid=valid,
        coverage=coverage,
        flags=track_flags,
        q_unit=q_unit,
        valid_fraction=valid_fraction,
        continuity_fraction=continuity_fraction,
        continuity_score=continuity_score,
    )


radial_ridge = measure_radial_ridges
extract_ridge_points = measure_radial_ridges


def extract_surface_curvature_ridges(*args: Any, **kwargs: Any) -> RidgeTrack:
    """Explicit public entry point for the principal-curvature ridge mode."""

    kwargs["ridge_method"] = "surface_curvature"
    return measure_radial_ridges(*args, **kwargs)


def _ridge_xy(points: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if isinstance(points, RidgeTrack):
        points = points.points
    # A numeric ``(n, 2)`` array follows the conventional optimizer contract:
    # its columns are Cartesian qx/qy coordinates.  The generic sequence
    # fallback below intentionally retains the legacy ``(angle, q)`` tuple
    # interpretation, so it must not be used for NumPy point clouds.
    if isinstance(points, np.ndarray):
        array = np.asarray(points, dtype=float)
        if array.ndim != 2 or array.shape[1] != 2:
            raise ValueError("ridge point array must have shape (n, 2) for qx/qy")
        xarr, yarr = array[:, 0], array[:, 1]
        return xarr, yarr, np.isfinite(xarr) & np.isfinite(yarr)
    if isinstance(points, Mapping):
        x = _array_field(points, ("x", "qx", "q_x"))
        y = _array_field(points, ("y", "qy", "q_y"))
        q = _array_field(points, ("q", "q_star", "radius"))
        angle = _array_field(points, ("angle", "azimuth", "phi"))
        if x is None or y is None:
            if q is None or angle is None:
                raise ValueError("ridge mapping needs x/y or q/angle")
            q, angle = np.broadcast_arrays(np.asarray(q, dtype=float), np.asarray(angle, dtype=float))
            x, y = q * np.cos(angle), q * np.sin(angle)
        else:
            x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        return x.ravel(), y.ravel(), np.isfinite(x.ravel()) & np.isfinite(y.ravel())
    seq = list(points)
    xs: list[float] = []
    ys: list[float] = []
    for point in seq:
        x = _get_field(point, ("x", "qx", "q_x"))
        y = _get_field(point, ("y", "qy", "q_y"))
        if _get_field(point, ("valid",), True) is False:
            # Shared RidgePoint intentionally rejects NaN coordinates.  Keep
            # invalid sectors out of a downstream ellipse fit explicitly.
            x, y = np.nan, np.nan
        if x is None or y is None:
            q = _get_field(point, ("q", "q_star", "radius"))
            angle = _get_field(point, ("angle", "azimuth", "phi"))
            if q is None or angle is None:
                try:
                    angle, q = point[0], point[1]
                except Exception as exc:
                    raise ValueError("each ridge point needs x/y or angle/q") from exc
            x, y = float(q) * np.cos(float(angle)), float(q) * np.sin(float(angle))
        xs.append(float(x))
        ys.append(float(y))
    xarr, yarr = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    return xarr, yarr, np.isfinite(xarr) & np.isfinite(yarr)


def _ridge_source(points: Any) -> Any:
    """Return the concrete point collection behind a ridge adapter."""

    if isinstance(points, RidgeTrack):
        return points.points
    if isinstance(points, Mapping):
        nested = _get_field(points, ("points", "ridges", "ridge_points"))
        if nested is not None:
            return nested
    return points


def _ridge_components(points: Any) -> np.ndarray | None:
    """Read aligned branch labels while retaining valid partial labels.

    ``NaN`` marks an invalid or unassigned point.  Returning the aligned
    array lets strict butterfly callers retain every valid label instead of
    silently discarding the whole topology because one masked sector is
    unlabeled.
    """

    source = _ridge_source(points)
    if isinstance(source, Mapping):
        value = _get_field(source, ("component", "components", "labels", "branch", "branch_id"))
        if value is None:
            return None
        array = np.asarray(value)
        if array.ndim == 0:
            return None
        try:
            numeric = array.astype(float, copy=False).ravel()
        except (TypeError, ValueError, OverflowError):
            return None
        valid = np.isfinite(numeric) & np.isin(numeric, (0, 1))
        if not np.any(valid):
            return None
        result = np.full(numeric.shape, np.nan, dtype=float)
        result[valid] = numeric[valid]
        return result
    if isinstance(source, np.ndarray) or isinstance(source, (str, bytes)):
        return None
    try:
        sequence = list(source)
    except TypeError:
        return None
    values: list[Any] = []
    found = False
    for point in sequence:
        component = _get_field(point, ("component", "branch", "label", "branch_id"))
        if component is None:
            metadata = _get_field(point, ("metadata",), {})
            component = _get_field(metadata, ("component", "branch", "label", "branch_id"))
        if component is None:
            values.append(np.nan)
        else:
            values.append(component)
            found = True
    if not found:
        return None
    array = np.asarray(values)
    try:
        numeric = array.astype(float, copy=False)
    except (TypeError, ValueError):
        return None
    valid = np.isfinite(numeric) & np.isin(numeric, (0, 1))
    if not np.any(valid):
        return None
    numeric = numeric.astype(float, copy=False)
    numeric[~valid] = np.nan
    return numeric


def _ridge_weights(points: Any) -> np.ndarray | None:
    """Read optional positive point weights for the canonical solver."""

    source = _ridge_source(points)
    if isinstance(source, Mapping):
        value = _get_field(source, ("weight", "weights"))
        if value is None:
            return None
        array = np.asarray(value, dtype=float)
        return array.ravel() if array.ndim else None
    if isinstance(source, np.ndarray) or isinstance(source, (str, bytes)):
        return None
    try:
        sequence = list(source)
    except TypeError:
        return None
    values: list[float] = []
    found = False
    for point in sequence:
        value = _get_field(point, ("weight",))
        if value is None:
            values.append(1.0)
        else:
            values.append(float(value))
            found = True
    return np.asarray(values, dtype=float) if found else None


def _infer_ridge_q_unit(points: Any) -> str:
    """Read q-unit provenance from a ridge adapter without guessing units."""

    value = _get_field(points, ("q_unit",), None)
    if value is not None:
        return str(value or "unknown")
    source = _ridge_source(points)
    if isinstance(source, Mapping) or isinstance(source, (str, bytes, np.ndarray)):
        return str(_get_field(source, ("q_unit",), "unknown") or "unknown")
    try:
        sequence = list(source)
    except TypeError:
        return "unknown"
    for point in sequence:
        value = _get_field(point, ("q_unit",), None)
        if value is not None:
            return str(value or "unknown")
        metadata = _get_field(point, ("metadata",), {})
        value = _get_field(metadata, ("q_unit", "unit"), None)
        if value is not None:
            return str(value or "unknown")
    return "unknown"


def _initial_to_canonical(initial: Any) -> Any:
    """Translate the legacy ``a,b,theta`` initialiser to canonical names."""

    if initial is None:
        return None
    # ParameterSet carries editable/fixed/tied specifications and must not be
    # flattened through ``dict()`` (its items are resolved numeric values).
    if hasattr(initial, "spec_items") and hasattr(initial, "free_names"):
        return initial
    if not isinstance(initial, Mapping):
        values = list(initial)
        if len(values) < 3:
            raise ValueError("ellipse initial must contain a, b, theta")
        a_value, b_value = float(values[0]), float(values[1])
        if a_value <= 0:
            raise ValueError("ellipse initial a must be positive")
        return {"a": a_value, "axis_ratio": b_value / a_value, "theta": float(values[2])}

    data = dict(initial)
    if "a" not in data:
        for alias in ("semi_major", "major", "a0"):
            if alias in data:
                data["a"] = data[alias]
                break
    if "b" not in data:
        for alias in ("semi_minor", "minor", "b0"):
            if alias in data:
                data["b"] = data[alias]
                break
    if "theta" not in data and "tilt" in data:
        data["theta"] = data["tilt"]
    if "center" in data:
        center = data.pop("center")
        if len(center) != 2:
            raise ValueError("ellipse center must contain (cx, cy)")
        data.setdefault("cx", center[0])
        data.setdefault("cy", center[1])
    if "centre" in data:
        center = data.pop("centre")
        if len(center) != 2:
            raise ValueError("ellipse centre must contain (cx, cy)")
        data.setdefault("cx", center[0])
        data.setdefault("cy", center[1])

    # ``ellipse.py`` accepts theta_deg directly.  Keep it in that form so a
    # ParameterSpec with min/max/vary remains editable in degree units.
    if "theta_deg" not in data and "theta" not in data and "tilt_deg" in data:
        data["theta_deg"] = data["tilt_deg"]
    if "axis_ratio" not in data and "ratio" in data:
        data["axis_ratio"] = data["ratio"]
    if "axis_ratio" not in data and "b" in data:
        if "a" not in data:
            raise ValueError("ellipse initial with b requires a")
        def _number(value: Any) -> float:
            if isinstance(value, Mapping):
                value = value.get("value", value.get("estimate"))
            elif hasattr(value, "value"):
                value = value.value
            return float(value)
        a_value, b_value = _number(data["a"]), _number(data["b"])
        if a_value <= 0:
            raise ValueError("ellipse initial a must be positive")
        data["axis_ratio"] = b_value / a_value
        # b is derived in the canonical parameterisation.  Removing a plain
        # legacy b also prevents it from overriding the tied expression.
        data.pop("b", None)
    return data


def _ellipse_center_hint(parameters: Any) -> tuple[tuple[float, float], bool]:
    """Return a fixed-center hint for pre-fit quadrant labeling."""

    if parameters is None:
        return (0.0, 0.0), False
    values: Mapping[str, Any] | None = None
    fixed = False
    if hasattr(parameters, "resolve"):
        try:
            resolved = parameters.resolve()
            if isinstance(resolved, Mapping):
                values = resolved
            specs = dict(parameters.spec_items()) if hasattr(parameters, "spec_items") else {}
            fixed = bool(
                specs
                and all(
                    name in specs and (not bool(getattr(specs[name], "vary", True)))
                    for name in ("cx", "cy")
                )
            )
        except (TypeError, ValueError):
            values = None
    elif isinstance(parameters, Mapping):
        values = parameters
        center = parameters.get("center", parameters.get("centre"))
        if isinstance(center, Sequence) and len(center) == 2:
            values = {**parameters, "cx": center[0], "cy": center[1]}
        fixed = bool(parameters.get("fixed_center", False))
        if not fixed:
            cx_spec = parameters.get("cx")
            cy_spec = parameters.get("cy")
            fixed = bool(
                isinstance(cx_spec, Mapping)
                and isinstance(cy_spec, Mapping)
                and cx_spec.get("vary") is False
                and cy_spec.get("vary") is False
            )
    if values is None:
        return (0.0, 0.0), fixed
    try:
        def numeric(value: Any) -> float:
            if isinstance(value, Mapping):
                value = value.get("value", value.get("estimate", 0.0))
            return float(value)

        return (numeric(values.get("cx", 0.0)), numeric(values.get("cy", 0.0))), fixed
    except (TypeError, ValueError):
        return (0.0, 0.0), False


def _empty_double_ellipse(
    n_points: int,
    message: str,
    *flags: str,
    q_unit: str = "unknown",
) -> DoubleEllipseFit:
    nan = float("nan")
    geometry = EllipseGeometry(nan, nan, nan, (nan, nan))
    unit = str(q_unit or "unknown")
    return DoubleEllipseFit(
        a=nan,
        b=nan,
        theta=nan,
        ellipses=(geometry, geometry),
        ellipticity=nan,
        axes_ratio=nan,
        rmse=nan,
        rss=nan,
        n_points=int(n_points),
        branch_counts=(0, 0),
        success=False,
        message=message,
        flags=_flags_with_q_unit(APPARENT_FLAGS + tuple(flags), unit),
        q_unit=unit,
    )


def ellipse_radius(angle: np.ndarray | float, a: float, b: float, theta: float) -> np.ndarray:
    """Polar radius of an origin-centred, rotated ellipse."""

    angle = np.asarray(angle, dtype=float)
    a, b = float(a), float(b)
    if a <= 0 or b <= 0:
        return np.full_like(angle, np.nan, dtype=float)
    delta = angle - float(theta)
    denominator = np.sqrt((b * np.cos(delta)) ** 2 + (a * np.sin(delta)) ** 2)
    return a * b / denominator


def _quadrant_labels(
    points: np.ndarray,
    *,
    center: tuple[float, float],
    reference_axis_deg: float,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Classify observed points in the local q-centred reference frame."""

    names = ("QI", "QII", "QIII", "QIV")
    relative = points - np.asarray(center, dtype=float)[None, :]
    angle = np.mod(
        np.arctan2(relative[:, 1], relative[:, 0]) - np.deg2rad(reference_axis_deg),
        2.0 * np.pi,
    )
    indices = np.floor(angle / (0.5 * np.pi)).astype(int)
    indices = np.clip(indices, 0, 3)
    return indices, names


def _symmetry_diagnostics(
    points: np.ndarray,
    labels: np.ndarray | None,
    *,
    center: tuple[float, float],
    reference_axis_deg: float,
    unassigned_count: int = 0,
    center_verified: bool = False,
    azimuthal: bool = False,
) -> dict[str, Any]:
    """Describe observed quadrant pairing without manufacturing counterparts."""

    if points.size == 0:
        return {
            "policy": "strict_butterfly_quadrant_pairing",
            "symmetry_status": "FAIL",
            "flags": ["no_observed_symmetry_points"],
            "quadrant_counts": {name: 0 for name in ("QI", "QII", "QIII", "QIV")},
            "branch_quadrant_counts": {"0": {}, "1": {}},
            "paired_support": {},
            "central_symmetry": {},
            "branch_leaks": {"direct": 0, "swapped": 0, "selected": 0},
            "unassigned_count": int(unassigned_count),
        }
    quadrant, names = _quadrant_labels(
        points,
        center=center,
        reference_axis_deg=reference_axis_deg,
    )
    quadrant_counts = {
        name: int(np.count_nonzero(quadrant == index))
        for index, name in enumerate(names)
    }
    diagnostics: dict[str, Any] = {
        "policy": "strict_butterfly_quadrant_pairing",
        "reference_axis_deg": float(reference_axis_deg),
        "center_qx": float(center[0]),
        "center_qy": float(center[1]),
        "center_verified": bool(center_verified),
        "quadrant_counts": quadrant_counts,
        "branch_quadrant_counts": {"0": {name: 0 for name in names}, "1": {name: 0 for name in names}},
        "paired_support": {},
        "central_symmetry": {},
        "branch_leaks": {"direct": 0, "swapped": 0, "selected": 0, "global_swap": False},
        "unassigned_count": int(unassigned_count),
        "q_difference_independent": not bool(azimuthal),
        "flags": [],
    }
    if labels is None:
        diagnostics["symmetry_status"] = "WARN"
        diagnostics["flags"].append("symmetry_unverified_unlabeled")
        return diagnostics
    labels = np.asarray(labels, dtype=float).ravel()
    if labels.size != points.shape[0]:
        diagnostics["symmetry_status"] = "FAIL"
        diagnostics["flags"].append("symmetry_label_alignment_error")
        return diagnostics
    relative = points - np.asarray(center, dtype=float)[None, :]
    local_angle = np.mod(
        np.arctan2(relative[:, 1], relative[:, 0]) - np.deg2rad(reference_axis_deg),
        2.0 * np.pi,
    )
    sector_phase = np.mod(local_angle, 0.5 * np.pi)
    boundary_ambiguous = np.minimum(sector_phase, 0.5 * np.pi - sector_phase) < np.deg2rad(2.0)
    valid_labels = np.isfinite(labels) & np.isin(labels, (0, 1)) & ~boundary_ambiguous
    diagnostics["unassigned_count"] = int(unassigned_count + np.count_nonzero(boundary_ambiguous))
    if np.any(boundary_ambiguous):
        diagnostics["flags"].append("quadrant_boundary_ambiguous")
    for branch_id in (0, 1):
        for index, name in enumerate(names):
            diagnostics["branch_quadrant_counts"][str(branch_id)][name] = int(
                np.count_nonzero(valid_labels & (labels == branch_id) & (quadrant == index))
            )
    expected = np.asarray((0, 1, 0, 1), dtype=int)
    direct_leaks = int(np.count_nonzero(valid_labels & (labels != expected[quadrant])))
    swapped_leaks = int(np.count_nonzero(valid_labels & (1 - labels != expected[quadrant])))
    selected_swap = bool(swapped_leaks < direct_leaks)
    selected_leaks = min(direct_leaks, swapped_leaks)
    diagnostics["branch_leaks"] = {
        "direct": direct_leaks,
        "swapped": swapped_leaks,
        "selected": selected_leaks,
        "global_swap": selected_swap,
    }
    if selected_swap:
        diagnostics["flags"].append("global_branch_swap_equivalent")
    working_labels = 1.0 - labels if selected_swap else labels
    for branch_id in (0, 1):
        for index, name in enumerate(names):
            diagnostics["branch_quadrant_counts"][str(branch_id)][name] = int(
                np.count_nonzero(valid_labels & (working_labels == branch_id) & (quadrant == index))
            )
    effective_unassigned = int(diagnostics["unassigned_count"])
    if effective_unassigned or np.count_nonzero(~valid_labels):
        diagnostics["flags"].append("unassigned_branch_points")

    radius = np.hypot(relative[:, 0], relative[:, 1])
    angle = np.mod(
        np.arctan2(relative[:, 1], relative[:, 0]) - np.deg2rad(reference_axis_deg),
        2.0 * np.pi,
    )
    radius_scale = max(float(np.nanmedian(radius[valid_labels])), np.finfo(float).eps)
    angle_tolerance = np.deg2rad(20.0)
    q_tolerance = max(0.05 * radius_scale, 1.0e-8)
    pair_specs = (("QI", "QIII", 0, 2), ("QII", "QIV", 1, 3))
    all_distance: list[float] = []
    all_angle_error: list[float] = []
    all_q_difference: list[float] = []
    missing_total = 0
    for branch_id, (first_name, second_name, first_quadrant, second_quadrant) in enumerate(pair_specs):
        # Greedy one-to-one matching keeps memory O(n); do not materialize a
        # full point-by-point distance matrix for large azimuthal tracks.
        first = np.flatnonzero(valid_labels & (working_labels == branch_id) & (quadrant == first_quadrant))
        second = np.flatnonzero(valid_labels & (working_labels == branch_id) & (quadrant == second_quadrant))
        unused = set(int(index) for index in second)
        pairs: list[tuple[int, int]] = []
        for first_index in first:
            if not unused:
                break
            candidates = sorted(
                (
                    (
                        float(
                            np.hypot(
                                angle[int(second_index)]
                                - angle[int(first_index)]
                                - np.pi,
                                0.0,
                            )
                        ),
                        int(second_index),
                    )
                    for second_index in unused
                ),
                key=lambda value: (abs((value[0] + np.pi) % (2.0 * np.pi) - np.pi), value[1]),
            )
            _, second_index = candidates[0]
            angular_error = abs(
                (angle[second_index] - angle[first_index] - np.pi + np.pi)
                % (2.0 * np.pi)
                - np.pi
            )
            q_difference = abs(float(radius[first_index] - radius[second_index]))
            if angular_error <= angle_tolerance and q_difference <= max(
                q_tolerance, 0.25 * max(radius[first_index], radius[second_index])
            ):
                pairs.append((int(first_index), second_index))
                unused.remove(second_index)
                all_distance.append(float(np.linalg.norm(relative[first_index] + relative[second_index])))
                all_angle_error.append(float(np.degrees(angular_error)))
                all_q_difference.append(q_difference)
        missing = int(len(first) + len(second) - 2 * len(pairs))
        missing_total += missing
        diagnostics["paired_support"][str(branch_id)] = {
            "quadrant_pair": f"{first_name}+{second_name}",
            "observed_side_counts": [int(len(first)), int(len(second))],
            "paired_count": int(len(pairs)),
            "missing_opposite_count": missing,
            "pair_fraction": float(2 * len(pairs) / max(1, len(first) + len(second))),
        }
    diagnostics["central_symmetry"] = {
        "matched_pair_count": len(all_distance),
        "median_deviation_q": float(np.median(all_distance)) if all_distance else None,
        "p95_deviation_q": float(np.percentile(all_distance, 95.0)) if all_distance else None,
        "median_angular_deviation_deg": float(np.median(all_angle_error)) if all_angle_error else None,
        "p95_angular_deviation_deg": float(np.percentile(all_angle_error, 95.0)) if all_angle_error else None,
        "median_q_difference": float(np.median(all_q_difference)) if all_q_difference else None,
        "p95_q_difference": float(np.percentile(all_q_difference, 95.0)) if all_q_difference else None,
        "q_difference_note": (
            "same-annulus sampling can impose q agreement; use angle/deviation for independent symmetry evidence"
            if azimuthal
            else "observed radial q difference between measured opposite points"
        ),
    }
    if missing_total:
        diagnostics["flags"].append("missing_opposite_support")
    if selected_leaks:
        diagnostics["flags"].append("branch_quadrant_leak")
    if not center_verified:
        diagnostics["flags"].append("symmetry_center_unverified")
    if selected_leaks:
        diagnostics["symmetry_status"] = "FAIL"
    elif missing_total or effective_unassigned or not center_verified:
        diagnostics["symmetry_status"] = "WARN"
    else:
        diagnostics["symmetry_status"] = "PASS"
    return diagnostics


def _legacy_fit_symmetric_double_ellipse(*args: Any, **kwargs: Any) -> DoubleEllipseFit:
    """Removed legacy implementation; public fitting is canonical."""

    return fit_symmetric_double_ellipse(*args, **kwargs)

def fit_symmetric_double_ellipse(
    points: Any,
    *,
    initial: Sequence[float] | Mapping[str, Any] | None = None,
    parameters: Any = None,
    params: Any = None,
    config: Any = None,
    robust_loss: str = "soft_l1",
    f_scale: float = 1.0,
    residual: str = "sampson",
    residual_kind: str | None = None,
    labels: Any = None,
    weights: Any = None,
    max_nfev: int | None = None,
    multistart: int = 7,
    reference_axis_deg: float = 0.0,
    q_unit: str | None = None,
    strict_symmetry: bool = False,
    cancel_event: Any = None,
) -> DoubleEllipseFit:
    """Fit a shared-centre pair of q-space ellipses at ``+/-theta``.

    This is the public observables adapter around
    :func:`butterfly_saxs.ellipse.fit_symmetric_ellipses`.  It retains the
    historical ``a/b/theta`` result fields while exposing canonical fit
    diagnostics (centre, covariance, standard errors, condition number,
    coverage and bound status).  Unlabelled ridge points are assigned to the
    nearer branch by the canonical residual function; no mirrored
    observations are fabricated.
    """

    if parameters is not None and params is not None:
        raise ValueError("supply only one of parameters or params")
    if parameters is not None and initial is not None:
        raise ValueError("supply only one of initial or parameters")
    if params is not None and initial is not None:
        raise ValueError("supply only one of initial or params")
    if config is not None and parameters is not None:
        raise ValueError("parameters are already supplied by config")
    if _fit_canonical_symmetric_ellipses is None:
        return _empty_double_ellipse(0, "canonical ellipse solver is unavailable", "solver_unavailable")

    resolved_q_unit = str(q_unit if q_unit is not None else _infer_ridge_q_unit(points) or "unknown")
    source_points = _ridge_source(points)
    point_flags: tuple[Any, ...] = ()
    if isinstance(points, Mapping) or isinstance(points, RidgeTrack):
        raw_flags = _get_field(points, ("flags",), ())
        try:
            point_flags = tuple(raw_flags or ())
        except TypeError:
            point_flags = ()
    azimuthal_trajectory = "azimuthal_peak_ridge" in point_flags
    if not azimuthal_trajectory and not isinstance(source_points, (Mapping, np.ndarray, str, bytes)):
        try:
            azimuthal_trajectory = any(
                str(_get_field(point, ("method",), "")).lower().replace("-", "_")
                == "azimuthal_peak"
                for point in source_points
            )
        except TypeError:
            azimuthal_trajectory = False

    source_parameters = parameters if parameters is not None else (params if params is not None else initial)
    center_hint, center_verified = _ellipse_center_hint(source_parameters)
    x, y, finite_xy = _ridge_xy(points)
    radius = np.hypot(x, y)
    inferred_labels = _ridge_components(points) if labels is None else np.asarray(labels)
    unassigned_count = 0
    if inferred_labels is not None:
        inferred_labels = np.asarray(inferred_labels, dtype=float).ravel()
        if inferred_labels.size != x.size:
            raise ValueError(f"labels must contain one value per ridge point ({x.size})")
        label_valid = np.isfinite(inferred_labels) & np.isin(inferred_labels, (0, 1))
        unassigned_count = int(np.count_nonzero(~label_valid))
        if labels is not None and unassigned_count and not strict_symmetry:
            raise ValueError("labels must contain only 0 or 1")
        if unassigned_count and not strict_symmetry:
            inferred_labels = None

    keep = finite_xy & np.isfinite(radius) & (radius > 0.0)
    if strict_symmetry and inferred_labels is None:
        return _empty_double_ellipse(
            int(np.count_nonzero(keep)),
            "strict symmetry requires aligned observed branch labels",
            "symmetry_unverified_unlabeled",
            q_unit=resolved_q_unit,
        )
    if strict_symmetry and inferred_labels is not None:
        keep &= np.isfinite(inferred_labels) & np.isin(inferred_labels, (0, 1))
        relative = np.column_stack((x - center_hint[0], y - center_hint[1]))
        local_angle = np.mod(
            np.arctan2(relative[:, 1], relative[:, 0]) - np.deg2rad(reference_axis_deg),
            2.0 * np.pi,
        )
        phase = np.mod(local_angle, 0.5 * np.pi)
        ambiguous = np.minimum(phase, 0.5 * np.pi - phase) < np.deg2rad(2.0)
        keep &= ~ambiguous
        unassigned_count += int(np.count_nonzero(ambiguous & finite_xy & (radius > 0.0)))
    xy = np.column_stack((x[keep], y[keep]))
    n_points = int(xy.shape[0])
    if n_points < 5:
        return _empty_double_ellipse(
            n_points,
            "at least five finite nonzero ridge points are required",
            "insufficient_points",
            q_unit=resolved_q_unit,
        )
    if inferred_labels is not None:
        inferred_labels = inferred_labels[keep].astype(int)

    point_weights = _ridge_weights(points) if weights is None else np.asarray(weights, dtype=float).ravel()
    if point_weights is not None:
        if point_weights.size != x.size:
            raise ValueError(f"weights must contain one value per ridge point ({x.size})")
        point_weights = point_weights[keep]

    canonical_parameters = _initial_to_canonical(source_parameters)
    if residual_kind is not None:
        residual = residual_kind
    label_options = [inferred_labels]
    if inferred_labels is not None and (labels is None or strict_symmetry):
        # Lobe-derived branch IDs are identities, not signed physical labels.
        # Try the one allowed global swap so a negative generator/model theta
        # is represented by the same canonical +|theta|/-|theta| pair.
        label_options.append(1 - inferred_labels)
    try:
        raise_if_cancelled(cancel_event, "observables:ellipse:start")
        labelled_results = [
            (
                option,
                _fit_canonical_symmetric_ellipses(
                    xy,
                    parameters=canonical_parameters,
                    residual=residual,
                    loss=robust_loss,
                    f_scale=f_scale,
                    labels=option,
                    weights=point_weights,
                    max_nfev=max_nfev,
                    multistart=multistart,
                    config=config,
                    reference_axis_deg=reference_axis_deg,
                    cancel_event=cancel_event,
                ),
            )
            for option in label_options
        ]
        selected_label_option, canonical_result = min(
            labelled_results,
            key=lambda item: (
                not bool(item[1].success),
                not bool(np.isfinite(item[1].cost)),
                float(item[1].cost) if np.isfinite(item[1].cost) else float("inf"),
            ),
        )
        swap_applied = bool(
            inferred_labels is not None
            and len(label_options) > 1
            and np.array_equal(selected_label_option, label_options[1])
        )
        inferred_labels = selected_label_option
    except Exception as exc:
        # Invalid editable configurations should be visible to a caller, but
        # a numerical failure for measured data remains a structured result
        # suitable for batch processing.
        if isinstance(exc, ValueError) and (canonical_parameters is not None or config is not None):
            raise
        return _empty_double_ellipse(
            n_points,
            f"ellipse solver failed: {exc}",
            "solver_failed",
            q_unit=resolved_q_unit,
        )

    values = dict(canonical_result.values)
    cx = float(values.get("cx", float("nan")))
    cy = float(values.get("cy", float("nan")))
    a = float(values.get("a", float("nan")))
    ratio = float(values.get("axis_ratio", float("nan")))
    b = float(values.get("b", a * ratio if np.isfinite(a * ratio) else float("nan")))
    theta = float(values.get("theta", float("nan")))
    centre = (cx, cy)
    reference_axis = float(getattr(canonical_result, "reference_axis_deg", reference_axis_deg))
    reference_rad = np.deg2rad(reference_axis)
    plus = EllipseGeometry(a, b, reference_rad + theta, centre)
    minus = EllipseGeometry(a, b, reference_rad - theta, centre)

    # Use the same q-space residual family as the canonical solver for branch
    # accounting.  Explicit labels take precedence over nearest-branch
    # assignment, which is important for supervised component comparisons.
    branch: np.ndarray
    if inferred_labels is not None:
        branch = inferred_labels == 0
    elif _CanonicalEllipseGeometry is not None:
        geometry = _CanonicalEllipseGeometry.from_values(values)
        residual_name = str(residual).lower()
        if residual_name in {"geometric", "distance", "closest"}:
            from .ellipse import ellipse_geometric_residuals as residual_function
        else:
            residual_function = _canonical_sampson_residuals
        plus_geometry = _CanonicalEllipseGeometry(
            geometry.cx,
            geometry.cy,
            geometry.a,
            geometry.axis_ratio,
            reference_rad + geometry.theta,
        )
        minus_geometry = _CanonicalEllipseGeometry(
            geometry.cx,
            geometry.cy,
            geometry.a,
            geometry.axis_ratio,
            reference_rad - geometry.theta,
        )
        r_plus = np.asarray(residual_function(xy, plus_geometry), dtype=float)
        r_minus = np.asarray(
            residual_function(xy, minus_geometry),
            dtype=float,
        )
        branch = np.abs(r_plus) <= np.abs(r_minus)
    else:  # pragma: no cover - canonical solver import guarantees this path
        branch = np.ones(n_points, dtype=bool)

    branch_counts = (int(np.count_nonzero(branch)), int(np.count_nonzero(~branch)))
    symmetry = _symmetry_diagnostics(
        xy,
        inferred_labels if inferred_labels is not None else None,
        center=(cx, cy),
        reference_axis_deg=reference_axis,
        unassigned_count=unassigned_count,
        center_verified=center_verified,
        azimuthal=azimuthal_trajectory,
    )
    if swap_applied:
        symmetry["branch_leaks"]["global_swap"] = True
        symmetry["flags"].append("global_branch_swap_applied")
    if strict_symmetry and unassigned_count:
        symmetry["flags"].append("partial_branch_labels_excluded")
    ellipticity = float(np.sqrt(max(0.0, 1.0 - ratio * ratio))) if np.isfinite(ratio) else float("nan")
    stderr = {str(name): float(value) for name, value in dict(canonical_result.stderr).items()}
    theta_stderr = stderr.get("theta", float("nan"))
    stderr.setdefault("theta_deg", float(np.degrees(theta_stderr)) if np.isfinite(theta_stderr) else float("nan"))
    bound_flags = {str(name): bool(value) for name, value in dict(canonical_result.bound_flags).items()}
    bound_status = {str(name): value for name, value in dict(canonical_result.bound_status).items()}
    unit = resolved_q_unit
    flags = list(_flags_with_q_unit(APPARENT_FLAGS, unit))
    if branch_counts[0] == 0 or branch_counts[1] == 0:
        flags.append("single_branch_supported")
    if any(bound_flags.values()):
        flags.append("parameter_at_bound")
    if not canonical_result.success:
        flags.append("solver_failed")

    q_scale = _q_to_nm_inverse_scale(unit)
    Ln_minor = float("nan")
    Lz = float("nan")
    if azimuthal_trajectory:
        flags.append("spacing_unavailable_azimuthal_trajectory")
    elif q_scale is not None and b > 0:
        # The reported spacing formulas describe an origin-centred q ellipse.
        # A fitted non-zero centre can represent beam-centre error or a real
        # translation; without an explicit physical model, suppress both
        # derived lengths instead of silently applying the origin formula.
        center_scale = max(1.0, abs(a), abs(b)) if np.isfinite(a) and np.isfinite(b) else 1.0
        center_tol = _SPACING_CENTER_REL_TOL * center_scale
        centered = (
            np.isfinite(cx)
            and np.isfinite(cy)
            and np.hypot(cx, cy) <= center_tol
        )
        if not centered:
            flags.extend(("spacing_unavailable_nonzero_center", "spacing_requires_origin_centered_ellipse_assumption"))
        else:
            Ln_minor = float(2.0 * np.pi / (b * q_scale))
            qz = float(ellipse_radius(np.pi / 2.0, a, b, theta))
            if qz > 0:
                Lz = float(2.0 * np.pi / (qz * q_scale))
            flags.append("spacing_requires_origin_centered_ellipse_assumption")

    residuals = np.asarray(canonical_result.residuals, dtype=float)
    rss = float(np.sum(np.square(residuals))) if residuals.size else float("nan")
    return DoubleEllipseFit(
        a=a,
        b=b,
        theta=theta,
        ellipses=(plus, minus),
        ellipticity=ellipticity,
        axes_ratio=ratio,
        rmse=float(canonical_result.rmse),
        rss=rss,
        n_points=n_points,
        branch_counts=branch_counts,
        success=bool(canonical_result.success),
        message=str(canonical_result.message),
        flags=tuple(flags),
        center=centre,
        stderr=stderr,
        covariance=canonical_result.covariance,
        condition_number=float(canonical_result.condition_number),
        coverage=canonical_result.coverage,
        bound_flags=bound_flags,
        bound_status=bound_status,
        free_names=tuple(canonical_result.free_names),
        parameter_values={
            **values,
            "reference_axis_deg": reference_axis,
            "ellipse_axis_tilt_deg": float(np.degrees(theta)),
        },
        reference_axis_deg=reference_axis,
        q_unit=unit,
        Ln_from_minor_axis_nm=Ln_minor,
        Lz_from_draw_axis_nm=Lz,
        branch_assignment=getattr(canonical_result, "branch_assignment", None),
        candidate_solutions=tuple(getattr(canonical_result, "candidate_solutions", ()) or ()),
        selected_start_index=int(getattr(canonical_result, "selected_start_index", 0)),
        multistart_count=int(getattr(canonical_result, "multistart_count", 1)),
        symmetry=symmetry,
        branch_assignment_indices=np.flatnonzero(keep),
    )


def _write_fit_branch_ids(ridge: RidgeTrack, fit: Any) -> None:
    """Copy an optional fit assignment onto the actual measured points."""

    assignment = _get_field(fit, ("branch_assignment", "branch_assignments"), None)
    if assignment is None:
        return
    if isinstance(assignment, Mapping):
        assignment = _get_field(assignment, ("point_branch", "labels", "branch_id", "values"), None)
    if assignment is None:
        return
    values = np.asarray(assignment).ravel()
    valid_indices = np.flatnonzero(ridge.valid)
    explicit_indices = _get_field(fit, ("branch_assignment_indices",), None)
    if explicit_indices is not None:
        explicit_indices = np.asarray(explicit_indices, dtype=int).ravel()
    if explicit_indices is not None and values.size == explicit_indices.size:
        target_indices = explicit_indices
    elif values.size == len(ridge.points):
        target_indices = np.arange(len(ridge.points), dtype=int)
    elif values.size == valid_indices.size:
        target_indices = valid_indices
    else:
        return
    for index, value in zip(target_indices, values):
        if value is None:
            continue
        try:
            branch_id = int(value)
        except (TypeError, ValueError):
            continue
        if branch_id not in (0, 1):
            continue
        point = ridge.points[int(index)]
        point.branch_id = branch_id
        quadrant = point.metadata.get("quadrant")
        if quadrant in {"QI", "QIII"}:
            point.metadata["quadrant_pair"] = "QI+QIII"
        elif quadrant in {"QII", "QIV"}:
            point.metadata["quadrant_pair"] = "QII+QIV"
        point.metadata["branch_id"] = branch_id


def _assign_symmetric_lobe_branches(
    ridge: RidgeTrack,
    lobes: Sequence[LobeMetrics],
    *,
    reference_axis_deg: float = 0.0,
    center: tuple[float, float] = (0.0, 0.0),
) -> None:
    """Assign fixed opposite-quadrant branch identity to observed points.

    The branch is determined once from the q-centred reference quadrants:
    QI+QIII -> 0 and QII+QIV -> 1.  It is never selected point-by-point by
    whichever ellipse happens to have the smaller residual near a crossing.
    """

    for point in ridge.points:
        if not point.valid or not np.isfinite(point.q):
            continue
        point_x = _get_field(point, ("qx", "x"), None)
        point_y = _get_field(point, ("qy", "y"), None)
        if point_x is not None and point_y is not None:
            relative_angle = float(
                (
                    np.arctan2(
                        float(point_y) - float(center[1]),
                        float(point_x) - float(center[0]),
                    )
                    - np.deg2rad(float(reference_axis_deg))
                )
                % (2.0 * np.pi)
            )
        else:
            relative_angle = float(
                (point.angle - np.deg2rad(float(reference_axis_deg))) % (2.0 * np.pi)
            )
        quadrant = int(np.floor(relative_angle / (0.5 * np.pi))) % 4
        boundary_phase = np.mod(relative_angle, 0.5 * np.pi)
        if min(boundary_phase, 0.5 * np.pi - boundary_phase) < np.deg2rad(2.0):
            point.branch_id = None
            point.metadata["quadrant"] = ("QI", "QII", "QIII", "QIV")[quadrant]
            point.metadata["quadrant_pair"] = (
                "QI+QIII" if quadrant in (0, 2) else "QII+QIV"
            )
            point.metadata["branch_assignment_source"] = "quadrant_boundary_ambiguous"
            point.metadata["symmetry_flags"] = ("quadrant_boundary_ambiguous",)
            continue
        point.branch_id = 0 if quadrant in (0, 2) else 1
        point.metadata["quadrant"] = ("QI", "QII", "QIII", "QIV")[quadrant]
        point.metadata["quadrant_pair"] = "QI+QIII" if point.branch_id == 0 else "QII+QIV"
        point.metadata["branch_assignment_source"] = "reference_quadrant"
        point.metadata["symmetry_flags"] = ("quadrant_pairing_observed",)
        point.metadata["branch_id"] = point.branch_id


fit_symmetric_ellipses = fit_symmetric_double_ellipse
fit_double_ellipse = fit_symmetric_double_ellipse


def _measure_lobe_radial_observables(
    frame: Any,
    qmap: Any,
    q_window: Any,
    lobes: Sequence[LobeMetrics],
    *,
    n_radial_bins: int,
    snr_threshold: float,
    min_coverage: float,
    mask: Any = None,
    cancel_event: Any = None,
) -> tuple[list[RadialProfile], list[RidgePoint]]:
    """Measure radial reflection quantities in each observed lobe sector.

    This companion measurement is intentionally separate from an
    ``azimuthal_peak`` trajectory.  The latter supplies a q-annulus coordinate
    for an angular maximum, while this routine supplies a genuine radial
    q_star (and hence optional spacing proxy) from a narrow, observed angular
    sector.  Invalid/edge peaks are retained with their reason so a caller can
    audit missing support without inventing a value.
    """

    profiles: list[RadialProfile] = []
    peaks: list[RidgePoint] = []
    for lobe in lobes:
        raise_if_cancelled(cancel_event, "lobe-radial:sector")
        if not lobe.valid or not np.isfinite(lobe.angle):
            continue
        fwhm = float(lobe.fwhm) if np.isfinite(lobe.fwhm) and lobe.fwhm > 0.0 else 0.0
        sector_width = max(np.deg2rad(6.0), min(np.deg2rad(30.0), 1.5 * fwhm))
        profile = measure_radial_profile(
            frame,
            qmap,
            float(lobe.angle),
            q_window,
            n_bins=n_radial_bins,
            sector_width=sector_width,
            mask=mask,
        )
        raise_if_cancelled(cancel_event, "lobe-radial:peak")
        peak = _radial_peak(
            profile,
            snr_threshold=snr_threshold,
            min_coverage=min_coverage,
        )
        peak.flags = tuple(dict.fromkeys(peak.flags + ("lobe_radial_peak",)))
        peak.metadata.update(
            flags=peak.flags,
            lobe_angle=float(lobe.angle),
            lobe_sector_width=float(sector_width),
            source_method="radial_peak_in_observed_lobe_sector",
        )
        profiles.append(profile)
        peaks.append(peak)
    raise_if_cancelled(cancel_event, "lobe-radial:complete")
    return profiles, peaks


def measure_observables(
    frame: Any,
    qmap: Any,
    q_window: Any,
    *,
    n_angular_bins: int = 360,
    n_ridge_angles: int = 72,
    n_radial_bins: int = 256,
    fit_ellipse: bool = True,
    ellipse_parameters: Any = None,
    ellipse_residual: str = "sampson",
    ellipse_multistart: int = 7,
    mask: Any = None,
    ridge_method: str = "radial_peak",
    ridge_snr_threshold: float | None = None,
    ridge_min_peak_fraction: float = 0.0,
    ridge_min_coverage: float = 0.0,
    draw_axis_deg: float = 90.0,
    curvature_sigma: float = 2.0,
    curvature_percentile: float = 25.0,
    curvature_normal_step: float = 1.0,
    p4_quality_thresholds: Mapping[str, Any] | None = None,
    cancel_event: Any = None,
) -> ObservableSet:
    """Run the standard angular, lobe, ridge, and ellipse measurement chain."""

    raise_if_cancelled(cancel_event, "observables:start")
    q_unit = _q_unit(qmap)
    angular = measure_angular_spectrum(frame, qmap, q_window, n_bins=n_angular_bins, mask=mask)
    lobes = measure_four_lobe_peaks(
        angular,
        symmetric_refine=True,
        reference_axis_deg=float(draw_axis_deg) - 90.0,
    )
    ridge = measure_radial_ridges(
        frame,
        qmap,
        q_window,
        n_angles=n_ridge_angles,
        n_bins=n_radial_bins,
        lobe_metrics=lobes,
        mask=mask,
        ridge_method=ridge_method,
        ridge_snr_threshold=ridge_snr_threshold,
        ridge_min_peak_fraction=ridge_min_peak_fraction,
        ridge_min_coverage=ridge_min_coverage,
        curvature_sigma=curvature_sigma,
        curvature_percentile=curvature_percentile,
        curvature_normal_step=curvature_normal_step,
        cancel_event=cancel_event,
    )
    lobe_radial_profiles, lobe_radial_peaks = _measure_lobe_radial_observables(
        frame,
        qmap,
        q_window,
        lobes,
        n_radial_bins=n_radial_bins,
        # Keep the shared ridge threshold as the acceptance contract for the
        # companion radial reflection.  A finite q_star is therefore never
        # silently retained below the user-selected SNR floor.
        snr_threshold=(
            float(ridge_snr_threshold)
            if ridge_snr_threshold is not None
            else 2.0
        ),
        min_coverage=ridge_min_coverage,
        mask=mask,
        cancel_event=cancel_event,
    )
    # Azimuthal maxima carry identities from observed q-wise tracks.  Lobe
    # centres are a separate angular summary and must not overwrite those
    # branch IDs (or create an implied symmetric counterpart).
    if str(ridge_method).lower().replace("-", "_") != "azimuthal_peak":
        center_hint, _ = _ellipse_center_hint(ellipse_parameters)
        _assign_symmetric_lobe_branches(
            ridge,
            lobes,
            reference_axis_deg=float(draw_axis_deg) - 90.0,
            center=center_hint,
        )
    ellipse = (
        fit_symmetric_double_ellipse(
            ridge,
            parameters=ellipse_parameters,
            residual=ellipse_residual,
            multistart=ellipse_multistart,
            reference_axis_deg=float(draw_axis_deg) - 90.0,
            q_unit=q_unit,
            strict_symmetry=True,
            cancel_event=cancel_event,
        )
        if fit_ellipse
        else None
    )
    if ellipse is not None:
        _write_fit_branch_ids(ridge, ellipse)
        from .p4_quality import evaluate_p4_ellipse_quality

        ellipse.quality = evaluate_p4_ellipse_quality(
            ridge,
            ellipse,
            thresholds=p4_quality_thresholds,
        )
    phi_app_deg, phi_app_std_deg = apparent_lamellar_tilt(lobes, draw_axis_deg=draw_axis_deg)
    return ObservableSet(
        angular=angular,
        lobes=lobes,
        ridge=ridge,
        ellipse=ellipse,
        phi_app_deg=phi_app_deg,
        phi_app_std_deg=phi_app_std_deg,
        phi_app_mad_deg=phi_app_std_deg,
        draw_axis_deg=float(draw_axis_deg),
        flags=_flags_with_q_unit(APPARENT_FLAGS, q_unit),
        q_unit=q_unit,
        lobe_radial_profiles=lobe_radial_profiles,
        lobe_radial_peaks=lobe_radial_peaks,
    )


__all__ = [
    "APPARENT_FLAGS",
    "AngularSpectrum",
    "LobeMetrics",
    "RadialProfile",
    "RidgePoint",
    "RidgeTrack",
    "EllipseGeometry",
    "DoubleEllipseFit",
    "ObservableSet",
    "measure_angular_spectrum",
    "angular_spectrum",
    "azimuthal_profile",
    "extract_azimuthal_spectrum",
    "measure_four_lobe_peaks",
    "find_four_lobe_peaks",
    "measure_four_peaks",
    "apparent_lamellar_tilt",
    "measure_radial_profile",
    "radial_profile",
    "measure_radial_ridges",
    "radial_ridge",
    "extract_ridge_points",
    "extract_surface_curvature_ridges",
    "ellipse_radius",
    "fit_symmetric_double_ellipse",
    "fit_symmetric_ellipses",
    "fit_double_ellipse",
    "measure_observables",
]
