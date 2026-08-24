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

try:  # scipy is a declared project dependency, but keep imports friendly.
    from scipy.ndimage import gaussian_filter, gaussian_filter1d, map_coordinates
    from scipy.signal import find_peaks
except Exception:  # pragma: no cover - exercised only in a partial install
    gaussian_filter = None
    gaussian_filter1d = None
    map_coordinates = None
    find_peaks = None

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
        )

    def keys(self) -> tuple[str, ...]:
        return (
            "angle", "q", "q_star", "lamellar_spacing", "intensity", "baseline", "snr",
            "radial_fwhm", "azimuthal_fwhm", "area", "coverage", "n_pixels", "valid", "source", "flags",
            "qx", "qy", "method", "curvature", "normal_slope", "support",
            "pixel_y", "pixel_x", "accepted", "reason", "q_unit", "q_star_Ainv", "q_star_nm_inv", "Ln", "Ln_nm",
        )

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


def measure_four_lobe_peaks(
    spectrum: AngularSpectrum | Mapping[str, Any],
    *,
    expected: int = 4,
    min_prominence: float | None = None,
    snr_threshold: float = 2.0,
    min_distance_fraction: float = 0.12,
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
                valid=bool(np.isfinite(snr) and snr >= snr_threshold),
            )
        )
    return sorted(out, key=lambda item: item.angle)


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


def _radial_peak(profile: RadialProfile, snr_threshold: float = 2.0) -> RidgePoint:
    q = np.asarray(profile.q, dtype=float)
    y = np.asarray(profile.intensity, dtype=float)
    finite = np.isfinite(y) & (np.asarray(profile.counts) > 0)
    if np.count_nonzero(finite) < 3:
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
            flags=APPARENT_FLAGS + ("low_coverage",),
            q_unit=profile.q_unit,
        )
    baseline = float(np.nanpercentile(y[finite], 10.0))
    noise = _robust_noise(y[finite] - baseline)
    if not np.isfinite(noise) or noise <= np.finfo(float).eps:
        noise = max(float(np.nanstd(y[finite])), np.finfo(float).eps)
    smooth = y.copy()
    if gaussian_filter1d is not None and np.count_nonzero(finite) > 8:
        fill = np.interp(q, q[finite], y[finite])
        smooth = gaussian_filter1d(fill, 1.0, mode="nearest")
    index = int(np.nanargmax(np.where(finite, smooth, np.nan)))
    peak_q, peak_intensity = _quadratic_peak(q, smooth, index)
    snr = (peak_intensity - baseline) / noise if noise else float("inf")
    fwhm = _linear_fwhm(q, smooth, index, baseline)
    area = float(np.trapezoid(np.maximum(y[finite] - baseline, 0.0), q[finite])) if np.count_nonzero(finite) > 1 else 0.0
    flags = list(_flags_with_q_unit(APPARENT_FLAGS, profile.q_unit))
    valid = bool(np.isfinite(snr) and snr >= snr_threshold and np.isfinite(peak_q))
    if not valid:
        flags.append("low_snr")
    coverage = float(np.sum(profile.counts) / max(1, np.sum(profile.candidate_counts)))
    q_star = float(peak_q)
    return RidgePoint(
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
        coverage=coverage,
        n_pixels=int(np.sum(profile.counts)),
        valid=valid,
        flags=tuple(flags),
        q_unit=profile.q_unit,
    )


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
        qx=q * np.cos(angle),
        qy=q * np.sin(angle),
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
        baseline=baseline,
        noise=noise,
        q_unit=_q_unit(qmap),
    )


def _curvature_point_for_sector(
    field: _CurvatureField,
    requested_angle: float,
    sector_width: float,
    snr_threshold: float,
) -> RidgePoint:
    """Select and refine the strongest principal-curvature ridge in a sector."""

    sector = _wrap_distance(field.angle, requested_angle) <= float(sector_width) / 2.0
    candidate_pixels = field.candidate_domain & sector
    valid_pixels = candidate_pixels & field.valid
    coverage = float(np.count_nonzero(valid_pixels) / max(1, np.count_nonzero(candidate_pixels)))
    accepted = field.ridge_candidate & sector
    if not np.any(accepted):
        return RidgePoint(
            angle=float(requested_angle),
            q=float("nan"),
            q_star=float("nan"),
            lamellar_spacing=float("nan"),
            coverage=coverage,
            n_pixels=int(np.count_nonzero(valid_pixels)),
            valid=False,
            flags=APPARENT_FLAGS + ("detector_pixel_principal_curvature", "no_curvature_candidate"),
            method="surface_curvature",
            accepted=False,
            reason="no_curvature_candidate",
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
        q_unit=field.q_unit,
    )


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
    ridge_method: str = "radial_peak",
    method: str | None = None,
    mask: Any = None,
    curvature_sigma: float = 2.0,
    curvature_percentile: float = 25.0,
    curvature_normal_step: float = 1.0,
) -> RidgeTrack:
    """Measure a radial peak for every requested, directly observed sector.

    If ``angles`` is omitted, uniformly spaced sectors cover the complete
    angular range.  Crucially, this routine never copies or mirrors a point
    from a counterpart quadrant; masked and low-SNR sectors stay invalid.
    """

    ridge_method = str(method if method is not None else ridge_method).lower().replace("-", "_")
    if ridge_method not in {"radial_peak", "surface_curvature", "curvature"}:
        raise ValueError("ridge_method must be 'radial_peak' or 'surface_curvature'")
    curvature_mode = ridge_method in {"surface_curvature", "curvature"}
    if angles is None:
        # Lobe locations are annotations, not a substitute for a densely
        # sampled trajectory.  A four-point-only track cannot constrain the
        # five geometric parameters of the shared-centre double ellipse.
        angles = np.linspace(-np.pi, np.pi, int(n_angles), endpoint=False).tolist()
    angle_array = np.asarray(angles, dtype=float).ravel()
    q_unit = _q_unit(qmap)
    if not angle_array.size:
        return RidgeTrack(
            points=[],
            angles=angle_array,
            q=np.array([]),
            valid=np.array([], dtype=bool),
            coverage=np.array([]),
            q_unit=q_unit,
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
    points: list[RidgePoint] = []
    for angle in angle_array:
        if curvature_mode:
            assert curvature_field is not None
            point = _curvature_point_for_sector(
                curvature_field,
                float(angle),
                float(sector_width),
                float(snr_threshold),
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
            point = _radial_peak(radial, snr_threshold=snr_threshold)
        if lobe_metrics:
            distances = [_wrap_distance(float(angle), lobe.angle) for lobe in lobe_metrics]
            nearest = int(np.argmin(distances))
            nearest_lobe = lobe_metrics[nearest]
            point.azimuthal_fwhm = float(nearest_lobe.fwhm)  # type: ignore[misc]
        points.append(point)
    q_values = np.asarray([point.q for point in points], dtype=float)
    valid = np.asarray([point.valid for point in points], dtype=bool)
    coverage = np.asarray([point.coverage for point in points], dtype=float)
    track_flags = APPARENT_FLAGS + (("detector_pixel_principal_curvature",) if curvature_mode else tuple())
    track_flags = _flags_with_q_unit(track_flags, q_unit)
    return RidgeTrack(
        points=points,
        angles=angle_array,
        q=q_values,
        valid=valid,
        coverage=coverage,
        flags=track_flags,
        q_unit=q_unit,
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
    """Read optional 0/1 branch labels without requiring them for fitting."""

    source = _ridge_source(points)
    if isinstance(source, Mapping):
        value = _get_field(source, ("component", "components", "labels", "branch"))
        if value is None:
            return None
        array = np.asarray(value)
        if array.ndim == 0:
            return None
        return array.ravel()
    if isinstance(source, np.ndarray) or isinstance(source, (str, bytes)):
        return None
    try:
        sequence = list(source)
    except TypeError:
        return None
    values: list[Any] = []
    found = False
    for point in sequence:
        component = _get_field(point, ("component", "branch", "label"))
        if component is None:
            metadata = _get_field(point, ("metadata",), {})
            component = _get_field(metadata, ("component", "branch", "label"))
        if component is None:
            values.append(np.nan)
        else:
            values.append(component)
            found = True
    if not found:
        return None
    return np.asarray(values)


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
    reference_axis_deg: float = 0.0,
    q_unit: str | None = None,
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

    x, y, finite_xy = _ridge_xy(points)
    radius = np.hypot(x, y)
    keep = finite_xy & np.isfinite(radius) & (radius > 0.0)
    xy = np.column_stack((x[keep], y[keep]))
    n_points = int(xy.shape[0])
    if n_points < 5:
        return _empty_double_ellipse(
            n_points,
            "at least five finite nonzero ridge points are required",
            "insufficient_points",
            q_unit=resolved_q_unit,
        )

    inferred_labels = _ridge_components(points) if labels is None else np.asarray(labels)
    if inferred_labels is not None:
        inferred_labels = np.asarray(inferred_labels).ravel()
        if inferred_labels.size != x.size:
            raise ValueError(f"labels must contain one value per ridge point ({x.size})")
        try:
            inferred_labels = inferred_labels.astype(int)
        except (TypeError, ValueError) as exc:
            raise ValueError("labels must contain only 0 or 1") from exc
        if not np.all(np.isin(inferred_labels, (0, 1))):
            raise ValueError("labels must contain only 0 or 1")
        inferred_labels = inferred_labels[keep]

    point_weights = _ridge_weights(points) if weights is None else np.asarray(weights, dtype=float).ravel()
    if point_weights is not None:
        if point_weights.size != x.size:
            raise ValueError(f"weights must contain one value per ridge point ({x.size})")
        point_weights = point_weights[keep]

    source_parameters = parameters if parameters is not None else (params if params is not None else initial)
    canonical_parameters = _initial_to_canonical(source_parameters)
    if residual_kind is not None:
        residual = residual_kind
    try:
        canonical_result = _fit_canonical_symmetric_ellipses(
            xy,
            parameters=canonical_parameters,
            residual=residual,
            loss=robust_loss,
            f_scale=f_scale,
            labels=inferred_labels,
            weights=point_weights,
            max_nfev=max_nfev,
            config=config,
            reference_axis_deg=reference_axis_deg,
        )
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
    if q_scale is not None and b > 0:
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
    )


fit_symmetric_ellipses = fit_symmetric_double_ellipse
fit_double_ellipse = fit_symmetric_double_ellipse


def measure_observables(
    frame: Any,
    qmap: Any,
    q_window: Any,
    *,
    n_angular_bins: int = 360,
    n_ridge_angles: int = 72,
    n_radial_bins: int = 256,
    fit_ellipse: bool = True,
    mask: Any = None,
    ridge_method: str = "radial_peak",
    draw_axis_deg: float = 90.0,
    curvature_sigma: float = 2.0,
    curvature_percentile: float = 25.0,
    curvature_normal_step: float = 1.0,
) -> ObservableSet:
    """Run the standard angular, lobe, ridge, and ellipse measurement chain."""

    q_unit = _q_unit(qmap)
    angular = measure_angular_spectrum(frame, qmap, q_window, n_bins=n_angular_bins, mask=mask)
    lobes = measure_four_lobe_peaks(angular)
    ridge = measure_radial_ridges(
        frame,
        qmap,
        q_window,
        n_angles=n_ridge_angles,
        n_bins=n_radial_bins,
        lobe_metrics=lobes,
        mask=mask,
        ridge_method=ridge_method,
        curvature_sigma=curvature_sigma,
        curvature_percentile=curvature_percentile,
        curvature_normal_step=curvature_normal_step,
    )
    ellipse = (
        fit_symmetric_double_ellipse(
            ridge,
            reference_axis_deg=float(draw_axis_deg) - 90.0,
            q_unit=q_unit,
        )
        if fit_ellipse
        else None
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
