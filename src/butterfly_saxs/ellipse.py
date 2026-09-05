"""Ellipse geometry and robust fitting for q-space SAXS ridge points.

The measured ridge coordinates are treated as points in the calibrated
``(q_x, q_y)`` plane.  A single ellipse uses a shared centre, a major
semi-axis ``a``, an axis ratio ``b/a`` and a rotation ``theta``.  The
butterfly model is the mirror-symmetric pair at ``+theta`` and ``-theta``;
both ellipses share centre and semi-axes.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import numpy as np
from scipy.optimize import OptimizeResult, brentq, least_squares, minimize_scalar
from scipy.spatial import ConvexHull

from .models import RidgePoint
from .parameters import ParameterSet, ParameterSpec, default_ellipse_parameters
from .cancellation import raise_if_cancelled

ResidualKind = Literal["sampson", "geometric"]


@dataclass(frozen=True)
class EllipseGeometry:
    """Numerical geometry of one ellipse in q space."""

    cx: float
    cy: float
    a: float
    axis_ratio: float
    theta: float = 0.0

    def __post_init__(self) -> None:
        values = (self.cx, self.cy, self.a, self.axis_ratio, self.theta)
        if not all(np.isfinite(values)):
            raise ValueError("ellipse geometry must be finite")
        if self.a <= 0:
            raise ValueError("a must be greater than zero")
        if not 0 < self.axis_ratio <= 1:
            raise ValueError("axis_ratio must be in (0, 1]")

    @property
    def b(self) -> float:
        return self.a * self.axis_ratio

    @property
    def theta_rad(self) -> float:
        return self.theta

    @property
    def theta_deg(self) -> float:
        return math.degrees(self.theta)

    @property
    def center(self) -> tuple[float, float]:
        return self.cx, self.cy

    @property
    def eccentricity(self) -> float:
        """Ellipse eccentricity ``sqrt(1-(b/a)^2)`` used in Grubb Table 3."""

        return math.sqrt(max(0.0, 1.0 - self.axis_ratio * self.axis_ratio))

    @property
    def ellipticity(self) -> float:
        """Compatibility name for the paper's eccentricity-valued column."""

        return self.eccentricity

    def point(self, phi: Any) -> np.ndarray:
        """Return q-space points for parametric ellipse angle(s) ``phi``."""

        phi = np.asarray(phi, dtype=float)
        c, s = np.cos(self.theta), np.sin(self.theta)
        u = self.a * np.cos(phi)
        v = self.b * np.sin(phi)
        x = self.cx + c * u - s * v
        y = self.cy + s * u + c * v
        return np.stack((x, y), axis=-1)

    @classmethod
    def from_values(cls, values: Mapping[str, Any]) -> "EllipseGeometry":
        try:
            return cls(
                cx=float(values["cx"]),
                cy=float(values["cy"]),
                a=float(values["a"]),
                axis_ratio=float(values["axis_ratio"]),
                theta=float(values.get("theta", 0.0)),
            )
        except KeyError as exc:
            raise ValueError(f"missing ellipse parameter: {exc.args[0]}") from exc


@dataclass(frozen=True)
class CoverageMetrics:
    """Simple geometry coverage diagnostics for a fitted track."""

    n_points: int
    angular_span: float
    angular_coverage: float
    radial_rms: float
    convex_hull_area: float
    components: int = 1

    def __getitem__(self, key: str) -> float | int:
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def as_dict(self) -> dict[str, float | int]:
        return {
            "n_points": self.n_points,
            "angular_span": self.angular_span,
            "angular_coverage": self.angular_coverage,
            "radial_rms": self.radial_rms,
            "convex_hull_area": self.convex_hull_area,
            "components": self.components,
        }


@dataclass
class EllipseFitResult:
    """Result and uncertainty/identifiability diagnostics of a fit."""

    parameters: ParameterSet
    values: dict[str, float]
    residuals: np.ndarray
    success: bool
    message: str
    cost: float
    optimality: float
    nfev: int
    covariance: np.ndarray | None
    stderr: dict[str, float]
    condition: float
    bound_flags: dict[str, bool]
    bound_status: dict[str, str | None]
    coverage: CoverageMetrics
    free_names: tuple[str, ...] = ()
    jacobian: np.ndarray | None = None
    scipy_result: OptimizeResult | None = field(default=None, repr=False)
    model: str = "ellipse"
    reference_axis: float = 0.0
    # The symmetric solver records its deterministic multistart audit trail.
    # These defaults keep the single-ellipse result/API unchanged.
    candidate_solutions: tuple[dict[str, Any], ...] = ()
    selected_start_index: int = 0
    multistart_count: int = 1
    branch_assignment: np.ndarray | None = None

    @property
    def parameter_set(self) -> ParameterSet:
        return self.parameters

    @property
    def best_values(self) -> dict[str, float]:
        return dict(self.values)

    @property
    def residual(self) -> np.ndarray:
        return self.residuals

    @property
    def rmse(self) -> float:
        return float(np.sqrt(np.mean(np.square(self.residuals)))) if self.residuals.size else float("nan")

    @property
    def reduced_chi_square(self) -> float:
        dof = max(self.residuals.size - len(self.free_names), 1)
        return float(2 * self.cost / dof)

    @property
    def covariance_names(self) -> tuple[str, ...]:
        return self.free_names

    @property
    def covariance_matrix(self) -> np.ndarray | None:
        return self.covariance

    @property
    def condition_number(self) -> float:
        return self.condition

    @property
    def x(self) -> np.ndarray:
        return self.parameters.free_vector()

    @property
    def at_bound(self) -> dict[str, bool]:
        return self.bound_flags

    @property
    def coverage_fraction(self) -> float:
        return self.coverage.angular_coverage

    @property
    def reference_axis_deg(self) -> float:
        return math.degrees(self.reference_axis)


def _coerce_points(points: Any) -> np.ndarray:
    if isinstance(points, Mapping):
        nested = points.get("points", points.get("ridges", points.get("ridge_points")))
        if nested is not None:
            points = nested
        elif "qx" in points and "qy" in points:
            points = np.column_stack((np.asarray(points["qx"]), np.asarray(points["qy"])))
    if isinstance(points, np.ndarray):
        array = np.asarray(points, dtype=float)
    elif isinstance(points, Sequence) and not isinstance(points, (str, bytes)):
        rows: list[tuple[float, float]] = []
        for point in points:
            if isinstance(point, RidgePoint):
                rows.append((point.qx, point.qy))
                continue
            if isinstance(point, Mapping):
                x = point.get("qx", point.get("q_x", point.get("x")))
                y = point.get("qy", point.get("q_y", point.get("y")))
                if x is None or y is None:
                    q = point.get("q", point.get("q_star", point.get("radius")))
                    angle = point.get("angle", point.get("azimuth", point.get("phi")))
                    if q is not None and angle is not None:
                        x, y = float(q) * math.cos(float(angle)), float(q) * math.sin(float(angle))
                if x is not None and y is not None:
                    rows.append((float(x), float(y)))
                    continue
            x = getattr(point, "qx", getattr(point, "x", None))
            y = getattr(point, "qy", getattr(point, "y", None))
            if x is None or y is None:
                q = getattr(point, "q", getattr(point, "q_star", None))
                angle = getattr(point, "angle", getattr(point, "azimuth", None))
                if q is not None and angle is not None:
                    x, y = float(q) * math.cos(float(angle)), float(q) * math.sin(float(angle))
            if x is not None and y is not None:
                rows.append((float(x), float(y)))
                continue
            # Preserve ordinary ``[(x, y), ...]`` and ``(x_array, y_array)``
            # inputs for the vectorized conversion below.
            rows = []
            break
        array = np.asarray(rows, dtype=float) if rows else np.asarray(points, dtype=float)
    else:
        array = np.asarray(points, dtype=float)
    if array.ndim != 2:
        raise ValueError("points must be a 2D array with shape (n, 2)")
    if array.shape[1] != 2 and array.shape[0] == 2:
        array = array.T
    if array.shape[1] != 2:
        raise ValueError("points must have shape (n, 2)")
    if array.shape[0] < 1:
        raise ValueError("at least one point is required")
    if not np.all(np.isfinite(array)):
        raise ValueError("points must contain only finite values")
    return array


def _coerce_weights(weights: Any, n: int) -> np.ndarray:
    if weights is None:
        return np.ones(n, dtype=float)
    result = np.asarray(weights, dtype=float)
    if result.ndim != 1 or result.size != n:
        raise ValueError(f"weights must have shape ({n},)")
    if not np.all(np.isfinite(result)) or np.any(result <= 0):
        raise ValueError("weights must be finite and positive")
    return result


def _values_from_parameters(parameters: Any) -> dict[str, float]:
    if isinstance(parameters, EllipseGeometry):
        return {
            "cx": parameters.cx,
            "cy": parameters.cy,
            "a": parameters.a,
            "axis_ratio": parameters.axis_ratio,
            "b": parameters.b,
            "theta": parameters.theta,
        }
    if isinstance(parameters, ParameterSet):
        return parameters.resolve()
    if isinstance(parameters, Mapping):
        if "center_x" in parameters and "cx" not in parameters:
            parameters = dict(parameters)
            parameters["cx"] = parameters.pop("center_x")
        if "center_y" in parameters and "cy" not in parameters:
            parameters = dict(parameters)
            parameters["cy"] = parameters.pop("center_y")
        if "ratio" in parameters and "axis_ratio" not in parameters:
            parameters = dict(parameters)
            parameters["axis_ratio"] = parameters.pop("ratio")
        if "b" in parameters and "axis_ratio" not in parameters and "a" in parameters:
            parameters = dict(parameters)
            parameters["axis_ratio"] = float(parameters["b"]) / float(parameters["a"])
        return ParameterSet({key: value for key, value in parameters.items()}).resolve()
    sequence = np.asarray(parameters, dtype=float)
    if sequence.ndim == 1 and sequence.size in (4, 5):
        cx, cy, a, ratio = sequence[:4]
        theta = sequence[4] if sequence.size == 5 else 0.0
        return {"cx": float(cx), "cy": float(cy), "a": float(a), "axis_ratio": float(ratio),
                "b": float(a * ratio), "theta": float(theta)}
    raise TypeError("parameters must be EllipseGeometry, ParameterSet, mapping or a 4/5-value sequence")


def _geometry(parameters: Any) -> EllipseGeometry:
    return EllipseGeometry.from_values(_values_from_parameters(parameters))


def ellipse_implicit(points: Any, parameters: Any) -> np.ndarray:
    """Return the normalized implicit ellipse equation ``F(x, y)``."""

    points = _coerce_points(points)
    geometry = _geometry(parameters)
    dx = points[:, 0] - geometry.cx
    dy = points[:, 1] - geometry.cy
    c, s = math.cos(geometry.theta), math.sin(geometry.theta)
    u = c * dx + s * dy
    v = -s * dx + c * dy
    return (u / geometry.a) ** 2 + (v / geometry.b) ** 2 - 1.0


def ellipse_sampson_residuals(points: Any, parameters: Any, *, weights: Any = None) -> np.ndarray:
    """Signed q-space Sampson distance to an ellipse.

    The implicit equation is normalized by the Cartesian gradient, giving a
    first-order geometric distance in the same q units as the input points.
    ``weights`` are applied as square-root weights for least-squares fitting.
    """

    points = _coerce_points(points)
    geometry = _geometry(parameters)
    dx = points[:, 0] - geometry.cx
    dy = points[:, 1] - geometry.cy
    c, s = math.cos(geometry.theta), math.sin(geometry.theta)
    u = c * dx + s * dy
    v = -s * dx + c * dy
    aa, bb = geometry.a * geometry.a, geometry.b * geometry.b
    f = u * u / aa + v * v / bb - 1.0
    gx = 2.0 * (u * c / aa - v * s / bb)
    gy = 2.0 * (u * s / aa + v * c / bb)
    denominator = np.maximum(np.hypot(gx, gy), np.finfo(float).tiny)
    result = f / denominator
    if weights is not None:
        result = result * np.sqrt(_coerce_weights(weights, points.shape[0]))
    return result


def _closest_ellipse_distance_scalar(u: float, v: float, a: float, b: float,
                                     *, grid_size: int = 256) -> float:
    """Return the unsigned global distance from ``(u, v)`` to an ellipse.

    A Newton iteration on the stationary angle is attractive because it is
    short and vectorizes well, but it can converge to the *farthest* or a
    saddle stationary point for a thin ellipse.  The implementation here
    brackets every sign-changing stationary root on a periodic grid and
    refines each bracket with Brent's method.  It also checks the lowest grid
    basins with a bounded scalar minimization, which covers the degenerate
    axis cases where a stationary root is tangent to zero and therefore does
    not change sign.  All values are scaled by ``a`` before evaluating the
    equation; this keeps ratios such as 0.02 well-conditioned.

    The scalar routine is intentionally deterministic.  ``grid_size`` is
    modest compared with a detector image but large enough to resolve the
    narrow minor-axis basin of the flat ellipses encountered in SAXS.
    """

    a = float(a)
    b = float(b)
    if not (np.isfinite(u) and np.isfinite(v) and a > 0.0 and 0.0 < b <= a):
        return float("nan")
    # Dimensionless local coordinates avoid a^2/b^2 overflow and give the
    # root function a consistent scale for q values in either unit system.
    ratio = b / a
    uu, vv = float(u / a), float(v / a)
    if ratio >= 1.0 - 1e-14:
        return float(a * abs(math.hypot(uu, vv) - 1.0))

    n_grid = max(64, int(grid_size))
    phi = np.linspace(-math.pi, math.pi, n_grid + 1, dtype=float)
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)

    def stationarity(angle: float) -> float:
        sin_value = math.sin(angle)
        cos_value = math.cos(angle)
        return (
            (1.0 - ratio * ratio) * sin_value * cos_value
            - uu * sin_value
            + ratio * vv * cos_value
        )

    def squared_distance(angle: float) -> float:
        du = math.cos(angle) - uu
        dv = ratio * math.sin(angle) - vv
        return du * du + dv * dv

    values = (
        (1.0 - ratio * ratio) * sin_phi * cos_phi
        - uu * sin_phi
        + ratio * vv * cos_phi
    )
    candidates: list[float] = []
    zero_tol = 2e-14 * max(1.0, abs(uu), abs(vv), ratio)
    for index in range(n_grid):
        left_value = float(values[index])
        right_value = float(values[index + 1])
        left_angle = float(phi[index])
        right_angle = float(phi[index + 1])
        if abs(left_value) <= zero_tol:
            candidates.append(left_angle)
        if left_value * right_value < 0.0:
            try:
                candidates.append(
                    float(brentq(stationarity, left_angle, right_angle,
                                 xtol=1e-14, rtol=4.0 * np.finfo(float).eps,
                                 maxiter=100))
                )
            except (ValueError, RuntimeError):
                # A grid endpoint can be numerically indistinguishable from a
                # pole for extremely flat ratios.  The local basin checks
                # below still provide a valid fallback in that case.
                pass
    if abs(float(values[-1])) <= zero_tol:
        candidates.append(float(phi[-1]))

    # Include all coarse local minima.  For an exact-axis point the nearest
    # stationary root may touch zero without a sign change; minimizing its
    # surrounding basin recovers the correct endpoint.  The explicit axes
    # also make the centre case well-defined (the minor endpoint is nearest).
    coarse = np.asarray([squared_distance(float(item)) for item in phi[:-1]], dtype=float)
    for index in range(n_grid):
        previous = coarse[(index - 1) % n_grid]
        current = coarse[index]
        following = coarse[(index + 1) % n_grid]
        if current <= previous and current <= following:
            left = float(phi[index] - (phi[1] - phi[0]))
            right = float(phi[index] + (phi[1] - phi[0]))
            try:
                refined = minimize_scalar(
                    squared_distance,
                    bounds=(left, right),
                    method="bounded",
                    options={"xatol": 2e-14, "maxiter": 100},
                )
                if refined.success and np.isfinite(refined.x):
                    candidates.append(float(refined.x))
            except (ValueError, RuntimeError):
                pass
    candidates.extend((0.0, 0.5 * math.pi, math.pi, -0.5 * math.pi))
    if not candidates:
        return float(a * math.sqrt(float(np.min(coarse))))
    distance_squared = min(squared_distance(float(angle)) for angle in candidates)
    return float(a * math.sqrt(max(0.0, distance_squared)))


def _closest_ellipse_distances(u: Any, v: Any, a: float, b: float) -> np.ndarray:
    """Vectorized global closest distances for a non-circular ellipse.

    In first-quadrant coordinates ``x=abs(u/a)`` and ``y=abs(v/a)``, the
    Lagrange multiplier of the constrained Euclidean projection satisfies

    ``(x/(1+lambda))**2 + (r*y/(r**2+lambda))**2 = 1``.

    The equation is monotone on the exterior interval ``[0, inf)`` and on
    the interior interval ``(-r**2, 0)`` whenever ``y != 0``.  Bisection on
    those intervals is both faster and more reliable than a per-point Newton
    solve.  Exact points on the major-axis symmetry line are handled by the
    closed-form stationary candidates (including the minor-axis branch, which
    is the case that defeated the former Newton routine).  Everything is
    normalized by ``a`` before the solve, so a ratio of .02 remains well
    scaled even when q values are large or small.
    """

    u_array, v_array = np.broadcast_arrays(
        np.asarray(u, dtype=float),
        np.asarray(v, dtype=float),
    )
    original_shape = u_array.shape
    u_flat = u_array.ravel()
    v_flat = v_array.ravel()
    a = float(a)
    b = float(b)
    if not (a > 0.0 and 0.0 < b <= a):
        return np.full(original_shape, np.nan, dtype=float)
    ratio = b / a
    x = np.abs(u_flat / a)
    y = np.abs(v_flat / a)
    if ratio >= 1.0 - 1e-14:
        return (a * np.abs(np.hypot(x, y) - 1.0)).reshape(original_shape)

    result = np.full(x.shape, np.nan, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    if not np.any(finite):
        return result.reshape(original_shape)
    ratio_sq = ratio * ratio
    # A value exactly on the symmetry line has a singular multiplier at the
    # minor axis.  Treat only numerical zero as this analytic case; a genuine
    # off-axis point, however small, has a regular root in (-r^2, 0).
    axis_tol = 1.0e-8 * np.maximum(1.0, x)
    axis_case = finite & (y <= axis_tol)
    regular = finite & ~axis_case

    if np.any(axis_case):
        axis_indices = np.flatnonzero(axis_case)
        exact_axis = axis_indices[y[axis_indices] == 0.0]
        near_axis = axis_indices[y[axis_indices] > 0.0]
        if near_axis.size:
            # The regular multiplier has two roots on either side of the
            # -r^2 pole as y approaches zero.  The scalar global fallback is
            # used only for this numerically tiny subset so the vectorized
            # bisection path remains fast for detector-sized point clouds.
            result[near_axis] = np.asarray(
                [
                    _closest_ellipse_distance_scalar(
                        float(u_flat[index]),
                        float(v_flat[index]),
                        a,
                        b,
                    )
                    for index in near_axis
                ],
                dtype=float,
            )
        if not exact_axis.size:
            exact_axis = np.asarray([], dtype=int)
        xx = x[exact_axis]
        denominator = max(1.0 - ratio_sq, np.finfo(float).eps)
        cosine = np.clip(xx / denominator, 0.0, 1.0)
        phi = np.arccos(cosine)
        off_axis = np.hypot(np.cos(phi) - xx, ratio * np.sin(phi))
        off_axis_valid = xx <= denominator
        off_axis = np.where(off_axis_valid, off_axis, np.inf)
        major = np.abs(1.0 - xx)
        minor = np.hypot(xx, ratio)
        if exact_axis.size:
            result[exact_axis] = a * np.minimum(np.minimum(off_axis, major), minor)

    def constraint(multiplier: np.ndarray, xx: np.ndarray, yy: np.ndarray) -> np.ndarray:
        first = np.divide(xx, 1.0 + multiplier)
        second = np.divide(ratio * yy, ratio_sq + multiplier)
        return first * first + second * second - 1.0

    outside = regular & ((x * x) + (y / ratio) ** 2 >= 1.0)
    if np.any(outside):
        xx = x[outside]
        yy = y[outside]
        lower = np.zeros_like(xx)
        upper = np.maximum(1.0, 2.0 * np.maximum(xx, ratio * yy) + 1.0)
        for _ in range(32):
            need_expand = constraint(upper, xx, yy) > 0.0
            if not np.any(need_expand):
                break
            upper = np.where(need_expand, upper * 2.0, upper)
        for _ in range(70):
            middle = 0.5 * (lower + upper)
            positive = constraint(middle, xx, yy) > 0.0
            lower = np.where(positive, middle, lower)
            upper = np.where(positive, upper, middle)
        multiplier = 0.5 * (lower + upper)
        closest_x = xx / (1.0 + multiplier)
        closest_y = ratio_sq * yy / (ratio_sq + multiplier)
        result[outside] = a * np.hypot(closest_x - xx, closest_y - yy)

    inside = regular & ~outside
    if np.any(inside):
        xx = x[inside]
        yy = y[inside]
        # ``yy != 0`` guarantees constraint(-r²+) -> +infinity, while
        # constraint(0) < 0 for an interior point.  The small offset keeps
        # the denominator finite for the first bisection evaluation.
        lower = np.full_like(xx, -ratio_sq)
        # Keep the lower endpoint close enough to the pole that even a tiny
        # but genuine off-axis coordinate brackets the monotone root.  Scaling
        # this offset with r^2 is essential: an absolute 16-eps offset is too
        # large for y~1e-14 and would leave F(lower)<0 instead of +infinity.
        lower += 16.0 * np.finfo(float).eps * max(ratio_sq, np.finfo(float).tiny)
        upper = np.zeros_like(xx)
        for _ in range(80):
            middle = 0.5 * (lower + upper)
            positive = constraint(middle, xx, yy) > 0.0
            lower = np.where(positive, middle, lower)
            upper = np.where(positive, upper, middle)
        multiplier = 0.5 * (lower + upper)
        closest_x = xx / (1.0 + multiplier)
        closest_y = ratio_sq * yy / (ratio_sq + multiplier)
        result[inside] = a * np.hypot(closest_x - xx, closest_y - yy)

    return result.reshape(original_shape)


def ellipse_geometric_residuals(points: Any, parameters: Any, *, weights: Any = None,
                                 max_iter: int = 32) -> np.ndarray:
    """Signed true closest-point distance to an ellipse.

    ``max_iter`` is retained as a compatibility argument from the former
    Newton implementation.  The current global bracket/refinement solver does
    not use it: unlike an unconstrained Newton step, it cannot silently select
    a wrong stationary point for a very flat ellipse.
    """

    del max_iter
    points = _coerce_points(points)
    geometry = _geometry(parameters)
    dx = points[:, 0] - geometry.cx
    dy = points[:, 1] - geometry.cy
    c, s = math.cos(geometry.theta), math.sin(geometry.theta)
    u = c * dx + s * dy
    v = -s * dx + c * dy
    a, b = geometry.a, geometry.b
    distance = _closest_ellipse_distances(u, v, a, b)
    implicit = ellipse_implicit(points, geometry)
    result = np.copysign(distance, implicit)
    if weights is not None:
        result = result * np.sqrt(_coerce_weights(weights, points.shape[0]))
    return result


# Short names are convenient in notebooks and make the residual choice clear.
sampson_residuals = ellipse_sampson_residuals
geometric_residuals = ellipse_geometric_residuals
ellipse_residuals = ellipse_sampson_residuals
ellipse_sampson_residual = ellipse_sampson_residuals
ellipse_geometric_residual = ellipse_geometric_residuals


def _canonical_parameter_mapping(parameters: Mapping[str, Any] | ParameterSet) -> dict[str, Any]:
    source = parameters._parameters if isinstance(parameters, ParameterSet) else parameters
    result: dict[str, Any] = {}
    aliases = {"center_x": "cx", "center_y": "cy", "x0": "cx", "y0": "cy", "ratio": "axis_ratio"}
    for key, value in source.items():
        canonical = aliases.get(key, key)
        if canonical in result and key != canonical:
            raise ValueError(f"duplicate ellipse parameter aliases for {canonical!r}")
        result[canonical] = value
    # Keep theta in radians internally, but accept a degree-valued editor
    # field as a first-class input.  Bounds and tied expressions are converted
    # as well, so a UI can vary theta_deg without knowing the solver units.
    if "theta_deg" in source and "theta" not in source:
        degree_spec = source["theta_deg"]
        if isinstance(degree_spec, Mapping):
            data = dict(degree_spec)
            data.setdefault("name", "theta_deg")
            degree_spec = ParameterSpec(**data)
        if isinstance(degree_spec, ParameterSpec):
            expr = None if degree_spec.expr is None else f"({degree_spec.expr})*pi/180"
            value = degree_spec.value * math.pi / 180 if expr is None else degree_spec.value
            lower = None if degree_spec.min is None else degree_spec.min * math.pi / 180
            upper = None if degree_spec.max is None else degree_spec.max * math.pi / 180
            result["theta"] = ParameterSpec(value=value, min=lower, max=upper,
                                             vary=degree_spec.vary, expr=expr, name="theta")
        else:
            result["theta"] = float(degree_spec) * math.pi / 180
        result.pop("theta_deg", None)
    elif "theta_deg" in source and "theta" in source:
        # A default set has a tied theta_deg field; an explicit second input is
        # ambiguous and should be surfaced instead of silently discarded.
        degree_spec = source["theta_deg"]
        if not (isinstance(degree_spec, ParameterSpec) and degree_spec.expr is not None):
            raise ValueError("supply either theta (radians) or theta_deg (degrees), not both")
        result.pop("theta_deg", None)
    return result


def _initial_guess(points: np.ndarray) -> tuple[float, float, float, float, float]:
    center = np.median(points, axis=0)
    centered = points - center
    if points.shape[0] > 2:
        covariance = np.cov(centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        major = eigenvectors[:, order[0]]
        spread = np.maximum(eigenvalues[order], np.finfo(float).eps)
        a = math.sqrt(2.0 * float(spread[0]))
        ratio = math.sqrt(float(spread[1] / spread[0]))
        theta = math.atan2(float(major[1]), float(major[0]))
    else:
        scale = max(float(np.max(np.ptp(points, axis=0))), 1.0)
        a, ratio, theta = scale, 0.7, 0.0
    a = max(a, np.finfo(float).eps * 10)
    # Real butterfly wings can be nearly line-like.  Keeping the historical
    # 0.05 floor silently excludes the .02/.05 synthetic and beamline cases,
    # so use a small positive numerical floor and let editable bounds decide
    # what a caller considers physically plausible.
    ratio = float(np.clip(ratio, 0.005, 1.0))
    # Major-axis orientation is periodic by pi; this range avoids a redundant
    # pair of equivalent solutions while retaining both signs for the pair.
    theta = (theta + math.pi / 2) % math.pi - math.pi / 2
    return float(center[0]), float(center[1]), a, ratio, theta


def _make_parameter_set(points: np.ndarray, parameters: Any) -> ParameterSet:
    guess = _initial_guess(points)
    defaults = default_ellipse_parameters(center=guess[:2], a=guess[2], axis_ratio=guess[3], theta=guess[4])
    if parameters is None:
        return defaults
    if not isinstance(parameters, (ParameterSet, Mapping)):
        raise TypeError("parameters must be a ParameterSet or mapping")
    provided = _canonical_parameter_mapping(parameters)
    raw = dict(defaults._parameters)
    for name, spec in provided.items():
        if name not in raw:
            raise ValueError(f"unknown ellipse parameter: {name}")
        raw[name] = spec
    # Mapping specifications are part of the public editor contract (for
    # example ``{"axis_ratio": {"value": .02, "min": .005}}``).  Coerce
    # them before applying the physical parameterization below; constructing
    # ``ParameterSpec(mapping)`` directly would treat the whole mapping as a
    # numeric value.
    for name, spec in tuple(raw.items()):
        if not isinstance(spec, ParameterSpec):
            raw[name] = ParameterSet._coerce_spec(name, spec)
    # Canonicalize bounds required by the physical parameterization.  A caller
    # can tighten these bounds, but cannot turn them into nonphysical ones.
    for name in ("a", "axis_ratio"):
        spec = raw[name]
        if not isinstance(spec, ParameterSpec):
            spec = ParameterSpec(spec, name=name)
        if name == "a":
            lower = np.finfo(float).eps if spec.min is None else spec.min
            if lower <= 0 or (spec.max is not None and spec.max <= 0):
                raise ValueError("a must be greater than zero")
            raw[name] = spec.copy(min=lower, name=name)
        else:
            lower = np.finfo(float).eps if spec.min is None else spec.min
            upper = 1.0 if spec.max is None else spec.max
            if lower <= 0 or upper > 1 or lower > upper:
                raise ValueError("axis_ratio must be in (0, 1]")
            raw[name] = spec.copy(min=lower, max=upper, name=name)
    b_spec = raw["b"]
    if not isinstance(b_spec, ParameterSpec):
        b_spec = ParameterSpec(float(b_spec), name="b")
    if b_spec.expr is None:
        if b_spec.vary:
            raise ValueError("b is derived and cannot be a free parameter")
        b_spec = b_spec.copy(expr="a*axis_ratio", vary=False, name="b")
    raw["b"] = b_spec
    return ParameterSet(raw)


def _residual_function(kind: str) -> Callable[..., np.ndarray]:
    kind = str(kind).lower()
    if kind in {"sampson", "sampson_distance"}:
        return ellipse_sampson_residuals
    if kind in {"geometric", "distance", "closest"}:
        return ellipse_geometric_residuals
    raise ValueError("residual must be 'sampson' or 'geometric'")


def _coverage(points: np.ndarray, residuals: np.ndarray, parameters: ParameterSet,
              *, components: int = 1, labels: np.ndarray | None = None,
              reference_axis: float = 0.0) -> CoverageMetrics:
    values = parameters.resolve()
    geometry = EllipseGeometry.from_values(values)
    dx, dy = points[:, 0] - geometry.cx, points[:, 1] - geometry.cy
    lab_theta = float(reference_axis) + geometry.theta
    c, s = math.cos(lab_theta), math.sin(lab_theta)
    u = c * dx + s * dy
    v = -s * dx + c * dy
    angles: list[np.ndarray] = []
    if components == 1:
        angles.append(np.arctan2(v / geometry.b, u / geometry.a))
    else:
        plus_geometry = EllipseGeometry(
            geometry.cx, geometry.cy, geometry.a, geometry.axis_ratio,
            float(reference_axis) + geometry.theta,
        )
        minus_geometry = EllipseGeometry(
            geometry.cx, geometry.cy, geometry.a, geometry.axis_ratio,
            float(reference_axis) - geometry.theta,
        )
        r_plus = ellipse_sampson_residuals(points, plus_geometry)
        r_minus = ellipse_sampson_residuals(points, minus_geometry)
        selected = np.abs(r_plus) <= np.abs(r_minus) if labels is None else labels == 0
        for sign, selected_mask in ((1.0, selected), (-1.0, ~selected)):
            if np.any(selected_mask):
                g = EllipseGeometry(
                    geometry.cx,
                    geometry.cy,
                    geometry.a,
                    geometry.axis_ratio,
                    float(reference_axis) + sign * geometry.theta,
                )
                ddx, ddy = points[selected_mask, 0] - g.cx, points[selected_mask, 1] - g.cy
                cc, ss = math.cos(g.theta), math.sin(g.theta)
                uu, vv = cc * ddx + ss * ddy, -ss * ddx + cc * ddy
                angles.append(np.arctan2(vv / g.b, uu / g.a))
    if angles:
        angle_array = np.concatenate(angles)
        sorted_angles = np.sort(np.mod(angle_array, 2 * math.pi))
        gaps = np.diff(np.concatenate((sorted_angles, sorted_angles[:1] + 2 * math.pi)))
        span = 2 * math.pi - float(np.max(gaps)) if gaps.size else 0.0
    else:
        span = 0.0
    if points.shape[0] >= 3:
        try:
            area = float(ConvexHull(points).volume)
        except Exception:
            area = 0.0
    else:
        area = 0.0
    return CoverageMetrics(
        n_points=int(points.shape[0]),
        angular_span=float(span),
        angular_coverage=float(np.clip(span / (2 * math.pi), 0.0, 1.0)),
        radial_rms=float(np.sqrt(np.mean(np.square(residuals)))) if residuals.size else float("nan"),
        convex_hull_area=area,
        components=components,
    )


def _bound_diagnostics(parameters: ParameterSet, values: Mapping[str, float]) -> tuple[dict[str, bool], dict[str, str | None]]:
    flags: dict[str, bool] = {}
    status: dict[str, str | None] = {}
    for name, spec in parameters.spec_items():
        value = values[name]
        scale = 1.0 + abs(value)
        at_lower = spec.min is not None and np.isfinite(spec.min) and abs(value - spec.min) <= 1e-8 * scale
        at_upper = spec.max is not None and np.isfinite(spec.max) and abs(value - spec.max) <= 1e-8 * scale
        state = "lower" if at_lower else "upper" if at_upper else None
        status[name] = state
        flags[name] = state is not None
    return flags, status


def _branch_assignment(points: np.ndarray, parameters: ParameterSet, labels: np.ndarray | None,
                       *, residual: ResidualKind, reference_axis: float,
                       components: int) -> np.ndarray | None:
    """Return stable component labels for the selected symmetric solution."""

    if components != 2:
        return None
    if labels is not None:
        # Explicit supervision is authoritative; do not recompute or reorder it.
        return np.asarray(labels, dtype=int).copy()
    values = parameters.resolve()
    geometry = EllipseGeometry.from_values(values)
    plus = EllipseGeometry(
        geometry.cx, geometry.cy, geometry.a, geometry.axis_ratio,
        float(reference_axis) + geometry.theta,
    )
    minus = EllipseGeometry(
        geometry.cx, geometry.cy, geometry.a, geometry.axis_ratio,
        float(reference_axis) - geometry.theta,
    )
    residual_function = _residual_function(residual)
    r_plus = np.asarray(residual_function(points, plus), dtype=float)
    r_minus = np.asarray(residual_function(points, minus), dtype=float)
    # Tie-breaking to branch 0 makes the result deterministic at the symmetry
    # axis and matches symmetric_ellipse_residuals.
    return np.where(np.abs(r_plus) <= np.abs(r_minus), 0, 1).astype(int)


def _make_result(parameters: ParameterSet, scipy_result: OptimizeResult | None, residuals: np.ndarray,
                 *, model: str, points: np.ndarray, components: int, labels: np.ndarray | None,
                 reference_axis: float = 0.0,
                 candidate_solutions: tuple[dict[str, Any], ...] = (),
                 selected_start_index: int = 0,
                 multistart_count: int = 1,
                 residual: ResidualKind = "sampson") -> EllipseFitResult:
    values = parameters.resolve()
    free_names = parameters.free_names
    jacobian = None if scipy_result is None else np.asarray(getattr(scipy_result, "jac", None), dtype=float)
    if jacobian is not None and (jacobian.ndim != 2 or jacobian.shape[1] != len(free_names)):
        jacobian = None
    covariance: np.ndarray | None = None
    condition = float("inf")
    stderr = {name: float("nan") for name in parameters.names}
    if jacobian is not None and jacobian.shape[1]:
        singular_values = np.linalg.svd(jacobian, compute_uv=False)
        if singular_values.size and singular_values[-1] > 0:
            condition = float(singular_values[0] / singular_values[-1])
        dof = max(residuals.size - jacobian.shape[1], 1)
        scale = float(2 * (scipy_result.cost if scipy_result is not None else np.sum(residuals ** 2) / 2) / dof)
        covariance = np.linalg.pinv(jacobian.T @ jacobian) * scale
        diagonal = np.diag(covariance)
        for name, variance in zip(free_names, diagonal):
            stderr[name] = float(math.sqrt(variance)) if variance >= 0 and np.isfinite(variance) else float("nan")
    flags, status = _bound_diagnostics(parameters, values)
    coverage = _coverage(
        points,
        residuals,
        parameters,
        components=components,
        labels=labels,
        reference_axis=reference_axis,
    )
    if scipy_result is None:
        success, message, cost, optimality, nfev = True, "no free parameters", float(np.sum(residuals ** 2) / 2), 0.0, 0
    else:
        success = bool(scipy_result.success)
        message = str(scipy_result.message)
        cost = float(scipy_result.cost)
        optimality = float(scipy_result.optimality)
        nfev = int(scipy_result.nfev)
    return EllipseFitResult(
        parameters=parameters,
        values=values,
        residuals=np.asarray(residuals, dtype=float),
        success=success,
        message=message,
        cost=cost,
        optimality=optimality,
        nfev=nfev,
        covariance=covariance,
        stderr=stderr,
        condition=condition,
        bound_flags=flags,
        bound_status=status,
        coverage=coverage,
        free_names=free_names,
        jacobian=jacobian,
        scipy_result=scipy_result,
        model=model,
        reference_axis=float(reference_axis),
        candidate_solutions=candidate_solutions,
        selected_start_index=int(selected_start_index),
        multistart_count=int(multistart_count),
        branch_assignment=_branch_assignment(
            points,
            parameters,
            labels,
            residual=residual,
            reference_axis=reference_axis,
            components=components,
        ),
    )


def _run_fit(points: np.ndarray, parameters: ParameterSet, objective: Callable[[ParameterSet], np.ndarray],
             *, loss: str, f_scale: float, max_nfev: int | None, model: str,
             components: int, labels: np.ndarray | None,
             reference_axis: float = 0.0,
             residual: ResidualKind = "sampson",
             cancel_event: Any = None) -> EllipseFitResult:
    raise_if_cancelled(cancel_event, f"ellipse:{model}:start")
    if f_scale <= 0 or not np.isfinite(f_scale):
        raise ValueError("f_scale must be finite and positive")
    loss = str(loss).lower()
    allowed_losses = {"linear", "soft_l1", "huber", "cauchy", "arctan"}
    if loss not in allowed_losses:
        raise ValueError(f"loss must be one of {sorted(allowed_losses)}")
    free_names = list(parameters.free_names)
    lower, upper = parameters.free_bounds()
    # Equal bounds are naturally fixed even when a GUI left vary checked.
    equal = np.isfinite(lower) & np.isfinite(upper) & np.isclose(lower, upper, rtol=0, atol=0)
    fixed_by_bound = {name for name, same in zip(free_names, equal) if same}
    if fixed_by_bound:
        for name in fixed_by_bound:
            parameters[name].vary = False
        free_names = list(parameters.free_names)
        lower, upper = parameters.free_bounds()
    if free_names:
        x0 = parameters.free_vector()
        for index, (value, lo, hi) in enumerate(zip(x0, lower, upper)):
            if np.isfinite(lo) and value <= lo:
                x0[index] = np.nextafter(lo, hi if np.isfinite(hi) else np.inf)
            if np.isfinite(hi) and value >= hi:
                x0[index] = np.nextafter(hi, lo if np.isfinite(lo) else -np.inf)
            if np.isfinite(lo) and np.isfinite(hi) and not lo < hi:
                raise ValueError("free parameter bounds must have min < max")
        def residual_at(vector: np.ndarray) -> np.ndarray:
            raise_if_cancelled(cancel_event, f"ellipse:{model}:residual")
            candidate = parameters.copy()
            candidate.set_free_vector(vector)
            try:
                return np.asarray(objective(candidate), dtype=float)
            except (ValueError, FloatingPointError, OverflowError):
                # Physical constraints expressed through a tied expression can
                # be tighter than scipy's direct bounds.  A finite penalty
                # keeps least_squares on the valid side of such a boundary.
                return np.full(points.shape[0], 1e12, dtype=float)
        # Keep the established Sampson trajectory for ordinary ellipses (it
        # is part of the public deterministic multistart contract), while
        # enabling Jacobian scaling whenever a geometric fit or an explicitly
        # flat ratio is being refined.
        flat_parameter = parameters._parameters.get("axis_ratio")
        flat_mode = bool(
            flat_parameter is not None
            and (
                float(flat_parameter.value) < 0.05
                or (flat_parameter.min is not None and float(flat_parameter.min) < 0.05)
            )
        )
        use_jac_scale = str(residual).lower() in {"geometric", "distance", "closest"} or flat_mode
        result = least_squares(
            residual_at,
            x0,
            bounds=(lower, upper),
            loss=loss,
            f_scale=float(f_scale),
            # q coordinates, angle (radians), and an axis ratio can differ by
            # several orders of magnitude for the thin ellipses in real
            # butterfly patterns.  Jacobian scaling keeps each editable
            # parameter visible to the trust-region step without changing the
            # reported units or the caller's bounds.
            x_scale="jac" if use_jac_scale else 1.0,
            max_nfev=max_nfev,
        )
        raise_if_cancelled(cancel_event, f"ellipse:{model}:complete")
        fitted = parameters.copy().set_free_vector(result.x)
        residuals = np.asarray(objective(fitted), dtype=float)
        return _make_result(
            fitted,
            result,
            residuals,
            model=model,
            points=points,
            components=components,
            labels=labels,
            reference_axis=reference_axis,
            residual=residual,
        )
    raise_if_cancelled(cancel_event, f"ellipse:{model}:residual")
    residuals = np.asarray(objective(parameters), dtype=float)
    return _make_result(
        parameters,
        None,
        residuals,
        model=model,
        points=points,
        components=components,
        labels=labels,
        reference_axis=reference_axis,
        residual=residual,
    )


def fit_ellipse(points: Any, parameters: ParameterSet | Mapping[str, Any] | None = None, *,
                params: ParameterSet | Mapping[str, Any] | None = None,
                residual: ResidualKind = "sampson", loss: str = "soft_l1", f_scale: float = 1.0,
                weights: Any = None, max_nfev: int | None = None, config: Any = None,
                residual_kind: str | None = None,
                cancel_event: Any = None) -> EllipseFitResult:
    """Fit one ellipse to q-space ridge points with robust least squares."""

    if parameters is not None and params is not None:
        raise ValueError("supply only one of parameters or params")
    if config is not None:
        if parameters is None and params is None:
            parameters = getattr(config, "parameters", None)
        residual = getattr(config, "residual", residual)
        loss = getattr(config, "loss", loss)
        f_scale = getattr(config, "f_scale", f_scale)
        max_nfev = getattr(config, "max_nfev", max_nfev)
        cancel_event = getattr(config, "cancel_event", cancel_event)
    if residual_kind is not None:
        residual = residual_kind
    points = _coerce_points(points)
    weights = _coerce_weights(weights, points.shape[0])
    parameter_set = _make_parameter_set(points, parameters if parameters is not None else params)
    residual_function = _residual_function(residual)
    def objective(candidate: Mapping[str, float]) -> np.ndarray:
        return residual_function(points, candidate, weights=weights)
    return _run_fit(points, parameter_set, objective, loss=loss, f_scale=f_scale, max_nfev=max_nfev,
                    model="ellipse", components=1, labels=None,
                    cancel_event=cancel_event)


def _coerce_labels(labels: Any, n: int) -> np.ndarray | None:
    if labels is None:
        return None
    result = np.asarray(labels)
    if result.ndim != 1 or result.size != n:
        raise ValueError(f"labels must have shape ({n},)")
    if not np.all(np.isin(result, [0, 1])):
        raise ValueError("labels must contain only 0 or 1")
    return result.astype(int)


def symmetric_ellipse_residuals(points: Any, parameters: Any, *, residual: ResidualKind = "sampson",
                                 labels: Any = None, weights: Any = None,
                                 reference_axis_deg: float = 0.0,
                                 reference_axis: float | None = None) -> np.ndarray:
    """Residual for shared-centre, shared-axis ``+/-theta`` ellipses."""

    points = _coerce_points(points)
    labels = _coerce_labels(labels, points.shape[0])
    weights_array = _coerce_weights(weights, points.shape[0])
    geometry = _geometry(parameters)
    reference = math.radians(float(reference_axis_deg)) if reference_axis is None else float(reference_axis)
    positive = EllipseGeometry(
        geometry.cx, geometry.cy, geometry.a, geometry.axis_ratio, reference + geometry.theta
    )
    negative = EllipseGeometry(
        geometry.cx, geometry.cy, geometry.a, geometry.axis_ratio, reference - geometry.theta
    )
    residual_function = _residual_function(residual)
    r_positive = residual_function(points, positive)
    r_negative = residual_function(points, negative)
    if labels is None:
        result = np.where(np.abs(r_positive) <= np.abs(r_negative), r_positive, r_negative)
    else:
        result = np.where(labels == 0, r_positive, r_negative)
    return result * np.sqrt(weights_array)


double_ellipse_residuals = symmetric_ellipse_residuals
symmetric_double_ellipse_residuals = symmetric_ellipse_residuals
symmetric_double_ellipse_residual = symmetric_ellipse_residuals


def _validate_multistart_count(multistart: Any) -> int:
    if isinstance(multistart, (bool, np.bool_)) or not isinstance(multistart, (int, np.integer)):
        raise TypeError("multistart must be an integer >= 1")
    count = int(multistart)
    if count < 1:
        raise ValueError("multistart must be an integer >= 1")
    return count


def _multistart_parameter_sets(parameters: ParameterSet, count: int) -> tuple[ParameterSet, ...]:
    """Build deterministic starts while preserving free/fixed/tied semantics."""

    if count == 1 or not parameters.free_names:
        return tuple(parameters.copy() for _ in range(count))
    base = parameters.free_vector()
    lower, upper = parameters.free_bounds()
    # Fractions are deliberately fixed (no RNG/seed dependency).  They cover
    # both sides and the centre of each finite interval, while the first start
    # remains exactly the caller's editable initial value.
    fractions = (0.20, 0.80, 0.35, 0.65, 0.50, 0.10, 0.90)
    offsets = (-1.5, 1.5, -0.75, 0.75, 0.0, -2.0, 2.0)
    starts: list[ParameterSet] = [parameters.copy()]
    for start_index in range(1, count):
        vector = base.copy()
        fraction = fractions[(start_index - 1) % len(fractions)]
        offset = offsets[(start_index - 1) % len(offsets)]
        for index, (name, value, lo, hi) in enumerate(zip(parameters.free_names, base, lower, upper)):
            if np.isfinite(lo) and np.isfinite(hi):
                if name == "axis_ratio" and lo > 0.0 and hi > 0.0 and lo < 0.05:
                    # Axis ratio is a scale parameter over decades in the
                    # flat regime.  Log interpolation gives deterministic
                    # starts near .02/.05 instead of spending every start in
                    # the visually near-circular half of the interval.
                    log_lo = math.log(max(float(lo), 0.005 * float(hi)))
                    log_hi = math.log(float(hi))
                    vector[index] = math.exp(log_lo + fraction * (log_hi - log_lo))
                else:
                    vector[index] = lo + fraction * (hi - lo)
            else:
                scale = max(abs(float(value)), abs(float(lo)) if np.isfinite(lo) else 0.0,
                            abs(float(hi)) if np.isfinite(hi) else 0.0, 1.0)
                vector[index] = float(value) + offset * scale
                if np.isfinite(lo):
                    vector[index] = max(vector[index], lo)
                if np.isfinite(hi):
                    vector[index] = min(vector[index], hi)
        candidate = parameters.copy()
        candidate.set_free_vector(vector)
        # Resolving here makes a tied-expression failure belong to the
        # offending deterministic start, before scipy is invoked.
        candidate.resolve()
        starts.append(candidate)
    return tuple(starts)


def _candidate_solution_record(start_index: int, start: ParameterSet, result: EllipseFitResult) -> dict[str, Any]:
    return {
        "start_index": int(start_index),
        "start_values": dict(start.resolve()),
        "values": dict(result.values),
        "success": bool(result.success),
        "finite_cost": bool(np.isfinite(result.cost)),
        "cost": float(result.cost),
        "message": str(result.message),
    }


def fit_symmetric_ellipses(points: Any, parameters: ParameterSet | Mapping[str, Any] | None = None, *,
                           params: ParameterSet | Mapping[str, Any] | None = None,
                           residual: ResidualKind = "sampson", loss: str = "soft_l1", f_scale: float = 1.0,
                           labels: Any = None, weights: Any = None, max_nfev: int | None = None,
                           config: Any = None, residual_kind: str | None = None,
                           reference_axis_deg: float = 0.0,
                           reference_axis: float | None = None,
                           multistart: int = 7,
                           cancel_event: Any = None) -> EllipseFitResult:
    """Fit a mirror-symmetric pair of ellipses sharing centre and axes."""

    if parameters is not None and params is not None:
        raise ValueError("supply only one of parameters or params")
    if config is not None:
        if parameters is None and params is None:
            parameters = getattr(config, "parameters", None)
        residual = getattr(config, "residual", residual)
        loss = getattr(config, "loss", loss)
        f_scale = getattr(config, "f_scale", f_scale)
        max_nfev = getattr(config, "max_nfev", max_nfev)
        reference_axis_deg = getattr(config, "reference_axis_deg", reference_axis_deg)
        multistart = getattr(config, "multistart", multistart)
        cancel_event = getattr(config, "cancel_event", cancel_event)
    if residual_kind is not None:
        residual = residual_kind
    points = _coerce_points(points)
    labels = _coerce_labels(labels, points.shape[0])
    weights = _coerce_weights(weights, points.shape[0])
    source_parameters = parameters if parameters is not None else params
    parameter_set = _make_parameter_set(points, source_parameters)
    reference = math.radians(float(reference_axis_deg)) if reference_axis is None else float(reference_axis)
    if source_parameters is None:
        relative = (parameter_set["theta"].value - reference + math.pi / 2) % math.pi - math.pi / 2
        parameter_set["theta"].set_value(relative)
    # In a mirror pair the geometries at +theta/-theta are unchanged when
    # theta changes sign.  Keeping both signs would therefore introduce a
    # duplicate optimum and an avoidable covariance singularity.  Store the
    # apparent pair tilt canonically as a magnitude in [0, pi/2]; the two
    # signed laboratory orientations remain reference +/- theta.
    theta_spec = parameter_set["theta"]
    theta_lower = max(0.0, theta_spec.min if theta_spec.min is not None else 0.0)
    theta_upper = min(math.pi / 2, theta_spec.max if theta_spec.max is not None else math.pi / 2)
    if theta_lower > theta_upper:
        raise ValueError("symmetric ellipse theta bounds must overlap [0, pi/2]")
    if theta_spec.expr is None:
        canonical_theta = abs((float(theta_spec.value) + math.pi / 2) % math.pi - math.pi / 2)
        canonical_theta = float(np.clip(canonical_theta, theta_lower, theta_upper))
        parameter_set["theta"] = theta_spec.copy(
            value=canonical_theta,
            min=theta_lower,
            max=theta_upper,
            name="theta",
        )
    else:
        parameter_set["theta"] = theta_spec.copy(
            min=theta_lower,
            max=theta_upper,
            name="theta",
        )
    def objective(candidate: Mapping[str, float]) -> np.ndarray:
        return symmetric_ellipse_residuals(
            points,
            candidate,
            residual=residual,
            labels=labels,
            weights=weights,
            reference_axis=reference,
        )
    multistart_count = _validate_multistart_count(multistart)
    starts = _multistart_parameter_sets(parameter_set, multistart_count)
    candidates: list[tuple[int, EllipseFitResult, dict[str, Any]]] = []
    for start_index, start in enumerate(starts):
        raise_if_cancelled(cancel_event, "ellipse:symmetric:multistart")
        result = _run_fit(
            points,
            start,
            objective,
            loss=loss,
            f_scale=f_scale,
            max_nfev=max_nfev,
            model="symmetric_double_ellipse",
            components=2,
            labels=labels,
            reference_axis=reference,
            residual=residual,
            cancel_event=cancel_event,
        )
        candidates.append((start_index, result, _candidate_solution_record(start_index, start, result)))
    # Prefer a solver-reported success, then finite cost, then the smallest
    # cost.  The final index tie-break is explicit for reproducibility.
    selected_index, selected, _ = min(
        candidates,
        key=lambda item: (
            not bool(item[1].success),
            not bool(np.isfinite(item[1].cost)),
            float(item[1].cost) if np.isfinite(item[1].cost) else float("inf"),
            item[0],
        ),
    )
    records = tuple(item[2] for item in candidates)
    return EllipseFitResult(
        **{
            field_name: getattr(selected, field_name)
            for field_name in EllipseFitResult.__dataclass_fields__
            if field_name not in {"candidate_solutions", "selected_start_index", "multistart_count"}
        },
        candidate_solutions=records,
        selected_start_index=int(selected_index),
        multistart_count=multistart_count,
    )


fit_double_ellipse = fit_symmetric_ellipses
fit_two_ellipses = fit_symmetric_ellipses
fit_symmetric_double_ellipse = fit_symmetric_ellipses


def ellipse_points(parameters: Any, phi: Any) -> np.ndarray:
    """Convenience function returning parametric points on one ellipse."""

    return _geometry(parameters).point(phi)


def symmetric_ellipse_points(parameters: Any, phi: Any) -> tuple[np.ndarray, np.ndarray]:
    geometry = _geometry(parameters)
    return geometry.point(phi), EllipseGeometry(geometry.cx, geometry.cy, geometry.a,
                                                geometry.axis_ratio, -geometry.theta).point(phi)


__all__ = [
    "CoverageMetrics",
    "EllipseFitResult",
    "EllipseGeometry",
    "ellipse_geometric_residuals",
    "ellipse_implicit",
    "ellipse_points",
    "ellipse_residuals",
    "ellipse_sampson_residuals",
    "ellipse_sampson_residual",
    "ellipse_geometric_residual",
    "fit_double_ellipse",
    "fit_ellipse",
    "fit_symmetric_double_ellipse",
    "fit_symmetric_ellipses",
    "fit_two_ellipses",
    "geometric_residuals",
    "sampson_residuals",
    "symmetric_double_ellipse_residuals",
    "symmetric_double_ellipse_residual",
    "symmetric_ellipse_points",
    "symmetric_ellipse_residuals",
]
