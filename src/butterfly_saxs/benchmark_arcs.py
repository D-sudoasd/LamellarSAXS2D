"""Independent image-level butterfly arc benchmark.

The generator in this module is deliberately separate from the application's
empirical intensity model.  It constructs short parametric polylines in q
space, rasterises their distance field with detector-pixel supersampling, and
applies a small detector blur.  The resulting image is useful for testing
topology and flat-ellipse failure modes because the observable arcs and their
ground truth are retained explicitly.

``True`` in ``mask`` means an invalid detector pixel.  Coordinates are in
``nm^-1``.  The generator describes observable image geometry only; it does
not claim a physical lamellar forward model or a calibrated uncertainty.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:  # scipy is a project dependency, but a small fallback keeps this module portable.
    from scipy.ndimage import gaussian_filter
except Exception:  # pragma: no cover - exercised only in a minimal NumPy runtime
    gaussian_filter = None


GENERATOR_VERSION = "arcs-independent-image-v1"
MODEL_SCOPE = "independent_known_truth_observable_arcs"
Q_UNIT = "nm^-1"
DEFAULT_SHAPE = (96, 96)
DEFAULT_CASE_IDS = (
    "ellipse_ratio_005",
    "ellipse_ratio_020",
    "ellipse_ratio_100",
    "ellipse_ratio_400",
    "partial_arcs",
    "missing_branch",
    "overlapping_unresolved",
    "asymmetric_branches",
    "miscentered_branches",
    "warped_qmap",
    "smooth_nonellipse",
    "null",
)


@dataclass(frozen=True)
class ArcCaseSpec(Mapping[str, Any]):
    """Public settings for one independent arc case."""

    case_id: str
    category: str
    axis_ratio: float | None
    partial: bool = False
    missing_branch: bool = False
    overlap: bool = False
    asymmetric: bool = False
    miscentered: bool = False
    warped_qmap: bool = False
    non_elliptic: bool = False
    null: bool = False
    seed: int = 0
    description: str = ""

    def __post_init__(self) -> None:
        if not str(self.case_id).strip():
            raise ValueError("case_id cannot be empty")
        if self.axis_ratio is not None:
            ratio = float(self.axis_ratio)
            if not np.isfinite(ratio) or ratio <= 0.0 or ratio > 1.0:
                raise ValueError("axis_ratio must be finite and in (0, 1]")
            object.__setattr__(self, "axis_ratio", ratio)
        if isinstance(self.seed, (bool, np.bool_)):
            raise TypeError("case seed must be an integer")
        if not isinstance(self.seed, (int, np.integer)):
            raise TypeError("case seed must be an integer")
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "case_id", str(self.case_id).strip())
        object.__setattr__(self, "category", str(self.category).strip())

    @property
    def name(self) -> str:
        return self.case_id

    def __getitem__(self, key: str) -> Any:
        aliases = {"name": "case_id", "id": "case_id", "b_over_a": "axis_ratio"}
        field_name = aliases.get(str(key), str(key))
        if field_name not in self.__dataclass_fields__:
            raise KeyError(key)
        return getattr(self, field_name)

    def __iter__(self):
        return iter(self.__dataclass_fields__)

    def __len__(self) -> int:
        return len(self.__dataclass_fields__)

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["name"] = self.name
        value["b_over_a"] = self.axis_ratio
        return value


@dataclass
class ArcQMap(Mapping[str, Any]):
    """Small q-map adapter returned by :func:`generate_arc_case`.

    It intentionally mirrors only the fields needed by image-level callers;
    keeping it local makes the benchmark independent of the fitting stack.
    """

    qx: np.ndarray
    qy: np.ndarray
    q: np.ndarray
    mask: np.ndarray
    q_unit: str = Q_UNIT
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        self.qx = np.asarray(self.qx, dtype=float)
        self.qy = np.asarray(self.qy, dtype=float)
        self.q = np.asarray(self.q, dtype=float)
        self.mask = np.asarray(self.mask, dtype=bool)
        if self.qx.ndim != 2 or self.qx.shape != self.qy.shape or self.qx.shape != self.q.shape:
            raise ValueError("qx, qy and q must be equal-shaped two-dimensional arrays")
        if self.mask.shape != self.qx.shape:
            raise ValueError("q-map mask must match q-map shape")
        self.q_unit = str(self.q_unit)
        self.metadata = dict(self.metadata or {})
        self.metadata.setdefault("q_unit", self.q_unit)

    @property
    def shape(self) -> tuple[int, int]:
        return self.qx.shape

    @property
    def angle(self) -> np.ndarray:
        return np.arctan2(self.qy, self.qx)

    @property
    def azimuth(self) -> np.ndarray:
        return self.angle

    @property
    def valid_mask(self) -> np.ndarray:
        return ~self.mask

    def __getitem__(self, key: str) -> Any:
        aliases = {"q_x": "qx", "q_y": "qy", "q_map": "q", "unit": "q_unit"}
        name = aliases.get(str(key), str(key))
        if name in {"qx", "qy", "q", "mask", "q_unit", "metadata"}:
            return getattr(self, name)
        if name == "valid_mask":
            return self.valid_mask
        if name in {"angle", "azimuth"}:
            return self.angle
        raise KeyError(key)

    def __iter__(self):
        return iter(("qx", "qy", "q", "mask", "q_unit", "metadata"))

    def __len__(self) -> int:
        return 6


DEFAULT_CASES: tuple[ArcCaseSpec, ...] = (
    ArcCaseSpec("ellipse_ratio_005", "ellipse", 0.005, seed=501, description="Very flat ellipse, b/a=0.005."),
    ArcCaseSpec("ellipse_ratio_020", "ellipse", 0.020, seed=502, description="Flat ellipse, b/a=0.02."),
    ArcCaseSpec("ellipse_ratio_100", "ellipse", 0.100, seed=503, description="Moderately flat ellipse, b/a=0.1."),
    ArcCaseSpec("ellipse_ratio_400", "ellipse", 0.400, seed=504, description="Ordinary ellipse, b/a=0.4."),
    ArcCaseSpec("partial_arcs", "partial_arcs", 0.10, partial=True, seed=505, description="Both branches are observed only over short arcs."),
    ArcCaseSpec("missing_branch", "missing_branch", 0.10, missing_branch=True, seed=506, description="One branch is absent from the observable image."),
    ArcCaseSpec("overlapping_unresolved", "overlap", 0.08, overlap=True, seed=507, description="Two close ellipses are intentionally unresolved at detector scale."),
    ArcCaseSpec("asymmetric_branches", "asymmetry", 0.10, asymmetric=True, seed=508, description="Branch amplitudes and widths differ."),
    ArcCaseSpec("miscentered_branches", "miscenter", 0.10, miscentered=True, seed=509, description="A deliberately miscentered negative control."),
    ArcCaseSpec("warped_qmap", "warped_qmap", 0.10, warped_qmap=True, seed=510, description="The detector-to-q map has a smooth spatial warp."),
    ArcCaseSpec("smooth_nonellipse", "non_elliptic", None, non_elliptic=True, seed=511, description="Smooth curved arcs with no true ellipse axes."),
    ArcCaseSpec("null", "null", None, null=True, seed=512, description="Background-only image with no observable arcs."),
)

# Descriptive aliases used by validation notebooks and external campaign
# scripts; all refer to the same immutable case definitions.
DEFAULT_ARC_CASES = DEFAULT_CASES
ARC_CASES = DEFAULT_CASES
CASE_MATRIX = DEFAULT_CASES


def default_cases() -> tuple[ArcCaseSpec, ...]:
    """Return the immutable default matrix."""

    return DEFAULT_CASES


def _normalise_name(value: Any) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def _resolve_case(case_id: str | ArcCaseSpec | Mapping[str, Any]) -> ArcCaseSpec:
    if isinstance(case_id, ArcCaseSpec):
        return case_id
    if isinstance(case_id, Mapping):
        raw = case_id.get("case_id", case_id.get("name", case_id.get("category")))
        if raw is None:
            raise ValueError("case mapping must contain case_id, name, or category")
        base = _resolve_case(str(raw))
        values = asdict(base)
        aliases = {"name": "case_id", "id": "case_id", "b_over_a": "axis_ratio"}
        for key, value in case_id.items():
            name = aliases.get(str(key), str(key))
            if name not in values:
                raise ValueError(f"unknown arc case key: {key}")
            values[name] = value
        return ArcCaseSpec(**values)
    normalized = _normalise_name(case_id)
    normalized_compact = normalized.replace(".", "p")
    aliases = {
        "ratio_005": "ellipse_ratio_005",
        "ratio_020": "ellipse_ratio_020",
        "ratio_100": "ellipse_ratio_100",
        "ratio_400": "ellipse_ratio_400",
        "ratio_0p005": "ellipse_ratio_005",
        "ratio_0p02": "ellipse_ratio_020",
        "ratio_0p1": "ellipse_ratio_100",
        "ratio_0p4": "ellipse_ratio_400",
        "b_a_005": "ellipse_ratio_005",
        "b_a_020": "ellipse_ratio_020",
        "b_a_100": "ellipse_ratio_100",
        "b_a_400": "ellipse_ratio_400",
        "b_over_a_005": "ellipse_ratio_005",
        "b_over_a_020": "ellipse_ratio_020",
        "b_over_a_100": "ellipse_ratio_100",
        "b_over_a_400": "ellipse_ratio_400",
        "flat_005": "ellipse_ratio_005",
        "flat_020": "ellipse_ratio_020",
        "flat_100": "ellipse_ratio_100",
        "flat_400": "ellipse_ratio_400",
        "nonellipse": "smooth_nonellipse",
        "smooth_non_ellipse": "smooth_nonellipse",
        "missingbranch": "missing_branch",
        "partial": "partial_arcs",
        "overlapping": "overlapping_unresolved",
        "warped": "warped_qmap",
    }
    normalized = aliases.get(normalized, aliases.get(normalized_compact, normalized))
    for spec in DEFAULT_CASES:
        if _normalise_name(spec.case_id) == normalized or _normalise_name(spec.category) == normalized:
            return spec
    choices = ", ".join(DEFAULT_CASE_IDS)
    raise ValueError(f"unknown arc case {case_id!r}; choose one of: {choices}")


def _validate_shape(shape: Sequence[int]) -> tuple[int, int]:
    try:
        values = tuple(shape)
    except TypeError as exc:
        raise ValueError("shape must contain two integer dimensions") from exc
    if len(values) != 2 or any(isinstance(value, (bool, np.bool_)) for value in values):
        raise ValueError("shape must contain two integer dimensions")
    if any(not isinstance(value, (int, np.integer)) or int(value) < 8 for value in values):
        raise ValueError("shape dimensions must be integers >= 8")
    return int(values[0]), int(values[1])


def _case_identity(case_id: str, seed: int, shape: tuple[int, int]) -> dict[str, Any]:
    """Return the dimensions that identify one generated benchmark image.

    The generator hash identifies the source code.  It must remain separate
    from the case/seed/shape identity so two noise draws cannot be mistaken
    for the same image, and changing a case definition cannot masquerade as a
    source-code change.
    """

    return {"case_id": str(case_id), "seed": int(seed), "shape": [int(value) for value in shape]}


def _case_identity_hash(identity: Mapping[str, Any]) -> str:
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _ellipse_polyline(
    center: tuple[float, float],
    a: float,
    b: float,
    theta_deg: float,
    phi_start: float,
    phi_stop: float,
    *,
    n_points: int = 96,
) -> np.ndarray:
    phi = np.linspace(float(phi_start), float(phi_stop), int(max(12, n_points)))
    angle = np.deg2rad(float(theta_deg))
    c, s = np.cos(angle), np.sin(angle)
    local_x = float(a) * np.cos(phi)
    local_y = float(b) * np.sin(phi)
    x = float(center[0]) + c * local_x - s * local_y
    y = float(center[1]) + s * local_x + c * local_y
    return np.column_stack((x, y))


def _nonellipse_polyline(
    center: tuple[float, float],
    phi_start: float,
    phi_stop: float,
    *,
    branch_sign: float,
    n_points: int = 96,
) -> np.ndarray:
    phi = np.linspace(float(phi_start), float(phi_stop), int(max(12, n_points)))
    # A smooth, visibly non-elliptic curve.  The third harmonic creates a
    # broad waist while the low-frequency term bends the observable ridge.
    radius = 0.68 + 0.10 * np.sin(3.0 * phi + 0.3 * branch_sign)
    x = float(center[0]) + radius * np.cos(phi)
    y = float(center[1]) + 0.18 * np.sin(phi) + branch_sign * 0.08 * np.sin(2.0 * phi)
    return np.column_stack((x, y))


def _segment_distance(x: np.ndarray, y: np.ndarray, polyline: np.ndarray) -> np.ndarray:
    """Return the nearest Euclidean distance from points to a polyline."""

    x_flat = np.asarray(x, dtype=float).ravel()
    y_flat = np.asarray(y, dtype=float).ravel()
    points0 = np.asarray(polyline[:-1], dtype=float)
    delta = np.asarray(polyline[1:], dtype=float) - points0
    denominator = np.sum(delta * delta, axis=1)
    denominator = np.where(denominator > np.finfo(float).eps, denominator, 1.0)
    best = np.full(x_flat.shape, np.inf, dtype=float)
    # A segment-at-a-time reduction avoids a large (pixel, segment, 2) array.
    for start, (dx, dy), den in zip(points0, delta, denominator):
        t = ((x_flat - start[0]) * dx + (y_flat - start[1]) * dy) / den
        t = np.clip(t, 0.0, 1.0)
        distance = np.hypot(x_flat - (start[0] + t * dx), y_flat - (start[1] + t * dy))
        best = np.minimum(best, distance)
    return best.reshape(np.asarray(x).shape)


def _fallback_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0.0:
        return np.asarray(image, dtype=float)
    radius = max(1, int(math.ceil(3.0 * float(sigma))))
    coordinate = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (coordinate / float(sigma)) ** 2)
    kernel /= np.sum(kernel)
    padded = np.pad(np.asarray(image, dtype=float), ((radius, radius), (radius, radius)), mode="reflect")
    result = np.empty_like(padded)
    # Two explicit separable passes are adequate for the fallback path.
    result[:] = padded
    for row in range(padded.shape[0]):
        result[row] = np.convolve(padded[row], kernel, mode="same")
    vertical = np.empty_like(result)
    for col in range(result.shape[1]):
        vertical[:, col] = np.convolve(result[:, col], kernel, mode="same")
    return vertical[radius:-radius, radius:-radius]


def _blur(image: np.ndarray, sigma: float) -> np.ndarray:
    if float(sigma) <= 0.0:
        return np.asarray(image, dtype=float)
    if gaussian_filter is not None:
        return np.asarray(gaussian_filter(np.asarray(image, dtype=float), sigma=float(sigma), mode="reflect"), dtype=float)
    return _fallback_blur(image, float(sigma))


def _q_grid(shape: tuple[int, int], *, warped: bool) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = shape
    extent = 1.15
    y = np.linspace(-extent, extent, rows, dtype=float)
    x = np.linspace(-extent, extent, cols, dtype=float)
    qx, qy = np.meshgrid(x, y)
    if warped:
        # The perturbation is smooth and invertible over the small detector
        # field, so every truth point remains an observable q-space point.
        qx = qx + 0.045 * np.sin(np.pi * qy / extent) * np.cos(np.pi * qx / extent)
        qy = qy + 0.040 * np.sin(np.pi * qx / extent) * np.cos(np.pi * qy / extent)
    return qx, qy


def _rasterize_arc(
    qx: np.ndarray,
    qy: np.ndarray,
    polyline: np.ndarray,
    *,
    amplitude: float,
    width: float,
    subpixel: int = 3,
) -> np.ndarray:
    """Rasterise a line using distance and detector-pixel integration."""

    rows, cols = qx.shape
    # Use the full local detector-to-q Jacobian.  A warped map can have
    # cross-components (dqx/drow and dqy/dcol), so diagonal median steps
    # silently misplace subpixel samples and produce a false calibration.
    qx_row, qx_col = np.gradient(qx)
    qy_row, qy_col = np.gradient(qy)
    count = max(1, int(subpixel))
    offsets = (np.arange(count, dtype=float) + 0.5) / count - 0.5
    result = np.zeros((rows, cols), dtype=float)
    for offset_y in offsets:
        for offset_x in offsets:
            shifted_x = qx + float(offset_y) * qx_row + float(offset_x) * qx_col
            shifted_y = qy + float(offset_y) * qy_row + float(offset_x) * qy_col
            distance = _segment_distance(shifted_x, shifted_y, polyline)
            result += float(amplitude) * np.exp(-0.5 * (distance / max(float(width), 1.0e-8)) ** 2)
    return result / float(count * count)


def _arc_record(
    *,
    arc_id: str,
    branch: str,
    side: str,
    polyline: np.ndarray,
    center: tuple[float, float],
    phi_start: float,
    phi_stop: float,
    amplitude: float,
    width: float,
    ellipse: bool,
    a: float | None = None,
    b: float | None = None,
    theta_deg: float | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "arc_id": str(arc_id),
        "branch": str(branch),
        "branch_id": 0 if str(branch) == "plus" else 1,
        "quadrant_pair": "QI+QIII" if str(branch) == "plus" else "QII+QIV",
        "side": str(side),
        "observable": True,
        "center": [float(center[0]), float(center[1])],
        "phi_start_deg": float(np.degrees(phi_start)),
        "phi_stop_deg": float(np.degrees(phi_stop)),
        "arc_span_deg": float(np.degrees(phi_stop - phi_start)),
        "amplitude": float(amplitude),
        "detector_width_q": float(width),
        "polyline_q": np.asarray(polyline, dtype=float).tolist(),
        "geometry": "ellipse" if ellipse else "smooth_nonellipse",
    }
    if ellipse:
        assert a is not None and b is not None and theta_deg is not None
        record.update({
            "a": float(a),
            "b": float(b),
            "b_over_a": float(b) / float(a),
            "axis_ratio": float(b) / float(a),
            "theta_deg": float(theta_deg),
        })
    return record


def _base_arc_ranges(spec: ArcCaseSpec, branch: str) -> tuple[tuple[str, float, float], ...]:
    if spec.partial:
        start, stop = 0.35, 0.75
    else:
        start, stop = 0.20, 1.20
    # ``upper``/``lower`` follow the application's observed-arc convention:
    # they are the two sides of the branch's own major-axis frame.  The
    # branch-specific quadrants below keep plus in QI+QIII and minus in
    # QII+QIV, so a topology tracer can recover both mirror components without
    # guessing from the generator's labels.
    if branch == "plus":
        return (
            ("upper", start, stop),
            ("lower", np.pi + start, np.pi + stop),
        )
    return (
        ("upper", np.pi - stop, np.pi - start),
        ("lower", 2.0 * np.pi - stop, 2.0 * np.pi - start),
    )


def generate_arc_case(
    case_id: str | ArcCaseSpec | Mapping[str, Any],
    seed: int | None = None,
    shape: Sequence[int] = DEFAULT_SHAPE,
) -> dict[str, Any]:
    """Generate one independent, known-observable arc image.

    Parameters
    ----------
    case_id:
        A default case name, :class:`ArcCaseSpec`, or a small override mapping.
    seed:
        Optional seed override.  The case's public seed is used by default.
    shape:
        Detector rows and columns.  The returned image, q map and mask always
        have exactly this shape.
    """

    spec = _resolve_case(case_id)
    detector_shape = _validate_shape(shape)
    if seed is not None and (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))):
        raise TypeError("seed must be an integer")
    actual_seed = int(spec.seed if seed is None else seed)
    identity = _case_identity(spec.case_id, actual_seed, detector_shape)
    rng = np.random.default_rng(actual_seed)
    qx, qy = _q_grid(detector_shape, warped=bool(spec.warped_qmap))
    mask = np.zeros(detector_shape, dtype=bool)
    qmap = ArcQMap(
        qx=qx,
        qy=qy,
        q=np.hypot(qx, qy),
        mask=mask.copy(),
        metadata={
            "q_unit": Q_UNIT,
            "generator_version": GENERATOR_VERSION,
            "warped": bool(spec.warped_qmap),
            "coordinate_scope": "detector_pixel_centres",
        },
    )

    background = 0.018
    image = np.full(detector_shape, background, dtype=float)
    truth: dict[str, Any] = {
        "truth_scope": "known_observable_arc_geometry_only",
        "model_scope": MODEL_SCOPE,
        "generator_version": GENERATOR_VERSION,
        "case_id": spec.case_id,
        "category": spec.category,
        "seed": actual_seed,
        "shape": list(detector_shape),
        "case_identity": identity,
        "case_identity_sha256": _case_identity_hash(identity),
        "q_unit": Q_UNIT,
        "mask_semantics": {"true": "invalid_detector_pixel", "valid_mask_true": "valid detector pixel"},
        "qmap": {"warped": bool(spec.warped_qmap), "coordinate_field": "qx/qy"},
        "detector_pixel_integration": {
            "subpixels_per_axis": 3,
            "method": "distance_field_area_average_local_jacobian",
            "cross_component_jacobian": True,
        },
        "blur": {"sigma_pixels": 0.65, "method": "gaussian"},
        "actual_observable_arcs": [],
        "observable_arcs": [],
        "branches": {},
        "centers": {},
        "is_elliptic": bool(not spec.non_elliptic and not spec.null),
    }

    if not spec.null:
        base_a = 0.72
        nominal_ratio = float(spec.axis_ratio) if spec.axis_ratio is not None else 0.10
        base_theta = 17.0
        branch_settings = {
            "plus": {
                "theta_deg": base_theta,
                "center": (0.0, 0.0),
                "a": base_a,
                "ratio": nominal_ratio,
                "amplitude": 1.0,
            },
            "minus": {
                "theta_deg": -base_theta,
                "center": (0.0, 0.0),
                "a": base_a,
                "ratio": nominal_ratio,
                "amplitude": 0.92,
            },
        }
        if spec.overlap:
            branch_settings["plus"].update({"theta_deg": 5.0, "center": (-0.018, 0.0), "amplitude": 0.88})
            branch_settings["minus"].update({"theta_deg": -5.0, "center": (0.018, 0.0), "amplitude": 0.84})
        if spec.asymmetric:
            branch_settings["plus"].update({"ratio": nominal_ratio, "amplitude": 1.15})
            branch_settings["minus"].update({"ratio": min(0.45, nominal_ratio * 1.45), "amplitude": 0.48})
        if spec.miscentered:
            branch_settings["plus"].update({"center": (0.015, -0.012)})
            branch_settings["minus"].update({"center": (-0.015, 0.012)})

        qx_row, qx_col = np.gradient(qx)
        qy_row, qy_col = np.gradient(qy)
        row_pixel_q = np.hypot(qx_row, qy_row)
        col_pixel_q = np.hypot(qx_col, qy_col)
        local_pixel_q = np.concatenate((row_pixel_q[np.isfinite(row_pixel_q)], col_pixel_q[np.isfinite(col_pixel_q)]))
        width = max(0.010, float(np.median(local_pixel_q)) * 0.80 if local_pixel_q.size else 0.010)
        for branch_index, (branch, values) in enumerate(branch_settings.items()):
            if spec.missing_branch and branch == "minus":
                truth["branches"][branch] = {"present": False, "sides": [], "reason": "branch_absent_by_generator"}
                continue
            center = tuple(float(value) for value in values["center"])
            a_value = float(values["a"])
            b_value = a_value * float(values["ratio"])
            truth["centers"][branch] = [center[0], center[1]]
            side_records: list[dict[str, Any]] = []
            for side, start, stop in _base_arc_ranges(spec, branch):
                # The missing-branch condition removes a whole branch.  It is
                # intentionally not encoded as a detector mask.
                phi_start = float(start)
                phi_stop = float(stop)
                if spec.non_elliptic:
                    polyline = _nonellipse_polyline(
                        center,
                        phi_start,
                        phi_stop,
                        branch_sign=1.0 if branch == "plus" else -1.0,
                    )
                else:
                    polyline = _ellipse_polyline(
                        center,
                        a_value,
                        b_value,
                        float(values["theta_deg"]),
                        phi_start,
                        phi_stop,
                    )
                amplitude = float(values["amplitude"])
                if side == "lower":
                    amplitude *= 0.95 if not spec.asymmetric else 0.82
                arc = _arc_record(
                    arc_id=f"{branch}_{side}",
                    branch=branch,
                    side=side,
                    polyline=polyline,
                    center=center,
                    phi_start=phi_start,
                    phi_stop=phi_stop,
                    amplitude=amplitude,
                    width=width,
                    ellipse=not spec.non_elliptic,
                    a=a_value if not spec.non_elliptic else None,
                    b=b_value if not spec.non_elliptic else None,
                    theta_deg=float(values["theta_deg"]) if not spec.non_elliptic else None,
                )
                side_records.append(arc)
                truth["actual_observable_arcs"].append(arc)
                truth["observable_arcs"].append(arc)
                image += _rasterize_arc(
                    qx,
                    qy,
                    polyline,
                    amplitude=amplitude,
                    width=width,
                    subpixel=3,
                )
            truth["branches"][branch] = {
                "present": True,
                "sides": [record["side"] for record in side_records],
                "arc_ids": [record["arc_id"] for record in side_records],
                "center": [center[0], center[1]],
                "quadrant_pair": "QI+QIII" if branch == "plus" else "QII+QIV",
                "theta_deg": float(values["theta_deg"]) if not spec.non_elliptic else None,
            }

        if not spec.non_elliptic:
            # Keep the canonical nominal axes available for ordinary ellipse
            # cases, while branch_axes retains asymmetric branch truth.
            truth["axes"] = {"a": base_a, "b": base_a * nominal_ratio, "b_over_a": nominal_ratio}
            truth["a"] = base_a
            truth["b"] = base_a * nominal_ratio
            truth["b_over_a"] = nominal_ratio
            truth["axis_ratio"] = nominal_ratio
            truth["branch_axes"] = {
                branch: {
                    "a": float(values["a"]),
                    "b": float(values["a"] * values["ratio"]),
                    "b_over_a": float(values["ratio"]),
                    "theta_deg": float(values["theta_deg"]),
                }
                for branch, values in branch_settings.items()
                if not (spec.missing_branch and branch == "minus")
            }

    truth["branch_side_truth"] = [
        {
            "branch": arc["branch"],
            "branch_id": arc["branch_id"],
            "quadrant_pair": arc["quadrant_pair"],
            "side": arc["side"],
            "arc_id": arc["arc_id"],
            "observable": True,
        }
        for arc in truth["actual_observable_arcs"]
    ]
    blur_sigma = 0.65
    image = _blur(image, blur_sigma)
    # Keep the baseline and weak background physically harmless for fitting
    # callbacks while retaining a small deterministic image-level noise term.
    noise_sigma = 0.004 if not spec.null else 0.002
    noise = rng.normal(0.0, noise_sigma, size=detector_shape)
    image = np.clip(image + noise, 0.0, None)
    image[mask] = np.nan
    truth["noise"] = {"model": "gaussian_image_noise", "sigma": noise_sigma, "seeded": True}
    truth["image_observable_arc_count"] = len(truth["actual_observable_arcs"])
    truth["calibration_claim"] = False
    # ``source_hash`` predates the explicit provenance contract and is kept
    # unchanged so old consumers and frozen run records remain readable.  It
    # is a legacy case recipe hash, not a hash of this module.  New records
    # must use ``generator_sha256`` for source provenance and the separate
    # identity fields above for case/seed/shape provenance.
    truth["source_hash"] = hashlib.sha256(
        (GENERATOR_VERSION + "|" + spec.case_id).encode("utf-8")
    ).hexdigest()
    truth["generator_sha256"] = GENERATOR_HASH
    return {
        "image": image,
        "qmap": qmap,
        "mask": mask,
        "valid_mask": ~mask,
        "truth": truth,
        "case_id": spec.case_id,
        "spec": spec,
        "qx": qx,
        "qy": qy,
        "q": qmap.q,
    }


generate_case = generate_arc_case


__all__ = [
    "ArcCaseSpec",
    "ArcQMap",
    "ARC_CASES",
    "CASE_MATRIX",
    "DEFAULT_CASES",
    "DEFAULT_ARC_CASES",
    "DEFAULT_CASE_IDS",
    "DEFAULT_SHAPE",
    "GENERATOR_HASH",
    "GENERATOR_VERSION",
    "MODEL_SCOPE",
    "Q_UNIT",
    "default_cases",
    "generate_arc_case",
    "generate_case",
]


GENERATOR_HASH = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
