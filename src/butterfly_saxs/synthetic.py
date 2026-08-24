"""Deterministic synthetic butterfly-pattern data for tests and demos."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .intensity import double_ellipse_intensity, parameter_values


SYNTHETIC_FLAGS = (
    "empirical_model_only",
    "nonunique_inverse_problem",
    "synthetic_data",
)


@dataclass
class SyntheticFrame:
    """Fallback frame used when the application data-model is not installed."""

    data: np.ndarray
    mask: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def intensity(self) -> np.ndarray:
        return self.data


@dataclass
class SyntheticQMap:
    """Deterministic q map whose synthetic model coordinates are nm^-1 by default."""

    qx: np.ndarray
    qy: np.ndarray
    # The synthetic model is parameterized in nm^-1 by default.  Keeping this
    # declaration on the q map makes the unit provenance explicit while still
    # allowing tests to exercise Å^-1, pixel-q, and unknown maps.
    q_unit: str = "nm^-1"

    @property
    def q(self) -> np.ndarray:
        return np.hypot(self.qx, self.qy)

    @property
    def angle(self) -> np.ndarray:
        return np.arctan2(self.qy, self.qx)

    @property
    def azimuth(self) -> np.ndarray:
        return self.angle


@dataclass
class SyntheticSequence:
    """Batch of frames, q maps, times, and parameter truth.

    The object is sequence-like, so existing batch code can use either
    ``sequence.frames`` or ``for frame in sequence``.
    """

    frames: tuple[Any, ...]
    qmaps: tuple[Any, ...]
    times: np.ndarray
    parameters: tuple[dict[str, Any], ...]
    seed: int
    flags: tuple[str, ...] = SYNTHETIC_FLAGS

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, index: int) -> Any:
        return self.frames[index]

    def __iter__(self):
        return iter(self.frames)

    @property
    def truth(self) -> tuple[dict[str, Any], ...]:
        return self.parameters

    def as_dict(self) -> dict[str, Any]:
        return {
            "frames": self.frames,
            "qmaps": self.qmaps,
            "times": self.times,
            "parameters": self.parameters,
            "seed": self.seed,
            "flags": self.flags,
        }


def _try_model_frame(data: np.ndarray, mask: np.ndarray, metadata: Mapping[str, Any]) -> Any:
    """Use the shared ImageFrame when its constructor is available."""

    try:
        from .models import ImageFrame  # type: ignore
    except Exception:
        return SyntheticFrame(data=data, mask=mask, metadata=dict(metadata))
    attempts = (
        {"data": data, "mask": mask, "metadata": dict(metadata)},
        {"intensity": data, "mask": mask, "metadata": dict(metadata)},
        {"data": data, "mask": mask},
        {"intensity": data, "mask": mask},
    )
    for kwargs in attempts:
        try:
            return ImageFrame(**kwargs)
        except Exception:
            continue
    return SyntheticFrame(data=data, mask=mask, metadata=dict(metadata))


def _try_model_qmap(qx: np.ndarray, qy: np.ndarray, q_unit: str = "nm^-1") -> Any:
    try:
        from .models import QMap  # type: ignore
    except Exception:
        return SyntheticQMap(qx=qx, qy=qy, q_unit=q_unit)
    attempts = (
        {"qx": qx, "qy": qy, "q_unit": q_unit},
        {"q_x": qx, "q_y": qy, "q_unit": q_unit},
    )
    for kwargs in attempts:
        try:
            return QMap(**kwargs)
        except Exception:
            continue
    return SyntheticQMap(qx=qx, qy=qy, q_unit=q_unit)


def _frame_parameters(base: Mapping[str, Any], index: int, n_frames: int, evolution: Any) -> dict[str, Any]:
    values = dict(base)
    if evolution is None:
        return values
    if callable(evolution):
        try:
            candidate = evolution(index, dict(values))
        except TypeError:
            candidate = evolution(index)
        if isinstance(candidate, Mapping):
            values.update(candidate)
        return values
    if isinstance(evolution, Mapping):
        fraction = 0.0 if n_frames <= 1 else float(index) / float(n_frames - 1)
        for key, trajectory in evolution.items():
            if isinstance(trajectory, Sequence) and not isinstance(trajectory, (str, bytes)) and len(trajectory) >= 2:
                try:
                    start, stop = float(trajectory[0]), float(trajectory[-1])
                    values[key] = start + fraction * (stop - start)
                except (TypeError, ValueError):
                    values[key] = trajectory[min(index, len(trajectory) - 1)]
            else:
                values[key] = trajectory
    return values


def make_butterfly_sequence(
    n_frames: int = 5,
    *,
    shape: tuple[int, int] = (128, 128),
    q_extent: float | None = None,
    parameters: Any = None,
    evolution: Mapping[str, Sequence[float]] | Callable[..., Mapping[str, Any]] | None = None,
    seed: int = 0,
    noise_sigma: float = 0.0,
    noise_model: str = "gaussian",
    poisson_scale: float = 1_000.0,
    mask_fraction: float = 0.0,
    time_step: float = 1.0,
    origin: tuple[float, float] | None = None,
    q_unit: str = "nm^-1",
) -> SyntheticSequence:
    """Create a deterministic sequence of synthetic four-lobe patterns.

    ``evolution`` accepts ``{"a": (start, stop), ...}`` or a callback.  Set
    ``noise_sigma=0`` for exact parameter-recovery tests.  Masks are invalid
    detector pixels (``True`` means masked), matching the observation adapter.
    """

    if int(n_frames) < 1:
        raise ValueError("n_frames must be positive")
    if len(shape) != 2 or any(int(size) < 8 for size in shape):
        raise ValueError("shape must be a two-dimensional size of at least 8x8")
    if noise_model not in {"gaussian", "poisson", "none"}:
        raise ValueError("noise_model must be 'gaussian', 'poisson', or 'none'")
    if not 0.0 <= float(mask_fraction) < 1.0:
        raise ValueError("mask_fraction must be in [0, 1)")
    rng = np.random.default_rng(int(seed))
    base = parameter_values(parameters)
    qscale = float(q_extent) if q_extent is not None else 1.35 * max(abs(float(base.get("a", 1.0))), abs(float(base.get("b", 0.7))), 1e-3)
    if qscale <= 0 or not np.isfinite(qscale):
        raise ValueError("q_extent must be finite and positive")
    rows, cols = int(shape[0]), int(shape[1])
    if origin is None:
        cy, cx = (rows - 1) / 2.0, (cols - 1) / 2.0
    else:
        cy, cx = map(float, origin)
    y = np.linspace(-qscale, qscale, rows) - (cy - (rows - 1) / 2.0) * (2.0 * qscale / max(rows - 1, 1))
    x = np.linspace(-qscale, qscale, cols) - (cx - (cols - 1) / 2.0) * (2.0 * qscale / max(cols - 1, 1))
    qx, qy = np.meshgrid(x, y)
    qmap = _try_model_qmap(qx, qy, q_unit=q_unit)
    frames: list[Any] = []
    qmaps: list[Any] = []
    truths: list[dict[str, Any]] = []
    for index in range(int(n_frames)):
        truth = _frame_parameters(base, index, int(n_frames), evolution)
        exact = np.asarray(double_ellipse_intensity(qx, qy, truth), dtype=float)
        if noise_model == "gaussian" and float(noise_sigma) > 0:
            observed = exact + rng.normal(0.0, float(noise_sigma), size=exact.shape)
        elif noise_model == "poisson":
            scale = max(float(poisson_scale), np.finfo(float).eps)
            observed = rng.poisson(np.clip(exact, 0.0, None) * scale) / scale
            if float(noise_sigma) > 0:
                observed = observed + rng.normal(0.0, float(noise_sigma), size=exact.shape)
        else:
            observed = exact.copy()
        mask = rng.random(exact.shape) < float(mask_fraction) if mask_fraction else np.zeros(exact.shape, dtype=bool)
        metadata = {
            "time": float(index) * float(time_step),
            "frame_index": index,
            "seed": int(seed),
            "truth": dict(truth),
            "flags": SYNTHETIC_FLAGS,
        }
        frames.append(_try_model_frame(observed, mask, metadata))
        qmaps.append(qmap)
        truths.append(dict(truth))
    return SyntheticSequence(
        frames=tuple(frames),
        qmaps=tuple(qmaps),
        times=np.arange(int(n_frames), dtype=float) * float(time_step),
        parameters=tuple(truths),
        seed=int(seed),
    )


generate_butterfly_sequence = make_butterfly_sequence
synthetic_butterfly_sequence = make_butterfly_sequence


def make_butterfly_frame(
    *,
    shape: tuple[int, int] = (128, 128),
    parameters: Any = None,
    seed: int = 0,
    noise_sigma: float = 0.0,
    **kwargs: Any,
) -> tuple[Any, Any, dict[str, Any]]:
    """Convenience wrapper returning one frame, q map, and truth mapping."""

    sequence = make_butterfly_sequence(
        1,
        shape=shape,
        parameters=parameters,
        seed=seed,
        noise_sigma=noise_sigma,
        **kwargs,
    )
    return sequence.frames[0], sequence.qmaps[0], sequence.parameters[0]


generate_butterfly_frame = make_butterfly_frame


__all__ = [
    "SYNTHETIC_FLAGS",
    "SyntheticFrame",
    "SyntheticQMap",
    "SyntheticSequence",
    "make_butterfly_sequence",
    "generate_butterfly_sequence",
    "synthetic_butterfly_sequence",
    "make_butterfly_frame",
    "generate_butterfly_frame",
]
