"""Deterministic lamellar SAXS sequences for pipeline validation.

The T2 sequence reuses the independent real-space stack/FFT source in
:mod:`butterfly_saxs.benchmark_t2`.  The oblique-stack sequence is generated
separately from finite anisotropic layer stacks with independently varied
lamellar tilt and stack rotation. Neither sequence uses a fitted intensity
model or draws a target ellipse.

Both generators are two-dimensional finite-window models. They are useful for
checking continuity, q calibration, missing-pixel handling and false-positive
behavior, but they are not a full three-dimensional Grubb forward model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from scipy.ndimage import gaussian_filter1d

from . import benchmark_t2


@dataclass(frozen=True)
class SequenceSettings:
    """Physical and acquisition settings for one reproducible sequence."""

    n_signal_frames: int = 10
    shape: tuple[int, int] = (512, 512)
    pixel_size_nm: float = 1.0
    spacing_start_nm: float = 11.4
    spacing_end_nm: float = 12.6
    orientation_spread_start: float = 0.85
    orientation_spread_end: float = 1.15
    orientation_center_start_deg: float = -3.0
    orientation_center_end_deg: float = 3.0
    target_first_order_snr: float = 15.0
    noise_sigma: float = 0.025
    spacing_jitter_fraction: float = 0.015
    seed: int = 20260926
    local_missing_lobe_frame: int = 5
    local_missing_lobe_center_deg: float = 23.0
    local_missing_lobe_half_width_deg: float = 13.0
    local_missing_lobe_q_low_fraction: float = 0.72
    local_missing_lobe_q_high_fraction: float = 1.30
    noise_control: bool = True
    noise_control_mean: float = 0.0
    noise_control_sigma: float | None = None
    buried_signal_stress: bool = True
    buried_signal_stress_sigma: float = 0.025
    beamstop_radius_fraction: float = 0.15

    def validate(self) -> None:
        if self.n_signal_frames < 2:
            raise ValueError("n_signal_frames must be at least 2")
        if len(self.shape) != 2 or min(self.shape) < 32:
            raise ValueError("shape must contain two dimensions, each at least 32")
        if not np.isfinite(self.pixel_size_nm) or self.pixel_size_nm <= 0:
            raise ValueError("pixel_size_nm must be finite and positive")
        if (not np.isfinite(self.spacing_start_nm) or self.spacing_start_nm <= 0
                or not np.isfinite(self.spacing_end_nm) or self.spacing_end_nm <= 0):
            raise ValueError("lamellar spacings must be finite and positive")
        if not np.isfinite(self.noise_sigma) or self.noise_sigma < 0:
            raise ValueError("noise_sigma must be finite and non-negative")
        if not np.isfinite(self.target_first_order_snr) or self.target_first_order_snr <= 0:
            raise ValueError("target_first_order_snr must be finite and positive")
        if self.noise_control_sigma is not None and (
            not np.isfinite(self.noise_control_sigma) or self.noise_control_sigma < 0
        ):
            raise ValueError("noise_control_sigma must be finite and non-negative")
        if not np.isfinite(self.buried_signal_stress_sigma) or self.buried_signal_stress_sigma < 0:
            raise ValueError("buried_signal_stress_sigma must be finite and non-negative")
        if not 0 <= self.local_missing_lobe_frame < self.n_signal_frames:
            raise ValueError("local_missing_lobe_frame must select a signal frame")
        if not 0 < self.beamstop_radius_fraction < 1:
            raise ValueError("beamstop_radius_fraction must be between 0 and 1")


def _fixed_beamstop(q: np.ndarray, settings: SequenceSettings) -> np.ndarray:
    """Return one detector-centered beamstop shared by every frame."""

    mean_spacing = 0.5 * (settings.spacing_start_nm + settings.spacing_end_nm)
    q_radius = settings.beamstop_radius_fraction * (2.0 * np.pi / mean_spacing)
    return np.asarray(q <= q_radius, dtype=bool)


def generate_sequence(settings: SequenceSettings | None = None) -> tuple[dict[str, Any], ...]:
    """Build continuous signal frames followed by an optional noise control.

    Signal intensities are produced by the independent T2 real-space density
    and FFT.  The local missing lobe is represented only in the exclusion
    mask; its generated intensity is preserved in the source array.
    """

    settings = settings or SequenceSettings()
    settings.validate()
    base = next(case for case in benchmark_t2.DEFAULT_CASES if case.category == "butterfly")
    base_offsets = np.asarray(base.orientation_offsets_deg, dtype=float)
    frame_fraction = np.linspace(0.0, 1.0, settings.n_signal_frames)
    spacings = np.linspace(
        settings.spacing_start_nm, settings.spacing_end_nm, settings.n_signal_frames
    )
    spreads = np.linspace(
        settings.orientation_spread_start,
        settings.orientation_spread_end,
        settings.n_signal_frames,
    )
    centers = np.linspace(
        settings.orientation_center_start_deg,
        settings.orientation_center_end_deg,
        settings.n_signal_frames,
    )
    generated: list[dict[str, Any]] = []

    for index, (fraction, spacing, spread, center) in enumerate(
        zip(frame_fraction, spacings, spreads, centers)
    ):
        offsets = base_offsets * float(spread)
        weight_sigma = 7.0 * float(spread)
        weight_center = 23.0 * float(spread)
        weights = np.exp(-0.5 * ((np.abs(offsets) - weight_center) / weight_sigma) ** 2)
        case = {
            "case_id": base.case_id,
            "layer_spacing_nm": float(spacing),
            "layer_count": base.layer_count,
            "layer_width_nm": base.layer_width_nm,
            "orientation_deg": float(center),
            "orientation_offsets_deg": tuple(float(value) for value in offsets),
            "orientation_weights": tuple(float(value) for value in weights),
            "curvature_nm": base.curvature_nm,
            "waviness_nm": base.waviness_nm,
            "spacing_jitter_fraction": settings.spacing_jitter_fraction,
            "asymmetry": base.asymmetry,
            "noise_sigma": 0.0,
        }
        frame = benchmark_t2.generate_case(
            case,
            shape=settings.shape,
            seed=settings.seed + index,
            noise_sigma=0.0,
            pixel_size_nm=settings.pixel_size_nm,
        )
        clean = np.asarray(frame["intensity_noiseless"], dtype=float)
        q0 = 2.0 * np.pi / float(spacing)
        noise_sigma = _reference_noise_sigma(
            clean,
            frame["q"],
            q0,
            settings.target_first_order_snr,
            settings.noise_sigma,
        )
        rng = np.random.default_rng(settings.seed + index)
        noise = rng.normal(0.0, noise_sigma, size=settings.shape)
        observed = np.clip(clean + noise, 0.0, None)
        frame["intensity_noisy"] = observed
        frame["intensity"] = observed.copy()
        frame["noise"] = noise
        frame["noise_sigma"] = noise_sigma
        frame["noise_target_first_order_snr"] = settings.target_first_order_snr
        frame["metadata"]["noise_sigma"] = noise_sigma
        frame["metadata"]["noise_target_first_order_snr"] = settings.target_first_order_snr
        beamstop = _fixed_beamstop(frame["q"], settings)
        missing_lobe = np.zeros(settings.shape, dtype=bool)
        if index == settings.local_missing_lobe_frame:
            angle = np.rad2deg(np.arctan2(frame["qy"], frame["qx"]))
            angular_delta = (angle - settings.local_missing_lobe_center_deg + 180.0) % 360.0 - 180.0
            missing_lobe = (
                (np.abs(angular_delta) <= settings.local_missing_lobe_half_width_deg)
                & (frame["q"] >= settings.local_missing_lobe_q_low_fraction * q0)
                & (frame["q"] <= settings.local_missing_lobe_q_high_fraction * q0)
            )
        frame["frame_index"] = index
        frame["frame_id"] = f"signal_{index:02d}"
        frame["sequence_role"] = "lamellar_signal"
        frame["structural_q0_nm_inv"] = q0
        frame["structure_truth"]["sequence_fraction"] = float(fraction)
        frame["structure_truth"]["orientation_spread_scale"] = float(spread)
        frame["structure_truth"]["nominal_orientation_deg"] = float(center)
        frame["noise_reference"] = _first_order_noise_reference(
            clean, frame["q"], q0, noise_sigma
        )
        frame["detector_beamstop_mask"] = beamstop
        frame["missing_lobe_mask"] = missing_lobe
        frame["mask"] = beamstop | missing_lobe
        frame["valid_mask"] = ~frame["mask"]
        frame["mask_diagnostics"] = {
            "beamstop_pixels": int(np.count_nonzero(beamstop)),
            "missing_lobe_pixels": int(np.count_nonzero(missing_lobe)),
            "masked_pixels": int(np.count_nonzero(frame["mask"])),
        }
        generated.append(frame)

    if settings.buried_signal_stress and generated:
        reference_signal = generated[-1]
        stress_index = len(generated)
        clean = np.asarray(reference_signal["intensity_noiseless"], dtype=float)
        rng = np.random.default_rng(settings.seed + stress_index)
        noise = rng.normal(0.0, settings.buried_signal_stress_sigma, size=settings.shape)
        observed = np.clip(clean + noise, 0.0, None)
        stress = dict(reference_signal)
        stress.update(
            {
                "frame_index": stress_index,
                "frame_id": "buried_signal_stress",
                "sequence_role": "buried_signal_stress",
                "seed": settings.seed + stress_index,
                "noise_sigma": settings.buried_signal_stress_sigma,
                "noise_target_first_order_snr": None,
                "intensity_noisy": observed,
                "intensity": observed.copy(),
                "noise": noise,
                "detector_beamstop_mask": np.asarray(reference_signal["detector_beamstop_mask"]).copy(),
                "missing_lobe_mask": np.asarray(reference_signal["missing_lobe_mask"]).copy(),
                "mask": np.asarray(reference_signal["mask"]).copy(),
                "valid_mask": np.asarray(reference_signal["valid_mask"]).copy(),
                "metadata": dict(reference_signal["metadata"], noise_sigma=settings.buried_signal_stress_sigma),
                "noise_reference": _first_order_noise_reference(
                    clean, reference_signal["q"], reference_signal["structural_q0_nm_inv"],
                    settings.buried_signal_stress_sigma,
                ),
            }
        )
        generated.append(stress)

    if settings.noise_control:
        reference = generated[-1]
        main_sigmas = [float(frame["noise_sigma"]) for frame in generated
                       if frame["sequence_role"] == "lamellar_signal"]
        control_sigma = (
            settings.noise_control_sigma
            if settings.noise_control_sigma is not None
            else float(np.median(main_sigmas)) if main_sigmas else settings.noise_sigma
        )
        control_seed = settings.seed + settings.n_signal_frames + 1
        rng = np.random.default_rng(control_seed)
        observed = np.clip(
            settings.noise_control_mean + rng.normal(0.0, control_sigma, size=settings.shape),
            0.0,
            None,
        )
        beamstop = _fixed_beamstop(reference["q"], settings)
        noise_control = {
            "frame_index": len(generated),
            "frame_id": "noise_control",
            "sequence_role": "noise_only_control",
            "seed": control_seed,
            "noise_sigma": control_sigma,
            "noise_control_mean": settings.noise_control_mean,
            "real_space_density": np.zeros(settings.shape, dtype=float),
            "intensity_noiseless": np.zeros(settings.shape, dtype=float),
            "intensity_clean": np.zeros(settings.shape, dtype=float),
            "intensity_noisy": observed,
            "intensity": observed.copy(),
            "noise": observed - settings.noise_control_mean,
            "qx": reference["qx"],
            "qy": reference["qy"],
            "q": reference["q"],
            "q_unit": benchmark_t2.T2_Q_UNIT,
            "mask": beamstop.copy(),
            "valid_mask": ~beamstop,
            "detector_beamstop_mask": beamstop,
            "missing_lobe_mask": np.zeros(settings.shape, dtype=bool),
            "structural_q0_nm_inv": None,
            "structure_truth": None,
            "projection_reference": np.empty((0, 2), dtype=float),
            "projection_truth": None,
            "mask_diagnostics": {
                "beamstop_pixels": int(np.count_nonzero(beamstop)),
                "missing_lobe_pixels": 0,
                "masked_pixels": int(np.count_nonzero(beamstop)),
            },
        }
        generated.append(noise_control)

    return tuple(generated)


def _local_first_order_values(
    intensity: np.ndarray, q: np.ndarray, q0: float, *, half_width_fraction: float = 0.12
) -> np.ndarray:
    image = np.asarray(intensity, dtype=float)
    q_values = np.asarray(q, dtype=float)
    keep = (
        np.isfinite(image)
        & np.isfinite(q_values)
        & (q_values >= q0 * (1.0 - half_width_fraction))
        & (q_values <= q0 * (1.0 + half_width_fraction))
    )
    return image[keep]


def _reference_noise_sigma(
    intensity: np.ndarray, q: np.ndarray, q0: float, target_snr: float, fallback: float
) -> float:
    values = _local_first_order_values(intensity, q, q0)
    reference = float(np.percentile(values, 90.0)) if values.size else 0.0
    return reference / target_snr if reference > 0 else float(fallback)


def _first_order_noise_reference(
    intensity: np.ndarray, q: np.ndarray, q0: float, noise_sigma: float
) -> dict[str, float | int | None]:
    values = _local_first_order_values(intensity, q, q0)
    if not values.size:
        return {"pixel_count": 0, "mean_intensity": None, "p90_intensity": None,
                "max_intensity": None, "noise_sigma": float(noise_sigma), "p90_snr": None}
    p90 = float(np.percentile(values, 90.0))
    return {
        "pixel_count": int(values.size),
        "mean_intensity": float(np.mean(values)),
        "p90_intensity": p90,
        "max_intensity": float(np.max(values)),
        "noise_sigma": float(noise_sigma),
        "p90_snr": float(p90 / noise_sigma) if noise_sigma > 0 else None,
    }


@dataclass(frozen=True)
class ObliqueStackSettings:
    """Settings for a finite tilted-lamella, rotated-stack 2D sequence."""

    n_signal_frames: int = 8
    shape: tuple[int, int] = (512, 512)
    pixel_size_nm: float = 1.0
    spacing_start_nm: float = 11.4
    spacing_end_nm: float = 12.6
    tilt_start_deg: float = 18.0
    tilt_end_deg: float = 30.0
    stack_rotation_start_deg: float = 25.0
    stack_rotation_end_deg: float = 12.0
    orientation_offsets_deg: tuple[float, ...] = (-8., -6., -4., -2., 0., 2., 4., 6., 8.)
    stack_width_nm: float = 70.0
    stack_height_nm: float = 24.0
    slab_sigma_nm: float = 1.2
    layer_count: int = 14
    target_first_order_snr: float = 15.0
    seed: int = 20260927
    local_missing_lobe_frame: int = 4
    local_missing_lobe_center_deg: float = 24.0
    local_missing_lobe_half_width_deg: float = 12.0
    beamstop_radius_fraction: float = 0.12
    include_clean_control: bool = True
    include_noise_control: bool = True
    noise_control_mean: float = 0.0

    def validate(self) -> None:
        if self.n_signal_frames < 2:
            raise ValueError("n_signal_frames must be at least 2")
        if len(self.shape) != 2 or min(self.shape) < 32:
            raise ValueError("shape must contain two dimensions, each at least 32")
        positive = (self.pixel_size_nm, self.spacing_start_nm, self.spacing_end_nm,
                    self.stack_width_nm, self.stack_height_nm, self.slab_sigma_nm,
                    self.target_first_order_snr)
        if any(not np.isfinite(value) or value <= 0 for value in positive):
            raise ValueError("pixel size, spacings, dimensions, slab width and SNR must be positive")
        if self.layer_count < 2:
            raise ValueError("layer_count must be at least 2")
        if not 0 <= self.local_missing_lobe_frame < self.n_signal_frames:
            raise ValueError("local_missing_lobe_frame must select a signal frame")
        if not 0 < self.beamstop_radius_fraction < 1:
            raise ValueError("beamstop_radius_fraction must be between 0 and 1")


def _oblique_component_density(
    x: np.ndarray,
    y: np.ndarray,
    spacing_nm: float,
    tilt_deg: float,
    stack_rotation_deg: float,
    settings: ObliqueStackSettings,
    phase_offset_nm: float,
) -> np.ndarray:
    """Make Gaussian lamellar slabs inside a finite anisotropic stack envelope."""

    phi = np.deg2rad(tilt_deg)
    alpha = np.deg2rad(stack_rotation_deg)
    normal_coord = x * np.cos(phi) + y * np.sin(phi)
    stack_coord = x * np.cos(alpha) + y * np.sin(alpha)
    transverse_coord = -x * np.sin(alpha) + y * np.cos(alpha)
    envelope = np.exp(
        -0.5 * (stack_coord / settings.stack_width_nm) ** 2
        -0.5 * (transverse_coord / settings.stack_height_nm) ** 2
    )
    layer_centers = (
        np.arange(settings.layer_count, dtype=float) - 0.5 * (settings.layer_count - 1)
    ) * spacing_nm + phase_offset_nm
    lamellae = np.zeros_like(envelope)
    for center in layer_centers:
        lamellae += np.exp(-0.5 * ((normal_coord - center) / settings.slab_sigma_nm) ** 2)
    return envelope * lamellae


def generate_oblique_stack_sequence(
    settings: ObliqueStackSettings | None = None,
) -> tuple[dict[str, Any], ...]:
    """Generate a literature-guided tilted lamella/rotated stack sequence.

    The positive and negative tilt populations use opposite stack rotations.
    Component densities are transformed independently and their powers are
    summed incoherently. The generated intensity therefore comes from finite
    real-space structure, not an analytic ellipse or a fitted intensity model.
    """

    settings = settings or ObliqueStackSettings()
    settings.validate()
    rows, cols = settings.shape
    y_axis = (np.arange(rows, dtype=float) - 0.5 * (rows - 1)) * settings.pixel_size_nm
    x_axis = (np.arange(cols, dtype=float) - 0.5 * (cols - 1)) * settings.pixel_size_nm
    x, y = np.meshgrid(x_axis, y_axis)
    qx, qy, q = benchmark_t2._q_grid(settings.shape, settings.pixel_size_nm)
    spacings = np.linspace(settings.spacing_start_nm, settings.spacing_end_nm,
                           settings.n_signal_frames)
    tilts = np.linspace(settings.tilt_start_deg, settings.tilt_end_deg,
                        settings.n_signal_frames)
    rotations = np.linspace(settings.stack_rotation_start_deg,
                            settings.stack_rotation_end_deg, settings.n_signal_frames)
    generated: list[dict[str, Any]] = []
    angle_offsets = np.asarray(settings.orientation_offsets_deg, dtype=float)
    if not angle_offsets.size:
        angle_offsets = np.asarray([0.0])
    component_weights = np.exp(-0.5 * (angle_offsets / max(np.ptp(angle_offsets) / 3.0, 1.0)) ** 2)
    component_weights /= np.sum(component_weights)

    for index, (spacing, tilt, rotation) in enumerate(zip(spacings, tilts, rotations)):
        q0 = 2.0 * np.pi / float(spacing)
        rng = np.random.default_rng(settings.seed + index)
        clean_power = np.zeros(settings.shape, dtype=float)
        for sign in (1.0, -1.0):
            for offset, weight in zip(angle_offsets, component_weights):
                phi = sign * (float(tilt) + float(offset))
                alpha = -sign * (float(rotation) + 0.5 * float(offset))
                phase = rng.uniform(-0.25, 0.25) * spacing
                density = _oblique_component_density(
                    x, y, spacing, phi, alpha, settings, phase
                )
                fourier = np.fft.fftshift(np.fft.fft2(density - float(np.mean(density))))
                clean_power += float(weight) * np.abs(fourier) ** 2
        clean_max = float(np.max(clean_power))
        if clean_max <= 0 or not np.isfinite(clean_max):
            raise ValueError("oblique stack produced an unusable FFT intensity")
        clean = clean_power / clean_max
        sigma = _reference_noise_sigma(clean, q, q0, settings.target_first_order_snr, 0.0)
        noise_rng = np.random.default_rng(settings.seed + 10_000 + index)
        noise = noise_rng.normal(0.0, sigma, size=settings.shape)
        observed = np.clip(clean + noise, 0.0, None)
        beamstop_q = settings.beamstop_radius_fraction * q0
        beamstop = q <= beamstop_q
        missing = np.zeros(settings.shape, dtype=bool)
        if index == settings.local_missing_lobe_frame:
            azimuth = np.rad2deg(np.arctan2(qy, qx))
            delta = (azimuth - settings.local_missing_lobe_center_deg + 180.0) % 360.0 - 180.0
            missing = (
                (np.abs(delta) <= settings.local_missing_lobe_half_width_deg)
                & (q >= 0.72 * q0)
                & (q <= 1.30 * q0)
            )
        truth = {
            "model": "finite_gaussian_slab_stacks_incoherent_fft_power_sum",
            "layer_spacing_nm": float(spacing),
            "layer_count": settings.layer_count,
            "slab_sigma_nm": settings.slab_sigma_nm,
            "stack_width_nm": settings.stack_width_nm,
            "stack_height_nm": settings.stack_height_nm,
            "positive_tilt_deg": float(tilt),
            "negative_tilt_deg": -float(tilt),
            "positive_population_stack_rotation_deg": -float(rotation),
            "negative_population_stack_rotation_deg": float(rotation),
            "orientation_offsets_deg": angle_offsets.tolist(),
            "orientation_weights": component_weights.tolist(),
            "q0_nm_inv": q0,
            "geometry_reference_type": "real_space_generator_parameters_not_projection_target",
        }
        generated.append({
            "frame_index": index,
            "frame_id": f"oblique_{index:02d}",
            "sequence_role": "oblique_stack_signal",
            "seed": settings.seed + index,
            "noise_sigma": sigma,
            "noise_target_first_order_snr": settings.target_first_order_snr,
            "noise_reference": _first_order_noise_reference(clean, q, q0, sigma),
            "real_space_density": None,
            "intensity_noiseless": clean,
            "intensity_clean": clean.copy(),
            "intensity_noisy": observed,
            "intensity": observed.copy(),
            "noise": noise,
            "qx": qx,
            "qy": qy,
            "q": q,
            "q_unit": benchmark_t2.T2_Q_UNIT,
            "mask": beamstop | missing,
            "valid_mask": ~(beamstop | missing),
            "detector_beamstop_mask": beamstop,
            "missing_lobe_mask": missing,
            "structural_q0_nm_inv": q0,
            "structure_truth": truth,
            "projection_reference": np.empty((0, 2), dtype=float),
            "projection_truth": None,
            "mask_diagnostics": {
                "beamstop_pixels": int(np.count_nonzero(beamstop)),
                "missing_lobe_pixels": int(np.count_nonzero(missing)),
                "masked_pixels": int(np.count_nonzero(beamstop | missing)),
            },
        })

    if settings.include_clean_control and generated:
        reference = generated[settings.local_missing_lobe_frame]
        clean = dict(reference)
        clean.update({
            "frame_index": len(generated),
            "frame_id": "clean_control",
            "sequence_role": "noise_free_signal_control",
            "seed": settings.seed,
            "noise_sigma": 0.0,
            "noise_target_first_order_snr": None,
            "intensity_noisy": np.asarray(reference["intensity_noiseless"]).copy(),
            "intensity": np.asarray(reference["intensity_noiseless"]).copy(),
            "noise": np.zeros(settings.shape, dtype=float),
            "missing_lobe_mask": np.zeros(settings.shape, dtype=bool),
            "mask": np.asarray(reference["detector_beamstop_mask"]).copy(),
            "valid_mask": ~np.asarray(reference["detector_beamstop_mask"]).copy(),
            "mask_diagnostics": {
                "beamstop_pixels": int(np.count_nonzero(reference["detector_beamstop_mask"])),
                "missing_lobe_pixels": 0,
                "masked_pixels": int(np.count_nonzero(reference["detector_beamstop_mask"])),
            },
            "noise_reference": _first_order_noise_reference(
                reference["intensity_noiseless"], reference["q"],
                float(reference["structural_q0_nm_inv"]), 0.0,
            ),
        })
        generated.append(clean)
    if settings.include_noise_control and generated:
        signal_sigmas = [float(frame["noise_sigma"]) for frame in generated
                         if frame["sequence_role"] == "oblique_stack_signal"]
        control_sigma = float(np.median(signal_sigmas)) if signal_sigmas else 0.0
        reference = generated[0]
        control_seed = settings.seed + 20_000
        rng = np.random.default_rng(control_seed)
        noise = rng.normal(0.0, control_sigma, size=settings.shape)
        observed = np.clip(settings.noise_control_mean + noise, 0.0, None)
        beamstop = np.asarray(reference["detector_beamstop_mask"], dtype=bool).copy()
        generated.append({
            "frame_index": len(generated),
            "frame_id": "noise_control",
            "sequence_role": "noise_only_control",
            "seed": control_seed,
            "noise_sigma": control_sigma,
            "noise_target_first_order_snr": None,
            "noise_reference": None,
            "noise_control_mean": settings.noise_control_mean,
            "real_space_density": np.zeros(settings.shape, dtype=float),
            "intensity_noiseless": np.zeros(settings.shape, dtype=float),
            "intensity_clean": np.zeros(settings.shape, dtype=float),
            "intensity_noisy": observed,
            "intensity": observed.copy(),
            "noise": noise,
            "qx": qx,
            "qy": qy,
            "q": q,
            "q_unit": benchmark_t2.T2_Q_UNIT,
            "mask": beamstop,
            "valid_mask": ~beamstop,
            "detector_beamstop_mask": beamstop,
            "missing_lobe_mask": np.zeros(settings.shape, dtype=bool),
            "structural_q0_nm_inv": None,
            "structure_truth": None,
            "projection_reference": np.empty((0, 2), dtype=float),
            "projection_truth": None,
            "mask_diagnostics": {
                "beamstop_pixels": int(np.count_nonzero(beamstop)),
                "missing_lobe_pixels": 0,
                "masked_pixels": int(np.count_nonzero(beamstop)),
            },
        })
    return tuple(generated)


def measure_annular_local_peak(
    image: np.ndarray,
    q: np.ndarray,
    excluded_mask: np.ndarray,
    q_center: float,
    *,
    half_width_fraction: float = 0.24,
    n_bins: int = 121,
    smooth_sigma_bins: float = 1.2,
) -> dict[str, Any]:
    """Measure a noise-free local radial peak without application fit code.

    The caller supplies a known q0 neighborhood to identify the first-order
    radial feature.  The output is a pixel-derived observable reference; it
    remains distinct from the structure's nominal ``2π / spacing`` value.
    """

    intensity = np.asarray(image, dtype=float)
    q_values = np.asarray(q, dtype=float)
    excluded = np.asarray(excluded_mask, dtype=bool)
    if intensity.ndim != 2 or intensity.shape != q_values.shape or intensity.shape != excluded.shape:
        raise ValueError("image, q, and excluded_mask must be matching 2D arrays")
    if not np.isfinite(q_center) or q_center <= 0:
        raise ValueError("q_center must be finite and positive")
    if not 0 < half_width_fraction < 1 or n_bins < 5:
        raise ValueError("peak search needs a positive local window and at least 5 bins")

    q_min = q_center * (1.0 - half_width_fraction)
    q_max = q_center * (1.0 + half_width_fraction)
    valid = (
        ~excluded
        & np.isfinite(intensity)
        & np.isfinite(q_values)
        & (q_values >= q_min)
        & (q_values <= q_max)
    )
    edges = np.linspace(q_min, q_max, n_bins + 1, dtype=float)
    counts, _ = np.histogram(q_values[valid], bins=edges)
    sums, _ = np.histogram(q_values[valid], bins=edges, weights=intensity[valid])
    populated = counts > 0
    if np.count_nonzero(populated) < 3:
        return {
            "status": "insufficient_radial_support",
            "q_peak_nm_inv": None,
            "spacing_nm": None,
            "search_interval_nm_inv": [float(q_min), float(q_max)],
            "valid_pixel_count": int(np.count_nonzero(valid)),
            "populated_bin_count": int(np.count_nonzero(populated)),
            "peak_prominence": None,
        }
    profile = np.divide(sums, counts, out=np.zeros_like(sums), where=populated)
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    # Interpolate empty bins from neighboring observed annuli for a stable
    # local peak estimate; counts remain separately available as support.
    profile = np.interp(
        np.arange(n_bins, dtype=float),
        np.flatnonzero(populated).astype(float),
        profile[populated],
    )
    smooth = gaussian_filter1d(profile, sigma=float(smooth_sigma_bins), mode="nearest")
    peak_index = int(np.argmax(smooth))
    delta = 0.0
    if 0 < peak_index < len(smooth) - 1:
        left, center, right = smooth[peak_index - 1 : peak_index + 2]
        denominator = left - 2.0 * center + right
        if denominator < 0:
            delta = float(np.clip(0.5 * (left - right) / denominator, -1.0, 1.0))
    dq = float(bin_centers[1] - bin_centers[0])
    q_peak = float(bin_centers[peak_index] + delta * dq)
    baseline = float(np.percentile(smooth, 25.0))
    return {
        "status": "measured_local_peak",
        "method": "independent_noise_free_annular_mean_local_peak",
        "q_peak_nm_inv": q_peak,
        "spacing_nm": float(2.0 * np.pi / q_peak),
        "search_interval_nm_inv": [float(q_min), float(q_max)],
        "valid_pixel_count": int(np.count_nonzero(valid)),
        "populated_bin_count": int(np.count_nonzero(populated)),
        "peak_prominence": float(smooth[peak_index] - baseline),
        "peak_at_search_boundary": bool(peak_index in {0, len(smooth) - 1}),
        "bin_centers_nm_inv": [float(value) for value in bin_centers],
        "annular_mean_intensity": [float(value) for value in profile],
        "smoothed_annular_mean_intensity": [float(value) for value in smooth],
        "annular_pixel_counts": [int(value) for value in counts],
    }


def pipeline_analysis_config(
    *, q_window: Sequence[float] = (0.15, 0.85), ridge_method: str = "butterfly_curvature"
) -> dict[str, Any]:
    """Return a compact, explicit recipe for the empirical pipeline run."""

    if len(q_window) != 2 or not (0 <= float(q_window[0]) < float(q_window[1])):
        raise ValueError("q_window must be an increasing pair")
    if ridge_method not in {"butterfly_curvature", "radial_peak"}:
        raise ValueError("ridge_method must be butterfly_curvature or radial_peak")
    return {
        "analysis": {
            "ridge_method": ridge_method,
            "q_window": [float(q_window[0]), float(q_window[1])],
            "draw_axis_deg": 90.0,
            "ellipse": {"preset": "standard"},
            "butterfly": {"stage": "evaluate", "resamples": 0, "sensitivity": False},
        }
    }


__all__ = [
    "SequenceSettings",
    "ObliqueStackSettings",
    "generate_sequence",
    "generate_oblique_stack_sequence",
    "measure_annular_local_peak",
    "pipeline_analysis_config",
]
