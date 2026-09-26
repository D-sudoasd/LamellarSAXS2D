"""Generate and analyse an independent continuous lamellar SAXS sequence.

Run from the repository root with ``py -3.13 scripts/validate_lamellar_sequence.py``.
The default output is an ignored validation directory under ``results/``.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import asdict
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import sys
import time
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np


REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from butterfly_saxs import benchmark_t2  # noqa: E402
from butterfly_saxs.benchmark_sequence import (  # noqa: E402
    ObliqueStackSettings,
    SequenceSettings,
    generate_oblique_stack_sequence,
    generate_sequence,
    measure_annular_local_peak,
    pipeline_analysis_config,
)
from butterfly_saxs.butterfly import analyze_butterfly  # noqa: E402
from butterfly_saxs.pipeline import analyze_frame  # noqa: E402


OUTPUT_FILES = (
    "summary.json",
    "sequence.csv",
    "t2_intensity_contact_sheet.png",
    "t2_truth_vs_observed.png",
    "oblique_intensity_contact_sheet.png",
    "oblique_truth_vs_observed.png",
    "annular_trajectory_overlay.png",
    "annular_profile_overlay.png",
)


def _finite(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if np.isfinite(parsed) else None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _jsonable(value: object) -> object:
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def _first_order_hint(result: object) -> dict[str, object] | None:
    butterfly = getattr(result, "butterfly", None)
    if not isinstance(butterfly, dict):
        return None
    diagnostics = butterfly.get("diagnostics")
    if not isinstance(diagnostics, dict):
        return None
    hint = diagnostics.get("first_order_q_hint")
    if not isinstance(hint, dict):
        return None
    return {
        key: hint.get(key)
        for key in (
            "q_star",
            "selection_status",
            "reason",
            "n_supported_bins",
            "n_supported_runs",
            "n_significant_peaks",
            "profile_noise_mad",
            "significance_prominence_threshold",
            "candidate_peaks",
        )
        if key in hint
    }


def _ridge_summary(result: object) -> dict[str, object]:
    ridges = getattr(result, "ridges", []) or []
    accepted = [point for point in ridges if point.get("accepted") and point.get("valid", True)]
    radii = [
        float(np.hypot(point["qx"], point["qy"]))
        for point in accepted
        if _finite(point.get("qx")) is not None and _finite(point.get("qy")) is not None
    ]
    return {
        "ridge_point_count": len(ridges),
        "accepted_ridge_point_count": len(accepted),
        "accepted_ridge_q_median_nm_inv": _finite(np.median(radii)) if radii else None,
        "accepted_ridge_q_p10_nm_inv": _finite(np.percentile(radii, 10)) if radii else None,
        "accepted_ridge_q_p90_nm_inv": _finite(np.percentile(radii, 90)) if radii else None,
        "accepted_side_counts": dict(
            Counter(f"{point.get('branch_id')}:{point.get('side')}" for point in accepted)
        ),
    }


def _analyze_curvature(
    frame: dict[str, object], q_window: tuple[float, float], multistart: int
) -> dict[str, object]:
    qmap = {
        "qx": frame["qx"],
        "qy": frame["qy"],
        "q": frame["q"],
        "q_unit": "nm^-1",
    }
    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        analysis_config = pipeline_analysis_config(q_window=q_window)
        analysis_config["analysis"]["ellipse"]["multistart"] = multistart
        result = analyze_frame(
            frame["intensity_noisy"],
            qmap=qmap,
            mask=frame["mask"],
            config=analysis_config,
            full2d=False,
        )
    butterfly = result.butterfly
    fit = butterfly.get("candidate_fit", {}) if isinstance(butterfly, dict) else {}
    quality = butterfly.get("quality", {}) if isinstance(butterfly, dict) else {}
    parameters = butterfly.get("quantitative_parameters", {}) if isinstance(butterfly, dict) else {}
    result_record = {
        "execution_status": "complete",
        "elapsed_s": float(time.perf_counter() - start),
        **_ridge_summary(result),
        "measurement_status": butterfly.get("measurement_status") if isinstance(butterfly, dict) else None,
        "candidate_fit": {
            key: _finite(fit.get(key))
            for key in ("a", "b", "axis_ratio", "theta_deg", "rmse", "condition", "q_star_from_arcs")
        },
        "q_star_source": fit.get("q_star_source"),
        "parameter_status": {
            name: {
                "value": _finite(parameter.get("value")),
                "status": parameter.get("status"),
                "confidence": parameter.get("confidence"),
                "reasons": parameter.get("reasons", []),
            }
            for name, parameter in parameters.items()
            if isinstance(parameter, dict)
        },
        "quality_status": quality.get("status") if isinstance(quality, dict) else None,
        "quality_flags": quality.get("flags", []) if isinstance(quality, dict) else [],
        "first_order_q_hint": _first_order_hint(result),
        "runtime_warnings": sorted({str(item.message) for item in caught}),
    }
    return result_record


def _analyze_radial_trace(
    frame: dict[str, object], q_window: tuple[float, float], multistart: int
) -> dict[str, object]:
    qmap = {
        "qx": frame["qx"],
        "qy": frame["qy"],
        "q": frame["q"],
        "q_unit": "nm^-1",
    }
    start = time.perf_counter()
    analysis_config = pipeline_analysis_config(q_window=q_window, ridge_method="radial_peak")
    analysis_config["analysis"]["ellipse"]["multistart"] = multistart
    result = analyze_frame(
        frame["intensity_noisy"],
        qmap=qmap,
        mask=frame["mask"],
        config=analysis_config,
        full2d=False,
    )
    summary = _ridge_summary(result)
    return {
        "execution_status": "complete",
        "elapsed_s": float(time.perf_counter() - start),
        **summary,
        "method": "application_pipeline_radial_peak",
    }


def _analyze_annular_trajectory(
    frame: dict[str, object], q_window: tuple[float, float], multistart: int
) -> dict[str, object]:
    qmap = {
        "qx": frame["qx"],
        "qy": frame["qy"],
        "q": frame["q"],
        "q_unit": "nm^-1",
    }
    start = time.perf_counter()
    result = analyze_butterfly(
        frame["intensity_noisy"],
        qmap,
        q_window,
        mask=frame["mask"],
        options={
            "trace_method": "annular_peak",
            "stage": "evaluate",
            "resamples": 0,
            "sensitivity": False,
            "annular_radial_bins": 40,
            "annular_angle_bins": 72,
        },
        reference_axis_deg=0.0,
        multistart=multistart,
    )
    points = result.get("points", [])
    accepted = [point for point in points if point.get("accepted")]
    diagnostics = result.get("diagnostics", {})
    return {
        "execution_status": "complete",
        "elapsed_s": float(time.perf_counter() - start),
        "method": "application_gui_annular_peak_trajectory",
        "method_version": result.get("method_version"),
        "measurement_status": result.get("measurement_status"),
        "candidate_fit": {
            key: _finite(result.get("candidate_fit", {}).get(key))
            for key in ("a", "b", "axis_ratio", "theta_deg", "rmse", "condition", "q_star_from_arcs")
        },
        "quality_status": result.get("quality", {}).get("status"),
        "quality_flags": result.get("quality", {}).get("flags", []),
        "diagnostics": _jsonable(
            {
                key: diagnostics.get(key)
                for key in (
                    "method",
                    "q_window",
                    "reference_axis_deg",
                    "n_annuli",
                    "n_points",
                    "n_accepted_points",
                    "n_arcs",
                    "n_raw_candidates",
                    "outer_window_truncated",
                    "outer_window_accepted_sides",
                    "first_order_q_hint",
                )
            }
        ),
        "points": _jsonable(
            [
                {
                    key: point.get(key)
                    for key in (
                        "point_id",
                        "annulus_index",
                        "trajectory_id",
                        "branch_id",
                        "side",
                        "q_annulus",
                        "chi_deg",
                        "qx",
                        "qy",
                        "accepted",
                        "reason",
                    )
                }
                for point in points
            ]
        ),
        "arcs": _jsonable(result.get("arcs", [])),
        "profiles": _jsonable(result.get("profiles", {})),
        "annular_peaks": _jsonable(result.get("annular_peaks", {})),
        "accepted_point_count": len(accepted),
        "point_count": len(points),
    }


def _safe_number(value: object) -> float | None:
    return _finite(value)


def _frame_record(
    frame: dict[str, object],
    q_window: tuple[float, float],
    radial_frames: set[int],
    annular_frames: set[int],
    multistart: int,
) -> dict[str, object]:
    model_id = str(frame.get("model_id", "unknown_model"))
    frame_id = str(frame.get("frame_id", "unknown_frame"))
    q0 = _finite(frame.get("structural_q0_nm_inv"))
    role = str(frame["sequence_role"])
    excluded = np.asarray(frame["mask"], dtype=bool)
    record: dict[str, object] = {
        "model_id": str(frame.get("model_id", "t2")),
        "frame_index": int(frame["frame_index"]),
        "frame_id": str(frame["frame_id"]),
        "sequence_role": role,
        "seed": int(frame.get("seed", 0)),
        "q_unit": frame.get("q_unit", "nm^-1"),
        "structural_q0_nm_inv": q0,
        "structural_layer_spacing_nm": (
            _safe_number(frame["structure_truth"].get("layer_spacing_nm"))
            if isinstance(frame.get("structure_truth"), dict)
            else None
        ),
        "orientation_spread_scale": (
            _safe_number(frame["structure_truth"].get("orientation_spread_scale"))
            if isinstance(frame.get("structure_truth"), dict)
            else None
        ),
        "nominal_orientation_deg": (
            _safe_number(frame["structure_truth"].get("nominal_orientation_deg"))
            if isinstance(frame.get("structure_truth"), dict)
            else None
        ),
        "mask_diagnostics": frame.get("mask_diagnostics"),
        "masked_fraction": float(np.mean(excluded)),
        "noise_sigma": _safe_number(frame.get("noise_sigma")),
        "noise_target_first_order_snr": _safe_number(frame.get("noise_target_first_order_snr")),
        "noise_reference": frame.get("noise_reference"),
        "structure_truth": frame.get("structure_truth"),
    }
    if q0 is not None:
        record["noise_free_annular_reference"] = measure_annular_local_peak(
            np.asarray(frame["intensity_noiseless"], dtype=float),
            np.asarray(frame["q"], dtype=float),
            excluded,
            q0,
        )
    else:
        record["noise_free_annular_reference"] = None
    print(f"[validation] {model_id}/{frame_id}: curvature start", file=sys.stderr, flush=True)
    try:
        record["curvature_pipeline"] = _analyze_curvature(frame, q_window, multistart)
    except Exception as exc:  # Keep the frame and failure reason in sequence outputs.
        record["curvature_pipeline"] = {
            "execution_status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    curvature_status = record["curvature_pipeline"].get("measurement_status")
    curvature_seconds = _finite(record["curvature_pipeline"].get("elapsed_s"))
    curvature_elapsed = f"{curvature_seconds:.1f}s" if curvature_seconds is not None else "n/a"
    print(
        f"[validation] {model_id}/{frame_id}: curvature done in {curvature_elapsed} "
        f"({curvature_status or record['curvature_pipeline'].get('execution_status')})",
        file=sys.stderr,
        flush=True,
    )
    if int(frame["frame_index"]) in radial_frames:
        print(f"[validation] {model_id}/{frame_id}: radial start", file=sys.stderr, flush=True)
        try:
            record["radial_peak_pipeline"] = _analyze_radial_trace(frame, q_window, multistart)
        except Exception as exc:
            record["radial_peak_pipeline"] = {
                "execution_status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        print(f"[validation] {model_id}/{frame_id}: radial done", file=sys.stderr, flush=True)
    else:
        record["radial_peak_pipeline"] = None
    if int(frame["frame_index"]) in annular_frames:
        print(f"[validation] {model_id}/{frame_id}: annular start", file=sys.stderr, flush=True)
        try:
            record["annular_trajectory"] = _analyze_annular_trajectory(
                frame, q_window, multistart
            )
        except Exception as exc:
            record["annular_trajectory"] = {
                "execution_status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        annular_status = record["annular_trajectory"].get("measurement_status")
        annular_seconds = _finite(record["annular_trajectory"].get("elapsed_s"))
        annular_elapsed = f"{annular_seconds:.1f}s" if annular_seconds is not None else "n/a"
        print(
            f"[validation] {model_id}/{frame_id}: annular done in {annular_elapsed} "
            f"({annular_status or record['annular_trajectory'].get('execution_status')})",
            file=sys.stderr,
            flush=True,
        )
    else:
        record["annular_trajectory"] = None

    curvature = record.get("curvature_pipeline")
    if isinstance(curvature, dict):
        measured_q = _finite(curvature.get("accepted_ridge_q_median_nm_inv"))
        record["curvature_ridge_q_median_minus_structural_q0_nm_inv"] = (
            None if measured_q is None or q0 is None else float(measured_q - q0)
        )
        reference = record.get("noise_free_annular_reference")
        reference_q = _finite(reference.get("q_peak_nm_inv")) if isinstance(reference, dict) else None
        record["curvature_ridge_q_median_minus_annular_reference_nm_inv"] = (
            None if measured_q is None or reference_q is None else float(measured_q - reference_q)
        )
    return record


def _write_csv(path: Path, records: list[dict[str, object]]) -> None:
    columns = (
        "model_id",
        "frame_index",
        "frame_id",
        "sequence_role",
        "structural_layer_spacing_nm",
        "structural_q0_nm_inv",
        "noise_sigma",
        "noise_target_first_order_snr",
        "noise_free_first_order_p90_snr",
        "noise_free_annular_q_peak_nm_inv",
        "noise_free_annular_spacing_nm",
        "curvature_median_ridge_q_nm_inv",
        "curvature_q_star_from_arcs_nm_inv",
        "curvature_q_star_source",
        "curvature_measurement_status",
        "curvature_quality_status",
        "curvature_ridge_point_count",
        "curvature_accepted_ridge_point_count",
        "curvature_axis_ratio_candidate",
        "radial_peak_median_ridge_q_nm_inv",
        "annular_point_count",
        "annular_accepted_point_count",
        "annular_trajectory_count",
        "annular_measurement_status",
        "masked_fraction",
    )
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for record in records:
            reference = record.get("noise_free_annular_reference")
            curvature = record.get("curvature_pipeline")
            radial = record.get("radial_peak_pipeline")
            annular = record.get("annular_trajectory")
            fit = curvature.get("candidate_fit", {}) if isinstance(curvature, dict) else {}
            writer.writerow(
                {
                    "model_id": record["model_id"],
                    "frame_index": record["frame_index"],
                    "frame_id": record["frame_id"],
                    "sequence_role": record["sequence_role"],
                    "structural_layer_spacing_nm": record["structural_layer_spacing_nm"],
                    "structural_q0_nm_inv": record["structural_q0_nm_inv"],
                    "noise_sigma": record["noise_sigma"],
                    "noise_target_first_order_snr": record["noise_target_first_order_snr"],
                    "noise_free_first_order_p90_snr": (
                        record.get("noise_reference", {}).get("p90_snr")
                        if isinstance(record.get("noise_reference"), dict)
                        else None
                    ),
                    "noise_free_annular_q_peak_nm_inv": (
                        reference.get("q_peak_nm_inv") if isinstance(reference, dict) else None
                    ),
                    "noise_free_annular_spacing_nm": (
                        reference.get("spacing_nm") if isinstance(reference, dict) else None
                    ),
                    "curvature_median_ridge_q_nm_inv": (
                        curvature.get("accepted_ridge_q_median_nm_inv")
                        if isinstance(curvature, dict)
                        else None
                    ),
                    "curvature_q_star_from_arcs_nm_inv": fit.get("q_star_from_arcs"),
                    "curvature_q_star_source": (
                        curvature.get("q_star_source") if isinstance(curvature, dict) else None
                    ),
                    "curvature_measurement_status": (
                        curvature.get("measurement_status") if isinstance(curvature, dict) else None
                    ),
                    "curvature_quality_status": (
                        curvature.get("quality_status") if isinstance(curvature, dict) else None
                    ),
                    "curvature_ridge_point_count": (
                        curvature.get("ridge_point_count") if isinstance(curvature, dict) else None
                    ),
                    "curvature_accepted_ridge_point_count": (
                        curvature.get("accepted_ridge_point_count")
                        if isinstance(curvature, dict)
                        else None
                    ),
                    "curvature_axis_ratio_candidate": fit.get("axis_ratio"),
                    "radial_peak_median_ridge_q_nm_inv": (
                        radial.get("accepted_ridge_q_median_nm_inv")
                        if isinstance(radial, dict)
                        else None
                    ),
                    "annular_point_count": (
                        annular.get("point_count") if isinstance(annular, dict) else None
                    ),
                    "annular_accepted_point_count": (
                        annular.get("accepted_point_count") if isinstance(annular, dict) else None
                    ),
                    "annular_trajectory_count": (
                        annular.get("diagnostics", {}).get("n_arcs")
                        if isinstance(annular, dict)
                        else None
                    ),
                    "annular_measurement_status": (
                        annular.get("measurement_status") if isinstance(annular, dict) else None
                    ),
                    "masked_fraction": record["masked_fraction"],
                }
            )


def _selected_annular_records(records: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        record
        for record in records
        if isinstance(record.get("annular_trajectory"), dict)
        and record["annular_trajectory"].get("execution_status") == "complete"
    ]


def _plot_annular_overlays(
    trajectory_path: Path,
    profile_path: Path,
    frames: tuple[dict[str, object], ...],
    records: list[dict[str, object]],
) -> None:
    annular_records = _selected_annular_records(records)
    if not annular_records:
        return
    by_index = {int(frame["frame_index"]): frame for frame in frames}
    n_panels = len(annular_records)
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad(color="0.72")

    fig, axes = plt.subplots(1, n_panels, figsize=(5.0 * n_panels, 4.7), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, record in zip(axes, annular_records):
        frame = by_index[int(record["frame_index"])]
        intensity = np.ma.array(
            np.maximum(np.asarray(frame["intensity_noisy"], dtype=float), 1e-6),
            mask=np.asarray(frame["mask"], dtype=bool),
        )
        ax.imshow(
            intensity,
            origin="lower",
            extent=(
                float(np.min(frame["qx"])),
                float(np.max(frame["qx"])),
                float(np.min(frame["qy"])),
                float(np.max(frame["qy"])),
            ),
            norm=LogNorm(vmin=1e-5, vmax=1.0),
            cmap=cmap,
            interpolation="nearest",
        )
        trace = record["annular_trajectory"]
        points = trace.get("points", [])
        grouped: dict[str, list[dict[str, object]]] = {}
        for point in points:
            if point.get("qx") is None or point.get("qy") is None:
                continue
            trajectory_id = str(point.get("trajectory_id"))
            grouped.setdefault(trajectory_id, []).append(point)
        for trajectory_id, group in grouped.items():
            qx = [float(point["qx"]) for point in group]
            qy = [float(point["qy"]) for point in group]
            is_accepted = all(bool(point.get("accepted")) for point in group)
            connected = is_accepted and trajectory_id != "None"
            ax.plot(
                qx,
                qy,
                marker="o",
                markersize=2.6,
                linewidth=0.8,
                linestyle="-" if connected else "None",
                alpha=0.9,
                label=trajectory_id if is_accepted else None,
                fillstyle="full" if is_accepted else "none",
            )
        q0 = _finite(frame.get("structural_q0_nm_inv"))
        if q0 is not None:
            theta = np.linspace(0.0, 2.0 * np.pi, 361)
            ax.plot(q0 * np.cos(theta), q0 * np.sin(theta), "w--", linewidth=0.7,
                    alpha=0.85, label="structural q₀")
        ax.set_aspect("equal")
        ax.set_xlim(-0.85, 0.85)
        ax.set_ylim(-0.85, 0.85)
        ax.set_xlabel("qₓ (nm⁻¹)")
        ax.set_ylabel("qᵧ (nm⁻¹)")
        ax.set_title(
            f"{frame['frame_id']} | {trace.get('measurement_status')}\n"
            f"{trace.get('accepted_point_count')}/{trace.get('point_count')} accepted annular points",
            fontsize=8,
        )
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=6, loc="best")
    fig.suptitle("Application annular trajectory overlay; grey = excluded pixels")
    fig.savefig(trajectory_path, dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, n_panels, figsize=(5.0 * n_panels, 4.7), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, record in zip(axes, annular_records):
        frame = by_index[int(record["frame_index"])]
        trace = record["annular_trajectory"]
        profiles = trace.get("profiles", {})
        profile_items = list(profiles.items()) if isinstance(profiles, dict) else []
        q0 = _finite(frame.get("structural_q0_nm_inv"))
        if q0 is None:
            target_q = float(np.median([profile.get("q_center", 0.0) for _, profile in profile_items])) if profile_items else 0.0
        else:
            target_q = q0
        selected = sorted(
            (
                (key, profile)
                for key, profile in profile_items
                if isinstance(profile, dict) and profile.get("profile_axis") == "azimuthal"
            ),
            key=lambda item: abs(float(item[1].get("q_center", 0.0)) - target_q),
        )[:4]
        for key, profile in selected:
            angle = np.asarray(profile.get("angle_deg", []), dtype=float)
            values = np.asarray(profile.get("smoothed_intensity", []), dtype=float)
            if angle.size and angle.shape == values.shape:
                ax.plot(angle, values, linewidth=0.9,
                        label=f"q={float(profile.get('q_center', np.nan)):.3f} nm⁻¹")
                peak_angles = np.asarray(profile.get("peak_angles_deg", []), dtype=float)
                for peak_angle in peak_angles:
                    ax.axvline(float(peak_angle), color="0.5", linewidth=0.45, alpha=0.35)
        ax.set_xlabel("azimuth χ (degree)")
        ax.set_ylabel("smoothed intensity")
        ax.set_title(f"{frame['frame_id']} annular azimuth profiles\nselected nearest q₀", fontsize=8)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=6, loc="best")
        ax.grid(True, alpha=0.2)
    fig.suptitle("Observed angular intensity profiles from prescribed q annuli")
    fig.savefig(profile_path, dpi=180)
    plt.close(fig)


def _plot_intensity_sheet(path: Path, frames: tuple[dict[str, object], ...]) -> None:
    cols = 4
    rows = int(np.ceil(len(frames) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(14, 3.65 * rows), constrained_layout=True)
    axes_array = np.asarray(axes).reshape(rows, cols)
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad(color="0.72")
    image = None
    for index, frame in enumerate(frames):
        ax = axes_array.flat[index]
        intensity = np.asarray(frame["intensity_noisy"], dtype=float)
        valid = ~np.asarray(frame["mask"], dtype=bool)
        displayed = np.ma.array(np.maximum(intensity, 1e-6), mask=~valid)
        image = ax.imshow(
            displayed,
            origin="lower",
            extent=(
                float(np.min(frame["qx"])),
                float(np.max(frame["qx"])),
                float(np.min(frame["qy"])),
                float(np.max(frame["qy"])),
            ),
            norm=LogNorm(vmin=1e-5, vmax=1.0),
            cmap=cmap,
            interpolation="nearest",
        )
        q0 = frame.get("structural_q0_nm_inv")
        suffix = f" | q₀={float(q0):.3f} nm⁻¹" if q0 is not None else " | noise only"
        ax.set_title(f"{frame['frame_id']}{suffix}", fontsize=9)
        ax.set_xlabel("qₓ (nm⁻¹)")
        ax.set_ylabel("qᵧ (nm⁻¹)")
        ax.set_aspect("equal")
        ax.set_xlim(-0.85, 0.85)
        ax.set_ylim(-0.85, 0.85)
    for ax in axes_array.flat[len(frames) :]:
        ax.remove()
    if image is not None:
        fig.colorbar(image, ax=[ax for ax in axes_array.flat if ax in fig.axes], shrink=0.8,
                     label="observed intensity (log scale)")
    fig.suptitle("Finite real-space layer-stack FFT sequence; grey = excluded pixels")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_truth_observed(path: Path, records: list[dict[str, object]]) -> None:
    indices = [int(record["frame_index"]) for record in records]
    structural_q = [_finite(record.get("structural_q0_nm_inv")) for record in records]
    annular_q = []
    curvature_q = []
    radial_q = []
    for record in records:
        reference = record.get("noise_free_annular_reference")
        curvature = record.get("curvature_pipeline")
        radial = record.get("radial_peak_pipeline")
        annular_q.append(
            _finite(reference.get("q_peak_nm_inv")) if isinstance(reference, dict) else None
        )
        curvature_q.append(
            _finite(curvature.get("accepted_ridge_q_median_nm_inv"))
            if isinstance(curvature, dict)
            else None
        )
        radial_q.append(
            _finite(radial.get("accepted_ridge_q_median_nm_inv"))
            if isinstance(radial, dict)
            else None
        )

    def plot_series(ax, values, label, marker, linestyle="none"):
        x_values = [idx for idx, value in zip(indices, values) if value is not None]
        y_values = [value for value in values if value is not None]
        if x_values:
            ax.plot(x_values, y_values, marker=marker, linestyle=linestyle, label=label)

    fig, ax = plt.subplots(figsize=(10.5, 5.2), constrained_layout=True)
    plot_series(ax, structural_q, "structure: 2π / spacing", "o", "-")
    plot_series(ax, annular_q, "noise-free observed annular peak", "s")
    plot_series(ax, curvature_q, "pipeline curvature: median accepted ridge radius", "^")
    plot_series(ax, radial_q, "pipeline radial_peak: median traced radius", "x")
    noise_indices = [
        int(record["frame_index"])
        for record in records
        if record["sequence_role"] == "noise_only_control"
    ]
    for noise_index in noise_indices:
        ax.axvspan(noise_index - 0.35, noise_index + 0.35, color="0.8", alpha=0.45)
        ax.text(noise_index, 0.98, "noise only", transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=8, rotation=90)
    missing_indices = [
        int(record["frame_index"])
        for record in records
        if isinstance(record.get("mask_diagnostics"), dict)
        and int(record["mask_diagnostics"].get("missing_lobe_pixels", 0)) > 0
    ]
    for missing_index in missing_indices:
        ax.axvspan(missing_index - 0.35, missing_index + 0.35, color="#e6a05a", alpha=0.18)
    ax.set_xlabel("sequence frame")
    ax.set_ylabel("q (nm⁻¹)")
    ax.set_title("Structural spacing reference and independently extracted q positions")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO / "results" / "validation" / "upgrade_sequence",
    )
    parser.add_argument("--frames", type=int, default=10, help="number of physical signal frames")
    parser.add_argument(
        "--oblique-frames", type=int, default=8,
        help="number of finite oblique-stack signal frames",
    )
    parser.add_argument("--size", type=int, default=512, help="square image side in pixels")
    parser.add_argument(
        "--resolution-control-size",
        type=int,
        default=128,
        help="additional low-resolution curvature control side in pixels",
    )
    parser.add_argument(
        "--no-resolution-control",
        action="store_true",
        help="skip the additional low-resolution curvature control",
    )
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument(
        "--multistart", type=int, default=1,
        help="ellipse optimizer starts per frame; standard benchmark uses one start with max_nfev=800",
    )
    parser.add_argument("--force", action="store_true", help="replace this script's known output files")
    parser.add_argument(
        "--radial-frames",
        default="",
        help="comma-separated signal frame indices also traced with pipeline radial_peak",
    )
    parser.add_argument(
        "--annular-frames",
        default=None,
        help="comma-separated frame indices for the GUI annular trajectory path; default: first, masked, noise control",
    )
    parser.add_argument(
        "--oblique-annular-frames",
        default=None,
        help="comma-separated oblique frame indices for the GUI annular path; default: first, masked, clean and noise controls",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _make_parser().parse_args(argv)
    if args.frames < 2 or args.oblique_frames < 2 or args.size < 32:
        raise ValueError("--frames and --oblique-frames must be >= 2 and --size must be >= 32")
    if args.multistart < 1:
        raise ValueError("--multistart must be at least 1")
    if not args.no_resolution_control and args.resolution_control_size < 32:
        raise ValueError("--resolution-control-size must be >= 32")
    output = args.output.resolve()
    try:
        output.relative_to(REPO / "data_local")
    except ValueError:
        pass
    else:
        raise ValueError("validation outputs cannot be written under data_local")
    output.mkdir(parents=True, exist_ok=True)
    existing = [output / filename for filename in OUTPUT_FILES if (output / filename).exists()]
    if existing and not args.force:
        raise FileExistsError(
            "output files already exist; use --force to replace only this script's known files"
        )

    radial_frames = {int(value) for value in args.radial_frames.split(",") if value.strip()}
    if any(index < 0 or index >= args.frames for index in radial_frames):
        raise ValueError("--radial-frames indices must identify physical signal frames")
    settings = SequenceSettings(
        n_signal_frames=args.frames,
        shape=(args.size, args.size),
        seed=args.seed,
        local_missing_lobe_frame=min(5, args.frames - 1),
    )
    frames = generate_sequence(settings)
    for frame in frames:
        frame["model_id"] = "t2_finite_layer_stack"
    if args.annular_frames is None:
        annular_frames = {0, settings.local_missing_lobe_frame}
        if settings.noise_control:
            noise_index = settings.n_signal_frames + int(settings.buried_signal_stress)
            annular_frames.add(noise_index)
    else:
        annular_frames = {int(value) for value in args.annular_frames.split(",") if value.strip()}
        if any(index < 0 or index >= len(frames) for index in annular_frames):
            raise ValueError("--annular-frames indices must identify generated signal/control frames")
    q_window = (0.15, 0.85)
    records = [
        _frame_record(frame, q_window, radial_frames, annular_frames, args.multistart)
        for frame in frames
    ]
    oblique_settings = ObliqueStackSettings(
        n_signal_frames=args.oblique_frames,
        shape=(args.size, args.size),
        seed=args.seed + 1000,
        local_missing_lobe_frame=min(4, args.oblique_frames - 1),
    )
    oblique_frames = generate_oblique_stack_sequence(oblique_settings)
    for frame in oblique_frames:
        frame["model_id"] = "oblique_lamella_stack"
    if args.oblique_annular_frames is None:
        oblique_annular_frames = {
            0,
            oblique_settings.local_missing_lobe_frame,
            oblique_settings.n_signal_frames,
            oblique_settings.n_signal_frames + int(oblique_settings.include_clean_control),
        }
    else:
        oblique_annular_frames = {
            int(value) for value in args.oblique_annular_frames.split(",") if value.strip()
        }
        if any(index < 0 or index >= len(oblique_frames) for index in oblique_annular_frames):
            raise ValueError("--oblique-annular-frames must identify generated frames/controls")
    oblique_records = [
        _frame_record(frame, q_window, set(), oblique_annular_frames, args.multistart)
        for frame in oblique_frames
    ]
    resolution_control = None
    if not args.no_resolution_control:
        low_settings = SequenceSettings(
            n_signal_frames=2,
            shape=(args.resolution_control_size, args.resolution_control_size),
            pixel_size_nm=settings.pixel_size_nm,
            spacing_start_nm=settings.spacing_start_nm,
            spacing_end_nm=settings.spacing_start_nm,
            orientation_spread_start=settings.orientation_spread_start,
            orientation_spread_end=settings.orientation_spread_start,
            orientation_center_start_deg=settings.orientation_center_start_deg,
            orientation_center_end_deg=settings.orientation_center_start_deg,
            noise_sigma=settings.noise_sigma,
            target_first_order_snr=settings.target_first_order_snr,
            spacing_jitter_fraction=settings.spacing_jitter_fraction,
            seed=settings.seed,
            local_missing_lobe_frame=1,
            local_missing_lobe_center_deg=settings.local_missing_lobe_center_deg,
            local_missing_lobe_half_width_deg=settings.local_missing_lobe_half_width_deg,
            local_missing_lobe_q_low_fraction=settings.local_missing_lobe_q_low_fraction,
            local_missing_lobe_q_high_fraction=settings.local_missing_lobe_q_high_fraction,
            noise_control=False,
            buried_signal_stress=False,
            beamstop_radius_fraction=settings.beamstop_radius_fraction,
        )
        low_frame = generate_sequence(low_settings)[0]
        low_frame["frame_id"] = f"resolution_{args.resolution_control_size}_signal_00"
        low_frame["sequence_role"] = "resolution_control"
        low_record = _frame_record(low_frame, q_window, set(), set(), args.multistart)
        resolution_control = {
            "shape": [args.resolution_control_size, args.resolution_control_size],
            "paired_with_main_frame": "signal_00",
            "purpose": "regenerate the same physical case on a smaller real-space grid at fixed pixel size, changing field of view, reciprocal-space pixel spacing, and finite-window truncation",
            "frame": low_record,
        }

    script_hashes = {
        "scripts/validate_lamellar_sequence.py": _sha256(Path(__file__).resolve()),
        "src/butterfly_saxs/benchmark_sequence.py": _sha256(
            REPO / "src" / "butterfly_saxs" / "benchmark_sequence.py"
        ),
        "src/butterfly_saxs/benchmark_t2.py": _sha256(
            REPO / "src" / "butterfly_saxs" / "benchmark_t2.py"
        ),
        "src/butterfly_saxs/pipeline.py": _sha256(
            REPO / "src" / "butterfly_saxs" / "pipeline.py"
        ),
        "src/butterfly_saxs/butterfly_ridge.py": _sha256(
            REPO / "src" / "butterfly_saxs" / "butterfly_ridge.py"
        ),
        "src/butterfly_saxs/butterfly.py": _sha256(
            REPO / "src" / "butterfly_saxs" / "butterfly.py"
        ),
        "src/butterfly_saxs/observables.py": _sha256(
            REPO / "src" / "butterfly_saxs" / "observables.py"
        ),
        "src/butterfly_saxs/analysis_config.py": _sha256(
            REPO / "src" / "butterfly_saxs" / "analysis_config.py"
        ),
        "src/butterfly_saxs/annular_trace.py": _sha256(
            REPO / "src" / "butterfly_saxs" / "annular_trace.py"
        ),
    }
    curvature_recipe = pipeline_analysis_config(q_window=q_window)
    curvature_recipe["analysis"]["ellipse"]["multistart"] = args.multistart
    summary = {
        "schema_version": "lamellarsaxs2d.validation.lamellar_sequence.v1",
        "model_scope": [
            "t2_2d_finite_layer_stack_density_fft",
            "finite_oblique_lamella_stack_incoherent_fft_power_sum",
        ],
        "generator_version": benchmark_t2.GENERATOR_VERSION,
        "generator_hash": benchmark_t2.GENERATOR_HASH,
        "source_hashes": script_hashes,
        "settings": {"t2": asdict(settings), "oblique_stack": asdict(oblique_settings)},
        "runtime": {
            "python": sys.version,
            "packages": {
                name: _package_version(name)
                for name in ("butterfly-saxs", "numpy", "scipy", "matplotlib")
            },
        },
        "q_unit": "nm^-1",
        "q_window_nm_inv": list(q_window),
        "analysis_recipes": {
            "curvature": curvature_recipe,
            "ellipse_multistart": args.multistart,
            "optimizer_max_nfev": 800,
            "radial_peak_pipeline_frames": sorted(radial_frames),
            "annular_peak_trajectory_frames": sorted(annular_frames),
            "oblique_annular_peak_trajectory_frames": sorted(oblique_annular_frames),
        },
        "references": [
            {
                "citation": "Grubb, Murthy & Francescangeli (2016)",
                "doi": "10.1002/polb.23930",
                "url": "https://doi.org/10.1002/polb.23930",
                "use": "context for lamellar stack intensity modeling; this script is not a reproduction of the full 3D model",
            },
            {
                "citation": "Grubb et al. (2021)",
                "doi": "10.1016/j.polymer.2021.123566",
                "url": "https://doi.org/10.1016/j.polymer.2021.123566",
                "use": "context for 2D scattering trajectory analysis and curvature-locus methods",
            },
            {
                "citation": "Murthy & Grubb (2024)",
                "doi": "10.1107/S1600576724004503",
                "url": "https://journals.iucr.org/j/issues/2024/04/00/tu5052/",
                "use": "spacing moves reflections radially; tilt changes azimuth; stack rotation relative to tilt distinguishes butterfly and eyebrow patterns; real-space stacks are Fourier transformed",
            },
        ],
        "interpretation": {
            "structural_q0": "known generator value 2π/layer_spacing_nm",
            "observable_reference": "noise-free FFT intensity annular mean local peak in a known q0 neighborhood; independent of pipeline ellipse/ridge fitting",
            "projection_truth_used_as_measurement_target": False,
            "masked_pixel_policy": "mask is exclusion only; source intensity is preserved and masked pixels are not recreated",
            "noise_control_policy": "no structural q0 is assigned; any fit is retained and reported as a noise false-positive candidate diagnostic",
            "noise_model": "additive Gaussian noise after clean-intensity normalization, then clip observed intensity to non-negative values",
            "optimizer": {
                "ellipse_multistart": args.multistart,
                "max_nfev": 800,
                "same_observations_and_candidate_rules": True,
            },
            "confidence": "engineering and synthetic-model evidence only; not experimental calibration or scientific acceptance",
        },
        "frame_count": len(records) + len(oblique_records),
        "signal_frame_count": settings.n_signal_frames,
        "noise_control_count": int(settings.noise_control),
        "resolution_control": resolution_control,
        "sequences": {
            "t2_finite_layer_stack": {
                "model_scope": "finite_2d_density_sum_then_fft_intensity",
                "frame_count": len(records),
                "settings": asdict(settings),
                "frames": records,
            },
            "oblique_lamella_stack": {
                "model_scope": "mirrored_tilt_populations_with_opposite_stack_rotation; incoherent_fft_power_sum",
                "frame_count": len(oblique_records),
                "settings": asdict(oblique_settings),
                "frames": oblique_records,
            },
        },
        "frames": records,
        "oblique_frames": oblique_records,
    }
    summary = _jsonable(summary)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    _write_csv(output / "sequence.csv", [*records, *oblique_records])
    _plot_intensity_sheet(output / "t2_intensity_contact_sheet.png", frames)
    _plot_truth_observed(output / "t2_truth_vs_observed.png", records)
    _plot_intensity_sheet(output / "oblique_intensity_contact_sheet.png", oblique_frames)
    _plot_truth_observed(output / "oblique_truth_vs_observed.png", oblique_records)
    _plot_annular_overlays(
        output / "annular_trajectory_overlay.png",
        output / "annular_profile_overlay.png",
        oblique_frames,
        oblique_records,
    )
    print(
        f"Wrote {len(records) + len(oblique_records)} frames "
        f"({len(records)} T2; {len(oblique_records)} oblique) and summaries to {output}"
    )
    for record in [*records, *oblique_records]:
        curvature = record.get("curvature_pipeline")
        q_star = None
        if isinstance(curvature, dict):
            fit = curvature.get("candidate_fit")
            if isinstance(fit, dict):
                q_star = fit.get("q_star_from_arcs")
        reference = record.get("noise_free_annular_reference")
        annular_q = reference.get("q_peak_nm_inv") if isinstance(reference, dict) else None
        status = curvature.get("measurement_status") if isinstance(curvature, dict) else None
        annular = record.get("annular_trajectory")
        annular_status = annular.get("measurement_status") if isinstance(annular, dict) else None
        print(
            f"{record['model_id']}/{record['frame_id']}: role={record['sequence_role']}, "
            f"curvature={status}, annular={annular_status}, "
            f"annular_q={annular_q}, curvature_q*={q_star}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
