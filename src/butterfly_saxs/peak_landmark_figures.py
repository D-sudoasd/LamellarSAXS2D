"""Standalone publication assets for measured SAXS peak landmarks."""

from __future__ import annotations

from collections.abc import Mapping
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .cancellation import raise_if_cancelled
from .serialization import json_safe


PEAK_FIGURE_METHOD_VERSION = "peak-landmark-figures-v1"
_PEAK_COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")


def _source_array(data: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in data and data[name] is not None:
            return data[name]
    raise ValueError(f"data must provide one of {', '.join(names)}")


def _source_payload(data: Mapping[str, Any], landmarks: Mapping[str, Any]) -> dict[str, Any]:
    observed = _source_array(data, "observed", "observed_raw", "observed_float")
    qx = _source_array(data, "qx", "qx_raw", "qx_float")
    qy = _source_array(data, "qy", "qy_raw", "qy_float")
    valid = data.get("effective_valid", data.get("valid_mask"))
    q_unit = str(data.get("q_unit", landmarks.get("q_unit", "unknown")) or "unknown")
    context = data.get("context")
    return {
        "observed": observed,
        "qx": qx,
        "qy": qy,
        "valid_mask": valid,
        "q_unit": q_unit,
        "context": context,
    }


def _model_shape(model: Any) -> tuple[int, ...] | None:
    if model is None:
        return None
    value = model
    if isinstance(model, Mapping):
        for key in ("model", "full2d", "predicted_intensity", "intensity", "prediction"):
            if key in model:
                value = model[key]
                break
        else:
            raise ValueError("model mapping must contain a supported intensity array")
    array = np.asanyarray(value)
    if np.asarray(np.ma.getdata(array)).dtype.kind not in "iuf":
        raise TypeError("model must contain real numeric values")
    return tuple(array.shape)


def _profile(landmarks: Mapping[str, Any], family: str) -> dict[str, np.ndarray]:
    profiles = landmarks.get("profiles", {})
    value = profiles.get(family, {}) if isinstance(profiles, Mapping) else {}
    if not isinstance(value, Mapping):
        return {}
    output: dict[str, np.ndarray] = {}
    for key, raw in value.items():
        try:
            if key == "supported":
                output[str(key)] = np.asarray(raw, dtype=bool)
            elif "count" in str(key):
                output[str(key)] = np.asarray(raw, dtype=np.int64)
            else:
                output[str(key)] = np.asarray(
                    [np.nan if item is None else float(item) for item in raw],
                    dtype=np.float64,
                )
        except (TypeError, ValueError, OverflowError):
            continue
    return output


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(
            json_safe(payload),
            stream,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        stream.write("\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_landmark_csv(path: Path, landmarks: Mapping[str, Any]) -> None:
    fields = (
        "record_type",
        "peak_id",
        "pixel_x",
        "pixel_y",
        "qx",
        "qy",
        "q",
        "raw_intensity",
        "smoothed_intensity",
        "angular_peak_deg",
        "chi_deg",
        "angular_prominence",
        "angular_noise_sigma",
        "angular_snr",
        "angular_fwhm_deg",
        "angular_coverage_fraction",
        "support_pixel_count",
        "model_pixel_x",
        "model_pixel_y",
        "model_qx",
        "model_qy",
        "model_q",
        "model_raw_intensity",
        "delta_qx",
        "delta_qy",
        "delta_q",
        "flags",
        "interpretation",
    )
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        raw_max = landmarks.get("raw_global_max")
        if isinstance(raw_max, Mapping):
            writer.writerow(
                {
                    "record_type": "raw_global_maximum_only",
                    "peak_id": raw_max.get("label", "G"),
                    "pixel_x": raw_max.get("pixel_x"),
                    "pixel_y": raw_max.get("pixel_y"),
                    "qx": raw_max.get("qx"),
                    "qy": raw_max.get("qy"),
                    "q": raw_max.get("q"),
                    "raw_intensity": raw_max.get("raw_intensity"),
                    "flags": ";".join(raw_max.get("flags", ())),
                    "interpretation": raw_max.get("interpretation"),
                }
            )
        for peak in landmarks.get("peaks", ()):
            if not isinstance(peak, Mapping):
                continue
            model_peak = peak.get("model_peak")
            if not isinstance(model_peak, Mapping):
                model_peak = {}
            writer.writerow(
                {
                    "record_type": "supported_angular_lobe_peak",
                    "peak_id": peak.get("peak_id"),
                    "pixel_x": peak.get("pixel_x"),
                    "pixel_y": peak.get("pixel_y"),
                    "qx": peak.get("qx"),
                    "qy": peak.get("qy"),
                    "q": peak.get("q"),
                    "raw_intensity": peak.get("raw_intensity"),
                    "smoothed_intensity": peak.get("smoothed_intensity"),
                    "angular_peak_deg": peak.get("angular_peak_deg"),
                    "chi_deg": peak.get("chi_deg"),
                    "angular_prominence": peak.get("angular_prominence"),
                    "angular_noise_sigma": peak.get("angular_noise_sigma"),
                    "angular_snr": peak.get("angular_snr"),
                    "angular_fwhm_deg": peak.get("angular_fwhm_deg"),
                    "angular_coverage_fraction": peak.get("angular_coverage_fraction"),
                    "support_pixel_count": peak.get("support_pixel_count"),
                    "model_pixel_x": model_peak.get("pixel_x"),
                    "model_pixel_y": model_peak.get("pixel_y"),
                    "model_qx": model_peak.get("qx"),
                    "model_qy": model_peak.get("qy"),
                    "model_q": model_peak.get("q"),
                    "model_raw_intensity": model_peak.get("raw_intensity"),
                    "delta_qx": peak.get("delta_qx"),
                    "delta_qy": peak.get("delta_qy"),
                    "delta_q": peak.get("delta_q"),
                    "flags": ";".join(peak.get("flags", ())),
                    "interpretation": "observed intensity landmark; not an identified reflection",
                }
            )


def _write_profile_csv(path: Path, landmarks: Mapping[str, Any]) -> None:
    angular = _profile(landmarks, "angular")
    radial = _profile(landmarks, "radial")
    fields = (
        "profile_type",
        "coordinate",
        "coordinate_unit",
        "intensity_raw_mean",
        "intensity_smoothed_mean",
        "intensity_isotropic_reference",
        "intensity_detection",
        "valid_pixel_count",
        "smoothed_pixel_count",
        "geometric_pixel_count",
        "coverage_fraction",
        "supported",
    )
    q_unit = str(landmarks.get("q_unit", "unknown"))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        angle = angular.get("angle_deg", np.asarray([], dtype=float))
        for index, coordinate in enumerate(angle):
            writer.writerow(
                {
                    "profile_type": "angular",
                    "coordinate": coordinate,
                    "coordinate_unit": "degree",
                    "intensity_raw_mean": angular.get("intensity_raw", [np.nan] * len(angle))[index],
                    "intensity_smoothed_mean": angular.get("intensity_smoothed", [np.nan] * len(angle))[index],
                    "intensity_isotropic_reference": angular.get(
                        "intensity_isotropic_reference", [""] * len(angle)
                    )[index],
                    "intensity_detection": angular.get(
                        "intensity_detection", [""] * len(angle)
                    )[index],
                    "valid_pixel_count": angular.get("valid_pixel_counts", [0] * len(angle))[index],
                    "smoothed_pixel_count": angular.get("smoothed_pixel_counts", [0] * len(angle))[index],
                    "geometric_pixel_count": angular.get("geometric_pixel_counts", [0] * len(angle))[index],
                    "coverage_fraction": angular.get("coverage_fraction", [0.0] * len(angle))[index],
                    "supported": angular.get("supported", [False] * len(angle))[index],
                }
            )
        q = radial.get("q", np.asarray([], dtype=float))
        for index, coordinate in enumerate(q):
            writer.writerow(
                {
                    "profile_type": "radial",
                    "coordinate": coordinate,
                    "coordinate_unit": q_unit,
                    "intensity_raw_mean": radial.get("mean_intensity_raw", [np.nan] * len(q))[index],
                    "intensity_smoothed_mean": radial.get("mean_intensity_smoothed", [np.nan] * len(q))[index],
                    "valid_pixel_count": radial.get("valid_pixel_counts", [0] * len(q))[index],
                    "smoothed_pixel_count": radial.get("smoothed_pixel_counts", [0] * len(q))[index],
                    "geometric_pixel_count": radial.get("geometric_pixel_counts", [0] * len(q))[index],
                    "coverage_fraction": radial.get("coverage_fraction", [0.0] * len(q))[index],
                    "supported": "",
                }
            )


def _write_profiles_npz(path: Path, landmarks: Mapping[str, Any]) -> None:
    angular = _profile(landmarks, "angular")
    radial = _profile(landmarks, "radial")
    payload: dict[str, Any] = {f"angular_{key}": value for key, value in angular.items()}
    payload.update({f"radial_{key}": value for key, value in radial.items()})
    payload["q_unit"] = np.asarray(str(landmarks.get("q_unit", "unknown")))
    payload["schema_version"] = np.asarray(str(landmarks.get("schema_version", "")))
    np.savez_compressed(path, **payload)


def _build_peak_map(
    *,
    base: Mapping[str, Any] | None,
    source: Mapping[str, Any],
    landmarks: Mapping[str, Any],
    width_mm: float,
    dpi: int,
) -> Any:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    from .figure_support import _draw_map

    height_mm = 110.0 if width_mm == 183.0 else 150.0
    fig = Figure(
        figsize=(width_mm / 25.4, height_mm / 25.4),
        dpi=dpi,
        facecolor="white",
        edgecolor="white",
    )
    FigureCanvasAgg(fig)
    if width_mm == 183.0:
        ax = fig.add_axes([0.07, 0.22, 0.79, 0.70])
        cax = fig.add_axes([0.88, 0.29, 0.018, 0.58])
    else:
        ax = fig.add_axes([0.15, 0.30, 0.68, 0.57])
        cax = fig.add_axes([0.89, 0.37, 0.025, 0.43])
    if base is not None:
        _draw_map(fig, ax, cax, base, draw_overlays=False, cmap="cividis")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, prune="both"))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, prune="both"))
    else:
        from .visualization import _diagnostic_q_axis_labels

        qx_label, qy_label = _diagnostic_q_axis_labels(source["q_unit"])
        ax.set_xlabel(qx_label)
        ax.set_ylabel(qy_label)
        ax.text(
            0.5,
            0.52,
            "No valid finite pixels in the supplied frame",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="#555555",
        )
        ax.set_aspect("equal", adjustable="box")
    peaks = [item for item in landmarks.get("peaks", ()) if isinstance(item, Mapping)]
    handles: list[Any] = []
    raw_max = landmarks.get("raw_global_max")
    if base is not None and isinstance(raw_max, Mapping):
        ax.scatter(
            [raw_max["qx"]],
            [raw_max["qy"]],
            marker="*",
            s=48,
            c="#D55E00",
            edgecolors="white",
            linewidths=0.55,
            zorder=7,
        )
        handles.append(
            Line2D(
                [0], [0], marker="*", linestyle="none", color="#D55E00",
                markeredgecolor="white", markersize=7,
                label="G: raw maximum (not a reflection assignment)",
            )
        )
        ax.annotate(
            str(raw_max.get("label", "G")),
            (float(raw_max["qx"]), float(raw_max["qy"])),
            xytext=(4, 4),
            textcoords="offset points",
            color="#D55E00",
            fontsize=6.0,
            fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.6},
            zorder=9,
        )
    model_marker_added = False
    has_lobe_peaks = bool(peaks)
    for index, peak in enumerate(peaks):
        color = _PEAK_COLORS[index % len(_PEAK_COLORS)]
        if base is not None:
            ax.scatter(
                [float(peak["qx"])],
                [float(peak["qy"])],
                marker="o",
                s=25,
                facecolors="none",
                edgecolors=color,
                linewidths=1.0,
                zorder=8,
            )
            ax.annotate(
                str(peak.get("peak_id", f"P{index + 1}")),
                (float(peak["qx"]), float(peak["qy"])),
                xytext=(4, 4),
                textcoords="offset points",
                color=color,
                fontsize=6.0,
                fontweight="bold",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 0.6},
                zorder=9,
            )
            model_peak = peak.get("model_peak")
            if isinstance(model_peak, Mapping):
                ax.scatter(
                    [float(model_peak["qx"])],
                    [float(model_peak["qy"])],
                    marker="x",
                    s=19,
                    c="#111111",
                    linewidths=0.8,
                    zorder=10,
                )
                model_marker_added = True
    if has_lobe_peaks:
        handles.append(
            Line2D(
                [0], [0], marker="o", linestyle="none", color="#0072B2",
                markerfacecolor="none", markersize=4.2,
                label="Numbered supported lobe peaks",
            )
        )
    if model_marker_added:
        handles.append(
            Line2D(
                [0], [0], marker="x", linestyle="none", color="#111111",
                markersize=4.2, label="Smoothed-model maximum in the same angular basin",
            )
        )
    signal_window = landmarks.get("domain", {}).get("effective_signal_q_window")
    if signal_window is None:
        signal_text = "signal band: all valid q (near-origin angular samples omitted)"
    else:
        signal_text = f"signal band: [{signal_window[0]:.5g}, {signal_window[1]:.5g}] {source['q_unit']}"
    title = (
        f"Observed intensity and peak landmarks\n{len(peaks)} supported lobe(s)"
        if width_mm == 89.0
        else f"Observed intensity and peak landmarks · {len(peaks)} supported lobe(s)"
    )
    fig.suptitle(
        title,
        x=0.5,
        y=0.97,
        fontsize=7.0,
        linespacing=0.95,
    )
    fig.text(0.5, 0.12 if width_mm == 183.0 else 0.19, signal_text, ha="center", fontsize=5.5)
    caption = (
        "G is unfiltered; P labels are supported intensity features, not reflection IDs."
        if width_mm == 183.0
        else "G is the unfiltered raw maximum.\nP labels mark supported intensity features, not reflection IDs."
    )
    fig.text(
        0.5,
        0.068 if width_mm == 183.0 else 0.115,
        caption,
        ha="center",
        va="center",
        fontsize=5.2,
        color="#555555",
    )
    if handles:
        if width_mm == 89.0:
            labels = {
                "G: raw maximum (not a reflection assignment)": "G · raw max",
                "Numbered supported lobe peaks": "Numbered P peaks",
                "Smoothed-model maximum in the same angular basin": "Smoothed model max (same basin)",
            }
            handles = [
                Line2D(
                    [0], [0],
                    marker=handle.get_marker(),
                    linestyle="none" if handle.get_marker() != "None" else "-",
                    color=handle.get_color(),
                    markerfacecolor=handle.get_markerfacecolor(),
                    markeredgecolor=handle.get_markeredgecolor(),
                    markersize=handle.get_markersize(),
                    linewidth=handle.get_linewidth(),
                    label=labels.get(handle.get_label(), handle.get_label()),
                )
                for handle in handles
            ]
        fig.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.012 if width_mm == 183.0 else 0.015),
            frameon=False,
            ncol=min(3, len(handles)) if width_mm == 183.0 else 1,
            fontsize=5.2,
            handletextpad=0.35,
            columnspacing=0.7,
        )
    return fig


def _build_peak_diagnostics(
    landmarks: Mapping[str, Any], *, width_mm: float, dpi: int
) -> Any:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    from .visualization import _diagnostic_q_axis_labels

    height_mm = 110.0 if width_mm == 183.0 else 145.0
    fig = Figure(
        figsize=(width_mm / 25.4, height_mm / 25.4),
        dpi=dpi,
        facecolor="white",
        edgecolor="white",
    )
    FigureCanvasAgg(fig)
    if width_mm == 183.0:
        angular_ax = fig.add_axes([0.08, 0.53, 0.38, 0.34])
        angular_count_ax = fig.add_axes([0.08, 0.34, 0.38, 0.12], sharex=angular_ax)
        radial_ax = fig.add_axes([0.57, 0.53, 0.36, 0.34])
        radial_count_ax = fig.add_axes([0.57, 0.34, 0.36, 0.12], sharex=radial_ax)
        title_y, legend_y, footer_y = 0.97, 0.075, 0.012
    else:
        angular_ax = fig.add_axes([0.16, 0.70, 0.75, 0.18])
        angular_count_ax = fig.add_axes([0.16, 0.62, 0.75, 0.06], sharex=angular_ax)
        radial_ax = fig.add_axes([0.16, 0.28, 0.75, 0.18])
        radial_count_ax = fig.add_axes([0.16, 0.20, 0.75, 0.06], sharex=radial_ax)
        title_y, legend_y, footer_y = 0.975, 0.055, 0.012
    angular = _profile(landmarks, "angular")
    radial = _profile(landmarks, "radial")
    peaks = [item for item in landmarks.get("peaks", ()) if isinstance(item, Mapping)]
    peak_legend_handles: list[Any] = []
    q_unit = str(landmarks.get("q_unit", "unknown"))
    qx_label, _ = _diagnostic_q_axis_labels(q_unit)
    common_style = {"direction": "out", "length": 2.0, "width": 0.5, "pad": 1.5}

    angle = angular.get("angle_deg", np.asarray([], dtype=float))
    angle_raw = angular.get("intensity_raw", np.asarray([], dtype=float))
    angle_smooth = angular.get("intensity_smoothed", np.asarray([], dtype=float))
    angle_reference = angular.get("intensity_isotropic_reference")
    angle_detection = angular.get("intensity_detection")
    if angle.size:
        angular_ax.plot(angle, angle_raw, color="#999999", linewidth=0.45, alpha=0.8, label="raw bin mean")
        angular_ax.plot(angle, angle_smooth, color="#0072B2", linewidth=0.9, label="smoothed bin mean")
        has_detection_profiles = False
        if angle_reference is not None and angle_reference.shape == angle.shape:
            angular_ax.plot(
                angle,
                angle_reference,
                color="#CC79A7",
                linewidth=0.75,
                linestyle="--",
                label="isotropic reference",
            )
            has_detection_profiles = True
        if angle_detection is not None and angle_detection.shape == angle.shape:
            angular_ax.plot(
                angle,
                angle_detection,
                color="#D55E00",
                linewidth=0.9,
                linestyle=":",
                label="detection (smoothed − reference)",
            )
            has_detection_profiles = True
        angular_ax.set_xlim(0.0, 360.0)
        angular_ax.set_ylabel("Mean intensity")
        angular_ax.grid(axis="y", color="#dddddd", linewidth=0.35)
        angular_ax.tick_params(**common_style)
        angular_ax.set_title(
            "Angular · raw / smoothed / reference / detection"
            if has_detection_profiles
            else "Angular · raw gray / smoothed blue / coverage green",
            fontsize=5.8,
        )
        angular_coverage = angular.get("coverage_fraction", np.zeros(angle.shape))
        coverage_ax = angular_ax.twinx()
        coverage_line, = coverage_ax.plot(
            angle,
            angular_coverage,
            color="#009E73",
            linewidth=0.55,
            linestyle="--",
            alpha=0.75,
            label="coverage (right axis)" if has_detection_profiles else None,
        )
        coverage_ax.set_ylim(0.0, 1.0)
        coverage_ax.set_ylabel("coverage", color="#009E73", fontsize=5.5)
        coverage_ax.tick_params(axis="y", colors="#009E73", labelsize=5.0, length=1.2, width=0.4, pad=1.0)
        if has_detection_profiles:
            handles, labels = angular_ax.get_legend_handles_labels()
            angular_ax.legend(
                handles + [coverage_line],
                labels + [coverage_line.get_label()],
                loc="upper left",
                bbox_to_anchor=(0.0, -0.80),
                frameon=False,
                ncol=2,
                fontsize=5.0,
                handlelength=1.0,
                handletextpad=0.3,
                columnspacing=0.7,
                labelspacing=0.2,
            )
        angular_counts = angular.get("valid_pixel_counts", np.zeros(angle.shape, dtype=int))
        angle_step = float(np.median(np.diff(angle))) if angle.size > 1 else 1.0
        angular_count_ax.bar(angle, angular_counts, width=angle_step * 0.86, color="#b8b8b8", edgecolor="none")
        angular_count_ax.set_ylabel("n", rotation=0, labelpad=6.0)
        angular_ax.tick_params(labelbottom=False)
        angular_count_ax.set_xticks(np.arange(0.0, 361.0, 90.0))
        angular_count_ax.set_xlabel("Azimuth χ (degrees)")
        angular_count_ax.tick_params(**common_style, labelsize=5.0)
        angular_count_ax.grid(axis="y", color="#eeeeee", linewidth=0.3)
        for index, peak in enumerate(peaks):
            color = _PEAK_COLORS[index % len(_PEAK_COLORS)]
            angular_ax.axvline(float(peak["angular_peak_deg"]), color=color, linewidth=0.75, alpha=0.9)
            angular_ax.text(
                float(peak["angular_peak_deg"]),
                0.96,
                str(peak.get("peak_id", f"P{index + 1}")),
                transform=angular_ax.get_xaxis_transform(),
                ha="center",
                va="top",
                color=color,
                fontsize=5.8,
                fontweight="bold",
            )
    else:
        angular_ax.text(0.5, 0.5, "No supported angular profile", transform=angular_ax.transAxes, ha="center", va="center")
        angular_count_ax.set_axis_off()

    q = radial.get("q", np.asarray([], dtype=float))
    radial_raw = radial.get("mean_intensity_raw", np.asarray([], dtype=float))
    radial_smooth = radial.get("mean_intensity_smoothed", np.asarray([], dtype=float))
    if q.size:
        radial_ax.plot(q, radial_raw, color="#999999", linewidth=0.45, alpha=0.8, label="raw bin mean")
        radial_ax.plot(q, radial_smooth, color="#0072B2", linewidth=0.9, label="smoothed bin mean")
        radial_ax.set_ylabel("Mean intensity")
        radial_ax.tick_params(labelbottom=False)
        radial_ax.set_title("Radial · raw gray / smoothed blue", fontsize=5.8)
        radial_ax.grid(axis="y", color="#dddddd", linewidth=0.35)
        radial_ax.tick_params(**common_style)
        radial_counts = radial.get("valid_pixel_counts", np.zeros(q.shape, dtype=int))
        radial_count_ax.bar(q, radial_counts, width=(q[1] - q[0]) * 0.86 if q.size > 1 else 1.0, color="#b8b8b8", edgecolor="none")
        radial_count_ax.set_ylabel("n", rotation=0, labelpad=6.0)
        radial_count_ax.set_xlabel(f"q radius ({q_unit})")
        radial_count_ax.tick_params(**common_style, labelsize=5.0)
        radial_count_ax.grid(axis="y", color="#eeeeee", linewidth=0.3)
        for index, peak in enumerate(peaks):
            color = _PEAK_COLORS[index % len(_PEAK_COLORS)]
            radial_ax.axvline(float(peak["q"]), color=color, linewidth=0.75, alpha=0.9)
            peak_legend_handles.append(
                Line2D(
                    [0], [0], color=color, linewidth=0.9,
                    label=f"{peak.get('peak_id', f'P{index + 1}')} local pixel q",
                )
            )
        raw_max = landmarks.get("raw_global_max")
        if isinstance(raw_max, Mapping) and q.min() <= float(raw_max["q"]) <= q.max():
            radial_ax.axvline(float(raw_max["q"]), color="#D55E00", linewidth=0.7, linestyle=":")
            radial_ax.text(
                float(raw_max["q"]),
                0.80,
                "raw max q",
                transform=radial_ax.get_xaxis_transform(),
                ha="left",
                va="top",
                color="#D55E00",
                fontsize=5.3,
                rotation=90,
            )
    else:
        radial_ax.text(0.5, 0.5, "No supported radial profile", transform=radial_ax.transAxes, ha="center", va="center")
        radial_count_ax.set_axis_off()
    fig.suptitle("Observed peak diagnostics", x=0.5, y=title_y, fontsize=7.0)
    if peak_legend_handles:
        fig.legend(
            handles=peak_legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, legend_y),
            frameon=False,
            ncol=min(4, len(peak_legend_handles)) if width_mm == 183.0 else 2,
            fontsize=5.2 if width_mm == 183.0 else 5.0,
            handlelength=1.1,
            handletextpad=0.35,
            columnspacing=0.7,
        )
    if not peaks:
        fig.text(0.5, title_y - 0.08, "No supported lobe peaks; missing lobes are not imputed.", ha="center", fontsize=5.8)
    footer = (
        "Colored q lines mark local 2D pixel maxima, not radial-profile fits; profiles do not assign reflections or establish model acceptance."
        if width_mm == 183.0
        else "Local pixel q lines are not radial-profile fits; no reflection is assigned."
    )
    fig.text(
        0.5,
        footer_y,
        footer,
        ha="center",
        va="bottom",
        fontsize=5.0,
        color="#555555",
    )
    return fig


def _build_peak_zooms(
    *,
    base: Mapping[str, Any] | None,
    source: Mapping[str, Any],
    landmarks: Mapping[str, Any],
    width_mm: float,
    dpi: int,
) -> Any:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.colors import Normalize

    from .visualization import _diagnostic_q_axis_labels, _display_transform

    height_mm = 108.0 if width_mm == 183.0 else 145.0
    fig = Figure(
        figsize=(width_mm / 25.4, height_mm / 25.4),
        dpi=dpi,
        facecolor="white",
        edgecolor="white",
    )
    FigureCanvasAgg(fig)
    peaks = [item for item in landmarks.get("peaks", ()) if isinstance(item, Mapping)]
    qx_label, qy_label = _diagnostic_q_axis_labels(source["q_unit"])
    if not peaks or base is None:
        ax = fig.add_axes([0.1, 0.15, 0.8, 0.74])
        message = "No supported peaks to zoom" if not peaks else "No valid source pixels to zoom"
        ax.text(0.5, 0.5, message, transform=ax.transAxes, ha="center", va="center", color="#555555")
        ax.set_axis_off()
        title = "Local peak neighborhoods\non supplied q map" if width_mm == 89.0 else "Local peak neighborhoods"
        fig.suptitle(title, y=0.96, fontsize=7.0, linespacing=0.95)
        return fig

    rows, cols = np.shape(base["observed_float"])
    grid_ncols = 2 if len(peaks) > 1 else 1
    grid_nrows = int(math.ceil(len(peaks) / grid_ncols))
    if width_mm == 183.0:
        left, right, colorbar_x = 0.075, 0.91, 0.94
    else:
        left, right, colorbar_x = 0.17, 0.82, 0.90
    bottom, top = 0.12, 0.87
    gap_x, gap_y = 0.055, 0.12
    cell_w = (right - left - gap_x * (grid_ncols - 1)) / grid_ncols
    cell_h = (top - bottom - gap_y * (grid_nrows - 1)) / grid_nrows
    norm = Normalize(float(base["display_low"]), float(base["display_high"]), clip=True)
    last_mesh = None
    observed = np.asarray(base["observed_float"], dtype=np.float64)
    qx = np.asarray(base["qx_float"], dtype=np.float64)
    qy = np.asarray(base["qy_float"], dtype=np.float64)
    valid = np.asarray(base["effective_valid"], dtype=bool)
    display_scale = str(base.get("display_scale", "asinh"))
    for index, peak in enumerate(peaks):
        grid_row, grid_col = divmod(index, grid_ncols)
        x = left + grid_col * (cell_w + gap_x)
        y = top - (grid_row + 1) * cell_h - grid_row * gap_y
        ax = fig.add_axes([x, y, cell_w, cell_h])
        row = int(peak["pixel_y"])
        col = int(peak["pixel_x"])
        half = 14
        y0, y1 = max(0, row - half), min(rows, row + half + 1)
        x0, x1 = max(0, col - half), min(cols, col + half + 1)
        valid_local = valid[y0:y1, x0:x1]
        raw_local = observed[y0:y1, x0:x1]
        qx_local = qx[y0:y1, x0:x1]
        qy_local = qy[y0:y1, x0:x1]
        display_local = _display_transform(np.where(valid_local, raw_local, np.nan), display_scale)
        masked = np.ma.masked_where(~valid_local | ~np.isfinite(display_local), display_local)
        if min(masked.shape) >= 2 and np.all(np.isfinite(qx_local)) and np.all(np.isfinite(qy_local)):
            last_mesh = ax.pcolormesh(
                qx_local,
                qy_local,
                masked,
                shading="auto",
                cmap="cividis",
                norm=norm,
                rasterized=True,
            )
        else:
            finite = valid_local & np.isfinite(qx_local) & np.isfinite(qy_local)
            if np.any(finite):
                last_mesh = ax.scatter(
                    qx_local[finite],
                    qy_local[finite],
                    c=display_local[finite],
                    cmap="cividis",
                    norm=norm,
                    marker="s",
                    s=8.0,
                    rasterized=True,
                )
        color = _PEAK_COLORS[index % len(_PEAK_COLORS)]
        ax.scatter([peak["qx"]], [peak["qy"]], marker="o", s=22, facecolors="none", edgecolors=color, linewidths=1.0, zorder=5)
        model_peak = peak.get("model_peak")
        if isinstance(model_peak, Mapping):
            ax.scatter([model_peak["qx"]], [model_peak["qy"]], marker="x", s=18, c="#111111", linewidths=0.8, zorder=6)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(qx_label)
        ax.set_ylabel(qy_label)
        ax.set_title(
            f"{peak.get('peak_id', f'P{index + 1}')} · χ={float(peak['chi_deg']):.1f}° · q={float(peak['q']):.5g}",
            fontsize=6.4,
        )
        ax.tick_params(direction="out", length=1.5, width=0.45, labelsize=5.0, pad=1.2)
        for spine in ax.spines.values():
            spine.set_linewidth(0.45)
    if last_mesh is not None:
        cax = fig.add_axes([colorbar_x, 0.28, 0.016 if width_mm == 183.0 else 0.025, 0.43])
        colorbar = fig.colorbar(last_mesh, cax=cax)
        colorbar.set_label(
            str(base.get("display_label", "Intensity display transform")),
            fontsize=5.6,
            labelpad=2.0 if width_mm == 183.0 else 0.8,
        )
        colorbar.ax.tick_params(labelsize=5.0, length=1.2, width=0.4, pad=1.0)
    title = (
        "Local peak neighborhoods\non supplied q map"
        if width_mm == 89.0
        else "Local peak neighborhoods on the supplied q map"
    )
    fig.suptitle(title, y=0.965, fontsize=7.0, linespacing=0.95)
    return fig


def export_peak_landmark_figures(
    stage: str | Path,
    *,
    data: Mapping[str, Any],
    landmarks: Mapping[str, Any],
    model: Any = None,
    cancel_event: Any = None,
) -> tuple[dict[str, Path], dict[str, Any]]:
    """Write peak figures and machine-readable evidence into an existing stage.

    The function never creates or publishes a target directory. The caller owns
    the staging directory and any atomic publication around it.
    """

    raise_if_cancelled(cancel_event, "peak-landmark-figures:validate")
    stage_path = Path(stage).expanduser().resolve(strict=True)
    if not stage_path.is_dir():
        raise ValueError("stage must be an existing directory")
    if not isinstance(data, Mapping) or not isinstance(landmarks, Mapping):
        raise TypeError("data and landmarks must be mappings")
    source = _source_payload(data, landmarks)
    width_mm = float(data.get("width_mm", 183.0))
    if width_mm not in {89.0, 183.0}:
        raise ValueError("data.width_mm must be 89 or 183")
    dpi_raw = data.get("dpi", 600)
    if isinstance(dpi_raw, bool) or int(dpi_raw) != dpi_raw or int(dpi_raw) <= 0:
        raise ValueError("data.dpi must be a positive integer")
    dpi = int(dpi_raw)

    from .figure_support import _EXPORT_LOCK, _FIGURE_RC, _prepare_inputs
    import matplotlib as mpl

    observed_raw = np.asanyarray(source["observed"])
    qx_raw = np.asanyarray(source["qx"])
    qy_raw = np.asanyarray(source["qy"])
    if observed_raw.ndim != 2 or qx_raw.shape != observed_raw.shape or qy_raw.shape != observed_raw.shape:
        raise ValueError("observed, qx, and qy must be shape-matched 2D arrays")
    obs = np.asarray(np.ma.getdata(observed_raw), dtype=np.float64)
    qx_values = np.asarray(np.ma.getdata(qx_raw), dtype=np.float64)
    qy_values = np.asarray(np.ma.getdata(qy_raw), dtype=np.float64)
    finite_valid = (
        np.isfinite(obs)
        & np.isfinite(qx_values)
        & np.isfinite(qy_values)
        & ~np.ma.getmaskarray(observed_raw)
        & ~np.ma.getmaskarray(qx_raw)
        & ~np.ma.getmaskarray(qy_raw)
    )
    if source["valid_mask"] is not None:
        mask_array = np.asanyarray(source["valid_mask"])
        mask_values = np.asarray(np.ma.getdata(mask_array))
        if mask_values.shape != obs.shape or mask_values.dtype.kind != "b":
            raise ValueError("valid_mask must be a shape-matched boolean array")
        finite_valid &= mask_values & ~np.ma.getmaskarray(mask_array)
    model_shape = _model_shape(model)
    if model_shape is not None and model_shape != obs.shape:
        raise ValueError("model must have the same 2D shape as observed")

    base: dict[str, Any] | None = None
    if np.any(finite_valid):
        base = _prepare_inputs(
            observed=source["observed"],
            qx=source["qx"],
            qy=source["qy"],
            valid_mask=finite_valid,
            result={},
            q_unit=source["q_unit"],
            context=source["context"],
            display_scale="asinh",
            width_mm=width_mm,
            dpi=dpi,
        )
    raise_if_cancelled(cancel_event, "peak-landmark-figures:prepare")

    filenames = {
        "peak_map_svg": "peak_map.svg",
        "peak_map_pdf": "peak_map.pdf",
        "peak_map_png": "peak_map.png",
        "peak_map_tiff": "peak_map.tiff",
        "peak_diagnostics_svg": "peak_diagnostics.svg",
        "peak_diagnostics_pdf": "peak_diagnostics.pdf",
        "peak_diagnostics_png": "peak_diagnostics.png",
        "peak_diagnostics_tiff": "peak_diagnostics.tiff",
        "peak_zooms_svg": "peak_zooms.svg",
        "peak_zooms_pdf": "peak_zooms.pdf",
        "peak_zooms_png": "peak_zooms.png",
        "peak_zooms_tiff": "peak_zooms.tiff",
        "peak_landmarks_json": "peak_landmarks.json",
        "peak_landmarks_csv": "peak_landmarks.csv",
        "peak_profiles_csv": "peak_profiles.csv",
        "peak_profiles_npz": "peak_profiles.npz",
        "peak_manifest": "peak_manifest.json",
    }
    outputs = {key: stage_path / name for key, name in filenames.items()}
    if any(path.exists() for path in outputs.values()):
        existing = next(path for path in outputs.values() if path.exists())
        raise FileExistsError(f"peak figure output already exists in stage: {existing.name}")
    figure_specs = (
        (
            "peak_map",
            lambda: _build_peak_map(
                base=base,
                source=source,
                landmarks=landmarks,
                width_mm=width_mm,
                dpi=dpi,
            ),
        ),
        (
            "peak_diagnostics",
            lambda: _build_peak_diagnostics(landmarks, width_mm=width_mm, dpi=dpi),
        ),
        (
            "peak_zooms",
            lambda: _build_peak_zooms(
                base=base,
                source=source,
                landmarks=landmarks,
                width_mm=width_mm,
                dpi=dpi,
            ),
        ),
    )
    with _EXPORT_LOCK, mpl.rc_context(_FIGURE_RC):
        for stem, build in figure_specs:
            raise_if_cancelled(cancel_event, f"peak-landmark-figures:render-{stem}")
            figure = build()
            figure.canvas.draw()
            for suffix, file_format in (("svg", "svg"), ("pdf", "pdf")):
                raise_if_cancelled(cancel_event, f"peak-landmark-figures:save-{stem}-{suffix}")
                figure.savefig(
                    outputs[f"{stem}_{suffix}"],
                    format=file_format,
                    facecolor="white",
                    transparent=False,
                )
            for suffix, file_format in (("png", "png"), ("tiff", "tiff")):
                raise_if_cancelled(cancel_event, f"peak-landmark-figures:save-{stem}-{suffix}")
                figure.savefig(
                    outputs[f"{stem}_{suffix}"],
                    format=file_format,
                    dpi=dpi,
                    facecolor="white",
                    transparent=False,
                )
            figure.clear()

    raise_if_cancelled(cancel_event, "peak-landmark-figures:write-data")
    _write_json(outputs["peak_landmarks_json"], landmarks)
    _write_landmark_csv(outputs["peak_landmarks_csv"], landmarks)
    _write_profile_csv(outputs["peak_profiles_csv"], landmarks)
    _write_profiles_npz(outputs["peak_profiles_npz"], landmarks)

    for path in outputs.values():
        if path.parent != stage_path:
            raise RuntimeError("peak exporter attempted to write outside its caller-owned stage")
    file_hashes = {
        path.name: _sha256(path)
        for key, path in outputs.items()
        if key != "peak_manifest"
    }
    angular_detection = landmarks.get("angular_detection")
    if not isinstance(angular_detection, Mapping):
        angular_detection = {}
    metadata = {
        "schema_version": "peak-landmark-export-v1",
        "method_version": PEAK_FIGURE_METHOD_VERSION,
        "q_unit": source["q_unit"],
        "input_shape": list(obs.shape),
        "valid_pixel_count": int(np.count_nonzero(finite_valid)),
        "raw_search_pixel_count": landmarks.get("domain", {}).get("raw_search_pixel_count"),
        "search_q_window": landmarks.get("domain", {}).get("search_q_window"),
        "signal_q_window": landmarks.get("domain", {}).get("signal_q_window"),
        "effective_signal_q_window": landmarks.get("domain", {}).get("effective_signal_q_window"),
        "signal_q_window_source": landmarks.get("domain", {}).get("signal_q_window_source"),
        "signal_q_window_origin": landmarks.get("domain", {}).get("signal_q_window_origin"),
        "supported_lobe_count": len(landmarks.get("peaks", ())),
        "peak_selection_method": landmarks.get("peak_selection_method"),
        "angular_detection_method": angular_detection.get("method"),
        "angular_detection_reference_method": angular_detection.get("reference_method"),
        "angular_reference_profile_key": angular_detection.get("reference_profile_key"),
        "angular_detection_profile_key": angular_detection.get("detection_profile_key"),
        "model_selection_method": landmarks.get("model_comparison", {}).get("selection_method"),
        "delta_q_definition": landmarks.get("model_comparison", {}).get("delta_q_definition"),
        "width_mm": width_mm,
        "dpi": dpi,
        "model_supplied": model is not None,
        "model_input_shape": list(model_shape) if model_shape is not None else None,
        "model_peak_pair_count": sum(
            isinstance(peak, Mapping) and isinstance(peak.get("model_peak"), Mapping)
            for peak in landmarks.get("peaks", ())
        ),
        "files": list(file_hashes),
        "sha256": file_hashes,
        "manifest_excluded_from_own_sha256": True,
    }
    _write_json(outputs["peak_manifest"], metadata)
    return outputs, json_safe(metadata)


__all__ = ["PEAK_FIGURE_METHOD_VERSION", "export_peak_landmark_figures"]
