"""Scientific diagnostic figures shared by the UI and exporters."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


OKABE_ITO = {
    "data": "#0072B2",
    "model": "#D55E00",
    "ridge": "#F0E442",
    "ellipse_a": "#56B4E9",
    "ellipse_b": "#E69F00",
    "failed": "#000000",
}


def _finite_limits(values: np.ndarray, lower: float = 1.0, upper: float = 99.5) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(finite, [lower, upper])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        center = float(finite[0])
        delta = max(abs(center) * 0.01, 1.0)
        return center - delta, center + delta
    return float(lo), float(hi)


def _extent(qx: np.ndarray, qy: np.ndarray) -> tuple[float, float, float, float]:
    finite_x = np.asarray(qx, dtype=float)[np.isfinite(qx)]
    finite_y = np.asarray(qy, dtype=float)[np.isfinite(qy)]
    if finite_x.size == 0 or finite_y.size == 0:
        raise ValueError("qx/qy 中没有有限坐标")
    return (
        float(np.min(finite_x)),
        float(np.max(finite_x)),
        float(np.min(finite_y)),
        float(np.max(finite_y)),
    )


def plot_fit_diagnostics(
    observed: np.ndarray,
    model: np.ndarray,
    qx: np.ndarray,
    qy: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
    ridge_xy: np.ndarray | Sequence[Sequence[float]] | None = None,
    ellipse_curves: Iterable[np.ndarray | Sequence[Sequence[float]]] = (),
    output: str | Path | None = None,
    title: str | None = None,
    dpi: int = 300,
) -> Any:
    """Create observed/model/residual/overlay diagnostics with honest shared scales.

    Observed and model panels use the same intensity limits. Residuals always use a
    zero-centered diverging scale. The function returns the Matplotlib ``Figure``
    and optionally writes a lossless PNG (or a vector format selected by suffix).
    """

    import matplotlib.pyplot as plt

    obs = np.asarray(observed, dtype=float)
    mod = np.asarray(model, dtype=float)
    if obs.shape != mod.shape:
        raise ValueError("observed 与 model 的 shape 必须一致")
    if np.shape(qx) != obs.shape or np.shape(qy) != obs.shape:
        raise ValueError("qx/qy 必须与图像 shape 一致")
    valid = np.isfinite(obs) & np.isfinite(mod)
    if valid_mask is not None:
        mask = np.asarray(valid_mask, dtype=bool)
        if mask.shape != obs.shape:
            raise ValueError("valid_mask 与图像 shape 不一致")
        valid &= mask
    residual = np.where(valid, obs - mod, np.nan)
    obs_show = np.where(valid, obs, np.nan)
    mod_show = np.where(valid, mod, np.nan)
    data_lo, data_hi = _finite_limits(np.concatenate([obs_show[valid], mod_show[valid]]))
    resid_finite = residual[np.isfinite(residual)]
    resid_lim = float(np.percentile(np.abs(resid_finite), 99.0)) if resid_finite.size else 1.0
    if not np.isfinite(resid_lim) or resid_lim <= 0:
        resid_lim = 1.0
    ext = _extent(np.asarray(qx), np.asarray(qy))

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.4), constrained_layout=True)
    panels = (
        (axes[0, 0], obs_show, "Observed", "cividis", data_lo, data_hi),
        (axes[0, 1], mod_show, "Model", "cividis", data_lo, data_hi),
        (axes[1, 0], residual, "Residual", "PuOr", -resid_lim, resid_lim),
        (axes[1, 1], obs_show, "Overlay", "cividis", data_lo, data_hi),
    )
    for label, (ax, array, panel_title, cmap, vmin, vmax) in zip("ABCD", panels):
        image = ax.imshow(
            array,
            origin="lower",
            extent=ext,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
            aspect="equal",
        )
        ax.set_title(panel_title, fontsize=9)
        ax.set_xlabel(r"$q_x$ (nm$^{-1}$)")
        ax.set_ylabel(r"$q_y$ (nm$^{-1}$)")
        ax.text(-0.13, 1.04, label, transform=ax.transAxes, fontweight="bold", fontsize=10)
        fig.colorbar(image, ax=ax, shrink=0.82, label="Intensity (input units)" if panel_title != "Residual" else "Data - model")

    overlay = axes[1, 1]
    if ridge_xy is not None:
        points = np.asarray(ridge_xy, dtype=float)
        if points.ndim == 2 and points.shape[1] == 2 and points.size:
            overlay.scatter(
                points[:, 0],
                points[:, 1],
                s=10,
                marker="o",
                facecolors="none",
                edgecolors=OKABE_ITO["ridge"],
                linewidths=0.7,
                label="Observed ridge",
            )
    for index, curve in enumerate(ellipse_curves):
        xy = np.asarray(curve, dtype=float)
        if xy.ndim != 2 or xy.shape[1] != 2 or not xy.size:
            continue
        color = OKABE_ITO["ellipse_a"] if index % 2 == 0 else OKABE_ITO["ellipse_b"]
        overlay.plot(xy[:, 0], xy[:, 1], color=color, lw=1.3, ls="-" if index % 2 == 0 else "--", label=f"Ellipse {index + 1}")
    handles, labels = overlay.get_legend_handles_labels()
    if handles:
        overlay.legend(frameon=False, fontsize=7, loc="best")
    if title:
        fig.suptitle(title, fontsize=10)
    if output is not None:
        target = Path(output)
        target.parent.mkdir(parents=True, exist_ok=True)
        save_kwargs: dict[str, Any] = {"bbox_inches": "tight"}
        if target.suffix.lower() in {".png", ".tif", ".tiff"}:
            save_kwargs["dpi"] = int(dpi)
        fig.savefig(target, **save_kwargs)
    return fig

def plot_parameter_evolution(
    rows: Sequence[Mapping[str, Any]],
    *,
    parameters: Sequence[str],
    x_key: str = "time_s",
    output: str | Path | None = None,
    dpi: int = 300,
) -> Any:
    """Plot fitted parameters without hiding failed frames or missing values."""

    import matplotlib.pyplot as plt

    if not parameters:
        raise ValueError("parameters 不能为空")
    x_values: list[float] = []
    for index, row in enumerate(rows):
        raw = row.get(x_key)
        try:
            x = float(raw) if raw is not None else float(index)
        except (TypeError, ValueError):
            x = float(index)
        x_values.append(x)

    fig, axes = plt.subplots(len(parameters), 1, figsize=(7.0, max(2.2, 2.1 * len(parameters))), sharex=True, constrained_layout=True)
    axes_arr = np.atleast_1d(axes)
    for ax, name in zip(axes_arr, parameters):
        y = np.full(len(rows), np.nan, dtype=float)
        err = np.full(len(rows), np.nan, dtype=float)
        failed_x: list[float] = []
        for i, row in enumerate(rows):
            status = str(row.get("status", "ok"))
            value = row.get(name)
            if status not in {"ok", "success", "partial", "recovered"}:
                failed_x.append(x_values[i])
                continue
            try:
                y[i] = float(value)
            except (TypeError, ValueError):
                continue
            for key in (f"{name}_stderr", f"stderr_{name}"):
                if row.get(key) is not None:
                    try:
                        err[i] = float(row[key])
                    except (TypeError, ValueError):
                        pass
                    break
        ok = np.isfinite(y)
        if np.any(ok):
            yerr = np.where(np.isfinite(err[ok]), err[ok], 0.0)
            ax.errorbar(
                np.asarray(x_values)[ok],
                y[ok],
                yerr=yerr,
                color=OKABE_ITO["data"],
                marker="o",
                ms=3.5,
                lw=1.2,
                capsize=2,
                label="Fit (error bar = stderr when available)",
            )
        if failed_x:
            bottom, top = ax.get_ylim()
            marker_y = bottom + 0.04 * (top - bottom if top > bottom else 1.0)
            ax.scatter(failed_x, np.full(len(failed_x), marker_y), marker="x", color=OKABE_ITO["failed"], s=22, label="Failed/missing")
        ax.set_ylabel(name)
        ax.spines[["top", "right"]].set_visible(False)
        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(frameon=False, fontsize=7, loc="best")
    axes_arr[-1].set_xlabel("Time (s)" if x_key == "time_s" else x_key)
    if output is not None:
        target = Path(output)
        target.parent.mkdir(parents=True, exist_ok=True)
        save_kwargs: dict[str, Any] = {"bbox_inches": "tight"}
        if target.suffix.lower() in {".png", ".tif", ".tiff"}:
            save_kwargs["dpi"] = int(dpi)
        fig.savefig(target, **save_kwargs)
    return fig
