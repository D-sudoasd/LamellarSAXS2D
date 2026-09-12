"""Pure task helpers for the butterfly measurement-figure worker."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any


def new_figure_export_target(parent: str | Path) -> Path:
    """Return a new child directory without creating or overwriting anything."""

    root = Path(parent).expanduser().resolve()
    candidate = root / "butterfly-figure"
    index = 2
    while candidate.exists():
        candidate = root / f"butterfly-figure-{index}"
        index += 1
    return candidate


def run_butterfly_figure_export(*, payload: Mapping[str, Any], **_: Any) -> dict[str, Path]:
    """Execute a detached figure snapshot without touching Qt widgets."""

    from ..butterfly_figure import export_butterfly_figure

    progress_callback = payload.get("progress")

    def report_progress(percent: int, phase: str) -> None:
        if callable(progress_callback):
            progress_callback({"percent": int(percent), "phase": str(phase)})

    comparison_inputs = {"model": payload["model"]} if payload.get("model") is not None else {}
    return export_butterfly_figure(
        payload["target"],
        observed=payload["observed"],
        qx=payload["qx"],
        qy=payload["qy"],
        valid_mask=payload.get("valid_mask"),
        result=payload["result"],
        q_unit=str(payload.get("q_unit", "unknown")),
        context=payload.get("context"),
        display_scale=str(payload.get("display_scale", "log1p")),
        width_mm=float(payload["width_mm"]),
        dpi=int(payload.get("dpi", 600)),
        cancel_event=payload["cancel_event"],
        progress=report_progress,
        **comparison_inputs,
    )


__all__ = ["new_figure_export_target", "run_butterfly_figure_export"]
