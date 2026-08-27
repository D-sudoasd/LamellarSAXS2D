"""Render compact, read-only visual overviews for P3 evidence directories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON 顶层必须是 object：{path}")
    return value


def _display(array: np.ndarray, mask: np.ndarray | None = None) -> np.ma.MaskedArray:
    values = np.asarray(array, dtype=float)
    finite = np.isfinite(values)
    if not np.any(finite):
        raise ValueError("证据图像没有有限像素")
    low = float(np.percentile(values[finite], 1.0))
    shifted = np.log1p(np.clip(values - low, 0.0, None))
    invalid = ~finite
    if mask is not None:
        invalid |= np.asarray(mask, dtype=bool)
    return np.ma.array(shifted, mask=invalid)


def _prepare_output(output: Path, names: tuple[str, ...], force: bool) -> None:
    targets = [output / name for name in names]
    existing = [path for path in targets if path.exists()]
    if existing and not force:
        raise FileExistsError("输出已存在，未覆盖：" + ", ".join(str(path) for path in existing))
    output.mkdir(parents=True, exist_ok=True)


def render_overviews(
    t1_manifest: Path,
    t2_manifest: Path,
    annotation_status: Path,
    output: Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    names = ("t1_overview.png", "t2_overview.png", "r0_annotation_overview.png", "overview_qc.json")
    _prepare_output(output, names, force)
    cmap = plt.get_cmap("magma").with_extremes(bad="black")

    t1 = _json(t1_manifest)
    t1_cases = t1["cases"]
    fig, axes = plt.subplots(3, 5, figsize=(13.5, 8.2), constrained_layout=True)
    t1_q_ok = True
    t1_finite = True
    for ax, record in zip(axes.flat, t1_cases, strict=True):
        with np.load(t1_manifest.parent / record["npz"], allow_pickle=False) as archive:
            intensity = np.asarray(archive["intensity"])
            mask = np.asarray(archive["mask"])
            qx = np.asarray(archive["qx"])
            qy = np.asarray(archive["qy"])
            q = np.asarray(archive["q"])
        t1_finite &= bool(np.isfinite(intensity).all())
        t1_q_ok &= bool(np.allclose(q, np.hypot(qx, qy), rtol=1e-12, atol=1e-12))
        ax.imshow(_display(intensity, mask), cmap=cmap, origin="lower")
        ax.set_title(str(record["name"]), fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("P3 / T1 same-model synthetic matrix (log display; masks black)", fontsize=13)
    t1_output = output / "t1_overview.png"
    fig.savefig(t1_output, dpi=200)
    plt.close(fig)

    t2 = _json(t2_manifest)
    t2_cases = t2["cases"]
    fig, axes = plt.subplots(2, 4, figsize=(12.5, 6.3), constrained_layout=True)
    t2_q_ok = True
    t2_nonnegative = True
    for column, record in enumerate(t2_cases):
        with np.load(t2_manifest.parent / record["npz_file"], allow_pickle=False) as archive:
            density = np.asarray(archive["real_space_density"])
            intensity = np.asarray(archive["intensity_noiseless"])
            mask = np.asarray(archive["mask"])
            qx = np.asarray(archive["qx"])
            qy = np.asarray(archive["qy"])
            q = np.asarray(archive["q"])
            reference = np.asarray(archive["projection_reference"])
        t2_nonnegative &= bool(np.isfinite(intensity).all() and np.all(intensity >= 0))
        t2_q_ok &= bool(np.allclose(q, np.hypot(qx, qy), rtol=1e-12, atol=1e-12))
        axes[0, column].imshow(density, cmap="gray", origin="lower")
        axes[0, column].set_title(f"{record['category']} / real space", fontsize=9)
        extent = [float(qx.min()), float(qx.max()), float(qy.min()), float(qy.max())]
        axes[1, column].imshow(
            _display(intensity, mask), cmap=cmap, origin="lower", extent=extent
        )
        axes[1, column].scatter(
            reference[:, 0], reference[:, 1], s=3, c="cyan", alpha=0.55, linewidths=0
        )
        q_crop = max(0.75, 1.25 * float(np.max(np.abs(reference))))
        axes[1, column].set_xlim(-q_crop, q_crop)
        axes[1, column].set_ylim(-q_crop, q_crop)
        axes[1, column].set_title("central FFT q + projection reference", fontsize=9)
        for row in range(2):
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
    fig.suptitle("P3 / T2 independent real-space lamellae + FFT", fontsize=13)
    t2_output = output / "t2_overview.png"
    fig.savefig(t2_output, dpi=200)
    plt.close(fig)

    annotation = _json(annotation_status)
    png_paths = sorted((annotation_status.parent / "blind_payload").glob("blind_*.png"))
    if len(png_paths) != 8:
        raise ValueError(f"盲标 PNG 应为 8 张，实际为 {len(png_paths)}")
    fig, axes = plt.subplots(2, 4, figsize=(12.5, 6.5), constrained_layout=True)
    for ax, path in zip(axes.flat, png_paths, strict=True):
        ax.imshow(mpimg.imread(path))
        ax.set_title(path.stem, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("R0 blind annotation pack (no fit overlays)", fontsize=13)
    annotation_output = output / "r0_annotation_overview.png"
    fig.savefig(annotation_output, dpi=200)
    plt.close(fig)

    input_hashes = annotation.get("input_hashes", [])
    report = {
        "schema_version": "lamellarsaxs2d.p3_overview_qc.v1",
        "t1": {
            "case_count": len(t1_cases),
            "all_intensity_finite": t1_finite,
            "q_identity_all_cases": t1_q_ok,
        },
        "t2": {
            "case_count": len(t2_cases),
            "categories": [record["category"] for record in t2_cases],
            "all_clean_intensity_finite_nonnegative": t2_nonnegative,
            "q_identity_all_cases": t2_q_ok,
        },
        "r0_annotation_pack": {
            "png_count": len(png_paths),
            "input_hashes_unchanged": bool(input_hashes)
            and all(item.get("unchanged") is True for item in input_hashes),
            "status": annotation.get("status"),
            "human_consensus": annotation.get("human_consensus"),
        },
        "outputs": {
            "t1_overview": t1_output.as_posix(),
            "t2_overview": t2_output.as_posix(),
            "r0_annotation_overview": annotation_output.as_posix(),
        },
    }
    (output / "overview_qc.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="生成 P3 证据总览图和只读 QC 摘要")
    parser.add_argument("--t1-manifest", type=Path, required=True)
    parser.add_argument("--t2-manifest", type=Path, required=True)
    parser.add_argument("--annotation-status", type=Path, required=True)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    report = render_overviews(
        args.t1_manifest,
        args.t2_manifest,
        args.annotation_status,
        args.output,
        force=args.force,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
