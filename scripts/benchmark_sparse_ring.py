"""Benchmark the sparse first-order sector kernel before and after packing.

The benchmark uses a deterministic synthetic ring and an in-script copy of
the pre-optimization sector scan.  It measures the 16-sector peak kernel;
point/arc bookkeeping and scientific acceptance are intentionally excluded.
Results are evidence for engineering performance only.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

from butterfly_saxs.butterfly_ridge import (
    METHOD_VERSION,
    _prepare_sparse_first_order_samples,
    _sector_first_order_peak,
)


SECTORS = tuple(
    30.0 + quadrant + offset
    for quadrant in (0.0, 90.0, 180.0, 270.0)
    for offset in (30.0, 45.0, 60.0, 75.0)
)


def legacy_sector_first_order_peak(
    qx: np.ndarray,
    qy: np.ndarray,
    q: np.ndarray,
    intensity: np.ndarray,
    valid: np.ndarray,
    *,
    sector_deg: float,
    halfwidth_deg: float,
    hint: float,
) -> dict[str, float] | None:
    """Reference sector scan corresponding to the pre-optimization path."""

    if not (np.isfinite(hint) and hint > 0.0):
        return None
    ang = np.degrees(np.arctan2(qy, qx))
    delta = np.abs(((ang - float(sector_deg) + 180.0) % 360.0) - 180.0)
    q_lo, q_hi = 0.70 * hint, 1.45 * hint
    selected = (
        np.asarray(valid, dtype=bool)
        & np.isfinite(q)
        & np.isfinite(intensity)
        & (delta <= float(halfwidth_deg))
        & (q >= q_lo)
        & (q <= q_hi)
    )
    if int(np.count_nonzero(selected)) < 12:
        return None
    radii = np.asarray(q[selected], dtype=float)
    values = np.asarray(intensity[selected], dtype=float)
    edges = np.linspace(q_lo, q_hi, 9)
    profile = np.full(edges.size - 1, np.nan, dtype=float)
    counts = np.zeros(edges.size - 1, dtype=int)
    idx = np.digitize(radii, edges) - 1
    for bin_i in range(edges.size - 1):
        in_bin = idx == bin_i
        counts[bin_i] = int(np.count_nonzero(in_bin))
        if counts[bin_i] >= 2:
            profile[bin_i] = float(np.nanmedian(values[in_bin]))
    usable = np.isfinite(profile)
    if int(np.count_nonzero(usable)) < 3:
        return None
    peak_i = int(np.nanargmax(np.where(usable, profile, -np.inf)))
    peak = float(profile[peak_i])
    baseline = float(np.nanmedian(profile[usable]))
    if not (np.isfinite(peak) and np.isfinite(baseline) and baseline > 0 and peak >= 1.30 * baseline):
        return None
    q_star = float(0.5 * (edges[peak_i] + edges[peak_i + 1]))
    in_bin = selected & (q >= edges[peak_i]) & (q < edges[peak_i + 1])
    if int(np.count_nonzero(in_bin)) < 3:
        in_bin = selected
    median_qx = float(np.nanmedian(qx[in_bin]))
    median_qy = float(np.nanmedian(qy[in_bin]))
    if not (np.isfinite(median_qx) and np.isfinite(median_qy)):
        return None
    rows, cols = np.nonzero(in_bin)
    nearest = int(np.argmin((qx[in_bin] - median_qx) ** 2 + (qy[in_bin] - median_qy) ** 2))
    return {
        "qx": median_qx,
        "qy": median_qy,
        "q_star": q_star,
        "intensity": peak,
        "contrast": peak / baseline,
        "pixel_x": float(cols[nearest]),
        "pixel_y": float(rows[nearest]),
        "sector_deg": float(sector_deg),
    }


def _result_key(result: dict[str, float] | None) -> tuple[tuple[str, float], ...] | None:
    if result is None:
        return None
    return tuple(sorted((key, float(value)) for key, value in result.items()))


def _benchmark_data(size: int) -> tuple[np.ndarray, ...]:
    axis = np.linspace(-1.0, 1.0, size)
    qx, qy = np.meshgrid(axis, axis)
    q = np.hypot(qx, qy)
    valid = (q >= 0.49) & (q <= 0.51)
    image = 0.2 + 4.0 * np.exp(-0.5 * ((q - 0.50) / 0.03) ** 2)
    return qx, qy, q, image, valid


def _time_call(callable_: object, repeats: int) -> float:
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        callable_()
        samples.append(time.perf_counter() - started)
    return float(np.median(samples))


def run_benchmark(*, size: int, rounds: int, repeats: int) -> dict[str, object]:
    qx, qy, q, image, valid = _benchmark_data(size)
    hint = 0.50

    def legacy_sweep() -> tuple[dict[str, float] | None, ...]:
        return tuple(
            legacy_sector_first_order_peak(
                qx, qy, q, image, valid, sector_deg=sector, halfwidth_deg=7.5, hint=hint
            )
            for sector in SECTORS
        )

    def optimized_sweep() -> tuple[dict[str, float] | None, ...]:
        prepared = _prepare_sparse_first_order_samples(qx, qy, q, image, valid, hint=hint)
        return tuple(
            _sector_first_order_peak(
                qx,
                qy,
                q,
                image,
                valid,
                sector_deg=sector,
                halfwidth_deg=7.5,
                hint=hint,
                prepared=prepared,
            )
            for sector in SECTORS
        )

    legacy_reference = tuple(map(_result_key, legacy_sweep()))
    optimized_reference = tuple(map(_result_key, optimized_sweep()))
    if legacy_reference != optimized_reference:
        raise AssertionError("optimized sparse-ring sector results differ from the legacy oracle")

    legacy_times: list[float] = []
    optimized_times: list[float] = []
    for round_index in range(rounds):
        first, second = (legacy_sweep, optimized_sweep) if round_index % 2 == 0 else (optimized_sweep, legacy_sweep)
        first_time = _time_call(first, repeats)
        second_time = _time_call(second, repeats)
        if first is legacy_sweep:
            legacy_times.append(first_time)
            optimized_times.append(second_time)
        else:
            optimized_times.append(first_time)
            legacy_times.append(second_time)
    legacy_median = float(np.median(legacy_times))
    optimized_median = float(np.median(optimized_times))
    return {
        "schema_version": "sparse_ring_benchmark.v1",
        "scope": "16-sector sparse first-order peak kernel; point/arc bookkeeping excluded",
        "method_version": METHOD_VERSION,
        "scientific_acceptance": False,
        "python": sys.version,
        "implementation": platform.python_implementation(),
        "numpy": np.__version__,
        "shape": list(q.shape),
        "effective_pixels": int(np.count_nonzero(valid)),
        "hint": hint,
        "sector_count": len(SECTORS),
        "rounds": rounds,
        "repeats_per_round": repeats,
        "legacy_times_s": legacy_times,
        "optimized_times_s": optimized_times,
        "legacy_median_s": legacy_median,
        "optimized_median_s": optimized_median,
        "speedup": legacy_median / optimized_median,
        "input": "deterministic synthetic ring; no experimental acceptance claim",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/benchmarks/sparse_ring.json"),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.size < 32 or args.rounds < 1 or args.repeats < 1:
        parser.error("size must be >= 32 and rounds/repeats must be positive")
    if args.output.exists() and not args.force:
        parser.error(f"output exists; pass --force to overwrite: {args.output}")
    report = run_benchmark(size=args.size, rounds=args.rounds, repeats=args.repeats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), **report}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
