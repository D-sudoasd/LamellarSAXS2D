"""Reproducible development/holdout evidence without changing source images.

Examples: --benchmark --seeds 20260906,20260907 --output G:/arc-validation
or --project project.toml --frames 61,67,110 --split development --output ...
Holdout runs require a frozen recipe produced by a previous --freeze run.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys
import time
import traceback

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from butterfly_saxs.benchmark_arcs import (  # noqa: E402
    DEFAULT_CASE_IDS,
    GENERATOR_HASH,
    generate_arc_case,
)
from butterfly_saxs.geometry import build_geometry  # noqa: E402
from butterfly_saxs.pipeline import _jsonable, analyze_frame  # noqa: E402
from butterfly_saxs.project import load_project  # noqa: E402
from butterfly_saxs.io import combine_masks  # noqa: E402


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_hashes():
    return {str(p.relative_to(REPO)).replace("\\", "/"): digest(p)
            for p in sorted((REPO / "src" / "butterfly_saxs").glob("*.py"))}


def record_fit(name, data, qmap, mask, window, options, reference, multistart, analysis=None):
    start = time.perf_counter()
    analysis = {**(analysis or {}), "ridge_method": "butterfly_curvature", "q_window": window,
                "draw_axis_deg": reference + 90., "ellipse_multistart": multistart,
                "butterfly": {**((analysis or {}).get("butterfly") or {}), **options}}
    result = analyze_frame(data, qmap=qmap, mask=mask, config={"analysis": analysis},
                           full2d=False).butterfly
    if result is None:
        raise RuntimeError("shared application pipeline returned no butterfly result")
    fit = result["candidate_fit"]
    summary = {"id": name, "elapsed_s": time.perf_counter() - start,
               "execution_status": "complete", "solver_success": bool(fit.get("success")),
               "candidate": {key: fit.get(key) for key in ("a", "b", "axis_ratio", "theta_deg", "rmse", "condition")},
               "point_count": len(result.get("points", [])),
               "side_counts": dict(Counter(f"{p['branch_id']}:{p['side']}" for p in result.get("points", []) if p.get("accepted"))),
               "measurement_status": result["measurement_status"],
               "quantitative_parameters": result["quantitative_parameters"],
               "intervals": result["uncertainty"].get("intervals", {}),
               "resampling_success_fraction": result["uncertainty"].get("success_fraction"),
               "quality": result["quality"]}
    return result, summary


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--cases", default=",".join(DEFAULT_CASE_IDS))
    parser.add_argument("--seeds", default="20260906")
    parser.add_argument("--shape", type=int, default=96)
    parser.add_argument("--project", type=Path)
    parser.add_argument("--frames", default="110")
    parser.add_argument("--split", choices=("development", "holdout"), default="development")
    parser.add_argument("--resamples", type=int, default=0)
    parser.add_argument("--sensitivity", action="store_true")
    parser.add_argument("--multistart", type=int, default=3)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--frozen-recipe", type=Path)
    return parser


def main(argv=None):
    args = make_parser().parse_args(argv)
    if bool(args.benchmark) == bool(args.project):
        raise ValueError("choose exactly one of --benchmark and --project")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = output / "report.json"
    if report.exists():
        raise FileExistsError(f"use a new output directory: {report}")
    options = {"stage": "evaluate", "resamples": args.resamples,
               "seed": 20260906, "sensitivity": args.sensitivity}
    hashes = code_hashes()
    context = {"method_version": "butterfly-curvature-arcs-v1", "code_sha256": hashes,
               "generator_sha256": GENERATOR_HASH,
               "options": options, "multistart": args.multistart, "split": args.split,
               "calibrated_confidence": False,
               "versions": {name: version(name) for name in ("numpy", "scipy", "pyFAI", "fabio", "tifffile")},
               "python": sys.version}
    jobs = []
    if args.benchmark:
        context["kind"] = "independent_image_benchmark"
        context["shape"] = [args.shape, args.shape]
        for case_name in args.cases.split(","):
            if case_name not in DEFAULT_CASE_IDS:
                raise ValueError(f"use a canonical benchmark case ID: {case_name!r}")
            for seed in (int(x) for x in args.seeds.split(",")):
                jobs.append((f"{case_name}_{seed}", case_name, seed))
    else:
        import re
        args.project = args.project.resolve()
        project = load_project(args.project).resolve_paths(args.project.parent)
        config = project.analysis
        context["project_analysis"] = config
        context.update(kind="real_frames", project_sha256=digest(args.project),
                       poni_sha256=digest(project.poni_path),
                       mask_sha256=digest(config["mask"]) if config.get("mask") else None,
                       reference_axis_deg=float(config.get("draw_axis_deg", 90.)) - 90.,
                       q_window=list(config.get("q_window", [.1, .5])))
        sample_path = Path(project.input_paths[0])
        if re.search(r"\d+$", sample_path.stem) is None:
            raise ValueError("frame selection requires a numeric filename suffix")
        context["dataset_directory"] = str(sample_path.parent.resolve())
        context["filename_pattern"] = re.sub(r"\d+$", "<frame>", sample_path.stem) + sample_path.suffix
        frame_ids = [int(x) for x in args.frames.split(",")]
        for frame_id in frame_ids:
            stem = re.sub(r"\d+$", f"{frame_id:05d}", sample_path.stem)
            path = sample_path.with_name(stem + sample_path.suffix)
            if not path.is_file():
                raise FileNotFoundError(path)
            jobs.append((str(frame_id), path, None))
        context["frame_ids"] = frame_ids
    context["job_ids"] = [job[0] for job in jobs]
    if len(set(context["job_ids"])) != len(jobs):
        raise ValueError("duplicate case/seed or frame identity in requested jobs")
    if args.split == "holdout":
        if args.frozen_recipe is None:
            raise ValueError("holdout requires --frozen-recipe")
        frozen = json.loads(args.frozen_recipe.read_text(encoding="utf-8"))
        for key in ("kind", "shape", "dataset_directory", "filename_pattern", "project_analysis", "method_version",
                    "code_sha256", "generator_sha256", "versions", "python", "options", "multistart", "poni_sha256", "mask_sha256", "reference_axis_deg", "q_window"):
            if frozen.get(key) != context.get(key):
                raise ValueError(f"frozen recipe differs at {key}; holdout must be replanned, not relabelled")
        if not frozen.get("job_ids"):
            raise ValueError("frozen recipe has no auditable job identities")
        if set(frozen["job_ids"]) & set(context["job_ids"]):
            raise ValueError("development and holdout case/seed or frame IDs overlap")
        context["frozen_recipe_sha256"] = digest(args.frozen_recipe)
    if args.freeze:
        if args.split != "development":
            raise ValueError("freeze only a development recipe")
        (output / "frozen_recipe.json").write_text(json.dumps(context, indent=2), encoding="utf-8")
    summaries = []
    qmap = mask = None
    for name, source, seed in jobs:
        print(f"{args.split}: {name}", flush=True)
        before = None
        try:
            if args.benchmark:
                case = generate_arc_case(source, seed=seed, shape=(args.shape, args.shape))
                data, qmap, mask = case["image"], case["qmap"], case["mask"]
                window, reference = (.05, 1.1), 0.
                local_options = {**options, "seed": seed}
            else:
                import tifffile
                before = digest(source)
                data = tifffile.imread(source)
                if qmap is None:
                    qmap = build_geometry(data.shape, project.poni_path)
                    mask = ~combine_masks(data.shape, external_mask=config.get("mask"))
                if data.shape != qmap.q.shape:
                    raise ValueError("frame/geometry shape changed")
                window, reference = context["q_window"], context["reference_axis_deg"]
                local_options = options
            result, summary = record_fit(name, data, qmap, mask, window, local_options, reference, args.multistart,
                                         analysis=None if args.benchmark else config)
            if args.benchmark:
                truth = case["truth"]
                summary["truth"] = {key: truth.get(key) for key in ("a", "b", "axis_ratio", "branch_axes", "is_elliptic", "category")}
                summary["relative_errors"] = {
                    key: (summary["candidate"][key] - truth[key]) / truth[key]
                    for key in ("a", "b", "axis_ratio") if truth.get(key)
                    and summary["candidate"][key] is not None}
                summary["interval_contains_truth"] = {
                    key: interval[0] <= truth[key] <= interval[1]
                    for key, interval in summary["intervals"].items()
                    if truth.get(key) is not None and interval and all(x is not None for x in interval)}
                summary.update(
                    case_id=truth["case_id"],
                    seed=int(truth["seed"]),
                    shape=list(truth["shape"]),
                    case_identity=truth.get("case_identity", {
                        "case_id": truth["case_id"],
                        "seed": int(truth["seed"]),
                        "shape": list(truth["shape"]),
                    }),
                    case_identity_sha256=truth.get("case_identity_sha256"),
                    generator_sha256=truth["generator_sha256"],
                )
                if summary["generator_sha256"] != GENERATOR_HASH:
                    raise RuntimeError("benchmark truth generator hash does not match imported generator")
            else:
                summary.update(source=str(source), source_sha256_before=before,
                               source_sha256_after=digest(source))
                summary["source_unchanged"] = summary["source_sha256_before"] == summary["source_sha256_after"]
                if not summary["source_unchanged"]:
                    raise RuntimeError("source changed during read-only analysis")
            (output / f"{name}.json").write_text(json.dumps(_jsonable(result, array_summary=False),
                ensure_ascii=False, allow_nan=False), encoding="utf-8")
        except Exception as exc:
            summary = {"id": name, "execution_status": "error", "error": str(exc), "traceback": traceback.format_exc()}
        summaries.append(summary)
        with (output / "progress.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_jsonable(summary, array_summary=False), ensure_ascii=False, allow_nan=False) + "\n")
        print(json.dumps({key: summary.get(key) for key in ("id", "elapsed_s", "execution_status", "solver_success", "measurement_status", "relative_errors")}), flush=True)
    context["code_unchanged"] = code_hashes() == hashes
    if not args.benchmark:
        context["calibration_unchanged"] = (digest(project.poni_path) == context["poni_sha256"]
                                             and (digest(config["mask"]) if config.get("mask") else None) == context["mask_sha256"])
    report.write_text(json.dumps(_jsonable({"context": context, "frames": summaries}, array_summary=False),
                                ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return int(any(row["execution_status"] == "error" for row in summaries) or not context["code_unchanged"])


if __name__ == "__main__":
    raise SystemExit(main())
