"""Command line entry points for the shared LamellarSAXS2D pipeline."""

from __future__ import annotations

import argparse
import glob
import json
from collections.abc import Mapping
from pathlib import Path
import os
import sys
from typing import Any, Sequence

import numpy as np

from .pipeline import (
    PipelineError,
    analyze_frame,
    inspect_frame,
    launch_gui,
    run_project,
    synthetic_butterfly,
)
from .project import ProjectConfig, ProjectConfigError, load_project


def _shape(value: str) -> tuple[int, int]:
    pieces = [item for item in value.replace("×", "x").replace(",", "x").split("x") if item]
    if len(pieces) != 2:
        raise argparse.ArgumentTypeError("shape 应为 HxW，例如 256x256")
    try:
        result = tuple(int(item) for item in pieces)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("shape 应为 HxW，例如 256x256") from exc
    if any(item < 4 for item in result):
        raise argparse.ArgumentTypeError("shape 的两个尺寸必须至少为 4")
    return result  # type: ignore[return-value]


def _config(value: str | None) -> ProjectConfig | None:
    if not value:
        return None
    source = Path(value)
    return load_project(source).resolve_paths(source.parent)


def _print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def _write_synthetic(array: np.ndarray, qmap: dict[str, Any], output: str | os.PathLike[str], *, force: bool) -> Path:
    destination = Path(output)
    if destination.exists() and not force:
        raise FileExistsError(f"输出已存在，未覆盖：{destination}（需要 --force）")
    destination.parent.mkdir(parents=True, exist_ok=True)
    suffix = destination.suffix.lower()
    if suffix == ".npy":
        np.save(destination, array)
    elif suffix == ".npz":
        payload = {
            key: value for key, value in qmap.items() if isinstance(value, np.ndarray)
        }
        # Unit metadata is part of the numerical contract.  In particular,
        # the built-in synthetic grid is pixel-q and must never be reopened
        # later as though it were calibrated nm^-1 data.
        if qmap.get("q_unit") is not None:
            payload["q_unit"] = np.asarray(str(qmap["q_unit"]))
        np.savez_compressed(destination, data=array, **payload)
    elif suffix in {".tif", ".tiff"}:
        import tifffile

        tifffile.imwrite(destination, array)
    else:
        raise PipelineError("synthetic 输出格式支持 .npy、.npz、.tif/.tiff")
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bsaxs",
        description="LamellarSAXS2D：蝴蝶状二维 SAXS 花样的定量测量与椭圆精修",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    inspect_parser = sub.add_parser("inspect", help="检查图像、q 空间和基础观测量")
    inspect_parser.add_argument("input", nargs="?", help="CBF/EDF/TIF/NPY/NPZ/HDF5 图像")
    inspect_parser.add_argument("-i", "--input-file", dest="input_file", help="输入图像（input 的显式别名）")
    inspect_parser.add_argument("-c", "--config", help="TOML 项目配置")
    inspect_parser.add_argument("--poni", help="PONI 几何文件")
    inspect_parser.add_argument("--frame", type=int, help="多帧文件中的零基帧索引")
    inspect_parser.add_argument("--dataset", help="HDF5/NPZ 数据集或键")
    inspect_parser.add_argument("--mask", help="外部掩膜路径（True=无效像素）")
    inspect_parser.add_argument("--mask-frame", type=int, help="多帧掩膜中的零基帧索引")
    inspect_parser.add_argument("--mask-dataset", help="HDF5/NPZ 掩膜数据集或键")
    inspect_parser.add_argument("--valid-mask", dest="valid_mask", help="有效像素掩膜路径（True=有效像素）")
    inspect_parser.add_argument("-o", "--output", help="可选 JSON 输出路径")
    inspect_parser.add_argument("--force", action="store_true", help="允许覆盖已有输出")

    analyze_parser = sub.add_parser("analyze", help="分析单帧并拟合对称双椭圆")
    analyze_parser.add_argument("input", nargs="?", help="输入图像")
    analyze_parser.add_argument("-i", "--input-file", dest="input_file", help="输入图像（input 的显式别名）")
    analyze_parser.add_argument("-c", "--config", help="TOML 项目配置")
    analyze_parser.add_argument("--poni", help="PONI 几何文件")
    analyze_parser.add_argument("--frame", type=int, help="多帧文件中的零基帧索引")
    analyze_parser.add_argument("--dataset", help="HDF5/NPZ 数据集或键")
    analyze_parser.add_argument("--mask", help="外部掩膜路径（True=无效像素）")
    analyze_parser.add_argument("--mask-frame", type=int, help="多帧掩膜中的零基帧索引")
    analyze_parser.add_argument("--mask-dataset", help="HDF5/NPZ 掩膜数据集或键")
    analyze_parser.add_argument("--valid-mask", dest="valid_mask", help="有效像素掩膜路径（True=有效像素）")
    analyze_parser.add_argument("-o", "--output", help="JSON/NPZ 文件或输出目录")
    analyze_parser.add_argument("--full2d", action="store_true", help="调用可选的 full2d 精修模块")
    analyze_parser.add_argument("--force", action="store_true", help="允许覆盖已有输出")

    batch_parser = sub.add_parser("batch", help="批量分析原位序列")
    batch_parser.add_argument("inputs", nargs="*", help="输入图像或通配符")
    batch_parser.add_argument("-c", "--config", help="TOML 项目配置（可提供 inputs.files）")
    batch_parser.add_argument("--poni", help="PONI 几何文件")
    batch_parser.add_argument("--frame", type=int, help="多帧文件中的零基帧索引")
    batch_parser.add_argument("--dataset", help="HDF5/NPZ 数据集或键")
    batch_parser.add_argument("--mask", help="外部掩膜路径（True=无效像素）")
    batch_parser.add_argument("--mask-frame", type=int, help="多帧掩膜中的零基帧索引")
    batch_parser.add_argument("--mask-dataset", help="HDF5/NPZ 掩膜数据集或键")
    batch_parser.add_argument("--valid-mask", dest="valid_mask", help="有效像素掩膜路径（True=有效像素）")
    batch_parser.add_argument("-o", "--output", help="输出目录")
    batch_parser.add_argument("--full2d", action="store_true", help="调用可选的 full2d 精修模块")
    batch_parser.add_argument("--mode", choices=("independent", "warm_start"), help="序列拟合模式")
    batch_parser.add_argument("--manifest", help="JSON/CSV 帧清单（含 time/frame_id 等元数据）")
    batch_parser.add_argument("--checkpoint", help="批量检查点 JSON 路径")
    batch_parser.add_argument("--resume", action="store_true", help="从已有检查点恢复")
    batch_parser.add_argument("--force", action="store_true", help="允许覆盖已有输出")

    synthetic_parser = sub.add_parser("synthetic", help="生成可重复的蝴蝶状二维测试花样")
    synthetic_parser.add_argument("-o", "--output", help=".npy/.npz/.tif 输出路径；不提供则只打印摘要")
    synthetic_parser.add_argument("--shape", type=_shape, default=(128, 128), help="图像尺寸 HxW")
    synthetic_parser.add_argument("--q0", type=float, default=28.0, help="椭圆特征 q 半径")
    synthetic_parser.add_argument("--width", type=float, default=2.0, help="峰脊宽度")
    synthetic_parser.add_argument("--ellipticity", type=float, default=2.0, help="椭圆长短轴比")
    synthetic_parser.add_argument("--angle", type=float, default=28.0, help="对称椭圆角度（度）")
    synthetic_parser.add_argument("--noise", type=float, default=0.0, help="高斯噪声标准差")
    synthetic_parser.add_argument("--seed", type=int, default=0, help="随机种子")
    synthetic_parser.add_argument("--force", action="store_true", help="允许覆盖已有输出")

    gui_parser = sub.add_parser("gui", help="打开与 CLI 共用 pipeline seam 的交互界面")
    gui_parser.add_argument("input", nargs="?", help="可选输入图像")
    gui_parser.add_argument("-c", "--config", help="可选 TOML 项目配置")
    gui_parser.add_argument("--poni", help="PONI 几何文件")

    project_parser = sub.add_parser("project", help="执行 TOML 项目配置")
    project_parser.add_argument("config", help="项目 TOML 文件")
    project_parser.add_argument("--force", action="store_true", help="允许覆盖已有输出")

    preflight_parser = sub.add_parser(
        "preflight", help="只读检查真实数据包、几何、掩膜、单位与清单"
    )
    preflight_parser.add_argument("package", help="真实数据包根目录")
    preflight_parser.add_argument("--manifest", help="CSV/JSON/TOML 帧清单")
    preflight_parser.add_argument("--poni", help="PONI 几何文件")
    preflight_parser.add_argument("--mask", help="外部掩膜文件")
    preflight_parser.add_argument("--context", help="project_context.yaml/yml")
    preflight_parser.add_argument("--image-glob", help="未使用清单时的图像通配符")
    preflight_parser.add_argument("--frame", type=int, help="图像多帧选择器")
    preflight_parser.add_argument("--dataset", help="图像 HDF5/NPZ 数据集或键")
    preflight_parser.add_argument("--mask-frame", type=int, help="掩膜多帧选择器")
    preflight_parser.add_argument("--mask-dataset", help="掩膜 HDF5/NPZ 数据集或键")
    preflight_parser.add_argument(
        "--q-window", type=float, nargs=2, metavar=("Q_MIN", "Q_MAX")
    )
    preflight_parser.add_argument(
        "--mask-convention",
        choices=("0_valid_1_invalid", "1_valid_0_invalid"),
    )
    preflight_parser.add_argument("--correction-state")
    preflight_parser.add_argument("--uncertainty-state")
    preflight_parser.add_argument("-o", "--output", help="预检证据输出目录")
    preflight_parser.add_argument("--force", action="store_true", help="允许覆盖已有预检输出")

    benchmark_parser = sub.add_parser(
        "benchmark", help="生成 P3 的 T1 同模型或 T2 独立 FFT 基准证据"
    )
    benchmark_parser.add_argument(
        "--suite", choices=("t1", "t2", "all"), default="all", help="要生成的基准套件"
    )
    benchmark_parser.add_argument("-o", "--output", required=True, help="新的证据输出目录")
    benchmark_parser.add_argument("--shape", type=_shape, help="可选统一图像尺寸 HxW")
    benchmark_parser.add_argument("--seed", type=int, help="可选统一起始随机种子")
    benchmark_parser.add_argument("--force", action="store_true", help="仅覆盖本命令的目标文件")

    annotation_parser = sub.add_parser(
        "annotation-pack", help="从 R0 清单生成 8 帧只读盲标包，不运行拟合"
    )
    annotation_parser.add_argument("package", help="真实数据包根目录")
    annotation_parser.add_argument("--rt-manifest", required=True, help="室温参考帧清单")
    annotation_parser.add_argument("--hold-manifest", required=True, help="保温序列帧清单")
    annotation_parser.add_argument("--preflight", help="可选原始强度预检 JSON，用于困难帧排序")
    annotation_parser.add_argument("--poni", help="PONI 文件，只记录来源而不参与选择")
    annotation_parser.add_argument("--mask", help="mask 文件，只记录来源而不参与选择")
    annotation_parser.add_argument("-o", "--output", required=True, help="新的盲标包输出目录")

    p3_parser = sub.add_parser("p3-status", help="只读评估 P3 Go/No-Go 证据门")
    p3_parser.add_argument("--t1-manifest", required=True, help="T1 truth_manifest.json")
    p3_parser.add_argument("--t2-manifest", required=True, help="T2 truth_manifest.json")
    p3_parser.add_argument("--annotation-status", required=True, help="annotation_status.json")
    p3_parser.add_argument("--thresholds", required=True, help="阈值 JSON（draft 或 frozen）")
    p3_parser.add_argument("-o", "--output", help="可选门禁报告 JSON")
    p3_parser.add_argument("--force", action="store_true", help="允许覆盖已有门禁报告")
    return parser


def _pick_input(args: argparse.Namespace, config: ProjectConfig | None) -> Any:
    value = args.input_file or args.input
    if value:
        return value
    if config and config.input_paths:
        return config.input_paths[0]
    raise PipelineError("请提供输入图像，或在 TOML 中填写 inputs.files")


def _handle_inspect(args: argparse.Namespace) -> int:
    config = _config(args.config)
    source = _pick_input(args, config)
    report = inspect_frame(
        source,
        poni=args.poni or (config.poni_path if config else None),
        config=config,
        frame=args.frame,
        dataset=args.dataset,
        mask=args.mask,
        mask_frame=args.mask_frame,
        mask_dataset=args.mask_dataset,
        valid_mask=args.valid_mask,
    )
    if args.output:
        destination = Path(args.output)
        if destination.exists() and not args.force:
            raise FileExistsError(f"输出已存在，未覆盖：{destination}（需要 --force）")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    _print_json(report)
    return 0


def _handle_analyze(args: argparse.Namespace) -> int:
    config = _config(args.config)
    source = _pick_input(args, config)
    result = analyze_frame(
        source,
        poni=args.poni or (config.poni_path if config else None),
        config=config,
        full2d=args.full2d or (config.full2d if config else False),
        frame=args.frame,
        dataset=args.dataset,
        mask=args.mask,
        mask_frame=args.mask_frame,
        mask_dataset=args.mask_dataset,
        valid_mask=args.valid_mask,
        output=args.output,
        force=args.force,
    )
    report = result.to_mapping()
    _print_json(report)
    from .batch import _quality_failure_reason

    return 1 if _quality_failure_reason(report) is not None else 0


def _handle_batch(args: argparse.Namespace) -> int:
    from . import batch as batch_module
    from . import export as export_module

    config = _config(args.config)
    inputs = list(args.inputs)
    if not inputs and config:
        inputs = list(config.input_paths)
    if not inputs:
        raise PipelineError("batch 没有输入；请提供路径或 TOML 的 inputs.files")
    # ``argparse`` receives quoted PowerShell globs literally.  Expand them
    # before handing a list to run_batch (its single-string form already has
    # this convenience, but a CLI naturally supplies a list).
    expanded_inputs: list[str] = []
    for value in inputs:
        text = os.fspath(value)
        if any(char in text for char in "*?[]"):
            expanded_inputs.extend(glob.glob(text))
        elif Path(text).is_dir():
            expanded_inputs.extend(os.fspath(item) for item in Path(text).iterdir() if item.is_file())
        else:
            expanded_inputs.append(text)
    inputs = expanded_inputs
    if not inputs:
        raise PipelineError("batch 输入通配符没有匹配任何文件")
    analysis = config.analysis if config is not None else {}
    mode = args.mode or str(analysis.get("batch_mode", analysis.get("mode", "independent")))
    manifest = args.manifest or analysis.get("manifest")
    checkpoint = args.checkpoint or analysis.get("checkpoint")
    resume = bool(args.resume or analysis.get("resume", False))
    output_dir = Path(args.output or (config.output_dir if config else "results"))
    if output_dir.exists() and any(output_dir.iterdir()) and not args.force:
        raise FileExistsError(f"输出目录已有内容，未覆盖：{output_dir}（需要 --force）")

    poni = args.poni or (config.poni_path if config else None)
    full2d = args.full2d or (config.full2d if config else False)

    batch_inputs: Any = inputs
    batch_manifest: Any = manifest
    batch_config: Any = config
    if args.frame is not None or args.dataset is not None:
        # Resolve the manifest before applying CLI overrides so the selected
        # frame/dataset become part of the FrameRef identity and input hash.
        resolved_refs = batch_module.build_frame_refs(inputs, manifest=manifest)
        resolved_with_cli_selectors = []
        for ref in resolved_refs:
            resolved_with_cli_selectors.append(
                batch_module.FrameRef(
                    ref.path,
                    time=ref.time,
                    frame_id=ref.frame_id,
                    metadata=ref.metadata,
                    order=ref.order,
                    source=ref.source,
                    dataset=args.dataset if args.dataset is not None else ref.dataset,
                    frame=args.frame if args.frame is not None else ref.frame,
                )
            )
        batch_inputs = resolved_with_cli_selectors
        # Feed the resolved records back as the manifest so an existing
        # manifest's acquisition order/time remains authoritative after the
        # CLI selector override.
        batch_manifest = [
            {**ref.to_dict(), "order": index}
            for index, ref in enumerate(resolved_with_cli_selectors)
        ]
        selector_config = {
            name: value
            for name, value in (("frame", args.frame), ("dataset", args.dataset))
            if value is not None
        }
        if config is None:
            batch_config = {"analysis": selector_config}
        else:
            batch_config = ProjectConfig(
                input_paths=config.input_paths,
                poni_path=config.poni_path,
                output_dir=config.output_dir,
                q_unit=config.q_unit,
                full2d=config.full2d,
                analysis={**config.analysis, **selector_config},
                export=config.export,
                metadata=config.metadata,
            )

    def analyze_for_batch(frame_ref: Any, initial_parameters: Any = None, config: Any = None) -> Any:
        source = getattr(frame_ref, "path", frame_ref)
        selected_frame = args.frame
        if selected_frame is None:
            selected_frame = getattr(frame_ref, "frame_selector", None)
        selected_dataset = args.dataset
        if selected_dataset is None:
            selected_dataset = getattr(frame_ref, "dataset", None)
            if selected_dataset is None:
                selected_dataset = getattr(frame_ref, "dataset_id", None) or None
        return analyze_frame(
            source,
            poni=poni,
            config=config,
            full2d=full2d,
            initial_parameters=initial_parameters,
            frame=selected_frame,
            dataset=selected_dataset,
            mask=args.mask,
            mask_frame=args.mask_frame,
            mask_dataset=args.mask_dataset,
            valid_mask=args.valid_mask,
        )

    run = batch_module.run_batch(
        batch_inputs,
        analyze_for_batch,
        mode=mode,
        config=batch_config,
        manifest=batch_manifest,
        checkpoint=checkpoint,
        resume=resume,
    )
    exports = export_module.export_batch(
        run,
        output_dir,
        provenance={"command": "bsaxs batch", "full2d": full2d},
        force=args.force,
    )
    compact_records = []
    for item in run.frame_results:
        record = item.to_record()
        if hasattr(item.result, "to_mapping"):
            # Avoid printing/duplicating full detector arrays at the CLI
            # boundary; lossless arrays remain in results.npz and the object
            # returned by the Python API.
            record["result"] = item.result.to_mapping()
        compact_records.append(record)
    report = {
        "mode": run.mode,
        "input_hash": run.input_hash,
        "config_hash": run.config_hash,
        "frames": compact_records,
        "n_frames": len(run.frame_results),
        "n_success": len(run.successful),
        "n_failed": len(run.failures),
        "checkpoint": str(run.checkpoint) if run.checkpoint is not None else None,
        "outputs": {key: str(path) for key, path in exports.items()},
    }
    _print_json(report)
    # Partial exports remain available for inspection, while automation gets
    # an honest non-zero status when any frame failed its load/fit quality gate.
    return 1 if run.failures else 0


def _handle_synthetic(args: argparse.Namespace) -> int:
    array, qmap = synthetic_butterfly(
        args.shape,
        q0=args.q0,
        width=args.width,
        ellipticity=args.ellipticity,
        angle_deg=args.angle,
        noise=args.noise,
        seed=args.seed,
        return_qmap=True,
    )
    destination = _write_synthetic(array, qmap, args.output, force=args.force) if args.output else None
    report = {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "output": os.fspath(destination) if destination else None,
        "intensity_min": float(np.min(array)),
        "intensity_max": float(np.max(array)),
        "seed": args.seed,
        "flags": {
            "empirical_model_only": True,
            "mechanism_under_determined": True,
            "forward_simulation_only": True,
            "nonunique_inverse_problem": True,
        },
    }
    _print_json(report)
    return 0


def _handle_gui(args: argparse.Namespace) -> int:
    config = _config(args.config)
    return int(
        launch_gui(
            input_path=args.input,
            poni=args.poni or (config.poni_path if config else None),
            config=config,
            config_path=args.config,
        )
        or 0
    )


def _handle_preflight(args: argparse.Namespace) -> int:
    from .service import ButterflyAnalysisService

    report = ButterflyAnalysisService().preflight(
        args.package,
        manifest=args.manifest,
        poni=args.poni,
        mask=args.mask,
        context=args.context,
        image_glob=args.image_glob,
        frame=args.frame,
        dataset=args.dataset,
        mask_frame=args.mask_frame,
        mask_dataset=args.mask_dataset,
        q_window=args.q_window,
        mask_convention=args.mask_convention,
        correction_state=args.correction_state,
        uncertainty_state=args.uncertainty_state,
        output=args.output,
        force=args.force,
    )
    _print_json(report)
    status = report.get("status")
    if isinstance(status, Mapping):
        return int(status["exit_code"])
    return 0 if status == "green" else 1


def _handle_benchmark(args: argparse.Namespace) -> int:
    from . import benchmark_t1, benchmark_t2

    output = Path(args.output)
    manifests: dict[str, str] = {}
    if args.suite in {"t1", "all"}:
        destination = output / "t1" if args.suite == "all" else output
        manifest = benchmark_t1.write_evidence_directory(
            destination,
            shape=args.shape,
            seed=args.seed,
            force=args.force,
        )
        manifests["t1"] = manifest.as_posix()
    if args.suite in {"t2", "all"}:
        destination = output / "t2" if args.suite == "all" else output
        manifest = benchmark_t2.write_evidence_directory(
            destination,
            shape=args.shape or benchmark_t2.DEFAULT_SHAPE,
            seed=args.seed,
            force=args.force,
        )
        manifests["t2"] = manifest.as_posix()
    _print_json({"suite": args.suite, "manifests": manifests})
    return 0


def _handle_annotation_pack(args: argparse.Namespace) -> int:
    from .annotation_pack import build_annotation_pack

    result = build_annotation_pack(
        args.package,
        args.rt_manifest,
        args.hold_manifest,
        args.output,
        preflight_json=args.preflight,
        poni=args.poni,
        mask=args.mask,
    )
    _print_json(
        {
            "output_directory": Path(result["output_directory"]).as_posix(),
            "candidate_count": result["candidate_count"],
            "status": result["status"]["status"],
            "human_consensus": result["status"]["human_consensus"],
        }
    )
    return 0


def _handle_p3_status(args: argparse.Namespace) -> int:
    from .p3_gate import evaluate_p3_gate, write_p3_gate_report

    if args.output:
        report = write_p3_gate_report(
            args.output,
            args.t1_manifest,
            args.t2_manifest,
            args.annotation_status,
            args.thresholds,
            force=args.force,
        )
    else:
        report = evaluate_p3_gate(
            args.t1_manifest,
            args.t2_manifest,
            args.annotation_status,
            args.thresholds,
        )
    _print_json(report)
    return int(report["exit_code"])


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "inspect":
            return _handle_inspect(args)
        if args.command == "analyze":
            return _handle_analyze(args)
        if args.command == "batch":
            return _handle_batch(args)
        if args.command == "synthetic":
            return _handle_synthetic(args)
        if args.command == "gui":
            return _handle_gui(args)
        if args.command == "project":
            results = run_project(args.config, force=args.force)
            _print_json([result.to_mapping() for result in results])
            return 0
        if args.command == "preflight":
            return _handle_preflight(args)
        if args.command == "benchmark":
            return _handle_benchmark(args)
        if args.command == "annotation-pack":
            return _handle_annotation_pack(args)
        if args.command == "p3-status":
            return _handle_p3_status(args)
        parser.error(f"未知命令：{args.command}")
    except (PipelineError, ProjectConfigError, FileExistsError, OSError, ValueError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2
    return 2


__all__ = ["build_parser", "main"]
