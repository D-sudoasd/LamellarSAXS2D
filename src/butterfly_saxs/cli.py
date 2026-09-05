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
    run_project_bounded,
    synthetic_butterfly,
)
from .project import ProjectConfig, ProjectConfigError, load_project
from .settings import deep_merge_mapping
from .path_utils import filter_supported_image_paths


_DEFAULT_LEGACY_PROJECT_RUNNER = run_project


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


def _analysis_overrides(args: argparse.Namespace) -> dict[str, Any]:
    """Collect explicit CLI refinement controls without overriding TOML defaults."""

    mapping: dict[str, Any] = {}
    for argument, key in (
        ("q_min", "q_min"),
        ("q_max", "q_max"),
        ("q_window", "q_window"),
        ("ridge_method", "ridge_method"),
        ("ridge_snr_threshold", "ridge_snr_threshold"),
        ("ridge_min_peak_fraction", "ridge_min_peak_fraction"),
        ("ridge_min_coverage", "ridge_min_coverage"),
        ("full2d_multistart", "full2d_multistart"),
        ("mask", "mask"),
        ("valid_mask", "valid_mask"),
        ("mask_frame", "mask_frame"),
        ("mask_dataset", "mask_dataset"),
    ):
        value = getattr(args, argument, None)
        if value is not None:
            mapping[key] = value
    ellipse: dict[str, Any] = {}
    for argument, key in (
        ("ellipse_preset", "preset"),
        ("ellipse_ratio_min", "axis_ratio_min"),
        ("ellipse_ratio_max", "axis_ratio_max"),
        ("ellipse_a", "a"),
        ("ellipse_b", "b"),
        ("ellipse_ratio", "axis_ratio"),
        ("ellipse_a_min", "a_min"),
        ("ellipse_a_max", "a_max"),
        ("ellipse_b_min", "b_min"),
        ("ellipse_b_max", "b_max"),
        ("ellipse_angle_min", "theta_min_deg"),
        ("ellipse_angle_max", "theta_max_deg"),
        ("ellipse_fixed_center", "fixed_center"),
        ("ellipse_fixed_a", "fixed_a"),
        ("ellipse_fixed_ratio", "fixed_axis_ratio"),
        ("ellipse_center_qx", "center_qx"),
        ("ellipse_center_qy", "center_qy"),
        ("ellipse_fixed_angle", "fixed_angle"),
        ("ellipse_angle_deg", "angle_deg"),
        ("ellipse_residual", "residual"),
        ("ellipse_multistart", "multistart"),
    ):
        value = getattr(args, argument, None)
        if value is not None:
            ellipse[key] = value
    if ellipse:
        mapping["ellipse"] = ellipse
    return mapping


def _with_analysis(config: ProjectConfig | None, overrides: Mapping[str, Any]) -> Any:
    """Return a config carrying explicit CLI overrides with TOML precedence."""

    if not overrides:
        return config
    if config is None:
        return ProjectConfig(analysis=dict(overrides))
    return ProjectConfig(
        input_paths=config.input_paths,
        poni_path=config.poni_path,
        output_dir=config.output_dir,
        q_unit=config.q_unit,
        full2d=config.full2d,
        analysis=deep_merge_mapping(config.analysis, overrides),
        export=config.export,
        metadata=config.metadata,
    )


def _add_refinement_options(parser: argparse.ArgumentParser) -> None:
    """Add the common flat-ellipse/ridge controls to a CLI subcommand."""

    parser.add_argument(
        "--q-window",
        type=float,
        nargs=2,
        metavar=("Q_MIN", "Q_MAX"),
        help="analysis q window in the active q-map unit",
    )
    parser.add_argument("--q-min", type=float, help="analysis q lower bound")
    parser.add_argument("--q-max", type=float, help="analysis q upper bound")
    parser.add_argument(
        "--ridge-method",
        choices=("radial_peak", "azimuthal_peak", "surface_curvature"),
        help="ridge localization method (radial_peak, azimuthal_peak, or surface_curvature)",
    )
    parser.add_argument("--ridge-snr-threshold", type=float, help="minimum ridge SNR")
    parser.add_argument(
        "--ridge-min-peak-fraction",
        type=float,
        help="minimum valid support fraction for a ridge candidate [0, 1]",
    )
    parser.add_argument(
        "--ridge-min-coverage",
        type=float,
        help="minimum detector coverage for a ridge candidate [0, 1]",
    )
    parser.add_argument(
        "--ellipse-preset",
        choices=("standard", "flat_ellipse", "very_flat_ellipse"),
        help="constrained measured-ellipse preset",
    )
    parser.add_argument("--ellipse-ratio-min", type=float, help="measured ellipse b/a lower bound")
    parser.add_argument("--ellipse-ratio-max", type=float, help="measured ellipse b/a upper bound")
    parser.add_argument("--ellipse-a", type=float, help="measured ellipse a starting value")
    parser.add_argument("--ellipse-b", type=float, help="measured ellipse b starting value")
    parser.add_argument("--ellipse-ratio", type=float, help="measured ellipse b/a starting value")
    parser.add_argument("--ellipse-a-min", type=float, help="measured ellipse a lower bound")
    parser.add_argument("--ellipse-a-max", type=float, help="measured ellipse a upper bound")
    parser.add_argument("--ellipse-b-min", type=float, help="derived measured ellipse b lower bound")
    parser.add_argument("--ellipse-b-max", type=float, help="derived measured ellipse b upper bound")
    parser.add_argument("--ellipse-angle-min", type=float, help="measured ellipse angle lower bound (deg)")
    parser.add_argument("--ellipse-angle-max", type=float, help="measured ellipse angle upper bound (deg)")
    parser.add_argument(
        "--ellipse-fixed-center",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="fix measured ellipse centre at ellipse-center-qx/qy",
    )
    parser.add_argument(
        "--ellipse-fixed-a",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="fix measured ellipse a at its explicit value",
    )
    parser.add_argument(
        "--ellipse-fixed-ratio",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="fix measured ellipse b/a at its explicit value",
    )
    parser.add_argument("--ellipse-center-qx", type=float, help="measured ellipse centre qx")
    parser.add_argument("--ellipse-center-qy", type=float, help="measured ellipse centre qy")
    parser.add_argument(
        "--ellipse-fixed-angle",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="fix the measured ellipse angle at ellipse-angle-deg",
    )
    parser.add_argument("--ellipse-angle-deg", type=float, help="measured ellipse angle initial value (deg)")
    parser.add_argument(
        "--ellipse-residual",
        choices=("sampson", "geometric"),
        help="residual used by measured ellipse fit",
    )
    parser.add_argument("--ellipse-multistart", type=int, help="deterministic measured ellipse starts")
    parser.add_argument("--full2d-multistart", type=int, help="deterministic full2d starts")


def _print_json(value: Any) -> None:
    # Console encodings on Windows are often GBK/CP936, which cannot encode
    # every scientific unit symbol (for example ``Å``).  Escaping non-ASCII
    # characters keeps stdout valid JSON on every terminal; exported files
    # remain human-readable UTF-8 through their dedicated writers.
    print(json.dumps(value, ensure_ascii=True, indent=2, allow_nan=False))


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
    _add_refinement_options(analyze_parser)

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
    batch_parser.add_argument(
        "--stream",
        action="store_true",
        help="逐帧写出参数/脊线和 NPZ，释放已处理帧的 detector 数组",
    )
    batch_parser.add_argument("--series", help="只处理 manifest 中指定的 series/group")
    batch_parser.add_argument("--start", type=int, help="选中有序序列的起始位置（含）")
    batch_parser.add_argument("--stop", type=int, help="选中有序序列的结束位置（含）")
    batch_parser.add_argument("--stride", type=int, default=None, help="有序序列步长")
    batch_parser.add_argument(
        "--range",
        dest="frame_range",
        help="序列范围 START:STOP[:STEP]，STOP 包含在内",
    )
    _add_refinement_options(batch_parser)

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
    project_parser.add_argument(
        "--legacy-json",
        action="store_true",
        help="以旧版 per-frame JSON 列表输出；默认输出带 schema_version 的批处理 envelope",
    )

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

    p4_parser = sub.add_parser(
        "p4-evaluate", help="运行 P4 ridge/lobe/双椭圆工程证据（不冒充科学验收）"
    )
    p4_parser.add_argument("--t1-manifest", required=True, help="T1 truth_manifest.json")
    p4_parser.add_argument("--t2-manifest", required=True, help="T2 truth_manifest.json")
    p4_parser.add_argument("--thresholds", required=True, help="draft/frozen 阈值 JSON")
    p4_parser.add_argument("-o", "--output", required=True, help="新的 P4 证据输出目录")
    p4_parser.add_argument("--r0-package", help="可选 R0 原始数据包根目录")
    p4_parser.add_argument("--r0-manifest", help="可选固定 8 帧 annotation_manifest.csv")
    p4_parser.add_argument("--poni", help="R0 使用的 PONI 文件")
    p4_parser.add_argument("--mask", help="R0 使用的外部 mask")
    p4_parser.add_argument(
        "--ridge-method",
        choices=("radial_peak", "surface_curvature"),
        default="radial_peak",
    )
    p4_parser.add_argument(
        "--skip-sensitivity", action="store_true", help="跳过单个 T1 方法敏感性对照"
    )
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
    config = _with_analysis(config, _analysis_overrides(args))
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
    config = _with_analysis(config, _analysis_overrides(args))
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
            expanded_inputs.extend(
                os.fspath(item) for item in filter_supported_image_paths(glob.glob(text))
            )
        elif Path(text).is_dir():
            expanded_inputs.extend(
                os.fspath(item)
                for item in filter_supported_image_paths(Path(text).iterdir())
            )
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
    series = args.series if args.series is not None else analysis.get("series")
    start = args.start if args.start is not None else analysis.get("start")
    stop = args.stop if args.stop is not None else analysis.get("stop")
    explicit_sequence = (
        args.start is not None
        or args.stop is not None
        or args.stride is not None
        or args.frame_range is not None
    )
    if explicit_sequence:
        start = args.start
        stop = args.stop
        stride = args.stride if args.stride is not None else 1
        frame_range = args.frame_range
    else:
        start = analysis.get("start")
        stop = analysis.get("stop")
        stride = analysis.get("stride", 1)
        frame_range = (
            args.frame_range
            if args.frame_range is not None
            else analysis.get("frame_range")
        )
    output_dir = Path(args.output or (config.output_dir if config else "results"))
    # A resumed run has already validated input/config/mode hashes before its
    # exports are written.  It may therefore refresh its own known bundle
    # targets; a fresh run still refuses a non-empty directory by default.
    if output_dir.exists() and any(output_dir.iterdir()) and not args.force and not resume:
        raise FileExistsError(f"输出目录已有内容，未覆盖：{output_dir}（需要 --force）")

    poni = args.poni or (config.poni_path if config else None)
    full2d = args.full2d or (config.full2d if config else False)

    # CLI path overrides are part of the batch configuration identity.  This
    # binds PONI/mask content fingerprints in ``config_fingerprint`` so a
    # resume cannot silently reuse results after calibration or mask changes.
    path_analysis = dict(config.analysis) if isinstance(config, ProjectConfig) else {}
    for name, value in (
        ("mask", args.mask),
        ("valid_mask", args.valid_mask),
        ("mask_frame", args.mask_frame),
        ("mask_dataset", args.mask_dataset),
    ):
        if value is not None:
            path_analysis[name] = value
    if explicit_sequence and frame_range is None:
        path_analysis.pop("frame_range", None)
    for name, value in (
        ("series", args.series),
        ("start", args.start),
        ("stop", args.stop),
        ("stride", stride if explicit_sequence else None),
        ("frame_range", frame_range if explicit_sequence else None),
    ):
        if value is not None:
            path_analysis[name] = value
    path_analysis["stage"] = "full2d" if full2d else "geometry"
    if isinstance(config, ProjectConfig) and (
        poni != config.poni_path or path_analysis != config.analysis
    ):
        batch_config = ProjectConfig(
            input_paths=config.input_paths,
            poni_path=poni,
            output_dir=config.output_dir,
            q_unit=config.q_unit,
            full2d=full2d,
            analysis=path_analysis,
            export=config.export,
            metadata=config.metadata,
        )
    elif config is None and (poni is not None or path_analysis):
        batch_config = ProjectConfig(poni_path=poni, analysis=path_analysis)
    else:
        batch_config = config

    batch_inputs: Any = inputs
    batch_manifest: Any = manifest
    if args.frame is not None or args.dataset is not None:
        # Resolve the manifest before applying CLI overrides so the selected
        # frame/dataset become part of the FrameRef identity and input hash.
        resolved_refs = batch_module.build_frame_refs(
            inputs,
            manifest=manifest,
            allow_mixed_series=series is not None,
        )
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
                input_paths=batch_config.input_paths,
                poni_path=batch_config.poni_path,
                output_dir=config.output_dir,
                q_unit=config.q_unit,
                full2d=full2d,
                analysis=deep_merge_mapping(batch_config.analysis, selector_config),
                export=batch_config.export,
                metadata=batch_config.metadata,
            )

    geometry_cache: dict[Any, Any] = {}

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
            geometry_cache=geometry_cache,
        )

    stream_writer = None
    if args.stream:
        stream_writer = export_module.StreamingBatchExporter(
            output_dir,
            provenance={"command": "bsaxs batch", "full2d": full2d, "stream": True},
            force=bool(args.force or resume),
            resume=resume,
        )
    try:
        run = batch_module.run_batch(
            batch_inputs,
            analyze_for_batch,
            mode=mode,
            config=batch_config,
            manifest=batch_manifest,
            checkpoint=checkpoint,
            resume=resume,
            series=series,
            start=start,
            stop=stop,
            stride=stride,
            frame_range=frame_range,
            progress=lambda update: print(
                f"batch progress {update.get('completed', 0)}/{update.get('total', 0)}",
                file=sys.stderr,
                flush=True,
            ),
            result_sink=None if stream_writer is None else stream_writer.write,
            retain_results=stream_writer is None,
        )
        exports = (
            stream_writer.finalize(run)
            if stream_writer is not None
            else export_module.export_batch(
                run,
                output_dir,
                provenance={"command": "bsaxs batch", "full2d": full2d},
                force=bool(args.force or resume),
            )
        )
    except Exception:
        if stream_writer is not None:
            stream_writer.abort()
        raise
    compact_records = []
    for item in run.frame_results:
        record = item.to_record()
        if hasattr(item.result, "to_mapping"):
            # Avoid printing/duplicating full detector arrays at the CLI
            # boundary; lossless arrays remain in results.npz and the object
            # returned by the Python API.
            result_mapping = item.result.to_mapping()
            if args.stream and isinstance(result_mapping, Mapping):
                # Stream mode already writes detector/profile arrays to NPZ;
                # keep stdout bounded to longitudinal diagnostics and rows.
                result_mapping = {
                    key: result_mapping.get(key)
                    for key in (
                        "metadata", "flags", "parameters", "ridges", "ridge_points",
                        "ellipse_fit", "lobe_radial_profiles", "lobe_radial_peaks",
                        "full2d", "analysis", "analysis_domain", "valid_mask",
                    )
                    if key in result_mapping
                }
            record["result"] = result_mapping
        elif args.stream and isinstance(item.result, Mapping):
            record["result"] = {
                key: item.result.get(key)
                for key in (
                    "metadata", "flags", "parameters", "ridges", "ridge_points",
                    "ellipse_fit", "lobe_radial_profiles", "lobe_radial_peaks",
                    "full2d", "analysis", "analysis_domain",
                )
                if key in item.result
            }
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
        "selection": run.selection,
        "cancelled": run.cancelled,
        "elapsed_s": run.elapsed_s,
        "processed_count": run.processed_count,
        "total_count": run.total_count,
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


def _handle_p4_evaluate(args: argparse.Namespace) -> int:
    from .p4_validation import run_p4_engineering

    report = run_p4_engineering(
        t1_manifest=args.t1_manifest,
        t2_manifest=args.t2_manifest,
        thresholds=args.thresholds,
        output=args.output,
        r0_package=args.r0_package,
        r0_manifest=args.r0_manifest,
        poni=args.poni,
        mask=args.mask,
        ridge_method=args.ridge_method,
        run_sensitivity=not args.skip_sensitivity,
        progress=lambda message: print(message, file=sys.stderr, flush=True),
    )
    _print_json(
        {
            "stage": report["stage"],
            "engineering_status": report["engineering_status"],
            "scientific_status": report["scientific_status"],
            "p4_go_no_go": report["p4_go_no_go"],
            "outputs": report["outputs"],
        }
    )
    return 0 if report["p4_go_no_go"] == "GO" else 1


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
            # The CLI uses the bounded runner explicitly.  Keep the legacy
            # symbol as an injection seam for older callers/tests that patch
            # ``cli.run_project``.
            project_runner = (
                run_project
                if run_project is not _DEFAULT_LEGACY_PROJECT_RUNNER
                else run_project_bounded
            )
            run = project_runner(args.config, force=args.force)
            compact_records = []
            for item in run.frame_results:
                record = item.to_record()
                if hasattr(item.result, "to_mapping"):
                    record["result"] = item.result.to_mapping()
                compact_records.append(record)
            if args.legacy_json:
                _print_json(compact_records)
            else:
                _print_json(
                    {
                        "schema_version": "lamellarsaxs2d.project_run.v2",
                        "mode": run.mode,
                        "input_hash": run.input_hash,
                        "config_hash": run.config_hash,
                        "frames": compact_records,
                        "n_frames": len(run.frame_results),
                        "n_success": len(run.successful),
                        "n_failed": len(run.failures),
                        "checkpoint": (
                            str(run.checkpoint) if run.checkpoint is not None else None
                        ),
                    }
                )
            return 1 if run.failures else 0
        if args.command == "preflight":
            return _handle_preflight(args)
        if args.command == "benchmark":
            return _handle_benchmark(args)
        if args.command == "annotation-pack":
            return _handle_annotation_pack(args)
        if args.command == "p3-status":
            return _handle_p3_status(args)
        if args.command == "p4-evaluate":
            return _handle_p4_evaluate(args)
        parser.error(f"未知命令：{args.command}")
    except (PipelineError, ProjectConfigError, FileExistsError, OSError, ValueError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2
    return 2


__all__ = ["build_parser", "main"]
