# 操作、输入输出、UI 与批处理指南

本指南面向当前 checkout 的 CLI、项目 TOML 和 Qt UI。先准备已知实验几何和 mask，再开始精修；PONI、mask、q 单位和输出目录都应随结果保存。科学量、符号、`full2d` 边界和不确定度解释见[科学量、符号、单位与可解释性边界](scientific_basis_zh.md)，模块关系见[软件架构与数据流](architecture_zh.md)。

## 1. 安装与启动

在项目根目录执行：

```powershell
py -3 -m pip install -e .
# 图形界面还需要 Qt/pyqtgraph 可选依赖
py -3 -m pip install -e ".[ui]"
```

安装后的 `bsaxs` 与 `py -3 -m butterfly_saxs` 是同一 CLI。若 Windows 的 `python` 指向 Store 占位程序，优先使用项目环境中的 `py -3` 或明确的虚拟环境解释器。

### 常用单帧命令

PONI 是物理 q 坐标的校准输入，CBF、EDF、TIF/TIFF 的最小示例：

```powershell
py -3 -m butterfly_saxs inspect data\frame_0001.cbf --poni geometry\detector.poni
py -3 -m butterfly_saxs analyze data\frame_0001.cbf --poni geometry\detector.poni -o results\frame_0001

py -3 -m butterfly_saxs analyze data\frame_0001.edf --poni geometry\detector.poni -o results\edf_0001
py -3 -m butterfly_saxs analyze data\frame_0001.tif --poni geometry\detector.poni -o results\tif_0001
```

`inspect` 只检查输入、q 范围和基础观测量；`analyze` 执行观测量、脊线和镜像双椭圆拟合。需要像素级经验模型时加 `--full2d`：

```powershell
py -3 -m butterfly_saxs analyze data\frame_0001.cbf --poni geometry\detector.poni --full2d -o results\frame_0001
```

不想写文件时可以省略 `-o`，结果 JSON 摘要打印到终端。`-o` 指向目录时写 `<输入stem>.json` 和 `<输入stem>.npz`；也可直接指定 `.json`、`.npz` 或 `.csv`。已有输出不会自动覆盖，确认覆盖时显式加 `--force`。

### 无 PONI 的边界

没有 PONI 时 pipeline 可以生成以图像中心为基准的确定性像素坐标，q 单位为 `pixel-q` 并带 `uncalibrated_pixel_q`。这适合算法烟测或同条件相对比较，不是物理 `nm^-1`。只有在用户掌握可靠标定并显式配置 `q_scale`/`q_unit` 时，才可把该坐标声明为物理 q；不要因为字段名叫 `q` 就自动换算纳米周期。`pixel-q` 与 `nm^-1`、论文 Table 3 的 pixel 半轴的区别见科学文档。

## 2. 输入格式与选择器

核心 I/O 支持：

| 格式 | 读取器/注意事项 |
|---|---|
| `.cbf`, `.edf` | 通过 FabIO 读取；多帧时使用零基 `--frame`。 |
| `.tif`, `.tiff` | 通过 tifffile 读取；多页时使用零基 `--frame`。 |
| `.npy` | 数组必须能选出严格二维图像；多帧数组显式给 `--frame`。 |
| `.npz` | 使用 `--dataset KEY` 选择键；文件含多个候选二维数组时必须明确选择。含 `data` 及 `qx/qy/q` 的 fixture 也可由 pipeline 读取其 qmap。 |
| `.h5`, `.hdf5`, `.hdf` | 通过 h5py 读取；多个数据集或路径不明确时显式给 `--dataset`。 |
| `.csv`, `.txt` | 读取二维数值表；不适合作为带实验元数据的通用容器。 |

所有选中的图像必须严格二维且与 qmap/mask 形状一致。读取阶段不做隐式归一化；若需暗场、曝光或监视器校正，应在项目外明确完成并记录。

命令行选择器：

```powershell
py -3 -m butterfly_saxs analyze data\scan.h5 --dataset "/entry/data" --frame 3 --poni geometry\detector.poni -o results\scan3
py -3 -m butterfly_saxs inspect data\stack.tif --frame 0 --poni geometry\detector.poni
```

掩膜参数的极性不同：`--valid-mask` 中 `True` 是有效像素；`--mask` 中 `True` 是无效像素。两者都是布尔数组或路径，必须与所选二维图像同形状。

## 3. 项目 TOML

项目配置把输入、PONI、分析开关和输出目录集中保存。相对路径按 TOML 文件所在目录解析。下面是一个可改写的最小骨架；其中 `rois` 是排除区，坐标是像素，`q_sector` 需要已生成 qmap：

```toml
[project]
q_unit = "1/nm"
full2d = false

[inputs]
files = ["data/frame_*.cbf"]
poni = "geometry/detector.poni"

[output]
directory = "results"

[analysis]
q_window = [0.02, 0.20]
ridge_method = "surface_curvature"
n_angles = 72
n_angular_bins = 360
n_radial_bins = 256
mask = "masks/detector_mask.npy"
batch_mode = "warm_start"
manifest = "sequence.csv"
checkpoint = "results/checkpoint.json"
resume = false
# 默认省略 max_pixels，使用全部有效像素。仅在交互试算或超大图像需要限时响应时，
# 才显式设置 max_pixels；此时 sampled_n/sample_rmse 会与全图 ndata/rmse 分开报告。
# max_pixels = 50000
robust_loss = "soft_l1"
# 自动生成初值时，软件只估计幅度/背景的起始数值尺度，不改变输入强度；
# 显式参数和 warm start 默认保持原值。需要强制重估时再设置为 true。
# auto_scale_initial = true

[[analysis.rois]]
type = "rectangle"
x0 = 0
x1 = 80
y0 = 0
y1 = 80

[[analysis.rois]]
type = "ellipse"
cx = 512
cy = 512
rx = 35
ry = 35
angle_deg = 0

# 有校准 qmap 时才使用 q-sector；它仍是排除区（True=无效）。
[[analysis.rois]]
type = "q_sector"
q_min = 0.02
q_max = 0.06
chi_min_deg = 170
chi_max_deg = -170
```

以 TOML 运行单帧/项目：

```powershell
py -3 -m butterfly_saxs inspect -c project.toml
py -3 -m butterfly_saxs project project.toml
py -3 -m butterfly_saxs project project.toml --force
```

`project` 会按 `inputs.files` 顺序/自然排序逐帧分析，并将单帧 JSON/NPZ 写入 `output.directory`。`--force` 只应在确认目标输出可覆盖时使用。若要获取批处理的统一 CSV/JSONL/NPZ 汇总，使用下一节的 `batch` 命令。

## 4. beamstop、streak 与 overlap mask

建议先做一张可复用外部 mask，再将实验性排除区写入 TOML：

- beamstop：中心圆/椭圆区域；
- equatorial streak：沿 streak 的狭长矩形或旋转椭圆；
- branch overlap/中心散射：只排除受污染的 q/chi 扇区，不要以镜像点填回；
- 热像素、饱和像素和 detector gap：外部 `mask`；
- 仅在有物理 qmap 时使用 `q_sector`，否则 q-sector 的数值没有物理依据。

当前公共 mask/ROI 约定是 `True=排除`，多个排除区 OR 合并；输入 `valid_mask` 在 I/O 边界转换为 `True=保留`。UI 的 ROI 编辑器只提供像素 `Rectangle` 和 `Ellipse`；q-sector 使用 TOML 的 `analysis.rois` 或 Python API。每次分析应在 JSON/provenance 中保留 mask 路径、ROI 参数和 q 单位。

## 5. Qt UI：载入、预览与精修

启动：

```powershell
py -3 -m butterfly_saxs gui
py -3 -m butterfly_saxs gui data\frame_0001.cbf --poni geometry\detector.poni
```

UI 文件选择器直接列出 CBF、EDF、TIF、TIFF；核心 I/O 还支持 NPY/NPZ/HDF5/CSV/TXT，后者可通过 CLI/API 或注入服务使用。典型顺序：

1. `Project → Open image…` 载入图像；多帧/数据集在项目或服务参数中指定 `frame`/`dataset`。
2. `Project → Select PONI…` 载入几何；`Project → Select external mask…` 载入外部无效像素 mask。
3. 在右侧参数表编辑 `Value`、`Min`、`Max`、`Vary`、`Expr`、`Unit`。角度的公共 UI 字段使用 degree 标注；内部求解器可用弧度，但不要把 degree 与 radian 混填。
4. 在 `Exclusion ROI (pixel)` 选择 `Rectangle` 或 `Ellipse`，填写边界/中心与半径，点击 `Apply`；`Clear` 清除 UI 排除区。
5. `Preview` 根据当前参数显示 observed/model/residual/overlay；`Optimize` 在后台精修可变参数；`Auto preview` 控制参数改变后的自动预览。状态栏显示 RMSE、ndata、flags、coverage。
6. `Project → Save project…` 保存 JSON 项目。它保存当前输入、PONI、mask、ROI、参数规格、batch 帧列表和 batch 设置；载入时，相对文件/目录路径按该 JSON 所在目录解析。CLI 的 TOML 仍是另一种项目格式，不要把二者当作同一 schema。

参数 `Vary=false` 是当前帧的固定参数；`Expr` 是受限、可审计的表达式绑定，绑定量不进入自由优化向量。固定参数的 stderr 不是零，见科学文档。

## 6. 批处理、warm start 与恢复

### CLI

PowerShell 中若使用通配符，建议加引号让 CLI 自己展开；CLI 也接受目录：

```powershell
py -3 -m butterfly_saxs batch "data\frame_*.cbf" --poni geometry\detector.poni -o results\batch --mode independent
py -3 -m butterfly_saxs batch "data\frame_*.cbf" --poni geometry\detector.poni -o results\batch --mode warm_start --manifest sequence.csv --checkpoint results\checkpoint.json
py -3 -m butterfly_saxs batch "data\frame_*.cbf" -c project.toml --resume --force
```

`manifest` 可用 CSV/JSON 提供 `path`、`frame_id`、`time`、`order`、`dataset`、零基 `frame`（或 `frame_index`）等元数据。manifest 文件中的相对 `path` 按 manifest 所在目录解析。它可以让同一个 HDF5/NPZ/TIFF 容器中的不同 dataset/frame 成为独立批处理记录；这些选择器会传给实际读取器并进入 checkpoint 身份。没有 manifest 时使用自然文件名排序。`checkpoint` 记录输入内容 hash、配置 hash、模式和每帧控制状态；`--resume` 只有 hash/mode 相符时才恢复。

批处理即使有失败帧也会先写出其余帧和失败记录，便于原位序列排查；只要存在失败帧，CLI 进程返回码为 `1`，全帧成功返回 `0`。因此自动化脚本既可读取 `frame_summary.csv` 做逐帧处理，也不会把部分失败误判为整批成功。

### 两种已实现模式

- `independent`：每帧独立调用分析器，不使用上一帧结果作为初值。
- `warm_start`：上一帧明确通过质量检查的结果作为下一帧初始状态；失败帧不会成为后续初值。checkpoint 恢复的成功帧也可以成为 warm start 链的一部分。

当前批处理**没有**跨全序列的 global/shared 联合优化：没有一个自由参数向量同时用所有帧求解。`fixed`/`Expr` 只定义每一帧精修的参数状态；如果希望参数在序列中保持同一固定值，请在配置中明确固定并把该配置随 provenance 保存，但不要把它称为 joint/shared refinement。真正的跨帧 shared free 参数拟合属于后续扩展。

UI 的 `Batch` 页提供 `Add frames…`、`Run batch`、`Independent/Warm start`、manifest、checkpoint、resume 和 output；`Evolution` 页可选择数值字段查看帧/时间演化。没有采集时间时，图表使用 frame/index，不把文件修改时间冒充实验时间。失败帧会保留 status/error，不静默插值。

## 7. 结果文件与字段

### 单帧

`analyze -o <目录>` 创建：

- `<stem>.json`：严格 JSON 摘要，包含 `metadata`、`flags`、`observables`、`ridges`、`ellipse_fit`、顶层 `parameters`、可选 `full2d`、`qmap`、`valid_mask` 和 `output_paths`；大数组在 JSON 中以摘要形式保存，非有限值转为 `null`。
- `<stem>.npz`：压缩数组 sidecar，保存图像、有效 mask、q/qx/qy、脊点数组，以及存在时的 `full2d_model`/`full2d_residual`；还保存 `observables_json`。

`-o` 指向 `.json`、`.npz` 或 `.csv` 时只写相应格式。JSON 是机器可读摘要，不应当作完整像素证据；需要复核像素、qmap、model/residual 时读取 NPZ。

### 批处理

`batch` 的统一导出目录包含以下逻辑文件（实际路径由 CLI JSON 的 `outputs` 返回）：

- `frame_summary.csv`：每帧 status、error、时间、warm-start lineage、标量摘要和 flags；
- `parameters_long.csv`：`parameter/value/stderr/uncertainty/fixed/unit/flags/bound_flags` 等长表；非有限数值留空，不写成 0；
- `ridge_points.csv`：逐帧逐点的 q、角度、强度、coverage、valid/reason 等；
- `ellipse_fit.json` 与 `ellipse_fit.jsonl`：逐帧椭圆拟合 JSON；
- `manifest.json`、`provenance.json`：输入/config hash、模式、版本、时间和用户 provenance；
- `results.npz`：批次的数组/结果 sidecar；其 `__metadata__` 明确给出 `complete` 与缺失帧列表。失败帧或只从不含数组的 checkpoint 恢复的帧会令 `complete=false`，不能把此文件误称为全帧像素证据；
- `evolution.png`：可用标量参数按单位分组的演化图。

不要只看 `status=ok` 或 `success=True`：同时检查 `scientific_flags`、`coverage`、`condition`、`bound_flags`、`stderr/uncertainty`、`q_unit` 和 mask 记录。`reduced_chi_square` 只是当前残差诊断，不是默认统计检验。

## 8. 最小排查路径

遇到结果异常时按以下顺序留证：

1. 先用 `inspect` 确认图像 shape、finite fraction、q 范围和 `q_unit`；
2. 确认 PONI 与图像 shape 相符，mask 极性和 shape 正确；
3. 先不用 `full2d` 检查 ridge/ellipse 的 `coverage`、`condition`、中心与 flags；
4. 对 beamstop/streak/overlap 增加排除区，比较 mask 前后的 ridge 支持，不用对称复制填洞；
5. 最后再运行 `full2d`，检查 `weighting`、`ndata`、`sampled_n`、`covariance_local_linear_approximation` 和整幅 residual；
6. 保存命令、TOML/JSON 项目、PONI、mask、输入 hash 和输出 provenance，确保后续结果可复核。
