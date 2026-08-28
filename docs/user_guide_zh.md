# 操作、输入输出、UI 与批处理指南

本指南面向当前 checkout 的 CLI、项目 TOML 和 Qt UI。先准备已知实验几何和 mask，再开始精修；PONI、mask、q 单位和输出目录都应随结果保存。科学量、符号、`full2d` 边界和不确定度解释见[科学量、符号、单位与可解释性边界](scientific_basis_zh.md)，模块关系见[软件架构与数据流](architecture_zh.md)。

## 1. 安装与启动

在项目根目录执行：

```powershell
python -m venv .venv-project
.\.venv-project\Scripts\python.exe -m pip install -c constraints\validation-py311-313.txt -e ".[all]"
```

安装后的 `bsaxs` 与 `python -m butterfly_saxs` 是同一 CLI。Windows 下建议直接使用 `.\.venv-project\Scripts\bsaxs.exe`；双击 `启动_LamellarSAXS2D.cmd` 也只使用这个已验证环境。旧 `.venv` 不会被删除或自动修复。

### 常用单帧命令

PONI 是物理 q 坐标的校准输入，CBF、EDF、TIF/TIFF 的最小示例：

```powershell
.\.venv-project\Scripts\python.exe -m butterfly_saxs inspect data\frame_0001.cbf --poni geometry\detector.poni
.\.venv-project\Scripts\python.exe -m butterfly_saxs analyze data\frame_0001.cbf --poni geometry\detector.poni -o results\frame_0001

.\.venv-project\Scripts\python.exe -m butterfly_saxs analyze data\frame_0001.edf --poni geometry\detector.poni -o results\edf_0001
.\.venv-project\Scripts\python.exe -m butterfly_saxs analyze data\frame_0001.tif --poni geometry\detector.poni -o results\tif_0001
```

`inspect` 只检查输入、q 范围和基础观测量；`analyze` 执行观测量、脊线和镜像双椭圆拟合。需要像素级经验模型时加 `--full2d`：

```powershell
.\.venv-project\Scripts\python.exe -m butterfly_saxs analyze data\frame_0001.cbf --poni geometry\detector.poni --full2d -o results\frame_0001
```

不想写文件时可以省略 `-o`，结果 JSON 摘要打印到终端。`-o` 指向目录时写 `<输入stem>.json` 和 `<输入stem>.npz`；也可直接指定 `.json`、`.npz` 或 `.csv`。已有输出不会自动覆盖，确认覆盖时显式加 `--force`。

### 真实数据包预检

在任何 ridge、ellipse 或 `full2d` 拟合前，先只读运行：

```powershell
.\.venv-project\Scripts\bsaxs.exe preflight data_local\real_validation\sample_package `
  --manifest manifest.csv `
  --poni geometry\detector.poni `
  --mask masks\detector_mask.npy `
  -o results\validation\preflight
```

输入是数据包、可选 context、manifest、PONI 和 mask；输出目录包含 `preflight.json`、`arrays.npz` 和 `run_report.md`。其中 `arrays.npz` 保存单位限定的 q-map，以及 finite、detector、external mask、q-window、ROI、weight、fit 和 sampled 共 8 类 analysis-domain 掩膜。成功标准是没有未解释的 `red` 检查项；`yellow` 表示只读预检完成，但必须阅读原因，例如 provisional 时间、`partial` 不确定度或未烧入强度的 solid-angle/polarization。该命令不运行拟合，也不改写图像或 mask。已有输出默认拒绝覆盖，只有确认后才用 `--force`。

退出码约定：`0` 为全部检查绿色通过；`1` 为预检完成但存在 `yellow/WARN`、质量失败或部分失败且证据已保留；`2` 为输入、配置、selector、单位、mask、PONI 或覆盖错误。

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
.\.venv-project\Scripts\python.exe -m butterfly_saxs analyze data\scan.h5 --dataset "/entry/data" --frame 3 --poni geometry\detector.poni -o results\scan3
.\.venv-project\Scripts\python.exe -m butterfly_saxs inspect data\stack.tif --frame 0 --poni geometry\detector.poni
```

掩膜参数的极性不同：`--valid-mask` 中 `True` 是有效像素；`--mask` 中 `True` 是无效像素。两者都是布尔数组或路径，必须与所选二维图像同形状。

图像的 `--frame/--dataset` 与掩膜的 `--mask-frame/--mask-dataset` 完全独立。NPY mask 不会继承 HDF5 图像的 dataset；多帧或多 dataset mask 必须显式给出自身 selector，否则快速失败。

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
.\.venv-project\Scripts\python.exe -m butterfly_saxs inspect -c project.toml
.\.venv-project\Scripts\python.exe -m butterfly_saxs project project.toml
.\.venv-project\Scripts\python.exe -m butterfly_saxs project project.toml --force
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
.\.venv-project\Scripts\python.exe -m butterfly_saxs gui
.\.venv-project\Scripts\python.exe -m butterfly_saxs gui data\frame_0001.cbf --poni geometry\detector.poni
```

首次启动默认显示中文。顶栏 `语言 / Language` 菜单可选择 `中文` 或 `English`，切换后窗口、按钮、页签、提示、状态栏、表头、测量图标签和 Fit session 会立即更新，无需重启。选择通过应用级 `QSettings` 的 `ui/language` 键全局记忆，不进入 schema 2 项目 JSON。参数键、q 单位、flags、`manual_status=unreviewed/accepted/rejected`、文件路径和 evidence provenance 始终保持机器可读原值，仅显示文字变化。

UI 文件选择器直接列出 CBF、EDF、TIF、TIFF；核心 I/O 还支持 NPY/NPZ/HDF5/CSV/TXT，后者可通过 CLI/API 或注入服务使用。典型顺序：

1. `Project → Open image…` 载入图像；多帧/数据集在项目或服务参数中指定 `frame`/`dataset`。
2. `Project → Select PONI…` 载入几何；`Project → Select external mask…` 载入外部无效像素 mask。
3. 在右侧参数表编辑 `Value`、`Min`、`Max`、`Vary`、`Expr`、`Unit`。角度的公共 UI 字段使用 degree 标注；内部求解器可用弧度，但不要把 degree 与 radian 混填。
4. 在 `Exclusion ROI (pixel)` 选择 `Rectangle` 或 `Ellipse`，填写边界/中心与半径，点击 `Apply`；`Clear` 清除 UI 排除区。
5. `Preview` 根据当前参数显示 observed/model/residual/overlay；Overlay 中青色虚线双椭圆来自右侧当前模型参数，橙色实线椭圆来自观测 ridge 的独立拟合，两者不得混作同一证据。`Optimize` 在后台精修可变参数；`Auto preview` 控制参数改变后的自动预览。状态栏显示 RMSE、ndata、flags、coverage。
6. 在 `Fit session` 中填写 `Snapshot note` 后点击 `Save snapshot`，可保存多个带顺序和备注的完整参数表；`Restore snapshot` 精确恢复所选参数。每次 Optimize 前软件还会自动保存一次完整状态，`Restore before optimize` 可撤销本次自动精修。取消任务或忽略迟到结果后，旧 worker 不会覆盖当前参数。
7. 最新一次 Preview 或 Optimize 成功后，填写 `Reviewer`，再点 `Accept current` 或 `Reject current`。修改参数、分析范围、输入、PONI、mask 或 ROI 后，状态会立即回到 `unreviewed`，必须重新 Preview 才能审核或导出。`accepted` 只表示该审核者接受当前人工拟合会话，不表示软件或 P3 科学门给出 PASS。
8. `Project → Export evidence…`（中文界面为“项目 → 导出证据…”）选择输出目录，固定生成 `observed.png`、`model.png`、`residual.png`、`overlay.png`、`parameters.csv`、`fit_session.json` 和 `provenance.json`。输入是当前屏幕对应的最新 Preview/Optimize、参数表和审核状态；输出默认不覆盖已有同名文件。成功标准是状态栏显示导出 7 个证据文件，七个文件均非空，`parameters.csv` 与当前参数表一致，两个 JSON 中 q 单位、输入路径/hash 和 `manual_status` 可复核。未审核结果也可以导出，但必须保持 `unreviewed`。
9. `Project → Save project…` 保存 schema 2 JSON 项目。它保存当前输入、PONI、mask、ROI、参数规格、人工审核状态、Optimize 前后摘要、参数快照、batch 帧列表和 batch 设置；不会把探测器尺寸的 observed/model/residual/qmap 数组塞进项目 JSON。重新打开后会恢复可复现输入与会话状态，再点击 Preview 重建四视图。相对路径按该 JSON 所在目录解析；CLI 的 TOML 仍是另一种项目格式，不要把二者当作同一 schema。

参数 `Vary=false` 是当前帧的固定参数；`Expr` 是受限、可审计的表达式绑定，绑定量不进入自由优化向量。固定参数的 stderr 不是零，见科学文档。

载入物理 PONI 后，`a`、`b`、径向宽度和背景宽度等 q 参数的 Unit 会刷新为当前 q 单位。`q min/q max` 只接受有限数值或自动范围标记（中文为 `自动`，英文为 `Auto`），非法文本以及 `q min >= q max` 会在任务启动前报错。Preview/Optimize 若失败或没有返回新 model/residual，界面会清空旧图并显示失败状态，避免把上一轮结果误认为本轮结果。

## 6. 批处理、warm start 与恢复

### CLI

PowerShell 中若使用通配符，建议加引号让 CLI 自己展开；CLI 也接受目录：

```powershell
.\.venv-project\Scripts\python.exe -m butterfly_saxs batch "data\frame_*.cbf" --poni geometry\detector.poni -o results\batch --mode independent
.\.venv-project\Scripts\python.exe -m butterfly_saxs batch "data\frame_*.cbf" --poni geometry\detector.poni -o results\batch --mode warm_start --manifest sequence.csv --checkpoint results\checkpoint.json
.\.venv-project\Scripts\python.exe -m butterfly_saxs batch "data\frame_*.cbf" -c project.toml --resume --force
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

- `<stem>.json`：严格 JSON 摘要，包含 `metadata`、`flags`、`observables`、`ridges`、`ellipse_fit`、顶层 `parameters`、可选 `full2d`、`qmap`、`analysis_domain`、`valid_mask` 和 `output_paths`；大数组在 JSON 中以摘要形式保存，非有限值转为 `null`。
- `<stem>.npz`：压缩数组 sidecar，保存图像、`fit_valid_mask`、`sampled_valid_mask`、q/qx/qy、脊点数组，以及存在时的 `full2d_model`/`full2d_residual`；还保存 `observables_json`。

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

1. 先用 `preflight` 确认 manifest、hash、图像 shape、PONI、mask 极性、校正/不确定度状态和 `q_unit`；
2. 确认 PONI 与图像 shape 相符，mask 极性和 shape 正确；
3. 先不用 `full2d` 检查 ridge/ellipse 的 `coverage`、`condition`、中心与 flags；
4. 对 beamstop/streak/overlap 增加排除区，比较 mask 前后的 ridge 支持，不用对称复制填洞；
5. 最后再运行 `full2d`，检查 `weighting`、`ndata`、`sampled_n`、`covariance_local_linear_approximation` 和整幅 residual；
6. 保存命令、TOML/JSON 项目、PONI、mask、输入 hash 和输出 provenance，确保后续结果可复核。

## 9. P3 基准、盲标包与证据门

P3 的用途是把“同模型实现正确”“对独立 FFT 图样有泛化能力”和“真实图像人工重复性”分开，不用软件自己的经验模型证明全部科学有效性。以下命令只生成测试证据或读取现有文件，不运行 R0 真实数据拟合。

生成 T1 同模型矩阵和 T2 独立实空间层片 + FFT 基准：

```powershell
.\.venv-project\Scripts\bsaxs.exe benchmark --suite t1 --seed 20260828 -o results\validation\synthetic_same_model\p3_run
.\.venv-project\Scripts\bsaxs.exe benchmark --suite t2 --shape 256x256 --seed 20260828 -o results\validation\synthetic_independent\p3_run
```

- T1 包含 15 个 case，覆盖噪声、参数变化、中心/q 范围/shape、探测器缺陷、低 SNR、重叠和非椭圆负例。它复用经验强度模型，只能验证实现、真值恢复和拒绝逻辑。
- T2 从实空间有限层片堆叠做二维 FFT，不导入 `intensity.py` 或旧 synthetic 模块。`projection_truth` 是只由层间距和取向分布计算的解析 Bragg 轨迹，不读取或吸附到生成的 FFT 像素；`structure_truth` 只描述生成器结构，不是经验参数的反演真值。`non_elliptical` 的轨迹仅用于负类识别，不作为定量反演真值。T2 仍是简化外部测试源，不是完整 3D 物理正演。
- 目标目录默认不得已有同名证据；`--force` 只覆盖本命令列出的目标文件，不删除无关文件。

从 R0 清单创建固定 8 帧盲标包：

```powershell
$projectRoot = (Get-Location).Path
.\.venv-project\Scripts\bsaxs.exe annotation-pack data_local\real_validation\2#_no50s_375_2_iso `
  --rt-manifest project_rt_reference_manifest.csv `
  --hold-manifest project_hold_375C_manifest.csv `
  --preflight "$projectRoot\results\validation\preflight\r0_hold\preflight.json" `
  --poni config\geometry\BL19B2_SAXS_Califile.poni `
  --mask masks\bl19b2_mask.npy `
  -o results\validation\annotations\r0_pilot
```

输出中的 `blind_payload/` 只包含 8 张不带拟合覆盖层的 `blind_*.png`、标注协议和两份待填写标注表，是唯一应分发给标注者的目录。上级目录另存选择审计表、待填写 consensus 表和 `annotation_status.json`，其中含帧角色/选择理由，只供协调者保管，consensus 前不得分发。有 preflight 时先用其中的原始强度摘要排序，只读取最终 8 帧图像；输入文件读取前后 SHA-256 必须一致。软件不会代替人填写标注。

评估 P3 门禁：

```powershell
.\.venv-project\Scripts\bsaxs.exe p3-status `
  --t1-manifest results\validation\synthetic_same_model\p3_run\truth_manifest.json `
  --t2-manifest results\validation\synthetic_independent\p3_run\truth_manifest.json `
  --annotation-status results\validation\annotations\r0_pilot\annotation_status.json `
  --thresholds configs\acceptance_thresholds_draft_v1.json `
  -o results\validation\p3_gate\p3_gate_report.json
```

返回码 `0` 表示四项证据检查均通过；团队可据报告决定是否进入 P4，工具本身不运行拟合，也不强制阻止后续阶段。返回码 `1` 表示科学证据 No-Go，`2` 表示路径、JSON 或参数错误。门禁会实际打开 NPZ 核对数组、mask、shape、q 恒等式、T2 FFT 重建和生成器 hash，并核对两份标注 CSV、consensus CSV、审计 manifest 都有 8 个相同盲号。只有完成 8 帧人工 consensus，并用人工重复性、仪器分辨率和 pilot 证据冻结 `acceptance_thresholds_v1.json` 后才可能返回 Go；draft 阈值不能用于最终 PASS/FAIL。正式阈值中的人工/仪器证据记录必须带 `status=complete`、存在的来源文件及其 SHA-256、有限数值和单位，pilot 记录必须明确 8 帧，且要记录 `frozen_by/frozen_at`；仅填写任意非空字符串不会通过门禁。

### 9.1 盲标模板的实际完成条件

生成器预填的 `blind_id`、`coordinate_system` 和 `image_version` 是固定身份字段，不要改写。完成盲标时，`annotator_a.csv`、`annotator_b.csv` 和 `consensus_review.csv` 必须各有且仅有 `blind_001`–`blind_008` 八行；前两份的 `valid_area`、`beamstop`、`streak`、`overlap`、`lobe_center_x`、`lobe_center_y`、`ridge_points`、`software`、`software_version`、`coordinate_system`、`image_version`、`annotation_time`、`annotator` 必须逐行非空，consensus 表的 `consensus_status`、上述标注字段、`reviewer` 和 `review_time` 也必须逐行非空。`valid_area` 必须由至少 3 个不同的有限 `[x,y]` 点构成，鞋带公式面积必须大于 0，不能写 `[]`、`unknown` 或共线/重复点；明确不存在的 beamstop/streak/overlap/ridge 可写 `[]`，其他确实无法判断的字段可写 `unknown`，但整行占位内容不会通过。`notes` 可留空。时间必须是可解析的带时区时间戳，坐标系必须保持 `image_pixel_x_right_y_up_origin_lower_left`，`image_version` 必须等于对应 PNG 的 SHA-256。

同时，`annotation_status.json` 必须从生成时的 `status=awaiting_human_annotations`、`human_consensus=false` 更新为门禁允许的完整状态：`schema_version=lamellarsaxs2d.annotation_pack.v2`、`human_consensus=true`、`candidate_count=8`、`consensus_records_count=8`，并在 `human_evidence` 中记录盲化方式。可接受的方式是两名不同标注者（`mode=two_independent_annotators`、`annotator_count>=2`），或同一专家间隔至少 7 天复标（`mode=one_expert_repeat`、`session_count>=2`、`interval_days>=7`、`lower_evidence=true`）。`files` 指向的 CSV、八个 PNG 和不可变输出哈希都必须能读回并与实际文件一致；每条 `input_hashes` 都必须有 64 位小写 SHA-256，且 `sha256_before == sha256_after`、`unchanged=true`。

### 9.2 `evidence_templates` 三种 JSON 契约

以下模板当前只是待补证据的骨架，不能把 `awaiting_*` 或 `null` 当作完成：

| 文件 | 完成时必须满足的字段 |
|---|---|
| `configs/evidence_templates/human_repeatability_template.json` | `schema_version=lamellarsaxs2d.human_repeatability.v1`；`status=complete`、`blinded=true`、`mode` 为 `two_independent_annotators` 或 `one_expert_repeat`、`frame_count=8`；`annotation_status_sha256` 等于完成后的状态文件哈希；`per_frame_error_px` 恰好记录 8 个盲号的有限非负误差；`metric.value` 必须等于声明的 `mean/median/p95/max/min` 聚合，单位为 `px`；`reviewed_by` 非空、`reviewed_at` 可解析。 |
| `configs/evidence_templates/instrument_resolution_template.json` | `schema_version=lamellarsaxs2d.instrument_resolution.v1`；`status=complete`；`measurements_nm_inv` 为非空有限正数序列；`metric.value` 等于声明的聚合、单位为 `nm^-1`；`calibration_record.source` 存在且 SHA-256 匹配；`method`、`reviewed_by` 非空且 `reviewed_at` 可解析。记录的是实测或 beamline 批准的 q 分辨率，不把 detector sampling 单独当成完整仪器分辨率。 |
| `configs/evidence_templates/pilot_evidence_template.json` | `schema_version=lamellarsaxs2d.pilot_evidence.v1`；`status=complete`、`frame_count=8`；`blind_ids` 恰好为 `blind_001`–`blind_008`；`annotation_status_sha256` 和 `consensus_sha256` 匹配当前文件；`frame_results` 的 8 个状态逐帧等于 consensus CSV；`reviewed_by` 非空、`reviewed_at` 可解析。 |

在冻结阈值 JSON 的 `evidence_sources` 中，每条记录都必须包含 `status=complete`、`source` 和 `sha256`，且记录的哈希与文件当前内容一致；人工重复性和仪器分辨率 wrapper 还必须有与 source 的数值、单位和聚合方式逐项匹配的 `metric`，pilot wrapper 必须有 `frame_count=8`。source 内容必须满足上表的逐帧、原始校准记录或 consensus 绑定合同。最终阈值本身需为 `schema_version=lamellarsaxs2d.acceptance_thresholds.v1`、`thresholds_version=v1`、`status=frozen`、`frozen=true`、`usable_for_final_pass_fail=true`，且保存 `frozen_by` 与可解析的 `frozen_at`。因此仓库现有 `acceptance_thresholds_draft_v1.json` 和三个 `awaiting_*` 模板只能用于占位/检查路径，不能产生最终 Go。

### 9.3 `p3-status` 的输入哈希与 provenance

`p3-status` 是只读证据汇总。报告的 `inputs` 对 `t1_manifest`、`t2_manifest`、`annotation_status` 和 `thresholds` 逐项保存绝对路径及当前 SHA-256；`provenance.gate_code_sha256` 记录门禁代码，`provenance.evidence_fingerprint_sha256` 是按输入名称排序后对四个输入哈希映射计算的 SHA-256，并同时记录 Python、NumPy 版本及 T1/T2 generator version/hash。请把 `p3_gate_report.json` 与这些输入放在同一证据目录中保存；若重新生成任何输入，应生成新的报告并重新核对哈希。它只读取和报告证据，不修改输入、不冻结阈值、不运行真实数据拟合。

生成只读总览图：

```powershell
.\.venv-project\Scripts\python.exe scripts\render_p3_overviews.py `
  --t1-manifest <T1 truth_manifest.json> `
  --t2-manifest <T2 truth_manifest.json> `
  --annotation-status <annotation_status.json> `
  -o <新的总览输出目录>
```

## 10. P4 ridge/lobe/双椭圆工程验证

`p4-evaluate` 运行固定 T1/T2 套件，并可选读取固定 8 帧 R0 包。它用于检查软件是否能定位 ridge/lobe、拟合或拒绝双椭圆以及保留输入哈希，不会自动填写人工标注，也不会把工程结果写成科学接受。

```powershell
.\.venv-project\Scripts\bsaxs.exe p4-evaluate `
  --t1-manifest results\validation\synthetic_same_model\p3_run\truth_manifest.json `
  --t2-manifest results\validation\synthetic_independent\p3_run\truth_manifest.json `
  --thresholds configs\acceptance_thresholds_draft_v1.json `
  --skip-sensitivity `
  -o results\validation\p4_engineering\p4_run
```

需要同时运行 R0 固定 8 帧时，再提供 `--r0-package`、`--r0-manifest`、`--poni` 和 `--mask`。输出目录必须是新目录，主要文件为 `p4_engineering_report.json` 和 `p4_summary.csv`；`0` 表示当前工程合同为 GO，`1` 表示工程 No-Go，`2` 表示输入、路径或参数错误。T1 的 detector-pixel 误差按合成线性 q 网格的 `(dq_y, dq_x)` 分轴换算；该口径不能直接冒充一般 PONI 曲线 q-map 的完整仪器分辨率。`results/validation/` 是本地证据目录，不应上传 GitHub。

即使 P4 工程主链可以执行，缺少真实人工 consensus、人工重复性、仪器 q 分辨率和冻结阈值时，P3 科学门禁仍为 No-Go。工程继续实施与正式科学验收是两件事。
