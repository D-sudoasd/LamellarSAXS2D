# P3 基准协议与 R0 边界

> 本文件是可执行的证据协议，不是结果报告。它不声明当前仓库已经完成 T1、T2、R0、P3、P4 或 P9；是否通过只能以实际生成的证据文件和复核记录为准。

## 1. 共同数据契约

- T1、T2 和 R0 的物理 q 序列化单位统一为 `nm^-1`。T1 的 q 网格由合成器声明，T2 的 q 网格由 `2π * fftfreq(pixel_size_nm)` 构造；不能把 `pixel-q` 或 detector pixel 当作物理 q。
- 基准数组中的 `mask=True` 表示探测器像素被排除/无效，`valid_mask`（若提供）应为 `~mask`。这与真实数据的原始 NPY 约定不同：R0 原始 mask 为 `0=有效、1=无效`，进入分析域后才转换为 `valid_mask=True` 有效、外部 mask/ROI 的 `True` 排除。
- 每次生成都要在 truth manifest 和 NPZ 中保存实际 `seed`、`generator_version`、`generator_hash` 及适用的依赖 hash；严格 JSON 不允许 NaN/Inf。证据目录默认不覆盖已有目标，`--force` 只允许明确覆盖本次命令列出的输出文件。

## 2. T1：同模型合成矩阵

T1（`P3.1/T1`）使用 `butterfly_saxs.intensity.double_ellipse_intensity` 生成同一经验模型族的数据，用于检查数据处理、q-map、mask 传播、序列化、参数/ridge 真值恢复和模型失配拒绝逻辑。当前默认矩阵有 15 个 case：

```text
noiseless_default, gaussian_parameter_sweep, poisson_counting,
center_offset, q_range_narrow, shape_rectangular, beamstop, streak,
gap, bad_points, missing_sector, combined_detector_artifacts, low_snr,
overlap, negative_non_elliptic
```

默认不传 `--seed` 时，当前默认 case 使用 `20260827` 的确定性起始种子；传入 `--seed S` 时，证据目录 writer 按 case 顺序使用 `S+i`（`i` 从 0 开始）。`mask=True` 仍表示排除/无效，q 数组和 truth 中均声明 `q_unit="nm^-1"`。其中 `negative_non_elliptic` 是同模型中的有意负例，只用于检查模型失配/拒绝逻辑，不是一个可定量回收的单椭圆真值。

T1 输出为每个 case 的 NPZ、truth JSON 和总 `truth_manifest.json`。门禁会核对 15 个 case 是否齐全，并实际检查要求的数组、shape、有限值、truth、生成器版本/hash 和依赖 hash；只看 manifest 文件名或一个低误差数字不能代替这些检查。T1 仍是同模型测试，不证明独立物理真实性、跨数据域泛化或材料机制。

## 3. T2：独立实空间层片 + 二维 FFT

T2 从有限实空间层片堆叠生成密度，再执行二维 FFT 得到 reciprocal-space intensity；实现不导入应用的经验 `intensity.py` 或旧 synthetic 模块。默认有四类且种子属于公开基准定义：

| `category` | 默认 seed | 用途 |
|---|---:|---|
| `2-point` | 2101 | 两个对称 Bragg 点 |
| `eyebrow` | 2102 | 有限取向分布形成的 eyebrow 轨迹 |
| `butterfly` | 2103 | 分布、弯曲、波纹和轻微不对称共同形成的 butterfly |
| `non_elliptical` | 2104 | 非椭圆负类，只用于分类/拒绝 |

不传 `--seed` 时使用表中的 case seed；传入 `--seed S` 时，writer 按 case 顺序使用 `S+i`。T2 的 `qx/qy/q` 单位为 `nm^-1`，`q=hypot(qx,qy)`，`mask=True` 表示排除/无效。`projection_truth` 只由生成器的层间距和取向分布计算，明确独立于生成的 FFT 像素；`structure_truth` 描述生成结构，不是经验椭圆参数的逆拟合真值。`non_elliptical` 的投影 truth 只能作 negative classification，不能作为定量反演目标。T2 是简化的独立测试源，不是完整三维物理正演。

T2 输出为每个 case 的 NPZ 和 `truth_manifest.json`。门禁会重建 q 网格、FFT、实空间 density 和 projection truth，并比较生成器版本/hash；只要代码或输入 hash 不匹配，就不能把旧证据当作当前实现的证据。

## 4. R0：真实数据包的只读边界

R0 是当前样品、仪器和处理流程组成的单一真实数据域。当前计划中的数据角色为 1 个 RT 帧加 120 个 375 ℃保温帧，但实际路径、帧数和 selector 必须以 manifest 为准，不能凭文件名猜测。R0/P9-A 的第一步是只读 `preflight`，输入包括图像、包内 PONI、最终 NPY mask、RT/hold manifests 和 `project_context.yaml`（若存在）。

R0 的 q 只有在 PONI 可读且几何/单位确认后才可声明 `nm^-1`；否则必须降级为 `pixel-q` 或 `unknown`，不得输出物理 `d` 或周期。原始 NPY mask 按 `0=有效、1=无效` 记录来源和 hash，内部转换后的 `valid_mask=True` 表示有效，外部 mask 与 ROI 的 `True` 表示排除。preflight 和 `annotation-pack` 都不得清洗、归一化、覆盖、移动或删除原始图像、PONI、mask 或 manifest；运行前后 hash 必须一致，结果写入新的结果目录。

R0 首批只做输入、几何、mask、selector、校正状态、不确定度状态和 provenance 检查：`fit=null`、`solver_status=not_run`，ridge/lobe/ellipse/full2d 为 `NOT_TESTED`。不能把 R0 preflight 的通过写成拟合准确、跨仪器泛化或材料机制证明。

`annotation-pack` 从 RT/hold manifest 固定选择 8 帧（RT、保温首帧/中间帧/末帧和 4 个困难候选），生成不带拟合覆盖层的 PNG 和待填写 CSV。生成时 `annotation_status.json` 为 `status=awaiting_human_annotations`、`human_consensus=false`；必须完成 8 帧两份独立盲标（或同一专家间隔至少 7 天复标）及 consensus，并通过实际内容、时间戳、坐标系和 hash 检查后，才可作为 P3 的 R0 人工证据。每行 `valid_area` 必须由至少 3 个不同有限点组成且鞋带公式面积大于 0，整行 `unknown`/`[]` 或共线点占位不算标注。模板填写规则见[操作、输入输出、UI 与批处理指南](../user_guide_zh.md)。

## 5. seed、hash 与 provenance

种子不是口头备注：应同时出现在 manifest、每个 case 的 truth/NPZ 元数据和运行命令中。T1 的 `generator_hash` 由当前 T1 生成器及 `intensity.py` 的依赖 hash 组合得到；T2 的 `generator_hash` 是当前 T2 生成器源码 SHA-256。`p3-status` 会比较 manifest、NPZ 与当前生成器版本/hash，并重算 T2 的独立构造；生成器源码变化后必须重新生成证据。

盲标包的 `annotation_status.json` 保存：

- `input_hashes`：每个输入的 `sha256_before`、`sha256_after` 和 `unchanged=true`；
- `blind_image_hashes`：8 个 PNG 的 hash；
- `immutable_output_hashes`：审计 manifest、协议和 PNG 等不可变输出的 hash；
- `files`：CSV、PNG 和协议的相对路径。

冻结阈值 JSON 的 `evidence_sources` 每条记录都必须有 `status=complete`、`source` 和 `sha256`，门禁会重新读取文件并比对 hash。人工重复性 source 必须提供 8 帧 `per_frame_error_px`，其聚合值与 wrapper `metric` 匹配；仪器 source 必须提供正数测量序列及可复算聚合，并绑定一个实际 calibration record 的 SHA-256；pilot source 必须同时绑定完成后的 `annotation_status.json`、consensus CSV 及 8 个逐帧 consensus 状态。只填写非空占位字符串或只列 8 个盲号不算证据。

`p3-status` 报告的 `inputs` 对 T1 manifest、T2 manifest、`annotation_status.json` 和 thresholds JSON 逐项保存当前绝对路径及 SHA-256。`provenance.gate_code_sha256` 记录门禁代码，`provenance.evidence_fingerprint_sha256` 对按输入名称排序的四个输入 hash 映射计算 SHA-256，并同时记录 Python、NumPy、T1/T2 generator version/hash。任何输入改变都应生成新的报告，不应沿用旧 fingerprint。

## 6. P3 Go/No-Go 判定

`p3-status` 是只读证据门，固定检查四项：

1. `t1_same_model_matrix`：15 个 T1 case、NPZ/JSON 数组与 truth/hash 完整；
2. `t2_independent_generator`：T2 四类齐全，独立于经验模型，FFT 与 projection/structure truth 可复核；
3. `r0_human_consensus`：8 个盲号的两份标注和 consensus 实际内容完整，盲化方式和 hash/时间规则满足；
4. `acceptance_thresholds_frozen`：阈值版本为正式 `v1`，状态为 frozen，可用于最终 PASS/FAIL，且人工重复性、仪器分辨率、pilot 三份来源均存在、内容有效、hash 匹配并记录冻结人/时间。

四项全部 `PASS` 才返回 `status=go`、退出码 `0`；任一缺失、内容为空、hash 不匹配、单位错误或阈值仍为 draft/provisional 都返回 `status=no_go`、退出码 `1`。路径、JSON 或参数错误返回退出码 `2`。No-Go 表示当前证据不足或不可复核，不等价于算法已被科学地证伪；它也不运行拟合、不冻结阈值、不改变输入，且不替用户强制阻止后续命令。

在 T1/T2/R0/P3 证据完整之前，不得据此宣称 P4 ridge/lobe/ellipse 已通过、P9 真实数据最终验收已运行，或从经验椭圆/序列趋势推出唯一三维结构和材料机制。P3 通过也只表示规定证据门通过；真实数据阶段仍须按单独的 P4–P9 验收协议记录适用数据域、失败帧、方法一致性、不确定度和外部证据。
