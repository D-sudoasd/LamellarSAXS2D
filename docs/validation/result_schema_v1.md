# `result_schema_v1`：严格 JSON 结果契约

> Schema ID：`lamellarsaxs2d.result.v1`
> 目标适用范围：`preflight`、`inspect`、单帧 `analysis` 和批处理 `batch_frame` 的 JSON 摘要。
> 当前实施边界：第一执行批次 P0–P2 只有 `preflight` 生产者可声明符合本 v1；现有 `inspect`、`analysis`、`batch_frame` 仍是旧接口，须在各自后续阶段迁移并通过同一 validator 后，才能写入 `schema_version="lamellarsaxs2d.result.v1"`。这不是对旧结果的隐式兼容声明。
> 本文件只定义数据结构、状态和解释边界；示例中的路径、hash 和数值均为 schema 示意，不是 R0 真实数据结论。

## 1. 为什么需要这个契约

`solver_status=success` 只能说明优化器返回，不能说明结果适合科研使用。v1 结果必须同时记录：输入选择、q/坐标单位、mask 与 analysis-domain、上游校正状态、不确定度、质量门、输出保护和 provenance。这样才能区分“程序跑完”“数值可用”和“科学上可以报告”。

在 P0–P2 首批中，符合本 v1 的 `result_type` 为 `preflight`，`fit` 必须为 `null`，`solver_status` 必须为 `not_run`；不能借此 schema 偷跑 ridge/lobe/ellipse/full2d，也不能把 120 帧 R0 序列写成已完成拟合。`inspect` 的目标字段仍在本文件冻结，供后续迁移复用，但旧 `inspect` 结果不得伪装成 v1。

## 2. 严格 JSON 序列化规则

以下规则对 `summary.json`、`quality.json` 和 `provenance.json` 均适用：

1. 文件必须是 UTF-8 的单个 JSON object；不允许注释、尾逗号、重复键或 JSON 之外的前后文本。
2. 顶层字段使用本文件规定的名称。v1 不允许静默添加未知顶层字段；未来扩展只能放入显式的 `extensions` object，并同时增加扩展版本。
3. **绝不允许 `NaN`、`Infinity`、`-Infinity` 或字符串形式的 `"NaN"`/`"Inf"`。** Python 实现应等价于 `json.dumps(payload, allow_nan=False)`，并在写文件前完成标准 JSON 解析回读。
4. 数值必须是有限 JSON number；缺失、未测量或不适用使用 `null`，绝不能用 `0` 冒充缺失。数组摘要中的非有限元素也必须计数并报告为 warning/failure，不能直接写入 JSON。
5. shape、计数、frame 和 order 必须为整数或有限数值；hash、状态和单位必须使用规定字符串。路径写字符串，数组写入 NPZ sidecar，不把大数组内嵌到 JSON。
6. JSON 中出现的相对输出路径都相对于 `outputs.directory`；输入路径、PONI、mask 和 manifest 的原始位置必须在 provenance 中保留。输出目录不得是原始数据目录。
7. 所有状态使用大小写敏感的枚举。机器判定看枚举和 flags，不根据自由文本猜测。

建议的最低写入/回读检查：

```python
import json

text = json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2)
json.loads(text)                 # 写文件前检查语法
with open("summary.json", "w", encoding="utf-8", newline="\n") as handle:
    handle.write(text + "\n")
```

## 3. 顶层字段

顶层必需字段如下。`measurements` 和 `fit` 虽然在 preflight 阶段没有内容，也应显式写成 `null`，避免调用方把“缺字段”误认为“没有问题”。

| 字段 | 类型/枚举 | 必需 | 约束 |
|---|---|---:|---|
| `schema_version` | string | 是 | 必须为 `lamellarsaxs2d.result.v1` |
| `result_type` | `preflight` / `inspect` / `analysis` / `batch_frame` | 是 | 表明本对象的运行类型 |
| `run_id` | string | 是 | 每次运行唯一；重跑默认新建，不复用旧结果目录 |
| `created_at` | ISO-8601 string | 是 | 必须带时区，记录生成时间 |
| `tool` | object | 是 | 至少含 `name`、`version` |
| `status` | object | 是 | 总状态、退出码、solver/numerical/scientific 状态和 flags |
| `input` | object | 是 | 输入文件、shape/dtype、强度单位和只读声明 |
| `selector` | object | 是 | 独立记录 image、mask、manifest selector |
| `geometry` | object | 是 | PONI、q 单位、q-window、轴标签和坐标系 |
| `mask` | object | 是 | 原始 mask 来源、极性、shape 和 NPZ mask 关系 |
| `correction_state` | 规定枚举 | 是 | 上游强度校正状态 |
| `correction` | object | 是 | 已做/未做校正步骤及可比性 |
| `uncertainty_state` | 规定枚举 | 是 | 测量/分析不确定度覆盖状态 |
| `uncertainty` | object | 是 | 来源、单位、误差分量和 stderr 范围 |
| `analysis_domain` | object | 是 | 统一像素域、分解计数和 NPZ mask 位置 |
| `quality` | object | 是 | 阈值、逐项检查、质量状态和质量指标 |
| `measurements` | object 或 `null` | 是 | ridge/lobe/ellipse 等表观测量；未运行时为 `null` |
| `fit` | object 或 `null` | 是 | full2d/其他拟合；P0–P2 必须为 `null` |
| `interpretation` | object | 是 | 模型范围、允许结论和禁止越界的解释 |
| `outputs` | object | 是 | 输出文件清单、覆盖策略和 force 记录 |
| `provenance` | object | 是 | 命令、版本、配置和所有关键 hash |
| `extensions` | object | 否 | 仅用于有版本的后续扩展，不能替代 v1 必需字段 |

缺少任何必需字段、字段类型错误或出现未声明的关键顶层字段时，结果不得标记 `PASS`。

## 4. `status`：颜色、质量状态与退出码

### 4.1 结构

```json
{
  "status_color": "green",
  "scientific_status": "PASS",
  "solver_status": "not_run",
  "numerical_status": "NOT_TESTED",
  "exit_code": 0,
  "flags": [],
  "failure_reasons": []
}
```

字段约束：

- `status_color` 只能是 `green`、`yellow`、`red`；
- `scientific_status` 只能是 `PASS`、`WARN`、`FAIL`；
- `solver_status` 只能是 `not_run`、`success`、`failed`、`cancelled`；
- `numerical_status` 为 `PASS`、`WARN`、`FAIL` 或 `NOT_TESTED`；没有运行拟合时使用 `NOT_TESTED`，不能填 `success`；
- `flags` 和 `failure_reasons` 都是结构化 object 数组，每项至少含 `code`、`severity`、`message`；`severity` 使用 `green`/`yellow`/`red`，可选 `evidence` 和 `action`；
- `failure_reasons` 在 `scientific_status=PASS` 时必须为空；有红色项时至少包含一个对应原因。

### 4.2 颜色与 `PASS/WARN/FAIL` 的统一映射

| 颜色 | scientific status | 语义 |
|---|---|---|
| `green` | `PASS` | 所有要求检查通过，没有未解释 warning 或 failure |
| `yellow` | `WARN` | 流程完成，但有已记录的 provisional 元数据、partial uncertainty、覆盖边界或可解释质量限制 |
| `red` | `FAIL` | 输入/单位/mask/analysis-domain 错误，或者质量门失败；不得把数值当作可靠结果 |

状态按 `red > yellow > green` 聚合。`solver_status=success` 不会自动改变 `scientific_status`；例如关键参数触界、留出区失败或残差有系统结构时，仍为 `FAIL`。

### 4.3 退出码

`exit_code` 是命令级结果，不能只由颜色机械推断：

| 退出码 | 含义 | 常见例子 |
|---:|---|---|
| `0` | 流程完成且所有要求质量门通过 | 绿色/PASS 的 T0 正例；没有未解释 warning |
| `1` | 流程完成，但有 WARN、质量 FAIL、部分失败或科学限制；证据仍保留 | R0 preflight 的 provisional 时间或 partial uncertainty；可运行帧的质量 FAIL |
| `2` | 输入、配置、selector、单位、mask、PONI 或输出覆盖错误，无法安全进入分析 | HDF5 dataset 歧义、mask shape 错、pixel-q 请求物理 d、目标已存在且无 `--force` |
| 其他非零 | 未处理的程序错误；视为发布阻断 | traceback、进程崩溃或输出不完整 |

因此，红色质量失败若流程已经完成并保存了证据可返回 `1`；红色输入/配置错误返回 `2`。无论返回 `1` 还是 `2`，都不能继续拟合或宣称 PASS。

## 5. 输入、selector 与 hash

### 5.1 `input`

`input` 至少包含：

```json
{
  "source_kind": "image|manifest|fixture",
  "images": [
    {
      "path": "相对或绝对输入路径",
      "sha256": "64 位小写十六进制",
      "frame_id": "可选帧标识",
      "shape": [1024, 1024],
      "dtype": "float32"
    }
  ],
  "manifest_path": null,
  "intensity_unit": "cm^-1",
  "read_only": true
}
```

真实数据 `read_only` 必须为 `true`。缺少强度单位不能偷偷写成 `cm^-1`；应写 `null` 并产生 warning，涉及绝对强度比较时阻断。

### 5.2 `selector`：图像和 mask 永不隐式继承

```json
{
  "image": {
    "path": "images/frame_0001.h5",
    "frame": 0,
    "dataset": "entry/data"
  },
  "mask": {
    "path": "masks/bl19b2_mask.npy",
    "frame": null,
    "dataset": null
  },
  "manifest": {
    "path": "project_hold_375C_manifest.csv",
    "row_id": "hold-0001",
    "frame_id": "0001",
    "order": 1,
    "time": null,
    "time_unit": "s",
    "time_source": "mtime_provisional"
  }
}
```

具体规则：

- 图像只使用 `selector.image.frame` 和 `selector.image.dataset`；mask 只使用自己的 `selector.mask.frame` 和 `selector.mask.dataset`。
- NPY 单帧 mask 的 `mask.frame`/`mask.dataset` 可以为 `null`，但语义是“没有 mask selector”，**不是继承图像 selector**。
- 多帧 mask 必须显式给出 `mask.frame`；多 dataset mask 必须显式给出 `mask.dataset`。歧义时 fail closed，返回退出码 2。
- 图像与 mask 同为 HDF5 时，两个 dataset 也必须分别记录；HDF5 图像 selector 不得传给 NPY mask。
- `manifest.order` 一旦存在必须转换为有限数值；`1, 2, 10` 按数值顺序，不按字符串排序。`NaN`、`Inf`、非数值 order 为 FAIL。没有 order 时按 time，再按原 manifest 顺序；mtime 只能写 `mtime_provisional`。

### 5.3 `provenance.hashes`

至少记录：输入图像、PONI、mask、manifest、配置、代码、schema 和依赖环境。建议结构：

```json
{
  "algorithm": "SHA-256",
  "files": [
    {"role": "image", "path": "...", "sha256": "64 位小写十六进制"},
    {"role": "poni", "path": "...", "sha256": "64 位小写十六进制"},
    {"role": "mask", "path": "...", "sha256": "64 位小写十六进制"},
    {"role": "manifest", "path": "...", "sha256": "64 位小写十六进制"}
  ],
  "config_sha256": "64 位小写十六进制",
  "code_sha256": "64 位小写十六进制",
  "schema_sha256": "64 位小写十六进制",
  "dependencies_sha256": "64 位小写十六进制"
}
```

实际 JSON 中 hash 值必须是 64 个小写十六进制字符；不能写“待定”或伪造 hash。若当前阶段无法计算某项，写 `null`、增加结构化 warning，并且不得把该结果标绿。真实数据运行还必须记录 `input_unchanged.before`、`input_unchanged.after` 或等价的前后 hash 比较，确认完全一致。

## 6. `geometry` 与单位规则

### 6.1 PONI、q-map 和规范单位

`geometry` 至少包含：

```json
{
  "poni": {
    "path": "config/geometry/BL19B2_SAXS_Califile.poni",
    "sha256": "64 位小写十六进制",
    "valid": true
  },
  "q_unit": "nm^-1",
  "source_q_unit": null,
  "q_conversion_factor_to_nm_inv": 1.0,
  "coordinate_system": "physical_q",
  "q_window": {"min": 0.1, "max": 2.0, "unit": "nm^-1"},
  "axis_labels": {"qx": "qx (nm^-1)", "qy": "qy (nm^-1)"}
}
```

规则如下：

- 内部约定为 `q = 2πs`；物理 q 的 v1 规范字符串为 **`nm^-1`**。上游写 `Å^-1` 时，先乘以 10 转为 `nm^-1`，并在 `source_q_unit`、转换因子和 provenance 中留痕。
- **PONI 物理 q 必须是 `q_unit="nm^-1"` 且 `poni.valid=true`。**PONI 不可读、几何不完整或单位未确认时，不能把数组命名为物理 q；使用 `pixel-q` 或 `unknown`，并产生 warning/failure。
- detector pixel、pixel-q 和物理 q 必须使用不同字段名。推荐：`x_pixel/y_pixel`、`qx_pixel_q/qy_pixel_q/q_pixel_q`、`qx_nm_inv/qy_nm_inv/q_nm_inv`；不得把一个无单位的 `q` 数值在不同上下文中重复解释。
- 物理 q 图轴只能写 `q (nm^-1)` 或等价明确标签；`pixel-q` 图轴写 `pixel-q`/`pixel-q (a.u.)`，不得写 `nm^-1`。
- 只有 `q_unit="nm^-1"`、PONI 有效、`q>0` 且使用的表观周期假设已在 flags 中声明时，才允许输出 `d_nm = 2π/q_nm_inv`。`d_nm` 是由 q 得到的表观长度，不自动等同于层片厚度或唯一结构周期。
- **`q_unit="pixel-q"` 时禁止输出物理 `d`：**`d_nm`、`spacing_nm`、`L_N_nm` 等物理周期字段必须省略或为 `null`，不能是数字；若调用方请求它，返回红色 FAIL/退出码 2。
- `q_unit="unknown"` 时同样不得输出物理周期。任何已有的兼容字段都必须带 `spacing_unavailable_unknown_q_unit` 等 flag，不能通过字段名推断单位。

### 6.2 NPZ 数组命名

JSON 只保存摘要，完整数组放在 `arrays.npz`。`geometry.arrays_npz` 和 `analysis_domain.arrays_npz` 必须列出实际 key；不能让调用方猜 key。建议使用单位限定 key：

| 坐标状态 | 必需 q-map key |
|---|---|
| `physical_q` | `qx_nm_inv`、`qy_nm_inv`、`q_nm_inv`、`chi_rad` |
| `pixel_q` | `qx_pixel_q`、`qy_pixel_q`、`q_pixel_q`、`chi_rad` |
| detector pixel | `x_pixel`、`y_pixel`（不能当作 q） |

如果实现暂时保留兼容 key `qx/qy/q`，必须在 `arrays_npz.key_units` 中逐项声明单位，且不能同时出现相互矛盾的 scoped key。

## 7. mask 与统一 analysis-domain

### 7.1 mask 极性

`mask` 至少包含：

```json
{
  "source": {
    "path": "masks/bl19b2_mask.npy",
    "sha256": "64 位小写十六进制",
    "raw_polarity": "0_valid_1_invalid"
  },
  "shape": [1024, 1024],
  "valid_mask_polarity": "true_valid",
  "external_mask_polarity": "true_invalid",
  "roi_exclusion_polarity": "true_invalid"
}
```

统一语义：raw NPY mask 的 `0=有效、1=无效`；内部 `valid_mask=True` 表示有效；`external_mask=True` 和 `ROI_exclusion=True` 表示排除。没有外部 mask 时，外部排除 mask 为同 shape 的全 False，不能省略极性说明。

### 7.2 `analysis_domain`

`analysis_domain` 至少包含：

```json
{
  "schema_version": "lamellarsaxs2d.analysis_domain.v1",
  "status": "computed",
  "image_shape": [1024, 1024],
  "q_window": {"min": 0.1, "max": 2.0, "unit": "nm^-1"},
  "weight_kind": "none",
  "counts": {
    "image_pixel_count": 1048576,
    "finite_pixel_count": 1048576,
    "detector_valid_count": 1048000,
    "external_mask_excluded_count": 1000,
    "external_valid_count": 1047000,
    "q_window_pixel_count": 900000,
    "roi_excluded_count": 100,
    "weight_invalid_count": 0,
    "fit_pixel_count": 899900,
    "sampled_pixel_count": 899900
  },
  "arrays_npz": {
    "path": "arrays.npz",
    "keys": {
      "finite_mask": "finite_mask",
      "detector_valid_mask": "detector_valid_mask",
      "external_valid_mask": "external_valid_mask",
      "q_window_mask": "q_window_mask",
      "roi_exclusion_mask": "roi_exclusion_mask",
      "weight_valid_mask": "weight_valid_mask",
      "fit_valid_mask": "fit_valid_mask",
      "sampled_valid_mask": "sampled_valid_mask"
    }
  }
}
```

最终拟合域必须逐像素满足：

```text
fit_valid_mask = finite(I, qx, qy, q)
                  AND detector_valid
                  AND NOT external_mask
                  AND NOT ROI_exclusion
                  AND (q_min <= q <= q_max)
                  AND weight_valid
```

计数定义和约束：

| 字段 | 定义/约束 |
|---|---|
| `image_pixel_count` | `prod(image_shape)` |
| `finite_pixel_count` | `finite(I,qx,qy,q)` 的像素数 |
| `detector_valid_count` | finite stage 与 detector-valid 的交集 |
| `external_mask_excluded_count` | detector-valid 数减去 external-valid 数 |
| `external_valid_count` | detector-valid 且不在 external mask 的像素数 |
| `q_window_pixel_count` | external-valid 且 q 落入闭区间的像素数 |
| `roi_excluded_count` | q-window stage 中被 ROI 排除的像素数 |
| `weight_invalid_count` | ROI 后 sigma/weights 非有限或非正的像素数；这类输入应快速失败 |
| `fit_pixel_count` | `count_nonzero(fit_valid_mask)` |
| `sampled_pixel_count` | `count_nonzero(sampled_valid_mask)`；必须满足 `sampled_valid_mask ⊆ fit_valid_mask` |

当 `analysis_domain.status="computed"` 时，NPZ 中必须有同 shape、布尔 dtype 的 `fit_valid_mask` 和 `sampled_valid_mask`，并能重算全部非空计数；任意 mismatch 为 FAIL。`status="not_run"` 只允许用于尚未建立域的 preflight/inspect，此时计数写 `null` 或明确可得的计数，不能用 0 假装没有像素。普通 analysis 的 q-window、fit domain 为空时必须快速失败。

`sigma` 与 `weights` 互斥；`weight_kind` 只能为 `none`、`sigma` 或 `weights`。权重不会静默删除像素；非法权重在 analysis-domain 内应返回输入/配置错误。

## 8. `correction_state` 与 `uncertainty_state`

### 8.1 correction state

`correction_state` 只能是：

| 值 | 含义 |
|---|---|
| `raw_counts` | 原始计数，未声明上游校正 |
| `external_recipe_declared` | 上游提供了校正 recipe；软件只记录，不重复执行 |
| `partially_corrected` | 已知只完成部分校正；剩余步骤必须列出 |
| `fully_corrected_external` | 上游明确完成全部约定校正，并提供可追溯证据 |
| `unknown` | 无法确认校正历史 |

`correction` object 至少包含 `source_files`、`declared_steps`、`not_applied_steps`、`software_reapply_prohibited` 和 `absolute_intensity_comparable`。软件不得因为 state 缺失而自动重复 dark、background、monitor、transmission、厚度或绝对 K 因子。`unknown` 至少为 WARN；如果任务比较绝对强度或关键校正步骤未确认，必须 FAIL/退出码 2。当前 R0 计划状态应记录为 `external_recipe_declared`，并把已做的 dark/background/monitor/transmission/固定厚度/绝对 K 因子与尚未烧入强度的 solid-angle/polarization 分开列出；这不是完整绝对强度验证。

### 8.2 uncertainty state

`uncertainty_state` 只能是：

| 值 | 含义 |
|---|---|
| `none` | 没有可用不确定度估计；不能显示完整误差条 |
| `partial` | 只覆盖部分来源，例如局部 covariance 或部分 detector 误差 |
| `complete` | 计划所需的噪声、仪器、mask/q-window/背景/初值等来源均有证据 |
| `unknown` | 来源或覆盖范围无法确认 |

`uncertainty` object 至少包含 `sources`、`components`、`units`、`stderr_scope` 和 `separate_from_selection_uncertainty=true`。局部拟合 covariance、bootstrap/Monte Carlo 区间、探测器测量不确定度和分析选择敏感性必须分栏；不能把局部 `stderr` 冒充完整测量误差。`partial`/`unknown` 通常为 WARN；当用户要求完整不确定度而来源缺失时为 FAIL。当前 R0 计划状态为 `partial`，不能把它写成完整测量误差。

## 9. `quality`、阈值和拟合字段

### 9.1 quality 结构

```json
{
  "status": "WARN",
  "status_color": "yellow",
  "thresholds_version": "p0-p2-contract-v1",
  "checks": [
    {
      "id": "P2-SEL-01",
      "name": "image_mask_selector_independent",
      "status": "PASS",
      "status_color": "green",
      "observed": true,
      "threshold": true,
      "comparison": "equals",
      "evidence": "tests/test_p2_io_batch_contracts.py"
    }
  ],
  "flags": [],
  "metrics": {
    "coverage": null,
    "condition_number_scaled": null,
    "bound_flags": [],
    "fit_ndata": null,
    "sampled_n": null,
    "withheld": null,
    "residual": null
  }
}
```

每个 `checks[]` 必须有稳定 `id`、`status`、`status_color`、`observed`、`threshold`、比较关系和 evidence。`thresholds_version` 区分固定接口门槛与尚待 P3 依据真值/标注/仪器分辨率冻结的科研阈值；不能为了让算法通过而事后放宽。

`checks[].status` 允许 `PASS`、`WARN`、`FAIL` 或 `NOT_TESTED`。`NOT_TESTED` 仅表示按阶段边界尚未执行，`status_color` 应为 `null`，不得聚合为 PASS；例如 P0–P2 的拟合、留出区和不确定度覆盖检查必须这样记录。`quality.status` 仍只使用 `PASS`、`WARN` 或 `FAIL`，由已执行检查和阶段允许的未测试项共同决定。

拟合结果（P0–P2 不运行）还应保留：fit/withheld normalized RMSE 或适用的 Poisson deviance、残差均值/尺度/空间结构、scaled-Jacobian condition、参数相关性、multi-start 解族和 `bound_flags`。条件数只有在无量纲参数缩放后才可解释；fixed/Expr 参数不进入自由列空间；非有限 condition 或关键参数触界不得 PASS。没有可靠计数/误差模型时，必须明确使用 robust relative metric，不能把 RMSE 当统计检验。

### 9.2 `fit` 与 `measurements`

- `fit=null` 表示没有运行拟合，不表示拟合失败，也不表示通过；此时 `status.solver_status=not_run`，相关质量项为 `NOT_TESTED`。
- 若 `fit` 非空，至少含 `ran=true`、模型范围、参数表、solver message、fit/withheld 域和局部/选择不确定度分栏。
- 当前经验模型只能写 `model_scope="empirical"`；物理正演完成并单独验证后才可写 `model_scope="physical_forward"`。不得把经验 full2d 写成物理模型或唯一反演。
- 参数名必须携带单位或使用明确后缀：detector pixel 用 `_pixel`，pixel-q 用 `_pixel_q`，物理 q 用 `_nm_inv`，物理长度用 `_nm`。例如 `center_pixel_x`、`q_star_pixel_q`、`q_star_nm_inv` 和 `d_nm` 不能互换。
- ridge 点必须保留 `pixel_x/pixel_y`、单位明确的 q 值、角度参考轴、`accepted`、`reason` 和 point-level flags；缺失分支不得用对称复制制造有效点。
- `theta_deg`、`phi_app_deg` 等表观角度必须带参考轴和 `apparent_geometry_only`/`nonunique_inverse_problem` flags，不能直接写成论文中的三维 `phi/alpha/psi`。

## 10. `interpretation`：科学解释边界

```json
{
  "model_scope": "empirical",
  "interpretation_limit": "apparent_geometry",
  "claims_allowed": [
    "q/lobe/ridge 的表观测量",
    "经验双椭圆参数及其质量状态"
  ],
  "claims_forbidden": [
    "单张二维 SAXS 唯一三维层片结构",
    "仅凭拟合趋势证明相变、位错、层片旋转或变形机制",
    "把表观椭圆半轴直接等同材料真实厚度/周期"
  ],
  "flags": ["apparent_geometry_only", "nonunique_inverse_problem"]
}
```

`interpretation_limit` 至少区分 `apparent_geometry`、`nonunique_inverse_problem` 和 `physical_forward_comparison`。R0 在 R1、外部表征和物理正演完成前只能是前两者；`claims_forbidden` 不能因 `scientific_status=PASS` 而删除。

## 11. `outputs`：默认不覆盖，`--force` 明确授权

`outputs` 至少包含：

```json
{
  "directory": "results/validation/preflight/20260827T120000+0800_abc123",
  "paths_relative": true,
  "files": {
    "summary_json": "summary.json",
    "quality_json": "quality.json",
    "provenance_json": "provenance.json",
    "arrays_npz": "arrays.npz"
  },
  "overwrite": false,
  "force": false,
  "overwritten_paths": []
}
```

规则：

- 默认 `overwrite=false`、`force=false`。目标文件或目录已有内容时，命令返回 2，不删除、不静默覆盖；建议生成新的 `run_id`/run 目录。
- 只有用户显式传入 `--force` 才允许覆盖**输出**，并将 `force=true` 和具体 `overwritten_paths` 写入 provenance；force 不能作用于原始图像、PONI、mask、manifest、`data_local/` 或 `CHANGELOG.md`。
- `outputs.files` 必须列出实际写出的证据文件；失败流程不能把缺失文件写成空白或零值冒充完整证据。
- 单帧完整证据包通常包括 `summary.json`、`arrays.npz`、`parameters.csv`、`quality.json`、`provenance.json` 和 observed/model/residual/overlay 图；preflight 可以没有 model/residual，但必须明确 `fit=null`。

## 12. `provenance` 最小字段

```json
{
  "command": "bsaxs preflight ...",
  "working_directory": "项目目录",
  "git_commit": "commit 或 null",
  "dependencies": {"python": "3.11.x", "pyFAI": "版本或 null"},
  "hashes": {
    "algorithm": "SHA-256",
    "files": [],
    "config_sha256": "64 位小写十六进制",
    "code_sha256": "64 位小写十六进制",
    "schema_sha256": "64 位小写十六进制",
    "dependencies_sha256": "64 位小写十六进制"
  },
  "input_unchanged": {"checked": true, "before_after_equal": true},
  "privacy": {"source_data_local": true, "upload_allowed": false}
}
```

P0–P2/R0 运行必须记录实际命令、环境、PONI/mask/manifest/config/code/schema hash、输入前后 hash 和只读结果。私有真实数据只留在 `data_local/`；结果和文档不应把真实图像复制进仓库、安装包或远程同步。

## 13. 最小完整示例（仅 schema 示意）

以下对象是一个 pixel-q T0 preflight 例子，故没有任何物理 `d_nm` 字段；它不代表 R0，也不代表真实实验已通过。

```json
{
  "schema_version": "lamellarsaxs2d.result.v1",
  "result_type": "preflight",
  "run_id": "example-t0-0001",
  "created_at": "2026-08-27T12:00:00+08:00",
  "tool": {"name": "LamellarSAXS2D", "version": "0.1.0"},
  "status": {
    "status_color": "yellow",
    "scientific_status": "WARN",
    "solver_status": "not_run",
    "numerical_status": "NOT_TESTED",
    "exit_code": 1,
    "flags": [
      {"code": "uncalibrated_pixel_q", "severity": "yellow", "message": "示例使用未标定 pixel-q", "evidence": "geometry.q_unit"}
    ],
    "failure_reasons": []
  },
  "input": {
    "source_kind": "fixture",
    "images": [{"path": "tests/data/t0/image.npy", "sha256": "0000000000000000000000000000000000000000000000000000000000000000", "frame_id": null, "shape": [2, 2], "dtype": "float32"}],
    "manifest_path": null,
    "intensity_unit": null,
    "read_only": true
  },
  "selector": {
    "image": {"path": "tests/data/t0/image.npy", "frame": null, "dataset": null},
    "mask": {"path": "tests/data/t0/mask.npy", "frame": null, "dataset": null},
    "manifest": {"path": null, "row_id": null, "frame_id": null, "order": null, "time": null, "time_unit": null, "time_source": "unknown"}
  },
  "geometry": {
    "poni": {"path": null, "sha256": null, "valid": false},
    "q_unit": "pixel-q",
    "source_q_unit": null,
    "q_conversion_factor_to_nm_inv": null,
    "coordinate_system": "pixel_q",
    "q_window": {"min": 0.0, "max": 2.0, "unit": "pixel-q"},
    "axis_labels": {"qx": "qx (pixel-q)", "qy": "qy (pixel-q)"},
    "arrays_npz": {"path": "arrays.npz", "keys": {"qx": "qx_pixel_q", "qy": "qy_pixel_q", "q": "q_pixel_q", "chi": "chi_rad"}, "key_units": {"qx_pixel_q": "pixel-q", "qy_pixel_q": "pixel-q", "q_pixel_q": "pixel-q", "chi_rad": "rad"}}
  },
  "mask": {
    "source": {"path": "tests/data/t0/mask.npy", "sha256": "1111111111111111111111111111111111111111111111111111111111111111", "raw_polarity": "0_valid_1_invalid"},
    "shape": [2, 2],
    "valid_mask_polarity": "true_valid",
    "external_mask_polarity": "true_invalid",
    "roi_exclusion_polarity": "true_invalid"
  },
  "correction_state": "raw_counts",
  "correction": {"source_files": [], "declared_steps": [], "not_applied_steps": [], "software_reapply_prohibited": true, "absolute_intensity_comparable": false},
  "uncertainty_state": "none",
  "uncertainty": {"sources": [], "components": {}, "units": null, "stderr_scope": "none", "separate_from_selection_uncertainty": true},
  "analysis_domain": {
    "schema_version": "lamellarsaxs2d.analysis_domain.v1",
    "status": "computed",
    "image_shape": [2, 2],
    "q_window": {"min": 0.0, "max": 2.0, "unit": "pixel-q"},
    "weight_kind": "none",
    "counts": {"image_pixel_count": 4, "finite_pixel_count": 4, "detector_valid_count": 4, "external_mask_excluded_count": 1, "external_valid_count": 3, "q_window_pixel_count": 3, "roi_excluded_count": 0, "weight_invalid_count": 0, "fit_pixel_count": 3, "sampled_pixel_count": 3},
    "arrays_npz": {"path": "arrays.npz", "keys": {"finite_mask": "finite_mask", "detector_valid_mask": "detector_valid_mask", "external_valid_mask": "external_valid_mask", "q_window_mask": "q_window_mask", "roi_exclusion_mask": "roi_exclusion_mask", "weight_valid_mask": "weight_valid_mask", "fit_valid_mask": "fit_valid_mask", "sampled_valid_mask": "sampled_valid_mask"}}
  },
  "quality": {"status": "WARN", "status_color": "yellow", "thresholds_version": "p0-p2-contract-v1", "checks": [], "flags": [], "metrics": {"coverage": null, "condition_number_scaled": null, "bound_flags": [], "fit_ndata": null, "sampled_n": null, "withheld": null, "residual": null}},
  "measurements": null,
  "fit": null,
  "interpretation": {"model_scope": "empirical", "interpretation_limit": "apparent_geometry", "claims_allowed": [], "claims_forbidden": ["物理周期", "唯一三维结构", "材料机制"], "flags": ["apparent_geometry_only", "nonunique_inverse_problem"]},
  "outputs": {"directory": "results/validation/preflight/example-t0-0001", "paths_relative": true, "files": {"summary_json": "summary.json", "quality_json": "quality.json", "provenance_json": "provenance.json", "arrays_npz": "arrays.npz"}, "overwrite": false, "force": false, "overwritten_paths": []},
  "provenance": {"command": "bsaxs preflight <fixture>", "working_directory": ".", "git_commit": null, "dependencies": {}, "hashes": {"algorithm": "SHA-256", "files": [], "config_sha256": null, "code_sha256": null, "schema_sha256": null, "dependencies_sha256": null}, "input_unchanged": {"checked": true, "before_after_equal": true}, "privacy": {"source_data_local": false, "upload_allowed": true}}
}
```

## 14. 实现验收检查

实现者应至少用 T0 正例、预期失败例和 R0 只读 preflight 验证：

- `json.dumps(..., allow_nan=False)` 与标准 JSON 回读均成功，任何 NaN/Inf 都被拒绝；
- image/mask selector 独立，HDF5+NPY、HDF5+HDF5 和多帧 mask 均按显式 selector 工作；
- analysis-domain 的 8 类 mask/count 与 NPZ `fit_valid_mask`/`sampled_valid_mask` 可逐项重算；
- PONI 物理 q 使用 `nm^-1`，pixel-q 图轴不含 `nm^-1`，pixel-q 请求 `d_nm` 返回 FAIL/退出码 2；
- `correction_state`、`uncertainty_state`、flags、quality status 和退出码一致；
- 默认输出不覆盖，已有目标无 `--force` 返回 2；force 的覆盖路径可审计且不触碰原始数据；
- P0–P2 首批中 `fit=null`、`solver_status=not_run`，没有 120 帧拟合或材料机理结论。
