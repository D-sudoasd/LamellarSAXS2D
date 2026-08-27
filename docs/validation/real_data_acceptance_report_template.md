# 真实实验数据验收报告模板

> 文档状态：`TEMPLATE`。本文件不包含任何真实实验结果，也不表示 P4 或 P9 已运行/通过。复制本模板后，只有在对应证据文件实际生成、哈希复核和人工签字后，才能把状态从 `NOT_RUN` 改为 `PASS`、`WARN` 或 `FAIL`；空白或占位符不等于通过。

## 0. 报告元数据

| 字段 | 待填写内容 |
|---|---|
| `report_id` | `[待填写]` |
| `report_status` | `TEMPLATE` |
| 数据域 | `R0：当前样品/当前仪器/当前处理流程` |
| 样品与实验批次 | `[待填写]` |
| 报告创建时间（含时区） | `[待填写]` |
| 代码版本/commit | `[待填写]` |
| Python 与依赖版本 | `[待填写]` |
| 主要执行者/复核者 | `[待填写]` |
| P4 状态 | `NOT_RUN` |
| P9 状态 | `NOT_RUN` |

## 1. 阶段结论摘要

| 阶段 | 状态（`NOT_RUN`/`PASS`/`WARN`/`FAIL`） | 证据目录/报告 | 备注与适用边界 |
|---|---|---|---|
| R0 只读 preflight | `NOT_RUN` | `[待填写]` | 不运行拟合；不得改写原始输入 |
| P3 T1/T2/R0 证据门 | `NOT_RUN` | `[待填写 p3_gate_report.json]` | 只读汇总，No-Go 不等于算法被证伪 |
| P4 ridge/lobe/ellipse | `NOT_RUN` | `[待填写]` | 未有证据前不要写成已验收 |
| P5 full2d/不确定度 | `NOT_RUN` | `[待填写]` | 记录局部与选择不确定度的分栏 |
| P6 批处理/恢复 | `NOT_RUN` | `[待填写]` | 120 帧分母与失败帧单独记录 |
| P7 GUI/导出 | `NOT_RUN` | `[待填写]` | 记录四视图和导出可复核性 |
| P8 发布候选 | `NOT_RUN` | `[待填写]` | 不把 Alpha 写成正式科研版 |
| P9 R0 最终验收 | `NOT_RUN` | `[待填写]` | 仅适用于当前 R0 数据域，不代表跨域泛化 |

**当前结论（未填前保持原文）：** `[尚未形成验收结论]`

## 2. 输入、单位和只读/provenance 记录

| 输入或证据 | 路径（建议相对项目根目录） | SHA-256（64 位小写） | 运行前 hash | 运行后 hash | unchanged | 备注 |
|---|---|---|---|---|---|---|
| 图像 manifest | `[待填写]` | `[待填写]` | `[待填写]` | `[待填写]` | `NOT_RUN` | 以 manifest 的路径和 selector 为准 |
| RT manifest | `[待填写]` | `[待填写]` | `[待填写]` | `[待填写]` | `NOT_RUN` | `[待填写]` |
| hold manifest | `[待填写]` | `[待填写]` | `[待填写]` | `[待填写]` | `NOT_RUN` | 当前计划为 120 个保温帧，实际以 manifest 为准 |
| PONI | `[待填写]` | `[待填写]` | `[待填写]` | `[待填写]` | `NOT_RUN` | 只有有效 PONI 才能声明物理 q |
| NPY mask | `[待填写]` | `[待填写]` | `[待填写]` | `[待填写]` | `NOT_RUN` | 原始约定 `0=有效、1=无效` |
| project context | `[待填写/不适用]` | `[待填写]` | `[待填写]` | `[待填写]` | `NOT_RUN` | 记录校正和不确定度来源 |
| 代码/schema/依赖清单 | `[待填写]` | `[待填写]` | `[待填写]` | `[待填写]` | `NOT_RUN` | 记录版本与 hash |

### 2.1 坐标、mask 和校正状态

| 项目 | 记录 |
|---|---|
| `q_unit` | `[待填写；物理 q 必须为 nm^-1，否则写 pixel-q/unknown]` |
| `q` 定义与转换 | `[待填写；如有 Å^-1→nm^-1，记录因子 10]` |
| PONI 有效性 | `[待填写：true/false + evidence]` |
| 原始 mask 极性 | `0=有效、1=无效` |
| 内部 `valid_mask` | `True=有效` |
| 外部 mask / ROI | `True=排除/无效` |
| `correction_state` | `[待填写：raw_counts/external_recipe_declared/partially_corrected/fully_corrected_external/unknown]` |
| 已做校正 | `[待填写]` |
| 尚未烧入强度的校正 | `[待填写]` |
| `uncertainty_state` | `[待填写：none/partial/complete/unknown]` |
| 物理周期是否可报告 | `[待填写；pixel-q/unknown 时必须为不可用]` |

## 3. R0 只读 preflight 检查

> 运行 `.\.venv-project\Scripts\bsaxs.exe preflight ...` 后填写。R0 首批不运行 ridge、lobe、ellipse 或 `full2d`；应保留 `fit=null`、`solver_status=not_run` 和相应 `NOT_TESTED` 质量项。

| 检查 ID | 验收问题 | 状态 | 观察值/原因 | 证据路径 |
|---|---|---|---|---|
| R0-PRE-01 | manifest 路径、帧数、角色、shape/dtype 和格式是否明确？ | `NOT_RUN` | `[待填写]` | `[待填写]` |
| R0-GEO-01 | PONI、q-map、q-window 和 q 单位是否明确且有限？ | `NOT_RUN` | `[待填写]` | `[待填写]` |
| R0-MASK-01 | mask shape/极性是否可重算？ | `NOT_RUN` | `[待填写]` | `[待填写]` |
| R0-MAN-01 | order/time/selector 是否明确，且未把 mtime 当正式实验时间？ | `NOT_RUN` | `[待填写]` | `[待填写]` |
| R0-CORR-01 | 校正历史、已做/未做步骤和绝对强度可比性是否明确？ | `NOT_RUN` | `[待填写]` | `[待填写]` |
| R0-UNC-01 | 不确定度来源、单位和覆盖范围是否分栏记录？ | `NOT_RUN` | `[待填写]` | `[待填写]` |
| R0-HASH-01 | 输入、PONI、mask、manifest 运行前后 hash 是否相同？ | `NOT_RUN` | `[待填写]` | `[待填写]` |
| R0-STATUS-01 | 是否无未解释红色项，黄色项是否有原因和证据？ | `NOT_RUN` | `[待填写]` | `[待填写]` |
| R0-SCOPE-01 | 是否保持 `fit=null`、`not_run`、`NOT_TESTED`，且未修改原始数据？ | `NOT_RUN` | `[待填写]` | `[待填写]` |

**R0 结果摘要：** `[未运行；不要填写“通过”]`

## 4. P3 证据与人工盲标记录

| 证据 | 路径 | 当前状态 | SHA-256/指纹 | 备注 |
|---|---|---|---|---|
| T1 `truth_manifest.json` | `[待填写]` | `NOT_RUN` | `[待填写]` | 默认 15 case；q 为 `nm^-1` |
| T2 `truth_manifest.json` | `[待填写]` | `NOT_RUN` | `[待填写]` | `2-point/eyebrow/butterfly/non_elliptical` |
| R0 `annotation_status.json` | `[待填写]` | `NOT_RUN` | `[待填写]` | 完成前应为 `awaiting_human_annotations` |
| thresholds JSON | `[待填写]` | `NOT_RUN` | `[待填写]` | draft/provisional 不能作为最终 Go |
| `p3_gate_report.json` | `[待填写]` | `NOT_RUN` | `[待填写 evidence_fingerprint]` | 门禁只读，不运行拟合 |

### 4.1 盲标内容核对

| 项目 | 实际核对结果 |
|---|---|
| `annotator_a.csv`、`annotator_b.csv`、`consensus_review.csv` 是否各有且仅有 `blind_001`–`blind_008` 八行 | `[待填写]` |
| `valid_area` 是否含至少 3 个不同有限点、面积大于 0，且不存在整行 `unknown`/`[]`/共线点占位 | `[待填写]` |
| 其他标注必填字段是否逐行非空；明确不存在是否写 `[]`、无法判断是否写 `unknown` | `[待填写]` |
| 时间戳是否可解析且带时区 | `[待填写]` |
| 坐标系是否为 `image_pixel_x_right_y_up_origin_lower_left` | `[待填写]` |
| `image_version` 是否等于对应 PNG SHA-256 | `[待填写]` |
| `input_hashes` 是否 `sha256_before == sha256_after` 且 `unchanged=true` | `[待填写]` |
| 是否满足双人独立，或同一专家间隔至少 7 天复标 | `[待填写]` |
| 三个 `evidence_templates` 是否由实际证据替换 `awaiting_*`/`null` 并通过 schema/单位/hash 检查 | `[待填写]` |

冻结阈值的 `evidence_sources` 每条记录还必须填写 `status=complete`、`source`、`sha256`；人工重复性 source 必须保存 8 帧误差并可复算聚合，仪器 source 必须保存测量序列与原始 calibration record 的 hash，pilot source 必须绑定当前 annotation status、consensus CSV 和 8 个逐帧共识状态。wrapper 的 metric/frame_count 必须与 source 匹配。

## 5. P4/P5 代表帧与方法一致性（待执行）

> 本节预置为 `NOT_RUN`，不表示 P4 或 P5 已运行。只有 P3 证据门、人工 consensus 和冻结阈值满足后，才按批准方案执行并填写。

| 计划项 | 状态 | 代表帧/参数 | 证据路径 | 备注 |
|---|---|---|---|---|
| 8 帧 pilot：RT、首/中/末保温帧和至少 4 个困难/负例 | `NOT_RUN` | `[待填写]` | `[待填写]` | 没有某类困难帧要写 `NOT AVAILABLE`，不能制造 |
| ridge/lobe/ellipse 与人工 consensus 对照 | `NOT_RUN` | `[待填写]` | `[待填写]` | 分别报告 precision/recall/F1 或适用指标 |
| q-window、mask 边界和方法扰动 | `NOT_RUN` | `[待填写]` | `[待填写]` | 超出阈值须解释或 WARN/FAIL |
| independent 与 warm-start 对照 | `NOT_RUN` | `[待填写]` | `[待填写]` | 路径依赖需显式标记 |
| 正序与逆序对照 | `NOT_RUN` | `[待填写]` | `[待填写]` | 不得产生超出合并不确定度的系统差 |
| multi-start、withheld 和 residual 检查 | `NOT_RUN` | `[待填写]` | `[待填写]` | 记录 condition、bound、结构化残差 |
| 局部拟合/统计与分析选择不确定度分栏 | `NOT_RUN` | `[待填写]` | `[待填写]` | 不把 stderr 当完整测量误差 |

## 6. P9 完整 120 帧与最终验收（待执行）

> 本节是 P9-E 报告骨架。当前仓库没有借此模板声明 P9-A/B/C/D/E 已运行；每项必须关联实际输出和复核人。

| P9 项目 | 状态 | 必须保留的证据 | 结果/备注 |
|---|---|---|---|
| P9-A 数据包 preflight | `NOT_RUN` | `preflight.json`、`arrays.npz`、`run_report.md`、输入 hash | `[待填写]` |
| P9-B 8 帧代表帧 pilot | `NOT_RUN` | inspect、拟合/质量、人工复核和困难帧说明 | `[待填写]` |
| P9-C 方法一致性 | `NOT_RUN` | radial/surface curvature、q-window、mask、mode、方向、初值、withheld 对照 | `[待填写]` |
| P9-D 120 帧 independent | `NOT_RUN` | frame summary、parameters、uncertainty、失败帧和 provenance | `[待填写]` |
| P9-D 120 帧 warm-start/resume | `NOT_RUN` | checkpoint、lineage、连续/恢复比较 | `[待填写]` |
| P9-E 最终报告 | `NOT_RUN` | `run_report.md`、`acceptance_matrix.csv`、environment/provenance、failed_frames、diagnostics、evolution、limitations | `[待填写]` |

### 6.1 P9 结果数字（未运行前保持空值）

| 指标 | 单位 | 观察值 | 冻结阈值 | 状态 | 证据 |
|---|---|---:|---:|---|---|
| 重复帧表观参数 CV | `%` | `[待填写]` | `[待填写]` | `NOT_RUN` | `[待填写]` |
| pilot ridge/lobe 指标 | `[待填写]` | `[待填写]` | `[待填写]` | `NOT_RUN` | `[待填写]` |
| independent/warm-start 差异 | `[待填写]` | `[待填写]` | `[待填写]` | `NOT_RUN` | `[待填写]` |
| 正序/逆序系统偏差 | `[待填写]` | `[待填写]` | `[待填写]` | `NOT_RUN` | `[待填写]` |
| 全序列可用帧数 / 120 | `frame` | `[待填写]` | `[待填写]` | `NOT_RUN` | `[待填写]` |
| 可用率 | `%` | `[待填写]` | `[待填写]` | `NOT_RUN` | `[待填写]` |
| false PASS | `%` | `[待填写]` | `[待填写]` | `NOT_RUN` | `[待填写]` |

## 7. 科学解释边界与失败项

- 适用数据域（只在证据完成后填写）：`[当前样品/仪器/处理流程]`
- 已通过的内容：`[待填写]`
- 未通过或未测试的内容：`[待填写]`
- `q_unit`、PONI、mask、correction、uncertainty 限制：`[待填写]`
- 尚缺外部证据（例如 WAXD、TEM/AFM、力学或温度日志）：`[待填写]`
- 不能由本报告宣称的内容：唯一三维层片结构、仅凭拟合趋势证明材料机制、跨图样/跨仪器泛化。

## 8. 复核与签字

| 角色 | 姓名 | 日期时间（含时区） | 签字/意见 |
|---|---|---|---|
| 执行者 | `[待填写]` | `[待填写]` | `[待填写]` |
| 数据/仪器复核者 | `[待填写]` | `[待填写]` | `[待填写]` |
| 科学结论批准者 | `[待填写]` | `[待填写]` | `[待填写]` |

**最终报告状态：** `TEMPLATE`（在所有必需证据、hash、人工复核和签字完成前不得改为 `COMPLETE`）。
