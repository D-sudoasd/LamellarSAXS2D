# v2.0 首批验收矩阵：P0–P2、T0 与 R0

> 本文件把《真实实验数据验证完整实施计划.md》（v2.0）中的首批边界落成可执行、可审计的验收合同。它只定义门槛和证据位置，不对真实数据下材料机理结论，也不替代后续 P3–P9 的科研阈值冻结。

## 1. 首批范围与停止边界

### 1.1 首批允许做什么

首批只做以下工作：

1. **P0：**冻结输入、单位、mask、校正状态、结果 schema、质量状态和证据包字段；
2. **P1：**建立可复现开发环境、静态检查和权威测试基线；
3. **P2：**修复 selector、manifest 排序、统一 analysis-domain、配置转发、覆盖语义和单位驱动绘图，并实现 package preflight；
4. **T0：**使用极小公开 fixture 验证 I/O、q-map、mask、JSON、selector 和退出码合同；
5. **R0：**对本地真实数据包做只读 preflight、文件/元数据/hash 检查。

### 1.2 首批明确不做什么

- **首批不进入 P3，不运行 T1/T2 基准拟合；**
- **首批不运行 ridge、lobe、双椭圆或 full2d 拟合；**
- **首批不做 120 帧保温序列拟合，也不做 P9-B pilot 或 P9-D 完整序列分析。**R0 的 1 个 RT 帧和 120 个保温帧可以被枚举、读取元数据和计算 hash，但不得进入拟合；
- 不做独立物理正演、三维结构反演、机制判断或全序列趋势解释；
- 不改写、清洗、覆盖、移动或删除 `data_local/` 下的原始真实数据；不打包、不提交、不上传真实数据和 `CHANGELOG.md`。

因此，首批的“通过”只表示输入与接口合同已满足，**不表示椭圆拟合准确，也不表示 R0 真实数据已经科研验收通过**。

## 2. 状态、门槛和责任约定

### 2.1 状态含义

| 状态 | 颜色 | 含义 | 允许进入拟合吗？ |
|---|---|---|---:|
| `PASS` | `green` | 要求的检查完成，且没有未解释的警告或红色项 | 是，但仍须满足下一阶段门槛 |
| `WARN` | `yellow` | 流程完成，但存在已记录的暂定信息或科学限制；证据必须保留 | 仅在后续阶段明确允许时 |
| `FAIL` | `red` | 检查失败、输入不可信、关键合同不满足或质量门失败 | 否 |
| `NOT TESTED` | — | 按首批边界尚未执行，不得当作通过 | 否 |
| `NOT AVAILABLE` | — | 计划要求的资料在当前数据包中不存在；必须记录原因，不得人工制造 | 否 |

状态优先级为 `red > yellow > green`。任何一个必需检查为 `FAIL`，阶段不得标记为通过。`WARN` 不能被“命令运行成功”“`success=True`”或绿色 GUI 文案覆盖。

### 2.2 负责人标记

- **自动化：**代码、测试、CLI 或 preflight 可重复执行并给出判定；
- **人工：**需要用户/课题负责人确认来源、授权、校正历史、阈值或科学解释边界；
- **自动化 + 人工：**软件先产生证据，人工再确认其科学含义。

### 2.3 证据目录约定

以下是计划中的证据位置；本次文档任务不生成这些运行结果。

| 证据 | 默认位置 |
|---|---|
| P0/P1 基线 | `results/validation/baseline/` |
| P2 preflight | `results/validation/preflight/<run_id>/` |
| T0 fixture | `tests/data/`、`tests/test_preflight.py`、`tests/test_realdata_contract.py` |
| 后续 pilot/拟合 | `results/validation/pilot/<run_id>/`（首批不得写入拟合结果） |
| 单帧 schema 证据包 | `<run_dir>/summary.json`、`arrays.npz`、`quality.json`、`provenance.json` |
| 变更说明 | 项目根目录已有的 `CHANGELOG.md`（由执行者按项目规则维护；不上传） |

每个运行目录必须保留命令、输入路径、输入 hash、配置 hash、代码/schema 版本和运行时间。若输出目录已存在，默认失败；只有用户明确传入 `--force` 才可覆盖输出，且 `--force` 绝不能作用于原始输入。

## 3. P0：冻结范围、输入与验收合同

P0 的输入为当前代码和 README、v2.0 实施计划、本地 R0 数据包及其 PONI/NPY mask/manifests，以及 Grubb 2016/2021 所界定的科学边界。P0 不拟合。

| ID | 检查内容 | 绿色通过门槛（PASS） | 黄色警告门槛（WARN） | 红色失败门槛（FAIL/暂停） | 证据路径 | 负责人 |
|---|---|---|---|---|---|---|
| P0-01 | 版本与范围冻结 | 记录 commit、`git status`、Python/依赖、测试基线；明确当前版本为 Alpha，R0 仅代表单一数据域 | 基线记录尚未完成，但不会被写成已通过 | 未记录版本/工作树状态，或把 Alpha 写成稳定科研版 | `results/validation/baseline/repository_state.txt`、`environment.json` | 自动化 + 人工 |
| P0-02 | R0 输入清单 | manifest、图像、PONI、mask、项目上下文文件的路径、角色和 hash 契约已定义 | 成分、文件名后缀 `_2`、正式时间零点或部分校正来源待用户确认 | PONI 来源、mask 极性、数据授权或输入角色不清 | `docs/validation/acceptance_matrix.md`、后续 `preflight.json` | 自动化 + 人工 |
| P0-03 | q/坐标契约 | 内部 `q=2πs`；物理 q 序列化为 `q_unit="nm^-1"`；Å^-1 输入转换因子为 10 并留 provenance；detector pixel、pixel-q、物理 q 分字段 | q 单位来自暂定/缺失元数据，先降为 `pixel-q` 或 `unknown` 并标黄 | 未标定却输出物理 q、物理周期或把软件角度直接当三维结构角 | `docs/validation/result_schema_v1.md`、`quality.json`、图轴检查 | 自动化 + 人工 |
| P0-04 | mask/analysis-domain 契约 | `valid_mask=True` 表示有效；外部 mask/ROI 的 `True` 表示排除；最终域按计划公式闭合，计数和 NPZ mask 可重算 | 某个可选排除区未提供，但其默认全 False 且已记录 | shape/极性不明、空 q-window、无效权重或计数无法与 mask 闭合 | `docs/validation/result_schema_v1.md`、`arrays.npz`、`quality.json` | 自动化 |
| P0-05 | correction/uncertainty 契约 | `correction_state` 和 `uncertainty_state` 为规定枚举，并记录来源、已做/未做步骤和误差分量 | `external_recipe_declared` + `partial` 等已知限制；不得冒充完整测量误差 | 状态缺失却要比较绝对强度，或软件自动重复上游校正 | `summary.json`、`provenance.json` | 自动化 + 人工 |
| P0-06 | 质量状态与证据包 | 同时保存 `solver_status`、`numerical_status`、`scientific_status`、结构化 flags、阈值和 provenance；`success=True` 不等于 PASS | 暂无拟合时标记 `solver_status=not_run`、`NOT TESTED` | 以低 RMSE/拟合成功掩盖 coverage、边界、病态或单位错误 | `docs/validation/result_schema_v1.md`、`quality.json` | 自动化 |
| P0-07 | 阈值审批 | 固定阈值与暂定科研阈值分开；冻结前由用户审核，不能按算法结果事后放宽 | 阈值仍为 provisional，后续 P3 依据标注/仪器分辨率冻结 | 阈值缺失、不可追溯或为让当前结果通过而修改 | `acceptance_matrix.csv`（后续 P9 输出）、用户确认记录 | 人工 |
| P0-08 | 科学边界 | 明确当前经验模型只给表观几何/经验参数；R0 不能证明 R1 泛化、唯一三维结构或材料机制 | 需要外部 WAXD/TEM/AFM/力学/温度日志，暂不解释 | 计划或报告把单张二维 SAXS 结果写成唯一结构/机制证明 | `limitations.md`（后续）、本文件第 1 节 | 人工 |

**P0 Go：**输入角色、q 单位、mask 极性、校正/不确定度状态、阈值与证据字段均已写清。**P0 No-Go：**任一 P0-02、P0-03、P0-04 或 P0-05 的关键事实无法确认。

## 4. P1：环境、CI 与权威基线

P1 的输入是源码、`pyproject.toml`、测试和可选依赖声明；不得删除旧 `.venv`，建议另建 `.venv-project`。安装依赖可能联网，执行前按项目规则确认。

| ID | 检查内容 | 绿色通过门槛（PASS） | 黄色警告门槛（WARN） | 红色失败门槛（FAIL/暂停） | 证据路径 | 负责人 |
|---|---|---|---|---|---|---|
| P1-ENV-01 | Python/环境 | 选定一个受支持版本（计划示例为 3.11；也可按实际支持的 3.12/3.13），新环境可安装 `.[all]`，`pip check` 返回码 0 | 本机缺少某个候选版本，已改用另一受支持版本并完整记录 | 依赖冲突、解释器路径失效或同一基线混用多个 Python | `results/validation/baseline/python-environment.json`、`pip-freeze.txt` | 自动化 |
| P1-CI-01 | 静态检查 | `compileall -q src`、`ruff check src tests` 均零错误 | 仅存在明确记录的非阻断 lint 提示（不得把错误当警告） | 任一语法错误或 lint 错误 | CI 日志、`results/validation/baseline/` | 自动化 |
| P1-CI-02 | 完整测试 | 全依赖 pytest 零失败；发布必测的 PySide6/pyqtgraph/fabio/pyFAI/h5py 项不因缺依赖静默 skip | 仅非发布必测项因环境原因 skip，且列出测试名和原因 | 发布必测项 skip、测试失败或结果不可重现 | `results/validation/baseline/pytest.txt` | 自动化 |
| P1-REP-01 | 运行可复现 | 保存 pip freeze、Python/平台、命令、测试参数和代码 hash；同环境重跑结果一致 | 时间、临时目录等非科学元数据不同，但数值与状态一致 | 无法重建环境，或只沿用旧的“117 passed, 7 skipped”作为新基线 | `results/validation/baseline/environment.json`、`repository_state.txt` | 自动化 + 人工 |
| P1-SAFE-01 | 原始数据保护 | P1 不触碰 `data_local/`，输入文件 hash 不变 | 仅读取私有数据目录做依赖/路径检查并保留证据 | 删除/覆盖/移动输入或把私有数据放入构建产物 | hash 清单、构建检查 | 自动化 + 人工 |

**P1 Go：**干净环境可安装，依赖、静态检查和必测项均通过，基线证据可复核。**P1 No-Go：**必测依赖缺失、测试静默跳过或环境不能复现。

## 5. P2：阻断性接口与 package preflight

P2 的输入为源码、T0 fixture、R0 数据包及 manifest/PONI/mask；输出为 preflight JSON/报告。P2 的任何错误都必须在拟合前失败关闭（fail closed）。

| ID | 检查内容 | 绿色通过门槛（PASS） | 黄色警告门槛（WARN） | 红色失败门槛（FAIL/暂停） | 证据路径 | 负责人 |
|---|---|---|---|---|---|---|
| P2-SEL-01 | 图像 selector 与 mask selector 隔离 | 图像使用自己的 `frame`/`dataset`；mask 使用自己的 `mask_frame`/`mask_dataset`；HDF5+NPY 与 HDF5+HDF5 均有回归测试 | 单帧 mask 的 selector 为 `null`，但明确表示“不继承图像 selector” | NPY mask 接收到 HDF5 dataset；多帧 mask 未显式选帧；歧义时静默猜测 | `tests/test_p2_io_batch_contracts.py`、`preflight.json` | 自动化 |
| P2-ORDER-01 | manifest 顺序/时间 | `order` 入口即转有限数值；`1,2,10`、前导零和小数顺序正确；缺 order 时按 time，再按原 manifest 顺序 | 时间来自文件 mtime，仅作为 `provisional`，并明确标注 | 非数值/NaN/Inf order、字典序错排、缺失路径或正式时间被 mtime 冒充 | `tests/test_p2_io_batch_contracts.py`、`preflight.json` 的 manifest check | 自动化 |
| P2-DOMAIN-01 | analysis-domain | 对每像素执行 `finite(I,qx,qy,q) ∧ detector_valid ∧ ¬external_mask ∧ ¬ROI_exclusion ∧ q_window ∧ weight_valid`；分解计数与 `fit_valid_mask` 逐点闭合；`sampled_valid_mask ⊆ fit_valid_mask` | 采样比例低但已记录 `sampled_pixel_count`；ROI 未启用且计数为 0 | shape 不一致、q-window 空集、权重非有限/非正、最终域为空或计数不能重算 | `src/butterfly_saxs/validation.py`、`tests/test_validation.py`、`arrays.npz` | 自动化 |
| P2-CONFIG-01 | service 配置转发 | `robust_loss`、`f_scale`、`max_nfev`、`scales`、`seed`、mask、q-window 在 CLI/API/GUI 共享同一配置快照 | 未启用的可选参数保留默认值并记录 | 任一入口丢参数、sigma 与 weights 同时出现、不同入口产生不可解释差异 | `tests/test_service.py`、结果中的 `analysis`/`analysis_domain` | 自动化 |
| P2-UNIT-01 | q 轴与周期单位 | PONI 有效时物理 q 写 `nm^-1`；pixel-q 图轴写 `pixel-q`/未标定，不写 `nm^-1`；只有物理 q 才可导出物理 `d` | q_unit 为 `unknown`，降级为 pixel-q/unknown 并给出警告 | pixel-q 生成物理 `d`、图轴误标 nm^-1、q 单位缺失却继续物理解读 | `tests/test_p2_export_visualization_contracts.py`、图像证据 | 自动化 |
| P2-OUTPUT-01 | no-overwrite/force | 默认目标已存在即返回退出码 2；新 run 目录不覆盖旧证据；`--force` 只显式作用于输出 | 用户明确 `--force`，覆盖行为由调用参数和输出路径共同记录 | 无 `--force` 覆盖输出，或 force 作用于原始输入/`data_local/` | `tests/test_preflight.py`、`tests/test_preflight_cli.py`、输出目录 | 自动化 + 人工 |
| P2-PREFLIGHT-01 | package preflight | 输出图像数量/格式/shape/dtype/finite/负值摘要、selector、PONI/q、mask、manifest、校正/不确定度、hash 和结构化状态 | provisional 时间、partial uncertainty、已声明但未烧入强度的校正步骤均明确为 yellow | 输入不可读、PONI/mask shape 不匹配、hash/selector/unit 错误、未解释 red | `results/validation/preflight/<run_id>/preflight.json`、`arrays.npz`、`run_report.md` | 自动化 + 人工 |
| P2-SAFE-01 | preflight 只读 | preflight 前后输入 hash 完全一致，不自动清洗/归一化/改 mask | 只报告负值和异常高值比例，不改变数组 | 任何输入被写入、替换或静默归一化 | preflight hash 清单 | 自动化 |

**P2 Go：**五项阻断性接口均有回归测试，T0 通过，R0 preflight 没有未解释红色项。**P2 No-Go：**selector、mask、单位或 correction state 仍靠调用方猜测，或 preflight 在输入有问题时仍继续拟合。

## 6. T0：极小 fixture 验收矩阵

T0 是公开、极小、可精确比较的测试数据，不代表真实材料数据。T0 同时包含正例和预期失败例；失败例“按约拒绝”是测试通过，不是把坏输入标为科研 PASS。

| ID | 输入/前置条件 | 检查 | PASS（绿色） | WARN（黄色） | FAIL（红色） | 证据路径 | 负责人 |
|---|---|---|---|---|---|---|---|
| T0-IO-01 | 2-D NPY/NPZ/TIFF 或最小 EDF/HDF5 fixture | 读取 shape、dtype、强度 | shape、dtype、强度逐元素一致；无隐式归一化 | fixture 明确没有可选 header，但不影响数组合同 | 读值改变、shape/dtype 被静默改写、无法识别歧义 | `tests/test_preflight.py`、`tests/test_p2_io_batch_contracts.py`、pytest 输出 | 自动化 |
| T0-GEO-01 | 固定 qx/qy/q/chi 参考数组 | q-map 数值、单位和角度周期 | float64 q 误差 `rtol≤1e-10`、`atol≤1e-12`；float32 按记录的 dtype 机器精度；chi 先按 `atan2(sin Δ, cos Δ)` 比较 | 仅有 pixel-q 参考，明确为未标定坐标 | 直接用未周期化 chi 差判定、单位错写或 q-map shape 不匹配 | q-map 差异报告、`quality.json` | 自动化 |
| T0-MASK-01 | 已知极性/shape 的外部 mask | 0/1 极性、valid mask 和 shape | raw external `0=有效、1=无效`；归一化 `valid_mask=True` 有效；逐点与参考一致 | 无 ROI 时 ROI exclusion 全 False | 极性反转、shape 不一致、被 mask 的像素进入 fit | `arrays.npz`、mask fixture 测试 | 自动化 |
| T0-DOMAIN-01 | 含 finite、detector、q-window、ROI、权重边界的 fixture | 计数分解与最终域 | 所有 count 与 `fit_valid_mask`/`sampled_valid_mask` 精确相符；空域和非法权重按约失败 | 采样是 fit 域子集并且比例已记录 | 任意计数不闭合，或空 q-window 被静默放行 | `quality.json`、`arrays.npz` | 自动化 |
| T0-SEL-01 | HDF5 多帧图像 + NPY/HDF5 mask | 独立 selector | 显式 selector 读到预期帧；NPY 不接受图像 dataset；歧义 fail closed | 单帧源 selector 为 null 且有“不继承”证据 | 错帧、错 dataset 或默默继承 | `tests/test_p2_io_batch_contracts.py` | 自动化 |
| T0-JSON-01 | 含缺失数值和数组摘要的结果对象 | 严格 JSON 序列化 | 标准 JSON 可解析；所有数值有限；缺失使用 `null`；没有 NaN/Inf/字符串 NaN | 数组放在 NPZ sidecar，JSON 只保留摘要 | JSON 写入 NaN/Infinity、重复字段、非法语法或把缺失写成 0 | `summary.json`、schema 测试 | 自动化 |
| T0-EXIT-01 | 正例、质量失败例、输入错误例 | 退出码 | 全部要求门通过为 0；流程完成但 WARN/质量 FAIL 为 1；输入/配置/selector/unit/mask/PONI/覆盖错误为 2 | 预期 yellow 且证据完整，退出 1 | 退出码与状态不一致，或未处理错误冒充 0 | CLI 测试日志 | 自动化 |
| T0-OUTPUT-01 | 已有同名输出目录 | 覆盖保护 | 无 `--force` 不覆盖并返回 2；显式 `--force` 只覆盖输出并留记录 | 新 run ID 写入新目录 | 覆盖原始数据、静默覆盖或无记录 force | 输出目录、provenance | 自动化 + 人工 |

## 7. R0：真实数据包 preflight（首批只读）

R0 输入按 manifest 为准，计划中包括 `images_edf/*.edf`、包内 PONI、最终 NPY mask、RT/hold manifests 和 `project_context.yaml`；当前数据包的预期角色是 1 个 RT 帧加 120 个 375 ℃保温帧。文件名不得由人工猜测。R0 的表观参数、拟合精度和序列趋势在首批均为 `NOT TESTED`。

| ID | 输入/检查 | PASS（绿色） | WARN（黄色） | FAIL（红色/停止） | 证据路径 | 负责人 |
|---|---|---|---|---|---|---|
| R0-PRE-01 | 包完整性与路径 | manifest 中每个路径可解析；帧数、角色、shape/dtype 和文件格式清楚；无重复/缺失路径 | 个别非分析辅助文件未纳入样品计数，但角色有记录 | manifest 缺项、路径指向错误文件、shape 不一致或无法读取 | `results/validation/preflight/<run_id>/preflight.json`、`run_report.md` | 自动化 |
| R0-GEO-01 | PONI 与 q | PONI 可读、geometry 字段完整、q-map 有限且 shape 正确；物理 q 序列化为 `nm^-1`；q-window（计划暂用 `[0.1, 2.0] nm^-1`）明确标注来源 | q-window 仍为 provisional；几何与 header 差异待人工确认；仅可降级为 pixel-q | PONI 不可读、几何字段缺失、q-map 非有限或把 pixel-q 当物理 q | `preflight.json` 的 geometry check、analysis-domain 和 PONI hash | 自动化 + 人工 |
| R0-MASK-01 | NPY mask | mask shape 与图像一致；原始极性为 `0=有效、1=无效`；归一化 valid mask 可重算 | 无法从文件注释确认极性，但由 fixture/用户确认并标注来源 | shape/极性不明或 mask 不能逐点应用；不得继续 | `preflight.json` 的 mask check、`arrays.npz`、mask hash | 自动化 + 人工 |
| R0-MAN-01 | order/time/selector | order 为有限数值并按数值排序；图像 selector 与 mask selector 独立；缺失路径数为 0 | 时间由文件 mtime 构造，`time_source=mtime_provisional`；正式设备日志待补 | order 非数值/非有限、字典序错误、selector 歧义或 mtime 被冒充正式实验时间 | `preflight.json` 的 manifest check 与 frame records | 自动化 + 人工 |
| R0-CORR-01 | correction state | 上游校正历史可追溯，状态写为 `external_recipe_declared`，已做和未做步骤分栏 | 当前计划已声明 dark/background/monitor/transmission/固定厚度/绝对 K 因子，solid-angle/polarization 尚未烧入强度；绝对强度比较不允许 | correction state unknown 却要比较绝对强度，或软件重复做已声明校正 | `project_context.yaml` 摘要、`preflight.json` 的 correction check | 人工 + 自动化 |
| R0-UNC-01 | 不确定度 | 来源、单位和覆盖的误差分量被记录；不把 covariance 当完整测量误差 | 当前计划为 `uncertainty_state=partial`；局部拟合/探测器/选择不确定度尚未完整 | 要求完整不确定度却无来源、单位或分量，仍输出高置信结论 | `preflight.json` 的 uncertainty check | 自动化 + 人工 |
| R0-HASH-01 | 只读/hash | 输入、PONI、mask、manifest hash 记录；运行前后完全一致；结果写新 run 目录 | 重新运行生成新 run ID；旧结果不覆盖 | 任一输入 hash 改变、输出写入输入目录或私有数据进入包 | `preflight.json` 的 `hashes` 清单 | 自动化 |
| R0-STATUS-01 | preflight 汇总状态 | 无未解释 red；所有黄项有 code、原因和证据 | 已知 provisional 时间、partial uncertainty、缺少强度校正步骤均为可解释 yellow；因此整体可能为 `WARN`/退出码 1 | 任一未解释 red、输入错误或关键 metadata 缺失；退出码 2（输入/配置错误） | `preflight.json`、`run_report.md` | 自动化 + 人工 |
| R0-SCOPE-01 | 首批禁止拟合 | 记录 `fit=null`、`solver_status=not_run`、ridge/lobe/ellipse/full2d 为 `NOT TESTED`；没有强度改写 | — | 任何 120 帧拟合、逐帧手调、物理/机制解释或用相邻帧代替重复曝光 | `preflight.json`、运行命令、空的 fit 记录 | 自动化 + 人工 |

R0 只能证明当前样品、仪器和处理流程的数据域接口是否可用，不能替代 R1 的多图样、多仪器或双人盲标，也不能宣称跨数据域泛化。`REAL-02` 重复曝光、`REAL-03` 8 帧 pilot、`REAL-08` 120 帧可用率均在首批标记 `NOT TESTED`。

## 8. 统一退出码与阶段 Go/No-Go

| 退出码 | 适用情形 | 结果证据要求 |
|---:|---|---|
| `0` | 命令完成，且所有要求的质量门为 PASS/green | JSON、NPZ（如适用）、quality、provenance 完整且可解析 |
| `1` | 流程完成，但有 WARN、质量 FAIL、部分失败或可解释的科学限制；证据仍保留 | 必须列出 flags、失败帧/检查、阈值和残留输出；不得把失败写成 0 |
| `2` | 输入、配置、selector、单位、mask、PONI 或输出覆盖错误；无法安全进入分析 | 不得开始拟合；保留可生成的诊断和错误原因 |
| 其他非零 | 未处理的程序错误 | 视为发布阻断，需修复后重跑 |

### 8.1 首批最终 Go

同时满足以下条件才可从首批进入 P3：

- P0 合同已冻结，所有影响科学判定的暂定阈值已标出并提交用户审核；
- P1 clean environment、静态检查和发布必测依赖基线可复现；
- P2 五项阻断性问题均有回归测试；
- T0 正例通过、负例按约拒绝，严格 JSON 不含 NaN/Inf；
- R0 preflight 没有未解释红色项，输入 hash 前后不变；
- 首批没有拟合或修改任何真实强度，没有打包、提交或上传私有数据/`CHANGELOG.md`。

### 8.2 任一即 No-Go/暂停

原始 hash 变化、PONI/q 单位不明、mask 极性或 shape 不明、未知 correction state 却比较绝对强度、selector 歧义、analysis-domain 为空、关键输出将覆盖旧结果、或有人要求在首批直接运行 120 帧拟合/下机制结论。

## 9. 人工签字前最小复核清单

用户/课题负责人在 P0 与 R0 preflight 后至少确认：

- PONI 来源、波长/距离/像素尺寸和物理 q 单位；
- mask 的原始极性、shape 以及 beamstop/streak/坏点含义；
- 上游 correction recipe、绝对强度是否可比、未执行步骤；
- 正式实验时间轴是否可用；mtime 只能作为 provisional；
- R0 数据授权和 `data_local/` 只读边界；
- 暂定阈值是否接受，哪些必须等 P3 标注/仪器分辨率后冻结；
- 明确 R0 结果只限当前数据域，不进入材料机理判断。
