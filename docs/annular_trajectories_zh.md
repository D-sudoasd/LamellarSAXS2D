# 固定 q 环的 `I(χ)` 花瓣轨迹

`annular_peak` 是当前 GUI 新会话的主蝴蝶路径。它把 q 窗口划成一组固定的 q 环，在每个 q 环上计算角向 profile `I(χ)`，从每个 profile 中提取实际支持的角向峰，再沿 q 方向连接成最多四条、支持充分时呈四条的长花瓣弧。这个观测定义对应用户在二维图上看到的 4 条蓝色花瓣轨迹；它不再把每个 bar 的局部最亮点直接当作主轨迹。

固定 χ 的 `radial_sector` 仍可用于独立的 `I(q)` 径向诊断，但其 `q*` 不属于默认 annular 主轨迹，也不是默认主椭圆输入。局部二维曲率 `curvature` 保留为历史项目和高级诊断路径。

## 默认采样与原始 profile

GUI 新建会话默认使用 40 个 q 环和 72 个方位角分箱。实际有效 q 像素会按 q 环和 χ 分箱；每个 q 环的原始角向强度定义为：

```text
I_raw(χₗ | qₖ) = sum(Iₚ) / count(Iₚ)
```

其中 `p` 只包括当前 mask、q 窗口和有限强度允许的像素。每个 q 环的 `raw_mean`、`raw_sum`、`counts` 和 `coverage` 都保留在结果中；`raw_sum` 不应脱离 count 单独比较。空分箱保持为空，不用零或镜像数据填充。

环积分层不额外施加探测器效率、固角、偏振或绝对强度校正，也不把输入强度重新解释为另一种 pyFAI 校正积分。输入数据已有的校正和强度单位随上下文记录。q 环中心来自 PONI 的 q 坐标；若只有 pixel-q，只能按采样坐标解释。

每个 q 环最多保留四个实际观测峰，每个参考象限最多一个 dominant 峰。少于四个时保留实际存在的峰，不补齐缺失象限；某个 q 环没有可靠峰时保留 profile 和缺环状态，不跨过缺环强行连接。当前默认最小连续轨迹长度为 3 个 q 环，短于该长度的候选不升级为连续花瓣弧。

## 峰筛选和轨迹连接

一个角向候选需要在该环上有足够 coverage、连续的角向支持、局部突出度和噪声支持，并通过有效像素贡献与热像素主导检查。相邻候选峰过于接近或强度证据不足时保留歧义/拒绝原因；程序不会为了得到四瓣而强行选峰。

轨迹只连接相邻 q 环中同一参考象限的实际峰，并限制角度跳变。mask、beam stop、探测器缝隙或低 coverage 造成的断开会原样记录。`I(χ)` profile 以及每个环的 candidates 仍可在 GUI 和离线图包中检查，即使该环最终没有进入花瓣弧。

当前真实 frame110/120 中看到的 point 数和 4 条 arc 只是某一组 q 环/角度分箱下的调校示例。它们不应写成最终验证数字、普适结构计数或科学接受结论；改变 q 窗口、mask、PONI 或分箱会改变可见支持。

## 固定对角配对与椭圆拟合

轨迹建立前先按用户给定的参考轴固定象限和 branch：

- family 0：参考轴内的 QI 与 QIII；
- family 1：参考轴内的 QII 与 QIV。

这是一种固定的对角配对。拟合接收已经分配好的 branch、side 和 arc 身份，不能在优化过程中把某一条花瓣重新指派给另一个 individual petal，也不能靠最近距离替换缺失象限。配对、缺口和支撑状态会写入结果诊断。

拟合得到的双椭圆仍是观测几何候选，不是唯一的三维结构反演。长轴、倾角和 `Ln/Lz` 只有在真实二维支持足够且参数可辨识时才可解释；求解器成功、残差较小或图形完整都不等于科学验收。

## q 环坐标和长轴边界

`q_annulus` 是预先规定的 q 环采样坐标，只说明该点来自哪个 q 环。它不是径向强度峰 `q*`，不能用 `2π/q_annulus` 换算层片间距，也不能自动命名为一阶反射。

分析 q 窗口的外边界是采样范围，不是真实长轴尖端。annular 路径已禁用共享的 `observed_tip_constraint`，避免把 q 窗口边界塞进长轴估计；用户明确给出的参数上下界仍然保留。若 q 支持不足以辨识很长的 `a`，结果应标记为不可辨识、ring-only 或候选限制状态，不强报长轴、`Ln` 或 `L`。

若至少两个独立侧边的已接受轨迹仍延伸到最外侧 q 环，`diagnostics.outer_window_truncated` 会记录这一事实，`quality.flags` 和候选拟合标记 `annular_outer_window_truncated`。此时分析窗口没有包住花瓣外端，长轴状态为 `not_identified`，椭圆参数只保留为未发表候选。逐弧留出检验使用与主拟合相同的无端点先验，不能在交叉检查时把窗口边界重新当成观测尖端。

候选拟合的 `symmetry` 还记录四象限计数、两对相对侧边的配对数和中心对称偏差。由于相对点可能来自同一个预设 q 环，`q_difference_independent=false`；径向差值不能单独作为独立的对称性证据。
对称性偏差采用候选拟合中心计算，并以 `center_source` 记录来源；数值中心存在不等于中心经过独立实验验证，因此 `center_verified=false`。若用户显式指定拓扑中心，轨迹诊断另记 `center_q_source=explicit_option`，也不自动提升为已验证。

## GUI、CLI 与旧项目

新建 GUI 会话的识别方式是 `annular_peak`，控制栏默认显示 q 环数量 40 和方位角分箱 72。历史项目配方缺少 `trace_method` 时，继续使用历史 `curvature` 语义；历史 `ridge_method=butterfly_curvature` 的 family、象限和分支标记仍保留。需要切换到 annular 时应明确选择方法并重新识别。

CLI 的主路径写法为：

```powershell
bsaxs analyze data/frame_0001.edf `
  --poni geometry/detector.poni `
  --mask masks/detector.npy `
  --butterfly-stage evaluate `
  --butterfly-trace-method annular_peak `
  --annular-rings 40 `
  --annular-angles 72 `
  --butterfly-resamples 0 `
  -o results/frame_0001_annular
```

`--annular-rings` 是请求的 q 环数量，实际数量仍受 q 像素采样限制；`--annular-angles` 是每个 q 环的方位角分箱数。显式指定 `--butterfly-trace-method` 时，CLI 会选择蝴蝶工作流；不要与其他 `--ridge-method` 混用，除非该值也是 `butterfly_curvature`。`radial_sector` 的 `--sector-width`/`--sector-step` 只改变独立径向诊断。

## 文献边界

本路线的文献依据只采用已核读的 Murthy 与 Grubb 2024 年 Journal of Applied Crystallography 文章[《Evolution of elliptical SAXS patterns in aligned systems》](https://journals.iucr.org/j/issues/2024/04/00/tu5052/)，尤其是 §4.1：先用 z slices 观察层片反射峰位，再以强度曲面的 minimum-curvature 轨迹扩展椭圆拟合；butterfly 需要两条椭圆。该文支持“峰位轨迹可以呈椭圆、蝴蝶需要两个椭圆”的物理和几何背景，不等同于本项目的 annular 工程算法。

本项目的 `annular_peak` 是固定 q 环、逐环 `I(χ)`、实际峰支持和连续轨迹连接的工程实现，不能称作对 2021 年算法的复刻。2007/2021 文献全文在本次调校中未取得并逐篇核读，因此不把它们写成已验证的实现依据。最终椭圆参数、反射级次和材料结构解释仍需独立实验与人工审查。

## 图稿和源数据

含 `annular_peaks` 的图包会新增：

- `annular_qchi.svg/pdf/png/tiff`：q–χ 原始均值和受支持角向峰；
- `annular_profiles.svg/pdf/png/tiff`：代表 q 环的原始/平滑 `I(χ)`、count 和 coverage；
- `annular_profiles.csv`、`annular_peaks.csv`、`annular_profiles.npz`：profile、候选峰、选择状态、覆盖和原始数组；
- `annular_caption.txt`、`annular_manifest.json`：方法定义、解释边界和文件哈希。

导出器只消费已完成的 annular 结果，不在图稿阶段重积分、补象限、跨缺环桥接或重新拟合椭圆。Nature 单栏/双栏尺寸和矢量输出有助于排版与复核，但图件生成成功不保证几何模型适用、参数可发表或科学验收。
