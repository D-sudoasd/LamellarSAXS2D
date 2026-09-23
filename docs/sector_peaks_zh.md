# 固定 χ 径向扇区：独立 `I(q)` 诊断

`radial_sector` 保留了“在一个方位角附近沿 q 方向看 profile”的测量，用来检查某个方向上的径向峰形、窗口截断和 q 位置稳定性。它是独立诊断路径，不是新 GUI 的主蝴蝶轨迹，也不替代默认的固定 q 环 `I(χ)` 花瓣轨迹；默认主路径见[固定 q 环花瓣轨迹](annular_trajectories_zh.md)。

## 何时使用

在 GUI 的识别方式中显式选择“扇区径向 `I(q)`”后，程序会在固定 χ 附近取一个有限角宽的扇区，将有效像素按 q 分箱，得到原始和平滑 profile。每个 profile 可以有受支持的径向峰，也可以只作为 profile-only 诊断保留。它适合检查：

- 某个花瓣方向的径向峰是否真的有两侧回落；
- q 窗口、mask 或探测器边界是否截断了峰；
- 低 q 尾部、宽峰、双峰或热像素是否让“最强点”不可靠；
- annular 主轨迹的某个 q 区间是否需要额外的方向性核对。

径向诊断产生的 `q*` 不自动进入默认主椭圆输入，也不自动分配反射级次。若用户明确选择它，结果必须与 annular 或历史曲率结果分开保存和解释。

## 计算定义

对中心角 `χ₀`、完整角宽 `w` 和 q 分箱 `Bⱼ`，程序使用 PONI 产生的逐像素 q/χ 坐标，并只保留有效 mask、有限强度和当前 q 窗口内的像素：

```text
I_raw(qⱼ, χ₀) = sum(Iₚ) / count(Iₚ),   p ∈ Bⱼ ∩ sector(χ₀, w)
```

原始 profile 的强度是有效像素的 `sum / count` 均值；`raw_sum`、`raw_count`、coverage 和几何覆盖也会保存。不同 q 分箱的像素数量不同，因此 `raw_sum` 不能直接当作强度比较。空分箱保持为空，不用零填充。

这条测量层不额外施加探测器效率、固角、偏振或绝对强度校正，也不把输入强度重新变成 pyFAI 校正积分。输入文件已经完成的校正和单位仍随上下文记录；q 轴没有 PONI 时只能按 pixel-q 解释。

## 角度设置与支持判据

径向模式默认扇区完整角宽为 10°，中心角步长为 5°。相邻 profile 因扇区重叠而相关，不能视为独立重复实验。q 分箱受到局部 q 像素步长限制，`sampling_sigma_q` 和 `half_bin_resolution` 是采样/分箱分辨率指标，不是置信区间。

程序不会强制每个角度都有 `q*`。候选峰需要有连续 q 支持、足够 coverage、相对于稳健原始 profile 基线的突出度和高度，并有可检查的两侧回落；还会检查有效贡献像素数和单个像素是否主导。常见的 profile-only 状态包括：

- 峰贴近 q 窗口边界，左侧或右侧回落不可见；
- mask、beam stop、探测器缝隙造成低覆盖或支持断裂；
- 相近候选峰无法区分，结果保持歧义；
- 一个热像素或极少数像素制造尖峰；
- profile 只有低 q 尾部、负向起伏或不足以支持峰的对比度。

连续支持足够时，宽峰可以保留；默认不会用窗口比例硬门槛把宽峰一概剔除。平滑曲线只服务定位和显示，原始均值、count、coverage 始终保留。没有峰的 profile 仍可在 GUI、CSV、NPZ 和离线图包中查看，不生成假 q 空间点。

## q* 的解释边界

径向模式的 `q*` 是该固定 χ profile 的观测峰位。它没有自动的反射级次；`2π/q*` 只有在单位、峰级次和物理模型另有证据时才可作相应长度解释。单独的径向 profile 不足以宣布一阶环、`Ln`、`Lz` 或长轴参数。

径向 FWHM 描述 profile 峰形，`sampling_sigma_q`/bin 分辨率描述采样限制；二者都不是置信区间，不能直接替代重复测量或实验误差分析。工程状态、SNR 或图件导出成功也不等于科学验收。

## GUI、CLI 与旧项目

新 GUI 会话默认是 `annular_peak`。旧项目配方没有 `trace_method` 时仍按历史 `curvature` 解释；历史 `ridge_method=butterfly_curvature` 的分支、象限和 family 语义继续保留。要运行径向诊断，必须在 GUI 明确切换方法，或通过 CLI 显式指定：

```powershell
bsaxs analyze data/frame_0001.edf `
  --poni geometry/detector.poni `
  --mask masks/detector.npy `
  --butterfly-stage trace `
  --butterfly-trace-method radial_sector `
  --sector-width 10 `
  --sector-step 5 `
  --butterfly-resamples 0 `
  -o results/frame_0001_radial_diagnostic
```

`--butterfly-trace-method` 接受 `curvature`、`radial_sector` 和 `annular_peak`。显式指定它时 CLI 选择蝴蝶工作流；不要再传入其他 `--ridge-method`，除非该值也是 `butterfly_curvature`。径向参数是 `--sector-width` 和 `--sector-step`；固定 q 环主路径的参数则是 `--annular-rings` 和 `--annular-angles`。

## 输出与人工检查

结果保留每个角度的 q 边界/中心、原始均值、平滑均值、count、coverage、候选峰、支持状态和失败原因。选中的峰会记录 prominence、SNR、径向 FWHM、代表支持像素和采样分辨率；代表像素只是二维图定位锚点，`q*` 来自 profile 统计量，不是像素最大值。

建议检查：

1. 原始 profile 在峰两侧是否有真实回落；
2. count/coverage 是否连续，是否被 q 边界或 mask 截断；
3. 相邻 5° profile 的变化是否考虑了重叠相关性；
4. 是否存在热像素、双峰歧义或低 q 尾部假峰；
5. PONI、q_unit、mask、q 窗口和输入强度校正是否与实验记录一致。

真实数据的窗口必须由目标峰和探测器覆盖决定。某一帧在 `q_max=0.5 nm^-1` 时出现的边界截断，不能推出所有帧都应使用 `0.8 nm^-1`；扩大窗口后也不能把不同窗口下的 SNR 直接比较。
