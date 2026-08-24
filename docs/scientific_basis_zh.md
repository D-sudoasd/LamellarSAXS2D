# 科学量、符号、单位与可解释性边界

本页把当前 LamellarSAXS2D 源码实际计算的量，与两篇 Grubb 论文中的结构符号分开。论文给出科学背景和模型动机；软件输出以源码、公开 API 和结果 flags 为准。椭圆拟合或 `full2d` 成功，不等于已经完成唯一的三维结构反演。

## 1. 当前软件真正报告什么

输入图像的原始数值由 I/O 层读取后保留，不自动按曝光、监视器或最大值归一化。若提供 PONI，pyFAI 生成每个像素的 `q`、`chi`、`qx`、`qy`；当前物理 q 单位为 `nm^-1`。约定为 `chi=0°` 在 `+qx`，`chi=90°` 在 `+qy`，正方向为 qx/qy 平面内逆时针。

主要观测量如下：

| 层级 | 字段/结果位置 | 计算含义 | 不应扩大解释为 |
|---|---|---|---|
| q 空间 | `q`, `qx`, `qy`, `chi` | PONI/pyFAI 的倒易空间坐标 | 未校准像素的物理 q |
| 方位剖面 | `angle`, `intensity`, `coverage` | 给定 q 窗内的像素统计 | 结构取向分布的唯一反演 |
| 峰/脊点 | `q_star`, `q_star_nm_inv`, `angle`, `radial_fwhm`, `azimuthal_fwhm`, `snr`, `coverage`, `n_pixels` | 实际保留像素支持的径向峰或脊点 | 没有仪器展宽/重叠影响的本征尺寸 |
| 椭圆 | `a`, `b`, `axis_ratio`, `ellipticity`, `theta_deg`, `center`, `coverage`, `condition` | q 平面中共享中心、共享半轴、镜像倾角的表观双椭圆 | 唯一的层片、堆栈或链结构 |
| 周期派生量 | `Ln_nm`, `Ln_from_minor_axis_nm`, `Lz_from_draw_axis_nm` | 在明确单位和原点假设下由 q 半径换算的长度 | 自动等同论文结构参数 |
| 精修 | `full2d.parameters`, `model`, `residual`, `rmse` 等 | 像素级经验强度模型和诊断 | Grubb 2016 的完整 3D 物理模型 |

方位峰角由角向观测量独立测量。`phi_app_deg`（若该结果层提供）是图样的表观 lobe/倾角统计；`alpha_candidate_deg`、`psi_candidate_deg` 没有被椭圆旋转自动推断。缺失或不可报告的量保持 `NaN`/`null`，不以 0 代替。

兼容字段 `lamellar_spacing` 可能保留原始 q 坐标下的 `2π/q` 数值；跨文件或写入论文时应优先使用带单位的 `Ln_nm`，并同时检查 `q_unit` 与 spacing flags。

常见科学/质量 flags 包括：

- `apparent_geometry_only`、`nonunique_inverse_problem`：结果是观测图样的表观几何，单个二维图样不能唯一决定真实结构。
- `empirical_model_only`：`full2d` 使用经验强度模型。
- `uncalibrated_pixel_q`：未提供 PONI 且未显式提供 `q_scale` 时，坐标是 `pixel-q`，不是物理单位。
- `spacing_unavailable_unknown_q_unit`、`spacing_unavailable_nonzero_center`、`spacing_requires_origin_centered_ellipse_assumption`：周期换算缺少物理 q 单位或不满足原点中心假设。
- `low_coverage`、`low_snr`、`no_curvature_candidate`、`single_branch_supported`、`parameter_at_bound`、`solver_failed`：分别提示像素支持不足、信噪不足、曲率脊未找到、仅一个椭圆分支有支持、参数碰到边界或求解失败。
- `covariance_local_linear_approximation`：`full2d` 的协方差是最优点附近 Jacobian 的局部线性近似。

flags 是结果的一部分，不能因 `success=True` 而清除。

## 2. 2016、2021 与软件符号对照

两篇论文的同名希腊字母不是同一套定义。尤其不能把软件的椭圆轴倾角当作论文中的结构角。

| 符号/字段 | Grubb 2016 | Grubb 2021 | 软件当前含义 |
|---|---|---|---|
| `theta`/`θ` | 分子轴 `m` 与拉伸/纤维轴 `z` 的夹角（错配角） | 不是该软件的结构输出字段 | `theta_deg` 是 q 平面双椭圆的镜像轴倾角参数（相对 `reference_axis_deg`） |
| `psi`/`ψ` | 分子轴 `m` 与层片法向 `n` 的夹角 | 定义为 `ψ = ϕ − α` 的相对层片法向/堆栈轴关系 | 软件不从椭圆拟合直接得到 |
| `phi`/`ϕ` | 观测 reciprocal-space 极角；与 2016 的 `ψ` 不是同一观测量 | 层片法向相对 draw 轴的倾角 | 可由角向峰得到 `phi_app_deg` 这种表观量，但不自动成为结构 `ϕ` |
| `alpha`/`α` | 非上述主角的通用软件字段 | 堆栈长轴相对 draw 轴的倾角 | `alpha_candidate_deg` 只有在另加结构假设/独立证据时才可填写 |
| `ellipse_axis_tilt_deg` | 论文没有把它定义成软件字段 | 2021 的拟合椭圆轴倾角是图样几何量，且可能受重叠影响 | q 平面拟合参数；**不是结构 `ϕ`、`α`、`ψ` 或 2016 `θ`** |

软件的 `reference_axis_deg` 用于说明 q 平面角度的参考方向；pipeline 默认把 `draw_axis_deg - 90°` 作为参考轴。双椭圆的两个成员使用参考轴加/减软件 `theta`。因此读取结果时应同时保存 `reference_axis_deg`、`theta_deg`、`q_unit` 和 flags，而不能只抄一个“倾角”。

2021 还提醒：单张图样的椭圆可能来自没有相关性的层片旋转分布；椭圆轴方向也不必等于 `α`，经线重叠可能移动拟合方向。这是软件保留 `apparent_geometry_only` 和 `nonunique_inverse_problem` 的原因。

## 3. q、s、周期与像素单位

对波长 `λ`、散射半角 `θ`，常用散射矢量模长为

\[
q = \frac{4\pi}{\lambda}\sin\theta .
\]

软件的 `q` 是**角频率约定**，单位是弧度/长度（通常写 `nm^-1`；弧度视为无量纲）。若用晶体学/空间频率约定 `s`，单位是 cycles/length，则

\[
q = 2\pi s,\qquad s = \frac{q}{2\pi},\qquad
L = \frac{1}{s} = \frac{2\pi}{q}.
\]

所以 `q` 不能直接当作 `s` 使用；漏掉 `2π` 会把周期整体算错。软件的 `Ln_nm` 与椭圆的 `Ln_from_minor_axis_nm` 都使用最后一个式子，并先把 `q` 转为 `nm^-1`。

Grubb 2021 Table 3 的 `a`、`b` 明确以 **pixel** 报告（例如 eyebrow 约 `164.2`、`117.9` pixel，butterfly 约 `232.7`、`93.2` pixel）。像素轴只能比较同一图像/同一几何条件下的相对形状；没有像素尺寸、距离、波长和校准转换时，**不能与 `nm^-1` 直接比较，也不能直接代入 `2π/q`**。Table 3 的 `ellipticity` 按该表数值对应

\[
e = \sqrt{1-(b/a)^2},
\]

软件同时提供 `axis_ratio=b/a` 与 `ellipticity`/`eccentricity`，不要把它与 `b/a` 混称。

经验模型参数的单位也要随 qmap/图像声明：`a`、`b`、`radial_sigma`、`radial_gamma` 和 q 窗在同一 q 坐标单位；`theta`、`lobe_angle`、`angular_width` 的核心表示是弧度，UI/导出可提供对应的 `_deg` 字段；`axis_ratio`、`eta` 和角向包络的相对幅度是无量纲；背景/幅度与输入图像强度使用同一数值尺度。若输入强度没有绝对校正，软件不会替用户赋予物理强度单位。

## 4. `L_N`、`L_z` 的公式和适用条件

### 4.1 单条脊点

当脊点 `q_star` 的单位可识别为 `nm^-1` 或 `Å^-1`，且 `q_star>0` 时，软件报告

\[
L_N = L_n = \frac{2\pi}{q_{*,\,nm^{-1}}}.
\]

这里的 `N/n` 只是软件对“沿该散射矢量法向的表观周期”的字段命名；它不是自动测出的晶片厚度或层片法向真实周期。`pixel-q`、`unknown` 或其他未声明单位时，周期保持不可用并带 `spacing_unavailable_unknown_q_unit`。

### 4.2 原点中心双椭圆

对拟合半轴 `a,b`（同一物理 q 单位）和轴倾角 `theta`，软件只在以下条件下计算椭圆周期派生量：

1. q 单位可转换为 `nm^-1`；
2. `a>0`、`b>0`；
3. 拟合中心在 q 原点容差内；
4. 接受“该椭圆是原点中心倒易空间轨迹”的明确假设。

此时

\[
L_N = \frac{2\pi}{b_{nm^{-1}}},
\]

并先求 draw 方向的椭圆半径

\[
q_z = R(\pi/2;a,b,\theta)
= \frac{ab}{\sqrt{[b\cos(\pi/2-\theta)]^2+[a\sin(\pi/2-\theta)]^2}},
\qquad
L_z = \frac{2\pi}{q_{z,\,nm^{-1}}}.
\]

当前字段名为 `Ln_from_minor_axis_nm` 和 `Lz_from_draw_axis_nm`，pipeline 还提供 `L_N`/`L_z` 别名。若中心不在原点，两个长度保持 `NaN` 并带 `spacing_unavailable_nonzero_center`；即使中心满足条件，也保留 `spacing_requires_origin_centered_ellipse_assumption`，提醒这不是无条件的结构结论。

### 4.3 与 Grubb 2021 关系式的边界

Grubb 2021 在不同几何操作/定义下写出 `L_N = L_z cosϕ`（图 2 的层间滑移关系）以及正文旋转状态中的 `L_N = L_z cosψ`。它们依赖所定义的 `ϕ`、`α`、`ψ` 和旋转顺序，不能合并成一个普适公式。软件的 `L_N`/`L_z` 是 q 椭圆的表观长度派生量，不会自动选择或验证上述结构机制。

## 5. 脊线、lobe 与 mask

`ridge_method="radial_peak"` 在角向扇区中做径向剖面/峰定位；`ridge_method="surface_curvature"` 在平滑强度场上用主曲率方向寻找像素级脊点。两者都只使用实际有效像素，不会把另一象限的点镜像复制进去。无有效像素或信噪不足的扇区仍可保留为无效记录，并写出 `coverage`、`valid`、`reason`。

束挡（beamstop）、赤道 streak、饱和/热像素、中心散射和相互重叠的 butterfly/eyebrow 分支都可能污染脊线或椭圆方向。应在分析前用外部 mask，或用排除 ROI 将它们排除；mask 是质量控制，不是“缺失数据的对称补全”。

掩膜极性必须写清：

- `valid_mask=True` 表示保留像素；
- 外部 `mask`/`external_mask=True` 表示排除像素；
- 多个 mask 按有效像素交集合并；ROI 排除区按逻辑 OR 合并。

像素 ROI 支持 `rectangle`、`ellipse`；有 qmap 时程序化配置还支持可跨 `-180°/180°` 的 `q_sector`。UI 当前直接绘制的是像素矩形/椭圆；q-sector 请在 TOML/API 中声明。建议把 beamstop/streak/overlap 的来源和坐标系记录在项目 provenance 中。

## 6. 经验 `full2d` 与 Grubb 2016 完整 3D 模型的边界

当前 `full2d` 是像素级**经验测量模型**，包含：共享 `a,b`、镜像 `+theta/-theta` 的两条原点中心椭圆，四个角向包络，径向 Gaussian/Lorentzian 混合（`eta`），以及非负、平滑的径向背景。椭圆角度相对 `reference_axis_deg` 定义；pipeline 默认采用 `draw_axis_deg - 90°`。它保留整幅 `model`/`residual` 图像用于比较，默认使用全部有效像素进行优化并据此计算 `ndata/rmse/weighted_rmse`。仅当用户显式设置 `max_pixels` 时才进行确定性抽样，并以 `sampled_n/sample_rmse` 单独报告抽样诊断。模型可直接配置 `sigma` 或 `weights`（二者不能同时给出），默认稳健损失为 `soft_l1`。对软件自动生成的初值，会用有效像素的稳健分位数估计幅度与背景的**起始尺度**；这不归一化、不重写输入强度，并以 `initial_intensity_scale_estimated` 标记。用户显式给出的初值和跨帧 warm start 默认不重估，除非配置 `auto_scale_initial=true`。

这不是 Grubb 2016 的完整模型。2016 模型还包括分子/层片取向能量分布、三维取向积分、多个尺寸方向的展宽、pseudo-Voigt、背景与尺度/强度等物理参数，并在完整二维像素上拟合。当前软件没有实现或验证那套 3D 能量-取向正演/唯一逆解，因此不能把 `full2d` 的 `theta`、`a/b`、`phi_app` 直接写成 2016 的结构 `θ/ψ` 或 2021 的 `ϕ/α/ψ`。

经验模型适合在相同 q 校准、mask、权重和配置下比较图样/时间序列。它不是机制判别器；任何结构解释都需要额外实验、几何假设或论文证据。

## 7. 精修结果、不确定度与 reduced chi-square

椭圆拟合使用 Sampson 或 geometric q-space residual，默认 `soft_l1` 稳健损失；`rmse` 是所用残差的均方根，`coverage` 与 `condition` 用于判断覆盖和可辨识性。

`covariance`/`stderr` 的含义是最优点附近的局部线性近似：用 Jacobian 的伪逆正规矩阵，再乘以由残差 cost 和自由度得到的尺度。`full2d` 同样返回 `covariance_local_linear_approximation`。它不是经过重复实验、bootstrap 或完整误差传播得到的统计置信区间；mask、q 校准、模型错配、重叠和鲁棒损失都可能主导真实误差。固定参数/表达式绑定参数不是“测得的零不确定度”，其 `stderr` 通常为 `NaN` 或空值，并应读取 `fixed`/tied 状态。

结果中的 `reduced_chi_square` 只是

\[
2\,\mathrm{cost}/\max(N_\mathrm{residual}-N_\mathrm{free},1)
\]

的残差诊断。默认 robust residual、未必经过实验方差归一化的 q/强度残差，以及可能的像素抽样，使它**不是统计意义的 reduced χ²**，不能据此给 p 值、接受概率或“模型正确”的结论。只有在用户明确提供有物理含义的 `sigma`/weights、并确认残差与误差模型匹配时，才能把加权诊断与该误差模型一起解释；软件仍不会自动产生正式统计检验。

## 8. 与本地论文的对应关系

- [Grubb, Murthy & Francescangeli (2016), DOI 10.1002/polb.23930](https://doi.org/10.1002/polb.23930)：完整二维 3D 取向/展宽模型、能量分布、pseudo-Voigt、背景与权重。
- [Grubb et al. (2021), DOI 10.1016/j.polymer.2021.123566](https://doi.org/10.1016/j.polymer.2021.123566)：`ϕ/α/ψ` 几何、曲率 ridge、镜像双椭圆、Table 3 像素半轴及单图样反问题非唯一性。

阅读论文时，先区分“观测 reciprocal-space 量”“软件经验模型参数”“论文结构解释”三层；这三层只有在额外假设成立时才可建立映射。
