# 蝴蝶 SAXS 测量图导出

`butterfly_saxs.butterfly_figure` 从观测强度、逐像素 `qx/qy` 和蝴蝶页 `current_result` 直接生成科研图，不依赖 Qt。主图使用输入的二维 q 坐标网格，因此支持非仿射网格和坐标递增、递减的情况；不会用图像尺寸推算 q 轴。

结果可以来自新 GUI 默认的 `annular_peak` 固定 q 环 `I(χ)` 花瓣轨迹、历史 `curvature` 脊线，或独立的 `radial_sector` 径向 profile。三者的点定义不能混用：annular 点来自逐环角向峰，`q_annulus` 是采样坐标；radial `q*` 来自固定 χ 的 `I(q)` profile，属于独立诊断，不是默认主椭圆输入。annular 原始强度与 radial profile 都按有效像素 `sum / count` 均值保存，并保留 counts/coverage；导出不额外施加 detector correction，输入强度的已有校正和单位按上下文记录。方法细节见[固定 q 环花瓣轨迹](annular_trajectories_zh.md)和[径向扇区诊断](sector_peaks_zh.md)。

主图只叠加结果提供的脊线点，以及满足来源条件的已观测弧段。每个点都必须带有效的源像素坐标，且落在调用方提供的有效域和当前显示 q 窗口内；仅有 q 坐标的旧结果不会猜测像素位置。缺少像素坐标、超出图像边界或落在无效像素上的点不绘制，但会原样保留在结果文件和逐点 CSV，并记录省略原因。弧段必须显式标记为有效、包含唯一且明确接受的有序源点，并且整段路径不能穿过显示有效域外的像素。接受、拒绝和状态缺失使用不同符号表示；未知支路或侧别保持原样，不镜像补点。主图不绘制候选椭圆。

同一导出目录另含四面板比较图：观测强度、单独的候选椭圆曲线、观测强度叠加测得轨迹与候选曲线、以及已有拟合残差诊断。有限的 ring-only、边界受限或未收敛几何可以作为虚线候选诊断曲线导出，并明确标记为“候选 / 非定量 / 非科学接受”；缺失、非有限或不物理的几何不画。候选曲线完整范围不是观测弧支持。若没有逐点投影/法向残差或可用法向强度 profile，诊断面板明确显示不可用，不构造残差。另有单独的 ellipse-only SVG/PDF/PNG/TIFF，供独立排版。

## 判断拟合是否可信，以及定位最强点

蝴蝶页现在可分别选择实测谱、annular 观测轨迹、历史曲率候选、几何候选、全像素强度模型椭圆及两者对比。几何椭圆使用已经固定 branch/family 的观测轨迹；强度模型椭圆使用当前优化结果的参数与参考轴。它们对应不同目标函数，不应混称为同一个拟合。角向包络、径向展宽和分量叠加都会使模型强度最大值偏离名义椭圆曲线。

页面显示几何残差相对于定位误差的比值，并沿用分析结果中已有的筛查限值。超过限值时明确显示匹配较差；优化器收敛不代表科学验收。强度模型的条件数、参数边界和参数来源随图一起记录。改变分析条件后，旧的峰标记与模型图层失效，重新计算后才能用于当前结果。

峰位表中的 `G` 是所选有效分析域内的原始最亮像素，可能位于低 q 尾部或窗口边缘。`P1–P4` 是受像素覆盖、角向局部突出程度及噪声检查支持的局部信号峰，最多四个，不会补造缺失峰。峰位用实际像素的 qx/qy 标定坐标表示；平滑只用于稳健定位，原始强度同时保留。选择表格行可定位相应标记并检查角向、径向曲线。P 编号按本帧方位排序，不能直接作为跨帧跟踪标识。

上段的 G/P 说明适用于历史曲率和像素峰诊断。使用 `annular_peak` 时，sector 表不再是主轨迹证据；图包会列出每个 q 环的完整 `I(χ)` profile、count、coverage、候选峰和实际选择的 0–4 个角向峰。少于四个峰和缺环都保留，不镜像补象限、不跨缺环桥接。使用 `radial_sector` 时，表中仍列出固定 χ 的 `I(q)` profile；没有受支持径向峰的角度仍可查看失败原因，不生成假 q 空间点。`q_annulus` 不是 `q*`，两者都不能仅凭导出自动等同于一阶周期或 `Ln`。

局部信号峰默认使用当前径向提示给出的信号带；G 的搜索仍覆盖完整分析窗口。导出记录两者各自的域与提示来源。G、局部像素峰、径向曲线峰值和 q* 是不同量，不会把一个最亮像素自动换算成层片周期。

v2 峰位诊断先用有效像素的径向中位数估计各向同性参考，将参考采样到同一个 q 网格，并经过相同的掩膜和平滑操作。用于识别的角向曲线是观测平滑剖面减去这个参考，避免像素采样把圆环变成四瓣。原始和平滑观测曲线仍完整保留；参考及检测差值随图、CSV、NPZ 一起输出。该参考用于识别，不替代实验背景扣除。局部角向噪声默认在一分箱尺度估计，保留原有四倍噪声突出度门槛与像素噪声传播下界；估计尺度及独立像素假设写入方法记录。

界面的剖面可在“曲线／数据”间切换，数据表包含对应坐标、单位和各条曲线；可用键盘访问。峰标记与脊线同时显示时按屏幕距离选择，附近的峰不再抢占脊线点击。动态流程提示和拟合警示同步提供给读屏工具。

模拟控制包含各向同性圆环、旋转／畸变网格、掩膜、噪声和二／三／四瓣，正对照覆盖 8–22° 的角向高斯宽度。更宽、相互重叠且对比度较低的分量仍可能无法分别识别，不能由返回峰数推断真实结构分量数。

模型峰与观测峰采用相同的掩膜归一化平滑定位规则，在观测峰对应的角向区域内配对。位移分量为模型坐标减观测坐标，`delta_q = hypot(delta_qx, delta_qy)` 表示二维位移长度，不是两个 q 半径的差。手动多边形排除/恢复按原顺序重放，导出峰位与测量域保持一致；源阵列仍完整保存。

新增图件均随同一次导出生成：

- `measured_only.*`：干净的实测二维谱。
- `geometry_overlay.*`：实测谱叠加几何候选，显示匹配状态。
- `intensity_model_overlay.*`：有同源模型参数和参考轴时，单独叠加强度模型椭圆。
- `peak_map.*`：G 与受支持峰的二维定位；有模型时叠加对应模型峰位及偏移。
- `peak_diagnostics.*`：角向/径向剖面与定位诊断。
- `peak_zooms.*`：峰附近的局部放大图。
- `peak_landmarks.csv/json`、`peak_profiles.csv/npz`：原始/平滑强度、像素与 q 坐标、覆盖、警示和实测—模型位移。批量测量 CSV 也包含独立的 `raw_pixel_maximum` 与 `supported_lobe_pixel` 记录，其 q 半径不冒充 q*。
- `fit_assessment.json`、`fit_source_parameters.csv`、`fit_overlay_curves.npz`：两种拟合的参数来源、状态和可重绘曲线。
- 若结果含 `annular_peaks`，另有 `annular_qchi.svg/pdf/png/tiff`（q–χ 原始均值与角向峰轨迹）、`annular_profiles.svg/pdf/png/tiff`（代表 q 环的 `I(χ)`、count/coverage 诊断）、`annular_profiles.csv`、`annular_peaks.csv`、`annular_profiles.npz`、`annular_caption.txt` 和 `annular_manifest.json`。这些图件消费已计算的 annular 结果，不在导出阶段重新积分、补象限或跨缺环连接。
- 若结果含 `sector_peaks`，另有 `sector_qchi.svg/pdf/png/tiff`（固定 χ 径向诊断热图）、`sector_profiles.svg/pdf/png/tiff`（径向 profile 与 count/coverage 诊断）、`sector_peaks.csv`、`sector_profiles.csv`、`sector_profiles.npz`、`sector_caption.txt` 和 `sector_manifest.json`。这些图件属于独立 `radial_sector` 诊断，不能代替 annular 主轨迹。

`*` 图件均有 SVG、PDF、PNG、TIFF。画板宽度和可编辑文字遵循 [Nature 官方科研图指南](https://research-figure-guide.nature.com/figures/building-and-exporting-figure-panels/) 的单/双栏规格；排版合规与实验模型有效性分别检查。

## 从界面直接生成图包

完成“识别轨迹 → 评估”并检查实测叠加后，点击“导出图稿”。在同一个窗口选择保存位置、单栏 89 mm 或双栏 183 mm，以及 300 / 600 / 1200 dpi。尺寸示意仅表示纸面比例；不会为了预览重新拟合或改变观测数据。SVG/PDF 保留矢量文字，DPI 主要决定嵌入图像与 PNG/TIFF 的像素数。通常可先用 600 dpi，再按目标期刊和具体图件要求调整。

输出写入新的图包子目录；已有目录不会被覆盖。后台导出期间可取消。完成后可打开图包中的 `index.html`，离线浏览实测图、annular q–χ 花瓣轨迹、各环 profile、历史曲率/像素峰诊断、独立径向 profile 与可用模型图，并访问对应图注、矢量文件和源数据。页面不依赖联网脚本或远程字体。

图包同时提供 `radial_profile.csv`，保存主图径向剖面的分箱边界、q 中心、原始强度和、均值及有效像素数。空分箱的均值留空，像素数不表示独立重复实验数，也不作为误差棒或置信区间。

`figure_qa.json` 核对导出配置并记录科学解释边界；它不逐个解析生成的图像或验证版面。配置检查通过不证明模型适用、结构参数可发表或所有标签均无重叠；投稿前仍应核对实际文件，并在最终版面尺寸下检查所有图件。Nature 各子刊要求可能不同，应以目标期刊的最新说明为准。

## Python

```python
from butterfly_saxs.butterfly_figure import export_butterfly_figure

files = export_butterfly_figure(
    "results/frame_0001_figure",  # 必须是尚不存在的新目录
    observed=observed,
    qx=qx,
    qy=qy,
    valid_mask=valid_mask,         # True 表示有效
    result=butterfly_page.current_result,
    model=full2d_model,             # 可选：调用方已有的全像素强度模型
    q_unit="nm^-1",
    context={"source": "frame_0001.edf", "frame": 1},
    display_scale="log1p",        # 也可选 linear 或 asinh
    width_mm=183.0,                 # 仅支持 Nature 单栏 89 mm 或双栏 183 mm
    dpi=600,
)
print(files["pdf"], files["source_data"], files["manifest"])
```

`render_butterfly_figure(...)` 接受相同的图像输入并返回 Matplotlib `Figure`，可用于预览或嵌入其他 Python 工作流。`model` 是可选的独立全像素强度模型数组，形状必须与 `observed` 相同；仅由蝴蝶椭圆几何推算的强度图不属于有效模型输入。提供时会额外生成观测 / 模型 / `observed - model` 三联图，共享强度变换和颜色限值，差值采用以零为中心的对称限值并保留原始强度单位。导出函数还接受 `cancel_event` 和 `progress`；进度回调签名为 `progress(percent: int, phase: str)`，取消会抛出项目共用的 `AnalysisCancelled`。

命令行示例（可在蝴蝶轨迹追踪后直接导出）：

```powershell
bsaxs analyze frame.edf --poni geometry.poni --mask mask.npy `
  --butterfly-stage trace --figure-output results/frame_0001_figure `
  --figure-width 183 --figure-dpi 600
```

图稿导出按 `current_result` 中实际记录的 `annular_peak`、`radial_sector` 或 `curvature` 方法生成对应章节；不会把一个方法的点重新解释成另一个方法的点。若从 CLI 新算结果，应显式选择目标方法。图稿尺寸选项独立于分析先验；annular 路径不会把 q 窗口外边界当成真实长轴尖端，用户明确给出的参数上下界仍按配置保留。

## 文件与解释

若结果含 `diagnostics.q_window`，图像和彩色限值使用该 q 半径窗口内的有效样本，并裁剪到这些样本的最小包围矩形；没有此窗口时显示完整有效区域。原始数组始终完整保存在 sidecar。若裁剪区域内有非有限 q 坐标，显示回退为有效 q 样本点，不会填造 q 坐标。

导出会在目标目录旁构建完整暂存目录，结束后以同卷目录重命名发布；目标已存在时拒绝覆盖。输出包括：

- `butterfly_figure.svg` / `butterfly_figure.pdf`：含可编辑矢量文字的图稿，PDF 使用 TrueType 42 字体；强度网格作为栅格图层嵌入。
- `butterfly_figure.tiff` / `butterfly_figure.png`：按指定 DPI 渲染。
- `butterfly_comparison.svg/pdf/png/tiff`：测量、候选几何、叠加与残差诊断四面板图。
- `ellipse_only.svg/pdf/png/tiff`：仅当有有限候选几何时输出；ring-only / bound-limited 曲线仍标记为诊断候选，不代表定量椭圆。
- `ellipse_curves.csv` / `ellipse_curves.npz`：源 `candidate_fit` 生成的曲线坐标、角度与来源支路标识；无可用候选时保存空曲线数组及 CSV 表头。
- `point_residuals.csv`：逐点保存实际 `projection_residual_q` / `normal_residual_q` 等字段；没有数据的字段留空。
- `normal_profiles.csv`：仅当输入结果包含可配对的 offset、raw 与 fit profile 时输出。诊断图若使用一个代表 profile，按“有效已接受点中 q 半径最接近中位数者”选取并写入设置，避免任意选择首条。
- `annular_qchi.*`、`annular_profiles.*` 与对应 CSV/NPZ/caption/manifest：固定 q 环 `I(χ)`、最多四个实际角向峰、branch/family 轨迹支持和缺环状态。`q_annulus` 是采样坐标，不转换为 `2π/q`。
- `sector_qchi.*`、`sector_profiles.*` 与对应 CSV/NPZ/caption/manifest：仅在结果含 `sector_peaks` 时生成，属于固定 χ `I(q)` 独立诊断。
- 若传入 `model`，另有 `model_comparison.svg/pdf/png/tiff`、`model_comparison_data.npz` 和 `model_comparison_caption.txt`。
- `source_data.npz`：原始 `observed`、`qx`、`qy`、调用方提供的选择 mask、完整有效像素 mask、数组自身的掩码、q 单位，以及径向分箱边缘、raw 强度和、raw 均值与计数。调用方的 `valid_mask` 可能已合并探测器、q 窗口、ROI 或其他分析域限制；导出器不把它称为探测器专用 mask。旧键 `valid_mask` 保留为兼容别名；新键 `supplied_valid_mask` 与 role 字段明确其来源；数组可用 `numpy.load(..., allow_pickle=False)` 读取。
- `ridge_points.csv`：逐点 q/像素坐标、来源标记、侧别、接受状态、是否绘出、未绘出的原因及完整源记录 JSON。
- `result.json`：完整蝴蝶页结果，非有限数值在 JSON 中记为 `null`。
- `settings.json`、caption 文件：画板规格、强度变换、mask 后的颜色限值与裁剪计数、径向统计定义、输入和点叠加计数、调用方提供的有效 mask 角色、候选曲线状态 / 支路 / 残差来源、上下文和科学解释边界。
- `manifest.json`：其他输出文件的 SHA-256。清单不包含自身哈希，避免自引用。

默认 `log1p` 变换仅用于主图显示，并保留负强度符号；原始阵列不会改写。颜色限值取显示区域有效像素变换后强度的 0.5 与 99.5 百分位，裁剪计数写入设置。辅助图 b 是以输入 q 坐标原点为中心的等宽径向分箱，显示未经强度变换的逐像素均值及每箱有效像素数；图中 `n` 表示每箱有效源像素数，纵轴使用有记录阈值的 symlog，原始有符号均值和各箱强度和均随 NPZ 导出。该曲线不是各向同性拟合，也不对像素面积或噪声进行额外加权。未知或像素 q 单位原样标在轴上，不据此换算层片周期。

径向扇区图中的 FWHM 是 `I(q)` profile 峰宽，annular 图中的角向 FWHM 是 `I(χ)` 峰宽；bin 分辨率是采样限制，二者都不是置信区间。annular 的 q 环与角向分箱是预设采样坐标，不能把 q 环中心或角向峰自动换算为层片周期。Nature 尺寸、矢量文字和离线源数据包便于排版与复核，但图稿格式检查或导出成功本身不保证模型适用、结构参数可发表或科学验收。

版式采用固定单栏 89 mm 或双栏 183 mm，画板高度不超过 170 mm，正文 6.5 pt 无衬线字体，面板字母为粗体小写 8 pt。该导出是可审阅的图稿工件；目标期刊的最终规范、图注、实验判断与科学验收仍需独立核对。
