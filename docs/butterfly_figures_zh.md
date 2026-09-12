# 蝴蝶 SAXS 测量图导出

`butterfly_saxs.butterfly_figure` 从观测强度、逐像素 `qx/qy` 和蝴蝶页 `current_result` 直接生成科研图，不依赖 Qt。主图使用输入的二维 q 坐标网格，因此支持非仿射网格和坐标递增、递减的情况；不会用图像尺寸推算 q 轴。

主图只叠加结果提供的脊线点，以及满足来源条件的已观测弧段。每个点都必须带有效的源像素坐标，且落在调用方提供的有效域和当前显示 q 窗口内；仅有 q 坐标的旧结果不会猜测像素位置。缺少像素坐标、超出图像边界或落在无效像素上的点不绘制，但会原样保留在结果文件和逐点 CSV，并记录省略原因。弧段必须显式标记为有效、包含唯一且明确接受的有序源点，并且整段路径不能穿过显示有效域外的像素。接受、拒绝和状态缺失使用不同符号表示；未知支路或侧别保持原样，不镜像补点。主图不绘制候选椭圆。

同一导出目录另含四面板比较图：观测强度、单独的候选椭圆曲线、观测强度叠加测得轨迹与候选曲线、以及已有拟合残差诊断。有限的 ring-only、边界受限或未收敛几何可以作为虚线候选诊断曲线导出，并明确标记为“候选 / 非定量 / 非科学接受”；缺失、非有限或不物理的几何不画。候选曲线完整范围不是观测弧支持。若没有逐点投影/法向残差或可用法向强度 profile，诊断面板明确显示不可用，不构造残差。另有单独的 ellipse-only SVG/PDF/PNG/TIFF，供独立排版。

## 判断拟合是否可信，以及定位最强点

蝴蝶页现在可分别选择实测谱、观测轨迹、几何候选、全像素强度模型椭圆及两者对比。几何椭圆来自脊线点；强度模型椭圆使用当前优化结果的参数与参考轴。它们对应不同目标函数，不应混称为同一个拟合。角向包络、径向展宽和分量叠加都会使模型强度最大值偏离名义椭圆曲线。

页面显示几何残差相对于定位误差的比值，并沿用分析结果中已有的筛查限值。超过限值时明确显示匹配较差；优化器收敛不代表科学验收。强度模型的条件数、参数边界和参数来源随图一起记录。改变分析条件后，旧的峰标记与模型图层失效，重新计算后才能用于当前结果。

峰位表中的 `G` 是所选有效分析域内的原始最亮像素，可能位于低 q 尾部或窗口边缘。`P1–P4` 是受像素覆盖、角向局部突出程度及噪声检查支持的局部信号峰，最多四个，不会补造缺失峰。峰位用实际像素的 qx/qy 标定坐标表示；平滑只用于稳健定位，原始强度同时保留。选择表格行可定位相应标记并检查角向、径向曲线。P 编号按本帧方位排序，不能直接作为跨帧跟踪标识。

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

`*` 图件均有 SVG、PDF、PNG、TIFF。画板宽度和可编辑文字遵循 [Nature 官方科研图指南](https://research-figure-guide.nature.com/figures/building-and-exporting-figure-panels/) 的单/双栏规格；排版合规与实验模型有效性分别检查。

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

当配置未指定其他脊线方法时，图稿导出使用 `butterfly_curvature`；如果用户明确配置了另一种方法，导出请求会在分析前被拒绝。图稿尺寸选项独立于分析先验：默认的 `b/a` 先验仍由分析配置控制（当前默认范围为 `0.005–0.35`），固定栏宽、字体或分辨率不会改变拟合结果。

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
- 若传入 `model`，另有 `model_comparison.svg/pdf/png/tiff`、`model_comparison_data.npz` 和 `model_comparison_caption.txt`。
- `source_data.npz`：原始 `observed`、`qx`、`qy`、调用方提供的选择 mask、完整有效像素 mask、数组自身的掩码、q 单位，以及径向分箱边缘、raw 强度和、raw 均值与计数。调用方的 `valid_mask` 可能已合并探测器、q 窗口、ROI 或其他分析域限制；导出器不把它称为探测器专用 mask。旧键 `valid_mask` 保留为兼容别名；新键 `supplied_valid_mask` 与 role 字段明确其来源；数组可用 `numpy.load(..., allow_pickle=False)` 读取。
- `ridge_points.csv`：逐点 q/像素坐标、来源标记、侧别、接受状态、是否绘出、未绘出的原因及完整源记录 JSON。
- `result.json`：完整蝴蝶页结果，非有限数值在 JSON 中记为 `null`。
- `settings.json`、caption 文件：画板规格、强度变换、mask 后的颜色限值与裁剪计数、径向统计定义、输入和点叠加计数、调用方提供的有效 mask 角色、候选曲线状态 / 支路 / 残差来源、上下文和科学解释边界。
- `manifest.json`：其他输出文件的 SHA-256。清单不包含自身哈希，避免自引用。

默认 `log1p` 变换仅用于主图显示，并保留负强度符号；原始阵列不会改写。颜色限值取显示区域有效像素变换后强度的 0.5 与 99.5 百分位，裁剪计数写入设置。辅助图 b 是以输入 q 坐标原点为中心的等宽径向分箱，显示未经强度变换的逐像素均值及每箱有效像素数；图中 `n` 表示每箱有效源像素数，纵轴使用有记录阈值的 symlog，原始有符号均值和各箱强度和均随 NPZ 导出。该曲线不是各向同性拟合，也不对像素面积或噪声进行额外加权。未知或像素 q 单位原样标在轴上，不据此换算层片周期。

版式采用固定单栏 89 mm 或双栏 183 mm，画板高度不超过 170 mm，正文 6.5 pt 无衬线字体，面板字母为粗体小写 8 pt。该导出是可审阅的图稿工件；目标期刊的最终规范、图注、实验判断与科学验收仍需独立核对。
