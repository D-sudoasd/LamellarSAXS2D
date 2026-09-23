# LamellarSAXS2D 首次启动与标准工作流

本文用于解决“安装后如何确认环境可用、双击为何无反应、进入界面后下一步做什么”三个常见问题。科学模型、参数定义和解释边界仍以[操作指南](user_guide_zh.md)与[科学量定义](scientific_basis_zh.md)为准。

## 1. 建立项目环境

支持 Python 3.11–3.13；Python 3.14 及更高版本不在当前支持范围内。建议在项目目录建立固定名称的虚拟环境：

```powershell
py -3.13 -m venv .venv-project
.\.venv-project\Scripts\python.exe -m pip install --upgrade pip
.\.venv-project\Scripts\python.exe -m pip install `
  -c constraints\validation-py311-313.txt -e ".[all]"
```

也可使用 `.venv` 或 `venv`。Windows 启动器依次查找 `.venv-project`、`.venv`、`venv`，最后才使用 `PATH` 中的 `python`，从而避免 README 安装方式与桌面启动方式不一致。

## 2. 启动前诊断

```powershell
bsaxs-doctor --require-ui
```

只有所有必需项均为 `OK` 时，GUI 才应启动。诊断会检查：

- Python 是否为 3.11、3.12 或 3.13；
- NumPy、SciPy、Matplotlib、FabIO、pyFAI、tifffile 与 PyYAML；
- GUI 所需的 PySide6 和 pyqtgraph；
- 可选的 HDF5 依赖 h5py。

机器可读报告：

```powershell
bsaxs-doctor --require-ui --json
bsaxs doctor --json
```

`bsaxs doctor` 与 `bsaxs-doctor` 相同；无 GUI 时不要加 `--require-ui`。无子命令的 `bsaxs` 会打印 JSON 命令清单（`bsaxs describe`）。

## 3. Windows 启动

双击 `启动_LamellarSAXS2D.cmd`，或在终端运行：

```powershell
.\启动_LamellarSAXS2D.cmd
```

可把图像路径和参数继续传给 GUI：

```powershell
.\启动_LamellarSAXS2D.cmd data\frame_0001.cbf `
  --poni geometry\detector.poni
```

仅检查启动链而不打开窗口：

```powershell
.\启动_LamellarSAXS2D.cmd --check
```

启动器不会自动修改环境。若检查失败，它会显示可复现的安装命令；若 `pythonw.exe` 启动阶段发生异常，完整 traceback 会写入用户目录下的 `LamellarSAXS2D/launcher.log`，同时显示日志位置，避免“双击后无反应”。

## 4. 界面中的推荐顺序

右侧 `工作流状态 / Workflow status` 会根据当前状态提示下一步。蝴蝶分析页的建议顺序为：

1. **打开图像**：确认帧和 HDF5/NPZ dataset 选择正确。
2. **加载 PONI**：只有物理 `q` 坐标建立后，间距和 reciprocal-space 尺度才可解释。`pixel-q` 仅用于算法检查或合成数据。
3. **设置 mask、ROI 和 q 范围**：先排除 beam stop、坏点、探测器缝隙和无效边界，再确定分析区间。
4. **识别（Identify）**：新建会话默认在固定 q 环上计算角向 `I(χ)`（默认 40 个 q 环、72 个角度分箱），从每个环提取最多四个实际支持峰并沿 q 连接花瓣轨迹；旧项目缺少 `trace_method` 时仍按历史曲率脊线处理。这是观测测量，不是 `full2d` 强度拟合。
5. **评估（Evaluate）**：在固定参考轴和对角 family 配对的观测轨迹上拟合镜像约束双椭圆，并查看缺环、占用边、flags 与质量诊断。拟合不能重新指派 individual petals；贴在先验边界上的解仍按候选限制或仅环处理。
6. **检查叠加与峰位**：核对实测图、q 环 `I(χ)` profile、观测花瓣轨迹、几何候选和（若已运行）`full2d` 椭圆。每个环少于四个峰时保留实际支持，不补象限；缺环不桥接。固定 χ 的 `radial_sector` 只作为独立 `I(q)` 诊断，不能与 annular 主轨迹混用。
7. **可选 Preview / Optimize**：高级强度页的 Preview/Optimize 只服务经验 `full2d` 模型，不能代替 Identify → Evaluate，也不能当作科学验收。
8. **人工接受或拒绝**：具名 `Accept/Reject` 仅记录当前会话审核，不等于 P3/P4 科学证据门通过。
9. **导出图包或进入批处理**：跨帧比较必须保持 PONI、mask、q 范围、权重和配置一致。

固定 q 环路径的原始角向强度定义为有效像素的 `sum / count`，并同时保存 count 和 coverage；本测量层不追加固角、偏振或探测器效率校正。每个 q 环最多四个实际支持峰，按参考轴固定为 QI+QIII 与 QII+QIV 两个对角 family；dominant 选择和至少连续 3 个 q 环的轨迹门槛用于阻止孤立噪声，少于四瓣或缺环都保留原状。`q_annulus` 是采样坐标，不是 `q*`，不能直接换算 `2π/q`。

annular 路径不把 q 窗口外边界当成长轴真实尖端，已禁用共享 `observed_tip_constraint`，但保留用户明确给出的参数上下界。长轴不可辨识时标记限制状态，不强报 `a`、`Ln` 或 `L`。完整判据与 CLI 选项见[固定 q 环花瓣轨迹](annular_trajectories_zh.md)。

固定 χ 的径向诊断见[径向扇区](sector_peaks_zh.md)：它默认使用 10° 扇区宽度、5° 步长，允许 profile-only、边界截断、低覆盖、歧义、热像素和宽峰诊断；FWHM 与 bin 分辨率不是置信区间。真实 frame110/120 当前显示的 points 与 4 条 arcs 只是当前分箱示例，不能写成最终验证数字或科学接受结论。

页面上方的质量摘要集中显示当前阶段、工程质量、观测支持与结果限制。只有当前分析明确提供且单位有效的 `q*` 和环周期才显示数值；未标定或过期的结果不显示旧的物理周期。较长的参数状态和原因可把指针停留在表格单元格上查看。摘要中的“候选”和工程 `PASS` 都不等于科学验收。

计算异常或取消会使旧测量失效；本次计算已返回但质量不通过的结果，仍保留为当前诊断，允许带失败标记导出。图稿导出失败不会改变已有测量质量。切换帧或设置后，已完成的旧图包会明确标为此前快照。

图稿导出窗口把保存位置、栏宽和分辨率放在一起；完成后可打开离线图包查看图件、图注与源数据，详见[测量图导出](butterfly_figures_zh.md)。图稿统一采用显示有效域的 0.5–99.5% 颜色范围，这与工作台临时调整的显示百分位可能不同；原始数值不会因此改变。

右侧控制栏已改为可滚动布局；在 980×680 或笔记本屏幕上，底部 ROI、人工审核和快照控件仍可访问。

## 5. 结果解释边界

- 未提供 PONI 时，软件会明确标记 `pixel-q`；此结果不能直接解释为 `nm^-1`、`Å^-1` 或真实空间尺度。
- 双椭圆与 `full2d` 是经验 reciprocal-space 测量/精修模型；良好拟合不等于唯一三维结构反演。
- 单帧拟合不能单独证明层片取向、相变或变形机制；应结合显微组织、衍射、力学或其他独立证据。
- 正式实数据分析前应先运行 `bsaxs preflight`，并依据 P3/P4 证据报告判断是否进入下一阶段。
