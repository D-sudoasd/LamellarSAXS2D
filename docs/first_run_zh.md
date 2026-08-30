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
```

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

右侧 `工作流状态 / Workflow status` 会根据当前状态提示下一步。建议顺序为：

1. **打开图像**：确认帧和 HDF5/NPZ dataset 选择正确。
2. **加载 PONI**：只有物理 `q` 坐标建立后，间距和 reciprocal-space 尺度才可解释。`pixel-q` 仅用于算法检查或合成数据。
3. **设置 mask、ROI 和 q 范围**：先排除 beam stop、坏点、探测器缝隙和无效边界，再确定分析区间。
4. **运行 Preview**：检查模型位置、双椭圆和剖面是否与观测花样基本一致。
5. **运行 Optimize**：仅在初值和分析域合理后进行；Optimize 不是科学有效性的自动判定。
6. **检查四视图**：同时检查 `Observed`、`Model`、`Residual` 和 `Overlay`，并查看 ridge、coverage、flags 与参数是否触边。
7. **人工接受或拒绝**：具名 `Accept/Reject` 仅记录当前会话审核，不等于 P3/P4 科学证据门通过。
8. **导出证据或进入批处理**：跨帧比较必须保持 PONI、mask、q 范围、权重和配置一致。

右侧控制栏已改为可滚动布局；在 980×680 或笔记本屏幕上，底部 ROI、人工审核和快照控件仍可访问。

## 5. 结果解释边界

- 未提供 PONI 时，软件会明确标记 `pixel-q`；此结果不能直接解释为 `nm^-1`、`Å^-1` 或真实空间尺度。
- 双椭圆与 `full2d` 是经验 reciprocal-space 测量/精修模型；良好拟合不等于唯一三维结构反演。
- 单帧拟合不能单独证明层片取向、相变或变形机制；应结合显微组织、衍射、力学或其他独立证据。
- 正式实数据分析前应先运行 `bsaxs preflight`，并依据 P3/P4 证据报告判断是否进入下一阶段。
