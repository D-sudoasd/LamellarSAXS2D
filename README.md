# LamellarSAXS2D

[![CI](https://github.com/D-sudoasd/LamellarSAXS2D/actions/workflows/ci.yml/badge.svg)](https://github.com/D-sudoasd/LamellarSAXS2D/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.11--3.13-blue)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Quantitative refinement and in-situ analysis of anisotropic lamellar 2D SAXS patterns.**

LamellarSAXS2D reads calibrated two-dimensional SAXS detector data, measures anisotropic butterfly/eyebrow features across several scales, fits mirror-constrained double ellipses, and tracks fitted parameters through an in-situ sequence. It preserves the input intensity scale and uses a supplied PONI file to construct physical `q`, `chi`, `qx`, and `qy` coordinates through pyFAI.

![LamellarSAXS2D refinement workbench using synthetic data](docs/assets/refinement-ui.png)

The screenshot uses synthetic data and demonstrates the observed, model, residual, and reciprocal-space overlay views.

## Features

- CBF, EDF, TIF/TIFF, NPY, NPZ, HDF5, CSV, and TXT input, including explicit frame/dataset selection.
- PONI-calibrated reciprocal-space maps and explicit masks or exclusion regions.
- Radial and azimuthal profiles, lobe measurements, ridge extraction, and q-space symmetric double-ellipse fitting.
- Optional full-pixel empirical 2D refinement with bounds, fixed parameters, expression constraints, weights, and residual diagnostics.
- Qt workbench with live preview, background optimization, parameter tables, overlays, batch controls, and evolution plots.
- Independent or quality-gated warm-start processing, checkpoints, failed-frame isolation, and auditable CSV/JSON/NPZ exports.

## Install from source

Python 3.11 or newer is required.

```bash
git clone https://github.com/D-sudoasd/LamellarSAXS2D.git
cd LamellarSAXS2D
python -m pip install -e ".[ui,hdf5]"
```

Core analysis does not require Qt. For core-only use, install with `python -m pip install -e .`.

## Quick start

Create a deterministic synthetic pattern and open it in the workbench:

```bash
bsaxs synthetic --shape 128x128 -o synthetic.npz
bsaxs inspect synthetic.npz
bsaxs gui synthetic.npz
```

Analyze one calibrated detector frame:

```bash
bsaxs inspect data/frame_0001.cbf --poni geometry/detector.poni
bsaxs analyze data/frame_0001.cbf --poni geometry/detector.poni --full2d -o results/frame_0001
```

Track a sequence with quality-gated warm starts:

```bash
bsaxs batch "data/frame_*.cbf" --poni geometry/detector.poni \
  -o results/batch --mode warm_start --checkpoint results/checkpoint.json
```

See the [user guide](docs/user_guide_zh.md), [scientific definitions and limits](docs/scientific_basis_zh.md), and [architecture](docs/architecture_zh.md) for the complete interface and output contracts.

## Compatibility names

`LamellarSAXS2D` is the application display name. Existing programmatic interfaces remain stable:

- Python distribution: `butterfly-saxs`
- Import package: `butterfly_saxs`
- Command-line program: `bsaxs`
- Existing machine-readable provenance identifiers and project/output contracts

## Scientific scope

The symmetric ellipses and optional `full2d` model are empirical reciprocal-space measurement/refinement models. A successful fit does **not** uniquely recover a three-dimensional lamellar structure or establish a deformation mechanism from one 2D pattern. Structural interpretation requires explicit geometric assumptions and independent experimental evidence.

The implementation is informed by the ellipse and lamellar-pattern analysis discussed by [Grubb, Murthy & Francescangeli (2016)](https://doi.org/10.1002/polb.23930) and [Grubb et al. (2021)](https://doi.org/10.1016/j.polymer.2021.123566). The papers themselves are not redistributed in this repository.

---

## 中文说明

LamellarSAXS2D 面向取向层片体系的各向异性二维 SAXS 花样，提供从像素、剖面、峰脊线到镜像双椭圆和整幅经验强度模型的定量测量，并可跟踪原位序列中的参数演化。软件保留输入强度的数值尺度；当提供 PONI 文件时，通过 pyFAI 生成物理 `q/chi/qx/qy` 坐标。

### 主要能力

- 读取 CBF、EDF、TIF/TIFF、NPY、NPZ、HDF5、CSV/TXT，并显式选择帧或数据集。
- 使用 PONI、外部 mask 和排除 ROI 管理真实探测器几何与有效像素。
- 提取径向/方位剖面、lobe、ridge，并在 q 空间拟合共享中心和半轴的镜像双椭圆。
- 可选像素级 `full2d` 经验精修，支持参数边界、固定、表达式绑定、权重和残差诊断。
- Qt 界面提供 Observed、Model、Residual、Overlay 四视图、参数表、预览、后台优化、批处理和演化图。
- 支持独立拟合或质量门控的 warm start、checkpoint 恢复、失败帧隔离及 CSV/JSON/NPZ 可审计导出。

### 安装与使用

```bash
git clone https://github.com/D-sudoasd/LamellarSAXS2D.git
cd LamellarSAXS2D
python -m pip install -e ".[ui,hdf5]"

bsaxs inspect data/frame_0001.cbf --poni geometry/detector.poni
bsaxs analyze data/frame_0001.cbf --poni geometry/detector.poni --full2d -o results/frame_0001
bsaxs gui data/frame_0001.cbf --poni geometry/detector.poni
```

批处理、mask、项目配置、字段和失败恢复见[操作与输入输出指南](docs/user_guide_zh.md)；模型参数、单位和可解释性边界见[科学量与解释边界](docs/scientific_basis_zh.md)。

### 科学边界与兼容性

椭圆拟合和 `full2d` 成功，只说明当前经验模型能够描述所选有效像素，不等于完成唯一的三维结构反演或机制判定。跨帧比较应保持 q 标定、mask、权重和配置一致，并结合独立实验解释。

公开展示名为 `LamellarSAXS2D`；为保持现有用户脚本和结果兼容，安装包仍为 `butterfly-saxs`，Python 包仍为 `butterfly_saxs`，命令仍为 `bsaxs`。

## License

[MIT](LICENSE)
