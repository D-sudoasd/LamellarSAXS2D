# LamellarSAXS2D

[![CI](https://github.com/D-sudoasd/LamellarSAXS2D/actions/workflows/ci.yml/badge.svg)](https://github.com/D-sudoasd/LamellarSAXS2D/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.11--3.13-blue)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Identify and parameterize butterfly-pattern 2D SAXS, then review an in-situ series without treating a solver bound as a measured structure.**

LamellarSAXS2D reads calibrated detector frames (CBF, EDF, TIFF, NPY/NPZ, HDF5), builds physical `q`, `chi`, `qx`, and `qy` from a PONI file through pyFAI, traces observed butterfly arcs, and reports what the image actually supports:

| A typical frame reports | Only when an interior ellipse is supported |
| --- | --- |
| First-order `q*` and **L ring** = `2π/q*` | Apparent `a`, `b/a`, `θ` |
| Occupied sides, quality, and flags | Unpublished **Ln / Lz / L major** candidates |
| `ring` when the fit sits on a bound or the major axis runs away | Never a silent overwrite of user bounds or of ring L into the Ln column |

`success=True` is not scientific acceptance. Pixel-q never invents a physical period. Opposite quadrants are never fabricated.

[中文说明](#中文说明) · [Install](#install) · [Quick start](#quick-start) · [Docs](#documentation) · [Scientific scope](#scientific-scope)

![Synthetic butterfly pattern with an origin-centred double-ellipse overlay](docs/assets/refinement-ui.png)

Synthetic demonstration (pixel-q): the Wang/Grubb double ellipse drawn on a generated butterfly. Real experimental frames more often support a first-order **ring** until an interior ellipse is actually identified.

## What it does

- **Butterfly arcs** (`ridge_method=butterfly_curvature`): curvature ridges, branch/side labels (QI+QIII vs QII+QIV), first-order family vs harmonics, sparse-ring fill. See the [butterfly arc guide](docs/butterfly_arcs_zh.md).
- **Honest ellipse publication**: `flat_ellipse` (editable `b/a` bounds, default `0.005–0.35`) and `very_flat_ellipse`. A cap, floor, or major axis longer than the observed first-order ridge is **ring-only** — `a`, tilt, and ellipticity stay unpublished.
- **Batch review**: independent frames, cancel/progress, checkpoints, streaming CSV/JSON/NPZ. The table shows `WARN · ring` / `WARN · ellipse`, L ring, and candidate-period tooltips. Warm-start stays quality-gated.
- **Workbench**: Identify arcs → Evaluate; bilingual UI; first-order ring overlay instead of a capped tilted ellipse; lamellar studio and 0.4 publication artboards are schematics, not a unique inversion ([studio](docs/lamellar_workbench_zh.md), [figures](docs/publication_figures_zh.md)).
- **Optional `full2d`**: empirical whole-pixel intensity refinement. It is a different model from the butterfly geometry measurement.
- **Measurement and fit figures**: export fixed-size SVG/PDF and high-resolution TIFF/PNG with source arrays, curve/profile CSVs, and checksums. Inspect measured data, candidate ellipses, overlays, and actual `full2d` predictions without promoting a candidate to a scientifically accepted result. See the [figure export guide](docs/butterfly_figures_zh.md).
- **Peak diagnostics**: locate the raw brightest pixel separately from supported lobe peaks; inspect measured/model peak positions, local zooms, angular/radial profiles, and clean overlays from each fit source. Peak coordinates and support flags also travel through batch exports.
- **Preflight and P3/P4 gates**: read-only package checks and evidence reports. They do not freeze science or replace named human review.

## Install

Python **3.11–3.13** (3.14+ is outside the support contract). Core analysis does not need Qt; the workbench does.

```powershell
git clone https://github.com/D-sudoasd/LamellarSAXS2D.git
cd LamellarSAXS2D
python -m venv .venv-project
.\.venv-project\Scripts\python.exe -m pip install --upgrade pip
.\.venv-project\Scripts\python.exe -m pip install `
  -c constraints\validation-py311-313.txt -e ".[all]"
.\.venv-project\Scripts\bsaxs-doctor.exe --require-ui
```

Linux / macOS:

```bash
python -m venv .venv-project
.venv-project/bin/python -m pip install --upgrade pip
.venv-project/bin/python -m pip install \
  -c constraints/validation-py311-313.txt -e ".[all]"
.venv-project/bin/bsaxs-doctor --require-ui
```

Core-only: `python -m pip install -e .` then `bsaxs-doctor` without `--require-ui`.

For Chinese figure text on Debian/Ubuntu, install a CJK font: `sudo apt-get install fonts-noto-cjk`. The renderer selects an installed CJK font; CI installs Noto CJK so missing-glyph checks run on Linux as well as Windows.

On Windows, after the doctor is green, double-click `启动_LamellarSAXS2D.cmd` or run `.\启动_LamellarSAXS2D.cmd --check`. The launcher uses `.venv-project` / `.venv` / `venv` first and writes start-up failures to a per-user `LamellarSAXS2D/launcher.log`. Details: [first-run guide](docs/first_run_zh.md).

## Quick start

```bash
bsaxs-doctor --require-ui
bsaxs synthetic --shape 128x128 -o synthetic.npz
bsaxs inspect synthetic.npz
bsaxs-gui synthetic.npz
```

Calibrated detector frame — geometry / butterfly measurement (not `--full2d`):

```bash
bsaxs inspect data/frame_0001.edf --poni geometry/detector.poni --mask masks/detector.npy
bsaxs analyze data/frame_0001.edf --poni geometry/detector.poni --mask masks/detector.npy \
  --ridge-method butterfly_curvature --ellipse-preset flat_ellipse \
  --butterfly-stage evaluate --butterfly-resamples 0 \
  -o results/frame_0001
```

Folder of frames:

```bash
bsaxs batch "data/frame_*.edf" --poni geometry/detector.poni --mask masks/detector.npy \
  --ridge-method butterfly_curvature --ellipse-preset flat_ellipse \
  --butterfly-stage evaluate --mode independent \
  -o results/batch --checkpoint results/checkpoint.json
```

Read-only package check before fitting real data:

```bash
bsaxs preflight data/package --manifest manifest.csv \
  --poni geometry.poni --mask mask.npy -o results/preflight
```

`bsaxs analyze ... --full2d` is the optional empirical intensity fit. `bsaxs gui` remains an alias of `bsaxs-gui`.

## Names

| Shown to people | Stable machine name |
| --- | --- |
| LamellarSAXS2D | PyPI / wheel: `butterfly-saxs` |
| | Import: `butterfly_saxs` |
| | CLI: `bsaxs`, `bsaxs-doctor`, `bsaxs-gui` |

## Documentation

| Topic | Page |
| --- | --- |
| First launch and recommended UI order | [docs/first_run_zh.md](docs/first_run_zh.md) |
| CLI, TOML, batch, masks, exports | [docs/user_guide_zh.md](docs/user_guide_zh.md) |
| Butterfly arcs and publication rules | [docs/butterfly_arcs_zh.md](docs/butterfly_arcs_zh.md) |
| Symbols, units, and interpretation limits | [docs/scientific_basis_zh.md](docs/scientific_basis_zh.md) |
| Architecture | [docs/architecture_zh.md](docs/architecture_zh.md) |
| Lamellar studio / publication artboards | [docs/lamellar_workbench_zh.md](docs/lamellar_workbench_zh.md), [docs/publication_figures_zh.md](docs/publication_figures_zh.md) |
| P3 / P4 evidence | [docs/validation/benchmark_protocol.md](docs/validation/benchmark_protocol.md) |

## Scientific scope

The double ellipse is an **empirical reciprocal-space measurement**, following the Wang/Grubb picture of a butterfly as a pair of origin-centred ellipses. One 2D pattern does not uniquely recover a 3D lamellar stack or a deformation mechanism.

- [Wang, Murthy & Grubb (2007)](https://doi.org/10.1016/j.polymer.2007.04.026)
- [Grubb, Murthy & Francescangeli (2016)](https://doi.org/10.1002/polb.23930)
- [Grubb et al. (2021)](https://doi.org/10.1016/j.polymer.2021.123566)

The papers are not redistributed here.

---

## 中文说明

LamellarSAXS2D 面向取向层片的各向异性二维 SAXS 蝴蝶纹：用 PONI（pyFAI）得到物理 `q/chi/qx/qy`，识别观测弧，并只发表图像真正支持的量。

多数实验帧给出的是**一阶环周期**（`q*` 与 **环 L**）。只有内凹椭圆真正成立时，才显示表观 `a`、`b/a`、`θ`，以及未发表的 **Ln / Lz / 长轴 L** 候选。贴在 `flat_ellipse` 上下界、或长轴超出一阶脊线范围的解，按 **仅环** 处理，不会把求解器的倾角或环 L 写进 Ln 列。`success=True` 不是科学验收；像素 q 不能冒充物理周期；缺失象限不会被镜像补齐。

### 安装与启动

支持 Python 3.11–3.13。Windows：

```powershell
git clone https://github.com/D-sudoasd/LamellarSAXS2D.git
cd LamellarSAXS2D
py -3.13 -m venv .venv-project
.\.venv-project\Scripts\python.exe -m pip install --upgrade pip
.\.venv-project\Scripts\python.exe -m pip install `
  -c constraints\validation-py311-313.txt -e ".[all]"
.\.venv-project\Scripts\bsaxs-doctor.exe --require-ui
.\启动_LamellarSAXS2D.cmd --check
.\启动_LamellarSAXS2D.cmd
```

蝴蝶页建议顺序：**识别弧 → 评估**。详见[首次启动](docs/first_run_zh.md)。

评估后可在「叠加图层」分别检查实测谱、观测轨迹、几何候选和全像素模型椭圆；峰位表区分原始最亮点 G 与受支持峰 P，选中行即可定位。导出同时包含干净叠加图、峰位图、局部放大、剖面及 CSV/NPZ 源数据。匹配差或参数不稳定时会保留明确提示，不能仅凭曲线看起来像蝴蝶判定拟合正确。详见[测量图与峰位导出](docs/butterfly_figures_zh.md)。

### 常用命令

```bash
bsaxs inspect data/frame_0001.edf --poni geometry/detector.poni --mask masks/detector.npy
bsaxs analyze data/frame_0001.edf --poni geometry/detector.poni --mask masks/detector.npy \
  --ridge-method butterfly_curvature --ellipse-preset flat_ellipse \
  --butterfly-stage evaluate --butterfly-resamples 0 -o results/frame_0001
bsaxs batch "data/frame_*.edf" --poni geometry/detector.poni --mask masks/detector.npy \
  --ridge-method butterfly_curvature --ellipse-preset flat_ellipse \
  --butterfly-stage evaluate --mode independent -o results/batch
bsaxs-gui data/frame_0001.edf --poni geometry/detector.poni
```

`--full2d` 才是可选的整幅经验强度精修，与蝴蝶几何测量不是同一条路径。批处理表用「警告 · 仅环 / 椭圆」区分发表状态；环 L 与 Ln 候选分列。

公开展示名是 `LamellarSAXS2D`；安装包仍为 `butterfly-saxs`，导入仍为 `butterfly_saxs`，主命令仍为 `bsaxs`。

科学量、单位与不可扩大解释的边界见[科学量与解释边界](docs/scientific_basis_zh.md)；操作与导出见[用户指南](docs/user_guide_zh.md)。

## License

[MIT](LICENSE)
