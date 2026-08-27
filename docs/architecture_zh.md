# 软件架构与数据流

使用入口：[操作、输入输出、UI 与批处理指南](user_guide_zh.md)；科学量和结构解释边界见[科学量、符号、单位与可解释性边界](scientific_basis_zh.md)。

```text
CBF / EDF / TIF / TIFF / NPY / NPZ / HDF5 / CSV/TXT
              + PONI + mask + 可选 sigma/weights
                         |
                         v
         只读 preflight + 输入 hash + 状态门
                         |
                         v
       ImageFrame + QMap (PONI: nm^-1;
              无 PONI/嵌入 qmap: pixel-q + flag)
                         |
          +--------------+--------------+
          |                             |
          v                             v
  多尺度观测量与真实 ridge       完整二维经验强度场（full2d）
          |                             |
          +------- 对称双椭圆 seed -----+
                         |
                         v
          bounds / fixed / expr 精修
                         |
           observed / model / residual
                         |
             单帧证据包与批次演化表
```

## 模块边界

- `io.py`：只负责读取原始数组、frame、dataset 和元数据，不做归一化或物理解释；`valid_mask=True` 为有效像素，外部 `mask=True` 为无效像素。
- `validation.py`：定义严格结果 schema 和统一 `analysis-domain`；测量、ridge、ellipse、`full2d` 与导出共享同一最终像素集合。
- `preflight.py`：在拟合前只读核验 package、manifest、PONI、mask、单位、校正/不确定度状态和 SHA-256，并生成 green/yellow/red 证据。
- `geometry.py`：只通过 PONI/pyFAI 产生 q/chi/qx/qy，不重新发明探测器旋转公式。
- `observables.py`：提取剖面、lobe、ridge 和直接派生量。
- `ellipse.py`：只拟合倒易空间曲线，返回残差、协方差和可辨识性。
- `intensity.py`：生成和拟合经验二维强度场；当前 `full2d` 是共享双椭圆、角向包络、径向线形和背景的经验模型，不是 Grubb 2016 的完整 3D 取向正演。
- `pipeline.py`：编排单帧公开 seam；CLI 和 UI 均调用它。
- `batch.py`：序列顺序、时间、`independent`/`warm_start`、失败隔离和 checkpoint 恢复；没有跨全序列的 global/shared 联合优化。
- `export.py`：写入可回读证据与 provenance，不覆盖原始输入。
- `ui/`：只管理交互和显示；后台 worker 不接触 QWidget。

## 输出原则

CSV 保存有界、可人工检查的表格；完整 `I/qx/qy/model/residual/fit_valid_mask/sampled_valid_mask` 保存在压缩 NPZ。统一分析域为 `finite(I,qx,qy) AND detector_valid AND NOT external_mask AND NOT ROI AND q_window AND weight_valid`，JSON 同时记录各阶段计数。批次的 `manifest.json`/`provenance.json` 保存输入/config hash、模式和版本；单帧 JSON 保存摘要，不能替代 NPZ 像素证据。执行成功与科学解释是两套状态：`fit_success=True` 不能清除 `mechanism_under_determined`、`apparent_geometry_only` 或 `nonunique_inverse_problem`。
