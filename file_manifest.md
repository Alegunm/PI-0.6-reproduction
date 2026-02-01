# File manifest: repository contents / 文件清单：仓库内容

**语言 / Language:** [English](#english) · [中文](#中文)

---

<a name="english"></a>

## English

This document lists what is included in this repository and what is not distributed. **Pi 0.6 is closed source**; this repo is **our own reproduction** and **can be published on GitHub** for others to view (code, docs, results).

### ✅ Contents of this repository

| File / folder | Description |
|---------------|-------------|
| `README.md` | Project overview, architecture, quick start |
| `FILE_MANIFEST.md` | This file |
| `model_stitched.py` | Stitched VF model definition |
| `dataset_alignment.py` | Stage 1 VQA / image-caption dataloader |
| `dataset_robot.py` | Stage 2 robot parquet + video dataloader (format only; no data) |
| `train_alignment.py` | Stage 1 training script |
| `train_robot_vf.py` | Stage 2 training script |
| `train.py` | Unified entry (stage 1 / 2 / both) |
| `eval_test.py` | Test-set evaluation (loss, Top-1, Top-5) |
| `visualize_value_function.py` | Value-over-time visualization |
| `draw_architectures.py` | Script to draw architecture diagrams |
| `architecture_current_stitched.png` | Stitched VF architecture figure |
| `architecture_prev_4b_vla.png` | Previous 4B VLA VF figure (comparison) |
| `value_function_visualization.png` | Example value curve (optional) |
| `start_train_tmux.sh` | Example script to run stage 2 in tmux (paths are placeholders) |
| `TRAINING.md` | Training guide (8:1:1 split, tmux, GPU tips) |
| `RESULTS.md` | Validation/test metrics |
| `.gitignore` | Patterns for data, outputs, logs, caches |

The **.py** files implement training and evaluation; no dataset or weights are bundled. Data paths in scripts and docs are placeholders.

### ❌ Not distributed with this repository

| Item | Note |
|------|------|
| **Robot / real-world datasets** | No `data/` with parquet, videos, or raw logs is included. |
| **Large checkpoints** | No `outputs/`, `*.pt`, or `*.safetensors` are included; only code and docs. |
| **Logs** | No training or eval logs (e.g. `train_stage2_tmux.log`) are included. |
| **Secrets / cache** | `.env`, tokens, and build/cache artifacts are excluded via `.gitignore`. |

### Optional extras (not currently included)

- A minimal dummy parquet + 1–2 dummy images to illustrate the data format.
- Example configs (e.g. `configs/stage2_example.yaml`) with placeholder paths and hyperparameters.
- A lightweight released adapter (e.g. LoRA only) under a clear license, if added in the future.

### Summary

This repository contains **code, documentation, and architecture figures** only. **No robot data, large weights, or internal logs** are distributed.

---

<a name="中文"></a>

## 中文

本文档说明本仓库包含哪些内容、哪些内容不随仓库提供。**π₀.6 为闭源**；本仓库为 **我们自行复现**，**可发布在 GitHub 上**供他人查看（代码、文档、结果）。

### ✅ 本仓库包含的内容

| 文件 / 目录 | 说明 |
|-------------|------|
| `README.md` | 项目概述、架构、快速开始 |
| `FILE_MANIFEST.md` | 本文件 |
| `model_stitched.py` | 缝合式 VF 模型定义 |
| `dataset_alignment.py` | 阶段 1 VQA / 图文 dataloader |
| `dataset_robot.py` | 阶段 2 机器人 parquet + 视频 dataloader（仅格式，无数据） |
| `train_alignment.py` | 阶段 1 训练脚本 |
| `train_robot_vf.py` | 阶段 2 训练脚本 |
| `train.py` | 统一入口（阶段 1 / 2 / 两阶段连跑） |
| `eval_test.py` | 测试集评估（loss、Top-1、Top-5） |
| `visualize_value_function.py` | 价值随时间可视化 |
| `draw_architectures.py` | 架构图绘制脚本 |
| `architecture_current_stitched.png` | 缝合式 VF 架构图 |
| `architecture_prev_4b_vla.png` | 此前 4B VLA VF 图（对比用） |
| `value_function_visualization.png` | 示例价值曲线（可选） |
| `start_train_tmux.sh` | 示例：在 tmux 中跑阶段 2（路径为占位） |
| `TRAINING.md` | 训练说明（8:1:1 划分、tmux、GPU 建议） |
| `RESULTS.md` | 验证/测试指标 |
| `.gitignore` | 用于忽略 data、outputs、日志、缓存等的规则 |

所有 **.py** 文件实现训练与评估逻辑；仓库中不包含数据集或权重，脚本与文档中的路径均为占位。

### ❌ 不随本仓库提供的内容

| 项目 | 说明 |
|------|------|
| **机器人 / 真实场景数据集** | 不包含任何含 parquet、视频或原始日志的 `data/`。 |
| **大型检查点** | 不包含 `outputs/`、`*.pt`、`*.safetensors`，仅提供代码与文档。 |
| **日志** | 不包含训练或评估日志（如 `train_stage2_tmux.log`）。 |
| **密钥 / 缓存** | `.env`、token 及构建/缓存产物由 `.gitignore` 排除。 |

### 可选补充（当前未包含）

- 最小化 dummy parquet + 1–2 张 dummy 图像，用于说明数据格式。
- 示例配置（如 `configs/stage2_example.yaml`），含占位路径与超参。
- 若后续发布轻量适配器（如仅 LoRA），可在明确许可下单独提供。

### 小结

本仓库仅包含 **代码、文档与架构图**。**不提供机器人数据、大型权重或内部日志**。
