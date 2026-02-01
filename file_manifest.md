# File manifest: what to publish vs keep private / 文件清单：可公开与需保密

**语言 / Language:** [English](#english) · [中文](#中文)

---

<a name="english"></a>

## English

Use this list when preparing the repo for GitHub. **Do not commit** anything in the "Private / do not show" section.

### ✅ Safe to publish (can show on GitHub)

| File / folder | Description |
|---------------|-------------|
| `README.md` | Project overview, architecture, quick start (no real data paths) |
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
| `start_train_tmux.sh` | **After redaction**: replace real paths with placeholders like `/path/to/parquet`, `/path/to/videos` |
| `TRAINING.md` | Training guide (8:1:1 split, tmux, GPU tips; placeholder paths only) |
| `RESULTS.md` | Validation/test metrics (no internal paths) |
| `.gitignore` | Ignore data, outputs, logs, caches (see below) |

**Notes:**

- All **.py** under this list: safe to show; they describe *how* to train/evaluate, not your actual data.
- **start_train_tmux.sh**: either redact paths before commit, or keep a generic example and document "replace with your paths" in README.
- **训练手册.md** / **训练与测试结果表.md**: you can keep them as internal docs; for GitHub, use TRAINING.md / RESULTS.md (bilingual, placeholder paths only) instead.

### ❌ Do not publish (private / do not show)

| Item | Reason |
|------|--------|
| **Robot / real-world datasets** | Sensitive; not for open source. This includes any `data/` with parquet, videos, or raw logs. |
| **Paths to internal data** | e.g. `/home/.../data/labeled_dataset_full/`, `/home/.../videos`. Use placeholders in committed scripts/docs. |
| **Large checkpoints** | `outputs/`, `outputs_alignment/`, `*.pt`, `*.safetensors`. Can document *structure* in README; do not commit full weights unless you intend to release them. |
| **Logs with local paths** | `train_stage2_tmux.log`, `eval_test_frame_skip1.log`, etc. May contain paths and internal info. |
| **Environment / secrets** | `.env`, tokens, internal server names, account names. |
| **Cache / build artifacts** | `__pycache__/`, `.pyc`, `*.egg-info`, virtualenvs. |

### Optional: what to add in a “release” or docs

- **Small example**: A minimal dummy parquet + 1–2 dummy images to show data format (no real robot data).
- **Config examples**: e.g. `configs/stage2_example.yaml` with placeholder paths and hyperparameters (batch size, LR, split seed).
- **One lightweight checkpoint**: If you later release a small adapter (e.g. LoRA only) for a specific task, you can add it under a clear license; keep full weights and raw data private.

### Summary

- **Show**: Code, architecture figures, README, FILE_MANIFEST, TRAINING.md, RESULTS.md, .gitignore.
- **Do not show**: Real data, real paths in committed files, large weights, logs that leak internal info.

After redacting paths in `start_train_tmux.sh`, the repo is safe to publish as a methodology and implementation showcase without exposing lab data or infrastructure.

---

<a name="中文"></a>

## 中文

准备将仓库发布到 GitHub 时请参照本清单。**请勿提交**「不可公开」部分的任何内容。

### ✅ 可公开（可在 GitHub 上展示）

| 文件 / 目录 | 说明 |
|-------------|------|
| `README.md` | 项目概述、架构、快速开始（无真实数据路径） |
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
| `start_train_tmux.sh` | **脱敏后**：将真实路径替换为占位路径如 `/path/to/parquet`、`/path/to/videos` |
| `TRAINING.md` | 训练说明（8:1:1 划分、tmux、GPU 建议；仅占位路径） |
| `RESULTS.md` | 验证/测试指标（无内部路径） |
| `.gitignore` | 忽略 data、outputs、日志、缓存等（见下文） |

**说明：**

- 上述所有 **.py**：可安全展示；描述的是*如何*训练/评估，不包含实际数据。
- **start_train_tmux.sh**：提交前将路径脱敏，或保留通用示例并在 README 中说明「请替换为你的路径」。
- **训练手册.md** / **训练与测试结果表.md**：可仅作内部文档；对外使用 TRAINING.md / RESULTS.md（双语言、仅占位路径）即可。

### ❌ 不可公开（需保密 / 勿展示）

| 项目 | 原因 |
|------|------|
| **机器人 / 真实场景数据集** | 敏感，不宜开源。包括任何含 parquet、视频或原始日志的 `data/`。 |
| **内部数据路径** | 如 `/home/.../data/labeled_dataset_full/`、`/home/.../videos`。在提交的脚本/文档中请使用占位路径。 |
| **大型检查点** | `outputs/`、`outputs_alignment/`、`*.pt`、`*.safetensors`。可在 README 中说明*结构*；除非打算发布，否则不要提交完整权重。 |
| **含本地路径的日志** | `train_stage2_tmux.log`、`eval_test_frame_skip1.log` 等。可能包含路径与内部信息。 |
| **环境 / 密钥** | `.env`、token、内部服务器名、账号名。 |
| **缓存 / 构建产物** | `__pycache__/`、`.pyc`、`*.egg-info`、虚拟环境等。 |

### 可选：发布或文档中可补充的内容

- **小示例**：最小化 dummy parquet + 1–2 张 dummy 图像，用于展示数据格式（非真实机器人数据）。
- **配置示例**：如 `configs/stage2_example.yaml`，含占位路径与超参（batch size、LR、split seed）。
- **轻量检查点**：若后续发布某任务的小型适配器（如仅 LoRA），可在明确许可下加入；完整权重与原始数据仍保密。

### 小结

- **可展示**：代码、架构图、README、FILE_MANIFEST、TRAINING.md、RESULTS.md、.gitignore。
- **不可展示**：真实数据、提交文件中的真实路径、大型权重、泄露内部信息的日志。

将 `start_train_tmux.sh` 中的路径脱敏后，仓库即可作为方法与实现展示安全发布，而不暴露实验室数据或基础设施。
