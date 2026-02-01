# Training guide / 训练说明

**语言 / Language:** [English](#english) · [中文](#中文)

---

<a name="english"></a>

## English

### Data split (required)

- **Fixed 8:1:1**: Train 80% : Val 10% : Test 10%, split by **episode** (no episode crosses sets).
- Args: `--train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1 --split_seed 42` (defaults in scripts).
- **Validation**: After each epoch, validation is run automatically; best checkpoint is saved by **validation loss**.
- **Test**: Used only for final evaluation; same `split_seed` as training so the test set is consistent.

### Run training in tmux

Always start long runs inside **tmux** so that SSH disconnect does not kill the process.

- Suggested session name: `yitong_train`.
- Attach: `tmux attach -t yitong_train`; detach without killing: `Ctrl+B` then `D`.

#### Stage 2 only (typical)

```bash
cd /path/to/value_function_stitched

tmux new-session -d -s yitong_train "CUDA_VISIBLE_DEVICES=0 python train.py --stage 2 \
  --projector_path ./outputs_alignment/best_projector.pt \
  --parquet_dir /path/to/your/lerobot_parquet/chunk-000 \
  --video_root /path/to/your/videos \
  --output_base ./outputs \
  --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1 --split_seed 42 \
  2>&1 | tee train_stage2_tmux.log"
```

Or use `start_train_tmux.sh` after editing paths and `CUDA_VISIBLE_DEVICES`.

#### Both stages (1 then 2)

```bash
tmux new-session -d -s yitong_train "CUDA_VISIBLE_DEVICES=0 python train.py --stage both \
  --parquet_dir /path/to/your/lerobot_parquet/chunk-000 \
  --video_root /path/to/your/videos \
  --output_base ./outputs \
  --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1 --split_seed 42 \
  2>&1 | tee train_both_tmux.log"
```

### Automatic validation

Stage 2 training runs validation at the end of each epoch and prints train loss, validation loss, and validation Top-1. If validation loss is the best so far, `best_*` weights are saved. No extra steps needed.

### After training: test-set evaluation

Using the same 8:1:1 split and `split_seed`:

```bash
python eval_test.py \
  --checkpoint ./outputs/robot_vf \
  --parquet_dir /path/to/your/lerobot_parquet/chunk-000 \
  --video_root /path/to/your/videos \
  --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1 --split_seed 42
```

For accurate metrics, use `frame_skip=1` on the test set (no frame skipping).

### GPU and batch size

- Aim to use most of the GPU memory without OOM.
- Single 24GB (e.g. RTX 3090/4090): `--batch_size 12` or 16, `--gradient_accumulation_steps 2`.
- Single 40GB+ (e.g. A6000): `--batch_size 16` or 24, `--gradient_accumulation_steps 2`.
- Check with `nvidia-smi`; increase `batch_size` if utilization and memory are low, decrease if near OOM.

---

<a name="中文"></a>

## 中文

### 数据划分（必读）

- **固定 8:1:1**：训练 80% : 验证 10% : 测试 10%，按 **episode** 划分（同一 episode 不跨集合）。
- 参数：`--train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1 --split_seed 42`（脚本中已默认，无需改则可不传）。
- **验证集**：每轮训练结束后自动跑验证，并按 **验证 loss** 保存 best 检查点。
- **测试集**：仅用于最终评估；与训练使用相同 `split_seed`，保证测试集一致。

### 在 tmux 中跑训练

长时间训练一律在 **tmux** 里启动，避免 SSH 断开后进程被杀。

- 建议会话名：`yitong_train`。
- 附着：`tmux attach -t yitong_train`；不杀进程退出：`Ctrl+B` 再按 `D`。

#### 仅阶段 2（常用）

```bash
cd /path/to/value_function_stitched

tmux new-session -d -s yitong_train "CUDA_VISIBLE_DEVICES=0 python train.py --stage 2 \
  --projector_path ./outputs_alignment/best_projector.pt \
  --parquet_dir /path/to/your/lerobot_parquet/chunk-000 \
  --video_root /path/to/your/videos \
  --output_base ./outputs \
  --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1 --split_seed 42 \
  2>&1 | tee train_stage2_tmux.log"
```

或编辑好路径和 `CUDA_VISIBLE_DEVICES` 后使用 `start_train_tmux.sh`。

#### 两阶段连跑（先 1 再 2）

```bash
tmux new-session -d -s yitong_train "CUDA_VISIBLE_DEVICES=0 python train.py --stage both \
  --parquet_dir /path/to/your/lerobot_parquet/chunk-000 \
  --video_root /path/to/your/videos \
  --output_base ./outputs \
  --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1 --split_seed 42 \
  2>&1 | tee train_both_tmux.log"
```

### 自动验证

阶段 2 训练在每轮结束时会自动跑验证集，输出训练 loss、验证 loss、验证 Top-1。若当前轮验证 loss 为历史最优，会保存 `best_*` 权重，无需额外操作。

### 训练结束后跑测试集评估

使用相同 8:1:1 与 `split_seed`：

```bash
python eval_test.py \
  --checkpoint ./outputs/robot_vf \
  --parquet_dir /path/to/your/lerobot_parquet/chunk-000 \
  --video_root /path/to/your/videos \
  --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1 --split_seed 42
```

为得到更准确指标，建议在测试集上使用 `frame_skip=1`（不跳帧）。

### GPU 与 batch_size

- 目标：尽量占满显存以缩短训练时间，但避免 OOM。
- 单卡 24GB（如 RTX 3090/4090）：`--batch_size 12` 或 16，`--gradient_accumulation_steps 2`。
- 单卡 40GB+（如 A6000）：`--batch_size 16` 或 24，`--gradient_accumulation_steps 2`。
- 训练跑起来后另开终端执行 `nvidia-smi` 查看该卡 **GPU-Util** 与 **Memory-Usage**；若利用率低且显存有余可适当增大 `batch_size`，若接近占满或 OOM 则减小 `batch_size` 或增大 `gradient_accumulation_steps`。
