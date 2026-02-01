#!/bin/bash
# 阶段 2 训练 - 在 tmux 中运行（8:1:1 划分 + 每轮自动验证）
# 使用前请确认：outputs_alignment/best_projector.pt 已存在（阶段 1 已完成）

cd "$(dirname "$0")"

# 指定 GPU（按需修改）
export CUDA_VISIBLE_DEVICES=5

# 数据与输出路径（按需修改）
PARQUET_DIR="/home/e230112/ssd/Yitong/data/labeled_dataset_full/data/chunk-000"
VIDEO_ROOT="/home/e230112/ssd/Yitong/data/videos"
PROJECTOR_PATH="./outputs_alignment/best_projector.pt"
OUTPUT_BASE="./outputs"

# 8:1:1 划分 + 每次启动自动验证
TRAIN_RATIO=0.8
VAL_RATIO=0.1
TEST_RATIO=0.1
SPLIT_SEED=42

# 尽量占满显卡：46GB 卡用 batch_size=24，gradient_accumulation_steps=1；若 OOM 可改为 16 或 2
BATCH_SIZE=24
GRAD_ACCUM=1
NUM_WORKERS=8

if tmux has-session -t yitong_train 2>/dev/null; then
    echo "tmux 会话 'yitong_train' 已存在，先关闭再启动"
    tmux kill-session -t yitong_train 2>/dev/null || true
fi

tmux new-session -d -s yitong_train "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python train.py --stage 2 \
  --projector_path $PROJECTOR_PATH \
  --parquet_dir $PARQUET_DIR \
  --video_root $VIDEO_ROOT \
  --output_base $OUTPUT_BASE \
  --train_ratio $TRAIN_RATIO --val_ratio $VAL_RATIO --test_ratio $TEST_RATIO --split_seed $SPLIT_SEED \
  --batch_size $BATCH_SIZE --gradient_accumulation_steps $GRAD_ACCUM --num_workers $NUM_WORKERS \
  2>&1 | tee train_stage2_tmux.log"

echo "=========================================="
echo "阶段 2 训练已在 tmux 中启动（8:1:1 + 每轮自动验证）"
echo "=========================================="
echo "会话名: yitong_train"
echo "查看进度: tmux attach -t yitong_train"
echo "查看日志: tail -f train_stage2_tmux.log"
