#!/usr/bin/env python3
"""
统一训练入口：阶段 1（视觉对齐）+ 阶段 2（机器人价值微调）

用法示例：
  # 只跑阶段 1
  python train.py --stage 1

  # 只跑阶段 2（需已有人训好的 projector）
  python train.py --stage 2 --parquet_dir /path/to/chunk-000 --video_root /path/to/videos

  # 两阶段连跑（先对齐，再机器人价值）
  python train.py --stage both --parquet_dir /path/to/chunk-000 --video_root /path/to/videos
"""

import argparse
import subprocess
import sys
from pathlib import Path


def _run_stage1(args):
    """运行阶段 1：视觉对齐。"""
    out_dir = Path(args.output_base) / "alignment"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent / "train_alignment.py"),
        "--output_dir", str(out_dir),
        "--batch_size", str(args.batch_size),
        "--max_epochs", str(args.max_epochs_align),
        "--lr", str(args.lr_align),
        "--dataset_name", args.dataset_name,
        "--max_length", str(args.max_length),
        "--gradient_accumulation_steps", str(args.gradient_accumulation_steps),
        "--logging_steps", str(args.logging_steps),
        "--save_steps", str(args.save_steps),
        "--max_grad_norm", str(args.max_grad_norm),
        "--num_workers", str(args.num_workers),
        "--device", args.device,
    ]
    if args.cauldron_subset:
        cmd += ["--cauldron_subset", args.cauldron_subset]
    if args.image_root:
        cmd += ["--image_root", args.image_root]
    if args.max_samples is not None:
        cmd += ["--max_samples", str(args.max_samples)]
    if args.cache_dir:
        cmd += ["--cache_dir", args.cache_dir]
    if args.load_in_4bit:
        cmd += ["--load_in_4bit"]

    print("=" * 60)
    print("阶段 1：视觉对齐 (Alignment)")
    print("=" * 60)
    return subprocess.run(cmd, check=True)


def _run_stage2(args, projector_path: str):
    """运行阶段 2：机器人价值微调。"""
    out_dir = Path(args.output_base) / "robot_vf"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent / "train_robot_vf.py"),
        "--parquet_dir", args.parquet_dir,
        "--video_root", args.video_root,
        "--projector_path", projector_path,
        "--output_dir", str(out_dir),
        "--batch_size", str(args.batch_size),
        "--max_epochs", str(args.max_epochs_robot),
        "--lr", str(args.lr_robot),
        "--gradient_accumulation_steps", str(args.gradient_accumulation_steps),
        "--logging_steps", str(args.logging_steps),
        "--save_steps", str(args.save_steps),
        "--max_grad_norm", str(args.max_grad_norm),
        "--num_workers", str(args.num_workers),
        "--device", args.device,
        "--lora_r", str(args.lora_r),
        "--lora_alpha", str(args.lora_alpha),
        "--lora_dropout", str(args.lora_dropout),
        "--train_ratio", str(args.train_ratio),
        "--val_ratio", str(args.val_ratio),
        "--test_ratio", str(args.test_ratio),
        "--split_seed", str(args.split_seed),
    ]
    if args.no_train_projector:
        cmd += ["--no_train_projector"]
    if args.no_lora:
        cmd += ["--no_lora"]
    if args.cache_dir:
        cmd += ["--cache_dir", args.cache_dir]
    if args.load_in_4bit:
        cmd += ["--load_in_4bit"]

    print("=" * 60)
    print("阶段 2：机器人价值微调 (Robot Value Fine-tuning)")
    print("=" * 60)
    return subprocess.run(cmd, check=True)


def parse_args():
    p = argparse.ArgumentParser(
        description="统一训练：阶段 1 视觉对齐 + 阶段 2 机器人价值微调",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--stage",
        type=str,
        choices=["1", "2", "both"],
        default="both",
        help="运行阶段：1=仅对齐, 2=仅机器人价值, both=先1后2",
    )
    p.add_argument(
        "--output_base",
        type=str,
        default="./outputs",
        help="输出根目录；阶段1 -> output_base/alignment，阶段2 -> output_base/robot_vf",
    )

    # 阶段 2 必需（当 stage 为 2 或 both 时）
    p.add_argument("--parquet_dir", type=str, default=None, help="机器人 Parquet 目录，如 labeled_dataset_full/data/chunk-000")
    p.add_argument("--video_root", type=str, default=None, help="视频根目录")
    p.add_argument(
        "--projector_path",
        type=str,
        default=None,
        help="阶段 1 的 projector 权重；stage=both 时可省略，默认用 output_base/alignment/best_projector.pt",
    )

    # 共用
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--gradient_accumulation_steps", type=int, default=2)
    p.add_argument("--logging_steps", type=int, default=50)
    p.add_argument("--save_steps", type=int, default=2000)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--load_in_4bit", action="store_true", help="Gemma 4-bit 量化")

    # 阶段 1 专用
    p.add_argument("--max_epochs_align", type=int, default=2, help="阶段 1 训练轮数")
    p.add_argument("--lr_align", type=float, default=2e-5, help="阶段 1 学习率")
    p.add_argument("--dataset_name", type=str, default="HuggingFaceM4/the_cauldron", help="阶段 1 数据集")
    p.add_argument("--cauldron_subset", type=str, default="vqav2", help="The Cauldron 子集，如 vqav2；空则不用流式")
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--image_root", type=str, default=None)
    p.add_argument("--max_samples", type=int, default=None, help="阶段 1 最多样本数（调试用）")

    # 阶段 2 专用（8:1:1 划分 + 每次启动自动验证）
    p.add_argument("--train_ratio", type=float, default=0.8, help="训练集比例（默认 0.8）")
    p.add_argument("--val_ratio", type=float, default=0.1, help="验证集比例（默认 0.1）")
    p.add_argument("--test_ratio", type=float, default=0.1, help="测试集比例（默认 0.1）")
    p.add_argument("--split_seed", type=int, default=42, help="划分随机种子，与 eval 一致以保证同一测试集")
    p.add_argument("--max_epochs_robot", type=int, default=10, help="阶段 2 训练轮数")
    p.add_argument("--lr_robot", type=float, default=1e-4, help="阶段 2 学习率")
    p.add_argument("--no_train_projector", action="store_true", help="阶段 2 不训 Projector")
    p.add_argument("--no_lora", action="store_true", help="阶段 2 关闭 LoRA")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)

    return p.parse_args()


def main():
    args = parse_args()
    stage = args.stage

    if stage in ("2", "both"):
        if not args.parquet_dir or not args.video_root:
            print("错误：阶段 2 需要指定 --parquet_dir 和 --video_root")
            sys.exit(1)

    # 阶段 1
    if stage in ("1", "both"):
        _run_stage1(args)
        if stage == "1":
            print("仅运行阶段 1，结束。")
            return

    # 阶段 2：确定 projector 路径
    if stage == "2":
        if not args.projector_path:
            print("错误：--stage 2 时必须指定 --projector_path，或先运行阶段 1 再使用 --stage both")
            sys.exit(1)
        projector_path = args.projector_path
    else:
        projector_path = args.projector_path or str(Path(args.output_base) / "alignment" / "best_projector.pt")
        if not Path(projector_path).exists():
            print(f"错误：未找到 Projector 权重 {projector_path}，请先完成阶段 1 或指定 --projector_path")
            sys.exit(1)

    _run_stage2(args, projector_path)
    print("两阶段训练全部完成。")


if __name__ == "__main__":
    main()
