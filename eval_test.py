"""
测试集评估：Test Loss、Top-1 与 Top-5 准确率。
使用与训练时相同的 8:1:1 划分（split_seed/ratio），确保测试集一致。
"""
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.amp import autocast
from tqdm import tqdm

from model_stitched import create_stitched_model
from dataset_robot import (
    RobotValueDatasetStitched,
    split_robot_dataset_by_episode,
    create_robot_dataloaders_split,
    collate_robot,
)


def main():
    p = argparse.ArgumentParser(description="测试集评估：Loss、Top-1 与 Top-5 准确率")
    p.add_argument("--checkpoint", type=str, default="./outputs/robot_vf", help="阶段 2 输出目录（含 best_*.pt、best_lora_adapter/）")
    p.add_argument("--parquet_dir", type=str, required=True, help="Parquet 目录（与训练一致）")
    p.add_argument("--video_root", type=str, required=True, help="视频根目录（与训练一致）")
    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--test_ratio", type=float, default=0.1)
    p.add_argument("--split_seed", type=int, default=42, help="与训练一致以保证同一测试集")
    p.add_argument("--batch_size", type=int, default=12)
    p.add_argument("--frame_skip", type=int, default=5)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--no_amp", action="store_true", help="禁用混合精度")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--max_samples", type=int, default=None, help="最多评估样本数（默认全部）")
    p.add_argument("--cache_dir", type=str, default=None)
    args = p.parse_args()

    ckpt = Path(args.checkpoint)
    if not (ckpt / "best_projector.pt").exists():
        raise FileNotFoundError(f"未找到 {ckpt / 'best_projector.pt'}")
    if not (ckpt / "best_value_head.pt").exists():
        raise FileNotFoundError(f"未找到 {ckpt / 'best_value_head.pt'}")
    projector_path = str(ckpt / "best_projector.pt")
    value_head_path = str(ckpt / "best_value_head.pt")
    lora_path = ckpt / "best_lora_adapter"
    if not lora_path.exists():
        lora_path = None

    device = args.device
    use_amp = not args.no_amp

    print("加载缝合 VLM...")
    model = create_stitched_model(device=device, cache_dir=args.cache_dir)
    model.freeze_vision_tower()
    model.load_projector(projector_path)
    model.load_value_head(value_head_path)
    if lora_path is not None:
        from peft import PeftModel
        model.language_model = PeftModel.from_pretrained(model.language_model, str(lora_path))
        print(f"  LoRA 已加载: {lora_path}")
    else:
        model.freeze_language_model()
    model.eval()

    print("构建测试集（与训练相同 8:1:1 划分）...")
    _, _, test_dataset = create_robot_dataloaders_split(
        parquet_dir=args.parquet_dir,
        video_root=args.video_root,
        image_processor=model.image_processor,
        tokenizer=model.tokenizer,
        batch_size=args.batch_size,
        frame_skip=args.frame_skip,
        num_workers=args.num_workers,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        split_seed=args.split_seed,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_robot,
        pin_memory=True,
    )
    print(f"  测试集样本数: {len(test_dataset)}")

    total_loss = 0.0
    total_correct_top1 = 0
    total_correct_top5 = 0
    total_n = 0

    print("在测试集上评估...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Test"):
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            n = labels.size(0)

            if use_amp:
                with autocast("cuda", dtype=torch.bfloat16):
                    out = model(
                        pixel_values=pixel_values,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        task_type="robot",
                        labels=labels,
                    )
            else:
                out = model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    task_type="robot",
                    labels=labels,
                )

            loss = out["loss"].item()
            logits = out["logits"].float()
            pred_top1 = logits.argmax(dim=-1)
            _, pred_top5 = logits.topk(5, dim=-1)
            correct_top1 = (pred_top1 == labels).sum().item()
            correct_top5 = (pred_top5 == labels.unsqueeze(1)).any(dim=1).sum().item()

            total_loss += loss * n
            total_correct_top1 += correct_top1
            total_correct_top5 += correct_top5
            total_n += n
            if args.max_samples is not None and total_n >= args.max_samples:
                break

    total_n = max(total_n, 1)
    avg_loss = total_loss / total_n
    accuracy_top1 = total_correct_top1 / total_n
    accuracy_top5 = total_correct_top5 / total_n

    print("\n" + "=" * 60)
    print("测试集评估结果")
    print("=" * 60)
    print(f"  测试集样本数: {total_n}")
    print(f"  Test Loss (交叉熵): {avg_loss:.4f}")
    print(f"  Test Top-1 准确率: {accuracy_top1:.4f} ({accuracy_top1*100:.2f}%)  正确: {total_correct_top1} / {total_n}")
    print(f"  Test Top-5 准确率: {accuracy_top5:.4f} ({accuracy_top5*100:.2f}%)  正确: {total_correct_top5} / {total_n}")
    print("=" * 60)


if __name__ == "__main__":
    main()
