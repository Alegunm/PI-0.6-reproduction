"""
阶段 1：视觉对齐训练 (Alignment Pre-training)

使用 LLaVA-Instruct-150K，只训练 Projector，让 Gemma 3 能“看懂” SigLIP 的视觉特征。
冻结：SigLIP + Gemma；可训练：Projector。
"""

import argparse
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from model_stitched import Pi06StitchedVLM, create_stitched_model
from dataset_alignment import (
    LLaVAInstruct150KDataset,
    create_alignment_dataloader,
    collate_alignment,
)


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, eta_min=1e-6, last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(eta_min, cosine_decay)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def parse_args():
    p = argparse.ArgumentParser(description="阶段 1：VLM Projector 对齐 (LLaVA-Instruct-150K)")
    p.add_argument("--output_dir", type=str, default="./outputs_alignment", help="输出与 checkpoint 目录")
    p.add_argument("--batch_size", type=int, default=8, help="每 GPU 批次大小")
    p.add_argument("--max_epochs", type=int, default=2, help="训练轮数，建议 1–2")
    p.add_argument("--lr", type=float, default=2e-5, help="Projector 学习率")
    p.add_argument("--warmup_steps", type=int, default=200, help="warmup 步数")
    p.add_argument("--max_length", type=int, default=512, help="最大序列长度")
    p.add_argument("--gradient_accumulation_steps", type=int, default=2, help="梯度累积步数")
    p.add_argument("--logging_steps", type=int, default=50, help="打印/日志间隔")
    p.add_argument("--save_steps", type=int, default=2000, help="保存 checkpoint 间隔")
    p.add_argument("--max_grad_norm", type=float, default=1.0, help="梯度裁剪")
    p.add_argument("--use_amp", action="store_true", default=True, help="混合精度")
    p.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    p.add_argument("--dataset_name", type=str, default="liuhaotian/LLaVA-Instruct-150K", help="HF 数据集名；用 HuggingFaceM4/the_cauldron 时配合 --cauldron_subset vqav2 流式边下边练")
    p.add_argument("--cauldron_subset", type=str, default=None, help="The Cauldron 子集名，如 vqav2；非空时用流式加载")
    p.add_argument("--split", type=str, default="train", help="数据集划分")
    p.add_argument("--cache_dir", type=str, default=None, help="HF 缓存目录")
    p.add_argument("--image_root", type=str, default=None, help="若数据集 image 为路径，则为此根目录")
    p.add_argument("--max_samples", type=int, default=None, help="最多使用样本数（调试用）")
    p.add_argument("--load_in_4bit", action="store_true", default=False, help="Gemma 4-bit 量化")
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 创建模型（只训 Projector）
    print("加载缝合 VLM（仅 Projector 可训练）...")
    model = create_stitched_model(
        device=device,
        load_in_4bit=args.load_in_4bit,
        cache_dir=args.cache_dir,
    )
    model.freeze_vision_tower()
    model.freeze_language_model()
    # Value Head 不参与阶段 1，可冻结
    for p in model.value_head.parameters():
        p.requires_grad = False
    # 4-bit 时 language_model 用 device_map，projector/value_head 需显式 to(device)
    model.projector.to(device)
    model.value_head.to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    scaler = GradScaler("cuda") if args.use_amp else None

    # 2. 数据
    train_loader = create_alignment_dataloader(
        image_processor=model.image_processor,
        tokenizer=model.tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        split=args.split,
        cache_dir=args.cache_dir,
        dataset_name=args.dataset_name,
        image_root=args.image_root,
        max_samples=args.max_samples,
        num_workers=args.num_workers,
        shuffle=True,
        cauldron_subset=args.cauldron_subset,
    )
    try:
        num_training_steps = (len(train_loader) * args.max_epochs) // args.gradient_accumulation_steps
    except TypeError:
        num_training_steps = 50000
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=num_training_steps,
        eta_min=1e-6,
    )

    global_step = 0
    best_loss = float("inf")

    for epoch in range(args.max_epochs):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.max_epochs}")

        optimizer.zero_grad()
        for step, batch in enumerate(pbar):
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            if args.use_amp:
                with autocast("cuda", dtype=torch.bfloat16):
                    out = model(
                        pixel_values=pixel_values,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        task_type="alignment",
                        labels=labels,
                    )
                    loss = out["loss"] / args.gradient_accumulation_steps
                scaler.scale(loss).backward()
            else:
                out = model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    task_type="alignment",
                    labels=labels,
                )
                (out["loss"] / args.gradient_accumulation_steps).backward()

            epoch_loss += out["loss"].item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.max_grad_norm > 0:
                    if args.use_amp:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
                if args.use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % args.logging_steps == 0:
                    lr = scheduler.get_last_lr()[0]
                    pbar.set_postfix(loss=f"{out['loss'].item():.4f}", lr=f"{lr:.2e}")

                if global_step % args.save_steps == 0:
                    ckpt_dir = output_dir / f"checkpoint-{global_step}"
                    ckpt_dir.mkdir(parents=True, exist_ok=True)
                    model.save_projector(str(ckpt_dir / "projector.pt"))
                    if out["loss"].item() < best_loss:
                        best_loss = out["loss"].item()
                        model.save_projector(str(output_dir / "best_projector.pt"))

        try:
            num_batches = len(train_loader)
        except TypeError:
            num_batches = step + 1
        avg_loss = epoch_loss / max(num_batches, 1)
        print(f"Epoch {epoch+1} 平均 loss: {avg_loss:.4f}")

    # 最终保存
    model.save_projector(str(output_dir / "projector_final.pt"))
    print(f"阶段 1 完成。Projector 已保存到: {output_dir}")


if __name__ == "__main__":
    main()
