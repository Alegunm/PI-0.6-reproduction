"""
阶段 2：机器人价值微调 (Robot Value Fine-tuning)

推荐方案（46G A6000）：Projector + Value Head + LoRA(Gemma)。
冻结：SigLIP；可训练：Projector + Value Head + Gemma LoRA 适配器。
"""

import argparse
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from model_stitched import Pi06StitchedVLM, create_stitched_model
from dataset_robot import create_robot_dataloaders_split, collate_robot


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, eta_min=1e-6, last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(eta_min, cosine_decay)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def parse_args():
    p = argparse.ArgumentParser(description="阶段 2：机器人价值微调")
    p.add_argument("--parquet_dir", type=str, required=True, help="Parquet 目录，如 labeled_dataset_full/data/chunk-000/")
    p.add_argument("--video_root", type=str, required=True, help="视频根目录")
    p.add_argument("--projector_path", type=str, required=True, help="阶段 1 保存的 projector 权重，如 outputs_alignment/best_projector.pt")
    p.add_argument("--output_dir", type=str, default="./outputs_robot_vf", help="输出与 checkpoint 目录")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=1000)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--use_amp", action="store_true", default=True)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_steps", type=int, default=500, help="价值归一化用最大步数")
    p.add_argument("--num_value_bins", type=int, default=201)
    p.add_argument("--frame_skip", type=int, default=5)
    p.add_argument("--train_projector", action="store_true", help="训 Projector（与 Value Head）")
    p.add_argument("--no_train_projector", dest="train_projector", action="store_false", help="不训 Projector（默认：训）")
    p.set_defaults(train_projector=True)
    p.add_argument("--use_lora", action="store_true", help="对 Gemma 使用 LoRA（推荐）")
    p.add_argument("--no_lora", dest="use_lora", action="store_false", help="关闭 LoRA（默认：开 LoRA）")
    p.set_defaults(use_lora=True)
    p.add_argument("--lora_r", type=int, default=16, help="LoRA 秩")
    p.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    p.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    p.add_argument("--train_lm", action="store_true", help="全参数解冻语言模型（显存大，慎用）")
    p.add_argument("--load_in_4bit", action="store_true", default=False)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--cache_dir", type=str, default=None)
    # 8:1:1 划分：训练 / 验证 / 测试（每次启动自动验证）
    p.add_argument("--train_ratio", type=float, default=0.8, help="训练集比例（默认 0.8）")
    p.add_argument("--val_ratio", type=float, default=0.1, help="验证集比例（默认 0.1）")
    p.add_argument("--test_ratio", type=float, default=0.1, help="测试集比例（默认 0.1）")
    p.add_argument("--split_seed", type=int, default=42, help="划分随机种子，与 eval 一致以保证同一测试集")
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 创建模型并加载阶段 1 的 Projector
    train_projector = args.train_projector
    use_lora = args.use_lora

    print("加载缝合 VLM...")
    model = create_stitched_model(
        device=device,
        load_in_4bit=args.load_in_4bit,
        cache_dir=args.cache_dir,
    )
    model.freeze_vision_tower()
    model.load_projector(args.projector_path)
    if not train_projector:
        model.freeze_projector()
    if use_lora:
        model.apply_lora(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout)
    elif not args.train_lm:
        model.freeze_language_model()

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    scaler = GradScaler("cuda") if args.use_amp else None

    # 2. 机器人数据（8:1:1 划分：训练 / 验证 / 测试）
    train_loader, val_loader, _test_dataset = create_robot_dataloaders_split(
        parquet_dir=args.parquet_dir,
        video_root=args.video_root,
        image_processor=model.image_processor,
        tokenizer=model.tokenizer,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        num_value_bins=args.num_value_bins,
        frame_skip=args.frame_skip,
        num_workers=args.num_workers,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        split_seed=args.split_seed,
    )
    num_training_steps = (len(train_loader) * args.max_epochs) // args.gradient_accumulation_steps
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=num_training_steps,
        eta_min=1e-6,
    )

    global_step = 0
    best_val_loss = float("inf")

    def run_validation():
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="验证", leave=False):
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
                val_loss += out["loss"].item() * labels.size(0)
                pred = out["logits"].argmax(dim=-1)
                val_correct += (pred == labels).sum().item()
                val_total += labels.size(0)
        model.train()
        avg_val_loss = val_loss / max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)
        return avg_val_loss, val_acc

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
                        task_type="robot",
                        labels=labels,
                    )
                    loss = out["loss"] / args.gradient_accumulation_steps
                scaler.scale(loss).backward()
            else:
                out = model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    task_type="robot",
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
                    model.save_value_head(str(ckpt_dir / "value_head.pt"))
                    if train_projector:
                        model.save_projector(str(ckpt_dir / "projector.pt"))
                    if use_lora:
                        model.language_model.save_pretrained(str(ckpt_dir / "lora_adapter"))

        avg_loss = epoch_loss / len(train_loader)
        # 每轮结束自动跑验证集，并依验证集 loss 保存 best
        val_loss, val_acc = run_validation()
        print(f"Epoch {epoch+1} 训练 loss: {avg_loss:.4f}  验证 loss: {val_loss:.4f}  验证 Top-1: {val_acc:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_value_head(str(output_dir / "best_value_head.pt"))
            if train_projector:
                model.save_projector(str(output_dir / "best_projector.pt"))
            if use_lora:
                model.language_model.save_pretrained(str(output_dir / "best_lora_adapter"))
            print(f"  -> 已按验证集保存 best (val_loss={val_loss:.4f})")

    model.save_value_head(str(output_dir / "value_head_final.pt"))
    if train_projector:
        model.save_projector(str(output_dir / "projector_final.pt"))
    if use_lora:
        model.language_model.save_pretrained(str(output_dir / "lora_adapter_final"))
    print(f"阶段 2 完成。输出目录: {output_dir}")


if __name__ == "__main__":
    main()
