"""
论文 Fig.4 风格：Value Function 可视化

- 横轴：Time (s)，纵轴：Value ∈ [-1, 0]（0=成功，与论文一致）
- 选 2 条测试集 episode：一条偏成功（末端 raw_value 高）、一条偏失败（末端 raw_value 低）
- 上图：关键帧缩略图；下图：VF 预测值随时间曲线，红=价值下降，绿=价值上升
- 归一化说明：我们与论文一致，价值在 [-1, 0]，[0,1] 仅用于内部 bin 下标计算
"""

import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from model_stitched import create_stitched_model
from dataset_robot import (
    RobotValueDatasetStitched,
    split_robot_dataset_by_episode,
    create_robot_dataloaders_split,
    collate_robot,
)

try:
    from decord import VideoReader, cpu
    DECORD_AVAILABLE = True
except ImportError:
    DECORD_AVAILABLE = False

# 假设原始视频 10 Hz
FPS = 10
NUM_VALUE_BINS = 201
MAX_STEPS = 500


def bin_to_value(bin_id: int) -> float:
    """bin_id ∈ [0, 200] -> 归一化价值 ∈ [-1, 0]（与论文一致）"""
    v_shifted = bin_id / (NUM_VALUE_BINS - 1)
    return v_shifted - 1.0


def load_episode_frame(video_root: Path, episode_index: int, frame_index: int) -> np.ndarray:
    """读取单帧 RGB [H,W,3]"""
    if not DECORD_AVAILABLE:
        return np.zeros((224, 224, 3), dtype=np.uint8)
    view_name = "observation.images.hand"
    video_path = video_root / "chunk-000" / view_name / f"episode_{episode_index:06d}.mp4"
    if not video_path.exists():
        return np.zeros((224, 224, 3), dtype=np.uint8)
    vr = VideoReader(str(video_path), ctx=cpu(0))
    if frame_index >= len(vr):
        frame_index = len(vr) - 1
    return vr[frame_index].asnumpy()


def main():
    p = argparse.ArgumentParser(description="Value Function 可视化（Fig.4 风格）")
    p.add_argument("--checkpoint", type=str, default="./outputs/robot_vf")
    p.add_argument("--parquet_dir", type=str, required=True)
    p.add_argument("--video_root", type=str, required=True)
    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--test_ratio", type=float, default=0.1)
    p.add_argument("--split_seed", type=int, default=42)
    p.add_argument("--frame_skip", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=8, help="推理时每批帧数")
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--out", type=str, default="./value_function_visualization.png")
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--dpi", type=int, default=150)
    args = p.parse_args()

    ckpt = Path(args.checkpoint)
    projector_path = str(ckpt / "best_projector.pt")
    value_head_path = str(ckpt / "best_value_head.pt")
    lora_path = ckpt / "best_lora_adapter"
    if not lora_path.exists():
        lora_path = None

    device = args.device
    use_amp = not args.no_amp
    video_root = Path(args.video_root)

    print("加载模型...")
    model = create_stitched_model(device=device, cache_dir=args.cache_dir)
    model.freeze_vision_tower()
    model.load_projector(projector_path)
    model.load_value_head(value_head_path)
    if lora_path is not None:
        from peft import PeftModel
        model.language_model = PeftModel.from_pretrained(model.language_model, str(lora_path))
    else:
        model.freeze_language_model()
    model.eval()

    print("构建测试集（8:1:1）...")
    _, _, test_dataset = create_robot_dataloaders_split(
        parquet_dir=args.parquet_dir,
        video_root=args.video_root,
        image_processor=model.image_processor,
        tokenizer=model.tokenizer,
        batch_size=args.batch_size,
        frame_skip=args.frame_skip,
        num_workers=0,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        split_seed=args.split_seed,
    )

    # 按 episode 分组：(episode_index -> [(idx, frame_index, raw_value), ...] 按 frame_index 排序)
    ep_to_items = defaultdict(list)
    for idx, s in enumerate(test_dataset.samples):
        ep_to_items[s["episode_index"]].append((idx, s["frame_index"], s["raw_value"]))
    for ep in ep_to_items:
        ep_to_items[ep].sort(key=lambda x: x[1])

    # 选两条：成功=末端 raw_value 最大，失败=末端 raw_value 最小
    episodes_sorted_by_outcome = sorted(
        ep_to_items.keys(),
        key=lambda ep: ep_to_items[ep][-1][2],
        reverse=True,
    )
    if len(episodes_sorted_by_outcome) < 2:
        print("测试集 episode 不足 2 条，无法画双图")
        return
    ep_success = episodes_sorted_by_outcome[0]
    ep_failure = episodes_sorted_by_outcome[-1]
    print(f"成功向 episode: {ep_success} (末端 raw_value={ep_to_items[ep_success][-1][2]})")
    print(f"失败向 episode: {ep_failure} (末端 raw_value={ep_to_items[ep_failure][-1][2]})")

    def run_episode(episode_index: int, items: list):
        """items: [(idx, frame_index, raw_value), ...] 已按 frame_index 排序"""
        indices = [x[0] for x in items]
        frame_indices = [x[1] for x in items]
        raw_values = [x[2] for x in items]
        times = [fi / FPS for fi in frame_indices]
        gt_values = [max(-1.0, rv / MAX_STEPS) for rv in raw_values]

        pred_values = []
        for start in range(0, len(indices), args.batch_size):
            batch_idx = indices[start : start + args.batch_size]
            batch = [test_dataset[i] for i in batch_idx]
            pixel_values = torch.stack([b["pixel_values"] for b in batch]).to(device)
            input_ids = torch.nn.utils.rnn.pad_sequence(
                [b["input_ids"] for b in batch], batch_first=True, padding_value=0
            ).to(device)
            attention_mask = torch.nn.utils.rnn.pad_sequence(
                [b["attention_mask"] for b in batch], batch_first=True, padding_value=0
            ).to(device)
            with torch.no_grad():
                if use_amp:
                    with autocast("cuda", dtype=torch.bfloat16):
                        out = model(
                            pixel_values=pixel_values,
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            task_type="robot",
                        )
                else:
                    out = model(
                        pixel_values=pixel_values,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        task_type="robot",
                    )
            logits = out["logits"].float()
            pred_bins = logits.argmax(dim=-1).cpu().numpy()
            for b in pred_bins:
                pred_values.append(bin_to_value(int(b)))
        return times, gt_values, pred_values, frame_indices, episode_index

    print("推理两条 episode...")
    success_times, success_gt, success_pred, success_frames, ep_succ = run_episode(ep_success, ep_to_items[ep_success])
    failure_times, failure_gt, failure_pred, failure_frames, ep_fail = run_episode(ep_failure, ep_to_items[ep_failure])

    # 选关键帧索引（首、1/4、1/2、3/4、末）
    def key_frame_indices(n: int):
        if n <= 5:
            return list(range(n))
        return [0, n // 4, n // 2, 3 * n // 4, n - 1]

    # 布局：2 行 2 列。上排左/右各 5 张关键帧；下排左/右各一条 Value vs Time 曲线（红=下降，绿=上升）
    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.25, height_ratios=[1, 1.2])

    # 成功向：上排 5 张关键帧
    k_succ = key_frame_indices(len(success_frames))
    sub_gs_succ = gs[0, 0].subgridspec(1, len(k_succ))
    for i, ki in enumerate(k_succ):
        ax = fig.add_subplot(sub_gs_succ[0, i])
        img = load_episode_frame(video_root, ep_succ, success_frames[ki])
        ax.imshow(img)
        ax.set_title(f"t={success_times[ki]:.1f}s", fontsize=8)
        ax.axis("off")

    # 失败向：上排 5 张关键帧
    k_fail = key_frame_indices(len(failure_frames))
    sub_gs_fail = gs[0, 1].subgridspec(1, len(k_fail))
    for i, ki in enumerate(k_fail):
        ax = fig.add_subplot(sub_gs_fail[0, i])
        img = load_episode_frame(video_root, ep_fail, failure_frames[ki])
        ax.imshow(img)
        ax.set_title(f"t={failure_times[ki]:.1f}s", fontsize=8)
        ax.axis("off")

    # 成功向：下排 Value vs Time
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.plot(success_times, success_pred, "k-", linewidth=1.5, label="Predicted value")
    ax1.set_ylim(-1.05, 0.05)
    ax1.set_ylabel("Value")
    ax1.set_xlabel("Time (s)")
    ax1.set_title("Successful-like episode (VF)")
    ax1.grid(True, alpha=0.3)
    for i in range(len(success_times) - 1):
        t0, t1 = success_times[i], success_times[i + 1]
        v1 = success_pred[i + 1]
        v0 = success_pred[i]
        if v1 < v0:
            ax1.axvspan(t0, t1, color="red", alpha=0.2)
        else:
            ax1.axvspan(t0, t1, color="green", alpha=0.2)
    ax1.legend(loc="lower right", fontsize=8)

    # 失败向：下排 Value vs Time
    ax2 = fig.add_subplot(gs[1, 1])
    ax2.plot(failure_times, failure_pred, "k-", linewidth=1.5, label="Predicted value")
    ax2.set_ylim(-1.05, 0.05)
    ax2.set_ylabel("Value")
    ax2.set_xlabel("Time (s)")
    ax2.set_title("Failure-like episode (VF)")
    ax2.grid(True, alpha=0.3)
    for i in range(len(failure_times) - 1):
        t0, t1 = failure_times[i], failure_times[i + 1]
        v1 = failure_pred[i + 1]
        v0 = failure_pred[i]
        if v1 < v0:
            ax2.axvspan(t0, t1, color="red", alpha=0.2)
        else:
            ax2.axvspan(t0, t1, color="green", alpha=0.2)
    ax2.legend(loc="lower right", fontsize=8)

    fig.suptitle("Value Function Visualization (Value ∈ [-1, 0], 0 = success; red = drop, green = rise)", fontsize=10)
    plt.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    plt.close()
    print(f"已保存: {args.out}")


if __name__ == "__main__":
    main()
