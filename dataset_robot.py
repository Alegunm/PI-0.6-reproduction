"""
阶段 2：机器人价值数据集

与 value_function 同格式（parquet + 视频），使用 SigLIP image_processor + Gemma tokenizer，
输出满足 Pi06StitchedVLM(task_type="robot") 的 pixel_values / input_ids / attention_mask / labels。
"""

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import copy

import numpy as np
import torch
import pyarrow.parquet as pq
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from transformers import AutoImageProcessor, AutoTokenizer

try:
    from decord import VideoReader, cpu
    DECORD_AVAILABLE = True
except ImportError:
    DECORD_AVAILABLE = False

# 占位符需与 model_stitched 中一致
IMAGE_PLACEHOLDER = "<image>"


def _ensure_image_token(tokenizer: AutoTokenizer) -> None:
    if IMAGE_PLACEHOLDER not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": [IMAGE_PLACEHOLDER]})


class RobotValueDatasetStitched(Dataset):
    """
    机器人价值数据集（缝合版）：parquet + 视频，SigLIP 图像 + Gemma 文本。
    """

    def __init__(
        self,
        parquet_dir: str,
        video_root: str,
        image_processor: AutoImageProcessor,
        tokenizer: AutoTokenizer,
        max_steps: int = 500,
        num_value_bins: int = 201,
        text_prompt: str = "Predict the success probability of this robot task.",
        frame_skip: int = 5,
    ):
        self.parquet_dir = Path(parquet_dir)
        self.video_root = Path(video_root)
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_steps = max_steps
        self.num_value_bins = num_value_bins
        self.text_prompt = text_prompt
        self.frame_skip = frame_skip
        self.view_names = ["observation.images.hand", "observation.images.view1", "observation.images.view2"]
        _ensure_image_token(tokenizer)
        self.samples = self._load_samples()
        print(f"RobotValueDatasetStitched: {len(self.samples)} 样本, parquet={parquet_dir}, video={video_root}")

    def _load_samples(self) -> List[Dict]:
        samples = []
        for parquet_file in sorted(self.parquet_dir.glob("episode_*.parquet")):
            try:
                table = pq.read_table(parquet_file)
                df = table.to_pandas()
                episode_index = int(parquet_file.stem.split("_")[-1])
                for _, row in df.iterrows():
                    frame_index = int(row.get("frame_index", row.get("index", row.name)))
                    if frame_index % self.frame_skip != 0:
                        continue
                    raw_value = int(row["raw_value"])
                    samples.append({
                        "episode_index": episode_index,
                        "frame_index": frame_index,
                        "raw_value": raw_value,
                    })
            except Exception as e:
                warnings.warn(f"读取 {parquet_file} 失败: {e}")
                continue
        if not samples:
            raise ValueError(f"未在 {self.parquet_dir} 中找到有效样本")
        return samples

    def _read_frame(self, video_path: Path, frame_index: int) -> np.ndarray:
        if not DECORD_AVAILABLE:
            raise ImportError("需要 decord: pip install decord")
        vr = VideoReader(str(video_path), ctx=cpu(0))
        if frame_index >= len(vr):
            frame_index = len(vr) - 1
        return vr[frame_index].asnumpy()

    def _discretize_value(self, raw_value: int) -> int:
        v_norm = max(-1.0, raw_value / self.max_steps)
        v_shifted = v_norm + 1.0
        bin_id = int(v_shifted * (self.num_value_bins - 1))
        return max(0, min(self.num_value_bins - 1, bin_id))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s = self.samples[idx]
        ep, frame, raw_value = s["episode_index"], s["frame_index"], s["raw_value"]
        video_path = self.video_root / "chunk-000" / self.view_names[0] / f"episode_{ep:06d}.mp4"
        if video_path.exists():
            frame_arr = self._read_frame(video_path, frame)
            image = Image.fromarray(frame_arr).convert("RGB")
        else:
            image = Image.new("RGB", (224, 224), color=(0, 0, 0))
        pixel_values = self.image_processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
        text = f"{IMAGE_PLACEHOLDER}\n{self.text_prompt}"
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        label = self._discretize_value(raw_value)
        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(label, dtype=torch.long),
            "task_type": "robot",
        }


def split_robot_dataset_by_episode(
    dataset: RobotValueDatasetStitched,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[RobotValueDatasetStitched, RobotValueDatasetStitched, RobotValueDatasetStitched]:
    """
    按 episode 划分：训练集 8 : 验证集 1 : 测试集 1（默认）。
    与 value_function 的 split_robot_dataset_by_episode 逻辑一致，保证同一 split_seed 得到相同划分。
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例之和必须为 1.0"

    episodes = sorted(list(set(s["episode_index"] for s in dataset.samples)))
    rng = np.random.RandomState(seed)
    rng.shuffle(episodes)

    n_episodes = len(episodes)
    n_train = int(n_episodes * train_ratio)
    n_val = int(n_episodes * val_ratio)

    train_episodes = set(episodes[:n_train])
    val_episodes = set(episodes[n_train : n_train + n_val])
    test_episodes = set(episodes[n_train + n_val :])

    print("\n数据集划分 (8:1:1 按 episode):")
    print(f"  - 总 episodes: {n_episodes}")
    print(f"  - 训练集 episodes: {len(train_episodes)} ({len(train_episodes)/n_episodes*100:.1f}%)")
    print(f"  - 验证集 episodes: {len(val_episodes)} ({len(val_episodes)/n_episodes*100:.1f}%)")
    print(f"  - 测试集 episodes: {len(test_episodes)} ({len(test_episodes)/n_episodes*100:.1f}%)")

    train_dataset = copy.copy(dataset)
    val_dataset = copy.copy(dataset)
    test_dataset = copy.copy(dataset)

    train_dataset.samples = [s for s in dataset.samples if s["episode_index"] in train_episodes]
    val_dataset.samples = [s for s in dataset.samples if s["episode_index"] in val_episodes]
    test_dataset.samples = [s for s in dataset.samples if s["episode_index"] in test_episodes]

    print(f"  - 训练集样本数: {len(train_dataset.samples)}")
    print(f"  - 验证集样本数: {len(val_dataset.samples)}")
    print(f"  - 测试集样本数: {len(test_dataset.samples)}")

    return train_dataset, val_dataset, test_dataset


def collate_robot(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    from torch.nn.utils.rnn import pad_sequence
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    input_ids = pad_sequence([b["input_ids"] for b in batch], batch_first=True, padding_value=0)
    attention_mask = pad_sequence([b["attention_mask"] for b in batch], batch_first=True, padding_value=0)
    labels = torch.stack([b["labels"] for b in batch])
    task_type = [b["task_type"] for b in batch]
    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "task_type": task_type,
    }


def create_robot_dataloader(
    parquet_dir: str,
    video_root: str,
    image_processor: AutoImageProcessor,
    tokenizer: AutoTokenizer,
    batch_size: int = 8,
    max_steps: int = 500,
    num_value_bins: int = 201,
    frame_skip: int = 5,
    num_workers: int = 4,
    shuffle: bool = True,
    dataset: Optional[RobotValueDatasetStitched] = None,
) -> torch.utils.data.DataLoader:
    """单个 DataLoader。若传入 dataset 则用其样本，否则从 parquet_dir/video_root 新建全量数据集。"""
    if dataset is None:
        dataset = RobotValueDatasetStitched(
            parquet_dir=parquet_dir,
            video_root=video_root,
            image_processor=image_processor,
            tokenizer=tokenizer,
            max_steps=max_steps,
            num_value_bins=num_value_bins,
            frame_skip=frame_skip,
        )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_robot,
        pin_memory=True,
    )


def create_robot_dataloaders_split(
    parquet_dir: str,
    video_root: str,
    image_processor: AutoImageProcessor,
    tokenizer: AutoTokenizer,
    batch_size: int = 8,
    max_steps: int = 500,
    num_value_bins: int = 201,
    frame_skip: int = 5,
    num_workers: int = 4,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    split_seed: int = 42,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, RobotValueDatasetStitched]:
    """
    8:1:1 划分：训练集、验证集、测试集（按 episode）。
    返回 (train_loader, val_loader, test_dataset)。
    测试集以 Dataset 返回，供 eval 脚本按相同划分评估。
    """
    full = RobotValueDatasetStitched(
        parquet_dir=parquet_dir,
        video_root=video_root,
        image_processor=image_processor,
        tokenizer=tokenizer,
        max_steps=max_steps,
        num_value_bins=num_value_bins,
        frame_skip=frame_skip,
    )
    train_ds, val_ds, test_ds = split_robot_dataset_by_episode(
        full, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio, seed=split_seed
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_robot,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_robot,
        pin_memory=True,
    )
    return train_loader, val_loader, test_ds
