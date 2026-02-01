"""
LLaVA-Instruct-150K 对齐数据集加载器

用于阶段 1 视觉对齐：加载 liuhaotian/LLaVA-Instruct-150K，
产出 pixel_values (SigLIP)、input_ids/labels (Gemma，含 <image> 占位符)。
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Iterator

import torch
from torch.utils.data import Dataset, IterableDataset
from PIL import Image
from transformers import AutoImageProcessor, AutoTokenizer


# LLaVA 原始占位符可能为 <<image>> 或 <image>
LLAVA_IMAGE_PLACEHOLDER = "<<image>>"
OUR_IMAGE_PLACEHOLDER = "<image>"


def _normalize_placeholder(text: str) -> str:
    """将 LLaVA 的 <<image>> 等统一为 <image>"""
    # 兼容 <<image>>、<image>、\n<<image>> 等
    text = re.sub(r"<<\s*image\s*>>", OUR_IMAGE_PLACEHOLDER, text, flags=re.IGNORECASE)
    text = text.replace("\n<image>\n", "\n" + OUR_IMAGE_PLACEHOLDER + "\n")
    return text


def _ensure_image_token(tokenizer: AutoTokenizer) -> None:
    """确保 tokenizer 有 <image> 单 token，没有则添加"""
    if OUR_IMAGE_PLACEHOLDER in tokenizer.get_vocab():
        return
    tokenizer.add_special_tokens({"additional_special_tokens": [OUR_IMAGE_PLACEHOLDER]})


class LLaVAInstruct150KDataset(Dataset):
    """
    LLaVA-Instruct-150K 数据集：图-文对话，用于 Projector 对齐。

    每条样本：一张图 + 多轮对话中的一轮（human + gpt）。
    输出格式满足 Pi06StitchedVLM.forward(task_type="alignment")：
    - pixel_values: SigLIP 图像输入
    - input_ids: 含 <image> 占位符的整序列（prompt + answer）
    - attention_mask: 同长度
    - labels: 同长度，prompt 部分为 -100，仅 answer 部分参与 loss
    """

    def __init__(
        self,
        split: str = "train",
        image_processor: Optional[AutoImageProcessor] = None,
        tokenizer: Optional[AutoTokenizer] = None,
        max_length: int = 512,
        cache_dir: Optional[str] = None,
        dataset_name: str = "liuhaotian/LLaVA-Instruct-150K",
        image_root: Optional[str] = None,
        max_samples: Optional[int] = None,
    ):
        """
        Args:
            split: 数据集划分，如 "train"
            image_processor: SigLIP 的 image processor
            tokenizer: Gemma 的 tokenizer
            max_length: 最大序列长度（含图像占位后的总长）
            cache_dir: HF 数据集缓存目录
            dataset_name: HuggingFace 数据集名
            image_root: 若数据集里 image 列为路径，则为此根目录；若为 None 且列为 path，则尝试用 HF 的 image 解码
            max_samples: 最多使用样本数（用于调试或子集）
        """
        from datasets import load_dataset

        if cache_dir is None:
            project_root = Path(__file__).resolve().parent.parent
            cache_dir = str(project_root / "data" / "llava_instruct_150k")
        Path(cache_dir).mkdir(parents=True, exist_ok=True)

        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.image_root = Path(image_root) if image_root else None

        _ensure_image_token(tokenizer)

        print(f"加载 LLaVA-Instruct-150K: {dataset_name} split={split} ...")
        self.hf_data = load_dataset(
            dataset_name,
            split=split,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )
        if max_samples is not None:
            self.hf_data = self.hf_data.select(range(min(max_samples, len(self.hf_data))))
        print(f"  样本数: {len(self.hf_data)}")

    def __len__(self) -> int:
        return len(self.hf_data)

    def _get_image(self, row: Dict) -> Image.Image:
        """从 row 中取到 PIL Image。支持 HF 的 Image 特征或路径."""
        img = row.get("image")
        if img is None:
            raise ValueError("row 中缺少 image 字段")
        if hasattr(img, "convert"):
            return img.convert("RGB")
        if isinstance(img, dict):
            if "path" in img and self.image_root:
                path = self.image_root / img["path"]
                return Image.open(path).convert("RGB")
            if "bytes" in img:
                import io
                return Image.open(io.BytesIO(img["bytes"])).convert("RGB")
        if isinstance(img, str):
            path = Path(img)
            if self.image_root:
                path = self.image_root / path
            return Image.open(path).convert("RGB")
        raise ValueError(f"无法解析 image 类型: {type(img)}")

    def _conversation_to_prompt_and_answer(self, conversations: List[Dict]) -> tuple:
        """取第一轮 human + gpt 作为 prompt 和 answer。human 中占位符统一为 <image>."""
        if not conversations or len(conversations) < 2:
            return "", ""
        human_msg = conversations[0].get("value", "")
        gpt_msg = conversations[1].get("value", "")
        human_msg = _normalize_placeholder(human_msg)
        return human_msg.strip(), gpt_msg.strip()

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.hf_data[idx]
        image = self._get_image(row)
        conversations = row.get("conversations", [])
        if isinstance(conversations, str):
            import json
            conversations = json.loads(conversations) if conversations else []
        prompt, answer = self._conversation_to_prompt_and_answer(conversations)

        # 先编码 prompt 与 answer，再拼接，保证 prompt_len 准确（labels 仅对 answer 算 loss）
        prompt_part = prompt + "\n"
        prompt_enc = self.tokenizer(
            prompt_part,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True,
        )
        answer_enc = self.tokenizer(
            answer,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length - prompt_enc["input_ids"].size(1),
            add_special_tokens=False,
        )
        input_ids = torch.cat(
            [prompt_enc["input_ids"].squeeze(0), answer_enc["input_ids"].squeeze(0)],
            dim=0,
        )
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        prompt_len = prompt_enc["input_ids"].size(1)
        labels = input_ids.clone()
        labels[:prompt_len] = -100

        # 图像：用 SigLIP 的 image processor
        if self.image_processor is not None:
            pixel_values = self.image_processor(
                images=image,
                return_tensors="pt",
            )["pixel_values"].squeeze(0)
        else:
            pixel_values = torch.zeros(3, 384, 384)  # SigLIP-384 占位

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def collate_alignment(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """对齐任务的 batch 整理：pad input_ids / attention_mask / labels."""
    from torch.nn.utils.rnn import pad_sequence

    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    input_ids = pad_sequence(
        [b["input_ids"] for b in batch],
        batch_first=True,
        padding_value=0,
    )
    attention_mask = pad_sequence(
        [b["attention_mask"] for b in batch],
        batch_first=True,
        padding_value=0,
    )
    labels = pad_sequence(
        [b["labels"] for b in batch],
        batch_first=True,
        padding_value=-100,
    )
    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


class VQAFormatAlignmentDataset(Dataset):
    """
    VQA 格式对齐数据集（备用）：Question + Answer，用于 LLaVA 加载失败时。
    支持 HuggingFaceM4/VQAv2、ChongyanChen/VQAonline 等。
    """
    def __init__(
        self,
        split: str,
        image_processor: Optional[AutoImageProcessor],
        tokenizer: AutoTokenizer,
        max_length: int = 512,
        cache_dir: Optional[str] = None,
        dataset_name: str = "ChongyanChen/VQAonline",
        hf_split: str = "train",
        max_samples: Optional[int] = None,
        image_root: Optional[str] = None,
    ):
        from datasets import load_dataset
        if cache_dir is None:
            project_root = Path(__file__).resolve().parent.parent
            cache_dir = str(project_root / "data" / "vqa_alignment")
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.image_root = Path(image_root) if image_root else None
        _ensure_image_token(tokenizer)
        print(f"加载 VQA 对齐数据集: {dataset_name} split={hf_split} ...")
        self.hf_data = load_dataset(dataset_name, split=hf_split, cache_dir=cache_dir)
        if max_samples is not None:
            self.hf_data = self.hf_data.select(range(min(max_samples, len(self.hf_data))))
        print(f"  样本数: {len(self.hf_data)}")

    def __len__(self) -> int:
        return len(self.hf_data)

    def _get_image(self, row: Dict) -> Image.Image:
        img = row.get("image")
        if img is None:
            raise ValueError("row 缺少 image")
        if hasattr(img, "convert"):
            return img.convert("RGB")
        if isinstance(img, dict) and "bytes" in img:
            import io
            return Image.open(io.BytesIO(img["bytes"])).convert("RGB")
        if isinstance(img, str):
            path = Path(img)
            if self.image_root:
                path = self.image_root / path.name if not path.is_absolute() else path
            if path.exists():
                return Image.open(path).convert("RGB")
            if self.image_root and (self.image_root / img).exists():
                return Image.open(self.image_root / img).convert("RGB")
            # 路径不存在时用占位图，保证训练可跑（与 value_function 一致）
            return Image.new("RGB", (384, 384), color=(128, 128, 128))
        raise ValueError(f"无法解析 image: {type(img)}")

    def _get_qa(self, row: Dict) -> tuple:
        q = row.get("question", row.get("Question", ""))
        a = row.get("answers", row.get("answer", row.get("Answer", "")))
        if isinstance(a, list):
            a = a[0].get("answer", str(a[0])) if a else ""
        return str(q).strip(), str(a).strip()

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.hf_data[idx]
        image = self._get_image(row)
        question, answer = self._get_qa(row)
        prompt_part = f"{OUR_IMAGE_PLACEHOLDER}\nQuestion: {question}\nAnswer: "
        prompt_enc = self.tokenizer(
            prompt_part,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True,
        )
        answer_enc = self.tokenizer(
            answer,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length - prompt_enc["input_ids"].size(1),
            add_special_tokens=False,
        )
        input_ids = torch.cat(
            [prompt_enc["input_ids"].squeeze(0), answer_enc["input_ids"].squeeze(0)],
            dim=0,
        )
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        prompt_len = prompt_enc["input_ids"].size(1)
        labels = input_ids.clone()
        labels[:prompt_len] = -100
        if self.image_processor is not None:
            pixel_values = self.image_processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
        else:
            pixel_values = torch.zeros(3, 384, 384)
        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class TheCauldronVQAv2StreamDataset(IterableDataset):
    """
    HuggingFaceM4/the_cauldron (name="vqav2") 流式数据集：边下边练，无需先下完几百 G。
    格式：images[0] + texts (list of {user, assistant})，取第一轮 user/assistant 作为 question/answer。
    """
    def __init__(
        self,
        image_processor: Optional[AutoImageProcessor],
        tokenizer: AutoTokenizer,
        max_length: int = 512,
        dataset_name: str = "HuggingFaceM4/the_cauldron",
        subset_name: str = "vqav2",
        split: str = "train",
        max_samples: Optional[int] = None,
    ):
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.dataset_name = dataset_name
        self.subset_name = subset_name
        self.split = split
        self.max_samples = max_samples
        _ensure_image_token(tokenizer)
        print(f"加载 The Cauldron 流式数据集: {dataset_name} name={subset_name} split={split} (streaming=True) ...")

    def _format_example(self, example: Dict[str, Any]) -> tuple:
        """从 the_cauldron 一条样本中取出 image, question, answer。"""
        image = example.get("images") or example.get("image")
        if image is not None and (isinstance(image, list) and len(image) > 0 or hasattr(image, "convert")):
            img = image[0] if isinstance(image, list) else image
            if hasattr(img, "convert"):
                image = img.convert("RGB")
            else:
                image = Image.new("RGB", (384, 384), color=(128, 128, 128))
        else:
            image = Image.new("RGB", (384, 384), color=(128, 128, 128))
        conversations = example.get("texts", example.get("conversations", []))
        question, answer = "", ""
        for turn in conversations:
            if isinstance(turn, dict):
                if turn.get("user"):
                    question = str(turn["user"]).strip()
                if turn.get("assistant"):
                    answer = str(turn["assistant"]).strip()
                    break
        return image, question, answer

    def _sample_to_tensors(self, image: Image.Image, question: str, answer: str) -> Dict[str, torch.Tensor]:
        """与 VQAFormatAlignmentDataset 一致的 tokenization 与 label 构造。"""
        prompt_part = f"{OUR_IMAGE_PLACEHOLDER}\nQuestion: {question}\nAnswer: "
        prompt_enc = self.tokenizer(
            prompt_part,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True,
        )
        answer_enc = self.tokenizer(
            answer,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length - prompt_enc["input_ids"].size(1),
            add_special_tokens=False,
        )
        input_ids = torch.cat(
            [prompt_enc["input_ids"].squeeze(0), answer_enc["input_ids"].squeeze(0)],
            dim=0,
        )
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        prompt_len = prompt_enc["input_ids"].size(1)
        labels = input_ids.clone()
        labels[:prompt_len] = -100
        if self.image_processor is not None:
            pixel_values = self.image_processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
        else:
            pixel_values = torch.zeros(3, 384, 384)
        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        from datasets import load_dataset
        stream = load_dataset(
            self.dataset_name,
            name=self.subset_name,
            split=self.split,
            streaming=True,
        )
        n = 0
        for example in stream:
            if self.max_samples is not None and n >= self.max_samples:
                break
            try:
                image, question, answer = self._format_example(example)
                if not question and not answer:
                    continue
                yield self._sample_to_tensors(image, question, answer)
                n += 1
            except Exception:
                continue


def create_alignment_dataloader(
    image_processor: AutoImageProcessor,
    tokenizer: AutoTokenizer,
    batch_size: int = 8,
    max_length: int = 512,
    split: str = "train",
    cache_dir: Optional[str] = None,
    dataset_name: str = "liuhaotian/LLaVA-Instruct-150K",
    image_root: Optional[str] = None,
    max_samples: Optional[int] = None,
    num_workers: int = 4,
    shuffle: bool = True,
    fallback_dataset: str = "ChongyanChen/VQAonline",
    fallback_split: str = "train",
    cauldron_subset: Optional[str] = None,
):
    """创建对齐 DataLoader。
    - 若 dataset_name 为 HuggingFaceM4/the_cauldron 且 cauldron_subset 非空（如 vqav2），使用流式数据集边下边练。
    - 否则优先 LLaVA-Instruct-150K；若失败则用 fallback_dataset。
    """
    from torch.utils.data import DataLoader
    from datasets.exceptions import DatasetGenerationError

    if dataset_name == "HuggingFaceM4/the_cauldron" and (cauldron_subset or "vqav2"):
        subset = cauldron_subset or "vqav2"
        dataset = TheCauldronVQAv2StreamDataset(
            image_processor=image_processor,
            tokenizer=tokenizer,
            max_length=max_length,
            dataset_name=dataset_name,
            subset_name=subset,
            split=split,
            max_samples=max_samples,
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_alignment,
            pin_memory=True,
        )
    dataset = None
    try:
        dataset = LLaVAInstruct150KDataset(
            split=split,
            image_processor=image_processor,
            tokenizer=tokenizer,
            max_length=max_length,
            cache_dir=cache_dir,
            dataset_name=dataset_name,
            image_root=image_root,
            max_samples=max_samples,
        )
    except (DatasetGenerationError, Exception) as e:
        print(f"LLaVA-Instruct-150K 加载失败 ({e})，使用备用数据集: {fallback_dataset}")
        dataset = VQAFormatAlignmentDataset(
            split=split,
            image_processor=image_processor,
            tokenizer=tokenizer,
            max_length=max_length,
            cache_dir=cache_dir,
            dataset_name=fallback_dataset,
            hf_split=fallback_split,
            max_samples=max_samples,
            image_root=image_root,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_alignment,
        pin_memory=True,
    )
