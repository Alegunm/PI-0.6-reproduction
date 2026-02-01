"""
π_0.6 多模态VLM - SigLIP + Gemma 3-270M

架构：
- Vision Tower: SigLIP-SO400M (输出 1152 维)
- Projector: MLP (1152 -> 640)
- Language Model: Gemma 3-270M (输入 640 维)
- Value Head: Linear (640 -> 201)，用于机器人价值预测

训练策略：
- 阶段 1 (Alignment): 冻结 SigLIP + Gemma，只训 Projector，用 VQA 数据
- 阶段 2 (Robot VF): 冻结 SigLIP，解冻 Projector + Value Head，用机器人数据
"""

import torch
import torch.nn as nn
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoImageProcessor
)
from typing import Optional, Dict, Tuple
from pathlib import Path
import os


class Pi06StitchedVLM(nn.Module):
    """
    手动缝合的 VLM：SigLIP (Vision) + Projector + Gemma 3-270M (Language) + Value Head
    
    组件：
    1. vision_tower: SigLIP-SO400M，冻结不训
    2. projector: MLP (1152->640)，阶段 1 训练对齐
    3. language_model: Gemma 3-270M，阶段 1 冻结，阶段 2 可选微调/LoRA
    4. value_head: Linear (640->201)，阶段 2 训练
    """
    
    def __init__(
        self,
        vision_model_name: str = "google/siglip-so400m-patch14-384",
        language_model_name: str = "google/gemma-3-270m-it",
        num_value_bins: int = 201,
        projector_hidden_size: int = 1152,
        language_hidden_size: int = 640,
        cache_dir: Optional[str] = None,
        device: str = "cuda",
        load_in_4bit: bool = False
    ):
        """
        初始化缝合 VLM
        
        Args:
            vision_model_name: SigLIP 模型路径
            language_model_name: Gemma 3 模型路径
            num_value_bins: 价值 bins 数量
            projector_hidden_size: Projector 输入维度（SigLIP 输出维度）
            language_hidden_size: 语言模型隐藏维度
            cache_dir: 模型缓存目录
            device: 设备
            load_in_4bit: 是否对语言模型使用 4-bit 量化
        """
        super().__init__()
        
        # 设置缓存目录
        if cache_dir is None:
            current_file = Path(__file__)
            project_root = current_file.parent.parent
            cache_dir = str(project_root / "models")
            Path(cache_dir).mkdir(exist_ok=True)
        
        self.device = device
        self.num_value_bins = num_value_bins
        self.projector_hidden_size = projector_hidden_size
        self.language_hidden_size = language_hidden_size
        
        # HuggingFace token
        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
        
        print(f"初始化缝合 VLM...")
        print(f"  Vision Tower: {vision_model_name}")
        print(f"  Language Model: {language_model_name}")
        print(f"  缓存目录: {cache_dir}")
        
        # 1. 加载视觉塔 (SigLIP)，仅用 vision 部分（完整 SigLIP 含 vision+text，需只取 vision_model）
        print("\n[1/4] 加载 SigLIP Vision Tower...")
        _vision = AutoModel.from_pretrained(
            vision_model_name,
            cache_dir=cache_dir,
            token=hf_token,
            torch_dtype=torch.bfloat16
        )
        if hasattr(_vision, "vision_model"):
            self.vision_tower = _vision.vision_model
        else:
            self.vision_tower = _vision
        self.vision_tower = self.vision_tower.to(device)
        # 冻结视觉塔（永远不训练）
        for param in self.vision_tower.parameters():
            param.requires_grad = False
        self.vision_tower.eval()
        
        # 获取视觉特征维度
        vision_config = getattr(self.vision_tower, "config", getattr(_vision, "config", None))
        if vision_config is not None and hasattr(vision_config, 'hidden_size'):
            actual_vision_dim = vision_config.hidden_size
        elif vision_config is not None and hasattr(vision_config, 'vision_config') and hasattr(vision_config.vision_config, 'hidden_size'):
            actual_vision_dim = vision_config.vision_config.hidden_size
        else:
            actual_vision_dim = projector_hidden_size
        
        print(f"  ✓ SigLIP 加载完成，输出维度: {actual_vision_dim}")
        
        # 2. 构建映射层 (Projector)
        print("\n[2/4] 构建 Projector (Vision -> Language)...")
        self.projector = nn.Sequential(
            nn.Linear(actual_vision_dim, actual_vision_dim),
            nn.GELU(),
            nn.Linear(actual_vision_dim, language_hidden_size)
        )
        # Projector 默认可训练，会在阶段 1 训练
        print(f"  ✓ Projector: {actual_vision_dim} -> {language_hidden_size}")
        
        # 3. 加载语言模型 (Gemma 3-270M)
        print("\n[3/4] 加载 Gemma 3-270M Language Model...")
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            self.language_model = AutoModelForCausalLM.from_pretrained(
                language_model_name,
                cache_dir=cache_dir,
                token=hf_token,
                quantization_config=bnb_config,
                device_map="auto"
            )
            print("  ✓ Gemma 3-270M 加载完成 (4-bit 量化)")
        else:
            self.language_model = AutoModelForCausalLM.from_pretrained(
                language_model_name,
                cache_dir=cache_dir,
                token=hf_token,
                torch_dtype=torch.bfloat16
            )
            print("  ✓ Gemma 3-270M 加载完成 (bfloat16)")
        
        # 默认冻结语言模型（阶段 1 不训练）
        for param in self.language_model.parameters():
            param.requires_grad = False
        
        # 4. 构建价值头 (Value Head)
        print("\n[4/4] 构建 Value Head...")
        self.value_head = nn.Linear(language_hidden_size, num_value_bins)
        print(f"  ✓ Value Head: {language_hidden_size} -> {num_value_bins}")
        
        # 保存配置
        self.vision_model_name = vision_model_name
        self.language_model_name = language_model_name
        
        # 获取 tokenizer（用于查找 image token id），并添加 <image> 占位符
        self.tokenizer = AutoTokenizer.from_pretrained(
            language_model_name,
            cache_dir=cache_dir,
            token=hf_token
        )
        if "<image>" not in self.tokenizer.get_vocab():
            self.tokenizer.add_special_tokens({"additional_special_tokens": ["<image>"]})
            self.language_model.resize_token_embeddings(len(self.tokenizer))
        
        # 获取 image processor
        self.image_processor = AutoImageProcessor.from_pretrained(
            vision_model_name,
            cache_dir=cache_dir,
            token=hf_token
        )
        
        print(f"\n缝合 VLM 初始化完成！")
        self._print_trainable_params()
    
    def _print_trainable_params(self):
        """打印可训练参数统计"""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"\n可训练参数: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")
        
        # 分组统计
        vision_params = sum(p.numel() for p in self.vision_tower.parameters())
        proj_params = sum(p.numel() for p in self.projector.parameters())
        lang_params = sum(p.numel() for p in self.language_model.parameters())
        value_params = sum(p.numel() for p in self.value_head.parameters())
        
        print(f"  - Vision Tower: {vision_params:,}")
        print(f"  - Projector: {proj_params:,}")
        print(f"  - Language Model: {lang_params:,}")
        print(f"  - Value Head: {value_params:,}")
    
    def freeze_vision_tower(self):
        """冻结视觉塔"""
        for param in self.vision_tower.parameters():
            param.requires_grad = False
        print("✓ Vision Tower 已冻结")
    
    def freeze_language_model(self):
        """冻结语言模型"""
        for param in self.language_model.parameters():
            param.requires_grad = False
        print("✓ Language Model 已冻结")
    
    def unfreeze_language_model(self):
        """解冻语言模型"""
        for param in self.language_model.parameters():
            param.requires_grad = True
        print("✓ Language Model 已解冻")
    
    def unfreeze_projector(self):
        """解冻 Projector"""
        for param in self.projector.parameters():
            param.requires_grad = True
        print("✓ Projector 已解冻")
    
    def freeze_projector(self):
        """冻结 Projector"""
        for param in self.projector.parameters():
            param.requires_grad = False
        print("✓ Projector 已冻结")
    
    def apply_lora(
        self,
        r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        target_modules: Optional[list] = None,
    ):
        """对语言模型应用 LoRA（阶段 2 推荐：Projector + Value Head + LoRA）。"""
        try:
            from peft import LoraConfig, get_peft_model, TaskType
        except ImportError:
            raise ImportError("请安装 peft: pip install peft")
        if target_modules is None:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        lora_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        self.language_model = get_peft_model(self.language_model, lora_config)
        print(f"✓ LoRA 已应用到 Language Model (r={r}, alpha={lora_alpha})")
        self._print_trainable_params()
    
    def get_image_token_id(self) -> Optional[int]:
        """获取 image token 的 ID"""
        # 尝试多种可能的 image token
        for token_name in ["<image>", "<|image|>", "<img>", "<image_soft_token>"]:
            try:
                token_id = self.tokenizer.convert_tokens_to_ids(token_name)
                if token_id != self.tokenizer.unk_token_id:
                    return token_id
            except:
                continue
        
        # 如果没有找到，可能需要添加
        # 这里我们用一个特殊 token 代替
        print("  ⚠️  未找到 <image> token，将使用 <unk> 占位")
        return self.tokenizer.unk_token_id
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        task_type: str = "alignment",
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            pixel_values: 图像输入 [batch_size, 3, H, W]
            input_ids: 文本输入 [batch_size, seq_len]，包含 <image> token 占位符
            attention_mask: 注意力掩码 [batch_size, seq_len]
            task_type: "alignment" (VQA) 或 "robot" (价值预测)
            labels: 标签
                - alignment: [batch_size, seq_len]，用于 next token prediction
                - robot: [batch_size]，用于 value bin 分类
            return_dict: 是否返回字典
        
        Returns:
            输出字典，包含 logits 和 loss
        """
        batch_size = pixel_values.size(0)
        device = pixel_values.device
        
        # ========== Step 1: 视觉编码 ==========
        # SigLIP 前向，得到视觉特征 [batch_size, num_patches, 1152]
        with torch.no_grad():  # Vision Tower 永远冻结
            vision_outputs = self.vision_tower(pixel_values=pixel_values)
            # 获取最后一层隐藏状态
            if hasattr(vision_outputs, 'last_hidden_state'):
                vision_features = vision_outputs.last_hidden_state  # [B, num_patches, 1152]
            elif hasattr(vision_outputs, 'pooler_output'):
                vision_features = vision_outputs.pooler_output.unsqueeze(1)  # [B, 1, 1152]
            else:
                # 兜底：直接取第一个返回值
                vision_features = vision_outputs[0]
        
        # ========== Step 2: Projector 映射 ==========
        # [batch_size, num_patches, 1152] -> [batch_size, num_patches, 640]
        image_embeds = self.projector(vision_features)
        num_image_tokens = image_embeds.size(1)
        
        # ========== Step 3: 文本编码 ==========
        # 获取语言模型的 embedding 层（兼容 Gemma2/Gemma3 等，用 get_input_embeddings）
        embed_layer = self.language_model.get_input_embeddings()
        text_embeds = embed_layer(input_ids)  # [B, seq_len, 640]
        
        # ========== Step 4: 模态拼接 (Stitching) ==========
        # 策略：找到 input_ids 中的 <image> token，用 image_embeds 替换它
        # 如果没有 <image> token，就简单地把图像 embeddings 拼在文本前面
        
        image_token_id = self.get_image_token_id()
        
        combined_embeds_list = []
        combined_attention_mask_list = []
        
        for i in range(batch_size):
            # 当前样本的 text embeds 和 attention mask
            curr_text_embeds = text_embeds[i]  # [seq_len, 640]
            curr_input_ids = input_ids[i]  # [seq_len]
            curr_attention = attention_mask[i] if attention_mask is not None else torch.ones_like(input_ids[i])
            
            # 查找 <image> token 的位置
            image_token_mask = (curr_input_ids == image_token_id)
            image_positions = torch.where(image_token_mask)[0]
            
            if len(image_positions) > 0:
                # 有 <image> token：用 image_embeds 替换第一个 <image> token 位置
                first_img_pos = image_positions[0].item()
                
                # 拼接：text[:img_pos] + image_embeds + text[img_pos+1:]
                combined = torch.cat([
                    curr_text_embeds[:first_img_pos],  # 前文
                    image_embeds[i],  # 图像特征
                    curr_text_embeds[first_img_pos + 1:]  # 后文
                ], dim=0)
                
                # 对应的 attention mask
                combined_mask = torch.cat([
                    curr_attention[:first_img_pos],
                    torch.ones(num_image_tokens, dtype=curr_attention.dtype, device=device),
                    curr_attention[first_img_pos + 1:]
                ], dim=0)
            else:
                # 没有 <image> token：直接把图像拼在文本前面
                combined = torch.cat([image_embeds[i], curr_text_embeds], dim=0)
                combined_mask = torch.cat([
                    torch.ones(num_image_tokens, dtype=curr_attention.dtype, device=device),
                    curr_attention
                ], dim=0)
            
            combined_embeds_list.append(combined)
            combined_attention_mask_list.append(combined_mask)
        
        # Pad 到同一长度
        max_len = max(emb.size(0) for emb in combined_embeds_list)
        padded_embeds = []
        padded_masks = []
        
        for emb, mask in zip(combined_embeds_list, combined_attention_mask_list):
            pad_len = max_len - emb.size(0)
            if pad_len > 0:
                # Pad embeddings with zeros
                emb_padded = torch.cat([
                    emb,
                    torch.zeros(pad_len, emb.size(1), dtype=emb.dtype, device=device)
                ], dim=0)
                # Pad mask with zeros
                mask_padded = torch.cat([
                    mask,
                    torch.zeros(pad_len, dtype=mask.dtype, device=device)
                ], dim=0)
            else:
                emb_padded = emb
                mask_padded = mask
            
            padded_embeds.append(emb_padded)
            padded_masks.append(mask_padded)
        
        combined_embeds = torch.stack(padded_embeds, dim=0)  # [B, max_len, 640]
        combined_attention_mask = torch.stack(padded_masks, dim=0)  # [B, max_len]
        
        # ========== Step 5: 语言模型前向 ==========
        # 直接传入 embeddings（不用 input_ids）
        lm_outputs = self.language_model(
            inputs_embeds=combined_embeds,
            attention_mask=combined_attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        
        # ========== Step 6: 任务路由 ==========
        result = {}
        
        if task_type == "alignment" or task_type == "vqa":
            # VQA / Alignment 任务：使用 LM Head 做 next token prediction
            lm_logits = lm_outputs.logits  # [B, max_len, vocab_size]
            result["logits"] = lm_logits
            result["task_type"] = "alignment"
            
            # 计算 Causal LM Loss：combined 序列中插入了 num_image_tokens，需构造与 logits 同长的 labels
            if labels is not None:
                # labels: [B, orig_seq_len]，原始 token 序列的 target；combined 长度为 orig_len-1+num_image_tokens
                expanded_labels_list = []
                for i in range(batch_size):
                    curr_input_ids = input_ids[i]
                    curr_labels = labels[i]
                    orig_len = (curr_input_ids != 0).sum().item()  # 非 pad 长度
                    if orig_len == 0:
                        orig_len = curr_input_ids.size(0)
                    image_token_mask = (curr_input_ids == image_token_id)
                    image_positions = torch.where(image_token_mask)[0]
                    if len(image_positions) > 0:
                        first_img_pos = image_positions[0].item()
                        # 扩展后长度
                        exp_len = orig_len - 1 + num_image_tokens
                        exp_labels = torch.full(
                            (exp_len,), -100, dtype=torch.long, device=device
                        )
                        # [0, first_img_pos): 对应原序列 shift target
                        exp_labels[:first_img_pos] = curr_labels[1 : first_img_pos + 1]
                        # [first_img_pos + num_image_tokens, exp_len): 原序列后半 target
                        tail_len = orig_len - 1 - first_img_pos
                        if tail_len > 0:
                            exp_labels[first_img_pos + num_image_tokens : first_img_pos + num_image_tokens + tail_len] = curr_labels[first_img_pos + 1 : orig_len]
                    else:
                        # 无 <image>：图像在句首，扩展长度 = num_image_tokens + orig_len - 1
                        exp_len = num_image_tokens + orig_len - 1
                        exp_labels = torch.full(
                            (exp_len,), -100, dtype=torch.long, device=device
                        )
                        exp_labels[num_image_tokens:] = curr_labels[1:orig_len]
                    expanded_labels_list.append(exp_labels)
                # Pad 到 max_len
                max_exp = max(el.size(0) for el in expanded_labels_list)
                padded_exp_labels = []
                for el in expanded_labels_list:
                    if el.size(0) < max_exp:
                        el = torch.cat([
                            el,
                            torch.full((max_exp - el.size(0),), -100, dtype=el.dtype, device=device)
                        ], dim=0)
                    padded_exp_labels.append(el)
                expanded_labels = torch.stack(padded_exp_labels, dim=0)  # [B, max_len]
                # shift: 预测下一 token
                shift_logits = lm_logits[..., :-1, :].contiguous()
                shift_labels = expanded_labels[..., 1:].contiguous()
                shift_logits_flat = shift_logits.view(-1, shift_logits.size(-1))
                shift_labels_flat = shift_labels.view(-1)
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(shift_logits_flat, shift_labels_flat)
                result["loss"] = loss
        
        elif task_type == "robot":
            # Robot 价值预测：使用 Value Head
            # 取最后一个 token 的 hidden state
            last_hidden_state = lm_outputs.hidden_states[-1]  # [B, seq_len, 640]
            
            # 找到每个样本中最后一个非 padding token 的位置
            if attention_mask is not None:
                # attention_mask: [B, seq_len]，1 表示有效 token
                sequence_lengths = attention_mask.sum(dim=1) - 1  # 最后一个有效 token 的索引
                sequence_lengths = sequence_lengths.clamp(min=0, max=last_hidden_state.size(1) - 1)
            else:
                sequence_lengths = torch.full(
                    (batch_size,), 
                    last_hidden_state.size(1) - 1, 
                    dtype=torch.long, 
                    device=device
                )
            
            # 取出每个样本的最后一个有效 token
            batch_indices = torch.arange(batch_size, device=device)
            final_hidden = last_hidden_state[batch_indices, sequence_lengths]  # [B, 640]
            
            # 通过 Value Head 得到价值 logits
            value_logits = self.value_head(final_hidden)  # [B, 201]
            result["logits"] = value_logits
            result["task_type"] = "robot"
            
            # 计算 Value Classification Loss
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(value_logits, labels)
                result["loss"] = loss
        
        else:
            raise ValueError(f"不支持的任务类型: {task_type}")
        
        return result
    
    def save_projector(self, save_path: str):
        """保存 Projector 权重（阶段 1 结束后）"""
        torch.save(self.projector.state_dict(), save_path)
        print(f"Projector 权重已保存到: {save_path}")
    
    def load_projector(self, load_path: str):
        """加载 Projector 权重（阶段 2 开始前）"""
        state_dict = torch.load(load_path, map_location="cpu", weights_only=True)
        self.projector.load_state_dict(state_dict)
        print(f"Projector 权重已加载: {load_path}")
    
    def save_value_head(self, save_path: str):
        """保存 Value Head 权重"""
        torch.save(self.value_head.state_dict(), save_path)
        print(f"Value Head 权重已保存到: {save_path}")
    
    def load_value_head(self, load_path: str):
        """加载 Value Head 权重"""
        state_dict = torch.load(load_path, map_location="cpu", weights_only=True)
        self.value_head.load_state_dict(state_dict)
        print(f"Value Head 权重已加载: {load_path}")


def create_stitched_model(
    vision_model: str = "google/siglip-so400m-patch14-384",
    language_model: str = "google/gemma-3-270m-it",
    num_value_bins: int = 201,
    device: str = "cuda",
    load_in_4bit: bool = False,
    cache_dir: Optional[str] = None
) -> Pi06StitchedVLM:
    """
    便捷函数：创建缝合 VLM
    
    Args:
        vision_model: SigLIP 模型名称
        language_model: Gemma 3 模型名称
        num_value_bins: 价值 bins 数量
        device: 设备
        load_in_4bit: 是否 4-bit 量化语言模型
        cache_dir: 缓存目录
    
    Returns:
        缝合好的 VLM 模型
    """
    model = Pi06StitchedVLM(
        vision_model_name=vision_model,
        language_model_name=language_model,
        num_value_bins=num_value_bins,
        cache_dir=cache_dir,
        device=device,
        load_in_4bit=load_in_4bit
    )
    
    if not load_in_4bit:
        model = model.to(device)
    
    return model


if __name__ == "__main__":
    """测试缝合模型的初始化"""
    print("="*80)
    print("测试缝合 VLM 初始化")
    print("="*80)
    
    # 创建模型（使用 4-bit 量化以节省显存）
    model = create_stitched_model(
        device="cuda" if torch.cuda.is_available() else "cpu",
        load_in_4bit=True
    )
    
    print("\n模型初始化成功！")
    print(f"Vision Tower 冻结: {not next(model.vision_tower.parameters()).requires_grad}")
    print(f"Projector 可训练: {next(model.projector.parameters()).requires_grad}")
    print(f"Language Model 冻结: {not next(model.language_model.parameters()).requires_grad}")
    print(f"Value Head 可训练: {next(model.value_head.parameters()).requires_grad}")
