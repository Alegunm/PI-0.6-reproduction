# Value Function for Advantage-Conditioned Policy (π-style Reproduction)

A lightweight **Value Function (VF)** implementation for advantage-conditioned robot policy learning, following the methodology of **Pi** (e.g. π₀.6): train a VF to score trajectories, then use high-advantage (positive) data only for policy fine-tuning.

This repo focuses on the **VF part**: a small multimodal model that predicts a **201-bin discrete value distribution** (time-to-go style) from robot-view images and task instructions. It is designed to work with **LeRobot-style** data and can be used as the critic in a RECAP / Advantage Conditioning pipeline.

---

## Overview

- **Input**: Robot camera image(s) + task text (e.g. "plug the cable").
- **Output**: A distribution over 201 value bins (normalized time-to-go in `[-1, 0]`).
- **Usage**: After training, the VF scores offline trajectories; trajectories above a threshold are labeled "Advantage: Positive" and used for policy SFT / offline RL.

**Design choices (reproduction-relevant):**

- **Labels**: Value labels are computed with a **Negative Time-to-Go (NTTG)** rule: successful episodes get values from 0 at success backward (e.g. -1, -2, …); failed episodes get a constant low value. Labels are normalized to `[-1, 0]` and discretized into 201 bins.
- **Loss**: Cross-entropy over the 201 bins (distributional value, not regression).
- **Co-training**: To reduce overfitting on small robot data, the VF is optionally co-trained on a small amount of **VQA / image-question data** (e.g. LLaVA-style or VQAv2), so the model keeps both a value head and a language head.

---

## Architecture (Stitched VF)

We use a **stitched** small-VLM setup (aligned with the paper’s Gemma 270M + SigLIP style):

| Component       | Model                               | Role                | Params  |
|----------------|--------------------------------------|---------------------|--------|
| Vision encoder | SigLIP-SO400M (patch14-384)          | Image features      | ~400M  |
| Projector      | MLP 1152→1152→640                    | Vision–language map | ~1.5M  |
| Language model | Gemma-3-270M-it                      | Causal LM + hidden  | ~270M  |
| Value head     | Linear(640 → 201)                    | Bin distribution    | ~130K  |

**Total ~670M parameters** (much smaller than a 4B VLA-based VF).

- **Stage 1 (alignment)**: Train only the Projector on VQA/image-caption data (e.g. LLaVA-Instruct or The Cauldron VQAv2); SigLIP and Gemma are frozen.
- **Stage 2 (robot VF)**: Load the Stage-1 projector; train Projector + Value head + optional **LoRA** on Gemma, on robot data with 201-bin labels. Data split: **8 : 1 : 1** (train / val / test) by episode.

See figures in the repo:

- `architecture_current_stitched.png` — Stitched SigLIP + Projector + Gemma 270M + Value head.
- `architecture_prev_4b_vla.png` — Previous 4B VLA-based VF (for comparison).

---

## Repository structure (what you can show)

```
value_function_stitched/
├── README.md                    # This file
├── FILE_MANIFEST.md             # What to publish / keep private
├── model_stitched.py            # Stitched VF model (SigLIP + Projector + Gemma + Value head)
├── dataset_alignment.py        # Stage 1: VQA / image-caption dataloader
├── dataset_robot.py            # Stage 2: Robot parquet + video dataloader (LeRobot-style)
├── train_alignment.py          # Stage 1 training (projector only)
├── train_robot_vf.py           # Stage 2 training (projector + value head + optional LoRA)
├── train.py                    # Unified entry: stage 1, stage 2, or both
├── eval_test.py                # Test-set evaluation (loss, Top-1, Top-5)
├── visualize_value_function.py # Value-over-time visualization (e.g. paper-style curves)
├── draw_architectures.py       # Script to draw architecture diagrams
├── start_train_tmux.sh         # Example: run stage 2 in tmux (replace paths before use)
├── architecture_current_stitched.png
├── architecture_prev_4b_vla.png
├── value_function_visualization.png   # Example value curve (optional to commit)
├── TRAINING.md                 # Training guide (8:1:1 split, tmux, GPU tips)
└── RESULTS.md                  # Validation/test metrics (no internal paths)
```

**Not included in the repo (do not commit):**

- Robot or real-world **datasets** (videos, parquet, raw logs). Only *scripts* and *data loaders* are public; data paths in docs/scripts are placeholders.
- Large **checkpoints** (e.g. `outputs/`, `*.pt`, `*.safetensors`). You can document the layout and add a small example if needed.
- **Logs** that contain internal paths or sensitive info.

See **FILE_MANIFEST.md** for a precise list of what to publish vs keep private.

---

## Data (placeholder; your data stays local)

Training and evaluation expect **LeRobot-style** data:

- **Parquet**: Episodes with frame indices, value labels (raw float from NTTG), task text, video keys.
- **Videos**: Frames referenced by the parquet (paths are configured via `--parquet_dir` and `--video_root`).

Value labels are assumed to be precomputed (e.g. by a separate `process_data` pipeline) with NTTG and optionally normalized to `[-1, 0]`; the dataloader maps them to 201-bin indices. **This repo does not include any real robot or real-world data**; only the data *format* and *loading code* are described.

---

## Quick start (with your own data)

**Environment:** Python 3.10+, PyTorch, Hugging Face `transformers`, `datasets`, and dependencies for SigLIP/Gemma (see your local env). Install and use a venv/conda as usual.

**1. Stage 1 — Alignment (VQA)**

```bash
python train.py --stage 1 \
  --output_base ./outputs
# Or: python train_alignment.py --dataset_name HuggingFaceM4/the_cauldron --cauldron_subset vqav2 ...
```

**2. Stage 2 — Robot value function**

Use placeholder paths; replace with your own parquet and video root:

```bash
python train.py --stage 2 \
  --projector_path ./outputs_alignment/best_projector.pt \
  --parquet_dir /path/to/your/lerobot_parquet/chunk-000 \
  --video_root /path/to/your/videos \
  --output_base ./outputs \
  --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1 --split_seed 42
```

**3. Run in tmux (recommended)**

Edit `start_train_tmux.sh`: set `CUDA_VISIBLE_DEVICES`, `parquet_dir`, `video_root`, then:

```bash
chmod +x start_train_tmux.sh && ./start_train_tmux.sh
```

**4. Test-set evaluation**

```bash
python eval_test.py \
  --checkpoint ./outputs/robot_vf \
  --parquet_dir /path/to/your/lerobot_parquet/chunk-000 \
  --video_root /path/to/your/videos \
  --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1 --split_seed 42
```

Use the same split seed as training so the test set is consistent. For accurate metrics, evaluate with `frame_skip=1` (no frame skipping) on the test set.

---

## Results (example)

On an internal robot dataset (not included), with 8:1:1 split and best checkpoint selected by **validation loss**:

- **Test loss**: ~4.07 (cross-entropy over 201 bins).
- **Test Top-1**: ~4.1%; **Test Top-5**: ~19.2%.

Random guess baseline: Top-1 ≈ 1/201 ≈ 0.5%, so the stitched VF clearly learns above chance. For more detail and epoch-by-epoch validation curves, see **RESULTS.md** (no internal paths or raw data).

---

## References and acknowledgments

- **Pi (π)** and **RECAP / Advantage Conditioning**: value functions for filtering high-advantage data and conditioning the policy.
- **OpenVLA / Pi0**: open-weight VLA policy used in the full pipeline (this repo only implements the VF).
- **SigLIP**, **Gemma**: vision and language models (Hugging Face).

This project is a **reproduction** of the value-function part of the pipeline; the full system (policy + simulation/real robot) is not included here. Collaboration with lab members; **robot and real-world data are not distributed with this repository.**

---

## License

Use and license as appropriate for your institution (e.g. research-only, non-commercial). Check model licenses for SigLIP and Gemma when redistributing weights.
