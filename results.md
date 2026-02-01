# Validation and test results / 验证与测试结果

**语言 / Language:** [English](#english) · [中文](#中文)

---

<a name="english"></a>

## English

- **Split**: 8:1:1 (train / val / test by episode).
- **Best checkpoint**: Selected by minimum validation loss (Epoch 3 in the run below).
- **Test metrics**: Evaluated once on the best checkpoint with the same `split_seed=42`.

### Summary

| Stage | Train loss | Val loss | Val Top-1 (%) | Test Top-1 (%) | Test Top-5 (%) |
|-------|------------|----------|---------------|----------------|----------------|
| Epoch 1 | 4.34 | 4.13 | 3.90 | — | — |
| Epoch 2 | 4.00 | 4.00 | 4.74 | — | — |
| Epoch 3 | 3.88 | **3.97** | 4.52 | — | — |
| Epoch 4 | 3.74 | 4.06 | 4.50 | — | — |
| Epoch 5 | 3.52 | 4.16 | 4.54 | — | — |
| Epoch 6 | 3.17 | 4.49 | 4.20 | — | — |
| Epoch 7 | 2.74 | 5.11 | 3.90 | — | — |
| Epoch 8 | 2.32 | 5.88 | 3.84 | — | — |
| Epoch 9 | 1.99 | 6.72 | 3.64 | — | — |
| Epoch 10 | 1.82 | 7.08 | 3.54 | — | — |
| **Test (best @ E3)** | — | **4.07** | — | **4.14** | **19.20** |

- Bold val loss: best epoch used for saving.
- Test row: Test Loss ≈ 4.07, Top-1 ≈ 4.14%, Top-5 ≈ 19.20%.

Random guess over 201 bins would give Top-1 ≈ 0.5%; the stitched VF clearly improves over this baseline.

---

<a name="中文"></a>

## 中文

- **划分**：8:1:1（训练 / 验证 / 测试，按 episode）。
- **Best 检查点**：按验证 loss 最小保存（下表对应 Epoch 3）。
- **测试指标**：在 best 检查点上用相同 `split_seed=42` 评估一次。

### 汇总表

| 阶段 | 训练 loss | 验证 loss | 验证 Top-1 (%) | 测试 Top-1 (%) | 测试 Top-5 (%) |
|------|-----------|-----------|----------------|----------------|----------------|
| Epoch 1 | 4.34 | 4.13 | 3.90 | — | — |
| Epoch 2 | 4.00 | 4.00 | 4.74 | — | — |
| Epoch 3 | 3.88 | **3.97** | 4.52 | — | — |
| Epoch 4 | 3.74 | 4.06 | 4.50 | — | — |
| Epoch 5 | 3.52 | 4.16 | 4.54 | — | — |
| Epoch 6 | 3.17 | 4.49 | 4.20 | — | — |
| Epoch 7 | 2.74 | 5.11 | 3.90 | — | — |
| Epoch 8 | 2.32 | 5.88 | 3.84 | — | — |
| Epoch 9 | 1.99 | 6.72 | 3.64 | — | — |
| Epoch 10 | 1.82 | 7.08 | 3.54 | — | — |
| **测试集 (best @ E3)** | — | **4.07** | — | **4.14** | **19.20** |

- 验证 loss 粗体：Epoch 3 最优，用于保存 best。
- 测试集一行：Test Loss ≈ 4.07，Top-1 ≈ 4.14%，Top-5 ≈ 19.20%。

201 个 bin 纯随机猜测 Top-1 ≈ 0.5%；缝合式 VF 明显优于该基线。
