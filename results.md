# Validation and test results (stitched VF)

- **Split**: 8:1:1 (train / val / test by episode).
- **Best checkpoint**: Selected by minimum validation loss (Epoch 3 in the run below).
- **Test metrics**: Evaluated once on the best checkpoint with the same `split_seed=42`.

## Summary

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
