# AFML Kaggle Hackathon — Weight Denoising & Machine Translation

We first denoise a corrupted Transformer weight vector, then use the recovered weights for a low-resource machine translation task.

## Structure

```text
README.md
26-part1.ipynb
26-part2.ipynb
```

The notebooks contain the complete implementation and experiments. Competition datasets and weight files are not included.

---

## Part 1 — Neural Network Weight Denoising

We were given noisy and clean flattened Transformer weights and had to learn a mapping from the noisy vector to the clean one.

### Approach

We first looked at local structure in the corruption. The noisy weights showed useful correlation at small offsets, including **1 and 128**, with 128 matching the Transformer row width. This suggested that neighbouring weights contained information about the noise.

We started with local **Ridge regression** using `side × side` neighbourhoods. A `21×21` neighbourhood worked better than smaller ones, and a scale-aware version gave another improvement.

| Method | Validation RMSE |
|---|---:|
| No denoising | 0.62051 |
| 1×1 Ridge | 0.48205 |
| 13×13 Ridge | 0.07821 |
| 21×21 Ridge | 0.07495 |
| Scale-aware 21×21 Ridge | **0.06900** |

The Ridge prediction still had structured residual error, so we trained **DnCNN-style residual CNNs** to learn the remaining correction. A RED-Net variant was also tested, but DnCNN performed slightly better.

A small ensemble of two independently trained DnCNN corrections was finally averaged, giving a validation RMSE of **0.04903**.

### Final Part 1 pipeline

```text
Noisy weights
    ↓
Scale-aware 21×21 Ridge
    ↓
2 × DnCNN residual correction
    ↓
Average predictions
    ↓
Denoised Transformer weights
```

We also tested a Gaussian/MMSE-style compensation as a sanity check, but it did not improve the final DnCNN result.

---

## Part 2 — Low-Resource Machine Translation

The denoised weights were loaded into the **fixed Transformer Encoder–Decoder** provided for the hackathon.

The architecture was kept unchanged:

- `d_model = 128`
- `4` attention heads
- `2` encoder layers
- `2` decoder layers
- `FFN = 256`
- `2,961,630` parameters

We also corrected the starter tokenizer's special-token IDs to match the supplied vocabulary:

```text
<pad> = 0    <bos> = 1    <eos> = 2    <unk> = 3
```

### Why the decoding strategy mattered

The 300 training sentence pairs showed a strong **position-preserving token mapping** between the mystery language and English. We used this structure alongside the Transformer rather than relying only on autoregressive generation.

We evaluated a sequence of methods on a **240 / 60 train-validation split**:

1. **Dictionary baseline** — direct token mapping.
2. **Restored Transformer** — denoised weights before fine-tuning.
3. **Fine-tuned Transformer** — fine-tuned on the 240 training examples.
4. **Length correction** — constrain output length based on source length.
5. **Post-hoc hybrid** — combine dictionary and Transformer output after generation.
6. **Position-constrained hybrid** — use the dictionary at reliable source positions and the Transformer for uncertain ones.

### Validation results

| Method | Token Acc. | Exact Match | BLEU-4 | ChrF++ |
|---|---:|---:|---:|---:|
| Dictionary | 78.44% | 6.67% | 30.89 | 56.57 |
| Restored Transformer | 31.44% | 3.33% | 26.85 | 54.97 |
| Fine-tuned Transformer | 63.37% | 31.67% | 52.71 | 72.75 |
| Length-corrected | 63.37% | 40.00% | 51.78 | 71.67 |
| Post-hoc hybrid | 88.82% | 45.00% | 57.82 | 74.27 |
| **Position-constrained hybrid** | **92.54%** | **53.33%** | **64.67** | **80.10** |

The final approach improved because it preserved reliable token mappings while avoiding the alignment and repetition problems seen in free-form generation.

## Final Setup

```text
Part 1:
Scale-aware Ridge → DnCNN ensemble

Part 2:
Denoised weights → 40-epoch fine-tuning (5e-5)
→ Position-constrained hybrid decoding
→ submission_part2.csv
```

The final notebook produces the required **200 translations** in `id,translation` format.


