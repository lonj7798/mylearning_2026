---
chapter: ch-02
course: llm-training
phase: read
excerpt_of: "Micikevicius, Stosic, Judd, Kamalu, Oberman, Shoeybi, et al. (NVIDIA, Arm, Intel) — FP8 Formats for Deep Learning"
source_url: https://arxiv.org/abs/2209.05433
created_at: "2026-09-15"
revised: "2026-09-15 (created from arXiv:2209.05433v2; generality revision)"
---

# Excerpt: FP8 Formats for Deep Learning (arXiv 2022-09)

Created on 2026-09-15 from arXiv:2209.05433v2 (2022-09-29). No library card exists for this paper; this chapter
excerpt holds the values ch-02 uses.

## Binary formats (§3, Table 1)
| Field | E4M3 | E5M2 |
|---|---|---|
| Exponent bias | 7 | 15 |
| Infinities | none | S.11111.00 |
| NaN | S.1111.111 | S.11111.{01, 10, 11} |
| Max normal | S.1111.110 = 1.75 × 2^8 = 448 | S.11110.11 = 1.75 × 2^15 = 57,344 |
| Min normal | S.0001.000 = 2^-6 | S.00001.00 = 2^-14 |
| Max subnormal | S.0000.111 = 0.875 × 2^-6 | S.00000.11 = 0.75 × 2^-14 |
| Min subnormal | S.0000.001 = 2^-9 | S.00000.01 = 2^-16 |

- E5M2 follows IEEE 754 conventions and "can be viewed as IEEE half precision with fewer mantissa bits" (§3).
- E4M3 reclaims most special-value bit patterns: no infinities and one NaN mantissa pattern. This adds one
  binade (17 → 18) and the magnitudes 256-448; without it the maximum would be 240 (§3.1).
- E5M2 provides 32 binades including subnormals (§3.1).
- Recommended use: E4M3 for weight and activation tensors, E5M2 for gradient tensors; some networks train with
  one type, others need both (§3).

## Scaling (§2, §3.2, §4.3)
- Higher-precision values are multiplied by a scaling factor before casting so that the tensor's maximum
  magnitude is close to the format's maximum; values that overflow are saturated (§2).
- Skipping weight updates on overflow and reducing the scale, as in FP16 automatic mixed precision, is "not a
  good choice for FP8" because the narrower range makes overflows more likely (§2).
- Per-tensor scaling in software is preferred over a programmable exponent bias, because a scale can be any
  real value while a bias is a power of 2 (§3.2).
- Mathematical operations on FP8 inputs are expected to produce higher-precision outputs (§2).
- Post-training E4M3 inference of a bfloat16-trained GPT-3 1.3B: casting GEMM inputs and residuals with a
  single exponent bias gives perplexity 12.59 or worse versus 10.19 for bfloat16; per-tensor calibrated
  scales give 10.29 (GEMM inputs) and 10.44 (GEMM inputs and residuals) (§4.3, Fig. 2).

## Training results (§4.1)
- Training used simulated FP8: tensors were clipped to FP8-representable values (with scaling and saturation)
  and arithmetic ran in FP16 or bfloat16 (§4). Inputs to GEMMs (activations, weights, activation gradients)
  were quantized; outputs stayed in higher precision (§4.1).
- Hyperparameters were unchanged from the 16-bit baselines (§4.1).

| Model | 16-bit baseline | FP8 | Metric | Locus |
|---|---|---|---|---|
| GPT 126M | 19.14 | 19.24 | training perplexity | Table 4 |
| GPT 1.3B | 10.62 | 10.66 | training perplexity | Table 4 |
| GPT 22B | 7.21 | 7.24 | training perplexity | Table 4 |
| GPT 175B | 6.65 | 6.68 | training perplexity at 75% of training (bfloat16 baseline not finished) | Table 4, §4.1 |
| Transformer Large (WMT16 En→De) | 28.43 | 28.35 | BLEU | Table 3 |
| MobileNet v2 (ILSVRC12) | 71.65 | 71.04 | top-1 | Table 2 |

- The authors state that FP8 results are within run-to-run variation of 16-bit training, except MobileNet v2
  (§4.1). Language models are evaluated by perplexity only; no downstream task evaluation of the GPT models is
  reported.

## Used in ch-02
- §1 (E4M3 and E5M2 limits), §4 (scaling rule, recommended format split, what the parity evidence measures).
