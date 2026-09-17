---
chapter: ch-02
course: llm-training
phase: read
excerpt_of: "DeepSeek-AI — DeepSeek-V3 Technical Report (arXiv:2412.19437), §3.3 FP8 Training and Appendix B"
source_url: https://arxiv.org/abs/2412.19437
created_at: "2026-04-23"
revised: "2026-09-15 (rewritten from arXiv:2412.19437v2 §3.3 and App. B; generality revision)"
---

# Excerpt: DeepSeek-V3 FP8 training (arXiv:2412.19437v2 §3.3, App. B)

Rewritten on 2026-09-15 from the primary text (v2, 2025-02-18). Model-level facts and the recipe ledger are in
the verified cards [[deepseek-v3]] and [[deepseek-v3-recipe]]. The previous version of this excerpt stated a
reversed scaling layout, an E4M3/E5M2 split, delayed scaling with a 16-step history, and FP32 optimizer
moments; the report states the opposite on each point (see below).

## Framework (§3.3, §3.3.1, Figure 6)
- Model: 671B total, 37B activated parameters; pre-trained on 14.8T tokens ([[deepseek-v3-recipe]]).
- The three GEMMs of each Linear operator run in FP8: Fprop (forward), Dgrad (activation gradient), Wgrad
  (weight gradient). They take FP8 inputs and produce BF16 or FP32 outputs (§3.3.1).
- The report states this design "theoretically doubles the computational speed compared with the original
  BF16 method" (§3.3.1). A measured end-to-end speedup is not reported.
- Kept in original precision (BF16 or FP32): the embedding module, the output head, MoE gating modules,
  normalization operators, and attention operators (§3.3.1).
- Master weights, weight gradients, and optimizer states are stored in higher precision (§3.3.1).

## Fine-grained quantization (§3.3.2, Figure 7a)
- Standard practice scales a tensor so its maximum absolute value maps to the FP8 maximum, which makes
  training sensitive to activation outliers (§3.3.2).
- Activations: scaled per 1×128 tile ("per token per 128 channels"). Weights: scaled per 128×128 block
  ("per 128 input channels per 128 output channels") (§3.3.2).
- Scales are computed online from the current maximum absolute value of each tile or block; the report
  contrasts this with delayed quantization, which infers the scale from a history of prior iterations (§3.3.2,
  "Online Quantization").
- Format: E4M3 on all tensors, in contrast to the hybrid E4M3 (Fprop) / E5M2 (Dgrad, Wgrad) scheme of prior
  work. The authors attribute the feasibility of E4M3 everywhere to tile- and block-wise scaling (§3.3.2,
  "Mantissa over Exponents").

## Accumulation precision (§3.3.2, Figure 7b)
- FP8 GEMM accumulation on NVIDIA H800 retains "around 14 bits". For GEMM of two random matrices with inner
  dimension K = 4096, the preliminary test gave a maximum relative error of nearly 2% (§3.3.2).
- Fix: partial results are copied to FP32 registers on CUDA cores every N_C elements; N_C = 128 (4 WGMMAs) is
  described as the smallest interval that improves precision without substantial overhead (§3.3.2).

## Low-precision storage and communication (§3.3.3)
- AdamW first and second moments are tracked in BF16 "without incurring observable performance degradation";
  master weights and gradients "(used for batch size accumulation)" stay in FP32 (§3.3.3).
- Inputs of the Linear after the attention operator are cached in a custom E5M6 format with power-of-2 scaling
  factors, because the attention backward pass also uses them (§3.3.3).
- Activations before MoE up-projections are quantized to FP8 for dispatch; forward and backward combine
  components stay in BF16 (§3.3.3).

## What the BF16 comparison measures (§3.3, App. B)
| Ablation | Setting | Reported result | Locus |
|---|---|---|---|
| FP8 vs BF16 | MoE ≈16B total params, 1.33T tokens | relative loss error below 0.25% | App. B.1, Fig. 10 |
| FP8 vs BF16 | MoE ≈230B total params, ≈0.9T tokens | relative loss error below 0.25% | App. B.1, Fig. 10 |
| Block-wise (128×128) quantization of all Dgrad tensors | MoE ≈16B, ≈300B tokens | model divergence | App. B.2 |

- The comparison is reported as training-loss curves smoothed by EMA with coefficient 0.9 (Fig. 10). No
  downstream benchmark comparison between the FP8 and BF16 runs is reported in §3.3 or App. B.
- The authors describe 0.25% as "well within the acceptable range of training randomness" (§3.3); no seed
  variance measurement is given.
- App. B.2 hypothesis: activation gradients are imbalanced among tokens, producing token-correlated outliers
  that block-wise grouping cannot handle.

## Used in ch-02
- §2 (components kept in higher precision), §4 (scaling granularity, formats, accumulation, validation scope),
  Recipe rows.
