---
chapter: ch-03
course: llm-training
phase: read
excerpt_of: primary source arXiv:2401.02954v1 (no library card as of 2026-09-15)
source_url: https://arxiv.org/abs/2401.02954
created_at: "2026-09-15"
---

# Excerpt: DeepSeek LLM: Scaling Open-Source Language Models with Longtermism

**Report:** DeepSeek-AI (Xiao Bi, Deli Chen, Guanting Chen, Shanhuang Chen, Damai Dai, Chengqi Deng, et al.).
arXiv v1 2024-01-05 read. Source type: official technical report. ch-03 uses §2.2–§3.1. The scaling-law
allocation results (§3.2–§3.3) are covered in the ch-08a excerpt of the same report.

## Pre-training settings (§2.2–§2.3, Table 2)

| Model | Layers | d_model | Heads / KV heads | Context | Batch (sequences) | Peak LR | Tokens |
|---|---|---|---|---|---|---|---|
| DeepSeek LLM 7B | 30 | 4096 | 32 / 32 | 4096 | 2304 | 4.2e-4 | 2.0T |
| DeepSeek LLM 67B | 95 | 8192 | 64 / 8 | 4096 | 4608 | 3.2e-4 | 2.0T |

- "DeepSeek LLM is initialized with a standard deviation of 0.006 and trained using the AdamW optimizer ... β1 = 0.9,
  β2 = 0.95, and weight_decay = 0.1" (§2.3). Pre-Norm with RMSNorm (§2.2). Gradient clipping 1.0 (§2.3).
- Schedule: maximum LR after 2000 warmup steps, "decreases to 31.6% of the maximum value after processing 80% of the
  training tokens. It further reduces to 10% of the maximum value after 90% of the tokens" (§2.3).
- For a 1.6B model on 100B tokens, final performance of the multi-step schedule is "essentially consistent" with
  cosine (Fig. 1a); other stage proportions are "slightly better" (Fig. 1b), but 80/10/10 was kept to balance reuse
  in continual training with performance (§2.3).

## Scaling laws for hyperparameters (§3.1, Eq. 1)

- Grid at 1e17 FLOPs (177M FLOPs/token model): generalization error "remains stable across a wide range of choices of
  batch sizes and learning rates" (Fig. 2a).
- Multi-step runs reuse the first stage across budgets 1e17 to 2e19; "near-optimal" = generalization error within
  0.25% of the minimum (§3.1).
- Fit: `η_opt = 0.3118 · C^−0.1250`, `B_opt = 0.2920 · C^0.3271` (Eq. 1). Validated at 1e20 FLOPs with a 2.94B
  FLOPs/token model (Fig. 2b).
- Limits stated: factors beyond compute C were not modeled, which the authors note is "inconsistent with some earlier
  works (Kaplan et al., 2020; McCandlish et al., 2018)" that relate optimal batch size to loss; at a fixed budget the
  optimal space "varies slightly" with the model/data split (§3.1).
- Model scale M = 72·n_layer·d_model² + 12·n_layer·d_model·l_seq (non-embedding FLOPs per token) and C = M·D (§3.2,
  Eq. 2).

## Derived check (not printed in the report)

The units of C and B in Eq. 1 are not printed. For the 7B model, M = 72·30·4096² + 12·30·4096·4096 = 4.23e10 FLOPs per
token and D = 2.0e12 tokens, so C = 8.46e22. Eq. 1 then gives η = 4.25e-4 and B = 9.22e6; Table 2 prints 4.2e-4 and
2304 × 4096 = 9.44e6 tokens. For the 67B model, M = 4.97e11, C = 9.95e23, η = 3.12e-4, B = 2.07e7; Table 2 prints
3.2e-4 and 4608 × 4096 = 1.89e7 tokens. The match supports reading C in FLOPs and B in tokens.

## Verification
- Read on 2026-09-15 against arXiv:2401.02954v1 PDF text (§2.2–§3.2).
