---
chapter: ch-01
course: llm-training
phase: read
excerpt_of: primary source, arXiv:2509.02046 (no library card as of 2026-09-15)
source_url: https://arxiv.org/abs/2509.02046
created_at: "2026-09-15"
---

# Excerpt: Fantastic Pretraining Optimizers and Where to Find Them

**Authors:** Kaiyue Wen, David Hall, Tengyu Ma, Percy Liang (Stanford University)
**Version read:** arXiv:2509.02046v2 (2025-09-04); v1 2025-09. Source type: paper (preprint).

## Claim (Abstract)
"the actual speedup of many proposed optimizers over well-tuned baselines is lower than claimed and decreases with model size to only 1.1× for 1.2B parameter models." "comparing intermediate checkpoints before reaching the target training budgets can be misleading, as rankings between two optimizers can flip during training due to learning rate decay."

## Setting (§1, §3)
Eleven optimizers (AdamW, NAdamW, Mars, Cautious, Lion, Adam-mini, Muon, Scion, Kron, Soap, Sophia), Llama 2 architecture at 130M, 300M, 520M, and 1.2B parameters, data mixture similar to OLMo 2, 1×, 2×, 4×, 8× the Chinchilla-optimal token count (20 tokens per non-embedding parameter). Primary metric: C4-EN validation loss; downstream accuracy tracked on ARC-E/C, BoolQ, COPA, CommonsenseQA, HellaSwag, LAMBADA, OpenBookQA, PIQA, WSC273, WinoGrande. Hyperparameters tuned per optimizer by coordinate descent, accepting a change when validation loss improves by more than 3e-3 (§3.2).

## Update rules as written in §3
- AdamW: w_{t+1} = w_t − η·m_t/(√v_t + ε) − ηλw_t.
- Lion: w_{t+1} = w_t − η·sign(β2·m_t + (1 − β2)·g_t).
- Muon: w_{t+1} = w_t − η·NS⁽⁵⁾(β2·m̃_t + (1 − β2)·g_t), applied to matrices except the token classification head and embedding; NS is the Newton-Schulz iteration.

## Findings
- Tuning one hyperparameter (LR) of the GPT-3 recipe for a 100M model gives up to a 2× speedup of AdamW itself (Fig. 1 top left).
- Hyperparameters do not transfer between optimizers: Lion's optimal weight decay is about 0.6 versus about 0.1 for AdamW (Fig. 1 top right).
- Against tuned AdamW the highest estimated speedup is 1.4× (Fig. 3). Matrix-based optimizers (Kron, Muon, Soap) give about 1.3× below 520M; at 1.2B and 8× Chinchilla, NAdamW, Muon, and Soap give about 1.1× and "no longer leads to downstream improvements" (§4.1, Fig. 4, Table 5).
- Scaling-law fit on 16 runs predicts Muon "will result in a slightly higher loss than AdamW in the 7B and 1× Chinchilla regime" (§4.1, App. B.1).
- Muon is best at low data-to-model ratios but is outperformed by Kron and Soap at 8× and 16× Chinchilla (§1, Fig. 4 right).

## Table 5 (1.2B models, 8× Chinchilla, 193B tokens), average over 10 benchmarks
AdamW 67.15; NAdamW 66.70; Muon 66.98.

## Table 36 (AdamW, 130M, 1× Chinchilla, C4-EN loss; one change from the base row at a time)
Base row: β1 0.9, β2 0.98, ε 1e-20, η 0.008, gnorm 1, batch 128, warmup 2000, λ 0.1 → 3.529.
β1 0.95 → 3.539; β1 0.98 → 3.882; β2 0.9 → 3.545; β2 0.95 → 3.535; ε 1e-10 → 3.531; η 0.004 → 3.550; η 0.016 → 3.538; η 0.032 → 7.781; gnorm 0 → 3.534; gnorm 2.0 → 3.534; batch 256 → 3.611; warmup 500 → 7.452; λ 0 → 3.545; λ 0.2 → 3.536.

## Limits
Largest model 1.2B; seeds per configuration not reported; downstream evaluation is multiple-choice and cloze-style base-model tasks.

## Verification
- Checked on 2026-09-15 against arXiv:2509.02046v2 (Abstract, §1, §3, §4.1, Fig. 1, Table 3, Table 5, Table 36).
