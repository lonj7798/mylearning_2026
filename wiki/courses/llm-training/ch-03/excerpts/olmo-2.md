---
chapter: ch-03
course: llm-training
phase: read
excerpt_of: OLMo 2 (arXiv:2501.00656) §2.1, §2.3, §3, §4.1–§4.3, Tables 1, 3, 8, 9, 11; released configs configs/official-1124 (allenai/OLMo)
source_url: https://arxiv.org/abs/2501.00656
created_at: "2026-09-15"
note: "The library card model-reports/olmo-2.md was not yet re-verified on 2026-09-15 and states that reordered norm is 'post-norm within residual' and that context was 'extended to 32K in cooldown'. Both are contradicted below (§2.1, §3.3.2, Table 3: sequence length 4096 for all sizes)."
---

# Excerpt: OLMo 2 — stability changes, schedule, and mid-training evidence

Source type: official technical report (Allen Institute for AI) plus released training configs. PDF text read on
2026-09-15; configs from `configs/official-1124/` in github.com/allenai/OLMo.

## Architecture changes for stability (§2.1, Table 1, §3)
- Layer norm "applied to" outputs instead of inputs; RMSNorm instead of non-parametric LayerNorm; QK-Norm instead of
  QKV clipping at 8 (OLMo-0424); z-loss; no weight decay on embeddings; RoPE θ 5·10^5 (Table 1).
- Block (§3.3.2): `h := x + RMSNorm(Attention(x))`, `h_out := h + RMSNorm(MLP(h))` versus OLMo-0424
  `h := x + Attention(LN(x))`, `h_out := h + MLP(LN(h))`.
- Evidence (single runs, figures without seeds): repeated n-gram filter (documents with ≥ 32 repeated n-grams of 1–13
  tokens removed; loss masking of such spans in the trainer) gave "a clear mitigation—though not complete
  elimination—of gradient spikes" and "no effect on the slow growth in gradient norm" (§3.1, Fig. 3). Init N(0, 0.02)
  vs OLMo-0424 scaled init: gradient spike score 0.40 → 0.03 in a reduced-warmup test (§3.2, Fig. 4). Reordered norm
  + QK-norm together: gradient spike score 0.108 → 0.069; "In isolation, neither of these changes yield good
  results" (§3.3.2, Fig. 7). Embedding weight decay on vs off: spike scores 0.16 vs 0.092 (§3.4.2, Fig. 10). AdamW ε
  10^-5 → 10^-8 lowers and stabilizes early gradient norm (§3.4.1, Fig. 9).
- z-loss coefficient: §3.3.3 prints 10^-4; Table 1 prints 10^-5; the 7B config sets `auxiliary_loss_multiplier: 1e-5`.

## Pre-training schedule (Table 3, §2.3; configs)

| | OLMo 2 7B | OLMo 2 13B | OLMo 2 32B |
|---|---|---|---|
| Layers / d_model | 32 / 4096 | 40 / 5120 | 64 / 5120 |
| Batch (instances) × sequence length | 1024 × 4096 | 2048 × 4096 | 2048 × 4096 |
| Gradient clipping | 1.0 | 1.0 | 1.0 |
| Peak LR | 3.0·10^-4 | 9.0·10^-4 | 6.0·10^-4 |
| Warmup | 2000 steps | 2000 steps | 2000 steps |
| Cosine horizon (to 10% of peak) | 5T tokens | 5T tokens | 6.5T tokens |
| Truncation | after 4T | n/a | after 6T |
| Stage-1 tokens; total tokens (§2.3) | 3.90T; 4.05T | 5T; 5.6T | 6.06T; 6.6T |

- `OLMo2-7B-stage1.yaml`: AdamW, `betas: [0.9, 0.95]`, `weight_decay: 0.1`, `eps: 1e-8`, `decay_embeddings: false`;
  `init_std: 0.02`, `init_cutoff_factor: 3`; `t_warmup: 8388608000` tokens, `t_max: 5e12`, `alpha_f: 0.1`;
  `max_grad_norm: 1.0`; `precision: amp_bf16`.
- `OLMo2-7B-stage2-seed42.yaml`: loads `step928646`; `learning_rate: 0.000061499`; `linear_with_warmup`,
  `t_warmup: 0`, `alpha_f: 0`; `max_duration: 50e9T`; `stop_at: 11931  # round(50e9 / (1024 * 4096)) + 10`.
- `OLMo2-13B-stage1.yaml`: `learning_rate: 3.0e-4`, `t_max: 5e12`, `alpha_f: 0.1`, `global_train_batch_size: 2048`
  (conflicts with Table 3's 9.0·10^-4). `OLMo2-13B-stage2-seed1110-100B.yaml`: loads `step596057`, `learning_rate: 9e-5`.
  Derived: 596,057 × 2048 × 4096 = 5.0T tokens, where a 9e-4 cosine with `alpha_f: 0.1` reaches 9e-5; a 3e-4 peak would
  give 3e-5. §4.1: "The 13B ran with a higher peak learning rate from the start".

## Learning rate and annealing (§4.1, Table 8)
- 7B-scale LR runs at 3, 6, 9, 12, and 30 ·10^-4; 30·10^-4 showed spikes during warmup and was abandoned. "Higher
  learning rates universally perform better early on ... but eventually the lower learning rate setting overtakes
  the others"; for 3 vs 6 ·10^-4 "the cross-over point is well past 200B tokens. A shorter hyperparameter experiment
  might come to the wrong conclusion."
- After linearly decaying to 0 over 50B or 100B tokens from 300B-token checkpoints, "a higher learning rate does make
  mid-training more effective, but it does so by exactly the amount that the pretraining is worse."
- Table 8 (OLMES, 9 MC tasks, validation, cloze): 300B + 50B anneal: 62.5 (3e-4), 63.9 (6e-4), 64.1 (9e-4), 63.6
  (12e-4); 300B + 100B: 64.6, 64.5, 64.2 (6, 9, 12 e-4); 2T + 100B high-quality: 73.8 (3e-4), 73.9 (6e-4). GSM8K after
  the 2T + 100B high-quality anneal: higher LR 2.8 points better; "More study is needed".

## Mid-training evidence (§4.2–§4.3, Tables 9, 11)
- Table 11 (7B, 4T-token checkpoint, 50B-token mid-training runs; OLMES / OLMES-Gen / MMLU / GSM*): no anneal 69.6 /
  63.2 / 59.8 / 28.5; PT Mix (same pre-training data, LR annealed) 74.0 / 64.5 / 61.8 / 27.0; Web FT FW72 75.2 / 63.8 /
  63.1 / 28.5; + Math 75.7 / 69.7 / 62.3 / 52.0; + Math + Ins 75.7 / 70.2 / 63.1 / 46.5. GSM* is 200 GSM8K questions used
  as a development set.
- Table 9, pretraining → pretraining + mid-training (Avg; dev MMLU, GSM8K; held-out MMLU PRO, TQA): 7B 53.0 → 62.9;
  MMLU 59.8 → 63.7; GSM8K 24.1 → 67.5; MMLU PRO 27.4 → 31.0; TQA 74.6 → 78.0. 13B 58.9 → 68.3; MMLU 63.4 → 67.5; GSM8K
  37.3 → 75.1; MMLU PRO 31.2 → 35.1; TQA 80.3 → 81.9. 7B final = average of 3 anneals on 50B tokens; 13B and 32B = 3
  runs on 100B + 1 run on 300B, averaged.
- Source-internal inconsistency: the §4.2 text says the 7B improves "on average by 10.6 points" and the 13B by 10.3,
  while the Table 9 averages give 53.0 → 62.9 (9.9) and 58.9 → 68.3 (9.4).
- Held-out suite: "not used for model development decisions"; GSM8K "only partially held-out" (200 of 1319 examples
  used for development) (§2.5 footnote 6).
