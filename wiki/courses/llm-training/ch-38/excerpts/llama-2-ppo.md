---
chapter: ch-38
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/model-reports/llama-2.md
source_url: https://arxiv.org/abs/2307.09288
revised_at: "2026-09-15"
---

# Excerpt: Llama 2 PPO: reward, KL per size, settings, false refusal

Checked against arXiv:2307.09288v2 on 2026-09-15; matches [[llama-2]] and [[llama-2-recipe]] (verified 2026-09-14). Used in read.md §4, §8 and Recipe.

## Schedule (§3.2.3)
- "Until RLHF (V4), we used only Rejection Sampling fine-tuning, and after that, we combined the two sequentially, applying PPO on top of the resulted Rejection Sampling checkpoint before sampling again." The published PPO comparison is RLHF-V5 with and without PPO (Figure 11).

## Reward (§3.2.3, Eq. 3-4)
- R(g|p) = R̃_c(g|p) − β D_KL(π_θ(g|p) ‖ π_0(g|p)); the penalty "is useful for training stability, and to reduce reward hacking".
- R_c(g|p) = R_s(g|p) if is_safety(p) or R_s(g|p) < 0.15, otherwise R_h(g|p); R̃_c = whiten(logit(R_c)). Threshold 0.15: precision 0.89, recall 0.55 on the Meta Safety test set.

## Settings (§3.2.3)
- "For all models, we use the AdamW optimizer ... with β1 = 0.9, β2 = 0.95, eps = 10^−5. We use a weight decay of 0.1, gradient clipping of 1.0, and a constant learning rate of 10^−6. For each PPO iteration we use a batch size of 512, a PPO clip threshold of 0.2, a mini-batch size of 64, and take one gradient step per mini-batch. For the 7B and 13B models, we set β = 0.01 (KL penalty), and for the 34B and 70B models, we set β = 0.005."
- 200-400 iterations with early stopping on held-out prompts; about 330 s per 70B iteration.

## False refusal (§4.2.3, Figure 33)
- Defined as "incorrectly refusing to answer legitimate user prompts due to irrelevant safety concerns"; measured with a refusal classifier on the helpfulness test sets and a 210-prompt borderline set.
- Measured in the safety-data scaling ablation (helpfulness data about 0.9M; safety data 0-100% of about 0.1M), not in a PPO-only comparison.
- Helpfulness set: 0.006% (1 occurrence) to 0.05% (8 occurrences). Borderline set: 15% to 27%.
