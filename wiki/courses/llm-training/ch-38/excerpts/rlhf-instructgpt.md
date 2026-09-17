---
chapter: ch-38
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/rlhf-instructgpt.md
source_url: https://arxiv.org/abs/2203.02155
revised_at: "2026-09-15"
---

# Excerpt: InstructGPT objective, settings, and alignment tax

Checked against arXiv:2203.02155v1 on 2026-09-15. Used in read.md §3, §5-§7 and Recipe. The library card's "Canonical hyperparameters" table (PPO LR 1.41e-5, 4 epochs per rollout, rollout length ≤ 2048, adaptive controller) does not match the paper and is not used.

## Model names and objective (§3.5)
- "We call these models 'PPO.'" (per-token KL penalty from the SFT model; value function initialized from the RM)
- "We also experiment with mixing the pretraining gradients into the PPO gradients ... We call these models 'PPO-ptx.'"
- Eq. 2: objective(φ) = E_{(x,y)∼D_{π_φ^RL}}[r_θ(x,y) − β log(π_φ^RL(y|x)/π^SFT(y|x))] + γ E_{x∼D_pretrain}[log(π_φ^RL(x))].
- "For 'PPO' models, γ is set to 0. Unless otherwise specified, in this paper InstructGPT refers to the PPO-ptx models."

## RL settings (App. C.4)
- β = 0.02; 256k episodes with about 31k unique prompts; batch 512, minibatch 64, "trained on for only a single inner epoch".
- Constant learning rate with a warm-up over the first 10 iterations starting at one tenth of the peak; EMA decay 0.992; no discount in GAE; clip ratio 0.2; sampling temperature 1.
- 6B RM and 6B value function, value initialized from the RM; value LR 9e-6 (1.3B, 6B policies), 5e-6 (175B).
- PPO-ptx: 8 times more pretraining examples than RL episodes; PPO and pretraining gradients accumulated per minibatch; γ = 27.8.
- Init models (C.3): SFT for 2 epochs with 10% pretraining data; batch 32 (1.3B, 6B) and 8 (175B); LRs 5e-6, 1.04e-5, 2.45e-6.

## Ablations (App. E)
- E.6: γ ≥ 20 recovers regressions at 1.3B; "a single value of 27.8 seems to work well across model sizes"; with γ = 0, β up to 2.0 ("100 times of the default value") does not fix the regressions; 512k episodes bring DROP and SQuADv2 slightly below GPT-3.
- E.7: "Both 0 and 2 for KL reward coefficient result in poor performance. The optimal value is around 0.01 and 0.02."
- E.9: LR sweep 2.55e-6 to 2.55e-5 (1.3B, 6B); "All runs with learning rate greater than 8.05e-6 diverged, for PPO models without pretraining data mix"; 175B: 2.55e-6 and 3.74e-6.
- E.11: pretraining data ratio 4 (pretraining loss often increased), 8 (chosen), 32 (better Likert, few-fold time); batch 512 best of 64-1024; minibatch 32 slightly better than 64, 64 kept for GPU utilization.

## Table 14, 175B columns (GPT / SFT / PPO / PPO-ptx)
| Task | Setting | GPT | SFT | PPO | PPO-ptx |
|---|---|---|---|---|---|
| SQuADv2 F1 | zero-shot | 64.30 | 57.67 | 43.68 | 59.85 |
| SQuADv2 F1 | few-shot | 69.75 | 65.90 | 51.95 | 69.93 |
| DROP F1 | zero-shot | 27.53 | 15.79 | 13.08 | 15.23 |
| DROP F1 | few-shot | 35.27 | 35.85 | 27.78 | 33.34 |
| HellaSwag acc | few-shot | 0.791 | 0.741 | 0.759 | 0.820 |
| FR→EN BLEU | zero-shot | 38.92 | 36.90 | 24.16 | 34.28 |
| FR→EN BLEU | few-shot | 39.93 | 35.07 | 26.58 | 36.76 |

## Generalization (§1, §3.3, §3.4, §4.1, §4.3, §5.4)
- 175B InstructGPT preferred to 175B GPT-3 85 ± 3% and to few-shot GPT-3 71 ± 4%.
- Agreement: training labelers 72.6 ± 1.5%, held-out labelers 77.3 ± 1.3%. RM cross-validation: 69.6 ± 0.9% on held-out groups vs 72.4 ± 0.4% within training groups.
- Dataset over 96% English; non-English and code behaviors "We do not track these behaviors quantitatively"; the model "often produces an output in English even when the instruction is in another language".
- FLAN/T0: InstructGPT 73.4 ± 2% win rate vs the SFT baseline; T0 26.8 ± 2%, FLAN 29.8 ± 2%.
- §5.4: the ptx proposal "does not completely mitigate performance regressions, and may make certain undesirable behaviors more likely for some tasks".
