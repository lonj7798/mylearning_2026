---
chapter: ch-40
course: llm-training
phase: read
excerpt_of: arXiv:2504.05118v3 (VAPO: Efficient and Reliable Reinforcement Learning for Advanced Reasoning Tasks)
source_url: https://arxiv.org/abs/2504.05118
created_at: "2026-09-15"
revised: "2026-09-15 (generality revision; written from the primary source — the library has no VAPO card)"
---

# Excerpt: VAPO — a value model that beats DAPO on the same base

Used by [[read]] §1, §8, and the Recipe. ByteDance Seed (full author list in the paper's Contributions; correspondence Yu Yue). Dated 14 April 2025; text read from arXiv v3 on 2026-09-15.

## Headline (Abstract, §5.2)
Qwen2.5-32B base, no SFT data, AIME 2024 avg@32 (temperature 1.0, top-p 0.7): VAPO 60.4 within 5,000 steps; DAPO 50; DeepSeek-R1-Zero-Qwen-32B 47; vanilla PPO 5. Peak scores of 60–61 across three repeated runs; no training crashes reported.

## Three stated problems with value models on long CoT (§3)
value-model bias when learning by bootstrapping over long trajectories; a single GAE `λ` cannot suit both short and long responses (short: variance-dominated; long: bias from bootstrapping); reward sparsity from verifier-only signals.

## Components (§4–§5.1) and the cost of removing each (Table 1, AIME24 avg@32)
| Component | What it does | Score without it |
|---|---|---|
| Value-Pretraining | 50 steps of value-network warm-up initialized from a reward model before policy training | 11 |
| Decoupled GAE | critic learns returns with `λ = 1.0`; policy uses its own `λ` | 33 |
| Length-adaptive GAE | `λ_policy = 1 − 1/(α l)` with `α = 0.05` and `l` the output length (Eqs. 4–5) | 45 |
| Clip-Higher | `ε_low = 0.2`, `ε_high = 0.28` (from DAPO) | 46 |
| Token-level loss | replaces the per-sequence average (Eq. 6, from DAPO) | 53 |
| Positive-example LM loss | NLL on correct answers added with weight 0.1 (Eqs. 9–10) | 54 |
| Group-Sampling | 512 prompts × 16 samples instead of 8,192 prompts × 1, "richer contrastive signals" | 55 |
| VAPO (all) | — | 60 |

## Training details (§5.1)
Vanilla PPO baseline: AdamW; actor LR 1e-6, critic LR 2e-6 ("the critic needs to update faster"); warmup-constant schedule; batch 8,192 prompts sampled once, mini-batch 512; value network initialized from a reward model; GAE `λ = 0.95`, `γ = 1.0`; sample-level loss; clip ε 0.2. VAPO changes: the seven items above, with 512 prompts × 16 samples and mini-batch 512.

## Why it matters for ch-40
The same group-sampling idea that motivates critic-free methods is worth 5 points inside a value-based system, and a warmed-up value model on this task and scale beats the best value-free system reported on the same base model. Claims that the critic is settled are not supported by this paper.

## Limits
One task family (competition mathematics), one base model, no seeds reported for the ablation rows, and the components are removed one at a time from the full system rather than added cumulatively, so interactions are not isolated.
