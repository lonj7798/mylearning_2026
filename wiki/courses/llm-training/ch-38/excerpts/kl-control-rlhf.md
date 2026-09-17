---
chapter: ch-38
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/kl-control-rlhf.md
source_url: https://arxiv.org/abs/2205.11275
revised_at: "2026-09-15"
---

# Excerpt: KL-regularized RL as Bayesian inference (Korbak et al. 2022)

Checked against arXiv:2205.11275v2 (Korbak, Perez, Buckley) on 2026-09-15. Used in read.md Core insight and §4. The library card's recommendation of the k3 estimator for the reward-side penalty is not supported by InstructGPT Eq. 2 or by TRL's default and is not used.

## Objective and reward form (§3, Eq. 3-4)
- J_KL-RL(θ) = E_{x∼π_θ}[r(x)] − β D_KL(π_θ, π_0).
- Equivalent reward: r'_θ(x) = r(x) + β(log π_0(x) − log π_θ(x)), the sampled log-ratio for a full sequence x.

## Target distribution (Eq. 5-7)
- π*_KL-RL(x) = (1/Z) π_0(x) exp(r(x)/β), "where π_0 is the prior, exp(r(x)/β) is the evidence provided by the reward function (scaled by temperature β)".
- π*_KL-RL = argmax_θ J_KL-RL(θ), and J_KL-RL(θ) ∝ −D_KL(π_θ, π*_KL-RL): maximizing the objective is variational inference toward π*.
