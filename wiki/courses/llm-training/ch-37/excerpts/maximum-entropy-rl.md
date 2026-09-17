---
chapter: ch-37
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/maximum-entropy-rl.md (verified card, 2026-09-14)
source_url: https://arxiv.org/abs/1801.01290
created_at: "2026-04-23"
revised: "2026-09-15 (generality revision; previous version replaced)"
---

# Excerpt: Soft Actor-Critic (maximum-entropy RL)

- **Authors:** Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, Sergey Levine
- **Year:** 2018 (arXiv v1 2018-01; ICML 2018)
- **Source type:** paper
- **Used in:** ch-37 §6 (entropy term). Continuous control only; no language-model experiments.

## Items used by ch-37, with loci (from the verified card)
- **Objective (Eq. 1).** J(π) = Σ_t E_{(s_t, a_t)∼ρ_π}[r(s_t, a_t) + α · H(π(·|s_t))]. α is the temperature; α → 0 recovers the standard objective. The paper absorbs α into the reward by scaling the reward by α⁻¹ (§3.2).
- **Policy improvement target (Eq. 4).** π_new = argmin_{π′} D_KL(π′(·|s) ‖ exp(Q^{π_old}(s, ·)) / Z(s)).
- **Reward-scale sensitivity (§5.2, Fig. 3b, Ant-v1).** A small reward scale (high effective temperature) gave a near-uniform policy that did not exploit the reward; a large scale gave a near-deterministic policy that stalled in poor local minima. Reward scale was the only hyperparameter tuned per environment (App. D Table 2).
- **Automatic temperature.** Not in this paper. The α loss and the target entropy −dim(A) are in the follow-up arXiv:1812.05905 (Eq. 18, App. D Table 1).

## Course interpretation used in ch-37
The KL-regularized RLHF optimum π_ref(y|x)·exp(r/β)/Z has the same exponential form as the Eq. 4 target, with π_ref in place of a uniform base (Interpretation; not in SAC).
