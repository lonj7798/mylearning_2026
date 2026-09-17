---
chapter: ch-43
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/maximum-entropy-rl.md
source_url: https://arxiv.org/abs/1801.01290
primary_version: arXiv:1801.01290v2 (SAC; ICML 2018)
created_at: "2026-04-23"
revised: "2026-09-15 (generality revision; rewritten to match the card verified on 2026-09-14)"
---

# Excerpt: Soft Actor-Critic — the maximum-entropy objective, and what LLM-RL does not take from it

- Objective (Eq. 1): `J(π) = Σ_t E_{(s_t,a_t)~ρ_π}[ r(s_t,a_t) + α·H(π(·|s_t)) ]`. `α` is the temperature;
  the standard RL objective is the `α → 0` limit. The paper then absorbs `α` into the reward by scaling the
  reward by `α⁻¹` (§3.2), and reports reward scale as the only per-environment hyperparameter it tuned
  (§5.2, App. D Table 2: 5 for Hopper, Walker2d, HalfCheetah and Ant; 20 for Humanoid-v1; 10 for Humanoid
  rllab).
- Soft Bellman backup (Eqs. 2-3): `V(s_t) = E_{a_t~π}[Q(s_t,a_t) − log π(a_t|s_t)]`; policy improvement
  (Eq. 4) projects onto `exp(Q(s_t,·))/Z(s_t)`, so the improved policy is Boltzmann in `Q`.
- Off-policy: replay buffer of 10⁶ transitions, two Q-networks, a target value network (§4.2, App. D Table 1).
- Sensitivity (§5.2, Fig. 3b, Ant-v1): a small reward scale (large effective `α`) gave a near-uniform policy
  that did not exploit the reward; a large scale gave a near-deterministic policy that stalled in poor local
  minima.
- Automatic temperature tuning is not in this paper; it is in arXiv:1812.05905, and the target-entropy rule
  for continuous control is stated there.

## Why this matters for ch-43
Two properties of the language-model setting break the analogy: the action space is a vocabulary with a very
uneven per-position entropy profile rather than a fixed-dimension continuous action space
([[high-entropy-minority-tokens]] §3), and production LLM-RL runs are on-policy with no soft-Q critic and
usually with the entropy coefficient set to 0 ([[entropy-logging-patterns]]). The entropy bonus itself
predates SAC: A3C (2016) credits it to Williams & Peng (1991) ([[entropy-regularization-ppo]]).

## Removed in the 2026-09 revision
"Every entropy knob in modern LLM-RL descends from SAC", "A3C/PPO ship the bonus as the small-α limit of
SAC", and the claim that Cui et al.'s interventions are a rediscovery of SAC-v2's temperature controller.
None is supported by the cited papers.
