---
chapter: ch-55
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/kl-control-rlhf.md
source_url: https://arxiv.org/abs/2203.02155
created_at: "2026-04-23"
updated_at: "2026-09-17"
---

# Excerpt: KL control in RLHF — the reference term and its two placements

**Canonical extract:** `wiki/raw-data/llm-training/papers/kl-control-rlhf.md`. This file was rewritten in the 2026-09 revision; its earlier version asserted that verl places KL in the reward and not in the loss, which is wrong.

---

## The objective

RLHF optimizes a KL-regularized objective rather than the reward alone: the policy maximizes `E[r(x, y)] − β·KL(π ‖ π_ref)`, where `π_ref` is the SFT checkpoint and β sets how far the policy may move from it. The closed-form optimum of that objective is `π*(y|x) ∝ π_ref(y|x)·exp(r(x, y)/β)`, so β is the inverse temperature of a reward-tilted version of the reference policy rather than a tuning constant with no interpretation.

## Two placements, and what verl does

- **In the reward:** the per-token reward becomes `r − β·KL_estimator`, and the advantage estimator sees the adjusted reward. verl implements this as `algorithm.use_kl_in_reward`, default `False`, estimator `kl` (k1), controller `fixed` with `kl_coef: 0.001` (ppo_trainer.yaml L98–101).
- **In the loss:** a KL term is added to the policy loss. verl implements this as `actor.use_kl_loss`, default `false`, with `kl_loss_coef: 0.001` and `kl_loss_type: low_var_kl` (k3) (actor.yaml L103, L113, L116). The verl GRPO documentation recommends this placement and says to set it True for GRPO (grpo.md L43–47).

Both default to off. A verl run that does not set one of them has no reference term at all.

## Estimators

k1 = δ, k2 = δ²/2, k3 = exp(−δ) − 1 + δ, with δ = log π_θ − log π_ref for a sampled token. k1 is unbiased for the KL but takes negative values per token; k3 is non-negative per token, so its curve reads directly as a drift magnitude. verl exposes k1, `abs`, k2, k3 and `+` variants that keep the value of one estimator with the gradient of k2.

## Why the term matters for breadth

The reward is defined on the trained domains; the reference policy is the only term that refers to behaviour anywhere else. [[prorl]] keeps the KL term for exactly this reason and additionally hard-resets the reference policy and optimizer when validation stagnates, on the argument that the KL term "may increasingly dominate the loss" over a long run (§2.3.1, §3.3). That is one training history, not an ablation.

## Corrections to the previous excerpt version

1. "verl subtracts β·KL from `token_level_rewards` before the advantage runs; it intentionally does not put KL in the loss" → verl supports both placements; the GRPO docs recommend the loss placement.
2. "verl defaults to β ≈ 0.04 — the DeepSeekMath value" → verl's defaults are `kl_loss_coef: 0.001` and `kl_coef: 0.001`. β = 0.04 is DeepSeekMath's setting.
3. "adding KL to the loss breaks the advantage-based policy gradient and empirically trains worse" — removed; no source supports it, and verl's own GRPO documentation recommends the loss form.
4. "k3 … recommended, used in modern TRL / OpenRLHF / verl" → TRL GRPO's β defaults to 0.0 (no reference model loaded); OpenRLHF defaults to k1 in the reward; verl defaults to both placements off.

## Connections

- [[entropy-logging-patterns]] — the per-framework defaults table.
- [[verl-ppo-loss]] — where the two placements are applied in code.
- [[prorl]] — a long multi-domain run that keeps the term and resets the reference.
