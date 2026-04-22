<!-- scope: Comparative reference — RLOO vs GRPO, when to pick which, shared lineage, empirical differences
     deps: [[rloo]], [[grpo]]
     see-also: [[dr-grpo]], [[reinforce-plus-plus]], [[ppo]]
-->

# RLOO vs GRPO — Comparative Reference
- **Core Insight:** RLOO and GRPO are two instantiations of the same idea — replace PPO's learned value baseline with a peer-sample baseline — and their advantages become equivalent (up to scaling by std and epsilon clipping) as the number of samples per prompt grows.
- **Guideline:** Pick RLOO when you want minimum overhead and deterministic-style reasoning rewards (correct/incorrect); pick GRPO when you have a continuous reward (RM, PRM) where std-normalization stabilizes gradient magnitudes — but always evaluate against Dr. GRPO or REINFORCE++ before committing.
- **Authors:** (synthesized from Ahmadian 2024, Shao 2024, Liu 2025, Hu 2025, plus practitioner blog posts from HF and OpenRLHF)
- **Year:** 2024–2025
- **URL:** (see component papers: [[rloo]], [[grpo]], [[dr-grpo]], [[reinforce-plus-plus]]; and [[nathan-lambert-grpo]])
- **Relevant topics:** critic-free RL, peer-baseline policy gradient, advantage normalization, LLM RLHF

## Overview
Both algorithms drop PPO's value network and use same-prompt peer samples as a baseline. They differ on four axes: advantage normalization, ratio clipping, KL placement, and epochs per rollout.

## Side-by-side

| Axis | RLOO (Ahmadian 2024) | GRPO (DeepSeekMath 2024) |
|------|----------------------|--------------------------|
| Samples per prompt | k ∈ {2, 4} typical | G ∈ {8, 16, 32, 64} |
| Baseline | `b_i = (1/(k−1)) Σ_{j≠i} R_j` | `mean(R_1..R_G)` |
| Normalization | none (raw reward − baseline) | / std(R_1..R_G) |
| Importance ratio clip | no | yes, ε = 0.2 |
| KL placement | shaped per-token reward | inside the loss, k3 estimator |
| Epochs per rollout | 1 | 1 |
| Value network | no | no |

## Loss formulas

### RLOO gradient
`∇J ≈ (1/k) Σ_i (R_i − b_i) · ∇ log π_θ(y_i | x)`

### GRPO objective
`J = E[ (1/G) Σ_i (1/|o_i|) Σ_t min(ρ_{i,t} Â_{i,t}, clip(ρ_{i,t}, 1-ε, 1+ε) Â_{i,t}) − β KL(π_θ || π_ref) ]`
`Â_{i,t} = (R_i − mean(R))/std(R)`

## Equivalence in the limit
- RLOO's baseline `b_i` ≈ mean(R) when k is large (off by one sample).
- If we further divide by std(R) → same advantage as GRPO (pre-clip).
- PPO-clip in GRPO becomes active only when the policy drifts substantially in one epoch; since both use 1 epoch per rollout, the clip rarely binds.
- Thus RLOO ≈ GRPO without std normalization and without clip ≈ **Dr. GRPO**.

## When each wins empirically

| Scenario | Winner | Why |
|----------|--------|-----|
| Continuous RM score, small k | GRPO | std-normalization stabilizes gradient magnitude across prompts of varying reward spread |
| Verifiable 0/1 reward, large G | Dr. GRPO > GRPO ≈ RLOO | std(R) degenerates when all-right or all-wrong; std-norm hurts |
| Small batch / tight memory | RLOO, k=2 | least overhead |
| Reasoning RL with long chains | Dr. GRPO or REINFORCE++ | avoid GRPO's length inflation |
| Very large global batch, single rollout | REINFORCE++ | global advantage normalization |

## Practitioner recommendations (from HF / OpenRLHF / verl docs)

1. Start with **Dr. GRPO** as the default for reasoning RL (verifiable rewards).
2. Use **RLOO k=2–4** for plain RLHF with a scalar RM and no process reward.
3. Reserve **PPO** for cases where you already have a trained value network (rare) or want explicit entropy control.
4. Try **REINFORCE++** if you can't afford k ≥ 2 rollouts per prompt.

## Connections
- Components: [[rloo]], [[grpo]], [[dr-grpo]], [[reinforce-plus-plus]].
- Parent: [[vanilla-pg]].
- Baseline: [[ppo]].
- Framework implementations: [[verl-grpo]], [[trl-grpo]], [[openrlhf-ppo]].
- Practitioner blog: [[nathan-lambert-grpo]].
