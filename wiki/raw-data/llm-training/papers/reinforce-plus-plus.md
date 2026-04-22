<!-- scope: REINFORCE++ — critic-free RLHF with global (batch-wide) advantage normalization, token-level KL, PPO-clip
     deps: [[vanilla-pg]], [[rloo]], [[grpo]]
     see-also: [[ppo]], [[dr-grpo]]
-->

# REINFORCE++: A Simple and Efficient Approach for Aligning Large Language Models
- **Core Insight:** Prompt-local advantage normalization (GRPO's per-group, RLOO's leave-one-out) is high-variance when groups are small; normalizing across the *global* batch gives a more accurate, lower-variance advantage, and combining that with token-level KL and PPO-clip recovers PPO's stability at REINFORCE's cost.
- **Guideline:** Use REINFORCE++ when you can only afford k=1–2 rollouts per prompt (large batch instead); apply token-level KL inside the advantage, PPO-clip on the importance ratio, and normalize advantages over the full mini-batch.
- **Authors:** Jian Hu
- **Year:** 2025
- **URL:** https://arxiv.org/abs/2501.03262
- **Relevant topics:** critic-free RLHF, advantage normalization, token-level KL, stable on-policy RL

## Abstract
Critic-free methods like RLOO and GRPO normalize advantages within small prompt-level groups, which can be inaccurate for small groups and lead to overfitting. REINFORCE++ normalizes advantages across the *global* mini-batch, retains PPO-style importance-ratio clipping for per-token stability, applies KL as a token-level shaped reward against π_ref, and drops the value network. It is simpler than GRPO (no grouping constraint), uses less memory than PPO (no critic), and trains stably on RLHF and reasoning benchmarks.

## Key Contributions
- **Global advantage normalization**: compute mean/std over the full mini-batch of sequences, not per-prompt.
- Keeps **PPO clip** to bound per-token ratio drift.
- Keeps **token-level KL** as a shaped per-token reward.
- Removes the value network entirely.
- Open implementation in OpenRLHF.

## Key Figures/Tables to Study
- **Algorithm 1:** REINFORCE++ pseudocode.
- **Figure 3:** Variance of advantages: group-normalized (GRPO) vs global (REINFORCE++).
- **Table 2:** RLHF benchmark results vs PPO, GRPO, RLOO.

## Technical Details

### Per-token shaped reward
`r̃_t = r(x,y) · 𝟙{t = T} − β · KL_t`
where `KL_t = log π_θ_old(y_t | ·) − log π_ref(y_t | ·)` (k1 estimator, one sign).
All terminal reward credited to the last token; KL penalty applied per token.

### Return and advantage
Cumulative return from step t:
`G_t = Σ_{t'≥t} γ^{t'−t} r̃_{t'}` (γ typically 1.0 for LLMs).

**Global advantage normalization** over the whole batch B:
`Â_t = (G_t − mean_{B}(G)) / std_{B}(G)`
No value network, no GAE, no group grouping.

### Loss (PPO-clip form)
`L(θ) = − E_t[ min( ρ_t(θ) Â_t,  clip(ρ_t(θ), 1-ε, 1+ε) Â_t ) ]`
where `ρ_t(θ) = π_θ(y_t | ·) / π_θ_old(y_t | ·)`.

### What's kept vs dropped (vs the family)
| Component | PPO | RLOO | GRPO | REINFORCE++ |
|-----------|-----|------|------|-------------|
| Value network | yes | no | no | **no** |
| Clip ε | yes | no | yes | **yes** |
| KL location | per-token reward | per-token reward | in-loss (k3) | **per-token reward** |
| Advantage baseline | learned V | leave-one-out | group mean/std | **global batch mean/std** |
| Group size requirement | — | k ≥ 2 | G ≥ 2 | **k = 1 OK** |

### Hyperparameters
| Knob | Value |
|------|-------|
| Clip ε | 0.2 |
| KL coef β | 0.01–0.05 |
| Learning rate | 5e-7 – 1e-6 |
| Global batch size | 512–2048 sequences |
| k (samples per prompt) | 1–4 |
| Epochs per rollout | 1 |
| Sampling T | 1.0 |

## Connections
- Parent of the family: [[vanilla-pg]].
- Direct predecessors: [[rloo]], [[grpo]].
- PPO baseline: [[ppo]].
- Bias-corrected GRPO: [[dr-grpo]].
- Framework: OpenRLHF (see [[openrlhf-ppo]]).
