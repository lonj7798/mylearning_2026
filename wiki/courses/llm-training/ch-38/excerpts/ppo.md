---
chapter: ch-38
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/ppo.md
source_url: https://arxiv.org/abs/1707.06347
revised_at: "2026-09-15"
---

# Excerpt: PPO clipped objective (Schulman et al. 2017)

Checked against arXiv:1707.06347v2 on 2026-09-15. Used in read.md §2 and §3.

## Objective (§3, Eq. 6-7)
- r_t(θ) = π_θ(a_t|s_t) / π_θold(a_t|s_t), so r(θ_old) = 1.
- L^CPI(θ) = Ê_t[r_t(θ) Â_t]. "Without a constraint, maximization of L^CPI would lead to an excessively large policy update."
- L^CLIP(θ) = Ê_t[min(r_t(θ)Â_t, clip(r_t(θ), 1 − ε, 1 + ε)Â_t)], "where epsilon is a hyperparameter, say, ε = 0.2".
- The paper states that L^CLIP is a lower (pessimistic) bound on the unclipped objective and equals L^CPI to first order around θ_old (§3). It is not a bound on the true return.

## Adaptive KL penalty (§4, Eq. 8)
- L^KLPEN = Ê_t[r_t Â_t − β KL[π_θold, π_θ]]; if d < d_targ/1.5 then β ← β/2; if d > 1.5 d_targ then β ← 2β.

## Combined objective and GAE (§5, Eq. 9-12)
- L^{CLIP+VF+S} = Ê_t[L^CLIP − c_1 (V_θ(s_t) − V_t^targ)² + c_2 S[π_θ](s_t)].
- δ_t = r_t + γV(s_{t+1}) − V(s_t); Â_t = δ_t + (γλ)δ_{t+1} + … (truncated GAE).

## Table 1 (7 MuJoCo tasks × 3 seeds, normalized score)
| Variant | Score |
|---|---|
| No clipping or penalty | −0.39 |
| Clipping ε = 0.1 / 0.2 / 0.3 | 0.76 / 0.82 / 0.70 |
| Adaptive KL, best (d_targ = 0.01) | 0.74 |
| Fixed KL, best (β = 3) | 0.72 |

## Settings by benchmark (App. A)
- MuJoCo (Table 3): horizon 2048, 10 epochs, minibatch 64, Adam 3e-4, γ 0.99, λ 0.95; no parameter sharing and no entropy bonus (§6.1).
- Atari (Table 5): 3 epochs, c_1 = 1, c_2 = 0.01, ε = 0.1·α with α annealed from 1 to 0.
- The paper has no language-model experiments.
