---
chapter: ch-40
course: llm-training
phase: read
excerpt_of: arXiv:2505.10978v3 (Group-in-Group Policy Optimization for LLM Agent Training; code verl-agent)
source_url: https://arxiv.org/abs/2505.10978
created_at: "2026-09-15"
revised: "2026-09-15 (generality revision; written from the primary source — the library has no GiGPO card)"
---

# Excerpt: GiGPO — episode groups plus anchor-state step groups

Used by [[read]] §1, §6, §10, and the Recipe. Authors: Lang Feng, Zhenghai Xue, Tingcong Liu, Bo An (NTU Singapore; Skywork AI). NeurIPS 2025; code at github.com/langfengQ/verl-agent. Text read from arXiv v3 (28 Oct 2025) on 2026-09-15.

## Episode-level advantage (Eqs. 2–3)
`N` trajectories are collected for one task from identical initial states. With return `R(τ_i)` (for a binary end-of-episode reward, 1 for success and 0 for failure):
```
A_E(τ_i) = ( R(τ_i) − mean{R(τ_j)}_{j=1..N} ) / F_norm({R(τ_j)})
```
`F_norm = std` reproduces GRPO. The paper notes that this "may introduce a difficulty bias, where trajectories from low-variance groups (e.g. very easy or hard tasks) receive disproportionately large gradients" and offers `F_norm = 1`, which App. C shows equals the RLOO advantage up to the factor `N/(N−1)`.

## Step-level advantage from anchor states (Eqs. 4–7)
Because all trajectories start from the same state, environment states recur across and within trajectories. Each distinct state `s̃` becomes an *anchor state*, and the actions taken from it across the group form a step-level group `G_S(s̃)`, built with a hash table and no extra rollouts. Each action is scored by its discounted return `R_t = Σ_{k≥t} γ^{k−t} r_k`, and
```
A_S(a_t) = ( R_t − mean{R_t : (a, R) ∈ G_S(s̃)} ) / F_norm({R_t : … })
A(a_t)   = A_E(τ_i) + ω · A_S(a_t)      (Eq. 8)
```
with `ω ≥ 0` a weighting coefficient. The clipped objective (Eq. 9) is otherwise GRPO's, with a KL term to `π_ref`.

## Settings (App. hyperparameters)
ALFWorld: prompt 2048 tokens, response 512, up to 50 environment steps, actor LR 1e-6 (critic 1e-5 for PPO only), reward 10 for success and 0 otherwise with −0.1 for invalid actions, group size 8 and 16 groups per rollout (128 environments), rollout temperature 1.0, mini-batch 256, KL loss coefficient 0.01, `ω = 1` without tuning, `γ = 0.95`. WebShop: same except prompt 4096 tokens and 15 steps per episode. Search-augmented QA follows Search-R1 with group size 5 and 4 turns, KL coefficient 0.001.

## Results (Table 1, averaged over 3 seeds)
Qwen2.5-7B-Instruct, ALFWorld overall success / WebShop success: PPO (with critic) 80.4 / 68.7; RLOO 75.5 / 65.7; GRPO 77.6 / 66.1; GiGPO with std 90.8 / 72.8; GiGPO without std 90.2 / 75.2. Qwen2.5-1.5B-Instruct, ALFWorld overall: PPO 54.4, RLOO 69.7, GRPO 72.8, GiGPO with std 86.7, without std 86.1. Search-augmented QA (Table 2, 7-dataset average): GiGPO 42.1 at 3B and 47.2 at 7B, above Search-R1 and ZeroSearch.

## Statements the chapter uses
- The normalization factor "is task-dependent rather than universally helpful": on harder subtasks (Look, Pick2, WebShop) `F_norm = std` "could exaggerate gradients from overly difficult samples or highly imbalanced groups", and `F_norm = 1` gives higher success; elsewhere the two are similar (§5.2).
- Critic-based PPO is not uniformly behind group baselines: at 7B it exceeds GRPO and RLOO on ALFWorld, while at 1.5B it is far behind (Table 1).
- Step-level group sizes concentrate around 6–8 (equal to `N`) as training converges, and extreme sizes from early repetitive loops disappear (App. figure discussion).
