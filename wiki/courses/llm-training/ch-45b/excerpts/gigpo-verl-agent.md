---
chapter: ch-45b
course: llm-training
phase: read
excerpt_of: arXiv:2505.10978v3 (no library card at the time of writing; chapter-local verified extract)
source_url: https://arxiv.org/abs/2505.10978
created_at: "2026-09-15"
---

# Excerpt: Group-in-Group Policy Optimization for LLM Agent Training (GiGPO)

- **Authors:** Lang Feng, Zhenghai Xue, Tingcong Liu, Bo An (Nanyang Technological University; Skywork AI)
- **Year:** 2025 (arXiv v1 2025-05; v3 2025-10-28; NeurIPS 2025). Code: github.com/langfengQ/verl-agent
- **Source type:** paper
- **Used in:** [[read]] §4, Recipe, Generalization lens, figure `gigpo-anchor-grouping.html`

## Problem setup (§3)
A trajectory is `τ = {(s_1, a_1, r_1), …, (s_T, a_T, r_T)}`; each action `a_t ∈ V^n` is a full token sequence,
and the policy is `π_θ(a_t | s_t, x)` for task prompt `x`. Rewards are "sparse or delayed (e.g., success and
failure signals at the end of an episode)".

## Two-level advantage
- **Episode level (Eq. 3):** `A^E(τ_i) = (R(τ_i) − mean{R(τ_j)}_{j=1..N}) / F_norm({R(τ_j)})`. `F_norm = std`
  reproduces GRPO; `F_norm = 1` "yields an unbiased Leave-One-Out estimator" (§4.1).
- **Anchor state grouping (Eq. 4, Eq. 6):** let `U` be the set of distinct environment states in the group of
  `N` trajectories. For each anchor state `s̃ ∈ U`,
  `G^S(s̃) = {(a_t^{(i)}, R_t^{(i)}) | s_t^{(i)} = s̃, 1 ≤ i ≤ N, 1 ≤ t ≤ T}`.
  Grouping "incurs no extra rollouts: it is entirely offline and requires only lightweight key-based grouping
  using hashmaps" (§4.2).
- **Discounted step return (Eq. 5):** `R_t^{(i)} = Σ_{k=t..T} γ^{k−t} r_k^{(i)}`.
- **Step advantage (Eq. 7):** `A^S(a_t^{(i)}) = (R_t^{(i)} − mean{R_t^{(j)} ∈ G^S(s̃)}) / F_norm({R_t^{(j)} ∈ G^S(s̃)})`.
- **Combination (Eq. 8):** `A(a_t^{(i)}) = A^E(τ_i) + ω · A^S(a_t^{(i)})`, with `ω ∈ R≥0`. The objective (Eq. 9) is
  the clipped PPO surrogate plus `−β D_KL(π_θ ‖ π_ref)`.

## Settings (§5.1)
Qwen2.5-1.5B/3B/7B-Instruct. `ω = 1` "with no further tuning". ALFWorld and WebShop: `N = 8`, identical
hyperparameters for GiGPO and every RL baseline. Search-augmented QA: settings follow Search-R1, `N = 5`,
max 4 turns, E5 retriever, similarity-based anchor grouping with a longest-matching-subsequence threshold of 0.9.

## Results
**Table 1 (ALFWorld overall success % / WebShop success %, mean ± std over 3 seeds).**

| Model | Method | ALFWorld All | WebShop Succ. |
|---|---|---|---|
| Qwen2.5-1.5B-Instruct | PPO (with critic) | 54.4 ± 3.1 | 51.5 ± 2.9 |
| Qwen2.5-1.5B-Instruct | RLOO | 69.7 ± 2.5 | 52.1 ± 6.7 |
| Qwen2.5-1.5B-Instruct | GRPO | 72.8 ± 3.6 | 56.8 ± 3.8 |
| Qwen2.5-1.5B-Instruct | GiGPO w/ std | 86.7 ± 1.7 | 65.0 ± 3.2 |
| Qwen2.5-1.5B-Instruct | GiGPO w/o std | 86.1 ± 4.7 | 67.4 ± 4.5 |
| Qwen2.5-7B-Instruct | GRPO | 77.6 ± 5.2 | 66.1 ± 3.7 |
| Qwen2.5-7B-Instruct | GiGPO w/ std | 90.8 ± 1.3 | 72.8 ± 3.2 |
| Qwen2.5-7B-Instruct | GiGPO w/o std | 90.2 ± 2.3 | 75.2 ± 3.8 |

§5.2 states the deltas over GRPO as +13.3 (ALFWorld) and +10.6 (WebShop) at 1.5B, +12.6 and +9.1 at 7B.

**Table 2 (search-augmented QA, trained on NQ and HotpotQA; ⋆ = out-of-domain).** Seven-dataset average:
3B — Search-R1 32.5, ZeroSearch 31.7, GiGPO 42.1; 7B — Search-R1 38.5, ZeroSearch 39.1, GiGPO 47.2.
Bamboogle⋆ at 7B: Search-R1 36.8, GiGPO 68.9.

**Ablation (Fig. 4, Qwen2.5-1.5B-Instruct).** Removing either level degrades every task; the gap between
`F_norm = std` and `F_norm = 1` is "comparatively minor compared to that observed in structural ablations" (§5.4).

**Step-group dynamics (§5.5, Fig. 5, ALFWorld).** Groups of size 1 stay below 35% throughout training, so
"over 65%" of states recur across trajectories. Sizes `10 ≤ |G^S| < 50` fall from 16.2% to 12.1% and `≥ 50` from
5.6% to 3.1% between iteration 10 and 75; by iteration 140 the distribution concentrates at sizes 6 to 8 with `N = 8`.

**Cost (§5.6, Fig. 6, ALFWorld, Qwen2.5-1.5B-Instruct).** Rollout, old- and reference-probability computation and
the policy update total 362.83 s per iteration; anchor state grouping adds 0.01 s and step-advantage computation
0.53 s, "< 0.002% of the total per-iteration training time". GPU memory and rollout cost are identical to GRPO.

**Stated limitation (§6).** Anchor grouping relies on state matching; "in the extreme case where no states are
repeated across trajectories (i.e., A^S = 0), it naturally degrades to GRPO".

## Verification
- Checked on 2026-09-15 against the cached PDF text of arXiv:2505.10978v3 (§3-§6, Tables 1-2, Figs. 3-6).
- Not reported in the cached text: learning rate, batch size, discount factor γ used in the experiments, KL
  coefficient β, and the Appendix E.1 hyperparameter table.
