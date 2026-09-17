---
chapter: ch-46a
course: llm-training
phase: read
excerpt_of: "Group-in-Group Policy Optimization for LLM Agent Training (arXiv:2505.10978v3, 2025-10-28; NeurIPS 2025)"
source_url: https://arxiv.org/abs/2505.10978
created_at: "2026-09-15"
note: "No library card exists for this source on 2026-09-15 (planned slug gigpo-verl-agent). Values read from the v3 PDF on 2026-09-15. Code: github.com/langfengQ/verl-agent."
---

# Excerpt: GiGPO — episode-level and step-level groups for multi-turn agent RL

**Authors:** Lang Feng, Zhenghai Xue, Tingcong Liu, Bo An (Nanyang Technological University; Skywork AI).

## Setting (§3, §5.1)

- An episode is a trajectory τ = {(s₁, a₁, r₁), …, (s_T, a_T, r_T)}; the policy π_θ(a_t | s_t, x) emits one textual action per turn, conditioned on the task prompt x and the current state.
- Base models: Qwen2.5-1.5B / 3B / 7B-Instruct. Environments: ALFWorld (3,827 task instances, six categories), WebShop, and search-augmented QA (NQ, TriviaQA, PopQA, HotpotQA, 2Wiki, MuSiQue, Bamboogle).
- All RL methods in the ALFWorld and WebShop comparison "use exactly the same hyperparameter configurations" (§5.1). Results are averaged over 3 random seeds (Table 1 caption).

## Advantage construction (§4.1-§4.3)

Episode-level (Eq. 3), with R(τ_i) the total return of trajectory i in a group of N trajectories rolled out from the same task and the same initial state:

```
A^E(τ_i) = ( R(τ_i) − mean{R(τ_j)}_{j=1..N} ) / F_norm({R(τ_j)}_{j=1..N})
```

`F_norm = std` is GRPO's default; `F_norm = 1` gives "an unbiased Leave-One-Out estimator" and the paper reports it as more stable on low-variance (very easy or very hard) groups.

Step-level: let U be the set of distinct environment states in the group. For each anchor state s̃ ∈ U, the group is G^S(s̃) = {(a_t^{(i)}, R_t^{(i)}) | s_t^{(i)} = s̃}, with discounted return R_t^{(i)} = Σ_{k=t..T} γ^{k−t} r_k^{(i)} (Eq. 5), and

```
A^S(a_t^{(i)}) = ( R_t^{(i)} − mean{R_t^{(j)} in G^S(s̃)} ) / F_norm({R_t^{(j)} in G^S(s̃)})     (Eq. 7)
A(a_t^{(i)}) = A^E(τ_i) + ω · A^S(a_t^{(i)})                                                    (Eq. 8)
```

Anchor grouping "incurs no extra rollouts: it is entirely offline and requires only lightweight key-based grouping using hashmaps" (§4.2). The objective (Eq. 9) is a clipped importance-ratio objective with a KL penalty term βD_KL(π_θ ‖ π_ref).

## Hyperparameters (App. E.1)

| Setting | ALFWorld | WebShop | Search QA |
|---|---|---|---|
| Max prompt / response tokens | 2,048 / 512 | 4,096 / 512 | 4,096 / 512 |
| Turn budget | ≤ 50 environment steps | ≤ 15 environment steps | max turn 4 |
| Actor LR (critic LR, PPO only) | 1e-6 (1e-5) | 1e-6 (1e-5) | 1e-6 |
| Reward | 10 success, 0 failure; invalid action −0.1 | same | 1 success, 0 failure; invalid −0.01 |
| Group size N × groups per rollout | 8 × 16 = 128 environments | 8 × 16 = 128 | 5 |
| Rollout / validation temperature | 1.0 / 0.4 | 1.0 / 0.4 | 1.0 / 0.0 |
| Mini-batch | 256 | 64 | 512 |
| KL-divergence loss coefficient | 0.01 | 0.01 | 0.001 |
| ω; discount γ | 1; 0.95 | 1; 0.95 | 1; 0.95 |
| History in prompt | last 2 observations and actions | last 2 | full history |

Compute (App. E.1): Qwen2.5-1.5B on 2×H100 and Qwen2.5-7B on 4×H100 for ALFWorld and WebShop, 150 iterations each; search QA runs 200 iterations on 4×H100 (3B) and 8×H100 (7B). The prompt template requires reasoning inside `<think> </think>` and the action inside `<action> </action>` (App. E.2, Fig. 8).

## Results (Table 1, Table 2, Table 5)

ALFWorld overall success rate (%) / WebShop success rate (%), mean ± s.d. over 3 seeds:

| Model | Method | ALFWorld All | WebShop Succ. |
|---|---|---|---|
| Qwen2.5-1.5B-Instruct | PPO (with critic) | 54.4 ± 3.1 | 51.5 ± 2.9 |
| Qwen2.5-1.5B-Instruct | RLOO | 69.7 ± 2.5 | 52.1 ± 6.7 |
| Qwen2.5-1.5B-Instruct | GRPO | 72.8 ± 3.6 | 56.8 ± 3.8 |
| Qwen2.5-1.5B-Instruct | GiGPO (F_norm = std) | 86.7 ± 1.7 | 65.0 ± 3.2 |
| Qwen2.5-1.5B-Instruct | GiGPO (F_norm = 1) | 86.1 ± 4.7 | 67.4 ± 4.5 |
| Qwen2.5-7B-Instruct | GRPO | 77.6 ± 5.2 | 66.1 ± 3.7 |
| Qwen2.5-7B-Instruct | GiGPO (F_norm = std) | 90.8 ± 1.3 | 72.8 ± 3.2 |
| Qwen2.5-7B-Instruct | GiGPO (F_norm = 1) | 90.2 ± 2.3 | 75.2 ± 3.8 |

Prompting baselines at 1.5B: Qwen2.5 4.1 ALFWorld / 5.2 WebShop; ReAct 12.8 / 11.3; Reflexion 21.8 / 21.9. Closed models: GPT-4o 48.0 / 23.7; Gemini-2.5-Pro 60.3 / 35.9.

Search QA average over 7 datasets (2 in-domain, 5 out-of-domain): Qwen2.5-3B GiGPO 42.1 vs Search-R1 32.5; Qwen2.5-7B GiGPO 47.2 vs Search-R1 38.5 (Table 2). Under a limit of at most 3 tool calls per query, the 7B model uses about 0.9 calls on single-hop and about 1.6 on multi-hop tasks (§5.3).

ω sensitivity on WebShop with Qwen2.5-1.5B-Instruct (Table 5): score 76.2 / 79.6 / 82.4 / 83.5 / 84.9 / 83.5 / 82.6 / 77.0 and success rate 56.6 / 63.1 / 65.8 / 67.2 / 68.3 / 67.4 / 66.5 / 56.3 at ω = 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4.

## Step-group dynamics and cost (§5.5, §5.6)

- Step-level groups of size 1 account for less than 35% throughout training, so over 65% of states recur across the 8 trajectories of a group.
- At iteration 10, groups with |G^S(s̃)| ≥ 10 exceed 20%; the authors write that "immature policies often produce invalid actions or fall into repetitive loops". By iteration 75, 10 ≤ |G^S(s̃)| < 50 falls from 16.2% to 12.1% and |G^S(s̃)| ≥ 50 from 5.6% to 3.1%. By iteration 140 the distribution concentrates at group sizes 6 to 8 (N = 8), together with a success-rate plateau above 80%.
- Per-iteration time on ALFWorld with Qwen2.5-1.5B-Instruct: rollouts, old and reference log-probabilities, and the policy update total 362.83 s; anchor state grouping adds 0.01 s and the step-advantage computation 0.53 s, "< 0.002% of the total per-iteration training time" (§5.6).

## Ablations and limits (§5.2, §5.4)

- Removing A^E or removing A^S both lower success across tasks (Fig. 4, no table of numbers).
- F_norm is "task-dependent rather than universally helpful": F_norm = 1 is higher on the harder tasks (Look, Pick2, WebShop) and comparable elsewhere (§5.2).
- Not reported: any evaluation outside the trained environments and the QA datasets (no general-capability benchmarks), context-management ablations, and the SFT stage (all runs start from the instruct checkpoints).

## Used in

ch-46a §3 (multi-turn RL configuration and the group-size instrumentation), §6 (loop and void turns), and the Recipe table.
