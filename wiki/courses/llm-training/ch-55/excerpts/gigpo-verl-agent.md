---
chapter: ch-55
course: llm-training
phase: read
excerpt_of: arXiv:2505.10978v3 (no library card yet)
source_url: https://arxiv.org/abs/2505.10978
created_at: "2026-09-17"
---

# Excerpt: Group-in-Group Policy Optimization for LLM Agent Training (GiGPO) and verl-agent

**Artifact:** Lang Feng, Zhenghai Xue, Tingcong Liu, Bo An. "Group-in-Group Policy Optimization for LLM Agent Training", arXiv:2505.10978v3 (v1 2025-05; NeurIPS 2025). Code: https://github.com/langfengQ/verl-agent
**Status:** the llm-training library has no `gigpo-verl-agent` card yet; this excerpt holds the verified extract the chapter cites.

---

## The problem, as the paper states it

Group-based RL (GRPO, RLOO) succeeded on single-turn tasks where reward arrives immediately. Agent episodes are long — "an ALFWorld episode may include up to 50 steps and over 20k tokens" — and rewards are sparse or delayed, so a single episode-level advantage copied to every token "collapses step-level distinctions" (§1).

## Mechanism

1. **Episode level.** Sample N trajectories from N environments initialized to identical states. With a rule-based reward `R(τ) = 1` on success and 0 on failure, the episode relative advantage is

   `A^E(τ_i) = (R(τ_i) − mean{R(τ_j)}) / F_norm({R(τ_j)})`   (Eq. 3)

   `F_norm = std` is GRPO's default; the paper notes this "may introduce a difficulty bias, where trajectories from low-variance groups (e.g., very easy or hard tasks) receive disproportionately large gradients" (§4).

2. **Anchor state grouping.** Let `U` be the set of distinct environment states appearing anywhere in the N trajectories. Each distinct state `s̃` is an anchor; the step-level group collects every (action, return) pair taken from that state across trajectories and time steps (Eqs. 4, 6). Recurrence is common because ineffective actions and loops revisit the same webpage, room, or scene (§4).

3. **Step level.** Each element gets the discounted return `R_t^(i) = Σ_{k≥t} γ^{k−t} r_k^(i)` (Eq. 5), and

   `A^S(a_t^(i)) = (R_t^(i) − mean over G_S(s̃)) / F_norm(over G_S(s̃))`   (Eq. 7)

4. **Combination.** `A(a_t^(i)) = A^E(τ_i) + ω·A^S(a_t^(i))` (Algorithm 1, line 15). No extra rollouts and no critic are required, so the method keeps the critic-free, low-memory properties of group RL; the paper reports the added cost as "only < 0.002% time cost" (§1).

## Reported results

Averaged over 3 random seeds (Table 1). Success rate (%), ALFWorld "All" column and WebShop score/success:

| Model | Method | ALFWorld All | WebShop Score | WebShop Succ. |
|---|---|---|---|---|
| Qwen2.5-1.5B-Instruct | PPO (with critic) | 54.4±3.1 | 73.8±3.0 | 51.5±2.9 |
| Qwen2.5-1.5B-Instruct | RLOO | 69.7±2.5 | 73.9±5.6 | 52.1±6.7 |
| Qwen2.5-1.5B-Instruct | GRPO | 72.8±3.6 | 75.8±3.5 | 56.8±3.8 |
| Qwen2.5-1.5B-Instruct | GiGPO (F_norm = std) | 86.7±1.7 | 83.1±1.6 | 65.0±3.2 |
| Qwen2.5-1.5B-Instruct | GiGPO (F_norm = 1) | 86.1±4.7 | 83.5±1.8 | 67.4±4.5 |
| Qwen2.5-7B-Instruct | PPO (with critic) | 80.4±2.7 | 81.4±3.1 | 68.7±5.1 |
| Qwen2.5-7B-Instruct | GRPO | 77.6±5.2 | 79.3±2.8 | 66.1±3.7 |

Summary sentence (§5.2): "GiGPO w/o std surpasses GRPO by 13.3% on ALFWorld and 10.6% on WebShop at 1.5B, and by 12.6% and 9.1%, respectively, at 7B."

Search-augmented QA (Table 2): GiGPO reaches 42.1% at 3B and 47.2% at 7B, above Search-R1, ZeroSearch and StepSearch. Under a limit of at most 3 tool calls per query, the 7B model uses about 0.9 calls on average (§5.3).

Ablation on the normalizer (§5.2): `F_norm = std` "could exaggerate gradients from overly difficult samples or highly imbalanced groups" on the harder subtasks (Look, Pick2, WebShop), where `F_norm = 1` gave higher success; on other tasks the two were similar.

## Training settings quoted

ALFWorld (App. E.1): max prompt length 2048 tokens, max response length 512 tokens, up to 50 environment steps per episode, actor LR 1e-6 (critic 1e-5, PPO only), rule-based reward, group size N = 8. Search-augmented QA (§5.1): settings follow Search-R1, E5 retriever, group size N = 5, max 4 turns, anchor states matched by longest-matching-subsequence similarity above 0.9.

## verl-agent (App. A)

Built on verl and extended with: (1) a step-wise multi-turn interaction paradigm that avoids concatenating full interaction histories, for memory control on long horizons; (2) a customizable memory module choosing which history each step sees; (3) parallel group-based environments behind a gym-style interface.

## Connections

- [[verl-rollout]] — verl's own multi-turn agent loop and response masks, which verl-agent replaces with a step-wise paradigm.
- [[verl-grpo]] — the episode-level estimator GiGPO extends.
- [[dr-grpo]] — the same difficulty-bias argument about std normalization.

## Verification

- Checked on 2026-09-17 against arXiv:2505.10978v3 (28 Oct 2025): Abstract, §1, §4, §5.1–5.3, §6, Table 1, Table 2, Algorithm 1, App. A, App. E.1.
- Not reported by the source: any evaluation outside ALFWorld, WebShop, and search-augmented QA; any measurement of ω sensitivity beyond the reported runs; any result at a model size above 7B.
