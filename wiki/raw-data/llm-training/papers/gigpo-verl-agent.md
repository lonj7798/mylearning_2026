<!-- scope: GiGPO — a two-level, critic-free advantage estimator for multi-turn LLM agents (episode-level group plus anchor-state step groups), its ALFWorld / WebShop / search-QA results, and the verl-agent framework (step-wise rollout with a memory module) it is implemented in
     deps: [[grpo]], [[ppo]]
     see-also: [[verl-rollout]], [[verl-grpo]], [[dr-grpo]], [[dapo]], [[skyrl-agent]], [[agentrl]], [[deepswe]]
-->

# Group-in-Group Policy Optimization for LLM Agent Training
- **Core Insight:** Grouping actions that were taken from the same recurring environment state across a group of trajectories produces a step-level advantage at no extra rollout cost, and adding it to the episode-level GRPO advantage raises ALFWorld overall success from 72.8 ± 3.6 (GRPO) to 86.1 ± 4.7 and WebShop success from 56.8 ± 3.8 to 67.4 ± 4.5 on Qwen2.5-1.5B-Instruct over 3 seeds (Table 1).
- **Guideline:** When training a multi-turn agent with a group-based critic-free algorithm and episodes in which states recur, add anchor-state step groups with ω = 1 and γ = 0.95, because the added computation is 0.54 s against 362.83 s per iteration (§5.6) and removing either advantage level lowers success on every ALFWorld subtask (§5.4, Fig. 4). When the tasks are among the harder subtasks (Look, Pick2, WebShop), set F_norm = 1 rather than the group standard deviation, because std scaling scored lower there while the two were similar elsewhere (§5.2, Table 1).
- **Authors:** Lang Feng, Zhenghai Xue, Tingcong Liu, Bo An (Nanyang Technological University; Skywork AI)
- **Year:** 2025 (arXiv v1 2025-05; NeurIPS 2025)
- **URL:** https://arxiv.org/abs/2505.10978 — code: https://github.com/langfengQ/verl-agent
- **Source type:** paper
- **Relevant topics:** multi-turn agent RL, credit assignment, group-based advantage estimation, anchor-state grouping, critic-free RL, step-wise rollout frameworks

## Abstract
Group-based RL has driven single-turn tasks such as mathematical reasoning, but its scalability to multi-turn agent training is limited: interactions run over many steps with sparse or delayed rewards, so credit assignment across steps is difficult. GiGPO estimates relative advantage at two levels. At the episode level it computes a macro advantage over groups of complete trajectories. At the step level it constructs groups retroactively by identifying repeated environment states across trajectories; actions taken from the same state form a group and receive a micro relative advantage. The structure requires no auxiliary model and no additional rollouts. Evaluated on ALFWorld, WebShop and search-augmented QA with Qwen2.5-1.5B/3B/7B-Instruct, GiGPO reports gains above 12% on ALFWorld and above 9% on WebShop over GRPO, and 42.1% (3B) and 47.2% (7B) average on QA, at the same GPU memory and rollout cost.

## Key Contributions
- Episode-level advantage A^E(τ_i) = (R(τ_i) − mean{R(τ_j)}) / F_norm({R(τ_j)}), with F_norm either the group standard deviation (GRPO's default) or 1 (§4.1, Eq. 3).
- Anchor-state grouping: hashmap-based collection of every (action, discounted return) pair whose state equals a given distinct state s̃, across all N trajectories and all time steps (§4.2, Eqs. 4, 6).
- Step-level advantage from the discounted return R_t^(i) = Σ_{k≥t} γ^{k−t} r_k^(i), normalized within the anchor-state group (§4.2, Eqs. 5, 7).
- Combined advantage A(a_t^(i)) = A^E(τ_i) + ω · A^S(a_t^(i)) inside a clipped surrogate with a KL penalty to a reference policy (§4.3, Eqs. 8–9).
- verl-agent: a verl-based framework with a step-wise multi-turn rollout that does not concatenate the full interaction history, a configurable memory module that selects each step's input, and parallel gym-style group environments; it supports GiGPO, GRPO, PPO, DAPO, RLOO among others (App. A, Fig. 7).

## Key Figures/Tables to Study
- Table 1 (§5.2) — ALFWorld six subtasks and overall, WebShop score and success, for prompting, PPO, RLOO, GRPO and both GiGPO variants at 1.5B and 7B, 3 seeds.
- Table 2 (§5.3) — search-augmented QA, in-domain (NQ, HotpotQA) and out-of-domain datasets.
- Fig. 4 (§5.4) — ablation of A^E and A^S.
- Fig. 5 (§5.5) — step-level group-size distribution at iterations 10, 75, 140.
- Fig. 6 (§5.6) — per-iteration time breakdown.
- Tables 3–5 (App. E.3–E.5) — VLM agents, DAPO combination, ω sensitivity.

## Technical Details
1. **Group construction.** N trajectories are rolled out for a fixed task x from identical initial states, so states recur across trajectories (§4.1). With a binary end-of-episode reward, R(τ_i) = 1 for success and 0 for failure (§4.1).
2. **Why F_norm = 1 is offered.** Standard-deviation normalization is described as introducing a difficulty bias in which trajectories from low-variance groups receive disproportionately large gradients; F_norm = 1 yields an unbiased leave-one-out estimator up to a constant factor of N/(N−1) (§4.1; App. C, Eqs. 10–11).
3. **Cost of grouping.** Grouping is offline and uses hashmap lookups only; no extra LLM rollouts and no critic (§4.2, §5.6).
4. **ALFWorld / WebShop results (Table 1, 3 seeds, overall success %).** 1.5B: prompting 4.1, ReAct 12.8, Reflexion 21.8, PPO 54.4 ± 3.1, RLOO 69.7 ± 2.5, GRPO 72.8 ± 3.6, GiGPO w/ std 86.7 ± 1.7, GiGPO w/o std 86.1 ± 4.7. 7B: PPO 80.4 ± 2.7, RLOO 75.5 ± 4.6, GRPO 77.6 ± 5.2, GiGPO w/ std 90.8 ± 1.3, w/o std 90.2 ± 2.3. WebShop success at 1.5B: GRPO 56.8 ± 3.8, GiGPO w/ std 65.0 ± 3.2, w/o std 67.4 ± 4.5 (scores 75.8 / 83.1 / 83.5); at 7B: GRPO 66.1 ± 3.7, GiGPO w/o std 75.2 ± 3.8 (score 86.2 ± 2.6). Reference points: Gemini-2.5-Pro 60.3 ALFWorld and 35.9 WebShop success; GPT-4o 48.0 and 23.7.
5. **Search-augmented QA (Table 2).** Trained on NQ and HotpotQA with F_norm = std; average over seven datasets 42.1 at 3B and 47.2 at 7B, against Search-R1 32.5 / 38.5 and ZeroSearch 31.7 / 39.1. Under a limit of at most 3 tool calls per query, the 7B model uses about 0.9 calls on single-hop and about 1.6 on multi-hop tasks (§5.3).
6. **Ablation (§5.4, Fig. 4, 1.5B).** Removing A^E drops performance across all tasks; removing A^S drops it most on Cool, Pick2 and WebShop. The gap between the two F_norm variants is smaller than either structural ablation. The figure reports success rates on a radar plot; per-task numbers are not printed.
7. **Step-group dynamics (§5.5, Fig. 5, ALFWorld, 1.5B).** Groups of size 1 stay below 35% of anchor states throughout training, so over 65% of states recur. At iteration 10, sizes ≥ 10 exceed 20%. By iteration 75, 10 ≤ size < 50 falls from 16.2% to 12.1% and size ≥ 50 falls from 5.6% to 3.1%. At iteration 140 the distribution concentrates at sizes 6–8 with N = 8, alongside a success-rate plateau above 80%.
8. **Computational budget (§5.6, Fig. 6, ALFWorld, 1.5B).** Rollout, old and reference log-probabilities, and the policy update total 362.83 s per iteration; anchor-state grouping adds 0.01 s and step-advantage computation adds 0.53 s. The paper states these are "< 0.002% of the total per-iteration training time"; 0.54 / 362.83 is 0.15%, so the printed percentage does not follow from the printed seconds (see Verification).
9. **ω sensitivity (App. E.5, Table 5, WebShop, 1.5B).** Success rate by ω: 0.0 → 56.6, 0.2 → 63.1, 0.4 → 65.8, 0.6 → 67.2, 0.8 → 68.3, 1.0 → 67.4, 1.2 → 66.5, 1.4 → 56.3. The runs elsewhere use ω = 1 without tuning (§5.1).
10. **Combination with DAPO (App. E.4, Table 4, WebShop, 1.5B).** GRPO 56.8 ± 3.8 success, DAPO 66.1 ± 3.2, GiGPO_dynamic (GiGPO plus dynamic sampling and clip-higher) 75.0 ± 3.5.
11. **VLM agents (App. E.3, Table 3, Qwen2.5-VL-3B-Instruct).** Sokoban 6×6: prompting 11.7, GRPO 67.1 ± 4.7, GiGPO w/o std 81.0 ± 3.6. EZPoints: GRPO 86.9 ± 3.4, both GiGPO variants 100.0 ± 0.0.
12. **Environments.** ALFWorld: 3,827 task instances across six categories, up to 50 environment steps per episode. WebShop: over 1.1 million products and 12k user instructions (§5.1).
13. **Prompting and memory (App. E.2).** History length is 2 for ALFWorld and WebShop and the full history for search-augmented QA; the agent emits `<think>` and `<action>` blocks.

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Qwen2.5-1.5B/7B-Instruct + GiGPO, ALFWorld | 1.5B / 7B | RL | max prompt; max response; env steps per episode | 2048 tokens; 512 tokens; 50 | arXiv:2505.10978v3 App. E.1 | verified 2026-09-18 | no ablation reported |
| Qwen2.5-1.5B/7B-Instruct + GiGPO, ALFWorld | 1.5B / 7B | RL | actor LR; critic LR (PPO only); mini-batch; KL loss coefficient | 1e-6; 1e-5; 256; 0.01 | App. E.1 | verified 2026-09-18 | no ablation reported |
| Qwen2.5-1.5B/7B-Instruct + GiGPO, ALFWorld | 1.5B / 7B | RL | group size N; groups per rollout; total environments; rollout / validation temperature | 8; 16; 128; 1.0 / 0.4 | App. E.1 | verified 2026-09-18 | no ablation reported |
| Qwen2.5-1.5B/7B-Instruct + GiGPO, ALFWorld | 1.5B / 7B | RL | reward: success; failure; invalid action | 10; 0; −0.1 | App. E.1 | verified 2026-09-18 | no ablation reported |
| Qwen2.5-1.5B/7B-Instruct + GiGPO, WebShop | 1.5B / 7B | RL | max prompt; max response; env steps per episode; mini-batch | 4096 tokens; 512 tokens; 15; 64 | App. E.1 | verified 2026-09-18 | no ablation reported |
| Qwen2.5-1.5B/7B-Instruct + GiGPO, WebShop | 1.5B / 7B | RL | actor LR; group size; groups per rollout; KL coefficient; reward (success / failure / invalid) | 1e-6; 8; 16; 0.01; 10 / 0 / −0.1 | App. E.1 | verified 2026-09-18 | no ablation reported |
| Qwen2.5-3B/7B-Instruct + GiGPO, search QA | 3B / 7B | RL | max prompt; max response; max turns; group size; train data size; mini-batch; KL coefficient; reward (success / failure / invalid); temperatures | 4096; 512; 4; 5; 256; 512; 0.001; 1 / 0 / −0.01; 1.0 train, 0.0 validation | App. E.1 | verified 2026-09-18 | no ablation reported |
| GiGPO (all tasks) | — | RL | weighting coefficient ω; discount γ; anchor-state similarity threshold (QA only) | 1 (no tuning); 0.95; 0.9 longest-matching-subsequence | §5.1, App. E.1 | verified 2026-09-18 | App. E.5 Table 5: WebShop success peaks at ω = 0.8 (68.3) with ω = 1.0 at 67.4, 1.5B |
| GiGPO training runs | 1.5B / 7B (ALFWorld, WebShop) | RL | GPUs; iterations | 2×H100 / 4×H100; 150 | App. E.1 Computing Details | verified 2026-09-18 | no ablation reported |
| GiGPO training runs | 3B / 7B (search QA) | RL | GPUs; iterations | 4×H100 / 8×H100; 200 | App. E.1 Computing Details | verified 2026-09-18 | no ablation reported |

## Findings relevant to generality, negative feedback and agentic training
- Transfer within the QA task family: training on NQ and HotpotQA and evaluating on five datasets marked out-of-domain (TriviaQA, PopQA, 2Wiki, MuSiQue, Bamboogle) gives higher scores than Search-R1 and ZeroSearch on all of them at 7B (Table 2). **Result (single study).**
- No evaluation is reported outside the trained environments and QA datasets: no general instruction-following, math, code, or safety benchmark appears, so the source gives no evidence about capability retained or lost outside the trained families.
- Negative signal is negative-as-gradient: failed trajectories stay in their episode group and receive a below-mean advantage, and an action taken from a shared anchor state is scored relative to the other actions taken from that state, so a failing trajectory's better-than-average action can still receive a positive step advantage (§4.2, Fig. 3 example ordering A^S(1st Item) > A^S(2nd Item) > A^S(Next Page)).
- Invalid actions are penalized at −0.1 rather than masked (App. E.1).
- The step-group size distribution is an observable of repeated states and therefore of loops; the paper reads its shift between iterations 10 and 140 as the agent leaving dead ends (§5.5). **Interpretation.**

## Connections
- [[verl-rollout]] — verl's own multi-turn agent loop, which verl-agent replaces with the step-wise paradigm of App. A.
- [[verl-grpo]] — the episode-level estimator that Eq. 3 reproduces.
- [[dr-grpo]] — the same difficulty-bias argument against standard-deviation normalization; GiGPO finds the choice task-dependent.
- [[dapo]] — dynamic sampling and clip-higher, combined with GiGPO in App. E.4.
- [[skyrl-agent]], [[deepswe]] — agentic RL stacks that use episode-level advantages only.

## Verification
- Created on 2026-09-18 from https://arxiv.org/abs/2505.10978 (arXiv v3, 2025-10-28; NeurIPS 2025).
- Corrections to the previous card version: none (new card). Chapter-local excerpts exist at `wiki/courses/llm-training/ch-55/excerpts/gigpo-verl-agent.md` and `ch-46a/excerpts/gigpo-verl-agent.md`, both written before this card and consistent with it.
- Removed as unsupported by the source: none.
- Chapter claims not found in the source: none. Every cited value was read at the stated locus — ch-40 (Table 1 PPO/RLOO/GRPO/GiGPO at both sizes; §5.2 F_norm; Eqs. 5–8; App. E.1 hyperparameters), ch-45b (Eqs. 3–7, §4), ch-46a (App. E.1 rewards, limits and hyperparameters; §5.5 group sizes; §5.6 timings; App. E.5 ω sweep), ch-55 (Eqs. 3–8, App. A, ω = 1 without tuning), ch-58 (Table 1, Fig. 6, App. A and E.1). One arithmetic note: ch-58's table already flags that the paper's "< 0.002%" does not match its own 0.54 s / 362.83 s (0.15%); the discrepancy is in the source, not in the chapter.
- Not reported by the source: results above 7B; any non-agentic benchmark; ω sensitivity outside WebShop; wall-clock or GPU-hour totals beyond per-iteration seconds and iteration counts; number of evaluation episodes per checkpoint.
