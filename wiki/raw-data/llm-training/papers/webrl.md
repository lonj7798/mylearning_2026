<!-- scope: WebRL (Qi et al., ICLR 2025): online RL for web agents in WebArena with a self-evolving curriculum built from failed tasks, a trained outcome reward model, a KL-constrained off-policy update, and a perplexity-filtered success-only replay buffer
     deps: [[ppo]], [[webarena-data]]
     see-also: [[agenttuning]], [[glm-4]], [[glm-4-5]], [[evol-instruct]], [[agentgym-rl]], [[agentic-finetuning-misalignment]], [[slime]]
-->

# WebRL: Training LLM Web Agents via Self-Evolving Online Curriculum Reinforcement Learning
- **Core Insight:** Online RL in WebArena with tasks generated from the agent's own failures, an 8B outcome reward model, and a KL-constrained update raises Llama-3.1-8B from 4.8% to 42.4% and GLM-4-9B from 6.1% to 43.0% success on WebArena-Lite, against 30.3% and 31.5% for DigiRL on the same backbones (Abstract; §3.2, Table 1).
- **Guideline:** When an agentic environment has few training tasks and only a binary final reward, generate each phase's tasks from previously failed instructions, keep only tasks whose critic value at the initial state is between 0.05 and 0.75, replay only successful past trajectories whose actions have actor perplexity in [1/0.95, 1/0.5], and constrain the update to the previous phase's policy (§2.1–§2.2; Table 2; App. C, Fig. 16).
- **Authors:** Zehan Qi, Xiao Liu, Iat Long Iong, Hanyu Lai, Xueqiao Sun, Wenyi Zhao, et al. (Tsinghua University, Zhipu AI; 14 authors, last-listed Yuxiao Dong)
- **Year:** 2024 (arXiv v1 2024-11-04; v3 2025-01-27; ICLR 2025)
- **URL:** https://arxiv.org/abs/2411.02337 (code, model, data: https://github.com/THUDM/WebRL)
- **Source type:** paper
- **Relevant topics:** agentic RL, web agents, curriculum task generation, outcome reward model, KL-constrained policy update, replay buffer, catastrophic forgetting, credit assignment with sparse reward

## Abstract
Open LLMs perform poorly as web agents, and strong web agents depend on proprietary APIs. WebRL is an online curriculum RL framework that trains open LLMs as web agents. It addresses three problems: few training tasks, sparse and costly feedback, and policy distribution drift during online learning. It uses (1) a self-evolving curriculum that generates new tasks from unsuccessful attempts, (2) an outcome-supervised reward model (ORM), and (3) adaptive RL strategies. On WebArena-Lite it raises Llama-3.1-8B from 4.8% to 42.4% and GLM-4-9B from 6.1% to 43%, above GPT-4-Turbo (17.6%), GPT-4o (13.9%) and AutoWebGLM (18.2%).

## Key Contributions
- Self-evolving curriculum: in-breadth evolution (Evol-Instruct style) seeded with failed instructions, then critic-based difficulty filtering and a GPT-4o feasibility filter (§2.1).
- ORM that answers YES/NO from instruction, action history and final-state HTML (§2; App. B.3).
- KL-constrained off-policy loss derived from the maximum-entropy RL view of the RLHF objective (§2.2, Eqs. 1–6).
- Success-only replay buffer filtered by actor perplexity (§2.2; Table 2).
- Error-type, step-count and complexity analyses showing gains on long-horizon tasks (§3.4–3.6).

## Key Figures/Tables to Study
- Fig. 2 (pipeline), Table 1 (per-site success rates), Fig. 5 (ablation curves over 8 phases), Table 2 (perplexity window), Fig. 7 (β with and without replay), Table 3 (ORM accuracy), Fig. 16 (success-only vs all-trajectory replay).

## Technical Details
**Setting (§2).** Finite-horizon MDP; state = HTML of the current page + action history; reward 1 on task success, else 0. Action space: Click, Hover, Type, Search, Press, Scroll, Select dropdown, New tab, Tab focus, Close tab, Goto, Go back, Go forward, Exit (App. B.1).

**Objective and loss (§2.2).** With π_ref = previous-phase policy and phase instruction distribution ρ(I):
`max_θ E[Σ_t (r(s_t,a_t,I) + β log π_ref(a_t|s_t,I)) + β H(π_θ)]` (Eq. 1). Using the optimal-policy relation β log(π*/π_ref) = A* (Eq. 4), the loss is
`L(π_θ) = E_ν[(β log(π_θ(a|s,I)/π_ref(a|s,I)) − A*(s,a,I))²]` (Eq. 5), off-policy over the phase's experience distribution ν. Its gradient is −2β E[(A* − β log(π_θ/π_ref)) ∇ log π_θ] (Eq. 6): positive advantage increases log-probability more when π_θ < π_ref; negative advantage decreases it less when π_θ < π_ref.

**Advantage (§2.2).** Value network trained with cross-entropy against the final binary reward (Eq. 7). Advantage mixes the one-step TD term and the final-reward term: A = λ(r_t + V(s_{t+1}) − V(s_t)) + (1 − λ)(r_T − V(s_t)), λ = 0.5 (Eq. 8). The critic is the SFT model plus a randomly initialized value head (§3.1).

**Curriculum (§2.1; App. B.1, Algorithm 1).**
1. SFT on WebArena-Lite demonstrations; roll out on its instructions; failed instructions initialize the failure set F; the replay buffer B is initialized with D₀ (Alg. 1 lines 1–5).
2. Each phase: GPT-4o generates instructions with examples from F; keep those with critic score in [0.05, 0.75]; remove infeasible ones with a GPT-4o prompt; select 500 instructions (§2.1; App. B.1).
3. Roll out; label with the ORM; add replay data with perplexity in [1/0.95, 1/0.5], capped at twice the new interaction data; train actor and critic; add successes to B and failures to F (Alg. 1 lines 15–20; App. B.1).

**ORM (§3.1; App. B.3).** WebArena-Lite has 1,186 training samples with reward functions; instructions are rewritten and variables changed; rollouts of baselines on these tasks plus the originals give 12,200 labeled samples. Input keeps only final-state HTML because of context limits. Prediction compares P(YES) vs P(NO). Accuracy (Table 3), test / rollout: ORM (8B) 80.8 / 79.4; GPT-4 71.9 / 71.2; Captioner + GPT-4 72.6 / 73.3; GPT-4V 71.2 / 70.5.

**Results (Table 1, WebArena-Lite 165 tasks; Reddit / GitLab / CMS / Map / OSS / Avg).** Llama-3.1-8B + WebRL 63.2 / 46.7 / 54.3 / 36.7 / 31.1 / 42.4; + DigiRL 30.3 avg; + AWR 28.5; + Filtered BC 23.0; + SFT 20.6. GLM-4-9B + WebRL 57.9 / 50.0 / 48.6 / 36.7 / 37.8 / 43.0; + DigiRL 31.5. Llama-3.1-70B + WebRL 49.1 avg (SFT 23.0).

**Ablations.** Replay perplexity window, phase-1 success rate (Table 2): all data [1, ∞] 29.1; [1, 1/0.95] 27.9; [1/0.95, 1/0.5] 31.5; [1/0.5, ∞] 23.0. β (Fig. 7): β = 0.01 degrades with or without replay; without replay, performance falls for β ≥ 1; with replay it stays high at large β. Without the replay buffer, performance declines over phases; REINFORCE with a value baseline (no KL) overfits each phase and stays below its initial value when replay is also removed (§3.7, Fig. 5).

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Llama-3.1-8B / GLM-4-9B / Llama-3.1-70B WebRL actor | 8B / 9B / 70B | SFT | LR; schedule; warmup; batch; epochs; cutoff | 1e-5; cosine; 0.1; 128; 1; 16384 | arXiv:2411.02337v3 App. B, Table 5 | verified 2026-11-24 | no ablation reported |
| WebRL actor and critic | 8B / 9B / 70B | RL | actor LR; critic LR; schedule; batch; discount; epochs per phase; rollout temperature | 1e-6; 1e-6; constant; 128; 0.9; 1 / 1; 1 | Table 5 | verified 2026-11-24 | no ablation reported |
| WebRL | — | RL | GAE mixing λ | 0.5 | §2.2, Eq. 8 | verified 2026-11-24 | no ablation reported |
| WebRL | — | RL | KL coefficient β used in main runs | not reported (β swept 0.01–10 in Fig. 7 only) | checked §2.2, §3.7, App. B | not reported | Fig. 7 |
| WebRL curriculum | — | data | instructions per phase; critic filter; phases | 500; [0.05, 0.75]; Fig. 5 x-axis runs to 8 (App. B.2: DigiRL "also conducts 8 rounds") | App. B.1; §2.1; Fig. 5 | verified 2026-11-24 | Fig. 5: w/o curriculum rises more slowly to a lower ceiling |
| WebRL replay | — | data | buffer content; perplexity window; cap | successes only; [1/0.95, 1/0.5]; ≤ 2× new data | §2.2; App. B.1 | verified 2026-11-24 | Table 2; Fig. 16 |
| ORM | 8B | reward-model | LR; schedule; warmup; batch; epochs; cutoff; samples | 5e-6; cosine; 0.1; 128; 4; 16384; 12,200 | Table 6; App. B.3 | verified 2026-11-24 | Table 3 accuracy |

## Findings relevant to generality and negative feedback
- **Negative feedback.** Failed instructions are not used as gradient signal from the replay buffer; they seed new tasks (Alg. 1 line 11). Adding failed past trajectories to replay made improvement "slower and more volatile", because correct actions inside failed trajectories receive negative advantage when the final reward is 0 (App. C, Fig. 16).
- **Forgetting.** Per-site curves for Llama-3.1-8B rise on all sites except Map, which rises then declines while OSS and CMS improve; GitLab also declines slightly (App. C, Fig. 12).
- **Generality.** Evaluation is WebArena-Lite only; no general-capability benchmarks are reported. Four seeds show an upward trend with phase-level fluctuation (App. C, Fig. 13).

## Connections
- [[agenttuning]]: same group; AgentTuning is SFT-only agent training, WebRL adds online RL.
- [[evol-instruct]]: in-breadth evolution is the task generator (§2.1).
- [[ppo]], [[dpo]]: App. A.1, Table 4 compares the WebRL loss with PPO (on-policy) and DPO (needs pairwise data, which needs page-state backtracking).
- [[agentic-finetuning-misalignment]]: uses the WebRL dataset of Qi et al. as its web SFT data.
- [[slime]], [[glm-4-5]]: later GLM agentic RL infrastructure.

## Verification
- Checked on 2026-11-24 against https://arxiv.org/abs/2411.02337 (v3, 27 Jan 2025; v1 4 Nov 2024, v2 3 Dec 2024 not compared). PDF read in full via text extraction; Table 3 orientation confirmed from the layout-preserving extraction.
- Not reported by the source: the β used in the final runs; ORM base model name (Table 3 says only "8B"); compute and wall-clock cost; number of rollouts per instruction per phase.
