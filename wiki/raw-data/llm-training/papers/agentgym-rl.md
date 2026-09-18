<!-- scope: AgentGym-RL — modular multi-turn RL framework (AgentGym + veRL) that trains Qwen2.5 instruct models as agents with outcome rewards and no agent SFT, across 5 scenarios; ScalingInter-RL staged interaction-turn cap
     deps: [[grpo]]
     see-also: [[agentgym-rl-recipe]], [[search-r1]], [[ragen-starpo]], [[skyrl-agent]], [[verl-grpo]], [[agenttuning]]
-->

# AgentGym-RL: Training LLM Agents for Long-Horizon Decision Making through Multi-Turn Reinforcement Learning
- **Core Insight:** Qwen2.5-7B-Instruct trained with GRPO and a staged cap on interaction turns (ScalingInter-7B) scored 26.00 on WebArena, 38.25 on Deep Search, 91.00 on TextCraft, 96.67 on BabyAI and 57.00 on SciWorld, versus 9.76, 18.75, 42.00, 66.67 and 1.50 for the untrained instruct model (Tables 1-5).
- **Guideline:** When training a multi-turn agent with outcome-only RL, start with a small maximum number of interaction turns and raise it in stages, because in the Deep Search environment a fixed cap of 10 turns gave higher early reward and then collapsed, a fixed cap of 5 plateaued, and the staged schedule reached the highest reward (§4.2, Fig. 7). This comparison is shown for one environment; seeds and variance are not reported.
- **Authors:** Zhiheng Xi, Jixuan Huang, Chenyang Liao, Baodai Huang, Honglin Guo, Jiaqi Liu, et al. (Fudan University, ByteDance Seed, Shanghai Innovation Institute)
- **Year:** 2025 (arXiv v1 2025-09; ICLR 2026 Oral according to the repository README)
- **URL:** https://arxiv.org/abs/2509.08755 (code: github.com/WooooDyy/AgentGym-RL)
- **Source type:** paper
- **Relevant topics:** multi-turn agent RL, horizon curriculum, GRPO vs REINFORCE++, web navigation, deep search, text games, embodied tasks, test-time interaction scaling, pass@k

## Abstract
The authors state that no unified interactive RL framework trains LLM agents without a preceding SFT stage across diverse, realistic environments. AgentGym-RL is a framework with decoupled environment, agent, and training modules that supports mainstream RL algorithms. ScalingInter-RL restricts the number of agent-environment interactions early in training (exploitation) and increases it later (exploration), which the authors report yields more diverse behavior and less collapse over long horizons. The trained agents match or surpass commercial models on 27 tasks across the environments (Abstract).

## Key Contributions
- A framework built on AgentGym and veRL: environments run as HTTP services with `/observation`, `/available_actions`, `/step`, `/reset`; many isolated environment clients roll out in parallel (§3.1, App. A, Fig. 3).
- Algorithm support: PPO, GRPO, RLOO, REINFORCE++ online; SFT, DPO, and AgentEvol (rejection sampling on successful self-generated trajectories) as other paradigms (§3.2.2).
- Engineering fixes: subprocess-based multi-Chromium WebArena server, full-reset interface for WebArena, parallel SciWorld instantiation, memory-leak fixes in TextCraft's crafting tree and SciWorld's clock (§3.2.3).
- ScalingInter-RL: a monotonic schedule h1 < h2 < ... < hn of maximum interaction turns, updated every Δ training steps (§3.3, Fig. 5).
- Evaluation of 3B and 7B RL agents against proprietary and open models in five scenarios, plus test-time scaling and algorithm comparisons (§4-§5).

## Key Figures/Tables to Study
- **Figure 7** — Deep Search reward and turn cap on an x-axis of 0-350 training steps: max turns 10 vs 5 vs ScalingInter-RL.
- **Tables 1-5** — per-benchmark results (WebArena, Deep Search, TextCraft, BabyAI, SciWorld).
- **Table 6** — GRPO vs REINFORCE++ at 3B and 7B.
- **Figures 8-9** — accuracy vs test-time interaction turns, and pass@K for K up to 64.
- **Appendix B** — per-environment data splits, turn caps, learning rates, KL coefficient, samples per query.

## Technical Details
**Formulation (§2.1-2.2, §3.3).** Tasks are POMDPs (U, S, A, O, T, r). After N turns the environment returns an outcome reward r(τ) ∈ [0, 1]. The objective is J(θ) = E_{τ∼πθ}[r(τ)]. ScalingInter-RL samples τ_t ∼ πθ(τ | h_t) subject to K_t ≤ h_t, where K_t is the number of turns and h_t is the phase-t cap; h_{t+1} = h_t + δh every Δ steps (§3.3). The authors set phase transition points from the total optimization steps rather than tuning them (§4.2). The paper gives no numeric h_t or Δ; released scripts do (see [[agentgym-rl-recipe]]).

**Environments and splits (App. B).**
- WebArena: 372 training and 50 test queries selected from 812; Content & Config (state-changing) tasks excluded for parallel rollout; max 15 turns (B.1).
- Deep Search: queries from NQ, TriviaQA, PopQA, HotpotQA, 2Wiki, Musique, Bamboogle following Search-R1; 400 examples sampled from development sets; max 4 turns (B.2).
- TextCraft: split by crafting-tree depth 1-4; max 20 turns (B.3). BabyAI: six subsets by goal; max 20 turns (B.4). SciWorld: 8 task subsets; max 20 turns (B.5).
- Hardware: NVIDIA A100 GPUs and Ascend 910B NPUs (App. B).

**Main results (7B unless noted).**
- WebArena overall: ScalingInter-7B 26.00, AgentGym-RL-7B 22.00, AgentGym-RL-3B 18.00; GPT-4o 16.00, Gemini-2.5-Pro 28.00, OpenAI o3 34.00, o4-mini 36.00 (Table 1).
- Deep Search overall: ScalingInter-7B 38.25, AgentGym-RL-7B 34.00; SearchR1-it-7B-v0.3 25.00; GPT-4o 26.75; o3 49.50; DeepSeek-R1-0528 40.25 (Table 2).
- TextCraft overall: ScalingInter-7B 91.00 (Depth 4: 33.33), AgentGym-RL-7B 89.00, AgentGym-RL-3B 75.00 vs Qwen2.5-3B-Instruct 14.00; Gemini-2.5-Pro 94.00 (Table 3).
- BabyAI overall: ScalingInter-7B 96.67, AgentGym-RL-3B 93.33, AgentGym-RL-7B 92.22; o3 94.44 (Table 4).
- SciWorld overall: ScalingInter-7B 57.00, AgentGym-RL-7B 50.50; o3 41.50 (Table 5).
- Average improvement of 33.65 points for open models trained with the framework (§1). Figure 1 (right): ScalingInter-7B average about 58.6% vs Llama3.1-70B about 47% and Qwen2.5-72B about 43% (§4.2).
- The authors report larger RL gains in rule-based simulated environments (TextCraft, BabyAI, SciWorld) than in open-ended ones (WebArena, Deep Search) (§4.2; Interpretation).

**Algorithm comparison (Table 6).** GRPO vs REINFORCE++: 3B TextCraft 75.00 vs 28.00, BabyAI 93.33 vs 70.00, SearchQA 25.75 vs 13.25; 7B 83.00 vs 73.00, 92.22 vs 84.44, 34.00 vs 24.00. The 3B GRPO model exceeds the 7B REINFORCE++ model on all three (§5.2). The authors attribute the gap to high-variance full-episode Monte Carlo returns in REINFORCE++ (§5.2; Interpretation).

**Test-time scaling (§5.1).** Accuracy rises with the allowed number of interaction turns for all tested models (Fig. 8). At K = 64 samples, the RL model is 5.5% above the untrained base in Deep Search and 7.05% above in SciWorld on pass@K (Fig. 9).

## Recipe ledger
Paper values (App. B) and released-script values (repository `examples/train/`) are separate rows in [[agentgym-rl-recipe]]. Summary of paper values: GRPO, KL coefficient 1e-3, temperature 1.0, policy LR 1e-6 (5e-7 for WebArena), 8 trajectories per query (4 for WebArena), SFT-baseline LR 1e-4 (B.1-B.5). Batch size, epochs, and step counts are not given in the paper.

## Findings relevant to generality and agentic training
- **In-domain only:** the authors state that the trained agents "perform well within in-domain settings" and list transfer to novel environments and unfamiliar tools as future work (§7). Settings are per environment (App. B), and no evaluation on held-out environments or on general non-agent benchmarks is reported.
- **"Without SFT" scope:** backbones are Qwen2.5-3B/7B instruct models (Table 6 row labels; released scripts load Qwen2.5-7B-Instruct), so the claim means no agent-trajectory SFT stage, not training from a base model.
- **Horizon and collapse:** Figure 7's caption attributes collapse at a 10-turn cap to high variance, credit-assignment difficulty, and overfitting to spurious behaviors (Interpretation; no ablation isolates these causes).
- **Remaining failure modes:** case studies show RL agents substituting factual recall for the required SciWorld procedure, stopping exploration early, and producing redundant clicks, hovers, and scrolls on WebArena (§5.3, Fig. 15-16).
- **Coverage:** pass@64 stays above the base model in the two environments shown (Fig. 9); no environment where the base model overtakes at large K is reported.

## Connections
- [[agentgym-rl-recipe]] — paper and released-script training settings.
- [[grpo]], [[rloo]], [[verl-grpo]] — algorithms and the veRL trainer the framework builds on (§3.2.2, §3.2.4).
- [[search-r1]] — source of the Deep Search setup and the SearchR1-it baseline (B.2, Table 2).
- [[webarena-data]] — WebArena environment used for web navigation (B.1).
- [[ragen-starpo]], [[skyrl-agent]], [[loop-appworld]], [[gigpo-verl-agent]] — other multi-turn agent RL systems.
- [[agenttuning]], [[fireact]], [[agent-flan]] — trajectory-imitation approaches the paper contrasts with RL (§6).
- [[agentic-benchmark-checklist]] — reports 1.4-5.2% overestimation from WebArena string matching and LLM judging; its effect on this paper's WebArena subset is not measured.
- [[rlvr-beyond-base-model]] — pass@k analysis of RL gains, relevant to Figure 9.

## Verification
- Created on 2026-09-14 from https://arxiv.org/abs/2509.08755 (arXiv v1, 2025-09-10; latest version at fetch). Released scripts read from github.com/WooooDyy/AgentGym-RL main branch on 2026-09-14 (no commit pinned).
- Audit claims not found in the source: "ScalingInter-RL schedule [10, 20, 30] increasing every 100 steps" is not in the paper; it is the README example and matches only the TextCraft and SciWorld scripts (WebArena [8,12,15]/80, Deep Search [5,8,10]/100, BabyAI [6,13,20]/100); "matches or surpasses GPT-4o across the 27 tasks" (the abstract says "commercial models on 27 tasks"; the 27 tasks are not enumerated).
- Internal inconsistencies in v1: §4.3 says AgentGym-RL-7B scored 16.00% on WebArena, Table 1 prints 22.00; §4.3 says every model scored zero on Chem-Mix, Table 5 prints nonzero values for some baselines (e.g., o3 40.00); Table 6 7B GRPO TextCraft 83.00 vs Table 3 AgentGym-RL-7B 89.00.
- Not reported by the source: seeds or variance, training prompt counts for Deep Search/TextCraft/BabyAI/SciWorld, batch sizes, number of training steps, compute hours, results for PPO and RLOO.
