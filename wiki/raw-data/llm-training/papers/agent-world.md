<!-- scope: Agent-World (Renmin University of China, ByteDance Seed): web-mined executable tool environments (1,978 environments, 19,822 tools), verifiable task synthesis, multi-environment GRPO, and a diagnosis-driven self-evolving arena; environment-count and evolution-round scaling
     deps: [[grpo]], [[dapo]]
     see-also: [[toucan-mcp]], [[agentscaler]], [[tau2-bench]], [[bfcl]], [[are-gaia2]]
-->

# Agent-World: Scaling Real-World Environment Synthesis for Evolving General Agent Intelligence
- **Core Insight:** After cold-start SFT and multi-environment GRPO on web-mined executable environments, Agent-World-8B (Qwen3-8B backbone) scores 61.8 on τ²-Bench, 51.4 on BFCL V4, and 8.9 on MCP-Mark, above every listed 7B-14B environment-scaling baseline on these three averages (Table 1); in the environment-scaling analysis the average over four subdomains rises from 18.4% with 0 training environments to 38.5% with 1,978 (§4.3.3, Fig. 8).
- **Guideline:** When training a tool-use agent with RL for breadth across unseen tool ecosystems, increase the number of distinct executable environments (database plus unit-tested tools), because in this paper the four-domain average rose from 18.4% to 38.5% between 0 and 1,978 environments, with the authors describing jumps at 10 → 100 and 100 → 500 and smaller positive gains after 500 (§4.3.3); add diagnosis-driven rounds of targeted task synthesis when held-out arena failures concentrate in specific environments, because two rounds raised MCP-Mark (Postgres) by 8.6 points for Agent-World-14B and 5.6 for EnvScaler-8B (Table 2, §4.3.4).
- **Authors:** Guanting Dong, Junting Lu, Junjie Huang, Wanjun Zhong, Longxiang Liu, Shijue Huang, et al. (Renmin University of China; ByteDance Seed)
- **Year:** 2026 (arXiv v1 2026-04; arXiv comment "Working in progress")
- **URL:** https://arxiv.org/abs/2604.18292
- **Source type:** paper
- **Relevant topics:** environment synthesis, MCP tool environments, verifiable task synthesis, rubric and code-verifier rewards, multi-environment agentic RL, self-evolving curriculum, environment scaling, agent generalization

## Abstract
General-purpose agents must act in external, stateful tool environments, and MCP and agent skills give a common interface to such services, but training is limited by the lack of realistic environments and of mechanisms for continued learning. Agent-World is a self-evolving training arena with two parts. Agentic Environment-Task Discovery explores topic-aligned databases and executable tool ecosystems from thousands of real-world environment themes and synthesizes verifiable tasks with controllable difficulty. Continuous Self-Evolving Agent Training combines multi-environment RL with an arena that finds capability gaps through dynamic task synthesis and drives targeted training, so policies and environments evolve together. Across 23 agent benchmarks, the authors report that Agent-World-8B and 14B outperform proprietary models and environment-scaling baselines, and they analyze scaling with environment diversity and self-evolution rounds.

## Key Contributions
- A pipeline that mines databases from the web with a deep-research agent, generates and unit-tests tools with a coding agent, and organizes environments in a 3-tier taxonomy (§3.1, Figs. 2-3).
- Two verifiable task generators: tool-graph random walks with rubrics, and programmatic tasks with Python solution and verification scripts; both filtered by repeated ReAct execution (§3.1.1).
- Multi-environment GRPO with rubric-judge rewards (graph tasks) and sandbox verification-script rewards (programmatic tasks) (§3.2.1).
- A self-evolving arena: fresh tasks in held-out environments each round, an LLM diagnosis agent that ranks weak environments and writes task-generation guidelines, then targeted synthesis and continued RL (§3.2.2, Alg. 1, App. A).
- Evaluation on 23 benchmarks plus scaling analyses over environment count and evolution rounds (§4).

## Key Figures/Tables to Study
- **Fig. 2:** environment-task discovery pipeline, with source counts for MCP servers, tool documentation, and PRDs.
- **Fig. 4:** environment, tool, file-type, turn-count, and Pass@10 difficulty statistics.
- **Fig. 5 and Alg. 1:** RL rollout module and the evaluate → diagnose → synthesize → continue-RL loop.
- **Table 1:** MCP-Mark, BFCL V4, τ²-Bench subdomain scores for proprietary, open, and environment-scaling models.
- **Fig. 8 and Table 2:** environment-count scaling and per-round self-evolution gains.
- **Fig. 9:** training reward and actor entropy for the 8B and 14B backbones.

## Technical Details
- **Formalism:** POMDP (U, S, A, O, P) following AgentSkiller; state = environment state × dialogue state; each environment e = (D, F) with database D and toolset F; actions are tool calls or language responses; the environment state is observed only through tool outputs (§2).
- **Theme sources:** MCP server specifications from Smithery, tool definitions from open-source tool-use datasets mapped to topics by an LLM, and industrial product requirement documents (§3.1). Fig. 2 labels the sources as about 2.8K MCP servers, 0.5K tool documentations, and 0.2K PRDs (Fig. 2).
- **Database mining:** a deep-research agent (search, browser, code compiler, OS tools) builds a topic database, then a complexification step φ expands it for N rounds; N is not reported (§3.1).
- **Tool filter:** a tool is kept only if it compiles, passes more than 50% of its generated unit tests (Acc > 0.5), and its environment has at least one valid tool and one valid test case (§3.1).
- **Taxonomy:** hierarchical clustering to 50 centers; GPT-OSS-120B names 50 second-tier labels; three annotators merge them into 20 first-tier types; over 2K third-tier labels (§3.1, Fig. 3).
- **Scale and difficulty:** over 2,000 environments, 1,978 retained; more than 10 tools per environment on average, some over 40; 19,822 tools; database files include json, csv, sql, html, tex, yaml; every task has at least 7 interaction turns, average over 20; under Pass@10 with Doubao-Seed-2.0-pro most tasks are solved once in 10 attempts and some are never solved (§3.1.1(3), Fig. 4).
- **Graph-based tasks:** LLM-assigned edge weights: strong dependency 3, weak 2, independent 1 (keeps the graph connected); weighted random walk; parameters passed from prior outputs or sampled from D; task text may not contain tool names or schema; sandbox execution yields a JSON answer and rubrics; kept if a ReAct agent reaches a consistent answer in at least 2 of 5 runs (§3.1.1(1)).
- **Programmatic tasks:** LLM writes a query and an executable Python solution with loops, branches, and aggregation, repaired in a ReAct loop; a verification script checks the answer and database state; kept if at least 2 of 5 ReAct runs pass (§3.1.1(2)).
- **Difficulty scaling:** longer random walks, more weak and independent edges, more tools and invocations, cross-database aggregation, and rewriting descriptions to remove API references (§3.1.1).
- **Reward:** graph tasks use a rubric-conditioned LLM judge and the prose defines the reward as the average of criterion pass indicators (the printed equation wraps this average in an indicator I[·] without stating a threshold); programmatic tasks use I[verification script passes] (§3.2.1).
- **Policy update:** GRPO objective with clipped ratio and a KL penalty β·D_KL(π_θ‖π_ref) (§3.2.1, Eq. 1); tasks in a global batch are paired with independent environments (§3.2.1).
- **Arena:** K = 5 environments sampled per first-tier category (20 categories, so 100 environments; derived); fresh graph and programmatic tasks each round; the diagnosis agent receives failure traces, error distributions by environment and category, and tool schemas, and outputs ranked weak environments and task-generation guidelines (§3.2.2, App. A).
- **Main results (Table 1, averages):** Agent-World-8B 8.9 / 51.4 / 61.8 (MCP-Mark / BFCL V4 / τ²-Bench); Agent-World-14B 13.3 / 55.8 / 65.4; Qwen3-8B 2.4 / 40.4 / 26.2; Qwen3-14B 3.4 / 41.0 / 32.4; EnvScaler-8B 5.6 / 47.6 / 37.9; AWM-14B 5.1 / 42.4 / 39.0; DeepSeek-V3.2-685B 36.7 / 54.1 / 80.3; GPT-5.2 High 53.1 / 62.9 / 80.2.

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Agent-World-8B / 14B | 8B, 14B | distill-SFT | cold-start data | 40K trajectories from an in-house Doubao-Seed-1.8 policy, using the environment-task discovery synthesis strategy | arXiv:2604.18292v1 §4.1 | verified 2026-09-14 | no ablation reported |
| Agent-World-8B / 14B | 8B, 14B | SFT | SFT LR, epochs, batch, sequence length | not reported (checked §4.1, §4.4, App. A-C) | — | not reported | none |
| Agent-World-8B / 14B | 8B, 14B | RL | backbone; RL samples; algorithm | Qwen3-8B / Qwen3-14B; 5K synthesized RL samples; GRPO | §4.1 | verified 2026-09-14 | no ablation reported |
| Agent-World-8B / 14B | 8B, 14B | RL | clip ε_low; ε_high | 0.2; 0.28 ("following prior work [118]", DAPO) | §4.1 | verified 2026-09-14 | no ablation reported |
| Agent-World-8B / 14B | 8B, 14B | RL | max trajectory length; max generation per step | 80K tokens; 32k tokens | §4.1 | verified 2026-09-14 | no ablation reported |
| Agent-World-8B / 14B | 8B, 14B | RL | tasks per step; rollouts per task; temperature; top_p | 32; 8; 1.0; 1.0 | §4.1 | verified 2026-09-14 | no ablation reported |
| Agent-World-8B / 14B | 8B, 14B | RL | KL coefficient β; LR; total steps; hardware | not reported (Fig. 9 x-axis spans 0-300 steps) | §3.2.1, §4.1, Fig. 9 | not reported | none |
| Agent-World-14B, EnvScaler-8B | 14B, 8B | RL | self-evolution rounds; arena size | 2 rounds; K = 5 environments per first-tier category | §3.2.2, §4.3.4 | verified 2026-09-14 | Table 2: round 1 and round 2 gains |
| GPT-OSS-120B | 120B | data generation | model for environment mining, task synthesis, code/rubric generation, diagnosis | GPT-OSS-120B | §4.1 | verified 2026-09-14 | not applicable |
| all evaluated models | — | eval-gate | decoding; repeats | temperature 1.0, top_p 1.0; 8 runs averaged; sampled subsets for GAIA and HLE; in-house framework aligned to official scores | §4.1 | verified 2026-09-14 | not applicable |

## Findings relevant to generality, negative feedback, agentic training, distillation
- **Environment-count scaling (Fig. 8):** 0, 10, 100, 500, 1000, 2000 (1,978) environments; four-domain average 18.4% → 38.5%; MCP-Mark (Postgres) 4.8% → 19.9%; BFCL (WebSearch) 7.0% → 47.0%; returns decrease beyond 500 (§4.3.3). The model size for this run is not stated. Result (single study).
- **Self-evolution (Table 2, τ²-Bench / BFCL-V4 / MCP-Mark Postgres):** Agent-World-14B 60.2 / 52.4 / 29.5 → 63.5 / 54.9 / 36.3 → 65.4 / 55.8 / 38.1; EnvScaler-8B 37.9 / 47.6 / 9.5 → 40.2 / 49.1 / 13.9 → 41.6 / 50.0 / 15.1; round-2 gains are smaller. The prose reports Agent-World-14B τ²-Bench as 45.3 → 50.5, which does not match Table 2 (§4.3.4). The round-2 values equal the Agent-World-14B row of Table 1.
- **Transfer to unseen benchmarks:** Agent-World-8B reaches 9.2 / 6.5 / 30.5 on SkillsBench / ARC-AGI-2 / Claw-Eval and 14B reaches 12.6 / 8.5 / 31.5, while Qwen3 drops on Claw-Eval from 8B to 14B (25.6 → 24.7) (§4.3.2, Fig. 7). On 17 reasoning, search/coding, and knowledge/MCP benchmarks the authors report gains over Qwen3-8B and EnvScaler-8B and "no degradation on core math reasoning"; per-benchmark values are only in Fig. 6 (§4.3.1). EnvScaler-8B is below its Qwen3-8B backbone on SWE-bench Verified and Terminal-Bench 1.0 (§4.3.1).
- **Claim vs table:** the abstract says Agent-World outperforms strong proprietary models, but in Table 1 GPT-5.2 High, Claude Sonnet-4.5, Gemini-3 Pro, and Seed 2.0 score higher than Agent-World-14B on all three suite averages; §4.2 compares against environment-scaling baselines and DeepSeek-V3.2-685B on BFCL V4 (55.8 vs 54.1).
- **Simulated vs executable environments:** the authors state that Simulator-8B achieves good results on τ²-Bench but performs poorly on MCP-Mark and BFCL V4 (Table 1 averages: 31.8, 2.4, 23.9), and interpret this as simulated environments not capturing real state transitions (Interpretation, §4.2).
- **Failures as data-generation signal:** failed arena traces go to the diagnosis agent and are converted into targeted new tasks; they are not used as a negative gradient (§3.2.2). Synthesized tasks are kept only if at least 2 of 5 ReAct runs succeed (§3.1.1).
- **Entropy during RL:** actor entropy rises over training for both backbones (Fig. 9b), which the authors interpret as continued exploration (§4.4).
- **Cold-start distillation:** teacher Doubao-Seed-1.8, 40K trajectories; teacher sampling settings and trajectory filtering are not reported (§4.1).

## Connections
- [[grpo]], [[dapo]]: policy update and the clip-higher setting (ε_high 0.28) (§3.2.1, §4.1).
- [[toucan-mcp]]: its taxonomy is the starting point for Agent-World's, its reverse-engineered task synthesis is cited, and TOUCAN-7B is a baseline (§3.1, §4.1).
- [[tau2-bench]], [[bfcl]], [[terminal-bench-2]]: evaluation suites (§4.1).
- [[are-gaia2]]: cited environment platform with asynchronous dynamics (§5.1).
- [[agentscaler]]: another environment-scaling approach for tool agents; not cited by this paper.
- [[qwen-agentworld]]: different artifact with a similar name (language world models from Qwen).
- [[clip-low-clip-high-entropy]]: analysis of how the two clip bounds change entropy, relevant to Fig. 9b.
- [[agent-early-experience]]: reward-free alternative for environments that lack verifiers.

## Verification
- Created on 2026-09-14 from https://arxiv.org/abs/2604.18292 (v1, 2026-04-20; only version).
- Audit claims not found in the source: "the 2026 successor to AgentScaler and DeepSeek-V3.2's environment loop" (AgentScaler is not cited; DeepSeek-V3.2-685B appears only as a baseline, §4.1). Other audit claims (two components, 23 benchmarks, scaling with environment diversity and evolution rounds, work in progress) are in the source.
- Not reported by the source: SFT hyperparameters, RL learning rate, KL coefficient, number of RL steps, compute, complexification rounds N, and the model size used in the environment-scaling run.
