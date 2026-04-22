<!-- scope: agentic trajectory synthesis — AgentLM + AgentInstruct corpus of 1.8K agent trajectories
     deps: [[fireact]]
     see-also: [[agentinstruct]], [[lumos]], [[agent-flan]]
-->

# AgentTuning: Enabling Generalized Agent Abilities for LLMs
- **Core Insight:** Agent capability transfers if you fine-tune on a relatively small but diverse set of high-quality agent trajectories (~1,866 across 6 environments) interleaved with general instruction data — the agent-data-to-general-data mixture ratio is the critical hyperparameter, not raw agent data volume.
- **Guideline:** For agent-SFT, assemble ~2K high-quality multi-turn trajectories across several environments (web, OS, DB, games) and mix them with 10× volume of general instruction data; the resulting model inherits both agent skills and chat quality.
- **Authors:** Aohan Zeng, Mingdao Liu, Rui Lu, Bowen Wang, Xiao Liu, Yuxiao Dong, Jie Tang (Tsinghua + Zhipu AI)
- **Year:** 2023
- **URL:** https://arxiv.org/abs/2310.12823
- **Relevant topics:** agent SFT, trajectory synthesis, AgentBench, AgentLM

## Abstract
AgentTuning introduces **AgentInstruct** (this paper's version — distinct from MS AgentInstruct), a dataset of 1,866 verified agent trajectories across six environments: ALFWorld, WebShop, Mind2Web, Knowledge Graph, OS bash, Database SQL. The trajectories are generated via GPT-4 + task-specific filters. Mixing them with general ShareGPT data and fine-tuning Llama-2 produces AgentLM-7B/13B/70B, which substantially outperforms base Llama-2-Chat on AgentBench while preserving general capabilities.

## Key Contributions
- **1,866-trajectory AgentInstruct corpus** (first public agent-SFT set at this scale).
- **Mixing recipe:** agent-data : general-data = 1 : 10 ratio is empirically optimal.
- AgentLM-70B closes much of the gap to GPT-3.5 on AgentBench.
- Evidence that agent fine-tuning does **not** harm general ability when properly mixed.

## Synthesis pipeline (REQUIRED — concrete, modality-specific)
- **Seed environments (6 total):**
  1. **ALFWorld** — household tasks in text simulator.
  2. **WebShop** — simulated e-commerce navigation.
  3. **Mind2Web** — real-website web tasks.
  4. **Knowledge Graph** — KG query / editing.
  5. **OS / bash** — shell commands.
  6. **Database / SQL** — DB queries.
- **Trajectory generation:**
  - Each environment runs GPT-4 in ReAct mode; GPT-4 produces `Thought: … ; Action: … ; Observation: …` trajectories.
  - Environment provides real observations (not simulated) by executing actions in the actual env.
- **Filtering:**
  - **Task-level reward:** only keep trajectories that solve the task (0/1).
  - **Trajectory-length filter:** reject abnormally long (>30 steps) trajectories.
  - **LLM-judge quality pass** on reasoning coherence.
- **Mixing:**
  - AgentInstruct + ShareGPT (general instructions) at 1:10 ratio by examples.
  - ShareGPT acts as a "ballast" preserving chat quality and preventing over-specialization.
- **Output shape:** 1,866 multi-turn trajectories; avg 8–15 turns per trajectory; total ~5M training tokens.
- **Teacher model:** GPT-4 (Oct-2023 version).
- **Cost:** ~$20K in GPT-4 API; days to weeks of env simulation.

## Modality-specific technical details (REQUIRED — agentic)
- **Environment:** six distinct sims; actions range from text commands (ALFWorld) to clicks (Mind2Web) to SQL queries.
- **Action space:** environment-specific; unified with ReAct `Thought / Action / Observation` wrapper at the prompt level.
- **Trajectory length:** avg 12 turns; min 3 (trivial tasks), max 30 (complex web navigation).
- **Success criterion:** environment's native reward (0/1 task-complete); trajectory filter keeps only successes.
- **Data scale:** 1,866 trajectories × avg 12 turns ≈ 22K (instruction, response) turn-level pairs.
- **Mixing ratio ablation:**
  - 1:1 AgentInstruct:ShareGPT → agent gains but 5-point drop in general quality (MT-Bench).
  - 1:10 → agent gains preserved, general quality preserved.
  - 1:50 → agent gains halved.

## Quality / diversity evaluation
- AgentBench overall score: AgentLM-70B 4.02 vs Llama-2-Chat-70B 1.58 — nearly 3× improvement.
- MMLU / MT-Bench / BBH preserved within 1 point of base Llama-2-Chat.
- Held-out environments (not in training) also improved — evidence of genuine agent-skill transfer.

## Risks + gotchas
- **Simulated envs differ from production:** ALFWorld and WebShop are stylized; transfer to real websites is partial.
- **GPT-4 teacher cap:** the dataset can only be as good as GPT-4's agent performance in 2023, which is well below 2025 frontier.
- **Narrow env coverage:** no code-repo agent (like SWE-Bench) or multi-agent task; later work (SWE-Gym, OpenHands) fills these.

## Connections
- Ancestor of [[agent-flan]] (scaled AgentFlan at millions of samples).
- Sibling trajectory-SFT: [[fireact]] (multi-task ReAct fine-tuning).
- Evaluation: AgentBench (same lab).
- Superseded for SWE by [[swe-gym]]; for multi-turn tool calling by [[apigen-mt]].
