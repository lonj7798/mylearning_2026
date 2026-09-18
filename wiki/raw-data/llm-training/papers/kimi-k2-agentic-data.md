<!-- scope: the tool-use data synthesis pipeline of the Kimi K2 technical report (arXiv:2507.20534 §3.1.1) — tool spec generation, agent and rubric-based task generation, simulated multi-turn trajectories, LLM-judge filtering, and real execution sandboxes
     deps: [[kimi-k2]]
     see-also: [[kimi-k2-recipe]], [[agentinstruct]], [[toolllm]], [[apigen-mt]], [[swe-gym]], [[glm-4-5]]
-->

# Kimi K2: Open Agentic Intelligence — §3.1.1 Large-Scale Agentic Data Synthesis for Tool Use Learning

- **Core Insight:** Kimi K2's tool-use ability is installed in **supervised fine-tuning**, not in pre-training: the report builds a tool repository of 3,000+ real MCP tools fetched from GitHub plus over 20,000 synthetic tools produced by hierarchical domain evolution, generates thousands of agents and rubric-paired tasks over that repository, and rolls out multi-turn trajectories against a stateful tool simulator that an LLM judge filters against the task rubric (§3.1.1).
- **Guideline:** When simulated tool responses cannot be trusted for a domain, keep the simulated pipeline for coverage but add real execution sandboxes for that domain, because the report uses real sandboxes with test-suite pass rates as ground-truth feedback for coding and software engineering while keeping simulation for the remaining domains (§3.1.1).
- **Authors:** Kimi Team (Moonshot AI). App. A lists contributors alphabetically by last name: Yifan Bai, Yiping Bao, Y. Charles, Cheng Chen, Guanduo Chen, Haiting Chen, et al.
- **Year:** 2025 (arXiv v1 2025-07; v2 2026-02)
- **URL:** https://arxiv.org/abs/2507.20534
- **Source type:** official technical report
- **Relevant topics:** agentic SFT data synthesis, MCP tool specs, synthetic tool evolution, rubric-based task generation, user simulation, tool simulator, LLM-judge filtering, rejection sampling, hybrid simulated/real environments

## Summary
This card extracts §3.1.1 of the Kimi K2 technical report. The section states the problem as follows: real environments give authentic interaction signals but are hard to construct at scale because of cost, complexity, privacy, and accessibility constraints (§3.1.1). The pipeline has three stages (§3.1.1, Figure 8): tool spec generation, agent and task generation, and trajectory generation. It is described as inspired by ACEBench's data synthesis framework and as building on AgentInstruct, Self-Instruct, StableToolBench, and ZeroSearch (§3.1.1). The section reports that the resulting data, used for supervised fine-tuning, improved tool-use capability, but gives no ablation number for the pipeline itself (§3.1.1). The whole-report card is [[kimi-k2]].

## Key Contributions
- A two-source tool repository: 3,000+ real MCP (Model Context Protocol) tool specs fetched from GitHub, and over 20,000 synthetic tools evolved through a hierarchical domain generation process (§3.1.1).
- Agent diversification: thousands of distinct agents built by synthesizing system prompts and pairing them with different tool combinations from the repository (§3.1.1).
- Rubric-based task generation: each task carries an explicit rubric naming success criteria, expected tool-use patterns, and evaluation checkpoints (§3.1.1).
- A multi-turn trajectory generator combining LLM-generated user personas with a stateful tool simulator that introduces controlled stochasticity (§3.1.1).
- A hybrid design: simulated environments for coverage plus real execution sandboxes for coding and software engineering, where correctness is measured by test-suite pass rates (§3.1.1).
- The authors characterize the LLM-judge filtering step as large-scale rejection sampling (§3.1.1).

## Key Figures/Tables to Study
- **Figure 8:** the two-part pipeline diagram — (a) tool specs → tool repository → agents → tasks with rubrics; (b) user agent, agent, tool simulator, trajectories, judge agent, filtered data.
- **Figure 9:** t-SNE of tool embeddings — (a) real MCP tools clustered by source category, (b) synthetic tools by pre-defined domain category, used to argue the two sets cover complementary regions.
- **Table 3:** the agentic benchmark results of Kimi-K2-Instruct that this data stage feeds into (Tau2-Bench 66.1, ACEBench En 76.5, SWE-bench Verified 65.8 agentic single attempt).

## Technical Details
**Stage 1 — tool spec generation.** Real tools: "3000+ real MCP (Model Context Protocol) tools from GitHub repositories" (§3.1.1). Synthetic tools: start from key categories, named in the text as financial trading, software applications, and robot control; evolve multiple specific application domains inside each category; then synthesize specialized tools per domain with interfaces, descriptions, and operational semantics. This produces "over 20,000 synthetic tools" (§3.1.1). The evolution step cites WizardLM (§3.1.1, ref. [83]).

**Stage 2 — agent and task generation.** Agents are produced by synthesizing system prompts and equipping each with a different combination of tools sampled from the repository; the report says "thousands of distinct agents" and gives no exact count (§3.1.1). Tasks per agent configuration range from simple to complex operations, and each is paired with a rubric (§3.1.1). No task count is reported.

**Stage 3 — trajectory generation.** Two components are named (§3.1.1):
1. *User simulation* — LLM-generated user personas with distinct communication styles and preferences hold multi-turn dialogues with the agent.
2. *Tool execution environment* — a tool simulator that the report calls "functionally equivalent to a world model". It executes tool calls, maintains and updates state after each execution so multi-step interactions have persistent effects, and introduces controlled stochasticity so outcomes include successes, partial failures, and edge cases.

**Filtering.** An LLM-based judge evaluates each trajectory against the task rubric, and only trajectories meeting the success criteria are retained (§3.1.1). The retention rate is not reported.

**Real execution environments.** The report states the limitation of simulation fidelity directly and complements simulated environments with real execution sandboxes for coding and software engineering, which run actual code and return objective metrics such as test suite pass rates (§3.1.1). The split between simulated and sandbox data is not reported.

**Scale.** The section says the pipeline enables "the generation of tens of thousands of diverse and high-quality training examples" (§3.1.1). No token count is given for this data, and no percentage of any training mixture is given.

**Placement in the pipeline.** §3.1.1 sits inside §3.1 Supervised Fine-Tuning, which is inside §3 Post-Training. Pre-training (§2.2) is 15.5T tokens of web text, code, mathematics, and knowledge, and does not include this data (§2.2, §3.1).

## Recipe ledger
Only pipeline-scale quantities are disclosed in §3.1.1; optimizer and schedule values for the SFT and RL stages are in [[kimi-k2-recipe]].

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Kimi-K2-Instruct | 1.04T total / 32.6B activated | SFT | real tool specs in repository | 3000+ MCP tools from GitHub | arXiv:2507.20534v2 §3.1.1 "Domain Evolution and Tool Generation" | verified 2026-09-18 | no ablation reported |
| Kimi-K2-Instruct | 1.04T total / 32.6B activated | SFT | synthetic tool specs in repository | over 20,000 | arXiv:2507.20534v2 §3.1.1 "Domain Evolution and Tool Generation" | verified 2026-09-18 | no ablation reported |
| Kimi-K2-Instruct | 1.04T total / 32.6B activated | SFT | distinct agents (system prompt + toolset) | "thousands" (no exact count) | arXiv:2507.20534v2 §3.1.1 "Agent Diversification" | verified 2026-09-18 | no ablation reported |
| Kimi-K2-Instruct | 1.04T total / 32.6B activated | SFT | generated training examples | "tens of thousands" | arXiv:2507.20534v2 §3.1.1 intro | verified 2026-09-18 | no ablation reported |
| Kimi-K2-Instruct | 1.04T total / 32.6B activated | SFT | agentic token count and mixture share | not reported | body, §3.1, §3.1.1 checked | not reported | — |
| Kimi-K2-Instruct | 1.04T total / 32.6B activated | SFT | judge retention rate of generated trajectories | not reported | §3.1.1 "Quality Evaluation and Filtering" checked | not reported | — |
| Kimi-K2-Instruct | 1.04T total / 32.6B activated | SFT | teacher models for SFT responses | K1.5 and in-house domain-specialized expert models | arXiv:2507.20534v2 §3.1 | verified 2026-09-18 | no ablation reported |

## Findings relevant to generality, negative feedback, agentic training, distillation
- **Generality.** The two tool sources are argued to cover complementary regions of the tool space, shown by t-SNE embeddings rather than by a downstream metric (§3.1.1, Figure 9). §5 lists as a limitation that K2's performance may decline on some tasks when tool use is enabled where it is not needed.
- **Negative feedback.** Trajectories that fail the rubric are discarded, not trained on with a negative gradient; this is negative marginal value in the sense of §6.1 (§3.1.1). The report gives no rejection rate and no false-negative rate for the judge.
- **Agentic training.** The simulator's controlled stochasticity deliberately produces partial failures and edge cases inside retained trajectories, so failure appears as content inside successful trajectories rather than as a training target of its own (§3.1.1). Agentic results the stage feeds: Tau2-Bench 66.1, ACEBench (En) 76.5, SWE-bench Verified 65.8 agentic single attempt and 71.6 with multiple attempts (Abstract, Table 3).
- **Distillation.** SFT candidate responses come from K1.5 and in-house domain expert models, and are filtered by LLM or human judges (§3.1).

## Connections
- [[kimi-k2]] — the full report card; this card is its §3.1.1 in detail. [[kimi-k2-recipe]] holds the training settings.
- [[agentinstruct]], [[self-instruct]], [[toolllm]] — cited in §3.1.1 as prior work on synthetic and tool-use data.
- [[apigen-mt]] — a separate pipeline that also pairs simulated users with tool execution; useful contrast for the verification step.
- [[swe-gym]], [[swe-rl]] — real-execution environments of the kind §3.1.1 adds for coding and software engineering.
- [[glm-4-5]] — another 2025 open report with an agentic data pipeline built on simulated tools and users.

## Verification
- Checked on 2026-09-18 against: https://arxiv.org/abs/2507.20534 (arXiv v2, 2026-02), §2.2, §3.1, §3.1.1, §5, Table 3, Figures 8-9.
- Corrections to the previous card version:
  - "~1T tokens of synthetic agent trajectories added to the pretraining mix" and "agentic trajectories mixed at ~3–5% of the pretraining token budget" → the report places all of this data in supervised fine-tuning; pre-training is 15.5T tokens of web text, code, mathematics, and knowledge with no agentic-trajectory component (§2.2, §3.1, §3.1.1).
  - "Multi-agent simulation: a 'user' agent … 'planner' … 'executor' … 'critic'" and "up to 5 sub-agents per trajectory" → the report describes an LLM-simulated user, the agent under training, a tool simulator, and an LLM judge; no planner/executor/critic roles and no sub-agent count appear (§3.1.1, Figure 8b).
  - "tens of thousands of synthetic tool schemas" → 3,000+ real MCP tools plus over 20,000 synthetic tools (§3.1.1).
  - "Teacher model(s): Moonshot's internal frontier model + external GPT-4o / Claude-3.5" → SFT responses are generated by K1.5 and in-house domain-specialized expert models; no external teacher is named (§3.1).
  - "RL: GRPO / RLHF with a reward model tuned for multi-turn tool-call correctness" → §3.2 describes a Gym-like framework of verifiable-reward tasks plus a self-critique pairwise rubric reward, and §3.2.3 gives a squared-error objective over K samples with a log-ratio regularizer, not GRPO.
  - "K2-Instruct SWE-Bench Verified: ~65%" → 65.8 agentic single attempt, 51.8 agentless, 71.6 with multiple attempts (Table 3).
  - "τ-bench: K2-Instruct leads open models, competitive with Claude-3.5-Sonnet" → the report gives Tau2-Bench 66.1 and compares against Claude Sonnet 4 and Claude Opus 4, not Claude 3.5 (Abstract, Table 3).
  - Title and authorship fields were "Kimi K2 Agentic Data Pipeline" / "Kimi Team / Moonshot AI" with no source type; replaced with the report's exact title, the App. A author listing, and source type.
- Removed as unsupported by the source: the ~1T-token agentic pre-training stage and its 3–5% mixture weight; the planner/executor/critic multi-agent generator and the "up to 5 sub-agents" figure; GPT-4/GPT-4o/Claude-3.5 teachers; GRPO; "MuonClip optimizer" as a contribution of this section (it belongs to §2.1, see [[kimi-k2]]); "median several thousand tokens" and "tail to tens of thousands" trajectory lengths; "1 to 20+" tool calls per trajectory; "rule-based validators (JSON schema, no repetition loops)"; "unified OpenAI-style `tool_calls` JSON"; "millions of SFT trajectories"; "multi-million-dollar data operation"; the claim that pre-training agent tokens give the base model a native tool-calling vocabulary; the contamination and decontamination claims; "Tool-calling BFCL: strong multi-turn scores" and "LiveCodeBench: high ranking" as unlocated numbers; the DeepSeek-V3 comparison claim about being "the most explicit agent-first pretrain".
- Not reported by the source: the token volume and mixture share of the agentic SFT data; the number of tasks and trajectories generated; the judge's retention rate and its false-negative rate; the identity of the models used as user simulator, tool simulator, and judge; the simulated-versus-sandbox data split; any ablation isolating the contribution of this pipeline.
