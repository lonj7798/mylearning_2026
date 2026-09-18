<!-- scope: Agent Lightning framework (Microsoft Research): RL training of the LLM inside existing agents by converting executions into per-call (input, output, reward) transitions; LightningRL credit assignment; trainer-agent disaggregation
     deps: [[grpo]], [[ppo]]
     see-also: [[verl-rollout]], [[gigpo-verl-agent]], [[ragen-starpo]], [[skyrl-agent]], [[turn-level-credit-assignment]]
-->

# Agent Lightning: Train ANY AI Agents with Reinforcement Learning
- **Core Insight:** Agent Lightning treats each LLM call in an agent execution as one action, stores executions as (input, output, reward) transitions, gives every call in an episode the episode's final return, and then applies a single-turn RL method grouped by task; with Llama-3.2-3B-Instruct on three agents (LangChain text-to-SQL on Spider, OpenAI Agents SDK retrieval QA on MuSiQue, AutoGen calculator math on Calc-X) the paper reports rising train and test reward curves, but no numeric tables (§3.3.2, §4, Figs. 5-7).
- **Guideline:** When an agent builds each LLM call's context separately (role prompts, summaries, several agents sharing one model), train on per-call transitions instead of one concatenated, masked trajectory, because this format required almost no agent-code modification in the paper's examples and allows optimizing a chosen subset of agents (2 of 3 in the text-to-SQL workflow, §4.1); the paper reports no comparison against masked multi-turn training and no general-capability evaluation, so the benefit over that baseline is not measured.
- **Authors:** Xufang Luo, Yuge Zhang, Zhiyuan He, Zilong Wang, Siyun Zhao, Dongsheng Li, et al. (Microsoft Research)
- **Year:** 2025 (arXiv v1 2025-08)
- **URL:** https://arxiv.org/abs/2508.03680
- **Source type:** paper (framework and algorithm; code at github.com/microsoft/agent-lightning, not checked for this card)
- **Relevant topics:** agentic RL infrastructure, multi-turn credit assignment, transition-level vs trajectory-level training, multi-agent training, rollout/trainer decoupling, observability-based trace capture, intermediate rewards

## Abstract
Agent Lightning is a framework for reinforcement learning of the LLMs used inside any AI agent. Existing methods couple RL training to the agent implementation or concatenate turns into one sequence with masks. Agent Lightning separates agent execution from training, so agents written with LangChain, OpenAI Agents SDK, AutoGen, or from scratch can be trained with almost no code modification. Agent execution is formulated as a Markov decision process with a unified data interface, and a hierarchical algorithm, LightningRL, uses a credit assignment module to decompose trajectories from any agent into training transitions, which supports multi-agent and dynamic workflows. The system uses a Training-Agent Disaggregation architecture and brings agent observability tooling into the agent runtime. Experiments on text-to-SQL, retrieval-augmented generation, and math tool use show stable, continuous improvement.

## Key Contributions
- A unified data interface: agent state as a set of "semantic variables"; each component invocation recorded as (meta, input, output) with an optional reward (§3.1, Eqs. 1-5).
- A POMDP view in which one action is the full token sequence of one policy-LLM call, and the RL data for an execution is {(input_t, output_t, r_t)} (§3.2, Eqs. 6-7).
- LightningRL: episode return assigned to calls, then token-level optimization by an unmodified single-turn algorithm such as GRPO, PPO, or REINFORCE++ (§3.3).
- Training-Agent Disaggregation: a Lightning Server attached to the RL framework that exposes an OpenAI-like API, and a Lightning Client that runs agents and returns traces (§3.4.1, Fig. 4).
- Agent runtime features: two-level data parallelism, OpenTelemetry/AgentOps or endpoint-based tracing, failure detection with retry or reassignment, Automatic Intermediate Rewarding (AIR), and pooled environment/reward services (§3.4.2).

## Key Figures/Tables to Study
- **Fig. 2:** a RAG agent execution rewritten as semantic-variable updates and the extracted transitions.
- **Fig. 3:** single-call GRPO vs previous multi-turn GRPO with masked non-generated tokens vs LightningRL grouping transitions by task.
- **Fig. 4 and App. B Fig. 8:** server/client architecture and process diagram.
- **Table 1:** the three tasks, frameworks, tools, and number of agents vs tuned agents.
- **Figs. 5-7:** train and test reward curves (the only reported results).
- **App. A Listing 2:** a train.py that wraps an existing agent function with `Client.train`.

## Technical Details
- **Agent definition:** a software system with one or more LLM calls; components are LLMs M = {M_i} and tools T = {T_j} (§2.1).
- **Execution record:** execution(x, k) = {call_i}, call_i = (meta_i, input_i, output_i) with output_i = C_i(input_i), C_i ∈ M ∪ T; meta holds name, version, endpoint, and sampling parameters (§3.1.1, Eqs. 2-3). With rewards: {(call_i, r_i)}; a terminal-only reward is the special case r_1..r_{N−1} absent (§3.1.2, Eq. 5).
- **POMDP:** observation = the part of the state passed to the policy LLM; action a_t = (y_{t,1}, …, y_{t,N_t}); return R = Σ_t r_t (§3.2.1).
- **Token loss used after credit assignment:** L(θ) = −E[Σ_j log π_θ(y_j | x, y_<j) · A_j], where A_j is the token-level advantage; importance ratio, clipping, and KL terms are omitted from the equation but "adopted accordingly" (§3.3.1, Eq. 8, footnote 3).
- **Credit assignment as implemented:** "each action within the episode has the same value, equal to the final return R"; for GRPO, each task is run several times, every execution is split into its calls, and all resulting samples of a task form one group (§3.3.2). A learned high-level value function is listed as future work (§3.3.2).
- **Author-stated reasons against concatenate-and-mask training (not measured):** masks require coupling training to agent logic, break the token continuity assumed by RoPE, complicate code and kernels, and accumulated turns produce long sequences; transitions allow batch accumulation (§3.3.2, §5.1).
- **Multi-agent handling:** one LLM acting as several agents through different prompts; selective optimization by including only chosen agents' transitions. For several distinct LLMs, the simple option is one independent MDP per LLM; MARL is named as more principled but not implemented (§3.2.2).
- **Server:** divides the task dataset into batches, dispatches tasks to ready clients with a per-task OpenAI-like endpoint, receives traces and rewards, and forwards them to the training framework (e.g., VeRL) (§3.4.1).
- **AIR:** converts system monitoring signals, for example tool-call return statuses, into intermediate rewards for transitions (§1, §3.4.2). Whether AIR was used in the three experiments is not stated.
- **Tasks (Table 1, §4):** Spider (over 10,000 questions, 200 databases, 138 domains; unseen databases at test time) with a 3-role LangChain workflow (SQL writer, checker, re-writer/answerer) where writer and re-writer are trained (§4.1); MuSiQue over a Wikipedia index of 21 million documents with BGE embeddings and cosine similarity (§4.2); Calc-X, built from GSM8K and Ape210K, with a calculator tool (§4.3).
- **Rewards:** Spider and Calc-X: final answer correctness (§4.1, §4.3). MuSiQue: R = 0.9 × R_correctness + 0.1 × R_format, where R_correctness is word-level F1 and R_format = 1 when output uses the <think>, <query>, <answer> tags (§4.2).

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Llama-3.2-3B-Instruct (all three tasks) | 3B | RL | base model | Llama-3.2-3B-Instruct for Spider, MuSiQue, Calc-X | arXiv:2508.03680v1 §4.1, §4.2, §4.3 | verified 2026-09-14 | not applicable |
| Llama-3.2-3B-Instruct, MuSiQue agent | 3B | RL | reward | 0.9 × word-level F1 + 0.1 × format score | §4.2 | verified 2026-09-14 | no ablation reported |
| Llama-3.2-3B-Instruct, all tasks | 3B | RL | credit assignment | final return copied to every call; grouping by task | §3.3.2 | verified 2026-09-14 | no ablation reported |
| Llama-3.2-3B-Instruct, all tasks | 3B | RL | RL algorithm used in the runs; LR; batch; rollouts per task; KL; clip; max lengths; steps; hardware | not reported (checked §3.3-§4, Figs. 5-7, App. A-B) | — | not reported | none |

## Findings relevant to agentic training and generality
- **Agentic training format:** the framework trains on per-call transitions rather than concatenated turns with masks, and the paper positions this against RAGEN, Trinity-RFT, rLLM, and Search-R1, which "typically concatenate agent turns" (§5.1). No experiment compares the two formats.
- **Multi-agent selective training:** in text-to-SQL, two of three prompt-defined roles of one LLM are optimized together (§4.1, Table 1).
- **Evidence strength:** results are reward curves on each task's train and test data only (Figs. 5-7); no numeric scores, baselines, seeds, or evaluations outside the three tasks are reported. Result (single study), curves only.

## Connections
- [[grpo]], [[ppo]], [[reinforce-plus-plus]]: the single-turn algorithms LightningRL plugs in for token-level updates (§3.3.1-3.3.2).
- [[verl-rollout]]: verl is the named training backend, and its multi-turn agent loop uses the response-mask approach the paper argues against (§3.4.1, §5.1).
- [[ragen-starpo]], [[search-r1]], [[simpletir]]: cited multi-turn or tool-integrated RL methods (§5.1).
- [[gigpo-verl-agent]], [[turn-level-credit-assignment]]: finer-grained credit assignment than copying the episode return to every call.
- [[areal-async-rl]], [[deepswe]]: cited RL systems and agent-RL efforts (§5.1).
- [[skyrl-agent]]: later report from the SkyRL team; the paper cites the earlier SkyRL-v0 as an RL framework with agent extensions (§5.1).
- [[agent-early-experience]]: reward-free alternative for agents whose environments lack rewards.

## Verification
- Created on 2026-09-14 from https://arxiv.org/abs/2508.03680 (v1, 2025-08-05; only version).
- Audit claims not found in the source: none. Wording note: the audit's "OpenAI-compatible endpoint" appears in the paper as "OpenAI-like API" (§1, §3.4.1). The audit's re-check request is resolved: Llama-3.2-3B-Instruct is stated for all three tasks (§4.1-4.3).
- Not reported by the source: training hyperparameters, compute, numeric results, comparison with masked multi-turn training, general-capability or held-out-domain evaluation.
