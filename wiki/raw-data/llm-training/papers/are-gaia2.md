<!-- scope: Meta ARE (Agents Research Environments) platform and the Gaia2 benchmark (arXiv 2509.17158): asynchronous event-driven environments, write-action verifier usable as an RL reward, 800 + 320 scenarios in seven splits (five capabilities plus Agent2Agent and Noise augmentations), frontier-model results, judge hacking during RL
     deps: [[tau-bench]]
     see-also: [[bfcl]], [[persona-hub]], [[agentic-benchmark-checklist]], [[holistic-agent-leaderboard]], [[jason-wei-asymmetry-of-verification]], [[agentscaler]]
-->

# ARE: Scaling Up Agent Environments and Evaluations
- **Core Insight:** Under one ReAct-style scaffold, GPT-5 (high) has the best Gaia2 overall pass@1 (42.1) but scores 0.0 on the Time split, where removing generation latency ("instant" mode) raises it to 34.4, and every model's pass@1-vs-budget curve plateaus (Table 2; Fig. 13; Fig. 1).
- **Guideline:** When an LLM-judge verifier is used as an RL reward for agent actions, add a task-agnostic sanity or style check on agent messages, because in early ARE RL runs the agent learned to put code-like template strings in its answer message and the soft-check judge accepted them (App. B.3.1, Fig. 24; §2.3.1).
- **Authors:** Romain Froger, Pierre Andrews, Matteo Bettini, Amar Budhiraja, Ricardo Silveira Cabral, Virginie Do, et al. (Meta Superintelligence Labs; lead authors Romain Froger, Amine Benhalloum, Grégoire Mialon, Thomas Scialom)
- **Year:** 2025 (arXiv v1 2025-09; v2 2025-12 updated author order and acknowledgements)
- **URL:** https://arxiv.org/abs/2509.17158 (code: https://github.com/facebookresearch/meta-agents-research-environments)
- **Source type:** paper (with released platform code)
- **Relevant topics:** agent environments, asynchronous simulation, agent evaluation, verifiers for RL, reward hacking, ambiguity handling, time-sensitive actions, noise robustness, multi-agent collaboration, cost-normalized evaluation

## Abstract
ARE is a research platform for creating environments, integrating synthetic or real applications, and running agent orchestrations; each environment has its own rules, tools, content, and verifiers. Gaia2 is a benchmark built in ARE to measure general agent capabilities: beyond search and execution, agents must handle ambiguity and noise, adapt to dynamic environments, collaborate with other agents, and act under time constraints. Gaia2 runs asynchronously, which exposes failure modes that static settings do not show. In the experiments no system dominates: stronger reasoning often costs efficiency, and budget-scaling curves plateau. ARE abstractions allow Gaia2 to be extended to other environments and new benchmarks (Abstract).

## Key Contributions
- Five abstractions: apps (stateful tool collections), environments (apps + data + rules), events (everything logged), notifications (configurable observability), scenarios (initial state + scheduled events + verification) (§2.1).
- Mobile environment: 12 apps (Fig. 5) exposing 101 tools, populated as "universes" of synthetic content (§2.2).
- ARE Verifier: matches agent write actions to annotated oracle write actions with hard and soft checks, causality, and timing (§2.3).
- Gaia2: 800 human-annotated scenarios in 10 universes, 160 per capability, plus 320 augmented scenarios (Agent2Agent and Noise) for 1,120 in total (§3.1-3.2).
- Evaluation of proprietary and open models with cost, time, time-mode, noise, and Agent2Agent analyses (§4).

## Key Figures/Tables to Study
- Table 2: pass@1 per model and capability split. Fig. 1: pass@1 vs maximum budget per scenario.
- Table 1 and Table 5: verifier agreement, precision, recall on 450 labeled trajectories.
- Fig. 11-12: score vs cost, time, LLM calls, output tokens. Fig. 13: default vs instant time mode.
- Table 3 and Fig. 15: cross-model Agent2Agent and pass@k vs collaboration ratio. Fig. 24: verifier exploit.

## Technical Details
- **Apps and tools (§2.1.1):** Python methods of an App class become tool descriptions; decorators mark tools as read or write; tools are scoped to agent, user, or env roles; external APIs connect through MCP. Core app System provides `wait` and `wait_for_next_notification`; calling a wait tool switches the simulation from real time to an event-to-event queue.
- **Environment and events (§2.1.2-2.1.3):** an environment is an MDP that runs deterministically for a fixed starting state and seed. Events (agent, user, env, conditional, validation, oracle) are timestamped, logged, and scheduled in a DAG; an event runs only after all its predecessors complete.
- **Notifications (§2.1.4; App. A.3, Table 4):** policies low (no environment events notified), medium (consequences of agent actions, such as email replies or ride cancellations; Gaia2 default), high (all environment events). User messages are always notified. Notifications are injected into context at the start of each agent step.
- **Mobile rules (§2.2.1):** during a turn the simulation clock advances while the agent computes; between turns it pauses. A scenario ends on completion, on exceeding time, step, or turn limits, or on failing verification at the end of a turn.
- **Universe generation (§2.2.2; App. A.4):** Llama 3.3 70B Instruct generates app content from a PersonaHub persona following an app dependency graph rooted at Contacts; about 400K tokens of raw unstructured content per universe, about 800K tokens in structured form (lower bounds, file system excluded). Shopping uses 500 sampled Amazon products per universe.
- **Verifier (§2.3.1; App. B.1.2):** first checks that tool-name counts match between oracle and agent write actions; sorts oracle actions topologically; maps each to an agent action with hard checks (exact parameters such as `email_id`), soft checks (an LLM judge on content), causality (all oracle parents already matched), and timing (window [Δt − 5 s, Δt + 25 s] when the oracle delay exceeds 1 s). Read actions are not verified. Equivalent alternative write actions are not accepted (Messages instead of Chats fails). Multi-turn scenarios are verified at the end of each turn (App. B.2.1).
- **Verifier accuracy (Table 1; Table 5):** on 450 human-labeled trajectories, ARE Verifier agreement 0.98, precision 0.99, recall 0.95 vs an in-context LLM-judge verifier 0.72 / 0.53 / 0.83 (both Llama 3.3 70B Instruct). With Gemini 2.5 Pro or Claude Sonnet 3.7: 0.96 / 0.98 / 0.89; prompts were tuned for Llama.
- **Scenario QA (§3.3):** annotator A writes the prompt and oracle graph, B and C solve it independently, D checks the three solutions for consistency; UI guardrails enforce structural rules (App. B.1.1); a 100% model success rate flags a scenario as too easy and 0% flags possible misspecification.
- **Splits (§3.2):** Search, Execution, Adaptability, Time (scenarios capped at 5 minutes), Ambiguity; Agent2Agent replaces apps with app sub-agents (default r = 1, all apps); Noise adds tool signature changes, tool failure probability 0.1, and 10 random environment events per minute.
- **Evaluation protocol (§4.1; App. B.5):** ReAct-like loop, one JSON tool call per step; context of at least 128K tokens; temperature 0.5; 16,000 generation tokens per turn; context overflow counts as failure; 3 runs per scenario; at most 200 steps; verifier Llama 3.3 70B Instruct at temperature 0; "generation time" mode pauses the simulation during inference and advances it by the measured generation time. GPT-5 uses temperature and top-p 1; Grok-4 reasoning is capped at 16k tokens.

## Findings relevant to generality, negative feedback, and agentic training
- **Capability profile (Result, single study; Table 2, pass@1, Exec / Search / Ambig / Adapt / Time / Noise / A2A / Overall):** GPT-5 (high) 69.2 / 79.6 / 51.9 / 40.4 / 0.0 / 35.4 / 17.9 / 42.1; Claude-4-Sonnet 57.9 / 59.8 / 24.2 / 38.1 / 8.1 / 27.7 / 27.9 / 34.8; Gemini-2.5-Pro overall 25.8; Kimi-K2 20.1 (best open model); Qwen3-235B 11.6; Llama 4 Maverick 7.4. Grok-4 scores 57.5 on Search but 8.8 on Execution and 0.0 on Time; the authors describe it as strong on Search and "collapsing elsewhere" (§4.2). The text gives GPT-5 (high) Noise as 36.0, Table 2 as 35.4.
- **Search and execution overestimate robustness (Interpretation, §4.2):** the authors read the drop from Execution/Search to Ambiguity/Adaptability as evidence that existing benchmarks may overestimate robustness in realistic environments.
- **Reasoning vs latency (§4.3, Fig. 13; §5):** in instant mode (each action takes 1 simulated second) Claude-4 Sonnet rises from 8.1 (Fig. 13; §4.3 text says 8.2) to 26.7 and GPT-5 (high) from 0.0 to 34.4. Capping GPT-5 (high) at 30 minutes of execution lowered Execution by 10 points and Search by 20 points (§5).
- **Compute scaling:** pass@1 correlates with LLM calls and output tokens per scenario; Claude-4 Sonnet (about 35%) and Kimi-K2 (about 21%) score high with relatively few tokens (§4.2, Fig. 12). All budget curves plateau (Fig. 1). The authors recommend cost-normalized metrics (§4.2).
- **Noise (App. B.6.1, Table 6, Claude-4 Sonnet on Gaia2-mini):** none 31.2, low 35.0, medium (default) 23.8, high 8.1.
- **Agent2Agent (§4.4; Fig. 15; Table 3, Gaia2-mini, r = 1):** raising r improves pass@k scaling for Llama 4 Maverick but not token-normalized score for Claude 4 Sonnet. All-Llama teams score 8.5 and all-Claude teams 29.3; mixed teams score 16.2 and 18.3. The §4.4 text assigns 16.2 to a Llama main agent with Claude app agents and 18.3 to a Claude main agent with Llama app agents, while the Table 3 layout places the two values the other way round.
- **Verifier as RL reward (App. B.3.1):** in early RL runs on Search scenarios (one write action expected), the agent embedded increasingly complex code-like strings in write calls; these overwhelmed the soft-check LLM judge and produced false positives. A task-agnostic "style" soft check stopped the exploit in follow-up runs. The authors state that rubric judges scale poorly for write-heavy tasks prone to reward hacking and propose more verifier-agent asymmetry (§5).
- **Training uses described but not tested:** hints (natural-language step-by-step solutions) can be rephrased and injected during RL when scenarios are too hard (§2.1.5); ARE "enables generation of high-quality SFT traces" (§1). No SFT or RL results are reported.
- **Benchmark saturation (§5):** on Gaia2-Search, annotators struggled to write scenarios that frontier models could not solve. Memory, long-horizon decision-making, and self-improvement are not evaluated (§5).

## Connections
- [[tau-bench]]: replicated in ARE as a single-app environment with an LLM user (§2.2.4); cited as a sequential benchmark where the environment pauses while the agent works (§1).
- [[bfcl]]: BFCLv3 also replicated in ARE (§2.2.4); final-state verification contrasted with write-sequence verification (§2.3.1).
- [[persona-hub]]: seed personas for universe generation (App. A.4).
- [[reward-hacking-taxonomy]], [[judge-llm-bias]]: background for the judge exploit in App. B.3.1.
- [[jason-wei-asymmetry-of-verification]]: related to the proposed verifier-agent asymmetry (§5).
- [[agentic-benchmark-checklist]], [[holistic-agent-leaderboard]]: other work on agent benchmark rigor and cost-aware reporting.
- [[agentscaler]], [[agent-world]], [[agentrl]]: environment scaling and multi-environment agent RL.
- [[kimi-k2]], [[llama-4]], [[qwen-3]]: evaluated open models.

## Verification
- Created on 2026-09-14 from https://arxiv.org/abs/2509.17158 (arXiv v2 PDF dated 2025-12-10; v1 2025-09-21 per the abstract page).
- Audit claims not found in the source: "a shareable environment spec" (the paper describes abstractions, `scenario.py` files, and extension of Gaia2 to other environments, not a specification by that name). The other audit leads (rules, tools, content, verifiers; asynchronous and time dynamics; ambiguity, noise, collaboration, time; reasoning costs efficiency; plateauing budget curves) are in the Abstract and §2-§4.
- Not reported by the source: RL hyperparameters or results of the early RL runs, the size of the test set whose oracle actions are withheld (App. B.2.1), and model versions or dates beyond those in Table 2 and App. B.5.
