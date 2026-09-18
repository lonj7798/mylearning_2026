<!-- scope: OpenHands platform paper — agent abstraction, Docker-sandboxed runtime, AgentSkills, and a 15-benchmark evaluation harness; the paper releases no training data and trains no models
     see-also: [[swe-gym]], [[webarena-data]], [[swe-rl]]
-->

# OpenHands: An Open Platform for AI Software Developers as Generalist Agents
- **Core Insight:** OpenHands connects an LLM agent to a Docker-sandboxed bash shell, IPython server, and Playwright Chromium browser through an event stream, and integrates 15 benchmarks; one CodeActAgent reaches 26.0% on SWE-Bench Lite, 15.3% on WebArena (via delegation), and 52.0% on GPQA diamond with claude-3-5-sonnet (§2, Tables 2, 4-6).
- **Guideline:** When one agent must be evaluated or run across software, web, and general-assistance tasks, a scaffold with generic code, bash, and browser actions lets the same agent and system prompt be used in all three categories (§4.1); this paper reports no training on the resulting trajectories, so evidence about trajectory data quality must come from other sources such as [[swe-gym]].
- **Authors:** Xingyao Wang, Boxuan Li, Yufan Song, Frank F. Xu, Xiangru Tang, Mingchen Zhuge, et al. (24 authors; UIUC, CMU, Yale, UC Berkeley, Contextual AI, KAUST, ANU, HCMUT, Alibaba, All Hands AI)
- **Year:** 2024 (arXiv v1 2024-07; v3 2025-04; ICLR 2025)
- **URL:** https://arxiv.org/abs/2407.16741 (code: https://github.com/All-Hands-AI/OpenHands)
- **Source type:** paper
- **Relevant topics:** agent scaffold, agent runtime, sandboxed code execution, web browsing agents, software-engineering agents, agent evaluation

## Abstract
The paper introduces OpenHands (formerly OpenDevin), a platform for building AI agents that act as software developers: they write code, use a command line, and browse the web. It describes how new agents are implemented, how actions run safely in sandboxed environments, how multiple agents coordinate, and how evaluation benchmarks are integrated. Agents are evaluated on 15 tasks, including SWE-Bench and WebArena. The platform is released under the MIT license and had more than 2.1K contributions from over 188 contributors at the time of writing. The paper does not train models or release trajectory datasets; the slug `openhands-data` is historical.

## Key Contributions
- An agent abstraction in which `step(state)` maps an event stream of past actions and observations to the next action (§2.1, Fig. 3).
- A runtime that starts one Docker container per task session and executes actions through a REST "action execution API" inside it; arbitrary user Docker images are supported (§2.2, App. F).
- AgentSkills, a Python library of tools imported into the IPython environment (§2.3, App. I), and `AgentDelegateAction` for multi-agent delegation (§2.4).
- An agent hub with over 10 agents, including CodeActAgent, BrowsingAgent, GPTSwarm, and micro agents (§1, §3).
- An evaluation framework with 15 benchmarks across software, web, and miscellaneous assistance (§4, Table 2), and integration tests with mocked LLM responses (App. E).

## Key Figures/Tables to Study
- Fig. 2: agent, event stream, and runtime components with an example action/observation sequence. Fig. 3: minimal agent code.
- Table 1: feature comparison with other agent frameworks. Table 2: the 15 benchmarks.
- Tables 3-6: success rate and average cost per benchmark. App. I-J: AgentSkills and BrowserGym action lists.

## Technical Details
**Actions and observations.** Core actions are `IPythonRunCellAction` (Python), `CmdRunAction` (bash), and `BrowserInteractiveAction`, which uses the BrowserGym browsing DSL (§2.1). The minimal agent in Fig. 3 also returns `AgentFinishAction` and `MessageAction`. The state also holds accumulated LLM cost and delegation metadata (§2.1).

**Runtime.** The API server in the container maintains a bash shell, a Jupyter IPython server, and a Playwright Chromium browser (§2.2). Browser observations include HTML, DOM, accessibility tree, screenshot, and open tabs (§2.2). Browsing actions are the BrowserGym v0.3.4 set (App. J); the BrowsingAgent prompt lists 16 action types (App. K). Runtime images use a hash-based tag (MD5 of the build folder) and a generic version tag (App. F.2.1).

**AgentSkills.** A tool is added only when the LLM cannot easily write the code directly or the tool calls an external model (§2.3). As of v0.6 the list includes `open_file`, `goto_line`, `scroll_up`/`scroll_down` (100 lines), `create_file`, `edit_file`, `search_dir`, `search_file`, `find_file`, and parsers for PDF, DOCX, LaTeX, audio, image, video, and PPTX (App. I). File-editing tools are adapted from SWE-Agent and Aider (§2.3).

**Agents.** CodeActAgent is the default generalist agent; at each step it either converses in natural language or executes bash, Python, or browser code (§3). BrowsingAgent uses zero-shot prompting (§3). CodeActSWEAgent reuses an in-context demonstration from a released SWE-agent trajectory (App. H).

**Evaluation settings.** SWE-Bench results use no hint text and the 300-instance Lite subset for cost (§4.2). Running full SWE-bench (2,294 instances) is estimated at $6.9k at $3 per instance (§4.2 fn. 2); one SWE-Bench Lite run with gpt-4o costs about 600 USD (App. E fn. 4). MINT allows up to five interaction iterations with two solution proposals (§4.4).

| Benchmark (instances) | Agent + model | Success % | Avg. cost $ | Locus |
|---|---|---|---|---|
| SWE-Bench Lite (300) | CodeActAgent v1.8 + claude-3-5-sonnet@20240620 | 26.0 | 1.10 | Table 4 |
| SWE-Bench Lite (300) | CodeActAgent v1.8 + gpt-4o-2024-05-13 | 22.0 | 1.72 | Table 4 |
| SWE-Bench Lite (300) | CodeActAgent v1.8 + gpt-4o-mini-2024-07-18 | 7.0 (Table 3 prints 6.3) | 0.01 | Tables 3-4 |
| HumanEvalFix (164) | CodeActAgent v1.5 + gpt-4o, 0-shot | 79.3 | 0.14 | Table 4 |
| WebArena (812) | BrowsingAgent v1.0 + claude-3-5-sonnet-20240620 | 15.5 | 0.10 | Table 5 |
| WebArena (812) | CodeActAgent v1.8 via delegation + claude-3-5-sonnet | 15.3 | – | Table 5 |
| MiniWoB++ (125 envs) | BrowsingAgent v1.0 + gpt-4o | 40.8 | 0.05 | Table 5 |
| GAIA L1 validation (53) | GPTSwarm v1.0 + gpt-4o | 32.1 | 0.050 | Table 6 |
| GPQA diamond (198) | CodeActAgent v1.8 + claude-3-5-sonnet | 52.0 | 0.065 | Table 6 |
| AgentBench OS (144) | CodeActAgent v1.5 + gpt-4o | 57.6 | 0.085 | Table 6 |

Selected baselines: Agentless + gpt-4o 27.3% and SWE-Agent + gpt-4-1106-preview 18.0% on SWE-Bench Lite (Table 3); SWE-agent 1-shot + gpt-4-turbo 87.7% on HumanEvalFix (Table 4); Auto Eval & Refine 20.2% on WebArena (Table 5).

## Findings relevant to generality, agentic training, long context
- **Generality (Result, single study).** The same CodeActAgent, without system-prompt changes, is competitive across software, web, and miscellaneous tasks, while most baselines target one category (§4.1, Table 3). It is not the top system in each category: Agentless scores 27.3% vs 26.0% on SWE-Bench Lite, and Auto Eval & Refine 20.2% vs 15.5% on WebArena (Tables 3, 5).
- **Measurement caveats.** HumanEvalFix compares OpenHands 0-shot with SWE-agent 1-shot, where the demonstration comes from the test set (§4.2.1). Tables mix agent versions v1.5 and v1.8 (Table 3 footnote). The gpt-4o-mini SWE-Bench Lite score differs between Table 3 (6.3) and Table 4 (7.0).
- **Agentic training.** The paper trains no models and reports no trajectory datasets. App. A lists building better agents "through both training and inference time techniques" as future work. Trained baselines appear only as comparisons (e.g., AutoWebGLM 7B, 18.2% on WebArena; Table 5).
- **Long context.** App. A reports that current agents perform poorly when editing long files and lists this as future work; no measurement is given.

## Connections
- [[swe-gym]]: uses OpenHands (CodeActAgent 2.1) as the scaffold for its agent fine-tuning experiments (arXiv:2412.21139 §4.2).
- [[webarena-data]]: WebArena (812 tasks) is one of the 15 integrated benchmarks (§4.3).
- [[swe-rl]]: a software-engineering RL method; this paper contains no RL.
- Agent Data Protocol (arXiv:2510.24702; no card in this library): converts 13 agent datasets into one schema and then into SFT formats for OpenHands, SWE-Agent, and AgentLab (ADP Abstract, Fig. 1, Table 7).

## Verification
- Checked on 2026-09-14 against: https://arxiv.org/abs/2407.16741 (v3, 18 Apr 2025; ICLR 2025).
- Corrections to the previous card version: title "OpenHands Agent Data" → no artifact has that title; the card's URL and author list identify the OpenHands platform paper, which this card now describes. "OpenHands (successor to SWE-agent)" → formerly OpenDevin, initially inspired by Devin (Abstract, §1 fn. 1); only its file-editing skills are adapted from SWE-Agent and Aider (§2.3). "CMU + UC Berkeley + various" → ten affiliations listed above. "Year: 2025 (paper)" → arXiv v1 July 2024, ICLR 2025. "OpenHands trajectories (collected via AgentHub rollouts)" → AgentHub is a set of agent implementations (§3); no trajectory collection is described. Action space "`str_replace_editor`, `execute_bash`, `browse`, `think`, `finish`" → the paper's actions are `IPythonRunCellAction`, `CmdRunAction`, `BrowserInteractiveAction`, `AgentDelegateAction`, plus finish and message actions (§2.1, §2.4, Fig. 3); the listed tool names do not appear in the paper. "SWE-Bench / WebArena / VisualWebArena evaluators" → VisualWebArena is not among the 15 benchmarks (Table 2).
- Removed as unsupported by the source: "de facto training data for Devstral, OpenHands-LM-32B"; "OpenHands-LM-32B (Mar 2025), SWE-Bench Verified 37.2%"; "Mistral Codestral-Agent trained on OpenHands trajectories"; "rejection-sampling SFT on OpenHands trajectories lifts 7B SWE-Bench by 10–15 points"; rollout teacher models (Claude-3.5, GPT-4o, Qwen-2.5-Coder-32B); step-tuple capture format; ChatML `tool_calls` conversion; 100K-token truncation; "10K–100K tokens per trajectory"; "median ~15 steps, tail to 100+"; "OpenHands-SWE-Gym / -Web / -Data-Science releases"; "$1–$5 per rollout, $50K+ collections"; "10K–100K-trajectory community corpora"; HumanEval-Plus as a task source; licensing and data-leakage risk statements.
- Not reported by the source: any training data, training recipe, trajectory statistics, or RL setup.
