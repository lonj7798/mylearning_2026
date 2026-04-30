<!-- scope: agentic trajectory synthesis — OpenHands agent scaffold and community training trajectories
     deps: [[swe-gym]]
     see-also: [[webarena-data]], [[swe-rl]]
-->

# OpenHands Agent Data
- **Core Insight:** An open, production-grade agent scaffold with generic action abstractions (file-edit, bash, browser, IPython) yields reusable trajectory data that transfers across SWE, web, and data-science tasks; decoupling the scaffold from model choice and from environment choice is the key design move.
- **Guideline:** Use OpenHands (successor to SWE-agent) as the trajectory-collection scaffold because its action abstractions — `str_replace_editor`, `execute_bash`, `browse`, `think`, `finish` — unify SWE/web/data-science, so one agent-data pipeline can train one model across all three.
- **Authors:** Xingyao Wang, Boxuan Li, Yufan Song, Frank F. Xu, Xiangru Tang, Mingchen Zhuge, Jiayi Pan, Yueqi Song, Bowen Li, Jaskirat Singh, Hoang H. Tran, Fuqiang Li, Ren Ma, Mingzhang Zheng, Bill Qian, Yanjun Shao, Niklas Muennighoff, Yizhe Zhang, Binyuan Hui, Junyang Lin, Robert Brennan, Hao Peng, Heng Ji, Graham Neubig (CMU + UC Berkeley + various)
- **Year:** 2024 (framework release), 2025 (paper: "OpenHands: An Open Platform for AI Software Developers as Generalist Agents")
- **URL:** https://arxiv.org/abs/2407.16741 ; https://github.com/All-Hands-AI/OpenHands
- **Relevant topics:** agent scaffold, generalist agents, SWE agents, trajectory collection

## Abstract
OpenHands (formerly OpenDevin) is an open-source platform for building AI agents that act as software developers: it provides a sandboxed Linux environment with file-editor, bash, browser, and IPython tools, and a generic agent loop that works across multiple LLMs. The 2025 paper documents OpenHands' architecture, the AgentSkills toolkit, and multi-agent orchestration. OpenHands trajectories (collected via AgentHub rollouts) are the de facto training data for many 2025 open SWE/agent models (Devstral, OpenHands-LM-32B).

## Key Contributions
- **Unified agent scaffold** with generic actions across SWE / web / data-science.
- **Sandboxed Docker runtime** — safe execution of arbitrary shell/Python.
- **AgentSkills** — a meta-tool library (e.g., `open_file`, `edit_file`, `search_dir`).
- Open trajectory releases via AgentHub and community HF datasets.
- OpenHands-LM-32B (Mar 2025) — released agent-specialist model.

## Synthesis pipeline (REQUIRED — concrete, modality-specific)

### Rollout setup
- Task source: SWE-Bench / SWE-Gym / WebArena / HumanEval-Plus / custom.
- OpenHands runtime spawns a Docker sandbox with the task's env.
- A teacher LLM (Claude-3.5, GPT-4o, Qwen-2.5-Coder-32B) drives the agent loop: sees observations, emits actions until `finish()`.

### Trajectory capture
- Each step records: `(agent_message, action, action_args, observation)` tuple.
- Full conversation saved as a multi-turn dialog where the assistant turns include tool calls and the "tool" role turns include observations.

### Filtering
- Task success via env-specific check (tests pass, URL/state predicate, gold-answer match).
- Trajectories exceeding token budget (typically 100K) truncated.

### Training-data formatting
- Convert trajectories into ChatML-style with `tool_calls` JSON for actions and `tool` role for observations.
- Typical token length: 10K–100K per trajectory.

- **Output shape:** public community releases include OpenHands-SWE-Gym (tens of thousands of SWE trajectories), OpenHands-Web trajectories, OpenHands-Data-Science trajectories.
- **Teacher model(s):** Claude-3.5-Sonnet and Qwen-2.5-Coder-32B are most commonly used.
- **Cost / compute:** per-rollout cost varies — an SWE-Bench rollout with Claude runs $1–$5; full-dataset collections reach $50K+.

## Modality-specific technical details (REQUIRED — agentic)
- **Environment:** Docker Linux sandbox with filesystem + shell + Python + optional browser (Playwright).
- **Action space (current):**
  - `str_replace_editor view / create / str_replace / insert / undo_edit`.
  - `execute_bash`.
  - `execute_ipython_cell`.
  - `browse [url]`, `click`, `type`, `scroll` (browser sub-agent).
  - `think` (internal reasoning, no env side-effect).
  - `finish` (terminate).
- **Trajectory length:** median ~15 steps, tail to 100+ steps.
- **Success criterion:** task-specific; OpenHands integrates SWE-Bench, WebArena, VisualWebArena evaluators.
- **Data scale:** community corpora in the 10K–100K-trajectory range; exact sizes vary by release.

## Quality / diversity evaluation
- OpenHands-LM-32B (Qwen-2.5-Coder base + OpenHands-SFT): SWE-Bench Verified 37.2%, strong on HumanEval+.
- OpenHands trajectories used as training ingredients for: Devstral, Mistral Codestral-Agent, several academic SWE agents.
- Scaling: rejection-sampling SFT on OpenHands trajectories lifts 7B SWE-Bench by 10–15 points (see [[swe-gym]]).

## Risks + gotchas
- **Environment / scaffold versioning:** OpenHands evolves quickly; trajectories from an old version may be incompatible with a new scaffold.
- **Safety:** the Docker sandbox isolates execution, but training data with real internet access can leak proprietary info.
- **Licensing heterogeneity:** trajectory datasets span many source licenses — careful filtering required for commercial use.
- **Not a single "dataset":** OpenHands-Data is a **scaffold + growing collection** of trajectory releases, not one paper.

## Connections
- Scaffold successor to SWE-agent (Princeton) and ReAct.
- Training-data lineage: [[swe-gym]] (primary SWE trajectory source).
- Sibling RL recipe: [[swe-rl]] (text-only, no environment).
- Eval envs: SWE-Bench, [[webarena-data]].
