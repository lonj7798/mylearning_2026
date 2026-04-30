<!-- scope: agentic trajectory synthesis — Magentic-One multi-agent orchestration data
     deps: [[agentinstruct]]
     see-also: [[kimi-k2-agentic-data]], [[openhands-data]]
-->

# Magentic-One: A Generalist Multi-Agent System for Solving Complex Tasks
- **Core Insight:** A **central Orchestrator agent** that plans, tracks progress, and delegates to specialized sub-agents (WebSurfer, FileSurfer, Coder, ComputerTerminal) produces agentic trajectories that succeed on complex multi-step tasks where monolithic agents fail; the orchestration protocol itself becomes reusable training data for smaller agents.
- **Guideline:** For complex multi-step agent tasks, use a hierarchical orchestrator pattern: a planner agent maintains a task ledger (facts, plan, progress); specialized sub-agents handle web, file, code, terminal actions; collect the full orchestration trace as training data.
- **Authors:** Adam Fourney, Gagan Bansal, Hussein Mozannar, Cheng Tan, Eduardo Salinas, Erkang (Eric) Zhu, Friederike Niedtner, Grace Proebsting, Griffin Bassman, Jack Gerrits, Jacob Alber, Peter Chang, Ricky Loynd, Robert West, Victor Dibia, Ahmed Awadallah, Ece Kamar, Rafah Hosn, Saleema Amershi (Microsoft AutoGen team)
- **Year:** 2024 (Nov)
- **URL:** https://arxiv.org/abs/2411.04468
- **Relevant topics:** multi-agent orchestration, Magentic-One, AutoGen, complex-task trajectories

## Abstract
Magentic-One is a generalist multi-agent system built on Microsoft's AutoGen framework. An **Orchestrator** LLM maintains a "Task Ledger" (facts known, plan, sub-task assignments) and delegates to specialized agents: WebSurfer (browse), FileSurfer (navigate files), Coder (write/run Python), ComputerTerminal (execute shell). On GAIA, WebArena, and Assistant-Bench, Magentic-One achieves leading complex-task success rates. While the paper focuses on the system, the orchestration traces it produces form a valuable dataset for training smaller agents in similar multi-agent patterns.

## Key Contributions
- **Orchestrator + specialist sub-agents** design pattern.
- **Task Ledger** — explicit state tracking across multi-agent delegation.
- **Complex-task benchmarks:** leading open scores on GAIA and Assistant-Bench at release.
- **AutoGen integration** — the orchestration framework is public.

## Synthesis pipeline (REQUIRED — concrete, modality-specific)

### System architecture
- **Orchestrator LLM:** receives task, builds/maintains Task Ledger.
- **Specialized sub-agents:**
  - **WebSurfer:** browser control.
  - **FileSurfer:** file-system navigation + content inspection.
  - **Coder:** Python code authoring + execution.
  - **ComputerTerminal:** shell command execution in sandbox.
- **Protocol:**
  1. Orchestrator drafts Task Ledger (known facts, plan).
  2. Selects next action → delegates to appropriate sub-agent.
  3. Sub-agent executes, returns observation.
  4. Orchestrator updates ledger, re-plans if needed.
  5. Repeat until task completion or max-step limit.

### Trajectory synthesis for data purposes
- Run Magentic-One (with GPT-4o or equivalent as all LLMs) on GAIA / WebArena tasks.
- Capture full orchestration trace: orchestrator messages, ledger snapshots, sub-agent calls and observations.
- Filter by task success (success predicate per benchmark).
- The resulting trajectories are richer than single-agent ReAct traces because they include planning/ledger-update segments.

- **Output shape:** per task, a full multi-agent trajectory ~10K–50K tokens. Several thousand successful trajectories collectible across public benchmarks.
- **Teacher model:** GPT-4o (primary), Claude-3.5-Sonnet (alternative).
- **Cost:** ~$5–$20 per trajectory depending on length.

## Modality-specific technical details (REQUIRED — agentic)
- **Environment:** mixed — web browser (WebSurfer), file system + Docker sandbox (FileSurfer, Coder, Terminal).
- **Action space:** per sub-agent — browser actions (click/type/scroll), file operations (open/search), code execution, shell commands.
- **Trajectory length:** median ~20 steps (including orchestrator turns and sub-agent turns); tail to 100+.
- **Success criterion:** task-specific (benchmark predicate).
- **Data scale:** rollout collections depend on use; specific public-dataset numbers are not centralized — users typically run Magentic-One on chosen tasks.
- **Unique training signal:** the orchestrator's "Task Ledger" update turns teach explicit state tracking — a structured skill that plain ReAct doesn't exhibit.

## Quality / diversity evaluation
- Magentic-One on GAIA: leading open-source result at release.
- Assistant-Bench, WebArena: strong scores.
- Smaller agents fine-tuned on Magentic-One trajectories inherit multi-agent coordination patterns.
- Qualitative: Task Ledger tokens in training data teach models to maintain structured state across turns.

## Risks + gotchas
- **Orchestrator-agent dependency cycle:** sub-agents assume the orchestrator's messages follow a specific format; student models must learn this format.
- **Cost:** running four LLM agents per step is expensive.
- **Not a packaged dataset:** Magentic-One is a system, and the public release includes code, not a pre-built SFT corpus — users must run it to produce data.
- **Sub-agent boundary rigidity:** adding a new sub-agent requires re-prompting the orchestrator.

## Connections
- Framework sibling: AutoGen (same MS team).
- Trajectory-training ecosystem: [[openhands-data]] (single-agent scaffold), [[kimi-k2-agentic-data]] (pretraining-scale agent data).
- Orchestration conceptual sibling: [[lumos]] (Plan/Ground/Execute — single-model modular).
- Benchmarks: GAIA, Assistant-Bench, WebArena.
