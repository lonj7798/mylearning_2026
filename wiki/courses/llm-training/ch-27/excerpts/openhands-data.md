---
chapter: ch-27
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/openhands-data.md
source_url: https://arxiv.org/abs/2407.16741
created_at: "2026-04-23"
---

# Excerpt: OpenHands — unified action abstractions across SWE / web / data-science

**Source library:** `wiki/raw-data/llm-training/papers/openhands-data.md`
**Paper:** Wang et al. 2024 (CMU + UC Berkeley + others), "OpenHands: An Open Platform for AI Software Developers as Generalist Agents."

---

## Why this source anchors ch-27 §2.2 and §5

Ch-27 §5's action-space table has one row for "Repo / SWE" and another for "Terminal." Both entries cite OpenHands. That's because OpenHands's action abstraction **unifies** SWE, web, terminal, and data-science into one scaffold. Understanding OpenHands means understanding the ch-27 §5 table's repo-and-terminal rows.

## The action vocab

From the source (lines 48-55):

> - **Action space (current):**
>   - `str_replace_editor view / create / str_replace / insert / undo_edit`.
>   - `execute_bash`.
>   - `execute_ipython_cell`.
>   - `browse [url]`, `click`, `type`, `scroll` (browser sub-agent).
>   - `think` (internal reasoning, no env side-effect).
>   - `finish` (terminate).

Six top-level actions plus browser sub-actions. `str_replace_editor` has six sub-operations (view, create, str_replace, insert, undo_edit) — it's the file-manipulation API. `execute_bash` is everything shell. `execute_ipython_cell` is Python without the subprocess overhead. `browse` opens a browser sub-agent. `think` is the no-op reasoning action. `finish` terminates with a final answer or patch.

## Why the unified vocab matters

From the source (line 7):

> - **Core Insight:** An open, production-grade agent scaffold with generic action abstractions (file-edit, bash, browser, IPython) yields reusable trajectory data that transfers across SWE, web, and data-science tasks; decoupling the scaffold from model choice and from environment choice is the key design move.

The bet: one scaffold, one action vocab, three task types. SWE (file-edit + bash + test execution), web (browse + click + type), data-science (IPython + file-edit). A single model trained on OpenHands trajectories from all three can deploy to any of them without scaffold change.

Compare to the alternative: train separate SWE-agent, web-agent, and data-science-agent models with separate action vocabs. Triple the training cost, triple the maintenance, no cross-transfer.

## The trajectory format

From the source (lines 32-34):

> ### Trajectory capture
> - Each step records: `(agent_message, action, action_args, observation)` tuple.
> - Full conversation saved as a multi-turn dialog where the assistant turns include tool calls and the "tool" role turns include observations.
>
> ### Training-data formatting
> - Convert trajectories into ChatML-style with `tool_calls` JSON for actions and `tool` role for observations.

ChatML format. `tool_calls` JSON for actions, `tool` role for observations. This is the 2025 de facto standard — you'll see the same format in Qwen-2.5-Coder, Llama 3 tool-use, Claude tool-use. Training data in this format transfers across SFT frameworks without reformatting.

The `action_args` sub-field is where tool-specific parameters live (`file_path`, `command`, `url`, `selector`). The scaffold validates args against the action's schema before dispatching to the runtime.

## Docker sandboxing

From the source (line 48):

> - **Environment:** Docker Linux sandbox with filesystem + shell + Python + optional browser (Playwright).

Every OpenHands trajectory runs in a Docker sandbox. Filesystem, shell, Python (IPython kernel), Playwright browser. This is the runtime abstraction that lets arbitrary bash commands be safe — even `rm -rf /` only destroys the container, not the host.

Playwright is the browser automation library of choice (not Selenium, not puppeteer). It supports headless Chromium and gives the scaffold programmatic control over click / type / scroll at the DOM level.

## OpenHands-LM-32B

From the source (line 61):

> - OpenHands-LM-32B (Qwen-2.5-Coder base + OpenHands-SFT): SWE-Bench Verified 37.2%, strong on HumanEval+.

Mar 2025 agent-specialist model from the OpenHands team. Qwen-2.5-Coder 32B base + OpenHands SFT on collected trajectories. 37.2% on SWE-Bench Verified is competitive with SWE-Gym-32B (32.0%) and below SWE-RL-70B (41.0%) — but uses a smaller model than SWE-RL and a different training recipe than SWE-Gym.

The delta vs SWE-Gym is small but real: OpenHands-LM-32B was trained on a broader trajectory mix (not just SWE-Gym tasks) and shows better generalization to tasks outside SWE-Bench.

## Scaffold versioning — the subtle hazard

From the source (line 66):

> - **Environment / scaffold versioning:** OpenHands evolves quickly; trajectories from an old version may be incompatible with a new scaffold.

The action vocab has changed across OpenHands releases (SWE-agent → OpenDevin → OpenHands; `edit` → `str_replace_editor`; new sub-operations added). Trajectory datasets from 2024 may use `edit` actions the 2026 scaffold doesn't recognize. Either re-format old data to match the new vocab (lossy — some operations don't round-trip) or pin the scaffold version alongside the dataset.

For long-lived corpora, scaffold versioning is as important as Docker-image pinning ([[excerpts/webarena-data]]). Both are operational details the papers tend to understate.

## What to take from OpenHands for ch-27

1. **Six top-level actions cover three task types.** file-edit + bash + IPython + browse + think + finish.
2. **ChatML with `tool_calls` JSON is the de facto format.** Same format Qwen/Llama/Claude use.
3. **Docker sandbox makes arbitrary bash safe.** Destroying the sandbox is free; destroying the host is bad.
4. **Playwright for browser, not Selenium.** Better DOM access, better async support.
5. **Scaffold versioning is a real hazard.** Pin the scaffold alongside the dataset.

## Connections

- [[ch-27]] §2.2, §5 — OpenHands is the scaffold underlying both SWE-Gym trajectories and the action-space table's Repo/Terminal rows.
- [[excerpts/swe-gym]] — SWE-Gym's rollouts use the OpenHands scaffold.
- [[excerpts/webarena-data]] — WebArena trajectories are collected via scaffolds that became OpenHands's browser sub-agent.
