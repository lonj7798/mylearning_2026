---
chapter: ch-51a
course: llm-training
phase: read
excerpt_of: "Hugging Face dataset card, yoonholee/terminalbench-trajectories (Terminal-Bench 2.0 Trajectories)"
source_url: https://huggingface.co/datasets/yoonholee/terminalbench-trajectories
created_at: "2026-09-15"
note: "The library card [[terminal-bench-trajectories]] predates the 2026-09 card standard, has no Verification section, and states only that the dataset holds 'tens of thousands of trajectories'. This excerpt records the dataset-card statistics read directly on 2026-09-15 so the chapter can cite exact numbers."
---

# Excerpt: Terminal-Bench 2.0 Trajectories — per-scaffold pass rates

Used by [[read]] §5 and §6. Read from the dataset card on 2026-09-15.

## Dataset summary (dataset card, "Dataset summary" table)

| Statistic | Value |
|---|---|
| Total trajectories | 52,104 |
| Tasks | 89 |
| Agent/model combinations | 109 |
| Scaffolds (agents) | 26 |
| Underlying models | 49 |
| Overall pass rate | 39.6% |
| Median steps per trajectory | 21 |
| Mean steps per trajectory | 47.1 |
| Trials with trajectory steps | 34,462 |
| Trials per (task, agent) | typically 5 |

"Each row is one trial: an agent attempting a task, with the complete step-by-step trace of messages, tool calls,
and observations." Columns include `task_name`, `agent`, `model`, `reward` (1/0), `duration_seconds`,
`input_tokens`, `output_tokens`, `cache_tokens`, `cost_cents`, `started_at`, `ended_at`, and `steps`
(JSON list of `{src, msg, tools, obs}`; observations truncated to 5,000 characters).

## Pass rate by scaffold ("Scaffolds" table)

| Scaffold | Model combos | Trajectories | Pass rate |
|---|---|---|---|
| terminus-2 | 32 | 17,431 | 33.6% |
| mini-swe-agent | 13 | 6,663 | 22.8% |
| openhands | 12 | 6,198 | 28.2% |
| codex | 8 | 3,532 | 45.2% |
| claude-code | 7 | 3,092 | 40.3% |
| Factory Droid | 5 | 2,224 | 67.3% |
| gemini-cli | 4 | 1,766 | 33.5% |
| mux | 4 | 1,068 | 66.2% |
| letta-code | 3 | 1,335 | 56.2% |
| goose | 3 | 1,332 | 44.2% |
| ruley | 2 | 890 | 66.3% |
| terminus-3-3 | 2 | 887 | 74.9% |

Fourteen further scaffolds contribute one model combination each.

## Top agent/model combinations ("Top agents by pass rate")

forge / gemini-3.1-pro-preview 78.4%; Factory Droid / gpt-5.3-codex 77.3%; simple_codex / gpt-5.3-codex 74.9%;
terminus-3-3 / claude-opus-4-6 74.9%; terminus-3-3 / gemini-3.1-pro-preview 74.8%; judy / claude-opus-4.6 71.9%.

## Reading limit

The per-scaffold rates are **not** paired: each scaffold row aggregates a different set of models, and the
strongest models are not evaluated under every scaffold. The table therefore shows how much a leaderboard row
depends on the model-scaffold pair; it does not isolate the scaffold effect. The paired comparison for that is
Table 2 of [[terminal-bench-2]], where the same model appears under several scaffolds.
