<!-- scope: released Terminal-Bench 2.0 agent trajectories (52,104 trials) as a terminal-agent data source
     see-also: [[openhands-data]], [[magnetic-one]]
-->

# Terminal-Bench 2.0 Trajectories (`yoonholee/terminalbench-trajectories`)
- **Core Insight:** The dataset contains 52,104 trials over the 89 Terminal-Bench 2.0 tasks with an overall pass rate of 39.6%, so about 60% of the rows are failed trajectories carrying a binary outcome label (dataset summary table).
- **Guideline:** When using this dataset for post-training, filter on `reward == 1` and on rows that actually contain step traces, because only 34,462 of the 52,104 trials have a populated `steps` field (dataset summary table).
- **Authors:** Hugging Face account `yoonholee`; the dataset card names no author list and no affiliation.
- **Year:** 2026 (dataset card; no publication date stated on the card)
- **URL:** https://huggingface.co/datasets/yoonholee/terminalbench-trajectories
- **Source type:** model/dataset card
- **Relevant topics:** terminal agents, agent trajectories, CLI tasks, failure analysis, benchmark-derived data

## Summary
The dataset releases full agent traces from Terminal-Bench 2.0, a benchmark that evaluates coding agents on terminal tasks. Each row is one trial: one agent scaffold with one underlying model attempting one task, with per-step messages, tool calls, and observations, plus a binary reward, wall-clock duration, token counts, and cost where the source leaderboard reported them. The card states the trajectories were scraped from tbench.ai using publicly available leaderboard data, and the license is Apache 2.0.

## Key Contributions
- Releases per-step traces rather than only leaderboard scores for a shared set of 89 tasks.
- Covers 109 agent/model combinations built from 26 scaffolds and 49 underlying models, so the same task can be compared across scaffolds.
- Labels every trial with a binary outcome, which makes the corpus usable for verifier or outcome-reward-model training as well as for imitation.
- Reports per-scaffold pass rates and a separate 30-task hard split, which allows difficulty-conditioned filtering.

## Key Figures/Tables to Study
- Dataset summary table: totals, pass rate, and step statistics.
- Schema table: the 14 columns and which of them are null for some agents.
- Step format block: the `src` / `msg` / `tools` / `obs` structure, with observations truncated to 5,000 characters.
- Scaffolds table: trajectory counts and pass rates per scaffold.
- Hard-split table: pass / total per agent over the 30 tasks tagged `difficulty:hard`.

## Technical Details
- 52,104 trajectories; 89 tasks; 109 agent/model combinations; 26 scaffolds; 49 underlying models (dataset summary table).
- Overall pass rate 39.6%; median 21 steps per trajectory; mean 47.1 steps; typically 5 trials per (task, agent) pair (dataset summary table).
- 34,462 trials have trajectory steps, so 17,642 rows carry outcome and cost metadata without a trace (dataset summary table).
- `reward` is `int64`, 1 if the agent solved the task and 0 otherwise (schema table). `duration_seconds`, `input_tokens`, `output_tokens`, `cache_tokens`, and `cost_cents` are null for some agents (schema table).
- `steps` is a JSON string; each step has `src` in {user, agent, system}, `msg`, an optional `tools` list of `{fn, cmd}` entries, and an optional `obs` truncated to 5,000 characters (step format section).
- Per-scaffold counts and pass rates (scaffolds table): terminus-2 17,431 trajectories at 33.6%; mini-swe-agent 6,663 at 22.8%; openhands 6,198 at 28.2%; codex 3,532 at 45.2%; claude-code 3,092 at 40.3%; Factory Droid 2,224 at 67.3%; gemini-cli 1,766 at 33.5%; mux 1,068 at 66.2%; letta-code 1,335 at 56.2%; goose 1,332 at 44.2%; ruley 890 at 66.3%; terminus-3-3 887 at 74.9%. Fourteen further scaffolds contribute one combination each.
- Highest full-set pass rates (top-agents table): forge with gemini-3.1-pro-preview 78.4%; Factory Droid with gpt-5.3-codex 77.3%; simple_codex with gpt-5.3-codex 74.9%; terminus-3-3 with claude-opus-4-6 74.9%.
- Hard split: 30 tasks tagged `difficulty:hard`; best reported entry is terminus-3-3 with claude-opus-4-6 at 99/147 (67.3%) (hard-split table).
- Format is Parquet; total file size 221 MB; license Apache 2.0 (files listing and license section).
- Sampling of the dataset preview shows trial timestamps between 2025-12-09 and 2026-02-07; the card itself states no collection window.

## Findings relevant to negative feedback and agentic training
- With a 39.6% overall pass rate, roughly 60% of rows are failures with an explicit label. This makes the corpus primarily a source of negatives and of failure analysis; the card describes no synthesis or filtering pipeline and provides no per-step correctness annotation.
- Pass rate varies by scaffold from 22.8% (mini-swe-agent) to 74.9% (terminus-3-3), so scaffold identity is confounded with outcome and an unfiltered mixture of scaffolds is not a uniform behavior distribution.
- Observations are truncated to 5,000 characters, so trajectories reconstructed from this dataset do not reproduce the full environment feedback the agent saw.
- Data are scraped from a public leaderboard rather than generated under a controlled sampling protocol, so temperature, prompt versions, and harness versions vary across rows and are not recorded.

## Connections
- The benchmark itself: Merrill et al., "Terminal-Bench: Benchmarking Agents on Hard, Realistic Tasks in Command Line Interfaces", arXiv:2601.11868 (2026-01), which defines the 89 tasks and reports frontier agents below 65%. The library has no card for it.
- [[openhands-data]] — one of the scaffolds represented here (6,198 trajectories) and a separate agent-trajectory corpus.
- [[magnetic-one]] — another multi-agent system evaluated on terminal-style tasks.

## Verification
- Checked on 2026-09-18 against: https://huggingface.co/datasets/yoonholee/terminalbench-trajectories (dataset card and Data Studio preview as served on that date)
- Corrections to the previous card version:
  - Title "Terminal-Bench Trajectories" → the card's own heading is "Terminal-Bench 2.0 Trajectories"; the artifact is the Hugging Face dataset `yoonholee/terminalbench-trajectories`.
  - "Authors: Terminal-Bench ecosystem / trajectory dataset maintainers" → the dataset is published by the Hugging Face account `yoonholee`; no author list is given.
  - "Dataset card reports tens of thousands of trajectories over the Terminal-Bench task set" → 52,104 trials over 89 tasks, of which 34,462 contain step traces.
  - "Each row stores a full trial including messages, tool calls, and observations" → true only for the 34,462 rows with a populated `steps` field, and observations are truncated to 5,000 characters.
  - "Covers many agent/model combinations" → 109 combinations from 26 scaffolds and 49 models.
- Removed as unsupported by the source: "A hard terminal benchmark becomes much more valuable for training once the full step-by-step agent traces are released" as a stated finding (the card reports contents, not a training result); "benchmark traces can double as high-value post-training data" (no training experiment is reported on this card); "valuable for imitation, trajectory filtering, and agent-behavior analysis" as a source claim.
- Not reported by the source: any training or fine-tuning result using the dataset; per-step correctness or error-type annotations; decoding parameters or harness versions per trial; the exact scrape window.
