<!-- scope: agentic trajectory synthesis — rule-based RL on software-engineering tasks with trajectories
     deps: [[grpo]]
     see-also: [[swe-gym]], [[agentinstruct]], [[openhands-data]]
-->

# SWE-RL: Advancing LLM Reasoning via Reinforcement Learning on Open Software Evolution
- **Core Insight:** Real software-engineering trajectories are a rich, automatically-verifiable RL substrate: scrape millions of (issue, pull-request) pairs from open source, use a simple similarity-based rule reward (Python difflib) against the human PR diff, and run GRPO — no unit-tests or executable benchmarks required for training.
- **Guideline:** For agentic SWE training via RL, skip expensive test-execution verification during rollout; a lightweight rule-based textual-similarity reward against ground-truth patches is sufficient signal for GRPO, enabling training on millions of issues cheaply.
- **Authors:** Yuxiang Wei, Olivier Duchenne, Jade Copet, Quentin Carbonneaux, Lingming Zhang, Daniel Fried, Gabriel Synnaeve, Rishabh Singh, Sida I. Wang (Meta FAIR + UIUC + CMU)
- **Year:** 2025
- **URL:** https://arxiv.org/abs/2502.18449
- **Relevant topics:** agentic RL, software engineering, rule-based reward, SWE-Bench, GRPO

## Abstract
SWE-RL trains Llama-3.1-70B (and later variants) as a coding agent using RL over rule-based rewards derived from real GitHub issue→PR pairs. The **Llama3-SWE-RL-70B** model achieves 41.0% on SWE-Bench Verified — the best score among open medium-sized models at release (early 2025) — using *only* rule-based similarity rewards, no unit-test execution, no human feedback. SWE-RL introduces (a) a large SWE trajectory corpus, (b) a simple rule-based reward, (c) empirical evidence that RL on SWE tasks transfers to reasoning outside software (math, function calling) — the paper's most provocative finding.

## Key Contributions
- **Scalable rule-based SWE reward** using Python `difflib.SequenceMatcher.ratio()` between predicted patch and ground-truth patch.
- **11M GitHub issue-PR-code triplets** scraped and prepared as RL training data.
- **Llama3-SWE-RL-70B** reaches 41.0% on SWE-Bench Verified — open SOTA at release.
- **Out-of-domain transfer:** model gains on math/function-calling/reasoning benchmarks despite training only on SWE tasks.

## Synthesis pipeline (REQUIRED — concrete, modality-specific)

### Data collection
- **Seed input:** GitHub Archive BigQuery export of all issues and PRs with code changes from top-starred repositories.
- **Pair extraction:** for each merged PR, extract (issue_description, code_context, ground_truth_patch).
- **Filters:**
  - PR merged and has linked issue.
  - Patch modifies ≤ 10 files and ≤ 500 lines (tractable for 7B context window).
  - Language is Python (primary) or other popular languages.
  - Dedup near-identical issues with MinHash.
- **Output:** ~11M (issue, context, patch) triples.

### RL training
- **Rollout format:** agent receives issue text + relevant code context; must emit a unified-diff patch.
- **Reward:** `r = difflib.SequenceMatcher(None, predicted_patch, ground_truth_patch).ratio()` — a scalar in [0, 1]. Clamped / shaped slightly (authors experiment with binary thresholding vs continuous reward; continuous wins).
- **Algorithm:** **GRPO** with group size G=8, KL coefficient β=0.02, LR 1e-6.
- **No execution:** unlike SWE-Gym / SWE-agent setups, the training loop does not run unit tests — this is a compute-saving design choice.

### Evaluation
- SWE-Bench Verified (500 real GitHub issues with hidden tests) — tests run only at eval time.
- BIG-Bench Hard, MATH, HumanEval+, MBPP+ — evaluated to measure out-of-domain transfer.

- **Output trajectory shape:** issue + code context ~4K–20K tokens; agent output is a unified diff (100–1000 tokens).
- **Teacher model(s):** no teacher; RL only. Base = Llama-3.1-70B-Instruct.
- **Cost / compute:** disclosed as ~1M H100-hours for the 70B RL run.

## Modality-specific technical details (REQUIRED — agentic)
- **Environment:** GitHub repository snapshots + issue text + human PR diff as gold. No sandboxed code execution during training; only at eval.
- **Action space:** emit a unified-diff patch. No multi-step tool calls (single-shot patch generation).
- **Trajectory length:** mostly single-turn (issue → patch). A chain-of-thought reasoning segment precedes the patch.
- **Success criterion:** training = similarity ratio; eval = test suite pass (SWE-Bench-Verified).
- **Data scale:** 11M issue-PR triples used for RL rollouts.
- **Key ablation:** rule-based reward vs execution-based reward — similarity reward wins because it provides dense signal on every sample, whereas execution reward is sparse (many tests fail for unrelated reasons).

## Quality / diversity evaluation
- **Llama3-SWE-RL-70B: 41.0% SWE-Bench Verified** (beats DeepSeek-Coder-V2-Instruct 18.0%, matches SWE-Gym-32B).
- **Out-of-domain transfer** — Llama3-SWE-RL-70B vs Llama-3.1-70B-Instruct baseline:
  - HumanEval+: +6 points.
  - MATH: +4 points.
  - BIG-Bench Hard: +3 points.
  - Multiple reasoning benchmarks show smaller but consistent gains.
- **Surprising result:** RL on software-engineering trajectories transfers positively to math and general reasoning — authors hypothesize it teaches "long-horizon grounded planning" transferable across domains.

## Risks + gotchas
- **Similarity reward can be gamed:** a predicted patch that copies context verbatim gets partial credit without fixing anything. Authors mitigate with format filters (must be diff, must modify code).
- **Data decontamination:** SWE-Bench Verified comes from the same GitHub universe; authors decontaminate by date (training data predates SWE-Bench issues) and commit-hash filter.
- **Single-turn limitation:** SWE-RL does not train multi-turn agent behavior (file navigation, test running). Paper notes this as future work.
- **Narrow languages:** training corpus is Python-dominant; transfer to Go/Rust is untested.

## Connections
- Complementary to execution-based SWE training: [[swe-gym]] (SWE-Gym uses full execution environments).
- Algorithm: [[grpo]].
- Agent-trajectory lineage: [[agentinstruct]], [[agent-flan]], [[agenttuning]].
- Eval: SWE-Bench Verified.
- Out-of-domain transfer finding ties to [[transferability-of-llm-reasoning]], [[front-loading-reasoning]].
