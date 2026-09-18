<!-- scope: SWE-rebench (arXiv:2505.20411, May 2025; NeurIPS 2025 D&B) — automated GitHub mining of 21,336 executable Python SWE tasks for agent RL, plus a 294-task date-tracked benchmark with fixed ReAct scaffold, 5 runs, SEM and pass@5
     deps: [[swe-gym]]
     see-also: [[swe-bench-illusion]], [[swe-smith]], [[r2e-gym]], [[kimi-dev]], [[deepswe]], [[livecodebench]]
-->

# SWE-rebench: An Automated Pipeline for Task Collection and Decontaminated Evaluation of Software Engineering Agents
- **Core Insight:** A four-stage automated pipeline (issue–PR mining, LLM-generated installation recipes, execution-based validation, LLM quality labels) produced 21,336 executable Python SWE tasks from 3,468 repositories; on its March–April 2025 benchmark slice, DeepSeek-V3-0324 and DeepSeek-V3-1226 resolve 21.3% and 21.9% although they score 39.7% and 35.2% on SWE-bench Verified (§2, §3.3, Table 2).
- **Guideline:** When building SWE tasks for agent RL, keep only tasks whose test patch fails before and passes after the gold patch, and rerun tests to drop flaky instances, because an underspecified issue or a flawed test produces trajectories that look like failures and penalize the agent incorrectly (§2.3, §2.4, App. B.5). The paper trains no agent, so the effect of these filters on RL results is not measured.
- **Authors:** Ibragim Badertdinov, Alexander Golubev, Maksim Nekrashevich, Anton Shevtsov, Simon Karasik, Andrei Andriushchenko, et al. (Nebius)
- **Year:** 2025 (arXiv v1 2025-05-26; v2 2025-11-04; NeurIPS 2025 Datasets and Benchmarks Track)
- **URL:** https://arxiv.org/abs/2505.20411 ; dataset https://huggingface.co/datasets/nebius/SWE-rebench ; leaderboard https://swe-rebench.com/leaderboard
- **Source type:** paper (dataset and benchmark)
- **Relevant topics:** SWE agent RL environments, executable task mining, automated environment setup, benchmark contamination, agent evaluation variance, task quality labeling

## Abstract
Interactive SWE training data, in which agents act in development environments and adapt to execution results, is scarce: existing datasets are one-shot code generation or small manually curated collections. Static benchmarks also become outdated through contamination. The authors present an automated, scalable pipeline that continuously extracts interactive SWE tasks from GitHub. It yields SWE-rebench, a public dataset of over 21,000 interactive Python tasks suitable for RL of SWE agents. A continuous supply of fresh tasks is used to build a contamination-free agentic SWE benchmark. Comparing results on this benchmark with SWE-bench Verified, the authors report that the performance of some models might be inflated by contamination.

## Key Contributions
- A fully automated pipeline for environment configuration, build setup, and test validation of GitHub-mined tasks (§2).
- SWE-rebench dataset: 21,336 tasks with pinned dependencies and LLM-predicted quality labels (§2.4; App. K).
- A continuously updated leaderboard with a fixed scaffold, repeated runs, and contamination flags based on issue creation dates (§3.2).
- Refinements to SWE-bench construction: head-commit patch diffs, deleted-test filtering, full tracebacks, dependency pinning (App. G).

## Key Figures/Tables to Study
- Fig. 1 (pipeline); Table 1 (January vs March–April 2025 slices); Table 2 (SWE-bench Verified vs SWE-rebench); App. C Table 3 (installation methods); App. F Table 4 (quality-label classifier); App. L Table 6 (funnel); App. M Table 7 (dataset statistics); App. H (benchmark filters).

## Technical Details
**Stage 1: task collection (§2.1)**
- Sources: GitHub Archive (issues, PRs, metadata) and full-history clones.
- About 450,000 PRs linked to issues created before May 1, 2025, from over 30,000 permissively licensed repositories where Python is over 75% of lines of code.
- Filters: resolved issue; PR merged into main; PR linked to one issue; issue longer than 10 characters; PR changes tests and non-test code; 1–15 files changed. About 153,400 candidates remain. Each PR is split into a solution patch (non-test files) and a test patch.

**Stage 2: installation recipes (§2.2; App. B, C)**
- Tasks are grouped by major.minor version from `git tag` (available for about 95%); each group uses the most recent base commit.
- An agentless LLM process finds installation-relevant files and writes a JSON recipe; Qwen2.5-72B-Instruct produces up to 3 candidate recipes per task and revises a recipe from error logs. At least one task gets a working recipe in 31% of repositories.
- Validation on 18 SWE-bench instances: agentless with 1 / 3 / 10 candidates configures 6 / 8 / 9; an interactive agent with 1 trial configures 8, at higher compute cost (App. C Table 3).

**Stage 3: execution validation (§2.3; App. B.5)**
- Valid only if (1) at least one test-patch test fails before the solution patch, (2) all initially failing test-patch tests pass after it, and (3) initially passing test-patch tests still pass.
- Containers built with buildah on tmpfs; dependency versions frozen with `pip freeze` and `conda env export`. Tests are run multiple times and instances failing in any run are excluded.

**Stage 4: quality labels (§2.4; App. F Table 4)**
- Qwen2.5-72B-Instruct is fine-tuned on over 3,800 SWE-bench Verified annotations (issue, gold patch, test patch as input), with a 75/25 split and 413 validation examples, to predict issue clarity, task complexity, and test-patch correctness separately.
- Validation accuracy (fine-tuned vs base): task complexity 81% vs 68% (weighted F1 0.82); test-patch correctness 67% vs 60% (F1 0.65); issue clarity 79% vs 80% (F1 0.76; recall on the high-score class 0.20). Labels are shipped as metadata for user filtering.

**Funnel and statistics (App. L Table 6; App. M Table 7)**
- About 10M PRs, 6M issues, 32K repositories → about 450K candidates (about 5% of PRs) → about 150K filtered (about 33%) → about 21K valid after install and validation (about 14%).
- Mean (p50) per task: issue 141.67 (91) words; files edited 3.46 (2); lines edited 142.17 (37); fail-to-pass tests 14.56 (2); total tests 105.43 (31). The App. M prose gives "added lines" mean about 97 with p75 76, which does not match Table 7 (lines edited p75 112).

**Benchmark and protocol (§3; App. H, J)**
- 294 tasks from 169 repositories. Filters: no AttributeError or ImportError before patching; ≤ 3 files; solution patch ≤ 500 words; English issue of 16–1,000 words; issue created in 2025; LLM difficulty label < 3; ≤ 50 fail-to-pass tests (App. H).
- Every model runs in the same minimal ReAct scaffold with identical prompts, developer-recommended generation settings, 128K context (unless shorter), and text commands instead of native function calling; 5 runs per model with SEM and pass@5 (§3.2). Evaluations that include issues created before a model's release are marked as potentially contaminated (§3.2).
- Open models served with vLLM on 2 nodes of 8×H200; one DeepSeek-V3 run over 294 tasks takes about 7 hours (App. J).

**Results (§3.3; Tables 1–2; App. F)**
- GPT-4.1 is the only model with a noticeable decline from the January to the March–April slice: 31.1% → 26.7% resolved (Table 1).
- SWE-bench Verified vs March–April slice, resolved: DeepSeek-V3-0324 39.7 vs 21.3; DeepSeek-V3-1226 35.2 vs 21.9; Llama-3.3-70B 18.1 vs 11.2; Llama-4-Maverick 16.0 vs 12.2; Qwen2.5-72B 11.3 vs 9.3; Qwen2.5-Coder-32B 4.9 vs 3.2 (Table 2).
- Interpretation (authors): the two DeepSeek-V3 versions are similar on SWE-rebench but differ on SWE-bench Verified, which "may suggest potential contamination effects on the older benchmark" (§3.3).
- Llama-4-Maverick: pass@5 27.6 at 12.2% mean resolved (low run-to-run reliability). Qwen2.5-Coder-32B hallucinates environment responses and loops on formatting errors. Qwen3 think and no-think modes score similarly (§3.3).
- DeepSeek-V3-0324 on January–July 2025 leaderboard tasks by files changed: 1 file 28.6% (n = 201), 2 files 20.6% (n = 144), ≥ 3 files 17.5% (n = 64) (App. F).

## Findings relevant to generality, negative feedback, and agentic training
- Evaluation confounds named by the authors: SWE-bench public since late 2023; scaffolds tuned on SWE-bench subsets ("implicit overfitting"); pass@N reported as pass@1; best-of-several-runs reporting (§3.1).
- Negative feedback (negative as gradient in RL, not measured here): failed trajectories caused by unsolvable issues or wrong tests are false negatives for the agent; mitigations are execution validation, flaky-test exclusion, and quality labels (§2.3, §2.4, App. B.5). The false-negative rate is not reported.
- Limits: automated labels are imperfect and some tasks may be unsolvable from the issue alone, which can lower absolute success rates versus manually curated benchmarks; Python only; the installation prompts were tuned on 18 repositories (§4).

## Connections
- [[swe-bench-illusion]] — memorization probes on SWE-Bench Verified; its "SWE-Bench Extra" set is not this team's earlier SWE-bench Extra dataset.
- [[swe-gym]] — manually configured SWE environments cited as limited in scale (§1).
- [[swe-smith]] — synthetic bug injection, cited as an alternative way to scale tasks (App. A); [[r2e-gym]] — another SWE environment construction method.
- [[deepswe]] — RL-trained SWE agent cited in App. A; [[kimi-dev]] — cited as work that uses the SWE-rebench dataset (App. A); [[skyrl-agent]] — later SkyRL agent work (the paper cites the earlier SkyRL-v0).
- [[swe-rl]] — RL on software-evolution data, cited in §1.
- [[livecodebench]] — continuously updated code benchmark cited in App. A; [[swe-bench-pro]] — later contamination-resistant SWE benchmark.
- [[agentic-benchmark-checklist]], [[holistic-agent-leaderboard]] — agent-evaluation reporting practices related to §3.1.

## Verification
- Created on 2026-09-14 from https://arxiv.org/abs/2505.20411 (arXiv v2, 2025-11-04, including Appendices A–M; v1 2025-05-26 and venue read on the abs page).
- Audit claims not found in the source: none.
- Not reported by the source: training hyperparameters for the quality-label fine-tune, any agent trained on the dataset, false-negative rate of task validation, per-model generation settings.
