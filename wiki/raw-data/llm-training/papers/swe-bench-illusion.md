<!-- scope: The SWE-Bench Illusion (arXiv:2506.12286, Jun 2025) — context-free file-path identification, function reproduction, and prefix completion probes showing SWE-Bench Verified-specific memorization across ten OpenAI and Anthropic models
     deps: [[data-contamination-survey]]
     see-also: [[swe-rebench]], [[swe-bench-pro]], [[livecodebench]], [[gsm1k]], [[quantifying-memorization]]
-->

# The SWE-Bench Illusion: When State-of-the-Art LLMs Remember Instead of Reason
- **Core Insight:** Given only the repository name and the issue text, ten OpenAI and Anthropic models named a file changed by the gold patch in 60–76% of SWE-Bench Verified instances, but in under 53% of 245 SWE-Bench-style tasks from seven popular repositories that are not in SWE-Bench (§4.1.1–4.1.2, Fig. 8).
- **Guideline:** When SWE-Bench Verified scores are used as evidence of general coding ability, also report results on tasks from repositories outside SWE-Bench or on issues created after the benchmark was built, because the same models score lower on those sets in file-path accuracy (<53% vs up to 76%) and in function-reproduction 5-gram overlap (maxima 13.9–18.2% vs 34.9%) (§4.1, §4.3).
- **Authors:** Shanchao Liang (Purdue University), Spandan Garg, Roshanak Zilouchian Moghaddam (Microsoft)
- **Year:** 2025 (arXiv v1 2025-06-14; v4 2025-12-01; no venue listed, the PDF carries an unfilled ACM template header)
- **URL:** https://arxiv.org/abs/2506.12286
- **Source type:** paper
- **Relevant topics:** benchmark contamination, memorization, SWE-Bench Verified, agentic coding evaluation, cross-benchmark generalization, n-gram overlap metrics

## Abstract
SWE-Bench Verified is a common benchmark for resolving real GitHub issues, and recent models score high on it. The authors argue that current protocols may overstate capability and that generalizable problem solving must be separated from memorized patterns. They introduce diagnostic tasks: file-path identification from the issue description alone, and reproduction of the ground-truth function from the current file context and the issue. Models reach up to 76% file-path accuracy on SWE-Bench Verified without repository structure, but only up to 53% on repositories not in SWE-Bench. Function reproduction shows up to 35% consecutive 5-gram overlap on SWE-Bench Verified and Full versus up to 18% on other benchmarks. The authors conclude that existing results may be inflated and that contamination-resistant benchmarks are needed.

## Key Contributions
- Cross-benchmark analysis: performance gaps between comparable benchmarks and repositories are used as a proxy for memorization, without access to training data or model internals (§1, §5.2).
- Three diagnostic tasks: context-free file-path identification, function reproduction without specification, and prefix completion (§2.1–2.3). The abstract names only the first two.
- Evidence from ten models of two memorization types: instance-specific (within SWE-Bench repositories) and repository-bias (SWE-Bench repositories vs other popular repositories) (§4.1).

## Key Figures/Tables to Study
- Fig. 8 (file-path accuracy by benchmark and model); Fig. 9 (filtered accuracy); Fig. 10 (5-gram overlap); Fig. 11 (Δ5, gold vs buggy similarity); Table 1 (repository size and issue length); Table 2 (instances whose issue text mentions the gold path); Table 3 (prefix-completion verbatim match); Figs. 6–7 (prompts).

## Technical Details
**Diagnostic tasks (§2, §3.5)**
1. File-path identification: the prompt contains the repository name and the issue text, with no code, file tree, or metadata; the model writes a DISCUSSION and one file path. A prediction is correct if it equals any file path in the gold patch (§2.1, Fig. 6).
2. Function reproduction: the model receives the issue and the file with every gold-patch-modified function removed; hints give only the function names and file paths; generation is single-round with no repository access (§2.2, §3.5.2, Fig. 7).
3. Prefix completion: the lines before each modified snippet are given; the continuation is compared with the gold code for up to N lines, where N is the number of modified lines (§2.3).

**Metrics (§3.3, §4.5)**
- Accuracy = exact-match instances / all instances × 100 (Eq. 1). Filtered accuracy uses only instances whose issue text contains no file path or import statement, found by heuristics (§3.3).
- 5-gram overlap = matched 5-grams / generated 5-grams × 100, where a 5-gram is 5 consecutive tokens and matching respects each 5-gram's count in the gold code (Eq. 2).
- Instance-level verbatim match: share of instances where at least one generated hunk equals the gold hunk (§3.3).
- Δ5 = overlap5(f̂, f_GT) − overlap5(f̂, f_buggy), where f̂ is the generated function, f_GT the patched function, and f_buggy the pre-patch function; positive values mean closer to the fix (Eq. 3).

**Evaluation sets (§3.1–3.2; Tables 1–2)**
- SWE-Bench Verified: 500 instances from 12 Python repositories; 763.5 files and 451.2-token issues on average; 135 (27.0%) issues mention a gold path.
- Full-SWE-Bench: 200 random instances outside Verified; 42 (21.0%) mention a path.
- SWE-Bench Extra (called "SWE-Repo Tasks" in Table 2 and §4.1.1): 217 recent issues from SWE-Bench repositories, mostly created after the SWE-Bench cutoff; 57 (26.3%). This is the authors' own collection.
- Outside-Repo Tasks: 245 instances from jupyter/notebook, celery, aiohttp, scipy, numpy, pytorch, pandas; 63 (25.7%).
- RefactorBench: 39 of 100 refactoring tasks from 9 Python repositories; 1,149.2 files, 14.6-token instructions; 1 (2.6%).
- SWE-Bench-C#: internal set of 75 tasks from 11 C# repositories; 716.7 files, 586.2-token issues; 11 (14.7%).

**Models and decoding (§3.4).** GPT-4o (2024-05 and 2024-08 snapshots), GPT-4.1 (2025-04-14), o3 (2025-04-16), o3-mini (2025-01-31), o4-mini (2025-04-16), Claude 3.5 Sonnet, 3.7 Sonnet, Sonnet 4, Opus 4. Maximum 2,048 completion tokens, 4,096 for o3, o3-mini, and o4-mini; provider-default sampling. Function reproduction was run for o3-mini only among OpenAI reasoning models (§4.3, footnote 1); prefix completion covers eight models (Table 3).

**Results**
- File path (§4.1.1–4.1.2): SWE-Bench Verified 60–76%, Full-SWE-Bench 57–71%, SWE-Repo tasks 50–68%, Outside-Repo below 53%. o3 reaches 76% on Verified (§1). All vendors show the ordering Verified > Full > SWE-Repo > {C#, RefactorBench, Outside-Repo} (§4.1.3). The drop to external repositories is up to 47 percentage points (§6).
- Filtered accuracy: the same ordering and the drop to external repositories remain after removing issues that mention paths or imports (§4.2, Fig. 9).
- Function reproduction maxima (§4.3): Verified 34.9%, Full 28.7%, SWE-Bench Extra 18.2%, RefactorBench 18.1%, Outside-Repo 13.9%. The text describes 18.2% as "outside the SWE-Bench ecosystem", although SWE-Bench Extra uses SWE-Bench repositories.
- Prefix completion (§4.4, Table 3): verbatim match from 11.7% (o3-mini) to 31.6% (Claude 4 Opus); Claude 3.5 Sonnet 12.1%, 3.7 Sonnet 12.3%, Sonnet 4 21.4%; GPT family 17.4–18.4%.
- Δ5 (§4.5, Fig. 11): Claude models are positive on Verified, Full, and Outside-Repo; OpenAI models are generally below +2 pp; on SWE-Bench Extra every model except Claude 3.5 Sonnet and GPT-4.1 is negative; on C# the GPT-4 family is about +3 to +5 pp.

## Findings relevant to generality and agentic training
- Measurement error (Result, single study): SWE-Bench Verified accuracy on subtasks is higher than on comparable tasks from the same repositories created later and from other popular repositories, for every vendor tested (§4.1.3).
- Interpretation (authors): the within-repository decay indicates instance-specific memorization of the curated Verified set; the drop on outside repositories indicates repository-bias memorization of the 12 SWE-Bench repositories (§4.1, §6).
- Confounds stated by the authors: RefactorBench instructions average 14.6 tokens, and part of the external drop is attributed to such differences (§4.1.2, Table 1). N-gram similarity is a noisy indicator because correct solutions share text with the gold patch (§5.2).
- Scope: the probes measure subtasks (localization, reproduction), not end-to-end agent resolve rates; only closed API models are tested; no training data is inspected.
- Recommendations (authors): temporal controls, cross-repository validation, and cross-benchmark analysis (§6).

## Connections
- [[swe-rebench]] — continuously collected, date-tracked SWE tasks with contamination flags; its authors' earlier "SWE-bench Extra" dataset is a different artifact from this paper's 217-task set.
- [[swe-bench-pro]] — later SWE benchmark built with contamination resistance as a goal.
- [[livecodebench]], [[livebench]], [[matharena]], [[gsm1k]] — fresh or date-filtered evaluation sets in other domains.
- [[data-contamination-survey]], [[rephrased-samples-contamination]], [[proving-test-set-contamination]], [[evaluation-data-contamination-contam]] — contamination detection and its effect on scores.
- [[quantifying-memorization]] — verbatim memorization measurement in language models.
- [[reasoning-or-memorization-rl-contamination]] — contamination as a confound in RL results.
- [[swe-gym]], [[swe-smith]], [[r2e-gym]], [[agent-data-protocol]], [[qwen3-coder-next]], [[glm-5]] — training work that reports SWE-bench Verified; this paper does not evaluate them.

## Verification
- Created on 2026-09-14 from https://arxiv.org/abs/2506.12286 (arXiv v4, 2025-12-01; version history v1 2025-06-14, v2 2025-06-27, v3 2025-08-06, v4 2025-12-01 read on the abs page).
- Audit claims not found in the source: "SWE-bench Verified gains after agentic mid-training (GLM-5, Qwen3-Coder-Next, ADP) may partly reflect memorization" (the paper discusses none of these models or mid-training); "two diagnostic probes" → the v4 body defines three tasks, adding prefix completion (§2.3); "consecutive 5-gram accuracy" → the metric is the 5-gram overlap ratio (Eq. 2); function reproduction "from only the current file context and the issue" → the prompt also names the removed functions and their paths (§3.5.2).
- Not reported by the source: end-to-end SWE-Bench resolve rates for the tested models, per-model numbers in the text for most figure bars (Figs. 8–11 print them only as chart labels), open-weight model results.
