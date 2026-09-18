<!-- scope: Agentic Benchmark Checklist (ABC) — task validity, outcome validity, and reporting checks; audit of 10 agentic benchmarks with measured over/underestimation; CVE-Bench revision
     deps: none
     see-also: [[tau-bench]], [[webarena-data]], [[swe-bench-illusion]], [[impossiblebench]], [[holistic-agent-leaderboard]], [[null-model-cheating-benchmarks]]
-->

# Establishing Best Practices for Building Rigorous Agentic Benchmarks
- **Core Insight:** Applying the Agentic Benchmark Checklist to ten agentic benchmarks found 7 with task-validity flaws, 7 with outcome-validity flaws, and all 10 with reporting limitations; for example, an agent that returns nothing passes 38% of τ-bench Airline tasks, and an agent that overwrites SWE-Lancer's test files scores 100% (§1, Fig. 5, App. E.2, E.4).
- **Guideline:** When an agentic benchmark score is used to compare models or checkpoints, first run a do-nothing agent and an answer-enumerating agent against its grader and check that required outputs cannot be matched trivially, because these checks exposed 38-40% overestimation on τ-bench Airline and 1.4-5.2% on WebArena (§5.2, App. E.2, E.5).
- **Authors:** Yuxuan Zhu, Tengjun Jin, Yada Pruksachatkun, Andy Zhang, Shu Liu, Sasha Cui, et al., Daniel Kang (UIUC, Stanford, UC Berkeley, Yale, Princeton, MIT, Transluce, ML Commons, Amazon, UK AI Safety Institute, University of Oxford)
- **Year:** 2025 (arXiv v1 2025-07; v5 2025-08 marked "Preprint")
- **URL:** https://arxiv.org/abs/2507.02825 (code: github.com/uiuc-kang-lab/agentic-benchmarks)
- **Source type:** paper
- **Relevant topics:** agentic evaluation validity, grader flaws, outcome validity, task validity, benchmark reporting, confidence intervals, contamination measures, reward hacking in evaluation

## Abstract
Agentic benchmarks measure agents by evaluating task outcomes with specific reward designs. The authors show that many such benchmarks have flaws in task setup or reward design: SWE-bench-Verified uses insufficient test cases, and τ-bench counts empty responses as successful. These flaws can under- or overestimate agent performance by up to 100% in relative terms. The Agentic Benchmark Checklist (ABC) is synthesized from the authors' benchmark-building experience, a survey of best practices, and previously reported issues. Applied to CVE-Bench, ABC reduces performance overestimation by 33% (Abstract; §1 specifies absolute terms).

## Key Contributions
- Two validity conditions (§1, §3, Fig. 1): **task validity** — a task is solvable if and only if the agent has the target capability; **outcome validity** — the evaluation result is positive if and only if the task succeeded.
- ABC in three parts: task validity T.1-T.10 (Fig. 2), outcome validity O.a.1-O.i.1 grouped by evaluation method (Fig. 3), and benchmark reporting R.1-R.13 (Fig. 4); the source of each item is listed in App. C, Table 4.
- An assessment of ten open-source benchmarks (Table 1) with per-item reports (App. D, Tables 5-14) and experiments quantifying each new issue (§5.2, App. E).
- A case study revising CVE-Bench during development (§5.3, Fig. 6) and a model reporting example based on BIRD (App. F).

## Key Figures/Tables to Study
- **Figures 2-4** — the checklist items. Counting the listed items gives 10 task-validity, 20 outcome-validity, and 13 reporting items (derived count).
- **Figure 5** — per-benchmark scores for the three parts.
- **§5.2 list** — six newly identified issues with their measured effect.
- **Table 3** — the 17 benchmarks with evaluated capability and evaluation design.
- **App. F, Eq. 1** — confidence interval for a success rate under noisy ground truth.

## Technical Details
**Benchmark collection (§3, App. B).** Release posts, reports, and papers from OpenAI, Anthropic, Google, Meta, xAI, Mistral, DeepSeek, and Amazon for models released 1 January 2024 to 18 March 2025, plus award-winning benchmarks, gave 78 benchmarks (Table 2). Excluding fact-seeking QA and short programming tasks gave 25 agentic benchmarks (App. B); Table 3 and §3 list 17. Ten were selected for full assessment for open-source availability and coverage of capabilities and evaluation methods (§3, App. B). Each satisfied item scores 1, and part scores are averages over applicable items (§3, §5.1).

**Checklist items with examples.**
- Task validity: tool versions specified (T.1); API tools accessible and failures handled (T.2-T.3); residual state cleared between runs (T.4; KernelBench left ground-truth results in GPU memory [35]); agent isolated from ground truth (T.5); setup frozen over time (T.6); ground truth verified (T.7); each task verified solvable (T.8); an oracle solver (T.9); no exploitable vulnerabilities (T.10) (§4.1, Fig. 2).
- Outcome validity: string matching handles equivalent expressions, redundant words, negation, enumeration of all answers, and guessing (O.a-O.b); LLM judges have documented accuracy, self-consistency, human agreement, and resistance to adversarial inputs and reward hacking (O.c.1-O.c.2); unit tests are verified and measured by coverage (O.d); fuzzers cover edge cases, data types, memory layouts, and inputs the code is sensitive to (O.e); end-to-end tests cover all workflow branches and avoid flaky results (O.f); state matching includes all success states, checks irrelevant states, and is not trivially satisfiable (O.g); answer formats are specified and guessing is minimized (O.h); quality metrics resist exploitation (O.i.1) (§4.2, Fig. 3).
- Reporting: open data and harness (R.1-R.2); contamination measures such as a private held-out test set (R.3); plans to update tasks to avoid overfitting (R.4); construct and evaluation-subject statements (R.5-R.6); flaw mitigation with qualitative and quantitative impact (R.7-R.9); statistical significance (R.10); interpretation guidance (R.11); non-AI and trivial-agent baselines (R.12-R.13) (§4.3, Fig. 4).

**Assessment results (§5.1-§5.2, App. E).**
- τ-bench: intentionally unsolvable tasks are 38% of Airline and 6% of Retail; success means an unchanged database, so a do-nothing agent passes them. A do-nothing agent scores 38% (Airline) and 6.0% (Retail) pass^k or pass@k for any k; an agent that outputs all data scores 40% and 9.6% (§5.2, App. E.2). A GPT-4o-based agent resolves 35% of τ-bench-Airline tasks [86] (§1).
- WebArena: substring matching ignores extraneous content, and the LLM judge accepts an empty reply for "N/A" tasks; overestimation 1.4-5.2% (§5.2). In App. E.5 the judge accepts an answer of 45 + 8 = 53 minutes when the ground truth is 63 minutes.
- SWE-Lancer: tests sit in a password-protected ZIP whose directory can be listed and whose files can be overwritten without the password; replacing tests with `assert 1 == 1` gives a 100% resolve rate (§5.2, App. E.4).
- KernelBench: the fuzzer varies tensor values but not shapes or memory layouts; o3-mini wrote extra tests for 3 sampled kernels per level-1 task, the tests were manually verified, and kernel correctness was found overestimated by 31% (§5.2, App. E.6).
- OSWorld: 13 of 46 Chrome-section problems are broken by website changes to layouts, URLs, and selectors, underestimating UI-TAR (as printed) by 28% in absolute terms (§5.2).
- SWE-bench: citing UTBoost [87], agents pass without resolving the issue on 5.3% (Verified) and 7.7% (Lite) of tasks, changing 40.9% and 24.4% of leaderboard rankings and overestimating performance by 2.3% and 1.6% (App. E.1). §1 reports 24% of the top-50 leaderboard positions as incorrect [31, 87].
- Part scores: more than half of benchmarks have implementation flaws in task validity; 80% do not acknowledge design or implementation weaknesses; none satisfies every reporting item (§5.1).

**CVE-Bench revision (§5.3).** The time-based SQL injection check looked for a SLEEP clause in the database log, which an agent can satisfy by adding SLEEP anywhere in a query (violates O.g.1); this overestimated performance by 32.5%. Mock runs for T.9 showed agents always passing the outbound-server attack because the server was reachable from the same Docker network; denying external requests lowered success rates by 10% (Fig. 6b).

**Confidence interval under noisy ground truth (App. F, R.9).**
μ = e + (1 − 2e)·p₀ ; σ² = μ(1 − μ) ; 95% interval μ ± 1.96·σ/√N
- e: fraction of incorrect ground-truth labels; p₀: success rate measured against the imperfect ground truth; p: true success rate, assumed normal with mean μ and standard deviation σ; N: test-set size.
- In the BIRD example, verification of 500 sampled tasks found 11.65% of ground-truth queries incorrect (results from Arcwise [7]), which is used as the plug-in e; a SQL expert scores 92.96% and a do-nothing agent 0% (App. F, R.7, R.12-R.13).
- Worked example (this card's arithmetic): e = 0.1165, p₀ = 0.50, N = 1000 gives μ = 0.1165 + 0.767 × 0.50 = 0.500, σ = 0.500, interval 0.500 ± 0.031.

## Findings relevant to generality and agentic training
- **Measurement error in capability estimates:** grading flaws move measured success by 1.4% (WebArena lower bound) up to 100% (SWE-Lancer) (§5.2). R.10 asks for confidence intervals (Fig. 4). A score difference smaller than a benchmark's known grader error therefore does not establish a capability difference (Interpretation).
- **Contamination and overfitting:** R.3 (private held-out test set) and R.4 (update tasks over time) are the checklist's controls against contamination and benchmark overfitting (Fig. 4, App. F).
- **Graders as rewards:** O.c.2 and O.i.1 name reward hacking and metric exploitation explicitly (Fig. 3, §4.2). The paper evaluates benchmarks, not training; applying the same checks to RL environment graders is this course's inference (Interpretation).
- **Scope limits:** only benchmarks used from January 2024 to March 2025 were analyzed, the checklist may not be exhaustive, and results reflect benchmark versions at the time of writing (App. A).

## Connections
- [[tau-bench]], [[tau2-bench]] — τ-bench unsolvable-task and substring issues (§5.2, App. E.2).
- [[webarena-data]], [[agentgym-rl]] — WebArena grading issues affect scores such as those reported by AgentGym-RL.
- [[swe-bench-illusion]], [[swe-rebench]], [[swe-gym]] — other SWE-bench validity and contamination work.
- [[impossiblebench]] — measures agents exploiting test cases, related to T.10 and O.d.
- [[null-model-cheating-benchmarks]] — trivial-output agents scoring on automatic benchmarks, related to R.13.
- [[holistic-agent-leaderboard]], [[benchmark-variance-quantified]] — agent-evaluation infrastructure and score variance, related to R.10.
- [[judge-llm-bias]] — LLM-judge reliability, related to O.c.1.
- [[reward-hacking-taxonomy]] — exploitation of reward functions, related to O.c.2 and O.i.1.

## Verification
- Created on 2026-09-14 from https://arxiv.org/abs/2507.02825 (arXiv v5, 2025-08-07; v1 2025-07).
- Audit claims not found in the source: "agent benchmark scores used to pick mixtures can be dominated by grading artifacts, so they need auditing before being trusted as held-out signals" (an audit interpretation; the paper does not discuss training-data mixtures). The audit's "~43 items" matches the count of items listed in Figures 2-4, but the paper does not state a total.
- Internal inconsistencies in v5: §2 cites 5.2% of SWE-bench-Verified tasks passing without correct patches, App. E.1 cites 5.3%; App. B reports 25 agentic benchmarks, §3 and Table 3 list 17.
- Not reported by the source: per-benchmark numeric ABC scores in text form (Figure 5 is a bar chart), inter-annotator agreement for the assessment, how many CVE-Bench tasks each fix affected.
