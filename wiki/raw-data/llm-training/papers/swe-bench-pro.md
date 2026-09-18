<!-- scope: SWE-Bench Pro (arXiv:2509.16941, Sep 2025; v2 Nov 2025) — 1,865 human-augmented multi-file issue-resolution tasks from 41 repositories in public / commercial / held-out subsets, with SWE-Agent results, an augmentation ablation, and an LLM-judged failure taxonomy
     deps: [[swe-bench-illusion]]
     see-also: [[swe-rebench]], [[swe-smith]], [[terminal-bench-2]], [[agentic-benchmark-checklist]], [[data-contamination-survey]]
-->

# SWE-Bench Pro: Can AI Agents Solve Long-Horizon Software Engineering Tasks?
- **Core Insight:** Under one SWE-Agent scaffold with a 50-turn limit, the best model resolves 43.6% of the 731 public SWE-Bench Pro tasks (Claude Sonnet 4.5) and 17.8% of the 276 commercial tasks from private startup repositories (Claude Opus 4.1), so the same generation of models scores lower on code that is not publicly available (§5, Tables 1–2).
- **Guideline:** When unit tests are used as the pass/fail signal for feature-addition tasks, give the agent the required behavior and the interface names that the tests expect, because removing these fields lowered GPT-5 (high) from 25.9% to 8.40% and Claude Opus 4.1 from 22.7% to 8.20%, which the authors attribute to verifier false negatives (§6.2, Table 3).
- **Authors:** Xiang Deng, Jeff Da (co-first), Edwin Pan, Yannis Yiming He, Charles Ide, Kanak Garg, et al. (22 authors, Scale AI)
- **Year:** 2025 (arXiv v1 2025-09; v2 2025-11-14; no venue listed)
- **URL:** https://arxiv.org/abs/2509.16941 (data: https://huggingface.co/datasets/ScaleAI/SWE-bench_Pro; code: https://github.com/scaleapi/SWE-bench_Pro-os)
- **Source type:** paper
- **Relevant topics:** agentic coding evaluation, benchmark contamination, held-out evaluation sets, verifier false negatives, long-horizon agents, failure-mode analysis

## Abstract
SWE-Bench Pro follows SWE-Bench but targets enterprise-level problems. It contains 1,865 problems from 41 actively maintained repositories (business applications, B2B services, developer tools). A public set uses 11 repositories, a held-out set uses 12, and a commercial set uses 18 proprietary repositories obtained through agreements with early-stage startups. Held-out and commercial problems are not public, but commercial-set results are published. Tasks may take a professional engineer hours to days and often need multi-file patches. All tasks are human-verified and augmented with context so that they are resolvable. Under a unified scaffold, the tested models stay below 45% Pass@1. The authors cluster failure modes from agent trajectories and present the benchmark as contamination-resistant.

## Key Contributions
- Contamination measures: public and held-out repositories use strong copyleft licenses (GPL), and the commercial set uses purchased private startup codebases (§1, §3.1).
- Difficulty and diversity constraints: 1–10-line edits are excluded; each repository contributes 50–100 instances with a cap of 100 (§1).
- A three-part task description (problem statement, requirements, interface) written by humans and grounded in the tests (§3.2, §4.2, App. B).
- Environment and test verification: expert-built Dockerfiles, repeated gold-test runs to remove flaky tests, human review of every fail2pass test (§4.3).
- Results on public and commercial sets, an augmentation ablation, and a GPT-5-judged failure taxonomy (§5–6).

## Key Figures/Tables to Study
- Table 1 (public, N=731) and Table 2 (commercial, N=276); Table 5 (public, 50 turns and $2 cap); Table 3 (augmentation ablation); Table 4 (failure modes); Fig. 3 (resolve rate by language, repository, files changed); App. B (example task); App. C.1 (judge prompt).

## Technical Details
**Composition (§1, §3.3)**
- Public: 731 instances from 11 copyleft repositories, released on Hugging Face. Commercial: 276 problems from 18 startup repositories, private, results published. Held-out: 858 problems from 12 separate repositories, private, "to test for overfitting in the future" (§3.3).
- Reference solutions average 107.4 lines across 4.1 files; every problem changes at least 10 lines; more than 100 tasks change more than 100 lines (§1).
- For comparison, the authors state that 161 of 500 SWE-Bench Verified problems need only 1–2-line changes (§1).
- Languages: Python, JavaScript, TypeScript, Go; Java, C++, and Rust are underrepresented (§3.2, §7.1).

**Construction (§4)**
1. Sourcing: consecutive commit pairs; the test patch is the diff of test files and the gold patch is the remaining diff (§4.1).
2. Task description: a problem statement limited to what the original sources contain, plus requirements (expected behavior, no implementation) and an optional interface (class and function names, signatures, file paths) (§4.2, App. B.2–B.3).
3. Environments: professional engineers write Dockerfiles; gold tests are run several times and problems with non-passing or flaky tests are dropped (§4.3).
4. Tests: fail2pass tests verify the fix and pass2pass tests check existing behavior; human reviewers drop tests that are irrelevant or too broad, and drop the problem if all are (§3.2, §4.3).

**Evaluation settings (§5)**
- Scaffold: SWE-Agent with its default prompt for all models. Agentless was tried and scored low because of difficulty with multi-file editing.
- Model versions as of 2025-09-18; open-weight models served with vLLM on one node of 8 H100 GPUs; tool calls for open-weight models by syntax parsing; maximum 50 turns; Claude Opus 4.1 without extended thinking.
- All three task-description fields are given to the agent, so the setting measures implementation, not resolution of ambiguity. Metric: Pass@1 reported as resolve rate; the number of runs per model is not stated.

**Results**
- Public, Table 1: Claude Sonnet 4.5 43.6, Claude Sonnet 4 42.7, GPT-5 (high) 41.8, Claude Haiku 4.5 39.5, Kimi K2 Instruct 27.7, GPT-OSS 120B 16.2 (%).
- Commercial, Table 2: Claude Opus 4.1 17.8, GPT-5 (high) 15.7, GPT-5 (medium) 14.9, Gemini 2.5 Pro Preview 10.1, Claude Sonnet 4 9.1, GPT-4o 3.6 (%).
- Public with 50 turns and a $2 cost cap, Table 5 (used for all §6 analyses): GPT-5 (high) 25.9, GPT-5 (medium) 23.3, Claude Opus 4.1 22.7, Claude Sonnet 4 17.6, GPT-OSS 20B 16.2, Gemini 2.5 Pro Preview 13.5, SWE-Smith-32B 6.8, GPT-4o 4.9, Qwen-3 32B 3.4 (%). The conclusion's "23%" for Opus 4.1 and GPT-5 refers to this capped setting (§8).
- By task property (§6.1, Fig. 3): Go and Python have higher resolve rates than JavaScript and TypeScript; some repositories stay below 10% for all models while others exceed 50% for some models; Claude Opus 4.1 and GPT-5 stay above 10% on 10+-file problems while open-source models approach zero. The 10+-file and 500+-LOC bins hold about 20–30 examples each (Fig. 3 caption).
- Failure modes (§6.3, Table 4): GPT-5 judges the last 20 turns of each unresolved trajectory (20 turns matched human validation better than 10 or 40). Examples from Table 4: Claude Opus 4.1 submitted 74.2% of failed runs, and 50.3% of those were wrong solutions; GPT-5 (high) did not submit in 72.8%, with 96.4% of those tool-use failures; Claude Sonnet 4 did not submit in 55.9%, with 57.4% of those long-context failures; Qwen 3 32B had 48.7% syntax errors among submitted and 78.8% tool-use among not-submitted runs.

## Findings relevant to generality and agentic training
- Measurement design (Result, single study): per-repository caps are chosen so that no model gains from being strong on one repository (§3.1); the held-out set is kept private for later overfitting checks, and v2 reports no held-out results (§3.3, §5).
- Public versus private code (Result): the best commercial-set score is 17.8% versus 43.6% on the public set, but the model lists differ between Tables 1 and 2, so no model has a paired public/commercial number in the same setting (§5).
- Verifier false negatives (Result): unit tests accept a narrow set of interfaces, so an agent's valid solution can fail; adding requirements and interface raises resolve rates about threefold for the two models tested (Table 3). This matters for any use of these tasks as RL rewards; the paper cites Agent-RLVR as training on SWE-style instances but does no training itself (§2.1).
- Agentic failure types (Interpretation, authors): stronger models fail mostly on solution correctness, weaker open models on syntax, formatting, and tool use (Table 4 caption).
- A SWE-bench-style fine-tuned model, SWE-Smith-32B, scores 6.8% in the capped setting (Table 5); the paper gives no training-data analysis for it.

## Connections
- [[swe-bench-illusion]] — memorization evidence on SWE-Bench Verified; SWE-Bench Pro does not cite it, but both argue against relying on Verified alone.
- [[swe-rebench]] — contamination control by issue date instead of licensing and private code.
- [[swe-smith]] — source of the SWE-Smith-32B model evaluated in Table 5.
- [[terminal-bench-2]] — agent benchmark with public tasks and a canary string, no private split.
- [[agentic-benchmark-checklist]], [[holistic-agent-leaderboard]] — rigor criteria and reporting for agent benchmarks.
- [[data-contamination-survey]], [[livecodebench]] — contamination mechanisms and date-filtered evaluation.
- [[r2e-gym]], [[swe-gym]], [[deepswe]], [[together-coderforge-agent-trajectories]] — SWE training data and agents that report SWE-Bench Verified.

## Verification
- Created on 2026-09-14 from https://arxiv.org/abs/2509.16941 (arXiv v2, 2025-11-14; version history v1 2025-09-21, v2 2025-11-14 read on the abs page).
- Audit claims not found in the source: "responds directly to SWE-bench Verified memorization (SWE-Bench Illusion)" (the paper cites general contamination work, not arXiv:2506.12286); "the held-out and commercial splits are the SWE analogue of held-out environments for agent-generality claims" (course interpretation; the paper describes the held-out set only as a future overfitting check).
- Internal inconsistency in v2: the §6.3 prose percentages (for example Opus 4.1 wrong solutions 35.9%, Sonnet 4 context overflow 35.6%, Qwen 3 32B tool errors 42.0%) do not match Table 4, which splits failures into submitted and not-submitted; this card uses Table 4.
- Not reported by the source: held-out-set results, number of runs or confidence intervals, per-language task counts, commercial-set results for Claude Sonnet 4.5 or Haiku 4.5.
