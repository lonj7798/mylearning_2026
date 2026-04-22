<!-- scope: reasoning-trace synthesis — 17K R1 distill curation for math/code/science
     deps: [[deepseek-r1]]
     see-also: [[openr1]], [[sky-t1]], [[s1]]
-->

# Bespoke-Stratos: A small, open replication of R1 distillation
- **Core Insight:** 17K curated (prompt, R1-trace) pairs are enough to distill o1-class reasoning into Qwen-32B-Instruct and 7B bases; the magic is in R1's traces, not in massive volume — but careful filtering (rejection sampling for correctness + domain balancing) is essential.
- **Guideline:** For R1 distillation on a budget, sample R1 traces at temperature 0.6, filter by gold-answer match (math) / unit-test pass (code) / LLM-judge (science), and keep ~10–20K of the best traces — you get 90%+ of the Stratos-Distill gain for 1/50 the data cost.
- **Authors:** Bespoke Labs team (Mahesh Sathiamoorthy, et al.)
- **Year:** 2025
- **URL:** https://www.bespokelabs.ai/blog/bespoke-stratos-the-unreasonable-effectiveness-of-reasoning-distillation
- **Relevant topics:** R1 distillation, reasoning-trace curation, rejection-sampling, open replication

## Abstract
Bespoke-Stratos is an open curation of 17,000 reasoning traces distilled from DeepSeek-R1 covering math (MATH, NuminaMath, AIME), code (APPS, CodeContests, TACO, LeetCode), and science (GPQA-style). The dataset was released in January 2025 as part of Bespoke Labs' Curator toolkit and was used to train Bespoke-Stratos-32B (SFT-only, Qwen2.5-32B-Instruct base) to within a few points of DeepSeek-R1-Distill-Qwen-32B (which used 800K traces). Companion 7B model also released.

## Key Contributions
- **Bespoke-Stratos-17k** dataset — public, Apache-2.0.
- Demonstration that R1 distillation is ~50× more data-efficient than the official R1-Distill recipe when filtering is careful.
- Release of **Curator**, a data-generation toolkit with built-in rejection-sampling / LLM-judge primitives.
- Matched Sky-T1 recipe at 1/5 the compute budget.

## Synthesis pipeline (REQUIRED — concrete, modality-specific)
- **Seed input:** problem pool stratified across domains:
  - **Math:** NuminaMath-CoT, MATH, AIME/AMC archive (~7K problems).
  - **Code:** APPS, CodeContests, TACO, LeetCode (~5K problems).
  - **Science/reasoning:** STILL-2 curated prompts, CoTLogic (~5K).
- **Trace generation:** query **DeepSeek-R1** (official API) for each seed problem at temperature 0.6, requesting the `<think>…</think><answer>…</answer>` format; retry up to 3× for failures.
- **Filtering (rejection sampling by verifier):**
  - **Math:** extract boxed answer, compare to gold via SymPy canonicalization; reject on mismatch.
  - **Code:** extract the candidate solution, run against public unit tests; reject if any test fails.
  - **Science:** use GPT-4o as LLM-judge comparing R1's final answer to the reference; require "correct" verdict.
- **Dedup + balance:** MinHash dedup across prompts; cap per-source count to enforce roughly balanced domain mix.
- **Output shape:** 17,000 (prompt, long-CoT trace) pairs. Average trace length ~3K tokens; tail to 10K+. Format preserves R1's `<think>` / `<answer>` wrappers.
- **Teacher model:** DeepSeek-R1 (official API, ~671B params MoE).
- **Cost / compute:** ~$800 in DeepSeek-R1 API credits (disclosed in blog); training cost for Stratos-32B is ~$4,000 on 8×H100 for a few hours.

## Modality-specific technical details (REQUIRED — reasoning-trace)
- **Reasoning length distribution:** 1K–10K tokens/trace; median ~3K. Long-CoT with heavy reflection ("Wait", "Let me recheck").
- **Trace style:** R1's native long-CoT with `<think>` / `</think>` / `<answer>` segmentation preserved verbatim.
- **Correctness verifier:** multi-modal — SymPy for math, test-execution for code, LLM judge for science.
- **Error-mode filter:** rejection rate ~30–50% of raw R1 outputs; majority of rejections are code failures and math extraction errors.
- **Why 17K suffices:** the authors argue (consistent with [[s1]] / [[limo]]) that base-model latent capability + R1 formatting template is what transfers — not the breadth of training prompts.

## Quality / diversity evaluation
- **Bespoke-Stratos-32B:** AIME24 ~63%, MATH500 ~93%, LiveCodeBench ~57% — within 2–3 points of R1-Distill-Qwen-32B despite using 1/47 of its data.
- **Bespoke-Stratos-7B:** AIME24 ~20%, MATH500 ~82%, LiveCodeBench ~37%.
- Ablations: removing code verification halves LiveCodeBench gain; removing math symbolic equivalence halves MATH gain.

## Risks + gotchas
- **R1 dependency:** entirely bottlenecked on R1 API availability and licensing; the dataset inherits R1's quirks and biases.
- **No step verification:** traces are accepted on final-answer correctness only — intermediate reflection content may contain plausible-sounding errors.
- **Narrow domain:** math+code+science; does not cover agentic or long-context.
- **Contamination risk:** AIME and MATH prompts are public; teacher may have memorized solutions.

## Connections
- Parallel 2025 R1-distill efforts: [[openr1]] (HF Open-R1), [[sky-t1]] ($450 recipe), [[qwen-qwq-traces]].
- "Less-is-more" kin: [[s1]], [[limo]].
- Foundational teacher: [[deepseek-r1]].
- Curator toolkit lineage: uses [[self-instruct]]-style seed pipelines under the hood.
