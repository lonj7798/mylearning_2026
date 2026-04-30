---
chapter: ch-35
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/bespoke-stratos.md
source_url: https://www.bespokelabs.ai/blog/bespoke-stratos-the-unreasonable-effectiveness-of-reasoning-distillation
created_at: "2026-04-23"
---

# Excerpt: Bespoke-Stratos — 17K curated traces + contamination discipline

**Source library:** `wiki/raw-data/llm-training/papers/bespoke-stratos.md`
**Team + date:** Bespoke Labs (Sathiamoorthy et al.), January 2025.

---

## Why this source anchors ch-35

Bespoke-Stratos is the "curate aggressively + SFT only + ship" prototype for the 2025 open-reasoning-distillation community. From the source header:

> **Core Insight:** 17K curated (prompt, R1-trace) pairs are enough to distill o1-class reasoning into Qwen-32B-Instruct and 7B bases; the magic is in R1's traces, not in massive volume — but careful filtering (rejection sampling for correctness + domain balancing) is essential.

Ch-35 §4.1 uses this as the cheap-frontier-reasoning recipe; the excerpt records the specific filter layers and the contamination-check discipline that separate this recipe from a naive rerun.

---

## The seed pool, exactly

From "Synthesis pipeline":

> **Seed input:** problem pool stratified across domains:
> - **Math:** NuminaMath-CoT, MATH, AIME/AMC archive (~7K problems).
> - **Code:** APPS, CodeContests, TACO, LeetCode (~5K problems).
> - **Science/reasoning:** STILL-2 curated prompts, CoTLogic (~5K).

Total seed pool ≈ 17K problems, not 17K traces. One trace per problem (retries up to 3× on format failure). The 17K final pairs correspond to ~17K problems that survived all verifier layers.

## Trace generation parameters

> **Trace generation:** query **DeepSeek-R1** (official API) for each seed problem at temperature 0.6, requesting the `<think>…</think><answer>…</answer>` format; retry up to 3× for failures.

Temperature 0.6 is the R1 community-default (matches what DeepSeek's own rejection-sampling stage used). The 3× retry budget on format failure is parsimonious — more retries would raise cost without improving trace quality.

## The three-layer verifier

This is the recipe's load-bearing component. From the source:

> **Filtering (rejection sampling by verifier):**
> - **Math:** extract boxed answer, compare to gold via SymPy canonicalization; reject on mismatch.
> - **Code:** extract the candidate solution, run against public unit tests; reject if any test fails.
> - **Science:** use GPT-4o as LLM-judge comparing R1's final answer to the reference; require "correct" verdict.

Three layers, three modalities. The ablation attested in "Quality / diversity evaluation":

> Ablations: removing code verification halves LiveCodeBench gain; removing math symbolic equivalence halves MATH gain.

Both math and code verifiers are load-bearing. Removing either halves the benchmark gain. No single verifier dominates; the three-layer stack is the minimum viable filter for a mixed-modality reasoning corpus.

## Dedup + domain balance

> MinHash dedup across prompts; cap per-source count to enforce roughly balanced domain mix.

MinHash runs cross-prompt (not within a single prompt's retries) to catch near-duplicate *problems* — a common pathology when seed sources overlap (e.g., NuminaMath contains problems that also appear in AIME archives). The per-source cap is a domain-balance mechanism, not a quality filter — it prevents the pool from collapsing onto whichever source has the most problems.

## Contamination checks

Implicit in the "Risks + gotchas" section:

> **Contamination risk:** AIME and MATH prompts are public; teacher may have memorized solutions.

The team's discipline is to report both in-distribution (AIME24, MATH500) and out-of-distribution (AIME25 — post-R1-training-cutoff) numbers. The gap between them is the chapter's proxy for contamination-level inflation. Ch-35 §4.1 flags AIME25 as the clean-eval target when the teacher's cutoff is 2024.

## Output shape + costs

> **Output shape:** 17,000 (prompt, long-CoT trace) pairs. Average trace length ~3K tokens; tail to 10K+. Format preserves R1's `<think>` / `<answer>` wrappers.
> **Teacher model:** DeepSeek-R1 (official API, ~671B params MoE).
> **Cost / compute:** ~$800 in DeepSeek-R1 API credits (disclosed in blog); training cost for Stratos-32B is ~$4,000 on 8×H100 for a few hours.

Total ≈ $4.8K. This is the reference cost point for "cheap frontier reasoning" in 2025. Sky-T1's $450 is cheaper but uses a weaker teacher (QwQ); the Stratos-Sky comparison quantifies the teacher-cost tradeoff in §4-5 of ch-35.

## Student results

> - **Bespoke-Stratos-32B:** AIME24 ~63%, MATH500 ~93%, LiveCodeBench ~57% — within 2–3 points of R1-Distill-Qwen-32B despite using 1/47 of its data.
> - **Bespoke-Stratos-7B:** AIME24 ~20%, MATH500 ~82%, LiveCodeBench ~37%.

The 2–3 points gap vs R1-Distill's 800K corpus is the chapter's headline evidence that 17K curated beats 800K rejection-sampled. Note the 7B numbers — the "small base" regime — sit much lower, which is consistent with LIMO's "Less-Is-More" claim that the base model must already have the latent capacity.

---

## Why ch-35 cites this as a reference recipe

The Bespoke-Stratos recipe is the 2025 baseline for "I have a weekend, a $5K budget, and I want a credible reasoning model." It is also the cleanest public case of three-layer verifier discipline: SymPy for math, unit-test exec for code, LLM-judge for open-ended. Every 2025 reasoning-distill recipe that does not use all three layers is implicitly betting that its chosen layer covers enough of the domain — and the source ablation says that bet is usually wrong.
