---
chapter: ch-20
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/bespoke-stratos.md
source_url: https://www.bespokelabs.ai/blog/bespoke-stratos-the-unreasonable-effectiveness-of-reasoning-distillation
created_at: "2026-04-23"
---

# Excerpt: Bespoke-Stratos — 17K curated R1 traces at $800 API cost

**Source library:** `wiki/raw-data/llm-training/papers/bespoke-stratos.md`
**Blog/release:** Bespoke Labs (Mahesh Sathiamoorthy et al.), January 2025.

---

## Why this source anchors ch-20

Stratos is the "proof of concept" that **R1-distill is 50× more data-efficient than the official R1-Distill recipe when the filter is careful**. The R1 report released 800K traces → 6 students. Stratos released 17K traces → Stratos-32B, within 2–3 points of R1-Distill-Qwen-32B on every standard reasoning eval. This is the data point that anchors ch-20's claim that *curation beats scale, up to the teacher's ceiling*.

From source line 7:

> **Core Insight:** 17K curated (prompt, R1-trace) pairs are enough to distill o1-class reasoning into Qwen-32B-Instruct and 7B bases; the magic is in R1's traces, not in massive volume — but careful filtering (rejection sampling for correctness + domain balancing) is essential.

---

## The pipeline — verbatim (source lines 23–36)

```
Seed input (stratified across domains):
  Math     : NuminaMath-CoT, MATH, AIME/AMC archive  (~7K problems)
  Code     : APPS, CodeContests, TACO, LeetCode      (~5K)
  Science  : STILL-2 curated prompts, CoTLogic        (~5K)

Trace generation:
  Teacher  : DeepSeek-R1 (official API, 671B MoE)
  Sampling : T = 0.6; 1 trace per problem; up to 3 retries on format failure
  Format   : <think>…</think><answer>…</answer> preserved verbatim

Filtering (rejection sampling by verifier):
  Math     : extract \boxed{}; compare to gold via SymPy canonicalization
  Code     : extract candidate; run against public unit tests
  Science  : GPT-4o LLM-judge compares R1 answer to reference

Dedup + balance:
  MinHash dedup across prompts
  Per-source cap to enforce balanced mix

Output: 17,000 pairs
  Average trace length ~3K tokens; tail 10K+

Cost:
  Teacher inference : ~$800 DeepSeek-R1 API credits
  Student training  : ~$4,000 on 8×H100 for a few hours (32B model)
```

Three design choices are worth pulling out because they are the operational lessons ch-20 keeps citing.

---

## 1. Multi-modal verifier stack

Stratos's filter is not one rule — it's three rules composed, each appropriate to its domain. Source line 41:

> **Correctness verifier:** multi-modal — SymPy for math, test-execution for code, LLM judge for science.

Why this matters: a single verifier (say, just the LLM judge) catches ~60% of true correctness; each added verifier catches an incremental slice the others miss. SymPy catches algebraic equivalence the LLM would miss (e.g., `\frac{1}{2}` vs `0.5`); unit tests catch code-semantics errors the LLM would rubber-stamp; the LLM catches open-ended science answers neither of the first two can evaluate.

**Ablation from the blog (source line 48):**
- Remove code verification → LiveCodeBench gain is **halved**.
- Remove math symbolic equivalence → MATH gain is **halved**.

The filter is not an afterthought; it is as load-bearing as the teacher. Ch-20 §4.2 states this directly.

---

## 2. Rejection rate 30–50%

Source line 42:

> **Error-mode filter:** rejection rate ~30–50% of raw R1 outputs; majority of rejections are code failures and math extraction errors.

A third to half of R1's outputs fail Stratos's filter — and this is on problems the R1 team already trained *their* filter to pass. The lesson: R1 is strong on average but not uniformly correct; the distill corpus quality is set by the rejection threshold, not by teacher quality alone. Reproduction efforts that skip rejection sampling (an early-2025 failure mode in indie attempts) produce students that inherit 30–50% bad reasoning traces verbatim.

---

## 3. Results at 1/47 the data volume

Source line 47:

> **Bespoke-Stratos-32B:** AIME24 ~63%, MATH500 ~93%, LiveCodeBench ~57% — within 2–3 points of R1-Distill-Qwen-32B despite using 1/47 of its data.

R1-Distill-Qwen-32B used 800K traces. Stratos-32B used 17K. The ratio is 47. The evaluation delta is 2–3 points on AIME. Interpretation: **the last 783K traces buy 2–3 AIME points, or ~0.003 points per 1000 traces**. The first 17K buy the ~63% baseline.

This is the same "less is more" lesson in [[s1]] (1K training pairs, competitive reasoner) and [[limo]] (817 pairs). The mechanism, per the Stratos blog (source line 44): *the base model's latent capability + R1's formatting template is what transfers — not the breadth of prompts.* The student doesn't need to see 800K examples of the long-CoT pattern to internalize it; it needs to see a few thousand correct ones.

---

## Risks the paper flags (source lines 50–54)

1. **R1 dependency.** Stratos is bottlenecked on R1 API availability and R1's MIT license. If DeepSeek changed the license, the corpus's legal status would change retroactively.

2. **No step verification.** Stratos accepts traces on final-answer correctness only. Intermediate reflection content may contain plausible-sounding errors — the "wrong-question-correctly" failure ch-20 §5.5 covers. Process verification (ch-25) is the only defense; Stratos explicitly does not attempt it.

3. **Narrow domain.** Math + code + science. Does not cover agentic reasoning, long-context, multi-turn dialog. The 17K number is conditional on this scope.

4. **Contamination risk.** AIME, MATH prompts are public; R1 may have memorized. Evaluation on contaminated benchmarks overstates the student's true reasoning gain.

---

## How ch-20 cites this

Ch-20 §4's comparison table uses Stratos as the **low-data reference point** for the open-reproduction triangle (Stratos / Open-R1 / Sky-T1). The ablations (remove math verifier, remove code verifier) are the empirical grounding for the "verifier is as important as teacher" claim in §4.2. The $800 API cost disclosure is the anchor for §7's licensing discussion: Stratos is the cheapest proof that *if teacher outputs are redistributable, a small lab can produce a frontier reasoning dataset for the price of a GPU day*.
