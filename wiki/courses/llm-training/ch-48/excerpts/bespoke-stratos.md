---
chapter: ch-48
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/bespoke-stratos.md
source_url: https://www.bespokelabs.ai/blog/bespoke-stratos-the-unreasonable-effectiveness-of-reasoning-distillation
created_at: "2026-04-23"
---

# Excerpt: Bespoke-Stratos — contamination check in distillation, the minimal case study

**Source library:** `wiki/raw-data/llm-training/papers/bespoke-stratos.md`
**Artifact:** Explicit contamination-risk admission; R1 trace generation pipeline; filter stages that decontamination must also cover.

---

## Why Bespoke-Stratos is ch-48's concrete downstream-contamination example

Ch-48 §5 describes the train → RM pref → eval pathway abstractly. Bespoke-Stratos is the two-paragraph version of that pathway, confined to a small, fully-documented 17K-trace dataset. It is the minimal case where every failure mode the chapter discusses is either present or explicitly acknowledged.

---

## The admission that drives ch-48's teacher-memorization section

Source §Risks + gotchas:

> **Contamination risk:** AIME and MATH prompts are public; teacher may have memorized solutions.

This single sentence encodes the downstream-contamination pathway:
1. AIME / MATH are public benchmarks.
2. DeepSeek-R1's pretraining corpus plausibly includes them (CC re-crawls of competition archives, discussion forums, solution sites).
3. R1's "reasoning traces" for those problems are mixtures of memorised solutions and genuine reasoning; there is no way to separate them from the trace.
4. Distilled traces become SFT data for Bespoke-Stratos-32B.
5. Bespoke-Stratos evaluates on AIME24 and MATH500 — the same benchmarks whose solutions may be upstream in R1.

Ch-48 §5's recommendation to decontaminate against the *trace content*, not only the *seed prompts*, comes directly from this pathway.

---

## The synthesis pipeline as a contamination-amplification stack

Source §Synthesis pipeline:

> - **Trace generation:** query DeepSeek-R1 (official API) for each seed problem at temperature 0.6 …
> - **Filtering (rejection sampling by verifier):** math via SymPy, code via unit tests, science via LLM-judge.
> - **Dedup + balance:** MinHash dedup across prompts

Two observations from a contamination lens:

1. The dedup step is **prompt-level, not trace-level**. Two prompts that differ in surface form but share the same gold answer (common in competition math — multiple AIME variants testing the same identity) survive dedup; their traces are correlated but not deduplicated. The defensive fix: hash the *answer + key solution pattern*, not only the problem statement.
2. The rejection-sampling verifier is aligned with the eval metric (SymPy answer match ≈ MATH grader). A trace that memorizes the gold answer passes the verifier by construction. The verifier is blind to the memorization it selects for.

Both observations are instances of the ch-48 §5 "downstream filter that amplifies contamination" pattern.

---

## The 30–50% rejection rate — what survives is not automatically clean

Source §Modality-specific technical details:

> **Error-mode filter:** rejection rate ~30–50% of raw R1 outputs; majority of rejections are code failures and math extraction errors.

The 50–70% of traces that pass the filter are *correct by verifier*; this does not mean they are *uncontaminated by memorization*. A memorized AIME solution passes the verifier perfectly; a de-novo reasoning chain that happens to be correct also passes. The filter provides no separation.

Ch-48's memo §7 "what the memo does NOT claim" section applied to this pipeline must include: "no audit of whether verifier-passing traces are reasoning outputs vs memorized outputs."

---

## Why 17K traces is not a small-number reprieve

Source §Abstract:

> 17,000 reasoning traces distilled from DeepSeek-R1 covering math (MATH, NuminaMath, AIME), code (APPS, CodeContests, TACO, LeetCode), and science (GPQA-style).

A common intuition: "17K is small, so contamination exposure is bounded." This is wrong. Contamination is about which *eval instances* the training data overlaps, not about total token count. A single memorized AIME problem in training is enough to inflate that problem's eval score by 100%. 17K traces targeting AIME / MATH / APPS is a *perfectly targeted* contamination surface if the teacher has memorized those benchmarks.

Ch-48 §3's per-instance overlap fraction (not per-corpus rate) is the right metric here.

---

## The matched-recipe efficiency as a red flag worth checking

Source §Quality / diversity evaluation:

> **Bespoke-Stratos-32B:** AIME24 ~63%, MATH500 ~93%, LiveCodeBench ~57% — within 2–3 points of R1-Distill-Qwen-32B despite using 1/47 of its data.

"50x more data-efficient" is a spectacular claim. It is *also* the claim you would make if the 17K traces were concentrated on benchmark-adjacent problems and the scoring was partly recall. Ch-48 §5's memo structure ("what the memo does NOT claim") is the responsible way to report numbers like this: the score is what the model achieves; the claim it generalises is separate and must be supported by a decontamination audit.

---

## What ch-48 takes from Bespoke-Stratos

| Source detail | Ch-48 adoption |
|---|---|
| Explicit "teacher may have memorized" admission | Memo template §7 teacher-memorization disclaim |
| Prompt-level MinHash dedup | Flags need for answer/trace-level hashing |
| Verifier passes memorized solutions | §5 "filter amplifies contamination" example |
| 50x data efficiency claim | Red-flag pattern requiring decontam audit |

---

## Connections

- **[[deepseek-r1]]** — the teacher model whose memorization surface drives the risk.
- **[[llama-3]]** — the full-scale pipeline where the same pathway has more stages.
- **[[anthropic-sleeper-agents-data]]** — adversarial sibling of the same conditional-memorization phenomenon.
- **[[deduplicating-training-data]]** — the MinHash primitive repurposed here.
- **[[faithful-synth-eval]]** — external-verifier framework that Bespoke's verifier instantiates.
