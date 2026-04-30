---
chapter: ch-18
course: llm-training
phase: read
excerpt_of: "OpenMathInstruct-2: Accelerating AI for Math with Massive Open-Source Instruction Data (Toshniwal, Du, Moshkov, Kisacanin, Ayrapetyan, Gitman, NVIDIA, 2024)"
source_url: https://arxiv.org/abs/2410.01560
created_at: "2026-04-23"
---

# Excerpt: OpenMathInstruct-2 — scaling stage 1 with a cheap modality-specific stage 4

**Authors:** Shubham Toshniwal, Wei Du, Ivan Moshkov, Branislav Kisacanin, Alexan Ayrapetyan, Igor Gitman (NVIDIA)
**Year:** 2024
**arXiv ID:** 2410.01560
**Raw-data source:** [[raw-data/openmathinstruct-2]]

---

## Why this paper is the stage-1-heavy flagship

APIGen shows what stage 4 looks like when it is expensive and the whole point. OpenMathInstruct-2 shows what stage 4 looks like when it is *cheap and off-the-shelf* — SymPy symbolic equivalence — and what that buys you at stage 1. The raw-data file's core insight:

> "Switching the math-synthesis teacher from Mixtral to Llama-3.1-405B-Instruct and redoing the pipeline at ~8x scale (14M examples) closes most of the open-vs-closed gap on MATH and exposes clean data-scaling laws for synthetic CoT."

The enabling condition is that math has a *free* verifier: you can check whether a reasoning trace's final answer is symbolically equivalent to the gold answer without an LLM in the loop. Freed from the stage-4 cost, the paper can pour budget into stage 1 — K=32 solutions per problem at frontier teacher scale.

---

## Mapping OMI-2 to the six-stage loop

- **(1) Generate.** Two-level. At the *problem* level: 15K seed problems (MATH + GSM8K train) are augmented into ~600K via paraphrase + novel-question generation. At the *solution* level: "Llama-3.1-405B-Instruct samples K ≈ 32 CoT solutions per augmented problem, temperature 1.0, top-p 0.95."
- **(2) Filter.** "Extract final boxed answer via regex." No boxed answer -> reject.
- **(3) Dedup.** "Near-duplicate suppression within a problem's accepted solutions." Cross-problem overlap is allowed by design — diversity is sought at the problem level, not the solution level.
- **(4) Verify.** "Gold-answer match using SymPy canonicalization (for MATH) or numeric-string match (GSM8K)." Cheap, off-the-shelf, modality-specific.
- **(5) Select.** Keep all verified solutions; no per-solution difficulty weighting. Selection happens upstream at stage 1 via question augmentation.
- **(6) Mix.** Single-stage SFT on Llama-3.1-{1.5B, 8B, 70B} bases. No real-data mixing.

Notice: stage 4 is one line of SymPy. Stage 1 is ~650K H100-hours. The budget allocation is the inverse of APIGen's.

---

## The two-level stage-1 idea

The raw-data file describes the augmentation:

> "Seed input: MATH train (7.5K) + GSM8K train (7.5K), **augmented** into ~600K problems via two operations:
> 1. **Question paraphrasing** (teacher rewrites each problem).
> 2. **Novel question generation** — teacher generates new problems in the style of the seed, grounded on topic/difficulty tags."

Notice: 15K seed problems -> 600K augmented problems is a 40x expansion *before* any solutions are sampled. Then each of the 600K augmented problems gets K=32 solutions, yielding ~19.2M candidates of which 14M survive verification.

This two-level structure is unique among the four flagship ch-18 pipelines. Self-Instruct and APIGen only scale at the (query, response) level; Nemotron scales at the task-family level but does not explicitly multiply problems from seeds. OMI-2 shows that stage 1 itself can be hierarchical — scale the problems first, then scale the solutions per problem — and this is the right decomposition for reasoning-trace modalities where a single "problem" admits many valid solutions.

---

## The teacher-strength ablation — the paper's most cited result

From the raw-data file:

> "Ablation: **stronger teacher > more data from weaker teacher** — Llama-3.1-405B at 1M samples beats Mixtral at 10M."

Restated: 10x less data from a 405B teacher beats 10x more data from a Mixtral (8x7B) teacher. This is the stage-1 quality-vs-quantity axis made quantitative. The ch-18 lesson is not just "buy a bigger teacher" — it is that **stage 1 quality compounds through stages 2–6 in a way stage-1 quantity does not**.

Notice: this is also a cost-reallocation argument. If the 10x-Mixtral run costs the same compute as the 1x-405B run, the paper is saying "route compute into teacher quality, not solution count." Put this next to the ch-18 cost profile: stage 1 is where the money goes, but *per-sample* expenditure on a stronger teacher beats per-sample spread across a weaker teacher.

---

## Stage 4 is cheap, but not perfect

From the raw-data file:

> "Correctness verifier: SymPy-based symbolic equivalence (MATH) + exact numeric match (GSM8K).
> Error mode: false-positive rate (right answer, wrong reasoning) measured at ~7% on a human-audited sample."

SymPy will accept a trace that has the right final answer but broken intermediate reasoning. 7% of accepted solutions are in this "right answer, wrong reasoning" category. The paper is forthright about this as the stage-4 blind spot specific to gold-answer-only verification.

The fix, which OMI-2 does not implement but later papers ([[rstar-math]], [[math-shepherd]], [[prm800k]]) do, is a *step-level* verifier — a process reward model that checks each reasoning step. That is a more expensive stage 4 and produces smaller datasets. OMI-2's bet is that 7% false-positive is tolerable when the dataset is 14M samples; whether that bet holds at the performance frontier is an open question that connects to ch-28 (iteration) and to the RL track (RLVR).

---

## Scaling-law result: stage-1 saturation

The raw-data file records a clean result:

> "Scaling laws: downstream MATH accuracy ≈ linear in log(dataset size) up to ~5M; flat beyond."

14M accepted solutions; accuracy saturates around 5M. The last 9M of solutions add little. This is not a pipeline failure — it is a teacher-ceiling. Once the 405B teacher's distribution of "solutions this 405B can produce correctly for MATH-style problems" is saturated, more samples don't help.

Notice: this is a **stage-1 ceiling**, not a stage-4 ceiling. The verifier is still correctly accepting / rejecting; there is just no new information in additional solutions. The ch-18 corollary: **stage 1 has a saturation point determined by the teacher**, and past that point you must either (a) switch to a stronger teacher (expensive), (b) move to a step-level verifier that extracts more information per solution, or (c) pivot to a different stage of the loop for gains.

---

## What OMI-2 deliberately does *not* do

From the raw-data file's "Risks + gotchas":

> "- **Short-CoT ceiling:** the dataset is non-reflective CoT; students trained on it do not acquire backtracking. Separate long-CoT sources (s1, LIMO, R1 distills) needed for o1-style behavior."

OMI-2's trace style is short-CoT, not o1-style long-CoT with reflection and backtracking. Students trained on OMI-2 inherit the trace style. This is a stage-1 choice (prompt the teacher for standard CoT) that permeates the whole pipeline. Long-CoT synthesis is a different stage-1 configuration, treated in ch-21 alongside [[s1]] and [[limo]].

Notice: the trace-style choice is made at stage 1, amplified through stage 4 (gold-answer match doesn't care about style), preserved through stage 5 (no style selection), and baked into the student at stage 6. One stage-1 decision propagates through the entire loop.

---

## The pipeline outcome

From the raw-data file:

> "OpenMath2-Llama3.1-8B: **91.7 GSM8K, 67.8 MATH** — SOTA among open 8B math models at release."

An 8B open model, within ~1 point of closed-teacher 8B distillations at the time. This is the ch-18 headline Nemotron offers for alignment: synthetic-only data, cheap verifier, frontier-adjacent results. The enabling condition is the free stage-4 verifier — a reminder that modality choice (having a symbolic verifier) is itself a design decision for any synthetic-data project.

---

## Contamination: the hidden stage-2 failure mode

The raw-data file flags:

> "**Question-aug contamination:** novel questions generated by the teacher may overlap with MATH/GSM8K test sets — authors run decontamination against test sets but 405B teacher may still leak."

This is a stage-2 (filter) concern that most stage-1-heavy pipelines underappreciate. When you ask a frontier teacher to "generate a new MATH-style problem in the style of this topic," there is a non-trivial chance the teacher reproduces (or near-reproduces) a problem from the held-out test set it was trained on. The OMI-2 team runs n-gram decontamination, but the raw-data file is frank: "405B teacher may still leak."

The ch-18 lesson: **scaling stage 1 scales stage-2 contamination risk roughly in proportion**. A stage-2 filter that was adequate at 15K problems may admit contamination at 600K problems generated in the source corpus's style. Decontamination is not a one-time setup; it is a stage-2 invariant that must be re-verified as stage 1 scales.

This generalises. APIGen's execution verifier is indirectly protected from contamination because the reference-API implementations are Salesforce-controlled — the teacher cannot leak exam answers because there is no exam in its training data for Salesforce's internal APIs. OMI-2's teacher, by contrast, has seen MATH publicly. Different modalities, different contamination pressures, same stage-2 concern.

## K = 32 per problem, and why that is the right number

OMI-2's per-problem sampling count is K ≈ 32. Not 1, not 100. Why 32?

This is a stage-1 hyperparameter that interacts directly with stage 4's yield rate. For a 405B teacher sampling math-CoT solutions, the solved-rate per sample is roughly 60–80% on augmented MATH-style problems (interpolating from the paper's acceptance numbers). Pushing K higher gives diminishing returns — you are asking the teacher to generate increasingly repetitive or degenerate variations on the same solution trace. Pushing K lower under-samples problems near the teacher's solved/unsolved boundary, where one or two attempts may fail but a handful will succeed.

32 is the bet that 1–2 correct solutions will reliably surface per problem across the 600K augmented pool, *without* spending sample budget on problems the teacher has already solved three different ways. It is a stage-1 tuning parameter that is, in effect, chosen by the expected stage-4 yield rate.

For ch-18: K is the right kind of hyperparameter to expose explicitly. Too-low K is under-sampling; too-high K is wasted compute. A well-designed stage-1 names its K and justifies it against the modality's stage-4 yield curve. OMI-2 is the clearest public instance of this kind of reasoning.

## Connections

- [[excerpts/self-instruct]] — same loop shape, but OMI-2's stage 4 is the thing Self-Instruct was missing.
- [[excerpts/apigen]] — the contrast: expensive stage 4 (three layers) vs OMI-2's cheap stage 4 (one SymPy check). Each is appropriate to its modality's verifier availability.
- [[excerpts/synthetic-scaling-laws]] — OMI-2's ~5M solution saturation is a modality-specific instance of the broader "teacher defines the ceiling" result.
- [[excerpts/nemotron-4]] — contrast at stage 4: Nemotron's 340B reward model (general-purpose, expensive) vs OMI-2's SymPy check (modality-specific, essentially free); also contrast at stage 1 scale allocation.
- [[excerpts/nathan-lambert-synth]] — OMI-2 is the canonical "hard-verifiable" case in Lambert's easy-vs-hard split.
- [[ch-18]] — parent. OMI-2 is the ch-18 reasoning-trace flagship and the cheap-stage-4 counterexample to APIGen's expensive stage 4.
- Forward references: the step-level verifier pipelines ([[rstar-math]], [[math-shepherd]], [[prm800k]]) live in ch-21; they upgrade OMI-2's gold-answer-only stage 4 into process-level verification at higher cost and smaller dataset scale.
- Forward reference: the long-CoT / reflective synthesis lineage ([[s1]], [[limo]], [[bespoke-stratos]]) deliberately chooses a different stage-1 style than OMI-2's short-CoT; covered in ch-21.
- [[excerpts/apigen]] — opposite budget allocation: APIGen spends at stage 4, OMI-2 spends at stage 1. Both work because their modalities admit that tradeoff.
- [[excerpts/nemotron-4]] — stage 4 as one RM; OMI-2's stage 4 as SymPy; two valid endpoints of the verifier-cost spectrum.
- [[excerpts/synthetic-scaling-laws]] — OMI-2's ~5M saturation is a stage-1 scaling-law datapoint aligned with the broader 2025 literature.
- [[ch-18]] — parent. OMI-2 is the reasoning-trace flagship and the cheap-stage-4 counterexample to APIGen's expensive stage 4.
