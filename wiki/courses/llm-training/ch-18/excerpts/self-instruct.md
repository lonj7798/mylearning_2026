---
chapter: ch-18
course: llm-training
phase: read
excerpt_of: "Self-Instruct: Aligning Language Models with Self-Generated Instructions (Wang, Kordi, Mishra, Liu, Smith, Khashabi, Hajishirzi, 2022)"
source_url: https://arxiv.org/abs/2212.10560
created_at: "2026-04-23"
---

# Excerpt: Self-Instruct — the first full loop instance, and where the loop shows its shape

**Authors:** Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A. Smith, Daniel Khashabi, Hannaneh Hajishirzi (UW / AI2 / Yejin Choi group)
**Year:** 2022
**Venue:** ACL 2023
**arXiv ID:** 2212.10560
**Raw-data source:** [[raw-data/self-instruct]]

---

## Why this paper is the origin point of the design-pattern lens

Self-Instruct is not the first synthetic-data paper, but it is the first one whose pipeline is **explicitly labelled as a pipeline** and whose authors name the stages. The four-step diagram in Figure 1 of the paper — seed tasks, instruction generation, classification/non-classification branching, instance generation, filtering — is the source-of-truth for almost every SFT data pipeline that follows it. Alpaca, Vicuna, WizardLM, Orca, Nemotron-4's Genetic Instruct, Magpie, Persona-Hub — every one of these is a modification of Self-Instruct's pipeline with one or two stages replaced.

For the design-pattern chapter (ch-18), Self-Instruct is useful because it instantiates every stage of the loop **except verify**, and the missing verify is exactly the weakness the lineage spent five years filling in.

---

## The four-step pipeline, mapped onto the six-stage loop

The raw-data file summarises the pipeline as:

> "1. **Instruction generation** — prompt the LM with 8 in-context examples (6 from seed, 2 from previously generated) and ask for a new task instruction.
> 2. **Classification-vs-non-classification branching** — ask the LM whether the instruction is a classification task; this changes the instance-generation prompt template (input-first for classification to avoid label bias, output-first otherwise).
> 3. **Instance generation** — for each accepted instruction, prompt the LM to produce an input and an output.
> 4. **Filtering**:
>    - Drop instructions with **ROUGE-L > 0.7** to any existing instruction (diversity filter).
>    - Drop instances where input == output, outputs too long/short, or the instruction contains 'image/graph/file'.
>    - Drop ill-formatted generations."

Map this onto our six stages:

- **(1) Generate** = steps 1 + 2 + 3 of the paper. The teacher (GPT-3 text-davinci-001) emits instructions first, then conditional on each instruction emits (input, output) pairs.
- **(2) Filter** = the format checks in step 4: input==output rejection, length bounds, "image/graph/file" keyword bans. These are surface-level.
- **(3) Dedup** = the ROUGE-L > 0.7 check. This is the single most-copied mechanism in the paper.
- **(4) Verify** = **empty**. There is no gold-answer check, no executor, no judge. Whatever the teacher emits that passes structural filters becomes training data.
- **(5) Select** = implicit. Everything that survives (2)+(3) is kept. No difficulty weighting, no coverage targeting.
- **(6) Mix** = single-stage SFT on vanilla GPT-3. The Self-Instruct corpus *is* the training mix.

Notice: four of the six stages are populated; stage 4 is empty and stage 5 is default-pass. That is the Self-Instruct fingerprint.

---

## Concrete numbers to remember

From the raw-data extract:

> "Final dataset: ~52K instructions x ~82K instances (after filtering from ~252K raw generations) produced using GPT-3 (text-davinci-001-era model)."

- **Yield rate:** 52K / 252K ≈ 21% of raw generations become final instructions. Most of the rejection is from the ROUGE-L dedup, not format filtering — diversity is the scarce resource.
- **Seed-to-output ratio:** 175 seed tasks fan out into 52K accepted instructions, a ~300x amplification. This is the bootstrap factor every later paper calibrates against.
- **Headline downstream result:** +33 absolute points on Super-NaturalInstructions vs vanilla GPT-3, matching InstructGPT-001 (trained with private human data).

Notice: the paper's claim to fame is a *training* result, but the engineering is entirely at stages 1–3 of our loop. All the leverage came from cheap structural/diversity filters plus a good teacher.

---

## Why the missing stage 4 matters

The raw-data file lists the observed failure modes:

> "**Failure modes observed:** hallucinated 'impossible' tasks, output-bias in classification, repetition — addressed via the diversity filter."

Two of these are stage-1 problems (template collapse, output bias); the diversity filter only addresses repetition. Hallucinated instructions — tasks the teacher can describe but cannot solve — slip through because nothing checks whether the output is *correct*, only whether it is *formatted*. This is the exact gap that APIGen (3-layer verification), Nemotron-4 (reward model), and OpenMathInstruct-2 (SymPy) close in their respective modalities.

The lesson for ch-18: **Self-Instruct is the proof that a loop with an empty verify stage can still ship a useful dataset**, but it is also the proof that the ceiling of such a dataset is bounded by teacher quality. Every headline-number-moving successor improved stage 4.

---

## The instruction-generation prompt template

The paper publishes the exact template, which became a template for a lineage:

```
Come up with a series of tasks:
Task 1: <seed 1>
Task 2: <seed 2>
...
Task 8: <seed 8>
Task 9:
```

Six slots are seed tasks; two are sampled from previously generated (and accepted) outputs. The two-slot feedback is the **iteration-within-stage-1** that gives Self-Instruct its bootstrap character — once the pool of accepted instructions grows, the prompt context increasingly conditions on the model's own outputs, which is exactly what model-collapse theory ([[ch-14]]) warns about. The ROUGE-L filter is the defence; it is load-bearing.

Notice: swap the 6:2 seed:generated ratio, or raise the ROUGE threshold, and you get a different bootstrap trajectory — this is one of the two "hyperparameters" of the Self-Instruct recipe that successor papers sweep.

---

## What Self-Instruct got right that the field kept

- **Pipeline-as-pipeline framing.** Later papers cite Figure 1 as a template, not a result.
- **ROUGE-L dedup as default.** Nearly every open-source SFT recipe (Alpaca, Vicuna, Tulu-1/2, OpenHermes) ships with a ROUGE-L dedup step borrowed verbatim from Self-Instruct.
- **Seed tasks as anchor.** The 175 human-written tasks are the first explicit anchor set in the genre. Nemotron's 20K human examples, APIGen's 3,673 API references, and OMI-2's 15K MATH+GSM8K seeds are direct descendants of the same concept.
- **Classification branching.** The input-first-vs-output-first template for classification tasks prevents label-bias in the synthetic data. This is a small detail but shows up repeatedly in later pipelines (including Nemotron's topic-following track).

---

## What it got wrong, and who fixed it

- **No executor / gold-answer check** -> APIGen (execution + LLM-judge), OpenMathInstruct-2 (SymPy), code-evol-instruct (unit tests).
- **No RM-based selection** -> Nemotron-4 (RM-as-selector for preference pairs), Tulu-3 (RM-filtered SFT).
- **Surface-level dedup only** -> InsTag (tag-based dedup), Persona-Hub (persona-space diversity), Magpie (query-distribution-based diversity).
- **Single-stage mix** -> Nemotron's staged code-then-general SFT; Tulu-3's curriculum.

Every one of these successors is an instance of the same loop with one or two stages upgraded.

---

## Re-reading the filter numbers through the loop lens

The raw-data file lists three filter actions without separating stage 2 from stage 3. Our loop decomposition makes the split explicit:

- **Stage 2 (format filter):** input == output rejection, length bounds, "image/graph/file" keyword ban, ill-formatted generation drops. These catch generations that are *structurally* invalid regardless of content.
- **Stage 3 (dedup filter):** ROUGE-L > 0.7 against any existing instruction. This catches generations that are *semantically redundant* regardless of validity.

Notice: the paper conflates them because both mechanisms are drop-rules, but the cost profile and failure mode are different. Stage 2 is O(1) per candidate; stage 3 is O(n) against a growing pool. Stage 2 over-prunes by banning keywords (see "image/graph/file" wiping valid math-diagram tasks); stage 3 over-prunes by rejecting legitimately rare paraphrases of common tasks. The mitigations are different too — for stage 2, you re-examine the keyword list; for stage 3, you raise the ROUGE threshold or switch to embedding-space dedup.

For ch-18's purposes: any paper that bundles "we filtered and deduplicated" without separating the two is hiding a design choice. Self-Instruct's paper does this; the successor lineage is clearer about which rejection happens where.

## The 175 seeds in perspective

The seed count deserves one more look. The raw-data file:

> "Seed pool: 175 human-written tasks (1 instruction + 1 instance each) covering classification, generation, open-ended, extraction."

175 is a small number. For ch-18: the seed count is not a quantity you grow; it is a quality you curate. Self-Instruct's seeds were written specifically to span four task families. Nemotron's anchor is larger (~20K) because the target task families are much broader (code, QA, topic-following, function-calling, refusal). APIGen's anchor is reference-API implementations, not natural-language task definitions, so direct comparison is apples-to-oranges.

The general rule the seed-count comparisons imply: anchor size scales with **target task-family breadth**, not with target dataset size. If you need 52K outputs across 4 task families, 175 curated seeds suffice. If you need 1.4M outputs across 10+ families, you need ~20K. The amplification factor settles in a band (~70x–300x) that seems robust across pipelines. That is a concrete datapoint worth carrying into future pipeline design.

## Connections

- [[excerpts/nemotron-4]] — stage 4 (RM) + stage 5 (RM-as-selector) + stage 6 (staged mix) all upgraded simultaneously; the industrial-scale counterpart to Self-Instruct.
- [[excerpts/apigen]] — the stage-4-in-three-layers response to Self-Instruct's missing executor.
- [[excerpts/openmathinstruct-2]] — the stage-4-via-SymPy response; also shows how to hierarchically scale Self-Instruct's single-level generate stage into problem-level + solution-level stages.
- [[excerpts/nathan-lambert-synth]] — the modern framing of why Self-Instruct's missing stage 4 was both tolerable in 2022 and intolerable by 2025.
- [[excerpts/synthetic-scaling-laws]] — the 2025 pretraining-scaling story continues the Self-Instruct bootstrap lineage into a different regime (pretraining rephrase) and suggests what its successors must do to avoid the pure-generated collapse signature.
- [[ch-18]] — parent. Self-Instruct is the chapter's anchor for "the first full loop instance, with one cell deliberately left empty."
- [[excerpts/apigen]] — stage 4 upgraded to three layers; shows what Self-Instruct's empty verify stage should have been for tool-calling data.
- [[excerpts/openmathinstruct-2]] — stage 1 scaled up (K=32 per problem via 405B teacher) and stage 4 upgraded to SymPy; direct demonstration that adding a verifier transforms a Self-Instruct-shape pipeline.
- [[excerpts/nathan-lambert-synth]] — "verification is the bottleneck" is the generalisation of the specific Self-Instruct gap.
- [[ch-18]] — parent. The loop Self-Instruct instantiates, and the doctrine derived from seeing successor papers fill in its gaps.
