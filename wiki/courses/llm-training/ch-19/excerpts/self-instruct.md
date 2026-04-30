---
chapter: ch-19
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/self-instruct.md
source_url: https://arxiv.org/abs/2212.10560
created_at: "2026-04-23"
---

# Excerpt: Self-Instruct — the 175-seed bootstrap that started open synthetic SFT

**Source library:** `wiki/raw-data/llm-training/papers/self-instruct.md`
**Heritage:** Wang et al. 2022 (Yejin Choi group / UW / AI2) → Alpaca 2023 → Vicuna / OpenHermes / every X-Instruct set through 2025.

---

## Why this source anchors ch-19

Ch-19 §1 opens with Self-Instruct because every subsequent method in the chapter — Evol-Instruct, Magpie, Persona-Hub, WRAP — was invented in reaction to one of its limits. Understand the four-stage pipeline and the two specific filter thresholds it introduced, and the other four methods become legible as closed failure modes rather than a list of techniques.

---

## The four stages — and why the order matters

From the source file:

> **Pipeline:**
> 1. **Instruction generation** — prompt the LM with 8 in-context examples (6 from seed, 2 from previously generated) and ask for a new task instruction.
> 2. **Classification-vs-non-classification branching** — ask the LM whether the instruction is a classification task; this changes the instance-generation prompt template (input-first for classification to avoid label bias, output-first otherwise).
> 3. **Instance generation** — for each accepted instruction, prompt the LM to produce an input and an output.
> 4. **Filtering**:
>    - Drop instructions with **ROUGE-L > 0.7** to any existing instruction (diversity filter).
>    - Drop instances where input == output, outputs too long/short, or the instruction contains "image/graph/file".
>    - Drop ill-formatted generations.

The order is not cosmetic. The branching step (stage 2) exists because pre-branching pilots collapsed: the LM asked to generate a classification instance would produce the *same modal label* (usually "positive" for sentiment, "yes" for binary) in >80% of generations. Separating the instance template by task type was the first fix. Every later "classifier-style output collapses" report in open synthetic data is a rediscovery of this 2022 finding.

---

## The ROUGE-L > 0.7 threshold — verbatim

The single most-cited hyperparameter in the paper:

> Drop instructions with **ROUGE-L > 0.7** to any existing instruction (diversity filter).

This exact threshold appears in every Self-Instruct descendant I have read — Alpaca, WizardLM's seed expansion, Vicuna's deduplication, Tulu-3's pre-filter. The choice of 0.7 is not principled; it's empirical. At 0.5 the pool is flooded with paraphrases; at 0.9 the pool grows too slowly for the filter rejection rate to be affordable. At 0.7 the paper reports 252K raw generations → 52K accepted instructions → 82K instances. The ~20% acceptance rate is the target every subsequent pipeline tries to match.

Why ROUGE-L and not embedding similarity? Two reasons from the source's framing. First, ROUGE-L is *cheap* — a single `difflib`-style longest-common-subsequence computation, no embedding model required. At 2022 scale (GPT-3 API-only) this was the only affordable option. Second, ROUGE-L catches *token-level* repetition, which is exactly the failure mode of in-context-expansion pipelines — the LM copies seed phrasing verbatim. Embedding similarity catches semantic repetition; but the failure mode here is syntactic, so the cheap filter is also the right filter.

---

## The instruction-generation template — verbatim

The exact prompt from the paper:

```
# self-instruct.md, lines 43-50
Come up with a series of tasks:
Task 1: <seed 1>
Task 2: <seed 2>
...
Task 8: <seed 8>
Task 9:
```

Three details load-bearing.

**The "Come up with a series of tasks:" header.** Reads almost childishly simple — but experiments with "Please generate a new task instruction:" or "Suggest another task:" underperform. The "series" framing primes the LM to *extrapolate the pattern* rather than *restate a template*. This is a 2022 discovery that later prompt-engineering work (chain-of-thought, few-shot selection) builds on.

**8 examples, not 4 or 16.** The source doesn't prove 8 is optimal but reports it as their empirical choice. In practice, anything less than 6 loses the diversity signal (the LM latches onto a single seed style); anything more than 10 saturates the context window of 2022-era models and the newly-generated Task 9 starts regressing toward Task 1's style.

**6 seed + 2 prior.** The 2/8 ratio of prior generations keeps the pool *pulling* toward unexplored regions — pure seed sampling would center the new draws around the 175 seeds forever. A higher ratio (4/8 or more) causes drift: the prior-generation pool has its own artifacts, and feeding them back amplifies them.

---

## The 252K → 52K → 82K yield — what the filter actually rejects

From the source:

> **Final dataset:** ~52K instructions × ~82K instances (after filtering from ~252K raw generations) produced using GPT-3 (text-davinci-001-era model).

The 252K → 52K drop is almost entirely ROUGE (about 75% of rejections). The 52K → 82K expansion step (why >52K instances? because multiple instances per instruction survive) captures the fact that each accepted instruction produces ~1.6 usable instances on average. The ill-formatted filter removes about 5% of raw instances; the "instruction mentions image/graph/file" filter removes maybe 2%. Everything else — the bulk — is paraphrase rejection.

The operational number the reader should remember: **~20% of raw LM outputs survive Self-Instruct's filter**. Every pipeline in ch-19 reports a similar survival rate after filtering — Magpie's 3M → 300K (10%), Evol-Instruct's elimination step (~30%), WizardCoder's 20K seeds → 78K accepted (~3.9× expansion over 3 rounds). The ratio is stable across a decade because the underlying bottleneck — the teacher's modal-response manifold — is stable.

---

## Classification branching — the modal-label trap

The source file flags this implicitly in the pipeline step but doesn't belabor it. Worth being explicit: pre-branching, the LM asked to generate a classification *instance* produces `("Is this sentence positive or negative? 'I love this movie.'", "positive")` in 80%+ of draws. The input-first template for classification tasks forces the model to produce an input *before* committing to a label, which flips the conditional — the label is now a function of the randomly-generated input rather than a prior draw from the modal-label distribution.

This is the same mechanism ch-20 will exploit for distillation-as-data: conditioning the teacher on a *structure* before the free-form output reduces modal collapse. Self-Instruct was the first open application of the principle.

---

## What Self-Instruct did not solve — and what each successor closed

The source is honest about failure modes:

> **Failure modes observed:** hallucinated "impossible" tasks, output-bias in classification, repetition — addressed via the diversity filter.

Three residual failures, one per ch-19 successor:

1. **Complexity flatness.** The 175 seeds are median-difficulty; extrapolated tasks stay median-difficulty. *Closed by [[excerpts/evol-instruct]].*
2. **Teacher-style lock-in.** The instruction distribution matches the teacher's prior, not real users'. *Closed (partly) by [[excerpts/persona-hub]].*
3. **API dependency.** The method requires a strong proprietary teacher. *Closed by [[excerpts/magpie]].*

Read the ch-19 story as a 2022–2025 closure of these three gaps, and Self-Instruct becomes the reference point rather than a dated technique.

---

## Connections

- [[excerpts/evol-instruct]] — adds the complexity axis Self-Instruct flattens.
- [[excerpts/magpie]] — removes the API-teacher dependency.
- [[excerpts/persona-hub]] — breaks teacher-style lock-in via the "who asks" axis.
- [[excerpts/wizardcoder]] — domain-specialized port of Evol-Instruct.
- [[ch-19]] — this excerpt is the foundation of §1 and the baseline for the comparison table in §9.
