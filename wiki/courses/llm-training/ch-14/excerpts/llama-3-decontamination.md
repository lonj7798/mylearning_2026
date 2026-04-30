---
chapter: ch-14
course: llm-training
phase: read
excerpt_of: Grattafiori et al. 2024 — "The Llama 3 Herd of Models" (data section)
source_url: https://arxiv.org/abs/2407.21783
created_at: "2026-04-23"
---

# Excerpt: Llama 3 Decontamination and Data Budget

**Source:** `wiki/raw-data/llm-training/model-reports/llama-3.md`
**Primary paper:** Aaron Grattafiori et al. (Meta Llama Team), "The Llama 3 Herd of Models", 2024
**arXiv:** https://arxiv.org/abs/2407.21783

---

## Bibliographic header

Llama 3 is the most thoroughly documented frontier pretraining run from 2024. For the ch-14 data-scaling topic, the load-bearing numbers are the token budget, the decontamination n-gram size, and the dropout rate.

From the raw-data notes:

> *"Llama 3 is pre-trained on 15.6T tokens and post-trained via six rounds of SFT + Rejection Sampling + DPO."*

15.6T unique tokens is roughly the upper end of what public-web English pretraining can reach in 2024 after aggressive filtering. Llama 4 and subsequent Meta releases will need to either (a) dip into the `R > 1` regime per Muennighoff (see [[excerpts/data-constrained-scaling]]) or (b) lean harder on synthetic data — the post-training recipe already does this.

---

## The data-budget signal

Llama 3 405B:
- **Pretraining tokens:** 15.6T (≈ `R = 1` on the unique-token pool)
- **Context:** 8K native, extended to 128K
- **Compute:** 3.8e25 FLOPs
- **Chinchilla ratio:** `D/N ≈ 15.6T / 405B ≈ 38.5` — *overtrained* (Chinchilla-optimal would be ~20).

The 2× overtraining is deliberate: Meta optimises for inference cost. Chinchilla tells you the compute-optimal model for a given training budget; they want the *inference*-optimal model for a given deployment use-case. A 2× smaller model trained 2× longer has the same training loss but ~half the inference cost.

**What Llama 3 does not do:** it does not repeat tokens. At `R = 1` for bulk pretraining, Muennighoff's decay is barely engaged (`w(1) = 1.0`). The repetition savings sit unused. Meta appears to judge that scraping another 5T tokens is harder than spending more compute.

---

## Decontamination procedure — 8-gram overlap

From the raw-data notes:

> *"Rejection sampling: for each prompt, sample K=10–30 completions from the best round-(N-1) chat model at temperature T=0.6–1.0, then keep the top by RM score. Filtering: topic classifier + quality classifier (both distilled from Llama 3) remove low-quality rejection-sampled text before SFT."*

The pretraining decontamination pipeline (documented in the Llama 3 paper's data section, not reproduced verbatim in the raw-data notes but summarised in the ch-14 recipe):

```
1. Enumerate all eval suites that Llama 3 will be evaluated on:
   MMLU, GSM8K, MATH, HumanEval, BBH, ARC, AGIEval, CommonsenseQA, ...

2. For each eval sample, extract 8-gram n-grams from (question + answer).
   Build a Bloom filter per eval.

3. Stream pretraining documents:
   For each document d:
     Count 8-grams from d that hit any eval Bloom filter.
     If overlap_fraction(d, eval) > 0.5 for any eval: drop d.
```

Meta reports dropping `< 0.1%` of pretraining tokens at this threshold — the procedure is not expensive in data yield. The 0.5 threshold is permissive — a document has to be more than half-composed of eval-suite n-grams to be dropped. For reasoning-sensitive evals, stricter thresholds (~0.1) are applied.

**Why 8-gram.** The 2024 standard settled at 8 because:
- 4-grams catch paraphrases but over-flag common phrases (FP >> FN).
- 8-grams catch substantial question-stem reproduction while passing common English.
- 13-grams catch only verbatim; paraphrases leak through.
- 20-grams essentially only catch copy-paste.

See `ch-14/read.md` §4 for the full false-positive / false-negative discussion.

---

## The iterative post-training recipe as a decontamination amplifier

The 6-round SFT → Rejection Sampling → DPO loop is not only about preference optimisation. Each round re-mines training data from the current best model's outputs, which:

1. Replaces scraped web content with model-generated content — reducing eval leakage from unexpected web sources.
2. Filters through Llama 3's own topic and quality classifiers, which are trained to reject low-quality / potentially contaminated content.
3. Uses the reward model to down-weight outputs that look "memorised" — an implicit anti-contamination signal.

The full post-training data mix (from the raw-data notes):

> *"~50–80% synthetic rejection-sampled data. Remainder: human SFT demonstrations, preference data, capability-specific synthetic (code-exec-filtered code, math with verifier, multi-turn tool use traces, long-context QA)."*

This composition is contamination-aware by construction: human-anchored, verifier-filtered, and iteratively re-mined.

---

## What Llama 3 is silent about

Three omissions worth noting for a learner:

1. **No explicit `R_T` fit.** Meta does not publish their own Muennighoff-style decay constant. The fact that they ran at `R = 1` suggests they did not feel the need to repeat, not that they measured decay.
2. **Decontamination audit is not open.** The dropped-document log is not released. OLMo 3's OlmoTrace ([[excerpts/olmo-3-decontamination]]) is the open counterpart.
3. **Synthetic-contamination estimate `p_synth` is not reported.** Meta does not claim what fraction of the 15.6T scrape is machine-generated. Given Dohmatob's 1% flatlining bound ([[excerpts/model-collapse]]), this is a non-trivial gap.

For a frontier lab replicating Llama 3 in 2026, these three numbers are where you would differ — publish them and you have a more scientifically defensible pretraining run.

---

## Connections

- The scaling law that Llama 3 sits on: [[excerpts/data-constrained-scaling]]
- The open-lab counterpart: [[excerpts/olmo-3-decontamination]]
- The contamination-weaponisation angle: [[excerpts/anthropic-sleeper-agents-data]]
- Chapter synthesis: [[ch-14]]
