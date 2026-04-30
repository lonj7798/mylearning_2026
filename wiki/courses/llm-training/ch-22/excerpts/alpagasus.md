---
chapter: ch-22
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/alpagasus.md
source_url: https://arxiv.org/abs/2307.08701
created_at: "2026-04-23"
---

# Excerpt: AlpaGasus — the LLM-as-rater baseline

**Source library:** `wiki/raw-data/llm-training/papers/alpagasus.md`
**Paper:** Chen et al. 2023, "AlpaGasus: Training a Better Alpaca with Fewer Data" (UMD + Samsung).

---

## Why this source anchors ch-22

AlpaGasus is the simplest filter in the chapter and the one every later paper cites as baseline. Three load-bearing facts for ch-22:

1. **52K → 9K produces a *better* model.** Marginal samples have negative value.
2. **5.7× training-time speedup** — the operational payoff that made filtering boring-industrial rather than research-curiosity.
3. **Threshold sensitivity.** 4.5 > 4.0 > 3.5; rater uncertainty clusters near the decision boundary.

---

## The rubric prompt — paraphrased

From the source (lines 28-34), ChatGPT receives the `(instruction, input, response)` triple and a rubric that asks it to rate on a 0-5 scale:

- **Relevance** of response to instruction.
- **Correctness** of factual claims.
- **Completeness** of coverage.
- **Format appropriateness** (code-block when code was requested, list when list was requested, etc.).

Final score = mean of the four sub-scores.

The averaging is not incidental. A 5/5-relevant / 3/5-correct response scores 4.0. At a threshold of 4.5 it is dropped; at 3.5 it survives. The AlpaGasus sweep (4.5 beats 4.0 beats 3.5) says: **the marginal-4.0 samples are a rater-uncertainty signal, not a weak-quality signal**. They are the samples the rater could not confidently approve, and they drag training down.

---

## Headline numbers

From the source (lines 14-15, 42-45):

- **Pool:** Alpaca 52K self-instruct set.
- **Survivors at threshold 4.5:** ~9K (17.3%).
- **Training speedup:** 5.7× (80 min → 14 min on 7B).
- **Win rate:** AlpaGasus-7B > Alpaca-7B on Vicuna, Koala, Self-Instruct, WizardLM benchmarks (all GPT-4-judged).
- **AlpaGasus-13B** reaches ≥90% of text-davinci-003's test performance.

The training-speedup number is more important than it reads. It means a lab can iterate on the SFT recipe 5–6× more in a week, which compounds the eventual quality lead. Filtering is a *velocity* multiplier, not just a quality one.

---

## What AlpaGasus misses — the ch-22 reading

From the source (lines 47-51):

- **Teacher rubric bias** — the rater's quality model is not aligned with downstream capabilities it cannot evaluate well (math, code-exec, factual precision).
- **Threshold tuning is dataset-specific** — 4.5 is *not* a universal constant.
- **Single-axis scoring** — DEITA later decomposes this into quality × complexity × diversity.
- **GPT-4-judge evaluation inherits known judge biases** (length, style, sycophancy).

This is why ch-22 §2 positions AlpaGasus as the *cheapest* baseline, not the *best* one. It answers "is this response acceptable to a reasonable rater?" — nothing more, nothing less. When that is the dominant failure mode of your pool (raw self-instruct output, 2023-style), it works. When your pool is filtered through a stronger teacher and the dominant failure mode is *redundancy* or *capability gap*, AlpaGasus cannot see it.

---

## Operational notes

- **API cost** is the dominant expense. Scoring 52K samples with ChatGPT at 2023 rates was a few hundred dollars; at 300K-sample pools it becomes non-trivial.
- **Parallelism** — the rating is embarrassingly parallel across samples. One request per sample with a ~200-token response; batch with high concurrency.
- **Deterministic seeding** — fix temperature = 0 on the rater, or average over 3 rater samples per item for more stable scores.
- **Log every rating** — keep the raw (score, reasoning) JSON per sample for later threshold sweeps and for diagnosing systematic mis-rates.

---

## Connections

- **[[ch-22]]** §2 — the LLM-as-rater slot in the filter taxonomy.
- **[[deita]]** — extends AlpaGasus with complexity and diversity axes.
- **[[cherry-llm]] / [[ifd]]** — alternative that eliminates the external rater.
- **[[ultrafeedback]]** — scales the AlpaGasus rating pattern 10× for preference data.
- **[[lima]]** — the human-curated limit of the same idea.
