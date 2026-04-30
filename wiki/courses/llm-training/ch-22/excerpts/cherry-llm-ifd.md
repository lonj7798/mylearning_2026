---
chapter: ch-22
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/cherry-llm.md, wiki/raw-data/llm-training/papers/ifd.md
source_urls:
  - https://arxiv.org/abs/2308.12032
created_at: "2026-04-23"
---

# Excerpt: Cherry-LLM and the IFD score

**Source library:** `wiki/raw-data/llm-training/papers/cherry-llm.md`, `wiki/raw-data/llm-training/papers/ifd.md`
**Paper:** Li et al. 2023/2024, "From Quantity to Quality: Boosting LLM Performance with Self-Guided Data Selection for Instruction Tuning" (NAACL 2024).

---

## Why this source anchors ch-22

IFD is the first filter that runs *without an external judge*. The target model scores its own SFT pool. This is a rare property — most later filters ([[alpagasus]] needs ChatGPT, [[deita]] needs a trained 13B scorer, [[less]] needs a target few-shot set) need outside signal. Cherry-LLM needs nothing but the target model and two forward passes per sample.

The ch-22 reading: IFD is the cheapest signal that is *data-intrinsic* rather than rater-dependent.

---

## The exact definition

From `ifd.md` (lines 14-18):

```
PPL_cond(a | q) = exp( -(1/|a|) · Σ_t log p_M(a_t | q, a_{<t}) )
PPL_uncond(a)   = exp( -(1/|a|) · Σ_t log p_M(a_t |    a_{<t}) )
IFD(q, a)       = PPL_cond(a | q) / PPL_uncond(a)
```

Two forward passes: one with the full prompt-response sequence, one with the response alone (no instruction). Per-token log-probs on the *response* tokens only. Ratio of perplexities.

---

## Interpretation of the three regimes

From `ifd.md` (lines 22-27):

| IFD range | Meaning | Keep? |
|---|---|---|
| `< 1` | Instruction reduces response uncertainty — informative | yes, if near the upper boundary |
| `≈ 1` | Instruction adds no signal — decoupled pair | drop |
| `> 1` | Instruction *hurts* — distributional mismatch | drop |

The cherry samples sit just below 1: the model still has uncertainty, but the instruction genuinely helps. These are the samples that carry real learning signal for SFT.

---

## Why warmup is mandatory

From `ifd.md` (lines 28-31):

> Warm the target LM on ~1K random samples for 1 epoch (avoids cold-start mis-calibration).

Cold `PPL_uncond` is badly calibrated on synthetic response distributions — the untuned base model does not know the chat template, does not know that responses start with "Sure, here's..." patterns, does not emit the tokenizer's special tokens. The warmup is small (~1K random) but aligns the target with the pool's surface conventions so the unconditional perplexity estimate is fair. Skip it and IFD becomes dominated by template-mismatch noise rather than real conditioning signal.

---

## Why IFD catches synthetic-pipeline failures

A dominant failure mode of [[self-instruct]]-style pipelines is **glued pairs**: the generator produced an instruction, then separately produced a response, and the two do not actually match. Surface-rater filters ([[alpagasus]]) miss this if the response *looks* fluent; the rater gives it 4/5 for relevance because it did not cross-check. IFD catches it: a decoupled pair has `PPL_cond ≈ PPL_uncond`, so `IFD ≈ 1`, and the sample is dropped.

This is why IFD pays on raw synthetic output that has not yet been quality-gated. It is picking up a *different* failure mode than LLM-as-rater — one that is structurally common in synthetic pipelines and structurally invisible to a rater reading one sample at a time.

---

## What IFD does not check

From the source (lines 48-52):

- **Not factual correctness.** A confidently-conditioned lie scores well.
- **Not diversity.** Top-K by IFD can select near-duplicates.
- **Not dataset-level coverage gaps** — selection subtracts; it cannot add a capability the pool lacks.

Compose with a verifier (math answer-check, code execution) for correctness and with a diversity constraint (embedding or gradient) for coverage. IFD alone is a first-stage de-noise, not a complete curation recipe.

---

## The Cherry-LLM pipeline

From `cherry-llm.md` (lines 35-41):

1. Fine-tune target on ~1K random subset for 1 epoch (warmup).
2. Compute IFD for every sample in the pool.
3. Sort desc within `IFD < 1`; keep top 5–15%.
4. Full SFT on the cherry set.

That is the whole recipe. No external rater, no gradient datastore, no rubric prompt.

---

## Connections

- **[[ch-22]]** §3 — the IFD slot; derivation of the ratio.
- **[[superfiltering]]** — GPT-2-125M IFD rankings transfer to Llama-2-7B; ~20× speedup.
- **[[alpagasus]]** — contrasting external-rater approach.
- **[[deita]]** — complementary; IFD is a cheap first-stage before DEITA's complexity × quality × diversity.
- **[[less]]** — orthogonal gradient-based signal.
- **[[lima]]** — the human-curated limit; Cherry-LLM is its machine-curated equivalent.
