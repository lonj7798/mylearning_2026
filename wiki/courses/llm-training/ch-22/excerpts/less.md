---
chapter: ch-22
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/less.md
source_url: https://arxiv.org/abs/2402.04333
created_at: "2026-04-23"
---

# Excerpt: LESS — gradient-similarity selection for targeted SFT

**Source library:** `wiki/raw-data/llm-training/papers/less.md`
**Paper:** Xia et al. 2024 (ICML 2024 Spotlight), "LESS: Selecting Influential Data for Targeted Instruction Tuning" (Princeton).

---

## Why this source anchors ch-22

LESS is the first paper to make influence-function-based data selection practical at LLM scale. Three contributions that every later gradient-based method (including [[prismatic-synthesis]]) builds on:

1. **Adam-adjusted gradients.** Vanilla influence-function theory assumes SGD; LESS shows this mis-ranks samples at LLM scale and fixes it.
2. **LoRA warmup.** A short LoRA pass on the pool stabilizes per-sample gradients enough for downstream similarity calculations.
3. **Random-projection datastore.** Store 8K-dim projected gradients once, reuse across many target queries.

The ch-22 reading: LESS is the *targeted* filter. It selects for gradient alignment with a named capability. Contrast with Prismatic's *coverage* objective.

---

## The influence-function setup

From the source (lines 28-33):

Classical influence theory: the effect on validation loss of up-weighting training sample `x_i` by `ε` is

```
d L_val / d ε  ≈  - η · g_val^T H^{-1} g_i
```

where `g_i = ∇_θ L_train(x_i)` and `g_val = ∇_θ L_val`. The Hessian inverse is intractable at LLM scale. LESS drops the H^{-1} and uses direct cosine similarity of projected gradients.

---

## The Adam adjustment — why it matters

From the source (lines 19, 60):

> Adam-aware influence formulation — fixes a bias present in naive SGD-influence at LLM scale.

Under Adam the effective per-step update is

```
θ ← θ - η · m̂ / (√v̂ + ε)
```

not `θ ← θ - η · g`. The per-dimension rescaling by `1 / (√v̂_k + ε)` means raw gradient magnitude is *not* what drives the update. LESS replaces each `g_{i,k}` by the Adam-effective quantity `g_{i,k} / (√v̂_k + ε)` (using the Adam state at the end of LoRA warmup) before similarity.

Empirically: SGD-influence-ranked vs Adam-influence-ranked top-5% subsets disagree on ~30% of samples, and the Adam-ranked subset trains measurably better. This is the subtle case where optimizer choice leaks into the data-selection algorithm — you cannot apply LESS correctly without matching its gradient-adjustment to the optimizer you train with.

---

## The datastore construction

From the source (lines 34-44):

1. **LoRA warmup.** Train LoRA adapters on the pool for ~4% of the full SFT budget. This reads all data cheaply and stabilizes per-sample gradients at a representative `θ`.
2. **Gradient computation.** For each pool sample, compute the Adam-adjusted per-sample gradient at the post-warmup `θ`.
3. **Random projection.** Project to `d ≈ 8K` dims via fixed Gaussian random projection (Johnson–Lindenstrauss — preserves inner products to additive ε).
4. **L2-normalize** and store.

The datastore is `|pool| × 8K` floats — ~12 GB for a 400K-sample pool at fp32. Reusable across many target queries.

---

## The target query

From the source (lines 39-41):

1. Take a few-shot target set (say, 5 MMLU exemplars).
2. Compute the same Adam-adjusted projected gradient for each.
3. Average and L2-normalize to get `g_val`.
4. Rank pool samples by `<g_i, g_val>` (cosine similarity).
5. Keep top 5%.

Five MMLU exemplars can steer a 400K-pool selection. The few-shot set is not training data — it is a *direction* in gradient space, not a capability definition.

---

## Headline results

From the source (lines 53-55):

- **5% LESS-selected** ≥ 100% random on MMLU, BBH, TydiQA.
- **Model-size transfer**: 7B-built datastore selects data that improves 13B training.
- **Model-family transfer**: Llama → Mistral datastore sharing works.

The transferability is surprising and important. It says the gradient-geometry of SFT data is *mostly* model-agnostic at the level of ranking — you can build the datastore once on the cheap model and reuse it across the frontier lineup.

---

## What LESS does not give you

From the source (lines 57-62):

- **Target-set dependence** — selection is only as good as the few-shot exemplars. Bad exemplars → bad selection.
- **No quality gate** — LESS optimizes for similarity, not correctness. Gradient-aligned hallucinations score highly.
- **No coverage** — LESS can happily pick 5% of samples that all cover the same MMLU corner.
- **Datastore build is nontrivial compute** — justifiable only if you run many selection queries against it.

Compose LESS with a verifier (answer-check for math, execution for code) and with an explicit coverage strategy (Prismatic's gradient entropy is the natural match).

---

## The LESS-vs-Prismatic contrast

LESS asks: *which samples align with this target?* — alignment objective.
Prismatic asks: *do my samples cover the gradient manifold?* — coverage objective.

Same geometry (projected Adam-adjusted gradients), opposite operations. Both are correct for different questions. Compose them: Prismatic for the generation-side coverage objective, LESS for the selection-side capability alignment.

---

## Connections

- **[[ch-22]]** §6 — the gradient-alignment slot; derivation of the Adam adjustment.
- **[[prismatic-synthesis]]** — same gradient primitive, coverage objective.
- **[[cherry-llm]] / [[ifd]]** — alternative that avoids gradients entirely.
- **[[deita]]** — scorer-based alternative; different geometry.
- **[[alpagasus]]** — no-gradient rater-based alternative.
- **[[tulu-3]]** — downstream consumer of targeted-SFT recipes.
