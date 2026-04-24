---
chapter: ch-23
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/strong-model-collapse.md
source_url: https://arxiv.org/abs/2410.04840
created_at: "2026-04-23"
---

# Excerpt: Dohmatob et al. (ICLR 2025 Spotlight) — Strong Model Collapse breaks scaling at 1%

**Source library:** `wiki/raw-data/llm-training/papers/strong-model-collapse.md`
**Authors:** Elvis Dohmatob, Yunzhen Feng, Arjun Subramonian, Julia Kempe (Meta / NYU)
**Venue:** ICLR 2025 Spotlight; arXiv 2410.04840

---

## Why this source anchors ch-23

Ch-23 §2 is this paper, condensed. The paper's contribution is not qualitative — everyone post-Shumailov already expected contamination to hurt — but *quantitative and asymptotic*: it gives the scaling-law-level statement that even a 1% synthetic fraction introduces an **irreducible** error term that no amount of real data can wash out. This upgrades Shumailov's "don't replace real with synthetic" warning to "don't mix *any* unverified synthetic into pretraining" — and forces the verifier-gate question to the center of the chapter.

---

## The main theorem — schematic form

```
# strong-model-collapse.md, lines 29-32
Let `E[R_test]` be expected test risk; in the real-only case
`E[R_test] ~ f(N)` decreases with `N` under standard scaling.
Under synthetic fraction `p > 0`,
  E[R_test] ≈ f(N) + c(p) · σ_synth²
with `c(p) > 0` for any `p > 0`.
```

The asymptote is now a function of `p`, not `N`. Scaling flatlines. The paper derives the explicit form of `c(p)` under a random-projection approximation of neural networks (operator-valued free probability — the technical machinery is dense but the takeaway is the formula).

**What this means for a practitioner.** If you add synthetic at fraction `p = 0.01` with `σ_synth²` comparable to real-data variance, your scaling law *departs from the real-only baseline* early and never catches up. You can spend 10× the compute on 10× the data and still sit at a worse asymptote than a 1×-data real-only run. The paper's empirical GPT-2-scale reproduction confirms this shape — the 1%-contamination curve visibly flatlines against the real-only baseline within the measured range.

---

## The interpolation-threshold phase diagram

```
# strong-model-collapse.md, line 26 + "Findings"
Below the interpolation threshold larger models amplify collapse;
beyond it, larger models partially mitigate but never eliminate it.
```

The second theorem (phase diagram) layers model size on top of synthetic fraction. Two regimes:

1. **Under-parameterized (below interpolation).** Larger models *worsen* collapse. Intuition: when the model cannot interpolate the training set, it spreads error across the data — including the synthetic portion — and synthetic error dominates the loss contribution.
2. **Over-parameterized (above interpolation).** Larger models partially mitigate collapse. Intuition: they can fit the real portion while treating synthetic as noise to average over. But the mitigation is partial; `c(p) > 0` for all `p > 0` even in this regime.

The policy-relevant corollary: "we'll just scale up to fix contamination" is mathematically wrong. Size helps only past interpolation and only partially. The only way to *eliminate* `c(p)·σ²` is to eliminate `p` or eliminate `σ_synth²` — i.e., remove contamination or verify synthetic into real-equivalent quality.

---

## Why this is a scaling-law statement, not just a collapse statement

Shumailov 2024 ([[excerpts/model-collapse]]) proved iterated recursion collapses. A natural defensive reading: "that's a closed-loop failure, real pipelines have open loops, we're safe." Strong Model Collapse blocks this defense. The setup is a *single training run* (not iterated) on a *fixed mixture* of real and synthetic. No recursion. No iteration. Just one run, one mixture, measured asymptotically in `N`.

The result: even the single-run, non-iterative setting loses its scaling benefit at 1% contamination. The mechanism is not sample compounding across generations; it is **asymptotic bias** introduced by synthetic's distributional offset from real.

This matters because it changes what pipelines are at risk:

- **Clean recursive-iteration loops (Shumailov's concern):** affected.
- **Open-web pretraining with accumulated LLM-generated text (the 2025 policy concern):** **also affected**. No iteration required.
- **Mid-training mixes with some synthetic:** affected.
- **SFT sets with any synthetic fraction:** affected.

The one escape: make `σ_synth²` small. A *verified* synthetic corpus — where every sample has passed an external check and is indistinguishable from real at the distribution level — has `σ_synth² → σ_real²` and the bias term collapses. This is the theoretical basis for ch-23's §4 (verification as defense) and §5 (gate vs no-gate table).

---

## The empirical reproduction — GPT-2 scale

The paper runs the theoretical prediction against real LM training at GPT-2 scale:

- Base: GPT-2-class model (exact size per paper).
- Training data: real corpus with 0% vs 1% synthetic injection (synthetic = prior GPT-2 outputs).
- Measurement: validation loss vs training-set size.

Result: the 1%-contamination curve **departs from the 0% baseline early and flatlines**. The departure is visible within the first order of magnitude of data scaling and widens with `N`. No model-size sweep closes the gap.

The paper explicitly cautions that the random-projection approximation is a simplification of real deep nets; the empirical reproduction supports but does not prove the full theory. But two independent lines of evidence (theory + experiment) agreeing is what makes this a ICLR Spotlight-class result.

---

## The synthetic-quality loophole — ch-23's §4 bridge

```
# strong-model-collapse.md, lines 47-48
Synthetic quality matters — the theory assumes "synthetic" = iid from
a model; high-quality filtered synthetic (e.g., via an external verifier)
behaves like real in the limit.
```

This is the most important sentence in the paper for ch-23's narrative. The pessimism is **assumption-dependent**. The assumption is that synthetic = iid samples from a generator model, untransformed. Under that assumption, `c(p) > 0` always.

If you relax the assumption — if "synthetic" means "samples that passed a verifier independent of the generator" — then the distributional gap `σ_synth² - σ_real²` shrinks with verifier strength. A perfect verifier makes verified-synthetic indistinguishable from real; an imperfect verifier reduces but does not eliminate the gap. The Zhang et al. 2025 analytical convergence guarantee ([[excerpts/faithful-synth-eval]]) is the formalization of this loophole.

This is why ch-23 §6 is titled "Canonical gate designs." The gates are not optional — they are what turns "synthetic" from a scaling-law poison into a scaling-law-neutral ingredient. Every production pipeline that scales without collapsing has a gate that makes this loophole real.

---

## The open-web contamination corollary

```
# strong-model-collapse.md, line 48
Policy implication (not the paper's claim): as open web accumulates
LLM-generated text, all future pretraining corpora will be contaminated
— hence active curation / provenance becomes mandatory.
```

This is the 2025 policy frame. Every subsequent pretraining corpus built from open-web crawl inherits some fraction of LLM-generated content. The question for the next generation of pretraining is not "how do we avoid synthetic" but "how do we filter or verify the synthetic we cannot avoid." Scale-up alone won't save us; Strong Model Collapse is a mathematical statement that `c(p) · σ²` doesn't die with bigger data.

Operationally for 2026 pretraining:

- Measure the synthetic fraction in your corpus (provenance-based if possible, detector-based as fallback).
- Apply filters that shrink `σ_synth²` (deduplication, fluency filters, tail-preservation checks).
- Budget the acceptable `c(p) · σ²` floor against your scaling target.

Ch-23 §7's dashboard is this recipe operationalized.

---

## What the paper does NOT claim

Three important non-claims:

1. **Not "all synthetic is worthless."** The theorem is about iid-model synthetic; verified synthetic is a different regime.
2. **Not "1% is the magic threshold."** 1% is the regime-specific empirical number for GPT-2-scale. Other regimes have different thresholds. The universal statement is "`c(p) > 0` for `p > 0`."
3. **Not "model size is irrelevant."** The phase diagram shows size matters; it just doesn't eliminate `c(p)`.

Readers who conflate the paper with "synthetic data is bad" miss the structural point, which is that the *loop* matters, not the content.

---

## Connections

- [[excerpts/model-collapse]] — the iterative-recursion paper this one sharpens into an asymptotic statement.
- [[excerpts/faithful-synth-eval]] — Zhang et al. 2025's convergence-under-verification is the formalization of the synthetic-quality loophole.
- [[excerpts/synthetic-data-scaling-laws]] — empirical evidence that rephrased synthetic (which is implicitly verified by paraphrase anchoring) survives scaling, while pure-generated (which satisfies this paper's assumptions) does not.
- [[excerpts/prismatic-synthesis]] — gradient-coverage selection shrinks `σ_synth²` by actively filling underpopulated regions; an anti-`c(p)` mechanism.
- [[ch-23]] — §2 is this paper, §5 and §6 are responses to its loophole.
