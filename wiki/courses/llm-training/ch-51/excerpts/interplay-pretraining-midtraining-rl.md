---
chapter: ch-51
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/interplay-pretraining-midtraining-rl.md
source_url: https://arxiv.org/abs/2512.07783
created_at: "2026-04-23"
---

# Excerpt: Interplay of pre-training / mid-training / RL — paired eval as the only way to separate true gains from rearranged mass

**Source library:** `wiki/raw-data/llm-training/papers/interplay-pretraining-midtraining-rl.md`
**Artifact:** RL yields capability gains only when pre-training leaves headroom; extrapolative vs contextual generalization; pass@128 as the right metric.

---

## Why this source is ch-51's "paired" anchor

Source §Core Insight:

> Reasoning gains are not attributable to RL alone: the paper's controlled setup shows that RL only produces real capability expansion when pre-training leaves headroom, mid-training installs usable priors, and the RL tasks sit near the model's edge of competence.

The ch-51 question the paper implicitly answers: when RL "improves" a model, is that (a) true new capability, or (b) rearranged probability mass on existing capability? The distinction is invisible to pass@1. It is detectable only with paired evaluation on the same prompts at higher pass@k. Ch-51 §4's paired-bootstrap requirement is the right statistical machinery for this distinction.

---

## Extrapolative vs contextual — two eval regimes, two CI interpretations

Source §Technical Details / Experimental design:

> Extrapolative: composing operations into harder problems.
> Contextual: reusing the same reasoning under different surface contexts.

Ch-51 §1's noise table needs an extension when the eval tests *extrapolative* generalization. Rare items (hard compositions) give per-item scores near 0; p(1-p)/N variance is actually *smaller* than at p=0.5, but the *effective* information per item is also smaller — one successful extrapolation is highly informative, one failed one is ambiguous between "can't do this" and "made an arithmetic slip." A paired sign test on matched hard items is the right robust estimator here.

---

## pass@128 as the CI-friendly metric for RL gains

Source §Key Figures/Tables to Study:

> Headroom / boundary experiments: the most important evidence for when RL truly adds capability.

pass@k with k large (128 in this paper) reduces decode variance — the test is "at least one success out of k samples" — which zeros out a major ch-51 §1 noise source. At k=128, per-seed σ collapses to near zero; the remaining noise is item-count. Ch-51's guideline: for "true capability" claims in RL, report pass@k at k≥16 paired against the baseline at the same k, not pass@1 at T=0.

---

## Mid-training as the compute-efficient alternative — and why its gains need CI-bracketing

Source §Key Contributions:

> Finds that mid-training can outperform RL-only training under fixed compute.

A "mid-training beats RL at same compute" claim is a *delta* claim, not a *level* claim. Ch-51 §4 paired bootstrap is the right test: run both arms at matched eval sets, compute per-item paired deltas, bootstrap. Because the compute budgets are matched, the variance sources are nearly identical across arms — the paired structure cancels the most noise possible, maximizing statistical power.

---

## Process-level rewards — ch-51 relevance

Source §Technical Details / Process supervision:

> The paper adds process-level verification to outcome rewards to reduce reward hacking.

Process rewards densify the signal, which means smaller per-item variance on the reward, which means tighter CIs on the downstream eval with the same N. Ch-51 §1's "effective N" framing (inherited from [[scaling-laws-data-quality]]) suggests that process-supervised models effectively have larger N_eff on reasoning evals because their per-item outputs are more self-consistent across seeds.

---

## What ch-51 takes from this paper

- **Pair, don't subtract.** RL-vs-baseline needs paired comparison at matched prompts, never unpaired CI subtraction.
- **pass@k at high k** collapses decode variance; prefer it for capability claims.
- **Boundary-at-competence** prompts have the highest information density — design the eval to oversample these prompts.

## What ch-51 does NOT take

- The synthetic-atomic-operation eval setup is task-specific; ch-51's machinery is benchmark-agnostic.
- The "extrapolative vs contextual" taxonomy is upstream of ch-51's slice definition; ch-50 owns it.

---

## Connections

- **[[rlvr-beyond-base-model]]** — the broader "does RL add capability or rearrange mass" debate.
- **[[prorl]]** — policy-intervention arguments that need paired evaluation to settle.
- **[[front-loading-reasoning]]** — motivates early-stage exposure; ch-51 is its eval-side dual.
- **[[math-shepherd]], [[lets-verify]]** — process-supervision lineage.
- **ch-51 §4 paired bootstrap** — the tool that makes "RL truly improved" vs "RL rearranged mass" decidable.
- **ch-51 §1 variance budget** — pass@k at high k is the decode-variance mitigation this paper's evals rely on.
