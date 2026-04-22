<!-- scope: theoretical tightening of model-collapse bounds; even 1% synthetic contamination breaks scaling
     deps: [[model-collapse]]
     see-also: [[prismatic-synthesis]], [[rephrasing-the-web]]
-->

# Strong Model Collapse (Dohmatob et al., ICLR 2025 Spotlight)
- **Core Insight:** Within the neural-scaling-laws paradigm, even a **fixed small fraction of synthetic contamination (≈1%)** in the training pool eliminates the expected test-error reduction from larger data — scaling laws flatline; collapse is not limited to pure-recursive settings.
- **Guideline:** When deciding whether to include open web data contaminated with synthetic text, or when mixing modern synthetic corpora into pretraining, treat the synthetic fraction as a *scaling-law-breaker* unless you can filter or accumulate with a large real-data dominant term; "some synthetic is fine" is mathematically fragile.
- **Authors:** Elvis Dohmatob, Yunzhen Feng, Arjun Subramonian, Julia Kempe (Meta / NYU)
- **Year:** 2024/2025 (ICLR 2025 Spotlight)
- **URL:** https://arxiv.org/abs/2410.04840
- **Relevant topics:** model collapse, scaling laws, operator-valued free probability, synthetic contamination thresholds

## Abstract
Strong Model Collapse sharpens Shumailov-style collapse into the scaling-laws regime. Using tools from operator-valued free probability theory, the authors derive a new bias-variance decomposition of test error for a model trained on a mixture of real and synthetic data under random-projection approximation of neural networks. Key conclusion: **any non-vanishing synthetic fraction introduces an irreducible error term** that prevents test error from decreasing with data size, regardless of model size — scaling-law flattening. Empirically reproduced on GPT-2-scale LM training with 1% synthetic injection. A size-dependent secondary effect is discussed: below the interpolation threshold larger models amplify collapse; beyond it, larger models partially mitigate but never eliminate it.

## Key Contributions
- Analytical bias-variance decomposition in a tractable random-projection approximation of neural networks.
- Proves **scaling-law breakdown** with arbitrarily small synthetic fraction.
- Characterizes how model size modulates (not eliminates) collapse.
- Empirical reproduction on language-model scale training.
- ICLR 2025 Spotlight; load-bearing reference for the theoretical side of the 2025 synthetic-data-risk debate.

## Key Figures/Tables to Study
- **Test-error vs training-set-size curve** with and without 1% synthetic — scaling benefit vanishes.
- **Interpolation-threshold phase diagram** — region where larger models worsen vs mitigate collapse.
- **Bias-variance decomposition** — the extra term induced by synthetic contamination.

## Core equation (schematic)
Let `E[R_test]` be expected test risk; in the real-only case `E[R_test] ~ f(N)` decreases with `N` under standard scaling. Under synthetic fraction `p > 0`,
`E[R_test] ≈ f(N) + c(p) · σ_synth^2`
with `c(p) > 0` for any `p > 0`, making the asymptote a function of `p` rather than `N`. The paper's main theorem gives the explicit form of `c(p)` under random-projection assumptions.

## Experimental setup
- Theoretical: random-projection regression, operator-valued free probability.
- Empirical: GPT-2-scale LMs trained with 1% synthetic injection; measures test-loss scaling vs real-data size.
- Ablations over model size and synthetic fraction.

## Findings
- Scaling-law break at 1% contamination is not a tail effect; it's the new asymptote.
- Model size helps only once past interpolation threshold, and only partially.
- Aligned with independent 2025 work (Gerstgrasser, Zhu, He, Garg) that accumulation + filtering is the escape.

## Risks + gotchas
- **Random-projection approximation** is a simplification — real deep nets may show different phase structure; the empirical GPT-2 results support but do not prove the full theory.
- **1% threshold is regime-dependent** — for very small training sets or weak models, tolerance is higher.
- **Synthetic quality matters** — the theory assumes "synthetic" = iid from a model; high-quality filtered synthetic (e.g., via an external verifier) behaves like real in the limit.
- **Policy implication (not the paper's claim):** as open web accumulates LLM-generated text, *all* future pretraining corpora will be contaminated — hence active curation / provenance becomes mandatory.

## Connections
- Tightens [[model-collapse]]: from "recursive replacement breaks models" to "any contamination breaks scaling."
- Provides the theoretical basis for 2025-era "real-data anchor" / "verifier-in-the-loop" recipes (see [[nemotron-4-synthetic]] HelpSteer2 anchor, [[tulu-3-sft-mix]]).
- [[prismatic-synthesis]] is a practical counter: generate synthetic that covers under-filled gradient regions, effectively synthesizing real-like structure rather than teacher-mode-centered samples.
- Applies orthogonally to [[rephrasing-the-web]]: paraphrase-augmentation is not self-iterative, but the scaling-law warning transfers once synthetic mass dominates.
