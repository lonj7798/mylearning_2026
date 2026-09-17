---
chapter: ch-41
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/reward-model-overoptimization.md (arXiv:2210.10760v1)
source_url: https://arxiv.org/abs/2210.10760
created_at: "2026-04-23"
revised: "2026-09-15 (generality revision; rewritten to match the verified card and primary source)"
---

# Excerpt: scaling laws for reward-model over-optimization (Gao, Schulman, Hilton)

Used by [[read]] Core insight, §2, the Recipe, and the Generalization lens. Source card: [[reward-model-overoptimization]].

The earlier version of this excerpt stated that both α and β shrink with RM size, that a 10× larger RM halves the slope, that bigger policies exploit the proxy faster, and that a 3M RM peaks near d ≈ 3. None of these is in the paper. Corrections: α_bon increases and β_bon, β_RL decrease with RM size, α_RL is held constant (§3.2, Fig. 3); the 6B policy gains less but peaks at almost the same KL (§3.4); exact α and β values are only plotted.

## Setup (§2)
- InstructGPT environment; GPT-3-series policies SFT-trained for 2 epochs; gold RM = 6B RM from Ouyang et al.; proxy RMs 3M–3B; 100,000 synthetic comparisons with 10% held out; deterministic labels by higher gold score (§2.1).
- KL penalty 0 except in §3.6; policy 1.2B for RM-size sweeps (§2, §3.2).

## Functional forms (§1)
- `d := √(D_KL(π ‖ π_init))`; `R_bon(d) = d(α_bon − β_bon·d)`; `R_RL(d) = d(α_RL − β_RL·log d)`; `R(0) := 0`.
- `KL_bon = log n − (n−1)/n`, taken from Stiennon et al. App. G.3 (§2).
- The BoN form was fit on data up to n = 1,000 (KL ≈ 6 nats) and confirmed on a run up to n = 60,000 (KL ≈ 10 nats) (§3.1).
- Derived (not printed): `d*_bon = α_bon/(2β_bon)`, `d*_RL = exp(α_RL/β_RL − 1)`.

## Results
- §3.2: "αbon and βbon change smoothly with RM size"; "we can hold αRL constant across all RM sizes, resulting in a clean scaling curve for βRL". Fig. 3 plots α_bon rising and β_bon, β_RL falling with RM parameters.
- §3.3 (data sweep with RM size held at 12M): "for all RM sizes ... for amounts of data less than around 2,000 comparisons, there is very little improvement over near-chance loss" (Fig. 6). Footnote 7: controlling for SGD steps, "running 4 epochs instead of 1 yields no change in gold score whatsoever, whereas 1 epoch of 4 times as much data performs substantially better" (the two data amounts are not stated).
- §3.4: "Larger policies see less benefit from optimization against an RM, but don't overoptimize more"; both policy sizes peak at almost the same KL and have almost the same proxy–gold gap (Fig. 7, Fig. 24).
- §3.5: RL consumes far more KL than BoN.
- §3.6: a KL penalty behaved like early stopping and did not improve the gold-vs-KL frontier (Fig. 9); the authors call this hyperparameter-sensitive.
- §4.2: α interpreted as regressional Goodhart, β as extremal Goodhart; "the smooth decrease in β for both BoN and RL with increased RM size" read as improved robustness.
- §4.3: under simplifying assumptions, k iterations add `β_RL · d · log k` to the gold score.

## Limits
One environment; synthetic gold labels; adversarial Goodhart not captured; no satisfactory fit for proxy scores (§3.1, §4.2.4, §4.5).
