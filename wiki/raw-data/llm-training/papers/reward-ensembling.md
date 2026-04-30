<!-- scope: ensembles of reward models as a defense against overoptimization
     deps: [[reward-model-overoptimization]]
     see-also: [[generative-reward-models]], [[reward-hacking-taxonomy]]
-->

# Reward Model Ensembles Help Mitigate Overoptimization (Coste et al.)
- **Core Insight:** Averaging (or taking the lower confidence bound over) multiple independently-trained reward models pushes the proxy-vs-gold divergence point further into the KL budget — ensembles partly immunize RLHF against Goodhart.
- **Guideline:** Train K ≥ 3 RMs with different seeds/data shards; use their mean for the reward signal in PPO and their min (or a pessimistic quantile) when selecting with Best-of-N; uncertainty disagreement can also be added as a negative shaping term to penalize extremal regions.
- **Authors:** Thomas Coste, Usman Anwar, Robert Kirk, David Krueger
- **Year:** 2023
- **URL:** https://arxiv.org/abs/2310.02743
- **Relevant topics:** RM ensembles, RM uncertainty, conservative reward, overoptimization, best-of-N

## Abstract
The paper replicates Gao et al.'s synthetic overoptimization benchmark with an ensemble defense. They train K RMs (K = 3, 5, 10) with different seeds on the same preference data, then combine them via mean, lower-confidence-bound (mean − λ·std), or min. On both Best-of-N and PPO, ensembling shifts the proxy-vs-gold peak to a higher KL and lifts the achievable gold reward — the overoptimization slope gets shallower. Conservative combinations (LCB, min) are uniformly safer but slightly lower in peak performance than mean.

## Key Contributions
- **Ensemble combinations studied:**
  - *Mean:* `r(x,y) = (1/K) Σ_k r_k(x,y)`.
  - *LCB:* `r(x,y) = mean_k r_k − λ · std_k r_k` (pessimistic under disagreement).
  - *Min:* `r(x,y) = min_k r_k(x,y)`.
  - *UWO (uncertainty-weighted objective):* reward minus a penalty on std.
- **Result:** all ensemble strategies delay overoptimization; LCB / min trade peak gold reward for robustness.
- **KL at peak shifts:** peak moves from `d ≈ 3` (Gao baseline) to `d ≈ 5–8` depending on K.
- **Limits:** ensembles correlate when they share data; if all RMs are systematically miscalibrated in the same direction (shared label noise, shared blind spot), ensembling does not help — demonstrated on adversarial prompts.
- **Practical recipe:** 3–5 RMs is usually enough; diminishing returns past that; seed diversity matters less than data-shard diversity.

## Key Figures/Tables to Study
- **Fig. 1** (proxy vs gold vs KL for single RM vs ensemble) — the peak shift.
- **Fig. 3** (mean vs LCB vs min) — LCB is the best compromise.
- **Fig. 5** (shared-blind-spot counterexample) — ensemble fails when biases are correlated.

## Technical Details
- **Loss unchanged:** each RM trained with standard BT loss (see **[[bradley-terry-rm]]**).
- **RL objective:** `r_ensemble(x,y) − β · KL(π‖π_ref)` fed into PPO; β same range as standard RLHF.
- **Best-of-N selection:** rerank N samples by ensemble score; min aggregator is robust to confidently-wrong RMs.
- **Disagreement as signal:** `std_k r_k` correlates with OOD-ness of the response; can be used as an anomaly flag.
- **Overhead:** K RMs roughly multiply the reward-forward-pass cost by K during RL; often affordable since the RM is smaller than the policy.

## Connections
- Direct defense against the laws of **[[reward-model-overoptimization]]**.
- Complementary to generative / LLM-as-judge RMs (**[[generative-reward-models]]**) — ensembling + CoT judging can compound.
- Philosophically aligned with the positive results in **[[reward-hacking-taxonomy]]** about restricting policy classes: the LCB aggregator implicitly penalizes policies pushing RMs apart.
- Used in production pipelines for safety-sensitive RLHF.
