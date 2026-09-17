---
chapter: ch-42
course: llm-training
phase: read
excerpt_of: arXiv:2209.13085v2 (Defining and Characterizing Reward Hacking), NeurIPS 2022; library card [[reward-hacking-taxonomy]]
source_url: https://arxiv.org/abs/2209.13085
created_at: "2026-04-23"
revised: "2026-09-15 (generality revision; rewritten from the primary source because the library card has no Verification section)"
---

# Excerpt: hackability, unhackability, and simplification (Skalse, Howe, Krasheninnikov, Krueger)

Used by [[read]] §1. Checked against arXiv v2 (2025-03-05) on 2026-09-15.

## Definitions (§4.2)
> **Definition 1.** A pair of reward functions `R1, R2` are **hackable** relative to policy set Π and an environment if there exist `π, π' ∈ Π` such that `J1(π) < J1(π')` and `J2(π) > J2(π')`; else they are **unhackable**.

`J_i(π)` is the expected discounted return of `π` under `R_i`. Unhackability is symmetric and is not transitive. `R1` and `R2` are **equivalent** on Π if `J1` and `J2` induce the same ordering of Π; `R` is **trivial** on Π if all policies have the same return.

> **Definition 2.** `R2` is a **simplification** of `R1` relative to Π if `J1(π) < J1(π') ⇒ J2(π) ≤ J2(π')` and `J1(π) = J1(π') ⇒ J2(π) = J2(π')`, and there exist `π, π'` with `J2(π) = J2(π')` but `J1(π) ≠ J1(π')`.

## Results (§5.1, §5.2)
> **Theorem 1.** In any MDP\R, if Π̂ contains an open set, then any pair of reward functions that are unhackable and non-trivial on Π̂ are equivalent on Π̂.

- **Corollary 1**: this applies to the set of all stationary policies.
- **Corollary 2**: it also applies to the set of all ε-suboptimal policies (ε > 0) and to the set of all δ-deterministic policies (δ < 1), because both sets contain open subsets. "Intuitively, Theorem 1 can be applied to any policy set with 'volume' in policy space."
- **Theorem 2**: for any finite policy set containing two policies with different feature counts, and any `R1`, there exists a non-trivial `R2` that is unhackable with respect to `R1` but not equivalent — so non-trivial unhackability exists on finite sets, including the set of deterministic policies.

## Simplification does not rescue a proxy (§1, Abstract)
> "Intuitively, it might be possible to create an unhackable proxy by leaving some terms out of the reward function (making it 'narrower') or overlooking fine-grained distinctions between roughly equivalent outcomes, but we show this is usually not the case."

The paper's worked examples: narrowing a cleaning reward `[1,1,1]` to "only clean the attic" `[1,0,0]` is hackable against the true reward, and coarsening fine distinctions stops being a simplification as soon as the true reward is slightly unbalanced (§1, Fig. 1).

## What the paper does not claim
- It does not show that a KL-bounded set of policies is unhackable. Corollary 2 goes the other way for sets with volume in policy space.
- It gives no empirical measurement; the empirical proxy-versus-gold curves come from [[reward-model-overoptimization]].
