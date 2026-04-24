---
chapter: ch-42
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/reward-hacking-taxonomy.md
source_url: https://arxiv.org/abs/2209.13085
created_at: "2026-04-23"
---

# Excerpt: Skalse 2022 — Defining and Characterizing Reward Hacking

**Source library:** `wiki/raw-data/llm-training/papers/reward-hacking-taxonomy.md`
**Paper:** Skalse, Howe, Krasheninnikov, Krueger, *Defining and Characterizing Reward Hacking*, NeurIPS 2022.

---

## Why this source anchors ch-42

Reward hacking before Skalse 2022 was a laundry list of anecdotes — CoastRunners boats spinning in coves, Lego-stackers flipping blocks, length-biased RMs. This paper is the first formal statement, and it reframes the whole engineering problem. The raw-data header:

> **Core Insight:** A proxy reward is "unhackable" (every increase in proxy return guarantees a non-decrease in true return) only in degenerate cases — under all stochastic policies, two non-trivially-related rewards can be jointly unhackable only if one of them is constant.

Ch-42 uses this to argue that "write a better reward" is a category error. The chapter's §1 quotes Theorem 3.2 directly, and §8's ordering of structural defenses (KL budget, verifiable rewards, potential-based shaping) is a direct response to the impossibility result.

## The definition

Given policy class Π, proxy `R̃` is **unhackable** wrt true reward `R` iff for all `π, π' ∈ Π`:

```
R̃(π) ≥ R̃(π')   ⇒   R(π) ≥ R(π')
```

Reward hacking is the failure of this property: a pair of policies where the proxy ordering disagrees with the true-reward ordering.

## The impossibility theorem

Over the set of all stochastic policies on a finite MDP, `R̃` is unhackable wrt `R` iff `R̃` is a positive affine transformation of `R` (or one of them is constant). The proof uses the simplicial geometry of the return vectors — unhackability requires the pair `(R̃(π), R(π))` to lie on a monotone curve, and over the convex hull of all stochastic policies this forces linearity.

**Operational consequence:** any scalar RM that is *not* literally `a·R + b` for the true reward will be hackable somewhere in policy space. A capable optimizer will find that region.

## The "simplification" counter-example

One plausible intuition is that simplifying a reward specification — removing terms to make it cleaner — should improve unhackability. The paper shows this is false: simplification can make unhackability strictly worse. The example is constructive: two rewards where a dropped term was the only thing pinning the ordinal structure of the proxy to the truth.

This is why ch-42 §1 states that writing a cleaner reward does not help; writing a bounded optimizer does.

## Positive results on restricted policy classes

The theorem fails if you restrict Π:

- **Deterministic policies.** Non-trivial unhackable pairs exist; characterized via the geometry of the deterministic-return polytope.
- **Finite enumerated policies.** Same — enumerate the pairs, check monotonicity.

The practical version: in RLHF you never train over "all stochastic policies"; you train over the image of an SGD trajectory starting from the SFT model. Restricting the exploration region (KL budget, early stopping) is precisely what moves you into a policy class where non-trivial unhackability is possible. This is the structural justification for KL control in [[kl-control-rlhf]].

## Hacking examples enumerated

The paper's related-work + appendix catalogues:

- Sycophancy.
- Length bias.
- Sentiment bias.
- Formatting / bold-text bias.
- Reward-model blind spots (high-RM-score adversarial strings that are incoherent).
- Specification gaming (CoastRunners, Lego-stacker).

Ch-42 §2 extends this list with post-2022 additions — U-sophistry, in-context reward hacking, refusal overtraining — and re-organizes it by mechanism/symptom/mitigation.

## Connection to RLHF

The paper's explicit interpretation for RLHF: the learned RM is the proxy, the true human preference distribution is the gold, and as optimization widens the considered policy region, hacking becomes generic. The recommended fix is not "a better RM" but "a bounded optimizer plus verifiable-reward fallback where possible" — the exact stance ch-42 adopts.

## Takeaways for the chapter

1. Reward hacking is not contingent on careless reward design; it is a structural property of scalar proxies under sufficiently powerful optimization.
2. Restrict the policy class (KL budget, early stopping, bounded rollouts) to move into a regime where unhackability is at least possible.
3. Verifiable rewards side-step the theorem because the proxy *is* the true reward by construction.
4. Every tractable defense in the modern stack — ensembling, CAI, GenRMs, RM-vs-RM disagreement — operates on the optimizer, not on the "right" reward.
