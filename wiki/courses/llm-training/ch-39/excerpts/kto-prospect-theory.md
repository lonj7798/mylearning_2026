---
chapter: ch-39
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/kto.md
source_url: https://arxiv.org/abs/2402.01306
created_at: "2026-04-23"
---

# Excerpt: KTO — prospect theory for unary preference labels

**Source library:** `wiki/raw-data/llm-training/papers/kto.md`
**Authors:** Kawin Ethayarajh, Winnie Xu, Niklas Muennighoff, Dan Jurafsky, Douwe Kiela
**Venue:** arXiv 2402.01306
**Year:** 2024

---

## Why this source anchors ch-39 §4

Every other variant in the chapter takes paired preferences. KTO is the single variant that works from *unary* labels: `{(x, y, b)}` with `b ∈ {desirable, undesirable}`. That matters because in production, feedback is usually one thumb, not a ranked pair: users mark a chatbot response up or down; moderators flag a completion as safe or unsafe; a verifier says the answer is correct or incorrect. You get a stream of unary labels; KTO is how you turn that stream into a preference-optimization gradient.

The derivation is not "drop the rejected side of DPO"; that would not work, because DPO's loss is symmetric and needs both halves of the pair. KTO uses the Kahneman-Tversky **prospect-theory value function** — concave on gains, convex and *steeper* on losses — and a batch-level reference point `z_0` as the "status quo" against which each example is scored.

## The HALO framing — DPO and PPO are already prospect-theoretic

Source lines 18–19: "We propose a family of human-aware loss functions (HALOs) and use it to derive KTO... DPO, PPO-Clip are HALOs." This is the paper's clever framing: prospect theory is not a new ingredient KTO invented; it is an ingredient DPO already had implicitly. KTO is what you get when you *derive* the loss directly from the KT utility instead of via Bradley-Terry.

## The reference point z_0

Source line 34–35:

```
z_0 = KL( π_θ(y' | x) || π_ref(y' | x) )
```

Computed over a batch-level moving statistic, no gradient through it. This is the prospect-theory "status quo": each example is scored relative to where the policy currently sits. In practice it's a detached estimate that stabilizes training without adding a real KL penalty.

## The value function

Source lines 37–40:

```
v(x, y) = { λ_D · σ( β · ( r_θ(x,y)  − z_0 ) )       if y is desirable
          { λ_U · σ( β · ( z_0 − r_θ(x,y) ) )        if y is undesirable
```

Where `r_θ(x,y) = β · log[π_θ(y|x) / π_ref(y|x)]` is the same implicit reward as DPO.

The asymmetry between desirable and undesirable is encoded via `λ_D` and `λ_U`. Prospect theory says humans are loss-averse — a $100 loss hurts ~2× more than a $100 gain feels good. KTO lets you set `λ_U > λ_D` to reflect that, but in the paper's default recipe both are 1.0.

## The loss

Source line 43:

```
L_KTO(π_θ; π_ref) = E_{(x, y) ~ D}[ λ_y − v(x, y) ]
```

where `λ_y ∈ {λ_D, λ_U}` is chosen by the label of the example.

## The class-balance recipe — the single most useful hyperparameter rule

Source line 58: **`λ_D / λ_U = N_U / N_D`**.

That is: if your desirable data is 10× as common as your undesirable, set `λ_U` 10× larger than `λ_D`. This keeps the two sides contributing equal expected loss to the gradient, which in turn prevents the policy from just memorizing "default to desirable."

In practice this is the KTO move that lets you use badly-imbalanced real-world feedback logs (90 % thumbs-up, 10 % thumbs-down) without the gradient degenerating.

## Hyperparameters

| Knob | Value |
|------|-------|
| β | 0.1 (same as DPO) |
| λ_D | 1.0 |
| λ_U | tune to `λ_D · N_D ≈ λ_U · N_U` |
| LR | 5e-7 |
| Batch | 32 examples (not pairs — unary) |
| Epochs | 1 |
| π_ref | SFT, frozen |

## How ch-39 uses this

§4 of `read.md` presents the value function and loss as Equations (10)–(11). The comparison table in §8 pins KTO as the single variant for unary data. The loss landscape in `figures/dpo-landscape.html` plots the desirable-vs-undesirable branches as mirrored curves (solid and dashed green) to make the asymmetry visible.

## Connections

- Logit-link ancestor: [[dpo]].
- Squared-link cousin: [[ipo]].
- Length-normalized reference-free variant: [[simpo]].
- Joint SFT + preference: [[orpo]].
- Framework flag: TRL `loss_type="kto_pair"` — see [[hf-dpo-zoo]].
