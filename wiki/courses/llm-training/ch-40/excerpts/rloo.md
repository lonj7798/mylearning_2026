---
chapter: ch-40
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/rloo.md
source_url: https://arxiv.org/abs/2402.14740
created_at: "2026-04-23"
---

# Excerpt: RLOO — the paper that killed PPO's critic

**Source library:** `wiki/raw-data/llm-training/papers/rloo.md`
**Artifact:** Ahmadian et al. 2024, "Back to Basics: Revisiting REINFORCE-Style Optimization for Learning from Human Feedback in LLMs."

---

## Why this source anchors ch-40

RLOO is the load-bearing paper for the entire §2–§8 narrative. It is the one that *stated the problem* — PPO's components assume stochastic dynamics and multi-step credit assignment that do not hold in LLM RLHF — and proposed the minimum-viable fix: k-sample REINFORCE with a leave-one-out baseline. Every subsequent paper in the chapter (REINFORCE++, GRPO, Dr.GRPO) either tweaks the baseline (global / group / no-std) or keeps everything else RLOO-identical.

---

## The derivation ch-40 §2 quotes line-by-line

Source lines 34–37 give the estimator; ch-40 reconstructs the derivation:

1. Start from REINFORCE with a constant baseline: `∇J = E[(R − b) · ∇log π]`. Any baseline `b` independent of `y` leaves the gradient unbiased — standard policy-gradient result.
2. Make `b` depend on the *other* samples to share information: `b_i = (1/(k−1)) Σ_{j≠i} R_j`. Since `b_i` is a deterministic function of `{R_j : j ≠ i}`, and those depend on `{y_j : j ≠ i}` which are independent of `y_i` given `x`, `b_i ⊥ y_i | x` — still unbiased.
3. Variance reduction: `b_i` is the minimum-variance unbiased estimator of `E[R]` using the other k-1 samples. Tighter than any moving-average baseline.

---

## The k=2 limit ch-40 highlights

With k=2, `b_1 = R_2` and `b_2 = R_1`. The advantage for sample 1 is `R_1 − R_2`; for sample 2 is `R_2 − R_1`. They are mirror images. This is why RLOO at k=2 is "online DPO without the log-sigmoid" — the same pairwise preference signal, taken as a raw policy-gradient step rather than binarized.

---

## What ch-40 §2's table came from

Source lines 45–51 enumerate what RLOO removes relative to PPO. Ch-40 reproduces the table verbatim because it is the single cleanest statement in the literature of "what was PPO overhead for LLMs." Value network gone, GAE gone, clip gone, epochs=1. Only the leave-one-out baseline and the per-token KL-shaped reward remain.

---

## Attested hyperparameters ch-40 uses

Source lines 54–62:

| Knob | Value (attested) |
|------|------------------|
| k (rollouts per prompt) | 2 or 4 (main: k=4) |
| KL coef β | 0.05 (tuned on Pareto curve) |
| Learning rate | 1e-6 to 3e-6 (AdamW) |
| Batch size (prompts) | 32–64 |
| Sampling T | 1.0 |
| Max new tokens | 53 (TL;DR), 256 (HH) |

Ch-40 reports these as RLOO's defaults. Note the batch size is *much smaller* than GRPO's 1024 prompts — RLOO is a small-batch method; GRPO scaled it up.

---

## Empirical dominance over PPO (§1 of ch-40 references this)

Source lines 20–25 and Figure 3: RLOO dominates PPO's Pareto frontier at every KL budget on TL;DR and HH-RLHF. The win rate gap is 5–20% at matched KL. This is the evidence ch-40 §1 uses to claim "PPO's overhead is a tax, not a feature, for LLM RLHF."

---

## The relationship-to-GRPO paragraph

Source lines 64–65 explicitly states: "GRPO's advantage `(r_i − mean(r))/std(r)` over a group of G is equivalent (up to scaling) to RLOO's leave-one-out when G is large." Ch-40 §6 turns this into the equivalence-in-the-limit table: RLOO (large k) ≈ GRPO without /std without clip ≈ Dr.GRPO. This paragraph is the theoretical bridge between the two lineages.

---

## Connections to the rest of the track

- [[grpo]], [[dr-grpo]] — the successors that inherited the leave-one-out idea.
- [[reinforce-plus-plus]] — same family, global normalization, k=1 variant.
- [[ppo]] — the baseline this paper systematically strips.
- [[on-off-policy-rlhf]] — why the field moved back to online methods, which made RLOO possible.
