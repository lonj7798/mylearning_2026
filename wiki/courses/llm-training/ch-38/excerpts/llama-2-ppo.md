---
chapter: ch-38
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/model-reports/llama-2.md
source_url: https://arxiv.org/abs/2307.09288
created_at: "2026-04-23"
---

# Excerpt: Llama-2 PPO — dual RM, LR 1e-6, KL 0.01

**Source library:** `wiki/raw-data/llm-training/model-reports/llama-2.md`
**Artifact:** V4/V5 PPO appendix hyperparameters, dual-RM scoring rule, margin-label RM loss.

---

## Why this source anchors ch-38

Ch-38 §5 is the Llama-2 instantiation of Equation 2. Llama-2 is the most thoroughly documented open RLHF recipe that used PPO (its successor Llama-3 switched to DPO) and the departures from InstructGPT — two RMs instead of one, LR 14× smaller, β half as large — all point at the iterative-RLHF schedule (V1–V5).

---

## PPO appendix quoted verbatim

From the source:

> **V4, V5:** PPO added on top of RSFT checkpoint.
> - **Learning rate:** 1e-6 (policy) for 70B.
> - **KL coefficient beta:** 0.01.
> - **Batch size:** 512.
> - **Sequence length:** 4K.
> - Standard PPO with clipped ratio, value function, GAE.

Ch-38 §5 puts this table in directly and contrasts against InstructGPT (LR 1.41e-5, β=0.02). The factor of ~14 on LR is the single load-bearing delta.

---

## Why the conservative LR — ch-38 §5's reasoning

Grounded in the iterative-RLHF schedule:

> **Iterative RLHF:** five successive checkpoints (V1..V5) with fresh weekly batches of human preferences.

Ch-38 §5 reasons: at 70B scale with weekly fresh batches, large per-iteration drift compounds across V4 → V5 and the dual-RM rule destabilizes. Smaller LR + smaller β + more iterations is the stabler point.

---

## Dual reward model — two models, one objective

From the source:

> Dual reward model: Helpfulness RM + Safety RM, both initialized from the LM base with a linear regression head replacing the LM head.
> At RLHF scoring time, a rule selects which RM (or a weighted combo) scores each prompt.

Ch-38 §5 frames this as "resolves the helpfulness-vs-safety tradeoff that a single RM forces into its scalar output." The piecewise rule (safety dominates on safety prompts) is what you get when you don't want to average helpfulness and safety into one score.

---

## Margin-weighted RM loss

From the source:

> **Margin labels:** "significantly better / better / slightly better / negligibly better" — used as a margin term in the RM loss.

Ch-38 §5 writes this as `L_RM = −E[log σ(r_w − r_l − m(label))]`. Upweights large-margin pairs, which shapes the reward surface PPO optimizes against.

---

## RSFT-before-PPO

From the source:

> **V1..V3:** Rejection-Sampling Fine-Tuning (RSFT). For each prompt, sample K outputs (K ~ 10+), score with combined RMs, SFT on the best sample. No policy-gradient.
> **V4, V5:** PPO added on top of RSFT.

Ch-38 §5 mentions this but keeps it brief — RSFT has its own chapter. The RLHF-track takeaway is that PPO is applied on top of an already RSFT-refined checkpoint, which makes the LR 1e-6 feasible.

---

## What ch-38 keeps, changes, drops

Keeps: LR 1e-6, β=0.01, batch 512, seq 4K, dual-RM framing. Drops: context distillation (safety chapter), margin-label preference protocol (preference-data chapter), detailed RM training (RM chapter).
