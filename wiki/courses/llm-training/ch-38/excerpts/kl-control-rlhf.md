---
chapter: ch-38
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/kl-control-rlhf.md
source_url: https://arxiv.org/abs/2205.11275
created_at: "2026-04-23"
---

# Excerpt: KL-control in RLHF — why β goes in the reward

**Source library:** `wiki/raw-data/llm-training/papers/kl-control-rlhf.md`
**Artifact:** Korbak Bayesian reformulation; k1/k2/k3 KL estimators; per-token KL reward shaping.

---

## Why this source anchors ch-38

Ch-38's thesis — RLHF is KL-regularized inference, not reward maximization — comes directly from [[kl-control-rlhf]]. The Korbak 2022 reformulation shows why β has a natural temperature interpretation and why the training objective has a closed-form tilted-posterior optimum. Ch-38 §4 states this explicitly as the justification for treating β as a *temperature*, not a loss coefficient.

---

## The core identity ch-38 §4 cites

From the source:

> `argmax_π E_π[r] − β · KL(π‖π_ref)` has closed-form optimum `π*(y|x) ∝ π_ref(y|x) · exp(r(x,y)/β)`. RLHF is therefore amortized sampling from this tilted posterior — not unconstrained reward maximization.

Ch-38 §4 uses this to frame PPO's job as "amortized sampling from a tilted posterior," which then makes DPO's closed-form implicit reward (ch-39) the same target approached differently.

---

## Why KL goes in the reward

From the source:

> **Why reward + not loss:** adding KL to the reward keeps the PPO advantage estimator well-defined per token; adding KL to the loss breaks the advantage-based policy gradient and empirically trains worse.

Ch-38 §6 makes this Costa-Huang detail #6. [[trl-ppo]]'s `non_score_reward` and [[verl-ppo-loss]]'s external `kl_penalty` both implement this.

---

## The k3 estimator

From the source:

> `k3 = (π_ref/π) − 1 − log(π_ref/π)` — unbiased AND always ≥ 0; **recommended**, used in modern TRL/OpenRLHF.

Ch-38 defaults to k3 for the per-token KL reward. k1 = `log(π/π_ref)` is cheap but high variance and can go negative; k2 = `0.5·(log π/π_ref)^2` is biased.

---

## β range ch-38 uses

From the source:

> **β coefficient:** InstructGPT used β ≈ 0.02 (in reward-scale units); practitioners tune in [0.01, 0.5].

Ch-38 §4 carries β=0.02 (InstructGPT), §5 carries β=0.01 (Llama-2). The figure `figures/ppo-clip.html` panel 2 sweeps this range and annotates the two regimes (reward hacking below ~0.005, mode collapse above ~1.0).

---

## What ch-38 keeps, changes, drops

Keeps: Korbak identity, k3 estimator, KL-in-reward rule, β=0.02/0.01. Changes: nothing material. Drops: DPO's implicit-KL discussion (moved to ch-39).
