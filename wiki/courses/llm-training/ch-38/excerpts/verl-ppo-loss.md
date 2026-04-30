---
chapter: ch-38
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/frameworks/verl-ppo-loss.md
source_url: https://github.com/verl-project/verl
created_at: "2026-04-23"
---

# Excerpt: verl PPO vanilla — asymmetric clip + dual-clip + loss-agg

**Source library:** `wiki/raw-data/llm-training/frameworks/verl-ppo-loss.md`
**Artifact:** `compute_policy_loss_vanilla` at `verl/trainer/ppo/core_algos.py` lines 1080–1140; dual-clip Ye 2020; loss aggregation modes.

---

## Why this source anchors ch-38

Ch-38 §6 (Costa-Huang tricks) and §7 (framework picture) both lean on verl because verl exposes the knobs that the Schulman 2017 paper hides: asymmetric `clip_ratio_low/high`, dual-clip bound on negative-advantage tokens, plug-in loss aggregation. These are the modern deltas that make PPO survive long LLM rollouts.

---

## The vanilla PPO loss ch-38 §7 quotes

From the source (lines 1080–1140 excerpt):

> `negative_approx_kl = log_prob - old_log_prob`
> `negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)`
> `ratio = torch.exp(negative_approx_kl)`
> `pg_losses1 = -advantages * ratio`
> `pg_losses2 = -advantages * torch.clamp(ratio, 1 - clip_ratio_low, 1 + clip_ratio_high)`
> `clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)`

Same `max`-of-negated-losses pattern as [[trl-ppo]]. The `clamp` on `negative_approx_kl` to `[-20, 20]` prevents `exp` overflow when a rare token has a huge log-ratio — this is a numerical stability trick TRL doesn't use because its batch sizes are smaller.

---

## Asymmetric clipping (DAPO recipe)

From the source §What to notice:

> **Asymmetric clipping:** `clip_ratio_low` and `clip_ratio_high` separate; DAPO uses ε_low ≈ 0.2 and ε_high ≈ 0.28 to allow more upward exploration on rare positive-advantage tokens.

Ch-38 §6 Costa-Huang detail #3 elevates this to "first thing to try when entropy collapses without reward gain." Symmetric ε=0.2 is the default; asymmetric is the modern patch.

---

## Dual-clip (Ye 2020)

From the source:

> **Dual clip** only fires when `advantages < 0` and the loss has already exceeded `clip_ratio_c · |advantage|`; it floors the loss to prevent ratio blow-ups on long off-policy rollouts.

Ch-38 §2 mentions this as the patch to PPO-clip's asymmetry: vanilla PPO does *not* floor the loss for negative-advantage tokens with huge ratios, and on long LLM rollouts (>2K tokens, K=4 epochs) that can blow up. Dual-clip floors the loss at `-clip_ratio_c · |adv|` (default `c=3.0`).

---

## Loss aggregation — the modern knob

From the source:

> **Loss aggregation is parametric:** `"token-mean"` (default), `"seq-mean-token-sum"` (Dr.GRPO style), or `"seq-mean-token-mean"` (length-normalized) — each materially changes gradients on long-tailed completion length distributions.

Ch-38 §6 Costa-Huang detail #4 (length normalization of the policy loss) is built directly on this. Choosing `token-mean` vs `seq-mean-token-sum` shifts gradient between short and long completions by a factor of `L_max / L_mean` on typical RLHF distributions.

---

## K3 KL, not in the loss

From the source:

> **K1 ratio is *not* the KL term in the loss** — `ppo_kl = mean(-Δlogp)` is logged for monitoring only; KL-to-reference is added either through the reward (token-level KL) or through a separate `kl_penalty(logprob, ref_logprob, "k3")` call (low-variance Schulman estimator), never inside this function.

Ch-38 §6 Costa-Huang detail #6 elevates the k3 estimator and the KL-in-reward rule — verl is the modern reference implementation.

---

## What ch-38 keeps, changes, drops

Keeps: `max`-of-negated-losses pattern, asymmetric clip, dual-clip, loss aggregation modes, k3 external KL. Changes: ch-38 recommends `token-mean` as the safe default; Dr.GRPO's `seq-mean-token-sum` is a later-chapter topic. Drops: `rollout_is_weights` vLLM IS correction (covered under engine-mismatch chapter).
