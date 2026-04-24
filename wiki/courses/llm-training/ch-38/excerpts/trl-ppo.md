---
chapter: ch-38
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/frameworks/trl-ppo.md
source_url: https://github.com/huggingface/trl
created_at: "2026-04-23"
---

# Excerpt: TRL PPO — the inline clipped surrogate, KL-in-reward

**Source library:** `wiki/raw-data/llm-training/frameworks/trl-ppo.md`
**Artifact:** `PPOTrainer.forward` inline update at lines 820–870; K2 approxkl diagnostic; value-loss clipping.

---

## Why this source anchors ch-38

TRL's `trl/experimental/ppo/ppo_trainer.py` is the cleanest single-file implementation of the InstructGPT PPO recipe in the modern ecosystem. Ch-38 §7 uses it as the reference for the `pg_loss = max(pg_losses1, pg_losses2)` pattern and as the canonical example of KL-in-reward via `non_score_reward`.

---

## The inline loss ch-38 §7 quotes

From the source (lines 820–870 excerpt):

> `logprobs_diff = new_logprobs - mb_logprobs`
> `ratio        = torch.exp(logprobs_diff)`
> `pg_losses    = -mb_advantage * ratio`
> `pg_losses2   = -mb_advantage * torch.clamp(ratio, 1.0 - args.cliprange, 1.0 + args.cliprange)`
> `pg_loss      = masked_mean(torch.max(pg_losses, pg_losses2), ~padding_mask[micro_batch_inds])`
> `loss = pg_loss + args.vf_coef * vf_loss`

The `torch.max` here is the sign-flipped equivalent of the paper's `torch.min` — because the losses are already negated (`-mb_advantage * ratio`), `max` on the negated quantities is `min` on the originals. This is the exact `L^{CLIP}` from ch-38 §2.

---

## Value clipping

From the source:

> `vpredclipped = torch.clamp(vpred, mb_values - args.cliprange_value, mb_values + args.cliprange_value)`
> `vf_losses1 = torch.square(vpred - mb_return)`
> `vf_losses2 = torch.square(vpredclipped - mb_return)`
> `vf_loss    = 0.5 * masked_mean(torch.max(vf_losses1, vf_losses2), ...)`

This is Costa-Huang detail #4 from ch-38 §6: clip value predictions around the old value to prevent the value head from overfitting a single rollout across K epochs.

---

## KL-in-reward — not in loss

From the source §What to notice:

> **No KL inside the loss** — KL penalty is added into the *reward* as `non_score_reward` before GAE, so it becomes part of `mb_advantage`.

This is Costa-Huang detail #6 (ch-38 §6) and the canonical instantiation of [[kl-control-rlhf]]'s "add KL to reward, not loss" rule.

---

## Two KL metrics

From the source:

> **Two KL metrics:** `kl = logprobs − ref_logprobs` (K1, biased but cheap) — used in reward shaping. `approxkl = 0.5·(Δlogp)^2` (K2, Schulman) — used in the clip-fraction diagnostic.

Ch-38 §6 notes that K3 `(π_ref/π) − 1 − log(π_ref/π)` is the *modern recommendation* (unbiased and ≥0); TRL still uses K1 in reward shaping and K2 in diagnostics. Not a bug — just a snapshot of the ecosystem.

---

## Entropy is a metric, not a loss term

From the source:

> **Entropy computed from logits** via `logsumexp − Σ p·logp` inside `torch.no_grad()` — it's a metric, not a training loss term (no entropy bonus).

Matches ch-38 §4's "no entropy bonus in InstructGPT PPO" — the β·KL term to π_ref is the regularizer.

---

## What ch-38 keeps, changes, drops

Keeps: `max` of negated losses pattern, value clipping, `non_score_reward` for KL-in-reward, K2 approxkl diagnostic. Changes: ch-38 recommends K3 over TRL's K1 for reward shaping. Drops: `INVALID_LOGPROB` sentinel handling (implementation detail).
