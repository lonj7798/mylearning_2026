---
chapter: ch-40
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/frameworks/trl-grpo.md
source_url: https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py
created_at: "2026-04-23"
---

# Excerpt: TRL GRPOTrainer — where the GRPO zoo lives in one file

**Source library:** `wiki/raw-data/llm-training/frameworks/trl-grpo.md`
**Artifact:** `trl/trainer/grpo_trainer.py` ≈ lines 2418–2610. The `_compute_loss` method is the unified entry point for GRPO, Dr.GRPO, DAPO, CISPO, BNPO, LUSPO, VESPO, SAPO — all selected by a `loss_type` string.

---

## Why this source anchors ch-40 §7

Ch-40 §7 needs to show the reader that the abstract equations in §4–§5 have concrete code realizations, and that the Dr.GRPO "fix" is a one-line change in the aggregator. TRL's trainer is the best vehicle because it keeps every variant in a single method — the diffs between GRPO and Dr.GRPO are inline.

---

## The aggregation branch ch-40 §5 and §7 both quote

Source lines 86–93:

```python
# ---- aggregation (varies per loss_type) ----
if self.loss_type == "grpo":
    loss = ((per_token_loss * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)).mean()
elif self.loss_type == "bnpo":
    loss = (per_token_loss * mask).sum() / mask.sum().clamp(min=1.0)
elif self.loss_type == "dr_grpo":
    loss = (per_token_loss * mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
```

Three lines. The **only** difference between `grpo` and `dr_grpo` is the denominator:
- `grpo`: per-sequence token-mean, then sample mean. `(Σ_t loss_t / |o_i|).mean()` — each response contributes its per-token mean, averaged across responses. Biased: short correct wins, long wrong underpenalized (ch-40 §5).
- `dr_grpo`: per-batch token-sum divided by `(B · L_max)`. Each token's loss contributes the same weight regardless of its host sequence's length. Unbiased.

The `bnpo` variant is the batch-normalized form (token-sum / total-valid-tokens) — an intermediate.

---

## The k3 KL inline

Source lines 58–63:

```python
if self.beta != 0.0:
    ref_per_token_logps = inputs["ref_per_token_logps"]
    per_token_kl = (
        torch.exp(ref_per_token_logps - per_token_logps)
        - (ref_per_token_logps - per_token_logps) - 1
    )
```

This is the Schulman k3 estimator (ch-40 §4 derivation): `e^x − x − 1` where `x = log(π_ref/π_θ)`. Always ≥ 0, unbiased, one extra reference forward pass. Ch-40 §4 reads the math directly off this tensor operation.

---

## The surrogate branch

Source lines 67–77:

```python
if self.loss_type in ["grpo", "bnpo", "dr_grpo", "dapo", "luspo"]:
    coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
    if self.args.delta is not None:                           # DAPO upper-clip cap
        coef_1 = torch.clamp(coef_1, max=self.args.delta)
    per_token_loss1 = coef_1 * advantages
    per_token_loss2 = coef_2 * advantages
    per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
```

Standard PPO-clip with asymmetric `epsilon_low` / `epsilon_high` (the DAPO generalization). The `-torch.min` is the PPO surrogate: always take the *less optimistic* of the clipped and unclipped objectives. Unchanged across GRPO/Dr.GRPO.

---

## Advantage broadcast — where length bias enters (ch-40 §5 references this)

Source lines 40–42:

```python
advantages = inputs["advantages"]               # (B,)  group-relative z-scores
if advantages.dim() == 1:
    advantages = advantages.unsqueeze(1)
```

The `(B,)` per-rollout scalar advantage becomes `(B, 1)` and broadcasts against the `(B, T)` per-token-loss tensor. Every token in a completion shares the same advantage. This is where the length bias enters: combined with the `(1/|o_i|)` per-sequence mean downstream, the aggregated per-sequence loss is independent of the sequence length, so per-token gradient magnitude is `|A|/|o_i|`.

---

## Top-entropy masking (DAPO trick, not GRPO-core but coexists)

Source lines 34–38 and 79–80:

```python
if self.top_entropy_quantile < 1.0:
    entropy_mask = self.get_high_entropy_mask(entropies, mask, 1 - self.top_entropy_quantile)
# ...
if entropy_mask is not None:
    per_token_loss = per_token_loss * entropy_mask
```

Keeps only the top-K% highest-entropy tokens for the gradient — the DAPO / Muon-paper observation that gradient signal on low-entropy tokens is mostly noise. Not in vanilla GRPO; ch-40 §7 notes it exists as an option.

---

## Why TRL fuses and verl splits

Ch-40 §7's claim: TRL bundles advantage normalization, KL term, clipped objective, aggregator all into `_compute_loss`; verl splits them into registry-pluggable `compute_advantage` and `compute_policy_loss` hooks (see [[verl-grpo]]). Algebraic equivalence for `loss_type="grpo"`; the split in verl makes it easier to add new advantage estimators.

---

## Attested implementation notes

- TRL always logs `masked_batch_mean(entropies)` to `_metrics[mode]["entropy"]` — ch-40 §5's "log `mean(|o_wrong|) − mean(|o_right|)` every epoch" guideline depends on this kind of instrumentation.
- K3 KL is always computed when β ≠ 0. Setting β=0 disables KL entirely (pure reward-only RL) — used in some ablations.
- `self.args.use_bias_correction_kl` multiplies `per_token_kl * coef_1` — a secondary correction some recipes enable; not part of vanilla GRPO.

---

## Connections to the rest of the track

- [[grpo]] — the paper whose Eq. 3 this file implements.
- [[dr-grpo]] — the `loss_type="dr_grpo"` branch.
- [[verl-grpo]] — the alternative registry-based implementation.
- [[entropy-logging-patterns]] — how the entropy log lines are used in practice.
