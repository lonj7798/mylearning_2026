---
chapter: ch-58
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/frameworks/verl-ppo-loss.md
source_url: https://github.com/verl-project/verl
created_at: "2026-04-23"
---

# Excerpt: verl PPO loss — the registry pattern as an architectural bet

**Source library:** `wiki/raw-data/llm-training/frameworks/verl-ppo-loss.md`
**Artifact:** `verl/trainer/ppo/core_algos.py` L1080–1140 (`compute_policy_loss_vanilla`), `kl_penalty()` with k1/k2/k3 switches.

---

## Why this source defines ch-58 matrix rows 4, 8, 10, 11

Ch-58's central thesis is "the algebra is the same; the architecture diverges". This source is the attested proof of that thesis for verl: the loss *function* is 60 lines of textbook PPO-clip; the architectural bet is the `@register_policy_loss` decorator.

## Row 4 — "PPO: @register_policy_loss('vanilla')"

Source code excerpt:

> ```python
> @register_policy_loss("vanilla")
> def compute_policy_loss_vanilla(
>     old_log_prob, log_prob, advantages, response_mask,
>     loss_agg_mode="token-mean", config=None, rollout_is_weights=None,
> ) -> tuple[torch.Tensor, dict[str, Any]]:
> ```

The registry decorator *is* the bet. Ch-58 §6 says "writing a new advantage estimator or policy loss as a 30-line `@register_*` function without modifying the trainer is a verl idiom". This source is the evidence that claim is attested, not speculative. A learner who wants to prototype GSPO or CISPO on verl literally adds one decorated function and flips a config key.

## Row 11 — "DAPO asymmetric clip: clip_ratio_low/high + delta"

Source code excerpt:

> ```python
> pg_losses2 = -advantages * torch.clamp(ratio, 1 - clip_ratio_low, 1 + clip_ratio_high)
> ```

Attested asymmetric clip. The source also documents `clip_ratio_c = 3.0` for dual-clip. Ch-58 matrix cell reproduces all three parameter names because they're the *user-facing* knobs in verl's config.

## Row 10 — "vLLM IS correction: rollout_is_weights"

Source:

> ```python
> if rollout_is_weights is not None:
>     pg_losses = pg_losses * rollout_is_weights  # off-policy IS correction (vLLM mismatch)
> ```

Different design from OpenRLHF's three-mode switch. verl accepts pre-computed per-token IS weights; the *policy* for how those weights are computed (TIS vs iCEPO vs seq-mask) lives outside the loss. Ch-58 contrasts this with OpenRLHF's in-loss switch in §3's "vLLM-vs-train KL" row.

## Row 8 — "KL location: reward shaping (k1/k2/k3 switch)"

Source:

> ```python
> def kl_penalty(logprob, ref_logprob, kl_penalty):
>     if kl_penalty == "k1":  return logprob - ref_logprob
>     if kl_penalty == "k2":  return 0.5 * (logprob - ref_logprob) ** 2
>     if kl_penalty == "k3":  # Schulman unbiased
>         diff = ref_logprob - logprob
>         return torch.exp(diff) - diff - 1
> ```

This exact switch is the §3 crib sheet's "K1/K2/K3" row. The source also states that KL is *never* in `compute_policy_loss_vanilla` itself — it enters the reward before GAE. Ch-58's §3 column header "KL(π‖π_ref)" maps to this `kl_penalty()` output.

## What the source says about loss aggregation (matrix row-independent)

> Loss aggregation is parametric: `"token-mean"` (default), `"seq-mean-token-sum"` (Dr.GRPO style), or `"seq-mean-token-mean"` (length-normalized) — each materially changes gradients on long-tailed completion length distributions.

This is a subtle point ch-58 does not elevate into the matrix (rows would balloon), but it's the reason a learner reading §6 graduation criteria "verl's `agg_loss` is more flexible" recognizes that the aggregation mode is an exposed knob, not a hard-coded choice.

## What ch-58 inherits verbatim

- `clip_ratio_low`, `clip_ratio_high`, `clip_ratio_c = 3.0` as the DAPO-family parameter vocabulary.
- The registry decorator name `@register_policy_loss("vanilla")` as the canonical verl idiom.
- `kl_penalty()` k1/k2/k3 switch names (§3 crib sheet cell).
- `rollout_is_weights` as the verl-specific IS-correction input channel.

## Connections

- **[[verl-grpo]]** — `@register_adv_est(GRPO)`: the same registry pattern applied to advantage estimation, not loss.
- **[[openrlhf-ppo]]** — the closest algebraic parallel; ch-58 matrix row 4 contrasts the `nn.Module`-based OpenRLHF design against this free-function + registry.
- **[[trl-ppo]]** — TRL's inlined-in-train-loop PPO; ch-58 §1 argues they share algebra, and this source is the verl half of that argument.
- **[[entropy-logging-patterns]]** — inherits the `actor/ppo_kl` and K1/K2/K3 terminology established here.
