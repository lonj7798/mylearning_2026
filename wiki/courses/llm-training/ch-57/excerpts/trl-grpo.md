---
chapter: ch-57
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/frameworks/trl-grpo.md
source_url: https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py
created_at: "2026-04-23"
---

# Excerpt: TRL GRPOTrainer — the monolithic `_compute_loss`

**Source library:** `wiki/raw-data/llm-training/frameworks/trl-grpo.md`
**Artifact:** `trl/trainer/grpo_trainer.py`. The file is ~2700 LOC; `_compute_loss` lives at ~L2418–2610. Generation + reward pipeline sits upstream at ~L1400–2290.

---

## Why this source anchors ch-57 §2 and §3

Ch-57 §2 introduces `GRPOTrainer` as the HF-ecosystem default for RLHF and the anchor of the stable-trainer set. Ch-57 §3 uses it to discuss the Accelerate-based orchestration story (vLLM server vs colocate, straggler behaviour, weight sync). This excerpt shows the code that bundles every responsibility — generate, score, compute advantage, compute loss, log — into one class. Contrast with verl's split-worker architecture ([[ch-55]]).

---

## The `_compute_loss` body that ch-57 §2 references

Source lines 2418–2580 (condensed):

```python
def _compute_loss(self, model, inputs):
    prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
    completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
    input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
    attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
    logits_to_keep = completion_ids.size(1)
    mask = completion_mask if "tool_mask" not in inputs else completion_mask * inputs["tool_mask"]

    per_token_logps, entropies = self._get_per_token_logps_and_entropies(
        model, input_ids, attention_mask, logits_to_keep, compute_entropy=True, ...
    )

    if self.top_entropy_quantile < 1.0:
        entropy_mask = self.get_high_entropy_mask(
            entropies, mask, 1 - self.top_entropy_quantile)
    else:
        entropy_mask = None

    advantages = inputs["advantages"]               # (B,)  group-relative z-scores
    if advantages.dim() == 1:
        advantages = advantages.unsqueeze(1)
    # ...
```

Four design choices matter for ch-57's framing:

1. **One method, many losses.** `loss_type` is a string: `"grpo"`, `"bnpo"`, `"dr_grpo"`, `"dapo"`, `"cispo"`, `"sapo"`, `"luspo"`, `"vespo"`. Every variant that has emerged since DeepSeekMath 2024 sits behind this switch. This is the TRL design philosophy in concentrated form.
2. **Recompute logprobs from scratch.** Unlike verl, where the rollout worker hands back `old_per_token_logps`, GRPOTrainer always recomputes `per_token_logps` inside `_compute_loss`. It will use `inputs["old_per_token_logps"]` if present; otherwise it uses `per_token_logps.detach()` — turning the ratio into 1.0 and reducing to straight policy gradient.
3. **Entropy is a first-class tensor.** `_get_per_token_logps_and_entropies` returns both per-token logprobs and per-token entropies. The `top_entropy_quantile` mask uses the entropy tensor to filter to the most-uncertain tokens (DAPO's trick).
4. **Advantage broadcast via `unsqueeze(1)`.** The `(B,)` group-relative z-score becomes `(B, 1)` and multiplies every token in the completion. This is where length bias enters; see ch-40 §5 for the Dr.GRPO analysis.

---

## K3 KL estimator inline (ch-57 §2 references this)

Source lines 58–65:

```python
if self.beta != 0.0:
    ref_per_token_logps = inputs["ref_per_token_logps"]
    per_token_kl = (
        torch.exp(ref_per_token_logps - per_token_logps)
        - (ref_per_token_logps - per_token_logps) - 1
    )
    if self.args.use_bias_correction_kl:
        per_token_kl = per_token_kl * coef_1
```

This is the Schulman K3 estimator: `e^x − x − 1` where `x = log(π_ref/π_θ)`. Always ≥ 0, unbiased, one extra reference forward pass. Ch-57 §2 contrasts this with [[trl-ppo]], which uses K1 (`log(π_θ/π_ref)`, biased sign) on the reward and K2 (`0.5·Δlogp²`) only for diagnostics.

`self.args.use_bias_correction_kl` multiplies `per_token_kl * coef_1` — an optional secondary correction; off by default.

---

## The surrogate-loss branch

Source lines 67–77:

```python
if self.loss_type in ["grpo", "bnpo", "dr_grpo", "dapo", "luspo"]:
    coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
    if self.args.delta is not None:                           # DAPO upper-clip cap
        coef_1 = torch.clamp(coef_1, max=self.args.delta)
    per_token_loss1 = coef_1 * advantages
    per_token_loss2 = coef_2 * advantages
    per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
elif self.loss_type == "cispo":
    clamped = torch.clamp(coef_1, max=self.epsilon_high).detach()
    per_token_loss = -clamped * advantages * per_token_logps
```

Standard PPO-clip with asymmetric `epsilon_low`/`epsilon_high` (the DAPO generalization). `torch.min` on `coef_1 * adv` vs `coef_2 * adv` is the pessimistic PPO surrogate. CISPO is a token-weighted importance form that keeps the weight outside the gradient via `.detach()`.

---

## Aggregation — where GRPO and Dr.GRPO differ by one denominator

Source lines 86–93:

```python
if self.loss_type == "grpo":
    loss = ((per_token_loss * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)).mean()
elif self.loss_type == "bnpo":
    loss = (per_token_loss * mask).sum() / mask.sum().clamp(min=1.0)
elif self.loss_type == "dr_grpo":
    loss = (per_token_loss * mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
```

Ch-57 §2 highlights this block: the `loss_type` abstraction is semantically meaningful *exactly because* one-line denominator changes encode whole algorithmic decisions. See ch-40 §5 for the bias analysis.

---

## vLLM IS correction (ch-57 §3 references this in the scaling discussion)

Source lines 81–82:

```python
if self.use_vllm and self.vllm_importance_sampling_correction and self.loss_type != "vespo":
    per_token_loss = per_token_loss * inputs["importance_sampling_ratio"]
```

When rollouts come from vLLM, the per-token logprobs from vLLM may not match the ones from the HF model forward — even with the same weights, numeric differences from CUDA kernel choice can bite. TRL exposes an IS-correction multiplier to fix this. `vllm_mode="colocate"` reduces but does not eliminate the drift; server mode can amplify it because the vLLM process may be several optimizer steps behind.

---

## Attested implementation notes

- TRL always logs `masked_batch_mean(entropies)` to `_metrics[mode]["entropy"]`. This is the canary for entropy collapse. Setting `log_completions=True` also logs full completions for manual inspection.
- `num_generations` in `GRPOConfig` is the `G` of [[grpo]] — typical values are 4, 8, 16, 32.
- Setting `beta=0.0` skips the KL term entirely. Used for pure reward-only RL ablations.
- The trainer supports both `processing_class=tokenizer` (new) and `tokenizer=tokenizer` (deprecated) — TRL transitioned to `processing_class` to align with multi-modal models.

---

## Connections to the rest of the track

- [[grpo]] — the Eq. 3 this file implements verbatim.
- [[trl-ppo]] — the demoted reference trainer.
- [[verl-grpo]] — the split-worker Ray counterpart; algebra identical, architecture inverted.
- [[dr-grpo]] — the `loss_type="dr_grpo"` branch explained.
- [[entropy-logging-patterns]] — how `_metrics[mode]["entropy"]` is interpreted in practice.
