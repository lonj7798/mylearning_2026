<!-- scope: GRPO loss in HuggingFace TRL
     deps: [[grpo]]
     see-also: [[verl-grpo]], [[trl-ppo]], [[trl-online-dpo]], [[entropy-logging-patterns]]
-->

# HuggingFace TRL — GRPO Trainer
- **Framework:** HuggingFace TRL
- **Repo URL:** https://github.com/huggingface/trl
- **Version/commit:** `main` branch (fetched 2026-04-21)
- **Relevant file(s):** `trl/trainer/grpo_trainer.py` (file ≈ 2700+ lines)
  - `_compute_loss` ≈ lines 2418–2610
  - generation + reward pipeline ≈ lines 1400–2290
- **Core pattern:** A single monolithic `_compute_loss` that (1) recomputes per-token logprobs and entropies, (2) forms token or sequence-level importance ratios, (3) builds the clipped objective with multiple `loss_type` branches (`grpo`, `dr_grpo`, `dapo`, `cispo`, `sapo`, `luspo`, `vespo`, `bnpo`), (4) adds β·per-token-KL, (5) aggregates with loss-type-specific normalizers. Entropy and KL are logged every step.
- **Why it matters:** TRL is the default RLHF trainer for most HF-ecosystem teams; this file is effectively the reference for how the modern GRPO family (including Dr.GRPO, DAPO, CISPO, and entropy-masked variants) is composed in one place.

## Context
TRL's GRPO trainer is where the 2024-2025 GRPO zoo gets unified. One base class, one `_compute_loss`, many `loss_type` switches. Advantages are computed upstream (group-relative reward z-scoring); the loss function itself focuses on the surrogate objective, KL regularizer, optional entropy-quantile masking (filter to top-k% highest-entropy tokens — the DAPO + Muon paper trick), and optional token-level vLLM IS correction.

## Code excerpt
```python
# trl/trainer/grpo_trainer.py, ≈ lines 2418–2580 (_compute_loss body, condensed)
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

    old_per_token_logps = inputs.get("old_per_token_logps")
    old_per_token_logps = (per_token_logps.detach()
                           if old_per_token_logps is None else old_per_token_logps)

    log_ratio = per_token_logps - old_per_token_logps
    if self.importance_sampling_level == "token":
        log_importance_weights = log_ratio
    elif self.importance_sampling_level == "sequence":            # GSPO-style
        log_importance_weights = (log_ratio * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)
        log_importance_weights = log_importance_weights.unsqueeze(-1)

    coef_1 = torch.exp(log_importance_weights)

    # KL-to-reference (Schulman k3 unbiased estimator)
    if self.beta != 0.0:
        ref_per_token_logps = inputs["ref_per_token_logps"]
        per_token_kl = (
            torch.exp(ref_per_token_logps - per_token_logps)
            - (ref_per_token_logps - per_token_logps) - 1
        )
        if self.args.use_bias_correction_kl:
            per_token_kl = per_token_kl * coef_1

    # ---- surrogate (GRPO family) ----
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

    if entropy_mask is not None:
        per_token_loss = per_token_loss * entropy_mask
    if self.use_vllm and self.vllm_importance_sampling_correction and self.loss_type != "vespo":
        per_token_loss = per_token_loss * inputs["importance_sampling_ratio"]
    if self.beta != 0.0:
        per_token_loss = per_token_loss + self.beta * per_token_kl

    # ---- aggregation (varies per loss_type) ----
    if self.loss_type == "grpo":
        loss = ((per_token_loss * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)).mean()
    elif self.loss_type == "bnpo":
        loss = (per_token_loss * mask).sum() / mask.sum().clamp(min=1.0)
    elif self.loss_type == "dr_grpo":
        loss = (per_token_loss * mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
    return loss
```

## What to notice
- **Advantage is broadcast via `unsqueeze(1)`** — the `(B,)` group-relative z-score becomes `(B, 1)` and multiplies every response token; length bias lives here (see Dr.GRPO aggregation fix).
- **Top-entropy masking** (`top_entropy_quantile < 1.0`) keeps only the highest-entropy tokens — the DAPO / Muon observation that gradient signal on low-entropy tokens is mostly noise.
- **K3 KL estimator** (`exp(Δ) − Δ − 1`) is always ≥ 0, low-variance, and only costs one extra ref forward pass. β is configurable and can be set to 0 (pure GRPO reward-only).
- **Loss aggregation is the core GRPO/Dr.GRPO distinction:**
  - `grpo`: per-sequence token-mean, then sample mean — length-biased.
  - `dr_grpo`: per-batch token-sum divided by `(B · max_completion_length)` — length-unbiased (Liu 2025).
  - `bnpo`: batch-normalized; `dapo`/`cispo`/`vespo`: token-sum normalized by `num_items_in_batch/num_processes`.
- **`coef_2 = clamp(coef_1, 1±ε)`** plus `torch.min` on `coef_1 * adv` vs `coef_2 * adv` is the standard PPO-clip trick, just with asymmetric ε.
- **Entropy is always logged** — `masked_batch_mean(entropies)` → `_metrics[mode]["entropy"]` and is the canary for entropy collapse.

## Comparison to paper / to other frameworks
- **vs DeepSeekMath GRPO:** matches the surrogate in Eq. 20 of Shao et al. 2024; TRL adds Dr.GRPO aggregation, DAPO asymmetric clip with `delta` upper cap, CISPO (token-weighted importance), and entropy masking.
- **vs verl `compute_grpo_outcome_advantage` + `compute_policy_loss_vanilla`:** verl splits advantage and loss into two registry hooks; TRL fuses them. Algebraically equivalent for `loss_type="grpo"`.
- **vs OpenRLHF (PolicyLoss + group baseline in exp buffer):** OpenRLHF keeps GRPO support through its PPO loss module plus a pre-advantage z-score step; no built-in Dr.GRPO / CISPO toggles.
