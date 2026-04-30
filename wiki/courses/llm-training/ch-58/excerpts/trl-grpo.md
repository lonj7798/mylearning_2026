---
chapter: ch-58
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/frameworks/trl-grpo.md
source_url: https://github.com/huggingface/trl
created_at: "2026-04-23"
---

# Excerpt: TRL GRPOTrainer — why TRL owns the §5 "zoo breadth" and "read / learn" leaves

**Source library:** `wiki/raw-data/llm-training/frameworks/trl-grpo.md`
**Artifact:** `trl/trainer/grpo_trainer.py` (file ≈ 2700+ lines), `_compute_loss` L2418–2610.

---

## Why this source defines ch-58 matrix rows 5, 8, 9, 10, 12 — and §5 Q5

Ch-58 §5 Q5 routes "need a feature TRL has and the others do not" to TRL. The feature list in Q5 — CISPO, SAPO, LUSPO, entropy-quantile masking, KTO, SimPO, ORPO — is attested by this source's loss_type enumeration.

## Row 5 — "GRPO: GRPOTrainer loss_type switch ({grpo, dr_grpo, bnpo, dapo, cispo, sapo, luspo, vespo})"

Source:

> A single monolithic `_compute_loss` that (1) recomputes per-token logprobs and entropies, (2) forms token or sequence-level importance ratios, (3) builds the clipped objective with multiple `loss_type` branches (`grpo`, `dr_grpo`, `dapo`, `cispo`, `sapo`, `luspo`, `vespo`, `bnpo`), (4) adds β·per-token-KL, (5) aggregates with loss-type-specific normalizers.

All eight names land in the ch-58 matrix cell. A learner re-typing the cell should be able to name which file-and-function unifies them (`_compute_loss` in `grpo_trainer.py`).

## Row 10 — "vLLM IS correction: vllm_importance_sampling_correction"

Source code:

> ```python
> if self.use_vllm and self.vllm_importance_sampling_correction and self.loss_type != "vespo":
>     per_token_loss = per_token_loss * inputs["importance_sampling_ratio"]
> ```

Different design from OpenRLHF (three modes) and verl (`rollout_is_weights`). TRL multiplies a precomputed `importance_sampling_ratio` into the per-token loss — simpler, single-mode. Ch-58 matrix cell attests this.

## Row 12 — "GSPO: importance_sampling_level='sequence'"

Source code:

> ```python
> elif self.importance_sampling_level == "sequence":            # GSPO-style
>     log_importance_weights = (log_ratio * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)
>     log_importance_weights = log_importance_weights.unsqueeze(-1)
> ```

Same algebra as OpenRLHF's `policy_loss_type="gspo"` (mean log-ratio over the response, then exp). The *name* of the knob differs — ch-58 cell captures both names for cross-framework debugging.

## Row 8 — "KL location: GRPO β·K3 added to loss"

Source code:

> ```python
> per_token_kl = (
>     torch.exp(ref_per_token_logps - per_token_logps)
>     - (ref_per_token_logps - per_token_logps) - 1
> )
> ...
> if self.beta != 0.0:
>     per_token_loss = per_token_loss + self.beta * per_token_kl
> ```

This is the *only* place across the three frameworks where KL enters the loss directly rather than reward-shaping (the §3 crib sheet explicitly flags this as "TRL GRPO only"). K3 is the Schulman unbiased estimator; β can be 0 for pure reward-only GRPO.

## Row 9 — "Entropy logging: true H (categorical)"

Source:

> Entropy is always logged — `masked_batch_mean(entropies)` → `_metrics[mode]["entropy"]` and is the canary for entropy collapse.

The `compute_entropy=True` flag in `_get_per_token_logps_and_entropies` returns true categorical entropy, same as verl `actor/entropy`. Ch-58 §3's "TRL GRPO entropy" cell maps here.

## The entropy-quantile masking feature — §5 Q5 depends on this

Source code:

> ```python
> if self.top_entropy_quantile < 1.0:
>     entropy_mask = self.get_high_entropy_mask(
>         entropies, mask, 1 - self.top_entropy_quantile)
> ```

This is the DAPO + Muon paper's trick: only the top-k% highest-entropy tokens get gradient signal. Ch-58 §5 Q5 routes here because no equivalent registry hook exists in OpenRLHF mainline, and verl would require writing one.

## DAPO upper cap — the `delta` parameter

Source:

> ```python
> if self.args.delta is not None:                           # DAPO upper-clip cap
>     coef_1 = torch.clamp(coef_1, max=self.args.delta)
> ```

Matrix row 11 cell "GRPO: epsilon_low, epsilon_high, delta cap" is this exact code path.

## What ch-58 inherits verbatim

- All eight `loss_type` names (grpo, dr_grpo, bnpo, dapo, cispo, sapo, luspo, vespo) into matrix row 5.
- `importance_sampling_level`, `vllm_importance_sampling_correction`, `top_entropy_quantile`, `delta` as TRL-specific knob names.
- `K3` (per-token KL in loss) as the §3 crib sheet's "TRL GRPO" KL-to-ref cell.
- `_metrics[mode]["entropy"]` as the §3 entropy logging key.

## Connections

- **[[trl-ppo]]** — the sister PPO trainer; matrix row 4.
- **[[trl-online-dpo]]** — the sister online-DPO trainer; matrix row 7.
- **[[verl-grpo]]** — the registry-split equivalent; ch-58 §1 "same algebra, different home" uses this pair as the attested argument.
- **[[entropy-logging-patterns]]** — the cross-framework KL-in-loss row originates here.
