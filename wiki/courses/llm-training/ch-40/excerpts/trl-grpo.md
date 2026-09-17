---
chapter: ch-40
course: llm-training
phase: read
excerpt_of: github.com/huggingface/trl@a08e713 trl/trainer/grpo_trainer.py and grpo_config.py; library card [[trl-grpo]] (verified 2026-09-14)
source_url: https://github.com/huggingface/trl/blob/a08e7139f933b770177fc2abc0b43118e26b260b/trl/trainer/grpo_trainer.py
created_at: "2026-04-23"
revised: "2026-09-15 (generality revision; aligned with the verified card, commit pinned)"
---

# Excerpt: TRL's GRPO trainer — the loss normalizer is one line

Used by [[read]] §6, the negatives section, and the Common-mistakes table.

## Loss normalizers (`_compute_loss`, L2548–2562, verbatim; gradient-accumulation lines omitted)
```python
if self.loss_type in ["grpo", "sapo"]:
    loss = ((per_token_loss * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)).mean()
elif self.loss_type == "bnpo":
    loss = (per_token_loss * mask).sum() / mask.sum().clamp(min=1.0)
elif self.loss_type == "dr_grpo":
    loss = (per_token_loss * mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
elif self.loss_type in ["cispo", "dapo", "vespo"]:
    normalizer = inputs["num_items_in_batch"] / self.accelerator.num_processes
    loss = (per_token_loss * mask).sum() / normalizer
```
`grpo` is the per-sequence mean of DeepSeekMath Eq. 3; `dr_grpo` divides by a constant equal to the completion budget; `dapo` divides by the number of completion tokens in the batch.

## Defaults that differ from the paper formulation
- `loss_type` default is `"dapo"` (grpo_config.py L709), not `"grpo"`.
- `beta` default is 0.0, so no reference model is created (L650–653); the docs cite Open-Reasoner-Zero, Dr. GRPO, and DAPO for this (docs L100).
- `scale_rewards` default is `"group"`: `A = r − group mean`, divided by `group std + 1e-4`; `"batch"` and `"none"` are the alternatives (L2127–2149).
- `epsilon` (lower clip) 0.2; `epsilon_high` optional, with the docstring noting DAPO's recommended 0.28 (grpo_config.py L171–177).
Reproducing a published GRPO run therefore requires setting `loss_type`, `beta`, and `scale_rewards` explicitly.

## Other mechanisms the chapter cites
- KL term: per-token `exp(ref − logp) − (ref − logp) − 1`, the DeepSeekMath Eq. 4 estimator, added after all masks and importance weights (L2493–2497, L2544–2545).
- Zero-variance groups: when every completion of a prompt gets the same reward, `A = 0` for that group and its policy term vanishes, but its tokens still count in `num_items_in_batch` (derived from L2147–2149, L1747–1754). `frac_reward_zero_std` logs the share of such samples (L2150, L2186).
- Sign-specific metrics: the low-clip metric counts `r < 1 − ε_low` only where `A < 0`, the high-clip metric counts `r > 1 + ε_high` only where `A > 0` (L2589–2590).
- `delta` (two-sided clipping, attributed to the INTELLECT-2 report) caps the ratio from above, which by construction affects only negative advantages (derived from L2508–2515).
- `mask_truncated_completions` removes truncated completions from the loss, citing DAPO (config L259–262).
- The entropy-quantile mask keeps tokens above the `(1 − ρ)` entropy quantile, attributed in the docstring to arXiv:2506.01939 "Beyond the 80/20 Rule" with paper value ρ = 0.2 (config L276–281) — this is not a DAPO technique.
- Tool-calling rollouts: tool-result tokens are masked out of the loss, the importance ratio, and the token count (L1472, L2425, L1747–1754).

## Claims removed from the earlier version of this excerpt
"k3 … low-variance, only costs one extra ref forward pass" and "algebraically equivalent to verl for loss_type='grpo'" were not supported by the source; the per-line loci above are from commit a08e713 (2026-04-21), not from an unpinned `main`.
