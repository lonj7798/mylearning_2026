<!-- scope: Hugging Face TRL GRPOTrainer loss family (_compute_loss), advantage computation, vLLM IS correction, and tool-calling rollout masking, pinned at commit a08e713
     deps: [[grpo]]
     see-also: [[trl-grpo-recipe]], [[dr-grpo]], [[verl-grpo]], [[trl-ppo]], [[openrlhf-ppo]], [[trl-online-dpo]], [[entropy-logging-patterns]]
-->

# Hugging Face TRL `trl/trainer/grpo_trainer.py` `GRPOTrainer` (commit a08e713)
- **Core Insight:** At commit a08e713, `GRPOTrainer._compute_loss` implements eight `loss_type` values (`grpo`, `bnpo`, `dr_grpo`, `dapo`, `cispo`, `sapo`, `luspo`, `vespo`) that differ in the per-token surrogate and in the loss normalizer; the defaults are `loss_type="dapo"`, β = 0.0 (no reference model), and group-std advantage scaling, and tool-result tokens are excluded from the loss by `tool_mask`.
- **Guideline:** When reproducing a GRPO result with TRL, set `loss_type`, `scale_rewards` and `beta` explicitly, because the defaults `loss_type="dapo"` and `beta=0.0` differ from the DeepSeekMath formulation (per-sequence length normalization with a KL term, [[grpo]] Eq. 3), and the config docstring states that `grpo` normalization "tends to prefer shorter completions with positive advantages and longer ones with negative advantages" (grpo_config.py L231–233).
- **Authors:** Hugging Face TRL maintainers (github.com/huggingface/trl); pinned commit authored by xuanduy04 with Quentin Gallouédec.
- **Year:** 2026 (pinned commit a08e713, 2026-04-21)
- **URL:** https://github.com/huggingface/trl/blob/a08e7139f933b770177fc2abc0b43118e26b260b/trl/trainer/grpo_trainer.py
- **Source type:** released config/code
- **Relevant topics:** GRPO loss variants, length normalization, asymmetric and two-sided clipping, k3 KL estimator, entropy-quantile masking, train–inference IS correction, off-policy sequence masking, tool-calling rollouts

## Summary
The artifact is TRL's GRPO trainer at commit a08e713: `trl/trainer/grpo_trainer.py` (2713 lines), `trl/trainer/grpo_config.py`, and `docs/source/grpo_trainer.md`. The trainer samples `num_generations` completions per prompt, optionally runs a tool-calling loop, scores completions with reward functions, computes group-relative advantages, and minimizes a clipped or soft-gated surrogate. The config docstring links most loss options to the paper they come from. The repository reports no training results for these defaults except a docs note that the quick-start run (Qwen2-0.5B-Instruct on DeepMath-103K, 8 GPUs) "takes approximately 1 day" (docs L50). Defaults are listed in [[trl-grpo-recipe]].

## Key Contributions
- One `_compute_loss` (L2418–2614) for eight loss types, with token- or sequence-level importance ratios (L2478–2490).
- Advantage scaling by group std, batch std, or none, and two multi-reward aggregation orders (L2124–2167).
- Optional k3 KL term, entropy-quantile token mask, off-policy sequence mask, and vLLM IS correction (L2444–2545, L2044–2073).
- Tool and environment rollouts whose tool-output tokens are masked from the loss (L1467–1672, L2425).
- Sign-split clipping metrics and entropy logging every step (L2571–2612).

## Key Figures/Tables to Study (code loci at a08e713)
- `grpo_trainer.py` L2418–2614 (`_compute_loss`), L2347–2416 (off-policy mask, VESPO weights), L988–1024 (entropy mask).
- `grpo_trainer.py` L2044–2073 (vLLM IS ratio), L2118–2186 (rewards → advantages), L1467–1672 (tool loop).
- `grpo_config.py` L163–311 (parameter docstring with paper links), L878–945 (batch-size validation).
- `docs/source/grpo_trainer.md` L54–166 (method and loss types), L264–293 (train–inference mismatch), L618–769 (agent training).

## Technical Details
Verbatim excerpt, `grpo_trainer.py` L2507–2515 and L2548–2562 (gradient-accumulation division lines omitted):
```python
elif self.loss_type in ["grpo", "bnpo", "dr_grpo", "dapo", "luspo"]:
    coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
    # Two-sided clipping
    if self.args.delta is not None:
        coef_1 = torch.clamp(coef_1, max=self.args.delta)
    per_token_loss1 = coef_1 * advantages
    per_token_loss2 = coef_2 * advantages
    per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
...
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
- **Mask.** `mask = completion_mask × tool_mask` when a tool mask exists (L2425).
- **Advantages.** A = r − mean over the prompt's group, divided by (std + 1e-4) where std is the group std (`scale_rewards="group"`, default), the batch std (`"batch"`), or not applied (`"none"`) (L2127–2149). A has shape (B,) and is broadcast to every token (L2450–2454). `frac_reward_zero_std` logs the fraction of samples with zero reward std (L2150, L2186).
- **Ratio.** r = exp(log π_θ − log π_old) per token; `"sequence"` uses the masked mean log ratio per sequence (L2478–2490). If `old_per_token_logps` is absent, π_old = π_θ.detach() (L2460–2461).
- **Clipping.** ε_low = `epsilon` (0.2); ε_high = `epsilon_high`, else `epsilon` (L616–617; grpo_config.py L601, L613). Docstring: DAPO recommends ε_high = 0.28; `delta` enables the upper bound of two-sided GRPO from the INTELLECT-2 report and should exceed 1 + ε (grpo_config.py L171–177). The code comment for `delta` reads "Two-sided clipping" (L2509).
- **Other surrogates.** `cispo`: −min(r, ε_high).detach()·A·log π (L2504–2506; docstring cites MiniMax-M1 and gives ε_high = 5.0 from ScaleRL, config L178–179, L244–247). `sapo`: −σ(τ(r − 1))·(4/τ)·A with τ = τ_pos if A > 0 else τ_neg (L2516–2519). `vespo`: −φ(w)·A·log π, φ(w) = exp(λ + k·log w − λ·w), log w = sum of per-token log ratios plus the log vLLM IS ratios, k and λ chosen by the sign of A (L2370–2416, L2520–2531). `luspo`: sequence loss × sequence length (L2563–2565).
- **KL.** Per-token exp(ref − logp) − (ref − logp) − 1 (L2493–2497), the estimator of DeepSeekMath Eq. (4) ([[grpo]] §4.1.1). `use_bias_correction_kl` multiplies it by r (L2499–2500). β·KL is added after all masks and IS weights (L2544–2545). With β = 0.0 no reference model is created (L650–653). The docs motivate β = 0 by citing Open-Reasoner-Zero, Dr. GRPO and DAPO (docs L100).
- **Masks applied to the policy term** in this order: off-policy sequence mask, entropy mask, vLLM IS ratio (skipped for `vespo`) (L2535–2542). Entropy mask: keep tokens with entropy ≥ the (1 − ρ) quantile of all valid tokens gathered across processes (L988–1024, L2444–2445); docstring: ρ from "Beyond the 80/20 Rule" (arXiv:2506.01939), paper value 0.2 (config L276–281).
- **vLLM IS ratio.** exp(old logp − sampling logp) over non-tool tokens; sequence modes sum the per-token log differences (L2046–2056). `*_truncate`: clamp at C; `*_mask`: set to 0 above C (L2062–2069). Defaults: on, `sequence_mask`, C = 3.0 (config L792–819); used only when `use_vllm=True` (default `False`, config L495; L2045, L2541).
- **Liger path.** With `use_liger_kernel`, `compute_loss` calls `compute_liger_loss` instead of `_compute_loss` (L2336–2344); `delta` is rejected with Liger (config L944–945).

## Recipe ledger
Framework defaults (not a trained model) are in [[trl-grpo-recipe]].

## Findings relevant to negative feedback (code behavior; no experiments in the source)
Meanings used: negative as gradient (A < 0) and negative as content (tool errors in context).
- **Sign-specific handling.** Low-clip metric counts r < 1 − ε_low only where A < 0; high-clip counts r > 1 + ε_high only where A > 0 (L2589–2590). `sapo` uses τ_neg = 1.05 vs τ_pos = 1.0 (config L622–635). The off-policy sequence mask drops a sequence only if A < 0 and its mean KL between the sampling and current policy exceeds the threshold (DeepSeek-V3.2 Eq. 9; L2347–2366; config L303–307).
- **`delta` acts on A < 0 only (derived from L2508–2515, δ > 1 + ε_high).** Example with ε = 0.2 and δ = 3 (example value, not a default): A = −1, r = 5 gives min(−3, −1.2) = −3, loss 3 with zero gradient in r; without δ the loss is 5. A = +1, r = 5 gives objective 1.2 (loss −1.2) with or without δ.
- **Zero-variance groups.** When all completions of a prompt get the same reward, A = 0 for the group and its policy term is 0; its tokens still count in `num_items_in_batch` (derived from L2147–2149, L1747–1754).
- **Truncated completions.** `mask_truncated_completions` removes truncated completions from the loss so they are not penalized; docstring cites DAPO (config L259–262; L1901–1909).

## Findings relevant to agentic training (code behavior; no experiments in the source)
- **Inputs.** `tools` (callables), experimental `environment_factory` (one environment per rollout), or experimental `rollout_func`, whose `env_mask` is used as `tool_mask` (L229–252, L439–452, L1741–1743). Docs: the loop needs a prefix-preserving chat template and TRL swaps in a patched template for known families (docs L636).
- **Loop.** Tool results are appended as `role: tool` messages, token ids are concatenated without re-tokenization, and generation resumes (L1536–1564, L1619–1622). `tool_mask` is 0 for tool-result tokens and 1 for model tokens; their logprobs are filled with 0.0 (L1472, L1638–1648). The loop ends when no tool call is emitted or after `max_tool_calling_iterations` (default `None` = no limit) (L541, L1479).
- **Failures as content.** Exceptions, unknown tool names and unsupported call types become `{"error": ...}` tool messages that the model reads on the next turn; `tools/call_frequency` and `tools/failure_frequency` are logged (L1502–1532, L1778–1786).
- **Length budget.** A tool result that would exceed `max_completion_length` or the model's maximum length is dropped and the sample leaves the loop (L1566–1592; commit 2761732, 2026-04-14); post-tool generation is truncated to the budget (L1624–1636).
- **Tokens excluded.** Tool tokens are excluded from the loss mask (L2425), the IS ratio (L2046), completion-length metrics, and `num_items_in_batch`, the `dapo`/`cispo`/`vespo` normalizer (L1747–1754; docs L172–179).
- Not in the source: a rationale or ablation for masking tool tokens, or any handling of out-of-distribution tool outputs beyond masking.

## Connections
- [[grpo]] — DeepSeekMath GRPO objective (Eq. 3) and KL estimator (Eq. 4) that `loss_type="grpo"` and the KL term follow.
- [[dr-grpo]] — source of `dr_grpo` normalization and of `scale_rewards="none"` (config L225–236).
- [[openrlhf-ppo]] — OpenRLHF `dual_clip` bounds A < 0 tokens as `delta` does here.
- [[verl-grpo]], [[trl-ppo]], [[trl-online-dpo]], [[entropy-logging-patterns]] — other trainers and metric conventions.
- DAPO (arXiv:2503.14476), INTELLECT-2 (arXiv:2505.07291), GSPO (arXiv:2507.18071), MiniMax-M1 (arXiv:2506.13585), Beyond the 80/20 Rule (arXiv:2506.01939), DeepSeek-V3.2 (arXiv:2512.02556) — cited in `grpo_config.py`; no card in this library.

## Verification
- Checked on 2026-09-14 against: github.com/huggingface/trl@a08e7139f933b770177fc2abc0b43118e26b260b (last `main` commit on 2026-04-21; `grpo_trainer.py` last changed 2026-04-20) — `trl/trainer/grpo_trainer.py`, `trl/trainer/grpo_config.py`, `docs/source/grpo_trainer.md`; also `grpo_trainer.py` and `grpo_config.py` at current `main` a04ffd3.
- Corrections to the previous card version:
  - "`main` branch (fetched 2026-04-21)", "`_compute_loss` ≈ lines 2418–2610", "generation + reward ≈ 1400–2290" → pinned to a08e713; `_compute_loss` L2418–2614; tool loop L1467–1672; `_generate_and_score_completions` L1800–2286.
  - "`delta` … DAPO upper-clip cap" → docstring attributes two-sided clipping to INTELLECT-2 (config L171–174).
  - "top-entropy masking … the DAPO / Muon observation" → docstring attributes ρ to "Beyond the 80/20 Rule" (config L276–281).
  - "matches the surrogate in Eq. 20 of Shao et al. 2024" → the clipped objective is DeepSeekMath Eq. (3) and the KL estimator Eq. (4); Eq. (20) is the gradient.
  - "OpenRLHF … no built-in Dr.GRPO toggle" → OpenRLHF@64c1cc4 has a `dr_grpo` advantage estimator (experience_maker.py L267–268).
  - "optional token-level vLLM IS correction" → when `use_vllm=True`, IS correction is on by default and uses `sequence_mask` (config L792–801).
  - Added: the default `loss_type` is `dapo` (config L709); the previous card did not state it.
- Removed as unsupported by the source: "TRL is the default RLHF trainer for most HF-ecosystem teams"; "gradient signal on low-entropy tokens is mostly noise"; "K3 … low-variance, only costs one extra ref forward pass"; "algebraically equivalent to verl for `loss_type="grpo"`"; "Entropy … is the canary for entropy collapse".
- Not reported by the source: ablations for any default; a reason for masking tool tokens. Internal inconsistency: docs L207 and config L295 call truncated IS the default, but the field default is `sequence_mask` (config L801).
- Later changes on `main` (a04ffd3, 2026-09-14): `_compute_loss` moves to L3113; the `dapo`/`cispo`/`vespo` normalizer is rescaled by gradient-accumulation steps / `steps_per_generation` (L3261–3266); `vllm_importance_sampling_cap` is deprecated for `vllm_importance_sampling_clip_max` (config L1034–1041); `max_completion_length` default is 512 (config L493).
