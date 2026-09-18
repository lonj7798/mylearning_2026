<!-- scope: recipe ledger for [[trl-grpo]]: TRL GRPOConfig / GRPOTrainer defaults at commit a08e713 (framework defaults, not a trained model)
     deps: [[trl-grpo]]
     see-also: [[openrlhf-ppo-recipe]], [[dr-grpo]], [[grpo]]
-->

# TRL GRPO defaults at commit a08e713 (recipe ledger for [[trl-grpo]])
- **Source:** https://github.com/huggingface/trl/blob/a08e7139f933b770177fc2abc0b43118e26b260b/trl/trainer/grpo_config.py (commit dated 2026-04-21)
- **Source type:** released config/code
- **Scope:** these are `GRPOConfig` field defaults (`grpo_config.py`) and constants in `grpo_trainer.py`. They are not the settings of any released checkpoint. The "Evidence" column quotes the docstring when it names a paper value; the repository contains no ablation for any row.

## Recipe ledger

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| TRL default (no model) | — | RL | loss type (`loss_type`) | `dapo` | `grpo_config.py` L709 @a08e713 | verified 2026-09-14 | docstring: DAPO token-level normalization "to eliminate length bias" (L237–239); no ablation in repo |
| TRL default (no model) | — | RL | KL coefficient (`beta`) | 0.0 (reference model not loaded) | `grpo_config.py` L589; `grpo_trainer.py` L650–653 | verified 2026-09-14 | docs L100 cite Open-Reasoner-Zero, Dr. GRPO, DAPO; docstring notes DeepSeek-R1 used 0.001 (L163–166) |
| TRL default (no model) | — | RL | lower clip ε (`epsilon`) | 0.2 | `grpo_config.py` L601; `grpo_trainer.py` L616 | verified 2026-09-14 | no ablation reported |
| TRL default (no model) | — | RL | upper clip ε_high (`epsilon_high`) | `None` → equals `epsilon` | `grpo_config.py` L613; `grpo_trainer.py` L617 | verified 2026-09-14 | docstring: DAPO recommends 0.28; for `cispo`, ScaleRL value 5.0 (L175–179) |
| TRL default (no model) | — | RL | two-sided clip bound (`delta`) | `None` (off); docstring recommends > 1 + ε | `grpo_config.py` L605, L171–174 | verified 2026-09-14 | docstring cites INTELLECT-2; no value given |
| TRL default (no model) | — | RL | policy updates per generation batch (`num_iterations`, μ) | 1 | `grpo_config.py` L597 | verified 2026-09-14 | no ablation reported |
| TRL default (no model) | — | RL | completions per prompt (`num_generations`) | 8 (minimum 2) | `grpo_config.py` L392, L938–942 | verified 2026-09-14 | no ablation reported |
| TRL default (no model) | — | RL | max completion length (`max_completion_length`) | 256 tokens | `grpo_config.py` L406 | verified 2026-09-14 | no ablation reported |
| TRL default (no model) | — | RL | sampling temperature; top-p; top-k | 1.0; 1.0; 0 (off) | `grpo_config.py` L440, L444, L451 | verified 2026-09-14 | no ablation reported |
| TRL default (no model) | — | RL | learning rate (`learning_rate`) | 1e-6 | `grpo_config.py` L352–355 | verified 2026-09-14 | no ablation reported |
| TRL default (no model) | — | RL | generation batch | `per_device_train_batch_size` × processes × `steps_per_generation`; `steps_per_generation` defaults to `gradient_accumulation_steps` | `grpo_config.py` L897–901 | verified 2026-09-14 | no ablation reported |
| TRL default (no model) | — | RL | advantage scaling (`scale_rewards`) | `group` (std + 1e-4 in the denominator) | `grpo_config.py` L697; `grpo_trainer.py` L2147–2149 | verified 2026-09-14 | docstring: Dr. GRPO recommends no scaling; PPO Lite recommends `batch` (L218–227) |
| TRL default (no model) | — | RL | multi-reward aggregation (`multi_objective_aggregation`) | `sum_then_normalize`; `reward_weights` `None` → 1.0 each | `grpo_config.py` L685, L678, L205–217 | verified 2026-09-14 | docstring cites GDPO for `normalize_then_sum` |
| TRL default (no model) | — | RL | importance-ratio level (`importance_sampling_level`) | `token` | `grpo_config.py` L668 | verified 2026-09-14 | docstring cites GSPO for `sequence` (L199–204) |
| TRL default (no model) | — | RL | entropy-quantile mask (`top_entropy_quantile`, ρ) | 1.0 (all tokens kept) | `grpo_config.py` L774 | verified 2026-09-14 | docstring: "Beyond the 80/20 Rule" recommends 0.2 (L276–281) |
| TRL default (no model) | — | RL | mask truncated completions (`mask_truncated_completions`) | `False` | `grpo_config.py` L744 | verified 2026-09-14 | docstring cites DAPO as good practice (L259–262) |
| TRL default (no model) | — | RL | vLLM generation (`use_vllm`; `vllm_mode`) | `False`; `colocate` | `grpo_config.py` L495, L502 | verified 2026-09-14 | no ablation reported |
| TRL default (no model) | — | RL | vLLM IS correction (`vllm_importance_sampling_correction`; `_mode`; `_cap` C) | `True`; `sequence_mask`; 3.0 | `grpo_config.py` L792, L801, L814 | verified 2026-09-14 | no ablation reported; docs L207 and docstring L295 describe truncated IS as default (conflicts with field default) |
| TRL default (no model) | — | RL | off-policy sequence mask threshold (`off_policy_mask_threshold`) | `None` (off) | `grpo_config.py` L821 | verified 2026-09-14 | docstring: DeepSeek-V3.2 Eq. 9, example 0.5 (L303–307) |
| TRL default (no model) | — | RL | KL IS bias correction (`use_bias_correction_kl`) | `False` | `grpo_config.py` L830 | verified 2026-09-14 | docstring cites DeepSeek-V3.2 |
| TRL default (no model) | — | RL | SAPO temperatures τ_pos; τ_neg | 1.0; 1.05 | `grpo_config.py` L630, L622 | verified 2026-09-14 | docstring cites SAPO paper |
| TRL default (no model) | — | RL | VESPO k_pos, λ_pos, k_neg, λ_neg | 2.0, 3.0, 3.0, 2.0 | `grpo_config.py` L638, L645, L653, L660 | verified 2026-09-14 | no ablation reported |
| TRL default (no model) | — | RL | reference-model sync (`sync_ref_model`; α; steps) | `False`; 0.6; 512 | `grpo_config.py` L752, L759, L767 | verified 2026-09-14 | docstring cites TR-DPO |
| TRL default (no model) | — | RL | tool-calling turn limit (`max_tool_calling_iterations`) | `None` (no limit; `sys.maxsize`) | `grpo_config.py` L784; `grpo_trainer.py` L541 | verified 2026-09-14 | no ablation reported |

## Verification
- Checked on 2026-09-14 against `trl/trainer/grpo_config.py`, `trl/trainer/grpo_trainer.py` and `docs/source/grpo_trainer.md` at commit a08e7139f933b770177fc2abc0b43118e26b260b.
- The ledger was created in this revision; it has no previous version.
- On `main` at a04ffd3 (2026-09-14) `max_completion_length` defaults to 512 (config L493) and `vllm_importance_sampling_cap` defaults to `None` and is deprecated for `vllm_importance_sampling_clip_max` (config L1034–1041); the other rows above were not re-checked at that commit.
