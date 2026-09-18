<!-- scope: recipe ledger for the TRL PPOTrainer card: framework defaults at commit a08e713 and the one documented benchmark run
     deps: [[trl-ppo]]
     see-also: [[ppo]], [[rlhf-instructgpt]]
-->

# Recipe ledger — TRL `PPOTrainer` (commit a08e713)
Companion to [[trl-ppo]]. Framework defaults and documented run values are separate facts.
Loci: `cfg` = `trl/experimental/ppo/ppo_config.py`, `trn` = `trl/experimental/ppo/ppo_trainer.py`,
`docs` = `docs/source/ppo_trainer.md`, all at github.com/huggingface/trl@a08e7139f933b770177fc2abc0b43118e26b260b.

## Framework defaults (`PPOConfig`)

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| TRL PPOConfig default | any | RL | peak learning rate (AdamW) | 3e-6 | cfg L124–127 (docstring L120) | verified 2026-09-14 | no ablation in TRL; PR #5174 (v0.29.0) changed the default from 5e-5 to 3e-6 citing arXiv:2403.17031 |
| TRL PPOConfig default | any | RL | PPO epochs per rollout batch (`num_ppo_epochs`) | 4 | cfg L232–235 | verified 2026-09-14 | no ablation reported |
| TRL PPOConfig default | any | RL | minibatches per rollout batch (`num_mini_batches`) | 1 | cfg L133–136 | verified 2026-09-14 | no ablation reported |
| TRL PPOConfig default | any | RL | KL coefficient (`kl_coef`), applied per token inside the reward | 0.05 | cfg L240–243; applied trn L777 | verified 2026-09-14 | no ablation reported |
| TRL PPOConfig default | any | RL | KL estimator (`kl_estimator`) | "k1" (option "k3") | cfg L244–252; trn L776 | verified 2026-09-14 | no ablation reported; docstring cites Schulman's KL-approximation note |
| TRL PPOConfig default | any | RL | policy ratio clip ε (`cliprange`) | 0.2 | cfg L253–256; trn L846 | verified 2026-09-14 | no ablation reported |
| TRL PPOConfig default | any | RL | value clip range (`cliprange_value`) | 0.2 | cfg L261–264; trn L831–835 | verified 2026-09-14 | no ablation reported |
| TRL PPOConfig default | any | RL | value loss coefficient (`vf_coef`) | 0.1 | cfg L257–260; trn L849 | verified 2026-09-14 | no ablation reported |
| TRL PPOConfig default | any | RL | discount γ (`gamma`) / GAE λ (`lam`) | 1.0 / 0.95 | cfg L265–272; trn L794–795 | verified 2026-09-14 | no ablation reported |
| TRL PPOConfig default | any | RL | sampling temperature | 0.7 | cfg L169–172; trn L622–628 (top_k 0.0, top_p 1.0) | verified 2026-09-14 | no ablation reported |
| TRL PPOConfig default | any | RL | max response length (`response_length`, tokens) | 53 | cfg L151–154 | verified 2026-09-14 | no ablation reported |
| TRL PPOConfig default | any | RL | reward whitening (`whiten_rewards`) | False | cfg L236–239 | verified 2026-09-14 | no ablation reported |
| TRL PPOConfig default | any | RL | missing-EOS penalty (`missing_eos_penalty`) | None (off) | cfg L173–180; trn L760–761 | verified 2026-09-14 | no ablation reported |
| TRL PPOTrainer (hard-coded) | any | RL | samples per prompt | 1 (each query is generated once; no repetition) | trn L671–756 | verified 2026-09-14 | no ablation reported |
| TRL example `examples/scripts/ppo/ppo_tldr.py` | any | RL | value-model and reference-policy initialisation | value model from `reward_model_path`; policy from `sft_model_path`; reference from `sft_model_path` when no PEFT config | ppo_tldr.py L100–120 | verified 2026-09-14 | no ablation reported |
| TRL PPOConfig default | any | RL | gradient checkpointing / bf16 / logging_steps | True / True if fp16 unset / 10 | cfg docstring L115–119 | verified 2026-09-14 | no ablation reported |

## Documented benchmark run (TL;DR summarization, Pythia 1B)

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| trl-lib/ppo_tldr (init `cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr`) | 1B | RL | reward model | `cleanrl/EleutherAI_pythia-1b-deduped__reward__tldr` | docs L188–189 | verified 2026-09-14 | SFT/RM taken from the N+ implementation-details paper (docs L175) |
| trl-lib/ppo_tldr | 1B | RL | dataset | `trl-lib/tldr` (validation split for eval) | docs L180–181 | verified 2026-09-14 | no ablation reported |
| trl-lib/ppo_tldr | 1B | RL | learning rate | 3e-6 | docs L183 | verified 2026-09-14 | no ablation reported |
| trl-lib/ppo_tldr | 1B | RL | per-device batch × gradient accumulation | 16 × 4 | docs L184–185 | verified 2026-09-14 | no ablation reported |
| trl-lib/ppo_tldr | 1B | RL | total episodes | 1000000 | docs L186 | verified 2026-09-14 | no ablation reported |
| trl-lib/ppo_tldr | 1B | RL | missing-EOS penalty / stop token | 1.0 / eos | docs L191–192 | verified 2026-09-14 | docs L63 recommend the "EOS trick"; no ablation reported |
| trl-lib/ppo_tldr | 1B | RL | parallelism | DeepSpeed ZeRO-2 accelerate config | docs L178 | verified 2026-09-14 | no ablation reported |
| trl-lib/ppo_tldr | 1B | RL | number of GPUs, global batch, KL coef, epochs, TRL version | not reported | checked docs L173–221 and cfg | not reported | the run is tagged `pr-1540` (docs L214); config defaults at that PR are not recorded in the docs |
| trl-lib/ppo_tldr | 1B | eval-gate | "preferred rate", GPT-4o mini as judge | PPO 64.7% vs SFT 33.0% | docs L202 | verified 2026-09-14 | single reported evaluation; number of prompts and seeds not reported |

Note: the benchmark command sets only the flags listed above. The values it inherited from the TRL version at PR #1540
are not stated in the docs, so the defaults table above must not be read as the configuration of that run.

## Verification
- Checked on 2026-09-14 against the files above at commit a08e713 (2026-04-21), TRL release notes v0.29.0, and PR #5174.
- This file is new; values were not transferred from the previous card, which listed no hyperparameters.
