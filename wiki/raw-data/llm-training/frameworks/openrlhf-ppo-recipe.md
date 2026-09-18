<!-- scope: recipe ledger for [[openrlhf-ppo]]: OpenRLHF PPO/GRPO-family CLI and loss defaults at commit 64c1cc4 (framework defaults, not a trained model)
     deps: [[openrlhf-ppo]]
     see-also: [[trl-grpo-recipe]], [[verl-ppo-loss]]
-->

# OpenRLHF PPO defaults at commit 64c1cc4 (recipe ledger for [[openrlhf-ppo]])
- **Source:** https://github.com/OpenRLHF/OpenRLHF/tree/64c1cc4f30d4c0dab772c8aa1dc11288c425e0da (commit dated 2026-04-19)
- **Source type:** released config/code
- **Scope:** these are framework defaults that apply when a flag is not passed. They are not the settings of any released checkpoint. No row has an ablation in the repository.

## Recipe ledger

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| OpenRLHF default (no model) | — | RL | PPO clip ε (`--actor.eps_clip`) | 0.2 | `openrlhf/cli/train_ppo_ray.py` L387 @64c1cc4 | verified 2026-09-14 | no ablation reported |
| OpenRLHF default (no model) | — | RL | lower/upper clip (`--actor.eps_clip_low_high`) | `None` → (eps_clip, eps_clip) | `train_ppo_ray.py` L388, L594–595 | verified 2026-09-14 | no ablation reported |
| OpenRLHF default (no model) | — | RL | dual-clip bound (`--actor.dual_clip`) | `None` (off); must be > 1.0 if set | `train_ppo_ray.py` L389; `models/loss.py` L106–107 | verified 2026-09-14 | no ablation reported |
| OpenRLHF default (no model) | — | RL | policy loss type (`--actor.policy_loss_type`) | `ppo` (choices `ppo`, `gspo`) | `train_ppo_ray.py` L420 | verified 2026-09-14 | no ablation reported |
| OpenRLHF default (no model) | — | RL | policy-loss aggregation | token-level masked mean for `ppo`; per-sequence mean then batch mean for `gspo` | `models/loss.py` L85, L101–103, L175–179; `trainer/ray/ppo_actor.py` L75–87 | verified 2026-09-14 | no ablation reported |
| OpenRLHF default (no model) | — | RL | value clip (`--critic.value_clip`) | 0.5 | `train_ppo_ray.py` L390 | verified 2026-09-14 | no ablation reported |
| OpenRLHF default (no model) | — | RL | advantage estimator (`--algo.advantage.estimator`) | `gae` (choices `gae`, `reinforce`, `rloo`, `reinforce_baseline`, `group_norm`, `dr_grpo`) | `train_ppo_ray.py` L477–483 | verified 2026-09-14 | no ablation reported |
| OpenRLHF default (no model) | — | RL | GAE γ, λ | 1, 1 | `train_ppo_ray.py` L391–392 | verified 2026-09-14 | no ablation reported |
| OpenRLHF default (no model) | — | RL | KL coefficient β (`--algo.kl.init_coef`) | 0.01 | `train_ppo_ray.py` L419 | verified 2026-09-14 | no ablation reported |
| OpenRLHF default (no model) | — | RL | KL placement (`--algo.kl.use_loss`) | `False` → KL added to per-token reward | `train_ppo_ray.py` L484–486; `experience_maker.py` L205–214 | verified 2026-09-14 | no ablation reported |
| OpenRLHF default (no model) | — | RL | KL estimator (`--algo.kl.estimator`) | `k1`; KL value clamped to [−10, 10] | `train_ppo_ray.py` L421–429; `models/utils.py` L79 | verified 2026-09-14 | no ablation reported |
| OpenRLHF default (no model) | — | RL | KL controller (`--algo.kl.target`, `--algo.kl.horizon`) | target `None` → `FixedKLController`; horizon 10000 | `train_ppo_ray.py` L417–418; `trainer/ppo_trainer.py` L171–176 | verified 2026-09-14 | no ablation reported |
| OpenRLHF default (no model) | — | RL | adaptive-controller error clip | ±0.2 | `trainer/ppo_utils/kl_controller.py` L17 | verified 2026-09-14 | docstring cites arXiv:1909.08593; no ablation in repo |
| OpenRLHF default (no model) | — | RL | entropy coefficient (`--actor.entropy_coef`) | `None` (no entropy term; 0 logs entropy only) | `train_ppo_ray.py` L431–436 | verified 2026-09-14 | no ablation reported |
| OpenRLHF default (no model) | — | RL | reward clip (`--reward.clip_range`) | (−10, 10) | `train_ppo_ray.py` L437 | verified 2026-09-14 | no ablation reported |
| OpenRLHF default (no model) | — | RL | samples per prompt (`--rollout.n_samples_per_prompt`) | 1 (must be > 1 for `rloo`, `reinforce_baseline`, `group_norm`) | `train_ppo_ray.py` L413–415, L608–611 | verified 2026-09-14 | no ablation reported |
| OpenRLHF default (no model) | — | RL | rollout batch (`--rollout.batch_size`; help text "Batch size for make experience") | 1024 | `train_ppo_ray.py` L373 | verified 2026-09-14 | no ablation reported |
| OpenRLHF default (no model) | — | RL | global training batch (`--train.batch_size`) | 128 | `train_ppo_ray.py` L394 | verified 2026-09-14 | no ablation reported |
| OpenRLHF default (no model) | — | RL | max total sequence length (`--data.max_len`) | 2048 tokens (prompt + response) | `train_ppo_ray.py` L379 | verified 2026-09-14 | no ablation reported |
| OpenRLHF default (no model) | — | RL | rollout temperature; top-p | 1.0; 1.0 | `train_ppo_ray.py` L398–399 | verified 2026-09-14 | no ablation reported |
| OpenRLHF default (no model) | — | RL | PPO epochs per rollout batch (`--train.max_epochs`) | 1 | `train_ppo_ray.py` L378 | verified 2026-09-14 | no ablation reported |
| OpenRLHF default (no model) | — | RL | AdamW LR actor; critic | 1e-6; 9e-6 | `train_ppo_ray.py` L465 | verified 2026-09-14 | no ablation reported |
| OpenRLHF default (no model) | — | RL | AdamW betas; eps; weight decay | (0.9, 0.95); 1e-8; 0.0 | `train_ppo_ray.py` L466–468 | verified 2026-09-14 | no ablation reported |
| OpenRLHF default (no model) | — | RL | LR schedule; warmup ratio; min LR ratio | `cosine_with_min_lr`; 0.03; 0.1 | `train_ppo_ray.py` L470–472 | verified 2026-09-14 | no ablation reported |
| OpenRLHF default (no model) | — | RL | gradient clip (`--{actor,critic}.max_norm`) | 1.0 | `train_ppo_ray.py` L474 | verified 2026-09-14 | no ablation reported |
| OpenRLHF default (no model) | — | RL | vLLM IS correction enable; band; type | `False`; [0.5, 5.0]; `tis` | `train_ppo_ray.py` L257–271 | verified 2026-09-14 | no ablation reported |

## Verification
- Checked on 2026-09-14 against the files named in the table at commit 64c1cc4f30d4c0dab772c8aa1dc11288c425e0da.
- The ledger was created in this revision; it has no previous version.
- On `main` at b117b2b (2026-09-14) the IS flags are replaced by `--algo.advantage.is_correction_level`, `_mode`, `_gating`, `_threshold` (commit e53dcf8); values above apply only to 64c1cc4.
