<!-- scope: default entropy, KL, clipping and advantage-normalization settings as released in the
     OpenRLHF, verl and TRL configuration files
     deps: [[entropy-regularization-ppo]]
     see-also: [[entropy-mechanism-llm-rl]], [[sampling-temperature-schedule]], [[kl-control-rlhf]]
-->

# Entropy, KL and advantage defaults in the OpenRLHF, verl and TRL released configurations
- **Core Insight:** In the released configuration files of all three frameworks the entropy bonus is off by
  default (OpenRLHF `actor.entropy_coef=None`, verl `entropy_coeff: 0`, TRL `entropy_coef=0.0`), so entropy is
  logged but not regularized unless the user sets the coefficient.
- **Guideline:** When a run starts from framework defaults, do not assume an entropy bonus or a
  KL-to-reference term is active; read the entropy coefficient, KL coefficient, KL estimator and
  advantage-normalization fields in the config actually used, because their defaults differ per framework.
- **Authors:** OpenRLHF maintainers; verl maintainers (ByteDance Seed); Hugging Face TRL maintainers
- **Year:** files retrieved 2026-09-18 from each project's `main` branch (no release tag pinned)
- **URL:** https://github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/cli/train_ppo_ray.py —
  https://github.com/volcengine/verl/blob/main/verl/trainer/config/actor/actor.yaml —
  https://github.com/volcengine/verl/blob/main/verl/trainer/config/ppo_trainer.yaml —
  https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_config.py
- **Source type:** released config/code
- **Relevant topics:** PPO defaults, GRPO defaults, entropy coefficient, KL estimator, advantage normalization, rollout temperature

## Summary
This card records the default values of the entropy, KL and advantage-normalization settings in three
open RL training frameworks, read directly from their argument parsers and configuration files. It covers
only values that appear at a named locus in those files. It is not a card for a single published artifact:
it is a comparison of three released configurations, and every row below names the file it came from.

## Key Contributions
- The three frameworks expose the same four controls (entropy coefficient, KL coefficient, KL estimator,
  advantage normalization), but their default values are not identical.
- The entropy bonus is disabled by default in all three, so entropy appears in logs as a metric only.
- The KL-to-reference term is disabled by default in verl (`use_kl_loss: false`, `use_kl_in_reward: False`)
  and in TRL (`beta=0.0`), and enabled by default in OpenRLHF (`algo.kl.init_coef=0.01`).
- The default KL estimator is not the same across frameworks: OpenRLHF defaults to `k1`, while verl's
  loss-side default is `low_var_kl` (the k3 form) and its reward-side default is `kl` (the k1 form).
- Advantage normalization by the group standard deviation is on by default in verl
  (`norm_adv_by_std_in_grpo: True`) and in TRL (`scale_rewards="group"`).

## Key Figures/Tables to Study
- OpenRLHF `openrlhf/cli/train_ppo_ray.py`: the `--actor.*`, `--algo.kl.*`, `--algo.advantage.*` block.
- verl `verl/trainer/config/actor/actor.yaml`: `clip_ratio`, `entropy_coeff`, `use_kl_loss`,
  `kl_loss_coef`, `kl_loss_type`; and `verl/trainer/config/ppo_trainer.yaml`: `algorithm.adv_estimator`,
  `algorithm.norm_adv_by_std_in_grpo`, `algorithm.use_kl_in_reward`, `algorithm.kl_penalty`.
- TRL `trl/trainer/grpo_config.py`: `num_generations`, `temperature`, `top_p`, `beta`, `epsilon`,
  `scale_rewards`, `entropy_coef`, `top_entropy_quantile`.

## Technical Details
All values below are defaults as printed in the named file, retrieved 2026-09-18.

- OpenRLHF: `--actor.eps_clip` default `0.2` (train_ppo_ray.py L406); `--actor.entropy_coef` default `None`,
  with the help text "set to 0 means only enable entropy logs" (L469-472); `--algo.kl.init_coef` default
  `0.01` (L438); `--algo.kl.target` default `None` and `--algo.kl.horizon` default `10000`, which are the
  adaptive-KL controls (L436-437); `--algo.kl.estimator` default `"k1"` with choices `k1/k2/k3` (L449-451);
  `--algo.kl.use_loss` default `False` (L524); `--algo.advantage.estimator` default `"gae"` (L515-519);
  `--algo.advantage.no_std_norm` default `False`, described as "disable dividing by std for advantages while
  keeping mean normalization" (L527-530); `--rollout.temperature` default `1.0` and `--rollout.top_p`
  default `1.0` (L417-418); `--rollout.n_samples_per_prompt` default `1` (L433); `--eval.temperature`
  default `0.6` (L576). The file prints a recommendation to use `k1` when KL is not used as a loss and
  `k2` or `k3` when it is (L730-734).
- verl actor config: `clip_ratio: 0.2`, `clip_ratio_low: 0.2`, `clip_ratio_high: 0.2` (actor.yaml L36-42);
  `entropy_coeff: 0` (L93); `calculate_entropy: false` (L96); `use_kl_loss: false` (L103);
  `kl_loss_coef: 0.001` (L113); `kl_loss_type: low_var_kl` (L116); `ppo_mini_batch_size: 256` (L18).
- verl algorithm config: `adv_estimator: gae` (ppo_trainer.yaml L74); `norm_adv_by_std_in_grpo: True` (L77);
  `use_kl_in_reward: False` (L98); `kl_penalty: kl` (L101).
- TRL `GRPOConfig`: `num_generations` default `8`; `temperature` default `1.0`; `top_p` default `1.0`;
  `beta` default `0.0` (the KL-to-reference coefficient); `epsilon` default `0.2` with `epsilon_high`
  defaulting to `epsilon`; `scale_rewards` default `"group"`; `entropy_coef` default `0.0`;
  `top_entropy_quantile` default `1.0`; `use_adaptive_entropy` default `False` with `entropy_target`
  default `0.2` nats and `entropy_coef_delta` default `0.005` (grpo_config.py L64-316).

## Findings relevant to generality and negative feedback
- TRL's `entropy_target` field is documented as a mean per-token entropy in nats and its docstring states
  that the default `0.2` "never triggers regularization (only on near-complete entropy collapse)", and
  recommends setting it near the entropy observed early in training (grpo_config.py L309-316). This is the
  only entropy-collapse threshold stated in any of the three files.
- None of the three files states an empirical result, so they carry no evidence about which value is better
  for any model or dataset. Evidence for entropy control is in [[entropy-mechanism-llm-rl]] and
  [[entropy-collapse-ppo]]; evidence for KL control is in [[kl-control-rlhf]].

## Connections
- [[entropy-regularization-ppo]] gives the entropy-bonus term these `entropy_coef` fields multiply.
- [[entropy-mechanism-llm-rl]] and [[entropy-collapse-ppo]] supply the measured entropy dynamics that these
  defaults do not.
- [[kl-control-rlhf]] defines the k1, k2 and k3 KL estimators named in the OpenRLHF and verl fields.
- [[sampling-temperature-schedule]] covers rollout temperature, whose default is `1.0` in all three files.
- [[grpo]] defines the group-relative advantage that `norm_adv_by_std_in_grpo` and `scale_rewards` normalize.

## Verification
- Checked on 2026-09-18 against: OpenRLHF `openrlhf/cli/train_ppo_ray.py`, verl
  `verl/trainer/config/actor/actor.yaml` and `verl/trainer/config/ppo_trainer.yaml`, TRL
  `trl/trainer/grpo_config.py`, each on the project `main` branch. No release tag or commit is pinned, so a
  later read of `main` may differ.
- Corrections to the previous card version:
  - "verl exposes it with default `1e-3` on some presets" → verl's released actor config sets
    `entropy_coeff: 0` (actor.yaml L93).
  - "Advantage normalization ... is ON by default in OpenRLHF and verl, OFF by default in TRL" → TRL's
    `scale_rewards` defaults to `"group"`, which divides by the group standard deviation
    (grpo_config.py L226); OpenRLHF's default advantage estimator is `gae`, not a group estimator
    (train_ppo_ray.py L515-519).
  - "all three default to k3" → OpenRLHF's `--algo.kl.estimator` defaults to `k1` (L449-451); verl's
    reward-side `kl_penalty` defaults to `kl` (k1) and only its loss-side `kl_loss_type` defaults to
    `low_var_kl` (k3), and that loss is off by default.
  - "Default KL coefficient β: around 0.01–0.1 of the reward scale" → OpenRLHF `algo.kl.init_coef` is
    `0.01` (L438), verl `kl_loss_coef` is `0.001` (actor.yaml L113), TRL `beta` is `0.0`. No file states a
    0.01–0.1 range.
  - "`group_size = 8` is typical small; `group_size = 16–32` is common for reasoning" → TRL
    `num_generations` defaults to `8` and OpenRLHF `rollout.n_samples_per_prompt` defaults to `1`. No file
    states a typical or common value.
- Removed as unsupported by the source: the five-step "community-standard triage" guideline; the claimed
  0.1-nat entropy threshold that is "diagnostic of collapse"; the "community norm for when to raise" the
  entropy coefficient (200 updates); the four "common failure patterns from issue trackers" (entropy crash
  within 100 steps, no entropy change after ~1000 steps, length explosion, NaN ratio); the learning-rate
  range 1e-6 to 5e-6 "halved for 70B"; the rollout-length ranges 1k-4k and 8k-32k; the evaluation sampler
  values `T = 0.0` or `T = 0.6` with `top_p = 0.95` for all three frameworks; the citation of "community
  Discord digests" and "GitHub issues tagged entropy / KL / collapse" as sources, which name no retrievable
  document; the claim that the three frameworks' conventions have "converged"; the statement that these
  defaults "underlie open replications of" [[rlvr-tulu3]] and [[deepseek-r1]].
- Not reported by the source: any measured entropy trajectory, any ablation comparing these defaults, any
  recommendation tied to model size.
