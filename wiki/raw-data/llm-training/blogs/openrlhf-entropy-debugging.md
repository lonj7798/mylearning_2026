<!-- scope: practitioner notes on entropy debugging in OpenRLHF / verl / TRL stacks
     deps: [[entropy-regularization-ppo]]
     see-also: [[entropy-mechanism-llm-rl]], [[sampling-temperature-schedule]]
-->

# OpenRLHF / verl / TRL Entropy-Debugging Notes
- **Core Insight:** Every major open-source LLM-RL stack (OpenRLHF, verl, TRL) exposes entropy as a logged metric and offers the same small set of knobs to fix collapse — entropy coefficient, KL coefficient, rollout temperature, advantage normalization — with very similar practitioner conventions emerging across their issue trackers.
- **Guideline:** When entropy collapses in an open-source RL run, follow the community-standard triage: (1) confirm KL-to-reference term is on and finite, (2) bump rollout temperature by 0.1–0.2, (3) raise entropy coefficient an order of magnitude, (4) check advantage normalization is per-batch zero-mean unit-var, (5) only then suspect the reward signal.
- **Authors / sources:** OpenRLHF maintainers (Jian Hu et al.); verl authors at ByteDance; HuggingFace TRL team. Primary documents: framework READMEs, GitHub issues tagged "entropy" / "KL" / "collapse", community Discord digests.
- **Year:** 2023–2025 (living documents)
- **URLs:**
  - OpenRLHF — https://github.com/OpenRLHF/OpenRLHF
  - verl — https://github.com/volcengine/verl
  - TRL — https://github.com/huggingface/trl
- **Relevant topics:** PPO tuning, GRPO tuning, entropy logging, advantage normalization, rollout temperature, framework parity

## Abstract (synthesis of practitioner literature)
Across the three leading open RLHF/RLVR stacks, the set of knobs to control entropy is nearly identical, and the best-practice defaults have converged. This page collects the common patterns in one place.

## Key Points
- **Standard logged metrics:** per-token entropy, per-batch KL(π‖π_ref), PPO ratio mean/std, reward mean/std, clipped-fraction, response length histogram. If a run drops per-token entropy below ~0.1 nats and reward hasn't already saturated, it's diagnostic of collapse, not of convergence.
- **Default entropy coefficient `c_H`:** OpenRLHF and TRL default to `0.0` for LLM-RL (counter to pre-LLM PPO practice). verl exposes it with default `1e-3` on some presets. Community norm for when to raise: entropy drops faster than reward rises within the first 200 updates.
- **Default KL coefficient `β`:** around 0.01–0.1 of the reward scale; adaptive-KL is supported in OpenRLHF (adjust `β` to hit a target KL per batch) and is a safer default than fixed-β for new reward functions.
- **Rollout sampler:** all three frameworks default to `T = 1.0`, `top_p = 1.0` for training rollouts. Framework-level integration with vLLM / SGLang handles KV-cache reuse; temperature is set on the sampler, not the policy.
- **Advantage normalization:** all three offer per-batch zero-mean unit-var normalization; it is ON by default in OpenRLHF and verl, OFF by default in TRL (a recurring footgun).
- **GRPO specifics (all three):** `group_size = 8` is typical small; `group_size = 16–32` is common for reasoning; no critic; advantages are group-relative z-scores.
- **Common failure patterns from issue trackers:**
  - Entropy crash within 100 steps → KL term accidentally off or β too small.
  - Reward-but-no-entropy-change after ~1000 steps → advantage normalization misconfigured.
  - Sudden length explosion in rollouts → entropy healthy but reward + length are confounded (reward hacking).
  - `NaN` in PPO ratio → very aggressive update; lower LR and clip range.

## Key Files / Configs to Read
- OpenRLHF: `examples/scripts/train_ppo_llama_ray.sh`, `openrlhf/trainer/ppo_trainer.py` (search for `entropy_loss`, `kl_loss`).
- verl: `verl/trainer/ppo/core_algos.py` (core advantage + KL math), `examples/grpo_trainer/` config yaml.
- TRL: `trl/trainer/ppo_trainer.py` (`entropy_coef`, `kl_ctl`), `trl/trainer/grpo_trainer.py`.

## Technical Details (community-standard defaults)
- **Entropy computation:** exact categorical entropy at each position, averaged over valid (non-pad) tokens.
- **Clip range `ε`:** 0.2 (PPO); GRPO often 0.2 on the ratio.
- **Learning rate:** 1e-6 to 5e-6 for 7B-class policies with `bf16`, halved for 70B.
- **Rollout length:** 1k–4k for standard RLHF; 8k–32k for reasoning RL (DeepSeek-R1-style).
- **KL estimator:** all three default to k3 (`(π_ref/π) − 1 − log(π_ref/π)`) — see **[[kl-control-rlhf]]**.
- **Evaluation vs training sampler:** eval at `T = 0.0` or `T = 0.6` with `top_p = 0.95`; training always at higher `T` / full support.

## Connections
- Framework-level counterpart to the formal analysis in **[[entropy-mechanism-llm-rl]]** and the sweep in **[[entropy-collapse-ppo]]**.
- Pairs with rollout temperature guidance (**[[sampling-temperature-schedule]]**).
- Implements KL-to-reference per **[[kl-control-rlhf]]** conventions.
- Provides the hyperparameters that underlie open replications of **[[rlvr-tulu3]]** and **[[deepseek-r1]]**.
