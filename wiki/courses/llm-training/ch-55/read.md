<!-- chapter: ch-55
     track: infra
     kind: content
     title: verl Internals: Losses, Rollouts, Multi-Domain Rewards, and In-Loop Validation
     deps: [ch-54]
     sources: [[verl-ppo-loss]], [[verl-grpo]], [[verl-rollout]], [[entropy-logging-patterns]], [[verl-dapo-recipe]], [[gigpo-verl-agent]], [[deepspeed-ulysses]], [[prorl]], [[dr-grpo]], [[grpo]], [[ppo]], [[kl-control-rlhf]], [[async-rollout]], [[rlvr-beyond-base-model]]
     figures: figures/verl-structure.html, figures/verl-loss-branches.html
     revised: 2026-09 (generality revision)
-->

# Chapter 55 — verl Internals: Losses, Rollouts, Multi-Domain Rewards, and In-Loop Validation

> **Core insight.** verl separates three things that are often fused in other code bases: the advantage estimator (`@register_adv_est`), the policy loss (`@register_policy_loss`), and the rollout engine (a separate worker process). Because of that separation, PPO, GRPO, Dr. GRPO, DAPO and their relatives differ by configuration rather than by a different training loop. The settings that decide whether a run produces a narrow task specialist or keeps broad capability are also configuration: which reward function each `data_source` routes to, whether the KL term to the reference policy is placed in the loss (`actor.use_kl_loss`) or in the reward (`algorithm.use_kl_in_reward`), and which validation sets run at `trainer.test_freq`.
>
> **Guideline.** When you read verl to understand an algorithm, start from `verl/trainer/ppo/core_algos.py` and the registry decorators, because every algorithm in the repository is a registered function selected by name from configuration (core_algos.py L53–67, L116–134). When you train on more than one domain, set `data.reward_fn_key` (default `data_source`) and route each value to its own scorer, because `NaiveRewardManager` reads that field per sample and `default_compute_score` raises `NotImplementedError` for an unregistered value (naive.py L121–136; reward_score/__init__.py L107). When the run must not lose capability outside the trained domain, set `trainer.test_freq` above 0 and include non-target validation files, because the validation path emits metrics per `data_source` and nothing else in the loop measures untrained domains (ray_trainer.py L718–740, L1727–1728).

---

## Why this chapter matters for a general-purpose model

This chapter sits at the RL stage of the pipeline (pre-training → mid-training → SFT → preference optimization → RL → evaluation). At this stage the training signal is one scalar per response, which carries less information than the per-token targets used in pre-training and SFT. If that scalar comes from one verifier on one domain, the optimizer has no term that resists the loss of ability elsewhere. Frameworks decide how easy it is to add the terms that do resist it: multiple reward routes, a KL term to a reference policy, and in-loop evaluation on domains that are not being trained.

verl is the framework several general-capability RL runs report using; the ProRL run that trains one 1.5B policy on math, code, STEM, logic puzzles and instruction following states verl as its framework ([[prorl]], §3.2). Reading verl therefore has a second purpose beyond implementation detail: the set of mechanisms verl exposes is close to the set of levers that an RL run has for staying general.

Facts in this chapter are read at commit `753aed3e1c286ba6825a74342b28669e72c083ea` (committed 2026-09-14) unless stated otherwise. Line numbers move; the function and configuration key names have been stable longer than the line numbers.

---

## §1 The repository at a pinned commit

**Definition.** verl is an RL library for LLM post-training in which a Ray-orchestrated trainer drives separate worker groups for generation, log-probability computation, and optimization.

**The four places where an algorithm lives.**

1. `verl/trainer/ppo/core_algos.py` — advantage estimators and policy losses, each registered by name. `@register_policy_loss` (L53–67) and `@register_adv_est` (L116–134) are plain decorators writing into a dictionary; the `AdvantageEstimator` enum lists 14 names (L88–110).
2. `verl/workers/utils/losses.py` — `ppo_loss` calls `get_policy_loss_fn(config.policy_loss.loss_mode)` (L101–112), then adds the entropy and KL terms after the policy loss returns (L57–144).
3. `verl/trainer/ppo/ray_trainer.py` — the step order: generate, recompute log-probabilities, score, compute advantages (L235–247), update, and validate (L596–770, L1727–1728).
4. `verl/workers/rollout/` — the generation engines, in their own processes.

The default policy loss is `vanilla` (`actor.yaml` L61); eleven other names are registered, including `gspo`, `sapo`, `clip_cov`, `kl_cov`, `cispo`, and `bypass_mode` (core_algos.py L1379–2414) ([[verl-ppo-loss]]).

**Two structural changes you will notice against older material.**

- Worker classes were removed in favor of engine classes in PR #6067, "deprecate workers, migrate to engines", merged 2026-04-20 in commit `044bbba`. Paths of the form `verl/workers/actor/*` no longer exist; the equivalent code is under `verl/workers/engine/`.
- The synchronous SPMD vLLM rollout was retired. `ServerAdapter.generate_sequences` now raises `NotImplementedError` with the message "The vLLM SPMD mode was retired in PR #4411" (`vllm_rollout.py` L325–341), and the rollout config default is `mode: async` (`rollout.yaml` L8) ([[verl-rollout]]). `ServerAdapter` itself was not removed: it is the adapter for server mode and still implements `update_weights` (`vllm_rollout.py` L209–253). The merge date of PR #4411 is not established by the source files read here.

The clickable module map at [figures/verl-structure.html](figures/verl-structure.html) lists each file above with its role, its key functions, and the configuration keys that select it; use it to find the file for a symptom you see in a log.

---

## §2 The policy loss, read end to end

**The problem it addresses.** A policy-gradient step computed from samples drawn by an older policy is a biased and high-variance estimate. The clipped surrogate of [[ppo]] bounds how far one update can move the probability of a sampled token relative to the sampling policy.

**Mechanism, as implemented.** Quoting `core_algos.py` L1336–1361 ([[verl-ppo-loss]]):

```python
    negative_approx_kl = log_prob - old_log_prob
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(
        torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask
    )

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
```

**Formula.** For a response token t,

- when `A_t ≥ 0`: `L_t = max(−A_t·r_t, −A_t·clip(r_t, 1−ε_low, 1+ε_high))`
- when `A_t < 0`: `L_t = min( max(−A_t·r_t, −A_t·clip(r_t, 1−ε_low, 1+ε_high)), −A_t·c )`

`r_t = exp(clamp(log π_θ(t) − log π_old(t), −20, 20))` is the token probability ratio; `π_θ` is the policy being updated and `π_old` the policy whose log-probabilities were stored with the rollout. `A_t` is the token advantage. `ε_low = clip_ratio_low`, `ε_high = clip_ratio_high`, both defaulting to `clip_ratio = 0.2` (`actor.yaml` L36, L39, L42). `c = clip_ratio_c`, default 3.0, asserted greater than 1.0 (`actor.yaml` L83; core_algos.py L1323–1334). On an unclipped branch `∂L_t/∂ log π_θ = −A_t·r_t`; on a clipped branch the gradient is 0.

**Worked example 1 — a negative-advantage token, `A = −1`, `ε_low = ε_high = 0.2`, `c = 3`.**

| `r` | `−A·r` | `−A·clip(r)` | max | dual-clip bound `−A·c` | `L` | gradient |
|---|---|---|---|---|---|---|
| 0.5 | 0.5 | 0.8 | 0.8 | 3.0 | 0.8 | 0 (lower clip active) |
| 1.0 | 1.0 | 1.0 | 1.0 | 3.0 | 1.0 | 1.0 |
| 2.0 | 2.0 | 1.2 | 2.0 | 3.0 | 2.0 | 2.0 |
| 5.0 | 5.0 | 1.2 | 5.0 | 3.0 | 3.0 | 0 (dual clip active) |

Two readings. A token already pushed below `(1−ε_low)·π_old` receives no further push-down in this update. A token whose ratio has run away receives no push-down either, because the loss is capped at `c·|A|` and the cap is constant in `r`. Ye et al. give the reason for the second bound: with off-policy data, "when Â_t < 0, such a large ratio r_t(θ) will introduce a big and unbounded variance" (arXiv:1912.09729v3, Algorithm Design, Eq. 5); their ε = 0.2 and c = 3 are verl's defaults.

**Worked example 2 — Clip-Higher, `A = +1`, `r = 1.25`.** With `ε_high = 0.2`, `clip(1.25, 0.8, 1.2) = 1.2`, the max selects the clipped term and the gradient is 0. With `ε_high = 0.28` the ratio is inside the band and the gradient is `−A·r = −1.25`. That single change is DAPO's Clip-Higher; the verl docs give the configuration as `clip_ratio_low: 0.2`, `clip_ratio_high: 0.28` (docs/algo/dapo.md L52–57) ([[verl-dapo-recipe]]). DAPO keeps `ε_low` at 0.2 "because increasing it will suppress the probability of these tokens to 0, resulting in the collapse of the sampling space" (arXiv:2503.14476v2 §3.1). **Result (single study):** on Qwen2.5-32B base, adding Clip-Higher to the progressive stack moved AIME24 avg@32 from 36 to 38, one run per row (arXiv:2503.14476v2 Table 1).

The interactive at [figures/verl-loss-branches.html](figures/verl-loss-branches.html) lets you move `A`, `r`, `ε_low`, `ε_high` and `c` and shows which of the four branches above is selected and what the gradient is; use it to check the two tables by hand.

**Aggregation.** `agg_loss` (L1140–1206) has five modes. `token-mean`, the default (`actor.yaml` L86), divides the masked sum by the global valid-token count (L1170–1175); the docstring states that aggregating over the global batch makes the loss invariant to the FSDP or Megatron parallelism layout (L1150). `seq-mean-token-sum-norm` divides sequence-summed losses by `loss_scale_factor`, whose default is the response-mask width (L1181–1193). The mode changes the gradient without changing any line of the algebra above, which is why the same objective can carry or not carry a length bias.

**Rollout importance weights.** `rollout_is_weights` multiply the per-token loss (L1364–1365). They exist only when `algorithm.rollout_correction` is configured, rollout log-probabilities are in the batch, and bypass mode is off (ray_trainer.py L1648–1658); the YAML default is `rollout_is: null`, with `"token"` or `"sequence"` as the options (config/algorithm.py L85–89). The configuration comments name two sources of off-policyness: precision mismatch between rollout and training engines, for example vLLM BF16 against FSDP FP32, and stale rollout checkpoints (L70–72). Precision is one cause among several; kernel and implementation differences produce a mismatch even when both engines run the same dtype, which is why the correction is written as a general importance weight rather than a dtype fix.

---

## §3 Where the KL term goes

verl supports both placements of the KL term to a reference policy, and both are off by default. Getting this right matters because the two placements have different gradients and different failure modes.

**In the loss.** `actor.use_kl_loss` (default `false`, `actor.yaml` L103) adds `kl_loss * kl_loss_coef` to the policy loss (losses.py L131–142), with `kl_loss_coef` default 0.001 and `kl_loss_type` default `low_var_kl`, the k3 estimator (`actor.yaml` L113, L116). The GRPO documentation recommends this placement and says to set `use_kl_loss: True` for GRPO (docs/algo/grpo.md L43–47).

**In the reward.** `algorithm.use_kl_in_reward` (default `False`, `ppo_trainer.yaml` L98) replaces token rewards with `token_level_scores − β·kl_penalty(old_log_probs, ref_log_prob)` and logs `actor/reward_kl_penalty` (ray_trainer.py L78–117). The estimator for this path is `algorithm.kl_penalty`, default `kl` (k1) (`ppo_trainer.yaml` L101); the `kl_ctrl` block defaults to `type: fixed`, `kl_coef: 0.001`, `horizon: 10000`, `target_kl: 0.1` (`ppo_trainer.yaml` L109–119). The adaptive controller updates `β ← β·(1 + clip(KL/target − 1, −0.2, 0.2)·n_steps/horizon)` (core_algos.py L153–174) ([[entropy-logging-patterns]]).

**Estimators.** With `δ = log π_θ − log π_ref` for a sampled token: k1 = δ, k2 = δ²/2, k3 = exp(−δ) − 1 + δ. verl clamps the input to ±20 and the k3 output to ±10 (core_algos.py L2239–2245). A `+` suffix such as `k3+` keeps the k3 value but uses the k2 gradient through a straight-through estimator (L2201–2213).

**Worked example.** Take one token with `δ = +0.1`: k1 = 0.1, k3 = exp(−0.1) − 1 + 0.1 = 0.90484 − 1 + 0.1 = 0.00484. Take another with `δ = −0.1`: k1 = −0.1, k3 = exp(0.1) − 1 − 0.1 = 1.10517 − 1 − 0.1 = 0.00517. Averaged over the two tokens, k1 gives 0.0 and k3 gives 0.005. k1 is unbiased for the KL but takes negative values per token, so a k1 curve near zero does not mean the policy is near the reference; k3 is non-negative per token, so its curve can be read directly as a drift magnitude ([[kl-control-rlhf]]).

**Implication for a general-purpose model.** The reference policy is the only term in the objective that refers to behaviour outside the reward's domain. With both placements off, no term in the objective opposes movement away from the starting checkpoint. [[prorl]] argues for keeping it: the authors observe that the view behind removing KL "often applies to base models prior to any supervised fine-tuning", and argue that from a checkpoint already producing coherent chains of thought the KL term helps stability and entropy (§2.3.1). They also hard-reset the reference policy and the optimizer state when validation stagnates or degrades, because the KL term "may increasingly dominate the loss" as training proceeds (§2.3.1, §3.3). Both claims are arguments plus one training history, not ablations; treat the reset rule as one recipe, not a measured effect. verl has no built-in reference-reset scheduler, so this is implemented in the driver script by re-pointing the reference model path at a recent checkpoint.

---

## §4 GRPO, Dr. GRPO, and the Pass@k estimator

**Definition.** GRPO replaces the value model with the mean reward of the G responses sampled for the same prompt ([[grpo]], §4.1.2).

**Mechanism.** `compute_grpo_outcome_advantage` sums each response's token rewards into one score, then normalizes within the group (core_algos.py L311–329) ([[verl-grpo]]):

```python
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                scores_tensor = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(scores_tensor)
                id2std[idx] = torch.std(scores_tensor)
        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]
        scores = scores.unsqueeze(-1) * response_mask
```

`A_i = (r_i − μ_g)/(σ_g + ε)` is copied to every token of response i. `r_i` is the summed token reward, `μ_g` and `σ_g` are the mean and `torch.std` (divisor G−1) over the G responses sharing a prompt id, and `ε = 1e-6` (L272).

**Worked example, binary reward, G = 4.**

1. One correct: μ = 0.25, σ = sqrt((0.75² + 3·0.25²)/3) = 0.5. The correct response gets +1.5, each wrong response −0.5.
2. Three correct: μ = 0.75, σ = 0.5. Each correct response gets +0.5, the wrong one −1.5.
3. Zero or four correct: σ = 0, so every advantage is 0/(0 + 1e-6) = 0 and the group contributes no policy gradient.
4. With `norm_adv_by_std_in_grpo=False`, case 1 gives +0.75 and −0.25, case 2 gives +0.25 and −0.75.

A group of size 1 is a separate case: mean 0 and std 1 are substituted, so the advantage equals the raw reward divided by 1 + 1e-6 (L315–317). The default `rollout.n` is 1 (`rollout.yaml` L127), so a GRPO configuration that forgets to raise it trains on raw rewards with no baseline and, with a 0/1 reward, produces no negative advantages at all.

**Dr. GRPO is four settings, not one.** The docs give the combination as `loss_agg_mode: seq-mean-token-sum-norm`, `loss_scale_factor` (marked optional) set to a constant such as the maximum response length, `use_kl_loss: False`, and `algorithm.norm_adv_by_std_in_grpo: False` (docs/algo/grpo.md L59–62). The std term and the length term are separate fixes at separate places: the std term is in the advantage function, the length term is in `agg_loss`. The two `seq-mean-token-sum*` modes share the same body — sum the loss over each sequence's tokens, then divide by the global batch size — and neither divides a sequence by its own length, so both remove the 1/|o_i| term that [[dr-grpo]] names as the source of the length bias (§3.1). `-norm` adds one further division by `loss_scale_factor`, which is a single constant applied to the whole loss; when `loss_scale_factor` is unset that constant is `loss_mask.shape[-1]`, the padded response width of the current batch, so it changes between batches and moves the effective step size with it (core_algos.py L1181–1193). Setting `loss_scale_factor` to a fixed integer is what makes the normalization constant across training. [[dr-grpo]] reports that removing both the length term and the std term keeps response length from growing during training and lowers the length of incorrect responses on its benchmarks (§3.2, Fig. 5; Qwen2.5-1.5B, R1 template, MATH training questions).

**Pass@k advantages.** `grpo_passk` (L471–530) implements arXiv:2503.19595. It raises an error for groups smaller than 2, gives only the highest-reward response an advantage of `r_max − r_second_max`, optionally divided by the group std, and gives every other response 0 (L516–527). No response receives a negative advantage. With a 0/1 reward the advantage is non-zero only when exactly one response in the group is correct; for rewards [1, 0, 0, 0] the correct response gets 1/0.5 = 2.0. This estimator optimizes coverage rather than the modal answer, which is the quantity [[rlvr-beyond-base-model]] argues is the one that RLVR often fails to improve.

**Multi-objective rewards.** `gdpo` (L361–468) normalizes per reward dimension, takes a weighted sum, and whitens over the batch (L466; arXiv:2601.05242). When a run has several reward components with different scales — a verifier, a format check, a length penalty — this is the entry point that keeps one component from dominating by scale alone.

---

## §5 Multi-domain training: mixed datasets, reward routing, per-domain metrics

**The measurable problem.** A run trained on one `data_source` optimizes one verifier. Nothing in the loss distinguishes "the answer is right" from "the answer satisfies this verifier", so breadth has to come from the data and the routing, not from the objective.

**Mechanism, step by step.**

1. `data.train_files` and `data.val_files` accept a list of parquet files or a single file (`legacy_data.yaml` L7–14). Multiple domains enter as multiple files concatenated into one dataset; `data.shuffle` is True by default (L68). There is no per-file sampling weight in this config, so the mixture ratio is the row count you write into the files.
2. Each row carries a `data_source` string. `data.reward_fn_key` selects which field is read, default `data_source` (`legacy_data.yaml` L29–30).
3. `NaiveRewardManager` reads that field per sample and passes it to the scorer (`naive.py` L121–136):

```python
            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            ...
                    score = self.compute_score(
                        data_source=data_source,
                        solution_str=response_str,
                        ground_truth=ground_truth,
                        extra_info=extra_info,
                    )
```

4. `default_compute_score` is a dispatch table over `data_source` values (`reward_score/__init__.py` L44–107): `openai/gsm8k` to `gsm8k`; `lighteval/MATH`, `DigitalLearningGmbH/MATH-lighteval`, `HuggingFaceH4/MATH-500` to `math_reward`; `math_dapo`, `math`, `math_dapo_reasoning` and anything starting with `aime` to `math_dapo`; six `numina_*` names to `prime_math`; `codecontests`, `apps`, `codeforces`, `taco` to a sandbox execution scorer when `sandbox_fusion.url` is set and to `prime_code` otherwise; `hiyouga/geometry3k` to `geo3k`; seven `searchR1_*` names to an exact-match QA scorer. Any other value raises `NotImplementedError` (L107).
5. Your own domains attach through `reward.custom_reward_function.path` and `.name` (default function name `compute_score`, `reward.yaml`), or through a registered reward manager (`reward.reward_manager.name`, default `naive`). Reward computation runs with `reward.num_workers: 8`.
6. If the scorer returns a dict, its `score` key becomes the reward and every other key is appended to `reward_extra_info` (`naive.py` L145–149). This is the channel that carries a per-domain `acc` field into the metrics described in §6.

**Worked configuration.** For a three-domain run — math, code, instruction following — write three parquet files with `data_source` set to `math_dapo`, `codecontests`, and `my_org/ifeval`; pass all three to `data.train_files`; set `reward.sandbox_fusion.url` so code executes rather than falling back to the in-process scorer; and supply `reward.custom_reward_function.path` with a function that handles `my_org/ifeval` and delegates everything else to `default_compute_score`. Set the row counts to the mixture you want; [[prorl]] used 136K prompts as math 40k, code 24k, STEM 25k, logic 37k, instruction following 10k (§3.1, Table 4), with no ablation of the mix reported.

**Conditions and limits.** A dict-returning scorer must return the same keys for every domain that you want compared, because the metric names are built from those keys. A scorer that returns a continuous score for one domain and a binary score for another produces per-domain metrics that are not on the same scale; the group-relative advantage makes this harmless within a group but not across the logged curves.

---

## §6 In-loop validation

**The measurable problem.** Training reward rises on the trained domains by construction. Whether anything else moved is not observable from the training curves.

**Mechanism.**

1. `trainer.val_before_train` is True by default and runs a full validation pass before step 1 (`ppo_trainer.yaml` L195; ray_trainer.py L1438–1443); `trainer.val_only` runs validation and exits (L198).
2. `trainer.test_freq` is **−1** by default, which disables periodic validation entirely (`ppo_trainer.yaml` L201). The gate is `if self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)` (ray_trainer.py L1727–1728). A run left at the default produces one validation number at the start and one at the end.
3. `_validate` repeats each validation prompt `actor_rollout_ref.rollout.val_kwargs.n` times, scores the responses with the same reward path as training, and records the `data_source` of each sample (ray_trainer.py L596–719).
4. `_val_metrics_update` groups by `data_source` and emits `"{val-core|val-aux}/{data_source}/{var_name}/{metric_name}"` (L721–740). A metric goes to `val-core` when its variable is the core variable — `acc` if present, otherwise `reward` — and its name starts with `mean`, `maj` or `best` at the largest sample count; everything else goes to `val-aux`.
5. `process_validation_metrics` computes, per prompt and then averaged over prompts: `mean@N`, `std@N`, and for each n in 2, 4, 8, … up to N, `best@n/mean`, `best@n/std`, `worst@n/mean`, `worst@n/std`, and `maj@n/*` when the scorer returned a `pred` field. The best/worst and majority statistics come from 1000 bootstrap resamples of size n (`metric_utils.py` L896–1040).
6. For agent runs, `val-aux/num_turns/{min,max,mean}` is emitted when the rollout recorded turn counts (L742–744).

**The default that silently removes coverage measurement.** `val_kwargs` defaults are `temperature: 0`, `n: 1`, `do_sample: False` (`rollout.yaml` L151–170). With N = 1 the code emits `mean@1` and nothing else: `std@N` and every `best@n` are inside the `n_resps > 1` branch (`metric_utils.py` L989). So the pass@k-style metric exists but produces nothing until you set `val_kwargs.do_sample: True`, a non-zero `val_kwargs.temperature`, and `val_kwargs.n` above 1.

**Worked example.** Set `val_kwargs.n: 8`, `temperature: 1.0`, `do_sample: True`, and validation files for `math_dapo` (trained), `openai/gsm8k` (trained), and `hiyouga/geometry3k` (not trained). `gen_ns(8)` returns [2, 4, 8], so for each `data_source` you get `mean@8`, `std@8`, and `best@{2,4,8}/mean` with their bootstrap standard deviations. `val-core/hiyouga/geometry3k/acc/best@8/mean` falling while `val-core/math_dapo/acc/mean@8` rises is narrowing, measured inside the training loop rather than after it. `best@8/mean` is a bootstrap estimate of pass@8 from 8 samples, so at n = N the resamples are drawn with replacement from the same 8 responses and the estimate is not independent of the sample; read the trend, not the absolute value.

**A metric worth adding.** Nothing in the default set measures drift from the starting checkpoint on general prompts. Add a fixed prompt set outside every trained domain, compute the mean per-token k3 KL between the current policy and the reference on it, and log it beside the validation metrics; with `use_kl_loss` on, this is the quantity the KL coefficient is supposed to control, and with it off, it is the quantity nothing controls.

---

## §7 Rollout: the reference path and the production path

**HFRollout** (`hf_rollout.py` L39–177) calls Hugging Face `generate` on the training module. FSDP modules are unsharded with `summon_full_params(writeback=False, recurse=False)` (L108–110), outputs are right-padded to `prompt_length + response_length` (L132–139), and the module docstring records that the class hangs under FSDP HybridShard (L16–18). Use it to compare log-probabilities against the async engine when you suspect a numerical divergence, not to train at scale.

**The async vLLM server.** `vLLMHttpServer.generate` takes prompt token ids and returns token ids with per-token log-probabilities (`vllm_async_server.py` L556–759) ([[verl-rollout]]). The steps that matter when reading a log:

1. `max_possible_tokens = max_model_len − len(prompt_ids)`; below 1 raises `ValueError` (L591–597).
2. `max_tokens` is taken from the request, else `max_new_tokens`, else `min(response_length, prompt_length + response_length − len(prompt_ids))` (L600–611).
3. The value is clamped to `[1, max_possible_tokens]` (L615–619). The clamp reduces `max_tokens` without a warning, and the assert that follows cannot fail after the clamp — a long prompt shortens the response budget with no log line.
4. Requests wait while `_submission_paused` is set (L662–665), then pass `priority` to `engine.generate` (L668–674). `AgentLoopManager` assigns `priority = np.arange(len(prompts))` before chunking, so priority is a per-sample index rather than a staleness signal (`agent_loop.py` L1210–1214).
5. An aborted request returns empty token ids with `stop_reason="aborted"` (L691–701).

**Weight synchronization.** `abort_all_requests` closes the submission gate, waits up to 60 s for in-flight admissions, then calls `pause_generation` (L932–1003); the code comment states that weight updates must not proceed unless every in-flight request was aborted and old-weight caches were cleared (L989–991). In-flight requests are aborted, not paused mid-decode.

**Placement modes.** HYBRID puts rollout and training engines in one process sharing GPUs with weight sync; COLOCATED uses the same placement group in a separate process with no weight sync, for an LLM judge; STANDALONE uses separate GPUs and is off-policy (`replica.py` L54–66).

**How much of a step is generation.** **Result (single study):** in verl's own fully-async report, on 128 H20 GPUs with Qwen2.5-Math-7B and DAPO, the colocated synchronous baseline spends 177.85 s of a 356.30 s step in `gen` (about 50%), and 400 steps take 1d 16h 48m colocated against 17h 22m fully async at a 64:64 rollout:train split, a 2.35× wall-clock ratio; the last reported AIME-2024 acc/mean@1 was 0.2958 colocated against 0.3094 async, with maxima 0.3573 and 0.3521 (`fully_async_policy/README.md` L333–369). One configuration, one benchmark; do not carry the ratio to another model size or cluster.

---

## §8 Multi-turn agent loops

**Mechanism.** Multi-turn and tool-calling rollouts run in `verl/experimental/agent_loop/`. `AgentLoopBase.run` returns `AgentLoopOutput(prompt_ids, response_ids, response_mask)`, with mask 1 for a model-generated token and 0 for a tool-response token (`agent_loop.rst` L33–72). `align_response_metadata` gives boundary tokens inserted by the builder mask 0, assistant tokens mask 1 with their rollout log-probabilities, and tool or user tokens mask 0 with log-probability 0.0 (`continuous_token.py` L390–413). The policy loss consumes exactly this mask (`losses.py` L93–108), so tool output never receives a gradient and never enters the token denominator of `token-mean`.

`ToolAgentLoop` is a state machine PENDING → GENERATING → PROCESSING_TOOLS → TERMINATED (`tool_agent_loop.py` L48–52, L166–178). It terminates when the response mask reaches `response_length` or `max_assistant_turns`/`max_user_turns` is reached (L282–287). Defaults: `multi_turn.enable: False`, `max_parallel_calls: 1`, `max_tool_response_length: 256`, `tool_response_truncate_side: middle`, `format: hermes`, `tokenization_sanity_check_mode: strict` (`rollout.yaml` L183–230); `agent.num_workers: 8` and `default_agent_loop: single_turn_agent` (L246–249).

**Why tokens in and tokens out.** The documentation reports that token ids from applying the chat template to the final message list may differ from the concatenation of per-turn prompt and response ids, because tool parsers rewrite assistant content and decode-then-encode is not always invertible, and states that using `apply_chat_template` on the final chat history "make PPO training not even converged in single-turn" (`agent_loop.rst` L155–182). This is an experience report with no numbers, but the mechanism is checkable: a mismatch shifts the mask by the number of inserted or removed tokens, so gradients land on the wrong positions.

**Context growth.** Every turn appends to one token sequence (`agent_data.prompt_ids = merge_result.token_ids`, L273, L388); `ToolAgentLoop` does not reset or summarize earlier turns. A 30-turn episode therefore prefills the whole history each turn, and the `max_tokens` clamp of §7 step 3 shrinks the response budget turn by turn.

**What agentic forks add.** [[gigpo-verl-agent]] (GiGPO, arXiv:2505.10978v3) is built on verl and adds a step-wise interaction paradigm that avoids concatenating full histories, a memory module that chooses what history each step sees, and gym-style parallel group environments (App. A). Its algorithmic change is a second level of grouping: episode-level advantages `A^E(τ_i) = (R(τ_i) − mean(R))/F_norm(R)` as in GRPO (Eq. 3), plus step-level advantages built by collecting, across the N trajectories of one task, all actions taken from the same recurring environment state — the "anchor state" — and normalizing their discounted returns within that set (Eqs. 4–7). The two combine as `A(a_t^(i)) = A^E(τ_i) + ω·A^S(a_t^(i))` (Eq. 8), where `ω ≥ 0` weights the step-level term against the episode-level term; the paper sets `ω = 1` "with no further tuning" and the rollout group size `N = 8` for ALFWorld and WebShop, `N = 5` for search-augmented QA (§5.1). **Result (single study, results averaged over 3 random seeds, Table 1):** with Qwen2.5-1.5B-Instruct, GiGPO without std normalization surpasses GRPO "by 13.3% on ALFWorld and 10.6% on WebShop at 1.5B, and by 12.6% and 9.1%, respectively, at 7B"; on search-augmented QA it reaches 42.1% at 3B and 47.2% at 7B (§5.2–5.3, Tables 1–2). The percentages are differences in success rate, which the paper writes with a percent sign. The authors report the added cost as "only < 0.002% time cost", since no extra rollouts are drawn (§1). They also report that `F_norm = std` hurt on the harder subtasks and `F_norm = 1` helped there, matching the Dr. GRPO argument about difficulty bias (§5.2).

**Implication for a general-purpose model.** Step-level credit is the mechanism that gives different steps of one episode different advantages. Without it, every token of a 50-step episode carries the same episode-level advantage, so the gradient does not distinguish a step that changed the outcome from a step that did not.

---

## §9 Long sequences: sequence parallelism and the length clamps

**When the clamp is enough.** Rollout length is bounded three times before generation (§7 step 3), and `rollout.response_length` defaults to `data.max_response_length` (`rollout.yaml` L38–39). If your responses fit in the context window and the per-GPU activation memory of one packed micro-batch fits, no parallelism change is needed; raise `response_length` and lower `ppo_max_token_len_per_gpu`.

**When it is not.** Training on very long sequences is limited by activation memory per GPU, which grows with sequence length and cannot be reduced by data parallelism. Sequence parallelism partitions the sequence itself.

**Mechanism (DeepSpeed-Ulysses).** Each of P GPUs holds `N/P` tokens of the sequence and projects them to Q, K, V. Before attention, an all-to-all redistributes so that each GPU holds the full sequence for a disjoint subset of the attention heads; attention is computed per head; a second all-to-all returns the output to the sequence-partitioned layout for the MLP and normalization layers ([[deepspeed-ulysses]], arXiv:2309.14509v2 §1, §3.1).

**Communication.** For hidden size h, sequence length N, and P devices, the per-link volume is `4Nh/P`, i.e. O(N/P), because an all-to-all of aggregate size M costs M/P per link; Megatron-LM sequence parallelism uses two all-gathers and two reduce-scatters of size Nh per layer, costing `4Nh` per link, i.e. O(N) regardless of P (§3.2). **Result (single study):** the authors report training at 4× the sequence length of the compared systems with over a million tokens per sequence, over 10× communication reduction, up to 2.5× throughput, and sustained throughput above 175 TFlops/GPU, over 54% of hardware peak (§1). These are 2023 measurements on the authors' hardware.

**In verl.** `ulysses_sequence_parallel_size` is a field of the FSDP engine config, default 1 (`engine/fsdp.yaml` L44–45; `workers/config/engine.py` L292). The head-count constraint from the mechanism above is enforced in code (`verl/utils/ulysses.py` L337–341):

```python
def validate_ulysses_config(num_heads, ulysses_sequence_size):
    if ulysses_sequence_size > 1:
        assert num_heads % ulysses_sequence_size == 0, (
            f"num_heads ({num_heads}) must be divisible by ulysses sequence size({ulysses_sequence_size})"
        )
```

Inputs are padded to a multiple of the SP size and sliced per rank before the forward pass (`ulysses_pad_and_slice_inputs`, L302–334); outputs are gathered and unpadded after (L247–280). When `pad_to_length` is on, `pad_to_length_bucket` is rounded up internally to a multiple of `ulysses_sequence_parallel_size` (`engine/fsdp.yaml` L69–72). Sequence parallelism applies to the training engines only: the rollout engine is a separate process with its own parallelism, so raising the SP size does not let the policy generate longer responses.

**Length as a reward term.** The DAPO recipe shapes rather than truncates: `data.max_response_length: 20480` with `overlong_buffer.enable: True`, `len: 4096`, `penalty_factor: 1.0`, so the penalty rises linearly from 0 to 1.0 as the response exceeds `max_response_length − 4096` by 0 to 4096 tokens (docs/algo/dapo.md L137–165) ([[verl-dapo-recipe]]). A response at 17,408 tokens takes a penalty of (17408 − 16384)/4096 = 0.25 added to its reward as a negative term. The verl FAQ notes that the paper's Overlong Filtering is not implemented here because it overlaps with this shaping (L169–171).

---

## Negative samples and negative feedback

**Which kind of negative.** verl's RL path uses negatives mainly as **gradient** (kind 4 in the taxonomy of the negative-feedback chapter): a wrong response in a mixed group receives a negative advantage and the policy loss decreases the log-probability of its tokens. It also produces **negative marginal value** decisions (kind 1) when `filter_groups` discards groups whose metric values are all equal. No part of verl trains failures as content or under a control token; that is an SFT-stage choice.

**Where negatives come from and how they are labeled.** From the verifier selected by `data_source` (§5): a math grader, a test-suite execution, an exact-match QA check. Two false-negative sources are visible in the code. `NaiveRewardManager` assigns reward 0.0 and continues when `compute_score` exceeds `compute_score_timeout` (`naive.py` L129–143), so a slow grader converts a possibly correct answer into a negative example. The sandbox path returns 0 on compile error, syntax error, or timeout, which conflates an unusable answer with a wrong one. Neither event is counted in a default metric; add a counter.

**Mechanism.** For the softmax, `∂ log p_y/∂z_j = 1[j = y] − p_j`. Lowering `log p_y` raises the other logits in proportion to their current probability, so the removed mass goes mostly to tokens that were already likely. Repeated push-down on a low-probability token therefore concentrates the distribution rather than redistributing it, which is the mechanism behind entropy decline under heavy negative gradient.

**What the code does with them.**

- The lower clip removes the gradient for a negative-advantage token once `r < 1 − ε_low` (§2, worked example 1, row 1). This is the bound that DAPO declines to loosen, on the argument that a larger `ε_low` drives these probabilities to 0 and collapses the sampling space (arXiv:2503.14476v2 §3.1).
- The dual clip removes the gradient once the loss would exceed `c·|A|` with `c = 3.0` (row 4). Only negative-advantage tokens reach this branch (`torch.where(advantages < 0, …)`).
- Zero-variance groups contribute no gradient but their tokens remain in `response_mask` (L329), so under `token-mean` they still count in the global token denominator (L1170–1175): an all-wrong batch reduces the magnitude of every other token's update.
- `algorithm.filter_groups` (default `enable: False`) removes all-equal groups and resamples with `gen_batch_size` until `train_batch_size` qualified groups exist, bounded by `max_num_gen_batches` (docs/algo/dapo.md L69–86). **Result (single study):** verl's DAPO reproduction on Qwen2.5-32B reports AIME 2024 accuracy 52% with dynamic sampling and 50% without, one run each (docs/algo/dapo.md L36–37).
- The Pass@k estimator issues no negative advantages at all (§4).

**Diagnostics verl already emits.** `actor/pg_clipfrac` is the fraction of tokens on a clipped branch; `actor/pg_clipfrac_lower` is the fraction of tokens with `A < 0` whose loss exceeded the dual-clip bound (core_algos.py L1353, L1357–1359). The pair splits clipping by advantage sign: `pg_clipfrac` rising while `pg_clipfrac_lower` stays near zero points at the positive-advantage side and at `ε_high`; the reverse points at runaway ratios on negative-advantage tokens. `actor/entropy` is categorical entropy from logits, not the sampled-token proxy ([[entropy-logging-patterns]]).

**Size of effect.** verl reports no measurement of the effect of negative advantages on pass@k, calibration, or held-out tasks; the card's Verification section lists this under "not reported". Do not attribute a share of an improvement to negatives from the framework's evidence alone.

---

## Recipe

Framework defaults are not model recipes. Each row below says which it is.

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| verl default | n/a | RL | `clip_ratio`; `clip_ratio_low`; `clip_ratio_high` | 0.2; 0.2; 0.2 | verl@753aed3 `config/actor/actor.yaml` L36, L39, L42 | verified 2026-09-14 | no ablation reported |
| verl default | n/a | RL | `clip_ratio_c` (dual clip) | 3.0 (asserted > 1.0) | actor.yaml L83; core_algos.py L1323–1334 | verified 2026-09-14 | no ablation reported |
| verl default | n/a | RL | `loss_agg_mode`; `entropy_coeff`; `ppo_epochs` | `token-mean`; 0; 1 | actor.yaml L86, L93, L119 | verified 2026-09-14 | no ablation reported |
| verl default | n/a | RL | `use_kl_loss`; `kl_loss_coef`; `kl_loss_type` | false; 0.001; `low_var_kl` (k3) | actor.yaml L103, L113, L116 | verified 2026-09-14 | no ablation reported |
| verl default | n/a | RL | `use_kl_in_reward`; `kl_penalty`; `kl_ctrl` | False; `kl` (k1); fixed, `kl_coef` 0.001, horizon 10000, target 0.1 | `ppo_trainer.yaml` L98, L101, L109–119 | verified 2026-09-14 | no ablation reported |
| verl default | n/a | RL | `adv_estimator`; `rollout.n`; `norm_adv_by_std_in_grpo` | `gae`; 1; True | ppo_trainer.yaml L74, L77; `rollout/rollout.yaml` L127 | verified 2026-09-14 | docs: set `grpo` and n > 1 for GRPO (grpo.md L29, L39) |
| verl default | n/a | RL | `val_before_train`; `val_only`; `test_freq` | True; False; −1 (disabled) | ppo_trainer.yaml L195, L198, L201 | verified 2026-09-14 | no ablation reported |
| verl default | n/a | RL | `val_kwargs.temperature`; `.n`; `.do_sample` | 0; 1; False | rollout.yaml L164, L167, L170 | verified 2026-09-14 | no ablation reported |
| verl default | n/a | RL | `ulysses_sequence_parallel_size` | 1 | `config/engine/fsdp.yaml` L45; `workers/config/engine.py` L292 | verified 2026-09-14 | no ablation reported |
| verl default | n/a | RL | `reward_fn_key`; `reward_manager.name`; `reward.num_workers` | `data_source`; `naive`; 8 | `config/data/legacy_data.yaml` L29–30; `config/reward/reward.yaml` | verified 2026-09-14 | no ablation reported |
| verl docs, DAPO recipe | n/a | RL | `clip_ratio_low`; `clip_ratio_high` | 0.2; 0.28 | docs/algo/dapo.md L52–57 | verified 2026-09-14 | none in docs |
| verl docs, DAPO recipe | n/a | RL | `gen_batch_size`; `train_batch_size`; `filter_groups.metric`; `max_num_gen_batches` | 1536; 512; `acc`; 10 | docs/algo/dapo.md L73–82 | verified 2026-09-14 | none in docs |
| verl docs, DAPO recipe | n/a | RL | `max_response_length`; `overlong_buffer.len`; `penalty_factor` | 20480; 4096; 1.0 | docs/algo/dapo.md L141–149 | verified 2026-09-14 | none in docs |
| verl docs, Dr. GRPO | n/a | RL | `loss_agg_mode`; `loss_scale_factor`; `use_kl_loss`; `norm_adv_by_std_in_grpo` | `seq-mean-token-sum-norm`; a constant integer, e.g. max response length (marked optional); False; False | docs/algo/grpo.md L59–62 | verified 2026-09-14 | no ablation reported in docs |
| DAPO reproduction on verl, Qwen2.5-32B base | 32B | RL | dynamic sampling on vs off | AIME 2024 acc. 52% vs 50% | docs/algo/dapo.md L36–37 (commit 4f80e4, 16×8×H800) | verified 2026-09-14 | one run per row; W&B record linked |
| Qwen2.5-Math-7B (verl fully_async experiment) | 7B | RL | algorithm; max response; `rollout.n`; hardware | DAPO; 28K tokens; 16; 128 H20 (64:64) | `fully_async_policy/README.md` L333–340, L361 | verified 2026-09-14 | 32/64/128-GPU comparison L353–361 |
| Qwen2.5-Math-7B (verl fully_async experiment) | 7B | RL | `staleness_threshold`; `partial_rollout` | 0.5; True | README.md L351–352 | verified 2026-09-14 | staleness ablation 0/0.1/0.3/0.5, L380–396 |
| Nemotron-Research-Reasoning-Qwen-1.5B | 1.5B | RL | framework; algorithm; ε_low/ε_high | verl; GRPO + decoupled clip + dynamic sampling; 0.2/0.4 | arXiv:2505.24864v1 §3.2 | verified 2026-09-14 | §2.3 "helps retain entropy"; no numbers |
| Nemotron-Research-Reasoning-Qwen-1.5B | 1.5B | RL | data (prompts, five domains) | 136K: math 40k, code 24k, STEM 25k, logic 37k, instruction following 10k | arXiv:2505.24864v1 §3.1, Table 4 | verified 2026-09-14 | no ablation of the mix reported |
| Nemotron-Research-Reasoning-Qwen-1.5B | 1.5B | eval-gate | validation blend | AIME2024, Codeforces, GPQA-diamond, IFEval, graph_color | arXiv:2505.24864v1 App. E | verified 2026-09-14 | final checkpoint selection rule not reported |
| Nemotron-Research-Reasoning-Qwen-1.5B | 1.5B | RL | KL coefficient β | not reported | arXiv:2505.24864v1 §2.3.1 | not reported (body, App. D–F, HF model card checked) | no ablation reported |
| GiGPO (verl-agent), Qwen2.5-7B-Instruct | 7B | RL | rollout group size N; weighting ω; ALFWorld gain over GRPO | 8; 1 (no tuning reported); 12.6% | arXiv:2505.10978v3 §5.1–5.2, Table 1 | verified 2026-09-14 | 3 seeds; ALFWorld and WebShop only |

**Starting point for a small general-purpose run.** Every number here comes from a `verified` row above. Use GRPO by setting `algorithm.adv_estimator: grpo` and `actor_rollout_ref.rollout.n: 16` (ProRL's value at 1.5B, §3.2); keep `clip_ratio_low: 0.2` and raise `clip_ratio_high: 0.28` (verl docs DAPO recipe); keep `clip_ratio_c: 3.0` and `loss_agg_mode: token-mean` (verl defaults); turn the reference term on with `actor.use_kl_loss: True`, `kl_loss_coef: 0.001`, `kl_loss_type: low_var_kl` (verl defaults, the placement the GRPO docs recommend); enable `algorithm.filter_groups` with `metric: acc`, `gen_batch_size: 1536`, `train_batch_size: 512`, `max_num_gen_batches: 10` (verl docs DAPO recipe); and set `trainer.test_freq` to a positive value with `val_kwargs.n: 8`, `do_sample: True` and a validation file per domain, trained and untrained. Conditions: the DAPO values were read at 32B on math with 16×8×H800; the ProRL values at 1.5B on a five-domain 136K set with 4×8 H100-80GB and about 16k GPU hours. Neither was tuned for your model size, and no row above carries an ablation that transfers.

---

## Generalization lens

**(a) What increases breadth.**

- Routing several `data_source` values to several verifiers in one run is the mechanism that puts more than one domain in the gradient (§5). **Result (single study):** the ProRL 1.5B run over five domains improved on every reported axis against its starting checkpoint — pass@1 for math 44.45 → 60.14, code 23.08 → 37.49 and GPQA-Diamond 15.86 → 41.78, and average verifier reward for IFEval 44.05 → 66.02 and Reasoning Gym 4.24 → 59.06 (§3.4, Tables 1–3) — and exceeded DeepCoder-1.5B on the code average and DeepScaleR-1.5B on the math average ([[prorl]]).
- Keeping the reference term and resetting the reference policy when validation stagnates is the recipe that report attaches to continued gains over 2k steps (§2.3.1, §3.3). No ablation isolates the reset from the KL term or from task diversity.
- Step-level credit in multi-turn settings adds a gradient signal that episode-level scalars do not carry ([[gigpo-verl-agent]], §5.2).

**(b) What causes narrowing.**

- A single-verifier run has no term that prefers general behaviour. The only counterweight in verl is the KL term, and both of its placements default to off (§3).
- Zero-variance groups and the default `rollout.n: 1` remove negative signal entirely or make every advantage the raw reward (§4), which changes what the run optimizes without changing any curve you are looking at.
- **Result (single study):** even in a multi-domain long RL run, tasks in the "Diminish" regime — mostly math — show pass@1 rising while pass@128 often declines, and the size of the boundary gain is negatively correlated with the starting checkpoint's pass@128 ([[prorl]] §4.1–4.2, Figs. 3–4). **Replicated in direction:** [[rlvr-beyond-base-model]] reports base models matching or exceeding RLVR models at large k on the benchmarks it tested, and interprets RLVR as raising the probability of paths the base model could already sample.
- Reward shaping terms narrow in their own way: the overlong penalty (§9) is a gradient against long responses, which is a gradient against the behaviour that long-context tasks need.

**(c) How to measure it at this stage.**

- Per-`data_source` validation metrics at a positive `test_freq`, with at least one `data_source` that is not in `train_files` (§6).
- `best@n` at `val_kwargs.n ≥ 8` as a coverage proxy alongside `mean@n`; a rising `mean@8` with a falling `best@8` is the narrowing signature described above. The bootstrap at n = N resamples the same N responses, so compare trends across steps, not the level against an external pass@k number.
- Mean k3 KL to the reference on a fixed general prompt set, logged per validation step (§6).
- `actor/entropy`, and `pg_clipfrac` against `pg_clipfrac_lower`, as early indicators that arrive before the validation metrics move.
- Known measurement errors: validation prompts that also appear in `train_files` make every metric optimistic; `temperature: 0` in `val_kwargs` makes `best@n` unavailable and `mean@1` a point estimate with no spread; grader timeouts register as zeros in both training and validation (§5, negatives section).

---

## Common mistakes and how to detect them

| Mistake | Observable symptom | Check |
|---|---|---|
| GRPO configured with `rollout.n` left at 1 | `actor/pg_clipfrac_lower` stays at 0; no response ever gets a negative advantage with a 0/1 reward | Print the resolved config for `actor_rollout_ref.rollout.n`; confirm group sizes at `core_algos.py` L315–317 |
| Expecting a KL term without setting either flag | `kl_loss` and `actor/reward_kl_penalty` absent from the metric dump; policy drifts with nothing opposing it | Check `actor.use_kl_loss` and `algorithm.use_kl_in_reward`; both default to off |
| Reading `actor/ppo_kl` as drift from the reference | Curve near zero while the model's general behaviour changes | `actor/ppo_kl` is k1 between the old and current policy (core_algos.py L1340), not against the reference |
| "Dr. GRPO" set as `loss_agg_mode` only, with the other three settings left at their defaults | Response length still grows during training, because `norm_adv_by_std_in_grpo` is still True | The documented combination is `seq-mean-token-sum-norm` + `loss_scale_factor` set to a constant + `norm_adv_by_std_in_grpo: False` + `use_kl_loss: False` (grpo.md L59–62) |
| `seq-mean-token-sum-norm` with `loss_scale_factor` unset | Loss and gradient-norm curves shift when the batch's longest response changes, with no change to the data | The divisor falls back to `loss_mask.shape[-1]` (core_algos.py L1190–1193); set `loss_scale_factor` to a fixed integer |
| New `data_source` value with no route | Run aborts with `NotImplementedError: Reward function is not implemented for data_source=…` | reward_score/__init__.py L107; register the value in a custom `compute_score` |
| Grader timeouts silently scored 0 | Accuracy for one domain drops with no change to the model | Log the `[NaiveRewardManager] compute_score exceeded` line as a counter (naive.py L138–142) |
| `test_freq` left at −1 | Exactly two validation points in the whole run | `ppo_trainer.yaml` L201; the gate at ray_trainer.py L1727–1728 |
| `val_kwargs` left at `temperature: 0, n: 1` | No `best@n` or `std@n` keys in the metric dump | `metric_utils.py` L989 computes them only when `n_resps > 1` |
| Long prompts silently shrinking responses | Truncated completions with no warning; reward drops for long-prompt rows | The `max_tokens` clamp at `vllm_async_server.py` L615–619 logs nothing; log `max_tokens` per request |
| Raising `ulysses_sequence_parallel_size` to a non-divisor of the head count | `AssertionError: num_heads (…) must be divisible by ulysses sequence size(…)` | `verl/utils/ulysses.py` L337–341 |
| Multi-turn training with the chat template re-applied at the end | Masks misaligned by the number of rewritten tokens; training fails to converge | Use the token-in/token-out path; `tokenization_sanity_check_mode: strict` (rollout.yaml L227) |

---

## Check your understanding

1. A run shows `actor/pg_clipfrac` rising toward 1 while `actor/pg_clipfrac_lower` stays near 0. Which branch of the loss is selecting, which sign of advantage is involved, and which single configuration value would you change first? Explain why that change affects this metric and not the other.
2. The default `token-mean` aggregation divides by the global valid-token count. Explain how an all-wrong group with zero advantage changes the size of the update applied to a different, correctly-scored group in the same batch, and why `filter_groups` addresses that.
3. Two teams report the same `kl` curve near 0.0. One used `kl_penalty: kl`, the other `kl_loss_type: low_var_kl`. Explain why the first team cannot conclude their policy is near the reference and the second can, using the per-token values for δ = ±0.1.
4. A validation dump contains `val-core/math_dapo/acc/mean@8` but no `best@8` key for any data source. Give the configuration cause and explain the code path that produces the absence.
5. You add a second domain by concatenating a parquet file into `data.train_files`. Training reward rises but the new domain's validation accuracy stays flat. Name three distinct causes consistent with that pattern — one in reward routing, one in the advantage estimator, one in the mixture — and the log line that would separate them.
6. Explain why raising `ulysses_sequence_parallel_size` lets you train on longer sequences but does not let the policy generate longer responses, referring to the process boundary between the training engine and the rollout engine.
7. GiGPO's step-level advantage requires environment states to recur across trajectories. Explain why that requirement is satisfiable in ALFWorld and WebShop but not in single-turn math RL, and what that implies about which verl-based stacks can adopt the method.
8. The Pass@k advantage estimator issues no negative advantages. Using the softmax gradient `∂ log p_y/∂z_j = 1[j = y] − p_j`, explain what that changes about where probability mass moves, and predict the effect on entropy relative to standard GRPO.

---

## Connections

- **ch-54 — Rollout Infrastructure: Off-Policy Data, Asynchronous RL, and Agentic Environments** — the architectural arguments (staleness, partial rollout, environment-in-the-loop) that §7 and §8 show implemented in one repository.
- **ch-56 — OpenRLHF Internals: PPO, DPO, Ray Orchestration, and Forgetting Controls** — the same algebra in a different code base, with KL placement and controller defaults that differ from the ones in §3.
- **ch-38 — KL-Controlled RLHF: PPO, InstructGPT, and the Alignment Tax** — the clipped surrogate and the KL-to-reference term that §2 and §3 implement.
- **ch-40 — Group-Baseline RL: RLOO, GRPO, Dr. GRPO, DAPO, and GSPO** — the algorithms that §4 shows as registry entries and configuration flags.
- **ch-43 — Entropy, Output Diversity, and KL Control in RL** — the failure mode the diagnostics in the negatives section are meant to catch early.
- **ch-53 — Lab: Evaluation Harness with a Held-Out Suite, Forgetting Report, and Perturbation Robustness** — the out-of-loop counterpart to the in-loop validation of §6.
- **ch-57 — TRL Internals: SFT, DPO, GRPO, and Distillation Trainers** — the third stack in the comparison.
- **ch-58 — Choosing and Instrumenting a Post-Training Stack to Measure and Protect General Capability** — where the per-framework readings become a selection and instrumentation decision.

---

## Sources

- [[verl-ppo-loss]] — `compute_policy_loss_vanilla` and `agg_loss` at commit 753aed3: the loss algebra of §2, the clip and dual-clip defaults, the aggregation modes, and the entropy/KL composition in `losses.py`.
- [[verl-grpo]] — `compute_grpo_outcome_advantage`, the Dr. GRPO switches, and the Pass@k estimator used in §4, with the worked G = 4 example.
- [[verl-rollout]] — `HFRollout`, the async vLLM server, weight-sync behaviour, placement modes, the agent loop and response masks, and the fully-async timing table used in §7 and §8.
- [[entropy-logging-patterns]] — the metric names and KL estimator formulas of §3, and the cross-framework defaults table.
- [[verl-dapo-recipe]] — verl's DAPO documentation: Clip-Higher values, dynamic sampling configuration, loss aggregation modes, the overlong reward shaping used in §9, and the reproduction table.
- [[gigpo-verl-agent]] — the two-level advantage, anchor-state grouping, verl-agent's step-wise interaction paradigm, and the ALFWorld/WebShop/QA numbers of §8.
- [[deepspeed-ulysses]] — the all-to-all sequence-parallel mechanism, the O(N/P) communication analysis, and the reported throughput numbers of §9.
- [[prorl]] — a multi-domain verl run: five-domain data mix, KL-in-loss with reference resets, the validation blend, and the breadth and narrowing results in the generalization lens.
- [[dr-grpo]] — the length and difficulty biases that the `loss_agg_mode` and `norm_adv_by_std_in_grpo` settings address.
- [[grpo]] — the group-relative advantage that §4 implements.
- [[ppo]] — the clipped surrogate of §2 with symmetric ε.
- [[kl-control-rlhf]] — the KL-regularized objective and the estimator background for §3.
- [[async-rollout]] — the asynchronous rollout design that the server in §7 realizes.
- [[rlvr-beyond-base-model]] — the pass@k critique that motivates the coverage metrics of §6 and the Pass@k estimator of §4.
