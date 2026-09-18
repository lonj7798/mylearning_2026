<!-- chapter: ch-58
     track: infra
     kind: content
     title: Choosing and Instrumenting a Post-Training Stack to Measure and Protect General Capability
     deps: [ch-57]
     sources: [[verl-grpo]], [[verl-ppo-loss]], [[verl-rollout]], [[openrlhf-ppo]], [[openrlhf-dpo]], [[trl-grpo]], [[trl-ppo]], [[trl-online-dpo]], [[entropy-logging-patterns]], [[async-rollout]], [[areal-async-rl]], [[skyrl-agent]], [[gigpo-verl-agent]], [[agent-lightning]], [[intellect-3]], [[thinkingmachines-defeating-nondeterminism]], [[rollout-training-mismatch-tis]]
     figures: figures/framework-compare.html
     revised: 2026-09 (generality revision)
-->

# Chapter 58 — Choosing and Instrumenting a Post-Training Stack to Measure and Protect General Capability

> **Core insight.** The clipped policy-gradient algebra is close to identical in verl, OpenRLHF and TRL, so a stack choice is not an algorithm choice. What differs is the operating range each stack supports (how many GPUs, how stale a rollout may be, whether an environment runs inside the loop) and the instrumentation: whether the stack can tell you, during the run, that general capability is falling. The instrumentation gap is measurable. AReaL's ablation at maximum staleness η = 4 without its decoupled objective scores AIME24 23.3 and MATH-500 66.9 against a synchronous oracle of 42.0 and 89.2; with the decoupled objective the same staleness gives 42.2 and 89.5 ([[areal-async-rl]] Table 2). A run that logged only training reward would not have separated those two configurations.
>
> **Guideline.** When the run must produce a generally capable model, require three things of the stack before any throughput comparison: periodic in-loop evaluation on held-out tasks the reward does not target, a logged rollout-versus-training log-probability diagnostic, and a reward layer that can route more than one domain. When those three are present in more than one candidate, choose by operating range: single node and no environment, use TRL; multi-node with a Ray-managed actor/critic split, use OpenRLHF; multi-node with Megatron parallelism, MoE routing replay, or a tool agent loop, use verl. When the run puts an environment in the loop for thousands of concurrent episodes, choose an agentic stack built for it (SkyRL-Agent, verl-agent, AReaL, prime-rl) rather than extending a single-turn trainer, because those systems report their scheduling gains with measurements ([[skyrl-agent]] §3.2; [[areal-async-rl]] §7.2).

---

## Why this chapter matters for a general-purpose model

This chapter sits at the end of the RL-infrastructure track, after [[verl-ppo-loss]], [[openrlhf-ppo]] and [[trl-grpo]] have been read one at a time. Its question is the one that comes before a run starts: which stack, and configured how, so that the run can be trusted.

The reason generality makes this a hard question is that RL optimizes the reward it is given, and the reward is almost always narrower than the capability you want to keep. A math verifier says nothing about instruction following, tool use, refusal behaviour, or output diversity. If the stack cannot evaluate anything except the training reward, then capability loss outside the reward's domain is invisible until the run ends. Three separate failure sources make that invisibility likely:

1. **Reward-directed narrowing.** The policy improves on the reward and may lose ability elsewhere. Detection requires evaluation on tasks the reward does not target.
2. **Off-policy drift.** Rollouts are produced by an inference engine and gradients by a training backend. The two disagree on token probabilities even under shared parameters, which turns nominally on-policy RL into off-policy RL ([[thinkingmachines-defeating-nondeterminism]], RL section). Uncorrected, this has produced reward collapse and entropy collapse in published runs.
3. **Unreliable measurement.** If generation is not reproducible, a before-and-after comparison of breadth mixes the model change with sampling noise. Sampling 1000 completions of one prompt at temperature 0 from Qwen3-235B-A22B-Instruct-2507 produced 80 distinct completions under default vLLM kernels, and one completion repeated across all 1000 under batch-invariant kernels ([[thinkingmachines-defeating-nondeterminism]], "Experiments": 1000 completions, 1000 tokens each).

The chapter therefore treats stack selection as an instrumentation problem first (§2), a capability matrix second (§3), and a throughput question third (§4).

---

## §1 What "the stack" is, and what is common to all of them

**Definition.** A post-training stack is four layers: a *trainer* that computes gradients, a *rollout engine* that generates samples, a *reward or environment layer* that scores them, and an *evaluation layer* that measures the model on tasks separate from training. The first three are what framework READMEs describe. The fourth is the one that decides whether generality can be observed.

**What is shared.** At the pinned commits read for this chapter — verl 753aed3 (2026-09-14), OpenRLHF 64c1cc4 (2026-04-19) with `main` b117b2b (2026-09-14) checked, TRL a08e713 (2026-04-21) with `main` a04ffd3 (2026-09-14) checked — the surrogate objective is the same object in all three. Each minimizes a clipped ratio times an advantage:

```
L = − min( r_t · A_t ,  clip(r_t, 1 − ε_low, 1 + ε_high) · A_t )
```

- `r_t` = exp(log π_θ(a_t) − log π_old(a_t)), the per-token probability ratio between the current policy and the policy whose log-probabilities were recorded for the batch.
- `A_t` = the advantage assigned to token t.
- `ε_low`, `ε_high` = lower and upper clip bounds, separately configurable in all three ([[verl-ppo-loss]]; [[openrlhf-ppo]] loss.py L136–148; [[trl-grpo]] grpo_trainer.py L2507–2515).

The group-relative advantage is also the same construction, with different constants. verl divides by `std + 1e-6` (core_algos.py L272, L326), TRL by `std + 1e-4` (grpo_trainer.py L2127–2149), OpenRLHF's `group_norm` by `std + 1e-9` (experience_maker.py L264–270). A group whose rewards are all equal gets advantage 0 in all three.

**Implication.** Do not choose a stack for its loss. Two runs that differ only in framework can still differ in outcome, but the cause will be in the rollout path, the KL placement, the aggregation denominator, or the evaluation schedule — not in the surrogate.

---

## §2 Generality instrumentation: four hooks to check before anything else

### §2.1 In-loop evaluation on tasks the reward does not target

**The problem, stated measurably.** RL runs for hundreds of steps. If the only logged scalars are reward, entropy, KL and clip fraction, a drop in an untargeted capability is detectable only after the run. A stack supports generality measurement when it can (a) run generation on a held-out set on a schedule, (b) score it with a different scorer than the training reward, and (c) select checkpoints on that score.

**What each stack provides.**

- **verl.** `trainer.val_before_train` defaults to `True`, `trainer.test_freq` defaults to `-1` (no periodic validation), `trainer.log_val_generations` defaults to `0`, and `trainer.validation_data_dir` defaults to `null` (`verl/trainer/config/ppo_trainer.yaml` L154–201 at 753aed3). A `trainer.checkpoint_callback_class` hook (default `null`, same file) runs after each checkpoint save, which is where an external evaluation job can be attached. So verl measures once before training by default and never again unless `test_freq` is set.
- **OpenRLHF.** `--eval.steps` defaults to `-1`, `--eval.dataset` to `None`, `--eval.temperature` to 0.6 and `--eval.n_samples_per_prompt` to 4 (`openrlhf/cli/train_ppo_ray.py` L286, L535–540 at 64c1cc4). There is a constraint worth knowing before planning: `if args.eval.dataset: assert args.reward.remote_url` (L672–673), so in-loop evaluation requires a remote reward-model service. `--ckpt.best_metric_key` selects the checkpoint by an eval metric such as `eval_default_pass1` (L297–300).
- **TRL.** `GRPOConfig` inherits `eval_strategy` and `per_device_eval_batch_size` from `transformers.TrainingArguments`, adds `num_generations_eval` (default `None`, falling back to `num_generations`), and validates that the global eval batch size is divisible by the number of eval generations (`grpo_config.py` L59, L399, L919–928 at a08e713; same check at L1105–1115 on `main`).

**Evidence that this is used in practice.** The INTELLECT-3 report plots model performance at 15-step intervals on AIME25, AIME24, LiveCodeBench, HLE and GPQA during the RL run, and states that the scores were still rising when the run ended (§3.3, "Online Evaluation"; Figure 9) ([[intellect-3]]). That is five benchmarks, of which at least two (HLE, GPQA) are outside the domain of most of the RL reward mixture. **Result (single study)**, one run, no seeds reported.

**Conditions and limits.** None of the three general frameworks ships a broad evaluation suite. Each gives a hook; the suite is the user's. The cost is real: generation for a five-benchmark suite at a useful number of samples per prompt competes with rollout for the same inference capacity.

**Implication for a general-purpose model.** Treat `test_freq` / `--eval.steps` / `eval_strategy` as required configuration, not optional, and point them at a set that includes at least one task family absent from the reward.

### §2.2 Reproducible evaluation: the rollout-versus-training log-probability gap

**Definition.** The sampler (vLLM, SGLang) and the learner (FSDP, Megatron, DeepSpeed) compute the probability of the same token under the same parameters with different kernels, different parallelism and sometimes different precision. The resulting difference is the rollout–training mismatch.

**The measured problem.** On a DAPO run with Qwen2.5-32B, the maximum per-token probability difference between the vLLM sampler and the FSDP learner reached 1.0 — tokens where the sampler assigned probability 1 and the learner 0 ([[rollout-training-mismatch-tis]], Fig. 1). On the smaller GSM8K example the maximum difference was about 0.4. The gap persisted after two vLLM patches (returning the probabilities actually used for sampling, and an fp32 `lm_head`).

**Mechanism.** The policy-gradient estimator assumes the sampling distribution and the differentiated distribution are the same. When they are not, the REINFORCE term becomes E_{a∼π_sampler}[R(a)·∇ log π_learner(a)], which is off-policy. Truncated importance sampling (TIS) restores the correction with a bounded weight:

```
E_{a∼π_sampler(θ)} [ min( π_learner(a,θ) / π_sampler(a,θ), C ) · R(a) · ∇_θ log π_learner(a,θ) ]
```

- `π_sampler(a,θ)`, `π_learner(a,θ)` = the probability assigned to action `a` by the inference engine and the training backend under the same parameters θ.
- `R(a)` = the reward of the sample.
- `C` = the truncation bound.

**Worked numeric example (variance, from the blog's callout).** Take one token whose learner/sampler ratio is 16. Gradient noise for that token is scaled by the square of the applied weight: untruncated importance sampling scales it by 16² = 256; TIS with C = 2 scales it by 2² = 4; TIS with C = 8 scales it by 8² = 64. The truncation trades a bias (the weight is capped below its true value) for a variance reduction of 256/4 = 64× at C = 2.

**Evidence.** A DAPO-32B run whose only difference was the truncated ratio showed higher downstream performance across the 250 steps that were run ([[rollout-training-mismatch-tis]], Fig. 1; the OPT 2025 workshop version names the benchmark as AIME 2024). On GSM8K with Qwen2.5-0.5B, INT8 rollouts degraded PPO relative to BF16 rollouts, and TIS brought the INT8 run back to similar accuracy (Fig. 2). Where the gap was already small (DeepSeek-R1-Distill-Qwen-1.5B, BF16 both sides), TIS gave no gain (Fig. 3). **Result (single study)**, practitioner evidence with released scripts.

The Thinking Machines report reaches the same place from the kernel side: in an RLVR setup on Bigmath with the policy initialized from Qwen2.5-VL-Instruct-8B and a 4096-token maximum rollout length, reward collapsed partway through training without off-policy correction, with a loss spike around step 318; with importance weighting, sampler–trainer KL stayed around 0.001 with occasional spikes; with bitwise-identical sampler and trainer kernels, KL stayed flat at 0 ([[thinkingmachines-defeating-nondeterminism]], RL section). **Replicated** across two independent sources for the direction of the effect.

**What each stack exposes.**

- **verl:** an `algorithm.rollout_correction` config group with `rollout_is: null` (disabled by default), `rollout_is_threshold: 2.0`, a `bypass_mode` switch between a three-policy decoupled formulation and a two-policy one, and rejection-sampling variants (`verl/trainer/config/algorithm/rollout_correction.yaml` at 753aed3). `rollout.calculate_log_probs` defaults to `True` (`rollout.yaml` L237).
- **OpenRLHF:** `--algo.advantage.is_correction_enable`, off by default, band [0.5, 5.0], types `tis`, `icepop`, `seq-mask-tis`, and a `vllm_kl` metric equal to the masked mean of (rollout log-prob − trainer log-prob) ([[openrlhf-ppo]] loss.py L150–173; `train_ppo_ray.py` L257–271). On `main` b117b2b the flags become `is_correction_level {off, token, seq}` with `mode {mask, clip}` and `gating {ratio, binary_kl, tv}`.
- **TRL:** in `GRPOTrainer`, IS correction is on by default when `use_vllm=True`, mode `sequence_mask`, cap 3.0 (`grpo_config.py` L792–819 at a08e713). In the experimental async trainer, the model `dtype` defaults to `"float32"` and the trainer logs a warning at start when the vLLM server's dtype does not match, with the docstring naming the training–inference mismatch as the reason (`trl/experimental/async_grpo/async_grpo_config.py` L37–44 at a04ffd3).

**Implication for a general-purpose model.** Two checkpoints can only be compared on breadth if the generation path is the same and the mismatch is either corrected or measured. Log the mismatch metric from step 1 (`vllm_kl`, or the verl rollout-correction ratio), and record sampler dtype, tensor-parallel size and sequence-parallel size next to every evaluation number, because the factor analysis shows all three change the mismatch ([[rollout-training-mismatch-tis]], Figs. 7–8).

### §2.3 Multi-domain reward routing

A generally capable model is trained against more than one scorer. The stack must be able to attach several and combine them without hand-written glue.

- **TRL** takes `reward_funcs` as a list, with `reward_weights` (default `None`, meaning weight 1.0 each) and `multi_objective_aggregation` (default `"sum_then_normalize"`, which sums the weighted rewards before applying `scale_rewards`) (`grpo_config.py` L205–212, L678–690 at a08e713). This is the most direct multi-domain path of the three.
- **OpenRLHF** routes through `--reward.remote_url`, a service boundary. The evaluation constraint in §2.1 makes the remote reward model the required path when in-loop evaluation is used.
- **verl** carries a `reward` config group and a legacy reward implementation group (`ppo_trainer.yaml` defaults list L34–39 at 753aed3); the routing detail is the subject of ch-55.
- **prime-rl** (the INTELLECT-3 stack) makes the environment the unit instead of the reward function: the orchestrator "utilizes verifiers environments to abstract multi-turn rollout generation and scoring, allowing any environment on the Environments Hub to plug into the training loop", and the report states the same environment interface covers synthetic data generation, supervised fine-tuning, RL and evaluation ([[intellect-3]] §2.1, §2.1.1). That property is what makes the training mixture and the evaluation suite comparable objects.

**Implication.** When the plan is a multi-domain reward mixture, prefer a stack where the domain identity survives into logging, so per-domain reward can be plotted separately. A single averaged reward hides a domain that is regressing while another improves.

### §2.4 Forgetting controls: KL placement, LoRA, checkpoint hooks

The three knobs the frameworks actually offer against capability loss are the KL term to a reference policy, low-rank adaptation, and checkpoint selection. Their defaults differ, and the defaults are what most runs use.

| Control | verl @753aed3 | OpenRLHF @64c1cc4 | TRL @a08e713 |
|---|---|---|---|
| KL to reference, in reward | `algorithm.use_kl_in_reward`, off by default | default path, `FixedKLController` `init_coef` 0.01; `AdaptiveKLController` only when `--algo.kl.target` is set (default `None`) | PPO path only: `kl_coef` 0.05 subtracted from the reward ([[trl-ppo]]) |
| KL to reference, in loss | `actor.use_kl_loss` (`false` by default; docs say `True` for GRPO), `kl_loss_coef` 0.001, `kl_loss_type` `low_var_kl` (`actor.yaml` L103–116) | `--algo.kl.use_loss`, adds `kl_loss * kl_ctl` to the actor loss (ppo_actor.py L296–314) | GRPO: β·k3 added to the loss, β default **0.0**, and with β = 0 no reference model is created (grpo_trainer.py L2493–2497, L650–653) |
| Entropy term in loss | `entropy_coeff: 0` (`actor.yaml` L93) | `--actor.entropy_coef` unset → logging only | no entropy term |
| LoRA | LoRA adapters supported in the rollout path (`lora_as_adapter`, adapter-only weight sync at sleep level 1) ([[verl-rollout]]) | `--ds.lora.rank` default 0, `--ds.lora.alpha` 16, `target_modules` `all-linear` (train_ppo_ray.py L363–368) | PEFT integration through the `transformers` trainer |
| Pretraining-loss mixing during RL | not present in the PPO trainer config read here | no such flag in `cli/train_ppo_ray.py` at 64c1cc4 | not present in `GRPOConfig` |
| Checkpoint selection / averaging / merging hook | `trainer.checkpoint_callback_class` (default `null`); no merge utility in the trainer | `--ckpt.best_metric_key`; `openrlhf/cli/lora_combiner.py` is a separate command, not called by the loop | `transformers` callbacks, `load_best_model_at_end`; `trl/experimental/merge_model_callback.py` at a04ffd3 merges the checkpoint with a target model through mergekit |

Two observations follow. First, the default configuration for GRPO-style RL carries **no KL term** in TRL (β = 0.0) and none in verl (`use_kl_loss` and `use_kl_in_reward` both off); the reference model is optional. The control most often named as the forgetting control is off unless it is switched on.

Second, exactly one of the three ships a merging hook that the training loop calls. TRL's `MergeModelCallback` (`trl/experimental/merge_model_callback.py` at a04ffd3) is a `transformers.TrainerCallback`: it merges the saved checkpoint with a target model — by default the model id read from the trained model's config — using mergekit, with `MergeConfig` methods `linear` (default; policy weight 0.5, target weight 0.5), `ties`, `dare_ties` and `slerp`. `merge_at_every_checkpoint` defaults to `False`, so by default the merge runs once in `on_train_end` rather than at each save, and the `mergekit` extra must be installed or the callback raises `ImportError` at construction. verl has no equivalent in the files read here, and OpenRLHF's `openrlhf/cli/lora_combiner.py` at b117b2b is a separate command-line utility. What merging accomplishes, and with which weights, is the subject of ch-58a.

---

## §3 The capability matrix, pinned to dated commits

Each cell below was read at the commit named in its column header, or is marked "not checked". "Not present" means the flag or module does not exist at that commit in the file read; it is not a claim about the whole repository or about later versions.

| # | Dimension | verl @753aed3 (2026-09-14) | OpenRLHF @64c1cc4 (2026-04-19), `main` b117b2b | TRL @a08e713 (2026-04-21), `main` a04ffd3 |
|---|---|---|---|---|
| 1 | Clipped surrogate with asymmetric bounds | yes ([[verl-ppo-loss]]) | yes, `clip_eps_low`/`clip_eps_high`, optional `dual_clip` (default `None`) | yes, `epsilon`/`epsilon_high`, optional `delta` |
| 2 | GRPO family | `@register_adv_est` with 14 registered estimators, incl. Pass@k and GDPO ([[verl-grpo]]) | `group_norm`, `dr_grpo`, `rloo`, `reinforce_baseline`, `gae`, `reinforce` | eight `loss_type` values, default `dapo` ([[trl-grpo]]) |
| 3 | Actor–critic PPO | yes (`adv_estimator=gae`, critic enabled) | yes, Ray actor/critic split | **removed from mainline**: PR #7020 deleted `trl/experimental/ppo/` on 2026-09-04, released in v1.13.0 ([[trl-ppo]]) |
| 4 | Mainline trainer set | RL trainers | at b117b2b `openrlhf/trainer/` holds `ppo_trainer.py`, `ppo_trainer_async.py`, `dpo_trainer.py`, `rm_trainer.py`, `sft_trainer.py` | at a04ffd3 `trl/trainer/` holds DPO, GRPO, KTO, RLOO, SFT, reward, distillation |
| 5 | Asynchronous rollout | `fully_async_policy` with partial rollout ([[verl-rollout]]) | `PPOTrainerAsync`, queue size default 1, optional partial rollout ([[async-rollout]]) | `trl/experimental/async_grpo/` at a04ffd3, `max_staleness` default 4, `queue_maxsize` 1024, `weight_sync_steps` 1 |
| 6 | Rollout–training IS correction | `algorithm.rollout_correction`, default off, threshold 2.0 | `is_correction_enable`, default off, band [0.5, 5.0] | on by default with `use_vllm=True`, `sequence_mask`, cap 3.0 |
| 7 | KL placement options | reward or loss, both off by default | reward (default) or loss | GRPO: loss only, β default 0.0 |
| 8 | In-loop evaluation | `test_freq` (default −1), `val_before_train` (default True) | `--eval.steps` (default −1), requires `--reward.remote_url` | `eval_strategy`, `num_generations_eval` |
| 9 | Multi-domain reward routing | reward config group (ch-55) | remote reward service | `reward_funcs` list + `reward_weights` + `multi_objective_aggregation` |
| 10 | LoRA | yes (adapter-only weight sync) | `--ds.lora.rank`, default 0 | PEFT |
| 11 | Long-sequence parallelism | Ulysses sequence parallelism used in the flash-rl DAPO script (learner SP 8) ([[rollout-training-mismatch-tis]] recipe ledger) | `--ds.ring_attn_size`, default 1, auto-enables `--ds.packing_samples` (L506, L638–641) | async trainer packs micro-batches balanced by Σ Lᵢ² |
| 12 | Tool / environment loop in-tree | `ToolAgentLoop` with response masks ([[verl-rollout]]) | `--train.agent_func_path` example script | `tools`, experimental `environment_factory`, `rollout_func`; async trainer has an OpenEnv `HarnessAdapter` bridge |
| 13 | MoE handling | routed-expert replay via `enable_rollout_routing_replay` (needs vLLM ≥ 0.22.0) | MoE aux loss preserved in DPO ([[openrlhf-dpo]]) | `router_aux_loss_coef` default 0.001 in the async GRPO config |
| 14 | Distillation config in-tree | `distillation` config group present in `ppo_trainer.yaml` defaults | not checked | `trl/trainer/distillation_trainer.py`, plus experimental `gkd`, `async_distillation`, `server_distillation`, `minillm` |
| 15 | Preference-optimization variants | not the focus of the RL trainer | DPO with cDPO smoothing, IPO, optional NLL term ([[openrlhf-dpo]]) | DPO and KTO in mainline; `cpo`, `orpo`, `bco`, `nash_md`, `xpo`, `online_dpo`, `sdpo`, `tpo` under `trl/experimental/` ([[trl-online-dpo]]) |
| 16 | Entropy and KL metric names | `actor/entropy`, `actor/ppo_kl`, `actor/reward_kl_penalty` | `ppo_kl`, `vllm_kl`, `clip_ratio` returned from `PolicyLoss.forward` | GRPO: entropy, sign-split clip metrics, `frac_reward_zero_std` ([[entropy-logging-patterns]]) |

Three rows change decisions that older descriptions of these frameworks would lead to. TRL's actor-critic PPO no longer exists in a released mainline, so using TRL for PPO means pinning a release before v1.13.0 (row 3). TRL has an asynchronous GRPO trainer with an explicit staleness bound (row 5). And the KL term to the reference policy can be placed in the loss, not only in the reward, in both verl and OpenRLHF (row 7).

**Interactive companion.** [framework-compare.html](figures/framework-compare.html) runs the decision procedure of §7 one question at a time, shows the path taken, and prints the reached leaf's configuration notes together with the instrumentation checklist, so a candidate stack can be checked against all three instrumentation questions before any throughput comparison.

---

## §4 Performance and staleness, using sourced measurements only

No framework has a throughput number independent of algorithm, model size, response length, straggler distribution and hardware. Every number below is a within-system comparison reported by the system's own authors, and none of them is a head-to-head benchmark of two frameworks under matched conditions.

| System | Comparison | Result | Setting |
|---|---|---|---|
| verl `fully_async_policy` | fully async 64:64 vs colocated sync | 400 steps in 17 h 22 m vs 1 d 16 h 48 m (2.35×) | Qwen2.5-Math-7B, DAPO, 28K max response, 128 H20 ([[verl-rollout]], `fully_async_policy/README.md` L333–361) |
| AReaL | async vs synchronous systems | up to 2.77× end-to-end training time reduction at matched benchmark scores | R1-Distilled-Qwen 1.5B–32B, H800 nodes of 8 GPUs, 16 nodes minimum and 48 nodes for the 32B model ([[areal-async-rl]] §7.1–7.2); the separate scaling study reports linear scaling to 512 GPUs (§7.3) |
| SkyRL-Agent | Async Pipeline vs Async Batch (Bounded) | ≈1.55×, GPU utilization ≈90% during generation | batch 64 × 8 rollouts (512 trajectories), 2×8 H100 ([[skyrl-agent]] §3.2, Fig. 1b) |
| GiGPO / verl-agent | GiGPO additions vs GRPO baseline | grouping 0.01 s and step-advantage computation 0.53 s of a 362.83 s iteration (the paper states these account for < 0.002% of iteration time; 0.54 / 362.83 is 0.15%) | Qwen2.5-1.5B-Instruct on ALFWorld ([[gigpo-verl-agent]] Fig. 6) |
| OpenRLHF async | async vs sync | **not reported** — no throughput or convergence measurement exists in the repository or in any of the six arXiv versions of the framework paper ([[async-rollout]]) |

**The staleness dimension, worked through.** Asynchrony buys throughput by letting the trainer consume rollouts produced by an older policy. How old is a configurable bound, and the bound interacts with the objective.

AReaL bounds staleness with a parameter η and uses a decoupled PPO objective that separates the behaviour policy (which sampled the data) from the proximal policy (the trust-region anchor), so one batch may mix policy versions ([[areal-async-rl]] §5.1). Table 2 varies η with and without that objective, on a 1.5B model. Five of its seven rows are reproduced here; the full sweep is 0/1/2/4/8/16/∞ and also reports AMC23 and AIME25:

| Max staleness η | AIME24 w/o | AIME24 with | MATH-500 w/o | MATH-500 with |
|---|---|---|---|---|
| 0 (oracle) | 42.0 | 42.0 | 89.2 | 89.2 |
| 1 | 41.8 | 42.1 | 89.9 | 89.8 |
| 4 | 23.3 | 42.2 | 66.9 | 89.5 |
| 8 | 35.7 | 41.0 | 87.8 | 89.2 |
| ∞ | 34.0 | 36.9 | 87.1 | 88.1 |

At η = 4 without the decoupled objective the model loses 18.7 AIME24 points and 22.3 MATH-500 points against the oracle; with the objective, both are within 1 point. At η = ∞ with the objective, unbounded staleness still costs 5.1 AIME24 points. The η = 4 row is not monotone with the η = 8 row, so the size of the drop at one η is not a smooth function of η in this sweep. The paper's default settings are η = 4 for coding and η = 8 for math (§7.1). **Result (single study)**, one model size, math and code only.

Compare the bounds the other systems choose: TRL async GRPO `max_staleness` default 4 (weight-update steps a sample may lag before being discarded); INTELLECT-3 `max_off_policy_steps` = 8, described as removing excessively off-policy rollouts ([[intellect-3]] §3.3); OpenRLHF `--train.async_queue_size` default 1, with the README describing larger queues and partial rollout as more off-policy and noting that async "may affect convergence", unquantified ([[async-rollout]]).

**Implication for a general-purpose model.** Staleness is not only a stability parameter. The AReaL table is a direct measurement of general-capability regression caused by an infrastructure setting, visible only because they evaluated on four benchmarks at each η. Any stack running async must log the achieved staleness distribution, not only its configured bound.

---

## §5 Long sequences and long rollouts

Long chain-of-thought RL and long-context training stress the stack in two ways that bear on generality.

**Mismatch grows with uninterrupted length.** In the factor analysis, responses capped at 20K tokens show a higher maximum sampler–learner mismatch than 4K caps, while the mean mismatch is similar; the first 4K tokens of a 20K response often exceed the mismatch of an independently generated 4K response ([[rollout-training-mismatch-tis]], Figs. 9–10). The reading is that long-CoT runs need the correction of §2.2 more than short-answer runs do, and that a correction validated on short responses does not transfer without re-measurement.

**Parallelism choice changes the mismatch.** With a vLLM TP1 sampler and an FSDP SP1 learner, one response of the first 512 DAPO-Math-17k prompts had maximum mismatch > 0.5; a TP2 sampler gave two; adding Ulysses SP8 on the learner gave a double-digit count (Fig. 7). Matching the parallelism between sampler and learner reduced the count (Fig. 8). So sequence parallelism, which is the standard answer to long sequences, is also a source of the numerical gap that makes evaluation non-reproducible.

**What each stack offers.** OpenRLHF has `--ds.ring_attn_size` (default 1) and auto-enables sample packing when it is greater than 1 (`train_ppo_ray.py` L506–511, L638–641). verl's long-sequence path in published runs is Ulysses sequence parallelism on the learner, with the sampler's tensor-parallel size set separately (flash-rl DAPO script: sampler TP 2, learner SP 8, 4 nodes × 8 GPUs; [[rollout-training-mismatch-tis]] recipe ledger). INTELLECT-3 trained at a 65,536-token context with a trainer that uses tensor, context and expert parallelism, and reported a step time of about 1500 s at that length with in-flight weight updating, rising by more than 2× without it ([[intellect-3]] §2.1.1, §3.3). TRL's async GRPO packs micro-batches balanced by the sum of squared row lengths rather than by token budget (`async_grpo_config.py` L113–116).

A separate clamp matters before any of this: verl's vLLM server computes `max_possible_tokens = max_model_len − len(prompt_ids)` and clamps `max_tokens` into [1, max_possible_tokens] without a warning ([[verl-rollout]], `vllm_async_server.py` L591–619). A long-context run whose prompts grow will silently generate shorter responses rather than fail.

---

## §6 Agentic stacks: the environment in the loop

**What changes when an environment enters the loop.** A single-turn RL step has one latency distribution: generation. An agentic step has three: environment setup, generation across many turns, and reward computation, each with a long tail, and any of them can fail for reasons unrelated to the policy. The frameworks in §3 were designed around the first case.

**Four designs, each with its own measured claim.**

1. **SkyRL-Agent** splits each rollout into runtime initialization, agent run, and reward calculation, with bounded queues per stage, and overlaps the CPU-bound stages with GPU generation. That pipeline is ≈1.55× faster than bounded async batching at ≈90% GPU utilization during generation ([[skyrl-agent]] §3.2). Its backend bridge records every LLM call as a transition of input tokens, output tokens and log-probabilities, which is what allows an inference–training mismatch correction and avoids re-tokenization drift (§3.3).
2. **verl-agent / GiGPO** makes the rollout step-wise: each step's prompt is built by a memory module rather than by concatenating the full history, so context stays roughly constant over a long episode, and environments run in parallel with a gym-style interface ([[gigpo-verl-agent]] App. A). The training result is that anchor-state grouping, which retroactively groups actions taken from repeated environment states across trajectories and normalizes within that group, raises ALFWorld overall success from 72.8 ± 3.6 (GRPO) to 86.1 ± 4.7 and WebShop success rate from 56.8 ± 3.8 to 67.4 ± 4.5 at Qwen2.5-1.5B-Instruct, averaged over 3 seeds (Table 1). Those two numbers are the GiGPO variant without standard-deviation normalization (F_norm = 1); the variant with it scores 86.7 ± 1.7 and 65.0 ± 3.2. Settings: group size 8, 16 groups per rollout (128 concurrent environments), ω = 1, γ = 0.95, KL-loss coefficient 0.01 (App. E.1), history length 2 (App. E.2).
3. **AReaL** makes the rollout worker interruptible: an `update_weights` request interrupts ongoing generations, the workers discard KV caches, load the new parameters and continue decoding the unfinished sequences (§4.1), and a rollout controller rejects generation requests that would violate the staleness constraint ([[areal-async-rl]] §5.1). This is the design that pairs with the η table in §4.
4. **prime-rl** separates an orchestrator (CPU) from the trainer (FSDP2) and an OpenAI-compatible vLLM inference pool, with `/update_weights` and `/reload_weights` endpoints, and abstracts multi-turn rollout and scoring behind verifiers environments so any Environments Hub entry plugs into the loop ([[intellect-3]] §2.1.1). INTELLECT-3 used 60 nodes of 8 H200s at roughly a 1:3 training-to-inference node ratio (16 training, 44 inference) (§3.3).

**Transition-level versus trajectory-level.** [[agent-lightning]] takes the same per-call transition view from the opposite direction: it converts executions of agents written in LangChain, the OpenAI Agents SDK or AutoGen into (input, output, reward) transitions and assigns every call in an episode the episode's return. The paper reports rising train and test reward curves on three agents with Llama-3.2-3B-Instruct but no numeric tables and no general-capability evaluation, so the benefit over masked trajectory training is not measured there.

**Generality evidence from agentic RL.** SkyRL-Agent's SA-SWE-32B was trained with RL only, on 4.5K R2E-Gym SWE instances, and evaluated outside that domain: Terminal-Bench 13.75 → 16.25, BrowseComp-Plus accuracy 18.1 → 19.4, WebArena 15.8 → 17.0 against the Qwen3-32B starting point ([[skyrl-agent]] Table 3). Each model is evaluated once with no variance reported. The counter-case is in the same paper: for the computer-use agent, training reward rose while validation accuracy showed "little to no gain", and the training tasks were a subset of the benchmark (§5.3). **Result (single study)** in both directions; single-domain agentic RL transferring to other tool domains is an **open question**.

**Instrumentation an agentic stack must add.** Environment crash rate separated from task failure rate (a crashed episode is not a negative example); trajectory length and turn-count distributions, not means; per-environment reward so a single broken environment does not move the average; and staleness measured in weight-update steps. SkyRL-Agent reports the reason directly: tool-serving timeouts contaminated rollouts around deep-research iteration 30 and recovery took steps 30–33, which the authors give as the reason to mask abnormal trajectories (§5.1).

---

## §7 A decision procedure that terminates

The procedure has seven questions and six leaf nodes naming four stacks: TRL and verl each appear twice, once for a run without an environment and once for a tools path. Every question has both answers defined, every answer moves forward or reaches a leaf, and no answer returns to an earlier question. The questions are answered in order.

```
Q1. Does the run put a stateful environment in the loop — episodes with more than one
    model call against an external system that holds state between calls?
      NO  -> Q2
      YES -> Q5

Q2. Does one node hold the training model, its optimizer state, and the rollout engine?
      YES -> LEAF A
      NO  -> Q3

Q3. Do you need Megatron-style tensor or pipeline parallelism, MoE routed-expert replay,
    or token-in/token-out rollout with per-request abort during weight sync?
      YES -> LEAF C
      NO  -> Q4

Q4. Do you need a Ray-managed actor/critic split with a separate value model, or a remote
    reward-model service as a process boundary?
      YES -> LEAF B
      NO  -> LEAF C

Q5. Does the run need more than about one hundred concurrent environment instances, or
    episodes longer than about ten turns or 30K tokens?
      YES -> LEAF D
      NO  -> Q6

Q6. Is the environment expressible as function-call tools that return text into the context?
      NO  -> LEAF D
      YES -> Q7

Q7. Does one node hold the training model, its optimizer state, and the rollout engine?
      YES -> LEAF A (tools path)
      NO  -> LEAF C (tools path)
```

The three thresholds in Q5 are read off the published agentic runs in §6, not from a measurement of where a single-turn trainer stops working: GiGPO runs 16 groups of 8 (128 concurrent environments) ([[gigpo-verl-agent]] App. E.1) and SkyRL-Agent's SWE training caps episodes at 50 turns and 32K tokens ([[skyrl-agent]] §4.2).

**Leaves.**

- **LEAF A — TRL.** `GRPOTrainer` for critic-free RL, `DPOTrainer`/`KTOTrainer` for preference optimization, `trl/experimental/async_grpo/` when rollout and training must overlap. The tools path uses `tools` / `environment_factory`, or the OpenEnv `HarnessAdapter` bridge in the async trainer. Not available: mainline actor-critic PPO (deleted in PR #7020, 2026-09-04).
- **LEAF B — OpenRLHF.** `PPOTrainerAsync` or the synchronous trainer, DeepSpeed ZeRO with a Ray actor/critic split, `--reward.remote_url` for the reward service. Note the coupling from §2.1: in-loop evaluation requires that remote service.
- **LEAF C — verl.** FSDP or Megatron backends, async vLLM rollout with per-request abort, `@register_adv_est` and `@register_policy_loss` for algorithm changes without touching the trainer, `ToolAgentLoop` for the tools path.
- **LEAF D — a stack built for environments.** SkyRL-Agent when the bottleneck is long-tail episode latency across stages; verl-agent when episodes are long enough that concatenating history is the constraint; AReaL when rollouts must be interrupted at weight sync and staleness bounded; prime-rl when environments should come from, and be shared with, a common environment interface.

**The instrumentation gate, applied once at the leaf you reach.** Before committing, confirm all three:

1. Periodic in-loop evaluation on a held-out suite containing at least one family the reward does not target, with per-benchmark series (§2.1).
2. A logged rollout-versus-training log-probability diagnostic, and a correction available if it is non-zero (§2.2).
3. Reward or environment routing that keeps the domain identity of each score (§2.3).

If a leaf fails one of these and the hook cannot be added in the time available, the next candidate is taken from the fixed order A → B → C → D, and the gate is applied once more. The order is fixed and the list has four entries, so the procedure ends after at most four gate evaluations.

---

## Negative samples and negative feedback

**Which meaning applies here.** This chapter is about instrumentation, so it covers two of the four meanings from the course standard: **negative as gradient** (tokens or sequences with A < 0), and **negative as content** (tool errors and corrective hints placed in the agent's context). Choosing a stack decides how well either can be observed.

**Mechanism.** For a sampled token with id y and logits z, the gradient of the log-probability with respect to logit j is

```
∂ log p_y / ∂ z_j = 1[j = y] − p_j
```

A negative advantage reverses the sign of the update on that token, so probability mass is removed from y and redistributed to the other entries in proportion to their current probabilities `p_j`. The mass therefore concentrates on whichever alternative is already most likely — which may be another wrong answer. Diagnosing this requires statistics split by advantage sign, not aggregated.

**What each stack lets you see.**

- **TRL GRPO** splits its clip metrics by sign: the low-clip metric counts `r < 1 − ε_low` only where A < 0, and the high-clip metric counts `r > 1 + ε_high` only where A > 0 (grpo_trainer.py L2589–2590). It also logs `frac_reward_zero_std`, the fraction of samples whose prompt group had zero reward variance and therefore contributed no policy term ([[trl-grpo]]).
- **OpenRLHF** returns a single `clip_ratio` that counts clipped tokens for both signs together and does not count dual-clip activations ([[openrlhf-ppo]] loss.py L180). Sign-split diagnostics have to be added.
- **verl** logs `actor/pg_clipfrac` and a separate `actor/pg_clipfrac_lower` (core_algos.py L1372–1374), and its GRPO estimator multiplies the group-normalized score by `response_mask` without removing zero-advantage groups (L1165–1170), so their tokens still count in the token denominator under the default `loss_agg_mode: token-mean` (L329; actor.yaml L86) ([[verl-grpo]]).

**Worked example (what a zero-variance group does).** Take a group of G = 4 with binary reward and one correct response. verl's estimator gives μ = 0.25 and σ = 0.5, so the correct response gets +1.5 and each wrong response −0.5 ([[verl-grpo]]). Now take a group where all four fail: σ = 0 and every advantage is 0/(0 + 1e-6) = 0. The prompt contributes nothing to the gradient but its tokens still enter the denominator, so the effective learning rate on the rest of the batch drops. Without `frac_reward_zero_std` or an equivalent, the observable symptom is a training curve that flattens for no visible reason.

**Controls that stacks provide.** TRL's `mask_truncated_completions` removes truncated completions from the loss instead of penalizing them (`grpo_config.py` L259–262). SkyRL-Agent excludes trajectories stopped by the 32K-token context limit or the 50-turn step limit from the gradient while leaving reward and advantage estimation unchanged ([[skyrl-agent]] §4.2). verl's agent loop gives tool and template tokens mask 0 so they are never trained on ([[verl-rollout]]). INTELLECT-3 masks rather than clips: importance ratios outside [α, β] = [0.5, 5] are set to 0, and a whole rollout is masked if any token ratio falls below 1e-5, which the report calls critical against the trainer–inference mismatch ([[intellect-3]] §3.3, Eq. 1–2).

**Negatives as content in agent loops.** TRL converts exceptions, unknown tool names and unsupported call types into `{"error": ...}` tool messages the model reads on the next turn, and logs `tools/call_frequency` and `tools/failure_frequency` ([[trl-grpo]] L1502–1532, L1778–1786). SkyRL-Agent injects corrective feedback for recoverable errors and hints covering tool failures, remaining step or context budget, invalid function calls and failed edits (§3.4, §4.2). These are trained with ordinary cross-entropy on the model's own subsequent tokens; no likelihood is pushed down, so the displacement risk of the gradient case does not apply.

**Effect on generality.** None of the framework sources measures the effect of its negative-sample handling on breadth. The measured evidence that exists is elsewhere in the course (ch-43, ch-44). What this chapter contributes is the requirement: choose a stack whose logs let you split by advantage sign and by domain, because otherwise a negative-gradient pathology confined to one domain averages away.

---

## Recipe

Rows are settings from published runs with their pinned loci, plus framework defaults recorded separately. A framework default and a paper value are different facts.

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| INTELLECT-3 (GLM-4.5-Air base) | 106B MoE | RL | prompts per step × rollouts per prompt; max context | 256 × 16; 65,536 tokens | arXiv:2512.16144 §3.3 | verified 2026-09-17 | no ablation reported |
| INTELLECT-3 | 106B MoE | RL | `max_off_policy_steps`; optimizer; LR | 8; Muon; 1e−6 | §3.3 | verified 2026-09-17 | stated as removing excessively off-policy rollouts; no sweep |
| INTELLECT-3 | 106B MoE | RL | IS masking band [α, β]; whole-rollout mask threshold | [0.5, 5]; 1e−5 | §3.3 Eq. 1–2 | verified 2026-09-17 | report states double-sided masking was critical against trainer–inference mismatch; no ablation table |
| INTELLECT-3 | 106B MoE | RL | nodes; train:inference split; step time | 60 × 8 H200; 16:44; ≈1500 s/step at 65,536 length | §3.3 | verified 2026-09-17 | without in-flight weight updating, step time rises > 2× |
| INTELLECT-3 | 106B MoE | eval-gate | in-loop eval interval and suite | every 15 steps on AIME24, AIME25, LiveCodeBench, HLE, GPQA | §3.3 "Online Evaluation", Fig. 9 | verified 2026-09-17 | scores still rising at end of run |
| AReaL (R1-Distilled-Qwen) | 1.5B–32B | RL | max staleness η | 4 for coding, 8 for math | arXiv:2505.24298 §7.1 | verified 2026-09-17 | Table 2 staleness sweep 0/1/2/4/8/16/∞ with and without decoupled PPO |
| verl `fully_async_policy` run (Qwen2.5-Math-7B) | 7B | RL | `staleness_threshold`; `partial_rollout`; rollout:train GPUs | 0.5; True; 64:64 of 128 H20 | verl@753aed3 `verl/experimental/fully_async_policy/README.md` L349–361 | verified 2026-09-14 ([[verl-rollout]]) | staleness ablation 0/0.1/0.3/0.5 in the same README |
| flash-rl DAPO run (Qwen2.5-32B) | 32B | RL | TIS cap C; sampler TP; learner Ulysses SP | 8; 2; 8 | github.com/yaof20/verl@fd02f7b `dapo_qwen32b_bf16.sh` L5, L41, L58, L63 | verified 2026-09-14 ([[rollout-training-mismatch-tis]]) | Fig. 1 TIS vs no TIS; no sweep over C |
| SA-SWE-32B (SkyRL-Agent, from Qwen3-32B) | 32B | RL | batch = mini-batch; rollouts per task; LR; KL / entropy | 64 = 64; 8; 1e−6; both disabled | arXiv:2511.16108 §4.2 | verified 2026-09-14 ([[skyrl-agent]]) | no ablation reported |
| SA-SWE-32B | 32B | RL | context / turn limits; truncated trajectories | 32K tokens, 50 turns; excluded from gradient | §4.2 | verified 2026-09-14 | no ablation reported |
| GiGPO on verl-agent (Qwen2.5-1.5B-Instruct) | 1.5B | RL | group size; groups per rollout; ω; γ; KL loss coef; history length | 8; 16 (128 environments); 1; 0.95; 0.01; 2 | arXiv:2505.10978v3 App. E.1 (history length: App. E.2) | verified 2026-09-17 | ω sensitivity in App. E.5 Table 5 (WebShop success rate peaks at 68.3 for ω = 0.8 and stays within 65.8–68.3 over ω ∈ [0.4, 1.2], against 56.6 at ω = 0 and 56.3 at ω = 1.4) |
| verl framework default | n/a | RL | `trainer.test_freq`; `val_before_train`; `log_val_generations` | −1; True; 0 | verl@753aed3 `ppo_trainer.yaml` L154–201 | verified 2026-09-17 | no ablation reported |
| verl framework default | n/a | RL | `rollout_is`; `rollout_is_threshold` | `null` (off); 2.0 | `algorithm/rollout_correction.yaml`@753aed3 | verified 2026-09-17 | no ablation reported |
| OpenRLHF framework default | n/a | RL | `--eval.steps`; `--eval.dataset`; eval temperature; eval samples/prompt | −1; None; 0.6; 4 | OpenRLHF@64c1cc4 `cli/train_ppo_ray.py` L286, L535–540 | verified 2026-09-17 | eval dataset requires `--reward.remote_url` (L672–673) |
| OpenRLHF framework default | n/a | RL | KL controller; `init_coef`; async queue size | `FixedKLController`; 0.01; 1 | `ppo_trainer.py` L171–176; `train_ppo_ray.py` L275, L417–419 | verified 2026-09-14 ([[openrlhf-ppo]], [[async-rollout]]) | no ablation reported |
| TRL framework default | n/a | RL | `loss_type`; β; `scale_rewards`; vLLM IS correction | `dapo`; 0.0; `group`; on with `use_vllm=True`, `sequence_mask`, cap 3.0 | trl@a08e713 `grpo_config.py` L709, L792–819 | verified 2026-09-14 ([[trl-grpo]]) | no ablation reported |
| TRL framework default | n/a | RL | async GRPO `max_staleness`; `queue_maxsize`; `weight_sync_steps`; `dtype` | 4; 1024; 1; `float32` | trl@a04ffd3 `trl/experimental/async_grpo/async_grpo_config.py` L37–44, L345–375 | verified 2026-09-17 | docstring cites the training–inference mismatch as the reason for the float32 default |

**Starting point for a small general-purpose run.** For a single-node GRPO run on a model of a few billion parameters, every number here comes from a `verified` row above, with its original condition stated. Use TRL with `use_vllm=True`, which turns on sequence-level IS correction at cap 3.0 by default (TRL default row; the default was set for vLLM-sampled rollouts, not measured on any specific model). Set `loss_type` and `beta` explicitly rather than accepting `dapo` and 0.0, because a run with β = 0 creates no reference model at all and therefore has no KL control against forgetting (TRL default row). Set `eval_strategy` to run on a schedule and `num_generations_eval` explicitly, and include at least one held-out family the reward does not target — the only published in-loop example of that discipline in these sources is INTELLECT-3's five-benchmark plot every 15 steps, at 106B MoE on 512 H200s, so the interval does not transfer; the practice does. If asynchrony is enabled, start at `max_staleness` 4, which matches both TRL's default and AReaL's coding setting (AReaL 1.5B–32B on H800s; the η = 4 row of Table 2 also shows this is the staleness at which a missing decoupled objective did the most damage).

---

## Generalization lens

**(a) What increases breadth.** Routing several reward sources into one run, with per-domain logging, so that the mixture can be rebalanced while it runs — TRL's `reward_funcs` plus `reward_weights` and prime-rl's verifiers-environment abstraction are the two in-tree designs for this ([[trl-grpo]]; [[intellect-3]] §2.1.1). Cross-domain transfer from agentic RL is reported once: SWE-only RL raised Terminal-Bench 13.75 → 16.25, BrowseComp-Plus 18.1 → 19.4 and WebArena 15.8 → 17.0 ([[skyrl-agent]] Table 3), single evaluation per cell.

**(b) What causes narrowing or forgetting.** Running with the default configuration: TRL GRPO's β = 0.0 means no reference model and no KL anchor, and verl's `use_kl_loss` and `use_kl_in_reward` are both off by default (§2.4). Unbounded or uncorrected staleness: the AReaL η = 4 row without the decoupled objective loses 18.7 AIME24 points and 22.3 MATH-500 points ([[areal-async-rl]] Table 2). Uncorrected rollout–training mismatch: reward collapse mid-run in the Thinking Machines Bigmath RLVR experiment, and entropy falling below 0.2 with abnormally long responses in the INT8 DAPO-32B run ([[thinkingmachines-defeating-nondeterminism]]; [[rollout-training-mismatch-tis]] Fig. 5). Narrow training sets: SkyRL-Agent's computer-use agent, trained on 32 OSWorld tasks that are a subset of the benchmark, showed rising training reward with "little to no gain" in validation accuracy (§5.3).

**(c) How to measure it for this stage.** Three instruments, in order of how often they are missing. First, in-loop evaluation on a fixed held-out suite with at least one family outside the reward's domain, run on a schedule from step 0 and logged as separate series per benchmark (verl `test_freq`, OpenRLHF `--eval.steps`, TRL `eval_strategy`). Second, the mismatch metric (`vllm_kl`, or verl's rollout-correction ratio) plotted next to reward, since the published failures announce themselves there first. Third, reproducibility discipline for the evaluation path itself: fixed sampler dtype, fixed tensor- and sequence-parallel sizes, and awareness that default kernels are not batch-invariant, so a temperature-0 evaluation is not deterministic unless batch-invariant kernels are used ([[thinkingmachines-defeating-nondeterminism]]). Known measurement errors: evaluation generation competes with rollout for inference capacity, so a short `test_freq` biases the throughput comparison between stacks; and per-benchmark scores at small sample counts move by several points between seeds, which none of the framework sources quantifies.

---

## Common mistakes and how to detect them

| Mistake | Observable symptom | Check |
|---|---|---|
| Choosing a stack by algorithm name | The chosen framework "has GRPO" but the default `loss_type`, β and normalizer differ from the paper being reproduced | Read the default values, not the trainer name: TRL default `loss_type="dapo"`, β = 0.0; verl `norm_adv_by_std_in_grpo=True`; OpenRLHF `group_norm` with ε = 1e-9 |
| Running with no in-loop evaluation | Only reward, KL, entropy and clip fraction appear in the dashboard | Confirm `test_freq` / `--eval.steps` / `eval_strategy` is not at its default off value before launching |
| Comparing checkpoints generated with different sampler settings | Breadth "improves" or "regresses" by a few points with no training change | Record sampler dtype, TP size and SP size with every eval number; re-run the baseline under the new settings before concluding |
| Leaving IS correction off because the run looks stable | Entropy falls steadily, responses lengthen, k1 KL is logged as negative, logged reward rises while benchmark accuracy does not | Plot `vllm_kl` (OpenRLHF) or the rollout-correction ratio (verl) from step 0; a negative k1 KL is a sign the log-ratio is being taken between mismatched engines ([[rollout-training-mismatch-tis]] Fig. 5) |
| Increasing async queue depth for throughput without bounding staleness | Throughput improves, held-out scores fall while training reward does not | Log achieved staleness per batch; compare against the configured bound; see the AReaL η table in §4 |
| Treating a zero-variance prompt group as harmless | Training curve flattens; gradient norm falls with no config change | Log the fraction of groups with zero reward std (`frac_reward_zero_std` in TRL; compute it manually elsewhere) |
| Extending a single-turn trainer into an agent loop | Wall time per step dominated by a small number of episodes; environment failures counted as task failures | Separate environment-crash rate from task-failure rate, and plot the turn-count distribution rather than its mean |
| Assuming a framework's stated capability is current | A configuration that worked six months ago no longer imports | Pin a commit and read the file: TRL's PPO trainer was deleted on 2026-09-04 (PR #7020) and TRL gained an async GRPO trainer in the same period |

---

## Check your understanding

1. The clipped surrogate is nearly identical in verl, OpenRLHF and TRL. Explain why two runs of the same algorithm on the two frameworks can still produce models with different breadth, naming three specific mechanisms outside the loss.
2. AReaL's η = 4 row shows AIME24 at 23.3 without the decoupled objective and 42.2 with it. Explain the causal chain from "batch mixes several policy versions" to "held-out benchmark score falls", and say why the decoupled objective breaks that chain.
3. OpenRLHF requires `--reward.remote_url` before `--eval.dataset` may be set. Explain what this constraint implies for the order in which you build a generality evaluation for an OpenRLHF run, compared with a TRL run.
4. TRL's GRPO default is β = 0.0, which creates no reference model. Give the argument for that default from the optimization side, and the argument against it from the generality side, and state what evidence would settle it.
5. The maximum sampler–learner mismatch grows with uninterrupted response length while the mean does not. Explain why a maximum matters more than a mean for training stability, using the softmax-gradient and importance-weight variance arguments.
6. A zero-variance prompt group contributes no policy-gradient term but its tokens still count in the `token-mean` denominator. Derive the effect on the effective step size for the remaining prompts in the batch, and say what you would log to catch it.
7. GiGPO's anchor-state grouping adds 0.54 s to a 362.83 s iteration and raises ALFWorld success by 13.3 points over GRPO. Explain why this is an argument about credit assignment rather than about compute, and what it implies for choosing between a single-turn trainer and an agentic one.
8. You are handed a stack that scores highest on throughput and lowest on instrumentation. Construct the argument for rejecting it that does not rely on the claim that generality matters more than speed.

---

## Connections

- **ch-54 — Rollout Infrastructure: Off-Policy Data, Asynchronous RL, and Agentic Environments.** The mechanisms this chapter selects between: staleness, partial rollout, environment integration.
- **ch-55 — verl Internals: Losses, Rollouts, Multi-Domain Rewards, and In-Loop Validation.** The verl column of §3 in full, including the reward routing summarized here.
- **ch-56 — OpenRLHF Internals: PPO, DPO, Ray Orchestration, and Forgetting Controls.** The OpenRLHF column, including the KL controller behaviour quoted in §2.4.
- **ch-57 — TRL Internals: SFT, DPO, GRPO, and Distillation Trainers.** Previous chapter; the TRL column, including the loss-type family and the tool loop.
- **ch-58a — Open General-Model Recipes End to End: Pretraining to Merge.** Next chapter; takes the stack chosen here and follows a published recipe through every stage, including the merge step whose framework support §2.4 records.
- **ch-02 — Numerical Precision, Determinism, and Train–Inference Mismatch.** The numerical causes of the §2.2 gap.
- **ch-43 — Entropy, Output Diversity, and KL Control in RL.** The theory behind the KL-placement row of §2.4 and the entropy diagnostics of §3 row 16.
- **ch-51a — Evaluating Agent Generality and Reliability.** The evaluation side of §6.
- **ch-59 — Capstone: Reproduce One Stage of an Open General-Model Recipe with a Generality Gate.** Uses this chapter's decision procedure as its stack-selection input.

---

## Sources

- [[verl-ppo-loss]] — verl's registered policy loss and clip parameters; the shared surrogate of §1.
- [[verl-grpo]] — verl's GRPO estimators, the ε = 1e-6 constant, zero-advantage group behaviour, and the token-denominator effect used in the negatives section.
- [[verl-rollout]] — the async vLLM server, the `max_tokens` clamp, the tool agent loop and response masks, and the `fully_async_policy` 2.35× timing table.
- [[openrlhf-ppo]] — `PolicyLoss`, the KL controllers and default coefficient, and the three IS-correction modes with the `vllm_kl` metric.
- [[openrlhf-dpo]] — the DPO path referenced in §3 row 15, including MoE aux-loss preservation.
- [[async-rollout]] — OpenRLHF's async trainer, queue semantics, partial rollout, and the absence of any published throughput measurement for it.
- [[trl-grpo]] — the eight loss types and their defaults, advantage scaling, vLLM IS correction, sign-split clip metrics, and the tool loop with error messages as content.
- [[trl-ppo]] — TRL's actor-critic PPO and its removal in PR #7020, used for §3 row 3.
- [[trl-online-dpo]] — the experimental preference-optimization trainers listed in §3 row 15.
- [[entropy-logging-patterns]] — the metric-name and KL-estimator differences across the three frameworks.
- [[rollout-training-mismatch-tis]] — the measured sampler–learner gap, the TIS formula and variance argument, the parallelism and length factor analysis, and the flash-rl DAPO recipe rows.
- [[thinkingmachines-defeating-nondeterminism]] — batch invariance, the 80-of-1000 completions measurement, and the RL section on off-policy correction and sampler–trainer KL.
- [[areal-async-rl]] — interruptible rollout, the staleness bound η, the decoupled PPO objective, and the Table 2 staleness sweep used throughout §4.
- [[skyrl-agent]] — the three-stage dispatcher and its 1.55× measurement, transition-based backend bridge, the SA-SWE-32B recipe, and the out-of-domain evaluation table.
- [[gigpo-verl-agent]] — verl-agent's step-wise rollout with a memory module, anchor-state grouping, the ALFWorld and WebShop results, and the per-iteration overhead breakdown.
- [[agent-lightning]] — transition-level training of agents written in external frameworks, cited for the design contrast in §6.
- [[intellect-3]] — prime-rl's orchestrator/trainer/inference architecture, the verifiers Environments Hub interface, the RL settings and masking thresholds, and the 15-step in-loop evaluation on five benchmarks.
