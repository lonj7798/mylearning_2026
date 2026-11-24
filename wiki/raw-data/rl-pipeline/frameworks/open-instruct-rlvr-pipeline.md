<!-- scope: allenai/open-instruct at commit f020e9a: end-to-end walk of open_instruct/grpo_fast.py RLVR — dataset row format, verifier registry and dispatch, IFEval constraint checking, code/LM-judge verifiers, mixed-domain reward, DataPreparationActor async bound, active sampling, ρ (TIS) loss weighting, NCCL weight broadcast
     deps: [[grpo]], [[rlvr-tulu3]]
     see-also: [[olmo-3-rl-pipeline]], [[allenai-olmo3-open-instruct-scripts]], [[ifbench]], [[verifiable-if-rl]], [[truncated-importance-sampling]], [[async-rollout]], [[slime]], [[verl-rollout]]
-->

# open-instruct `grpo_fast.py`: the multi-domain RLVR pipeline behind Olmo 3
- **Core Insight:** One string column, `dataset`, selects the verifier for each row: `build_all_verifiers` registers every `VerifierFunction` subclass by lower-cased name, and `apply_verifiable_reward` looks up `reward_fn_mapping[ds.lower()]` per row, so math, code, IFEval and LM-judge prompts share one batch and one loss (`ground_truth_utils.py` L1239–1277, L1298–1362).
- **Guideline:** To add a domain, write a `VerifierFunction` subclass returning a score in [0, 1] and tag rows with its name; keep `async_steps > 1` when `active_sampling` is on (the config asserts it) and keep `mask_tool_use=True` when using the ρ correction (asserted in `grpo_fast.py` L1016–1017), because tool-output tokens have no sampler log-probs to correct against.
- **Authors:** Allen Institute for AI (allenai/open-instruct maintainers).
- **Year:** 2024–2026 (repository); code read at commit f020e9a9, committed 2026-11-24.
- **URL:** https://github.com/allenai/open-instruct/tree/f020e9a9a8402a220a67646e3254fc23fa121c27
- **Source type:** released config/code
- **Relevant topics:** RLVR, verifier registry, IFEval constraint verification, code-execution reward, LM-judge reward, async actor/learner, active sampling, truncated importance sampling, NCCL weight broadcast

## Summary
`grpo_fast.py` runs a DeepSpeed (or OLMo-core) learner and a pool of vLLM actors under Ray. A `DataPreparationActor` pushes prompts into a queue, collects finished groups from the actors, scores them, filters zero-variance groups, and hands packed batches to the learner; a separate weight-sync thread broadcasts new weights after every step. Olmo 3 Think/Instruct/RL-Zero runs use this file with per-domain mixes (see [[olmo-3-rl-pipeline]] and [[allenai-olmo3-open-instruct-scripts]]).

## Key Contributions
- Verifier-tag dispatch with weights and a `remap_verifier` option (`ground_truth_utils.py` L1239–1277).
- IFEval-style verifier over 54 registered constraint classes (`IFEvalG/instructions_registry.py` L125–191) plus a training-admission screen for malformed or unsuitable constraints (`ifeval_constraints.py`).
- Code verifier that posts the program and tests to an HTTP execution service (`code_utils/api.py` L50–65, `/test_program` and `/test_program_stdio`).
- LM-judge verifiers `general-<type>` for quality, reference-based quality, safety, factuality, creative writing, refusal (`judge_utils.py` L278–287); scores on a 1–10 scale divided by 10 (L272–275).
- Async bound `async_steps`, active sampling, and the ρ = π_train_old / π_infer_old correction with clamp and mask bounds (`data_loader.py` L405–445; `grpo_utils.py` L129–153, L445–496).

## Key Figures/Tables to Study
- `ground_truth_utils.py` L333–381 (IFEval), L967–1035 (code), L1298–1362 (dispatch), L1365–1473 (`RewardConfig.build`).
- `data_loader.py` L800–825 (`process_group`), L1105–1160 (accumulate + filter), L1405–1460 (prep loop).
- `grpo_fast.py` L1455–1513 (weight-sync thread), L1926–2022 (main loop); `grpo_utils.py` L780–817 (`perform_weight_sync`).

## Technical Details
**Pipeline at a glance**
1. Row → tokens: `rlvr_tokenize_v1` builds the prompt from `messages[:-1]` and copies `ground_truth` and `dataset` (`dataset_transformation.py` L1013–1014, L2209–2251). Sample rows: `{messages, ground_truth, dataset, constraint_type, constraint}` (`test_data/rlvr_sample.jsonl`).
2. Prompt queue: the prep actor pre-fills `async_steps × global_batch_size` prompts, then blocks while `training_step - last_consumed_step > async_steps` (`data_loader.py` L1412–1430).
3. Generation: vLLM actors; a prefetch thread refills the engine whenever active tasks fall below `inference_batch_size` and `should_stop` is false (`vllm_utils.py` L435–440).
4. Reward: `RewardConfig.build()` → `apply_verifiable_reward` → per-row verifier `async_call` (`ground_truth_utils.py` L1365–1473).
5. Filter: groups with `np.std(reward_scores) == 0` are dropped; with `active_sampling` a replacement prompt is pulled (`data_loader.py` L817, L1131–1135). Results older than `async_steps` model versions are stale (`result_is_stale`, L996–1005).
6. Loss: DAPO clip or CISPO, multiplied by ρ weights, masked mean over tokens (`grpo_utils.py` L539–561; `grpo_fast.py` L720).
7. Weight sync: main thread calls `weight_sync_trigger.notify(step)` (`grpo_fast.py` L2022); the sync thread runs `broadcast_to_vllm` on every learner rank (L1479–1513).

**Verifier dispatch.** Each row may carry a list of `(ground_truth, dataset)` pairs; rewards are summed with `reward_mult * score * reward_weight`.
```python
# ground_truth_utils.py L1323-1335
for gt, ds in zip(ground_truth_list, dataset_list):
    reward_func = reward_fn_mapping.get(ds.lower())
    if reward_func is None:
        logger.warning("No reward function found for dataset %s. Skipping reward.", ds)
        continue
    task = reward_func.async_call(
        tokenized_prediction=tok_prediction, prediction=prediction,
        label=gt, query=query, rollout_state=rollout_state,
    )
```
`reward_mult` is `verification_reward`, default 10.0 (L1371). Optional R1 format reward is additive or a gate (L1409–1452); `non_stop_penalty` overwrites the score of non-`stop` finishes (L1465–1469). Per-turn tool rewards are combined by `last` or `sum` aggregators (L1095–1108, L1444).

**How a constraint string becomes a check (IFEval).** The label is a Python-literal list holding one dict with parallel `instruction_id` and `kwargs` arrays. Each id maps to a checker class; `build_description(**kwargs)` sets its parameters; `check_following(answer)` returns a bool on the response with the thinking section removed.
```python
# ground_truth_utils.py L370-381
for instruction_key, args in zip(instruction_keys, args_list):
    if args is None:
        args = {}
    args = {k: v for k, v in args.items() if v is not None}
    instruction_cls = instruction_dict[instruction_key]
    instruction_instance = instruction_cls(instruction_key)
    instruction_instance.build_description(**args)
    if prediction.strip() and instruction_instance.check_following(answer):
        rewards.append(1.0)
    else:
        rewards.append(0.0)
return VerificationResult(score=sum(rewards) / max(len(rewards), 1))
```
Example checker: `keywords:frequency` → `KeywordFrequencyChecker`, which counts `re.findall(keyword, value, re.IGNORECASE)` and compares with `less than` / `at least` the given frequency (`IFEvalG/instructions.py` L758, L811–818). `ifeval_constraints.if_training_issues` rejects malformed bundles and specific bad arguments (e.g. `letters:letter_counting` with relation `less than` and N ≤ 3), noting that "checkers otherwise choose random defaults" when arguments are missing (L63–104); its callers at this commit are in `open_instruct/miles/` (`run_data.py` L374, `source_preparation.py` L156), not in `grpo_fast.py`.

**Code and judge.** `CodeVerifier` posts `{program, tests, max_execution_time}` and scores the pass rate, zeroed when below `code_pass_rate_reward_threshold` (L1018); Olmo 3 scripts set the threshold to 0.99 and point `--code_api_url` at an AWS endpoint (`scripts/train/olmo3/7b_think_rl.sh` L62–63). `LMJudgeVerifier` calls an OpenAI-compatible judge via litellm; Olmo 3 uses `hosted_vllm/Qwen/Qwen3-32B` (L58).

**Weight broadcast.** `broadcast_weights_to_vllm` gathers parameters on all ranks (DeepSpeed ZeRO-3 / FSDP collective) and only rank 0 sends; NCCL when a model-update group exists, CUDA IPC otherwise (`vllm_utils.py` L1383–1410). Engines are paused through the actor manager, not drained:
```python
# grpo_utils.py L796-810
ray.get(actor_manager.set_should_stop.remote(True))
try:
    results, actor_sync_times = utils.ray_get_with_progress(broadcast_refs, ...)
    if not inflight_updates:
        utils.ray_get_with_progress(itertools.chain.from_iterable(results), ...)
    utils.ray_get_with_progress([e.wake_up.remote() for e in vllm_engines], ...)
finally:
    ray.get(actor_manager.set_should_stop.remote(False))
```
`inflight_updates` defaults to True (`data_loader.py` L443): the thread does not wait for inner engine-update RPCs, so generation in progress continues.

**ρ correction.** `rho = exp(clamp(old_logprob - vllm_logprobs, -10, 10))`, optionally sequence-level; tokens outside `[rho_mask_lower, rho_mask_upper]` get weight 0, others are clamped to `[rho_clamp_lower, rho_clamp_upper]` (default upper 2.0) (`grpo_utils.py` L445–496).

## Verification
- Checked on 2026-11-24 against the local shallow clone at f020e9a9 (2026-11-24). All loci refer to that commit.
- IFEval reward is the fraction of constraints satisfied in code; the Olmo 3 paper describes it as binary ([[olmo-3-rl-pipeline]] Verification).
- `IFEvalVerifierOld` (L384) and `if_functions.py` (483 lines) are an older constraint path; not traced.
- Not found: a per-domain reward weight in the Olmo 3 scripts (all verifier weights are 1.0 in constructors); a built-in search tool verifier for web search (only `R1SearchVerifier` / `ReSearchVerifierF1` answer matchers, L563–661).
