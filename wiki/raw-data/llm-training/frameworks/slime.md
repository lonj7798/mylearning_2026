<!-- scope: slime (THUDM/slime) RL post-training framework at a pinned commit: Megatron training + SGLang/router rollout + Data Buffer, colocated vs disaggregated placement, fully-async rollout, dynamic-sampling filter, partial rollout, weight sync
     deps: [[async-rollout]], [[grpo]]
     see-also: [[glm-4-5]], [[glm-4-5-recipe]], [[glm-4]], [[verl-rollout]], [[openrlhf-ppo]], [[areal-async-rl]], [[glm-4-1v-thinking]]
-->

# slime (THUDM/slime): Megatron training, SGLang rollout, and the Data Buffer
- **Core Insight:** slime keeps one training entry point (`train.py`) for both synchronous and asynchronous RL: the loop is generate → train → pause admission → push weights → resume (`train.py` L47–99), and fully-async behaviour is selected by swapping the rollout function (`--rollout-function-path slime.rollout.fully_async_rollout.generate_rollout_fully_async`), not by a second trainer (`docs/en/advanced/rl-systems.md` L43; `docs/en/blogs/introducing_slime.md` L45).
- **Guideline:** Use `--colocate` (rollout and training share GPUs and alternate residency with offload) when rollouts are similar in length and GPU count is the constraint; use separate GPU pools plus the fully-async rollout function when sample lengths vary widely (agentic, long-tail generations), and monitor `staleness/mean` and `staleness/max`, because the docs state that overlap is paid for with policy staleness and that the async speedup bound is optimistic (`rl-systems.md` L7, L21–43).
- **Authors:** Zilin Zhu, Chengxing Xie, Xin Lv, and slime contributors (citation block, `README.md` L191–199); THUDM / Z.ai.
- **Year:** 2025 (citation `year = {2025}`, `README.md` L195). Code read at commit 79aa41a8, committed 2026-11-06.
- **URL:** https://github.com/THUDM/slime/tree/79aa41a884ab0bf65222c94418f6357d3c31192b
- **Source type:** released config/code
- **Relevant topics:** rollout infrastructure, colocated vs disaggregated RL, asynchronous rollout and staleness, dynamic sampling (DAPO-style filter), partial rollout, weight synchronization, Megatron/SGLang integration

## Summary
slime connects Megatron-LM (training) and SGLang servers behind an sgl-router (rollout). The README defines three modules: training reads data from the Data Buffer and synchronizes parameters to rollout after training; rollout generates samples, rewards and verifier outputs and stores them in the Data Buffer; the Data Buffer manages prompt initialization, custom data and rollout generation methods (`README.md` L67–71). The README lists slime as the RL framework behind GLM-4.5, GLM-4.6, GLM-4.7 and the GLM-5 series (`README.md` L21). Megatron arguments are passed through directly; SGLang arguments are passed through with a `--sglang-` prefix (`README.md` L144–148).

## Key Contributions
- Single training loop for sync and async RL (`train.py` L13–99; `introducing_slime.md` L45).
- Placement by one flag: `--colocate` puts rollout engines on the actor GPUs and forces offload (`slime/utils/arguments.py` L72–79, L2141–2156).
- DAPO-style dynamic sampling as a pluggable group filter with oversampling (`arguments.py` L480–505; `filter_hub/dynamic_sampling_filters.py` L9–25).
- Partial rollout via SGLang `/abort_request`: unfinished generations are recycled into the buffer (`sglang_rollout.py` L442–489; `introducing_slime.md` L86–90).
- Custom generate / reward functions as the extension point for multi-turn and agentic rollouts (`README.md` L70, L81–88).

## Key Figures/Tables to Study
- `train.py` L47–99 (the full RL step), `slime/ray/placement_group.py` L103–154 (GPU layout), `slime/rollout/sglang_rollout.py` L492–618 (`generate_rollout_async`), `slime/data/data_source.py` L191–266 (buffer), `docs/en/advanced/rl-systems.md` L5–43 (placement and async cost model).

## Technical Details
**Training loop (`train.py`).** For each `rollout_id`: (1) `rollout_manager.generate.remote(rollout_id)` returns a data reference (L51); (2) rollout is offloaded if `--offload-rollout` (L53–54); (3) critic trains first when `--use-critic`, and the actor trains only after `num_critic_only_steps` (L59–67); (4) `training_completed` releases queue capacity (L71); (5) optional checkpoint (L73–83); (6) `pause_rollout_admission`, then `actor_model.update_weights()`, then `resume_rollout_admission` unless this was the last rollout (L88–93); (7) periodic eval (L98–99).

**Placement (`placement_group.py`).** Without `--colocate`, actor and rollout get separate placement groups "so a restarted trainer [can] resize without moving the serving engines" (L126–137). With `--colocate`, one group of `max(actor_num_gpus, rollout_num_gpus)` GPUs is shared with rollout offset 0 (L117–118). Colocate defaults: `offload_train = True`, `offload_rollout = True`, and `rollout_num_gpus = actor_num_gpus_per_node * actor_num_nodes` if unset (`arguments.py` L2141–2156).

**Rollout step (`generate_rollout_async`, `sglang_rollout.py` L492–618).**
1. Target is `rollout_batch_size` prompt groups; each group must hold `n_samples_per_prompt` samples (L520, L550; `data_source.py` L229–232).
2. While submitted work is below target, fetch `over_sampling_batch_size` prompts from the data source and submit them (L531–535).
3. As groups finish (`asyncio.FIRST_COMPLETED`, L539), run the dynamic filter; a rejected group is discarded and `remaining_batch_size` is decremented so more prompts are fetched (L555–571).
4. When the batch is full, `abort()` cancels remaining requests (L588). With `--partial-rollout`, aborted groups with non-empty responses get `metadata["start_rollout_id"]` and are returned to the buffer (L472–484).

```python
def check_reward_nonzero_std(args, samples, **kwargs):
    rewards = [(sample[-1] if isinstance(sample, list) else sample).get_reward_value(args) for sample in samples]
    keep = torch.tensor(rewards, dtype=torch.float64).std() > 1e-6
    return DynamicFilterOutput(keep=keep, reason=None if keep else f"zero_std_{round(rewards[0], 1)}")
```
(`slime/rollout/filter_hub/dynamic_sampling_filters.py` L9–17.) The `_with_fallback` variant keeps a rejected group when dropping it would force another sampling round (L20–25).

**Data Buffer (`data_source.py`).** `RolloutDataSourceWithBuffer.get_samples` first pops from `self.buffer` (recycled partial groups) and fills the rest from the prompt dataset (L200–212). Default pop order is `pop_first`; `--buffer-sort-by-staleness` selects `pop_oldest`, which sorts groups by the minimum `weight_versions` of their samples (L195–197, L248–259); `--buffer-filter-path` replaces both (L197).

**Fully-async rollout (`fully_async_rollout.py` L1–29).** A background asyncio worker keeps a fixed pool of in-flight trajectories across rollout boundaries. Admission pause during weight update leaves in-flight tasks alive; rejected groups under the dynamic filter are replaced from the warm queue. `RolloutManager.pause_rollout_admission` pauses the `fully_async` consumer with `drain=False` (`slime/ray/rollout.py` L151–158).

**Weight sync.** `UpdateWeightFromTensor` docstring: colocated path serializes GPU→CPU, gathers with Gloo, then sends by Ray IPC to the engine; distributed path uses NCCL broadcast to remote engines (`update_weight/update_weight_from_tensor.py` L50–56). Disk and delta-disk updaters also exist (`update_weight/` directory listing).

**Algorithms exposed.** `--advantage-estimator` ∈ {grpo, gspo, cispo, reinforce_plus_plus, reinforce_plus_plus_baseline, ppo}, default grpo (`arguments.py` L1064–1074); `--eps-clip` default 0.2, `--eps-clip-high` defaults to `eps_clip` (L1022–1023, L2079).

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| slime defaults (framework) | — | RL | advantage estimator; PPO clip | grpo; 0.2 (high = low unless set) | `slime/utils/arguments.py` L1064–1074, L1022–1023, L2079 @79aa41a | verified 2026-11-24 | no ablation reported |
| slime colocate mode | — | infra | offload_train / offload_rollout / rollout GPUs | True / True / actor GPUs | `arguments.py` L2141–2156 | verified 2026-11-24 | not applicable |
| Qwen3-4B example | 4B | RL | rollout_batch_size; n_samples_per_prompt; global_batch_size | 32; 8; 256 | `scripts/run-qwen3-4B.sh` L58–63 | verified 2026-11-24 | example script, no ablation |
| Qwen3-4B example | 4B | RL | max response len; temperature; lr; KL coef; entropy coef | 131072; 1; 1e-6; 0.00; 0.00 | `run-qwen3-4B.sh` L60–61, L104, L106, L114 | verified 2026-11-24 | example script, no ablation |
| GLM-4.5 production run | 355B-A32B | RL | slime hyperparameters | not reported in the repository (model config only: `scripts/models/glm4.5-355B-A32B.sh`) | checked README, scripts/, docs/en | not reported | see [[glm-4-5]] §3.5 |

## Findings relevant to generality and negative feedback
- **Negative feedback / zero-signal groups.** The default dynamic filter drops groups whose reward standard deviation is ≤ 1e-6, which removes all-correct and all-wrong groups before they reach the GRPO loss (`dynamic_sampling_filters.py` L9–17; `arguments.py` L493–505 cites DAPO).
- **Staleness.** The docs decompose train–rollout log-prob mismatch into parameter staleness and same-version engine mismatch, and state that deterministic kernels reduce only the second term (`rl-systems.md` L33–41).
- **Generality.** No benchmark numbers are reported in the repository for any model; the framework is evaluated in the model reports that use it ([[glm-4-5]]).

## Connections
- [[glm-4-5]], [[glm-4-5-recipe]]: report §3.5 describes slime's colocated synchronous mode for reasoning RL and disaggregated asynchronous mode for agentic RL.
- [[verl-rollout]], [[openrlhf-ppo]]: other open RL stacks; verl's HYBRID/COLOCATED/STANDALONE modes correspond to slime's colocate/separate placement.
- [[async-rollout]], [[areal-async-rl]]: asynchronous RL designs; the `/abort_request` endpoint was designed with the AReaL team (`introducing_slime.md` L86).

## Verification
- Checked on 2026-11-24 against a shallow clone of https://github.com/THUDM/slime at commit 79aa41a884ab0bf65222c94418f6357d3c31192b (2026-11-06). All file/line references refer to that commit.
- `train_async.py` is not present at this commit; async is the rollout-function swap above. Older descriptions of a separate async trainer do not apply to this commit.
- Inconsistency: `rl-systems.md` L15 says "slime uses CUDA IPC for colocation", while the `UpdateWeightFromTensor` docstring describes GPU→CPU serialization plus Ray IPC (L54). Both are quoted; the implementation was not traced further.
- Not reported by the source: throughput or speedup numbers for colocated vs async modes; GLM-4.5 RL hyperparameters.
