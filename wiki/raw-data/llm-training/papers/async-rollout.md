<!-- scope: OpenRLHF asynchronous RL trainer (ppo_trainer_async.py) — generator and trainer Ray actors joined by a bounded queue, vLLM lock, partial-rollout pause/resume, and the vLLM-vs-trainer importance-sampling correction it is paired with; pinned at commit 64c1cc4
     deps: [[ppo]], [[openrlhf-ppo]]
     see-also: [[verl-rollout]], [[on-off-policy-rlhf]], [[minibatch-sharing-rl]]
-->

# OpenRLHF `openrlhf/trainer/ppo_trainer_async.py` — asynchronous PPO/REINFORCE trainer (commit 64c1cc4)
- **Core Insight:** OpenRLHF's async mode runs rollout generation and policy training as two concurrent Ray actors connected by a queue of capacity `--train.async_queue_size` (default 1), with a cross-actor lock or, in partial-rollout mode, vLLM pause/resume around weight broadcast; the repository reports no throughput or convergence measurement for this mode (ppo_trainer_async.py L268–350; README L665–679).
- **Guideline:** When throughput matters more than strict on-policy sampling and convergence has already been validated in synchronous mode, enable `--train.async_enable` with queue size 1, because the README describes larger queues and partial rollout as more off-policy and warns that async training may affect stability (README L667, L679, L731–732). Otherwise use the synchronous mode (README: "better stability") or the colocated hybrid engine (README: "strictly on-policy", L730).
- **Authors:** OpenRLHF project (organization); the framework paper credits "Asynchronous Agentic RL" to Haotian Xu and Jian Hu (arXiv:2405.11143v6 App. A)
- **Year:** 2026 (pinned commit 64c1cc4, 2026-04-19; the async feature is announced for OpenRLHF 0.8.0 in the README news list)
- **URL:** https://github.com/OpenRLHF/OpenRLHF/blob/64c1cc4f30d4c0dab772c8aa1dc11288c425e0da/openrlhf/trainer/ppo_trainer_async.py
- **Source type:** released config/code
- **Relevant topics:** asynchronous actor-learner RL, rollout queue, weight synchronization, partial rollout, off-policy importance sampling, agent RL

## Summary
The artifact is the asynchronous trainer of OpenRLHF and the code paths it depends on at commit 64c1cc4: `ppo_trainer_async.py`, the `train_ppo_ray.py` CLI flags, `BasePPOTrainer.train_step` in `ppo_trainer.py`, the vLLM engine wrapper, the importance-sampling (IS) correction in `models/loss.py`, the README async section, and the example script `train_reinforce_baseline_ray_agent_async.sh`. `ppo_trainer_async.py` is byte-identical at current `main` b117b2b (2026-09-14); the IS-correction flags were renamed between the two commits (see Verification). The repository and all six arXiv versions of the OpenRLHF paper contain no async-versus-sync throughput or convergence result.

## Key Contributions
- Two Ray actors, `GenerateSamplesActor` and `TrainingActor`, run their `fit` loops concurrently under `PPOTrainerAsync` (L37–350).
- Backpressure through a slot-token queue of the same capacity as the rollout queue (L292–297).
- `VLLMLock`, a Ray actor wrapping `asyncio.Lock`, so that generation and weight broadcast do not overlap and each batch is generated with one set of weights (L19–34).
- Partial-rollout mode: generation runs without the lock and weight broadcast pauses and resumes vLLM, so in-flight samples may contain tokens from old and new weights (L128–140, L255–265; CLI L277–283).
- A separate, optional IS correction in `PolicyLoss` between trainer and vLLM log-probabilities, with a `vllm_kl` diagnostic (loss.py L150–173).

## Key Figures/Tables to Study (code loci at 64c1cc4)
- `ppo_trainer_async.py` L100–164 (`GenerateSamplesActor.fit`), L202–253 (`TrainingActor.fit`), L255–265 (`broadcast_to_vllm`), L268–350 (`PPOTrainerAsync`).
- `cli/train_ppo_ray.py` L145–148 (trainer selection), L256–283 (IS and async flags), L663–670 and L686–688 (argument checks).
- `models/loss.py` L150–182 (IS correction and returned metrics); `trainer/ppo_trainer.py` L213–252 (`train_step`).
- README L665–679 and L727–733 (async options and execution-mode table).

## Technical Details
1. **Selection:** `--train.async_enable` imports `PPOTrainerAsync` in place of `PPOTrainer` (train_ppo_ray.py L145–148). Async mode cannot be combined with `--vllm.enable_sleep` (L666–667); with `--train.colocate_all` it colocates only the DeepSpeed models (L663–664); `--rollout.vllm_generate_batch_size > --rollout.batch_size` requires async mode (L686–688).
2. **Queues:** `rollout_queue = Queue(maxsize=queue_size)`; `rollout_slots = Queue(maxsize=queue_size)` is pre-filled with `queue_size` tokens and acts as a counting semaphore; `queue_size` must be positive (ppo_trainer_async.py L287–297).
3. **Generator loop:** block on `rollout_slots.get()`; the token carries the trainer's latest `global_step` (L109–110). If an evaluation is due, generate eval samples under the lock and enqueue `("eval", step, metrics)` (L116–125). Otherwise acquire the lock (normal mode only), call `SamplesGenerator.generate_samples`, release the lock (L128–140), and enqueue `(rollout_samples, client_states, filter_pass_rate, generation_time)` (L142–151). If no samples were produced, the token is returned to avoid a deadlock (L154–157).
4. **Trainer loop:** dequeue a batch, return one slot token before training (L229–230), then call `train_step` (L232). `timing/step_total` is wall-clock time including queue wait (L234–237).
5. **Weight sync:** `train_step` runs experience making, the PPO update, then `broadcast_to_vllm()` (ppo_trainer.py L213–247). The async trainer's override acquires the lock and, in partial-rollout mode, calls vLLM `pause_generation` before and `resume_generation` after the broadcast (ppo_trainer_async.py L255–265). The engine wrapper calls `pause_generation(mode="keep")` (vllm_engine.py L132–136).
6. **Policy version:** the enqueued payload has no policy-version field, and neither loop filters samples by age (L149–151, L227–232). The README states that larger `async_queue_size` is "more off-policy" (L667).
7. **IS correction (optional, off by default):** with `--algo.advantage.is_correction_enable`, `PolicyLoss` computes `log_ratio = old_log_probs − rollout_log_probs` (loss.py L154). Type `tis` clamps exp(log_ratio) to [low, high] and multiplies the loss (L169–172); `icepop` zeroes tokens whose ratio is outside [low, high] (L155–160); `seq-mask-tis` drops sequences whose geometric-mean ratio is outside the band and keeps token ratios for the rest (L161–168). `vllm_kl = masked_mean(rollout_log_probs − old_log_probs)` (L173). Defaults: threshold [0.5, 5.0], type `tis` (train_ppo_ray.py L257–270). The code cites the blog "Your Efficient RL Framework Secretly Brings You Off-Policy RL Training" (loss.py L150).
8. **KL controller note:** `train_step` updates the KL controller with a TODO stating it must be `FixedKLController` because `AdaptiveKLController` is incompatible there (ppo_trainer.py L252).

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| OpenRLHF framework default (no checkpoint) | — | RL | async enable; queue size; partial rollout | False; 1; False | github.com/OpenRLHF/OpenRLHF@64c1cc4 `openrlhf/cli/train_ppo_ray.py` L274–283 | verified 2026-09-14 | no ablation reported |
| OpenRLHF framework default (no checkpoint) | — | RL | IS correction enable; type; thresholds [low, high] | False; tis; [0.5, 5.0] | same file L257–270 | verified 2026-09-14 | no ablation reported |
| Example script `train_reinforce_baseline_ray_agent_async.sh` (Qwen/Qwen3-4B-Thinking-2507; no released result) | 4B | RL | mode flags | async_enable + partial_rollout_enable + agent_func_path; IS correction type icepop | `examples/scripts/…_agent_async.sh` @64c1cc4 L33, L62–63, L93–94 | verified 2026-09-14 | no ablation reported |
| same | 4B | RL | advantage estimator; actor LR; KL | reinforce_baseline; 5e-7; init_coef 1e-5, k2, used as loss | L86–92 | verified 2026-09-14 | no ablation reported |
| same | 4B | RL | prompts per step; samples per prompt; train batch; max new tokens; episodes | 128; 8; 1024; 64000; 1 | L40–58 | verified 2026-09-14 | no ablation reported |
| same | 4B | RL | data; placement | zhuzilin/dapo-math-17k; actor 4 GPUs, ref 4 GPUs, 2 vLLM engines × TP 2, colocate_all, ds.enable_sleep | L9, L65–73 | verified 2026-09-14 | no ablation reported |
| OpenRLHF async mode | — | RL | throughput vs sync; staleness bound | not reported | checked README, code above, arXiv:2405.11143 v1–v6 | not reported | — |

## Findings relevant to agentic training and off-policy stability
- **Agentic training:** async mode is used together with `--train.agent_func_path` in the example script that the README lists as its multi-turn example (script L33, L62–63; README L684); the README describes token-in-token-out agent execution to keep sampling and training tokens consistent (README L115, L676). The framework paper states the async design is "readily extensible for scalable agent RL training" without a measurement (arXiv:2405.11143v6 §3.2).
- **Stability:** the README lists synchronous as the default with "better stability", async as "higher throughput, may affect convergence", and partial rollout as "most aggressive off-policy", to be paired with IS correction (README L671–679, L731–732). No experiment in the repository supports or quantifies these statements.

## Connections
- [[openrlhf-ppo]] — `PolicyLoss`, KL wiring, and IS-correction variants at the same commit.
- [[verl-rollout]] — verl's asynchronous vLLM server, a separate artifact with its own weight-sync hooks.
- [[on-off-policy-rlhf]] — evidence on on-policy versus off-policy preference training, relevant to queue-induced policy lag.
- [[ppo]] — the clipped objective that `PolicyLoss` implements.
- Not in the library: HybridFlow (arXiv:2409.19256 §4.1) uses "asynchronous dataflow execution" for data futures between models on separate devices (Fig. 5b), not for rollout staleness; IMPALA (arXiv:1802.01561, §1) introduces V-trace to correct the lag between actor and learner policies.

## Verification
- Checked on 2026-09-14 against: github.com/OpenRLHF/OpenRLHF@64c1cc4f30d4c0dab772c8aa1dc11288c425e0da (`trainer/ppo_trainer_async.py`, `cli/train_ppo_ray.py`, `models/loss.py`, `trainer/ppo_trainer.py`, `trainer/ray/vllm_engine.py`, `trainer/ppo_utils/samples_generator.py`, `README.md`, the example script); the same files at `main` b117b2b; arXiv:2405.11143 v1–v6; arXiv:2409.19256; arXiv:1802.01561.
- Corrections to the previous card version:
  - Card mixed three artifacts (OpenRLHF paper, HybridFlow paper, IMPALA) and verl code → rewritten as one artifact, the OpenRLHF async trainer code; verl material moved to [[verl-rollout]].
  - "OpenRLHF paper §3.3 async; Figure 5: 1.9× at 7B, 1.6× at 70B" → no version of the paper has an async section with measurements; §3.3 in v2–v4 is "PPO Implementation Tricks" and Fig. 5 is "PPO training curves"; v5–v6 describe async dataflow in prose only.
  - "`RolloutActor`" → `GenerateSamplesActor` (L37–38). "`vllm_lock`: asyncio.Lock" → `VLLMLock` Ray actor wrapping `asyncio.Lock` (L19–34). "queue capacity 1–2" → `maxsize = async_queue_size`, default 1 (L292; CLI L275).
  - "Partial rollout interrupts incomplete generations" → vLLM is paused with `mode="keep"` and resumed; in-flight samples may mix weights (vllm_engine.py L132–136; CLI L277–283).
  - "iCEPO" → `icepop` (CLI L266–270). At b117b2b the flags are `--algo.advantage.is_correction_level/mode/gating/threshold` (train_ppo_ray.py@b117b2b L256–290) and the example script uses level `token`, mode `mask` (L93–94).
  - "HybridFlow dataflow supports sync, async, and partial-rollout modes" → HybridFlow's asynchronous execution refers to data futures between models (§4.1).
- Removed as unsupported by the source: "1.6–2.0× throughput vs sync on 8×H100 for ~7B"; "use async for any RL run >30 min"; "staleness k = queue_depth + partial_rollout_depth, typical 1–2"; "sequence dropped if IS weight exceeds a hard cap"; "`vllm_kl` > 0.1 with clipfrac pegged at 1 is the failure signature"; "IS clip is mathematically the same object as V-trace"; verl `priority` and "pause state ≈ line 628" (belongs to [[verl-rollout]], not checked here); "verl blog Continuous batching RL"; "used by every 2025 production RL stack"; "TRL forthcoming async trainer"; verl engine "paused state" hook.
- Not reported by the source: throughput gain, maximum staleness, convergence comparison, turn-level partial rollout for agents. README news dates the 0.8.0 async release as "2025/5" at b117b2b (L65) and "2026/5" at 64c1cc4 (L62).
