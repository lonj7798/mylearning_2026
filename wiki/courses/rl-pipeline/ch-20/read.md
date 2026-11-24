<!-- chapter: ch-20
     track: rl-pipeline
     kind: content
     title: Fault Tolerance: Recover at the Smallest Scope
     deps: [ch-02]
     sources: [[mai-rl-infrastructure]], [[mai-thinking-1]], [[olmo-3-rl-pipeline]], [[slime]], [[verl-rollout]], [[areal-async-rl]], [[open-instruct-rlvr-pipeline]], [[async-rollout]]
     figures: figures/recovery-ladder.html
-->

# Chapter 20 — Fault Tolerance: Recover at the Smallest Scope

> **Core insight.** In an asynchronous RL job the components hold different kinds of state, and they fail at different rates. The learner holds the only copy of optimizer state and trains on thousands of GPUs bound by collective communication, so one bad rank stalls all of them. An inference replica holds only weights it can re-receive and the in-flight rollouts it can drop. The controller holds the curriculum, admission state and the bookkeeping of which rollouts are in flight. A recovery design that restarts the whole job on any failure throws away the state of every healthy component. Rocket instead repairs the smallest component whose state was lost and escalates only when that fails: replace one inference replica; rebuild only the learner while inference keeps generating ("roughly five minutes"); rebuild inside the Ray job; requeue at cluster level ([[mai-rl-infrastructure]] p. 11–12). On a 6,000-GPU basis, Table 1 attributes 7.07% of training time (1,056,000 GPU-hours) to 88 learner-only recoveries and 0.42% (63,000 GPU-hours) to 21 in-job recoveries.
>
> **Guideline.** When the inference fleet and the learner are separate processes, make the learner restartable without the inference fleet, because Rocket's learner recovery is the largest fault-tolerance saving it reports (n = 88, 2 h per event, 176 h, 7.07%; Table 1) and slime implements the same split by keeping its serving cluster as a detached Ray actor that survives the training job (`docs/en/advanced/fault-tolerance.md` L9, L72). When a process can stall without exiting, treat a stall as a failure with a timeout instead of waiting for collective-communication timeouts, because Rocket kills learner ranks "without waiting for NCCL timeouts" (p. 11) and slime's health monitor retires an engine whose Ray call does not return within its timeout (`health_monitor.py` L144–154). When the inference fleet shrinks, set a live-fraction threshold below which the job escalates, because a fleet too small to keep the learner fed changes staleness and throughput together (ch-04). Place the control plane on nodes that fail less often than GPU nodes, because Rocket runs Ray heads and CPU actors on CPU-only nodes that were "more stable than GPU nodes in our deployments" (p. 12); no failure-rate numbers are given.

---

## Why this chapter matters

Rocket's runs last "many thousands of training steps spanning weeks to months on thousands of GPUs" ([[mai-rl-infrastructure]] p. 1). The post argues that a run that climbs from a base model cannot use the escape a distilled start has: a distilled model starts RL near its ceiling, so a failed run can be restarted cheaply, while a climb from base loses weeks of progress (p. 1; ch-01). MAI-Thinking-1 adds a second reason that restarts are not free: resuming from a pre-collapse checkpoint "is often not viable" because instabilities are embedded "many steps before the actual collapse" ([[mai-thinking-1]] Sec. 3.1.4). Recovery must therefore preserve state that took the run time to produce.

The accounting in Table 1 (p. 11) makes the cost concrete. It covers "a portion of RL training, assuming 6,000 GPUs":

| Event type | n | Saved per event | Time saved (h) | % | GPU-hours saved |
|---|---|---|---|---|---|
| In-job recovery | 21 | 0.5 h | 10.5 | 0.42% | 63,000 |
| Learner recovery | 88 | 2 h | 176 | 7.07% | 1,056,000 |
| Runtime elasticity (ch-21) | 123 | 2 h | 246 | 9.88% | 1,476,000 |
| Overall training time | — | — | 2,490 | 100.00% | 14,940,000 |
| Total estimated savings | — | — | 432.5 | 17.37% | 2,595,000 |

The card verified that GPU-hours = hours × 6,000 and % = hours / 2,490 for every row. The table states savings per event; the card does not record how each per-event saving was estimated (for example, which counterfactual restart time was assumed). Treat the numbers as MAI's estimates, not as measured A/B differences.

Use the figure [recovery-ladder.html](figures/recovery-ladder.html) while reading §3–§6. It shows the ladder rung by rung, what keeps running at each rung, and the Table 1 row for it, with a GPU-count control that rescales GPU-hours.

---

## §1 What each component holds, and what is lost when it fails

The recovery scope follows from the state each component owns. Using Rocket's Fig. 1 components ([[mai-rl-infrastructure]] p. 3; ch-02):

| Component | State it owns | Can the state be re-derived? | Cost of losing it |
|---|---|---|---|
| Learner (GPU pool) | weights, optimizer moments, RNG, the step counter | only from the last checkpoint | all steps since the checkpoint |
| Inference replica (SGLang on GPU) | a copy of weights; KV caches; in-flight decode | weights: re-sent by weight transfer (ch-05). In-flight rollouts: no | the rollouts it was serving |
| Rollout workers (CPU) | agent-loop state, tool sessions, per-rollout grading | partly (tool sessions are external) | the rollouts they were running |
| Problem workers (CPU) | group membership, group grading, advantages | no, for groups in flight | groups in flight |
| Controller (CPU) | curriculum position, admission, packing, segment references | from recovery checkpoints (Fig. 1 lists "controller + learner state") | admission state, curriculum position |
| Rollout store | token IDs, sampling log-probs, model-turn payloads | no | trained-but-unreferenced data is harmless; untrained data is lost |

Two facts from this table drive the design. First, the learner and the controller are the only components whose state cannot be rebuilt from elsewhere, so they are checkpointed (Fig. 1, "Recovery checkpoints"). Second, an inference replica's state is either re-sendable (weights) or droppable (in-flight rollouts), so a replica can be replaced without touching anything else. Dropping its in-flight rollouts is the same policy as dropping failed rollouts in ch-04, with the same cost: noise or bias in the GRPO baseline, because failures "may share features that never make their way to the learner" (p. 7).

---

## §2 Detection: stalls are failures

**Rocket.** "Health checks treat stalled processes as unhealthy" (p. 11). The important case is a stalled learner rank. In a collective operation every rank waits for every other; a rank that hangs without crashing produces no error until the collective timeout fires. Rocket's learner recovery therefore kills ranks without waiting for NCCL timeouts (p. 11). The post does not give the health-check interval or the stall threshold.

**Rocket v1 (MAI-Thinking-1).** The earlier Rocket lists five stability mechanisms: a replica self-watchdog, a router circuit breaker, per-class liveness thresholds, a step-progress watchdog, and mid-flight retry of a request on another replica ([[mai-thinking-1]] Sec. 3.6). Its problem worker *retried* failed rollouts; the Base 2 post drops them instead (ch-04). No thresholds are given.

**slime.** The serving side has an explicit monitor. It probes every engine concurrently and retires any engine whose probe fails or whose Ray call does not return in time:

```python
# slime@79aa41a slime/utils/health_monitor.py:L130–L154
    def _run_health_checks(self, *, force=False) -> None:
        # The background thread and rollout-completion RPC share this lock, so
        # they cannot retire the same engine while another check uses its handle.
        with self._check_lock:
            if self._stop_event.is_set():
                return
            if not force and self._pause_event.is_set():
                return
            checks = {
                engine.health_generate.remote(timeout=self._check_timeout): rollout_engine_id
                for rollout_engine_id, engine in enumerate(self._server_group.engines)
                if engine is not None
            }
            if checks:
                # Bound queued/wedged actor RPCs as well as the underlying HTTP
                # requests. All engines are probed concurrently.
                ray.wait(list(checks), num_returns=len(checks), timeout=self._check_timeout)
            for handle, rollout_engine_id in checks.items():
                try:
                    ray.get(handle, timeout=0)
                except Exception as error:
                    # Both HTTP failures and unresponsive/dead actor RPCs retire
                    # the engine; HTTP timeout alone cannot detect a wedged actor.
                    logger.error("Health check failed for engine %s: %s", rollout_engine_id, error)
                    self._server_group.retire_engine(rollout_engine_id, timeout=self._check_timeout)
```

The comment on the last branch states the same principle as Rocket: a wedged process is detected by a bounded wait, not by an error. slime's defaults are 600 s for the first wait after resume, the interval, and the timeout; the doc's example for a large MoE sets the interval to 10 s (`docs/en/advanced/fault-tolerance.md` L94–106). The monitor pauses while weights or memory ownership change (L79–86 of `health_monitor.py`), because a probe sent during an offload or weight update would see a legitimately unresponsive engine.

**open-instruct.** The GRPO driver checks vLLM engine health inside the training loop and fails the job if a weight sync does not finish in time:

```python
# open-instruct@f020e9a open_instruct/grpo_fast.py:L1843–L1861 (WEIGHT_SYNC_TIMEOUT_S = 120.0 at L161)
    def health_check_fn(weight_sync_future: futures.Future | None, expect_new_weight_sync: bool = False):
        seen_actors_paused = False
        start = time.perf_counter()
        while True:
            if weight_sync_future is not None and weight_sync_future.done():
                weight_sync_future.result()
            should_stop = ray.get(actor_manager.should_stop.remote())
            if should_stop:
                seen_actors_paused = True
            if not should_stop and (not expect_new_weight_sync or seen_actors_paused):
                break
            if time.perf_counter() - start > WEIGHT_SYNC_TIMEOUT_S:
                raise RuntimeError(f"Weight sync timed out after {WEIGHT_SYNC_TIMEOUT_S}s - vLLM engines may be stuck")
            time.sleep(0.1)
        ray_get_with_progress(
            [engine.check_background_threads.remote() for engine in vllm_engines],
            desc="Checking vLLM engine health",
            enable=False,
        )
```

Detection is bounded (120 s), but the response is a `RuntimeError` in the driver, which ends the job. Recovery is then a whole-job restart from `checkpoint_state_dir` (§5). The Olmo 3 RL card ([[olmo-3-rl-pipeline]]) reports no fault-tolerance mechanism or failure statistics for OlmoRL: not reported.

---

## §3 Learner-only recovery

**Rocket's procedure** (p. 11): pause admission of new work; kill the learner ranks without waiting for NCCL timeouts; rebuild the learner; restore the latest checkpoint; reconnect to the live controller. It takes "roughly five minutes". Inference keeps running throughout, "and accepts the staleness cost".

**Why pause admission.** Rollouts admitted while the learner is down would be generated, graded and queued, but no new weights reach the sampler until the learner trains again. Pausing admission bounds how much queued data accumulates on the pre-failure weights. In-flight rollouts continue; hour-long agentic rollouts (Fig. 2: SWE median 93 min, web development 58 min) are exactly the work a whole-job restart would destroy.

**Where the staleness cost comes from.** The post does not decompose it. Two mechanisms are consistent with the design described in ch-03 and ch-05 (this is reasoning, not a statement from the post):

1. Rollouts completed during the outage wait in the controller queue and are trained over several steps after the learner returns, so the later-trained ones have a larger `t_trained − t_generated`.
2. If the restored checkpoint is older than the learner's last completed step, the sampler holds weights from a version the learner no longer has; the learner retrains from an earlier version while rollouts in flight were generated by a later one. Whether Rocket retrains, discards, or reuses the data trained between the checkpoint and the failure is not stated in the card.

At Rocket's cadence (sampler refresh every 4 learner steps; a sample step of 3 min in Fig. 4) a 5-minute outage is on the order of one or two learner steps (derived from Fig. 4's step time, which varies by step).

**slime's equivalent.** slime separates a `ServingCluster`, "a named, detached Ray actor that survives the job that created it", from the `RolloutManager` and the trainer (`fault-tolerance.md` L70–74). After a Megatron failure, the user resubmits `train.py` to the same Ray cluster; slime finds the retained serving cluster by session name and resumes (L9, L35–47). Training resumes from the last saved model checkpoint and "retrains the batches after that checkpoint", including batches that trained but whose updates were not saved (L51). The `TrainingRecovery` journal releases replay data only when a checkpoint commits:

```python
# slime@79aa41a slime/ray/training_recovery.py:L431–L444
    def checkpoint_committed(self, rollout_id):
        # Runtime training completion is insufficient: only a durable training
        # checkpoint lets us release batches needed for replay.
        self.checkpoint_step = rollout_id
        self.checkpoint = replace(
            self.checkpoint,
            load=self.checkpoint.save,
            ckpt_step=rollout_id,
            start_rollout_id=rollout_id + 1,
            finetune=False,
            no_load_optim=getattr(self.args, "use_stateless_adam", False),
            no_load_rng=False,
        )
        self.release_batches(through=rollout_id)
```

| Property | Rocket learner recovery | slime retained-serving recovery |
|---|---|---|
| Inference during recovery | keeps running | serving cluster retained; generation admission paused by the manager if it survives (L76) |
| Trigger | automatic (health check) | manual resubmit; slime "does not automatically retry the training job" (L9) |
| Data after the checkpoint | not stated | replayed from durable storage; requires straw transport or saved rollout dumps (L13–25) |
| Requirements | not stated | `torch_dist` checkpoints with RNG and optimizer state; `--no-save-rng` rejected (L29; `training_recovery.py` L47–57) |
| Weight versions | not stated | "Serving weight versions continue to increase across restarts" (L53) |
| Bitwise reproducibility after restart | not stated | not guaranteed when TP/PP changes (L55) |
| Reported duration | about 5 min | not reported |

The checkpoint interval sets the replay cost in slime: "Longer intervals mean more batches to retrain after a failure and more recovery data to retain" (L27). The same trade-off exists for Rocket but the post gives no interval.

---

## §4 Inference recovery and the live-fraction threshold

**Rocket.** Inference recovery replaces one replica. If the live fraction of the fleet drops below a threshold, the job restarts at cluster level (p. 11). The threshold value is not given. The reason a threshold is needed follows from ch-04: Rocket deliberately keeps the job inference-bound with an overallocated learner. Losing replicas lowers generation throughput, which lengthens the wait for slow GRPO groups and raises staleness; below some fraction the learner is starved and a cluster-level restart with a full fleet costs less than running degraded.

**slime.** Failed engines are removed from the router, their Ray handles cleared, and "missing engines restart before the next weight update and then load the trainer's weights" (`fault-tolerance.md` L88–90). The restart is tied to the weight-update boundary so that a restarted engine never serves stale weights after rejoining. External engines (not launched by slime) "are not recovered by slime fault tolerance" (`external-rollout-engines.md` L103). This matters for a setup that serves the policy from separately managed OpenAI-compatible endpoints (ch-22).

**verl v1 async trainer.** A group whose rollout "finishes with `failure`" is evicted from the replay buffer and refilled with one new prompt from the dataloader, the same as groups that exceed the staleness threshold or fail the DAPO filter (`docs/advance/v1_async_trainer.md` L84–96). This is the drop-and-refill policy at group granularity; the doc does not describe replica replacement.

**AReaL.** The library cards record a staleness manager and decoupled PPO ([[areal-async-rl]]) but no per-replica replacement; recovery is whole-job (§5).

---

## §5 Escalation: in-job rebuild, then cluster requeue

When a narrower recovery fails, Rocket escalates to an **in-job rebuild** within the same Ray job, then to a **cluster requeue** (p. 11–12). Table 1 attributes 21 events and 0.5 h each to in-job recovery: 10.5 h, 0.42%, 63,000 GPU-hours. Cluster requeue has no row; it is the baseline the other rows save against.

Open frameworks implement the top of the ladder (whole-job restart from a checkpoint) and differ in automation:

```python
# AReaL@bcef492 areal/infra/launcher/ray.py:L1024–L1033 (RECOVER_TIME_INTERVAL = 10 s at L63)
        recover_states = [JobState.FAILED]
        if isinstance(e, JobException):
            recover_this = (
                e.reason in recover_states
                and run_id < config.recover.retries
                and config.recover.mode in ("on", "auto")
            )
            if recover_this:
                time.sleep(RECOVER_TIME_INTERVAL)
                ray_main(config, run_id=run_id + 1)
```

AReaL relaunches the whole experiment (`ray_main`) up to `recover.retries` times (default 3) after a 10 s wait; `RecoverHandler` saves model, optimizer, RNG, dataloader position and step counters and resumes from the last published generation (`docs/en/reference/checkpointing.md` L143–229). Its DCP recovery checkpoints "require identical parallelism configuration" (L254–255), so the restarted job cannot change its layout.

| Framework | Smallest recovery scope | Automatic? | Checkpoint cadence knob | Source |
|---|---|---|---|---|
| Rocket (Base 2) | one inference replica; learner only | yes | not stated | [[mai-rl-infrastructure]] p. 11–12 |
| Rocket v1 | request retried on another replica | yes | not stated | [[mai-thinking-1]] Sec. 3.6 |
| slime | one SGLang engine; trainer only (serving retained) | engines: yes; trainer: manual resubmit | `--save-interval` | `fault-tolerance.md` L3–53 |
| verl (v1 async) | failed group dropped and refilled; whole job resume | resume: `resume_mode: auto` | `save_freq` (default −1) | `ppo_trainer.yaml` L174, L187; `v1_async_trainer.md` L88 |
| AReaL | whole job | yes, up to 3 retries | `recover.freq_steps / freq_secs` | `ray.py` L1024–1033; `checkpointing.md` L153–159 |
| open-instruct | whole job | no (driver raises) | `checkpoint_state_freq` default 200 | `grpo_utils.py` L225; `grpo_fast.py` L395–415 |
| OlmoRL (paper) | not reported | not reported | not reported | [[olmo-3-rl-pipeline]] |

verl adds one mechanism the others do not document: for preemptible "elastic server instances" with a fixed time budget, it starts a checkpoint early, at "Longest historical step duration + Checkpoint save duration + esi_redundant_time" before shutdown (`ppo_trainer.yaml` L176–181). On recovery, verl's async trainer reissues pending and running groups from their saved prompts and restores finished groups as-is (`v1_async_trainer.md` L98).

---

## §6 The control plane on stable nodes

Rocket runs the Ray head and CPU actors (controller, problem workers, rollout workers, router) on CPU-only nodes, which were "more stable than GPU nodes in our deployments" (p. 12). The reasoning is that every recovery path above depends on the controller being alive: learner recovery "reconnects to the live controller"; inference recovery is decided by it. If the control plane shared nodes with GPU workers, a GPU-node failure could take down the component that coordinates recovery from that failure, forcing a cluster requeue. The post gives no failure rates for CPU versus GPU nodes; the claim is an operational observation.

slime has the same dependency in a different form. Its retained-serving recovery "requires the Ray cluster to remain running" (`fault-tolerance.md` L9), and "if `ServingCluster` or the entire Ray cluster is lost, you must start serving again and recover from a checkpoint" (L127). The doc also warns against launcher scripts that run `pkill sglang` or restart Ray between attempts (L47), because that destroys the state the recovery depends on.

---

## §7 Reading the accounting

Derived quantities from Table 1 (computed here, not stated in the post; they assume the events are spread over the 2,490 h window the table reports):

| Quantity | Value | Computation |
|---|---|---|
| Mean interval between learner recoveries | about 28 h | 2,490 / 88 |
| Mean interval between in-job recoveries | about 119 h | 2,490 / 21 |
| Learner recoveries' share of the fault-tolerance savings | 94% | 176 / (176 + 10.5) |
| Learner downtime actually incurred | about 7.3 h | 88 × 5 min |

The last row uses the "roughly five minutes" figure; it is an order of magnitude, not a measured total. What the table cannot tell you: how often each event type occurred per GPU-hour of each component (no per-component fleet sizes are given for Base 2); what the restart time of a cluster requeue was; and how much model quality, as opposed to wall time, a full restart would have cost. The numbers show that, in MAI's runs, learner-side failures were frequent enough that recovering the learner alone was worth more than all other fault-tolerance mechanisms combined.

---

## Recipe

1. List every component and the state it owns (§1 table). Mark which state is re-sendable, droppable, or checkpoint-only.
2. Make the inference fleet and the learner separate processes with separate allocations, so the learner can be rebuilt alone (slime: separate placement groups; ch-21).
3. Add stall detection with bounded waits for every component: engine probes, learner step-progress watchdog, weight-sync timeout. Do not rely on collective timeouts.
4. On a replica failure, drop its in-flight rollouts (ch-04) and replace it before the next weight transfer, so it rejoins with current weights.
5. On a learner failure, pause admission, kill all learner ranks, rebuild, restore the latest checkpoint, and reconnect to the live controller.
6. Decide what happens to data trained between the last checkpoint and the failure (replay, as slime does, or discard), and write it down.
7. Set a live-fraction threshold for the inference fleet and escalate to an in-job rebuild, then to a cluster requeue.
8. Put the control plane (Ray head, controller, CPU workers) on nodes separate from GPU workers, and make launchers never kill retained processes.
9. Choose the checkpoint interval from the replay cost you accept after a learner failure.
10. Log every recovery event with type, duration and estimated saving, so a Table 1 equivalent can be computed for your own run.

---

## Common mistakes and how to detect them

| Mistake | Symptom | Detection |
|---|---|---|
| Waiting for NCCL timeouts | job idle for the collective timeout after one rank hangs | step-progress watchdog fires before the collective timeout |
| Health probes during weight updates | healthy engines retired after every sync | correlate retirements with weight-transfer times; pause probes during transfers (slime `health_monitor.py` L79–92) |
| Restarted replica serves old weights | log-prob gap spikes on a subset of rollouts | per-replica weight version tag on each rollout; restart only before the next weight update |
| Launcher kills retained serving | every learner failure becomes a cold restart | count of cold restarts vs learner-only recoveries in the event log |
| No replay or discard policy for post-checkpoint data | duplicated or missing training batches after recovery | batch IDs trained per step across a recovery boundary |
| No live-fraction threshold | throughput and staleness degrade for hours after replica losses | live fraction and staleness plotted together (ch-03) |
| Treating Table 1 as measured A/B | over-trusting the 17.37% | note the card records no estimation method for per-event savings |

---

## Check your understanding

1. Why is an inference replica cheaper to replace than a learner rank, in terms of the state each owns?
2. Rocket pauses admission during learner recovery but keeps inference running. Explain what would happen to the queue and to staleness if admission were not paused during a long outage.
3. Why does a stalled rank need a separate detector when a crashed rank does not?
4. slime restarts missing engines "before the next weight update". What failure would occur if a restarted engine rejoined the router between weight updates?
5. The control plane runs on CPU-only nodes. Construct a failure sequence that a co-located control plane would turn from a learner-only recovery into a cluster requeue.
6. Learner recoveries account for 94% of Rocket's fault-tolerance savings in Table 1. What does that suggest about where failures occur in a 6,000-GPU run, and what does the table not let you conclude?

---

## Connections

- Course chapters: ch-02 (components and the rollout store), ch-03 (staleness definition used in §3), ch-04 (drop-failed policy and inference-bound sizing), ch-05 (weight transfer that a replaced replica depends on), ch-21 (elasticity; the runtime-elasticity row of Table 1), ch-22 (capstone recovery row).
- llm-training: ch-45b (multi-turn agentic RL; why long rollouts make restarts expensive).
- Sources: [[mai-rl-infrastructure]] (Table 1, recovery ladder), [[mai-thinking-1]] (Rocket v1 stability mechanisms), [[olmo-3-rl-pipeline]] (no fault-tolerance data reported), [[slime]], [[verl-rollout]], [[areal-async-rl]], [[open-instruct-rlvr-pipeline]].

---

## Sources

- [[mai-rl-infrastructure]] — Rocket fault tolerance (p. 11–12) and Table 1 savings on a 6,000-GPU basis.
- [[mai-thinking-1]] — Rocket v1 stability mechanisms; retry policy; pre-collapse resume note.
- [[olmo-3-rl-pipeline]] — OlmoRL infrastructure; no fault-tolerance mechanism reported.
- [[slime]] — slime framework card; code and docs quoted at 79aa41a (health monitor, training recovery, fault-tolerance guide).
- [[verl-rollout]] — verl rollout card; v1 async trainer doc and `ppo_trainer.yaml` quoted at 8357f1e.
- [[areal-async-rl]] — AReaL card; launcher retry loop and RecoverHandler doc quoted at bcef492.
- [[open-instruct-rlvr-pipeline]] — open-instruct pipeline card; health check and checkpoint code quoted at f020e9a.
- [[async-rollout]] — OpenRLHF async mode, for comparison of a framework with no documented recovery scope.
