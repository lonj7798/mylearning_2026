<!-- chapter: ch-04
     track: rl-pipeline
     kind: content
     title: Staleness Levers: Priority, Dropping Failures, and Inference-Bound Sizing
     deps: [ch-03]
     sources: [[mai-rl-infrastructure]], [[grpo]], [[slime]], [[olmo-3-rl-pipeline]], [[areal-agent-pipelines]], [[scalerl]], [[mai-thinking-1]], [[areal-async-rl]], [[slime-agentic-and-mismatch-examples]], [[open-instruct-rlvr-pipeline]], [[verl-rollout]], [[deepseek-train-infer-consistency]]
     figures: figures/inference-vs-staleness.html
-->

# Chapter 4 — Staleness Levers: Priority, Dropping Failures, and Inference-Bound Sizing

> **Core insight.** ch-03 split a token's staleness into sampler lag plus the time from generation to training measured in learner steps. Rocket attacks the second term with three levers, each aimed at one phase of a rollout's life ([[mai-rl-infrastructure]] p. 7). (1) **Priority to in-flight rollouts.** Every request inherits a priority from its rollout's start time, so a rollout that has already generated tokens is not left waiting behind new ones. Waiting before the first token is free, so new rollouts absorb the queueing. (2) **Drop failed rollouts.** Retrying a failed rollout makes its whole group wait for the retry. Dropping it shortens the wait, at the price of noise or bias in the GRPO baseline, because "the failed rollouts may share features that never make their way to the learner". (3) **Stay inference-bound.** Overallocate learner GPUs so that completed groups do not queue; the Fig. 4 group waited 12 s, and the cost is partly idle learner GPUs. The same formula has a counter-intuitive consequence: adding inference servers at a fixed batch size shortens the time between learner steps while a long rollout's wall-clock time stays the same, so the rollout spans more steps and its tokens become *staler* (p. 7).
>
> **Guideline.** When rollouts of very different lengths share an inference fleet, order requests by rollout start time rather than arrival or batch index, because Rocket reports that without this, multi-turn rollouts "sometimes paused for hours between inference requests while waiting for inference capacity" (p. 7). When you set rollout timeouts, include the pre-first-token wait, because in Fig. 4 every rollout waited 24 minutes before its first request (p. 7), and AReaL's SWE example wraps the whole episode in one timeout and discards the trajectory when it fires (AReaL@bcef492 `examples/swe/agent.py` L185–205). When a rollout fails for an infrastructure reason, drop it rather than retry it, because a retry can add "a large gap" between the retry and the siblings that succeeded first (p. 7). When the failure is caused by the policy's own actions, keep the trajectory with a failure reward, as slime's coding-agent example does for terminal sandbox failures (slime@79aa41a `examples/coding_agent_rl/generate.py` L369–391). When you add inference capacity, scale the batch size with it or tighten the staleness bound, because at fixed batch size the step interval shrinks and staleness per rollout grows in proportion (p. 7; derivation in §5).

---

## Why this chapter matters

The levers in this chapter are configuration and scheduling decisions, not algorithms, and each one has a cost that shows up somewhere else. Priority makes new rollouts wait longer and changes what timeouts mean. Dropping failures reduces staleness but conditions the baseline on survival. Overallocating the learner costs idle GPUs. Adding inference to "speed things up" can make data staler. A team that pulls these levers without understanding which term each one affects will move staleness from one place to another and may not notice.

**Interactive figure.** [figures/inference-vs-staleness.html](figures/inference-vs-staleness.html) lets you scale inference capacity with the batch size fixed or scaled, and shows the step interval, learner utilization, and the staleness each Fig. 2 problem type accumulates while its rollout runs. Use it with §5.

---

## §1 Which lever acts on which term

From ch-03 §2, a token's staleness is (s − v) + (T − s), where the second part accumulates during four phases. Rocket's levers map onto them as follows (p. 6–7):

| Phase where staleness accumulates | Rocket lever | Cost of the lever |
|---|---|---|
| Sampler lag at generation | frequent weight transfer: two-stage, 15 s, every 4 steps (ch-05) | transfer time, engineering |
| Rollout still running (more turns, tool calls, requeues) | priority by rollout start time; auto-scaled in-flight count | new rollouts wait longer before their first token; timeouts must allow for it |
| Completed rollout waits for its group | drop failed rollouts instead of retrying | noise or bias in the GRPO baseline |
| Completed group waits for the learner | overallocate the learner (inference-bound) | learner GPUs partly idle |
| Denominator: time between steps | (no lever; a consequence) | more inference at fixed batch size raises staleness |

---

## §2 Priority to in-flight rollouts

### 2.1 What Rocket does and why

The post's reasoning, in order (p. 7):

1. "A long wait before a rollout's first token is generated does not contribute to staleness."
2. "Once a token is generated, however, it becomes increasingly stale as the rollout performs additional inference calls, tool calls, and requeues."
3. Observed failure: "In early experiments, multi-turn rollouts sometimes paused for hours between inference requests while waiting for inference capacity."
4. Fix: "we prioritize finishing existing rollouts over starting new ones. Each request inherits a priority based on its rollout's start time, so earlier rollouts are served first."
5. Companion control: "We also automatically scale the number of in-flight rollouts up or down to match available inference capacity, keeping queues short under normal conditions."

The causal chain behind step 4: under first-come-first-served scheduling, the next turn of a 200-turn rollout queues behind new rollouts' first turns. Each such wait adds staleness to every token the rollout has already generated. Under start-time priority, that wait is moved onto new rollouts, and for them it costs nothing. The total amount of queueing does not change. The change is in who waits, and that changes the total staleness.

A side effect: priority also bounds how long a rollout runs, because its turns are not delayed by newer work. That bounds the group wait of its siblings (§3).

### 2.2 Consequences for timeouts and resources

"Even with this automatic scaling, new rollouts can wait for a long time before their first request. Each rollout in Figure 4 waits 24 minutes before its first inference request begins. Rollout-level timeouts must account for this wait, and rollouts may hold VMs and other resources before generation starts" (p. 7).

Two concrete problems follow.

1. **A wall-clock timeout started at dispatch includes queueing.** A rollout that queued for 24 minutes has 24 fewer minutes for its own work. Under priority scheduling, the newest rollouts queue longest, so a dispatch-based timeout kills them first. AReaL's SWE workflow shows the pattern: the whole episode is wrapped in `asyncio.wait_for` with a default of 1,800 s (3,600 s in the YAML), and on timeout it logs "Discarding trajectory" and re-raises (AReaL@bcef492 `examples/swe/agent.py` L129, L185–205; [[areal-agent-pipelines]] SWE agent). slime's coding-agent example states that its 3,600 s agent budget "includes inference waiting and tool execution", and its 7,200 s whole-rollout guard "includes sandbox preparation, agent execution, patch extraction, fresh-sandbox grading and trajectory export" (slime@79aa41a `examples/coding_agent_rl/README.md` L227–242). Both budgets count queueing.
2. **Resources held before generation.** A rollout may boot a sandbox VM before its first model call. If it then waits 24 minutes for inference, the VM is held idle. With many in-flight agentic rollouts, the sandbox fleet has to be sized for that holding time, not only for execution time. ch-08 returns to this.

### 2.3 How open frameworks order requests

| Framework | Ordering | What it is keyed on | Source |
|---|---|---|---|
| Rocket | earlier rollouts served first | rollout start time, inherited by each request | p. 7 |
| verl | per-sample priority forwarded to vLLM scheduling | position in the batch (`np.arange`), not age | verl@8357f1e `verl/experimental/agent_loop/agent_loop.py` L1361–1365 |
| slime (buffer) | `--buffer-sort-by-staleness` resumes oldest groups first | minimum weight version of the group's samples | slime@79aa41a `slime/data/data_source.py` L195–196, L248–259 |
| slime (straw queue) | completed groups, then partial groups, then fresh prompts; older versions first within a stage | completion stage, then version | slime@79aa41a `docs/en/advanced/rollout-scheduling.md` L76 |
| AReaL | older trajectories prioritized when a batch is formed | trajectory age | [[areal-async-rl]] §5.1 |

verl's priority is the closest in mechanism, since it reaches the engine's request scheduler. The code assigns it by index:

```python
# verl@8357f1e verl/experimental/agent_loop/agent_loop.py L1361–1365
# Attach per-sample priority to the batch (like ``uid``) so each sample gets
# a globally-unique priority that flows to vLLM request scheduling. Assigned
# before chunking so chunks own disjoint ranges without per-worker offsets.
if "priority" not in prompts.non_tensor_batch:
    prompts.non_tensor_batch["priority"] = np.arange(len(prompts), dtype=np.int64)
```

The priority comes from position within one batch, so it does not by itself order work from different steps by age. slime's buffer ordering uses age, but it acts when groups are taken from the buffer, not on each inference request:

```python
# slime@79aa41a slime/data/data_source.py L248–259
def pop_oldest(args, rollout_id, buffer: list[list[Sample]], num_samples: int) -> list[list[Sample]]:
    """Resume groups with the oldest generated tokens first; preserve ties."""

    def oldest_version(group):
        versions = (str(v) for sample in iter_samples(group) for v in (sample.weight_versions or []))
        return min(
            (int(v) for v in versions if v.isascii() and v.removeprefix("-").isdigit()),
            default=float("inf"),
        )

    buffer.sort(key=oldest_version)
    return pop_first(args, rollout_id, buffer, num_samples)
```

A group with no generated tokens has `oldest_version = inf` and sorts last. This is the same rule as Rocket's: work that has started goes first. slime's docs add that "Ordering does not automatically discard stale samples" (`rollout-scheduling.md` L76). Ordering and bounding are separate controls.

---

## §3 Drop, retry, or keep as a failure

### 3.1 Rocket's choice and its stated cost

"Across a GRPO group, tokens from completed rollouts become stale while waiting for the rest of their group to finish so that training can occur. In Figure 4, the fastest rollout waits 7.6 hours for the slowest one to finish. Retrying failed rollouts can add a large gap between the retry finishing and the rollouts that succeeded on the first try. We therefore drop failed rollouts rather than hold back the group's training data. This reduces staleness, but dropping failed rollouts can add noise to or bias the baseline for GRPO's group advantage, since the failed rollouts may share features that never make their way to the learner" (p. 7).

This reverses Rocket v1. MAI-Thinking-1's problem worker sent 16 then 128 requests, "retrying failed requests as needed" ([[mai-thinking-1]] Sec. 3.6.2). The post does not give a measured comparison between the two policies.

"Failed" here means the rollout did not complete: an infrastructure or execution failure. A rollout that completes with a wrong answer is not failed. It gets a low reward and stays in the group.

### 3.2 Why a retry costs a full rollout of staleness

A retried rollout starts from its first token again. Its duration is a fresh draw from the task's duration distribution, and it starts late. For the siblings, the group-completion time becomes (failure time + retry duration) instead of the original rollout's duration. For an SWE task with a 93-minute median (Fig. 2), one retry can add on the order of the median duration to the wait of every completed sibling. Under the step interval Δt, that is 93 min / Δt extra learner steps of staleness for all of them.

### 3.3 Why dropping changes the baseline

The GRPO advantage of rollout i is (R_i − mean(R)) / std(R) over the group ([[grpo]] §4.1.1; llm-training ch-40). If some rollouts are dropped, the mean and std are taken over the survivors. Two cases:

- **Failure independent of reward.** The survivors are a random subset. The baseline is unbiased but estimated from fewer samples, so it is noisier. This is the "noise" in the post's sentence.
- **Failure correlated with reward.** The survivors are a selected subset. The baseline estimates E[R | x, survived], not E[R | x]. Suppose long attempts are both more likely to fail (timeouts, VM loss) and more likely to be correct. Then the survivors' mean is lower than the true mean, and every survivor's advantage is shifted up. The failed attempts themselves get no gradient, positive or negative. This is the "bias", and the post's phrase "features that never make their way to the learner" describes it.

The example in the second case is an illustration of the mechanism. The post does not report which direction the correlation has in its runs, or its size.

DeepSeek-V4 makes the same survival argument for interrupted rollouts. Regenerating a preempted rollout from scratch "is mathematically incorrect" because "shorter responses are more likely to survive interruption", so DeepSeek resumes from a token-granular write-ahead log instead ([[deepseek-train-infer-consistency]] Sec. 5.2.3). Dropping and regenerating both condition the training data on survival. Resuming does not, but it requires saving and restoring the full rollout state.

### 3.4 Three policies in open code

**Keep as a failure (slime coding agent).** When the sandbox terminates after the agent has acted, the trajectory is kept with reward 0:

```python
# slime@79aa41a examples/coding_agent_rl/generate.py L369–391
except SandboxTerminatedError as error:
    # A terminal sandbox failure is an observed trajectory outcome. Keep
    # its sampled actions and train them with zero reward instead of
    # discarding the trajectory and retrying until execution succeeds.
    metadata = {
        "grading_solved": False,
        "instance_id": instance_id,
        "agent": agent_name,
        "agent_error": str(error),
        "agent_error_type": "sandbox_terminated",
    }
    if evaluation:
        samples = _eval_result(
            base_sample, reward=0.0, applied_cleanly=False, agent_exit_code=None, instance_id=instance_id
        )
        samples[0].metadata.update(metadata)
    else:
        with trace_span(base_sample, "trajectory_finalize", record_error=False):
            samples = await adapter.finish_session(
                session_id, base_sample=base_sample, reward=0.0, extra_metadata=metadata
            )
        if not samples:
            return _abort_result(base_sample, "sandbox_terminated_before_generation", instance_id)
```

The comment gives the reasoning: the failure is "an observed trajectory outcome". If the agent's own commands killed the sandbox, a zero reward teaches it not to do that. If an infrastructure fault killed the sandbox, the same zero reward penalizes actions that did nothing wrong. The code aborts (loss mask `[0]`, `remove_sample = True`) only when no generation happened (`_abort_result`, L446–456).

**Drop and refill the group (verl V1).** A training group is evicted, and one replacement prompt is fetched, when "its rollout finishes with `failure`", when it is stale beyond `max_off_policy_threshold` under `drop`, or when the DAPO filter rejects it (verl@8357f1e `docs/advance/v1_async_trainer.md` L84–92). The eviction unit is the prompt group, not the single rollout. This removes the baseline-bias problem of §3.3 for that prompt, because no partial group is trained. It throws away the completed siblings' compute, and it changes the prompt distribution if some prompts fail more often than others.

**Requeue (slime fully async).** "A generator that does return an ABORTED/PENDING result is requeued instead of sent to training", and "segmented agent failures retry their original inputs" (slime@79aa41a `slime/rollout/fully_async_rollout.py` L22–25). This is retry, with the staleness cost of §3.2.

| Policy | Staleness | Baseline | Wasted compute | Who uses it |
|---|---|---|---|---|
| Retry | adds a fresh rollout duration to the group wait | unbiased if retries eventually succeed | the failed attempt | Rocket v1; slime fully async (segmented agents) |
| Drop the rollout | lowest | noisy; biased if failure correlates with reward | the failed attempt | Rocket (Base 2) |
| Drop the group, refill | low for the batch | unbiased for the remaining groups; prompt mix shifts | all completed siblings | verl V1 |
| Keep with reward 0 | lowest | biased toward penalizing failure-prone behaviour | none | slime coding agent |
| Resume from saved state | none added beyond the outage | unbiased | none | DeepSeek-V4 (preemption) |

---

## §4 Keeping the job inference-bound

### 4.1 Rocket's statement

"Completed rollouts that wait in a long internal queue before they're sent to the learner accumulate further staleness. We overallocate GPUs to our learner so that RL jobs are bottlenecked by inference capacity, not training capacity. In Figure 4, the completed group spends just 12 seconds in the controller queue before it is trained on. The trade-off is that the learner's GPUs sit partially idle" (p. 7).

The reasoning is queueing. If the learner consumes batches more slowly than inference produces them, a backlog forms and every queued token ages. If the learner is faster, the queue stays near empty and the learner waits. Rocket chooses the second case and pays for it in learner idle time. slime's docs state the same queueing fact: with production rate λ above consumption rate μ, "an unbounded backlog grows at roughly λ − μ; backpressure is needed" (slime@79aa41a `docs/en/advanced/rollout-scheduling.md` L78).

### 4.2 How large the inference share is in practice

| System | Inference : learner | Observed learner state | Source |
|---|---|---|---|
| Rocket v1, largest job | 4,096 : 768 GB300 chips; "can be as large as 5:1" | not stated | [[mai-thinking-1]] Sec. 3.6.4 |
| Olmo 3 32B Think | 20 : 8 nodes (≈5× GPU use) | learner waited for data 75% of the time; a representative step is 1,000 s with 125 s of training | [[olmo-3-rl-pipeline]] §4.4.3 |
| Olmo 3 7B | 7 : 2 nodes (≈14×) | not stated | same |
| AReaL | three quarters of devices to inference, chosen over 50–50 in early experiments | not stated | [[areal-async-rl]] §7.1 |
| Rocket (Base 2) | not stated; learner overallocated | 12 s controller queue in Fig. 4 | p. 7 |

Olmo 3 shows what inference-bound means quantitatively: 125 s of training in a 1,000 s step means the learner is busy 12.5% of the time. Its report presents this as the bottleneck it then optimized on the generation side ([[olmo-3-rl-pipeline]] Core Insight). Rocket presents the same state as a deliberate choice.

### 4.3 Open-framework mechanisms at the same boundary

- **open-instruct blocks generation when the learner falls behind** and logs "Consider increasing training compute" (open-instruct@f020e9a `open_instruct/data_loader.py` L1427–1430; quoted in ch-02). Its default response to a learner-bound job is to stop producing, which bounds the queue at `async_steps` batches.
- **verl V1 lends idle trainer GPUs to rollout.** With `hybrid_rollout.enable_switch`, hybrid replicas on the training GPUs join the rollout load balancer while the trainer waits for its next batch (verl@8357f1e `docs/advance/v1_async_trainer.md` L125–130). On 3 × 8 H100 with Qwen3.5-35B-A3B, 100 steps took 13.15 h without switching and 11.79 h with it (−10.3%), tokens/s rose from 14,604 to 16,445, and mean reward was 0.7755 vs 0.7762 (L308–322). This recovers some of the idle learner time that Rocket accepts. The mechanism adds the switching cost, and the doc notes that it "cannot be combined with rollout PD disaggregation" (L170).

---

## §5 More inference at fixed batch size raises staleness

### 5.1 The statement

"The formula's denominator has a counterintuitive consequence: adding inference servers while leaving batch size unchanged can increase staleness. With more inference servers, it takes less time to collect enough rollouts for a training step. Since our jobs are typically inference-bound, this shortens the time between training steps. More training steps can be completed in the time it takes for a rollout to finish, even if the rollout's wall-clock time is unchanged" (p. 7).

### 5.2 Derivation

Let N be inference capacity (servers), B the batch size, and D a rollout's wall-clock duration.

1. **Inference-bound step interval.** If the learner waits for data, the step interval is the time to produce one batch. With throughput proportional to N, Δt ∝ B / N.
2. **Rollout duration.** D is set by the rollout's sequential turns: decode latency per turn, tool latency, and waits. If queues are already short (Rocket auto-scales in-flight count to keep them short, p. 7), adding servers does not shorten D.
3. **Staleness accumulated while the rollout runs.** The waiting term for a token generated near the start is about D / Δt ∝ D · N / B.
4. **Result.** Doubling N at fixed B halves Δt and doubles the staleness each rollout accumulates. A fixed-duration SWE rollout spans twice as many learner steps.

The post gives no measurement of this effect. It is an analytical consequence of the decomposition, labelled "counterintuitive" by the authors.

### 5.3 Where the derivation stops holding

- **Learner-bound regime.** Once Δt reaches the learner's own step time, more inference no longer shortens it. Δt = max(c · B / N, T_train) for some constant c. Past that point, extra inference does not raise staleness through the denominator. It fills the controller queue instead, which raises staleness through the queue term, and inference becomes the idle resource.
- **Queue-dominated rollouts.** If D is dominated by waiting for inference capacity (Rocket's "early experiments", p. 7), more servers shorten D. Then D / Δt can stay flat or fall. The statement holds "even if the rollout's wall-clock time is unchanged", which is the case after priority and auto-scaling have removed most queueing.
- **Scaling the batch.** If B grows with N, Δt stays constant and so does staleness. The extra capacity then gives more samples per step rather than more steps. That is not a neutral change to the optimizer. ScaleRL reports that a 2,048-prompt batch reached a higher fitted asymptote than smaller batches, while smaller batches looked better early ([[scalerl]] Fig. 10). Batch size has effects on learning beyond staleness.

### 5.4 Practical reading

When you request more inference GPUs for an inference-bound run, decide in advance what the new capacity is for: more steps per hour (at higher staleness) or larger batches (at the same staleness). Then set the staleness bound (ch-03 §5) and the batch size together. A runtime-configurable staleness limit and batch packing are both on Rocket's list of typed runtime settings (p. 13; ch-21), which is consistent with changing them together when capacity changes.

---

## Recipe

1. **Measure the four waiting phases** per group: pre-first-token wait, generation time, group-completion wait (fastest sibling), controller-queue time. Fig. 4 is the template.
2. **Order inference requests by rollout start time.** If your engine supports request priority (verl forwards `priority` to vLLM), set it from the rollout's first-token timestamp, not from batch position.
3. **Auto-scale in-flight rollouts** to inference capacity so that queues stay short, and log queue length per replica.
4. **Start rollout timeouts at the first sampled token,** or set dispatch-based timeouts to median duration + expected pre-first-token wait. Do not boot sandboxes until generation is about to start, if the environment allows it.
5. **Classify failures.** Infrastructure failures (node loss, sandbox fleet error, inference crash): drop the rollout. Policy-caused failures (agent killed its own sandbox, malformed tool call): keep with a failure reward. Log the failure rate per task type and per reward bucket.
6. **Check for survival bias.** Compare the reward distribution of groups with and without dropped rollouts on the same prompts. If they differ, consider dropping the whole group (verl V1) for those task types.
7. **Size the learner so the controller queue stays near zero** (Rocket: 12 s). Log learner busy fraction (Olmo 3: 125 s of 1,000 s) to know what the choice costs.
8. **When adding inference capacity, change batch size or the staleness bound in the same change,** and record which one you chose.

---

## Common mistakes and how to detect them

| Mistake | Observable symptom | Check |
|---|---|---|
| FIFO request scheduling with mixed-length rollouts | Multi-turn rollouts stall for long periods between turns; staleness tail grows | Log per-turn queue time by rollout age (Rocket saw hours-long pauses, p. 7) |
| Dispatch-based timeouts under priority scheduling | Newest rollouts time out at a high rate without doing work | Log time to first token vs timeout; count timeouts with zero generated tokens |
| Retrying failed agentic rollouts by default | Fast siblings wait for retries; group-wait time has a second mode near one rollout duration | Histogram of group-completion wait; mark groups with retries |
| Dropping failures without checking correlation | Baseline biased; reward trends on survivors differ from full-group evaluation | Failure rate by task type and by partial-reward bucket |
| Treating policy-caused failures as infrastructure noise | The policy keeps the behaviour that crashes environments | Classify failure causes; keep policy-caused ones with a failure reward (slime coding agent) |
| Learner-bound job | Controller queue grows; staleness rises with no change in rollout length | Controller-queue time per group; learner busy fraction near 100% |
| Adding inference at fixed batch size to "go faster" | Steps per hour rise, staleness per rollout rises in proportion | Recompute D / Δt per task type before and after; see the figure |

---

## Check your understanding

1. Rocket gives priority by rollout start time. Explain why this reduces total staleness even though the total amount of queueing in the system is unchanged.
2. Why does priority scheduling make dispatch-based rollout timeouts kill the newest rollouts first? Propose a timeout definition that avoids this, and state what resource it still fails to protect.
3. Dropping a failed rollout "can add noise to or bias the baseline". Give a concrete case in which failure probability correlates with reward, and work out the sign of the shift in the survivors' advantages.
4. slime keeps a sandbox-terminated trajectory with reward 0, while Rocket drops failed rollouts. Under what condition on the failure's cause is each choice correct, and why?
5. verl V1 evicts the whole prompt group when a rollout fails. Which problem from §3.3 does this remove, and what new distortion does it introduce?
6. Derive why adding inference at fixed batch size raises staleness, and identify the two assumptions in the derivation. For each, describe a regime in which it fails and what happens to staleness there.
7. Olmo 3's learner trained for 125 s of a 1,000 s step. Explain why Rocket would call this state a design choice rather than a bottleneck, and what Rocket gives up by accepting it.

---

## Connections

- **Previous: ch-03** — the staleness decomposition these levers act on.
- **Next: ch-05** — the remaining lever, weight transfer, which bounds sampler lag.
- **ch-08** — agentic rollouts that hold sandboxes and VMs before their first token.
- **ch-10** — group size and rollout length, which set how long the group barrier lasts.
- **ch-12** — prompt selection; dropping groups changes the effective prompt mix.
- **ch-21** — staleness limits and batch packing as runtime-configurable settings.
- **llm-training ch-40** — GRPO group advantages, the baseline that dropping failures alters.
- **llm-training ch-44a** — length in RL; length and failure rates are often correlated.
- **llm-training ch-45b** — multi-turn agentic RL and trajectory-level masking.
- **llm-training ch-16** — prompt distribution, which group-level eviction shifts.

---

## Sources

- [[mai-rl-infrastructure]] — the three levers and the inference-vs-staleness consequence (p. 7); Fig. 2 durations; Fig. 4 waits.
- [[mai-thinking-1]] — Rocket v1 retry policy, inference:learner ratio up to 5:1, 4,096 / 768 chips.
- [[grpo]] — the group-mean baseline.
- [[slime]] — `pop_oldest` buffer ordering; queueing note; slime@79aa41a.
- [[slime-agentic-and-mismatch-examples]] — coding-agent failure handling and timeouts; fully async requeue.
- [[verl-rollout]] — request priority by batch index; V1 failure eviction and hybrid switching (verl@8357f1e docs).
- [[areal-agent-pipelines]] — SWE episode timeout that discards the trajectory; AReaL@bcef492.
- [[areal-async-rl]] — 3:1 inference:training split, older-first batch formation.
- [[olmo-3-rl-pipeline]] — 20:8 nodes, learner idle 75%, 1,000 s step with 125 s training.
- [[open-instruct-rlvr-pipeline]] — generation blocked when the learner is behind.
- [[scalerl]] — batch size changes the fitted asymptote, so scaling batch with inference is not neutral.
- [[deepseek-train-infer-consistency]] — survival bias of regenerating interrupted rollouts; WAL resume.
