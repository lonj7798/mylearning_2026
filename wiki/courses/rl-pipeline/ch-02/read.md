<!-- chapter: ch-02
     track: rl-pipeline
     kind: content
     title: Anatomy of the Async Loop: Controller, Workers, Sampler, Learner
     deps: [ch-01]
     sources: [[mai-rl-infrastructure]], [[mai-thinking-1]], [[slime]], [[verl-rollout]], [[areal-async-rl]], [[async-rollout]], [[olmo-3-rl-pipeline]], [[open-instruct-rlvr-pipeline]], [[pipelinerl]], [[slime-agentic-and-mismatch-examples]], [[areal-agent-pipelines]]
     figures: figures/async-loop.html
-->

# Chapter 2 — Anatomy of the Async Loop: Controller, Workers, Sampler, Learner

> **Core insight.** Rocket splits one GRPO step into components that each own one unit of work and run concurrently: the controller owns the curriculum and batch formation, a problem worker owns one GRPO group, a rollout worker owns one trajectory and its agent loop, the sampler (router + SGLang) owns token generation, and the learner owns the policy update ([[mai-rl-infrastructure]] Fig. 1, p. 2–4). Data does not flow along the control path. Trajectory payloads go to a rollout store, scores and references go back up to the problem worker, and the batches the controller sends to the learner carry only "segment references and advantages"; the learner fetches "the corresponding token IDs and sampling log probabilities from the rollout store" (p. 3). Because no component waits for a global barrier, GPUs stay busy while rollouts of very different lengths run, and the price is staleness (ch-03). Open designs place the same pieces differently: verl's HYBRID mode and slime's `--colocate` time-share GPUs and synchronize weights each step, while slime's fully async rollout, AReaL, PipelineRL and open-instruct decouple generation from training, as Rocket does.
>
> **Guideline.** When rollout lengths within one batch differ by more than an order of magnitude, as Rocket's medians do (2 min to 93 min, Fig. 2), use disaggregated generation with a persistent pool of in-flight rollouts instead of a generate-then-train loop, because verl's V1 benchmark measured 22.79 h for 150 synchronous steps against 14.10 h with separate async rollout on the same four nodes (verl@8357f1e `docs/advance/v1_async_trainer.md` L288–292), and AReaL measured 41.0 h against 14.8 h at equal AIME24 accuracy ([[areal-async-rl]] Table 1). When graders need several rollouts at once (pairwise LLM comparison), run them where the group is assembled (the problem worker), and run isolated graders next to the trajectory (the rollout worker), because MAI-Thinking-1 places them exactly this way ([[mai-thinking-1]] Sec. 3.6.3). When trajectories are long, pass references to stored payloads through the control path instead of token tensors, as Rocket's controller does (p. 3) and as slime's trainer does with a Ray object reference (slime@79aa41a `train.py` L51).

---

## Why this chapter matters

ch-03 to ch-08 each change one box of Fig. 1. Staleness depends on where rollouts wait. Weight transfer depends on who triggers it. Precision and parity live in the sampler and the learner. Agentic tools live in the rollout worker. Fault tolerance depends on which boxes can restart alone. You need the full data path in your head before any of those chapters makes sense.

The open frameworks use different names for the same jobs: rollout manager, data buffer, agent loop, DataPreparationActor, rollout controller, replay buffer. Mapping them to Fig. 1 lets you read their code and their configuration flags against one reference.

**Interactive figure.** [figures/async-loop.html](figures/async-loop.html) steps through the Fig. 1 flow one message at a time, then shows the colocated synchronous loop for contrast. Open it before §1.

---

## §1 The loop, step by step

The post numbers the components (1)–(6) in Fig. 1 and walks through them in prose (p. 2–4). In order:

1. **Controller (1) selects a problem.** The controller "selects a problem from the curriculum and sends it to a problem worker". In Fig. 1 the controller box lists "Select problems", "Admit work" and "Filter + pack". It runs on the CPU pool.
2. **Problem worker (2) dispatches rollout requests.** The problem worker "dispatches a set of rollout requests to the rollout worker pool". "Each problem and rollout worker handles many requests concurrently" (p. 2).
3. **Rollout worker (3) runs the agent loop.** "Each rollout worker tracks the state of the ongoing rollouts it manages" (p. 3).
4. **Sampler calls (4).** "For agentic tasks, an agent loop requests model responses from the sampler." Fig. 1 shows the sampler as a CPU router in front of SGLang GPU servers doing prefill and decode. The arrow into the sampler is labelled "Context + settings". The arrow back is labelled "Tokens / logprobs / versions".
5. **Tool calls to environments (5).** The agent loop "routes tool calls through the tool dispatcher to the task environment". The agent "can execute actions in sandbox execution environments (SEE), query search services, or invoke tools exposed by MCP servers". Tool results become context for the next model call, and the loop continues "until the agent finishes or reaches a configured limit" (p. 3).
6. **Per-rollout grading.** "Per-rollout graders then evaluate the trajectory and, where needed, the environment state" (p. 3).
7. **Store the payload, return references.** "The rollout worker (3) stores the trajectory payload and returns scores and references to the stored rollouts to the problem worker (2)" (p. 3).
8. **Group grading and advantages.** The problem worker "collects a problem's rollouts, runs configured group graders on all the rollouts together, and then computes advantages for all rollouts in the GRPO group" (p. 3).
9. **Filter and pack.** The problem worker returns results to the controller (1), "which selects eligible data and packs training segments into learner batches" (p. 3).
10. **Learner fetch (6).** "The batches carry segment references and advantages; the learner (6) retrieves the corresponding token IDs and sampling log probabilities from the rollout store" (p. 3).
11. **Update and refresh.** "The learner updates the policy, and the controller periodically initiates a weight transfer to refresh the sampler" (p. 4).
12. **Supervision.** "Recovery checkpoints preserve both learner and controller state. Runtime management monitors worker health and coordinates recovery and resource allocation" (p. 4).

Placement of each component, from Fig. 1:

| Component | Hardware pool | Holds | Receives | Emits |
|---|---|---|---|---|
| Controller (1) | CPU | curriculum, admission state, eligible results | problems from the mixture; results + advantages | problems; segment refs + advantages; weight-transfer trigger |
| Problem worker (2) | CPU | one GRPO group per problem | a problem; scores + rollout refs | rollout requests; results + advantages |
| Rollout worker (3) | CPU | per-trajectory agent state | rollout requests; tokens; observations | sampler requests; tool actions; payloads; scores + refs |
| Router + SGLang (4) | CPU router, GPU servers | weights, KV caches | context + settings | tokens / logprobs / versions |
| Environments (5) | sandboxes, external tool services | tool state | actions / checks | observations |
| Rollout store | not stated | tokens / logprobs; model-turn payloads | payloads from rollout workers | payloads to the learner |
| Learner (6) | GPU | policy and optimizer state | segment refs + advantages | updated weights |

The post does not say what hardware the rollout store runs on. Fig. 1 does not place it in a CPU or GPU pool.

---

## §2 The problem worker owns the group

A GRPO advantage needs every reward in the group, so some component has to collect them. In Rocket that component is the problem worker, which is why each GRPO group has a single owner.

MAI-Thinking-1 gives the Rocket v1 flow for one problem ([[mai-thinking-1]] Sec. 3.6.2, p. 50–51):

1. **Early-exit rollouts.** Send 16 requests to the rollout workers, "retrying failed requests as needed". If the pass rate is in a set interval, continue; otherwise abort the task.
2. **Full rollouts.** Send 128 more requests, again with retries. If the pass rate is in a set interval, continue; otherwise abort.
3. **Postprocessing.** Compute normalized advantages; "this step may include applying length penalties for long rollouts, applying GRPO reward normalization, or other heuristics".

The hyperparameters behind these numbers are G = 128, G_early = 16, early range [0.05, 0.8], full range [0.1, 0.8] (Sec. 3.1.5). The pass-rate filter is a property of the group, so it also has to run where the group is assembled.

Two changes in the Base 2 post matter for later chapters. First, failed rollouts are now dropped rather than retried (p. 7; ch-04). Second, group graders run on the problem worker "on all the rollouts together" (p. 3). MAI-Thinking-1 gives the rule for where grading runs: "If individual rollouts can be evaluated in isolation, grading is performed by the rollout worker", while graders that "need to look at the answers for multiple rollouts simultaneously, e.g., so that they can ask an LLM to perform pairwise comparisons" run on the problem worker (Sec. 3.6.3). ch-11 covers grader design. The placement rule follows from data dependency: a pairwise grader cannot run before both rollouts exist.

**Open counterparts.** slime collects groups in its rollout function. It submits prompts, waits on `asyncio.FIRST_COMPLETED`, checks that each finished group holds `n_samples_per_prompt` samples, and applies the dynamic filter to the group as a unit (slime@79aa41a `slime/rollout/sglang_rollout.py` L530–571). verl V1's replay buffer treats "one prompt group" as the refill unit (verl@8357f1e `docs/advance/v1_async_trainer.md` L82). In each case the group is the smallest unit that can be accepted, filtered, or dropped.

---

## §3 The rollout worker and the sampler

### 3.1 Rollout worker

MAI-Thinking-1 describes the rollout-worker loop: it builds an initial prompt, sends it to an inference server, parses the response, "performs any tool calls, and incorporates the results into a follow-up request". This continues "until the model stops making tool calls and outputs a final answer, or exceeds some pre-determined limit on the number of steps or the amount of time the rollout can take" ([[mai-thinking-1]] Sec. 3.6.3, p. 51).

A worker holds many trajectories at once (p. 2), so its state is a table of in-flight rollouts. Each entry holds the conversation so far, the pending tool call, and the sampler versions seen. Turn counts reach 249 at the median for software engineering (Fig. 2) and 784 for the slowest rollout in Fig. 4. A worker therefore spends most of its wall-clock time waiting on the sampler or on tools. This is why rollout workers sit on a CPU pool.

### 3.2 Sampler

The sampler returns "Tokens / logprobs / versions" (Fig. 1 label). The version label lets staleness be measured per token (ch-03). The log-probs are what the learner later reads from the store, so the learner can compare its own probabilities with the sampler's.

Rocket v1 describes the serving side ([[mai-thinking-1]] Sec. 3.6.4, p. 51–52):

- SGLang workers behind the SGLang router, which "handle[s] load balancing, traffic control, and prefix caching".
- For single-turn workloads, prefix caching is disabled "so that sliding-window-attention tokens can be fully evicted during long generations". For multi-turn workloads, "our prefix cache hit rate reaches 97–98% on production RL runs".
- "When a rollout worker's request to an inference server fails mid-flight, it is retried against another replica, so individual replica failures degrade pool throughput but do not lose rollouts."

The last point separates two failure scopes. One request that fails inside the sampler is retried, and no rollout is lost. A rollout that fails as a whole is dropped in Base 2 (ch-04).

**Open counterparts.** verl's async vLLM server is token-in/token-out: it takes prompt token ids and returns token ids, per-token log-probs, and optionally routed experts ([[verl-rollout]] Key Contributions). Its `LLMServerClient` uses least-in-flight load balancing and sticky sessions, so later turns reach the same server for prefix caching ([[verl-rollout]] Findings). The multi-turn agent loop sits in `verl/experimental/agent_loop/`. slime's coding-agent example runs an HTTP adapter per agent CLI that "turns the CLI's model calls into SGLang token calls and records the trajectory" ([[slime-agentic-and-mismatch-examples]] Pipeline step 3). AReaL lets the agent call the policy through an OpenAI-compatible client and exports interactions with loss masks ([[areal-agent-pipelines]] Guideline). Each of these puts a recording layer between the agent and the inference engine, as Rocket's rollout worker does.

---

## §4 The rollout store and reference batches

Rocket's learner batches carry "segment references and advantages", not token data (p. 3). The learner fetches token IDs and sampling log-probs from the store.

**The reason is not stated.** The post describes the design and does not say why it was chosen. Two consequences follow from the description itself:

1. **The controller stays small.** The controller is a CPU actor (Fig. 1). If every token of every trajectory passed through it, its memory and network load would scale with total generated tokens. With references, its load scales with the number of segments.
2. **The learner reads exactly what the sampler produced.** The stored values are the sampler's own token IDs and log-probs. The learner never re-tokenizes text, and the sampler-side log-probs are available for any ratio or correction computed in the loss. verl's agent-loop docs report the failure mode this avoids: re-applying the chat template to the final messages "make PPO training not even converged in single-turn" ([[verl-rollout]] Findings; verl `docs/advance/agent_loop.rst`).

These two points are consequences of the stated design. They are not reasons the authors gave.

**Open counterparts.** slime passes a reference as well. The rollout manager returns a Ray object reference, and the actor trains from it:

```python
# slime@79aa41a train.py L51, L67
rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))
...
ray.get(actor_model.async_train(rollout_id, rollout_data_ref))
```

verl's V1 async trainer moves samples through a `TransferQueue` with an asynchronous replay buffer (verl@8357f1e `docs/advance/v1_async_trainer.md` L10). AReaL's trainers read from a replay buffer filled by rollout workers ([[areal-async-rl]] §4.1). open-instruct's `DataPreparationActor` scores, filters and packs batches before handing them to the learner ([[open-instruct-rlvr-pipeline]] Summary). The common pattern is that a CPU-side component decides what is trained, and the tensors travel by a separate path.

---

## §5 Training segments

"Each segment is a training example containing one or more model turns" (p. 3). A trajectory is therefore not necessarily one training example. A 249-turn software-engineering rollout can be split into several segments.

The post does not say how segments are cut or why. Two constraints in the open frameworks show what segmentation must respect:

- **Token alignment.** AReaL's concat exporter builds one sequence per trajectory, and it refuses to merge a child turn whose prompt does not preserve the parent's sampled token prefix ([[areal-agent-pipelines]] Guideline; AReaL@bcef492 `areal/experimental/openai/types.py` L190–232). When the prefix is not preserved (for example, a context-management step rewrote earlier turns), the turns must become separate training examples.
- **Loss masks.** In verl's `ToolAgentLoop`, model-generated tokens get response_mask 1 and tool or template tokens get mask 0 ([[verl-rollout]] Core Insight). Any segment must carry these masks for its own span.

slime's fully async rollout also distinguishes segmented agent trajectories. "Segmented agent failures retry their original inputs because their closed sandbox sessions cannot be restored from tokens" (slime@79aa41a `slime/rollout/fully_async_rollout.py` L24–25). ch-08 returns to multi-turn segments and masking. ch-45b of llm-training covers the masking mechanics.

---

## §6 Three placements in open frameworks

Rocket is one point in a design space with two axes. The first axis is whether learner and sampler share GPUs. The second is whether a training step waits for a fixed batch to be generated with its current weights.

### 6.1 Colocated, synchronous

verl's `HYBRID` mode fuses rollout and training "in same process" and they "share GPUs, switch context with weight synchronization", with the stated use case "on-policy training" (verl@8357f1e `verl/workers/rollout/replica.py` L68–72; full enum quoted in ch-01). slime's `--colocate` puts the rollout engines on the actor GPUs and forces offload of both sides by default (slime@79aa41a `slime/utils/arguments.py` L2141–2154). The step is generate → (offload rollout) → train → pause admission → update weights → resume (`train.py` L47–93).

The cost model in slime's docs: a serial round takes `t_sync = t_R + t_T + t_W` (rollout + reward, training, weight update), and the optimistic async lower bound is `max(t_R, t_T) + t_W`. The docs add: "Variable-length samples make synchronous barriers especially costly" (slime@79aa41a `docs/en/advanced/rl-systems.md` L21–31). This is a bound, not a measured speedup.

### 6.2 Disaggregated, bounded asynchrony

Generation runs on its own GPUs, and a bound limits how far it may run ahead of training. open-instruct's data-preparation loop shows the simplest version:

```python
# open-instruct@f020e9a open_instruct/data_loader.py L1412–1431
num_initial_prompts = self.config.async_steps * self.global_batch_size
logger.info(f"[DataPreparationActor] Pushing {num_initial_prompts} initial prompts to param_prompt_Q")
for _ in range(num_initial_prompts):
    add_prompt_to_generator(
        next(self.iter_dataloader),
        self.iter_dataloader._epoch,
        self.param_prompt_Q,
        self.generation_config,
        is_eval=False,
        base_env_config=self.base_env_config,
        ground_truth_overrides=self.ground_truth_overrides,
    )

while self.training_step < self.num_training_steps:
    generation_wait_start_time = time.perf_counter()
    while self.training_step - self._last_consumed_step > self.config.async_steps:
        logger.info(
            f"[DataPreparationActor] Step {self.training_step}: waiting for step {self._last_consumed_step + self.config.async_steps} to be consumed. Consider increasing training compute."
        )
        time.sleep(0.1)
```

The queue is pre-filled with `async_steps × global_batch_size` prompts, and production blocks when generation is more than `async_steps` ahead (default 8, `data_loader.py` L415). The log message "Consider increasing training compute" is the open-instruct form of Rocket's advice to stay inference-bound (ch-04). Olmo 3's 7B Think run used `async_steps 1` and the other columns used 8 ([[olmo-3-rl-pipeline]] Table 49).

AReaL bounds asynchrony at admission time with a staleness parameter η: η = 0 is synchronous, and the defaults are 4 for code and 8 for math ([[areal-async-rl]] Technical Details 1–2). ch-03 quotes its capacity code.

### 6.3 Fully asynchronous with a persistent in-flight pool

slime's fully async rollout keeps a pool of trajectories alive across training-step boundaries:

```python
# slime@79aa41a slime/rollout/fully_async_rollout.py L1–8, L20–29
"""Fully-async rollout for slime.

Decouples ``max_concurrent_tasks`` from ``rollout_batch_size``: a background
asyncio worker keeps a fixed pool of in-flight trajectories across rollout
boundaries, so the next training step doesn't have to wait for the slowest
in-flight sample to finish.

Use with ``--rollout-function-path slime.rollout.fully_async_rollout.generate_rollout_fully_async``.
...
Admission pause leaves in-flight tasks alive across training rounds. Agent
adapters with ``--partial-rollout`` continue interrupted model turns within
those tasks, retaining the live sandbox and CLI. A generator that does return
an ABORTED/PENDING result is requeued instead of sent to training. Flat samples
retain their partial response; segmented agent failures retry their original
inputs because their closed sandbox sessions cannot be restored from tokens.
Completed trajectories in the same group retain their results across retries.

``--dynamic-sampling-filter-path`` is honoured DAPO-style: rejected groups do
not count toward ``rollout_batch_size`` and are replaced from the warm queue.
"""
```

This is the closest open analogue to Rocket's loop. The pool corresponds to Rocket's in-flight rollouts. The admission pause during weight update corresponds to Rocket's admission control. Group completion still gates training. PipelineRL takes one more step: generation does not pause for training at all, and new weights are loaded "between decoding steps without discarding in-progress sequences" ([[pipelinerl]] Core Insight; ch-05).

### 6.4 Measured effect of decoupling

| System | Setting | Synchronous | Decoupled | Accuracy | Source |
|---|---|---|---|---|---|
| verl V1 | Qwen3.5-35B-A3B, DAPO-Math-17k, 4 nodes, 150 steps | 22.79 h (sync) | 14.72 h colocate_async; 14.10 h separate_async | per-step quality shown only as figures | verl@8357f1e `docs/advance/v1_async_trainer.md` L280–292 |
| AReaL | 1.5B math, 16 nodes, 250 steps | 41.0 h | 14.8 h | AIME24 42.0 vs 42.2 | [[areal-async-rl]] Table 1 |
| AReaL | 7B math, 24 nodes | 57.7 h | 25.4 h | 63.0 vs 63.1 | [[areal-async-rl]] Table 1 |
| PipelineRL | Qwen2.5-7B math, 128 GPUs | G = 32 conventional | in-flight | same reward about 2× faster | [[pipelinerl]] §5, Fig. 5 |
| Olmo 3 infra | fixed 2-node benchmark | 881 tok/s (OLMo 2) | 2,949 tok/s with continuous batching, threading, in-flight updates | throughput only | [[olmo-3-rl-pipeline]] Table 23 |

These numbers come from math and code RL with rollouts of up to 32K tokens. None of them has Rocket's hour-long agentic tail. The direction is consistent across systems. The size of the gain depends on the length spread, and PipelineRL's authors state that gains "shrink when generation lengths are uniform" ([[pipelinerl]] Limitations).

**Correlation and causation.** The AReaL and verl comparisons are controlled: same hardware, same data, one design variable changed. Each is one run per setting. AReaL reports no seed count ([[areal-async-rl]] Verification).

---

## §7 Where the barriers remain

An asynchronous loop removes the global step barrier. Three smaller barriers remain, and each turns into a staleness term in ch-03:

| Barrier | Rocket | slime fully async | AReaL | verl V1 separate_async |
|---|---|---|---|---|
| Group completion | problem worker waits for all rollouts | group must complete; rejected groups replaced | group of `n_samples` | prompt group is the refill unit |
| Batch formation | controller packs eligible segments | rollout function collects `rollout_batch_size` groups | trainer waits for a batch of B prompts | one PPO mini-batch at a time (doc L39) |
| Weight transfer | controller triggers periodically; 15 s | pause admission → update → resume | interrupt, discard KV, recompute | `parameter_sync_step` (default 4) |

The group barrier cannot be removed while the algorithm is GRPO, because the advantage of each rollout depends on all of its siblings (llm-training ch-40). It can only be shortened. ch-04 covers how: drop failed rollouts, prioritize in-flight rollouts.

---

## Recipe

1. **Draw your Fig. 1.** For your stack, name the component that owns each of: problem selection, group assembly, trajectory state, generation, tool execution, per-rollout grading, group grading, advantage computation, batch packing, policy update, weight-transfer trigger.
2. **Keep the group owner single.** One component must hold all rewards of a group before advantages are computed. Put pass-rate filters and group graders there.
3. **Put isolated graders next to the trajectory** (rollout worker) and group graders at the group owner (problem worker).
4. **Record sampler outputs exactly.** Store sampler token IDs, per-token log-probs, and the weight version used for each turn. Never re-tokenize text for training.
5. **Separate control and data paths.** Pass references (Ray object refs, store keys) through the controller. Move token tensors from the store directly to the learner.
6. **Choose placement by length spread.** If rollout lengths are similar, colocated synchronous (verl `HYBRID`, slime `--colocate`) is simpler and strictly on-policy. If they differ widely, use separate pools and a persistent in-flight pool (slime fully async, verl `separate_async`, AReaL).
7. **Bound asynchrony explicitly** (open-instruct `async_steps`, AReaL η, verl `max_off_policy_threshold`) and log the bound's effect from the first run (ch-03).
8. **Checkpoint controller state together with learner state**, so a restored learner reconnects to a controller that knows which work is pending (Rocket p. 4; ch-20).

---

## Common mistakes and how to detect them

| Mistake | Observable symptom | Check |
|---|---|---|
| Re-applying the chat template to final messages to build training sequences | Ratios far from 1 at θ = θ_old; training does not converge | Compare stored sampler token IDs with re-tokenized IDs; verl reports non-convergence even single-turn ([[verl-rollout]]) |
| Running a pairwise or group grader inside the rollout worker | Grader blocks waiting for siblings, or grades against a partial group | Place graders by their data dependency (Sec. 3.6.3 rule) |
| Routing token tensors through a CPU controller | Controller memory grows with rollout length; controller becomes the throughput limit | Measure controller RSS and message size per batch; switch to references |
| Assuming async removes all waiting | Fast rollouts still wait for slow siblings; staleness high | Log per-rollout time from completion to group completion (Fig. 4's 7.6 h) |
| Losing per-turn version labels | Staleness cannot be computed per token; mixed-version trajectories cannot be corrected | Check that every stored turn carries the sampler weight version (Fig. 1 "versions"; slime `weight_versions`) |
| Using colocated mode for agentic tails | Step time equals the slowest trajectory; GPUs idle during tool waits | Compare `timing_s/gen` with training time per step (verl V1 metrics, doc L243) |

---

## Check your understanding

1. Rocket's learner batches carry references and advantages, not tokens. The post does not give a reason. Construct the argument from the components' hardware placement and from the turn counts in Fig. 2, and state one cost of the design (a component that must exist and stay available).
2. Why must a pairwise LLM grader run on the problem worker rather than the rollout worker? What would happen to staleness if a group grader were slow?
3. In MAI-Thinking-1 the problem worker sends 16 early requests and then 128 more. Explain why the pass-rate decision has to be made by the component that owns the group, and what would go wrong if rollout workers made it independently.
4. open-instruct's data-preparation loop logs "Consider increasing training compute" when generation is ahead. Explain which component is the bottleneck in that case and what happens to staleness if the bound `async_steps` is raised instead of adding learner GPUs.
5. slime retries segmented agent failures from their original inputs but continues flat samples from their partial response. Explain the difference in terms of what state can be restored from tokens alone.
6. The group barrier remains in every design in §7. Explain why no placement choice can remove it under GRPO, and name an algorithmic change that would remove it, with its cost.
7. AReaL's and verl's speedups were measured on math and code with rollouts up to 32K tokens. Give one reason the gain could be larger and one reason it could be smaller for Rocket's agentic mixture.

---

## Connections

- **Previous: ch-01** — why the recipe regime requires this loop.
- **Next: ch-03** — staleness as a per-token quantity, measured on this loop.
- **ch-04** — priority, dropping failures and inference-bound sizing act on the barriers of §7.
- **ch-05** — the weight transfer the controller triggers.
- **ch-08** — the agent loop, tool dispatcher and graders inside the rollout worker.
- **ch-11** — grader design; this chapter fixes only where graders run.
- **ch-20** — learner recovery reconnecting to the live controller.
- **llm-training ch-40** — GRPO group advantages, the reason the group barrier exists.
- **llm-training ch-45b** — response masks and observation masking for multi-turn segments.
- **llm-training ch-54** — earlier survey of async rollout systems; **ch-55** — verl internals.
- **llm-training ch-16** — prompt distribution, which the controller's curriculum samples from.

---

## Sources

- [[mai-rl-infrastructure]] — Fig. 1 components and the numbered flow (p. 2–4); Fig. 2 durations.
- [[mai-thinking-1]] — Rocket v1 controller, problem worker (16 + 128 requests, retries), rollout worker grading placement, router and inference details (Sec. 3.6.1–3.6.4).
- [[slime]] — training loop, colocate defaults, async cost model; code from slime@79aa41a.
- [[slime-agentic-and-mismatch-examples]] — fully async rollout, agent adapter recording trajectories.
- [[verl-rollout]] — rollout modes, token-in/token-out server, agent-loop masks; V1 async benchmark from verl@8357f1e docs.
- [[areal-async-rl]] — admission-bounded asynchrony and the 41.0 h vs 14.8 h comparison.
- [[areal-agent-pipelines]] — concat exporter and prefix-preservation rule.
- [[open-instruct-rlvr-pipeline]] — `DataPreparationActor` and the `async_steps` bound; code from open-instruct@f020e9a.
- [[olmo-3-rl-pipeline]] — async settings per run (Table 49) and infra throughput (Table 23).
- [[pipelinerl]] — in-flight weight updates; gains shrink for uniform lengths.
- [[async-rollout]] — OpenRLHF's queue-bounded async trainer, for comparison.
