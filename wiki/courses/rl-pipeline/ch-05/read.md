<!-- chapter: ch-05
     track: rl-pipeline
     kind: content
     title: Weight Transfer: Keeping the Sampler Current
     deps: [ch-04]
     sources: [[mai-rl-infrastructure]], [[mai-thinking-1]], [[slime]], [[verl-rollout]], [[deepseek-open-infra]], [[open-instruct-rlvr-pipeline]], [[pipelinerl]], [[slime-agentic-and-mismatch-examples]], [[areal-async-rl]], [[olmo-3-rl-pipeline]]
     figures: none
-->

# Chapter 5 — Weight Transfer: Keeping the Sampler Current

> **Core insight.** Sampler lag, the first term of ch-03's staleness decomposition, is set by how often the sampler receives new weights and how long each transfer takes. With a refresh every k learner steps, lag ranges over 0 … k − 1 steps; Rocket's Fig. 4 run refreshed every 4 steps, so "sampler lag is 0–3 steps" ([[mai-rl-infrastructure]] p. 6). Frequent refresh needs a transfer whose cost does not grow with the inference fleet. Rocket's reason for its design is stated directly: "sending weights separately to each one would make learner-side communication cost grow with the inference fleet size". It therefore transfers in two stages: "selected sampler engines first pull packed parameter chunks from the learner, and then distribute them to corresponding ranks across sampler replicas", which takes 15 s "in our current configuration and training run" (p. 6–7). Rocket v1 solved the other half of the problem, resharding between different learner and inference layouts, with a transfer plan computed once at job start and expanded at runtime to all live replicas ([[mai-thinking-1]] Sec. 3.6.5). Open frameworks make the same choices at smaller scale: a pause → push → resume protocol around the transfer (slime, verl, open-instruct), a single-root NCCL broadcast (slime, open-instruct) or relay-based fan-out (verl's multi-sender and kimi checkpoint engines), and CUDA IPC when learner and sampler share GPUs.
>
> **Guideline.** When the inference fleet has many replicas, do not send a full copy of the weights from the learner to each replica; send one copy into the fleet and fan out inside it, because Rocket identifies learner-side cost growth with fleet size as the problem (p. 6) and verl's NCCL engine adds same-node relays for the same reason, so that NCCL "can push on over each peer's own NIC" (verl@8357f1e `verl/checkpoint_engine/nccl_checkpoint_engine.py` L215–222). When you choose the sync cadence, set it so that k − 1 steps of lag is small compared with the generation-to-training term of your longest rollouts, because Fig. 4 shows 0–3 steps of lag against about 20 steps of waiting for one group (p. 6; ch-03 §3). When generation continues across a weight update, decide explicitly whether the KV cache is kept, recomputed, or flushed, because PipelineRL measured that keeping the stale cache raised KL only slightly over recomputing it ([[pipelinerl]] Fig. 7), while verl refuses to update weights unless in-flight requests were aborted and caches cleared (verl@8357f1e `verl/workers/rollout/vllm_rollout/vllm_async_server.py` L1249–1253). When learner and inference shard parameters differently, compute the resharding plan once and reuse it, because Rocket v1 does this and keeps the plan valid "as replicas join, leave, or are replaced" ([[mai-thinking-1]] Sec. 3.6.5).

---

## Why this chapter matters

Weight transfer is the only lever on sampler lag, and it is the one synchronization point that remains in every asynchronous design in ch-02. Its cost grows with model size and fleet size. MAI Base 2 has three trillion parameters and an inference fleet in the thousands of GPUs. A transfer design that is fine for a 7B model on 16 GPUs can take minutes at that scale. Each minute of transfer is either idle inference or a longer lag.

The open frameworks expose the decisions in this chapter as flags: transport backend, bucket size, cache flushing interval, partial rollout, in-flight updates. Reading their code shows what each flag does to the rollouts that are running when the weights change.

---

## §1 How weight transfer sets sampler lag

### 1.1 Lag as a function of cadence

Let the learner be at step s when a token is generated, and let the sampler serve the weights of step v. If the sampler is refreshed every k learner steps and the transfer completes instantly, v is the most recent multiple of k and s − v ∈ {0, …, k − 1}. In Fig. 4 the refreshes are at steps 104, 108, …, 124 and k = 4, so the lag is 0–3 (p. 6).

A transfer of duration τ adds lag in one of two ways. If generation continues on old weights during the transfer, tokens generated in that window carry about τ / Δt extra steps of lag. If generation pauses, no stale tokens are produced, but inference is idle for τ. MAI-Thinking-1 states both costs: transfer time "creates inference idleness and additional staleness" (Sec. 3.6.5). The Base 2 post does not say which of the two Rocket does during its 15 s transfer.

### 1.2 How large the lag term is

With τ = 15 s, the transfer adds about 15 s / Δt steps if generation continues during it. The post does not state Δt; ch-03 §3.2 derives only an upper bound of about 37 min for Fig. 4. The transfer term is below 0.1 step for any Δt above 2.5 min, so unless steps are very short, the cadence dominates: 0–3 steps. The waiting term for the same group is about 20 steps or more (ch-03 §3). So in Rocket's agentic run, faster weight transfer can remove at most a few steps of a 23-step staleness. It cannot fix the long tail.

The cadence matters more when rollouts are short and steps are fast. With k = 4 and a step of a few minutes, 0–3 steps of lag is a large fraction of the total staleness of a math rollout that finishes within one or two steps. Fast transfer is what allows k to be small.

### 1.3 Cadences in practice

| System | Cadence | Transfer behaviour | Source |
|---|---|---|---|
| Rocket (Base 2) | every 4 learner steps (Fig. 4 run) | two-stage; 15 s | p. 6–7 |
| Rocket v1 | "5 gradient steps between inference model updates" | planned resharding over DP-group subsets | [[mai-thinking-1]] Sec. 3.1.5, 3.6.5 |
| verl V1 `separate_async` | `parameter_sync_step`, default 4; must satisfy `train_batch_size = parameter_sync_step × ppo_mini_batch_size` | checkpoint engine (non-`naive` backend required) | verl@8357f1e `docs/advance/v1_async_trainer.md` L41–49, L230; `verl/trainer/config/ppo_trainer.yaml` L261–262 |
| open-instruct | every step, sync thread notified by the main loop | NCCL broadcast; `inflight_updates` default True | [[open-instruct-rlvr-pipeline]] Pipeline step 7 |
| PipelineRL | after each optimizer step | in-flight, stale KV kept | [[pipelinerl]] Summary |
| slime coding agent | after every optimizer update; cache flushed every third update | `FLUSH_CACHE_INTERVAL=3` | slime@79aa41a `examples/coding_agent_rl/README.md` L222–224 |

---

## §2 Why per-replica sends do not scale

### 2.1 The fan-out cost

Let S be the bytes of one model copy as the sampler stores it, and R the number of sampler replicas. If the learner sends a full copy to each replica, it sends R · S bytes. If those bytes leave through a fixed set of learner NICs with total bandwidth b, the transfer takes at least R · S / b. Doubling the fleet doubles the transfer time, and the learner is the bottleneck. This is the situation Rocket describes: "sending weights separately to each one would make learner-side communication cost grow with the inference fleet size" (p. 6).

If the learner sends one copy into the fleet and the replicas forward it among themselves, the learner sends about S bytes, independent of R. The forwarding work is spread over the inference nodes' own NICs. With a pipelined chain or tree, each node forwards a bounded number of copies, and total time grows slowly with R.

### 2.2 Why NIC bandwidth is the constraint

Within a node, GPUs are connected by NVLink. Between nodes, traffic goes through the NICs. DeepSeek's DeepEP measurements show the size of the gap on one platform: on H800 with about 160 GB/s NVLink and one 400 Gb/s ConnectX-7 InfiniBand NIC per GPU (about 50 GB/s), intranode dispatch reached 153 GB/s and internode dispatch 43–46 GB/s ([[deepseek-open-infra]] DeepEP table, README @84d3d6f). These are MoE all-to-all measurements, not weight-transfer measurements. They show the ratio that a weight-transfer design must respect: moving bytes across nodes costs about 3–4× more time than moving them within a node on that hardware. A design that crosses the inter-node fabric once per model copy, and fans out inside nodes over NVLink, uses the fast links for the repeated work.

### 2.3 Rocket's two stages

Rocket's description (p. 6–7), as steps:

1. **Pull.** "Selected sampler engines first pull packed parameter chunks from the learner." The learner does not push to every engine. A subset of engines fetches the parameters, packed into chunks.
2. **Distribute.** The selected engines "then distribute them to corresponding ranks across sampler replicas". Each replica is sharded the same way, so rank j of the selected engine forwards its chunk to rank j of every other replica.
3. **Result.** Learner-side cost does not grow with the number of replicas. Transfer takes 15 s in the stated configuration.

**Not stated:** how many engines are "selected", chunk size, the transport (NCCL, RDMA, other), whether the learner quantizes to MXFP8 before sending or the sampler quantizes on receipt (ch-06), what happens to in-flight generation during the 15 s, and the model bytes moved. The 15 s figure cannot be turned into a bandwidth without the bytes.

---

## §3 Rocket v1: one transfer plan, expanded at runtime

MAI-Thinking-1 describes the earlier Rocket weight transfer in more detail ([[mai-thinking-1]] Sec. 3.6.5, p. 52–53). The base problem it addresses is resharding.

1. **The problem.** "Learner and inference allocations shard the same tensors differently: Each side may use different combinations or degrees of FSDP, pipeline parallelism, data-parallel attention, and tensor layouts, while parameters may also differ in precision, quantization state, or matrix layout. Every transfer must therefore both reshard bytes and perform per-tensor transformations."
2. **The plan.** "We compile resharding, data movement, and per-tensor transformation into a single transfer plan computed once at job start and reused thereafter." For each parameter, the planner "intersects the learner's and inference server's sharding layouts and emits one entry per non-empty sub-shard overlap, recording source rank, destination rank, byte extent, and required operations such as dtype casts or layout permutations".
3. **No full-tensor materialization.** "Each learner rank sends only the slices that its paired inference rank needs, eliminating separate materialization of full tensors."
4. **Placement of transforms and pipelining.** "Transforms run on whichever side minimizes their cost, and the runtime pipelines packing, transfer, and unpacking so successive sub-shards overlap in time."
5. **Fleet independence.** "The plan targets an idealized topology of one learner and one inference server; at runtime each source–destination pair expands into a transfer group covering all live replicas of the destination rank, so the plan remains valid as replicas join, leave, or are replaced."
6. **DP subsets.** Because data parallelism replicates parameters, "one DP group already holds every model parameter", so the plan uses "a subset of DP groups, each paired with a disjoint slice of inference replicas and running independently. A 36-server fleet split four ways performs four 9-server transfers in parallel, improving throughput while containing the failure blast radius."

Step 5 is the precursor of the Base 2 two-stage design. The plan is written for one destination replica, and fan-out to the other replicas is a runtime expansion. Step 6 adds a second form of parallelism: several independent source groups, each feeding a slice of the fleet. A failure in one 9-server transfer does not stall the other 27 servers.

Compare the open frameworks. slime gathers TP shards to the PP source rank and converts each parameter to Hugging Face layout before sending (§5.1). open-instruct gathers the whole model with a ZeRO-3 or FSDP collective and has rank 0 send (§5.3). Both materialize full parameters on the sending side. Rocket v1's sub-shard intersection avoids that.

---

## §4 What happens to rollouts in flight

A weight update arrives while many requests are mid-generation. A design has to choose what happens to them, and the choice sets both staleness and KV-cache consistency. The four open choices:

### 4.1 Abort, clear caches, park new requests (verl)

verl closes a submission gate, lets admissions already past it land, then aborts everything in flight and clears caches:

```python
# verl@8357f1e verl/workers/rollout/vllm_rollout/vllm_async_server.py L1196–1204, L1234–1253
try:
    # Close the gate first, then let admissions already past it land, so the pause
    # below actually covers them. Closing the gate and declaring how late arrivals
    # are handled happen together, so no request can observe one without the other.
    self._submission_paused = True
    self._rejecting = reject_request
    self._resume_event.clear()
    deadline = time.monotonic() + _GATE_BARRIER_TIMEOUT_S
    while self._admitting > 0:
        ...
    reset_connector = self.config.external_kv_cache_invalidation == "reset"
    clear_cache = reset_prefix_cache and reset_connector
    if _VLLM_VERSION >= version.parse("0.32.0"):
        await self.engine.pause_generation(mode="abort", clear_cache=clear_cache)
    else:
        await self.engine.pause_generation(
            wait_for_inflight_requests=False,
            clear_cache=clear_cache,
        )
    if reset_prefix_cache and not reset_connector:
        await self.clear_kv_cache()

    logger.info(f"Aborted {len(request_ids)} requests: {request_ids}")
    return {"aborted_count": len(request_ids), "request_ids": request_ids}

except Exception:
    # Weight updates must not proceed unless every in-flight request was
    # actually aborted and the old-weight caches were cleared.
    logger.exception("Error aborting requests")
    raise
```

The aborted requests are not lost in the async trainers. verl V1's partial-rollout client keeps "tokens and log probabilities produced before the abort", retries "the unfinished trajectory through the load balancer", the resumed request runs "with a newer model version", and "the KV cache is reconstructed for the retained prefix before decoding continues" (verl@8357f1e `docs/advance/v1_async_trainer.md` L53–60). The cost is "extra prefill work and within-trajectory policy-version changes" (L60). The KV for the old prefix is recomputed with new weights, so the cache is consistent with the new policy, while the stored log-probs of the prefix remain those of the old policy.

### 4.2 Pause in place, flush periodically (slime)

slime's distributed updater bumps the version, pauses engines, optionally flushes caches, sends the weights, and continues:

```python
# slime@79aa41a slime/backends/megatron_utils/update_weight/update_weight_from_distributed.py L103–122, L132–133, L143
@torch.no_grad()
def update_weights(self) -> None:
    """
    Pause → optionally flush → _send_weights → continue. Progress on PP source.
    """
    self.weight_version += 1
    flush_cache = should_flush_cache(
        self.args.flush_cache_interval, self.weight_version, getattr(self.args, "update_weight_start_version", 0)
    )

    if dist.get_rank() == 0:
        ray.get(
            [
                engine.pause_generation.remote(mode="abort" if flush_cache else "in_place")
                for engine in self.rollout_engines
            ]
        )
        if flush_cache:
            ray.get([engine.flush_cache.remote() for engine in self.rollout_engines])
    ...
    pbar = tqdm(desc=f"[{self._group_name}] Update weights", total=0) if self._is_pp_src_rank else None
    self._send_weights(pbar)
    ...
        ray.get([engine.continue_generation.remote() for engine in self.rollout_engines])
```

The helper's docstring states the policy: "Refresh at startup and every N subsequent syncs; nonpositive preserves KV … A full refresh aborts requests; other updates pause them in place" (slime@79aa41a `slime/utils/weight_sync.py` L4–10). Between flushes, requests resume on the new weights with a KV cache computed under old weights.

### 4.3 Update in flight without waiting (open-instruct, PipelineRL)

open-instruct stops its actors from taking new work, broadcasts, and with `inflight_updates` does not wait for the inner engine-update calls:

```python
# open-instruct@f020e9a open_instruct/grpo_utils.py L796–810
start = time.perf_counter()
ray.get(actor_manager.set_should_stop.remote(True))
try:
    results, actor_sync_times = utils.ray_get_with_progress(
        broadcast_refs, desc="Broadcasting weights to vLLM engines", enable=progress
    )
    if not inflight_updates:
        utils.ray_get_with_progress(
            itertools.chain.from_iterable(results), desc="Waiting for vLLM engine update RPCs", enable=progress
        )
    utils.ray_get_with_progress(
        [e.wake_up.remote() for e in vllm_engines], desc="Waking up vLLM engines", enable=progress
    )
finally:
    ray.get(actor_manager.set_should_stop.remote(False))
```

`inflight_updates` defaults to True (`data_loader.py` L443). Olmo 3 cites PipelineRL for this mode and measured its throughput effect: on a fixed 2-node benchmark, adding in-flight updates raised throughput from 1,358 to 2,949 tokens/s, after continuous batching and better threading had taken it from 881 to 1,358 ([[olmo-3-rl-pipeline]] Table 23). PipelineRL keeps the stale KV cache by default and measured that sequences continued on stale KV had "only slightly higher KL divergence than recomputed ones" ([[pipelinerl]] §5.1, Fig. 7).

### 4.4 Interrupt, discard KV, recompute (AReaL)

AReaL's paper: an `update_weights` request interrupts ongoing generations, the workers discard KV caches computed under the old weights, recompute them under the new weights, and continue decoding ([[areal-async-rl]] §4.1). Its ablation measured that interruptible generation raised generation throughput by 12% (1.5B) and 17% (7B) on 4 nodes, 231k vs 207k and 130k vs 111k tokens/s ([[areal-async-rl]] Fig. 6b). In the current code, the SGLang pause is sent in two steps: a default-mode pause that "aborts in-flight requests and returns their partial output so the client resumes them by extending the prompt", then an in-place pause that raises the scheduler flag (AReaL@bcef492 `areal/engine/sglang_remote.py` L495–515).

### 4.5 Comparison

| Policy | KV after update | Tokens of one turn from | Extra compute | Who |
|---|---|---|---|---|
| Abort + retry with prefix | recomputed under new weights | old prefix, new continuation | prefill of the prefix | verl V1, AReaL |
| Pause in place | stale (old-weight KV), flushed every N syncs | old prefix, new continuation | none until flush | slime |
| In-flight, keep KV | stale | old prefix, new continuation | none | PipelineRL, open-instruct |
| Rocket (Base 2) | not stated | per-turn versions recorded (Fig. 1 "versions") | not stated | — |

In every row the stored log-probs of already-generated tokens belong to the old policy. That is why the per-token version labels of ch-02 and ch-03 are required whichever policy is used.

---

## §5 Transport and topology in open frameworks

### 5.1 slime: one broadcast group per pipeline stage

For disaggregated engines, slime creates a process group containing one training rank (the PP source, DP = TP = 0) and every GPU of every engine:

```python
# slime@79aa41a slime/backends/megatron_utils/update_weight/update_weight_from_distributed.py L291–298, L317–325
if engine_gpu_counts is None:
    engine_gpu_counts = [args.rollout_num_gpus_per_engine] * len(rollout_engines)

master_address = ray._private.services.get_node_ip_address()
with socket.socket() as sock:
    sock.bind(("", 0))
    master_port = sock.getsockname()[1]
world_size = sum(engine_gpu_counts) + 1  # +1 for training rank 0
...
model_update_groups = init_process_group(
    backend=backend,
    init_method=f"tcp://{_wrap_ipv6(master_address)}:{master_port}",
    world_size=world_size,
    rank=0,
    group_name=group_name,
)
ray.get(refs)
return model_update_groups
```

The training rank then issues `dist.broadcast` of a flattened bucket to the whole group (L369–376). The learner issues one collective per bucket, not one send per replica, so NCCL's broadcast algorithm handles the fan-out. The group size is the whole fleet plus one, so group setup and the broadcast's span grow with fleet size. Before sending, TP shards are all-gathered and converted to Hugging Face layout (`UpdateWeightFromTensor` docstring, `update_weight_from_tensor.py` L53–58).

### 5.2 verl: checkpoint engines with relays

verl puts weight transfer behind a "Checkpoint Engine" abstraction with `send_weights` / `receive_weights` / `get_weights` APIs (verl@8357f1e `verl/checkpoint_engine/README.md` L4–9). Its backend table (L15–22):

| Backend | Topology | Elastic | Use case |
|---|---|---|---|
| naive | all_gather | NA | on-policy, colocated |
| nccl | all_gather + broadcast | low: rebuild NCCL group | disaggregated, fixed clusters |
| nixl | all_gather + ring p2p | high: dynamic ring | disaggregated, elastic rollout, rollout fault tolerance |
| kimi_ckpt_engine | p2p + broadcast | low | disaggregated, checkpoint each time |
| mooncake | all_gather + ring p2p | high | disaggregated, fixed clusters |

Two of them address the fan-out problem directly.

- **kimi_ckpt_engine** is a two-stage design: "the actor first offloads the weights to the CPU, and the rollout creates a sub communication group that includes all the cards for the rollout. Then, using Mooncake transfer engine, these weights are transmitted via P2P to a specific worker in the rollout, followed by a broadcast to all other rollout workers" (README L26). This is the open design closest to Rocket's "pull, then distribute".
- **nccl with `multi_sender`** widens the root of the broadcast:

```python
# verl@8357f1e verl/checkpoint_engine/nccl_checkpoint_engine.py L214–233
@staticmethod
def _multi_sender_ranks(actor_wg_world_size: int, metadata: list[WorkerMetadata]) -> list[int]:
    """Actor rank 0 joins, plus every actor worker on its node, which are its NVLink peers.

    Those peers carry no data the rollout needs. They join so NCCL has somewhere to fan a
    bucket out to over NVLink, from which it can push on over each peer's own NIC. A peer on
    any other node would have to pull a full copy over the fabric to contribute nothing, so it
    gets rank -1 instead.
    """
    source_node = metadata[0].node_id

    # Ranks must stay contiguous, so number the survivors as we walk the actor workers.
    ranks, next_rank = [], 0
    for i in range(actor_wg_world_size):
        if i == 0 or metadata[i].node_id == source_node:
            ranks.append(next_rank)
            next_rank += 1
        else:
            ranks.append(-1)
    return ranks
```

The docstring gives the NIC argument of §2.2 in code form. Copies are made over NVLink inside the source node, and each copy then leaves through a different NIC.

verl's measured transfer times (README L55–74), Qwen3-30B-A3B-Base, 30 rollout receivers, weights received without CUDA IPC into the engine:

| Hardware | Backend | Time (s) | Bandwidth (GB/s) |
|---|---|---|---|
| 4 × 8 H100, ConnectX-7 400 Gbps IB | NCCL | ~7 | 8.25 |
| 4 × 8 H100, ConnectX-7 400 Gbps IB | NIXL | ~7 | 8.25 |
| 2 × 16 Ascend 910C | HCCL | ~11 | 5.3 |
| 2 × 16 Ascend 910C | kimi_ckpt_engine | offload 7, update 3.5 | 16.5 |
| 2 × 8 H100, ConnectX-7 400 Gbps IB | mooncake | 5.93 | 9.44 |

The model is about 1% of MAI Base 2's parameter count, so these times do not predict Rocket's. They show that a 30B MoE moves in single-digit seconds on one rack-scale setup, so the transfer is not the dominant cost at that size.

### 5.3 open-instruct and AReaL

open-instruct's `broadcast_weights_to_vllm` "must be called on ALL ranks when using DeepSpeed stage 3 or FSDP, since gathering is a collective operation. Only rank 0 sends weights", with NCCL when a model-update group exists and CUDA IPC otherwise (open-instruct@f020e9a `open_instruct/vllm_utils.py` L1391–1410). Engines are put to sleep before the broadcast (L1406–1407).

AReaL's `weight_update_mode` takes `disk`, `xccl` (NCCL broadcast) or `awex`, default `xccl` (AReaL@bcef492 `areal/api/cli_args.py` L1588–1595), with an option for sparse delta updates (L1596–1598).

### 5.4 Colocated transfer and delta transfer

When learner and sampler share GPUs, the bytes do not cross a network. slime's docs state "slime uses CUDA IPC for colocation" (`docs/en/advanced/rl-systems.md` L15), while the `UpdateWeightFromTensor` docstring describes the colocated path as "GPU→CPU serialize → gather_object(Gloo CPU …) → Ray IPC to engine" (`update_weight_from_tensor.py` L56). The two descriptions conflict; the [[slime]] card records the same inconsistency and does not resolve it.

Delta transfer sends only changed bytes. slime's delta mode diffs gathered tensors against a CPU snapshot, compresses with zstd, writes a versioned directory on a shared filesystem, and has every engine pull and apply it with per-tensor checksums ([[slime-agentic-and-mismatch-examples]] Delta weight sync). Its docs give the condition under which this helps: delta "wins only when the bytes saved outweigh" the snapshot, diff, encoding and reconstruction time (`rl-systems.md` L15). No byte-reduction measurement is reported in-tree. The rough cost model in the same doc is t_W ≈ S_W / B_W + t_0 (L11–13).

---

## Recipe

1. **Measure S and b.** Record the bytes of one sampler-format model copy and the per-NIC bandwidth from learner nodes to inference nodes. Compute R · S / b for your fleet size R. If it exceeds your target lag budget, a per-replica design will not work.
2. **Send one copy into the fleet.** Use a relay or two-stage transport (verl `kimi_ckpt_engine`, NCCL `multi_sender`, or a pull-then-fan-out design like Rocket's). Fan out inside nodes over NVLink where possible.
3. **Precompute the resharding plan** at job start when learner and sampler layouts differ (Rocket v1). Send sub-shards, not gathered full tensors.
4. **Split the fleet into independent transfer groups** (Rocket v1's four 9-server transfers) so a single failure does not stall all replicas.
5. **Choose the cadence k** so that the transfer time τ is a small fraction of k · Δt, and k − 1 steps of lag is small compared with your typical generation-to-training wait (ch-03).
6. **Choose the in-flight policy** (§4.5) and record per-token versions in every case. If you keep stale KV, flush it on a fixed interval and measure the KL between continued and recomputed sequences on a sample, as PipelineRL did.
7. **Quantize on the side that matches the sampler's expected format** (ch-06), and verify bit-equality of the received tensors against the learner's export on a test parameter.
8. **Log** transfer time, bytes, inference idle time during transfer, and the number of aborted or paused requests per sync.

---

## Common mistakes and how to detect them

| Mistake | Observable symptom | Check |
|---|---|---|
| Learner sends to every replica | Transfer time grows linearly as replicas are added | Plot transfer time vs replica count; compare with R · S / b |
| Single-root broadcast through one NIC | Root NIC saturated; other learner NICs idle | NIC counters per learner node during sync; try relays (verl `multi_sender`) |
| Updating weights while requests are half-admitted | Some requests run new weights with old caches or old salt | verl's gate barrier and exception path (L1196–1253) as a reference; count requests admitted during sync |
| Keeping stale KV without measurement | Hidden sampler–learner gap on continued sequences | Recompute KV for a sample of continued requests and compare log-probs (PipelineRL Fig. 7 method) |
| Losing per-token versions across an abort-and-retry | Prefix tokens attributed to the new version | Check that retried trajectories carry the original versions for the prefix |
| Large cadence to save transfer cost | Short rollouts see most of their staleness as sampler lag | Split staleness into lag and waiting terms per task type (ch-03 recipe) |
| Reading a transfer time without the bytes | Wrong bandwidth estimate; wrong scaling forecast | Always record bytes with time (verl's table does; Rocket's 15 s does not) |

---

## Check your understanding

1. With refresh every k steps and instantaneous transfer, sampler lag is 0 … k − 1. Derive this, and explain how a non-zero transfer time τ changes lag differently depending on whether generation pauses during the transfer.
2. In Fig. 4, sampler lag is at most 3 steps while the group's waiting term is about 20. Explain why Rocket still invests in fast transfer, using the case of short rollouts.
3. Explain why per-replica sends make learner-side cost grow with fleet size, and why a single NCCL broadcast from one learner rank is not by itself a full solution. What does verl's `multi_sender` change?
4. Rocket v1's plan "targets an idealized topology of one learner and one inference server". Explain what this buys when replicas fail or are added, and how it relates to the Base 2 two-stage design.
5. PipelineRL keeps stale KV after an in-flight update; verl aborts and recomputes the prefix KV. For each, state which tensors are computed by which policy version, and which of them the learner's importance ratio must account for.
6. Why does every in-flight policy in §4.5 still require per-token version labels?

---

## Connections

- **Previous: ch-04** — the levers on the waiting terms; this chapter covers the lag term.
- **Next: ch-06** — MXFP8 sampler storage, which determines the format the transfer must deliver.
- **ch-03** — the staleness decomposition and Fig. 4.
- **ch-07** — bitwise parity requires the sampler to receive exactly the learner's weights.
- **ch-20** — learner recovery; the transfer path must reconnect to a rebuilt learner.
- **ch-21** — elasticity; a fleet-independent plan is what lets inference resize immediately.
- **llm-training ch-54** — async rollout systems and in-flight updates in the earlier course.
- **llm-training ch-55** — verl internals, including the rollout server.

---

## Sources

- [[mai-rl-infrastructure]] — two-stage transfer, 15 s, 4-step cadence, lag 0–3 (p. 6–7).
- [[mai-thinking-1]] — Rocket v1 transfer plan, DP subsets, 36-server example, "inference idleness and additional staleness" (Sec. 3.6.5); 5 gradient steps per update (Sec. 3.1.5).
- [[slime]] — pause → update → resume loop; distributed process group; colocated path inconsistency; slime@79aa41a.
- [[slime-agentic-and-mismatch-examples]] — delta weight sync; flush interval in the coding-agent example.
- [[verl-rollout]] — abort/clear/park protocol; V1 partial rollout; checkpoint engines and benchmarks (verl@8357f1e).
- [[open-instruct-rlvr-pipeline]] — rank-0 NCCL broadcast and in-flight updates; open-instruct@f020e9a.
- [[olmo-3-rl-pipeline]] — throughput effect of in-flight updates (Table 23).
- [[pipelinerl]] — in-flight update with stale KV; KL measurement (Fig. 7).
- [[areal-async-rl]] — interruptible generation and its throughput ablation (Fig. 6b).
- [[deepseek-open-infra]] — intranode vs internode bandwidth measurements used in §2.2.
