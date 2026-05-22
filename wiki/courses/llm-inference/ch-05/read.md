<!-- chapter: ch-05
     track: scheduling
     title: Chunked Prefill + Mixed-Batch Scheduling (Sarathi-Serve)
     sources: [[sarathi-serve]], [[vllm-scheduler]], [[continuous-batching]]
     back: [[continuous-batching]] (ch-04)
     forward: [[pagedattention]] (ch-06), [[distserve]] (ch-09), [[vllm-scheduler]] (ch-16)
     figures: figures/chunked-prefill-timeline.html
-->

# Chapter 5 — Chunked Prefill + Mixed-Batch Scheduling (Sarathi-Serve)

> **Core insight.** Continuous batching alone cannot fix the prefill/decode interference problem: a single long prompt's prefill step takes 10–100× longer than a decode step, so the moment one arrives, every running decode stalls for it. Sarathi-Serve cuts the long prefill into bounded *chunks* and packs each chunk into a step alongside decode tokens, so decode traffic keeps flowing and the GPU stays compute-bound. The whole serving step becomes a single mixed-batch forward pass whose cost is a token-budget knob you control.
>
> **Guideline.** In vLLM V1 / Sarathi-Serve / SGLang, enable chunked prefill and set `max_num_batched_tokens` to the smallest value that still keeps your GPU compute-saturated (typically 2048–8192 on H100, 512–2048 on A100). Set `max_chunk_size` (or rely on the engine's auto-tune) so a single prefill chunk fits inside one step's token budget after subtracting the running decode count. TTFT rises slightly (∼10–30 %), TPOT drops dramatically (often 2–5×), and goodput@p99 wins.

---

## Why this chapter exists

[[continuous-batching]] (ch-04) gave us iteration-level scheduling: at every decode step the scheduler can admit new requests and retire finished ones, without forcing the static batch to drain. That alone is a 2–5× throughput win over static batching. But continuous batching answers only one question — *who runs together?* — and leaves a second question unsolved: *what work goes into one step?*

The unsolved problem is concrete. A single decode step on Llama-3-8B at batch 32 takes roughly 20 ms on an H100. A single prefill step on a 4096-token prompt for the same model takes roughly 300 ms. If you admit one such prompt while 32 decodes are in flight, the next 300 ms of all 32 streaming requests are silently lost — the user sees `TPOT` spike from 20 ms to 320 ms for one token, mid-stream. The naive iteration-level scheduler still treats "prefill the new request" and "decode one token for every running request" as separate, sequential steps, and the prefill blocks the decode.

[[sarathi-serve]] (Agrawal et al. 2024) is the paper that closes this gap. The trick is mechanically simple — split the long prefill into chunks of, say, 512 tokens each, and pack each chunk into a forward pass that *also* contains all the running decode tokens — but the consequences for serving design are large. After Sarathi-Serve, the scheduling unit is no longer "a request" or even "an iteration"; it's "a token budget," and prefill and decode are two different ways to spend it.

This chapter follows the paper and the production realization in [[vllm-scheduler]] (the vLLM V1 scheduler implements chunked prefill as a first-class scheduling primitive).

---

## 1. The prefill/decode asymmetry that breaks pure continuous batching

Recall from ch-03 the two phases of autoregressive generation:

- **Prefill**: one forward pass over the entire prompt of length `L_p`. Attention cost is `O(L_p²)`. FFN cost is `O(L_p)`. For a typical prompt (1k–8k tokens) prefill is **compute-bound**: the GPU's tensor cores can be fully utilized by a single request.
- **Decode**: one forward pass over **one** new token, with attention against the cached `L_p + t` past tokens. Compute per step is `O(L_p + t)`, but the dominant cost is **memory bandwidth** — every layer must stream all of its KV cache and weights from HBM for each decode step. A single request leaves >90 % of the H100's tensor cores idle.

The asymmetry has three operational consequences a serving system has to live with:

1. **Prefill is hostile to latency.** One forward pass takes 100s of ms for any prompt over a few thousand tokens. Anyone whose decode happens to share that step waits the full prefill latency.
2. **Decode is hostile to throughput.** A single decode wastes the GPU. You need batch sizes of 32–256 concurrent decodes to push the tensor cores into the compute-bound regime — but each running request adds KV cache memory pressure.
3. **The two compete for the same SM.** A simple round-robin "decode for a while, then prefill" alternation is what pure continuous batching does; this is exactly what causes the TPOT spikes above.

Worse, the spikes are not occasional. In a typical chat workload, prefill happens on every new turn — every time a user hits send. So even at modest load (a few requests per second per GPU) the spike happens dozens of times per minute.

The Sarathi-Serve paper formalizes the failure mode as **"generation stalls"** — moments where running decodes have to wait for an unrelated prefill — and shows they dominate p99 TPOT under conventional continuous batching.

---

## 2. The chunked-prefill mechanism

The core idea ([[sarathi-serve]] §3) is to make prefill latency a knob, not a forced cost.

Replace the operation "prefill the entire prompt of `L_p` tokens in one step" with a loop:

```
for chunk in split(prompt, chunk_size):
    schedule one model step that does:
        - prefill `len(chunk)` tokens for this request
        - decode one token for every running request that has capacity
```

Each chunk advances the new request's KV cache by `chunk_size` positions. After `ceil(L_p / chunk_size)` chunks the request has finished prefill and transitions into decode. Every chunk-bearing step also runs decode for everyone else, so the *visible* TPOT for running requests is bounded by one chunk's worth of compute plus the decode batch — not by the whole 4096-token prefill.

The mechanism rests on two architectural facts about transformer attention:

- **Causal mask + KV reuse.** Tokens within a prefill chunk attend to all earlier tokens *including those processed by previous chunks of the same request*. Those earlier K/V tensors are already in the KV cache. So a chunk of size 512 doing prefill at position 1536–2047 is computationally identical to "a request whose existing context is 1536 tokens, now extending by 512 new tokens" — exactly the same kernel shape the decode-with-extend path already supports.
- **Variable-length attention kernels.** Production attention kernels (FlashAttention, FlashInfer, vLLM's `triton_attention`) already accept a `cu_seqlens` tensor describing the per-sequence start/end positions inside a packed batch. So one prefill chunk + N decode tokens *is* one packed batch — no separate kernel path.

The price you pay is one extra forward pass per `chunk_size` tokens of prompt. With `chunk_size = 512` and a 4096-token prompt you launch 8 forward passes instead of 1. Per-token prefill compute is unchanged; only the per-launch overhead increases. On modern GPUs that overhead is small (∼1–2 %) compared to the latency-stability win.

---

## 3. Mixed-batch scheduling — what fits in one forward pass

The Sarathi-Serve scheduler treats each step as a **mixed batch** described by one number: the total token budget `T`. For a step:

```
T = num_prefill_tokens_this_step + num_decode_tokens_this_step
  ≤ max_num_batched_tokens
```

`max_num_batched_tokens` is the user-set ceiling — the largest number of tokens any single forward pass on this GPU will process. The scheduler picks the split each step subject to this single constraint and to KV-cache memory availability:

```python
# Sketch of the per-step scheduler decision (vLLM V1 / Sarathi-Serve style)

def build_step(running, waiting, T_max):
    decode_tokens = len(running)           # 1 per running request
    chunk_budget  = T_max - decode_tokens  # what's left for prefill

    if chunk_budget <= 0:
        # Decode is already saturating compute; defer all new prefills
        return Step(prefill=[], decode=running)

    prefill_chunks = []
    while waiting and chunk_budget > 0:
        req = waiting.peek()
        tokens_left_in_prefill = req.prompt_len - req.prefill_pos
        take = min(tokens_left_in_prefill, chunk_budget, max_chunk_size)
        if kv_cache_has_room_for(req, take):
            prefill_chunks.append((req, take))
            chunk_budget -= take
            req.prefill_pos += take
        else:
            break  # KV pressure → don't admit; possibly preempt
    return Step(prefill=prefill_chunks, decode=running)
```

Two design choices in that sketch are worth pausing on:

- **Decodes go first.** Running requests are guaranteed their one decode token per step. The remaining token budget is what prefill chunks compete for. This is what makes Sarathi-Serve a *stall-free* scheduler: TPOT for the active fleet is bounded above by one mixed step's latency.
- **Token budget is a hardware property.** `max_num_batched_tokens` is empirically tuned to keep the GPU compute-bound. Below that level, the kernel runs in the bandwidth-bound regime and adding more tokens is free. Above it, the kernel runs in the compute-bound regime and each extra token adds latency. The sweet spot is the inflection point.

The vLLM V1 scheduler (`vllm/v1/core/sched/scheduler.py`) is the canonical production implementation. Its `Scheduler.schedule()` method walks running and waiting queues exactly this way, asks `kv_cache_manager.allocate_slots(...)` whether memory is available, and emits a `SchedulerOutput` describing the mixed batch the worker should execute. See [[vllm-scheduler]] for the code-level walkthrough that ch-16 expands on.

---

## 4. Tuning `max_num_batched_tokens` and `max_chunk_size`

These are the two production knobs you actually turn.

**`max_num_batched_tokens`** — the per-step total token budget. The right value is workload- and hardware-dependent but follows a predictable rule:

| GPU | Llama-3-8B (bf16, GQA-8) | Llama-3-70B (TP=4) |
|---|---|---|
| A100 80GB | 2048–4096 | 1024–2048 |
| H100 80GB | 4096–8192 | 2048–4096 |
| H200 / B100 | 8192–16384 | 4096–8192 |

Two heuristics for picking the value:

- *Start at* `2 × max_concurrent_decodes`. With 32 concurrent decodes that gives 64 minimum; the remainder is prefill chunk headroom.
- *Profile the per-step latency curve* vs token count: pick the largest token count where adding 256 more tokens adds less than 5 % step latency. Below that point the GPU is bandwidth-bound; above it the GPU is compute-bound and you're paying linearly for every extra token.

Too low and prefill admission slows to a crawl (TTFT regresses). Too high and a single step's latency exceeds your TPOT SLO budget (TPOT regresses on tail).

**`max_chunk_size`** (vLLM exposes this as `long_prefill_token_threshold` / `chunked_prefill_max_seq_len`; SGLang uses `chunked_prefill_size`) — the largest single prefill chunk allowed in one step. Bounding it prevents one giant prefill from monopolizing the step's token budget. Typical values: 512–2048.

In practice modern engines auto-derive `chunk_size` from `max_num_batched_tokens` minus the current running-decode count. You only set it manually if you want a hard per-step TPOT ceiling.

**Calibration recipe.**

1. Pick a TPOT SLO (e.g. p99 ≤ 50 ms).
2. Measure single-step latency as a function of total tokens for your model and batch shape. Find `T_max` where step latency ≈ 0.8 × your TPOT SLO. (0.8 leaves headroom for jitter.)
3. Set `max_num_batched_tokens = T_max`.
4. Leave `max_chunk_size` at the engine default unless your prompts are all extremely long *and* you want a tighter per-step latency cap than the budget alone provides.

This recipe is the operational backbone of every chunked-prefill deployment.

---

## 5. The TTFT-vs-TPOT trade-off

Chunked prefill does not give you free latency; it redistributes it. The trade-off ([[sarathi-serve]] §5) has a clean shape:

- **TPOT improves a lot.** A single long-prompt admission that previously cost 300 ms of stall now costs 1–2 mixed-batch step latencies (~25–40 ms each). p99 TPOT drops 5–10× under bursty workloads.
- **TTFT regresses a little.** A 4096-token prompt that previously prefilled in one ~300 ms step now prefills over 8 chunks of ~25 ms each = ~200 ms total compute, but spread across many scheduling steps that also contain decode work for other requests. The end-to-end TTFT for that particular request can rise by 10–30 %.
- **Aggregate throughput improves.** Because the mixed batch keeps the GPU compute-bound on *every* step (not just on prefill steps), and because fewer requests get preempted under memory pressure, total tokens/sec rises 1.3–2.5× on realistic ShareGPT-style traces.

The paper's Figure 9 shows this directly: at the same offered load, Sarathi-Serve simultaneously achieves lower p99 TPOT (because of stall-free scheduling) and higher goodput@SLO (because of better compute utilization) than vanilla continuous batching. The only metric that worsens is mean TTFT for the longest prompts in the trace.

For most production workloads — chat, agent, RAG — this is the right trade. Users care more about smooth streaming than first-token. For SLO-bound workloads where TTFT is the hard constraint (e.g. voice assistants), the answer is prefill/decode disaggregation, covered in ch-09 ([[distserve]]).

---

## 6. Why this changes the rest of the stack

Chunked prefill is the reason a number of later features in vLLM and SGLang look the way they do:

- **Prefix caching (ch-07).** A prefix-cache hit converts the entire prefill into a no-op; the request enters decode immediately. Combined with chunked prefill, this means the scheduler can mix "new requests with cached prefixes (decode-only)" + "new requests with cold prefixes (prefill chunks)" + "running requests (decode tokens)" in the same step. The token-budget abstraction stays the same.
- **Speculative decoding (ch-14, ch-15).** A speculation step is "verify K candidate tokens for this running request." That's `K` decode tokens for one request in one step — directly compatible with the mixed-batch budget.
- **PagedAttention (ch-06).** The KV-cache manager in [[vllm-kv-cache-manager]] has to allocate blocks for a prefill chunk that's only partly through its prompt — a non-trivial memory-accounting problem that block-based allocation makes tractable.
- **Disaggregation (ch-09).** If chunked prefill still isn't enough — for example when prefill compute is so heavy it crowds out decode even one chunk at a time — the answer is to separate the two phases onto different GPUs entirely. Sarathi-Serve is the strongest colocated baseline that disaggregation has to beat.

These connections explain why ch-04 (continuous batching) and ch-05 (chunked prefill) are foundational: every later optimization is described in terms of how it composes with the mixed-batch scheduler.

---

## 7. The pseudocode you should be able to redraw from memory

```
# One serving step under Sarathi-Serve / vLLM V1 chunked prefill.
# Inputs: running (active decoding requests), waiting (queued prompts).

def step(running, waiting):
    T_max = config.max_num_batched_tokens
    decode_tokens = list(running)               # 1 token each
    budget = T_max - len(decode_tokens)

    chunks = []
    while waiting and budget > 0 and kv_has_capacity():
        req  = waiting.peek()
        left = req.prompt_len - req.prefill_pos
        take = min(left, budget, config.max_chunk_size)
        if not kv_cache_manager.allocate_slots(req, take):
            break                                # KV full → defer
        chunks.append((req, take))
        budget -= take
        req.prefill_pos += take
        if req.prefill_pos == req.prompt_len:
            waiting.pop(); running.append(req)

    batch = pack(prefill=chunks, decode=decode_tokens)
    out   = model.forward(batch)                 # single mixed forward pass
    for req in running: req.append_token(out[req])
    for req, taken in chunks: req.cache_K_V(out[req, :taken])
    return scheduler_output(batch, out)
```

Notice four things:

1. **One forward pass per step.** Mixed batch is the unit of work; there is no "prefill phase" or "decode phase" as a separate kernel.
2. **Decode is always served first.** Running TPOT is bounded by `step_latency(T_max)`.
3. **Chunks are admitted greedily under both the token budget and KV memory.** Either constraint can stop admission.
4. **A finished prefill becomes a decode request on the very next step.** No special promotion logic — the request just appears in `running` next iteration.

You should be able to redraw this loop from memory before moving on to ch-06. It's the scaffolding every subsequent serving feature attaches to.

---

## Connections and what's next

- **Back to [[continuous-batching]] (ch-04)** — chunked prefill is the upgrade that turns continuous batching from "iteration-level scheduling" into "token-budget scheduling." Same scheduler skeleton, finer-grained work unit.
- **Forward to [[pagedattention]] (ch-06)** — the mixed batch is only feasible because KV cache can be allocated and indexed per-block rather than as one contiguous tensor per request.
- **Forward to [[distserve]] (ch-09)** — the disaggregation answer to "what if chunked prefill still isn't enough?" Splits prefill and decode onto different GPU pools entirely.
- **Forward to [[vllm-scheduler]] (ch-16)** — production deep dive into the code that implements this loop, including preemption, structured-output bitmasks, and KV connector outputs.
- **Forward to [[sglang-scheduler]] (ch-17)** — SGLang's `chunked_prefill_size` / `schedule_policy` realize the same algorithm with a slightly different memory-pool front end.

## Further reading

- [[sarathi-serve]] — Agrawal et al. 2024, the canonical reference for chunked prefill in online serving; the SOSP 2024 paper.
- [[vllm-scheduler]] — vLLM V1 scheduler source `vllm/v1/core/sched/scheduler.py`; the production realization.
- [[continuous-batching]] — synthesis card for the iteration-level scheduling that chunked prefill builds on.
- [[orca]] — the OSDI 2022 paper that introduced iteration-level scheduling; predates chunked prefill but is the conceptual ancestor.

## Companion visualization

**[figures/chunked-prefill-timeline.html](figures/chunked-prefill-timeline.html)** — interactive timeline contrasting "static batch," "continuous batch (prefill blocks decode)," and "chunked-prefill mixed batch." Drag the `max_num_batched_tokens` slider to see how the per-step latency and the per-request TPOT respond.
