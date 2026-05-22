<!-- chapter: ch-16
     phase: framework-internals
     title: vLLM Internals — Scheduler + Cache Manager + Structured Output
     sources: [[vllm]], [[vllm-scheduler]], [[vllm-kv-cache-manager]], [[vllm-structured-output]], [[vllm-project]]
     figures: figures/vllm-architecture.html
-->

# Chapter 16 — vLLM Internals: Scheduler + Cache Manager + Structured Output

> **Core insight.** vLLM is the *reference open-source serving stack* for the entire course so far. Its V1 engine (2024+) bundles every technique you've read about — [[continuous-batching]] (ch-04), [[pagedattention]] (ch-06), prefix caching ([[ch-07]]), chunked prefill ([[ch-05]]), [[cuda-graphs-inference]] (ch-12), speculative decoding ([[ch-14]] / [[ch-15]]), TP/PP/EP ([[ch-13]]) — into one OpenAI-compatible engine. The architecture is **API server → engine → scheduler → executor → worker**; the scheduler holds three queues (`WAITING`, `RUNNING`, optionally `SWAPPED`), the cache manager owns the block pool with hash-based prefix reuse, and structured output is grammar-bitmask masking on the sampler.
>
> **Guideline.** Don't tune vLLM by twiddling random flags. The four flags that matter are `--gpu-memory-utilization`, `--max-num-batched-tokens`, `--max-num-seqs`, and `--enable-prefix-caching`. Everything else is downstream of these. Read the scheduler source (`vllm/v1/core/sched/scheduler.py`) before opening a ticket.

---

## Why this chapter exists

You've now covered every technique vLLM uses. This chapter shows where each one *lives in the code*: which file owns the scheduling decision, which file owns the KV cache, which file owns the structured-output bitmask. The point is not to memorize line numbers — it's to give you the mental map you need to (a) debug production incidents, (b) read the source when docs are stale, and (c) understand how the systems story compresses into one engine.

Three things to walk away with:

1. The V1 engine's **control flow**: API server submits requests → engine puts them in a queue → scheduler builds a step → executor runs the model → outputs stream back.
2. The scheduler's three-queue state machine and how it interacts with the KV cache manager's block-pool allocator.
3. How structured output (`guided_json`, `guided_regex`, `guided_grammar`) snaps into the sampling step via per-step token bitmasks.

This chapter assumes you've already read ch-04 (continuous batching), ch-06 (PagedAttention), ch-12 (CUDA Graphs), and ch-14 (speculative decoding). It synthesizes them into vLLM's concrete architecture.

---

## 1. Architecture overview

The V1 engine (the rewrite that replaced V0 starting late 2024) has a clean process model:

```
        ┌──────────────────┐
        │  HTTP client     │  (OpenAI Python client, curl, etc.)
        └────────┬─────────┘
                 │   POST /v1/chat/completions
                 ▼
        ┌──────────────────┐
        │  API server      │  vllm/entrypoints/openai/api_server.py
        │  (FastAPI)       │  → validates request, builds RequestInput
        └────────┬─────────┘
                 │   submit(request)
                 ▼
        ┌──────────────────┐
        │  AsyncLLM /      │  vllm/v1/engine/async_llm.py
        │  AsyncEngineCore │  → owns input/output queues per request
        └────────┬─────────┘
                 │   add_request(request)
                 ▼
        ┌──────────────────┐
        │  EngineCore loop │  vllm/v1/engine/core.py
        │  (event loop)    │  → tight while-True: scheduler.schedule() → executor.execute_model()
        └────────┬─────────┘
                 │
                 ▼
        ┌──────────────────┐         ┌──────────────────┐
        │   Scheduler      │ ◄─────► │  KVCacheManager  │
        │   vllm/v1/core/  │ allocate│  vllm/v1/core/   │
        │   sched/         │  slots  │  kv_cache_       │
        │   scheduler.py   │         │  manager.py      │
        └────────┬─────────┘         └──────────────────┘
                 │   SchedulerOutput (which requests, how many tokens, block tables)
                 ▼
        ┌──────────────────┐
        │  Executor        │  vllm/v1/executor/{uniproc,multiproc,ray}_executor.py
        │  (rank manager)  │  → spawns / coordinates one worker per (TP, PP) rank
        └────────┬─────────┘
                 │   execute_model(scheduler_output)
                 ▼
        ┌──────────────────┐
        │  Worker          │  vllm/v1/worker/gpu_worker.py + gpu_model_runner.py
        │  (one per rank)  │  → loads model, runs forward, returns sampled tokens
        └────────┬─────────┘
                 │   ModelRunnerOutput
                 ▼
        (back up the chain, streamed to client)
```

Process model: **one process per executor rank** (multi-process executor, default), plus the API server. Inter-process communication via ZMQ for control + shared memory for tensors.

### 1.1 The single-stream invariant

The V1 engine has one critical invariant: **at most one model forward pass at a time across all ranks**. The scheduler emits a SchedulerOutput; the executor runs the model; the result returns; the scheduler emits the next SchedulerOutput. No overlapping forwards.

This is what makes continuous batching coherent — the scheduler has a complete view of state when deciding what to schedule next.

---

## 2. The scheduler (`vllm/v1/core/sched/scheduler.py`)

This is the heart of vLLM. ~1500 lines (as of v0.7), but the core logic is one function: `Scheduler.schedule()`.

### 2.1 Request states

A request's lifecycle:

```
                add_request()
                     │
                     ▼
              ┌──────────────┐
              │   WAITING    │   (not yet running; waiting for cache capacity)
              └──────┬───────┘
                     │   capacity available + admission ok
                     ▼
              ┌──────────────┐
              │   RUNNING    │   (in the active batch each step)
              └──────┬───────┘
                     │   (a) finishes naturally    (b) preempted under cache pressure
                     ▼                              ▼
                ┌─────────┐                  ┌───────────┐
                │ FINISHED│                  │  SWAPPED  │ (KV blocks freed; can resume)
                └─────────┘                  └─────┬─────┘
                                                   │   capacity available again
                                                   ▼
                                           (back to WAITING or RUNNING)
```

`SWAPPED` is the preemption state. When cache pressure forces eviction, running requests' KV blocks are either swapped to CPU (`--swap-space` configured) or recomputed from scratch (default).

### 2.2 The schedule() loop

Each step, in order:

```python
def schedule(self) -> SchedulerOutput:
    # 1. Pop any finished / preempted requests, freeing their blocks.
    self._free_finished()

    # 2. Try to schedule RUNNING requests first (decode tokens for each).
    for req in self.running:
        num_new_tokens = self._get_num_new_tokens(req)  # usually 1 for decode
        if not self.kv_cache_manager.has_capacity(req, num_new_tokens):
            self._preempt(req)  # frees its blocks
            continue
        blocks = self.kv_cache_manager.allocate_slots(req, num_new_tokens)
        scheduled_running.append((req, num_new_tokens, blocks))
        token_budget -= num_new_tokens
        if token_budget <= 0:
            break

    # 3. Admit WAITING requests if token budget allows.
    while self.waiting and token_budget > 0:
        req = self.waiting[0]
        # Prefix cache lookup: maybe we can skip recomputing the first N tokens
        computed_blocks = self.kv_cache_manager.get_computed_blocks(req)
        num_new_tokens = min(
            req.num_prompt_tokens - len(computed_blocks) * block_size,
            token_budget,
            self.max_num_batched_tokens,
        )
        # Chunked prefill: maybe schedule only part of the prompt this step
        if num_new_tokens < req.remaining_prompt_tokens:
            scheduled_chunked_prefill.append(...)
        else:
            scheduled_full.append(...)
        token_budget -= num_new_tokens

    # 4. Build SchedulerOutput: which requests, how many tokens each,
    #    block tables for the attention kernel, sampling params.
    return SchedulerOutput(
        scheduled_running=scheduled_running,
        scheduled_new=scheduled_new,
        finished_req_ids=...,
        ...
    )
```

The order is: **finish first, then keep running requests going, then admit new ones**. The point is *liveness*: don't starve in-progress requests by always preferring new ones.

### 2.3 Three queues, one token budget

The scheduler caps how much compute it commits per step via `max_num_batched_tokens` (default ~2048). This is the **token budget** — total tokens (prefill + decode) that can be in one forward pass.

```
Example token budget = 2048
RUNNING decode requests: 32 × 1 token each = 32 tokens
Remaining budget = 2016 tokens

WAITING request A: prefill 5000 tokens
  → schedule 2016 tokens of A as a chunked prefill
  → A stays in WAITING (with progress marker)
  → A will resume on the next step with its remaining 2984 tokens

WAITING request B: prefill 200 tokens
  → could not schedule (budget exhausted)
  → B waits next step
```

The chunked-prefill integration ([[ch-05]] Sarathi-Serve idea) is built into this loop: prefill is just "scheduled tokens that aren't decode tokens". The budget accommodates both.

### 2.4 Preemption strategies

Under cache pressure, vLLM has two preemption modes:

- **Recompute** (default): free the request's KV blocks; on resume, recompute the prompt from scratch. Cheaper memory, more compute on resume.
- **Swap** (with `--swap-space N`): copy the request's KV blocks to CPU memory; on resume, copy back to GPU. Saves recompute but adds CPU↔GPU bandwidth.

Recompute is generally preferred unless prompts are very long (>10k tokens) where recompute cost dominates.

### 2.5 Priority and fair scheduling

V1 has a priority field on requests; higher priority requests get scheduled first within the WAITING queue. This is the hook for SLO-aware scheduling ([[ch-10]]: VTC, Niyama). Default priority is FIFO.

---

## 3. The KV cache manager (`vllm/v1/core/kv_cache_manager.py`)

The cache manager owns the **block pool** — a pre-allocated bank of fixed-size KV blocks (typically 16 tokens each). Requests get blocks; the attention kernel reads through block tables ([[pagedattention]] from ch-06).

### 3.1 Block pool structure

```
Total blocks = (GPU memory - model weights - activations) / block_size_bytes

block_size_bytes = 2 (K + V) × num_layers × kv_heads × head_dim × block_tokens × dtype_bytes

For Llama-3-70B on H100 (80 layers, 8 GQA KV-heads, head_dim=128, block_tokens=16, bf16):
  block_size_bytes = 2 × 80 × 8 × 128 × 16 × 2 = 5,242,880 bytes = 5 MB per block

H100 80 GB - 70B weights × 2 bytes - activations ≈ 30 GB free
Total blocks ≈ 30 GB / 5 MB = 6000 blocks

Total tokens served = 6000 × 16 = 96,000 tokens across the fleet
```

This is the per-rank capacity at TP=1. At TP=8, KV cache shards 8-way → 8× more block capacity per rank (same physical pool size, fewer bytes per block per rank).

### 3.2 Block tables — per request

Each request maintains a list of block IDs:

```python
class Request:
    block_table: list[int]   # e.g. [42, 1733, 89, 902, ...]
                              # → tokens [0..15] live in physical block 42, etc.
```

The attention kernel (PagedAttention) accepts the block table as an indirect-index argument and reads KV from non-contiguous physical blocks. No memcpy on allocation; no fragmentation.

### 3.3 The block pool free-list

```python
class BlockPool:
    def __init__(self, num_blocks):
        self.free_blocks = deque(range(num_blocks))
        self.allocated = {}   # block_id → ref_count

    def allocate(self, n) -> list[int]:
        return [self.free_blocks.popleft() for _ in range(n)]

    def free(self, block_ids):
        for bid in block_ids:
            self.allocated[bid] -= 1
            if self.allocated[bid] == 0:
                self.free_blocks.append(bid)
```

Reference counting handles **copy-on-write** for parallel sampling: a single base prompt shared by N samples references the same blocks until divergence; on first divergent token in sample i, allocate a new block, copy the existing one's content for the divergence point, and unshare. APC integrates here too — a finished request's blocks aren't freed if their content might be reused.

### 3.4 Automatic Prefix Cache (APC)

When `--enable-prefix-caching` is set, the cache manager hashes block content and indexes blocks by hash. On admission, `get_computed_blocks(request)` walks the prompt forward, hashing 16-token chunks and looking each up:

```python
def get_computed_blocks(request):
    computed = []
    for chunk in chunks_of_16(request.prompt_tokens):
        h = hash(prefix_hash, chunk)
        if h in self.cached_blocks:
            computed.append(self.cached_blocks[h])
            prefix_hash = h
        else:
            break
    return computed
```

The walk stops at the first chunk that doesn't match → the matched prefix is "for free", remaining tokens are prefilled normally.

Eviction is LRU on cached blocks: when the free-list is empty, evict the least-recently-used cached block. Cached blocks are only evicted when in-use blocks need capacity.

**Production impact (vLLM benchmarks)**:
- Chat with fixed system prompt: 50-90% APC hit rate → 5-10× TTFT speedup.
- RAG with shared retrieval context: 60-80% APC hit rate → 4-6× TTFT speedup.
- Arbitrary user prompts: 10-30% APC hit rate → 1.5-2× TTFT.

### 3.5 KV cache groups

V1 supports **multiple KV cache groups** per model — e.g. one group for full attention, another for sliding-window attention (used in Mistral 7B, Gemma). The block pool is partitioned across groups; each group has its own block_size.

`SingleTypeKVCacheManager` (`vllm/v1/core/single_type_kv_cache_manager.py`) is the per-group manager; the top-level `KVCacheManager` dispatches to the appropriate per-group instance.

---

## 4. The executor and worker

### 4.1 Executor flavors

- **UniprocExecutor**: single process, single rank. For TP=PP=1 dev work.
- **MultiprocExecutor**: spawns one subprocess per rank. The default for TP>1.
- **RayExecutor**: uses Ray actors as workers. Used for PP>1 across nodes.

The executor's job: receive a SchedulerOutput, broadcast to all workers, collect ModelRunnerOutputs, return.

### 4.2 GPU worker

Each rank's worker (`vllm/v1/worker/gpu_worker.py`):

```python
class GPUWorker:
    def __init__(self, config, rank):
        self.model_runner = GPUModelRunner(config, device=f"cuda:{rank}")
        self.kv_caches = self.model_runner.initialize_kv_cache(num_blocks)

    def execute_model(self, scheduler_output):
        input_ids, positions, attn_metadata = build_inputs(scheduler_output)
        with self.kv_cache_context():
            logits = self.model_runner.execute(input_ids, positions, attn_metadata, self.kv_caches)
        sampled = self.model_runner.sample(logits, scheduler_output.sampling_params)
        return ModelRunnerOutput(sampled_tokens=sampled, logprobs=...)
```

`GPUModelRunner` (`vllm/v1/worker/gpu_model_runner.py`) is the bridge between scheduler-level data and the model's actual `forward()`. It builds the attention metadata (block tables, slot mapping, sequence lengths), invokes the attention backend (FlashAttention, FlashInfer, or xformers), and runs the model.

### 4.3 CUDA graph capture

For decode steps with stable shapes, GPUModelRunner captures **piecewise CUDA graphs** ([[cuda-graphs-inference]] from ch-12) — one graph per (batch-size bucket, num_kv_cache_blocks). At runtime, the engine selects the right graph and replays it; saves ~10-30% on decode latency.

Configuration: `--enforce-eager` disables CUDA graphs (useful for debugging).

---

## 5. Structured output (`vllm/v1/structured_output/`)

vLLM supports five constraint types via the OpenAI-extension request fields:

- `guided_choice`: output must be one of N strings
- `guided_regex`: output must match a regex
- `guided_json`: output must satisfy a JSON schema
- `guided_grammar`: output must satisfy a context-free grammar (EBNF)
- `structural_tag`: structured XML-style tags

### 5.1 The token bitmask pipeline

For each constrained request, at each decode step:

```
1. Compute grammar state for the request (advance from previous state).
2. Compute allowed-token bitmask: vocab-sized bool tensor, True if token can follow.
3. Apply mask to logits: logits[~mask] = -inf
4. Sample as usual.
```

Two backends supported:

- **xgrammar** (default, since v0.6): the fastest grammar-bitmask compiler; precomputes a FSM and per-state bitmasks. ~5-15 µs per step overhead.
- **outlines**: the original backend; slower compile but supports a wider grammar surface.

### 5.2 Backend selection

```bash
vllm serve <model> --guided-decoding-backend xgrammar
# or
vllm serve <model> --guided-decoding-backend outlines
```

### 5.3 JSON schema example

```python
import requests

response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "meta-llama/Llama-3-70B-Instruct",
        "messages": [{"role": "user", "content": "Extract a person's name and age from: John is 30."}],
        "guided_json": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name", "age"],
        },
    },
)
# Output guaranteed to be {"name": "John", "age": 30} or some other schema-valid JSON.
```

### 5.4 Integration with scheduler

The structured-output manager is consulted *before* sampling each step. The scheduler tracks per-request grammar state; on each forward pass, it asks the manager for the bitmask and passes it through to the sampler.

**Caveat**: structured output is *not* free. Per-step bitmask computation adds 5-50 µs depending on grammar complexity. For large schemas (>1000 productions), this can exceed 5% of decode latency.

---

## 6. The four production knobs

```bash
vllm serve meta-llama/Llama-3-70B-Instruct \
    --tensor-parallel-size 8 \
    --gpu-memory-utilization 0.92 \       # fraction of GPU memory to use for vLLM
    --max-num-batched-tokens 8192 \       # token budget per scheduler step
    --max-num-seqs 256 \                  # cap on RUNNING requests
    --enable-prefix-caching \             # turn on APC
    --swap-space 16                       # CPU swap in GB per rank (default 4)
```

### 6.1 `gpu-memory-utilization` (default 0.9)

The fraction of GPU memory vLLM is allowed to use. Higher = more KV cache blocks = more concurrency. Lower = safer (less risk of OOM from activation spikes).

**Tune**: start at 0.9. If you see OOM on long sequences, drop to 0.85. If you have lots of headroom, push to 0.95.

### 6.2 `max-num-batched-tokens` (default 2048 or model-dependent)

The token budget per scheduler step. Higher = more prefill throughput, but each step is longer (worse decode latency for already-running requests).

**Tune**: for chat-heavy (decode-dominated): 2048 is fine. For prefill-heavy (large prompts): 8192-16384.

### 6.3 `max-num-seqs` (default 256)

The maximum number of RUNNING requests. Caps the decode batch size. Higher = more throughput, more KV memory.

**Tune**: rarely binding because KV cache capacity binds first. Leave at 256 unless you have explicit memory headroom.

### 6.4 `enable-prefix-caching`

Turn on APC. Almost always a win for chat / RAG. Default off historically (V0), on by default in some V1 builds.

---

## 7. Pitfalls

- **Don't read V0 code thinking it's V1.** V0 files live in `vllm/core/`, `vllm/engine/` (no `v1/`). The V1 rewrite is in `vllm/v1/`. Major API and architecture differences.
- **`--enforce-eager` is for debugging only.** Disables CUDA graphs → 30-50% decode-latency regression. Don't ship to prod.
- **APC requires identical tokenization.** Two clients sending the same prompt text but with different chat templates → cache miss. Normalize the template at the client.
- **Preemption causes TPOT spikes.** If you see `preempted_count` rising in metrics, you're cache-pressured. Drop `gpu-memory-utilization` or add ranks.
- **Structured-output bitmask compile time is per-grammar.** Cache compiled grammars across requests using the same schema; vLLM does this automatically by schema hash.
- **`max-num-batched-tokens` lower than your longest prompt** means chunked prefill triggers for every long prompt. That's fine for throughput but increases TTFT for those requests.
- **MultiprocExecutor + spec-dec.** Some early V1 builds had bugs with spec-dec under multi-process executor; verify on your version.
- **Spec-dec interacts with chunked prefill.** During the chunked-prefill phase, spec-dec is paused (only the target's prefill chunk runs). Spec-dec resumes once the request enters decode.

---

## 8. Practitioner's cheat-sheet

```bash
# Production deployment for Llama-3-70B on 8×H100
vllm serve meta-llama/Llama-3-70B-Instruct \
    --tensor-parallel-size 8 \
    --gpu-memory-utilization 0.92 \
    --max-num-batched-tokens 8192 \
    --enable-prefix-caching \
    --port 8000

# Production deployment with EAGLE-2 speculative decoding
vllm serve meta-llama/Llama-3-70B-Instruct \
    --tensor-parallel-size 8 \
    --enable-prefix-caching \
    --speculative-config '{"method":"eagle","model":"yuhuili/EAGLE-LLaMA3-Instruct-70B","num_speculative_tokens":5}'

# Structured output (xgrammar backend)
vllm serve meta-llama/Llama-3-70B-Instruct \
    --tensor-parallel-size 8 \
    --guided-decoding-backend xgrammar
```

```python
# Offline API — same engine, no HTTP overhead
from vllm import LLM, SamplingParams
llm = LLM(
    model="meta-llama/Llama-3-70B-Instruct",
    tensor_parallel_size=8,
    gpu_memory_utilization=0.92,
    enable_prefix_caching=True,
    max_num_batched_tokens=8192,
)
outputs = llm.generate(
    ["Tell me about transformers"],
    SamplingParams(temperature=0.7, top_p=0.9, max_tokens=512),
)
```

---

## Connections and what's next

- **[[continuous-batching]] / ch-04** — the scheduler implements this; understand the algorithm before reading the code.
- **[[pagedattention]] / ch-06** — the KV cache manager is its production implementation.
- **[[cuda-graphs-inference]] / ch-12** — piecewise capture lives in `gpu_model_runner.py`.
- **[[ch-13]]** — TP/PP/EP knobs are passed through the executor.
- **[[ch-14]] / [[ch-15]]** — speculative decoding integrates via `speculative_config` on the engine.
- **[[sglang]] / ch-17** — the next chapter's framework, contrast with vLLM (RadixAttention vs APC, Python frontend DSL vs OpenAI API).

## Further reading

- [[vllm]] — public docs + architecture overview.
- [[vllm-scheduler]] — V1 scheduler source path.
- [[vllm-kv-cache-manager]] — KV cache + block pool + APC.
- [[vllm-structured-output]] — xgrammar / outlines integration.
- [[vllm-project]] — project-level summary.

Code paths to bookmark:
- `vllm/v1/core/sched/scheduler.py` — the scheduler.
- `vllm/v1/core/kv_cache_manager.py` — block pool + APC.
- `vllm/v1/core/block_pool.py` — free-list mechanics.
- `vllm/v1/structured_output/` — grammar backends.
- `vllm/v1/worker/gpu_model_runner.py` — model invocation bridge.
- `vllm/v1/engine/core.py` — the engine loop.

## Companion visualization

**[figures/vllm-architecture.html](figures/vllm-architecture.html)** — interactive diagram of the V1 engine flow with hover-to-explain for each component (API server, engine, scheduler, KV cache manager, executor, worker).

## Excerpts

- [excerpts/vllm-scheduler.md](excerpts/vllm-scheduler.md) — V1 schedule() loop, three queues, chunked prefill, preemption.
- [excerpts/vllm-kv-cache-manager.md](excerpts/vllm-kv-cache-manager.md) — block pool, block tables, APC hash chain.
- [excerpts/vllm-structured-output.md](excerpts/vllm-structured-output.md) — xgrammar bitmask pipeline, guided_json example.
- [excerpts/vllm-production-knobs.md](excerpts/vllm-production-knobs.md) — the four production flags, when to tune which.
