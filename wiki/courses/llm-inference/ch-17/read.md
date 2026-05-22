<!-- chapter: ch-17
     track: framework-internals
     title: SGLang Internals — RadixAttention + HiCache + Frontend Language
     sources: [[sglang]], [[sglang-scheduler]], [[sglang-radixattention]], [[sglang-hicache]], [[sglang-structured-output]], [[sglang-project]]
     figures: figures/sglang-architecture.html
-->

# Chapter 17 — SGLang Internals: RadixAttention + HiCache + Frontend Language

> **Core insight.** SGLang is a co-design: a Python frontend DSL (`gen` / `select` / `fork`) lets the *user* expose program structure (shared system prompt, parallel branches, multi-turn state); the runtime uses that structure to make prefix reuse a first-class scheduling primitive via **RadixAttention** (token-keyed radix tree of KV blocks) extended by **HiCache** (L1 GPU → L2 host → L3 distributed-storage tiers). Where vLLM treats every request as opaque and recovers prefix reuse post-hoc via block-hash, SGLang co-designs frontend semantics + backend cache so the prefix structure is *known*, not inferred.
>
> **Guideline.** Pick SGLang for workloads where prompts share heavy prefixes: agent loops, few-shot templates, multi-turn chat with long system prompts, RAG over a small document corpus, multi-question QA over the same context. Expect 2–6× higher cache hit rate and 1.5–5× higher throughput over vLLM on these workloads. For arbitrary user prompts with no overlap, RadixAttention degrades to vLLM-equivalent block-cache; the lift is workload-shape-dependent, not free.

---

## Why this chapter exists

Most production serving systems treat KV cache as private per-request state. SGLang treats it as a shared filesystem indexed by token prefix. That single architectural decision — keeping a **token-keyed radix tree of KV blocks across all live and recently-finished requests** — is what makes SGLang faster than vLLM on agent / few-shot / multi-turn workloads, and why DeepSeek, Moonshot, and 01.AI all build production serving on SGLang.

Three things to walk away with:

1. **The frontend → runtime contract.** Why `fork()` / `gen()` are not Python conveniences but *cache hints* — they tell the runtime which prefixes are about to be shared, so the scheduler can pre-warm them.
2. **The RadixAttention data structure and lifecycle.** Insert / match / split / evict on a token-keyed radix tree, with reference counting for live requests. This is the entire SGLang prefix-cache machinery in ~600 lines of `radix_cache.py`.
3. **The HiCache 3-tier story.** GPU is too small to hold all reusable prefixes; recomputing prefill is too expensive. The 2025 answer is to spill cold prefixes to CPU and to a cluster-shared L3 (Mooncake / 3FS / NIXL / AIBrix). The cache hit rate per tier is the metric that matters.

This sits across [[sglang]], [[sglang-scheduler]], [[sglang-radixattention]], [[sglang-hicache]], [[sglang-structured-output]], and [[sglang-project]] in the raw-data library.

---

## 1. Architecture overview — the four-layer stack

SGLang is structurally a four-layer pipeline:

```
┌─────────────────────────────────────────────────────────────┐
│ Frontend Language (optional)                                │
│   Python DSL: gen / select / fork / system / user / role    │
│   Compiles user programs into request graphs that expose    │
│   prefix structure to the backend                           │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Server / API layer                                          │
│   • OpenAI-compatible /v1/chat/completions, /v1/completions │
│   • Native /generate (sampling_params, json_schema, regex)  │
│   • Offline Engine API (in-process, no HTTP)                │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Scheduler (python/sglang/srt/managers/scheduler.py)         │
│   • Waiting / running queues                                │
│   • Cache-aware admission (RadixAttention prefix match)     │
│   • Chunked prefill + continuous batching                   │
│   • Coordinates grammar backends, LoRA, spec decoding,      │
│     PD-disaggregation                                       │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Backend / Model Worker                                      │
│   • RadixAttention KV cache (radix_cache.py)                │
│   • HiCache tiers: L1 GPU → L2 host → L3 distributed        │
│   • Paged-attention kernels (FlashInfer / Triton / CUTLASS) │
│   • Tensor parallel / expert parallel comms                 │
└─────────────────────────────────────────────────────────────┘
```

The boundary line that matters is between scheduler and backend: **the scheduler queries the radix cache *before* admitting a request**, so the token budget it estimates already accounts for prefix-cache hits. This is the difference between "prefix caching as an optimization" (vLLM's APC, bolted onto block hash) and "prefix caching as scheduling input" (SGLang's RadixAttention, queried at admission).

You can see this in [[sglang-scheduler]]: cache-aware scheduling is selected via `schedule_policy=lpm` ("longest prefix match") rather than `fcfs`. Under `lpm`, requests with longer cached prefixes get priority — they consume less GPU work, so admitting them first improves both per-request TTFT *and* aggregate throughput.

---

## 2. The frontend language — `gen` / `select` / `fork`

The SGLang frontend is a Python DSL that looks like async/await but compiles to a request graph the backend can pre-plan. The three primitives are:

```python
import sglang as sgl

@sgl.function
def multi_turn_qa(s, doc, questions):
    s += sgl.system("You are an expert reading-comprehension assistant.")
    s += sgl.user(f"Document:\n{doc}")          # ← long shared prefix
    forks = s.fork(len(questions))               # ← parallel branches
    for f, q in zip(forks, questions):
        f += sgl.user(q)
        f += sgl.assistant(sgl.gen("answer", max_tokens=256))
    return [f["answer"] for f in forks]
```

What this buys you:

| Primitive | What it does | Cache effect |
|-----------|--------------|--------------|
| `gen(name, ...)` | Issue a generation call; bind the result to `name` | Each `gen` becomes one runtime request; the prefix up to this point is the cache key |
| `select(name, choices)` | Constrained generation — pick one of `choices` | Same cache effect as `gen` + grammar mask |
| `fork(N)` | Branch the conversation N ways | All N branches share the same prefix; the runtime guarantees the radix tree holds the shared KV blocks until all branches finish |
| `s += sgl.system(...)` / `user(...)` / `assistant(...)` | Append messages to the program state | Extends the cached prefix path |

The frontend never has to *tell* the scheduler "please reuse this prefix" — the request graph it emits already expresses the structure, and the radix cache exploits it automatically.

**Why coroutine-style.** Each `gen` is a suspension point. The Python interpreter doesn't block waiting for the model; it yields, the runtime schedules the call, and resumes the coroutine when tokens arrive. This is the same pattern as `asyncio`, but specialized so the runtime sees the whole program graph (including unexecuted future branches), not just the current await.

**The opt-out path.** You don't have to use the frontend — the OpenAI-compatible `/v1/chat/completions` and native `/generate` endpoints expose the same backend without DSL. Most production traffic to SGLang is OpenAI-compatible. The frontend value is concentrated in agent / multi-step workflows where the DSL exposes prefix structure that a raw API call cannot.

---

## 3. RadixAttention — the data structure

[[sglang-radixattention]] is the SGLang signature mechanism. It's a **radix tree (compressed prefix trie) keyed by token sequences**, where each tree edge owns one or more KV-cache blocks.

### 3.1 The tree shape

A radix tree compresses chains of single-child nodes into one edge. For LLM serving this matches reality: most prompts share a long common prefix (the system prompt) and diverge late. The tree:

```
root
 └── ["You are an expert ...", "Document: <doc-tokens>"]  ← edge holds KV blocks for ~5k tokens
      ├── ["Question 1: ..."]    → KV blocks for question 1 + its answer
      ├── ["Question 2: ..."]    → ...
      └── ["Question 3: ..."]    → ...
```

The shared edge stores the KV blocks for the system prompt + document *exactly once*. Every leaf request "owns" a reference to the shared edge but pays no extra memory and no extra prefill compute for those tokens.

### 3.2 The five operations

From [[sglang-radixattention]] + `python/sglang/srt/mem_cache/radix_cache.py`:

| Operation | What it does | When called |
|-----------|--------------|-------------|
| `match_prefix(token_ids)` | Walk the tree, return (longest matching prefix length, node) | Scheduler admission: how much prefill can we skip? |
| `insert(token_ids, kv_blocks)` | Add a new path to the tree, possibly splitting an existing edge | After a request completes its prefill |
| `_split_node(node, prefix_len)` | Edge has 5k tokens but new request matches only first 3k — split the edge | During `insert` when partial-match occurs |
| `evict(num_tokens_to_free)` | LRU-evict leaf nodes whose ref count = 0 | When KV memory pressure exceeds threshold |
| `inc_ref / dec_ref` | Pin / unpin a node so it can't be evicted | Live requests pin their prefix path |

The critical correctness property: **a node can only be evicted when ref_count == 0**. Live requests pin their prefix back to root, so the system prompt KV blocks stay resident as long as any in-flight request uses them. Eviction is LRU among unpinned leaves.

### 3.3 The lifecycle in one paragraph

A request arrives with token sequence `T = [t_0, ..., t_n]`. The scheduler calls `match_prefix(T)` → returns `(k, node)` meaning the first `k` tokens are cached. The scheduler increments the ref count on `node` (pinning the prefix), allocates KV blocks for the remaining `n − k` tokens, and prefills only those `n − k` tokens. When the request finishes, the scheduler calls `insert` to add the new completion path (allowing future reuse), then `dec_ref` on `node`. If memory pressure is high, `evict` walks the LRU list of unpinned leaves and frees their blocks.

### 3.4 Concrete hit-rate numbers

From [[sglang-radixattention]] + the SGLang paper (arXiv 2312.07104):

| Workload | RadixAttention hit rate | vLLM APC hit rate | Throughput uplift |
|----------|------------------------:|------------------:|------------------:|
| Multi-turn chat with 2k-token system prompt | 60–80 % | 50–70 % | 1.3–1.5× |
| Few-shot prompting (8 shots, varying queries) | 80–95 % | 70–85 % | 2–3× |
| Tree-of-thought / agent loops with `fork()` | 90–95 % | not applicable (no fork hint) | 3–5× |
| Multi-QA over shared 100k document | 95–99 % | 60–80 % | 4–6× |
| Arbitrary user chat (no shared prefix) | 0–20 % | 0–20 % | ~1× (parity) |

The lift is **workload-shape-dependent**, not a constant speedup. When prompts don't overlap, RadixAttention degrades gracefully to vLLM-equivalent behaviour — same paged blocks, no extra cost.

---

## 4. Scheduler — `lpm` vs `fcfs` and the chunked-prefill knobs

From [[sglang-scheduler]] (`python/sglang/srt/managers/scheduler.py`):

The scheduler runs an event loop. Each iteration:

1. Drain finished requests from the running batch; `dec_ref` their prefix nodes.
2. Look at the waiting queue. For each request, call `match_prefix` to get cached-prefix length.
3. Under `schedule_policy=lpm` (longest prefix match), sort waiting requests by `(prefix_match_len / total_prompt_len)` descending. This admits cache-friendly requests first.
4. Under `schedule_policy=fcfs`, ignore prefix length and admit in arrival order.
5. Subject to token budget (`max_num_batched_tokens`), KV-memory budget, and request-count budget (`max_running_requests`), admit as many new requests as fit.
6. Form a mixed prefill+decode batch (chunked prefill — see ch-05 + [[sglang-scheduler]]).
7. Forward pass. Append generated tokens. Loop.

The user-facing knobs that matter:

| Flag | Default | What it tunes |
|------|--------:|---------------|
| `--schedule-policy` | `lpm` | `lpm` (cache-aware) vs `fcfs` (latency-fair) |
| `--schedule-conservativeness` | `1.0` | Multiplier on token-budget estimate; <1 admits more aggressively, >1 admits less |
| `--max-running-requests` | model-dependent | Hard cap on concurrent in-flight requests |
| `--max-total-tokens` | derived from `gpu_memory_utilization` | Total KV-cache slot budget |
| `--chunked-prefill-size` | `8192` | Max tokens of prefill mixed into one forward pass |
| `--max-prefill-tokens` | `16384` | Hard cap on single-request prefill chunk |
| `--disable-radix-cache` | off | Kill switch — disables RadixAttention, falls back to per-request KV |

**Production pattern.** Leave `--schedule-policy lpm` on. Raise `--max-running-requests` until KV-cache thrash appears (tail TTFT spikes). Tune `--chunked-prefill-size` to balance prefill burst latency vs decode throughput — see ch-05 for the chunked-prefill mechanics; SGLang's defaults are an inheritance of Sarathi-Serve.

---

## 5. HiCache — the three-tier KV hierarchy

[[sglang-hicache]] extends RadixAttention from a single GPU-resident tree to a **three-tier hierarchy** modeled on CPU cache design:

| Tier | Storage | Capacity | Latency to access | Role |
|------|---------|---------:|------------------:|------|
| **L1** | GPU HBM | 10–80 GB | ~µs (compute speed) | Hottest KV; immediate decode reuse |
| **L2** | Host RAM | 100–1000 GB | ~ms (PCIe transfer) | Warm prefixes; reusable across requests |
| **L3** | Distributed storage | TB–PB | ~10–100 ms | Cluster-shared; cross-instance reuse |

L3 backends supported in SGLang HiCache: **Mooncake** (KV-centric KVStore from Moonshot, see ch-09), **3FS** (DeepSeek's distributed FS), **NIXL** (NVIDIA's KV transfer library), **AIBrix KVCache**.

### 5.1 HiRadixTree — the metadata

Standard RadixAttention's `RadixCache` becomes `HiRadixCache`. Each node now carries a *location set*:

```python
class HiRadixNode:
    children: dict[token_chunk → HiRadixNode]
    kv_locations: set[Location]   # {GPU(blocks), CPU(blocks), L3(uri)}
    last_access_time: float
    ref_count: int
```

A single prefix's KV can live simultaneously at multiple tiers — e.g. recently-used prefix is both in GPU L1 *and* in CPU L2, so it can be evicted from GPU without recompute (just rehydrate from CPU).

### 5.2 The match → fetch → write-back workflow

```
1. Request arrives → HiRadixCache.match_prefix(tokens)
2. Walk tree; for matched segments, check kv_locations:
   - L1 hit:        zero work, KV blocks already in GPU
   - L2 hit:        async DMA copy host → GPU
   - L3 hit:        async fetch from distributed store → host → GPU
   - miss:          standard prefill on the unmatched suffix
3. Compute the new suffix's KV during prefill
4. Write-back: enqueue eviction-survivors to L2; long-term-hot prefixes to L3
```

The async fetch overlaps with the (smaller, suffix-only) prefill compute, so an L2/L3 hit typically still beats a full recompute.

### 5.3 Per-tier hit-rate metrics

The metric you must monitor in production:

| Metric | Healthy range |
|--------|---------------|
| L1 (GPU) hit rate | 40–80 % depending on workload |
| L2 (CPU) hit rate (over L1 misses) | 20–60 % |
| L3 (distributed) hit rate (over L1+L2 misses) | 10–40 % |
| End-to-end hit rate (any tier) | 60–95 % for shared-prefix workloads |
| Offload bandwidth utilization (PCIe / RDMA) | <50 % of link peak (else thrash) |

**The bottleneck is bandwidth, not capacity.** A 64 GB/s PCIe link can move ~8 GB of KV per second. A Llama-3-70B request with 8k cached tokens has ~1 GB of KV (see ch-03 KV formula). At 8 cached-prefix-hit requests/sec sustained, you saturate PCIe. HiCache works because most requests *don't* need a full prefix rehydrate — partial L1 hits are common.

---

## 6. Structured output — grammar backends + prefix cache interaction

From [[sglang-structured-output]]: SGLang supports JSON schema, regex, EBNF, and structural tags via three pluggable grammar backends:

- **XGrammar** (default) — fastest; precomputes token masks per grammar state.
- **Outlines** — broader feature surface, slightly slower.
- **llguidance** — Microsoft's backend; tight integration with chat-template constraints.

Selected via `--grammar-backend xgrammar|outlines|llguidance`.

### 6.1 How constraints reach the runtime

Two API surfaces, same backend:

```python
# OpenAI-compatible — JSON schema as response_format
client.chat.completions.create(
    model="...",
    messages=[...],
    response_format={"type": "json_schema",
                     "json_schema": {"schema": my_schema}},
)

# Native /generate — full constraint surface
requests.post("http://server/generate", json={
    "text": prompt,
    "sampling_params": {
        "json_schema": my_schema,     # or:
        "regex": r"\d{3}-\d{4}",      # or:
        "ebnf": "<grammar>",          # or:
        "structural_tag": {...},      # for tool-call shaped outputs
    },
})
```

### 6.2 Interaction with prefix cache

The grammar state is per-request, but the *grammar definition* is often shared (e.g. all tool-call requests use the same function-schema). XGrammar precomputes a token-mask table per grammar; this table is reused across requests with the same schema. Combined with RadixAttention's prefix cache, an agent that re-issues identical tool-call requests pays grammar-compile cost once, prefill cost on the suffix only, and decode cost normally — total speedup vs a stateless implementation is typically 3–6× for tool-heavy agents.

**Gotcha.** A complex grammar can slow decode by 20–40 % due to the mask-application overhead (see [[sglang-structured-output]] limitations). When TPOT matters more than schema-strictness, prefer post-hoc validation over grammar-constrained generation.

---

## 7. Production knobs — the cheat sheet

```bash
# Standard high-throughput chat server, RadixAttention on, fp8 KV cache, FlashInfer kernels
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3-8B-Instruct \
    --tp 1 \
    --port 30000 \
    --schedule-policy lpm \
    --chunked-prefill-size 8192 \
    --max-running-requests 256 \
    --kv-cache-dtype fp8_e4m3 \
    --attention-backend flashinfer \
    --enable-metrics

# Agent / shared-prefix server — bigger RadixCache, longer prefixes
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3-8B-Instruct \
    --schedule-policy lpm \
    --max-prefill-tokens 65536 \
    --chunked-prefill-size 16384 \
    --max-total-tokens 524288

# HiCache with L2 (host) + L3 (Mooncake) tiers
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3-70B-Instruct \
    --tp 8 \
    --enable-hicache \
    --hicache-storage-backend mooncake \
    --hicache-write-policy write_through_selective \
    --hicache-ratio 2.0          # L2 size = 2× GPU KV capacity

# Structured-output server with XGrammar
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3-8B-Instruct \
    --grammar-backend xgrammar

# Offline engine (no HTTP, in-process)
from sglang import Engine
engine = Engine(model_path="meta-llama/Llama-3-8B-Instruct",
                schedule_policy="lpm")
out = engine.generate(prompt, sampling_params={"max_new_tokens": 256})
```

---

## 8. SGLang vs vLLM — when each wins

| Workload | Winner | Reason |
|----------|--------|--------|
| Arbitrary user chat, no shared prefix | vLLM | Marginal; SGLang has slightly more scheduler overhead from radix tree walks |
| Multi-turn chat with long system prompt | SGLang | Higher prefix-cache hit rate; faster prefill |
| Few-shot prompting (same exemplars, varying query) | SGLang | RadixAttention picks up shared shots automatically |
| Agent loops, ReAct, tree-of-thought | SGLang | `fork()` makes branching explicit; cache pinned across branches |
| RAG over a small corpus | SGLang | Documents become shared prefixes |
| Constrained decoding heavy (JSON-mode tool calls) | SGLang | XGrammar is faster than vLLM's xgrammar bindings, and integrates better with prefix cache |
| MoE inference (Mixtral, DeepSeek) | SGLang | Better expert-parallel scheduling integration |
| Bare-metal max-throughput on H100, single shape | vLLM | Mature CUDA-Graph capture pipeline; smaller variance |
| Production cluster with cross-node KV sharing | SGLang | HiCache is the only OSS multi-tier story; vLLM has the KV offloading connector but no distributed L3 |

The forward look: vLLM has been catching up on prefix-cache features (vLLM 1.x APC is good), and SGLang has been catching up on raw single-shape throughput. The structural advantage SGLang retains is the *frontend / runtime co-design* — vLLM cannot match the `fork()`-as-cache-hint pattern without a frontend of its own.

---

## Common pitfalls

- **Running `--schedule-policy fcfs` on shared-prefix workloads.** Defeats RadixAttention's value. The default `lpm` is correct for ~all production workloads.
- **`--disable-radix-cache` for debugging and forgetting to turn it back on.** SGLang sans RadixAttention is just slower vLLM. Watch the cache-hit-rate metric to confirm RadixAttention is active.
- **Underestimating `--max-running-requests` impact on radix tree size.** More concurrent requests = more ref-counted nodes = less evictable KV = more cache pressure. Tune jointly with `--max-total-tokens`.
- **Mixing the frontend DSL with explicit HTTP calls.** The DSL caches across `fork()` branches; raw HTTP calls don't share state. Pick one per service.
- **Grammar backend choice without measurement.** XGrammar is default and usually fastest; only switch to Outlines / llguidance if you hit a specific grammar-syntax compatibility issue.
- **HiCache L2 hit rate < 10 %.** Means working set fits in GPU L1 — L2 overhead is pure cost. Disable HiCache unless L2 hit rate justifies the bandwidth budget.
- **Ignoring `cached_tokens` in the response.** SGLang returns `cached_tokens` in `usage`; if it's 0 across many requests, your workload has no overlap and SGLang's main feature isn't helping.

---

## Connections and what's next

- **Back: [[sglang-radixattention]] / ch-07** — the algorithm-level treatment of prefix caching + RadixAttention vs vLLM APC. This chapter is the *production-runtime* counterpart.
- **Back: [[pagedattention]] / ch-06** — RadixAttention re-uses paged-block storage; the radix tree is metadata on top of vLLM-style block allocator.
- **Back: [[continuous-batching]] / ch-04** + **[[sarathi-serve]] / ch-05** — the scheduler this chapter discusses is the SGLang implementation of chunked-prefill continuous batching.
- **Back: [[mooncake]] / ch-09** — HiCache's L3 backend; the distributed KV-cache architecture that motivated the 3-tier design.
- **Lateral: ch-16 (vLLM)** — direct architectural comparison; same problem, different cache structure (hash vs radix).
- **Forward: ch-19 (benchmarks)** — `python -m sglang.bench_serving` is the canonical way to measure all of the above; ShareGPT + request-rate sweep with TTFT/TPOT/goodput.
- **Forward: ch-21 (lab)** — SGLang vs vLLM head-to-head on Llama-3-8B with the same ShareGPT trace; the canonical comparison.

## Further reading

- [[sglang]] — framework-level overview and serving APIs.
- [[sglang-radixattention]] — the data structure, with paper + source links.
- [[sglang-scheduler]] — scheduler internals and policy knobs.
- [[sglang-hicache]] — three-tier hierarchical KV cache design.
- [[sglang-structured-output]] — grammar backends and constrained decoding.
- [[sglang-project]] — project context and benchmark/profile pointers.
- SGLang paper: https://arxiv.org/abs/2312.07104 — RadixAttention + frontend co-design.

## Companion visualization

**[figures/sglang-architecture.html](figures/sglang-architecture.html)** — interactive walk-through of the four-layer SGLang stack: frontend program → request graph → cache-aware scheduler → backend with RadixAttention + HiCache tiers. Click a node to see what it owns; toggle `lpm`/`fcfs` to see admission-order change; slide the workload-overlap parameter to watch hit-rate per tier respond.
