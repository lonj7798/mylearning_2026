<!-- chapter: ch-07
     track: kv-cache
     title: Prefix Caching + RadixAttention (SGLang)
     sources: [[sglang-radixattention]], [[sglang]], [[sglang-scheduler]], [[sglang-hicache]], [[vllm-kv-cache-manager]]
     back: [[pagedattention]] (ch-06)
     forward: [[h2o]] (ch-08), [[sglang-scheduler]] (ch-17)
     figures: figures/radixattention-tree.html
-->

# Chapter 7 — Prefix Caching + RadixAttention (SGLang)

> **Core insight.** In real serving workloads, a large fraction of every prompt is *recycled text*: a fixed system prompt, a few-shot exemplar block, a tool schema, an agent's loop preamble, or the previous turns of a multi-turn chat. Recomputing the KV cache for that text every time wastes the most expensive resource in the system (prefill compute) for content that is bit-identical to something the GPU just computed five seconds ago. Prefix caching makes that recomputation a no-op: hash the prompt blocks, look the hashes up in a cache, and reuse the existing physical KV blocks for the matching prefix. vLLM's APC does this with a flat hash table keyed by block hashes; SGLang's **RadixAttention** does it with a radix tree keyed by token sequences, which makes prefix matching automatic across siblings, branches, and partial overlaps.
>
> **Guideline.** Enable prefix caching by default for any production deployment with non-trivial prompt structure: chat applications with system prompts, agent loops, RAG with shared document context, few-shot pipelines. In vLLM use `--enable-prefix-caching` (default on in V1). In SGLang use the radix cache (default on; disable with `--disable-radix-cache`). Expect 30–80 % TTFT reduction for chat/agent workloads and effectively zero quality regression. For long shared contexts that exceed GPU HBM, layer **SGLang HiCache** (GPU → CPU → distributed) on top.

---

## Why this chapter exists

[[pagedattention]] (ch-06) gave us block-granular KV allocation. Sharing KV blocks across **siblings of one request** (parallel sampling, beam search) was the immediate win. But the same machinery suggests a more aggressive use: share blocks across **independent requests** that happen to begin with the same tokens.

The opportunity is enormous. Consider a typical chat application:

- Every request starts with a ~500-token system prompt that the operator never changes.
- A typical multi-turn conversation appends ~50 tokens of user turn at a time.
- Turn N of a conversation shares the first `(500 + sum of turns 1..N-1)` tokens with turn N-1.

If you recompute prefill from scratch every turn, you are paying for the *same* attention computation over and over. The system prompt's K and V tensors at layer 23, head 4, position 312 are bit-identical across every request — and yet most serving systems pre-RadixAttention recomputed them.

Prefix caching turns this insight into a runtime primitive: at admission time, scan the prompt against the existing cache, find the longest matching prefix, *reuse* its blocks, and only prefill the remaining suffix. For a 1000-token prompt with a 900-token cached prefix, you save 90 % of prefill compute and the request's TTFT drops nearly to zero.

Two systems realize this with different data structures:

- **vLLM's Automatic Prefix Cache (APC)** — flat hash table keyed by block-content hashes; LRU eviction; the path we already saw in [[vllm-kv-cache-manager]] (ch-06).
- **SGLang's RadixAttention** — radix tree keyed by token sequences; matches the longest common prefix automatically across any branching pattern; per-node reference counts for safe eviction.

Both work. RadixAttention is the better fit for branchy workloads (agents that fork into multiple sub-queries from the same context); APC is the simpler model and is plenty for most chat/RAG workloads. This chapter walks both.

---

## 1. The prefix-cache use case in numbers

To anchor the discussion, here is what shared prefixes look like in practice:

| Workload | Typical cached fraction | Mechanism |
|---|---|---|
| ChatGPT-style chat with fixed system prompt | 30–60 % per turn-1 request | Shared system prompt only |
| Multi-turn chat, turn N≥2 | 80–95 % per request | All previous turns + system prompt |
| RAG with shared documents | 50–90 % when the same doc appears multiple times in the trace | Shared document chunk |
| Few-shot pipeline (e.g., classifier with 20 exemplars) | 90–98 % | Exemplars + instructions are 1k+ tokens of fixed prefix |
| Agent loop (ReAct-style) over same task | 80–95 % across iterations | Persistent scratchpad + tool definitions |
| Code completion against same file | 70–90 % | Cursor moves but file context persists |

These numbers (from the SGLang paper §6, the vLLM APC blog post, and replications in the [[sglang-hicache]] design doc) explain why prefix caching is now table stakes: a 50–90 % prefill reduction is too large to leave on the table.

For workloads where prefix caching does *not* help — single-turn, no system prompt, arbitrary user-provided long prompts — the cost is small: a hash computation per block during admission. Disable only if profiling shows the hashing itself is contended.

---

## 2. vLLM's APC: hash-based block matching

vLLM's prefix cache reuses the block-allocation machinery from [[vllm-kv-cache-manager]] (ch-06). The extension is a single hash table from block-hash to physical block ID, plus a chain-hash scheme that makes prefix matching token-exact.

**Block hashing.** Each block holds 16 tokens. Its hash incorporates *both* its own 16 tokens *and* its parent block's hash:

```python
# vllm/v1/core/kv_cache_utils.py — paraphrased
def hash_block(parent_hash: BlockHash, token_ids: tuple[int, ...]) -> BlockHash:
    return blake3((parent_hash, token_ids).serialize()).digest()
```

The chain means block `k`'s hash depends on the entire token prefix up to and including block `k`. Two requests share block `k` iff their entire prompts agree on tokens 0..16(k+1)-1 — that is, exact-token-prefix match at 16-token granularity.

**Lookup path** (`KVCacheManager.get_computed_blocks`, ch-06):

```
for each prompt block i = 0, 1, 2, ...:
    h_i = hash_block(h_{i-1}, prompt_tokens[16i : 16(i+1)])
    if h_i in cache_hash_to_block:
        cached_blocks.append(cache_hash_to_block[h_i])
        cache_hash_to_block[h_i].ref_cnt += 1     # pin against eviction
    else:
        break                                      # contiguous prefix only
return cached_blocks, len(cached_blocks) * 16
```

The integer return value `num_cached_tokens` flows straight into the scheduler's `request.num_computed_tokens` and bypasses the corresponding prefill work entirely. Chunked prefill (ch-05) only needs to schedule the *suffix*.

**LRU eviction.** When the free pool empties, the cache walks unreferenced cached blocks in LRU order and frees them back to the pool. A reference count > 0 means the block is in active use by a running request; those blocks are never evicted.

**Hit-rate characteristics:**

- 16-token block granularity means partial-prefix matches round down. A prompt sharing 947 tokens with a cached prompt registers as `947 // 16 = 59` cached blocks (944 tokens); the last 3 tokens are recomputed.
- Hash collisions are statistically negligible (blake3, 256-bit space).
- Cache is content-addressed, so reordering tools that produce the same tokenization still hit.

This is the cleanest prefix cache to reason about, and it covers the dominant case (single-prefix matching). What it does *not* handle elegantly:

- **Branching contexts.** If 10 requests share the first 500 tokens, then diverge into 10 different middle sections, then share another 200 tokens at the end — APC catches the first 500 (shared prefix) but cannot reuse the trailing 200.
- **Sub-prefix sharing across unrelated requests.** APC only matches from token 0. Two requests sharing tokens 200–700 but differing in tokens 0–199 register as zero shared blocks.

For branchy workloads (agents, multi-path search, tree-of-thought), the radix-tree variant is strictly more general.

---

## 3. SGLang's RadixAttention: tree-keyed prefix sharing

[[sglang-radixattention]] (Zheng et al. 2023) replaces the hash table with a **radix tree** keyed by token sequences. Each node holds:

- A token-sequence span (the edge label from its parent).
- A list of KV blocks holding the K/V for that span.
- A reference count (>0 means in use by a running request).
- Child pointers keyed by the first token of each child's span.

```
root
├── [system_prompt_500_tokens]  (cached blocks B0..B31)
│   ├── [user_turn_A_50_tokens]  (cached blocks B32..B34)
│   │   └── [assistant_reply_300]   (cached blocks B35..B53)
│   └── [user_turn_B_80_tokens]  (cached blocks B36..B40)
└── [different_system_prompt_300]  (cached blocks B100..B118)
```

When a new request arrives, the radix tree's `match_prefix(tokens)` walks from root, following the child whose span continues to match the request's tokens. When the request's tokens diverge mid-span, the tree **splits** the node at the divergence point, giving the request its own child for the suffix and leaving the matched prefix shared.

This is more powerful than flat hashing in two ways:

- **Automatic longest-prefix match across any topology.** Two requests with the same first 500 tokens share those 500 — whether they then diverge or not.
- **Mid-span splits give token-level granularity** (not 16-token-block granularity). A request matching 947 tokens shares 947, not 944.

The trade is a more complex data structure and more per-step bookkeeping (matching is O(L) per request, but with a much smaller constant than rebuilding KV from scratch).

**Reference counting.** Each radix node has a `ref_cnt` field tracking how many in-flight requests are using that span. Eviction picks nodes with `ref_cnt == 0` (LRU among them). Active requests effectively "lock" their entire matched-prefix chain against eviction until completion.

**Mid-span split** (the operation that gives token-level granularity):

```text
Before: root → [tokens 0..499] (one node, blocks B0..B31)
        New request matches 0..299 of this node, diverges at 300.

After:  root → [tokens 0..299]       (new node, blocks B0..B18)
                ├── [tokens 300..499]  (old continuation, blocks B19..B31)
                └── [new request suffix]  (new node, fresh blocks)
```

The split is a metadata-only operation; physical KV blocks don't move.

---

## 4. The SGLang scheduler integrates radix matching

[[sglang-scheduler]] (the `python/sglang/srt/managers/scheduler.py` runtime) integrates RadixAttention into admission:

```python
# Sketch — actual code is in sglang/srt/managers/scheduler.py

def admit_request(req):
    matched_blocks, matched_len = radix_cache.match_prefix(req.input_ids)
    req.cached_tokens   = matched_len
    req.prefill_to_do   = req.input_len - matched_len
    radix_cache.lock(matched_blocks)               # pin against eviction
    waiting.append(req)
```

A request that matches a 900-token prefix from a 1000-token prompt enters with `prefill_to_do = 100`. The scheduler's per-step token-budget logic (ch-05's mixed-batch scheduling) treats that 100 as the chunk-prefill workload.

**Cache-aware scheduling.** SGLang's `schedule_policy` flag also lets the scheduler order admissions by *expected cache hit rate*. A waiting request that will hit a 95 % cached prefix is admitted earlier than one with no cache match, because admitting the high-hit one frees scheduling tokens for actual new work. Production deployments report ~10–20 % TTFT improvement from cache-aware ordering on top of the cache itself.

This is one of the design choices that justifies the radix tree over a flat hash: the scheduler can cheaply estimate hit length per waiting request (one tree walk) before committing to an admission order.

---

## 5. Hit rates in practice

Combining published measurements ([[sglang-radixattention]] §6, the SGLang blog, the vLLM 0.6 APC release notes, the [[sglang-hicache]] design doc, and a handful of production deployment writeups):

| Workload | Cache hit rate | TTFT reduction |
|---|---|---|
| ChatGPT-style with fixed system prompt | 50–70 % | 40–60 % |
| Multi-turn chat (averaged over turns) | 70–90 % | 60–85 % |
| Few-shot classifier (1k-token exemplars + per-request 100 tokens) | 90–95 % | 85–93 % |
| RAG with frequent document re-use | 60–85 % | 50–75 % |
| Agent loop with tool-call scratchpad | 80–95 % | 75–90 % |
| Arbitrary user prompts, no shared structure | 5–20 % | 0–15 % |

A few caveats:

- "Hit rate" here is the fraction of *prompt tokens* served from cache, not the fraction of requests with any cache hit. The latter is usually much higher.
- TTFT reduction is sub-linear because the suffix still needs prefill, and TTFT also includes scheduling latency.
- Hit rates depend on **tokenization stability**. If the operator changes the chat template or system prompt, every cached prefix invalidates on the next request.

The headline implication: for **any production workload with structured prompts**, prefix caching is the single highest-ROI optimization after paged KV itself.

---

## 6. SGLang HiCache: when the KV pool isn't enough

[[sglang-hicache]] extends RadixAttention from a GPU-only cache to a **three-tier hierarchy**:

| Tier | Storage | Latency | Capacity |
|---|---|---|---|
| L1 | GPU HBM | sub-microsecond load | tens of GB |
| L2 | CPU memory (DDR5) | ~10 µs load (PCIe Gen5) | hundreds of GB to TBs |
| L3 | Distributed storage (Mooncake, 3FS, NIXL, AIBrix KVCache) | sub-ms via RDMA | unbounded |

The metadata structure becomes a **HiRadixTree**: same radix tree as RadixAttention, but each node tags its KV with the tier currently holding it. On a request:

1. Walk the HiRadixTree to find the longest matching prefix.
2. For each matched span, the tier metadata says where the KV lives.
3. If all matched spans are L1-resident, no transfer needed.
4. If a span is L2 or L3, prefetch it to L1 in the background while the scheduler computes the suffix prefill.

**Why this matters for long-context.** A 100k-token document used as RAG context is too big to keep in HBM permanently — it would crowd out active request KV. But recomputing 100k tokens of prefill every time a user asks a different question about the same document costs ~10 seconds of GPU compute. HiCache demotes the cached KV to host memory (L2) when GPU pressure is high, then promotes it back when the document is queried again. The cost is a few hundred ms of PCIe transfer vs ~10 seconds of recompute.

**The capacity hierarchy maps directly to the workload hierarchy:**

- L1 holds **hot** prefixes: live system prompts, in-flight conversations.
- L2 holds **warm** prefixes: long documents the fleet has seen recently, agent state across paused sessions.
- L3 holds **cold but valuable** prefixes shared across the cluster: corporate knowledge bases, large few-shot exemplar sets, base prompts for fine-tuned families.

Mooncake (ch-09) is the production L3 backend most-often paired with HiCache — a KV-cache-centric serving architecture with RDMA-backed cross-node KV transfer.

---

## 7. Comparing vLLM APC and SGLang RadixAttention

A side-by-side for picking a default:

| Dimension | vLLM APC | SGLang RadixAttention |
|---|---|---|
| Data structure | Flat hash table | Radix tree |
| Granularity | 16-token blocks | 1-token splits within a node |
| Prefix-matching scope | Token-0 forward only | Any common prefix in the tree |
| Branching workloads | Catches only the deepest single chain | Catches all common ancestors |
| Lookup cost | O(L/16) hash computations | O(L) tree walk (small constants) |
| Eviction | LRU on free pool refill | LRU among `ref_cnt==0` nodes |
| Multi-tier (CPU/distributed) | KVConnector for CPU offload | HiCache (L1/L2/L3 first-class) |
| Default in framework | On since vLLM V1 | On unless `--disable-radix-cache` |

**Recommendation for production:**

- *Chat / RAG / single-turn workloads* — either works; pick by your framework preference. APC is simpler to operate.
- *Agent loops, tree-of-thought, multi-branch reasoning* — RadixAttention wins meaningfully because branches share more than one prefix.
- *Long-context (>30k tokens) with document reuse* — pair RadixAttention with HiCache; the L2 tier turns an HBM-bound problem into a PCIe-bandwidth-bound problem, which is easier.

For workloads where prompts share *content* but not *exact tokens* (e.g., the same paragraph rephrased), no caching strategy will help. The cache demands bit-exact tokens.

---

## 8. The operational checklist

Things to verify before declaring prefix caching working:

1. **Enable the cache.** vLLM: `--enable-prefix-caching`. SGLang: default on; don't pass `--disable-radix-cache`.
2. **Verify tokenization stability.** A chat template change invalidates every cached prefix. Track `vllm:prefix_cache_hit_rate` (vLLM) or `sglang:cache_hit_rate` (SGLang); drops to zero correlate with template changes.
3. **Size the cache.** Default behavior reserves all post-weights HBM as one pool shared between cached blocks and active-request blocks. If hit rate is plateaued and free pool is always near-empty, increase `gpu_memory_utilization`. If preemption rate is high, decrease it.
4. **Match block size to prompt structure.** vLLM's 16-token blocks are usually fine; SGLang's radix tree is granularity-free.
5. **For HiCache:** size L2 (CPU memory) to ~10× L1 capacity; L3 only if cluster-shared workloads dominate.

Metrics to watch in production:

- `prefix_cache_hit_rate` — should match the workload expectation; drop signals a content shift.
- `prefix_cache_evictions_per_sec` — high values mean the cache is undersized.
- `num_preemptions_total` — high values mean cache is starving live requests; lower `gpu_memory_utilization` or reserve more free-pool headroom.

---

## Connections and what's next

- **Back to [[pagedattention]] (ch-06)** — prefix caching is the cross-request generalization of intra-request COW. Same block pool, same kernel, smarter accounting.
- **Back to [[vllm-kv-cache-manager]] (ch-06)** — `get_computed_blocks` is the integration point for APC; this chapter reused that machinery.
- **Forward to [[h2o]] / [[snapkv]] / [[quest-kv]] (ch-08)** — KV compression and sparsity work on the *active* cache; prefix caching works on the *cold-shared* cache. They compose.
- **Forward to [[sglang-scheduler]] (ch-17)** — full SGLang internals tour: cache-aware admission, hyperparameter knobs, end-to-end flow.
- **Forward to [[mooncake]] (ch-09)** — the canonical L3 distributed KV-cache backend HiCache plugs into.
- **Forward to [[vllm-scheduler]] (ch-16)** — `num_computed_tokens` is the scheduler hook this chapter relied on.

## Further reading

- [[sglang-radixattention]] — the RadixAttention paper and `radix_cache.py` source.
- [[sglang]] — overall SGLang architecture and APIs.
- [[sglang-scheduler]] — scheduler integration; cache-aware policy.
- [[sglang-hicache]] — multi-tier extension.
- [[vllm-kv-cache-manager]] — vLLM APC implementation.
- vLLM blog: "Automatic Prefix Caching in vLLM" (v0.6 release notes).

## Companion visualization

**[figures/radixattention-tree.html](figures/radixattention-tree.html)** — interactive radix tree: drag requests onto a shared root and watch nodes split as prompts diverge. Hover any node to see token range, block IDs, and reference count. Toggle "evict all `ref_cnt==0`" to see LRU pruning in action.
