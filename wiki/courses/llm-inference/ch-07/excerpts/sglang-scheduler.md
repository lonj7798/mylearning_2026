---
chapter: ch-07
course: llm-inference
phase: read
excerpt_of: "SGLang scheduler — python/sglang/srt/managers/scheduler.py"
source_url: https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/scheduler.py
created_at: "2026-05-21"
---

# Excerpt: SGLang scheduler — radix-cache-aware admission

**Authors:** SGLang project
**Year:** 2023-present
**URL:** https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/scheduler.py
**Raw-data source:** [[raw-data/sglang-scheduler]]

---

## Why this module matters for prefix caching

The radix cache ([[excerpts/sglang-radixattention]]) is a data structure. The scheduler is what decides *when* to query it, *which* matches to honor, and *how to order* requests by expected cache hit. Most of RadixAttention's headline numbers come from the data structure + the scheduler's cache-aware policy *together*.

This excerpt focuses on what the scheduler adds on top of the radix tree.

---

## The main loop (paraphrase)

```python
# python/sglang/srt/managers/scheduler.py — simplified to the radix-cache path

class Scheduler:
    def event_loop(self):
        while True:
            self.recv_new_requests()              # pull from HTTP/native API queue
            batch = self.get_next_batch()
            if batch is None:
                continue
            output = self.tp_worker.run_batch(batch)
            self.process_output(batch, output)

    def get_next_batch(self):
        # 1. Decode-only batch path (high-frequency case).
        if self.running and not self.waiting:
            return self.build_decode_batch(self.running)

        # 2. Mixed prefill+decode (with radix cache).
        new_admissions = self.get_new_prefills(token_budget=self.max_prefill_tokens)
        return self.build_mixed_batch(new_admissions, self.running)

    def get_new_prefills(self, token_budget):
        admitted = []
        # Step 1: query radix cache for every waiting request; sort by hit rate.
        for req in self.waiting:
            matched, matched_len = self.tree_cache.match_prefix(req.input_ids)
            req.cached_tokens = matched_len
            req.matched_blocks = matched
        sorted_waiting = self.policy.sort(self.waiting)   # cache-aware ordering

        # Step 2: admit greedily under prefill token budget + KV memory.
        for req in sorted_waiting:
            prefill_left = req.input_len - req.cached_tokens
            take = min(prefill_left, token_budget, self.chunked_prefill_size)
            if not self.req_to_token_pool.has_room(take):
                break
            self.tree_cache.inc_lock_ref(req.matched_blocks)  # pin matched prefix
            admitted.append((req, take))
            token_budget -= take
        return admitted
```

Three SGLang-specific design points:

- **Radix match happens in admission, not after.** The matched length determines what the scheduler treats as "prefill work" — without this hook, the cache would be a memory feature but not a latency feature.
- **`policy.sort` is cache-aware.** Implementations (in `policy_scheduler.py`) order waiting requests by expected hit ratio; high-hit requests get admitted earlier because they free per-step token budget faster.
- **`chunked_prefill_size` is the per-request cap** — analogous to vLLM's `max_chunk_size`. The scheduler can't dedicate the full token budget to a single huge prefill.

---

## The scheduling-policy options

SGLang exposes `--schedule-policy` with several modes:

| Policy | Behavior |
|---|---|
| `lpm` (longest prefix match) | Sort waiting by absolute matched token count. Maximizes cache reuse. |
| `random` | Random order (debugging baseline). |
| `fcfs` | First-come, first-served. Ignores cache. |
| `dfs-weight` | DFS over the radix tree weighted by waiting requests; groups branch siblings together. |

`lpm` is the default for chat/RAG-style workloads. `dfs-weight` is preferred for tree-search workloads where many requests share branches in the radix tree (the SGLang paper §5 prefers DFS-weight for tree-of-thought).

---

## How chunked prefill composes with prefix cache

Chunked prefill (ch-05) and prefix caching compose multiplicatively. A request with prompt length 1000 and matched-prefix 900:

- Without cache, without chunked: one 1000-token prefill step.
- With cache, without chunked: one 100-token prefill step.
- With cache, with chunked (`chunk_size = 512`): one 100-token chunk (fits in one step). Same as no-chunk in this case.
- Without cache, with chunked: two chunks (512 + 488), interleaved with decodes.

The cache turns "long prompts" into "short suffixes" before chunked prefill even has to act. SGLang's scheduler runs both, so a 10k-token prompt with a 9.5k-token cached prefix becomes one chunk (500 tokens) — single-step admission, no decode stalls.

---

## Tunable knobs

| Server arg | Default | Effect |
|---|---|---|
| `--schedule-policy lpm` | `lpm` | Ordering of waiting requests |
| `--schedule-conservativeness 1.0` | 1.0 | Backoff factor on KV admission; lower → more aggressive |
| `--max-running-requests` | auto | Cap on parallel-decode count |
| `--max-total-tokens` | auto from GPU | Total KV pool size in tokens |
| `--chunked-prefill-size` | 8192 | Per-step prefill chunk cap |
| `--max-prefill-tokens` | 16384 | Per-step prefill token budget |
| `--disable-radix-cache` | (off) | Turn the entire cache off; only for benchmarking |
| `--mem-fraction-static` | 0.88 | GPU memory fraction for static allocation (model + cache) |

The two that production operators most often touch: `--mem-fraction-static` (sets the pool size) and `--schedule-policy` (rare; default `lpm` is right for most workloads).

---

## Interaction with HiCache

When HiCache is enabled, the scheduler's `match_prefix` query becomes:

1. Local radix-tree match (L1 GPU) — returns matched blocks immediately resident.
2. If unmatched suffix could match an L2/L3 entry, schedule a prefetch and *also* schedule the suffix prefill compute. Whichever completes first wins.
3. Once L2/L3 blocks land in L1, the radix tree updates and subsequent admissions see the full match.

This means a "cold" workload that starts with no cache hits warms up quickly — every completed prefill leaves blocks both in L1 and (asynchronously) in L2/L3, so subsequent matching requests within the same minute can hit the warm cache.

---

## Connections

- [[excerpts/sglang-radixattention]] — the cache data structure this scheduler uses.
- [[excerpts/sglang-hicache]] — the multi-tier extension.
- [[excerpts/sglang]] — overall SGLang architecture.
- [[ch-07]] — parent synthesis.
- Forward to [[ch-17]] — full SGLang internals.
