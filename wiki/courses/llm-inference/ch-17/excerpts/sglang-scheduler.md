---
chapter: ch-17
course: llm-inference
phase: read
excerpt_of: "SGLang scheduler — python/sglang/srt/managers/scheduler.py"
source_url: https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/scheduler.py
created_at: "2026-05-21"
---

# Excerpt: SGLang Scheduler — cache-aware admission + chunked prefill

**Authors:** SGLang project
**Year:** 2023–present
**URLs:** https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/scheduler.py / https://docs.sglang.ai/advanced_features/server_arguments.html
**Raw-data source:** [[raw-data/sglang-scheduler]]

---

## The event loop, in plain English

```python
# Pseudocode condensing managers/scheduler.py
def run_step():
    # 1. Drain finished requests from the running batch.
    for req in running:
        if req.finished():
            radix_cache.insert(req.tokens, req.kv_blocks)
            radix_cache.dec_lock_ref(req.cached_node)
            running.remove(req)

    # 2. Score waiting requests by prefix-cache hit length (under lpm).
    if schedule_policy == "lpm":
        scored = []
        for req in waiting:
            matched_blocks, node = radix_cache.match_prefix(req.tokens)
            score = len(matched_blocks) / max(len(req.tokens), 1)
            scored.append((score, req, node))
        scored.sort(key=lambda x: -x[0])      # cache-friendly first
    else:  # fcfs
        scored = [(0, req, radix_cache.match_prefix(req.tokens)[1])
                  for req in waiting]

    # 3. Admit while budgets allow.
    token_budget = max_num_batched_tokens
    for _, req, node in scored:
        new_tokens = len(req.tokens) - cache_hit_len(req, node)
        if (new_tokens <= token_budget
            and free_kv_blocks() >= ceil(new_tokens / block_size)
            and len(running) < max_running_requests):
            radix_cache.inc_lock_ref(node)
            req.cached_node = node
            allocate_kv_blocks(req, new_tokens)
            running.append(req)
            waiting.remove(req)
            token_budget -= new_tokens

    # 4. Build mixed prefill+decode batch (chunked prefill).
    batch = build_mixed_batch(running, max_chunk=chunked_prefill_size)

    # 5. Forward pass.
    model_worker.forward(batch)
```

---

## The flags that matter

| Flag | Default | Effect |
|------|--------:|--------|
| `--schedule-policy` | `lpm` | Cache-aware (`lpm`) vs arrival-order (`fcfs`) |
| `--schedule-conservativeness` | `1.0` | KV-budget safety margin; >1 admits fewer requests |
| `--max-running-requests` | model-dep | Hard cap on concurrent requests |
| `--max-total-tokens` | derived | Total KV-slot budget (~`gpu_memory_utilization × KV-capacity`) |
| `--chunked-prefill-size` | `8192` | Max tokens of prefill mixed into one forward pass |
| `--max-prefill-tokens` | `16384` | Hard cap on per-request prefill chunk |
| `--disable-radix-cache` | off | Kill switch — disables RadixAttention |

---

## Why `lpm` beats `fcfs` on shared-prefix workloads

Under FCFS, a workload of 32 in-flight requests against a 5k-token document might process them in arrival order. Request 1 takes a full 5k+question prefill; the cache then holds the document. Requests 2–32 hit the cache and only prefill the question, but they queued behind request 1.

Under LPM, the *first* admitted request is whichever one has the longest cached prefix — but on a cold start this is symmetric. The win comes from steady-state: when 32 requests arrive in a burst, LPM keeps admitting cache-friendly ones first, so the cache stays hot and never evicts the document. Total time-to-completion is lower for the whole batch because total prefill work is lower.

The opposite tradeoff: LPM can starve a cache-cold request behind a stream of cache-warm ones. SGLang mitigates this with an age boost (requests waiting too long get score boost) — see `schedule_conservativeness` and the scheduler source.

---

## Coordination with other features

- **Chunked prefill (ch-05 / Sarathi-Serve).** Scheduler bounds each request's prefill chunk to `chunked_prefill_size`; mixed batches contain decode tokens + prefill chunks. Decode latency stays low even when a huge prefill is in flight.
- **Grammar backends (XGrammar/Outlines).** Grammar state is per-request; scheduler attaches grammar context at admission. Mask application happens in the model worker.
- **LoRA.** Requests carry a `lora_adapter` field; scheduler batches requests by adapter to avoid swap thrash.
- **Speculative decoding.** Scheduler treats drafted+verified tokens as decode work; chunk size accounts for the K-fold multiplier.
- **PD-disaggregation.** Scheduler routes prefill requests to prefill workers, decode requests to decode workers, with KV transfer between pools (see ch-09).

---

## Connections

- [[excerpts/sglang-architecture]] — where this scheduler sits.
- [[excerpts/sglang-radixattention]] — the cache it queries.
- [[excerpts/sarathi-serve]] (ch-05) — chunked-prefill foundation.
- [[ch-04]] — continuous batching theory.
