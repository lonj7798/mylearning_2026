---
chapter: ch-12
course: llm-inference
phase: read
excerpt_of: "SGLang — RadixAttention-aware CUDA graph scheduling"
source_url: https://docs.sglang.ai/
created_at: "2026-05-21"
---

# Excerpt: SGLang — CUDA graphs with scheduler-side bucket stability

**Authors:** SGLang project
**Year:** 2023–present
**URL:** https://docs.sglang.ai/
**Raw-data source:** [[raw-data/sglang]]

---

## What SGLang adds beyond vLLM's piecewise graphs

SGLang captures CUDA graphs the same way vLLM does (one per batch-size bucket). The interesting difference is in **how the scheduler picks the next batch composition** to maximize graph hits.

### The vLLM problem: bucket thrashing

vLLM's scheduler picks the running batch greedily based on (priority, KV availability). The resulting batch size fluctuates step-by-step: 12 → 17 → 15 → 11 → 19. Each fluctuation crosses a bucket boundary and forces a different captured graph — but more importantly, the *padding overhead* changes with each transition.

Empirically, on a chat-heavy workload, vLLM hits each bucket roughly uniformly, with ~25% average padding waste.

### The SGLang fix: scheduler-side bucket affinity

SGLang's scheduler (`python/sglang/srt/managers/scheduler.py`) tracks the current bucket and tries to keep the next batch *in the same bucket*:

```text
For each scheduler iteration:
    current_running = N_running
    current_bucket  = next_pow2(current_running)
    pending = waiting_queue
    
    target_size = current_bucket
    # If finishes will drop us below current_bucket - 1, OK to keep bucket
    # If we have room to add and it stays in current_bucket, prefer to add
    # Only cross bucket if forced (KV pressure, request finished cluster)
    
    pick_requests_to_match(target_size, pending)
```

This stabilizes the bucket across iterations.

---

## RadixAttention as the enabler

The reason SGLang can do this without sacrificing fairness: **RadixAttention** ([[sglang-radixattention]], ch-07) means most requests share prefixes. Scheduling decisions can prioritize prefix-cache-hot requests, which often cluster naturally into similar batch sizes.

The scheduler's bucket-stability heuristic stacks with prefix-hit-rate maximization. Both push toward similar batch compositions across iterations.

---

## Reported numbers

| Setting | TPOT (Llama-3-8B, A100) |
|---|---|
| vLLM with graphs | 18 ms (25% padding waste avg) |
| SGLang with graphs + bucket stability | 17 ms (15% padding waste avg) |
| **Improvement** | **~5–10% TPOT** |

Small but real — and entirely from scheduler decisions, no kernel changes.

---

## Other SGLang runtime wins

Beyond graph stability:

- **Zero-overhead scheduling**: the scheduler runs on a separate Python thread that overlaps with GPU compute. The CPU never blocks waiting for the GPU.
- **Fast Python-side tokenizer**: pre-loads tokenizer in a worker thread.
- **Continuous batching with prefix-aware admission**: prefer requests whose prefix is already cached.

---

## Connections

- [[ch-12]] — parent chapter.
- [[excerpts/cuda-graphs-inference]] — the underlying mechanism.
- [[excerpts/vllm-piecewise-graphs]] — the baseline SGLang improves on.
- [[ch-07]] / [[sglang-radixattention]] — the prefix-cache structure that enables the scheduling trick.
- [[ch-17]] — full SGLang internals (scheduler + RadixAttention end-to-end).
