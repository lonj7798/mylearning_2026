---
chapter: ch-05
course: llm-inference
phase: read
excerpt_of: "vLLM V1 Scheduler — vllm/v1/core/sched/scheduler.py"
source_url: https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py
created_at: "2026-05-21"
---

# Excerpt: vLLM V1 scheduler — chunked prefill in production code

**Authors:** vLLM project
**Year:** 2024-present (V1 engine, default since v0.6)
**URLs:** https://docs.vllm.ai/en/latest/api/vllm/v1/core/sched/scheduler/ ; https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py
**Raw-data source:** [[raw-data/vllm-scheduler]]

---

## What this module owns

The V1 scheduler is the policy module that decides **what each engine step computes**. Concretely, on each engine tick it:

1. Walks the `waiting` queue (queued requests) and the `running` collection (active requests).
2. Allocates per-token budgets up to `max_num_batched_tokens`.
3. Asks the `KVCacheManager` whether physical KV blocks are free for new work.
4. Preempts (frees KV from) lower-priority running requests if memory pressure prevents admission.
5. Emits a `SchedulerOutput` consumed by the model runner.

The contract is intentionally narrow: scheduler decides *what runs*; worker runs it. Tracing one step in the source code is the fastest way to understand chunked prefill as a system.

---

## The scheduling loop (paraphrase of `Scheduler.schedule()`)

```python
# vllm/v1/core/sched/scheduler.py — simplified to the chunked-prefill skeleton

def schedule(self) -> SchedulerOutput:
    token_budget = self.scheduler_config.max_num_batched_tokens
    scheduled_running: list[CachedRequestData] = []
    scheduled_new:     list[NewRequestData]    = []

    # 1) RUNNING — every active request gets at least one decode token.
    for req in self.running:
        n = 1  # decode one token
        # Note: speculative decoding bumps `n` to K + 1 for verify steps.
        if not self.kv_cache_manager.allocate_slots(req, n):
            self._preempt_to_make_room(req)        # frees KV from another req
            if not self.kv_cache_manager.allocate_slots(req, n):
                continue                            # still no room: skip this req
        scheduled_running.append(req); token_budget -= n

    # 2) WAITING — admit new prefills (possibly chunked) up to remaining budget.
    while self.waiting and token_budget > 0:
        req = self.waiting[0]
        # `num_computed_tokens` reflects prefix-cache hits (see ch-07).
        prefill_left = req.num_prompt_tokens - req.num_computed_tokens
        take = min(prefill_left, token_budget, self.scheduler_config.max_chunk_size)

        if not self.kv_cache_manager.allocate_slots(req, take):
            break                                   # KV full; defer admission
        scheduled_new.append(req); token_budget -= take
        req.num_computed_tokens += take
        if req.num_computed_tokens == req.num_prompt_tokens:
            self.waiting.popleft(); self.running.append(req)

    return SchedulerOutput(scheduled_new, scheduled_running, ...)
```

Three details that catch new readers:

- **`num_computed_tokens` not `num_prefilled_tokens`.** A prefix-cache hit also bumps this counter, so a "new" request can enter `running` immediately with no actual compute. Same scheduler path, different cost.
- **Decode promotion is implicit.** Once a `waiting` request's prefill is fully scheduled, it moves to `running` and is treated as a normal decode request next iteration. There is no separate "promote" RPC.
- **Token budget is shared.** Decode tokens and prefill-chunk tokens both deduct from `token_budget`. There is no separate prefill budget; it's whatever decode leaves behind.

---

## The KV-cache memory check (`allocate_slots`)

```python
# vllm/v1/core/kv_cache_manager.py — paraphrased

def allocate_slots(self, request, num_tokens):
    blocks_needed = self._blocks_needed(request, num_tokens)  # block_size=16
    if blocks_needed > len(self.free_block_pool):
        return None                  # caller must preempt or defer
    new_blocks = self.free_block_pool.pop(blocks_needed)
    request.block_table.extend(new_blocks)
    return new_blocks
```

KV allocation is *exactly* block-granular. A request extending by 17 tokens with `block_size=16` and 1 used slot in its current last block: needs 1 new block (16 - 1 = 15 free slots in the current block, plus 2 more from a new block). This is the bookkeeping ch-06 covers in depth.

---

## Server flags that map directly to this loop

| CLI / `VllmConfig` flag | What it changes in the loop |
|---|---|
| `--max-num-batched-tokens N` | The initial `token_budget` per step. |
| `--max-num-seqs N` | Cap on `len(running)` — i.e., parallel decode batch size. |
| `--long-prefill-token-threshold N` | Effective `max_chunk_size` per prefill admission. |
| `--enable-chunked-prefill` (default True in V1) | If False, falls back to "fill the entire prompt in one step." |
| `--gpu-memory-utilization 0.9` | Sizes the KV-block pool; smaller pool → more preemption pressure. |
| `--enable-prefix-caching` | Lets `num_computed_tokens` count cache hits (ch-07). |

These are the flags you actually turn in production deployments. Everything else is convenience.

---

## Preemption — the safety valve

When a running request needs to allocate but `free_block_pool` is empty:

```python
def _preempt_to_make_room(self, req):
    victim = self._pick_preemption_victim()         # LIFO by default
    if self.preemption_mode == "recompute":
        self.kv_cache_manager.free(victim)          # blocks go back to pool
        self.waiting.appendleft(victim)             # restart prefill later
        victim.num_computed_tokens = 0              # ⚠ loses partial prefill
    else:  # "swap"
        self._swap_to_cpu(victim)
```

Preemption is bounded — a request can be preempted at most a few times before the scheduler escalates (e.g., 429 to client). The metric to monitor is `vllm:num_preemptions_total`; a spike here means `gpu_memory_utilization` is too tight or `max_num_seqs` is too high.

---

## What this code does *not* do

The scheduler is intentionally policy-thin. Things it does *not* own:

- The attention kernel (FlashAttention / FlashInfer / Triton — ch-11).
- The forward pass itself (`vllm/v1/worker/`).
- Sampling (`vllm/v1/sample/`).
- The structured-output bitmask (xgrammar / outlines — separate manager, fed into `SchedulerOutput`).
- Speculative decoding's verify/accept logic — the scheduler only allocates `K+1` slots; the worker decides accept/reject (ch-14).

Keeping these out of `Scheduler.schedule()` is what lets vLLM swap features in/out without rewriting the loop.

---

## Connections

- [[excerpts/sarathi-serve]] — the algorithm this code implements.
- [[excerpts/continuous-batching]] — the iteration-level scheduling pattern the loop instantiates.
- [[ch-05]] — parent synthesis.
- Forward to [[ch-16]] — full vLLM internals tour including the worker, engine, and async server.
- Forward to [[ch-07]] — prefix caching path (`num_computed_tokens` is the integration point).
