---
chapter: ch-04
course: llm-inference
phase: read
excerpt_of: "vLLM V1 Scheduler (vllm/v1/core/sched/scheduler.py + design docs)"
source_url: https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py
created_at: "2026-05-21"
---

# Excerpt: vLLM scheduler internals — WAITING / RUNNING / SWAPPED state machine

**Authors:** vLLM project (Kwon et al. + community)
**Year:** 2024-present (V1 engine)
**URLs:** https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py ; https://docs.vllm.ai/en/latest/api/vllm/v1/core/sched/scheduler/
**Raw-data source:** [[raw-data/vllm-scheduler]]

---

## The three-state machine

```
                        admit (KV available + budget)
              WAITING ──────────────────────────────→ RUNNING
                ↑                                       │
                │                                       │ finish (EOS / max_tokens)
                │                                       ↓
                │                                      DONE
                │                                       │
                │                          preempt (KV pressure)
                │                                       ↓
              SWAPPED ←──────────────────────── (swap-out: KV → CPU)
                │
                └──── re-admit (KV available) ────→ RUNNING
                       (swap-in: KV ← CPU)
```

Each request (technically each "sequence group" — a request plus all its beam children) lives in exactly one state at a time.

---

## The schedule loop (paraphrased from `scheduler.py`)

```python
def schedule(self) -> SchedulerOutput:
    token_budget = self.scheduler_config.max_num_batched_tokens
    seq_budget = self.scheduler_config.max_num_seqs
    scheduled = []

    # PASS 1: Continue running decodes and chunked prefills.
    for seq_group in list(self.running):
        if len(scheduled) >= seq_budget or token_budget <= 0:
            break
        n_tokens = self._next_step_token_count(seq_group)   # 1 for decode, ≤ chunk_size for prefill
        if self.kv_cache_manager.allocate_slots(seq_group, n_tokens):
            scheduled.append((seq_group, n_tokens))
            token_budget -= n_tokens
        else:
            # KV exhausted; preempt and retry next step.
            self._preempt(seq_group)
            self.running.remove(seq_group)

    # PASS 2: Admit waiting prompts (new requests).
    while self.waiting and len(scheduled) < seq_budget and token_budget > 0:
        new = self.waiting[0]
        # Maybe chunk the prefill if it exceeds remaining budget.
        n_tokens = min(new.unprocessed_prompt_len, token_budget)
        if self.kv_cache_manager.allocate_slots(new, n_tokens):
            self.waiting.popleft()
            self.running.append(new)
            scheduled.append((new, n_tokens))
            token_budget -= n_tokens
        else:
            break

    # PASS 3: Re-admit swapped (after pressure relieved).
    while self.swapped and len(scheduled) < seq_budget and token_budget > 0:
        swapped = self.swapped[0]
        if self.kv_cache_manager.allocate_slots(swapped, 1) and \
                self.kv_cache_manager.swap_in(swapped):
            self.swapped.popleft()
            self.running.append(swapped)
            scheduled.append((swapped, 1))
            token_budget -= 1
        else:
            break

    return SchedulerOutput(scheduled=scheduled, ...)
```

The `SchedulerOutput` is consumed by the model executor, which runs one batched forward pass containing the scheduled token positions for each request.

---

## Preemption policies

When KV is exhausted, the scheduler must free blocks. Two strategies:

### Recompute

- Drop the preempted sequence's KV cache entirely.
- On re-admit, re-prefill from the prompt (which the scheduler remembers).
- **Cheap for**: short prompts (re-prefill cost low).
- **Expensive for**: long prompts (full prefill cost paid again).

### Swap

- Move the preempted sequence's KV blocks to CPU memory (over PCIe).
- On re-admit, swap blocks back to GPU.
- **Cheap for**: long prompts (avoid re-prefill).
- **Expensive for**: high preemption rate (PCIe bandwidth saturates).

vLLM picks per-sequence based on a length threshold (recompute below, swap above). Configurable via `preemption_mode`.

---

## KV cache manager integration

The scheduler doesn't allocate raw memory — it asks `kv_cache_manager` for blocks. The manager:

- Maintains a pool of physical KV blocks (typically 16 tokens each — PagedAttention; ch-06).
- Per-sequence block tables map logical token positions to physical blocks.
- Supports prefix-cache lookups (APC; ch-07) for shared prompt prefixes.

Schedule decisions thus become block-allocation decisions:

```python
def allocate_slots(self, seq_group, num_new_tokens) -> bool:
    new_blocks_needed = ceil((seq_group.current_len + num_new_tokens) / block_size) - len(seq_group.block_table)
    if self.free_blocks >= new_blocks_needed:
        self.free_blocks -= new_blocks_needed
        seq_group.block_table.extend(self.allocate_n_blocks(new_blocks_needed))
        return True
    return False
```

If admission fails, the scheduler tries preemption or keeps the sequence waiting.

---

## What other managers attach to the scheduler

The V1 scheduler is the orchestration hub. It coordinates with:

- `KVCacheManager` — block allocation, prefix matching, free-pool management
- `EncoderCacheManager` — for encoder-decoder / multimodal models with cached encoder outputs
- `StructuredOutputManager` — grammar bitmask updates per decode step (ch-01 §5)
- `SpecDecodeManager` — speculative decoding state (ch-14)
- `KVConnector` — for disaggregated serving, ships KV blocks to remote decode pool (ch-09)

The `SchedulerOutput` carries enough metadata for all of these to do their per-step updates.

---

## Tuning knobs exposed via CLI

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --max-num-seqs 256 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --max-num-batched-tokens 8192 \
  --preemption-mode swap \
  --swap-space 16  # GiB of CPU swap space
```

Each knob corresponds to a scheduler-level decision. `--enable-chunked-prefill` activates the SARATHI-style mixed batching (ch-05); `--enable-prefix-caching` activates APC (ch-07).

---

## What changed in V1 vs V0

vLLM rewrote the scheduler in 2024 (V1). The V0 scheduler had:
- Separate phases for prefill and decode (couldn't mix)
- Single-pass admission (no chunked prefill in the same pass)
- Less hookable extension points

V1's design ([[raw-data/vllm-scheduler]]):
- Unified token-budget accounting (prefill chunks and decode tokens compete for the same budget)
- Explicit pass structure (PASS 1: running, PASS 2: waiting, PASS 3: swapped)
- Hookable cache/structured-output/spec-decode managers
- Better preemption policies

---

## Common pitfalls

- **`max_num_seqs` set arbitrarily high**. Without enough KV pool, sequences thrash between SWAPPED and RUNNING. Symptom: high tail latency, frequent preemption logs.
- **Confusing `max_num_seqs` with `max_num_batched_tokens`**. They cap different things (sequence count vs token compute). Both matter.
- **Disabling chunked prefill on long-prompt workloads**. TPOT will spike each time a long prompt is admitted.
- **Not reading scheduler logs**. vLLM logs preemption events; if you see them, you're under-provisioned.

---

## Connections

- [[excerpts/orca-iteration-scheduling]] — the algorithm; vLLM V1 implements at production scale.
- [[excerpts/continuous-batching-gains]] — measured throughput improvements from this design.
- [[raw-data/vllm-kv-cache-manager]] — the block allocator the scheduler delegates to.
- [[raw-data/pagedattention]] — the underlying KV-block design.
- [[ch-05]] — chunked prefill: how the scheduler mixes prefill chunks with decode in one pass.
- [[ch-16]] — full code-level deep dive of vLLM internals (scheduler + cache manager + structured output).
