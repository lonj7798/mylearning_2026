<!-- chapter: ch-04
     track: scheduling
     title: Continuous Batching — Iteration-Level Scheduling
     sources: [[orca]], [[continuous-batching]], [[vllm-scheduler]]
     figures: figures/continuous-batching-timeline.html
-->

# Chapter 4 — Continuous Batching: Iteration-Level Scheduling

> **Core insight.** Orca's 2022 contribution was to rebuild the batch *at every decode step* rather than at every request boundary. Combined with PagedAttention's KV management, this single scheduling change pushed production GPU utilization from ~15% (static batching) to ~70% — the 2–5× throughput claim that defines the modern LLM serving era. Every production stack (vLLM, SGLang, TGI, TensorRT-LLM) implements some variant of the WAITING / RUNNING / SWAPPED state machine.
>
> **Guideline.** When tuning a continuously-batched serving system, the three knobs that matter are `max_num_seqs` (concurrent sequence cap), `max_num_batched_tokens` (per-step compute budget), and `gpu_memory_utilization` (KV pool fraction of HBM). Set them in that order: KV pool first (~0.85 of free HBM), then derive `max_num_seqs` from your context length, then set `max_num_batched_tokens` based on whether you want TTFT-favored (large, e.g. 8192) or TPOT-favored (small, e.g. 1024) scheduling.

---

## Why this chapter exists

Continuous batching is the load-bearing scheduling primitive of modern LLM serving. Every later optimization sits on top of it: chunked prefill (ch-05) adds prefill-decode mixing to the same loop; PagedAttention (ch-06) makes KV allocation feasible at this scheduling granularity; disaggregation (ch-09) splits the loop across GPU pools but keeps the iteration-level structure inside each. If you only learn one serving algorithm, learn this one.

Three things you should walk away with:

1. The exact iteration-level scheduling loop (Orca's contribution) and why "selective batching" is the second half of it.
2. The vLLM scheduler's WAITING / RUNNING / SWAPPED state machine and the preemption logic that handles KV-cache pressure.
3. The 2–5× throughput claim, what workload regime it holds in, and the tuning knobs (`max_num_seqs`, `max_num_batched_tokens`, priority, preemption policy) that determine the actual gain.

Sources: [[orca]], [[continuous-batching]], [[vllm-scheduler]] in the raw-data library.

---

## 1. The static-batching pathology (recap)

Static batching: take `N` requests, pad to common length, run prefill + decode in lockstep until all finish. From ch-03 we already saw the failure modes — head-of-line blocking, padding waste, slow admission. The numbers in production:

```
100 chat requests, output lengths ~ Geom(0.005), mean=200, max=1000

static batching (B=100):
  batch latency:        T(prefill) + 1000 · T(decode)  ≈  30 seconds
  GPU utilization:      ~15% (most decode steps no-op)
  fairness:             worst — short requests wait for longest

continuous batching:
  per-request latency:  mean 6 s
  GPU utilization:      ~70%
  fairness:             FIFO admission, short requests finish first
```

The 30-second-vs-6-second comparison is the entire reason Orca exists.

---

## 2. Orca's two contributions: iteration-level scheduling + selective batching

[[orca]] (Yu et al., OSDI 2022) names both ideas precisely.

### Iteration-level scheduling

Don't form a "batch of requests"; form a "batch of one decode step". At every model step:

```
for ever:
    1. Drop finished requests; free their KV.
    2. Admit waiting requests if KV budget allows.
    3. Forward pass for the current active set (1 step).
    4. Sample and emit deltas.
```

Steps 1–2 happen between every model invocation, so the "batch" is dynamic at decode-step granularity (~50 ms). A finished request leaves immediately; a new request joins within one step.

### Selective batching

Not every operation in the generation loop can be batched across requests with arbitrary state. Some operations have shapes that match across requests (matmul on the same model weights); some are per-request and shape-dependent (sampling with different temperatures, repetition penalties, stop-string matching, structured-output mask updates).

Orca's solution: batch the model-weight operations (everything inside the decoder layer) and keep the per-request logic outside the batch. Concretely:

```
batched:        Q/K/V projection, attention, FFN, layernorm  (same weights, all requests)
unbatched:      sampling, EOS check, max-tokens check, grammar advance, delta emission
```

Modern stacks fuse the unbatched logic into per-request CUDA streams that run in parallel with the next forward pass.

---

## 3. The 2–5× throughput claim — where it holds

Orca's headline: **>10× throughput** on GPT-3-175B at the same SLO vs FasterTransformer (the pre-Orca state of the art). vLLM's headline: **24× throughput** vs HF Transformers, **2.2–3.5× vs Orca** at the same SLO ([[raw-data/pagedattention]]).

The gain is workload-dependent:

| Workload | Gain over static batching |
|---|---|
| Mixed prompt/output lengths (chat) | 5–10× |
| Uniform short outputs (embeddings) | 1.1–1.5× |
| Uniform long outputs (story) | 2–3× |
| Bursty arrival (production) | 5–10× |
| Long-context with high concurrency | 3–5× (KV-bound) |

The maximum gain comes from the heterogeneity of real workloads. If every request is the same shape, static and continuous batching converge.

---

## 4. The vLLM scheduler — WAITING / RUNNING / SWAPPED state machine

[[vllm-scheduler]] V1 (the rewritten 2024+ engine) implements continuous batching as a three-state machine over sequence groups (a "sequence group" is one request, possibly with multiple beams):

```
                       admit
              WAITING ─────────→ RUNNING ─────────→ DONE
                ↑                   │       finish
                │                   │
                │ admit on swap-in  │ preempt on KV pressure
                │                   ↓
              SWAPPED ←─────────────┘
                                  swap-out
```

- **WAITING**: request arrived, prompt not yet processed. Scheduler tries to admit at each step.
- **RUNNING**: currently in the active batch; gets a forward pass per step.
- **SWAPPED**: was running, KV cache evicted to CPU memory due to pressure. Re-admitted when KV frees up.

### The schedule loop (paraphrased from [`vllm/v1/core/sched/scheduler.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py))

```python
def schedule(self) -> SchedulerOutput:
    # 1. Drop finished from RUNNING; free their KV blocks.
    self._reap_finished()

    # 2. Decide budget for this step.
    token_budget = self.scheduler_config.max_num_batched_tokens
    seq_budget = self.scheduler_config.max_num_seqs

    # 3. Schedule running decodes first (1 token each); chunked prefills if any.
    scheduled = []
    for seq_group in self.running:
        if token_budget <= 0 or len(scheduled) >= seq_budget:
            break
        # Decode needs 1 token; prefill chunk needs up to chunk_size.
        n_tokens = self._next_step_tokens(seq_group)
        # Allocate KV slots; preempt lower-priority running if needed.
        if not self.kv_cache_manager.allocate_slots(seq_group, n_tokens):
            self._preempt_one()
            continue
        scheduled.append(seq_group)
        token_budget -= n_tokens

    # 4. Admit from WAITING (subject to remaining budget + KV availability).
    while self.waiting and token_budget > 0 and len(scheduled) < seq_budget:
        new = self.waiting[0]
        n_tokens = min(new.prompt_len, token_budget)  # may chunk
        if self.kv_cache_manager.allocate_slots(new, n_tokens):
            self.waiting.popleft()
            self.running.append(new)
            scheduled.append(new)
            token_budget -= n_tokens
        else:
            break  # KV exhausted; stop trying

    # 5. Re-admit SWAPPED if KV available now.
    while self.swapped and token_budget > 0 and len(scheduled) < seq_budget:
        ...

    return SchedulerOutput(scheduled=scheduled, ...)
```

### Preemption policy

When KV is exhausted and a new decode step needs blocks:
- **Recompute**: drop KV for some running sequence; re-prefill from prompt next time it's scheduled (cheap if prompt is short).
- **Swap**: move KV blocks to CPU memory; restore on re-admission (cheap if PCIe bandwidth is fast and sequences are long).

vLLM defaults to recompute below a threshold sequence length, swap above. The threshold is configurable.

---

## 5. The three production knobs

```
max_num_seqs                # concurrent sequence cap (e.g. 256)
max_num_batched_tokens      # per-step compute budget (e.g. 8192)
gpu_memory_utilization      # KV pool fraction of HBM (e.g. 0.90)
```

### Tuning order

1. **Set `gpu_memory_utilization` first** (typically 0.85–0.90). This determines KV pool size.
2. **Compute `max_num_seqs` from KV budget and your target context length**:
   ```
   max_num_seqs = (gpu_memory_utilization · HBM − weights) / (per_token_KV · max_context)
   ```
3. **Set `max_num_batched_tokens`** based on TTFT vs TPOT preference:
   - Large (8192+): TTFT-favored, prefills run fast, decodes may stall briefly
   - Small (1024–2048): TPOT-favored, smoother decode at cost of slower prefill
   - Default vLLM: 8192; SGLang: 2048

### Don't tune these in isolation

- `max_num_seqs > KV-budget allows` → constant preemption → tail latency spikes
- `max_num_batched_tokens < typical_prompt_size` → forces chunked prefill (good if intended; bad if accidental)
- `max_num_batched_tokens > KV-budget allows` → KV exhaustion mid-batch

---

## 6. Continuous batching + streaming

The Orca scheduler enables per-request streaming naturally: every decode step samples one token per active sequence; emit that token as an SSE delta on the request's stream. Requests finish independently; streams close independently. From [[openai-streaming-and-token-usage]] (ch-01) the client sees:

```
client A:  delta delta delta delta delta [stop]
client B:        delta delta delta delta delta delta delta [length]
client C:          delta delta [stop]
client D:            delta delta delta delta delta [length]
                      ...
```

Each client is reading from a single decode loop that's also serving the others. The "batch" is invisible to the client.

---

## 7. Limitations + when continuous batching is NOT enough

Continuous batching alone gives a big speedup but does not solve every problem:

1. **Prefill-decode interference**: a long prefill admitted into the batch blocks all decodes for that step. → chunked prefill (ch-05) or disaggregation (ch-09).
2. **KV fragmentation**: contiguous KV allocators waste 30–60% of cache memory. → PagedAttention (ch-06).
3. **Multi-tenant fairness**: FCFS admission doesn't prevent one client from starving others. → VTC fair scheduling (ch-10).
4. **SLO-aware admission**: admitting a request you can't meet wastes resources. → admission control + goodput optimization (ch-10).
5. **CUDA graph capture**: dynamic batch shapes break static-graph capture. → piecewise CUDA graphs per discrete batch bucket (ch-12).

Every chapter in phase 2–4 of this course is a refinement of continuous batching. They compose: vLLM today = continuous batching + chunked prefill + PagedAttention + APC prefix caching + piecewise CUDA graphs + structured output bitmasks + (optionally) disaggregation.

---

## 8. Cheat-sheet

```
CONTINUOUS BATCHING SCHEDULE:
  every step:
    1. drop finished, free KV
    2. admit waiting (if KV budget allows)
    3. one forward pass for active set
    4. sample + check stop + emit deltas

ORCA'S TWO IDEAS:
  - iteration-level scheduling (not request-level)
  - selective batching (model ops batched; per-request logic isolated)

VLLM SCHEDULER STATES:
  WAITING  ─[admit]─→ RUNNING ─[finish]─→ DONE
                       ↓ [preempt KV pressure]
                    SWAPPED  ─[admit]─→ RUNNING

THROUGHPUT GAINS (over static batching):
  chat workload:        5–10×
  uniform output:       1.5–3×
  long context:         3–5× (KV-bound)

THREE TUNING KNOBS (set in this order):
  1. gpu_memory_utilization  (0.85–0.90)  → KV pool size
  2. max_num_seqs            (derived)    → concurrent cap
  3. max_num_batched_tokens  (1k–8k)      → TTFT/TPOT tradeoff

PREEMPTION POLICIES:
  recompute  → drop KV, re-prefill on next admit (good for short prompts)
  swap       → KV to CPU memory (good for long prompts)

WHAT IT DOESN'T SOLVE:
  - prefill-decode interference  → ch-05 chunked prefill
  - KV fragmentation             → ch-06 PagedAttention
  - multi-tenant fairness        → ch-10 VTC
  - SLO-aware admission          → ch-10 admission control
```

---

## Connections and what's next

- **[[sarathi-serve]] / ch-05** — chunked prefill: extends the scheduler to mix prefill chunks with decode tokens in one forward pass, smoothing TTFT/TPOT.
- **[[pagedattention]] / ch-06** — block-based KV allocator that makes continuous batching's dynamic KV needs actually feasible; eliminates 30–60% fragmentation.
- **[[sglang-radixattention]] / ch-07** — RadixAttention adds shared-prefix matching to the same scheduler family.
- **[[distserve]] / ch-09** — disaggregation: separate prefill and decode onto different GPU pools, each running its own iteration-level scheduler.
- **[[vtc]] / ch-10** — Virtual Token Counter: fair scheduling on top of the same loop.
- **[[vllm-scheduler]] / ch-16** — full code-level deep dive of vLLM's scheduler implementation, with file paths and config flags.
- **[[ttft-tpot-itl]] / ch-19** — measuring what continuous batching's gains actually look like in production.

## Further reading

- [[orca]] — Yu et al. OSDI 2022; the founding paper for iteration-level scheduling.
- [[continuous-batching]] — synthesis card covering Orca + vLLM + HF docs as a single story.
- [[vllm-scheduler]] — V1 engine reference; the modern production implementation.
- [[pagedattention]] — vLLM paper; pairs continuous batching with paged KV for the 24× throughput claim.

## Companion visualization

**[figures/continuous-batching-timeline.html](figures/continuous-batching-timeline.html)** — animated timeline comparing static batching (lockstep, head-of-line blocking visible) with continuous batching (sequences entering / leaving the batch at every decode step). Drag the slider to advance time; see how a single long sequence in a static batch stalls all the short ones for 800+ ms, while continuous batching lets each finish at its natural latency.
