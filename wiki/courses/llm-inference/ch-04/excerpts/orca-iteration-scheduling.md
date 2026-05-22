---
chapter: ch-04
course: llm-inference
phase: read
excerpt_of: "Orca: A Distributed Serving System for Transformer-Based Generative Models (Yu et al. OSDI 2022)"
source_url: https://www.usenix.org/conference/osdi22/presentation/yu
created_at: "2026-05-21"
---

# Excerpt: Orca — iteration-level scheduling and selective batching

**Authors:** Gyeong-In Yu, Joo Seong Jeong, Geon-Woo Kim, Soojeong Kim, Byung-Gon Chun
**Year:** 2022
**Venue:** OSDI 2022
**URL:** https://www.usenix.org/conference/osdi22/presentation/yu
**Raw-data source:** [[raw-data/orca]]

---

## The contribution in one sentence

Schedule autoregressive Transformer inference at the granularity of **one decode iteration**, not one whole request — so completed requests release capacity immediately and waiting requests can join immediately.

---

## The static-batching pathology Orca diagnoses

The Orca paper opens with a worked example showing why request-level batching fails for autoregressive workloads. Suppose 4 requests arrive simultaneously with output lengths `[100, 300, 1000, 100]`:

```
Static (request-level) batching timeline:
  step 1..100:    all 4 active        (full utilization)
  step 101..300:  3 active            (request 1 done, but stays in batch consuming KV)
  step 301..1000: 1 active            (only request 3 still generating)

Total wall time: 1000 decode steps × T(step)
GPU time wasted: requests 1, 2, 4 contribute (900 + 700 + 900) = 2500 wasted forward-pass slots
Utilization: ~37% over the full batch lifetime
```

Variable output lengths are the norm in chat; uniform output length is the exception. The pathology is unavoidable in static batching.

---

## Iteration-level scheduling — the algorithm

```
loop forever:
    active_set = all currently-running requests with KV still allocated
    finished_set = {r in active_set : r.output[-1] == EOS or r.length == max_tokens}
    for r in finished_set:
        emit(r.output)
        free_kv(r)
        active_set.remove(r)

    while len(active_set) < max_seqs and queue.nonempty() and kv_budget_ok():
        new = queue.pop()
        allocate_kv(new)
        active_set.add(new)

    # ONE forward pass over the current active_set
    logits = model.step(active_set)
    for r in active_set:
        token = sample(logits[r], r.sampling_params)
        r.append(token)
        emit_delta(r)
```

The "batch" lives for exactly one decode step. It is rebuilt completely on the next iteration.

---

## Selective batching — the operations-level subtlety

Not every operation in the generation loop can naturally be batched across requests with arbitrary state. Orca's selective batching:

**Batched** (uniform tensor shapes across requests):
- Q, K, V projections (one large matmul against shared model weights)
- Attention (one batched attention call with per-request KV)
- FFN gate, up, down projections
- LayerNorm, residual adds

**Unbatched** (per-request, different shapes or logic):
- Sampling (different temperature, top-p, top-k per request)
- Stop-string matching (different stop strings per request)
- EOS / max-tokens checks
- Structured-output mask updates (different grammars per request)
- Logits-bias application (different bias vectors per request)

Modern implementations (vLLM, SGLang) handle the unbatched part on the host CPU in parallel with the next forward pass, so unbatched logic doesn't serialize the batched compute.

---

## Distributed execution

Orca was designed for very large models (GPT-3 175B at the time) and supports distributed execution via tensor + pipeline parallelism. The scheduler is centralized; workers execute the model in TP/PP groups.

The scheduling decisions happen at the granularity of model microbatches under pipeline parallelism: each pipeline stage's forward pass is one "iteration" from the scheduler's perspective. Selective batching ensures that operations needing per-request state (sampling) happen at well-defined boundaries.

---

## Reported gains

On GPT-3 175B with FasterTransformer as baseline:
- **Throughput**: >10× at the same latency
- **Latency**: comparable to FasterTransformer at low load; substantially lower at high load (no head-of-line blocking)

These numbers became the standard against which subsequent serving papers compared (vLLM/PagedAttention, SGLang, TensorRT-LLM in-flight batching).

---

## What Orca did NOT do

- **No paged KV management**: Orca uses contiguous KV per request. KV fragmentation is significant under heterogeneous lengths. → solved by [[raw-data/pagedattention]] (ch-06).
- **No prefill-decode mixing**: a long prefill blocks all decodes for that iteration. → solved by [[raw-data/sarathi-serve]] (ch-05).
- **No fairness across tenants**: FCFS admission. → solved by [[raw-data/vtc]] (ch-10).
- **No prefix sharing**: shared system prompts are recomputed for every request. → solved by [[raw-data/sglang-radixattention]] (ch-07).

These follow-ons all *inherit* iteration-level scheduling. None of them replace it.

---

## Why this is the lasting contribution

Every modern serving stack implements iteration-level scheduling: vLLM, SGLang, TGI, TensorRT-LLM ("in-flight batching" is NVIDIA's name for the same idea), DeepSpeed-FastGen, LightLLM. The vocabulary varies but the algorithm is essentially identical.

The reason it lasted: it's the *only* batching strategy that matches the structure of autoregressive generation. Static batching is fundamentally incompatible with variable-length outputs; dynamic batching is just static batching with better admission; iteration-level scheduling is the structurally correct solution.

---

## Common pitfalls

- **Confusing "continuous batching" with the broader family**. "Continuous batching" is the practitioner term; "iteration-level scheduling" is the research term. Same algorithm.
- **Assuming Orca solves KV fragmentation**. It doesn't. Without PagedAttention, you get iteration-level scheduling on top of a contiguous KV allocator, which wastes 30–60% of cache. The combination (Orca scheduling + PagedAttention) is what vLLM ships.
- **Tuning batch size as if static**. Under iteration-level scheduling, "batch size" is a dynamic distribution, not a fixed number. Tune `max_num_seqs` (cap) and let the scheduler vary the active set.

---

## Connections

- [[excerpts/vllm-scheduler-states]] — vLLM's WAITING / RUNNING / SWAPPED implements Orca's scheduling at production scale.
- [[excerpts/continuous-batching-gains]] — quantifies the 2–5× throughput claim on real workloads.
- [[raw-data/pagedattention]] — the matched KV-management innovation.
- [[ch-05]] — chunked prefill: refinement that handles prefill-decode interference inside the same scheduler.
- [[ch-09]] — disaggregation: separate scheduler per phase, but iteration-level inside each.
