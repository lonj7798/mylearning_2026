---
chapter: ch-03
course: llm-inference
phase: read
excerpt_of: "Batching for Inference (synthesis: Orca + vLLM + HF docs)"
source_url: https://www.usenix.org/conference/osdi22/presentation/yu
created_at: "2026-05-21"
---

# Excerpt: Static, dynamic, continuous batching — the three baselines

**Authors:** Gyeong-In Yu et al. (Orca, 2022); Woosuk Kwon et al. (vLLM/PagedAttention, 2023); HF Transformers team
**Year:** 2022–present
**URLs:** https://www.usenix.org/conference/osdi22/presentation/yu ; https://arxiv.org/abs/2309.06180
**Raw-data source:** [[raw-data/batching-for-inference]]

---

## Static batching — the worst baseline

```python
def static_batch_serve(requests, model):
    # 1. Wait for N requests.
    batch = requests[:N]
    # 2. Pad all to max prompt length.
    padded_inputs = pad_to_max([req.prompt for req in batch])
    # 3. Run prefill in lockstep.
    kv_cache = model.prefill(padded_inputs)
    # 4. Generate in lockstep until ALL hit EOS or max_tokens.
    outputs = [[] for _ in batch]
    while not all_done(outputs):
        next_tokens = model.decode_step(kv_cache)
        for i, tok in enumerate(next_tokens):
            if not outputs[i] or outputs[i][-1] != EOS:
                outputs[i].append(tok)
    # 5. Return all responses simultaneously.
    return [(req, out) for req, out in zip(batch, outputs)]
```

**Pathology 1 — head-of-line blocking**: if 9 requests finish at 100 tokens and 1 finishes at 1000, all 9 short ones stay in the batch wasting compute for 900 extra decode steps.

**Pathology 2 — padding waste**: lengths `[100, 200, 500, 1000]` padded to 1000 waste `2200/4000 = 55%` of prefill compute.

**Pathology 3 — admission delay**: new requests wait for the entire current batch to finish. At 30 s P99 batch latency, that's a 30 s admission tax.

**GPU utilization**: typically 10–30% on production traffic shapes. **Where it survives**: offline batch inference (e.g. embeddings, scoring) where requests are uniform.

---

## Dynamic batching — incremental but still pathological

```python
def dynamic_batch_serve(request_queue, model):
    while True:
        # Wait for either N requests OR timeout
        batch = collect_batch(request_queue, max_size=N, max_wait=T)
        # Run as static batch from here.
        run_static_batch(batch, model)
```

Same fundamental problems as static, but improves admission delay. **GPU utilization**: 20–40%. Still the wrong design for autoregressive LLMs.

---

## Continuous batching (Orca's iteration-level scheduling) — the standard

```python
def continuous_batch_serve(request_queue, model, kv_pool, max_seqs):
    active = {}                                       # seq_id -> state
    while True:
        # 1. Sweep finished sequences out.
        for seq_id, state in list(active.items()):
            if state.is_done():
                emit_final(state)
                kv_pool.free(state.kv_blocks)
                del active[seq_id]

        # 2. Admit waiting sequences (subject to KV budget).
        while len(active) < max_seqs:
            new_req = request_queue.try_pop()
            if not new_req:
                break
            kv_needed = predict_kv_blocks(new_req)
            if not kv_pool.can_allocate(kv_needed):
                request_queue.put_back(new_req)
                break
            active[new_req.id] = SeqState(new_req, kv_pool.allocate(kv_needed))

        if not active:
            continue

        # 3. ONE forward pass for the current active set.
        logits = model.step(active)

        # 4. Sample one token per active sequence; check stop conditions.
        for seq_id, state in active.items():
            token = state.sampler(logits[seq_id])
            state.append(token)
            if state.is_done():
                state.mark_finished()

        # 5. Stream deltas to clients.
        for seq_id, state in active.items():
            state.emit_delta_if_new()
```

**Key properties**:
- The batch is reconstituted every decode step (no static commitment).
- Sequences run at their own lengths (no padding).
- New requests admitted within one decode-step latency (~50 ms).
- KV budget is the binding constraint (not nominal request count).

**Reported gains** (Orca paper, Llama-class models):
- 2–5× throughput vs static batching at same TPOT
- 70–90% GPU utilization (vs 10–30% for static)
- Latency improved because tail-blocking is eliminated

This is what vLLM, SGLang, TGI, TensorRT-LLM all implement. Modern LLM serving is essentially: continuous batching + PagedAttention + chunked prefill + CUDA Graphs.

---

## The schedule decisions that continuous batching exposes

The scheduler must decide at every iteration:

1. **Admission**: which waiting requests to admit (FCFS, priority, fairness)
2. **Preemption**: when KV is exhausted, which active requests to swap out
3. **Prefill mix**: how many prefill tokens to compute alongside decode tokens (chunked prefill)
4. **Token budget**: `max_num_batched_tokens` — total compute budget per step
5. **Memory budget**: `max_num_seqs` — concurrent sequence cap; bounded by KV pool

vLLM's scheduler (ch-04 and ch-16) implements all of these as policy hooks.

---

## Why static batching is unrecoverable for chat

The numbers are stark. Suppose 100 chat requests arrive with output lengths distributed as `Geom(0.005)` (mean 200 tokens, max ~1000 tokens). Static batching at `B=100`:

```
batch latency  =  T(prefill) + max(output_lens) · T(decode_step)
              ≈  100 ms + 1000 · 30 ms
              =  30.1 seconds
total compute  =  100 · 1000 · T(decode_step) = 3000 s of compute
useful compute =  100 · 200 · T(decode_step) = 600 s
waste          =  80% of decode steps were no-op for finished sequences
```

Continuous batching at the same arrival rate maintains average batch size ~20–30 (sequences finish and new ones admit), each running at its own length. Mean response latency: ~6 s. GPU utilization: 70%+.

---

## Common pitfalls

- **Setting `max_num_seqs` too high without enlarging KV pool**: scheduler admits beyond capacity, preempts immediately. Symptom: tail latency spikes.
- **Setting `max_num_batched_tokens` too low**: limits prefill chunk size; TTFT degrades.
- **Setting them too high**: prefill bursts crash decode TPOT (the SARATHI motivation; ch-05).
- **Ignoring KV pool fragmentation**: in pre-PagedAttention systems, continuous batching helps less because allocator can't pack diverse sequence lengths efficiently.

---

## Connections

- [[excerpts/kv-cache-formula]] — KV pool is the binding constraint on continuous batch size.
- [[excerpts/prefill-vs-decode]] — admission decisions must consider both phases' resource needs.
- [[raw-data/orca]] — the founding paper.
- [[raw-data/vllm-scheduler]] — modern reference implementation.
- [[ch-04]] — full chapter on continuous batching internals.
- [[ch-05]] — chunked prefill: the second-order refinement.
