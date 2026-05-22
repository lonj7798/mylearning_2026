---
chapter: ch-05
course: llm-inference
phase: read
excerpt_of: "Continuous Batching / Iteration-Level Scheduling (synthesis card around Orca + vLLM + TGI)"
source_url: https://www.usenix.org/conference/osdi22/presentation/yu
created_at: "2026-05-21"
---

# Excerpt: continuous batching — the substrate chunked prefill upgrades

**Authors:** Synthesis around Orca (Yu et al., OSDI 2022), vLLM, HF TGI / Transformers serving
**Year:** 2022–present
**URL:** https://www.usenix.org/conference/osdi22/presentation/yu
**Raw-data source:** [[raw-data/continuous-batching]]

---

## Why "iteration-level scheduling" is the right name

Static batching schedules **at the request level**: a batch is admitted together and must finish together. The longest output stalls every short output. New arrivals wait until the whole batch drains.

Continuous batching schedules **at the iteration level**: at every model step the scheduler rebuilds the active batch from currently running decode requests plus any newly admitted prompts. Finished requests release capacity *immediately*; new requests enter as soon as token budget and KV-cache room allow.

The name shift matters: it implies the scheduling decision happens once per token, not once per request.

```text
Static batching                     Continuous batching
=================                   ===================
batch = [req1, req2, req3]          step 1: batch = [req1, req2, req3]
run prefills together               step 2: batch = [req1, req2, req3]
run all decodes until last EOS      step 3: req2 done → batch = [req1, req3, req4 (newly admitted prefill)]
return all together                 step 4: batch = [req1, req3, req4]
                                    step 5: req1 done → batch = [req3, req4, req5]
                                    ...
```

---

## What it gives you

From Orca's headline number and subsequent production replications (vLLM, TGI):

- **2–5× throughput at iso-latency vs static batching.** On GPT-3 175B with realistic length distributions, Orca measured ~37× normalized latency improvement at fixed throughput vs FasterTransformer.
- **Zero head-of-line blocking from variable output lengths.** A 10-token reply and a 1000-token reply no longer share a deadline.
- **Streaming-friendly.** Each request emits tokens as soon as its decode step completes, independent of others.

These wins are why continuous batching is the default in every modern serving stack (vLLM, SGLang, TensorRT-LLM in-flight batching, TGI, LightLLM).

---

## The scheduler invariants

Every continuous-batching scheduler enforces, simultaneously:

| Invariant | Why |
|---|---|
| `len(running) ≤ max_num_seqs` | Bounds parallel-decode batch size for GPU compute efficiency |
| `sum(per-step tokens) ≤ max_num_batched_tokens` | Bounds per-step latency |
| `sum(per-request KV blocks) ≤ kv_block_pool_size` | Bounds physical KV memory |
| each request's context ≤ `max_model_len` | Bounds attention compute |

Of these, the *token budget* and the *KV-cache budget* are the two that bind in practice. The token budget is a compute-utilization knob; the KV budget is a memory-utilization knob. Tuning either changes both throughput and tail latency.

---

## The mix-prefill-with-decode problem (the gap chunked prefill fills)

Vanilla continuous batching admits a new prompt by running its entire prefill in a single step. For long prompts, that step's latency is dominated by prefill compute — and *all running decodes in that step pay the cost*.

```text
step latency = max(prefill_cost(longest_admitted_prompt),
                   decode_cost(num_running_decodes))
```

When a 4k-token prefill enters, step latency jumps from ~20 ms to ~300 ms; every streaming user sees a one-token spike that big. Pure continuous batching gives you no knob to bound this — prefill is atomic.

Chunked prefill (ch-05) is the upgrade: break the prefill into chunks, treat each chunk as a discrete token-budget consumer, and let decodes co-execute with each chunk. After chunked prefill, the formula becomes:

```text
step latency ≈ kernel_cost(max_num_batched_tokens)   # bounded by config knob
```

That replacement — from "depends on the longest admitted prompt" to "bounded by a configured constant" — is what makes p99 TPOT predictable.

---

## Continuous batching ≠ a single algorithm

The synthesis card emphasizes: **there is no single canonical "continuous batching" paper**. Orca introduces the term *iteration-level scheduling* and *selective batching*. vLLM ships *paged continuous batching*. TGI ships *in-flight batching*. SGLang ships *cache-aware continuous batching*. All four share the iteration-level scheduling skeleton; they differ in:

| System | Distinguishing addition |
|---|---|
| Orca (OSDI 2022) | Selective batching for ops that can't share shapes |
| vLLM | PagedAttention block tables → no per-request KV reservation |
| TGI | Rust-frontend router + per-shard scheduling |
| SGLang | RadixAttention tree-keyed prefix matching |
| TensorRT-LLM | Ahead-of-time engine + inflight batching |

What unifies them is the answer to *when do we decide what to run?* (every iteration) and *what consumes resources?* (tokens and KV blocks, not requests).

---

## Why this is the right substrate for ch-05

Chunked prefill (Sarathi-Serve) inherits *everything* in this card and changes one thing: prefill is no longer atomic. The scheduler's invariants stay the same; what changes is that a single waiting request can be admitted across multiple steps. Once you internalize iteration-level scheduling, chunked prefill is "one more degree of freedom in how you spend a step's token budget."

---

## Connections

- [[excerpts/orca]] — the OSDI 2022 paper that introduced iteration-level scheduling.
- [[excerpts/sarathi-serve]] — chunked prefill, the natural upgrade.
- [[excerpts/vllm-scheduler]] — the production realization in vLLM V1.
- [[ch-04]] — chapter dedicated to continuous batching alone.
- [[ch-05]] — this chapter, where the upgrade lives.
