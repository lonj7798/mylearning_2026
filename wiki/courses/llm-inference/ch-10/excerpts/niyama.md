---
chapter: ch-10
course: llm-inference
phase: read
excerpt_of: "Niyama: Breaking the Silos of LLM Inference Serving"
source_url: https://arxiv.org/abs/2503.22562
created_at: "2026-05-21"
---

# Excerpt: Niyama — QoS-driven co-scheduling of interactive and batch traffic

**Authors:** Kanishk Goel, Jayashree Mohan, Nipun Kwatra, Ravi Shreyas Anupindi, Ramachandran Ramjee (MSR-India)
**Year:** 2025
**Venue:** arXiv 2503.22562
**URL:** https://arxiv.org/abs/2503.22562
**Raw-data source:** [[raw-data/niyama]]

---

## The siloing waste

Production deployments commonly split into separate fleets per latency tier:

```
   Fleet A: interactive chat  (provisioned for peak)
   Fleet B: nightly batch     (provisioned for peak)
```

Niyama measures the waste empirically: each fleet is sized for its own peak, so the combined fleet has ~2× the average load capacity needed. Utilization sits at 40–60% in each fleet.

**Niyama's claim**: co-run all classes on shared infrastructure with class-aware scheduling, recover the missing 40%.

---

## QoS classes (the load-bearing abstraction)

Each request carries a class:

| Class | TTFT SLO | TPOT SLO | Preemptible? | Chunk size |
|---|---|---|---|---|
| `interactive` | 500 ms | 50 ms | no | 256 |
| `standard` | 2 s | 100 ms | yes (between iters) | 512 |
| `batch` | 30 s+ | 1 s+ | yes (anywhere) | 2048 |

The chunk size column is the key knob: `batch` requests can run very large prefill chunks (high throughput per chunk) because their TPOT SLO is loose; `interactive` requests need small chunks so they don't stall co-batched decodes.

---

## Hybrid prioritization

```text
Each iteration:
    1. Reserve up to interactive_admit_ceiling (e.g., 50%) of batch slots for interactive.
    2. Within interactive: VTC across tenants.
    3. Fill remaining slots with standard (also VTC).
    4. Use any spare compute capacity (chunked-prefill style) for batch prefills.
    5. If interactive arrives mid-step and a standard request can be preempted, preempt at the next iteration boundary.
```

Preemption is cheap because it happens at iteration boundaries — no KV ship, just leave the request's state in the cache and resume later.

---

## Dynamic chunking — per-class chunk-size adaptation

A direct extension of [[sarathi-serve]]: each class has its own chunk size, and the scheduler can adapt chunk size within a class based on current load:

```text
chunk_size(class, load) =
    base_chunk[class] / (1 + α · interactive_pressure)
```

When interactive pressure is high, batch chunks shrink so decode tokens get more compute. When interactive pressure is low, batch chunks grow back to maximize throughput.

---

## Selective relegation under overload

When the system passes a load threshold:

```text
if overload_signal:
    1. Stop admitting new batch requests (queue them or reject).
    2. Demote in-flight batch to "background" — runs only on spare compute.
    3. Standard requests continue but with stricter chunk sizes.
    4. Interactive requests untouched.
    5. Only if interactive itself exceeds capacity: invoke per-tier admission control.
```

This is **graceful degradation**: latency-tolerant traffic absorbs the overload before latency-sensitive traffic is touched.

---

## Reported numbers

- **32% more capacity** at the same SLO-attainment vs siloed deployment.
- **50% fewer SLO violations** under extreme load (1.5× nominal).
- Interactive p95 TTFT held within SLO even when batch class is saturated.

---

## Connections

- [[ch-10]] — parent chapter; Niyama is the QoS layer.
- [[excerpts/vtc]] — composable: VTC within a class, Niyama across classes.
- [[sarathi-serve]] — chunk-size knob; Niyama makes it per-class and dynamic.
- [[excerpts/serving-optimization-foundations-2026]] — Niyama is the position paper's first concrete instantiation.
