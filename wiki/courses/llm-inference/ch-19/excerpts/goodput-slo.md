---
chapter: ch-19
course: llm-inference
phase: read
excerpt_of: "Goodput under SLO — the SLO-aware capacity metric"
source_url: https://docs.vllm.ai/en/latest/contributing/benchmarks.html
created_at: "2026-05-21"
---

# Excerpt: Goodput@SLO — the metric that matches business value

**Authors:** vLLM project + LMSYS + academic serving literature
**Year:** 2023–2026
**URLs:** https://docs.vllm.ai/en/latest/contributing/benchmarks.html
**Raw-data source:** [[raw-data/goodput-slo]]

---

## The definition

```
A request is "good" iff it satisfies all SLO predicates:
    TTFT     < SLO_ttft
    AND TPOT < SLO_tpot
    AND e2e  < SLO_e2e
    AND no error

goodput     = good_requests / benchmark_duration         [req/sec]
goodput_tok = good_output_tokens / benchmark_duration    [tok/sec]
```

The metric counts *only* requests that landed inside the SLO. Requests that "completed" but at p99-blowup latency are excluded.

---

## Why throughput alone misleads

```
System A (max-throughput-tuned):
    Throughput @ saturation:    8000 tok/s
    p99 TTFT @ saturation:      25 seconds
    p99 TPOT @ saturation:      250 ms/token

System B (SLO-tuned):
    Throughput @ saturation:    5500 tok/s
    p99 TTFT @ saturation:      1.8 seconds
    p99 TPOT @ saturation:      45 ms/token

User SLO: TTFT < 2 s, TPOT < 100 ms

System A goodput @ SLO:   ~800 tok/s    (most requests violate TTFT)
System B goodput @ SLO:   ~5200 tok/s   (almost all requests pass)
```

System A's "throughput win" is fake. **Goodput is what counts.**

The intuition: in heavy-load regimes a serving system can accept and complete many requests by letting their tail latency explode. They complete *eventually* — counted in throughput, useless to users.

---

## The measurement loop

```python
def goodput_sweep(server, dataset, slo,
                  rates=[1, 2, 4, 8, 16, 32, 64, 128]):
    results = []
    for rate in rates:
        run = bench_serve(server, dataset, request_rate=rate, duration=300)
        good_reqs = [
            r for r in run.requests
            if r.success
               and r.ttft_ms < slo.ttft_ms
               and r.tpot_ms < slo.tpot_ms
               and r.e2e_ms  < slo.e2e_ms
        ]
        goodput = len(good_reqs) / run.duration_sec
        results.append({
            "offered_rate":  rate,
            "goodput":       goodput,
            "throughput":    run.throughput,
            "ttft_p99":      run.ttft_p99,
            "tpot_p99":      run.tpot_p99,
            "error_rate":    1.0 - run.success_rate,
        })
    return results
```

Plot offered_rate vs goodput. The curve shape:

```
goodput (good req/s)
       ↑
       |       ___________
       |      /           \___
       |     /                \___
       |    /                     \____
       |   /                           \___
       |  /                                \___
       | /                                     \_
       +─────────────────────────────────────────── offered rate
       0   1   2   4   8  16  32  64  128
                          ↑                ↑
                       knee (peak goodput)  collapse (everything misses SLO)
```

The **peak goodput** value is your system's real capacity. The corresponding offered rate is your safe operating point.

---

## Standard SLO presets

| Workload | TTFT SLO | TPOT SLO | Notes |
|----------|---------:|---------:|-------|
| Interactive chat | 1 s | 50 ms (≥20 tok/s) | Perception threshold |
| Voice assistant | 500 ms | 33 ms (≥30 tok/s) | TTS pipeline downstream |
| Code completion (Cursor / Copilot) | 200 ms | 25 ms (≥40 tok/s) | Tight feedback loop |
| Long-form RAG | 3 s | 100 ms | Long output, user tolerates initial wait |
| Tool-call agent | 2 s | 80 ms | Multi-step, each call adds |
| Batch summarization | none | none, e2e<5 min | No streaming SLO |
| MLPerf Server scenario | task-specific (e.g. 2 s, 100 ms) | per-task | Audited |

Set the SLO based on user research and product requirements, not benchmark convenience.

---

## Conjunctive vs disjunctive predicates

The standard form is **conjunctive** — request is good iff ALL predicates pass. This matches user perception: a fast first token with stuttering follow-up is bad; a snappy stream after a slow start is bad. Both must be acceptable.

A **disjunctive** definition ("either TTFT or TPOT is good") is weaker and reports inflated numbers. Don't use it unless you have a specific product justification.

---

## What goodput hides

Even goodput@SLO is one number. It doesn't capture:

- **Fairness across users.** Goodput can be high while one user starves.
- **Per-tier behavior.** Premium-tier vs free-tier need separate SLOs.
- **Cost.** A system meeting goodput at 8× the hardware cost isn't winning.

For full reporting, combine goodput with (throughput, p99 latencies, error rate, GPU-hour cost). See HELM for the multi-dimension reporting discipline.

---

## Connections

- [[excerpts/ttft-tpot-itl]] — the predicates goodput conjuncts.
- [[excerpts/vllm-benchmarks]] — vLLM's `benchmark_serving.py --goodput ttft:2000 tpot:50` flag.
- [[excerpts/admission-control-goodput]] (ch-10) — admission policies designed to maximize this exact metric.
- [[ch-19]] — parent synthesis.
