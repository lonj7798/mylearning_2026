---
chapter: ch-09
course: llm-inference
phase: read
excerpt_of: "Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving"
source_url: https://arxiv.org/abs/2407.00079
created_at: "2026-05-21"
---

# Excerpt: Mooncake — KV cache as a fleet-wide distributed resource

**Authors:** Ruoyu Qin, Zheming Li, Weiran He, Mingxing Zhang, Yongwei Wu, Weimin Zheng, Xinran Xu (Moonshot AI + Tsinghua)
**Year:** 2024
**Venue:** arXiv 2024 (production at Kimi)
**URL:** https://arxiv.org/abs/2407.00079
**Raw-data source:** [[raw-data/mooncake]]

---

## The KV-store layer (the load-bearing architectural idea)

Mooncake takes the DistServe disaggregation pattern and inserts a **disaggregated KV-cache store** between the prefill and decode pools:

```
Prefill workers ──► KV-store layer ──► Decode workers

KV-store tiers (per node):
   HBM   : ~30 GB (hot, sub-ms access)
   DRAM  : ~512 GB (warm, ~10 μs access via RDMA)
   NVMe  : ~8 TB (cold, ~100 μs access)
```

The store is fleet-wide and addressable. A request's prefix can be cached *somewhere* across the fleet, and the scheduler can either pull it to the decode worker or schedule decode where the cache lives.

This is the architectural extension that makes long-context economical. At 200k context, recomputing the prefill on a miss costs ~30s of H100 time; restoring from CPU-DRAM cache costs <1s. CPU DRAM is essentially free relative to those numbers.

---

## KV-centric scheduling

Conventional load balancers minimize **worker load**. Mooncake's scheduler minimizes **expected end-to-end latency under cache-aware routing**:

```text
For each incoming request R with prompt P:
    candidate_workers = workers_with_capacity()
    for w in candidate_workers:
        cache_hit = longest_prefix_match(P, w.local_cache + w.reachable_remote_cache)
        cost_w    = ttft_estimate(P, cache_hit, w.load, network_to(w))
    schedule R on argmin_w(cost_w)
```

The trick is `reachable_remote_cache`: a decode worker can pull cache from another worker's DRAM in ~50 ms for a multi-GiB prefix — usually cheaper than re-prefilling.

---

## Prediction-based early rejection (the overload story)

Mooncake explicitly handles overload — the regime that DistServe and Splitwise assume away. Under load, Mooncake **predicts whether a new request will meet SLO** given the current state:

```text
For each incoming request:
    est_TTFT = queue_delay + prefill_time(P, current_TP)
    est_TPOT = decode_time(batch_decode_now, KV_pressure_now)
    if est_TTFT > SLO_TTFT or est_TPOT > SLO_TPOT:
        reject(R)  # early — don't spend compute on a doomed request
    else:
        admit(R)
```

The intuition: in overload, every admitted request degrades the next admitted request. Better to fail fast on requests you can't serve than to thrash on all of them. Mooncake reports ~50% more requests *meeting SLO* under 1.5× overload by rejecting ~30% of arrivals early.

This is the production-grade version of [[admission-control-goodput]] (ch-10).

---

## Reported numbers

- **Up to 525% more requests handled** under SLO vs colocated baselines on long-context simulated workloads.
- **75% more requests during peak load** in production at Kimi vs the prior architecture.
- Long-context (≥32k) is where the gap widens — the cache-store layer pays for itself most at long context.

---

## Connections

- [[excerpts/distserve]] — direct architectural ancestor (phase split + KV ship).
- [[excerpts/splitwise]] — workload-characterization motivation.
- [[ch-09]] — parent synthesis on disaggregation.
- [[ch-10]] — admission control / goodput; Mooncake's early rejection is one instantiation.
- [[ch-07]] / [[sglang-radixattention]] — the GPU-local prefix cache that Mooncake generalizes to fleet scale.
- [[ch-20]] — DeepSeek V3 MLA shrinks KV by ~10×, changing the KV-store economics.
