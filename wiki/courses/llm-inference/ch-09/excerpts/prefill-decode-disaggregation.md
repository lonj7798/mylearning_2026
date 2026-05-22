---
chapter: ch-09
course: llm-inference
phase: read
excerpt_of: "Prefill-Decode Disaggregation — design-pattern synthesis"
source_url: synthesis card; references arXiv 2311.18677, 2401.09670, 2407.00079
created_at: "2026-05-21"
---

# Excerpt: Disaggregation as a serving design pattern

**Authors:** Synthesis card around Splitwise, DistServe, Mooncake, and vLLM disaggregated-prefill docs
**Year:** 2023–present
**URLs:** see ch-09 sources
**Raw-data source:** [[raw-data/prefill-decode-disaggregation]]

---

## The state boundary

Disaggregation splits the inference computation at the **post-prefill KV cache**. The transfer size, per request:

```math
\text{bytes}_{KV} = 2 \cdot L \cdot H_{kv} \cdot D_h \cdot N_{prompt} \cdot \text{dtype\_bytes}
```

Worked values (Llama-3-70B, `L=80, H_kv=8, D_h=128`, bf16):

| Prompt length | KV size | Transfer time @ 200 GB/s RDMA |
|---|---|---|
| 512 | 168 MiB | 0.8 ms |
| 4,096 | 1.34 GiB | 7 ms |
| 8,192 | 2.68 GiB | 13 ms |
| 32,768 | 10.7 GiB | 54 ms |
| 131,072 | 42.8 GiB | 214 ms |

The first three are negligible next to prefill compute; the last two are not.

---

## When disaggregation pays — the rule

Disaggregate iff the **phase-specific optimization gain** exceeds the **KV transfer + orchestration cost**:

```text
Gain   = (TTFT_colocated - TTFT_disagg) + α · (TPOT_colocated - TPOT_disagg)
Cost   = T_transfer + T_orchestration
Decide: disaggregate if Gain > Cost  AND  the colocated p99 violates SLO
```

The second condition matters: if colocated p99 is fine, don't add operational complexity. Disaggregation is a tax you only pay when the colocated frontier can't reach the SLO at acceptable cost.

---

## Optimizations that flatten the transfer tax

1. **Layer-wise streaming.** Start shipping layer-0 KV while prefill is still computing layer-1. Effective overlap ~70%. Without it, transfer is serialized after prefill.
2. **Same-node co-placement.** NVLink-Switch intra-node is ~600 GB/s, RDMA inter-node is ~200 GB/s. The placement scheduler maximizes intra-node prefill→decode pairs.
3. **In-transit compression.** Ship in fp8 (or int8), expand at decode. Halves the transfer time at <0.05% perplexity cost.
4. **Shared-prefix coalescing.** If 100 requests share a 2k-token system prompt, ship its KV once.

---

## What disaggregation does NOT solve

- **Variable output length.** A request with `max_tokens=2048` may finish in 50 tokens or run the full 2048; decode pool sizing still has the same prediction problem.
- **Tenant fairness.** Disaggregation says nothing about which user/tenant gets served first; that's the [[vtc]] / [[niyama]] axis (ch-10).
- **Overload behavior.** Without explicit admission control (Mooncake), a saturated decode pool still degrades all in-flight requests; disaggregation alone is not graceful under overload.

---

## Connections

- [[excerpts/distserve]] — goodput-optimized scheduler over the split.
- [[excerpts/splitwise]] — workload characterization + hardware-tier argument.
- [[excerpts/mooncake]] — production KV-store layer + early rejection.
- [[ch-09]] — parent synthesis.
- [[ch-05]] / [[sarathi-serve]] — the colocated alternative.
- [[ch-10]] / [[admission-control-goodput]] — what to add on top of disaggregation for graceful overload.
