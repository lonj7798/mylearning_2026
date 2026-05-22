---
chapter: ch-09
course: llm-inference
phase: read
excerpt_of: "DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving"
source_url: https://arxiv.org/abs/2401.09670
created_at: "2026-05-21"
---

# Excerpt: DistServe — goodput-optimized phase splitting

**Authors:** Yinmin Zhong, Shengyu Liu, Junda Chen, Jianbo Hu, Yibo Zhu, Xuanzhe Liu, Xin Jin, Hao Zhang
**Year:** 2024
**Venue:** OSDI 2024
**URL:** https://arxiv.org/abs/2401.09670
**Raw-data source:** [[raw-data/distserve]]

---

## The goodput redefinition (the load-bearing metric)

Conventional serving optimizes raw throughput `λ = requests/sec` or `tokens/sec`. DistServe argues the metric is wrong: a token delivered after its SLO is *not capacity*. It defines goodput as

```math
G(\lambda, T_{\text{ttft}}, T_{\text{tpot}}) = \lambda \cdot \Pr[\text{TTFT} \le T_{\text{ttft}} \land \text{TPOT} \le T_{\text{tpot}}]
```

The serving objective is `max λ s.t. G(λ, T_ttft, T_tpot) / λ ≥ q`, with `q` typically 0.9 or 0.99 (the SLO attainment target).

The redefinition matters because colocated systems can *increase* throughput while *decreasing* goodput — they bat through more requests by missing SLOs on a long tail of them. DistServe's evaluation always reports goodput, not throughput.

---

## Why colocation cannot meet both SLOs simultaneously

The paper's central measurement: under a Poisson arrival of mixed long-and-short prompts on a single colocated engine,

- pushing TTFT inside SLO requires giving prefill enough compute → starves decode → TPOT misses,
- pushing TPOT inside SLO requires giving decode the batch size it needs → blocks long prefills → TTFT misses.

There is no single configuration of `max_num_batched_tokens` that satisfies both at high load. The colocated engine is on a Pareto frontier; disaggregation moves the frontier outward.

---

## Phase-specific resource model

For each phase, DistServe builds an offline performance model:

- **Prefill latency**: `L_prefill(N_prompt, TP) ≈ α(TP) · N_prompt^β(TP)` — close to linear in prompt length, sublinear in TP (TP=4 is ~3× faster than TP=1 due to all-reduce overhead).
- **Decode TPOT**: `TPOT(batch, TP) ≈ γ(TP) + δ(TP) · batch` — slow growth with batch since decode is memory-bound; saturates around batch=64.

Given target SLOs, DistServe searches `(N_prefill, TP_prefill, N_decode, TP_decode, batch_decode)` to maximize goodput. The search is small (typically <100 configurations to enumerate) and offline.

Typical result for OPT-13B at 4k average prompt, 200-token average output:
- `TP_prefill = 4, N_prefill = 2`
- `TP_decode = 1, N_decode = 6, batch_decode = 64`
- Goodput at 4-second TTFT / 100 ms TPOT SLO: **4.48× the colocated baseline**.

---

## Placement-aware KV transfer

DistServe co-locates prefill→decode worker pairs on the same physical node when possible. The reason is the bandwidth gap:

- intra-node NVLink-Switch (H100): ~600 GB/s
- inter-node InfiniBand (NDR 400G): ~50 GB/s effective for a single flow

A 2.7 GiB KV cache (Llama-3-70B, 8k context) ships in ~5 ms intra-node, ~55 ms inter-node. For chat workloads with 200 ms TTFT budgets, inter-node placement is unacceptable.

DistServe's placement algorithm formulates the prefill→decode pairing as a bipartite matching problem with bandwidth-aware edge weights.

---

## Connections

- [[excerpts/splitwise]] — earlier workload characterization that motivates the same split.
- [[excerpts/mooncake]] — production extension with KV-store layer and predictive admission.
- [[ch-09]] — parent synthesis on disaggregation architectures.
- [[ch-10]] — goodput as the metric for admission control beyond DistServe.
