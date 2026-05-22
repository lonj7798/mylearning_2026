<!-- chapter: ch-09
     phase: serving-architecture
     title: Prefill/Decode Disaggregation
     sources: [[distserve]], [[splitwise]], [[mooncake]], [[prefill-decode-disaggregation]]
     back: [[sarathi-serve]] (ch-05), [[continuous-batching]] (ch-04)
     forward: [[admission-control-goodput]] (ch-10), framework PD-disagg (ch-16/17)
-->

# Chapter 9 — Prefill/Decode Disaggregation (DistServe / Splitwise / Mooncake)

> **Core insight.** Even with chunked prefill ([[sarathi-serve]], ch-05) holding the colocated case together, prefill and decode want *different machines*. They have different ideal batch sizes, different ideal memory/compute mixes, different SLO targets, and — once load grows or contexts get long — different ideal hardware. Disaggregation pulls them onto separate GPU pools and ships the KV cache between them; once you've paid the KV transfer tax, you can independently optimize TTFT and TPOT, and your goodput-per-dollar improves anywhere from 2× (DistServe) to 5×+ in long-context production (Mooncake).
>
> **Guideline.** Disaggregate when (a) your workload mixes long prompts and many short decodes, (b) TTFT and TPOT SLOs cannot both be met by a single colocated engine, and (c) you have RDMA / NVLink between the pools. Stay colocated with Sarathi-style chunked prefill when prompts are short and uniform, when load is low, or when interconnect is weak enough that KV transfer eats the gain.

---

## Why this chapter exists

By chapter 5 we had a serving stack that could handle a mixed prefill/decode workload on one engine: continuous batching ([[continuous-batching]]) at iteration granularity, chunked prefill ([[sarathi-serve]]) to split big prompts into scheduler-visible chunks. That gets you a long way.

It is not enough at production scale. Three pressures break the colocated model:

1. **Different ideal batch sizes.** Prefill saturates an H100's tensor cores at batch 1–4 (because every prompt token contributes work). Decode needs batch 32–64 to amortize weight reads from HBM. One engine cannot run both batch sizes simultaneously.
2. **Different SLO targets.** TTFT (time-to-first-token) is a prefill metric and is dominated by tail prompt length. TPOT (time-per-output-token) is a decode metric and is dominated by KV-cache size and memory bandwidth. Optimizing one in a single colocated engine compresses headroom on the other.
3. **Different hardware sweet spots.** Decode is memory-bandwidth-bound — older or cheaper GPUs with strong memory subsystems can serve decode at near-H100 throughput. Prefill is compute-bound — it wants the freshest tensor cores. A colocated fleet has to choose one hardware tier; a disaggregated fleet can mix.

Three papers, in chronological order, drive the field:

- **Splitwise** (Patel et al., MSR, Nov 2023, [[splitwise]]) characterizes the prefill/decode asymmetry and proposes phase splitting across hardware tiers.
- **DistServe** (Zhong et al., PKU/UCSD, Jan 2024, [[distserve]]) operationalizes the split as a goodput-optimized scheduler with phase-specific parallelism.
- **Mooncake** (Moonshot AI, Jul 2024, [[mooncake]]) takes it to long-context production: treats KV cache as a first-class distributed resource (GPU → CPU → SSD tiers) and adds prediction-based early rejection under overload.

This chapter builds the architecture from first principles, then walks each system's contribution, then gives the decision rubric.

---

## 1. Why prefill and decode interfere — even with chunked prefill

Recall from ch-05 that chunked prefill ([[sarathi-serve]]) splits a long prompt into 512- or 1024-token chunks and interleaves them with decode steps in the same forward pass. This works — and it is the right answer at *small* scale. It still has three structural limits:

**(a) Shared GPU = shared batch size.** A single forward pass has one batch size. Sarathi-Serve uses the chunked-prefill tokens to "fill" the spare compute the decode batch leaves on the table. If your prefill chunks are too small to saturate the tensor cores, you waste compute. If they're too big, decode TPOT spikes. There is exactly one knob (`max_num_batched_tokens`), and there is no setting that is right for *both* a long-prompt-heavy minute and a chat-heavy minute.

**(b) Decode-side memory pressure leaks into prefill capacity.** Every decoding request occupies KV-cache blocks proportional to its context length. With Llama-3-70B at 8k context, one decoding request holds ~10 GB of KV (see [[kv-cache-memory-formula]]). On an 80 GB H100, after weights, you have ~30 GB for KV — about 3 concurrent long-context decodes. There is no KV left for a long prompt's prefill state, so prefills queue and TTFT inflates.

**(c) Phase-specific parallelism conflicts.** Prefill benefits from high tensor parallelism (split one matmul across 8 GPUs to make the big prompt fast). Decode benefits from low tensor parallelism (avoid all-reduce overhead per token; instead, *replicate* the model across GPUs and run many concurrent decode batches). A colocated cluster picks one TP degree and lives with the cost in the other phase.

The conclusion: chunked prefill makes colocated serving *acceptable*. It does not make it *optimal*. Once your QPS or your context length crosses a threshold, the only way out is two separate pools.

See [[excerpts/splitwise]] for the workload-characterization measurements behind these claims.

---

## 2. Disaggregation architecture (the canonical picture)

```
                ┌─────────────────────────┐
                │      router / LB        │
                │  classifies + admits    │
                └─────────────┬───────────┘
                              │ request (prompt + sampling params)
                              ▼
        ┌─────────────────────────────────────┐
        │       PREFILL POOL (e.g., H100×N)   │
        │   - high TP                          │
        │   - small batch (1–4 prompts)        │
        │   - compute-bound                    │
        │   - output: KV cache + first token   │
        └────────────────┬────────────────────┘
                         │ KV transfer
                         │ (RDMA / NVLink-network /
                         │  cache store)
                         ▼
        ┌─────────────────────────────────────┐
        │       DECODE POOL (e.g., A100×M)    │
        │   - low TP, replicated              │
        │   - large batch (32–64 sequences)   │
        │   - memory-bandwidth-bound          │
        │   - emits stream of tokens          │
        └────────────────┬────────────────────┘
                         │ token stream
                         ▼
                       client
```

The state boundary is **the KV cache after prefill**. Its size, per request, is:

```
bytes_KV = 2 · layers · kv_heads · head_dim · prompt_tokens · dtype_bytes
```

For Llama-3-70B (`layers=80`, `kv_heads=8`, `head_dim=128`, bf16):

```
bytes_KV / token = 2 · 80 · 8 · 128 · 2 = 327,680 B ≈ 0.31 MiB/token
```

A 4k-token prompt → ~1.25 GiB KV; a 32k-token prompt → ~10 GiB. **That is the per-request cost of disaggregation.** Anything you save by phase-specific optimization must beat this transfer time. With 200 GB/s RDMA, 10 GiB ships in ~50 ms — fast next to a 1-second decode budget for a long generation, slow next to a 50 ms chat reply.

This is why every disaggregation system, without exception, talks at length about **interconnect and placement**.

---

## 3. DistServe — TTFT and TPOT, optimized independently

[[distserve]] (Zhong et al., OSDI 2024, arXiv 2401.09670) is the canonical "phase splitting as a goodput-optimized scheduler" paper. Its three contributions:

**(a) The goodput objective.** DistServe reframes serving capacity. Instead of measuring tokens/sec, it measures **request rate that satisfies both a TTFT SLO and a TPOT SLO**. A token that arrives 100 ms late is *not capacity* if the SLO was 200 ms. This is the same redefinition that motivates [[admission-control-goodput]] (ch-10).

**(b) Phase-specific resource search.** Given a TTFT SLO `T_ttft` and a TPOT SLO `T_tpot`, DistServe searches over `(N_prefill, TP_prefill, N_decode, TP_decode, batch_decode)` to maximize sustained QPS. The search uses a per-phase performance model derived from offline profiling: prefill latency as a function of TP and prompt length, decode TPOT as a function of TP and batch size.

The result is qualitatively that **prefill TP is high (4 or 8) and decode TP is low (1 or 2)**. Decode replicates the model across GPUs because per-token all-reduce overhead is brutal: an 8-way TP all-reduce takes ~50 μs at H100/NVLink-Switch speeds, and decode latency budget is ~10–30 ms per token. Multiply by 80 layers and you've burned the whole TPOT budget on collectives.

**(c) Placement-aware KV transfer.** Prefill and decode workers are placed to minimize cross-NIC KV traffic. On a node with 8 H100s and NVLink-Switch, intra-node KV transfer can be 600 GB/s; inter-node over RDMA is closer to 200–400 GB/s. DistServe co-locates prefill→decode pairs on the same node when possible.

**Numbers.** DistServe reports up to **4.48× higher request rate** at the same SLO vs colocated baselines on OPT-13B/66B and Llama-2-7B/13B, and a **10.2× tighter SLO at the same request rate**. The big wins are on long-prompt workloads (where prefill dominates) and on mixed workloads (where colocated systems thrash).

See [[excerpts/distserve]] for the goodput equation and the phase-specific resource model.

---

## 4. Splitwise — characterization first, hardware specialization second

[[splitwise]] (Patel et al., MSR, ISCA 2024, arXiv 2311.18677) came first and is the more measurement-heavy paper. Its main intellectual contribution is the **profile** of prefill vs decode on production traces.

Key measurements:

- **Prefill saturates an A100 at batch size 1** for prompts > 1024 tokens.
- **Decode at batch size 1 uses ~10% of A100 compute** — the rest is idle waiting for HBM.
- **Power draw**: prefill at ~360 W, decode at ~250 W on the same A100. Decode is memory-bound and the cores are underclocked.
- **Per-token cost**: prefill is cheaper per token (~1 ms/token at decent batch), decode is far more expensive per token (~20 ms/token because each step pays the full weight-read cost).

The architectural conclusion: not just split *across machines*, but split *across hardware tiers*. Run prefill on H100s where compute matters; run decode on A100s (or even older GPUs with strong memory) where memory bandwidth is what you need. This is the **hardware-specialized disaggregation** thesis.

Splitwise also introduces a **mixed pool** as a hedge: if load on one side spikes, machines in the mixed pool can swing to prefill or decode duty. This is critical operationally — pure-disaggregated fleets stranded capacity 20–40% of the time in the original measurements.

Numbers (Splitwise):
- 1.4× higher throughput at the same cost vs Llama-2-70B colocated on H100.
- 2.35× more requests/sec at the same cost using mixed H100/A100 tiers.
- 25% lower power for the same throughput.

The Splitwise paper is your reference for **why** you'd disaggregate; DistServe is your reference for **how to schedule** the result.

See [[excerpts/splitwise]] for the workload-characterization table.

---

## 5. Mooncake — KV cache as a first-class distributed resource

[[mooncake]] (Qin et al., Moonshot AI, arXiv 2407.00079) describes the production serving stack behind Kimi. The setting is long-context (often 32k–200k tokens) production traffic — the regime where KV cache *dominates* total memory and KV transfer dominates the disaggregation cost.

Mooncake's three additions beyond DistServe:

**(a) Disaggregated KV-cache store.** Instead of just shipping KV from prefill→decode worker, Mooncake routes KV into a global cache layer that lives in underutilized CPU DRAM and SSDs across the fleet. Per-token KV at long context is small enough (~0.3 MiB/token at Llama-3-70B) that even SSD-resident KV is competitive with re-prefilling when prompts are reused.

```
   Prefill ──► KV-store layer ──► Decode
                  │
                  ├─ GPU HBM (hot, ~30 GB/worker)
                  ├─ CPU DRAM (warm, ~512 GB/worker)
                  └─ NVMe SSD (cold, ~8 TB/worker)
```

The KV cache becomes a **first-class object the scheduler can route, replicate, and evict** — not a private GPU-only buffer. Prefix-cache hits (see ch-07's [[sglang-radixattention]] for the GPU-local version) extend across the entire fleet.

**(b) KV-centric scheduler.** Scheduling decisions are made by reasoning about **where the cache lives**, not where compute is cheapest. If a request's prefix is already cached on node A's CPU DRAM, run its decode there (or pull the cache to the nearest decode worker over RDMA — whichever is cheaper). Conventional load balancers pick the least-loaded worker; Mooncake's picks the cache-closest worker.

**(c) Prediction-based early rejection.** Under overload, Mooncake predicts whether a newly arriving request can meet its TTFT/TPOT SLO given the current state. If the answer is "no with high probability", reject the request immediately. This is the production version of the [[admission-control-goodput]] (ch-10) pattern.

Numbers (Mooncake): on simulated production traces, **up to 525% more requests handled** under the same SLO vs colocated baselines, with the gap widening at longer context. In production at Kimi, Mooncake handles 75% more requests during peak load than the prior architecture.

The Mooncake paper is essential because it is the first system to operate disaggregation at the **scale and context length** where the KV cache itself is the dominant operational concern.

See [[excerpts/mooncake]] for the KV-store layer description and the early-rejection logic.

---

## 6. KV transfer in detail — the tax you pay

The disaggregation cost is dominated by per-request KV ship time. The arithmetic is:

```
T_transfer = bytes_KV / interconnect_bw

           = (2 · L · H_kv · D_h · N_prompt · 2) / BW    [bf16]
```

Worked example: Llama-3-70B (`L=80, H_kv=8, D_h=128`), 8k prompt, RDMA at 200 GB/s:

```
bytes_KV    = 2 · 80 · 8 · 128 · 8192 · 2  = 2.68 GiB
T_transfer  = 2.68 GiB / 200 GB/s          ≈ 13 ms
```

13 ms is small next to the prefill compute time for 8k tokens (~150 ms on a 4-way TP H100 group) and tiny next to the multi-second decode budget. **The tax pays for itself easily at long contexts.**

For short contexts the math flips. A 256-token chat prompt:

```
bytes_KV    = 2 · 80 · 8 · 128 · 256 · 2   = 84 MiB
T_transfer  = 84 MiB / 200 GB/s            ≈ 0.4 ms
```

Still fine — but the prefill itself is only ~5 ms. The transfer is now 8% of prefill time, and you may have been better off colocated.

**The transfer optimizations that matter in practice**:

- **Layer-wise streaming.** Don't wait for prefill to finish to start shipping. As each layer's KV is computed, ship it; decode can start once layer-0 KV arrives. DistServe and Mooncake both implement this. Effective transfer-prefill overlap is ~70%.
- **Same-node co-placement.** Intra-node NVLink is ~600 GB/s; inter-node RDMA is ~200 GB/s. A 3× difference. Schedule prefill/decode pairs same-node when possible.
- **Compression / quantization in transit.** Mooncake ships fp8 KV across slow links and decompresses at the decode side; saves ~50% of bandwidth.
- **KV-store coalescing.** If 100 requests share the same system prompt, ship its KV once and broadcast.

---

## 7. When disaggregation pays — and when it doesn't

The decision rubric:

| Workload signal | Disaggregate? | Why |
|---|---|---|
| Long prompts (≥4k) + many short outputs | **yes** | Prefill cost dominates; phase-specific TP wins big |
| Mixed long+short prompts under high QPS | **yes** | Colocated thrashes between batch-size regimes |
| Tight TTFT *and* tight TPOT SLOs | **yes** | Can't optimize both in one engine |
| Production long-context (32k+) with shared prefixes | **yes (Mooncake-style)** | Cache layer becomes a fleet-wide asset |
| Uniform short prompts (<512 tokens) at moderate QPS | **no** | Transfer tax not worth it; chunked prefill is fine |
| Weak interconnect (1× 100 GbE) | **no** | RDMA matters; without it, transfer eats the gain |
| Low QPS (<2× single-engine capacity) | **no** | One Sarathi-Serve engine handles it without complexity |
| You don't have homogeneous high-bandwidth networking | **case by case** | Mooncake-style cache tiers help; DistServe-style direct ship breaks |

The cleaner heuristic: **start colocated with chunked prefill. Move to disaggregation when your TTFT p99 or TPOT p99 SLO is consistently violated and adding GPUs to the colocated pool no longer fixes it.** That's the moment the phase asymmetry has outrun the engine's ability to absorb it.

---

## 8. Side-by-side comparison

| Property | DistServe | Splitwise | Mooncake |
|---|---|---|---|
| Year / venue | OSDI 2024 | ISCA 2024 | arXiv 2407, production at Kimi |
| Primary objective | goodput under TTFT+TPOT SLO | cost/throughput, energy | long-context production SLO + overload |
| KV transfer path | direct prefill→decode RDMA | direct, mixed pool | global KV store (GPU/CPU/SSD tiers) |
| Phase-specific TP | yes, searched | yes, manual | yes, plus per-tenant policies |
| Admission / rejection | implicit (queue limits) | none discussed | **explicit predictive early rejection** |
| Hardware split | homogeneous | **heterogeneous (H100 prefill, A100 decode)** | homogeneous + tiered storage |
| Best for | medium-context, throughput-critical | cost-optimized fleet design | long-context production with overload |
| Headline number | 4.48× req rate at SLO | 1.4× throughput, 25% less power | 525% req rate (long-ctx sim) |

---

## 9. Practitioner's deployment recipe

```
# Decision tree for a new disaggregated deployment

1. Profile your colocated baseline first.
   - Measure TTFT p50/p95/p99, TPOT p50/p95/p99 at target QPS.
   - If both p99s are under SLO, do NOT disaggregate.

2. Identify the bottleneck phase.
   - TTFT misses → prefill-side under-provisioned.
   - TPOT misses → decode-side under-provisioned (often KV memory exhausted).

3. Provision pools with phase-specific parallelism.
   - Prefill: TP=4 or 8; small batch; aim for compute saturation.
   - Decode: TP=1 or 2 (replicated); batch=32–64; aim for memory-bw saturation.
   - Ratio (prefill:decode workers) ≈ avg_prompt_len / (avg_output_len · TP_ratio).
   - Typical: 1 prefill worker per 2–4 decode workers for chat-style workloads.

4. Choose transfer mode.
   - Same node: NVLink direct, layer-wise streaming.
   - Different node: RDMA (RoCE or InfiniBand), layer-wise streaming.
   - Long-context production: KV-store layer (Mooncake) with CPU/SSD tiers.

5. Implement admission control (always).
   - Even before SLO violations, cap concurrent prefills and concurrent KV-residency.
   - Under overload, reject — do not queue indefinitely (see ch-10).

6. Monitor the new failure modes:
   - KV-transfer queue depth (saturated interconnect).
   - Decode-pool KV exhaustion (forces preemption).
   - Prefill-pool head-of-line (long prompt blocks subsequent prefills).
```

---

## Connections and what's next

- **Back to [[sarathi-serve]] (ch-05)** — disaggregation is what you reach for when chunked prefill is no longer enough; the two are not mutually exclusive (you can run chunked prefill *inside* a disaggregated prefill pool).
- **Back to [[continuous-batching]] (ch-04)** — both pools still run iteration-level scheduling; disaggregation is a topology change, not a scheduler replacement.
- **Back to [[kv-cache-memory-formula]] (ch-03)** — the formula `2·L·H_kv·D_h·N·dtype_bytes` is the same one driving KV transfer cost.
- **Forward to [[admission-control-goodput]] (ch-10)** — Mooncake's early rejection is one instance; goodput is the metric disaggregation tries to maximize.
- **Forward to [[vllm-disaggregated-prefill-2026]] / vLLM (ch-16)** — vLLM ships a disaggregated-prefill mode in production releases; SGLang has PD-disaggregation in ch-17.
- **Forward to ch-20** — DeepSeek V3's MLA (Multi-head Latent Attention) compresses KV by ~10×; this changes the disaggregation math by making the transfer tax much smaller.

## Further reading

- [[distserve]] — Zhong et al. 2024; OSDI; the goodput-optimized scheduler.
- [[splitwise]] — Patel et al. 2024; ISCA; the workload characterization that started it.
- [[mooncake]] — Qin et al. 2024; the Kimi production KV-centric architecture.
- [[prefill-decode-disaggregation]] — the synthesis card across all three.
- [[sarathi-serve]] — chunked prefill, the colocated alternative that disaggregation eventually outgrows.

## Companion visualization

**[figures/disaggregation-architecture.html](figures/disaggregation-architecture.html)** — interactive comparison of colocated vs disaggregated topologies. Sliders for prompt length, output length, and QPS show when the KV-transfer tax is recovered by phase-specific batching wins.
