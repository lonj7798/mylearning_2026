<!-- chapter: ch-03
     track: generation-foundations
     title: KV Cache Memory Formula + Prefill vs Decode Bottlenecks
     sources: [[kv-cache-memory-formula]], [[prefill-vs-decode]], [[batching-for-inference]]
     figures: figures/kv-cache-calculator.html
-->

# Chapter 3 — KV Cache Memory Formula + Prefill vs Decode

> **Core insight.** The KV cache size in bytes is `2 · L · H_kv · d_head · T · B · bytes_per_elem` — that single formula is the binding constraint on every serving deployment. Prefill is compute-bound (the full prompt's attention matrix runs once, arithmetic intensity ~1000 FLOPs/byte); decode is memory-bandwidth-bound (one query, but reads all cached KV every step, arithmetic intensity ~0.3 FLOPs/byte). The two phases sit on opposite sides of the GPU roofline and need fundamentally different schedules.
>
> **Guideline.** Before deploying: (1) compute KV bytes per request at your max context with the formula above, (2) subtract weight bytes + activation overhead from total HBM to get the KV budget, (3) divide to get your maximum batch size — that's the upper bound on concurrent requests. (4) Measure prefill TTFT and decode TPOT separately; if either one is your SLO bottleneck, the optimizations live in different chapters (ch-05 for prefill, ch-06/08 for decode).

---

## Why this chapter exists

The formula and the asymmetry are not optional knowledge for an LLM inference engineer — they're the *only* numbers you actually need to size a deployment. PagedAttention (ch-06), continuous batching (ch-04), chunked prefill (ch-05), disaggregation (ch-09), KV compression (ch-08) — every one of these is a targeted relaxation of either the formula or the prefill/decode imbalance. If you can derive both cold, you can read serving papers in their native language.

Three things you should walk away with:

1. The exact formula and how to plug in any open model's config to predict KV bytes — for Llama-3-70B at 8k context, the answer is ~10 GB *per request*.
2. Why prefill and decode have arithmetic intensities ~3 orders of magnitude apart, and what "roofline knee" means for each.
3. Static vs dynamic vs continuous batching as the three baseline schedules, and why continuous batching is the only one that doesn't waste 50%+ of GPU time at production traffic shapes.

Sources: [[kv-cache-memory-formula]], [[prefill-vs-decode]], [[batching-for-inference]] in the raw-data library.

---

## 1. The KV cache memory formula

For a single sequence at context length `T` in a decoder-only transformer:

```
KV bytes = 2 · L · H_kv · d_head · T · bytes_per_elem
```

| Symbol | Meaning |
|---|---|
| `2` | one for K, one for V |
| `L` | number of transformer layers (`n_layers` in config) |
| `H_kv` | number of KV heads (`num_key_value_heads` in config — see ch-02) |
| `d_head` | head dimension (`hidden_size / num_attention_heads`) |
| `T` | tokens cached so far (prompt + generated) |
| `bytes_per_elem` | 2 for bf16/fp16, 1 for int8/fp8, 0.5 for int4 KV |

For a batch of `B` independent sequences each at context `T`:

```
KV bytes total = 2 · L · B · T · H_kv · d_head · bytes_per_elem
```

This is what [[kv-cache-memory-formula]] tags as the operational formula every distributed-serving engineer must internalize.

---

## 2. Worked examples — three frontier models

### Llama-3-8B at 8k context, bf16

Config: `L=32, H_q=32, H_kv=8, d_head=128`.

```
per-token KV  =  2 · 32 · 8 · 128 · 2  =  131,072 bytes  ≈  128 KB
per-request   =  128 KB × 8192          =  1.0 GB
```

On an 80 GB H100, weights are ~16 GB (bf16) → ~60 GB available for KV → fits ~60 concurrent 8k-context requests.

### Llama-3-70B at 8k context, bf16

Config: `L=80, H_q=64, H_kv=8, d_head=128`.

```
per-token KV  =  2 · 80 · 8 · 128 · 2  =  327,680 bytes  ≈  320 KB
per-request   =  320 KB × 8192          ≈  2.5 GB
```

On 8×H100 (640 GB total) with weight bytes ~140 GB and TP=8 weight sharding overhead, KV budget is roughly `640 − 140 − 60 (overhead) = 440 GB`. That fits ~176 concurrent 8k-context requests across the 8 GPUs.

The same model at 128k context per request:
```
per-request   =  320 KB × 131072  ≈  40 GB per request
```
You're now at ~10 concurrent requests across all 8 GPUs. **One order of magnitude reduction in batch size from 16× context.**

### DeepSeek V3 671B with MLA, 32k context, fp8

DeepSeek's MLA compresses KV to ~70 KB / token at fp8 (vs ~600 KB for naive MHA-equivalent at this scale). At 32k:
```
per-request   =  70 KB × 32768  ≈  2.3 GB
```
This is *less* than Llama-3-70B at 8k! MLA + fp8 is what makes 671B serving practical.

---

## 3. Prefill is compute-bound; decode is bandwidth-bound

The single most important asymmetry in LLM inference. Let's quantify on H100 SXM (1979 TFLOPS bf16, 3.35 TB/s HBM):

**Roofline knee**: `1979e12 FLOPs / 3.35e12 B = 590 FLOPs/byte`. Operations above 590 FLOPs/byte are compute-bound; below 590 are bandwidth-bound.

### Prefill arithmetic intensity (Llama-3-70B, `L=4096, d=8192, B=1`)

```
FLOPs/layer:   ~7·B·L·d² + 2·B·L²·d
             =  7·4096·8192² + 2·4096²·8192
             ≈  1.9e12 + 1.1e12  =  3.0e12 FLOPs

Bytes/layer:   weights read + KV write
             ≈  7·d²·2 + 2·L·d_kv·2
             =  7·8192²·2 + 2·4096·1024·2
             ≈  9.4e8 + 1.7e7  ≈  9.6e8 bytes

Intensity:     3.0e12 / 9.6e8  ≈  3100 FLOPs/byte  →  COMPUTE-BOUND
```

Prefill operates near peak compute throughput. The optimization lever is FLOPs: smarter attention kernels (FlashAttention), fewer redundant passes (prefix caching), or fewer prefill tokens (chunked prefill).

### Decode arithmetic intensity (Llama-3-70B, `L=4096, d=8192, B=1`)

```
FLOPs/layer:   ~7·B·d² + 2·B·L·d
             =  7·8192² + 2·4096·8192
             ≈  4.7e8 + 6.7e7  ≈  5.4e8 FLOPs

Bytes/layer:   weights read + KV read
             ≈  14·d²·2 + 2·L·d_kv·2
             =  14·8192²·2 + 2·4096·1024·2
             ≈  1.9e9 + 1.7e7  ≈  1.9e9 bytes

Intensity:     5.4e8 / 1.9e9  ≈  0.28 FLOPs/byte  →  MEMORY-BOUND
```

Decode operates near peak bandwidth. The optimization lever is *bytes moved*: shrink KV (MQA/GQA, KV compression), batch up (amortize weight load), eliminate redundant passes (speculative decoding moves multiple tokens per weight load).

### The four-order-of-magnitude gap

| Phase | Intensity | Roofline regime | Optimization lever |
|---|---:|---|---|
| Prefill | ~3100 FLOPs/B | compute-bound | reduce FLOPs |
| Decode (B=1) | ~0.3 FLOPs/B | bandwidth-bound | reduce bytes; batch |
| Decode (B=32) | ~9 FLOPs/B | still bandwidth-bound | batch up further |
| Decode (B=512, short L) | ~150 FLOPs/B | mixed | depends on regime |

This is why every serving paper splits prefill from decode. Two physically different optimization problems, one shared GPU, and naive batching forces them to interfere ([[prefill-vs-decode]] is the SARATHI/DistServe formalization).

---

## 4. TTFT vs TPOT — the two latency metrics

These are the only two latency numbers most serving SLOs care about ([[prefill-vs-decode]]):

```
TTFT (time-to-first-token):  prefill latency + scheduler queueing
                             = time from request submitted to first delta emitted
                             = dominated by prefill FLOPs at long prompts

TPOT (time-per-output-token): median decode-step latency
                              = (1 / decode_throughput_at_current_batch)
                              = dominated by KV-bandwidth + weight-bandwidth
```

Production SLOs often look like: TTFT p99 < 1 s; TPOT median < 50 ms. The two metrics live on opposite sides of the roofline:

- **Improve TTFT**: faster prefill kernels (FlashAttention, ch-11), shorter prefill (prefix caching, ch-07), parallelize prefill (chunked prefill, ch-05), dedicate GPUs to prefill (disaggregation, ch-09).
- **Improve TPOT**: smaller KV (GQA, KV compression), larger batch (continuous batching, ch-04), fewer decode steps (speculative decoding, ch-14), CUDA Graphs (ch-12).

Optimizing one can hurt the other. Chunked prefill (ch-05) trades TTFT for TPOT — splitting a long prefill into chunks delays first-token by a few ms but stops it from blocking decodes for 100s of ms.

---

## 5. Per-fleet implications: KV is the budget

Treat KV memory as the canonical fleet capacity unit, not "requests" or "QPS".

```
fleet KV budget   =  num_GPUs · (HBM_per_GPU − weights_per_GPU − overhead_per_GPU)
batch capacity    =  fleet KV budget / (per_token_KV · avg_context_length)
peak QPS          =  batch capacity / avg_request_latency
```

For Llama-3-70B on 8×H100 with avg context 4k and avg latency 2s:
- Fleet KV: `8 · (80 − 18 − 10) = 416 GB`
- Per-request KV: `320 KB · 4096 = 1.25 GB`
- Batch capacity: `416 / 1.25 ≈ 333 concurrent requests`
- Peak QPS: `333 / 2 ≈ 167 req/s`

These back-of-envelope numbers are usually within 2× of measured production capacity. Reasoning about a fleet in any other unit (e.g. "we provisioned 50 instances") obscures the actual binding constraint.

**Long-context tax.** Doubling context doesn't double cost — it doubles KV per request, which can drop batch capacity by 2×, which doubles per-request latency at saturation, which can drop QPS by 4×. The cubic-ish scaling is why long-context serving is a totally different cost regime from short-context.

---

## 6. The three batching baselines

Every serving system implements one of three batching strategies. They are not interchangeable.

### Static batching (the worst baseline)

```
1. Accept N requests.
2. Pad all to max length.
3. Run prefill + N · max_decode forwards in lockstep.
4. Return all N responses simultaneously.
```

The pathology: if 9 requests finish at 100 tokens and 1 finishes at 1000 tokens, the 9 short ones consume KV cache and GPU time for 900 wasted decode steps. Throughput is wildly suboptimal for any heterogeneous workload.

Padding waste: if requests have lengths `[100, 200, 500, 1000]`, you pad to 1000 and waste `(1000-100) + (1000-200) + (1000-500) + 0 = 2200` token-positions of compute.

**Where you still see it**: offline batch inference where uniform workloads exist. Never in production serving.

### Dynamic batching

```
1. Maintain a queue.
2. Form a batch when (count ≥ N) OR (wait ≥ T_max).
3. Run prefill + decode for that batch.
4. Wait for entire batch to finish.
```

Same pathology as static; just better admission control. Marginally better because the wait timeout limits worst-case latency.

### Continuous (iteration-level) batching — Orca 2022

```
loop forever:
    swap out finished sequences
    swap in waiting sequences (subject to KV budget)
    run ONE forward pass for the current active set
    sample next token for each active sequence
    emit deltas to streams
```

The batch is reconstituted at every decode step. No padding (sequences run at their own lengths). No head-of-line blocking (short requests finish whenever they finish). New requests join within one decode step latency (~50 ms). Reported 2–5× throughput improvement vs static batching at the same TPOT.

This is the entirety of ch-04. PagedAttention (ch-06) and continuous batching are the two innovations that turned LLM serving from a 10% utilization toy into a 60–80% utilization production system.

---

## 7. Cheat-sheet

```
KV CACHE FORMULA:
  bytes  =  2 · L · H_kv · d_head · T · B · bytes_per_elem
         =  per_token_KV · context · batch_size

LLAMA-3 KV PER TOKEN (bf16):
  8B:   128 KB
  70B:  320 KB
  405B: 800 KB
  (DeepSeek V3 with MLA + fp8: ~70 KB/token — 4–5× cheaper)

ROOFLINE (H100 SXM):
  peak compute (bf16):  1979 TFLOPS
  HBM bandwidth:        3.35 TB/s
  knee:                 590 FLOPs/byte

PREFILL vs DECODE:
                  prefill    decode (B=1)
  intensity       ~3100      ~0.3   FLOPs/byte
  regime          compute    bandwidth
  metric          TTFT       TPOT
  optimization    kernels    smaller KV + batch up

BATCHING:
  static       → 30–50% util, pathological at heterogeneous load
  dynamic      → same shape, better admission
  continuous   → 60–80% util; rebuilds batch at every decode step

CAPACITY ESTIMATE (fleet):
  batch_cap = (HBM_total − weights − overhead) / (per_token_KV · avg_ctx)
  peak_QPS  = batch_cap / avg_latency
```

---

## Connections and what's next

- **[[continuous-batching]] / ch-04** — the iteration-level scheduling pattern that turns the formula into actual fleet capacity.
- **[[sarathi-serve]] / ch-05** — chunked prefill: mix prefill chunks with decode tokens in one forward pass, smoothing the TTFT/TPOT tradeoff.
- **[[pagedattention]] / ch-06** — block-allocate KV cache to eliminate ~40% fragmentation waste; vLLM's load-bearing innovation.
- **[[h2o]] / ch-08** — KV compression: the per-token cost has a *third* factor (sparsity) we can exploit; H2O keeps only the heavy hitters.
- **[[distserve]] / ch-09** — disaggregate prefill and decode onto separate GPU pools so they stop interfering at the scheduler level.
- **[[flashattention]] / ch-11** — kills the `O(L²)` attention-matrix materialization that dominates long-prefill memory.
- **[[mha-mqa-gqa]] / ch-02** — the only knob that directly shrinks `H_kv` in the formula; GQA-8 is the universal default.

## Further reading

- [[kv-cache-memory-formula]] — the canonical synthesis card.
- [[prefill-vs-decode]] — SARATHI and DistServe framings.
- [[batching-for-inference]] — Orca, vLLM, HF docs as a single survey.
- [[attention-complexity]] — Tay et al. 2020; the `O(L²)` foundation that drives the prefill arithmetic intensity.

## Companion visualization

**[figures/kv-cache-calculator.html](figures/kv-cache-calculator.html)** — interactive calculator: choose model (Llama-3-8B/70B/405B, Qwen-3-32B, Mistral-7B), context length, dtype, batch size; outputs per-request KV, per-fleet KV at 8×H100, and predicted max concurrency. Compares MHA / GQA-{4,8} / MQA / MLA variants side-by-side. Use it to internalize how each design knob moves the binding constraint.
