---
chapter: ch-04
course: llm-inference
phase: read
excerpt_of: "Continuous Batching synthesis (Orca + vLLM + Anyscale + HF docs)"
source_url: https://www.anyscale.com/blog/continuous-batching-llm-inference
created_at: "2026-05-21"
---

# Excerpt: Continuous batching gains — the 2–5× number and where it holds

**Authors:** Synthesis across Orca (Yu et al. 2022), vLLM (Kwon et al. 2023), Anyscale blog (Aminabadi et al. 2023), HF Transformers docs
**Year:** 2022–present
**URLs:** https://www.anyscale.com/blog/continuous-batching-llm-inference ; https://arxiv.org/abs/2309.06180 ; https://huggingface.co/docs/transformers/main//continuous_batching
**Raw-data source:** [[raw-data/continuous-batching]]

---

## The headline numbers (production reports)

| Source | Comparison | Throughput gain |
|---|---|---|
| Orca paper (2022) | vs FasterTransformer on GPT-3 175B | >10× |
| vLLM paper (2023) | vs HF Transformers on Llama-7B/13B | up to 24× |
| vLLM paper (2023) | vs FasterTransformer on Llama-7B/13B | 2.2–3.5× |
| vLLM paper (2023) | vs TGI (which already had continuous batching) | 1.6–2.7× |
| Anyscale blog | Llama-7B on A100, ShareGPT workload | ~3× |
| HF continuous batching docs (2026) | Default vs `generate()` loop | 2–5× |

The "2–5×" headline that gets quoted in talks is the gain from continuous batching alone (Orca-style). The "20×" headline is continuous batching + paged KV + good kernels (the full vLLM stack) vs the naive HF baseline.

---

## Where the gain comes from — three mechanisms

### 1. Eliminating head-of-line blocking

Heterogeneous output lengths (chat: short answers + long stories + medium replies) mean static batching wastes most of the longest-request's lifetime. Continuous batching lets short requests finish at their natural latency.

Empirical impact: ~3× throughput improvement at workload heterogeneity typical of chat APIs.

### 2. Eliminating padding waste

Static batching pads all requests to max length for prefill. Continuous batching processes each prompt at its own length.

Empirical impact: 1.5–2× compute savings on heterogeneous prompts.

### 3. Maximizing instantaneous batch size

When a short request finishes, its KV is freed immediately and a waiting request can be admitted on the next iteration (~50 ms). The decode batch stays near the KV-budget maximum.

Empirical impact: GPU utilization 70%+ vs 15–30% for static.

---

## When the gain is small

Continuous batching does NOT help when:

| Scenario | Why |
|---|---|
| Uniform single-request workload | No heterogeneity to exploit; static and continuous are equivalent |
| Heavily prefill-bound (very long prompts, short outputs) | Most time is in prefill, not decode; scheduling doesn't matter |
| KV budget exhausted at 2–4 sequences | Can't admit new requests anyway; effective batch is small |
| Real-time low-latency single-stream | Latency-optimized, batch=1; no batching to do |

For these cases, other optimizations (FlashAttention prefill, chunked prefill, KV compression) matter more than scheduling.

---

## Anyscale's measurement methodology (the right way to benchmark)

The Anyscale blog formalized the canonical benchmarking setup that became the standard ([[raw-data/llmperf]]):

1. **Workload**: ShareGPT-style traces with realistic prompt/output length distributions (mean ~500 input, ~150 output tokens; high variance).
2. **Arrival pattern**: Poisson at rate λ (requests/second), sweep λ from low to high.
3. **Metrics**: TTFT median + p99, TPOT median + p99, throughput (tok/s), goodput at SLO.
4. **Saturation analysis**: find the QPS at which TTFT p99 violates SLO; report throughput at that point.

A common bad benchmark: measure throughput at QPS = ∞ (steady-state saturation). This number is decoupled from any SLO and overstates production capacity by 2–5×.

---

## Tuning for the gain

| Knob | Effect on continuous-batching gain |
|---|---|
| `max_num_seqs` higher | Larger steady-state batch, more amortization, but more preemption risk |
| `max_num_batched_tokens` higher | Faster TTFT but bigger TPOT blip when prefill admitted |
| `gpu_memory_utilization` higher | Bigger KV pool → bigger batch → more amortization |
| Enable prefix caching | Reduces prefill cost for shared system prompts; multiplies decode throughput gain |
| Enable chunked prefill | Smooths prefill-decode interference, recovers TPOT regularity |

A common production config (vLLM serving Llama-3-8B on A100):
```
--max-num-seqs 256
--max-num-batched-tokens 8192
--gpu-memory-utilization 0.90
--enable-prefix-caching
--enable-chunked-prefill
```

This stack reliably hits ~3000 tok/s decode throughput on a single A100 80GB at ShareGPT-like workloads.

---

## Common pitfalls

- **Measuring throughput at peak load without SLO**. Production capacity is throughput-at-SLO, not raw throughput.
- **Reporting tok/s without splitting prefill from decode**. They have totally different cost models.
- **Forgetting that gains scale with workload heterogeneity**. A homogeneous benchmark understates the win by 3×+.
- **Comparing against "HF Transformers `generate()`" alone**. That's not even dynamic batching — it's batch-of-one. The honest comparison is against TGI / DeepSpeed-FastGen / TensorRT-LLM in-flight batching.

---

## Connections

- [[excerpts/orca-iteration-scheduling]] — the founding algorithm.
- [[excerpts/vllm-scheduler-states]] — production implementation that delivers the headline numbers.
- [[raw-data/pagedattention]] — paged KV is the matched primitive that enables the >2× over plain Orca.
- [[ch-05]] — chunked prefill: how to keep the gains while also satisfying TPOT SLOs.
- [[ch-19]] — TTFT/TPOT/goodput benchmarking that makes "2–5×" mean something.
