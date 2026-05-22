---
chapter: ch-16
course: llm-inference
phase: read
excerpt_of: "Synthesis of vLLM production tuning from docs + vllm-project benchmark guides"
source_url: synthesis
created_at: "2026-05-21"
---

# Excerpt: vLLM Production Knobs — The Four That Matter

**Sources synthesized:**
- vLLM serve args docs: https://docs.vllm.ai/en/stable/configuration/serve_args/
- vLLM benchmarks repository
- LMSYS / Anyscale production tuning blog posts
- Field reports from vLLM Discord

**Raw-data source:** [[raw-data/vllm-project]]

---

## The four knobs

There are ~50 CLI flags vLLM accepts. **Four** account for ~90% of production tuning impact:

1. `--gpu-memory-utilization` (default 0.9)
2. `--max-num-batched-tokens` (default depends on model)
3. `--max-num-seqs` (default 256)
4. `--enable-prefix-caching`

Everything else is downstream of these.

---

## 1. `--gpu-memory-utilization` (gmu)

What it does: fraction of GPU memory vLLM may use. After loading the model + reserving activation headroom, the *remaining* memory becomes the KV cache pool.

```
KV cache memory = (total GPU memory × gmu) - model weights - activation reserve
KV cache blocks = KV cache memory / block_size_bytes
Total KV tokens served = KV cache blocks × block_tokens
```

### Tuning rule

- Start at **0.92** for production. Default 0.9 is conservative.
- If you OOM under long-sequence load → drop to 0.85.
- If you have lots of headroom (`vllm_gpu_cache_usage` stays below 80%) → push to 0.95.
- **Don't go to 1.0**: vLLM needs a small reserve for kernel scratch.

### Per-model defaults

| Model | GPU | Suggested gmu |
|-------|-----|---------------|
| Llama-3-8B (TP=1) | H100 80GB | 0.92 |
| Llama-3-70B (TP=8) | 8×H100 80GB | 0.92 |
| Llama-3-70B (TP=4) | 4×H100 80GB | 0.90 (less headroom) |
| Mixtral-8x22B (TP=4) | 4×H100 80GB | 0.92 |
| DeepSeek-V3 (TP=8 EP=64) | 64×H100 | 0.88 (MoE has more activation variance) |

---

## 2. `--max-num-batched-tokens` (mnbt)

What it does: cap on total tokens (prefill + decode) per scheduler step. The "token budget" in the scheduler loop.

### Tuning rule

- **Chat-heavy workload** (mostly short decode, occasional moderate prefill): 2048 is fine.
- **Prefill-heavy workload** (RAG, long-context, summarization): 8192-16384.
- **Mixed**: 4096-8192.

Higher → better prefill throughput, but each scheduler step takes longer → decode TPOT degrades for already-running requests.

### Worked example

For a 4096 max-num-batched-tokens budget:
- 64 RUNNING decode requests × 1 token = 64 tokens used.
- Remaining: 4032 tokens.
- A new request with 5000-token prompt gets chunked: 4032 tokens this step, 968 next step.
- TTFT for the new request: 2 × step_latency.

If budget were 8192, same prompt fits in one chunk → TTFT = 1 × step_latency.

But the step is now larger → decode TPOT for the 64 RUNNING requests is higher.

---

## 3. `--max-num-seqs` (mns)

What it does: cap on concurrent RUNNING requests.

### Tuning rule

- Default 256 is rarely binding because KV cache typically runs out first.
- Lower this (e.g. 64) only if you want to *force* lower concurrency for latency reasons.
- Higher this only if you've verified KV cache has headroom.

### Diagnostic

Check `vllm_running_requests_count` vs `vllm_waiting_requests_count`:
- If running ≈ mns and waiting > 0: you're seq-bound; consider raising mns or scaling out.
- If running < mns and waiting > 0: you're KV-cache-bound; consider raising gmu or adding ranks.

---

## 4. `--enable-prefix-caching` (APC)

What it does: turns on Automatic Prefix Cache. Cache blocks across requests; reuse them via hash-chain matching.

### When to enable

- **Chat with system prompt**: 5-10× TTFT speedup. Always on.
- **RAG / long-document Q&A**: 3-6× TTFT speedup. Always on.
- **Agent / multi-turn**: 2-4× TTFT speedup. Always on.
- **Arbitrary user prompts (no sharing)**: ~5% overhead, ~1.5× TTFT speedup. Still on usually.

### Cost

APC adds ~5% per-step overhead from hash computation. The TTFT savings dwarf this in almost all real workloads.

### Default in V1

V1 default: **on**. V0 default was off. Confirm with `--no-enable-prefix-caching` to disable if needed.

---

## Other knobs worth knowing

### `--swap-space N`

CPU memory (GB per rank) reserved for KV swap on preemption. Default 4. Raise to 16-32 if you have lots of preemption and long prompts.

### `--block-size N`

KV cache block size in tokens. Default 16. Larger → less per-block overhead but more wasted bytes per request. Don't change unless you've profiled.

### `--enforce-eager`

Disables CUDA graphs. Only use for debugging — 30-50% decode-latency regression in prod.

### `--num-scheduler-steps N`

Multi-step scheduling: scheduler emits decisions for N steps at a time, model runner runs N steps before checking back. Reduces scheduler overhead at low concurrency.

Default 1. Raise to 4-8 for low-concurrency latency optimization (saves ~10% on TPOT at batch=1-4).

### `--speculative-config`

Speculative decoding config. See [[ch-14]], [[ch-15]].

### `--guided-decoding-backend {xgrammar,outlines}`

Structured output backend. Default xgrammar.

---

## Tuning workflow

```
1. Load the model with default flags. Measure TTFT, TPOT, throughput at your workload.
2. If KV cache utilization is < 70%: → raise --gpu-memory-utilization.
3. If TTFT is bad and prompts are long: → raise --max-num-batched-tokens.
4. If TPOT is bad and prompts are long: → consider lowering --max-num-batched-tokens
                                          or enabling chunked prefill explicitly.
5. If RUNNING ≈ mns and waiting backlog grows: → scale out or lower QPS.
6. Always: --enable-prefix-caching (unless you have measured it hurts your specific workload).
```

---

## Common production-tuning mistakes

- **Raising gmu without checking activation variance.** Long sequences spike activations; if you raised gmu to 0.97, you may OOM at p99.
- **Tuning mnbt without measuring chunked-prefill impact.** Lowering mnbt forces more chunking → higher TTFT for long prompts.
- **Enabling APC without checking hash overhead.** For workloads with zero shared prefixes (random user prompts), APC overhead doesn't get paid back.
- **Setting --enforce-eager in prod.** "Just in case" → 30-50% decode regression. Never.
- **Spec-dec at high concurrency.** Spec-dec helps only when target is bandwidth-bound (low-medium batch). At batch=64+, disable it; it adds overhead without saving time.

---

## Connections

- [[excerpts/vllm-scheduler]] — token budget and seq cap consumed here.
- [[excerpts/vllm-kv-cache-manager]] — gmu and prefix caching are KV cache settings.
- [[ch-21]] (lab) — production benchmarks that exercise these knobs.
- [[ch-16]] — parent chapter.
