---
chapter: ch-21
course: llm-inference
phase: read
excerpt_of: "Required ablation design: prefix cache, max_num_batched_tokens, chunk size"
source_url: https://docs.vllm.ai/en/stable/configuration/serve_args/
created_at: "2026-05-21"
---

# Excerpt: Three ablation cells with expected shapes

**Source:** vLLM 0.6 serve args + SGLang 0.3 launch flags
**Raw-data source:** [[raw-data/vllm-scheduler]], [[raw-data/sglang-radixattention]], [[raw-data/sharegpt-workload]]

---

## Why these three and not others

The full vLLM / SGLang knob space has ~50 dials. The three picked for this lab are the ones that:
1. Are *expected* to move the latency / throughput numbers visibly.
2. Map to concepts from earlier chapters ([[ch-05]] chunked prefill, [[ch-07]] prefix caching).
3. Behave differently across the two frameworks, so the comparison is meaningful.

Other interesting knobs (CUDA graph capture sizes, kernel backend, speculative decoding) are out of scope.

---

## Ablation (a) — prefix caching on / off

**Hypothesis.** ShareGPT chat templates produce a small but non-zero shared prefix (`<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant...`). With prefix caching off, every request re-prefills this. With it on, only the first request pays the prefill cost.

**Run config.** Same baseline launch as [[excerpts/launch-cheatsheet]], with one flag flipped:

```bash
# vLLM: drop --enable-prefix-caching
vllm serve ... --max-num-batched-tokens 8192 --enable-chunked-prefill
# (no --enable-prefix-caching)

# SGLang: add --disable-radix-cache
python -m sglang.launch_server ... --chunked-prefill-size 8192 --disable-radix-cache
```

Sweep request rate as in the baseline. Report TTFT p50, TTFT p99, throughput, goodput at rate 16 (near-knee).

**Expected shape.**

| Cell | TTFT p99 | Throughput | Goodput |
|------|---------:|-----------:|--------:|
| vLLM, APC off | ~600 ms | 1.0× | 1.0× |
| vLLM, APC on  | ~450 ms | 1.05× | 1.10× |
| SGLang, Radix off | ~580 ms | 1.0× | 1.0× |
| SGLang, Radix on | ~380 ms | 1.10× | 1.20× |

(Rough numbers, on H100 + Llama-3-8B-Instruct + ShareGPT. Yours will differ; the *shape* should be similar.)

SGLang typically extracts more from prefix caching than vLLM does on this workload because RadixAttention does *cross-request, token-granularity* matching while vLLM's APC does block-granularity hash matching — the radix tree finds more re-usable prefix.

For workloads with more aggressive shared prefixes (system prompts, few-shot exemplars, agent tool histories), the gap widens further. ShareGPT is a *minimal* case.

---

## Ablation (b) — `max_num_batched_tokens` sweep

**Hypothesis.** This is the per-step token budget the scheduler can spend on prefill + decode combined (see [[vllm-scheduler]]). Bigger budget → bigger batches → higher throughput, but also bigger per-step latency → worse p99 TPOT.

**Run config.** Fix request rate at 16 (near-knee from baseline). Sweep `max_num_batched_tokens` over {2048, 4096, 8192, 16384, 32768}. Keep prefix caching ON.

```bash
for tokens in 2048 4096 8192 16384 32768; do
  # restart vLLM with the new value
  pkill -f "vllm serve"; sleep 5
  vllm serve ... --max-num-batched-tokens ${tokens} --enable-chunked-prefill --enable-prefix-caching &
  sleep 90  # weights reload
  python benchmark_serving.py ... --request-rate 16 \
    --result-filename "vllm_tokens${tokens}.json"
done
```

For SGLang, the equivalent flag is `--chunked-prefill-size`.

**Expected shape.**

| `max_num_batched_tokens` | TTFT p99 | TPOT p50 | Throughput |
|-------------------------:|---------:|---------:|-----------:|
| 2048  | low | high | low |
| 4096  | low-mid | mid | mid |
| 8192  | mid | low-mid | high ← typical sweet spot |
| 16384 | high | low | highest |
| 32768 | very high | low | plateau |

Why: at small budgets, prefill chunks block decode → high TTFT recovery time but each decode step is fast. At very large budgets, decode steps see many tokens to compute and start to feel the per-step latency cost (longer kernel runs).

The "sweet spot" on H100 + Llama-3-8B + ShareGPT is consistently ~8 192. On long-context workloads (RAG, 32 k prompts) the sweet spot moves up to 16 384–32 768.

---

## Ablation (c) — chunked-prefill chunk size

**Hypothesis.** The chunked-prefill *chunk* controls how a single long prompt is sliced. Small chunks let other work interleave between chunks (good for TPOT of decode tokens), at the cost of more total prefill steps.

**Run config.** Fix request rate at 16. In vLLM, chunk size is implicit (it's bounded by `max_num_batched_tokens`), so this ablation is more naturally done in SGLang:

```bash
for chunk in 512 1024 2048 4096 8192; do
  pkill -f "sglang.launch_server"; sleep 5
  python -m sglang.launch_server ... --chunked-prefill-size ${chunk} &
  sleep 90
  python benchmark_serving.py ... --base-url http://localhost:8002 \
    --request-rate 16 --result-filename "sgl_chunk${chunk}.json"
done
```

**Expected shape.**

| chunk_size | TTFT p99 (first chunk) | TTFT p99 (full prefill) | TPOT (decode tokens) |
|-----------:|----------------------:|-----------------------:|---------------------:|
| 512  | very low (first chunk fires fast) | high (many steps) | low (decode interleaves) |
| 1024 | low | mid | low |
| 2048 | mid | mid | mid ← Sarathi sweet spot |
| 4096 | mid-high | lower | mid-high |
| 8192 | high (single-step prefill) | lowest | high (decode starved) |

The Sarathi-Serve paper ([[ch-05]]) shows the qualitative shape: there's a sweet spot around 1–4k where TTFT-to-first-chunk is acceptable, total prefill is fast, and decode tokens don't starve. Below 512 the per-chunk overhead dominates.

---

## Interpreting the ablation results

For each ablation, the memo should answer **three** questions:

1. **Did the expected shape appear?** If not, the most likely cause is mis-set baseline flags (e.g., you accidentally left prefix caching on during ablation (a)) — verify the launch logs.
2. **Which framework is more sensitive to this knob?** SGLang's RadixAttention is more sensitive to ablation (a) than vLLM's APC; vLLM's scheduler is typically more sensitive to ablation (b) than SGLang's.
3. **Did the ablation move the *knee* of the rate sweep?** If yes, the knob is operationally important (re-run the full rate sweep with the new value). If no, it's a fine-tune dial.

---

## What this ablation suite does *not* cover

- **CUDA graph piecewise capture buckets.** Big effect on TPOT consistency, but tuning is per-architecture and outside scope.
- **Quantization (W4 / FP8).** Different format → different best-knob values; conflates the comparison.
- **Tensor parallelism.** Single-GPU lab; TP scaling is a separate study.
- **Attention backend choice.** FlashAttention 2 vs 3 vs FlashInfer — significant impact, but a kernel ablation, not a serving-framework ablation.

For the lab, three ablations on the configuration surface that both frameworks expose with comparable flags is the right scope.

---

## Connections

- [[ch-05]] — Sarathi-Serve mixed-batch scheduling: the academic origin of chunked prefill, source of the qualitative expected shape in ablation (c).
- [[ch-07]] — RadixAttention vs APC algorithm-level comparison; explains why ablation (a) usually moves more for SGLang.
- [[ch-16]] / [[ch-17]] — scheduler source-code pointers: `vllm/v1/core/sched/scheduler.py`, `sglang/srt/managers/scheduler.py`.
- [[excerpts/memo-template]] — where to record each ablation's numbers.
