<!-- chapter: ch-19
     track: benchmarks-production
     title: Inference Benchmarks + Metrics (TTFT / TPOT / ITL / Goodput)
     sources: [[ttft-tpot-itl]], [[sharegpt-workload]], [[vllm-benchmarks]], [[sglang-benchmarks]], [[genai-perf]], [[llmperf]], [[mlperf-inference-llm]], [[helm-inference]], [[goodput-slo]]
     figures: figures/ttft-tpot-tradeoff.html
-->

# Chapter 19 — Inference Benchmarks + Metrics: TTFT / TPOT / ITL / Goodput

> **Core insight.** Throughput (tokens/sec) is what providers report; **goodput under SLO** (requests/sec that meet a latency promise) is what users experience. Between them sit four orthogonal latency metrics — **TTFT** (time to first token, the prefill bottleneck), **TPOT** (time per output token, the decode steady state), **ITL** (inter-token latency, the jitter), and **end-to-end latency** (the sum users actually feel) — each measuring a different thing the scheduler can break. A serving system with great average throughput can have catastrophic p99 TTFT under burst load; a system with great p50 numbers can have a p99 cliff that makes it unusable for chat. The 2025 frontier metric is goodput, the only one that aligns with business value.
>
> **Guideline.** Always report (TTFT_p50, TTFT_p95, TPOT_p50, TPOT_p95, throughput_tokens_per_sec, goodput@SLO) — six numbers, never one. Sweep request rate (1 → saturation) instead of picking a fixed concurrency. Match the workload distribution to your production traffic (ShareGPT is the OSS default; capture your own if it's available). Use the framework-native benchmark for engine-level numbers (`vllm bench serve`, `python -m sglang.bench_serving`), GenAI-Perf or LLMPerf for endpoint-level comparison across stacks, and MLPerf only when you need audited cross-vendor hardware comparison. Define the SLO *first*, then measure goodput against it — not the other way around.

---

## Why this chapter exists

LLM serving benchmarks are unusually easy to misreport. The same engine on the same hardware can produce a "10× faster" headline or a "no improvement" headline depending on which metric you pick, what request rate you tested at, whether you warmed the cache, whether you used p50 or p99, and whether you defined an SLO. Three of the most-cited 2024 vLLM-vs-TGI-vs-TRT-LLM benchmarks contradict each other for exactly these reasons.

Three things to walk away with:

1. **The four metrics, the two phases.** TTFT measures prefill; TPOT measures decode; ITL measures decode jitter; throughput measures aggregate. Each one is the right metric for a different question.
2. **The traps.** Percentile lies, warm-up cache contamination, request-rate sweeps that miss saturation, tokenizer drift, mixing streaming and non-streaming runs, ignoring failed requests.
3. **Goodput under SLO as the unifying business metric.** All the per-request latency metrics matter only because they predict whether a request lands inside an SLO; goodput counts only the ones that do. This is the metric that aligns capacity planning with user experience.

The sources are [[ttft-tpot-itl]] (the canonical metric definitions), [[sharegpt-workload]] (the de-facto workload distribution), [[vllm-benchmarks]] + [[sglang-benchmarks]] (framework-native scripts), [[genai-perf]] + [[llmperf]] (cross-framework endpoint testers), [[mlperf-inference-llm]] (audited cross-vendor), [[helm-inference]] (multi-dimension reporting discipline), and [[goodput-slo]] (the unifying metric).

---

## 1. The two phases — why we need more than one latency number

A streaming LLM request has two qualitatively different phases:

```
Wall clock →

|<------ TTFT ------>|<------------------- decode ------------------>|
|                    |                                                |
prompt arrives       first token emitted                       request done
       prefill (compute-bound, O(L²·d))     decode (memory-bound, ~constant per token)
```

**Prefill** is the GPU running attention over the whole prompt — compute-bound, scales as O(L²·d). For a 4k-token prompt on Llama-3-8B at H100 batch 16, prefill takes 100–300 ms.

**Decode** is the GPU emitting tokens one at a time — memory-bandwidth-bound (re-read all KV each step), roughly constant per token. For Llama-3-8B at H100, ~5–15 ms per token depending on batch size and KV cache state.

These two phases have different scaling, different bottlenecks, and different SLO targets. Reporting one "average latency" obscures both.

---

## 2. The four core metrics

### 2.1 TTFT — Time To First Token

**Definition:** wall-clock from client request to receiving first generated token.

**What it includes by default:** request queuing + scheduling + tokenization (sometimes) + prefill compute + first decode step + network transit.

**When it matters:** every interactive use case (chat, autocomplete, voice assistant). Sub-second TTFT is the perception bar for "responsive"; above 2 s users notice the wait.

**Typical numbers (Llama-3-8B, H100, vLLM, 1k-token prompt, idle server):**
- p50: 80–150 ms
- p95: 150–300 ms
- p99: 300–800 ms

Under load (request rate above saturation), p99 can blow up to 5–30 seconds — the prefill queue is the worst tail in LLM serving.

### 2.2 TPOT — Time Per Output Token (steady-state decode)

**Definition:** `(end_time - first_token_time) / (output_tokens - 1)` — average per-token time *after* the first token.

**What it measures:** decode steady state, excluding the prefill spike.

**When it matters:** long-output workloads (RAG, code generation, reasoning chains). At 50 ms TPOT a 500-token answer takes 25 seconds; at 20 ms it takes 10 seconds.

**Typical numbers (Llama-3-8B, H100, batch 16 decode):**
- TPOT ≈ 10–25 ms/token at moderate batch
- TPOT ≈ 25–60 ms/token at large batch (memory bandwidth saturated)

Reciprocal of TPOT is the user-perceived **token rate**: 1000 / TPOT_ms = tokens/sec for a single user. 20 ms TPOT = 50 tok/s perceived rate.

### 2.3 ITL — Inter-Token Latency

**Definition:** distribution of gaps between successive streamed tokens.

**Why it's distinct from TPOT:** TPOT is a *mean*. ITL is the *distribution*. A system can have great mean TPOT and terrible ITL p99 — meaning most tokens arrive on time but occasional ones stall for hundreds of ms, producing visible "stutter" in streaming UI.

**What causes ITL spikes:**
- Scheduler interruption: another request's prefill chunk landed in your batch
- Cache eviction: KV pages got reshuffled
- CUDA-Graph recapture: batch size crossed a bucket boundary
- Garbage collection in Python serving code
- KV offload swap-in from CPU

**When to report:** anywhere streaming UX matters. Report ITL_p50, ITL_p95, ITL_p99 — the tail matters more than the mean for perceived smoothness.

### 2.4 Throughput — aggregate tokens/sec or requests/sec

**Definition:** total output tokens / wall-clock time across a benchmark run; or, equivalently, requests completed per second.

**What it measures:** engine capacity, *without* SLO consideration. The number providers like to report.

**When it matters:** capacity planning (how many GPUs do we need to serve N tokens/day?). For interactive serving it's the wrong primary metric.

**Typical numbers (Llama-3-8B on 1× H100):**
- Offline throughput (no SLO): 6,000–12,000 output tokens/sec
- Online throughput at sustainable SLO: 2,000–5,000 output tokens/sec
- 2× gap between offline and online is real — see goodput, section 7.

---

## 3. The common traps

### 3.1 The percentile lie

> "vLLM has 80 ms p50 TTFT" → true and useless.

Most LLM serving distributions are heavy-tailed. p50 hides the catastrophic tail.

```
TTFT distribution (idle vs loaded):
  Idle:    p50=80ms, p95=150ms, p99=300ms       ← healthy
  Loaded:  p50=120ms, p95=4500ms, p99=22000ms   ← cliff
```

The p50 looks roughly fine in both cases. The p99 reveals the saturation cliff.

**Rule.** Report at least (p50, p95, p99). For consumer-facing SLOs, p99.9 is sometimes required.

### 3.2 Warm-up cache contamination

Most benchmark scripts have a warm-up phase: run N requests, discard them, then measure. Two failure modes:

- **Warm-up too short.** First measured requests still pay one-time costs (CUDA graph capture, kernel JIT, weight cache fill, prefix cache cold). Numbers look worse than steady state.
- **Warm-up contaminates measurement.** The same prompts used in warm-up sit in the prefix cache during measurement, so measured TTFT is artificially low (almost-zero prefill).

**Rule.** Use *different* prompts for warm-up and measurement; or randomize prompts; or disable prefix caching explicitly for a *clean* throughput number, then re-run with it enabled.

### 3.3 Request-rate sweeps that miss saturation

A benchmark at one fixed request rate tells you nothing about behavior under burst. The standard methodology:

```
for rate in [1, 2, 4, 8, 16, 32, 64]:           # req/s sweep
    results = run_bench(duration=300, rate=rate)
    plot(rate, results.ttft_p99, results.throughput, results.goodput)
```

Look for the **knee**: the request rate at which TTFT_p99 explodes. That's saturation. Operate at 60–80 % of saturation in production.

A sweep that stops at rate=8 when saturation is at 24 missed the whole story.

### 3.4 Tokenizer drift

Two benchmarks "on Llama-3-8B" using different chat templates produce different input-token counts → different prefill cost → different TTFT. A common error.

**Rule.** Pin model revision, tokenizer revision, and chat template explicitly. Report all three.

### 3.5 Streaming vs non-streaming mixed

Streaming uses SSE; the wire protocol has per-chunk overhead. Non-streaming returns all tokens at once. They are not comparable.

**Rule.** Pick one mode per benchmark. Streaming is the right choice for chat SLOs; non-streaming for batch jobs.

### 3.6 Ignoring failed requests

Requests that timeout or error don't appear in latency percentiles — they vanish. A system at 10 % error rate can look great on TTFT.

**Rule.** Report error rate explicitly. Goodput counts only successful + SLO-meeting requests, which is the right denominator.

---

## 4. Workload design — ShareGPT and beyond

[[sharegpt-workload]] is the OSS default trace.

### 4.1 What ShareGPT is

A scrape of real user-assistant conversations from ChatGPT-era public dumps. Used to extract realistic prompt + response length distributions:

```
Prompt length distribution:    median ~250 tokens, p95 ~2k, p99 ~8k
Response length distribution:  median ~400 tokens, p95 ~1.5k, p99 ~4k
```

Heavy-tailed in both dimensions — short conversations are common, long ones rare but not negligible.

### 4.2 How it's replayed

```python
# vllm/benchmarks/benchmark_serving.py simplified
prompts, output_lens = sample_sharegpt(num_samples=1000, tokenizer=tok)
# Two replay modes:
# (a) request rate (open-loop): issue prompts at Poisson(rate) regardless of completion
# (b) fixed concurrency (closed-loop): keep N in flight at all times
```

**Open-loop (request rate)** is the right mode for serving benchmarks — it captures queuing effects. **Closed-loop (concurrency)** is the right mode for throughput-only benchmarks where you want to measure engine capacity in isolation.

### 4.3 Disclosure required

When reporting a ShareGPT-based benchmark, disclose:
- Source JSON file + revision
- Tokenizer used to compute lengths
- Filter: min/max prompt and output length
- Whether output lengths were replayed exactly or generated to EOS
- Random seed
- Chat template applied (or not)

Without these, results are not reproducible.

### 4.4 Alternatives

- **Synthetic uniform length** — easy to control, unrealistic.
- **LongBench / RULER** — long-context benchmarks (up to 128k tokens), useful for KV-bound workloads.
- **Production trace replay** — if you have it, use it. Anonymize, tokenize, replay.
- **Domain-specific traces** — code completion (CodeAlpaca), summarization (CNN/DM), math reasoning (GSM-8k).

---

## 5. The benchmark stacks

### 5.1 vLLM built-in — `vllm bench` / `benchmark_serving.py`

[[vllm-benchmarks]]: the framework-native benchmark, paired with [[vllm]].

Two scripts:

```bash
# Offline engine throughput (no HTTP overhead)
python benchmarks/benchmark_throughput.py \
    --model meta-llama/Llama-3-8B-Instruct \
    --dataset ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 1000 \
    --output-len 256

# Online serving (against a running vllm serve)
python benchmarks/benchmark_serving.py \
    --backend vllm \
    --base-url http://localhost:8000 \
    --model meta-llama/Llama-3-8B-Instruct \
    --dataset-name sharegpt \
    --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 1000 \
    --request-rate 16 \
    --percentile-metrics ttft,tpot,itl,e2el
```

Output:

```
============ Serving Benchmark Result ============
Successful requests:                     1000
Benchmark duration (s):                  62.51
Total input tokens:                      213412
Total generated tokens:                  148927
Request throughput (req/s):              15.99
Output token throughput (tok/s):         2382.55
Total Token throughput (tok/s):          5796.78
---------------Time to First Token----------------
Mean TTFT (ms):                          146.32
Median TTFT (ms):                        112.85
P99 TTFT (ms):                           892.41
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          18.74
Median TPOT (ms):                        16.92
P99 TPOT (ms):                           48.31
---------------Inter-token Latency----------------
Mean ITL (ms):                           18.62
Median ITL (ms):                         16.88
P99 ITL (ms):                            52.13
==================================================
```

This output format is the de-facto standard — SGLang, GenAI-Perf, LLMPerf all converge on similar tables.

### 5.2 SGLang built-in — `python -m sglang.bench_serving`

[[sglang-benchmarks]]: same shape as vLLM's serving benchmark, plus prefix-cache-aware modes.

```bash
python -m sglang.bench_serving \
    --backend sglang \
    --host 127.0.0.1 --port 30000 \
    --dataset-name sharegpt \
    --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 1000 \
    --request-rate 16 \
    --random-input-len 1024 \
    --random-output-len 256
```

Plus a prefix-shared workload mode that highlights RadixAttention value:

```bash
python -m sglang.bench_serving \
    --dataset-name generated-shared-prefix \
    --random-input-len 4096 \
    --gen-shared-prefix-num-groups 8 \
    --gen-shared-prefix-prompts-per-group 32
```

This generates a workload where 8 different prefixes are each used by 32 requests — directly tests prefix-cache behavior.

### 5.3 GenAI-Perf (NVIDIA)

[[genai-perf]]: NVIDIA's standardized client-side endpoint tester. Talks to Triton, TensorRT-LLM, OpenAI-compatible, or any HTTP endpoint.

```bash
genai-perf profile \
    --model meta-llama/Llama-3-8B-Instruct \
    --endpoint-type chat \
    --backend openai \
    --url http://localhost:8000 \
    --num-prompts 500 \
    --random-seed 42 \
    --synthetic-input-tokens-mean 1024 --synthetic-input-tokens-stddev 256 \
    --output-tokens-mean 256 --output-tokens-stddev 64 \
    --concurrency 16 \
    --measurement-interval 30000 \
    --streaming \
    --verbose
```

Strengths: NVIDIA-blessed metric definitions; works against many backends; precise control over synthetic input length distributions.

### 5.4 LLMPerf (Anyscale)

[[llmperf]]: Python client for hosted-API comparison. Useful for comparing OpenAI vs Anthropic vs your-deployed-vLLM through the same harness.

```bash
python token_benchmark_ray.py \
    --model meta-llama/Llama-3-8B-Instruct \
    --mean-input-tokens 1024 --stddev-input-tokens 256 \
    --mean-output-tokens 256 --stddev-output-tokens 64 \
    --max-num-completed-requests 500 \
    --num-concurrent-requests 16 \
    --llm-api openai \
    --additional-sampling-params '{}' \
    --results-dir results
```

Strengths: provider-agnostic; reports per-token throughput, percentile latency, error rate.

### 5.5 MLPerf Inference (LLM tracks)

[[mlperf-inference-llm]]: MLCommons-audited cross-vendor benchmark.

Scenarios:
- **Offline** — maximum throughput (no latency constraint)
- **Server** — Poisson arrivals + per-task latency SLO; reports tokens/sec at SLO

LLM tasks (round 4.0+):
- Llama-2-70B (Q&A)
- Mixtral-8x7B (code + math + Q&A)
- Llama-3.1-405B (long-context summarization)
- GPT-J-6B (legacy)

Use when you need audited hardware/software comparison numbers; do not use as a tuning loop.

### 5.6 HELM Inference

[[helm-inference]]: Stanford CRFM's multi-dimensional evaluation framework. Less a load generator, more a reporting discipline.

Key contribution: **multi-metric reporting** — accuracy + calibration + robustness + fairness + bias + toxicity + efficiency, all reported separately. Borrow this discipline: never report a single number when six dimensions exist.

---

## 6. Goodput under SLO — the unifying metric

[[goodput-slo]]: the metric that aligns capacity planning with user experience.

### 6.1 Definition

```
goodput = (requests/sec that satisfy SLO predicates) / total_request_rate
goodput@SLO = goodput when offered_rate = saturation
```

SLO predicates are typically conjunctive:
```
SLO satisfied if:
   TTFT < 2000 ms        AND
   TPOT < 100 ms/token   AND
   end_to_end < 30000 ms AND
   request did not error
```

A request is *good* iff it meets all predicates. Goodput counts only good requests.

### 6.2 Why throughput alone misleads

Consider two systems:

| Metric | System A | System B |
|--------|---------:|---------:|
| Output tokens/sec at saturation | 8,000 | 5,500 |
| p99 TTFT at saturation | 25 s | 1.8 s |
| p99 TPOT at saturation | 250 ms | 45 ms |
| Goodput@SLO (TTFT<2s, TPOT<100ms) | 800 tok/s | 5,200 tok/s |

System A's "throughput win" disappears once you require requests to be acceptable to users. **Goodput is what counts.**

### 6.3 How to measure it

```python
def goodput_sweep(server, dataset, slo, rates=[1, 2, 4, 8, 16, 32, 64]):
    results = []
    for rate in rates:
        run = bench_serve(server, dataset, request_rate=rate, duration=300)
        good_reqs = sum(
            1 for r in run.requests
            if r.ttft_ms < slo.ttft_ms
               and r.tpot_ms < slo.tpot_ms
               and r.e2e_ms  < slo.e2e_ms
               and r.success
        )
        goodput = good_reqs / run.duration_sec
        results.append((rate, goodput, run.ttft_p99, run.tpot_p99))
    return results
```

Plot offered rate (x) vs goodput (y). The peak goodput value is your system's real capacity.

### 6.4 Standard SLO presets

| Workload | Typical SLO |
|----------|-------------|
| Interactive chat | TTFT < 1 s, TPOT < 50 ms (perceived 20+ tok/s) |
| Voice assistant (TTS pipeline) | TTFT < 500 ms, TPOT < 33 ms (30+ tok/s) |
| Code completion (Cursor / Copilot) | TTFT < 200 ms, TPOT < 25 ms |
| Long-form RAG | TTFT < 3 s, TPOT < 100 ms, end-to-end < 60 s |
| Batch summarization | end-to-end < 5 min, no streaming SLO |

Set the SLO based on user research, not benchmark convenience.

---

## 7. End-to-end methodology — the checklist

```
Pre-benchmark:
  ☐ Pin model + revision + tokenizer + chat template + framework version
  ☐ Pin hardware (GPU SKU, driver, CUDA, kernel version)
  ☐ Choose workload (ShareGPT / custom / synthetic) + disclose filters + seed
  ☐ Choose streaming vs non-streaming (pick one)
  ☐ Decide whether prefix cache is on or off (measure both)
  ☐ Define SLO predicates explicitly

Warm-up:
  ☐ Use distinct prompts for warm-up vs measurement
  ☐ Run until throughput plateaus (typically 60-120 s)
  ☐ Discard warm-up window from results

Sweep:
  ☐ Sweep request rate [1, 2, 4, 8, 16, ..., 2×expected_saturation]
  ☐ Each rate runs for at least 5 min or 500 requests
  ☐ Capture (TTFT, TPOT, ITL, e2e, throughput, goodput) per rate

Report:
  ☐ For each metric: p50, p95, p99 (and p99.9 if SLO requires)
  ☐ Error rate explicitly
  ☐ Goodput@SLO at peak
  ☐ Plot rate vs goodput; mark the knee
  ☐ Disclose all preconditions (model, hardware, workload, knobs, seed)
```

---

## 8. Concrete benchmark numbers — Llama-3-8B-Instruct, 1× H100, ShareGPT

From vLLM 0.6.x + SGLang 0.4.x reports, March 2026 (representative; numbers move):

| Framework | Throughput (tok/s) | TTFT_p99 @ rate=16 | TPOT_p99 @ rate=16 | Goodput@SLO (TTFT<2s, TPOT<50ms) |
|-----------|-------------------:|-------------------:|-------------------:|---------------------------------:|
| vLLM 0.6 (APC on) | 5,800 | 380 ms | 32 ms | 14.2 req/s |
| vLLM 0.6 (APC off) | 4,900 | 450 ms | 35 ms | 13.8 req/s |
| SGLang 0.4 (RadixAttn) | 6,200 | 290 ms | 28 ms | 15.3 req/s |
| SGLang 0.4 (shared-prefix workload) | 9,800 | 180 ms | 24 ms | 27.5 req/s |
| TGI 2.4 | 4,800 | 510 ms | 38 ms | 12.1 req/s |
| TensorRT-LLM 0.13 (FP8) | 8,400 | 240 ms | 22 ms | 19.4 req/s |
| TensorRT-LLM 0.13 (BF16) | 5,900 | 320 ms | 30 ms | 14.6 req/s |

Reading: TRT-LLM FP8 wins absolute throughput; SGLang on shared-prefix workloads wins goodput by a wide margin; vLLM is the OSS reference; TGI trails on numbers but wins on ops. **Don't extrapolate** — your workload will show different ratios.

---

## Common pitfalls

- **One number, no percentiles.** "We get 8k tok/s" is unfalsifiable. Always p50 + p95 + p99.
- **Fixed concurrency only.** Misses queueing effects; use a request-rate sweep.
- **Closed-loop benchmark labeled "online".** Open-loop (Poisson arrivals) is the right model for serving.
- **Prefix cache enabled, same prompts in warm-up and measurement.** Measured TTFT is fake; rerun with distinct prompts.
- **Tokenizer not pinned across runs.** Different chat templates → different prefill cost → different TTFT.
- **Reporting goodput without disclosing SLO.** "Goodput = 15 req/s" without saying "(TTFT<2s, TPOT<50ms)" is meaningless.
- **Throughput numbers from offline mode used to predict online SLO.** Offline is the easy regime; online is what matters for users.
- **Cross-framework comparison with mismatched flags.** vLLM `--enable-prefix-caching` vs SGLang's `lpm` policy are not automatic — match feature sets explicitly.
- **Ignoring failed requests.** A 5 % error rate is a serving failure even if "successful TTFT_p99 = 400 ms" looks great.
- **MLPerf numbers used for chat sizing.** MLPerf scenarios don't model conversational arrival patterns; chat is bursty in ways MLPerf doesn't capture.

---

## Connections and what's next

- **Back: [[continuous-batching]] / ch-04** — TTFT/TPOT are the metrics scheduling decisions move. Continuous batching is what makes goodput possible.
- **Back: [[sarathi-serve]] / ch-05** — chunked prefill trades small TTFT increase for big TPOT decrease; ch-19 metrics are how you verify the trade.
- **Back: [[goodput-slo]] / ch-10** — VTC / admission-control / Niyama treatments work by *defining* goodput; this chapter is the measurement counterpart.
- **Back: [[vllm-scheduler]] / ch-16** + **[[sglang-scheduler]] / ch-17** — what `--max-num-batched-tokens`, `--chunked-prefill-size`, `--schedule-policy` do, measured.
- **Lateral: ch-18** — framework choice + benchmark methodology together; never decide one without the other.
- **Forward: ch-20** — apply this methodology to production model reports (Llama 3, DeepSeek V3, Qwen 3, GPT-OSS).
- **Forward: ch-21 (lab)** — vLLM vs SGLang on ShareGPT with this exact methodology.

## Further reading

- [[ttft-tpot-itl]] — canonical metric definitions.
- [[goodput-slo]] — the SLO-aware capacity metric.
- [[sharegpt-workload]] — workload distribution and replay methodology.
- [[vllm-benchmarks]] — `vllm bench serve` and `benchmark_throughput.py`.
- [[sglang-benchmarks]] — `python -m sglang.bench_serving`.
- [[genai-perf]] — NVIDIA's standardized endpoint tester.
- [[llmperf]] — Anyscale's cross-provider API tester.
- [[mlperf-inference-llm]] — audited cross-vendor.
- [[helm-inference]] — multi-dimensional reporting discipline.

## Companion visualization

**[figures/ttft-tpot-tradeoff.html](figures/ttft-tpot-tradeoff.html)** — interactive request-rate sweep plot. Sliders for request rate (1 → 64 req/s), prefix-cache hit ratio (0 → 0.95), batch size cap, chunked-prefill chunk size. Overlay: TTFT_p50 / TTFT_p99 / TPOT_p50 / TPOT_p99 / throughput / goodput@SLO. SLO thresholds are user-settable; the goodput line drops to 0 above saturation. Use it to see why chunked prefill raises TTFT slightly and lowers TPOT a lot.
