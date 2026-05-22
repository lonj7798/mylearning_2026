---
chapter: ch-19
course: llm-inference
phase: read
excerpt_of: "Benchmark stacks — vLLM-bench, SGLang-bench, GenAI-Perf, LLMPerf, MLPerf, HELM"
source_url: composite
created_at: "2026-05-21"
---

# Excerpt: The benchmark-stack landscape

**Authors:** vLLM project + SGLang project + NVIDIA + Anyscale + MLCommons + Stanford CRFM
**Year:** 2023–2026
**Raw-data sources:** [[raw-data/vllm-benchmarks]] + [[raw-data/sglang-benchmarks]] + [[raw-data/genai-perf]] + [[raw-data/llmperf]] + [[raw-data/mlperf-inference-llm]] + [[raw-data/helm-inference]]

---

## Six stacks, each with a different purpose

| Stack | Purpose | Use when |
|-------|---------|----------|
| **vLLM built-in** | Engine + serving capacity, vLLM-specific | Tuning vLLM knobs |
| **SGLang built-in** | Engine + serving capacity, SGLang-specific | Tuning SGLang knobs; prefix-cache workloads |
| **GenAI-Perf** (NVIDIA) | Cross-backend endpoint comparison | Comparing TRT-LLM / vLLM / Triton |
| **LLMPerf** (Anyscale) | Hosted-API + self-hosted comparison | Comparing OpenAI vs Anthropic vs your-vLLM |
| **MLPerf Inference** | Audited cross-vendor hardware comparison | Hardware selection / certification |
| **HELM** | Multi-dimensional model + system reporting | Reporting discipline; multi-metric tables |

---

## vLLM built-in benchmarks

```bash
# Offline throughput (no HTTP overhead) — measures engine in isolation
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
    --percentile-metrics ttft,tpot,itl,e2el \
    --goodput ttft:2000 tpot:50          # report goodput@SLO
```

Output table convention:
```
============ Serving Benchmark Result ============
Successful requests:                     1000
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
P99 ITL (ms):                            52.13
==================================================
```

This table format is the de-facto standard.

---

## SGLang built-in benchmark

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

# Prefix-shared variant (highlights RadixAttention)
python -m sglang.bench_serving \
    --dataset-name generated-shared-prefix \
    --random-input-len 4096 \
    --gen-shared-prefix-num-groups 8 \
    --gen-shared-prefix-prompts-per-group 32
```

Same output table shape as vLLM. The shared-prefix dataset is the SGLang-specific addition that lets you measure prefix-cache benefit directly.

---

## GenAI-Perf (NVIDIA)

```bash
genai-perf profile \
    --model meta-llama/Llama-3-8B-Instruct \
    --endpoint-type chat \
    --backend openai \
    --url http://localhost:8000 \
    --num-prompts 500 \
    --random-seed 42 \
    --synthetic-input-tokens-mean 1024 \
    --synthetic-input-tokens-stddev 256 \
    --output-tokens-mean 256 \
    --output-tokens-stddev 64 \
    --concurrency 16 \
    --measurement-interval 30000 \
    --streaming
```

Strengths:
- Talks to many backends (Triton, TRT-LLM, OpenAI-compat, raw HTTP)
- Precise control over synthetic length distributions
- NVIDIA-blessed metric definitions
- Produces machine-readable JSON for downstream analysis

Outputs (excerpt):
```
Time to first token (ms):
  avg=143.21, min=42.15, max=1832.04
  p25=89.12, p50=121.34, p75=178.92, p90=234.56, p95=312.45, p99=892.41
Inter token latency (ms):
  avg=18.62, p50=16.88, p95=23.41, p99=52.13
Request throughput (per sec): 15.92
Output token throughput (per sec): 2378.04
```

---

## LLMPerf (Anyscale)

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

Strengths:
- Provider-agnostic (OpenAI, Anthropic, Together, your-deployed-vLLM through one harness)
- Ray-based concurrent execution
- Reports per-token throughput, percentile latency, error rate

Use when: comparing hosted API providers, or your self-hosted endpoint against a hosted baseline.

---

## MLPerf Inference (LLM tracks)

Scenarios:

| Scenario | What it tests |
|----------|---------------|
| **Offline** | Maximum throughput, no latency constraint |
| **Server** | Poisson arrivals + per-task latency SLO; reports tokens/sec subject to SLO |
| **SingleStream** (edge) | One request at a time, latency only |
| **MultiStream** (edge) | Fixed concurrency, latency-bounded |

LLM tasks (round 4.0+):
- Llama-2-70B (Q&A)
- Mixtral-8x7B (code + math + Q&A)
- Llama-3.1-405B (long-context summarization)
- GPT-J-6B (legacy)

Use when: you need audited cross-vendor hardware/software comparison numbers. Don't use as a tuning loop — overhead is too high.

---

## HELM Inference

Stanford CRFM's framework. Not a load generator — a *reporting discipline*. Multi-dimensional evaluation:

- Accuracy (per task)
- Calibration
- Robustness
- Fairness
- Bias
- Toxicity
- **Efficiency** (latency, cost, throughput)

The key contribution: **never report a single number when six dimensions exist**. Borrow this discipline even for systems benchmarks — pair latency + throughput + goodput + accuracy + cost.

---

## Composition pattern

Real benchmarks compose multiple stacks:

```
1. vLLM/SGLang built-in for engine-knob tuning (fast iteration)
2. GenAI-Perf for cross-framework comparison (TRT-LLM vs vLLM vs SGLang)
3. LLMPerf for end-to-end with realistic network path (cross-region, cross-provider)
4. MLPerf when you need an audited comparison number
5. HELM-style reporting discipline throughout
```

---

## Connections

- [[excerpts/ttft-tpot-itl]] — the metric definitions all stacks share.
- [[excerpts/sharegpt-workload]] — the workload most stacks use by default.
- [[excerpts/goodput-slo]] — the SLO-aware metric.
- [[ch-19]] — parent synthesis.
