---
chapter: ch-19
course: llm-inference
phase: read
excerpt_of: "TTFT / TPOT / ITL metric definitions — GenAI-Perf and vLLM benchmark conventions"
source_url: https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/perf_analyzer/genai-perf/README.html
created_at: "2026-05-21"
---

# Excerpt: TTFT / TPOT / ITL — the four-metric vocabulary

**Authors:** NVIDIA GenAI-Perf + vLLM + Anyscale (LLMPerf) + the broader serving community
**Year:** 2023–2026
**URLs:** https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/perf_analyzer/genai-perf/README.html / https://docs.vllm.ai/en/latest/contributing/benchmarks.html
**Raw-data source:** [[raw-data/ttft-tpot-itl]]

---

## The four metrics, one canonical definition each

### TTFT — Time To First Token
```
TTFT = first_token_received_time − request_sent_time
```
Includes: client→server network, request validation, queueing, tokenization (server-side), scheduling, prefill compute, first decode step, first-token network return.

Often the dominant user-perceived delay for short responses.

### TPOT — Time Per Output Token (steady state)
```
TPOT = (request_finished_time − first_token_received_time) / (output_tokens − 1)
```
Average per-token decode time, *excluding* the first token (which contains all prefill cost).

This is the right metric for "how fast does it generate" once the first token has arrived.

### ITL — Inter-Token Latency (distribution)
```
ITL_i = token_i_received_time − token_{i-1}_received_time   for i ≥ 1
ITL distribution = {ITL_1, ITL_2, ..., ITL_{N-1}}
```
The distribution of per-token gaps. Mean(ITL) ≈ TPOT, but the **tail** (ITL_p99) reveals stutter.

### End-to-end Latency
```
e2e = request_finished_time − request_sent_time
    = TTFT + TPOT × (output_tokens − 1)
```
What the user actually waits for the full response.

### Throughput (aggregate)
```
output_throughput  = total_output_tokens / benchmark_duration
input_throughput   = total_input_tokens  / benchmark_duration
request_throughput = total_requests      / benchmark_duration
```

---

## Why TPOT excludes the first token

The first token's wall-clock time conflates prefill (compute-bound, O(L²·d)) with decode (memory-bound, ~constant). If you include it in the "per-token" average, the average is dominated by prompt length, not decode speed.

Concretely, for a 4k-token prompt + 200-token response on Llama-3-8B / H100:
- TTFT ≈ 200 ms (mostly prefill)
- Per-decode-token time ≈ 18 ms
- Naive average = (200 + 199 × 18) / 200 = 18.9 ms — *almost right by coincidence*
- For a 32k-token prompt + 200-token response:
- TTFT ≈ 1600 ms (prefill scales)
- Per-decode-token time ≈ 25 ms (KV cache larger)
- Naive average = (1600 + 199 × 25) / 200 = 32.9 ms — *massively misleading about decode speed*

TPOT's exclusion of the first token cleanly separates the two concerns.

---

## Percentiles, not means

A benchmark must report at least (p50, p95, p99) for each metric. Means hide tails; serving tails are the failure mode.

```
Healthy idle system:
   TTFT  p50=80ms   p95=130ms   p99=210ms
   TPOT  p50=18ms   p95=24ms    p99=32ms
   ITL   p50=18ms   p95=23ms    p99=58ms

Saturated system:
   TTFT  p50=180ms  p95=4800ms  p99=22000ms     ← cliff
   TPOT  p50=22ms   p95=85ms    p99=320ms       ← cliff
   ITL   p50=20ms   p95=300ms   p99=2400ms      ← visible stutter
```

The p50s look fine in both cases; the p99s tell the truth.

---

## What to disclose alongside the numbers

- Network path (client and server colocated? same DC? cross-region?)
- Streaming or non-streaming mode
- Tokenization location (client-side or server-side; uses Hugging Face tokenizer or framework tokenizer)
- Warm-up policy (how many requests, what prompts, discarded or not)
- Prefix cache state (cold, warm-shared, warm-distinct)
- Whether failed/timeout requests are included in percentiles

Two benchmark runs that disagree on any of these are not comparable.

---

## Connections

- [[excerpts/sharegpt-workload]] — the dataset providing realistic prompt/output lengths to test on.
- [[excerpts/goodput-slo]] — the SLO-aware metric that combines all four.
- [[excerpts/genai-perf]] — the NVIDIA tool that codified these definitions.
- [[ch-19]] — parent synthesis.
