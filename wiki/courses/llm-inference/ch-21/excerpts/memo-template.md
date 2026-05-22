---
chapter: ch-21
course: llm-inference
phase: read
excerpt_of: "Lab memo template + reflection scaffold"
source_url: internal
created_at: "2026-05-21"
---

# Excerpt: One-page lab memo template

**Source:** assembled from [[goodput-slo]] + [[ttft-tpot-itl]] reporting conventions
**Raw-data source:** [[raw-data/vllm-benchmarks]], [[raw-data/sglang-benchmarks]], [[raw-data/goodput-slo]]

---

## The template

Copy-paste this into `lab-memo.md`. Fill the underscores. Do not restructure — the standard form is what makes the lab comparable across cohorts.

```markdown
# Lab Memo — vLLM vs SGLang on Llama-3-8B (or Qwen-1.8B)

**Author:**           ____
**Path:**             full-budget | resource-constrained
**Hardware:**         GPU model, count, CUDA version
**Model:**            <model id + HF commit hash>
**Workload:**         ShareGPT V3 cleaned, <N> prompts/run, seed 42
**SLO:**              TTFT p99 ≤ <ms>, TPOT p99 ≤ <ms>, ITL p99 ≤ <ms>
**vLLM version:**     ____
**SGLang version:**   ____
**Date:**             ____

## 1. Headline table — three saturation points

| Stack  | Rate | TTFT p50 | TTFT p99 | TPOT p50 | TPOT p99 | ITL p99 | Tput tok/s | Goodput@p99 (req/s) |
|--------|-----:|---------:|---------:|---------:|---------:|--------:|-----------:|--------------------:|
| vLLM   |   4  |  ____    |  ____    |  ____    |  ____    |  ____   |  ____      |  ____               |
| SGLang |   4  |  ____    |  ____    |  ____    |  ____    |  ____   |  ____      |  ____               |
| vLLM   |  16  |  ____    |  ____    |  ____    |  ____    |  ____   |  ____      |  ____               |
| SGLang |  16  |  ____    |  ____    |  ____    |  ____    |  ____   |  ____      |  ____               |
| vLLM   |  32  |  ____    |  ____    |  ____    |  ____    |  ____   |  ____      |  ____               |
| SGLang |  32  |  ____    |  ____    |  ____    |  ____    |  ____   |  ____      |  ____               |

Knee detection (rate at which TTFT p99 crosses SLO): vLLM = ____, SGLang = ____.

## 2. Ablation result

(a) Prefix caching off → on, at rate 16:

| Stack  | TTFT p99 OFF | TTFT p99 ON | Δ %  | Throughput OFF | Throughput ON | Δ %  |
|--------|-------------:|------------:|-----:|---------------:|--------------:|-----:|
| vLLM   |  ____        |  ____       |  ____ |  ____         |  ____         |  ____ |
| SGLang |  ____        |  ____       |  ____ |  ____         |  ____         |  ____ |

(Optional) (b) `max_num_batched_tokens` sweep at rate 16, sweet spot identified: ____.
(Optional) (c) Chunk-size sweep at rate 16 (SGLang only), sweet spot identified: ____.

## 3. One specific failure mode I observed

(Required. Specific = framework + workload + knob + observation + fix. NOT "SGLang was slower than I expected." Something like:

  "vLLM TTFT p99 spiked to 4500 ms at rate 24, exceeding the 2000 ms SLO.
  Inspection of `/metrics`/scheduler stats showed `swap-out events` rising
  to ~30 per second around the knee. Lowering `--max-num-seqs` from 256 to 128
  reduced KV-cache pressure and dropped TTFT p99 to 1800 ms at the same rate.
  The lesson: `max_num_seqs` must be calibrated against the per-request KV
  cost for the chosen `max_model_len`, not left at the default.")

## 4. Recommendation

- For workload class A (e.g., short independent chats, no shared prefix):  ____ wins, because ____.
- For workload class B (e.g., long shared system prompts, agent loops):    ____ wins, because ____.
- For workload class C (e.g., very long context, low-concurrency RAG):     ____ wins, because ____.

## 5. What would flip my conclusion

Three concrete scenarios that would change the recommendation:

1. ____ (e.g., "context > 32k": the framework with better long-context scheduling would pull ahead).
2. ____ (e.g., "prefix-share > 50%": SGLang's RadixAttention advantage would dominate).
3. ____ (e.g., "TPOT SLO tighter than 50 ms": the chunked-prefill knob becomes critical, and the framework whose default chunk size is closer wins by accident).
```

---

## What a "good" memo looks like

A passing memo answers all the cells. A good memo also:

- States the SLO **before** showing the numbers, so the reader knows what "goodput@p99" means.
- Reports the knee rate, not just point-in-time numbers — the knee is the operationally relevant capacity.
- Includes the failure-mode section with a specific diagnosed cause and fix.
- Names the *tipping point* (the axis along which the conclusion would flip), not just abstract caveats.

A bad memo:
- Reports only mean latency or only throughput.
- Says "vLLM is faster" or "SGLang is more efficient" with no SLO context.
- Has the failure-mode section as "I ran into some weirdness, not sure why."
- Lists ablation numbers but does not say which knob is operationally important.

---

## How to use the result for actual deployment decisions

The memo's recommendation should be actionable. Three workload classes from the lab translate to three real deployment patterns:

| Workload class | Real-world example | Pick |
|----------------|--------------------|------|
| Short independent chats | Customer support, code completion | Either; pick by ecosystem fit |
| Long shared system prompts / agent loops | Agent frameworks, RAG with fixed instructions | SGLang (RadixAttention) |
| Long context, low concurrency | Document QA, 100 k+ context summarisation | vLLM (broader long-context tuning surface) |

These defaults will be wrong sometimes. The memo's *flip conditions* are how you know when.

---

## Connections

- [[goodput-slo]] — how to define the SLO predicate that makes the goodput column meaningful.
- [[ttft-tpot-itl]] — which metric to report and at which percentile.
- [[sharegpt-workload]] — the workload caveats that limit how generalisable the memo is.
- [[ch-22]] — the capstone uses the same memo discipline ("did your numbers match the paper within 10%?").
