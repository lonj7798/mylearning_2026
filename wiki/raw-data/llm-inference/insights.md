<!-- scope: aggregated cross-source insights for the llm-inference raw library
     deps: [[README]], [[COLLECTION-PLAN]]
     see-also: (filled after pass 1)
-->

# LLM Inference — Insights Index

This page is the cross-source map for the raw library. It is built after the first collection pass and updated as the course outline stabilizes.

> Status: first-pass skeleton plus collection notes. Re-populate with full synthesis during course outlining.

## 1. Generation Loop

(Will summarize autoregressive generation, prefill/decode separation, logits processing, sampling, streaming, stop rules, and structured-output constraints.)

## 2. KV Cache

(Will summarize KV cache memory formulas, layout choices, PagedAttention, prefix caching, RadixAttention, eviction, offload, and long-context consequences.)

## 3. Scheduling

(Will summarize continuous batching, iteration-level scheduling, chunked prefill, prefill/decode interference, disaggregation, goodput, and SLO-aware admission control.)

## 4. Frameworks

(Will compare vLLM, SGLang, TensorRT-LLM, TGI, llama.cpp, LightLLM, LMDeploy, and DeepSpeed-FastGen by scheduler, cache manager, kernel backend, quantization support, and structured-output support.)

## 5. Decoding Acceleration

(Will summarize speculative decoding, Medusa, EAGLE, prompt lookup, n-gram speculation, multi-token prediction, and when these help or hurt.)

## 6. Metrics

(Will summarize TTFT, TPOT, ITL, throughput, request-rate sweeps, goodput, cache-hit rate, and benchmark workload design.)

## 7. Open Gaps

- First pass collected fundamentals, KV cache, scheduling, kernels, decoding acceleration, frameworks, benchmarks, model reports, labs, and practitioner docs.
- Mandatory topics called out by the learner are covered: [[vllm]], [[sglang]], [[pagedattention]], [[sglang-radixattention]], [[kv-cache-memory-formula]], [[prefill-vs-decode]], [[flashattention]], [[flashattention-2]], [[flashattention-3]], [[flashdecoding]], and [[flashinfer]].
- Some pages are intentionally synthesis cards rather than single artifacts: [[continuous-batching]], [[prefill-decode-disaggregation]], [[admission-control-goodput]], [[structured-generation-constrained-decoding]], and the parallelism pages.
