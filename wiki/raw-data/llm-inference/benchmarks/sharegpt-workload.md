<!-- scope: ShareGPT-style request traces for LLM serving benchmarks
     see-also: ttft-tpot-itl, vllm-benchmarks, sglang-benchmarks
-->

# ShareGPT Workload
- **Core Insight:** Realistic serving benchmarks need heavy-tailed prompt and response lengths, not a single fixed sequence length.
- **Guideline:** Use ShareGPT-style traces when evaluating scheduler behavior, but disclose filtering, truncation, tokenizer, and request-rate model.
- **Authors:** ShareGPT dataset community; vLLM and SGLang benchmark maintainers
- **Year:** 2023-2026
- **URL:** https://docs.vllm.ai/en/latest/contributing/benchmarks.html
- **Relevant topics:** workload modeling, prompt length distribution, output length distribution, open-loop load

## Abstract
ShareGPT conversations became a common source for synthetic LLM serving traces because they contain multi-turn user/assistant text with varied input and output lengths. Serving benchmarks typically tokenize the first user prompt and assistant answer, filter by length, and replay the resulting requests under fixed or swept request rates.

## Key Contributions
- Replaces uniform prompt/output sizes with a more realistic length distribution.
- Exercises batching and KV-cache allocation across mixed short and long requests.
- Exposes head-of-line blocking and tail latency under open-loop arrivals.
- Provides a common workload option in vLLM and SGLang benchmark scripts.

## Key Figures/Tables to Study
- vLLM `benchmark_serving.py` dataset modes: shows how ShareGPT prompts are sampled and replayed.
- SGLang benchmark serving docs: comparable use of ShareGPT traces for request-rate experiments.

## Technical Details
The workload is not a standard benchmark by itself. Results depend on the source JSON, tokenizer, prompt/output token filters, random seed, maximum model length, and whether output lengths are replayed or generated to an EOS. Because the traces are text conversations, model-specific chat templates can change input token counts. The workload is best used with TTFT, TPOT, ITL, request latency, and goodput metrics.

## Connections
- [[vllm-benchmarks]] and [[sglang-benchmarks]] use ShareGPT-style serving benchmarks.
- [[goodput-slo]] evaluates how many ShareGPT requests satisfy latency constraints.
