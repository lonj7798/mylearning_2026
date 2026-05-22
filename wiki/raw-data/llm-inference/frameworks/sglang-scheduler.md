<!-- scope: SGLang runtime scheduler and request batching source page
     deps: [[sglang]]
     see-also: [[sglang-radixattention]], [[vllm-scheduler]]
-->

# SGLang Scheduler
- **Core Insight:** SGLang's scheduler builds continuous batches while using radix-cache hits, token budgets, and memory availability to reduce prefill cost.
- **Guideline:** Tune scheduler flags jointly with radix cache and chunked prefill; scheduling is where latency, throughput, and cache locality collide.
- **Authors:** SGLang project
- **Year:** 2023-present
- **URL:** https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/scheduler.py
- **Relevant topics:** continuous batching, FCFS/cache-aware policy, chunked prefill, radix cache, request queues

## Abstract
The SGLang runtime scheduler owns the core lifecycle of generation requests after frontend parsing. It receives requests from server/API layers, maintains waiting and running queues, uses memory/cache managers to estimate and allocate capacity, forms batches for model workers, updates request state after each step, and coordinates constrained decoding and metrics.

## Key Contributions
- Implements runtime scheduling for both prefill and decode phases.
- Integrates RadixAttention prefix matching into request preparation and memory accounting.
- Provides policy flags such as `schedule_policy`, `schedule_conservativeness`, `max_running_requests`, `max_total_tokens`, `chunked_prefill_size`, and `max_prefill_tokens`.
- Coordinates with grammar backends, LoRA state, speculative decoding, metrics, and distributed modes.
- Supports advanced serving modes such as prefill/decode disaggregation and overlap scheduling.

## Key Figures/Tables to Study
- `scheduler.py`: main event loop and batch construction.
- Server args emitted in docs examples: default scheduler-related flags.
- `schedule_policy`: where FCFS or cache-aware behavior is selected.
- Batch preparation paths: how prefill/decode and radix-cache metadata enter model execution.

## Technical Details
Public/serving entrypoint:
- Users launch `python -m sglang.launch_server`; scheduler settings are supplied as server arguments.
- Requests arrive through OpenAI-compatible endpoints or SGLang's native `/generate` API.

Scheduling approach:
- Maintain queued requests and active running requests.
- Estimate token and memory pressure before admitting new requests.
- Use prefix-cache hits from the radix cache to avoid recomputing shared prompt tokens.
- Chunk large prefills so decode traffic can continue and TTFT remains bounded.
- Continue decode steps for running requests while admitting new prefills when budgets allow.

Relevant code/docs:
- Scheduler source: https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/scheduler.py
- Runtime managers directory: https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/managers
- Server args docs: https://docs.sglang.ai/advanced_features/server_arguments.html
- Hyperparameter tuning: https://docs.sglang.ai/advanced_features/hyperparameter_tuning.html

Strengths:
- Cache-aware scheduling can improve TTFT for shared-prefix workloads.
- Rich control over request limits and token budgets.
- Designed with production features, not just single-node batch formation.

Limitations:
- Many policy knobs are coupled; changing one limit can expose a different bottleneck.
- Cache-aware choices can improve aggregate throughput while shifting fairness between requests.
- Exact behavior is best learned from code because release docs focus more on user flags than algorithm walkthroughs.

## Connections
- Uses [[sglang-radixattention]] for prefix-cache locality.
- Compare with [[vllm-scheduler]] and [[tensorrt-llm]] in-flight batching.
- Scheduler decisions determine whether structured decoding overhead in [[sglang-structured-output]] is visible in tail latency.
