<!-- scope: vLLM V1 scheduler code path and serving policy
     deps: [[vllm]]
     see-also: [[vllm-kv-cache-manager]], [[continuous-batching]]
-->

# vLLM Scheduler
- **Core Insight:** vLLM's scheduler converts many asynchronous requests into step-sized model work while enforcing token, KV-cache, and modality budgets.
- **Guideline:** Study the scheduler when tuning TTFT/TPOT tradeoffs, chunked prefill, prefix caching, preemption, or request admission.
- **Authors:** vLLM project
- **Year:** 2024-present V1 engine
- **URL:** https://docs.vllm.ai/en/latest/api/vllm/v1/core/sched/scheduler/ and https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py
- **Relevant topics:** continuous batching, chunked prefill, preemption, prefix-cache-aware scheduling, encoder cache, structured output bitmasks

## Abstract
The V1 scheduler is the central policy module that decides what each engine step computes. It maintains waiting and running requests, computes token budgets, allocates KV-cache slots through the KV cache manager, handles preemption when cache pressure is high, and emits a `SchedulerOutput` consumed by workers/model runners.

## Key Contributions
- Performs iteration-level scheduling rather than forming static batches for complete requests.
- Schedules both prefill and decode work under a maximum scheduled-token budget.
- Integrates KV-cache allocation directly into admission decisions.
- Coordinates with prefix caching, encoder cache, speculative decoding, and structured-output managers.
- Emits explicit scheduler outputs so worker execution can stay mostly policy-free.

## Key Figures/Tables to Study
- `Scheduler.schedule()`: main loop for deciding running work and newly admitted work.
- `SchedulerOutput`: contract between scheduler and model execution.
- Calls to `kv_cache_manager.allocate_slots(...)`: where token scheduling becomes cache allocation.
- Preemption paths: how running requests may be paused to free cache.

## Technical Details
Public/serving entrypoint:
- Users do not call the scheduler directly; it is reached through `vllm serve`, the OpenAI server, or the offline engine.
- Main configuration flows from `VllmConfig`, `SchedulerConfig`, and server CLI arguments.

Scheduling approach:
- Keep a `waiting` queue for not-yet-running requests and a `running` collection for active requests.
- Each step chooses how many tokens to compute per request, bounded by global token budget and per-request remaining work.
- Decode requests usually need one new token; prefill requests may need many prompt tokens and can be chunked.
- Before scheduling tokens, the scheduler asks whether KV-cache slots are available.
- If cache allocation fails, vLLM can preempt lower-priority/running work and retry.

Cache and other managers:
- `KVCacheManager` supplies block IDs and computed-prefix information.
- `EncoderCacheManager` handles encoder-side cache for encoder-decoder or multimodal cases.
- Structured output constraints can add grammar bitmasks to scheduler output.
- KV connector outputs allow remote/disaggregated KV movement to affect scheduling state.

Relevant code/docs:
- Scheduler source: https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py
- Scheduler package: https://github.com/vllm-project/vllm/tree/main/vllm/v1/core/sched
- API docs: https://docs.vllm.ai/en/latest/api/vllm/v1/core/sched/scheduler/

Strengths:
- Integrates admission control with exact KV-cache availability rather than guessing from request count.
- Chunked prefill prevents a single large prompt from monopolizing a step.
- Explicit outputs make it easier to reason about model-runner inputs and cache tables.

Limitations:
- Tuning is workload-specific; prefill-heavy, decode-heavy, and mixed traffic want different policies.
- Preemption protects liveness under cache pressure but can increase tail latency.
- Reading behavior from docs alone is insufficient because the scheduler is under active development.

## Connections
- Depends on [[vllm-kv-cache-manager]] for block allocation and prefix reuse.
- Implements [[continuous-batching]] in a production framework.
- Compare with [[sglang-scheduler]], [[tensorrt-llm]], and [[hf-tgi]].
