<!-- scope: Static, dynamic, and continuous batching for LLM serving
     deps: [[prefill-vs-decode]], [[kv-cache-memory-formula]]
     see-also: [[multi-query-attention]], [[grouped-query-attention]], [[openai-streaming-and-token-usage]]
-->

# Batching for Inference
- **Core Insight:** LLM serving throughput depends on keeping GPUs busy across many variable-length requests while managing each request's growing KV cache.
- **Guideline:** Prefer iteration-level or continuous batching for online serving; static request-level batches waste compute when outputs have different lengths.
- **Authors:** Gyeong-In Yu et al. (Orca); Woosuk Kwon et al. (vLLM/PagedAttention); Hugging Face Transformers docs
- **Year:** 2022-2026
- **URL:** https://www.usenix.org/conference/osdi22/presentation/yu ; https://arxiv.org/abs/2309.06180 ; https://huggingface.co/docs/transformers/continuous_batching
- **Relevant topics:** batching, continuous batching, iteration-level scheduling, PagedAttention, serving throughput

## Abstract
Autoregressive generation creates variable-length workloads: each request may finish at a different decode step, and new requests arrive continuously. Orca introduced iteration-level scheduling to batch at each generation iteration. vLLM combined high-throughput batching with PagedAttention to reduce KV-cache fragmentation. Hugging Face now exposes continuous batching APIs in Transformers.

## Key Contributions
- Orca: schedules at iteration granularity rather than waiting for whole requests to finish.
- Orca: selective batching handles operations with different shapes across requests.
- vLLM: PagedAttention stores KV cache in blocks to reduce fragmentation and enable larger batches.
- Continuous batching: removes finished requests and admits waiting requests between generation steps.
- Modern serving: batch size is constrained by KV memory, latency SLOs, and prompt/output length distributions.

## Key Figures/Tables to Study
- **Orca scheduling diagrams:** Request-level vs iteration-level batching.
- **vLLM memory diagrams:** Logical KV blocks mapped to physical blocks.
- **Throughput/latency graphs:** Batch size improves throughput until memory or latency limits bind.
- **HF continuous batching docs:** `ContinuousBatchingManager` and `generate_batch` API flow.

## Technical Details
Static batching:

```text
collect N requests -> run until all finish -> return
```

Continuous batching:

```text
each decode iteration:
  remove finished sequences
  admit queued sequences if memory allows
  run one model step for active batch
```

The scheduler must track per-sequence state: prompt progress, generated tokens, EOS status, sampling parameters, and KV-cache block allocation. Larger batches improve GPU utilization, but each active sequence consumes KV memory proportional to its current context length.

## Connections
- [[prefill-vs-decode]]: good schedulers decide how to mix prefill chunks and decode steps.
- [[kv-cache-memory-formula]]: memory, not arithmetic, often caps active batch size.
- [[multi-query-attention]] and [[grouped-query-attention]]: smaller KV tensors allow more active sequences.
- [[openai-streaming-and-token-usage]]: streaming responses are usually backed by token-by-token continuous serving.
