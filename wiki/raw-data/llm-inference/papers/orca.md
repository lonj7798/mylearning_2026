<!-- scope: OSDI 2022 serving system introducing iteration-level scheduling and selective batching
     deps: transformer-generation
     see-also: [[continuous-batching]], [[pagedattention]], [[vtc]]
-->

# Orca: A Distributed Serving System for Transformer-Based Generative Models
- **Core Insight:** Autoregressive serving should schedule each generation iteration, not whole requests, so completed requests can leave and new requests can join immediately.
- **Guideline:** For online LLM serving, prefer iteration-level scheduling with operation-aware batching over static request-level batches.
- **Authors:** Gyeong-In Yu, Joo Seong Jeong, Geon-Woo Kim, Soojeong Kim, Byung-Gon Chun
- **Year:** 2022
- **URL:** https://www.usenix.org/conference/osdi22/presentation/yu
- **Relevant topics:** continuous batching, iteration-level scheduling, selective batching, distributed serving, pipeline/model parallelism

## Abstract
Orca targets Transformer generative models whose requests run for many autoregressive iterations. Conventional inference servers batch entire requests, forcing short outputs to wait for long ones and preventing new arrivals from joining until the whole batch finishes. Orca introduces iteration-level scheduling, where the scheduler invokes one model iteration at a time over the currently active requests. It also introduces selective batching, because not every operation in a generation loop can be batched the same way across requests with different states.

## Key Contributions
- Defines the multi-iteration serving problem for generative Transformer workloads.
- Introduces iteration-level scheduling, the conceptual basis of continuous batching.
- Uses selective batching to batch compatible operations while handling per-request control flow.
- Supports distributed execution for very large models, including models with hundreds of billions of parameters.
- Demonstrates large throughput improvement over FasterTransformer on GPT-3-scale models.

## Key Figures/Tables to Study
- Figure 1: request-level batching pathologies for variable output lengths.
- Architecture figure: scheduler, execution engine, and distributed workers.
- Selective batching diagrams: which operators can share a batch and which remain request-specific.
- Evaluation on GPT-3 175B: latency-throughput curves versus FasterTransformer.

## Technical Details
Orca separates the serving loop into repeated iterations. At each iteration the scheduler chooses active requests, invokes the execution engine for one token step, returns completed requests, and fills open slots with queued requests. This avoids head-of-line blocking from long generations.

Selective batching handles the fact that generative inference includes both shared dense tensor operations and request-specific logic such as sampling, stopping, and sequence bookkeeping. Orca batches operations where tensor shapes and semantics align, and keeps other work separate.

The paper predates PagedAttention, so KV-cache management is less flexible than modern vLLM-style paged allocators. Its key scheduler idea still underlies vLLM, TGI, TensorRT-LLM in-flight batching, and many later fairness/SLO schedulers.

## Connections
- [[continuous-batching]] is the broader practitioner term for Orca's iteration-level scheduling.
- [[pagedattention]] removes a major memory-management bottleneck left by dynamic scheduling.
- [[vtc]] adapts continuous batching to multi-tenant fairness.
