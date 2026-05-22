<!-- scope: Character.AI inference deployment with INT8 + KV quantization
     deps: [[int8]], [[fp8-e4m3]]
     see-also: [[vllm-quant]], [[transformer-engine-blog]]
-->

# Character.AI — Optimizing AI Inference at a Character.AI Scale
- **Core Insight:** Character.AI's June 2024 deployment post showed that aggressive INT8 weight + INT8 activation + INT8 KV quantization paired with multi-query attention and cross-layer KV-cache sharing cut serving cost per query by ~33× compared to a baseline FP16 deployment, enabling 20k QPS at consumer-affordable margins.
- **Guideline:** When KV cache dominates memory at production scale, attack it first — INT8 KV + MQA + cross-layer KV reuse compound multiplicatively, often saving more than weight quantization alone.
- **Authors:** Character.AI engineering team (Noam Shazeer's team)
- **Year:** 2024
- **URL:** https://research.character.ai/optimizing-inference/
- **Relevant topics:** INT8 KV cache, MQA, cross-layer KV, production inference, cost engineering

## Summary
Character.AI's "Optimizing AI Inference" post is one of the most-cited production deployment write-ups in the LLM inference space. The company runs >20,000 queries-per-second of inference globally and was cost-bound by KV cache memory at the prefill+generate join. Their solution stack: (1) multi-query attention (MQA) reduces KV cache by num_heads factor; (2) cross-layer KV sharing reduces by additional 2-3× depending on grouping; (3) INT8 weight + activation quantization halves weight-memory bandwidth; (4) INT8 KV cache halves KV-memory bandwidth. The compounded effect is ~33× lower cost per query versus a naive FP16 baseline. The blog also discusses sliding-window attention (only attend to recent N tokens) as a fallback for long contexts.

## Key Points
- KV cache, not weights, is the production-scale bottleneck.
- MQA: all heads share one K and one V → KV memory / num_heads.
- Cross-layer KV sharing: groups of consecutive layers share a single KV cache.
- INT8 weights + INT8 activations: half-precision matmul bandwidth.
- INT8 KV cache: half the cache footprint and 2× attention bandwidth.
- Combined: ~33× cost reduction vs FP16 baseline at Character's scale.

## Technical Details

### The four optimizations

#### 1. Multi-Query Attention (MQA)
- Standard GQA / MHA: each head has its own K, V.
- MQA: all heads share a single K and single V.
- Saves: KV memory / num_heads (typically 32-64×).
- Cost: small accuracy hit; mitigated by fine-tuning after architectural change.

#### 2. Cross-layer KV sharing
- Group consecutive transformer layers; only one set of KV per group.
- Layers within a group reuse the same cached KV.
- Saves: additional 2-3× depending on group size.
- Cost: another small accuracy hit; jointly tuned with MQA.

#### 3. INT8 weight + activation
- Per-channel weight quant, per-token activation quant (similar to SmoothQuant).
- Halves bandwidth for the W·X matmul.
- Custom CUDA kernels at production-deployment time (not open-sourced).

#### 4. INT8 KV cache
- Per-token INT8 scaling for both K and V.
- Halves cache memory; halves the bandwidth of the attention QKᵀ + AV matmuls.
- Critical for the long-context regime where KV memory exceeds weight memory.

### Multiplicative effect
| Optimization | Cost reduction | Cumulative |
|--------------|----------------|------------|
| Baseline FP16 | 1× | 1× |
| + MQA | 4× | 4× |
| + cross-layer KV (group=4) | 2× | 8× |
| + INT8 weights/acts | 2× | 16× |
| + INT8 KV | 2× | 32× |

(Approximate; exact numbers depend on context length and batch size.)

### Sliding-window attention
- For contexts > sliding window, only attend to most recent N tokens.
- Trades retention vs latency; suitable for chat / agent dialogue but not long-document QA.
- Compounds with cross-layer KV sharing.

### Why this post is influential
- Shows that the prevailing focus on weight quantization (GPTQ, AWQ) misses the actual bottleneck at production scale.
- Concretizes the KV-cache cost story with real numbers from real serving.
- Inspires subsequent open-source work on KV quant (KIVI, KVQuant, GEAR) and KV sharing (cross-layer attention).

### What's missing from the post
- No open-source code release.
- Doesn't disclose exact per-stage perplexity / win-rate tradeoff numbers.
- Hardware fleet composition is not detailed.

## Connections
- [[kivi]] / [[kvquant]] / [[gear]] — open-source successors implementing the KV-quant idea.
- [[int8]] — KV cache and weight format.
- [[vllm-quant]] — open-source serving with similar (less aggressive) KV quant options.
- [[transformer-engine-blog]] — FP8 alternative to INT8 path for similar use.
