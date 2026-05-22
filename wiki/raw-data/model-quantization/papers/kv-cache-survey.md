<!-- scope: dedicated KV-cache compression survey covering quantization, sparsity, eviction
     deps: [[kivi]], [[kvquant]], [[gear]]
     see-also: [[per-channel-vs-per-token-kv]], [[skvq]], [[wkvquant]]
-->

# Keep the Cost Down: A Review on Methods to Optimize LLM's KV-Cache Consumption
- **Core Insight:** KV-cache optimisation splits into three axes — *quantization* (low-bit representation), *sparsity / eviction* (drop tokens), and *architectural changes* (MQA/GQA, latent attention) — and the optimal recipe combines all three rather than relying on any single dimension.
- **Guideline:** When designing a long-context serving system, allocate a memory budget across the three KV-compression axes simultaneously: e.g. 4-bit (quant) + 50% retention (eviction) + GQA (architectural) = 16× cumulative reduction with bounded quality loss.
- **Authors:** Luohe Shi, Hongyi Zhang, Yao Yao, Zuchao Li, Hai Zhao
- **Year:** 2024 (COLM 2024)
- **URL:** https://arxiv.org/abs/2407.18003
- **Relevant topics:** KV-cache survey, quantization, eviction, sparsity, architectural compression

## Abstract
This review surveys methods to reduce KV-cache memory consumption in LLMs, organised across three phases: pre-training (architectural — MQA/GQA, multi-latent), deployment (post-training — quantization, low-rank decomposition), and inference (runtime — eviction, sliding window). It provides evaluation metrics for long-context capability (memory, throughput, accuracy on RULER / LongBench) and curates a paper repository. The paper's organising contribution is the three-axis taxonomy that makes the compression budget explicit.

## Key Contributions
- Three-axis taxonomy: pre-train architectural / deployment quantization / inference eviction.
- Evaluation framework for "long-context capability" combining efficiency (memory, throughput) and quality (LongBench / RULER scores) into a single Pareto view.
- Curated paper repository (GitHub) tracking the rapidly-evolving KV-cache literature.
- Identifies open problems: coupling between axes (does quantization hurt eviction policies?), dynamic-vs-static bit allocation, cross-batch sharing.

## Key Figures/Tables to Study
- The three-axis taxonomy table: where each method (KIVI / KVQuant / H2O / StreamingLLM / GQA) sits.
- Memory-vs-accuracy Pareto plot: 2-bit + eviction outperforms 4-bit + no eviction at the same memory.
- Long-context benchmark summary at iso-memory.

## Technical Details

### Axis 1 — Architectural (pre-training)
- **MQA (Multi-Query Attention):** all heads share one K and one V → 32× KV reduction for 32-head model.
- **GQA (Grouped-Query):** g heads share one K/V pair → factor-g reduction; LLaMA-3/2-70B use GQA-8.
- **Multi-Latent Attention (MLA):** project K/V into a low-dimensional latent → 8–16× reduction; DeepSeek V2/V3.
- Trades: architectural changes require pretraining; not retroactively applicable to existing FP models.

### Axis 2 — Deployment (post-training)
- **Quantization:** the bulk of this library's bucket 11. KIVI / KVQuant / GEAR / SKVQ / QAQ / WKVQuant.
- **Low-rank decomposition:** GEAR's residual; LESS factorization.
- **Distillation into smaller K/V:** DistillKV-style, trains a compact replacement KV head.

### Axis 3 — Inference (runtime)
- **Eviction:** H2O (heavy-hitter eviction), Scissorhands, StreamingLLM (sink + recent window).
- **Sliding window:** Mistral's SWA, Longformer-style fixed-size context.
- **Block-sparse retrieval:** retrieve only top-K blocks per query (Infinite Attention).

### Three-axis combination (example budget)
Target: serve 7B LLM with 1M context on a single 80GB GPU.
- Axis 1: GQA-8 (8× from architecture).
- Axis 2: KVQuant at 2-bit (8× from quant).
- Axis 3: StreamingLLM with 4K sink + 4K recent (variable from eviction).
Combined: > 100× compression vs full FP16 KV, enabling the target context.

### Coupling concerns (open problem)
- Quantization noise affects eviction's heavy-hitter detection — H2O scores on quantized K may differ from FP K.
- Architectural reduction (GQA) reduces redundancy that quant relied on for averaging out noise.
- Sliding window changes the relative importance of channel-wise vs token-wise outliers.

The survey flags this coupling as an under-studied frontier.

## Connections
- Quant-side papers covered: [[kivi]], [[kvquant]], [[gear]], [[wkvquant]], [[qaq]], [[skvq]], [[coupling-kv-quant]].
- Per-channel vs per-token analytical companion: [[per-channel-vs-per-token-kv]].
- Architectural compression (out of scope but referenced): MQA, GQA, MLA.
- Eviction (out of scope but referenced): H2O, StreamingLLM, Scissorhands.
