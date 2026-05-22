<!-- scope: Quest paper on query-aware sparse KV-cache page selection
     deps: [[pagedattention]]
     see-also: [[snapkv]], [[h2o]], [[attention-sinks]]
-->

# Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference
- **Core Insight:** Token importance depends on the current query, so long-context inference should select KV cache pages dynamically rather than using only query-agnostic eviction.
- **Guideline:** For long contexts, page KV cache and load only query-relevant pages when attention bandwidth dominates latency.
- **Authors:** Jiaming Tang, Yilong Zhao, Kan Zhu, Guangxuan Xiao, Baris Kasikci, Song Han
- **Year:** 2024
- **URL:** https://arxiv.org/abs/2406.10774
- **Relevant topics:** query-aware sparsity, KV-cache pages, long context, sparse attention, memory bandwidth

## Abstract
Quest targets long-context LLM inference where attention slows down because each step must load a large KV cache. Prior work shows attention depends on a subset of critical tokens, but Quest observes that criticality depends on the current query. It tracks summary statistics for KV pages and selects the most relevant pages for each query, reducing KV bandwidth while preserving accuracy.

## Key Contributions
- Introduces query-aware KV-cache selection for long-context inference.
- Uses page-level metadata, including min/max key values, to estimate relevance.
- Loads only selected KV pages for attention computation.
- Aligns naturally with paged KV-cache layouts.
- Demonstrates latency gains with small quality loss on long-context workloads.

## Key Figures/Tables to Study
- Query-dependent token criticality evidence.
- Quest page metadata and selection pipeline.
- Latency breakdown showing KV bandwidth reduction.
- Accuracy/latency curves under different page budgets.

## Technical Details
Quest groups KV cache into pages and maintains compact statistics per page. Given a query vector, it estimates which pages may contain keys with high attention scores, then computes attention over selected pages. This avoids reading the full KV cache at every decode step.

The method differs from permanent eviction: pages can remain stored but not loaded for a given query. This makes it attractive for systems with paged cache managers, because sparse page access becomes a scheduling/kernel problem rather than only a compression policy.

## Connections
- Uses a page concept related to [[pagedattention]], but for sparse selection rather than allocation alone.
- [[snapkv]] compresses prompt KV once after prefill; Quest selects pages per query.
- [[h2o]] motivates attention sparsity but uses query-agnostic heavy hitters.

## Notes
Quest is a kernel and cache-access problem as much as a pruning policy because selected pages must be loaded efficiently.
