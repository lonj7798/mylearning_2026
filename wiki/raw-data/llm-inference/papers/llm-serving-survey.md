<!-- scope: survey of efficient generative LLM serving from algorithms to systems
     deps: none
     see-also: [[continuous-batching]], [[pagedattention]], [[distserve]]
-->

# Towards Efficient Generative Large Language Model Serving: A Survey from Algorithms to Systems
- **Core Insight:** Efficient LLM serving is a stack problem spanning model algorithms, memory/KV-cache management, scheduling, parallelism, and deployment systems.
- **Guideline:** Use surveys to map optimization families, but return to primary system papers for exact mechanisms and assumptions.
- **Authors:** Xupeng Miao, Gabriele Oliaro, Zhihao Zhang, Xinhao Cheng, Hongyi Jin, Tianqi Chen, Zhihao Jia
- **Year:** 2023
- **URL:** https://arxiv.org/abs/2312.15234
- **Relevant topics:** LLM serving survey, algorithms, systems, inference optimization, scheduling, memory

## Abstract
This survey reviews efficient generative LLM serving from an MLSys perspective. It covers the pressure caused by high compute and memory demands, then organizes techniques across algorithms and systems: model compression, attention and KV-cache optimizations, batching/scheduling, parallelism, and serving frameworks.

## Key Contributions
- Provides a taxonomy for LLM serving optimizations.
- Connects algorithmic methods with systems constraints such as latency and throughput.
- Reviews batching, scheduling, memory, and parallel execution techniques.
- Helps place PagedAttention, Orca-style scheduling, speculative decoding, and compression in one map.
- Identifies open problems around heterogeneous workloads and production constraints.

## Key Figures/Tables to Study
- Taxonomy figure of serving optimization categories.
- Tables summarizing serving systems and algorithmic accelerators.
- Sections on memory/KV-cache optimizations.
- Sections on scheduling and parallelism.

## Technical Details
The survey is useful for course structure because it separates optimizations by where they act: reducing model work, reducing memory movement, improving batching, exploiting parallel hardware, or changing deployment architecture. For this track, the most relevant sections are KV-cache memory management, continuous batching, and prefill/decode scheduling.

The survey should not be treated as a substitute for implementation details. For example, exact block-table semantics come from PagedAttention, and exact iteration-level scheduling semantics come from Orca and serving framework docs.

## Connections
- Overview source for [[pagedattention]], [[orca]], [[sarathi-serve]], [[distserve]], and KV-cache compression papers.
- Useful bridge to benchmark/metrics pages because it frames latency-throughput tradeoffs.

## Notes
This page intentionally summarizes the survey at a taxonomy level; use individual paper pages for implementation details.
