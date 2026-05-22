<!-- scope: raw source library for course/llm-inference
     deps: [[COLLECTION-PLAN]]
     see-also: [[insights]], [[wiki/courses/llm-inference/outline]]
-->

# LLM Inference — Raw Source Library

This directory holds the primary source material the `course/llm-inference` course is built from. Every page here is an **extracted summary** of one artifact: paper, framework module, serving-system design, benchmark, blog, model report, or hardware/runtime note. Course chapters cite these pages via wikilinks.

The course is **systems-first with mathematical grounding**: the first chapters explain how an autoregressive transformer generates text, why prefill and decode have different bottlenecks, how KV cache changes complexity, and how batch scheduling turns a model into a serving system. Frameworks such as `vLLM` and `SGLang` appear after those foundations, so learners understand what PagedAttention, RadixAttention, continuous batching, and prefix caching are solving.

## Scope

End-to-end LLM inference and serving:

- **Generation fundamentals**: autoregressive decoding loop, logits, sampling, greedy/beam/top-k/top-p/min-p/temperature/repetition penalties, stop conditions, streaming, structured outputs, and token accounting.
- **Transformer inference math**: attention, causal masks, positional encodings, MHA/MQA/GQA, prefill vs decode, prompt processing, KV cache memory formulas, batch-size/sequence-length tradeoffs.
- **Serving systems**: continuous batching, iteration-level scheduling, chunked prefill, disaggregated prefill/decode, prefix caching, request queues, admission control, latency SLOs, multi-tenant serving.
- **KV cache**: cache layout, paging, block tables, prefix reuse, radix-tree reuse, eviction, offload, quantization boundary, long-context scaling, context windows.
- **Kernels and runtimes**: FlashAttention, FlashDecoding, PagedAttention kernels, CUDA graphs, NCCL collectives, tensor/pipeline/expert parallel inference, speculative decoding kernels.
- **Frameworks**: `vLLM`, `SGLang`, TensorRT-LLM, Hugging Face TGI, llama.cpp/llama-server, LightLLM, LMDeploy, DeepSpeed-FastGen, Ray Serve / KServe integration.
- **Acceleration methods**: speculative decoding, assisted generation, Medusa, EAGLE, n-gram speculation, prompt lookup, multi-token prediction, early exit, draft-model selection.
- **Benchmarks and metrics**: TTFT, TPOT, ITL, throughput, goodput, request-rate sweeps, ShareGPT-style workloads, GenAI-Perf, LLMPerf, MLPerf Inference, HELM serving metrics.
- **Production reports**: inference details from Llama, Qwen, DeepSeek, Gemini, GPT-OSS, Claude/OpenAI disclosures when public, and major serving-platform case studies.

## Directory layout

```
raw-data/llm-inference/
├── README.md             this file
├── COLLECTION-PLAN.md    master topic checklist + source targets
├── insights.md           aggregated core-insights index
├── classics/             foundational transformer/generation papers and formulas
├── papers/               arxiv + conference papers
├── frameworks/           code-level framework summaries
├── systems/              serving architecture and platform summaries
├── benchmarks/           benchmark suites and metric methodology
├── model-reports/        frontier-model reports with inference details
├── blogs/                practitioner posts, vendor notes, tutorials
└── labs/                 labs / projects / maintainers by capability
```

## File-naming convention

- Slug-cased, no prefixes: `pagedattention.md`, `continuous-batching.md`, `speculative-decoding.md`.
- One artifact per file. Framework code goes in `frameworks/<framework>-<module>.md`.
- Runtime/system design pages go in `systems/` unless they are tied to one codebase.

## File format (required for every source page)

```markdown
<!-- scope: one-line description of what this source covers
     deps: prereq-source (optional)
     see-also: related-source
-->

# <Artifact title>
- **Core Insight:** one sentence — the thing this source is famous for
- **Guideline:** one sentence — what a practitioner should actually do
- **Authors:** ...
- **Year:** ...
- **URL:** ...
- **Relevant topics:** ...

## Abstract
(for papers) faithful paraphrase; for framework/system pages, concise summary

## Key Contributions
- 3–6 bullets

## Key Figures/Tables to Study
- which figure/table/code path + one-line why

## Technical Details
(varies by source type: for generation include loop/pseudocode and logits transforms;
for KV cache include memory formula, layout, block size, eviction/prefix rules;
for schedulers include queueing policy, batch construction, prefill/decode treatment;
for frameworks include public APIs, relevant modules, and runtime architecture)

## Connections
- where this connects to other sources
```

## How this library is used

1. **Planner** reads `COLLECTION-PLAN.md` + this library to decide chapter granularity.
2. **Course chapters** (`wiki/courses/llm-inference/ch-*/read.md`) cite these pages via wikilinks and lift equations, diagrams, and code paths from them.
3. **Insights index** (`insights.md`) summarizes the recurring design principles once the first collection pass is complete.

Do not rewrite source pages to fit a chapter narrative. If a source's interpretation changes during course writing, add a `Notes` section instead of replacing the extract.
