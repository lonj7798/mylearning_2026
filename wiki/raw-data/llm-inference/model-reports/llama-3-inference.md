<!-- scope: inference-relevant facts from Meta Llama 3 and Llama 3.1 model reports
     see-also: vllm-project, nvidia-inference
-->

# Llama 3 Inference
- **Core Insight:** Llama 3 popularized high-quality dense decoder-only open models with grouped-query attention as a default inference-efficient architecture.
- **Guideline:** Treat Llama 3/3.1 as a baseline dense GQA serving target; size KV cache from context length, layer count, hidden size, and KV head count.
- **Authors:** Meta
- **Year:** 2024
- **URL:** https://ai.meta.com/blog/meta-llama-3/
- **Relevant topics:** dense transformer, GQA, RoPE, 8k and 128k context, BF16/quantization

## Abstract
Llama 3 models are decoder-only transformer LLMs released in 8B and 70B sizes, followed by Llama 3.1 with 8B, 70B, and 405B variants. The family uses grouped-query attention for inference efficiency, a large tokenizer vocabulary, RoPE-style positional encoding, and chat-tuned variants. Llama 3.1 extends the supported context length to 128k.

## Key Contributions
- Provides strong open dense-model baselines for serving systems.
- Uses GQA to reduce KV-cache size relative to full multi-head attention.
- Supports common runtimes including vLLM, TensorRT-LLM, TGI, llama.cpp variants, and cloud endpoints.
- Llama 3.1 adds long-context serving pressure with 128k context.

## Key Figures/Tables to Study
- Meta Llama 3 model card: architecture, context, and intended use.
- Llama 3.1 report/model cards: 128k context and 405B deployment constraints.

## Technical Details
Inference bottlenecks differ by size: 8B is often memory-bandwidth and batching friendly, 70B commonly needs tensor parallelism or quantization, and 405B requires multi-node or highly optimized serving. GQA lowers KV memory by sharing key/value heads across query-head groups. For long context, prefill cost and KV-cache residency dominate scheduler and memory design.

## Connections
- [[vllm-project]] and [[tensorrt-llm-docs]] commonly benchmark Llama models.
- [[mlperf-inference-llm]] uses Llama-family tasks for standardized system comparisons.
