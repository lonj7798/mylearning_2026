<!-- scope: Hugging Face inference serving projects and APIs
     see-also: hf-llm-inference-optimization, gemma-inference, phi-inference
-->

# Hugging Face Inference
- **Core Insight:** Hugging Face provides both model distribution and serving stacks, making model cards, tokenizers, chat templates, and runtime APIs part of inference correctness.
- **Guideline:** Treat the Hugging Face model card and tokenizer config as runtime inputs, not just documentation.
- **Authors:** Hugging Face
- **Year:** 2020-2026
- **URL:** https://huggingface.co/docs/text-generation-inference/
- **Relevant topics:** Transformers, Text Generation Inference, chat templates, model cards, quantization

## Abstract
Hugging Face is a central distribution and deployment ecosystem for LLMs. Inference paths include Transformers generation, Text Generation Inference (TGI), hosted Inference Endpoints, Optimum, quantization integrations, and model-specific chat templates. The ecosystem is especially important because many official model cards disclose context length, dtype, generation examples, and serving caveats on Hugging Face.

## Key Contributions
- Standardizes model packaging, tokenizer files, and model cards.
- Provides TGI as a production text-generation server.
- Supports quantization and hardware-specific integrations through the broader ecosystem.
- Makes chat templates a first-class part of correct inference.

## Key Figures/Tables to Study
- TGI documentation: continuous batching, streaming, tensor parallelism, and supported models.
- Transformers generation docs: logits processors, stopping, sampling, and caches.
- Model cards: exact prompt format and dtype/runtime notes.

## Technical Details
Transformers is the reference Python API but may not be the fastest serving path. TGI provides server-level batching, streaming, and deployment controls. Chat templates can materially change prompt tokens and output behavior. Benchmarking Hugging Face-hosted or local endpoints should pin revision, tokenizer, dtype, quantization, and generation config.

## Connections
- [[hf-llm-inference-optimization]] summarizes Hugging Face optimization guidance.
- [[gemma-inference]] and [[phi-inference]] often rely on Hugging Face model cards.
