<!-- scope: inference-relevant facts from Microsoft Phi model reports and model cards
     see-also: huggingface-inference, llama-cpp
-->

# Phi Inference
- **Core Insight:** Phi models target high capability at small parameter counts, making local and low-latency inference a primary use case.
- **Guideline:** For Phi serving, check the exact generation and variant because context length, modality, license, and chat template vary substantially.
- **Authors:** Microsoft
- **Year:** 2023-2025
- **URL:** https://huggingface.co/microsoft
- **Relevant topics:** small language model, dense transformer, long context, local inference, ONNX/DirectML

## Abstract
Microsoft's Phi family includes compact dense language models such as Phi-2, Phi-3, Phi-3.5, and Phi-4 variants. The family is inference-relevant because many checkpoints are designed for local, edge, or latency-sensitive use while preserving strong reasoning and coding performance for their size.

## Key Contributions
- Establishes small language models as serious serving targets.
- Provides variants with long-context support relative to parameter count.
- Ships through Hugging Face and Microsoft tooling with quantized/local deployment paths.
- Includes multimodal and mini variants, so runtime requirements differ by checkpoint.

## Key Figures/Tables to Study
- Microsoft model cards on Hugging Face: context length, dtype, chat template, and intended use.
- Phi technical reports: architecture and benchmark tables by model size.
- ONNX/AI Toolkit docs: local deployment examples.

## Technical Details
Phi checkpoints are generally dense, so serving is simpler than MoE models. Their small size enables high concurrency on a single GPU or CPU/GPU edge deployment, but long-context variants can still become KV-cache-bound. For production comparisons, normalize by tokenizer, prompt template, and max output tokens because small models are often used with tighter latency targets.

## Connections
- [[llama-cpp]] and [[huggingface-inference]] are common deployment routes.
- [[ttft-tpot-itl]] still applies because small models are often optimized for first-token latency.
