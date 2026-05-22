<!-- scope: inference-relevant facts from Google Gemma model reports
     see-also: hf-llm-inference-optimization, llama-cpp
-->

# Gemma Inference
- **Core Insight:** Gemma provides compact open dense models whose deployment profile is shaped by GQA/sliding-window attention choices and edge-friendly sizes.
- **Guideline:** Select the exact Gemma generation because context length and attention pattern differ across Gemma, Gemma 2, and Gemma 3.
- **Authors:** Google DeepMind
- **Year:** 2024-2025
- **URL:** https://ai.google.dev/gemma/docs
- **Relevant topics:** dense transformer, GQA, sliding window attention, local deployment, quantization

## Abstract
Gemma is Google's open model family derived from Gemini research and released in multiple sizes and generations. Inference-relevant traits include compact dense checkpoints, grouped-query attention in later models, sliding/local attention patterns in some variants, and broad support in Hugging Face, Google AI tooling, and local runtimes.

## Key Contributions
- Supplies small and mid-size dense models for local and server inference.
- Uses architecture choices that reduce KV/cache and attention cost in later releases.
- Offers instruction-tuned variants with documented chat formatting.
- Has broad quantization and runtime support for edge deployment.

## Key Figures/Tables to Study
- Gemma docs model table: size, context length, modality, and recommended use.
- Technical reports for Gemma 2/3: attention pattern, GQA, and long-context details.
- Hugging Face model cards: dtype, tokenizer, and chat template.

## Technical Details
Gemma serving is often memory-limited on consumer GPUs or edge accelerators, so quantized weights and KV-cache size matter. Sliding-window attention can cap attention computation for part of the context but changes how cache eviction and long-context behavior should be understood. Always verify whether the selected checkpoint is base, instruction-tuned, multimodal, or text-only.

## Connections
- [[llama-cpp]] is a common local-runtime path for quantized Gemma variants.
- [[huggingface-inference]] covers hosted and Transformers-based serving.
