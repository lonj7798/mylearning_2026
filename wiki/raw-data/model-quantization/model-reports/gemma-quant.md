<!-- scope: Google Gemma official quantization (Gemma 2 / Gemma 3 QAT)
     deps: [[gguf-k-quants]], [[awq]]
     see-also: [[llama-3-quantization]], [[qwen-2.5-quant]]
-->

# Gemma 2/3 Official Quantization (Google QAT)
- **Core Insight:** Google's official Gemma quantization is **QAT** (quantization-aware training) rather than PTQ — a 5,000-step fine-tune of the BF16 checkpoint using the non-quantized model's logits as targets — and ships specifically as `q4_0` GGUF for llama.cpp / Ollama / MLX consumption, recovering ~ 54 % of the perplexity degradation that pure post-training INT4 would incur.
- **Guideline:** When deploying Gemma 2 / Gemma 3, prefer Google's QAT `q4_0` GGUF builds over community PTQ ports; expect ~ 4× memory reduction vs BF16 and meaningfully better quality than llama.cpp's stock RTN-INT4.
- **Authors:** Google DeepMind (Gemma team)
- **Year:** 2024-2025 (Gemma 2 in 2024-06, Gemma 3 in 2025-03)
- **URL:** Gemma 3 QAT blog: https://developers.googleblog.com/en/gemma-3-quantized-aware-trained-state-of-the-art-ai-to-consumer-gpus/
- **Relevant topics:** QAT, q4_0 GGUF, knowledge distillation calibration, consumer-GPU deployment

## Abstract
Gemma 2 (2B / 9B / 27B, June 2024) and Gemma 3 (1B / 4B / 12B / 27B, March 2025) are Google's open-weight transformer family. Quantization releases follow a Google-specific pattern: instead of relying on community AWQ / GPTQ ports, Google publishes its own **QAT** checkpoints — the BF16 model is fine-tuned for ~ 5,000 steps with the loss = KL(student_quantized_logits || teacher_BF16_logits), so the model learns to be quantization-friendly. The release format is `q4_0` GGUF (a particular llama.cpp 4-bit quant: per-block float scale, no zero-point, 32 elements per block) — chosen for compatibility with llama.cpp / Ollama / MLX rather than for top-end serving throughput. Google reports the QAT step recovers 54 % of the perplexity degradation a naïve q4_0 PTQ would incur on Gemma 3.

## Key Contributions
- **QAT recipe**: ~ 5,000 steps fine-tune; loss = KL(quant_logits ‖ teacher_BF16_logits); produces a quant-friendly checkpoint.
- **q4_0 GGUF release format**: per-block FP16 scale, no zero-point (symmetric), 32 elements per block; the simplest GGUF format, broadly compatible.
- **54 % perplexity-degradation recovery** vs naïve q4_0 PTQ on Gemma 3.
- **Consumer-GPU targeting**: Gemma 3 27B QAT-INT4 = 14.1 GB → fits RTX 3090 (24 GB); 12B QAT-INT4 = 6.6 GB → fits RTX 4060 laptop (8 GB).
- **Family coverage**: QAT release for every Gemma 3 size (1B / 4B / 12B / 27B) and Gemma 2 sizes.
- **Multimodal Gemma 3** (PaliGemma successors): same QAT recipe applied; vision encoder kept at higher precision.

## Key Figures/Tables to Study
- The Gemma 3 QAT blog memory table (BF16 vs INT4 per size).
- The perplexity-degradation-recovery chart (54 % figure cited).
- The hardware-target column (RTX 3090 / 4060 / mobile) for each Gemma 3 size.

## Technical Details

### QAT recipe
- **Base**: BF16 fine-tuned checkpoint of Gemma.
- **Step count**: ~ 5,000 (small fraction of a full fine-tune; cheap).
- **Loss**: KL divergence between the quantized student's logits and the non-quantized teacher's logits, per token.
- **Quant simulation**: fake-quant the weights to q4_0 in the forward pass (with STE for backward).
- **Effect**: weights learn to be more uniformly distributed within each 32-element block → less precision loss when actually quantized to q4_0.

### q4_0 format (llama.cpp legacy 4-bit)
- 32 elements per block.
- One FP16 scale per block.
- No zero-point (symmetric).
- Storage: 4 bits/weight + 16 bits/block = 4 + 0.5 = 4.5 bpw effective.
- Simplest GGUF format; widely supported in llama.cpp, MLX, Ollama, LMStudio.

### Why q4_0 (not k-quants)
- q4_0 is the lowest-common-denominator GGUF; runs everywhere.
- k-quants (q4_K_M etc.) are higher-quality but require more recent llama.cpp.
- Google's choice is portability over peak quality; the QAT step closes the gap.

### Memory per Gemma 3 size
| Model | BF16 | INT4 (q4_0) | Target hardware |
|-------|------|-------------|-----------------|
| Gemma3-1B | 2 GB | 0.5 GB | mobile, anything |
| Gemma3-4B | 8 GB | 2.6 GB | laptop GPU |
| Gemma3-12B | 24 GB | 6.6 GB | RTX 4060 (8 GB) |
| Gemma3-27B | 54 GB | 14.1 GB | RTX 3090 (24 GB) |

### KV cache caveat
- The memory figures cover weight only; KV cache adds proportional memory at long context.
- Gemma 3 supports 128 K context (1M for some sizes); KV cache at 128 K is large (~ 10 GB for 27B at FP16).
- Community guidance: pair Google's QAT q4_0 weights with FP8 / INT8 KV cache for full deployment.

### Serving stacks
- **llama.cpp**: direct consumption of the GGUF files.
- **Ollama**: pulls from Google's HF org; user-friendly CLI.
- **MLX (Apple Silicon)**: re-conversion of GGUF; Apple's MLX has its own quant path that competes.
- **vLLM**: less common for Gemma than for Llama / Qwen; community AWQ ports exist.

## Connections
- [[gguf-k-quants]] / [[gguf-q4-0]] — the GGUF format family Gemma releases in.
- [[awq]] / [[gptq]] — alternative community paths; Google chose QAT instead.
- [[llama-cpp-ggml]] (in `frameworks/`) — the consuming runtime.
- [[llama-3-quantization]] — counterpart for Meta's Llama; uses PTQ (AWQ/GPTQ/FP8), not QAT.
- [[qwen-2.5-quant]] — counterpart for Alibaba's Qwen; uses PTQ (AWQ/GPTQ/GGUF).
- [[lsq]] (classical) — the QAT precedent (learned step size); Gemma's QAT is conceptually similar but uses KD loss instead of task loss.
- [[bitdistiller]] — academic predecessor of KD-driven quant fine-tuning.
