<!-- scope: Qwen 3 official quantization releases (AWQ / GPTQ / GGUF / FP8)
     deps: [[qwen-2.5-quant]], [[awq]], [[gptq]]
     see-also: [[fp8-formats-paper]], [[deepseek-v3-fp8]]
-->

# Qwen 3 Quantization (Alibaba, 2025)
- **Core Insight:** Qwen 3 (April 2025) extends the Qwen 2.5 quantization matrix with FP8 row-wise weight + activation builds for Hopper / Blackwell serving — the first major non-DeepSeek open-model family to ship lab-blessed FP8 weights — while keeping the AWQ / GPTQ / GGUF coverage Qwen 2.5 established.
- **Guideline:** When deploying Qwen 3, use the FP8 build on H100 / H200 / B200; fall back to AWQ-INT4 (group_size=128) on Ampere; use the GGUF builds for llama.cpp / Ollama. Qwen 3's reasoning-mode models follow the [[deepseek-r1-quantization]] sensitivity caveat — be conservative on activation quant.
- **Authors:** Qwen team (Alibaba)
- **Year:** 2025 (Qwen 3 released 2025-04-29)
- **URL:** https://huggingface.co/Qwen • https://qwenlm.github.io/blog/qwen3/ • https://qwen.readthedocs.io/
- **Relevant topics:** FP8 inference, AWQ, GPTQ, GGUF, reasoning-mode quant sensitivity

## Abstract
Qwen 3 ships six dense sizes (0.6B / 1.7B / 4B / 8B / 14B / 32B) plus two MoE models (30B-A3B activation, 235B-A22B activation), with a wide quant matrix per size: AWQ-INT4, GPTQ-INT4 + INT8, GGUF k-quants (q2_K → q8_0), and — new vs Qwen 2.5 — **FP8 row-wise** builds for Hopper / Blackwell serving. The FP8 path uses E4M3 weights + activations with per-row scales via FBGEMM-FP8 / TRT-LLM, similar to Meta's Llama-3.1-405B-FP8 recipe. The MoE models add per-expert quant decisions; Qwen's reported recipe keeps the routing gate at BF16 and applies the same FP8 / W4 treatment to expert linears. Qwen 3 also introduced an explicit "thinking mode" toggle that switches the model into long-CoT reasoning; the quant docs flag that thinking-mode generation is more quant-sensitive than chat mode, echoing the DeepSeek-R1 observation.

## Key Contributions
- **First lab-blessed FP8 builds outside DeepSeek**: `Qwen/Qwen3-…-FP8` checkpoints, E4M3 per-row, FBGEMM-FP8 backend.
- **Continued AWQ-INT4 / GPTQ-INT4 / GGUF coverage** following the Qwen 2.5 matrix.
- **MoE quant**: 30B-A3B and 235B-A22B MoE models with per-expert FP8 / AWQ treatment; routing gate retained at higher precision.
- **Thinking-mode caveat**: documentation warns that the reasoning-mode generation degrades under W4A4 / W4A8 faster than chat-mode; recommends FP8 or W4A16 for thinking mode.
- **Long-context preservation**: 32 K / 128 K context preserved across all quant builds; RoPE settings carried through.
- **Multilingual calibration**: continued from Qwen 2.5 — calibration dataset spans the same ~30 languages.

## Key Figures/Tables to Study
- The Qwen 3 quantization benchmark table per size showing MMLU, MATH, HumanEval drops per quant tier.
- The thinking-mode vs non-thinking-mode quality delta under W4 quant (the new evidence row vs Qwen 2.5).
- MoE-specific quant table for 30B-A3B and 235B-A22B.

## Technical Details

### FP8 recipe
- **Format**: E4M3 weights, E4M3 activations, per-row weight scales.
- **Backend**: FBGEMM-FP8 (PyTorch) / TRT-LLM FP8 path / vLLM FP8.
- **Selective precision**: LM head, final norm in BF16; routing gate in BF16 (for MoE).
- **Calibration**: per-row scales fit on a multilingual instruct subset.
- **Quality**: < 0.5 pt MMLU drop, similar to Llama-3.1-405B-FP8.

### AWQ / GPTQ recipe
- Same as Qwen 2.5: group_size=128, per-channel symmetric (AWQ) or per-group asymmetric (GPTQ).
- Calibrated on Qwen's multilingual instruct dataset.

### MoE quant (30B-A3B, 235B-A22B)
- Routing gate: BF16 (gates need precise tie-breaking; quant breaks routing stability).
- Expert linears: FP8 / AWQ-INT4 / GGUF — same treatment as dense.
- Expert quantization is **uniform** across experts in the released checkpoints; per-expert bit allocation (QMoE-style) is left to community.

### Thinking-mode sensitivity
- Qwen 3 has a `/think` token that switches to long-CoT generation (similar in spirit to R1).
- Empirically (Qwen + community): thinking mode under W4A8 drops AIME pass@1 by 5-10 pt where chat mode drops < 1 pt.
- Recommendation: FP8 base + ≥ FP8 KV cache for thinking-mode-heavy deployments.

### Serving stacks
- vLLM: FP8 path + AWQ path; both vendor-tested by Alibaba.
- SGLang: same.
- TensorRT-LLM: FP8 + AWQ + NVFP4 (Blackwell) all available.
- llama.cpp: GGUF builds for the dense sizes; MoE builds are partial (community work).

### Storage per size
| Model | FP16 | FP8 | AWQ-INT4 |
|-------|------|-----|----------|
| Qwen3-8B | 16 GB | 8 GB | 5 GB |
| Qwen3-14B | 28 GB | 14 GB | 9 GB |
| Qwen3-32B | 64 GB | 32 GB | 19 GB |
| Qwen3-30B-A3B (MoE) | 60 GB | 30 GB | 18 GB (W4 expert) |
| Qwen3-235B-A22B (MoE) | 470 GB | 235 GB | 140 GB |

## Connections
- [[qwen-2.5-quant]] — direct predecessor with the same matrix minus FP8.
- [[awq]] / [[gptq]] — algorithms behind the W4 checkpoints.
- [[fp8-formats-paper]] — E4M3 spec used by the FP8 builds.
- [[deepseek-r1-quantization]] — reasoning-model quant sensitivity precedent that informs Qwen 3 thinking-mode guidance.
- [[deepseek-v3-fp8]] — DSV3 FP8 training precedent; Qwen 3 is BF16-trained then FP8-quantized (not FP8-native trained).
- [[mixtral-quant]] — MoE-quant precedent for the per-expert + routing-gate treatment.
- [[gguf-k-quants]] / [[llama-cpp-gguf-releases]] — k-quant releases for Qwen 3.
