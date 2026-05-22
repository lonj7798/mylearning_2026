<!-- scope: Qwen 2.5 official quantization releases (GPTQ / AWQ / GGUF)
     deps: [[gptq]], [[awq]]
     see-also: [[qwen-3-quant]], [[llama-3-quantization]]
-->

# Qwen 2.5 Quantization (Alibaba)
- **Core Insight:** Qwen 2.5 ships the most comprehensive lab-blessed quantization matrix of any 2024 open model family — every Instruct size (0.5B / 1.5B / 3B / 7B / 14B / 32B / 72B) has official AWQ-INT4, GPTQ-INT4 (per-channel and group-128), GPTQ-INT8, and GGUF (q2_K through q8_0) variants on the Qwen HF org, all calibrated by Alibaba with consistent recipes.
- **Guideline:** When deploying Qwen 2.5, default to Qwen's official `Qwen/Qwen2.5-…-Instruct-AWQ` for vLLM serving; use the GGUF builds (q4_k_m / q5_k_m) for llama.cpp / Ollama / LMStudio; only re-quantize from scratch if you need a non-standard target.
- **Authors:** Qwen team (Alibaba)
- **Year:** 2024 (Qwen 2.5 released 2024-09)
- **URL:** https://huggingface.co/Qwen • https://qwen.readthedocs.io/en/latest/quantization/awq.html
- **Relevant topics:** official AWQ / GPTQ recipes, GGUF, group_size 128, calibration

## Abstract
Qwen 2.5 is the second-generation Qwen release (after Qwen 2 in 2024-06) and shipped with the most extensive set of official quantization variants in the open-LLM space. Every Instruct model size has companion AWQ-INT4, GPTQ-INT4 (group_size=128), GPTQ-INT8, and GGUF (k-quant ladder q2_K → q8_0) on the QwenLM HF org. The recipes are all standard (AWQ group_size=128 per channel, GPTQ with the same group_size, GGUF k-quants with the standard 16×16 super-block layout) but calibrated by Alibaba on a multilingual chat-style dataset, so they avoid the English-only calibration bias of most community ports. The deployment story for Qwen 2.5 is: pick the closest HF release to your target, no re-calibration needed for typical serving.

## Key Contributions
- **AWQ-INT4 official builds** for all Instruct sizes; group_size=128, per-channel symmetric.
- **GPTQ-INT4 / GPTQ-INT8 official builds** with consistent group sizes.
- **GGUF k-quant ladder** (q2_K, q3_K, q4_K_M, q5_K_M, q6_K, q8_0) for every size — pre-built for llama.cpp / Ollama / LMStudio.
- **Multilingual calibration**: calibration set drawn from Qwen's own multilingual instruct corpus, so non-English quality holds up better than community AWQ ports calibrated on English-only data.
- **Qwen-specific architecture handled**: Qwen 2.5 has bias terms in QKV projections (unlike Llama 3), and the quant configs include the bias treatment.
- **Long-context (131 K) preserved**: the quant builds keep RoPE settings intact so the 131 K context works in the quantized form.

## Key Figures/Tables to Study
- The Qwen 2.5 quantization benchmark table (in the Qwen docs) showing MMLU / Math / Coding deltas for each quant tier per model size.
- The model-card tensor-type table on the HF page (I32 + F16 for AWQ; F16 + I4 for GPTQ; varied for GGUF).
- The throughput comparison table (vLLM with AWQ vs FP16) on H100 / A100.

## Technical Details

### Recipes
- **AWQ-INT4**: group_size=128 along the channel axis; per-group FP16 scale; symmetric (no zero-point); zero_point=None or False in the config. Activation-magnitude calibration on a held-out subset of Qwen's instruct data.
- **GPTQ-INT4**: same group_size; Hessian-aware sequential weight rounding; per-group FP16 scale + zero-point (asymmetric).
- **GPTQ-INT8**: same algorithm at higher precision; rarely used (FP16 is competitive at 8-bit weight).
- **GGUF k-quants**: see [[gguf-k-quants]] in `formats/`. The Qwen GGUF builds use the standard `q4_k_m` (mixed) and `q5_k_m` (mixed) recipes that put critical tensors (attention.K, attention.V) at higher bit width.

### Per-size storage
| Model | FP16 | AWQ-INT4 | GGUF q4_k_m |
|-------|------|----------|-------------|
| Qwen2.5-7B | 15 GB | 5 GB | 4.7 GB |
| Qwen2.5-14B | 28 GB | 9 GB | 8.4 GB |
| Qwen2.5-32B | 65 GB | 19 GB | 18.4 GB |
| Qwen2.5-72B | 145 GB | 41 GB | 40.5 GB |

### Quality at AWQ-INT4 (Qwen's reported numbers)
- MMLU drop typically < 1 pt vs FP16.
- Math / Coding drop similar or slightly larger (2-3 pt on GSM8K for 7B; smaller for 72B).
- Multilingual evals (MMLU-multi) tracked separately; multilingual calibration helps here.

### Serving
- vLLM: `--quantization awq_marlin` or `awq`; auto-picks Marlin/Machete on Ampere/Hopper.
- SGLang: same backends as vLLM for AWQ/GPTQ.
- TensorRT-LLM: official AWQ + per-channel INT8 path.
- llama.cpp: GGUF builds directly consumable.
- MLX (Apple Silicon): community GGUF re-conversion.

### Qwen 2.5 family
| Size | Variants |
|------|----------|
| 0.5B / 1.5B / 3B / 7B / 14B / 32B / 72B | Base + Instruct + AWQ + GPTQ-INT4 + GPTQ-INT8 + GGUF (×6 k-quants) |

## Connections
- [[awq]] — algorithm behind the AWQ checkpoints.
- [[gptq]] — algorithm behind the GPTQ checkpoints.
- [[gguf-k-quants]] (in `formats/`) — k-quant ladder layout that the GGUF builds use.
- [[qwen-3-quant]] — sibling page for Qwen 3.
- [[llama-3-quantization]] — counterpart for Meta's Llama 3; similar AWQ/GPTQ pattern.
- [[marlin-kernel]] / [[machete-kernel]] — kernels used to serve Qwen 2.5 quant builds on Ampere/Hopper.
