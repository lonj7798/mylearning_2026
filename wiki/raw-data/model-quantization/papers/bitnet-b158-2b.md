<!-- scope: scaled BitNet b1.58 2B/4B models with full evals (Ma et al. / Microsoft 2025)
     deps: [[bitnet-b158]], [[bitnet]]
     see-also: [[bitnet-a48]], [[bitnet-scaling-laws]], [[bitnet-models]]
-->

# BitNet b1.58 2B-4T (Microsoft Native 1.58-bit LLM)
- **Core Insight:** A 2B-parameter transformer trained from scratch with ternary {-1, 0, +1} weights (1.58 bpw) and 8-bit activations, on 4 trillion tokens, can match the average benchmark performance of fully-trained dense BF16 2B models — proving native 1.58-bit pretraining is competitive at the 2B / 4T-token scale, not just a small-scale curiosity.
- **Guideline:** Train BitNet from scratch (post-training quantization of an FP16 model to 1.58 bits does *not* work); use BitLinear layers with absmean quantization on weights and absmax per-token quantization on activations; keep embeddings and LM-head in higher precision; deploy via bitnet.cpp for actual speedup (HuggingFace transformers won't give you the inference win).
- **Authors:** Shuming Ma, Hongyu Wang, Lingxiao Ma, Lei Wang, Wenhui Wang, Shaohan Huang, Li Dong, Furu Wei et al. (Microsoft Research)
- **Year:** 2025 (released April 16, 2025)
- **URL:** https://huggingface.co/microsoft/bitnet-b1.58-2B-4T • https://github.com/microsoft/BitNet
- **Relevant topics:** native 1.58-bit LLM, BitLinear, absmean quantization, ternary weights, MIT-licensed open weights

## Abstract
The microsoft/bitnet-b1.58-2B-4T release is the first open-source, native 1-bit-class (1.58 bpw) LLM at 2B scale, trained from scratch on 4 T tokens. The architecture is a standard transformer (Llama-3 tokenizer, RoPE, ReLU² FFN, RMSNorm, no bias) with all linear layers replaced by `BitLinear` (ternary weights {-1, 0, +1}, INT8 activations). Training stages are pretrain → SFT → DPO. On average across 16 standard evals (MMLU, ARC, GSM8K, HumanEval, …) the model scores 54.19 — within striking distance of Llama-3.2 1B (~42) and competitive with Gemma-3 1B / Qwen-2.5 1.5B (~55). Memory footprint is 0.4 GB (vs 1.4-4.8 GB for FP16 2B class), CPU latency 29 ms (vs 41-124 ms), and estimated energy is ~7× lower. The model proves that the 1.58-bit scaling story (BitNet b1.58 2024 paper) generalizes from research-scale runs to a full 4T-token pretrain.

## Key Contributions
- **First open-source native 1.58-bit LLM at 2B / 4T-token scale**, MIT licensed, with full eval transparency.
- Demonstrates BitNet b1.58 quality claims hold at scale: ARC-Challenge 49.9, MMLU 53.2, GSM8K 58.4, MATH-500 43.4, HumanEval+ 38.4.
- Releases three variants: packed 1.58-bit weights for deployment, BF16 master weights for fine-tuning, GGUF for bitnet.cpp.
- Validates the BitLinear training recipe (absmean weights + absmax INT8 activations + per-token activation scale + Squared ReLU FFN) at production scale.
- Releases bitnet.cpp — a fork of llama.cpp with custom W1.58 kernels that actually realize the theoretical speedup on x86 (2.4-6.2× over llama.cpp FP16) and ARM CPUs (1.4-5.1×).
- 4-trillion-token training run — comparable in token budget to mid-2024 dense models, refuting the worry that 1.58-bit training would require absurd token budgets to converge.

## Key Figures/Tables to Study
- The benchmark table comparing BitNet-b1.58-2B-4T to Llama-3.2 1B, Qwen-2.5 1.5B, Gemma-3 1B, Phi-2 across MMLU / ARC / GSM8K / HumanEval / MATH-500 — the headline parity claim.
- The CPU latency / memory / energy table — the operational case for 1-bit deployment.
- The bitnet.cpp speedup table on x86 and ARM CPUs.

## Technical Details

### Architecture
- Transformer with **BitLinear** replacing every dense layer in the block (Q, K, V, O projections; FFN gate, up, down).
- Llama-3 tokenizer (vocab 128 256), context 4 096.
- RoPE positional encoding, squared ReLU (ReLU²) in FFN (vs SwiGLU), subln RMS normalization, no bias terms.
- ~ 2B parameters, 4T training tokens.

### BitLinear quantization
- **Weights:** ternary {-1, 0, +1} via **absmean** quantization — quantize to {-1, 0, +1} based on sign(w) where |w| > mean(|W|) threshold; pack as 1.58 bpw (log₂ 3 ≈ 1.585).
- **Activations:** 8-bit integers via **absmax per-token** quantization (one scale per token, applied to all hidden channels).
- Forward pass: y = (W_ternary · x_INT8) * (scale_w * scale_x). The W·x matmul itself is integer add/subtract only (no multiplies, since weight entries are 0 or ±1).
- During training, the quantization uses straight-through estimator (STE) so gradients flow back through the quantizer.

### Training stages
1. **Pretrain** on 4 T tokens (large web corpus + math + code mix).
2. **SFT** on instruction data.
3. **DPO** for alignment.

### Selective high precision (not BitLinear)
- Token embedding table: BF16.
- LM head: BF16.
- RMSNorm scales: BF16.
- Everything else (40 transformer blocks × {QKVO + FFN}): BitLinear.

### Variants released
- `microsoft/bitnet-b1.58-2B-4T` — packed 1.58-bit weights for deployment.
- `microsoft/bitnet-b1.58-2B-4T-bf16` — BF16 master weights, intended for fine-tuning / research.
- `microsoft/bitnet-b1.58-2B-4T-gguf` — GGUF format for bitnet.cpp.

### Key eval numbers
| Benchmark | BitNet-b1.58-2B-4T |
|-----------|--------------------|
| Average (16 evals) | 54.19 |
| MMLU | 53.17 |
| ARC-Challenge | 49.91 |
| GSM8K | 58.38 |
| MATH-500 | 43.40 |
| HumanEval+ | 38.40 |
| Memory (non-embed) | 0.4 GB |
| Latency (CPU, one token) | 29 ms |
| Energy estimate | 0.028 J |

### Deployment caveat
- Standard HuggingFace transformers loads the model in BF16 — no speedup gain.
- bitnet.cpp uses custom W1.58 × INT8 kernels on x86 / ARM CPUs that give the actual operational win.

## Connections
- [[bitnet-b158]] — the original BitNet b1.58 paper (Ma et al. 2024) whose claims this release validates at scale.
- [[bitnet]] — the 2023 BitNet paper that introduced BitLinear (binary {-1, +1} weights).
- [[bitnet-a48]] — the 4-bit activation extension; this 2B-4T release uses INT8 activations, not 4-bit.
- [[bitnet-scaling-laws]] — companion scaling-law analysis explaining the bit budget tradeoff.
- [[bitnet-models]] — the model-report page indexing all official BitNet releases.
- [[era-of-1bit-llms]] — the consolidation paper / Microsoft Research blog framing for the 1-bit line.
