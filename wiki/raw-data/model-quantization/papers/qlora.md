<!-- scope: QLoRA — 4-bit NF4 frozen base + LoRA adapters; democratized 65B-model fine-tuning
     deps: [[lora]], [[llm-int8]], [[nf4]]
     see-also: [[qa-lora]], [[loftq]], [[peqa]], [[bitsandbytes-nf4]]
-->

# QLoRA: Efficient Finetuning of Quantized LLMs
- **Core Insight:** A 4-bit base model can be fine-tuned with full-precision-equivalent quality by (a) using **NF4**, an information-theoretically optimal 4-bit quantile code for the empirically near-Gaussian weight distribution, (b) **double-quantizing** the per-block scales to save ~0.4 bits/weight, and (c) routing gradients through frozen 4-bit weights into BF16 LoRA adapters — so the base never leaves 4-bit memory.
- **Guideline:** For SFT/instruction-tuning under tight GPU budgets, use QLoRA with NF4 weights, double quantization, paged AdamW, and LoRA rank r=64 on all linear layers (including attention and FFN); fits LLaMA-65B fine-tuning into a single 48 GB GPU.
- **Authors:** Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke Zettlemoyer
- **Year:** 2023 (NeurIPS 2023)
- **URL:** https://arxiv.org/abs/2305.14314
- **Relevant topics:** NormalFloat 4-bit, double quantization, paged optimizer, LoRA, 65B fine-tuning, Guanaco

## Abstract
QLoRA enables fine-tuning a 65B-parameter LLM on a single 48 GB GPU while preserving full 16-bit fine-tuning quality. It contributes three pieces stacked on top of LoRA: (1) **4-bit NormalFloat (NF4)**, a quantile-based 4-bit format optimal for zero-mean unit-variance normal weights; (2) **Double Quantization**, which quantizes the FP32 block scales themselves to 8-bit, saving ~0.37 bits per weight on average; (3) **Paged Optimizers**, which page AdamW states between GPU and CPU memory during gradient spikes. The frozen base lives in 4-bit; gradients backprop through dequantized weights (computed on-the-fly in BF16) into BF16 LoRA adapters of rank r=64. The Guanaco family fine-tuned this way reaches 99.3% of ChatGPT's Vicuna benchmark in 24 GPU-hours.

## Key Contributions
- **NF4 format** — first quantile-based 4-bit code derived from the standard normal CDF; provably min-MSE for Gaussian weights.
- **Double Quantization** — quantize the 32-bit block-scales of the NF4 quantization to FP8 (256 outer group) for ~0.4 bits/weight savings.
- **Paged Optimizers** — NVIDIA unified memory keeps optimizer states resident on CPU during pre-emptable peaks.
- **Frozen-4-bit base + BF16 LoRA** — the standard fine-tune recipe of 2023–2025; backprop through dequantized weights but never store FP weights.
- Demonstrates that QLoRA does not regress against 16-bit full fine-tuning across MMLU, BBH, and chatbot benchmarks (Vicuna evaluator + human).
- Releases Guanaco (7B / 13B / 33B / 65B), the first open instruction-tuned 65B chat model.

## Key Figures/Tables to Study
- **Figure 2:** the QLoRA dataflow — NF4 weight → dequant to BF16 → matmul with BF16 activation → LoRA path adds.
- **Figure 3 (NF4 derivation):** 16 quantiles of the standard normal — the actual code values.
- **Table 6:** Guanaco vs ChatGPT — 99.3% Vicuna with 24 hr fine-tune on a single A100.
- **Table 5:** QLoRA-FT vs full FT — within 0.1 of each other on MMLU/BBH.

## Technical Details

### NormalFloat 4-bit (NF4)
LLM weights, after normalisation, are well modelled by a zero-mean unit-variance normal. The information-theoretically optimal 4-bit code partitions the real line by **equal quantile mass** under the normal CDF, and represents each bin by its midpoint quantile:
```
NF4 levels q_i = Φ⁻¹( (i / 16) − offset ),  i = 1..16,  with symmetric placement around 0
```
After normalising a block of 64 weights by their absmax `s`, each weight is mapped to the nearest NF4 level. NF4 has **better empirical MSE than INT4 or FP4** on real LLM weights (Dettmers measures ~0.3% reduction in perplexity vs equivalent 4-bit alternatives).

The 16 NF4 values (symmetric, ±0 absent because zero is one of the asymmetric quantile midpoints):
```
[-1.0, -0.6961928, -0.5250730, -0.39491748, -0.28444138, -0.18477343,
 -0.09105475, 0.0, 0.07958029, 0.16093020, 0.24611230, 0.33791524,
 0.44070983, 0.56261432, 0.72295684, 1.0]
```

### Block-wise quantization
- Each tensor partitioned into blocks of B = 64 weights.
- Per-block scale `s_b = max|w_b|` stored in FP32 (or FP16).
- Quant: `q_i = NF4_round(w_i / s_b)`; dequant: `w_i ≈ s_b · NF4[q_i]`.

### Double Quantization
Per-block FP32 scales `{s_b}` are themselves quantized:
- Group every 256 block-scales together.
- Quantize them to FP8 with one FP32 outer scale per 256-block group.
- Saving: `(32 − 8) / 64 + 32 / (64·256) ≈ 0.373 bits/weight`.

### Paged Optimizer
AdamW state (m, v) for a 65B model is ~520 GB — vastly exceeds GPU memory even before activations. Paged Optimizer leverages NVIDIA unified memory so the OS pages optimizer chunks between GPU HBM and CPU DRAM on demand; the gradient-spike phase that causes OOM in standard mixed-precision FT is absorbed.

### LoRA on top of frozen NF4 base
For each Linear layer `y = W x`:
- W is frozen 4-bit NF4 (with double-quantized scales).
- Add `ΔW = α/r · B A`, with A ∈ R^{r × d_in} and B ∈ R^{d_out × r} in BF16, both trainable.
- Forward: `y = dequant(W) · x + (α/r) · B (A x)`.
- Backward: gradients flow through the BF16 dequant op into A, B; W is never updated.
- Default: r = 64 on **all** linear layers (q, k, v, o, gate, up, down) — not just attention.

### Hyperparameters (Guanaco recipe)
| Knob | Value |
|------|-------|
| Weight format | NF4 |
| Block size | 64 |
| Double quant | yes (FP8 inner scale, FP32 outer) |
| Effective bits/weight | ~4.127 (NF4 + DQ overhead) |
| LoRA rank r | 64 |
| LoRA target | all linear layers |
| LoRA α | 16 |
| Optimizer | paged AdamW 32-bit |
| LR | 2e-4 |
| LR schedule | constant |
| Batch (sequence) | 16 |
| Max seq | 512 (Guanaco) |
| Wall-clock 65B | ~24 hr on 1×A100 48GB |

## Connections
- Frozen-base PEFT predecessor: LoRA (Hu 2021) — QLoRA quantizes the frozen base.
- Same author's 8-bit predecessor: [[llm-int8]] (mixed-precision INT8 inference, no fine-tune).
- Format reference: [[nf4]].
- Joint-init successor (better LoRA init for quantized base): [[loftq]].
- Quant-aware PEFT successor (merges adapters into the INT4 weight): [[qa-lora]].
- Scale-only PEFT alternative: [[peqa]].
- Implementation: [[bitsandbytes-nf4]].
- Quantized-outlier sibling from same author: [[spqr]].
