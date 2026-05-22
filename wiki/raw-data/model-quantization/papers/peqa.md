<!-- scope: PEQA — parameter-efficient fine-tuning that updates only the per-channel quantization scales
     deps: [[qlora]], [[lora]]
     see-also: [[qa-lora]], [[loftq]]
-->

# PEQA: Memory-Efficient Fine-Tuning of Compressed LLMs via sub-4-bit Integer Quantization
- **Core Insight:** A surprising amount of task adaptation can be captured purely by changing the **per-output-channel quantization scales** while keeping the INT2/INT3/INT4 weight indices frozen — the scales are an O(d_out) per-layer affine knob that already covers task-specific dynamic-range shifts, and updating only them gives a LoRA-rivalling PEFT with no extra parameters at inference.
- **Guideline:** When the deployment must remain a pure low-bit artifact (no FP16 LoRA branch, no merge step), use PEQA: freeze the INT2/3/4 weight indices, make only the per-channel FP scales trainable, fine-tune with AdamW; trainable params ≈ d_out per layer (orders of magnitude below LoRA).
- **Authors:** Jeonghoon Kim, Jung Hyun Lee, Sungdong Kim, Joonsuk Park, Kang Min Yoo, Se Jung Kwon, Dongsoo Lee
- **Year:** 2023 (NeurIPS 2023)
- **URL:** https://arxiv.org/abs/2305.14152
- **Relevant topics:** scale-only PEFT, sub-4-bit fine-tuning, no-LoRA quantized adaptation

## Abstract
PEQA fine-tunes a pre-quantized LLM by updating only the per-output-channel scales of the quantization, leaving the integer weight indices frozen. The trainable parameter count is `Σ_layer d_out` — for LLaMA-65B that is ~5M params, vs ~350M for LoRA r=64. Despite the extreme reduction, PEQA recovers and often exceeds full-precision performance on language modeling, few-shot learning, and natural-language comprehension benchmarks across LLM sizes up to 65B. Because no FP adapter exists, the deployed model is a clean sub-4-bit integer artifact with the same kernel path as PTQ.

## Key Contributions
- The most parameter-efficient PEFT for quantized LLMs of 2023: only per-channel scales train.
- Demonstrates that **scale-only updates** are sufficient for task adaptation on top of a quantized base — fine-tuning is essentially a per-channel calibration repair plus task-specific dynamic-range shift.
- Inference artifact is pure sub-4-bit integer; no LoRA branch, no merge overhead.
- Scales to 65B with single-GPU fine-tuning.

## Key Figures/Tables to Study
- **Figure 2:** the parameter count comparison — PEQA vs LoRA vs full FT — orders of magnitude smaller.
- **Table 4:** LLaMA-65B sub-4-bit downstream — PEQA matches or beats LoRA-on-INT4-base.

## Technical Details

### Quantized weight representation
For each linear layer, after PTQ:
```
W ≈ s ⊙ W_int          (s ∈ R^{d_out × 1}, per-channel; W_int ∈ INT-b)
```
b ∈ {2, 3, 4}. Group-wise variant: s has shape `(d_out, d_in / G)`.

### PEQA fine-tuning
- **Frozen**: `W_int` (the INT2/3/4 weight indices).
- **Trainable**: `s` (the per-channel scales; total `d_out` or `d_out · d_in/G` params per layer).
- Standard task loss, AdamW.

Forward pass during training:
```
y = (s ⊙ W_int) · x
∂L/∂s computed normally; ∂L/∂W_int is not used (frozen).
```

### Optional bias channel
Some PEQA variants also unfreeze the per-channel zero-point (asymmetric quant), giving `2 · d_out` trainable params per layer.

### Parameter count
| Model | Full FT | LoRA r=64 | PEQA (per-channel) |
|-------|---------|-----------|---------------------|
| LLaMA-7B | 7B | 33M | ~700K |
| LLaMA-30B | 30B | 156M | ~2.5M |
| LLaMA-65B | 65B | 350M | ~5.4M |

### Inference
Identical to the pre-fine-tune PTQ artifact — same kernels (Marlin/AutoGPTQ/AutoAWQ), same memory layout. Only the FP per-channel scales differ.

### Hyperparameters (recipe)
| Knob | Value |
|------|-------|
| Base PTQ | RTN / GPTQ / NF4 (any) |
| Bits | 2, 3, 4 |
| Trainable | per-channel scale(s) |
| Optimizer | AdamW |
| LR | 1e-3 to 1e-2 (high — small parameter count) |
| Datasets tested | C4, MMLU, GSM8K, Alpaca |

## Connections
- LoRA-on-quantized-base alternative: [[qlora]] (BF16 adapter, larger).
- Merge-friendly alternative: [[qa-lora]] (group-wise adapter that merges into scales).
- Joint init alternative: [[loftq]].
- Full QAT alternative: [[llm-qat]].
- The kernel substrate inherits from: [[gptq]], [[autogptq]], [[autoawq]].
