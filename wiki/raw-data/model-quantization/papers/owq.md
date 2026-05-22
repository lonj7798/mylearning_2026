<!-- scope: OWQ — outlier-aware weight-only quant; keep activation-outlier-aligned weight columns in FP16
     deps: [[gptq]], [[llm-int8]]
     see-also: [[spqr]], [[squeezellm]], [[awq]]
-->

# OWQ: Outlier-aware Weight Quantization for Efficient Fine-Tuning and Inference of Large Language Models
- **Core Insight:** A weight *column* is "weak" if it multiplies an activation channel with extreme magnitude — keeping just those weak columns in FP16 (a structured 1–5% fraction) and aggressively quantizing the rest matches the accuracy of a uniformly higher-bit baseline at half the average bit-width.
- **Guideline:** Pick weak columns by `score_j = max_t |X_{t,j}| · ||W_{·,j}||₂`, keep the top 1–5% in FP16, GPTQ-quantize the rest at 3 bits (group_size=128). Use OWQ's "weak column tuning" (WCT) to fine-tune *only those FP16 columns* for parameter-efficient adaptation.
- **Authors:** Changhun Lee, Jungyu Jin, Taesu Kim, Hyungjun Kim, Eunhyeok Park
- **Year:** 2023 (AAAI 2024 oral)
- **URL:** https://arxiv.org/abs/2306.02272
- **Relevant topics:** outlier-aware weight quant, structured mixed precision, weak column tuning, parameter-efficient fine-tuning

## Abstract
OWQ identifies a structured form of outlier in LLM weight matrices: entire *columns* (input dims) that multiply against high-magnitude activation channels are disproportionately sensitive to quantization error. Rather than per-weight outlier extraction ([[spqr]], [[squeezellm]]), OWQ stores these whole columns in FP16 and aggressively quantizes the rest with GPTQ. The structured nature makes the dequant kernel a simple concat: dense INT-k columns + sparse FP16 columns. The same FP16 columns then serve as the *only* trainable parameters for OWQ's "Weak Column Tuning" (WCT), giving parameter-efficient adaptation in the OWQ format with minimal memory overhead. Achieves 3.1-bit-average performance matching 4-bit GPTQ on OPT/LLaMA.

## Key Contributions
- Structured column-level outlier identification — easier to kernelise than per-weight sparse formats.
- 3.1-bit average competitive with 4-bit GPTQ.
- **Weak Column Tuning (WCT)**: parameter-efficient fine-tuning that updates only the small FP16 outlier columns — a LoRA-free PEFT alternative that lives natively in the quantized format.
- Compatible with existing GPTQ pipelines for the dense path.

## Key Figures/Tables to Study
- **Figure 3:** distribution of per-column sensitivity scores — sharp tail (1–5%) carries the loss.
- **Table 4:** OPT/LLaMA OWQ-3.01 vs GPTQ-4 — comparable or better at lower average bits.
- **Table 7:** WCT vs LoRA fine-tuning — comparable downstream accuracy with smaller trainable param count.

## Technical Details

### Weak column score
For weight `W ∈ R^{d_out × d_in}` and calibration activations `X ∈ R^{N × d_in}`, per input channel j:
```
score_j = max_t |X_{t, j}|  ·  ||W_{·, j}||_2
```
This combines activation outlier magnitude with weight contribution. Pick the top `k = ⌈p · d_in⌉` columns (p = 1–5%) as the **weak set** W*.

### Mixed-precision storage
- Weak columns (1–5% of d_in): stored as FP16, no quantization.
- Dense columns (95–99%): GPTQ-quantized at b = 3 bits, group_size = 128, with the weak columns masked out so the Hessian update only propagates within the dense set.

Effective bits/weight ≈ `0.95 · 3.125 + 0.05 · 16 ≈ 3.77`, drops further at p=1%.

### Inference
```
y_dense = GPTQ-decoded GEMV(W_dense_q, x)
y_outlier = FP16-GEMV(W_weak_fp16, x[weak_idx])
y = y_dense + y_outlier
```
The weak indices are fixed at quant time → simple gather + small dense matmul.

### Weak Column Tuning (WCT)
For fine-tuning:
- Freeze all dense (quantized) weights.
- Make only `W_weak_fp16` trainable (shape `d_out × k`, k = 1–5% · d_in).
- Standard task-specific loss; AdamW.
- Trainable parameter count: ~3–5% of full-fine-tune, comparable to LoRA at r=8 but already lives in the quantized format → no merge step.

### Hyperparameters
| Knob | Value |
|------|-------|
| Weak fraction p | 0.01–0.05 |
| Dense bits | 3 |
| Group size | 128 |
| Score | `max|X_j| · ||W_{·,j}||_2` |
| Calibration | 128 sequences C4 |
| WCT optimizer | AdamW, lr 1e-4, batch 16 |

## Connections
- Per-weight outlier rivals (finer-grained but harder to kernelise): [[spqr]], [[squeezellm]].
- Activation outlier in FP16 column form (input X side, not weight side): [[llm-int8]].
- Activation-aware weight quant without storing outliers in FP16: [[awq]].
- LoRA alternative for quantized PEFT: [[qlora]], [[loftq]], [[peqa]].
- Backend uses [[gptq]] for the dense path.
