<!-- scope: EfficientQAT — block-wise then end-to-end QAT for LLMs, 2-bit LLaMA-2-70B in 41h on one A100
     deps: [[lsq]], [[straight-through-estimator]]
     see-also: [[bitdistiller]], [[llm-qat]], [[pv-tuning]]
-->

# EfficientQAT: Efficient Quantization-Aware Training for Large Language Models
- **Core Insight:** Full-model QAT is infeasible at 70B scale (gradients + optimizer state for both quantized and FP weights blow up memory), but the cost can be decomposed: train every parameter *block-wise* first (one transformer block at a time, fitting in single-GPU memory), then end-to-end fine-tune *only the quantization step sizes* to capture inter-block interactions.
- **Guideline:** When you need true QAT (not PTQ) at sub-4-bit for ≥70B models on a single A100, use EfficientQAT = (1) Block-AP: block-wise full-parameter training, (2) E2E-QP: end-to-end training of only step sizes; gets 2-bit LLaMA-2-70B in 41h on one A100-80GB.
- **Authors:** Mengzhao Chen, Wenqi Shao, Peng Xu, Jiahao Wang, Peng Gao, Kaipeng Zhang, Yu Qiao, Ping Luo
- **Year:** 2024 (ACL 2025)
- **URL:** https://arxiv.org/abs/2407.11062
- **Relevant topics:** block-wise QAT, end-to-end step-size training, 2-bit QAT, single-GPU 70B

## Abstract
EfficientQAT decomposes QAT into two consecutive phases: (1) Block-wise training of All Parameters (Block-AP) — the first method to enable direct training of all parameters in a block-wise manner, reducing accuracy loss in low-bit scenarios by enlarging the optimization solution space; and (2) End-to-end training of Quantization Parameters (E2E-QP) — trains only quantization step sizes end-to-end, capturing inter-module interactions. Achieves 2-bit Llama-2-70B on a single A100-80GB in 41 hours with <3 points of accuracy degradation versus full precision.

## Key Contributions
- Block-AP: novel block-wise QAT that trains *all* parameters in each transformer block (not just LoRA-adapter slices), giving the optimizer full freedom inside the block.
- E2E-QP: cheap end-to-end pass training only the quantization step sizes (LSQ-style), capturing cross-block dependencies that block-wise training misses.
- Demonstrates 2-bit LLaMA-2-70B at 69.48 vs 72.41 FP zero-shot avg, on a single A100-80GB in 41h.
- Validated across base / instruction-tuned / multimodal LLMs from 7B to 70B.

## Key Figures/Tables to Study
- **Figure 2:** Block-AP + E2E-QP pipeline diagram.
- **Table 3:** LLaMA-2-7B/13B/70B 2-bit results — EfficientQAT vs GPTQ/AWQ/OmniQuant.
- **Figure 5:** Memory and time cost vs traditional QAT.

## Technical Details

### Phase 1 — Block-AP (Block-wise All Parameter training)
For each transformer block b in isolation:
1. Cache input activations h_{b-1} on the calibration corpus (one forward pass, then frozen).
2. Load block b's FP weights into GPU.
3. Initialize quantized weights W_q via RTN.
4. Loss: `L_b = || f_b^FP(h_{b-1}) − f_b^quant(h_{b-1}) ||²`.
5. SGD on *all* parameters of block b (W, layer-norm scales, quantization scales) with STE through the quantizer.
6. After convergence, store W_q + scales; release FP weights from memory.

Memory: only one block's worth of FP gradients + optimizer state at a time. For LLaMA-70B that's ~2 GB per block vs 280 GB for the full model.

### Phase 2 — E2E-QP (End-to-end Quantization Parameter training)
Freeze all weights W_q at their Block-AP values. Make only the *quantization step sizes* s ∈ ℝ (one per group/per-channel) trainable. Forward pass: full quantized model; loss: causal-LM next-token prediction on calibration data.

Memory cost is tiny: gradients only flow through s (a few hundred KB total parameters), so backprop fits easily. Captures cross-block interactions that per-block calibration cannot.

### Quantization rule (LSQ-style)
`W_q = round(W / s)`, `Ŵ = s · W_q`, with s learnable per group of 64–128 weights. STE gradient: `dŴ/ds = round(W/s) − (W/s)` (LSQ derivation).

### Wall clock
- Block-AP for Llama-2-70B: ~32 hours on A100-80GB.
- E2E-QP: ~9 hours additional.
- Total: 41 hours, single GPU.

### Why this beats PTQ at 2-bit
PTQ (GPTQ, AWQ) only adjusts quantization scales and a few zero-points — too few free parameters to recover 2-bit accuracy. EfficientQAT adjusts *all* block weights to fit the quantized output, giving billions of free parameters to compensate.

## Connections
- LSQ ancestor for step-size training: [[lsq]].
- Data-free QAT line: [[llm-qat]], [[quest]].
- Self-distillation QAT sibling: [[bitdistiller]].
- LoRA-aware QAT alternatives: [[qa-lora]], [[lq-lora]], [[loftq]].
- Discrete-code fine-tuning alternative: [[pv-tuning]].
