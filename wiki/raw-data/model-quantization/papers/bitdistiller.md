<!-- scope: BitDistiller — self-distillation QAT for sub-4-bit LLMs with Confidence-Aware KL Divergence
     deps: [[llm-qat]], [[straight-through-estimator]]
     see-also: [[efficientqat]], [[quest]], [[pv-tuning]]
-->

# BitDistiller: Unleashing the Potential of Sub-4-Bit LLMs via Self-Distillation
- **Core Insight:** Self-distillation (student quantized model learning from FP teacher logits) recovers sub-4-bit accuracy far better than plain QAT cross-entropy, but vanilla KL-divergence over-weights confident teacher predictions where the student already agrees — Confidence-Aware KL Divergence (CAKLD) re-weights so the loss focuses on the disagreement frontier.
- **Guideline:** For 2-3 bit LLM QAT, pair asymmetric clipping (per-channel) with self-distillation using CAKLD as the loss — converges in fewer steps and on less data than vanilla KL, and surpasses prior 2-3 bit PTQ/QAT.
- **Authors:** Dayou Du, Yijia Zhang, Shijie Cao, Jiaqi Guo, Ting Cao, Xiaowen Chu, Ningyi Xu
- **Year:** 2024
- **URL:** https://arxiv.org/abs/2402.10631
- **Relevant topics:** self-distillation, CAKLD, asymmetric clipping, sub-4-bit QAT

## Abstract
BitDistiller combines quantization-aware training with knowledge distillation for ultra-low-precision (2-3 bit) LLMs. Two technical contributions: (1) asymmetric quantization with per-channel clipping that preserves the long tail of LLM weight distributions; (2) Confidence-Aware KL Divergence (CAKLD) — a self-distillation objective that adaptively re-weights tokens based on teacher confidence. Significantly surpasses prior PTQ and QAT methods at 2- and 3-bit on general language and reasoning benchmarks, with less data and compute than competing approaches.

## Key Contributions
- Asymmetric quantization (separate clipping for positive and negative ranges, per channel) — important for weights with skewed distributions, which is typical of LLMs post-LayerNorm.
- Confidence-Aware KL Divergence (CAKLD) — re-weights the per-token KL by the teacher's prediction confidence, focusing learning on tokens where teacher and student disagree.
- Demonstrates that self-distillation against the model's own FP teacher dominates supervised fine-tuning at low bit-widths.
- Practical recipe: short fine-tuning on synthetic / general corpora rather than task-specific data.

## Key Figures/Tables to Study
- **Figure 1:** Asymmetric clipping diagram — per-channel positive/negative bounds.
- **Figure 3:** CAKLD weighting curves — confidence-vs-weight mapping.
- **Table 2:** LLaMA / Mistral 2/3-bit results — BitDistiller vs OmniQuant vs AWQ vs LLM-QAT.

## Technical Details

### Asymmetric per-channel clipping
For each output channel c, learn separate positive and negative clipping bounds α_c^+, α_c^−:
`W'_{c,j} = clip(W_{c,j}, −α_c^−, +α_c^+)`
Then quantize W' with symmetric INT scheme. Bounds α^± learned per channel during QAT.

This matters because LayerNorm + linear weights are not symmetric — the positive and negative tails differ in extent, and symmetric clipping wastes precision.

### Confidence-Aware KL Divergence (CAKLD)
Standard KL distillation:
`L = Σ_t KL(p_T(·|x_{<t}) || p_S(·|x_{<t}))`
where p_T is teacher, p_S is student.

CAKLD reweights:
`L_CAKLD = Σ_t w(c_t) · KL_t`
with c_t = teacher confidence = `max_y p_T(y|x_{<t})`, and w a learned monotone weighting:
- w(c_t) low when c_t ≈ 1 (teacher is very confident; student usually matches; little to learn).
- w(c_t) high when c_t in the 0.3–0.7 "interesting" range (teacher uncertain, student gradient most informative).

### Training recipe
- Teacher = FP base model.
- Student = quantized base model with asymmetric clipping bounds trainable.
- Loss = CAKLD on a generic corpus (Wikipedia or self-generated).
- ~10B tokens of fine-tuning.
- STE through quantizer.

### Performance
LLaMA-2-7B at 3-bit: WikiText-2 PPL ~5.8 vs FP 5.5 — within 0.3.
At 2-bit: BitDistiller PPL ~7.5 — significantly better than OmniQuant 2-bit (~8.5).

## Connections
- Self-distillation QAT line: [[quest]], [[llm-qat]].
- Asymmetric quant lineage: [[lsq-plus]] (asymmetric LSQ).
- Sub-4-bit competitor lines: [[aqlm]], [[quip-sharp]], [[pv-tuning]].
- Block-wise QAT companion: [[efficientqat]].
- Confidence-weighted distillation parallel: temperature-scaled KD (Hinton 2015).
