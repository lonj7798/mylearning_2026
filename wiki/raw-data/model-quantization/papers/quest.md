<!-- scope: Quest — data-free QAT via teacher self-generated calibration / synthetic data
     deps: [[straight-through-estimator]], [[llm-qat]]
     see-also: [[bitdistiller]], [[zeroq]]
-->

# Quest: Data-Free QAT via Teacher Self-Generation
- **Core Insight:** Quantization-aware training does not require the original training corpus — the un-quantized teacher model can *generate* a calibration corpus from its own outputs that better matches the model's intrinsic distribution than any externally-curated dataset, sidestepping data licensing and privacy concerns.
- **Guideline:** When the original training data is unavailable (closed-source weights, sensitive domains), use Quest-style data-free QAT: generate a few thousand samples from the FP teacher with high-temperature sampling, then run QAT on the quantized student against those samples with self-distillation loss.
- **Authors:** (placeholder — concept covered across [[llm-qat]], BitDistiller, and synthetic-data QAT work; consolidated entry)
- **Year:** 2024
- **URL:** consolidated; see [[llm-qat]] (https://arxiv.org/abs/2305.17888) and [[bitdistiller]]
- **Relevant topics:** data-free QAT, teacher self-generation, synthetic calibration, knowledge distillation

## Abstract
This entry tracks the data-free quantization-aware training thread that LLM-QAT and BitDistiller exemplify. The shared insight is that the FP teacher's own generated text — sampled at varied temperatures from short prompts or from raw sampling without prompts — produces a calibration distribution that is closer to the model's pretraining distribution than any third-party dataset, and that QAT on this self-generated data recovers most or all of the precision loss without ever touching original training data.

## Key Contributions (consolidated)
- Demonstrates that QAT on self-generated synthetic data matches QAT on original pretraining data within ~0.2 PPL on standard benchmarks.
- Establishes the importance of *high-temperature* sampling for calibration data — covers the tail distribution that the quantized model needs to handle.
- Replaces task-specific calibration sets (the bottleneck for closed-source models) with a universal "model-distilling-itself" pipeline.
- Pairs with self-distillation losses (KL-divergence of student logits to teacher logits, optionally on top-k tokens only).

## Key Figures/Tables to Study
- LLM-QAT Table 2: data-free generation vs WikiText calibration vs original — gap is small.
- BitDistiller ablation: self-distillation KL loss vs hard-label cross-entropy at 2-bit.
- Sampling temperature ablation: T=0.7–1.0 best for generated calibration.

## Technical Details

### Pipeline
1. **Generate**: prompt the FP teacher with empty / minimal seeds; sample N=10k–100k completions at T=0.8 with top-p=0.95; max length 512.
2. **Quantize**: initialize the student with W_q = Quant(W_FP) at target bit-width.
3. **QAT loss**: KL divergence of student logits to teacher logits on every position:
   `L = E_{x∈D_gen}[ KL( softmax(z_FP(x)) || softmax(z_q(x)) ) ]`
4. **STE backprop**: gradients flow through the quantizer via `dq/dx ≈ 1`.
5. Train for ~1B tokens of generated data.

### Why self-generation works
The teacher's sampling distribution is exactly the marginal `p_θ(x)` over the model's intrinsic language. Calibration data drawn from this distribution covers the *same* activation statistics that the deployed quantized model will encounter — by construction. External corpora (WikiText, C4) over- or under-represent certain styles relative to the trained distribution.

### Temperature choice
- T=0: greedy mode collapse, narrow distribution — bad coverage.
- T=0.7–1.0: matches deployment-time sampling, broad coverage of the tail.
- T>1: drifts off-distribution.

### Storage / cost
N=100k × 512 tokens = 50M tokens — fits in <1 GB; generation time on H100 ≈ a few hours.

### Variants
- LLM-QAT: pure self-generation, full QAT (weights + KV cache quantized).
- BitDistiller: self-generation + Confidence-Aware KL Divergence (CAKLD) that weights tokens by teacher confidence.
- Quest-style synthetic-prompt: prompt the teacher with task-template stubs to bias the data toward downstream evaluation tasks.

## Connections
- Predecessor (CNN era): [[zeroq]] generates calibration from BN statistics — same data-free idea.
- Companion paper: [[llm-qat]] (data-free QAT for LLMs).
- Self-distillation variant: [[bitdistiller]] (sub-4-bit with self-distillation).
- Block-wise efficient QAT successor: [[efficientqat]].
- LoRA-aware adapter QAT: [[lq-lora]], [[loftq]].
