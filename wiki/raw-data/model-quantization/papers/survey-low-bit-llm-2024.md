<!-- scope: 2024 survey of sub-8-bit LLM quantization (Gong / Liang lineage); deep coverage of W4 and below
     deps: [[survey-llm-quantization-2024]], [[survey-gholami-2021]]
     see-also: [[gptq]], [[awq]], [[quip-sharp]], [[aqlm]], [[bitnet-b158]]
-->

# A Survey on Low-Bit Large Language Models (Gong, Liang et al. 2024)
- **Core Insight:** Below 4 bits/weight, the dominant quality lever is no longer the quantization *algorithm* (GPTQ vs AWQ vs HQQ) but the underlying *code geometry* — uniform integer codes hit a hard wall around 3 bits, non-uniform scalar codes extend to 2.5 bits, and only vector quantization (AQLM, QuIP#) or end-to-end pretraining (BitNet) cross into the sub-2-bit regime without catastrophic degradation.
- **Guideline:** When targeting sub-4-bit weight quantization, first check whether your model is large enough (≥ 70B is the rough threshold for AQLM/QuIP# to recover quality); below 7B, plan for end-to-end QAT (BitNet-style) instead of PTQ.
- **Authors:** Various 2024 surveys focused on <8-bit (e.g. Gong et al. "A Survey on Low-Bit Large Language Models", Liang et al. low-bit consolidations)
- **Year:** 2024
- **URL:** https://arxiv.org/abs/2402.18158 ("Low-Bit Quantization of Neural Networks for Efficient Inference", Gong); also https://arxiv.org/abs/2409.16694 (LLM compression surveys 2024)
- **Relevant topics:** sub-4-bit, 2-bit, 1.58-bit, vector quantization, low-bit LLM survey

## Abstract
The 2024 low-bit LLM surveys consolidate the rapid push from 4-bit (GPTQ/AWQ established 2022–2023) down to 2-bit (QuIP, QuIP#, AQLM, PV-Tuning) and 1.58-bit (BitNet b1.58). They identify the **bit-width quality cliff** at ~3 bits/weight, below which uniform scalar quantization (INT3, INT2) breaks for any model size and any calibration method. The surveys partition the sub-4-bit literature into four classes: (1) **scalar uniform with heroic calibration** (GPTQ extensions), (2) **scalar non-uniform** (SqueezeLLM, NF3/NF2 codes), (3) **vector / incoherence-processed** (QuIP, QuIP#, AQLM, VPTQ), and (4) **end-to-end pretrained binary/ternary** (BitNet, BitNet b1.58, OneBit). The headline finding: vector quantization is the only PTQ route to sub-2-bit, and end-to-end QAT is the only route to true 1-bit.

## Key Contributions
- Systematic taxonomy of sub-8-bit methods specifically.
- Empirical "bit-width cliff" analysis: shows the discontinuous quality drop at ~3 bits for uniform codes.
- Coverage of vector / incoherence-processing methods (QuIP, QuIP#, AQLM) absent from broader LLM-quant surveys.
- Identifies model-scale dependencies: AQLM works well at 70B but lags at 7B.
- Maps out the BitNet 1.58-bit ecosystem (training recipe, inference kernels, scaling).

## Key Figures/Tables to Study
- **Bit-width vs perplexity cliff** at ~3 bits — the defining figure of the survey.
- **Method-class quality vs bit-width** (uniform vs non-uniform vs vector vs end-to-end): vector and end-to-end methods dominate below 2.5 bits.
- **Model-scale × method × bit-width 3D plot**: shows AQLM-7B underperforms vs AQLM-70B at the same bit-width.

## Technical Details

### The bit-width cliff
Uniform scalar quant (INT*) perplexity gap vs FP16:
| Bits | Llama-7B PPL gap | Llama-70B PPL gap |
|------|------------------|---------------------|
| 8 | ~0.05 | ~0.02 |
| 4 (GPTQ) | ~0.15 | ~0.05 |
| 3 (GPTQ) | ~0.7 | ~0.3 |
| 2 (GPTQ) | catastrophic (NaN-like) | ~3.0 |

The discontinuous jump from 3 to 2 bits is the **bit-width cliff** — fundamental to scalar quantization.

### Four method classes for sub-4-bit

**1. Scalar uniform with heroic calibration**
- GPTQ + finer groups + better calibration sets.
- Helpful at 4-bit, no help at 2-bit (hits the cliff).

**2. Scalar non-uniform**
- SqueezeLLM (Fisher-weighted k-means LUT).
- Extended NF3 / NF2 quantile codes.
- HQQ at 3-bit.
- Pushes the cliff from ~3 bits to ~2.5 bits, no further.

**3. Vector / incoherence-processed quantization**
- QuIP (Chee 2023): random rotations + 2-bit lattice quant.
- QuIP# (Tseng 2024): E8 lattice codebook + Hadamard rotations.
- AQLM (Egiazarian 2024): additive vector quantization with small codebooks.
- VPTQ / GPTVQ: vector PTQ variants.
- These cross into the 2-bit and 1.5-bit regimes with manageable quality loss at ≥70B.

**4. End-to-end pretrained low-bit**
- BitNet (Wang 2023): binary {-1, +1} from scratch.
- BitNet b1.58 (Ma 2024): ternary {-1, 0, +1}; matches FP16 at ≥3B.
- OneBit (Xu 2024): 1-bit via SVD initialization.
- BitDistiller (Du 2024): self-distillation to sub-4-bit.

### Why vector quantization crosses the cliff
Scalar quantization is bounded below by the Lloyd-Max code for the source distribution; in 1-D, the space-filling loss (1.53 dB) cannot be recovered. Vector quantization in dimension d > 1 closes this gap (see [[vector-quantization]]). At 2 bits/weight, the gap matters quantitatively — d = 8 VQ achieves ~0.3 PPL where scalar 2-bit achieves catastrophic failure.

### Why end-to-end QAT enables true 1-bit
PTQ assumes the FP model is fixed and finds the best quantized approximation. QAT lets the model *learn around* the quantization constraint. At 1-bit, no fixed FP model has a good 1-bit approximation — but a model *trained* with the 1-bit constraint develops representations that work natively at 1-bit. This is the BitNet b1.58 lesson.

### Activation handling at sub-4 bits
- Sub-4-bit weight methods almost universally keep activations at 8-bit or 16-bit.
- **W4A4** (4-bit activations + 4-bit weights) requires rotation (QuaRot) or learned equalization (OmniQuant).
- **W2A4 / W2A8** is the typical sub-2-bit deployment.
- True end-to-end 1-bit weights + 1-bit activations is only demonstrated by binary BitNet (not yet matched in quality).

### Hardware implications
- Sub-4-bit benefits are dominated by *memory bandwidth* (smaller weights = faster decode-bound inference).
- Compute throughput is bounded by activation precision, not weight precision.
- ⇒ W2A16 doesn't speed up *compute*, only memory; W4A4 does both.
- Special hardware (BitNet accelerators, MX-FP4 on Blackwell) needed to capture true sub-4-bit compute gains.

### Calibration-set sensitivity at low bits
- 4-bit: calibration set choice changes results by ~0.05 PPL.
- 3-bit: changes by ~0.2 PPL.
- 2-bit: changes by ~0.5 PPL; choice of calibration distribution becomes a hyperparameter.

### Open problems
- Sub-2-bit on 7B-scale models (PTQ class limitation).
- KV cache below 2 bits.
- Sub-2-bit on instruction-tuned models (chat/RL-trained models more sensitive).
- Universal sub-2-bit hardware support (currently fragmented).

## Connections
- [[survey-llm-quantization-2024]] — broader LLM-quant survey; sub-4-bit is a chapter there.
- [[survey-gholami-2021]] — pre-LLM foundation; doesn't cover sub-4-bit at all.
- [[gptq]] — the 4-bit workhorse that hits the cliff at 3-bit.
- [[squeezellm]] — non-uniform scalar code; pushes the cliff a bit lower.
- [[quip]] / [[quip-sharp]] / [[aqlm]] — vector / incoherence methods crossing the cliff.
- [[bitnet]] / [[bitnet-b158]] — end-to-end QAT lineage.
- [[vector-quantization]] — theoretical justification for why VQ crosses the cliff.
