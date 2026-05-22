<!-- scope: 2024 LLM-specific quantization survey (Zhu et al. or Gong/Liang); consolidates 2022–2024 LLM PTQ methods
     deps: [[survey-gholami-2021]]
     see-also: [[gptq]], [[awq]], [[smoothquant]], [[quip]], [[bitnet-b158]]
-->

# A Survey on Model Compression for Large Language Models (Zhu et al. 2024)
- **Core Insight:** LLM-era quantization splits cleanly along three lines absent from pre-LLM surveys — (a) weight-only vs weight+activation methods (driven by activation-outlier breakdown at ≥6.7B params), (b) calibration-free vs calibration-driven methods, and (c) integer-format methods vs equivalent-transformation methods (rotation / scaling) — and the 2022–2024 explosion is fundamentally about *handling activation outliers*, not about better integer codes.
- **Guideline:** Use this survey to navigate the LLM-quant landscape; map any new paper onto its 4×4 taxonomy (W-only/WA × calibrated/calibration-free × uniform/non-uniform × per-tensor/per-channel) before reading.
- **Authors:** Xunyu Zhu, Jian Li, Yong Liu, Can Ma, Weiping Wang (also other 2024 LLM-quant surveys, e.g. Gong / Liang) — multiple competing surveys appeared in 2024
- **Year:** 2024
- **URL:** https://arxiv.org/abs/2308.07633 (Zhu 2024 "A Survey on Model Compression for Large Language Models", TACL); also https://arxiv.org/abs/2402.18158 (Gong 2024 "A Survey on Low-Bit Large Language Models")
- **Relevant topics:** LLM quantization survey, weight-only vs W+A, outlier handling, 2022–2024 PTQ explosion

## Abstract
The 2024 LLM-quantization surveys consolidate the explosion of PTQ methods that followed LLM.int8() (Dettmers 2022). Unlike pre-LLM surveys (which focused on uniform/non-uniform × PTQ/QAT × per-tensor/per-channel), the LLM-era surveys add a critical fourth axis: **activation outliers and how each method handles them**. Methods cluster into three classes: (1) **isolation** — keep outlier channels in higher precision (LLM.int8(), SpQR, OWQ); (2) **equivalent transformation** — pre-multiply weights/activations by per-channel scales or rotations to flatten the outlier distribution (SmoothQuant, AWQ, QuaRot, SpinQuant, OmniQuant); (3) **better-than-uniform codes** — non-uniform quantization or vector quantization that allocates resolution to the outliers (SqueezeLLM, AQLM, QuIP). The surveys also cover QAT (LLM-QAT, BitDistiller), KV-cache quantization (KIVI, KVQuant), and the BitNet 1-bit lineage.

## Key Contributions
- LLM-era taxonomy: weight-only vs weight+activation × calibrated vs calibration-free × static vs dynamic.
- Methods catalog: 30+ methods covering 2022–2024 (LLM.int8, GPTQ, AWQ, SmoothQuant, ZeroQuant, QuIP, SpQR, OmniQuant, KIVI, QuaRot, BitNet, etc.).
- Empirical comparison tables: bit-width × method × model × downstream task accuracy.
- Identifies the four "method classes" (isolation, equivalent transformation, non-uniform code, vector quant) that organize the literature.
- Discusses calibration-set sensitivity, hardware support, and deployment considerations.

## Key Figures/Tables to Study
- **Method taxonomy tree**: the canonical 2024-era diagram organizing GPTQ/AWQ/SmoothQuant/etc. into branches.
- **Bit-width vs perplexity** comparison on Llama-1/2 across all major methods (canonical "leaderboard" plot).
- **Outlier histograms**: per-channel activation max-abs values across layers; motivates every equivalent-transformation method.

## Technical Details

### Taxonomy (LLM-era extended)
| Axis | Options |
|------|---------|
| **Target** | Weights only (W-only) / Weights + Activations (W+A) / KV cache only |
| **Calibration** | Calibration-free (RTN, HQQ, NF4) / Data-driven (GPTQ, AWQ, SmoothQuant) |
| **Code** | Uniform (INT*) / Non-uniform (NF4, SqueezeLLM LUT, FP4) / Vector (AQLM, QuIP) |
| **Granularity** | Per-tensor / per-channel / per-group / per-token |
| **Outlier handling** | Isolation / equivalent transformation / better code |

### Method classes
**1. Isolation methods** (keep outliers in higher precision)
- LLM.int8(): outlier channels in FP16, bulk in INT8.
- SpQR: outlier weights stored as sparse FP16, bulk as 3-bit.
- OWQ: outlier-aware weight quantization.

**2. Equivalent-transformation methods** (pre-multiply to flatten)
- SmoothQuant: migrate per-channel activation outliers to weights via scaling.
- AWQ: scale weight channels by activation magnitude before quant.
- OmniQuant: learnable per-channel scales + clip thresholds.
- QuaRot / SpinQuant: rotate weights/activations by Hadamard / learned orthogonal to suppress outliers entirely.

**3. Non-uniform / vector code methods**
- NF4 / SqueezeLLM: non-uniform 4-bit code matched to weight distribution.
- AQLM / QuIP# / VPTQ / GPTVQ: vector quantization with small codebooks; sub-2-bit.

**4. KV cache methods** (own track)
- KIVI: per-channel K + per-token V.
- KVQuant: ultra-low-bit KV cache with dense-and-sparse decomposition.
- GEAR: error compensation.

### Calibration objectives surveyed
- MSE (most common).
- Hessian-weighted MSE (GPTQ, AdaRound).
- Activation-aware MSE (AWQ, OmniQuant).
- KL divergence on output distribution.
- End-task accuracy (rare; expensive but most-aligned).

### Outlier characterization
The surveys document the empirical finding (originally from [[llm-int8]]):
- At ≥6.7B params, ~0.1% of activation channels carry magnitudes 10–100× larger than the bulk.
- Outliers are *consistent across tokens* within a channel → per-channel scale can isolate them.
- Outlier channels are not constant across layers — typically a few hundred channels in early/late layers.

### Bit-width quality trends (consolidated)
| Method | W-bit | A-bit | Llama-7B PPL gap |
|--------|-------|-------|------------------|
| FP16 | 16 | 16 | 0 |
| LLM.int8() | 8 (+ FP16 outliers) | 8 | ~0.05 |
| SmoothQuant | 8 | 8 | ~0.1 |
| GPTQ | 4 | 16 | ~0.15 |
| AWQ | 4 | 16 | ~0.1 |
| QuaRot | 4 | 4 | ~0.3 |
| SpinQuant | 4 | 4 | ~0.2 |
| AQLM | 2 | 16 | ~0.5 |
| QuIP# | 2 | 16 | ~0.5 |
| BitNet b1.58 (QAT) | 1.58 | 8 | parity (trained from scratch) |

### Open problems identified
- Reliable W4A4 on small models (< 7B).
- KV-cache below 2 bits.
- Sub-2-bit weight quantization without major quality loss on instruction-tuned models.
- Quantization stability across long-context / many-step reasoning.
- Native sub-FP8 training at frontier scale.

### Critique
- Surveys vary in coverage: some emphasize PTQ over QAT, others vice versa.
- BitNet b1.58 and FP8 training are sometimes excluded as "not strictly quantization."
- Empirical comparisons sensitive to calibration-set choice; absolute numbers vary by 0.1–0.3 PPL across reproductions.

## Connections
- [[survey-gholami-2021]] — pre-LLM canonical taxonomy; this survey extends it.
- [[survey-low-bit-llm-2024]] — sister survey focused on <8-bit LLM.
- [[survey-efficient-llm-inference-2024]] — broader efficient-inference survey with a quantization section.
- [[gptq]] / [[awq]] / [[smoothquant]] — three most-cited methods in this survey.
- [[quip]] / [[aqlm]] / [[bitnet-b158]] — sub-2-bit and 1-bit lineage.
- [[kvquant]] / [[kivi]] — KV-cache compression covered as own track.
