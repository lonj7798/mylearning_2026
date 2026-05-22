<!-- scope: fractional-bit research consolidation (sub-1-bit LLMs) — 2026 placeholder
     deps: [[bitnet-b158]], [[qmoe]]
     see-also: [[quant-2026-frontier]], [[bitnet-models]]
-->

# Sub-Bit LLMs (2026) — Placeholder
- **Core Insight:** Sub-1-bit territory is no longer a single regime — it splits into (a) ternary-sparse (BitNet b1.58 with structured sparsity → ~1.0 bpw), (b) MoE-amortized sub-bit (QMoE → 0.8 bpw via per-expert compression), and (c) variable-bit codebooks (most weights at 1 bit, rare outliers at 4-8). 2026 work is collating these into a unified "fractional bpw" framework with explicit codebook vs sparsity vs MoE-amortization tradeoffs.
- **Guideline:** When picking a sub-bit method, decide the dominant axis first: dense + ternary + sparsity (BitNet line), MoE + per-expert compression (QMoE line), or hybrid codebook + outlier (SqueezeLLM / AQLM line); each axis has different memory / latency / accuracy tradeoffs.
- **Authors:** placeholder — to be populated as 2026 consolidation settles
- **Year:** 2026
- **URL:** anchored on QMoE (https://arxiv.org/abs/2310.16795), BitNet b1.58 (https://arxiv.org/abs/2402.17764), AQLM, and 2026 consolidation work
- **Relevant topics:** sub-1-bit LLM, fractional bpw, MoE compression, ternary + sparsity

## Abstract
This is a placeholder for 2026 sub-bit consolidation work. The three already-canonical sub-bit lines are: (1) BitNet b1.58 (1.585 bpw) — ternary dense; QMoE (~0.8 bpw average) — per-expert compression of large MoEs; AQLM / QuIP# (~ 2 bpw) — codebook-based with outlier handling. 2026 consolidation papers (placeholder) are expected to combine these into a single unified analysis showing where each method's sweet spot lies on a "memory-vs-quality" Pareto frontier across model class (dense, MoE) and bit budget.

## Key Contributions (to be populated)
- Unified Pareto frontier across BitNet / QMoE / AQLM / QuIP# at the same memory budget.
- Per-axis selection criteria for sub-bit methods.
- New 2026 sub-bit methods (likely fusing ternary + per-expert + codebook).
- Hardware co-design papers showing sub-bit hardware (FPGAs, ASICs, BitNet-specific accelerators).

## Key Figures/Tables to Study
- (placeholder — populate when canonical 2026 consolidation paper exists)

## Technical Details
**Gap log entry**: no canonical 2026 sub-bit consolidation paper at library freeze; closest concrete anchors are the BitNet b1.58 2B-4T release and the QMoE 2023 paper for the MoE-compression side.

### Sub-bit method taxonomy (as of 2025)
| Method | bpw | Class | Mechanism |
|--------|-----|-------|-----------|
| BitNet b1.58 | 1.585 | dense | ternary {-1, 0, +1} weights |
| OneBit | 1.0 | dense | binary weight via SVID + scale rebroadcast |
| QMoE | 0.8 (avg) | MoE | per-expert codebook + sparsity |
| AQLM 2-bit | 2 | dense | additive vector quant + codebook |
| QuIP# 2-bit | 2 | dense | E8 lattice + incoherence processing |
| VPTQ | 2 | dense | vector PTQ via GPTQ-style updates |
| SqueezeLLM W2 | 2 (avg) | dense | sensitivity codebook + sparse outliers |

### Why MoE allows sub-bit averages
MoE models have many "cold" experts that activate rarely → high-rate codebook compression is cheap (decode latency hits only when expert is hot). QMoE pushes the average bpw below 1 by aggressive compression of cold experts while keeping hot experts at higher bpw.

### Where sub-bit research is heading (2026 expectations)
- Hybrid dense + MoE sub-bit recipes for the new generation of large MoEs (DeepSeek-V3-class, ~ 600B total / 30B active).
- Hardware-co-designed BitNet accelerators in academic prototypes.
- Quality-recovery via small high-precision adapters on top of sub-bit bases (extends [[qlora]] / [[pv-tuning]] direction).

## Connections
- [[bitnet-b158]] / [[bitnet-b158-2b]] — the dense ternary anchor.
- [[onebit]] — sub-1-bit binary attempt.
- [[qmoe]] — MoE sub-bit anchor.
- [[aqlm]] / [[quip-sharp]] / [[vptq]] — codebook-based 2-bit anchors.
- [[squeezellm]] / [[squeezellm-followups]] — sensitivity-based sub-bit line.
- [[pv-tuning]] — fine-tuning sub-2-bit models with small adapters.
- [[quant-2026-frontier]] — parent placeholder index.
