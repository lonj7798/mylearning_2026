<!-- scope: 2026 frontier LLM quantization research — rolling index after May 2026 crawl
     deps: [[deepseek-v3-fp8]], [[nvfp4-training]], [[quartet-ii]]
     see-also: [[mxfp4-native-hardware-2026]], [[nvfp4-qad]], [[statistically-lossless-quantization]], [[adaptive-kv-cache-quant]]
-->

# 2026 Frontier LLM Quantization — Rolling Index
- **Core Insight:** The May 2026 frontier has concrete anchors now: NVFP4 training is moving from stochastic rounding to MS-EDEN ([[quartet-ii]]), MXFP4 training diagnostics point to Wgrad as the hard path ([[mxfp4-native-hardware-2026]]), NVFP4 inference recovery uses QAD ([[nvfp4-qad]]), and evaluation is shifting from benchmark deltas to distribution fidelity ([[statistically-lossless-quantization]]).
- **Guideline:** Use this file only as the 2026 map; cite the concrete pages for chapter substance. Keep looking for public >50B FP4 pretraining runs, multi-lab NVFP4 production reports, and MoE-specific per-expert quantization papers.
- **Authors:** rolling index
- **Year:** 2026
- **URL:** see Connections for the closest concrete sources
- **Relevant topics:** FP4 native pretraining, NVFP4/MXFP4 production, distribution-lossless quantization, KV-cache compression, MoE quantization

## Abstract
This file used to be a placeholder for the 2026 crawl. After the May 2026 freshness pass, the course now has concrete 2026 anchors: Quartet II for NVFP4 gradient estimation, native-hardware MXFP4 training diagnosis, NVFP4 quantization-aware distillation, FP4 inference sensitivity analysis, statistically-lossless quantization, adaptive KV-cache precision, and KV cache transform coding. It remains useful as a high-level index and gap log, not as a primary source page.

## Key Contributions
- Tracks the 2026 shift from "Can FP4 work?" to "Which tensor path, block layout, and recovery method makes FP4 reliable?"
- Separates NVFP4 training ([[nvfp4-training]], [[quartet-ii]]) from NVFP4 inference recovery ([[nvfp4-qad]]).
- Separates MXFP4 training claims ([[mxfp4-pretraining]], [[mxfp4-native-hardware-2026]]) from model-release claims ([[gpt-oss-mxfp4]]).
- Records open gaps that still need single canonical artifacts.

## Key Figures/Tables to Study
- Use the concrete source pages linked below; this index has no standalone figures.

## Technical Details
Open gaps after the May 2026 crawl:
- No public >50B dense FP4 pretraining run with full loss curves has displaced [[nvfp4-training]]'s 12B/10T evidence.
- No broad, multi-lab Blackwell production report has displaced [[nvfp4-qad]] as the most concrete NVFP4 production-recovery source.
- MoE-specific bit allocation beyond MXFP4 MoE-weight release cases remains under-covered.
- Sub-bit LLM work is still represented mainly by BitNet/AQLM-style 2024-2025 lines, not a clean 2026 consolidation.

## Connections
- [[quartet-ii]] — strongest 2026 NVFP4 training-method addition.
- [[mxfp4-native-hardware-2026]] — newest MXFP4 native-hardware training diagnosis (May 2026).
- [[nvfp4-qad]] — production NVFP4 inference recovery with quantization-aware distillation.
- [[fp4-inference-diagnosis]] — layer/block sensitivity map for MXFP4 and NVFP4 inference.
- [[statistically-lossless-quantization]] — new evaluation vocabulary for "lossless" quantization.
- [[adaptive-kv-cache-quant]] and [[kvtc]] — 2026 KV-cache compression additions.
- [[gpt-oss-mxfp4]] — production model-release case for MXFP4 MoE weight quantization.
- [[nvfp4-training]] — the current frontier reference (12B / 10T NVFP4 pretraining run, 2025-09).
- [[deepseek-v3-fp8]] — the FP8 frontier ancestor (2024-12); 2026 frontier reports will be measured against its loss-gap baseline.
- [[mxfp4-pretraining]] — academic MXFP4 pretrain (up to 6.7B); 2026 may push this above 50B.
- [[bitnet-b158-2b]] — Microsoft's 2025 2B / 4T release; the BitNet line's 2026 follow-up (rumored larger model) will go here.
- [[nvfp4-production-2026]] — sibling placeholder for production deployment reports.
- [[sub-bit-llm-2026]] — sibling placeholder for fractional-bit research consolidation.
- [[blackwell-quantization]] — hardware report; 2026 frontier work depends on the B200 / B300 / GB300 fleet rollout.
