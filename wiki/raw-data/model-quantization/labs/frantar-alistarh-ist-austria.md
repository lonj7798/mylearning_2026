<!-- scope: Elias Frantar + Dan Alistarh (IST Austria) — Hessian-based PTQ + production GEMM kernels
     deps: [[gptq]], [[obc]], [[aqlm]], [[marlin-kernel]]
     see-also: [[sparsegpt]]
-->

# Frantar + Alistarh (IST Austria) — Hessian PTQ + Production Kernels
- **Core Insight:** This research line turned second-order, one-shot compression into the default 4-bit LLM PTQ workflow and paired it with kernels that deliver real W4 inference speedups.
- **Guideline:** Read this lab track for the OBC/GPTQ/AQLM/Marlin lineage: Hessian-aware local objectives first, then production GEMM kernels.
- **Authors:** Elias Frantar, Dan Alistarh, ISTA DAS Lab, and collaborators
- **Year:** 2022–2026
- **URL:** https://github.com/IST-DASLab/gptq ; https://github.com/IST-DASLab/marlin
- **Relevant topics:** GPTQ, OBC, SparseGPT, AQLM, Marlin, Machete, second-order PTQ

## Summary
Elias Frantar and Dan Alistarh at the Institute of Science and Technology Austria (IST Austria) run the research line behind **Hessian-based post-training quantization at LLM scale** — [[gptq]] is the foundational paper of the entire 4-bit-LLM-PTQ era — and the companion line of **production W4 GEMM kernels** ([[marlin-kernel]], Machete) that turn the quantized weights into actual inference speedups. The lab also produced [[aqlm]] (sub-2-bit additive quantization) and SparseGPT (one-shot LLM pruning), establishing the broader theme of "second-order, one-shot compression for LLMs."

## Notable Works
- [[obs-obd]] (Hassibi 1993 / LeCun 1989) — the theoretical ancestor the lab modernized; Optimal Brain Surgeon.
- [[obc]] (Frantar 2022) — Optimal Brain Compression; unified second-order pruning + quantization.
- [[gptq]] (Frantar 2022, ICLR 2023) — one-shot Hessian-based PTQ for OPT/BLOOM; 4-bit at 175B scale.
- SparseGPT (Frantar 2023) — one-shot magnitude pruning of LLMs to 50-60% sparsity at minimal quality loss; same second-order framework.
- [[aqlm]] (Egiazarian / Frantar 2024) — additive vector quantization for sub-2-bit LLMs.
- [[marlin-kernel]] (Frantar 2024) — W4A16 GEMM kernel for Ampere/Hopper achieving near-FP16 throughput at low batch.
- Quality-Adaptive variants (QQQ, etc.) — follow-ups exploring smart bit-allocation per layer.

## Recurring themes
- **One-shot, layer-local, second-order**: the lab's signature recipe is "minimize a layer-local quadratic with the calibration Hessian, in one pass." Used in OBC, GPTQ, SparseGPT, AQLM-prep.
- **Make the kernel ship**: GPTQ would have been an academic result without Marlin; the lab consistently follows up algorithmic papers with production-grade CUDA kernels.
- **Push the bit-width frontier**: GPTQ took 4-bit from "research" to "default"; AQLM is doing the same for 2-bit.

## Open Resources
- GPTQ code: https://github.com/IST-DASLab/gptq (original); https://github.com/AutoGPTQ/AutoGPTQ (community fork, ubiquitous)
- SparseGPT: https://github.com/IST-DASLab/sparsegpt
- AQLM: https://github.com/Vahe1994/AQLM
- Marlin: https://github.com/IST-DASLab/marlin
- Lab homepage: http://people.csail.mit.edu/alistarh/ (Dan Alistarh) — also linked from IST Austria's machine-learning faculty page

## Connections
- [[dettmers-group]] — adjacent; Dettmers does distribution-aware PTQ, Frantar/Alistarh does Hessian-aware PTQ. The two approaches are complementary and often combined.
- [[han-song-mit]] — adjacent; HAN Lab also produces algorithm + kernel pairs (AWQ + TinyChat).
- [[nvidia-quantization]] — Marlin / Machete eventually integrated into TensorRT-LLM and vLLM, putting the lab's kernels in production NVIDIA pipelines.
