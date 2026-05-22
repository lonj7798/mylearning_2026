<!-- scope: Statistically-lossless LLM quantization; EAR metric and SLQ asymmetric non-uniform method
     deps: [[gptq]], [[awq]], [[rate-distortion-theory]], [[information-theoretic-bounds]]
     see-also: [[aqlm]], [[spqr]], [[squeezellm]]
-->

# Statistically-Lossless Quantization of Large Language Models
- **Core Insight:** "Lossless" quantization should be measured at multiple behavioral levels: task scores can be preserved below 4 bits, but distribution-level fidelity needs stricter metrics such as Expected Acceptance Rate and usually 5-6 average bits.
- **Guideline:** Do not judge near-lossless quantization only by zero-shot benchmark deltas; also test next-token distribution fidelity, especially for speculative decoding, distillation, and applications sensitive to sampling behavior.
- **Authors:** Michael Helcig, Eldar Kurtic, Dan Alistarh
- **Year:** 2026
- **URL:** https://arxiv.org/abs/2605.02404
- **Relevant topics:** statistically-lossless quantization, expected acceptance rate, asymmetric quantization, non-uniform quantization, optimized kernels

## Abstract
This paper formalizes a middle ground between lossy practical quantization and exact lossless compression. It separates task-lossless compression, which preserves benchmark accuracy within natural sampling variance, from distribution-lossless compression, which preserves the model's next-token distribution. The paper proposes Expected Acceptance Rate (EAR) as an interpretable distribution-fidelity metric and introduces SLQ, a layer-wise non-uniform method with asymmetric quantization and broad bitwidth search. Reported results include task-lossless compression below 4 bits per parameter, distribution-lossless compression around 5-6 bits on average, and 1.7x-3.6x speedups with optimized kernels.

## Key Contributions
- Defines several operational notions of "lossless" for LLM quantization instead of treating the word as binary.
- Introduces **Expected Acceptance Rate (EAR)** as a metric for next-token distribution agreement under optimal coupling.
- Proves a gamma-squared variance law explaining why symmetric quantization can inflate noise and why asymmetry matters for distribution-level fidelity.
- Proposes **SLQ**, a layer-wise non-uniform quantization method with asymmetric quantization and wide bitwidth search.
- Reports inference speedups relative to FP16, so the method is not only a storage compression result.

## Key Figures/Tables to Study
- Metric-definition section: distinguishes task-lossless, distribution-lossless, and exact lossless.
- EAR vs benchmark accuracy plot: shows when task metrics hide distribution drift.
- Bitwidth allocation table: indicates which layers need extra bits for distribution fidelity.
- Kernel speedup table: connects the statistical definition to deployable inference.

## Technical Details

### Three levels of losslessness
| Level | What must be preserved | Typical bit budget reported |
|-------|------------------------|-----------------------------|
| Task-lossless | benchmark accuracy within run variance | below 4 bits/parameter in some settings |
| Distribution-lossless | next-token distribution effectively unchanged | about 5-6 average bits |
| Exact lossless | exact weights/logits | storage compression, usually no inference speedup |

### Expected Acceptance Rate
EAR is an interpretable next-token fidelity metric: an EAR of 0.99 means the quantized and original models can be coupled to agree on the next sampled token about 99% of the time. This is more relevant than MMLU deltas when the downstream use cares about sampling distribution, speculative decoding acceptance, or logit-level distillation.

### Why asymmetry matters
The paper argues that symmetric quantization can inflate variance by a factor tied to the distribution asymmetry. For distribution-level fidelity, this makes asymmetric quantization more than an engineering detail: it can be necessary to avoid systematic next-token drift.

### SLQ
SLQ combines:
- layer-wise bitwidth search,
- non-uniform quantization,
- asymmetric quantization,
- optimized kernels to recover real inference speed.

## Connections
- [[gptq]] and [[awq]] — practical lossy baselines; SLQ asks when those are statistically indistinguishable from full precision.
- [[rate-distortion-theory]] — the course's theoretical foundation for why bitwidth and distribution fidelity trade off.
- [[aqlm]] / [[spqr]] / [[squeezellm]] — adjacent methods that also chase near-lossless low-bit behavior through non-uniformity, sparsity, or additive codebooks.
- [[marlin-kernel]] — the kernel lesson: compression claims matter more when there is a fast runtime path.

## Notes
This is one of the most valuable May 2026 additions because it improves the course's evaluation vocabulary. It gives students a way to explain why "same benchmark score" is not the same as "same model."
