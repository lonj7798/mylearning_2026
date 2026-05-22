<!-- scope: Adaptive per-token KV-cache bit allocation for lightweight on-device LLMs
     deps: [[kivi]], [[kvquant]], [[qaq]], [[kv-cache-compression-survey-2025]]
     see-also: [[turboquant]], [[polarquant]], [[kvtc]]
-->

# Don't Waste Bits! Adaptive KV-Cache Quantization for Lightweight On-Device LLMs
- **Core Insight:** KV-cache bitwidth should be token-dependent: a lightweight controller can assign 2-bit, 4-bit, 8-bit, or FP16 precision based on token importance instead of spending the same bit budget everywhere.
- **Guideline:** For on-device and edge LLMs, consider adaptive KV precision when static KV quantization drops accuracy; preserve high precision only for tokens whose features indicate high downstream impact.
- **Authors:** Sayed Pedram Haeri Boroujeni, Niloufar Mehrabi, Patrick Woods, Gabriel Hillesheim, Abolfazl Razi
- **Year:** 2026 (CVPR 2026)
- **URL:** https://arxiv.org/abs/2604.04722
- **Relevant topics:** KV-cache quantization, adaptive bit allocation, on-device LLMs, edge inference, token importance

## Abstract
This paper proposes adaptive KV-cache quantization for lightweight on-device LLMs. Instead of a fixed KV precision, a compact controller selects per-token precision from `{2-bit, 4-bit, 8-bit, FP16}` during decoding. The controller uses lightweight features such as token frequency, quality score, attention variance, and entropy-based uncertainty. Experiments on SmolLM models show improved accuracy-latency trade-offs against static KV quantization and rule-based baselines.

## Key Contributions
- Introduces a learned token-level policy for KV-cache bit allocation.
- Uses inexpensive online features rather than full calibration or expensive teacher passes.
- Supports multiple precisions in one cache: 2-bit, 4-bit, 8-bit, and FP16.
- Focuses on mobile, embedded, and edge-device constraints rather than datacenter-only serving.
- Reports improved accuracy and latency compared with static KV quantization on SmolLM-135M, SmolLM-360M, and SmolLM-1.7B.

## Key Figures/Tables to Study
- Controller architecture diagram: shows the feature-to-bitwidth decision path.
- Bit allocation histogram: reveals which tokens receive FP16 or 2-bit treatment.
- Accuracy-latency frontier vs static baselines: the key deployment result.
- SmolLM-360M HellaSwag example: useful concrete case for teaching adaptive precision.

## Technical Details

### Controller inputs
The policy uses token-level features including:
- token frequency,
- token quality score,
- attention variance,
- entropy or uncertainty signal.

These features estimate whether a token's cached key/value will matter enough to justify extra bits.

### Precision menu
| Precision | Intended use |
|-----------|--------------|
| 2-bit | low-impact tokens |
| 4-bit | default compressed cache |
| 8-bit | moderately important tokens |
| FP16 | highly important tokens |

### How it differs from KIVI/KVQuant
[[kivi]] and [[kvquant]] are primarily fixed-method quantizers with specific per-channel/per-token layouts. Adaptive KV quantization is a bit-allocation policy layered over the idea that not all tokens deserve equal precision.

## Connections
- [[qaq]] — quality-adaptive KV-cache quantization is the closest earlier theme.
- [[kivi]] / [[kvquant]] / [[gear]] — fixed or method-specific KV quantization baselines.
- [[turboquant]] / [[polarquant]] — data-oblivious vector/KV approaches; adaptive KV quantization is orthogonal because it chooses precision based on token importance.
- [[kvtc]] — another 2026 KV-storage direction, but focused on transform coding for reusable caches.

## Notes
This is narrower than TurboQuant: it is most relevant for edge models and adaptive precision policy, not as a universal KV quantizer. It is still worth including because it captures the 2026 move from "choose one KV bitwidth" to "allocate bits dynamically."
