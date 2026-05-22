<!-- scope: 1-bit Quantized Johnson-Lindenstrauss transform for KV-cache quantization; sign-bit JL sketch with asymmetric inner-product estimator, no per-block scale overhead
     deps: [[rate-distortion-theory]]
     see-also: [[polarquant]], [[turboquant]], [[kivi]], [[kvquant]]
-->

# QJL: 1-Bit Quantized JL Transform for KV Cache Quantization with Zero Overhead
- **Core Insight:** Apply a Johnson-Lindenstrauss random projection to each KV vector and keep only the **sign** of each output coordinate; pair this 1-bit sketch with an unquantized JL sketch on the other side of an inner product, and the resulting **asymmetric estimator** is unbiased with minimal distortion — and stores no per-block scale or zero-point at all.
- **Guideline:** When attention-score reconstruction (inner-product preservation) is what you need from your KV cache, QJL gives you 3-bit equivalent quality at zero metadata overhead, with a tight CUDA kernel; it is the natural drop-in for KV-cache compression on serving systems that can re-implement the attention dot product.
- **Authors:** Amir Zandieh, Majid Daliri, Insu Han (Google Research)
- **Year:** 2024 (arxiv Jun)
- **URL:** https://arxiv.org/abs/2406.03482
- **Relevant topics:** KV-cache quantization, Johnson-Lindenstrauss, sign-bit sketching, inner-product estimation, data-oblivious quant

## Abstract
Serving LLMs requires substantial memory due to the storage requirements of KV embeddings in the KV cache, which grows with sequence length. Quantization is an effective compression approach, but traditional methods face significant memory overhead from needing to store quantization constants (zero point + scale) in full precision per data block — adding 1 or 2 extra bits per quantized number depending on block size. QJL is a new approach combining a **Johnson-Lindenstrauss transform followed by sign-bit quantization** — it eliminates the per-block metadata overhead entirely. The paper proposes an **asymmetric estimator** for the inner product: apply QJL (sign-only) to one vector and a standard JL transform (no quantization) to the other; this yields an unbiased estimator with minimal distortion. An efficient CUDA implementation is provided. When applied to quantize the KV cache to **3 bits**, QJL achieves more than **5×** KV cache memory reduction without compromising accuracy, with faster runtime.

## Key Contributions
- A **data-oblivious, zero-overhead** sketch — the JL projection matrix is fixed at startup and shared globally; no per-block scale or zero-point stored.
- The **asymmetric inner-product estimator**: only one operand is sign-quantized, the other is a standard JL projection. This breaks the symmetry that would otherwise inflate variance and is what makes the estimator unbiased.
- A **lightweight CUDA kernel** combining the JL projection, sign quantization, and the asymmetric inner-product computation into a single fused operation, beating the unquantized FP16 attention baseline on long contexts.
- > 5× KV memory reduction on LLMs across NLP tasks at 3-bit equivalent, with no measured accuracy loss.

## Key Figures/Tables to Study
- The asymmetric-estimator schematic: one side `sign(Π · v)`, other side `Π · u`. The diagram makes clear *why* you store the sign-only on the cached side and keep the live-query side full-precision.
- The bit-budget table: traditional KIVI/KVQuant pay 1–2 bits/element of overhead, QJL pays zero.
- Accuracy vs compression curves on QA / summarization tasks.

## Technical Details

### The QJL sketch
Given a vector `v ∈ R^d` and a random JL projection matrix `Π ∈ R^{m × d}` (entries iid `±1/sqrt(m)` or Gaussian):
- **Stored**: `s(v) = sign(Π · v) ∈ {-1, +1}^m`. One bit per output coordinate. No scale, no zero-point.
- **Live (query side)**: `Π · u` computed full-precision on the query vector `u`.

### Asymmetric inner-product estimator
For two vectors `u, v`:
`⟨u, v⟩ ≈ α · ⟨s(v), Π · u⟩ = α · Σ_j s(v)_j · (Π · u)_j`
where `α` is a known scaling constant depending on `d`, `m`, and the JL distribution. The asymmetry — sign-only on one side, full real on the other — gives an **unbiased** estimator. (A symmetric sign-only on both sides would be biased by a factor of `2/π`-like terms via the "sign-product = arcsin of cosine" identity.)

### Why zero overhead works
Traditional KIVI/KVQuant store, per block of `B` elements:
- `B · k` bits for quantized values (`k` = bit-width)
- 1 scale (FP16, 16 bits) + 1 zero-point (FP16, 16 bits) ⇒ 32 bits per block ⇒ `32/B` extra bits per element.
For `B = 32` and `k = 3`, the overhead alone adds 1 bit/element — meaning "3-bit" KV is really ~4-bit effective.

QJL stores `m` sign bits per vector and *nothing else*. The scaling constant `α` is fixed at startup (depends only on dimensions), not per-block. Effective rate: pure 1 bit per JL-output coordinate, with `m ≈ d/4` to `d/2` giving "3-bit equivalent" effective rate per input coordinate.

### Bit budget
| Sketch width m | Effective bits / input coord | Memory reduction |
|----------------|------------------------------|-------------------|
| m = d/4 | ~0.25 bit | ~16× over FP16 |
| m = d/2 | ~0.5 bit | ~8× over FP16 |
| m = d | ~1 bit | ~5× over FP16 (paper headline 3-bit equivalent quality) |

### CUDA kernel
The released kernel fuses: (1) JL projection of `K`, (2) sign quantization, (3) packing into bytes, (4) at decode time, the unpacked `±1` signs are multiplied by `Π · q` for the current query. This avoids materializing the dequantized K cache and saves both memory bandwidth and SM time.

## Connections
- [[polarquant]] — same group; uses polar coords instead of JL signs; sister approach to "data-oblivious KV quant with no per-block scale".
- [[turboquant]] — supersedes/subsumes QJL as **Stage 2** of a two-stage pipeline (Stage 1 = scalar quant after rotation, Stage 2 = 1-bit QJL on the residual).
- [[kivi]] / [[kvquant]] — per-channel/per-token baselines that pay the per-block metadata overhead QJL eliminates.
- [[rate-distortion-theory]] — sign-only quantization is the 1-bit limit of any scalar quantizer; QJL gets away with it via the asymmetric estimator and high-dimensional concentration.

## Notes
QJL is the **earliest** paper in the Zandieh / Mirrokni KV-cache trilogy: QJL (Jun 2024) → [[polarquant]] (Feb 2025) → [[turboquant]] (Apr 2025, ICLR 2026). Read these three together for the conceptual arc from "1-bit JL sketches" → "polar coords with closed-form angle distribution" → "unified rate-distortion-optimal two-stage quantizer".
